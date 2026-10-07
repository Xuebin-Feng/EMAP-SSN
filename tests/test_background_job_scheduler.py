"""Background job scheduler (Background_Job_Scheduler): FIFO non-overlapping
jobs, status updates on the GUI thread, prompt shutdown, and output-path
reservations."""

import os
import sys
import tempfile
import threading
import time
import unittest
import gc
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtCore

from Background_Job_Scheduler import BackgroundJobScheduler


class ViewerStub:
    def __init__(self):
        self.console_text = type("Console", (), {"text": "Cmd: untouched_"})()
        self.background_job_status = ""
        self.status_updates = []
        self.update_threads = []

    def set_background_job_status(self, message):
        self.background_job_status = str(message)
        self.status_updates.append(self.background_job_status)
        self.update_threads.append(threading.get_ident())


# Generous deadlines: waits return as soon as their condition holds, and the
# "does not block" checks compare against jobs that stay blocked for
# BLOCKED_JOB_SECONDS, so a loaded CPU cannot turn a pass into a failure.
STARTUP_TIMEOUT = 10.0
BLOCKED_JOB_SECONDS = 10.0
NON_BLOCKING_LIMIT = 5.0


def wait_for(predicate, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    application = QtCore.QCoreApplication.instance()
    while time.monotonic() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    application.processEvents()
    return predicate()


class BackgroundJobSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse an application another test module created and leave it alive;
        # only an application created here is shut down in tearDownClass.
        cls._owns_application = QtCore.QCoreApplication.instance() is None
        cls.application = (
            QtCore.QCoreApplication.instance()
            or QtCore.QCoreApplication([])
        )

    @classmethod
    def tearDownClass(cls):
        application = cls.application
        cls.application = None
        if cls._owns_application:
            application.quit()
            application.deleteLater()
            application.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(
                None,
                QtCore.QEvent.Type.DeferredDelete,
            )
        del application
        gc.collect()

    def test_mixed_jobs_are_fifo_nonoverlapping_and_continue_after_failure(self):
        viewer = ViewerStub()
        gui_thread = threading.get_ident()
        scheduler = BackgroundJobScheduler(viewer)
        release_first = threading.Event()
        first_started = threading.Event()
        execution = []
        active = 0
        max_active = 0
        lock = threading.Lock()
        failures = []
        successes = []
        reveal_threads = []
        scheduler.job_failed.connect(
            lambda job, error, detail, elapsed: failures.append(job.job_id)
        )
        scheduler.job_succeeded.connect(
            lambda job, result, elapsed: successes.append(job.job_id)
        )

        def worker(payload):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
                execution.append(("start", payload))
            try:
                if payload == "label":
                    first_started.set()
                    release_first.wait(BLOCKED_JOB_SECONDS)
                if payload == "broken":
                    raise RuntimeError("expected failure")
                result = {"message": f"finished {payload}", "save_path": payload}
                if payload == "label":
                    result["reveal_directory"] = directory
                return result
            finally:
                with lock:
                    execution.append(("end", payload))
                    active -= 1

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "Background_Job_Scheduler.open_in_file_manager",
            side_effect=lambda _path: reveal_threads.append(threading.get_ident()),
        ):
            scheduler.enqueue(
                "label", "label test", "label", worker,
                os.path.join(directory, "one.xlsx"),
            )
            self.assertTrue(first_started.wait(STARTUP_TIMEOUT))
            enqueue_started = time.monotonic()
            scheduler.enqueue(
                "logo", "failure test", "broken", worker,
                os.path.join(directory, "two.svg"),
            )
            scheduler.enqueue(
                "logo", "logo test", "logo", worker,
                os.path.join(directory, "three.svg"),
            )
            self.assertLess(time.monotonic() - enqueue_started, NON_BLOCKING_LIMIT)
            release_first.set()
            self.assertTrue(
                wait_for(lambda: successes == [1, 3] and failures == [2])
            )

        self.assertEqual(max_active, 1)
        self.assertEqual(
            execution,
            [
                ("start", "label"), ("end", "label"),
                ("start", "broken"), ("end", "broken"),
                ("start", "logo"), ("end", "logo"),
            ],
        )
        self.assertIn("completed", viewer.background_job_status)
        self.assertEqual(viewer.console_text.text, "Cmd: untouched_")
        self.assertTrue(any("Queued background job" in update for update in viewer.status_updates))
        self.assertTrue(any("Running background job" in update for update in viewer.status_updates))
        self.assertTrue(any("failed after" in update for update in viewer.status_updates))
        self.assertEqual(set(viewer.update_threads), {gui_thread})
        self.assertEqual(reveal_threads, [gui_thread])
        scheduler.shutdown()

    def test_shutdown_returns_promptly_and_discards_queued_jobs(self):
        viewer = ViewerStub()
        scheduler = BackgroundJobScheduler(viewer)
        release = threading.Event()
        started = threading.Event()
        executed = []

        def blocking_worker(payload):
            executed.append(payload)
            started.set()
            release.wait(BLOCKED_JOB_SECONDS)
            return {"message": "finished", "save_path": payload}

        with tempfile.TemporaryDirectory() as directory:
            active_path = os.path.join(directory, "active.svg")
            queued_path = os.path.join(directory, "queued.xlsx")
            scheduler.enqueue(
                "logo", "active", "active", blocking_worker, active_path
            )
            self.assertTrue(started.wait(STARTUP_TIMEOUT))
            scheduler.enqueue(
                "label", "queued", "queued", blocking_worker, queued_path
            )
            shutdown_started = time.monotonic()
            scheduler.shutdown()
            self.assertLess(time.monotonic() - shutdown_started, NON_BLOCKING_LIMIT)
            self.assertFalse(scheduler.is_output_path_reserved(queued_path))
            release.set()
            self.assertTrue(wait_for(lambda: scheduler.queue_depth == 0))

        self.assertEqual(executed, ["active"])

    def test_existing_and_reserved_outputs_are_rejected(self):
        viewer = ViewerStub()
        scheduler = BackgroundJobScheduler(viewer)
        release = threading.Event()

        def worker(_payload):
            release.wait(BLOCKED_JOB_SECONDS)
            return {"message": "done", "save_path": "ignored"}

        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "existing.svg")
            with open(existing, "wb") as output:
                output.write(b"present")
            with self.assertRaises(FileExistsError):
                scheduler.enqueue("logo", "existing", None, worker, existing)

            reserved = os.path.join(directory, "reserved.svg")
            scheduler.enqueue("logo", "reserved", None, worker, reserved)
            with self.assertRaises(FileExistsError):
                scheduler.enqueue("logo", "reserved", None, worker, reserved)
            release.set()
            self.assertTrue(wait_for(lambda: scheduler.queue_depth == 0))
        scheduler.shutdown()

    def test_allow_overwrite_permits_existing_file_but_rejects_reserved_path(self):
        viewer = ViewerStub()
        scheduler = BackgroundJobScheduler(viewer)
        release = threading.Event()

        def worker(_payload):
            release.wait(BLOCKED_JOB_SECONDS)
            return {"message": "done", "save_path": "ignored"}

        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "existing.svg")
            with open(existing, "wb") as output:
                output.write(b"present")

            # With allow_overwrite=True, enqueuing with an existing file on disk succeeds
            job_id = scheduler.enqueue(
                "logo",
                "existing",
                None,
                worker,
                existing,
                allow_overwrite=True,
            )
            self.assertEqual(job_id, 1)

            # Concurrent reservation for the same path is still rejected even if allow_overwrite=True
            with self.assertRaises(FileExistsError):
                scheduler.enqueue(
                    "logo",
                    "existing",
                    None,
                    worker,
                    existing,
                    allow_overwrite=True,
                )

            release.set()
            self.assertTrue(wait_for(lambda: scheduler.queue_depth == 0))
        scheduler.shutdown()


if __name__ == "__main__":
    unittest.main()
