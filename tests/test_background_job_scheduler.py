"""Background job scheduler (Background_Job_Scheduler): non-overlapping jobs in
the order of the line, status updates on the GUI thread, prompt shutdown, and
output-path reservations; and the line itself: waiting jobs moved or removed
while the thread takes the next one, a running job told to stop at its next
check point, job numbers and the command portal's record through it all."""

import os
import random
import sys
import tempfile
import threading
import time
import unittest
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtCore

from Background_Job_Scheduler import BackgroundJobScheduler
from Viewer_Command_Portal import ExecutionContext, ViewerCommandPortal, bind
from utilities.Job_Cancellation import JobCancelled, cancel_requested, raise_if_cancelled
from utilities.Localization import Message, display_text
from tests.viewer_fixtures import NO_WINDOW_STATES, WINDOW_STATES, apply_window_state


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

    def test_a_folder_a_job_asks_to_reveal_opens_only_where_a_user_is_at_the_viewer(self):
        # print and label finish with reveal_directory; the scheduler is the one place that acts on it.
        for state in WINDOW_STATES:
            with self.subTest(state=state):
                viewer = apply_window_state(ViewerStub(), state)
                scheduler = BackgroundJobScheduler(viewer)
                self.addCleanup(scheduler.shutdown)
                done = []
                scheduler.job_succeeded.connect(lambda job, result, elapsed: done.append(job.job_id))
                with tempfile.TemporaryDirectory() as directory, mock.patch(
                    "Background_Job_Scheduler.open_in_file_manager"
                ) as reveal:
                    scheduler.enqueue(
                        "print", "reveal test", "payload",
                        lambda payload: {"message": "finished", "reveal_directory": directory},
                        os.path.join(directory, "one.png"),
                    )
                    self.assertTrue(wait_for(lambda: bool(done)))
                    wait_for(lambda: False, 0.1)   # let the success report run
                self.assertEqual(reveal.called, state not in NO_WINDOW_STATES)
                if reveal.called:
                    reveal.assert_called_once_with(directory)

    def test_a_viewer_that_is_gone_still_reveals(self):
        scheduler = BackgroundJobScheduler(ViewerStub())
        self.addCleanup(scheduler.shutdown)
        scheduler._viewer_ref = lambda: None
        done = []
        scheduler.job_succeeded.connect(lambda job, result, elapsed: done.append(job.job_id))
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "Background_Job_Scheduler.open_in_file_manager"
        ) as reveal:
            scheduler.enqueue(
                "print", "reveal test", "payload",
                lambda payload: {"message": "finished", "reveal_directory": directory},
                os.path.join(directory, "one.png"),
            )
            self.assertTrue(wait_for(lambda: bool(done)))
            wait_for(lambda: False, 0.1)
        reveal.assert_called_once_with(directory)

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


class LineTestCase(unittest.TestCase):
    """A scheduler whose first job holds the thread until a test lets it go, so the line can be arranged."""

    @classmethod
    def setUpClass(cls):
        cls._owns_application = QtCore.QCoreApplication.instance() is None
        cls.application = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    @classmethod
    def tearDownClass(cls):
        application = cls.application
        cls.application = None
        if cls._owns_application:
            application.quit()
            application.deleteLater()
            application.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        del application
        gc.collect()

    def setUp(self):
        self.scheduler = BackgroundJobScheduler(ViewerStub())
        self.addCleanup(self.stop_scheduler)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.release = threading.Event()
        self.started = threading.Event()
        self.ran = []
        self.reports = []
        self.scheduler.job_started.connect(lambda job: self.reports.append(("started", job.job_id)))
        self.scheduler.job_succeeded.connect(lambda job, result, elapsed: self.reports.append(("succeeded", job.job_id)))
        self.scheduler.job_failed.connect(lambda job, error, detail, elapsed: self.reports.append(("failed", job.job_id)))
        self.scheduler.job_cancelled.connect(
            lambda job, started: self.reports.append(("stopped" if started else "removed", job.job_id)))

    def stop_scheduler(self):
        self.release.set()
        self.scheduler.shutdown()
        wait_for(lambda: self.scheduler.queue_depth == 0)

    def path(self, name):
        return os.path.join(self.directory.name, name)

    def worker(self, payload):
        self.ran.append(payload)
        if payload == "blocker":
            self.started.set()
            self.release.wait(BLOCKED_JOB_SECONDS)
        return {"message": f"done {payload}"}

    def enqueue(self, payload, worker=None, **options):
        return self.scheduler.enqueue("logo", f"logo -> {payload}.svg", payload, worker or self.worker,
                                      self.path(f"{payload}.svg"), **options)

    def hold_the_thread(self):
        """Queue the job that runs until self.release is set, and wait until it runs."""
        job_id = self.enqueue("blocker")
        self.assertTrue(self.started.wait(STARTUP_TIMEOUT))
        return job_id

    def line(self):
        return [entry.job.job_id for entry in self.scheduler.snapshot()]

    def wait_idle(self):
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))
        wait_for(lambda: False, 0.1)  # the reports queued behind it


class LineOrderTests(LineTestCase):
    def test_a_waiting_job_moves_along_the_line_and_the_jobs_then_run_in_that_order(self):
        blocker = self.hold_the_thread()
        first, second, third = (self.enqueue(name) for name in ("a", "b", "c"))
        self.assertEqual((blocker, first, second, third), (1, 2, 3, 4))
        self.assertEqual(self.line(), [1, 2, 3, 4])
        self.assertTrue(self.scheduler.move(third, -2))
        self.assertEqual(self.line(), [1, 4, 2, 3])
        self.assertTrue(self.scheduler.move(first, 1))
        self.assertEqual(self.line(), [1, 4, 3, 2])
        self.release.set()
        self.wait_idle()
        self.assertEqual(self.ran, ["blocker", "c", "b", "a"])
        self.assertEqual([job for kind, job in self.reports if kind == "succeeded"], [1, 4, 3, 2])

    def test_the_ends_of_the_line_and_the_running_job_do_not_move(self):
        blocker = self.hold_the_thread()
        first, second, third = (self.enqueue(name) for name in ("a", "b", "c"))
        self.assertFalse(self.scheduler.move(first, -1), "the first waiting job cannot pass the running one")
        self.assertFalse(self.scheduler.move(third, 1))
        self.assertFalse(self.scheduler.move(second, 0))
        self.assertFalse(self.scheduler.move(blocker, 1), "the running job never moves")
        self.assertFalse(self.scheduler.move(blocker, -1))
        self.assertFalse(self.scheduler.move(99, 1), "a job that is not there")
        self.assertEqual(self.line(), [blocker, first, second, third])
        # A move past an end stops at the end.
        self.assertTrue(self.scheduler.move(third, -10))
        self.assertEqual(self.line(), [blocker, third, first, second])
        self.assertTrue(self.scheduler.move(third, 10))
        self.assertEqual(self.line(), [blocker, first, second, third])

    def test_a_job_that_has_run_cannot_be_moved(self):
        blocker = self.hold_the_thread()
        waiting = self.enqueue("a")
        self.release.set()
        self.wait_idle()
        self.assertFalse(self.scheduler.move(blocker, 1))
        self.assertFalse(self.scheduler.move(waiting, -1))

    def test_the_snapshot_lists_the_running_job_first_with_each_place_in_the_line(self):
        blocker = self.hold_the_thread()
        waiting = [self.enqueue(name) for name in ("a", "b")]
        after = time.monotonic()
        entries = self.scheduler.snapshot()
        self.assertEqual([(entry.job.job_id, entry.running, entry.position) for entry in entries],
                         [(blocker, True, 1), (waiting[0], False, 2), (waiting[1], False, 3)])
        self.assertLessEqual(entries[0].started_at, after)
        self.assertIsNone(entries[1].started_at)
        self.assertFalse(any(entry.stopping for entry in entries))
        # listing() keeps its older shape: (job, running) pairs in the same order.
        self.assertEqual([(job.job_id, running) for job, running in self.scheduler.listing()],
                         [(blocker, True), (waiting[0], False), (waiting[1], False)])
        self.assertEqual(self.scheduler.queue_depth, 3)

    def test_moves_made_while_the_thread_takes_the_next_job_lose_and_repeat_nothing(self):
        # Many quick jobs, and a second thread that moves and removes jobs as fast as it can while
        # the scheduler's thread takes the next one: each job ends exactly one way (completed,
        # removed from the line, or stopped), none twice, none at the same time as another, and
        # every output is free again afterwards.
        generator = random.Random(42)
        count = 120
        active, overlap = [0], [0]
        lock = threading.Lock()
        completed = []

        def worker(payload):
            with lock:
                active[0] += 1
                overlap[0] = max(overlap[0], active[0])
            try:
                time.sleep(0.0005)
                raise_if_cancelled()
                with lock:
                    completed.append(payload)
            finally:
                with lock:
                    active[0] -= 1
            return {"message": "done"}

        blocker = self.hold_the_thread()
        ids = [self.enqueue(f"job{index}", worker) for index in range(count)]
        stop = threading.Event()

        def stir():
            while not stop.is_set():
                job_id = generator.choice(ids)
                if generator.random() < 0.85:
                    self.scheduler.move(job_id, generator.choice((-3, -1, 1, 3)))
                else:
                    self.scheduler.cancel(job_id)
                time.sleep(0.0002)  # as fast as is fair: the lock is not a queue

        mover = threading.Thread(target=stir, daemon=True)
        mover.start()
        self.release.set()
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0, 60))
        stop.set()
        mover.join(10)
        self.assertFalse(mover.is_alive())
        self.wait_idle()
        ended = [job for kind, job in self.reports if kind in ("succeeded", "removed", "stopped")]
        self.assertEqual(sorted(ended), sorted([blocker, *ids]), "each job ended exactly once")
        ran = [int(payload[3:]) + blocker + 1 for payload in completed]
        self.assertEqual(len(ran), len(set(ran)), "no job completed twice")
        self.assertEqual(sorted(ran), sorted(job for kind, job in self.reports if kind == "succeeded" and job != blocker),
                         "the jobs that completed are the ones reported as completed")
        self.assertEqual(overlap[0], 1)
        self.assertEqual(self.scheduler.listing(), [])
        for index in range(count):
            self.assertFalse(self.scheduler.is_output_path_reserved(self.path(f"job{index}.svg")))

    def test_job_numbers_never_change_or_repeat_through_moves_and_removals(self):
        blocker = self.hold_the_thread()
        first = self.enqueue("a")
        second = self.enqueue("b")
        self.scheduler.move(second, -1)
        self.assertTrue(self.scheduler.cancel(first))
        third = self.enqueue("c")
        self.assertEqual((blocker, first, second, third), (1, 2, 3, 4), "a removed job's number is not reused")
        self.assertEqual(self.line(), [1, 3, 4])
        self.release.set()
        self.wait_idle()


class LineRemovalTests(LineTestCase):
    def test_a_waiting_job_is_removed_at_once_and_its_output_is_free_again(self):
        self.hold_the_thread()
        waiting = self.enqueue("a")
        output = self.path("a.svg")
        self.assertTrue(self.scheduler.is_output_path_reserved(output))
        self.assertTrue(self.scheduler.cancel(waiting))
        # At once: while the thread is still busy with another job.
        self.assertFalse(self.scheduler.is_output_path_reserved(output))
        self.assertEqual(self.scheduler.queue_depth, 1)
        self.assertEqual(self.line(), [1])
        self.assertTrue(wait_for(lambda: ("removed", waiting) in self.reports))
        # Free to be queued again, under a number of its own.
        again = self.enqueue("a")
        self.assertEqual(again, waiting + 1)
        self.release.set()
        self.wait_idle()
        self.assertEqual(self.ran, ["blocker", "a"])
        self.assertEqual(self.reports.count(("removed", waiting)), 1)

    def test_nothing_is_cancelled_twice_or_after_it_is_over_or_when_it_does_not_exist(self):
        self.hold_the_thread()
        waiting = self.enqueue("a")
        self.assertTrue(self.scheduler.cancel(waiting))
        self.assertFalse(self.scheduler.cancel(waiting))
        self.assertFalse(self.scheduler.cancel(404))
        self.release.set()
        self.wait_idle()
        self.assertFalse(self.scheduler.cancel(1), "a job that finished")

    def test_cancelling_several_jobs_takes_out_the_waiting_ones_before_the_running_one_stops(self):
        # The running job ends the moment it is told to; the job behind it must not start in between.
        started = threading.Event()

        def worker(payload):
            started.set()
            while True:
                time.sleep(0.005)
                raise_if_cancelled()

        running = self.enqueue("a", worker)
        first, second = self.enqueue("b"), self.enqueue("c")
        self.assertTrue(started.wait(STARTUP_TIMEOUT))

        cancelled = self.scheduler.cancel_many([running, first, second, 404])

        self.assertEqual(sorted(cancelled), sorted([running, first, second]), "404 is no job")
        self.assertTrue(wait_for(lambda: ("stopped", running) in self.reports))
        self.wait_idle()
        self.assertEqual(self.ran, [], "neither waiting job started")
        self.assertEqual(
            sorted(job for kind, job in self.reports if kind == "removed"), sorted([first, second]))
        self.assertEqual(self.scheduler.snapshot(), [])
        self.assertEqual(self.scheduler.cancel_many([running, first]), [], "nothing is left to cancel")

    def test_a_running_job_is_told_to_stop_and_ends_at_its_next_check_point_without_its_file(self):
        reached, stopping_seen = threading.Event(), threading.Event()
        partial = self.path(".partial-file")

        def worker(payload):
            with open(partial, "wb") as handle:
                handle.write(b"half")
            try:
                reached.set()
                while True:
                    if stopping_seen.wait(0.005):
                        raise_if_cancelled()      # the job's own check point
            finally:
                os.remove(partial)                # what each real worker's finally does

        job_id = self.enqueue("a", worker)
        waiting = self.enqueue("b")
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.assertTrue(os.path.exists(partial))
        self.assertTrue(self.scheduler.cancel(job_id))
        entries = self.scheduler.snapshot()
        self.assertTrue(entries[0].running and entries[0].stopping, "the card shows it stopping")
        self.assertFalse(self.scheduler.cancel(job_id), "already told to stop")
        self.assertFalse(self.scheduler.move(job_id, 1))
        self.assertTrue(self.scheduler.is_output_path_reserved(self.path("a.svg")), "its output is held until it ends")
        stopping_seen.set()
        self.assertTrue(wait_for(lambda: ("stopped", job_id) in self.reports))
        self.wait_idle()
        self.assertFalse(os.path.exists(partial), "the unfinished file is gone")
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("a.svg")), "the reservation is released")
        # The line goes on with the next job; the stopped one is neither a success nor a failure.
        self.assertEqual(self.ran, ["b"])
        self.assertEqual([kind for kind, job in self.reports if job == job_id], ["started", "stopped"])
        self.assertIn(("succeeded", waiting), self.reports)
        self.assertEqual(self.scheduler.snapshot(), [])

    def test_a_job_with_no_check_point_finishes_and_is_reported_as_completed(self):
        # A worker that never asks (a stand-in worker, or the VR scheduler's) cannot be stopped midway:
        # it writes what it was going to write, and the report says it completed.
        release = threading.Event()
        entered = threading.Event()

        def worker(payload):
            entered.set()
            release.wait(BLOCKED_JOB_SECONDS)
            return {"message": "done anyway"}

        job_id = self.enqueue("a", worker)
        self.assertTrue(entered.wait(STARTUP_TIMEOUT))
        self.assertTrue(self.scheduler.cancel(job_id))
        release.set()
        self.wait_idle()
        self.assertEqual([kind for kind, job in self.reports if job == job_id], ["started", "succeeded"])

    def test_a_job_told_to_stop_before_its_worker_asks_ends_at_the_first_check(self):
        gate = threading.Event()

        def worker(payload):
            gate.wait(BLOCKED_JOB_SECONDS)
            self.assertTrue(cancel_requested())
            raise_if_cancelled()
            self.ran.append("not reached")

        job_id = self.enqueue("a", worker)
        self.assertTrue(wait_for(lambda: ("started", job_id) in self.reports))
        self.scheduler.cancel(job_id)
        gate.set()
        self.wait_idle()
        self.assertEqual(self.ran, [])
        self.assertIn(("stopped", job_id), self.reports)

    def test_a_stopped_job_a_failure_and_a_success_each_end_as_what_they_were(self):
        def stops(payload):
            raise JobCancelled()

        def breaks(payload):
            raise RuntimeError("expected failure")

        with mock.patch("builtins.print"):
            ids = [self.enqueue("stop", stops), self.enqueue("fail", breaks), self.enqueue("fine")]
            self.wait_idle()
        outcomes = {job: kind for kind, job in self.reports if kind != "started"}
        self.assertEqual(outcomes, {ids[0]: "stopped", ids[1]: "failed", ids[2]: "succeeded"})

    def test_shutdown_tells_the_running_job_to_stop_and_discards_the_line(self):
        reached = threading.Event()
        seen = []

        def worker(payload):
            reached.set()
            for _ in range(2000):
                if cancel_requested():
                    seen.append("told")
                    return {"message": "stopped itself"}
                time.sleep(0.005)
            return {"message": "never told"}

        self.enqueue("a", worker)
        waiting = self.enqueue("b")
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.scheduler.shutdown()
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("b.svg")))
        self.assertTrue(wait_for(lambda: seen == ["told"]))
        self.wait_idle()
        self.assertNotIn(("started", waiting), self.reports)


class ShutdownCleanupTests(LineTestCase):
    """Closing the Viewer while a job writes: the job stops and no half-written file is left, within a bound."""

    def stop_scheduler(self):
        self.release.set()
        self.scheduler.shutdown(wait=0.2)
        wait_for(lambda: self.scheduler.queue_depth == 0)

    def partial_job(self, name, *, cooperative, hold=None):
        """A worker that writes a partial file named to the scheduler, then renames it into place if it ever finishes.

        A cooperative one checks for the stop between its steps and deletes the file itself; the
        other ignores the stop, as a job stuck in one long call does.
        """
        partial, target = self.path(f".{name}.partial"), self.path(name)
        reached = threading.Event()
        gate = hold if hold is not None else self.release

        def worker(payload, job):
            job.note_partial(partial)
            try:
                with open(partial, "wb") as handle:
                    handle.write(b"half")
                reached.set()
                while not gate.wait(0.005):
                    if cooperative:
                        job.raise_if_cancelled()
                os.replace(partial, target)
                job.forget_partial(partial)
            finally:
                if cooperative and os.path.exists(partial):
                    os.remove(partial)
            return {"message": "done"}

        return worker, partial, target, reached

    def test_the_worker_gets_its_handle_when_it_names_a_second_parameter_job(self):
        seen = []

        def with_handle(payload, job):
            seen.append(("handle", type(job).__name__))
            return {}

        def without(payload):
            seen.append(("payload", payload))
            return {}

        self.enqueue("a", with_handle)
        self.enqueue("b", without)
        self.wait_idle()
        self.assertEqual(seen, [("handle", "JobHandle"), ("payload", "b")])

    def test_closing_stops_a_cooperative_job_at_once_and_its_partial_file_is_gone(self):
        worker, partial, target, reached = self.partial_job("report.xlsx", cooperative=True)
        self.enqueue("report.xlsx", worker)
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.assertTrue(os.path.exists(partial))
        started = time.monotonic()
        self.scheduler.shutdown()
        self.assertLess(time.monotonic() - started, 1.5, "a job that checks stops long before the bound")
        self.assertEqual(os.listdir(self.directory.name), [])

    def test_closing_returns_within_the_bound_when_the_job_ignores_the_stop_and_its_partial_file_is_deleted(self):
        worker, partial, target, reached = self.partial_job("stuck.png", cooperative=False)
        self.enqueue("stuck.png", worker)
        waiting = self.enqueue("queued")
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.assertTrue(os.path.exists(partial))
        started = time.monotonic()
        self.scheduler.shutdown(wait=0.3)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.25, "it waited for the job to stop")
        self.assertLess(elapsed, 2.0, "and no longer than the bound")
        self.assertFalse(os.path.exists(partial), "the scheduler deleted the partial file the job named")
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("queued.svg")))
        self.assertNotIn("queued", self.ran)
        # What the job does when it finally gets on cannot put a file in place: its partial file is gone.
        self.release.set()
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))
        self.assertFalse(os.path.exists(target))

    def test_a_file_the_worker_renamed_into_place_before_the_close_is_left_alone(self):
        finished = self.path("finished.png")
        gate = threading.Event()
        after = threading.Event()

        def worker(payload, job):
            partial = self.path(".finished.partial")
            job.note_partial(partial)
            with open(partial, "wb") as handle:
                handle.write(b"whole")
            os.replace(partial, finished)
            job.forget_partial(partial)
            after.set()
            gate.wait(BLOCKED_JOB_SECONDS)   # a later step that ignores the stop
            return {}

        self.enqueue("finished", worker)
        self.assertTrue(after.wait(STARTUP_TIMEOUT))
        self.scheduler.shutdown(wait=0.2)
        gate.set()
        self.assertEqual(Path(finished).read_bytes(), b"whole")
        self.assertEqual(os.listdir(self.directory.name), ["finished.png"])

    def test_closing_with_no_job_running_returns_at_once_and_twice_is_harmless(self):
        started = time.monotonic()
        self.scheduler.shutdown()
        self.scheduler.shutdown()
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(self.scheduler._thread.is_alive())


class LaneTests(LineTestCase):
    """Lanes: lines of jobs that run side by side, each in its own order, made by the first job that names them."""

    def lane_job(self, name, lane, worker=None, **options):
        return self.scheduler.enqueue("test", f"{name} description", name, worker or self.worker,
                                      self.path(f"{name}.out"), lane=lane, **options)

    def hold(self, name, lane, gate):
        """A job in ``lane`` that runs until ``gate`` is set; returns its number once it runs."""
        entered = threading.Event()

        def worker(payload):
            entered.set()
            gate.wait(BLOCKED_JOB_SECONDS)
            return {"message": name}

        job_id = self.lane_job(name, lane, worker)
        self.assertTrue(entered.wait(STARTUP_TIMEOUT))
        return job_id

    def lines(self):
        grouped = {}
        for entry in self.scheduler.snapshot():
            grouped.setdefault(entry.lane, []).append(entry.job.job_id)
        return grouped

    def test_a_job_that_names_a_new_lane_makes_it_and_run_beside_the_first_lane(self):
        files_gate, fold_gate = threading.Event(), threading.Event()
        self.addCleanup(files_gate.set)
        self.addCleanup(fold_gate.set)
        first = self.hold("a", "files", files_gate)
        second = self.hold("b", "folding", fold_gate)
        running = {entry.lane: entry.running for entry in self.scheduler.snapshot()}
        self.assertEqual(running, {"files": True, "folding": True}, "both run at the same time")
        self.assertEqual([name for name, _title in self.scheduler.lanes()], ["files", "folding"])
        self.assertEqual((first, second), (1, 2), "one count of job numbers for every lane")
        self.assertEqual(self.scheduler.queue_depth, 2)
        files_gate.set()
        fold_gate.set()
        self.wait_idle()

    def test_each_lane_runs_one_job_at_a_time_in_its_own_order_while_the_lanes_overlap(self):
        lock = threading.Lock()
        running = {"files": 0, "folding": 0}
        most = {"files": 0, "folding": 0, "both": 0}
        order = {"files": [], "folding": []}

        def make(lane):
            def worker(payload):
                with lock:
                    running[lane] += 1
                    most[lane] = max(most[lane], running[lane])
                    most["both"] = max(most["both"], sum(running.values()))
                    order[lane].append(payload)
                time.sleep(0.02)
                with lock:
                    running[lane] -= 1
                return {}
            return worker

        gate = threading.Event()
        self.hold("start-files", "files", gate)
        self.hold("start-folding", "folding", gate)
        for index in range(4):
            self.lane_job(f"f{index}", "files", make("files"))
            self.lane_job(f"g{index}", "folding", make("folding"))
        gate.set()
        self.wait_idle()
        self.assertEqual(order["files"], ["f0", "f1", "f2", "f3"])
        self.assertEqual(order["folding"], ["g0", "g1", "g2", "g3"])
        self.assertEqual((most["files"], most["folding"]), (1, 1))
        self.assertEqual(most["both"], 2, "the two lanes ran at the same time")

    def test_a_move_or_a_removal_in_one_lane_leaves_the_other_alone(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.hold("a-run", "files", gate)
        self.hold("b-run", "folding", gate)
        a1, a2, a3 = (self.lane_job(name, "files") for name in ("a1", "a2", "a3"))
        b1, b2 = (self.lane_job(name, "folding") for name in ("b1", "b2"))
        self.assertEqual(self.lines(), {"files": [1, a1, a2, a3], "folding": [2, b1, b2]})
        self.assertTrue(self.scheduler.move(a3, -2))
        self.assertEqual(self.lines(), {"files": [1, a3, a1, a2], "folding": [2, b1, b2]})
        # A job cannot be moved out of its lane: past the first waiting job it stays at the front of its own.
        self.assertFalse(self.scheduler.move(b1, -5))
        self.assertTrue(self.scheduler.move(b2, -5))
        self.assertEqual(self.lines(), {"files": [1, a3, a1, a2], "folding": [2, b2, b1]})
        self.assertTrue(self.scheduler.cancel(a1))
        self.assertEqual(self.lines(), {"files": [1, a3, a2], "folding": [2, b2, b1]})
        # Each lane counts its own places, the running job of each being place 1.
        places = {(entry.lane, entry.job.job_id): entry.position for entry in self.scheduler.snapshot()}
        self.assertEqual(places[("files", a3)], 2)
        self.assertEqual(places[("folding", b2)], 2)
        self.assertEqual(places[("folding", 2)], 1)
        gate.set()
        self.wait_idle()
        # Each lane ran its own jobs in the order it was left in.
        self.assertEqual([name for name in self.ran if name.startswith("a")], ["a3", "a2"])
        self.assertEqual([name for name in self.ran if name.startswith("b")], ["b2", "b1"])

    def test_the_snapshot_lists_lane_after_lane_in_the_order_the_lanes_were_made(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.hold("a", "folding", gate)   # the folding lane is made first by a job, after the default lane
        self.hold("b", "files", gate)
        self.lane_job("c", "folding")
        self.assertEqual([(entry.lane, entry.job.description) for entry in self.scheduler.snapshot()],
                         [("files", "b description"), ("folding", "a description"), ("folding", "c description")])
        self.assertEqual([(job.lane, running) for job, running in self.scheduler.listing()],
                         [("files", True), ("folding", True), ("folding", False)])
        gate.set()
        self.wait_idle()

    def test_a_lane_has_the_title_it_was_registered_with_or_the_job_gave(self):
        self.scheduler.register_lane("folding", "Folding")
        self.assertEqual(dict(self.scheduler.lanes())["folding"], "Folding")
        self.lane_job("x", "analysis", lane_title="Analysis")
        self.wait_idle()
        self.assertEqual(dict(self.scheduler.lanes())["analysis"], "Analysis")
        self.lane_job("y", "plain")
        self.wait_idle()
        self.assertEqual(dict(self.scheduler.lanes())["plain"], "plain", "a lane nobody titled is shown by its name")
        self.assertEqual(str(dict(self.scheduler.lanes())["files"]), "Files")

    def test_a_job_can_go_to_the_front_of_its_line_but_not_before_the_running_job(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        running = self.hold("run", "folding", gate)
        waiting = [self.lane_job(name, "folding") for name in ("w1", "w2")]
        urgent = self.lane_job("urgent", "folding", front=True)
        self.assertEqual(self.lines()["folding"], [running, urgent, *waiting])
        gate.set()
        self.wait_idle()

    def test_every_path_a_job_reserves_is_held_until_it_ends_and_a_second_job_may_not_take_one(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        extra = [self.path("one.pdb"), self.path("one.residues.json")]
        entered = threading.Event()

        def worker(payload):
            entered.set()
            gate.wait(BLOCKED_JOB_SECONDS)
            return {}

        job_id = self.scheduler.enqueue("test", "multi", None, worker, self.path("main.pdb"),
                                        lane="folding", extra_paths=extra)
        self.assertTrue(entered.wait(STARTUP_TIMEOUT))
        for path in (self.path("main.pdb"), *extra):
            self.assertTrue(self.scheduler.is_output_path_reserved(path), path)
            self.assertEqual(self.scheduler.reservation_holder(path), job_id)
            with self.assertRaises(FileExistsError):
                # Taken in another lane too, whichever of the job's paths it is.
                self.scheduler.enqueue("test", "clash", None, worker, self.path("other.out"),
                                       lane="files", extra_paths=[path])
        self.assertEqual(self.scheduler.queue_depth, 1, "a refused job is not queued")
        self.assertIsNone(self.scheduler.reservation_holder(self.path("free.pdb")))
        gate.set()
        self.wait_idle()
        for path in (self.path("main.pdb"), *extra):
            self.assertFalse(self.scheduler.is_output_path_reserved(path))

    def test_the_paths_of_a_waiting_job_are_released_when_it_is_removed(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.hold("run", "folding", gate)
        waiting = self.scheduler.enqueue("test", "multi", None, self.worker, self.path("w.pdb"), lane="folding",
                                         extra_paths=[self.path("w.residues.json")])
        self.assertTrue(self.scheduler.cancel(waiting))
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("w.pdb")))
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("w.residues.json")))
        gate.set()
        self.wait_idle()

    def test_the_cleanup_function_is_called_once_however_the_job_ends(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        cleaned = []

        def job(name, worker=None, **options):
            return self.lane_job(name, "folding", worker, cleanup=lambda: cleaned.append(name), **options)

        def breaks(payload):
            raise RuntimeError("expected")

        def stops(payload):
            raise JobCancelled()

        self.hold("run", "folding", gate)
        removed = job("removed")
        self.scheduler.cancel(removed)
        job("done")
        job("failed", breaks)
        job("stopped", stops)
        with mock.patch("builtins.print"):
            gate.set()
            self.wait_idle()
        self.assertEqual(sorted(cleaned), ["done", "failed", "removed", "stopped"])

    def test_the_cleanup_function_of_a_job_discarded_by_closing_is_called(self):
        gate = threading.Event()
        cleaned = []
        self.hold("run", "folding", gate)
        self.lane_job("waiting", "folding", cleanup=lambda: cleaned.append("waiting"))
        self.scheduler.shutdown(wait=0.2)
        gate.set()
        self.assertEqual(cleaned, ["waiting"])

    def test_a_cleanup_that_fails_does_not_stop_the_lane(self):
        def broken():
            raise OSError("cannot delete")

        with mock.patch("builtins.print") as said:
            self.lane_job("a", "folding", cleanup=broken)
            self.lane_job("b", "folding")
            self.wait_idle()
        self.assertIn("cannot delete", str(said.call_args_list))
        self.assertEqual(self.ran, ["a", "b"])

    def test_on_finished_is_told_how_the_job_ended_on_the_gui_thread(self):
        told, threads = [], []

        def finished(job, outcome, error):
            told.append((job.job_id, outcome, type(error).__name__ if error is not None else None))
            threads.append(threading.get_ident())

        def breaks(payload):
            raise ValueError("expected")

        def stops(payload):
            raise JobCancelled()

        gate = threading.Event()
        self.hold("run", "folding", gate)
        dropped = self.lane_job("dropped", "folding", on_finished=finished)
        self.scheduler.cancel(dropped)
        ok = self.lane_job("ok", "folding", on_finished=finished)
        bad = self.lane_job("bad", "folding", breaks, on_finished=finished)
        halted = self.lane_job("halted", "folding", stops, on_finished=finished)
        with mock.patch("builtins.print"):
            gate.set()
            self.wait_idle()
        self.assertEqual(sorted(told), sorted([(dropped, "cancelled", None), (ok, "succeeded", None),
                                              (bad, "failed", "ValueError"), (halted, "cancelled", None)]))
        self.assertEqual(set(threads), {threading.get_ident()})

    def test_a_follow_up_that_fails_does_not_stop_the_reports(self):
        def broken(job, outcome, error):
            raise RuntimeError("follow-up failed")

        with mock.patch("builtins.print") as said:
            self.lane_job("a", "folding", on_finished=broken)
            self.lane_job("b", "folding")
            self.wait_idle()
        self.assertIn("follow-up failed", str(said.call_args_list))
        self.assertIn(("succeeded", 2), self.reports)

    def test_a_failed_job_is_kept_with_its_reason_and_log_until_it_is_dismissed(self):
        log = self.path("run.log")

        def breaks(payload, job):
            job.set_log_path(log)
            raise RuntimeError(Message("Folding failed: {why}", why="out of memory"))

        with mock.patch("builtins.print"):
            job_id = self.lane_job("bad", "folding", breaks)
            self.wait_idle()
        (failure,) = self.scheduler.failures()
        self.assertEqual((failure.job_id, failure.lane, failure.log_path), (job_id, "folding", log))
        self.assertEqual(display_text(failure.error), "Folding failed: out of memory")
        self.assertEqual(self.scheduler.log_path_of(job_id), log, "the log is found by the job's number")
        self.assertTrue(self.scheduler.dismiss_failure(job_id))
        self.assertFalse(self.scheduler.dismiss_failure(job_id))
        self.assertEqual(self.scheduler.failures(), [])
        self.assertIsNone(self.scheduler.log_path_of(job_id))

    def test_only_the_last_few_failures_are_kept_and_a_log_named_at_enqueue_is_kept_too(self):
        from Background_Job_Scheduler import KEPT_FAILURES

        def breaks(payload):
            raise RuntimeError("expected")

        with mock.patch("builtins.print"):
            for index in range(KEPT_FAILURES + 2):
                self.lane_job(f"bad{index}", "folding", breaks, log_path=self.path(f"bad{index}.log"))
            self.wait_idle()
        kept = self.scheduler.failures()
        self.assertEqual([failure.job_id for failure in kept], list(range(3, KEPT_FAILURES + 3)))
        self.assertEqual(kept[0].log_path, self.path("bad2.log"))

    def test_a_job_that_succeeds_or_is_stopped_leaves_no_failure_behind(self):
        def stops(payload):
            raise JobCancelled()

        self.lane_job("ok", "folding")
        self.lane_job("halted", "folding", stops)
        self.wait_idle()
        self.assertEqual(self.scheduler.failures(), [])

    def test_progress_and_a_log_a_worker_reports_reach_the_snapshot_and_tell_the_card(self):
        told, gate, entered = [], threading.Event(), threading.Event()
        self.addCleanup(gate.set)
        self.scheduler.changed.connect(lambda: told.append(1))

        def worker(payload, job):
            job.progress("Folding 2 of 3: node_b")
            job.set_log_path(self.path("fold.log"))
            entered.set()
            gate.wait(BLOCKED_JOB_SECONDS)
            return {}

        job_id = self.lane_job("fold", "folding", worker)
        self.assertTrue(entered.wait(STARTUP_TIMEOUT))
        self.assertTrue(wait_for(lambda: bool(told)))
        (entry,) = self.scheduler.snapshot()
        self.assertEqual((entry.progress, entry.log_path), ("Folding 2 of 3: node_b", self.path("fold.log")))
        self.assertEqual(self.scheduler.log_path_of(job_id), self.path("fold.log"))
        gate.set()
        self.wait_idle()

    def test_a_job_that_wraps_a_process_registers_the_function_that_stops_it(self):
        # The scheduler knows nothing of the process: the worker hands it the stop function.
        stopped, process_running, entered = threading.Event(), threading.Event(), threading.Event()
        calls = []

        def worker(payload, job):
            process_running.set()
            job.on_cancel(lambda: (calls.append("stop the process"), stopped.set()))
            entered.set()
            stopped.wait(BLOCKED_JOB_SECONDS)
            job.raise_if_cancelled()
            return {}

        job_id = self.lane_job("wrapper", "folding", worker)
        self.assertTrue(entered.wait(STARTUP_TIMEOUT))
        self.assertTrue(self.scheduler.cancel(job_id))
        self.wait_idle()
        self.assertEqual(calls, ["stop the process"])
        self.assertIn(("stopped", job_id), self.reports)

    def test_closing_stops_the_running_job_of_every_lane_and_discards_every_line(self):
        reached = {"files": threading.Event(), "folding": threading.Event()}
        told = []

        def make(lane):
            def worker(payload, job):
                reached[lane].set()
                for _ in range(2000):
                    if job.requested:
                        told.append(lane)
                        return {}
                    time.sleep(0.005)
                return {}
            return worker

        self.lane_job("a", "files", make("files"))
        self.lane_job("b", "folding", make("folding"))
        waiting = self.lane_job("c", "folding")
        for event in reached.values():
            self.assertTrue(event.wait(STARTUP_TIMEOUT))
        self.scheduler.shutdown()
        self.assertEqual(sorted(told), ["files", "folding"])
        self.assertFalse(self.scheduler.is_output_path_reserved(self.path("c.out")))
        self.assertEqual(self.scheduler.queue_depth, 0)
        self.assertNotIn(("started", waiting), self.reports)

    def test_a_lane_that_is_stuck_does_not_make_closing_wait_longer_than_the_bound_for_all_of_them(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.hold("stuck-files", "files", gate)
        self.hold("stuck-folding", "folding", gate)
        started = time.monotonic()
        self.scheduler.shutdown(wait=0.4)
        self.assertLess(time.monotonic() - started, 1.5, "one bound for every lane, not one for each")
        gate.set()


class SchedulerNamesNoCommandTests(unittest.TestCase):
    """The scheduler and what it shares with a job know no command: a plugin uses them as they are."""

    SRC = Path(__file__).resolve().parents[1] / "src"
    FILES = ("Background_Job_Scheduler.py", "utilities/Job_Cancellation.py")

    @staticmethod
    def command_words():
        """The names of the commands that queue jobs: the command modules that use the scheduler."""
        commands = Path(__file__).resolve().parents[1] / "src" / "commands"
        return sorted(path.stem for path in commands.glob("*.py")
                      if path.stem != "__init__" and "scheduler.enqueue(" in path.read_text(encoding="utf-8"))

    def mentions(self, relative):
        """Where the file names a command: an import of a command module, or a string a program shows or compares.

        Docstrings and comments are prose and may use the words (a "label" is also a label); the code may not.
        """
        return self.mentions_in((self.SRC / relative).read_text(encoding="utf-8"))

    def mentions_in(self, source):
        import ast
        import re

        words = self.command_words()
        pattern = re.compile(r"\b(" + "|".join(words) + r")\b", re.IGNORECASE)
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                found += [(node.lineno, match.group(0)) for match in pattern.finditer(node.value)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    found += [(node.lineno, part) for part in (name or "").split(".") if part in words]
        return found

    def test_the_commands_that_queue_jobs_are_found(self):
        self.assertEqual(self.command_words()[:4], ["esmfold", "label", "logo", "print"])

    def test_the_check_finds_a_command_named_in_code_and_leaves_prose_alone(self):
        flagged = "\n".join([
            '"""A label is a label."""',
            "import commands.logo",
            'value = "print -> map.png"',
            "if name == 'esmfold':",
            "    pass",
        ])
        self.assertEqual([word for _line, word in self.mentions_in(flagged)], ["logo", "print", "esmfold"])
        prose = "\n".join([
            '"""The print of a label."""',
            "# print label logo",
            'print("a message")',
            'label = {"key": "value"}',
        ])
        self.assertEqual(self.mentions_in(prose), [])

    def test_no_file_of_the_scheduler_names_a_command(self):
        for relative in self.FILES:
            with self.subTest(file=relative):
                self.assertEqual(self.mentions(relative), [])


class CloseWiringTests(unittest.TestCase):
    """Every way the Viewer closes ends at the scheduler's shutdown, which stops the running job and deletes its partial files."""

    SRC = Path(__file__).resolve().parents[1] / "src"

    def test_the_window_and_the_application_quit_both_close_the_scheduler(self):
        source = (self.SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8")
        # The window's close button (the canvas's close event) and the application's quit, which every
        # other way of closing comes to: the launcher's, and the MCP server's shutdown request below.
        self.assertRegex(source, r"canvas\.events\.close\.connect\(\s*lambda _event: self\.background_job_scheduler\.shutdown\(\)")
        self.assertIn("qapp.aboutToQuit.connect(self.background_job_scheduler.shutdown)", source)

    def test_the_mcp_servers_close_asks_the_application_to_quit_before_it_ends_the_process(self):
        server = (self.SRC / "web_ui" / "Web_Server.py").read_text(encoding="utf-8")
        shutdown = server[server.index('"/api/mcp/v1/shutdown"'):]
        self.assertIn('qapp, "quit"', shutdown[:2000])
        client = (self.SRC / "mcp_server" / "viewer" / "Viewer_Client.py").read_text(encoding="utf-8")
        close = client[client.index("async def close_session"):]
        # It posts the shutdown request first and ends the process only after the timeout.
        self.assertLess(close.index("/api/mcp/v1/shutdown"), close.index("_terminate_tree"))

    def test_the_wait_is_shorter_than_the_five_seconds_the_mcp_server_gives_a_viewer_to_exit(self):
        from Background_Job_Scheduler import SHUTDOWN_WAIT_SECONDS

        self.assertGreaterEqual(SHUTDOWN_WAIT_SECONDS, 2.0)
        self.assertLessEqual(SHUTDOWN_WAIT_SECONDS, 3.0)


class GuiCallTests(LineTestCase):
    """A job asks for a function to be run on the GUI thread, a dialog for instance, and waits for its answer."""

    def run_job(self, body):
        """Queue a job whose worker is body(job); return (job id, what the body saw)."""
        seen = {}

        def worker(payload, job):
            seen["thread"] = threading.current_thread()
            try:
                seen["result"] = body(job)
            except BaseException as error:  # JobCancelled included: the job reports it
                seen["error"] = error
                raise
            return {"message": "done"}

        job_id = self.enqueue("gui", worker)
        return job_id, seen

    def test_the_function_runs_on_the_gui_thread_and_its_answer_comes_back(self):
        ran_on = []
        job_id, seen = self.run_job(lambda job: job.call_in_gui(lambda: ran_on.append(threading.current_thread()) or "typed"))
        self.wait_idle()
        self.assertEqual(seen["result"], "typed")
        self.assertIsNot(seen["thread"], threading.main_thread(), "the job ran on a lane thread")
        self.assertEqual(ran_on, [threading.main_thread()], "the function ran on the GUI thread")

    def test_an_error_of_the_function_is_raised_in_the_job(self):
        def fails():
            raise ValueError("no dialog")

        job_id, seen = self.run_job(lambda job: job.call_in_gui(fails))
        self.wait_idle()
        self.assertIsInstance(seen["error"], ValueError)
        self.assertIn(("failed", job_id), self.reports)

    def test_a_stop_while_the_function_waits_closes_it_and_cancels_the_job(self):
        shown, closed = threading.Event(), threading.Event()
        order = []
        job = []

        def dialog():
            # The GUI thread, inside a modal loop, where the user's click on Stop is handled: that stops the job,
            # and the dialog ends when close() is run from the same event loop.
            shown.set()
            QtCore.QTimer.singleShot(0, lambda: self.scheduler.cancel(job[0]))
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while not closed.is_set() and time.monotonic() < deadline:
                QtCore.QCoreApplication.processEvents()
                time.sleep(0.005)
            order.append("dialog ended")
            return "late answer"

        def close():
            order.append("closed")
            closed.set()

        job_id, seen = self.run_job(lambda handle: handle.call_in_gui(dialog, close=close))
        job.append(job_id)
        self.assertTrue(wait_for(shown.is_set))
        self.wait_idle()
        self.assertIsInstance(seen["error"], JobCancelled)
        self.assertEqual(order, ["closed", "dialog ended"])
        self.assertIn(("stopped", job_id), self.reports)

    def test_a_job_already_told_to_stop_asks_for_nothing(self):
        ran = []
        started = threading.Event()

        def body(job):
            started.set()
            while not job.requested:
                time.sleep(0.005)
            return job.call_in_gui(lambda: ran.append(1))

        job_id, seen = self.run_job(body)
        self.assertTrue(started.wait(STARTUP_TIMEOUT))
        self.scheduler.cancel(job_id)
        self.wait_idle()
        self.assertIsInstance(seen["error"], JobCancelled)
        self.assertEqual(ran, [])

    def test_without_a_scheduler_the_function_is_just_called(self):
        from utilities.Job_Cancellation import JobHandle
        self.assertEqual(JobHandle().call_in_gui(lambda: 7), 7)


class LinePortalTests(LineTestCase):
    """The command portal's record of each job: its place in the line, and the events it gets."""

    @staticmethod
    def portal_context(record):
        portal = SimpleNamespace(append_output=lambda *args: None, changed=SimpleNamespace(emit=lambda *args: None))
        return ExecutionContext(portal, "request", record)

    def portal_job(self, name, worker=None):
        record = ViewerCommandPortal.new_command(name)
        with bind(self.portal_context(record)):
            job_id = self.enqueue(name, worker)
        return job_id, record["jobs"][0]

    @staticmethod
    def places(entries):
        return [(entry["job_id"], entry.get("position"), entry.get("waiting_for")) for entry in entries]

    @staticmethod
    def settle():
        wait_for(lambda: False, 0.1)

    def test_every_waiting_job_records_its_place_and_the_job_ahead_as_the_line_changes(self):
        blocker = self.hold_the_thread()
        first, one = self.portal_job("a")
        second, two = self.portal_job("b")
        third, three = self.portal_job("c")
        self.settle()
        self.assertEqual(self.places([one, two, three]),
                         [(str(first), 2, str(blocker)), (str(second), 3, str(first)), (str(third), 4, str(second))])
        self.scheduler.move(third, -2)
        self.settle()
        self.assertEqual(self.places([one, two, three]),
                         [(str(first), 3, str(third)), (str(second), 4, str(first)), (str(third), 2, str(blocker))])
        self.scheduler.cancel(first)
        self.settle()
        self.assertEqual((two["position"], two["waiting_for"]), (3, str(third)))
        self.assertEqual((three["position"], three["waiting_for"]), (2, str(blocker)))
        self.assertEqual(one["status"], "cancelled")
        self.release.set()
        self.wait_idle()

    def test_the_record_gets_succeeded_for_a_job_that_ran_and_cancelled_for_one_removed(self):
        self.hold_the_thread()
        _kept, kept = self.portal_job("a")
        dropped, removed = self.portal_job("b")
        self.scheduler.cancel(dropped)
        self.release.set()
        self.wait_idle()
        self.assertEqual(kept["status"], "succeeded")
        self.assertEqual(removed["status"], "cancelled")
        self.assertIn("cancelled before it started", removed["message"])

    def test_a_running_job_stopped_is_cancelled_in_the_record_with_its_reason(self):
        reached = threading.Event()
        record = ViewerCommandPortal.new_command("logo")

        def worker(payload):
            reached.set()
            while True:
                raise_if_cancelled()
                time.sleep(0.005)

        with bind(self.portal_context(record)):
            job_id = self.enqueue("stopme", worker)
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.scheduler.cancel(job_id)
        self.wait_idle()
        entry = record["jobs"][0]
        self.assertEqual(entry["status"], "cancelled")
        self.assertIn("stopped before it finished", entry["message"])
        self.assertEqual(record["outcome"], "cancelled")
        self.assertEqual(entry["position"], 1)

    def test_the_record_says_what_the_job_is_its_lane_its_progress_and_when_it_started_and_ended(self):
        reached, go_on = threading.Event(), threading.Event()

        def worker(payload, job):
            job.progress(Message("Half way"))
            reached.set()
            go_on.wait(BLOCKED_JOB_SECONDS)
            return {"message": "done"}

        self.addCleanup(go_on.set)
        job_id, entry = self.portal_job("a", worker)
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))
        self.settle()
        self.assertEqual(
            {key: entry[key] for key in ("kind", "lane", "lane_title", "description", "status", "progress",
                                         "cancellable", "stopping", "position", "waiting_for", "finished_at")},
            {"kind": "logo", "lane": "files", "lane_title": "Files", "description": "logo -> a.svg",
             "status": "running", "progress": "Half way", "cancellable": True, "stopping": False,
             "position": 1, "waiting_for": None, "finished_at": None})
        self.assertTrue(entry["started_at"], "the time it began")
        go_on.set()
        self.wait_idle()
        self.assertEqual(entry["status"], "succeeded")
        self.assertTrue(entry["finished_at"] >= entry["started_at"])
        self.assertEqual((entry["cancellable"], entry["stopping"]), (False, False))

    def test_a_job_with_no_check_point_told_to_stop_is_stopping_until_it_ends_and_then_completed_not_cancelled(self):
        reached, go_on = threading.Event(), threading.Event()
        record = ViewerCommandPortal.new_command("logo")

        def worker(payload):
            reached.set()
            go_on.wait(BLOCKED_JOB_SECONDS)        # a worker that never asks whether it was told to stop
            return {"message": "written"}

        self.addCleanup(go_on.set)
        with bind(self.portal_context(record)):
            job_id = self.enqueue("stubborn", worker)
        entry = record["jobs"][0]
        self.assertTrue(reached.wait(STARTUP_TIMEOUT))

        self.assertEqual(self.scheduler.cancel_many([job_id]), [job_id])
        self.settle()

        # Not cancelled: the job still runs, and the record says it is being told to stop.
        self.assertEqual((entry["status"], entry["stopping"], entry["cancellable"]), ("running", True, False))
        self.assertEqual(self.scheduler.cancel_many([job_id]), [], "told once")
        self.assertEqual([(listed.job.job_id, listed.stopping) for listed in self.scheduler.snapshot()],
                         [(job_id, True)])
        self.assertNotIn(("stopped", job_id), self.reports)

        go_on.set()
        self.wait_idle()
        self.settle()
        # It could not stop, so it completed; the record's final status and message say so.
        self.assertEqual(self.reports, [("started", job_id), ("succeeded", job_id)])
        self.assertEqual((entry["status"], entry["stopping"], entry["stop_requested"]), ("succeeded", False, True))
        self.assertIn("told to stop but had no step left to stop at", entry["message"])
        self.assertIn("written", entry["message"])


if __name__ == "__main__":
    unittest.main()
