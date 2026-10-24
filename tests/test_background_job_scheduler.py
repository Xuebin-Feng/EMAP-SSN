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


if __name__ == "__main__":
    unittest.main()
