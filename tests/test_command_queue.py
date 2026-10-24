"""The Viewer's command queue (Command_Queue): order, selection, cancelling, undo, failures, MCP.

The compute steps run on a ManualExecutor, which runs one only when a test says
so, on the test's own thread: every interleaving below is decided by the test,
with no sleeps. The last class runs a real Leiden clustering on the queue's
worker thread and measures the GUI thread's timer meanwhile.
"""
import io
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtCore, QtWidgets  # noqa: E402
import Command_Engine as ce  # noqa: E402
from Command_Queue import CommandQueue, ManualExecutor, ThreadExecutor  # noqa: E402
from Viewer_Command_Portal import ViewerCommandPortal, bind  # noqa: E402

# Two 4-cliques joined by the bridge 3-4, and two nodes with no edge. Jaccard
# at 0.5 with MIN_SIZE 3 makes the cliques cluster_1 (nodes 0-3) and
# cluster_2 (4-7); nodes 8 and 9 are noise.
CLIQUE_EDGES = [(a, b) for group in ((0, 1, 2, 3), (4, 5, 6, 7)) for i, a in enumerate(group) for b in group[i + 1:]]
EDGES = CLIQUE_EDGES + [(3, 4)]
CLUSTER = "cluster jaccard 0.5 3"
RED = (1.0, 0.0, 0.0, 1.0)


class QueueViewer:
    """A stand-in Viewer with a real command queue; drawing is counted, not drawn."""

    def __init__(self, executor=None):
        n = 10
        self.n_nodes = n
        self.full_headers = [f"n{i}" for i in range(n)]
        self.edges = np.array(EDGES, dtype=np.int32)
        self.edge_scores = np.full(len(EDGES), 50.0, dtype=np.float32)
        self.visible_mask = np.ones(n, dtype=bool)
        self.selected_indices = []
        self.selection_epoch = 0
        self.cluster_labels = None
        self.group_labels = [set() for _ in range(n)]
        self.metadata = {}
        self.alignment = None
        self.current_colors = np.zeros((n, 4))
        self.current_sizes = np.ones(n)
        self.current_shapes = np.full(n, "disc", dtype=object)
        self.console_text = SimpleNamespace(text="")
        self.console_mode = False
        self.tooltip = SimpleNamespace(text="")
        self.hovered_node_idx = None
        self.selected_node_idx = None
        self.command_history = []
        self._history_folder = tempfile.TemporaryDirectory()
        self.history_file = str(Path(self._history_folder.name) / "history.txt")
        self.saved = 0
        self.undone = 0
        self.redone = 0
        self.status_lines = []
        self.is_multi_dragging = False
        self.is_box_selecting = False
        self.command_queue = CommandQueue(self, executor=executor if executor is not None else ManualExecutor())

    def _save_state(self):
        self.saved += 1

    def _do_undo(self, announce=True):
        self.undone += 1
        return True

    def _do_redo(self, announce=True):
        self.redone += 1
        return True

    def _refresh_after_history(self):
        pass

    def update_nodes(self):
        pass

    def update_edges(self):
        pass

    def update_selection_visual(self, broadcast=True):
        pass

    def promote_nodes(self, indices):
        pass

    def broadcast_metadata_state(self):
        pass

    def broadcast_event(self, event):
        pass

    def update_console_background(self):
        pass

    def set_background_job_status(self, message):
        self.status_lines.append(str(message))

    # --- what the user does ------------------------------------------------

    def type(self, line):
        """Type line at the command line and press Enter."""
        with bind(None):
            ce.execute_command(self, line, record_history=False)

    def select_by_hand(self, indices):
        """Select nodes with the mouse, as the Viewer's handlers do."""
        self.selected_indices = list(indices)
        self.selection_epoch += 1

    def close(self):
        """Remove the history folder now: a scheduler that points back at the viewer keeps it alive to the end of the run."""
        self._history_folder.cleanup()

    @property
    def queue(self):
        return self.command_queue

    @property
    def executor(self):
        return self.command_queue.executor


class QueueTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)
        self.viewer = QueueViewer()
        self.addCleanup(self.viewer.close)
        self.addCleanup(self.viewer.command_queue.shutdown)

    def settle(self, timeout=10):
        """Process Qt events until the queue has no step left to take without a compute finishing."""
        deadline = time.monotonic() + timeout
        queue = self.viewer.command_queue
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not queue._wake.isActive() and not any(job.status == 'applying' for job in queue.jobs):
                self.app.processEvents()
                if not queue._wake.isActive():
                    return
            time.sleep(0.002)
        self.fail("the command queue did not settle")

    def finish_compute(self):
        """Let the oldest compute step run, and its result apply."""
        job = self.viewer.executor.finish()
        self.settle()
        return job

    def red_nodes(self):
        return [i for i in range(self.viewer.n_nodes) if tuple(self.viewer.current_colors[i]) == RED]


class OrderTests(QueueTestCase):
    def test_commands_given_while_a_cluster_computes_run_after_it_on_its_result(self):
        """cluster, select #cluster_1#, color red typed fast: as if typed one by one."""
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("select #cluster_1#")
        viewer.type("color red")
        self.settle()
        self.assertEqual([job.status for job in viewer.queue.pending()], ['running', 'queued', 'queued'])
        self.assertIsNone(viewer.cluster_labels)
        self.assertEqual(viewer.selected_indices, [])
        self.assertEqual(self.red_nodes(), [])
        self.assertIn("waiting for #1 cluster jaccard 0.5 3", self.output.getvalue())

        self.finish_compute()

        self.assertEqual(viewer.queue.pending(), [])
        np.testing.assert_array_equal(viewer.cluster_labels, [1, 1, 1, 1, 2, 2, 2, 2, -1, -1])
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2, 3])
        self.assertEqual(self.red_nodes(), [0, 1, 2, 3])
        self.assertEqual(viewer.saved, 2)  # cluster and color; select makes no undo step
        self.assertEqual([job.status for job in viewer.queue.finished], ['succeeded'] * 3)

    def test_with_nothing_waiting_a_light_command_runs_at_once_and_is_no_job(self):
        self.viewer.select_by_hand([8])
        self.viewer.type("color red")
        self.assertEqual(self.red_nodes(), [8])
        self.assertEqual(list(self.viewer.queue.finished), [])

    def test_a_heavy_command_given_while_nothing_waits_starts_at_once(self):
        self.viewer.type(CLUSTER)
        job = self.viewer.queue.pending()[0]
        self.assertEqual(job.status, 'running')
        self.assertEqual(len(self.viewer.executor.tasks), 1)
        self.assertEqual(self.viewer.console_text.text, "Clustering (JACCARD)...")

    def test_typos_are_refused_at_once_even_while_commands_wait(self):
        self.viewer.type(CLUSTER)
        self.viewer.type("cluster mcl 0.5")
        self.viewer.type("colr red")
        self.assertEqual(len(self.viewer.queue.pending()), 1)
        self.assertIn("MCL inflation must be between 1.1 and 10.0; got 0.5.", self.output.getvalue())
        self.assertEqual(self.viewer.console_text.text, "Unknown command: colr")

    def test_a_cluster_list_given_while_a_cluster_computes_lists_the_new_clusters(self):
        self.viewer.type(CLUSTER)
        self.viewer.type("cluster list")
        self.finish_compute()
        self.assertEqual(self.viewer.console_text.text, "Listed 2 clusters in console.")

    def test_help_runs_at_once_while_commands_wait(self):
        self.viewer.type(CLUSTER)
        self.viewer.type("color help")
        self.assertEqual(len(self.viewer.queue.pending()), 1)
        self.assertIn("Advanced Coloring & Highlighting Tool", self.output.getvalue())

    def test_a_command_module_with_a_waiting_job_is_not_reloaded(self):
        self.viewer.type(CLUSTER)
        with mock.patch("importlib.reload") as reload:
            self.viewer.type("cluster list")
            self.viewer.type("select #cluster_1#")
        reloaded = [call.args[0].__name__ for call in reload.call_args_list]
        self.assertNotIn("commands.cluster", reloaded)
        self.assertIn("commands.select", reloaded)


class SelectionTimelineTests(QueueTestCase):
    def test_a_selection_made_by_hand_while_a_cluster_computes_is_what_a_later_color_uses(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.select_by_hand([8, 9])
        viewer.type("color red")
        self.finish_compute()
        self.assertEqual(self.red_nodes(), [8, 9])
        self.assertEqual(viewer.selected_indices, [8, 9])

    def test_a_hand_selection_after_a_queued_select_stays_and_only_later_commands_use_it(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("select #cluster_1#")
        viewer.type("color red")
        viewer.select_by_hand([9])
        viewer.type("color 2x")
        self.finish_compute()
        # color red used the select's nodes; the select did not replace the newer hand selection.
        self.assertEqual(self.red_nodes(), [0, 1, 2, 3])
        self.assertEqual(viewer.selected_indices, [9])
        self.assertEqual([i for i in range(10) if viewer.current_sizes[i] != 1], [9])

    def test_select_modes_read_the_selection_they_were_given(self):
        viewer = self.viewer
        viewer.select_by_hand([8])
        viewer.type(CLUSTER)
        viewer.type("select add #cluster_2#")
        viewer.select_by_hand([0])
        viewer.type("color red")
        self.finish_compute()
        # select add added cluster_2 to [8], the selection it was given, not to the newer [0].
        job = viewer.queue.finished[1]
        self.assertEqual(sorted(job.selection_out.tolist()), [4, 5, 6, 7, 8])
        self.assertEqual(viewer.selected_indices, [0])
        self.assertEqual(self.red_nodes(), [0])

    def test_hiding_the_selection_given_leaves_none_for_the_next_command(self):
        viewer = self.viewer
        viewer.select_by_hand([8, 9])
        viewer.type(CLUSTER)
        viewer.type("hide")
        viewer.type("color red")
        self.finish_compute()
        self.assertFalse(viewer.visible_mask[8] or viewer.visible_mask[9])
        self.assertEqual(viewer.selected_indices, [])
        self.assertEqual(self.red_nodes(), [])
        self.assertEqual(viewer.queue.finished[-1].status, 'succeeded')  # no match is no failure

    def test_a_hide_keeps_hidden_nodes_out_of_a_newer_hand_selection(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type('hide "n9"')
        viewer.select_by_hand([8, 9])
        self.finish_compute()
        self.assertEqual(viewer.selected_indices, [8])

    def test_a_cancelled_select_passes_on_the_selection_it_was_given(self):
        viewer = self.viewer
        viewer.select_by_hand([9])
        viewer.type(CLUSTER)
        viewer.type("select #cluster_1#")
        viewer.type("color red")
        viewer.queue.cancel(viewer.queue.pending()[1])
        self.finish_compute()
        self.assertEqual(self.red_nodes(), [9])

    def test_query_reads_the_selection_it_was_given_for_its_fallback(self):
        job = SimpleNamespace(bound_selection=lambda viewer: np.array([1, 2]))
        with ce.command_job(job):
            self.assertEqual(ce.command_selection(self.viewer), [1, 2])
            np.testing.assert_array_equal(np.flatnonzero(ce.get_selected_mask(self.viewer)), [1, 2])
        self.assertEqual(ce.command_selection(self.viewer), [])


class CancelAndUndoTests(QueueTestCase):
    def test_jobs_cancel_removes_a_waiting_command_and_the_others_run(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("select #cluster_1#")
        viewer.type("color red")
        viewer.type("jobs cancel 3")
        self.assertIn("Cancelled 1 job: #3 color red.", self.output.getvalue())
        self.finish_compute()
        self.assertEqual(self.red_nodes(), [])
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2, 3])
        self.assertEqual([job.status for job in viewer.queue.finished], ['cancelled', 'succeeded', 'succeeded'])

    def test_cancelling_a_running_compute_throws_its_result_away(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.type("jobs cancel 1")
        # The command after it went on at once; the compute's result, when it comes, is dropped.
        self.settle()
        self.assertEqual(viewer.queue.pending(), [])
        self.finish_compute()
        self.assertIsNone(viewer.cluster_labels)
        self.assertEqual(viewer.saved, 0)  # color red matched nothing: no selection

    def test_jobs_cancel_with_an_unknown_number_cancels_nothing(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.type("jobs cancel 2 9")
        self.assertEqual(len(viewer.queue.pending()), 2)
        self.assertIn("Error: No job #9 waits or runs; nothing was cancelled.", self.output.getvalue())

    def test_jobs_cancel_all_and_the_listing(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.type("jobs")
        listing = self.output.getvalue()
        self.assertIn("#1     running", listing)
        self.assertIn("#2     waiting", listing)
        self.assertEqual(viewer.console_text.text, "2 jobs waiting or running; the list is in the Console.")
        viewer.type("jobs cancel all")
        self.assertEqual(viewer.queue.pending(), [])
        viewer.type("jobs cancel")
        self.assertEqual(viewer.console_text.text, "Nothing to cancel: no command waits or runs.")

    def test_undo_by_hand_cancels_the_newest_waiting_command_then_undoes(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.type("undo")
        self.assertEqual([job.line for job in viewer.queue.pending()], [CLUSTER])
        self.assertIn("Undo: cancelled 1 command that had not finished: #2 color red.", self.output.getvalue())
        viewer.type("undo")
        self.assertEqual(viewer.queue.pending(), [])
        self.assertEqual(viewer.undone, 0)
        self.finish_compute()
        self.assertIsNone(viewer.cluster_labels)
        viewer.type("undo")
        self.assertEqual(viewer.undone, 1)

    def test_undo_n_cancels_what_waits_and_undoes_the_rest(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.type("undo 3")
        self.assertEqual(viewer.queue.pending(), [])
        self.assertEqual(viewer.undone, 1)

    def test_ctrl_z_and_the_sheet_undo_cancel_the_newest_waiting_command(self):
        from EMAPSSN_Viewer import MainViewer
        from web_ui import meta_backend
        viewer = self.viewer
        viewer.undo_by_hand = types.MethodType(MainViewer.undo_by_hand, viewer)
        viewer.redo_by_hand = types.MethodType(MainViewer.redo_by_hand, viewer)
        viewer.type(CLUSTER)
        viewer.type("color red")
        self.assertTrue(viewer.undo_by_hand())
        self.assertEqual(len(viewer.queue.pending()), 1)
        self.assertTrue(meta_backend.handle_metadata_undo(viewer, {}))
        self.assertEqual(viewer.queue.pending(), [])
        self.assertEqual(viewer.undone, 0)
        self.assertTrue(viewer.undo_by_hand())
        self.assertEqual(viewer.undone, 1)

    def test_redo_by_hand_does_nothing_while_commands_wait(self):
        from EMAPSSN_Viewer import MainViewer
        viewer = self.viewer
        viewer.redo_by_hand = types.MethodType(MainViewer.redo_by_hand, viewer)
        viewer.type(CLUSTER)
        viewer.type("redo")
        self.assertFalse(viewer.redo_by_hand())
        self.assertEqual(viewer.redone, 0)
        self.assertEqual(viewer.console_text.text,
                         "Nothing to redo: the undone steps were followed by 1 waiting command.")
        self.finish_compute()
        viewer.type("redo")
        self.assertEqual(viewer.redone, 1)

    def test_an_undo_from_a_script_waits_its_turn_and_undoes(self):
        viewer = self.viewer
        from commands import run as run_command
        viewer.type(CLUSTER)
        run_command.run_script_lines(viewer, ["undo"])
        self.assertEqual([job.line for job in viewer.queue.pending()], [CLUSTER, "undo"])
        self.finish_compute()
        self.assertEqual(viewer.undone, 1)
        self.assertEqual(viewer.saved, 1)


class FailureTests(QueueTestCase):
    def test_a_failure_cancels_the_commands_typed_after_it(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("subcluster cluster_9")
        viewer.type("color red")
        viewer.type("hide")
        self.finish_compute()
        statuses = [(job.line, job.status) for job in viewer.queue.finished]
        self.assertEqual(statuses, [(CLUSTER, 'succeeded'), ("subcluster cluster_9", 'failed'),
                                    ("color red", 'cancelled'), ("hide", 'cancelled')])
        self.assertIn("Cancelled 2 commands given after #2, which failed: #3 color red, #4 hide.",
                      self.output.getvalue())
        self.assertEqual(viewer.saved, 1)

    def test_a_failing_script_line_cancels_the_rest_of_the_script(self):
        from commands import run as run_command
        viewer = self.viewer
        viewer.type(CLUSTER)
        run_command.run_script_lines(viewer, ["select #cluster_7#", "color red"])
        self.assertIn("Batch given: 2 commands. Waiting their turn: 2", self.output.getvalue())
        self.finish_compute()
        self.assertEqual([job.status for job in viewer.queue.finished], ['succeeded', 'failed', 'cancelled'])

    def test_a_compute_that_raises_fails_its_command(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        with mock.patch("utilities.Network_Kernels.jaccard_partition", side_effect=RuntimeError("kernel broke")):
            self.finish_compute()
        self.assertEqual([job.status for job in viewer.queue.finished], ['failed', 'cancelled'])
        self.assertIn("kernel broke", self.output.getvalue())
        self.assertEqual(viewer.saved, 0)


class FileJobTests(QueueTestCase):
    """jobs lists and cancels the background scheduler's file jobs too, numbered with the commands."""

    def setUp(self):
        super().setUp()
        import itertools
        from Background_Job_Scheduler import BackgroundJobScheduler
        numbers = itertools.count(1)
        self.viewer.command_queue.shutdown()
        self.viewer.command_queue = CommandQueue(self.viewer, executor=ManualExecutor(), job_numbers=numbers)
        self.addCleanup(self.viewer.command_queue.shutdown)
        self.scheduler = BackgroundJobScheduler(self.viewer, job_numbers=numbers)
        self.viewer.background_job_scheduler = self.scheduler
        self.addCleanup(self.scheduler.shutdown)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.release = threading.Event()
        self.started = threading.Event()
        self.ran = []
        self.cancelled = []
        self.scheduler.job_cancelled.connect(lambda job, started: self.cancelled.append((job.job_id, started)))

    def path(self, name):
        return Path(self.folder.name) / name

    def worker(self, payload):
        """A file job that holds its thread: a partial file exists while it runs, and the job stops at its check point."""
        from utilities.Job_Cancellation import raise_if_cancelled
        if payload != "first":
            self.ran.append(payload)
            return {"message": payload}
        partial = self.path(".first.partial.svg")
        partial.write_bytes(b"half")
        try:
            self.started.set()
            while not self.release.wait(0.005):
                pass
            raise_if_cancelled()
            self.ran.append(payload)
        finally:
            partial.unlink()
        return {"message": payload}

    def enqueue(self, payload):
        return self.scheduler.enqueue("logo", f"logo -> {payload}.svg", payload, self.worker,
                                      str(self.path(f"{payload}.svg")))

    def wait_for(self, condition, timeout=10):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.app.processEvents()
        return condition()

    def test_a_file_job_not_started_is_dropped_and_one_being_written_is_stopped(self):
        self.viewer.type(CLUSTER)  # job 1
        writing = self.enqueue("first")  # job 2
        self.assertTrue(self.started.wait(10))
        waiting = self.enqueue("second")  # job 3
        after = self.enqueue("third")  # job 4
        self.assertEqual((writing, waiting, after), (2, 3, 4))
        self.viewer.type("jobs")
        listing = self.output.getvalue()
        self.assertIn("#2     running", listing)
        self.assertIn("#3     waiting", listing)
        self.viewer.type("jobs cancel 3")
        self.assertIn("Cancelled 1 job: #3 logo -> second.svg.", self.output.getvalue())
        self.assertFalse(self.scheduler.is_output_path_reserved(str(self.path("second.svg"))))
        # The one being written is told to stop, and is not called cancelled before it has: it stops at its
        # next check point, not before.
        self.output.truncate(0)
        self.output.seek(0)
        self.viewer.type("jobs cancel 2")
        self.assertIn("Told 1 running job to stop: #2 logo -> first.svg.", self.output.getvalue())
        self.assertNotIn("Cancelled", self.output.getvalue())
        self.assertNotIn("cannot be cancelled", self.output.getvalue())
        self.output.truncate(0)
        self.output.seek(0)
        self.viewer.type("jobs cancel 2")
        self.assertIn("Already stopping: #2 logo -> first.svg.", self.output.getvalue())
        self.assertNotIn("Told", self.output.getvalue())
        self.assertTrue(self.path(".first.partial.svg").exists())
        self.output.truncate(0)
        self.output.seek(0)
        self.viewer.type("jobs")
        self.assertIn("#2     stopping", self.output.getvalue())
        self.release.set()
        self.assertTrue(self.wait_for(lambda: (2, True) in self.cancelled and "third" in self.ran))
        self.assertEqual(sorted(self.cancelled), [(2, True), (3, False)])
        self.assertEqual(self.ran, ["third"], "the line goes on with the job after it")
        self.assertFalse(self.path(".first.partial.svg").exists(), "the unfinished file is deleted")
        self.assertFalse(self.path("first.svg").exists())
        self.assertTrue(self.wait_for(lambda: self.scheduler.listing() == []))
        self.assertFalse(self.scheduler.is_output_path_reserved(str(self.path("first.svg"))))

    def test_cancel_all_takes_in_the_file_being_written_and_the_ones_waiting(self):
        self.viewer.type(CLUSTER)  # job 1
        self.enqueue("first")  # job 2
        self.assertTrue(self.started.wait(10))
        self.enqueue("second")  # job 3
        self.viewer.type("jobs cancel all")
        text = self.output.getvalue()
        # The waiting ones are cancelled; the one being written is told to stop, which is not the same.
        self.assertIn("Cancelled 2 jobs: #1 cluster jaccard 0.5 3, #3 logo -> second.svg.", text)
        self.assertIn("Told 1 running job to stop: #2 logo -> first.svg.", text)
        self.release.set()
        self.assertTrue(self.wait_for(lambda: len(self.cancelled) == 2))
        self.assertEqual(sorted(self.cancelled), [(2, True), (3, False)])
        self.assertEqual(self.ran, [])
        self.assertFalse(self.path(".first.partial.svg").exists())

    def test_the_jobs_waiting_behind_a_running_one_are_taken_out_before_it_is_told_to_stop(self):
        # The running job ends the moment it is told to, while its stop function (ending a process, say) is
        # still running; a job behind it that was still in the line would start in that moment.
        from utilities.Job_Cancellation import raise_if_cancelled
        started = threading.Event()

        def running(payload, job):
            job.on_cancel(lambda: time.sleep(0.3))
            started.set()
            while True:
                time.sleep(0.002)
                raise_if_cancelled()

        def behind(payload):
            self.ran.append(payload)
            return {}

        self.scheduler.enqueue("test", "running", "running", running, str(self.path("running.svg")))
        self.assertTrue(started.wait(10))
        self.scheduler.enqueue("test", "behind", "behind", behind, str(self.path("behind.svg")))
        self.scheduler.enqueue("test", "behind too", "behind too", behind, str(self.path("behind_too.svg")))

        self.viewer.type("jobs cancel all")

        self.assertTrue(self.wait_for(lambda: self.scheduler.listing() == []))
        self.assertEqual(self.ran, [], "neither job behind the running one started")

    def test_a_number_that_names_no_job_cancels_nothing_not_even_the_file_being_written(self):
        self.enqueue("first")  # job 1
        self.assertTrue(self.started.wait(10))
        self.viewer.type("jobs cancel 1 99")
        self.assertIn("Error: No job #99 waits or runs; nothing was cancelled.", self.output.getvalue())
        self.assertEqual([entry.stopping for entry in self.scheduler.snapshot()], [False])
        self.release.set()
        self.assertTrue(self.wait_for(lambda: self.ran == ["first"]))


class TurnTests(QueueTestCase):
    def test_a_result_waits_while_nodes_are_dragged(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.is_multi_dragging = True
        viewer.executor.finish()
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(viewer.queue.pending()[0].status, 'applying')
        self.assertIsNone(viewer.cluster_labels)
        viewer.is_multi_dragging = False
        viewer.queue.resume()
        self.settle()
        self.assertIsNotNone(viewer.cluster_labels)

    def test_feedback_of_a_later_command_does_not_replace_a_command_being_typed(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.console_mode = True
        viewer.console_text.text = "colo"
        self.finish_compute()
        self.assertEqual(viewer.console_text.text, "colo")
        self.assertIn("Done! Found 2 clusters via JACCARD.", viewer.status_lines)

    def test_closing_cancels_every_job(self):
        viewer = self.viewer
        viewer.type(CLUSTER)
        viewer.type("color red")
        viewer.queue.shutdown()
        self.assertEqual(viewer.queue.pending(), [])
        self.assertEqual([job.status for job in viewer.queue.finished], ['cancelled', 'cancelled'])


class PortalMappingTests(QueueTestCase):
    def setUp(self):
        super().setUp()
        self.viewer.command_portal = self.portal = ViewerCommandPortal(self.viewer)
        self.addCleanup(self.portal.shutdown)

    def wait_request(self, request_id, statuses=('succeeded', 'failed', 'cancelled')):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            request = self.portal.get(request_id)
            if request['status'] in statuses:
                return request
            time.sleep(0.002)
        self.fail(f"request still {self.portal.get(request_id)['status']}")

    def test_a_request_waits_for_its_job_and_reports_its_states(self):
        request = self.portal.submit('s1', [CLUSTER, "select #cluster_1#"])
        for _ in range(20):
            self.app.processEvents()
        record = self.portal.get(request['request_id'])
        self.assertEqual(record['status'], 'running')
        self.assertEqual(record['commands'][0]['jobs'][0]['status'], 'running')
        self.assertEqual(record['commands'][1]['status'], 'queued')
        self.finish_compute()
        record = self.wait_request(request['request_id'])
        self.assertEqual(record['status'], 'succeeded')
        self.assertEqual([c['status'] for c in record['commands']], ['succeeded', 'succeeded'])
        self.assertEqual(record['commands'][0]['jobs'][0]['status'], 'succeeded')
        self.assertEqual(sorted(self.viewer.selected_indices), [0, 1, 2, 3])

    def start_cluster(self, submission):
        """Submit a clustering through the portal and let it start: its compute step waits for finish_compute."""
        request = self.portal.submit(submission, [CLUSTER])
        for _ in range(20):
            self.app.processEvents()
        return request['request_id'], self.viewer.command_queue.running()

    def entry(self, request_id):
        return self.portal.get(request_id)['commands'][0]['jobs'][0]

    def test_a_computing_commands_progress_reaches_its_job_entry_once_per_status_tick(self):
        queue = self.viewer.command_queue
        request_id, job = self.start_cluster('p1')
        self.assertIsNone(self.entry(request_id)['progress'])
        queue._show_running()
        self.assertIsNone(self.entry(request_id)['progress'])  # nothing reported yet
        job.progress_text = 'pass 1 of 3'  # what the worker thread's progress('pass 1 of 3') sets
        self.assertIsNone(self.entry(request_id)['progress'])  # the record is only written on the GUI thread
        queue._show_running()
        self.assertEqual(self.entry(request_id)['progress'], 'pass 1 of 3')
        job.progress_text = 'pass 2 of 3'
        self.assertEqual(self.entry(request_id)['progress'], 'pass 1 of 3')  # held until the next tick
        queue._show_running()
        self.assertEqual(self.entry(request_id)['progress'], 'pass 2 of 3')
        self.finish_compute()
        self.assertEqual(self.wait_request(request_id)['status'], 'succeeded')

    def test_progress_reported_after_the_last_tick_is_kept_when_the_job_finishes(self):
        request_id, job = self.start_cluster('p2')
        job.progress_text = 'pass 1 of 3'
        self.viewer.command_queue._show_running()
        job.progress_text = 'last pass'
        self.finish_compute()
        record = self.wait_request(request_id)
        self.assertEqual(record['commands'][0]['jobs'][0]['status'], 'succeeded')
        self.assertEqual(record['commands'][0]['jobs'][0]['progress'], 'last pass')

    def test_a_cancelled_job_keeps_what_it_reported(self):
        request_id, job = self.start_cluster('p3')
        job.progress_text = 'pass 2 of 3'
        self.viewer.command_queue.cancel(job)
        entry = self.entry(request_id)
        self.assertEqual((entry['status'], entry['progress']), ('cancelled', 'pass 2 of 3'))

    def test_a_job_that_reports_nothing_and_a_typed_job_have_no_progress_and_no_error(self):
        queue = self.viewer.command_queue
        request_id, job = self.start_cluster('p4')
        queue._show_running()
        self.assertIsNone(self.entry(request_id)['progress'])
        self.finish_compute()
        self.assertIsNone(self.wait_request(request_id)['commands'][0]['jobs'][0]['progress'])
        self.viewer.type(CLUSTER)  # typed: no portal entry to copy into
        typed = queue.running()
        self.assertIsNone(typed.context)
        typed.progress_text = 'pass 1 of 3'
        queue._show_running()
        self.finish_compute()

    def test_the_status_timer_calls_the_copy(self):
        """The 500 ms status timer is the throttle: it is connected to the method that copies the text."""
        from Command_Queue import STATUS_REFRESH_MS
        queue = self.viewer.command_queue
        self.assertEqual(queue._status_timer.interval(), STATUS_REFRESH_MS)
        request_id, job = self.start_cluster('p5')
        self.assertTrue(queue._status_timer.isActive())
        job.progress_text = 'from the timer'
        deadline = time.monotonic() + 10
        while self.entry(request_id)['progress'] is None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(self.entry(request_id)['progress'], 'from the timer')
        self.finish_compute()

    def test_a_portal_command_behind_a_typed_one_says_where_it_waits(self):
        self.viewer.type(CLUSTER)
        request = self.portal.submit('s2', ["color red"])
        for _ in range(20):
            self.app.processEvents()
        job = self.portal.get(request['request_id'])['commands'][0]['jobs'][0]
        self.assertEqual((job['status'], job['position'], job['waiting_for']), ('queued', 2, '1'))
        self.finish_compute()
        self.assertEqual(self.wait_request(request['request_id'])['status'], 'succeeded')

    def test_a_failure_skips_the_rest_of_its_request_and_leaves_typed_commands_alone(self):
        self.viewer.type(CLUSTER)
        request = self.portal.submit('s3', ["subcluster cluster_9", "color red"])
        for _ in range(20):
            self.app.processEvents()
        self.viewer.type("select #cluster_2#")
        self.finish_compute()
        record = self.wait_request(request['request_id'])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual([c['status'] for c in record['commands']], ['failed', 'skipped'])
        self.assertEqual(sorted(self.viewer.selected_indices), [4, 5, 6, 7])

    def test_a_cancelled_job_cancels_its_command_and_a_jobs_request_runs_ahead(self):
        request = self.portal.submit('s4', [CLUSTER, "color red"])
        for _ in range(20):
            self.app.processEvents()
        cancel = self.portal.submit('s5', ["jobs cancel all"])
        self.assertEqual(self.wait_request(cancel['request_id'])['status'], 'succeeded')
        record = self.wait_request(request['request_id'])
        self.assertEqual(record['status'], 'cancelled')
        self.assertEqual(record['commands'][0]['jobs'][0]['status'], 'cancelled')
        self.assertEqual(record['commands'][1]['status'], 'skipped')


def synthetic_network(n_nodes=10000, edges_per_cluster_node=90, seed=42):
    """A clustered network: dense random clusters of 50-400 nodes, 5% random edges between them."""
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < n_nodes:
        sizes.append(int(min(n_nodes - sum(sizes), rng.integers(50, 400))))
    starts = np.cumsum([0] + sizes[:-1])
    parts = []
    for start, size in zip(starts, sizes):
        count = size * edges_per_cluster_node // 2
        u = rng.integers(0, size, count) + start
        v = rng.integers(0, size, count) + start
        parts.append(np.column_stack((u, v)))
    bridges = rng.integers(0, n_nodes, (n_nodes * edges_per_cluster_node // 40, 2))
    edges = np.concatenate(parts + [bridges])
    edges = edges[edges[:, 0] != edges[:, 1]]
    key = np.unique(np.minimum(edges[:, 0], edges[:, 1]).astype(np.int64) * n_nodes
                    + np.maximum(edges[:, 0], edges[:, 1]))
    edges = np.column_stack((key // n_nodes, key % n_nodes)).astype(np.int32)
    scores = rng.uniform(30, 90, len(edges)).astype(np.float32)
    return edges, scores


class GuiResponsivenessTests(QueueTestCase):
    """A real Leiden clustering on the queue's worker thread leaves the GUI thread's timer ticking."""

    MAX_GAP_MS = 100

    def setUp(self):
        super().setUp()
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")
        self.viewer.command_queue.shutdown()
        self.viewer = QueueViewer(executor=ThreadExecutor())
        self.addCleanup(self.viewer.close)
        self.addCleanup(self.viewer.command_queue.shutdown)
        edges, scores = synthetic_network()
        self.viewer.n_nodes = 10000
        self.viewer.full_headers = [f"n{i}" for i in range(10000)]
        self.viewer.edges, self.viewer.edge_scores = edges, scores
        self.viewer.visible_mask = np.ones(10000, dtype=bool)
        self.viewer.group_labels = [set() for _ in range(10000)]
        self.viewer.current_colors = np.zeros((10000, 4))

    def timer_gaps(self, until):
        """Run the GUI thread's event loop, with a 5 ms timer ticking, until until() is true.

        Returns the gaps between ticks, in ms, and the seconds it took.
        """
        ticks = []
        timer = QtCore.QTimer()
        timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
        loop = QtCore.QEventLoop()
        poll = QtCore.QTimer()
        poll.setInterval(10)
        start = time.perf_counter()
        poll.timeout.connect(lambda: loop.quit() if until() or time.perf_counter() - start > 60 else None)
        timer.start()
        poll.start()
        loop.exec()
        timer.stop()
        poll.stop()
        stamps = np.array([start] + ticks + [time.perf_counter()])
        return np.diff(stamps) * 1000.0, time.perf_counter() - start

    def test_the_gui_timer_keeps_ticking_while_leiden_computes(self):
        from utilities import Network_Kernels
        threads = []
        real = Network_Kernels.leiden_partition

        def leiden(*args, **kwargs):
            threads.append(threading.get_ident())
            return real(*args, **kwargs)

        with mock.patch.object(Network_Kernels, "leiden_partition", side_effect=leiden):
            self.viewer.type("cluster leiden 1.0 10")
            gaps, seconds = self.timer_gaps(lambda: not self.viewer.queue.pending())
        self.assertNotEqual(threads, [threading.get_ident()])  # it computed on the worker thread
        self.assertIsNotNone(self.viewer.cluster_labels)
        self.assertEqual(self.viewer.queue.finished[-1].status, 'succeeded')
        sys.__stdout__.write(f"\n[jitter] cluster leiden on 10,000 nodes / {len(self.viewer.edges):,} edges: "
                             f"{seconds * 1000:.0f} ms, largest GUI timer gap {gaps.max():.1f} ms, "
                             f"median {np.median(gaps):.1f} ms\n")
        self.assertLess(gaps.max(), self.MAX_GAP_MS)

    def test_the_same_clustering_on_the_gui_thread_would_stop_the_timer(self):
        """The control: the measurement sees a freeze when there is one."""
        from commands import cluster
        done = []

        def block():
            cluster.run(self.viewer, ["leiden", "1.0", "10"])
            done.append(True)

        started = time.perf_counter()
        QtCore.QTimer.singleShot(50, block)
        gaps, _seconds = self.timer_gaps(lambda: bool(done))
        blocked_ms = (time.perf_counter() - started) * 1000 - 50
        self.assertGreater(gaps.max(), min(self.MAX_GAP_MS, 0.5 * blocked_ms))


if __name__ == "__main__":
    unittest.main()
