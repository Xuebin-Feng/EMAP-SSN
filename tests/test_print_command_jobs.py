"""Print command background job, transient marks, lazy import, and cooperative
command-queue tests (src/commands/print.py): real background job queue writes and
cancellation, hiding hover/click marks during PNG capture, lazy matplotlib.image
import on first use, and cooperative tile slicing in the command queue."""

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.command_fixtures import end_scheduler
from tests.print_command_fixtures import (
    FixedClock,
    PROJECT_ROOT,
    SRC_DIR,
    StatusStub,
    TILE_SHAPE,
    drawn,
    export_svg,
    make_viewer,
    marked_viewer,
    node_slot,
    print_command,
    setUpModule,
    square_network,
    tearDownModule,
    wait_for,
    white_renders,
    zoom_viewer,
)


class PrintBackgroundJobTests(unittest.TestCase):
    """print in the viewer's real background job queue: the command returns
    once the picture is taken, and the queue's thread finishes and writes
    the file from what print copied, then reports it."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtCore

        # Reuse an application another test module created and leave it alive;
        # only an application created here is shut down in tearDownClass.
        cls._owns_application = QtCore.QCoreApplication.instance() is None
        cls.application = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    @classmethod
    def tearDownClass(cls):
        from PySide6 import QtCore

        application = cls.application
        cls.application = None
        if cls._owns_application:
            application.quit()
            application.deleteLater()
            application.processEvents()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        del application
        import gc
        gc.collect()

    def setUp(self):
        from Background_Job_Scheduler import BackgroundJobScheduler

        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.save_dir = os.path.join(root.name, "Saved_Images")
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        self.engine = engine
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(print_command, "PRINT_DIRECTORY", self.save_dir))
        self.open_folder = stack.enter_context(mock.patch("Background_Job_Scheduler.open_in_file_manager"))
        stack.enter_context(mock.patch.multiple(
            print_command.cfg, create=True, EDGE_COLOR="#000000",
            NODE_BOUNDARY_COLOR="#000000", EDGE_ALPHA=0.2, EDGE_WIDTH=0.5, UMAP_MODE=False,
        ))
        stack.enter_context(redirect_stdout(io.StringIO()))
        self.status = StatusStub()
        self.scheduler = BackgroundJobScheduler(self.status)
        # First of the cleanups: no job runs once the patches and folder go.
        self.addCleanup(self.stop_scheduler)

    def stop_scheduler(self, release=None):
        """Discard the queued jobs, let a blocked one end, wait for the queue to empty, and delete the scheduler."""
        self.scheduler.shutdown()
        if release is not None:
            release.set()
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))
        end_scheduler(self.scheduler)

    def viewer(self, svg=False):
        if svg:
            viewer = square_network()
            for name, value in vars(make_viewer()).items():
                if name not in ("canvas", "view"):
                    setattr(viewer, name, value)
            viewer.canvas.update = mock.Mock()
        else:
            viewer = zoom_viewer()
        viewer.background_job_scheduler = self.scheduler
        return viewer

    def block_the_queue(self):
        """Put a job in front that runs until the returned event is set."""
        started, release = threading.Event(), threading.Event()

        def blocker(_payload):
            started.set()
            release.wait(10)
            return {"message": "blocker done"}

        self.scheduler.enqueue(
            command_name="blocker", description="blocker", payload=None, worker=blocker,
            output_path=os.path.join(self.save_dir, "blocker.txt"),
        )
        self.assertTrue(started.wait(10))
        self.addCleanup(self.stop_scheduler, release)
        return release

    def finished(self, count=1):
        """Wait until count jobs have reported, and return the status lines."""
        done = lambda: sum(
            "completed" in line or "failed" in line for line in self.status.statuses
        ) >= count
        self.assertTrue(wait_for(done), self.status.statuses)
        return self.status.statuses

    def test_print_returns_while_the_queue_thread_encodes_the_png(self):
        viewer = self.viewer()
        threads, release = [], threading.Event()

        def encode(target, image, **kwargs):
            threads.append(threading.current_thread().name)
            self.assertTrue(release.wait(10))
            original_imsave(target, image, **kwargs)

        original_imsave = print_command.mpimg.imsave
        path = os.path.join(self.save_dir, "queued.png")
        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), \
                mock.patch.object(print_command.mpimg, "imsave", side_effect=encode):
            print_command.run(viewer, ["queued", "full"])
            # The command is done; the job waits in the encoder.
            self.engine.command_failed.assert_not_called()
            self.assertEqual(
                str(self.engine.command_succeeded.call_args.args[1]),
                f"Queued print for {path}; waiting for the background job.",
            )
            self.assertTrue(wait_for(lambda: threads))
            self.assertFalse(os.path.exists(path))
            self.assertTrue(self.scheduler.is_output_path_reserved(path))
            release.set()
            statuses = self.finished()

        self.assertEqual(threads, ["SSN-Background-Jobs"])
        self.assertEqual(os.listdir(self.save_dir), ["queued.png"])
        self.assertRegex(statuses[-1], r"^Background job #1 completed in [\d.]+s: Saved PNG: queued\.png$")
        self.open_folder.assert_called_once_with(self.save_dir)
        self.assertFalse(self.scheduler.is_output_path_reserved(path))

    def test_the_file_shows_the_network_as_it_was_when_print_ran(self):
        expected = export_svg(square_network())
        release = self.block_the_queue()
        viewer = self.viewer(svg=True)

        print_command.run(viewer, ["net", "svg"])
        # While the job waits behind another, the network changes in place
        # and in whole: nodes move, recolour, hide; edges and threshold change.
        viewer.pos[:] += 1000.0
        viewer.current_colors[:] = 0.0
        viewer.current_sizes[:] = 3.0
        viewer.visible_mask[1] = False
        viewer.edges = np.array([[0, 2]], dtype=np.int32)
        viewer.current_slider_threshold = 0.0
        viewer.canvas.bgcolor = SimpleNamespace(rgba=(0.0, 0.0, 0.0, 1.0))
        release.set()
        self.finished(count=2)

        with open(os.path.join(self.save_dir, "net.svg"), encoding="utf-8") as handle:
            self.assertEqual(handle.read(), expected)

    def test_a_name_a_queued_print_holds_is_refused(self):
        self.block_the_queue()
        viewer = self.viewer()
        path = os.path.join(self.save_dir, "same.png")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)):
            print_command.run(viewer, ["same"])
            print_command.run(viewer, ["same"])

        self.engine.command_succeeded.assert_called_once()
        self.assertEqual(
            str(self.engine.command_failed.call_args.args[1]),
            f"Error: Output file is already reserved by a background job: {path}",
        )
        self.assertEqual(self.scheduler.queue_depth, 2)

    def test_timestamped_names_skip_the_ones_queued_prints_hold(self):
        release = self.block_the_queue()
        viewer = self.viewer()
        clock = SimpleNamespace(datetime=FixedClock)
        with mock.patch.object(print_command, "datetime", clock), \
                mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)):
            print_command.run(viewer, [])
            print_command.run(viewer, [])
            release.set()
            self.finished(count=3)

        self.assertEqual(
            sorted(os.listdir(self.save_dir)),
            ["test_sequences_20260102_030405.png", "test_sequences_20260102_030405_2.png"],
        )

    def test_a_timestamped_name_taken_while_queued_is_not_written_over(self):
        # Another program saves a file of that name while the print waits.
        release = self.block_the_queue()
        viewer = self.viewer()
        clock = SimpleNamespace(datetime=FixedClock)
        path = os.path.join(self.save_dir, "test_sequences_20260102_030405.png")
        with mock.patch.object(print_command, "datetime", clock), \
                mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)), \
                redirect_stderr(io.StringIO()):
            print_command.run(viewer, [])
            os.makedirs(self.save_dir)
            with open(path, "w") as handle:
                handle.write("someone else's")
            release.set()
            statuses = self.finished(count=2)

        self.assertIn("Output file already exists", statuses[-1])
        with open(path) as handle:
            self.assertEqual(handle.read(), "someone else's")
        self.assertEqual(os.listdir(self.save_dir), ["test_sequences_20260102_030405.png"])

    def test_a_failed_job_is_reported_and_leaves_no_file(self):
        viewer = self.viewer()
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((4, 4, 4), dtype=np.float32)), \
                mock.patch.object(print_command.mpimg, "imsave", side_effect=OSError("disk full")), \
                redirect_stderr(io.StringIO()):
            print_command.run(viewer, ["broken"])
            statuses = self.finished()

        self.engine.command_succeeded.assert_called_once()
        self.assertRegex(statuses[-1], r"^Background job #1 failed after [\d.]+s \(print\): disk full$")
        self.assertEqual(os.listdir(self.save_dir), [])
        self.open_folder.assert_not_called()

    # --- a job stopped while it runs ---------------------------------------------------------

    def stop_job(self):
        """Tell the running print job to stop, as the Session card's remove does."""
        (running,) = [entry for entry in self.scheduler.snapshot() if entry.running]
        self.assertTrue(self.scheduler.cancel(running.job.job_id))

    def stopped(self):
        """Wait until the job has ended as stopped, and the line is empty."""
        reported = lambda: any("stopped before it finished" in line for line in self.status.statuses)
        self.assertTrue(wait_for(reported), self.status.statuses)
        self.assertTrue(wait_for(lambda: self.scheduler.queue_depth == 0))

    def stop_when_called(self, name, **patch_options):
        """Patch print_command.<name> to ask for the stop on its first call, and then do what it did."""
        real = getattr(print_command, name)
        calls = []

        def stopping(*args, **kwargs):
            if not calls:
                self.stop_job()
            calls.append(name)
            return real(*args, **kwargs)

        return mock.patch.object(print_command, name, side_effect=stopping, **patch_options), calls

    def test_a_png_stopped_while_it_is_encoded_deletes_its_partial_file(self):
        for arguments in (["half"], ["half", "full"]):
            with self.subTest(arguments=arguments):
                viewer = self.viewer()
                seen, writes, test = [], [], self

                class StopsOnWrite(print_command.CancellableFile):
                    def write(inner, data):
                        writes.append(1)
                        # The partial file is opened by the first write, so it is there for the second.
                        if len(writes) == 2:
                            seen.extend(os.listdir(test.save_dir))
                            test.stop_job()
                        return super().write(data)

                writes.clear()
                seen.clear()
                self.status.statuses.clear()
                with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), \
                        mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)), \
                        mock.patch.object(print_command, "CancellableFile", StopsOnWrite):
                    print_command.run(viewer, arguments)
                    self.stopped()
                self.assertEqual(len(seen), 1)
                self.assertIn(".partial.png", seen[0], "the partial file existed when the stop came")
                self.assertEqual(os.listdir(self.save_dir), [])
                self.assertFalse(self.scheduler.is_output_path_reserved(os.path.join(self.save_dir, "half.png")))
                self.open_folder.assert_not_called()

    def test_a_full_png_stopped_while_it_is_finished_writes_nothing(self):
        viewer = self.viewer()
        patch, calls = self.stop_when_called("_content_mask")
        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)), patch:
            print_command.run(viewer, ["finishing", "full"])
            self.stopped()
        self.assertTrue(calls, "the picture was being finished")
        self.assertEqual(os.listdir(self.save_dir) if os.path.exists(self.save_dir) else [], [])

    def test_an_svg_stopped_while_it_is_written_deletes_its_partial_file(self):
        viewer = self.viewer(svg=True)
        seen = []

        def stops_at_the_nodes(*args, **kwargs):
            # The header and the edges are in the partial file by now.
            seen.extend(name for name in os.listdir(self.save_dir))
            self.stop_job()
            yield from real_chunks(*args, **kwargs)

        real_chunks = print_command._svg_node_chunks
        with mock.patch.object(print_command, "_svg_node_chunks", side_effect=stops_at_the_nodes):
            print_command.run(viewer, ["drawing", "svg"])
            self.stopped()
        self.assertEqual(len(seen), 1)
        self.assertIn(".partial.svg", seen[0])
        self.assertEqual(os.listdir(self.save_dir), [])

    def close_the_viewer_during_the_save(self, arguments, *, job_obeys, wait=2.5, at_write=1):
        """Run print and close the scheduler, as closing the Viewer does, at a write of its PNG.

        With ``job_obeys`` the job waits at that write until it is told to stop; without, it
        holds there until the test lets it go, as a job stuck in one long encode does. The
        file is opened by the first write, so it exists from the second on. Returns (how long
        closing took, what the folder held when closing returned, which is before the job is
        let go).
        """
        from Background_Job_Scheduler import BackgroundJobScheduler
        from utilities.Job_Cancellation import cancel_requested

        self.scheduler = BackgroundJobScheduler(self.status)
        self.addCleanup(self.stop_scheduler)
        viewer = self.viewer()
        entered, release = threading.Event(), threading.Event()

        class HoldsAtFirstWrite(print_command.CancellableFile):
            writes = 0

            def write(inner, data):
                if data:
                    HoldsAtFirstWrite.writes += 1
                    if HoldsAtFirstWrite.writes == at_write:
                        entered.set()
                        if job_obeys:
                            while not cancel_requested():
                                time.sleep(0.005)
                        else:
                            release.wait(30)
                return super().write(data)

        with mock.patch.object(print_command, "_render_tile", side_effect=lambda v, t: white_renders(v, t, TILE_SHAPE)),                 mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)),                 mock.patch.object(print_command, "CancellableFile", HoldsAtFirstWrite):
            print_command.run(viewer, arguments)
            self.assertTrue(entered.wait(30))
            started = time.monotonic()
            self.scheduler.shutdown(wait=wait)
            took = time.monotonic() - started
            left_when_closed = sorted(os.listdir(self.save_dir)) if os.path.isdir(self.save_dir) else []
            release.set()
            deadline = time.monotonic() + 30
            while self.scheduler._thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.005)
        return took, left_when_closed

    def test_closing_the_viewer_during_the_save_leaves_neither_a_partial_file_nor_the_picture(self):
        for arguments in (["closing"], ["closing", "full"]):
            for job_obeys in (True, False):
                with self.subTest(arguments=arguments, job_obeys=job_obeys):
                    if os.path.isdir(self.save_dir):
                        shutil.rmtree(self.save_dir)
                    took, left_when_closed = self.close_the_viewer_during_the_save(arguments, job_obeys=job_obeys, wait=0.4)
                    self.assertLess(took, 3.0, "closing returns within the bound")
                    self.assertEqual(left_when_closed, [])
                    self.assertEqual(os.listdir(self.save_dir), [])

    def test_a_picture_the_job_is_stuck_in_the_middle_of_is_deleted_by_the_scheduler_where_the_system_allows(self):
        # Held after the first write, the PNG's partial file is open in the job's thread. Closing
        # deletes it where a file can be deleted while open; Windows cannot, and there it goes
        # when the job gets on (the stop is then waiting for it), which is all the process needs.
        took, left_when_closed = self.close_the_viewer_during_the_save(["held"], job_obeys=False, wait=0.4, at_write=2)
        self.assertLess(took, 3.0)
        if os.name != "nt":
            self.assertEqual(left_when_closed, [])
        self.assertEqual(os.listdir(self.save_dir), [])

    def test_closing_the_viewer_during_a_print_leaves_an_earlier_finished_picture_as_it_was(self):
        viewer = self.viewer()
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)):
            print_command.run(viewer, ["first"])
            self.finished()
        first = os.path.join(self.save_dir, "first.png")
        with open(first, "rb") as handle:
            before = handle.read()
        self.close_the_viewer_during_the_save(["second"], job_obeys=False, wait=0.3)
        self.assertEqual(os.listdir(self.save_dir), ["first.png"])
        with open(first, "rb") as handle:
            self.assertEqual(handle.read(), before)

    def test_a_stop_leaves_an_existing_picture_of_that_name_as_it_was(self):
        os.makedirs(self.save_dir)
        existing = os.path.join(self.save_dir, "keep.png")
        with open(existing, "wb") as handle:
            handle.write(b"previous picture")
        viewer = self.viewer()
        patch, calls = self.stop_when_called("_trim_png_margins")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)), patch:
            print_command.run(viewer, ["keep"])
            self.stopped()
        self.assertTrue(calls)
        self.assertEqual(os.listdir(self.save_dir), ["keep.png"])
        with open(existing, "rb") as handle:
            self.assertEqual(handle.read(), b"previous picture")

    def test_the_name_is_free_again_and_the_next_print_runs_after_a_stop(self):
        viewer = self.viewer()
        patch, _calls = self.stop_when_called("_trim_png_margins")
        with mock.patch.object(print_command, "_capture_tile", return_value=np.ones((40, 60, 4), dtype=np.float32)):
            with patch:
                print_command.run(viewer, ["again"])
                self.stopped()
            path = os.path.join(self.save_dir, "again.png")
            self.assertFalse(self.scheduler.is_output_path_reserved(path))
            self.status.statuses.clear()
            print_command.run(viewer, ["again"])
            self.finished()
        self.assertEqual(os.listdir(self.save_dir), ["again.png"])
        self.assertEqual(self.engine.command_failed.call_count, 0)

    def test_without_a_scheduler_nothing_is_drawn_or_written(self):
        viewer = self.viewer()
        del viewer.background_job_scheduler
        with mock.patch.object(print_command, "_render_tile") as render:
            for arguments in (["picture"], ["picture", "full"], ["picture", "svg"]):
                with self.subTest(arguments=arguments):
                    self.engine.command_failed.reset_mock()
                    print_command.run(viewer, arguments)
                    self.assertEqual(
                        str(self.engine.command_failed.call_args.args[1]),
                        "Error: The background job scheduler is unavailable.",
                    )
        render.assert_not_called()
        self.engine.command_succeeded.assert_not_called()
        self.assertFalse(os.path.exists(self.save_dir))


class PrintTransientMarksTests(unittest.TestCase):
    """A PNG leaves out the hover colour and the left-click halo, and keeps
    the selection borders; the live view gets both marks back afterwards."""

    def setUp(self):
        import matplotlib.colors as mcolors
        import EMAPSSN_Config

        self.hover_rgba = np.array(mcolors.to_rgba(EMAPSSN_Config.HOVER_COLOR), dtype=np.float32)
        patcher = mock.patch.multiple(
            EMAPSSN_Config, CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_print(self, viewer, arguments, capture_error=None, interrupt=False, write_png=None):
        engine = print_command.Command_Engine
        for reporter in (engine.command_artifact, engine.command_succeeded, engine.command_failed):
            reporter.reset_mock()
        during = []

        def capture(captured_viewer, is_transparent):
            hidden = getattr(captured_viewer, "transient_marks_hidden", False)
            during.append((drawn(captured_viewer), hidden))
            if capture_error is not None:
                raise capture_error
            return white_renders(captured_viewer, is_transparent, TILE_SHAPE)

        save_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, save_dir, True)
        patches = [
            mock.patch.object(print_command, "PRINT_DIRECTORY", save_dir),
            mock.patch.object(print_command, "_render_tile", side_effect=capture),
            mock.patch.object(print_command, "_write_png", side_effect=write_png),
        ]
        if interrupt:
            patches.append(mock.patch.object(
                print_command._CaptureEvents, "pause_if_due",
                side_effect=print_command._CaptureInterrupted(print_command.Message("window resized")),
            ))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            print_command.run(viewer, arguments)
        return engine, during

    def assert_drawn_without_transient_marks(self, viewer, before, during):
        self.assertTrue(during)
        for data, hidden in during:
            self.assertTrue(hidden)
            # No halo: one marker per visible node, in the same order as before.
            self.assertFalse(data["rings"].any())
            np.testing.assert_array_equal(
                data["node_order"], before["node_order"][~before["rings"]]
            )
            # The hovered node has its own colour, not the hover colour.
            np.testing.assert_array_equal(
                data["face_color"][node_slot(data, 1)], viewer.current_colors[1]
            )
            # The clicked node is drawn as before, without its ring.
            np.testing.assert_array_equal(
                data["face_color"][node_slot(data, 2)],
                before["face_color"][node_slot(before, 2)],
            )
            # Selection borders stay: colour and width of the selected node.
            for name in ("edge_color", "edge_width"):
                np.testing.assert_array_equal(
                    data[name][node_slot(data, 0)], before[name][node_slot(before, 0)]
                )
            np.testing.assert_array_equal(data["edge_color"][node_slot(data, 0)], self.hover_rgba)
            self.assertEqual(data["edge_width"][node_slot(data, 0)], 2.0)

    def assert_restored(self, viewer, before):
        after = drawn(viewer)
        self.assertFalse(viewer.transient_marks_hidden)
        self.assertEqual(viewer.hovered_node_idx, 1)
        self.assertEqual(viewer.selected_node_idx, 2)
        self.assertEqual(viewer.selected_indices, [0])
        self.assertEqual(set(after), set(before))
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        # The mouse is still over node 1, so it has the hover colour again,
        # and node 2 its halo.
        np.testing.assert_array_equal(after["face_color"][node_slot(after, 1)], self.hover_rgba)
        self.assertEqual(after["node_order"][after["rings"]].tolist(), [2])

    def test_the_viewer_draws_the_marks_this_test_relies_on(self):
        before = drawn(marked_viewer())
        np.testing.assert_array_equal(before["face_color"][node_slot(before, 1)], self.hover_rgba)
        self.assertEqual(before["node_order"][before["rings"]].tolist(), [2])

    def test_a_png_leaves_out_hover_and_halo_and_gets_them_back(self):
        for arguments in (["marks"], ["marks", "transparent"], ["marks", "full"],
                          ["marks", "full", "zoom", "2000"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)

                engine, during = self.run_print(viewer, arguments)

                engine.command_failed.assert_not_called()
                engine.command_succeeded.assert_called_once()
                self.assert_drawn_without_transient_marks(viewer, before, during)
                self.assert_restored(viewer, before)

    def test_the_marks_are_back_while_the_png_is_encoded(self):
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)
                while_encoding = []

                def encode(_path, _image):
                    while_encoding.append(viewer.transient_marks_hidden)
                    self.assert_restored(viewer, before)

                engine, _during = self.run_print(viewer, arguments, write_png=encode)

                engine.command_succeeded.assert_called_once()
                self.assertEqual(viewer.background_job_scheduler.errors, [])
                self.assertEqual(while_encoding, [False])

    def test_the_marks_come_back_when_the_capture_fails(self):
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer()
                before = drawn(viewer)

                engine, during = self.run_print(
                    viewer, arguments, capture_error=RuntimeError("render failed")
                )

                engine.command_failed.assert_called_once()
                engine.command_succeeded.assert_not_called()
                self.assert_drawn_without_transient_marks(viewer, before, during)
                self.assert_restored(viewer, before)

    def test_the_marks_come_back_when_a_full_capture_is_cancelled(self):
        viewer = marked_viewer()
        before = drawn(viewer)

        engine, during = self.run_print(viewer, ["marks", "full"], interrupt=True)

        engine.command_failed.assert_called_once()
        self.assertIn("window resized", str(engine.command_failed.call_args.args[1]))
        self.assert_drawn_without_transient_marks(viewer, before, during)
        self.assert_restored(viewer, before)

    def test_an_svg_and_a_refusal_leave_the_drawing_alone(self):
        viewer = marked_viewer()
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update, \
                mock.patch.object(print_command, "_svg_snapshot"), \
                mock.patch.object(print_command, "_write_svg"):
            self.run_print(viewer, ["marks", "svg"])
            self.run_print(viewer, ["marks", "zoom", "200"])
        update.assert_not_called()

    def test_a_network_without_marks_is_not_redrawn_but_keeps_new_ones_out_of_the_tiles(self):
        # The mouse moves and clicks while a full capture is drawn, and neither may reach a tile drawn after it.
        for arguments in (["marks"], ["marks", "full"]):
            with self.subTest(arguments=arguments):
                viewer = marked_viewer(hovered=None, clicked=None)
                with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
                    engine, during = self.run_print(viewer, arguments)
                engine.command_failed.assert_not_called()
                update.assert_not_called()
                self.assertTrue(all(hidden for _data, hidden in during))
                self.assertFalse(viewer.transient_marks_hidden)

    def test_a_hover_or_click_that_comes_during_the_capture_is_drawn_when_it_ends(self):
        viewer = marked_viewer(hovered=None, clicked=None)
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
            restore = print_command._hide_transient_marks(viewer)
            self.assertIsNotNone(restore)
            self.assertTrue(viewer.transient_marks_hidden)
            update.assert_not_called()
            viewer.hovered_node_idx = 1
            viewer.selected_node_idx = 2
            restore()
        update.assert_called_once()
        after = drawn(viewer)
        self.assertFalse(viewer.transient_marks_hidden)
        np.testing.assert_array_equal(after["face_color"][node_slot(after, 1)], self.hover_rgba)
        self.assertEqual(after["node_order"][after["rings"]].tolist(), [2])

    def test_a_hover_or_click_that_is_over_by_then_is_not_redrawn(self):
        viewer = marked_viewer()
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as update:
            restore = print_command._hide_transient_marks(viewer)
            update.reset_mock()  # the marks that were shown were redrawn away
            viewer.hovered_node_idx = None
            viewer.selected_node_idx = None
            restore()
        update.assert_not_called()
        self.assertFalse(viewer.transient_marks_hidden)

    def test_a_viewer_that_draws_no_marks_has_nothing_to_hide(self):
        viewer = zoom_viewer()
        self.assertIsNone(print_command._hide_transient_marks(viewer))
        self.assertFalse(getattr(viewer, "transient_marks_hidden", False))


class PrintFirstUseTests(unittest.TestCase):
    def test_the_module_does_not_load_the_image_writer_on_the_gui_thread(self):
        # The first print of a session loads the module on the GUI thread: matplotlib's image writer waits for the job.
        script = (
            "import sys\n"
            f"sys.path[:0] = [{PROJECT_ROOT!r}, {SRC_DIR!r}]\n"
            "import tests  # noqa: F401\n"
            "from commands import print as print_command\n"
            "assert 'matplotlib.image' not in sys.modules\n"
            "writer = print_command.mpimg\n"
            "assert writer is sys.modules['matplotlib.image']\n"
            "print('LOADED WHEN ASKED FOR')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=PROJECT_ROOT,
                                timeout=300, encoding="utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:] + result.stderr[-2500:])
        self.assertIn("LOADED WHEN ASKED FOR", result.stdout)


class PrintCooperativeTests(unittest.TestCase):
    """print as the command queue runs it: start() gives a generator that yields after each tile of a full capture."""

    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.save_dir = os.path.join(root.name, "Saved_Images")
        self.engine = print_command.Command_Engine
        for reporter in (self.engine.command_artifact, self.engine.command_succeeded, self.engine.command_failed):
            reporter.reset_mock()
        self.rects = []
        self.render_mock = mock.Mock(side_effect=self.render)
        patches = (
            mock.patch.object(print_command, "PRINT_DIRECTORY", self.save_dir),
            mock.patch.object(print_command, "_render_tile", self.render_mock),
            mock.patch.object(print_command, "_write_png"),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.save = print_command._write_png
        hidden = redirect_stdout(io.StringIO())
        hidden.__enter__()
        self.addCleanup(hidden.__exit__, None, None, None)

    def render(self, viewer, is_transparent):
        self.rects.append(viewer.view.camera.rect)
        return white_renders(viewer, is_transparent, TILE_SHAPE)

    def steps(self, viewer, arguments):
        """The generator the command queue steps for `print arguments`."""
        return print_command.start(viewer, print_command.plan(viewer, arguments))

    def test_the_plan_and_start_are_what_the_queue_reads_and_nothing_is_drawn_until_it_steps(self):
        import inspect
        viewer = zoom_viewer()

        plan = print_command.plan(viewer, ["fig1", "full"])
        steps = print_command.start(viewer, plan)

        self.assertTrue(plan.cooperative)
        self.assertFalse(getattr(plan, "heavy", False))
        self.assertTrue(inspect.isgenerator(steps))
        self.render_mock.assert_not_called()
        self.engine.command_succeeded.assert_not_called()

    def saved_images(self):
        """The pictures _write_png was given since the last call."""
        images = [call.args[1] for call in self.save.call_args_list]
        self.save.reset_mock()
        return images

    def test_stepping_it_to_the_end_saves_what_run_saves(self):
        for arguments in (["fig1", "full"], ["fig1", "full", "zoom", "2000"], ["fig1", "full", "transparent"],
                          ["fig1"], ["fig1", "transparent"]):
            with self.subTest(arguments=arguments):
                print_command.run(zoom_viewer(), arguments)
                expected, expected_rects = self.saved_images(), list(self.rects)
                self.rects.clear()

                stepped = zoom_viewer()
                for _ in self.steps(stepped, arguments):
                    pass

                images = self.saved_images()
                self.assertEqual(len(images), 1)
                self.assertEqual(len(images), len(expected))
                np.testing.assert_array_equal(images[0], expected[0])
                self.assertEqual(self.rects, expected_rects)
                self.assertEqual(len(stepped.background_job_scheduler.jobs), 1)
                self.rects.clear()

    def test_it_yields_the_progress_after_each_tile(self):
        # 100 x 75 units at 0.125 per pixel: a 2 x 2 grid of tiles.
        progress = list(self.steps(zoom_viewer(), ["fig1", "full"]))

        self.assertEqual([str(message) for message in progress], [f"Capturing tile {n} of 4" for n in range(1, 5)])
        self.assertTrue(all(isinstance(message, print_command.Message) for message in progress))

    def test_a_plain_view_and_an_svg_are_taken_in_one_step(self):
        with mock.patch.object(print_command, "_svg_snapshot"), mock.patch.object(print_command, "_write_svg"):
            for arguments in (["fig1"], ["fig1", "svg"], ["help"]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(list(self.steps(zoom_viewer(), arguments)), [])

    def test_a_resize_between_tiles_cancels_it_and_nothing_is_saved(self):
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)
        next(steps)

        viewer.canvas.size = (900, 600)
        self.assertEqual(list(steps), [])  # it reported the failure itself and went no further

        self.engine.command_failed.assert_called_once()
        self.assertIn("window was resized", str(self.engine.command_failed.call_args.args[1]))
        self.engine.command_succeeded.assert_not_called()
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.assertEqual(self.render_mock.call_count, 3)  # the measuring render and two tiles

    def capture_events(self):
        """Keep the _CaptureEvents a print makes, to tell it what the event loop did."""
        made = []
        real = print_command._CaptureEvents

        def make(*arguments, **keywords):
            made.append(real(*arguments, **keywords))
            return made[-1]

        patch = mock.patch.object(print_command, "_CaptureEvents", side_effect=make)
        patch.start()
        self.addCleanup(patch.stop)
        return made

    def test_a_change_of_the_network_between_tiles_cancels_it(self):
        made = self.capture_events()
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)

        made[0]._mark_changed()  # a visual of the network was updated while the event loop ran
        self.assertEqual(list(steps), [])

        self.engine.command_failed.assert_called_once()
        self.assertIn("network display changed", str(self.engine.command_failed.call_args.args[1]))
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])

    def test_a_camera_move_between_tiles_does_not(self):
        made = self.capture_events()
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)

        made[0]._mark_changed()  # every visual is updated by a camera move too
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        remaining = list(steps)

        self.assertEqual(len(remaining), 3)
        self.engine.command_failed.assert_not_called()
        self.engine.command_succeeded.assert_called_once()
        self.save.assert_called_once()

    def test_the_input_that_changes_nothing_cancels_nothing(self):
        made = self.capture_events()
        steps = self.steps(zoom_viewer(), ["fig1", "full"])
        for _ in steps:
            # The event loop ran between every two tiles, and nothing was updated.
            self.assertFalse(made[0]._changed)
        self.engine.command_failed.assert_not_called()
        self.save.assert_called_once()

    def test_events_that_changed_the_network_are_noted_from_the_moment_the_event_loop_is_let_run(self):
        viewer = zoom_viewer()
        events = print_command._CaptureEvents(viewer, [])

        events._mark_changed()  # something before the pause is no part of it
        events.release()
        events.reclaim()

        events._mark_changed()
        events.release()
        events._mark_changed()
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def composing_viewer(self):
        """A zoom_viewer() whose canvas widget has the signals the QOpenGLWidget sends as its window composes and swaps."""
        viewer = zoom_viewer()
        viewer.canvas.native = SimpleNamespace(
            aboutToCompose=mock.Mock(), frameSwapped=mock.Mock(), window=mock.Mock(return_value=mock.Mock()),
        )
        return viewer

    def compose_for(self, events, clock, seconds):
        """The window composes, its swap taking `seconds`, and the event loop's pass ends."""
        events.release()
        events._compose_began()
        clock[0] += seconds
        events._compose_ended()
        events.reclaim()

    def test_a_window_whose_swap_waits_for_the_desktop_is_not_painted_again_until_the_capture_ends(self):
        # Covered, under XWayland, the swap of every paint the event loop runs takes a second: 144 tiles took 71 s.
        viewer = self.composing_viewer()
        window = viewer.canvas.native.window.return_value
        clock = [100.0]
        with mock.patch.object(print_command, "time", SimpleNamespace(perf_counter=lambda: clock[0])):
            events = print_command._CaptureEvents(viewer, [])
            viewer.canvas.native.aboutToCompose.connect.assert_called_once()
            viewer.canvas.native.frameSwapped.connect.assert_called_once()

            self.compose_for(events, clock, 0.02)  # a window that shows its frames is left alone
            window.setUpdatesEnabled.assert_not_called()

            self.compose_for(events, clock, 1.04)
            window.setUpdatesEnabled.assert_called_once_with(False)

            clock[0] += print_command.CAPTURE_PRESENT_PROBE_S / 2
            events.release()
            events.reclaim()
            window.setUpdatesEnabled.assert_called_once_with(False)  # still held

            clock[0] += print_command.CAPTURE_PRESENT_PROBE_S
            events.release()
            events.reclaim()
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(True))  # let paint, to see

            self.compose_for(events, clock, 1.04)  # and it still does not show a frame
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(False))

            events.close()
            self.assertEqual(window.setUpdatesEnabled.call_args_list[-1], mock.call(True))
            events.close()  # closing twice does no more
            self.assertEqual(window.setUpdatesEnabled.call_count, 4)
            viewer.canvas.native.aboutToCompose.disconnect.assert_called_once()
            viewer.canvas.native.frameSwapped.disconnect.assert_called_once()

    def test_a_window_that_was_never_held_is_not_touched_when_the_capture_ends(self):
        viewer = self.composing_viewer()
        window = viewer.canvas.native.window.return_value
        clock = [100.0]
        with mock.patch.object(print_command, "time", SimpleNamespace(perf_counter=lambda: clock[0])):
            events = print_command._CaptureEvents(viewer, [])
            for _ in range(20):
                self.compose_for(events, clock, 0.03)
            events.close()
        window.setUpdatesEnabled.assert_not_called()

    def test_a_canvas_without_composing_signals_is_left_to_the_event_loop_alone(self):
        events = print_command._CaptureEvents(zoom_viewer(), [])
        events.release()
        events.reclaim()
        events.close()

    def scene_viewer(self):
        """A zoom_viewer() whose canvas holds a scene: the edge layer and the node markers the capture watches."""
        from vispy import scene
        from desktop.Viewer_Edges import ViewerEdges

        root = scene.Node()
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        layer = ViewerEdges(root, np.array([[0, 1], [1, 2]], dtype=np.int32), np.array([0.5, 0.9]), positions,
                            (0, 0, 0, 0.5), 1.0, use_position_texture=False)
        markers = scene.visuals.Markers(parent=root)
        viewer = zoom_viewer()
        viewer.canvas.scene = root
        return viewer, layer, markers

    def test_an_edge_layer_that_only_repaints_is_no_change_and_one_that_draws_other_edges_is(self):
        # After a pan with fewer edges drawn the layer draws every edge again, and updates its visual to say so:
        # the picture it draws is the same, which its generation, bumped by every real change, tells.
        viewer, layer, _markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        layer.visual.update()
        events.reclaim()  # a repaint

        events.release()
        layer.visible = not layer.visible  # what draws changed
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def test_that_holds_with_the_camera_moved_too(self):
        viewer, layer, _markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        layer.visual.update()
        events.reclaim()
        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(20.0, -12.5), width=100.0, height=75.0)
        layer.visible = not layer.visible
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

    def test_a_node_marker_update_is_still_a_change_unless_the_camera_moved(self):
        viewer, _layer, markers = self.scene_viewer()
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)

        events.release()
        markers.update()
        with self.assertRaises(print_command._CaptureInterrupted):
            events.reclaim()

        events.release()
        viewer.view.camera.rect = SimpleNamespace(pos=(10.0, -12.5), width=100.0, height=75.0)
        markers.update()
        events.reclaim()

    def test_it_closed_between_tiles_leaves_no_job_and_no_report(self):
        viewer = zoom_viewer()
        steps = self.steps(viewer, ["fig1", "full"])
        next(steps)
        next(steps)

        steps.close()

        viewer.canvas.update.assert_called()
        self.save.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.engine.command_failed.assert_not_called()
        self.engine.command_succeeded.assert_not_called()
        # The live view is as it was.
        self.assertEqual(viewer.view.camera.rect.width, 100.0)
        self.assertTrue(viewer.zoom_text.visible)


class PrintInTheCommandQueueTests(unittest.TestCase):
    """The real command queue steps a print's tiles in slices: the event loop, and with it the Viewer's input, runs between them."""

    @classmethod
    def setUpClass(cls):
        from PySide6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        import Command_Engine as real_engine
        from Command_Queue import CommandQueue
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.log = []
        self.engine = print_command.Command_Engine
        for reporter in (self.engine.command_artifact, self.engine.command_succeeded, self.engine.command_failed):
            reporter.reset_mock()
        # The queue records the outcome the real engine is told of.
        self.engine.command_succeeded.side_effect = lambda viewer, message=None: real_engine._record_outcome("succeeded")
        self.addCleanup(setattr, self.engine.command_succeeded, "side_effect", None)

        def render(viewer, is_transparent):
            self.log.append("tile")
            time.sleep(0.01)
            return white_renders(viewer, is_transparent, TILE_SHAPE)

        for patch in (mock.patch.object(print_command, "PRINT_DIRECTORY", os.path.join(root.name, "Saved_Images")),
                      mock.patch.object(print_command, "_render_tile", side_effect=render),
                      mock.patch.object(print_command, "_write_png"),
                      redirect_stdout(io.StringIO())):
            patch.__enter__()
            self.addCleanup(patch.__exit__, None, None, None)
        class QueuedViewer(SimpleNamespace):
            """The queue keeps a weak reference to its viewer, which a SimpleNamespace itself cannot be given."""

        self.viewer = QueuedViewer(**vars(zoom_viewer()))
        self.queue = CommandQueue(self.viewer)
        self.viewer.command_queue = self.queue
        self.addCleanup(self.queue.shutdown)

    def pump_until(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, "the command queue did not get there")
            self.app.processEvents()
            time.sleep(0.001)

    def test_the_event_loop_runs_between_the_tiles_and_the_picture_is_the_same(self):
        from PySide6 import QtCore
        timer = QtCore.QTimer()
        timer.setInterval(1)
        timer.timeout.connect(lambda: self.log.append("tick"))
        timer.start()
        self.addCleanup(timer.stop)
        arguments = ["fig1", "full", "zoom", "2000"]  # 3 x 3 tiles of 10 ms each, in slices of 30 ms

        job = self.queue.dispatch(print_command, "print", arguments, "print " + " ".join(arguments))
        self.assertIsNotNone(job)
        self.pump_until(lambda: not self.queue.pending())
        timer.stop()

        self.assertEqual(self.log.count("tile"), 10)  # the render that measures the canvas, and nine tiles
        first, last = self.log.index("tile"), len(self.log) - 1 - self.log[::-1].index("tile")
        self.assertGreaterEqual(self.log[first:last].count("tick"), 2)
        self.assertEqual(self.queue.finished[-1].status, "succeeded")
        self.assertEqual(len(self.viewer.background_job_scheduler.jobs), 1)
        saved = print_command._write_png.call_args.args[1]
        self.assertEqual(saved.shape, (960, 1760, 4))

    def test_a_command_given_meanwhile_waits_for_the_picture(self):
        import Command_Engine as real_engine
        order = []

        def later_start(viewer, plan):
            order.append(self.log.count("tile"))
            real_engine._record_outcome("succeeded")

        arguments = ["fig1", "full", "zoom", "2000"]
        self.queue.dispatch(print_command, "print", arguments, "print fig1 full zoom 2000")
        later = SimpleNamespace(plan=lambda viewer, args: SimpleNamespace(cooperative=False, heavy=False),
                                start=later_start)
        self.queue.dispatch(later, "later", [], "later")
        self.assertEqual([job.name for job in self.queue.pending()], ["print", "later"])
        self.pump_until(lambda: not self.queue.pending())
        # `later` took its turn when every tile had been drawn.
        self.assertEqual(order, [10])

    def test_undo_stops_a_print_between_tiles_and_nothing_is_saved(self):
        arguments = ["fig1", "full", "zoom", "2000"]
        job = self.queue.dispatch(print_command, "print", arguments, "print fig1 full zoom 2000")
        self.pump_until(lambda: self.log.count("tile") >= 3)
        self.assertTrue(job.cooperation is not None)

        self.queue.take_back(1)

        self.assertEqual(self.queue.pending(), [])
        self.assertEqual(job.status, "cancelled")
        drawn = self.log.count("tile")
        for _ in range(20):
            self.app.processEvents()
        self.assertEqual(self.log.count("tile"), drawn)
        print_command._write_png.assert_not_called()
        self.assertEqual(self.viewer.background_job_scheduler.jobs, [])
        self.assertEqual(self.viewer.view.camera.rect.width, 100.0)


if __name__ == "__main__":
    unittest.main()
