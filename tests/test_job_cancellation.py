# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Stopping a background job part-way (utilities.Job_Cancellation).

The check point does nothing outside a job and raises inside one that was told
to stop, in the thread of that job only; the exception is not an Exception, so
a worker's own error handling cannot take it for a failure; and a file object
whose every write is a check point stops the encoders and writers that cannot
be given check points of their own (a PNG, an SVG figure, a workbook) within a
chunk of their output, leaving nothing to fail on when they are collected.
"""
import gc
import io
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from utilities.Job_Cancellation import (  # noqa: E402
    CancellableFile,
    JobCancelled,
    JobHandle,
    bound_to,
    cancel_requested,
    current_job,
    forget_partial_path,
    note_partial_path,
    raise_if_cancelled,
)


class StopAfter(io.BytesIO):
    """A file that asks for the job to stop once it has had ``writes`` writes."""

    def __init__(self, token, writes):
        super().__init__()
        self.token, self.remaining, self.count = token, writes, 0

    def write(self, data):
        self.count += 1
        result = super().write(data)
        self.remaining -= 1
        if self.remaining == 0:
            self.token.request()
        return result


class StopAfterOnDisk:
    """The same for a real file, which the owner closes: it has the file's seek and tell too."""

    def __init__(self, handle, token, writes):
        self.handle, self.token, self.remaining, self.count = handle, token, writes, 0

    def write(self, data):
        self.count += 1
        result = self.handle.write(data)
        self.remaining -= 1
        if self.remaining == 0:
            self.token.request()
        return result

    def __getattr__(self, name):
        return getattr(self.handle, name)


class CheckPointTests(unittest.TestCase):
    def test_outside_a_job_every_call_does_nothing(self):
        self.assertFalse(cancel_requested())
        raise_if_cancelled()

    def test_a_job_told_to_stop_raises_at_its_next_check_point_and_not_before(self):
        token = JobHandle()
        with bound_to(token):
            raise_if_cancelled()
            self.assertFalse(cancel_requested())
            token.request()
            self.assertTrue(cancel_requested())
            with self.assertRaises(JobCancelled):
                raise_if_cancelled()
        raise_if_cancelled()  # the token is unbound again

    def test_the_exception_is_not_an_exception_so_error_handlers_cannot_swallow_it(self):
        self.assertTrue(issubclass(JobCancelled, BaseException))
        self.assertFalse(issubclass(JobCancelled, Exception))
        token = JobHandle()
        token.request()
        swallowed = []
        with self.assertRaises(JobCancelled), bound_to(token):
            try:
                raise_if_cancelled()
            except Exception as error:  # a worker's "skip this subset" or "report the failure"
                swallowed.append(error)

        self.assertEqual(swallowed, [])

    def test_the_token_belongs_to_the_thread_that_bound_it(self):
        token = JobHandle()
        token.request()
        seen = []

        def other_thread():
            seen.append(cancel_requested())
            raise_if_cancelled()

        with bound_to(token):
            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join(10)
            self.assertTrue(cancel_requested())
        self.assertEqual(seen, [False])

    def test_bound_blocks_nest_and_restore_the_one_outside(self):
        outer, inner = JobHandle(), JobHandle()
        outer.request()
        with bound_to(outer):
            with bound_to(inner):
                self.assertFalse(cancel_requested())
            self.assertTrue(cancel_requested())
        self.assertFalse(cancel_requested())

    def test_the_token_is_unbound_when_the_job_ends_by_an_exception(self):
        token = JobHandle()
        token.request()
        with self.assertRaises(JobCancelled), bound_to(token):
            raise_if_cancelled()
        self.assertFalse(cancel_requested())


class CancellableFileTests(unittest.TestCase):
    def test_writes_pass_through_until_the_job_is_told_to_stop(self):
        token = JobHandle()
        target = io.BytesIO()
        wrapper = CancellableFile(target)
        with bound_to(token):
            self.assertEqual(wrapper.write(b"abc"), 3)
            wrapper.flush()
            token.request()
            with self.assertRaises(JobCancelled):
                wrapper.write(b"def")
        self.assertEqual(target.getvalue(), b"abc")

    def test_it_writes_normally_outside_a_job(self):
        target = io.BytesIO()
        CancellableFile(target).write(b"abc")
        self.assertEqual(target.getvalue(), b"abc")

    def test_after_the_stop_it_does_nothing_more_so_a_half_done_writer_ends_quietly(self):
        token = JobHandle()
        token.request()
        target = io.BytesIO(b"keep")
        target.seek(4)
        wrapper = CancellableFile(target)
        with bound_to(token), self.assertRaises(JobCancelled):
            wrapper.write(b"x")
        target.close()  # the owner closes the file; the writer may still be collected
        self.assertEqual(wrapper.write(b"more"), 4)
        # A position of its own, which moves as a file's would.
        self.assertEqual(wrapper.tell(), 8)
        self.assertEqual(wrapper.seek(2), 2)
        self.assertEqual(wrapper.write(b"xyz"), 3)
        self.assertEqual(wrapper.tell(), 5)
        wrapper.flush()
        wrapper.close()

    def test_it_has_no_file_descriptor_so_an_encoder_writes_through_it(self):
        with self.assertRaises(io.UnsupportedOperation):
            CancellableFile(io.BytesIO()).fileno()
        with tempfile.TemporaryFile() as real:
            with self.assertRaises(io.UnsupportedOperation):
                CancellableFile(real).fileno()

    def test_closing_it_leaves_the_file_to_its_owner(self):
        target = io.BytesIO()
        wrapper = CancellableFile(target)
        wrapper.close()
        self.assertFalse(target.closed)
        self.assertFalse(wrapper.closed)


class JobHandleTests(unittest.TestCase):
    """What a job's worker and the scheduler share: the stop request and its stop functions, partial files, progress."""

    def test_the_stop_functions_a_worker_registered_are_called_once_when_the_job_is_told_to_stop(self):
        handle, calls = JobHandle(), []
        handle.on_cancel(lambda: calls.append("process"))
        handle.on_cancel(lambda: calls.append("second"))
        self.assertEqual(calls, [])
        handle.request()
        handle.request()
        self.assertEqual(calls, ["process", "second"])
        self.assertTrue(handle.requested)

    def test_a_stop_function_registered_after_the_request_is_called_at_once(self):
        handle, calls = JobHandle(), []
        handle.request()
        handle.on_cancel(lambda: calls.append("late"))
        self.assertEqual(calls, ["late"])

    def test_a_stop_function_that_fails_does_not_keep_the_others_from_running(self):
        handle, calls = JobHandle(), []

        def broken():
            raise OSError("process already gone")

        handle.on_cancel(broken)
        handle.on_cancel(lambda: calls.append("second"))
        with mock.patch("builtins.print") as said:
            handle.request()
        self.assertIn("process already gone", str(said.call_args))
        self.assertEqual(calls, ["second"])
        self.assertTrue(handle.requested)

    def test_partial_files_are_kept_in_the_order_named_until_forgotten(self):
        handle = JobHandle()
        handle.note_partial(Path("a.partial"))
        handle.note_partial("b.partial")
        self.assertEqual(handle.partial_paths(), ["a.partial", "b.partial"])
        handle.forget_partial("a.partial")
        handle.forget_partial("never named")
        self.assertEqual(handle.partial_paths(), ["b.partial"])

    def test_progress_and_the_log_path_tell_the_scheduler_only_when_they_change(self):
        told = []
        handle = JobHandle(on_change=lambda: told.append(1))
        handle.progress("Folding 1 of 3")
        handle.progress("Folding 1 of 3")
        handle.set_log_path(Path("run.log"))
        handle.set_log_path("run.log")
        handle.progress(None)
        self.assertEqual(len(told), 3)
        self.assertEqual((handle.progress_text, handle.log_path), (None, "run.log"))

    def test_the_module_functions_reach_the_handle_bound_to_the_thread_and_do_nothing_outside(self):
        note_partial_path("outside.partial")
        forget_partial_path("outside.partial")
        self.assertIsNone(current_job())
        handle = JobHandle()
        with bound_to(handle):
            self.assertIs(current_job(), handle)
            note_partial_path("inside.partial")
            self.assertEqual(handle.partial_paths(), ["inside.partial"])
            forget_partial_path("inside.partial")
        self.assertEqual(handle.partial_paths(), [])
        self.assertIsNone(current_job())


class OpenAtFirstWriteTests(unittest.TestCase):
    """CancellableFile.at opens the file only when something is written, and never once the job was told to stop."""

    def test_nothing_exists_until_the_first_write_and_the_file_is_closed_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.png")
            with CancellableFile.at(path, "wb") as target:
                target.write(b"")
                target.flush()
                self.assertEqual((target.tell(), os.path.exists(path)), (0, False), "an empty write opens nothing")
                target.write(b"abc")
                self.assertTrue(os.path.exists(path))
                target.write(b"def")
            self.assertTrue(target.closed)
            self.assertEqual(Path(path).read_bytes(), b"abcdef")

    def test_a_zip_writers_seek_to_the_start_opens_nothing_either(self):
        # ZipFile seeks to the start of the file it is given before it writes anything.
        import zipfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "book.xlsx")
            with CancellableFile.at(path, "wb") as target:
                archive = zipfile.ZipFile(target, "w")
                self.assertFalse(os.path.exists(path))
                archive.writestr("a.txt", "x")
                archive.close()
            with zipfile.ZipFile(path) as written:
                self.assertEqual(written.read("a.txt"), b"x")

    def test_a_job_told_to_stop_never_opens_the_file(self):
        handle = JobHandle()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.png")
            with bound_to(handle):
                with self.assertRaises(JobCancelled), CancellableFile.at(path, "wb") as target:
                    handle.request()
                    target.write(b"abc")
            self.assertEqual(os.listdir(directory), [])

    def test_a_text_file_takes_text_and_a_binary_one_bytes_as_files_do(self):
        # matplotlib tells a text file from a binary one by whether a bytes write is refused.
        with tempfile.TemporaryDirectory() as directory:
            with CancellableFile.at(os.path.join(directory, "t.svg"), "w", "utf-8") as text:
                with self.assertRaises(TypeError):
                    text.write(b"")
                text.write("é")
            with CancellableFile.at(os.path.join(directory, "b.png"), "wb") as binary:
                with self.assertRaises(TypeError):
                    binary.write("text")
            self.assertEqual(Path(directory, "t.svg").read_bytes(), "é".encode("utf-8"))


class EncodersThroughItTests(unittest.TestCase):
    """The writers print, logo and label hand it to: the same bytes, and a stop within a chunk."""

    def test_a_png_through_it_is_the_png_imsave_writes_to_a_path(self):
        image = (np.random.default_rng(7).random((60, 80, 4)) * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            direct = os.path.join(directory, "direct.png")
            mpimg.imsave(direct, image)
            through = os.path.join(directory, "through.png")
            with open(through, "wb") as handle:
                mpimg.imsave(CancellableFile(handle), image, format="png")
            self.assertEqual(Path(direct).read_bytes(), Path(through).read_bytes())

    def test_a_png_encode_stops_within_a_few_chunks_of_the_output(self):
        # Noise does not compress, so the PNG goes out in many 64 kB writes.
        image = (np.random.default_rng(7).random((900, 1200, 4)) * 255).astype(np.uint8)
        token = JobHandle()
        sink = StopAfter(token, 4)
        with bound_to(token), self.assertRaises(JobCancelled):
            mpimg.imsave(CancellableFile(sink), image, format="png")
        complete = io.BytesIO()
        mpimg.imsave(complete, image, format="png")
        self.assertGreater(len(complete.getvalue()), 20 * 65536, "the picture takes many chunks")
        self.assertLessEqual(sink.count, 6, "the encoder stopped within a chunk or two of the request")
        self.assertLess(len(sink.getvalue()), len(complete.getvalue()) // 10)

    def figure(self):
        figure = Figure(figsize=(4, 3))
        axes = figure.subplots()
        axes.plot(*np.random.default_rng(3).random((2, 4000)))
        return figure

    def test_an_svg_figure_through_it_is_the_svg_savefig_writes_to_a_path(self):
        meta = {"Date": None}
        figure = self.figure()
        with tempfile.TemporaryDirectory() as directory, matplotlib.rc_context({"svg.hashsalt": "fixed"}):
            direct = os.path.join(directory, "direct.svg")
            figure.savefig(direct, format="svg", metadata=meta)
            through = os.path.join(directory, "through.svg")
            # Opened as matplotlib opens a path for an SVG: text, so line ends are the platform's.
            with open(through, "w", encoding="utf-8") as handle:
                figure.savefig(CancellableFile(handle), format="svg", metadata=meta)
            self.assertEqual(Path(direct).read_bytes(), Path(through).read_bytes())

    def test_a_png_figure_through_it_is_the_png_savefig_writes_to_a_path(self):
        figure = self.figure()
        with tempfile.TemporaryDirectory() as directory:
            direct = os.path.join(directory, "direct.png")
            figure.savefig(direct, format="png", transparent=True, dpi=120, bbox_inches="tight")
            through = os.path.join(directory, "through.png")
            with open(through, "wb") as handle:
                figure.savefig(CancellableFile(handle), format="png", transparent=True, dpi=120, bbox_inches="tight")
            self.assertEqual(Path(direct).read_bytes(), Path(through).read_bytes())

    def test_an_svg_figure_stops_while_it_is_drawn(self):
        token = JobHandle()
        sink = StopAfter(token, 20)
        with bound_to(token), self.assertRaises(JobCancelled):
            self.figure().savefig(CancellableFile(sink), format="svg")
        whole = io.BytesIO()
        self.figure().savefig(whole, format="svg")
        self.assertLess(len(sink.getvalue()), len(whole.getvalue()) // 2)

    def workbook(self):
        book = openpyxl.Workbook()
        sheet = book.active
        rng = np.random.default_rng(5)
        for row in rng.random((400, 30)):
            sheet.append([float(value) for value in row])
        return book

    def test_a_workbook_through_it_loads_back_as_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "book.xlsx")
            with open(path, "wb") as handle:
                self.workbook().save(CancellableFile(handle))
            loaded = openpyxl.load_workbook(path)
            self.assertEqual(loaded.active.max_row, 400)
            self.assertEqual(loaded.active.max_column, 30)

    def test_a_workbook_stops_while_it_is_saved_and_its_writer_collects_quietly(self):
        # The workbook's ZipFile closes itself when it is collected, after the owner closed the file:
        # that must not raise (an exception there is printed as "Exception ignored" and lost).
        unraisable = []
        original = sys.unraisablehook
        sys.unraisablehook = lambda info: unraisable.append(info)
        self.addCleanup(setattr, sys, "unraisablehook", original)
        token = JobHandle()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "book.xlsx")
            with bound_to(token), self.assertRaises(JobCancelled):
                with open(path, "wb") as handle:
                    sink = StopAfterOnDisk(handle, token, 3)
                    self.workbook().save(CancellableFile(sink))
            self.assertTrue(handle.closed)
            gc.collect()
            self.assertEqual(unraisable, [])
            self.assertGreaterEqual(sink.count, 3)


class EveryQueuingCommandCanBeStoppedTests(unittest.TestCase):
    """A command that queues jobs gives each a way to be stopped, or "stop" would only ever say "stopping".

    A job whose worker has no check point finishes whatever it was told, and is reported as completed. This
    holds every command module that queues jobs to name at least one check point, found without naming a command.
    """

    COMMANDS = Path(__file__).resolve().parents[1] / "src" / "commands"
    CHECK_POINTS = ("raise_if_cancelled", "cancel_requested", "CancellableFile", "on_cancel")

    def test_each_command_module_that_queues_a_job_has_a_check_point(self):
        queuing = [path for path in sorted(self.COMMANDS.glob("*.py"))
                   if "scheduler.enqueue(" in path.read_text(encoding="utf-8")]
        self.assertGreaterEqual(len(queuing), 4, "the modules that queue jobs were found")
        for path in queuing:
            with self.subTest(module=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertTrue(any(point in source for point in self.CHECK_POINTS),
                                f"{path.name} queues jobs but names no way to stop one")


if __name__ == "__main__":
    unittest.main()
