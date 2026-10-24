# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Output capture (utilities.Console_Capture): secrets are redacted in the ring and the log, carriage returns collapse, lines are split across chunk boundaries, the ring and the rotating log keep what they should, the original stream is passed on only when someone reads it, and every source of output reaches the ring in the contexts the windows start in (pipes, a file, no handles at all, a hidden console, NUL), with Qt's warnings, a native crash and the last lines before exit included. No capture runs in the MCP server."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
PROBE = PROJECT_ROOT / "tests" / "console_capture_probe.py"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Console_Capture  # noqa: E402
from utilities.Console_Capture import (  # noqa: E402
    Recorder,
    Splitter,
    collapse_carriage_returns,
    prune_logs,
    redact,
)

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
# Everything the probe writes: one marker for each way a window's process can print.
MARKERS_ANY_PLATFORM = (
    "python stdout", "python stderr", "unicode", "progress 100% done", "os.write stdout", "os.write stderr",
    "c puts", "child python stdout", "child python stderr", "python warning", "ZeroDivisionError",
)
MARKERS_WINDOWS = ("win32 stdout handle", "win32 stderr handle", "child cmd stdout", "child cmd stderr")
MARKERS_POSIX = ("child sh stdout", "child sh stderr")
ALL_MARKERS = MARKERS_ANY_PLATFORM + (MARKERS_WINDOWS if sys.platform == "win32" else MARKERS_POSIX)


def child_environment():
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.pop("QT_FORCE_STDERR_LOGGING", None)
    return environment


class Ring(Recorder):
    """A recorder with no log file, for the parts that do not need one."""

    def __init__(self, **options):
        super().__init__("test", None, **options)


class RedactTests(unittest.TestCase):
    def test_the_keys_of_the_services_the_program_uses_are_replaced(self):
        cases = {
            "sk-proj-abcdEFGH1234567890abcdEFGH": "key sk-proj-abcdEFGH1234567890abcdEFGH used",
            "sk-ant-api03-AbCdEfGhIjKlMnOpQrSt": "ANTHROPIC sk-ant-api03-AbCdEfGhIjKlMnOpQrSt",
            "hf_AbCdEfGhIjKlMnOpQrStUvWxYz": "token hf_AbCdEfGhIjKlMnOpQrStUvWxYz here",
            "AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q": "google AIzaSyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
            "ghp_" + "a1B2" * 9: "github ghp_" + "a1B2" * 9,
            "AKIAABCDEFGHIJKLMNOP": "aws AKIAABCDEFGHIJKLMNOP",
        }
        for secret, line in cases.items():
            with self.subTest(secret=secret[:8]):
                shown = redact(line)
                self.assertNotIn(secret, shown)
                self.assertIn(Console_Capture.REDACTED, shown)

    def test_a_named_value_loses_its_value_and_keeps_its_name(self):
        self.assertEqual(redact('api_key = "abcdef123456"'), 'api_key = "<redacted>"')
        self.assertEqual(redact("Authorization: Bearer abcdefghijklmnop1234"), "Authorization: Bearer <redacted>")
        self.assertEqual(redact("Bearer abcdefghijklmnop1234"), "Bearer <redacted>")
        self.assertEqual(redact("GET /v1?key=abc123def456&x=1"), "GET /v1?key=<redacted>&x=1")
        self.assertEqual(redact('{"password": "hunter2hunter2"}'), '{"password": "<redacted>"}')

    def test_ordinary_text_is_left_alone(self):
        for line in (
            "Loading PyTorch...",
            "Building network display: 8,000 nodes, 2,987,354 edges.",
            "the key frame moved; token count 5; a secret garden",
            "C:\\Users\\me\\.venv\\Lib\\site-packages\\numba\\core\\typed_passes.py:336: NumbaPerformanceWarning",
            "Traceback (most recent call last):",
            "sk-short",
        ):
            with self.subTest(line=line):
                self.assertEqual(redact(line), line)

    def test_redacting_twice_changes_nothing_more(self):
        once = redact("hf_AbCdEfGhIjKlMnOpQrStUvWxYz and api_key=abcdef123456")
        self.assertEqual(redact(once), once)


class SplitterTests(unittest.TestCase):
    def test_lines_split_at_newlines_and_a_crlf_ends_a_line_too(self):
        splitter = Splitter()
        self.assertEqual(splitter.feed(b"one\ntwo\r\nthr"), [("one", False), ("two", False)])
        self.assertEqual(splitter.feed(b"ee\n"), [("three", False)])

    def test_a_character_cut_between_chunks_is_not_lost(self):
        splitter = Splitter()
        data = "α-amylase 蛋白质\n".encode("utf-8")
        found = []
        for index in range(len(data)):  # one byte at a time: every possible cut
            found += splitter.feed(data[index:index + 1])
        self.assertEqual(found, [("α-amylase 蛋白质", False)])

    def test_invalid_bytes_become_replacement_characters(self):
        self.assertEqual(Splitter().feed(b"a\xffb\n"), [("a\ufffdb", False)])

    def test_carriage_returns_collapse_to_the_last_state(self):
        self.assertEqual(collapse_carriage_returns("10%\r50%\r100% done"), "100% done")
        self.assertEqual(collapse_carriage_returns("done\r"), "done")
        self.assertEqual(collapse_carriage_returns("plain"), "plain")
        self.assertEqual(Splitter().feed(b"10%\r50%\r100%\n"), [("100%", False)])

    def test_a_progress_line_is_partial_until_its_newline(self):
        splitter = Splitter()
        self.assertEqual(splitter.feed(b"step 1\rstep 2"), [("step 2", True)])
        self.assertEqual(splitter.feed(b"\rstep 3"), [("step 3", True)])
        self.assertEqual(splitter.feed(b"\n"), [("step 3", False)])
        self.assertEqual(splitter.feed(b"text without a newline"), [])  # shown when the line ends

    def test_what_is_left_at_the_end_of_the_stream_is_a_line(self):
        splitter = Splitter()
        splitter.feed(b"no newline at the end")
        self.assertEqual(splitter.finish(), [("no newline at the end", False)])
        self.assertEqual(splitter.finish(), [])

    def test_output_without_any_newline_is_cut_at_a_limit(self):
        with mock.patch.object(Console_Capture, "MAX_LINE_CHARACTERS", 10):
            found = Splitter().feed(b"x" * 25)
        self.assertEqual(found, [("x" * 25, False)])


class RecorderTests(unittest.TestCase):
    def test_the_ring_keeps_the_newest_lines_and_counts_the_rest(self):
        ring = Ring(ring_lines=3)
        for number in range(5):
            ring.add("stdout", f"line {number}")
        self.assertEqual([line.text for line in ring.snapshot()], ["line 2", "line 3", "line 4"])
        self.assertEqual((ring.dropped, ring.total), (2, 5))

    def test_each_line_knows_its_stream_and_the_ring_numbers_them(self):
        ring = Ring()
        first = ring.add("stdout", "a")
        second = ring.add("stderr", "b")
        self.assertEqual((first.stream, second.stream), ("stdout", "stderr"))
        self.assertEqual(second.seq, first.seq + 1)

    def test_a_progress_line_is_replaced_in_place_by_its_next_state_and_by_the_finished_line(self):
        ring = Ring()
        ring.add("stdout", "before")
        first = ring.add("stdout", "10%", partial=True)
        ring.add("stdout", "50%", partial=True)
        done = ring.add("stdout", "100% done")
        texts = [line.text for line in ring.snapshot()]
        self.assertEqual(texts, ["before", "100% done"])
        self.assertEqual(done.seq, first.seq)
        self.assertEqual(ring.total, 2)

    def test_listeners_hear_every_line_and_one_that_fails_stops_nothing(self):
        ring = Ring()
        heard = []
        ring.subscribe(lambda line: 1 / 0)
        ring.subscribe(lambda line: heard.append(line.text))
        ring.add("stdout", "one")
        ring.add("stderr", "two")
        self.assertEqual(heard, ["one", "two"])
        callback = heard.append
        ring.subscribe(callback)
        ring.unsubscribe(callback)
        ring.unsubscribe(callback)  # no error
        ring.add("stdout", "three")
        self.assertEqual(heard, ["one", "two", "three"])

    def test_progress_states_reach_listeners_at_most_every_fifty_milliseconds(self):
        ring = Ring()
        heard = []
        ring.subscribe(lambda line: heard.append((line.text, line.partial)))
        for state in ("1%", "2%", "3%"):
            ring.add("stdout", state, partial=True)
        ring.add("stdout", "done")
        self.assertEqual(heard, [("1%", True), ("done", False)])
        self.assertEqual([line.text for line in ring.snapshot()], ["done"])

    def test_the_ring_holds_the_redacted_text(self):
        ring = Ring()
        ring.add("stderr", "calling with api_key=abcdef123456")
        self.assertEqual(ring.snapshot()[0].text, "calling with api_key=<redacted>")


class LogFileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log_dir = Path(self.directory.name) / "logs"

    def test_the_log_is_named_for_the_app_the_time_and_the_process(self):
        recorder = Recorder("viewer", self.log_dir)
        self.addCleanup(recorder.close_log)
        name = Path(recorder.log_path).name
        self.assertRegex(name, rf"^viewer-\d{{8}}-\d{{6}}-{os.getpid()}\.log$")
        self.assertEqual(Path(recorder.log_path).parent, self.log_dir)

    def test_each_line_is_written_with_its_time_and_stream(self):
        recorder = Recorder("viewer", self.log_dir)
        recorder.add("stdout", "hello α")
        recorder.add("stderr", "oops")
        recorder.close_log()
        lines = Path(recorder.log_path).read_text(encoding="utf-8").splitlines()
        self.assertRegex(lines[0], r"^\d\d:\d\d:\d\d O hello α$")
        self.assertRegex(lines[1], r"^\d\d:\d\d:\d\d E oops$")

    def test_the_log_holds_the_redacted_text_and_no_progress_states(self):
        recorder = Recorder("viewer", self.log_dir)
        recorder.add("stdout", "key hf_AbCdEfGhIjKlMnOpQrStUvWxYz")
        recorder.add("stdout", "10%", partial=True)
        recorder.add("stdout", "100% done")
        recorder.close_log()
        text = Path(recorder.log_path).read_text(encoding="utf-8")
        self.assertNotIn("hf_AbCd", text)
        self.assertNotIn("10%", text)
        self.assertIn("100% done", text)

    def test_a_full_log_rolls_over_once_and_keeps_going(self):
        recorder = Recorder("viewer", self.log_dir, log_bytes=200)
        for number in range(60):
            recorder.add("stdout", f"line {number:03d} " + "x" * 20)
        recorder.close_log()
        current = Path(recorder.log_path).read_text(encoding="utf-8")
        rolled = Path(recorder.log_path + ".1").read_text(encoding="utf-8")
        self.assertIn("line 059", current)
        self.assertLessEqual(len(current.encode()), 200)
        self.assertLessEqual(len(rolled.encode()), 200)
        self.assertNotIn("line 000", rolled + current)  # only the last rolled file is kept

    def test_the_newest_ten_logs_of_each_app_stay(self):
        for number in range(13):
            for app in ("viewer", "tools"):
                self.log_dir.mkdir(exist_ok=True)
                stem = f"{app}-20261024-1200{number:02d}-{1000 + number}"
                (self.log_dir / (stem + ".log")).write_text("x")
                (self.log_dir / (stem + ".log.1")).write_text("x")
                (self.log_dir / (stem + ".log.crash")).write_text("x")
        (self.log_dir / "notes.txt").write_text("keep")
        prune_logs(self.log_dir, "viewer")
        names = sorted(os.listdir(self.log_dir))
        viewer_logs = [name for name in names if name.startswith("viewer-") and name.endswith(".log")]
        self.assertEqual(len(viewer_logs), 10)
        self.assertEqual(viewer_logs[0], "viewer-20261024-120003-1003.log")
        self.assertNotIn("viewer-20261024-120002-1002.log.crash", names)  # a pruned log's other files go too
        self.assertNotIn("viewer-20261024-120002-1002.log.1", names)
        self.assertEqual(len([name for name in names if name.startswith("tools-")]), 39)  # another app: untouched
        self.assertIn("notes.txt", names)

    def test_opening_a_log_prunes_its_apps_older_logs(self):
        self.log_dir.mkdir()
        for number in range(Console_Capture.LOG_KEEP):
            (self.log_dir / f"viewer-20200101-0000{number:02d}-{number}.log").write_text("old")
        recorder = Recorder("viewer", self.log_dir)
        self.addCleanup(recorder.close_log)
        viewer_logs = [name for name in os.listdir(self.log_dir) if name.startswith("viewer-")]
        self.assertEqual(len(viewer_logs), Console_Capture.LOG_KEEP)
        self.assertIn(Path(recorder.log_path).name, viewer_logs)

    def test_a_folder_that_cannot_be_made_leaves_the_ring_working(self):
        blocker = Path(self.directory.name) / "file"
        blocker.write_text("x")
        recorder = Recorder("viewer", blocker / "logs")
        recorder.add("stdout", "still here")
        self.assertIsNone(recorder.log_path)
        self.assertIsNotNone(recorder.log_error)
        self.assertEqual(recorder.snapshot()[0].text, "still here")

    def test_a_log_that_fails_to_write_stops_writing_but_not_recording(self):
        recorder = Recorder("viewer", self.log_dir)
        recorder._log.close()  # the next write raises ValueError... which a closed file does; use OSError instead
        recorder._log = mock.Mock()
        recorder._log.write.side_effect = OSError("disk full")
        recorder.add("stdout", "one")
        recorder.add("stdout", "two")
        self.assertEqual([line.text for line in recorder.snapshot()], ["one", "two"])
        self.assertEqual(recorder.log_error, "disk full")


class OriginalStreamTests(unittest.TestCase):
    def test_a_missing_stream_is_not_read(self):
        with mock.patch.object(Console_Capture.os, "fstat", side_effect=OSError):
            self.assertFalse(Console_Capture.original_is_read(1))

    def test_a_pipe_or_file_is_read(self):
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        self.addCleanup(os.close, write_end)
        self.assertTrue(Console_Capture.original_is_read(write_end))
        with tempfile.TemporaryFile() as handle:
            self.assertTrue(Console_Capture.original_is_read(handle.fileno()))

    @unittest.skipIf(sys.platform == "win32", "NUL is told by its console on Windows")
    def test_the_null_device_is_not_read(self):
        with open(os.devnull, "wb") as handle:
            self.assertFalse(Console_Capture.original_is_read(handle.fileno()))

    def test_a_terminal_is_read_unless_it_is_a_hidden_console(self):
        with tempfile.TemporaryFile() as handle, mock.patch.object(
            Console_Capture.os, "isatty", return_value=True
        ), mock.patch.object(Console_Capture, "_is_null_device", return_value=False):
            descriptor = handle.fileno()
            with mock.patch.object(Console_Capture.sys, "platform", "linux"):
                self.assertTrue(Console_Capture.original_is_read(descriptor))
            with mock.patch.object(Console_Capture.sys, "platform", "win32"):
                for state, expected in (("shown", True), ("hidden", False), ("none", False)):
                    with mock.patch.object(Console_Capture, "_console_window_state", return_value=state):
                        self.assertEqual(Console_Capture.original_is_read(descriptor), expected, state)


@unittest.skipUnless(sys.platform == "win32", "console windows are Windows'")
class ConsoleWindowStateTests(unittest.TestCase):
    """What the process's console window is, told with stand-ins for the two Windows calls: a test must not open
    a console of its own (even a hidden one) on the screen of whoever runs it."""

    def state(self, handle, class_name="ConsoleWindowClass", visible=True):
        import ctypes
        from types import SimpleNamespace

        def write_class_name(window, buffer, size):
            buffer.value = class_name
            return len(class_name)

        kernel32 = SimpleNamespace(GetConsoleWindow=mock.MagicMock(return_value=handle))
        user32 = SimpleNamespace(IsWindowVisible=mock.MagicMock(return_value=int(visible)),
                                 GetClassNameW=mock.MagicMock(side_effect=write_class_name))
        with mock.patch.object(ctypes, "windll", SimpleNamespace(kernel32=kernel32, user32=user32), create=True):
            return Console_Capture._console_window_state()

    def test_no_window_is_none_a_visible_one_is_shown_and_a_hidden_one_is_hidden(self):
        self.assertEqual(self.state(0), "none")
        self.assertEqual(self.state(0x1234, visible=True), "shown")
        self.assertEqual(self.state(0x1234, visible=False), "hidden")

    def test_the_window_a_terminal_hosts_its_console_in_counts_as_shown_though_it_is_never_visible(self):
        self.assertEqual(self.state(0x1234, "PseudoConsoleWindow", visible=False), "shown")


class LogFolderTests(unittest.TestCase):
    def test_the_logs_go_to_temp_logs_in_the_project_unless_a_folder_is_named(self):
        self.assertEqual(Console_Capture.log_directory({}), PROJECT_ROOT / "temp" / "logs")
        self.assertEqual(Console_Capture.log_directory({"SSN_LOG_DIR": " "}), PROJECT_ROOT / "temp" / "logs")
        self.assertEqual(Console_Capture.log_directory({"SSN_LOG_DIR": "/elsewhere"}), Path("/elsewhere"))

    def test_a_test_run_keeps_the_logs_of_every_window_it_starts_out_of_the_project(self):
        named = Path(os.environ["SSN_LOG_DIR"])
        self.assertNotEqual(named, PROJECT_ROOT / "temp" / "logs")
        self.assertEqual(Console_Capture.log_directory(), named)


class EntryPointTests(unittest.TestCase):
    def test_capture_is_wanted_unless_headless_console_or_switched_off(self):
        self.assertTrue(Console_Capture.wanted(["EMAPSSN_Config.py"], {}))
        self.assertFalse(Console_Capture.wanted(["EMAPSSN_Config.py", "--headless"], {}))
        self.assertFalse(Console_Capture.wanted(["EMAPSSN_Config.py", "--console"], {}))
        self.assertFalse(Console_Capture.wanted(["x"], {"SSN_NO_CONSOLE_CAPTURE": "1"}))
        self.assertTrue(Console_Capture.wanted(["x"], {"SSN_NO_CONSOLE_CAPTURE": "0"}))

    def test_start_gives_a_headless_run_only_utf8_streams(self):
        with mock.patch.object(Console_Capture, "install") as install, mock.patch(
            "utilities.Output_Streams.configure_output_streams"
        ) as configure:
            self.assertIsNone(Console_Capture.start("viewer", ["x", "--headless"]))
            self.assertIsNone(Console_Capture.start("viewer", ["x"], headless=True))
            self.assertIsNone(Console_Capture.start("config", ["x", "--console"]))
        install.assert_not_called()
        self.assertEqual(configure.call_count, 3)

    def test_start_installs_the_capture_for_a_window(self):
        sentinel = object()
        with mock.patch.object(Console_Capture, "install", return_value=sentinel) as install:
            self.assertIs(Console_Capture.start("tools", ["x"]), sentinel)
        install.assert_called_once_with("tools", None)

    def test_a_capture_that_cannot_start_never_keeps_the_window_from_starting(self):
        with mock.patch.object(Console_Capture, "install", side_effect=OSError("no pipes")), mock.patch(
            "utilities.Output_Streams.configure_output_streams"
        ) as configure, mock.patch.object(Console_Capture.sys, "stderr", new_callable=lambda: __import__("io").StringIO()) as err:
            self.assertIsNone(Console_Capture.start("tools", ["x"]))
        configure.assert_called_once()
        self.assertIn("no pipes", err.getvalue())

    def test_capture_refuses_to_start_in_the_mcp_server(self):
        with mock.patch.object(Console_Capture.sys, "argv", [str(SRC_DIR / "EMAPSSN_MCP_Server.py")]), mock.patch.object(
            Console_Capture, "_capture", None
        ), mock.patch.object(Console_Capture, "Capture") as capture:
            with self.assertRaises(RuntimeError):
                Console_Capture.install("mcp")
        capture.assert_not_called()

    def test_only_the_window_entry_points_start_a_capture(self):
        """The MCP server, whose standard output is the protocol, and every library module never do."""
        allowed = {"EMAPSSN_Config.py", "EMAPSSN_Tools.py", "EMAPSSN_Viewer.py"}
        found = set()
        for path in SRC_DIR.rglob("*.py"):
            if path.name == "Console_Capture.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("Console_Capture"):
                    if {alias.name for alias in node.names} & {"start", "install"}:
                        found.add(path.name)
                elif isinstance(node, ast.Attribute) and node.attr in {"start", "install"}:
                    if getattr(node.value, "id", "") == "Console_Capture":
                        found.add(path.name)
        # The VR program's entry point is in opt_vr and is checked there.
        self.assertEqual(found - allowed, set(), "only a window's entry point may start the capture")
        self.assertEqual(allowed - found, set(), "every window's entry point starts the capture")

    def test_the_mcp_server_does_not_load_the_capture(self):
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); import EMAPSSN_MCP_Server; "
            "print('utilities.Console_Capture' in sys.modules)"
        )
        done = subprocess.run([sys.executable, "-c", program, str(SRC_DIR)], capture_output=True, text=True,
                              env=child_environment(), timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "False")


class ProbeRun:
    """One run of console_capture_probe.py in a context, and what it left behind."""

    def __init__(self, test, context, options=(), forward="auto"):
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.result_path = self.directory / "result.json"
        self.log_dir = self.directory / "logs"
        self.stdout_bytes = self.stderr_bytes = self.file_bytes = b""
        executable, flags, streams = self._context(context)
        command = [executable, "-u", str(PROBE), str(self.result_path), str(self.log_dir), forward, *options]
        self.process = None
        if streams == "pipes":
            done = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL, env=child_environment(),
                                  creationflags=flags, timeout=120)
            self.code, self.stdout_bytes, self.stderr_bytes = done.returncode, done.stdout, done.stderr
        elif streams == "file":
            path = self.directory / "both.log"
            with open(path, "wb") as handle:
                done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                      env=child_environment(), creationflags=flags, timeout=120)
            self.code, self.file_bytes = done.returncode, path.read_bytes()
        elif streams == "null":
            done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, env=child_environment(), creationflags=flags, timeout=120)
            self.code = done.returncode
        else:  # "inherit": no standard stream given at all, so a new hidden console or no handles
            done = subprocess.run(command, env=child_environment(), creationflags=flags, timeout=120)
            self.code = done.returncode
        self.result = json.loads(self.result_path.read_text(encoding="utf-8")) if self.result_path.exists() else None

    @staticmethod
    def _context(name):
        flags = 0
        if name == "hidden_console":
            flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
            return sys.executable, flags, "inherit"
        if name == "no_handles":
            pythonw = Path(sys.base_prefix) / "pythonw.exe"
            return str(pythonw), DETACHED_PROCESS, "inherit"
        if sys.platform == "win32":
            flags = CREATE_NO_WINDOW
        return sys.executable, flags, name

    @property
    def ring_texts(self):
        return [text for _, text in self.result["ring"]]

    def logged(self):
        """The text of the one log the probe wrote (plus its rolled-over part)."""
        logs = sorted(self.log_dir.glob("probe-*.log"))
        return "".join(path.read_text(encoding="utf-8") for path in logs)


class ContextTests(unittest.TestCase):
    """Every source of output, in each context the windows start in."""

    def assertAllSourcesCaptured(self, run):
        self.assertEqual(run.code, 0, run)
        self.assertIsNotNone(run.result, "the probe died before its result")
        self.assertEqual(run.result["missing_in_ring"], [])
        logged = run.logged()
        for marker in ALL_MARKERS:
            with self.subTest(marker=marker):
                self.assertTrue(any(marker in text for text in run.ring_texts), f"{marker} is not in the ring")
                self.assertIn(marker, logged)

    def test_pipes_read_by_the_parent_keep_getting_every_byte(self):
        """An MCP client's Viewer_Terminal reads the Viewer's two pipes: the tee is unchanged."""
        run = ProbeRun(self, "pipes", options=("qt", "close"))
        self.assertAllSourcesCaptured(run)
        self.assertEqual(run.result["forwarding"], {"stdout": True, "stderr": True})
        out, err = run.stdout_bytes.decode("utf-8"), run.stderr_bytes.decode("utf-8")
        self.assertIn("SRC python stdout", out)
        self.assertIn("SRC unicode \u03b1-amylase \u2713 \u86cb\u767d\u8d28 \u00e9", out)
        self.assertIn("SRC c puts", out)
        self.assertIn("SRC child python stdout", out)
        self.assertIn("SRC last line before close", out)
        self.assertIn("SRC python stderr", err)
        self.assertIn("SRC child python stderr", err)
        self.assertIn("ZeroDivisionError: division by zero", err)
        self.assertNotIn("SRC python stdout", err)
        self.assertNotIn("SRC python stderr", out)
        # What is passed on is the raw output, progress states and all.
        self.assertIn("progress 10%\rprogress 50%\rSRC progress 100% done", out)

    def test_a_qt_warning_lands_in_the_ring_tagged(self):
        run = ProbeRun(self, "pipes", options=("qt",))
        self.assertEqual(run.code, 0)
        self.assertIn("[Qt warning] SRC qt warning", run.ring_texts)
        self.assertIn("[Qt critical] SRC qt critical", run.ring_texts)
        self.assertIn("[Qt warning] SRC qt warning", run.logged())

    def test_without_the_handler_qt_still_writes_to_stderr(self):
        program = (
            "import sys, time; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "c = C.install('probe', sys.argv[2], False); from PySide6 import QtCore; QtCore.qWarning('bare qt warning'); "
            "time.sleep(0.5); sys.__stdout__.write(repr([line.text for line in c.snapshot()]))"
        )
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run([sys.executable, "-c", program, str(SRC_DIR), directory], capture_output=True,
                                  text=True, env=child_environment(), timeout=120,
                                  creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self.assertIn("bare qt warning", done.stdout)

    def test_one_file_for_both_streams_gets_everything_as_the_launcher_monitors_log_does(self):
        run = ProbeRun(self, "file")
        self.assertAllSourcesCaptured(run)
        self.assertEqual(run.result["forwarding"], {"stdout": True, "stderr": True})
        text = run.file_bytes.decode("utf-8")
        for marker in ALL_MARKERS:
            self.assertIn(marker, text)

    def test_nul_is_not_read_so_nothing_is_passed_on(self):
        run = ProbeRun(self, "null")
        self.assertAllSourcesCaptured(run)
        self.assertEqual(run.result["forwarding"], {"stdout": False, "stderr": False})

    @unittest.skipUnless(sys.platform == "win32", "a hidden console is Windows' CREATE_NO_WINDOW")
    def test_a_hidden_console_is_not_read(self):
        run = ProbeRun(self, "hidden_console")
        self.assertAllSourcesCaptured(run)
        self.assertEqual(run.result["forwarding"], {"stdout": False, "stderr": False})
        self.assertIn("TextIOWrapper", run.result["stdout_before"])

    @unittest.skipUnless(sys.platform == "win32" and (Path(sys.base_prefix) / "pythonw.exe").is_file(),
                         "pythonw.exe of the base interpreter starts with no standard handles")
    def test_pythonw_with_no_standard_handles_is_captured_too(self):
        run = ProbeRun(self, "no_handles")
        self.assertEqual(run.result["stdout_before"], "None")
        self.assertCapturedWithoutChildren(run)

    def assertCapturedWithoutChildren(self, run):
        self.assertIsNotNone(run.result)
        self.assertEqual(run.result["forwarding"], {"stdout": False, "stderr": False})
        # (A C library's stdout stream is not reopened: with no handle at start the C runtime keeps none.)
        for marker in ("python stdout", "python stderr", "unicode", "os.write stdout", "os.write stderr",
                       "win32 stdout handle", "win32 stderr handle", "python warning", "ZeroDivisionError"):
            with self.subTest(marker=marker):
                self.assertTrue(any(marker in text for text in run.ring_texts), marker)

    def test_progress_lines_collapse_in_the_ring_and_the_log(self):
        run = ProbeRun(self, "null")
        self.assertIn("SRC progress 100% done", run.ring_texts)
        self.assertNotIn("10%", "".join(run.ring_texts))
        self.assertNotIn("progress 10%", run.logged())

    def test_a_native_crash_leaves_a_dump_beside_the_log(self):
        run = ProbeRun(self, "pipes", options=("crash",))
        self.assertNotEqual(run.code, 0)
        crashes = list(run.log_dir.glob("probe-*.log.crash"))
        self.assertEqual(len(crashes), 1)
        dump = crashes[0].read_text(encoding="utf-8", errors="replace")
        self.assertRegex(dump, r"(?i)fatal exception|fatal python error|access violation|segmentation fault")
        self.assertIn("console_capture_probe.py", dump)
        self.assertIn("about to crash natively", run.logged())

    def test_a_clean_exit_leaves_no_crash_file_and_puts_the_streams_back(self):
        run = ProbeRun(self, "pipes", options=("close",))
        self.assertEqual(run.code, 0)
        self.assertEqual(list(run.log_dir.glob("*.crash")), [])
        self.assertEqual(run.result["streams_restored"], [True, True])
        self.assertIn("last line before close", run.result["log_after_close"])  # drained before the process ended
        self.assertIn("SRC last line before close", run.stdout_bytes.decode("utf-8"))
        # After close() the descriptors are the process's own again: this goes to the parent's pipe, not the log.
        self.assertIn("SRC after close", run.stdout_bytes.decode("utf-8"))
        self.assertNotIn("SRC after close", run.logged())

    def test_a_viewer_an_mcp_client_starts_keeps_its_tee_through_viewer_terminal(self):
        """Viewer_Terminal.py's stdout.log and stderr.log hold what its parent's pipes got, as without the capture."""
        program = (
            "import os, sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "C.install('viewer', sys.argv[2]); print('python output'); os.write(1, b'native partial'); "
            "os.write(2, b'native error'); print('Viewer Ready', flush=True)"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SRC_DIR / "mcp_server" / "viewer" / "Viewer_Terminal.py"), directory,
                 sys.executable, "-u", "-c", program, str(SRC_DIR), os.path.join(directory, "logs")],
                capture_output=True, timeout=60, env=child_environment(),
                creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((Path(directory) / "stdout.log").read_bytes(), result.stdout)
            self.assertEqual((Path(directory) / "stderr.log").read_bytes(), result.stderr)
            self.assertIn(b"python output", result.stdout)
            self.assertIn(b"native partial", result.stdout)
            self.assertIn(b"Viewer Ready", result.stdout)
            self.assertIn(b"native error", result.stderr)
            logs = list((Path(directory) / "logs").glob("viewer-*.log"))
            self.assertIn("Viewer Ready", logs[0].read_text(encoding="utf-8"))

    def test_the_original_stream_is_not_passed_on_when_asked_not_to(self):
        run = ProbeRun(self, "pipes", forward="off")
        self.assertEqual(run.result["forwarding"], {"stdout": False, "stderr": False})
        self.assertEqual(run.stdout_bytes, b"")
        self.assertEqual(run.stderr_bytes, b"")
        self.assertEqual(run.result["missing_in_ring"], [])

    def test_a_burst_keeps_the_ring_and_the_pipe_complete(self):
        run = ProbeRun(self, "pipes", options=("burst=30000",))
        self.assertEqual(run.code, 0)
        self.assertEqual(run.result["burst_in_ring"], Console_Capture.RING_LINES)  # the newest 20,000
        out = run.stdout_bytes.decode("utf-8").splitlines()
        self.assertEqual(sum(1 for line in out if line.startswith("burst line")), 30000)
        self.assertGreater(run.result["ring_dropped"], 0)  # the ring is a window, the log has it all
        self.assertIn("burst line 29999", run.logged())
        self.assertIn("burst line 0", run.logged())

    def test_a_reader_that_stalls_cannot_stall_the_process(self):
        program = (
            "import sys, time; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "C.FORWARD_QUEUE_CHUNKS = 4; c = C.install('probe', sys.argv[2], True); "
            "[print('x' * 200 + str(i)) for i in range(20000)]; time.sleep(0.5); "
            "open(sys.argv[3], 'w').write(str(c.forward_dropped) + ' ' + c.snapshot()[-1].text[-5:])"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = os.path.join(directory, "result.txt")
            # The parent holds the read ends and never reads: the pipes fill up.
            process = subprocess.Popen([sys.executable, "-c", program, str(SRC_DIR), directory, result],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       env=child_environment(),
                                       creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            try:
                deadline = time.time() + 60
                while process.poll() is None and time.time() < deadline:
                    time.sleep(0.1)
                finished = process.poll() is not None
            finally:
                process.kill()
                process.communicate()
            self.assertTrue(finished, "the process hung behind a reader that never reads")
            dropped, last = Path(result).read_text().split()
            self.assertGreater(int(dropped), 0)
            self.assertEqual(last, "19999")


if __name__ == "__main__":
    unittest.main()
