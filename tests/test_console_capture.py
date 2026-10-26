# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Output capture (utilities.Console_Capture): secrets are redacted in the ring and the log, carriage returns collapse, lines are split across chunk boundaries, the ring and the rotating log keep what they should, the original stream is passed on only when someone reads it, and every source of output reaches the ring in the contexts the windows start in (pipes, a file, no handles at all, a hidden console, NUL), with Qt's warnings, a native crash and the last lines before exit included. No capture runs in the MCP server."""

from __future__ import annotations

import ast
import io
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

    def _stale_crash_files(self, dead_pids):
        """Four logs of one app, each with the crash file a run leaves, and a log of another app with one."""
        self.log_dir.mkdir()
        for stem, size in (
            ("viewer-20261024-120000-111", 0),   # dead, nothing crashed: a killed run's leftover
            ("viewer-20261024-120001-222", 0),   # a run that is still going
            ("viewer-20261024-120002-333", 40),  # dead, with a dump
            ("viewer-20261024-120003-444", None),  # dead, no crash file at all
            ("tools-20261024-120000-111", 0),    # another app's, left for its own start
        ):
            (self.log_dir / (stem + ".log")).write_text("x")
            if size is not None:
                (self.log_dir / (stem + ".log.crash")).write_bytes(b"d" * size)
        return mock.patch.object(Console_Capture, "process_is_gone", side_effect=lambda pid: pid in dead_pids)

    def test_an_empty_crash_file_of_a_process_that_is_gone_is_removed_and_nothing_else(self):
        with self._stale_crash_files({111, 333, 444}):
            prune_logs(self.log_dir, "viewer")
        names = sorted(os.listdir(self.log_dir))
        self.assertEqual(names, [
            "tools-20261024-120000-111.log", "tools-20261024-120000-111.log.crash",
            "viewer-20261024-120000-111.log",  # the log stays, only the empty crash file goes
            "viewer-20261024-120001-222.log", "viewer-20261024-120001-222.log.crash",  # still running
            "viewer-20261024-120002-333.log", "viewer-20261024-120002-333.log.crash",  # a dump is kept
            "viewer-20261024-120003-444.log",
        ])

    def test_opening_a_log_removes_the_empty_crash_files_a_killed_run_left(self):
        with self._stale_crash_files({111, 333, 444}):
            recorder = Recorder("viewer", self.log_dir)
            self.addCleanup(recorder.close_log)
        names = os.listdir(self.log_dir)
        self.assertNotIn("viewer-20261024-120000-111.log.crash", names)
        self.assertIn("viewer-20261024-120001-222.log.crash", names)
        self.assertIn("viewer-20261024-120002-333.log.crash", names)

    def test_a_crash_file_that_cannot_be_checked_or_removed_does_no_harm(self):
        with self._stale_crash_files({111}):
            with mock.patch.object(Console_Capture.os, "remove", side_effect=PermissionError("held open")):
                prune_logs(self.log_dir, "viewer")
        self.assertIn("viewer-20261024-120000-111.log.crash", os.listdir(self.log_dir))

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


class ColourPlumbingTests(unittest.TestCase):
    def test_the_log_gets_the_text_without_colours_and_the_ring_keeps_them(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder("viewer", directory)
            recorder.add("stdout", "\x1b[1;31mred\x1b[0m and \x1b[38;2;1;2;3m\u25cf\x1b[0m")
            recorder.add("stdout", "no colour")
            recorder.close_log()
            logged = Path(recorder.log_path).read_text(encoding="utf-8")
        self.assertNotIn("\x1b", logged)
        self.assertRegex(logged, r"\d\d:\d\d:\d\d O red and \u25cf\n")
        self.assertEqual(recorder.snapshot()[0].text, "\x1b[1;31mred\x1b[0m and \x1b[38;2;1;2;3m\u25cf\x1b[0m")

    def test_only_a_terminal_that_can_show_colours_is_passed_them(self):
        with mock.patch.object(Console_Capture.os, "isatty", return_value=False):
            self.assertFalse(Console_Capture._takes_colours(5))
        with mock.patch.object(Console_Capture.os, "isatty", side_effect=OSError):
            self.assertFalse(Console_Capture._takes_colours(5))
        with mock.patch.object(Console_Capture.os, "isatty", return_value=True):
            with mock.patch.object(Console_Capture.sys, "platform", "linux"):
                self.assertTrue(Console_Capture._takes_colours(5))
            with mock.patch.object(Console_Capture.sys, "platform", "win32"), mock.patch.object(
                Console_Capture, "_windows_console_takes_ansi", return_value=False, create=True
            ):
                self.assertFalse(Console_Capture._takes_colours(5))  # a console that cannot read them gets none
            with mock.patch.object(Console_Capture.sys, "platform", "win32"), mock.patch.object(
                Console_Capture, "_windows_console_takes_ansi", return_value=True, create=True
            ):
                self.assertTrue(Console_Capture._takes_colours(5))

    def test_the_capture_knows_its_own_streams_and_nothing_else(self):
        own = io.StringIO()
        with mock.patch.object(Console_Capture, "_capture", mock.Mock(text_streams=(own,))):
            self.assertTrue(Console_Capture.is_capture_stream(own))
            self.assertFalse(Console_Capture.is_capture_stream(io.StringIO()))
        with mock.patch.object(Console_Capture, "_capture", None):
            self.assertFalse(Console_Capture.is_capture_stream(own))
        self.assertEqual(Recorder("viewer", None).text_streams, ())  # no capture, no streams


class ProcessGoneTests(unittest.TestCase):
    """Console_Capture.process_is_gone: the pruning's question, on both systems (a probe that ends the process on Windows would be a bug)."""

    @staticmethod
    def _start(code):
        return subprocess.Popen(
            [sys.executable, "-c", code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    def test_this_process_and_a_running_one_are_not_gone_and_are_not_disturbed(self):
        self.assertFalse(Console_Capture.process_is_gone(os.getpid()))
        child = self._start("import time; time.sleep(60)")
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        self.assertFalse(Console_Capture.process_is_gone(child.pid))
        self.assertFalse(Console_Capture.process_is_gone(child.pid))
        self.assertIsNone(child.poll(), "asking must not end the process")

    def test_a_process_that_has_ended_is_gone(self):
        # Windows hands a finished process's number out again quickly: ask about a few.
        for _attempt in range(5):
            child = self._start("pass")
            child.wait()
            if Console_Capture.process_is_gone(child.pid):
                return
        self.fail("a process that ended was never reported gone")

    def test_a_number_that_means_no_process_is_not_called_gone(self):
        for pid in (0, -1):
            with self.subTest(pid=pid):
                self.assertFalse(Console_Capture.process_is_gone(pid))

    @unittest.skipIf(sys.platform == "win32", "os.kill(pid, 0) is the POSIX probe")
    def test_a_process_of_another_user_is_not_gone(self):
        with mock.patch.object(Console_Capture.os, "kill", side_effect=PermissionError("not yours")):
            self.assertFalse(Console_Capture.process_is_gone(4242))
        with mock.patch.object(Console_Capture.os, "kill", side_effect=ProcessLookupError()):
            self.assertTrue(Console_Capture.process_is_gone(4242))


class CrashFileTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "file modes are POSIX")
    def test_the_crash_file_is_for_its_owner_alone_as_the_log_is_whatever_the_umask(self):
        program = (
            "import os, sys; sys.path.insert(0, sys.argv[1]); os.umask(0); from utilities import Console_Capture as C; "
            "c = C.install('probe', sys.argv[2], False); "
            "open(sys.argv[3], 'w').write(' '.join(oct(os.stat(p).st_mode & 0o777) for p in (c._crash_path, c.log_path)))"
        )
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run(
                [sys.executable, "-c", program, str(SRC_DIR), os.path.join(directory, "logs"),
                 os.path.join(directory, "modes")],
                capture_output=True, stdin=subprocess.DEVNULL, env=child_environment(), timeout=120,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(Path(directory, "modes").read_text().split(), ["0o600", "0o600"])


class QtPortalNoticeTests(unittest.TestCase):
    """The one Qt message GNOME makes on every start, and what the handler does with it and with its neighbours."""

    MESSAGE = (
        'Failed to register with host portal QDBusError("org.freedesktop.portal.Error.Failed", '
        '"Could not register app ID: Connection already associated with an application ID")'
    )

    def test_that_message_is_matched(self):
        self.assertTrue(Console_Capture.is_portal_already_registered("qt.qpa.services", self.MESSAGE))
        self.assertTrue(Console_Capture.is_portal_already_registered(
            "qt.qpa.services", self.MESSAGE.replace("already associated", "Already Associated")))

    def test_its_neighbours_are_not(self):
        other_failure = self.MESSAGE.replace(
            "Connection already associated with an application ID", "App info not found for 'emapssn_tools'")
        for category, message in (
            ("qt.qpa.services", other_failure),
            ("qt.qpa.services", 'Failed to register with host portal QDBusError("org.freedesktop.DBus.Error.'
                                'ServiceUnknown", "The name is not activatable")'),
            ("qt.qpa.services", "Could not find a program to open the URL"),
            ("qt.qpa.services", "Connection already associated with an application ID"),  # not the portal's
            ("qt.qpa.xcb", self.MESSAGE),
            ("default", self.MESSAGE),
            ("", self.MESSAGE),
            (None, self.MESSAGE),
        ):
            with self.subTest(category=category, message=message[:50]):
                self.assertFalse(Console_Capture.is_portal_already_registered(category, message))

    def _handler(self):
        from PySide6 import QtCore

        installed = []
        with mock.patch.object(Console_Capture, "_capture", object()), mock.patch.object(
            QtCore, "qInstallMessageHandler", side_effect=installed.append
        ):
            handler = Console_Capture.install_qt_message_handler()
        self.assertEqual(installed, [handler])
        return QtCore.QtMsgType, handler

    def test_the_handler_drops_that_message_and_tags_the_rest(self):
        from types import SimpleNamespace

        kinds, handler = self._handler()
        services = SimpleNamespace(category="qt.qpa.services")
        written = io.StringIO()
        with mock.patch.object(Console_Capture.sys, "stderr", written):
            handler(kinds.QtWarningMsg, services, self.MESSAGE)
            handler(kinds.QtWarningMsg, services, self.MESSAGE.replace("Connection already associated", "No portal"))
            handler(kinds.QtWarningMsg, SimpleNamespace(category="default"), "GPUInfo not initialized")
            handler(kinds.QtCriticalMsg, SimpleNamespace(category="qt.qpa.xcb"), self.MESSAGE)
        self.assertEqual(written.getvalue().splitlines(), [
            "[Qt warning] qt.qpa.services: " + self.MESSAGE.replace("Connection already associated", "No portal"),
            "[Qt warning] GPUInfo not initialized",
            "[Qt critical] qt.qpa.xcb: " + self.MESSAGE,
        ])


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


class _FullStream:
    """A raw writer that is full for its first ``refusals`` calls, as a non-blocking pipe is: it says "try
    again" with None (what FileIO.write gives) or a BlockingIOError, then takes ``chunk`` bytes at a time."""

    def __init__(self, refusals, *, raises=False, chunk=7):
        self.refusals, self.raises, self.chunk = refusals, raises, chunk
        self.received = bytearray()
        self.calls = 0

    def write(self, data):
        self.calls += 1
        if self.refusals > 0:
            self.refusals -= 1
            if self.raises:
                raise BlockingIOError(11, "Resource temporarily unavailable")
            return None
        taken = bytes(data[: self.chunk])
        self.received += taken
        return len(taken)


class NonBlockingStreamTests(unittest.TestCase):
    """A stream the process inherited in non-blocking mode that is full is waited for, not given up on.

    Forwarding used to stop for good at the first "try again" (EAGAIN), so one burst of output into a
    slow reader silenced the original stream until the process ended.
    """

    def test_a_full_stream_is_retried_until_it_takes_everything(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                writer = _FullStream(refusals=5, raises=raises)
                pauses = []
                Console_Capture._write_all(writer, b"0123456789" * 3, pause=pauses.append)
                self.assertEqual(bytes(writer.received), b"0123456789" * 3)
                self.assertEqual(len(pauses), 5)
                self.assertEqual(pauses, sorted(pauses))  # the waits grow
                self.assertLessEqual(max(pauses), Console_Capture.EAGAIN_LONGEST_PAUSE_SECONDS)

    def test_each_refusal_counts_from_the_last_byte_taken(self):
        """Refusals between bytes taken never add up to a reader that is gone."""
        class Alternating(_FullStream):
            def write(self, data):
                self.turn = getattr(self, "turn", 0) + 1
                return None if self.turn % 2 else super().write(data)

        writer = Alternating(refusals=0)
        with mock.patch.object(Console_Capture, "EAGAIN_PATIENCE_SECONDS", 0.0):
            Console_Capture._write_all(writer, b"x" * 70, pause=lambda seconds: None)
        self.assertEqual(bytes(writer.received), b"x" * 70)

    def test_a_reader_that_takes_nothing_for_the_patience_ends_the_forwarding(self):
        writer = _FullStream(refusals=10 ** 9)
        started = time.monotonic()
        with self.assertRaisesRegex(OSError, "nobody reads"):
            Console_Capture._write_all(writer, b"data", patience=0.2)
        self.assertGreaterEqual(time.monotonic() - started, 0.2)
        self.assertLess(time.monotonic() - started, 5)

    def test_a_failed_write_still_ends_the_forwarding_at_once(self):
        class Closed:
            def write(self, data):
                raise BrokenPipeError(32, "Broken pipe")

        class Stuck:
            def write(self, data):
                return 0

        with self.assertRaises(BrokenPipeError):
            Console_Capture._write_all(Closed(), b"data", pause=lambda seconds: self.fail("waited"))
        with self.assertRaisesRegex(OSError, "stopped taking output"):
            Console_Capture._write_all(Stuck(), b"data", pause=lambda seconds: self.fail("waited"))

    def test_a_slow_reader_of_a_non_blocking_pipe_gets_every_byte(self):
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, read_end)
        os.set_blocking(write_end, False)
        writer = io.FileIO(write_end, "wb", closefd=False)
        self.addCleanup(os.close, write_end)
        data = b"".join(b"%06d " % number + b"y" * 90 + b"\n" for number in range(4000))  # past a pipe's capacity
        received = bytearray()

        def read_slowly():
            time.sleep(0.3)  # the pipe is full and the writer has been told "try again" by now
            while len(received) < len(data):
                received.extend(os.read(read_end, 4096))
                time.sleep(0.001)

        reader = threading.Thread(target=read_slowly, daemon=True)
        reader.start()
        Console_Capture._write_all(writer, data)
        reader.join(30)
        self.assertEqual(bytes(received), data)

    @unittest.skipIf(sys.platform == "win32", "Windows hands a non-blocking pipe 4 KB at a time: too slow to finish in the time close allows")
    def test_a_capture_keeps_forwarding_into_a_non_blocking_stdout(self):
        # 30 writes of 100 lines each: the forward queue holds chunks (512 of them) and drops the newest when
        # a reader stalls, as it should, so the burst must not be many tiny writes for this to be about EAGAIN.
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "c = C.install('probe', sys.argv[2], True); "
            "[sys.stdout.write(''.join('line %05d ' % i + 'x' * 90 + '\\n' for i in range(n, n + 100))) "
            "for n in range(0, 3000, 100)]; "
            "print('END', flush=True)"
        )
        read_end, write_end = os.pipe()
        os.set_blocking(write_end, False)  # one pipe end, shared by every copy of it: the child's stdout is non-blocking too
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen(
                [sys.executable, "-c", program, str(SRC_DIR), directory], stdin=subprocess.DEVNULL,
                stdout=write_end, stderr=subprocess.DEVNULL, env=child_environment(),
                creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            os.close(write_end)
            received = bytearray()
            try:
                time.sleep(1.0)  # the reader is slow: the pipe fills, and the forwarder is told "try again"
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    chunk = os.read(read_end, 65536)
                    if not chunk:
                        break
                    received.extend(chunk)
                process.wait(timeout=30)
            finally:
                os.close(read_end)
                if process.poll() is None:
                    process.kill()
                    process.wait()
        lines = received.decode("utf-8").splitlines()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(lines[-1], "END")
        self.assertEqual(sum(line.startswith("line ") for line in lines), 3000)


class InheritedPipeTests(unittest.TestCase):
    """close() does not wait for a program this process started that holds descriptors 1 and 2.

    On Linux QtWebEngine's helper processes keep copies of Tools' capture pipes, so the readers, which
    waited for the pipes' end, made close() wait out its two seconds every time Tools closed.
    """

    PROGRAM = (
        "import json, subprocess, sys, time; sys.path.insert(0, sys.argv[1]); "
        "from utilities import Console_Capture as C; "
        "c = C.install('probe', sys.argv[2], False); "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdin=subprocess.DEVNULL); "
        "print('LAST LINE', flush=True); "
        "started = time.monotonic(); c.close(); seconds = time.monotonic() - started; "
        "child.kill(); child.wait(); "
        "json.dump({'seconds': seconds, 'log': open(c.log_path, encoding='utf-8').read()}, open(sys.argv[3], 'w'))"
    )

    @unittest.skipIf(sys.platform == "win32", "Windows reads its pipes to their end, as it always did")
    def test_close_ends_at_once_with_the_last_line_while_a_child_holds_the_pipes(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = os.path.join(directory, "result.json")
            done = subprocess.run(
                [sys.executable, "-c", self.PROGRAM, str(SRC_DIR), os.path.join(directory, "logs"), result_path],
                capture_output=True, stdin=subprocess.DEVNULL, env=child_environment(), timeout=120,
            )
            self.assertEqual(done.returncode, 0, done.stderr.decode("utf-8", "replace"))
            result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        self.assertLess(result["seconds"], 1.0)
        self.assertIn("LAST LINE", result["log"])


class HardExitTests(unittest.TestCase):
    """``hard_exit`` ends the process with os._exit only after what it printed has left the pipes.

    A hidden Viewer that cannot get an OpenGL context prints a marker line and exits at once; the MCP
    client reads that line to start the headless fallback. With the capture on, descriptors 1 and 2 are
    pipes that threads of the process empty, and a bare os._exit ended those threads first: an audit
    lost the line in 59 of 60 runs on Linux and 58 of 60 on Windows.
    """

    PROGRAM = (
        "import sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
        "{install}"
        "print('MARKER out ' + sys.argv[3], flush=True); "
        "print('MARKER err ' + sys.argv[3], file=sys.stderr, flush=True); "
        "{leave}"
    )
    EXIT_CODE = 75
    RUNS = 5  # the unfixed exit loses the line nearly every time; a few runs make a pass certain

    def run_child(self, directory, number, *, capture, leave="C.hard_exit(75)"):
        program = self.PROGRAM.format(
            install="C.install('probe', sys.argv[2]); " if capture else "", leave=leave
        )
        return subprocess.run(
            [sys.executable, "-c", program, str(SRC_DIR), os.path.join(directory, "logs"), str(number)],
            capture_output=True, stdin=subprocess.DEVNULL, env=child_environment(), timeout=120,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    def test_the_last_lines_before_a_hard_exit_reach_the_parent_and_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            for number in range(self.RUNS):
                with self.subTest(run=number):
                    done = self.run_child(directory, number, capture=True)
                    self.assertEqual(done.returncode, self.EXIT_CODE, done.stderr)
                    self.assertIn(f"MARKER out {number}".encode(), done.stdout)
                    self.assertIn(f"MARKER err {number}".encode(), done.stderr)
            logged = "".join(path.read_text(encoding="utf-8") for path in (Path(directory) / "logs").glob("probe-*.log"))
            for number in range(self.RUNS):
                self.assertIn(f"MARKER out {number}", logged)
                self.assertIn(f"MARKER err {number}", logged)

    def test_without_a_capture_it_flushes_the_buffered_streams_and_exits(self):
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "sys.stdout.write('buffered line'); sys.stderr.write('buffered error'); C.hard_exit(3)"
        )
        done = subprocess.run(
            [sys.executable, "-c", program, str(SRC_DIR)], capture_output=True, stdin=subprocess.DEVNULL,
            env=child_environment(), timeout=120, creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self.assertEqual(done.returncode, 3)
        self.assertEqual(done.stdout, b"buffered line")
        self.assertEqual(done.stderr, b"buffered error")

    def test_the_exit_comes_even_when_closing_the_capture_fails(self):
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "C.install('probe', sys.argv[2]); "
            "C._capture.close = lambda: 1 / 0; "
            "C.hard_exit(4)"
        )
        with tempfile.TemporaryDirectory() as directory:
            done = subprocess.run(
                [sys.executable, "-c", program, str(SRC_DIR), directory], capture_output=True,
                stdin=subprocess.DEVNULL, env=child_environment(), timeout=120,
                creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        self.assertEqual(done.returncode, 4)

    def test_it_is_the_only_hard_exit_in_the_program_and_the_hidden_viewers_fallback_uses_it(self):
        found = set()
        roots = [SRC_DIR]
        if (PROJECT_ROOT / "opt_vr" / "src").is_dir():
            roots.append(PROJECT_ROOT / "opt_vr" / "src")
        for root in roots:
            for path in root.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Attribute) and node.attr == "_exit" and getattr(node.value, "id", "") == "os":
                        found.add(path.relative_to(PROJECT_ROOT).as_posix())
        self.assertEqual(found, {"src/utilities/Console_Capture.py"})
        viewer = ast.parse((SRC_DIR / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(viewer)
                        if isinstance(node, ast.FunctionDef) and node.name == "_hidden_window_unavailable")
        calls = {getattr(call.func, "id", getattr(call.func, "attr", "")) for call in ast.walk(function)
                 if isinstance(call, ast.Call)}
        self.assertIn("hard_exit", calls)


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
        """An MCP client's launch monitor reads the Viewer's two pipes: the tee is unchanged."""
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

    def test_the_portal_notice_gnome_starts_print_is_left_out_with_its_neighbour_kept(self):
        """Through the real Qt: a category's warnings reach the handler as Qt sends them."""
        run = ProbeRun(self, "pipes", options=("qtportal",))
        self.assertEqual(run.code, 0, run)
        self.assertIsNotNone(run.result, "the probe died before its result")
        self.assertEqual(run.result["missing_in_ring"], [])
        stderr = run.stderr_bytes.decode("utf-8", "replace")
        for where, text in (("the ring", "\n".join(run.ring_texts)), ("the log", run.logged()), ("stderr", stderr)):
            with self.subTest(where=where):
                self.assertNotIn("already associated", text)
                self.assertIn("App info not found", text)  # another failure of the same registration shows
        shown = [text for text in run.ring_texts if "host portal" in text]
        self.assertEqual(len(shown), 1, shown)
        self.assertTrue(shown[0].startswith("[Qt warning] qt.qpa.services: Failed to register with host portal"), shown)

    def test_colours_reach_the_ring_but_neither_the_log_nor_a_pipe(self):
        run = ProbeRun(self, "pipes", options=("ansi",))
        self.assertEqual(run.code, 0, run)
        self.assertIsNotNone(run.result, "the probe died before its result")
        self.assertEqual(run.result["missing_in_ring"], [])
        # The question a command asks: yes under the capture, though both streams are pipes; no for a buffer.
        self.assertTrue(run.result["colours_enabled"])
        self.assertFalse(run.result["colours_enabled_in_a_buffer"])
        esc = chr(27)

        def ring_line(marker):
            return next(text for text in run.ring_texts if marker in text)

        # The Console window reads these.
        self.assertEqual(ring_line("ansi marker"), f"{esc}[31mSRC ansi marker{esc}[0m {esc}[38;2;1;2;3mdot{esc}[0m")
        self.assertEqual(ring_line("ansi stderr"), f"{esc}[1mSRC ansi stderr{esc}[0m")
        self.assertEqual(ring_line("ansi split"), f"SRC ansi split {esc}[31mcoloured{esc}[0m")
        log = run.logged()
        self.assertNotIn(esc, log)
        self.assertIn("SRC ansi marker dot", log)
        self.assertIn("SRC ansi split coloured", log)
        # Standard output and error are pipes here: what is passed on has no colour, even across a cut sequence.
        out, err = run.stdout_bytes.decode("utf-8"), run.stderr_bytes.decode("utf-8")
        self.assertNotIn(esc, out + err)
        self.assertIn("SRC ansi marker dot", out)
        self.assertIn("SRC ansi split coloured", out)
        self.assertIn("SRC ansi stderr", err)

    @unittest.skipUnless(os.name == "posix", "a pseudo-terminal is POSIX")
    def test_a_terminal_is_passed_the_colours_unchanged(self):
        import pty
        import select

        master, slave = pty.openpty()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        result = Path(directory.name) / "result.json"
        command = [sys.executable, "-u", str(PROBE), str(result), str(Path(directory.name) / "logs"), "auto", "ansi"]
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave,
                                   env=child_environment())
        os.close(slave)
        received = b""
        deadline = time.monotonic() + 60
        try:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([master], [], [], 0.5)
                if ready:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:  # the other end closed (Linux)
                        break
                    if not chunk:
                        break
                    received += chunk
                elif process.poll() is not None:
                    break
        finally:
            os.close(master)
            process.wait(timeout=30)
        self.assertEqual(process.returncode, 0)
        esc = chr(27).encode()
        self.assertIn(esc + b"[31mSRC ansi marker" + esc + b"[0m", received)
        self.assertIn(esc + b"[38;2;1;2;3mdot" + esc + b"[0m", received)
        self.assertIn(b"SRC ansi split " + esc + b"[31mcoloured" + esc + b"[0m", received)
        log = "".join(path.read_text(encoding="utf-8") for path in (Path(directory.name) / "logs").glob("probe-*.log"))
        self.assertNotIn(chr(27), log)  # the log has none whoever is looking

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

    def test_a_viewer_an_mcp_client_starts_keeps_its_tee_through_the_launch_monitor(self):
        """The monitor's stdout.log and stderr.log hold what the Viewer's two pipes got, as without the capture."""
        from desktop import Desktop_Launcher_Monitor

        program = (
            "import os, sys; sys.path.insert(0, sys.argv[1]); from utilities import Console_Capture as C; "
            "C.install('viewer', sys.argv[2]); print('python output'); os.write(1, b'native partial'); "
            "os.write(2, b'native error'); print('Viewer Ready', flush=True); sys.exit(3)"
        )
        start = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch"
            environment = dict(child_environment())
            environment[Desktop_Launcher_Monitor.SUPERVISED_DIRECTORY_VARIABLE] = str(launch)
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                Desktop_Launcher_Monitor.subprocess, "Popen",
                side_effect=lambda command, **kwargs: start(
                    [sys.executable, "-u", "-c", program, str(SRC_DIR), os.path.join(directory, "logs")], **kwargs),
            ):
                code = Desktop_Launcher_Monitor.run_monitor("session", Path(directory) / "state")
            out = (launch / "stdout.log").read_bytes()
            err = (launch / "stderr.log").read_bytes()
            self.assertEqual(code, 3, err)
            self.assertIn(b"python output", out)
            self.assertIn(b"native partial", out)
            self.assertIn(b"Viewer Ready", out)
            self.assertIn(b"native error", err)
            self.assertNotIn(b"native error", out)
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
