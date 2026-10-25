# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The one opener for files, folders and web pages (utilities.Desktop_Open).

On Linux and macOS the program that opens a target is started with /dev/null for all three
standard streams and a session of its own, so it never holds the pipes of the Console capture;
on Windows each kind of target goes where it always went (Qt, or the webbrowser module for the
Viewer's own pages). The program's call sites reach the desktop only through it.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import webbrowser

# Qt and the webbrowser module read the platform when they are first imported: not while a test says win32.
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tests import desktop_openers  # noqa: E402
from utilities import Desktop_Open  # noqa: E402

CREATE_NO_WINDOW = 0x08000000


def found(*names):
    """A stand-in shutil.which that knows only these programs."""
    return lambda name: f"/usr/bin/{name}" if name in names else None


class CommandTests(unittest.TestCase):
    def test_the_commands_are_those_of_the_programs_the_desktop_has(self):
        cases = (
            ("linux", ("xdg-open", "gio"), [["xdg-open", "T"], ["gio", "open", "T"]]),
            ("linux", ("gio",), [["gio", "open", "T"]]),
            ("linux", ("xdg-open",), [["xdg-open", "T"]]),
            ("linux", (), []),
            ("darwin", ("open", "xdg-open"), [["open", "T"]]),
            ("darwin", (), []),
        )
        for platform_name, installed, expected in cases:
            with self.subTest(platform=platform_name, installed=installed):
                self.assertEqual(Desktop_Open._desktop_commands("T", platform_name, found(*installed)), expected)

    def test_the_next_program_is_tried_when_one_does_not_start(self):
        with mock.patch.object(Desktop_Open, "_start", side_effect=[False, True]) as start, mock.patch.object(
            Desktop_Open.shutil, "which", side_effect=found("xdg-open", "gio")
        ):
            self.assertTrue(Desktop_Open._open_with_the_desktop("/a/b", browser=False, platform_name="linux"))
        self.assertEqual([call.args[0] for call in start.call_args_list], [["xdg-open", "/a/b"], ["gio", "open", "/a/b"]])

    def test_nothing_opens_when_no_program_starts_or_none_is_there(self):
        with mock.patch.object(Desktop_Open, "_start", return_value=False), mock.patch.object(
            Desktop_Open.shutil, "which", side_effect=found("xdg-open")
        ):
            self.assertFalse(Desktop_Open._open_with_the_desktop("/a/b", browser=False, platform_name="linux"))
        with mock.patch.object(Desktop_Open, "_start") as start, mock.patch.object(
            Desktop_Open.shutil, "which", side_effect=found()
        ):
            self.assertFalse(Desktop_Open._open_with_the_desktop("/a/b", browser=False, platform_name="linux"))
        start.assert_not_called()

    def test_a_target_that_would_read_as_an_option_is_not_handed_to_a_program(self):
        with mock.patch.object(Desktop_Open, "_start") as start, mock.patch.object(
            Desktop_Open.shutil, "which", side_effect=found("xdg-open")
        ):
            self.assertFalse(Desktop_Open._open_with_the_desktop("--version", browser=True, platform_name="linux"))
        start.assert_not_called()

    def test_a_relative_path_is_made_absolute_before_a_program_gets_it(self):
        with mock.patch.object(Desktop_Open.sys, "platform", "linux"), mock.patch.object(
            Desktop_Open, "_start", return_value=True
        ) as start, mock.patch.object(Desktop_Open.shutil, "which", side_effect=found("xdg-open")):
            self.assertTrue(Desktop_Open.open_path(Path("-odd") / "folder"))
        self.assertEqual(start.call_args.args[0], ["xdg-open", os.path.abspath(os.path.join("-odd", "folder"))])


class BrowserVariableTests(unittest.TestCase):
    URL = "http://127.0.0.1:1/page"

    def commands(self, value):
        return Desktop_Open._browser_commands(self.URL, {"BROWSER": value})

    def test_the_variable_is_read_as_the_webbrowser_module_reads_it(self):
        value = os.pathsep.join(["one --new-tab %s", "two", "", "three '%s' --private", "four 'unclosed"])
        self.assertEqual(self.commands(value), [
            ["one", "--new-tab", self.URL],
            ["two", self.URL],
            ["three", self.URL, "--private"],
        ])
        self.assertEqual(Desktop_Open._browser_commands(self.URL, {}), [])
        self.assertEqual(self.commands("   "), [])

    def test_a_web_address_tries_the_users_browser_first_and_a_path_never_does(self):
        calls = []

        def start(command):
            calls.append(command)
            return command[0] == "xdg-open"

        with mock.patch.dict(os.environ, {"BROWSER": "mybrowser %s"}), mock.patch.object(
            Desktop_Open, "_start", side_effect=start
        ), mock.patch.object(Desktop_Open.shutil, "which", side_effect=found("xdg-open")):
            self.assertTrue(Desktop_Open._open_with_the_desktop(self.URL, browser=True, platform_name="linux"))
            self.assertEqual(calls, [["mybrowser", self.URL], ["xdg-open", self.URL]])
            del calls[:]
            self.assertTrue(Desktop_Open._open_with_the_desktop("/a/b", browser=False, platform_name="linux"))
            self.assertEqual(calls, [["xdg-open", "/a/b"]])


class WindowsTests(unittest.TestCase):
    """Windows keeps what it always did, so these run on any host with the platform named win32."""

    def setUp(self):
        self.enterContext(mock.patch.object(Desktop_Open.sys, "platform", "win32"))
        self.start = self.enterContext(mock.patch.object(Desktop_Open, "_start"))
        self.qt = self.enterContext(mock.patch.object(QDesktopServices, "openUrl", return_value=True))
        self.browser = self.enterContext(mock.patch.object(webbrowser, "open", return_value=True))

    def test_a_file_or_folder_goes_to_qt_as_a_local_file_url(self):
        self.assertTrue(Desktop_Open.open_path(Path("C:/data/folder")))
        self.qt.assert_called_once_with(QUrl.fromLocalFile(str(Path("C:/data/folder"))))

    def test_a_web_address_goes_to_qt_and_the_viewers_pages_to_the_webbrowser_module(self):
        self.assertTrue(Desktop_Open.open_url("https://example.org/licence"))
        self.qt.assert_called_once_with(QUrl("https://example.org/licence"))
        self.browser.assert_not_called()
        self.qt.reset_mock()
        self.assertTrue(Desktop_Open.open_url("http://127.0.0.1:1/meta.html", new_tab=True))
        self.browser.assert_called_once_with("http://127.0.0.1:1/meta.html", new=2)
        self.qt.assert_not_called()

    def test_the_answer_is_the_one_the_backend_gave(self):
        self.qt.return_value = False
        self.browser.return_value = None
        self.assertIs(Desktop_Open.open_path("C:/x"), False)
        self.assertIs(Desktop_Open.open_url("https://example.org"), False)
        self.assertIs(Desktop_Open.open_url("https://example.org", new_tab=True), False)

    def test_no_program_is_started_by_the_opener_itself(self):
        Desktop_Open.open_path("C:/x")
        Desktop_Open.open_url("https://example.org", new_tab=True)
        self.start.assert_not_called()


class StartTests(unittest.TestCase):
    """What the one function that starts a program asks the system for."""

    def start(self):
        start = desktop_openers.real_opener_start()
        self.assertIsNotNone(start, "the test guard keeps the real function")
        return start

    def test_the_program_gets_no_streams_a_session_of_its_own_and_is_collected(self):
        with mock.patch.object(Desktop_Open.subprocess, "Popen") as popen, mock.patch.object(
            Desktop_Open, "_reap"
        ) as reap:
            self.assertTrue(self.start()(["xdg-open", "/a/b"]))
        popen.assert_called_once_with(
            ["xdg-open", "/a/b"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        reap.assert_called_once_with(popen.return_value)

    def test_a_program_that_cannot_start_is_not_an_error(self):
        for error in (FileNotFoundError("gone"), PermissionError("no"), ValueError("embedded null")):
            with self.subTest(error=type(error).__name__), mock.patch.object(
                Desktop_Open.subprocess, "Popen", side_effect=error
            ), mock.patch.object(Desktop_Open, "_reap") as reap:
                self.assertFalse(self.start()(["xdg-open", "/a/b"]))
            reap.assert_not_called()


# A stand-in opener (a desktop's xdg-open, a user's browser): it records what its streams and session are, then
# stays alive, as a file manager or a browser does, until the test lets it go.
_RECORDING_OPENER = """\
import json, os, stat, sys, time

null = os.stat(os.devnull)
streams = {}
for number in (0, 1, 2):
    info = os.fstat(number)
    streams[number] = {"null": os.path.samestat(info, null), "pipe": stat.S_ISFIFO(info.st_mode)}
entry = {"program": os.path.basename(sys.argv[1]), "arguments": sys.argv[2:], "streams": streams,
         "own_session": os.getsid(0) == os.getpid()}
with open(os.environ["FAKE_OPENER_RECORD"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry) + "\\n")
release = os.environ["FAKE_OPENER_RELEASE"]
deadline = time.monotonic() + 60
while not os.path.exists(release) and time.monotonic() < deadline:
    time.sleep(0.05)
"""

# A window that captures its output (pipes over descriptors 1 and 2) opens a folder and a web page, then ends.
_CAPTURING_WINDOW = """\
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from utilities import Console_Capture, Desktop_Open

record = os.environ["FAKE_OPENER_RECORD"]
capture = Console_Capture.install("probe", sys.argv[2], True)
started = [Desktop_Open.open_path(sys.argv[2]), Desktop_Open.open_url("https://example.org/page", new_tab=True)]
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    if os.path.exists(record) and len(open(record, encoding="utf-8").read().splitlines()) >= 2:
        break
    time.sleep(0.02)
began = time.monotonic()
capture.close()  # as at the end of the window's process, with both programs still running
print(json.dumps({"started": started, "close_seconds": time.monotonic() - began}))
"""


@unittest.skipIf(sys.platform == "win32", "Windows opens through Qt; this is what Linux and macOS start")
class OpenedProgramTests(unittest.TestCase):
    """A folder or a page opened by a window that captures its output does not hold the capture's pipes.

    The audit's finding for the other long-lived children: webbrowser.open and Qt's QDesktopServices
    started the opener (and so a file manager or a browser) with the window's own descriptors, which
    are the capture's pipes: closing the capture waited two seconds for them at exit, and the program
    died of SIGPIPE on its next write once the window had gone.
    """

    def test_the_opened_program_gets_dev_null_and_a_session_of_its_own(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_dir = root / "bin"
        bin_dir.mkdir()
        script = root / "recording_opener.py"
        script.write_text(_RECORDING_OPENER, encoding="utf-8")
        # macOS's own opener is `open`: a stand-in goes first on PATH there, so the real one never runs.
        for name in ("xdg-open", "open", "mybrowser"):
            program = bin_dir / name
            program.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$0" "$@"\n', encoding="utf-8"
            )
            program.chmod(0o755)
        record, release = root / "record.jsonl", root / "release"
        self.addCleanup(release.touch)  # a program that outlives a failed test ends at once
        environment = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
        environment.update(
            PATH=f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}",
            BROWSER="mybrowser %s",
            FAKE_OPENER_RECORD=str(record),
            FAKE_OPENER_RELEASE=str(release),
        )
        done = subprocess.run(
            [sys.executable, "-c", _CAPTURING_WINDOW, str(SRC_DIR), str(root)],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, env=environment, timeout=120,
        )
        release.touch()
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(report["started"], [True, True])
        entries = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]
        by_program = {entry["program"]: entry for entry in entries}
        # The folder went to the desktop's opener, the page to the user's own browser first.
        desktop_opener = "open" if sys.platform == "darwin" else "xdg-open"
        self.assertEqual(sorted(by_program), sorted(["mybrowser", desktop_opener]))
        self.assertEqual(by_program[desktop_opener]["arguments"], [str(root)])
        self.assertEqual(by_program["mybrowser"]["arguments"], ["https://example.org/page"])
        for entry in entries:
            for number in ("0", "1", "2"):
                with self.subTest(program=entry["program"], descriptor=number):
                    self.assertTrue(entry["streams"][number]["null"], "not /dev/null")
                    self.assertFalse(entry["streams"][number]["pipe"])
            self.assertTrue(entry["own_session"], f"{entry['program']} shares this window's session")
        self.assertLess(report["close_seconds"], 1.0, "closing the capture waited for an opened program")


class RoutingTests(unittest.TestCase):
    """The program asks the desktop to open something in one place only."""

    # Splash.py is the Windows launcher's own process (no capture, nothing to inherit); Desktop_App's
    # os.startfile is the Windows fallback of open_in_file_manager.
    ALLOWED = {
        "src/utilities/Desktop_Open.py": "the opener",
        "src/desktop/Splash.py": "the launcher's start-up window, on Windows",
        "src/desktop/Desktop_App.py": "os.startfile, Windows' last resort",
    }

    def calls(self, root):
        found = {}
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                owner = getattr(node.value, "id", "")
                if node.attr in {"openUrl", "startDetached", "startfile"} or (
                    owner == "webbrowser" and node.attr.startswith("open")
                ):
                    found.setdefault(path.relative_to(PROJECT_ROOT).as_posix(), []).append(f"{owner}.{node.attr}")
        return found

    def test_only_the_opener_and_its_documented_exceptions_reach_the_desktop_directly(self):
        roots = [SRC_DIR]
        if (PROJECT_ROOT / "opt_vr" / "src").is_dir():
            roots.append(PROJECT_ROOT / "opt_vr" / "src")
        offenders = {}
        for root in roots:
            for name, calls in self.calls(root).items():
                if name not in self.ALLOWED:
                    offenders[name] = calls
        self.assertEqual(offenders, {}, "route these through utilities.Desktop_Open")


if __name__ == "__main__":
    unittest.main()
