# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The first stage of a Windows launch (desktop.Desktop_Launcher): it finds an open window and ends quietly, holds a lock so a double-click starts one launch, shows the splash and the dependency check's lines live, hands over to the visible console when the environment needs setup, starts the monitor and closes the splash when the window is ready, turns the splash into an error view when the window ends first, passes a Viewer session its arguments, and with --console runs the old launch with no splash. The names it computes match the ones the windows use. The kinds are a table that a program beside this one adds to: a kind has its own steps (a step that fails is a warning that opens the splash's details and lets the start go on), batch file, single-instance key, taskbar identity, catalogs and monitor."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtNetwork import QLocalServer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop import Desktop_App, Desktop_Launcher, Splash  # noqa: E402
from desktop.Desktop_Launcher import Launch, LaunchKind, LaunchStep  # noqa: E402

COPIED = (
    "src/desktop/__init__.py", "src/desktop/Desktop_Launcher_Monitor.py", "src/utilities/__init__.py",
    "src/utilities/Localization.py",
)

# The monitor of a test tree gets this in place of utilities/Terminal_Launcher.py: a crash it reports the old way
# (or a launcher mutated to ask for that) must leave a record in the state folder and never open a terminal on
# the screen of whoever runs the tests.
STUB_TERMINAL_LAUNCHER = textwrap.dedent(
    '''
    import json
    import os
    from enum import Enum

    class HoldMode(str, Enum):
        NEVER = "never"
        ALWAYS = "always"
        ON_ERROR = "on_error"

    class TerminalUnavailableError(RuntimeError):
        pass

    def launch_in_terminal(command, *, cwd, env=None, hold=HoldMode.NEVER, title=None, platform_name=None, **ignored):
        with open(os.path.join(str(cwd), "terminal.requested"), "w", encoding="utf-8") as handle:
            json.dump({"title": title, "hold": str(hold.value)}, handle)
    '''
)

# What a fake window does, as a script: report, print, create gui.ready or fail. The monitor passes
# SSN_GUI_READY_FILE and SSN_STARTUP_STATUS_FILE, as it does to the real ones.
FAKE_WINDOW = textwrap.dedent(
    '''
    import json, os, sys, time
    log = os.path.join(os.path.dirname(os.environ["SSN_GUI_READY_FILE"]), "argv.json")
    with open(log, "w") as handle:
        json.dump(sys.argv[1:], handle)
    def report(text, fraction=None):
        with open(os.environ["SSN_STARTUP_STATUS_FILE"], "ab") as handle:
            handle.write((json.dumps({"text": text, "fraction": fraction, "values": {}}) + "\\n").encode())
    mode = os.environ.get("FAKE_WINDOW", "ready")
    print("fake window starts", flush=True)
    report("Loading PyTorch\u2026", 0.45)
    if mode == "fail":
        print("Traceback (most recent call last):", file=sys.stderr, flush=True)
        print("ValueError: the window broke", file=sys.stderr, flush=True)
        sys.exit(1)
    if mode == "handover":
        print("already open", flush=True)
        sys.exit(0)
    if mode == "hang":
        time.sleep(60)
    report("Building the window\u2026", 0.8)
    time.sleep(float(os.environ.get("FAKE_WINDOW_DELAY", "0.2")))
    with open(os.environ["SSN_GUI_READY_FILE"], "w") as handle:
        handle.write("ready\\n")
    time.sleep(float(os.environ.get("FAKE_WINDOW_STAY", "0.5")))
    sys.exit(int(os.environ.get("FAKE_WINDOW_EXIT", "0")))
    '''
)
FAKE_CHECK = textwrap.dedent(
    '''
    import os, sys
    print("Managed Python: fake", flush=True)
    print("Detected: a fake GPU", flush=True)
    print("Dependency environment is ready (fake).", flush=True)
    sys.exit(int(os.environ.get("FAKE_CHECK_EXIT", "0")))
    '''
)


def make_tree(test):
    """A project folder of fake windows and a fake dependency check, with the real monitor and launcher modules."""
    directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    test.addCleanup(directory.cleanup)
    root = Path(directory.name)
    for relative in COPIED:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, root / relative)
    (root / "src" / "utilities" / "Terminal_Launcher.py").write_text(STUB_TERMINAL_LAUNCHER, encoding="utf-8")
    for name in ("EMAPSSN_Config.py", "EMAPSSN_Tools.py", "EMAPSSN_Viewer.py"):
        (root / "src" / name).write_text(FAKE_WINDOW, encoding="utf-8")
    (root / "src" / "Install_Dependencies.py").write_text(FAKE_CHECK, encoding="utf-8")
    (root / "src" / "bin").mkdir()
    (root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.bat").write_text("@echo off\r\n", encoding="utf-8")
    return root


class PureHelperTests(unittest.TestCase):
    def test_the_single_instance_names_are_the_windows_own(self):
        with tempfile.TemporaryDirectory() as home, tempfile.TemporaryDirectory() as other:
            for application_id in ("SSN_Config", "SSN_Tools", "SSN Odd/Name"):
                for root in (PROJECT_ROOT, Path(other)):
                    with self.subTest(application_id=application_id, root=str(root)):
                        expected = Desktop_App._single_instance_server_name(application_id, root)
                        self.assertEqual(Desktop_Launcher.single_instance_server_name(application_id, root), expected)

    def test_the_names_depend_on_the_user_and_the_checkout(self):
        names = {
            Desktop_Launcher.single_instance_server_name("SSN_Config", "/a", home="/h1"),
            Desktop_Launcher.single_instance_server_name("SSN_Config", "/b", home="/h1"),
            Desktop_Launcher.single_instance_server_name("SSN_Config", "/a", home="/h2"),
            Desktop_Launcher.single_instance_server_name("SSN_Tools", "/a", home="/h1"),
        }
        self.assertEqual(len(names), 4)

    def test_the_kinds_are_the_batch_files_and_the_window_it_starts_is_the_monitors(self):
        kinds = Desktop_Launcher.LAUNCH_KINDS
        self.assertEqual(set(kinds), {"viewer", "tools", "session"})
        from desktop import Desktop_Launcher_Monitor

        self.assertEqual(set(kinds), set(Desktop_Launcher_Monitor.APP_SCRIPTS))
        self.assertEqual({kind.splash_kind for kind in kinds.values()}, set(Splash.KINDS))
        self.assertEqual(kinds["viewer"].splash_kind, "config")  # "viewer" is Config, as in the batch file
        for name, kind in kinds.items():
            with self.subTest(kind=name):
                self.assertEqual(kind.name, name)
                self.assertEqual(kind.steps, ())  # the windows of this program add no steps of their own
                self.assertEqual(kind.batch_file, Desktop_Launcher.BATCH_FILE)
                self.assertIsNone(kind.app_user_model_id)
                self.assertEqual(kind.catalogs, ())
        # The single-instance channels are the windows' own; a session may be open many times, and runs no check.
        self.assertEqual((kinds["viewer"].application_id, kinds["tools"].application_id, kinds["session"].application_id),
                         ("SSN_Config", "SSN_Tools", None))
        self.assertEqual((kinds["viewer"].check, kinds["tools"].check, kinds["session"].check), (True, True, False))

    def test_this_module_names_no_program_beside_it(self):
        source = Path(Desktop_Launcher.__file__).read_text(encoding="utf-8")
        for name in ("EMAPSSN_Config_VR", "Detect_GPU_VR", "Launcher_VR", '"vr"'):
            self.assertNotIn(name, source)

    def test_the_check_is_the_one_the_batch_file_runs(self):
        command = Desktop_Launcher.check_command("C:/p", "C:/p/.venv/Scripts/python.exe", "uv")
        self.assertEqual(command[1], "-u")
        self.assertEqual(command[2:5], [str(Path("C:/p/src/Install_Dependencies.py")), "--check-only", "--uv-executable"])
        self.assertEqual(command[5:], ["uv", "--venv", str(Path("C:/p/.venv"))])
        batch = (PROJECT_ROOT / "src" / "bin" / "EMAPSSN.bat").read_text(encoding="utf-8")
        self.assertIn('src\\Install_Dependencies.py --check-only --uv-executable "!UV_EXE!" --venv .venv', batch)

    def test_uv_is_found_on_the_path_or_in_the_users_local_bin(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertIsNone(Desktop_Launcher.find_uv({"PATH": str(home / "nothing"), "USERPROFILE": str(home)}))
            (home / ".local" / "bin").mkdir(parents=True)
            (home / ".local" / "bin" / "uv.exe").write_text("x")
            self.assertEqual(Desktop_Launcher.find_uv({"PATH": str(home / "nothing"), "USERPROFILE": str(home)}),
                             str(home / ".local" / "bin" / "uv.exe"))
            tools = home / "tools"
            tools.mkdir()
            program = tools / ("uv.exe" if sys.platform == "win32" else "uv")
            program.write_text("x")
            program.chmod(0o755)
            self.assertEqual(Path(Desktop_Launcher.find_uv({"PATH": str(tools), "USERPROFILE": str(home)})), program)

    def test_the_visible_console_launch_is_the_batch_file_with_its_kind(self):
        command = Desktop_Launcher.visible_console_command("C:/p", "tools")
        self.assertEqual(command[1:3], ["/d", "/c"])
        self.assertEqual(command[3], str(Path("C:/p/src/bin/EMAPSSN_Desktop_Launcher.bat")))
        self.assertEqual(command[4], "tools")

    @unittest.skipUnless(sys.platform == "win32", "a new console window is a Windows thing")
    def test_the_old_launch_gets_a_console_window_of_its_own(self):
        popen = mock.Mock()
        Desktop_Launcher.open_visible_console("C:/p", "viewer", popen=popen)
        self.assertEqual(popen.call_args.kwargs["creationflags"], Desktop_Launcher.CREATE_NEW_CONSOLE)
        self.assertEqual(popen.call_args.kwargs["cwd"], "C:/p")

    def test_the_stub_exits_with_a_code_that_means_use_the_console(self):
        self.assertEqual(Desktop_Launcher.EXIT_FALLBACK, 3)

    def test_the_lines_of_a_process_are_split_on_newlines_and_a_steps_keep_their_indent(self):
        class Process:
            def __init__(self, data):
                self.data = data

            def readAllStandardOutput(self):
                data, self.data = self.data, b""
                return data

        for keep, expected in ((True, ["  [ok] a", "       reason with é"]), (False, ["[ok] a", "reason with é"])):
            with self.subTest(keep_indent=keep):
                buffer = bytearray()
                process = Process("  [ok] a\r\n       reason with é\n\n   \npart".encode("utf-8"))
                self.assertEqual(Desktop_Launcher._read_lines(process, buffer, keep_indent=keep), expected)
                self.assertEqual(bytes(buffer), b"part")  # a line with no newline yet waits for it
                process.data = b"ial\n"
                self.assertEqual(Desktop_Launcher._read_lines(process, buffer, keep_indent=keep), ["partial"])
                process.data = b"the last, with no newline"
                self.assertEqual(Desktop_Launcher._read_lines(process, buffer, keep_indent=keep), [])
                self.assertEqual(Desktop_Launcher._read_lines(process, buffer, final=True, keep_indent=keep),
                                 ["the last, with no newline"])
                self.assertEqual(bytes(buffer), b"")

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            Launch("elsewhere", "C:/p")
        with self.assertRaises(SystemExit), mock.patch.object(sys, "stderr"):
            Desktop_Launcher.main(["elsewhere"])

    def test_a_launcher_script_gets_the_stubs_fallback_when_main_fails_and_a_usage_error_passes_through(self):
        with mock.patch.object(Desktop_Launcher, "main", side_effect=RuntimeError("boom")), mock.patch.object(
            Desktop_Launcher.sys, "stderr"
        ):
            self.assertEqual(Desktop_Launcher.entry(["viewer"]), Desktop_Launcher.EXIT_FALLBACK)
        with mock.patch.object(Desktop_Launcher, "main", return_value=0) as main:
            self.assertEqual(Desktop_Launcher.entry(["tools"]), 0)
        main.assert_called_once_with(["tools"])
        with self.assertRaises(SystemExit):
            with mock.patch.object(sys, "stderr"):
                Desktop_Launcher.entry(["elsewhere"])

    @unittest.skipUnless(sys.platform == "win32", "taskbar identities are a Windows thing")
    def test_the_process_takes_the_taskbar_identity_it_is_given_before_it_has_a_window(self):
        program = (
            "import ctypes, sys; sys.path.insert(0, sys.argv[1]); from desktop import Desktop_Launcher; "
            "print(Desktop_Launcher.set_process_app_user_model_id('ca.test.one')); "
            "p = ctypes.c_wchar_p(); ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(p)); print(p.value)"
        )
        done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=120,
                              creationflags=Desktop_Launcher.CREATE_NO_WINDOW, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
        self.assertEqual(done.stdout.split(), ["True", "ca.test.one"], done.stderr)
        with mock.patch.object(Desktop_Launcher.sys, "platform", "linux"):
            self.assertFalse(Desktop_Launcher.set_process_app_user_model_id("x"))


class LockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QApplication([])

    def test_a_second_launch_of_the_same_window_does_not_get_the_lock(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            first = Desktop_Launcher.acquire_launch_lock(root, "viewer")
            self.assertIsNotNone(first)
            self.assertIsNone(Desktop_Launcher.acquire_launch_lock(root, "viewer"))
            other = Desktop_Launcher.acquire_launch_lock(root, "tools")  # each window has its own
            self.assertIsNotNone(other)
            first.unlock()
            again = Desktop_Launcher.acquire_launch_lock(root, "viewer")
            self.assertIsNotNone(again)
            again.unlock()
            other.unlock()

    def test_a_lock_whose_launcher_has_gone_is_taken_over(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
            Desktop_Launcher.lock_path(root, "viewer").parent.mkdir(parents=True)
            holder = subprocess.Popen([sys.executable, "-c", (
                "import sys; sys.path.insert(0, sys.argv[1]); from PySide6.QtCore import QCoreApplication, QLockFile; "
                "a = QCoreApplication([]); l = QLockFile(sys.argv[2]); l.setStaleLockTime(0); "
                "assert l.tryLock(0); print('held', flush=True); sys.stdin.readline()"
            ), str(SRC), str(Desktop_Launcher.lock_path(root, "viewer"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, creationflags=0x08000000 if sys.platform == "win32" else 0)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "held")
                self.assertIsNone(Desktop_Launcher.acquire_launch_lock(root, "viewer"))  # alive: still held
                # Crashed without unlocking. (A venv's python.exe starts the real interpreter as its child, which
                # is the process that holds the lock: both go.)
                import psutil

                for child in psutil.Process(holder.pid).children(recursive=True):
                    child.kill()
                holder.kill()
                holder.wait(timeout=20)
                time.sleep(0.3)
                lock = Desktop_Launcher.acquire_launch_lock(root, "viewer")
                self.assertIsNotNone(lock)
                lock.unlock()
            finally:
                if holder.poll() is None:
                    holder.kill()
                holder.communicate()


class LaunchCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        Splash.load_fonts(None)

    def setUp(self):
        self.root = make_tree(self)
        patcher = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.root / "app_settings.json")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        import psutil

        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                if str(self.root) in " ".join(process.info["cmdline"] or []):
                    process.kill()
            except (psutil.Error, OSError):
                pass

    def launch(self, kind="viewer", *, environment=None, **options):
        options.setdefault("show", False)
        options.setdefault("uv", "uv")
        launch = Launch(kind, self.root, python=sys.executable, popen=mock.Mock(name="popen"), **options)
        self.codes = []
        launch.finished.connect(self.codes.append)
        self.addCleanup(lambda: launch.splash is not None and launch.splash.deleteLater())
        patcher = mock.patch.dict(os.environ, environment or {})
        patcher.start()
        self.addCleanup(patcher.stop)
        return launch

    def run_launch(self, launch, seconds=60, until=None):
        launch.begin()
        deadline = time.monotonic() + seconds
        while not self.codes and time.monotonic() < deadline:
            if until is not None and until():
                break
            self.app.processEvents()
            time.sleep(0.01)
        return self.codes[0] if self.codes else None

    def state_dirs(self):
        return sorted((self.root / "temp").glob("viewer_*")) + sorted((self.root / "temp").glob("tools_*")) \
            + sorted((self.root / "temp").glob("session_*"))


class WindowStartTests(LaunchCase):
    def test_a_good_start_shows_the_check_then_what_the_window_reports_and_closes_on_ready(self):
        launch = self.launch("viewer")
        seen = []
        launch.splash = None
        original = Launch.open_splash

        def watching(self):
            original(self)
            shown = self.splash.show_text
            self.splash.show_text = lambda text, fraction=None: (seen.append(text), shown(text, fraction))

        with mock.patch.object(Launch, "open_splash", watching):
            code = self.run_launch(launch)
        self.assertEqual(code, 0)
        self.assertEqual(seen[:3], ["Managed Python: fake", "Detected: a fake GPU", "Dependency environment is ready (fake)."])
        self.assertIn("Loading PyTorch…", seen)
        self.assertIn("Building the window…", seen)
        self.assertLess(seen.index("Dependency environment is ready (fake)."), seen.index("Loading PyTorch…"))
        self.assertLess(seen.index("Loading PyTorch…"), seen.index("Building the window…"))
        for step in ("splash_shown", "check_started", "check_finished", "monitor_started", "gui_ready", "finished"):
            self.assertIn(step, launch.marks)
        self.assertLess(launch.marks["splash_shown"], launch.marks["check_finished"])
        self.assertLess(launch.marks["check_finished"], launch.marks["gui_ready"])
        self.assertFalse(launch.splash.error)
        self.assertTrue(launch.check_lines)

    def test_the_window_gets_the_files_it_reports_through_and_the_monitor_is_told_the_launcher_is_done(self):
        launch = self.launch("tools", environment={"FAKE_WINDOW_STAY": "4"})
        self.assertEqual(self.run_launch(launch), 0)
        state = launch.state_dir
        self.assertEqual(state.parent, self.root / "temp")
        self.assertTrue(state.name.startswith("tools_"))
        # The window is still open, so the monitor is waiting: the launcher has left the word it is done, which is
        # what lets the monitor report a crash of the open window (it reports none before that).
        self.assertTrue((state / "terminal.dismissed").is_file())
        # The monitor removes the state folder once the window has ended well and the launcher said it was done.
        deadline = time.monotonic() + 30
        while state.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertFalse(state.exists(), "the monitor left the state folder")
        # A normal start and end leave nothing in temp/ (no state folder, no lock file).
        self.assertEqual(sorted(path.name for path in (self.root / "temp").iterdir()), [])

    def test_the_monitor_is_asked_to_report_a_crash_of_the_open_window_in_a_dialog_not_a_terminal(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.1"})
        real = Desktop_Launcher.Monitor.launch_detached_monitor
        with mock.patch.object(Desktop_Launcher.Monitor, "launch_detached_monitor", wraps=real) as started:
            self.assertEqual(self.run_launch(launch), 0)
        started.assert_called_once()
        self.assertEqual(started.call_args.kwargs["failure_report"], "dialog")
        self.assertEqual(started.call_args.args[0], "viewer")

    def test_a_crash_after_the_window_opened_is_told_in_a_dialog_and_asks_for_no_terminal(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.3", "FAKE_WINDOW_EXIT": "1"})
        self.assertEqual(self.run_launch(launch), 0)
        state = launch.state_dir
        deadline = time.monotonic() + 30
        while not (state / "application.exit").is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual((state / "application.exit").read_text(encoding="utf-8").strip(), "1")
        time.sleep(2.0)  # the monitor has seen the exit and reports it
        self.assertFalse((state / "terminal.requested").exists(), "the monitor asked for a terminal")
        self.assertIsNone(launch.monitor.poll(), "the monitor is meant to be waiting on its dialog (offscreen here)")

    def test_a_window_that_fails_before_it_is_ready_turns_the_splash_into_an_error_view(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW": "fail"})
        launch.begin()
        deadline = time.monotonic() + 60
        while not (launch.splash and launch.splash.error) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(launch.splash.error)
        self.assertTrue(launch.splash.details_open, "the section did not open by itself")
        text = launch.splash.log_view.toPlainText()
        self.assertIn("Managed Python: fake", text)  # the dependency check's lines are in it too, first
        self.assertLess(text.index("Managed Python: fake"), text.index("fake window starts"))
        self.assertIn("fake window starts", text)
        self.assertIn("ValueError: the window broke", text)
        self.assertEqual(launch.splash.details.kind_of(text.splitlines().index("ValueError: the window broke")), "error")
        self.assertEqual(launch.splash.log_path, str(launch.state_dir / "application.log"))
        self.assertTrue(launch.splash.log_button.isEnabled())
        self.assertEqual(self.codes, [], "the launch ended before the error was read")
        launch.splash.close_button.click()  # read it: the launch is over, handled
        self.assertEqual(self.codes, [0])
        self.assertTrue((launch.state_dir / "application.log").is_file(), "the log is kept")
        self.assertEqual(launch.marks["failure_code"], 1)

    def test_a_window_that_ends_well_without_opening_closes_the_splash_quietly(self):
        # The window found another one open and handed over to it.
        launch = self.launch("viewer", environment={"FAKE_WINDOW": "handover"})
        self.assertEqual(self.run_launch(launch), 0)
        self.assertFalse(launch.splash.error)
        self.assertEqual(launch.marks["exit_code"], 0)
        self.assertNotIn("gui_ready", launch.marks)

    def test_a_window_that_never_opens_ends_in_the_error_view_after_the_wait(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW": "hang"}, timeout=1.5)
        launch.begin()
        deadline = time.monotonic() + 60
        while not launch.splash.error and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(launch.splash.error)
        self.assertIn("did not open a window within", launch.splash.error_title)
        self.assertEqual(launch.marks["failure_code"], Desktop_Launcher.Monitor.STARTUP_TIMEOUT)
        launch.splash.close_button.click()


class SectionTests(LaunchCase):
    """The splash's details: the start's output, and a splash that is open at ready staying until Close."""

    def run_until(self, launch, condition, seconds=60, open_section=True):
        launch.begin()
        if open_section:
            launch.splash.set_details_open(True)
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(condition(), "the condition was not met")

    def test_the_section_holds_the_check_lines_then_what_the_window_printed_as_it_started(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "5"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)
        text = launch.splash.details.toPlainText()
        self.assertEqual(text.splitlines()[:3], ["Managed Python: fake", "Detected: a fake GPU",
                                                 "Dependency environment is ready (fake)."])
        self.assertIn("fake window starts", text)
        self.assertLess(text.index("Dependency environment is ready (fake)."), text.index("fake window starts"))
        self.assertEqual(launch.splash.log_path, str(launch.state_dir / "application.log"))
        launch.splash.close_button.click()

    def test_open_at_ready_the_splash_says_ready_and_stays_until_close_is_pressed(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "8"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)
        for _ in range(60):  # a second: nothing closes it
            self.app.processEvents()
            time.sleep(0.02)
        splash = launch.splash
        self.assertEqual(self.codes, [], "the launch ended on its own")
        self.assertTrue(splash.isVisible())
        self.assertTrue(splash.ready)
        self.assertEqual(splash.status, "Ready")
        self.assertIn("ready_kept_open", launch.marks)
        self.assertNotIn("finished", launch.marks)
        # The monitor was told the launcher is done, as it is when the splash closes at once.
        self.assertTrue((launch.state_dir / "terminal.dismissed").is_file())
        self.assertTrue(splash.close_button.isVisibleTo(splash))
        splash.close_button.click()
        self.assertEqual(self.codes, [0])
        self.assertIn("finished", launch.marks)

    def test_collapsed_at_ready_it_closes_as_it_always_has(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        self.assertNotIn("ready_kept_open", launch.marks)
        self.assertFalse(launch.splash.ready)

    def test_opened_and_closed_again_before_ready_it_still_closes_at_ready(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.1", "FAKE_WINDOW_DELAY": "1.5"})
        launch.begin()
        launch.splash.set_details_open(True)
        launch.splash.set_details_open(False)
        deadline = time.monotonic() + 60
        while not self.codes and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(self.codes, [0])
        self.assertNotIn("ready_kept_open", launch.marks)

    def test_the_error_view_reads_what_the_window_printed_last_though_no_tick_has_yet(self):
        launch = self.launch("viewer")
        launch.state_dir = launch.root / "temp" / "viewer_last_words"
        launch.state_dir.mkdir(parents=True)
        launch.open_splash()
        log = launch.state_dir / "application.log"
        launch.splash.follow_log(log)
        log.write_bytes(b"Traceback (most recent call last):\nValueError: right at the very end")  # no final newline
        launch.show_failure(1)
        text = launch.splash.details.toPlainText()
        self.assertTrue(text.endswith("ValueError: right at the very end"), text)
        self.assertTrue(launch.splash.error)

    def test_a_session_launch_follows_the_viewers_output_too(self):
        launch = self.launch("session", app_args=["--settings", "x.json"], environment={"FAKE_WINDOW_STAY": "5"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)
        self.assertIn("fake window starts", launch.splash.details.toPlainText())
        self.assertNotIn("Managed Python", launch.splash.details.toPlainText())  # a session runs no check
        launch.splash.close_button.click()


class FallbackTests(LaunchCase):
    def test_an_environment_that_needs_setup_hands_over_to_the_visible_console(self):
        launch = self.launch("tools", environment={"FAKE_CHECK_EXIT": "10"})
        self.assertEqual(self.run_launch(launch), 0)
        launch.popen.assert_called_once()
        command = launch.popen.call_args.args[0]
        self.assertEqual(command[-1], "tools")
        self.assertTrue(command[-2].endswith("EMAPSSN_Desktop_Launcher.bat"))
        self.assertEqual(launch.marks["console_reason"], "the environment needs setup or repair")
        self.assertNotIn("monitor_started", launch.marks)
        self.assertEqual(self.state_dirs(), [])

    def test_a_missing_managed_python_does_too_before_any_check_runs(self):
        launch = Launch("viewer", self.root, python=self.root / ".venv" / "nothing.exe", uv="uv", show=False,
                        popen=mock.Mock())
        launch.finished.connect(lambda code: None)
        launch.begin()
        launch.popen.assert_called_once()
        self.assertIn("missing", launch.marks["console_reason"])
        self.assertNotIn("check_started", launch.marks)

    def test_so_does_a_missing_uv(self):
        launch = self.launch("viewer", uv=None)
        with mock.patch.object(Desktop_Launcher, "find_uv", return_value=None):
            launch.begin()
        launch.popen.assert_called_once()
        self.assertEqual(launch.marks["console_reason"], "uv is not installed")

    def test_a_python_that_cannot_start_does_too_instead_of_waiting_for_a_check_that_never_runs(self):
        broken = self.root / "broken_python.exe"
        broken.write_bytes(b"this is not a program")
        launch = Launch("viewer", self.root, python=broken, uv="uv", show=False, popen=mock.Mock(name="popen"))
        codes = []
        launch.finished.connect(codes.append)
        self.addCleanup(lambda: launch.splash is not None and launch.splash.deleteLater())
        launch.begin()
        deadline = time.monotonic() + 30
        while not codes and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(codes, [0])
        launch.popen.assert_called_once()
        self.assertEqual(launch.marks["console_reason"], "the environment check could not run")
        self.assertNotIn("monitor_started", launch.marks)

    def test_a_console_that_cannot_open_leaves_the_stub_to_try(self):
        launch = self.launch("viewer", environment={"FAKE_CHECK_EXIT": "10"})
        launch.popen.side_effect = OSError("no console")
        self.assertEqual(self.run_launch(launch), Desktop_Launcher.EXIT_FALLBACK)


def short_socket_folder(test):
    """POSIX: Qt's temp folder for a test that listens on a real local socket, one whose path is short.

    A local socket is a Unix socket file named ``<temp folder>/<name>``, and a Unix socket path may hold
    103 characters on macOS (107 on Linux). The test run's own temp folder (tests/session_folder.py) gives
    112 to 120 and more there, so the test would fail to listen on the platform the windows run on.
    Qt reads TMPDIR whenever it makes the path, for the server and for a launcher's connection alike.
    """
    folder = tempfile.mkdtemp(prefix="ssn-", dir="/tmp")
    test.addCleanup(shutil.rmtree, folder, True)
    return folder


class QuietEndTests(LaunchCase):
    def test_an_open_window_of_this_checkout_is_brought_forward_and_nothing_else_starts(self):
        if os.name != "nt":  # Windows' local sockets are named pipes, with no such limit
            self.enterContext(mock.patch.dict(os.environ, {"TMPDIR": short_socket_folder(self)}))
        name = Desktop_Launcher.single_instance_server_name("SSN_Config", self.root)
        activated = []
        server = QLocalServer()
        QLocalServer.removeServer(name)
        self.assertTrue(server.listen(name), server.errorString())
        if os.name != "nt":
            self.assertLess(len(server.fullServerName()), 100)
        self.addCleanup(server.close)
        server.newConnection.connect(lambda: activated.append(server.nextPendingConnection().readAll()))
        launch = self.launch("viewer")
        self.assertEqual(self.run_launch(launch, seconds=10), 0)
        deadline = time.monotonic() + 3
        while not activated and time.monotonic() < deadline:
            self.app.processEvents()
        self.assertEqual(len(activated), 1)
        self.assertIn("activated_existing", launch.marks)
        self.assertIsNone(launch.splash)
        self.assertNotIn("check_started", launch.marks)
        self.assertEqual(self.state_dirs(), [])

    def test_a_launch_under_way_makes_a_second_one_end_without_a_splash(self):
        held = Desktop_Launcher.acquire_launch_lock(self.root, "viewer")
        self.assertIsNotNone(held)
        self.addCleanup(held.unlock)
        launch = self.launch("viewer")
        self.assertEqual(self.run_launch(launch, seconds=10), 0)
        self.assertIn("launch_in_progress", launch.marks)
        self.assertIsNone(launch.splash)
        self.assertEqual(self.state_dirs(), [])

    def test_the_lock_is_let_go_when_the_launch_ends(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        again = Desktop_Launcher.acquire_launch_lock(self.root, "viewer")
        self.assertIsNotNone(again)
        again.unlock()

    def test_the_lock_is_let_go_after_an_error_view_too(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW": "fail"})
        launch.begin()
        deadline = time.monotonic() + 60
        while not launch.splash.error and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        launch.splash.close_button.click()
        again = Desktop_Launcher.acquire_launch_lock(self.root, "viewer")
        self.assertIsNotNone(again)
        again.unlock()


class SessionTests(LaunchCase):
    def test_a_viewer_session_skips_the_check_and_the_lock_and_gives_the_viewer_its_arguments(self):
        launch = self.launch("session", app_args=["--settings", "C:/tmp/ssn_viewer_1.json", "--delete-settings"],
                             environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        self.assertNotIn("check_started", launch.marks)
        self.assertIn("gui_ready", launch.marks)
        self.assertEqual(launch.splash.kind, "viewer")
        self.assertFalse(Desktop_Launcher.lock_path(self.root, "session").exists())
        # Two Viewers may be open: a second session starts while the first still runs.
        import json

        recorded = json.loads((launch.state_dir / "argv.json").read_text(encoding="utf-8")) \
            if (launch.state_dir / "argv.json").exists() else None
        if recorded is not None:
            self.assertEqual(recorded, ["--settings", "C:/tmp/ssn_viewer_1.json", "--delete-settings"])

    def test_two_sessions_can_run_at_once(self):
        first = self.launch("session", environment={"FAKE_WINDOW_STAY": "1.5"})
        second = Launch("session", self.root, python=sys.executable, popen=mock.Mock(), show=False)
        codes = []
        second.finished.connect(codes.append)
        self.addCleanup(lambda: second.splash is not None and second.splash.deleteLater())
        first.begin()
        second.begin()
        deadline = time.monotonic() + 60
        while (not self.codes or not codes) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual((self.codes, codes), ([0], [0]))
        self.assertIn("gui_ready", first.marks)
        self.assertIn("gui_ready", second.marks)
        self.assertNotEqual(first.state_dir, second.state_dir)


# A step of a plug-in's start: prints its name's lines and exits with FAKE_STEP_EXIT if it is the "second".
FAKE_STEP = textwrap.dedent(
    '''
    import os, sys
    name = sys.argv[1]
    print(name + ": line one", flush=True)
    print(name + ": cwd is " + os.path.basename(os.getcwd()), flush=True)
    sys.exit(int(os.environ.get("FAKE_STEP_EXIT", "0")) if name == "second" else 0)
    '''
)
# What opt_vr's monitor script does, as a stand-in: register the window with the monitor, call its main().
FAKE_PLUGIN_MONITOR = textwrap.dedent(
    '''
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "src"))
    from desktop import Desktop_Launcher_Monitor as monitor
    monitor.register_application("plug", here / "Window.py", here, monitor=Path(__file__).resolve())
    raise SystemExit(monitor.main())
    '''
)


class PluginKindTests(LaunchCase):
    """A program beside this one adds a kind: its own steps, batch file, key and monitor, with this file naming none."""

    FIRST, SECOND = "Checking the first thing…", "Checking the second thing…"
    WARNING = "The plug will still open, but may not work here."

    def setUp(self):
        super().setUp()
        self.plug = self.root / "plug"
        self.plug.mkdir()
        (self.plug / "Window.py").write_text(FAKE_WINDOW, encoding="utf-8")
        (self.plug / "Step.py").write_text(FAKE_STEP, encoding="utf-8")
        (self.plug / "Monitor_Plug.py").write_text(FAKE_PLUGIN_MONITOR, encoding="utf-8")
        (self.plug / "Launch_Plug.bat").write_text("@echo off\r\n", encoding="utf-8")
        from desktop import Desktop_Launcher_Monitor

        Desktop_Launcher_Monitor.register_application("plug", self.plug / "Window.py", self.plug,
                                                      monitor=self.plug / "Monitor_Plug.py")
        self.addCleanup(Desktop_Launcher_Monitor.EXTRA_APPLICATIONS.pop, "plug", None)
        self.register()

    def step(self, name, label, **options):
        return LaunchStep(label, lambda root, python: [python, "-u", str(Path(root) / "plug" / "Step.py"), name],
                          self.WARNING, **options)

    def register(self, steps=None, **options):
        steps = (self.step("first", self.FIRST), self.step("second", self.SECOND, working_dir=self.plug)) \
            if steps is None else steps
        kind = Desktop_Launcher.register_kind(LaunchKind(
            "plug", "config", "SSN_Plug", batch_file=self.plug / "Launch_Plug.bat", steps=steps,
            app_user_model_id="ca.test.plug", **options))
        self.addCleanup(Desktop_Launcher.LAUNCH_KINDS.pop, "plug", None)
        return kind

    def run_until(self, launch, condition, seconds=60):
        launch.begin()
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline and not self.codes:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(condition(), "the condition was not met")

    def watch_statuses(self):
        seen = []
        original = Launch.open_splash

        def watching(launch):
            original(launch)
            shown = launch.splash.show_text
            launch.splash.show_text = lambda text, fraction=None: (seen.append((text, fraction)), shown(text, fraction))

        patcher = mock.patch.object(Launch, "open_splash", watching)
        patcher.start()
        self.addCleanup(patcher.stop)
        return seen

    def test_the_kind_starts_its_own_window_through_its_own_monitor_in_its_own_state_folder(self):
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "5"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)
        self.assertTrue(launch.state_dir.name.startswith("plug_"))
        self.assertIn("fake window starts", launch.splash.details.toPlainText())  # the plug's window, not Config's
        self.assertEqual(launch.splash.kind, "config")
        self.assertEqual(Path(launch.monitor.args[2]), self.plug / "Monitor_Plug.py")
        self.assertEqual(launch.monitor.args[3:6], ["--failure-report", "dialog", "plug"])

    def test_its_steps_run_after_the_check_and_before_the_window_each_with_a_stretch_of_the_bar(self):
        seen = self.watch_statuses()
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        texts = [text for text, _ in seen]
        self.assertEqual(texts[:3], ["Managed Python: fake", "Detected: a fake GPU", "Dependency environment is ready (fake)."])
        order = [texts.index(text) for text in ("Dependency environment is ready (fake).", self.FIRST, self.SECOND,
                                                "Starting…", "Loading PyTorch…")]
        self.assertEqual(order, sorted(order))
        fractions = dict(seen)
        checked, started = Splash.FRACTION_CHECKED, Splash.FRACTION_STARTED
        self.assertAlmostEqual(fractions[self.FIRST], checked)
        self.assertAlmostEqual(fractions[self.SECOND], checked + (started - checked) / 2)
        self.assertAlmostEqual(fractions["Starting…"], started)
        for step in ("step_1_started", "step_1_ended", "step_2_started", "step_2_ended", "monitor_started"):
            self.assertIn(step, launch.marks)
        order = ["check_finished", "step_1_started", "step_1_ended", "step_2_started", "step_2_ended", "monitor_started"]
        self.assertEqual([launch.marks[name] for name in order], sorted(launch.marks[name] for name in order))
        self.assertLess(launch.marks["step_1_started"], launch.marks["step_1_ended"])  # a step takes some time
        self.assertEqual(launch.warnings, [])
        self.assertNotIn("warnings", launch.marks)

    def test_a_kind_with_no_steps_goes_from_the_check_to_the_window_as_every_kind_did(self):
        self.register(steps=())
        seen = self.watch_statuses()
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        fractions = dict(seen)
        self.assertAlmostEqual(fractions["Starting…"], Splash.FRACTION_CHECKED)
        self.assertNotIn("step_1_started", launch.marks)

    def test_the_output_of_a_step_that_passed_is_in_the_details_in_order_in_the_plain_colour(self):
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "5"})
        launch.begin()
        launch.splash.set_details_open(True)
        deadline = time.monotonic() + 60
        while "gui_ready" not in launch.marks and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        lines = launch.splash.details.toPlainText().splitlines()
        self.assertEqual(lines[3:7], ["first: line one", "first: cwd is " + self.root.name,
                                      "second: line one", "second: cwd is plug"])  # a step's folder: its own, or the project's
        self.assertEqual([launch.splash.details.kind_of(index) for index in range(3, 7)], ["info"] * 4)
        self.assertNotIn(self.WARNING, lines)
        self.assertTrue(launch.splash.ready)  # opened by the reader: it stays
        launch.splash.close_button.click()

    def test_a_step_that_fails_is_a_warning_the_details_open_by_themselves_and_the_start_goes_on(self):
        launch = self.launch("plug", environment={"FAKE_STEP_EXIT": "11", "FAKE_WINDOW_STAY": "8"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)  # the window opened though the step failed
        splash = launch.splash
        self.assertEqual(launch.warnings, [self.SECOND])
        self.assertEqual(launch.marks["warnings"], [self.SECOND])
        lines = splash.details.toPlainText().splitlines()
        first = lines.index("first: line one")
        self.assertEqual(lines[first + 2:first + 5], ["second: line one", "second: cwd is plug", self.WARNING])
        kinds = [splash.details.kind_of(index) for index in range(len(lines))]
        self.assertEqual(kinds[first:first + 2], ["info", "info"])        # the step that passed
        self.assertEqual(kinds[first + 2:first + 5], ["warning"] * 3)     # the one that did not, and what it means
        self.assertIn("fake window starts", lines)                        # the window went on to start
        self.assertEqual(kinds[lines.index("fake window starts")], "info")
        # Nobody opened the details: the warning did, and at the end the splash stays at Ready until Close is pressed.
        for _ in range(60):
            self.app.processEvents()
            time.sleep(0.02)
        self.assertEqual(self.codes, [], "the launch ended on its own")
        self.assertTrue(splash.isVisible())
        self.assertTrue(splash.details_open)
        self.assertTrue(splash.ready)
        self.assertEqual(splash.status, "Ready")
        self.assertIn("ready_kept_open", launch.marks)
        self.assertNotIn("finished", launch.marks)
        self.assertFalse(splash.error)
        self.assertTrue((launch.state_dir / "terminal.dismissed").is_file())  # the monitor is told the launcher is done
        splash.close_button.click()
        self.assertEqual(self.codes, [0])

    def test_a_reader_who_hides_the_details_after_the_warning_has_the_splash_close_at_ready_as_it_does_for_anyone(self):
        launch = self.launch("plug", environment={"FAKE_STEP_EXIT": "11", "FAKE_WINDOW_STAY": "0.1", "FAKE_WINDOW_DELAY": "1.5"})
        self.run_until(launch, lambda: "step_2_ended" in launch.marks)
        self.assertTrue(launch.splash.details_open)
        launch.splash.set_details_open(False)
        deadline = time.monotonic() + 60
        while not self.codes and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertEqual(self.codes, [0])
        self.assertNotIn("ready_kept_open", launch.marks)

    def test_a_step_that_cannot_start_or_cannot_be_put_together_is_a_warning_too(self):
        def broken(root, python):
            raise RuntimeError("cannot put it together")

        steps = (LaunchStep(self.FIRST, lambda root, python: [str(self.root / "no-such-program.exe")], self.WARNING),
                 LaunchStep(self.SECOND, broken, self.WARNING))
        self.register(steps=steps)
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "5"})
        self.run_until(launch, lambda: "gui_ready" in launch.marks)
        self.assertEqual(launch.warnings, [self.FIRST, self.SECOND])
        text = launch.splash.details.toPlainText()
        self.assertIn("cannot put it together", text)
        self.assertEqual(text.count(self.WARNING), 2)
        self.assertIn("fake window starts", text)
        launch.splash.close_button.click()

    def test_the_kinds_own_batch_file_is_the_fallback_not_configs_and_no_step_runs_before_it(self):
        launch = self.launch("plug", environment={"FAKE_CHECK_EXIT": "10"})
        self.assertEqual(self.run_launch(launch), 0)
        launch.popen.assert_called_once()
        command = launch.popen.call_args.args[0]
        self.assertEqual(Path(command[-2]), self.plug / "Launch_Plug.bat")
        self.assertEqual(command[-1], "plug")
        self.assertNotIn("step_1_started", launch.marks)
        self.assertNotIn("monitor_started", launch.marks)
        self.assertEqual(Desktop_Launcher.visible_console_command("C:/p", "tools")[3],
                         str(Path("C:/p") / Desktop_Launcher.BATCH_FILE))

    def test_the_console_option_runs_the_kinds_batch_file_in_this_console(self):
        with mock.patch.object(Desktop_Launcher.subprocess, "call", return_value=5) as call, mock.patch.object(
            Desktop_Launcher, "QApplication"
        ) as application:
            self.assertEqual(Desktop_Launcher.main(["plug", "--console", "--root", str(self.root)]), 5)
        application.assert_not_called()
        self.assertEqual(Path(call.call_args.args[0][-2]), self.plug / "Launch_Plug.bat")

    def test_a_second_launch_brings_the_open_window_forward_by_the_kinds_own_key(self):
        name = Desktop_Launcher.single_instance_server_name("SSN_Plug", self.root)
        activated = []
        server = QLocalServer()
        QLocalServer.removeServer(name)
        self.assertTrue(server.listen(name))
        self.addCleanup(server.close)
        server.newConnection.connect(lambda: activated.append(server.nextPendingConnection().readAll()))
        launch = self.launch("plug")
        self.assertEqual(self.run_launch(launch, seconds=10), 0)
        deadline = time.monotonic() + 3
        while not activated and time.monotonic() < deadline:
            self.app.processEvents()
        self.assertEqual(len(activated), 1)
        self.assertIn("activated_existing", launch.marks)
        self.assertIsNone(launch.splash)
        # Config's window under the same checkout is not the plug's: it is another key.
        self.assertFalse(Desktop_Launcher.activate_existing_instance("viewer", self.root))
        self.assertNotEqual(name, Desktop_Launcher.single_instance_server_name("SSN_Config", self.root))

    def test_the_kinds_catalogs_are_installed_with_the_splashs_own(self):
        self.register(catalogs=((str(self.plug), "emapssn_plug"),))
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "0.1"})
        with mock.patch.object(Splash, "install_extra_catalogs", wraps=Splash.install_extra_catalogs) as install:
            self.assertEqual(self.run_launch(launch), 0)
        install.assert_called_once()
        self.assertEqual(install.call_args.args[1], ((str(self.plug), "emapssn_plug"),))

    def test_the_command_line_accepts_the_kind_and_gives_the_process_its_taskbar_identity_before_the_window(self):
        for kind, expected in (("plug", ["ca.test.plug"]), ("viewer", [])):
            with self.subTest(kind=kind), mock.patch.object(Desktop_Launcher, "QApplication") as application, \
                    mock.patch.object(Desktop_Launcher, "Launch"), mock.patch.object(Desktop_Launcher, "QTimer"), \
                    mock.patch.object(Desktop_Launcher, "set_process_app_user_model_id") as identity:
                application.side_effect = lambda *arguments: (self.assertEqual(identity.called, kind == "plug"), mock.DEFAULT)[1]
                Desktop_Launcher.main([kind, "--root", str(self.root), "--hidden"])
            self.assertEqual([call.args[0] for call in identity.call_args_list], expected)

    def test_a_step_is_a_check_the_window_can_do_without_so_a_kind_may_run_steps_with_no_dependency_check(self):
        self.register(steps=(self.step("first", self.FIRST),), check=False)
        launch = self.launch("plug", environment={"FAKE_WINDOW_STAY": "0.1"})
        self.assertEqual(self.run_launch(launch), 0)
        self.assertNotIn("check_started", launch.marks)
        self.assertIn("step_1_started", launch.marks)
        self.assertIn("gui_ready", launch.marks)


class BrokenLaunchTests(LaunchCase):
    """A step that raises ends the launch instead of leaving the event loop, and the splash, waiting for ever."""

    def test_a_step_that_raises_before_anything_started_asks_the_stub_for_the_visible_console(self):
        launch = self.launch("viewer")
        with mock.patch.object(Launch, "open_splash", side_effect=RuntimeError("boom")), mock.patch.object(
            Desktop_Launcher.sys, "stderr"
        ):
            self.assertEqual(self.run_launch(launch, seconds=10), Desktop_Launcher.EXIT_FALLBACK)
        self.assertIn("boom", launch.marks["exception"])
        self.assertEqual(self.state_dirs(), [])
        again = Desktop_Launcher.acquire_launch_lock(self.root, "viewer")
        self.assertIsNotNone(again, "the lock stayed held")
        again.unlock()

    def test_one_that_raises_once_the_window_is_starting_ends_the_launch_as_handled(self):
        launch = self.launch("viewer", environment={"FAKE_WINDOW_STAY": "0.1"})
        with mock.patch.object(Launch, "window_ready", side_effect=RuntimeError("boom")), mock.patch.object(
            Desktop_Launcher.sys, "stderr"
        ):
            self.assertEqual(self.run_launch(launch), 0)
        self.assertIn("boom", launch.marks["exception"])
        self.assertTrue(launch.started_anything)


class ConsoleOptionTests(unittest.TestCase):
    def test_console_runs_the_old_launch_in_this_console_with_no_splash(self):
        with mock.patch.object(Desktop_Launcher.subprocess, "call", return_value=7) as call, mock.patch.object(
            Desktop_Launcher, "QApplication"
        ) as application:
            code = Desktop_Launcher.main(["tools", "--console", "--root", "C:/p"])
        self.assertEqual(code, 7)
        application.assert_not_called()  # no Qt, so no splash
        command = call.call_args.args[0]
        self.assertEqual(command[-1], "tools")
        self.assertTrue(command[-2].endswith("EMAPSSN_Desktop_Launcher.bat"))
        self.assertEqual(call.call_args.kwargs["cwd"], "C:/p")

    def test_only_a_session_takes_arguments_for_the_window(self):
        with self.assertRaises(SystemExit), mock.patch.object(sys, "stderr"):
            Desktop_Launcher.main(["tools", "--settings", "x"])


class ProcessTests(LaunchCase):
    """The launcher as the stub runs it: a process that ends with 0 once the window is up."""

    def test_the_launcher_process_ends_with_zero_after_a_start(self):
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
        environment.update({
            "QT_QPA_PLATFORM": "offscreen", "SSN_SPLASH_HIDDEN": "1", "FAKE_WINDOW_STAY": "0.2",
            "SSN_LAUNCH_MARKS_FILE": str(self.root / "marks.json"), "SSN_APP_SETTINGS_PATH": str(self.root / "a.json"),
        })
        shutil.copyfile(SRC / "desktop" / "Desktop_Launcher.py", self.root / "src" / "desktop" / "Desktop_Launcher.py")
        done = subprocess.run(
            [sys.executable, "-u", str(SRC / "desktop" / "Desktop_Launcher.py"), "viewer", "--root", str(self.root),
             "--python", sys.executable, "--hidden"],
            capture_output=True, text=True, env=environment, timeout=120, stdin=subprocess.DEVNULL,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        import json

        marks = json.loads((self.root / "marks.json").read_text(encoding="utf-8"))
        self.assertIn("gui_ready", marks)
        self.assertLess(marks["splash_shown"], 5000)


if __name__ == "__main__":
    unittest.main()
