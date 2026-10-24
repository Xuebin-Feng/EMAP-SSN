# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Desktop launcher monitor: the launch-and-wait handshake, the launch environment (managed path overrides, Linux Qt platform), terminal fallback and its dialog's language, exit codes and the application log's encoding."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from desktop import Desktop_Launcher_Monitor  # noqa: E402


class DesktopLauncherMonitorTests(unittest.TestCase):
    @staticmethod
    def _process(returncode):
        process = mock.Mock(pid=12345)
        process.wait.return_value = returncode
        return process

    def test_clean_exit_removes_log_and_state_after_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()
            (state / "gui.ready").write_text("ready\n", encoding="utf-8")
            (state / "terminal.dismissed").write_text("dismissed\n", encoding="utf-8")
            with mock.patch.object(
                Desktop_Launcher_Monitor.subprocess,
                "Popen",
                return_value=self._process(0),
            ):
                result = Desktop_Launcher_Monitor.run_monitor("viewer", state)

            self.assertEqual(result, 0)
            self.assertFalse(state.exists())

    def test_failure_before_ready_is_left_for_startup_terminal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            with mock.patch.object(
                Desktop_Launcher_Monitor.subprocess,
                "Popen",
                return_value=self._process(7),
            ), mock.patch.object(
                Desktop_Launcher_Monitor, "_open_error_terminal"
            ) as open_error:
                result = Desktop_Launcher_Monitor.run_monitor("tools", state)

            self.assertEqual(result, 7)
            self.assertEqual((state / "application.exit").read_text().strip(), "7")
            self.assertTrue((state / "application.log").is_file())
            open_error.assert_not_called()

    def test_failure_after_terminal_dismissal_opens_retained_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()
            (state / "gui.ready").write_text("ready\n", encoding="utf-8")
            (state / "terminal.dismissed").write_text("dismissed\n", encoding="utf-8")
            with mock.patch.object(
                Desktop_Launcher_Monitor.subprocess,
                "Popen",
                return_value=self._process(9),
            ), mock.patch.object(
                Desktop_Launcher_Monitor, "_open_error_terminal", return_value=True
            ) as open_error:
                result = Desktop_Launcher_Monitor.run_monitor("viewer", state)

            self.assertEqual(result, 9)
            open_error.assert_called_once_with(state / "application.log")
            self.assertTrue(state.exists())

    def test_windows_gui_child_uses_no_console_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.sys, "platform", "win32"
        ), mock.patch.object(
            Desktop_Launcher_Monitor, "_console_python", return_value="python.exe"
        ), mock.patch.object(
            Desktop_Launcher_Monitor.subprocess,
            "Popen",
            return_value=self._process(2),
        ) as popen:
            Desktop_Launcher_Monitor.run_monitor(
                "viewer", Path(temp_dir) / "state"
            )

        self.assertEqual(popen.call_args.kwargs["creationflags"], 0x08000000)

    def test_windows_monitor_bootstrap_is_detached_without_pythonw(self):
        process = self._process(0)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.sys,
            "executable",
            r"C:\project\.venv\Scripts\python.exe",
        ), mock.patch.object(
            Desktop_Launcher_Monitor.subprocess,
            "Popen",
            return_value=process,
        ) as popen:
            result = Desktop_Launcher_Monitor.launch_detached_monitor(
                "viewer",
                Path(temp_dir) / "state",
                platform_name="win32",
            )

        self.assertIs(result, process)
        command = popen.call_args.args[0]
        kwargs = popen.call_args.kwargs
        self.assertEqual(command[0], r"C:\project\.venv\Scripts\python.exe")
        self.assertNotIn("pythonw.exe", " ".join(command).lower())
        self.assertEqual(kwargs["creationflags"], 0x08000000)
        self.assertIs(kwargs["stdin"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertTrue(kwargs["close_fds"])
        self.assertNotIn("start_new_session", kwargs)

    def test_posix_monitor_bootstrap_starts_a_new_session(self):
        process = self._process(0)
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.subprocess,
            "Popen",
            return_value=process,
        ) as popen:
            result = Desktop_Launcher_Monitor.launch_detached_monitor(
                "tools",
                Path(temp_dir) / "state",
                platform_name="linux",
            )

        self.assertIs(result, process)
        kwargs = popen.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("creationflags", kwargs)
        self.assertIs(kwargs["stdin"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], Desktop_Launcher_Monitor.subprocess.DEVNULL)
        self.assertTrue(kwargs["close_fds"])

    def test_startup_waiter_observes_ready_at_fifty_millisecond_intervals(self):
        process = mock.Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()

            def signal_ready(delay):
                self.assertEqual(delay, 0.05)
                (state / "gui.ready").write_text("ready\n", encoding="utf-8")

            with mock.patch.object(
                Desktop_Launcher_Monitor.time, "monotonic", side_effect=[0.0, 0.0]
            ), mock.patch.object(
                Desktop_Launcher_Monitor.time, "sleep", side_effect=signal_ready
            ) as sleep:
                result = Desktop_Launcher_Monitor.wait_for_startup_state(
                    state, process, timeout=1.0
                )

        self.assertEqual(result, "ready")
        sleep.assert_called_once_with(0.05)

    def test_startup_waiter_distinguishes_exit_monitor_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()
            process = mock.Mock()
            process.poll.return_value = None
            (state / "application.exit").write_text("7\n", encoding="utf-8")
            self.assertEqual(
                Desktop_Launcher_Monitor.wait_for_startup_state(state, process),
                "application_exited",
            )
            (state / "application.exit").unlink()
            process.poll.return_value = 2
            self.assertEqual(
                Desktop_Launcher_Monitor.wait_for_startup_state(state, process),
                "monitor_exited",
            )
            process.poll.return_value = None
            self.assertEqual(
                Desktop_Launcher_Monitor.wait_for_startup_state(
                    state, process, timeout=0
                ),
                "timeout",
            )

    def test_launch_and_wait_maps_startup_states_and_spawn_failure(self):
        process = self._process(0)
        expected = {
            "ready": Desktop_Launcher_Monitor.STARTUP_READY,
            "application_exited": Desktop_Launcher_Monitor.STARTUP_APPLICATION_EXITED,
            "timeout": Desktop_Launcher_Monitor.STARTUP_TIMEOUT,
            "monitor_exited": Desktop_Launcher_Monitor.STARTUP_LAUNCH_FAILED,
        }
        for state, exit_code in expected.items():
            with self.subTest(state=state), mock.patch.object(
                Desktop_Launcher_Monitor,
                "launch_detached_monitor",
                return_value=process,
            ), mock.patch.object(
                Desktop_Launcher_Monitor,
                "wait_for_startup_state",
                return_value=state,
            ), mock.patch.object(Desktop_Launcher_Monitor.sys.stderr, "write"):
                self.assertEqual(
                    Desktop_Launcher_Monitor.launch_and_wait("viewer", Path("state")),
                    exit_code,
                )

        with mock.patch.object(
            Desktop_Launcher_Monitor,
            "launch_detached_monitor",
            side_effect=OSError("blocked"),
        ), mock.patch.object(Desktop_Launcher_Monitor.sys.stderr, "write"):
            self.assertEqual(
                Desktop_Launcher_Monitor.launch_and_wait("tools", Path("state")),
                Desktop_Launcher_Monitor.STARTUP_LAUNCH_FAILED,
            )


class SessionAndStatusTests(unittest.TestCase):
    """The monitor runs a Viewer for Config too, tells the window where to report its start, passes the window's
    arguments, and reports a crash after the window opened in a dialog when no terminal is open."""

    @staticmethod
    def _process(returncode):
        process = mock.Mock(pid=12345)
        process.wait.return_value = returncode
        return process

    def test_a_session_runs_the_viewer_script_with_the_arguments_it_was_given(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(2)
        ) as popen:
            Desktop_Launcher_Monitor.run_monitor(
                "session", Path(temp_dir) / "state", app_args=["--settings", "C:/t/ssn_viewer_1.json", "--delete-settings"]
            )
        command = popen.call_args.args[0]
        self.assertEqual(Path(command[2]), PROJECT_ROOT / "src" / "EMAPSSN_Viewer.py")
        self.assertEqual(command[3:], ["--settings", "C:/t/ssn_viewer_1.json", "--delete-settings"])

    def test_the_window_is_told_where_to_report_its_readiness_and_its_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(2)
        ) as popen:
            state = Path(temp_dir) / "state"
            Desktop_Launcher_Monitor.run_monitor("tools", state)
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["SSN_GUI_READY_FILE"], str((state / "gui.ready").resolve()))
            self.assertEqual(environment["SSN_STARTUP_STATUS_FILE"], str((state / "startup.status").resolve()))

    def test_every_kind_has_a_script_that_exists(self):
        self.assertEqual(sorted(Desktop_Launcher_Monitor.APP_SCRIPTS), ["session", "tools", "viewer"])
        for kind, script in Desktop_Launcher_Monitor.APP_SCRIPTS.items():
            with self.subTest(kind=kind):
                self.assertTrue((PROJECT_ROOT / script).is_file())

    def test_a_crash_after_the_window_opened_is_reported_in_a_dialog_when_asked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()
            (state / "gui.ready").write_text("ready\n", encoding="utf-8")
            (state / "terminal.dismissed").write_text("dismissed\n", encoding="utf-8")
            with mock.patch.object(
                Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(9)
            ), mock.patch.object(Desktop_Launcher_Monitor, "_open_error_terminal") as terminal, mock.patch.object(
                Desktop_Launcher_Monitor, "_report_application_failure"
            ) as dialog:
                result = Desktop_Launcher_Monitor.run_monitor("viewer", state, failure_report="dialog")
        self.assertEqual(result, 9)
        terminal.assert_not_called()
        dialog.assert_called_once_with(state.resolve() / "application.log")

    def test_a_failure_before_the_window_opened_leaves_the_report_to_the_launcher_whichever_way(self):
        for report in ("terminal", "dialog"):
            with self.subTest(report=report), tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
                Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(7)
            ), mock.patch.object(Desktop_Launcher_Monitor, "_open_error_terminal") as terminal, mock.patch.object(
                Desktop_Launcher_Monitor, "_report_application_failure"
            ) as dialog:
                Desktop_Launcher_Monitor.run_monitor("tools", Path(temp_dir) / "state", failure_report=report)
            terminal.assert_not_called()
            dialog.assert_not_called()

    def test_the_default_report_is_still_the_terminal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state = Path(temp_dir) / "state"
            state.mkdir()
            (state / "gui.ready").write_text("ready\n", encoding="utf-8")
            (state / "terminal.dismissed").write_text("dismissed\n", encoding="utf-8")
            with mock.patch.object(
                Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(9)
            ), mock.patch.object(Desktop_Launcher_Monitor, "_open_error_terminal", return_value=True) as terminal:
                Desktop_Launcher_Monitor.run_monitor("viewer", state)
        terminal.assert_called_once()

    def test_the_crash_dialog_shows_the_last_lines_and_names_the_log_in_the_windows_language(self):
        from PySide6.QtWidgets import QApplication
        from tests.translation_fixtures import outside_the_catalog, pseudo_language

        app = QApplication.instance() or QApplication([])
        pseudo_language(self, app)
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "application.log"
            log.write_text("".join(f"line {number}\n" for number in range(30)) + "\n   \nValueError: boom\n", encoding="utf-8")
            with mock.patch("PySide6.QtWidgets.QMessageBox.critical") as critical:
                Desktop_Launcher_Monitor._report_application_failure(log)
        critical.assert_called_once()
        _parent, title, text = critical.call_args.args
        self.assertIn("ValueError: boom", text)
        self.assertIn("line 20", text)
        self.assertNotIn("line 5\n", text)  # only the last lines
        self.assertIn(str(log), text)
        self.assertEqual(outside_the_catalog(title), "")
        self.assertEqual(text.count("\n\n"), 2)  # the lead, the lines, the log's name

    def test_a_log_that_cannot_be_read_still_gets_a_dialog(self):
        with mock.patch("PySide6.QtWidgets.QMessageBox.critical") as critical:
            Desktop_Launcher_Monitor._report_application_failure(Path("no/such/application.log"))
        critical.assert_called_once()
        self.assertIn("application.log", critical.call_args.args[2])

    def test_the_detached_monitor_is_started_with_the_report_and_arguments_it_was_given(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(0)
        ) as popen:
            Desktop_Launcher_Monitor.launch_detached_monitor(
                "session", Path(temp_dir) / "state", platform_name="win32", failure_report="dialog",
                app_args=["--settings", "x.json"],
            )
        command = popen.call_args.args[0]
        self.assertEqual(command[1], "-u")
        self.assertEqual(command[3:], ["--failure-report", "dialog", "session", str((Path(temp_dir) / "state").resolve()),
                                       "--settings", "x.json"])

    def test_the_default_start_adds_nothing_to_the_command(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(0)
        ) as popen:
            Desktop_Launcher_Monitor.launch_detached_monitor("tools", Path(temp_dir) / "state", platform_name="linux")
        self.assertEqual(popen.call_args.args[0][3:], ["tools", str((Path(temp_dir) / "state").resolve())])

    def test_an_unknown_report_or_kind_is_refused(self):
        with self.assertRaises(ValueError):
            Desktop_Launcher_Monitor.launch_detached_monitor("tools", Path("state"), failure_report="popup")
        with self.assertRaises(ValueError):
            Desktop_Launcher_Monitor.launch_detached_monitor("elsewhere", Path("state"))

    def test_the_command_line_passes_the_report_and_the_windows_arguments_on(self):
        with mock.patch.object(Desktop_Launcher_Monitor, "run_monitor", return_value=0) as run:
            Desktop_Launcher_Monitor.main(["--failure-report", "dialog", "session", "state", "--settings", "x.json", "--delete-settings"])
        run.assert_called_once_with("session", Path("state"), app_args=["--settings", "x.json", "--delete-settings"],
                                    failure_report="dialog")
        with mock.patch.object(Desktop_Launcher_Monitor, "run_monitor", return_value=0) as run:
            Desktop_Launcher_Monitor.main(["viewer", "state"])
        run.assert_called_once_with("viewer", Path("state"), app_args=[], failure_report="terminal")

    def test_a_viewer_session_for_config_has_no_console_a_launcher_and_the_settings_to_delete(self):
        env = {"SSN_TARGET_CACHE_PATH": "a/b.h5"}
        with mock.patch.object(Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(0)) as popen:
            Desktop_Launcher_Monitor.launch_viewer_session(
                PROJECT_ROOT, env, settings_path="C:/t/ssn_viewer_1.json", executable="C:/py/python.exe"
            )
        command = popen.call_args.args[0]
        self.assertEqual(command[:2], ["C:/py/python.exe", "-u"])
        self.assertEqual(Path(command[2]), PROJECT_ROOT / "src" / "desktop" / "Desktop_Launcher.py")
        self.assertEqual(command[3:], ["session", "--settings", "C:/t/ssn_viewer_1.json", "--delete-settings"])
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["env"], env)
        self.assertEqual(Path(kwargs["cwd"]), PROJECT_ROOT)
        self.assertEqual(kwargs["creationflags"], 0x08000000)
        for stream in ("stdin", "stdout", "stderr"):
            self.assertIs(kwargs[stream], Desktop_Launcher_Monitor.subprocess.DEVNULL)

    def test_a_viewer_session_without_settings_gets_no_settings_arguments(self):
        with mock.patch.object(Desktop_Launcher_Monitor.subprocess, "Popen", return_value=self._process(0)) as popen:
            Desktop_Launcher_Monitor.launch_viewer_session(PROJECT_ROOT, {})
        self.assertEqual(popen.call_args.args[0][3:], ["session"])


class ManagedEnvironmentTests(unittest.TestCase):
    monitor = Desktop_Launcher_Monitor

    def test_monitor_removes_only_managed_path_overrides(self):
        environment = {
            "PYTHONHOME": "/external/python",
            "PYTHONPATH": "/external/modules",
            "QT_PLUGIN_PATH": "/external/qt/plugins",
            "QT_QPA_PLATFORM_PLUGIN_PATH": "/external/qt/platforms",
            "QML_IMPORT_PATH": "/external/qml",
            "QML2_IMPORT_PATH": "/external/qml2",
            "PATH": "/managed/path",
            "CONDA_PREFIX": "/conda/base",
            "CUDA_VISIBLE_DEVICES": "0",
            "DISPLAY": ":0",
            "QT_QPA_PLATFORM": "wayland",
        }

        result = self.monitor._sanitize_managed_environment(
            environment, platform_name="linux"
        )

        self.assertIs(result, environment)
        for name in self.monitor._MANAGED_ENVIRONMENT_PATH_OVERRIDES:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["PATH"], "/managed/path")
        self.assertEqual(environment["CONDA_PREFIX"], "/conda/base")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["DISPLAY"], ":0")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "wayland")

    def test_monitor_removes_dyld_overrides_only_on_macos(self):
        mac_environment = {
            "DYLD_LIBRARY_PATH": "/external/lib",
            "DYLD_FRAMEWORK_PATH": "/external/frameworks",
        }
        linux_environment = dict(mac_environment)

        self.monitor._sanitize_managed_environment(
            mac_environment, platform_name="darwin"
        )
        self.monitor._sanitize_managed_environment(
            linux_environment, platform_name="linux"
        )

        self.assertEqual(mac_environment, {})
        self.assertEqual(linux_environment["DYLD_LIBRARY_PATH"], "/external/lib")
        self.assertEqual(
            linux_environment["DYLD_FRAMEWORK_PATH"], "/external/frameworks"
        )

    def test_monitor_passes_sanitized_environment_to_managed_child(self):
        process = mock.Mock(pid=12345)
        process.wait.return_value = 0
        inherited = {
            "PYTHONHOME": "/external/python",
            "QT_PLUGIN_PATH": "/external/qt/plugins",
            "CONDA_PREFIX": "/conda/base",
            "QT_QPA_PLATFORM": "offscreen",
        }
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            state_dir.mkdir()
            (state_dir / "terminal.dismissed").write_text(
                "dismissed\n", encoding="utf-8"
            )
            with mock.patch.dict(os.environ, inherited, clear=False), mock.patch.object(
                self.monitor.subprocess, "Popen", return_value=process
            ) as popen:
                result = self.monitor.run_monitor("viewer", state_dir)

        self.assertEqual(result, 0)
        child_environment = popen.call_args.kwargs["env"]
        self.assertNotIn("PYTHONHOME", child_environment)
        self.assertNotIn("QT_PLUGIN_PATH", child_environment)
        self.assertEqual(child_environment["CONDA_PREFIX"], "/conda/base")
        self.assertEqual(child_environment["QT_QPA_PLATFORM"], "offscreen")


class DesktopPlatformPolicyTests(unittest.TestCase):
    def test_wayland_desktop_launch_prefers_xcb(self):
        environment = {"XDG_SESSION_TYPE": "wayland"}

        result = Desktop_Launcher_Monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertIs(result, environment)
        self.assertEqual(result["QT_QPA_PLATFORM"], "xcb")

    def test_explicit_qt_platform_override_is_preserved(self):
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "QT_QPA_PLATFORM": "wayland",
        }

        Desktop_Launcher_Monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertEqual(environment["QT_QPA_PLATFORM"], "wayland")

    def test_non_wayland_session_is_unchanged(self):
        environment = {"XDG_SESSION_TYPE": "x11"}

        Desktop_Launcher_Monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertNotIn("QT_QPA_PLATFORM", environment)


class ApplicationLogEncodingTests(unittest.TestCase):
    """What Config and Tools print reaches application.log, which the error
    terminal reads back, as UTF-8.

    The log is the application's stdout and stderr, and Python writes a file in
    the ANSI code page on Windows: "é" came back as U+FFFD, and printing "α"
    raised UnicodeEncodeError, which closed Config from a Qt slot and stopped
    every Tools launch from a project folder with such a name.
    """

    SAMPLE = "Résistance α-amylase 中文"
    PROBE_EXIT = 3  # not 0, so the monitor keeps the log and returns at once
    # Runs an application script's real startup as the monitor starts it, but
    # with --headless --help, which exits before any window, single-instance
    # lock or event loop; then prints SAMPLE to both streams. The stream setup
    # has to come first in the script to cover that branch and the GUI alike.
    PROBE = "\n".join([
        "import os, runpy, sys",
        "script = sys.argv[1]",
        "sys.path.insert(0, os.path.dirname(script))",
        "sys.argv = [script, '--headless', '--help']",
        "try:",
        "    runpy.run_path(script, run_name='__main__')",
        "except SystemExit:",
        "    pass",
        f"print('stdout:', {ascii(SAMPLE)})",
        f"print('stderr:', {ascii(SAMPLE)}, file=sys.stderr)",
        f"sys.exit({PROBE_EXIT})",
    ])

    def test_application_log_is_utf8_whatever_the_file_encoding(self):
        start = Desktop_Launcher_Monitor.subprocess.Popen

        def start_probe(command, **kwargs):
            python, unbuffered, script = command
            return start([python, unbuffered, "-c", self.PROBE, script], **kwargs)

        # cp1252 is a file's encoding on Western Windows; the variable sets it
        # on any platform.
        environment = {
            name: value for name, value in os.environ.items() if name != "PYTHONUTF8"
        }
        environment["PYTHONIOENCODING"] = "cp1252"
        for app_kind in sorted(Desktop_Launcher_Monitor.APP_SCRIPTS):
            with self.subTest(app=app_kind):
                with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
                    os.environ, environment, clear=True
                ), mock.patch.object(
                    Desktop_Launcher_Monitor.subprocess, "Popen", side_effect=start_probe
                ):
                    state = Path(temp_dir) / "state"
                    result = Desktop_Launcher_Monitor.run_monitor(app_kind, state)
                    log = (state / "application.log").read_bytes().decode(
                        "utf-8", errors="replace"
                    )

                self.assertEqual(result, self.PROBE_EXIT, log)
                self.assertIn(f"stdout: {self.SAMPLE}", log)
                self.assertIn(f"stderr: {self.SAMPLE}", log)


class FailureDialogTests(unittest.TestCase):
    """The dialog shown when no terminal can open to show a failed application's
    log: in the language the windows show, while the log keeps English."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def log_file(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        log_path = Path(folder.name) / "application.log"
        log_path.write_text("Traceback\n", encoding="utf-8")
        return log_path

    def report(self):
        """The dialog's title and text, and the log, once a failure is reported."""
        log_path = self.log_file()
        with mock.patch("PySide6.QtWidgets.QMessageBox.critical") as critical:
            Desktop_Launcher_Monitor._report_terminal_failure(log_path, OSError(2))
        critical.assert_called_once()
        _parent, title, text = critical.call_args.args
        return title, text, log_path.read_text(encoding="utf-8")

    def test_the_dialog_shows_in_the_windows_language_and_the_log_in_english(self):
        from tests.translation_fixtures import outside_the_catalog, pseudo_language

        pseudo_language(self, self.app)
        title, text, log = self.report()
        for shown in (title, text):
            with self.subTest(shown=shown):
                self.assertTrue(shown)
                self.assertEqual(outside_the_catalog(shown), "", shown)
        self.assertIn("SSN could not open a terminal to display the application failure.", log)
        self.assertIn("Terminal error: 2", log)

    def test_in_its_own_process_the_monitor_installs_the_language_first(self):
        events = []

        class OwnApplication:
            """The application the monitor creates in its own process."""

            @staticmethod
            def instance():
                return None

            def __init__(self, arguments):
                self.quit = mock.Mock()

        with mock.patch("PySide6.QtWidgets.QApplication", OwnApplication), \
                mock.patch.object(Desktop_Launcher_Monitor, "_install_chosen_language",
                                  side_effect=lambda application: events.append(("language", application))), \
                mock.patch("PySide6.QtWidgets.QMessageBox.critical",
                           side_effect=lambda *arguments: events.append(("dialog", None))):
            Desktop_Launcher_Monitor._report_terminal_failure(self.log_file(), OSError(2))
        self.assertEqual([event for event, _ in events], ["language", "dialog"])
        application = events[0][1]
        self.assertIsInstance(application, OwnApplication)
        application.quit.assert_called_once_with()

    def test_the_dialog_shows_in_english_when_the_language_cannot_load(self):
        with mock.patch("desktop.Desktop_App.startup_language", return_value="de"), \
                mock.patch("desktop.Desktop_App.install_translations", side_effect=OSError("unreadable")) as install:
            Desktop_Launcher_Monitor._install_chosen_language(self.app)
        install.assert_called_once_with(self.app, "de")
        title, text, log = self.report()
        self.assertEqual(title, "SSN Application Failure")
        self.assertTrue(text.startswith("SSN could not open a terminal"), text)


if __name__ == "__main__":
    unittest.main()
