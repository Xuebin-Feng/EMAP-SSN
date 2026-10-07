# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Desktop launcher monitor: the launch-and-wait handshake, the launch environment (managed path overrides, Linux Qt platform), terminal fallback and exit codes."""

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


if __name__ == "__main__":
    unittest.main()
