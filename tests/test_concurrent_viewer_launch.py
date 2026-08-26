# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Install_Dependencies  # noqa: E402
from utilities import Desktop_Launcher_Monitor  # noqa: E402
from web_ui import Web_Server  # noqa: E402
from web_ui import Browser_Page  # noqa: E402
from web_ui import esmfold_backend  # noqa: E402
from commands import agent, meta  # noqa: E402


class DependencyReadinessTests(unittest.TestCase):
    def test_ready_environment_is_checked_without_installing(self):
        completed = mock.Mock(returncode=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            state = {"active_backend": {"backend": "cpu"}}
            active = Install_Dependencies.backend_spec({"backend": "cpu"})
            with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                    mock.patch.object(Install_Dependencies, "verify_bundled_artifacts"), \
                    mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                    mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                    mock.patch.object(Install_Dependencies, "read_state", return_value=state), \
                    mock.patch.object(Install_Dependencies, "_state_mismatches", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "_backend_from_state", return_value=active), \
                    mock.patch.object(Install_Dependencies, "validate_backend", return_value={"devices": []}), \
                    mock.patch.object(
                        Install_Dependencies,
                        "_installed_version",
                        side_effect=lambda _python, package: (
                            Install_Dependencies.TRANSFORMERS_VERSION
                            if package == "transformers"
                            else Install_Dependencies.ESM_VERSION
                        ),
                    ), mock.patch.object(
                        Install_Dependencies, "validate_package_consistency", return_value=True
                    ) as consistency, mock.patch.object(
                        Install_Dependencies, "validate_esm_stack", return_value=True
                    ):
                ready = Install_Dependencies.environment_is_ready(
                    project_root=root, venv=root, uv_executable="uv"
                )

        self.assertTrue(ready)
        consistency.assert_called_once_with("uv", python)

    def test_changed_state_requires_setup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            python = root / "python"
            python.touch()
            with mock.patch.object(Install_Dependencies, "venv_python", return_value=python), \
                    mock.patch.object(Install_Dependencies, "verify_bundled_artifacts"), \
                    mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value={}), \
                    mock.patch.object(Install_Dependencies, "backend_specs", return_value=[]), \
                    mock.patch.object(Install_Dependencies, "hardware_fingerprint", return_value="fp"), \
                    mock.patch.object(Install_Dependencies, "read_state", return_value={}), \
                    mock.patch.object(
                        Install_Dependencies,
                        "_state_mismatches",
                        return_value=["hardware_fingerprint"],
                    ), \
                    mock.patch.object(Install_Dependencies, "validate_backend") as validate:
                ready = Install_Dependencies.environment_is_ready(
                    project_root=root, venv=root, uv_executable="uv"
                )

        self.assertFalse(ready)
        validate.assert_not_called()


class ConcurrentWebServerTests(unittest.TestCase):
    def _close_server(self, server):
        server.shutdown()
        server.server_close()

    def test_server_uses_requested_available_port(self):
        server = Web_Server.start_server(object(), preferred_port=0)
        self.addCleanup(self._close_server, server)
        self.assertGreater(server.server_address[1], 0)

    def test_occupied_preferred_port_falls_back_to_free_port(self):
        occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied.bind(("localhost", 0))
        occupied.listen(1)
        self.addCleanup(occupied.close)
        occupied_port = occupied.getsockname()[1]

        server = Web_Server.start_server(object(), preferred_port=occupied_port)
        self.addCleanup(self._close_server, server)
        self.assertNotEqual(server.server_address[1], occupied_port)
        self.assertGreater(server.server_address[1], 0)


class InstanceUrlRoutingTests(unittest.TestCase):
    class Viewer:
        def __init__(self):
            self.console_text = mock.Mock(text="")
            self.web_server = None

        def get_web_url(self, path):
            return f"http://localhost:49123/{path.lstrip('/')}"

        def update_console_background(self):
            pass

        def _open_web_ui(self, path, label, client_id):
            return Browser_Page.open_browser_page(self, path, label, client_id)

        def open_agent_ui(self):
            return self._open_web_ui("/agent.html", "Agent UI", "agent")

        def open_metadata_ui(self):
            return self._open_web_ui("/meta.html", "Metadata UI", "meta")

    def test_agent_meta_and_esmfold_use_viewer_instance_port(self):
        viewer = self.Viewer()
        expected = [
            "http://localhost:49123/agent.html",
            "http://localhost:49123/meta.html",
            "http://localhost:49123/esmfold.html",
        ]
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(agent, "register"), \
                mock.patch.object(meta, "register"), \
                mock.patch.object(meta.cfg, "METADATA_DIR", temp_dir), \
                mock.patch("webbrowser.open") as browser_open:
            agent.run(viewer, [])
            meta.run(viewer, [])
            esmfold_backend.open_esmfold_ui(viewer)

        self.assertEqual(
            [call.args[0] for call in browser_open.call_args_list], expected
        )

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


class LauncherStructureTests(unittest.TestCase):
    def test_desktop_launchers_show_then_dismiss_and_keep_direct_diagnostics(self):
        install_bat = (PROJECT_ROOT / "install.bat").read_text(encoding="utf-8")
        install_sh = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
        shell_supervisor = (SRC_DIR / "bin" / "SSN_Desktop_Launcher.sh").read_text(encoding="utf-8")
        windows_supervisor = (SRC_DIR / "bin" / "SSN_Desktop_Launcher.bat").read_text(encoding="utf-8")
        direct_windows = (SRC_DIR / "bin" / "SSN_Viewer.bat").read_text(encoding="utf-8")
        direct_posix = (SRC_DIR / "bin" / "SSN_Viewer.sh").read_text(encoding="utf-8")

        self.assertIn("cmd.exe", install_bat)
        self.assertIn("SSN_Desktop_Launcher.bat", install_bat)
        self.assertNotIn("wscript.exe", install_bat)
        self.assertEqual(install_sh.count("Terminal=false"), 2)
        self.assertIn("SSN_Desktop_Launcher.sh", install_sh)
        self.assertIn("SSN Viewer.app", install_sh)
        self.assertIn("SSN Tools.app", install_sh)
        self.assertIn("ssn_install_macos", install_sh)
        self.assertIn("ssn_install_linux", install_sh)
        self.assertIn('ssn_install_macos "$@"', install_sh)
        self.assertIn('ssn_install_linux "$@"', install_sh)
        self.assertIn("--open-terminal", shell_supervisor)
        self.assertIn("--terminal-session", shell_supervisor)
        self.assertIn("--check-only", shell_supervisor)
        self.assertIn("--setup-only", shell_supervisor)
        self.assertIn("terminal.dismissed", shell_supervisor)
        self.assertIn("--check-only", windows_supervisor)
        self.assertIn("--setup-only", windows_supervisor)
        self.assertIn("terminal.dismissed", windows_supervisor)
        self.assertIn("Starting SSN_Config", direct_windows)
        self.assertIn("Starting SSN_Config", direct_posix)
        self.assertNotIn("terminal.dismissed", direct_windows)
        self.assertNotIn("terminal.dismissed", direct_posix)


if __name__ == "__main__":
    unittest.main()
