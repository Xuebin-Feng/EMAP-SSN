from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utilities import Terminal_Launcher as launcher  # noqa: E402
from utilities import Application_Identity as identity  # noqa: E402
from utilities import Application_Fonts as application_fonts  # noqa: E402
from utilities import Desktop_Launcher_Monitor as desktop_monitor  # noqa: E402


class FakeApplication:
    def __init__(self):
        self.application_name = None
        self.desktop_file_name = None

    def setApplicationName(self, name):
        self.application_name = name

    def setDesktopFileName(self, name):
        self.desktop_file_name = name


class ApplicationIdentityTests(unittest.TestCase):
    def test_canonical_product_and_component_names(self):
        self.assertEqual(identity.PRODUCT_NAME, "EMAP-SSN")
        self.assertEqual(
            identity.PRODUCT_LONG_NAME,
            "EMAP-SSN: Embedding- and Multiple-Alignment-integrated Protein "
            "Sequence Similarity Network Platform",
        )
        self.assertEqual(identity.CONFIG_DISPLAY_NAME, "EMAP-SSN Configuration")
        self.assertEqual(identity.VIEWER_DISPLAY_NAME, "EMAP-SSN Viewer")
        self.assertEqual(identity.TOOLS_DISPLAY_NAME, "EMAP-SSN Tools")

    def test_linux_identity_matches_desktop_file_basename(self):
        application = FakeApplication()

        with mock.patch.object(identity.sys, "platform", "linux"):
            identity.configure_linux_qt_desktop_identity(
                application, identity.VIEWER_DESKTOP_FILE_NAME
            )

        self.assertEqual(application.application_name, "emapssn")
        self.assertEqual(application.desktop_file_name, "emapssn")

    def test_non_linux_platform_is_unchanged(self):
        application = FakeApplication()

        with mock.patch.object(identity.sys, "platform", "darwin"):
            identity.configure_linux_qt_desktop_identity(
                application, identity.VIEWER_DESKTOP_FILE_NAME
            )

        self.assertIsNone(application.application_name)
        self.assertIsNone(application.desktop_file_name)

    def test_installer_generates_matching_wm_classes(self):
        installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn(
            f"StartupWMClass={identity.VIEWER_DESKTOP_FILE_NAME}\n", installer
        )
        self.assertIn(
            f"StartupWMClass={identity.TOOLS_DESKTOP_FILE_NAME}\n", installer
        )


class DesktopPlatformPolicyTests(unittest.TestCase):
    def test_wayland_desktop_launch_prefers_xcb(self):
        environment = {"XDG_SESSION_TYPE": "wayland"}

        result = desktop_monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertIs(result, environment)
        self.assertEqual(result["QT_QPA_PLATFORM"], "xcb")

    def test_explicit_qt_platform_override_is_preserved(self):
        environment = {
            "XDG_SESSION_TYPE": "wayland",
            "QT_QPA_PLATFORM": "wayland",
        }

        desktop_monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertEqual(environment["QT_QPA_PLATFORM"], "wayland")

    def test_non_wayland_session_is_unchanged(self):
        environment = {"XDG_SESSION_TYPE": "x11"}

        desktop_monitor._apply_linux_qt_platform_policy(
            environment, platform_name="linux"
        )

        self.assertNotIn("QT_QPA_PLATFORM", environment)


class VispyTextScalingTests(unittest.TestCase):
    def test_logical_pixel_size_is_independent_of_canvas_dpi(self):
        logical_pixels = 16.0

        for dpi in (72.0, 96.0, 144.0, 192.0, 220.0):
            with self.subTest(dpi=dpi):
                points = application_fonts.vispy_points_for_logical_pixels(
                    logical_pixels, dpi
                )
                rendered_pixels = points / 72.0 * dpi
                self.assertAlmostEqual(rendered_pixels, logical_pixels)

    def test_reference_point_size_preserves_96_dpi_appearance(self):
        for dpi in (96.0, 144.0, 192.0):
            with self.subTest(dpi=dpi):
                points = application_fonts.vispy_points_at_reference_dpi(8.0, dpi)
                rendered_pixels = points / 72.0 * dpi
                self.assertAlmostEqual(rendered_pixels, 8.0 / 72.0 * 96.0)

    def test_invalid_canvas_dpi_uses_reference_dpi(self):
        points = application_fonts.vispy_points_for_logical_pixels(16.0, 0.0)

        self.assertEqual(points, 12.0)


class TerminalRegistryTests(unittest.TestCase):
    def test_default_terminal_precedence(self):
        installed = {
            "xdg-terminal-exec": "/bin/xdg-terminal-exec",
            "x-terminal-emulator": "/bin/x-terminal-emulator",
            "xterm": "/bin/xterm",
        }

        spec, executable = launcher._find_linux_terminal(installed.get)

        self.assertEqual(spec.name, "xdg-terminal-exec")
        self.assertEqual(executable, "/bin/xdg-terminal-exec")

    def test_distribution_default_precedes_named_terminal(self):
        installed = {
            "x-terminal-emulator": "/bin/x-terminal-emulator",
            "ptyxis": "/bin/ptyxis",
        }

        spec, _ = launcher._find_linux_terminal(installed.get)

        self.assertEqual(spec.name, "x-terminal-emulator")

    def test_missing_terminal_raises_clear_error(self):
        with self.assertRaisesRegex(
            launcher.TerminalUnavailableError, "No supported terminal emulator"
        ):
            launcher._find_linux_terminal(lambda _name: None)

    def test_every_registered_terminal_builds_an_argv(self):
        command = ["/tmp/program with spaces", "quote'", 'double"', "; touch nope", "雪"]
        for spec in launcher._LINUX_TERMINALS:
            with self.subTest(terminal=spec.name):
                argv = launcher._build_linux_argv(
                    command, terminal=(spec, f"/mock/{spec.name}")
                )
                self.assertEqual(argv[0], f"/mock/{spec.name}")
                if spec.argument_mode is launcher._ArgumentMode.SINGLE_STRING:
                    self.assertEqual(shlex.split(argv[-1]), command)
                else:
                    self.assertEqual(argv[-len(command) :], command)

    def test_argument_families_use_documented_switches(self):
        command = ["python", "worker.py", "a b"]
        cases = {
            "ptyxis": ["/mock/ptyxis", "--", *command],
            "konsole": ["/mock/konsole", "-e", *command],
            "xfce4-terminal": ["/mock/xfce4-terminal", "-x", *command],
            "wezterm": ["/mock/wezterm", "start", "--", *command],
            "qterminal": ["/mock/qterminal", "-e", shlex.join(command)],
        }
        specs = {spec.name: spec for spec in launcher._LINUX_TERMINALS}
        for name, expected in cases.items():
            with self.subTest(terminal=name):
                self.assertEqual(
                    launcher._build_linux_argv(
                        command, terminal=(specs[name], f"/mock/{name}")
                    ),
                    expected,
                )


class TerminalPolicyTests(unittest.TestCase):
    def test_structured_command_is_required(self):
        with self.assertRaises(TypeError):
            launcher._normalize_command("python worker.py")
        with self.assertRaises(ValueError):
            launcher._normalize_command([])

    def test_never_hold_keeps_posix_command_direct(self):
        command = ["python", "worker.py", "a b"]
        self.assertEqual(
            launcher._posix_child_command(command, launcher.HoldMode.NEVER, None),
            command,
        )

    def test_posix_hold_wrapper_preserves_each_argument(self):
        command = ["/tmp/python path", "worker's.py", "; echo unsafe", "雪"]
        child = launcher._posix_child_command(
            command, launcher.HoldMode.ON_ERROR, "EMAP-SSN Viewer"
        )
        self.assertEqual(child[:2], ["bash", "-c"])
        self.assertEqual(child[3:6], ["ssn-terminal", "on_error", "EMAP-SSN Viewer"])
        self.assertEqual(child[6:], command)

    def test_windows_never_hold_runs_command_directly_without_title(self):
        command = [r"C:\Program Files\Python\python.exe", "worker.py"]
        self.assertEqual(
            launcher._build_windows_argv(command, launcher.HoldMode.NEVER, None),
            command,
        )

    def test_windows_wrapper_payload_preserves_metacharacters(self):
        command = ["python.exe", "worker.py", "a&b", "%PATH%", "bang!", "雪"]
        always = launcher._build_windows_argv(
            command, launcher.HoldMode.ALWAYS, "SSN & Tools"
        )
        on_error = launcher._build_windows_argv(
            command, launcher.HoldMode.ON_ERROR, "EMAP-SSN Viewer"
        )
        self.assertEqual(always[-2], "--windows-child")
        self.assertEqual(on_error[-2], "--windows-child")
        self.assertEqual(
            launcher._decode_windows_payload(always[-1]),
            (command, launcher.HoldMode.ALWAYS, "SSN & Tools"),
        )
        self.assertEqual(
            launcher._decode_windows_payload(on_error[-1]),
            (command, launcher.HoldMode.ON_ERROR, "EMAP-SSN Viewer"),
        )

    def test_macos_command_quotes_paths_and_closes_when_requested(self):
        argv = launcher._build_macos_argv(
            ["/tmp/python path", "worker's.py", "; echo unsafe", "雪"],
            cwd="/tmp/project path",
            env=None,
            close_on_exit=True,
        )
        script = "\n".join(argv)
        self.assertEqual(argv[0], "osascript")
        self.assertIn("close launchWindow", argv)
        self.assertIn("project path", script)
        self.assertIn("echo unsafe", script)

    def test_macos_startup_terminal_checks_for_exit_every_fifty_milliseconds(self):
        source = (SRC_DIR / "bin" / "EMAPSSN_Terminal_Launcher.sh").read_text(
            encoding="utf-8"
        )
        busy_loop = source.split("repeat while busy of launchTab", 1)[1]
        self.assertIn("delay 0.05", busy_loop.split("end repeat", 1)[0])

    def test_launch_missing_terminal_does_not_spawn_background_process(self):
        with mock.patch.object(launcher.shutil, "which", return_value=None), mock.patch.object(
            launcher.subprocess, "Popen"
        ) as popen:
            with self.assertRaises(launcher.TerminalUnavailableError):
                launcher.launch_in_terminal(
                    ["python", "worker.py"], cwd=PROJECT_ROOT, platform_name="linux"
                )
        popen.assert_not_called()

    def test_spawn_error_propagates_without_fallback(self):
        def fake_which(name):
            return "/mock/xterm" if name == "xterm" else None

        with mock.patch.object(launcher.shutil, "which", side_effect=fake_which), mock.patch.object(
            launcher.subprocess, "Popen", side_effect=FileNotFoundError("gone")
        ) as popen:
            with self.assertRaisesRegex(FileNotFoundError, "gone"):
                launcher.launch_in_terminal(
                    ["python", "worker.py"], cwd=PROJECT_ROOT, platform_name="linux"
                )
        popen.assert_called_once()

    def test_launch_forwards_environment_and_absolute_cwd(self):
        environment = {"PATH": os.environ.get("PATH", ""), "SSN_TEST": "value"}
        with mock.patch.object(
            launcher, "_build_linux_argv", return_value=["/mock/terminal"]
        ), mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.launch_in_terminal(
                ["python", "worker.py"],
                cwd=PROJECT_ROOT,
                env=environment,
                platform_name="linux",
            )
        popen.assert_called_once_with(
            ["/mock/terminal"], cwd=str(PROJECT_ROOT), env=environment
        )


class CallerIntegrationTests(unittest.TestCase):
    def test_python_callers_use_shared_helper_without_terminal_lists(self):
        callers = (
            SRC_DIR / "EMAPSSN_Tools.py",
            SRC_DIR / "EMAPSSN_Config.py",
            SRC_DIR / "commands" / "esmfold.py",
            SRC_DIR / "utilities" / "Desktop_Launcher_Monitor.py",
        )
        for path in callers:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("launch_in_terminal", source)
                self.assertNotIn("x-terminal-emulator", source)
                self.assertNotIn("gnome-terminal", source)
                self.assertNotIn('f"bash -c', source)


if __name__ == "__main__":
    unittest.main()
