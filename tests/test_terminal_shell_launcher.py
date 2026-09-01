from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "src" / "bin" / "emapssn_terminal_launcher.sh"


class ShellTerminalLauncherTests(unittest.TestCase):
    def _run_with_terminal(self, terminal: str) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "argv.bin"
            fake = directory / terminal
            fake.write_text(
                "#!/bin/bash\nprintf '%s\\0' \"$@\" >\"$SSN_TEST_LOG\"\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = str(directory)
            environment["SSN_TEST_LOG"] = str(log_path)
            result = subprocess.run(
                [
                    "/bin/bash",
                    str(LAUNCHER),
                    "--cwd",
                    "/tmp/project path",
                    "--title",
                    "SSN Test",
                    "--",
                    "/tmp/program path",
                    "apostrophe'",
                    "; echo unsafe",
                    "雪",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            deadline = time.monotonic() + 2
            while not log_path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(log_path.exists(), f"fake {terminal} was not invoked")
            return [
                item.decode("utf-8")
                for item in log_path.read_bytes().split(b"\0")
                if item
            ]

    def test_shell_argument_families(self):
        cases = {
            "xdg-terminal-exec": ["--"],
            "x-terminal-emulator": ["-e"],
            "ptyxis": ["--"],
            "gnome-terminal": ["--"],
            "kgx": ["--"],
            "konsole": ["-e"],
            "xfce4-terminal": ["-x"],
            "mate-terminal": ["-x"],
            "kitty": ["--"],
            "alacritty": ["-e"],
            "wezterm": ["start", "--"],
            "foot": ["-e"],
            "footclient": ["-e"],
            "tilix": ["-e"],
            "terminator": ["-x"],
            "xterm": ["-e"],
            "urxvt": ["-e"],
            "rxvt": ["-e"],
            "st": ["-e"],
        }
        for terminal, prefix in cases.items():
            with self.subTest(terminal=terminal):
                argv = self._run_with_terminal(terminal)
                self.assertEqual(argv[: len(prefix)], prefix)
                child = argv[len(prefix) :]
                self.assertEqual(child[:2], ["bash", "-c"])
                self.assertEqual(child[-4:], [
                    "/tmp/program path",
                    "apostrophe'",
                    "; echo unsafe",
                    "雪",
                ])

    def test_single_string_terminal_families_round_trip_arguments(self):
        for terminal in ("qterminal", "lxterminal"):
            with self.subTest(terminal=terminal):
                argv = self._run_with_terminal(terminal)
                self.assertEqual(argv[0], "-e")
                self.assertEqual(len(argv), 2)
                child = shlex.split(argv[1])
                self.assertEqual(child[:2], ["bash", "-c"])
                self.assertEqual(child[-4:], [
                    "/tmp/program path",
                    "apostrophe'",
                    "; echo unsafe",
                    "雪",
                ])

    def test_standard_default_precedes_named_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            selected_log = directory / "selected"
            for terminal in ("xdg-terminal-exec", "x-terminal-emulator", "xterm"):
                fake = directory / terminal
                fake.write_text(
                    f"#!/bin/bash\nprintf '%s' {shlex.quote(terminal)} >\"$SSN_TEST_LOG\"\n",
                    encoding="utf-8",
                )
                fake.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = str(directory)
            environment["SSN_TEST_LOG"] = str(selected_log)
            result = subprocess.run(
                ["/bin/bash", str(LAUNCHER), "--", "/tmp/program"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            deadline = time.monotonic() + 2
            while not selected_log.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(selected_log.read_text(encoding="utf-8"), "xdg-terminal-exec")

    def test_missing_terminal_returns_nonzero_with_clear_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment["PATH"] = temporary
            result = subprocess.run(
                ["/bin/bash", str(LAUNCHER), "--", "/tmp/program"],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
        self.assertEqual(result.returncode, 127)
        self.assertIn("No supported terminal emulator was found", result.stderr)

    def test_macos_waits_until_terminal_reports_command_started(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log_path = directory / "osascript-argv.bin"
            fake_uname = directory / "uname"
            fake_uname.write_text("#!/bin/bash\nprintf Darwin\n", encoding="utf-8")
            fake_uname.chmod(0o755)
            fake_osascript = directory / "osascript"
            fake_osascript.write_text(
                "#!/bin/bash\nprintf '%s\\0' \"$@\" >\"$SSN_TEST_LOG\"\n",
                encoding="utf-8",
            )
            fake_osascript.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{directory}:{environment.get('PATH', '')}"
            environment["SSN_TEST_LOG"] = str(log_path)

            result = subprocess.run(
                [
                    "/bin/bash",
                    str(LAUNCHER),
                    "--cwd",
                    "/tmp/project path",
                    "--title",
                    "SSN Test",
                    "--",
                    "/tmp/program path",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            argv = [
                item.decode("utf-8")
                for item in log_path.read_bytes().split(b"\0")
                if item
            ]
            script = "\n".join(argv)
            self.assertIn("/bin/sleep 1;", script)
            self.assertIn("repeat with launchAttempt from 1 to 100", argv)
            self.assertIn("repeat while busy of launchTab", argv)
            self.assertLess(
                argv.index("repeat with launchAttempt from 1 to 100"),
                argv.index("repeat while busy of launchTab"),
            )

    def test_generated_macos_apps_open_command_file_without_apple_events(self):
        installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn('exec /usr/bin/open -a Terminal', installer)
        self.assertIn('Contents/Resources/start.command', installer)
        self.assertIn('"$app_kind" --terminal-session', installer)
        self.assertNotIn("NSAppleEventsUsageDescription", installer)

    def test_clean_cutover_has_no_legacy_macos_installer(self):
        self.assertFalse((PROJECT_ROOT / "install.command").exists())


if __name__ == "__main__":
    unittest.main()
