"""Shared terminal launcher (utilities.Terminal_Launcher): terminal discovery,
per-platform argv construction, hold modes, launch errors, and the rule that
Python callers delegate to it instead of naming terminals themselves."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utilities import Terminal_Launcher as launcher  # noqa: E402


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

    def test_launch_forwards_supplied_standard_streams(self):
        log = object()
        with mock.patch.object(
            launcher, "_build_linux_argv", return_value=["/mock/terminal"]
        ), mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.launch_in_terminal(
                ["python", "worker.py"],
                cwd=PROJECT_ROOT,
                platform_name="linux",
                stdin=launcher.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        popen.assert_called_once_with(
            ["/mock/terminal"],
            cwd=str(PROJECT_ROOT),
            stdin=launcher.subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )

    def test_unsupported_platform_raises_without_spawning(self):
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaises(launcher.TerminalUnavailableError) as raised:
                launcher.launch_in_terminal(
                    ["python", "worker.py"], cwd=PROJECT_ROOT, platform_name="sunos5"
                )
        self.assertEqual(
            str(raised.exception),
            "Terminal launching is not supported on platform 'sunos5'.",
        )
        popen.assert_not_called()


class WindowsLaunchTests(unittest.TestCase):
    """The win32 path: a new console running the --windows-child wrapper,
    which runs the command and holds the console as requested."""

    COMMAND = ["python.exe", "worker.py", "a&b", "%PATH%"]

    def test_win32_launch_opens_a_new_console_running_the_child_wrapper(self):
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.launch_in_terminal(
                self.COMMAND,
                cwd=PROJECT_ROOT,
                hold="on_error",
                title="EMAP-SSN Viewer",
                platform_name="win32",
            )
        argv = popen.call_args.args[0]
        self.assertEqual(
            argv[:3],
            [launcher._console_python(), "-u", os.path.abspath(launcher.__file__)],
        )
        self.assertEqual(argv[-2], "--windows-child")
        self.assertEqual(
            launcher._decode_windows_payload(argv[-1]),
            (self.COMMAND, launcher.HoldMode.ON_ERROR, "EMAP-SSN Viewer"),
        )
        # CREATE_NEW_CONSOLE is 0x10, also where subprocess lacks the name.
        self.assertEqual(
            popen.call_args.kwargs, {"cwd": str(PROJECT_ROOT), "creationflags": 0x10}
        )

    def test_win32_launch_without_hold_or_title_runs_the_command_itself(self):
        with mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.launch_in_terminal(
                self.COMMAND, cwd=PROJECT_ROOT, platform_name="win32"
            )
        popen.assert_called_once_with(
            self.COMMAND, cwd=str(PROJECT_ROOT), creationflags=0x10
        )

    def test_child_uses_the_console_python_beside_pythonw(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pythonw = os.path.join(temp_dir, "pythonw.exe")
            with mock.patch.object(launcher.sys, "executable", pythonw):
                self.assertEqual(launcher._console_python(), pythonw)
                python = os.path.join(temp_dir, "python.exe")
                Path(python).touch()
                self.assertEqual(launcher._console_python(), python)

    def run_child(self, hold, title="", returncode=0, run_error=None, prompt_error=None):
        """Run _run_windows_child with the command, prompt and console faked."""
        payload = launcher._encode_windows_payload(self.COMMAND, hold, title)
        completed = subprocess.CompletedProcess(self.COMMAND, returncode)
        run = mock.Mock(side_effect=run_error or (lambda *args, **kwargs: completed))
        prompt = mock.Mock(side_effect=prompt_error)
        windll = mock.Mock()
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(launcher.subprocess, "run", run), \
                mock.patch("builtins.input", prompt), \
                mock.patch("ctypes.windll", windll, create=True), \
                mock.patch.dict(os.environ, {"COMSPEC": "test-shell.exe"}), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            status = launcher._run_windows_child(payload)
        return status, run, prompt, windll, stdout.getvalue(), stderr.getvalue()

    def test_on_error_child_prompts_only_after_a_failure(self):
        status, run, prompt, windll, output, _ = self.run_child(
            launcher.HoldMode.ON_ERROR, title="EMAP-SSN Viewer", returncode=3
        )
        self.assertEqual(status, 3)
        run.assert_called_once_with(self.COMMAND, check=False)
        prompt.assert_called_once_with("Press Enter to close...")
        self.assertEqual(output, "\nProcess exited with code 3.\n")
        windll.kernel32.SetConsoleTitleW.assert_called_once_with("EMAP-SSN Viewer")

        status, run, prompt, windll, output, _ = self.run_child(
            launcher.HoldMode.ON_ERROR, returncode=0
        )
        self.assertEqual(status, 0)
        prompt.assert_not_called()
        self.assertEqual(output, "")
        windll.kernel32.SetConsoleTitleW.assert_not_called()

    def test_child_that_cannot_start_exits_127_and_holds(self):
        status, _run, prompt, _windll, _output, errors = self.run_child(
            launcher.HoldMode.ON_ERROR, run_error=FileNotFoundError(2, "not found")
        )
        self.assertEqual(status, 127)
        self.assertEqual(errors, "Failed to start python.exe: [Errno 2] not found\n")
        prompt.assert_called_once_with("Press Enter to close...")

    def test_closed_stdin_at_the_prompt_is_tolerated(self):
        status, _run, prompt, _windll, _output, _errors = self.run_child(
            launcher.HoldMode.ON_ERROR, returncode=3, prompt_error=EOFError
        )
        self.assertEqual(status, 3)
        prompt.assert_called_once_with("Press Enter to close...")

    def test_always_child_opens_the_comspec_shell_after_the_command(self):
        status, run, prompt, _windll, output, _ = self.run_child(
            launcher.HoldMode.ALWAYS, returncode=0
        )
        self.assertEqual(status, 0)
        self.assertEqual(
            run.call_args_list,
            [mock.call(self.COMMAND, check=False), mock.call(["test-shell.exe"], check=False)],
        )
        prompt.assert_not_called()
        self.assertEqual(output, "\nProcess exited with code 0.\n")

    def test_script_entry_point_returns_the_child_status(self):
        payload = launcher._encode_windows_payload(
            self.COMMAND, launcher.HoldMode.ON_ERROR, ""
        )
        completed = subprocess.CompletedProcess(self.COMMAND, 5)
        script = os.path.abspath(launcher.__file__)
        with mock.patch.object(subprocess, "run", return_value=completed), \
                mock.patch("builtins.input"), \
                mock.patch("ctypes.windll", create=True), \
                redirect_stdout(io.StringIO()):
            with mock.patch.object(sys, "argv", [script, "--windows-child", payload]):
                with self.assertRaises(SystemExit) as child_exit:
                    runpy.run_path(script, run_name="__main__")
            with mock.patch.object(sys, "argv", [script]):
                with self.assertRaises(SystemExit) as direct_exit:
                    runpy.run_path(script, run_name="__main__")
        self.assertEqual(child_exit.exception.code, 5)
        self.assertEqual(
            direct_exit.exception.code, "Terminal_Launcher.py is an internal utility."
        )


class CallerIntegrationTests(unittest.TestCase):
    def test_python_callers_use_shared_helper_without_terminal_lists(self):
        # Every module that opens a terminal is found by scanning src, so a new
        # caller cannot be missed the way a hand-written list missed one.
        helper = SRC_DIR / "utilities" / "Terminal_Launcher.py"
        callers = sorted(
            path
            for path in SRC_DIR.rglob("*.py")
            if path != helper
            and "launch_in_terminal" in path.read_text(encoding="utf-8")
        )
        self.assertTrue(callers, "no launch_in_terminal callers found under src")
        for path in callers:
            with self.subTest(path=path.relative_to(SRC_DIR).as_posix()):
                source = path.read_text(encoding="utf-8")
                self.assertIn("Terminal_Launcher import", source)
                self.assertNotIn("def launch_in_terminal", source)
                self.assertNotIn("x-terminal-emulator", source)
                self.assertNotIn("gnome-terminal", source)
                self.assertNotIn('f"bash -c', source)


if __name__ == "__main__":
    unittest.main()
