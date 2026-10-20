"""Shared terminal launcher (utilities.Terminal_Launcher): terminal discovery,
per-platform argv construction, hold modes, launch errors, and the rule that
Python callers delegate to it instead of naming terminals themselves."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from utilities import Terminal_Launcher as launcher  # noqa: E402

XDG_TERMINAL = (launcher._LINUX_TERMINALS[0], "/mock/xdg-terminal-exec")


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
    def setUp(self):
        # Linux launches write a job file there (LinuxJobEnvironmentTests).
        self.runtime = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": self.runtime}))

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
        # The terminal program gets the environment minus the job's SSN_
        # variables; the command itself gets all of it (LinuxJobEnvironmentTests).
        environment = {"PATH": os.environ.get("PATH", ""), "SSN_TEST": "value"}
        with mock.patch.object(
            launcher, "_find_linux_terminal", return_value=XDG_TERMINAL
        ), mock.patch.object(
            launcher, "_build_linux_argv", return_value=["/mock/terminal"]
        ), mock.patch.object(launcher.subprocess, "Popen") as popen:
            launcher.launch_in_terminal(
                ["python", "worker.py"],
                cwd=PROJECT_ROOT,
                env=environment,
                platform_name="linux",
            )
        popen.assert_called_once_with(
            ["/mock/terminal"],
            cwd=str(PROJECT_ROOT),
            env={"PATH": environment["PATH"]},
        )

    def test_launch_forwards_supplied_standard_streams(self):
        log = object()
        with mock.patch.object(
            launcher, "_find_linux_terminal", return_value=XDG_TERMINAL
        ), mock.patch.object(
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
            env=mock.ANY,
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


class LinuxJobEnvironmentTests(unittest.TestCase):
    """A Linux command takes its environment and folder from a private job file
    that --posix-child reads inside the terminal, not from the terminal.

    Ptyxis, Ubuntu 26.04's default terminal behind xdg-terminal-exec, runs a
    --new-window command with the environment of whichever process started its
    running instance; ForeignEnvironmentTerminalTests runs a stand-in for it.
    """

    COMMAND = ["/opt/python", "worker.py", "a b"]
    JOB = {
        "PATH": "/job/bin:/usr/bin",
        "SSN_VIEWER_SETTINGS_PATH": "/tmp/ssn_viewer_job.json",
        "QT_QPA_PLATFORM": "xcb",
        "HF_TOKEN": "hf_secret_value",
    }

    def setUp(self):
        self.runtime = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": self.runtime}))

    def launch(self, env=None, popen_error=None, **options):
        with mock.patch.object(
            launcher, "_find_linux_terminal", return_value=XDG_TERMINAL
        ), mock.patch.object(
            launcher.subprocess, "Popen", side_effect=popen_error
        ) as popen:
            launcher.launch_in_terminal(
                self.COMMAND, cwd=PROJECT_ROOT, env=env, platform_name="linux", **options
            )
        return popen

    def job_files(self, folder=None):
        return sorted(Path(folder or self.runtime).iterdir())

    def test_command_runs_through_the_child_with_a_private_job_file(self):
        popen = self.launch(dict(self.JOB), hold="on_error", title="EMAP-SSN Viewer")
        argv = popen.call_args.args[0]
        [job_file] = self.job_files()
        self.assertEqual(argv[:2], ["/mock/xdg-terminal-exec", "--"])
        self.assertEqual(
            argv[2:8],
            [
                launcher._console_python(),
                "-I",
                os.path.abspath(launcher.__file__),
                "--posix-child",
                "on_error",
                os.path.join(self.runtime, job_file.name),
            ],
        )
        self.assertEqual(
            argv[8:],
            launcher._posix_child_command(
                self.COMMAND, launcher.HoldMode.ON_ERROR, "EMAP-SSN Viewer"
            ),
        )
        self.assertEqual(
            json.loads(job_file.read_text(encoding="utf-8")),
            {"cwd": str(PROJECT_ROOT), "environment": self.JOB},
        )
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(job_file.stat().st_mode), 0o600)
        # ps shows a command line to every user, so no value travels on it.
        self.assertFalse(
            [part for part in argv if "hf_secret_value" in part or "ssn_viewer_job" in part]
        )

    def test_terminal_program_gets_the_environment_without_job_variables(self):
        # A single-instance terminal this starts would hand SSN_ variables to
        # every window opened later, the user's own shells included.
        popen = self.launch(dict(self.JOB))
        expected = dict(self.JOB)
        del expected["SSN_VIEWER_SETTINGS_PATH"]
        self.assertEqual(popen.call_args.kwargs["env"], expected)

    def test_default_environment_is_this_process_environment(self):
        with mock.patch.dict(
            os.environ, {"SSN_TARGET_CACHE_MODE": "new", "QT_SCALE_FACTOR": "2"}
        ):
            snapshot = dict(os.environ)
            popen = self.launch()
        [job_file] = self.job_files()
        self.assertEqual(
            json.loads(job_file.read_text(encoding="utf-8"))["environment"], snapshot
        )
        self.assertEqual(
            popen.call_args.kwargs["env"],
            {key: value for key, value in snapshot.items() if not key.startswith("SSN_")},
        )

    def test_failed_terminal_start_removes_the_job_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "gone"):
            self.launch(dict(self.JOB), popen_error=FileNotFoundError("gone"))
        self.assertEqual(self.job_files(), [])

    def test_missing_terminal_writes_no_job_file(self):
        with mock.patch.object(launcher.shutil, "which", return_value=None):
            with self.assertRaises(launcher.TerminalUnavailableError):
                launcher.launch_in_terminal(
                    self.COMMAND, cwd=PROJECT_ROOT, env=dict(self.JOB), platform_name="linux"
                )
        self.assertEqual(self.job_files(), [])

    def test_job_file_goes_to_the_temporary_folder_without_a_runtime_folder(self):
        fallback = self.enterContext(tempfile.TemporaryDirectory())
        missing = os.path.join(self.runtime, "missing")
        for runtime in (None, "", missing):
            with self.subTest(runtime=runtime), mock.patch.object(
                launcher.tempfile, "tempdir", fallback
            ), mock.patch.dict(os.environ):
                os.environ.pop("XDG_RUNTIME_DIR")
                if runtime is not None:
                    os.environ["XDG_RUNTIME_DIR"] = runtime
                path = launcher._write_linux_job_file("/project", {"A": "1"})
                self.addCleanup(launcher._remove_quietly, path)
                self.assertEqual(os.path.dirname(path), fallback)
        self.assertEqual(self.job_files(), [])


class _Exec(BaseException):
    """Stands in for a successful exec, which never returns."""


class PosixChildTests(unittest.TestCase):
    """--posix-child, run first inside the new terminal: it reads and deletes the
    job file, then becomes the command in the job's folder and environment."""

    COMMAND = ["bash", "-c", "wrapper", "ssn-terminal", "on_error", "", "/opt/python", "w.py"]
    # What a running Ptyxis instance started by an earlier job hands a command.
    TERMINAL = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/user",
        "TERM": "xterm-256color",
        "COLORTERM": "truecolor",
        "VTE_VERSION": "8000",
        "SSN_PROBE": "job-ONE",
        "SSN_VIEWER_SETTINGS_PATH": "/tmp/ssn_viewer_earlier.json",
        "PYTHONPATH": "/stale/python",
    }
    # The caller's environment; it was itself started from some terminal.
    JOB = {
        "PATH": "/job/bin:/usr/bin",
        "HOME": "/home/user",
        "SSN_PROBE": "job-TWO",
        "TERM": "dumb",
        "WINDOWID": "4242",
        "LINES": "10",
    }
    CWD = "/projects/emap ssn 数据"

    def setUp(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.job_file = folder / "ssn_terminal_job.json"
        self.job_file.write_text(
            json.dumps({"cwd": self.CWD, "environment": self.JOB}), encoding="utf-8"
        )

    def run_child(self, hold="on_error", exec_error=_Exec):
        execvpe = mock.Mock(side_effect=exec_error)
        chdir = mock.Mock()
        prompt = mock.Mock(side_effect=EOFError)
        stdout, stderr = io.StringIO(), io.StringIO()
        status = None
        with mock.patch.dict(os.environ, self.TERMINAL, clear=True), \
                mock.patch.object(launcher.os, "execvpe", execvpe), \
                mock.patch.object(launcher.os, "chdir", chdir), \
                mock.patch("builtins.input", prompt), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                status = launcher._run_posix_child(hold, str(self.job_file), self.COMMAND)
            except _Exec:
                pass
        return status, execvpe, chdir, prompt, stdout.getvalue(), stderr.getvalue()

    def test_command_gets_the_job_environment_and_folder(self):
        status, execvpe, chdir, prompt, _output, _errors = self.run_child()
        self.assertIsNone(status)
        execvpe.assert_called_once_with(
            "bash",
            self.COMMAND,
            {
                # The job's own values; the earlier job's SSN_ variables and the
                # terminal's PYTHONPATH are gone.
                "PATH": "/job/bin:/usr/bin",
                "HOME": "/home/user",
                "SSN_PROBE": "job-TWO",
                # What describes the terminal comes from the terminal only.
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "VTE_VERSION": "8000",
                "PWD": self.CWD,
            },
        )
        chdir.assert_called_once_with(self.CWD)
        prompt.assert_not_called()
        self.assertFalse(self.job_file.exists())

    def test_missing_job_file_is_reported_and_held_open_unless_never(self):
        self.job_file.unlink()
        for hold, prompts in (("on_error", 1), ("always", 1), ("never", 0)):
            with self.subTest(hold=hold):
                status, execvpe, _chdir, prompt, output, errors = self.run_child(hold)
                self.assertEqual(status, 127)
                execvpe.assert_not_called()
                self.assertIn("Could not start the command in this terminal:", errors)
                self.assertEqual(prompt.call_count, prompts)
                self.assertEqual(
                    output, "\nProcess exited with code 127.\n" if prompts else ""
                )

    def test_corrupt_job_file_is_deleted_and_reported(self):
        for content in ("{", '{"cwd": "/x"}', '{"cwd": "/x", "environment": []}'):
            with self.subTest(content=content):
                self.job_file.write_text(content, encoding="utf-8")
                status, execvpe, *_rest, errors = self.run_child()
                self.assertEqual(status, 127)
                execvpe.assert_not_called()
                self.assertIn("Could not start the command in this terminal:", errors)
                self.assertFalse(self.job_file.exists())

    def test_command_that_cannot_start_is_reported(self):
        status, execvpe, *_rest, errors = self.run_child(
            exec_error=FileNotFoundError(2, "No such file or directory")
        )
        self.assertEqual(status, 127)
        execvpe.assert_called_once()
        self.assertIn("No such file or directory", errors)
        self.assertFalse(self.job_file.exists())

    def test_script_entry_point_runs_the_posix_child(self):
        script = os.path.abspath(launcher.__file__)
        argv = [script, "--posix-child", "never", str(self.job_file), *self.COMMAND]
        execvpe = mock.Mock(side_effect=_Exec)
        with mock.patch.object(os, "execvpe", execvpe), mock.patch.object(os, "chdir"), \
                mock.patch.object(sys, "argv", argv):
            with self.assertRaises(_Exec):
                runpy.run_path(script, run_name="__main__")
            self.assertEqual(execvpe.call_args.args[:2], ("bash", self.COMMAND))
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as missing:
                runpy.run_path(script, run_name="__main__")
        self.assertEqual(missing.exception.code, 127)


# A stand-in for Ptyxis behind xdg-terminal-exec (Ubuntu 26.04 GNOME). A running
# Ptyxis starts a --new-window command with the environment of the process that
# started the instance, here an earlier job's, not the caller's. This one also
# starts it in /. It records the environment it was itself given first.
_FOREIGN_ENVIRONMENT_TERMINAL = """#!/bin/sh
env > "$FAKE_TERMINAL_RECORD"
[ "$1" = "--" ] && shift
cd / || exit 1
exec env -i PATH=/usr/bin:/bin HOME=/nonexistent TERM=xterm-ptyxis \\
    SSN_PROBE=job-ONE SSN_VIEWER_SETTINGS_PATH=/tmp/ssn_viewer_earlier.json \\
    PYTHONPATH=/stale/python "$@"
"""

_ENVIRONMENT_PROBE = """
import json, os, sys
names = ("SSN_PROBE", "SSN_VIEWER_SETTINGS_PATH", "PATH", "PYTHONPATH", "TERM", "PWD",
         "FAKE_TERMINAL_RECORD", "HF_TOKEN")
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({"cwd": os.getcwd(), "environment": {name: os.environ.get(name) for name in names}}, handle)
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "runs a stand-in Linux terminal")
class ForeignEnvironmentTerminalTests(unittest.TestCase):
    """Real launches through a terminal that ignores the caller's environment and
    folder, as Ptyxis does (v0.4.0 Linux report, F1)."""

    def test_command_gets_the_caller_environment_and_folder(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_dir = root / "bin"
        bin_dir.mkdir()
        terminal = bin_dir / "xdg-terminal-exec"
        terminal.write_text(_FOREIGN_ENVIRONMENT_TERMINAL, encoding="utf-8")
        terminal.chmod(0o755)
        runtime = root / "runtime"
        runtime.mkdir(mode=0o700)
        work = root / "project dir 数据"
        work.mkdir()
        path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        self.enterContext(
            mock.patch.dict(os.environ, {"PATH": path, "XDG_RUNTIME_DIR": str(runtime)})
        )
        for hold, title in (("on_error", "EMAP-SSN Layout Generator"), ("never", None)):
            with self.subTest(hold=hold):
                record = root / f"terminal-{hold}.env"
                output = root / f"probe-{hold}.json"
                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if key not in ("PYTHONPATH", "PYTHONHOME")
                }
                environment.update(
                    SSN_PROBE="job-TWO",
                    FAKE_TERMINAL_RECORD=str(record),
                    HF_TOKEN="hf_secret_value",
                    TERM="xterm-caller",
                )
                process = launcher.launch_in_terminal(
                    [sys.executable, "-c", _ENVIRONMENT_PROBE, str(output)],
                    cwd=work,
                    env=environment,
                    hold=hold,
                    title=title,
                    platform_name="linux",
                    stdout=subprocess.DEVNULL,
                )
                self.assertEqual(process.wait(timeout=60), 0)
                seen = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(
                    seen["environment"],
                    {
                        "SSN_PROBE": "job-TWO",
                        "SSN_VIEWER_SETTINGS_PATH": None,
                        "PATH": path,
                        "PYTHONPATH": None,
                        "TERM": "xterm-ptyxis",
                        "PWD": str(work),
                        "FAKE_TERMINAL_RECORD": str(record),
                        "HF_TOKEN": "hf_secret_value",
                    },
                )
                self.assertEqual(os.path.realpath(seen["cwd"]), os.path.realpath(work))
                terminal_variables = record.read_text(encoding="utf-8").splitlines()
                self.assertIn(f"FAKE_TERMINAL_RECORD={record}", terminal_variables)
                self.assertEqual(
                    [line for line in terminal_variables if line.startswith("SSN_")], []
                )
                self.assertEqual(list(runtime.iterdir()), [])


# A stand-in for a single-instance terminal's client, such as Ptyxis's: it hands
# the command to a running instance and exits at once.
_HAND_OFF_TERMINAL = """#!/bin/sh
[ "$1" = "--" ] && shift
"$@" >/dev/null 2>&1 &
exit 0
"""


@unittest.skipUnless(sys.platform.startswith("linux"), "runs a stand-in Linux terminal")
class HandOffTerminalTests(unittest.TestCase):
    """A terminal command that ends at once is collected without the caller
    waiting on it (v0.4.0 Linux report, round 2, N3: one [ptyxis] <defunct>
    per launch until the launching app exited)."""

    def test_ended_terminal_command_is_not_left_a_zombie(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_dir = root / "bin"
        bin_dir.mkdir()
        terminal = bin_dir / "xdg-terminal-exec"
        terminal.write_text(_HAND_OFF_TERMINAL, encoding="utf-8")
        terminal.chmod(0o755)
        runtime = root / "runtime"
        runtime.mkdir(mode=0o700)
        output = root / "probe.json"
        path = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        self.enterContext(
            mock.patch.dict(os.environ, {"PATH": path, "XDG_RUNTIME_DIR": str(runtime)})
        )
        process = launcher.launch_in_terminal(
            [sys.executable, "-c", _ENVIRONMENT_PROBE, str(output)],
            cwd=root,
            platform_name="linux",
            stdout=subprocess.DEVNULL,
        )
        # Watched through /proc, not poll() or wait(): either would collect it.
        entry = Path(f"/proc/{process.pid}")
        deadline = time.monotonic() + 10
        while entry.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(entry.exists(), "The ended terminal command is still a zombie")
        self.assertEqual(process.wait(timeout=5), 0)
        self.assertEqual(process.poll(), 0)
        # The command it handed on still ran and took its job file.
        deadline = time.monotonic() + 30
        while not output.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(output.exists(), "The handed-on command did not run")
        self.assertEqual(list(runtime.iterdir()), [])


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


class WindowlessLaunchTests(unittest.TestCase):
    """launch_without_window: a headless Viewer's workers get no console, no
    terminal and no window, and write their output to a log file."""

    COMMAND = ["python.exe", "worker.py", "a&b", "%PATH%"]
    NEW_CONSOLE = 0x10
    DETACHED = 0x8
    NO_WINDOW = 0x08000000

    def launch_mocked(self, platform_name, **options):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        log_path = directory / "logs" / "worker.log"
        with mock.patch.object(launcher.subprocess, "Popen") as popen, \
                mock.patch.object(launcher, "_find_linux_terminal") as find_terminal, \
                mock.patch.object(launcher, "_reap_when_done") as reap:
            process = launcher.launch_without_window(
                self.COMMAND, cwd=PROJECT_ROOT, log_path=log_path,
                platform_name=platform_name, **options,
            )
        find_terminal.assert_not_called()
        return popen, process, reap, log_path

    def test_windows_starts_the_command_itself_with_a_hidden_console(self):
        popen, process, reap, log_path = self.launch_mocked("win32")

        self.assertEqual(popen.call_args.args[0], self.COMMAND)
        kwargs = popen.call_args.kwargs
        flags = kwargs["creationflags"]
        # CREATE_NO_WINDOW: the console exists, hidden, and a venv's real
        # interpreter inherits it. DETACHED_PROCESS would leave the redirector
        # without one and Windows would give that child a visible console.
        self.assertEqual(flags & self.NO_WINDOW, self.NO_WINDOW)
        self.assertEqual(flags & (self.NEW_CONSOLE | self.DETACHED), 0)
        self.assertEqual(kwargs["cwd"], str(PROJECT_ROOT))
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertNotIn("start_new_session", kwargs)
        # Windows has no zombies to collect.
        reap.assert_not_called()
        self.assertIs(process, popen.return_value)

    def test_posix_leads_its_own_session_and_is_collected(self):
        for platform_name in ("linux", "darwin"):
            with self.subTest(platform=platform_name):
                popen, process, reap, _ = self.launch_mocked(platform_name)

                self.assertEqual(popen.call_args.args[0], self.COMMAND)
                kwargs = popen.call_args.kwargs
                self.assertIs(kwargs["start_new_session"], True)
                self.assertNotIn("creationflags", kwargs)
                self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                reap.assert_called_once_with(popen.return_value)
                self.assertIs(process, reap.return_value)

    def test_output_goes_to_the_log_in_a_folder_that_is_created(self):
        popen, _, _, log_path = self.launch_mocked("win32")

        log = popen.call_args.kwargs["stdout"]
        self.assertEqual(Path(log.name), log_path)
        self.assertEqual(log.mode, "ab")
        self.assertTrue(log_path.parent.is_dir())
        # The child has its own copy of the handle; ours is closed.
        self.assertTrue(log.closed)

    def test_python_output_is_unbuffered_utf8_and_the_environment_is_the_callers(self):
        popen, _, _, _ = self.launch_mocked("win32", env={"KEEP": "1", "PYTHONUNBUFFERED": "0"})

        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["KEEP"], "1")
        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8:replace")
        # A caller that chose a buffering keeps it.
        self.assertEqual(environment["PYTHONUNBUFFERED"], "0")

        with mock.patch.dict(os.environ, {"ONLY_HERE": "yes"}):
            os.environ.pop("PYTHONUNBUFFERED", None)
            popen, _, _, _ = self.launch_mocked("linux")
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["ONLY_HERE"], "yes")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")

    def test_a_command_must_be_a_sequence_and_a_failed_start_propagates(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with self.assertRaises(TypeError):
            launcher.launch_without_window(
                "python worker.py", cwd=directory, log_path=directory / "w.log"
            )
        with mock.patch.object(
            launcher.subprocess, "Popen", side_effect=FileNotFoundError("no such python")
        ):
            with self.assertRaises(FileNotFoundError):
                launcher.launch_without_window(
                    ["missing"], cwd=directory, log_path=directory / "w.log",
                    platform_name="win32",
                )

    def stop_tree(self, process):
        """End process and what it started (behind a venv's redirector, the interpreter is a child)."""
        try:
            root = psutil.Process(process.pid)
            targets = [root, *root.children(recursive=True)]
        except psutil.Error:
            targets = []
        for target in targets:
            try:
                target.kill()
            except psutil.Error:
                pass
        process.wait(timeout=30)

    def run_real(self, directory, code, name="worker.log", **options):
        script = directory / "stand_in.py"
        script.write_text(code, encoding="utf-8")
        log_path = directory / "logs" / name
        process = launcher.launch_without_window(
            [sys.executable, str(script)], cwd=directory, log_path=log_path, **options
        )
        self.addCleanup(self.stop_tree, process)
        self.assertEqual(process.wait(timeout=60), 0)
        return log_path

    def test_stdout_and_stderr_both_reach_the_log_and_a_second_launch_appends(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        code = (
            "import sys\n"
            "print('to stdout')\n"
            "print('to stderr', file=sys.stderr)\n"
            "print('stdin is', repr(sys.stdin.read()))\n"
        )
        log_path = self.run_real(directory, code)
        self.run_real(directory, code)

        text = log_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("to stdout"), 2, text)
        self.assertEqual(text.count("to stderr"), 2, text)
        # No stdin: a worker that asks for input gets end of file, not a hang.
        self.assertEqual(text.count("stdin is ''"), 2, text)

    def test_a_name_outside_the_ansi_code_page_does_not_crash_the_worker(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # The caller's environment forces the Windows file encoding on every platform.
        environment = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
        code = "print('\\u03b1-amylase \\u96ea')\n"

        log_path = self.run_real(directory, code, env=environment)

        self.assertIn("α-amylase 雪", log_path.read_text(encoding="utf-8"))

    def test_output_is_in_the_log_while_the_command_still_runs(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = directory / "stand_in.py"
        script.write_text("import time\nprint('started')\ntime.sleep(60)\n", encoding="utf-8")
        log_path = directory / "worker.log"
        process = launcher.launch_without_window(
            [sys.executable, str(script)], cwd=directory, log_path=log_path
        )
        self.addCleanup(self.stop_tree, process)

        deadline = time.monotonic() + 30
        while "started" not in (
            log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        ):
            self.assertLess(time.monotonic(), deadline, "The log stayed empty while the command ran")
            time.sleep(0.05)
        self.assertIsNone(process.poll())

    @unittest.skipUnless(sys.platform == "win32", "Windows consoles")
    def test_no_console_window_exists_for_the_command_or_a_process_it_starts(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # The real interpreter behind a venv's redirector, and a child of its
        # own: with a hidden console there is no window to show.
        code = (
            "import ctypes, subprocess, sys\n"
            "probe = 'import ctypes; print(ctypes.windll.kernel32.GetConsoleWindow())'\n"
            "child = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)\n"
            "print('command', ctypes.windll.kernel32.GetConsoleWindow())\n"
            "print('child', child.stdout.strip())\n"
        )

        log_path = self.run_real(directory, code)

        lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["command 0", "child 0"])


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
