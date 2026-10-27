"""Launcher scripts (install.sh/.bat and src/bin launchers): the desktop launcher's start without a terminal and
its terminal fallbacks, the macOS app bundles, shared environment
sanitizing and setup-lock policy, the managed-venv interpreter probe, the
existing-instance probe that runs before dependency validation, the desktop
launcher structure, the POSIX dependency-setup lock, and how many readiness
checks a setup runs."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
import venv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
INSTALLER = PROJECT_ROOT / "install.sh"
MANAGED_LAUNCHERS = ("EMAPSSN.bat", "EMAPSSN_Tools.bat", "EMAPSSN.sh", "EMAPSSN_Tools.sh")


def run_interpreter_probe(probe, version_info):
    """Run a launcher's `-c` interpreter probe as the given Python version would."""
    fake_sys = types.SimpleNamespace(version_info=version_info)

    def fake_exit(code=0):
        raise SystemExit(code)

    fake_sys.exit = fake_exit
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        return fake_sys if name == "sys" else real_import(name, *args, **kwargs)

    try:
        exec(probe, {"__builtins__": {**vars(builtins), "__import__": fake_import}})
    except SystemExit as stop:
        return stop.code
    return 0


class LauncherScriptPolicyTests(unittest.TestCase):
    def test_existing_launchers_share_environment_and_lock_policy(self):
        sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in (
                PROJECT_ROOT / "install.sh",
                SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.sh",
                SRC_DIR / "bin" / "EMAPSSN.sh",
                SRC_DIR / "bin" / "EMAPSSN_Tools.sh",
                SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.bat",
                SRC_DIR / "bin" / "EMAPSSN.bat",
                SRC_DIR / "bin" / "EMAPSSN_Tools.bat",
            )
        }

        self.assertIn("ssn_sanitize_managed_environment", sources["install.sh"])
        self.assertIn(
            "ssn_sanitize_managed_environment", sources["EMAPSSN_Desktop_Launcher.sh"]
        )
        self.assertIn(":SANITIZE_MANAGED_ENVIRONMENT", sources["EMAPSSN_Desktop_Launcher.bat"])
        for name in ("EMAPSSN.bat", "EMAPSSN_Tools.bat"):
            self.assertIn("dependency_setup.lock", sources[name])
            self.assertIn("--locked-setup", sources[name])
            self.assertIn('"%COMSPEC%" /d /c', sources[name])

    def test_launchers_recreate_a_venv_that_is_not_the_managed_python(self):
        # An in-place upgrade kept a working Python 3.12 .venv: the probes only
        # ran `import sys`, so any interpreter passed and the new pins were
        # installed into it instead of a recreated Python 3.13 environment.
        for name in MANAGED_LAUNCHERS:
            source = (SRC_DIR / "bin" / name).read_text(encoding="utf-8")
            with self.subTest(launcher=name):
                self.assertNotIn('-c "import sys"', source)
                self.assertIn(
                    '-c "!VENV_PROBE!"' if name.endswith(".bat") else '-c "$VENV_PROBE"',
                    source,
                )
                probe = re.search(r"VENV_PROBE='?(import sys;[^'\"\r\n]*)", source).group(1)
                created = tuple(
                    int(part)
                    for part in re.search(
                        r"venv --clear --python (\d+)\.(\d+)", source
                    ).groups()
                )

                self.assertEqual(run_interpreter_probe(probe, created + (0,)), 0)
                self.assertEqual(run_interpreter_probe(probe, (3, 12, 3)), 1)
                self.assertEqual(run_interpreter_probe(probe, (3, 14, 0)), 1)
                # Windows delayed expansion would strip a literal "!" from the probe.
                self.assertNotIn("!", probe)

    def test_launchers_probe_for_existing_windows_before_validation(self):
        paths = (
            SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.bat",
            SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.sh",
            SRC_DIR / "bin" / "EMAPSSN.bat",
            SRC_DIR / "bin" / "EMAPSSN.sh",
            SRC_DIR / "bin" / "EMAPSSN_Tools.bat",
            SRC_DIR / "bin" / "EMAPSSN_Tools.sh",
        )
        for path in paths:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("Single_Instance_Probe.py", source)
                validation_markers = (
                    "Detecting hardware and validating dependencies",
                    "ssn_require_linux_gui_dependencies",
                    "where uv",
                    "command -v uv",
                )
                validation_position = min(
                    source.index(marker)
                    for marker in validation_markers
                    if marker in source
                )
                probe_position = source.index(
                    "call :ACTIVATE_EXISTING_INSTANCE"
                    if path.suffix == ".bat"
                    else "activate_existing_instance()"
                )
                self.assertLess(
                    probe_position,
                    validation_position,
                )

    def test_shell_scripts_are_committed_executable(self):
        # install.sh runs `chmod +x src/bin/*.sh`. A script committed without
        # the executable bit then shows as modified, which can block a later
        # `git pull` (found on a fresh Linux clone). Windows checkouts keep
        # core.fileMode false, so git takes the bit from the index, never from
        # the file: set it with `git update-index --chmod=+x`.
        try:
            listed = subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "ls-files", "-s", "--", "*.sh"],
                capture_output=True, text=True,
            )
        except OSError:
            self.skipTest("git is not installed")
        modes = {}
        for line in listed.stdout.splitlines() if listed.returncode == 0 else ():
            entry, path = line.split("\t", 1)
            modes[path] = entry.split()[0]
        if modes.get("install.sh") != "100755":
            # A snapshot whose index came from `git add` on Windows has no modes.
            self.skipTest("this index does not carry the clone's file modes")
        self.assertIn("src/bin/EMAPSSN.sh", modes)
        self.assertEqual({path: mode for path, mode in modes.items() if mode != "100755"}, {})


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class PosixSetupLockTests(unittest.TestCase):
    def _run_bash(self, program: str, *, env=None):
        return subprocess.run(
            ["/bin/bash", "-c", program],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_shell_sanitizer_preserves_supported_overrides(self):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHOME": "/external/python",
                "PYTHONPATH": "/external/modules",
                "QT_PLUGIN_PATH": "/external/qt/plugins",
                "QT_QPA_PLATFORM_PLUGIN_PATH": "/external/qt/platforms",
                "QML_IMPORT_PATH": "/external/qml",
                "QML2_IMPORT_PATH": "/external/qml2",
                "CONDA_PREFIX": "/conda/base",
                "QT_QPA_PLATFORM": "wayland",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
        removed = (
            "PYTHONHOME",
            "PYTHONPATH",
            "QT_PLUGIN_PATH",
            "QT_QPA_PLATFORM_PLUGIN_PATH",
            "QML_IMPORT_PATH",
            "QML2_IMPORT_PATH",
        )
        preserved = ("CONDA_PREFIX", "QT_QPA_PLATFORM", "CUDA_VISIBLE_DEVICES")
        program = (
            f". {shlex.quote(str(INSTALLER))}\n"
            "ssn_sanitize_managed_environment\n"
            + "\n".join(
                f"printf '%s=%s\\n' {name} \"${{{name}-}}\""
                for name in (*removed, *preserved)
            )
        )

        result = self._run_bash(program, env=environment)

        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines())
        for name in removed:
            self.assertEqual(values[name], "")
        self.assertEqual(values["CONDA_PREFIX"], "/conda/base")
        self.assertEqual(values["QT_QPA_PLATFORM"], "wayland")
        self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "0")

    def test_setup_lock_serializes_contenders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events"
            owner_program = "\n".join(
                (
                    f". {shlex.quote(str(INSTALLER))}",
                    f"ssn_acquire_dependency_setup_lock {shlex.quote(str(root))} || exit 1",
                    f"printf 'first\\n' >> {shlex.quote(str(events))}",
                    "sleep 1",
                    f"printf 'first_done\\n' >> {shlex.quote(str(events))}",
                    "ssn_release_dependency_setup_lock",
                )
            )
            contender_program = "\n".join(
                (
                    f". {shlex.quote(str(INSTALLER))}",
                    f"ssn_acquire_dependency_setup_lock {shlex.quote(str(root))} || exit 1",
                    f"printf 'second\\n' >> {shlex.quote(str(events))}",
                    "ssn_release_dependency_setup_lock",
                )
            )
            owner = subprocess.Popen(
                ["/bin/bash", "-c", owner_program],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + 5
            while not events.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(events.exists(), "first contender did not acquire the lock")

            contender = self._run_bash(contender_program)
            owner_stdout, owner_stderr = owner.communicate(timeout=5)

            self.assertEqual(owner.returncode, 0, owner_stderr or owner_stdout)
            self.assertEqual(contender.returncode, 0, contender.stderr)
            self.assertEqual(
                events.read_text(encoding="utf-8").splitlines(),
                ["first", "first_done", "second"],
            )
            self.assertIn("waiting", contender.stdout)

    def test_stale_lock_is_reclaimed_and_exit_trap_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_dir = (
                root / "temp" / "dependency_setup.lock"
            )
            lock_dir.mkdir(parents=True)
            (lock_dir / "owner.pid").write_text("99999999\n", encoding="utf-8")
            reclaim = "\n".join(
                (
                    f". {shlex.quote(str(INSTALLER))}",
                    f"ssn_acquire_dependency_setup_lock {shlex.quote(str(root))} || exit 1",
                    "ssn_release_dependency_setup_lock",
                )
            )

            result = self._run_bash(reclaim)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Recovered stale dependency setup lock", result.stdout)
            self.assertFalse(lock_dir.exists())

            trapped_exit = "\n".join(
                (
                    f". {shlex.quote(str(INSTALLER))}",
                    "ssn_enable_desktop_failure_pause",
                    f"ssn_acquire_dependency_setup_lock {shlex.quote(str(root))} || exit 1",
                    "exit 7",
                )
            )
            trapped = self._run_bash(trapped_exit)
            self.assertEqual(trapped.returncode, 7)
            self.assertFalse(lock_dir.exists())


class LauncherStructureTests(unittest.TestCase):
    def test_the_installers_and_launcher_scripts_contain_their_entries_flags_and_markers(self):
        install_bat = (PROJECT_ROOT / "install.bat").read_text(encoding="utf-8")
        install_sh = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
        shell_supervisor = (SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.sh").read_text(encoding="utf-8")
        windows_supervisor = (SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.bat").read_text(encoding="utf-8")
        direct_windows = (SRC_DIR / "bin" / "EMAPSSN.bat").read_text(encoding="utf-8")
        direct_posix = (SRC_DIR / "bin" / "EMAPSSN.sh").read_text(encoding="utf-8")

        self.assertIn("cmd.exe", install_bat)
        self.assertIn("EMAPSSN_Desktop_Launcher.bat", install_bat)
        self.assertNotIn("wscript.exe", install_bat)
        self.assertEqual(install_sh.count("Terminal=false"), 2)
        self.assertIn("EMAPSSN_Desktop_Launcher.sh", install_sh)
        self.assertIn("EMAP-SSN.app", install_sh)
        self.assertIn("EMAP-SSN Tools.app", install_sh)
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
        self.assertIn("--launch-and-wait", shell_supervisor)
        self.assertIn("--launch-and-wait", windows_supervisor)
        self.assertNotIn("nohup", shell_supervisor)
        self.assertNotIn("start \"\" /b", windows_supervisor)
        self.assertNotIn("pythonw.exe", windows_supervisor)
        self.assertIn("Starting EMAPSSN_Config", direct_windows)
        self.assertIn("Starting EMAPSSN_Config", direct_posix)
        self.assertNotIn("terminal.dismissed", direct_windows)
        self.assertNotIn("terminal.dismissed", direct_posix)

    def test_the_desktop_launcher_has_a_start_without_a_terminal_and_keeps_the_terminal_as_the_fallback(self):
        shell_supervisor = (SRC_DIR / "bin" / "EMAPSSN_Desktop_Launcher.sh").read_text(encoding="utf-8")
        self.assertIn("src/desktop/Desktop_Launcher.py", shell_supervisor)
        self.assertIn("start_without_terminal", shell_supervisor)
        self.assertIn("--console", shell_supervisor)
        self.assertIn("--open-terminal", shell_supervisor)  # the older name of --console
        # The check and the terminal session of the old start are still there, in the same order.
        self.assertLess(shell_supervisor.index("start_without_terminal"), shell_supervisor.index("open_startup_terminal\n    exit"))
        self.assertLess(shell_supervisor.index("open_startup_terminal\n    exit"), shell_supervisor.index("--check-only"))

    def test_the_mac_app_runs_the_python_launcher_and_keeps_terminal_as_its_fallback(self):
        install_sh = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")
        self.assertIn('-u "\\$LAUNCHER" $app_kind --caller-opens-console', install_sh)
        self.assertIn("exec /usr/bin/open -a Terminal", install_sh)
        self.assertIn("<key>LSUIElement</key><true/>", install_sh)
        self.assertNotIn("NSAppleEventsUsageDescription", install_sh)


# The Python launcher, the probe and the monitor are stand-ins that record how they were run. The stand-in for the
# managed Python tells them apart by what the script under test passes it.
_RECORDING_PYTHON = """\
#!/bin/bash
log=$SSN_TEST_PYTHON_LOG
case "$1" in
    -u)
        case "$2" in
            */Desktop_Launcher.py)
                printf 'launcher|%s|cwd=%s|PYTHONHOME=%s|PYTHONPATH=%s|QT_QPA_PLATFORM=%s|PATH=%s\\n' \\
                    "$*" "$PWD" "${PYTHONHOME-unset}" "${PYTHONPATH-unset}" "${QT_QPA_PLATFORM-unset}" "$PATH" >>"$log"
                exit "${SSN_TEST_LAUNCHER_EXIT:-0}" ;;
            */Desktop_Launcher_Monitor.py)
                printf 'monitor|%s\\n' "$*" >>"$log"
                exit "${SSN_TEST_MONITOR_EXIT:-0}" ;;
        esac ;;
    */Single_Instance_Probe.py)
        printf 'probe|%s\\n' "$*" >>"$log"
        exit "${SSN_TEST_PROBE_EXIT:-1}" ;;
esac
printf 'other|%s\\n' "$*" >>"$log"
exit 0
"""

# A terminal emulator that only records the command it was given (one argument per NUL-ended item).
_RECORDING_TERMINAL = """\
#!/bin/bash
printf '%s\\0' "$@" >"$SSN_TEST_TERMINAL_LOG"
"""


def _lf_bytes(path):
    """A file's bytes with LF line endings, as a Linux checkout has them (a Windows one may hold CRLF)."""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class DesktopLauncherScriptTests(unittest.TestCase):
    """EMAPSSN_Desktop_Launcher.sh with its own logic, over a scratch project whose Python launcher is a stand-in.

    The default mode starts the Python launcher with no terminal; a terminal opens (the fake emulator records the
    command it gets) only when the managed Python or the launcher script is missing, with --console, and when the
    launcher ends with anything but 0, which includes its exit code 3, "open the console".
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "my project"  # a space: every path in the script is quoted
        (self.root / "src" / "bin").mkdir(parents=True)
        (self.root / "src" / "desktop").mkdir()
        for relative in ("src/bin/EMAPSSN_Desktop_Launcher.sh", "src/bin/EMAPSSN_Terminal_Launcher.sh", "install.sh"):
            (self.root / relative).write_bytes(_lf_bytes(PROJECT_ROOT / relative))
        for relative in ("src/desktop/Desktop_Launcher.py", "src/desktop/Single_Instance_Probe.py"):
            (self.root / relative).write_text("# stand-in\n", encoding="utf-8")
        self.python = self.root / ".venv" / "bin" / "python"
        self.python.parent.mkdir(parents=True)
        self.python.write_text(_RECORDING_PYTHON, encoding="utf-8", newline="\n")
        self.python.chmod(0o755)
        self.fake_bin = base / "fake-bin"
        self.fake_bin.mkdir()
        # The script must not reach a real terminal emulator or, on a Mac, Terminal through osascript: it believes
        # it is on Linux, and the first emulator it finds is the recording one.
        (self.fake_bin / "uname").write_text("#!/bin/bash\nprintf Linux\n", encoding="utf-8", newline="\n")
        (self.fake_bin / "xdg-terminal-exec").write_text(_RECORDING_TERMINAL, encoding="utf-8", newline="\n")
        for name in ("uname", "xdg-terminal-exec"):
            (self.fake_bin / name).chmod(0o755)
        self.python_log = base / "python.log"
        self.terminal_log = base / "terminal.argv"

    def run_script(self, *arguments, launcher_exit=0, probe_exit=1, environment=None, expect_terminal=True):
        """Run the script; returns (result, the Python calls, the terminal's argument list or None).

        The fake emulator is started in the background, so a run that expects one waits for it to record its
        command; a run that does not gives it a moment to show up wrongly.
        """
        env = os.environ.copy()
        env.pop("QT_QPA_PLATFORM", None)
        env.update({
            "PATH": f"{self.fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "SSN_TEST_PYTHON_LOG": str(self.python_log),
            "SSN_TEST_TERMINAL_LOG": str(self.terminal_log),
            "SSN_TEST_LAUNCHER_EXIT": str(launcher_exit),
            "SSN_TEST_PROBE_EXIT": str(probe_exit),
        })
        env.update(environment or {})
        result = subprocess.run(
            ["/bin/bash", str(self.root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.sh"), *arguments],
            check=False, capture_output=True, text=True, env=env, cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL,
            timeout=60,
        )
        calls = self.python_log.read_text(encoding="utf-8").splitlines() if self.python_log.exists() else []
        deadline = time.monotonic() + (3 if expect_terminal else 0.3)
        while not self.terminal_log.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        terminal = None
        if self.terminal_log.exists():
            time.sleep(0.05)
            terminal = [item.decode("utf-8") for item in self.terminal_log.read_bytes().split(b"\0") if item]
        return result, calls, terminal

    def kinds_of(self, calls):
        return [call.split("|", 1)[0] for call in calls]

    def assert_terminal_session(self, terminal, kind, title):
        """The terminal runs the .sh again in its --terminal-session mode, in the project folder, with the window's title."""
        script = str(self.root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.sh")
        self.assertEqual(terminal[0], "--")
        self.assertEqual(terminal[1:3], ["bash", "-c"])
        self.assertEqual(terminal[-5:], [str(self.root), title, script, kind, "--terminal-session"])

    # --- the start without a terminal
    def test_a_launcher_that_handled_the_start_leaves_no_terminal_and_no_other_work(self):
        for kind in ("viewer", "tools"):
            with self.subTest(kind=kind):
                self.python_log.unlink(missing_ok=True)
                result, calls, terminal = self.run_script(kind, expect_terminal=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertEqual(self.kinds_of(calls), ["launcher"])  # not even the existing-instance probe
                launcher = str(self.root / "src" / "desktop" / "Desktop_Launcher.py")
                self.assertEqual(calls[0].split("|")[1], f"-u {launcher} {kind}", calls[0])
                self.assertIsNone(terminal)

    def test_the_launcher_runs_in_the_project_folder_with_a_clean_managed_environment(self):
        result, calls, _ = self.run_script(
            "viewer", environment={"PYTHONHOME": "/x", "PYTHONPATH": "/y", "XDG_SESSION_TYPE": "wayland"},
            expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = dict(part.split("=", 1) for part in calls[0].split("|")[2:])
        self.assertEqual(os.path.realpath(fields["cwd"]), os.path.realpath(self.root))
        self.assertEqual((fields["PYTHONHOME"], fields["PYTHONPATH"]), ("unset", "unset"))
        # The splash is plain widgets and runs on Wayland itself: nothing here moves it to XWayland (the monitor
        # does that for the window it starts).
        self.assertEqual(fields["QT_QPA_PLATFORM"], "unset")

    def test_a_platform_the_user_chose_reaches_the_launcher(self):
        result, calls, _ = self.run_script("tools", environment={"QT_QPA_PLATFORM": "xcb"}, expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QT_QPA_PLATFORM=xcb", calls[0])

    # --- the fallbacks: a terminal
    def test_the_launchers_console_code_and_any_other_failure_open_the_terminal(self):
        for code in (3, 1, 2, 134, 139):  # EXIT_FALLBACK, a Python error, a bad option, an abort, a crash
            with self.subTest(code=code):
                self.python_log.unlink(missing_ok=True)
                self.terminal_log.unlink(missing_ok=True)
                result, calls, terminal = self.run_script("tools", launcher_exit=code)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.kinds_of(calls), ["launcher", "probe"])  # then the old start: probe, terminal
                self.assertIn(f"exit code {code}", result.stderr)
                self.assertIsNotNone(terminal, "no terminal was opened")
                self.assert_terminal_session(terminal, "tools", "EMAP-SSN Tools")

    def test_a_window_that_is_open_after_the_launcher_failed_is_not_answered_with_a_terminal(self):
        result, calls, terminal = self.run_script("viewer", launcher_exit=3, probe_exit=0, expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kinds_of(calls), ["launcher", "probe"])
        self.assertIsNone(terminal)

    def test_with_no_managed_python_yet_the_terminal_does_the_setup_and_the_launcher_is_not_tried(self):
        self.python.unlink()
        result, calls, terminal = self.run_script("viewer")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])
        self.assertNotIn("exit code", result.stderr)  # the launcher was not tried, so it did not fail either
        self.assertIsNotNone(terminal)
        self.assert_terminal_session(terminal, "viewer", "EMAP-SSN Configuration")

    def test_a_python_that_is_not_executable_is_no_managed_python(self):
        self.python.chmod(0o644)
        result, calls, terminal = self.run_script("viewer")
        self.assertEqual(calls, [])
        self.assertNotIn("exit code", result.stderr)
        self.assertIsNotNone(terminal)

    def test_a_missing_launcher_script_goes_to_the_terminal_too(self):
        (self.root / "src" / "desktop" / "Desktop_Launcher.py").unlink()
        result, calls, terminal = self.run_script("viewer")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kinds_of(calls), ["probe"])
        self.assertIsNotNone(terminal)

    def test_console_opens_the_terminal_and_never_tries_the_launcher(self):
        for option in ("--console", "--open-terminal"):  # the second is the older name
            with self.subTest(option=option):
                self.python_log.unlink(missing_ok=True)
                self.terminal_log.unlink(missing_ok=True)
                result, calls, terminal = self.run_script("viewer", option)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.kinds_of(calls), ["probe"])
                self.assertIsNotNone(terminal)
                self.assert_terminal_session(terminal, "viewer", "EMAP-SSN Configuration")

    def test_console_still_brings_an_open_window_forward_instead(self):
        result, calls, terminal = self.run_script("viewer", "--console", probe_exit=0, expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kinds_of(calls), ["probe"])
        self.assertIsNone(terminal)

    def test_a_machine_with_no_terminal_emulator_reports_it_and_fails(self):
        (self.fake_bin / "xdg-terminal-exec").unlink()
        # Only the standard tools are on the path: no emulator of any kind.
        tools = Path(tempfile.mkdtemp(prefix="ssn-tools-"))
        self.addCleanup(shutil.rmtree, tools, True)
        (tools / "dirname").symlink_to(shutil.which("dirname"))
        (tools / "uname").write_text("#!/bin/bash\nprintf Linux\n", encoding="utf-8", newline="\n")
        (tools / "uname").chmod(0o755)
        result, calls, terminal = self.run_script(
            "viewer", "--console", environment={"PATH": str(tools)}, expect_terminal=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No supported terminal emulator was found", result.stderr)
        self.assertIsNone(terminal)

    # --- usage, and the session the terminal runs
    def test_an_unknown_kind_or_mode_is_a_usage_error_naming_the_options(self):
        result, calls, terminal = self.run_script("elsewhere")
        self.assertEqual(result.returncode, 2)
        self.assertIn("viewer|tools [--console|--terminal-session]", result.stderr)
        result, calls, terminal = self.run_script("viewer", "--nonsense")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown desktop launcher mode: --nonsense", result.stderr)
        self.assertEqual(calls, [])
        self.assertIsNone(terminal)

    def test_the_terminal_session_checks_then_runs_the_monitor_as_it_always_did(self):
        # What the terminal runs: the kind's portable launcher checks (stand-in), then the monitor starts the window.
        portable = self.root / "src" / "bin" / "EMAPSSN.sh"
        portable.write_text(
            '#!/bin/bash\nprintf "check|%s\\n" "$*" >>"$SSN_TEST_PYTHON_LOG"\nexit 0\n', encoding="utf-8", newline="\n")
        portable.chmod(0o755)
        result, calls, terminal = self.run_script("viewer", "--terminal-session", expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.kinds_of(calls), ["probe", "check", "probe", "monitor"])
        self.assertIn("--launch-and-wait viewer", calls[-1])
        self.assertIn("EMAP-SSN Configuration is ready.", result.stdout)
        self.assertIsNone(terminal)

    def test_the_terminal_session_that_needs_a_setup_runs_it_in_the_terminal(self):
        portable = self.root / "src" / "bin" / "EMAPSSN.sh"
        portable.write_text(
            '#!/bin/bash\nprintf "portable|%s\\n" "$*" >>"$SSN_TEST_PYTHON_LOG"\n'
            '[ "$1" = "--check-only" ] && exit "${SSN_TEST_CHECK_EXIT:-0}"\nexit 0\n',
            encoding="utf-8", newline="\n")
        portable.chmod(0o755)
        result, calls, _ = self.run_script(
            "viewer", "--terminal-session", environment={"SSN_TEST_CHECK_EXIT": "10"}, expect_terminal=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call for call in calls if call.startswith("portable")],
                         ["portable|--check-only", "portable|--setup-only"])
        self.assertIn("Setup or repair is required.", result.stdout)


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class MacosAppBundleTests(unittest.TestCase):
    """The .app bundles install.sh writes: what they contain, and how the executable behaves over a scratch project.

    Nothing here has run on a Mac. The executable is a bash script, and the tests run it under bash with the
    managed Python and /usr/bin/open replaced by stand-ins that record how they were called.
    """

    KINDS = (("EMAP-SSN", "viewer", "ca.utoronto.emapssn"), ("EMAP-SSN Tools", "tools", "ca.utoronto.emapssn.tools"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "my project"  # the .apps live in the project root
        self.root.mkdir()
        (self.root / "install.sh").write_bytes(_lf_bytes(PROJECT_ROOT / "install.sh"))
        self.log = self.base / "calls.log"

    def build(self, name, kind, bundle_id):
        done = subprocess.run(
            ["/bin/bash", "-c", f'. "{self.root}/install.sh" && ssn_create_macos_app_launcher "{self.root}" "{name}" "{kind}" "{bundle_id}"'],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        return self.root / f"{name}.app"

    def stand_ins(self, app, *, python=True, launcher=True):
        """A managed Python and Desktop_Launcher.py that record their calls, and an `open` the executable runs instead of /usr/bin/open."""
        executable = app / "Contents" / "MacOS" / "launcher"
        text = executable.read_text(encoding="utf-8")
        fake_open = self.base / "open"
        fake_open.write_text('#!/bin/bash\nprintf \'open|%s\\n\' "$*" >>"$SSN_TEST_LOG"\n', encoding="utf-8", newline="\n")
        fake_open.chmod(0o755)
        self.assertIn("/usr/bin/open", text)
        executable.write_text(text.replace("/usr/bin/open", str(fake_open)), encoding="utf-8", newline="\n")
        if python:
            interpreter = self.root / ".venv" / "bin" / "python"
            interpreter.parent.mkdir(parents=True, exist_ok=True)
            interpreter.write_text(
                '#!/bin/bash\nprintf \'python|%s|cwd=%s|PYTHONHOME=%s|PYTHONPATH=%s|DYLD_LIBRARY_PATH=%s|PATH=%s\\n\' '
                '"$*" "$PWD" "${PYTHONHOME-unset}" "${PYTHONPATH-unset}" "${DYLD_LIBRARY_PATH-unset}" "$PATH" >>"$SSN_TEST_LOG"\n'
                'exit "${SSN_TEST_LAUNCHER_EXIT:-0}"\n',
                encoding="utf-8", newline="\n")
            interpreter.chmod(0o755)
        if launcher:
            (self.root / "src" / "desktop").mkdir(parents=True, exist_ok=True)
            (self.root / "src" / "desktop" / "Desktop_Launcher.py").write_text("# stand-in\n", encoding="utf-8")
        return executable

    def run_executable(self, executable, *, launcher_exit=0, environment=None):
        env = os.environ.copy()
        env.update({"SSN_TEST_LOG": str(self.log), "SSN_TEST_LAUNCHER_EXIT": str(launcher_exit)})
        env.update(environment or {})
        done = subprocess.run(["/bin/bash", str(executable)], check=False, capture_output=True, text=True, env=env,
                              cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL, timeout=60)
        calls = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return done, calls

    def test_the_bundles_are_written_for_each_window_with_the_dock_hidden_and_no_apple_events(self):
        for name, kind, bundle_id in self.KINDS:
            with self.subTest(app=name):
                app = self.build(name, kind, bundle_id)
                plist = (app / "Contents" / "Info.plist").read_text(encoding="utf-8")
                self.assertIn("<key>CFBundleExecutable</key><string>launcher</string>", plist)
                self.assertIn(f"<key>CFBundleIdentifier</key><string>{bundle_id}</string>", plist)
                self.assertIn("<key>LSUIElement</key><true/>", plist)  # the bundle itself has no Dock icon
                self.assertNotIn("NSAppleEventsUsageDescription", plist)
                self.assertIn("<key>LSMinimumSystemVersion</key><string>14.0</string>", plist)
                executable = app / "Contents" / "MacOS" / "launcher"
                self.assertTrue(os.access(executable, os.X_OK))
                self.assertTrue(os.access(app / "Contents" / "Resources" / "start.command", os.X_OK))
                text = executable.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("#!/bin/bash\n"))
                self.assertIn(f'-u "$LAUNCHER" {kind} --caller-opens-console', text)
                self.assertIn('exec /usr/bin/open -a Terminal "$APP_ROOT/Contents/Resources/start.command"', text)
                self.assertNotIn("osascript", text)
                for script in (executable, app / "Contents" / "Resources" / "start.command"):
                    self.assertEqual(subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True).returncode, 0)
                # The command file is what it was: the .sh's terminal session, for this window.
                self.assertIn(f'"$PROJECT_ROOT/src/bin/EMAPSSN_Desktop_Launcher.sh" "{kind}" --terminal-session',
                              (app / "Contents" / "Resources" / "start.command").read_text(encoding="utf-8"))

    def test_a_launcher_that_handled_the_start_leaves_terminal_alone(self):
        for name, kind, bundle_id in self.KINDS:
            with self.subTest(app=name):
                self.log.unlink(missing_ok=True)
                app = self.build(name, kind, bundle_id)
                executable = self.stand_ins(app)
                done, calls = self.run_executable(
                    executable, environment={"PYTHONHOME": "/x", "PYTHONPATH": "/y", "DYLD_LIBRARY_PATH": "/z"})
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual([call.split("|")[0] for call in calls], ["python"])
                launcher = self.root / "src" / "desktop" / "Desktop_Launcher.py"
                fields = calls[0].split("|")
                self.assertEqual(fields[1], f"-u {launcher} {kind} --caller-opens-console")
                values = dict(part.split("=", 1) for part in fields[2:])
                self.assertEqual(os.path.realpath(values["cwd"]), os.path.realpath(self.root))
                for name_ in ("PYTHONHOME", "PYTHONPATH", "DYLD_LIBRARY_PATH"):
                    self.assertEqual(values[name_], "unset")
                # Finder's short PATH gets the Homebrew folders added after it, for a uv installed there.
                self.assertTrue(values["PATH"].endswith(":/opt/homebrew/bin:/usr/local/bin"), values["PATH"])

    def test_a_launcher_that_asks_for_the_console_or_fails_has_terminal_open_the_command_file(self):
        for code in (3, 1, 2, 134, 139):
            with self.subTest(code=code):
                self.log.unlink(missing_ok=True)
                app = self.build("EMAP-SSN", "viewer", "ca.utoronto.emapssn")
                executable = self.stand_ins(app)
                done, calls = self.run_executable(executable, launcher_exit=code)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual([call.split("|")[0] for call in calls], ["python", "open"])
                self.assertEqual(calls[1], f"open|-a Terminal {app}/Contents/Resources/start.command")

    def test_with_no_managed_python_or_no_launcher_script_terminal_opens_at_once(self):
        for python, launcher in ((False, True), (True, False)):
            with self.subTest(python=python, launcher=launcher):
                self.log.unlink(missing_ok=True)
                shutil.rmtree(self.root / ".venv", ignore_errors=True)
                shutil.rmtree(self.root / "src", ignore_errors=True)
                app = self.build("EMAP-SSN Tools", "tools", "ca.utoronto.emapssn.tools")
                executable = self.stand_ins(app, python=python, launcher=launcher)
                done, calls = self.run_executable(executable)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(calls, [f"open|-a Terminal {app}/Contents/Resources/start.command"])

    def test_the_command_file_runs_the_terminal_session_of_the_shell_launcher(self):
        app = self.build("EMAP-SSN Tools", "tools", "ca.utoronto.emapssn.tools")
        script = self.root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.sh"
        script.parent.mkdir(parents=True)
        script.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >"$SSN_TEST_LOG"\n', encoding="utf-8", newline="\n")
        script.chmod(0o755)
        done, calls = self.run_executable(app / "Contents" / "Resources" / "start.command")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(calls, ["tools --terminal-session"])


# Stand-ins for the scripts a launcher runs; each call is recorded in a log.
_STUB_INSTALLER = """\
import os
import sys

checking = "--check-only" in sys.argv
with open(os.environ["SSN_LAUNCHER_TEST_LOG"], "a", encoding="utf-8") as log:
    log.write("check\\n" if checking else "install\\n")
print("stub readiness report" if checking else "stub installer report")
sys.exit(int(os.environ["SSN_LAUNCHER_TEST_CHECK_EXIT"]) if checking else 0)
"""
_STUB_APPLICATION = """\
import os

with open(os.environ["SSN_LAUNCHER_TEST_LOG"], "a", encoding="utf-8") as log:
    log.write("app\\n")
"""
_STUB_INSTANCE_PROBE = "raise SystemExit(1)\n"


class _ReadinessCheckCountMixin:
    """Run the real launchers against stand-in scripts and count readiness checks.

    Each Install_Dependencies.py --check-only detects the hardware again and
    prints its report. A desktop launch whose saved state had changed printed
    that report three times after "Setup or repair is required": --setup-only
    repeated the desktop launcher's check, the Windows --locked-setup child
    repeated its parent's, and the re-check under the setup lock printed it too.
    """

    LAUNCHERS = ()  # (launcher file name, application script) pairs

    @classmethod
    def _build_root(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        (cls.root / "src" / "bin").mkdir(parents=True)
        (cls.root / "src" / "desktop").mkdir()
        (cls.root / "stub-bin").mkdir()
        for launcher, application in cls.LAUNCHERS:
            (cls.root / "src" / "bin" / launcher).write_bytes(
                (SRC_DIR / "bin" / launcher).read_bytes()
            )
            (cls.root / "src" / application).write_text(_STUB_APPLICATION, encoding="utf-8")
        (cls.root / "src" / "Install_Dependencies.py").write_text(
            _STUB_INSTALLER, encoding="utf-8"
        )
        (cls.root / "src" / "desktop" / "Single_Instance_Probe.py").write_text(
            _STUB_INSTANCE_PROBE, encoding="utf-8"
        )

    def _launch(self, launcher, mode, *, check_exit):
        log = self.root / "launcher.log"
        log.unlink(missing_ok=True)
        environment = os.environ.copy()
        environment["PATH"] = str(self.root / "stub-bin") + os.pathsep + environment["PATH"]
        environment["SSN_LAUNCHER_TEST_LOG"] = str(log)
        environment["SSN_LAUNCHER_TEST_CHECK_EXIT"] = str(check_exit)
        result = subprocess.run(
            self._command(launcher, mode),
            cwd=self.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=180,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertFalse((self.root / "temp" / "dependency_setup.lock").exists(), output)
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return output, calls

    def test_setup_only_checks_once_quietly_before_installing(self):
        for launcher, _application in self.LAUNCHERS:
            with self.subTest(launcher=launcher):
                output, calls = self._launch(launcher, "--setup-only", check_exit=10)

                self.assertEqual(calls, ["check", "install"], output)
                self.assertNotIn("stub readiness report", output)
                self.assertIn("stub installer report", output)

    def test_setup_only_stops_when_the_environment_became_ready(self):
        for launcher, _application in self.LAUNCHERS:
            with self.subTest(launcher=launcher):
                output, calls = self._launch(launcher, "--setup-only", check_exit=0)

                self.assertEqual(calls, ["check"], output)
                self.assertIn("Dependency setup was completed by another launcher.", output)

    def test_direct_launch_reports_its_failed_check_once(self):
        for launcher, _application in self.LAUNCHERS:
            with self.subTest(launcher=launcher):
                output, calls = self._launch(launcher, "", check_exit=10)

                # The fast-path check, then the re-check under the setup lock.
                self.assertEqual(calls, ["check", "check", "install", "app"], output)
                self.assertEqual(output.count("stub readiness report"), 1, output)
                self.assertEqual(output.count("stub installer report"), 1, output)


@unittest.skipUnless(os.name == "nt", "runs the Windows batch launchers")
class WindowsReadinessCheckCountTests(_ReadinessCheckCountMixin, unittest.TestCase):
    LAUNCHERS = (("EMAPSSN.bat", "EMAPSSN_Config.py"), ("EMAPSSN_Tools.bat", "EMAPSSN_Tools.py"))

    @classmethod
    def setUpClass(cls):
        source = (SRC_DIR / "bin" / "EMAPSSN.bat").read_text(encoding="utf-8")
        managed = tuple(
            int(part)
            for part in re.search(r"venv --clear --python (\d+)\.(\d+)", source).groups()
        )
        if sys.version_info[:2] != managed:
            raise unittest.SkipTest("the launchers' interpreter probe rejects this Python")
        cls._build_root()
        # The launchers run .venv\Scripts\python.exe, so it must be a real interpreter.
        venv.EnvBuilder(with_pip=False).create(cls.root / ".venv")
        (cls.root / "stub-bin" / "uv.cmd").write_text("@exit /b 0\r\n", encoding="utf-8")

    def _command(self, launcher, mode):
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return f'"{comspec}" /d /s /c ""{self.root / "src" / "bin" / launcher}" {mode}"'


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class PosixReadinessCheckCountTests(_ReadinessCheckCountMixin, unittest.TestCase):
    LAUNCHERS = (("EMAPSSN.sh", "EMAPSSN_Config.py"), ("EMAPSSN_Tools.sh", "EMAPSSN_Tools.py"))

    @classmethod
    def setUpClass(cls):
        cls._build_root()
        # The launchers source the project's install.sh; skip its Linux GUI
        # library preflight, which depends on the host's packages.
        (cls.root / "install.sh").write_text(
            f". {shlex.quote(str(INSTALLER))}\n"
            "ssn_require_linux_gui_dependencies() { return 0; }\n",
            encoding="utf-8",
        )
        # Accept the interpreter-version probe; run the stand-ins with this Python.
        python = cls.root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            '#!/bin/sh\n[ "$1" = "-c" ] && exit 0\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n',
            encoding="utf-8",
        )
        uv = cls.root / "stub-bin" / "uv"
        uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        for path in (python, uv):
            path.chmod(0o755)

    def _command(self, launcher, mode):
        return ["/bin/bash", str(self.root / "src" / "bin" / launcher), *([mode] if mode else [])]


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class PosixGuiPackageCheckTests(unittest.TestCase):
    """The Tools preflight accepts either name of the t64-renamed libraries.

    It used to pick the libasound2t64/libcups2t64 names only for ID=ubuntu
    24.04+, so Debian 13 (which has only the new names) and Ubuntu-based
    derivatives such as Linux Mint 22 failed the check on every launch, even
    after running the suggested apt command.

    The old code read the host's own /etc/os-release, so which of the two
    "satisfy" tests catches a return to it depends on the host: on a t64-era
    host (Ubuntu 24.04+) the old-names test fails, elsewhere the new-names test
    does. Keep both.
    """

    RENAMED = ("libasound2", "libcups2")
    NEW_NAMES = ("libasound2t64", "libcups2t64")

    def _missing(self, kind, *, absent=(), apt_knows=(), os_release_fallback=None):
        """Run ssn_collect_missing_gui_packages against stub dpkg/apt.

        Every package is installed except those in `absent`; apt-cache knows only
        the names in `apt_knows`. Returns (missing packages, fallback consulted).
        """
        fallback = (
            ""
            if os_release_fallback is None
            else "ssn_os_release_uses_t64_packages() { echo FALLBACK >&2; return "
            f"{0 if os_release_fallback else 1}; }}"
        )
        program = "\n".join(
            (
                f". {shlex.quote(str(INSTALLER))}",
                "uname() { echo Linux; }",
                "apt() { :; }",
                f"ABSENT=' {' '.join(absent)} '",
                f"KNOWN=' {' '.join(apt_knows)} '",
                # dpkg-query -W -f=... NAME; apt-cache show NAME
                'dpkg-query() { case "$ABSENT" in *" ${!#} "*) return 1 ;; esac; printf "ii "; }',
                'apt-cache() { case "$KNOWN" in *" $2 "*) return 0 ;; esac; return 100; }',
                fallback,
                f"ssn_collect_missing_gui_packages {kind}",
                "printf '%s\\n' \"${SSN_MISSING_PACKAGES[@]}\"",
            )
        )
        result = subprocess.run(
            ["/bin/bash", "-c", program], check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return [line for line in result.stdout.splitlines() if line], "FALLBACK" in result.stderr

    def test_new_names_satisfy_the_tools_check_on_debian_13_and_derivatives(self):
        # Debian 13 and Mint 22 have only the t64 packages installed.
        missing, consulted = self._missing(
            "tools", absent=self.RENAMED, apt_knows=self.NEW_NAMES, os_release_fallback=False
        )
        self.assertEqual(missing, [])
        self.assertFalse(consulted)

    def test_old_names_still_satisfy_it_on_debian_12_and_ubuntu_22_04(self):
        missing, _ = self._missing("tools", absent=self.NEW_NAMES, apt_knows=self.RENAMED)
        self.assertEqual(missing, [])

    def test_a_missing_library_is_suggested_under_the_name_apt_offers(self):
        both_absent = self.RENAMED + self.NEW_NAMES
        cases = {
            "t64 system": (self.NEW_NAMES, list(self.NEW_NAMES)),
            "older system": (self.RENAMED, list(self.RENAMED)),
        }
        for label, (known, expected) in cases.items():
            with self.subTest(label):
                missing, consulted = self._missing(
                    "tools", absent=both_absent, apt_knows=known, os_release_fallback=False
                )
                self.assertEqual(missing, expected)
                self.assertFalse(consulted)

    def test_os_release_decides_only_when_apt_knows_neither_name(self):
        both_absent = self.RENAMED + self.NEW_NAMES
        for uses_t64, expected in ((True, list(self.NEW_NAMES)), (False, list(self.RENAMED))):
            with self.subTest(uses_t64=uses_t64):
                missing, consulted = self._missing(
                    "tools", absent=both_absent, os_release_fallback=uses_t64
                )
                self.assertEqual(missing, expected)
                self.assertTrue(consulted)

    def test_the_viewer_check_does_not_need_the_webengine_libraries(self):
        missing, _ = self._missing("viewer", absent=self.RENAMED + self.NEW_NAMES)
        self.assertEqual(missing, [])

    def test_every_launcher_needs_qts_core_libraries(self):
        # A minimal Debian 13 (a server or WSL image) lacks these; PySide6 then
        # failed to import with "libglib-2.0.so.0: cannot open shared object file"
        # after the check had passed.
        for kind in ("viewer", "tools"):
            with self.subTest(kind=kind):
                missing, _ = self._missing(
                    kind,
                    absent=("libgl1", "libfontconfig1", "libfreetype6", "fontconfig",
                            "libglib2.0-0", "libglib2.0-0t64"),
                    apt_knows=("libglib2.0-0t64",),
                )
                self.assertEqual(
                    missing,
                    ["libgl1", "libfontconfig1", "libfreetype6", "fontconfig", "libglib2.0-0t64"],
                )
                # Either GLib name satisfies it, like the other renamed libraries.
                for absent in ("libglib2.0-0", "libglib2.0-0t64"):
                    missing, _ = self._missing(kind, absent=(absent,))
                    self.assertEqual(missing, [])

    def test_the_tools_check_needs_qtwebengines_x_libraries(self):
        missing, _ = self._missing("tools", absent=("libxfixes3", "libxkbfile1"))
        self.assertEqual(missing, ["libxfixes3", "libxkbfile1"])
        missing, _ = self._missing("viewer", absent=("libxfixes3", "libxkbfile1"))
        self.assertEqual(missing, [])

    @unittest.skipUnless(shutil.which("dpkg"), "dpkg --compare-versions is Debian-only")
    def test_os_release_fallback_knows_the_renaming_releases(self):
        cases = {
            ("ubuntu", "24.04"): 0,
            ("ubuntu", "22.04"): 1,
            ("debian", "13"): 0,
            ("debian", "12"): 1,
            ("linuxmint", "22"): 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            for (distribution, version), expected in cases.items():
                os_release = Path(temporary) / f"{distribution}-{version}"
                os_release.write_text(
                    f'ID={distribution}\nVERSION_ID="{version}"\n', encoding="utf-8"
                )
                with self.subTest(distribution=distribution, version=version):
                    result = subprocess.run(
                        [
                            "/bin/bash",
                            "-c",
                            f". {shlex.quote(str(INSTALLER))}\n"
                            f"ssn_os_release_uses_t64_packages {shlex.quote(str(os_release))}",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)


@unittest.skipUnless(Path("/bin/bash").is_file(), "requires a POSIX bash")
class LinuxInstallerEntryWarningTests(unittest.TestCase):
    """install.sh's Linux part names the menu and Desktop entries of another checkout, in a scratch HOME.

    GNOME kept starting a previous checkout after install.sh had replaced the menu entries, until the user logged out
    and back in, and old copies on the Desktop kept pointing at that checkout. The installer now says so.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.home = self.base / "home"
        (self.home / ".local" / "share" / "applications").mkdir(parents=True)
        self.root = self.base / "this checkout"  # a space in the path: the entries quote it
        (self.root / "src" / "bin").mkdir(parents=True)
        (self.root / "src" / "bin" / "stand-in.sh").write_text("#!/bin/bash\n", encoding="utf-8", newline="\n")
        (self.root / "install.sh").write_bytes(_lf_bytes(INSTALLER))
        self.other = self.base / "other checkout"
        self.menu = self.home / ".local" / "share" / "applications"
        self.desktop = self.home / "Desktop"

    @staticmethod
    def write_entry(path, folder, name="viewer"):
        Path(path).write_text(
            "[Desktop Entry]\nType=Application\nName=EMAP-SSN\n"
            f'Exec="{folder}/src/bin/EMAPSSN_Desktop_Launcher.sh" {name}\nPath={folder}\nTerminal=false\n',
            encoding="utf-8", newline="\n")

    def install(self, answers):
        """Run the Linux part with the Qt system-library check stubbed out, answering its questions in order."""
        script = (f". {shlex.quote(str(self.root / 'install.sh'))}\n"
                  "ssn_require_linux_gui_dependencies() { return 0; }\nssn_install_linux\n")
        environment = {**os.environ, "HOME": str(self.home)}
        return subprocess.run(["/bin/bash", "-c", script], input=answers, capture_output=True, text=True,
                              env=environment, cwd=str(self.base), timeout=60, check=False)

    def old_menu_entries(self):
        self.other.mkdir(exist_ok=True)
        self.write_entry(self.menu / "emapssn.desktop", self.other)
        self.write_entry(self.menu / "emapssn_tools.desktop", self.other, "tools")

    def test_the_script_is_valid_bash(self):
        self.assertEqual(subprocess.run(["/bin/bash", "-n", str(INSTALLER)], capture_output=True).returncode, 0)

    def test_a_menu_entry_of_another_checkout_is_named_and_a_logout_is_advised_when_it_is_replaced(self):
        self.old_menu_entries()
        done = self.install("y\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[NOTE]", done.stdout)
        self.assertIn(f"emapssn.desktop starts {self.other}", done.stdout)
        self.assertIn(f"emapssn_tools.desktop starts {self.other}", done.stdout)
        self.assertIn(f"now starts {self.root}", done.stdout)
        self.assertIn("log out and back in", done.stdout)
        self.assertIn(f'Exec="{self.root}/src/bin/EMAPSSN_Desktop_Launcher.sh" viewer',
                      (self.menu / "emapssn.desktop").read_text(encoding="utf-8"))  # and it did replace them

    def test_entries_of_this_folder_or_a_link_to_it_say_nothing(self):
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        for folder in (self.root, alias):
            with self.subTest(folder=folder):
                self.write_entry(self.menu / "emapssn.desktop", folder)
                self.write_entry(self.menu / "emapssn_tools.desktop", folder, "tools")
                done = self.install("y\n")
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertNotIn("[NOTE]", done.stdout)
                self.assertNotIn("log out", done.stdout)

    def test_no_old_entries_say_nothing(self):
        done = self.install("y\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("[NOTE]", done.stdout)
        self.assertNotIn("[WARNING]", done.stderr)
        self.assertTrue((self.menu / "emapssn.desktop").is_file())

    def test_a_folder_that_is_gone_is_another_folder_too(self):
        self.write_entry(self.menu / "emapssn.desktop", self.base / "deleted checkout")
        done = self.install("y\n")
        self.assertIn(f"emapssn.desktop starts {self.base / 'deleted checkout'}", done.stdout)

    def test_declining_the_menu_says_the_entries_still_start_the_other_folder_and_leaves_them(self):
        self.old_menu_entries()
        before = (self.menu / "emapssn.desktop").read_bytes()
        done = self.install("n\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("still starts another folder", done.stdout)
        self.assertIn(f"emapssn.desktop starts {self.other}", done.stdout)
        self.assertIn("answer y", done.stdout)
        self.assertEqual((self.menu / "emapssn.desktop").read_bytes(), before)
        self.assertNotIn("log out", done.stdout)

    def test_a_desktop_copy_of_another_checkout_is_warned_about_when_it_was_not_replaced(self):
        self.desktop.mkdir()
        self.other.mkdir()
        self.write_entry(self.desktop / "emapssn.desktop", self.other)
        done = self.install("n\nn\n")  # not the menu, not the Desktop
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("[WARNING]", done.stderr)
        self.assertIn(f"{self.desktop / 'emapssn.desktop'} starts {self.other}", done.stderr)
        self.assertIn("answer y to the Desktop question", done.stderr)
        self.assertIn("copy those two files", done.stderr)
        self.assertIn(f"Exec=\"{self.other}", (self.desktop / "emapssn.desktop").read_text(encoding="utf-8"))  # untouched

    def test_a_yes_replaces_the_desktop_copies_but_a_differently_named_one_is_still_warned_about(self):
        self.desktop.mkdir()
        self.other.mkdir()
        self.write_entry(self.desktop / "emapssn.desktop", self.other)
        self.write_entry(self.desktop / "emapssn-old.desktop", self.other)
        done = self.install("n\ny\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn(f'Exec="{self.root}/src/bin/', (self.desktop / "emapssn.desktop").read_text(encoding="utf-8"))
        self.assertIn(f"{self.desktop / 'emapssn-old.desktop'} starts {self.other}", done.stderr)
        self.assertNotIn(f"{self.desktop / 'emapssn.desktop'} starts", done.stderr)

    def test_desktop_files_that_start_this_folder_or_are_not_ours_are_not_warned_about(self):
        self.desktop.mkdir()
        self.write_entry(self.desktop / "emapssn.desktop", self.root)
        (self.desktop / "emapssn-custom.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=Mine\nExec=/usr/bin/true\n", encoding="utf-8")
        (self.desktop / "other.desktop").write_text("[Desktop Entry]\nExec=/x/src/bin/y.sh\n", encoding="utf-8")
        done = self.install("n\nn\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("[WARNING]", done.stderr)

    def test_without_a_desktop_folder_there_is_no_desktop_question_and_no_warning(self):
        done = self.install("n\n")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("Desktop?", done.stdout)
        self.assertNotIn("[WARNING]", done.stderr)

    def test_the_entry_parser_reads_quoted_and_plain_exec_lines(self):
        for text, expected in (
            ('Exec="/a b/src/bin/EMAPSSN_Desktop_Launcher.sh" viewer', "/a b"),
            ("Exec=/plain/src/bin/EMAPSSN.sh", "/plain"),
            ("Exec=/usr/bin/firefox", None),
            ("Name=No exec line", None),
        ):
            with self.subTest(text=text):
                entry = self.base / "probe.desktop"
                entry.write_text("[Desktop Entry]\n" + text + "\n", encoding="utf-8")
                done = subprocess.run(
                    ["/bin/bash", "-c", f". {shlex.quote(str(INSTALLER))}\nssn_desktop_entry_folder {shlex.quote(str(entry))}"],
                    capture_output=True, text=True, check=False)
                if expected is None:
                    self.assertNotEqual(done.returncode, 0)
                    self.assertEqual(done.stdout, "")
                else:
                    self.assertEqual((done.returncode, done.stdout.strip()), (0, expected))


if __name__ == "__main__":
    unittest.main()
