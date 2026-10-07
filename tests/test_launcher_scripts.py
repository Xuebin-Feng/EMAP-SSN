"""Launcher scripts (install.sh/.bat and src/bin launchers): shared environment
sanitizing and setup-lock policy, the managed-venv interpreter probe, the
existing-instance probe that runs before dependency validation, the desktop
launcher structure, and the POSIX dependency-setup lock."""

from __future__ import annotations

import builtins
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import types
import unittest


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
    def test_desktop_launchers_show_then_dismiss_and_keep_direct_diagnostics(self):
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


if __name__ == "__main__":
    unittest.main()
