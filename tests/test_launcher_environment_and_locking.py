from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
INSTALLER = PROJECT_ROOT / "install.sh"


class ManagedEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        sys.path.insert(0, str(SRC_DIR))
        from utilities import Desktop_Launcher_Monitor

        cls.monitor = Desktop_Launcher_Monitor

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

    def test_existing_launchers_share_environment_and_lock_policy(self):
        sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in (
                PROJECT_ROOT / "install.sh",
                SRC_DIR / "bin" / "SSN_Desktop_Launcher.sh",
                SRC_DIR / "bin" / "SSN_Viewer.sh",
                SRC_DIR / "bin" / "SSN_Tools.sh",
                SRC_DIR / "bin" / "SSN_Desktop_Launcher.bat",
                SRC_DIR / "bin" / "SSN_Viewer.bat",
                SRC_DIR / "bin" / "SSN_Tools.bat",
            )
        }

        self.assertIn("ssn_sanitize_managed_environment", sources["install.sh"])
        self.assertIn(
            "ssn_sanitize_managed_environment", sources["SSN_Desktop_Launcher.sh"]
        )
        self.assertIn(":SANITIZE_MANAGED_ENVIRONMENT", sources["SSN_Desktop_Launcher.bat"])
        for name in ("SSN_Viewer.bat", "SSN_Tools.bat"):
            self.assertIn("dependency_setup.lock", sources[name])
            self.assertIn("--locked-setup", sources[name])
            self.assertIn('"%COMSPEC%" /d /c', sources[name])


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
                root / "Cache_Files" / "Launcher_State" / "dependency_setup.lock"
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


if __name__ == "__main__":
    unittest.main()
