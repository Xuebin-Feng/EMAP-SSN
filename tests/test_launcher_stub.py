# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Windows launcher stub (src/bin/Launcher.cs, built into EMAP-SSN.exe by install.bat) and the installer that builds it and points the shortcuts at it. The stub is a GUI-subsystem program that starts the Python launcher hidden, with the managed environment's path overrides removed, and falls back to the old visible console when asked (--console), when there is no managed Python, and when the Python launcher fails; install.bat builds it with the .NET Framework compiler, falls back to the batch-file shortcuts when it cannot, and gives each shortcut the AppUserModelID of the process it starts. Skipped where the compiler is absent."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LAUNCHER_SOURCE = SRC / "bin" / "Launcher.cs"
APP_ID_SCRIPT = SRC / "bin" / "Set_Shortcut_AppId.ps1"
INSTALLER = PROJECT_ROOT / "install.bat"
CREATE_NO_WINDOW = 0x08000000


def find_csc():
    windir = os.environ.get("WINDIR")
    if sys.platform != "win32" or not windir:
        return None
    for folder in ("Framework64", "Framework"):
        path = Path(windir) / "Microsoft.NET" / folder / "v4.0.30319" / "csc.exe"
        if path.is_file():
            return path
    return None


CSC = find_csc()

# A stand-in for python.exe: records how it was started and exits with the code it is told to.
FAKE_PYTHON = textwrap.dedent(
    """
    using System;
    using System.IO;
    using System.Runtime.InteropServices;

    static class FakePython
    {
        [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
        [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr window);

        static int Main(string[] args)
        {
            IntPtr console = GetConsoleWindow();
            string seen = "args=" + string.Join("|", args)
                + "\\ncwd=" + Directory.GetCurrentDirectory()
                + "\\nconsole_visible=" + (console != IntPtr.Zero && IsWindowVisible(console))
                + "\\nPYTHONHOME=" + (Environment.GetEnvironmentVariable("PYTHONHOME") ?? "(unset)")
                + "\\nQT_PLUGIN_PATH=" + (Environment.GetEnvironmentVariable("QT_PLUGIN_PATH") ?? "(unset)")
                + "\\nKEEP=" + (Environment.GetEnvironmentVariable("EMAPSSN_KEEP") ?? "(unset)") + "\\n";
            File.AppendAllText(Environment.GetEnvironmentVariable("FAKE_LOG"), seen);
            string code = Environment.GetEnvironmentVariable("FAKE_PYTHON_EXIT");
            return code == null ? 0 : int.Parse(code);
        }
    }
    """
)

# The stub's visible-console launch starts the command interpreter named by COMSPEC in a console of its own,
# which would flash a window on the screen of whoever runs the tests. Tests name this program instead: a
# program with no window that runs the real interpreter, hidden, with the arguments exactly as it was given them.
FAKE_SHELL = textwrap.dedent(
    """
    using System;
    using System.Diagnostics;

    static class FakeShell
    {
        static int Main()
        {
            string line = Environment.CommandLine;
            int at;
            if (line.StartsWith(((char)34).ToString()))
            {
                at = line.IndexOf((char)34, 1);
                at = at < 0 ? line.Length : at + 1;
            }
            else
            {
                at = line.IndexOf(' ');
                at = at < 0 ? line.Length : at;
            }
            var info = new ProcessStartInfo
            {
                FileName = Environment.GetEnvironmentVariable("REAL_COMSPEC"),
                Arguments = line.Substring(at).TrimStart(),
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using (Process process = Process.Start(info))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
    }
    """
)


def build(source: Path, target: Path, *options: str):
    done = subprocess.run(
        [str(CSC), "/nologo", "/optimize+", f"/out:{target}", *options, str(source)],
        capture_output=True, text=True, timeout=120,
    )
    if done.returncode != 0:
        raise AssertionError(f"csc failed: {done.stdout}{done.stderr}")


def subsystem(path: Path) -> int:
    """The Subsystem field of a Windows executable's header (2 is a GUI program, 3 a console program)."""
    data = path.read_bytes()
    header = struct.unpack_from("<I", data, 0x3C)[0]
    return struct.unpack_from("<H", data, header + 24 + 68)[0]


@unittest.skipUnless(CSC is not None, "the .NET Framework C# compiler is not installed")
class StubTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls.directory.cleanup)
        cls.work = Path(cls.directory.name)
        source = cls.work / "FakePython.cs"
        source.write_text(FAKE_PYTHON, encoding="utf-8")
        cls.fake_python = cls.work / "FakePython.exe"
        build(source, cls.fake_python)
        shell_source = cls.work / "FakeShell.cs"
        shell_source.write_text(FAKE_SHELL, encoding="utf-8")
        cls.fake_shell = cls.work / "FakeShell.exe"
        build(shell_source, cls.fake_shell, "/target:winexe")
        cls.stub = cls.work / "EMAP-SSN.exe"
        build(LAUNCHER_SOURCE, cls.stub, "/target:winexe", f"/win32icon:{SRC / 'bin' / 'logos' / 'viewer_logo_large.ico'}")

    def project(self, *, with_python=True, with_batch=True):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        shutil.copyfile(self.stub, root / "EMAP-SSN.exe")
        (root / "src" / "desktop").mkdir(parents=True)
        (root / "src" / "desktop" / "Desktop_Launcher.py").write_text("# stands in for the launcher\n", encoding="utf-8")
        (root / "src" / "bin").mkdir()
        if with_python:
            (root / ".venv" / "Scripts").mkdir(parents=True)
            shutil.copyfile(self.fake_python, root / ".venv" / "Scripts" / "python.exe")
            self.warm_up(root / ".venv" / "Scripts" / "python.exe", root)
        if with_batch:
            (root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.bat").write_text(
                '@echo off\r\n>>"%FAKE_BAT_LOG%" echo args=%* cwd=%CD%\r\n', encoding="utf-8")
        return root

    @staticmethod
    def warm_up(program, root):
        """Start a fresh copy once before the stub does. A scanner may hold a program that was just written for a
        moment, and the stub would take that for a Python that cannot start and open the console (as it should)."""
        for _ in range(40):
            try:
                subprocess.run([str(program)], env=dict(os.environ, FAKE_LOG=str(root / "warmup.log")),
                               capture_output=True, timeout=30, creationflags=CREATE_NO_WINDOW)
                return
            except OSError:
                time.sleep(0.25)

    def run_stub(self, root, *arguments, exit_code=None, wait=True):
        environment = dict(os.environ, FAKE_LOG=str(root / "python.log"), FAKE_BAT_LOG=str(root / "bat.log"),
                           PYTHONHOME="C:\\somewhere\\else", QT_PLUGIN_PATH="C:\\elsewhere", EMAPSSN_KEEP="kept",
                           COMSPEC=str(self.fake_shell), REAL_COMSPEC=os.environ.get("COMSPEC", "cmd.exe"))
        if exit_code is not None:
            environment["FAKE_PYTHON_EXIT"] = str(exit_code)
        else:
            environment.pop("FAKE_PYTHON_EXIT", None)
        done = subprocess.run([str(root / "EMAP-SSN.exe"), *arguments], env=environment, capture_output=True,
                              timeout=60, creationflags=CREATE_NO_WINDOW, cwd=str(self.work))
        return done.returncode

    @staticmethod
    def read(path, wait=10.0):
        deadline = time.monotonic() + wait
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(0.2)
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def test_the_stub_is_a_gui_program_with_the_viewer_icon(self):
        self.assertEqual(subsystem(self.stub), 2)  # no console window of its own
        self.assertGreater(self.stub.stat().st_size, (SRC / "bin" / "logos" / "viewer_logo_large.ico").stat().st_size)
        self.assertLess(subsystem(self.fake_python) - 2, 2)  # (the stand-in is a console program: 3)
        self.assertEqual(subsystem(self.fake_python), 3)

    def test_it_starts_the_python_launcher_hidden_for_the_configuration_window_by_default(self):
        root = self.project()
        self.assertEqual(self.run_stub(root), 0)
        log = self.read(root / "python.log")
        launcher = root / "src" / "desktop" / "Desktop_Launcher.py"
        self.assertIn(f"args=-u|{launcher}|viewer", log)
        self.assertIn(f"cwd={root}", log)
        self.assertIn("console_visible=False", log)
        self.assertFalse((root / "bat.log").exists(), "the console launch ran too")

    def test_tools_and_viewer_name_their_window_and_other_words_are_ignored(self):
        root = self.project()
        self.assertEqual(self.run_stub(root, "tools"), 0)
        self.assertIn("|tools", self.read(root / "python.log"))
        root = self.project()
        self.assertEqual(self.run_stub(root, "nonsense", "viewer"), 0)
        self.assertIn("|viewer", self.read(root / "python.log"))

    def test_the_managed_environment_is_not_redirected_but_the_rest_is_kept(self):
        root = self.project()
        self.run_stub(root)
        log = self.read(root / "python.log")
        self.assertIn("PYTHONHOME=(unset)", log)
        self.assertIn("QT_PLUGIN_PATH=(unset)", log)
        self.assertIn("KEEP=kept", log)

    def test_a_launcher_that_fails_hands_over_to_the_visible_console(self):
        root = self.project()
        self.assertEqual(self.run_stub(root, "tools", exit_code=3), 0)
        self.assertIn("|tools", self.read(root / "python.log"))
        bat = self.read(root / "bat.log")
        self.assertIn("args=tools", bat)
        self.assertIn(f"cwd={root}", bat)

    def test_any_failure_does_it_not_only_the_launchers_own_code(self):
        root = self.project()
        self.run_stub(root, exit_code=1)  # as Python's own failure to start the launcher is
        self.assertIn("args=viewer", self.read(root / "bat.log"))

    def test_console_goes_straight_to_the_visible_console(self):
        root = self.project()
        self.assertEqual(self.run_stub(root, "viewer", "--console"), 0)
        self.assertIn("args=viewer", self.read(root / "bat.log"))
        self.assertFalse((root / "python.log").exists())

    def test_with_no_managed_python_yet_the_first_start_is_the_console_which_sets_it_up(self):
        root = self.project(with_python=False)
        self.assertEqual(self.run_stub(root), 0)
        self.assertIn("args=viewer", self.read(root / "bat.log"))

    def test_the_console_launch_is_run_by_the_command_interpreter_with_the_batch_files_path_quoted(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name) / "a folder with spaces"
        root.mkdir()
        shutil.copyfile(self.stub, root / "EMAP-SSN.exe")
        (root / "src" / "bin").mkdir(parents=True)
        (root / "src" / "bin" / "EMAPSSN_Desktop_Launcher.bat").write_text(
            '@echo off\r\n>>"%FAKE_BAT_LOG%" echo args=%* cwd=%CD%\r\n', encoding="utf-8")
        self.run_stub(root, "tools")
        self.assertIn("args=tools", self.read(root / "bat.log"))

    def test_the_source_names_the_one_batch_file_and_launcher_it_starts(self):
        text = LAUNCHER_SOURCE.read_text(encoding="utf-8")
        self.assertIn('Path.Combine(root, "src", "desktop", "Desktop_Launcher.py")', text)
        self.assertIn('Path.Combine(root, "src", "bin", "EMAPSSN_Desktop_Launcher.bat")', text)
        self.assertTrue((SRC / "desktop" / "Desktop_Launcher.py").is_file())
        self.assertTrue((SRC / "bin" / "EMAPSSN_Desktop_Launcher.bat").is_file())


class InstallerTextTests(unittest.TestCase):
    """What install.bat says, which needs no compiler."""

    def setUp(self):
        self.text = INSTALLER.read_bytes().decode("utf-8")

    def test_the_binary_is_not_committed(self):
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("EMAP-SSN.exe", ignored)
        listed = subprocess.run(["git", "-C", str(PROJECT_ROOT), "ls-files", "--", "*.exe"], capture_output=True, text=True)
        if listed.returncode == 0:
            self.assertEqual(listed.stdout.split(), [])

    def test_the_installer_uses_the_compiler_that_ships_with_windows_and_embeds_the_icon(self):
        for part in ("Framework64", "Framework", "v4.0.30319", "csc.exe", "/target:winexe", "/win32icon:", "src\\bin\\Launcher.cs"):
            self.assertIn(part, self.text)
        self.assertNotIn("curl", self.text.lower())  # nothing is downloaded for it
        self.assertNotIn("Invoke-WebRequest", self.text)

    def test_it_keeps_the_old_shortcuts_when_the_build_fails_and_the_batch_file_is_still_there(self):
        self.assertIn("EMAPSSN_Desktop_Launcher.bat", self.text)
        self.assertIn("if ($env:USE_LAUNCHER_EXE)", self.text)
        self.assertIn("System32\\cmd.exe", self.text)

    def test_the_shortcuts_get_the_ids_the_processes_set(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        self.assertIn(f"-AppId {APP_USER_MODEL_IDS['config']}", self.text)
        self.assertIn(f"-AppId {APP_USER_MODEL_IDS['tools']}", self.text)
        self.assertEqual(len(set(APP_USER_MODEL_IDS.values())), len(APP_USER_MODEL_IDS))

    def test_the_helper_files_the_installer_runs_exist(self):
        self.assertTrue(APP_ID_SCRIPT.is_file())
        self.assertTrue(LAUNCHER_SOURCE.is_file())
        self.assertIn("Set_Shortcut_AppId.ps1", self.text)

    def test_line_endings_are_what_cmd_needs(self):
        self.assertIn(b"\r\n", INSTALLER.read_bytes())
        self.assertNotIn(b"\r\r", INSTALLER.read_bytes())


@unittest.skipUnless(CSC is not None, "the .NET Framework C# compiler is not installed")
@unittest.skipUnless(sys.platform == "win32", "install.bat is Windows' installer")
class InstallerRunTests(unittest.TestCase):
    """install.bat run in a scratch copy of the files it needs: nothing is made outside it."""

    def tree(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name) / "project root"
        (root / "src" / "bin" / "logos").mkdir(parents=True)
        shutil.copyfile(INSTALLER, root / "install.bat")
        for name in ("Launcher.cs", "Set_Shortcut_AppId.ps1"):
            shutil.copyfile(SRC / "bin" / name, root / "src" / "bin" / name)
        for icon in (SRC / "bin" / "logos").glob("*.ico"):
            shutil.copyfile(icon, root / "src" / "bin" / "logos" / icon.name)
        return root

    def install(self, root, windir=None):
        environment = dict(os.environ)
        if windir is not None:
            environment["WINDIR"] = windir
        return subprocess.run(
            ["cmd.exe", "/d", "/c", "echo N| call .\\install.bat"], cwd=str(root), env=environment,
            capture_output=True, text=True, timeout=180, creationflags=CREATE_NO_WINDOW,
        )

    @staticmethod
    def shortcut(root, name):
        script = (
            "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK); "
            "$i = (New-Object -ComObject Shell.Application).NameSpace((Split-Path $env:LNK)).ParseName((Split-Path $env:LNK -Leaf)); "
            "'TARGET=' + $s.TargetPath; 'ARGS=' + $s.Arguments; 'CWD=' + $s.WorkingDirectory; 'ICON=' + $s.IconLocation; "
            "'AUMID=' + $i.ExtendedProperty('System.AppUserModel.ID')"
        )
        done = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True,
                              env=dict(os.environ, LNK=str(root / name)), timeout=60, creationflags=CREATE_NO_WINDOW)
        return dict(line.split("=", 1) for line in done.stdout.splitlines() if "=" in line)

    def test_it_builds_the_launcher_and_points_both_shortcuts_at_it_with_their_ids(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        root = self.tree()
        done = self.install(root)
        self.assertIn("Built EMAP-SSN.exe", done.stdout, done.stdout + done.stderr)
        exe = root / "EMAP-SSN.exe"
        self.assertTrue(exe.is_file())
        self.assertEqual(subsystem(exe), 2)
        for name, kind, icon in (("EMAP-SSN.lnk", "viewer", "viewer_logo_large.ico"),
                                 ("EMAP-SSN Tools.lnk", "tools", "tool_logo_large.ico")):
            with self.subTest(shortcut=name):
                found = self.shortcut(root, name)
                self.assertEqual(Path(found["TARGET"]), exe)
                self.assertEqual(found["ARGS"], kind)
                self.assertEqual(Path(found["CWD"]), root)
                self.assertTrue(found["ICON"].lower().endswith(icon + ",0"), found["ICON"])
                self.assertEqual(found["AUMID"], APP_USER_MODEL_IDS["config" if kind == "viewer" else "tools"])
        # Nothing but this folder was touched: the Desktop copy was declined.
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["EMAP-SSN Tools.lnk", "EMAP-SSN.exe", "EMAP-SSN.lnk", "install.bat", "src"])

    def test_without_the_compiler_the_shortcuts_run_the_batch_file_as_before_and_still_get_their_ids(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        root = self.tree()
        empty = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(empty.cleanup)
        (Path(empty.name) / "System32").mkdir()
        done = self.install(root, windir=empty.name)
        self.assertIn("Could not build EMAP-SSN.exe", done.stdout, done.stdout + done.stderr)
        self.assertFalse((root / "EMAP-SSN.exe").exists())
        found = self.shortcut(root, "EMAP-SSN.lnk")
        self.assertTrue(found["TARGET"].lower().endswith("cmd.exe"), found["TARGET"])
        self.assertIn("EMAPSSN_Desktop_Launcher.bat", found["ARGS"])
        self.assertTrue(found["ARGS"].endswith(" viewer\""), found["ARGS"])
        self.assertEqual(found["AUMID"], APP_USER_MODEL_IDS["config"])

    def test_a_failed_build_leaves_no_stale_launcher_behind(self):
        root = self.tree()
        (root / "src" / "bin" / "Launcher.cs").write_text("this is not C#", encoding="utf-8")
        (root / "EMAP-SSN.exe").write_bytes(b"an old build")
        done = self.install(root)
        self.assertIn("Could not build EMAP-SSN.exe", done.stdout)
        self.assertFalse((root / "EMAP-SSN.exe").exists(), "the shortcuts would have started an old build")
        self.assertTrue(self.shortcut(root, "EMAP-SSN Tools.lnk")["TARGET"].lower().endswith("cmd.exe"))


class AppIdTests(unittest.TestCase):
    def test_every_window_sets_the_id_its_shortcut_carries_before_its_first_window(self):
        import ast

        from desktop.Desktop_App import APP_USER_MODEL_IDS

        expected = {"EMAPSSN_Config.py": "config", "EMAPSSN_Tools.py": "tools", "EMAPSSN_Viewer.py": "viewer"}
        for name, key in expected.items():
            with self.subTest(window=name):
                source = (SRC / name).read_text(encoding="utf-8")
                tree = ast.parse(source)
                calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and getattr(node.func, "id", None) == "set_app_user_model_id"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(ast.unparse(calls[0].args[0]), f"APP_USER_MODEL_IDS['{key}']")
        self.assertEqual(sorted(APP_USER_MODEL_IDS), ["config", "tools", "viewer", "vr_config"])

    def test_setting_the_id_is_a_windows_only_call_that_reports_whether_it_took(self):
        from unittest import mock

        from desktop import Desktop_App

        with mock.patch.object(Desktop_App.sys, "platform", "linux"):
            self.assertFalse(Desktop_App.set_app_user_model_id("x"))
        if sys.platform == "win32":
            program = (
                "import ctypes, sys; sys.path.insert(0, sys.argv[1]); from desktop import Desktop_App; "
                "print(Desktop_App.set_app_user_model_id('ca.test.one')); "
                "p = ctypes.c_wchar_p(); ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(p)); print(p.value)"
            )
            done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=60,
                                  creationflags=CREATE_NO_WINDOW)
            self.assertEqual(done.stdout.split(), ["True", "ca.test.one"], done.stderr)


if __name__ == "__main__":
    unittest.main()
