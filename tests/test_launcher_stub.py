# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Windows launcher stub (src/bin/Launcher.cs, built into EMAP-SSN.exe by install.bat) and the installer that builds it and points the shortcuts at it. The stub is a GUI-subsystem program that starts the Python launcher hidden, with the managed environment's path overrides removed, and falls back to the old visible console when asked (--console), when there is no managed Python, and when the Python launcher fails; install.bat builds it (src/bin/Build_Launcher.bat, which opt_vr's installer runs too) with the .NET Framework compiler, falls back to the batch-file shortcuts when it cannot, and gives the shortcuts no AppUserModelID (the windows set their own, which tests below cover with the taskbar identity of each main window). The stub also starts the VR Configuration window's launcher, which is opt_vr's own script, and falls back to opt_vr's own batch file. Skipped where the compiler is absent."""

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
BUILD_SCRIPT = SRC / "bin" / "Build_Launcher.bat"
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

    def project(self, *, with_python=True, with_batch=True, with_vr=False, with_vr_script=True):
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
        if with_vr:  # opt_vr: its own launcher script and its own console launch
            (root / "opt_vr" / "src" / "bin").mkdir(parents=True)
            if with_vr_script:
                (root / "opt_vr" / "src" / "bin" / "Desktop_Launcher_VR.py").write_text(
                    "# stands in for opt_vr's launcher\n", encoding="utf-8")
            (root / "opt_vr" / "src" / "bin" / "EMAPSSN_Desktop_Launcher_VR.bat").write_text(
                '@echo off\r\n>>"%FAKE_BAT_LOG%" echo vr_batch args=%* cwd=%CD%\r\n', encoding="utf-8")
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

    def test_the_source_names_the_launcher_and_batch_file_of_each_window_it_starts(self):
        text = LAUNCHER_SOURCE.read_text(encoding="utf-8")
        for kind, script, batch in (
            ("viewer", r"src\desktop\Desktop_Launcher.py", r"src\bin\EMAPSSN_Desktop_Launcher.bat"),
            ("tools", r"src\desktop\Desktop_Launcher.py", r"src\bin\EMAPSSN_Desktop_Launcher.bat"),
            ("vr", r"opt_vr\src\bin\Desktop_Launcher_VR.py", r"opt_vr\src\bin\EMAPSSN_Desktop_Launcher_VR.bat"),
        ):
            with self.subTest(kind=kind):
                self.assertIn(f'new[] {{ "{kind}", @"{script}", @"{batch}" }}', text)
        self.assertTrue((SRC / "desktop" / "Desktop_Launcher.py").is_file())
        self.assertTrue((SRC / "bin" / "EMAPSSN_Desktop_Launcher.bat").is_file())
        # opt_vr's own are where the stub looks, when the submodule is checked out beside it.
        opt_vr = PROJECT_ROOT / "opt_vr" / "src" / "bin"
        if (opt_vr / "Desktop_Launcher_VR.py").exists() or (opt_vr / "EMAPSSN_Desktop_Launcher_VR.bat").exists():
            self.assertTrue((opt_vr / "Desktop_Launcher_VR.py").is_file())
            self.assertTrue((opt_vr / "EMAPSSN_Desktop_Launcher_VR.bat").is_file())

    def test_vr_starts_the_vr_launcher_that_opt_vr_brings_hidden_and_not_the_main_one(self):
        root = self.project(with_vr=True)
        self.assertEqual(self.run_stub(root, "vr"), 0)
        log = self.read(root / "python.log")
        launcher = root / "opt_vr" / "src" / "bin" / "Desktop_Launcher_VR.py"
        self.assertIn(f"args=-u|{launcher}|vr", log)
        self.assertNotIn("Desktop_Launcher.py", log.replace("Desktop_Launcher_VR.py", ""))
        self.assertIn(f"cwd={root}", log)
        self.assertIn("console_visible=False", log)
        self.assertIn("PYTHONHOME=(unset)", log)
        self.assertFalse((root / "bat.log").exists(), "a console launch ran too")

    def test_vr_falls_back_to_the_vr_batch_file_when_asked_when_python_is_missing_and_when_its_launcher_fails(self):
        root = self.project(with_vr=True)
        self.assertEqual(self.run_stub(root, "vr", "--console"), 0)
        bat = self.read(root / "bat.log")
        self.assertIn("vr_batch args=vr", bat)
        self.assertIn(f"cwd={root}", bat)
        self.assertNotIn("\nargs=", "\n" + bat)  # Config's batch file did not run
        self.assertFalse((root / "python.log").exists())
        root = self.project(with_vr=True)
        self.assertEqual(self.run_stub(root, "vr", exit_code=3), 0)  # the launcher's code for "use the console"
        self.assertIn("|vr", self.read(root / "python.log"))
        self.assertIn("vr_batch args=vr", self.read(root / "bat.log"))
        root = self.project(with_vr=True, with_python=False)  # the first start: the console sets the environment up
        self.assertEqual(self.run_stub(root, "vr"), 0)
        self.assertIn("vr_batch args=vr", self.read(root / "bat.log"))
        self.assertFalse((root / "python.log").exists())

    def test_vr_without_its_launcher_script_opens_its_batch_file_and_the_other_kinds_do_not_need_opt_vr(self):
        root = self.project(with_vr=True, with_vr_script=False)
        self.assertEqual(self.run_stub(root, "vr"), 0)
        self.assertIn("vr_batch args=vr", self.read(root / "bat.log"))
        self.assertFalse((root / "python.log").exists())
        root = self.project()  # no opt_vr at all
        self.assertEqual(self.run_stub(root, "tools"), 0)
        self.assertIn("|tools", self.read(root / "python.log"))

    def test_the_last_window_named_wins_and_a_word_that_names_none_is_ignored(self):
        root = self.project(with_vr=True)
        self.assertEqual(self.run_stub(root, "tools", "vr"), 0)
        log = self.read(root / "python.log")
        self.assertIn("Desktop_Launcher_VR.py|vr", log)
        self.assertNotIn("|tools", log)
        root = self.project(with_vr=True)
        self.assertEqual(self.run_stub(root, "vr", "nonsense"), 0)
        self.assertIn("Desktop_Launcher_VR.py|vr", self.read(root / "python.log"))


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

    def test_the_build_script_uses_the_compiler_that_ships_with_windows_and_embeds_the_icon(self):
        build = BUILD_SCRIPT.read_bytes().decode("utf-8")
        for part in ("Framework64", "Framework", "v4.0.30319", "csc.exe", "/target:winexe", "/win32icon:", "Launcher.cs"):
            self.assertIn(part, build)
        self.assertNotIn("curl", build.lower())  # nothing is downloaded for it
        self.assertNotIn("Invoke-WebRequest", build)
        self.assertIn(b"\r\n", BUILD_SCRIPT.read_bytes())
        self.assertNotIn(b"\r\r", BUILD_SCRIPT.read_bytes())

    def test_the_installer_builds_through_that_script_instead_of_a_copy_of_the_build(self):
        self.assertIn('call "src\\bin\\Build_Launcher.bat"', self.text)
        self.assertNotIn("csc.exe", self.text)
        self.assertNotIn("/target:winexe", self.text)
        self.assertIn("if !ERRORLEVEL! equ 0", self.text)
        self.assertIn('set "USE_LAUNCHER_EXE=1"', self.text)

    def test_it_keeps_the_old_shortcuts_when_the_build_fails_and_the_batch_file_is_still_there(self):
        self.assertIn("EMAPSSN_Desktop_Launcher.bat", self.text)
        self.assertIn("if ($env:USE_LAUNCHER_EXE)", self.text)
        self.assertIn("System32\\cmd.exe", self.text)

    def test_the_shortcuts_carry_no_id_because_a_window_whose_id_a_shortcut_carries_gets_its_large_icon(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        self.assertNotIn("-AppId", self.text)
        self.assertNotIn("Set_Shortcut_AppId", self.text)
        for app_id in APP_USER_MODEL_IDS.values():
            self.assertNotIn(app_id, self.text)
        self.assertEqual(len(set(APP_USER_MODEL_IDS.values())), len(APP_USER_MODEL_IDS))
        self.assertFalse((SRC / "bin" / "Set_Shortcut_AppId.ps1").exists())
        # The shortcuts keep the large icons: those are for the desktop.
        self.assertIn("viewer_logo_large.ico", self.text)
        self.assertIn("tool_logo_large.ico", self.text)

    def test_the_helper_files_the_installer_runs_exist(self):
        self.assertTrue(LAUNCHER_SOURCE.is_file())
        self.assertTrue(BUILD_SCRIPT.is_file())

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
        for name in ("Launcher.cs", "Build_Launcher.bat"):
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

    def test_it_builds_the_launcher_and_points_both_shortcuts_at_it_with_no_id(self):
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
                self.assertEqual(found["AUMID"], "")  # a window whose ID it carried would get its large icon
        # Nothing but this folder was touched: the Desktop copy was declined.
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["EMAP-SSN Tools.lnk", "EMAP-SSN.exe", "EMAP-SSN.lnk", "install.bat", "src"])
        self.assertIn("Built EMAP-SSN.exe, the launcher with no console window", done.stdout)

    def test_without_the_compiler_the_shortcuts_run_the_batch_file_as_before(self):
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
        self.assertEqual(found["AUMID"], "")

    def test_a_failed_build_leaves_no_stale_launcher_behind(self):
        root = self.tree()
        (root / "src" / "bin" / "Launcher.cs").write_text("this is not C#", encoding="utf-8")
        (root / "EMAP-SSN.exe").write_bytes(b"an old build")
        done = self.install(root)
        self.assertIn("Could not build EMAP-SSN.exe", done.stdout)
        self.assertFalse((root / "EMAP-SSN.exe").exists(), "the shortcuts would have started an old build")
        self.assertTrue(self.shortcut(root, "EMAP-SSN Tools.lnk")["TARGET"].lower().endswith("cmd.exe"))


@unittest.skipUnless(CSC is not None, "the .NET Framework C# compiler is not installed")
@unittest.skipUnless(sys.platform == "win32", "Build_Launcher.bat is Windows'")
class BuildLauncherRunTests(unittest.TestCase):
    """src/bin/Build_Launcher.bat, which install.bat and opt_vr's install_vr.bat share, run in a scratch copy."""

    def tree(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        root = Path(folder.name) / "project root"
        (root / "src" / "bin" / "logos").mkdir(parents=True)
        for name in ("Launcher.cs", "Build_Launcher.bat"):
            shutil.copyfile(SRC / "bin" / name, root / "src" / "bin" / name)
        shutil.copyfile(SRC / "bin" / "logos" / "viewer_logo_large.ico", root / "src" / "bin" / "logos" / "viewer_logo_large.ico")
        return root

    @staticmethod
    def build(root, *arguments, windir=None):
        environment = dict(os.environ)
        if windir is not None:
            environment["WINDIR"] = windir
        # One string, so cmd gets the quotes as they are: it strips the outer pair and runs call "<script>" <arguments>.
        command = f'cmd.exe /d /c "call "{root / "src" / "bin" / "Build_Launcher.bat"}" {" ".join(arguments)}"'
        return subprocess.run(command, cwd=str(root.parent), env=environment, capture_output=True, text=True,
                              timeout=180, creationflags=CREATE_NO_WINDOW)

    def test_it_builds_a_gui_launcher_from_wherever_it_is_run_and_says_so(self):
        root = self.tree()
        done = self.build(root)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Built EMAP-SSN.exe", done.stdout)
        self.assertEqual(subsystem(root / "EMAP-SSN.exe"), 2)
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["EMAP-SSN.exe", "src"])  # nothing else is made

    def test_it_always_replaces_the_launcher_unless_asked_to_build_only_when_needed(self):
        root = self.tree()
        (root / "EMAP-SSN.exe").write_bytes(b"an old build")
        self.assertEqual(self.build(root).returncode, 0)
        self.assertEqual(subsystem(root / "EMAP-SSN.exe"), 2)  # a real launcher now

    def test_if_needed_leaves_a_launcher_that_is_current_alone(self):
        root = self.tree()
        self.assertEqual(self.build(root).returncode, 0)
        before = (root / "EMAP-SSN.exe").stat().st_mtime_ns
        done = self.build(root, "--if-needed")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("up to date", done.stdout)
        self.assertEqual((root / "EMAP-SSN.exe").stat().st_mtime_ns, before)

    def test_if_needed_builds_when_the_launcher_is_missing(self):
        root = self.tree()
        done = self.build(root, "--if-needed")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Built EMAP-SSN.exe", done.stdout)
        self.assertEqual(subsystem(root / "EMAP-SSN.exe"), 2)

    def test_if_needed_rebuilds_a_launcher_older_than_its_source_as_one_from_before_a_new_window_is(self):
        root = self.tree()
        self.assertEqual(self.build(root).returncode, 0)
        old = (root / "src" / "bin" / "Launcher.cs").stat().st_mtime - 3600
        os.utime(root / "EMAP-SSN.exe", (old, old))
        done = self.build(root, "--if-needed")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Built EMAP-SSN.exe", done.stdout)
        self.assertGreater((root / "EMAP-SSN.exe").stat().st_mtime, old + 60)

    def test_without_the_compiler_a_missing_launcher_is_reported_with_a_failing_status(self):
        root = self.tree()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as empty:
            done = self.build(root, windir=empty)
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("Could not build EMAP-SSN.exe", done.stdout)
        self.assertFalse((root / "EMAP-SSN.exe").exists())

    def test_without_the_compiler_a_stale_launcher_does_not_count_but_a_current_one_does(self):
        root = self.tree()
        self.assertEqual(self.build(root).returncode, 0)
        old = (root / "src" / "bin" / "Launcher.cs").stat().st_mtime - 3600
        os.utime(root / "EMAP-SSN.exe", (old, old))
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as empty:
            stale = self.build(root, "--if-needed", windir=empty)
            self.assertEqual(stale.returncode, 1, stale.stdout)  # it would open the wrong window for "vr"
            self.assertIn("Could not build EMAP-SSN.exe", stale.stdout)
            now = (root / "src" / "bin" / "Launcher.cs").stat().st_mtime + 60
            os.utime(root / "EMAP-SSN.exe", (now, now))
            current = self.build(root, "--if-needed", windir=empty)
            self.assertEqual(current.returncode, 0, current.stdout)
            self.assertIn("up to date", current.stdout)

    def test_a_failed_build_leaves_no_stale_launcher_behind(self):
        root = self.tree()
        (root / "src" / "bin" / "Launcher.cs").write_text("this is not C#", encoding="utf-8")
        (root / "EMAP-SSN.exe").write_bytes(b"an old build")
        done = self.build(root)
        self.assertEqual(done.returncode, 1)
        self.assertFalse((root / "EMAP-SSN.exe").exists())

    def test_it_changes_nothing_about_the_callers_folder(self):
        root = self.tree()
        command = f'cmd.exe /d /c "cd /d "{root.parent}" && call "{root / "src" / "bin" / "Build_Launcher.bat"}" >nul && cd"'
        done = subprocess.run(command, capture_output=True, text=True, timeout=180, creationflags=CREATE_NO_WINDOW)
        self.assertEqual(Path(done.stdout.strip()), root.parent)


class AppIdTests(unittest.TestCase):
    def test_every_window_sets_its_process_id_before_its_first_window(self):
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
        self.assertEqual(sorted(APP_USER_MODEL_IDS), ["config", "tools", "viewer"])  # VR Config declares its own

    def test_setting_the_id_is_a_windows_only_call_that_reports_whether_it_took(self):
        from unittest import mock

        from desktop import Taskbar_Identity

        with mock.patch.object(Taskbar_Identity.sys, "platform", "linux"):
            self.assertFalse(Taskbar_Identity.set_app_user_model_id("x"))
        if sys.platform == "win32":
            program = (
                "import ctypes, sys; sys.path.insert(0, sys.argv[1]); from desktop import Taskbar_Identity; "
                "print(Taskbar_Identity.set_app_user_model_id('ca.test.one')); "
                "p = ctypes.c_wchar_p(); ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(ctypes.byref(p)); print(p.value)"
            )
            done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=60,
                                  creationflags=CREATE_NO_WINDOW)
            self.assertEqual(done.stdout.split(), ["True", "ca.test.one"], done.stderr)


def read_window_texts(hwnd, property_ids):
    """What Windows holds for a native window under System.AppUserModel.* (IPropertyStore::GetValue)."""
    import ctypes

    from desktop import Taskbar_Identity

    shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
    shell32.SHGetPropertyStoreForWindow.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(Taskbar_Identity._Guid), ctypes.POINTER(ctypes.c_void_p)]
    shell32.SHGetPropertyStoreForWindow.restype = ctypes.c_long
    ole32.PropVariantClear.argtypes = [ctypes.POINTER(Taskbar_Identity._PropVariant)]
    store = ctypes.c_void_p()
    iid = Taskbar_Identity._guid(Taskbar_Identity._PROPERTY_STORE_IID)
    assert shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid), ctypes.byref(store)) == 0
    methods = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    get_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(Taskbar_Identity._PropertyKey),
                                   ctypes.POINTER(Taskbar_Identity._PropVariant))(methods[5])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(methods[Taskbar_Identity._PROPERTY_STORE_RELEASE])
    found = {}
    try:
        for property_id in property_ids:
            key = Taskbar_Identity._PropertyKey(Taskbar_Identity._guid(Taskbar_Identity._TASKBAR_PROPERTY_FORMAT), property_id)
            value = Taskbar_Identity._PropVariant()
            assert get_value(store, ctypes.byref(key), ctypes.byref(value)) >= 0  # S_FALSE (1) comes with a value too
            found[property_id] = ctypes.wstring_at(value.pointer) if value.type == Taskbar_Identity._VT_LPWSTR else None
            ole32.PropVariantClear(ctypes.byref(value))
    finally:
        release(store)
    return found


class TaskbarIdentityTests(unittest.TestCase):
    """What each main window tells the taskbar: its own ID, the small icon, its name, and how to start it again.

    A window whose ID a shortcut carries gets that shortcut's icon in the taskbar, whatever icon the window sets,
    and the desktop shortcuts carry the large icons; so the windows' IDs are not any shortcut's. No window is
    shown here, so no taskbar button appears.
    """

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.application = QApplication.instance() or QApplication([])

    def test_the_properties_go_to_windows_in_its_order_with_the_id_last(self):
        from unittest import mock

        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        widget = QWidget()
        with mock.patch.object(Taskbar_Identity.sys, "platform", "win32"), \
                mock.patch.object(Taskbar_Identity, "_set_window_texts", return_value=True) as store:
            self.assertTrue(Taskbar_Identity.set_window_taskbar_identity(
                widget, "ca.test.id", "C:/logos/small.ico", "EMAP-SSN Test", '"C:/x.exe" go'))
        hwnd, texts = store.call_args.args
        self.assertEqual(hwnd, int(widget.winId()))
        # The property IDs of PKEY_AppUserModel_RelaunchCommand, ...RelaunchIconResource, ...RelaunchDisplayNameResource
        # and ...ID in propkey.h; the icon is a path to an .ico and the index in it.
        self.assertEqual(texts, ((2, '"C:/x.exe" go'), (3, "C:/logos/small.ico,0"), (4, "EMAP-SSN Test"), (5, "ca.test.id")))

    def test_a_window_that_cannot_be_started_again_gets_its_id_alone(self):
        from unittest import mock

        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        widget = QWidget()
        with mock.patch.object(Taskbar_Identity.sys, "platform", "win32"),                 mock.patch.object(Taskbar_Identity, "_set_window_texts", return_value=True) as store:
            self.assertTrue(Taskbar_Identity.set_window_taskbar_identity(widget, "ca.test.id", "small.ico", "EMAP-SSN Test", None))
        self.assertEqual(store.call_args.args[1], ((5, "ca.test.id"),))  # no name or icon without a command to go with them

    def test_the_windows_module_gives_the_names_the_windows_and_opt_vr_import(self):
        from desktop import Desktop_App, Taskbar_Identity

        for name in ("APP_USER_MODEL_IDS", "LAUNCHER_BATCH", "LAUNCHER_EXE", "TaskbarIdentity", "launcher_relaunch_command",
                     "set_app_user_model_id", "set_window_taskbar_identity"):
            with self.subTest(name=name):
                self.assertIs(getattr(Desktop_App, name), getattr(Taskbar_Identity, name))

    def test_the_module_loads_without_the_windows_heavy_ones_so_the_splash_can_use_it(self):
        program = ("import sys; sys.path.insert(0, sys.argv[1]); from desktop import Taskbar_Identity; "
                   "print(sorted(name for name in ('desktop.Desktop_App', 'desktop.Studio_Theme', 'utilities.App_Settings') "
                   "if name in sys.modules))")
        done = subprocess.run([sys.executable, "-c", program, str(SRC)], capture_output=True, text=True, timeout=60,
                              creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        self.assertEqual(done.stdout.strip(), "[]", done.stderr)

    def test_elsewhere_it_does_nothing_and_says_so(self):
        from unittest import mock

        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        with mock.patch.object(Taskbar_Identity.sys, "platform", "linux"), \
                mock.patch.object(Taskbar_Identity, "_set_window_texts") as store:
            self.assertFalse(Taskbar_Identity.set_window_taskbar_identity(QWidget(), "id", "icon.ico", "Name", "command"))
        store.assert_not_called()

    def test_a_refusal_from_windows_is_reported_and_not_raised(self):
        from unittest import mock

        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        for failure in (False, OSError("no property store"), AttributeError("no shell32")):
            with self.subTest(failure=failure):
                options = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with mock.patch.object(Taskbar_Identity.sys, "platform", "win32"), \
                        mock.patch.object(Taskbar_Identity, "_set_window_texts", **options):
                    self.assertFalse(Taskbar_Identity.set_window_taskbar_identity(QWidget(), "id", "icon.ico", "Name", "command"))

    @unittest.skipUnless(sys.platform == "win32", "the taskbar properties are Windows'")
    def test_windows_holds_what_was_set_on_the_window(self):
        import ctypes
        from types import SimpleNamespace

        from desktop import Taskbar_Identity

        # A real, hidden native window: Qt's offscreen platform has no window handles, and nothing here may reach the taskbar.
        user32 = ctypes.windll.user32
        user32.CreateWindowExW.restype = ctypes.c_void_p
        user32.CreateWindowExW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32] + [ctypes.c_int] * 4 \
            + [ctypes.c_void_p] * 4
        user32.DestroyWindow.argtypes = [ctypes.c_void_p]
        hwnd = user32.CreateWindowExW(0, "STATIC", "EMAP-SSN identity test", 0, 0, 0, 0, 0, None, None, None, None)
        self.assertTrue(hwnd)
        self.addCleanup(user32.DestroyWindow, hwnd)
        icon = str(SRC / "bin" / "logos" / "viewer_logo.ico")
        command = Taskbar_Identity.launcher_relaunch_command("viewer")
        self.assertTrue(Taskbar_Identity.set_window_taskbar_identity(
            SimpleNamespace(winId=lambda: hwnd), "ca.test.identity", icon, "EMAP-SSN Test", command))
        self.assertEqual(read_window_texts(hwnd, (2, 3, 4, 5)), {
            2: command, 3: icon + ",0", 4: "EMAP-SSN Test", 5: "ca.test.identity"})

    def test_the_helper_object_sets_it_when_the_window_gets_its_handle_and_again_for_another(self):
        from unittest import mock

        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        widget = QWidget()
        arguments = ("ca.test.id", "icon.ico", "Name", "command")
        with mock.patch.object(Taskbar_Identity.sys, "platform", "win32"), \
                mock.patch.object(Taskbar_Identity, "set_window_taskbar_identity") as setter:
            Taskbar_Identity.TaskbarIdentity(widget, *arguments)
            setter.assert_not_called()  # no handle yet, and none is made for it
            widget.winId()
            setter.assert_called_once_with(widget, *arguments)
            self.application.sendEvent(widget, QEvent(QEvent.Type.WinIdChange))  # Qt made another
            self.assertEqual(setter.call_count, 2)
            self.application.sendEvent(widget, QEvent(QEvent.Type.Show))
            self.assertEqual(setter.call_count, 2)

    def test_the_helper_object_stays_out_of_the_way_off_windows(self):
        from unittest import mock

        from PySide6.QtWidgets import QWidget

        from desktop import Taskbar_Identity

        widget = QWidget()
        with mock.patch.object(Taskbar_Identity.sys, "platform", "linux"), \
                mock.patch.object(Taskbar_Identity, "set_window_taskbar_identity") as setter:
            Taskbar_Identity.TaskbarIdentity(widget, "id", "icon.ico", "Name", "command")
            widget.winId()
        setter.assert_not_called()

    def test_a_pinned_copy_starts_the_window_as_its_shortcut_does(self):
        from unittest import mock

        from desktop import Taskbar_Identity

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            exe = Path(folder) / "EMAP SSN" / "EMAP-SSN.exe"
            batch = Path(folder) / "EMAP SSN" / "launch.bat"
            exe.parent.mkdir()
            with mock.patch.object(Taskbar_Identity, "LAUNCHER_EXE", exe), mock.patch.dict(os.environ, {"COMSPEC": "C:\\Windows\\cmd.exe"}):
                # Built by install.bat: what the shortcut runs, with the word for the window.
                exe.write_bytes(b"MZ")
                self.assertEqual(Taskbar_Identity.launcher_relaunch_command("tools"), f'"{exe}" tools')
                self.assertEqual(Taskbar_Identity.launcher_relaunch_command("vr", batch, ""), f'"{exe}" vr')
                # Not built: the batch file in a console, as the shortcut then runs it (install.bat gives it the word,
                # opt_vr's batch file takes none).
                exe.unlink()
                self.assertEqual(Taskbar_Identity.launcher_relaunch_command("tools", batch),
                                 f'"C:\\Windows\\cmd.exe" /d /c ""{batch}" tools"')
                self.assertEqual(Taskbar_Identity.launcher_relaunch_command("vr", batch, ""),
                                 f'"C:\\Windows\\cmd.exe" /d /c ""{batch}""')

    def test_the_words_the_windows_start_with_are_the_ones_the_launcher_knows(self):
        import ast

        source = LAUNCHER_SOURCE.read_text(encoding="utf-8")
        for name, word in (("EMAPSSN_Config.py", "viewer"), ("EMAPSSN_Tools.py", "tools")):
            with self.subTest(window=name):
                calls = [node for node in ast.walk(ast.parse((SRC / name).read_text(encoding="utf-8")))
                         if isinstance(node, ast.Call) and ast.unparse(node.func) == "launcher_relaunch_command"]
                self.assertEqual([ast.unparse(call.args[0]) for call in calls], [repr(word)])
                self.assertIn(f'new[] {{ "{word}",', source)

    def test_config_and_tools_set_their_own_id_with_the_small_icon_and_no_window_opens_a_large_one(self):
        import ast

        for name, key, icon in (("EMAPSSN_Config.py", "config", "viewer_logo.ico"), ("EMAPSSN_Tools.py", "tools", "tool_logo.ico")):
            with self.subTest(window=name):
                text = (SRC / name).read_text(encoding="utf-8")
                calls = [node for node in ast.walk(ast.parse(text))
                         if isinstance(node, ast.Call) and ast.unparse(node.func) == "TaskbarIdentity"]
                self.assertEqual(len(calls), 1)
                self.assertEqual([ast.unparse(argument) for argument in calls[0].args[:3]],
                                 ["self", f"APP_USER_MODEL_IDS['{key}']", "icon_path"])
                self.assertIn(f'"{icon}"', text)
        for path in SRC.rglob("*.py"):
            with self.subTest(source=path.name):
                self.assertNotIn("_logo_large", path.read_text(encoding="utf-8"))

    def test_no_window_uses_an_id_the_earlier_installers_gave_the_shortcuts_that_are_still_on_desktops(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        # install.bat and install_vr.bat did this for a few days; a window with one of these gets the large icon again.
        earlier = {"ca.utoronto.emapssn.config", "ca.utoronto.emapssn.tools", "ca.utoronto.emapssn.vrconfig"}
        self.assertEqual(earlier & set(APP_USER_MODEL_IDS.values()), set())

    def test_the_ids_differ_from_each_other_and_the_viewer_stays_a_group_apart(self):
        from desktop.Desktop_App import APP_USER_MODEL_IDS

        self.assertEqual(len(set(APP_USER_MODEL_IDS.values())), len(APP_USER_MODEL_IDS))
        for app_id in APP_USER_MODEL_IDS.values():
            self.assertRegex(app_id, r"^ca\.utoronto\.emapssn\.[a-z.]+$")
            self.assertLessEqual(len(app_id), 128)  # Windows' limit


if __name__ == "__main__":
    unittest.main()
