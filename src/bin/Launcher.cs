// Copyright 2026 Xuebin Feng
// Author affiliation: University of Toronto
// SPDX-License-Identifier: Apache-2.0
//
// EMAP-SSN.exe: what the desktop shortcuts start.
//
// A program with no console of its own (the Windows GUI subsystem), a few kilobytes, built by install.bat with
// the C# compiler that ships with Windows (csc.exe, no download; src\bin\Build_Launcher.bat does it). It starts
// the Python launcher of the window it was asked for, hidden:
//
//     .venv\Scripts\python.exe -u src\desktop\Desktop_Launcher.py <viewer|tools>
//
// which shows the splash within a third of a second and starts the window (see that file). The shortcut
// therefore opens no console window, as one that runs the batch file does.
//
// It falls back to the old visible console (src\bin\EMAPSSN_Desktop_Launcher.bat), which also sets up or repairs
// the environment, when there is no managed Python yet (the first start), when the Python launcher fails before
// it shows anything, and when asked: EMAP-SSN.exe viewer --console. "viewer" is the EMAP-SSN Configuration
// window, as in the batch file, "tools" is EMAP-SSN Tools, and "vr" is the VR Configuration window of opt_vr:
// the one thing here that knows opt_vr's folder. Its launcher script and its console launch are opt_vr's own
// (opt_vr\src\bin), and everything else the VR start does (its steps, its monitor, its single-instance key)
// is declared there; see the Kinds table below.
//
// Build (install.bat does this): csc /nologo /target:winexe /win32icon:src\bin\logos\viewer_logo_large.ico
//     /out:EMAP-SSN.exe src\bin\Launcher.cs

using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

static class Launcher
{
    // Variables that would point the managed Python or Qt at another installation (the batch file clears them too).
    static readonly string[] ManagedOverrides =
    {
        "PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML_IMPORT_PATH", "QML2_IMPORT_PATH"
    };

    // The windows it can start, each as: the word that names it, the Python launcher that shows the splash and
    // starts it, and the batch file that does the same in a visible console (the fallback), the last two relative
    // to the folder this program is in. The first is what no word at all means.
    static readonly string[][] Kinds =
    {
        new[] { "viewer", @"src\desktop\Desktop_Launcher.py", @"src\bin\EMAPSSN_Desktop_Launcher.bat" },
        new[] { "tools", @"src\desktop\Desktop_Launcher.py", @"src\bin\EMAPSSN_Desktop_Launcher.bat" },
        new[] { "vr", @"opt_vr\src\bin\Desktop_Launcher_VR.py", @"opt_vr\src\bin\EMAPSSN_Desktop_Launcher_VR.bat" },
    };

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int MessageBox(IntPtr window, string text, string caption, uint type);

    [STAThread]
    static int Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
        string[] chosen = Kinds[0];
        bool console = false;
        foreach (string argument in args)
        {
            if (argument == "--console") console = true;
            else foreach (string[] known in Kinds) if (argument == known[0]) chosen = known;
        }
        string kind = chosen[0];

        string python = Path.Combine(root, ".venv", "Scripts", "python.exe");
        string script = Path.Combine(root, chosen[1]);
        string batch = Path.Combine(root, chosen[2]);

        if (console || !File.Exists(python) || !File.Exists(script))
            return OpenConsole(root, batch, kind);

        int code = Run(python, "-u \"" + script + "\" " + kind, root);
        // 0: the launcher showed the window, or something it could report. Anything else, including Python
        // failing to start, means nothing was shown: the visible console launch tells what is wrong.
        return code == 0 ? 0 : OpenConsole(root, batch, kind);
    }

    static ProcessStartInfo Start(string program, string arguments, string root, bool window)
    {
        var info = new ProcessStartInfo
        {
            FileName = program,
            Arguments = arguments,
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = !window,
        };
        foreach (string name in ManagedOverrides) info.EnvironmentVariables.Remove(name);
        return info;
    }

    // Runs the Python launcher with no window and waits for it; -1 if it could not be started.
    static int Run(string python, string arguments, string root)
    {
        try
        {
            using (Process process = Process.Start(Start(python, arguments, root, false)))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (Exception)
        {
            return -1;
        }
    }

    // The old launch: the batch file in a console window of its own, which stays while it works.
    static int OpenConsole(string root, string batch, string kind)
    {
        if (!File.Exists(batch))
        {
            MessageBox(IntPtr.Zero, "EMAP-SSN cannot start: " + batch + " is missing. Reinstall it, or run install.bat again.",
                       "EMAP-SSN", 0x10);
            return 1;
        }
        string shell = Environment.GetEnvironmentVariable("COMSPEC");
        if (string.IsNullOrEmpty(shell)) shell = "cmd.exe";
        try
        {
            Process.Start(Start(shell, "/d /c \"\"" + batch + "\" " + kind + "\"", root, true));
            return 0;
        }
        catch (Exception error)
        {
            MessageBox(IntPtr.Zero, "EMAP-SSN could not open its launcher console: " + error.Message, "EMAP-SSN", 0x10);
            return 1;
        }
    }
}
