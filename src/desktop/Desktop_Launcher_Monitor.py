# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Detached monitor for Config, Tools and Viewer processes started by desktop launchers.

It runs the window with a hidden console, keeps what it prints in application.log (the window
captures its own output for the Console window as well, utilities/Console_Capture.py), writes
application.exit when the window ends, and reports a crash after the window opened: in a terminal
that shows the log (the launch scripts' terminal start, batch and .sh), or, with --failure-report
dialog, in a message box (the Python launcher, desktop/Desktop_Launcher.py, which has no terminal, on every platform).
The launcher that started it follows the window's progress in startup.status (utilities/Startup_Status.py)
and its readiness in gui.ready.

A program that sits beside this one, as opt_vr does, adds its own window with register_application: the
script to run, the folder to run it in and the monitor script that is started for it (a few lines that
register the window and call main()). This file names none of them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import NamedTuple

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from utilities.Localization import Message
    from utilities.Terminal_Launcher import (
        HoldMode,
        TerminalUnavailableError,
        launch_in_terminal,
    )
except ModuleNotFoundError:
    from Localization import Message  # type: ignore[no-redef]
    from Terminal_Launcher import (  # type: ignore[no-redef]
        HoldMode,
        TerminalUnavailableError,
        launch_in_terminal,
    )


APP_SCRIPTS = {
    "viewer": Path("src") / "EMAPSSN_Config.py",    # "viewer" is the Configuration window, as in the batch file
    "tools": Path("src") / "EMAPSSN_Tools.py",
    "session": Path("src") / "EMAPSSN_Viewer.py",   # a Viewer window that Config starts
}
FAILURE_REPORTS = ("terminal", "dialog")

# The windows other programs add (register_application): kind -> Application.
EXTRA_APPLICATIONS: dict = {}


class Application(NamedTuple):
    """A window a program beside this one adds: its script, the folder it runs in and the monitor script for it."""

    script: Path
    working_dir: Path
    monitor: Path | None = None


def register_application(kind, script, working_dir, monitor=None):
    """Teach the monitor a window of another program; the monitor script re-enters itself by ``monitor`` (default: this file).

    The monitor is a process of its own that starts the window, so the program's monitor script, not the
    launcher that started it, makes this call in that process, and the launcher's process makes it too so
    that launch_detached_monitor finds the script to start.
    """
    EXTRA_APPLICATIONS[kind] = Application(
        Path(script), Path(working_dir), None if monitor is None else Path(monitor)
    )


def application_kinds():
    """Every kind the monitor can run: its own windows and the ones registered."""
    return sorted({*APP_SCRIPTS, *EXTRA_APPLICATIONS})


STATUS_FILE_NAME = "startup.status"
STARTUP_READY = 0
STARTUP_APPLICATION_EXITED = 20
STARTUP_TIMEOUT = 21
STARTUP_LAUNCH_FAILED = 22
STARTUP_TIMEOUT_SECONDS = 600.0
STARTUP_POLL_SECONDS = 0.05

_MANAGED_ENVIRONMENT_PATH_OVERRIDES = (
    "PYTHONHOME",
    "PYTHONPATH",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QML_IMPORT_PATH",
    "QML2_IMPORT_PATH",
)


def _sanitize_managed_environment(environment, *, platform_name=None):
    """Remove inherited paths that can redirect the managed Python/Qt stack."""
    platform_name = platform_name or sys.platform
    for name in _MANAGED_ENVIRONMENT_PATH_OVERRIDES:
        environment.pop(name, None)
    if platform_name == "darwin":
        environment.pop("DYLD_LIBRARY_PATH", None)
        environment.pop("DYLD_FRAMEWORK_PATH", None)
    return environment


def _apply_linux_qt_platform_policy(environment, *, platform_name=None):
    """Prefer XWayland for Qt/OpenGL when a desktop launch bypasses the shell."""
    platform_name = platform_name or sys.platform
    if (
        platform_name.startswith("linux")
        and environment.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        and not environment.get("QT_QPA_PLATFORM")
    ):
        environment["QT_QPA_PLATFORM"] = "xcb"
    return environment


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _console_python() -> str:
    executable = Path(sys.executable)
    if sys.platform == "win32" and executable.name.lower() == "pythonw.exe":
        console_python = executable.with_name("python.exe")
        if console_python.is_file():
            return str(console_python)
    return str(executable)


_ERROR_DISPLAY_PROGRAM = """\
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
print(log_path.read_text(encoding="utf-8", errors="replace"), end="")
print(f"\\n\\nSSN application failed. Log retained at: {log_path}")
"""


def _open_error_terminal(log_path: Path, *, platform_name: str | None = None) -> bool:
    launch_in_terminal(
        [_console_python(), "-u", "-c", _ERROR_DISPLAY_PROGRAM, str(log_path)],
        cwd=log_path.parent,
        hold=HoldMode.ALWAYS,
        title="SSN Error",
        platform_name=platform_name,
    )
    return True


def _install_chosen_language(application) -> None:
    """Show the monitor's dialog in the language the windows show.

    The monitor runs in a process of its own, which no window's language
    reaches. If the language can't be loaded, the dialog shows in English.
    """
    try:
        from desktop.Desktop_App import install_translations, startup_language

        install_translations(application, startup_language())
    except Exception:
        pass


def _show_failure_dialog(message: Message) -> None:
    """A message box with the failure's text, in the language the windows show; nothing if Qt cannot show one."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        owns_application = QApplication.instance() is None
        application = QApplication.instance() or QApplication([])
        if owns_application:
            _install_chosen_language(application)
        QMessageBox.critical(None, Message("SSN Application Failure").display(), message.display())
        if owns_application:
            application.quit()
    except Exception:
        pass


def _report_terminal_failure(log_path: Path, error: Exception) -> None:
    message = Message(
        "SSN could not open a terminal to display the application failure. "
        "Review the retained log at {log}. Terminal error: {error}",
        log=log_path,
        error=error,
    )
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            print(f"\n{message}", file=log_handle)  # The log stays English.
    except OSError:
        pass
    _show_failure_dialog(message)


# How many of the log's last lines the crash dialog shows.
CRASH_DIALOG_LINES = 12


def _report_application_failure(log_path: Path) -> None:
    """A window ended in failure after it opened, and no terminal is open to show why: tell it in a dialog.

    The dialog is the one _report_terminal_failure uses when a terminal cannot open. It names the
    log, which keeps the whole output, and shows the end of it.
    """
    try:
        lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        lines = []
    _show_failure_dialog(Message(
        "EMAP-SSN stopped unexpectedly. The last lines it printed:\n\n{lines}\n\nThe whole output is in {log}.",
        lines="\n".join(lines[-CRASH_DIALOG_LINES:]) or "-",
        log=log_path,
    ))


def _wait_for_dismissal(path: Path, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return path.exists()


def _cleanup_success(state_dir: Path) -> None:
    try:
        shutil.rmtree(state_dir)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def launch_detached_monitor(
    app_kind: str,
    state_dir: Path,
    *,
    platform_name: str | None = None,
    failure_report: str = "terminal",
    app_args: tuple[str, ...] | list[str] = (),
    project_root: Path | None = None,
    python: str | None = None,
) -> subprocess.Popen:
    """Start the long-lived monitor without inheriting the startup terminal.

    failure_report says how it reports a crash after the window opened ("terminal" or "dialog"), and
    app_args go to the window's script. The monitor of project_root's own copy runs, with its own
    interpreter (python), so a launcher can start a project other than the one it sits in (tests). A
    window another program added (register_application) is watched by the monitor script it named.
    """
    registered = EXTRA_APPLICATIONS.get(app_kind)
    if app_kind not in APP_SCRIPTS and registered is None:
        raise ValueError(f"Unknown desktop application kind: {app_kind}")
    if failure_report not in FAILURE_REPORTS:
        raise ValueError(f"Unknown failure report: {failure_report}")

    own_root = Path(__file__).resolve().parents[2]
    project_root = own_root if project_root is None else Path(project_root).resolve()
    monitor = Path(__file__).resolve() if project_root == own_root else project_root / "src" / "desktop" / Path(__file__).name
    if registered is not None and registered.monitor is not None:
        monitor = registered.monitor
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    platform_name = platform_name or sys.platform
    command = [
        python or sys.executable,
        "-u",
        str(monitor),
        *(["--failure-report", failure_report] if failure_report != "terminal" else []),
        app_kind,
        str(state_dir),
        *app_args,
    ]
    popen_kwargs: dict[str, object] = {
        "cwd": str(project_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if platform_name == "win32":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000
        )
    else:
        popen_kwargs["start_new_session"] = True

    return subprocess.Popen(command, **popen_kwargs)


def launch_viewer_session(
    project_root: Path | str,
    env: dict,
    *,
    settings_path: Path | str | None = None,
    executable: str | None = None,
    platform_name: str | None = None,
) -> subprocess.Popen:
    """Start a Viewer window for Config with no terminal or console of its own, and a splash while it loads.

    The Python launcher (desktop/Desktop_Launcher.py, "session") shows the splash, runs the window under
    the monitor with a hidden console and keeps its output in a log: the window keeps its own output for
    its Console window (utilities/Console_Capture.py), so nothing needs a terminal. ``env`` is the
    window's environment (its cache and settings snapshot).

    On Windows the launcher runs with a hidden console; on Linux and macOS it leads a session of its own
    (as the monitor does), so that closing Config does not end it. The caller waits on the returned process
    to collect it, and may start the terminal launch if it ends with a failure before it started anything
    (EMAPSSN_Config._handoff_to_viewer).
    """
    project_root = Path(project_root).resolve()
    command = [
        executable or sys.executable,
        "-u",
        str(project_root / "src" / "desktop" / "Desktop_Launcher.py"),
        "session",
    ]
    if settings_path:
        command.extend(["--settings", os.fspath(settings_path), "--delete-settings"])
    options: dict[str, object] = {}
    if (platform_name or sys.platform) == "win32":
        options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    else:
        options["start_new_session"] = True
    return subprocess.Popen(
        command,
        cwd=str(project_root),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        **options,
    )


def wait_for_startup_state(
    state_dir: Path,
    monitor_process: subprocess.Popen,
    *,
    timeout: float = STARTUP_TIMEOUT_SECONDS,
    poll_interval: float = STARTUP_POLL_SECONDS,
) -> str:
    """Wait for readiness, an early application exit, or monitor failure."""
    state_dir = state_dir.resolve()
    ready_path = state_dir / "gui.ready"
    exit_path = state_dir / "application.exit"
    deadline = time.monotonic() + max(0.0, float(timeout))
    poll_interval = max(0.001, float(poll_interval))

    while True:
        if ready_path.is_file():
            return "ready"
        if exit_path.is_file():
            return "application_exited"
        if monitor_process.poll() is not None:
            return "monitor_exited"

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "timeout"
        time.sleep(min(poll_interval, remaining))


def launch_and_wait(app_kind: str, state_dir: Path) -> int:
    """Detach the monitor, then keep the startup terminal until Qt is ready."""
    try:
        monitor_process = launch_detached_monitor(app_kind, state_dir)
    except (OSError, ValueError) as error:
        print(f"Failed to launch the detached desktop monitor: {error}", file=sys.stderr)
        return STARTUP_LAUNCH_FAILED

    state = wait_for_startup_state(state_dir, monitor_process)
    if state == "ready":
        return STARTUP_READY
    if state == "application_exited":
        return STARTUP_APPLICATION_EXITED
    if state == "timeout":
        return STARTUP_TIMEOUT

    print(
        "The detached desktop monitor exited before reporting GUI readiness.",
        file=sys.stderr,
    )
    return STARTUP_LAUNCH_FAILED


def run_monitor(
    app_kind: str,
    state_dir: Path,
    *,
    app_args: tuple[str, ...] | list[str] = (),
    failure_report: str = "terminal",
) -> int:
    project_root = Path(__file__).resolve().parents[2]
    registered = EXTRA_APPLICATIONS.get(app_kind)
    if registered is not None:
        app_script, working_dir = registered.script, registered.working_dir
    else:
        app_script, working_dir = project_root / APP_SCRIPTS[app_kind], project_root
    state_dir = state_dir.resolve()
    ready_path = state_dir / "gui.ready"
    dismissed_path = state_dir / "terminal.dismissed"
    exit_path = state_dir / "application.exit"
    log_path = state_dir / "application.log"
    state_dir.mkdir(parents=True, exist_ok=True)

    env = _sanitize_managed_environment(os.environ.copy())
    env = _apply_linux_qt_platform_policy(env)
    env["SSN_GUI_READY_FILE"] = str(ready_path)
    env["SSN_STARTUP_STATUS_FILE"] = str(state_dir / STATUS_FILE_NAME)
    return_code = 1
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                if sys.platform == "win32"
                else 0
            )
            process = subprocess.Popen(
                [_console_python(), "-u", str(app_script), *app_args],
                cwd=str(working_dir),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            _atomic_write(state_dir / "application.pid", f"{process.pid}\n")
            return_code = int(process.wait())
    except Exception as error:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            print(f"Desktop launcher monitor failed: {error}", file=log_handle)

    ready_seen = ready_path.exists()
    _atomic_write(exit_path, f"{return_code}\n")

    if return_code == 0:
        _wait_for_dismissal(dismissed_path)
        _cleanup_success(state_dir)
        return 0

    if ready_seen and _wait_for_dismissal(dismissed_path):
        if failure_report == "dialog":
            _report_application_failure(log_path)
        else:
            try:
                _open_error_terminal(log_path)
            except (OSError, TerminalUnavailableError) as error:
                _report_terminal_failure(log_path, error)
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch-and-wait",
        action="store_true",
        help="detach the monitor and wait for GUI startup state",
    )
    parser.add_argument(
        "--failure-report",
        choices=FAILURE_REPORTS,
        default="terminal",
        help="how to report a crash after the window opened: a terminal that shows the log, or a dialog",
    )
    parser.add_argument("app_kind", choices=application_kinds())
    parser.add_argument("state_dir", type=Path)
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="arguments for the window's script")
    args = parser.parse_args(argv)
    if args.launch_and_wait:
        return launch_and_wait(args.app_kind, args.state_dir)
    return run_monitor(
        args.app_kind, args.state_dir, app_args=args.app_args, failure_report=args.failure_report
    )


if __name__ == "__main__":
    raise SystemExit(main())
