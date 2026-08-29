# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Detached monitor for Config/Tools processes started by desktop launchers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from utilities.Terminal_Launcher import (
        HoldMode,
        TerminalUnavailableError,
        launch_in_terminal,
    )
except ModuleNotFoundError:
    # This monitor is also executed directly by the bootstrap launchers.
    from Terminal_Launcher import (  # type: ignore[no-redef]
        HoldMode,
        TerminalUnavailableError,
        launch_in_terminal,
    )


APP_SCRIPTS = {
    "viewer": Path("src") / "SSN_Config.py",
    "tools": Path("src") / "SSN_Tools.py",
}
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


def _report_terminal_failure(log_path: Path, error: Exception) -> None:
    message = (
        "SSN could not open a terminal to display the application failure. "
        f"Review the retained log at {log_path}. Terminal error: {error}"
    )
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
            print(f"\n{message}", file=log_handle)
    except OSError:
        pass

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        owns_application = QApplication.instance() is None
        application = QApplication.instance() or QApplication([])
        QMessageBox.critical(None, "SSN Application Failure", message)
        if owns_application:
            application.quit()
    except Exception:
        # The retained log is the final fallback if Qt itself cannot start.
        pass


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
        # Stale successful state is harmless and can be cleaned next launch.
        pass


def launch_detached_monitor(
    app_kind: str,
    state_dir: Path,
    *,
    platform_name: str | None = None,
) -> subprocess.Popen:
    """Start the long-lived monitor without inheriting the startup terminal."""
    if app_kind not in APP_SCRIPTS:
        raise ValueError(f"Unknown desktop application kind: {app_kind}")

    project_root = Path(__file__).resolve().parents[2]
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    platform_name = platform_name or sys.platform
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        app_kind,
        str(state_dir),
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


def run_monitor(app_kind: str, state_dir: Path) -> int:
    project_root = Path(__file__).resolve().parents[2]
    app_script = project_root / APP_SCRIPTS[app_kind]
    state_dir = state_dir.resolve()
    ready_path = state_dir / "gui.ready"
    dismissed_path = state_dir / "terminal.dismissed"
    exit_path = state_dir / "application.exit"
    log_path = state_dir / "application.log"
    state_dir.mkdir(parents=True, exist_ok=True)

    env = _sanitize_managed_environment(os.environ.copy())
    env = _apply_linux_qt_platform_policy(env)
    env["SSN_GUI_READY_FILE"] = str(ready_path)
    return_code = 1
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            creationflags = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                if sys.platform == "win32"
                else 0
            )
            process = subprocess.Popen(
                [_console_python(), "-u", str(app_script)],
                cwd=str(project_root),
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
    parser.add_argument("app_kind", choices=sorted(APP_SCRIPTS))
    parser.add_argument("state_dir", type=Path)
    args = parser.parse_args(argv)
    if args.launch_and_wait:
        return launch_and_wait(args.app_kind, args.state_dir)
    return run_monitor(args.app_kind, args.state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
