# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Detached monitor for Config/Tools processes started by desktop launchers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


APP_SCRIPTS = {
    "viewer": Path("src") / "SSN_Config.py",
    "tools": Path("src") / "SSN_Tools.py",
}


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


def _error_command(log_path: Path) -> str:
    quoted_log = shlex.quote(str(log_path))
    return (
        f"cat {quoted_log}; printf '\\n\\nSSN application failed. "
        f"Log retained at: %s\\n' {quoted_log}; exec \"${{SHELL:-/bin/bash}}\" -l"
    )


def _open_error_terminal(log_path: Path, *, platform_name: str | None = None) -> bool:
    platform_name = platform_name or sys.platform
    if platform_name == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)
        command = (
            f'title SSN Error & type "{log_path}" & echo. & echo. & '
            f'echo SSN application failed. Log retained at: "{log_path}"'
        )
        subprocess.Popen(
            ["cmd.exe", "/d", "/k", command],
            creationflags=creationflags,
            cwd=str(log_path.parent),
        )
        return True

    command = _error_command(log_path)
    if platform_name == "darwin":
        escaped = command.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(
            [
                "osascript",
                "-e", 'tell application "Terminal"',
                "-e", "activate",
                "-e", f'do script "{escaped}"',
                "-e", "end tell",
            ]
        )
        return True

    terminals = (
        "gnome-terminal", "konsole", "xfce4-terminal", "mate-terminal",
        "lxterminal", "kitty", "alacritty", "xterm", "x-terminal-emulator",
    )
    terminal = next((name for name in terminals if shutil.which(name)), None)
    if terminal is None:
        return False
    if terminal in {"gnome-terminal", "kitty", "alacritty"}:
        argv = [terminal, "--", "bash", "-c", command]
    elif terminal == "konsole":
        argv = [terminal, "-e", "bash", "-c", command]
    else:
        argv = [terminal, "-e", "bash", "-c", command]
    subprocess.Popen(argv)
    return True


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


def run_monitor(app_kind: str, state_dir: Path) -> int:
    project_root = Path(__file__).resolve().parents[2]
    app_script = project_root / APP_SCRIPTS[app_kind]
    state_dir = state_dir.resolve()
    ready_path = state_dir / "gui.ready"
    dismissed_path = state_dir / "terminal.dismissed"
    exit_path = state_dir / "application.exit"
    log_path = state_dir / "application.log"
    state_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
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
        except OSError:
            pass
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app_kind", choices=sorted(APP_SCRIPTS))
    parser.add_argument("state_dir", type=Path)
    args = parser.parse_args(argv)
    return run_monitor(args.app_kind, args.state_dir)


if __name__ == "__main__":
    raise SystemExit(main())
