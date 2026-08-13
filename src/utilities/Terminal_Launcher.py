# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Launch structured commands in a new terminal on supported desktops."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
from dataclasses import dataclass
from enum import Enum
import json
import os
import shlex
import shutil
import subprocess
import sys
from typing import Callable


class HoldMode(str, Enum):
    """Control how long the new terminal remains open."""

    NEVER = "never"
    ALWAYS = "always"
    ON_ERROR = "on_error"


class TerminalUnavailableError(RuntimeError):
    """Raised when no supported terminal emulator can be launched."""


class _ArgumentMode(str, Enum):
    DASH_DASH = "dash_dash"
    E_ARGV = "e_argv"
    X_ARGV = "x_argv"
    SINGLE_STRING = "single_string"
    WEZTERM = "wezterm"


@dataclass(frozen=True)
class _TerminalSpec:
    name: str
    argument_mode: _ArgumentMode


# Prefer desktop standards/defaults before probing named applications.
_LINUX_TERMINALS = (
    _TerminalSpec("xdg-terminal-exec", _ArgumentMode.DASH_DASH),
    _TerminalSpec("x-terminal-emulator", _ArgumentMode.E_ARGV),
    _TerminalSpec("ptyxis", _ArgumentMode.DASH_DASH),
    _TerminalSpec("gnome-terminal", _ArgumentMode.DASH_DASH),
    _TerminalSpec("kgx", _ArgumentMode.DASH_DASH),
    _TerminalSpec("konsole", _ArgumentMode.E_ARGV),
    _TerminalSpec("xfce4-terminal", _ArgumentMode.X_ARGV),
    _TerminalSpec("mate-terminal", _ArgumentMode.X_ARGV),
    _TerminalSpec("kitty", _ArgumentMode.DASH_DASH),
    _TerminalSpec("alacritty", _ArgumentMode.E_ARGV),
    _TerminalSpec("wezterm", _ArgumentMode.WEZTERM),
    _TerminalSpec("foot", _ArgumentMode.E_ARGV),
    _TerminalSpec("footclient", _ArgumentMode.E_ARGV),
    _TerminalSpec("tilix", _ArgumentMode.E_ARGV),
    _TerminalSpec("terminator", _ArgumentMode.X_ARGV),
    _TerminalSpec("qterminal", _ArgumentMode.SINGLE_STRING),
    _TerminalSpec("lxterminal", _ArgumentMode.SINGLE_STRING),
    _TerminalSpec("xterm", _ArgumentMode.E_ARGV),
    _TerminalSpec("urxvt", _ArgumentMode.E_ARGV),
    _TerminalSpec("rxvt", _ArgumentMode.E_ARGV),
    _TerminalSpec("st", _ArgumentMode.E_ARGV),
)


_POSIX_WRAPPER = r'''
hold_mode=$1
shift
title=$1
shift
if [ -n "$title" ]; then
    printf '\033]0;%s\007' "$title"
fi
"$@"
status=$?
if [ "$hold_mode" = "always" ]; then
    printf '\nProcess exited with code %s.\n' "$status"
    exec "${SHELL:-/bin/bash}" -l
fi
if [ "$hold_mode" = "on_error" ] && [ "$status" -ne 0 ]; then
    printf '\nProcess exited with code %s.\n' "$status"
    read -r -p 'Press Enter to close...' _ </dev/tty || true
fi
exit "$status"
'''.strip()


def _normalize_command(command: Sequence[os.PathLike[str] | str]) -> list[str]:
    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a sequence of arguments, not a shell string")
    normalized = [os.fspath(part) for part in command]
    if not normalized:
        raise ValueError("command must contain at least one argument")
    if not normalized[0]:
        raise ValueError("command executable must not be empty")
    return normalized


def _posix_child_command(
    command: Sequence[str], hold: HoldMode, title: str | None
) -> list[str]:
    if hold is HoldMode.NEVER and not title:
        return list(command)
    return [
        "bash",
        "-c",
        _POSIX_WRAPPER,
        "ssn-terminal",
        hold.value,
        title or "",
        *command,
    ]


def _find_linux_terminal(
    which: Callable[[str], str | None] | None = None,
) -> tuple[_TerminalSpec, str]:
    which = which or shutil.which
    for spec in _LINUX_TERMINALS:
        executable = which(spec.name)
        if executable:
            return spec, executable
    supported = ", ".join(spec.name for spec in _LINUX_TERMINALS)
    raise TerminalUnavailableError(
        "No supported terminal emulator was found. Install or configure one of: "
        f"{supported}."
    )


def _build_linux_argv(
    command: Sequence[str],
    *,
    terminal: tuple[_TerminalSpec, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> list[str]:
    spec, executable = terminal or _find_linux_terminal(which)
    if spec.argument_mode is _ArgumentMode.DASH_DASH:
        return [executable, "--", *command]
    if spec.argument_mode is _ArgumentMode.E_ARGV:
        return [executable, "-e", *command]
    if spec.argument_mode is _ArgumentMode.X_ARGV:
        return [executable, "-x", *command]
    if spec.argument_mode is _ArgumentMode.WEZTERM:
        return [executable, "start", "--", *command]
    if spec.argument_mode is _ArgumentMode.SINGLE_STRING:
        return [executable, "-e", shlex.join(command)]
    raise AssertionError(f"Unsupported terminal argument mode: {spec.argument_mode}")


def _console_python() -> str:
    executable = os.path.abspath(sys.executable)
    if os.path.basename(executable).lower() == "pythonw.exe":
        console_python = os.path.join(os.path.dirname(executable), "python.exe")
        if os.path.isfile(console_python):
            return console_python
    return executable


def _encode_windows_payload(
    command: Sequence[str], hold: HoldMode, title: str | None
) -> str:
    serialized = json.dumps(
        {"command": list(command), "hold": hold.value, "title": title or ""},
        ensure_ascii=False,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii")


def _decode_windows_payload(payload: str) -> tuple[list[str], HoldMode, str]:
    values = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
    return (
        _normalize_command(values["command"]),
        HoldMode(values["hold"]),
        str(values.get("title", "")),
    )


def _build_windows_argv(
    command: Sequence[str], hold: HoldMode, title: str | None
) -> list[str]:
    if hold is HoldMode.NEVER and not title:
        return list(command)
    payload = _encode_windows_payload(command, hold, title)
    return [_console_python(), "-u", os.path.abspath(__file__), "--windows-child", payload]


def _run_windows_child(payload: str) -> int:
    command, hold, title = _decode_windows_payload(payload)
    if title:
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except (AttributeError, OSError):
            pass
    try:
        status = int(subprocess.run(command, check=False).returncode)
    except OSError as error:
        print(f"Failed to start {command[0]}: {error}", file=sys.stderr)
        status = 127

    if hold is HoldMode.ALWAYS:
        print(f"\nProcess exited with code {status}.")
        shell = os.environ.get("COMSPEC", "cmd.exe")
        try:
            subprocess.run([shell], check=False)
        except OSError as error:
            print(f"Failed to start interactive shell {shell}: {error}", file=sys.stderr)
    elif hold is HoldMode.ON_ERROR and status != 0:
        print(f"\nProcess exited with code {status}.")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
    return status


def _mac_environment_prefix(env: Mapping[str, str] | None) -> list[str]:
    if env is None:
        return []
    current = os.environ
    removed = sorted(
        key
        for key in current
        if key not in env and (key.startswith("SSN_") or key.startswith("QT_"))
    )
    changed = sorted(
        (key, str(value))
        for key, value in env.items()
        if current.get(key) != str(value)
        or key.startswith("SSN_")
        or key.startswith("QT_")
    )
    prefix = ["env"]
    for key in removed:
        prefix.extend(["-u", key])
    prefix.extend(f"{key}={value}" for key, value in changed)
    return prefix if len(prefix) > 1 else []


def _apple_script_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_macos_argv(
    command: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str] | None,
    close_on_exit: bool,
) -> list[str]:
    terminal_command = [*_mac_environment_prefix(env), *command]
    shell_command = f"cd {shlex.quote(cwd)} && {shlex.join(terminal_command)}"
    escaped = _apple_script_escape(shell_command)
    argv = [
        "osascript",
        "-e",
        'tell application "Terminal"',
        "-e",
        "activate",
        "-e",
    ]
    if close_on_exit:
        argv.extend(
            [
                "set launchTab to do script \"\"",
                "-e",
                "set launchWindow to front window",
                "-e",
                f'do script "{escaped}" in launchTab',
                "-e",
                "repeat while busy of launchTab",
                "-e",
                "delay 0.2",
                "-e",
                "end repeat",
                "-e",
                "try",
                "-e",
                "close launchWindow",
                "-e",
                "end try",
            ]
        )
    else:
        argv.append(f'do script "{escaped}"')
    argv.extend(["-e", "end tell"])
    return argv


def launch_in_terminal(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: os.PathLike[str] | str,
    env: Mapping[str, str] | None = None,
    hold: HoldMode = HoldMode.NEVER,
    title: str | None = None,
    platform_name: str | None = None,
) -> subprocess.Popen:
    """Launch ``command`` in a new terminal without flattening its arguments."""

    normalized = _normalize_command(command)
    hold = HoldMode(hold)
    cwd_text = os.path.abspath(os.fspath(cwd))
    platform_name = platform_name or sys.platform

    popen_kwargs: dict[str, object] = {"cwd": cwd_text}
    if env is not None:
        popen_kwargs["env"] = dict(env)

    if platform_name == "win32":
        argv = _build_windows_argv(normalized, hold, title)
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_CONSOLE", 0x10
        )
    elif platform_name == "darwin":
        child = _posix_child_command(normalized, hold, title)
        argv = _build_macos_argv(
            child,
            cwd=cwd_text,
            env=env,
            close_on_exit=hold is not HoldMode.ALWAYS,
        )
    elif platform_name.startswith("linux"):
        child = _posix_child_command(normalized, hold, title)
        argv = _build_linux_argv(child)
    else:
        raise TerminalUnavailableError(
            f"Terminal launching is not supported on platform {platform_name!r}."
        )

    return subprocess.Popen(argv, **popen_kwargs)


__all__ = ["HoldMode", "TerminalUnavailableError", "launch_in_terminal"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--windows-child":
        raise SystemExit(_run_windows_child(sys.argv[2]))
    raise SystemExit("Terminal_Launcher.py is an internal utility.")
