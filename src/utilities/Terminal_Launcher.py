# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Launch structured commands in a new terminal on supported desktops.

A headless Viewer has no screen, so ``launch_without_window`` runs a command
with no window at all and its output in a log file.
"""

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
import tempfile
import threading
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


# Variables that describe the terminal a command runs in. The terminal sets them
# for its own child, so they come from the terminal, never from the caller.
_TERMINAL_VARIABLES = frozenset(
    {
        "TERM",
        "COLORTERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "VTE_VERSION",
        "WINDOWID",
        "COLUMNS",
        "LINES",
    }
)

# Callers hand a job its settings snapshot, session folder and launch id in
# SSN_ variables. The terminal program never needs them, and a single-instance
# terminal that this launch starts would pass them to every later window.
_JOB_VARIABLE_PREFIX = "SSN_"


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


def _remove_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_linux_job_file(cwd: str, environment: Mapping[str, str]) -> str:
    """Write a command's folder and complete environment for ``--posix-child``.

    The file is private (mode 0600, from mkstemp) and goes to XDG_RUNTIME_DIR,
    a per-user tmpfs, when the session has one. The child deletes it as soon as
    it has read it. A file keeps the environment, secrets included, off the
    command line, which ``ps`` shows to every user, and has no length limit.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    directory = (
        runtime
        if runtime and os.path.isdir(runtime) and os.access(runtime, os.W_OK | os.X_OK)
        else None
    )
    descriptor, path = tempfile.mkstemp(
        prefix="ssn_terminal_", suffix=".json", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"cwd": cwd, "environment": dict(environment)}, handle)
    except BaseException:
        _remove_quietly(path)
        raise
    return path


def _linux_child_command(
    command: Sequence[str], hold: HoldMode, title: str | None, job_file: str
) -> list[str]:
    # -I: the terminal's own PYTHONPATH or PYTHONHOME must not reach this step.
    return [
        _console_python(),
        "-I",
        os.path.abspath(__file__),
        "--posix-child",
        hold.value,
        job_file,
        *_posix_child_command(command, hold, title),
    ]


def _terminal_job_environment(
    job: Mapping[str, str], terminal: Mapping[str, str]
) -> dict[str, str]:
    """The job's environment, with the variables that describe the terminal."""
    environment = {
        key: value for key, value in job.items() if key not in _TERMINAL_VARIABLES
    }
    environment.update(
        (key, value) for key, value in terminal.items() if key in _TERMINAL_VARIABLES
    )
    return environment


def _run_posix_child(hold_value: str, job_file: str, command: Sequence[str]) -> int:
    """Start ``command`` in the folder and environment the job file records.

    This runs first inside the new Linux terminal. Ptyxis, which Ubuntu's
    xdg-terminal-exec starts with ``--new-window``, is single-instance: it runs
    the command with the environment of whichever process started the running
    instance, which can be an earlier job's, not the caller's. A terminal can
    also ignore the caller's folder. The command therefore takes both from the
    job file, and only the variables that describe the terminal from the
    terminal. On success this process becomes the command and never returns.
    """
    hold = HoldMode(hold_value)
    try:
        try:
            with open(job_file, encoding="utf-8") as handle:
                job = json.load(handle)
        finally:
            _remove_quietly(job_file)
        cwd = job["cwd"]
        environment = _terminal_job_environment(job["environment"], os.environ)
        environment["PWD"] = cwd
        os.chdir(cwd)
        os.execvpe(command[0], list(command), environment)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"Could not start the command in this terminal: {error}", file=sys.stderr)
    if hold is not HoldMode.NEVER:
        print("\nProcess exited with code 127.")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
    return 127


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


def _reap_when_done(process: subprocess.Popen) -> subprocess.Popen:
    """Collect the terminal command's exit status as soon as it ends.

    Most callers let the Popen go, and until something waits on it an ended
    command stays a zombie: on Ubuntu 26.04 each Save & Run left one
    ``[ptyxis] <defunct>`` under Config until Config exited (v0.4.0 Linux
    report, round 2, N3). poll() and wait() keep working for callers that
    use them.
    """
    threading.Thread(
        target=process.wait, name="terminal-command-reaper", daemon=True
    ).start()
    return process


def launch_in_terminal(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: os.PathLike[str] | str,
    env: Mapping[str, str] | None = None,
    hold: HoldMode = HoldMode.NEVER,
    title: str | None = None,
    platform_name: str | None = None,
    stdin=None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    """Launch ``command`` in a new terminal without flattening its arguments.

    ``stdin``, ``stdout`` and ``stderr`` go to ``subprocess.Popen``. On Linux
    and macOS they are the terminal program's own streams (its diagnostics),
    while ``command`` runs on the new terminal's own console; a stream the
    caller does not give is /dev/null there. The terminal program (or, on
    macOS, osascript) outlives the call, and one that inherited this
    process's descriptors would hold the pipes of the Console capture
    (utilities/Console_Capture.py): the capture's close would wait for it at
    exit, and it would die of SIGPIPE on its next write after this process
    ended. On Windows the command inherits them in place of the new console.

    ``env`` (default: this process's environment) is the command's
    environment. On Linux the command gets it, complete, and ``cwd`` from a
    private file that ``--posix-child`` reads and deletes inside the terminal,
    whatever environment and folder the terminal itself starts commands with;
    only TERM and the other variables that describe the terminal come from the
    terminal. The terminal program gets ``env`` without its SSN_ variables.

    On Linux and macOS a background thread waits on the returned process, so
    an ended terminal command never lingers as a zombie (_reap_when_done).
    """

    normalized = _normalize_command(command)
    hold = HoldMode(hold)
    cwd_text = os.path.abspath(os.fspath(cwd))
    platform_name = platform_name or sys.platform

    popen_kwargs: dict[str, object] = {"cwd": cwd_text}
    if env is not None:
        popen_kwargs["env"] = dict(env)
    for name, stream in (("stdin", stdin), ("stdout", stdout), ("stderr", stderr)):
        if stream is not None:
            popen_kwargs[name] = stream
        elif platform_name != "win32":
            popen_kwargs[name] = subprocess.DEVNULL  # not the caller's pipes (see above)

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
        terminal = _find_linux_terminal()
        environment = dict(os.environ if env is None else env)
        job_file = _write_linux_job_file(cwd_text, environment)
        try:
            argv = _build_linux_argv(
                _linux_child_command(normalized, hold, title, job_file),
                terminal=terminal,
            )
            popen_kwargs["env"] = {
                key: value
                for key, value in environment.items()
                if not key.startswith(_JOB_VARIABLE_PREFIX)
            }
            process = subprocess.Popen(argv, **popen_kwargs)
        except BaseException:
            _remove_quietly(job_file)
            raise
        return _reap_when_done(process)
    else:
        raise TerminalUnavailableError(
            f"Terminal launching is not supported on platform {platform_name!r}."
        )

    process = subprocess.Popen(argv, **popen_kwargs)
    # Windows has no zombies: an ended process goes once its handles close.
    return process if platform_name == "win32" else _reap_when_done(process)


def launch_without_window(
    command: Sequence[os.PathLike[str] | str],
    *,
    cwd: os.PathLike[str] | str,
    log_path: os.PathLike[str] | str,
    env: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> subprocess.Popen:
    """Run ``command`` where no window can appear, with its output in a log.

    This is for a headless Viewer, which has no screen. ``command`` gets no
    stdin, and its stdout and stderr both go to ``log_path``, which is opened
    for appending and whose folder is created. A Python ``command`` prints
    unbuffered UTF-8 (the log is current, and a name outside the Windows ANSI
    code page cannot crash it).

    On Windows the command starts with CREATE_NO_WINDOW. Never use
    DETACHED_PROCESS here: a venv's python.exe is a redirector that starts the
    real interpreter as its child, and a redirector without a console makes
    Windows give that child a new, visible one. A hidden console is inherited.
    On Linux and macOS the command leads its own session, and a background
    thread collects its exit status (_reap_when_done).

    ``env`` (default: this process's environment) is the command's environment.
    """

    normalized = _normalize_command(command)
    platform_name = platform_name or sys.platform
    environment = dict(os.environ if env is None else env)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    popen_kwargs: dict[str, object] = {
        "cwd": os.path.abspath(os.fspath(cwd)),
        "env": environment,
        "stdin": subprocess.DEVNULL,
    }
    if platform_name == "win32":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0x08000000
        )
    else:
        popen_kwargs["start_new_session"] = True

    log_path = os.path.abspath(os.fspath(log_path))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # The child gets its own copy of the handle, so ours can close at once.
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            normalized, stdout=log, stderr=subprocess.STDOUT, **popen_kwargs
        )
    return process if platform_name == "win32" else _reap_when_done(process)


__all__ = [
    "HoldMode",
    "TerminalUnavailableError",
    "launch_in_terminal",
    "launch_without_window",
]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--windows-child":
        raise SystemExit(_run_windows_child(sys.argv[2]))
    if len(sys.argv) >= 5 and sys.argv[1] == "--posix-child":
        raise SystemExit(_run_posix_child(sys.argv[2], sys.argv[3], sys.argv[4:]))
    raise SystemExit("Terminal_Launcher.py is an internal utility.")
