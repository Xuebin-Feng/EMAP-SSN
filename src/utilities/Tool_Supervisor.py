# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Run a tool script under a light parent that restarts it once after a GPU out-of-memory crash.

A GPU out-of-memory error can leave the CUDA context unusable ("sticky"), and
PyTorch then ends the whole process from C++ while it frees memory (Windows exit
code 0xC0000409, POSIX SIGABRT). Nothing in that process can go on. The tool
scripts therefore call :func:`supervise` first: the first process becomes a
parent that starts the same command again as a child and waits. The parent
starts the child once more when both hold:

* the child wrote a note that it had caught a GPU out-of-memory error
  (:func:`record_note`), and
* the child then ended abnormally (:func:`is_abnormal_exit`).

Every other end (exit 0, a normal failure, Ctrl+C, a kill, an abnormal end with no
note) is passed on unchanged and prints nothing. The restart happens at most once.

Only the standard library is imported here, so the parent never loads torch or
numpy; psutil is imported by the child's parent watcher alone.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

OFF_VARIABLE = "SSN_TOOL_SUPERVISOR"  # "0" runs the tool in one process
PARENT_VARIABLE = "SSN_TOOL_SUPERVISOR_PARENT"  # the parent's PID, set for the child
NOTE_VARIABLE = "SSN_TOOL_SUPERVISOR_NOTE"  # the path of the note the child may write
WAIT_VARIABLE = "SSN_TOOL_SUPERVISOR_WAIT"  # the failed run's memory, set for the restarted child
_MARKERS = (PARENT_VARIABLE, NOTE_VARIABLE, WAIT_VARIABLE)

NOTE_PREFIX = "ssn_tool_oom_"
STALE_NOTE_SECONDS = 7 * 24 * 3600  # longer than any run: a live tool's note is never taken for a stale one
WATCH_INTERVAL_SECONDS = 1.0
GPU_WAIT_SECONDS = 60.0
RESTART_MEMORY_FRACTION = 0.5  # the share of the free memory a restarted tool plans with
GPU_WAIT_INTERVAL_SECONDS = 1.0
GIB = 1024 ** 3

_WINDOWS_ABORT_CODE = 3  # the C runtime's abort()
_WINDOWS_CONTROL_C = 0xC000013A  # Ctrl+C, or the console closed
_STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGKILL") if hasattr(signal, name)
)
_POSIX_CRASH_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGABRT", "SIGSEGV", "SIGBUS", "SIGILL", "SIGFPE")
    if hasattr(signal, name)
)

# What the child read from its environment (None in a process that is not a supervised child).
_note_path = None
_restart_info = None
_parent_pid = None
_observed = {}  # device key -> (free_bytes, total_bytes) of the largest free memory this process saw
_wait_done = False
_reduction_announced = False


# ---------------------------------------------------------------------------
# Which ends count as a crash
# ---------------------------------------------------------------------------
def is_abnormal_exit(code, platform=None):
    """Whether a child's exit status is a crash a GPU memory error can cause.

    Windows: an NTSTATUS error code (0xC0000000 and above) other than 0xC000013A
    (Ctrl+C, console closed), or 3 (the C runtime's abort). POSIX: ended by SIGABRT,
    SIGSEGV, SIGBUS, SIGILL or SIGFPE. A tool's own failure (exit 1), a signal sent
    to stop it (SIGINT, SIGTERM, SIGKILL, SIGHUP) and a clean exit are not crashes.
    """
    if not isinstance(code, int) or isinstance(code, bool):
        return False
    if (platform or sys.platform) == "win32":
        unsigned = code & 0xFFFFFFFF
        if unsigned == _WINDOWS_CONTROL_C:
            return False
        return unsigned == _WINDOWS_ABORT_CODE or unsigned >= 0xC0000000
    return code < 0 and -code in _POSIX_CRASH_SIGNALS


def _describe_exit(code, platform=None):
    if (platform or sys.platform) == "win32":
        return f"exit code 0x{code & 0xFFFFFFFF:08X}"
    if isinstance(code, int) and code < 0:
        try:
            return f"signal {-code} ({signal.Signals(-code).name})"
        except ValueError:
            return f"signal {-code}"
    return f"exit code {code}"


# ---------------------------------------------------------------------------
# The child's side: the note, the parent watcher, the wait
# ---------------------------------------------------------------------------
def _device_key(device):
    """"cuda:0" for 0, "cuda", "cuda:0" and torch.device("cuda:0")."""
    text = str(device).strip().lower()
    if text == "cuda":
        return "cuda:0"
    return text


def observe_gpu_memory(device, free_bytes, total_bytes):
    """Remember the largest free memory seen on a device, for a note that has no reading of its own."""
    key = _device_key(device)
    seen = _observed.get(key)
    if seen is None or int(free_bytes) > seen[0]:
        _observed[key] = (int(free_bytes), int(total_bytes))


def record_note(device, error, *, where="", free_bytes=None, total_bytes=None, system=None, system_before=None):
    """Tell the supervising parent that a GPU out-of-memory error was caught.

    Returns True when a note was written, and never raises. The first note of a run
    stays: the memory read before the first failing test is the one worth keeping.
    ``system`` and ``system_before`` are the computer's memory (commit and available
    RAM, Hardware_Acceleration.read_system_memory) at the failure and before the test.
    """
    path = _note_path
    if not path:
        return False
    key = _device_key(device)
    if free_bytes is None or total_bytes is None:
        seen = _observed.get(key)
        if seen is not None:
            free_bytes, total_bytes = seen
    note = {
        "device": key,
        "where": str(where),
        "error": f"{type(error).__name__}: {error}"[:500],
        "free_bytes": None if free_bytes is None else int(free_bytes),
        "total_bytes": None if total_bytes is None else int(total_bytes),
    }
    for name, value in (("system", system), ("system_before", system_before)):
        if isinstance(value, dict):
            note[name] = {key: int(item) for key, item in value.items() if isinstance(item, (int, float))}
    try:
        if os.path.exists(path):
            return True
        partial = f"{path}.{os.getpid()}.part"
        with open(partial, "w", encoding="utf-8") as handle:
            json.dump(note, handle)
        os.replace(partial, path)
        return True
    except OSError:
        return False


def _kill_tree_and_exit():
    """End this process and everything it started, at once."""
    import psutil

    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.kill()
        except psutil.Error:
            pass
    me.kill()  # no os._exit: the program has one sanctioned hard exit (Console_Capture), and a kill needs no cleanup


def _watch_parent(parent_pid, interval=WATCH_INTERVAL_SECONDS, on_gone=_kill_tree_and_exit):
    """End this process within a few seconds of its supervising parent ending alone."""
    try:
        import psutil

        parent = psutil.Process(parent_pid)  # remembers the creation time: a reused PID is not the parent
    except Exception:
        return None

    def watch():
        while True:
            time.sleep(interval)
            try:
                gone = not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                gone = True
            except psutil.Error:
                continue
            if gone:
                on_gone()
                return

    thread = threading.Thread(target=watch, name="tool-parent-watch", daemon=True)
    thread.start()
    return thread


def _become_child():
    """Read and remove the parent's marker variables, then watch the parent."""
    global _note_path, _restart_info, _parent_pid
    parent_text = os.environ.pop(PARENT_VARIABLE, "")
    _note_path = os.environ.pop(NOTE_VARIABLE, "") or None
    wait_text = os.environ.pop(WAIT_VARIABLE, "")
    if wait_text:
        try:
            info = json.loads(wait_text)
            _restart_info = info if isinstance(info, dict) else None
        except ValueError:
            _restart_info = None
    try:
        parent_pid = int(parent_text)
    except ValueError:
        return
    _parent_pid = parent_pid
    _watch_parent(parent_pid)


def restart_info():
    """What the failed run recorded (device, free and total bytes) in a restarted child, else None."""
    return None if _wait_done else _restart_info


def _gib(value):
    return f"{value / GIB:.1f} GB"


def planning_free_bytes(device, free_bytes, total_bytes, out=print):
    """The free memory a plan is sized from: all of it, or in a tool restarted after a GPU crash a fraction.

    Only the device the failed run's note names, and for the whole restarted run (the wait for the
    memory to come back uses the real figure and comes first). Whatever shows or records the real free
    memory (the diagnostic, the note) does not call this. The reduction is announced once.
    """
    global _reduction_announced
    info = _restart_info
    free_bytes = int(free_bytes)
    if not info or _device_key(device) != _device_key(info.get("device") or ""):
        return free_bytes
    reduced = int(free_bytes * RESTART_MEMORY_FRACTION)
    if not _reduction_announced:
        _reduction_announced = True
        share = "half" if RESTART_MEMORY_FRACTION == 0.5 else f"{RESTART_MEMORY_FRACTION:.0%}"
        out(
            f"[Hardware] After the restart, GPU plans use {share} of the free memory "
            f"({reduced / GIB:.1f} of {free_bytes / GIB:.1f} GB)."
        )
    return reduced


def wait_for_gpu_memory(device, read_memory, *, timeout=GPU_WAIT_SECONDS, interval=GPU_WAIT_INTERVAL_SECONDS,
                        sleep=None, clock=None, out=print):
    """In a restarted child, once: wait until the device's free memory is back to its old level.

    ``read_memory()`` returns (free_bytes, total_bytes) or None. The target is the free
    memory the failed run recorded before its failed test, less a margin for what the
    desktop takes. The wait is bounded; the run goes on either way, with one line of
    numbers. Returns True when the wait ran. Does nothing in any other process, for any
    other device, and after the first call.
    """
    global _wait_done
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    info = _restart_info
    if _wait_done or not info or not info.get("free_bytes"):
        return False
    if _device_key(device) != _device_key(info.get("device", "")):
        return False
    _wait_done = True
    wanted = int(info["free_bytes"])
    total = int(info.get("total_bytes") or 0)
    target = max(0, wanted - max(512 * 1024 ** 2, int(total * 0.05)))
    started = clock()
    free = None
    while True:
        try:
            reading = read_memory()
        except Exception:
            reading = None
        if reading is not None:
            free = int(reading[0])
            if free >= target:
                waited = clock() - started
                out(
                    f"[Hardware] GPU memory is back after {waited:.0f} s: {_gib(free)} free "
                    f"({_gib(wanted)} before the failed test)."
                )
                return True
        if clock() - started >= timeout:
            have = "no reading" if free is None else f"{_gib(free)} free"
            out(
                f"[Hardware] GPU memory did not come back within {timeout:g} s ({have}, "
                f"{_gib(wanted)} before the failed test); going on."
            )
            return True
        sleep(interval)


# ---------------------------------------------------------------------------
# The parent's side
# ---------------------------------------------------------------------------
def _supervision_wanted(module_name, module_file):
    if module_name != "__main__":
        return False
    if os.environ.get(OFF_VARIABLE, "").strip() == "0":
        return False
    # Only a real run of the script: not an exec of its text inside another program.
    script = sys.argv[0] if sys.argv else ""
    if not script or not module_file:
        return False
    try:
        return os.path.samefile(script, module_file)
    except OSError:
        return False


def _command():
    arguments = getattr(sys, "orig_argv", None)
    if arguments:
        return [sys.executable, *arguments[1:]]
    return [sys.executable, *sys.argv]


def _sweep_stale_notes(folder):
    try:
        names = os.listdir(folder)
    except OSError:
        return
    cutoff = time.time() - STALE_NOTE_SECONDS
    for name in names:
        if name.startswith(NOTE_PREFIX):
            path = os.path.join(folder, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
            except OSError:
                pass


def _read_note(path):
    """The child's note as a dict ({} for an unreadable one), or None when there is none."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            note = json.load(handle)
        return note if isinstance(note, dict) else {}
    except (OSError, ValueError):
        return {}


def _remove(path):
    """Delete the note and any half-written copy of it a killed child left."""
    folder, name = os.path.split(path)
    try:
        leftovers = [entry for entry in os.listdir(folder) if entry == name or entry.startswith(name + ".")]
    except OSError:
        leftovers = [name]
    for entry in leftovers:
        try:
            os.remove(os.path.join(folder, entry))
        except OSError:
            pass


def _wait_for_child(process):
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            # Ctrl+C reaches the child too; it decides how to end, and its code is passed on.
            continue


def _forward_stop_signals(process):
    """POSIX: a SIGTERM or SIGHUP sent to the parent alone reaches the child."""
    if os.name == "nt":
        return
    def forward(signum, _frame):
        try:
            process.send_signal(signum)
        except OSError:
            pass
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), forward)
            except (ValueError, OSError):
                pass


def _pass_on(code):
    """End this process with the child's status.

    Windows: the same code. POSIX: a child that someone stopped (SIGINT, SIGTERM, SIGHUP, SIGKILL)
    is mirrored by ending the same way, so the status stays truthful. Any other signal is passed on
    as 128+n and not raised again: re-raising a crash signal would write a second core dump (Linux)
    or crash report (macOS) for a process that did not crash.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if os.name == "nt":
        sys.exit(code - 0x100000000 if code >= 0x80000000 else code)
    if code < 0:
        if -code in _STOP_SIGNALS:
            try:
                try:
                    signal.signal(-code, signal.SIG_DFL)
                except (ValueError, OSError):
                    pass  # Python refuses a handler for SIGKILL; the signal needs none, so still raise it
                os.kill(os.getpid(), -code)
                time.sleep(1.0)
            except (ValueError, OSError):
                pass
        sys.exit(128 - code)
    sys.exit(code)


def signal_of_exit(code, platform=None):
    """The signal number n when ``code`` is what :func:`_pass_on` makes of a child ended by signal n, else None.

    The inverse of the rule in ``_pass_on``: a signal that is not a stop signal ends the parent with
    exit code 128+n, so a reader of that code (an MCP job) can name the signal again. Codes up to 128,
    codes that are no signal number, a stop signal (mirrored, never turned into a code) and Windows give None.
    """
    if (platform or sys.platform) == "win32":
        return None
    if not isinstance(code, int) or isinstance(code, bool) or code <= 128:
        return None
    number = code - 128
    try:
        signal.Signals(number)
    except ValueError:
        return None
    return None if number in _STOP_SIGNALS else number


@functools.lru_cache(maxsize=64)
def _script_calls_supervise(path, size, mtime_ns):
    """Whether the script at ``path`` calls ``Tool_Supervisor.supervise`` at the top of its module."""
    try:
        with open(path, "rb") as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, ValueError):
        return False
    return any(
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "Tool_Supervisor.supervise"
        for node in tree.body
    )


def is_supervised_script(script_path):
    """Whether a run of this script starts under the watching parent: it calls ``supervise`` and the switch is on.

    The knowledge is in the tool scripts themselves, so a tool that gains the call is recognised with no list.
    The script's text is parsed (never run) once per file version.
    """
    if os.environ.get(OFF_VARIABLE, "").strip() == "0":
        return False
    try:
        path = os.path.abspath(os.fspath(script_path))
        info = os.stat(path)
    except (OSError, TypeError, ValueError):
        return False
    return _script_calls_supervise(path, info.st_size, info.st_mtime_ns)


def shortage_sentence(note):
    """What the note says about the computer's own memory, when it was short; else "".

    Named only for under 1 GiB of commit headroom (Windows) or else under 1 GiB of available RAM at the
    failure, and then as the fact the numbers show and nothing more: they say the computer was short of
    memory, not that the GPU was not. A note without numbers says nothing.
    """
    if not isinstance(note, dict):
        return ""
    system = note.get("system") or note.get("system_before")
    if not isinstance(system, dict):
        return ""
    used, limit = system.get("commit_used"), system.get("commit_limit")
    if isinstance(used, (int, float)) and isinstance(limit, (int, float)) and limit > 0:
        headroom = limit - used
        if headroom < GIB:
            return (f"Windows had {max(0.0, headroom) / GIB:.1f} GB of virtual memory left: "
                    "the computer itself was short of memory.")
    available = system.get("ram_available")
    if isinstance(available, (int, float)) and available < GIB:
        return f"The computer had only {available / GIB:.1f} GB of RAM available."
    return ""


def _say(text):
    """Print one line; a console that cannot show a character gets a replacement, never a crash."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding), flush=True)


def _run_parent():
    """Run the command as a child, once more after a GPU crash; return the exit status to pass on."""
    command = _command()
    folder = tempfile.gettempdir()
    _sweep_stale_notes(folder)
    restarted = False
    wait_info = None
    code = None
    while True:
        note_path = os.path.join(folder, f"{NOTE_PREFIX}{os.getpid()}_{uuid.uuid4().hex[:8]}.json")
        environment = dict(os.environ)
        environment[PARENT_VARIABLE] = str(os.getpid())
        environment[NOTE_VARIABLE] = note_path
        environment.pop(WAIT_VARIABLE, None)
        if wait_info:
            environment[WAIT_VARIABLE] = json.dumps(wait_info)
        try:
            try:
                process = subprocess.Popen(command, env=environment)
            except OSError:
                # No child: the tool runs in this process, as with the switch off (or, after a restart, ends as it did).
                return code
            _forward_stop_signals(process)
            code = _wait_for_child(process)
            note = _read_note(note_path)
        finally:
            _remove(note_path)
        abnormal = is_abnormal_exit(code)
        if restarted:
            if abnormal:
                _say(
                    f"❌ Error: the tool ended abnormally again ({_describe_exit(code)}) after one restart. "
                    "Finished work is kept. Start it again when nothing else is using the GPU, "
                    "or choose the CPU as the device."
                )
            return code
        if note is None or not abnormal:
            return code
        restarted = True
        wait_info = {key: note.get(key) for key in ("device", "free_bytes", "total_bytes")}
        shortage = shortage_sentence(note)
        _say(
            f"[Hardware] Warning: the tool ended abnormally ({_describe_exit(code)}) after a GPU "
            "out-of-memory error. " + (shortage + " " if shortage else "")
            + "Finished work is kept. Starting it once more."
        )


def supervise(module_name, module_file):
    """The hook at the top of a tool script: ``supervise(__name__, __file__)``.

    In a process that is not running the script (an import, a spawn worker) it does
    nothing. In the script's own process it either becomes the supervised child (the
    parent set the marker variables) or becomes the parent and never returns.
    """
    if module_name != "__main__":
        return
    if os.environ.get(PARENT_VARIABLE):
        _become_child()
        return
    if not _supervision_wanted(module_name, module_file):
        return
    code = _run_parent()
    if code is not None:  # None: no child could be started, so this process runs the tool itself
        _pass_on(code)
