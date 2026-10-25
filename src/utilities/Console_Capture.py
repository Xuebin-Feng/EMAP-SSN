# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Keep everything a desktop window's process prints, for the Console window.

The three windows (Config, Tools, Viewer) no longer have a terminal of their
own on Windows: the launcher starts them with a hidden console, and anything
they print would vanish. ``install`` points file descriptors 1 and 2 at two
pipes, so output from Python, Qt, Numba, CUDA libraries, child processes that
inherit the handles and ``faulthandler`` all arrives here, where one reader
thread per stream

* keeps the last RING_LINES lines in memory, for the Console window;
* writes every line to a log file, ``temp/logs/<app>-<time>-<pid>.log``, rolled
  over at LOG_BYTES, of which LOG_KEEP per app are kept;
* passes the bytes on to the stream the process started with, but only when
  someone reads it: an MCP client's pipe or log file (Viewer_Terminal.py), a
  redirect, a visible terminal. A hidden console or NUL is not read, so
  nothing is passed on to it.

The log and the ring hold the text with API keys and tokens replaced (see
``redact``); what is passed on is left as it was, byte for byte, except for
colour: commands colour some output with ANSI escape sequences (see
utilities/Terminal_Colors.py), which the ring keeps for the Console window to
draw, and which the log and a stream that is not a terminal never get.

``start`` is what an entry point calls, first in its ``__main__`` block and
before anything prints. It does nothing but ``configure_output_streams`` for a
headless run or ``--console``. Only the entry points of windows call it, and
``install`` refuses in the MCP server, whose standard output is the protocol.

Imports only the standard library, so an entry point can call it before Qt,
NumPy or torch load (torch must load before Qt on Windows).
"""

from __future__ import annotations

import atexit
import codecs
import collections
import faulthandler
import io
import itertools
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time

from utilities.Terminal_Colors import AnsiStripper, strip_ansi

RING_LINES = 20000
LOG_BYTES = 8 * 1024 * 1024
LOG_KEEP = 10
# A "line" with no end after this many characters is cut there, so output with no
# newline (a binary dump) cannot grow without bound.
MAX_LINE_CHARACTERS = 1 << 20
# A progress line (one that is rewritten with a carriage return) tells its readers
# at most this often; the ring always holds its latest state.
PARTIAL_NOTIFY_SECONDS = 0.05
# The chunks of output waiting to be passed on to the original stream. A reader
# that stalls (a console in selection mode) loses the newest chunks instead of
# stalling the process.
FORWARD_QUEUE_CHUNKS = 512
# A non-blocking original stream that is full ("try again") is waited for, with pauses that grow from
# the first to the longest, until it has taken nothing for this long: then its reader counts as gone.
EAGAIN_PATIENCE_SECONDS = 10.0
EAGAIN_FIRST_PAUSE_SECONDS = 0.001
EAGAIN_LONGEST_PAUSE_SECONDS = 0.01

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR_VARIABLE = "SSN_LOG_DIR"

# Windows: asking whether a process still runs (process_is_gone).
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_INVALID_PARAMETER = 87
STILL_ACTIVE = 259
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

_capture: "Capture | None" = None
# Scripts whose standard output is a protocol, not text for a person: capture refuses to start in them.
_PROTOCOL_SCRIPTS = frozenset({"EMAPSSN_MCP_Server.py"})


# =====================================================================
# Secrets
# =====================================================================

REDACTED = "<redacted>"

# Whole tokens of the formats the program's services use: OpenAI, DeepSeek and
# Anthropic (sk-...), Hugging Face, Google, GitHub, Slack and AWS keys.
_TOKEN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
# "Bearer <token>", `api_key = "..."`, `"authorization": "..."`, `?key=...`:
# the name stays, the value goes.
_NAMED_PATTERNS = (
    re.compile(r"(?i)(\bBearer\s+)([A-Za-z0-9._~+/=\-]{16,})"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|secret|password|passwd"
        r"|x-api-key|authorization)[\"']?\s*[:=]\s*[\"']?(?:Bearer\s+)?)([^\s\"',;&}\)]{6,})"
    ),
    re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token|signature)=)([^&\s\"']+)"),
)
# Most lines hold none of these; one search over them spares the patterns above.
_SECRET_HINT = re.compile(
    r"sk-|hf_|AIza|gh[pousr]_|xox[abprs]-|AKIA|(?i:bearer|key|token|secret|passw|authorization|signature)"
)


def redact(text: str) -> str:
    """text with API keys and tokens replaced by <redacted>."""
    if not _SECRET_HINT.search(text):
        return text
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)
    for pattern in _NAMED_PATTERNS:
        text = pattern.sub(lambda match: match.group(1) + REDACTED, text)
    return text


# =====================================================================
# Lines
# =====================================================================

class Line:
    """One line of output: its number, time, stream ("stdout" or "stderr") and text.

    ``partial`` marks a progress line that is still being rewritten with
    carriage returns; the next state of it, and in the end the finished line,
    carries the same ``seq``, and replaces it where it is shown.
    """

    __slots__ = ("seq", "time", "stream", "text", "partial")

    def __init__(self, seq, when, stream, text, partial=False):
        self.seq, self.time, self.stream, self.text, self.partial = seq, when, stream, text, partial

    def __repr__(self):
        return f"Line({self.seq}, {self.stream}, {self.text!r}{', partial' if self.partial else ''})"


def collapse_carriage_returns(text: str) -> str:
    """What a terminal would show of a line written with carriage returns: its last non-empty state."""
    if "\r" not in text:
        return text
    for state in reversed(text.split("\r")):
        if state:
            return state
    return ""


class Splitter:
    """Bytes in, lines out: decodes UTF-8 across chunk boundaries and splits at newlines.

    ``feed`` gives (text, partial) pairs. A line that ends in a newline is
    complete (carriage returns inside it collapse to the last state, and a
    CRLF ends a line as a newline does). The text after the last newline
    gives a partial pair only when it holds a carriage return, as a progress
    bar's states do; ``finish`` gives what is left at the end of the stream.
    """

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""

    def feed(self, data: bytes):
        self._pending += self._decoder.decode(data)
        *complete, self._pending = self._pending.split("\n")
        found = [(collapse_carriage_returns(line), False) for line in complete]
        if len(self._pending) > MAX_LINE_CHARACTERS:
            found.append((collapse_carriage_returns(self._pending), False))
            self._pending = ""
        elif "\r" in self._pending:
            state = collapse_carriage_returns(self._pending)
            if state:
                found.append((state, True))
        return found

    def finish(self):
        text = self._pending + self._decoder.decode(b"", final=True)
        self._pending = ""
        return [(collapse_carriage_returns(text), False)] if text else []


# =====================================================================
# The record: ring, log file, listeners
# =====================================================================

def log_directory(environment=None) -> Path:
    """Where the logs go: temp/logs in the project folder, or the folder SSN_LOG_DIR names (the tests do).."""
    environment = os.environ if environment is None else environment
    named = environment.get(LOG_DIR_VARIABLE, "").strip()
    return Path(named) if named else PROJECT_ROOT / "temp" / "logs"


def _log_stem(app: str, when: float, pid: int) -> str:
    return f"{app}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(when))}-{pid}"


def process_is_gone(pid: int) -> bool:
    """Whether no process has this id any more. False when that cannot be told: keeping a file is the safe answer.

    (``os.kill(pid, 0)`` is no probe on Windows, where any signal but Ctrl+C and Ctrl+Break ends the process.)
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if not handle:
            return ctypes.get_last_error() == ERROR_INVALID_PARAMETER  # no such process (not: access denied)
        try:
            code = ctypes.c_uint32()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value != STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # PermissionError: it exists, and belongs to someone else
    return False


def prune_logs(log_dir, app, keep=LOG_KEEP):
    """Remove the oldest of an app's logs beyond ``keep``, with their rolled-over and crash files.

    An empty crash file of a log that stays goes too when its process is gone: a run that was killed
    (a log-out, a task kill) never reached the close that removes it, and nothing crashed in it.
    """
    try:
        names = os.listdir(log_dir)
    except OSError:
        return
    pattern = re.compile(rf"^{re.escape(app)}-\d{{8}}-\d{{6}}-(\d+)\.log$")
    logs = sorted(name for name in names if pattern.match(name))
    older = logs[:-keep] if keep > 0 else logs
    for name in older:
        for suffix in ("", ".1", ".crash"):
            try:
                os.remove(os.path.join(log_dir, name + suffix))
            except OSError:
                pass
    for name in logs[len(older):]:
        crash = os.path.join(log_dir, name + ".crash")
        try:
            if os.path.getsize(crash) != 0:
                continue  # a dump: that is what the file is for
        except OSError:
            continue  # none
        if process_is_gone(int(pattern.match(name).group(1))):
            try:
                os.remove(crash)
            except OSError:
                pass


class Recorder:
    """Keeps lines in a ring and a log file, and tells listeners about each."""

    def __init__(self, app: str, log_dir=None, *, ring_lines=RING_LINES, log_bytes=LOG_BYTES, keep=LOG_KEEP):
        self.app = app
        self.lines: collections.deque = collections.deque(maxlen=ring_lines)
        self.dropped = 0
        self.text_streams: tuple = ()  # sys.stdout and sys.stderr, once a Capture has made them
        self.total = 0
        self._seq = itertools.count(1)
        self._lock = threading.Lock()
        self._listeners = []
        self._open_partial: dict = {}
        self._last_notice: dict = {}
        self.log_dir = None if log_dir is None else str(log_dir)
        self.log_path: str | None = None
        self._log_bytes = log_bytes
        self._log = None
        self._log_size = 0
        self._log_lock = threading.Lock()
        self.log_error: str | None = None
        if self.log_dir is not None:
            self._open_log(keep)

    # --- log file
    def _open_log(self, keep):
        try:
            os.makedirs(self.log_dir, mode=0o700, exist_ok=True)
            path = os.path.join(self.log_dir, _log_stem(self.app, time.time(), os.getpid()) + ".log")
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._log = os.fdopen(descriptor, "ab", buffering=0)
            self.log_path = path
            prune_logs(self.log_dir, self.app, keep)
        except OSError as error:
            self.log_error = str(error)  # the Console still works from the ring

    def _write_log(self, when, stream, text):
        if self._log is None:
            return
        stamp = time.strftime("%H:%M:%S", time.localtime(when))
        data = f"{stamp} {'E' if stream == 'stderr' else 'O'} {strip_ansi(text)}\n".encode("utf-8", "replace")
        try:
            if self._log_size + len(data) > self._log_bytes and self._log_size:
                self._log.close()
                os.replace(self.log_path, self.log_path + ".1")
                descriptor = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                self._log = os.fdopen(descriptor, "ab", buffering=0)
                self._log_size = 0
            self._log.write(data)
            self._log_size += len(data)
        except OSError as error:
            self.log_error = str(error)
            self._log = None

    # --- recording
    def add(self, stream: str, text: str, partial: bool = False) -> Line:
        text = redact(text)
        now = time.time()
        notify = True
        with self._lock:
            seq = self._open_partial.get(stream)
            replacing = seq is not None
            if seq is None:
                seq = next(self._seq)
                self.total += 1
            if partial:
                self._open_partial[stream] = seq
                if now - self._last_notice.get(stream, 0.0) < PARTIAL_NOTIFY_SECONDS:
                    notify = False
                else:
                    self._last_notice[stream] = now
            else:
                self._open_partial.pop(stream, None)
                self._last_notice.pop(stream, None)
            line = Line(seq, now, stream, text, partial)
            self._store(line, replacing)
            listeners = list(self._listeners) if notify else ()
        if not partial:
            self._write_log(now, stream, text)
        for listener in listeners:
            try:
                listener(line)
            except Exception:
                pass  # a listener that fails must not stop the reader
        return line

    def _store(self, line, replacing):
        # The next state of a progress line replaces it, and it is nearly always among the last.
        if replacing:
            for position in range(len(self.lines) - 1, max(-1, len(self.lines) - 6), -1):
                if self.lines[position].seq == line.seq:
                    self.lines[position] = line
                    return
        if len(self.lines) == self.lines.maxlen:
            self.dropped += 1
        self.lines.append(line)

    # --- what the Console window reads
    def snapshot(self):
        with self._lock:
            return list(self.lines)

    def subscribe(self, callback):
        with self._lock:
            self._listeners.append(callback)

    def unsubscribe(self, callback):
        with self._lock:
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

    def close_log(self):
        log, self._log = self._log, None
        if log is not None:
            try:
                log.close()
            except OSError:
                pass


# =====================================================================
# Who reads the original streams
# =====================================================================

def _console_window_state():
    """Windows: "none" (no console window), "hidden" (CREATE_NO_WINDOW) or "shown" (a terminal)."""
    import ctypes

    kernel32, user32 = ctypes.windll.kernel32, ctypes.windll.user32
    kernel32.GetConsoleWindow.restype = ctypes.c_void_p
    user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user32.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    window = kernel32.GetConsoleWindow()
    if not window:
        return "none"
    name = ctypes.create_unicode_buffer(64)
    user32.GetClassNameW(window, name, 64)
    # Windows Terminal and other ConPTY hosts give the console a window that is
    # never visible, in front of a terminal that is.
    if name.value == "PseudoConsoleWindow":
        return "shown"
    return "shown" if user32.IsWindowVisible(window) else "hidden"


def _is_null_device(descriptor: int) -> bool:
    try:
        stat = os.fstat(descriptor)
        null = os.stat(os.devnull)
    except OSError:
        return False
    device = getattr(stat, "st_rdev", 0)  # Windows' NUL has none: a hidden console says the same
    return device != 0 and device == getattr(null, "st_rdev", -1)


def original_is_read(descriptor: int) -> bool:
    """Whether bytes written to the stream the process started with will be read by someone."""
    try:
        os.fstat(descriptor)
    except OSError:
        return False  # pythonw: no stream at all
    if _is_null_device(descriptor):
        return False
    if os.isatty(descriptor):
        # Windows also reports NUL as a terminal; the console it belongs to
        # tells whether anyone looks.
        return sys.platform != "win32" or _console_window_state() == "shown"
    return True  # a pipe or file the launcher (MCP, the monitor, a test, a redirect) set up


def _takes_colours(descriptor: int) -> bool:
    """Whether the stream behind ``descriptor`` shows colour sequences: a terminal does, a file or a pipe does not."""
    try:
        if not os.isatty(descriptor):
            return False
    except OSError:
        return False
    return sys.platform != "win32" or _windows_console_takes_ansi(descriptor)


def _windows_console_takes_ansi(descriptor: int) -> bool:
    """Windows: switch the console behind ``descriptor`` to ANSI processing, if it has none yet; whether it has it."""
    import ctypes
    import msvcrt

    try:
        kernel32 = ctypes.windll.kernel32
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(descriptor))
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if mode.value & ENABLE_VIRTUAL_TERMINAL_PROCESSING:
            return True
        return bool(kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
    except (OSError, ValueError, AttributeError):
        return False


def _forward_writer(descriptor: int):
    """A raw binary writer on ``descriptor`` that does not close it; a Windows console takes UTF-8 through its own writer."""
    if sys.platform == "win32" and os.isatty(descriptor):
        try:
            return io._WindowsConsoleIO(descriptor, "wb", closefd=False)  # type: ignore[attr-defined]
        except Exception:
            pass
    return io.FileIO(descriptor, "wb", closefd=False)


def _write_all(writer, data: bytes, *, patience=None, pause=time.sleep):
    """Write every byte of ``data``; ``patience`` seconds without any progress is an error.

    A stream the process inherited in non-blocking mode (a pipe whose reader is slow, with
    O_NONBLOCK set by whoever made it) takes what fits and then says "try again": ``FileIO.write``
    gives None, or raises BlockingIOError. That is a full pipe, not a gone reader, so the writer
    waits a moment and tries again; only a reader that takes nothing for ``patience`` seconds
    (default EAGAIN_PATIENCE_SECONDS) ends the forwarding, as a failed write does.
    """
    patience = EAGAIN_PATIENCE_SECONDS if patience is None else patience
    blocked_since = None
    delay = EAGAIN_FIRST_PAUSE_SECONDS
    while data:
        try:
            written = writer.write(data)
        except BlockingIOError as error:
            # (characters_written is only set when some bytes went before it filled up: progress)
            written = getattr(error, "characters_written", 0) or None
        if written is None:
            now = time.monotonic()
            if blocked_since is None:
                blocked_since = now
            elif now - blocked_since >= patience:
                raise OSError("the stream is full and nobody reads it")
            pause(delay)
            delay = min(delay * 2, EAGAIN_LONGEST_PAUSE_SECONDS)
            continue
        if not written:
            raise OSError("the stream stopped taking output")
        blocked_since, delay = None, EAGAIN_FIRST_PAUSE_SECONDS
        data = data[written:]


def _set_windows_std_handle(descriptor: int):
    """Windows: native code that asks the OS for its standard handles must see the pipes too."""
    import ctypes
    import msvcrt

    kernel32 = ctypes.windll.kernel32
    kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    handle = msvcrt.get_osfhandle(descriptor) if descriptor >= 0 else -1
    kernel32.SetStdHandle({0: -10, 1: -11, 2: -12}[descriptor], ctypes.c_void_p(handle))


# =====================================================================
# The capture
# =====================================================================

class Capture(Recorder):
    """The file-descriptor plumbing: two pipes over descriptors 1 and 2, a reader and a forwarder for each."""

    def __init__(self, app: str, log_dir=None, forward: bool | None = None, **options):
        super().__init__(app, log_dir, **options)
        self._closed = False
        # What each descriptor was: one copy for the forwarder to write to and one to put back at the
        # end. The C runtime locks a descriptor while a write to it is blocked, so putting the original
        # back from the forwarder's own copy would wait for a reader that never reads.
        self._saved: dict = {}          # descriptor -> a copy to put back, or None
        self._forwarded: dict = {}      # descriptor -> the copy the forwarder writes to, or None
        self._python_streams = (sys.stdout, sys.stderr)
        self.forwarding: dict = {}
        self._queues: dict = {}
        self._threads: list = []
        self.forward_dropped = 0
        self._crash = None
        self._crash_path = None
        names = (("stdout", 1), ("stderr", 2))
        for name, descriptor in names:
            self.forwarding[name] = original_is_read(descriptor) if forward is None else bool(forward)
            try:
                self._saved[descriptor] = os.dup(descriptor)
                self._forwarded[descriptor] = os.dup(descriptor) if self.forwarding[name] else None
            except OSError:
                self._saved[descriptor] = self._forwarded[descriptor] = None
                self.forwarding[name] = False
        for stream in self._python_streams:
            try:
                stream.flush()
            except Exception:
                pass
        for name, descriptor in names:
            read_end, write_end = os.pipe()
            os.dup2(write_end, descriptor)  # inheritable, as descriptors 0 to 2 always are
            os.close(write_end)
            if sys.platform == "win32":
                _set_windows_std_handle(descriptor)
            forward_queue = None
            if self.forwarding[name]:
                forward_queue = self._queues[name] = queue.Queue(FORWARD_QUEUE_CHUNKS)
                self._start(self._forward, f"console-forward-{name}", forward_queue, self._forwarded[descriptor], name)
            self._start(self._pump, f"console-{name}", name, read_end, forward_queue)
        # Python-level streams over the new descriptors: UTF-8, what cannot be encoded
        # escaped, every write passed on at once (as -u does).
        sys.stdout = self._text_stream(1)
        sys.stderr = self._text_stream(2)
        self.text_streams = (sys.stdout, sys.stderr)
        os.environ.setdefault("QT_FORCE_STDERR_LOGGING", "1")  # Qt stays silent on a hidden console without it
        if self.log_path is not None:
            # A native crash: Python's own dump, straight to a file, not through a pipe whose reader may be gone.
            self._crash_path = self.log_path + ".crash"
            try:
                # Owner only, as the log is: a dump holds paths, and a copied log folder keeps the mode.
                descriptor = os.open(
                    self._crash_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0), 0o600
                )
                self._crash = os.fdopen(descriptor, "ab", buffering=0)
                faulthandler.enable(file=self._crash, all_threads=True)
            except OSError:
                self._crash = None
        atexit.register(self.close)

    @staticmethod
    def _text_stream(descriptor):
        raw = io.FileIO(descriptor, "wb", closefd=False)
        return io.TextIOWrapper(raw, encoding="utf-8", errors="backslashreplace", write_through=True)

    def _start(self, target, name, *arguments):
        thread = threading.Thread(target=target, args=arguments, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    # --- reader
    def _pump(self, name, read_end, forward_queue):
        splitter = Splitter()
        try:
            while True:
                try:
                    chunk = os.read(read_end, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                if forward_queue is not None and self.forwarding[name]:
                    try:
                        forward_queue.put_nowait(chunk)
                    except queue.Full:
                        self.forward_dropped += 1
                for text, partial in splitter.feed(chunk):
                    self.add(name, text, partial)
            for text, partial in splitter.finish():
                self.add(name, text, partial)
        finally:
            if forward_queue is not None:
                try:
                    forward_queue.put(None, timeout=1.0)  # the forwarder ends; it may already have
                except queue.Full:
                    pass
            try:
                os.close(read_end)
            except OSError:
                pass

    # --- forwarder
    def _forward(self, forward_queue, descriptor, name):
        try:
            writer = _forward_writer(descriptor)
        except OSError:
            self.forwarding[name] = False
            return
        # A terminal gets the output as it came, colours and all; anything else gets it with no colours.
        stripper = None if _takes_colours(descriptor) else AnsiStripper()
        while True:
            chunk = forward_queue.get()
            if chunk is None:
                return
            if stripper is not None:
                chunk = stripper.feed(chunk)
                if not chunk:
                    continue
            try:
                _write_all(writer, chunk)
            except OSError:
                self.forwarding[name] = False  # the reader is gone
                return

    # --- shutdown: put the process back as it was, so the last lines are read before it ends
    def close(self):
        if self._closed:
            return
        self._closed = True
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        for descriptor, saved in self._saved.items():
            try:
                if saved is not None:
                    os.dup2(saved, descriptor)
                else:
                    os.close(descriptor)
            except OSError:
                pass
            if sys.platform == "win32" and saved is not None:
                try:
                    _set_windows_std_handle(descriptor)
                except Exception:
                    pass
        sys.stdout, sys.stderr = self._python_streams
        deadline = time.monotonic() + 2.0
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        for saved in self._saved.values():
            if saved is not None:
                try:
                    os.close(saved)
                except OSError:
                    pass
        # A forwarder still blocked in a write holds its descriptor: that copy is left to the process's end.
        if not any(thread.is_alive() for thread in self._threads if thread.name.startswith("console-forward")):
            for forwarded in self._forwarded.values():
                if forwarded is not None:
                    try:
                        os.close(forwarded)
                    except OSError:
                        pass
        if self._crash is not None:
            try:
                faulthandler.disable()
                self._crash.close()
                if os.path.getsize(self._crash_path) == 0:
                    os.remove(self._crash_path)  # nothing crashed: no file to find
            except OSError:
                pass
            self._crash = None
        self.close_log()


# =====================================================================
# Entry points
# =====================================================================

def install(app: str, log_dir=None, forward: bool | None = None) -> "Capture":
    """Capture this process's output, once. ``forward`` forces passing it on (True) or not (False).

    The log goes to ``log_dir``, by default log_directory().

    Refuses in the MCP server, whose standard output carries the protocol.
    """
    global _capture
    if os.path.basename(sys.argv[0] if sys.argv else "") in _PROTOCOL_SCRIPTS:
        raise RuntimeError("the output of the MCP server is its protocol, so it is never captured")
    if _capture is None:
        _capture = Capture(app, log_directory() if log_dir is None else log_dir, forward)
    return _capture


def current() -> "Capture | None":
    """The capture this process installed, or None."""
    return _capture


def is_capture_stream(stream) -> bool:
    """Whether ``stream`` is the sys.stdout or sys.stderr this process's capture installed.

    A pass-through wrapper that keeps the stream it wraps as ``.stream`` counts as it does: the Viewer's
    command portal wraps both, to copy a command's output to the request that asked for it.
    """
    if _capture is None:
        return False
    for _ in range(4):
        if any(stream is own for own in _capture.text_streams):
            return True
        stream = getattr(stream, "stream", None)
        if stream is None:
            return False
    return False


def hard_exit(code: int) -> None:
    """End the process at once, with ``os._exit``, after what it printed has been passed on.

    With a capture, descriptors 1 and 2 are pipes that daemon threads empty into the log and the
    original streams. ``os._exit`` ends those threads wherever they are, so a line printed just
    before it is lost: the MCP client never sees the Viewer's "hidden window unavailable" line, and
    the headless fallback it triggers never starts. ``close`` puts the descriptors back, which ends
    the pipes, and waits (two seconds at most) until the threads have passed everything on. Without
    a capture the streams are just flushed. Nothing here can keep the process from ending.
    """
    try:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        if _capture is not None:
            _capture.close()
        for stream in (sys.stdout, sys.stderr):  # close() gave back the process's own streams
            try:
                stream.flush()
            except Exception:
                pass
    finally:
        os._exit(code)


def wanted(argv=None, environment=None) -> bool:
    """Whether an entry point should capture: not for a headless run, ``--console`` or SSN_NO_CONSOLE_CAPTURE."""
    argv = sys.argv if argv is None else argv
    environment = os.environ if environment is None else environment
    if "--headless" in argv or "--console" in argv:
        return False
    return environment.get("SSN_NO_CONSOLE_CAPTURE", "").strip().lower() not in {"1", "true", "yes", "on"}


def start(app: str, argv=None, *, headless: bool = False, log_dir=None):
    """What a window's entry point calls first in ``__main__``, before anything prints.

    Captures the output (see ``install``) and returns the Capture; for a
    headless run, ``--console`` or anything that stops the capture, it only
    gives the streams UTF-8 and returns None.
    """
    from utilities.Output_Streams import configure_output_streams

    if headless or not wanted(argv):
        configure_output_streams()
        return None
    try:
        return install(app, log_dir)
    except Exception as error:  # never keep a window from starting
        configure_output_streams()
        print(f"Warning: the Console could not capture this process's output: {error}", file=sys.stderr)
        return None


def is_portal_already_registered(category, message) -> bool:
    """Whether a Qt message is the desktop portal's "this connection already has an application ID".

    Qt registers the application with the desktop's portal once ``setDesktopFileName`` has named it
    (qt.qpa.services: "Failed to register with host portal ... Could not register app ID: Connection
    already associated with an application ID"). On GNOME the session has often registered the
    process already, through the systemd scope its launcher puts it in, so the portal refuses the
    second registration. The name is in place either way (the taskbar groups and names the window as
    it should), so the message only alarms, on every start. Any other failure of the registration, or
    of the services, is not matched and still shows.
    """
    return (
        category == "qt.qpa.services"
        and "host portal" in message
        and "already associated with an application id" in message.lower()
    )


def install_qt_message_handler():
    """Route Qt's messages through Python's stderr, tagged "[Qt warning]" and so on, so the Console can tell them apart.

    Call it once PySide6 is loaded, as soon as the entry point can (before
    that, QT_FORCE_STDERR_LOGGING gives the messages without the tag). It
    does nothing in a process that captures no output. One harmless message
    is left out (is_portal_already_registered).
    """
    if _capture is None:
        return None
    from PySide6 import QtCore

    names = {
        QtCore.QtMsgType.QtDebugMsg: "debug",
        QtCore.QtMsgType.QtInfoMsg: "info",
        QtCore.QtMsgType.QtWarningMsg: "warning",
        QtCore.QtMsgType.QtCriticalMsg: "critical",
        QtCore.QtMsgType.QtFatalMsg: "fatal",
    }

    def handler(kind, context, message):
        try:
            category = getattr(context, "category", None)
            if is_portal_already_registered(category, str(message)):
                return
            prefix = f"[Qt {names.get(kind, 'message')}] "
            if category and category != "default":
                prefix += f"{category}: "
            print(prefix + str(message), file=sys.stderr)
        except Exception:
            pass

    QtCore.qInstallMessageHandler(handler)
    return handler
