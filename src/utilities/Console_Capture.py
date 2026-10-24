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
``redact``); what is passed on is left as it was, byte for byte.

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR_VARIABLE = "SSN_LOG_DIR"

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


def prune_logs(log_dir, app, keep=LOG_KEEP):
    """Remove the oldest of an app's logs beyond ``keep``, with their rolled-over and crash files."""
    try:
        names = os.listdir(log_dir)
    except OSError:
        return
    pattern = re.compile(rf"^{re.escape(app)}-\d{{8}}-\d{{6}}-\d+\.log$")
    logs = sorted(name for name in names if pattern.match(name))
    for name in logs[:-keep] if keep > 0 else logs:
        for suffix in ("", ".1", ".crash"):
            try:
                os.remove(os.path.join(log_dir, name + suffix))
            except OSError:
                pass


class Recorder:
    """Keeps lines in a ring and a log file, and tells listeners about each."""

    def __init__(self, app: str, log_dir=None, *, ring_lines=RING_LINES, log_bytes=LOG_BYTES, keep=LOG_KEEP):
        self.app = app
        self.lines: collections.deque = collections.deque(maxlen=ring_lines)
        self.dropped = 0
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
        data = f"{stamp} {'E' if stream == 'stderr' else 'O'} {text}\n".encode("utf-8", "replace")
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


def _forward_writer(descriptor: int):
    """A raw binary writer on ``descriptor`` that does not close it; a Windows console takes UTF-8 through its own writer."""
    if sys.platform == "win32" and os.isatty(descriptor):
        try:
            return io._WindowsConsoleIO(descriptor, "wb", closefd=False)  # type: ignore[attr-defined]
        except Exception:
            pass
    return io.FileIO(descriptor, "wb", closefd=False)


def _write_all(writer, data: bytes):
    while data:
        written = writer.write(data)
        if not written:
            raise OSError("the stream stopped taking output")
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
        os.environ.setdefault("QT_FORCE_STDERR_LOGGING", "1")  # Qt stays silent on a hidden console without it
        if self.log_path is not None:
            # A native crash: Python's own dump, straight to a file, not through a pipe whose reader may be gone.
            self._crash_path = self.log_path + ".crash"
            try:
                self._crash = open(self._crash_path, "ab", buffering=0)
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
        while True:
            chunk = forward_queue.get()
            if chunk is None:
                return
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


def install_qt_message_handler():
    """Route Qt's messages through Python's stderr, tagged "[Qt warning]" and so on, so the Console can tell them apart.

    Call it once PySide6 is loaded, as soon as the entry point can (before
    that, QT_FORCE_STDERR_LOGGING gives the messages without the tag). It
    does nothing in a process that captures no output.
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
            prefix = f"[Qt {names.get(kind, 'message')}] "
            if category and category != "default":
                prefix += f"{category}: "
            print(prefix + str(message), file=sys.stderr)
        except Exception:
            pass

    QtCore.qInstallMessageHandler(handler)
    return handler
