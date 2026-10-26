# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What the desktop probe's parts share: where its files are, how they are written and read, and the redaction.

The probe is test-only. A test agent runs tests/desktop_probe/probe.py, and the .pth it installs loads the
in-process part into EMAP-SSN's own processes (see README.md). Both sides use this module, so it imports the
standard library only and nothing from src/.

The gate lives in <checkout>/temp/desktop_probe/:

    enabled            the gate: its presence turns the probe on, its JSON says where the output goes
    last-out           where the last run wrote, so `sheet` and `timeline` still work after `off`
    boot-errors.log    a failure of the loader itself (before a process has a folder of its own)

and the output folder (default <home>/out, or the --out of `on`) holds:

    step.json, steps.jsonl          the step label the agent last set, and all of them
    requests/<target>--<id>.json    requests an agent writes; target is pid-<n> or kind-<kind>
    responses/<id>.json             the answers
    captures/*.png                  every capture
    files/                          where {files} in a dialog answer points
    procs/<pid>-<kind>/             one folder per process: process.json, events.jsonl, windows.json, errors.log
    contact-sheet.html              what `sheet` writes
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

PACKAGE = "emapssn_desktop_probe"
HOME_NAME = "desktop_probe"
GATE_NAME = "enabled"
LAST_OUT_NAME = "last-out"
OUT_MARKER = ".emapssn-desktop-probe-out"
SETTLE_MS = 300  # how long a new window is left to finish laying itself out before its automatic capture
SCHEMA = 1


def home_dir(checkout) -> Path:
    return Path(checkout) / "temp" / HOME_NAME


def gate_file(checkout) -> Path:
    return home_dir(checkout) / GATE_NAME


def default_out(checkout) -> Path:
    return home_dir(checkout) / "out"


def read_gate(checkout) -> dict | None:
    """The gate's settings when the probe is on for this checkout, else None.

    An empty or unreadable gate file still counts as on, with the defaults: the file's presence is the switch.
    """
    path = gate_file(checkout)
    if not os.path.exists(path):
        return None
    gate: dict = {}
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8") or "{}")
        if isinstance(loaded, dict):
            gate = loaded
    except (OSError, ValueError):
        pass
    gate.setdefault("out", str(default_out(checkout)))
    gate.setdefault("auto_capture", True)
    gate.setdefault("settle_ms", SETTLE_MS)
    return gate


def write_gate(checkout, out, *, auto_capture=True, settle_ms=SETTLE_MS) -> dict:
    gate = {"schema": SCHEMA, "out": str(Path(out).resolve()), "auto_capture": bool(auto_capture),
            "settle_ms": int(settle_ms), "written_ms": wall_ms()}
    home = home_dir(checkout)
    home.mkdir(parents=True, exist_ok=True)
    atomic_write_json(gate_file(checkout), gate, attempts=50)
    (home / LAST_OUT_NAME).write_text(gate["out"], encoding="utf-8")
    return gate


def remove_gate(checkout) -> bool:
    try:
        os.remove(gate_file(checkout))
        return True
    except FileNotFoundError:
        return False


def out_for(checkout, override=None) -> Path:
    """The output folder to read: the one asked for, the gate's, the last one used, else the default."""
    if override:
        return Path(override).resolve()
    gate = read_gate(checkout)
    if gate:
        return Path(gate["out"])
    try:
        remembered = (home_dir(checkout) / LAST_OUT_NAME).read_text(encoding="utf-8").strip()
        if remembered:
            return Path(remembered)
    except OSError:
        pass
    return default_out(checkout)


def ensure_out(out) -> Path:
    out = Path(out)
    for name in ("requests", "responses", "captures", "files", "procs"):
        (out / name).mkdir(parents=True, exist_ok=True)
    marker = out / OUT_MARKER
    if not marker.exists():
        marker.write_text("A folder made by tests/desktop_probe; `probe.py on --fresh` may empty it.\n", encoding="utf-8")
    return out


def wall_ms() -> int:
    return int(time.time() * 1000)


def mono_ms() -> float:
    return round(time.monotonic() * 1000, 1)


def atomic_write_json(path, data, *, attempts: int = 1, pause: float = 0.01) -> None:
    """Write ``data`` so that a reader sees the old file or the whole new one, never half.

    On Windows a file that someone has open for reading cannot be replaced (Python opens without
    FILE_SHARE_DELETE), and the command line reads these files all the time; ``attempts`` tries the
    replacement again after ``pause`` seconds. A writer in the GUI thread keeps the default of one try, because
    the next write will do, and a writer in a thread of its own asks for more.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(redact_data(data), indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt >= attempts:
                with contextlib.suppress(OSError):
                    temporary.unlink()
                raise
            time.sleep(pause)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_events(path):
    """The events of one events.jsonl, in order; a half-written last line is skipped."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def slug(text, limit=40) -> str:
    """A file-name-safe piece of ``text``: letters, digits and dashes."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "")).strip("-")
    return (cleaned[:limit].strip("-")) or "none"


# ---------------------------------------------------------------------------------------------
# The step label
# ---------------------------------------------------------------------------------------------

def read_step(out) -> dict:
    """The step the agent last set (``probe.py step "..."``): {"seq", "label", "wall_ms"}; seq 0 before the first."""
    step = read_json(Path(out) / "step.json")
    if isinstance(step, dict) and "label" in step:
        return step
    return {"seq": 0, "label": "", "wall_ms": 0}


def write_step(out, label) -> dict:
    out = ensure_out(out)
    step = {"seq": read_step(out)["seq"] + 1, "label": str(label), "wall_ms": wall_ms()}
    atomic_write_json(out / "step.json", step, attempts=50)
    with open(out / "steps.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(redact_data(step), ensure_ascii=False) + "\n")
    return step


def read_steps(out) -> list[dict]:
    return read_events(Path(out) / "steps.jsonl")


class StepReader:
    """The current step, re-read only when step.json changes (a stat per call)."""

    def __init__(self, out):
        self._path = Path(out) / "step.json"
        self._stamp = None
        self._step = {"seq": 0, "label": "", "wall_ms": 0}

    def current(self) -> dict:
        try:
            info = os.stat(self._path)
        except OSError:
            return self._step
        stamp = (info.st_mtime_ns, info.st_size)
        if stamp != self._stamp:
            self._stamp = stamp
            self._step = read_json(self._path, self._step) or self._step
        return self._step


# ---------------------------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------------------------
# Everything the probe writes (events, answers, the contact sheet's captions) goes through redact_data. It is a
# pattern filter, a best effort: it knows the common key shapes and the words that name a secret, not every
# secret. The probe never reads the files that hold keys (model_card.json, Biohub_API.json, ankh_license.json),
# so a key can only reach it through a window's text or a command line.

MASK = "<redacted>"
_KEY_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{8,}"),
)
# A secret's name followed by "=" (or by a closing quote, a colon and an optional opening quote, as in JSON, or by
# a colon and a quote), then its value. A label such as "Token: paste it here" is text, not an assignment.
_NAMED_VALUE = re.compile(
    r"""(?ix)
    (?P<name>[\w.\-]*(?:api[_\-]?key|apikey|token|secret|passw(?:or)?d|credential|authorization)[\w.\-]*)
    (?P<sep>["']\s*[:=]\s*["']?|\s*=\s*["']?|\s*:\s*["'])
    (?P<value>[^\s"',;}\]]{4,})
    """
)
_LONG_TOKEN = re.compile(r"(?<![\w/\\.\-])[A-Za-z0-9+_\-]{32,}={0,2}(?![\w/\\.\-])")
_SECRET_FLAG = re.compile(r"(?i)^--?[\w\-]*(?:api[_\-]?key|apikey|token|secret|passw(?:or)?d|credential|key)$")


def _looks_like_token(text: str) -> bool:
    """A long run of letters and digits, which a key is and an amino-acid sequence or a word is not."""
    has_digit = any(character.isdigit() for character in text)
    has_letter = any(character.isalpha() for character in text)
    return has_digit and has_letter


def redact_text(text: str) -> str:
    """``text`` with key-shaped values and the values of secret-named assignments masked."""
    if not text:
        return text
    for shape in _KEY_SHAPES:
        text = shape.sub(MASK, text)
    text = _NAMED_VALUE.sub(lambda match: match["name"] + match["sep"] + MASK, text)
    return _LONG_TOKEN.sub(lambda match: MASK if _looks_like_token(match.group(0)) else match.group(0), text)


def redact_data(value, _depth=0):
    """``value`` with every string in it redacted (dicts, lists and tuples are copied; other values pass through)."""
    if _depth > 40:
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: redact_data(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(item, _depth + 1) for item in value]
    return value


def redact_argv(argv) -> list[str]:
    """A command line with the value after a --api-key, --token, --password ... flag masked, and the rest redacted."""
    result = []
    mask_next = False
    for argument in argv:
        argument = str(argument)
        if mask_next:
            result.append(MASK)
            mask_next = False
            continue
        if _SECRET_FLAG.match(argument):
            mask_next = True
            result.append(argument)
            continue
        result.append(redact_text(argument))
    return result


# ---------------------------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------------------------

class EventLog:
    """One process's events.jsonl: a line per event, flushed at once, safe to write from two threads."""

    def __init__(self, path, base: dict, steps: StepReader):
        self._path = Path(path)
        self._base = dict(base)
        self._steps = steps
        self._lock = threading.Lock()
        self._handle = None
        self.count = 0

    def write(self, name: str, **fields) -> dict:
        record = {"ev": name, "wall_ms": wall_ms(), "mono_ms": mono_ms(), **self._base,
                  "step": self._steps.current()["label"], **fields}
        line = json.dumps(redact_data(record), ensure_ascii=False, default=str)
        with self._lock:
            if self._handle is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = open(self._path, "a", encoding="utf-8", buffering=1)
            self._handle.write(line + "\n")
            self.count += 1
        return record

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None


# ---------------------------------------------------------------------------------------------
# Processes and requests
# ---------------------------------------------------------------------------------------------

def _windows_api():
    """kernel32 and ntdll with the argument and result types the calls here need (handles are pointers, not ints)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    return ctypes, wintypes, kernel32, ntdll


def own_start_time() -> float | None:
    """When this process began, in seconds since the epoch, read from the operating system without psutil.

    psutil takes tens of milliseconds to import and the probe must not slow a start-up. Windows: GetProcessTimes.
    Linux: the start tick in /proc/self/stat against /proc/uptime. Elsewhere: None (the probe's boot time is used).
    """
    try:
        if os.name == "nt":
            ctypes, wintypes, kernel32, _ntdll = _windows_api()
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel32.GetProcessTimes(kernel32.GetCurrentProcess(), *(ctypes.byref(item) for item in (created, exited, kernel, user))):
                return None
            ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
            return (ticks - 116444736000000000) / 10_000_000
        if sys.platform.startswith("linux"):
            with open("/proc/self/stat", encoding="utf-8") as handle:
                fields = handle.read().rsplit(")", 1)[1].split()
            started = int(fields[19]) / os.sysconf("SC_CLK_TCK")
            with open("/proc/uptime", encoding="utf-8") as handle:
                uptime = float(handle.read().split()[0])
            return time.time() - (uptime - started)
    except Exception:
        return None
    return None


def _parent_of(pid: int) -> int | None:
    """The parent of process ``pid`` (Windows: NtQueryInformationProcess; Linux: /proc), or None."""
    try:
        if os.name == "nt":
            ctypes, _wintypes, kernel32, ntdll = _windows_api()

            class BasicInformation(ctypes.Structure):
                _fields_ = [(name, ctypes.c_void_p) for name in ("status", "peb", "affinity", "priority", "pid", "parent")]

            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                information = BasicInformation()
                if ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(information), ctypes.sizeof(information), None) != 0:
                    return None
                return int(information.parent or 0) or None
            finally:
                kernel32.CloseHandle(handle)
        if sys.platform.startswith("linux"):
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                return int(handle.read().rsplit(")", 1)[1].split()[1]) or None
        if pid == os.getpid():
            return os.getppid()
    except Exception:
        return None
    return None


def lineage(depth: int = 6) -> list[int]:
    """The parent of this process, its parent and so on (nearest first). On Windows the first is often the venv's launcher stub."""
    chain = []
    pid = os.getpid()
    for _ in range(depth):
        pid = _parent_of(pid) or (os.getppid() if not chain and pid == os.getpid() else None)
        if not pid or pid in chain:
            break
        chain.append(pid)
    return chain


def process_folders(out) -> list[Path]:
    procs = Path(out) / "procs"
    try:
        return sorted((entry for entry in procs.iterdir() if entry.is_dir()), key=lambda entry: entry.name)
    except OSError:
        return []


def process_alive(record: dict) -> bool:
    """Whether the process a process.json describes is still running (a reused pid is told apart by its start time)."""
    pid = record.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        created = record.get("created")
        if isinstance(created, (int, float)) and abs(psutil.Process(pid).create_time() - created) > 2:
            return False
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def list_processes(out, *, live_only=False) -> list[dict]:
    """The process.json of every process that wrote to ``out``, oldest first, each with "alive" and "folder"."""
    records = []
    for folder in process_folders(out):
        record = read_json(folder / "process.json")
        if not isinstance(record, dict):
            continue
        record["folder"] = str(folder)
        record["alive"] = process_alive(record)
        if record["alive"] or not live_only:
            records.append(record)
    records.sort(key=lambda record: record.get("boot_wall_ms", 0))
    return records


def request_name(target: dict, request_id: str) -> str:
    if "pid" in target:
        return f"pid-{int(target['pid'])}--{request_id}.json"
    return f"kind-{slug(target['kind'], 20)}--{request_id}.json"


def write_request(out, target: dict, request: dict, request_id: str) -> Path:
    """Drop a request where the target process will find it. The file appears whole (written, then renamed)."""
    out = ensure_out(out)
    path = out / "requests" / request_name(target, request_id)
    body = dict(request)
    body["id"] = request_id
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def response_path(out, request_id: str) -> Path:
    return Path(out) / "responses" / f"{request_id}.json"
