# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""One process's record: its folder under procs/, process.json, events.jsonl, errors.log and windows.json.

This part runs from the first moment of the process (before Qt is imported), so it uses the standard library
only. Whatever goes wrong inside the probe ends up in errors.log through ``fail``, and the probe then turns
itself off in that process (``disabled``): a test tool must never change what the program does.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

from emapssn_desktop_probe import store

# The only environment variables the record keeps: they say which Qt platform and scaling the process ran with.
RECORDED_ENVIRONMENT = (
    "QT_QPA_PLATFORM", "QT_SCALE_FACTOR", "QT_AUTO_SCREEN_SCALE_FACTOR", "QT_FONT_DPI", "QT_QPA_EGLFS_PHYSICAL_WIDTH",
    "DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP", "GDK_SCALE",
)


class Recorder:
    """The folder and the event log of one process."""

    def __init__(self, checkout, kind, gate, argv, *, detail=None):
        self.checkout = Path(checkout)
        self.kind = kind
        self.gate = gate
        self.pid = os.getpid()
        self.out = store.ensure_out(gate["out"])
        self.folder = self._new_folder()
        self.steps = store.StepReader(self.out)
        self.disabled = False
        self.stopping = False  # the process is exiting: the waiting thread must not touch Qt any more
        self._exited = False
        self._lock = threading.Lock()
        base = {"pid": self.pid, "kind": kind}
        self.log = store.EventLog(self.folder / "events.jsonl", base, self.steps)
        self.boot_wall_ms = store.wall_ms()
        self.boot_mono_ms = store.mono_ms()
        self.process = {
            "schema": store.SCHEMA, "pid": self.pid, "ppid": os.getppid(), "kind": kind, "detail": detail,
            "argv": store.redact_argv(argv), "cwd": os.getcwd(), "checkout": str(self.checkout),
            "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
            "boot_wall_ms": self.boot_wall_ms, "boot_mono_ms": self.boot_mono_ms,
            "environment": {name: os.environ[name] for name in RECORDED_ENVIRONMENT if name in os.environ},
            "auto_capture": bool(gate.get("auto_capture", True)), "settle_ms": int(gate.get("settle_ms", store.SETTLE_MS)),
        }
        self._write_process()
        self.event("process_start", argv=self.process["argv"], detail=detail, ppid=self.process["ppid"])

    def _new_folder(self) -> Path:
        """procs/<pid>-<kind>; a folder an earlier process with the same pid left gets a number after it."""
        base = self.out / "procs" / f"{self.pid}-{self.kind}"
        folder, number = base, 1
        while folder.exists():
            number += 1
            folder = base.with_name(f"{base.name}-{number}")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _write_process(self) -> None:
        store.atomic_write_json(self.folder / "process.json", self.process)

    def update_process(self, **fields) -> None:
        with self._lock:
            self.process.update(fields)
            self._write_process()

    def event(self, name: str, **fields) -> dict | None:
        if os.getpid() != self.pid:  # a forked child (multiprocessing) shares this object: it writes nothing
            return None
        try:
            return self.log.write(name, **fields)
        except Exception:
            return None

    def write_windows(self, windows: list) -> None:
        """The window inventory, for `probe.py wait` and `status`; a file that is replaced whole."""
        try:
            store.atomic_write_json(self.folder / "windows.json", {"wall_ms": store.wall_ms(), "windows": windows})
        except Exception:
            pass

    def fail(self, where: str, error: BaseException) -> None:
        """Note a failure of the probe itself, and turn the probe off in this process."""
        text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        try:
            with open(self.folder / "errors.log", "a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%H:%M:%S')}] {where}\n{store.redact_text(text)}\n")
        except OSError:
            pass
        self.event("probe_error", where=where, error=f"{type(error).__name__}: {error}")
        self.disable(f"error in {where}")

    def disable(self, reason: str) -> None:
        if not self.disabled:
            self.disabled = True
            self.event("probe_disabled", reason=reason)

    def finish(self, how: str) -> None:
        """Write the exit event once, however the process ends (normal exit, os._exit)."""
        if os.getpid() != self.pid:
            return
        self.stopping = True
        with self._lock:
            if self._exited:
                return
            self._exited = True
        self.event("exit", how=how)
        self.log.close()
