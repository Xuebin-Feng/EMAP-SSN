# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""One process's record: its folder under procs/, process.json, events.jsonl, errors.log and windows.json.

This part runs from the first moment of the process (before Qt is imported), so it uses the standard library
only. It is made, and its files are made, in the thread boot.py starts, not in the program's main thread: the
program's start-up pays for deciding whether to record, and nothing else. An event written before the files
exist waits in memory and is written first. Whatever goes wrong inside the probe ends up in errors.log through
``fail``, and the probe then turns itself off in that process (``disabled``): a test tool must never change what
the program does.
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

    def __init__(self, checkout, kind, gate, argv, *, detail=None, boot_wall_ms=None, boot_mono_ms=None):
        """Only names and numbers: ``prepare`` makes the folder and the files."""
        self.checkout = Path(checkout)
        self.kind = kind
        self.gate = gate
        self.pid = os.getpid()
        self.out = Path(gate["out"])
        self.folder: Path | None = None
        self.steps: store.StepReader | None = None
        self.log: store.EventLog | None = None
        self.process: dict = {}
        self.disabled = False
        self.stopping = False  # the process is exiting: the waiting thread must not touch Qt any more
        self._exited = False
        self._lock = threading.Lock()
        self._queued: list | None = []
        self._argv = list(argv)
        self._detail = detail
        self.boot_wall_ms = boot_wall_ms if boot_wall_ms is not None else store.wall_ms()
        self.boot_mono_ms = boot_mono_ms if boot_mono_ms is not None else store.mono_ms()

    def prepare(self) -> None:
        """Make procs/<pid>-<kind>/ and its files, then write what happened meanwhile."""
        store.ensure_out(self.out)
        self.folder = self._new_folder()
        self.steps = store.StepReader(self.out)
        self.process = {
            "schema": store.SCHEMA, "pid": self.pid, "ppid": os.getppid(), "kind": self.kind, "detail": self._detail,
            "argv": store.redact_argv(self._argv), "cwd": os.getcwd(), "checkout": str(self.checkout),
            "python": sys.version.split()[0], "executable": sys.executable, "platform": sys.platform,
            "boot_wall_ms": self.boot_wall_ms, "boot_mono_ms": self.boot_mono_ms,
            "environment": {name: os.environ[name] for name in RECORDED_ENVIRONMENT if name in os.environ},
            "auto_capture": bool(self.gate.get("auto_capture", True)),
            "settle_ms": int(self.gate.get("settle_ms", store.SETTLE_MS)),
        }
        self._write_process(attempts=20)
        log = store.EventLog(self.folder / "events.jsonl", {"pid": self.pid, "kind": self.kind}, self.steps)
        log.write("process_start", at=(self.boot_wall_ms, self.boot_mono_ms), argv=self.process["argv"],
                  detail=self._detail, ppid=self.process["ppid"])
        with self._lock:
            queued, self._queued = self._queued or [], None
            for name, stamp, fields in queued:
                log.write(name, at=stamp, **fields)
            self.log = log

    def _new_folder(self) -> Path:
        """procs/<pid>-<kind>; a folder an earlier process with the same pid left gets a number after it."""
        base = self.out / "procs" / f"{self.pid}-{self.kind}"
        folder, number = base, 1
        while folder.exists():
            number += 1
            folder = base.with_name(f"{base.name}-{number}")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _write_process(self, attempts) -> None:
        store.atomic_write_json(self.folder / "process.json", self.process, attempts=attempts)

    def update_process(self, **fields) -> None:
        """Add to process.json (the waiting thread does this; the file may be open in the command line meanwhile)."""
        with self._lock:
            self.process.update(fields)
            try:
                self._write_process(attempts=200)
            except OSError:
                pass  # it is information for the timeline; the process goes on without it

    def add_details(self) -> None:
        """What is slow to find out (the platform's full name), written once the program is going."""
        self.update_process(platform=platform.platform())

    def event(self, name: str, **fields) -> dict | None:
        if os.getpid() != self.pid:  # a forked child (multiprocessing) shares this object: it writes nothing
            return None
        stamp = (store.wall_ms(), store.mono_ms())
        with self._lock:
            if self._queued is not None:
                self._queued.append((name, stamp, fields))
                return None
        try:
            return self.log.write(name, at=stamp, **fields)
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
        target = self.folder / "errors.log" if self.folder is not None else store.home_dir(self.checkout) / "boot-errors.log"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%H:%M:%S')}] pid {self.pid} {where}\n{store.redact_text(text)}\n")
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
            if self._exited or self.log is None:
                return
            self._exited = True
        self.event("exit", how=how)
        self.log.close()
