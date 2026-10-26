# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The in-process part of the desktop probe: it lives in the GUI thread of one EMAP-SSN process.

boot.py waits for the program's QApplication in a thread and calls ``attach``. From then on one timer in the
GUI thread does everything: every tick it scans the windows (windows.py), every fourth tick it looks for
requests (actions.py) and answers the ones that have settled, and about once a second it checks that the gate
file is still there. Nothing here changes what the program does:

* no window of its own, no focus change, no swallowed event (the event filters return False);
* every entry point is guarded: an error is written to errors.log and the probe turns itself off, restoring
  what it replaced (hooks.py);
* nothing keeps a window alive (only C++ addresses are kept), and no thread outlives the hand-over.
"""

from __future__ import annotations

import atexit
import functools
import re
import sys

import PySide6
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from emapssn_desktop_probe import actions, capture, hooks, store, targets, windows

TICK_MS = 25
REQUEST_EVERY = 4        # ticks between looks at the requests folder (about 100 ms)
GATE_EVERY = 40          # ticks between looks at the gate file (about 1 s)
SECRET_TITLE = re.compile(r"(?i)api[ _-]?key|token|passw|secret|credential")

_probe = None


def attach(recorder, core, application) -> None:
    """Called from the thread that waited for the QApplication: have the GUI thread build the probe.

    QTimer.singleShot with the application as context runs the function in the application's thread, on its
    event loop, as soon as that loop (or a processEvents call) next runs.
    """
    core.QTimer.singleShot(0, application, functools.partial(_build, recorder, application))


def _build(recorder, application) -> None:
    global _probe
    if _probe is not None or recorder.disabled:
        return
    try:
        _probe = Probe(recorder, application)
        _probe.start()
    except Exception as error:
        recorder.fail("starting", error)


def guarded(method):
    """A method of Probe whose failure turns the probe off instead of reaching the program."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.recorder.disabled:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as error:
            self.recorder.fail(method.__name__, error)
            self.shutdown()
            return None

    return wrapper


class Probe:
    def __init__(self, recorder, application):
        self.recorder = recorder
        self.application = application
        self.auto_capture = bool(recorder.process.get("auto_capture", True))
        self.settle_ms = int(recorder.process.get("settle_ms", store.SETTLE_MS))
        self.windows = windows.WindowTracker(self)
        self.actions = actions.Actions(self)
        self.file_dialogs = hooks.FileDialogAnswers(self)
        self.desktop_open = hooks.DesktopOpenRecorder(self)
        self._timer = None
        self._ticks = 0
        self._captures = 0
        self._focus = None
        self._inventory_json = None
        self._stopped = False

    # --- starting and stopping
    def start(self) -> None:
        self._describe_application()
        self.file_dialogs.install()
        self.desktop_open.refresh()
        self._timer = QtCore.QTimer(self.application)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        atexit.register(self.shutdown)  # runs before boot's exit record: this is registered later
        self._tick()

    def _describe_application(self) -> None:
        screens = []
        if isinstance(self.application, QtGui.QGuiApplication):
            for screen in self.application.screens():
                geometry = screen.geometry()
                screens.append({"name": screen.name(), "scale": round(screen.devicePixelRatio(), 4),
                                "geometry": [geometry.x(), geometry.y(), geometry.width(), geometry.height()],
                                "logical_dpi": round(screen.logicalDotsPerInch(), 2),
                                "physical_dpi": round(screen.physicalDotsPerInch(), 2)})
        self.recorder.event(
            "app_ready", app_class=type(self.application).__name__,
            platform=self.application.platformName() if hasattr(self.application, "platformName") else None,
            qt=QtCore.qVersion(), pyside=PySide6.__version__, screens=screens,
            application_name=QtCore.QCoreApplication.applicationName(),
            desktop_file_name=QtGui.QGuiApplication.desktopFileName() if isinstance(self.application, QtGui.QGuiApplication) else None,
            style=self.application.style().objectName() if isinstance(self.application, QtWidgets.QApplication) else None,
        )

    def shutdown(self) -> None:
        """Stop the timer, take every hook out, and leave the program as it was."""
        if self._stopped:
            return
        self._stopped = True
        for step in (self._stop_timer, self.windows.release, self.file_dialogs.restore, self.desktop_open.restore):
            try:
                step()
            except Exception:
                pass

    def _stop_timer(self) -> None:
        if self._timer is not None and shiboken6.isValid(self._timer):
            self._timer.stop()

    # --- the timer
    @guarded
    def _tick(self) -> None:
        if self._stopped:
            return
        self._ticks += 1
        if self._ticks % GATE_EVERY == 0 and store.read_gate(self.recorder.checkout) is None:
            self.recorder.disable("the gate file was removed")
            self.shutdown()
            return
        if self._ticks % GATE_EVERY == 0:
            self.desktop_open.refresh()
        self.windows.scan()
        self._check_focus()
        self._write_inventory()
        if self._ticks % REQUEST_EVERY == 0:
            self.actions.poll()
        self.actions.finish_pending()

    def _check_focus(self) -> None:
        focus = targets.focus_description(self.application)
        marker = None if focus is None else (focus["class"], focus["objectName"], focus["window"], focus["draws_focus"])
        if marker != self._focus:
            self._focus = marker
            self.recorder.event("focus_changed", focus=focus)

    def _write_inventory(self) -> None:
        """Keep windows.json current (`probe.py wait` reads it); rewritten when the windows change, and every few seconds."""
        if not self.windows.inventory_changed and self._ticks % (GATE_EVERY * 5) != 0:
            return
        self.windows.inventory_changed = False
        listing = self.windows.inventory()
        self.recorder.write_windows(listing)

    # --- what the other parts use
    def next_capture_number(self) -> int:
        self._captures += 1
        return self._captures

    def capture(self, widget, *, reason, title=None) -> dict:
        return capture.capture_widget(self, widget, reason=reason, title=title)

    def is_sensitive(self, window) -> bool:
        """A window that probably shows a key or a password: it is not captured unless asked for with force."""
        text = " ".join((targets.window_title(window), window.objectName() or "", type(window).__name__))
        return bool(SECRET_TITLE.search(text))

    def viewer(self):
        """The MainViewer of a Viewer process (the ``viewer`` the script makes in __main__), else None."""
        candidate = getattr(sys.modules.get("__main__"), "viewer", None)
        return candidate if hasattr(candidate, "process_command") else None
