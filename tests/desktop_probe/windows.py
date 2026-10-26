# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Keeping track of the program's top-level windows: when each is shown, painted, changed, hidden and gone.

Two things work together. A scan, every few milliseconds, lists QApplication.topLevelWidgets() and compares
each window with what it was; that finds new windows and every change without touching one event. A window
the scan has found also gets an event filter on itself and on its QWindow, which stamps show, hide, close and
expose at the moment they happen. Neither keeps a window alive (only C++ addresses are kept), and the filter
never swallows an event.

A window's first show is seen by the next scan, so its "shown" time is late by up to one scan interval, which
the event says (``via: "scan"``). The "first painted" time is the first event-loop turn after the platform
exposed the window: it is exact on the platforms that expose windows, and absent for a window that is never
mapped (the Viewer's hidden mode, the tests' WA_DontShowOnScreen).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PySide6 import QtCore, QtGui, QtWidgets

from emapssn_desktop_probe import targets

QEvent = QtCore.QEvent
WATCHED_WIDGET_EVENTS = (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.Close)
MAX_AUTO_CAPTURES_PER_WINDOW = 20
MIN_SECONDS_BETWEEN_AUTO_CAPTURES = 1.0


@dataclass
class WindowRecord:
    key: int
    serial: int
    snapshot: dict | None = None
    shown: bool = False
    shows: int = 0
    exposed: bool = False
    painted: bool = False
    close_requested: bool = False
    handle_key: int | None = None
    captures: int = 0
    last_capture: float = 0.0
    state: dict = field(default_factory=dict)


class _Watcher(QtCore.QObject):
    """The event filter: it reports to the tracker and always lets the event go on."""

    def __init__(self, tracker):
        super().__init__()
        self._tracker = tracker

    def eventFilter(self, watched, event):  # noqa: N802 (Qt's name)
        try:
            self._tracker.on_event(watched, event)
        except Exception as error:
            self._tracker.probe.recorder.fail("window event filter", error)
        return False


class WindowTracker:
    def __init__(self, probe):
        self.probe = probe
        self.recorder = probe.recorder
        self.records: dict[int, WindowRecord] = {}
        self.serials: dict[int, int] = {}
        self._by_handle: dict[int, int] = {}
        self._next_serial = 1
        self._watcher = _Watcher(self)
        self.inventory_changed = True

    # --- the scan
    def scan(self) -> None:
        app = QtWidgets.QApplication.instance()
        present = set()
        for widget in app.topLevelWidgets():
            if not targets.alive(widget) or not widget.isWindow():
                continue
            key = targets.cpp_key(widget)
            present.add(key)
            record = self.records.get(key)
            visible = widget.isVisible()
            if record is None:
                if not visible:
                    continue
                record = self._adopt(widget, key)
            self._observe(record, widget, visible, via="scan")
        for key in [key for key in self.records if key not in present]:
            self._gone(self.records.pop(key))

    def _adopt(self, widget, key) -> WindowRecord:
        record = WindowRecord(key=key, serial=self._next_serial)
        self._next_serial += 1
        self.records[key] = record
        self.serials[key] = record.serial
        widget.installEventFilter(self._watcher)
        self.inventory_changed = True
        return record

    def _watch_handle(self, record, widget) -> None:
        """Put the filter on the window's QWindow once it has one (it is made when the window is first shown)."""
        if record.handle_key is not None:
            return
        handle = widget.windowHandle()
        if handle is not None and targets.alive(handle):
            record.handle_key = targets.cpp_key(handle)
            self._by_handle[record.handle_key] = record.key
            handle.installEventFilter(self._watcher)

    def _observe(self, record, widget, visible, *, via) -> None:
        self._watch_handle(record, widget)
        snapshot = targets.describe_window(widget)
        if visible and not record.shown:
            self._shown(record, widget, snapshot, via)
        elif not visible and record.shown:
            self._hidden(record, widget, snapshot)
        elif visible and record.snapshot is not None and snapshot != record.snapshot:
            changes = {name: [record.snapshot.get(name), value] for name, value in snapshot.items()
                       if record.snapshot.get(name) != value}
            self.recorder.event("window_changed", window=snapshot, changed=changes, serial=record.serial)
            self.inventory_changed = True
        record.snapshot = snapshot
        if visible and record.shown and not record.exposed:
            handle = widget.windowHandle()
            if handle is not None and handle.isExposed():
                self._exposed(record, widget)

    # --- transitions
    def _shown(self, record, widget, snapshot, via) -> None:
        record.shown = True
        record.shows += 1
        record.exposed = record.painted = record.close_requested = False
        record.snapshot = snapshot
        self.inventory_changed = True
        self.recorder.event(
            "window_shown", window=snapshot, serial=record.serial, via=via, late=via == "scan",
            first=record.shows == 1, focus=targets.focus_description(), popup=targets.is_popup_like(widget),
        )
        self._describe_dialog(widget, record)
        self._schedule_capture(record, widget)

    def _hidden(self, record, widget, snapshot) -> None:
        record.shown = False
        record.snapshot = snapshot
        self.inventory_changed = True
        self.recorder.event("window_hidden", window=snapshot, serial=record.serial)
        if record.close_requested:
            self.recorder.event("window_closed", window=snapshot, serial=record.serial)
            record.close_requested = False

    def _exposed(self, record, widget) -> None:
        record.exposed = True
        key = record.key
        QtCore.QTimer.singleShot(0, lambda: self._painted(key, record.shows))

    def _painted(self, key, shows) -> None:
        record = self.records.get(key)
        if record is None or record.painted or not record.shown or record.shows != shows:
            return
        record.painted = True
        self.recorder.event("window_first_painted", serial=record.serial, window=record.snapshot)

    def _gone(self, record) -> None:
        if record.shown:
            self.recorder.event("window_hidden", window=record.snapshot, serial=record.serial, destroyed=True)
        self.recorder.event("window_destroyed", window=record.snapshot, serial=record.serial)
        self.serials.pop(record.key, None)
        if record.handle_key is not None:
            self._by_handle.pop(record.handle_key, None)
        self.inventory_changed = True

    # --- the event filter
    def on_event(self, watched, event) -> None:
        kind = event.type()
        if kind == QEvent.Type.Expose:
            key = self._by_handle.get(targets.cpp_key(watched))
            record = self.records.get(key)
            if record is not None and record.shown and not record.exposed:
                widget = self._widget_for(key)
                if widget is not None:
                    self._exposed(record, widget)
            return
        if kind not in WATCHED_WIDGET_EVENTS or not isinstance(watched, QtWidgets.QWidget) or not watched.isWindow():
            return
        key = targets.cpp_key(watched)
        record = self.records.get(key)
        if record is None:
            return
        if kind == QEvent.Type.Show and not record.shown:
            self._shown(record, watched, targets.describe_window(watched), "event")
        elif kind == QEvent.Type.Hide and record.shown and not watched.isVisible():
            self._hidden(record, watched, targets.describe_window(watched))
        elif kind == QEvent.Type.Close:
            record.close_requested = True
            self.recorder.event("window_close_requested", serial=record.serial, window=targets.describe_window(watched),
                                spontaneous=event.spontaneous())

    def _widget_for(self, key):
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if targets.alive(widget) and targets.cpp_key(widget) == key:
                return widget
        return None

    # --- dialogs
    def _describe_dialog(self, widget, record) -> None:
        if not isinstance(widget, QtWidgets.QDialog):
            return
        info = {"class": type(widget).__name__, "title": targets.window_title(widget), "modal": widget.isModal()}
        if isinstance(widget, QtWidgets.QMessageBox):
            default = widget.defaultButton()
            info.update(
                text=widget.text(), informative_text=widget.informativeText(), detailed_text=widget.detailedText(),
                icon=widget.icon().name, buttons=[button.text() for button in widget.buttons()],
                default_button=default.text() if default is not None else None,
            )
        else:
            info["buttons"] = [button.text() for button in widget.findChildren(QtWidgets.QAbstractButton) if button.isVisible()]
            info["labels"] = [label.text() for label in widget.findChildren(QtWidgets.QLabel) if label.isVisible()][:12]
        self.recorder.event("dialog_shown", serial=record.serial, dialog=info)

    # --- the automatic capture
    def _schedule_capture(self, record, widget) -> None:
        probe = self.probe
        if not probe.auto_capture or targets.is_popup_like(widget):
            return
        if probe.is_sensitive(widget):
            self.recorder.event("capture_skipped", reason="the window looks like it holds a key or password",
                                serial=record.serial)
            return
        key, shows = record.key, record.shows
        QtCore.QTimer.singleShot(probe.settle_ms, lambda: self._auto_capture(key, shows))

    def _auto_capture(self, key, shows) -> None:
        try:
            record = self.records.get(key)
            if record is None or not record.shown or record.shows != shows:
                return
            now = time.monotonic()
            if (record.captures >= MAX_AUTO_CAPTURES_PER_WINDOW
                    or now - record.last_capture < MIN_SECONDS_BETWEEN_AUTO_CAPTURES):
                return
            widget = self._widget_for(key)
            if widget is None or not widget.isVisible():
                return
            record.captures += 1
            record.last_capture = now
            self.probe.capture(widget, reason="auto")
        except Exception as error:  # a picture that failed is noted; the probe goes on
            self.recorder.event("capture_failed", error=f"{type(error).__name__}: {error}")

    # --- what other parts ask for
    def inventory(self) -> list[dict]:
        """The visible top-level windows, oldest first, each described with its index."""
        windows = [widget for widget in targets.top_level_windows() if targets.cpp_key(widget) in self.records]
        windows.sort(key=lambda widget: self.serials.get(targets.cpp_key(widget), 0))
        result = []
        for index, widget in enumerate(windows):
            entry = targets.describe_window(widget)
            entry["index"] = index
            entry["serial"] = self.serials.get(targets.cpp_key(widget))
            result.append(entry)
        return result

    def release(self) -> None:
        """Take the filters off every window (the probe is turning itself off)."""
        for widget in QtWidgets.QApplication.topLevelWidgets():
            try:
                if targets.alive(widget):
                    widget.removeEventFilter(self._watcher)
                    handle = widget.windowHandle()
                    if handle is not None and targets.alive(handle):
                        handle.removeEventFilter(self._watcher)
            except Exception:
                pass
        self.records.clear()
