# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Console window (desktop.Console_View) tests.

Provides WindowCase (with QApplication setup, isolated app_settings.json,
clipboard cleanup, language switching, and header-row assertion) and helpers
for flushing Qt events, reading text fragments, layout rectangles, and icons.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.Console_View import ConsoleWindow  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from utilities.Console_Capture import Recorder  # noqa: E402


# The capture a WindowCase gives its windows, unless a test names another (None: a program that keeps no output).
RECORDER = object()
ESC = "\x1b"
THREE_BUTTONS = (("metaDataBtn", "Meta Data", "chart-column"), ("agentBtn", "Agent", "bot"),
                 ("fold_view_btn", "Fold View", "dna"))


def flush(app, cycles=6):
    for _ in range(cycles):
        app.processEvents()
        time.sleep(0.01)


def wait_for(app, condition, seconds=3.0):
    """Run the event loop until condition() holds (the window draws what arrives every FLUSH_MS)."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    return condition()


def fragments(window, number):
    """The runs of text of one line, as (text, foreground, background or None, bold, underline, italic)."""
    found = []
    iterator = window.text.document().findBlockByNumber(number).begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        fmt = fragment.charFormat()
        background = fmt.background().color().name() if fmt.background().style() != Qt.BrushStyle.NoBrush else None
        found.append((fragment.text(), fmt.foreground().color().name(), background,
                      fmt.fontWeight() >= 700, fmt.fontUnderline(), fmt.fontItalic()))
        iterator += 1
    return found


def output_lines(window):
    """The window's lines, blank lines included (a line of ``splitlines`` is lost when it is the last and empty)."""
    if window._empty:
        return []
    return window.text.toPlainText().split("\n")


def header_widgets(layout):
    """The widgets of a header layout, left to right, without its blank gaps."""
    return [layout.itemAt(index).widget() for index in range(layout.count()) if layout.itemAt(index).widget() is not None]


def QRect_of(widget, window):
    """A widget's rectangle in the window's coordinates."""
    from PySide6.QtCore import QRect
    return QRect(widget.mapTo(window, widget.rect().topLeft()), widget.size())


def icon_colour(button):
    """The colour a button's icon is drawn in: the commonest of its (nearly) opaque pixels."""
    image = button.icon().pixmap(16, 16).toImage()
    counts = {}
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.alpha() > 200:
                counts[colour.name()] = counts.get(colour.name(), 0) + 1
    return max(counts, key=counts.get) if counts else None


class WindowCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def setUp(self):
        self.recorder = Recorder("viewer", None)
        # The offscreen platform keeps the clipboard's data in a store of the process, freed after Python has gone:
        # data made in Python (Copy's) that is still there ends the process with exit code 139. Empty it after each test.
        self.addCleanup(QGuiApplication.clipboard().clear)
        # The window keeps its choice of showing the times in app_settings.json: not the user's own.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.settings_path = Path(directory.name) / "app_settings.json"
        patch = mock.patch.dict(os.environ, {"SSN_APP_SETTINGS_PATH": str(self.settings_path)})
        patch.start()
        self.addCleanup(patch.stop)

    def window(self, capture=RECORDER, **options):
        window = ConsoleWindow(self.recorder if capture is RECORDER else capture, **options)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.show()
        flush(self.app, 2)
        return window

    def lines(self, window):
        return window.text.toPlainText().splitlines()

    def drain(self, window):
        """Draw what has arrived now, instead of waiting for the window's timer."""
        window._flush()
        self.app.processEvents()

    @contextlib.contextmanager
    def in_language(self, language):
        """Windows built inside are in ``language`` (None: English as written). The bundled CJK font is not
        registered: its registering and removing would make Qt look fonts up again for every later window."""
        if language is None:
            yield
            return
        from desktop import Desktop_App
        with mock.patch.dict(Desktop_App.LANGUAGE_FONTS, clear=True):
            installed = Desktop_App.install_translations(self.app, language)
        try:
            yield
        finally:
            installed.remove()

    def assertOneRow(self, window):
        controls = window.header_controls()
        self.assertEqual(len(controls), 10)  # search, previous, next, filter, times, copy, save, clear, log, folder
        root = window.rect()
        centres, previous_right = [], None
        for control in controls:
            with self.subTest(control=control.accessibleName() or getattr(control, 'text', lambda: '')() or type(control).__name__):
                self.assertTrue(control.isVisibleTo(window))
                box = QRect_of(control, window)
                centres.append(box.center().y())
                self.assertGreaterEqual(box.left(), 12)                   # inside the window's margins
                self.assertLessEqual(box.right(), root.width() - 12)
                self.assertGreater(box.width(), 0)
                if previous_right is not None:
                    self.assertGreater(box.left(), previous_right)        # beside the one before it, never over or under it
                previous_right = box.right()
        self.assertLessEqual(max(centres) - min(centres), 1, centres)       # one row, on one centre line
        header_bottom = max(QRect_of(control, window).bottom() for control in controls)
        self.assertLess(header_bottom, window.text.mapTo(window, window.text.rect().topLeft()).y())
