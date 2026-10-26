# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The command line's text cursor: a thin bar between two characters that blinks as the system's does.

The bar is a QWidget on the canvas, not a VisPy visual. It takes no room in
the text, and its blink repaints the widget alone: the window composes the
canvas's last frame under it, so the network, which is fill-rate bound with
many edges, is not drawn twice a second. canvas.render() (print, MCP
captures, the edge cache's pictures) draws the scene only, so the bar is
never in a picture.

The widget covers the canvas and never moves or changes size on its own, only
when the canvas does: Qt draws the canvas's scene again whenever a widget over
it is moved or resized, but not when part of the widget is repainted. The bar
is painted at its place in the widget, and a move repaints the bar's old and
new rectangles.

The Viewer says where the cursor falls (show_at) each time the command line's
text is laid out, and calls steady after each key: the bar lights at once and
the blink restarts from there, as in a text field. It blinks at the system's
rate (QStyleHints.cursorFlashTime, a whole cycle, lit for the first half); a
rate of 0 or less keeps it lit. While the command line is closed (away), or the
canvas has lost the keyboard (another window or a control took it), nothing is
painted and the timer is stopped, so an idle Viewer runs nothing for it.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter
from PySide6.QtWidgets import QApplication, QStyle, QWidget

from desktop.Studio_Theme import TOKENS

# The bar is at most this wide, in logical px, whatever the style asks for.
MAX_CARET_WIDTH = 4


def caret_width():
    """The bar's width in logical px: the style's text cursor width (1 unless the system sets another)."""
    width = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_TextCursorWidth)
    return min(max(int(width), 1), MAX_CARET_WIDTH)


def blink_interval_ms():
    """How long the bar stays lit, and then dark, in ms; 0 when the system sets no blinking."""
    flash = QGuiApplication.styleHints().cursorFlashTime()
    return flash // 2 if flash >= 2 else 0


def _snap(value):
    """value rounded to a whole number, halves up (Python's round() takes halves to the even number)."""
    return math.floor(value + 0.5)


class CommandCaret(QWidget):
    """The cursor of the command line, a bar over the canvas.

    parent is the canvas's native widget; coordinates are canvas px, as the
    HUD's text is placed in. The bar is snapped to whole device pixels, so it
    is sharp at any display scale.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("commandCaret")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._bar = QRectF()          # empty while the command line is closed
        self._lit = True
        self._active = False
        self._focused = True          # until the canvas loses the keyboard
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._flip)
        self.setGeometry(parent.rect())
        parent.installEventFilter(self)
        self.show()

    @property
    def lit(self):
        """Whether the bar is drawn now (False in the dark half of a blink)."""
        return self._lit

    @property
    def blinking(self):
        """Whether the blink timer runs."""
        return self._timer.isActive()

    @property
    def bar(self):
        """The bar's rectangle in canvas px (empty while the command line is closed)."""
        return QRectF(self._bar)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget():
            kind = event.type()
            if kind == QEvent.Type.Resize:
                self.setGeometry(watched.rect())
            elif kind in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
                self._focused = kind == QEvent.Type.FocusIn
                self.steady()
        return False

    def show_at(self, x, top, height):
        """Put the bar x px from the canvas's left, from top down height px.

        x is the boundary between two characters: the bar is centred on it. A
        bar that was away comes lit, blinking from here; a bar already shown
        keeps its blink, so laying the text out again (a resize) does not
        restart it.
        """
        dpr = self.devicePixelRatioF()
        width = caret_width()
        left = _snap((x - width / 2.0) * dpr)
        upper = _snap(top * dpr)
        bar = QRectF(left / dpr, upper / dpr, max(1, _snap(width * dpr)) / dpr,
                     max(1, _snap((top + height) * dpr) - upper) / dpr)
        if bar != self._bar:
            self._repaint(self._bar, bar)
            self._bar = bar
        if not self._active:
            self._active = True
            self.steady()

    def steady(self):
        """Light the bar now and restart the blink from here, as a text field does after each key."""
        self._lit = True
        self._timer.stop()
        interval = blink_interval_ms()
        if self._active and self._focused and interval:
            self._timer.start(interval)
        self._repaint(self._bar)

    def away(self):
        """The command line is closed: erase the bar and stop its blink."""
        self._active = False
        self._timer.stop()
        self._lit = True
        if not self._bar.isNull():
            self._repaint(self._bar)
            self._bar = QRectF()

    def _flip(self):
        self._lit = not self._lit
        self._repaint(self._bar)

    def _repaint(self, *bars):
        """Repaint the part of the widget the bars cover, with a pixel's margin for the edges."""
        for bar in bars:
            if not bar.isNull():
                self.update(bar.adjusted(-1, -1, 1, 1).toAlignedRect())

    def paintEvent(self, event):
        painter = QPainter(self)
        self.draw(painter)
        painter.end()

    def draw(self, painter):
        """Draw the bar with painter, in the theme's text colour, while it is lit and the canvas has the keys."""
        if self._lit and self._focused and not self._bar.isNull():
            painter.fillRect(self._bar, QColor(TOKENS["text"]))
