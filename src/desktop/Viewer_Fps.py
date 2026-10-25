# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's frame rate, over the canvas's top right (Config's Show Frame Rate (FPS)).

The display is a QLabel on the canvas, not a VisPy visual: a new text
repaints the label alone, never the canvas, and canvas.render() (print, MCP
captures, the edge cache's pictures) draws the scene only, so it is never in
a picture.

A frame counts when the canvas has drawn its scene into the window (not into
a framebuffer object pushed for canvas.render()) and the window has then
shown it (the QOpenGLWidget's frameSwapped). The window also composes when a
widget over the canvas repaints, this label's new text included; those
compositions drew nothing and are not counted, so the display never feeds
itself.

The rate is that of the frames shown in the FPS_WINDOW_S before the last
one. The text changes every FPS_UPDATE_MS while frames come; once none has
come for FPS_IDLE_S (or twice the last frame interval, for a slow pan) it
reads Idle and its timer stops, so an idle Viewer runs nothing for it.

With the rate go the graphics card's time for a recent frame (GL
timestamps around the scene's draw, desktop.Viewer_Motion.GpuTimer, read a
frame or two later without waiting; left out where the GL lacks them), and,
while the edges are thinned, the share drawn (desktop.Viewer_Motion).
"""

from __future__ import annotations

import time
from collections import deque

from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel

from desktop.Desktop_App import qt_monospace_font
from desktop.Studio_Theme import TOKENS

# The rate is that of the frames shown in this long, back from the last one.
FPS_WINDOW_S = 1.0
# The text changes this often while frames come.
FPS_UPDATE_MS = 500
# With no frame for this long (or twice the last interval, if longer) the
# display reads Idle.
FPS_IDLE_S = 0.5
# Between the display and the sidebar's toggle, in px.
FPS_GAP = 8
# The display's height, the toggle's.
FPS_HEIGHT = 30
# The text that sets the room the command line leaves the display.
_WIDEST = (888.0, 888.8, 64)


def fps_text(rate, gpu_ms=None, detail=1):
    """What the display reads: the rate, the GPU time and the thinning, or Idle (rate None)."""
    translate = QtCore.QCoreApplication.translate
    if rate is None:
        return translate("ViewerFps", "Idle")
    parts = [translate("ViewerFps", "{fps} fps").format(fps=f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}")]
    if gpu_ms is not None:
        parts.append(translate("ViewerFps", "GPU {ms} ms").format(
            ms=f"{gpu_ms:.2f}" if gpu_ms < 1 else f"{gpu_ms:.1f}"))
    if detail and detail > 1:
        parts.append(translate("ViewerFps", "edges 1/{step}").format(step=int(detail)))
    return " · ".join(parts)


class FrameRate:
    """The frames the canvas drew into the window and the window showed."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._times = deque()
        self._pending = None          # the step of a frame drawn but not yet shown
        self.detail = 1               # the step the last frame shown was drawn with
        self.gpu_ms = None            # the GPU time of a recent frame

    def drew(self, detail=1):
        """The canvas drew a frame into the window, with every detail-th edge."""
        self._pending = int(detail or 1)

    def shown(self):
        """The window showed what it had; True if that was a frame the canvas drew."""
        if self._pending is None:
            return False
        now = self._clock()
        self.detail, self._pending = self._pending, None
        self._times.append(now)
        while len(self._times) > 2 and self._times[-1] - self._times[0] > FPS_WINDOW_S:
            self._times.popleft()
        return True

    def rate(self):
        """Frames per second, or None while idle."""
        if not self._times:
            return None
        idle_after = FPS_IDLE_S
        if len(self._times) >= 2:
            idle_after = max(idle_after, 2.0 * (self._times[-1] - self._times[-2]))
        if self._clock() - self._times[-1] > idle_after:
            self._times.clear()
            return None
        if len(self._times) < 2:
            return None
        return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)


class FpsOverlay(QLabel):
    """The frame-rate display: a small card at the canvas's top right, left of the sidebar's toggle where that shows.

    canvas is the VisPy canvas; edge_detail() gives the step the edges were
    last drawn with (1 for every edge). attach() hooks it to the canvas's
    draws and shown frames; place() puts it by the toggle.
    """

    def __init__(self, canvas, edge_detail=None, *, clock=time.perf_counter, gpu_timer=None):
        super().__init__(canvas.native)
        self.setObjectName("fpsOverlay")
        self.canvas = canvas
        self.edge_detail = edge_detail or (lambda: 1)
        self.frames = FrameRate(clock)
        self._gpu = gpu_timer
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(qt_monospace_font(self.font()))
        self.setFixedHeight(FPS_HEIGHT)
        t = TOKENS
        self.setStyleSheet(f"""
            QLabel#fpsOverlay {{
                background: {t['tint']}; color: {t['text_muted']};
                border: 1px solid {t['tint']}; border-radius: {t['radius']}px;
                padding: 0px 8px;
            }}
        """)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(FPS_UPDATE_MS)
        self.timer.timeout.connect(self.refresh)
        self.setText(fps_text(None))

    # --- hooks -----------------------------------------------------------

    def attach(self):
        """Count the canvas's frames: its scene's draws into the window, as the window shows them."""
        native = self.canvas.native
        swapped = getattr(native, "frameSwapped", None)
        if swapped is not None:
            swapped.connect(self.frame_shown)
        draw_scene = getattr(self.canvas, "_draw_scene", None)
        if draw_scene is None:
            return
        if self._gpu is None:
            from desktop.Viewer_Motion import GpuTimer
            self._gpu = GpuTimer()
        from desktop import Viewer_Edges

        def counted_draw_scene(*args, **kwargs):
            if not Viewer_Edges._draws_into_window(self.canvas):
                return draw_scene(*args, **kwargs)   # canvas.render(): not a frame
            self._gpu.time(lambda: draw_scene(*args, **kwargs), "frame", self._gpu_result)
            self.frames.drew(self.edge_detail())
        self.canvas._draw_scene = counted_draw_scene

    def _gpu_result(self, _tag, gpu_ms):
        self.frames.gpu_ms = gpu_ms

    def frame_shown(self):
        """The window showed a frame: count it if the canvas drew it, and wake the display."""
        if self.frames.shown() and not self.timer.isActive():
            self.timer.start()

    # --- display ---------------------------------------------------------

    def refresh(self):
        """Show the rate now; Idle, with the timer stopped, once frames stop."""
        rate = self.frames.rate()
        if rate is None:
            self.timer.stop()
            self.show_text(fps_text(None))
            return
        self.show_text(fps_text(rate, self.frames.gpu_ms, self.frames.detail))

    def show_text(self, text):
        if text != self.text():
            self.setText(text)
            self.adjustSize()
            self._keep_right_edge()

    def widest_width(self):
        """The display's width with its longest text: the room the command line leaves it."""
        widest = getattr(self, "_widest", None)
        if widest is None:
            # Measured as the display measures itself: a label like it, with that text.
            probe = QLabel(fps_text(*_WIDEST))
            probe.setObjectName(self.objectName())
            probe.setFont(self.font())
            probe.setStyleSheet(self.styleSheet())
            probe.ensurePolished()
            widest = self._widest = probe.sizeHint().width()
            probe.deleteLater()
        return widest

    def place(self, right, centre_y):
        """Put the display's right edge at right, centred on centre_y (canvas px)."""
        self._right = int(round(right))
        self.adjustSize()
        self.setGeometry(self._right - self.width(), int(round(centre_y - FPS_HEIGHT / 2)),
                         self.width(), FPS_HEIGHT)

    def _keep_right_edge(self):
        right = getattr(self, "_right", None)
        if right is not None:
            self.move(right - self.width(), self.y())
