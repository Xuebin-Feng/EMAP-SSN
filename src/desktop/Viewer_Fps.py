# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's frame rate, over the canvas's top right (Config's Show Frame Rate (FPS)).

The display is a QLabel on the canvas, not a VisPy visual: a new text
repaints the label alone, never the canvas, and canvas.render() (print, MCP
captures, the edge cache's pictures) draws the scene only, so it is never in
a picture.

A frame counts when the canvas has drawn its scene into the window (not into
a framebuffer object pushed for canvas.render()), the picture differs from
that of the last frame counted, and the window has then shown it (the
QOpenGLWidget's frameSwapped). The window also composes when a widget over
the canvas repaints, this label's new text included; those compositions drew
nothing and are not counted, so the display never feeds itself. A redraw of
a picture identical to the last counted one is not counted either.

"Identical" is a fingerprint taken once per drawn frame: the number of times
a scene node asked for a redraw since the last frame (every visual's change
goes through that request: the nodes, the edges, the selection rings and the
overlays in the scene), and the Viewer's picture_state (the camera's
rectangle, the canvas size and background, the edge layer's generation,
visibility and motion thinning step). It can miss a change that neither
sees (one made in GL state alone, with no node asked to redraw), and the
display then reads low. It can count a redraw a node asked for that changed
nothing (set_data with the same data), and then reads high. It cannot see
the screen's refresh: of two pictures drawn within one refresh the screen
presents one, and both count.

The rate is the new pictures per second shown in the FPS_WINDOW_S before the
last one, and the slowest is the longest gap between two of them in that
window, which a stutter the average hides shows in. The text changes every
FPS_UPDATE_MS while frames come; once none has come for FPS_IDLE_S (or twice
the last frame interval, for a slow pan) it reads Idle and its timer stops,
so an idle Viewer runs nothing for it.

With the rate go the graphics card's time for a recent frame (GL
timestamps around the scene's draw, desktop.Viewer_Motion.GpuTimer, read a
frame or two later without waiting; left out where the GL lacks them), and,
while the edges are thinned, the share drawn (desktop.Viewer_Motion). In a
window too narrow for those beside the command line, it shows the rate alone
(show_rate_only).
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
# Between the top-right row's items, in px: the display, the Undo and Redo buttons
# (desktop.Viewer_History) and the sidebar's toggle.
FPS_GAP = 8
# The display's height, the toggle's.
FPS_HEIGHT = 30
# The text that sets the room the command line leaves the display (rate, GPU ms, thinning step, slowest ms);
# the rate alone when that is all it shows.
_WIDEST = (888.0, 888.8, 64, 888.0)
_WIDEST_RATE = (888.0,)


def fps_rate_text(rate):
    """The display in a narrow window: the rate alone, as short as the old frames-a-second text, or Idle."""
    if rate is None:
        return QtCore.QCoreApplication.translate("ViewerFps", "Idle")
    return QtCore.QCoreApplication.translate("ViewerFps", "{fps}/s").format(
        fps=f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}")


def fps_text(rate, gpu_ms=None, detail=1, slowest_ms=None):
    """What the display reads: the rate of new pictures, the slowest gap between two, the GPU time and
    the thinning, or Idle (rate None)."""
    translate = QtCore.QCoreApplication.translate
    if rate is None:
        return translate("ViewerFps", "Idle")
    parts = [translate("ViewerFps", "{fps} new/s").format(fps=f"{rate:.0f}" if rate >= 10 else f"{rate:.1f}")]
    if slowest_ms is not None:
        parts.append(translate("ViewerFps", "slowest {ms} ms").format(
            ms=f"{slowest_ms:.0f}" if slowest_ms >= 10 else f"{slowest_ms:.1f}"))
    if gpu_ms is not None:
        parts.append(translate("ViewerFps", "GPU {ms} ms").format(
            ms=f"{gpu_ms:.2f}" if gpu_ms < 1 else f"{gpu_ms:.1f}"))
    if detail and detail > 1:
        parts.append(translate("ViewerFps", "edges 1/{step}").format(step=int(detail)))
    return " · ".join(parts)


class FrameRate:
    """The new pictures the canvas drew into the window and the window showed."""

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._times = deque()
        self._pending = None          # the step of a frame drawn but not yet shown
        self._picture = None          # the fingerprint of the last picture drawn as new
        self.detail = 1               # the step the last frame shown was drawn with
        self.gpu_ms = None            # the GPU time of a recent frame

    def drew(self, detail=1, picture=None):
        """The canvas drew a frame into the window, with every detail-th edge. picture is the fingerprint
        of what it shows: equal to the last new picture's, the frame is a redraw and is not counted.
        None (no fingerprint) always counts."""
        if picture is not None:
            if picture == self._picture:
                return
            self._picture = picture
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

    def slowest_ms(self):
        """The longest gap between two counted frames in the window, in ms (None before two)."""
        if len(self._times) < 2:
            return None
        times = list(self._times)
        return 1000.0 * max(b - a for a, b in zip(times, times[1:]))


class FpsOverlay(QLabel):
    """The frame-rate display: a small card at the canvas's top right, left of the Undo and Redo buttons
    (and of the sidebar's toggle where that shows).

    canvas is the VisPy canvas; edge_detail() gives the step the edges were
    last drawn with (1 for every edge). attach() hooks it to the canvas's
    draws and shown frames; place() puts it by the buttons.
    """

    def __init__(self, canvas, edge_detail=None, *, clock=time.perf_counter, gpu_timer=None,
                 picture_state=None):
        super().__init__(canvas.native)
        self.setObjectName("fpsOverlay")
        self.canvas = canvas
        self.edge_detail = edge_detail or (lambda: 1)
        # What else the picture depends on, as a comparable tuple (camera, size, edge state); with None
        # only the scene nodes' redraw requests tell a new picture.
        self.picture_state = picture_state
        self._redraws = 0             # redraws scene nodes asked for outside a draw
        self.frames = FrameRate(clock)
        self._gpu = gpu_timer
        self.rate_only = False        # a narrow window: the rate, without the GPU time and the thinning
        self._widest = {}
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
        self._count_node_updates()
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
            self.frames.drew(self.edge_detail(), self.picture())
        self.canvas._draw_scene = counted_draw_scene

    def _count_node_updates(self):
        """Count the redraws scene nodes ask for (Node.update calls canvas.update(node=...)): a change to
        any visual in the scene goes through that, whatever the picture_state lists. A request made during
        a draw (a visual preparing itself) is no change."""
        update = getattr(self.canvas, "update", None)
        if update is None:
            return

        def counted_update(*args, **kwargs):
            if (args or kwargs.get("node") is not None) and not getattr(self.canvas, "_drawing", False):
                self._redraws += 1
            return update(*args, **kwargs)
        self.canvas.update = counted_update

    def picture(self):
        """The fingerprint of what the canvas shows now: the node redraw count and the picture_state."""
        state = None
        if self.picture_state is not None:
            try:
                state = self.picture_state()
            except Exception:
                state = object()      # never equal: a picture that cannot be read counts as new
        return (self._redraws, state)

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
        self.show_text(self.text_for(rate, self.frames.gpu_ms, self.frames.detail, self.frames.slowest_ms()))

    def text_for(self, rate, gpu_ms=None, detail=1, slowest_ms=None):
        """What the display reads for this rate (None: Idle): fps_text, the rate alone while rate_only."""
        return fps_rate_text(rate) if self.rate_only else fps_text(rate, gpu_ms, detail, slowest_ms)

    def show_rate_only(self, rate_only):
        """Show the rate alone (True: the window is narrow) or with the GPU time and the thinning."""
        if rate_only != self.rate_only:
            self.rate_only = rate_only
            self.refresh()

    def show_text(self, text):
        if text != self.text():
            self.setText(text)
            self.adjustSize()
            self._keep_right_edge()

    def widest_width(self, rate_only=None):
        """The display's width with its longest text: the room the command line leaves it. rate_only
        asks for the display that shows the rate alone, or the full one; None, as it is now."""
        rate_only = self.rate_only if rate_only is None else rate_only
        widest = self._widest.get(rate_only)
        if widest is None:
            # Measured as the display measures itself: a label like it, with the longest of its texts.
            probe = QLabel()
            probe.setObjectName(self.objectName())
            probe.setFont(self.font())
            probe.setStyleSheet(self.styleSheet())
            widths = []
            for text in (fps_rate_text(*_WIDEST_RATE) if rate_only else fps_text(*_WIDEST), fps_text(None)):
                probe.setText(text)
                probe.ensurePolished()
                widths.append(probe.sizeHint().width())
            widest = self._widest[rate_only] = max(widths)
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
