# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Smooth mouse-wheel zoom for the Viewer's camera.

VisPy's PanZoomCamera zooms a whole wheel notch, about a fifth of the view,
in one frame, so wheeling reads as a run of jumps. SmoothWheelZoom takes each
notch's zoom instead and plays it out over the next frames, easing off: each
frame applies the share of what is left that WHEEL_ZOOM_TAU_S gives for the
time since the last one, so a slow frame catches up rather than lagging. A
notch adds to what is left, so fast wheeling zooms faster, and a notch the
other way takes back from it. The zoom keeps the point under the cursor
where it is, as VisPy's does, and goes through the camera's zoom, so the
Viewer's width limits (commands.zoom.limited_zoom_factor) still hold; at a
limit the rest is dropped. A view moved by anything else meanwhile (a pan,
the zoom command, the View Width box, a print) ends the zoom where it is.
"""

from __future__ import annotations

import math
import time

from PySide6 import QtCore

# The time constant of the easing, in seconds: after it, about two thirds of a
# notch's zoom has been applied, and the whole after about five times it.
WHEEL_ZOOM_TAU_S = 0.06
# The frame interval, in ms: about 60 frames a second.
WHEEL_ZOOM_FRAME_MS = 15
# Once a frame would leave less than this, as the log of the zoom factor
# (0.1%), it applies the rest too and the zoom ends.
WHEEL_ZOOM_DONE = 1e-3


def _view(camera):
    rect = camera.rect
    return (rect.pos[0], rect.pos[1], rect.width, rect.height)


class SmoothWheelZoom:
    """Plays out wheel zooms on camera over a few frames.

    zoom(factor, center) zooms the camera, factor being new width / width,
    and returns the factor it applied, which is less than asked at a limit.
    after_step runs after each frame's zoom, for the HUD.
    """

    def __init__(self, camera, zoom, after_step=None, *, clock=time.monotonic, timer=None):
        self._camera = camera
        self._zoom = zoom
        self._after_step = after_step
        self._clock = clock
        self._pending = 0.0  # The log of the zoom factor still to apply.
        self._center = None
        self._view = None    # The view as the last frame left it.
        self._last = 0.0
        if timer is None:
            timer = QtCore.QTimer()
            timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            timer.setInterval(WHEEL_ZOOM_FRAME_MS)
        self._timer = timer
        self._timer.timeout.connect(self.step)

    @property
    def active(self):
        return self._timer.isActive()

    def add(self, factor, center):
        """Zoom by factor (new width / width) about center, a point in scene coordinates, over the next frames."""
        if not (math.isfinite(factor) and factor > 0) or factor == 1:
            return
        if self.active and _view(self._camera) != self._view:
            self._pending = 0.0  # Something else moved the view: start from where it is.
        self._pending += math.log(factor)
        self._center = center
        self._view = _view(self._camera)
        if not self.active:
            self._last = self._clock()
            self._timer.start()

    def step(self):
        """Apply this frame's share of the zoom left."""
        if _view(self._camera) != self._view:
            self.stop()
            return
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        amount = self._pending * (1.0 - math.exp(-elapsed / WHEEL_ZOOM_TAU_S))
        if abs(self._pending - amount) <= WHEEL_ZOOM_DONE:
            amount = self._pending  # The little left goes with this frame.
        if amount:
            factor = math.exp(amount)
            applied = self._zoom(factor, self._center)
            self._pending -= amount
            if applied is not None and not math.isclose(applied, factor, rel_tol=1e-9):
                self._pending = 0.0  # Stopped at a limit.
            self._view = _view(self._camera)
            if self._after_step is not None:
                self._after_step()
        if not self._pending:
            self.stop()

    def stop(self):
        """End the zoom where it is."""
        self._pending = 0.0
        self._timer.stop()


__all__ = ["SmoothWheelZoom", "WHEEL_ZOOM_DONE", "WHEEL_ZOOM_FRAME_MS", "WHEEL_ZOOM_TAU_S"]
