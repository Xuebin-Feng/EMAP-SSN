# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's zoom: what a zoom is, the limits it keeps to, and the wheel's smooth zoom.

The Viewer's zoom is a percentage of a fixed view (ZOOM_100_PERCENT_WIDTH),
and everything that sets or reads it goes through here: the Zoom box, the
zoom command, print's `zoom N`, the wheel and a pinch. It is the Viewer's
own view maths, which the Zoom box and the commands that zoom share, so
neither depends on the other. set_zoom is the one place a zoom is applied to
the camera; the zoom command (commands/zoom.py) only reads its words, calls
it and reports.

SmoothWheelZoom: VisPy's PanZoomCamera zooms a whole wheel notch, about a
fifth of the view, in one frame, so wheeling reads as a run of jumps.
SmoothWheelZoom takes each notch's zoom instead and plays it out over the
next frames, easing off: each frame applies the share of what is left that
WHEEL_ZOOM_TAU_S gives for the time since the last one, so a slow frame
catches up rather than lagging. A notch adds to what is left, so fast
wheeling zooms faster, and a notch the other way takes back from it. The
zoom keeps the point under the cursor where it is, as VisPy's does, and goes
through the camera's zoom, so the Viewer's width limits
(limited_zoom_factor) still hold; at a limit the rest is dropped. A view
moved by anything else meanwhile (a pan, the zoom command, the Zoom box, a
print) ends the zoom where it is.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PySide6 import QtCore

from utilities.Localization import Message

# The camera stores its scale, canvas width / view width, as float32. Below float32's
# smallest normal value the scale loses precision, and then it is zero, which makes
# the camera's matrix singular; above its largest value the scale is infinite.
_FLOAT32_SMALLEST_NORMAL = float(np.finfo(np.float32).tiny)
_FLOAT32_LARGEST = float(np.finfo(np.float32).max)
# The camera also places the view's edges by their offset from the origin, in float32.
# That offset is good to about |centre| * canvas_width / view_width * eps / 2 pixels, so
# a view narrower than |centre| * canvas_width * eps is off by half a pixel or more.
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)

# The Viewer's zoom is a percentage of a fixed view: 100 % is a view
# ZOOM_100_PERCENT_WIDTH layout (scene) units wide, so a bigger percentage is a
# narrower view, closer in, and the same percentage is the same view width in
# every session and window shape: zoom = 100 * ZOOM_100_PERCENT_WIDTH / width.
# The zoom command, print's zoom N, the Zoom box, the wheel and a pinch all go
# by it. Change this one number to move what 100 % means (the highest zoom,
# MAX_ZOOM_PERCENT, follows). 300 puts the layouts the Viewer opens on, a few
# hundred units across, near 100 %.
ZOOM_100_PERCENT_WIDTH = 300.0
# The narrowest view the Viewer zooms to, whether by the wheel, a pinch, the Zoom
# box or the zoom command: MIN_VIEW_WIDTH units across, which is MAX_ZOOM_PERCENT.
# print's zoom N doesn't take the limits.
MIN_VIEW_WIDTH = 0.5
# The zoom out that shows every node, hidden ones too, as the Viewer opens on them
# (opening_view), is the widest view and the lowest zoom (zoom_limits). It follows the
# nodes as they are moved, and is rounded down to this many significant figures, so
# the whole network still fits at the limit.
ZOOM_LIMIT_DIGITS = 2
# The view the Viewer opens on: every node, a margin of FIT_MARGIN_FRACTION of
# the network's extent on each side (at least FIT_MARGIN_MIN units), and then
# OPENING_RANGE_MARGIN of that range on each side, as the camera's set_range
# added when it opened the view.
FIT_MARGIN_FRACTION = 0.05
FIT_MARGIN_MIN = 10.0
OPENING_RANGE_MARGIN = 0.05
# A hair under a limit is that limit: the width a percentage gives, taken back to
# a percentage, may be a float's last digit off.
LIMIT_SLACK = 1e-9


def zoom_percent(width):
    """The zoom of a view `width` scene units wide, as a percentage."""
    return 100.0 * ZOOM_100_PERCENT_WIDTH / width


def width_at_zoom(percent):
    """The width, in scene units, of the view at a zoom of `percent`."""
    return 100.0 * ZOOM_100_PERCENT_WIDTH / percent


# The highest zoom: the view MIN_VIEW_WIDTH wide.
MAX_ZOOM_PERCENT = zoom_percent(MIN_VIEW_WIDTH)


def format_percent(percent):
    """A zoom as text for a message: six significant figures at most, with no exponent."""
    text = f"{percent:.6g}"
    if "e" in text:
        text = f"{percent:.12f}".rstrip("0").rstrip(".")
    return text


def round_down(value, digits=ZOOM_LIMIT_DIGITS):
    """A positive value rounded down to `digits` significant figures: 23.26 is 23, 1.87 is 1.8."""
    exponent = math.floor(math.log10(value)) - (digits - 1)
    # A hair under a whole count is that count, as 0.57 * 100 is 56.99999999999999.
    count = math.floor((value * 10.0 ** -exponent if exponent < 0 else value / 10.0 ** exponent) + LIMIT_SLACK)
    # The count is a whole number, so dividing by a power of ten gives the float nearest the decimal.
    return count / 10 ** -exponent if exponent < 0 else float(count * 10 ** exponent)


def visible_width(camera):
    """The width, in scene units, the camera shows now.

    The camera widens a view to the canvas's shape (its real rectangle), so what
    shows is at least as wide as the rectangle it was asked for; the zoom goes by
    what shows. A camera with no real rectangle shows what it was asked for.
    """
    real = getattr(camera, "_real_rect", None)
    return abs(float((real if real is not None else camera.rect).width))


def current_zoom(viewer):
    """The zoom the camera shows now, as a percentage, or None while it shows no width."""
    width = visible_width(viewer.view.camera)
    return zoom_percent(width) if math.isfinite(width) and width > 0 else None


def opening_view(positions):
    """The view the Viewer opens on, (left, bottom, width, height): every position with the margins above."""
    points = np.asarray(positions, dtype=float)[:, :2]
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((high - low) * FIT_MARGIN_FRACTION, FIT_MARGIN_MIN)
    low, high = low - margin, high + margin
    extra = (high - low) * OPENING_RANGE_MARGIN
    low, high = low - extra, high + extra
    return (float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))


def zoom_limits(viewer):
    """The lowest and highest zoom, as percentages, the Viewer zooms to.

    The highest is MAX_ZOOM_PERCENT. The lowest is the zoom whose view, at the
    canvas's shape, shows every node, hidden or not, as the Viewer opens on
    them, rounded down to ZOOM_LIMIT_DIGITS significant figures so that view
    still shows every node. It is None while there is no node to show.
    """
    positions = getattr(viewer, "pos", None)
    if positions is None or len(positions) == 0:
        return None, MAX_ZOOM_PERCENT
    _left, _bottom, needed, height = opening_view(positions)
    canvas_width, canvas_height = viewer.canvas.size
    if canvas_width > 0 and canvas_height > 0:
        needed = max(needed, height * canvas_width / canvas_height)
    if not (math.isfinite(needed) and needed > 0):
        return None, MAX_ZOOM_PERCENT
    whole_network = zoom_percent(needed)
    if not whole_network > 0:  # So wide that it underflows.
        return None, MAX_ZOOM_PERCENT
    return round_down(whole_network), MAX_ZOOM_PERCENT


def view_width_limits(viewer):
    """The narrowest and widest view width, in scene units, the Viewer zooms to.

    These are zoom_limits as widths: the widest is None while there is no node to show.
    """
    lowest, _highest = zoom_limits(viewer)
    return MIN_VIEW_WIDTH, None if lowest is None else width_at_zoom(lowest)


def limited_zoom_factor(viewer, factor):
    """The camera's zoom factor (new width / width), cut to keep the view within view_width_limits.

    A zoom that would cross a limit stops at it. One going further past a limit
    the view is already beyond, as after the nodes were moved closer together,
    does nothing; the other way, it zooms as asked.
    """
    scale = float(factor) if np.isscalar(factor) else float(factor[-1])
    width = visible_width(viewer.view.camera)
    if not (math.isfinite(scale) and scale > 0 and width > 0):
        return factor
    narrowest, widest = view_width_limits(viewer)
    if scale < 1:
        return max(scale, min(1.0, narrowest / width))
    if scale > 1 and widest is not None:
        return min(scale, max(1.0, widest / width))
    return scale


def parse_zoom(value):
    """The zoom `value` asks for, as a percentage.

    A text is a number with an optional % sign after it, and a space may
    come before the sign: "200", "200%" and "200 %" are all 200 %. A number is
    taken as it is. Anything else, or a percentage that is not positive and finite,
    raises ValueError holding the Message to show.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            text = text[:-1].rstrip()
        try:
            if not text.isascii() or "_" in text:  # float() reads 1_0 and Arabic-Indic digits.
                raise ValueError(text)
            percent = float(text)
        except ValueError:
            raise ValueError(Message(
                "Error: Zoom must be a number, with an optional % sign, as in {example}.",
                example="zoom 200 or zoom 200%",
            )) from None
    else:
        percent = float(value)
    # float() also accepts nan, inf and 1e400; none of them is a zoom.
    if not math.isfinite(percent) or percent <= 0:
        raise ValueError(Message("Error: Zoom must be a positive, finite percentage."))
    return percent


def view_for_zoom(viewer, value):
    """Return the view `zoom value` sets, as (left, bottom, width, height) in scene units.

    The view is the width the zoom gives, keeps the current centre and takes the
    canvas's aspect ratio. Nothing is changed here. A value that is not a zoom
    the camera can draw raises ValueError holding the Message to show. The
    print command checks its own `zoom N` here too.
    """
    new_width = width_at_zoom(parse_zoom(value))

    canvas_width, canvas_height = viewer.canvas.size
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError(Message("Error: The canvas has no visible area, so the zoom cannot be applied."))

    rect = viewer.view.camera.rect

    center_x = rect.pos[0] + (rect.width / 2.0)
    center_y = rect.pos[1] + (rect.height / 2.0)

    aspect_ratio = canvas_width / canvas_height

    half_w = new_width / 2.0
    half_h = (new_width / aspect_ratio) / 2.0
    x_range = (center_x - half_w, center_x + half_w)
    y_range = (center_y - half_h, center_y + half_h)
    # A tiny zoom can make a width that overflows to inf, a huge finite width can
    # still overflow to inf on a tall canvas, and a scale below float32's normal
    # range is too large.
    if (not all(math.isfinite(value) for value in x_range + y_range)
            or canvas_width < new_width * _FLOAT32_SMALLEST_NORMAL):
        raise ValueError(Message("Error: Zoom is too low for the current view."))
    if (canvas_width > new_width * _FLOAT32_LARGEST
            or new_width < max(abs(center_x), abs(center_y)) * canvas_width * _FLOAT32_EPSILON):
        raise ValueError(Message(
            "Error: Zoom is too high to draw accurately at the current view centre."
        ))
    return (x_range[0], y_range[0], new_width, new_width / aspect_ratio)


def set_zoom(viewer, value):
    """Set the view `zoom value` asks for, and return its zoom as a percentage.

    The zoom command and the Viewer's Zoom box both zoom here. Every check runs
    before the camera is touched: a zoom the camera can't draw, or one outside
    zoom_limits, raises ValueError holding the Message to show, and the view
    stays as it was.
    """
    percent = parse_zoom(value)
    new_rect = view_for_zoom(viewer, percent)
    lowest, highest = zoom_limits(viewer)
    if lowest is not None and not lowest * (1 - LIMIT_SLACK) <= percent <= highest * (1 + LIMIT_SLACK):
        raise ValueError(Message(
            "Error: Zoom must be between {lowest}% and {highest}%, the lowest being the zoom that shows every node.",
            lowest=format_percent(lowest), highest=format_percent(highest),
        ))

    camera = viewer.view.camera
    rect = camera.rect

    # The rectangle is set directly, as the print command does: set_range adds a margin
    # to the requested range, and its margin=0 still adds 0.1 to each side.
    previous = (rect.pos[0], rect.pos[1], rect.width, rect.height)
    try:
        camera.rect = new_rect
    except Exception as error:
        # The camera may already hold the new rectangle, so put the previous one back.
        camera.rect = previous
        raise ValueError(Message("Error: Zoom could not be applied: {error}", error=error)) from error

    viewer._hud_timer.start()
    return percent


# ---- The wheel's smooth zoom ----

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


__all__ = [
    "FIT_MARGIN_FRACTION", "FIT_MARGIN_MIN", "LIMIT_SLACK", "MAX_ZOOM_PERCENT", "MIN_VIEW_WIDTH",
    "OPENING_RANGE_MARGIN", "SmoothWheelZoom", "WHEEL_ZOOM_DONE", "WHEEL_ZOOM_FRAME_MS", "WHEEL_ZOOM_TAU_S",
    "ZOOM_100_PERCENT_WIDTH", "ZOOM_LIMIT_DIGITS", "current_zoom", "format_percent", "limited_zoom_factor",
    "opening_view", "parse_zoom", "round_down", "set_zoom", "view_for_zoom", "view_width_limits",
    "visible_width", "width_at_zoom", "zoom_limits", "zoom_percent",
]
