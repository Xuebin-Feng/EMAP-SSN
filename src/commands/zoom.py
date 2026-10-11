# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np

import Command_Engine
from utilities.Localization import JoinedMessage, Message

# The camera stores its scale, canvas width / view width, as float32. Below float32's
# smallest normal value the scale loses precision, and then it is zero, which makes
# the camera's matrix singular; above its largest value the scale is infinite.
_FLOAT32_SMALLEST_NORMAL = float(np.finfo(np.float32).tiny)
_FLOAT32_LARGEST = float(np.finfo(np.float32).max)
# The camera also places the view's edges by their offset from the origin, in float32.
# That offset is good to about |centre| * canvas_width / view_width * eps / 2 pixels, so
# a view narrower than |centre| * canvas_width * eps is off by half a pixel or more.
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)

# The view widths the Viewer keeps to, whether zoomed by the wheel, a pinch or
# this command (view_width_limits): no narrower than MIN_VIEW_WIDTH, and no
# wider than the nearest multiple of MAX_VIEW_WIDTH_STEP that shows every node,
# hidden ones too, as the Viewer opens on them (opening_view). That widest view
# follows the nodes as they are moved. print's zoom N doesn't take them.
MIN_VIEW_WIDTH = 0.5
MAX_VIEW_WIDTH_STEP = 100
# The view the Viewer opens on: every node, a margin of FIT_MARGIN_FRACTION of
# the network's extent on each side (at least FIT_MARGIN_MIN units), and then
# OPENING_RANGE_MARGIN of that range on each side, as the camera's set_range
# added when it opened the view.
FIT_MARGIN_FRACTION = 0.05
FIT_MARGIN_MIN = 10.0
OPENING_RANGE_MARGIN = 0.05


def opening_view(positions):
    """The view the Viewer opens on, (left, bottom, width, height): every position with the margins above."""
    points = np.asarray(positions, dtype=float)[:, :2]
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((high - low) * FIT_MARGIN_FRACTION, FIT_MARGIN_MIN)
    low, high = low - margin, high + margin
    extra = (high - low) * OPENING_RANGE_MARGIN
    low, high = low - extra, high + extra
    return (float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1]))


def view_width_limits(viewer):
    """The narrowest and widest view width, in scene units, the Viewer zooms to.

    The widest is the nearest multiple of MAX_VIEW_WIDTH_STEP at or above the
    width whose view, at the canvas's shape, shows every node, hidden or not,
    as the Viewer opens on them; it is None while there is no node to show.
    """
    positions = getattr(viewer, "pos", None)
    if positions is None or len(positions) == 0:
        return MIN_VIEW_WIDTH, None
    _left, _bottom, needed, height = opening_view(positions)
    canvas_width, canvas_height = viewer.canvas.size
    if canvas_width > 0 and canvas_height > 0:
        needed = max(needed, height * canvas_width / canvas_height)
    if not math.isfinite(needed):
        return MIN_VIEW_WIDTH, None
    # A hair over a multiple, as floating point leaves 1100 * 1.21 / 1.21, is that multiple.
    steps = max(1, math.ceil(needed / MAX_VIEW_WIDTH_STEP - 1e-9))
    return MIN_VIEW_WIDTH, float(steps * MAX_VIEW_WIDTH_STEP)


def limited_zoom_factor(viewer, factor):
    """The camera's zoom factor (new width / width), cut to keep the view within view_width_limits.

    A zoom that would cross a limit stops at it. One going further past a limit
    the view is already beyond, as after the nodes were moved closer together,
    does nothing; the other way, it zooms as asked.
    """
    scale = float(factor) if np.isscalar(factor) else float(factor[-1])
    width = abs(viewer.view.camera.rect.width)
    if not (math.isfinite(scale) and scale > 0 and width > 0):
        return factor
    narrowest, widest = view_width_limits(viewer)
    if scale < 1:
        return max(scale, min(1.0, narrowest / width))
    if scale > 1 and widest is not None:
        return min(scale, max(1.0, widest / width))
    return scale


def _report_error(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg)


def view_for_width(viewer, text):
    """Return the view `zoom text` sets, as (left, bottom, width, height) in scene units.

    The view is text scene units wide, keeps the current centre and takes the
    canvas's aspect ratio. Nothing is changed here. A text that is not a width
    the camera can draw raises ValueError holding the Message to show. The
    print command checks the width of its own `zoom N` here too.
    """
    try:
        new_width = float(text)
    except ValueError:
        raise ValueError(Message("Error: Zoom width must be a valid number.")) from None
    # float() also accepts nan, inf and 1e400; none of them is a view width.
    if not math.isfinite(new_width) or new_width <= 0:
        raise ValueError(Message("Error: Zoom width must be a positive, finite number."))

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
    # A huge finite width can still overflow to inf on a tall canvas, and a
    # scale below float32's normal range is too large.
    if (not all(math.isfinite(value) for value in x_range + y_range)
            or canvas_width < new_width * _FLOAT32_SMALLEST_NORMAL):
        raise ValueError(Message("Error: Zoom width is too large for the current view."))
    if (canvas_width > new_width * _FLOAT32_LARGEST
            or new_width < max(abs(center_x), abs(center_y)) * canvas_width * _FLOAT32_EPSILON):
        raise ValueError(Message(
            "Error: Zoom width is too small to draw accurately at the current view centre."
        ))
    return (x_range[0], y_range[0], new_width, new_width / aspect_ratio)


def run(viewer, args):
    if not args or args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="zoom <width>"),
            "Description: Sets the camera view width to exactly <width> while keeping the current "
            "center point and canvas aspect ratio. The width must be at least 0.5 and at most the "
            "widest view that shows every node, hidden ones too, rounded up to a multiple of 100; "
            "the mouse wheel stops at the same limits.\n"
            "Examples:\n  zoom 500  (Sets the view width to 500 units)",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # Every check runs before the camera is touched.
    try:
        new_rect = view_for_width(viewer, args[0])
    except ValueError as error:
        _report_error(viewer, error.args[0])
        return
    new_width = new_rect[2]
    narrowest, widest = view_width_limits(viewer)
    if widest is not None and not narrowest <= new_width <= widest:
        _report_error(viewer, Message(
            "Error: Zoom width must be between {narrowest} and {widest}, the widest view that shows every node.",
            narrowest=f"{narrowest:g}", widest=f"{widest:g}",
        ))
        return

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
        _report_error(viewer, Message("Error: Zoom could not be applied: {error}", error=error))
        return

    viewer._hud_timer.start()

    msg = Message("Zoom snapped to View Width: {width}", width=new_width)
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
