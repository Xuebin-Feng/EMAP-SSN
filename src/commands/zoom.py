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

# How the Viewer dispatches this command (Command_Declarations): the camera reads nothing a waiting
# command would change, so zoom runs at once.
QUEUE = 'instant'

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


def _report_error(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg)


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


def run(viewer, args):
    if not args or args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        full = f"{ZOOM_100_PERCENT_WIDTH:g}"
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="zoom <percent>"),
            "Description: Sets the camera zoom to exactly <percent> while keeping the current "
            "center point and canvas aspect ratio. A bigger percentage is closer: "
            f"100% is a view {full} layout units wide and 200% is half as wide. The % sign is "
            "optional, and a space may come before it. The zoom must be at most "
            f"{format_percent(MAX_ZOOM_PERCENT)}% (a view {MIN_VIEW_WIDTH:g} units wide) and "
            "at least the zoom that shows every node, hidden ones too, rounded down to "
            f"{ZOOM_LIMIT_DIGITS} significant figures; the mouse wheel stops at the same limits.\n"
            "Examples:\n"
            f"  zoom 200   (Sets the zoom to 200%, a view {ZOOM_100_PERCENT_WIDTH / 2:g} units wide)\n"
            f"  zoom 50%   (Sets the zoom to 50%, a view {ZOOM_100_PERCENT_WIDTH * 2:g} units wide)",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    # `zoom 200 %` reaches here as two words: the sign may stand apart. Any other
    # second word is not part of the zoom.
    text = args[0] + args[1] if len(args) == 2 and args[1] == "%" else " ".join(args)
    try:
        percent = set_zoom(viewer, text)
    except ValueError as error:
        _report_error(viewer, error.args[0])
        return

    msg = Message("Zoom set to {zoom}%", zoom=format_percent(percent))
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
