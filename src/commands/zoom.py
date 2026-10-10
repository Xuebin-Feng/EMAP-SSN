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


def _report_error(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg)


def run(viewer, args):
    if not args or args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="zoom <width>"),
            "Description: Sets the camera view width to exactly <width> while keeping the current "
            "center point and canvas aspect ratio.\nExamples:\n  zoom 500  (Sets the view width to 500 units)",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    try:
        new_width = float(args[0])
    except ValueError:
        _report_error(viewer, Message("Error: Zoom width must be a valid number."))
        return
    # float() also accepts nan, inf and 1e400; none of them is a view width.
    if not math.isfinite(new_width) or new_width <= 0:
        _report_error(viewer, Message("Error: Zoom width must be a positive, finite number."))
        return

    canvas_width, canvas_height = viewer.canvas.size
    if canvas_width <= 0 or canvas_height <= 0:
        _report_error(viewer, Message("Error: The canvas has no visible area, so the zoom cannot be applied."))
        return

    camera = viewer.view.camera
    rect = camera.rect

    center_x = rect.pos[0] + (rect.width / 2.0)
    center_y = rect.pos[1] + (rect.height / 2.0)

    aspect_ratio = canvas_width / canvas_height

    half_w = new_width / 2.0
    half_h = (new_width / aspect_ratio) / 2.0
    x_range = (center_x - half_w, center_x + half_w)
    y_range = (center_y - half_h, center_y + half_h)
    # Every check runs before the camera is touched. A huge finite width can still
    # overflow to inf on a tall canvas, and a scale below float32's normal range is too large.
    if (not all(math.isfinite(value) for value in x_range + y_range)
            or canvas_width < new_width * _FLOAT32_SMALLEST_NORMAL):
        _report_error(viewer, Message("Error: Zoom width is too large for the current view."))
        return
    if (canvas_width > new_width * _FLOAT32_LARGEST
            or new_width < max(abs(center_x), abs(center_y)) * canvas_width * _FLOAT32_EPSILON):
        _report_error(viewer, Message(
            "Error: Zoom width is too small to draw accurately at the current view centre."
        ))
        return

    # The rectangle is set directly, as the print command does: set_range adds a margin
    # to the requested range, and its margin=0 still adds 0.1 to each side.
    previous = (rect.pos[0], rect.pos[1], rect.width, rect.height)
    try:
        camera.rect = (x_range[0], y_range[0], new_width, new_width / aspect_ratio)
    except Exception as error:
        # The camera may already hold the new rectangle, so put the previous one back.
        camera.rect = previous
        _report_error(viewer, Message("Error: Zoom could not be applied: {error}", error=error))
        return

    viewer._hud_timer.start()

    msg = Message("Zoom snapped to View Width: {width}", width=new_width)
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
