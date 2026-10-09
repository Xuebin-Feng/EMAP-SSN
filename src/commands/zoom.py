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

import Command_Engine
from utilities.Localization import JoinedMessage, Message


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

    rect = viewer.view.camera.rect

    center_x = rect.pos[0] + (rect.width / 2.0)
    center_y = rect.pos[1] + (rect.height / 2.0)

    aspect_ratio = canvas_width / canvas_height

    half_w = new_width / 2.0
    half_h = (new_width / aspect_ratio) / 2.0
    x_range = (center_x - half_w, center_x + half_w)
    y_range = (center_y - half_h, center_y + half_h)
    # A huge finite width can still overflow to inf on a tall canvas.
    if not all(math.isfinite(value) for value in x_range + y_range):
        _report_error(viewer, Message("Error: Zoom width is too large for the current view."))
        return

    viewer.view.camera.set_range(x=x_range, y=y_range)

    viewer._hud_timer.start()

    msg = Message("Zoom snapped to View Width: {width}", width=new_width)
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
