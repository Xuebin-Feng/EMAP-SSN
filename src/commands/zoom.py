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

"""The zoom command: `zoom <percent>` sets the Viewer's zoom.

The command only reads its words, reports and carries its help. What a zoom is, the limits it keeps to and
how it is applied to the camera belong to the Viewer (desktop.Viewer_Zoom), which the Zoom box and print's
`zoom N` use as well, so none of them depends on this module. The names imported below are re-exported for
code that still takes them from here.
"""

import Command_Engine
from desktop.Viewer_Zoom import (  # noqa: F401  (the names this module does not use are re-exported)
    FIT_MARGIN_FRACTION,
    FIT_MARGIN_MIN,
    LIMIT_SLACK,
    MAX_ZOOM_PERCENT,
    MIN_VIEW_WIDTH,
    OPENING_RANGE_MARGIN,
    ZOOM_100_PERCENT_WIDTH,
    ZOOM_LIMIT_DIGITS,
    current_zoom,
    format_percent,
    limited_zoom_factor,
    opening_view,
    parse_zoom,
    round_down,
    set_zoom,
    view_for_zoom,
    view_width_limits,
    visible_width,
    width_at_zoom,
    zoom_limits,
    zoom_percent,
)
from utilities.Localization import JoinedMessage, Message
from desktop.Command_Metadata import argument, entry

# How the Viewer dispatches this command (Command_Declarations): the camera reads nothing a waiting
# command would change, so zoom runs at once.
QUEUE = 'instant'

# --- What this command declares about itself ---
# Changes only the camera.
AGENT_POLICY = 'run'

CATALOG_ENTRY = entry('Set the camera zoom, a percentage, while retaining its center and canvas aspect ratio.',
    argument('percent', 'Zoom as a percentage with an optional % sign: zoom 200 and zoom 200% both mean 200%, and a space may come before the sign. 100% is a view 300 layout units wide, and a bigger percentage is closer (200% is half as wide). From the zoom that shows every node, hidden ones too, rounded down to two significant figures, up to 60000% (the mouse wheel stops at the same limits); omission prints help.'))

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 50
AGENT_PROMPT = """
`zoom <PERCENT>`
- Sets the camera zoom to the requested percentage while preserving the current center and the canvas aspect ratio. The `%` sign is optional (`zoom 200` and `zoom 200%` are the same). 100% is a view 300 viewer-coordinate units wide; a bigger percentage is closer, and 200% is half as wide.
- PERCENT runs from the lowest zoom that shows every node, hidden ones too, rounded down to two significant figures, up to 60000%; that lowest zoom follows the nodes as they move. A percentage outside those limits is refused with both limits named. The mouse wheel stops at the same limits.
"""


def _report_error(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg)


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
