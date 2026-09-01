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

import Command_Engine
def run(viewer, args):
    if not args or args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: zoom <width>\nDescription: Sets the camera view width to exactly <width>+10% while keeping the current center point.\nExamples:\n  zoom 500  (Sets the view width to 500 units)"
        Command_Engine.print_help(viewer, msg)
        return
    
    try:
        new_width = float(args[0])
        rect = viewer.view.camera.rect
        
        center_x = rect.pos[0] + (rect.width / 2.0)
        center_y = rect.pos[1] + (rect.height / 2.0)
        
        canvas_width, canvas_height = viewer.canvas.size
        aspect_ratio = canvas_width / canvas_height
        
        half_w = new_width / 2.0
        half_h = (new_width / aspect_ratio) / 2.0
        
        viewer.view.camera.set_range(
            x=(center_x - half_w, center_x + half_w),
            y=(center_y - half_h, center_y + half_h)
        )
        
        viewer._hud_timer.start()
        
        msg = f"Zoom snapped to View Width: {new_width}"
        Command_Engine.print_help(viewer, msg)
        
    except ValueError:
        msg = "Error: Zoom width must be a valid number."
        Command_Engine.print_help(viewer, msg)
