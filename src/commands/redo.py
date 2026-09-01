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
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = "Usage: redo\nDescription: Reapplies a state that was previously undone using the `undo` command."
        Command_Engine.print_help(viewer, msg)
        return
    viewer._do_redo()
