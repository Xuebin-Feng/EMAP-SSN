# Copyright 2026 Xuebin Feng
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
        msg = "Usage: undo\nDescription: Reverts the visual and spatial state of the network to the previous action.\nMost commands automatically save state before execution, allowing them to be undone."
        Command_Engine.print_help(viewer, msg)
        return
    viewer._do_undo()
