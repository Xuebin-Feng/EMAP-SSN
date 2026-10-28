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

"""The Viewer's commands: each module here is one command, named by its file, with run(viewer, args).

The program never names a command. A module that needs to be dispatched
differently says so itself, by declaring any of QUEUE / queue_kind(args, source)
(runs at once instead of waiting its turn), STARTS_MODEL_REQUEST /
starts_model_request(args) (refused when a model generated the command) and
ACTS_ON_JOBS / acts_on_jobs(args) (the portal runs it ahead of a request that
waits for jobs), or CONSOLE_BUTTON = True (the command adds a Console button, and
a loaded session runs `<command> --register-only` to bring it back). A module that
declares nothing waits its turn, starts no model request, acts on no jobs and has
no Console button. The convention is documented in Command_Declarations.
"""

import Command_Engine
