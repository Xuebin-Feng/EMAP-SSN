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

A command is a plug-in. What the program needs to know about it, the module says
itself, in declarations at its top level, and the program asks the module instead
of naming the command: a module dropped into this folder is dispatched, listed,
described to the Agent, put on the Session card and made resettable by what it
declares, with no change to the program. A module that declares nothing is treated
cautiously: it waits its turn, starts no model request, acts on no jobs, has no
Console button, picks no file, resets nothing, is catalogued with defaults, waits
for the user's approval in an Agent reply and gets a short generated item in the
Agent's instructions. What a module can declare, and where it is read:

How it is dispatched (Command_Declarations, which imports the module and asks it):
    QUEUE = 'instant', or queue_kind(args, source)
        runs at once instead of waiting its turn in the command queue.
    STARTS_MODEL_REQUEST, or starts_model_request(args)
        refused when a model generated the command (it cannot start another model request).
    ACTS_ON_JOBS, or acts_on_jobs(args)
        the command portal runs it ahead of a request that waits for jobs.
    CONSOLE_BUTTON = True
        the command adds a Console button; a loaded session runs `<command> --register-only` to bring it back.
    PICKS_SESSION_FILE = 'msa'
        the Session card's pick button for that file types this command.

How its work is split for the command queue (Command_Queue):
    plan(viewer, args), start(viewer, plan), compute(inputs, progress), apply(viewer, plan, result)
        all optional; a module without plan() runs whole, at its turn.

What it resets (commands/reset.py, which imports the modules that mention RESET_TARGETS):
    RESET_TARGETS = ('cluster',), reset(viewer, target), reset_changes_state(viewer, target)
        the reset targets it owns, how it restores each, and when that changes nothing.
    RESET_HELP = {'clusters': (40, 'Clears all cluster labels.')}
        what `reset help` and the usage line of a refusal list for its targets.

What the command catalog and the Agent are told about it (desktop.Command_Metadata, which
reads these literals from the source without importing the module):
    CATALOG_ENTRY
        its summary and arguments in the catalog (and in `help`).
    AGENT_POLICY
        'run', 'approve' or 'refuse' for a command in an Agent reply, by default and by first argument.
    AGENT_PROMPT, AGENT_PROMPT_ORDER
        its item in the Agent's instructions, and its place among them.

Commands may import one another (cluster needs group, label needs cluster and logo); a command whose import
is missing cannot be imported, and the program reports it when it is typed. tests/test_command_removal.py takes
each command away in turn, and renames some, to show that nothing else of the program breaks.

The rest of the program names no command, and tests/test_command_declarations.py
(CentralCodeNamesNoCommandTests) fails when a file of it does, or imports a command module, or holds a table of
command names. Its ALLOWED lists the few cases left, each with its reason, as the selection grammar's
own words (a #label# or an alignment or embedding position), the 'jobs' and 'offset' keys of a command request's
record, the 'run' verdict and entry point, the one place the engine hands a reset to the reset command, and a web page
that shares a name with its command (agent, esmfold, meta) in the page's own two files.
"""

import Command_Engine
