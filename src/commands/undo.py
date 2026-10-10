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

import re

import Command_Engine
from utilities.Localization import JoinedMessage, Message

# ASCII digits only: int() would also take signs, underscores and other digits.
_STEP_COUNT = re.compile(r"[0-9]+")
# Ten digits or more is far beyond any history, so such a count means all of it.
_ALL_STEPS = 10 ** 9


def _steps_argument(args):
    """(steps, None) for the arguments of `undo [N]`, or (None, refusal) naming what is wrong with them."""
    count = args[0]
    digits = count.lstrip("0")
    if not _STEP_COUNT.fullmatch(count) or not digits:
        return None, Message(
            "Error: {syntax} takes a positive whole number of steps, not '{value}'.",
            syntax="undo [N]", value=count,
        )
    if len(args) > 1:
        return None, Message(
            "Error: {syntax} takes one number of steps; '{argument}' was not used.",
            syntax="undo [N]", argument=args[1],
        )
    return (int(digits) if len(digits) < 10 else _ALL_STEPS), None


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="undo [N]"),
            "Description: Reverts the visual and spatial state of the network to the previous action.\n"
            "Most commands automatically save state before execution, allowing them to be undone.\n"
            "  - With N, a positive whole number, it undoes up to N steps and stops early when the history runs out.",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return
    if not args:
        changed = viewer._do_undo()
        Command_Engine.command_succeeded(viewer, "Undo successful." if changed else "Nothing to undo.")
        return

    steps, refusal = _steps_argument(args)
    if refusal is not None:
        # Refused before anything changes; the usage follows the error in the terminal.
        message = JoinedMessage([refusal, Message("Usage: {syntax}", syntax="undo [N]")], separator="\n")
        Command_Engine.print_help(viewer, message, report_message=False)
        Command_Engine.command_failed(viewer, message)
        return

    done = 0
    while done < steps and viewer._do_undo():
        done += 1
    if done:
        Command_Engine.command_succeeded(viewer, Message("Undid %n step(s).", n=done))
    else:
        Command_Engine.command_succeeded(viewer, "Nothing to undo.")
