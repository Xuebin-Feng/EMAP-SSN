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
import EMAPSSN_Config as cfg
from utilities.Localization import JoinedMessage, Message


def print_help():
    print("""
    Alignment Numbering Offset Tool
    ===============================
    Usage:
      offset
          Displays the current alignment offset and whether it is active.
      offset <INTEGER>
          Changes the alignment offset for the current viewer session.
      offset help
          Displays this help message.

    Description:
      Adds an integer offset to reference-anchored alignment numbering without
      changing the alignment itself. Displayed positions are calculated as:

          displayed position = reference position + offset

      For example, an offset of 10 changes position 1 to 11 and insertion
      position 1.1 to 11.1. Setting the offset to 0 restores the original
      reference numbering.

    Requirements:
      A multiple-sequence alignment and a valid Alignment Reference ID must be
      loaded. Use 'reference <ID>' to select a reference during a session.
      The offset cannot be changed while reference numbering is inactive.

    Affected Commands:
      The updated numbering is used immediately by position-aware commands,
      including query, label, logo, color, select, group, hide, and spectrum.
      Existing alignment columns and sequence data are not modified.

    Notes:
      Positive and negative integers are accepted. Changes made with this
      command apply to the current viewer session. Configure Alignment Offset
      in EMAPSSN_Config to set the value used when launching a new session.
      In Boolean amino-acid expressions, parentheses are required around a
      negative displayed position: use K(-1) or K(-1.1), never K-1 or K-1.1.
      Grouped alternatives use (RHK)(-1), where the first parentheses define
      the residue set and the second parentheses contain the negative position.

    Examples:
      offset
          Reports the current offset.
      offset 10
          Starts reference numbering at 11 instead of 1.
      offset -5
          Subtracts 5 from every reference-anchored position.
      offset 0
          Restores the unshifted reference numbering.
    """)


def _current_offset(viewer):
    alignment = getattr(viewer, 'alignment', None)
    if alignment is not None and getattr(alignment, 'has_reference', False):
        return getattr(alignment, 'offset', 0), True
    return getattr(viewer, 'alignment_offset', getattr(cfg, 'ALIGNMENT_OFFSET', 0)), False


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    current_offset, is_active = _current_offset(viewer)
    if not args:
        if is_active:
            message = Message("Current Alignment Offset: {offset}", offset=current_offset)
        else:
            message = Message(
                "Current Alignment Offset: {offset} (inactive: no valid alignment reference is loaded)",
                offset=current_offset,
            )
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_succeeded(viewer, message)
        return

    if len(args) != 1:
        # The console line shows the first line; the usage is for the terminal.
        message = JoinedMessage(
            [Message("Error: Offset accepts exactly one integer."), "Usage: offset [INTEGER]"], separator="\n"
        )
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return

    try:
        new_offset = int(args[0])
    except (TypeError, ValueError):
        message = Message("Error: Alignment offset must be an integer, not '{value}'.", value=args[0])
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or not getattr(alignment, 'has_reference', False):
        message = Message(
            "Error: Alignment offset requires a correctly loaded reference. Use 'reference <ID>' first."
        )
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return

    if not alignment.set_offset(new_offset):
        message = Message("Error: Alignment offset could not be applied to the active reference.")
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return

    viewer.alignment_offset = new_offset
    cfg.ALIGNMENT_OFFSET = new_offset
    message = Message("Alignment Offset set to {offset}. Position numbering updated.", offset=new_offset)
    Command_Engine.print_help(viewer, message)
    Command_Engine.command_succeeded(viewer, message)
