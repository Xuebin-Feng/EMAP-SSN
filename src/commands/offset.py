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
import EMAPSSN_Config as cfg


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
            viewer.console_text.text = "Help information printed to the terminal"
        return

    current_offset, is_active = _current_offset(viewer)
    if not args:
        suffix = "" if is_active else " (inactive: no valid alignment reference is loaded)"
        Command_Engine.print_help(
            viewer,
            f"Current Alignment Offset: {current_offset}{suffix}",
        )
        return

    if len(args) != 1:
        Command_Engine.print_help(
            viewer,
            "Error: Offset accepts exactly one integer.\nUsage: offset [INTEGER]",
        )
        return

    try:
        new_offset = int(args[0])
    except (TypeError, ValueError):
        Command_Engine.print_help(
            viewer,
            f"Error: Alignment offset must be an integer, not '{args[0]}'.",
        )
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or not getattr(alignment, 'has_reference', False):
        Command_Engine.print_help(
            viewer,
            "Error: Alignment offset requires a correctly loaded reference. "
            "Use 'reference <ID>' first.",
        )
        return

    if not alignment.set_offset(new_offset):
        Command_Engine.print_help(
            viewer,
            "Error: Alignment offset could not be applied to the active reference.",
        )
        return

    viewer.alignment_offset = new_offset
    cfg.ALIGNMENT_OFFSET = new_offset
    Command_Engine.print_help(
        viewer,
        f"Alignment Offset set to {new_offset}. Position numbering updated.",
    )
