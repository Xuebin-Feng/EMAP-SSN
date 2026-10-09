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
from utilities.Localization import JoinedMessage, Message
from utilities.Sequence_Utils import pick_reference_header


def _resolve_reference_header(viewer, target):
    """Resolve TARGET once to the exact full network header that anchors numbering.

    The alignment holds only rows whose headers are network headers, so the
    network headers are the complete candidate set. The alignment matches the
    chosen full header exactly, so it anchors on the row that is reported, or
    leaves the reference inactive when the MSA lacks that sequence.
    """
    return pick_reference_header(viewer.full_headers, target)


def _current_reference_message(viewer):
    alignment = getattr(viewer, 'alignment', None)
    if alignment is not None and getattr(alignment, 'has_reference', False):
        return Message("Current Reference: {reference}", reference=alignment.resolved_ref_full)
    configured = getattr(viewer, 'active_reference', None)
    if configured and str(configured).strip().lower() != 'none':
        return Message(
            "Current Reference: {reference} (inactive; not resolved in the current MSA)", reference=configured
        )
    return Message("Current Reference: None")


def run(viewer, args):
    if not args:
        msg = _current_reference_message(viewer)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="reference [TARGET]"),
            "Description: Changes the reference sequence for alignment mapping.\n"
            "  - Call without arguments to see the current reference and whether it is active.\n"
            "  - Pass a full header, a leading identifier such as WP_0123.1, a partial header, or a wildcard\n"
            "    pattern such as WP_01* to set a new reference. An exact header or identifier takes priority;\n"
            "    otherwise the first match is used and a warning names it.\n"
            "Examples:\n  reference\n  reference SeqA",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    target = args[0]
    resolved_header = _resolve_reference_header(viewer, target)

    if resolved_header:
        viewer.active_reference = resolved_header

        print(f"\nReloading alignment...")
        Command_Engine.show_status(
            viewer, Message("Reloading alignment with new reference: {reference}...", reference=resolved_header)
        )

        viewer.load_global_alignment()

        if (
            viewer.alignment
            and viewer.alignment.aln is not None
            and getattr(viewer.alignment, 'has_reference', False)
        ):
            viewer.resolved_ref_full = viewer.alignment.resolved_ref_full
            msg = Message("Reference successfully set: {reference}.", reference=viewer.alignment.resolved_ref_full)
            Command_Engine.show_status(viewer, Message("Reference successfully set."))
        elif viewer.alignment and viewer.alignment.aln is not None:
            viewer.resolved_ref_full = None
            msg = Message(
                "Reference '{reference}' is configured but inactive because it is not "
                "present in the current MSA. Pure occupancy mode remains active.",
                reference=resolved_header,
            )
            Command_Engine.show_status(viewer, msg)
            print(f"\nWarning: {msg}")
        else:
            msg = Message("Error: Could not reload the current MSA for reference '{reference}'.", reference=target)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.show_status(viewer, msg)
            print(f"\n{msg}")
            return
    else:
        err = Message("Error: Reference '{reference}' not found.", reference=target)
        Command_Engine.command_failed(viewer, err)
        Command_Engine.show_status(viewer, err)
        print(f"\n{err}")
        return
    Command_Engine.command_succeeded(viewer, msg)
