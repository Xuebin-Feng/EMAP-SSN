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
from utilities.Sequence_Utils import reference_header_matches


def _resolve_reference_header(viewer, target):
    """Resolve TARGET once to the exact full network header that anchors numbering.

    Returns (header, matches). MATCHES are the network headers TARGET names, in
    network order, and HEADER is the one to anchor on, or None.

    The alignment holds only rows whose headers are network headers, so the
    network headers are the complete candidate set. The alignment matches the
    chosen full header exactly, so it anchors on the row that is reported.
    With an alignment loaded, HEADER is the first match that has a row in it,
    so a pattern is not stuck on an unaligned header that happens to come
    first, and is None when no match has one. Without an alignment, HEADER is
    the first match.
    """
    matches = reference_header_matches(viewer.full_headers, target)
    aln = getattr(getattr(viewer, 'alignment', None), 'aln', None)
    if aln is None:
        header = matches[0] if matches else None
    else:
        # The alignment's own test for whether a header has a row to anchor on.
        header = next((match for match in matches if aln.find_reference_index(match) != -1), None)
    if header is not None and len(matches) > 1:
        print(Message(
            "Warning: Multiple matches found for '{target}'. Using '{reference}'.",
            target=target, reference=header,
        ))
    return header, matches


def _refuse_unaligned_reference(viewer, name):
    """Report that NAME has no sequence in the loaded alignment; nothing has changed."""
    err = Message(
        "Error: '{name}' is not in the loaded alignment, so it cannot be the reference. "
        "The reference is unchanged.",
        name=name,
    )
    Command_Engine.command_failed(viewer, err)
    Command_Engine.show_status(viewer, err)
    print(f"\n{err}")


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
            "    otherwise the first match is used and a warning names it. When several headers match, the\n"
            "    first one that has a sequence in the loaded alignment is used.\n"
            "  - The reference must have a sequence in the loaded alignment. A target without one is refused,\n"
            "    and the reference and offset stay as they were.\n"
            "  - Only one TARGET is accepted. A further argument is refused, and the reference and offset\n"
            "    stay as they were.\n"
            "Examples:\n  reference\n  reference SeqA",
        ], separator="\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    if len(args) > 1:
        # Refused before anything changes; the first argument after TARGET is named.
        message = Message(
            "Error: {syntax} takes one target; '{argument}' was not used. The reference is unchanged.",
            syntax="reference [TARGET]", argument=args[1],
        )
        Command_Engine.print_help(viewer, message, report_message=False)
        Command_Engine.command_failed(viewer, message)
        return

    target = args[0]
    resolved_header, matches = _resolve_reference_header(viewer, target)

    if resolved_header:
        # A failed reload restores the alignment and reference that were in effect.
        backup_alignment = viewer.alignment
        backup_active_ref = viewer.active_reference
        backup_resolved_ref = getattr(viewer, 'resolved_ref_full', None)

        def restore_previous_state():
            viewer.alignment = backup_alignment
            viewer.active_reference = backup_active_ref
            viewer.resolved_ref_full = backup_resolved_ref

        viewer.active_reference = resolved_header

        print(f"\nReloading alignment...")
        Command_Engine.show_status(
            viewer, Message("Reloading alignment with new reference: {reference}...", reference=resolved_header)
        )

        try:
            # A new reference renumbers the columns; the rows already loaded
            # are kept when the MSA file has not changed.
            viewer.load_global_alignment(reuse_loaded=True)
        except Exception:
            restore_previous_state()
            raise

        if (
            viewer.alignment
            and viewer.alignment.aln is not None
            and getattr(viewer.alignment, 'has_reference', False)
        ):
            viewer.resolved_ref_full = viewer.alignment.resolved_ref_full
            msg = Message("Reference successfully set: {reference}.", reference=viewer.alignment.resolved_ref_full)
            Command_Engine.show_status(viewer, Message("Reference successfully set."))
        elif viewer.alignment and viewer.alignment.aln is not None:
            # The header has a row in the alignment, so the reload anchors on it;
            # were it not to, the previous reference stays rather than going inactive.
            restore_previous_state()
            _refuse_unaligned_reference(viewer, resolved_header)
            return
        else:
            # The loader says why on the manager it leaves behind, which the
            # restore replaces.
            reason = getattr(viewer.alignment, 'load_failure', None)
            restore_previous_state()
            msg = Message("Error: Could not reload the current MSA for reference '{reference}'.", reference=target)
            if reason is not None:
                msg = JoinedMessage([msg, reason])
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.show_status(viewer, msg)
            print(f"\n{msg}")
            return
    elif matches:
        # The network has the node but the loaded alignment has no sequence for it.
        _refuse_unaligned_reference(viewer, matches[0] if len(matches) == 1 else target)
        return
    else:
        err = Message("Error: Reference '{reference}' not found.", reference=target)
        Command_Engine.command_failed(viewer, err)
        Command_Engine.show_status(viewer, err)
        print(f"\n{err}")
        return
    Command_Engine.command_succeeded(viewer, msg)
