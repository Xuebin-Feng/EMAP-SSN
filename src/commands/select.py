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

import os
import numpy as np
import EMAPSSN_Config as cfg
import Command_Engine
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import validate_output_basename


def _nodes(count):
    """'1 node' or '2 nodes', for a message that holds more than one count."""
    return Message("%n node(s)", n=count)


def print_help():
    print("""
    Advanced Selection Tool
    =======================
    Usage: select [MODE] <EXPRESSION>
           select <EXPRESSION> [MODE]
           select invert
           select save <FILENAME>
           select help

    Description:
      Selects nodes using complex boolean logic. Matches can be Amino Acid 
      positions, Header substrings, Clusters, Groups, or external Files.
      
      * IMPORTANT: This command only applies to and selects visible nodes.
      * IMPORTANT: Do NOT use spaces inside your expressions!

    Modes:
      change (default)            : Clears current selection and selects the new matches.
      add / plus / include        : Adds matches to the current selection.
      subtract / minus / remove   : Removes matches from the current selection.
      filter / keep / intersect   : Keeps ONLY currently selected nodes that match the expression.
      invert                      : Inverts current selection (takes no expression).

    Saving:
      select save <FILENAME>      : Saves the current selection to the header list
                                    directory (Input_Files/Header_Lists/ by default).
                                    FILENAME is a plain file name, not a path.
                                    Use .txt for headers or .fasta for sequences.
                                    'save' must come first; to save new matches,
                                    select them before saving.

    Syntax & Targets:
      1. AA Position:  [AA][Pos] (e.g., P106, _100), or ([AA...])[Pos] for
                       alternatives (e.g., (RHK)71); negative positions require
                       parentheses (e.g., K(-1), (RHK)(-1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list.txt@)
      4. NCBI/PDB:     @[NCBI][File]@ or @[PDB][File]@
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#)
      6. UI Selection: $sele$    (Explicitly targets selected nodes)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*})

    Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist in the current SSN. Invalid references abort the command
      without changing the selection. A valid expression may match zero nodes.

    Examples:
      select P106                       (Selects only P106 nodes)
      select (RHK)71                    (Selects nodes with R, H, or K at pos 71)
      select add "ATA"                  (Adds nodes with "ATA" to selection)
      select remove #noise#             (Drops noise nodes from current selection)
      select keep P106                  (Filters current selection, keeping ONLY P106 nodes)
      select {Length>=500}&!#noise#     (Selects nodes with length >= 500 that are not noise)
    """)

def run(viewer, args):
    if not args:
        # The console line shows the first line; the usage is for the terminal.
        msg = JoinedMessage([
            Message("Error: Select command requires an expression or invert/save action."),
            "Usage: select [MODE] <EXPRESSION>",
        ], separator="\n")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    if args[0].lower() == "save":
        if len(args) < 2:
            msg = Message("Error: Please provide a filename to save (e.g., 'select save top_nodes.txt' or 'my_seqs.fasta').")
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return
            
        # A plain name keeps the file in the header list directory; the web
        # agent and MCP clients name files too, not only the console.
        try:
            filename = validate_output_basename(args[1])
        except ValueError as error:
            msg = Message("Error: {error}", error=error)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return
        is_fasta = False
        
        if filename.lower().endswith('.fasta'):
            is_fasta = True
        elif not filename.lower().endswith('.txt'):
            filename += ".txt"
            
        save_dir = getattr(
            cfg,
            "HEADER_LIST_DIR",
            os.path.join("Input_Files", "Header_Lists"),
        )
        save_path = os.path.join(save_dir, filename)
        
        selected_indices = getattr(viewer, 'selected_indices', [])
        if not selected_indices:
            msg = Message("Warning: No nodes are currently selected.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_succeeded(viewer, msg)
            return
            
        try:
            if is_fasta:
                from commands.export import _get_in_memory_sequence_records
                from utilities.Sequence_Utils import write_fasta_atomic

                # Use the records the viewer loaded and checked against the
                # cache, keyed by the canonical headers the cache stores. The
                # source FASTA itself may never have been sanitized, so a raw
                # re-read missed every record whose header sanitizing changed.
                source_records = _get_in_memory_sequence_records(viewer)
                if not source_records:
                    raise ValueError(Message(
                        "no in-memory sequence set is available; use a .txt "
                        "filename to save the headers instead"
                    ))

                headers_to_save = []
                sequences_to_save = []
                missing_count = 0
                # Ascending node order: the selection is a set, whose order varies.
                for idx in sorted(selected_indices):
                    header = viewer.full_headers[idx]
                    sequence = source_records.get(header)
                    if sequence is None:
                        missing_count += 1
                    else:
                        headers_to_save.append(header)
                        sequences_to_save.append(sequence)

                write_fasta_atomic(save_path, headers_to_save, sequences_to_save)
                if missing_count > 0:
                    msg = Message(
                        "Saved %n sequence(s) to {path} ({missing} missing from the loaded sequences)",
                        n=len(headers_to_save),
                        path=save_path,
                        missing=missing_count,
                    )
                else:
                    msg = Message("Saved %n sequence(s) to {path}", n=len(headers_to_save), path=save_path)
            else:
                os.makedirs(save_dir, exist_ok=True)
                with open(save_path, "w", encoding="utf-8", newline="\n") as f:
                    for idx in sorted(selected_indices):
                        f.write(f"{viewer.full_headers[idx]}\n")
                msg = Message("Saved %n header(s) to {path}", n=len(selected_indices), path=save_path)
                
            Command_Engine.command_artifact(viewer, save_path)
            Command_Engine.print_help(viewer, msg)
        except Exception as e:
            msg = Message("Error saving file: {error}", error=e)
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return
        Command_Engine.command_succeeded(viewer, msg)
        return

    # 'save' is only an action on the current selection, never a mode that
    # follows an expression.
    if any(arg.lower() == "save" for arg in args[1:]):
        msg = Message(
            "Error: 'save' must come first. Use 'select save <FILENAME>' to save the "
            "current selection; to save new matches, select them first."
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    mode = "change"
    expr_args = []
    
    # Map keywords to their respective modes
    mode_map = {
        "change": "change",
        "add": "add", "plus": "add", "include": "add",
        "subtract": "subtract", "minus": "subtract", "remove": "subtract",
        "filter": "filter", "keep": "filter", "intersect": "filter",
        "invert": "invert"
    }

    for arg in args:
        clean_arg = arg.lower()
        if clean_arg in mode_map:
            mode = mode_map[clean_arg]
        else:
            expr_args.append(arg)

    if len(expr_args) > 1:
        msg = Message("Error: Select accepts exactly one whitespace-free Boolean expression.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return
    expr = expr_args[0] if expr_args else None

    if expr:
        classification = Command_Engine.classify_selection_expression(expr)
        if classification.kind != Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
            error = classification.error or Command_Engine.SelectionExpressionError(
                Message("'{expression}' is not a Boolean selection expression.", expression=expr)
            )
            Command_Engine.report_selection_error(viewer, expr, error, Message("Selection"))
            return

    # --- Strict Invert Mode ---
    if mode == "invert":
        if expr:
            msg = Message("Error: 'invert' does not take expressions. Use '!EXPR' instead.")
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return
            
        current_selection = set(getattr(viewer, 'selected_indices', []))
        all_visible = set(np.where(viewer.visible_mask)[0].tolist())
        
        final_selection = all_visible.difference(current_selection)
        
        viewer.selected_indices = list(final_selection)
        viewer.update_selection_visual()
        
        new_selected = len(final_selection)
        un_selected = len(current_selection)
        msg = Message(
            "Inverted selection. Selected {selected}, Un-selected {unselected}.",
            selected=_nodes(new_selected),
            unselected=_nodes(un_selected),
        )
        
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    if not expr:
        msg = Message("Error: No logic expression provided.")
        Command_Engine.show_status(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        print("\nError: Please provide a valid boolean expression.")
        Command_Engine.command_failed(viewer, '\nError: Please provide a valid boolean expression.')
        return

    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)

    try:
        mask = Command_Engine.parse_advanced_expression(
            expr,
            viewer_to_aln,
            valid_indices,
            viewer.full_headers,
            getattr(viewer, 'cluster_labels', None),
            getattr(viewer, 'group_labels', None),
            getattr(viewer, 'alignment', None),
            metadata=getattr(viewer, 'metadata', None),
            selection_mask=Command_Engine.get_selected_mask(viewer),
        )
        visible_indices = set(np.where(viewer.visible_mask)[0].tolist())
        new_indices = set(np.where(mask)[0].tolist()).intersection(visible_indices)
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, Message("Selection"))
        return

    current_selection = set(getattr(viewer, 'selected_indices', []))
    
    if mode == "change":
        final_selection = new_indices
        unselected_count = len(current_selection.difference(final_selection))
        msg = Message(
            "Selected {selected}, Un-selected {unselected}.",
            selected=_nodes(len(final_selection)),
            unselected=_nodes(unselected_count),
        )
        
    elif mode in ["add", "plus", "include"]:
        final_selection = current_selection.union(new_indices)
        added_count = len(final_selection.difference(current_selection))
        msg = Message(
            "Added %n node(s) to selection (current total: {total}).",
            n=added_count,
            total=_nodes(len(final_selection)),
        )
        
    elif mode in ["subtract", "minus", "remove"]:
        final_selection = current_selection.difference(new_indices)
        removed_count = len(current_selection.difference(final_selection))
        msg = Message(
            "Removed %n node(s) from selection (remaining: {remaining}).",
            n=removed_count,
            remaining=len(final_selection),
        )

    # ---> NEW LOGIC: The Filter/Keep Mode <---
    elif mode in ["filter", "keep", "intersect"]:
        if not current_selection:
            msg = Message("Nothing to filter: No nodes are currently selected.")
            final_selection = set()
        else:
            final_selection = current_selection.intersection(new_indices)
            removed_count = len(current_selection) - len(final_selection)
            msg = Message(
                "Filtered selection: Kept {kept}, removed {removed}.",
                kept=_nodes(len(final_selection)),
                removed=_nodes(removed_count),
            )

    else:
        # Every keyword in mode_map must have a branch above.
        msg = Message("Error: Unsupported selection mode '{mode}'.", mode=mode)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    viewer.selected_indices = list(final_selection)
    viewer.update_selection_visual()
    
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
