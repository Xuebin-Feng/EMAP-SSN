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

import os
import re
import fnmatch
import numpy as np
import SSN_Config as cfg
import SSN_Utils as utils
import Command_Engine

def print_help():
    print("""
    Advanced Selection Tool
    =======================
    Usage: select [MODE] <EXPRESSION>
           select <EXPRESSION> [MODE]
           select invert
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
      save [filename]             : Saves selected nodes to Input_Files/Header_Lists/
                                    (Supports .txt for headers or .fasta for sequences)

    Syntax & Targets:
      1. AA Position:  [AA][Pos] (e.g., P106, _100); negative positions require
                       parentheses (e.g., K(-1), K(-1.1))
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
      select add "ATA"                  (Adds nodes with "ATA" to selection)
      select remove #noise#             (Drops noise nodes from current selection)
      select keep P106                  (Filters current selection, keeping ONLY P106 nodes)
      select {Length>=500}&!#noise#     (Selects nodes with length >= 500 that are not noise)
    """)

def run(viewer, args):
    if not args:
        msg = "Error: Select command requires an expression or invert/save action.\nUsage: select [MODE] <EXPRESSION>"
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    if args[0].lower() == "save":
        if len(args) < 2:
            msg = "Error: Please provide a filename to save (e.g., 'select save top_nodes.txt' or 'my_seqs.fasta')."
            Command_Engine.print_help(viewer, msg)
            return
            
        filename = args[1]
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
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        selected_indices = getattr(viewer, 'selected_indices', [])
        if not selected_indices:
            msg = "Warning: No nodes are currently selected."
            Command_Engine.print_help(viewer, msg)
            return
            
        try:
            if is_fasta:
                from Bio import SeqIO
                # Lazy load original sequences if not already in memory
                if getattr(viewer, 'original_seqs', None) is None:
                    viewer.console_text.text = "Loading original sequences..."
                    viewer.original_seqs = {}
                    fasta_to_load = getattr(cfg, 'NODE_FASTA_FILE', getattr(cfg, 'SEQUENCES_FILE', ''))
                    if os.path.exists(fasta_to_load):
                        for r in SeqIO.parse(fasta_to_load, "fasta"):
                            viewer.original_seqs[r.id] = r
                            viewer.original_seqs[r.description] = r
                    else:
                        raise FileNotFoundError(f"Source FASTA not found at {fasta_to_load}")
                
                records_to_save = []
                missing_count = 0
                for idx in selected_indices:
                    full_h = viewer.full_headers[idx]
                    simple_h = viewer.full_headers[idx]
                    
                    s = viewer.original_seqs.get(full_h)
                    if not s: s = viewer.original_seqs.get(simple_h)
                    if not s: s = viewer.original_seqs.get(simple_h.split()[0])
                    
                    if s:
                        records_to_save.append(s)
                    else:
                        missing_count += 1
                        
                SeqIO.write(records_to_save, save_path, "fasta")
                msg = f"Saved {len(records_to_save)} sequences to {save_path}"
                if missing_count > 0:
                    msg += f" ({missing_count} missing from source FASTA)"
            else:
                with open(save_path, "w", encoding="utf-8", newline="\n") as f:
                    for idx in selected_indices:
                        f.write(f"{viewer.full_headers[idx]}\n")
                msg = f"Saved {len(selected_indices)} headers to {save_path}"
                
            Command_Engine.print_help(viewer, msg)
        except Exception as e:
            msg = f"Error saving file: {e}"
            Command_Engine.print_help(viewer, msg)
        return

    mode = "change"
    expr = None
    
    # Map keywords to their respective modes
    mode_map = {
        "change": "change",
        "add": "add", "plus": "add", "include": "add",
        "subtract": "subtract", "minus": "subtract", "remove": "subtract",
        "filter": "filter", "keep": "filter", "intersect": "filter",
        "save": "save",
        "invert": "invert"
    }

    for arg in args:
        clean_arg = arg.lower()
        if clean_arg in mode_map:
            mode = mode_map[clean_arg]
        else:
            expr = arg

    # --- Strict Invert Mode ---
    if mode == "invert":
        if expr:
            msg = "Error: 'invert' does not take expressions. Use '!EXPR' instead."
            Command_Engine.print_help(viewer, msg)
            return
            
        current_selection = set(getattr(viewer, 'selected_indices', []))
        all_visible = set(np.where(viewer.visible_mask)[0].tolist())
        
        final_selection = all_visible.difference(current_selection)
        
        viewer.selected_indices = list(final_selection)
        viewer.update_selection_visual()
        
        new_selected = len(final_selection)
        un_selected = len(current_selection)
        msg = f"Inverted selection. Selected {new_selected} nodes, Un-selected {un_selected} nodes."
        
        Command_Engine.print_help(viewer, msg)
        return

    if not expr:
        viewer.console_text.text = "Error: No logic expression provided."
        print("\nError: Please provide a valid boolean expression.")
        return

    if "$sele$" in expr.lower():
        header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
        os.makedirs(header_dir, exist_ok=True)
        sele_path = os.path.join(header_dir, "_sele.txt")
        
        with open(sele_path, "w", encoding="utf-8", newline="\n") as f:
            if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
                for idx in viewer.selected_indices:
                    f.write(viewer.full_headers[idx] + "\n")
                    
        expr = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)

    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    
    if expr:
        expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)

    try:
        mask = Command_Engine.parse_advanced_expression(expr, viewer_to_aln, valid_indices, viewer.full_headers, getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None), getattr(viewer, 'alignment', None), metadata=getattr(viewer, 'metadata', None))
        visible_indices = set(np.where(viewer.visible_mask)[0].tolist())
        new_indices = set(np.where(mask)[0].tolist()).intersection(visible_indices)
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, "Selection")
        return

    current_selection = set(getattr(viewer, 'selected_indices', []))
    
    if mode == "change":
        final_selection = new_indices
        unselected_count = len(current_selection.difference(final_selection))
        msg = f"Selected {len(final_selection)} nodes, Un-selected {unselected_count} nodes."
        
    elif mode in ["add", "plus", "include"]:
        final_selection = current_selection.union(new_indices)
        added_count = len(final_selection.difference(current_selection))
        msg = f"Added {added_count} nodes to selection (current total: {len(final_selection)} nodes)."
        
    elif mode in ["subtract", "minus", "remove"]:
        final_selection = current_selection.difference(new_indices)
        removed_count = len(current_selection.difference(final_selection))
        msg = f"Removed {removed_count} nodes from selection (remaining: {len(final_selection)})."

    # ---> NEW LOGIC: The Filter/Keep Mode <---
    elif mode in ["filter", "keep", "intersect"]:
        if not current_selection:
            msg = "Nothing to filter: No nodes are currently selected."
            final_selection = set()
        else:
            final_selection = current_selection.intersection(new_indices)
            removed_count = len(current_selection) - len(final_selection)
            msg = f"Filtered selection: Kept {len(final_selection)} nodes, removed {removed_count} nodes."

    viewer.selected_indices = list(final_selection)
    viewer.update_selection_visual()
    
    Command_Engine.print_help(viewer, msg)
