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
import os
import re
import fnmatch
import numpy as np
from collections import Counter
import SSN_Config as cfg
import SSN_Utils as utils
import Command_Engine
from utilities.Position_Parsing import (
    DISPLAYED_POSITION_ATOM_PATTERN,
    normalize_displayed_position_atom,
    reject_bare_negative_positions,
)


_QUERY_POSITION_ENDPOINT_PATTERN = (
    rf"(?:{DISPLAYED_POSITION_ATOM_PATTERN}|E(?:ND)?)"
)
_QUERY_POSITION_RANGE_RE = re.compile(
    rf"^({_QUERY_POSITION_ENDPOINT_PATTERN})\s*-\s*"
    rf"({_QUERY_POSITION_ENDPOINT_PATTERN})$",
    re.IGNORECASE,
)


def parse_query_positions(position_spec, valid_labels):
    """Expand a query position list against mapped alignment labels."""
    text = str(position_spec).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    reject_bare_negative_positions(text)

    parsed_args = [part.strip() for part in text.split(',') if part.strip()]
    expanded_positions = []
    max_val = valid_labels[-1][0] if valid_labels else (0, 0)

    def parse_to_tuple(value):
        normalized = normalize_displayed_position_atom(value, allow_end=True)
        if normalized in {"E", "END"}:
            return max_val
        major_text, separator, insertion_text = normalized.partition('.')
        return int(major_text), int(insertion_text) if separator else 0

    for part in parsed_args:
        range_match = _QUERY_POSITION_RANGE_RE.fullmatch(part)
        if range_match:
            start_value, end_value = sorted(
                [parse_to_tuple(range_match.group(1)), parse_to_tuple(range_match.group(2))]
            )
            for value, label in valid_labels:
                if start_value <= value <= end_value and label not in expanded_positions:
                    expanded_positions.append(label)
            continue

        normalized = normalize_displayed_position_atom(part, allow_end=True)
        if normalized in {"E", "END"} and valid_labels:
            normalized = valid_labels[-1][1]
        if normalized not in expanded_positions:
            expanded_positions.append(normalized)

    return expanded_positions

def print_help():
    print("""
    Subsection Query & Alignment Statistics Tool
    ==========================================
    Usage: query [EXPRESSION] [POSITIONS]
           query [EXPRESSION] [LOGIC_ARGUMENT]
           query help

    Description:
      Queries the loaded alignment for amino acid distribution at specified reference
      positions (Mode 1), OR searches for alignment positions matching specific amino 
      acid frequency criteria (Mode 2).
      
      Can query globally OR on a subset of nodes using logical sequence selection.

      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer. If no nodes are selected, it 
        defaults to querying ALL nodes in the entire network.

    Syntax Modes:
      1. Position Breakdown Mode:
         [POSITIONS] - Comma-separated list or ranges enclosed in brackets.
         Accepts decimal positions, and 'E' or 'END' for the last residue.
         Negative positions must be enclosed individually in parentheses.
         Example: [(-1), 0, 15.1, 20-30, 250-E, END] or [(-3)-(-1)]

      2. Position Frequency Search Mode:
         [LOGIC_ARGUMENT] - Frequency criteria with operators (>, <, >=, <=) and 
         logical operators (&, |, !, ^). Single arguments do NOT require ().
         Multi-condition queries MUST enclose each individual argument in ().
         Spaces are allowed. Accepts percentages (e.g. 10%) or decimals (e.g. 0.1).
         Accepts residue codes (A-Z) and 'GAP' or '_' for gaps (case-insensitive).
         Example: [K>10%], [(K>0.1) & (R>0.05)], [!(GAP>50%) & ((K>10%) | (R>10%))]

    Sequence Selection Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100); negative positions require
                       parentheses (e.g., K(-1), K(-1.1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list@, @my_seqs.fasta@)
      4. NCBI List:    @[NCBI][File]@ (Extracts & matches NCBI IDs from file and headers)
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#, #my_group#)
      6. UI Selection: $sele     (Targets nodes currently selected in viewer)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*})

    Logic Operators:
      & (AND), | (OR), ! (NOT), ^ (XOR)

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort the query. A valid selection
      expression may match zero nodes.

    Examples:
      query [10, 15, 20-30]                         (Queries pos 10, 15, and 20 to 30)
      query [(-1),0,10.1]                           (Queries negative, zero, and insertion positions)
      query #cluster_1# [(-3)-2]                    (Queries a range crossing zero in cluster 1)
      query [K>10%]                                 (Finds positions where Lysine > 10%)
      query [(K>0.1) & (R>0.05)]                    (Finds positions where K > 10% and R > 5%)
      query P106 [(K>20%) | (R>20%)]                (Finds positions with K or R > 20% in Pro106 subset)
      query {Length>500} [!(GAP>30%) & (K>5%)]     (Finds positions with <30% gaps and >5% Lys in length>500)
    """)

def run(viewer, args):
    if not args:
        msg = "Error: Query command requires a POSITIONS or LOGIC_ARGUMENT parameter.\nUsage: query [POSITIONS] or query [LOGIC_ARGUMENT]"
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        msg = "Error: No alignment loaded in the viewer."
        viewer.console_text.text = msg
        print(msg)
        return

    if len(alignment.aln) == 0:
        msg = (
            "Error: The selected MSA contains no aligned rows for the current network. "
            "Query analysis is unavailable."
        )
        viewer.console_text.text = msg
        print(msg)
        return

    # --- Reconstruct bracketed arguments (in case of spaces within brackets) ---
    reconstructed_args = []
    temp_bracket = []
    in_bracket = False
    
    for a in args:
        if '[' in a and not in_bracket:
            if a.count('[') > a.count(']'):
                in_bracket = True
                temp_bracket.append(a)
            else:
                reconstructed_args.append(a)
        elif in_bracket:
            temp_bracket.append(a)
            if ']' in a:
                joined = " ".join(temp_bracket)
                if joined.count('[') <= joined.count(']'):
                    reconstructed_args.append(joined)
                    temp_bracket = []
                    in_bracket = False
        else:
            reconstructed_args.append(a)
            
    if temp_bracket:
        reconstructed_args.extend(temp_bracket)
    args = reconstructed_args

    # --- 1. Extract Positions/Logic Argument (First argument containing brackets) ---
    bracket_indices = [i for i, a in enumerate(args) if a.strip().startswith('[') and a.strip().endswith(']')]
    
    if not bracket_indices:
        msg = "Error: No bracketed argument provided. Use [...] syntax (e.g., [10-20] or [K>10%])."
        Command_Engine.print_help(viewer, msg)
        return
        
    pos_idx = bracket_indices[0]
    pos_str = args.pop(pos_idx).strip()
    
    # --- 2. Isolate Expression & Apply Smart Fallbacks ---
    expr = "$sele$"
    if len(args) > 0:
        expr = "".join(args) # Join remaining to reconstruct expression without spaces

    # Smart Fallback to ALL Nodes
    if expr == "$sele$" and not getattr(viewer, 'selected_indices', []):
        expr = '"*"'  # The wildcard string matches all headers
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "No selection found. Defaulting to ALL nodes."
        print("No nodes selected. Defaulting to ALL nodes in the network.")

    # Handle UI Selection dynamically
    if "$sele$" in expr.lower():
        header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
        os.makedirs(header_dir, exist_ok=True)
        sele_path = os.path.join(header_dir, "_sele.txt")
        
        with open(sele_path, "w", encoding="utf-8", newline="\n") as f:
            if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
                for idx in viewer.selected_indices:
                    f.write(viewer.full_headers[idx] + "\n")
                    
        # Replace $sele$ shorthand with explicit file mask syntax
        expr = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)

    # --- 3. Compute Subset Rows ---
    target_rows = None
    n_seqs = len(viewer.alignment.aln)
    subset_mode = False

    if expr:
        viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
        
        expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)
        try:
            mask = Command_Engine.parse_advanced_expression(expr, viewer_to_aln, valid_indices, viewer.full_headers, getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None), getattr(viewer, 'alignment', None), metadata=getattr(viewer, 'metadata', None))
        except Exception as e:
            Command_Engine.report_selection_error(viewer, expr, e, "Query")
            return
            
        valid_nodes = np.where(mask)[0]
        aln_rows = viewer_to_aln[valid_nodes]
        target_rows = aln_rows[aln_rows != -1]
        
        n_seqs = len(target_rows)
        subset_mode = True
        
        if n_seqs == 0:
            msg = f"No sequences matched the expression '{expr}'. Aborting query."
            viewer.console_text.text = msg
            print("-" * 50)
            print(msg)
            print("-" * 50)
            return

    # --- 4. Detect Mode: Position Breakdown (Mode 1) vs Frequency Search (Mode 2) ---
    inner = pos_str[1:-1].strip()
    is_logic_mode = bool(re.search(r'[><]', inner))

    # Retrieve alignment metadata for printing
    msa_file = getattr(viewer.alignment, 'msa_file', None) or getattr(cfg, 'MSA_FILE', 'None')
    if isinstance(msa_file, str) and msa_file:
        msa_file_display = os.path.basename(msa_file)
    else:
        msa_file_display = str(msa_file)

    if getattr(viewer.alignment, 'has_reference', False):
        ref_display = getattr(viewer.alignment, 'resolved_ref_full', None) or getattr(viewer, 'active_reference', 'None')
    else:
        ref_display = "None (Unanchored)"

    offset_display = utils.get_alignment_offset_display(viewer)
    is_sparse = hasattr(viewer.alignment.aln, 'matrix')

    # Get mapped position labels in order
    label_to_col = getattr(viewer, 'alignment', None).label_to_col if getattr(viewer, 'alignment', None) else {}
    valid_labels = []
    for lbl in label_to_col.keys():
        try:
            parts = str(lbl).split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            valid_labels.append(((major, minor), lbl))
        except ValueError:
            pass
    valid_labels.sort(key=lambda x: x[0])
    ordered_pos_labels = [lbl for _, lbl in valid_labels]

    if is_logic_mode:
        # =====================================================================
        # MODE 2: POSITION FREQUENCY SEARCH MODE
        # =====================================================================
        print("-" * 50)
        if subset_mode:
            print(f"QUERY POSITION SEARCH SUBSET: '{expr}' ({n_seqs} sequences mapped)")
        else:
            print(f"QUERY POSITION SEARCH GLOBAL: All Mapped Alignment Sequences ({n_seqs} sequences)")
        print(f"Alignment File:   {msa_file_display}")
        print(f"Active Reference: {ref_display}")
        print(f"Alignment Offset: {offset_display}")
        print(f"Search Criteria:  [{inner}]")
        print("-" * 50)

        n_cols = len(ordered_pos_labels)
        if n_cols == 0:
            print("No valid alignment columns mapped.")
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "No valid alignment columns mapped."
            return

        # Precompute AA and Gap frequencies for all mapped columns
        all_gap_fracs = np.zeros(n_cols, dtype=float)
        all_aa_fracs = {} # aa_char -> 1D numpy array of length n_cols

        for idx, pos_label in enumerate(ordered_pos_labels):
            col_idx = label_to_col[pos_label]
            if is_sparse:
                if subset_mode:
                    sliced = viewer.alignment.aln.matrix[target_rows, col_idx]
                    if hasattr(sliced, 'toarray'):
                        dense_col = sliced.toarray().flatten()
                    else:
                        dense_col = np.array(sliced).flatten()
                    n_gaps = np.sum(dense_col == 0)
                    residues = dense_col[dense_col != 0]
                    counts = Counter(residues)
                else:
                    col_vec = viewer.alignment.aln.matrix[:, col_idx]
                    residues = col_vec.data
                    n_gaps = n_seqs - len(residues)
                    counts = Counter(residues)

                gap_frac = n_gaps / n_seqs if n_seqs > 0 else 0.0
                all_gap_fracs[idx] = gap_frac

                for aa_int, count in counts.items():
                    aa_char = viewer.alignment.aln.int_to_aa.get(aa_int, 'X').upper()
                    if aa_char not in all_aa_fracs:
                        all_aa_fracs[aa_char] = np.zeros(n_cols, dtype=float)
                    all_aa_fracs[aa_char][idx] += (count / n_seqs) if n_seqs > 0 else 0.0
            else:
                if subset_mode:
                    col_chars = [viewer.alignment.aln[row].seq[col_idx].upper() for row in target_rows]
                else:
                    col_chars = [rec.seq[col_idx].upper() for rec in viewer.alignment.aln]

                raw_counts = Counter(col_chars)
                n_gaps = sum(raw_counts[g] for g in cfg.GAP_CHARS if g in raw_counts)
                gap_frac = n_gaps / n_seqs if n_seqs > 0 else 0.0
                all_gap_fracs[idx] = gap_frac

                for aa_char, count in raw_counts.items():
                    if aa_char not in cfg.GAP_CHARS:
                        aa_clean = aa_char.upper()
                        if aa_clean not in all_aa_fracs:
                            all_aa_fracs[aa_clean] = np.zeros(n_cols, dtype=float)
                        all_aa_fracs[aa_clean][idx] += (count / n_seqs) if n_seqs > 0 else 0.0

        # Tokenize and evaluate atomic logical conditions
        masks = {}
        mask_idx = 0

        def cond_repl(match):
            nonlocal mask_idx
            target_aa_raw, op, val_str = match.groups()
            aa_clean = target_aa_raw.strip().upper()
            is_gap = (aa_clean == '_' or aa_clean == 'GAP')

            val_clean = val_str.strip()
            if val_clean.endswith('%'):
                thresh = float(val_clean[:-1]) / 100.0
            else:
                v = float(val_clean)
                thresh = v if v <= 1.0 else (v / 100.0)

            if is_gap:
                col_freqs = all_gap_fracs
            else:
                col_freqs = all_aa_fracs.get(aa_clean, np.zeros(n_cols, dtype=float))

            if op == '>':
                cond_mask = col_freqs > thresh
            elif op == '<':
                cond_mask = col_freqs < thresh
            elif op == '>=':
                cond_mask = col_freqs >= thresh
            elif op == '<=':
                cond_mask = col_freqs <= thresh
            else:
                cond_mask = np.zeros(n_cols, dtype=bool)

            mask_key = f"M_{mask_idx}"
            masks[mask_key] = cond_mask
            mask_idx += 1
            return f"masks['{mask_key}']"

        # 1. Replace parenthesized frequency arguments (e.g. (K>10%), ( GAP <= 0.5 ))
        expr_sub = re.sub(r'\(\s*([A-Za-z_]+)\s*(>=|<=|>|<)\s*(\d+(?:\.\d+)?%?)\s*\)', cond_repl, inner)

        # 2. Fallback: single unparenthesized argument (e.g. [K>10%], [gap<=0.2])
        if mask_idx == 0:
            single_match = re.fullmatch(r'\s*([A-Za-z_]+)\s*(>=|<=|>|<)\s*(\d+(?:\.\d+)?%?)\s*', inner)
            if single_match:
                expr_sub = cond_repl(single_match)

        # 3. Enforcement: ensure all comparison operators in multi-condition queries are inside ()
        if re.search(r'[><]', expr_sub) or mask_idx == 0:
            msg = f"Error: Individual frequency arguments in multi-condition queries must be enclosed in parentheses '()'.\nExample: [K>10%], [(K>0.1) & (R>0.05)], [!(GAP>50%) & ((K>10%) | (R>10%))]"
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "Error: Individual frequency arguments must be enclosed in ()"
            print("-" * 50)
            print(msg)
            print("-" * 50)
            return

        try:
            final_expr = expr_sub.replace("!", "~").replace("&", "&").replace("|", "|").replace("^", "^")
            pos_mask = eval(final_expr, {"__builtins__": {}}, {"masks": masks})
        except Exception as e:
            msg = f"Error parsing position logic '[{inner}]': {e}"
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = msg
            print(msg)
            return

        matching_indices = np.where(pos_mask)[0]
        matching_labels = [ordered_pos_labels[i] for i in matching_indices]

        print(f"Matching Positions ({len(matching_labels)} found):")
        if matching_labels:
            print(", ".join(str(lbl) for lbl in matching_labels))
            print("-" * 50)

            for idx in matching_indices:
                pos_label = ordered_pos_labels[idx]
                gap_pct = all_gap_fracs[idx] * 100.0

                valid_aas = []
                for aa_char, fracs in all_aa_fracs.items():
                    pct = fracs[idx] * 100.0
                    if pct >= 1.0:
                        valid_aas.append((aa_char, pct))
                valid_aas.sort(key=lambda x: x[1], reverse=True)

                out_str = f"Pos {pos_label:<8}\tGap {gap_pct:>5.1f}%"
                for aa, pct in valid_aas:
                    out_str += f" | {aa} {pct:>5.1f}%"
                print(out_str)
        else:
            print("[No positions matched the search criteria]")

        print("-" * 50)
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = f"Found {len(matching_labels)} matching position(s). Check terminal."
        return


    # =========================================================================
    # MODE 1: POSITION BREAKDOWN MODE (Existing Behavior)
    # =========================================================================
    found_count = 0

    print("-" * 50)
    if subset_mode:
        print(f"QUERY SUBSET: '{expr}' ({n_seqs} sequences mapped)")
    else:
        print(f"QUERY GLOBAL: All Mapped Alignment Sequences ({n_seqs} sequences)")
    print(f"Alignment File:   {msa_file_display}")
    print(f"Active Reference: {ref_display}")
    print(f"Alignment Offset: {offset_display}")
    print("-" * 50)

    try:
        expanded_positions = parse_query_positions(inner, valid_labels)
    except ValueError as exc:
        Command_Engine.print_help(viewer, f"Error: {exc}")
        return

    # Query the Matrix
    for pos in expanded_positions:
        if pos not in (getattr(viewer, 'alignment', None).label_to_col if getattr(viewer, 'alignment', None) else {}):
            print(f"Pos {pos: >5}: [Not found in active alignment mapping]")
            continue
            
        col_idx = viewer.alignment.label_to_col[pos]
        found_count += 1
        
        if is_sparse:
            if subset_mode:
                # Slicing specific rows returns a dense matrix or array
                sliced = viewer.alignment.aln.matrix[target_rows, col_idx]
                if hasattr(sliced, 'toarray'):
                    dense_col = sliced.toarray().flatten()
                else:
                    dense_col = np.array(sliced).flatten()
                
                n_gaps = np.sum(dense_col == 0)
                residues = dense_col[dense_col != 0]
                counts = Counter(residues)
            else:
                col_vec = viewer.alignment.aln.matrix[:, col_idx]
                residues = col_vec.data
                n_gaps = n_seqs - len(residues)
                counts = Counter(residues)
                
            aa_counts = {}
            for aa_int, count in counts.items():
                aa_char = viewer.alignment.aln.int_to_aa.get(aa_int, 'X')
                aa_counts[aa_char] = aa_counts.get(aa_char, 0) + count
        else:
            if subset_mode:
                col_chars = [viewer.alignment.aln[row].seq[col_idx].upper() for row in target_rows]
            else:
                col_chars = [rec.seq[col_idx].upper() for rec in viewer.alignment.aln]
                
            raw_counts = Counter(col_chars)
            n_gaps = sum(raw_counts[g] for g in cfg.GAP_CHARS if g in raw_counts)
            aa_counts = {aa: count for aa, count in raw_counts.items() if aa not in cfg.GAP_CHARS}
            
        # Gap-Diluted Calculation
        gap_pct = (n_gaps / n_seqs) * 100.0 if n_seqs > 0 else 0.0
        
        valid_aas = []
        for aa, count in aa_counts.items():
            pct = (count / n_seqs) * 100.0 if n_seqs > 0 else 0.0
            if pct >= 1.0:
                valid_aas.append((aa, pct))
                
        valid_aas.sort(key=lambda x: x[1], reverse=True)
        
        out_str = f"Pos {pos:<8}\tGap {gap_pct:>5.1f}%"
        for aa, pct in valid_aas:
            out_str += f" | {aa} {pct:>5.1f}%"
            
        print(out_str)
        
    print("-" * 50)
    
    if found_count > 0:
        viewer.console_text.text = f"Queried {found_count} position(s). Check terminal."
    else:
        viewer.console_text.text = "No valid positions queried."
