import os
import re
import fnmatch
import numpy as np
from collections import Counter
import SSN_Config as cfg
import SSN_Utils as utils
import Command_Engine

def print_help():
    print("""
    Subsection Query & Alignment Statistics Tool
    ==========================================
    Usage: query [EXPRESSION] [POSITIONS]
           query help

    Description:
      Queries the loaded alignment for the amino acid distribution at the specified 
      reference positions. Can query globally OR on a subset of nodes using logic.

      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer. If no nodes are selected, it 
        defaults to querying ALL nodes in the entire network.

    Syntax & Targets:
      1. Positions:  Comma-separated list or ranges enclosed in brackets. 
                     Accepts decimal positions, and 'E' or 'END' for the last residue.
                     Example: [10, 15.1, 20-30, 250-E, END]
      2. Expression: Boolean logic (e.g., #cluster_1#, "ATA", or $sele$).
                     Do NOT use spaces inside your logical expression.

    Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100)
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list@, @my_seqs.fasta@)
      4. NCBI List:    @[NCBI][File]@ (Extracts & matches NCBI IDs from file and headers)
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#, #my_group#)
      6. UI Selection: $sele     (Targets nodes currently selected in viewer)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*})

    Logic Operators:
      & (AND), | (OR), ! (NOT), ^ (XOR)

    Examples:
      query [10, 15, 20-30]         (Queries pos 10, 15, and 20 to 30 for selected/all nodes)
      query P106 [150-160]          (Sub-query pos 150-160 ONLY for nodes with Pro at 106)
      query "ATA"&#kinase# [45-50]  (Sub-query pos 45-50 for nodes with "ATA" AND in #kinase#)
      query {Length>500} [10-20]    (Sub-query pos 10-20 for nodes with Length > 500)
    """)

def run(viewer, args):
    if not args:
        msg = "Error: Query command requires a POSITIONS parameter.\nUsage: query [POSITIONS]"
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    if not viewer.alignment.aln:
        msg = "Error: No alignment loaded in the viewer."
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

    # --- 1. Extract Positions Argument (First argument containing brackets) ---
    bracket_indices = [i for i, a in enumerate(args) if a.startswith('[') and a.endswith(']')]
    
    if not bracket_indices:
        msg = "Error: No positions provided. Use [...] syntax (e.g., [10-20])."
        Command_Engine.print_help(viewer, msg)
        return
        
    pos_idx = bracket_indices[0]
    pos_str = args.pop(pos_idx)
    
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
        
        with open(sele_path, "w", encoding="utf-8") as f:
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
        viewer_to_aln = np.full(len(viewer.full_headers), -1, dtype=int)
        for i, h in enumerate(viewer.full_headers):
            if h in viewer.alignment.seq_map:
                viewer_to_aln[i] = viewer.alignment.seq_map[h]
                
        valid_indices = np.where(viewer_to_aln != -1)[0]
        
        expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)
        try:
            mask = Command_Engine.parse_advanced_expression(expr, viewer_to_aln, valid_indices, viewer.full_headers, getattr(viewer, 'cluster_labels', None), getattr(viewer, 'group_labels', None), getattr(viewer, 'alignment', None), metadata=getattr(viewer, 'metadata', None))
        except Exception as e:
            viewer.console_text.text = f"Query Logic Error: {e}"
            print(f"Error parsing subset logic '{expr}': {e}")
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

    # --- 4. Parse Ranges and Positions ---
    is_sparse = hasattr(viewer.alignment.aln, 'matrix')
    found_count = 0

    print("-" * 50)
    if subset_mode:
        print(f"QUERY SUBSET: '{expr}' ({n_seqs} sequences mapped)")
    else:
        print(f"QUERY GLOBAL: All Mapped Alignment Sequences ({n_seqs} sequences)")
    print(f"Alignment Offset: {utils.get_alignment_offset_display(viewer)}")
    print("-" * 50)

    # NEW: Extract the inner string from the brackets and split by comma
    inner = pos_str[1:-1]
    parsed_args = [x.strip() for x in inner.split(',') if x.strip()]
    
    expanded_positions = []
    valid_labels = []
    
    # Store valid labels as Tuples for version-like sorting (e.g., 188.10 > 188.6)
    for lbl in (getattr(viewer, 'alignment', None).label_to_col if getattr(viewer, 'alignment', None) else {}).keys():
        try:
            parts = str(lbl).split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            valid_labels.append(((major, minor), lbl))
        except ValueError:
            pass
    valid_labels.sort(key=lambda x: x[0])

    max_val = valid_labels[-1][0] if valid_labels else (0, 0)

    def parse_to_tuple(s):
        s_clean = str(s).strip().upper()
        if s_clean in ["E", "END"]:
            return max_val
        p = s_clean.split('.')
        return (int(p[0]), int(p[1]) if len(p) > 1 else 0)

    for part in parsed_args:
        if '-' in part and not part.startswith('-'):
            try:
                s_str, e_str = part.split('-', 1)
                s_val, e_val = sorted([parse_to_tuple(s_str), parse_to_tuple(e_str)])
                
                for val, lbl in valid_labels:
                    # Tuple comparison naturally handles Major/Minor logic correctly
                    if s_val <= val <= e_val and lbl not in expanded_positions:
                        expanded_positions.append(lbl)
            except ValueError:
                if part not in expanded_positions:
                    expanded_positions.append(part)
        else:
            part_clean = part.strip().upper()
            if part_clean in ["E", "END"] and valid_labels:
                last_lbl = valid_labels[-1][1]
                if last_lbl not in expanded_positions:
                    expanded_positions.append(last_lbl)
            else:
                if part not in expanded_positions:
                    expanded_positions.append(part)

    # --- 5. Query the Matrix ---
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
