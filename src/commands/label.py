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
import os
import glob
import re
import traceback
import datetime
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment 
import SSN_Config as cfg
import SSN_Utils as utils
import Cache_Manifest as cache_manifest
try:
    import commands.cluster as cluster_cmd
except ImportError:
    import cluster as cluster_cmd


GLOBAL_CONSERVATION_THRESHOLD = 0.97


def print_help():
    print("""
    Differential Labeling & Statistics Tool
    =======================================
    Generates a comprehensive XLSX report comparing the sequence properties and 
    conserved residues of each subset against the global dataset. Output is saved 
    to the 'Results/Cluster_Label/' directory.

    * PREREQUISITES: 
      1. A Multiple Sequence Alignment (MSA) must be loaded.
      2. A Reference Sequence must be set (use the 'reference' command).

    Usage: label [TARGET] [GLOBAL_MAX] [CLUSTER_MIN] [NAME]
       or: label [TARGET] [key value] [<key 2> <value 2> ...] [NAME]

    Targets (Default: clusters):
      clusters : Analyzes all defined topology clusters AND any custom groups.
      groups   : Analyzes ONLY custom groups (topology clusters not required).

    Arguments (Accepts decimals '0.4' or percentages '40%'):
      gmax (Outside Max)  : Default 40%. Max frequency the subset's dominant
                            residue can have among all aligned sequences OUTSIDE
                            that subset to be considered "Subset Specific".
      cmin (Cluster Min)  : Default 98%. Min frequency a residue must have WITHIN 
                            a subset to be reported as conserved.
      NAME                 : Optional final XLSX filename. '.xlsx' is added if
                            omitted. Numeric or reserved names must include the
                            extension, for example '0.4.xlsx' or 'groups.xlsx'.
                            The default remains Label_Output_YYYYMMDD_HHMMSS.xlsx.

    Fixed behavior:
      Globally conserved residues are reported when their frequency is greater
      than 97% across all aligned sequences. This threshold is not configurable.

    Examples:
      label                       (Uses gmax=40%, cmin=98%, and timestamp naming)
      label 0.4 0.9               (Positional: gmax=40%, cmin=90%)
      label groups cmin 90%       (Keyword: Analyzes groups, sets cmin to 90%)
      label groups report         (Writes report.xlsx)
      label 0.4 0.9 report        (Sets thresholds and writes report.xlsx)
      
    Note: Do not mix positional numbers after using keywords. A custom filename
          must be the final argument.
    """)

def parse_percentage(val_str):
    try:
        clean_str = val_str.replace('%', '')
        val = float(clean_str)
        if val > 1.0: return val / 100.0
        return val
    except ValueError: return None


def _normalize_output_filename(filename):
    """Return a safe XLSX basename for the configured label output directory."""
    filename = str(filename).strip()
    if not filename:
        raise ValueError("Filename cannot be empty.")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("Filename must not include a directory or path separators.")
    if re.search(r'[<>:"|?*\x00-\x1f]', filename):
        raise ValueError(f"Filename contains unsupported characters: '{filename}'.")
    if not filename.lower().endswith(".xlsx"):
        filename += ".xlsx"
    return filename

def get_sequence_stats(aln):
    lengths = []
    gap_chars = set(cfg.GAP_CHARS)
    for record in aln:
        seq_str = str(record.seq)
        ungapped_len = sum(1 for c in seq_str if c not in gap_chars)
        lengths.append(ungapped_len)
    if not lengths: return 0, 0, 0.0, 0.0
    arr = np.array(lengths)
    return int(np.min(arr)), int(np.max(arr)), np.mean(arr), np.std(arr)


def _get_amino_acid_counts(aln, col_idx):
    """Return non-gap amino-acid counts for one alignment column."""
    if hasattr(aln, 'matrix'):
        counts = Counter(aln.matrix[:, col_idx].data)
        aa_counts = {}
        for aa_int, count in counts.items():
            aa = aln.int_to_aa.get(aa_int, 'X')
            if aa not in cfg.GAP_CHARS:
                aa_counts[aa] = aa_counts.get(aa, 0) + count
    else:
        raw_counts = Counter(record.seq[col_idx].upper() for record in aln)
        aa_counts = {
            aa: count
            for aa, count in raw_counts.items()
            if aa not in cfg.GAP_CHARS
        }

    return {
        str(aa).upper(): int(count)
        for aa, count in aa_counts.items()
    }


def _get_amino_acid_frequencies(aln, col_idx):
    """Return occupancy-diluted, non-gap amino-acid frequencies for one column."""
    n_seqs = len(aln)
    if n_seqs == 0:
        return {}
    return {
        aa: count / n_seqs
        for aa, count in _get_amino_acid_counts(aln, col_idx).items()
    }


def _format_global_amino_acid_profile(aln, col_idx, frequencies=None):
    """Format a non-gap column profile using query.py's reporting semantics."""
    if frequencies is None:
        frequencies = _get_amino_acid_frequencies(aln, col_idx)

    profile = []
    for aa, frequency in frequencies.items():
        percentage = frequency * 100.0
        if percentage >= 1.0:
            profile.append((aa, percentage))
    profile.sort(key=lambda item: item[1], reverse=True)

    if not profile:
        return "-"
    return " | ".join(f"{aa} {percentage:>5.1f}%" for aa, percentage in profile)


def _is_subset_specific_residue(
    subset_aa,
    subset_frequency,
    subset_size,
    global_counts,
    global_size,
    cluster_min,
    global_max,
):
    """Apply cmin and compare gmax against the leave-subset-out background."""
    if subset_frequency < cluster_min:
        return False

    outside_size = global_size - subset_size
    if outside_size <= 0:
        return False

    subset_count = int(round(subset_frequency * subset_size))
    global_count = global_counts.get(str(subset_aa).upper(), 0)
    outside_count = global_count - subset_count
    if outside_count < 0:
        return False

    outside_frequency = outside_count / outside_size
    return outside_frequency < global_max


def _append_workbook_metadata(
    worksheet,
    out_filename,
    ref_display,
    offset_display,
    global_list,
    network_node_count=None,
    aligned_node_count=None,
    excluded_node_count=None,
):
    worksheet.append([f"Filename: {out_filename}"])
    worksheet.append([f"Reference: {ref_display}"])
    worksheet.append([f"Alignment Offset: {offset_display}"])
    if network_node_count is not None:
        worksheet.append([f"Network Nodes: {network_node_count}"])
        worksheet.append([f"Aligned Nodes: {aligned_node_count}"])
        worksheet.append([f"Excluded Unaligned Nodes: {excluded_node_count}"])
    worksheet.append([f"Global Conserved (>{int(GLOBAL_CONSERVATION_THRESHOLD * 100)}%)"])
    worksheet.append(global_list if global_list else ["None"])
    worksheet.append([])


def run(viewer, args):
    if args and args[0].lower() == 'reset':
        Command_Engine.execute_reset(viewer, ["clusters"])
        return

    try:
        alignment = getattr(viewer, 'alignment', None)
        if alignment is None or alignment.aln is None:
            viewer.console_text.text = "Error: Global Alignment not loaded."
            print("Error: Global Alignment not loaded.")
            return

        if len(alignment.aln) == 0:
            msg = (
                "Error: The selected MSA contains no aligned rows for the current "
                "network. Label analysis is unavailable."
            )
            viewer.console_text.text = msg
            print(msg)
            return

        if not getattr(alignment, 'has_reference', False):
            viewer.console_text.text = (
                "Error: No active alignment reference. Use 'reference <ID>' with "
                "a node present in the current MSA."
            )
            print(viewer.console_text.text)
            return

        if args and args[0].lower() in ['help', '-h', '-?']:
            print_help()
            if hasattr(viewer, 'console_text'):
                viewer.console_text.text = "Help information printed to the terminal"
            return

        # --- Parameters ---
        global_max = 0.40
        cluster_min = 0.98
        forced_target = "clusters"
        requested_filename = None

        valid_keys = {"gmax", "global_max", "g_max", "cmin", "cluster_min", "c_min"}
        fixed_keys = {"gmin", "global_min", "g_min"}
        valid_targets = {"cluster", "clusters", "group", "groups"}
        
        positional_args = []
        keyword_args = {}
        keyword_mode = False
        
        i = 0
        while i < len(args):
            arg = args[i].lower()
                
            # Catch targets anywhere in the command
            if arg in valid_targets:
                forced_target = "clusters" if arg in ["cluster", "clusters"] else "groups"
                i += 1
                continue

            if arg in fixed_keys:
                msg = "Error: gmin is fixed at 97% and cannot be set by the label command."
                viewer.console_text.text = msg
                print(msg)
                return

            # Catch space-separated keywords
            if arg in valid_keys:
                keyword_mode = True
                if i + 1 >= len(args):
                    msg = f"Error: Missing numerical value for '{arg}'."
                    viewer.console_text.text = msg
                    print(msg)
                    return
                
                val_str = args[i+1]
                val = parse_percentage(val_str)
                if val is None:
                    msg = f"Error: Invalid percentage '{val_str}' for '{arg}'."
                    viewer.console_text.text = msg
                    print(msg)
                    return
                
                # Standardize key names
                if arg in ["gmax", "global_max", "g_max"]: key_name = "gmax"
                elif arg in ["cmin", "cluster_min", "c_min"]: key_name = "cmin"
                
                if key_name in keyword_args:
                    msg = f"Error: Duplicate assignment for '{key_name}'."
                    viewer.console_text.text = msg
                    print(msg)
                    return
                    
                keyword_args[key_name] = val
                i += 2
                continue
                
            # Numeric tokens retain their historical positional gmax/cmin meaning.
            val = parse_percentage(arg)
            if val is not None:
                if keyword_mode:
                    msg = f"Error: Ambiguous input. Positional argument '{arg}' found after keywords."
                    viewer.console_text.text = msg
                    print(msg)
                    return
                positional_args.append(val)
                i += 1
                continue

            # One non-numeric final token is the custom output basename.
            if requested_filename is not None:
                msg = "Error: Provide only one custom output filename."
                viewer.console_text.text = msg
                print(msg)
                return
            if i != len(args) - 1:
                msg = "Error: A custom output filename must be the final argument."
                viewer.console_text.text = msg
                print(msg)
                return
            try:
                requested_filename = _normalize_output_filename(args[i])
            except ValueError as exc:
                msg = f"Error: {exc}"
                viewer.console_text.text = msg
                print(msg)
                return
            i += 1

        # Map positionals strictly to order: 1. gmax, 2. cmin
        pos_map = ["gmax", "cmin"]
        if len(positional_args) > 2:
            msg = "Error: Too many positional numerical arguments."
            viewer.console_text.text = msg
            print(msg)
            return
            
        for idx, p_val in enumerate(positional_args):
            target_key = pos_map[idx]
            if target_key in keyword_args:
                msg = f"Error: Ambiguous input. '{target_key}' defined both positionally and via keyword."
                viewer.console_text.text = msg
                print(msg)
                return
            keyword_args[target_key] = p_val
            
        # Apply final parsed variables (falling back to defaults)
        global_max = keyword_args.get("gmax", 0.40)
        cluster_min = keyword_args.get("cmin", 0.98)

        # --- Validations ---
        if forced_target == "clusters" and viewer.cluster_labels is None:
            viewer.console_text.text = "Error: Run 'cluster' first."
            print("Error: Run 'cluster' first to use cluster mode.")
            return
            
        if forced_target == "groups" and getattr(viewer, 'group_labels', None) is None:
            viewer.console_text.text = "Error: No groups defined."
            print("Error: No groups defined. Use the 'group' command first.")
            return

        # --- 1. Global Statistics ---
        print("Calculating Global Stats...")
        offset_display = utils.get_alignment_offset_display(viewer)
        print(f"Alignment Offset: {offset_display}")
        g_stats = viewer.alignment.calculate_frequencies(viewer.alignment.col_to_label)
        total_global_seqs = len(viewer.alignment.aln)
        total_network_nodes = getattr(viewer, 'n_nodes', len(viewer.full_headers))
        excluded_unaligned_nodes = total_network_nodes - total_global_seqs
        g_min, g_max, g_avg, g_std = get_sequence_stats(viewer.alignment.aln)

        global_count_cache = {}
        global_frequency_cache = {}

        def get_global_counts(label):
            if label not in global_count_cache:
                col_idx = viewer.alignment.label_to_col.get(label)
                global_count_cache[label] = (
                    _get_amino_acid_counts(viewer.alignment.aln, col_idx)
                    if col_idx is not None
                    else {}
                )
            return global_count_cache[label]

        def get_global_frequencies(label):
            if label not in global_frequency_cache:
                global_frequency_cache[label] = {
                    aa: count / total_global_seqs
                    for aa, count in get_global_counts(label).items()
                } if total_global_seqs else {}
            return global_frequency_cache[label]

        # --- 2. Resolve Base Directories ---
        fasta_file = getattr(cfg, 'NODE_FASTA_FILE', None)
        fasta_base = os.path.splitext(os.path.basename(fasta_file))[0] if fasta_file else getattr(cfg, 'SEQUENCE_SET', 'Network')
        metadata = cache_manifest.validate_network_schema(cfg.INPUT_HDF5)
        model_label = re.sub(
            r'[<>:"/\\|?*]', "_", metadata.model_name
        )
        lvl1_name = f"{fasta_base}_[{model_label}]"
        is_blast = metadata.network_type == "blast"
        if not is_blast:
            norm_m = getattr(cfg, 'NORM_MODE', None)
            if norm_m: lvl1_name += f"_{norm_m}"
            score_m = getattr(cfg, 'ALIGNMENT_SCORE', None)
            if score_m: lvl1_name += f"_{score_m}"
            
        lvl2_name_base = ""
        top_val = getattr(cfg, 'TOP_EDGE_PERCENT', None)
        if top_val is not None and str(top_val).strip() != "None":
            try: lvl2_name_base += f"Top{float(top_val)}Pct"
            except: pass
        else:
            thresh = getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0)
            try: lvl2_name_base += f"Score{float(thresh)}"
            except: pass
            
        if forced_target == "clusters":
            if getattr(viewer, 'last_cluster_params', None):
                c_mode_param, c_min_param = viewer.last_cluster_params
                if lvl2_name_base:
                    lvl2_name = f"{lvl2_name_base}_{c_mode_param}_Min{c_min_param}"
                else:
                    lvl2_name = f"{c_mode_param}_Min{c_min_param}"
            else:
                lvl2_name = lvl2_name_base
        else:
            lvl2_name = lvl2_name_base

        # --- 3. Prepare Tasks ---
        tasks = [] 
        
        print(f"Splitting Global Alignment for {forced_target.upper()}...")
        viewer_to_aln, _ = Command_Engine.get_alignment_mapping(viewer)
        
        if forced_target == "clusters":
            aln_idx_to_cid = {}
            for i, _header in enumerate(viewer.full_headers):
                if i >= len(viewer.cluster_labels): break
                cid = viewer.cluster_labels[i]
                aln_idx = int(viewer_to_aln[i])
                if aln_idx >= 0:
                    aln_idx_to_cid[aln_idx] = cid

            clusters_records = {}
            for i, record in enumerate(viewer.alignment.aln):
                if i in aln_idx_to_cid:
                    found_cid = aln_idx_to_cid[i]
                    if found_cid != -1:
                        if found_cid not in clusters_records: clusters_records[found_cid] = []
                        clusters_records[found_cid].append(record)
            
            for cid in sorted(clusters_records.keys()):
                sub_aln = MultipleSeqAlignment(clusters_records[cid])
                tasks.append(('cluster', cid, sub_aln, viewer.alignment.col_to_label))
        
        # Group splitting
        if getattr(viewer, 'group_labels', None):
            aln_idx_to_groups = {}
            for i, _header in enumerate(viewer.full_headers):
                if i >= len(viewer.group_labels): break
                groups = viewer.group_labels[i]
                aln_idx = int(viewer_to_aln[i])
                if groups and aln_idx >= 0:
                    aln_idx_to_groups[aln_idx] = groups
            
            groups_records = {}
            for i, record in enumerate(viewer.alignment.aln):
                if i in aln_idx_to_groups:
                    for g_name in aln_idx_to_groups[i]:
                        if g_name not in groups_records: groups_records[g_name] = []
                        groups_records[g_name].append(record)
            
            for g_name in sorted(groups_records.keys()):
                sub_aln = MultipleSeqAlignment(groups_records[g_name])
                tasks.append(('group', g_name, sub_aln, viewer.alignment.col_to_label))

        # --- 4. Process Tasks ---
        master_labels = set()
        cluster_results = []
        
        # Build color map for topology clusters using cluster_cmd
        cluster_ids = [entity_id for entity_type, entity_id, _, _ in tasks if entity_type == 'cluster']
        cluster_color_map = cluster_cmd.get_cluster_color_map(cluster_ids)

        for entity_type, entity_id, c_aln, c_map in tasks:
            try:
                # Calculate sequence stats
                c_size = len(c_aln)
                c_min_len, c_max_len, c_avg_len, c_std_len = get_sequence_stats(c_aln)

                # Calculate Residue Frequencies (using global map implicitly)
                c_stats = viewer.alignment.calculate_frequencies(c_map, exclude=[], aln=c_aln)
                c_dict = {}
                c_occ_dict = {} 
                
                for lbl in utils.sort_labels(c_stats.keys()):
                    if lbl not in c_stats: continue
                    c_aa, c_freq, c_occ = c_stats[lbl]
                    
                    c_occ_dict[lbl] = c_occ 
                    if lbl not in g_stats:
                        continue
                    if _is_subset_specific_residue(
                        c_aa,
                        c_freq,
                        c_size,
                        get_global_counts(lbl),
                        total_global_seqs,
                        cluster_min,
                        global_max,
                    ):
                        c_dict[lbl] = {"text": f"{c_aa}{lbl}", "occ": c_occ}
                        master_labels.add(lbl)
                
                # Format Output Styling
                if entity_type == 'cluster':
                    if entity_id in cluster_color_map:
                        rgb = cluster_color_map[entity_id]
                        hex_code = mcolors.to_hex(rgb)
                    else:
                        hex_code = "-"
                    name_str = f"Cluster {entity_id}"
                    sort_key = (0, entity_id)
                else:
                    hex_code = "-"
                    name_str = f"Group {entity_id}"
                    # Sort groups by count descending (-c_size), then alphabetically
                    sort_key = (1, -c_size, str(entity_id))

                cluster_results.append({
                    "type": entity_type,
                    "id": entity_id,
                    "name": name_str,
                    "sort_key": sort_key,
                    "count": c_size,
                    "hex": hex_code,
                    "min": c_min_len,
                    "max": c_max_len,
                    "avg": c_avg_len,
                    "std": c_std_len,
                    "data": c_dict,
                    "occ_data": c_occ_dict
                })
                    
            except Exception as e: 
                print(f"Skipping {entity_type} {entity_id} due to error: {e}")
                continue

        # --- 5. Export XLSX ---
        out_dir = cfg.CLUSTER_LABEL_DIR
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        # Preserve timestamp naming unless the user explicitly supplies a basename.
        if requested_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_filename = f"Label_Output_{timestamp}.xlsx"
        else:
            out_filename = requested_filename
        out_path = os.path.join(out_dir, out_filename)
        if os.path.exists(out_path):
            msg = f"Error: Output file already exists: {out_path}"
            viewer.console_text.text = msg
            print(msg)
            return

        global_list = []
        for lbl in utils.sort_labels(g_stats.keys()):
            aa, freq, occ = g_stats[lbl]
            if freq > GLOBAL_CONSERVATION_THRESHOLD:
                global_list.append(f"{aa}{lbl}")

        sorted_cols = utils.sort_labels(list(master_labels))
        cluster_results.sort(key=lambda x: x["sort_key"])
        all_occ_labels = utils.sort_labels(list(g_stats.keys()))
        
        ref_display = (
            getattr(viewer.alignment, 'resolved_ref_full', None)
            or getattr(viewer, 'active_reference', None)
            or "None"
        )
        
        # Prepare Metadata details
        fasta_file = getattr(cfg, 'NODE_FASTA_FILE', None)
        fasta_name = os.path.basename(fasta_file) if fasta_file else getattr(cfg, 'SEQUENCE_SET', 'N/A')
        
        network_file = getattr(cfg, 'INPUT_HDF5', None)
        network_name = os.path.basename(network_file) if network_file else "N/A"
        
        msa_file = getattr(cfg, 'MSA_FILE', None)
        alignment_name = os.path.basename(msa_file) if msa_file else "N/A"
        
        if forced_target == "clusters" and getattr(viewer, 'last_cluster_params', None):
            c_mode_param, c_min_param = viewer.last_cluster_params
            parts = c_mode_param.split('_')
            cluster_mode = parts[0] if parts else c_mode_param
            param_val = parts[1] if len(parts) > 1 else ""
            cluster_params = f"Param: {param_val}, Min Size: {c_min_param}" if param_val else f"Min Size: {c_min_param}"
        else:
            cluster_mode = "Groups" if forced_target == "groups" else "N/A"
            cluster_params = "N/A"

        label_params = (
            f"gmax_outside={int(global_max*100)}%, cmin={int(cluster_min*100)}%, "
            f"global_conservation_threshold={int(GLOBAL_CONSERVATION_THRESHOLD*100)}% (fixed), "
            f"target={forced_target}"
        )

        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font
        except ImportError:
            viewer.console_text.text = "Error: 'openpyxl' is required for XLSX export. Run: pip install openpyxl"
            print("Error: openpyxl not installed.")
            return

        try:
            wb = openpyxl.Workbook()
            
            # ==========================================
            # TAB 1: Meta Data
            # ==========================================
            ws_meta = wb.active
            ws_meta.title = "Meta Data"
            
            total_nodes = getattr(viewer, 'n_nodes', total_global_seqs)
            n_clusters = len([r for r in cluster_results if r['type'] == 'cluster'])
            n_groups = len([r for r in cluster_results if r['type'] == 'group'])
            if n_clusters > 0 and n_groups > 0:
                topology_str = f"{total_nodes} nodes with {n_clusters} Clusters, {n_groups} Groups"
            elif n_clusters > 0:
                topology_str = f"{total_nodes} nodes with {n_clusters} Clusters"
            else:
                topology_str = f"{total_nodes} nodes with {n_groups} Groups"

            ws_meta.append(["Metadata Field", "Value"])
            ws_meta.append(["Cluster Mode", cluster_mode])
            ws_meta.append(["Cluster Parameters", cluster_params])
            ws_meta.append(["Network Topology", topology_str])
            ws_meta.append(["Fasta Name", fasta_name])
            ws_meta.append(["Network Name", network_name])
            ws_meta.append(["Alignment Name", alignment_name])
            ws_meta.append(["Network Nodes", total_network_nodes])
            ws_meta.append(["Aligned Nodes", total_global_seqs])
            ws_meta.append(["Excluded Unaligned Nodes", excluded_unaligned_nodes])
            ws_meta.append(["Label Parameters", label_params])
            
            # Style header row for Meta Data tab
            header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            for cell in ws_meta[1]:
                cell.fill = header_fill
                cell.font = header_font
            ws_meta.column_dimensions['A'].width = 25
            ws_meta.column_dimensions['B'].width = 50

            percent_column = 2
            hex_color_column = 4
            position_start_column = 10
            percent_number_format = "0.00%"

            def node_fraction(count):
                return count / total_network_nodes if total_network_nodes else 0.0

            # ==========================================
            # TAB 2: Subset Specific Matrix
            # ==========================================
            ws1 = wb.create_sheet(title="Subset Stats")
            ws1.freeze_panes = "C1"  # Keep columns A and B visible during horizontal scrolling.
            
            # Write Metadata 
            _append_workbook_metadata(
                ws1,
                out_filename,
                ref_display,
                offset_display,
                global_list,
                total_network_nodes,
                total_global_seqs,
                excluded_unaligned_nodes,
            )
            
            # Write Headers 
            ws1.append(["Subset Specific Matrix"])
            headers1 = ["Subset Name", "Percent", "Count", "Hex Color", "Min Len", "Max Len", "Avg Len", "Std Dev", ""] + [f"#{c}" for c in sorted_cols]
            ws1.append(headers1)
            
            try:
                occ_cmap1 = cm.get_cmap('Reds_r')
            except AttributeError:
                occ_cmap1 = matplotlib.colormaps['Reds_r']
                
            def get_occ_fill1(occ_value):
                rgba = occ_cmap1(occ_value)
                hex_color = mcolors.to_hex(rgba)[1:].upper()
                return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")

            # Write Global Row 
            global_freq_row1 = [
                "Global Stats", 
                node_fraction(total_global_seqs),
                total_global_seqs, 
                "-",
                g_min, 
                g_max, 
                round(g_avg, 1), 
                round(g_std, 1),
                ""
            ]
            
            g_occ_dict1 = {}
            for col in sorted_cols:
                if col in g_stats:
                    _, _, g_occ = g_stats[col]
                    col_idx = viewer.alignment.label_to_col[col]
                    global_freq_row1.append(
                        _format_global_amino_acid_profile(
                            viewer.alignment.aln,
                            col_idx,
                            frequencies=get_global_frequencies(col),
                        )
                    )
                    g_occ_dict1[col] = g_occ
                else: 
                    global_freq_row1.append("-")
                    
            ws1.append(global_freq_row1)
            g_row_idx1 = ws1.max_row
            ws1.cell(row=g_row_idx1, column=percent_column).number_format = percent_number_format
            
            for c_idx, col in enumerate(sorted_cols):
                if col in g_occ_dict1:
                    col_letter_idx = c_idx + position_start_column
                    ws1.cell(row=g_row_idx1, column=col_letter_idx).fill = get_occ_fill1(g_occ_dict1[col])
                    
            ws1.append([]) # Blank row below Global Stats

            # Write Subset Rows
            last_type = None
            for res in cluster_results:
                if last_type == 'cluster' and res['type'] == 'group':
                    ws1.append([]) # Blank row separating Clusters and Groups
                last_type = res['type']
                row1 = [
                    res['name'], 
                    node_fraction(res['count']),
                    res['count'], 
                    res['hex'],
                    res['min'],
                    res['max'],
                    round(res['avg'], 1),
                    round(res['std'], 1),
                    ""
                ]
                
                row_occs1 = {}
                for c_idx, col in enumerate(sorted_cols): 
                    if col in res['data']:
                        row1.append(res['data'][col]["text"])
                    else:
                        row1.append("")
                        
                    if col in res['occ_data']:
                        row_occs1[c_idx + position_start_column] = res['occ_data'][col]
                    else:
                        row_occs1[c_idx + position_start_column] = 0.0
                        
                ws1.append(row1)
                current_row1 = ws1.max_row
                ws1.cell(row=current_row1, column=percent_column).number_format = percent_number_format
                
                if res['hex'] != "-":
                    hex_val = res['hex'].replace("#", "").upper()
                    ws1.cell(row=current_row1, column=hex_color_column).fill = PatternFill(start_color=hex_val, end_color=hex_val, fill_type="solid")
                
                for col_index, occ_val in row_occs1.items():
                    ws1.cell(row=current_row1, column=col_index).fill = get_occ_fill1(occ_val)
                    
            # ==========================================
            # TAB 2: Occupancy Stats
            # ==========================================
            ws2 = wb.create_sheet(title="Occupancy Stats")
            ws2.freeze_panes = "C1"  # Keep columns A and B visible during horizontal scrolling.
            
            # Write Metadata 
            _append_workbook_metadata(
                ws2,
                out_filename,
                ref_display,
                offset_display,
                global_list,
                total_network_nodes,
                total_global_seqs,
                excluded_unaligned_nodes,
            )
            
            # Write Headers 
            ws2.append(["Occupancy Matrix"])
            headers2 = ["Subset Name", "Percent", "Count", "Hex Color", "Min Len", "Max Len", "Avg Len", "Std Dev", ""] + [f"#{c}" for c in all_occ_labels]
            ws2.append(headers2)
            
            try:
                occ_cmap2 = cm.get_cmap('Greens')
            except AttributeError:
                occ_cmap2 = matplotlib.colormaps['Greens']
                
            def get_occ_fill2(occ_value):
                rgba = occ_cmap2(occ_value)
                hex_color = mcolors.to_hex(rgba)[1:].upper()
                return PatternFill(start_color=hex_color, end_color=hex_color, fill_type="solid")

            # Write Global Row 
            global_freq_row2 = [
                "Global Stats", 
                node_fraction(total_global_seqs),
                total_global_seqs, 
                "-",
                g_min, 
                g_max, 
                round(g_avg, 1), 
                round(g_std, 1),
                ""
            ]
            
            g_occ_dict2 = {}
            for col in all_occ_labels:
                global_freq_row2.append("") # Keep text blank
                if col in g_stats:
                    g_occ_dict2[col] = g_stats[col][2] # Occupancy is the 3rd item in the tuple
                else: 
                    g_occ_dict2[col] = 0.0
                    
            ws2.append(global_freq_row2)
            g_row_idx2 = ws2.max_row
            ws2.cell(row=g_row_idx2, column=percent_column).number_format = percent_number_format
            
            for c_idx, col in enumerate(all_occ_labels):
                col_letter_idx = c_idx + position_start_column
                ws2.cell(row=g_row_idx2, column=col_letter_idx).fill = get_occ_fill2(g_occ_dict2[col])
                
            ws2.append([]) # Blank row below Global Stats

            # Write Subset Rows
            last_type = None
            for res in cluster_results:
                if last_type == 'cluster' and res['type'] == 'group':
                    ws2.append([]) # Blank row separating Clusters and Groups
                last_type = res['type']
                row2 = [
                    res['name'], 
                    node_fraction(res['count']),
                    res['count'], 
                    res['hex'],
                    res['min'],
                    res['max'],
                    round(res['avg'], 1),
                    round(res['std'], 1),
                    ""
                ]
                
                row_occs2 = {}
                for c_idx, col in enumerate(all_occ_labels): 
                    row2.append("") # Keep text blank
                    row_occs2[c_idx + position_start_column] = res['occ_data'].get(col, 0.0)
                        
                ws2.append(row2)
                current_row2 = ws2.max_row
                ws2.cell(row=current_row2, column=percent_column).number_format = percent_number_format
                
                if res['hex'] != "-":
                    hex_val = res['hex'].replace("#", "").upper()
                    ws2.cell(row=current_row2, column=hex_color_column).fill = PatternFill(start_color=hex_val, end_color=hex_val, fill_type="solid")
                
                for col_index, occ_val in row_occs2.items():
                    ws2.cell(row=current_row2, column=col_index).fill = get_occ_fill2(occ_val)

            # Auto-fit Column A width based on the longest cluster or group name
            name_lengths = [len("Subset Name"), len("Global Stats")] + [len(res['name']) for res in cluster_results]
            max_name_len = max(name_lengths) if name_lengths else 15
            col_a_width = max(max_name_len + 3, 15)

            ws1.column_dimensions['A'].width = col_a_width
            ws2.column_dimensions['A'].width = col_a_width
            ws1.column_dimensions['B'].width = 10
            ws2.column_dimensions['B'].width = 10

            wb.save(out_path)
            
            msg = f"Exported to {out_path}"
            viewer.console_text.text = msg
            print(msg)
            utils.open_in_file_manager(out_dir)
        except Exception as e:
            viewer.console_text.text = f"IO Error: {e}"

    except Exception as e:
        viewer.console_text.text = f"Error: {e}"
        traceback.print_exc()
