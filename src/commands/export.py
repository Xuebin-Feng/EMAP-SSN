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
import re
from Bio import SeqIO
import SSN_Config as cfg
import SSN_Utils as utils
import Cache_Manifest as cache_manifest

def print_help():
    print("""
    FASTA Export Tool
    =================
    Usage: export [TARGET] [<TARGET_2> ...]
           export help

    Description:
      Extracts sequence subsets from the currently active viewer state and saves them 
      as standalone .fasta files. Files are automatically routed to strictly organized 
      subdirectories within that specified in the GUI.
      
    [TARGET] Arguments (Default: clusters):
      clusters : Exports sequences based on their assigned topology cluster ID. 
                 (Note: Unclustered 'Noise' nodes are automatically ignored).
      groups / group : Exports separate .fasta files for ALL custom group labels currently defined.
      group:<Name> : Exports only specific groups by prefixing with group: (e.g., group:kinase).
                     You can chain multiple specific groups (e.g., export group:kinase group:receptor).

    Examples:
      export             (Defaults to exporting all clusters)
      export group       (Exports all custom groups)
      export group:human (Exports only the sequences in the 'human' group)
    """)
    
def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '-?']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- 1. Parse Arguments ---
    target_mode = "clusters"
    specific_groups = []

    for arg in args:
        arg_lower = arg.lower()
        if arg_lower == "clusters":
            target_mode = "clusters"
        elif arg_lower in ["group", "groups"]:
            target_mode = "groups"
        elif arg_lower.startswith("group:"):
            target_mode = "specific"
            specific_groups.append(arg_lower[6:].strip())

    # --- Validations ---
    if target_mode == "clusters" and getattr(viewer, 'cluster_labels', None) is None:
        viewer.console_text.text = "Error: Run 'cluster' first."
        print("Error: Run 'cluster' first to export clusters.")
        return
        
    if target_mode in ["groups", "specific"] and getattr(viewer, 'group_labels', None) is None:
        viewer.console_text.text = "Error: No groups defined."
        print("Error: No groups defined. Use the 'group' command first.")
        return

    # --- 2. Load Source FASTA ---
    fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None)
    if not fasta_path or not os.path.exists(fasta_path):
        fasta_path = getattr(cfg, 'SEQUENCES_FILE', None)
        
    if not fasta_path or not os.path.exists(fasta_path):
        msg = "Error: Cannot find source FASTA file."
        viewer.console_text.text = msg
        print(msg)
        return

    print(f"Loading source FASTA: {os.path.basename(fasta_path)}...")
    source_records = {}
    try:
        for rec in SeqIO.parse(fasta_path, "fasta"):
            # Store by full header to ensure perfect mapping
            source_records[rec.description] = rec
    except Exception as e:
        msg = f"Error reading FASTA: {e}"
        viewer.console_text.text = msg
        print(msg)
        return

    # --- 3. Resolve Target Directory (NO Reference Injection) ---
    fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
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
        
    if target_mode == "clusters":
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

    sequence_export_dir = getattr(
        cfg,
        "SEQUENCE_EXPORT_DIR",
        os.path.join("Analysis_Results", "Sequence_Export"),
    )
    out_dir = os.path.join(sequence_export_dir, lvl1_name)
    
    if target_mode in ["groups", "specific"]:
        final_dir_name = f"{lvl2_name}_GROUPS" if lvl2_name else "GROUPS"
        out_dir = os.path.join(out_dir, final_dir_name)
    else:
        out_dir = os.path.join(out_dir, lvl2_name) if lvl2_name else out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # --- 4. Group Sequences ---
    file_map = {}
    missing_count = 0
    
    print("Mapping sequences...")
    for i, full_header in enumerate(viewer.full_headers):
        if full_header not in source_records:
            missing_count += 1
            continue
            
        record = source_records[full_header]
        
        if target_mode == "clusters":
            if i >= len(viewer.cluster_labels): continue
            cid = viewer.cluster_labels[i]
            if cid == -1: continue # Skip noise
            
            file_name = f"Cluster_{cid}.fasta"
            if file_name not in file_map: file_map[file_name] = []
            file_map[file_name].append(record)
            
        elif target_mode == "groups":
            if i >= len(viewer.group_labels): continue
            for g_name in viewer.group_labels[i]:
                file_name = f"{g_name}.fasta"
                if file_name not in file_map: file_map[file_name] = []
                file_map[file_name].append(record)
                
        elif target_mode == "specific":
            if i >= len(viewer.group_labels): continue
            for g_name in viewer.group_labels[i]:
                if g_name.lower() in specific_groups:
                    file_name = f"{g_name}.fasta"
                    if file_name not in file_map: file_map[file_name] = []
                    file_map[file_name].append(record)

    if missing_count > 0:
        print(f"Warning: {missing_count} viewer nodes were not found in the original FASTA file.")

    if not file_map:
        msg = "No valid subsets found to export."
        viewer.console_text.text = msg
        print(msg)
        return

    # --- 5. Write Files ---
    print(f"Exporting to: {out_dir}")
    files_written = 0
    seqs_written = 0
    
    for filename, recs in file_map.items():
        out_path = os.path.join(out_dir, filename)
        try:
            SeqIO.write(recs, out_path, "fasta")
            files_written += 1
            seqs_written += len(recs)
        except Exception as e:
            print(f"Failed to write {filename}: {e}")

    msg = f"Exported {files_written} files ({seqs_written} sequences)."
    viewer.console_text.text = msg
    print(f"\nSuccess! {msg}")
    
    # Auto-open the output folder in the system file manager
    utils.open_in_file_manager(out_dir)
