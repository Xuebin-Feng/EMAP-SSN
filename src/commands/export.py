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
import os
import re
import EMAPSSN_Config as cfg
import Cache_Manifest as cache_manifest
from utilities.Application_Windows import open_in_file_manager
from utilities.FASTA_Sanitization import write_fasta_atomic

SEQUENCE_EXPORT_DIRECTORY = os.path.join(
    "$analysis_result$", "Sequence_Export"
)


def _get_in_memory_sequence_records(viewer):
    """Return canonical header/sequence pairs already loaded by the viewer."""
    selected_records = getattr(viewer, "_selected_fasta_records", None)
    if selected_records is not None:
        return {
            str(header): str(sequence)
            for header, sequence in selected_records
        }

    sequence_map = getattr(viewer, "sequences_map", None)
    if sequence_map is None:
        return {}
    return {
        header: str(sequence_map[header])
        for header in viewer.full_headers
        if header in sequence_map
    }

def print_help():
    print("""
    FASTA Export Tool
    =================
    Usage: export [clusters | groups | #LABEL# ...]
           export help

    Description:
      Extracts sanitized sequence subsets from the currently active viewer state and
      saves them as standalone .fasta files. Files are automatically routed to strictly
      organized subdirectories beneath the configured Analysis Results directory.
      
    [TARGET] Arguments (Default: clusters):
      clusters : Exports sequences based on their assigned topology cluster ID. 
                 (Note: Unclustered 'Noise' nodes are automatically ignored).
      groups / group : Exports separate .fasta files for ALL custom group labels currently defined.
      #LABEL# : Exports a specific custom group, topology cluster, or noise label.
                Multiple labels may be mixed and repeated labels are deduplicated.

    Examples:
      export             (Defaults to exporting all clusters)
      export group       (Exports all custom groups)
      export #human# (Exports only the sequences in the 'human' group)
      export #cluster_1# #noise# (Exports one cluster and the explicit noise subset)
    """)
    
def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '-?']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- 1. Parse Arguments ---
    target_mode = "clusters"
    specific_targets = []
    mode_tokens = []
    label_tokens = []

    for arg in args:
        arg_lower = arg.lower()
        if arg_lower.startswith("group:"):
            Command_Engine.print_help(
                viewer,
                "Error: Legacy export group:NAME syntax is no longer supported. "
                "Use export #NAME#.",
            )
            return
        if arg_lower == "clusters":
            mode_tokens.append("clusters")
            continue
        if arg_lower in ["group", "groups"]:
            mode_tokens.append("groups")
            continue
        label_match = re.fullmatch(r'#([^#]+)#', arg)
        if label_match:
            label_tokens.append(label_match.group(1))
            continue
        Command_Engine.print_help(
            viewer,
            f"Error: Unrecognized export target '{arg}'. Use clusters, groups, or #LABEL#.",
        )
        return

    if mode_tokens and label_tokens:
        Command_Engine.print_help(
            viewer,
            "Error: Export all-target modes cannot be combined with specific #LABEL# targets.",
        )
        return
    if len(mode_tokens) > 1:
        Command_Engine.print_help(
            viewer, "Error: Export accepts only one all-target mode: clusters or groups."
        )
        return
    if mode_tokens:
        target_mode = mode_tokens[0]
    elif label_tokens:
        target_mode = "specific"
        seen_targets = set()
        try:
            for label in label_tokens:
                resolved = Command_Engine.resolve_label_target(
                    getattr(viewer, 'cluster_labels', None),
                    getattr(viewer, 'group_labels', None),
                    label,
                )
                key = (
                    resolved.kind,
                    resolved.cluster_id
                    if resolved.kind in ("cluster", "noise")
                    else resolved.name.lower(),
                )
                if key not in seen_targets:
                    seen_targets.add(key)
                    specific_targets.append(resolved)
        except Command_Engine.SelectionExpressionError as error:
            Command_Engine.report_selection_error(
                viewer, " ".join(f"#{label}#" for label in label_tokens), error, "Export"
            )
            return

    # --- Validations ---
    if target_mode == "clusters" and getattr(viewer, 'cluster_labels', None) is None:
        viewer.console_text.text = "Error: Run 'cluster' first."
        print("Error: Run 'cluster' first to export clusters.")
        return
        
    if target_mode == "groups" and getattr(viewer, 'group_labels', None) is None:
        viewer.console_text.text = "Error: No groups defined."
        print("Error: No groups defined. Use the 'group' command first.")
        return

    # --- 2. Load Canonical Records Already Held by the Viewer ---
    source_records = _get_in_memory_sequence_records(viewer)
    if not source_records:
        msg = "Error: No in-memory sequence set is available for export."
        viewer.console_text.text = msg
        print(msg)
        return

    fasta_path = (
        getattr(cfg, 'NODE_FASTA_FILE', None)
        or getattr(cfg, 'SEQUENCES_FILE', None)
        or "loaded_sequences.fasta"
    )
    print(f"Using {len(source_records)} in-memory sanitized sequences...")

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
        
    uses_cluster_directory = target_mode == "clusters" or (
        target_mode == "specific"
        and specific_targets
        and all(target.kind in ("cluster", "noise") for target in specific_targets)
    )
    uses_group_directory = target_mode == "groups" or (
        target_mode == "specific"
        and specific_targets
        and all(target.kind == "group" for target in specific_targets)
    )

    if uses_cluster_directory:
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

    sequence_export_dir = cfg.resolve_directory_path(SEQUENCE_EXPORT_DIRECTORY)
    out_dir = os.path.join(sequence_export_dir, lvl1_name)
    
    if uses_group_directory:
        final_dir_name = f"{lvl2_name}_GROUPS" if lvl2_name else "GROUPS"
        out_dir = os.path.join(out_dir, final_dir_name)
    elif target_mode == "specific" and not uses_cluster_directory:
        final_dir_name = f"{lvl2_name}_LABELS" if lvl2_name else "LABELS"
        out_dir = os.path.join(out_dir, final_dir_name)
    else:
        out_dir = os.path.join(out_dir, lvl2_name) if lvl2_name else out_dir

    # --- 4. Group Sequences ---
    file_map = {}
    missing_count = 0
    
    print("Mapping sequences...")
    for i, full_header in enumerate(viewer.full_headers):
        if full_header not in source_records:
            missing_count += 1
            continue
            
        record = (full_header, source_records[full_header])
        
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
                
    if target_mode == "specific":
        file_map = {}
        for target in specific_targets:
            if target.kind == "cluster":
                filename = f"Cluster_{target.cluster_id}.fasta"
            elif target.kind == "noise":
                filename = "Noise.fasta"
            else:
                filename = f"{target.name}.fasta"
            target_mask = Command_Engine.evaluate_label_mask(
                viewer.full_headers,
                getattr(viewer, 'cluster_labels', None),
                getattr(viewer, 'group_labels', None),
                target.name,
            )
            records = [
                (header, source_records[header])
                for index, header in enumerate(viewer.full_headers)
                if target_mask[index] and header in source_records
            ]
            if records:
                file_map[filename] = records

    if missing_count > 0:
        print(f"Warning: {missing_count} viewer nodes were not found in the original FASTA file.")

    if not file_map:
        msg = "No valid subsets found to export."
        viewer.console_text.text = msg
        print(msg)
        return

    os.makedirs(out_dir, exist_ok=True)

    # --- 5. Write Files ---
    print(f"Exporting to: {out_dir}")
    files_written = 0
    seqs_written = 0
    
    for filename, recs in file_map.items():
        out_path = os.path.join(out_dir, filename)
        try:
            write_fasta_atomic(
                out_path,
                [header for header, _ in recs],
                [sequence for _, sequence in recs],
            )
            files_written += 1
            seqs_written += len(recs)
        except Exception as e:
            print(f"Failed to write {filename}: {e}")

    msg = f"Exported {files_written} files ({seqs_written} sequences)."
    viewer.console_text.text = msg
    print(f"\nSuccess! {msg}")
    
    # Auto-open the output folder in the system file manager
    open_in_file_manager(out_dir)
