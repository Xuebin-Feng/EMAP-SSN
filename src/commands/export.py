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
from desktop.Desktop_App import open_in_file_manager
from utilities.Localization import Message
from utilities.Output_Names import validate_output_basename
from utilities.Sequence_Utils import write_fasta_atomic

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


def _refused_output_name(viewer, name, source):
    """Report `source` and return True unless `name` is a plain file name.

    Group labels and clustering parameters become export file and folder
    names. The group and cluster commands never produce a path, but a layout
    cache restores whatever labels and parameters it carries, and os.path.join
    drops the export folder for an absolute name or climbs out of it through
    '..'. The name checked is the one written, so a group named '..' still
    exports as '...fasta'. `source` is a Message, such as "group label 'x'".
    """
    try:
        validate_output_basename(name)
    except ValueError as error:
        msg = Message("Error: Export refused {source}: {error}", source=source, error=error)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return True
    return False

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
      Hidden nodes are included in the exported subsets.
      Group labels become file names and clustering parameters a folder name; if one
      would be a path (as a hand-edited layout cache can carry), the export is
      refused and nothing is written.
      
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
    if args and args[0].lower() in ['help', '-h', '-?', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # --- 1. Parse Arguments ---
    target_mode = "clusters"
    specific_targets = []
    mode_tokens = []
    label_tokens = []

    for arg in args:
        arg_lower = arg.lower()
        if arg_lower.startswith("group:"):
            msg = Message("Error: Legacy export group:NAME syntax is no longer supported. Use export #NAME#.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
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
        msg = Message("Error: Unrecognized export target '{target}'. Use clusters, groups, or #LABEL#.", target=arg)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    if mode_tokens and label_tokens:
        msg = Message("Error: Export all-target modes cannot be combined with specific #LABEL# targets.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return
    if len(mode_tokens) > 1:
        msg = Message("Error: Export accepts only one all-target mode: clusters or groups.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
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
                viewer, " ".join(f"#{label}#" for label in label_tokens), error, Message("Export")
            )
            return

    # --- Validations ---
    if target_mode == "clusters" and getattr(viewer, 'cluster_labels', None) is None:
        msg = Message("Error: Run 'cluster' first.")
        Command_Engine.show_status(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        print("Error: Run 'cluster' first to export clusters.")
        Command_Engine.command_failed(viewer, "Error: Run 'cluster' first to export clusters.")
        return
        
    if target_mode == "groups" and not any(getattr(viewer, 'group_labels', None) or ()):
        msg = Message("Error: No groups defined.")
        Command_Engine.show_status(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        print("Error: No groups defined. Use the 'group' command first.")
        Command_Engine.command_failed(viewer, "Error: No groups defined. Use the 'group' command first.")
        return

    # --- 2. Load Canonical Records Already Held by the Viewer ---
    source_records = _get_in_memory_sequence_records(viewer)
    if not source_records:
        msg = Message("Error: No in-memory sequence set is available for export.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
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
            if _refused_output_name(
                viewer, lvl2_name,
                Message("clustering parameters ({mode}, {minimum})", mode=c_mode_param, minimum=c_min_param),
            ):
                return
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
                if file_name not in file_map:
                    if _refused_output_name(viewer, file_name, Message("group label '{group}'", group=g_name)):
                        return
                    file_map[file_name] = []
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
                if _refused_output_name(viewer, filename, Message("group label '{group}'", group=target.name)):
                    return
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
        msg = Message("No valid subsets found to export.")
        Command_Engine.show_status(viewer, msg)
        print(msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    # Names that differ only in letter case are one file on Windows, so one subset
    # would overwrite the other. Refuse before any folder or file is created.
    first_name_by_lower = {}
    for file_name in (file_map if os.name == "nt" else ()):
        first_name = first_name_by_lower.setdefault(file_name.lower(), file_name)
        if first_name != file_name:
            msg = Message(
                "Error: Export refused: {first} and {second} differ only in letter case, "
                "so Windows would write them to one file.",
                first=first_name,
                second=file_name,
            )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

    os.makedirs(out_dir, exist_ok=True)

    # --- 5. Write Files ---
    print(f"Exporting to: {out_dir}")
    files_written = 0
    seqs_written = 0
    files_failed = 0

    for filename, recs in file_map.items():
        out_path = os.path.join(out_dir, filename)
        try:
            write_fasta_atomic(
                out_path,
                [header for header, _ in recs],
                [sequence for _, sequence in recs],
            )
            Command_Engine.command_artifact(viewer, out_path)
            files_written += 1
            seqs_written += len(recs)
        except Exception as e:
            files_failed += 1
            print(f"Failed to write {filename}: {e}")
            Command_Engine.command_failed(
                viewer, Message("Failed to write {file}: {error}", file=filename, error=e)
            )

    if files_failed:
        # Some files are missing, so report an error rather than the export's success.
        msg = Message(
            "Error: Failed to write %n file(s); {written} written.",
            n=files_failed,
            written=Message("%n file(s)", n=files_written),
        )
        Command_Engine.show_status(viewer, msg)
        print(f"\n{msg}")
        if files_written:
            open_in_file_manager(out_dir)
        Command_Engine.command_failed(viewer, msg)
        return

    msg = Message(
        "Exported %n file(s) ({sequences}).",
        n=files_written,
        sequences=Message("%n sequence(s)", n=seqs_written),
    )
    Command_Engine.show_status(viewer, msg)
    print(f"\nSuccess! {msg}")
    
    # Auto-open the output folder in the system file manager
    open_in_file_manager(out_dir)
    Command_Engine.command_succeeded(viewer, msg)
