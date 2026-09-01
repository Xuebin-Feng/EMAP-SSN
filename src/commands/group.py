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

import re
import numpy as np
import Command_Engine


_RESERVED_GROUP_NAMES = {
    "noise",
    "reset",
    "remove",
    "delete",
    "list",
    "help",
    "cluster",
    "group",
    "groups",
    "clusters",
}
_CANONICAL_CLUSTER_NAME_RE = re.compile(r'^cluster_(0|[1-9]\d*)$')
_GENERATED_SUBCLUSTER_NAME_RE = re.compile(
    r'^subcluster_[1-9]\d*_[1-9]\d*$'
)


def print_help():
    print("""
    Custom Group Labeling Tool
    ==========================
    Usage: group [EXPRESSION] <GROUP_NAME> [<EXPR_2> <NAME_2> ...]
           group list
           group remove <NAME_1> [<NAME_2> ...]
           group help

    Description:
      Assigns custom, searchable group labels to nodes. Unlike topology clusters, 
      a single node can belong to multiple groups simultaneously. Group names must 
      NOT contain spaces or special characters (use underscores).
      
      * QUICK USE: If no expression is provided, the command automatically applies
        the group name to the nodes currently selected in the viewer.

    Expression Targets (Do NOT use spaces inside expressions!):
      1. AA Position:  [AA][Pos] (e.g., P106, _100 for gap), or ([AA...])[Pos]
                       for alternatives (e.g., (RHK)71); negative positions require
                       parentheses (e.g., K(-1), (RHK)(-1))
      2. Header Text:  "[Text]"  (e.g., "3HMU", "*4A6T*")
      3. File Search:  @[File]@  (e.g., @my_list.txt@)
      4. NCBI/PDB:     @[NCBI][File]@ or @[PDB][File]@
      5. Labels:       #[Name]#  (e.g., #cluster_1#, #noise#)
      6. UI Selection: $sele$    (Explicitly targets selected nodes)
      7. Metadata:     {Key Op Val} (e.g., {Length>500}, {Organism=*coli*})

    Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. If any expression is invalid, no groups are applied.
      A valid expression may match zero nodes.
      Reserved group names are: noise, reset, remove, delete, list, help, cluster,
      group, groups, and clusters. A canonical cluster name is rejected only when
      that cluster currently exists (for example, cluster_1 while cluster 1 is
      loaded). Noncanonical names with leading zeros, such as cluster_001, are
      custom-group names. Canonical subcluster names remain reserved.

    Commands:
      list          - Prints current group statistics to the console.
      remove/delete - Deletes the specified group(s) entirely.

    Examples:
      group active_site                        (Assigns to currently selected nodes)
      group P106 mutant                        (Assigns "mutant" to P106 nodes)
      group (RHK)71 basic                      (Groups nodes with R, H, or K at pos 71)
      group "ATA"&#cluster_2# kinase           (Assigns "kinase" using boolean logic)
      group {Length<300} short_seqs            (Assigns "short_seqs" to nodes with Length < 300)
      group list                               (Prints all active groups)
      group remove active_site mutant          (Deletes both groups from the network)
    """)

def run(viewer, args):
    if args and args[0].lower() == 'reset':
        Command_Engine.execute_reset(viewer, ["groups"])
        return

    if not args or args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- LIST COMMAND ---
    if args[0].lower() == 'list':
        if getattr(viewer, 'group_labels', None) is None:
            msg = "No groups are currently defined."
            Command_Engine.print_help(viewer, msg)
            return
            
        group_counts = {}
        for g_set in viewer.group_labels:
            for g in g_set:
                group_counts[g] = group_counts.get(g, 0) + 1
                
        if not group_counts:
            msg = "No groups are currently defined."
            Command_Engine.print_help(viewer, msg)
            return
            
        n_nodes = viewer.n_nodes
        
        print(f"\n{'='*52}")
        print(f"--- Current Group Statistics (Total Nodes: {n_nodes}) ---")
        print(f"{'='*52}")
        print(f"| {'Group Name':<20} | {'Node Count':>10} | {'Percent':>10} |")
        print(f"|{'-'*22}+{'-'*12}+{'-'*12}|")
        
        # Sort by count descending (-x[1]), then alphabetically by name (x[0])
        sorted_groups = sorted(group_counts.items(), key=lambda x: (-x[1], x[0]))
        
        for g_name, c_count in sorted_groups:
            c_pct = (c_count / n_nodes) * 100
            # Truncate excessively long names to keep the table clean
            disp_name = g_name if len(g_name) <= 20 else g_name[:17] + "..."
            print(f"| {disp_name:<20} | {c_count:>10} | {c_pct:>9.2f}% |")
        print(f"{'='*52}\n")
        
        msg = f"Listed {len(sorted_groups)} groups in console."
        viewer.console_text.text = msg
        return

    # --- REMOVE / DELETE COMMAND ---
    if args[0].lower() in ['remove', 'delete']:
        if len(args) < 2:
            msg = "Error: Please specify one or more groups to remove (e.g., 'group remove group1 group2')."
            Command_Engine.print_help(viewer, msg)
            return
            
        if getattr(viewer, 'group_labels', None) is None:
            msg = "No groups are currently defined."
            Command_Engine.print_help(viewer, msg)
            return

        groups_to_remove = [g.lower() for g in args[1:]]
        total_removed = 0
        
        # Save state before deleting so the user can Undo
        viewer._save_state()
        
        for g_set in viewer.group_labels:
            for g_target in groups_to_remove:
                if g_target in g_set:
                    g_set.remove(g_target)
                    total_removed += 1
                    
        if total_removed > 0:
            viewer.update_nodes()
            msg = f"Removed {len(groups_to_remove)} group(s) from {total_removed} total node instances."
        else:
            msg = f"None of the specified groups were found."
            
        Command_Engine.print_help(viewer, msg)
        return

    # Handle missing selection logic (Default to $sele$)
    if len(args) == 1:
        args = ["$sele$", args[0]]
        
    # Handle other odd number of arguments
    elif len(args) % 2 != 0:
        msg = "Error: Arguments must be in pairs of [expression] [group_name]."
        Command_Engine.print_help(viewer, msg)
        return

    # Prepare mapping for boolean logic evaluations
    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    
    total_modified = 0
    stats = []
    warnings_issued = []

    current_group_labels = getattr(viewer, 'group_labels', None)
    if current_group_labels is None:
        staged_group_labels = [set() for _ in range(viewer.n_nodes)]
    else:
        staged_group_labels = [set(groups) for groups in current_group_labels]

    evaluated_pairs = []

    # Preflight every pair against staged labels. This preserves sequential
    # define-then-use behavior without mutating the viewer until all expressions
    # have been validated.
    for i in range(0, len(args), 2):
        expr = args[i]
        raw_name = args[i+1]
        
        # Validation checks
        name = raw_name.lower()
        if name in _RESERVED_GROUP_NAMES:
            msg = f"Group name '{raw_name}' is a reserved keyword. Skipping."
            print(f"Warning: {msg}")
            warnings_issued.append(msg)
            continue
        cluster_name_match = _CANONICAL_CLUSTER_NAME_RE.fullmatch(name)
        if (
            cluster_name_match
            and getattr(viewer, 'cluster_labels', None) is not None
            and np.any(
                np.asarray(viewer.cluster_labels)
                == int(cluster_name_match.group(1))
            )
        ):
            msg = f"Group name '{raw_name}' conflicts with an existing topology cluster. Skipping."
            print(f"Warning: {msg}")
            warnings_issued.append(msg)
            continue
        if _GENERATED_SUBCLUSTER_NAME_RE.fullmatch(name):
            msg = f"Group name '{raw_name}' is reserved for subclusters. Skipping."
            print(f"Warning: {msg}")
            warnings_issued.append(msg)
            continue
        if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
            msg = f"Group name '{raw_name}' contains invalid characters. Skipping."
            print(f"Warning: {msg}")
            warnings_issued.append(msg)
            continue
            
        try:
            mask = Command_Engine.parse_advanced_expression(
                expr,
                viewer_to_aln,
                valid_indices,
                viewer.full_headers,
                getattr(viewer, 'cluster_labels', None),
                staged_group_labels,
                getattr(viewer, 'alignment', None),
                metadata=getattr(viewer, 'metadata', None),
                selection_mask=Command_Engine.get_selected_mask(viewer),
            )
            count = int(np.sum(mask))

            if count > 0:
                for idx in np.where(mask)[0]:
                    staged_group_labels[idx].add(name)

            evaluated_pairs.append((name, mask, count))
        except Exception as e:
            Command_Engine.report_selection_error(viewer, expr, e, "Group")
            return

    if any(count > 0 for _, _, count in evaluated_pairs):
        viewer._save_state()
        if current_group_labels is None:
            viewer.group_labels = [set() for _ in range(viewer.n_nodes)]

        for name, mask, count in evaluated_pairs:
            if count <= 0:
                continue
            for idx in np.where(mask)[0]:
                viewer.group_labels[idx].add(name)
            total_modified += count
            stats.append(f"{count} nodes -> '{name}'")
            
    if total_modified > 0:
        viewer.update_nodes()
        msg = f"Groups Applied: {'; '.join(stats)}"
        if warnings_issued:
            msg += f" ({len(warnings_issued)} skipped)"
        viewer.console_text.text = msg
        print(f"\nSuccess! {msg}")
    elif warnings_issued:
        # If nothing was modified but we had warnings, show the first warning on the HUD
        viewer.console_text.text = f"Skipped: {warnings_issued[0]}"
        print(f"\nOperation skipped or aborted due to warnings.")
    else:
        viewer.console_text.text = "No nodes matched criteria for grouping."
        print("\nNo nodes matched your criteria.")
