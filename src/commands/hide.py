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
import Command_Engine
import numpy as np
import SSN_Config as cfg

def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        msg = ("Usage: hide [EXPRESSION / single / free]\n\n"
               "Description:\n"
               "  Without arguments: Immediately hides all currently selected nodes and their connected edges.\n"
               "  With 'single' or 'free': Hides all visible nodes that have no active edges at the current similarity threshold.\n"
               "  With EXPRESSION: Hides all visible nodes matching the logical expression.\n\n"
               "To unhide nodes, use the `reset hide` command.")
        Command_Engine.print_help(viewer, msg)
        return

    if args and args[0].lower() == 'reset':
        Command_Engine.execute_reset(viewer, ["hidden"])
        return

    if args and args[0].lower() in ['single', 'free']:
        current_slider_val = getattr(viewer, 'current_slider_threshold', getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0))
        
        # Calculate which edges are active (visible endpoints and score >= threshold)
        nodes_visible_mask = viewer.visible_mask[viewer.edges[:, 0]] & viewer.visible_mask[viewer.edges[:, 1]]
        if hasattr(viewer, 'edge_scores') and len(viewer.edge_scores) > 0:
            threshold_visible_mask = viewer.edge_scores >= current_slider_val
            valid_edges_mask = nodes_visible_mask & threshold_visible_mask
        else:
            valid_edges_mask = nodes_visible_mask
            
        active_edges = viewer.edges[valid_edges_mask]
        
        # Find which nodes are endpoints of active edges
        has_active_edges = np.zeros(viewer.n_nodes, dtype=bool)
        if len(active_edges) > 0:
            has_active_edges[active_edges[:, 0]] = True
            has_active_edges[active_edges[:, 1]] = True
            
        # Select currently visible nodes that have no active edges
        single_nodes_mask = viewer.visible_mask & ~has_active_edges
        num_hidden = np.sum(single_nodes_mask)
        
        if num_hidden == 0:
            msg = "No single/free nodes found to hide at the current edge threshold."
            Command_Engine.print_help(viewer, msg)
            return
            
        viewer._save_state()
        viewer.visible_mask[single_nodes_mask] = False
        
        # Clean up selection if any selected nodes were hidden
        if hasattr(viewer, 'selected_indices'):
            viewer.selected_indices = [i for i in viewer.selected_indices if viewer.visible_mask[i]]
            
        viewer.hovered_node_idx = None
        viewer.selected_node_idx = None
        viewer.tooltip.text = ""
        
        viewer.update_selection_visual()
        viewer.update_edges()
        
        msg = f"Hidden {num_hidden} single/free nodes."
        Command_Engine.print_help(viewer, msg)
        return

    # If logic argument is given, parse it to find nodes to hide.
    # If not, default to currently selected nodes.
    if args:
        expr = " ".join(args)
        
        viewer_to_aln = np.full(len(viewer.full_headers), -1, dtype=int)
        if (getattr(viewer, 'alignment', None).aln if getattr(viewer, 'alignment', None) else None) is not None:
            for i, h in enumerate(viewer.full_headers):
                if h in viewer.alignment.seq_map:
                    viewer_to_aln[i] = viewer.alignment.seq_map[h]
        valid_indices = np.where(viewer_to_aln != -1)[0]
        
        # Update _sele.txt for $sele$ references
        if "$sele$" in expr.lower():
            header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
            os.makedirs(header_dir, exist_ok=True)
            sele_path = os.path.join(header_dir, "_sele.txt")
            
            if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
                with open(sele_path, "w", encoding="utf-8", newline="\n") as f:
                    for idx in viewer.selected_indices:
                        f.write(viewer.full_headers[idx] + "\n")
            else:
                if os.path.exists(sele_path):
                    open(sele_path, "w", encoding="utf-8").close()
            
            expr = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)

        expr = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr)
        
        try:
            mask = Command_Engine.parse_advanced_expression(
                expr, 
                viewer_to_aln, 
                valid_indices, 
                viewer.full_headers, 
                cluster_labels=getattr(viewer, 'cluster_labels', None), 
                group_labels=getattr(viewer, 'group_labels', None), 
                alignment=getattr(viewer, 'alignment', None), 
                metadata=getattr(viewer, 'metadata', None)
            )
        except Exception as e:
            msg = f"Error processing expression '{expr}': {e}"
            Command_Engine.print_help(viewer, msg)
            return

        previous_visible = viewer.visible_mask.copy()
        viewer._save_state()
        viewer.visible_mask[mask] = False
        num_hidden = np.sum(previous_visible & ~viewer.visible_mask)
        
        if num_hidden == 0:
            msg = f"No visible nodes matched '{expr}' to hide."
            Command_Engine.print_help(viewer, msg)
            return
            
        # Clean up selection if any selected nodes were hidden
        if hasattr(viewer, 'selected_indices'):
            viewer.selected_indices = [i for i in viewer.selected_indices if viewer.visible_mask[i]]
            
        viewer.hovered_node_idx = None
        viewer.selected_node_idx = None
        viewer.tooltip.text = ""
        
        viewer.update_selection_visual()
        viewer.update_edges()
        
        msg = f"Hidden {num_hidden} nodes matching expression."
        Command_Engine.print_help(viewer, msg)
        
    else:
        # Default to hiding selected nodes
        if not getattr(viewer, 'selected_indices', []):
            msg = "Error: No nodes currently selected."
            Command_Engine.print_help(viewer, msg)
            return
        
        viewer._save_state()
        viewer.visible_mask[viewer.selected_indices] = False
        num_hidden = len(viewer.selected_indices)
        viewer.selected_indices = []
        
        viewer.hovered_node_idx = None
        viewer.selected_node_idx = None
        viewer.tooltip.text = ""
        
        viewer.update_selection_visual()
        viewer.update_edges()
        
        msg = f"Hidden {num_hidden} selected nodes."
        Command_Engine.print_help(viewer, msg)

