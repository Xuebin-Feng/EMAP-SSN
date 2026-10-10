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
import numpy as np
import EMAPSSN_Config as cfg
from utilities.Localization import JoinedMessage, Message

def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        # The console line shows the first line; the terminal shows it all, in English.
        msg = JoinedMessage([
            Message("Usage: {syntax}", syntax="hide [EXPRESSION / single / free]"),
            "Description:\n"
            "  Without arguments: Immediately hides all currently selected nodes and their connected edges.\n"
            "  With 'single' or 'free': Hides all visible nodes that have no active edges at the current similarity threshold.\n"
            "  With EXPRESSION: Hides all visible nodes matching the logical expression.\n"
            "  The expression is one word: do not use spaces inside it.\n"
            "  'single', 'free' and 'reset' take no other arguments.\n\n"
            "Validation:\n"
            "  Referenced clusters, groups, alignment positions, metadata properties, and files must exist.\n"
            "  An invalid reference aborts without hiding nodes; a valid expression may match zero nodes.\n\n"
            "To unhide nodes, use the `reset hide` command.",
        ], separator="\n\n")
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # reset, single and free are whole commands: a word after one would be
    # ignored, so it is refused before anything changes.
    if len(args) > 1 and args[0].lower() in ['reset', 'single', 'free']:
        msg = Message(
            "Error: Unrecognized hide argument '{argument}'. '{keyword}' takes no other arguments.",
            argument=args[1], keyword=args[0],
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if args and args[0].lower() == 'reset':
        msg = Command_Engine.execute_reset(viewer, ["hidden"])
        Command_Engine.command_succeeded(viewer, msg)
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
            msg = Message("No single/free nodes found to hide at the current edge threshold.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_succeeded(viewer, msg)
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
        
        msg = Message("Hidden %n single/free node(s).", n=num_hidden)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    # If logic argument is given, parse it to find nodes to hide.
    # If not, default to currently selected nodes.
    if args:
        # The console splits a line at spaces, so a quoted name with a space,
        # "node 1", arrives as two words and would match no header. select
        # refuses such input; so does hide, before anything changes.
        if len(args) > 1:
            msg = Message("Error: Hide accepts exactly one whitespace-free Boolean expression.")
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return
        expr = args[0]

        viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
        
        try:
            mask = Command_Engine.parse_advanced_expression(
                expr, 
                viewer_to_aln, 
                valid_indices, 
                viewer.full_headers, 
                cluster_labels=getattr(viewer, 'cluster_labels', None), 
                group_labels=getattr(viewer, 'group_labels', None), 
                alignment=getattr(viewer, 'alignment', None), 
                metadata=getattr(viewer, 'metadata', None),
                selection_mask=Command_Engine.get_selected_mask(viewer),
            )
        except Exception as e:
            Command_Engine.report_selection_error(viewer, expr, e, Message("Hide"))
            return

        newly_hidden = mask & viewer.visible_mask
        num_hidden = int(np.sum(newly_hidden))

        # Nothing to hide is not an error, but it must not add an undo step.
        if num_hidden == 0:
            msg = Message("No visible nodes matched '{expression}' to hide.", expression=expr)
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_succeeded(viewer, msg)
            return

        viewer._save_state()
        viewer.visible_mask[newly_hidden] = False

        # Clean up selection if any selected nodes were hidden
        if hasattr(viewer, 'selected_indices'):
            viewer.selected_indices = [i for i in viewer.selected_indices if viewer.visible_mask[i]]
            
        viewer.hovered_node_idx = None
        viewer.selected_node_idx = None
        viewer.tooltip.text = ""
        
        viewer.update_selection_visual()
        viewer.update_edges()
        
        msg = Message("Hidden %n node(s) matching expression.", n=num_hidden)
        Command_Engine.print_help(viewer, msg)
        
    else:
        # Default to hiding selected nodes
        if not getattr(viewer, 'selected_indices', []):
            msg = Message("Error: No nodes currently selected.")
            Command_Engine.command_failed(viewer, msg)
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
        
        msg = Message("Hidden %n selected node(s).", n=num_hidden)
        Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)
