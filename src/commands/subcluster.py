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
import numpy as np
import re
import os
import SSN_Utils as utils
import sys
import colorsys
import math

if sys.platform == 'win32':
    os.system('')

def get_colored_subcluster_name(sub_id, name_str, color_map=None):
    if sub_id == -1:
        # Grey for noise
        r, g, b = 204, 204, 204
    elif color_map and sub_id in color_map:
        rgba = color_map[sub_id]
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    else:
        r, g, b = 180, 180, 180
    return f"\033[38;2;{r};{g};{b}m●\033[0m {name_str}"

def get_subcluster_colors(n_subclusters):
    if n_subclusters <= 0:
        return []
    shades_per_hue = math.ceil(n_subclusters / 12)
    n_hues = math.ceil(n_subclusters / shades_per_hue)
    colors = []
    for s_idx in range(shades_per_hue):
        for h_idx in range(n_hues):
            hue = h_idx / n_hues
            if shades_per_hue == 1:
                lightness = 0.50  # Strong/bold tone
            else:
                lightness = 0.35 + (s_idx / (shades_per_hue - 1)) * 0.30  # Bold tones (0.35 to 0.65)
            saturation = 0.95  # Highly saturated
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(rgb)
    return colors

def print_help():
    print("""
    Cluster Subclustering Tool
    ==========================
    Usage: subcluster <CLUSTER_NAME> [MODE] [PARAM_1] [MIN_SIZE]
           subcluster clear
           subcluster help

    Description:
      Performs subclustering on a specific topology cluster, creating custom group labels 
      named 'subcluster_N_M' (where N is the original cluster ID, and M is the subcluster ID).
      Unlike main clusters, these are saved as custom group labels so nodes can keep their 
      original cluster identities.

    Arguments:
      <CLUSTER_NAME>    - Name of the cluster to subcluster (e.g., cluster_2, cluster_5).
      clear             - Clears all subcluster groups (subcluster_N_M) from the viewer session.

    Modes:
      leiden (Default)  - Leiden Community Detection. PARAM_1: Resolution (Default: 1.0)
      mcl               - Markov Clustering Algorithm. PARAM_1: Inflation (Default: 2.0)
      jaccard           - Topology Jaccard filtering. PARAM_1: Threshold (Default: 0.2)

    [MIN_SIZE]          - (Optional) Minimum size of subclusters to keep (Default: 10).
                          Smaller groups are treated as Noise.

    Examples:
      subcluster cluster_2
      subcluster cluster_2 mcl 2.0 5
      subcluster cluster_5 leiden 1.5
      subcluster clear
    """)

def run(viewer, args):
    # --- 1. Help Check ---
    if not args or args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # --- CLEAR COMMAND ---
    if args[0].lower() == 'clear':
        if not hasattr(viewer, 'group_labels') or viewer.group_labels is None:
            Command_Engine.print_help(viewer, "No groups are currently defined.")
            return
            
        viewer._save_state()
        
        pattern = re.compile(r'^subcluster_\d+_\d+$')
        total_removed = 0
        for g_set in viewer.group_labels:
            to_remove = [g for g in g_set if pattern.match(g)]
            for g in to_remove:
                g_set.remove(g)
                total_removed += 1
                
        viewer.update_nodes()
        
        msg = f"Cleared all subcluster groups (removed {total_removed} label instances)."
        Command_Engine.print_help(viewer, msg)
        return

    # --- Parse Cluster Name ---
    match = re.match(r'^cluster_(\d+)$', args[0].lower())
    if not match:
        print_help()
        Command_Engine.print_help(viewer, f"Error: First argument must be 'clear' or a cluster name like 'cluster_N' (got '{args[0]}').")
        return
        
    cluster_id = int(match.group(1))

    if getattr(viewer, 'cluster_labels', None) is None:
        Command_Engine.print_help(viewer, "Error: No clusters are currently defined. Run 'cluster' first.")
        return

    target_mask = (viewer.cluster_labels == cluster_id)
    subgraph_nodes = np.where(target_mask)[0]

    if len(subgraph_nodes) == 0:
        Command_Engine.print_help(viewer, f"Error: Cluster {cluster_id} is empty or does not exist.")
        return

    # --- 2. Parse Other Parameters ---
    sub_args = args[1:]
    mode = "leiden"
    param1 = None
    min_sz = 10
    
    if len(sub_args) >= 1:
        first_arg = sub_args[0].lower()
        if first_arg in ['jaccard', 'mcl', 'leiden']:
            mode = first_arg
            if len(sub_args) >= 2: 
                try: param1 = float(sub_args[1])
                except ValueError: print("Error: Parameter must be a number."); return
            if len(sub_args) >= 3: 
                try: min_sz = int(sub_args[2])
                except ValueError: print("Error: Min Size must be an integer."); return
        else:
            # Fallback to default Jaccard logic if first argument is a number
            try: param1 = float(sub_args[0])
            except ValueError:
                print(f"Error: Unknown mode or invalid number '{sub_args[0]}'")
                return
            if len(sub_args) >= 2: 
                try: min_sz = int(sub_args[1])
                except ValueError: print("Error: Min Size must be an integer."); return

    # Apply defaults if param1 wasn't provided
    if param1 is None:
        if mode == "jaccard": param1 = 0.2
        elif mode == "mcl": param1 = 2.0
        elif mode == "leiden": param1 = 1.0

    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = f"Subclustering cluster_{cluster_id} ({mode.upper()})..."
    print(f"Running {mode.upper()} Subclustering for cluster_{cluster_id} (Param={param1}, MinSize={min_sz})...")

    # --- 3. Extract Subgraph Edges ---
    edges = np.array(viewer.edges, dtype=np.int32)
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(subgraph_nodes)}
    local_to_global = {l_idx: g_idx for l_idx, g_idx in enumerate(subgraph_nodes)}
    
    subgraph_edges = []
    subgraph_edge_scores = []
    for e_idx, edge in enumerate(edges):
        u, v = edge
        if target_mask[u] and target_mask[v]:
            subgraph_edges.append(edge)
            if hasattr(viewer, 'edge_scores'):
                subgraph_edge_scores.append(viewer.edge_scores[e_idx])

    if len(subgraph_edges) == 0:
        Command_Engine.print_help(viewer, f"Error: No edges exist within cluster_{cluster_id} to perform subclustering.")
        return

    local_edges = np.array([[global_to_local[u], global_to_local[v]] for u, v in subgraph_edges], dtype=np.int32)
    local_edge_scores = np.array(subgraph_edge_scores, dtype=np.float64) if subgraph_edge_scores else None

    n_sub = len(subgraph_nodes)
    local_labels = np.full(n_sub, -1, dtype=int)

    # =======================================================
    # MODE 1: JACCARD (Topology Filtering + BFS)
    # =======================================================
    if mode == "jaccard":
        thresh = param1
        if not utils.NUMBA_AVAILABLE:
            print("Error: Numba required for topology clustering.")
            viewer.console_text.text = "Error: Numba library missing."
            return

        # Prepare adjacency data for Numba
        degrees = np.zeros(n_sub, dtype=np.int32)
        for u, v in local_edges:
            degrees[u] += 1; degrees[v] += 1
            
        indptr = np.zeros(n_sub + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(degrees)
        indices = np.zeros(indptr[-1], dtype=np.int32)
        
        temp_counts = np.zeros(n_sub, dtype=np.int32)
        for u, v in local_edges:
            indices[indptr[u] + temp_counts[u]] = v; temp_counts[u] += 1
            indices[indptr[v] + temp_counts[v]] = u; temp_counts[v] += 1
            
        for i in range(n_sub): 
            indices[indptr[i]:indptr[i+1]].sort()

        # Numba Filter
        keep_mask = utils.fast_jaccard_filter(local_edges, indptr, indices, thresh)
        
        filtered_adj = {i: [] for i in range(n_sub)}
        for i, keep in enumerate(keep_mask):
            if keep:
                u, v = local_edges[i]
                filtered_adj[u].append(v)
                filtered_adj[v].append(u)

        # BFS Connected Components
        visited = np.zeros(n_sub, dtype=bool)
        sub_id = 1
        
        for i in range(n_sub):
            if not visited[i]:
                stack = [i]
                visited[i] = True
                component = []
                
                while stack:
                    node = stack.pop()
                    component.append(node)
                    for neighbor in filtered_adj[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                
                if len(component) >= min_sz:
                    for node in component: 
                        local_labels[node] = sub_id
                    sub_id += 1

    # =======================================================
    # MODE 2: MARKOV CLUSTERING (MCL)
    # =======================================================
    elif mode == "mcl":
        inflation = param1
        try:
            import markov_clustering as mc
            import scipy.sparse as sp
        except ImportError:
            msg = "Missing libraries! Run: pip install markov_clustering networkx scipy"
            print(f"Error: {msg}")
            viewer.console_text.text = msg
            return
            
        print("Building Sparse Adjacency Matrix...")
        row = np.concatenate([local_edges[:, 0], local_edges[:, 1]])
        col = np.concatenate([local_edges[:, 1], local_edges[:, 0]])
        
        if local_edge_scores is not None:
            d_vals = np.concatenate([local_edge_scores, local_edge_scores])
        else:
            d_vals = np.ones(len(row))
            
        matrix = sp.csr_matrix((d_vals, (row, col)), shape=(n_sub, n_sub))
        
        import warnings
        from scipy.sparse import SparseEfficiencyWarning
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
        
        print(f"Running MCL (Inflation = {inflation}). This may take a moment...")
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        
        sub_id = 1
        for comp in clusters:
            if len(comp) >= min_sz:
                for node in comp:
                    local_labels[node] = sub_id
                sub_id += 1

    # =======================================================
    # MODE 3: LEIDEN COMMUNITY DETECTION
    # =======================================================
    elif mode == "leiden":
        resolution = param1
        try:
            import graspologic_native  # noqa: F401  (availability check)
        except ImportError:
            msg = "Missing library! Run: pip install graspologic-native"
            print(f"Error: {msg}")
            viewer.console_text.text = msg
            return

        print("Building Edge List & Mapping Edge Weights...")

        if local_edge_scores is not None:
            print(f"Running Leiden (Resolution = {resolution}, Weighted).")
        else:
            print(f"Running Leiden (Resolution = {resolution}, Unweighted).")

        # Nodes with no edges are always returned as Noise (-1), even if
        # min_sz is 1. See SSN_Utils.leiden_partition for the guarantee.
        local_labels = utils.leiden_partition(
            n_sub, local_edges, local_edge_scores, resolution, min_sz, seed=42
        )

    # --- 4. Update Viewer State ---
    viewer._save_state()

    # Ensure group_labels is initialized
    if not hasattr(viewer, 'group_labels') or viewer.group_labels is None:
        viewer.group_labels = [set() for _ in range(viewer.n_nodes)]

    # Remove existing groups matching subcluster_N_* for all nodes
    pattern = re.compile(rf'^subcluster_{cluster_id}_\d+$')
    for g_set in viewer.group_labels:
        to_remove = [g for g in g_set if pattern.match(g)]
        for g in to_remove:
            g_set.remove(g)

    # Assign new ones and calculate counts
    sub_counts = {}
    for l_idx, m in enumerate(local_labels):
        if m != -1:
            g_name = f"subcluster_{cluster_id}_{m}"
            global_idx = local_to_global[l_idx]
            viewer.group_labels[global_idx].add(g_name)
            sub_counts[m] = sub_counts.get(m, 0) + 1

    # Compute color map for subclusters
    sorted_subs = sorted(sub_counts.keys())
    n_subclusters = len(sorted_subs)
    if n_subclusters > 0:
        sub_colors = get_subcluster_colors(n_subclusters)
        color_map = {sid: sub_colors[idx % len(sub_colors)] for idx, sid in enumerate(sorted_subs)}
    else:
        color_map = {}

    # Assign colors to viewer.current_colors for nodes in the subclustered target
    for l_idx, m in enumerate(local_labels):
        global_idx = local_to_global[l_idx]
        if m == -1:
            viewer.current_colors[global_idx] = (0.8, 0.8, 0.8, 0.4)  # Grey for noise
        else:
            r, g, b = color_map[m]
            viewer.current_colors[global_idx] = (r, g, b, 1.0)

    viewer.update_nodes()

    # --- 5. Print Statistics ---
    print(f"\n{'='*54}")
    print(f"--- {mode.upper()} Subclustering Stats for Cluster {cluster_id} (Total Nodes: {n_sub}) ---")
    print(f"{'='*54}")
    print(f"| {'Subcluster Name':<22} | {'Node Count':>10} | {'Percent':>10} |")
    print(f"|{'-'*24}+{'-'*12}+{'-'*12}|")
    
    noise_count = np.sum(local_labels == -1)
    noise_pct = (noise_count / n_sub) * 100
    
    noise_name_padded = get_colored_subcluster_name(-1, f"{'Noise (Unclustered)':<20}", color_map)
    print(f"| {noise_name_padded} | {noise_count:>10} | {noise_pct:>9.2f}% |")
    
    for m in sorted_subs:
        count = sub_counts[m]
        pct = (count / n_sub) * 100
        sub_name_padded = get_colored_subcluster_name(m, f"{f'subcluster_{cluster_id}_{m}':<20}", color_map)
        print(f"| {sub_name_padded} | {count:>10} | {pct:>9.2f}% |")
    print(f"{'='*54}\n")

    msg = f"Done! Found {n_subclusters} subclusters in cluster_{cluster_id} via {mode.upper()}."
    if hasattr(viewer, 'console_text'):
        viewer.console_text.text = msg
    print(msg)

