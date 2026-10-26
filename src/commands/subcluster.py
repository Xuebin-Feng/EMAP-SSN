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

from dataclasses import dataclass

import Command_Engine
import numpy as np
import re
import os
from utilities import Network_Kernels as network_clustering
from utilities.Localization import JoinedMessage, Message
from utilities.Terminal_Colors import colours_enabled
import sys
import colorsys
import math
try:
    import commands.group as group_cmd
except ImportError:
    import group as group_cmd
try:
    import commands.cluster as cluster_cmd
except ImportError:
    import cluster as cluster_cmd

if sys.platform == 'win32' and not globals().get("_WINDOWS_ANSI_ENABLED", False):
    os.system('')
    _WINDOWS_ANSI_ENABLED = True


def get_colored_subcluster_name(sub_id, name_str, color_map=None):
    if sub_id == -1:
        # Grey for noise
        r, g, b = 204, 204, 204
    elif color_map and sub_id in color_map:
        rgba = color_map[sub_id]
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    else:
        r, g, b = 180, 180, 180
    if not colours_enabled():
        return f"● {name_str}"
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

def extra_argument_error(argument, syntax):
    """Return the error Message for an argument the command does not take, with its usage."""
    return JoinedMessage([
        Message("Error: Unrecognized subcluster argument '{argument}'.", argument=argument),
        Message("Usage: {syntax}", syntax=syntax),
    ], separator="\n")


def cluster_name_digits(argument):
    """The digits N of a cluster written #cluster_N# or cluster_N, or None for anything else.

    The label is read as an expression reads it (Command_Engine.resolve_label_target):
    in any case, with any space inside the # trimmed. Anything else is None, because
    the command works on one cluster: the groups it makes, subcluster_N_M, and the
    rule that a new cluster run removes them go by that cluster's N. So another
    label (#noise#, #my_group#), an expression (#cluster_2#|#cluster_3#, $sele$)
    and a name with one # missing (#cluster_2, cluster_2#) are all None.
    """
    delimited = re.fullmatch(r'#([^#]*)#', argument)
    name = (delimited.group(1) if delimited else argument).strip().lower()
    match = re.fullmatch(r'cluster_(\d+)', name)
    return match.group(1) if match else None


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
      original cluster identities. Nodes in the target cluster are recolored by subcluster;
      nodes below MIN_SIZE are gray. 'subcluster clear' removes labels but leaves colors as-is.
      Subclusters are numbered by size, so subcluster_N_1 is the largest; equal sizes are
      ordered by their lowest member node index.

    Arguments:
      <CLUSTER_NAME>    - The one cluster to subcluster, written as in an expression
                          (e.g., #cluster_2#, #cluster_5#); the bare cluster_2 is accepted
                          too and means the same. Without leading zeros: cluster_01 is
                          refused, in either form. Another label (#noise#, #my_group#) or
                          an expression (#cluster_2#|#cluster_3#, $sele$) is refused.
      clear             - Clears the generated subcluster groups (subcluster_N_M) from the
                          viewer session. Custom groups with lookalike names, such as
                          subcluster_0_2 or subcluster_001_2, are kept.

    Modes:
      leiden (Default)  - Leiden Community Detection. PARAM_1: Resolution, a finite number above 0 (Default: 1.0)
      mcl               - Markov Clustering Algorithm. PARAM_1: Inflation, 1.1 - 10.0 (Default: 2.0)
      jaccard           - Topology Jaccard filtering. PARAM_1: Threshold, 0.0 - 1.0 (Default: 0.2)
                          Uses Numba when installed, and a slower pure-Python fallback otherwise.

    [MIN_SIZE]          - (Optional) Minimum size of subclusters to keep, at least 1 (Default: 10).
                          Smaller groups are treated as Noise. When no subcluster reaches
                          MIN_SIZE, nothing changes: the cluster keeps its colours and
                          its earlier subclusters, and no undo step is added.

    Anything after the arguments above (or after clear) is refused with an error that
    names the first extra argument, and nothing changes.

    What is clustered:
      - Subclustering uses every loaded edge between the cluster's own nodes,
        regardless of the similarity slider and hidden nodes.
      - Jaccard compares closed neighbourhoods (a node counts as its own
        neighbour). An edge whose endpoints have a and b neighbours, c of them
        shared, scores (c + 2) / (a + b - c), so an edge outside every triangle
        still scores above 0, while an isolated pair and every edge of a clique
        score 1.
      - Isolated nodes (no edge within the cluster) are alike in all three modes:
        singleton subclusters when MIN_SIZE is 1, Noise when it is larger.
      - MCL gives every node a self-loop as heavy as its strongest edge within
        the cluster (1 for a node with no edge), so multiplying every score by a
        constant does not change the subclusters.
      - MCL needs every edge score within the cluster to be finite and above 0.

    Examples:
      subcluster #cluster_2#
      subcluster #cluster_2# mcl 2.0 5
      subcluster cluster_5 leiden 1.5    (the bare form of #cluster_5#)
      subcluster clear
    """)

# --- The command, in the steps the Viewer's command queue runs (Command_Queue) ---
# plan() reads the arguments when the command is given; start() checks, at its
# turn, that the cluster exists (an earlier cluster command may make it) and
# copies the clusters; compute() subclusters, on the queue's worker thread;
# apply() writes the groups and colours and saves the undo state, on the GUI
# thread. run() does all four at once, as the VR viewer and the tests run it.


@dataclass(frozen=True)
class SubclusterPlan:
    cluster_id: int
    mode: str
    param1: float
    min_size: int
    heavy = True


@dataclass(frozen=True)
class SubclusterClearPlan:
    """subcluster clear: it clears the groups at its turn, after the commands given before it."""
    heavy = False


@dataclass(frozen=True)
class SubclusterInputs:
    plan: SubclusterPlan
    cluster_labels: object
    edges: object
    scores: object


@dataclass(frozen=True)
class SubclusterResult:
    refusal: object = None
    nothing_reached: bool = False
    subgraph_nodes: object = None
    local_labels: object = None
    sub_counts: dict = None
    color_map: dict = None


def plan(viewer, args):
    """Read the arguments: a SubclusterPlan or SubclusterClearPlan, or None when help ran or they were refused."""
    # --- 1. Help Check ---
    if not args or args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return None

    # --- CLEAR COMMAND ---
    if args[0].lower() == 'clear':
        if len(args) > 1:
            refusal = extra_argument_error(args[1], "subcluster clear")
            Command_Engine.print_help(viewer, refusal, report_message=False)
            Command_Engine.command_failed(viewer, refusal)
            return None
        return SubclusterClearPlan()

    # --- Parse Cluster Name ---
    digits = cluster_name_digits(args[0])
    if digits is None:
        print_help()
        msg = Message(
            "Error: First argument must be 'clear' or exactly one cluster, "
            "written cluster_N or #cluster_N# (got '{argument}').",
            argument=args[0],
        )
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_failed(viewer, msg)
        return None

    # Expressions know #cluster_N# only without leading zeros, as group.py's
    # canonical cluster names are, so cluster_01 names a cluster no expression
    # can select. Refused before anything changes.
    if len(digits) > 1 and digits.startswith('0'):
        msg = Message(
            "Error: Write the cluster as {canonical}, without leading zeros.",
            canonical=f"cluster_{digits.lstrip('0') or '0'}",
        )
        Command_Engine.print_help(viewer, msg, report_message=False)
        Command_Engine.command_failed(viewer, msg)
        return None

    cluster_id = int(digits)

    # --- 2. Parse Other Parameters ---
    sub_args = args[1:]
    mode = "leiden"
    param1 = None
    min_sz = 10
    # [MODE] PARAM_1 MIN_SIZE take three arguments; a bare PARAM_1 MIN_SIZE two.
    taken = 3

    if len(sub_args) >= 1:
        first_arg = sub_args[0].lower()
        if first_arg in ['jaccard', 'mcl', 'leiden']:
            mode = first_arg
            if len(sub_args) >= 2:
                try: param1 = float(sub_args[1])
                except ValueError:
                    print("Error: Parameter must be a number.")
                    Command_Engine.command_failed(viewer, "Error: Parameter must be a number.")
                    return None
            if len(sub_args) >= 3:
                try: min_sz = int(sub_args[2])
                except ValueError:
                    print("Error: Min Size must be an integer.")
                    Command_Engine.command_failed(viewer, "Error: Min Size must be an integer.")
                    return None
        else:
            taken = 2
            # A bare number is the resolution of the default mode, Leiden.
            try: param1 = float(sub_args[0])
            except ValueError:
                print(f"Error: Unknown mode or invalid number '{sub_args[0]}'")
                Command_Engine.command_failed(viewer, f"Error: Unknown mode or invalid number '{sub_args[0]}'")
                return None
            if len(sub_args) >= 2:
                try: min_sz = int(sub_args[1])
                except ValueError:
                    print("Error: Min Size must be an integer.")
                    Command_Engine.command_failed(viewer, "Error: Min Size must be an integer.")
                    return None

    # Nothing may follow the arguments the form takes.
    if len(sub_args) > taken:
        refusal = extra_argument_error(
            sub_args[taken], "subcluster <CLUSTER_NAME> [MODE] [PARAM_1] [MIN_SIZE]"
        )
        Command_Engine.print_help(viewer, refusal, report_message=False)
        Command_Engine.command_failed(viewer, refusal)
        return None

    # Apply defaults if param1 wasn't provided
    if param1 is None:
        if mode == "jaccard": param1 = 0.2
        elif mode == "mcl": param1 = 2.0
        elif mode == "leiden": param1 = 1.0

    refusal = cluster_cmd.parameter_error(mode, param1, min_sz)
    if refusal:
        Command_Engine.print_help(viewer, refusal, report_message=False)
        Command_Engine.command_failed(viewer, refusal)
        return None
    return SubclusterPlan(cluster_id, mode, param1, min_sz)


def clear_subclusters(viewer):
    """subcluster clear: remove the generated subcluster groups, as one undo step."""
    if not hasattr(viewer, 'group_labels') or viewer.group_labels is None:
        msg = Message("No groups are currently defined.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    generated = [
        [g for g in g_set if group_cmd.is_generated_subcluster_name(g)]
        for g_set in viewer.group_labels
    ]
    total_removed = sum(len(names) for names in generated)

    # Nothing to clear is not an error, but it must not add an undo step.
    if total_removed == 0:
        msg = Message("No subcluster groups to clear.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    viewer._save_state()

    for g_set, names in zip(viewer.group_labels, generated):
        for g in names:
            g_set.remove(g)

    viewer.update_nodes()

    msg = Message("Cleared all subcluster groups (removed %n label instance(s)).", n=total_removed)
    Command_Engine.print_help(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def _refuse(viewer, msg):
    Command_Engine.print_help(viewer, msg, report_message=False)
    Command_Engine.command_failed(viewer, msg)


def start(viewer, plan):
    """At its turn: check that the cluster exists and the libraries load; return SubclusterInputs, or None."""
    if isinstance(plan, SubclusterClearPlan):
        clear_subclusters(viewer)
        return None

    cluster_id, mode, param1, min_sz = plan.cluster_id, plan.mode, plan.param1, plan.min_size
    if getattr(viewer, 'cluster_labels', None) is None:
        _refuse(viewer, Message("Error: No clusters are currently defined. Run 'cluster' first."))
        return None

    # A copy: the worker reads it while the GUI thread carries on.
    cluster_labels = np.array(viewer.cluster_labels)
    if not np.any(cluster_labels == cluster_id):
        _refuse(viewer, Message("Error: Cluster {cluster} is empty or does not exist.", cluster=cluster_id))
        return None

    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(
            viewer, Message("Subclustering {cluster} ({mode})...", cluster=f"cluster_{cluster_id}", mode=mode.upper())
        )
    print(f"Running {mode.upper()} Subclustering for cluster_{cluster_id} (Param={param1}, MinSize={min_sz})...")

    if mode == "mcl":
        try:
            import markov_clustering as mc  # noqa: F401  (availability check)
            import scipy.sparse as sp  # noqa: F401
        except ImportError:
            msg = Message("Missing libraries! Run: {command}", command="pip install markov_clustering networkx scipy")
            print(f"Error: {msg}")
            Command_Engine.command_failed(viewer, f'Error: {msg}')
            Command_Engine.show_status(viewer, msg)
            return None
    elif mode == "leiden":
        try:
            import graspologic_native  # noqa: F401  (availability check)
        except ImportError:
            msg = Message("Missing library! Run: {command}", command="pip install graspologic-native")
            print(f"Error: {msg}")
            Command_Engine.command_failed(viewer, f'Error: {msg}')
            Command_Engine.show_status(viewer, msg)
            return None

    # The edges and scores never change while the Viewer runs, so the worker reads them as they are.
    scores = viewer.edge_scores if hasattr(viewer, 'edge_scores') else None
    return SubclusterInputs(plan, cluster_labels, viewer.edges, scores)


def compute(inputs, progress):
    """Subcluster the cluster. No viewer and no Qt: this runs on the command queue's worker thread.

    A refusal found here (no edge inside the cluster, unusable scores) comes
    back as the result's refusal, which apply() reports before anything changes.
    """
    cluster_id, mode, param1, min_sz = (
        inputs.plan.cluster_id, inputs.plan.mode, inputs.plan.param1, inputs.plan.min_size)
    target_mask = (inputs.cluster_labels == cluster_id)
    subgraph_nodes = np.where(target_mask)[0]

    # --- 3. Extract Subgraph Edges ---
    edges = np.array(inputs.edges, dtype=np.int32).reshape(-1, 2)
    inside = target_mask[edges[:, 0]] & target_mask[edges[:, 1]]

    if not inside.any():
        return SubclusterResult(refusal=Message(
            "Error: No edges exist within {cluster} to perform subclustering.", cluster=f"cluster_{cluster_id}"
        ))

    global_to_local = np.full(len(target_mask), -1, dtype=np.int32)
    global_to_local[subgraph_nodes] = np.arange(len(subgraph_nodes), dtype=np.int32)
    local_edges = global_to_local[edges[inside]]
    if inputs.scores is not None:
        local_edge_scores = np.asarray(inputs.scores)[inside]
    else:
        local_edge_scores = None

    n_sub = len(subgraph_nodes)
    local_labels = np.full(n_sub, -1, dtype=int)

    # =======================================================
    # MODE 1: JACCARD (Topology Filtering + BFS)
    # =======================================================
    if mode == "jaccard":
        thresh = param1

        # Connected components of the edges the Jaccard filter keeps.
        local_labels = network_clustering.jaccard_partition(
            n_sub, local_edges, thresh, min_sz
        )

    # =======================================================
    # MODE 2: MARKOV CLUSTERING (MCL)
    # =======================================================
    elif mode == "mcl":
        inflation = param1
        score_error = cluster_cmd.mcl_edge_score_error(local_edge_scores)
        if score_error:
            return SubclusterResult(refusal=score_error)

        print("Building Sparse Adjacency Matrix...")
        # As cluster builds it: float32, so MCL starts from the same precision.
        matrix = cluster_cmd.mcl_adjacency(local_edges, local_edge_scores, n_sub)

        print(f"Running MCL (Inflation = {inflation}). This may take a moment...")
        clusters = network_clustering.markov_clusters(matrix, inflation, progress=progress)

        local_labels = cluster_cmd.label_mcl_clusters(clusters, n_sub, min_sz)

    # =======================================================
    # MODE 3: LEIDEN COMMUNITY DETECTION
    # =======================================================
    elif mode == "leiden":
        resolution = param1
        print("Building Edge List & Mapping Edge Weights...")

        if local_edge_scores is not None:
            print(f"Running Leiden (Resolution = {resolution}, Weighted).")
        else:
            print(f"Running Leiden (Resolution = {resolution}, Unweighted).")

        # A node with no edge inside the cluster is a singleton subcluster
        # when min_sz is 1, as in MCL and Jaccard, and Noise (-1) otherwise.
        local_labels = network_clustering.leiden_partition(
            n_sub, local_edges, local_edge_scores, resolution, min_sz, seed=42
        )

    # Number subclusters by size in every mode, as cluster numbers clusters.
    local_labels = cluster_cmd.renumber_clusters_by_size(local_labels)

    # No subcluster reaches MIN_SIZE: change nothing, so the cluster keeps its
    # colours and its earlier subclusters, and no undo step is added.
    if not np.any(local_labels != -1):
        return SubclusterResult(nothing_reached=True)

    sub_ids, sub_sizes = np.unique(local_labels[local_labels != -1], return_counts=True)
    sub_counts = dict(zip(sub_ids.tolist(), sub_sizes.tolist()))

    # Compute color map for subclusters
    sorted_subs = sorted(sub_counts.keys())
    n_subclusters = len(sorted_subs)
    if n_subclusters > 0:
        sub_colors = get_subcluster_colors(n_subclusters)
        color_map = {sid: sub_colors[idx % len(sub_colors)] for idx, sid in enumerate(sorted_subs)}
    else:
        color_map = {}
    return SubclusterResult(
        subgraph_nodes=subgraph_nodes, local_labels=local_labels, sub_counts=sub_counts, color_map=color_map,
    )


def apply(viewer, plan, result):
    """Write the subcluster groups and colours, as one undo step, and report them."""
    cluster_id, mode, min_sz = plan.cluster_id, plan.mode, plan.min_size
    if result.refusal is not None:
        _refuse(viewer, result.refusal)
        return
    if result.nothing_reached:
        msg = Message(
            "No subcluster of {cluster} reached the minimum size {min_size}; "
            "nothing was changed.",
            cluster=f"cluster_{cluster_id}",
            min_size=min_sz,
        )
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, msg)
        print(msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    subgraph_nodes, local_labels = result.subgraph_nodes, result.local_labels
    sub_counts, color_map = result.sub_counts, result.color_map
    n_sub = len(subgraph_nodes)

    # --- 4. Update Viewer State ---
    viewer._save_state()

    # Ensure group_labels is initialized
    if not hasattr(viewer, 'group_labels') or viewer.group_labels is None:
        viewer.group_labels = [set() for _ in range(viewer.n_nodes)]

    # Replace the labels an earlier run generated for this cluster; custom
    # lookalikes such as subcluster_N_002 or subcluster_N_0 are kept.
    pattern = re.compile(rf'^subcluster_{cluster_id}_[1-9]\d*$')
    for g_set in viewer.group_labels:
        to_remove = [g for g in g_set if pattern.match(g)]
        for g in to_remove:
            g_set.remove(g)

    # Assign new ones
    group_names = {m: f"subcluster_{cluster_id}_{m}" for m in sub_counts}
    for global_idx, m in zip(subgraph_nodes.tolist(), local_labels.tolist()):
        if m != -1:
            viewer.group_labels[global_idx].add(group_names[m])

    sorted_subs = sorted(sub_counts.keys())
    n_subclusters = len(sorted_subs)

    # Assign colors to viewer.current_colors for nodes in the subclustered target
    unique_local, local_slots = np.unique(local_labels, return_inverse=True)
    slot_colors = np.empty((len(unique_local), 4))
    for slot, m in enumerate(unique_local.tolist()):
        if m == -1:
            slot_colors[slot] = (0.8, 0.8, 0.8, 0.4)  # Grey for noise
        else:
            r, g, b = color_map[m]
            slot_colors[slot] = (r, g, b, 1.0)
    viewer.current_colors[subgraph_nodes] = slot_colors[local_slots]

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

    msg = Message(
        "Done! Found %n subcluster(s) in {cluster} via {mode}.",
        n=n_subclusters,
        cluster=f"cluster_{cluster_id}",
        mode=mode.upper(),
    )
    if hasattr(viewer, 'console_text'):
        Command_Engine.show_status(viewer, msg)
    print(msg)
    Command_Engine.command_succeeded(viewer, msg)


def _no_progress(text=None):
    """run() reports no progress."""


def run(viewer, args):
    """Run the command at once: its plan, start, compute and apply steps in a row."""
    command_plan = plan(viewer, args)
    if command_plan is None:
        return
    inputs = start(viewer, command_plan)
    if inputs is None:
        return
    apply(viewer, command_plan, compute(inputs, _no_progress))
