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
from desktop.Node_Colors import NOISE_COLOR, cluster_colors, declare_categories
from utilities.Localization import JoinedMessage, Message
from utilities.Terminal_Colors import colours_enabled
import importlib.util
import sys
import textwrap
import os
import math
try:
    import commands.group as group_cmd
except ImportError:
    import group as group_cmd
from desktop.Command_Metadata import argument, choice, entry, CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE

# --- What this command declares about itself ---
# Changes only the session's in-memory state.
AGENT_POLICY = 'run'

CATALOG_ENTRY = entry('Identify topology clusters and number retained clusters by size.',
    argument('action', 'Use list for current cluster statistics; otherwise perform clustering, which also removes the generated subcluster_N_M groups of the previous clustering (custom groups with lookalike names such as subcluster_0_2 are kept; undo restores them). Arguments beyond MODE, PARAMETER and MIN_SIZE are refused, as are extras after list.', choice('list')),
    CLUSTER_MODE, CLUSTER_PARAMETER, MIN_SIZE)

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 150
AGENT_PROMPT = """
`cluster [MODE] [PARAM_1] [MIN_SIZE] | cluster list`
- Clusters the current network topology and assigns mutually exclusive cluster labels. These are topology-derived candidate communities, not guaranteed functional families. Communities smaller than MIN_SIZE are labeled noise; MIN_SIZE is an integer of at least `1` and defaults to `10`. A node with no edge is its own community in all three modes: a singleton cluster when MIN_SIZE is `1`, noise when it is larger. Arguments beyond MODE, PARAM_1 and MIN_SIZE, or after `cluster list`, are refused with an error naming the first one, and nothing changes.
- `leiden` is the default mode and uses resolution `1.0`; a higher resolution generally yields more communities. `mcl` uses inflation `2.0` in the range `1.1` to `10.0`; higher inflation generally yields tighter communities. `jaccard` uses shared-neighbor threshold `0.2` in the range `0.0` to `1.0`; higher thresholds discard more weakly supported edges. In `jaccard`, an edge outside every triangle still scores above `0`, and an isolated pair or any clique edge scores `1`.
- Leiden and MCL use network edge scores as weights when available. MODE may be omitted for Leiden. Reclustering replaces prior cluster labels and applies cluster/noise colors; it also removes the generated `subcluster_N_M` groups of the previous clustering (custom groups with lookalike names such as `subcluster_0_2` are kept), and the change, including the removal, participates in undo/redo. `cluster list` prints current cluster sizes and proportions without reclustering.
- Clustering uses every loaded edge and node regardless of the threshold slider or hidden nodes. Retained clusters are renumbered by size, so `cluster_1` is the largest.
"""

if sys.platform == 'win32' and not globals().get("_WINDOWS_ANSI_ENABLED", False):
    os.system('')
    _WINDOWS_ANSI_ENABLED = True


def __getattr__(name):
    """cluster.network_clustering, the clustering kernels, loads when it is first asked for.

    The kernels' module imports Numba, which takes seconds on a cold start. Loading this module,
    and a command's start step, run on the Viewer's GUI thread, which must not wait for that:
    compute() asks for the kernels itself, on the command queue's worker thread.
    """
    if name == "network_clustering":
        from utilities import Network_Kernels
        return Network_Kernels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def library_missing(name):
    """Whether the top-level package `name` is not installed, found without importing it.

    A command checks its libraries at its turn, on the GUI thread, and importing them there holds
    the Viewer for as long as they take to load (markov_clustering loads scikit-learn, SciPy's
    statistics and matplotlib: seconds, and ten times that from a cold disk). The libraries
    load in compute(), on the worker thread. A package set to None in sys.modules is missing,
    as an import of it would say.
    """
    if name in sys.modules:
        return sys.modules[name] is None
    return importlib.util.find_spec(name) is None


def get_colored_cluster_name(cid, name_str, color_map=None):
    if cid == -1:
        # Grey for noise
        r, g, b = 204, 204, 204
    elif color_map and cid in color_map:
        rgba = color_map[cid]
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    else:
        r, g, b = 180, 180, 180
    if not colours_enabled():
        return f"● {name_str}"
    return f"\033[38;2;{r};{g};{b}m●\033[0m {name_str}"

def get_combined_colors(n_clusters):
    return cluster_colors(n_clusters)

def get_cluster_color_map(cluster_ids):
    sorted_clusters = sorted([cid for cid in cluster_ids if cid != -1])
    n_clusters = len(sorted_clusters)
    if n_clusters > 0:
        combined_colors = get_combined_colors(n_clusters)
        return {cid: combined_colors[idx % len(combined_colors)] for idx, cid in enumerate(sorted_clusters)}
    return {}


def renumber_clusters_by_size(labels):
    """Return 1-based cluster IDs ordered by size with stable tie-breaking."""
    labels = np.asarray(labels)
    renumbered = np.full(labels.shape, -1, dtype=int)
    clustered_mask = labels != -1
    if not np.any(clustered_mask):
        return renumbered

    clustered_labels = labels[clustered_mask]
    _cluster_ids, inverse, counts = np.unique(
        clustered_labels,
        return_inverse=True,
        return_counts=True,
    )

    member_indices = np.flatnonzero(clustered_mask)
    minimum_member_indices = np.full(len(counts), labels.size, dtype=int)
    np.minimum.at(minimum_member_indices, inverse, member_indices)

    # np.lexsort uses the final key as primary: largest size first, then the
    # lowest member node index for deterministic equal-size ordering.
    size_order = np.lexsort((minimum_member_indices, -counts))
    new_ids = np.empty(len(counts), dtype=int)
    new_ids[size_order] = np.arange(1, len(counts) + 1, dtype=int)
    renumbered[clustered_mask] = new_ids[inverse]
    return renumbered


# The documented MCL inflation range, bounds included; subcluster shares it.
MCL_INFLATION_RANGE = (1.1, 10.0)


def mcl_inflation_error(inflation):
    """Return the error Message for an MCL inflation outside its range, or None."""
    low, high = MCL_INFLATION_RANGE
    # Written so that nan, which compares false, is refused too.
    if low <= inflation <= high:
        return None
    return Message(
        "Error: MCL inflation must be between {low} and {high}; got {inflation}.",
        low=low, high=high, inflation=inflation,
    )


def leiden_resolution_error(resolution):
    """Return the error Message for a Leiden resolution that is not a finite number above 0, or None."""
    # Written so that nan, which compares false, is refused too.
    if math.isfinite(resolution) and resolution > 0:
        return None
    return Message(
        "Error: Leiden resolution must be a finite number above 0; got {resolution}.",
        resolution=resolution,
    )


def jaccard_threshold_error(threshold):
    """Return the error Message for a Jaccard threshold outside 0.0 - 1.0, or None."""
    low, high = 0.0, 1.0
    # Written so that nan, which compares false, is refused too.
    if low <= threshold <= high:
        return None
    return Message(
        "Error: Jaccard threshold must be between {low} and {high}; got {threshold}.",
        low=low, high=high, threshold=threshold,
    )


def min_size_error(min_size):
    """Return the error Message for a MIN_SIZE below 1, or None."""
    if min_size >= 1:
        return None
    return Message("Error: Min Size must be at least 1; got {min_size}.", min_size=min_size)


def parameter_error(mode, param1, min_size):
    """Return the error Message for a MIN_SIZE or mode parameter the clustering cannot use, or None."""
    error = min_size_error(min_size)
    if error is not None:
        return error
    if mode == "mcl":
        return mcl_inflation_error(param1)
    if mode == "jaccard":
        return jaccard_threshold_error(param1)
    return leiden_resolution_error(param1)


def extra_argument_error(argument, syntax):
    """Return the error Message for an argument the command does not take, with its usage."""
    return JoinedMessage([
        Message("Error: Unrecognized cluster argument '{argument}'.", argument=argument),
        Message("Usage: {syntax}", syntax=syntax),
    ], separator="\n")


def mcl_edge_score_error(scores):
    """Return the error Message for edge scores MCL cannot use, or None.

    MCL needs every score finite and above 0; scores is None for an
    unweighted network. A score of 0 or below (-log10 of an E-value of 1 or
    more is 0 or negative, and so is the EFI alignment score of a weak hit
    between long sequences) or a non-finite one makes MCL fail with a NaN error.
    """
    if scores is None:
        return None
    scores = np.asarray(scores, dtype=float)
    # Written so that nan, which compares false, is counted too.
    unusable = int(np.count_nonzero(~(np.isfinite(scores) & (scores > 0))))
    if unusable == 0:
        return None
    return JoinedMessage([
        Message(
            "Error: MCL needs every edge score to be finite and above 0, but found %n edge(s) with a score that is not.",
            n=unusable,
        ),
        "MCL cannot use a score of 0 or below, such as -log10 of an E-value of 1 or more or the EFI alignment score of a weak hit between long sequences, or a score that is not finite.",
    ], separator="\n")


def mcl_adjacency(edges, scores, n_nodes):
    """The symmetric adjacency matrix MCL runs on, in float32 for cluster and subcluster alike.

    scores is None for an unweighted network, whose edges all weigh 1. Both
    commands build it here, so MCL starts from the same precision in each.
    """
    import scipy.sparse as sp

    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    if scores is None:
        values = np.ones(len(row), dtype=np.float32)
    else:
        values = np.asarray(scores, dtype=np.float32)
        values = np.concatenate([values, values])
    return sp.csr_matrix((values, (row, col)), shape=(n_nodes, n_nodes))


def label_mcl_clusters(clusters, n_nodes, min_size):
    """Label nodes by the clusters markov_clusters returns, 1-based; the rest are -1.

    MCL clusters can overlap. A cluster of at least min_size nodes labels its
    members, a later cluster overwriting an earlier one on shared nodes, and
    a cluster left with fewer than min_size nodes by that is Noise.
    """
    labels = np.full(n_nodes, -1, dtype=int)
    cluster_id = 1
    for comp in clusters:
        if len(comp) >= min_size:
            for node in comp:
                labels[node] = cluster_id
            cluster_id += 1
    kept_ids, kept_sizes = np.unique(labels[labels != -1], return_counts=True)
    labels[np.isin(labels, kept_ids[kept_sizes < min_size])] = -1
    return labels


def mcl_attractor_centres(labels, clusters, attractors, min_size):
    """The attractors (markov_clusters_and_attractors) that are still members
    of their own cluster in labels, the labelling label_mcl_clusters gave.

    A cluster's attractor is its centre. An overlapping later cluster can
    take the attractor over, and a cluster an overlap left too small is
    Noise; such a cluster's attractor is not returned.
    """
    centres = []
    cluster_id = 1
    for comp, attractor in zip(clusters, attractors):
        if len(comp) >= min_size:
            if labels[attractor] == cluster_id:
                centres.append(int(attractor))
            cluster_id += 1
    return centres


def mcl_centres(labels, attractor_centres, edges, scores, kernels):
    """{cluster id: centre node} for an MCL clustering numbered by size.

    A cluster's centre is its attractor (mcl_attractor_centres). A cluster
    whose attractor went to another cluster takes the member with the
    strongest edges to the rest of its cluster instead.
    """
    centres = {int(labels[node]): node for node in attractor_centres}
    missing = set(np.unique(labels[labels != -1]).tolist()) - centres.keys()
    if missing:
        strongest = kernels.modularity_centres(labels, edges, scores, 0.0)
        centres.update({cid: strongest[cid] for cid in missing})
    return centres


def centre_name(viewer, node):
    """The name of node, its full header, as the cluster table prints it."""
    headers = getattr(viewer, 'full_headers', None)
    if headers is not None and 0 <= node < len(headers):
        return str(headers[node])
    return f"Node {node}"


# --- The % identity columns of the cluster and subcluster tables ---

NO_MSA_NOTE = "No MSA is loaded, so the table gives no % identity."


@dataclass(frozen=True)
class AlignmentInputs:
    """The loaded MSA as the identity columns read it, taken on the GUI thread."""
    matrix: object      # the alignment's CSR code matrix; code 0 is a gap
    int_to_aa: dict
    node_rows: object   # node i's matrix row, or -1 when the MSA has none


@dataclass(frozen=True)
class IdentityColumns:
    """The tables' Mean %ID and Min %ID columns.

    identities is {group id: Alignment_Profiles.GroupIdentity}, or None when
    the table has no such columns, and note then says why. exact_pairs and
    sample_pairs are the limits the values were computed with.
    """
    identities: object = None
    note: str = None
    exact_pairs: int = 0
    sample_pairs: int = 0


def alignment_inputs(viewer):
    """The loaded MSA for the identity columns, or None when none is loaded."""
    aln = getattr(getattr(viewer, 'alignment', None), 'aln', None)
    matrix = getattr(aln, 'matrix', None)
    if matrix is None or getattr(matrix, 'format', None) != 'csr':
        return None
    node_rows, _mapped = Command_Engine.get_alignment_mapping(viewer)
    node_rows = np.array(node_rows, dtype=np.int64)
    if node_rows.shape != (viewer.n_nodes,):
        return None
    return AlignmentInputs(matrix, dict(getattr(aln, 'int_to_aa', None) or {}), node_rows)


def identity_columns(alignment, labels, nodes=None):
    """IdentityColumns for the groups labels gives; Noise (-1) has none.

    alignment comes from alignment_inputs. nodes, when given, are the network
    nodes labels describes, in order (subcluster's subgraph). On the worker
    thread: the identity kernels' module imports Numba (see __getattr__).
    """
    if alignment is None:
        return IdentityColumns(note=NO_MSA_NOTE)
    from utilities import Alignment_Profiles
    node_rows = alignment.node_rows if nodes is None else alignment.node_rows[nodes]
    exact_pairs = Alignment_Profiles.IDENTITY_EXACT_PAIRS
    sample_pairs = Alignment_Profiles.IDENTITY_SAMPLE_PAIRS
    try:
        identities = Alignment_Profiles.group_identities(
            alignment.matrix, alignment.int_to_aa, node_rows, labels,
            exact_pairs=exact_pairs, sample_pairs=sample_pairs)
    except Exception as exc:  # a cancelled job still stops: JobCancelled is no Exception
        return IdentityColumns(note=f"The % identity columns could not be computed: {exc}")
    return IdentityColumns(identities, None, exact_pairs, sample_pairs)


def has_identity_columns(columns):
    return columns is not None and columns.identities is not None


def table_rule(columns):
    """The table's ==== rule, longer by the identity columns when it has them."""
    return "=" * (78 if has_identity_columns(columns) else 54)


def identity_heading(columns):
    if not has_identity_columns(columns):
        return ""
    return f" {'Mean %ID':>9} | {'Min %ID':>9} |"


def identity_divider(columns):
    return f"+{'-'*11}+{'-'*11}" if has_identity_columns(columns) else ""


def identity_cells(columns, group_id):
    """A row's identity cells: blank for Noise, '-' without a pair to compare,
    and marked '~' when they come from a sample of the pairs."""
    if not has_identity_columns(columns):
        return ""
    identity = columns.identities.get(group_id)
    if identity is None:
        mean = low = ""
    elif not identity.pairs:
        mean = low = "-"
    else:
        mark = "~" if identity.sampled else ""
        mean, low = f"{mark}{identity.mean:.1f}%", f"{mark}{identity.minimum:.1f}%"
    return f" {mean:>9} | {low:>9} |"


def identity_notes(columns, sizes):
    """The lines under the table: what its identity columns hold, or why it has none.

    sizes is {group id: member count}.
    """
    if columns is None:
        return []
    if not has_identity_columns(columns):
        return textwrap.wrap(columns.note, len(table_rule(columns))) if columns.note else []
    identities = columns.identities
    notes = ["% ID: identical standard residues / alignment length incl. internal gaps, as the "
             "tools count a local alignment: for two MSA rows, the columns from their first to "
             "their last aligned residue pair where either has a residue. Mean and minimum over "
             "all member pairs."]
    if any(identity.sampled for identity in identities.values()):
        notes.append(f"~ : over {columns.exact_pairs:,} member pairs, so {columns.sample_pairs:,} "
                     "random pairs (seed 42) were compared; the minimum may be above the true one.")
    if any(not identity.pairs for identity in identities.values()):
        notes.append("- : fewer than two members have an MSA row, or no pair shares an aligned residue.")
    missing = sum(sizes.get(group_id, 0) - identity.rows for group_id, identity in identities.items())
    if missing:
        notes.append(f"{missing} member(s) have no row in the MSA and are left out of % ID.")
    # Each note wrapped to the table's width.
    return [line for note in notes for line in textwrap.wrap(note, len(table_rule(columns)))]


def print_help():
    print("""
    Topology Clustering Tool
    ========================
    Usage: cluster [MODE] [PARAM_1] [MIN_SIZE]
           cluster list
           cluster help

    Modes:
      leiden (Default)
          - Leiden Community Detection (Modularity & Density optimization).
          - Automatically uses network edge scores as structural weights if available.
          - Requires: pip install graspologic-native
          - PARAM_1: Resolution (a finite number above 0; higher = more clusters). Default: 1.0
          - Example: cluster leiden 1.0 10  (OR simply: cluster 1.0 10)
          
      mcl
          - Markov Clustering Algorithm (Simulates random walks).
          - Automatically uses network edge scores as structural weights if available.
          - Requires: pip install markov_clustering networkx
          - PARAM_1: Inflation (1.1 - 10.0, higher = tighter clusters). Default: 2.0
          - Example: cluster mcl 2.0 10

      jaccard
          - Filters edges based on shared neighbors and cuts weak connections.
          - Uses Numba (JIT compilation) when installed, and a slower pure-Python
            fallback otherwise.
          - PARAM_1: Threshold (0.0 - 1.0). Default: 0.2
          - Example: cluster jaccard 0.3 10
          
    Commands:
      list          - Prints current cluster statistics and node distributions to the console.

    Arguments:
      [MIN_SIZE]    (Optional) Minimum Cluster Size (Integer, at least 1).
                    - Groups smaller than this are designated as 'Noise' (Cluster -1).
                    - Default: 10

      Anything after the arguments above (or after list) is refused with an
      error that names the first extra argument, and nothing changes.

    What is clustered:
      - Clustering uses every loaded edge and node, regardless of the similarity
        slider and hidden nodes.
      - Jaccard compares closed neighbourhoods (a node counts as its own
        neighbour). An edge whose endpoints have a and b neighbours, c of them
        shared, scores (c + 2) / (a + b - c), so an edge outside every triangle
        still scores above 0, while an isolated pair and every edge of a clique
        score 1.
      - Isolated nodes (no edge) are alike in all three modes: singleton
        clusters when MIN_SIZE is 1, Noise when it is larger.
      - MCL gives every node a self-loop as heavy as its strongest edge (1 for a
        node with no edge), so multiplying every score by a constant does not
        change the clusters.
      - MCL needs every edge score to be finite and above 0.

    Cluster numbering:
      Retained clusters are numbered from largest to smallest. Equal-size
      clusters are ordered by their lowest member node index.

    The statistics table:
      - Each cluster's row gives its node count and its share of all nodes.
      - With an MSA loaded, Mean %ID and Min %ID give the pairwise sequence
        identity of the cluster's members, counted as the Pairwise Alignment
        and Embedding Search tools count a local alignment: identical standard
        residues / alignment length incl. internal gaps. For two MSA rows that
        length is the columns from their first to their last aligned residue
        pair (both have a residue) where either has a residue; overhanging
        ends are not counted. X and other non-standard letters count toward
        the length but never as identical. The tools align each pair
        themselves, so their value for a pair can differ from the MSA's.
      - The columns hold the mean and the minimum over all member pairs.
        Members without an MSA row are left out; '-' marks a cluster with fewer
        than two of them. Without an MSA the table has no such columns.
      - A cluster with over 50,000,000 member pairs (about 10,000 members)
        compares 2,000,000 random pairs instead (seed 42), marked '~': its mean
        stays close to the full one, but its minimum may be above the true one.
      - MCL and Leiden name each cluster's centre node to the right of the
        table. For MCL it is the attractor the flow settles on; for Leiden, the
        member whose leaving would cost the most modularity at the resolution
        used. Jaccard names none.

    Subclusters:
      Running cluster replaces the current clusters, so it also removes the
      subcluster groups (subcluster_N_M) the subcluster command generated for
      them; undo brings them back. Custom groups with lookalike names, such as
      subcluster_0_2 or subcluster_001_2, are kept.
    """)

# --- The command, in the steps the Viewer's command queue runs (Command_Queue) ---
# plan() reads the arguments when the command is given; start() checks the
# network and the libraries at its turn; compute() clusters, on the queue's
# worker thread (Leiden, Jaccard and MCL release the GIL); apply() writes the
# clusters and colours and saves the undo state, on the GUI thread. run() does
# all four at once, as the VR viewer and the tests run the command.


@dataclass(frozen=True)
class ClusterPlan:
    mode: str
    param1: float
    min_size: int
    heavy = True


@dataclass(frozen=True)
class ClusterListPlan:
    """cluster list: it reads the clusters at its turn, after a clustering given before it."""
    heavy = False


@dataclass(frozen=True)
class ClusterInputs:
    plan: ClusterPlan
    n_nodes: int
    edges: object
    scores: object
    # The loaded MSA for the identity columns (alignment_inputs), or None.
    alignment: object = None


@dataclass(frozen=True)
class ClusterResult:
    labels: object
    unique_labels: object
    label_slots: object
    counts: object
    color_map: dict
    # {cluster id: centre node} for MCL and Leiden; None for Jaccard, which finds none.
    centres: object = None
    # The table's identity columns (identity_columns).
    identity: object = None


def plan(viewer, args):
    """Read the arguments: a ClusterPlan or ClusterListPlan, or None when help ran or they were refused."""
    # --- 1. Help Check ---
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return None

    # --- LIST COMMAND ---
    if args and args[0].lower() == 'list':
        if len(args) > 1:
            refusal = extra_argument_error(args[1], "cluster list")
            Command_Engine.print_help(viewer, refusal, report_message=False)
            Command_Engine.command_failed(viewer, refusal)
            return None
        return ClusterListPlan()

    # --- 2. Parse Arguments ---
    mode = "leiden"
    param1 = None
    min_sz = 10
    # [MODE] PARAM_1 MIN_SIZE take three arguments; a bare PARAM_1 MIN_SIZE two.
    taken = 3

    if len(args) >= 1:
        first_arg = args[0].lower()
        if first_arg in ['jaccard', 'mcl', 'leiden']:
            mode = first_arg
            if len(args) >= 2:
                try: param1 = float(args[1])
                except ValueError:
                    refusal = Message("Error: Parameter must be a number.")
                    Command_Engine.print_help(viewer, refusal, report_message=False)
                    Command_Engine.command_failed(viewer, refusal)
                    return None
            if len(args) >= 3:
                try: min_sz = int(args[2])
                except ValueError:
                    refusal = Message("Error: Min Size must be an integer.")
                    Command_Engine.print_help(viewer, refusal, report_message=False)
                    Command_Engine.command_failed(viewer, refusal)
                    return None
        else:
            taken = 2
            # A bare number is the resolution of the default mode, Leiden.
            try: param1 = float(args[0])
            except ValueError:
                refusal = Message("Error: Unknown mode or invalid number '{argument}'", argument=args[0])
                Command_Engine.print_help(viewer, refusal, report_message=False)
                Command_Engine.command_failed(viewer, refusal)
                return None
            if len(args) >= 2:
                try: min_sz = int(args[1])
                except ValueError:
                    refusal = Message("Error: Min Size must be an integer.")
                    Command_Engine.print_help(viewer, refusal, report_message=False)
                    Command_Engine.command_failed(viewer, refusal)
                    return None

    # Nothing may follow the arguments the form takes.
    if len(args) > taken:
        refusal = extra_argument_error(args[taken], "cluster [MODE] [PARAM_1] [MIN_SIZE]")
        Command_Engine.print_help(viewer, refusal, report_message=False)
        Command_Engine.command_failed(viewer, refusal)
        return None

    # Apply defaults if param1 wasn't provided
    if param1 is None:
        if mode == "jaccard": param1 = 0.2
        elif mode == "mcl": param1 = 2.0
        elif mode == "leiden": param1 = 1.0

    refusal = parameter_error(mode, param1, min_sz)
    if refusal:
        Command_Engine.print_help(viewer, refusal, report_message=False)
        Command_Engine.command_failed(viewer, refusal)
        return None
    return ClusterPlan(mode, param1, min_sz)


def list_clusters(viewer):
    """cluster list: print the current cluster statistics."""
    if getattr(viewer, 'cluster_labels', None) is None:
        msg = Message("No clusters are currently defined.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    labels = viewer.cluster_labels
    n_nodes = viewer.n_nodes
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    sorted_clusters = sorted([k for k in label_counts.keys() if k != -1])
    color_map = get_cluster_color_map(sorted_clusters)

    print(f"\n{'='*54}")
    print(f"--- Current Cluster Statistics (Total: {n_nodes}) ---")
    print(f"{'='*54}")
    print(f"| {'Cluster Name':<22} | {'Node Count':>10} | {'Percent':>10} |")
    print(f"|{'-'*24}+{'-'*12}+{'-'*12}|")

    noise_count = label_counts.get(-1, 0)
    noise_pct = (noise_count / n_nodes) * 100
    noise_name_padded = get_colored_cluster_name(-1, f"{'Noise (Unclustered)':<20}", color_map)
    print(f"| {noise_name_padded} | {noise_count:>10} | {noise_pct:>9.2f}% |")

    for cid in sorted_clusters:
        c_count = label_counts[cid]
        c_pct = (c_count / n_nodes) * 100
        c_name_padded = get_colored_cluster_name(cid, f"{f'Cluster {cid}':<20}", color_map)
        print(f"| {c_name_padded} | {c_count:>10} | {c_pct:>9.2f}% |")
    print(f"{'='*54}\n")

    msg = Message("Listed %n cluster(s) in console.", n=len(sorted_clusters))
    Command_Engine.show_status(viewer, msg)
    Command_Engine.command_succeeded(viewer, msg)


def start(viewer, plan):
    """At its turn: check the libraries and the scores the mode needs; return ClusterInputs, or None."""
    if isinstance(plan, ClusterListPlan):
        list_clusters(viewer)
        return None

    mode, param1, min_sz = plan.mode, plan.param1, plan.min_size
    Command_Engine.show_status(viewer, Message("Clustering ({mode})...", mode=mode.upper()))
    print(f"Running {mode.upper()} Clustering (Param={param1}, MinSize={min_sz})...")
    scores = viewer.edge_scores if hasattr(viewer, 'edge_scores') else None

    if mode == "mcl":
        if any(library_missing(name) for name in ("markov_clustering", "networkx", "scipy")):
            msg = Message("Missing libraries! Run: {command}", command="pip install markov_clustering networkx scipy")
            print(f"Error: {msg}")
            Command_Engine.command_failed(viewer, f'Error: {msg}')
            Command_Engine.show_status(viewer, msg)
            return None

        score_error = mcl_edge_score_error(scores)
        if score_error:
            Command_Engine.print_help(viewer, score_error, report_message=False)
            Command_Engine.command_failed(viewer, score_error)
            return None

    elif mode == "leiden":
        if library_missing("graspologic_native"):
            msg = Message("Missing library! Run: {command}", command="pip install graspologic-native")
            print(f"Error: {msg}")
            Command_Engine.command_failed(viewer, f'Error: {msg}')
            Command_Engine.show_status(viewer, msg)
            return None

    # The edges and scores never change while the Viewer runs, so the worker reads them as they are.
    return ClusterInputs(plan, viewer.n_nodes, viewer.edges, scores, alignment_inputs(viewer))


def compute(inputs, progress):
    """Cluster the network. No viewer and no Qt: this runs on the command queue's worker thread."""
    # Here, on the worker thread, and not when the module loads: see __getattr__.
    from utilities import Network_Kernels as network_clustering

    mode, param1, min_sz = inputs.plan.mode, inputs.plan.param1, inputs.plan.min_size
    n_nodes = inputs.n_nodes
    edges = np.array(inputs.edges, dtype=np.int32).reshape(-1, 2)
    labels = np.full(n_nodes, -1, dtype=int)

    # =======================================================
    # MODE 1: JACCARD (Topology Filtering + BFS)
    # =======================================================
    if mode == "jaccard":
        thresh = param1

        # Connected components of the edges the Jaccard filter keeps.
        labels = network_clustering.jaccard_partition(
            n_nodes, edges, thresh, min_sz
        )

    # =======================================================
    # MODE 2: MARKOV CLUSTERING (MCL)
    # =======================================================
    elif mode == "mcl":
        inflation = param1
        print("Building Sparse Adjacency Matrix...")
        # The edge scores weigh the edges when the viewer has them; otherwise each weighs 1.
        matrix = mcl_adjacency(edges, inputs.scores, n_nodes)

        print(f"Running MCL (Inflation = {inflation}). This may take a moment...")
        clusters, attractors = network_clustering.markov_clusters_and_attractors(
            matrix, inflation, progress=progress
        )

        labels = label_mcl_clusters(clusters, n_nodes, min_sz)
        attractor_centres = mcl_attractor_centres(labels, clusters, attractors, min_sz)

    # =======================================================
    # MODE 3: LEIDEN COMMUNITY DETECTION
    # =======================================================
    elif mode == "leiden":
        resolution = param1
        print("Building Edge List & Mapping Edge Weights...")

        # Automatically harness the physics edge weights if available!
        weights = inputs.scores
        if weights is not None:
            print(f"Running Leiden (Resolution = {resolution}, Weighted).")
        else:
            print(f"Running Leiden (Resolution = {resolution}, Unweighted).")

        # A node with no edge is a singleton cluster when min_sz is 1, as in
        # MCL and Jaccard, and Noise (-1) otherwise.
        labels = network_clustering.leiden_partition(
            n_nodes, edges, weights, resolution, min_sz, seed=42
        )

    # Canonicalize IDs across every clustering mode without changing membership.
    labels = renumber_clusters_by_size(labels)

    # Each cluster's centre node, found on the final labels: a centre is a
    # member, so its cluster id is simply its label.
    centres = None
    if mode == "mcl":
        centres = mcl_centres(labels, attractor_centres, edges, inputs.scores, network_clustering)
    elif mode == "leiden":
        centres = network_clustering.modularity_centres(labels, edges, inputs.scores, param1)

    unique_labels, label_slots, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )
    unique_clusters = [k for k in unique_labels if k != -1]
    return ClusterResult(labels, unique_labels, label_slots, counts,
                         get_cluster_color_map(unique_clusters), centres,
                         identity_columns(inputs.alignment, labels))


def removed_subclusters_message(removed):
    """The note that names how many generated subcluster groups a new or cleared clustering removed, or None."""
    if not removed:
        return None
    return Message(
        "Removed %n generated subcluster group(s) of the previous clustering.",
        n=len(removed),
    )


def apply(viewer, plan, result):
    """Write the clusters and their colours, as one undo step, and report them."""
    mode, param1, min_sz = plan.mode, plan.param1, plan.min_size
    n_nodes = viewer.n_nodes
    labels = result.labels
    unique_labels, label_slots, counts, color_map = (
        result.unique_labels, result.label_slots, result.counts, result.color_map)

    # --- 6. Update Viewer ---

    # ---> NEW: Save the state before applying new clusters and colors for Undo/Redo
    viewer._save_state()

    viewer.cluster_labels = labels

    # The generated subcluster groups name nodes of the clusters just replaced.
    removed_subclusters = group_cmd.remove_generated_subcluster_groups(viewer)

    # Store parameters as strings so external commands (align.py, etc.) know what was used
    viewer.last_cluster_params = (f"{mode.upper()}_{param1}", min_sz)

    # Apply Colors
    slot_colors = np.empty((len(unique_labels), 4))
    for slot, lbl in enumerate(unique_labels):
        if lbl == -1:
            slot_colors[slot] = NOISE_COLOR  # Grey for noise
        else:
            r, g, b = color_map[lbl]
            slot_colors[slot] = (r, g, b, 1.0)
    viewer.current_colors[:n_nodes] = slot_colors[label_slots]

    # What the colors mean: a category for each cluster, by id, and Noise last; two clusters that share a color stay two.
    named = [slot for slot, lbl in enumerate(unique_labels) if lbl != -1]
    categories = [(Message("Cluster {number}", number=int(unique_labels[slot])), slot_colors[slot]) for slot in named]
    category_of_slot = np.full(len(unique_labels), len(named))
    category_of_slot[named] = np.arange(len(named))
    if len(named) < len(unique_labels):
        categories.append((Message("Noise"), NOISE_COLOR))
    declare_categories(viewer, Message("Clusters"), categories, category_of_slot[label_slots])

    viewer.update_nodes()

    # --- 7. Print Statistics ---
    label_counts = dict(zip(unique_labels, counts))

    # MCL and Leiden name each cluster's centre node to the right of the
    # table, outside its border and unpadded, so a long header neither widens
    # the table nor is cut short. Jaccard finds no centres.
    centre_names = {}
    if result.centres is not None:
        centre_names = {cid: centre_name(viewer, node) for cid, node in result.centres.items()}

    def beside(text):
        return f" {text}" if text else ""

    # With an MSA loaded, the mean and minimum pairwise % identity of each
    # cluster's members follow Percent, inside the table.
    identity = result.identity
    rule = table_rule(identity)
    print(f"\n{rule}")
    print(f"--- {mode.upper()} Clustering Stats (Total Nodes: {n_nodes}) ---")
    print(rule)
    print(f"| {'Cluster Name':<22} | {'Node Count':>10} | {'Percent':>10} |"
          f"{identity_heading(identity)}{beside('Centre' if centre_names else '')}")
    print(f"|{'-'*24}+{'-'*12}+{'-'*12}{identity_divider(identity)}|")

    noise_count = label_counts.get(-1, 0)
    noise_pct = (noise_count / n_nodes) * 100
    noise_name_padded = get_colored_cluster_name(-1, f"{'Noise (Unclustered)':<20}", color_map)
    print(f"| {noise_name_padded} | {noise_count:>10} | {noise_pct:>9.2f}% |{identity_cells(identity, -1)}")

    sorted_clusters = sorted([k for k in label_counts.keys() if k != -1])
    for cid in sorted_clusters:
        c_count = label_counts[cid]
        c_pct = (c_count / n_nodes) * 100
        c_name_padded = get_colored_cluster_name(cid, f"{f'Cluster {cid}':<20}", color_map)
        print(f"| {c_name_padded} | {c_count:>10} | {c_pct:>9.2f}% |"
              f"{identity_cells(identity, cid)}{beside(centre_names.get(cid, ''))}")
    print(rule)
    for note in identity_notes(identity, label_counts):
        print(note)
    print()

    n_clusters = len(sorted_clusters)
    msg = Message("Done! Found %n cluster(s) via {mode}.", n=n_clusters, mode=mode.upper())
    parts = [msg]
    removed_note = removed_subclusters_message(removed_subclusters)
    if removed_note is not None:
        parts.append(removed_note)

    # A custom group made before this run may already carry a new cluster's
    # name, which makes #cluster_N# ambiguous. Custom groups are not renamed or removed.
    group_labels = getattr(viewer, 'group_labels', None)
    group_names = set().union(*group_labels) if group_labels is not None else set()
    shared_names = [f"cluster_{cid}" for cid in sorted_clusters if f"cluster_{cid}" in group_names]
    if shared_names:
        parts.append(Message(
            "Warning: custom groups and clusters now share %n name(s): {names}. "
            "Selecting them with #name# is ambiguous until the group is removed.",
            n=len(shared_names), names=", ".join(shared_names),
        ))
    if len(parts) > 1:
        msg = JoinedMessage(parts)
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


# What "reset clusters" restores: commands/reset.py asks each command module for its targets.
RESET_TARGETS = ('cluster',)
RESET_HELP = {'clusters': (40, 'Clears all cluster labels, and removes the subcluster groups\n'
                               '(subcluster_N_M) the subcluster command generated for them. Custom\n'
                               'groups with lookalike names, such as subcluster_0_2, are kept.')}


def reset_changes_state(viewer, target):
    """False when there are no clusters, no clustering parameters and no generated subcluster groups."""
    groups = getattr(viewer, 'group_labels', None) or []
    return (getattr(viewer, 'cluster_labels', None) is not None
            or getattr(viewer, 'last_cluster_params', None) is not None
            or any(group_cmd.is_generated_subcluster_name(g) for g_set in groups for g in g_set))


def reset(viewer, target):
    """Clear the clusters, the parameters they were made with, and the subcluster groups generated for them."""
    from commands.reset import ResetOutcome
    viewer.cluster_labels = None
    # The parameters belong to the clusters just cleared, and so do the groups
    # the subcluster command generated for them.
    viewer.last_cluster_params = None
    removed_subclusters = group_cmd.remove_generated_subcluster_groups(viewer)
    group_cmd.clear_label_visuals(viewer)
    return ResetOutcome("clusters", note=removed_subclusters_message(removed_subclusters))
