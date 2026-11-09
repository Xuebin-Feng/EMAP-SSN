# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View tests' shared fixtures: an alignment folder holding the guide tree the real writer saves, and a Viewer over it.

The trees are written by Embedding_MSA.write_guide_tree, the code that makes the files the page reads, so a change
to the file format shows up here. random_linkage makes a tree of any size at once: scipy's linkage needs every
pairwise distance, which at 44,000 leaves is 8 GB.

Not collected by unittest; import with ``from tests.tree_fixtures import ...``.
"""

import http.client
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, SRC / "tools"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

ALIGNMENT_NAME = "Set_[model]_alignment.fasta"
TREE_NAME = "Set_[model]_guide_tree.nex"


def writer():
    """Embedding_MSA, which writes the tree files."""
    import Embedding_MSA

    return Embedding_MSA


def random_linkage(count, seed=1):
    """A valid UPGMA-style linkage matrix of ``count`` leaves, made without any distances.

    Random pairs of the clusters still open are joined at rising heights, so the heights never fall as a
    dendrogram's must not.
    """
    rng = np.random.default_rng(seed)
    pool = list(range(count))
    sizes = {leaf: 1 for leaf in range(count)}
    rows = np.zeros((count - 1, 4))
    height = 0.0
    for step in range(count - 1):
        first = pool.pop(int(rng.integers(len(pool))))
        second = pool.pop(int(rng.integers(len(pool))))
        height += float(rng.random()) + 0.01
        new = count + step
        sizes[new] = sizes[first] + sizes[second]
        rows[step] = [first, second, height, sizes[new]]
        pool.append(new)
    return rows


def shaped_linkage(shape, count, seed=1):
    """A valid UPGMA-style linkage matrix of ``count`` leaves in a given shape, made without any distances.

    "bushy" is random_linkage. "caterpillar" joins one leaf at a time to a growing clade, so it is as deep as it
    has leaves. "balanced" joins neighbours level by level. "chains" is 8 caterpillars joined as a balanced tree:
    long chains, thousands of nodes deep at 44,000 leaves, hung from a shallow trunk. Heights rise with every row, so a clade is never above its parent.
    """
    if shape == "bushy":
        return random_linkage(count, seed)
    rows = []
    sizes = {leaf: 1 for leaf in range(count)}

    def join(first, second):
        new = count + len(rows)
        sizes[new] = sizes[first] + sizes[second]
        rows.append([first, second, 1.0 + len(rows) * 0.01, sizes[new]])
        return new

    def chain(leaves):
        clade = leaves[0]
        for leaf in leaves[1:]:
            clade = join(clade, leaf)
        return clade

    def balance(clades):
        while len(clades) > 1:
            joined = [join(clades[i], clades[i + 1]) for i in range(0, len(clades) - 1, 2)]
            clades = joined + ([clades[-1]] if len(clades) % 2 else [])
        return clades[0]

    if shape == "caterpillar":
        chain(list(range(count)))
    elif shape == "balanced":
        balance(list(range(count)))
    elif shape == "chains":
        groups = max(1, min(8, count // 2))
        bounds = [round(count * g / groups) for g in range(groups + 1)]
        balance([chain(list(range(bounds[g], bounds[g + 1]))) for g in range(groups)])
    else:
        raise ValueError(f"no tree shape {shape!r}")
    return np.asarray(rows, dtype=float)


def neighbor_joining_tree(count, seed=4):
    """(linkage, branch lengths) of the neighbor-joining tree of ``count`` random points: what a [&U] file is made from."""
    from scipy.spatial.distance import pdist

    points = np.random.default_rng(seed).random((count, 4))
    return writer().neighbor_joining_condensed(pdist(points), count, return_branch_lengths=True)


def headers_of(count, prefix="sp|Q"):
    """Distinct headers shaped like UniProt ones, with the characters that need quoting in a tree file."""
    return [f"{prefix}{i:05d}|PROT_{i} Some protein, organism {i}" for i in range(count)]


def save_guide_tree(path, headers, *, seed=1, notes=(), linkage=None, **options):
    """Save a guide tree of ``headers`` at ``path`` (its folder is made), as the writer does; returns the path."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    link = random_linkage(len(headers), seed) if linkage is None else linkage
    writer().write_guide_tree(path, link, headers, notes=list(notes), **options)
    return path


def make_alignment_folder(root, headers, *, alignment=ALIGNMENT_NAME, tree=TREE_NAME, notes=None, **options):
    """A folder with an (empty) alignment file and, unless tree is None, the tree the tool saves for it.

    notes default to the ones the tool records, named for the alignment; pass notes=[] for a tree without any.
    Returns (alignment path, tree path or None).
    """
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    msa_path = root / alignment
    msa_path.write_text(">x\nA\n", encoding="utf-8")
    tree_path = None
    if tree is not None:
        if notes is None:
            notes = [("ALIGNMENT", alignment), ("SEQUENCE_SET", "Set"), ("MODEL", "model")]
        tree_path = save_guide_tree(root / "Guide_Trees" / tree, headers, notes=notes, **options)
    return msa_path, tree_path


def node_colors(count, painted=None):
    """The (count, 4) float32 colours of a network's nodes: the Viewer's initial node colour, with ``painted`` over it.

    painted maps a node number to a colour (a name, or #rrggbb), or to an RGBA tuple of 0 to 1.
    """
    from matplotlib import colors as mcolors

    import EMAPSSN_Config as cfg

    colors = np.tile(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), (count, 1)).astype(np.float32)
    for node, color in (painted or {}).items():
        colors[node] = mcolors.to_rgba(color)
    return colors


def make_viewer(msa_path, headers, *, aligned=None, loaded=True, colors=None, hidden=None, clusters=None, key=None):
    """The little of a Viewer that Tree View reads: its alignment manager and its network's headers.

    aligned: which nodes of the network the alignment holds (all by default). loaded=False is an alignment that
    failed to load: its manager has a path and no rows. colors, hidden and clusters are what the Viewer holds of the
    nodes' colours (an array of RGBA), the nodes that are hidden (a list of node numbers) and their clusters
    (a list, -1 for none); left out, the Viewer has none of them. key is a function that declares what the colours
    mean, called with the viewer once it holds them (desktop.Node_Colors.declare_categories and declare_gradient).
    """
    mask = np.ones(len(headers), dtype=bool) if aligned is None else np.asarray(aligned, dtype=bool)
    alignment = SimpleNamespace(
        msa_file=None if msa_path is None else os.fspath(msa_path),
        aln=object() if loaded else None,
        aligned_node_mask=mask,
    )
    viewer = SimpleNamespace(alignment=alignment, full_headers=list(headers), n_nodes=len(headers))
    if colors is not None:
        viewer.current_colors = np.asarray(colors, dtype=np.float32)
        viewer.visible_mask = np.ones(len(headers), dtype=bool)
        viewer.visible_mask[list(hidden or [])] = False
    if clusters is not None:
        viewer.cluster_labels = np.asarray(clusters, dtype=np.int32)
    if key is not None:
        key(viewer)
    return viewer


def give_edges(viewer, pairs, scores, *, threshold=None, evalue=False):
    """Give a Viewer the edges the threshold tree is built from: an edge layer holding the real EdgeOrder (the edges
    by descending score), the slider's threshold, and the kind of score. Returns the viewer."""
    from desktop.Viewer_Edges import EdgeOrder

    scores = np.asarray(scores, dtype=np.float32)
    viewer.edge_layer = SimpleNamespace(order=EdgeOrder(np.asarray(pairs, dtype=np.int32), scores))
    viewer.min_threshold = float(scores.min()) if len(scores) else 0.0
    viewer.current_slider_threshold = viewer.min_threshold if threshold is None else threshold
    viewer.is_evalue = evalue
    return viewer


class TemporaryTreeFolder:
    """A temporary folder, removed with the test that made it (use with addCleanup)."""

    def __init__(self):
        self._folder = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self._folder.name)

    def cleanup(self):
        self._folder.cleanup()


# ---- The real web server over a Viewer that has only what the server and the tree backend read ----------------------

def http_get(server, path, headers=None):
    """GET path from the real server; returns (status, headers, body bytes)."""
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        for name, value in {"Host": f"127.0.0.1:{port}", **(headers or {})}.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def page_headers(server):
    """What Chromium sends for a same-origin fetch from a bundled page."""
    port = server.server_address[1]
    return {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}


def server_viewer():
    """The little a Viewer must have for the real web server to run over it.

    Its plugin registry holds the pages the bundled manifests declare (their policy, theme and texts); the test
    registers the routes of the plugin it serves.
    """
    from desktop.Viewer_Inspection import ViewerInspectionService
    from web_ui import Web_Server
    from web_ui.Plugin_Manager import WebPluginRegistry, register_bundled_pages

    viewer = SimpleNamespace(
        n_nodes=2, full_headers=["a", "b"], metadata={}, edges=[], visible_mask=[True, True], selected_indices=[],
        group_labels=[[], []], cluster_labels=None)
    viewer.web_plugin_registry = register_bundled_pages(WebPluginRegistry(viewer))
    viewer.viewer_inspection = ViewerInspectionService(viewer)
    viewer.communicator = Web_Server.QtCommunicator(viewer)
    viewer.get_initial_web_state = lambda: {"rows": []}
    viewer.handle_web_action = lambda data: None
    return viewer


# ---- A Viewer a snapshot can be taken of, holding the tree part --------------------------------------------------------

# A tree of six leaves and four clades, in post-order: A B (A B)=2 C D E (D E)=6 (C (D E))=7 F and the root 9 over 2, 7 and F.
SMALL_TREE = {
    "labels": list("ABCDEF"),
    "parent": [2, 2, 9, 7, 6, 6, 7, 9, 9, -1],
    "length": [1, 1, 2, 1, 1, 1, 2, 3, 5, 0],
    "support": [-1, -1, 90, -1, -1, -1, 70, 80, -1, -1],
}


def snapshot_viewer(headers, *, base=None, visible=None, clusters=None, metadata=None):
    """A Viewer with the little a snapshot reads, and Tree View's plugin registered in its registry.

    ``base`` is a Viewer from make_viewer to build on (its alignment and network are kept). Returns (viewer, service);
    ``service.capture_snapshot`` takes the snapshot.
    """
    from desktop.Viewer_Inspection import ViewerInspectionService
    from web_ui import tree_backend
    from web_ui.Plugin_Manager import WebPluginRegistry

    viewer = base if base is not None else make_viewer(None, headers)
    count = len(headers)
    viewer.visible_mask = np.ones(count, dtype=bool) if visible is None else np.asarray(visible, dtype=bool)
    viewer.cluster_labels = None if clusters is None else np.asarray(clusters)
    viewer.metadata = {} if metadata is None else metadata
    viewer.selected_indices, viewer.group_labels = [], None
    viewer.edges = getattr(viewer, "edges", [])
    viewer.current_slider_threshold = 0.5
    viewer.web_plugin_registry = WebPluginRegistry(viewer)
    tree_backend.register_backend(viewer.web_plugin_registry, viewer)
    service = ViewerInspectionService(viewer)
    viewer.viewer_inspection = service
    return viewer, service


def upload_tree(viewer, tree=None, **overrides):
    """Open a tree in the Viewer as the page's upload does (tree_backend.handle_tree_upload); returns the answer."""
    from web_ui import tree_backend

    data = {"name": "small.nwk", "format": "Newick", "rooted": True, **(SMALL_TREE if tree is None else tree)}
    data.update(overrides)
    return tree_backend.handle_tree_upload(viewer, data)
