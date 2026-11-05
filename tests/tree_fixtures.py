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


def make_viewer(msa_path, headers, *, aligned=None, loaded=True):
    """The little of a Viewer that Tree View reads: its alignment manager and its network's headers.

    aligned: which nodes of the network the alignment holds (all by default). loaded=False is an alignment that
    failed to load: its manager has a path and no rows.
    """
    mask = np.ones(len(headers), dtype=bool) if aligned is None else np.asarray(aligned, dtype=bool)
    alignment = SimpleNamespace(
        msa_file=None if msa_path is None else os.fspath(msa_path),
        aln=object() if loaded else None,
        aligned_node_mask=mask,
    )
    return SimpleNamespace(alignment=alignment, full_headers=list(headers), n_nodes=len(headers))


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
