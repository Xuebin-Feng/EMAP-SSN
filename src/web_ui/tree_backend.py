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

"""Tree View's side of the Viewer: the guide tree of the alignment that is loaded.

The page names no file. The tree is the one the Embedding MSA tool saved for the
alignment the Viewer has loaded (utilities/Guide_Tree_File.py maps an alignment's
path to its tree's), so the tree follows the alignment, and an alignment that did
not come from the tool has none. /api/tree/data answers with the tree as compact
arrays, how it matches the Viewer's network and the alignment, and, when there is
nothing to draw, why. The page puts every word of it in its own language.
"""

import base64
import datetime
import os
import threading

import numpy as np

from desktop.Desktop_App import translate
from utilities.Guide_Tree_File import GuideTreeError, guide_tree_path_for_alignment, read_guide_tree
from web_ui.Plugin_Manager import ensure_registry

DATA_ROUTE = "/api/tree/data"

# A tree is saved before the alignment is merged, so a finished alignment is as new as its tree, or newer. This
# many seconds of difference are the file system's and the clocks', not a sign that the alignment is older.
STALE_TOLERANCE_SECONDS = 2.0
# What of the tree's notes the page is sent: a handful of short lines about the run.
MAX_NOTES, MAX_NOTE_NAME, MAX_NOTE_VALUE = 40, 64, 600

# The tree last read, by the file's place, size and modification time, so that a page opened again, or the
# same page asking twice, does not parse the file again.
_TREE_CACHE = {"key": None, "tree": None}
_TREE_CACHE_LOCK = threading.Lock()


def loaded_alignment(viewer):
    """The Viewer's alignment manager and the path of its alignment file, or (None, None) when none loaded.

    An alignment that failed to load is not one: the tree follows the alignment the Viewer holds, and a
    failed one holds nothing.
    """
    alignment = getattr(viewer, "alignment", None)
    if alignment is None or getattr(alignment, "aln", None) is None:
        return None, None
    path = getattr(alignment, "msa_file", None)
    if not path or not str(path).strip():
        return None, None
    return alignment, os.fspath(path)


def _read_cached(path):
    """The guide tree in the file ``path``, parsed once per version of the file."""
    status = os.stat(path)
    key = (os.path.abspath(path), status.st_size, status.st_mtime_ns)
    with _TREE_CACHE_LOCK:
        if _TREE_CACHE["key"] == key:
            return _TREE_CACHE["tree"]
    tree = read_guide_tree(path)
    with _TREE_CACHE_LOCK:
        _TREE_CACHE["key"], _TREE_CACHE["tree"] = key, tree
    return tree


def alignment_as_fasta_name(name):
    """The FASTA name an alignment file stands for: the sparse form ``..._sparse.h5`` or ``.h5`` is of the ``.fasta``."""
    lowered = name.lower()
    if lowered.endswith("_sparse.h5"):
        return name[: -len("_sparse.h5")] + ".fasta"
    if lowered.endswith(".h5"):
        return name[: -len(".h5")] + ".fasta"
    return name


def _set_name_of_tree(tree_path):
    """The ``<set>_[<model>]`` part of a tree's file name, which the alignment it follows shares."""
    return os.path.basename(tree_path)[: -len("_guide_tree.nex")]


def written_time(text):
    """The seconds since the epoch of a tree's WRITTEN note (``2026-11-05 14:03:22 -0500``), or None."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    for candidate, pattern in ((text, "%Y-%m-%d %H:%M:%S %z"), (text[:19], "%Y-%m-%d %H:%M:%S")):
        try:
            return datetime.datetime.strptime(candidate, pattern).timestamp()
        except ValueError:
            continue
    return None


def _packed(array, dtype):
    """Base64 of the array's little-endian bytes: what the page reads as a typed array."""
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def consistency(tree, alignment, msa_path, tree_path, viewer_node, network_size):
    """How the tree fits the loaded alignment and the network: (counts, warnings).

    counts are plain numbers for the page to say. Each warning is a dict with a ``code`` and the values its
    sentence needs; a mismatch is shown as a warning and never stops the tree from being drawn.
    """
    warnings = []
    name = os.path.basename(msa_path)

    note = tree.notes.get("ALIGNMENT")
    if note is not None:
        if note.casefold() != alignment_as_fasta_name(name).casefold():
            warnings.append({"code": "alignment_differs", "tree": note, "loaded": name})
    else:
        # A tree saved before the tool recorded its alignment: its set and model, against the alignment's name.
        sequence_set, model = tree.notes.get("SEQUENCE_SET"), tree.notes.get("MODEL")
        if sequence_set is not None and model is not None:
            written, loaded = f"{sequence_set}_[{model}]", _set_name_of_tree(tree_path)
            if written.casefold() != loaded.casefold():
                warnings.append({"code": "name_differs", "tree": written, "loaded": loaded})

    tree_time = written_time(tree.notes.get("WRITTEN"))
    try:
        alignment_time = os.path.getmtime(msa_path)
    except OSError:
        alignment_time = None
    if tree_time is not None and alignment_time is not None and alignment_time < tree_time - STALE_TOLERANCE_SECONDS:
        warnings.append({"code": "alignment_older"})

    shown = viewer_node >= 0
    in_network = int(np.count_nonzero(shown))
    leaves = tree.leaf_count
    not_in_tree = int(network_size - len(np.unique(viewer_node[shown])))
    counts = {
        "leaves": leaves,
        "nodes": tree.node_count,
        "leaves_in_network": in_network,
        "leaves_not_in_network": leaves - in_network,
        "network_nodes": int(network_size),
        "network_nodes_not_in_tree": not_in_tree,
    }
    if counts["leaves_not_in_network"]:
        warnings.append({"code": "leaves_not_in_network", "count": counts["leaves_not_in_network"]})
    if not_in_tree:
        warnings.append({"code": "network_nodes_not_in_tree", "count": not_in_tree})
    aligned = getattr(alignment, "aligned_node_mask", None)
    if aligned is not None and len(aligned) == network_size and in_network:
        outside = int(in_network - np.count_nonzero(np.asarray(aligned, dtype=bool)[viewer_node[shown]]))
        counts["leaves_not_in_alignment"] = outside
        if outside:
            warnings.append({"code": "leaves_not_in_alignment", "count": outside})
    return counts, warnings


def tree_state(viewer):
    """What the Tree View page is sent: the tree of the loaded alignment, or why there is none to draw.

    ``status`` is "ok" (everything below is there), "no_alignment" (none loaded), "unnamed_alignment" (the
    alignment's name is not the tool's, so it has no tree), "no_tree" (the file the name leads to is not
    there), "unreadable" (it is there, but is not a tree this reader takes: ``reason``) or "no_overlap" (none
    of its leaves is a node of the network). The path of the expected file comes with every status that has one.
    """
    alignment, msa_path = loaded_alignment(viewer)
    if alignment is None:
        return {"status": "no_alignment"}
    name = os.path.basename(msa_path)
    tree_path = guide_tree_path_for_alignment(msa_path)
    if tree_path is None:
        return {"status": "unnamed_alignment", "alignment": name}
    shown_path = os.path.abspath(tree_path)
    base = {"alignment": name, "tree_path": shown_path}
    try:
        tree = _read_cached(tree_path)
    except FileNotFoundError:
        return {"status": "no_tree", **base}
    except GuideTreeError as error:
        return {"status": "unreadable", "reason": str(error), **base}
    except OSError as error:
        return {"status": "unreadable", "reason": error.strerror or str(error), **base}

    headers = list(getattr(viewer, "full_headers", ()) or ())
    node_of = {}
    for node, header in enumerate(headers):
        node_of.setdefault(str(header), node)
    viewer_node = np.fromiter((node_of.get(label, -1) for label in tree.labels), dtype=np.int32, count=tree.leaf_count)
    counts, warnings = consistency(tree, alignment, msa_path, tree_path, viewer_node, len(headers))
    if not counts["leaves_in_network"]:
        return {"status": "no_overlap", "counts": counts, **base}
    return {
        "status": "ok",
        **base,
        "name": tree.name,
        "rooted": tree.rooted,
        "has_lengths": tree.has_lengths,
        "notes": {
            str(key)[:MAX_NOTE_NAME]: str(value)[:MAX_NOTE_VALUE] for key, value in list(tree.notes.items())[:MAX_NOTES]
        },
        "counts": counts,
        "warnings": warnings,
        "parent": _packed(tree.parent, "<i4"),
        "length": _packed(tree.length, "<f4"),
        "support": _packed(tree.support, "i1"),
        "viewer_node": _packed(viewer_node, "<i4"),
        "labels": tree.labels,
    }


def handle_tree_data(viewer, request):
    """GET /api/tree/data: the page's whole picture of the tree (tree_state). The web server has checked the request's origin."""
    return 200, tree_state(viewer)


def register_backend(registry, viewer):
    """Register Tree View's web capabilities without adding a Console button."""
    registry.register_api_route("tree", DATA_ROUTE, lambda request: handle_tree_data(viewer, request))


def activate(viewer):
    """Add the Tree View button to the Console and keep it for the session, as the pages' buttons are."""
    if hasattr(viewer, "add_console_button"):
        viewer.add_console_button(
            name="treeViewBtn",
            label=translate("Viewer", "Tree View"),
            callback=lambda: viewer.open_page("tree"),
            tooltip=translate("Viewer", "Open the guide tree of the loaded alignment in browser"),
            icon="layers",
        )
        if not hasattr(viewer, "sidebar_buttons_to_persist"):
            viewer.sidebar_buttons_to_persist = []
        if "tree" not in viewer.sidebar_buttons_to_persist:
            viewer.sidebar_buttons_to_persist.append("tree")


def register(viewer):
    """Compatibility wrapper: ensure backend registration, then activate its UI."""
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("tree")
    return activate(viewer)
