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

"""Tree View's side of the Viewer: the guide tree of the alignment that is loaded, and the colours of the network.

The page names no file. The tree is the one the Embedding MSA tool saved for the
alignment the Viewer has loaded (utilities/Guide_Tree_File.py maps an alignment's
path to its tree's), so the tree follows the alignment, and an alignment that did
not come from the tool has none. /api/tree/data answers with the tree as compact
arrays, how it matches the Viewer's network and the alignment, and, when there is
nothing to draw, why. The page puts every word of it in its own language.

The leaves are drawn in the colours of the nodes they stand for. /api/tree/colors
answers with the colour of every node of the network, which nodes are hidden, and
the clusters; the page works out for itself what a clade looks like. Whatever
changes a colour, in whatever command, ends by drawing the nodes again, so one
generic listener on that (viewer.node_update_listeners) is all that follows them:
it tells a connected page, once per burst of changes, that the colours changed,
and the page asks for them again. No colour travels in the event stream, which
every open page receives.

The selection goes both ways. The Viewer already tells every page which nodes are
selected (selection_changed) and which has the click focus (highlight_row), and
/api/tree/selection gives a page that opens later what it missed. The other way,
the page asks the Viewer to select the nodes of a clade (the tree_select action)
or to focus on a leaf's node (tree_focus), and the Viewer does what a box
selection or a click does; the page learns what came of it from the same two events.
"""

import base64
import datetime
import json
import os
import threading
import time
import weakref
import zlib
from types import SimpleNamespace
from urllib.parse import parse_qs

import numpy as np

from desktop.Desktop_App import translate
from utilities.Guide_Tree_File import (
    MAX_LABEL_CHARACTERS, MAX_LEAVES, GuideTreeError, guide_tree_path_for_alignment, read_guide_tree,
)
from web_ui.Plugin_Manager import ensure_registry

DATA_ROUTE = "/api/tree/data"
COLORS_ROUTE = "/api/tree/colors"
SELECTION_ROUTE = "/api/tree/selection"
UPLOAD_ROUTE = "/api/tree/upload"
MISMATCH_ROUTE = "/api/tree/mismatch"
# The ways a tree_select request combines its nodes with the selection the network has.
SELECT_MODES = ("replace", "add", "toggle")
# The event that says the colours (or which nodes are hidden, or the clusters) are not what the page was sent.
COLORS_EVENT = "tree_colors_changed"
# A colour change is announced this long after the last one seen, so that a command that draws in a burst sends one event.
COLOR_COALESCE_SECONDS = 0.15

# The trees Tree View can show, and the one it shows when it is not asked for another: the guide tree when the loaded
# alignment has one, else the one built from the network's edges.
SOURCE_THRESHOLD = "threshold"
SOURCE_GUIDE = "guide"
SOURCE_UPLOADED = "uploaded"
SOURCES = (SOURCE_THRESHOLD, SOURCE_GUIDE, SOURCE_UPLOADED)

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


def locate_tree(viewer):
    """Where the tree of the loaded alignment should be, said without reading it.

    ``status`` is "no_alignment" (none loaded), "unnamed_alignment" (the alignment's name is not the tool's, so
    it has no tree; ``alignment`` names it), "no_tree" (the file the name leads to is not there) or "present"
    (something is there; whether it is a tree is for tree_state to find out). The last two come with
    ``alignment`` and ``tree_path``, the absolute path of the expected file.
    """
    alignment, msa_path = loaded_alignment(viewer)
    if alignment is None:
        return {"status": "no_alignment"}
    name = os.path.basename(msa_path)
    tree_path = guide_tree_path_for_alignment(msa_path)
    if tree_path is None:
        return {"status": "unnamed_alignment", "alignment": name}
    base = {"alignment": name, "tree_path": os.path.abspath(tree_path)}
    return {"status": "present" if os.path.exists(tree_path) else "no_tree", **base}


def guide_viewer_node(tree, headers):
    """The node of the network each leaf of the guide tree names, by exact full header, or -1 (an int32 array)."""
    node_of = {}
    for node, header in enumerate(headers):
        node_of.setdefault(str(header), node)
    return np.fromiter((node_of.get(label, -1) for label in tree.labels), dtype=np.int32, count=tree.leaf_count)


def guide_state(viewer):
    """What the Tree View page is sent for the guide tree: the tree of the loaded alignment, or why there is none to draw.

    ``status`` is "ok" (everything below is there), "no_alignment" (none loaded), "unnamed_alignment" (the
    alignment's name is not the tool's, so it has no tree), "no_tree" (the file the name leads to is not
    there), "unreadable" (it is there, but is not a tree this reader takes: ``reason``) or "no_overlap" (none
    of its leaves is a node of the network). The path of the expected file comes with every status that has one.
    """
    found = locate_tree(viewer)
    if found["status"] != "present":
        return found
    alignment, msa_path = loaded_alignment(viewer)
    tree_path = guide_tree_path_for_alignment(msa_path)
    base = {"alignment": found["alignment"], "tree_path": found["tree_path"]}
    try:
        tree = _read_cached(tree_path)
    except FileNotFoundError:
        return {"status": "no_tree", **base}
    except GuideTreeError as error:
        return {"status": "unreadable", "reason": str(error), **base}
    except OSError as error:
        return {"status": "unreadable", "reason": error.strerror or str(error), **base}

    headers = list(getattr(viewer, "full_headers", ()) or ())
    viewer_node = guide_viewer_node(tree, headers)
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


# ---- The threshold tree: single linkage over the network's edges --------------------------------------------------

# The threshold tree last built, kept for as long as the network's edge order is: it is the same object until the
# network is drawn again, so a new network (or a new score) builds a new tree. A weak reference, so that the tree
# does not keep a replaced network's edges alive.
_THRESHOLD_CACHE = {"order": None, "count": 0, "built": None}
_THRESHOLD_LOCK = threading.Lock()


def edge_order_of(viewer):
    """The Viewer's edges by descending score (desktop.Viewer_Edges.EdgeOrder), or None when it has no edges."""
    layer = getattr(viewer, "edge_layer", None)
    return getattr(layer, "order", None) if layer is not None else None


def threshold_unavailable(viewer):
    """Why the threshold tree cannot be built, as a status ("no_network", "no_edges"), or None when it can."""
    if not getattr(viewer, "full_headers", None):
        return "no_network"
    order = edge_order_of(viewer)
    if order is None or not len(order):
        return "no_edges"
    return None


def _kept_edges(viewer):
    """How the edges the network keeps were chosen: ``{"kept": "threshold", "cutoff"}`` for a layout that keeps the
    edges at or above a cutoff, or ``{"kept": "top_k", "k"}`` for one that keeps each node's best k (UMAP, PaCMAP)."""
    from desktop.Viewer_State import knn_topology_layout, strongest_edge_limit

    import EMAPSSN_Config as cfg

    if knn_topology_layout(cfg):
        return {"kept": "top_k", "k": int(strongest_edge_limit(cfg))}
    cutoff = getattr(viewer, "min_threshold", None)
    return {"kept": "threshold", "cutoff": None if cutoff is None else float(cutoff)}


def _build_threshold(order, headers):
    """The page's payload for the threshold tree of ``order`` and the facts about it, built once (threshold_state)."""
    from utilities.Threshold_Tree import threshold_tree

    started = time.perf_counter()
    count = len(headers)
    tree = threshold_tree(order.edges, order.scores, count, check=False)
    labels = [str(headers[node]) for node in tree.leaf_order]
    payload = {
        "status": "ok",
        "source": SOURCE_THRESHOLD,
        "name": None,
        "rooted": True,
        "has_lengths": True,
        "notes": {},
        "counts": {
            "leaves": count, "nodes": tree.node_count, "leaves_in_network": count, "leaves_not_in_network": 0,
            "network_nodes": count, "network_nodes_not_in_tree": 0,
        },
        "warnings": [],
        "parent": _packed(tree.parent, "<i4"),
        "length": _packed(tree.length, "<f4"),
        "support": _packed(np.full(tree.node_count, -1), "i1"),
        "viewer_node": _packed(tree.leaf_order, "<i4"),
        "score": _packed(tree.height, "<f4"),
        "labels": labels,
    }
    details = {
        "edges": int(len(order)),
        "groups": tree.groups,
        "top": tree.top,
        "bottom": tree.bottom,
        "root_score": tree.root_score,
        "forest": tree.groups > 1,
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    return payload, details


def threshold_state(viewer):
    """What the Tree View page is sent for the tree of the network's own edges: single linkage over them, strongest first.

    ``status`` is "ok", or "no_network" / "no_edges" when there is nothing to build it from. Built the first time it
    is asked for and kept for as long as the Viewer keeps that network's edge order. The leaves are the network's
    nodes, in the tree's leaf order (``viewer_node`` says which), and ``score`` is each tree node's height, the score
    at which its groups first share an edge, in the same post-order as ``parent``. ``threshold`` holds what the page
    says about the tree: the score the Viewer's slider is at now (``current``), the highest score of the edges
    (``top``, where the leaves stand) and the lowest join (``bottom``), whether the network falls into separate
    groups (``forest``, ``groups``; the forest's root is at ``root_score``), how the kept edges were chosen
    (``kept``, with ``cutoff`` or ``k``), the kind of score, and the seconds the build took.
    """
    reason = threshold_unavailable(viewer)
    if reason is not None:
        return {"status": reason, "source": SOURCE_THRESHOLD}
    order = edge_order_of(viewer)
    headers = list(viewer.full_headers)
    with _THRESHOLD_LOCK:
        cache = _THRESHOLD_CACHE
        reference = cache["order"]
        if reference is None or reference() is not order or cache["count"] != len(headers):
            cache["built"] = _build_threshold(order, headers)
            cache["order"], cache["count"] = weakref.ref(order), len(headers)
        payload, details = cache["built"]
    current = getattr(viewer, "current_slider_threshold", None)
    info = {
        **details,
        **_kept_edges(viewer),
        "score_kind": "blast" if getattr(viewer, "is_evalue", False) else "similarity",
        "current": None if current is None else float(current),
    }
    return {**payload, "threshold": info}


# ---- An uploaded tree: a tree the user opened in the page, matched to the network by full sequence name ----------------

# The most of the report's lists the page is sent: the labels that could not be matched uniquely, and the examples shown
# beside each other when nothing matched. The page has every label itself, so it lists the unmatched ones in full.
MAX_REPORTED_LABELS = 1000
MAX_EXAMPLES = 5
# What of a file's own words the Viewer keeps: its name, its format, a tree's name, and the notations it used.
MAX_TEXT, MAX_WARNINGS, MAX_DIALECTS = 260, 20, 20
# The notes of a refused upload; the page says each in its own words.
UPLOAD_REFUSALS = (
    "bad_request", "too_large", "no_leaves", "too_few_leaves", "too_many_leaves", "bad_labels", "empty_label", "label_too_long",
    "duplicate_label", "bad_arrays", "size_mismatch", "too_many_nodes", "not_post_order", "leaf_count_mismatch", "bad_length",
    "bad_support", "bad_rooted", "bad_text", "bad_warnings", "bad_clade_names",
)

_UPLOAD_LOCK = threading.Lock()


class UploadRefused(ValueError):
    """An upload that is not a tree the Viewer takes: ``note`` is one of UPLOAD_REFUSALS, ``detail`` the facts a sentence needs."""

    def __init__(self, note, **detail):
        super().__init__(note)
        self.note = note
        self.detail = detail


def _action_limit():
    """The largest request body the web server reads: an upload is no larger."""
    from web_ui import Web_Server

    return int(Web_Server.MAX_ACTION_BYTES)


def _text_of(value, *, limit=MAX_TEXT):
    """A file's own word (a name, a format) as one line of at most ``limit`` characters; a value that is no text refuses the upload."""
    if not isinstance(value, str):
        raise UploadRefused("bad_text")
    return "".join(" " if character.isspace() or ord(character) < 32 else character for character in value).strip()[:limit]


def _array_of(value, name, kinds, element_size):
    """The array a payload names ``name``: a list of numbers, or the base64 of their little-endian bytes (``element_size``
    bytes each). Returns a NumPy array; anything else, or a list holding anything but numbers of ``kinds``, refuses."""
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError):
            raise UploadRefused("bad_arrays", array=name) from None
        if len(raw) % element_size:
            raise UploadRefused("size_mismatch", array=name)
        dtype = {"parent": "<i4", "length": "<f4", "support": "i1"}[name]
        return np.frombuffer(raw, dtype=dtype)
    if not isinstance(value, list):
        raise UploadRefused("bad_arrays", array=name)
    try:
        array = np.asarray(value)
    except (ValueError, OverflowError):
        raise UploadRefused("bad_arrays", array=name) from None
    if array.ndim != 1 or array.dtype.kind not in kinds:
        # An empty list is a list of nothing, whatever dtype NumPy gives it.
        if not (array.ndim == 1 and array.size == 0):
            raise UploadRefused("bad_arrays", array=name)
    return array


def _clade_names_of(value, parent, internal):
    """The names of internal nodes: {node: name} from a dict (keys are node numbers, in text) or a list of [node, name]."""
    if value is None:
        return {}
    pairs = value.items() if isinstance(value, dict) else value
    if not isinstance(value, (dict, list)):
        raise UploadRefused("bad_clade_names")
    names = {}
    for pair in pairs:
        try:
            node, name = pair
            node = int(node) if isinstance(node, str) and node.lstrip("-").isdigit() else node
        except (TypeError, ValueError):
            raise UploadRefused("bad_clade_names") from None
        if not _is_node_number(node) or not 0 <= int(node) < len(parent) or not internal[int(node)] or not isinstance(name, str):
            raise UploadRefused("bad_clade_names")
        if len(name) > MAX_LABEL_CHARACTERS:
            raise UploadRefused("label_too_long", limit=MAX_LABEL_CHARACTERS)
        names[int(node)] = name
    return names


def validate_upload(data):
    """The tree a ``tree_upload`` request holds, checked from nothing (the page is never trusted), or UploadRefused.

    ``data`` is {"name", "labels", "parent", "length", "support", "rooted"} and, optionally, "format", "tree_name",
    "clade_names", "dialects", "support_name", "root_stated", "has_lengths" and "warnings" (each of the reader's {"code", "count"}).
    ``root_stated`` says whether the file said it (else the root's branches decided ``rooted``). ``parent``, ``length`` and ``support`` are lists of numbers or the base64 of their bytes ("<i4", "<f4", "i1"), one
    entry for each node of the tree. A tree is accepted when:
    its size is within the web server's limit for a request, and it has 2 to MAX_LEAVES leaves and at most twice as many
    nodes; every label is a text of 1 to MAX_LABEL_CHARACTERS characters and no two are the same; the three arrays have one
    entry for each node; the nodes are in post-order, every parent after its child and the last node the one root, so
    that the nodes form one tree and no cycle; the number of leaves (nodes with no child) is the number of labels; every
    length is finite, not negative and fits 4 bytes; every support is -1 or 0 to 100; and ``rooted`` is true, false or null.
    Returns a SimpleNamespace of what is kept: the texts, ``labels``, ``parent`` (int32), ``length`` (float32), ``support``
    (int8), ``rooted``, ``has_lengths``, ``clade_names``, ``warnings`` and ``fingerprint``.
    """
    leaf_limit, label_limit = MAX_LEAVES, MAX_LABEL_CHARACTERS
    if not isinstance(data, dict):
        raise UploadRefused("bad_request")
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise UploadRefused("bad_labels")
    if len(labels) > leaf_limit:
        raise UploadRefused("too_many_leaves", limit=leaf_limit)
    # What the request weighs, as near as can be told without writing it out again: the largest the server reads.
    weight = sum(len(label) + 4 for label in labels if isinstance(label, str))
    for key in ("parent", "length", "support"):
        value = data.get(key)
        weight += len(value) if isinstance(value, str) else 12 * len(value) if isinstance(value, list) else 0
    if weight > _action_limit():
        raise UploadRefused("too_large", limit=_action_limit())

    name = _text_of(data.get("name", ""))
    file_format = _text_of(data.get("format", ""), limit=40)
    tree_name = _text_of(data.get("tree_name", ""))
    support_name = _text_of(data.get("support_name", ""), limit=60)
    rooted = data.get("rooted")
    if rooted is not None and not isinstance(rooted, bool):
        raise UploadRefused("bad_rooted")
    root_stated = data.get("root_stated", False)
    if not isinstance(root_stated, bool):
        raise UploadRefused("bad_rooted")

    if len(labels) < 1:
        raise UploadRefused("no_leaves")
    if len(labels) < 2:
        raise UploadRefused("too_few_leaves")
    seen = {}
    for index, label in enumerate(labels):
        if not isinstance(label, str):
            raise UploadRefused("bad_labels")
        if not label:
            raise UploadRefused("empty_label", leaf=index)
        if len(label) > label_limit:
            raise UploadRefused("label_too_long", leaf=index, limit=label_limit)
        if label in seen:
            raise UploadRefused("duplicate_label", label=label[:120], first=seen[label], second=index)
        seen[label] = index

    parent = _array_of(data.get("parent"), "parent", "iu", 4)
    length = _array_of(data.get("length"), "length", "iuf", 4)
    support = _array_of(data.get("support"), "support", "iu", 1)
    count = len(parent)
    if len(length) != count or len(support) != count:
        raise UploadRefused("size_mismatch", nodes=count, lengths=len(length), supports=len(support))
    if count > 2 * leaf_limit:
        raise UploadRefused("too_many_nodes", limit=2 * leaf_limit)
    if count < len(labels):
        raise UploadRefused("leaf_count_mismatch", leaves=len(labels), found=0)

    parent = parent.astype(np.int64, copy=False)
    # Post-order: every node but the last has a parent that comes after it, and the last has none. So following parents
    # only ever goes up the numbers, ends at the last node, and there is one root and no cycle.
    if parent[-1] != -1:
        raise UploadRefused("not_post_order", node=count - 1)
    if count > 1:
        above = parent[:-1]
        bad = np.flatnonzero((above <= np.arange(count - 1)) | (above >= count))
        if bad.size:
            raise UploadRefused("not_post_order", node=int(bad[0]))
    child_counts = np.bincount(parent[:-1], minlength=count) if count > 1 else np.zeros(count, dtype=np.int64)
    leaves = int(np.count_nonzero(child_counts == 0))
    if leaves != len(labels):
        raise UploadRefused("leaf_count_mismatch", leaves=len(labels), found=leaves)

    length = length.astype(np.float64, copy=False)
    if not np.all(np.isfinite(length)) or np.any(length < 0) or np.any(length > np.finfo(np.float32).max):
        raise UploadRefused("bad_length")
    support = support.astype(np.int64, copy=False)
    if np.any(support < -1) or np.any(support > 100):
        raise UploadRefused("bad_support")

    dialects = data.get("dialects", [])
    if not isinstance(dialects, list) or len(dialects) > MAX_DIALECTS:
        raise UploadRefused("bad_text")
    dialects = [_text_of(item, limit=40) for item in dialects]
    warnings = data.get("warnings", [])
    if not isinstance(warnings, list) or len(warnings) > MAX_WARNINGS:
        raise UploadRefused("bad_warnings")
    kept = []
    for warning in warnings:
        if not isinstance(warning, dict) or not isinstance(warning.get("code"), str) or len(warning["code"]) > 40:
            raise UploadRefused("bad_warnings")
        number = warning.get("count")
        if number is not None and not (_is_node_number(number) and 0 <= number <= 2**31):
            raise UploadRefused("bad_warnings")
        kept.append({"code": warning["code"]} if number is None else {"code": warning["code"], "count": int(number)})
    has_lengths = data.get("has_lengths")
    if has_lengths is None:
        has_lengths = bool(np.any(length > 0))
    elif not isinstance(has_lengths, bool):
        raise UploadRefused("bad_text")
    clade_names = _clade_names_of(data.get("clade_names"), parent, child_counts > 0)

    parent32, length32, support8 = parent.astype("<i4"), length.astype("<f4"), support.astype("i1")
    fingerprint = zlib.crc32(parent32.tobytes())
    fingerprint = zlib.crc32(length32.tobytes(), fingerprint)
    fingerprint = zlib.crc32(support8.tobytes(), fingerprint)
    fingerprint = zlib.crc32("\n".join(labels).encode("utf-8", "surrogatepass"), fingerprint)
    fingerprint = zlib.crc32(repr((name, file_format, tree_name, rooted, root_stated, has_lengths)).encode("utf-8", "surrogatepass"), fingerprint)
    return SimpleNamespace(
        name=name, format=file_format, tree_name=tree_name, support_name=support_name, dialects=dialects, warnings=kept,
        labels=labels, parent=parent32, length=length32, support=support8, rooted=rooted, root_stated=root_stated,
        has_lengths=has_lengths, clade_names=clade_names, fingerprint=fingerprint,
    )


def match_labels(labels, headers):
    """Which node of the network each leaf label names, by full sequence name: (viewer_node, report). See ``_match``."""
    viewer_node, report, _ = _match(labels, headers)
    return viewer_node, report


def _match(labels, headers):
    """Which node of the network each leaf label names, by full sequence name: (viewer_node, report).

    A label names a node when (1) it equals the node's header exactly, else (2) the label after the Viewer's own
    sanitize_header (the rules the headers were written by) equals the header. Nothing else: no accession, no case
    folding, no underscore for space. A node can be one leaf's: when two labels reach the same header by step 2, or one
    reaches by step 2 a header a label already has exactly, none of them is matched and they are listed (``ambiguous``).
    ``viewer_node`` holds the node number for each label or -1. ``report`` counts what happened: ``matched``, ``exact``,
    ``sanitized``, ``unmatched`` (leaves of no node, ambiguous ones among them) and ``ambiguous`` with ``ambiguous_labels``
    (at most MAX_REPORTED_LABELS), and ``network_nodes`` with ``network_nodes_not_in_tree``. The third value is the
    ambiguous leaves' numbers, all of them and ascending.
    """
    from utilities.Sequence_Utils import sanitize_header

    node_of = {}
    for node, header in enumerate(headers):
        node_of.setdefault(str(header), node)
    count = len(labels)
    viewer_node = np.full(count, -1, dtype=np.int32)
    exact = 0
    pending = []
    for leaf, label in enumerate(labels):
        node = node_of.get(label)
        if node is None:
            pending.append(leaf)
        else:
            viewer_node[leaf] = node
            exact += 1
    taken = set(viewer_node[viewer_node >= 0].tolist())
    reached = {}
    for leaf in pending:
        node = node_of.get(sanitize_header(labels[leaf])[0])
        if node is not None:
            reached.setdefault(node, []).append(leaf)
    sanitized, ambiguous = 0, []
    for node, leaves in reached.items():
        if len(leaves) > 1 or node in taken:
            ambiguous.extend(leaves)
        else:
            viewer_node[leaves[0]] = node
            sanitized += 1
    ambiguous.sort()
    matched = exact + sanitized
    report = {
        "leaves": count,
        "matched": matched,
        "exact": exact,
        "sanitized": sanitized,
        "unmatched": count - matched,
        "ambiguous": len(ambiguous),
        "ambiguous_labels": [labels[leaf][:MAX_LABEL_CHARACTERS] for leaf in ambiguous[:MAX_REPORTED_LABELS]],
        "network_nodes": len(headers),
        "network_nodes_not_in_tree": len(headers) - len(set(viewer_node[viewer_node >= 0].tolist())),
    }
    return viewer_node, report, ambiguous


def stored_upload(viewer):
    """The tree the user opened in the page, as validate_upload kept it, or None; kept for the Viewer's session."""
    with _UPLOAD_LOCK:
        return getattr(viewer, "_tree_upload", None)


def uploaded_state(viewer):
    """What the Tree View page is sent for the tree the user opened: the tree and how it matches the network, or why not.

    ``status`` is "ok", "no_upload" (none was opened) or "no_overlap" (none of its leaves is a node of the network: the
    answer has the report and a few labels beside a few headers, so that a renamed export is easy to see). The match is
    made against the network the Viewer holds when it is asked, by match_labels. ``uploaded`` holds what the page says
    of the tree: its file, format, tree name, notations, the support's name, and the number of names of clades; ``match``
    the report; ``warnings`` the reader's, then those of the match.
    """
    upload = stored_upload(viewer)
    if upload is None:
        return {"status": "no_upload", "source": SOURCE_UPLOADED}
    headers = list(getattr(viewer, "full_headers", ()) or ())
    viewer_node, report = match_labels(upload.labels, headers)
    leaves = len(upload.labels)
    counts = {
        "leaves": leaves,
        "nodes": int(len(upload.parent)),
        "leaves_in_network": report["matched"],
        "leaves_not_in_network": report["unmatched"],
        "network_nodes": len(headers),
        "network_nodes_not_in_tree": report["network_nodes_not_in_tree"],
    }
    info = {
        "file": upload.name,
        "format": upload.format,
        "tree_name": upload.tree_name,
        "dialects": list(upload.dialects),
        "root_stated": upload.root_stated,
        "support_name": upload.support_name,
        "clade_names": len(upload.clade_names),
    }
    base = {"source": SOURCE_UPLOADED, "counts": counts, "match": report, "uploaded": info}
    if not report["matched"]:
        report = {**report, "tree_examples": list(upload.labels[:MAX_EXAMPLES]), "network_examples": [str(h) for h in headers[:MAX_EXAMPLES]]}
        return {"status": "no_overlap", **base, "match": report}
    warnings = [dict(item) for item in upload.warnings]
    if report["unmatched"]:
        warnings.append({"code": "leaves_not_in_network", "count": report["unmatched"]})
    if report["ambiguous"]:
        warnings.append({"code": "ambiguous_names", "count": report["ambiguous"]})
    if report["network_nodes_not_in_tree"]:
        warnings.append({"code": "network_nodes_not_in_tree", "count": report["network_nodes_not_in_tree"]})
    return {
        "status": "ok",
        **base,
        "name": upload.tree_name or None,
        "rooted": upload.rooted,
        "has_lengths": upload.has_lengths,
        "notes": {},
        "warnings": warnings,
        "parent": _packed(upload.parent, "<i4"),
        "length": _packed(upload.length, "<f4"),
        "support": _packed(upload.support, "i1"),
        "viewer_node": _packed(viewer_node, "<i4"),
        "labels": upload.labels,
    }


def handle_tree_upload(viewer, data):
    """The ``tree_upload`` action: keep a tree the user opened in the page, and make it the tree Tree View shows.

    The tree is checked from nothing (validate_upload) and nothing is changed unless all of it is well formed: a
    refusal answers {"ok": False, "changed": False, "note": <one of UPLOAD_REFUSALS>, "detail": {...}} and leaves the tree
    that was kept, and the choice of source, as they were. A tree is kept in the Viewer's memory only, for as long as the
    Viewer runs, and replaces the one before. The same tree again is no change: it succeeds with the note "unchanged",
    and is shown all the same. Else {"ok": True, "changed": True, "note": None}, with ``status`` ("ok", or "no_overlap"
    when no leaf is a node of the network) and ``counts``, ``match`` of uploaded_state.
    """
    try:
        upload = validate_upload(data)
    except UploadRefused as refusal:
        return {"ok": False, "changed": False, "note": refusal.note, "detail": refusal.detail}
    with _UPLOAD_LOCK:
        before = getattr(viewer, "_tree_upload", None)
        unchanged = before is not None and before.fingerprint == upload.fingerprint
        if not unchanged:
            viewer._tree_upload = upload
        viewer._tree_source = SOURCE_UPLOADED
    state = uploaded_state(viewer)
    return {
        "ok": True,
        "changed": not unchanged,
        "note": "unchanged" if unchanged else None,
        "status": state["status"],
        "counts": state["counts"],
        "match": state["match"],
    }


def handle_tree_upload_request(viewer, request):
    """POST /api/tree/upload: the ``tree_upload`` action answered at once, with the tree it made the page's (``state``).

    An action is answered with nothing but its acceptance; the page that sends a tree needs to know what came of it.
    The web server has checked the request's origin, its size and its Content-Type.
    """
    try:
        data = json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return 200, {"ok": False, "changed": False, "note": "bad_request", "detail": {}}
    result = handle_tree_upload(viewer, data)
    if result["ok"]:
        result["state"] = tree_state(viewer, SOURCE_UPLOADED)
    return 200, result


# ---- Which tree is shown ----------------------------------------------------------------------------------------


def source_choices(viewer, guide):
    """The trees the page's chooser offers, each ``{"id", "available", "reason"}``: ``reason`` is the status the
    tree cannot be shown for (the same words an empty card says), None when it can. ``guide`` is guide_state(viewer).
    The tree the user opened in the page, once there is one, is a third entry, with the name of its file (``name``)."""
    threshold_reason = threshold_unavailable(viewer)
    guide_reason = None if guide.get("status") == "ok" else guide.get("status")
    choices = [
        {"id": SOURCE_THRESHOLD, "available": threshold_reason is None, "reason": threshold_reason},
        {"id": SOURCE_GUIDE, "available": guide_reason is None, "reason": guide_reason},
    ]
    upload = stored_upload(viewer)
    if upload is not None:
        choices.append({"id": SOURCE_UPLOADED, "available": True, "reason": None, "name": upload.name})
    return choices


def default_source(kept, available):
    """The tree the page opens on when it asks for none: the source the Viewer kept for the session while it can be
    shown, else the guide tree when it can, else the threshold tree; with none to show, the guide tree (whose card says
    why). ``available`` is the set of sources that can be shown."""
    if kept in available:
        return kept
    if SOURCE_GUIDE in available:
        return SOURCE_GUIDE
    if SOURCE_THRESHOLD in available:
        return SOURCE_THRESHOLD
    return SOURCE_GUIDE


def opening_source(viewer):
    """Which tree the page opens on now, said without reading any tree file: ``(source, guide, threshold_reason)``.

    ``source`` is what default_source chooses. ``guide`` is locate_tree(viewer): the guide tree counts as available
    when its file is there, whether it can be read is for the page to say. ``threshold_reason`` is
    threshold_unavailable(viewer) ("no_network", "no_edges" or None).
    """
    guide = locate_tree(viewer)
    threshold_reason = threshold_unavailable(viewer)
    available = set()
    if guide["status"] == "present":
        available.add(SOURCE_GUIDE)
    if threshold_reason is None:
        available.add(SOURCE_THRESHOLD)
    if stored_upload(viewer) is not None:
        available.add(SOURCE_UPLOADED)
    return default_source(getattr(viewer, "_tree_source", None), available), guide, threshold_reason


def tree_state(viewer, source=None):
    """What the Tree View page is sent: one tree, or why there is none to draw, and the trees it can choose from.

    ``source`` is "threshold" (the tree of the network's own edges), "guide" (the guide tree of the loaded
    alignment), "uploaded" (the tree the user opened in the page) or None. A source that is asked for is the one the
    Viewer keeps for the session, when it can be shown; without one, the kept source is used while it can be shown,
    else the guide tree when the loaded alignment has one, else the threshold tree. The answer says which it is
    (``source``) and lists them (``sources``). A source that is asked for and cannot be shown gets the card that says
    why, and is not kept. An unknown source raises ValueError, which the web server answers as a bad request.
    """
    if source is not None and source not in SOURCES:
        raise ValueError(translate("Viewer", "Tree View does not know a tree called {name}.").format(name=str(source)[:60]))
    guide = guide_state(viewer)
    sources = source_choices(viewer, guide)
    available = {entry["id"] for entry in sources if entry["available"]}
    if source is not None:
        chosen = source
        if source in available:
            viewer._tree_source = source
    else:
        chosen = default_source(getattr(viewer, "_tree_source", None), available)
    if chosen == SOURCE_THRESHOLD:
        answer = threshold_state(viewer)
    elif chosen == SOURCE_UPLOADED:
        answer = uploaded_state(viewer)
    else:
        answer = {**guide, "source": SOURCE_GUIDE}
    return {**answer, "sources": sources}


def handle_tree_data(viewer, request):
    """GET /api/tree/data[?source=threshold|guide|uploaded]: the page's whole picture of the tree (tree_state). The web server has checked the request's origin."""
    values = parse_qs(request.query or "", keep_blank_values=True).get("source", [])
    return 200, tree_state(viewer, values[0] if values else None)


# ---- Which leaves and nodes the warnings count -----------------------------------------------------------------------

# The lists of mismatch_state, in the order the page shows them; each is what the warning of the same name counts.
MISMATCH_LISTS = ("leaves_not_in_network", "network_nodes_not_in_tree", "leaves_not_in_alignment", "ambiguous_names")


def _names_of(names):
    """One list of the mismatch answer: how many, and the names themselves."""
    names = list(names)
    return {"count": len(names), "names": names}


def _nodes_not_named(headers, viewer_node):
    """The network's full headers of the nodes no leaf is matched to, in node order."""
    named = np.zeros(len(headers), dtype=bool)
    named[viewer_node[viewer_node >= 0]] = True
    return [headers[node] for node in np.flatnonzero(~named).tolist()]


def mismatch_state(viewer, source):
    """The leaves and nodes the tree's warnings count, by their full names, from the data tree_state reads.

    ``source`` is "threshold", "guide" or "uploaded" (anything else raises ValueError, which the web server answers as a
    bad request). The answer holds ``source``, ``status`` ("ok", or the status the tree cannot be shown for, with empty
    lists) and four lists, each ``{"count", "names"}`` and none longer than the tree or the network:
    ``leaves_not_in_network`` (the tree's labels that name no node), ``network_nodes_not_in_tree`` (the full headers of
    the nodes no leaf names), ``leaves_not_in_alignment`` (labels of leaves whose node has no sequence in the loaded
    alignment: the guide tree only) and ``ambiguous_names`` (labels whose name reaches a node another leaf has too: an
    opened tree only). The threshold tree has every node as a leaf, so all its lists are empty.
    """
    if source not in SOURCES:
        raise ValueError(translate("Viewer", "Tree View does not know a tree called {name}.").format(name=str(source)[:60]))
    lists = {key: [] for key in MISMATCH_LISTS}
    status = "ok"
    headers = [str(header) for header in (getattr(viewer, "full_headers", ()) or ())]
    if source == SOURCE_UPLOADED:
        upload = stored_upload(viewer)
        if upload is None:
            status = "no_upload"
        else:
            viewer_node, _, ambiguous = _match(upload.labels, headers)
            labels = upload.labels
            lists["leaves_not_in_network"] = [labels[leaf] for leaf in np.flatnonzero(viewer_node < 0).tolist()]
            lists["ambiguous_names"] = [labels[leaf] for leaf in ambiguous]
            lists["network_nodes_not_in_tree"] = _nodes_not_named(headers, viewer_node)
    elif source == SOURCE_GUIDE:
        status = locate_tree(viewer)["status"]
        if status == "present":
            alignment, msa_path = loaded_alignment(viewer)
            try:
                tree = _read_cached(guide_tree_path_for_alignment(msa_path))
            except FileNotFoundError:
                status = "no_tree"
            except (GuideTreeError, OSError):
                status = "unreadable"
            else:
                status = "ok"
                labels = tree.labels
                viewer_node = guide_viewer_node(tree, headers)
                lists["leaves_not_in_network"] = [labels[leaf] for leaf in np.flatnonzero(viewer_node < 0).tolist()]
                lists["network_nodes_not_in_tree"] = _nodes_not_named(headers, viewer_node)
                aligned = getattr(alignment, "aligned_node_mask", None)
                shown = viewer_node >= 0
                if aligned is not None and len(aligned) == len(headers) and shown.any():
                    unaligned = np.zeros(len(labels), dtype=bool)
                    unaligned[shown] = ~np.asarray(aligned, dtype=bool)[viewer_node[shown]]
                    lists["leaves_not_in_alignment"] = [labels[leaf] for leaf in np.flatnonzero(unaligned).tolist()]
    else:
        status = threshold_unavailable(viewer) or "ok"
    return {"source": source, "status": status, **{key: _names_of(lists[key]) for key in MISMATCH_LISTS}}


def handle_tree_mismatch(viewer, request):
    """GET /api/tree/mismatch?source=threshold|guide|uploaded: the names the tree's warnings count (mismatch_state).

    ``source`` must be given once; a missing, repeated or unknown one raises ValueError, which the web server answers as a
    bad request. The web server has checked the request's origin.
    """
    values = parse_qs(request.query or "", keep_blank_values=True).get("source", [])
    if len(values) != 1:
        raise ValueError(translate("Viewer", "Tree View needs the name of one tree."))
    return 200, mismatch_state(viewer, values[0])


# ---- The colours of the network ---------------------------------------------------------------------------------


def _plain_color_bytes():
    """The red, green, blue and alpha (0 to 255) a node has until something colours it: the Viewer's initial node colour."""
    from matplotlib import colors as mcolors

    import EMAPSSN_Config as cfg

    return [int(round(value * 255)) for value in mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)]


def _color_bytes(name):
    """The red, green, blue and alpha (0 to 255) of the Viewer's colour setting ``name``, read when asked."""
    from matplotlib import colors as mcolors

    import EMAPSSN_Config as cfg

    return [int(round(value * 255)) for value in mcolors.to_rgba(getattr(cfg, name))]


def connected_identification_on(viewer):
    """Whether the Viewer outlines the neighbours of the selected nodes: its own rule, asked of it.

    The Viewer does so while the node boundary colour differs from the connected node colour, and that is
    all this says; a Viewer that cannot say is taken not to.
    """
    rule = getattr(viewer, "_connected_node_identification_enabled", None)
    if not callable(rule):
        return False
    try:
        return bool(rule())
    except Exception:
        return False


def marks_state(viewer):
    """The colours the Viewer outlines the nodes it has selected in (``selected``), and their neighbours in
    (``connected``), each as red, green, blue and alpha (0 to 255), read from its settings when asked; and
    ``connected_on``, whether it outlines the neighbours at all (connected_identification_on).
    """
    return {
        "selected": _color_bytes("HOVER_COLOR"),
        "connected": _color_bytes("CONNECTED_NODE_COLOR"),
        "connected_on": connected_identification_on(viewer),
    }


def colors_snapshot(viewer):
    """What the page is told of the nodes' colours, as arrays, and the version that names it; None when the Viewer has none.

    A copy of what the Viewer holds, taken at once: the Viewer changes its arrays on its own thread. The version is a
    checksum of everything that is sent, so two states that look alike have one version, whoever asks, and a page
    that holds a version asks for nothing when it hears of the same one again. Colours are as the nodes are drawn,
    without the hover colour (the Viewer paints that on a copy).
    """
    colors = getattr(viewer, "current_colors", None)
    count = int(getattr(viewer, "n_nodes", 0) or 0)
    if colors is None or count < 1:
        return None
    colors = np.array(colors, dtype=np.float64, copy=True)
    if colors.shape != (count, 4):
        return None
    rgba = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    mask = getattr(viewer, "visible_mask", None)
    shown = np.ones(count, dtype=bool) if mask is None or len(mask) != count else np.array(mask, dtype=bool, copy=True)
    hidden = np.packbits(~shown, bitorder="little")
    labels = getattr(viewer, "cluster_labels", None)
    cluster = None
    if labels is not None and len(labels) == count:
        cluster = np.array(labels, dtype=np.int64, copy=True)
        cluster = np.clip(cluster, -(2**31), 2**31 - 1).astype(np.int32)
    marks = marks_state(viewer)
    version = zlib.crc32(rgba.tobytes())
    version = zlib.crc32(hidden.tobytes(), version)
    version = zlib.crc32(repr(sorted(marks.items())).encode("ascii"), version)
    if cluster is not None:
        version = zlib.crc32(cluster.tobytes(), version)
    return {
        "count": count, "rgba": rgba, "hidden": hidden, "cluster": cluster, "marks": marks, "version": version & 0xFFFFFFFF,
    }


def colors_state(viewer):
    """What /api/tree/colors answers: ``status`` "ok" with the colours of every node of the network, or "no_colors".

    ``rgba`` holds 4 bytes a node, ``hidden`` one bit a node (set when the node is hidden, least significant bit
    first), ``cluster`` the cluster number of each node (-1 for none) when the Viewer has clusters, in 2 bytes each
    (``cluster_type`` "i2") unless one of them needs 4 ("i4"), ``plain`` the colour of a node nothing has coloured, and
    ``marks`` the colours the selection and its neighbours are outlined in (marks_state), which the answer for no
    colours has too.
    The nodes are the Viewer's, in its numbering; the tree's leaves find theirs in the viewer_node array of
    /api/tree/data.
    """
    found = colors_snapshot(viewer)
    if found is None:
        return {"status": "no_colors", "version": 0, "marks": marks_state(viewer)}
    cluster = found["cluster"]
    small = cluster is not None and (int(cluster.min()) >= -(2**15) and int(cluster.max()) < 2**15)
    return {
        "status": "ok",
        "version": found["version"],
        "count": found["count"],
        "plain": _plain_color_bytes(),
        "marks": found["marks"],
        "rgba": _packed(found["rgba"], "u1"),
        "hidden": _packed(found["hidden"], "u1"),
        "cluster": None if cluster is None else _packed(cluster, "<i2" if small else "<i4"),
        "cluster_type": None if cluster is None else ("i2" if small else "i4"),
    }


class ColorWatch:
    """Tells a connected Tree View page that the colours changed: the viewer's node update listener.

    It runs each time the nodes are drawn, so it costs nothing while no Tree View page is connected, and else only
    the checksum of the colours. A change is announced once, a short while after the last of a burst; the event
    carries the new version, and the page asks /api/tree/colors for what it is. The hover colour is not part of
    the colours, so a pointer moving over the nodes announces nothing.
    """

    def __init__(self, viewer, *, delay=COLOR_COALESCE_SECONDS, timer_factory=threading.Timer):
        self.viewer = viewer
        self.delay = delay
        self.timer_factory = timer_factory
        self.latest = None
        self.announced = None
        self._timer = None
        self._lock = threading.Lock()

    def __call__(self):
        server = getattr(self.viewer, "web_server", None)
        has_client = getattr(server, "has_event_client", None)
        if not callable(has_client) or not has_client("tree"):
            return
        found = colors_snapshot(self.viewer)
        version = 0 if found is None else found["version"]
        with self._lock:
            if version == self.latest:
                return
            self.latest = version
            if self._timer is not None:
                return
            self._timer = self.timer_factory(self.delay, self._announce)
            self._timer.daemon = True
            self._timer.start()

    def served(self, version):
        """The page was sent this version: nothing is owed to it until the colours are something else."""
        with self._lock:
            self.latest = self.announced = version

    def _announce(self):
        with self._lock:
            self._timer = None
            version = self.latest
            if version == self.announced:
                return
            self.announced = version
        self.viewer.broadcast_event({"type": COLORS_EVENT, "version": version})


def handle_tree_colors(viewer, request):
    """GET /api/tree/colors: the colours of the network's nodes (colors_state). The web server has checked the request's origin."""
    state = colors_state(viewer)
    watch = getattr(viewer, "_tree_color_watch", None)
    if watch is not None:
        watch.served(state["version"])
    return 200, state


# ---- The selection, both ways -----------------------------------------------------------------------------------


def _is_node_number(value):
    """Whether value is a whole number a JSON request can carry: an int, never a boolean."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


# The most neighbour entries gathered at once: a selection of every node of a dense network is read in pieces.
NEIGHBOUR_CHUNK = 4_000_000


def neighbours_of_nodes(viewer, nodes):
    """The nodes next to ``nodes`` in the Viewer's whole network (not ``nodes`` themselves), ascending, as 4-byte integers.

    The Viewer's own answer (_connected_to_selected_indices) is kept up to date by the Viewer's thread as the
    selection changes, and this runs on the web server's, so it counts afresh from the Viewer's adjacency index
    instead of sharing what that thread changes: the cost of the summed degree of ``nodes``, read in pieces. A Viewer
    that has no adjacency index has no neighbours to name.
    """
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    index = getattr(viewer, "_adjacency_index", None)
    if not nodes.size or not callable(index):
        return np.empty(0, dtype=np.int32)
    indptr, indices = index()
    count = len(indptr) - 1
    nodes = nodes[(nodes >= 0) & (nodes < count)]
    if not nodes.size:
        return np.empty(0, dtype=np.int32)
    starts = indptr[nodes]
    degrees = indptr[nodes + 1] - starts
    ends = np.cumsum(degrees)
    reached = np.zeros(count, dtype=bool)
    first = 0
    while first < nodes.size:
        before = int(ends[first - 1]) if first else 0
        last = max(first + 1, int(np.searchsorted(ends, before + NEIGHBOUR_CHUNK, side="right")))
        piece = degrees[first:last]
        total = int(piece.sum())
        if total:
            # Entry j of node g reads indices[starts[g] + j]: every node's slice, side by side.
            offsets = np.cumsum(piece) - piece
            gather = np.arange(total, dtype=np.int64) + np.repeat(starts[first:last] - offsets, piece)
            reached[indices[gather]] = True
        first = last
    reached[nodes] = False
    return np.flatnonzero(reached).astype(np.int32)


def selection_state(viewer):
    """What /api/tree/selection answers: the nodes selected in the network, the ones next to them, and the click focus.

    ``indices`` are the Viewer's node numbers, sorted and without repeats, as 4-byte integers (``count`` of them);
    ``focus`` is the node the network's click focus is on, or None. ``connected_on`` is whether the Viewer outlines the
    neighbours of the selected nodes (connected_identification_on); only then, and with a selection, ``connected``
    holds those neighbours as ``indices`` does (``connected_count`` of them, the selected nodes not among them): the
    answer is the nodes, never the edges between them. A page asks for this when it opens and when the selection
    changes, as it hears of the selection through selection_changed and highlight_row.
    """
    count = int(getattr(viewer, "n_nodes", 0) or 0)
    selected = getattr(viewer, "selected_indices", None)
    found = np.zeros(0, dtype=np.int64)
    if selected is not None and len(selected) and count > 0:
        found = np.unique(np.asarray(list(selected), dtype=np.int64))
        found = found[(found >= 0) & (found < count)]
    focus = getattr(viewer, "selected_node_idx", None)
    if not _is_node_number(focus) or not 0 <= int(focus) < count:
        focus = None
    connected_on = connected_identification_on(viewer)
    connected = neighbours_of_nodes(viewer, found) if connected_on and found.size else np.empty(0, dtype=np.int32)
    return {
        "status": "ok",
        "count": int(found.size),
        "indices": _packed(found, "<i4"),
        "focus": None if focus is None else int(focus),
        "connected_on": connected_on,
        "connected_count": int(connected.size),
        "connected": _packed(connected, "<i4"),
    }


def handle_tree_selection(viewer, request):
    """GET /api/tree/selection: the network's selection and focus (selection_state). The web server has checked the request's origin."""
    return 200, selection_state(viewer)


def handle_tree_select(viewer, data):
    """The ``tree_select`` action: select nodes of the network from the tree, as a box selection does.

    ``indices`` is a list of the Viewer's node numbers, and ``mode`` says what is done with the selection the
    network has: "replace" (the nodes become the selection, an empty list clearing it), "add" (they join it) or
    "toggle" (those selected leave it, the others join it). A request that is not well formed changes nothing:
    ``indices`` not a list, longer than the network, or holding anything but whole numbers (a boolean is not one) in
    0..n-1, or another ``mode``. Nodes that are hidden cannot be selected and are left out, as are repeats; when that
    leaves nothing of a request that named some, or the selection comes out as it was, nothing is changed and the
    request succeeds with a note. Returns {"ok", "changed", "selected", "note"}; the page learns the new selection
    from the Viewer's selection_changed event, not from this.
    """
    def refuse(note):
        return {"ok": False, "changed": False, "selected": None, "note": note}

    indices = data.get("indices") if isinstance(data, dict) else None
    mode = data.get("mode", "replace") if isinstance(data, dict) else None
    if mode not in SELECT_MODES:
        return refuse("bad_mode")
    count = int(getattr(viewer, "n_nodes", 0) or 0)
    if not isinstance(indices, list):
        return refuse("bad_indices")
    if len(indices) > count:
        return refuse("too_many")
    for value in indices:
        if not _is_node_number(value) or not 0 <= int(value) < count:
            return refuse("bad_indices")
    visible = getattr(viewer, "visible_mask", None)
    if visible is None or len(visible) != count:
        return refuse("no_network")
    requested = np.unique(np.asarray(indices, dtype=np.int64))
    wanted = requested[np.asarray(visible, dtype=bool)[requested]] if requested.size else requested

    current = np.unique(np.asarray(list(getattr(viewer, "selected_indices", None) or []), dtype=np.int64))
    if requested.size and not wanted.size:
        return {"ok": True, "changed": False, "selected": int(current.size), "note": "nothing_selectable"}
    if mode == "replace":
        result = wanted
    elif mode == "add":
        result = np.union1d(current, wanted)
    else:
        result = np.setxor1d(current, wanted)
    if np.array_equal(result, current):
        return {"ok": True, "changed": False, "selected": int(current.size), "note": "unchanged"}

    viewer.selected_indices = [int(value) for value in result]
    viewer._note_hand_selection()
    viewer.update_selection_visual()
    show = getattr(viewer, "_show_selection_status", None)
    if callable(show):
        show()
    return {"ok": True, "changed": True, "selected": int(result.size), "note": None}


def handle_tree_focus(viewer, data):
    """The ``tree_focus`` action: put the network's click focus on the node of a leaf, as a click on the node does.

    ``index`` (it must be there) is a Viewer node number that is shown in the network, or None to clear the click
    focus (the selection is not touched either way). Anything else, or a node out of range or hidden, changes
    nothing. Returns
    {"ok", "changed", "note"}.
    """
    if not isinstance(data, dict) or "index" not in data:
        return {"ok": False, "changed": False, "note": "bad_index"}
    index = data["index"]
    count = int(getattr(viewer, "n_nodes", 0) or 0)
    if index is None:
        clear = getattr(viewer, "clear_left_click_focus", None)
        if not callable(clear):
            return {"ok": False, "changed": False, "note": "no_network"}
        if getattr(viewer, "selected_node_idx", None) is None:
            return {"ok": True, "changed": False, "note": "unchanged"}
        clear()
        return {"ok": True, "changed": True, "note": None}
    if not _is_node_number(index) or not 0 <= int(index) < count:
        return {"ok": False, "changed": False, "note": "bad_index"}
    index = int(index)
    visible = getattr(viewer, "visible_mask", None)
    if visible is None or len(visible) != count or not bool(visible[index]):
        return {"ok": False, "changed": False, "note": "hidden"}
    focus = getattr(viewer, "apply_left_click_focus", None)
    if not callable(focus):
        return {"ok": False, "changed": False, "note": "no_network"}
    focus(index)
    return {"ok": True, "changed": True, "note": None}


def follow_colors(viewer):
    """Have the viewer tell this module when the nodes are drawn again; once, however often it is asked."""
    if getattr(viewer, "_tree_color_watch", None) is not None:
        return
    viewer._tree_color_watch = ColorWatch(viewer)
    if getattr(viewer, "node_update_listeners", None) is None:
        viewer.node_update_listeners = []
    viewer.node_update_listeners.append(viewer._tree_color_watch)


# ---- The tree in an inspection snapshot ------------------------------------------------------------------------------

# What a source that cannot be shown says, by the status the tree states give.
SNAPSHOT_REASONS = {
    "no_alignment": "no alignment is loaded",
    "unnamed_alignment": "the loaded alignment's name is not one the Embedding MSA tool writes, so it has no guide tree",
    "no_tree": "the loaded alignment has no guide tree file",
    "unreadable": "the guide tree file cannot be read",
    "no_overlap": "none of the tree's leaves is a node of the network",
    "no_network": "no network is loaded",
    "no_edges": "the network has no edges to build the tree from",
    "no_upload": "no tree file has been opened in Tree View",
}
SNAPSHOT_THRESHOLD_FIELDS = ("current", "kept", "cutoff", "k", "score_kind", "top", "bottom", "groups", "forest")


def snapshot_overview(viewer):
    """What a snapshot's summary says of the trees, without reading any: which can be shown, which opens first, why one cannot.

    ``available`` lists the sources (guide, threshold, uploaded) that can be captured as far as can be told without
    reading a file or building the tree (a guide tree file that is there may still be unreadable), ``opens_on``
    the one Tree View opens on (None when there is none), and ``unavailable`` the status of each source that cannot be shown.
    """
    opens_on, guide, threshold_reason = opening_source(viewer)
    unavailable = {}
    if guide["status"] != "present":
        unavailable[SOURCE_GUIDE] = guide["status"]
    if threshold_reason is not None:
        unavailable[SOURCE_THRESHOLD] = threshold_reason
    if stored_upload(viewer) is None:
        unavailable[SOURCE_UPLOADED] = "no_upload"
    available = [source for source in (SOURCE_GUIDE, SOURCE_THRESHOLD, SOURCE_UPLOADED) if source not in unavailable]
    return {"available": available, "opens_on": opens_on if available else None, "unavailable": unavailable}


def _unpacked(payload, key, dtype):
    """The array of a page payload's base64 text, read only."""
    return np.frombuffer(base64.b64decode(payload[key]), dtype=dtype)


def _fingerprint(parent, length, support, labels):
    """Eight hex digits that name a tree: the same arrays and leaf names give the same ones."""
    digest = zlib.crc32(parent.tobytes())
    digest = zlib.crc32(length.tobytes(), digest)
    digest = zlib.crc32(support.tobytes(), digest)
    return f"{zlib.crc32(chr(10).join(labels).encode('utf-8', 'surrogatepass'), digest):08x}"


def capture_snapshot_part(viewer, options, budget_bytes):
    """Freeze one tree for an inspection snapshot: ``options["tree_source"]`` (guide, threshold or uploaded), else the one Tree View opens on.

    Read only: the choice of source Tree View keeps is not touched. Raises ValueError, in words, for a source that cannot
    be shown or a tree too big for ``budget_bytes``. Returns what tree_snapshot reads: the arrays Tree View is sent,
    the leaf labels, the clade names an uploaded file gives, ``artificial`` (the nodes whose place stands for no score),
    ``not_aligned`` and ``ambiguous`` leaves, ``index`` (tree_snapshot.tree_index) and the ``summary`` the snapshot's
    overview shows.
    """
    from web_ui import tree_snapshot

    source = options.get("tree_source") or opening_source(viewer)[0]
    if source not in SOURCES:
        raise ValueError(f"Tree View does not know a tree called {str(source)[:60]}.")
    upload = None
    if source == SOURCE_THRESHOLD:
        state = threshold_state(viewer)
    elif source == SOURCE_UPLOADED:
        upload = stored_upload(viewer)
        state = uploaded_state(viewer)
    else:
        state = guide_state(viewer)
    if state["status"] != "ok":
        words = SNAPSHOT_REASONS.get(state["status"], state["status"])
        detail = f": {state['reason']}" if state.get("reason") else ""
        shown = ", ".join(other for other in snapshot_overview(viewer)["available"] if other != source) or "none"
        raise ValueError(f"The {source} tree cannot be shown: {words} ({state['status']}{detail}). Trees that can be shown: {shown}.")
    counts = state["counts"]
    if counts["nodes"] * 40 + sum(len(label) + 56 for label in state["labels"]) > budget_bytes:
        raise ValueError("The tree is too large for the snapshot memory budget.")
    parent = _unpacked(state, "parent", "<i4")
    length = _unpacked(state, "length", "<f4")
    support = _unpacked(state, "support", "i1")
    viewer_node = _unpacked(state, "viewer_node", "<i4")
    labels = list(state["labels"])
    index = tree_snapshot.tree_index(parent, length)
    root = len(parent) - 1
    artificial = np.zeros(len(parent), dtype=bool)
    threshold = None
    if source == SOURCE_THRESHOLD:
        info = state["threshold"]
        threshold = {key: info[key] for key in SNAPSHOT_THRESHOLD_FIELDS if key in info}
        if info["forest"]:
            # The groups that share no kept edge hang from one root put below the lowest score: its place and the
            # lengths of the branches to it stand for no score.
            artificial[root] = True
            artificial[np.flatnonzero(parent == root)] = True
        used = "artificial" if info["forest"] else "network"
        support_kind = None
    elif source == SOURCE_UPLOADED:
        used = "file"
        support_kind = "file" if (support >= 0).any() else None
    else:
        used = "file" if state["rooted"] is True else "midpoint"
        support_kind = "split_stability" if (support >= 0).any() else None
    shown = viewer_node >= 0
    not_aligned = np.zeros(len(labels), dtype=bool)
    if source == SOURCE_GUIDE:
        aligned = getattr(loaded_alignment(viewer)[0], "aligned_node_mask", None)
        headers = getattr(viewer, "full_headers", ()) or ()
        if aligned is not None and len(aligned) == len(headers) and shown.any():
            not_aligned[shown] = ~np.asarray(aligned, dtype=bool)[viewer_node[shown]]
    ambiguous = np.zeros(0, dtype=np.int64)
    if upload is not None and state["match"]["ambiguous"]:
        ambiguous = np.asarray(_match(upload.labels, [str(header) for header in (getattr(viewer, "full_headers", ()) or ())])[2], dtype=np.int64)
    summary = {
        "source": source,
        "name": state.get("name"),
        "file": state.get("tree_path") if source == SOURCE_GUIDE else upload.name if upload is not None else None,
        "rooted": state["rooted"],
        "root": used,
        "has_lengths": bool(state["has_lengths"]),
        "support_kind": support_kind,
        "counts": {**counts, "clades": counts["nodes"] - counts["leaves"]},
        "warnings": [{key: warning[key] for key in ("code", "count") if key in warning} for warning in state["warnings"]],
        "threshold": threshold,
        "fingerprint": _fingerprint(parent, length, support, labels),
    }
    return {
        "source": source, "summary": summary, "parent": parent, "length": length, "support": support,
        "viewer_node": viewer_node, "labels": labels, "has_lengths": summary["has_lengths"],
        "clade_names": dict(upload.clade_names) if upload is not None else {}, "artificial": artificial,
        "not_aligned": not_aligned, "ambiguous": ambiguous, "index": index,
    }


def register_snapshot_part(registry):
    """Declare Tree View's part of an inspection snapshot: ``include_tree`` asks for it, ``tree_source`` picks the tree."""
    from web_ui import tree_snapshot

    registry.register_snapshot_part(
        "tree", "tree",
        overview=snapshot_overview,
        capture=capture_snapshot_part,
        queries={"query_clades": tree_snapshot.clade_rows, "query_tree_names": tree_snapshot.name_rows},
        summary_arguments={"include_tree": "switch", "tree_source": "option"},
        capability="tree_snapshots_v1",
    )


def register_backend(registry, viewer):
    """Register Tree View's web capabilities without adding a Console button."""
    registry.register_api_route("tree", DATA_ROUTE, lambda request: handle_tree_data(viewer, request))
    registry.register_api_route("tree", COLORS_ROUTE, lambda request: handle_tree_colors(viewer, request))
    registry.register_api_route("tree", MISMATCH_ROUTE, lambda request: handle_tree_mismatch(viewer, request))
    registry.register_api_route("tree", SELECTION_ROUTE, lambda request: handle_tree_selection(viewer, request))
    registry.register_action("tree", "tree_select", lambda data: handle_tree_select(viewer, data))
    registry.register_action("tree", "tree_focus", lambda data: handle_tree_focus(viewer, data))
    registry.register_action("tree", "tree_upload", lambda data: handle_tree_upload(viewer, data))
    from web_ui import Web_Server

    registry.register_api_route(
        "tree", UPLOAD_ROUTE, lambda request: handle_tree_upload_request(viewer, request), ("POST",),
        max_body=Web_Server.MAX_ACTION_BYTES, content_types=("application/json",),
    )
    register_snapshot_part(registry)
    follow_colors(viewer)


def activate(viewer):
    """Add the Tree View button to the Console and keep it for the session, as the pages' buttons are."""
    if hasattr(viewer, "add_console_button"):
        viewer.add_console_button(
            name="treeViewBtn",
            label=translate("Viewer", "Tree View"),
            callback=lambda: viewer.open_page("tree"),
            tooltip=translate("Viewer", "Open Tree View in browser: the guide tree, the network threshold tree or a tree file"),
            icon="list-tree",
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
