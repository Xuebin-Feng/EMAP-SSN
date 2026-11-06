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
import os
import threading
import zlib

import numpy as np

from desktop.Desktop_App import translate
from utilities.Guide_Tree_File import GuideTreeError, guide_tree_path_for_alignment, read_guide_tree
from web_ui.Plugin_Manager import ensure_registry

DATA_ROUTE = "/api/tree/data"
COLORS_ROUTE = "/api/tree/colors"
SELECTION_ROUTE = "/api/tree/selection"
# The ways a tree_select request combines its nodes with the selection the network has.
SELECT_MODES = ("replace", "add", "toggle")
# The event that says the colours (or which nodes are hidden, or the clusters) are not what the page was sent.
COLORS_EVENT = "tree_colors_changed"
# A colour change is announced this long after the last one seen, so that a command that draws in a burst sends one event.
COLOR_COALESCE_SECONDS = 0.15

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


def tree_state(viewer):
    """What the Tree View page is sent: the tree of the loaded alignment, or why there is none to draw.

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


def register_backend(registry, viewer):
    """Register Tree View's web capabilities without adding a Console button."""
    registry.register_api_route("tree", DATA_ROUTE, lambda request: handle_tree_data(viewer, request))
    registry.register_api_route("tree", COLORS_ROUTE, lambda request: handle_tree_colors(viewer, request))
    registry.register_api_route("tree", SELECTION_ROUTE, lambda request: handle_tree_selection(viewer, request))
    registry.register_action("tree", "tree_select", lambda data: handle_tree_select(viewer, data))
    registry.register_action("tree", "tree_focus", lambda data: handle_tree_focus(viewer, data))
    follow_colors(viewer)


def activate(viewer):
    """Add the Tree View button to the Console and keep it for the session, as the pages' buttons are."""
    if hasattr(viewer, "add_console_button"):
        viewer.add_console_button(
            name="treeViewBtn",
            label=translate("Viewer", "Tree View"),
            callback=lambda: viewer.open_page("tree"),
            tooltip=translate("Viewer", "Open the guide tree of the loaded alignment in browser"),
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
