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
    version = zlib.crc32(rgba.tobytes())
    version = zlib.crc32(hidden.tobytes(), version)
    if cluster is not None:
        version = zlib.crc32(cluster.tobytes(), version)
    return {"count": count, "rgba": rgba, "hidden": hidden, "cluster": cluster, "version": version & 0xFFFFFFFF}


def colors_state(viewer):
    """What /api/tree/colors answers: ``status`` "ok" with the colours of every node of the network, or "no_colors".

    ``rgba`` holds 4 bytes a node, ``hidden`` one bit a node (set when the node is hidden, least significant bit
    first), ``cluster`` the cluster number of each node (-1 for none) when the Viewer has clusters, in 2 bytes each
    (``cluster_type`` "i2") unless one of them needs 4 ("i4"), and ``plain`` the colour of a node nothing has coloured.
    The nodes are the Viewer's, in its numbering; the tree's leaves find theirs in the viewer_node array of
    /api/tree/data.
    """
    found = colors_snapshot(viewer)
    if found is None:
        return {"status": "no_colors", "version": 0}
    cluster = found["cluster"]
    small = cluster is not None and (int(cluster.min()) >= -(2**15) and int(cluster.max()) < 2**15)
    return {
        "status": "ok",
        "version": found["version"],
        "count": found["count"],
        "plain": _plain_color_bytes(),
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
    follow_colors(viewer)


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
