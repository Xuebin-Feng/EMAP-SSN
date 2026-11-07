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

"""A tree held in an inspection snapshot, and the reads that answer from it (query_clades, query_tree_names).

tree_backend freezes one tree into a snapshot (the arrays Tree View is sent, plus the facts about it). Here is what is
worked out from those arrays and what the two reads do with them. Everything is NumPy over leaf ranges: the nodes are in
post-order, so a clade's leaves are a block of the leaf numbers (first_leaf to last_leaf), and the leaves of a clade, the
clades below it and the smallest clade holding some leaves are all arithmetic on those two arrays. A clade's number is its
node's number in the arrays, the same one Tree View's model uses for it.

The reads take (frozen, snapshot, indices, args), which is what a snapshot part's query is given
(web_ui.Plugin_Manager.SnapshotPart): the frozen part, the snapshot's data (headers, metadata, visibility, clusters, its
subsets), the nodes asked for and the arguments of the request.
"""

import numpy as np

from desktop.Viewer_Inspection import SnapshotStore, ViewerInspectionError

# Numbers in a row are rounded to this many significant digits.
SIGNIFICANT_DIGITS = 4
# The longest text a row holds of a leaf's name or a category; a longer one is cut and says so.
MAX_ROW_TEXT = 500
MAX_CATEGORY_TEXT = 80
TOP_CLUSTERS = 3
CLADE_ORDERS = ("size", "tree")
NAME_LISTS = ("leaves_not_in_network", "network_nodes_not_in_tree", "leaves_not_in_alignment", "ambiguous_names")


def _ancestor_sums(parent, weight):
    """For each node, the sum of ``weight`` over the node and its ancestors, by doubling the jump along the parents."""
    total = np.asarray(weight, dtype=np.float64).copy()
    ahead = np.asarray(parent, dtype=np.int64).copy()
    while True:
        live = np.flatnonzero(ahead >= 0)
        if not live.size:
            return total
        target = ahead[live]
        total[live] += total[target]
        ahead[live] = ahead[target]


def tree_index(parent, length):
    """What a clade's reads need of the tree, from its post-order ``parent`` (the root last, -1) and branch ``length``.

    Returns a dict of arrays over the nodes: ``is_leaf``; ``leaf_node`` (the node of each leaf, in leaf order);
    ``first_leaf`` and ``last_leaf`` (the leaf numbers a node's leaves run from and to, for a leaf its own);
    ``start`` (the first node of a node's block, so the node's subtree is the nodes ``start`` to the node); ``depth``
    (edges from the root) and ``distance`` (the branch lengths from the root, float64).
    """
    parent = np.asarray(parent, dtype=np.int64)
    count = len(parent)
    nodes = np.arange(count)
    has_child = np.zeros(count, dtype=bool)
    if count > 1:
        has_child[parent[:-1]] = True
    is_leaf = ~has_child
    leaf_node = np.flatnonzero(is_leaf).astype(np.int32)
    seen = np.cumsum(is_leaf)  # leaves up to and including each node
    # The first node of a block is the leftmost leaf below the node: follow first children, doubling the jump.
    first_child = nodes.copy()
    if count > 1:
        order = np.argsort(parent[:-1], kind="stable")
        above = parent[:-1][order]
        firsts = np.flatnonzero(np.r_[True, above[1:] != above[:-1]])
        first_child[above[firsts]] = order[firsts]
    start = first_child
    while True:
        jumped = start[start]
        if np.array_equal(jumped, start):
            break
        start = jumped
    weight = np.zeros(count)
    if count > 1:
        weight[:-1] = 1
    depth = _ancestor_sums(parent, weight)
    lengths = np.asarray(length, dtype=np.float64).copy()
    lengths[parent < 0] = 0.0
    return {
        "is_leaf": is_leaf,
        "leaf_node": leaf_node,
        "first_leaf": (seen[start] - 1).astype(np.int32),
        "last_leaf": (seen - 1).astype(np.int32),
        "start": start.astype(np.int32),
        "depth": depth.astype(np.int32),
        "distance": _ancestor_sums(parent, lengths),
    }


def smallest_clade(index, leaves):
    """The clade (an internal node) with the fewest leaves that holds every one of ``leaves``, which are leaf numbers.

    Among clades with the same leaves (a chain of one-child nodes) it is the deepest. None when no leaf is given.
    """
    leaves = np.asarray(leaves)
    if not leaves.size:
        return None
    low, high = int(leaves.min()), int(leaves.max())
    first, last = index["first_leaf"], index["last_leaf"]
    holds = (~index["is_leaf"]) & (first <= low) & (last >= high)
    nodes = np.flatnonzero(holds)
    if not nodes.size:
        return None
    # Ordered by leaves held, then by the node's number: a deeper node has the smaller number.
    size = (last[nodes] - first[nodes]).astype(np.int64)
    return int(nodes[np.argmin(size * (len(first) + 1) + nodes)])


def significant(value, digits=SIGNIFICANT_DIGITS):
    """``value`` as a float of at most ``digits`` significant digits, or None for a missing or non-finite number."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return float(f"{value:.{digits}g}")


def _text(value, limit):
    value = str(value)
    return value if len(value) <= limit else value[:limit]


class Rows:
    """The rows of one read: ``count`` of them, each made when it is reached, so a page costs what it holds.

    A snapshot's pager starts at ``rows_from(offset)`` and does not make the rows before it.
    """

    def __init__(self, count, make):
        self.count, self.make = count, make

    def rows_from(self, offset):
        for position in range(offset, self.count):
            yield self.make(position)

    def __iter__(self):
        return self.rows_from(0)

    def __len__(self):
        return self.count


class _Leaves:
    """The leaves of a frozen tree seen through one snapshot: which are nodes of the network, hidden, in which cluster."""

    def __init__(self, frozen, snapshot):
        self.viewer_node = frozen["viewer_node"]
        self.shown = self.viewer_node >= 0
        nodes = np.where(self.shown, self.viewer_node, 0)
        self.shown_before = np.concatenate(([0], np.cumsum(self.shown)))
        visible = np.asarray(snapshot["visible"], dtype=bool)
        hidden = self.shown & ~visible[nodes]
        self.hidden_before = np.concatenate(([0], np.cumsum(hidden)))
        clusters = snapshot.get("clusters")
        # A leaf that is no node of the network is -2, below every cluster label (noise is -1).
        self.cluster = None if clusters is None else np.where(self.shown, np.asarray(clusters, dtype=np.int64)[nodes], -2)
        self.nodes = nodes


def _column_reader(snapshot, name, leaves):
    """``read(first, last)`` for a metadata column: the top category with its count, or the median, over a clade's network leaves."""
    entry = snapshot["metadata"][name]
    values = entry["values"]
    classify = SnapshotStore._classify
    numeric = entry["type"] == "number"
    nodes = leaves.nodes
    if numeric:
        valid = np.array([classify(value, True) == "valid" for value in values], dtype=bool)
        numbers = np.array([float(value) if ok else np.nan for value, ok in zip(values, valid)], dtype=np.float64)
        leaf_values = np.where(leaves.shown, numbers[nodes], np.nan)

        def read_number(first, last):
            part = leaf_values[first:last + 1]
            part = part[~np.isnan(part)]
            return {"median": significant(np.median(part)) if part.size else None}

        return read_number
    texts = [str(value) if classify(value, False) == "valid" else None for value in values]
    present = np.array([text is not None for text in texts], dtype=bool)
    categories, codes = np.unique(np.array([text for text in texts if text is not None], dtype=object), return_inverse=True)
    node_code = np.full(len(texts), -1, dtype=np.int64)
    node_code[present] = codes
    leaf_codes = np.where(leaves.shown, node_code[nodes], -1)

    def read_text(first, last):
        part = leaf_codes[first:last + 1]
        part = part[part >= 0]
        if not part.size:
            return {"top": None}
        counts = np.bincount(part)
        best = int(np.argmax(counts))  # the first of equal counts is the smallest text
        return {"top": [_text(categories[best], MAX_CATEGORY_TEXT), int(counts[best])]}

    return read_text


def _count(value, name, minimum=0):
    if value is None:
        return None
    if type(value) is not int or value < minimum:
        raise ViewerInspectionError(f"{name} must be an integer of at least {minimum}")
    return value


def clade_rows(frozen, snapshot, indices, args):
    """The rows of query_clades: one for each clade (internal node) asked for, as the tree is drawn from its root.

    ``args`` may hold ``min_leaves`` and ``max_leaves`` (the leaves a clade holds in the tree), ``within`` (a clade:
    only the clades below it), ``contains`` (a subset id of the snapshot: only the smallest clade that holds every
    node of the subset that is a leaf, with how the subset sits in it; it takes no other filter), ``columns``
    (metadata columns, summarised for each clade's network leaves) and ``order`` ("size", the default: most leaves
    first, then by clade number, or "tree": from the root, each clade before the clades below it, left to right).
    """
    index = frozen["index"]
    unknown = sorted(set(args) - {"min_leaves", "max_leaves", "within", "contains", "columns", "order"})
    if unknown:
        raise ViewerInspectionError(f"Unknown query_clades argument: {', '.join(unknown)}")
    min_leaves, max_leaves = _count(args.get("min_leaves"), "min_leaves", 1), _count(args.get("max_leaves"), "max_leaves", 1)
    within, contains = args.get("within"), args.get("contains")
    order = args.get("order") or "size"
    if order not in CLADE_ORDERS:
        raise ViewerInspectionError("order must be size or tree")
    columns = args.get("columns") or []
    if any(name not in snapshot["metadata"] for name in columns):
        raise ViewerInspectionError("Unknown metadata column")
    if contains is not None and (within is not None or min_leaves is not None or max_leaves is not None or args.get("order") is not None):
        raise ViewerInspectionError("contains answers one clade; it takes no within, min_leaves, max_leaves or order")
    leaves = _Leaves(frozen, snapshot)
    readers = {name: _column_reader(snapshot, name, leaves) for name in columns}
    first, last, parent = index["first_leaf"], index["last_leaf"], frozen["parent"]
    names, artificial = frozen["clade_names"], frozen["artificial"]
    clusters = leaves.cluster
    has_lengths = frozen["has_lengths"]
    total_nodes = len(parent)

    def row(clade):
        low, high = int(first[clade]), int(last[clade])
        in_network = int(leaves.shown_before[high + 1] - leaves.shown_before[low])
        result = {
            "clade": clade,
            "parent": None if parent[clade] < 0 else int(parent[clade]),
            "depth": int(index["depth"][clade]),
            "height": significant(index["distance"][clade]) if has_lengths else None,
            "branch_length": significant(frozen["length"][clade]) if has_lengths and parent[clade] >= 0 else None,
            "support": None if frozen["support"][clade] < 0 else int(frozen["support"][clade]),
        }
        if clade in names:
            result["name"] = _text(names[clade], MAX_ROW_TEXT)
        result["leaves"] = {"total": high - low + 1, "in_network": in_network, "not_in_network": high - low + 1 - in_network,
                            "hidden": int(leaves.hidden_before[high + 1] - leaves.hidden_before[low])}
        top = []
        if clusters is not None:
            part = clusters[low:high + 1]
            counts = np.bincount(part[part >= -1] + 1)
            for label in np.argsort(-counts, kind="stable")[:TOP_CLUSTERS]:
                if counts[label]:
                    top.append([int(label) - 1, int(counts[label])])
        result["top_clusters"] = top
        if artificial[clade]:
            result["artificial"] = True
        if readers:
            result["columns"] = {name: read(low, high) for name, read in readers.items()}
        return result

    if contains is not None:
        members = snapshot["subsets"].get(contains)
        if members is None:
            raise ViewerInspectionError("Unknown subset for this snapshot")
        in_subset = np.zeros(len(snapshot["headers"]), dtype=bool)
        in_subset[np.asarray(members, dtype=np.int64)] = True
        held = leaves.shown & in_subset[leaves.nodes]
        subset_leaves = np.flatnonzero(held)
        clade = smallest_clade(index, subset_leaves)
        if clade is None:
            raise ViewerInspectionError("No node of the subset is a leaf of this tree")
        found = row(clade)
        inside = int(held[first[clade]:last[clade] + 1].sum())
        found["in_subset"] = inside
        found["outside_subset"] = found["leaves"]["in_network"] - inside
        found["subset_not_in_tree"] = int(len(members) - len(np.unique(leaves.viewer_node[held])))
        return Rows(1, lambda position: found)

    chosen = np.flatnonzero(~index["is_leaf"])
    if within is not None:
        if type(within) is not int or not 0 <= within < total_nodes or index["is_leaf"][within]:
            raise ViewerInspectionError("within must be the number of a clade of this tree")
        chosen = chosen[(chosen >= index["start"][within]) & (chosen < within)]
    size = last[chosen] - first[chosen] + 1
    keep = np.ones(len(chosen), dtype=bool)
    if min_leaves is not None:
        keep &= size >= min_leaves
    if max_leaves is not None:
        keep &= size <= max_leaves
    chosen, size = chosen[keep], size[keep]
    if order == "size":
        chosen = chosen[np.lexsort((chosen, -size))]
    else:
        chosen = chosen[np.lexsort((-chosen, index["start"][chosen]))]
    return Rows(len(chosen), lambda position: row(int(chosen[position])))


def name_rows(frozen, snapshot, indices, args):
    """The rows of query_tree_names: the names one of the fit lists counts (tree_backend.mismatch_state), by ``list``.

    The leaves are rows of ``name`` (the leaf's label), ``leaf`` (its node number in the tree) and, when it names
    a node of the network, ``index``; the network nodes no leaf names are rows of ``name`` (the full header) and ``index``.
    """
    unknown = sorted(set(args) - {"list"})
    if unknown:
        raise ViewerInspectionError(f"Unknown query_tree_names argument: {', '.join(unknown)}")
    which = args.get("list")
    if which not in NAME_LISTS:
        raise ViewerInspectionError("list must be one of " + ", ".join(NAME_LISTS))
    labels, viewer_node, leaf_node = frozen["labels"], frozen["viewer_node"], frozen["index"]["leaf_node"]

    def cut(text):
        text = str(text)
        return (text, False) if len(text) <= MAX_ROW_TEXT else (text[:MAX_ROW_TEXT], True)

    def leaf_row(leaf):
        name, cut_short = cut(labels[leaf])
        result = {"name": name, "leaf": int(leaf_node[leaf])}
        if viewer_node[leaf] >= 0:
            result["index"] = int(viewer_node[leaf])
        if cut_short:
            result["truncated"] = True
        return result

    if which == "network_nodes_not_in_tree":
        named = np.zeros(len(snapshot["headers"]), dtype=bool)
        named[viewer_node[viewer_node >= 0]] = True
        missing = np.flatnonzero(~named)

        def node_row(position):
            node = int(missing[position])
            name, cut_short = cut(snapshot["headers"][node])
            return {"name": name, "index": node, **({"truncated": True} if cut_short else {})}

        return Rows(len(missing), node_row)
    if which == "leaves_not_in_network":
        leaves = np.flatnonzero(viewer_node < 0)
    elif which == "leaves_not_in_alignment":
        leaves = np.flatnonzero(frozen["not_aligned"])
    else:
        leaves = np.asarray(frozen["ambiguous"], dtype=np.int64)
    return Rows(len(leaves), lambda position: leaf_row(int(leaves[position])))
