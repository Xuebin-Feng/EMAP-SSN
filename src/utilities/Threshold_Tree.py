# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The threshold tree of a network: single linkage over its edges, strongest first.

Cutting the network at a score keeps the edges scoring at least that much, and the
groups of nodes those edges connect are the connected components. Lowering the cut
only ever joins groups, so the components at every cut are the clades of one tree:
the single-linkage dendrogram of the network, with similarity in place of distance.
Each internal node is where two groups first share an edge, and its height is that
edge's score.

The network's edges are already in order (desktop.Viewer_Edges.EdgeOrder sorts them
once, by descending score, when the network is drawn), so no sort is paid for here: one
union-find pass (path halving) over the edges, stopped as soon as every node is in one
group. 44,000 nodes and 48.7 million edges take about 0.14 s in the worst case and a few
milliseconds when the network connects early.

Joins at the same score are one branch point with several children, as the cut at that
score cannot tell them apart. Leaves stand at the highest score of the edges, a branch's
length is the score it spans, and a node's height is its join score, so the tree is a
dendrogram with every leaf aligned. Nodes that never share a kept edge with the rest are
a forest: the groups hang from one root a little below the lowest kept score (``gap``
below it), and ``groups`` says how many there are. Lengths to that root stand for no score.

The arrays are what Tree View takes: nodes in post-order (every child before its parent,
the root last, a node's leaves one block), ``parent`` (-1 for the root), ``length`` and
``leaf_order``, the network node of each leaf in leaf order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

# How far below the lowest kept score a forest's root is put, as a share of the scores' range.
FOREST_GAP_SHARE = 0.05


@dataclass(frozen=True)
class ThresholdTree:
    """The tree: post-order arrays over ``node_count`` nodes (the first ``leaf_count`` leaves are not
    necessarily the first nodes; ``leaf_order[k]`` is the network node of the k-th leaf in node order)."""

    parent: np.ndarray  # int32, -1 for the root
    length: np.ndarray  # float32, the score a branch spans; 0 for the root
    height: np.ndarray  # float32, each node's score (a leaf's is ``top``)
    leaf_order: np.ndarray  # int32, network node of each leaf, in leaf order
    leaf_count: int
    node_count: int
    groups: int  # separate groups of nodes: 1 for a connected network
    top: float  # the highest kept score: where the leaves stand
    bottom: float  # the lowest score a join happened at (== top for no joins)
    root_score: float  # the root's height: ``bottom`` for a connected network, below it for a forest
    joins: int  # joins of two groups made


@njit(nogil=True, cache=True)
def _find(parent, node):
    """The root of ``node``'s group, halving the path on the way."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


@njit(nogil=True, cache=True)
def _join(edges, scores, node_count, left, right, height):
    """Single linkage over edges in descending score order.

    Join j makes node ``node_count + j`` of two groups' nodes ``left[j]`` and ``right[j]`` at
    ``height[j]``. Returns the number of joins, and the number of edges looked at is the
    number up to the last join. A NaN score joins nothing.
    """
    group = np.arange(node_count)
    top = np.arange(node_count)  # the binary-tree node a group's root stands for
    joins = 0
    for k in range(len(edges)):
        score = scores[k]
        if score != score:
            continue
        a = _find(group, edges[k, 0])
        b = _find(group, edges[k, 1])
        if a == b:
            continue
        left[joins] = top[a]
        right[joins] = top[b]
        height[joins] = score
        group[b] = a
        top[a] = node_count + joins
        joins += 1
        if joins == node_count - 1:
            break
    return joins


@njit(nogil=True, cache=True)
def _arrange(node_count, joins, left, right, height, top, root_score):
    """Merge equal-height joins, hang separate groups from one root, and number the nodes in post-order.

    Binary nodes are 0..node_count-1 (leaves) and node_count+j (join j); with more than one group the
    last binary node, node_count + joins, is the forest's root. Returns the arrays of ThresholdTree
    (the post-order ones) and how many nodes there are.
    """
    total = node_count + joins
    # The groups that are left: the binary nodes nothing joined into.
    joined = np.zeros(total, dtype=np.uint8)
    for j in range(joins):
        joined[left[j]] = 1
        joined[right[j]] = 1
    groups = 0
    for v in range(total):
        if joined[v] == 0:
            groups += 1
    forest = groups > 1
    size = total + 1 if forest else total
    height_of = np.empty(size, dtype=np.float64)
    bin_parent = np.full(size, -1, dtype=np.int64)
    for v in range(node_count):
        height_of[v] = top
    for j in range(joins):
        height_of[node_count + j] = height[j]
        bin_parent[left[j]] = node_count + j
        bin_parent[right[j]] = node_count + j
    if forest:
        height_of[total] = root_score
        for v in range(total):
            if joined[v] == 0:
                bin_parent[v] = total
    # Parents are made after their children, so counting down from the root, a node whose parent has its
    # height is absorbed into it: ``rep`` is the node that stands for it.
    rep = np.arange(size)
    for v in range(size - 2, node_count - 1, -1):
        p = bin_parent[v]
        if p >= 0 and height_of[p] == height_of[v]:
            rep[v] = rep[p]
    # The effective parent of each kept node, and its children as lists side by side.
    kept = 0
    degree = np.zeros(size + 1, dtype=np.int64)
    eff_parent = np.full(size, -1, dtype=np.int64)
    for v in range(size):
        if rep[v] != v:
            continue
        kept += 1
        p = bin_parent[v]
        if p >= 0:
            p = rep[p]
            eff_parent[v] = p
            degree[p + 1] += 1
    for v in range(size):
        degree[v + 1] += degree[v]
    fill = degree[:-1].copy()
    child_list = np.empty(degree[size], dtype=np.int64)
    for v in range(size):
        if rep[v] == v and eff_parent[v] >= 0:
            child_list[fill[eff_parent[v]]] = v
            fill[eff_parent[v]] += 1
    root = size - 1
    # Post-order, without recursion: a stack of (node, next child).
    new_id = np.full(size, -1, dtype=np.int64)
    stack = np.empty(kept, dtype=np.int64)
    cursor = np.empty(kept, dtype=np.int64)
    out_parent = np.empty(kept, dtype=np.int32)
    out_height = np.empty(kept, dtype=np.float32)
    leaf_order = np.empty(node_count, dtype=np.int32)
    depth = 0
    stack[0] = root
    cursor[0] = degree[root]
    made = 0
    leaves = 0
    while depth >= 0:
        v = stack[depth]
        if cursor[depth] < degree[v + 1]:
            child = child_list[cursor[depth]]
            cursor[depth] += 1
            depth += 1
            stack[depth] = child
            cursor[depth] = degree[child]
            continue
        new_id[v] = made
        out_height[made] = height_of[v]
        if v < node_count:
            leaf_order[leaves] = v
            leaves += 1
        made += 1
        depth -= 1
    for v in range(size):
        if new_id[v] >= 0:
            p = eff_parent[v]
            out_parent[new_id[v]] = new_id[p] if p >= 0 else -1
    return out_parent, out_height, leaf_order, groups


def threshold_tree(edges, scores, node_count, *, check=True) -> ThresholdTree:
    """The threshold tree of a network of ``node_count`` nodes whose edges (an (E, 2) array of node
    numbers) are in descending score order, ``scores`` their scores; ``scores`` None means every edge
    scores the same. Raises ValueError for no nodes or (with ``check``, which costs a pass over the edges that
    a caller whose edges are already checked can skip) an edge outside the network."""
    node_count = int(node_count)
    if node_count < 1:
        raise ValueError("A network with no nodes has no tree.")
    edges = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
    if scores is None:
        scores = np.zeros(len(edges), dtype=np.float32)
    elif not isinstance(scores, np.ndarray) or scores.dtype not in (np.float32, np.float64):
        scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(edges):
        raise ValueError("The edge scores must match the edges.")
    if check and len(edges) and (int(edges.min()) < 0 or int(edges.max()) >= node_count):
        raise ValueError("An edge names a node outside the network.")

    left = np.empty(max(node_count - 1, 0), dtype=np.int64)
    right = np.empty(max(node_count - 1, 0), dtype=np.int64)
    height = np.empty(max(node_count - 1, 0), dtype=np.float64)
    joins = int(_join(edges, scores, node_count, left, right, height))

    # The leaves stand at the highest kept score. A forest hangs below the lowest kept score of the whole
    # network (an edge inside a group is a kept score too), by a share of the range.
    highest, lowest = _highest_finite(scores), _lowest_finite(scores)
    top = 0.0 if highest is None else highest
    kept_low = top if lowest is None else lowest
    gap = (top - kept_low) * FOREST_GAP_SHARE
    root_score = kept_low - (gap if gap > 0 else 1.0)
    bottom = float(height[joins - 1]) if joins else top
    parent, node_height, leaf_order, groups = _arrange(
        node_count, joins, left[:joins], right[:joins], height[:joins], top, root_score,
    )
    length = np.zeros(len(parent), dtype=np.float32)
    has_parent = parent >= 0
    length[has_parent] = (
        node_height[has_parent].astype(np.float64) - node_height[parent[has_parent]].astype(np.float64)
    ).astype(np.float32)
    return ThresholdTree(
        parent=parent, length=length, height=node_height, leaf_order=leaf_order,
        leaf_count=node_count, node_count=len(parent), groups=int(groups),
        top=top, bottom=bottom, root_score=float(node_height[-1]), joins=joins,
    )


def _lowest_finite(scores):
    """The lowest score that is not NaN, or None. The edges are in descending order, so it is the last such."""
    for k in range(len(scores) - 1, -1, -1):
        value = scores[k]
        if value == value:
            return float(value)
    return None


def _highest_finite(scores):
    for k in range(len(scores)):
        value = scores[k]
        if value == value:
            return float(value)
    return None


def groups_at(tree: ThresholdTree, threshold: float) -> np.ndarray:
    """The group number of each leaf (in leaf order) when the network is cut at ``threshold``: the groups
    the edges scoring at least that connect. For tests and for anything that wants the cut of the tree."""
    count = tree.node_count
    cut = tree.height >= threshold
    cut[count - 1] = cut[count - 1] and tree.groups == 1
    labels = np.empty(count, dtype=np.int64)
    # Parents come after children, so counting down from the root a node starts a group unless its
    # parent is also joined at or above the threshold.
    for node in range(count - 1, -1, -1):
        parent = tree.parent[node]
        if parent < 0 or not cut[parent]:
            labels[node] = node
        else:
            labels[node] = labels[parent]
    leaf_nodes = np.flatnonzero(_is_leaf(tree))
    return labels[leaf_nodes]


def _is_leaf(tree: ThresholdTree) -> np.ndarray:
    children = np.zeros(tree.node_count, dtype=np.int64)
    kids = tree.parent[tree.parent >= 0]
    np.add.at(children, kids, 1)
    return children == 0
