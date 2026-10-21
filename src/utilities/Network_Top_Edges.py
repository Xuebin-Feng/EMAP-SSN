# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Each node's strongest edges, found while the network's pairs are streamed.

A UMAP layout uses only the ``UMAP_NEIGHBORS`` strongest edges of every node, as a
source and as a target. A network can hold close to a billion pairs, so nothing here
keeps a column of the file: the pairs are read and scored a slice at a time
(utilities.Network_Score_Statistics.ScoreSource), and each worker thread keeps, for
every node and direction, a small heap of its best edges so far. The memory used is
a few slices per worker plus ``node_count * keep_limit`` entries per buffer set, and
the result (at most twice that many edges), whatever the size of the network. The
buffers of the worker threads are merged at the end. A network of 974 million pairs
takes 5 to 11 s and 0.5 GB this way; sorting every pair with pandas took about 90
bytes per pair and minutes.

Ranking is a total order, so the result does not depend on how the pairs were split
between threads or on the slice size: higher score first; equal scores (ties are
common in quantized scores and in capped -log10 E-values) by the earlier pair in the
file; a NaN score after every other score, again by pair order. That is the order
``DataFrame.sort_values("score", ascending=False, kind="stable")`` gives, so
``groupby(...).head(keep_limit)`` on such a frame keeps the same edges.

This module imports no Qt, so the Viewer, the layout job and the tests use it as it is.
"""

from __future__ import annotations

import threading

import numpy as np
from numba import njit

from utilities import Network_Score_Statistics as scan

# The most memory the per-thread buffer sets may take together. Each set holds
# 2 * node_count * keep_limit entries of score + pair position + other node, so a large
# UMAP_NEIGHBORS on a large network gets fewer threads rather than more memory.
BUFFER_BUDGET_BYTES = 1 << 30
# The reads of the worker threads share the HDF5 library's lock, so beyond a few threads
# a pass only gets more buffers: 974M pairs took 6.3 s with 2 threads, 5.2 s with 4 and
# 4.8 s with 8, from the file cache (a cold read of the 14.5 GB file took 11 s).
DEFAULT_MAX_WORKERS = 8
_NO_POSITION = np.iinfo(np.int64).max


@njit(nogil=True, cache=True)
def _better(score_a, position_a, score_b, position_b):
    """Whether edge a ranks strictly above edge b (see the module docstring)."""
    a_nan = score_a != score_a
    b_nan = score_b != score_b
    if a_nan or b_nan:
        if a_nan and b_nan:
            return position_a < position_b
        return b_nan
    if score_a > score_b:
        return True
    if score_a < score_b:
        return False
    return position_a < position_b


@njit(nogil=True, cache=True)
def _offer(node, other, score, position, keep, state):
    """Put an edge into its node's buffer if it is one of the node's best ``keep`` so far.

    state is (count, worst_score, worst_position, scores, positions, others). A node's
    row of the last three arrays is a binary heap whose root, slot 0, is the lowest
    ranked entry: a full buffer replaces it in O(log keep) instead of searching for the
    next lowest. worst_score and worst_position copy the root of a full buffer; a buffer
    that is not yet full has worst_score at the lowest value, so nothing is turned away.
    """
    count, worst_score, worst_position, scores, positions, others = state
    filled = count[node]
    if filled < keep:
        slot = filled
        count[node] = filled + 1
        while slot > 0:                        # sift up past every better parent
            parent = (slot - 1) >> 1
            if not _better(scores[node, parent], positions[node, parent], score, position):
                break
            scores[node, slot] = scores[node, parent]
            positions[node, slot] = positions[node, parent]
            others[node, slot] = others[node, parent]
            slot = parent
        scores[node, slot] = score
        positions[node, slot] = position
        others[node, slot] = other
        if filled + 1 < keep:
            return
    else:
        if not _better(score, position, worst_score[node], worst_position[node]):
            return
        slot = 0                               # the new entry takes the root, then sinks
        while True:
            child = 2 * slot + 1
            if child >= keep:
                break
            if child + 1 < keep and _better(
                scores[node, child], positions[node, child],
                scores[node, child + 1], positions[node, child + 1],
            ):
                child += 1                     # the lower ranked of the two children
            if not _better(score, position, scores[node, child], positions[node, child]):
                break
            scores[node, slot] = scores[node, child]
            positions[node, slot] = positions[node, child]
            others[node, slot] = others[node, child]
            slot = child
        scores[node, slot] = score
        positions[node, slot] = position
        others[node, slot] = other
    worst_score[node] = scores[node, 0]
    worst_position[node] = positions[node, 0]


@njit(nogil=True, cache=True)
def _accumulate(sources, targets, scores, positions, keep, as_source, as_target):
    """Offer every pair to its source's and its target's buffer.

    The test before ``_offer`` is the cheap rejection almost every pair meets once the
    buffers are full: strictly below the node's lowest entry, or equal to it but later
    in the file. NaN comparisons are false, so a NaN score always reaches ``_offer``.
    """
    source_worst, source_position = as_source[1], as_source[2]
    target_worst, target_position = as_target[1], as_target[2]
    for index in range(len(scores)):
        score = scores[index]
        position = positions[index]
        source = sources[index]
        target = targets[index]
        floor = source_worst[source]
        if not (score < floor or (score == floor and position > source_position[source])):
            _offer(source, target, score, position, keep, as_source)
        floor = target_worst[target]
        if not (score < floor or (score == floor and position > target_position[target])):
            _offer(target, source, score, position, keep, as_target)


@njit(nogil=True, cache=True)
def _merge(into, other, keep):
    """Offer every entry of another thread's buffers to these."""
    count, _, _, scores, positions, others = other
    for node in range(len(count)):
        for slot in range(count[node]):
            _offer(node, others[node, slot], scores[node, slot], positions[node, slot], keep, into)


def _lowest_value(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return -np.inf
    if dtype.kind in "iu":
        return np.iinfo(dtype).min
    raise TypeError(f"Pair scores of dtype {dtype} cannot be ranked.")


def _new_direction(node_count, keep, dtype):
    return (
        np.zeros(node_count, dtype=np.int32),                              # entries held
        np.full(node_count, _lowest_value(dtype), dtype=dtype),            # lowest score
        np.full(node_count, _NO_POSITION, dtype=np.int64),                 # its pair position
        np.zeros((node_count, keep), dtype=dtype),                         # scores
        np.zeros((node_count, keep), dtype=np.int64),                      # pair positions
        np.zeros((node_count, keep), dtype=np.int32),                      # the node at the other end
    )


class _Buffers:
    def __init__(self, node_count, keep, dtype):
        self.as_source = _new_direction(node_count, keep, dtype)
        self.as_target = _new_direction(node_count, keep, dtype)


def _entries(direction, as_source):
    """(positions, sources, targets, scores) of a direction's buffered edges."""
    count, _, _, scores, positions, others = direction
    filled = np.arange(scores.shape[1]) < count[:, None]
    nodes = np.nonzero(filled)[0].astype(np.int32)
    held_others = others[filled]
    if as_source:
        return positions[filled], nodes, held_others, scores[filled]
    return positions[filled], held_others, nodes, scores[filled]


def buffer_thread_count(node_count, keep_limit, itemsize, workers):
    """How many worker threads fit the buffer budget (at least one)."""
    per_set = 2 * int(node_count) * int(keep_limit) * (int(itemsize) + 8 + 4)
    return max(1, min(int(workers), BUFFER_BUDGET_BYTES // max(1, per_set)))


def strongest_edges(source, node_count, keep_limit, *, workers=None):
    """The edges that are among the best ``keep_limit`` of their source or of their target.

    source is a ScoreSource. Returns (positions, sources, targets, scores) of the union
    over all nodes of each node's best edges as a source and as a target, in the order
    of the stored pairs; sources and targets are the stored node indices.
    """
    keep = int(keep_limit)
    chunks = scan.chunk_ranges(source.edge_count)
    if keep <= 0 or not chunks:
        # Nothing is kept; the empty scores still carry the dtype a pass would produce.
        empty = source.read(0, 0, nodes=True, positions=True)
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32), empty.scores[:0])

    worker_limit = max(1, int(workers or min(scan.worker_count(), DEFAULT_MAX_WORKERS)))
    local = threading.local()
    held = []
    lock = threading.Lock()

    def task(start, stop):
        piece = source.read(start, stop, nodes=True, check_nodes=True, positions=True)
        buffers = getattr(local, "buffers", None)
        if buffers is None:
            buffers = _Buffers(node_count, keep, piece.scores.dtype)
            local.buffers = buffers
            with lock:
                held.append(buffers)
        _accumulate(piece.sources, piece.targets, piece.scores, piece.positions, keep,
                    buffers.as_source, buffers.as_target)

    # The dtype of the scores is known from a one-pair read; size the pool from it.
    probe = source.read(0, 1, positions=True)
    threads = buffer_thread_count(node_count, keep, probe.scores.dtype.itemsize, worker_limit)
    for _ in scan.run_pass(task, chunks, workers=threads):
        pass

    merged = held[0]
    for other in held[1:]:
        _merge(merged.as_source, other.as_source, keep)
        _merge(merged.as_target, other.as_target, keep)

    parts = [_entries(merged.as_source, True), _entries(merged.as_target, False)]
    positions = np.concatenate([part[0] for part in parts])
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    # An edge in both its source's and its target's best is listed twice.
    first = np.ones(len(positions), dtype=bool)
    first[1:] = positions[1:] != positions[:-1]
    chosen = order[first]
    sources = np.concatenate([part[1] for part in parts])[chosen]
    targets = np.concatenate([part[2] for part in parts])[chosen]
    scores = np.concatenate([part[3] for part in parts])[chosen]
    return positions[first], sources, targets, scores
