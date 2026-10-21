# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The three pair sets of a PaCMAP layout, built from a network's own scores.

PaCMAP (Wang, Huang, Rudin and Shaposhnik, JMLR 22(201), 2021) lays points out from three
kinds of pairs: near pairs (each node's nearest neighbours), mid-near pairs (a node and a
node of middling distance) and further pairs (a node and a random other node). The
reference package builds them from a feature matrix. A network has no features, only
scores between nodes, so here:

* near pairs are each node's ``neighbors`` strongest edges, taken from the edges the
  network's strongest-edge pass keeps (utilities.Network_Top_Edges);
* a mid-near partner is the second best of six random nodes by the network's score
  with the node, which is the package's rule with "distance" read as "negated score".
  The scores of the sampled pairs are looked up in the stored pairs while they are
  streamed once more (``mid_near_pairs``); a pair that is not stored counts as the
  farthest. This pass is what gives the layout its global arrangement: with random
  mid-near partners the layout is as local as UMAP's;
* further pairs are random nodes that are not among the node's near partners.

Isolated nodes, the ones with no edge, take part in no pair set: the engine places them
by its own rule.

Every random choice comes from a small counter-based generator (SplitMix64) seeded by the
layout seed, the kind of pair and the node, so the pairs are the same whatever the number
of threads, and on every platform and Numba version, which Numba's own random generator
does not promise. This module imports no Qt.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:                                              # pragma: no cover
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        def decorator(function):
            return function
        return decorator

from utilities import Network_Score_Statistics as scan
from utilities import Numba_Threads


# Module aliasing to prevent duplicate JIT caches in long-running processes
_THIS_MODULE = sys.modules[__name__]
if __name__ == "PaCMAP_Pairs":
    sys.modules.setdefault("utilities.PaCMAP_Pairs", _THIS_MODULE)
else:
    sys.modules.setdefault("PaCMAP_Pairs", _THIS_MODULE)


# --- The constants of the method -------------------------------------------------------
# Fixed, not settings: they are what "PaCMAP" means here. Changing one changes every
# layout, so Cache_Manifest.PACMAP_REVISION is raised with it.

# Mid-near partners per node: half the near ones, rounded up (mid_near_count).
FURTHER_RATIO = 2             # further partners per node, as a multiple of the near ones
MID_NEAR_CANDIDATES = 6       # the nodes sampled for each mid-near partner

STREAM_MID_NEAR = 1
STREAM_FURTHER = 2
STREAM_START = 3


def mid_near_count(neighbors):
    """Mid-near partners per node: half the near ones, rounded up."""
    return (int(neighbors) + 1) // 2


def further_count(neighbors):
    """Further partners per node: twice the near ones."""
    return FURTHER_RATIO * int(neighbors)


def effective_mid_near_count(node_count, neighbors):
    """Mid-near partners per node among ``node_count`` connected nodes.

    Each takes six distinct sampled nodes other than its own node, and a node's samples
    are distinct from each other, so a small network gets fewer partners per node
    (none below seven nodes).
    """
    return max(0, min(mid_near_count(neighbors), (int(node_count) - 1) // MID_NEAR_CANDIDATES))


def random_seed():
    """A seed for LAYOUT_SEED=null, which opts out of reproducibility."""
    return int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)


# --- SplitMix64 -------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MULTIPLY_1 = np.uint64(0xBF58476D1CE4E5B9)
_MULTIPLY_2 = np.uint64(0x94D049BB133111EB)
_FILTER_MULTIPLY = np.uint64(0xC2B2AE3D27D4EB4F)


@njit(nogil=True, cache=True)
def _mix(value):
    """SplitMix64's finalizer: a well-spread 64-bit value from a 64-bit one."""
    value = (value ^ (value >> np.uint64(30))) * _MULTIPLY_1
    value = (value ^ (value >> np.uint64(27))) * _MULTIPLY_2
    return value ^ (value >> np.uint64(31))


@njit(nogil=True, cache=True)
def node_state(seed, stream, node):
    """The generator state of one node in one stream of draws."""
    state = _mix(np.uint64(seed) + np.uint64(stream) * _GOLDEN)
    return _mix(state + (np.uint64(node) + np.uint64(1)) * _GOLDEN)


@njit(nogil=True, cache=True)
def next_value(state):
    """(the next state, the next 64-bit value)."""
    state = state + _GOLDEN
    return state, _mix(state)


@njit(nogil=True, cache=True)
def below(state, bound):
    """(the next state, a uniform integer in 0..bound-1, bound at least 1): masked rejection, so no modulo bias."""
    limit = np.uint64(bound) - np.uint64(1)
    mask = np.uint64(0)
    while mask < limit:
        mask = (mask << np.uint64(1)) | np.uint64(1)
    while True:
        state, value = next_value(state)
        value = value & mask
        if value <= limit:
            return state, np.int64(value)


@njit(nogil=True, cache=True)
def _fill_normal_start(count, dimensions, seed, scale, out):
    for node in range(count):
        state = node_state(seed, STREAM_START, node)
        for axis in range(dimensions):
            state, first = next_value(state)
            state, second = next_value(state)
            # Box-Muller from two uniform values in (0, 1].
            radius = math.sqrt(-2.0 * math.log((np.float64(first >> np.uint64(11)) + 1.0) / 9007199254740992.0))
            angle = 2.0 * math.pi * (np.float64(second >> np.uint64(11)) / 9007199254740992.0)
            out[node, axis] = np.float32(scale * radius * math.cos(angle))


def normal_start(count, dimensions, seed, scale):
    """(count, dimensions) float32 random normal cloud of standard deviation ``scale``, one stream per node."""
    out = np.empty((int(count), int(dimensions)), dtype=np.float32)
    _fill_normal_start(int(count), int(dimensions), int(seed), float(scale), out)
    return out


# --- Near pairs ---------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def _is_better(score_a, position_a, score_b, position_b):
    """Whether edge a ranks above edge b: the higher score, then the earlier edge; NaN last."""
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
def _incident_lists(compact_a, compact_b, count):
    """CSR lists of the edges at each node: (pointer, edge numbers), self-loops left out."""
    pointer = np.zeros(count + 1, dtype=np.int64)
    for edge in range(len(compact_a)):
        if compact_a[edge] != compact_b[edge]:
            pointer[compact_a[edge] + 1] += 1
            pointer[compact_b[edge] + 1] += 1
    for node in range(count):
        pointer[node + 1] += pointer[node]
    fill = pointer[:-1].copy()
    edges = np.empty(pointer[count], dtype=np.int64)
    for edge in range(len(compact_a)):
        a = compact_a[edge]
        b = compact_b[edge]
        if a != b:
            edges[fill[a]] = edge
            fill[a] += 1
            edges[fill[b]] = edge
            fill[b] += 1
    return pointer, edges


@njit(nogil=True, parallel=True, cache=True)
def _best_partners(compact_a, compact_b, scores, pointer, edges, neighbors, out):
    """Each node's ``neighbors`` strongest edges' other ends, best first, -1 where it has fewer.

    A node's edges are ranked by score, then by their order in the edge list; an edge
    listed twice counts once, at its best rank.
    """
    count = len(pointer) - 1
    for node in prange(count):
        best_partner = np.full(neighbors, -1, dtype=np.int64)
        best_edge = np.full(neighbors, -1, dtype=np.int64)
        filled = 0
        for slot in range(pointer[node], pointer[node + 1]):
            edge = edges[slot]
            other = compact_b[edge] if compact_a[edge] == node else compact_a[edge]
            score = scores[edge]
            if filled == neighbors and not _is_better(
                score, edge, scores[best_edge[neighbors - 1]], best_edge[neighbors - 1]
            ):
                continue
            # A partner already held keeps its better edge.
            held = -1
            for place in range(filled):
                if best_partner[place] == other:
                    held = place
                    break
            if held >= 0:
                if not _is_better(score, edge, scores[best_edge[held]], best_edge[held]):
                    continue
                for place in range(held, filled - 1):
                    best_partner[place] = best_partner[place + 1]
                    best_edge[place] = best_edge[place + 1]
                filled -= 1
            place = filled if filled < neighbors else neighbors - 1
            while place > 0 and _is_better(
                score, edge, scores[best_edge[place - 1]], best_edge[place - 1]
            ):
                best_partner[place] = best_partner[place - 1]
                best_edge[place] = best_edge[place - 1]
                place -= 1
            best_partner[place] = other
            best_edge[place] = edge
            if filled < neighbors:
                filled += 1
        for place in range(neighbors):
            out[node, place] = best_partner[place]


def connected_nodes(sources, targets, node_count):
    """The nodes with at least one edge to another node, sorted: (ids, compact index of every node).

    The compact index is -1 for a node with no edge.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    linked = np.zeros(int(node_count), dtype=bool)
    real = sources != targets
    linked[sources[real]] = True
    linked[targets[real]] = True
    ids = np.flatnonzero(linked).astype(np.int32)
    compact = np.full(int(node_count), -1, dtype=np.int32)
    compact[ids] = np.arange(len(ids), dtype=np.int32)
    return ids, compact


def near_partners(compact_sources, compact_targets, scores, count, neighbors):
    """(count, neighbors) int32: each node's strongest partners among the edges, -1 padded.

    The edges are in compact node numbers (every node has an edge), in a fixed order that
    breaks ties between equal scores.
    """
    out = np.full((int(count), int(neighbors)), -1, dtype=np.int32)
    if count == 0 or neighbors <= 0:
        return out
    compact_sources = np.ascontiguousarray(compact_sources, dtype=np.int32)
    compact_targets = np.ascontiguousarray(compact_targets, dtype=np.int32)
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    pointer, edges = _incident_lists(compact_sources, compact_targets, int(count))
    _best_partners(compact_sources, compact_targets, scores, pointer, edges, int(neighbors), out)
    return out


# --- Further pairs --------------------------------------------------------------------------


@njit(nogil=True, parallel=True, cache=True)
def _sample_further(near, count, wanted, seed, out):
    """``wanted`` distinct random nodes per node, none the node itself or one of its near partners."""
    neighbors = near.shape[1]
    for node in prange(count):
        state = node_state(seed, STREAM_FURTHER, node)
        taken = 0
        near_count = 0
        for place in range(neighbors):
            if near[node, place] >= 0:
                near_count += 1
        available = count - 1 - near_count
        need = wanted if wanted < available else available
        if need <= 0:
            continue
        if need * 4 >= available:
            # Nearly every other node is wanted: a partial shuffle of the candidates is exact and short.
            pool = np.empty(available, dtype=np.int64)
            filled = 0
            for other in range(count):
                if other == node:
                    continue
                skip = False
                for place in range(neighbors):
                    if near[node, place] == other:
                        skip = True
                        break
                if not skip:
                    pool[filled] = other
                    filled += 1
            for index in range(need):
                state, offset = below(state, available - index)
                pick = index + offset
                chosen = pool[pick]
                pool[pick] = pool[index]
                pool[index] = chosen
                out[node, index] = chosen
            continue
        while taken < need:
            state, other = below(state, count - 1)
            if other >= node:
                other += 1
            repeated = False
            for place in range(neighbors):
                if near[node, place] == other:
                    repeated = True
                    break
            if not repeated:
                for place in range(taken):
                    if out[node, place] == other:
                        repeated = True
                        break
            if not repeated:
                out[node, taken] = other
                taken += 1


def further_partners(near, count, neighbors, seed):
    """(count, further_count) int32: random non-neighbours of every node, -1 padded."""
    wanted = further_count(neighbors)
    out = np.full((int(count), wanted), -1, dtype=np.int32)
    if count > 1 and wanted > 0:
        _sample_further(np.ascontiguousarray(near, dtype=np.int32), int(count), wanted, int(seed), out)
    return out


# --- Mid-near pairs ---------------------------------------------------------------------------


@njit(nogil=True, parallel=True, cache=True)
def _draw_candidates(count, per_node, seed, out):
    """``per_node`` distinct random nodes per node, none the node itself."""
    for node in prange(count):
        state = node_state(seed, STREAM_MID_NEAR, node)
        taken = 0
        while taken < per_node:
            state, other = below(state, count - 1)
            if other >= node:
                other += 1
            repeated = False
            for place in range(taken):
                if out[node, place] == other:
                    repeated = True
                    break
            if not repeated:
                out[node, taken] = other
                taken += 1


def mid_near_candidates(count, neighbors, seed):
    """(count, slots, 6) int32: the random nodes each mid-near partner is chosen from."""
    slots = effective_mid_near_count(count, neighbors)
    per_node = slots * MID_NEAR_CANDIDATES
    out = np.empty((int(count), per_node), dtype=np.int32)
    if per_node:
        _draw_candidates(int(count), per_node, int(seed), out)
    return out.reshape(int(count), slots, MID_NEAR_CANDIDATES)


@njit(nogil=True, cache=True)
def _pair_key(a, b, node_count):
    low = a if a < b else b
    high = b if a < b else a
    return np.int64(low) * np.int64(node_count) + np.int64(high)


@njit(nogil=True, cache=True)
def _build_table(keys, bits, table_keys, table_slots):
    """Open-addressing hash table of the sorted unique ``keys``; the slot is the key's rank."""
    mask = len(table_keys) - 1
    shift = np.uint64(64 - bits)
    for slot in range(len(keys)):
        key = keys[slot]
        place = np.int64((np.uint64(key) * _GOLDEN) >> shift)
        while table_keys[place] >= 0:
            place = (place + 1) & mask
        table_keys[place] = key
        table_slots[place] = slot


@njit(nogil=True, cache=True)
def _build_filter(keys, bits, bit_array):
    shift = np.uint64(64 - bits)
    for slot in range(len(keys)):
        place = np.int64((np.uint64(keys[slot]) * _FILTER_MULTIPLY) >> shift)
        bit_array[place >> 3] |= np.uint8(1 << (place & 7))


@njit(nogil=True, cache=True)
def _probe(first, second, node_count, filter_bits, bit_array, table_bits, table_keys, table_slots, out):
    """out[t] = the slot of pair (first[t], second[t]) among the sampled pairs, -1 when it is not one."""
    filter_shift = np.uint64(64 - filter_bits)
    table_shift = np.uint64(64 - table_bits)
    mask = len(table_keys) - 1
    for t in range(len(first)):
        key = _pair_key(first[t], second[t], node_count)
        place = np.int64((np.uint64(key) * _FILTER_MULTIPLY) >> filter_shift)
        if not (bit_array[place >> 3] >> np.uint8(place & 7)) & np.uint8(1):
            out[t] = -1
            continue
        place = np.int64((np.uint64(key) * _GOLDEN) >> table_shift)
        found = -1
        while table_keys[place] >= 0:
            if table_keys[place] == key:
                found = table_slots[place]
                break
            place = (place + 1) & mask
        out[t] = found


class PairLookup:
    """Which of a set of node pairs a stream of stored pairs holds, and their best scores.

    The sampled pairs go into a hash table behind a small bit filter. The filter stays in
    the CPU cache and turns most stored pairs away without touching the table; the
    table, a few times the number of sampled pairs, settles the rest exactly. Nothing is
    sized by the square of the node count.
    """

    FILTER_BITS_PER_KEY = 16
    MIN_FILTER_BITS = 16
    MAX_FILTER_BITS = 30

    def __init__(self, first, second, node_count, *, table_bits=None, filter_bits=None):
        self.node_count = int(node_count)
        if self.node_count and self.node_count > 3_000_000_000:
            raise ValueError("Too many nodes for pair keys.")
        first = np.asarray(first, dtype=np.int64).ravel()
        second = np.asarray(second, dtype=np.int64).ravel()
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        self.keys = np.unique(low * np.int64(self.node_count) + high)
        wanted = len(self.keys)
        if table_bits is None:
            table_bits = max(4, int(np.ceil(np.log2(max(2, 2 * wanted)))))
        if filter_bits is None:
            filter_bits = int(np.ceil(np.log2(max(2, self.FILTER_BITS_PER_KEY * wanted))))
            filter_bits = min(max(filter_bits, self.MIN_FILTER_BITS), self.MAX_FILTER_BITS)
        self.table_bits = int(table_bits)
        self.filter_bits = int(filter_bits)
        if (1 << self.table_bits) <= wanted:
            raise ValueError("The hash table needs room for every sampled pair.")
        self.table_keys = np.full(1 << self.table_bits, -1, dtype=np.int64)
        self.table_slots = np.zeros(1 << self.table_bits, dtype=np.int32)
        self.bit_array = np.zeros(max(1, (1 << self.filter_bits) // 8), dtype=np.uint8)
        _build_table(self.keys, self.table_bits, self.table_keys, self.table_slots)
        _build_filter(self.keys, self.filter_bits, self.bit_array)
        self.best = np.full(wanted, -np.inf, dtype=np.float64)
        self.found = np.zeros(wanted, dtype=bool)

    def slots_of(self, first, second):
        """The slot of each pair among the sampled pairs, -1 for one that is not sampled."""
        # The stored node columns are probed as they are (uint16 for a network of fewer than 65,536
        # nodes), not widened first: the pass is bound by memory, not by arithmetic.
        first = np.ascontiguousarray(first)
        second = np.ascontiguousarray(second)
        out = np.empty(len(first), dtype=np.int32)
        _probe(first, second, self.node_count, self.filter_bits, self.bit_array,
               self.table_bits, self.table_keys, self.table_slots, out)
        return out

    def hits(self, first, second, scores):
        """The (slots, scores) of the pairs of one slice that are sampled; NaN scores are left out."""
        slots = self.slots_of(first, second)
        hit = np.flatnonzero(slots >= 0)
        if len(hit) == 0:
            return hit.astype(np.int32), np.zeros(0, dtype=np.float64)
        values = np.asarray(scores)[hit].astype(np.float64)
        keep = values == values
        return slots[hit][keep], values[keep]

    def add(self, slots, values):
        """Keep the best score seen for each sampled pair (the same whatever the order of the slices)."""
        if len(slots):
            np.maximum.at(self.best, slots, values)
            self.found[slots] = True

    def scores_of(self, first, second):
        """The best stored score of each pair, -inf for a pair the stream did not hold."""
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        slot = np.searchsorted(self.keys, low * np.int64(self.node_count) + high)
        return np.where(self.found[slot], self.best[slot], -np.inf)


def stream_pair_scores(source, index_map, lookup, *, workers=None, cancelled=None):
    """Read every stored pair of ``source`` once, and note the scores of the pairs ``lookup`` samples.

    index_map takes the source's stored node numbers to the lookup's (None: the same).
    """
    node_map = None if index_map is None else np.asarray(index_map)

    def task(start, stop):
        piece = source.read(start, stop, nodes=True)
        first, second = piece.sources, piece.targets
        if node_map is not None:
            first, second = node_map[first], node_map[second]
        return lookup.hits(first, second, piece.scores)

    for slots, values in scan.run_pass(
        task, scan.chunk_ranges(source.edge_count), workers=workers, cancelled=cancelled
    ):
        lookup.add(slots, values)
    return lookup


@njit(nogil=True, parallel=True, cache=True)
def _pick_second_best(candidates, scores, out):
    """For each node and slot, the candidate with the second best score; the earlier one wins a tie."""
    count, slots, per_slot = candidates.shape
    for node in prange(count):
        for slot in range(slots):
            best = -1
            second = -1
            for entry in range(per_slot):
                score = scores[node, slot, entry]
                if best < 0 or score > scores[node, slot, best]:
                    second = best
                    best = entry
                elif second < 0 or score > scores[node, slot, second]:
                    second = entry
            out[node, slot] = candidates[node, slot, second]


def pick_mid_near(candidates, scores):
    """(count, slots) int32: the second best sampled node of each slot by score."""
    count, slots, _ = candidates.shape
    out = np.empty((count, slots), dtype=np.int32)
    if count and slots:
        _pick_second_best(np.ascontiguousarray(candidates),
                          np.ascontiguousarray(scores, dtype=np.float64), out)
    return out


def random_mid_near_pairs(ids, neighbors, seed):
    """Mid-near pairs without scores: every partner is a random sampled node. (pairs, 2) int32."""
    ids = np.asarray(ids, dtype=np.int64)
    candidates = mid_near_candidates(len(ids), neighbors, seed)
    slots = candidates.shape[1]
    if slots == 0:
        return np.zeros((0, 2), dtype=np.int32)
    picked = ids[pick_mid_near(candidates, np.full(candidates.shape, -np.inf))]
    return np.column_stack((np.repeat(ids, slots).astype(np.int32), picked.ravel().astype(np.int32)))


def mid_near_pairs(source, index_map, node_count, ids, neighbors, seed, *, workers=None,
                   cancelled=None, lookup_options=None):
    """The mid-near pairs of a layout: (pairs, 2) int32 of node numbers in the network.

    ``source`` is a ScoreSource (or anything with its read and edge_count) over the pairs
    of the nodes in the layout; ``index_map`` takes its stored node numbers to numbers
    below ``node_count`` (None: the same); ``ids`` are the connected nodes, the only
    ones that take part. Each of them gets ``mid_near_count(neighbors)`` pairs (fewer in
    a very small network), found by sampling six nodes per pair and keeping the second
    best by stored score. One pass over the stored pairs looks the scores up.
    """
    ids = np.asarray(ids, dtype=np.int64)
    count = len(ids)
    with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
        candidates = mid_near_candidates(count, neighbors, seed)
    slots = candidates.shape[1]
    if slots == 0:
        return np.zeros((0, 2), dtype=np.int32)
    owners = np.broadcast_to(ids[:, None, None], candidates.shape)
    sampled = ids[candidates]
    lookup = PairLookup(owners.ravel(), sampled.ravel(), node_count, **(lookup_options or {}))
    stream_pair_scores(source, index_map, lookup, workers=workers, cancelled=cancelled)
    scores = lookup.scores_of(owners.ravel(), sampled.ravel()).reshape(candidates.shape)
    with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
        picked = ids[pick_mid_near(candidates, scores)]
    owner_column = np.repeat(ids, slots).astype(np.int32)
    return np.column_stack((owner_column, picked.ravel().astype(np.int32)))


# --- Incidence lists --------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def _incidence(first, second, count):
    """CSR lists of the pairs at each node: (pointer, the node at the other end).

    A pair is in the lists of both its nodes. Within a node's list the pairs are in the order
    of the pair rows, so the lists do not depend on anything but the rows.
    """
    pointer = np.zeros(count + 1, dtype=np.int64)
    for row in range(len(first)):
        pointer[first[row] + 1] += 1
        pointer[second[row] + 1] += 1
    for node in range(count):
        pointer[node + 1] += pointer[node]
    fill = pointer[:-1].copy()
    other = np.empty(pointer[count], dtype=np.int32)
    for row in range(len(first)):
        a = first[row]
        b = second[row]
        other[fill[a]] = b
        fill[a] += 1
        other[fill[b]] = a
        fill[b] += 1
    return pointer, other


def incidence_lists(first, second, count):
    """(pointer, partners) of the pairs (first[r], second[r]) over ``count`` nodes."""
    return _incidence(np.ascontiguousarray(first, dtype=np.int32),
                      np.ascontiguousarray(second, dtype=np.int32), int(count))


def rows_of_partners(partners):
    """The pair rows (first, second) of a (count, slots) partner table, -1 entries left out."""
    count, slots = partners.shape
    owner = np.repeat(np.arange(count, dtype=np.int32), slots)
    other = np.ascontiguousarray(partners).ravel()
    present = other >= 0
    return owner[present], other[present]


__all__ = [
    "FURTHER_RATIO", "MID_NEAR_CANDIDATES", "mid_near_count", "further_count",
    "effective_mid_near_count", "random_seed", "connected_nodes", "near_partners", "further_partners",
    "mid_near_candidates", "PairLookup", "stream_pair_scores", "pick_mid_near", "mid_near_pairs",
    "random_mid_near_pairs", "normal_start",
    "incidence_lists", "rows_of_partners",
]
