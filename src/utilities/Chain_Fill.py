# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Chain fill: the missing pairs of a sparse network, estimated from chains of computed pairs.

The computed set C is the network's edges plus the calibration pairs aligned at the start of the run, each
with its normalized score y. The chain value of a pair (i, j) is

    b_ij = max over k of min(y_ik, y_kj),   over every k for which (i, k) and (k, j) are both in C.

A pair with no such k is uncovered. A pair that is itself in C never uses its own value (the diagonal is
"not computed"). With base the calibrated isotonic prediction from the similarity, and for the calibration
pairs c_oof the out-of-fold isotonic prediction, a weighted least-squares fit on the calibration pairs of

    y ~ c0 + c1 * c_oof + c2 * (b if covered else c_oof) + c3 * [uncovered]

gives four coefficients, and every missing pair is filled with

    c0 + c1 * base + c2 * (b if covered else base) + c3 * [uncovered].

The chain step is exact: each sequence has the list of its computed neighbours sorted by score, highest
first, and a bit row of them; for a pair an empty AND of the two bit rows means uncovered, otherwise the walk
takes the shorter list and stops at the first score that is not above the best chain value found, because no
later term can exceed it. That gives every bit the full scan over k gives. No kernel uses fastmath, each
pair is computed on its own, and sums are taken in numpy, so no result depends on the number of threads.

Memory: the (n, n) float32 score matrix, the lists (int32 index and float32 score, both directions of every
computed pair), the bit rows, and row blocks of the fill; never an (n, n) float64 array. estimate_bytes
gives the total.

This module is used by Embedding_MSA and imports no tool.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from utilities import Numba_Threads

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a pinned requirement
    psutil = None


NEG = np.float32(-1e30)
HALF_NEG = np.float32(-5e29)

# The most pairs of one row block of the fill (the similarity values, the isotonic predictions and the
# temporaries of the prediction are held for one block at a time).
BLOCK_PAIRS = 2_000_000
# Bytes a block holds per pair (the authors' 16 to 20 per query pair are index arrays, which this fill does not
# make): the float64 similarity and prediction of each side (32, the second side only for a similarity matrix
# that is not symmetric) and about three float64 temporaries of the isotonic prediction (24).
BLOCK_BYTES_PER_PAIR = 56

# What the step may not take from the memory the machine has free: the rule the alignment engine's host cache
# follows (Embedding_Alignment_Engine.resolve_host_cache_bytes): at least 8 GiB, or a quarter of the total.
GIB = 1024 ** 3
MIN_HOST_RESERVE = 8 * GIB
HOST_RESERVE_FRACTION = 0.25


# ----------------------------------------------------------------------------------------------- kernels
@njit(cache=True)
def _scatter_scores(edge_i, edge_j, y, square):
    """Write each edge's score at (i, j) and (j, i); in edge order, so a pair listed twice keeps its last score."""
    for k in range(edge_i.size):
        i = edge_i[k]
        j = edge_j[k]
        if i != j:
            square[i, j] = y[k]
            square[j, i] = y[k]


@njit(parallel=True, cache=True)
def _degrees(square, counts):
    n = square.shape[0]
    for i in prange(n):
        c = 0
        for k in range(n):
            if square[i, k] > HALF_NEG:
                c += 1
        counts[i] = c


@njit(parallel=True, cache=True)
def _fill_lists(square, start, idx, val):
    """Row i's computed neighbours, by score descending, into idx and val at start[i]:start[i + 1]."""
    n = square.shape[0]
    for i in prange(n):
        s = start[i]
        m = start[i + 1] - s
        if m == 0:
            continue
        tk = np.empty(m, dtype=np.int32)
        tv = np.empty(m, dtype=np.float32)
        c = 0
        for k in range(n):
            if square[i, k] > HALF_NEG:
                tk[c] = k
                tv[c] = square[i, k]
                c += 1
        order = np.argsort(tv, kind="mergesort")
        for q in range(m):
            o = order[m - 1 - q]
            idx[s + q] = tk[o]
            val[s + q] = tv[o]


@njit(parallel=True, cache=True)
def _fill_bits(square, bits):
    n = square.shape[0]
    for i in prange(n):
        for w in range(bits.shape[1]):
            bits[i, w] = 0
        for k in range(n):
            if square[i, k] > HALF_NEG:
                bits[i, k >> 6] |= np.uint64(1) << np.uint64(k & 63)


@njit(cache=True)
def _pair_chain(square, start, idx, val, bits, i, j):
    """The chain value of pair (i, j): NEG when no k has both (i, k) and (k, j) computed."""
    words = bits.shape[1]
    shared = False
    for w in range(words):
        if bits[i, w] & bits[j, w]:
            shared = True
            break
    best = NEG
    if shared:
        # Walk the shorter list.
        if start[j + 1] - start[j] < start[i + 1] - start[i]:
            t = i
            i = j
            j = t
        s = start[i]
        e = start[i + 1]
        q = s
        while q < e:
            a = val[q]
            if a <= best:
                break
            b = square[j, idx[q]]
            v = a if a < b else b
            if v > best:
                best = v
            q += 1
    return best


@njit(parallel=True, cache=True)
def _chain_pairs(square, start, idx, val, bits, rows, cols, out):
    for t in prange(rows.size):
        out[t] = _pair_chain(square, start, idx, val, bits, rows[t], cols[t])


@njit(parallel=True, cache=True)
def _gather_upper(x, r0, r1, offsets, out):
    """The similarities of the pairs (a, b), a in r0..r1 - 1 and b > a, in condensed order."""
    n = x.shape[0]
    for t in prange(r1 - r0):
        a = r0 + np.int64(t)
        p = offsets[t]
        for b in range(a + 1, n):
            out[p] = x[a, b]
            p += 1


@njit(cache=True)
def _one_distance(c0, c1, c2, c3, bs, chain, has_chain, u, max_norm):
    """float32(max(0, max_norm - pred)) for one base prediction bs (the tool's inversion of a fill value)."""
    w = np.float64(chain) if has_chain else bs
    pred = c0 + c1 * bs + c2 * w + c3 * u
    d = max_norm - pred
    if d < 0.0:
        d = 0.0
    return np.float32(d)


@njit(parallel=True, cache=True)
def _gather_lower(x, r0, r1, offsets, out):
    """The similarities x[b, a] of the same pairs, for a similarity matrix that is not symmetric."""
    n = x.shape[0]
    for t in prange(r1 - r0):
        a = r0 + np.int64(t)
        p = offsets[t]
        for b in range(a + 1, n):
            out[p] = x[b, a]
            p += 1


@njit(parallel=True, cache=True)
def _symmetry_flags(x, flags):
    """flags[t] is 1 when no element of the 64-row tile t differs from its mirror (tiled for the cache)."""
    n = x.shape[0]
    tile = 64
    for t in prange(flags.size):
        r0 = np.int64(t) * tile
        r1 = min(r0 + tile, n)
        same = 1
        for c0 in range(0, n, tile):
            c1 = min(c0 + tile, n)
            for a in range(r0, r1):
                for b in range(c0, c1):
                    if x[a, b] != x[b, a]:
                        same = 0
        flags[t] = same


@njit(parallel=True, cache=True)
def _fill_block(square, start, idx, val, bits, r0, r1, offsets, base, base_mirror, c0, c1, c2, c3, max_norm,
                out, covered_rows, missing_rows):
    """Distances of the pairs of rows r0..r1 - 1 into out (condensed order; computed pairs get 0.0, to be overwritten).

    A pair's distance is 0.5 * (d_ab + d_ba) in float32, the tool's own symmetrising of the dense matrix, where
    d_ab is made from base (the isotonic prediction at x[a, b]) and d_ba from base_mirror (at x[b, a]); the chain
    value is one for both. With a symmetric similarity base_mirror is base, and the average of two equal
    float32 values is exactly that value.
    """
    n = square.shape[0]
    for t in prange(r1 - r0):
        a = r0 + np.int64(t)
        p = offsets[t]
        covered = 0
        missing = 0
        for b in range(a + 1, n):
            if square[a, b] > HALF_NEG:
                out[p] = 0.0
            else:
                missing += 1
                chain = _pair_chain(square, start, idx, val, bits, a, b)
                has_chain = chain > HALF_NEG
                if has_chain:
                    covered += 1
                    u = 0.0
                else:
                    u = 1.0
                d_ab = _one_distance(c0, c1, c2, c3, base[p], chain, has_chain, u, max_norm)
                d_ba = _one_distance(c0, c1, c2, c3, base_mirror[p], chain, has_chain, u, max_norm)
                out[p] = np.float32(0.5) * (d_ab + d_ba)
            p += 1
        covered_rows[t] = covered
        missing_rows[t] = missing


# ---------------------------------------------------------------------------------------------- building
def _threads():
    return Numba_Threads.limited_threads(Numba_Threads.default_thread_count())


@dataclass
class ChainIndex:
    """Each sequence's computed neighbours, best score first (start, idx, val), and their bit rows."""

    start: np.ndarray
    idx: np.ndarray
    val: np.ndarray
    bits: np.ndarray


def score_matrix(n, edge_i, edge_j, y):
    """The (n, n) float32 matrix of computed scores, NEG where a pair is not computed and on the diagonal.

    A pair listed more than once keeps the score of its last listing, as populate_condensed_matrix does
    for the distance the tree's baseline holds for it.
    """
    square = np.full((n, n), NEG, dtype=np.float32)
    # The edge arrays are used as they are (int32 in a run): a converted copy would cost 8 bytes an edge.
    _scatter_scores(
        np.ascontiguousarray(edge_i),
        np.ascontiguousarray(edge_j),
        np.ascontiguousarray(y, dtype=np.float32),
        square,
    )
    return square


def build_index(square):
    """The neighbour lists and bit rows of a score matrix."""
    n = square.shape[0]
    counts = np.empty(n, dtype=np.int64)
    with _threads():
        _degrees(square, counts)
        start = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=start[1:])
        idx = np.empty(int(start[-1]), dtype=np.int32)
        val = np.empty(int(start[-1]), dtype=np.float32)
        _fill_lists(square, start, idx, val)
        bits = np.empty((n, (n + 63) // 64), dtype=np.uint64)
        _fill_bits(square, bits)
    return ChainIndex(start, idx, val, bits)


def chain_values(square, index, rows, cols):
    """The chain value of each pair (rows[t], cols[t]) as float32: NEG where uncovered."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    out = np.empty(rows.size, dtype=np.float32)
    with _threads():
        _chain_pairs(square, index.start, index.idx, index.val, index.bits, rows, cols, out)
    return out


def is_symmetric(matrix):
    """Whether every element of the square matrix equals its mirror (exactly)."""
    n = matrix.shape[0]
    flags = np.ones((n + 63) // 64, dtype=np.int64)
    with _threads():
        _symmetry_flags(matrix, flags)
    return bool(flags.all())


def is_covered(chain):
    """Which chain values belong to a pair with a chain."""
    return np.asarray(chain) > HALF_NEG


# ------------------------------------------------------------------------------------------------ memory
def estimate_bytes(n, edges, block_pairs=BLOCK_PAIRS):
    """Peak bytes of the chain fill's own arrays for n sequences and at most `edges` computed pairs.

    The score matrix (4 n^2), the lists (an int32 index and a float32 score, 8 bytes, for each of the two
    directions of a pair: 16 per pair) with their offsets (8 (n + 1)), the bit rows (8 n ceil(n / 64)), one
    row block of the fill (BLOCK_BYTES_PER_PAIR per pair of the block) and the condensed float32 distances the
    fill writes (4 per pair of sequences). The similarity matrix the tool already holds is not included.
    """
    n = int(n)
    pairs = n * (n - 1) // 2
    return (
        4 * n * n
        + 16 * int(edges)
        + 8 * (n + 1)
        + 8 * n * ((n + 63) // 64)
        + BLOCK_BYTES_PER_PAIR * min(int(block_pairs), pairs)
        + 4 * pairs
    )


def available_memory_bytes():
    """(total, available) physical memory in bytes, (0, 0) when it cannot be read. Tests patch this."""
    if psutil is None:
        return 0, 0
    try:
        memory = psutil.virtual_memory()
    except (psutil.Error, OSError):
        return 0, 0
    return int(memory.total), int(memory.available)


@dataclass
class MemoryVerdict:
    fits: bool
    need: int
    usable: int
    available: int
    reserve: int


def memory_verdict(n, edges, block_pairs=BLOCK_PAIRS):
    """Whether estimate_bytes fits in the memory free now minus the reserve. Unreadable memory counts as not fitting."""
    total, available = available_memory_bytes()
    reserve = max(MIN_HOST_RESERVE, int(total * HOST_RESERVE_FRACTION)) if total else MIN_HOST_RESERVE
    usable = max(0, available - reserve) if available else 0
    need = estimate_bytes(n, edges, block_pairs)
    return MemoryVerdict(need <= usable, need, usable, available, reserve)


# ------------------------------------------------------------------------------------------------ the fit
def combine(coefficients, base, chain):
    """The fill value: c0 + c1 * base + c2 * (chain if covered else base) + c3 * [uncovered], in float64."""
    base = np.asarray(base, dtype=np.float64)
    chain = np.asarray(chain, dtype=np.float32)
    covered = is_covered(chain)
    return (
        coefficients[0]
        + coefficients[1] * base
        + coefficients[2] * np.where(covered, chain.astype(np.float64), base)
        + coefficients[3] * (~covered)
    )


def fit_coefficients(out_of_fold, chain, y, weights):
    """The four coefficients, by weighted least squares on the calibration pairs (float64)."""
    out_of_fold = np.asarray(out_of_fold, dtype=np.float64)
    chain = np.asarray(chain, dtype=np.float32)
    covered = is_covered(chain)
    design = np.column_stack([
        np.ones(out_of_fold.size),
        out_of_fold,
        np.where(covered, chain.astype(np.float64), out_of_fold),
        (~covered).astype(np.float64),
    ])
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], np.asarray(y, dtype=np.float64) * root, rcond=None)
    return coefficients


def held_out_predictions(out_of_fold, chain, y, weights):
    """The chain + calibration estimates of the odd-position calibration pairs, from coefficients fitted on the even half.

    Positions are those of the sample order. The coefficients are fitted on the even half (its out-of-fold
    isotonic predictions, chain values and weights, by lstsq, the fourth column kept even when it is all
    zero) and applied to the odd half's [1, c_oof, chain if covered else c_oof, uncovered]. The odd half is
    the one the isotonic held-out check scores, so the two checks compare. The scheme has one small indirect
    leak: the even half's out-of-fold predictions come from the isotonic fit made on the odd half, so the
    evaluated half's targets reach the four coefficients indirectly (small with four coefficients and
    thousands of pairs). No nested scheme removes it.
    """
    out_of_fold = np.asarray(out_of_fold, dtype=np.float64)
    chain = np.asarray(chain, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    even = np.arange(0, out_of_fold.size, 2)
    odd = np.arange(1, out_of_fold.size, 2)
    coefficients = fit_coefficients(out_of_fold[even], chain[even], y[even], weights[even])
    return combine(coefficients, out_of_fold[odd], chain[odd])


@dataclass
class ChainPlan:
    """A prepared chain fill: the score matrix and index it reads, and what it learnt from the calibration pairs."""

    n: int
    square: np.ndarray
    index: ChainIndex
    coefficients: np.ndarray
    calibration_chain: np.ndarray
    calibration_coverage: float
    held_out: np.ndarray

    def release(self):
        """Drop the arrays of the fill (the matrix, lists and bit rows)."""
        self.square = None
        self.index = None


def prepare(n, edge_i, edge_j, y, calibration_positions, out_of_fold, calibration_weights):
    """Build the chain index over the computed pairs and fit the coefficients on the calibration pairs.

    edge_i, edge_j and y are every computed pair (the calibration pairs last), calibration_positions the
    calibration pairs' places among them in sample order, out_of_fold their out-of-fold isotonic
    predictions and calibration_weights how many missing pairs each stands for.
    """
    square = score_matrix(n, edge_i, edge_j, y)
    index = build_index(square)
    positions = np.asarray(calibration_positions, dtype=np.int64)
    rows = np.asarray(edge_i)[positions]
    cols = np.asarray(edge_j)[positions]
    chain = chain_values(square, index, rows, cols)
    target = np.asarray(y)[positions]
    weights = np.asarray(calibration_weights, dtype=np.float64)
    coefficients = fit_coefficients(out_of_fold, chain, target, weights)
    held_out = held_out_predictions(out_of_fold, chain, target, weights)
    coverage = float(np.count_nonzero(is_covered(chain))) / max(chain.size, 1)
    return ChainPlan(n, square, index, coefficients, chain, coverage, held_out)


# ------------------------------------------------------------------------------------------ the fill
def _row_blocks(n, block_pairs):
    """Consecutive row ranges (r0, r1) covering rows 0..n-2, each with about block_pairs pairs or one row."""
    blocks = []
    r0 = 0
    pairs = 0
    for a in range(n - 1):
        pairs += n - 1 - a
        if pairs >= block_pairs:
            blocks.append((r0, a + 1))
            r0 = a + 1
            pairs = 0
    if r0 < n - 1:
        blocks.append((r0, n - 1))
    return blocks


def fill_distances(plan, condensed, similarity, iso, max_norm_score, block_pairs=BLOCK_PAIRS):
    """Write the distance max(0, max_norm_score - fill) of every missing pair into `condensed` (float32).

    similarity is the (n, n) matrix the isotonic fit was made on, iso that fit. Computed pairs get 0.0 and
    must be overwritten by the caller with their own distances. Row blocks keep every temporary small.
    Returns (covered, missing): how many of the missing pairs had a chain, and how many there were.
    """
    n = plan.n
    index = plan.index
    c0, c1, c2, c3 = (float(c) for c in plan.coefficients)
    max_norm = float(max_norm_score)
    covered_total = 0
    missing_total = 0
    symmetric = is_symmetric(similarity)
    for r0, r1 in _row_blocks(n, block_pairs):
        rows = np.arange(r0, r1, dtype=np.int64)
        sizes = n - 1 - rows
        offsets = np.zeros(rows.size, dtype=np.int64)
        np.cumsum(sizes[:-1], out=offsets[1:])
        pairs = int(sizes.sum())
        first = n * r0 - r0 * (r0 + 1) // 2  # the condensed index of the block's first pair
        similarities = np.empty(pairs, dtype=np.float64)
        covered_rows = np.empty(rows.size, dtype=np.int64)
        missing_rows = np.empty(rows.size, dtype=np.int64)
        with _threads():
            _gather_upper(similarity, r0, r1, offsets, similarities)
            base = np.asarray(iso.predict(similarities), dtype=np.float64)
            if symmetric:
                base_mirror = base
            else:
                _gather_lower(similarity, r0, r1, offsets, similarities)
                base_mirror = np.asarray(iso.predict(similarities), dtype=np.float64)
            del similarities
            _fill_block(
                plan.square, index.start, index.idx, index.val, index.bits, r0, r1, offsets, base, base_mirror,
                c0, c1, c2, c3, max_norm, condensed[first:first + pairs], covered_rows, missing_rows,
            )
        covered_total += int(covered_rows.sum())
        missing_total += int(missing_rows.sum())
    return covered_total, missing_total
