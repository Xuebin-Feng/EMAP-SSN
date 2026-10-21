# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Edges at or above a score threshold, and the threshold of a network's best edges, streamed.

A layout keeps the pairs scoring at least ``SIMILARITY_THRESHOLD``, a number the user
gives or one the best ``TOP_EDGE_PERCENT`` of the possible edges sets. A network can
hold close to a billion pairs, so nothing here keeps a column of the file: the pairs
are read and scored a slice at a time (utilities.Network_Score_Statistics.ScoreSource),
and what a pass keeps is only what the next step needs. Loading the columns took 15 to
28 bytes per pair, 14 to 27 GB at 974 million pairs.

Both answers are exactly what loading every score gave:

* ``edges_at_least`` applies the same comparison to each slice, ``scores >= threshold``
  on the scores in the dtype the normalization gives them, so each pair is kept or
  dropped as before; the kept pairs come back in the order they are stored.
* ``cutoff_of_best_edges`` is ``float(np.partition(scores, n - e)[n - e])``: the e-th
  largest score, found by radix selection (Network_Score_Statistics.RankSelector, one
  16-bit digit of the scores' keys per pass: two passes for float32 scores, four for
  float64). NumPy orders NaN after every number, so NaN scores are counted apart and
  take the highest ranks; with that many, the cutoff is NaN, as it was.

The one value this cannot reproduce is the sign of a zero cutoff when scores of -0.0
and +0.0 both lie at the cutoff rank: np.partition leaves either there, depending on its
partitioning, and the two compare equal as a threshold.

This module imports no Qt and no Numba, so the Viewer, the layout job and the tests use
it as it is.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from utilities import Network_Score_Statistics as scan

# The reads of the worker threads share the HDF5 library's lock, so beyond a few threads a
# pass only gets more slices in flight (see Network_Top_Edges.DEFAULT_MAX_WORKERS).
DEFAULT_MAX_WORKERS = 8


def _pass_workers(workers):
    return max(1, int(workers or min(scan.worker_count(), DEFAULT_MAX_WORKERS)))


def _rankable(scores):
    """(the scores that are numbers, as float32 or float64; how many were NaN).

    The selector's keys order numbers only; NumPy sorts NaN after them. Integer scores are
    ranked as float64, which keeps their order, so the rounded value of the selected score
    is the value float() makes of it.
    """
    values = scores if scores.dtype == np.float32 else scores.astype(np.float64, copy=False)
    is_nan = np.isnan(values)
    nan_count = int(np.count_nonzero(is_nan))
    if nan_count:
        values = values[~is_nan]
    return values, nan_count


def cutoff_of_best_edges(source, edge_count_for, *, workers=None):
    """(the number of pairs, the score of the e-th largest pair), without keeping the scores.

    e is ``edge_count_for(pairs)`` limited to 1 .. pairs. The cutoff is
    ``float(np.partition(scores, pairs - e)[pairs - e])`` of all the scores, and 0.0
    when there are no pairs. ``edge_count_for`` is called once, after the first pass.
    """
    ranges = scan.chunk_ranges(source.edge_count)
    limit = _pass_workers(workers)

    def first_pass(start, stop):
        # The first look at every pair also checks that the pair indices fit the headers.
        scores = source.read(start, stop, check_nodes=True).scores
        if len(scores) == 0:
            return None
        values, nan_count = _rankable(scores)
        return len(scores), nan_count, scan.RankSelector.first_digit_counts(values), values.dtype

    pairs = nan_total = 0
    dtype = None
    digits = np.zeros(scan.RankSelector.SIZE, dtype=np.int64)
    for result in scan.run_pass(first_pass, ranges, workers=limit):
        if result is None:
            continue
        pairs += result[0]
        nan_total += result[1]
        digits += result[2]
        dtype = result[3]

    requested = edge_count_for(pairs)
    if pairs == 0:
        return 0, 0.0
    best = max(1, min(int(requested), pairs))
    rank = pairs - best
    if rank >= pairs - nan_total:
        return pairs, float("nan")

    selector = scan.RankSelector([rank], dtype)
    selector.advance(digits)
    for level in range(1, selector.levels):
        def pass_at_level(start, stop, level=level):
            scores = source.read(start, stop).scores
            if len(scores) == 0:
                return None
            return selector.count(_rankable(scores)[0], level)

        counts = selector.empty_histogram(level)
        for part in scan.run_pass(pass_at_level, ranges, workers=limit):
            if part is not None:
                selector.merge(counts, part, level)
        selector.advance(counts)
    return pairs, float(selector.values[rank])


def edges_at_least(source, threshold, index_map, *, workers=None):
    """``(edges, scores)`` of the pairs scoring at least ``threshold``, in their stored order.

    edges is an int32 array of ``index_map[source], index_map[target]`` rows; scores keep
    the dtype of the normalized scores. The comparison is ``scores >= threshold`` for
    each slice, whatever threshold is, so an unusable one (None) fails as it did.
    """
    ranges = scan.chunk_ranges(source.edge_count)

    def pick(start, stop):
        piece = source.read(start, stop, nodes=True, check_nodes=True)
        wanted = piece.scores >= threshold
        return (index_map[piece.sources[wanted]], index_map[piece.targets[wanted]],
                piece.scores[wanted])

    pieces = deque(scan.run_pass(pick, ranges, workers=_pass_workers(workers)))
    if not pieces:
        # No pairs at all: the comparison still runs, on the empty scores of the right dtype.
        empty = source.read(0, 0, nodes=True)
        wanted = empty.scores >= threshold
        pieces.append((index_map[empty.sources[wanted]], index_map[empty.targets[wanted]],
                       empty.scores[wanted]))
    total = sum(len(piece[2]) for piece in pieces)
    edges = np.empty((total, 2), dtype=np.int32)
    scores = np.empty(total, dtype=pieces[0][2].dtype)
    filled = 0
    while pieces:  # each piece is released as it is copied
        first, second, piece_scores = pieces.popleft()
        stop = filled + len(piece_scores)
        edges[filled:stop, 0] = first
        edges[filled:stop, 1] = second
        scores[filled:stop] = piece_scores
        filled = stop
    return edges, scores
