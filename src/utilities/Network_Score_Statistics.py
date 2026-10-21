# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Score statistics of a similarity network, computed in bounded memory.

Both Config windows ("Compute Network Statistics") and the MCP ``network_statistics``
action read a network's stored pairs, score them, and summarise the scores. A network
can hold close to a billion pairs, so nothing here keeps a column of the file: the
pairs are read a slice (chunk) at a time, scored, reduced, and dropped. The memory
used is a few chunks per worker thread, whatever the size of the network.

What is exact, and how:

* Counts, the minimum, the maximum and the stored-edge count are exact. The scores of
  a chunk are computed with the same expressions, in the same dtypes, as when a whole
  column was scored at once, so each score is the same number.
* A threshold count is "scores >= threshold". float32 scores met a float64
  threshold, so numpy compared them as float64; the count here compares the
  float32 scores with the smallest float32 not below the threshold (ceil_float32),
  which gives the same answer without widening the scores.
* The mean is a float64 running sum divided by the count. It can differ from numpy's
  float32 pairwise mean in the last digits (relative 1e-7 or less).
* Order statistics (RankSelector) are exact: a radix selection on the order-preserving
  integer key of the scores, one 16-bit digit per pass over the data.

This module imports no Qt, so the VR Config, the MCP helper process and the tests use
it as it is.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys
import threading

import numpy as np

# Pairs per chunk. A worker holds a few arrays of this length (about 30 B per pair for
# one alignment chunk, with its column buffers), so 18 workers hold about 0.3 GB.
# Measured at 100M pairs: 500,000 is as fast as 1,000,000 or 2,000,000 and a third smaller.
CHUNK_EDGES = 500_000
EPSILON = 1e-6
SEQUENCE_NORMALIZATIONS = frozenset({"shorter_sequence", "longer_sequence", "average_sequence"})
# The CPUs left free for the desktop, like Numba_Threads.RESERVED_CPUS. That module
# imports Numba, which these windows do not otherwise load, so the rule is restated
# here and a test keeps the two equal.
RESERVED_CPUS = 2


class StatisticsCancelled(Exception):
    """A statistics run was stopped by its owner (progress or cancelled said so)."""


def usable_cpu_count():
    counter = getattr(os, "process_cpu_count", None)
    if counter is not None:
        count = counter()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, int(count or 1))


def worker_count():
    """The worker threads to use: every CPU this process may run on but two, at least one."""
    return max(1, usable_cpu_count() - RESERVED_CPUS)


def chunk_ranges(total, chunk_edges=None):
    """The (start, stop) slices of ``total`` pairs, ``chunk_edges`` each."""
    chunk = max(1, int(chunk_edges or CHUNK_EDGES))
    return [(start, min(start + chunk, total)) for start in range(0, int(total), chunk)]


# --- Threshold comparison without widening the scores -------------------------------


def ceil_float32(thresholds):
    """The smallest float32 that is not below each (float64) threshold.

    For a float32 score s, ``s >= t`` compared as float64 is the same as
    ``s >= ceil_float32(t)`` compared as float32. Rounding to the nearest float32
    instead would change counts for scores sitting on a threshold.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    rounded = thresholds.astype(np.float32)
    low = rounded.astype(np.float64) < thresholds
    rounded[low] = np.nextafter(rounded[low], np.float32(np.inf))
    return rounded


def cutoffs_for(thresholds, dtype):
    """``thresholds`` in the form to compare with scores of ``dtype`` (see ceil_float32)."""
    if np.dtype(dtype) == np.float32:
        return ceil_float32(thresholds)
    return np.asarray(thresholds, dtype=np.float64)


def count_at_least(sorted_scores, cutoffs):
    """How many of the sorted scores are >= each cutoff."""
    return len(sorted_scores) - np.searchsorted(sorted_scores, cutoffs, side="left")


# --- Nodes ----------------------------------------------------------------------------


def read_fasta_names(fasta_path):
    """The ids and the full headers of a FASTA file's records, as Bio.SeqIO reads them."""
    from Bio import SeqIO

    ids, headers = set(), set()
    # The default encoding, as SeqIO.parse(path) opens the file, but closed when done.
    with open(fasta_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.add(record.id)
            headers.add(record.description)
    return ids, headers


def kept_node_mask(network_headers, fasta_ids, fasta_headers):
    """Which network nodes a FASTA keeps, and how many of its names the network lacks.

    A node is kept when its whole header is one of the FASTA's headers or its first
    word is one of the FASTA's ids. Returns (mask, missing).
    """
    network_header_set = set(network_headers)
    network_id_set = {header.split()[0] for header in network_headers}
    missing = sum(
        1 for name in fasta_ids if name not in network_id_set and name not in network_header_set
    )
    mask = np.zeros(len(network_headers), dtype=bool)
    kept = [
        index for index, header in enumerate(network_headers)
        if header in fasta_headers or header.split()[0] in fasta_ids
    ]
    mask[kept] = True
    return mask, missing


# --- Reading and scoring a slice ------------------------------------------------------


def statistics_normalize(raw_scores, alignment_lengths, source_lengths, target_lengths, mode):
    """The Config statistics' normalization: float32 throughout, no division by less than 1e-6."""
    if mode == "alignment_length":
        denominator = alignment_lengths
    elif mode == "shorter_sequence":
        denominator = np.minimum(source_lengths, target_lengths)
    elif mode == "longer_sequence":
        denominator = np.maximum(source_lengths, target_lengths)
    elif mode == "average_sequence":
        denominator = (source_lengths + target_lengths) / 2.0
    else:
        denominator = alignment_lengths
    denominator = np.maximum(denominator, EPSILON)
    return (raw_scores / denominator).astype(np.float32)


class Slice:
    """The scored pairs of one slice: scores, and the nodes and stored positions of each pair when asked for."""

    __slots__ = ("scores", "sources", "targets", "positions")

    def __init__(self, scores, sources=None, targets=None, positions=None):
        self.scores = scores
        self.sources = sources
        self.targets = targets
        self.positions = positions


class ScoreSource:
    """The scores of a network's stored pairs, read and scored a slice at a time.

    kept_mask drops pairs with an end outside the kept nodes. normalize is the
    normalization (raw scores, alignment lengths, source lengths, target lengths,
    mode); the Config statistics use statistics_normalize with cast_float32, which
    reads float32 columns as the Config always has, and the layout's own
    normalization (desktop.Viewer_State._normalize_score) takes the columns as
    stored. Only the columns the settings need are read: the alignment-length column
    only when the normalization divides by it, the sequence lengths only when it
    divides by those, and the pair indices only for the mask, for sequence lengths,
    for the node check (check_nodes=True, against node_count), or when a caller asks
    for them (nodes=True).
    """

    def __init__(self, network, *, is_blast, score_mode="global", norm_mode="alignment_length",
                 kept_mask=None, normalize=statistics_normalize, cast_float32=True,
                 node_count=None):
        self.network = network
        self.is_blast = is_blast
        self.norm_mode = norm_mode
        self.kept_mask = kept_mask
        self.normalize = normalize
        self.cast_float32 = cast_float32
        # When set, read(check_nodes=True) checks each pair index against it, as a layout job does.
        self.node_count = node_count
        self.edge_count = int(network["i"].shape[0])
        self.score_name = "score" if is_blast else ("l_score" if score_mode == "local" else "g_score")
        self.length_name = "l_len" if score_mode == "local" else "g_len"
        self.by_sequence = (not is_blast) and norm_mode in SEQUENCE_NORMALIZATIONS
        self.by_alignment = (not is_blast) and not self.by_sequence
        self._sequence_lengths = None
        self._buffers = threading.local()

    @property
    def sequence_lengths(self):
        if self._sequence_lengths is None:
            if "seq_lens" in self.network:
                self._sequence_lengths = self.network["seq_lens"][:]
            else:
                # No lengths stored: every sequence counts as one residue.
                highest = 0
                for start, stop in chunk_ranges(self.edge_count):
                    highest = max(highest, int(self.network["i"][start:stop].max(initial=0)),
                                  int(self.network["j"][start:stop].max(initial=0)))
                self._sequence_lengths = np.ones(highest + 1)
        return self._sequence_lengths

    def _needs_indices(self, nodes, check_nodes):
        return (nodes or self.kept_mask is not None or self.by_sequence
                or (check_nodes and self.node_count is not None))

    def _read(self, name, start, stop):
        """Pairs start:stop of a column, read into this thread's own buffer for it.

        Reading into a buffer the thread keeps spares every slice the cost of a fresh
        array (its pages are faulted in by the first write), which is about half of
        what reading a column costs. The result is valid until this thread reads the
        same column again, so read() never hands it on without a copy.
        """
        dataset = self.network[name]
        count = stop - start
        if count <= 0:
            return np.empty(0, dtype=dataset.dtype)
        buffers = self._buffers.__dict__
        buffer = buffers.get(name)
        if buffer is None or len(buffer) < count:
            buffer = buffers[name] = np.empty(count, dtype=dataset.dtype)
        dataset.read_direct(buffer, source_sel=np.s_[start:stop], dest_sel=np.s_[0:count])
        return buffer[:count]

    def read(self, start, stop, nodes=False, check_nodes=False, positions=False):
        """The scored pairs of slice start:stop. check_nodes also checks the pair indices against node_count.

        positions=True also returns where each kept pair sits among the stored pairs
        (start + its offset), as int64: the pair order a caller can break ties by.
        """
        raw_scores = self._read(self.score_name, start, stop)
        if not self.is_blast and self.cast_float32:
            raw_scores = raw_scores.astype(np.float32, copy=False)
        sources = targets = None
        if self._needs_indices(nodes, check_nodes):
            sources = self._read("i", start, stop)
            targets = self._read("j", start, stop)
            if check_nodes and self.node_count is not None and len(sources) and max(
                int(sources.max()), int(targets.max())
            ) >= self.node_count:
                raise IndexError("Network pair indices exceed the header count.")
        alignment_lengths = None
        if self.by_alignment:
            alignment_lengths = self._read(self.length_name, start, stop)
            if self.cast_float32:
                alignment_lengths = alignment_lengths.astype(np.float32)
        pair_positions = None
        if self.kept_mask is not None:
            valid = self.kept_mask[sources] & self.kept_mask[targets]
            if positions:
                pair_positions = np.flatnonzero(valid) + np.int64(start)
            raw_scores = raw_scores[valid]
            sources = sources[valid]
            targets = targets[valid]
            if alignment_lengths is not None:
                alignment_lengths = alignment_lengths[valid]
        if self.is_blast:
            scores = raw_scores.astype(np.float32) if self.cast_float32 else raw_scores.copy()
        else:
            source_lengths = target_lengths = None
            if self.by_sequence:
                lengths = self.sequence_lengths
                source_lengths = lengths[sources]
                target_lengths = lengths[targets]
                if self.cast_float32:
                    source_lengths = source_lengths.astype(np.float32)
                    target_lengths = target_lengths.astype(np.float32)
            scores = self.normalize(
                raw_scores, alignment_lengths, source_lengths, target_lengths, self.norm_mode
            )
        if positions and pair_positions is None:
            pair_positions = np.arange(start, start + len(scores), dtype=np.int64)
        if nodes:
            if self.kept_mask is None:  # still views of the buffers
                sources, targets = sources.copy(), targets.copy()
            return Slice(scores, sources, targets, pair_positions)
        return Slice(scores, positions=pair_positions)


# --- Running a pass over the slices ---------------------------------------------------


def run_pass(function, ranges, *, workers=None, cancelled=None, progress=None):
    """function(start, stop) for each range on worker threads; the results in order.

    At most 2 * workers slices are in flight, so a pass holds a few chunks however
    many there are. ``cancelled`` is asked before each slice and raises
    StatisticsCancelled; ``progress(done_pairs)`` is called after each result, from
    the calling thread. The reads share the HDF5 library's lock, so one slice reads
    while the others score and reduce theirs.
    """
    workers = max(1, int(workers or worker_count()))

    def task(span):
        if cancelled is not None and cancelled():
            raise StatisticsCancelled()
        return function(*span)

    ranges = list(ranges)
    done = 0
    if workers == 1 or len(ranges) <= 1:
        for span in ranges:
            result = task(span)
            done += span[1] - span[0]
            if progress is not None:
                progress(done)
            yield result
        return
    pool = ThreadPoolExecutor(workers)
    try:
        pending = deque()
        spans = iter(ranges)
        for span in spans:
            pending.append((span, pool.submit(task, span)))
            if len(pending) >= 2 * workers:
                break
        while pending:
            span, future = pending.popleft()
            result = future.result()
            done += span[1] - span[0]
            nxt = next(spans, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(task, nxt)))
            if progress is not None:
                progress(done)
            yield result
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# --- The Config window's statistics ---------------------------------------------------


# The most rows the report's threshold table has. A table is made for people to choose a cutoff
# from; the steps it has always used (0.1, or 1 then 5 for E-value scores) give some 10 to 150
# rows for real networks, and they are kept up to this many rows (scores up to 20 above the
# lowest, or -log10(E) up to about 595). Scores that range further, from a network with lengths
# of zero or with huge scores, would have hundreds of millions of rows, so the table takes the
# smallest step of 1, 2 or 5 times a power of ten that fits.
MAX_REPORT_ROWS = 200


def _default_row_count(is_blast, min_score, max_score):
    """How many rows the report's usual steps would make, counted without making them."""
    if is_blast:
        count = max(0, min(100, int(math.ceil(max_score))) + 1 - int(math.floor(min_score)))
        if max_score > 100:
            count += max(0, -((105 - (int(math.ceil(max_score)) + 5)) // 5))
        return count
    start_val = math.floor(min_score * 10) / 10.0
    return int(math.ceil((float(max_score + 0.1) - start_val) / 0.1))


def _nice_steps(smallest):
    """1, 2, 5, 10, 20, 50, ... times a power of ten, as (digit, exponent), above ``smallest``."""
    exponent = math.floor(math.log10(smallest))
    while True:
        for digit in (1, 2, 5):
            if digit * 10.0 ** exponent > smallest * (1 + 1e-9):
                yield digit, exponent
        exponent += 1


def _coarse_thresholds(is_blast, min_score, max_score, limit):
    """Evenly spaced thresholds from just below the minimum to just above the maximum, in at most ``limit`` rows.

    The step is 1, 2 or 5 times a power of ten; each threshold is the float64 nearest to its
    decimal value (E-value scores take whole numbers, as the usual steps do).
    """
    low, high = float(min_score), float(max_score)
    for digit, exponent in _nice_steps(1.0 if is_blast else 0.1):
        if is_blast and exponent < 0:
            continue
        step = digit * 10.0 ** exponent
        first = math.floor(low / step)
        last = math.ceil((high + step) / step) - 1  # the thresholds below high + step, as np.arange makes
        if last - first + 1 > limit:
            continue
        indices = np.arange(first, last + 1, dtype=np.int64) * digit
        if is_blast:
            return indices * np.int64(10 ** exponent)
        return indices / 10.0 ** -exponent if exponent < 0 else indices * 10.0 ** exponent
    raise ValueError("No threshold table fits these scores.")


def report_thresholds(is_blast, min_score, max_score, limit=MAX_REPORT_ROWS):
    """The thresholds the report lists. min_score and max_score are numpy float32 scalars.

    The expressions are the ones the report always used, which stay on numpy scalars because
    ``min_score * 10`` and ``max_score + 0.1`` follow numpy's float32 rules; they give the table
    whenever it has at most ``limit`` rows. A wider range gets coarser steps instead (see
    MAX_REPORT_ROWS).
    """
    if not (math.isfinite(float(min_score)) and math.isfinite(float(max_score))):
        raise ValueError("The scores are not all finite, so no threshold table can be made.")
    if _default_row_count(is_blast, min_score, max_score) <= limit:
        if is_blast:
            start_val = int(math.floor(min_score))
            limit_step_1 = min(100, int(math.ceil(max_score)))
            thresh_low = np.arange(start_val, limit_step_1 + 1, 1)
            if max_score > 100:
                thresh_high = np.arange(105, int(math.ceil(max_score)) + 5, 5)
                return np.concatenate([thresh_low, thresh_high])
            return thresh_low
        start_val = math.floor(min_score * 10) / 10.0
        thresholds = np.arange(start_val, max_score + 0.1, 0.1)
        if len(thresholds) <= limit:
            return thresholds
    return _coarse_thresholds(is_blast, min_score, max_score, limit)


class ScoreStatistics:
    """What the report needs of the scores: stored edges, min, max, mean and counts per threshold."""

    def __init__(self, stored_edges, minimum, maximum, mean, thresholds, counts):
        self.stored_edges = stored_edges
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.thresholds = thresholds
        self.counts = counts


def compute_score_statistics(source, *, workers=None, chunk_edges=None, progress=None, cancelled=None):
    """Stored edges, minimum, maximum, mean and the count at each threshold, in two passes.

    progress(fraction) is called from the calling thread as the passes advance.
    Raises StatisticsCancelled when ``cancelled()`` says so.
    """
    ranges = chunk_ranges(source.edge_count, chunk_edges)
    total_work = max(1, 2 * source.edge_count)

    def first_pass(start, stop):
        scores = source.read(start, stop).scores
        if len(scores) == 0:
            return 0, None, None, 0.0
        return (len(scores), np.max(scores), np.min(scores), float(scores.sum(dtype=np.float64)))

    stored_edges = 0
    max_score = min_score = None
    running_sum = 0.0
    report = (lambda done: progress(done / total_work)) if progress else None
    for count, part_max, part_min, part_sum in run_pass(
        first_pass, ranges, workers=workers, cancelled=cancelled, progress=report
    ):
        if count == 0:
            continue
        stored_edges += count
        max_score = part_max if max_score is None else max(max_score, part_max)
        min_score = part_min if min_score is None else min(min_score, part_min)
        running_sum += part_sum
    if stored_edges == 0:
        empty = np.zeros(0, dtype=np.int64)
        return ScoreStatistics(0, None, None, None, empty, empty)

    thresholds = report_thresholds(source.is_blast, min_score, max_score)
    cutoffs = cutoffs_for(thresholds, np.float32)

    def second_pass(start, stop):
        scores = source.read(start, stop).scores
        scores.sort()
        return count_at_least(scores, cutoffs)

    counts = np.zeros(len(thresholds), dtype=np.int64)
    report = (lambda done: progress((source.edge_count + done) / total_work)) if progress else None
    for part_counts in run_pass(
        second_pass, ranges, workers=workers, cancelled=cancelled, progress=report
    ):
        counts += part_counts
    return ScoreStatistics(
        stored_edges, min_score, max_score, running_sum / stored_edges, thresholds, counts
    )


def statistics_report(stats, *, model_name, fasta_name, node_count, is_blast, score_mode, norm_mode):
    """The dict ConfigGUI.show_statistics takes, in no language, from ScoreStatistics."""
    node_count = int(node_count)
    theoretical_max_edges = (node_count * (node_count - 1)) / 2.0
    rows = []
    for thresh, count in zip(stats.thresholds, stats.counts):
        pct = (count / theoretical_max_edges) * 100.0 if theoretical_max_edges > 0 else 0
        value = f"{int(thresh)}" if is_blast else f"{thresh:.1f}"
        rows.append((value, f"{count}", f"{pct:.2f}%"))
    return {
        "model": model_name,
        "fasta": fasta_name,
        "nodes": node_count,
        "blast": is_blast,
        "score_mode": score_mode,
        "normalization": norm_mode,
        "stored_edges": int(stats.stored_edges),
        "possible_edges": int(theoretical_max_edges),
        "maximum": float(stats.maximum),
        "minimum": float(stats.minimum),
        "mean": float(stats.mean),
        "rows": rows,
    }


# --- Exact order statistics ------------------------------------------------------------


def float_keys(values):
    """Integer keys that sort like the floats: float32 give uint32, float64 give uint64."""
    values = np.ascontiguousarray(values)
    if values.dtype == np.float32:
        unsigned, sign_bit = values.view(np.uint32), np.uint32(1 << 31)
    elif values.dtype == np.float64:
        unsigned, sign_bit = values.view(np.uint64), np.uint64(1 << 63)
    else:
        raise TypeError(f"scores must be float32 or float64, not {values.dtype}")
    return np.where(unsigned & sign_bit, ~unsigned, unsigned | sign_bit)


def top_digit_keys(scores):
    """The top 16 bits of each score's key (see float_keys), as uint16, without building the keys.

    The top half of a float's bits holds its sign, exponent and first mantissa bits, so one
    cheap view and mask give what a full key shifted right would.
    """
    scores = np.ascontiguousarray(scores)
    step = {np.dtype(np.float32): 2, np.dtype(np.float64): 4}[scores.dtype]
    high = scores.view(np.uint16)[(step - 1 if sys.byteorder == "little" else 0)::step]
    keys = high | np.uint16(0x8000)
    negative = high >= np.uint16(0x8000)
    keys[negative] = ~high[negative]
    return keys


def float_from_key(key, dtype):
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        sign_bit, unsigned_type = 1 << 31, np.uint32
    else:
        sign_bit, unsigned_type = 1 << 63, np.uint64
    key = int(key)
    unsigned = key ^ sign_bit if key & sign_bit else ~key & ((sign_bit << 1) - 1)
    return np.array([unsigned], dtype=unsigned_type).view(dtype)[0]


class RankSelector:
    """The scores of given ranks, and how many scores lie below given values, without keeping the scores.

    ranks are positions in the ascending order (0 is the smallest); probes are values whose
    number of smaller scores is wanted (the edges of a histogram). Each 16-bit digit of the
    scores' keys costs one pass over the data: a float32 needs two passes (levels 0 and 1) and a
    float64 four. In every pass each slice gives ``count(scores, level)``, the counts of the
    next digit among the scores whose earlier digits are those the ranks were narrowed to; the
    slices' counts are added into ``empty_histogram(level)`` with ``merge`` and the total goes
    to ``advance``. Afterwards ``values[rank]`` is the score at that rank, ``below[rank]`` the
    number of scores that are smaller than it, and ``probe_below[i]`` the number of scores smaller
    than ``probes[i]``.
    """

    DIGIT = 16
    SIZE = 1 << DIGIT

    def __init__(self, ranks, dtype, probes=()):
        self.dtype = np.dtype(dtype)
        self.bits = 32 if self.dtype == np.float32 else 64
        self.levels = self.bits // self.DIGIT
        self.ranks = sorted(set(int(rank) for rank in ranks))
        self.probes = np.asarray(list(probes), dtype=self.dtype)
        probe_keys = [int(key) for key in float_keys(self.probes)] if len(self.probes) else []
        # Per entry (the ranks, then the probes): the prefix of its key found so far, the rank
        # within the scores sharing that prefix (ranks only; a probe's key is known), and the
        # number of scores below the prefix.
        self._fixed = [None] * len(self.ranks) + probe_keys
        count = len(self._fixed)
        self._prefix = [0] * count
        self._rank_in = list(self.ranks) + [0] * len(probe_keys)
        self._below = [0] * count
        self._level = 0
        self._live = np.zeros(1, dtype=np.uint64)  # the distinct prefixes still being narrowed
        self._live_top = None  # which top-16-bit values some live prefix starts with
        self.values = {}
        self.below = {}
        self.probe_below = []

    @classmethod
    def first_digit_counts(cls, scores):
        """The level-0 counts of one slice: how many scores have each top 16 key bits.

        They do not depend on the ranks, so they can be taken in the pass that also
        counts the scores, before the ranks (which depend on that count) are known.
        """
        return np.bincount(top_digit_keys(scores), minlength=cls.SIZE)

    def empty_histogram(self, level):
        """The array a level's counts are added into (level 0: the digits; later: live prefixes x digits)."""
        return np.zeros(self.SIZE if level == 0 else len(self._live) * self.SIZE, dtype=np.int64)

    def merge(self, total, part, level):
        """Add one slice's count() into ``total``."""
        if level == 0:
            total += part
        else:
            indices, counts = part
            total[indices] += counts

    def count(self, scores, level):
        """One slice's counts: level 0 the digits' histogram, later (flat indices, counts) of the hits."""
        if level == 0:
            return self.first_digit_counts(scores)
        # Almost no score lies in a live prefix, so find the candidates by their top 16 bits,
        # which are cheap, and build full keys for those alone.
        candidates = scores[self._live_top[top_digit_keys(scores)]]
        keys = float_keys(candidates).astype(np.uint64, copy=False)
        high = keys >> np.uint64(self.bits - self.DIGIT * level)
        position = np.searchsorted(self._live, high)
        position[position == len(self._live)] = 0
        hit = self._live[position] == high
        digit = ((keys[hit] >> np.uint64(self.bits - self.DIGIT * (level + 1))) & np.uint64(self.SIZE - 1)).astype(np.int64)
        flat = position[hit].astype(np.int64) * self.SIZE + digit
        indices, counts = np.unique(flat, return_counts=True)
        return indices, counts

    def advance(self, histogram):
        """Narrow every rank by the next digit, given the added-up counts of this level."""
        level = self._level
        if level == 0:
            row_of = lambda prefix: 0
            rows = histogram.reshape(1, self.SIZE)
        else:
            rows = histogram.reshape(len(self._live), self.SIZE)
            position = {int(prefix): row for row, prefix in enumerate(self._live)}
            row_of = lambda prefix: position[prefix]
        cumulative = {}
        for index in range(len(self._fixed)):
            row = row_of(self._prefix[index])
            if row not in cumulative:
                cumulative[row] = np.cumsum(rows[row])
            running = cumulative[row]
            if self._fixed[index] is None:
                digit = int(np.searchsorted(running, self._rank_in[index], side="right"))
            else:
                digit = (self._fixed[index] >> (self.bits - self.DIGIT * (level + 1))) & (self.SIZE - 1)
            before = int(running[digit - 1]) if digit > 0 else 0
            self._below[index] += before
            self._rank_in[index] -= before
            self._prefix[index] = (self._prefix[index] << self.DIGIT) | digit
        self._level += 1
        if self._level == self.levels:
            for index, rank in enumerate(self.ranks):
                self.values[rank] = float_from_key(self._prefix[index], self.dtype)
                self.below[rank] = self._below[index]
            self.probe_below = self._below[len(self.ranks):]
            return
        self._live = np.array(sorted(set(self._prefix)), dtype=np.uint64)
        self._live_top = np.zeros(self.SIZE, dtype=bool)
        self._live_top[(self._live >> np.uint64(self.DIGIT * (self._level - 1))).astype(np.intp)] = True


def linear_quantiles(count, quantiles, value_at_rank):
    """numpy's default (linear) quantiles, from the values at the ranks they need.

    ``value_at_rank(rank)`` gives the score at an ascending rank. The expressions are
    numpy's own (np.quantile), so the results are the same numbers.
    """
    quantiles = np.asarray(quantiles, dtype=np.float64)
    virtual = (count - 1) * quantiles  # numpy's index for the linear method
    previous, following = quantile_ranks(count, quantiles)
    gamma = np.asanyarray(virtual - previous, dtype=virtual.dtype)
    low = np.array([value_at_rank(int(rank)) for rank in previous], dtype=np.float64)
    high = np.array([value_at_rank(int(rank)) for rank in following], dtype=np.float64)
    difference = high - low
    result = np.add(low, difference * gamma)
    np.subtract(high, difference * (1 - gamma), out=result, where=gamma >= 0.5,
                casting="unsafe", dtype=np.float64)
    return result


def quantile_ranks(count, quantiles):
    """The ascending ranks (before, after) numpy's linear quantile interpolates between."""
    quantiles = np.asarray(quantiles, dtype=np.float64)
    virtual = (count - 1) * quantiles  # numpy's index for the linear method
    previous = np.asanyarray(np.floor(virtual))
    following = previous + 1
    above = virtual >= count - 1
    previous[above] = -1
    following[above] = -1
    below = virtual <= 0
    previous[below] = 0
    following[below] = 0
    previous = previous.astype(np.int64)
    following = following.astype(np.int64)
    previous[previous < 0] += count
    following[following < 0] += count
    return previous, following
