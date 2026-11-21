# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0
"""The shared similarity rule, and the calibration sample of a run's missing pairs.

pair_similarity is the one rule every tool uses for the length-adjusted pooled-embedding cosine
similarity: Align_Similarity_Matrix's Edge Pre-filtering keeps the pairs that reach the
PREFILTER_STRENGTH percentile of it, and Embedding_MSA fills a sparse network's missing pairs with
an isotonic fit of the normalized score on it.

Embedding_MSA draws its calibration sample from the run's own missing pairs (missing_pairs,
select_among): an equal number from each of BINS similarity bands, the lowest band included, chosen
with SEED, then aligned with the network's own scoring. Builds from 89c914b until this change had Align
store a sample of the skipped pairs in the network file instead (select_calibration drew it,
write_record stored it with the prefilter's settings); Align no longer does, but read_record,
locate_sample and pair_keys still read such a file (Network_Injection keeps the stored sample out
of its cut), and write_record and select_calibration stay so that tests can build one.

This module imports no torch, numba or Qt.
"""

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits  # pinned in src/requirements.txt

# The sample: FRACTION of the skipped pairs, at least MIN_PAIRS and at most
# MAX_PAIRS (all of them when fewer are skipped), spread over BINS equal-count
# similarity bands. Internal constants, not Tools settings.
FRACTION = 0.01
MIN_PAIRS = 2000
MAX_PAIRS = 50000
BINS = 20
SEED = 42
# The bounds of the sample Embedding_MSA draws from a run's missing pairs (select_among).
# MIN_PAIRS and MAX_PAIRS stay the bounds of the sample a network stores until Align
# stops storing one.
DRAW_MIN_PAIRS = 5000
DRAW_MAX_PAIRS = 100000
# Calibration is used when at least this share of a network's (or a run's) pairs is missing: Embedding_MSA
# judges its own sequences with it, the Tools window judges the whole network with it.
MIN_MISSING_FRACTION = 0.10

# The similarity rule's two constants (see pair_similarity): shared code, not settings of any tool.
POOLING = "max"            # ("mean", "max") - how a sequence's residue embeddings become one vector
LENGTH_RATIO_POWER = 2.0   # the exponent of the shorter/longer length ratio the cosine is multiplied by

# The network file's record (root attributes and datasets).
ATTR_STRENGTH = "prefilter_strength"
ATTR_THRESHOLD = "prefilter_threshold"
ATTR_POOLING = "prefilter_pooling"
ATTR_LENGTH_POWER = "prefilter_length_ratio_power"
ATTR_KEPT = "prefilter_kept_pairs"
ATTR_SKIPPED = "prefilter_skipped_pairs"
ATTR_BINS = "calibration_bins"
ATTR_SEED = "calibration_seed"
DATA_I = "calibration_i"
DATA_J = "calibration_j"
DATA_WEIGHT = "calibration_weight"

# Edges matched against the sample per step, to bound the temporary arrays.
_MATCH_CHUNK = 8_000_000
# Elements of the length-ratio block length_adjusted_similarity makes at a time (a few int32 and float64
# temporaries of this size, about 100 MB), and values assigned to bands per step in _draw_by_bands.
_SIMILARITY_BLOCK = 4_000_000
_BAND_CHUNK = 8_000_000


@dataclass
class CalibrationSample:
    """Pairs (rows[k] < cols[k]) and how many skipped pairs each stands for."""

    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray

    def __len__(self):
        return int(self.rows.size)


@dataclass
class PrefilterRecord:
    """What a prefiltered network records about its cut and its calibration sample."""

    strength: float
    threshold: float
    pooling: str
    length_power: float
    kept: int
    skipped: int
    sample: CalibrationSample
    bins: int = BINS
    seed: int = SEED


def pooled_vectors(embeddings, names, pooling, progress=None):
    """One vector per sequence: the max or mean of its residue embeddings, as float32.

    embeddings is the HDF5 group (or a mapping) of per-residue arrays, names
    the dataset names in order; progress, when given, wraps the names (tqdm).
    """
    if pooling not in ("max", "mean"):
        raise ValueError(f"Unknown pooling method '{pooling}': use 'max' or 'mean'.")
    pool = np.max if pooling == "max" else np.mean
    iterable = progress(names) if progress is not None else names
    return np.array([pool(embeddings[name][:], axis=0) for name in iterable], dtype=np.float32)


def length_adjusted_similarity(pooled, seq_lens, power):
    """All-vs-all cosine similarity of pooled vectors times (shorter / longer length) ** power.

    pooled is float32 (n, d); the cosine stays float32, clipped to [-1, 1],
    and the length ratio is float64, so the (n, n) result is float64. Zero
    vectors count as length 1e-8.

    The matrix product runs with BLAS on one thread, whatever the process has: a threaded product
    adds the float32 terms in another order, differs in the last bits, and so can move a pair across
    a band edge of the calibration draw or the prefilter's cut. Every tool therefore gets the bits a
    process with one BLAS thread gets (the way Align_Similarity_Matrix has always run).

    Memory: the one float32 product (4 bytes per pair of sequences, clipped in place) and the float64
    result (8 bytes) are the only n x n arrays; the length ratio and its power are made for a block of
    rows at a time and multiplied into the result. Every element still passes through the same operations
    on the same values (integer minimum and maximum, one division, one power, one multiplication, each
    correctly rounded and none combining elements), so the order in which blocks are made cannot change a bit.
    """
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    unit = pooled / norms
    with threadpool_limits(limits=1, user_api="blas"):
        similarity = np.dot(unit, unit.T)
    del unit
    np.clip(similarity, -1.0, 1.0, out=similarity)
    lengths = np.asarray(seq_lens, dtype=np.int32)
    n = similarity.shape[0]
    result = np.empty((n, n), dtype=np.float64)
    block = max(1, _SIMILARITY_BLOCK // max(n, 1))
    for start in range(0, n, block):
        stop = min(start + block, n)
        mine = lengths[start:stop, None]
        shorter = np.minimum(mine, lengths[None, :])
        longer = np.maximum(np.maximum(mine, lengths[None, :]), 1)
        ratio = shorter / longer
        if power != 1.0:
            ratio = ratio ** power
        np.multiply(similarity[start:stop], ratio, out=result[start:stop])
    return result


def pair_similarity(embeddings, headers, lengths, progress=None):
    """The similarity rule every tool uses: the (n, n) float64 matrix of the sequences' length-adjusted pooled cosine.

    This is the one place that owns the rule, so Align_Similarity_Matrix's prefilter, Embedding_MSA (and,
    later, Network_Injection) cannot drift apart: the pooling is POOLING, the power LENGTH_RATIO_POWER,
    neither an argument. headers are the sequences' headers in matrix order, as they are in the file or
    already mapped to dataset names ('/' and '\\' become '_', which mapping twice leaves alone); lengths
    the sequences' residue counts, in the same order; embeddings the HDF5 group (or a mapping) of
    per-residue arrays; progress, when given, wraps the dataset names (tqdm).
    """
    names = [header.replace("/", "_").replace("\\", "_") for header in headers]
    pooled = pooled_vectors(embeddings, names, POOLING, progress=progress)
    return length_adjusted_similarity(pooled, lengths, LENGTH_RATIO_POWER)


def sample_size(skipped, minimum=None, maximum=None):
    """How many skipped pairs the calibration sample aligns: FRACTION of them between the bounds.

    The bounds default to MIN_PAIRS and MAX_PAIRS, read when called.
    """
    minimum = MIN_PAIRS if minimum is None else minimum
    maximum = MAX_PAIRS if maximum is None else maximum
    skipped = int(skipped)
    size = int(round(FRACTION * skipped))
    return min(max(size, int(minimum)), int(maximum), skipped)


def condensed_pairs(positions, n):
    """(row, col) of positions in the row-major upper triangle (col > row) of an n x n matrix."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    n = int(n)

    def row_start(row):
        return row * n - row * (row + 1) // 2

    total = n * (n - 1)
    root = np.sqrt(-8.0 * positions + 4.0 * total - 7.0)
    rows = (n - 2 - np.floor(root / 2.0 - 0.5)).astype(np.int64)
    rows = np.clip(rows, 0, n - 2)
    # Floating point can land one row off near a row's end; step back into place.
    rows -= positions < row_start(rows)
    rows += positions >= row_start(rows + 1)
    cols = positions - row_start(rows) + rows + 1
    return rows, cols


def select_calibration(values, threshold, n, size=None, bins=BINS, seed=SEED):
    """The calibration sample of the pairs whose similarity is below threshold.

    values holds every pair's similarity in row-major upper-triangle order
    (similarity[np.triu(ones, 1)]). The skipped pairs are split into bins
    equal-count bands by similarity, and the same number is drawn from each
    with a generator seeded by seed, so a resumed run draws the same pairs.
    Each pair's weight is its band's skipped count over the pairs drawn from
    it. size defaults to sample_size(skipped); every skipped pair is taken,
    with weight 1, when there are no more than that.
    """
    values = np.asarray(values)
    skipped_positions = np.flatnonzero(values < threshold)
    skipped = skipped_positions.size
    if size is None:
        size = sample_size(skipped)
    size = min(int(size), skipped)
    if size <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return CalibrationSample(empty, empty.copy(), np.zeros(0, dtype=np.float64))
    if size == skipped:
        rows, cols = condensed_pairs(skipped_positions, n)
        return CalibrationSample(rows, cols, np.ones(skipped, dtype=np.float64))

    picked, weights = _draw_by_bands(values[skipped_positions], size, bins, seed)
    rows, cols = condensed_pairs(skipped_positions[picked], n)
    return CalibrationSample(rows, cols, weights)


def _draw_by_bands(candidate_values, size, bins, seed):
    """Indices into candidate_values, the same number from each of bins equal-count bands, and weights.

    The bands are the quantiles of the values; a generator seeded by seed draws
    them band by band, so the same candidates in the same order give the same
    draw. A weight is its band's candidate count over the pairs drawn from it.
    """
    # np.quantile partitions one copy of the values (8 bytes per candidate, freed on return, exact); the
    # band of each candidate is one byte, found a chunk at a time, so no int64 array of the size of the values exists.
    edges = np.quantile(candidate_values, np.linspace(0.0, 1.0, bins + 1))
    count = len(candidate_values)
    band = np.empty(count, dtype=np.uint8 if bins <= 255 else np.int64)
    for start in range(0, count, _BAND_CHUNK):
        chunk = candidate_values[start:start + _BAND_CHUNK]
        band[start:start + len(chunk)] = np.clip(np.searchsorted(edges, chunk, side="right") - 1, 0, bins - 1)
    rng = np.random.default_rng(seed)
    base, extra = divmod(size, bins)
    picked, weights = [], []
    for b in range(bins):
        members = np.flatnonzero(band == b)
        if members.size == 0:
            continue
        take = min(base + (1 if b < extra else 0), members.size)
        if take == 0:
            continue
        chosen = rng.choice(members.size, size=take, replace=False)
        picked.append(members[chosen])
        weights.append(np.full(take, members.size / take, dtype=np.float64))
    return np.concatenate(picked), np.concatenate(weights)


def pair_presence(edge_i, edge_j, n):
    """An (n, n) boolean array, True at [min, max] of every pair that has an edge (the upper triangle only)."""
    present = np.zeros((n, n), dtype=bool)
    for start in range(0, len(edge_i), _MATCH_CHUNK):
        a = np.asarray(edge_i[start:start + _MATCH_CHUNK], dtype=np.int64)
        b = np.asarray(edge_j[start:start + _MATCH_CHUNK], dtype=np.int64)
        keep = a != b
        a, b = a[keep], b[keep]
        present[np.minimum(a, b), np.maximum(a, b)] = True
    return present


def missing_count(present):
    """How many of the n(n-1)/2 pairs have no edge, given pair_presence."""
    n = present.shape[0]
    return n * (n - 1) // 2 - int(np.count_nonzero(present))


def missing_pairs(similarity, present):
    """(values, locate) of the pairs without an edge, in row-major upper-triangle order.

    similarity is the (n, n) adjusted similarity, present the pair_presence of the run's edges.
    values holds the similarity of each missing pair (float64, 8 bytes per pair) and nothing else:
    locate(positions) gives the (rows, cols) of the missing pairs at those positions (int64), found
    from a bit-packed copy of present (n x n / 8 bytes) and the missing count of each row, so only
    the pairs a draw picks ever get their rows and columns.
    """
    n = present.shape[0]
    counts = (n - 1 - np.arange(n, dtype=np.int64)) - np.count_nonzero(present, axis=1)
    total = int(counts.sum())
    values = np.empty(total, dtype=np.float64)
    filled = 0
    for row in range(n - 1):
        columns = np.flatnonzero(~present[row, row + 1:]) + row + 1
        stop = filled + columns.size
        values[filled:stop] = similarity[row, columns]
        filled = stop
    packed = np.packbits(present, axis=1)
    ends = np.cumsum(counts)

    def locate(positions):
        positions = np.asarray(positions, dtype=np.int64)
        rows = np.searchsorted(ends, positions, side="right")
        offsets = positions - (ends[rows] - counts[rows])
        cols = np.empty(positions.size, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        distinct, first = np.unique(sorted_rows, return_index=True)
        last = np.append(first[1:], sorted_rows.size)
        for row, begin, end in zip(distinct.tolist(), first.tolist(), last.tolist()):
            absent = np.flatnonzero(np.unpackbits(packed[row], count=n)[row + 1:] == 0) + row + 1
            chosen = order[begin:end]
            cols[chosen] = absent[offsets[chosen]]
        return rows, cols

    return values, locate


def select_among(values, locate, size=None, bins=BINS, seed=SEED, minimum=None, maximum=None):
    """The calibration sample of a run's missing pairs (missing_pairs gives its arguments).

    The same rule as select_calibration, applied to the missing pairs in
    row-major order: size defaults to sample_size(count, minimum, maximum)
    with the bounds DRAW_MIN_PAIRS and DRAW_MAX_PAIRS (read when called), and
    every pair is taken, with weight 1, when there are no more than that.
    With the pairs a prefilter skipped, and the stored bounds, it draws what
    select_calibration drew.
    """
    values = np.asarray(values)
    count = values.size
    if size is None:
        size = sample_size(
            count,
            DRAW_MIN_PAIRS if minimum is None else minimum,
            DRAW_MAX_PAIRS if maximum is None else maximum,
        )
    size = min(int(size), count)
    if size <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return CalibrationSample(empty, empty.copy(), np.zeros(0, dtype=np.float64))
    if size == count:
        rows, cols = locate(np.arange(count, dtype=np.int64))
        return CalibrationSample(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64),
                                 np.ones(count, dtype=np.float64))
    picked, weights = _draw_by_bands(values, size, bins, seed)
    rows, cols = locate(picked)
    return CalibrationSample(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), weights)


def write_record(hf, record, idx_dtype):
    """Store record in an open network file: its settings as attributes, its sample as datasets."""
    hf.attrs[ATTR_STRENGTH] = float(record.strength)
    hf.attrs[ATTR_THRESHOLD] = float(record.threshold)
    hf.attrs[ATTR_POOLING] = str(record.pooling)
    hf.attrs[ATTR_LENGTH_POWER] = float(record.length_power)
    hf.attrs[ATTR_KEPT] = int(record.kept)
    hf.attrs[ATTR_SKIPPED] = int(record.skipped)
    hf.attrs[ATTR_BINS] = int(record.bins)
    hf.attrs[ATTR_SEED] = int(record.seed)
    hf.create_dataset(DATA_I, data=np.asarray(record.sample.rows, dtype=idx_dtype))
    hf.create_dataset(DATA_J, data=np.asarray(record.sample.cols, dtype=idx_dtype))
    hf.create_dataset(DATA_WEIGHT, data=np.asarray(record.sample.weights, dtype=np.float64))


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_record(hf):
    """The prefilter record of an open network file, or None when it has none.

    A file whose attributes are present but whose sample datasets are missing
    (or the reverse) has no usable record either.
    """
    needed = (ATTR_STRENGTH, ATTR_THRESHOLD, ATTR_POOLING, ATTR_LENGTH_POWER,
              ATTR_KEPT, ATTR_SKIPPED)
    if any(name not in hf.attrs for name in needed):
        return None
    if any(name not in hf for name in (DATA_I, DATA_J, DATA_WEIGHT)):
        return None
    rows = np.asarray(hf[DATA_I][:], dtype=np.int64)
    cols = np.asarray(hf[DATA_J][:], dtype=np.int64)
    weights = np.asarray(hf[DATA_WEIGHT][:], dtype=np.float64)
    if not rows.size == cols.size == weights.size:
        return None
    return PrefilterRecord(
        strength=float(hf.attrs[ATTR_STRENGTH]),
        threshold=float(hf.attrs[ATTR_THRESHOLD]),
        pooling=_text(hf.attrs[ATTR_POOLING]),
        length_power=float(hf.attrs[ATTR_LENGTH_POWER]),
        kept=int(hf.attrs[ATTR_KEPT]),
        skipped=int(hf.attrs[ATTR_SKIPPED]),
        sample=CalibrationSample(rows, cols, weights),
        bins=int(hf.attrs.get(ATTR_BINS, BINS)),
        seed=int(hf.attrs.get(ATTR_SEED, SEED)),
    )


def pair_keys(rows, cols, n):
    """One int64 key per unordered pair: min * n + max."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    return np.minimum(rows, cols) * int(n) + np.maximum(rows, cols)


def locate_sample(sample, index_map, edge_i, edge_j, n):
    """Where each sample pair sits among a run's edges, in sample order (-1 if absent).

    index_map takes the network's sequence indices to the run's (-1 for a
    sequence the run dropped); edge_i and edge_j are the run's edges, n its
    sequence count. A pair stored twice is found at its first edge.
    """
    index_map = np.asarray(index_map)
    found = np.full(len(sample), -1, dtype=np.int64)
    if len(sample) == 0:
        return found
    rows = index_map[sample.rows]
    cols = index_map[sample.cols]
    present = (rows >= 0) & (cols >= 0) & (rows != cols)
    keys = np.full(len(sample), -1, dtype=np.int64)
    keys[present] = pair_keys(rows[present], cols[present], n)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    for start in range(0, len(edge_i), _MATCH_CHUNK):
        stop = min(start + _MATCH_CHUNK, len(edge_i))
        edge_keys = pair_keys(edge_i[start:stop], edge_j[start:stop], n)
        slot = np.searchsorted(sorted_keys, edge_keys)
        slot = np.minimum(slot, sorted_keys.size - 1)
        hits = np.flatnonzero(sorted_keys[slot] == edge_keys)
        for edge, k in zip(hits + start, order[slot[hits]]):
            if keys[k] >= 0 and found[k] < 0:
                found[k] = edge
    return found
