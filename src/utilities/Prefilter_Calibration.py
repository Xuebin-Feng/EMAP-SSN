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
import math

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
# Threshold (bytes of dense 12 * n^2 float32+float64 similarity matrices) above which
# callers switch automatically from dense pair_similarity to StreamedPairSimilarity.
STREAMED_SIMILARITY_BYTES = 256 * 1024 * 1024
_HISTOGRAM_BINS = 65536
_MAX_BRACKET_VALUES = 4_000_000


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
    pooled = np.asarray(pooled, dtype=np.float32)
    if pooled.size == 0 or pooled.shape[0] == 0:
        n = int(pooled.shape[0]) if pooled.ndim >= 1 else 0
        return np.zeros((n, n), dtype=np.float64)
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


class StreamedPairSimilarity:
    """Memory-bounded length-adjusted pooled cosine similarity without any N x N matrix.

    Stores only the (n, d) float32 unit-normalized pooled vectors and (n,) int32
    sequence lengths, streaming row strips of at most ``_SIMILARITY_BLOCK`` cells
    on demand under single-threaded BLAS. Supports exact streaming percentiles
    (bit-identical to ``np.percentile`` over the streamed upper triangle) and
    direct extraction of sorted ``int64`` pair keys ``r * n + c`` (``r < c``).
    """

    def __init__(self, pooled, seq_lens, power):
        pooled = np.asarray(pooled, dtype=np.float32)
        if pooled.ndim == 1:
            pooled = pooled.reshape(pooled.shape[0], 0)
        if pooled.shape[0] == 0 or pooled.shape[1] == 0:
            self.unit = np.ascontiguousarray(pooled, dtype=np.float32)
        else:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-8, norms)
            self.unit = np.ascontiguousarray(pooled / norms, dtype=np.float32)
        self.lengths = np.asarray(seq_lens, dtype=np.int32)
        self.power = float(power)
        self.n = int(self.unit.shape[0])
        self.shape = (self.n, self.n)
        self._last_threshold_count = None

    def _upper_block_rows(self):
        return max(1, _SIMILARITY_BLOCK // max(self.n, 1))

    def _dot_block(self, start, stop, col_start):
        """Compute (stop - start, n - col_start) float32 cosines via float64 accumulation.

        Accumulating in float64 before rounding to float32 eliminates float32 BLAS
        blocking/order drift so ``rows()``, ``upper_strip()``, and ``at()`` agree
        identically on every pair while keeping ``self.unit`` compact in float32.
        """
        width = self.n - col_start
        rows_count = stop - start
        if rows_count <= 0 or width <= 0:
            return np.empty((max(0, rows_count), max(0, width)), dtype=np.float32)
        dim = int(self.unit.shape[1]) if self.unit.ndim == 2 else 0
        if dim == 0:
            return np.zeros((rows_count, width), dtype=np.float32)
        row_64 = np.asarray(self.unit[start:stop], dtype=np.float64)
        # Bound temporary float64 column slice to ~32 MiB.
        col_chunk = max(1, (4 * 1024 * 1024) // dim)
        with threadpool_limits(limits=1, user_api="blas"):
            if width <= col_chunk:
                col_64 = np.asarray(self.unit[col_start:], dtype=np.float64)
                return np.dot(row_64, col_64.T).astype(np.float32, copy=False)
            cos = np.empty((rows_count, width), dtype=np.float32)
            for c0 in range(col_start, self.n, col_chunk):
                c1 = min(c0 + col_chunk, self.n)
                col_64 = np.asarray(self.unit[c0:c1], dtype=np.float64)
                cos[:, c0 - col_start:c1 - col_start] = np.dot(row_64, col_64.T).astype(
                    np.float32, copy=False
                )
            return cos

    def _scale_block(self, cos_block, row_lengths, col_lengths):
        np.clip(cos_block, -1.0, 1.0, out=cos_block)
        shorter = np.minimum(row_lengths[:, None], col_lengths[None, :])
        longer = np.maximum(np.maximum(row_lengths[:, None], col_lengths[None, :]), 1)
        ratio = shorter / longer
        if self.power != 1.0:
            ratio = ratio ** self.power
        return np.multiply(cos_block, ratio, dtype=np.float64)

    def rows(self, start, stop):
        """Return the (stop - start, n) float64 similarity slice for rows [start, stop)."""
        start = int(start)
        stop = int(stop)
        if stop <= start:
            return np.empty((0, self.n), dtype=np.float64)
        cos = self._dot_block(start, stop, 0)
        return self._scale_block(cos, self.lengths[start:stop], self.lengths)

    def upper_strip(self, start, stop):
        """Return (stop - start, n - start) float64 similarities for cols >= start."""
        start = int(start)
        stop = int(stop)
        if stop <= start:
            return np.empty((0, max(0, self.n - start)), dtype=np.float64)
        cos = self._dot_block(start, stop, start)
        return self._scale_block(cos, self.lengths[start:stop], self.lengths[start:])

    def iter_upper_blocks(self):
        """Yield ``(start, stop, block)`` covering all upper-triangle entries ``c > r``."""
        if self.n < 2:
            return
        step = self._upper_block_rows()
        for start in range(0, self.n - 1, step):
            stop = min(start + step, self.n - 1)
            yield start, stop, self.upper_strip(start, stop)

    @staticmethod
    def _upper_mask(start, stop, width):
        rows_rel = np.arange(stop - start, dtype=np.int64)[:, None]
        cols_rel = np.arange(width, dtype=np.int64)[None, :]
        return cols_rel > rows_rel

    def at(self, rows, columns):
        """Return the 1D float64 length-adjusted similarity at each (row, column) pair."""
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        count = rows.size
        if count == 0:
            return np.empty(0, dtype=np.float64)
        out = np.empty(count, dtype=np.float64)
        dim = int(self.unit.shape[1]) if self.unit.ndim == 2 else 0
        if dim == 0:
            out.fill(0.0)
            return out
        # Keep temporary float64 (chunk, d) gathers bounded (~32 MiB each).
        chunk = max(1, (4 * 1024 * 1024) // dim)
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            r = rows[start:stop]
            c = columns[start:stop]
            u_r = np.asarray(self.unit[r], dtype=np.float64)
            u_c = np.asarray(self.unit[c], dtype=np.float64)
            cos = np.einsum("ij,ij->i", u_r, u_c).astype(np.float32, copy=False)
            np.clip(cos, -1.0, 1.0, out=cos)
            lr = self.lengths[r]
            lc = self.lengths[c]
            shorter = np.minimum(lr, lc)
            longer = np.maximum(np.maximum(lr, lc), 1)
            ratio = shorter / longer
            if self.power != 1.0:
                ratio = ratio ** self.power
            out[start:stop] = np.multiply(cos, ratio, dtype=np.float64)
        return out

    def count_upper_at_or_above(self, threshold):
        """Count upper-triangle pairs (r < c) with similarity >= threshold."""
        if self.n < 2:
            return 0
        threshold = float(threshold)
        if math.isnan(threshold):
            return 0
        if (
            self._last_threshold_count is not None
            and self._last_threshold_count[0] == threshold
        ):
            return self._last_threshold_count[1]
        total = 0
        for start, stop, block in self.iter_upper_blocks():
            mask = self._upper_mask(start, stop, block.shape[1])
            mask &= block >= threshold
            total += int(np.count_nonzero(mask))
        self._last_threshold_count = (threshold, total)
        return total

    def upper_keys_at_or_above(self, threshold):
        """Return sorted int64 pair keys ``r * n + c`` (r < c) with similarity >= threshold."""
        if self.n < 2:
            return np.zeros(0, dtype=np.int64)
        threshold = float(threshold)
        if math.isnan(threshold):
            return np.zeros(0, dtype=np.int64)
        n = self.n
        pieces = []
        for start, stop, block in self.iter_upper_blocks():
            mask = self._upper_mask(start, stop, block.shape[1])
            mask &= block >= threshold
            rr, cc = np.nonzero(mask)
            if rr.size:
                pieces.append((rr.astype(np.int64) + start) * n + (cc.astype(np.int64) + start))
        keys = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
        self._last_threshold_count = (threshold, int(keys.size))
        return keys

    def _find_percentile_bracket(self, q):
        """Locate exact order-statistic bracket for ``np.percentile(..., q)`` over upper triangle.

        Returns either:
          - ``("exact", float_value)`` when the percentile is already known exactly, or
          - ``("bracket", (lo, hi, scale, n_bins, b_lo, b_hi, strictly_below, bracket_count, k_lo, k_hi, frac))``
            where ``bracket_count <= _MAX_BRACKET_VALUES`` and a single scan collects
            all elements in bins ``[b_lo, b_hi]`` for exact ``np.partition``.
        """
        q = float(q)
        if not np.isfinite(q) or q < 0.0 or q > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100].")
        total = self.n * (self.n - 1) // 2
        if total == 0:
            return "exact", None

        h = (q / 100.0) * (total - 1)
        k_lo = max(0, min(total - 1, int(math.floor(h))))
        k_hi = max(0, min(total - 1, int(math.ceil(h))))
        frac = h - k_lo

        # Fast 1-pass shortcuts for exact min (q == 0) and exact max (q == 100).
        if k_lo == 0 and k_hi == 0:
            best = None
            for start, stop, block in self.iter_upper_blocks():
                vals = block[self._upper_mask(start, stop, block.shape[1])]
                if np.isnan(vals).any():
                    return "exact", float("nan")
                cur = float(vals.min())
                best = cur if best is None or cur < best else best
            return "exact", best
        if k_lo == total - 1 and k_hi == total - 1:
            best = None
            for start, stop, block in self.iter_upper_blocks():
                vals = block[self._upper_mask(start, stop, block.shape[1])]
                if np.isnan(vals).any():
                    return "exact", float("nan")
                cur = float(vals.max())
                best = cur if best is None or cur > best else best
            return "exact", best

        lo = -1.0
        hi = 1.0
        n_bins = _HISTOGRAM_BINS
        for _ in range(12):
            span = hi - lo
            if span <= 0.0:
                return "exact", float(lo)
            scale = n_bins / span
            hist = np.zeros(n_bins, dtype=np.int64)
            below = 0
            in_min = None
            in_max = None
            for start, stop, block in self.iter_upper_blocks():
                vals = block[self._upper_mask(start, stop, block.shape[1])]
                if np.isnan(vals).any():
                    return "exact", float("nan")
                below += int(np.count_nonzero(vals < lo))
                in_vals = vals[(vals >= lo) & (vals <= hi)]
                if in_vals.size == 0:
                    continue
                cur_min = float(in_vals.min())
                cur_max = float(in_vals.max())
                in_min = cur_min if in_min is None or cur_min < in_min else in_min
                in_max = cur_max if in_max is None or cur_max > in_max else in_max
                idx = np.floor((in_vals - lo) * scale).astype(np.int64)
                np.clip(idx, 0, n_bins - 1, out=idx)
                hist += np.bincount(idx, minlength=n_bins)

            if in_min is not None and in_min == in_max:
                return "exact", float(in_min)

            cum = below + np.cumsum(hist)
            b_lo = int(np.searchsorted(cum, k_lo, side="right"))
            b_hi = int(np.searchsorted(cum, k_hi, side="right"))
            b_lo = min(b_lo, n_bins - 1)
            b_hi = min(b_hi, n_bins - 1)
            strictly_below = below + (int(cum[b_lo - 1] - below) if b_lo > 0 else 0)
            bracket_count = int(hist[b_lo:b_hi + 1].sum())

            if bracket_count <= _MAX_BRACKET_VALUES:
                return "bracket", (
                    lo, hi, scale, n_bins, b_lo, b_hi,
                    strictly_below, bracket_count, k_lo, k_hi, frac,
                )

            # Rare edge case: k_lo is the last item of b_lo and k_hi is the first of b_hi
            # with > _MAX_BRACKET_VALUES items across those two bins -> 1-pass min/max.
            if b_lo < b_hi:
                v_lo = None
                v_hi = None
                for start, stop, block in self.iter_upper_blocks():
                    vals = block[self._upper_mask(start, stop, block.shape[1])]
                    in_vals = vals[(vals >= lo) & (vals <= hi)]
                    if in_vals.size == 0:
                        continue
                    idx = np.floor((in_vals - lo) * scale).astype(np.int64)
                    np.clip(idx, 0, n_bins - 1, out=idx)
                    lo_bin_vals = in_vals[idx == b_lo]
                    if lo_bin_vals.size:
                        m = float(lo_bin_vals.max())
                        v_lo = m if v_lo is None or m > v_lo else v_lo
                    hi_bin_vals = in_vals[idx == b_hi]
                    if hi_bin_vals.size:
                        m = float(hi_bin_vals.min())
                        v_hi = m if v_hi is None or m < v_hi else v_hi
                return "exact", self._lerp(v_lo, v_hi, frac)

            # Massive ties inside a single bin b_lo == b_hi: narrow [lo, hi].
            step = span / n_bins
            new_lo = max(lo, lo + b_lo * step - step * 1e-6)
            new_hi = min(hi, lo + (b_hi + 1) * step + step * 1e-6)
            if new_hi - new_lo <= 1e-15 * max(1.0, abs(new_lo), abs(new_hi)):
                # At ULP resolution: at most a handful of distinct float64 values remain.
                distinct = {}
                below_ul = 0
                for start, stop, block in self.iter_upper_blocks():
                    vals = block[self._upper_mask(start, stop, block.shape[1])]
                    below_ul += int(np.count_nonzero(vals < new_lo))
                    in_vals = vals[(vals >= new_lo) & (vals <= new_hi)]
                    if in_vals.size:
                        uniq, cnts = np.unique(in_vals, return_counts=True)
                        for u_val, u_cnt in zip(uniq.tolist(), cnts.tolist()):
                            distinct[u_val] = distinct.get(u_val, 0) + u_cnt
                acc = below_ul
                v_lo = v_hi = None
                for u_val in sorted(distinct):
                    acc += distinct[u_val]
                    if v_lo is None and acc > k_lo:
                        v_lo = u_val
                    if v_hi is None and acc > k_hi:
                        v_hi = u_val
                        break
                return "exact", self._lerp(v_lo, v_hi, frac)
            lo, hi = new_lo, new_hi

        return "exact", float(0.5 * (lo + hi))

    @staticmethod
    def _lerp(v_lo, v_hi, frac):
        """NumPy 2.x exact linear interpolation formula for 0-ULP ``np.percentile`` parity."""
        diff = v_hi - v_lo
        if frac >= 0.5:
            return float(v_hi - diff * (1.0 - frac))
        return float(v_lo + frac * diff)

    @classmethod
    def _finish_bracket(cls, buf, strictly_below, k_lo, k_hi, frac):
        pos_lo = k_lo - strictly_below
        pos_hi = k_hi - strictly_below
        if pos_lo == pos_hi:
            v_lo = float(np.partition(buf, pos_lo)[pos_lo])
            return v_lo
        part = np.partition(buf, (pos_lo, pos_hi))
        v_lo = float(part[pos_lo])
        v_hi = float(part[pos_hi])
        return cls._lerp(v_lo, v_hi, frac)

    def upper_percentile(self, q):
        """Return the exact q-th percentile (0..100) of upper-triangle similarities, or None if n < 2."""
        kind, payload = self._find_percentile_bracket(q)
        if kind == "exact":
            return payload
        lo, hi, scale, n_bins, b_lo, b_hi, strictly_below, bracket_count, k_lo, k_hi, frac = payload
        buf = np.empty(bracket_count, dtype=np.float64)
        filled = 0
        for start, stop, block in self.iter_upper_blocks():
            vals = block[self._upper_mask(start, stop, block.shape[1])]
            in_vals = vals[(vals >= lo) & (vals <= hi)]
            if in_vals.size == 0:
                continue
            idx = np.floor((in_vals - lo) * scale).astype(np.int64)
            np.clip(idx, 0, n_bins - 1, out=idx)
            picked = in_vals[(idx >= b_lo) & (idx <= b_hi)]
            if picked.size:
                buf[filled:filled + picked.size] = picked
                filled += picked.size
        return self._finish_bracket(buf[:filled], strictly_below, k_lo, k_hi, frac)

    def upper_percentile_and_keys(self, q):
        """Compute exact percentile ``threshold`` and sorted int64 keys ``>= threshold`` in 2 passes."""
        kind, payload = self._find_percentile_bracket(q)
        if kind == "exact":
            if payload is None:
                return None, np.zeros(0, dtype=np.int64)
            return payload, self.upper_keys_at_or_above(payload)

        lo, hi, scale, n_bins, b_lo, b_hi, strictly_below, bracket_count, k_lo, k_hi, frac = payload
        n = self.n
        buf = np.empty(bracket_count, dtype=np.float64)
        filled = 0
        # Per strip we keep sorted candidate keys (in bracket OR strictly above) plus the
        # tiny subset of border positions/values so we never need a global sort after Pass 2.
        strip_records = []
        for start, stop, block in self.iter_upper_blocks():
            upper_mask = self._upper_mask(start, stop, block.shape[1])
            rr_u, cc_u = np.nonzero(upper_mask)
            if rr_u.size == 0:
                continue
            vals = block[rr_u, cc_u]
            above_hi = vals > hi
            in_range = (vals >= lo) & (vals <= hi)
            cand_rel = np.zeros(vals.size, dtype=bool)
            cand_rel |= above_hi
            border_rel = np.zeros(vals.size, dtype=bool)
            if in_range.any():
                in_pos = np.flatnonzero(in_range)
                in_vals = vals[in_pos]
                idx = np.floor((in_vals - lo) * scale).astype(np.int64)
                np.clip(idx, 0, n_bins - 1, out=idx)
                ge_lo = idx >= b_lo
                if ge_lo.any():
                    cand_rel[in_pos[ge_lo]] = True
                in_br = ge_lo & (idx <= b_hi)
                if in_br.any():
                    br_pos_global = in_pos[in_br]
                    border_rel[br_pos_global] = True
                    picked = in_vals[in_br]
                    buf[filled:filled + picked.size] = picked
                    filled += picked.size
            cand_idx = np.flatnonzero(cand_rel)
            if cand_idx.size == 0:
                continue
            cand_keys = (rr_u[cand_idx].astype(np.int64) + start) * n + (
                cc_u[cand_idx].astype(np.int64) + start
            )
            br_local = np.flatnonzero(border_rel[cand_idx])
            br_vals = vals[cand_idx[br_local]] if br_local.size else None
            strip_records.append((cand_keys, br_local, br_vals))

        threshold = self._finish_bracket(buf[:filled], strictly_below, k_lo, k_hi, frac)
        del buf
        total_kept = 0
        for cand_keys, _br_local, br_vals in strip_records:
            n_drop = (
                int(np.count_nonzero(br_vals < threshold))
                if (br_vals is not None and br_vals.size)
                else 0
            )
            total_kept += int(cand_keys.size) - n_drop
        keys = np.empty(total_kept, dtype=np.int64)
        cursor = 0
        for idx_rec in range(len(strip_records)):
            cand_keys, br_local, br_vals = strip_records[idx_rec]
            strip_records[idx_rec] = None
            if br_vals is not None and br_vals.size:
                drop_mask = br_vals < threshold
                if drop_mask.any():
                    keep = np.ones(cand_keys.size, dtype=bool)
                    keep[br_local[drop_mask]] = False
                    cand_keys = cand_keys[keep]
            if cand_keys.size:
                keys[cursor:cursor + cand_keys.size] = cand_keys
                cursor += cand_keys.size
            del cand_keys, br_local, br_vals
        strip_records.clear()
        self._last_threshold_count = (threshold, total_kept)
        return threshold, keys


def streamed_pair_similarity(embeddings, headers, lengths, progress=None):
    """Build a memory-bounded ``StreamedPairSimilarity`` using the shared rule (no N x N matrix)."""
    names = [header.replace("/", "_").replace("\\", "_") for header in headers]
    pooled = pooled_vectors(embeddings, names, POOLING, progress=progress)
    return StreamedPairSimilarity(pooled, lengths, LENGTH_RATIO_POWER)


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
