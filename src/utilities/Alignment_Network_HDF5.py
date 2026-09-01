# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Resumable single-file storage for embedding-alignment networks.

The completed file intentionally retains the established alignment-network
schema.  Resume state exists only inside the ``.partial`` working file and is
removed before publication; no format or producer-origin marker is written.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import hashlib
import json
import math
import os
import struct
import zlib

import h5py
import numpy as np


RESULT_DATASETS = (
    "i",
    "j",
    "l_score",
    "l_len",
    "g_score",
    "g_len",
)
DEFAULT_WRITE_CHUNK_EDGES = 65_536
_RESUME_GROUP = "_resume"
_CHECKPOINT_DATASET = "checkpoints"
_CHECKPOINT_DTYPE = np.dtype(
    [
        ("generation", "<u8"),
        ("chunk_start", "<u8"),
        ("committed_count", "<u8"),
        ("next_i", "<u8"),
        ("next_j", "<u8"),
        ("data_crc", "<u4"),
        ("record_crc", "<u4"),
        ("valid", "u1"),
    ]
)


def _decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalized_precision(value):
    value = _decode_text(value).strip().lower()
    return {
        "float32": "ieee_fp32",
        "fp32": "ieee_fp32",
        "ieee": "ieee_fp32",
    }.get(value, value)


def _canonical_pair_ordinal(i, j, sequence_count):
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    return i * sequence_count - (i * (i + 1)) // 2 + (j - i - 1)


@dataclass(frozen=True)
class AlignmentNetworkIdentity:
    headers: tuple[str, ...]
    sequence_lengths: tuple[int, ...]
    embedding_checksum: str
    model_name: str
    gap_penalties: tuple[float, float]
    saving_mode: str | None = None

    @property
    def sequence_count(self):
        return len(self.headers)


@dataclass(frozen=True)
class SparsityProfile:
    enabled: bool
    sparsity_percent: float
    total_pair_count: int
    keep_count: int
    cutoff: float
    pooling_method: str
    length_ratio_power: float
    selection_fingerprint: str


@dataclass(frozen=True)
class CompatibleNetworkCandidate:
    path: str
    matmul_precision: str
    edge_count: int
    reusable_edge_count: int


def exact_top_k_mask(adjusted_scores, sparsity_percent, *, enabled=True):
    """Return an exact, nested upper-triangle top-K mask and its cutoff.

    Equal scores are resolved by canonical row-major pair order.  Consequently,
    every smaller K is a strict prefix of every larger K for the same matrix.
    """

    scores = np.asarray(adjusted_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Adjusted pair scores must be a square matrix.")
    n = int(scores.shape[0])
    tri_i, tri_j = np.triu_indices(n, k=1)
    values = scores[tri_i, tri_j]
    total = int(values.size)
    if not enabled or float(sparsity_percent) <= 0.0:
        keep_count = total
        normalized_sparsity = 0.0
    else:
        normalized_sparsity = float(sparsity_percent)
        if not 0.0 <= normalized_sparsity <= 100.0:
            raise ValueError("Sparsity percentage must be between 0 and 100.")
        keep_count = min(
            total,
            max(
                0,
                int(math.ceil(total * (100.0 - normalized_sparsity) / 100.0)),
            ),
        )

    selected = np.zeros(total, dtype=bool)
    if keep_count == total:
        selected.fill(True)
        cutoff = float(np.min(values)) if total else float("-inf")
    elif keep_count == 0:
        cutoff = float("inf")
    else:
        cutoff_index = total - keep_count
        cutoff = float(np.partition(values, cutoff_index)[cutoff_index])
        selected = values > cutoff
        remaining = keep_count - int(np.count_nonzero(selected))
        if remaining:
            equal_positions = np.flatnonzero(values == cutoff)
            selected[equal_positions[:remaining]] = True

    mask = np.zeros((n, n), dtype=bool)
    mask[tri_i[selected], tri_j[selected]] = True
    if int(np.count_nonzero(mask)) != keep_count:
        raise RuntimeError("Exact sparsity selection produced an invalid edge count.")
    return mask, keep_count, cutoff


def make_selection_fingerprint(
    *,
    embedding_checksum,
    sparsity_percent,
    keep_count,
    cutoff,
    pooling_method,
    length_ratio_power,
):
    payload = {
        "embedding_checksum": str(embedding_checksum),
        "sparsity_percent": float(sparsity_percent),
        "keep_count": int(keep_count),
        "cutoff_float32_hex": np.float32(cutoff).tobytes().hex(),
        "pooling_method": str(pooling_method),
        "length_ratio_power": float(length_ratio_power),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_sparsity_profile(
    adjusted_scores,
    *,
    embedding_checksum,
    enabled,
    sparsity_percent,
    pooling_method,
    length_ratio_power,
):
    mask, keep_count, cutoff = exact_top_k_mask(
        adjusted_scores,
        sparsity_percent,
        enabled=enabled,
    )
    n = int(np.asarray(adjusted_scores).shape[0])
    total = n * (n - 1) // 2
    normalized = float(sparsity_percent) if enabled else 0.0
    profile = SparsityProfile(
        enabled=bool(enabled and normalized > 0.0),
        sparsity_percent=normalized if enabled else 0.0,
        total_pair_count=total,
        keep_count=int(keep_count),
        cutoff=float(np.float32(cutoff)),
        pooling_method=str(pooling_method),
        length_ratio_power=float(length_ratio_power),
        selection_fingerprint=make_selection_fingerprint(
            embedding_checksum=embedding_checksum,
            sparsity_percent=normalized if enabled else 0.0,
            keep_count=keep_count,
            cutoff=cutoff,
            pooling_method=pooling_method,
            length_ratio_power=length_ratio_power,
        ),
    )
    return mask, profile


def iter_mask_pairs(required_mask, *, start_i=0, start_j=1):
    """Yield selected canonical pairs without materializing pair tuples."""

    required = np.asarray(required_mask, dtype=bool)
    n = int(required.shape[0])
    for i in range(max(0, int(start_i)), n):
        first_j = max(i + 1, int(start_j) if i == int(start_i) else i + 1)
        if first_j >= n:
            continue
        for j in np.flatnonzero(required[i, first_j:]) + first_j:
            yield i, int(j)


def _candidate_reason(
    hf,
    identity,
    target_mask,
    required_precision,
    chunk_edges,
):
    missing = [name for name in RESULT_DATASETS + ("headers", "seq_lens") if name not in hf]
    if missing:
        return None, f"missing dataset '{missing[0]}'"
    if _decode_text(hf.attrs.get("embedding_checksum", "")) != identity.embedding_checksum:
        return None, "embedding checksum differs"
    if _decode_text(hf.attrs.get("model_name", "")) != identity.model_name:
        return None, "model name differs"
    cached_gaps = hf.attrs.get("gap_penalties")
    if cached_gaps is None or not np.allclose(
        np.asarray(cached_gaps, dtype=np.float32).reshape(-1),
        np.asarray(identity.gap_penalties, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    ):
        return None, "gap penalties differ"
    precision = _normalized_precision(hf.attrs.get("matmul_precision", "ieee_fp32"))
    if precision not in {"ieee_fp32", "tf32"}:
        return None, f"unsupported matmul precision '{precision}'"
    if required_precision is not None and precision != required_precision:
        return None, "matmul precision differs"

    headers = tuple(_decode_text(value) for value in hf["headers"][:])
    if headers != identity.headers:
        return None, "ordered headers differ"
    lengths = tuple(int(value) for value in hf["seq_lens"][:])
    if lengths != identity.sequence_lengths:
        return None, "sequence lengths differ"

    edge_count = len(hf["i"])
    if any(len(hf[name]) != edge_count for name in RESULT_DATASETS[1:]):
        return None, "alignment dataset lengths differ"
    n = identity.sequence_count
    index_dtype = np.dtype(np.uint16 if n <= 65_535 else np.uint32)
    expected_dtypes = {
        "i": index_dtype,
        "j": index_dtype,
        "l_score": np.dtype(np.float32),
        "l_len": np.dtype(np.uint16),
        "g_score": np.dtype(np.float32),
        "g_len": np.dtype(np.uint16),
    }
    for name, expected_dtype in expected_dtypes.items():
        if np.dtype(hf[name].dtype) != expected_dtype:
            return None, f"dataset '{name}' has incompatible dtype"
    previous = -1
    reusable = 0
    for start in range(0, edge_count, chunk_edges):
        end = min(edge_count, start + chunk_edges)
        arr_i = np.asarray(hf["i"][start:end], dtype=np.int64)
        arr_j = np.asarray(hf["j"][start:end], dtype=np.int64)
        if np.any(arr_i < 0) or np.any(arr_j >= n) or np.any(arr_i >= arr_j):
            return None, "pair indices are invalid or noncanonical"
        ordinals = _canonical_pair_ordinal(arr_i, arr_j, n)
        if ordinals.size:
            if int(ordinals[0]) <= previous or np.any(ordinals[1:] <= ordinals[:-1]):
                return None, "pairs are duplicated or not strictly ordered"
            previous = int(ordinals[-1])
        for name in ("l_score", "g_score"):
            if not np.isfinite(np.asarray(hf[name][start:end], dtype=np.float32)).all():
                return None, f"dataset '{name}' contains non-finite values"
        reusable += int(np.count_nonzero(target_mask[arr_i, arr_j]))
    return (precision, edge_count, reusable), None


def discover_compatible_alignment_networks(
    network_dir,
    *,
    identity,
    target_mask,
    active_target=None,
    required_precision=None,
    diagnostics=None,
    chunk_edges=DEFAULT_WRITE_CHUNK_EDGES,
):
    """Return compatible top-level networks ranked by reusable target edges."""

    required_precision = (
        None if required_precision is None else _normalized_precision(required_precision)
    )
    active_target = (
        None if active_target is None else os.path.abspath(os.path.normpath(active_target))
    )
    candidates = []
    pattern = os.path.join(glob.escape(os.path.abspath(network_dir)), "*.h5")
    for path in sorted(glob.glob(pattern), key=lambda value: os.path.abspath(value)):
        resolved = os.path.abspath(os.path.normpath(path))
        if resolved == active_target or resolved.endswith(".partial"):
            continue
        try:
            with h5py.File(resolved, "r") as hf:
                values, reason = _candidate_reason(
                    hf,
                    identity,
                    target_mask,
                    required_precision,
                    int(chunk_edges),
                )
            if values is None:
                if diagnostics is not None:
                    diagnostics.append((resolved, reason))
                continue
            precision, edge_count, reusable = values
            candidates.append(
                CompatibleNetworkCandidate(
                    path=resolved,
                    matmul_precision=precision,
                    edge_count=int(edge_count),
                    reusable_edge_count=int(reusable),
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            if diagnostics is not None:
                diagnostics.append((resolved, f"{type(error).__name__}: {error}"))
    return sorted(
        candidates,
        key=lambda candidate: (-candidate.reusable_edge_count, candidate.path),
    )


class CanonicalAlignmentNetworkReader:
    """Streaming canonical lookup against one compatible source network."""

    def __init__(self, path, sequence_count, chunk_edges=DEFAULT_WRITE_CHUNK_EDGES):
        self.path = os.path.abspath(path)
        self.sequence_count = int(sequence_count)
        self.chunk_edges = max(1, int(chunk_edges))
        self.hf = h5py.File(self.path, "r")
        self.edge_count = len(self.hf["i"])
        self._offset = 0
        self._chunk = None
        self._ordinals = np.empty(0, dtype=np.int64)

    def _load_next(self):
        if self._offset >= self.edge_count:
            self._chunk = None
            self._ordinals = np.empty(0, dtype=np.int64)
            return False
        start = self._offset
        end = min(self.edge_count, start + self.chunk_edges)
        values = tuple(np.asarray(self.hf[name][start:end]) for name in RESULT_DATASETS)
        self._offset = end
        self._chunk = values
        self._ordinals = _canonical_pair_ordinal(
            values[0], values[1], self.sequence_count
        )
        return True

    def lookup_many(self, pairs):
        """Return ``position -> six-column result`` for increasing pairs."""

        pairs = list(pairs)
        if not pairs:
            return {}
        target_ordinals = _canonical_pair_ordinal(
            np.fromiter((pair[0] for pair in pairs), dtype=np.int64),
            np.fromiter((pair[1] for pair in pairs), dtype=np.int64),
            self.sequence_count,
        )
        found = {}
        cursor = 0
        while cursor < len(pairs):
            if self._chunk is None and not self._load_next():
                break
            if self._ordinals.size == 0:
                self._chunk = None
                continue
            if int(self._ordinals[-1]) < int(target_ordinals[cursor]):
                self._chunk = None
                continue
            chunk_limit = int(self._ordinals[-1])
            target_end = int(np.searchsorted(target_ordinals, chunk_limit, side="right"))
            if target_end <= cursor:
                break
            positions = np.searchsorted(
                self._ordinals,
                target_ordinals[cursor:target_end],
                side="left",
            )
            for relative, source_position in enumerate(positions):
                target_position = cursor + relative
                if (
                    source_position < self._ordinals.size
                    and int(self._ordinals[source_position])
                    == int(target_ordinals[target_position])
                ):
                    found[target_position] = tuple(
                        column[source_position].item() for column in self._chunk
                    )
            cursor = target_end
            if cursor < len(pairs):
                self._chunk = None
        return found

    def close(self):
        if self.hf is not None:
            self.hf.close()
            self.hf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _checkpoint_crc(generation, chunk_start, committed_count, next_i, next_j, data_crc):
    packed = struct.pack(
        "<QQQQQI",
        int(generation),
        int(chunk_start),
        int(committed_count),
        int(next_i),
        int(next_j),
        int(data_crc),
    )
    return zlib.crc32(packed) & 0xFFFFFFFF


def _arrays_crc(arrays):
    checksum = 0
    for array in arrays:
        checksum = zlib.crc32(np.ascontiguousarray(array).tobytes(), checksum)
    return checksum & 0xFFFFFFFF


def _backup_path(path, label):
    candidate = f"{path}.{label}"
    suffix = 1
    while os.path.exists(candidate):
        candidate = f"{path}.{label}_{suffix}"
        suffix += 1
    os.replace(path, candidate)
    return candidate


class ResumableAlignmentNetworkWriter:
    """Write and atomically publish one canonical alignment network."""

    def __init__(
        self,
        target_path,
        *,
        identity,
        sparsity_profile,
        matmul_precision,
        chunk_edges=DEFAULT_WRITE_CHUNK_EDGES,
        failure_hook=None,
    ):
        self.target_path = os.path.abspath(os.path.normpath(target_path))
        self.partial_path = self.target_path + ".partial"
        self.identity = identity
        self.profile = sparsity_profile
        self.matmul_precision = _normalized_precision(matmul_precision)
        self.chunk_edges = max(1, int(chunk_edges))
        self.failure_hook = failure_hook
        self.hf = None
        self.committed_count = 0
        self.generation = 0
        self.next_i = 0
        self.next_j = 1
        self._last_chunk_start = 0
        self._open_or_create()

    def _fail(self, stage):
        if self.failure_hook is not None:
            self.failure_hook(stage)

    def _identity_matches(self, hf):
        if any(name not in hf for name in RESULT_DATASETS + ("headers", "seq_lens")):
            return False
        if any(len(hf[name]) != self.profile.keep_count for name in RESULT_DATASETS):
            return False
        if _decode_text(hf.attrs.get("embedding_checksum", "")) != self.identity.embedding_checksum:
            return False
        if _decode_text(hf.attrs.get("model_name", "")) != self.identity.model_name:
            return False
        if _normalized_precision(hf.attrs.get("matmul_precision", "ieee_fp32")) != self.matmul_precision:
            return False
        gaps = hf.attrs.get("gap_penalties")
        if gaps is None or not np.allclose(
            np.asarray(gaps, dtype=np.float32).reshape(-1),
            np.asarray(self.identity.gap_penalties, dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        ):
            return False
        if tuple(_decode_text(value) for value in hf["headers"][:]) != self.identity.headers:
            return False
        if tuple(int(value) for value in hf["seq_lens"][:]) != self.identity.sequence_lengths:
            return False
        return (
            int(hf.attrs.get("sparsity_keep_count", -1)) == self.profile.keep_count
            and _decode_text(hf.attrs.get("selection_fingerprint", ""))
            == self.profile.selection_fingerprint
        )

    def _create(self):
        os.makedirs(os.path.dirname(self.target_path), exist_ok=True)
        self.hf = h5py.File(self.partial_path, "w")
        hf = self.hf
        hf.attrs["embedding_checksum"] = self.identity.embedding_checksum
        hf.attrs["model_name"] = self.identity.model_name
        if self.identity.saving_mode is not None:
            hf.attrs["saving_mode"] = self.identity.saving_mode
        hf.attrs["gap_penalties"] = np.asarray(
            self.identity.gap_penalties, dtype=np.float32
        )
        hf.attrs["matmul_precision"] = self.matmul_precision
        hf.attrs["sparsity_percent"] = float(self.profile.sparsity_percent)
        hf.attrs["sparsity_keep_count"] = int(self.profile.keep_count)
        hf.attrs["sparsity_cutoff"] = np.float32(self.profile.cutoff)
        hf.attrs["pooling_method"] = self.profile.pooling_method
        hf.attrs["length_ratio_power"] = float(self.profile.length_ratio_power)
        hf.attrs["selection_fingerprint"] = self.profile.selection_fingerprint
        string_dtype = h5py.string_dtype(encoding="utf-8")
        hf.create_dataset(
            "headers",
            data=np.asarray(self.identity.headers, dtype=object),
            dtype=string_dtype,
        )
        hf.create_dataset(
            "seq_lens", data=np.asarray(self.identity.sequence_lengths, dtype=np.uint16)
        )
        index_dtype = np.uint16 if self.identity.sequence_count <= 65_535 else np.uint32
        dtypes = {
            "i": index_dtype,
            "j": index_dtype,
            "l_score": np.float32,
            "l_len": np.uint16,
            "g_score": np.float32,
            "g_len": np.uint16,
        }
        for name, dtype in dtypes.items():
            kwargs = {}
            if self.profile.keep_count:
                kwargs["chunks"] = (min(self.chunk_edges, self.profile.keep_count),)
            hf.create_dataset(
                name,
                shape=(self.profile.keep_count,),
                dtype=dtype,
                **kwargs,
            )
        resume = hf.create_group(_RESUME_GROUP)
        slots = resume.create_dataset(
            _CHECKPOINT_DATASET, shape=(2,), dtype=_CHECKPOINT_DTYPE
        )
        initial = np.zeros((), dtype=_CHECKPOINT_DTYPE)
        initial["generation"] = 0
        initial["chunk_start"] = 0
        initial["committed_count"] = 0
        initial["next_i"] = 0
        initial["next_j"] = 1
        initial["data_crc"] = 0
        initial["record_crc"] = _checkpoint_crc(0, 0, 0, 0, 1, 0)
        initial["valid"] = 1
        slots[0] = initial
        slots[1] = np.zeros((), dtype=_CHECKPOINT_DTYPE)
        hf.flush()

    def _valid_checkpoint(self, record):
        if int(record["valid"]) != 1:
            return False
        expected = _checkpoint_crc(
            record["generation"],
            record["chunk_start"],
            record["committed_count"],
            record["next_i"],
            record["next_j"],
            record["data_crc"],
        )
        if expected != int(record["record_crc"]):
            return False
        start = int(record["chunk_start"])
        end = int(record["committed_count"])
        if start < 0 or end < start or end > self.profile.keep_count:
            return False
        if end > start:
            arrays = tuple(self.hf[name][start:end] for name in RESULT_DATASETS)
            if _arrays_crc(arrays) != int(record["data_crc"]):
                return False
        elif int(record["data_crc"]) != 0:
            return False
        return True

    def _resume(self):
        if not self._identity_matches(self.hf):
            raise ValueError("Partial network metadata does not match the requested target.")
        if _RESUME_GROUP not in self.hf:
            raise ValueError("Partial network is missing its resume journal.")
        records = self.hf[_RESUME_GROUP][_CHECKPOINT_DATASET][:]
        valid = [record for record in records if self._valid_checkpoint(record)]
        if not valid:
            raise ValueError("Partial network has no valid checkpoint.")
        record = max(valid, key=lambda value: int(value["generation"]))
        self.generation = int(record["generation"])
        self._last_chunk_start = int(record["chunk_start"])
        self.committed_count = int(record["committed_count"])
        self.next_i = int(record["next_i"])
        self.next_j = int(record["next_j"])

    def _open_or_create(self):
        if os.path.exists(self.partial_path):
            try:
                self.hf = h5py.File(self.partial_path, "r+")
                self._resume()
                return
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                if self.hf is not None:
                    try:
                        self.hf.close()
                    except OSError:
                        pass
                    self.hf = None
                _backup_path(self.partial_path, "corrupt")
        self._create()

    def commit(self, records, *, next_pair):
        records = list(records)
        if not records:
            return self.committed_count
        self._fail("before_data_write")
        start = self.committed_count
        end = start + len(records)
        if end > self.profile.keep_count:
            raise ValueError("Result chunk exceeds the target network size.")
        columns = tuple(zip(*records))
        arrays = {
            name: np.asarray(column, dtype=self.hf[name].dtype)
            for name, column in zip(RESULT_DATASETS, columns)
        }
        arr_i = arrays["i"].astype(np.int64, copy=False)
        arr_j = arrays["j"].astype(np.int64, copy=False)
        ordinals = _canonical_pair_ordinal(
            arr_i, arr_j, self.identity.sequence_count
        )
        if np.any(arr_i >= arr_j) or np.any(ordinals[1:] <= ordinals[:-1]):
            raise ValueError("Result chunk is not in strict canonical pair order.")
        if start:
            previous = int(
                _canonical_pair_ordinal(
                    [self.hf["i"][start - 1]],
                    [self.hf["j"][start - 1]],
                    self.identity.sequence_count,
                )[0]
            )
            if int(ordinals[0]) <= previous:
                raise ValueError("Result chunk overlaps the committed pair prefix.")
        for name in RESULT_DATASETS:
            self.hf[name][start:end] = arrays[name]
        self.hf.flush()
        self._fail("after_data_flush")
        data_crc = _arrays_crc(tuple(arrays[name] for name in RESULT_DATASETS))
        generation = self.generation + 1
        next_i, next_j = next_pair
        record = np.zeros((), dtype=_CHECKPOINT_DTYPE)
        record["generation"] = generation
        record["chunk_start"] = start
        record["committed_count"] = end
        record["next_i"] = int(next_i)
        record["next_j"] = int(next_j)
        record["data_crc"] = data_crc
        record["record_crc"] = _checkpoint_crc(
            generation, start, end, next_i, next_j, data_crc
        )
        record["valid"] = 1
        self._fail("before_checkpoint_write")
        slots = self.hf[_RESUME_GROUP][_CHECKPOINT_DATASET]
        slots[generation % 2] = record
        self.hf.flush()
        self._fail("after_checkpoint_flush")
        self.generation = generation
        self._last_chunk_start = start
        self.committed_count = end
        self.next_i = int(next_i)
        self.next_j = int(next_j)
        return end

    def _validate_complete(self):
        if self.committed_count != self.profile.keep_count:
            raise RuntimeError(
                f"Cannot publish {self.committed_count} committed edges as a "
                f"{self.profile.keep_count}-edge target."
            )
        previous = -1
        for start in range(0, self.profile.keep_count, self.chunk_edges):
            end = min(self.profile.keep_count, start + self.chunk_edges)
            arr_i = np.asarray(self.hf["i"][start:end], dtype=np.int64)
            arr_j = np.asarray(self.hf["j"][start:end], dtype=np.int64)
            ordinals = _canonical_pair_ordinal(
                arr_i, arr_j, self.identity.sequence_count
            )
            if ordinals.size:
                if int(ordinals[0]) <= previous or np.any(ordinals[1:] <= ordinals[:-1]):
                    raise RuntimeError("Completed network pair order is invalid.")
                previous = int(ordinals[-1])
            if np.any(arr_i >= arr_j):
                raise RuntimeError("Completed network contains a noncanonical pair.")
            for name in ("l_score", "g_score"):
                if not np.isfinite(self.hf[name][start:end]).all():
                    raise RuntimeError(
                        f"Completed network dataset '{name}' contains non-finite values."
                    )

    def finalize(self):
        self._validate_complete()
        if _RESUME_GROUP in self.hf:
            del self.hf[_RESUME_GROUP]
        self.hf.flush()
        self.hf.close()
        self.hf = None
        os.replace(self.partial_path, self.target_path)
        return self.target_path

    def close(self):
        if self.hf is not None:
            self.hf.close()
            self.hf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def completed_target_matches(
    path,
    *,
    identity,
    sparsity_profile,
    matmul_precision=None,
    target_mask=None,
):
    """Validate the identity and exact selection of a completed target."""

    if not os.path.exists(path) or str(path).endswith(".partial"):
        return False
    try:
        with h5py.File(path, "r") as hf:
            precision = _normalized_precision(
                hf.attrs.get("matmul_precision", "ieee_fp32")
            )
            if matmul_precision is not None and precision != _normalized_precision(matmul_precision):
                return False
            if target_mask is None:
                target_mask = np.triu(
                    np.ones(
                        (identity.sequence_count, identity.sequence_count),
                        dtype=bool,
                    ),
                    k=1,
                )
            values, reason = _candidate_reason(
                hf,
                identity,
                target_mask,
                precision,
                DEFAULT_WRITE_CHUNK_EDGES,
            )
            if values is None or reason is not None:
                return False
            precision, edge_count, reusable = values
            return (
                edge_count == sparsity_profile.keep_count
                and reusable == sparsity_profile.keep_count
                and int(hf.attrs.get("sparsity_keep_count", -1))
                == sparsity_profile.keep_count
                and _decode_text(hf.attrs.get("selection_fingerprint", ""))
                == sparsity_profile.selection_fingerprint
                and _RESUME_GROUP not in hf
            )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def backup_incompatible_target(path):
    return _backup_path(path, "incompatible")
