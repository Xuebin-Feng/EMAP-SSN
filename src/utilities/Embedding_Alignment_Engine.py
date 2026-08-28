# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Memory-bounded CUDA execution for residue-embedding alignments.

The engine is intentionally independent from a particular tool's cache and
output schema.  Callers supply the pair tasks and the CPU alignment callback;
the engine owns only embedding caching, CUDA batching, and pipeline overlap.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import ctypes
import os
import sys
import time
from typing import Callable

import h5py
import numpy as np
import torch
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


GIB = 1024 ** 3
DEFAULT_HOST_CACHE_CAP = 128 * GIB
MIN_HOST_RESERVE = 8 * GIB
MIN_CUDA_RESERVE = 2 * GIB
PADDING_OVERHEAD_LIMIT = 0.15
MATRIX_WORKSPACE_MULTIPLIER = 8
CUDA_TILE_FRACTION = 0.30
CUDA_MATRIX_FRACTION = 0.50
SUPPORTED_TILED_BACKENDS = frozenset({"cuda", "xpu"})


class BenchmarkPhaseTimer:
    """Record the timed phase of one warm-up/timed pipeline invocation."""

    def __init__(self):
        self.started_at = None
        self.stopped_at = None

    def start(self):
        if self.started_at is not None:
            raise RuntimeError("Benchmark timed phase has already started.")
        self.started_at = time.perf_counter()

    def stop(self):
        if self.started_at is None:
            raise RuntimeError("Benchmark timed phase has not started.")
        if self.stopped_at is not None:
            raise RuntimeError("Benchmark timed phase has already stopped.")
        self.stopped_at = time.perf_counter()

    @property
    def elapsed(self):
        if self.started_at is None or self.stopped_at is None:
            raise RuntimeError("Benchmark timed phase did not complete.")
        return max(float(self.stopped_at - self.started_at), 1e-9)


def evenly_spaced_task_subset(tasks, count):
    """Return a deterministic evenly spaced subset without reordering it."""
    tasks = list(tasks)
    count = max(0, min(int(count), len(tasks)))
    if count == 0:
        return []
    if count == len(tasks):
        return tasks
    positions = np.linspace(0, len(tasks) - 1, num=count, dtype=np.int64)
    return [tasks[int(position)] for position in positions]


def matched_benchmark_task_halves(
    headers,
    lengths,
    pending_counts,
    pending_columns_for_row,
    half_pairs,
    *,
    row_limit=32,
):
    """Select disjoint, row/cost-matched warm-up and timed task halves.

    ``pending_counts`` describes the complete pending workload by left row.
    ``pending_columns_for_row`` is called only for the globally representative
    rows selected here, avoiding materialization of the full pair workload.
    """
    headers = list(headers)
    lengths = [int(value) for value in lengths]
    counts = np.asarray(pending_counts, dtype=np.int64)
    if len(headers) != len(lengths) or len(headers) != len(counts):
        raise ValueError("Benchmark headers, lengths, and row counts must match.")
    if np.any(counts < 0):
        raise ValueError("Benchmark pending row counts cannot be negative.")

    total = int(counts.sum())
    if total == 0:
        return [], []
    requested_half = max(1, int(half_pairs))
    if total == 1:
        row = int(np.flatnonzero(counts)[0])
        columns = np.asarray(pending_columns_for_row(row), dtype=np.int64)
        if len(columns) != 1:
            raise ValueError("Pending column provider disagrees with row counts.")
        column = int(columns[0])
        return [], [(row, column, headers[row], headers[column])]

    half = min(requested_half, total // 2)
    selected_total = half * 2
    populated = np.flatnonzero(counts)
    row_slots = min(max(1, int(row_limit)), len(populated), selected_total)
    cumulative = np.cumsum(counts)
    ordinal_targets = (
        (np.arange(row_slots, dtype=np.float64) + 0.5) * total / row_slots
    ).astype(np.int64)
    selected_rows = set(
        int(row) for row in np.searchsorted(cumulative, ordinal_targets, side="right")
    )

    selected_capacity = sum(int(counts[row]) for row in selected_rows)
    paired_capacity = sum(int(counts[row]) // 2 for row in selected_rows)
    remaining = sorted(
        (int(row) for row in populated if int(row) not in selected_rows),
        key=lambda row: (-int(counts[row]), row),
    )
    for row in remaining:
        if selected_capacity >= selected_total and paired_capacity >= half:
            break
        selected_rows.add(row)
        selected_capacity += int(counts[row])
        paired_capacity += int(counts[row]) // 2
    if selected_capacity < selected_total:
        raise ValueError("Pending row counts cannot supply the benchmark sample.")

    ordered_rows = sorted(selected_rows)
    selected_counts = np.asarray([counts[row] for row in ordered_rows], dtype=np.int64)
    paired_counts = selected_counts // 2
    paired_capacity = int(paired_counts.sum())
    if paired_capacity >= half:
        paired_cumulative = np.cumsum(paired_counts)
        quota_targets = (
            (np.arange(half, dtype=np.float64) + 0.5)
            * paired_capacity
            / half
        ).astype(np.int64)
        quota_indices = np.searchsorted(
            paired_cumulative, quota_targets, side="right"
        )
        quotas = 2 * np.bincount(
            quota_indices, minlength=len(ordered_rows)
        )
    else:
        selected_cumulative = np.cumsum(selected_counts)
        quota_targets = (
            (np.arange(selected_total, dtype=np.float64) + 0.5)
            * selected_capacity
            / selected_total
        ).astype(np.int64)
        quota_indices = np.searchsorted(
            selected_cumulative, quota_targets, side="right"
        )
        quotas = np.bincount(quota_indices, minlength=len(ordered_rows))

    warmup = []
    timed = []
    for row_offset, (row, quota) in enumerate(zip(ordered_rows, quotas)):
        quota = int(quota)
        if quota == 0:
            continue
        columns = np.asarray(pending_columns_for_row(row), dtype=np.int64)
        if len(columns) != int(counts[row]):
            raise ValueError("Pending column provider disagrees with row counts.")
        ordered_columns = sorted(
            (int(column) for column in columns),
            key=lambda column: (lengths[row] * lengths[column], column),
        )
        positions = np.linspace(
            0, len(ordered_columns) - 1, num=quota, dtype=np.int64
        )
        chosen = [ordered_columns[int(position)] for position in positions]
        start_with_warmup = row_offset % 2 == 0
        for offset in range(0, len(chosen) - 1, 2):
            left = (row, chosen[offset], headers[row], headers[chosen[offset]])
            right = (
                row,
                chosen[offset + 1],
                headers[row],
                headers[chosen[offset + 1]],
            )
            if start_with_warmup:
                warmup.append(left)
                timed.append(right)
            else:
                timed.append(left)
                warmup.append(right)
        if len(chosen) % 2:
            column = chosen[-1]
            task = (row, column, headers[row], headers[column])
            if len(warmup) <= len(timed):
                warmup.append(task)
            else:
                timed.append(task)

    if len(warmup) != half or len(timed) != half:
        raise RuntimeError("Benchmark sampler failed to produce equal task halves.")
    warmup.sort(key=lambda task: (int(task[0]), int(task[1])))
    timed.sort(key=lambda task: (int(task[0]), int(task[1])))
    return warmup, timed


def system_memory_bytes():
    """Return ``(total, available)`` physical-memory bytes when detectable."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except (AttributeError, OSError):
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except (AttributeError, OSError, ValueError):
        return 0, 0


def resolve_host_cache_bytes(setting):
    """Resolve ``auto``/GiB input to a safe persistent host-cache budget."""
    total, available = system_memory_bytes()
    reserve = max(MIN_HOST_RESERVE, int(total * 0.25)) if total else MIN_HOST_RESERVE
    safe_limit = max(0, available - reserve) if available else 0
    safe_limit = min(DEFAULT_HOST_CACHE_CAP, safe_limit)

    if setting is None or str(setting).strip().lower() in {"", "auto"}:
        return safe_limit
    try:
        requested = float(setting)
    except (TypeError, ValueError) as error:
        raise ValueError("HOST_CACHE_GB must be 'auto' or a non-negative number.") from error
    if not np.isfinite(requested) or requested < 0:
        raise ValueError("HOST_CACHE_GB must be 'auto' or a non-negative number.")
    return min(int(requested * GIB), safe_limit) if safe_limit else int(requested * GIB)


def normalize_precision_setting(value):
    normalized = "auto" if value is None else str(value).strip().lower()
    aliases = {"ieee": "float32", "fp32": "float32", "ieee_fp32": "float32"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"auto", "float32", "tf32"}:
        raise ValueError("ACCELERATOR_PRECISION must be auto, float32, or tf32.")
    return normalized


def normalize_execution_mode(value):
    """Return a validated alignment execution-mode setting."""
    normalized = "auto" if value is None else str(value).strip().lower()
    if normalized not in {"auto", "scalar", "tiled"}:
        raise ValueError(
            "Execution mode must be 'auto', 'scalar', or 'tiled'."
        )
    return normalized


def is_nvidia_cuda(device=None):
    """Return whether PyTorch exposes native NVIDIA CUDA (not ROCm/HIP)."""
    if device is not None and getattr(device, "type", None) != "cuda":
        return False
    return (
        getattr(torch.version, "cuda", None) is not None
        and getattr(torch.version, "hip", None) is None
        and torch.cuda.is_available()
    )


class AcceleratorBackend:
    """Minimal CUDA/XPU runtime adapter for the Aug 22 tiled pipeline."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.device_type = self.device.type
        if self.device_type not in SUPPORTED_TILED_BACKENDS:
            raise ValueError(
                "Tiled execution requires a CUDA/ROCm or XPU accelerator; "
                f"received '{self.device}'."
            )
        self.module = getattr(torch, self.device_type, None)
        if self.module is None:
            raise RuntimeError(
                f"This PyTorch build does not provide torch.{self.device_type}."
            )
        self._memory_module = getattr(self.module, "memory", None)

    def _runtime_function(self, name):
        function = getattr(self.module, name, None)
        if function is None and self._memory_module is not None:
            function = getattr(self._memory_module, name, None)
        return function

    def device_context(self):
        context = getattr(self.module, "device", None)
        return context(self.device) if context is not None else nullcontext()

    def create_stream(self):
        stream_type = getattr(self.module, "Stream", None)
        if stream_type is None:
            raise RuntimeError(
                f"torch.{self.device_type}.Stream is unavailable."
            )
        try:
            return stream_type(device=self.device)
        except TypeError:
            with self.device_context():
                return stream_type()

    def stream_context(self, stream):
        context = getattr(self.module, "stream", None)
        if context is None:
            raise RuntimeError(
                f"torch.{self.device_type}.stream is unavailable."
            )
        return context(stream)

    def create_event(self):
        event_type = getattr(self.module, "Event", None)
        if event_type is None:
            raise RuntimeError(
                f"torch.{self.device_type}.Event is unavailable."
            )
        return event_type()

    def memory_info(self):
        function = self._runtime_function("mem_get_info")
        if function is None:
            raise RuntimeError(
                f"torch.{self.device_type} does not expose allocator memory info."
            )
        with self.device_context():
            try:
                free_bytes, total_bytes = function(self.device)
            except TypeError:
                free_bytes, total_bytes = function()
        return int(free_bytes), int(total_bytes)

    def empty_cache(self):
        function = self._runtime_function("empty_cache")
        if function is not None:
            function()

    def is_out_of_memory(self, error):
        exception_types = []
        for owner in (torch, self.module):
            exception_type = getattr(owner, "OutOfMemoryError", None)
            if isinstance(exception_type, type):
                exception_types.append(exception_type)
        if exception_types and isinstance(error, tuple(set(exception_types))):
            return True
        return isinstance(error, RuntimeError) and any(
            signature in str(error).lower()
            for signature in (
                "out of memory",
                "cannot allocate memory",
                "allocation failed",
            )
        )

    def supports_tiled(self, *, require_memory=True):
        missing = [
            name for name in ("Stream", "Event", "stream")
            if getattr(self.module, name, None) is None
        ]
        if require_memory and self._runtime_function("mem_get_info") is None:
            missing.append("mem_get_info")
        if missing:
            return False, (
                f"torch.{self.device_type} is missing " + ", ".join(missing)
            )
        return True, "stream, event, and allocator APIs are available"


def get_accelerator_backend(device):
    return AcceleratorBackend(device)


def tiled_accelerator_support(device, *, require_memory=True):
    try:
        backend = get_accelerator_backend(device)
    except (RuntimeError, ValueError) as error:
        return False, str(error)
    return backend.supports_tiled(require_memory=require_memory)


@contextmanager
def cuda_matmul_precision(mode):
    """Temporarily select IEEE FP32 or TF32 CUDA matmul semantics."""
    normalized = str(mode).strip().lower()
    use_tf32 = normalized == "tf32"
    new_api = (
        hasattr(torch.backends, "cuda")
        and hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "fp32_precision")
    )
    if new_api:
        previous = torch.backends.cuda.matmul.fp32_precision
        torch.backends.cuda.matmul.fp32_precision = "tf32" if use_tf32 else "ieee"
        try:
            yield
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous
        return

    matmul_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul_backend is None or not hasattr(matmul_backend, "allow_tf32"):
        yield
        return
    previous = bool(matmul_backend.allow_tf32)
    matmul_backend.allow_tf32 = use_tf32
    try:
        yield
    finally:
        matmul_backend.allow_tf32 = previous


def compute_score_matrix_torch(emb_i, emb_j, device, precision="float32"):
    """Compute the canonical finite float32 residue-alignment score matrix."""
    precision_context = (
        nullcontext() if precision is None else cuda_matmul_precision(precision)
    )
    with precision_context, torch.inference_mode():
        t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float32)
        t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float32)
        t_i = torch.nn.functional.normalize(t_i, p=2, dim=-1)
        t_j = torch.nn.functional.normalize(t_j, p=2, dim=-1)
        cosine = torch.mm(t_i, t_j.T).clamp(-1.0, 1.0)
        similarity = torch.exp(-(1.0 - cosine))
        epsilon = 1e-8
        row_mean = similarity.mean(dim=1, keepdim=True)
        row_std = similarity.std(dim=1, keepdim=True, correction=0)
        col_mean = similarity.mean(dim=0, keepdim=True)
        col_std = similarity.std(dim=0, keepdim=True, correction=0)
        score = (
            (similarity - row_mean) / (row_std + epsilon)
            + (similarity - col_mean) / (col_std + epsilon)
        ) / 2.0
        result = score.to(dtype=torch.float32, device="cpu").numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Residue score calculation produced non-finite values.")
    return result


@dataclass(frozen=True)
class CudaMemoryPlan:
    free_bytes: int
    total_bytes: int
    usable_bytes: int
    tile_cache_bytes: int
    matrix_pool_bytes: int
    matrix_bytes: int
    reserve_bytes: int
    lanes: int
    inflight_slots: int


@dataclass(frozen=True)
class CudaWorkloadEstimate:
    variant: str
    lanes: int
    inflight_slots: int
    tile_bytes: int
    transient_bytes: int
    additional_bytes: int
    projected_peak_bytes: int
    safe_peak_bytes: int
    per_microbatch_bytes: int
    largest_microbatch_bytes: int
    feasible: bool
    reason: str


def cuda_memory_plan(
    device,
    lanes=1,
    memory_info=None,
    *,
    tile_fraction=None,
    matrix_fraction=None,
):
    """Divide free accelerator memory across tiles and microbatches."""
    lanes = max(1, int(lanes))
    tile_fraction = (
        CUDA_TILE_FRACTION if tile_fraction is None else float(tile_fraction)
    )
    matrix_fraction = (
        CUDA_MATRIX_FRACTION
        if matrix_fraction is None
        else float(matrix_fraction)
    )
    if tile_fraction <= 0 or matrix_fraction <= 0:
        raise ValueError("CUDA tile and matrix fractions must be positive.")
    if tile_fraction + matrix_fraction > 0.80 + 1e-12:
        raise ValueError(
            "CUDA tile and matrix fractions may use at most 80% of usable VRAM."
        )
    if memory_info is None:
        free_bytes, total_bytes = get_accelerator_backend(device).memory_info()
    else:
        free_bytes, total_bytes = memory_info
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    reserve = max(MIN_CUDA_RESERVE, int(total_bytes * 0.15))
    usable = max(0, free_bytes - reserve)
    inflight_slots = max(2, lanes * 2)
    matrix_pool = max(1, int(usable * matrix_fraction))
    return CudaMemoryPlan(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        usable_bytes=usable,
        tile_cache_bytes=max(1, int(usable * tile_fraction)),
        matrix_pool_bytes=matrix_pool,
        matrix_bytes=max(1, matrix_pool // inflight_slots),
        reserve_bytes=reserve,
        lanes=lanes,
        inflight_slots=inflight_slots,
    )


class EmbeddingTileStore:
    """Describe an embedding database and optionally keep it packed in RAM."""

    def __init__(self, path, headers, host_cache_setting="auto"):
        self.path = os.fspath(path)
        self.headers = list(headers)
        self.host_cache_bytes = resolve_host_cache_bytes(host_cache_setting)
        self.shapes = []
        self.dtypes = []
        self.source_bytes = []
        self.float32_bytes = []
        self.feature_dimension = None
        with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
            group = hf["embeddings"]
            for header in self.headers:
                dataset = group[header]
                shape = tuple(int(value) for value in dataset.shape)
                if len(shape) != 2:
                    raise ValueError(f"Embedding '{header}' is not two-dimensional.")
                if self.feature_dimension is None:
                    self.feature_dimension = shape[1]
                elif shape[1] != self.feature_dimension:
                    raise ValueError("Embedding feature dimensions are inconsistent.")
                dtype = np.dtype(dataset.dtype)
                self.shapes.append(shape)
                self.dtypes.append(dtype)
                self.source_bytes.append(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize)
                self.float32_bytes.append(int(np.prod(shape, dtype=np.int64)) * 4)

        self.total_source_bytes = int(sum(self.source_bytes))
        self.total_float32_bytes = int(sum(self.float32_bytes))
        self._packed = None
        self._offsets = None
        if self.host_cache_bytes and self.total_source_bytes <= self.host_cache_bytes:
            self._load_full_cache()

    @property
    def fully_cached(self):
        return self._packed is not None

    @property
    def cached_bytes(self):
        return self.total_source_bytes if self.fully_cached else 0

    def _load_full_cache(self):
        if not self.headers:
            return
        common_dtype = self.dtypes[0]
        if any(dtype != common_dtype for dtype in self.dtypes):
            return
        total_rows = sum(shape[0] for shape in self.shapes)
        packed = np.empty((total_rows, self.feature_dimension), dtype=common_dtype)
        offsets = np.zeros(len(self.headers) + 1, dtype=np.int64)
        with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
            group = hf["embeddings"]
            cursor = 0
            for index, header in enumerate(self.headers):
                rows = self.shapes[index][0]
                packed[cursor:cursor + rows] = group[header][:]
                cursor += rows
                offsets[index + 1] = cursor
        self._packed = packed
        self._offsets = offsets

    def get(self, index, h5_group=None):
        index = int(index)
        if self._packed is not None:
            start, end = self._offsets[index:index + 2]
            return self._packed[int(start):int(end)]
        if h5_group is None:
            with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
                return hf["embeddings"][self.headers[index]][:]
        return h5_group[self.headers[index]][:]

    def load_indices(self, indices, h5_group=None):
        return {int(index): self.get(index, h5_group) for index in sorted(set(indices))}

    def block_ids(self, max_block_bytes):
        """Greedily assign sequences to contiguous float32-byte blocks."""
        max_block_bytes = max(1, int(max_block_bytes))
        block_ids = np.zeros(len(self.headers), dtype=np.int32)
        block = 0
        used = 0
        for index, size in enumerate(self.float32_bytes):
            if used and used + size > max_block_bytes:
                block += 1
                used = 0
            block_ids[index] = block
            used += size
        return block_ids


def _to_normalized_cuda(array, device):
    contiguous = np.ascontiguousarray(array)
    cpu_tensor = torch.from_numpy(contiguous)
    try:
        cpu_tensor = cpu_tensor.pin_memory()
    except RuntimeError:
        pass
    tensor = cpu_tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
    tensor.div_(norms.clamp_min_(torch.finfo(torch.float32).tiny))
    return tensor


def _microbatch_workspace_bytes(
    row_length,
    target_lengths,
    feature_dimension=0,
):
    """Conservatively estimate padded targets plus score/statistic tensors."""
    if not target_lengths:
        return 0
    count = len(target_lengths)
    max_columns = max(int(length) for length in target_lengths)
    matrix_bytes = (
        count
        * int(row_length)
        * max_columns
        * 4
        * MATRIX_WORKSPACE_MULTIPLIER
    )
    padded_target_bytes = (
        count * max_columns * max(0, int(feature_dimension)) * 4
    )
    return int(matrix_bytes + padded_target_bytes)


def _length_microbatches(
    tasks,
    lengths,
    row_length,
    matrix_budget,
    feature_dimension=0,
):
    """Bucket one row by target length, padding by at most 15 percent."""
    ordered = sorted(tasks, key=lambda task: (lengths[int(task[1])], int(task[1])))
    current = []
    real_columns = 0
    max_columns = 0
    for task in ordered:
        length = int(lengths[int(task[1])])
        next_count = len(current) + 1
        next_real = real_columns + length
        next_max = max(max_columns, length)
        padded = next_count * next_max
        padding_ok = padded <= next_real * (1.0 + PADDING_OVERHEAD_LIMIT)
        workspace = int(
            next_count
            * next_max
            * 4
            * (
                int(row_length) * MATRIX_WORKSPACE_MULTIPLIER
                + max(0, int(feature_dimension))
            )
        )
        memory_ok = workspace <= max(1, int(matrix_budget))
        if current and (not padding_ok or not memory_ok):
            yield current
            current = []
            real_columns = 0
            max_columns = 0
        current.append(task)
        real_columns += length
        max_columns = max(max_columns, length)
    if current:
        yield current


def _batched_score_matrices(row_tensor, target_tensors, target_lengths):
    """Compute padded score matrices while excluding padding from statistics."""
    batch_size = len(target_tensors)
    max_length = max(int(length) for length in target_lengths)
    feature_dimension = int(row_tensor.shape[1])
    targets = torch.zeros(
        (batch_size, max_length, feature_dimension),
        dtype=torch.float32,
        device=row_tensor.device,
    )
    for index, (target, length) in enumerate(zip(target_tensors, target_lengths)):
        targets[index, :int(length)].copy_(target)

    cosine = torch.matmul(row_tensor.unsqueeze(0), targets.transpose(1, 2))
    similarity = torch.exp(-(1.0 - cosine.clamp_(-1.0, 1.0)))
    lengths_tensor = torch.as_tensor(
        target_lengths,
        dtype=torch.int64,
        device=row_tensor.device,
    )
    mask = torch.arange(max_length, device=row_tensor.device).unsqueeze(0) < lengths_tensor.unsqueeze(1)
    mask3 = mask.unsqueeze(1)
    divisor = lengths_tensor.to(torch.float32).view(-1, 1, 1)

    row_mean = (similarity * mask3).sum(dim=2, keepdim=True) / divisor
    row_delta = (similarity - row_mean) * mask3
    row_std = torch.sqrt((row_delta * row_delta).sum(dim=2, keepdim=True) / divisor)
    col_std, col_mean = torch.std_mean(similarity, dim=1, keepdim=True, correction=0)
    epsilon = 1e-8
    final = (
        (similarity - row_mean) / (row_std + epsilon)
        + (similarity - col_mean) / (col_std + epsilon)
    ) / 2.0
    final.masked_fill_(~mask3, 0.0)
    return final


def _host_output_buffer(tensor):
    try:
        return torch.empty(tensor.shape, dtype=torch.float32, device="cpu", pin_memory=True)
    except RuntimeError:
        return torch.empty(tensor.shape, dtype=torch.float32, device="cpu")


def _partition_tiles(tasks, block_ids):
    grouped = OrderedDict()
    for task in tasks:
        key = (int(block_ids[int(task[0])]), int(block_ids[int(task[1])]))
        grouped.setdefault(key, []).append(task)
    return grouped.values()


def _drain_completed_alignment_futures(
    cpu_pending,
    results,
    *,
    progress=None,
    block=False,
):
    """Collect every completed alignment immediately, independent of order."""
    if block and cpu_pending:
        completed, _ = wait(cpu_pending, return_when=FIRST_COMPLETED)
    else:
        completed = {future for future in cpu_pending if future.done()}
    for future in completed:
        cpu_pending.remove(future)
        results.append(future.result())
        if progress is not None:
            progress.update(1)
    return len(completed)


def estimate_cuda_working_set(
    tasks,
    *,
    store: EmbeddingTileStore,
    lengths,
    device,
    lanes,
    variant="tiled",
    memory_info=None,
    memory_plan_override=None,
):
    """Estimate peak CUDA use for one workload/variant without allocating."""
    if memory_plan_override is not None and memory_info is not None:
        raise ValueError(
            "memory_info and memory_plan_override cannot be supplied together."
        )
    plan = memory_plan_override or cuda_memory_plan(
        device, lanes=lanes, memory_info=memory_info
    )
    if int(plan.lanes) != max(1, int(lanes)):
        raise ValueError("CUDA memory plan lane count does not match execution lanes.")
    tasks = list(tasks)
    baseline_bytes = max(0, plan.total_bytes - plan.free_bytes)
    safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
    if not tasks:
        return CudaWorkloadEstimate(
            variant=str(variant),
            lanes=int(lanes),
            inflight_slots=plan.inflight_slots,
            tile_bytes=0,
            transient_bytes=0,
            additional_bytes=0,
            projected_peak_bytes=baseline_bytes,
            safe_peak_bytes=safe_peak,
            per_microbatch_bytes=plan.matrix_bytes,
            largest_microbatch_bytes=0,
            feasible=True,
            reason="empty workload",
        )

    feature_dimension = int(store.feature_dimension or 0)
    variant = str(variant).strip().lower()
    tile_bytes = 0
    workspaces = []
    if variant == "tiled":
        per_block = max(1, plan.tile_cache_bytes // 2)
        block_ids = store.block_ids(per_block)
        for tile_tasks in _partition_tiles(tasks, block_ids):
            tile_indices = {int(task[0]) for task in tile_tasks}
            tile_indices.update(int(task[1]) for task in tile_tasks)
            tile_bytes = max(
                tile_bytes,
                sum(store.float32_bytes[index] for index in tile_indices),
            )
            rows = OrderedDict()
            for task in tile_tasks:
                rows.setdefault(int(task[0]), []).append(task)
            for row, row_tasks in rows.items():
                for microbatch in _length_microbatches(
                    row_tasks,
                    lengths,
                    lengths[row],
                    plan.matrix_bytes,
                    feature_dimension,
                ):
                    target_lengths = [
                        int(lengths[int(task[1])]) for task in microbatch
                    ]
                    workspaces.append(
                        _microbatch_workspace_bytes(
                            lengths[row],
                            target_lengths,
                            feature_dimension,
                        )
                    )
        active_slots = plan.inflight_slots
    else:
        for task in tasks:
            left = int(task[0])
            right = int(task[1])
            workspace = _microbatch_workspace_bytes(
                lengths[left],
                [lengths[right]],
                0,
            )
            embedding_bytes = (
                store.float32_bytes[left] + store.float32_bytes[right]
            )
            workspaces.append(workspace + embedding_bytes)
        active_slots = max(1, int(lanes))

    largest = max(workspaces, default=0)
    transient_bytes = sum(
        sorted(workspaces, reverse=True)[:active_slots]
    )
    additional = tile_bytes + transient_bytes
    projected_peak = baseline_bytes + additional
    microbatch_fits = variant != "tiled" or largest <= plan.matrix_bytes
    feasible = microbatch_fits and projected_peak <= safe_peak
    if not microbatch_fits:
        reason = "one minimum-size microbatch exceeds its per-slot budget"
    elif projected_peak > safe_peak:
        reason = "projected peak exceeds the reserved-VRAM boundary"
    else:
        reason = "within reserved-VRAM boundary"
    return CudaWorkloadEstimate(
        variant=variant,
        lanes=int(lanes),
        inflight_slots=active_slots,
        tile_bytes=int(tile_bytes),
        transient_bytes=int(transient_bytes),
        additional_bytes=int(additional),
        projected_peak_bytes=int(projected_peak),
        safe_peak_bytes=int(safe_peak),
        per_microbatch_bytes=int(plan.matrix_bytes),
        largest_microbatch_bytes=int(largest),
        feasible=bool(feasible),
        reason=reason,
    )


def run_tiled_accelerator_pipeline(
    tasks,
    *,
    store: EmbeddingTileStore,
    lengths,
    device,
    workers,
    lanes,
    alignment_callback: Callable,
    precision="float32",
    progress=None,
    matrix_budget_override=None,
    result_callback=None,
    result_chunk_size=65536,
    memory_plan_override=None,
    warmup_task_count=0,
    benchmark_timer=None,
):
    """Run the Aug 22 tiled producer and completion-driven CPU consumers."""
    backend = get_accelerator_backend(device)
    supported, reason = backend.supports_tiled(require_memory=True)
    if not supported:
        raise RuntimeError(
            f"Tiled execution is unavailable on '{device}': {reason}."
        )
    tasks = list(tasks)
    if not tasks:
        return []
    warmup_task_count = int(warmup_task_count)
    if warmup_task_count < 0 or warmup_task_count >= len(tasks):
        if benchmark_timer is not None:
            raise ValueError(
                "Benchmark warm-up count must leave at least one timed task."
            )
        warmup_task_count = 0

    plan = memory_plan_override or cuda_memory_plan(device, lanes=lanes)
    if int(plan.lanes) != max(1, int(lanes)):
        raise ValueError("CUDA memory plan lane count does not match execution lanes.")
    matrix_budget = int(matrix_budget_override or plan.matrix_bytes)
    per_block = max(1, plan.tile_cache_bytes // 2)
    block_ids = store.block_ids(per_block)
    results = []
    cpu_pending = set()
    inflight = deque()
    max_inflight = max(2, int(lanes) * 2)
    streams = [backend.create_stream() for _ in range(max(1, int(lanes)))]
    stream_cursor = 0

    def collect_cpu(block=False):
        _drain_completed_alignment_futures(
            cpu_pending,
            results,
            progress=progress,
            block=block,
        )
        if result_callback is not None and len(results) >= int(result_chunk_size):
            result_callback(results)
            results.clear()

    def submit_oldest(block):
        if not inflight:
            return
        event, host_tensor, metadata = inflight[0]
        if not block and not event.query():
            return
        if block:
            event.synchronize()
        inflight.popleft()
        matrix_array = host_tensor.numpy()
        for offset, (idx_i, idx_j, length) in enumerate(metadata):
            matrix = matrix_array[offset, :, :int(length)]
            if not np.isfinite(matrix).all():
                raise FloatingPointError(
                    "Batched accelerator scoring produced non-finite values."
                )
            cpu_pending.add(
                cpu_executor.submit(
                    alignment_callback,
                    (idx_i, idx_j, matrix),
                )
            )
        while len(cpu_pending) >= max(1, int(workers)) * 2:
            collect_cpu(block=True)

    with cuda_matmul_precision(precision), torch.inference_mode(), \
            h5py.File(store.path, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="alignment-cpu",
            ) as cpu_executor:
        group = hf["embeddings"]

        def run_phase(phase_tasks):
            nonlocal stream_cursor
            for tile_tasks in _partition_tiles(phase_tasks, block_ids):
                tile_indices = {int(task[0]) for task in tile_tasks}
                tile_indices.update(int(task[1]) for task in tile_tasks)
                host_embeddings = store.load_indices(tile_indices, group)
                preload_stream = streams[stream_cursor % len(streams)]
                with backend.stream_context(preload_stream):
                    gpu_embeddings = {
                        index: _to_normalized_cuda(array, device)
                        for index, array in host_embeddings.items()
                    }
                    preload_event = backend.create_event()
                    preload_event.record(preload_stream)
                preload_event.synchronize()
                del host_embeddings

                rows = OrderedDict()
                for task in tile_tasks:
                    rows.setdefault(int(task[0]), []).append(task)
                for idx_i, row_tasks in rows.items():
                    for microbatch in _length_microbatches(
                        row_tasks,
                        lengths,
                        lengths[idx_i],
                        matrix_budget,
                        store.feature_dimension,
                    ):
                        stream = streams[stream_cursor % len(streams)]
                        stream_cursor += 1
                        target_lengths = [
                            int(lengths[int(task[1])]) for task in microbatch
                        ]
                        targets = [
                            gpu_embeddings[int(task[1])] for task in microbatch
                        ]
                        with backend.stream_context(stream):
                            matrices = _batched_score_matrices(
                                gpu_embeddings[idx_i],
                                targets,
                                target_lengths,
                            )
                            host_tensor = _host_output_buffer(matrices)
                            host_tensor.copy_(
                                matrices, non_blocking=host_tensor.is_pinned()
                            )
                            event = backend.create_event()
                            event.record(stream)
                        metadata = [
                            (int(task[0]), int(task[1]), length)
                            for task, length in zip(microbatch, target_lengths)
                        ]
                        inflight.append((event, host_tensor, metadata))
                        while len(inflight) >= max_inflight:
                            submit_oldest(block=True)
                        submit_oldest(block=False)
                        collect_cpu(block=False)

                while inflight:
                    submit_oldest(block=True)
                del gpu_embeddings
                backend.empty_cache()

            while cpu_pending:
                collect_cpu(block=True)

        if benchmark_timer is None:
            run_phase(tasks)
        else:
            if warmup_task_count:
                run_phase(tasks[:warmup_task_count])
            benchmark_timer.start()
            run_phase(tasks[warmup_task_count:])
            benchmark_timer.stop()

    if result_callback is not None and results:
        result_callback(results)
        results.clear()

    return results


def run_tiled_cuda_pipeline(*args, **kwargs):
    """Compatibility name for the CUDA/XPU tiled alignment pipeline."""
    return run_tiled_accelerator_pipeline(*args, **kwargs)


def _fixed_query_task_index(task):
    return int(task[0])


def estimate_fixed_query_cuda_working_set(
    tasks,
    *,
    query_embedding,
    store: EmbeddingTileStore,
    lengths,
    device,
    lanes=1,
    memory_info=None,
    task_index: Callable = _fixed_query_task_index,
):
    """Estimate a fixed-query tiled search without allocating CUDA tensors."""
    plan = cuda_memory_plan(device, lanes=lanes, memory_info=memory_info)
    tasks = list(tasks)
    baseline = max(0, plan.total_bytes - plan.free_bytes)
    safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
    query = np.asarray(query_embedding)
    query_length = int(query.shape[0])
    query_bytes = int(query.size) * 4
    if not tasks:
        return CudaWorkloadEstimate(
            variant="fixed_query_tiled",
            lanes=int(lanes),
            inflight_slots=plan.inflight_slots,
            tile_bytes=query_bytes,
            transient_bytes=0,
            additional_bytes=query_bytes,
            projected_peak_bytes=baseline + query_bytes,
            safe_peak_bytes=safe_peak,
            per_microbatch_bytes=plan.matrix_bytes,
            largest_microbatch_bytes=0,
            feasible=baseline + query_bytes <= safe_peak,
            reason="empty workload",
        )

    indexed = [(task_index(task), task) for task in tasks]
    per_block = max(1, plan.tile_cache_bytes - query_bytes)
    block_ids = store.block_ids(per_block)
    grouped = OrderedDict()
    for index, task in indexed:
        grouped.setdefault(int(block_ids[index]), []).append(task)

    tile_bytes = query_bytes
    workspaces = []
    for block_tasks in grouped.values():
        indices = {task_index(task) for task in block_tasks}
        tile_bytes = max(
            tile_bytes,
            query_bytes + sum(store.float32_bytes[index] for index in indices),
        )
        pseudo_tasks = [(0, task_index(task), task) for task in block_tasks]
        for microbatch in _length_microbatches(
            pseudo_tasks,
            lengths,
            query_length,
            plan.matrix_bytes,
            store.feature_dimension,
        ):
            target_lengths = [int(lengths[int(item[1])]) for item in microbatch]
            workspaces.append(
                _microbatch_workspace_bytes(
                    query_length,
                    target_lengths,
                    store.feature_dimension,
                )
            )

    largest = max(workspaces, default=0)
    transient = sum(sorted(workspaces, reverse=True)[:plan.inflight_slots])
    additional = tile_bytes + transient
    projected = baseline + additional
    feasible = largest <= plan.matrix_bytes and projected <= safe_peak
    if largest > plan.matrix_bytes:
        reason = "one minimum-size microbatch exceeds its per-slot budget"
    elif projected > safe_peak:
        reason = "projected peak exceeds the reserved-VRAM boundary"
    else:
        reason = "within reserved-VRAM boundary"
    return CudaWorkloadEstimate(
        variant="fixed_query_tiled",
        lanes=int(lanes),
        inflight_slots=plan.inflight_slots,
        tile_bytes=int(tile_bytes),
        transient_bytes=int(transient),
        additional_bytes=int(additional),
        projected_peak_bytes=int(projected),
        safe_peak_bytes=int(safe_peak),
        per_microbatch_bytes=int(plan.matrix_bytes),
        largest_microbatch_bytes=int(largest),
        feasible=bool(feasible),
        reason=reason,
    )


def run_fixed_query_cuda_pipeline(
    tasks,
    *,
    query_embedding,
    store: EmbeddingTileStore,
    lengths,
    device,
    workers,
    lanes,
    alignment_callback: Callable,
    precision="float32",
    progress=None,
    matrix_budget_override=None,
    task_index: Callable = _fixed_query_task_index,
):
    """Batch a fixed query against target embeddings and overlap CPU scoring."""
    if getattr(device, "type", None) != "cuda":
        raise ValueError("The fixed-query tiled engine currently requires CUDA.")
    tasks = list(tasks)
    if not tasks:
        return []

    plan = cuda_memory_plan(device, lanes=lanes)
    matrix_budget = int(matrix_budget_override or plan.matrix_bytes)
    query_array = np.asarray(query_embedding)
    query_length = int(query_array.shape[0])
    query_bytes = int(query_array.size) * 4
    per_block = max(1, plan.tile_cache_bytes - query_bytes)
    block_ids = store.block_ids(per_block)
    grouped = OrderedDict()
    for task in tasks:
        index = task_index(task)
        grouped.setdefault(int(block_ids[index]), []).append(task)

    results = []
    cpu_pending = set()
    inflight = deque()
    max_inflight = max(2, int(lanes) * 2)
    streams = [torch.cuda.Stream(device=device) for _ in range(max(1, int(lanes)))]
    stream_cursor = 0

    def collect_cpu(block=False):
        nonlocal cpu_pending
        if block and cpu_pending:
            completed, _ = wait(cpu_pending, return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in cpu_pending if future.done()}
        for future in completed:
            cpu_pending.remove(future)
            results.append(future.result())
            if progress is not None:
                progress.update(1)

    def submit_oldest(block):
        if not inflight:
            return
        event, host_tensor, metadata = inflight[0]
        if not block and not event.query():
            return
        if block:
            event.synchronize()
        inflight.popleft()
        matrices = host_tensor.numpy()
        for offset, (task, target_length) in enumerate(metadata):
            matrix = matrices[offset, :, :int(target_length)]
            if not np.isfinite(matrix).all():
                raise FloatingPointError(
                    "Fixed-query accelerator scoring produced non-finite values."
                )
            cpu_pending.add(
                cpu_executor.submit(
                    alignment_callback,
                    task,
                    query_length,
                    int(target_length),
                    matrix,
                )
            )
        while len(cpu_pending) >= max(1, int(workers)) * 2:
            collect_cpu(block=True)

    with cuda_matmul_precision(precision), torch.inference_mode(), \
            h5py.File(store.path, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="search-alignment-cpu",
            ) as cpu_executor:
        preload_stream = streams[0]
        with torch.cuda.stream(preload_stream):
            query_tensor = _to_normalized_cuda(query_array, device)
            query_event = torch.cuda.Event()
            query_event.record(preload_stream)
        query_event.synchronize()
        group = hf["embeddings"]

        for block_tasks in grouped.values():
            indices = {task_index(task) for task in block_tasks}
            host_embeddings = store.load_indices(indices, group)
            preload_stream = streams[stream_cursor % len(streams)]
            with torch.cuda.stream(preload_stream):
                gpu_embeddings = {
                    index: _to_normalized_cuda(array, device)
                    for index, array in host_embeddings.items()
                }
                preload_event = torch.cuda.Event()
                preload_event.record(preload_stream)
            preload_event.synchronize()
            del host_embeddings

            pseudo_tasks = [(0, task_index(task), task) for task in block_tasks]
            for microbatch in _length_microbatches(
                pseudo_tasks,
                lengths,
                query_length,
                matrix_budget,
                store.feature_dimension,
            ):
                stream = streams[stream_cursor % len(streams)]
                stream_cursor += 1
                target_lengths = [int(lengths[int(item[1])]) for item in microbatch]
                target_tensors = [gpu_embeddings[int(item[1])] for item in microbatch]
                with torch.cuda.stream(stream):
                    matrices = _batched_score_matrices(
                        query_tensor,
                        target_tensors,
                        target_lengths,
                    )
                    host_tensor = _host_output_buffer(matrices)
                    host_tensor.copy_(matrices, non_blocking=host_tensor.is_pinned())
                    event = torch.cuda.Event()
                    event.record(stream)
                metadata = [
                    (item[2], length)
                    for item, length in zip(microbatch, target_lengths)
                ]
                inflight.append((event, host_tensor, metadata))
                while len(inflight) >= max_inflight:
                    submit_oldest(block=True)
                submit_oldest(block=False)
                collect_cpu(block=False)

            while inflight:
                submit_oldest(block=True)
            del gpu_embeddings
            torch.cuda.empty_cache()

        while cpu_pending:
            collect_cpu(block=True)
    return results


def compare_precision_results(fp32_results, tf32_results, per_residue_tolerance=1e-3):
    """Validate TF32 alignment decisions and length-normalized score drift."""
    fp32 = {tuple(result[:2]): result for result in fp32_results}
    tf32 = {tuple(result[:2]): result for result in tf32_results}
    if fp32.keys() != tf32.keys():
        return False, "pair identities differ"
    for pair in fp32:
        baseline = fp32[pair]
        candidate = tf32[pair]
        if int(baseline[3]) != int(candidate[3]) or int(baseline[5]) != int(candidate[5]):
            return False, f"alignment length changed for pair {pair}"
        for score_index, length_index in ((2, 3), (4, 5)):
            baseline_score = float(baseline[score_index])
            candidate_score = float(candidate[score_index])
            if not np.isfinite(candidate_score):
                return False, f"non-finite TF32 score for pair {pair}"
            scale = max(1, int(baseline[length_index]))
            if abs(candidate_score - baseline_score) / scale > per_residue_tolerance:
                return False, f"score tolerance exceeded for pair {pair}"
    return True, "alignment lengths and per-residue scores passed"
