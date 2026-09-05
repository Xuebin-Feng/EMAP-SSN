# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Memory-bounded accelerator execution for residue-embedding alignments.

The engine is intentionally independent from a particular tool's cache and
output schema.  Callers supply the pair tasks and the CPU alignment callback;
the engine owns only embedding caching, accelerator batching, and overlap.
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
MIN_MPS_RESERVE = 2 * GIB
PADDING_OVERHEAD_LIMIT = 0.15
MATRIX_WORKSPACE_MULTIPLIER = 8
CUDA_TILE_FRACTION = 0.30
CUDA_MATRIX_FRACTION = 0.50
TILE_MEMORY_PROFILES = (
    ("matrix-heavy", 0.20, 0.60),
    ("balanced", 0.30, 0.50),
    ("tile-heavy", 0.40, 0.40),
)
SUPPORTED_TILED_BACKENDS = frozenset({"cuda", "xpu", "mps"})
BF16_ACCELERATOR_BACKENDS = frozenset({"cuda", "xpu", "mps"})
bf16_support_cache = {}


class BF16ValidationIntegrityError(ValueError):
    """Raised when an FP32/BF16 comparison cannot produce a valid report."""


@dataclass(frozen=True)
class BF16ValidationExtreme:
    """Worst observed change for one metric in a BF16 validation sample."""

    metric: str
    identity: object
    baseline_value: object
    candidate_value: object
    signed_difference: float
    percentage_change: float


@dataclass(frozen=True)
class BF16DistributionStatistics:
    """Descriptive statistics for one non-negative BF16 drift series."""

    count: int
    mean: float
    median: float
    p95: float
    p99: float
    maximum: float


@dataclass(frozen=True)
class BF16ModeStatistics:
    """Alignment-length and raw-score drift for one alignment mode."""

    mode: str
    exact_length_count: int
    changed_length_count: int
    absolute_length_difference: BF16DistributionStatistics
    length_percentage_drift: BF16DistributionStatistics
    changed_absolute_length_difference: BF16DistributionStatistics | None
    changed_length_percentage_drift: BF16DistributionStatistics | None
    score_percentage_drift: BF16DistributionStatistics
    worst_absolute_length: BF16ValidationExtreme
    worst_relative_length: BF16ValidationExtreme
    worst_score: BF16ValidationExtreme


@dataclass(frozen=True)
class BF16ValidationReport:
    """Informational FP32/BF16 comparison with no numerical rejection gate."""

    sample_count: int
    changed_case_count: int
    modes: tuple[BF16ModeStatistics, ...]


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


BENCHMARK_TRIAL_SECONDS = 5.0


class BenchmarkTrial(BenchmarkPhaseTimer):
    """Soft submission deadline; elapsed time includes draining accepted work."""

    def __init__(self, seconds=BENCHMARK_TRIAL_SECONDS, clock=None):
        super().__init__()
        self.seconds = float(seconds)
        if not self.seconds > 0:
            raise ValueError("Benchmark duration must be positive.")
        self.clock = clock or time.perf_counter
        self.submitted = 0
        self.completed = 0
        self.tiles = 0
        self.microbatches = 0
        self.deadline = None
        self.stop_reason = None

    @staticmethod
    def warmup_count(tasks, workers, lanes=1):
        return min(len(tasks), max(32, 2 * int(workers), 2 * int(lanes)))

    def start(self):
        if self.started_at is not None:
            raise RuntimeError("Benchmark trial has already started.")
        self.started_at = self.clock()
        self.deadline = self.started_at + self.seconds

    def can_submit(self):
        if self.started_at is None:
            raise RuntimeError("Benchmark trial has not started.")
        return self.submitted == 0 or self.clock() < self.deadline

    def stop(self, total_pairs):
        if self.started_at is None or self.stopped_at is not None:
            raise RuntimeError("Benchmark trial is not running.")
        if not self.completed or self.completed != self.submitted:
            raise RuntimeError("Benchmark trial did not drain all submitted pairs.")
        self.stopped_at = self.clock()
        self.stop_reason = (
            "batch finished" if self.completed == total_pairs else "deadline"
        )

    @property
    def rate(self):
        return self.completed / self.elapsed

    def status(self, tiled=False):
        detail = f"{self.completed} pairs, {self.elapsed:.3f}s, {self.stop_reason}"
        if tiled:
            detail += f", {self.tiles} tiles, {self.microbatches} microbatches"
        return detail


def run_bounded_cpu_trial(pool, callback, tasks, workers, trial):
    """Warm and time one existing process pool with bounded ten-pair chunks."""
    def run_phase(phase_tasks, active=None):
        pending = deque()
        offset = 0
        results = []
        while offset < len(phase_tasks) or pending:
            while offset < len(phase_tasks) and len(pending) < max(1, workers):
                if active is not None and not active.can_submit():
                    offset = len(phase_tasks)
                    break
                chunk = phase_tasks[offset:offset + 10]
                pending.append(pool.map_async(callback, chunk, chunksize=10))
                offset += len(chunk)
                if active is not None:
                    active.submitted += len(chunk)
            if pending:
                completed = pending.popleft().get()
                results.extend(completed)
                if active is not None:
                    active.completed += len(completed)
        return results

    run_phase(tasks[:trial.warmup_count(tasks, workers)])
    trial.start()
    results = run_phase(tasks, trial)
    trial.stop(len(tasks))
    return results


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
    normalized = (
        "automatic_32bit" if value is None else str(value).strip().lower()
    )
    aliases = {
        "auto": "automatic_32bit",
        "ieee": "float32",
        "fp32": "float32",
        "ieee_fp32": "float32",
        "tfloat32": "tf32",
        "bfloat16": "bf16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"automatic_32bit", "float32", "tf32", "bf16"}:
        raise ValueError(
            "ACCELERATOR_PRECISION must be automatic_32bit, float32, tf32, "
            "or bf16."
        )
    return normalized


def precision_compute_dtype(precision):
    """Return the operand dtype for a canonical configured/result precision."""
    normalized = str(precision).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def precision_element_bytes(precision):
    """Return resident embedding bytes per element for ``precision``."""
    return 2 if precision_compute_dtype(precision) == torch.bfloat16 else 4


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
    """Portable CUDA/ROCm, XPU, and MPS tiled-runtime adapter."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.device_type = self.device.type
        if self.device_type not in SUPPORTED_TILED_BACKENDS:
            raise ValueError(
                "Tiled execution requires a CUDA/ROCm, XPU, or MPS accelerator; "
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
        if not self.supports_async_streams:
            raise RuntimeError(
                f"torch.{self.device_type} uses the default synchronous queue."
            )
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
        if not self.supports_async_streams:
            raise RuntimeError(
                f"torch.{self.device_type} does not use tiled pipeline events."
            )
        event_type = getattr(self.module, "Event", None)
        if event_type is None:
            raise RuntimeError(
                f"torch.{self.device_type}.Event is unavailable."
            )
        return event_type()

    def memory_info(self):
        if self.device_type == "mps":
            snapshot = self.memory_snapshot()
            return snapshot.free_bytes, snapshot.total_bytes
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

    def synchronize(self):
        function = self._runtime_function("synchronize")
        if function is None:
            return
        try:
            function(self.device)
        except TypeError:
            function()

    @property
    def supports_async_streams(self):
        return self.device_type in {"cuda", "xpu"}

    def memory_snapshot(self):
        """Return a conservative backend-neutral accelerator memory view."""
        if self.device_type != "mps":
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
            total_bytes = int(total_bytes)
            free_bytes = int(free_bytes)
            reserve = max(MIN_CUDA_RESERVE, int(total_bytes * 0.15))
            return AcceleratorMemorySnapshot(
                backend=self.device_type,
                free_bytes=free_bytes,
                total_bytes=total_bytes,
                reserve_bytes=reserve,
                source="mem_get_info",
                unified_memory=False,
            )

        recommended = self._runtime_function("recommended_max_memory")
        driver_allocated = self._runtime_function("driver_allocated_memory")
        if recommended is None or driver_allocated is None:
            raise RuntimeError(
                "torch.mps must expose recommended_max_memory and "
                "driver_allocated_memory for safe tiled execution."
            )
        total_bytes = int(recommended())
        allocated_bytes = int(driver_allocated())
        if (
            total_bytes <= 0
            or allocated_bytes < 0
            or allocated_bytes > total_bytes
        ):
            raise RuntimeError("torch.mps returned invalid memory information.")
        _system_total, system_available = system_memory_bytes()
        free_bytes = max(0, total_bytes - allocated_bytes)
        if system_available > 0:
            free_bytes = min(free_bytes, int(system_available))
        reserve = max(MIN_MPS_RESERVE, int(total_bytes * 0.20))
        return AcceleratorMemorySnapshot(
            backend="mps",
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            reserve_bytes=reserve,
            source="recommended_max_memory/driver_allocated_memory/system_ram",
            unified_memory=True,
        )

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
        missing = []
        if self.supports_async_streams:
            missing.extend(
                name for name in ("Stream", "Event", "stream")
                if getattr(self.module, name, None) is None
            )
        if require_memory:
            if self.device_type == "mps":
                missing.extend(
                    name for name in (
                        "recommended_max_memory", "driver_allocated_memory"
                    )
                    if self._runtime_function(name) is None
                )
            elif self._runtime_function("mem_get_info") is None:
                missing.append("mem_get_info")
        if missing:
            return False, (
                f"torch.{self.device_type} is missing " + ", ".join(missing)
            )
        if require_memory:
            try:
                snapshot = self.memory_snapshot()
            except Exception as error:
                return False, str(error)
            if snapshot.total_bytes <= 0 or snapshot.usable_bytes <= 0:
                return False, (
                    f"torch.{self.device_type} reported no safely usable "
                    "accelerator memory"
                )
        if self.supports_async_streams:
            return True, "stream, event, and allocator APIs are available"
        return True, "default queue and working-set memory APIs are available"


def get_accelerator_backend(device):
    return AcceleratorBackend(device)


def tiled_accelerator_support(device, *, require_memory=True):
    try:
        backend = get_accelerator_backend(device)
    except (RuntimeError, ValueError) as error:
        return False, str(error)
    return backend.supports_tiled(require_memory=require_memory)


def bf16_accelerator_support(device, *, refresh=False):
    """Probe and cache usable BF16 matmul support on one accelerator device."""
    resolved = torch.device(device)
    cache_key = (resolved.type, resolved.index)
    if not refresh and cache_key in bf16_support_cache:
        return bf16_support_cache[cache_key]
    if resolved.type not in BF16_ACCELERATOR_BACKENDS:
        result = (False, "BF16 execution requires CUDA/ROCm, XPU, or MPS")
        bf16_support_cache[cache_key] = result
        return result

    backend = None
    try:
        backend = get_accelerator_backend(resolved)
        with torch.inference_mode():
            left = torch.arange(
                32, device=resolved, dtype=torch.float32
            ).reshape(4, 8).to(torch.bfloat16)
            right = torch.arange(
                24, device=resolved, dtype=torch.float32
            ).reshape(3, 8).to(torch.bfloat16)
            product = torch.mm(left, right.T).to(torch.float32)
            backend.synchronize()
            finite = bool(torch.isfinite(product).all().to("cpu").item())
        if not finite:
            raise FloatingPointError("the BF16 probe produced non-finite values")
        result = (True, "BF16 matmul and FP32 conversion succeeded")
    except Exception as error:
        result = (False, f"BF16 runtime probe failed: {error}")
    finally:
        if backend is not None:
            try:
                backend.empty_cache()
            except Exception:
                pass
    bf16_support_cache[cache_key] = result
    return result


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
        compute_dtype = precision_compute_dtype(precision)
        if compute_dtype != torch.float32:
            t_i = t_i.to(compute_dtype)
            t_j = t_j.to(compute_dtype)
        cosine = torch.mm(t_i, t_j.T).to(torch.float32).clamp(-1.0, 1.0)
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
class AcceleratorMemorySnapshot:
    backend: str
    free_bytes: int
    total_bytes: int
    reserve_bytes: int
    source: str
    unified_memory: bool = False

    @property
    def usable_bytes(self):
        return max(0, int(self.free_bytes) - int(self.reserve_bytes))


@dataclass(frozen=True)
class AcceleratorMemoryPlan:
    free_bytes: int
    total_bytes: int
    usable_bytes: int
    tile_cache_bytes: int
    matrix_pool_bytes: int
    matrix_bytes: int
    reserve_bytes: int
    lanes: int
    inflight_slots: int
    profile_name: str = "balanced"
    memory_source: str = "mem_get_info"
    compute_element_bytes: int = 4


# Compatibility for existing Network Injection, SSEARCH, and tests.
CudaMemoryPlan = AcceleratorMemoryPlan


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
    embedding_reload_bytes: int = 0
    microbatch_count: int = 0
    padded_elements: int = 0
    real_elements: int = 0
    schedule_signature: tuple = ()

    @property
    def padding_ratio(self):
        if self.real_elements <= 0:
            return 0.0
        return max(0.0, self.padded_elements / self.real_elements - 1.0)


@dataclass(frozen=True)
class AdaptiveTilePlan:
    """One memory-safe tiled execution candidate."""

    memory_plan: AcceleratorMemoryPlan
    estimate: CudaWorkloadEstimate

    @property
    def profile_name(self):
        return self.memory_plan.profile_name

    @property
    def lanes(self):
        return self.memory_plan.lanes

    @property
    def microbatch_workspace_bytes(self):
        """Expose tile memory as the final benchmark tie-break quantity."""
        return self.memory_plan.tile_cache_bytes


def accelerator_memory_plan(
    device,
    lanes=1,
    memory_info=None,
    *,
    memory_snapshot=None,
    tile_fraction=None,
    matrix_fraction=None,
    profile_name="balanced",
    compute_element_bytes=4,
):
    """Divide safe accelerator memory across tiles and microbatches."""
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
        raise ValueError("Accelerator tile and matrix fractions must be positive.")
    if tile_fraction + matrix_fraction > 0.80 + 1e-12:
        raise ValueError(
            "Accelerator tile and matrix fractions may use at most 80% of "
            "usable device memory."
        )
    if memory_info is not None and memory_snapshot is not None:
        raise ValueError("Provide memory_info or memory_snapshot, not both.")
    backend = get_accelerator_backend(device)
    if memory_snapshot is not None:
        snapshot = memory_snapshot
    elif memory_info is None:
        snapshot = backend.memory_snapshot()
    else:
        free_bytes, total_bytes = memory_info
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
        reserve = (
            max(MIN_MPS_RESERVE, int(total_bytes * 0.20))
            if backend.device_type == "mps"
            else max(MIN_CUDA_RESERVE, int(total_bytes * 0.15))
        )
        snapshot = AcceleratorMemorySnapshot(
            backend=backend.device_type,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            reserve_bytes=reserve,
            source="provided",
            unified_memory=backend.device_type == "mps",
        )
    free_bytes = int(snapshot.free_bytes)
    total_bytes = int(snapshot.total_bytes)
    reserve = int(snapshot.reserve_bytes)
    if (
        total_bytes <= 0
        or free_bytes < 0
        or free_bytes > total_bytes
        or reserve < 0
    ):
        raise ValueError("Accelerator memory snapshot contains invalid values.")
    usable = snapshot.usable_bytes
    inflight_slots = 1 if backend.device_type == "mps" else max(2, lanes * 2)
    matrix_pool = max(1, int(usable * matrix_fraction))
    return AcceleratorMemoryPlan(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        usable_bytes=usable,
        tile_cache_bytes=max(1, int(usable * tile_fraction)),
        matrix_pool_bytes=matrix_pool,
        matrix_bytes=max(1, matrix_pool // inflight_slots),
        reserve_bytes=reserve,
        lanes=lanes,
        inflight_slots=inflight_slots,
        profile_name=str(profile_name),
        memory_source=snapshot.source,
        compute_element_bytes=max(1, int(compute_element_bytes)),
    )


def cuda_memory_plan(*args, **kwargs):
    """Compatibility wrapper for the backend-neutral memory planner."""
    return accelerator_memory_plan(*args, **kwargs)


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

    def compute_bytes(self, element_bytes=4):
        """Return per-sequence resident bytes for the selected compute dtype."""
        element_bytes = max(1, int(element_bytes))
        return [
            int(np.prod(shape, dtype=np.int64)) * element_bytes
            for shape in self.shapes
        ]

    def block_ids(self, max_block_bytes, *, element_bytes=4):
        """Greedily assign sequences to contiguous compute-byte blocks."""
        max_block_bytes = max(1, int(max_block_bytes))
        compute_bytes = self.compute_bytes(element_bytes)
        block_ids = np.zeros(len(self.headers), dtype=np.int32)
        block = 0
        used = 0
        for index, size in enumerate(compute_bytes):
            if used and used + size > max_block_bytes:
                block += 1
                used = 0
            block_ids[index] = block
            used += size
        return block_ids


def _store_resident_bytes(store, lengths, element_bytes):
    """Return resident sizes while preserving compatibility with test stores."""
    if isinstance(store, EmbeddingTileStore):
        return store.compute_bytes(element_bytes)
    float32_bytes = getattr(store, "float32_bytes", None)
    if isinstance(float32_bytes, (list, tuple, np.ndarray)):
        scale = max(1, int(element_bytes)) / 4.0
        return [int(value * scale) for value in float32_bytes]
    feature_dimension = getattr(store, "feature_dimension", 1)
    if not isinstance(feature_dimension, (int, np.integer)):
        feature_dimension = 1
    return [
        int(length) * max(1, int(feature_dimension)) * max(1, int(element_bytes))
        for length in lengths
    ]


def _store_block_ids(store, max_block_bytes, lengths, element_bytes):
    if isinstance(store, EmbeddingTileStore):
        return store.block_ids(
            max_block_bytes, element_bytes=element_bytes
        )
    candidate = getattr(store, "block_ids", None)
    if callable(candidate):
        try:
            block_ids = np.asarray(candidate(max_block_bytes), dtype=np.int32)
            if block_ids.shape == (len(lengths),):
                return block_ids
        except (TypeError, ValueError):
            pass
    resident_bytes = _store_resident_bytes(store, lengths, element_bytes)
    block_ids = np.zeros(len(resident_bytes), dtype=np.int32)
    block = 0
    used = 0
    for index, size in enumerate(resident_bytes):
        if used and used + size > max(1, int(max_block_bytes)):
            block += 1
            used = 0
        block_ids[index] = block
        used += size
    return block_ids


def _to_normalized_cuda(array, device, precision="float32"):
    """Transfer and normalize one embedding using backend-safe copy semantics."""
    contiguous = np.ascontiguousarray(array)
    cpu_tensor = torch.from_numpy(contiguous)
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    use_pinned_copy = device_type in {"cuda", "xpu"}
    if use_pinned_copy:
        try:
            cpu_tensor = cpu_tensor.pin_memory()
        except RuntimeError:
            use_pinned_copy = False
    tensor = cpu_tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=use_pinned_copy,
    )
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
    tensor.div_(norms.clamp_min_(torch.finfo(torch.float32).tiny))
    return tensor.to(precision_compute_dtype(precision))


def _to_normalized_accelerator(array, device, precision):
    """Preserve the legacy two-argument helper call for 32-bit execution."""
    if precision_compute_dtype(precision) == torch.float32:
        return _to_normalized_cuda(array, device)
    return _to_normalized_cuda(array, device, precision)


def _microbatch_workspace_bytes(
    row_length,
    target_lengths,
    feature_dimension=0,
    compute_element_bytes=4,
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
        count
        * max_columns
        * max(0, int(feature_dimension))
        * max(1, int(compute_element_bytes))
    )
    return int(matrix_bytes + padded_target_bytes)


def _length_microbatches(
    tasks,
    lengths,
    row_length,
    matrix_budget,
    feature_dimension=0,
    compute_element_bytes=4,
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
            * (
                4 * int(row_length) * MATRIX_WORKSPACE_MULTIPLIER
                + max(0, int(feature_dimension))
                * max(1, int(compute_element_bytes))
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
        dtype=row_tensor.dtype,
        device=row_tensor.device,
    )
    for index, (target, length) in enumerate(zip(target_tensors, target_lengths)):
        targets[index, :int(length)].copy_(target)

    cosine = torch.matmul(
        row_tensor.unsqueeze(0), targets.transpose(1, 2)
    ).to(torch.float32)
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
    compute_element_bytes=4,
):
    """Estimate accelerator use and tiled schedule shape without allocating."""
    if memory_plan_override is not None and memory_info is not None:
        raise ValueError(
            "memory_info and memory_plan_override cannot be supplied together."
        )
    plan = memory_plan_override or accelerator_memory_plan(
        device,
        lanes=lanes,
        memory_info=memory_info,
        compute_element_bytes=compute_element_bytes,
    )
    if int(plan.lanes) != max(1, int(lanes)):
        raise ValueError(
            "Accelerator memory plan lane count does not match execution lanes."
        )
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

    feature_dimension = getattr(store, "feature_dimension", 0)
    if not isinstance(feature_dimension, (int, np.integer)):
        feature_dimension = 0
    feature_dimension = int(feature_dimension or 0)
    compute_element_bytes = int(
        getattr(plan, "compute_element_bytes", compute_element_bytes)
    )
    resident_bytes = _store_resident_bytes(
        store, lengths, compute_element_bytes
    )
    variant = str(variant).strip().lower()
    tile_bytes = 0
    workspaces = []
    embedding_reload_bytes = 0
    microbatch_count = 0
    padded_elements = 0
    real_elements = 0
    schedule_parts = []
    normalization_transient_bytes = 0
    if variant == "tiled":
        per_block = max(1, plan.tile_cache_bytes // 2)
        block_ids = _store_block_ids(
            store,
            per_block,
            lengths,
            compute_element_bytes,
        )
        for tile_tasks in _partition_tiles(tasks, block_ids):
            tile_indices = {int(task[0]) for task in tile_tasks}
            tile_indices.update(int(task[1]) for task in tile_tasks)
            resident = sum(resident_bytes[index] for index in tile_indices)
            embedding_reload_bytes += resident
            tile_bytes = max(
                tile_bytes,
                resident,
            )
            if compute_element_bytes < 4 and tile_indices:
                # Each embedding is normalized in FP32 before its resident
                # BF16 copy is retained. Loading is sequential, so only the
                # largest such FP32 normalization source is concurrent.
                normalization_transient_bytes = max(
                    normalization_transient_bytes,
                    max(
                        int(lengths[index]) * feature_dimension * 4
                        for index in tile_indices
                    ),
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
                    compute_element_bytes,
                ):
                    target_lengths = [
                        int(lengths[int(task[1])]) for task in microbatch
                    ]
                    microbatch_count += 1
                    max_columns = max(target_lengths)
                    real_elements += sum(target_lengths)
                    padded_elements += len(target_lengths) * max_columns
                    schedule_parts.append(
                        (
                            int(row),
                            tuple(int(task[1]) for task in microbatch),
                        )
                    )
                    workspaces.append(
                        _microbatch_workspace_bytes(
                            lengths[row],
                            target_lengths,
                            feature_dimension,
                            compute_element_bytes,
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
                resident_bytes[left] + resident_bytes[right]
            )
            normalization_transient = 0
            if compute_element_bytes < 4:
                normalization_transient = max(
                    int(lengths[left]) * feature_dimension * 4,
                    int(lengths[right]) * feature_dimension * 4,
                )
            workspaces.append(
                workspace + embedding_bytes + normalization_transient
            )
        active_slots = max(1, int(lanes))

    largest = max(workspaces, default=0)
    transient_bytes = sum(
        sorted(workspaces, reverse=True)[:active_slots]
    )
    if variant == "tiled":
        transient_bytes += normalization_transient_bytes
    additional = tile_bytes + transient_bytes
    projected_peak = baseline_bytes + additional
    microbatch_fits = variant != "tiled" or largest <= plan.matrix_bytes
    tile_fits = variant != "tiled" or tile_bytes <= plan.tile_cache_bytes
    feasible = tile_fits and microbatch_fits and projected_peak <= safe_peak
    if not tile_fits:
        reason = "one embedding block pair exceeds its tile-cache budget"
    elif not microbatch_fits:
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
        embedding_reload_bytes=int(embedding_reload_bytes),
        microbatch_count=int(microbatch_count),
        padded_elements=int(padded_elements),
        real_elements=int(real_elements),
        schedule_signature=(
            tuple(block_ids.tolist()) if variant == "tiled" else (),
            tuple(schedule_parts),
        ),
    )


def _tile_plan_dominates(left, right):
    """Return whether ``left`` is no worse than ``right`` on dry-run costs."""
    if left.lanes != right.lanes:
        return False
    left_values = (
        left.estimate.embedding_reload_bytes,
        left.estimate.microbatch_count,
        left.estimate.padded_elements,
        left.estimate.projected_peak_bytes,
    )
    right_values = (
        right.estimate.embedding_reload_bytes,
        right.estimate.microbatch_count,
        right.estimate.padded_elements,
        right.estimate.projected_peak_bytes,
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def build_adaptive_tile_plans(
    tasks,
    *,
    store,
    lengths,
    device,
    lane_candidates,
    memory_info=None,
    memory_snapshot=None,
    compute_element_bytes=4,
):
    """Build, deduplicate, and Pareto-prune safe tile-plan candidates."""
    tasks = list(tasks)
    candidates = []
    seen = set()
    for lanes in sorted(set(max(1, int(value)) for value in lane_candidates)):
        for profile_name, tile_fraction, matrix_fraction in TILE_MEMORY_PROFILES:
            plan = accelerator_memory_plan(
                device,
                lanes=lanes,
                memory_info=memory_info,
                memory_snapshot=memory_snapshot,
                tile_fraction=tile_fraction,
                matrix_fraction=matrix_fraction,
                profile_name=profile_name,
                compute_element_bytes=compute_element_bytes,
            )
            estimate = estimate_cuda_working_set(
                tasks,
                store=store,
                lengths=lengths,
                device=device,
                lanes=lanes,
                variant="tiled",
                memory_plan_override=plan,
                compute_element_bytes=compute_element_bytes,
            )
            if not estimate.feasible:
                continue
            signature = (lanes, estimate.schedule_signature)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(AdaptiveTilePlan(plan, estimate))

    frontier = []
    for candidate in candidates:
        if any(_tile_plan_dominates(other, candidate) for other in candidates):
            continue
        frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda candidate: (
            candidate.lanes,
            candidate.estimate.projected_peak_bytes,
            candidate.memory_plan.tile_cache_bytes,
            candidate.profile_name,
        ),
    )


def _run_synchronous_tiled_pipeline(
    tasks,
    *,
    store,
    lengths,
    device,
    workers,
    alignment_callback,
    precision,
    progress,
    result_callback,
    result_chunk_size,
    plan,
    matrix_budget,
    warmup_task_count,
    benchmark_timer,
    benchmark_trial=None,
):
    """Run tiled work on a default-queue backend such as Apple MPS."""
    backend = get_accelerator_backend(device)
    element_bytes = int(plan.compute_element_bytes)
    per_block = max(1, plan.tile_cache_bytes // 2)
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
    results = []
    active_trial = None
    cpu_pending = set()

    def collect_cpu(block=False):
        completed_count = _drain_completed_alignment_futures(
            cpu_pending,
            results,
            progress=progress,
            block=block,
        )
        if active_trial is not None:
            active_trial.completed += completed_count
        if result_callback is not None and len(results) >= int(result_chunk_size):
            result_callback(results)
            results.clear()

    with cuda_matmul_precision(precision), torch.inference_mode(), \
            h5py.File(store.path, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="alignment-cpu",
            ) as cpu_executor:
        group = hf["embeddings"]

        def run_phase(phase_tasks):
            for tile_tasks in _partition_tiles(phase_tasks, block_ids):
                if active_trial is not None:
                    if not active_trial.can_submit():
                        break
                    active_trial.tiles += 1
                tile_indices = {int(task[0]) for task in tile_tasks}
                tile_indices.update(int(task[1]) for task in tile_tasks)
                host_embeddings = store.load_indices(tile_indices, group)
                gpu_embeddings = {
                    index: _to_normalized_accelerator(array, device, precision)
                    for index, array in host_embeddings.items()
                }
                del host_embeddings

                rows = OrderedDict()
                for task in tile_tasks:
                    rows.setdefault(int(task[0]), []).append(task)
                for idx_i, row_tasks in rows.items():
                    if active_trial is not None and not active_trial.can_submit():
                        break
                    for microbatch in _length_microbatches(
                        row_tasks,
                        lengths,
                        lengths[idx_i],
                        matrix_budget,
                        store.feature_dimension,
                        element_bytes,
                    ):
                        if active_trial is not None:
                            if not active_trial.can_submit():
                                break
                            active_trial.submitted += len(microbatch)
                            active_trial.microbatches += 1
                        target_lengths = [
                            int(lengths[int(task[1])]) for task in microbatch
                        ]
                        targets = [
                            gpu_embeddings[int(task[1])] for task in microbatch
                        ]
                        matrices = _batched_score_matrices(
                            gpu_embeddings[idx_i], targets, target_lengths
                        )
                        backend.synchronize()
                        matrix_array = matrices.to(
                            dtype=torch.float32, device="cpu"
                        ).numpy()
                        for offset, (task, length) in enumerate(
                            zip(microbatch, target_lengths)
                        ):
                            matrix = matrix_array[offset, :, :int(length)]
                            if not np.isfinite(matrix).all():
                                raise FloatingPointError(
                                    "Batched accelerator scoring produced "
                                    "non-finite values."
                                )
                            cpu_pending.add(
                                cpu_executor.submit(
                                    alignment_callback,
                                    (int(task[0]), int(task[1]), matrix),
                                )
                            )
                        while len(cpu_pending) >= max(1, int(workers)) * 2:
                            collect_cpu(block=True)
                        collect_cpu(block=False)

                del gpu_embeddings
                backend.empty_cache()

            while cpu_pending:
                collect_cpu(block=True)

        if benchmark_trial is not None:
            run_phase(tasks[:benchmark_trial.warmup_count(tasks, workers, plan.lanes)])
            results.clear()
            active_trial = benchmark_trial
            benchmark_trial.start()
            run_phase(tasks)
            benchmark_trial.stop(len(tasks))
        elif benchmark_timer is None:
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
    benchmark_trial=None,
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

    plan = memory_plan_override or accelerator_memory_plan(
        device,
        lanes=lanes,
        compute_element_bytes=precision_element_bytes(precision),
    )
    if int(plan.lanes) != max(1, int(lanes)):
        raise ValueError(
            "Accelerator memory plan lane count does not match execution lanes."
        )
    matrix_budget = int(matrix_budget_override or plan.matrix_bytes)
    per_block = max(1, plan.tile_cache_bytes // 2)
    element_bytes = int(getattr(plan, "compute_element_bytes", 4))
    if getattr(backend, "supports_async_streams", True) is False:
        return _run_synchronous_tiled_pipeline(
            tasks,
            store=store,
            lengths=lengths,
            device=device,
            workers=workers,
            alignment_callback=alignment_callback,
            precision=precision,
            progress=progress,
            result_callback=result_callback,
            result_chunk_size=result_chunk_size,
            plan=plan,
            matrix_budget=matrix_budget,
            warmup_task_count=warmup_task_count,
            benchmark_timer=benchmark_timer,
            benchmark_trial=benchmark_trial,
        )
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
    results = []
    active_trial = None
    cpu_pending = set()
    inflight = deque()
    max_inflight = max(2, int(lanes) * 2)
    streams = [backend.create_stream() for _ in range(max(1, int(lanes)))]
    stream_cursor = 0

    def collect_cpu(block=False):
        completed_count = _drain_completed_alignment_futures(
            cpu_pending,
            results,
            progress=progress,
            block=block,
        )
        if active_trial is not None:
            active_trial.completed += completed_count
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
                if active_trial is not None:
                    if not active_trial.can_submit():
                        break
                    active_trial.tiles += 1
                tile_indices = {int(task[0]) for task in tile_tasks}
                tile_indices.update(int(task[1]) for task in tile_tasks)
                host_embeddings = store.load_indices(tile_indices, group)
                preload_stream = streams[stream_cursor % len(streams)]
                with backend.stream_context(preload_stream):
                    gpu_embeddings = {
                        index: _to_normalized_accelerator(
                            array, device, precision
                        )
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
                    if active_trial is not None and not active_trial.can_submit():
                        break
                    for microbatch in _length_microbatches(
                        row_tasks,
                        lengths,
                        lengths[idx_i],
                        matrix_budget,
                        store.feature_dimension,
                        element_bytes,
                    ):
                        if active_trial is not None:
                            if not active_trial.can_submit():
                                break
                            active_trial.submitted += len(microbatch)
                            active_trial.microbatches += 1
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

        if benchmark_trial is not None:
            run_phase(tasks[:benchmark_trial.warmup_count(tasks, workers, plan.lanes)])
            results.clear()
            active_trial = benchmark_trial
            benchmark_trial.start()
            run_phase(tasks)
            benchmark_trial.stop(len(tasks))
        elif benchmark_timer is None:
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
    precision="float32",
):
    """Estimate a fixed-query tiled search without allocating CUDA tensors."""
    element_bytes = precision_element_bytes(precision)
    plan = cuda_memory_plan(
        device,
        lanes=lanes,
        memory_info=memory_info,
        compute_element_bytes=element_bytes,
    )
    tasks = list(tasks)
    baseline = max(0, plan.total_bytes - plan.free_bytes)
    safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
    query = np.asarray(query_embedding)
    query_length = int(query.shape[0])
    query_bytes = int(query.size) * element_bytes
    normalization_transient = int(query.size) * 4 if element_bytes < 4 else 0
    if not tasks:
        return CudaWorkloadEstimate(
            variant="fixed_query_tiled",
            lanes=int(lanes),
            inflight_slots=plan.inflight_slots,
            tile_bytes=query_bytes,
            transient_bytes=normalization_transient,
            additional_bytes=query_bytes + normalization_transient,
            projected_peak_bytes=baseline + query_bytes + normalization_transient,
            safe_peak_bytes=safe_peak,
            per_microbatch_bytes=plan.matrix_bytes,
            largest_microbatch_bytes=0,
            feasible=baseline + query_bytes + normalization_transient <= safe_peak,
            reason="empty workload",
        )

    indexed = [(task_index(task), task) for task in tasks]
    per_block = max(1, plan.tile_cache_bytes - query_bytes)
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
    grouped = OrderedDict()
    for index, task in indexed:
        grouped.setdefault(int(block_ids[index]), []).append(task)

    tile_bytes = query_bytes
    workspaces = []
    target_normalization_transient = 0
    resident_bytes = _store_resident_bytes(store, lengths, element_bytes)
    float32_resident_bytes = _store_resident_bytes(store, lengths, 4)
    for block_tasks in grouped.values():
        indices = {task_index(task) for task in block_tasks}
        tile_bytes = max(
            tile_bytes,
            query_bytes
            + sum(resident_bytes[index] for index in indices),
        )
        if element_bytes < 4 and indices:
            target_normalization_transient = max(
                target_normalization_transient,
                max(float32_resident_bytes[index] for index in indices),
            )
        pseudo_tasks = [(0, task_index(task), task) for task in block_tasks]
        for microbatch in _length_microbatches(
            pseudo_tasks,
            lengths,
            query_length,
            plan.matrix_bytes,
            store.feature_dimension,
            element_bytes,
        ):
            target_lengths = [int(lengths[int(item[1])]) for item in microbatch]
            workspaces.append(
                _microbatch_workspace_bytes(
                    query_length,
                    target_lengths,
                    store.feature_dimension,
                    element_bytes,
                )
            )

    largest = max(workspaces, default=0)
    transient = normalization_transient + target_normalization_transient + sum(
        sorted(workspaces, reverse=True)[:plan.inflight_slots]
    )
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

    element_bytes = precision_element_bytes(precision)
    plan = cuda_memory_plan(
        device, lanes=lanes, compute_element_bytes=element_bytes
    )
    matrix_budget = int(matrix_budget_override or plan.matrix_bytes)
    query_array = np.asarray(query_embedding)
    query_length = int(query_array.shape[0])
    query_bytes = int(query_array.size) * element_bytes
    per_block = max(1, plan.tile_cache_bytes - query_bytes)
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
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
            query_tensor = _to_normalized_accelerator(
                query_array, device, precision
            )
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
                    index: _to_normalized_accelerator(array, device, precision)
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
                element_bytes,
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


def compare_precision_results(
    fp32_results,
    candidate_results,
    per_residue_tolerance=1e-3,
    candidate_label="candidate",
):
    """Validate alignment decisions and length-normalized precision drift."""
    fp32 = {tuple(result[:2]): result for result in fp32_results}
    candidate_map = {tuple(result[:2]): result for result in candidate_results}
    if fp32.keys() != candidate_map.keys():
        return False, "pair identities differ"
    for pair in fp32:
        baseline = fp32[pair]
        candidate = candidate_map[pair]
        if int(baseline[3]) != int(candidate[3]) or int(baseline[5]) != int(candidate[5]):
            return False, f"alignment length changed for pair {pair}"
        for score_index, length_index in ((2, 3), (4, 5)):
            baseline_score = float(baseline[score_index])
            candidate_score = float(candidate[score_index])
            if not np.isfinite(baseline_score) or not np.isfinite(candidate_score):
                return False, f"non-finite {candidate_label} score for pair {pair}"
            scale = max(1, int(baseline[length_index]))
            if abs(candidate_score - baseline_score) / scale > per_residue_tolerance:
                return False, f"score tolerance exceeded for pair {pair}"
    return True, "alignment lengths and per-residue scores passed"


def _bf16_percentage_change(baseline, candidate):
    """Return absolute candidate drift as a percentage of the FP32 value."""
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return abs(candidate - baseline) / abs(baseline) * 100.0


def _bf16_distribution(values):
    """Summarize a non-empty series using deterministic nearest-rank tails."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty BF16 drift series.")
    count = len(ordered)
    if any(np.isinf(value) for value in ordered):
        mean = float("inf")
    else:
        mean = float(sum(ordered) / count)
    midpoint = count // 2
    if count % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0

    def nearest_rank(fraction):
        index = max(0, min(count - 1, int(np.ceil(fraction * count)) - 1))
        return ordered[index]

    return BF16DistributionStatistics(
        count=count,
        mean=mean,
        median=float(median),
        p95=float(nearest_rank(0.95)),
        p99=float(nearest_rank(0.99)),
        maximum=float(ordered[-1]),
    )


def _bf16_extreme(metric, identity, baseline, candidate, percentage_change):
    return BF16ValidationExtreme(
        metric=str(metric),
        identity=identity,
        baseline_value=baseline,
        candidate_value=candidate,
        signed_difference=float(candidate) - float(baseline),
        percentage_change=float(percentage_change),
    )


def compare_bf16_metric_results(
    fp32_results,
    candidate_results,
    *,
    identity_label="pair",
    candidate_label="BF16",
):
    """Build an informational BF16 report from normalized alignment metrics.

    Each input maps an identity to an ordered mapping whose values are
    ``(raw_score, alignment_length)`` tuples. Finite numerical differences of
    any magnitude are reported and never rejected.
    """
    sample_count = len(fp32_results)
    if sample_count == 0:
        raise BF16ValidationIntegrityError("BF16 validation sample is empty")
    if fp32_results.keys() != candidate_results.keys():
        raise BF16ValidationIntegrityError(
            f"{identity_label} identities differ"
        )

    first_metrics = next(iter(fp32_results.values()))
    try:
        mode_order = tuple(first_metrics.keys())
    except AttributeError as error:
        raise BF16ValidationIntegrityError(
            f"invalid FP32 alignment metrics: {error}"
        ) from error
    if not mode_order:
        raise BF16ValidationIntegrityError("BF16 validation has no alignment modes")
    accumulators = {
        str(mode): {
            "absolute_length": [],
            "length_percentage": [],
            "changed_absolute_length": [],
            "changed_length_percentage": [],
            "score_percentage": [],
            "worst_absolute_length": None,
            "worst_relative_length": None,
            "worst_score": None,
        }
        for mode in mode_order
    }
    changed_case_count = 0
    for identity, baseline_metrics in fp32_results.items():
        candidate_metrics = candidate_results[identity]
        try:
            baseline_modes = tuple(baseline_metrics.keys())
            candidate_modes = tuple(candidate_metrics.keys())
        except AttributeError as error:
            raise BF16ValidationIntegrityError(
                f"invalid alignment metrics for {identity_label} {identity}: {error}"
            ) from error
        if set(baseline_modes) != set(mode_order) or set(candidate_modes) != set(
            mode_order
        ):
            raise BF16ValidationIntegrityError(
                f"alignment modes differ for {identity_label} {identity}",
            )

        parsed = []
        for mode in mode_order:
            baseline_values = baseline_metrics[mode]
            candidate_values = candidate_metrics[mode]
            try:
                baseline_score = float(baseline_values[0])
                baseline_length = int(baseline_values[1])
                candidate_score = float(candidate_values[0])
                candidate_length = int(candidate_values[1])
            except (IndexError, OverflowError, TypeError, ValueError) as error:
                raise BF16ValidationIntegrityError(
                    f"invalid {mode} result for {identity_label} {identity}: {error}",
                ) from error
            if baseline_length < 0 or candidate_length < 0:
                raise BF16ValidationIntegrityError(
                    f"negative {mode} alignment length for {identity_label} {identity}",
                )
            if not np.isfinite(baseline_score) or not np.isfinite(candidate_score):
                raise BF16ValidationIntegrityError(
                    f"non-finite {candidate_label} {mode} score for "
                    f"{identity_label} {identity}",
                )
            parsed.append(
                (
                    str(mode),
                    baseline_score,
                    baseline_length,
                    candidate_score,
                    candidate_length,
                )
            )

        case_changed = any(
            baseline_length != candidate_length
            for (
                _mode,
                _baseline_score,
                baseline_length,
                _candidate_score,
                candidate_length,
            ) in parsed
        )
        if case_changed:
            changed_case_count += 1
        for (
            mode,
            baseline_score,
            baseline_length,
            candidate_score,
            candidate_length,
        ) in parsed:
            accumulator = accumulators[mode]
            signed_length_difference = candidate_length - baseline_length
            absolute_length_difference = abs(signed_length_difference)
            length_percentage = _bf16_percentage_change(
                baseline_length, candidate_length
            )
            score_percentage = _bf16_percentage_change(
                baseline_score, candidate_score
            )
            accumulator["absolute_length"].append(absolute_length_difference)
            accumulator["length_percentage"].append(length_percentage)
            accumulator["score_percentage"].append(score_percentage)
            if signed_length_difference:
                accumulator["changed_absolute_length"].append(
                    absolute_length_difference
                )
                accumulator["changed_length_percentage"].append(length_percentage)

            worst_absolute = accumulator["worst_absolute_length"]
            if (
                worst_absolute is None
                or absolute_length_difference
                > abs(worst_absolute.signed_difference)
            ):
                accumulator["worst_absolute_length"] = _bf16_extreme(
                    f"{mode} absolute length difference",
                    identity,
                    baseline_length,
                    candidate_length,
                    length_percentage,
                )
            worst_relative = accumulator["worst_relative_length"]
            if (
                worst_relative is None
                or length_percentage > worst_relative.percentage_change
            ):
                accumulator["worst_relative_length"] = _bf16_extreme(
                    f"{mode} relative length drift",
                    identity,
                    baseline_length,
                    candidate_length,
                    length_percentage,
                )
            worst_score = accumulator["worst_score"]
            if worst_score is None or score_percentage > worst_score.percentage_change:
                accumulator["worst_score"] = _bf16_extreme(
                    f"{mode} raw-score drift",
                    identity,
                    baseline_score,
                    candidate_score,
                    score_percentage,
                )

    mode_statistics = []
    for mode in mode_order:
        accumulator = accumulators[str(mode)]
        changed_count = len(accumulator["changed_absolute_length"])
        changed_absolute = accumulator["changed_absolute_length"]
        changed_percentage = accumulator["changed_length_percentage"]
        mode_statistics.append(
            BF16ModeStatistics(
                mode=str(mode),
                exact_length_count=sample_count - changed_count,
                changed_length_count=changed_count,
                absolute_length_difference=_bf16_distribution(
                    accumulator["absolute_length"]
                ),
                length_percentage_drift=_bf16_distribution(
                    accumulator["length_percentage"]
                ),
                changed_absolute_length_difference=(
                    _bf16_distribution(changed_absolute)
                    if changed_absolute
                    else None
                ),
                changed_length_percentage_drift=(
                    _bf16_distribution(changed_percentage)
                    if changed_percentage
                    else None
                ),
                score_percentage_drift=_bf16_distribution(
                    accumulator["score_percentage"]
                ),
                worst_absolute_length=accumulator["worst_absolute_length"],
                worst_relative_length=accumulator["worst_relative_length"],
                worst_score=accumulator["worst_score"],
            )
        )
    return BF16ValidationReport(
        sample_count=sample_count,
        changed_case_count=changed_case_count,
        modes=tuple(mode_statistics),
    )


def compare_bf16_precision_results(
    fp32_results,
    candidate_results,
    *,
    candidate_label="BF16",
):
    """Report Align/Network Injection global and local BF16 differences."""
    fp32_results = list(fp32_results)
    candidate_results = list(candidate_results)
    try:
        fp32 = {tuple(result[:2]): result for result in fp32_results}
        candidate_map = {
            tuple(result[:2]): result for result in candidate_results
        }
    except TypeError as error:
        raise BF16ValidationIntegrityError(
            f"invalid alignment identity: {error}"
        ) from error
    if len(fp32) != len(fp32_results) or len(candidate_map) != len(candidate_results):
        raise BF16ValidationIntegrityError(
            "duplicate pair identities in validation results"
        )
    try:
        fp32_metrics = {
            pair: {
                "global": (result[2], result[3]),
                "local": (result[4], result[5]),
            }
            for pair, result in fp32.items()
        }
        candidate_metrics = {
            pair: {
                "global": (result[2], result[3]),
                "local": (result[4], result[5]),
            }
            for pair, result in candidate_map.items()
        }
    except (IndexError, TypeError) as error:
        raise BF16ValidationIntegrityError(
            f"invalid alignment result: {error}"
        ) from error
    return compare_bf16_metric_results(
        fp32_metrics,
        candidate_metrics,
        identity_label="pair",
        candidate_label=candidate_label,
    )


def bf16_validation_notice(*, tool_name, sample_count):
    """Return the once-per-job warning shown before informational validation."""
    return (
        f"[Precision] WARNING: {tool_name} is using explicit low-precision BF16. "
        f"The FP32 comparison on up to {int(sample_count)} representative cases "
        "is informational: finite alignment-length and score differences of any "
        "magnitude will be reported but will not prevent BF16 production work."
    )


def _format_bf16_number(value):
    value = float(value)
    if np.isposinf(value):
        return "inf"
    return f"{value:.3f}"


def _format_bf16_distribution(label, statistics, unit=""):
    suffix = str(unit)
    return (
        f"[Precision]   {label}: n={statistics.count}, "
        f"mean={_format_bf16_number(statistics.mean)}{suffix}, "
        f"median={_format_bf16_number(statistics.median)}{suffix}, "
        f"P95={_format_bf16_number(statistics.p95)}{suffix}, "
        f"P99={_format_bf16_number(statistics.p99)}{suffix}, "
        f"max={_format_bf16_number(statistics.maximum)}{suffix}"
    )


def _format_bf16_extreme(extreme, identity_label):
    return (
        f"[Precision]   worst {extreme.metric}: {identity_label} "
        f"{extreme.identity}, FP32={extreme.baseline_value}, "
        f"BF16={extreme.candidate_value}, "
        f"signed difference={extreme.signed_difference:+.6g}, "
        f"absolute change={_format_bf16_number(extreme.percentage_change)}%"
    )


def format_bf16_validation_report(
    report,
    *,
    context,
    identity_label="pair",
):
    """Format a complete informational BF16 validation report."""
    lines = [
        f"[Precision] BF16 validation report: {context}; "
        f"cases={report.sample_count}; any-length-changed="
        f"{report.changed_case_count}/{report.sample_count} "
        f"({report.changed_case_count / report.sample_count * 100.0:.3f}%)."
    ]
    for mode in report.modes:
        lines.append(
            f"[Precision] {mode.mode}: exact lengths="
            f"{mode.exact_length_count}/{report.sample_count}; changed lengths="
            f"{mode.changed_length_count}/{report.sample_count} "
            f"({mode.changed_length_count / report.sample_count * 100.0:.3f}%)."
        )
        lines.append(
            _format_bf16_distribution(
                "all-case absolute length difference",
                mode.absolute_length_difference,
                " residues",
            )
        )
        lines.append(
            _format_bf16_distribution(
                "all-case FP32-relative length drift",
                mode.length_percentage_drift,
                "%",
            )
        )
        if mode.changed_absolute_length_difference is not None:
            lines.append(
                _format_bf16_distribution(
                    "changed-only absolute length difference",
                    mode.changed_absolute_length_difference,
                    " residues",
                )
            )
            lines.append(
                _format_bf16_distribution(
                    "changed-only FP32-relative length drift",
                    mode.changed_length_percentage_drift,
                    "%",
                )
            )
        else:
            lines.append("[Precision]   changed-only length statistics: n=0")
        lines.append(
            _format_bf16_distribution(
                "all-case FP32-relative raw-score drift",
                mode.score_percentage_drift,
                "%",
            )
        )
        lines.extend(
            _format_bf16_extreme(extreme, identity_label)
            for extreme in (
                mode.worst_absolute_length,
                mode.worst_relative_length,
                mode.worst_score,
            )
        )
    return "\n".join(lines)
