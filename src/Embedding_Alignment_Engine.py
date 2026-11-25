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
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
from itertools import chain
import os
import queue
import sys
import threading
import time
import traceback
from typing import Callable

import h5py
import numpy as np
import psutil
import torch
from utilities import Align_GPU_Kernels
from utilities import GPU_Memory_View
from utilities import Tool_Supervisor
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


GIB = 1024 ** 3
DEFAULT_HOST_CACHE_CAP = 128 * GIB
MIN_HOST_RESERVE = 8 * GIB
MIN_CUDA_RESERVE = 2 * GIB
MIN_MPS_RESERVE = 2 * GIB
# The allocator cap: a tiled run may not reserve more device memory than its plan needs, plus this slack
# for fragmentation. SSN_GPU_ALLOCATOR_CAP=0 turns the cap off (read once), for comparing runs.
ALLOCATOR_CAP_SWITCH = "SSN_GPU_ALLOCATOR_CAP"
ALLOCATOR_CAP_SLACK_BYTES = GIB // 2
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


def _read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _parse_int_limit(text):
    if not text or text.lower() == "max":
        return None
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _read_cgroup_stat_field(stat_path, *field_names, default=0):
    text = _read_text_file(stat_path)
    if not text:
        return default
    wanted = set(field_names)
    values = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in wanted:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                pass
    for name in field_names:
        if name in values:
            return max(0, values[name])
    return default


def _cgroup_ancestors(root_dir, rel_path):
    root = os.path.normpath(root_dir)
    rel = (rel_path or "").strip().lstrip("/")
    current = os.path.normpath(os.path.join(root, rel)) if rel else root
    try:
        if os.path.commonpath([root, current]) != root:
            current = root
    except ValueError:
        current = root
    dirs = []
    while True:
        dirs.append(current)
        if current == root:
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return dirs


def _cgroup_v2_limits(
    host_total,
    *,
    proc_cgroup="/proc/self/cgroup",
    cgroup_root="/sys/fs/cgroup",
):
    text = _read_text_file(proc_cgroup)
    rel_path = ""
    if text:
        for line in text.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                rel_path = parts[2]
                break
    best_total = None
    best_avail = None
    for directory in _cgroup_ancestors(cgroup_root, rel_path):
        limit = _parse_int_limit(_read_text_file(os.path.join(directory, "memory.max")))
        if limit is None or limit >= host_total:
            continue
        current = _parse_int_limit(_read_text_file(os.path.join(directory, "memory.current")))
        if current is None:
            avail = limit
        else:
            reclaimable = _read_cgroup_stat_field(
                os.path.join(directory, "memory.stat"), "inactive_file"
            )
            avail = max(0, limit - max(0, current - reclaimable))
        best_total = limit if best_total is None else min(best_total, limit)
        best_avail = avail if best_avail is None else min(best_avail, avail)
    if best_total is None:
        return None
    return best_total, min(best_total, best_avail)


def _cgroup_v1_limits(
    host_total,
    *,
    proc_cgroup="/proc/self/cgroup",
    cgroup_root="/sys/fs/cgroup/memory",
):
    text = _read_text_file(proc_cgroup)
    rel_path = ""
    if text:
        for line in text.splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and "memory" in parts[1].split(","):
                rel_path = parts[2]
                break
    best_total = None
    best_avail = None
    # Linux cgroup v1 reports ~2^63 - PAGE_SIZE when unlimited.
    unlimited_sentinel = 1 << 60
    for directory in _cgroup_ancestors(cgroup_root, rel_path):
        stat_path = os.path.join(directory, "memory.stat")
        raw_limit = _parse_int_limit(
            _read_text_file(os.path.join(directory, "memory.limit_in_bytes"))
        )
        valid_limit = (
            raw_limit
            if raw_limit is not None
            and 0 < raw_limit < host_total
            and raw_limit < unlimited_sentinel
            else None
        )
        raw_h_limit = _read_cgroup_stat_field(
            stat_path, "hierarchical_memory_limit", default=None
        )
        valid_h_limit = (
            raw_h_limit
            if raw_h_limit is not None
            and 0 < raw_h_limit < host_total
            and raw_h_limit < unlimited_sentinel
            else None
        )
        if valid_limit is None and valid_h_limit is None:
            continue
        if valid_limit is None:
            limit = valid_h_limit
        elif valid_h_limit is None:
            limit = valid_limit
        else:
            limit = min(valid_limit, valid_h_limit)
        usage = _parse_int_limit(
            _read_text_file(os.path.join(directory, "memory.usage_in_bytes"))
        )
        if usage is None:
            avail = limit
        else:
            reclaimable = _read_cgroup_stat_field(
                stat_path,
                "total_inactive_file",
                "inactive_file",
            )
            avail = max(0, limit - max(0, usage - reclaimable))
        best_total = limit if best_total is None else min(best_total, limit)
        best_avail = avail if best_avail is None else min(best_avail, avail)
    if best_total is None:
        return None
    return best_total, min(best_total, best_avail)


def _parse_slurm_mem_bytes(raw_value):
    if not raw_value:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    lower = text.lower()
    suffix_multipliers = (
        ("kib", 1024),
        ("mib", 1024 ** 2),
        ("gib", 1024 ** 3),
        ("tib", 1024 ** 4),
        ("kb", 1024),
        ("mb", 1024 ** 2),
        ("gb", 1024 ** 3),
        ("tb", 1024 ** 4),
        ("k", 1024),
        ("m", 1024 ** 2),
        ("g", 1024 ** 3),
        ("t", 1024 ** 4),
    )
    # SLURM_MEM_PER_NODE and SLURM_MEM_PER_CPU are reported in MiB by default.
    scale = 1024 ** 2
    for suffix, mult in suffix_multipliers:
        if lower.endswith(suffix):
            scale = mult
            text = text[:-len(suffix)].strip()
            break
    try:
        amount = float(text)
    except ValueError:
        return None
    if not np.isfinite(amount) or amount <= 0:
        return None
    scaled = amount * scale
    if not np.isfinite(scaled) or scaled <= 0:
        return None
    return int(scaled)


def _slurm_memory_limits(host_total, environ=None):
    env = os.environ if environ is None else environ
    limit = _parse_slurm_mem_bytes(env.get("SLURM_MEM_PER_NODE"))
    if limit is None:
        per_cpu = _parse_slurm_mem_bytes(env.get("SLURM_MEM_PER_CPU"))
        if per_cpu is not None:
            cpus = 1
            for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
                raw_cpus = env.get(key)
                if raw_cpus:
                    try:
                        token = str(raw_cpus).split("(", 1)[0].split(",", 1)[0].strip()
                        parsed = int(token)
                        if parsed > 0:
                            cpus = parsed
                            break
                    except ValueError:
                        pass
            limit = per_cpu * cpus
    if limit is None or limit <= 0 or limit >= host_total:
        return None
    try:
        rss = int(psutil.Process().memory_info().rss)
    except (psutil.Error, OSError):
        rss = 0
    return limit, max(0, limit - max(0, rss))


def _container_memory_limits(host_total):
    """Return tighter ``(limit_total, limit_available)`` from Linux cgroups or SLURM."""
    if host_total <= 0:
        return None
    candidates = []
    if sys.platform.startswith("linux"):
        for probe in (_cgroup_v2_limits, _cgroup_v1_limits):
            result = probe(host_total)
            if result is not None:
                candidates.append(result)
    slurm = _slurm_memory_limits(host_total)
    if slurm is not None:
        candidates.append(slurm)
    if not candidates:
        return None
    best_total = min(item[0] for item in candidates)
    best_avail = min(item[1] for item in candidates)
    return best_total, min(best_total, best_avail)


def system_memory_bytes():
    """Return ``(total, available)`` physical-memory or container/job quota bytes.

    psutil's available figure counts memory the OS can hand out without
    swapping, such as reclaimable page cache: Linux's MemAvailable, macOS's
    free plus inactive pages and Windows's available physical memory. When
    running inside a Linux cgroup (v1 or v2) or SLURM job with a tighter
    memory quota, both totals are clamped to that quota so ``HOST_CACHE_GB='auto'``
    never oversubscribes the job allocation.
    """
    try:
        memory = psutil.virtual_memory()
    except (psutil.Error, OSError):
        return 0, 0
    total = int(memory.total)
    available = int(memory.available)
    quota = _container_memory_limits(total)
    if quota is not None:
        limit_total, limit_available = quota
        total = min(total, int(limit_total))
        available = min(available, int(limit_available), total)
    return total, available


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
    return min(int(requested * GIB), safe_limit) if total else int(requested * GIB)


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

    def _wait_for_recorded_memory(self, function):
        """In a tool restarted after a GPU out-of-memory crash, wait once, bounded, for the memory the failed run had free.

        Does nothing in any other process (utilities/Tool_Supervisor.py decides), and after the first call.
        """
        if self.device_type != "cuda":
            return

        def read():
            with self.device_context():
                try:
                    free_bytes, total_bytes = function(self.device)
                except TypeError:
                    free_bytes, total_bytes = function()
            return int(free_bytes), int(total_bytes)

        Tool_Supervisor.wait_for_gpu_memory(self.device, read)

    def memory_info(self):
        if self.device_type == "mps":
            snapshot = self.memory_snapshot()
            return snapshot.free_bytes, snapshot.total_bytes
        function = self._runtime_function("mem_get_info")
        if function is None:
            raise RuntimeError(
                f"torch.{self.device_type} does not expose allocator memory info."
            )
        self._wait_for_recorded_memory(function)
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
            self._wait_for_recorded_memory(function)
            with self.device_context():
                try:
                    free_bytes, total_bytes = function(self.device)
                except TypeError:
                    free_bytes, total_bytes = function()
            total_bytes = int(total_bytes)
            free_bytes = int(free_bytes)
            reserve = max(MIN_CUDA_RESERVE, int(total_bytes * 0.15))
            if self.device_type == "cuda":
                free_bytes = _free_beside_other_programs(self.device, free_bytes, total_bytes, reserve)
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


_other_programs_announced = set()  # device keys the "other programs" line was printed for


def _free_beside_other_programs(device, free_bytes, total_bytes, reserve, out=print):
    """The free figure a CUDA snapshot plans from: lowered so that plans leave out what other programs hold (Windows and WSL2).

    On Windows and under WSL2 the free figure of mem_get_info does not see other programs, so the usable memory (free minus
    reserve) is taken as the smaller of today's and ``room - margin``, where room is the card minus what other
    programs hold and torch's reserved pool. The correction travels in the free figure and not in the reserve
    because plans are rebuilt from a plan's free and total with the default reserve (memory_info=), which
    would drop a raised reserve but keeps a lowered free figure. Every other case (not Windows or WSL2, reader
    unknown, switch off) returns ``free_bytes`` unchanged. Never raises.
    """
    try:
        view = GPU_Memory_View.room_beside_other_programs(device, total_bytes)
        if view is None:
            return free_bytes
        room, others = view
        planned = max(0, int(room) - GPU_Memory_View.card_margin_bytes(total_bytes)) + int(reserve)
        lowered = min(int(free_bytes), planned)
        if lowered < int(free_bytes):
            key = str(device)
            if key not in _other_programs_announced:
                _other_programs_announced.add(key)
                out(
                    f"[Hardware] Other programs hold {others / GIB:.1f} GB of the "
                    f"{int(total_bytes) / GIB:.1f} GB on {device}; GPU plans leave that memory to them."
                )
        return lowered
    except Exception:
        return free_bytes


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
    # What the dry run says this plan needs, plus slack; 0 = derive it from the budgets (allocator_cap_bytes).
    allocator_cap_bytes: int = 0
    unified_memory: bool = False


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


_allocator_cap_enabled = None


def allocator_cap_enabled():
    """Whether the allocator cap is on: always, unless SSN_GPU_ALLOCATOR_CAP=0. Read once."""
    global _allocator_cap_enabled
    if _allocator_cap_enabled is None:
        _allocator_cap_enabled = os.environ.get(ALLOCATOR_CAP_SWITCH, "").strip() != "0"
    return _allocator_cap_enabled


def _clear_failed_frames(error):
    """Let go of the local variables of the frames a failed run's error carries in its traceback.

    The traceback keeps those frames, and their locals (padded targets, score matrices, row tensors) are
    device memory, for as long as anyone holds the error: a caller that stores it for a retry or reports it
    from inside its except block. Clearing them lets the cache be emptied and the retry start clean. The
    error itself, its type, message and traceback lines, is untouched; nothing here can raise.
    """
    try:
        traceback.clear_frames(error.__traceback__)
    except Exception:
        pass


def allocator_cap_bytes(plan, held=0):
    """The most device memory the caching allocator may reserve for one plan, or None.

    What the run needs, plus what this process already holds on the device (``held``): the plan's own need (the
    dry run's tile + transient bytes, when the plan carries it) or else its budgets (tile cache + matrix pool),
    plus slack for fragmentation. The fraction limits everything the process has on the device, while the plan
    counts only what the run adds. Never more than the card minus its reserve.
    """
    wanted = int(getattr(plan, "allocator_cap_bytes", 0) or 0)
    if wanted <= 0:
        wanted = int(plan.tile_cache_bytes) + int(plan.matrix_pool_bytes) + ALLOCATOR_CAP_SLACK_BYTES
    ceiling = int(plan.total_bytes) - int(plan.reserve_bytes)
    if ceiling <= 0 or int(plan.total_bytes) <= 0:
        return None
    return max(1, min(ceiling, max(0, int(held)) + wanted))


def _held_device_bytes(backend):
    """Bytes this process has allocated on the backend's device right now; 0 when that cannot be read."""
    try:
        function = getattr(getattr(backend, "module", None), "memory_allocated", None)
        return max(0, int(function(backend.device))) if function is not None else 0
    except Exception:
        return 0


@contextmanager
def allocator_cap(backend, plan):
    """Cap what the caching allocator may reserve while one tiled plan runs; yield the cap in bytes or None.

    Reserved memory ran to 2.3 times the allocated peak (microbatches grow in size, the allocator keeps taking
    new blocks) and filled the card; under WDDM the excess spills into system memory and is charged to the
    system's commit. With a cap the allocator frees its cache and retries instead, and an exhausted cap is a
    clean OutOfMemoryError that the callers already answer by halving the matrix budget. The cap starts from
    what the process already holds on the device (a retry after an error can still hold tensors of the failed
    run through its traceback). Only CUDA and ROCm
    (torch.cuda) are capped: MPS, XPU and any other backend run exactly as before. The cap is lifted again
    when the plan ends, also after an error, and nothing here can raise.
    """
    function = None
    cap = None
    applied = False
    try:
        if getattr(backend, "device_type", None) == "cuda" and allocator_cap_enabled():
            function = getattr(getattr(backend, "module", None), "set_per_process_memory_fraction", None)
            if function is not None:
                cap = allocator_cap_bytes(plan, _held_device_bytes(backend))
                if cap:
                    function(min(1.0, cap / float(plan.total_bytes)), backend.device)
                    applied = True
    except Exception:
        applied = False
    try:
        yield cap if applied else None
    finally:
        if applied:
            try:
                function(1.0, backend.device)
            except Exception:
                pass


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
    # A tool restarted after a GPU out-of-memory crash plans from a fraction of the free memory. Only the
    # budgets see it: free_bytes stays the real free memory, so the estimate's baseline and its
    # "Est. VRAM peak" show the real size of the plan.
    planning_free = Tool_Supervisor.planning_free_bytes(
        getattr(backend, "device", device), free_bytes, total_bytes
    )
    if planning_free != free_bytes:
        usable = max(0, planning_free - reserve)
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
        unified_memory=bool(snapshot.unified_memory),
    )


def cuda_memory_plan(*args, **kwargs):
    """Compatibility wrapper for the backend-neutral memory planner."""
    return accelerator_memory_plan(*args, **kwargs)


class _WindowSlabPrefetcher(threading.Thread):
    """Background HDF5 reader filling standby RAM slab buffers for windowed alignment.

    Owns its own read-only SWMR ``h5py.File`` handle so HDF5 objects are never
    shared across threads, and always closes that handle when stopped or joined.
    """

    def __init__(self, store, slab_bufs):
        super().__init__(name="embedding-window-prefetch", daemon=True)
        self._store = store
        self._slab_bufs = slab_bufs
        self._req = queue.Queue(maxsize=1)
        self._done = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._fatal_error = None
        self._pending = False

    def submit(self, slot, indices):
        if self._pending:
            raise RuntimeError("A slab prefetch is already in flight.")
        if self._fatal_error is not None:
            raise self._fatal_error
        if self._stop.is_set() or not self.is_alive():
            raise RuntimeError("Window slab prefetcher thread is not running.")
        self._pending = True
        self._req.put((int(slot), list(indices)))

    def wait_result(self):
        if not self._pending:
            return None
        while True:
            try:
                slot, slab_map, read_count, read_bytes, error = self._done.get(timeout=0.1)
                break
            except queue.Empty:
                if self._fatal_error is not None:
                    self._pending = False
                    raise self._fatal_error
                if not self.is_alive():
                    try:
                        slot, slab_map, read_count, read_bytes, error = self._done.get_nowait()
                        break
                    except queue.Empty:
                        self._pending = False
                        if self._fatal_error is not None:
                            raise self._fatal_error
                        raise RuntimeError("Window slab prefetcher thread exited unexpectedly.")
        self._pending = False
        if error is not None:
            raise error
        return slot, slab_map, read_count, read_bytes

    def run(self):
        try:
            with h5py.File(self._store.path, "r", libver="latest", swmr=True) as hf:
                group = hf["embeddings"]
                self._ready.set()
                while not self._stop.is_set():
                    try:
                        item = self._req.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if item is None:
                        return
                    slot, indices = item
                    try:
                        buf = self._slab_bufs[slot]
                        slab_map = {}
                        cursor = 0
                        read_bytes = 0
                        for idx in indices:
                            if self._stop.is_set():
                                self._done.put((slot, {}, 0, 0, None))
                                return
                            idx = int(idx)
                            rows = self._store.shapes[idx][0]
                            out = buf[cursor:cursor + rows]
                            dataset = h5py.h5d.open(
                                group.id, self._store.headers[idx].encode("utf-8")
                            )
                            dataset.read(
                                h5py.h5s.ALL,
                                h5py.h5s.ALL,
                                out,
                                h5py.h5t.py_create(out.dtype),
                            )
                            slab_map[idx] = (cursor, cursor + rows)
                            cursor += rows
                            read_bytes += self._store.source_bytes[idx]
                        self._done.put((slot, slab_map, len(indices), read_bytes, None))
                    except BaseException as error:
                        self._done.put((slot, None, 0, 0, error))
        except BaseException as error:
            self._fatal_error = error
            self._ready.set()
            if self._pending:
                self._done.put((-1, None, 0, 0, error))

    def close(self):
        self._stop.set()
        try:
            while not self._req.empty():
                self._req.get_nowait()
        except queue.Empty:
            pass
        try:
            self._req.put_nowait(None)
        except queue.Full:
            pass
        if self.ident is not None:
            self.join(timeout=10.0)
        self._pending = False


class EmbeddingTileStore:
    """Describe an embedding database and keep all or a sliding window in RAM."""

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
        self.disk_reads = 0
        self.disk_read_bytes = 0
        self._packed = None
        self._offsets = None
        self._anchor_buf = None
        self._anchor_map = {}
        self._slab_bufs = [None, None]
        self._active_slab_slot = -1
        self._active_slab_map = {}
        self._prefetcher = None
        if self.host_cache_bytes and self.total_source_bytes <= self.host_cache_bytes:
            self._load_full_cache()

    @property
    def fully_cached(self):
        return self._packed is not None

    @property
    def cached_bytes(self):
        if self.fully_cached:
            return self.total_source_bytes
        return self.window_cached_bytes

    @property
    def uniform_dtype(self):
        if not self.dtypes:
            return None
        first = self.dtypes[0]
        if any(dtype != first for dtype in self.dtypes):
            return None
        return first

    @property
    def supports_windowed_cache(self):
        return (
            not self.fully_cached
            and self.host_cache_bytes > 0
            and bool(self.headers)
            and self.uniform_dtype is not None
        )

    @property
    def window_active(self):
        return self._anchor_buf is not None or self._active_slab_slot >= 0

    @property
    def window_cached_bytes(self):
        total = 0
        if self._anchor_buf is not None:
            total += int(self._anchor_buf.nbytes)
        for buf in self._slab_bufs:
            if buf is not None:
                total += int(buf.nbytes)
        return total

    def all_cached(self, indices):
        """Return True if every sequence index in ``indices`` is currently resident in RAM."""
        if self._packed is not None:
            return True
        if not self._anchor_map and not self._active_slab_map:
            return False
        anchor = self._anchor_map
        slab = self._active_slab_map
        return all((int(idx) in anchor or int(idx) in slab) for idx in indices)

    def _load_full_cache(self):
        if not self.headers:
            return
        common_dtype = self.uniform_dtype
        if common_dtype is None:
            return
        total_rows = sum(shape[0] for shape in self.shapes)
        try:
            packed = np.empty((total_rows, self.feature_dimension), dtype=common_dtype)
        except MemoryError:
            self._packed = None
            self._offsets = None
            return
        offsets = np.zeros(len(self.headers) + 1, dtype=np.int64)
        with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
            group = hf["embeddings"]
            cursor = 0
            for index, header in enumerate(self.headers):
                rows = self.shapes[index][0]
                packed[cursor:cursor + rows] = group[header][:]
                cursor += rows
                offsets[index + 1] = cursor
                self.disk_reads += 1
                self.disk_read_bytes += self.source_bytes[index]
        self._packed = packed
        self._offsets = offsets

    def open_window(self, anchor_rows_cap, slab_rows_cap):
        """Allocate reusable RAM anchor + double-buffered slab arrays and start prefetcher."""
        self.clear_window()
        common_dtype = self.uniform_dtype
        if common_dtype is None or self.feature_dimension is None:
            return False
        anchor_rows_cap = max(0, int(anchor_rows_cap))
        slab_rows_cap = max(0, int(slab_rows_cap))
        if anchor_rows_cap <= 0:
            return False
        try:
            anchor_buf = np.empty((anchor_rows_cap, self.feature_dimension), dtype=common_dtype)
            if slab_rows_cap > 0:
                slab_bufs = [
                    np.empty((slab_rows_cap, self.feature_dimension), dtype=common_dtype),
                    np.empty((slab_rows_cap, self.feature_dimension), dtype=common_dtype),
                ]
            else:
                slab_bufs = [None, None]
        except MemoryError:
            self.clear_window()
            return False
        self._anchor_buf = anchor_buf
        self._anchor_map = {}
        self._slab_bufs = slab_bufs
        self._active_slab_slot = -1
        self._active_slab_map = {}
        if slab_rows_cap > 0:
            prefetcher = _WindowSlabPrefetcher(self, self._slab_bufs)
            self._prefetcher = prefetcher
            prefetcher.start()
            ready = prefetcher._ready.wait(timeout=30.0)
            if not ready or prefetcher._fatal_error is not None or not prefetcher.is_alive():
                self.clear_window()
                return False
        return True

    def load_anchor(self, indices):
        """Populate the reusable anchor buffer with ``indices`` (reusing RAM hits if present)."""
        if self._anchor_buf is None:
            raise RuntimeError("Windowed anchor buffer is not open.")
        indices = [int(i) for i in indices]
        need_rows = sum(self.shapes[i][0] for i in indices)
        if need_rows > self._anchor_buf.shape[0]:
            raise ValueError(
                f"Anchor load requires {need_rows} rows, exceeding capacity {self._anchor_buf.shape[0]}."
            )
        old_slab_map = self._active_slab_map
        old_slab_buf = (
            self._slab_bufs[self._active_slab_slot]
            if self._active_slab_slot >= 0
            else None
        )
        new_map = {}
        disk_needed = []
        cursor = 0
        for idx in indices:
            rows = self.shapes[idx][0]
            span = old_slab_map.get(idx) if old_slab_buf is not None else None
            if span is not None:
                self._anchor_buf[cursor:cursor + rows] = old_slab_buf[span[0]:span[1]]
            else:
                disk_needed.append((idx, cursor, rows))
            new_map[idx] = (cursor, cursor + rows)
            cursor += rows
        if disk_needed:
            with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
                group = hf["embeddings"]
                for idx, dst_start, rows in disk_needed:
                    out = self._anchor_buf[dst_start:dst_start + rows]
                    dataset = h5py.h5d.open(
                        group.id, self.headers[idx].encode("utf-8")
                    )
                    dataset.read(
                        h5py.h5s.ALL,
                        h5py.h5s.ALL,
                        out,
                        h5py.h5t.py_create(out.dtype),
                    )
                    self.disk_reads += 1
                    self.disk_read_bytes += self.source_bytes[idx]
        self._anchor_map = new_map
        # Clear any stale active slab from a previous anchor to avoid holding mismatched indices.
        self._active_slab_map = {}

    def prefetch_slab(self, indices):
        """Start asynchronous disk read of ``indices`` into the standby slab buffer."""
        if self._prefetcher is None:
            raise RuntimeError("Window slab prefetcher is not active.")
        standby = 0 if self._active_slab_slot != 0 else 1
        indices = [int(i) for i in indices]
        need_rows = sum(self.shapes[i][0] for i in indices)
        if need_rows > self._slab_bufs[standby].shape[0]:
            raise ValueError(
                f"Slab prefetch requires {need_rows} rows, exceeding capacity {self._slab_bufs[standby].shape[0]}."
            )
        self._prefetcher.submit(standby, indices)

    def activate_prefetched_slab(self):
        """Wait for the in-flight slab prefetch and swap it in as the active RAM slab."""
        if self._prefetcher is None:
            return
        result = self._prefetcher.wait_result()
        if result is None:
            return
        slot, slab_map, read_count, read_bytes = result
        self._active_slab_slot = slot
        self._active_slab_map = slab_map
        self.disk_reads += read_count
        self.disk_read_bytes += read_bytes

    def clear_window(self):
        """Stop the prefetch thread and release all windowed RAM buffers."""
        if self._prefetcher is not None:
            try:
                self._prefetcher.close()
            finally:
                self._prefetcher = None
        self._anchor_map = {}
        self._active_slab_map = {}
        self._active_slab_slot = -1
        self._anchor_buf = None
        self._slab_bufs = [None, None]

    def get(self, index, h5_group=None):
        index = int(index)
        if self._packed is not None:
            start, end = self._offsets[index:index + 2]
            return self._packed[int(start):int(end)]
        span = self._anchor_map.get(index)
        if span is not None:
            return self._anchor_buf[span[0]:span[1]]
        span = self._active_slab_map.get(index)
        if span is not None:
            return self._slab_bufs[self._active_slab_slot][span[0]:span[1]]
        self.disk_reads += 1
        self.disk_read_bytes += self.source_bytes[index]
        if h5_group is None:
            with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
                return hf["embeddings"][self.headers[index]][:]
        return h5_group[self.headers[index]][:]

    def load_indices(self, indices, h5_group=None):
        return {int(index): self.get(index, h5_group) for index in sorted(set(indices))}

    def read_into(self, index, out, h5_group=None):
        """Read embedding ``index`` into ``out``, a C-contiguous array of its stored dtype and shape.

        Reading into a buffer the caller owns (a pinned one, say) saves the array
        ``get`` allocates for every embedding.
        """
        index = int(index)
        if self._packed is not None:
            start, end = self._offsets[index:index + 2]
            out[...] = self._packed[int(start):int(end)]
            return out
        span = self._anchor_map.get(index)
        if span is not None:
            out[...] = self._anchor_buf[span[0]:span[1]]
            return out
        span = self._active_slab_map.get(index)
        if span is not None:
            out[...] = self._slab_bufs[self._active_slab_slot][span[0]:span[1]]
            return out
        if h5_group is not None:
            # h5py's low-level calls: the Dataset wrapper and the selections of read_direct
            # cost about as much as the read itself (91 us against 194 us per 0.5 MB embedding).
            dataset = h5py.h5d.open(h5_group.id, self.headers[index].encode("utf-8"))
            dataset.read(h5py.h5s.ALL, h5py.h5s.ALL, out, h5py.h5t.py_create(out.dtype))
            self.disk_reads += 1
            self.disk_read_bytes += self.source_bytes[index]
        else:
            with h5py.File(self.path, "r", libver="latest", swmr=True) as hf:
                return self.read_into(index, out, hf["embeddings"])
        return out

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


def plan_windowed_budgets(
    store,
    *,
    device=None,
    compute_element_bytes=4,
    num_tasks=None,
    batch_size=None,
):
    """Return ``(anchor_bytes, slab_bytes)`` for partial-RAM windowing, or ``None``.

    Activates only when ``store`` has a positive host-cache budget that cannot
    hold the whole embedding file, has uniform dtype, and can fit at least one
    anchor plus two double-buffered streaming slabs (aligned to GPU half-tile
    byte size when an accelerator is available). Skips windowing for unified-
    memory devices (such as Apple Silicon MPS) so host RAM is not double-booked
    alongside GPU tiles, and for single-batch runs whose entire compute working
    set already fits in the GPU tile cache.
    """
    if not isinstance(store, EmbeddingTileStore) or not store.supports_windowed_cache:
        return None
    budget = int(store.host_cache_bytes)
    if budget <= 0 or store.total_source_bytes <= budget:
        return None
    max_seq_bytes = max(store.source_bytes) if store.source_bytes else 0
    if max_seq_bytes <= 0 or budget < max_seq_bytes * 3:
        return None

    src_itemsize = max(1, int(store.uniform_dtype.itemsize))
    elem_bytes = max(1, int(compute_element_bytes))
    gpu_half_src = None
    if device is not None:
        dev_type = str(
            getattr(device, "type", str(device).split(":", 1)[0])
        ).strip().lower()
        if dev_type == "mps":
            return None
        if dev_type in SUPPORTED_TILED_BACKENDS:
            try:
                plan = accelerator_memory_plan(
                    device, lanes=2, compute_element_bytes=elem_bytes
                )
                if getattr(plan, "unified_memory", False) is True:
                    return None
                total_compute_bytes = int(sum(store.compute_bytes(elem_bytes)))
                if (
                    num_tasks is not None
                    and batch_size is not None
                    and int(num_tasks) <= int(batch_size)
                    and total_compute_bytes <= int(plan.tile_cache_bytes)
                ):
                    return None
                gpu_half_src = max(
                    max_seq_bytes,
                    int((plan.tile_cache_bytes // 2) * src_itemsize // elem_bytes),
                )
            except Exception:
                gpu_half_src = None
    if gpu_half_src is None:
        gpu_half_src = max(max_seq_bytes, budget // 5)

    slab_bytes = min(gpu_half_src, max(max_seq_bytes, int(budget * 0.15)))
    anchor_bytes = budget - 2 * slab_bytes
    if anchor_bytes < max_seq_bytes:
        slab_bytes = max(max_seq_bytes, (budget - max_seq_bytes) // 2)
        anchor_bytes = budget - 2 * slab_bytes
    if anchor_bytes < max_seq_bytes or slab_bytes < max_seq_bytes:
        return None
    return int(anchor_bytes), int(slab_bytes)


def _byte_partition_indices(indices, source_bytes, max_bytes):
    """Group ``indices`` in given order into contiguous byte-bounded chunks."""
    max_bytes = max(1, int(max_bytes))
    groups = []
    cur = []
    used = 0
    for idx in indices:
        idx = int(idx)
        size = int(source_bytes[idx])
        if cur and used + size > max_bytes:
            groups.append(cur)
            cur = []
            used = 0
        cur.append(idx)
        used += size
    if cur:
        groups.append(cur)
    return groups


def _rows_for_indices(store, indices):
    return sum(int(store.shapes[int(idx)][0]) for idx in indices)


class WindowedPairSession:
    """Two-level partial-RAM (anchor + double-buffered slab) batch producer.

    Supports both mask-based all-vs-all / sparse prefiltered runs
    (``from_masks``) and sorted ``int64`` key runs (``from_pending_keys``,
    e.g. ``Network_Injection``). Every emitted batch belongs strictly to
    ``anchor_p x anchor_p`` or ``anchor_p x slab_q`` (with small diagonals
    optionally combined with ``slab_0``), guaranteeing 100% RAM cache hits
    during GPU execution while the next slab prefetches asynchronously.
    """

    def __init__(
        self,
        store: EmbeddingTileStore,
        safe_headers,
        *,
        anchor_bytes: int,
        slab_bytes: int,
        batch_size: int,
        mode: str,
        computed_mask=None,
        required_mask=None,
        pending_keys=None,
    ):
        self.store = store
        self.safe_headers = list(safe_headers)
        self.anchor_bytes = max(1, int(anchor_bytes))
        self.slab_bytes = max(1, int(slab_bytes))
        self.batch_size = max(1, int(batch_size))
        self.mode = mode
        self._computed_mask = computed_mask
        self._required_mask = required_mask
        self._pending_keys = pending_keys
        self._window_enabled = False
        self._closed = False
        self.anchors_loaded = 0
        self.slabs_loaded = 0
        self.slabs_skipped = 0
        self.batches_emitted = 0

        if mode == "masks":
            self._prepare_mask_schedule()
        elif mode == "keys":
            self._prepare_key_schedule()
        else:
            raise ValueError(f"Unknown WindowedPairSession mode: {mode!r}")

    @classmethod
    def from_masks(
        cls,
        store: EmbeddingTileStore,
        safe_headers,
        computed_mask,
        required_mask,
        *,
        anchor_bytes: int,
        slab_bytes: int,
        batch_size: int,
    ):
        return cls(
            store,
            safe_headers,
            anchor_bytes=anchor_bytes,
            slab_bytes=slab_bytes,
            batch_size=batch_size,
            mode="masks",
            computed_mask=computed_mask,
            required_mask=required_mask,
        )

    @classmethod
    def from_pending_keys(
        cls,
        store: EmbeddingTileStore,
        safe_headers,
        pending_keys,
        *,
        anchor_bytes: int,
        slab_bytes: int,
        batch_size: int,
    ):
        return cls(
            store,
            safe_headers,
            anchor_bytes=anchor_bytes,
            slab_bytes=slab_bytes,
            batch_size=batch_size,
            mode="keys",
            pending_keys=np.asarray(pending_keys, dtype=np.int64),
        )

    def _prepare_mask_schedule(self):
        n = len(self.safe_headers)
        # Partition 0..n-1 into contiguous slab-sized GPU blocks, then group into RAM anchors.
        self._gpu_blocks = _byte_partition_indices(
            range(n), self.store.source_bytes, self.slab_bytes
        )
        block_bytes = [
            sum(int(self.store.source_bytes[i]) for i in blk) for blk in self._gpu_blocks
        ]
        self._anchors = []  # list of lists of gpu_block indices
        cur_blocks = []
        used = 0
        for b_idx, b_sz in enumerate(block_bytes):
            if cur_blocks and used + b_sz > self.anchor_bytes:
                self._anchors.append(cur_blocks)
                cur_blocks = []
                used = 0
            cur_blocks.append(b_idx)
            used += b_sz
        if cur_blocks:
            self._anchors.append(cur_blocks)

        self._anchor_rows_cap = max(
            (
                sum(_rows_for_indices(self.store, self._gpu_blocks[b]) for b in ab)
                for ab in self._anchors
            ),
            default=0,
        )
        # Packed sparse slabs never exceed slab_bytes; cap slab rows to max possible rows in slab_bytes.
        max_blk_rows = max(
            (_rows_for_indices(self.store, blk) for blk in self._gpu_blocks),
            default=0,
        )
        row_bytes = max(
            1,
            int(self.store.feature_dimension or 1)
            * max(1, int(self.store.uniform_dtype.itemsize if self.store.uniform_dtype is not None else 4)),
        )
        budget_slab_rows = self.slab_bytes // row_bytes
        self._slab_rows_cap = (
            max(max_blk_rows, budget_slab_rows) if len(self._anchors) > 1 else 0
        )

    def _prepare_key_schedule(self):
        keys = self._pending_keys
        n = len(self.safe_headers)
        self._key_stages = []
        if keys.size == 0 or n == 0:
            self._pending_keys = None
            self._anchor_rows_cap = 0
            self._slab_rows_cap = 0
            return

        u = keys // n
        v = keys % n
        deg = np.bincount(u, minlength=n).astype(np.int64)
        deg += np.bincount(v, minlength=n).astype(np.int64)
        active = np.flatnonzero(deg > 0)
        # Put high-degree hubs (e.g., new sequences in Network Injection) into Anchor 0
        # while keeping remaining sequences in ascending index order for sequential disk I/O
        # and tight intra-block locality.
        max_deg = int(deg[active].max()) if active.size else 0
        hub_cut = max(2, int(max_deg * 0.35))
        order = sorted(
            active.tolist(),
            key=lambda idx: (0 if int(deg[idx]) >= hub_cut else 1, -int(deg[idx]) if int(deg[idx]) >= hub_cut else idx),
        )
        del deg, active
        anchors = _byte_partition_indices(order, self.store.source_bytes, self.anchor_bytes)
        del order

        anchor_of = np.full(n, -1, dtype=np.int32)
        for a_idx, seqs in enumerate(anchors):
            anchor_of[seqs] = a_idx

        au = anchor_of[u]
        av = anchor_of[v]
        del u, v
        a_lo = np.minimum(au, av)
        a_hi = np.maximum(au, av)
        del au, av

        # Group all pending keys by their lower anchor in one O(E log E) sort.
        sort_idx = np.argsort(a_lo, kind="stable")
        keys_sorted = keys[sort_idx]
        self._pending_keys = None
        del keys
        a_lo_s = a_lo[sort_idx]
        del a_lo
        a_hi_s = a_hi[sort_idx]
        del a_hi, sort_idx

        starts = np.searchsorted(a_lo_s, np.arange(len(anchors)), side="left")
        ends = np.searchsorted(a_lo_s, np.arange(len(anchors)), side="right")
        del a_lo_s

        max_anchor_rows = 0
        max_slab_rows = 0
        for a_idx, (s_pos, e_pos) in enumerate(zip(starts.tolist(), ends.tolist())):
            if s_pos >= e_pos:
                continue
            stage_keys = keys_sorted[s_pos:e_pos]
            stage_hi = a_hi_s[s_pos:e_pos]
            diag_mask = stage_hi == a_idx
            diag_keys = stage_keys[diag_mask]
            ext_keys = stage_keys[~diag_mask]

            # Collect only sequences of this anchor that participate in this stage.
            su = stage_keys // n
            sv = stage_keys % n
            in_anchor = np.concatenate((su[anchor_of[su] == a_idx], sv[anchor_of[sv] == a_idx]))
            active_anchor = np.unique(in_anchor).tolist()
            del su, sv, in_anchor
            max_anchor_rows = max(max_anchor_rows, _rows_for_indices(self.store, active_anchor))

            # Order diagonal keys cleanly by (u, v).
            if diag_keys.size > 1:
                diag_keys = np.sort(diag_keys)

            # Group external keys by byte-bounded slabs of their external endpoints.
            slabs = []
            if ext_keys.size > 0:
                eu = ext_keys // n
                ev = ext_keys % n
                ext_end = np.where(anchor_of[eu] == a_idx, ev, eu)
                del eu, ev
                unique_ext = np.unique(ext_end).tolist()
                slab_groups = _byte_partition_indices(
                    unique_ext, self.store.source_bytes, self.slab_bytes
                )
                slab_of = np.full(n, -1, dtype=np.int32)
                for sl_idx, sl_seqs in enumerate(slab_groups):
                    slab_of[sl_seqs] = sl_idx
                    max_slab_rows = max(max_slab_rows, _rows_for_indices(self.store, sl_seqs))
                ext_slab_ids = slab_of[ext_end]
                del ext_end, slab_of
                sl_order = np.argsort(ext_slab_ids, kind="stable")
                ext_sorted = ext_keys[sl_order]
                ext_sid_sorted = ext_slab_ids[sl_order]
                del ext_slab_ids, sl_order, ext_keys
                sl_starts = np.searchsorted(ext_sid_sorted, np.arange(len(slab_groups)), side="left")
                sl_ends = np.searchsorted(ext_sid_sorted, np.arange(len(slab_groups)), side="right")
                del ext_sid_sorted
                for sl_seqs, sl_s, sl_e in zip(slab_groups, sl_starts.tolist(), sl_ends.tolist()):
                    if sl_s < sl_e:
                        slabs.append((sl_seqs, ext_sorted[sl_s:sl_e]))

            self._key_stages.append((active_anchor, diag_keys, slabs))

        del keys_sorted, a_hi_s, anchor_of
        self._anchor_rows_cap = max_anchor_rows
        self._slab_rows_cap = max_slab_rows

    def __enter__(self):
        if self._anchor_rows_cap > 0 and self.store.supports_windowed_cache:
            self._window_enabled = self.store.open_window(
                self._anchor_rows_cap, self._slab_rows_cap
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def release_window(self):
        """Release RAM buffers (e.g., on CPU fallback) while keeping batch iteration active."""
        if self._window_enabled:
            self._window_enabled = False
            self.store.clear_window()

    def close(self):
        if not self._closed:
            self._closed = True
            self._window_enabled = False
            self.store.clear_window()

    def _chunk_pair_tuples(self, pair_iter):
        batch = []
        limit = self.batch_size
        for item in pair_iter:
            batch.append(item)
            if len(batch) >= limit:
                self.batches_emitted += 1
                yield batch
                batch = []
        if batch:
            self.batches_emitted += 1
            yield batch

    def _keys_to_tuples(self, keys):
        headers = self.safe_headers
        n = len(headers)
        for start in range(0, keys.size, 65536):
            chunk = keys[start:start + 65536]
            for r, c in zip((chunk // n).tolist(), (chunk % n).tolist()):
                yield (r, c, headers[r], headers[c])

    def _iter_mask_diag_tuples(self, anchor_blocks):
        headers = self.safe_headers
        comp = self._computed_mask
        req = self._required_mask
        for idx_a, b_r in enumerate(anchor_blocks):
            rows_r = self._gpu_blocks[b_r]
            r0, r1 = rows_r[0], rows_r[-1] + 1
            for b_c in anchor_blocks[idx_a:]:
                cols_c = self._gpu_blocks[b_c]
                c0, c1 = cols_c[0], cols_c[-1] + 1
                sub = ~comp[r0:r1, c0:c1]
                if req is not None:
                    sub &= req[r0:r1, c0:c1]
                if b_r == b_c:
                    sub = np.triu(sub, k=1)
                rr, cc = np.nonzero(sub)
                if rr.size == 0:
                    continue
                for r, c in zip((rr + r0).tolist(), (cc + c0).tolist()):
                    yield (r, c, headers[r], headers[c])

    def _iter_mask_slab_tuples(self, anchor_blocks, packed_slab_cols):
        headers = self.safe_headers
        comp = self._computed_mask
        req = self._required_mask
        cols_arr = np.asarray(packed_slab_cols, dtype=np.int64)
        for b_r in anchor_blocks:
            rows_r = self._gpu_blocks[b_r]
            r0, r1 = rows_r[0], rows_r[-1] + 1
            sub = ~comp[r0:r1, cols_arr]
            if req is not None:
                sub &= req[r0:r1, cols_arr]
            rr, cc = np.nonzero(sub)
            if rr.size == 0:
                continue
            for r, c in zip((rr + r0).tolist(), cols_arr[cc].tolist()):
                yield (r, c, headers[r], headers[c])

    def _iter_mask_batches(self):
        comp = self._computed_mask
        req = self._required_mask
        n = len(self.safe_headers)
        src_bytes = self.store.source_bytes
        merge_diag_threshold = min(self.batch_size, 16384)

        for a_idx, anchor_blocks in enumerate(self._anchors):
            r0 = self._gpu_blocks[anchor_blocks[0]][0]
            r1 = self._gpu_blocks[anchor_blocks[-1]][-1] + 1

            # Determine active sequences in this anchor and total diagonal pairs.
            row_active = np.zeros(r1 - r0, dtype=bool)
            diag_count = 0
            # Diagonal scan
            diag_sub = ~comp[r0:r1, r0:r1]
            if req is not None:
                diag_sub &= req[r0:r1, r0:r1]
            diag_sub = np.triu(diag_sub, k=1)
            diag_count = int(np.count_nonzero(diag_sub))
            if diag_count > 0:
                row_active |= diag_sub.any(axis=1)
                row_active |= diag_sub.any(axis=0)
            del diag_sub

            # Off-diagonal scan across subsequent GPU blocks; pack sparse active columns up to slab_bytes.
            packed_slabs = []
            cur_cols = []
            cur_bytes = 0
            last_blk = anchor_blocks[-1]
            for b_q in range(last_blk + 1, len(self._gpu_blocks)):
                cols_q = self._gpu_blocks[b_q]
                c0, c1 = cols_q[0], cols_q[-1] + 1
                sub = ~comp[r0:r1, c0:c1]
                if req is not None:
                    sub &= req[r0:r1, c0:c1]
                col_any = sub.any(axis=0)
                if not col_any.any():
                    self.slabs_skipped += 1
                    continue
                row_active |= sub.any(axis=1)
                active_cols = (np.flatnonzero(col_any) + c0).tolist()
                for col in active_cols:
                    sz = int(src_bytes[col])
                    if cur_cols and cur_bytes + sz > self.slab_bytes:
                        packed_slabs.append(cur_cols)
                        cur_cols = []
                        cur_bytes = 0
                    cur_cols.append(col)
                    cur_bytes += sz
            if cur_cols:
                packed_slabs.append(cur_cols)

            if diag_count == 0 and not packed_slabs:
                continue

            active_anchor = (np.flatnonzero(row_active) + r0).tolist()
            if self._window_enabled and not self.store.window_active:
                self._window_enabled = False
            if self._window_enabled:
                self.store.load_anchor(active_anchor)
                self.anchors_loaded += 1
                if packed_slabs:
                    self.store.prefetch_slab(packed_slabs[0])

            # Decide whether to run diagonal standalone (overlapping slab_0 prefetch) or combine with slab_0.
            start_slab = 0
            if diag_count > 0 and (not packed_slabs or diag_count >= merge_diag_threshold):
                yield from self._chunk_pair_tuples(self._iter_mask_diag_tuples(anchor_blocks))
            elif diag_count > 0 and packed_slabs:
                if self._window_enabled and not self.store.window_active:
                    self._window_enabled = False
                if self._window_enabled:
                    self.store.activate_prefetched_slab()
                    self.slabs_loaded += 1
                    if len(packed_slabs) > 1:
                        self.store.prefetch_slab(packed_slabs[1])
                yield from self._chunk_pair_tuples(
                    chain(
                        self._iter_mask_diag_tuples(anchor_blocks),
                        self._iter_mask_slab_tuples(anchor_blocks, packed_slabs[0]),
                    )
                )
                start_slab = 1

            for k in range(start_slab, len(packed_slabs)):
                if self._window_enabled and not self.store.window_active:
                    self._window_enabled = False
                if self._window_enabled:
                    self.store.activate_prefetched_slab()
                    self.slabs_loaded += 1
                    if k + 1 < len(packed_slabs):
                        self.store.prefetch_slab(packed_slabs[k + 1])
                yield from self._chunk_pair_tuples(
                    self._iter_mask_slab_tuples(anchor_blocks, packed_slabs[k])
                )

    def _iter_key_batches(self):
        merge_diag_threshold = min(self.batch_size, 16384)
        for active_anchor, diag_keys, slabs in self._key_stages:
            if self._window_enabled and not self.store.window_active:
                self._window_enabled = False
            if self._window_enabled:
                self.store.load_anchor(active_anchor)
                self.anchors_loaded += 1
                if slabs:
                    self.store.prefetch_slab(slabs[0][0])

            diag_count = int(diag_keys.size)
            start_slab = 0
            if diag_count > 0 and (not slabs or diag_count >= merge_diag_threshold):
                yield from self._chunk_pair_tuples(self._keys_to_tuples(diag_keys))
            elif diag_count > 0 and slabs:
                if self._window_enabled and not self.store.window_active:
                    self._window_enabled = False
                if self._window_enabled:
                    self.store.activate_prefetched_slab()
                    self.slabs_loaded += 1
                    if len(slabs) > 1:
                        self.store.prefetch_slab(slabs[1][0])
                yield from self._chunk_pair_tuples(
                    chain(self._keys_to_tuples(diag_keys), self._keys_to_tuples(slabs[0][1]))
                )
                start_slab = 1

            for k in range(start_slab, len(slabs)):
                if self._window_enabled and not self.store.window_active:
                    self._window_enabled = False
                if self._window_enabled:
                    self.store.activate_prefetched_slab()
                    self.slabs_loaded += 1
                    if k + 1 < len(slabs):
                        self.store.prefetch_slab(slabs[k + 1][0])
                yield from self._chunk_pair_tuples(self._keys_to_tuples(slabs[k][1]))

    def iter_batches(self):
        if self.mode == "masks":
            yield from self._iter_mask_batches()
        else:
            yield from self._iter_key_batches()


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
    use_pinned_copy = _pins_host_memory(device_type)
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


def _int64_to_device(values, device):
    """Copy small integer metadata to ``device`` without a stream sync.

    A pageable host-to-device copy makes the host wait until the current
    stream is idle, which serialized the tiled producer once per microbatch.
    Pinned memory makes the copy asynchronous on CUDA and XPU.
    """
    host = torch.as_tensor(values, dtype=torch.int64)
    device = torch.device(device)
    if _pins_host_memory(device.type):
        try:
            host = host.pin_memory()
        except RuntimeError:
            return host.to(device)
        return host.to(device, non_blocking=True)
    return host.to(device)


def _gather_padded_targets(packed, target_starts, target_lengths, pad_row):
    """Gather zero-padded ``(batch, max_length, features)`` targets in one
    operation from a packed tile whose row ``pad_row`` is all zeros."""
    batch = len(target_lengths)
    max_length = max(int(length) for length in target_lengths)
    metadata = _int64_to_device([target_starts, target_lengths], packed.device)
    positions = torch.arange(max_length, device=packed.device)
    valid = positions.unsqueeze(0) < metadata[1].unsqueeze(1)
    rows = torch.where(
        valid, metadata[0].unsqueeze(1) + positions.unsqueeze(0), pad_row
    )
    return packed.index_select(0, rows.reshape(-1)).view(
        batch, max_length, packed.shape[1]
    )


def _copy_tile_row(packed, start, length):
    """Copy one row embedding out of the packed tile into its own tensor.

    The copy runs on the current lane's stream; other lanes must wait for it.
    A separate tensor matches the per-sequence embeddings earlier versions
    passed to the matrix multiply.
    """
    return packed[start:start + length].clone()


def _batched_score_matrices(row_tensor, target_tensors, target_lengths):
    """Compute padded score matrices while excluding padding from statistics.

    ``target_tensors`` is either a list of per-target embeddings, padded here
    one copy per target, or an already padded ``(batch, max_length,
    features)`` tensor such as ``_gather_padded_targets`` returns.
    """
    if torch.is_tensor(target_tensors):
        targets = target_tensors
        max_length = int(targets.shape[1])
    else:
        batch_size = len(target_tensors)
        max_length = max(int(length) for length in target_lengths)
        feature_dimension = int(row_tensor.shape[1])
        targets = torch.zeros(
            (batch_size, max_length, feature_dimension),
            dtype=row_tensor.dtype,
            device=row_tensor.device,
        )
        for index, (target, length) in enumerate(
            zip(target_tensors, target_lengths)
        ):
            targets[index, :int(length)].copy_(target)

    cosine = torch.matmul(
        row_tensor.unsqueeze(0), targets.transpose(1, 2)
    ).to(torch.float32)
    similarity = torch.exp(-(1.0 - cosine.clamp_(-1.0, 1.0)))
    lengths_tensor = _int64_to_device(target_lengths, row_tensor.device)
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


_PIN_BACKEND_USABLE = {}


def _pins_host_memory(device):
    """True only for CUDA or XPU that this process can really use.

    Where PyTorch was built with MPS but cannot use it, asking for pinned memory
    crashes the process outright, which no except can catch; so the device type
    alone is not enough. The availability of each backend is asked once.
    """
    backend = getattr(device, "type", str(device).split(":", 1)[0])
    if backend not in {"cuda", "xpu"}:
        return False
    if backend not in _PIN_BACKEND_USABLE:
        module = getattr(torch, backend, None)
        try:
            _PIN_BACKEND_USABLE[backend] = bool(module is not None and module.is_available())
        except Exception:
            _PIN_BACKEND_USABLE[backend] = False
    return _PIN_BACKEND_USABLE[backend]


def _host_output_buffer(tensor):
    if _pins_host_memory(tensor.device):
        try:
            return torch.empty(tensor.shape, dtype=torch.float32, device="cpu", pin_memory=True)
        except RuntimeError:
            pass
    return torch.empty(tensor.shape, dtype=torch.float32, device="cpu")


class _PinnedHostBufferPool:
    """Reuse pinned float32 staging buffers across microbatches to avoid per-batch cudaHostAlloc."""

    def __init__(self, device, max_buffers=16):
        self._pin = _pins_host_memory(device)
        self._max_buffers = max(1, int(max_buffers))
        self._pool = []

    def acquire(self, shape):
        need = int(np.prod(shape, dtype=np.int64))
        best_idx = -1
        best_cap = None
        for idx, buf in enumerate(self._pool):
            cap = buf.numel()
            if cap >= need and (best_cap is None or cap < best_cap):
                best_idx = idx
                best_cap = cap
                if cap <= need * 2:
                    break
        if best_idx >= 0:
            buf = self._pool.pop(best_idx)
            return buf, buf[:need].view(shape)
        if self._pin:
            try:
                buf = torch.empty(need, dtype=torch.float32, device="cpu", pin_memory=True)
                return buf, buf.view(shape)
            except RuntimeError:
                self._pin = False
        buf = torch.empty(need, dtype=torch.float32, device="cpu")
        return buf, buf.view(shape)

    def release(self, buf):
        if buf is None:
            return
        if len(self._pool) < self._max_buffers:
            self._pool.append(buf)

    def clear(self):
        self._pool.clear()


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
    fused_gpu=False,
):
    """Build, deduplicate, and Pareto-prune safe tile-plan candidates."""
    tasks = list(tasks)
    candidates = []
    seen = set()
    normalized_lanes = sorted(set(max(1, int(value)) for value in lane_candidates))
    if fused_gpu:
        # Fused GPU alignment executes end-to-end on GPU with 1-2 async streams and fixed
        # block-GEMM workspaces; testing 3 memory profiles x 5 lanes adds pure startup overhead.
        normalized_lanes = [l for l in normalized_lanes if l <= 2] or [normalized_lanes[0]]
        profiles = (("balanced", 0.30, 0.50),)
    else:
        profiles = TILE_MEMORY_PROFILES

    for lanes in normalized_lanes:
        for profile_name, tile_fraction, matrix_fraction in profiles:
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
            if fused_gpu:
                baseline_bytes = max(0, plan.total_bytes - plan.free_bytes)
                safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
                resident_bytes = _store_resident_bytes(store, lengths, compute_element_bytes)
                per_block = max(1, plan.tile_cache_bytes // 2)
                block_ids = _store_block_ids(store, per_block, lengths, compute_element_bytes)
                tile_bytes = 0
                reload_bytes = 0
                max_len = 1
                for tile_tasks in _partition_tiles(tasks, block_ids):
                    tile_indices = {int(t[0]) for t in tile_tasks} | {int(t[1]) for t in tile_tasks}
                    if tile_indices:
                        resident = sum(resident_bytes[i] for i in tile_indices)
                        tile_bytes = max(tile_bytes, resident)
                        reload_bytes += resident
                        max_len = max(max_len, max(int(lengths[i]) for i in tile_indices))
                # The planner inspects only the first pending batch, where a cross-block tile
                # may touch only a partial slice of block 0. Later batches can load two full
                # blocks, so size the peak/cap for the worst-case tile across all batches.
                if len(set(block_ids.tolist())) > 1:
                    tile_bytes = max(
                        tile_bytes,
                        min(int(sum(resident_bytes)), int(plan.tile_cache_bytes)),
                    )
                n_streams = max(1, min(2, int(lanes)))
                blk_side = Align_GPU_Kernels._choose_block_residues(
                    plan.usable_bytes, n_streams, max_len
                )
                # Per stream: blk_side^2 float32 cosine mat + stats/DP scratch (~128 MiB)
                # plus cuBLAS workspace and chunked-upload staging (~64 MiB) slack so the
                # allocator cap never causes false OOM retries across batches.
                per_slot = int(blk_side) * int(blk_side) * 4 + 128 * 1024 * 1024
                transient_bytes = per_slot * n_streams + 256 * 1024 * 1024
                additional = tile_bytes + transient_bytes
                projected_peak = baseline_bytes + additional
                feasible = tile_bytes <= plan.tile_cache_bytes and projected_peak <= safe_peak
                reason = (
                    "within reserved-VRAM boundary"
                    if feasible
                    else (
                        "one embedding block pair exceeds its tile-cache budget"
                        if tile_bytes > plan.tile_cache_bytes
                        else "projected peak exceeds the reserved-VRAM boundary"
                    )
                )
                estimate = CudaWorkloadEstimate(
                    variant="tiled",
                    lanes=int(lanes),
                    inflight_slots=n_streams,
                    tile_bytes=int(tile_bytes),
                    transient_bytes=int(transient_bytes),
                    additional_bytes=int(additional),
                    projected_peak_bytes=int(projected_peak),
                    safe_peak_bytes=int(safe_peak),
                    per_microbatch_bytes=int(per_slot),
                    largest_microbatch_bytes=int(per_slot),
                    feasible=bool(feasible),
                    reason=reason,
                    embedding_reload_bytes=int(reload_bytes),
                    microbatch_count=1,
                    padded_elements=0,
                    real_elements=0,
                    schedule_signature=(tuple(block_ids.tolist()), ("fused", n_streams, blk_side)),
                )
            else:
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
            # What this plan was dry-run to need, for the allocator cap while it runs.
            try:
                plan = replace(
                    plan,
                    allocator_cap_bytes=int(estimate.additional_bytes) + ALLOCATOR_CAP_SLACK_BYTES,
                )
            except (TypeError, ValueError):
                pass  # a plan that is not an AcceleratorMemoryPlan keeps the cap derived from its budgets
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


def _open_store_h5_if_needed(store, tasks):
    """Return an HDF5 handle context manager only when ``tasks`` needs disk reads."""
    if not tasks:
        return nullcontext(None)
    all_in_ram = getattr(store, "fully_cached", False) is True
    if not all_in_ram and callable(getattr(store, "all_cached", None)):
        all_task_indices = {int(task[0]) for task in tasks}
        all_task_indices.update(int(task[1]) for task in tasks)
        all_in_ram = store.all_cached(all_task_indices) is True
    if all_in_ram:
        return nullcontext(None)
    return h5py.File(store.path, "r", libver="latest", swmr=True)


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
            _open_store_h5_if_needed(store, tasks) as hf, \
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="alignment-cpu",
            ) as cpu_executor:
        group = None if hf is None else hf["embeddings"]

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
        else:
            run_phase(tasks)

    if result_callback is not None and results:
        result_callback(results)
        results.clear()
    return results


def _stream_wait(stream, event):
    """Order ``stream`` after ``event`` on the device; wait on the host when the
    stream object cannot express a device-side dependency."""
    wait_event = getattr(stream, "wait_event", None)
    if callable(wait_event):
        wait_event(event)
    else:
        event.synchronize()


def _tile_layout(tile_tasks, lengths):
    """Return sorted tile indices, their packed start rows, and total rows."""
    indices = sorted(
        {int(task[0]) for task in tile_tasks}
        | {int(task[1]) for task in tile_tasks}
    )
    starts = {}
    rows = 0
    for index in indices:
        starts[index] = rows
        rows += int(lengths[index])
    return indices, starts, rows


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
    benchmark_trial=None,
    batch_alignment_callback=None,
    gpu_dp_gaps=None,
):
    """Run the tiled producer and completion-driven CPU consumers.

    When ``gpu_dp_gaps=(global_gap, local_gap)`` is supplied on a CUDA/ROCm device
    whose runtime alignment kernels pass self-check, the entire cosine GEMM, exact
    two-pass Z-score normalization, and global+local DP run fused on the GPU with
    zero score-matrix PCIe transfer.

    Otherwise each tile's embeddings are uploaded once into a packed device buffer, and
    every microbatch gathers its zero-padded targets from it in one
    operation. With ``batch_alignment_callback`` one CPU task aligns a whole
    microbatch, ``(row, targets, target_lengths, matrices) -> [result, ...]``;
    otherwise ``alignment_callback`` runs once per pair. Tile changes are
    ordered on the device with events, so the host never drains the queue
    between tiles.
    """
    backend = get_accelerator_backend(device)
    supported, reason = backend.supports_tiled(require_memory=True)
    if not supported:
        raise RuntimeError(
            f"Tiled execution is unavailable on '{device}': {reason}."
        )
    tasks = list(tasks)
    if not tasks:
        return []

    if (
        gpu_dp_gaps is not None
        and backend.device_type == "cuda"
        and Align_GPU_Kernels.align_kernels(device) is not None
    ):
        global_gap, local_gap = gpu_dp_gaps
        return Align_GPU_Kernels.run_fused_gpu_alignment_pipeline(
            tasks,
            store=store,
            lengths=lengths,
            device=device,
            lanes=lanes,
            global_gap=global_gap,
            local_gap=local_gap,
            precision=precision,
            progress=progress,
            result_callback=result_callback,
            result_chunk_size=result_chunk_size,
            matrix_budget_override=matrix_budget_override,
            memory_plan_override=memory_plan_override,
            benchmark_trial=benchmark_trial,
            cuda_matmul_precision_ctx=cuda_matmul_precision,
            precision_compute_dtype_fn=precision_compute_dtype,
            precision_element_bytes_fn=precision_element_bytes,
            accelerator_memory_plan_fn=accelerator_memory_plan,
            allocator_cap_ctx=allocator_cap,
            clear_failed_frames_fn=_clear_failed_frames,
            partition_tiles_fn=_partition_tiles,
            backend=backend,
        )

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
            benchmark_trial=benchmark_trial,
        )
    block_ids = store.block_ids(per_block, element_bytes=element_bytes)
    feature_dimension = int(store.feature_dimension)
    all_tiles = [list(tile) for tile in _partition_tiles(tasks, block_ids)]
    all_layouts = [_tile_layout(tile, lengths) for tile in all_tiles]
    # Any subset of the tasks (benchmark warm-up) yields tiles whose index
    # sets are subsets of these, so one buffer of this size serves them all.
    pad_row = max(layout[2] for layout in all_layouts)
    packed = None

    results = []
    active_trial = None
    cpu_pending = set()
    future_buffers = {}
    inflight = deque()
    max_inflight = max(2, int(lanes) * 2)
    batch_pending_limit = max(1, int(workers)) + max(1, int(lanes))
    host_pool = _PinnedHostBufferPool(device, max_buffers=max_inflight + batch_pending_limit + 2)
    streams = [backend.create_stream() for _ in range(max(1, int(lanes)))]
    stream_cursor = 0
    cap_stack = ExitStack()
    cap_stack.enter_context(allocator_cap(backend, plan))

    def collect_cpu(block=False):
        if block and cpu_pending:
            completed, _ = wait(cpu_pending, return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in cpu_pending if future.done()}
        completed_count = 0
        for future in completed:
            cpu_pending.remove(future)
            pooled_buf = future_buffers.pop(future, None)
            try:
                if batch_alignment_callback is None:
                    results.append(future.result())
                    completed_count += 1
                else:
                    batch_results = future.result()
                    results.extend(batch_results)
                    completed_count += len(batch_results)
            finally:
                if pooled_buf is not None:
                    host_pool.release(pooled_buf)
        if progress is not None and completed_count:
            progress.update(completed_count)
        if active_trial is not None:
            active_trial.completed += completed_count
        if result_callback is not None and len(results) >= int(result_chunk_size):
            result_callback(results)
            results.clear()

    def submit_oldest(block):
        if not inflight:
            return
        event, pooled_buf, host_tensor, idx_i, target_indices, target_lengths, _row = inflight[0]
        if not block and not event.query():
            return
        if block:
            event.synchronize()
        # Dropping the entry also releases its row tensor, which is safe now
        # that the event (and every earlier entry's) has completed.
        inflight.popleft()
        matrix_array = host_tensor.numpy()
        if batch_alignment_callback is not None:
            future = cpu_executor.submit(
                batch_alignment_callback,
                (idx_i, target_indices, target_lengths, matrix_array),
            )
            future_buffers[future] = pooled_buf
            cpu_pending.add(future)
            while len(cpu_pending) >= batch_pending_limit:
                collect_cpu(block=True)
            return
        for offset, (idx_j, length) in enumerate(
            zip(target_indices, target_lengths)
        ):
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

    def upload_tile(layout, group):
        nonlocal packed
        indices, starts, _rows = layout
        host_embeddings = store.load_indices(indices, group)
        preload_stream = streams[0]
        # The packed buffer is reused: overwrite it only after every lane's
        # queued work on the previous tile, without blocking the host.
        for stream in streams[1:]:
            queued = backend.create_event()
            queued.record(stream)
            _stream_wait(preload_stream, queued)
        with backend.stream_context(preload_stream):
            for index in indices:
                normalized = _to_normalized_accelerator(
                    host_embeddings[index], device, precision
                )
                if packed is None:
                    # Allocated like the embeddings themselves; the extra
                    # last row stays zero and pads every gathered batch.
                    packed = normalized.new_zeros(
                        (pad_row + 1, feature_dimension)
                    )
                start = starts[index]
                packed[start:start + normalized.shape[0]].copy_(normalized)
            uploaded = backend.create_event()
            uploaded.record(preload_stream)
        del host_embeddings
        for stream in streams:
            _stream_wait(stream, uploaded)
        return starts

    try:
        with cuda_matmul_precision(precision), torch.inference_mode(), \
                _open_store_h5_if_needed(store, tasks) as hf, \
                ThreadPoolExecutor(
                    max_workers=max(1, int(workers)),
                    thread_name_prefix="alignment-cpu",
                ) as cpu_executor:
            group = None if hf is None else hf["embeddings"]

            def run_phase(phase_tasks):
                nonlocal stream_cursor
                if phase_tasks is tasks:
                    phase_tiles, phase_layouts = all_tiles, all_layouts
                else:
                    phase_tiles = [
                        list(tile) for tile in _partition_tiles(phase_tasks, block_ids)
                    ]
                    phase_layouts = [_tile_layout(tile, lengths) for tile in phase_tiles]
                for tile_tasks, layout in zip(phase_tiles, phase_layouts):
                    if active_trial is not None:
                        if not active_trial.can_submit():
                            break
                        active_trial.tiles += 1
                    starts = upload_tile(layout, group)
                    row_tensor = None
                    row_index = None
                    row_ready = None
                    row_lanes = set()
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
                            feature_dimension,
                            element_bytes,
                        ):
                            if active_trial is not None:
                                if not active_trial.can_submit():
                                    break
                                active_trial.submitted += len(microbatch)
                                active_trial.microbatches += 1
                            stream = streams[stream_cursor % len(streams)]
                            stream_cursor += 1
                            target_indices = [int(task[1]) for task in microbatch]
                            target_lengths = [
                                int(lengths[index]) for index in target_indices
                            ]
                            with backend.stream_context(stream):
                                if row_index != idx_i:
                                    row_tensor = _copy_tile_row(
                                        packed, starts[idx_i], int(lengths[idx_i])
                                    )
                                    row_ready = backend.create_event()
                                    row_ready.record(stream)
                                    row_lanes = {id(stream)}
                                    row_index = idx_i
                                elif id(stream) not in row_lanes:
                                    # The copy was queued on another lane;
                                    # order this lane after it before any
                                    # kernel here reads the row.
                                    _stream_wait(stream, row_ready)
                                    row_lanes.add(id(stream))
                                targets = _gather_padded_targets(
                                    packed,
                                    [starts[index] for index in target_indices],
                                    target_lengths,
                                    pad_row,
                                )
                                matrices = _batched_score_matrices(
                                    row_tensor, targets, target_lengths
                                )
                                if batch_alignment_callback is not None:
                                    pooled_buf, host_tensor = host_pool.acquire(matrices.shape)
                                else:
                                    pooled_buf, host_tensor = None, _host_output_buffer(matrices)
                                host_tensor.copy_(
                                    matrices, non_blocking=host_tensor.is_pinned()
                                )
                                event = backend.create_event()
                                event.record(stream)
                            # The entry keeps the shared row tensor alive until
                            # this microbatch has finished on its stream.
                            inflight.append((
                                event,
                                pooled_buf,
                                host_tensor,
                                int(idx_i),
                                target_indices,
                                target_lengths,
                                row_tensor,
                            ))
                            while len(inflight) >= max_inflight:
                                submit_oldest(block=True)
                            submit_oldest(block=False)
                            collect_cpu(block=False)

                while inflight:
                    submit_oldest(block=True)
                while cpu_pending:
                    collect_cpu(block=True)

            if benchmark_trial is not None:
                run_phase(tasks[:benchmark_trial.warmup_count(tasks, workers, plan.lanes)])
                results.clear()
                active_trial = benchmark_trial
                benchmark_trial.start()
                run_phase(tasks)
                benchmark_trial.stop(len(tasks))
            else:
                run_phase(tasks)
    except BaseException as error:
        _clear_failed_frames(error)  # before the cache is emptied below; the error goes on unchanged
        raise
    finally:
        # Hand the cached blocks back to the driver, also after a failure:
        # callers check mem_get_info before the next batch or plan, and the
        # caching allocator's reserve would otherwise read as used memory.
        try:
            inflight.clear()
            future_buffers.clear()
            host_pool.clear()
            packed = None
            backend.empty_cache()
        finally:
            cap_stack.close()  # the allocator cap is lifted whatever happened

    if result_callback is not None and results:
        result_callback(results)
        results.clear()

    return results


def run_tiled_cuda_pipeline(*args, **kwargs):
    """Compatibility name for the CUDA/XPU tiled alignment pipeline."""
    return run_tiled_accelerator_pipeline(*args, **kwargs)


def _fixed_query_task_index(task):
    return int(task[0])


def fixed_query_block_ids(store, query_array, plan, element_bytes):
    """Tile block of every database embedding in a fixed-query tiled search.

    The one place that decides the blocks, so the search, its VRAM estimate
    and the hardware tuner's timing sample all see the same ones: each block
    is as many consecutive embeddings as fit beside the query in the tile cache.
    """
    query_bytes = int(np.asarray(query_array).size) * int(element_bytes)
    return store.block_ids(
        max(1, plan.tile_cache_bytes - query_bytes),
        element_bytes=element_bytes,
    )


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
    block_ids = fixed_query_block_ids(store, query, plan, element_bytes)
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
    # The device staging buffer one chunk of raw embeddings is copied into before it is normalized.
    source_bytes = getattr(store, "source_bytes", None)
    largest_source = (
        max((int(source_bytes[i]) for i in range(len(source_bytes))), default=0)
        if isinstance(source_bytes, (list, tuple, np.ndarray)) else 0
    )
    staging_bytes = max(FIXED_QUERY_CHUNK_BYTES, largest_source)
    additional = tile_bytes + transient + staging_bytes
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


# A fixed-query search reads its embeddings in chunks of about this many bytes into a ring of pinned
# buffers, at most FIXED_QUERY_RING_BYTES of them, while the device works on what it already has.
FIXED_QUERY_CHUNK_BYTES = 64 * 1024 ** 2
FIXED_QUERY_RING_BYTES = 256 * 1024 ** 2
FIXED_QUERY_ALIGNMENT = 16
# A small search is read in finer chunks, at least this small and about this many to a search, so that
# reading and copying up overlap (and its first chunk arrives soon) even when it is a few hundred megabytes.
FIXED_QUERY_MIN_CHUNK_BYTES = 4 * 1024 ** 2
FIXED_QUERY_CHUNKS_PER_RUN = 16
_TORCH_SOURCE_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def plan_upload_chunks(indices, nbytes, chunk_bytes=FIXED_QUERY_CHUNK_BYTES, alignment=FIXED_QUERY_ALIGNMENT):
    """Cut ``indices`` into consecutive chunks that fill a buffer of about ``chunk_bytes``.

    Returns ``[(entries, used_bytes), ...]`` where ``entries`` lists ``(index, offset, size)``.
    Offsets are aligned to ``alignment`` bytes so each embedding can be viewed in its own dtype.
    An embedding larger than ``chunk_bytes`` is a chunk of its own.
    """
    chunks = []
    entries = []
    used = 0
    for index in indices:
        size = int(nbytes[index])
        padded = -(-size // alignment) * alignment
        if entries and used + padded > chunk_bytes:
            chunks.append((entries, used))
            entries, used = [], 0
        entries.append((int(index), used, size))
        used += padded
    if entries:
        chunks.append((entries, used))
    return chunks


class _ChunkReader(threading.Thread):
    """Reads a fixed-query search's embeddings, chunk by chunk, into reusable host buffers.

    The chunks are read in their planned order, so reading runs ahead of the consumer
    (into the next tile block too) until every buffer holds a chunk it has not taken.
    ``next_chunk`` returns ``(chunk number, entries, used bytes, slot)``; ``release`` gives
    the slot's buffer back once its contents are on the device. A failure while reading
    surfaces from ``next_chunk``; ``close`` stops the thread.
    """

    def __init__(self, store, chunks, buffers):
        super().__init__(name="search-embedding-reader", daemon=True)
        self.store = store
        self.chunks = chunks
        self.buffers = buffers
        self.free = queue.Queue()
        self.ready = queue.Queue()
        self.stopping = threading.Event()
        for slot in range(len(buffers)):
            self.free.put(slot)

    def _take_free(self):
        while not self.stopping.is_set():
            try:
                return self.free.get(timeout=0.05)
            except queue.Empty:
                continue
        return None

    def run(self):
        try:
            with h5py.File(self.store.path, "r", libver="latest", swmr=True) as hf:
                group = hf["embeddings"]
                for number, (entries, used) in enumerate(self.chunks):
                    slot = self._take_free()
                    if slot is None:
                        return
                    raw = self.buffers[slot]
                    for index, offset, size in entries:
                        if self.stopping.is_set():
                            return
                        view = raw[offset:offset + size].view(self.store.dtypes[index])
                        self.store.read_into(index, view.reshape(self.store.shapes[index]), group)
                    self.ready.put((number, entries, used, slot))
        except BaseException as error:  # handed to the consumer, which re-raises it
            self.ready.put(error)
        finally:
            self.ready.put(None)

    def next_chunk(self, idle=None, interval=0.005):
        """The next chunk. While it is read, ``idle()`` runs every ``interval`` seconds."""
        while True:
            try:
                item = self.ready.get(timeout=interval)
            except queue.Empty:
                if not self.is_alive() and self.ready.empty():
                    raise RuntimeError("The embedding reader stopped before every chunk was read.")
                if idle is not None:
                    idle()
                continue
            if item is None:
                raise RuntimeError("The embedding reader stopped before every chunk was read.")
            if isinstance(item, BaseException):
                raise item
            return item

    def release(self, slot):
        self.free.put(slot)

    def close(self):
        self.stopping.set()
        if self.ident is not None:
            self.join()


def run_fixed_query_cuda_pipeline(
    tasks,
    *,
    query_embedding,
    store: EmbeddingTileStore,
    lengths,
    device,
    workers,
    lanes,
    batch_alignment_callback: Callable,
    precision="float32",
    progress=None,
    matrix_budget_override=None,
    task_index: Callable = _fixed_query_task_index,
    timing=None,
    gpu_search_config=None,
):
    """Batch a fixed query against target embeddings and overlap CPU scoring.

    When ``gpu_search_config`` is supplied and CUDA/ROCm runtime alignment kernels
    pass self-check (and lengths/gap fit the fused search kernel), scoring and
    dynamic programming run end-to-end on the GPU with zero score-matrix transfer.
    """
    if getattr(device, "type", None) != "cuda":
        raise ValueError("The fixed-query tiled engine currently requires CUDA.")
    tasks = list(tasks)
    if not tasks:
        return []

    if (
        gpu_search_config is not None
        and Align_GPU_Kernels.align_kernels(device) is not None
        and Align_GPU_Kernels.can_use_fused_search(
            tasks, int(np.asarray(query_embedding).shape[0]), lengths, gpu_search_config, task_index
        )
    ):
        return Align_GPU_Kernels.run_fused_fixed_query_gpu_pipeline(
            tasks,
            query_embedding=query_embedding,
            store=store,
            lengths=lengths,
            device=device,
            lanes=lanes,
            search_config=gpu_search_config,
            precision=precision,
            progress=progress,
            matrix_budget_override=matrix_budget_override,
            task_index=task_index,
            timing=timing,
            cuda_matmul_precision_ctx=cuda_matmul_precision,
            precision_compute_dtype_fn=precision_compute_dtype,
            precision_element_bytes_fn=precision_element_bytes,
            accelerator_memory_plan_fn=accelerator_memory_plan,
            fixed_query_block_ids_fn=fixed_query_block_ids,
            plan_upload_chunks_fn=plan_upload_chunks,
            chunk_reader_cls=_ChunkReader,
            to_normalized_accelerator_fn=_to_normalized_accelerator,
            torch_source_dtypes=_TORCH_SOURCE_DTYPES,
            fixed_query_chunk_bytes=FIXED_QUERY_CHUNK_BYTES,
            fixed_query_min_chunk_bytes=FIXED_QUERY_MIN_CHUNK_BYTES,
            fixed_query_chunks_per_run=FIXED_QUERY_CHUNKS_PER_RUN,
            fixed_query_ring_bytes=FIXED_QUERY_RING_BYTES,
            pins_host_memory_fn=_pins_host_memory,
            clear_failed_frames_fn=_clear_failed_frames,
        )

    element_bytes = precision_element_bytes(precision)
    compute_dtype = precision_compute_dtype(precision)
    plan = cuda_memory_plan(
        device, lanes=lanes, compute_element_bytes=element_bytes
    )
    matrix_budget = int(matrix_budget_override or plan.matrix_bytes)
    query_array = np.asarray(query_embedding)
    query_length = int(query_array.shape[0])
    feature_dimension = int(store.feature_dimension)
    block_ids = fixed_query_block_ids(store, query_array, plan, element_bytes)
    grouped = OrderedDict()
    for task in tasks:
        index = task_index(task)
        grouped.setdefault(int(block_ids[index]), []).append(task)

    # Every block's embeddings, in index order, laid out in the packed buffer and cut into chunks.
    nbytes = store.source_bytes
    chunk_bytes = min(
        FIXED_QUERY_CHUNK_BYTES,
        max(
            FIXED_QUERY_MIN_CHUNK_BYTES,
            sum(int(nbytes[index]) for index in {task_index(task) for task in tasks})
            // FIXED_QUERY_CHUNKS_PER_RUN,
        ),
    )
    block_layouts = []
    chunks = []
    for block_tasks in grouped.values():
        indices = sorted({task_index(task) for task in block_tasks})
        starts, rows = {}, 0
        for index in indices:
            starts[index] = rows
            rows += int(lengths[index])
        planned = plan_upload_chunks(indices, nbytes, chunk_bytes)
        block_layouts.append((starts, rows, len(planned)))
        chunks.extend(planned)
    pad_row = max(rows for _starts, rows, _count in block_layouts)
    capacity = max(used for _entries, used in chunks)
    ring_depth = max(2, min(8, FIXED_QUERY_RING_BYTES // max(1, capacity)))
    buffers = None
    if _pins_host_memory(device):
        try:
            buffers = [
                torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
                for _ in range(ring_depth)
            ]
        except RuntimeError:
            buffers = None
    if buffers is None:
        buffers = [torch.empty(capacity, dtype=torch.uint8) for _ in range(ring_depth)]
    reader = _ChunkReader(store, chunks, [buffer.numpy() for buffer in buffers])

    results = []
    cpu_pending = set()
    inflight = deque()
    max_inflight = max(2, int(lanes) * 2)
    batch_pending_limit = max(1, int(workers)) + max(1, int(lanes))
    streams = [torch.cuda.Stream(device=device) for _ in range(max(1, int(lanes)))]
    upload_stream = torch.cuda.Stream(device=device)
    stream_cursor = 0
    pending_release = deque()
    tiny = torch.finfo(torch.float32).tiny
    packed = raw_device = query_tensor = targets = matrices = host_tensor = event = None

    def collect_cpu(block=False):
        if block and cpu_pending:
            completed, _ = wait(cpu_pending, return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in cpu_pending if future.done()}
        for future in completed:
            cpu_pending.remove(future)
            batch_results = future.result()
            results.extend(batch_results)
            if progress is not None:
                progress.update(len(batch_results))

    def submit_oldest(block):
        if not inflight:
            return
        event, host_tensor, batch_tasks, target_lengths = inflight[0]
        if not block and not event.query():
            return
        if block:
            event.synchronize()
        inflight.popleft()
        cpu_pending.add(
            cpu_executor.submit(
                batch_alignment_callback,
                batch_tasks,
                target_lengths,
                host_tensor.numpy(),
            )
        )
        while len(cpu_pending) >= batch_pending_limit:
            collect_cpu(block=True)

    def release_completed():
        while pending_release and pending_release[0][0].query():
            reader.release(pending_release.popleft()[1])

    def idle():
        submit_oldest(block=False)
        collect_cpu(block=False)
        release_completed()

    def upload_block(layout, indices, after):
        """Queue one block's chunks onto the upload stream; return the event its data is ready at."""
        starts, _rows, chunk_count = layout
        with torch.cuda.stream(upload_stream):
            waited = False
            for _ in range(chunk_count):
                _number, entries, used, slot = reader.next_chunk(idle)
                raw_device[:used].copy_(buffers[slot][:used], non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(upload_stream)
                pending_release.append((copied, slot))
                if not waited:
                    # The packed buffer holds the previous block until every lane has
                    # finished with it: wait on the device, not on the host.
                    for event in after:
                        upload_stream.wait_event(event)
                    waited = True
                for index, offset, size in entries:
                    length = int(lengths[index])
                    source = raw_device[offset:offset + size].view(
                        _TORCH_SOURCE_DTYPES[store.dtypes[index]]
                    ).view(length, feature_dimension)
                    values = source.to(torch.float32)
                    norms = torch.linalg.vector_norm(values, ord=2, dim=-1, keepdim=True)
                    norms.clamp_min_(tiny)
                    target = packed[starts[index]:starts[index] + length]
                    if compute_dtype == torch.float32:
                        torch.div(values, norms, out=target)
                    else:
                        target.copy_(torch.div(values, norms))
                release_completed()
            uploaded = torch.cuda.Event()
            uploaded.record(upload_stream)
        return uploaded

    try:
        with cuda_matmul_precision(precision), torch.inference_mode(), \
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
            with torch.cuda.stream(upload_stream):
                # The packed buffer ends in one all-zero row, which pads every gathered batch.
                packed = torch.zeros((pad_row + 1, feature_dimension), dtype=compute_dtype, device=device)
                raw_device = torch.empty(capacity, dtype=torch.uint8, device=device)
            if timing is not None:
                timing.warm_executor(cpu_executor, max(1, int(workers)))
                timing.start()
            reader.start()

            previous_lanes = []
            for layout, block_tasks in zip(block_layouts, grouped.values()):
                starts = layout[0]
                indices = sorted(starts)
                uploaded = upload_block(layout, indices, previous_lanes)
                for stream in streams:
                    stream.wait_event(uploaded)

                pseudo_tasks = [(0, task_index(task), task) for task in block_tasks]
                for microbatch in _length_microbatches(
                    pseudo_tasks,
                    lengths,
                    query_length,
                    matrix_budget,
                    feature_dimension,
                    element_bytes,
                ):
                    stream = streams[stream_cursor % len(streams)]
                    stream_cursor += 1
                    target_indices = [int(item[1]) for item in microbatch]
                    target_lengths = [int(lengths[index]) for index in target_indices]
                    with torch.cuda.stream(stream):
                        targets = _gather_padded_targets(
                            packed,
                            [starts[index] for index in target_indices],
                            target_lengths,
                            pad_row,
                        )
                        matrices = _batched_score_matrices(
                            query_tensor, targets, target_lengths
                        )
                        host_tensor = _host_output_buffer(matrices)
                        host_tensor.copy_(matrices, non_blocking=host_tensor.is_pinned())
                        event = torch.cuda.Event()
                        event.record(stream)
                    inflight.append(
                        (event, host_tensor, [item[2] for item in microbatch], target_lengths)
                    )
                    while len(inflight) >= max_inflight:
                        submit_oldest(block=True)
                    submit_oldest(block=False)
                    collect_cpu(block=False)

                # The next block may overwrite the packed buffer once these are reached.
                previous_lanes = []
                for stream in streams:
                    reached = torch.cuda.Event()
                    reached.record(stream)
                    previous_lanes.append(reached)

            while inflight:
                submit_oldest(block=True)
            while cpu_pending:
                collect_cpu(block=True)
            if timing is not None:
                timing.stop()
    except BaseException as error:
        _clear_failed_frames(error)  # before the cache is emptied below; the error goes on unchanged
        raise
    finally:
        reader.close()
        inflight.clear()
        pending_release.clear()
        try:
            torch.cuda.synchronize(device)
        except RuntimeError:
            pass
        # Let go of every device tensor first, so the cache can really be emptied.
        packed = raw_device = query_tensor = targets = matrices = host_tensor = event = None
        torch.cuda.empty_cache()
    if timing is not None:
        timing.finish()
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
