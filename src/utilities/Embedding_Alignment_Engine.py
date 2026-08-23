# Copyright 2026 Xuebin Feng
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

from array import array
from collections import OrderedDict, defaultdict, deque
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
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
MIB = 1024 ** 2
DEFAULT_HOST_CACHE_CAP = 128 * GIB
MIN_HOST_RESERVE = 8 * GIB
MIN_ACCELERATOR_RESERVE = 2 * GIB
MAX_HOST_STAGING_BYTES = 512 * MIB
MIN_HOST_STAGING_BYTES = 16 * MIB
MAX_PINNED_STAGING_BYTES = 256 * MIB
MAX_MICROBATCH_WORKSPACE_BYTES = 512 * MIB
MICROBATCH_WORKSPACE_CANDIDATES = (
    256 * MIB,
    512 * MIB,
    768 * MIB,
    1024 * MIB,
)
LENGTH_BUCKET_QUANTUM = 32
MAX_CPU_CHUNK_BYTES = 64 * MIB
CPU_CHUNK_CANDIDATES = (1, 2, 4, 8)
AUTOTUNE_MIN_MEASUREMENT_SECONDS = 0.35
MAX_COMPILED_SHAPE_FAMILIES = 8
MAX_CUDA_GRAPH_SHAPE_FAMILIES = 4
ACCELERATOR_MEMORY_HIGH_WATERMARK = 0.80
ALLOCATOR_TRIM_MIN_BYTES = 512 * MIB
PADDING_OVERHEAD_LIMIT = 0.15
MATRIX_WORKSPACE_MULTIPLIER = 8
ACCELERATOR_TILE_FRACTION = 0.30
ACCELERATOR_MATRIX_FRACTION = 0.50
SUPPORTED_TILED_BACKENDS = frozenset({"cuda", "xpu"})

# Compatibility constants retained for third-party callers.
MIN_CUDA_RESERVE = MIN_ACCELERATOR_RESERVE
CUDA_TILE_FRACTION = ACCELERATOR_TILE_FRACTION
CUDA_MATRIX_FRACTION = ACCELERATOR_MATRIX_FRACTION


class CompactPairTasks(Sequence):
    """Two uint32 index arrays presenting the historical four-field tasks."""

    def __init__(self, capacity, headers):
        self.capacity = max(1, int(capacity))
        self.headers = headers
        self.left = np.empty(self.capacity, dtype=np.uint32)
        self.right = np.empty(self.capacity, dtype=np.uint32)
        self.count = 0

    def append(self, task_or_left, right=None):
        if right is None:
            left, right = int(task_or_left[0]), int(task_or_left[1])
        else:
            left, right = int(task_or_left), int(right)
        if self.count >= self.capacity:
            raise OverflowError("Compact pair batch exceeded its fixed capacity.")
        self.left[self.count] = left
        self.right[self.count] = right
        self.count += 1

    def __len__(self):
        return self.count

    def _task(self, index):
        left = int(self.left[index])
        right = int(self.right[index])
        return left, right, self.headers[left], self.headers[right]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._task(value) for value in range(*index.indices(self.count))]
        index = int(index)
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return self._task(index)

    def __iter__(self):
        for index in range(self.count):
            yield self._task(index)


class _OrdinalTaskView(Sequence):
    """Lazy view of task ordinals belonging to one embedding tile."""

    def __init__(self, tasks, ordinals):
        self.tasks = tasks
        self.ordinals = ordinals

    def __len__(self):
        return len(self.ordinals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[value] for value in range(*index.indices(len(self)))]
        return self.tasks[int(self.ordinals[int(index)])]

    def __iter__(self):
        for ordinal in self.ordinals:
            yield self.tasks[int(ordinal)]


class _CombinedTaskView(Sequence):
    """Zero-copy view over at most two open publication batches."""

    def __init__(self, sequences):
        self.sequences = tuple(sequences)
        self.boundaries = []
        total = 0
        for sequence in self.sequences:
            total += len(sequence)
            self.boundaries.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[value] for value in range(*index.indices(self.total))]
        index = int(index)
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        start = 0
        for sequence, end in zip(self.sequences, self.boundaries):
            if index < end:
                return sequence[index - start]
            start = end
        raise IndexError(index)


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


def resolve_host_staging_bytes():
    """Bound transient result/pinned buffers using current free host memory."""
    _total, available = system_memory_bytes()
    if not available:
        return MAX_HOST_STAGING_BYTES
    return max(
        1,
        min(
            MAX_HOST_STAGING_BYTES,
            max(MIN_HOST_STAGING_BYTES, int(available * 0.05)),
        ),
    )


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
    """Small adapter over PyTorch accelerator-specific runtime APIs.

    PyTorch represents both native CUDA and ROCm devices as ``cuda`` while
    Intel GPUs use ``xpu``.  Keeping those module differences here lets the
    tiled scheduler and scorer remain backend-neutral.
    """

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

    @property
    def display_name(self):
        function = getattr(self.module, "get_device_name", None)
        if function is None:
            return str(self.device)
        try:
            return str(function(self.device))
        except (RuntimeError, TypeError, ValueError):
            return str(self.device)

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

    def create_event(self, *, enable_timing=False):
        event_type = getattr(self.module, "Event", None)
        if event_type is None:
            raise RuntimeError(
                f"torch.{self.device_type}.Event is unavailable."
            )
        if enable_timing:
            try:
                return event_type(enable_timing=True)
            except TypeError:
                pass
        return event_type()

    @staticmethod
    def record_event(event, stream):
        event.record(stream)

    @staticmethod
    def wait_event(stream, event):
        wait_function = getattr(stream, "wait_event", None)
        if wait_function is not None:
            wait_function(event)
            return
        wait_function = getattr(event, "wait", None)
        if wait_function is not None:
            wait_function(stream)
            return
        # This fallback is correct but removes upload/compute overlap.  It is
        # retained for future PyTorch accelerator modules with partial APIs.
        event.synchronize()

    @staticmethod
    def elapsed_time(start_event, end_event):
        """Return elapsed event time in seconds, or ``None`` if unsupported."""
        function = getattr(start_event, "elapsed_time", None)
        if function is None:
            return None
        try:
            return max(0.0, float(function(end_event)) / 1000.0)
        except (RuntimeError, TypeError, ValueError, NotImplementedError):
            return None

    def compile_callable(self, function, *, dynamic=False, mode="default"):
        """Compile one accelerator callable through PyTorch when available."""
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise RuntimeError("This PyTorch build does not expose torch.compile.")
        return compiler(function, dynamic=dynamic, mode=mode)

    def supports_compilation(self):
        return callable(getattr(torch, "compile", None))

    def supports_graph_compilation(self):
        """Return whether reduce-overhead compilation may use CUDA graphs."""
        return bool(
            self.device_type == "cuda"
            and getattr(torch.version, "hip", None) is None
            and self.supports_compilation()
        )

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

    def memory_allocated(self):
        function = self._runtime_function("memory_allocated")
        if function is None:
            return 0
        try:
            return int(function(self.device))
        except TypeError:
            with self.device_context():
                return int(function())

    def memory_reserved(self):
        function = self._runtime_function("memory_reserved")
        if function is None:
            return self.memory_allocated()
        try:
            return int(function(self.device))
        except TypeError:
            with self.device_context():
                return int(function())

    def max_memory_allocated(self):
        function = self._runtime_function("max_memory_allocated")
        if function is None:
            return self.memory_allocated()
        try:
            return int(function(self.device))
        except TypeError:
            with self.device_context():
                return int(function())

    def max_memory_reserved(self):
        function = self._runtime_function("max_memory_reserved")
        if function is None:
            return self.memory_reserved()
        try:
            return int(function(self.device))
        except TypeError:
            with self.device_context():
                return int(function())

    def reset_peak_memory_stats(self):
        function = self._runtime_function("reset_peak_memory_stats")
        if function is None:
            return
        try:
            function(self.device)
        except TypeError:
            with self.device_context():
                function()

    def empty_cache(self):
        function = self._runtime_function("empty_cache")
        if function is not None:
            function()

    def synchronize(self):
        function = getattr(self.module, "synchronize", None)
        if function is None:
            return
        try:
            function(self.device)
        except TypeError:
            with self.device_context():
                function()

    def utilization(self):
        function = getattr(self.module, "utilization", None)
        if function is None:
            return None
        try:
            return float(function(self.device))
        except (RuntimeError, TypeError, ValueError, ImportError, OSError):
            return None

    def is_out_of_memory(self, error):
        exception_types = []
        for owner in (torch, self.module):
            exception_type = getattr(owner, "OutOfMemoryError", None)
            if isinstance(exception_type, type):
                exception_types.append(exception_type)
        if exception_types and isinstance(error, tuple(set(exception_types))):
            return True
        if not isinstance(error, RuntimeError):
            return False
        message = str(error).lower()
        signatures = (
            "out of memory",
            "cannot allocate memory",
            "allocation failed",
        )
        return any(signature in message for signature in signatures)

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
    """Return the registered tiled-runtime adapter for ``device``."""
    return AcceleratorBackend(device)


def tiled_accelerator_support(device, *, require_memory=True):
    """Return ``(supported, reason)`` without initializing accelerator work."""
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
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    reclaimable_bytes: int = 0
    transient_pool_bytes: int = 0
    backend: str = "accelerator"


@dataclass(frozen=True)
class AcceleratorWorkloadEstimate:
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


@dataclass(frozen=True)
class AcceleratorExecutionPlan:
    """One benchmarkable tiled scheduling and scoring configuration."""

    lanes: int
    microbatch_workspace_bytes: int
    length_bucket_quantum: int = LENGTH_BUCKET_QUANTUM
    length_bucket_policy: str = "row_safe"
    scorer_variant: str = "eager"
    active_cpu_workers: int = 1
    cpu_chunk_size: int = 1

    def __post_init__(self):
        if int(self.lanes) < 1:
            raise ValueError("Accelerator execution lanes must be positive.")
        if int(self.microbatch_workspace_bytes) < 1:
            raise ValueError("Microbatch workspace bytes must be positive.")
        if int(self.length_bucket_quantum) < 1:
            raise ValueError("Length bucket quantum must be positive.")
        if self.length_bucket_policy not in {"row_safe", "multirow"}:
            raise ValueError(
                "Length bucket policy must be row_safe or multirow."
            )
        if self.scorer_variant not in {
            "eager",
            "compiled",
            "compiled_graph",
        }:
            raise ValueError(
                "Scorer variant must be eager, compiled, or compiled_graph."
            )
        if int(self.active_cpu_workers) < 1:
            raise ValueError("Active CPU workers must be positive.")
        if int(self.cpu_chunk_size) not in CPU_CHUNK_CANDIDATES:
            raise ValueError(
                "CPU chunk size must be one of "
                f"{', '.join(str(value) for value in CPU_CHUNK_CANDIDATES)}."
            )


@dataclass(frozen=True)
class PairWorkPacket:
    """A memory-estimated, length-bucketed group of independent pairs."""

    tasks: tuple
    ordinals: tuple
    query_lengths: tuple
    target_lengths: tuple
    query_bucket: int
    target_bucket: int
    workspace_bytes: int
    output_bytes: int
    padded_cells: int
    real_cells: int
    estimated_flops: int
    batch_id: int = 0

    @property
    def shape_family(self):
        return (
            len(self.tasks),
            int(self.query_bucket),
            int(self.target_bucket),
        )


@dataclass
class AcceleratorPipelineMetrics:
    """Low-overhead cumulative timing and resource counters."""

    pairs: int = 0
    packets: int = 0
    tiles: int = 0
    prefetched_tiles: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    oom_retries: int = 0
    allocator_trims: int = 0
    workspace_resets: int = 0
    upload_seconds: float = 0.0
    upload_wait_seconds: float = 0.0
    gpu_score_seconds: float = 0.0
    download_seconds: float = 0.0
    cpu_queue_stall_seconds: float = 0.0
    cpu_alignment_seconds: float = 0.0
    writer_seconds: float = 0.0
    tile_boundary_stall_seconds: float = 0.0
    batch_boundary_stall_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    upload_bytes: int = 0
    download_bytes: int = 0
    padded_cells: int = 0
    real_cells: int = 0
    estimated_flops: int = 0
    peak_inflight_packets: int = 0
    upload_queue_high_watermark: int = 0
    result_queue_high_watermark: int = 0
    peak_cpu_pending_tasks: int = 0
    peak_cpu_pending_bytes: int = 0
    peak_host_staging_bytes: int = 0
    peak_pinned_staging_bytes: int = 0
    scratch_growths: int = 0
    compiled_packets: int = 0
    eager_packets: int = 0
    compiled_fallbacks: int = 0
    compilation_seconds: float = 0.0
    packet_shapes: dict = field(default_factory=dict)

    def classification(self):
        elapsed = max(float(self.elapsed_seconds), 1e-9)
        if self.cpu_queue_stall_seconds / elapsed >= 0.15:
            return "cpu-dp-bound"
        transfer = self.upload_seconds + self.download_seconds
        if transfer > self.gpu_score_seconds and transfer / elapsed >= 0.20:
            return "transfer-bound"
        if self.gpu_score_seconds / elapsed < 0.45:
            return "launch-or-scheduler-bound"
        return "device-compute-bound"

    def as_dict(self):
        values = asdict(self)
        values["bottleneck"] = self.classification()
        values["padding_fraction"] = (
            0.0
            if not self.padded_cells
            else max(
                0.0,
                1.0 - float(self.real_cells) / float(self.padded_cells),
            )
        )
        return values


@dataclass(frozen=True)
class AcceleratorPlanMeasurement:
    """One staged autotuning observation."""

    execution_plan: AcceleratorExecutionPlan
    pairs_per_second: float | None
    peak_memory_bytes: int | None = None
    compilation_seconds: float = 0.0
    measured_pairs: int = 0
    results: tuple = ()
    error: str | None = None

    @property
    def succeeded(self):
        return self.error is None and self.pairs_per_second is not None


def microbatch_workspace_candidates(matrix_pool_bytes, lanes):
    """Return unique bounded workspace candidates for one lane count."""
    per_lane = max(1, int(matrix_pool_bytes) // max(1, int(lanes)))
    candidates = [
        min(per_lane, int(candidate))
        for candidate in MICROBATCH_WORKSPACE_CANDIDATES
    ]
    return tuple(sorted(set(max(1, value) for value in candidates)))


def default_accelerator_execution_plan(
    memory_plan,
    workers,
    *,
    microbatch_workspace_bytes=None,
    scorer_variant="eager",
    cpu_chunk_size=1,
):
    """Build the conservative default execution plan for one memory plan."""
    candidates = microbatch_workspace_candidates(
        memory_plan.matrix_pool_bytes,
        memory_plan.lanes,
    )
    default_limit = min(MAX_MICROBATCH_WORKSPACE_BYTES, candidates[-1])
    requested = int(microbatch_workspace_bytes or default_limit)
    return AcceleratorExecutionPlan(
        lanes=int(memory_plan.lanes),
        microbatch_workspace_bytes=max(1, min(candidates[-1], requested)),
        scorer_variant=str(scorer_variant),
        active_cpu_workers=max(1, int(workers)),
        cpu_chunk_size=int(cpu_chunk_size),
    )


def lane_probe_execution_plan(memory_plan, workers):
    """Stage-one eager plan with a 512-MiB-clamped packet budget."""
    return default_accelerator_execution_plan(
        memory_plan,
        workers,
        microbatch_workspace_bytes=MAX_MICROBATCH_WORKSPACE_BYTES,
        scorer_variant="eager",
        cpu_chunk_size=1,
    )


def microbatch_refinement_plans(memory_plan, workers):
    """Stage-two bounded microbatch candidates for one retained lane count."""
    return tuple(
        default_accelerator_execution_plan(
            memory_plan,
            workers,
            microbatch_workspace_bytes=limit,
            scorer_variant="eager",
            cpu_chunk_size=1,
        )
        for limit in microbatch_workspace_candidates(
            memory_plan.matrix_pool_bytes,
            memory_plan.lanes,
        )
    )


def worker_refinement_plans(execution_plan, workers):
    """Benchmark half and all configured CPU workers without a Cartesian grid."""
    worker_counts = tuple(
        sorted({max(1, int(workers) // 2), max(1, int(workers))})
    )
    return tuple(
        replace(execution_plan, active_cpu_workers=count)
        for count in worker_counts
    )


def chunk_refinement_plans(execution_plan):
    """Benchmark supported CPU chunk sizes after worker selection."""
    return tuple(
        replace(execution_plan, cpu_chunk_size=size)
        for size in CPU_CHUNK_CANDIDATES
    )


def _rank_execution_measurements(measurements, limit=None):
    successful = [item for item in measurements if item.succeeded]
    ranked = []
    while successful:
        fastest = max(float(item.pairs_per_second) for item in successful)
        competitive = [
            item
            for item in successful
            if float(item.pairs_per_second) >= fastest * 0.97
        ]
        selected = min(
            competitive,
            key=lambda item: (
                (
                    int(item.peak_memory_bytes)
                    if item.peak_memory_bytes is not None
                    else 2 ** 63 - 1
                ),
                {
                    "eager": 0,
                    "compiled": 1,
                    "compiled_graph": 2,
                }[item.execution_plan.scorer_variant],
                int(item.execution_plan.lanes),
                int(item.execution_plan.microbatch_workspace_bytes),
                -int(item.execution_plan.cpu_chunk_size),
            ),
        )
        ranked.append(selected)
        successful.remove(selected)
        if limit is not None and len(ranked) >= int(limit):
            break
    return ranked


def _execution_results_exact(reference, candidate):
    """Require exact IDs, scores, and DP lengths for an automatic fast path."""
    if len(reference) != len(candidate):
        return False
    for left, right in zip(reference, candidate):
        if len(left) != len(right):
            return False
        for left_value, right_value in zip(left, right):
            if isinstance(left_value, (float, np.floating)) or isinstance(
                right_value, (float, np.floating)
            ):
                if not np.array_equal(
                    np.asarray(left_value), np.asarray(right_value)
                ):
                    return False
            elif left_value != right_value:
                return False
    return True


def benchmark_accelerator_execution_plans(
    *,
    lane_candidates,
    memory_plan_factory,
    workers,
    short_tasks,
    confirmation_tasks,
    remaining_pairs,
    measure,
    allow_compilation=True,
    allow_graph_compilation=False,
):
    """Run the staged lane/cap/CPU/compile autotuner from the approved plan.

    ``measure(plan, tasks)`` returns a mapping containing ``rate``, optional
    ``peak_memory_bytes``, ``compilation_seconds``, and ``results``.  Candidate
    failures are isolated and returned as failed measurements.
    """
    short_tasks = tuple(short_tasks)
    confirmation_tasks = tuple(confirmation_tasks)
    observations = []

    def observe(plan, tasks):
        try:
            values = measure(plan, tasks)
            observation = AcceleratorPlanMeasurement(
                execution_plan=plan,
                pairs_per_second=float(values["rate"]),
                peak_memory_bytes=(
                    None
                    if values.get("peak_memory_bytes") is None
                    else int(values["peak_memory_bytes"])
                ),
                compilation_seconds=float(
                    values.get("compilation_seconds", 0.0)
                ),
                measured_pairs=max(
                    1, int(values.get("measured_pairs", len(tasks)))
                ),
                results=tuple(values.get("results", ())),
            )
        except Exception as error:
            observation = AcceleratorPlanMeasurement(
                execution_plan=plan,
                pairs_per_second=None,
                error=f"{type(error).__name__}: {error}",
            )
        observations.append(observation)
        return observation

    # Stage 1: lanes at the existing 512-MiB-clamped eager capacity.
    lane_results = []
    plans_by_lane = {}
    for lanes in tuple(dict.fromkeys(max(1, int(v)) for v in lane_candidates)):
        memory_plan = memory_plan_factory(lanes)
        plans_by_lane[lanes] = memory_plan
        lane_results.append(
            observe(lane_probe_execution_plan(memory_plan, workers), short_tasks)
        )
    retained_lanes = _rank_execution_measurements(lane_results, limit=2)

    # Stage 2: capacity only for the two retained lane counts.
    microbatch_results = []
    seen = set()
    for item in retained_lanes:
        memory_plan = plans_by_lane[item.execution_plan.lanes]
        for plan in microbatch_refinement_plans(memory_plan, workers):
            key = (
                plan.lanes,
                plan.microbatch_workspace_bytes,
                plan.active_cpu_workers,
                plan.cpu_chunk_size,
                plan.scorer_variant,
                plan.length_bucket_policy,
            )
            if key in seen:
                continue
            seen.add(key)
            if plan == item.execution_plan:
                microbatch_results.append(item)
            else:
                microbatch_results.append(observe(plan, short_tasks))
    # Preserve the established 512-MiB packet boundary for automatic
    # scientific execution. Larger candidates remain benchmarked and visible
    # in telemetry, but changing BMM batch boundaries was observed to alter a
    # rare downstream DP tie. Smaller devices retain their largest feasible
    # clamped boundary and OOM recovery may still halve it.
    compatibility_results = [
        item
        for item in microbatch_results
        if int(item.execution_plan.microbatch_workspace_bytes)
        == int(
            lane_probe_execution_plan(
                plans_by_lane[item.execution_plan.lanes], workers
            ).microbatch_workspace_bytes
        )
    ]
    retained_complete = _rank_execution_measurements(
        compatibility_results or retained_lanes, limit=2
    )

    # CPU stages need enough canonical work to expose queueing and chunking;
    # the 2,048-pair confirmation sample remains bounded but is substantially
    # less noisy than the short accelerator probe.
    cpu_tuning_tasks = confirmation_tasks or short_tasks

    # Stage 5a: choose active worker count for the retained complete plans.
    worker_results = []
    for item in retained_complete:
        for plan in worker_refinement_plans(item.execution_plan, workers):
            worker_results.append(observe(plan, cpu_tuning_tasks))
    retained_workers = _rank_execution_measurements(
        worker_results or retained_complete, limit=2
    )

    # Stage 5b: choose chunk size only after the worker count is selected.
    chunk_results = []
    for item in retained_workers:
        for plan in chunk_refinement_plans(item.execution_plan):
            chunk_results.append(observe(plan, cpu_tuning_tasks))
    retained_chunks = _rank_execution_measurements(
        chunk_results or retained_workers, limit=2
    )

    # Stage 3: benchmark multi-row packing only after the conservative packet
    # and CPU plan is known. Batched GEMM shape can change FP32 reduction bits
    # and, rarely, a DP tie outside a finite tuning sample. Keep the
    # measurement for telemetry, but do not automatically rank it until an
    # exact-matrix validation mechanism is available.
    policy_results = list(retained_chunks)
    for eager_item in retained_chunks:
        multirow_plan = replace(
            eager_item.execution_plan,
            length_bucket_policy="multirow",
        )
        multirow_item = observe(multirow_plan, cpu_tuning_tasks)
        if not multirow_item.succeeded:
            continue
        if (
            eager_item.results
            and multirow_item.results
            and not _execution_results_exact(
                eager_item.results, multirow_item.results
            )
        ):
            continue
    retained_policies = _rank_execution_measurements(
        policy_results, limit=2
    )

    # Stage 7: compile only finalists, and charge cold compilation against the
    # remaining work.  Numerical mismatch permanently rejects that candidate.
    finalists = list(retained_policies)
    if allow_compilation:
        for eager_item in retained_policies:
            compiled_plan = replace(
                eager_item.execution_plan, scorer_variant="compiled"
            )
            compiled_item = observe(compiled_plan, cpu_tuning_tasks)
            if not compiled_item.succeeded:
                continue
            if eager_item.results and compiled_item.results:
                if not _execution_results_exact(
                    eager_item.results, compiled_item.results
                ):
                    continue
            compile_seconds = max(0.0, compiled_item.compilation_seconds)
            measured_elapsed = compiled_item.measured_pairs / max(
                float(compiled_item.pairs_per_second), 1e-9
            )
            warm_elapsed = max(1e-9, measured_elapsed - compile_seconds)
            warm_rate = compiled_item.measured_pairs / warm_elapsed
            projected_compiled = compile_seconds + (
                max(0, int(remaining_pairs)) / max(warm_rate, 1e-9)
            )
            projected_eager = max(0, int(remaining_pairs)) / max(
                float(eager_item.pairs_per_second), 1e-9
            )
            if projected_compiled <= projected_eager / 1.03:
                finalists.append(compiled_item)
                if allow_graph_compilation:
                    graph_plan = replace(
                        eager_item.execution_plan,
                        scorer_variant="compiled_graph",
                    )
                    graph_item = observe(graph_plan, cpu_tuning_tasks)
                    if graph_item.succeeded:
                        equivalent = True
                        if eager_item.results and graph_item.results:
                            equivalent = _execution_results_exact(
                                eager_item.results, graph_item.results
                            )
                        graph_elapsed = graph_item.measured_pairs / max(
                            float(graph_item.pairs_per_second), 1e-9
                        )
                        graph_warm_elapsed = max(
                            1e-9,
                            graph_elapsed
                            - max(0.0, graph_item.compilation_seconds),
                        )
                        graph_rate = (
                            graph_item.measured_pairs / graph_warm_elapsed
                        )
                        projected_graph = max(
                            0.0, graph_item.compilation_seconds
                        ) + max(0, int(remaining_pairs)) / max(
                            graph_rate, 1e-9
                        )
                        if (
                            equivalent
                            and projected_graph <= projected_eager / 1.03
                        ):
                            finalists.append(graph_item)

    # Stage 4 confirmation: exactly the two fastest complete plans advance to
    # the existing production-ordered confirmation sample.
    confirmed = []
    for item in _rank_execution_measurements(finalists, limit=2):
        if confirmation_tasks and confirmation_tasks != short_tasks:
            confirmed.append(observe(item.execution_plan, confirmation_tasks))
        else:
            confirmed.append(item)
    return _rank_execution_measurements(confirmed), tuple(observations)


def measure_accelerator_session(
    session,
    tasks,
    *,
    minimum_seconds=AUTOTUNE_MIN_MEASUREMENT_SECONDS,
):
    """Measure a persistent session long enough to suppress short-probe noise."""
    tasks = tasks if isinstance(tasks, Sequence) else tuple(tasks)
    if not tasks:
        return [], 0.0, 0
    minimum_seconds = max(0.0, float(minimum_seconds))
    first_results = None
    completed = 0
    started = time.perf_counter()
    while first_results is None or (
        time.perf_counter() - started < minimum_seconds
    ):
        values = session.run(tasks)
        if first_results is None:
            first_results = values
        completed += len(tasks)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return first_results, completed / elapsed, completed


def accelerator_memory_plan(
    device,
    lanes=1,
    memory_info=None,
    *,
    allocator_info=None,
    tile_fraction=None,
    matrix_fraction=None,
):
    """Plan one shared transient pool from current accelerator memory.

    ``free`` memory does not include bytes already reserved by PyTorch.  The
    allocator's reserved-but-unallocated bytes are therefore added back as
    reclaimable capacity before the safety reserve is applied.
    """
    lanes = max(1, int(lanes))
    tile_fraction = (
        ACCELERATOR_TILE_FRACTION
        if tile_fraction is None
        else float(tile_fraction)
    )
    matrix_fraction = (
        ACCELERATOR_MATRIX_FRACTION
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
    backend = get_accelerator_backend(device)
    if memory_info is None:
        free_bytes, total_bytes = backend.memory_info()
    else:
        free_bytes, total_bytes = memory_info
    if allocator_info is None and memory_info is None:
        allocated_bytes = backend.memory_allocated()
        reserved_bytes = backend.memory_reserved()
    elif allocator_info is None:
        allocated_bytes = 0
        reserved_bytes = 0
    else:
        allocated_bytes, reserved_bytes = allocator_info
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    allocated_bytes = max(0, int(allocated_bytes))
    reserved_bytes = max(allocated_bytes, int(reserved_bytes))
    reclaimable = max(0, reserved_bytes - allocated_bytes)
    reserve = max(MIN_ACCELERATOR_RESERVE, int(total_bytes * 0.15))
    usable = max(0, free_bytes + reclaimable - reserve)
    inflight_slots = max(2, lanes * 2)
    matrix_pool = max(1, int(usable * matrix_fraction))
    tile_cache = max(1, int(usable * tile_fraction))
    transient_pool = max(1, tile_cache + matrix_pool)
    return AcceleratorMemoryPlan(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        usable_bytes=usable,
        tile_cache_bytes=tile_cache,
        matrix_pool_bytes=matrix_pool,
        # This is a shared capacity, not a fixed per-lane partition.  Runtime
        # scheduling admits concurrent microbatches by their actual estimates.
        matrix_bytes=matrix_pool,
        reserve_bytes=reserve,
        lanes=lanes,
        inflight_slots=inflight_slots,
        allocated_bytes=allocated_bytes,
        reserved_bytes=reserved_bytes,
        reclaimable_bytes=reclaimable,
        transient_pool_bytes=transient_pool,
        backend=backend.device_type,
    )


# Compatibility types/functions for integrations written against the original
# CUDA-only engine.  They now support both CUDA/ROCm and XPU devices.
CudaMemoryPlan = AcceleratorMemoryPlan
CudaWorkloadEstimate = AcceleratorWorkloadEstimate


def cuda_memory_plan(*args, **kwargs):
    """Return a compatibility plan with the historical per-slot cap.

    New tiled all-vs-all code uses :func:`accelerator_memory_plan` and its
    shared pool.  Fixed-query and third-party CUDA callers retain the previous
    conservative ``matrix_bytes`` meaning while seeing the same reserve data.
    """
    plan = accelerator_memory_plan(*args, **kwargs)
    return replace(
        plan,
        matrix_bytes=max(1, plan.matrix_pool_bytes // plan.inflight_slots),
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


def _to_normalized_accelerator(array, device, *, pin_memory=True):
    contiguous = np.ascontiguousarray(array)
    cpu_tensor = torch.from_numpy(contiguous)
    if pin_memory:
        try:
            cpu_tensor = cpu_tensor.pin_memory()
        except (RuntimeError, NotImplementedError):
            pin_memory = False
    tensor = cpu_tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=bool(pin_memory),
    )
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=-1, keepdim=True)
    tensor.div_(norms.clamp_min_(torch.finfo(torch.float32).tiny))
    return tensor, cpu_tensor


def _to_normalized_cuda(array, device):
    """Compatibility wrapper returning only the accelerator tensor."""
    return _to_normalized_accelerator(array, device)[0]


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


def _length_bucket(length, quantum=LENGTH_BUCKET_QUANTUM):
    """Round a length only when the bucket itself stays within 15% padding."""
    length = max(1, int(length))
    quantum = max(1, int(quantum))
    rounded = ((length + quantum - 1) // quantum) * quantum
    if rounded <= length * (1.0 + PADDING_OVERHEAD_LIMIT):
        return rounded
    return length


def _length_group(length, quantum=LENGTH_BUCKET_QUANTUM):
    """Return the lower bucket boundary used only for grouping nearby lengths."""
    length = max(1, int(length))
    quantum = max(1, int(quantum))
    return ((length - 1) // quantum) * quantum


def _pair_packet_workspace_bytes(
    count,
    query_bucket,
    target_bucket,
    feature_dimension,
    query_count=None,
):
    count = max(0, int(count))
    query_bucket = max(1, int(query_bucket))
    target_bucket = max(1, int(target_bucket))
    feature_dimension = max(0, int(feature_dimension))
    matrix_bytes = (
        count
        * query_bucket
        * target_bucket
        * 4
        * MATRIX_WORKSPACE_MULTIPLIER
    )
    query_count = count if query_count is None else max(0, int(query_count))
    embedding_bytes = (
        (query_count * query_bucket + count * target_bucket)
        * feature_dimension
        * 4
    )
    return int(matrix_bytes + embedding_bytes)


def _build_pair_work_packet(
    entries,
    *,
    batch_id,
    feature_dimension,
):
    tasks = tuple(entry[1] for entry in entries)
    ordinals = tuple(int(entry[0]) for entry in entries)
    query_lengths = tuple(int(entry[2]) for entry in entries)
    target_lengths = tuple(int(entry[3]) for entry in entries)
    query_bucket = max(int(entry[4]) for entry in entries)
    target_bucket = max(int(entry[5]) for entry in entries)
    padded_cells = len(entries) * query_bucket * target_bucket
    real_cells = sum(
        query_length * target_length
        for query_length, target_length in zip(
            query_lengths, target_lengths
        )
    )
    workspace_bytes = _pair_packet_workspace_bytes(
        len(entries),
        query_bucket,
        target_bucket,
        feature_dimension,
        query_count=(
            0
            if len({int(task[0]) for task in tasks}) == 1
            else len(entries)
        ),
    )
    output_bytes = int(padded_cells * 4)
    estimated_flops = int(
        2 * real_cells * max(0, int(feature_dimension))
        + 24 * real_cells
    )
    return PairWorkPacket(
        tasks=tasks,
        ordinals=ordinals,
        query_lengths=query_lengths,
        target_lengths=target_lengths,
        query_bucket=query_bucket,
        target_bucket=target_bucket,
        workspace_bytes=workspace_bytes,
        output_bytes=output_bytes,
        padded_cells=padded_cells,
        real_cells=real_cells,
        estimated_flops=estimated_flops,
        batch_id=int(batch_id),
    )


def iter_pair_work_packets(
    tasks,
    lengths,
    workspace_budget,
    feature_dimension,
    *,
    batch_id=0,
    length_bucket_quantum=LENGTH_BUCKET_QUANTUM,
    task_ordinals=None,
    multirow=True,
):
    """Lazily pack multiple query rows into bounded length-bucketed packets."""
    workspace_budget = max(1, int(workspace_budget))
    buckets = OrderedDict()
    indexed_tasks = (
        enumerate(tasks)
        if task_ordinals is None
        else zip(task_ordinals, tasks)
    )
    for ordinal, task in indexed_tasks:
        idx_i, idx_j = int(task[0]), int(task[1])
        query_length = int(lengths[idx_i])
        target_length = int(lengths[idx_j])
        # Exact query lengths retain the canonical column ``std_mean`` path,
        # which is required for alignment-length tie parity. Target lengths
        # remain bucketed because their padding is masked from row statistics.
        query_group = query_length
        target_group = _length_group(target_length, length_bucket_quantum)
        key = (
            query_group,
            target_group,
            None if multirow else idx_i,
        )
        state = buckets.setdefault(
            key,
            {
                "entries": [],
                "real_cells": 0,
                "max_query": 0,
                "max_target": 0,
                "queries": set(),
            },
        )
        entries = state["entries"]
        new_entry = (
            ordinal,
            task,
            query_length,
            target_length,
            query_length,
            target_length,
        )
        candidate_count = len(entries) + 1
        max_query = max(int(state["max_query"]), query_length)
        max_target = max(int(state["max_target"]), target_length)
        padded_cells = candidate_count * max_query * max_target
        real_cells = int(state["real_cells"]) + query_length * target_length
        padding_ok = padded_cells <= real_cells * (
            1.0 + PADDING_OVERHEAD_LIMIT
        )
        workspace_ok = _pair_packet_workspace_bytes(
            candidate_count,
            max_query,
            max_target,
            feature_dimension,
            query_count=(
                1
                if len(state["queries"] | {idx_i}) == 1
                else candidate_count
            ),
        ) <= workspace_budget
        if entries and (not padding_ok or not workspace_ok):
            yield _build_pair_work_packet(
                entries,
                batch_id=batch_id,
                feature_dimension=feature_dimension,
            )
            state = {
                "entries": [],
                "real_cells": 0,
                "max_query": 0,
                "max_target": 0,
                "queries": set(),
            }
            buckets[key] = state
            entries = state["entries"]
        entries.append(new_entry)
        state["real_cells"] = int(state["real_cells"]) + (
            query_length * target_length
        )
        state["max_query"] = max(int(state["max_query"]), query_length)
        state["max_target"] = max(int(state["max_target"]), target_length)
        state["queries"].add(idx_i)

    for state in buckets.values():
        entries = state["entries"]
        if entries:
            yield _build_pair_work_packet(
                entries,
                batch_id=batch_id,
                feature_dimension=feature_dimension,
            )


def schedule_pair_work_packets(packets, lanes):
    """Lazily cost-balance a bounded packet window across compute lanes."""
    iterator = iter(packets)
    window_size = max(2, int(lanes) * 2)
    while True:
        window = []
        for _ in range(window_size):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        # Largest FLOP packets go first so round-robin streams finish their
        # current wave at similar times. Padding/bytes were already enforced
        # while each packet was formed.
        window.sort(
            key=lambda packet: (
                int(packet.estimated_flops),
                int(packet.padded_cells),
                int(packet.workspace_bytes),
            ),
            reverse=True,
        )
        yield from window


def iter_row_safe_work_packets(
    tasks,
    task_ordinals,
    lengths,
    workspace_budget,
    feature_dimension,
    *,
    batch_id=0,
):
    """Wrap the established row packet boundaries in ``PairWorkPacket``."""
    rows = OrderedDict()
    for ordinal, task in zip(task_ordinals, tasks):
        # The wrapper is deliberately unique even if a caller repeats the same
        # tuple object.  The first two fields preserve the legacy
        # ``_length_microbatches`` indexing contract.
        wrapped = (int(task[0]), int(task[1]), int(ordinal), task)
        rows.setdefault(int(task[0]), []).append(wrapped)
    for query_index, wrapped_row_tasks in rows.items():
        for microbatch in _length_microbatches(
            wrapped_row_tasks,
            lengths,
            lengths[query_index],
            workspace_budget,
            feature_dimension,
        ):
            entries = []
            # Stable grouping makes every target slab occupy one contiguous
            # result-buffer range.  The BMM packet shape is unchanged, and
            # ordinals restore publication order after CPU alignment.
            ordered_microbatch = sorted(
                microbatch,
                key=lambda wrapped: _length_group(
                    lengths[int(wrapped[1])]
                ),
            )
            for wrapped in ordered_microbatch:
                task = wrapped[3]
                query_length = int(lengths[int(task[0])])
                target_length = int(lengths[int(task[1])])
                entries.append(
                    (
                        int(wrapped[2]),
                        task,
                        query_length,
                        target_length,
                        query_length,
                        target_length,
                    )
                )
            yield _build_pair_work_packet(
                entries,
                batch_id=batch_id,
                feature_dimension=feature_dimension,
            )


def _prepare_padded_pair_inputs(
    query_tensors,
    target_tensors,
    query_lengths,
    target_lengths,
    query_bucket,
    target_bucket,
    workspace,
):
    """Gather normalized ragged embeddings into reusable padded tensors."""
    batch_size = len(query_tensors)
    feature_dimension = int(query_tensors[0].shape[1])
    device = query_tensors[0].device

    def workspace_tensor(name, shape, dtype):
        tensor = workspace.get(name)
        if (
            tensor is None
            or tensor.dtype != dtype
            or len(tensor.shape) != len(shape)
            or any(
                int(current) < int(required)
                for current, required in zip(tensor.shape, shape)
            )
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            workspace[name] = tensor
        return tensor[tuple(slice(0, int(size)) for size in shape)]

    queries = workspace_tensor(
        "queries",
        (batch_size, int(query_bucket), feature_dimension),
        torch.float32,
    )
    targets = workspace_tensor(
        "targets",
        (batch_size, int(target_bucket), feature_dimension),
        torch.float32,
    )
    queries.zero_()
    targets.zero_()
    for index, (query, target, query_length, target_length) in enumerate(
        zip(query_tensors, target_tensors, query_lengths, target_lengths)
    ):
        queries[index, :int(query_length)].copy_(query)
        targets[index, :int(target_length)].copy_(target)

    query_length_tensor = workspace_tensor(
        "query_lengths", (batch_size,), torch.int64
    )
    target_length_tensor = workspace_tensor(
        "target_lengths", (batch_size,), torch.int64
    )
    # Length metadata is tiny.  Reusing its destination avoids accumulating
    # device allocations while retaining a portable PyTorch copy path.
    query_length_tensor.copy_(
        torch.as_tensor(query_lengths, dtype=torch.int64, device=device)
    )
    target_length_tensor.copy_(
        torch.as_tensor(target_lengths, dtype=torch.int64, device=device)
    )
    return queries, targets, query_length_tensor, target_length_tensor


def _batched_score_matrices(
    row_tensor,
    target_tensors,
    target_lengths,
    *,
    workspace=None,
    capabilities=None,
):
    """Compute padded score matrices while excluding padding from statistics."""
    batch_size = len(target_tensors)
    max_length = max(int(length) for length in target_lengths)
    feature_dimension = int(row_tensor.shape[1])
    row_length = int(row_tensor.shape[0])
    workspace = {} if workspace is None else workspace
    capabilities = {} if capabilities is None else capabilities

    target_shape = (batch_size, max_length, feature_dimension)
    targets = workspace.get("targets")
    if targets is None or any(
        int(current) < required
        for current, required in zip(targets.shape, target_shape)
    ):
        targets = torch.empty(
            target_shape,
            dtype=torch.float32,
            device=row_tensor.device,
        )
        workspace["targets"] = targets
    targets = targets[
        :batch_size,
        :max_length,
        :feature_dimension,
    ]
    targets.zero_()
    for index, (target, length) in enumerate(zip(target_tensors, target_lengths)):
        targets[index, :int(length)].copy_(target)

    matrix_shape = (batch_size, row_length, max_length)
    similarity = workspace.get("similarity")
    if similarity is None or any(
        int(current) < required
        for current, required in zip(similarity.shape, matrix_shape)
    ):
        similarity = torch.empty(
            matrix_shape,
            dtype=torch.float32,
            device=row_tensor.device,
        )
        workspace["similarity"] = similarity
    similarity = similarity[:batch_size, :row_length, :max_length]
    expanded_row = row_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    use_bmm_out = capabilities.get("bmm_out", True)
    if use_bmm_out:
        try:
            torch.bmm(expanded_row, targets.transpose(1, 2), out=similarity)
        except (RuntimeError, NotImplementedError):
            capabilities["bmm_out"] = False
            similarity.copy_(torch.bmm(expanded_row, targets.transpose(1, 2)))
    else:
        similarity.copy_(torch.bmm(expanded_row, targets.transpose(1, 2)))
    similarity.clamp_(-1.0, 1.0).sub_(1.0).exp_()

    final = workspace.get("score")
    if final is None or any(
        int(current) < required
        for current, required in zip(final.shape, matrix_shape)
    ):
        final = torch.empty(
            matrix_shape,
            dtype=torch.float32,
            device=row_tensor.device,
        )
        workspace["score"] = final
    final = final[:batch_size, :row_length, :max_length]

    lengths_tensor = torch.as_tensor(
        target_lengths,
        dtype=torch.int64,
        device=row_tensor.device,
    )
    positions = workspace.get("positions")
    if positions is None or int(positions.numel()) < max_length:
        positions = torch.arange(max_length, device=row_tensor.device)
        workspace["positions"] = positions
    positions = positions[:max_length]
    mask_shape = (batch_size, max_length)
    mask = workspace.get("mask")
    if mask is None or any(
        int(current) < required
        for current, required in zip(mask.shape, mask_shape)
    ):
        mask = torch.empty(
            mask_shape,
            dtype=torch.bool,
            device=row_tensor.device,
        )
        workspace["mask"] = mask
    mask = mask[:batch_size, :max_length]
    if capabilities.get("comparison_out", True):
        try:
            torch.lt(
                positions.unsqueeze(0),
                lengths_tensor.unsqueeze(1),
                out=mask,
            )
        except (RuntimeError, NotImplementedError):
            capabilities["comparison_out"] = False
            mask.copy_(
                positions.unsqueeze(0) < lengths_tensor.unsqueeze(1)
            )
    else:
        mask.copy_(positions.unsqueeze(0) < lengths_tensor.unsqueeze(1))
    mask3 = mask.unsqueeze(1)
    divisor = lengths_tensor.to(torch.float32).view(-1, 1, 1)

    torch.mul(similarity, mask3, out=final)
    row_mean = final.sum(dim=2, keepdim=True) / divisor
    torch.sub(similarity, row_mean, out=final)
    final.mul_(mask3).square_()
    row_std = torch.sqrt(final.sum(dim=2, keepdim=True) / divisor)
    use_std_mean = capabilities.get("std_mean", True)
    if use_std_mean:
        try:
            col_std, col_mean = torch.std_mean(
                similarity, dim=1, keepdim=True, correction=0
            )
        except (RuntimeError, NotImplementedError):
            capabilities["std_mean"] = False
            col_mean = similarity.mean(dim=1, keepdim=True)
            col_std = similarity.std(dim=1, keepdim=True, correction=0)
    else:
        col_mean = similarity.mean(dim=1, keepdim=True)
        col_std = similarity.std(dim=1, keepdim=True, correction=0)
    epsilon = 1e-8
    torch.sub(similarity, row_mean, out=final)
    final.div_(row_std + epsilon)
    similarity.sub_(col_mean).div_(col_std + epsilon)
    final.add_(similarity).mul_(0.5)
    final.masked_fill_(~mask3, 0.0)
    return final


def _batched_pair_score_matrices(
    query_tensors,
    target_tensors,
    query_lengths,
    target_lengths,
    *,
    query_bucket=None,
    target_bucket=None,
    workspace=None,
    capabilities=None,
):
    """Score independent ragged pairs from multiple query rows in one BMM."""
    batch_size = len(query_tensors)
    if not batch_size or batch_size != len(target_tensors):
        raise ValueError("Pair batches require equally sized nonempty inputs.")
    query_lengths = [int(value) for value in query_lengths]
    target_lengths = [int(value) for value in target_lengths]
    if len(query_lengths) != batch_size or len(target_lengths) != batch_size:
        raise ValueError("Pair-batch lengths do not match its tensor count.")
    max_query = max(query_lengths)
    max_target = max(target_lengths)
    query_bucket = max(max_query, int(query_bucket or max_query))
    target_bucket = max(max_target, int(target_bucket or max_target))
    feature_dimension = int(query_tensors[0].shape[1])
    workspace = {} if workspace is None else workspace
    capabilities = {} if capabilities is None else capabilities
    device = query_tensors[0].device

    def workspace_tensor(name, shape, dtype):
        tensor = workspace.get(name)
        if (
            tensor is None
            or tensor.dtype != dtype
            or len(tensor.shape) != len(shape)
            or any(
                int(current) < int(required)
                for current, required in zip(tensor.shape, shape)
            )
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            workspace[name] = tensor
        return tensor[tuple(slice(0, int(size)) for size in shape)]

    queries = workspace_tensor(
        "queries",
        (batch_size, query_bucket, feature_dimension),
        torch.float32,
    )
    targets = workspace_tensor(
        "targets",
        (batch_size, target_bucket, feature_dimension),
        torch.float32,
    )
    queries.zero_()
    targets.zero_()
    for index, (query, target, query_length, target_length) in enumerate(
        zip(query_tensors, target_tensors, query_lengths, target_lengths)
    ):
        queries[index, :query_length].copy_(query)
        targets[index, :target_length].copy_(target)

    matrix_shape = (batch_size, query_bucket, target_bucket)
    similarity = workspace_tensor(
        "similarity", matrix_shape, torch.float32
    )
    if capabilities.get("bmm_out", True):
        try:
            torch.bmm(queries, targets.transpose(1, 2), out=similarity)
        except (RuntimeError, NotImplementedError):
            capabilities["bmm_out"] = False
            similarity.copy_(torch.bmm(queries, targets.transpose(1, 2)))
    else:
        similarity.copy_(torch.bmm(queries, targets.transpose(1, 2)))
    similarity.clamp_(-1.0, 1.0).sub_(1.0).exp_()

    query_length_tensor = torch.as_tensor(
        query_lengths, dtype=torch.int64, device=device
    )
    target_length_tensor = torch.as_tensor(
        target_lengths, dtype=torch.int64, device=device
    )
    query_positions = workspace_tensor(
        "query_positions", (query_bucket,), torch.int64
    )
    target_positions = workspace_tensor(
        "target_positions", (target_bucket,), torch.int64
    )
    torch.arange(query_bucket, device=device, out=query_positions)
    torch.arange(target_bucket, device=device, out=target_positions)
    query_mask = workspace_tensor(
        "query_mask", (batch_size, query_bucket), torch.bool
    )
    target_mask = workspace_tensor(
        "target_mask", (batch_size, target_bucket), torch.bool
    )
    torch.lt(
        query_positions.unsqueeze(0),
        query_length_tensor.unsqueeze(1),
        out=query_mask,
    )
    torch.lt(
        target_positions.unsqueeze(0),
        target_length_tensor.unsqueeze(1),
        out=target_mask,
    )
    valid = query_mask.unsqueeze(2) & target_mask.unsqueeze(1)
    score = workspace_tensor("score", matrix_shape, torch.float32)
    temporary = workspace_tensor("temporary", matrix_shape, torch.float32)
    query_divisor = query_length_tensor.to(torch.float32).view(-1, 1, 1)
    target_divisor = target_length_tensor.to(torch.float32).view(-1, 1, 1)

    torch.mul(similarity, valid, out=score)
    row_mean = score.sum(dim=2, keepdim=True) / target_divisor
    col_mean = score.sum(dim=1, keepdim=True) / query_divisor
    torch.sub(similarity, row_mean, out=temporary)
    temporary.mul_(valid).square_()
    row_std = torch.sqrt(
        temporary.sum(dim=2, keepdim=True) / target_divisor
    )
    torch.sub(similarity, col_mean, out=temporary)
    temporary.mul_(valid).square_()
    col_std = torch.sqrt(
        temporary.sum(dim=1, keepdim=True) / query_divisor
    )
    epsilon = 1e-8
    torch.sub(similarity, row_mean, out=score)
    score.div_(row_std + epsilon)
    similarity.sub_(col_mean).div_(col_std + epsilon)
    score.add_(similarity).mul_(0.5)
    score.masked_fill_(~valid, 0.0)
    return score


def _host_output_buffer(tensor, *, pin_memory=True):
    if pin_memory:
        try:
            return torch.empty(
                tensor.shape,
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
        except (RuntimeError, NotImplementedError):
            pass
    return torch.empty(tensor.shape, dtype=torch.float32, device="cpu")


def _score_padded_pair_tensors(
    queries,
    targets,
    query_lengths,
    target_lengths,
):
    """Functional scorer used by capability-probed compiled shape families."""
    similarity = torch.bmm(queries, targets.transpose(1, 2))
    similarity = torch.exp(similarity.clamp(-1.0, 1.0) - 1.0)
    query_positions = torch.arange(
        queries.shape[1], device=queries.device
    )
    target_positions = torch.arange(
        targets.shape[1], device=targets.device
    )
    query_mask = query_positions.unsqueeze(0) < query_lengths.unsqueeze(1)
    target_mask = target_positions.unsqueeze(0) < target_lengths.unsqueeze(1)
    valid = query_mask.unsqueeze(2) & target_mask.unsqueeze(1)
    masked = similarity * valid
    query_divisor = query_lengths.to(torch.float32).view(-1, 1, 1)
    target_divisor = target_lengths.to(torch.float32).view(-1, 1, 1)
    row_mean = masked.sum(dim=2, keepdim=True) / target_divisor
    col_mean = masked.sum(dim=1, keepdim=True) / query_divisor
    row_delta = (similarity - row_mean) * valid
    col_delta = (similarity - col_mean) * valid
    row_std = torch.sqrt(
        row_delta.square().sum(dim=2, keepdim=True) / target_divisor
    )
    col_std = torch.sqrt(
        col_delta.square().sum(dim=1, keepdim=True) / query_divisor
    )
    score = (
        (similarity - row_mean) / (row_std + 1e-8)
        + (similarity - col_mean) / (col_std + 1e-8)
    ) * 0.5
    return score.masked_fill(~valid, 0.0)


def _score_fixed_query_tensors(query, targets, target_lengths):
    """Functional fixed-row scorer for compiled row-safe shape families."""
    batch_size = targets.shape[0]
    similarity = torch.bmm(
        query.unsqueeze(0).expand(batch_size, -1, -1),
        targets.transpose(1, 2),
    )
    similarity = torch.exp(similarity.clamp(-1.0, 1.0) - 1.0)
    positions = torch.arange(targets.shape[1], device=targets.device)
    mask = positions.unsqueeze(0) < target_lengths.unsqueeze(1)
    mask3 = mask.unsqueeze(1)
    divisor = target_lengths.to(torch.float32).view(-1, 1, 1)
    masked = similarity * mask3
    row_mean = masked.sum(dim=2, keepdim=True) / divisor
    row_delta = (similarity - row_mean) * mask3
    row_std = torch.sqrt(
        row_delta.square().sum(dim=2, keepdim=True) / divisor
    )
    col_std, col_mean = torch.std_mean(
        similarity, dim=1, keepdim=True, correction=0
    )
    score = (
        (similarity - row_mean) / (row_std + 1e-8)
        + (similarity - col_mean) / (col_std + 1e-8)
    ) * 0.5
    return score.masked_fill(~mask3, 0.0)


def _eager_score_padded_pair_tensors(
    queries,
    targets,
    query_lengths,
    target_lengths,
    *,
    workspace,
    capabilities,
    equal_query_lengths=False,
):
    """Allocation-reducing eager form of the pure padded-pair scorer."""
    batch_size, query_bucket = int(queries.shape[0]), int(queries.shape[1])
    target_bucket = int(targets.shape[1])
    device = queries.device

    def workspace_tensor(name, shape, dtype):
        tensor = workspace.get(name)
        if (
            tensor is None
            or tensor.dtype != dtype
            or len(tensor.shape) != len(shape)
            or any(
                int(current) < int(required)
                for current, required in zip(tensor.shape, shape)
            )
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            workspace[name] = tensor
        return tensor[tuple(slice(0, int(size)) for size in shape)]

    matrix_shape = (batch_size, query_bucket, target_bucket)
    similarity = workspace_tensor("similarity", matrix_shape, torch.float32)
    if capabilities.get("bmm_out", True):
        try:
            torch.bmm(queries, targets.transpose(1, 2), out=similarity)
        except (RuntimeError, NotImplementedError):
            capabilities["bmm_out"] = False
            similarity.copy_(torch.bmm(queries, targets.transpose(1, 2)))
    else:
        similarity.copy_(torch.bmm(queries, targets.transpose(1, 2)))
    similarity.clamp_(-1.0, 1.0).sub_(1.0).exp_()

    query_positions = workspace_tensor(
        "query_positions", (query_bucket,), torch.int64
    )
    target_positions = workspace_tensor(
        "target_positions", (target_bucket,), torch.int64
    )
    if capabilities.get("arange_out", True):
        try:
            torch.arange(query_bucket, device=device, out=query_positions)
            torch.arange(target_bucket, device=device, out=target_positions)
        except (RuntimeError, NotImplementedError, TypeError):
            capabilities["arange_out"] = False
            query_positions.copy_(torch.arange(query_bucket, device=device))
            target_positions.copy_(torch.arange(target_bucket, device=device))
    else:
        query_positions.copy_(torch.arange(query_bucket, device=device))
        target_positions.copy_(torch.arange(target_bucket, device=device))

    query_mask = workspace_tensor(
        "query_mask", (batch_size, query_bucket), torch.bool
    )
    target_mask = workspace_tensor(
        "target_mask", (batch_size, target_bucket), torch.bool
    )
    if capabilities.get("comparison_out", True):
        try:
            torch.lt(
                query_positions.unsqueeze(0),
                query_lengths.unsqueeze(1),
                out=query_mask,
            )
            torch.lt(
                target_positions.unsqueeze(0),
                target_lengths.unsqueeze(1),
                out=target_mask,
            )
        except (RuntimeError, NotImplementedError, TypeError):
            capabilities["comparison_out"] = False
            query_mask.copy_(
                query_positions.unsqueeze(0) < query_lengths.unsqueeze(1)
            )
            target_mask.copy_(
                target_positions.unsqueeze(0) < target_lengths.unsqueeze(1)
            )
    else:
        query_mask.copy_(
            query_positions.unsqueeze(0) < query_lengths.unsqueeze(1)
        )
        target_mask.copy_(
            target_positions.unsqueeze(0) < target_lengths.unsqueeze(1)
        )
    valid = workspace_tensor("valid", matrix_shape, torch.bool)
    torch.logical_and(
        query_mask.unsqueeze(2), target_mask.unsqueeze(1), out=valid
    )
    score = workspace_tensor("score", matrix_shape, torch.float32)
    temporary = workspace_tensor("temporary", matrix_shape, torch.float32)
    query_divisor_values = workspace_tensor(
        "query_divisors", (batch_size,), torch.float32
    )
    target_divisor_values = workspace_tensor(
        "target_divisors", (batch_size,), torch.float32
    )
    query_divisor_values.copy_(query_lengths)
    target_divisor_values.copy_(target_lengths)
    query_divisor = query_divisor_values.view(-1, 1, 1)
    target_divisor = target_divisor_values.view(-1, 1, 1)

    torch.mul(similarity, valid, out=score)
    row_mean = score.sum(dim=2, keepdim=True) / target_divisor
    if equal_query_lengths and capabilities.get("std_mean", True):
        try:
            col_std, col_mean = torch.std_mean(
                similarity, dim=1, keepdim=True, correction=0
            )
        except (RuntimeError, NotImplementedError):
            capabilities["std_mean"] = False
            col_mean = score.sum(dim=1, keepdim=True) / query_divisor
            col_std = None
    else:
        col_mean = score.sum(dim=1, keepdim=True) / query_divisor
        col_std = None
    torch.sub(similarity, row_mean, out=temporary)
    temporary.mul_(valid).square_()
    row_std = torch.sqrt(
        temporary.sum(dim=2, keepdim=True) / target_divisor
    )
    if col_std is None:
        torch.sub(similarity, col_mean, out=temporary)
        temporary.mul_(valid).square_()
        col_std = torch.sqrt(
            temporary.sum(dim=1, keepdim=True) / query_divisor
        )
    torch.sub(similarity, row_mean, out=score)
    score.div_(row_std + 1e-8)
    similarity.sub_(col_mean).div_(col_std + 1e-8)
    score.add_(similarity).mul_(0.5)
    score.masked_fill_(~valid, 0.0)
    return score


def _eager_fixed_query_score_tensors(
    query,
    targets,
    target_lengths,
    *,
    workspace,
    capabilities,
):
    """Fast eager scorer when every pair shares one true-length query row."""
    batch_size, target_bucket = int(targets.shape[0]), int(targets.shape[1])
    query_length = int(query.shape[0])
    device = query.device

    def workspace_tensor(name, shape, dtype):
        tensor = workspace.get(name)
        if (
            tensor is None
            or tensor.dtype != dtype
            or len(tensor.shape) != len(shape)
            or any(
                int(current) < int(required)
                for current, required in zip(tensor.shape, shape)
            )
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            workspace[name] = tensor
        return tensor[tuple(slice(0, int(size)) for size in shape)]

    matrix_shape = (batch_size, query_length, target_bucket)
    similarity = workspace_tensor("similarity", matrix_shape, torch.float32)
    expanded_query = query.unsqueeze(0).expand(batch_size, -1, -1)
    if capabilities.get("bmm_out", True):
        try:
            torch.bmm(
                expanded_query, targets.transpose(1, 2), out=similarity
            )
        except (RuntimeError, NotImplementedError):
            capabilities["bmm_out"] = False
            similarity.copy_(
                torch.bmm(expanded_query, targets.transpose(1, 2))
            )
    else:
        similarity.copy_(
            torch.bmm(expanded_query, targets.transpose(1, 2))
        )
    similarity.clamp_(-1.0, 1.0).sub_(1.0).exp_()

    positions = workspace_tensor(
        "target_positions", (target_bucket,), torch.int64
    )
    if capabilities.get("arange_out", True):
        try:
            torch.arange(target_bucket, device=device, out=positions)
        except (RuntimeError, NotImplementedError, TypeError):
            capabilities["arange_out"] = False
            positions.copy_(torch.arange(target_bucket, device=device))
    else:
        positions.copy_(torch.arange(target_bucket, device=device))
    mask = workspace_tensor(
        "target_mask", (batch_size, target_bucket), torch.bool
    )
    if capabilities.get("comparison_out", True):
        try:
            torch.lt(
                positions.unsqueeze(0),
                target_lengths.unsqueeze(1),
                out=mask,
            )
        except (RuntimeError, NotImplementedError, TypeError):
            capabilities["comparison_out"] = False
            mask.copy_(
                positions.unsqueeze(0) < target_lengths.unsqueeze(1)
            )
    else:
        mask.copy_(positions.unsqueeze(0) < target_lengths.unsqueeze(1))
    mask3 = mask.unsqueeze(1)
    divisor_values = workspace_tensor(
        "target_divisors", (batch_size,), torch.float32
    )
    divisor_values.copy_(target_lengths)
    divisor = divisor_values.view(-1, 1, 1)
    score = workspace_tensor("score", matrix_shape, torch.float32)
    torch.mul(similarity, mask3, out=score)
    row_mean = score.sum(dim=2, keepdim=True) / divisor
    torch.sub(similarity, row_mean, out=score)
    score.mul_(mask3).square_()
    row_std = torch.sqrt(
        score.sum(dim=2, keepdim=True) / divisor
    )
    if capabilities.get("std_mean", True):
        try:
            col_std, col_mean = torch.std_mean(
                similarity, dim=1, keepdim=True, correction=0
            )
        except (RuntimeError, NotImplementedError):
            capabilities["std_mean"] = False
            col_mean = similarity.mean(dim=1, keepdim=True)
            col_std = similarity.std(dim=1, keepdim=True, correction=0)
    else:
        col_mean = similarity.mean(dim=1, keepdim=True)
        col_std = similarity.std(dim=1, keepdim=True, correction=0)
    torch.sub(similarity, row_mean, out=score)
    score.div_(row_std + 1e-8)
    similarity.sub_(col_mean).div_(col_std + 1e-8)
    score.add_(similarity).mul_(0.5)
    score.masked_fill_(~mask3, 0.0)
    return score


class _HostBufferPool:
    """Bounded best-fit host buffers shared by result-buffer leases."""

    def __init__(self, capacity_bytes, max_free_buffers=8):
        self.capacity_bytes = max(1, int(capacity_bytes))
        self._free = defaultdict(deque)
        self._leased_storage = {}
        self.max_free_buffers = max(1, int(max_free_buffers))
        self.allocated_bytes = 0
        self.pinned_bytes = 0

    @staticmethod
    def _key(shape, pinned):
        return (
            int(np.prod(tuple(int(value) for value in shape), dtype=np.int64)),
            bool(pinned),
        )

    @staticmethod
    def _bytes(shape):
        return int(np.prod(tuple(shape), dtype=np.int64)) * 4

    def _evict_free(self, required_bytes, capacity_bytes=None):
        capacity = max(
            1,
            min(
                self.capacity_bytes,
                int(capacity_bytes or self.capacity_bytes),
            ),
        )
        while self.allocated_bytes + required_bytes > capacity:
            victim_key = None
            victim_tensor = None
            for key, buffers in self._free.items():
                if buffers:
                    victim_key = key
                    victim_tensor = buffers.pop()
                    break
            if victim_tensor is None:
                return False
            released = int(victim_tensor.numel()) * victim_tensor.element_size()
            self.allocated_bytes = max(0, self.allocated_bytes - released)
            if victim_tensor.is_pinned():
                self.pinned_bytes = max(0, self.pinned_bytes - released)
            if victim_key is not None and not self._free[victim_key]:
                del self._free[victim_key]
        return True

    def acquire(self, shape, *, pin_memory, capacity_bytes=None):
        requested_bytes = self._bytes(shape)
        requested_elements = requested_bytes // 4
        pinned_preferences = (
            (True, False) if pin_memory else (False, True)
        )
        for pinned in pinned_preferences:
            compatible = [
                key
                for key, buffers in self._free.items()
                if buffers and key[1] == pinned
                and int(key[0]) >= requested_elements
            ]
            if compatible:
                key = min(compatible, key=lambda value: int(value[0]))
                storage = self._free[key].pop()
                if not self._free[key]:
                    del self._free[key]
                view = storage[:requested_elements].view(
                    tuple(int(value) for value in shape)
                )
                self._leased_storage[int(view.data_ptr())] = storage
                return view, True
        capacity = max(
            1,
            min(
                self.capacity_bytes,
                int(capacity_bytes or self.capacity_bytes),
            ),
        )
        pooled = requested_bytes <= capacity and self._evict_free(
            requested_bytes, capacity
        )
        if not pooled:
            # Returning no buffer applies backpressure.  An unpooled result
            # allocation here would bypass the aggregate host-staging cap and
            # can eventually page host memory to disk.
            return None, False
        try:
            storage = torch.empty(
                (requested_elements,),
                dtype=torch.float32,
                device="cpu",
                pin_memory=bool(pin_memory),
            )
        except (RuntimeError, NotImplementedError):
            storage = torch.empty(
                (requested_elements,),
                dtype=torch.float32,
                device="cpu",
            )
        if pooled:
            actual_bytes = int(storage.numel()) * storage.element_size()
            self.allocated_bytes += actual_bytes
            if storage.is_pinned():
                self.pinned_bytes += actual_bytes
        view = storage.view(tuple(int(value) for value in shape))
        if pooled:
            self._leased_storage[int(view.data_ptr())] = storage
        return view, pooled

    def release(self, tensor, pooled):
        if not pooled:
            return
        storage = self._leased_storage.pop(
            int(tensor.data_ptr()), tensor.reshape(-1)
        )
        free_count = sum(len(buffers) for buffers in self._free.values())
        if free_count >= self.max_free_buffers:
            smallest_key = min(
                (
                    key for key, buffers in self._free.items() if buffers
                ),
                key=lambda value: int(value[0]),
            )
            if int(smallest_key[0]) < int(storage.numel()):
                victim = self._free[smallest_key].pop()
                if not self._free[smallest_key]:
                    del self._free[smallest_key]
                released = int(victim.numel()) * victim.element_size()
                self.allocated_bytes = max(
                    0, self.allocated_bytes - released
                )
                if victim.is_pinned():
                    self.pinned_bytes = max(0, self.pinned_bytes - released)
            else:
                released = int(storage.numel()) * storage.element_size()
                self.allocated_bytes = max(
                    0, self.allocated_bytes - released
                )
                if storage.is_pinned():
                    self.pinned_bytes = max(0, self.pinned_bytes - released)
                return
        self._free[
            (int(storage.numel()), bool(storage.is_pinned()))
        ].append(storage)

    def clear(self):
        self._free.clear()
        self._leased_storage.clear()
        self.allocated_bytes = 0
        self.pinned_bytes = 0


class _HostBufferLease:
    """Keep a result buffer alive until all CPU chunks release it."""

    def __init__(self, pool, tensor, pooled, references):
        self.pool = pool
        self.tensor = tensor
        self.pooled = bool(pooled)
        self.references = max(0, int(references))
        self.bytes = int(tensor.numel()) * tensor.element_size()
        self.released = False

    def release_reference(self):
        if self.released:
            return False
        self.references -= 1
        if self.references <= 0:
            self.pool.release(self.tensor, self.pooled)
            self.released = True
            return True
        return False

    def release_all(self):
        if not self.released:
            self.pool.release(self.tensor, self.pooled)
            self.released = True


@dataclass(frozen=True)
class _EmbeddingSlabBundle:
    """Normalized padded slabs and lookup positions for one tile."""

    key: tuple
    slabs: dict
    index_map: dict
    bytes: int


def _partition_tiles(tasks, block_ids):
    grouped = OrderedDict()
    for task in tasks:
        key = (int(block_ids[int(task[0])]), int(block_ids[int(task[1])]))
        grouped.setdefault(key, []).append(task)
    return grouped.values()


def estimate_accelerator_working_set(
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
    """Estimate peak device use for one workload/variant without allocating."""
    if memory_plan_override is not None and memory_info is not None:
        raise ValueError(
            "memory_info and memory_plan_override cannot be supplied together."
        )
    plan = memory_plan_override or accelerator_memory_plan(
        device, lanes=lanes, memory_info=memory_info
    )
    if int(plan.lanes) != max(1, int(lanes)):
        raise ValueError(
            "Accelerator memory plan lane count does not match execution lanes."
        )
    tasks = list(tasks)
    baseline_bytes = max(0, plan.total_bytes - plan.free_bytes)
    safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
    if not tasks:
        return AcceleratorWorkloadEstimate(
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
    microbatch_fits = variant != "tiled" or largest <= plan.matrix_pool_bytes
    feasible = microbatch_fits and projected_peak <= safe_peak
    if not microbatch_fits:
        reason = "one minimum-size microbatch exceeds the shared matrix pool"
    elif projected_peak > safe_peak:
        reason = "projected peak exceeds the reserved device-memory boundary"
    else:
        reason = "within reserved device-memory boundary"
    return AcceleratorWorkloadEstimate(
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


def estimate_cuda_working_set(*args, **kwargs):
    """Compatibility wrapper for :func:`estimate_accelerator_working_set`."""
    return estimate_accelerator_working_set(*args, **kwargs)


class _LegacyTiledAcceleratorSession:
    """Persistent tiled producer/consumer pipeline for one accelerator plan."""

    def __init__(
        self,
        *,
        store: EmbeddingTileStore,
        lengths,
        device,
        workers,
        lanes,
        alignment_callback: Callable,
        precision="float32",
        matrix_budget_override=None,
        memory_plan_override=None,
    ):
        self.store = store
        self.lengths = lengths
        self.device = torch.device(device)
        self.workers = max(1, int(workers))
        self.lanes = max(1, int(lanes))
        self.alignment_callback = alignment_callback
        self.precision = precision
        self.backend = get_accelerator_backend(self.device)
        supported, reason = self.backend.supports_tiled(require_memory=True)
        if not supported:
            raise RuntimeError(
                f"Tiled execution is unavailable on {self.device}: {reason}."
            )
        self.plan = memory_plan_override or accelerator_memory_plan(
            self.device, lanes=self.lanes
        )
        if int(self.plan.lanes) != self.lanes:
            raise ValueError(
                "Accelerator memory plan lane count does not match execution lanes."
            )
        # Matrix work may use the shared matrix pool, but it must not borrow
        # the embedding allowance wholesale.  Doing that can create one very
        # large device workspace and an equally large host result buffer.
        self.matrix_budget = max(
            1,
            min(
                int(matrix_budget_override or self.plan.matrix_pool_bytes),
                int(self.plan.matrix_pool_bytes),
            ),
        )
        self.microbatch_workspace_limit = max(
            1,
            min(
                MAX_MICROBATCH_WORKSPACE_BYTES,
                self.matrix_budget // self.lanes,
            ),
        )
        self.host_staging_budget = resolve_host_staging_bytes()
        self.pinned_staging_budget = max(
            1,
            min(MAX_PINNED_STAGING_BYTES, self.host_staging_budget // 2),
        )
        self.device_high_watermark = max(
            1,
            min(
                int(self.plan.total_bytes) - int(self.plan.reserve_bytes),
                int(self.plan.total_bytes * ACCELERATOR_MEMORY_HIGH_WATERMARK),
            ),
        )
        self.upload_stream = self.backend.create_stream()
        self.compute_streams = [
            self.backend.create_stream() for _ in range(self.lanes)
        ]
        self._stream_cursor = 0
        self._workspaces = [{} for _ in range(self.lanes)]
        self._workspace_bytes = [0 for _ in range(self.lanes)]
        self._embedding_cache = OrderedDict()
        self._embedding_cache_bytes = 0
        self._upload_staging = deque()
        self._upload_staging_bytes = 0
        self._result_staging_bytes = 0
        self._result_pinned_staging_bytes = 0
        self._capabilities = {
            "bmm_out": True,
            "comparison_out": True,
            "std_mean": True,
        }
        self._h5 = h5py.File(
            self.store.path, "r", libver="latest", swmr=True
        )
        self._group = self._h5["embeddings"]
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="alignment-cpu",
        )
        self._closed = False
        self._pinned_transfers = self._probe_pinned_host_memory()
        self.telemetry = {
            "pairs": 0,
            "tiles": 0,
            "prefetched_tiles": 0,
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "oom_retries": 0,
            "allocator_trims": 0,
            "peak_host_staging_bytes": 0,
            "peak_pinned_staging_bytes": 0,
        }
        self.backend.reset_peak_memory_stats()

    def _probe_pinned_host_memory(self):
        try:
            source = torch.ones((1,), dtype=torch.float32, pin_memory=True)
            if not source.is_pinned():
                return False
            target = torch.empty_like(source, device="cpu", pin_memory=True)
            with self.backend.stream_context(self.upload_stream):
                device_probe = source.to(
                    device=self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                target.copy_(device_probe, non_blocking=True)
                event = self.backend.create_event()
                self.backend.record_event(event, self.upload_stream)
            event.synchronize()
            return bool(target.item() == 1.0)
        except (RuntimeError, NotImplementedError):
            try:
                self.backend.synchronize()
            except (RuntimeError, NotImplementedError):
                pass
            return False

    def _collect_upload_staging(self, block=False):
        while self._upload_staging:
            event, staging, staging_bytes = self._upload_staging[0]
            if not block and not event.query():
                break
            if block:
                event.synchronize()
            self._upload_staging.popleft()
            self._upload_staging_bytes = max(
                0, self._upload_staging_bytes - int(staging_bytes)
            )
            del staging

    def _evict_embedding_cache(self, required_indices, incoming_bytes):
        required_indices = set(required_indices)
        capacity = max(1, int(self.plan.tile_cache_bytes))
        while (
            self._embedding_cache
            and self._embedding_cache_bytes + incoming_bytes > capacity
        ):
            evicted = None
            for index in self._embedding_cache:
                if index not in required_indices:
                    evicted = index
                    break
            if evicted is None:
                break
            tensor = self._embedding_cache.pop(evicted)
            self._embedding_cache_bytes -= int(tensor.numel()) * tensor.element_size()

    def _load_tile(self, indices):
        ordered_indices = sorted(set(int(index) for index in indices))
        missing = [
            index for index in ordered_indices
            if index not in self._embedding_cache
        ]
        self.telemetry["embedding_cache_hits"] += len(ordered_indices) - len(missing)
        self.telemetry["embedding_cache_misses"] += len(missing)
        for index in ordered_indices:
            if index in self._embedding_cache:
                self._embedding_cache.move_to_end(index)
        if not missing:
            return {
                index: self._embedding_cache[index]
                for index in ordered_indices
            }, None

        incoming_bytes = sum(self.store.float32_bytes[index] for index in missing)
        self._evict_embedding_cache(ordered_indices, incoming_bytes)
        staging = []
        staging_bytes = 0
        with self.backend.stream_context(self.upload_stream):
            for index in missing:
                host_embedding = self.store.get(index, self._group)
                source_bytes = int(np.asarray(host_embedding).nbytes)
                use_pinned = bool(
                    self._pinned_transfers
                    and self._upload_staging_bytes
                    + self._result_pinned_staging_bytes
                    + staging_bytes
                    + source_bytes
                    <= self.pinned_staging_budget
                )
                tensor, host_tensor = _to_normalized_accelerator(
                    host_embedding,
                    self.device,
                    pin_memory=use_pinned,
                )
                self._embedding_cache[index] = tensor
                self._embedding_cache_bytes += (
                    int(tensor.numel()) * tensor.element_size()
                )
                if use_pinned and host_tensor.is_pinned():
                    staging.append(host_tensor)
                    staging_bytes += (
                        int(host_tensor.numel()) * host_tensor.element_size()
                    )
                del host_embedding
            event = self.backend.create_event()
            self.backend.record_event(event, self.upload_stream)
        if staging:
            self._upload_staging.append((event, staging, staging_bytes))
            self._upload_staging_bytes += staging_bytes
            self.telemetry["peak_pinned_staging_bytes"] = max(
                int(self.telemetry["peak_pinned_staging_bytes"]),
                int(self._upload_staging_bytes)
                + int(self._result_pinned_staging_bytes),
            )
            self.telemetry["peak_host_staging_bytes"] = max(
                int(self.telemetry["peak_host_staging_bytes"]),
                int(self._upload_staging_bytes)
                + int(self._result_staging_bytes),
            )
        return {
            index: self._embedding_cache[index]
            for index in ordered_indices
        }, event

    def _workspace_requires_reset(
        self, stream_index, estimated_bytes, capacity=None
    ):
        estimated_bytes = max(1, int(estimated_bytes))
        capacity = max(1, int(capacity or self.matrix_budget))
        previous = self._workspace_bytes[stream_index]
        projected = sum(self._workspace_bytes) - previous + max(
            previous, estimated_bytes
        )
        return projected > capacity

    def _prepare_workspace(self, stream_index, estimated_bytes, capacity=None):
        estimated_bytes = max(1, int(estimated_bytes))
        self._workspace_bytes[stream_index] = max(
            self._workspace_bytes[stream_index], estimated_bytes
        )
        return self._workspaces[stream_index]

    def _reset_workspaces(self, *, trim_allocator=False):
        self._workspaces = [{} for _ in range(self.lanes)]
        self._workspace_bytes = [0 for _ in range(self.lanes)]
        if trim_allocator:
            self.backend.empty_cache()
            self.telemetry["allocator_trims"] += 1

    def _trim_allocator_if_needed(self, *, force=False):
        allocated = max(0, int(self.backend.memory_allocated()))
        reserved = max(allocated, int(self.backend.memory_reserved()))
        reclaimable = max(0, reserved - allocated)
        should_trim = bool(
            force
            or reserved > self.device_high_watermark
            or (
                reclaimable >= ALLOCATOR_TRIM_MIN_BYTES
                and reserved > int(self.plan.total_bytes * 0.75)
            )
        )
        if should_trim:
            self.backend.empty_cache()
            self.telemetry["allocator_trims"] += 1
        return should_trim

    def run(
        self,
        tasks,
        *,
        progress=None,
        result_callback=None,
        result_chunk_size=65536,
        matrix_budget_override=None,
    ):
        """Run one output batch while retaining streams, buffers, and tiles."""
        if self._closed:
            raise RuntimeError("The tiled accelerator session is closed.")
        tasks = tasks if isinstance(tasks, Sequence) else tuple(tasks)
        if not tasks:
            return []
        started = time.perf_counter()
        starting_pairs = int(self.telemetry["pairs"])
        matrix_budget = max(
            1,
            min(
                int(matrix_budget_override or self.matrix_budget),
                int(self.matrix_budget),
            ),
        )
        per_block = max(1, int(self.plan.tile_cache_bytes) // 2)
        block_ids = self.store.block_ids(per_block)
        results = []
        cpu_pending = set()
        inflight = deque()
        max_inflight = max(2, self.lanes * 2)
        cpu_pending_limit = max(1, self.workers * 2)

        def collect_cpu(block=False):
            nonlocal cpu_pending
            if block and cpu_pending:
                completed, _ = wait(cpu_pending, return_when=FIRST_COMPLETED)
            else:
                completed = {future for future in cpu_pending if future.done()}
            for future in completed:
                cpu_pending.remove(future)
                results.append(future.result())
                self.telemetry["pairs"] += 1
                if progress is not None:
                    progress.update(1)
            if result_callback is not None and len(results) >= int(result_chunk_size):
                result_callback(results)
                results.clear()

        def submit_oldest(block):
            if not inflight:
                return False
            (
                event,
                host_tensor,
                metadata,
                _workspace_bytes,
                _output_bytes,
                _pinned_bytes,
            ) = inflight[0]
            if not block and not event.query():
                return False
            if block:
                event.synchronize()
            inflight.popleft()
            self._result_staging_bytes = max(
                0, self._result_staging_bytes - int(_output_bytes)
            )
            self._result_pinned_staging_bytes = max(
                0, self._result_pinned_staging_bytes - int(_pinned_bytes)
            )
            matrix_array = host_tensor.numpy()
            for offset, (idx_i, idx_j, length) in enumerate(metadata):
                while len(cpu_pending) >= cpu_pending_limit:
                    collect_cpu(block=True)
                matrix = matrix_array[offset, :, :int(length)]
                if not np.isfinite(matrix).all():
                    raise FloatingPointError(
                        "Batched accelerator scoring produced non-finite values."
                    )
                cpu_pending.add(
                    self._cpu_executor.submit(
                        self.alignment_callback,
                        (idx_i, idx_j, matrix),
                    )
                )
            return True

        precision_context = (
            cuda_matmul_precision(self.precision)
            if self.backend.device_type == "cuda"
            else nullcontext()
        )
        try:
            with precision_context, torch.inference_mode(), self.backend.device_context():
                tile_groups = list(_partition_tiles(tasks, block_ids))
                prefetched_tiles = {}
                for tile_position, tile_tasks in enumerate(tile_groups):
                    self.telemetry["tiles"] += 1
                    tile_indices = {int(task[0]) for task in tile_tasks}
                    tile_indices.update(int(task[1]) for task in tile_tasks)
                    if tile_position in prefetched_tiles:
                        device_embeddings, preload_event = prefetched_tiles.pop(
                            tile_position
                        )
                    else:
                        device_embeddings, preload_event = self._load_tile(
                            tile_indices
                        )
                    next_tile_indices = None
                    prefetch_extra_bytes = 0
                    if tile_position + 1 < len(tile_groups):
                        next_tile = tile_groups[tile_position + 1]
                        next_tile_indices = {
                            int(task[0]) for task in next_tile
                        }
                        next_tile_indices.update(
                            int(task[1]) for task in next_tile
                        )
                        union_indices = tile_indices | next_tile_indices
                        union_bytes = sum(
                            self.store.float32_bytes[index]
                            for index in union_indices
                        )
                        if union_bytes <= int(self.plan.tile_cache_bytes):
                            prefetch_extra_bytes = sum(
                                self.store.float32_bytes[index]
                                for index in next_tile_indices
                                if index not in self._embedding_cache
                            )
                        else:
                            next_tile_indices = None
                    tile_matrix_budget = max(
                        1,
                        min(
                            matrix_budget,
                            int(self.plan.transient_pool_bytes)
                            - int(self._embedding_cache_bytes)
                            - int(prefetch_extra_bytes),
                        ),
                    )
                    prefetch_started = False

                    rows = OrderedDict()
                    for task in tile_tasks:
                        rows.setdefault(int(task[0]), []).append(task)
                    for idx_i, row_tasks in rows.items():
                        for microbatch in _length_microbatches(
                            row_tasks,
                            self.lengths,
                            self.lengths[idx_i],
                            min(
                                tile_matrix_budget,
                                self.microbatch_workspace_limit,
                            ),
                            self.store.feature_dimension,
                        ):
                            target_lengths = [
                                int(self.lengths[int(task[1])])
                                for task in microbatch
                            ]
                            workspace_bytes = _microbatch_workspace_bytes(
                                self.lengths[idx_i],
                                target_lengths,
                                self.store.feature_dimension,
                            )
                            while (
                                inflight
                                and sum(item[3] for item in inflight)
                                + workspace_bytes > tile_matrix_budget
                            ):
                                submit_oldest(block=True)
                            stream_index = self._stream_cursor % self.lanes
                            stream = self.compute_streams[stream_index]
                            self._stream_cursor += 1
                            if self._workspace_requires_reset(
                                stream_index,
                                workspace_bytes,
                                tile_matrix_budget,
                            ):
                                while inflight:
                                    submit_oldest(block=True)
                                self._reset_workspaces(trim_allocator=True)
                            workspace = self._prepare_workspace(
                                stream_index,
                                workspace_bytes,
                                tile_matrix_budget,
                            )
                            targets = [
                                device_embeddings[int(task[1])]
                                for task in microbatch
                            ]
                            output_bytes = int(
                                len(microbatch)
                                * int(self.lengths[idx_i])
                                * max(target_lengths)
                                * 4
                            )
                            self._collect_upload_staging(block=False)
                            while (
                                self._upload_staging
                                and self._upload_staging_bytes
                                + self._result_staging_bytes
                                + output_bytes
                                > self.host_staging_budget
                            ):
                                self._collect_upload_staging(block=True)
                            while (
                                inflight
                                and self._result_staging_bytes
                                + output_bytes
                                > self.host_staging_budget
                            ):
                                submit_oldest(block=True)
                            pin_output = bool(
                                self._pinned_transfers
                                and self._upload_staging_bytes
                                + self._result_pinned_staging_bytes
                                + output_bytes
                                <= self.pinned_staging_budget
                            )
                            with self.backend.stream_context(stream):
                                if preload_event is not None:
                                    self.backend.wait_event(stream, preload_event)
                                matrices = _batched_score_matrices(
                                    device_embeddings[idx_i],
                                    targets,
                                    target_lengths,
                                    workspace=workspace,
                                    capabilities=self._capabilities,
                                )
                                host_tensor = _host_output_buffer(
                                    matrices,
                                    pin_memory=pin_output,
                                )
                                non_blocking = bool(
                                    self._pinned_transfers
                                    and host_tensor.is_pinned()
                                )
                                host_tensor.copy_(
                                    matrices, non_blocking=non_blocking
                                )
                                event = self.backend.create_event()
                                self.backend.record_event(event, stream)
                            metadata = [
                                (int(task[0]), int(task[1]), length)
                                for task, length in zip(
                                    microbatch, target_lengths
                                )
                            ]
                            inflight.append(
                                (
                                    event,
                                    host_tensor,
                                    metadata,
                                    workspace_bytes,
                                    output_bytes,
                                    output_bytes if host_tensor.is_pinned() else 0,
                                )
                            )
                            self._result_staging_bytes += output_bytes
                            if host_tensor.is_pinned():
                                self._result_pinned_staging_bytes += output_bytes
                            self.telemetry["peak_host_staging_bytes"] = max(
                                int(self.telemetry["peak_host_staging_bytes"]),
                                int(self._upload_staging_bytes)
                                + int(self._result_staging_bytes),
                            )
                            self.telemetry["peak_pinned_staging_bytes"] = max(
                                int(self.telemetry["peak_pinned_staging_bytes"]),
                                int(self._upload_staging_bytes)
                                + int(self._result_pinned_staging_bytes),
                            )
                            if (
                                not prefetch_started
                                and next_tile_indices is not None
                            ):
                                prefetched_tiles[tile_position + 1] = (
                                    self._load_tile(next_tile_indices)
                                )
                                prefetch_started = True
                                self.telemetry["prefetched_tiles"] += 1
                            while len(inflight) >= max_inflight:
                                submit_oldest(block=True)
                            submit_oldest(block=False)
                            collect_cpu(block=False)

                    while inflight:
                        submit_oldest(block=True)
                    self._collect_upload_staging(block=False)
                    self._trim_allocator_if_needed()

                while cpu_pending:
                    collect_cpu(block=True)
                self._collect_upload_staging(block=True)
        except Exception:
            while inflight:
                pending_count = len(inflight)
                try:
                    submit_oldest(block=True)
                except Exception:
                    if len(inflight) == pending_count:
                        inflight.popleft()
            if cpu_pending:
                wait(cpu_pending)
            self.telemetry["pairs"] = starting_pairs
            self.telemetry["elapsed_seconds"] = (
                float(self.telemetry.get("elapsed_seconds", 0.0))
                + max(0.0, time.perf_counter() - started)
            )
            raise

        if result_callback is not None and results:
            result_callback(results)
            results.clear()
        self.telemetry["elapsed_seconds"] = (
            float(self.telemetry.get("elapsed_seconds", 0.0))
            + max(0.0, time.perf_counter() - started)
        )
        return results

    def recover_from_oom(self, minimum_budget=1024 ** 2):
        """Clear reusable state, refresh memory, and halve the shared pool."""
        self.telemetry["oom_retries"] += 1
        self.backend.synchronize()
        self._embedding_cache.clear()
        self._embedding_cache_bytes = 0
        self._upload_staging.clear()
        self._upload_staging_bytes = 0
        self._result_staging_bytes = 0
        self._result_pinned_staging_bytes = 0
        self._reset_workspaces()
        self.backend.empty_cache()
        refreshed = accelerator_memory_plan(self.device, lanes=self.lanes)
        self.plan = refreshed
        halved = max(1, int(self.matrix_budget) // 2)
        requested = max(int(minimum_budget), halved)
        self.matrix_budget = max(
            1,
            min(int(refreshed.matrix_pool_bytes), requested),
        )
        self.microbatch_workspace_limit = max(
            1,
            min(
                MAX_MICROBATCH_WORKSPACE_BYTES,
                self.matrix_budget // self.lanes,
            ),
        )
        self.host_staging_budget = resolve_host_staging_bytes()
        self.pinned_staging_budget = max(
            1,
            min(MAX_PINNED_STAGING_BYTES, self.host_staging_budget // 2),
        )
        self.device_high_watermark = max(
            1,
            min(
                int(refreshed.total_bytes) - int(refreshed.reserve_bytes),
                int(
                    refreshed.total_bytes
                    * ACCELERATOR_MEMORY_HIGH_WATERMARK
                ),
            ),
        )
        return self.matrix_budget

    def metrics(self):
        """Return backend-neutral runtime telemetry for diagnostics."""
        metrics = dict(self.telemetry)
        elapsed = max(float(metrics.get("elapsed_seconds", 0.0)), 1e-9)
        metrics.update(
            {
                "backend": self.backend.device_type,
                "device": self.backend.display_name,
                "lanes": self.lanes,
                "matrix_budget_bytes": int(self.matrix_budget),
                "microbatch_workspace_limit_bytes": int(
                    self.microbatch_workspace_limit
                ),
                "host_staging_budget_bytes": int(self.host_staging_budget),
                "pinned_staging_budget_bytes": int(
                    self.pinned_staging_budget
                ),
                "device_high_watermark_bytes": int(
                    self.device_high_watermark
                ),
                "tile_cache_bytes": int(self._embedding_cache_bytes),
                "peak_allocated_bytes": self.backend.max_memory_allocated(),
                "peak_reserved_bytes": self.backend.max_memory_reserved(),
                "utilization_percent": self.backend.utilization(),
                "throughput_pairs_per_second": (
                    int(metrics.get("pairs", 0)) / elapsed
                ),
            }
        )
        return metrics

    def close(self):
        if self._closed:
            return
        self.backend.synchronize()
        self._collect_upload_staging(block=True)
        self._cpu_executor.shutdown(wait=True)
        self._h5.close()
        self._embedding_cache.clear()
        self._workspaces.clear()
        self.backend.empty_cache()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TiledAcceleratorSession(_LegacyTiledAcceleratorSession):
    """Length-bucketed, bounded persistent accelerator session.

    The legacy implementation above remains as a compatibility reference for
    fixed-query callers and old tests.  This class replaces its one-query-row
    producer with multi-row packets, reusable host buffers, byte-aware CPU
    backpressure, event timing, and optional fixed-shape compilation.
    """

    def __init__(
        self,
        *,
        store: EmbeddingTileStore,
        lengths,
        device,
        workers,
        lanes,
        alignment_callback: Callable,
        precision="float32",
        matrix_budget_override=None,
        memory_plan_override=None,
        execution_plan=None,
        alignment_chunk_callback=None,
        enable_compilation=True,
        print_summary=True,
    ):
        super().__init__(
            store=store,
            lengths=lengths,
            device=device,
            workers=workers,
            lanes=lanes,
            alignment_callback=alignment_callback,
            precision=precision,
            matrix_budget_override=matrix_budget_override,
            memory_plan_override=memory_plan_override,
        )
        if execution_plan is None:
            execution_plan = default_accelerator_execution_plan(
                self.plan,
                self.workers,
            )
        if int(execution_plan.lanes) != self.lanes:
            raise ValueError(
                "Accelerator execution-plan lane count does not match the "
                "session lane count."
            )
        self.execution_plan = replace(
            execution_plan,
            microbatch_workspace_bytes=max(
                1,
                min(
                    int(execution_plan.microbatch_workspace_bytes),
                    int(self.matrix_budget) // self.lanes,
                ),
            ),
            active_cpu_workers=max(
                1,
                min(int(execution_plan.active_cpu_workers), self.workers),
            ),
        )
        self.microbatch_workspace_limit = int(
            self.execution_plan.microbatch_workspace_bytes
        )
        if self.execution_plan.active_cpu_workers != self.workers:
            self._cpu_executor.shutdown(wait=True)
            self._cpu_executor = ThreadPoolExecutor(
                max_workers=self.execution_plan.active_cpu_workers,
                thread_name_prefix="alignment-cpu",
            )
        self._alignment_chunk_callback = alignment_chunk_callback
        self._metrics = AcceleratorPipelineMetrics()
        self._host_pool = _HostBufferPool(
            self.host_staging_budget,
            max_free_buffers=max(2, self.lanes),
        )
        self._slab_cache = OrderedDict()
        self._slab_cache_bytes = 0
        self._compiled_scorers = OrderedDict()
        self._compiled_failures = set()
        self._compiled_new_shapes = set()
        self._enable_compilation = bool(enable_compilation)
        self._print_summary = bool(print_summary)
        self._summary_printed = False

    def _load_tile(self, indices):
        while len(self._upload_staging) >= 2:
            wait_started = time.perf_counter()
            self._collect_upload_staging(block=True)
            self._metrics.upload_wait_seconds += max(
                0.0, time.perf_counter() - wait_started
            )
        ordered = tuple(sorted(set(int(index) for index in indices)))
        if ordered in self._slab_cache:
            bundle, event = self._slab_cache.pop(ordered)
            self._slab_cache[ordered] = (bundle, event)
            self._metrics.embedding_cache_hits += len(ordered)
            return bundle, event

        self._metrics.embedding_cache_misses += len(ordered)
        started = time.perf_counter()
        grouped = OrderedDict()
        quantum = int(self.execution_plan.length_bucket_quantum)
        for index in ordered:
            group_key = _length_group(self.lengths[index], quantum)
            grouped.setdefault(group_key, []).append(index)
        for bucket_indices in grouped.values():
            bucket_indices.sort(
                key=lambda index: (int(self.lengths[index]), int(index))
            )
        feature_dimension = int(self.store.feature_dimension)
        slabs = {}
        index_map = {}
        staging = []
        staging_bytes = 0
        bundle_bytes = 0
        with self.backend.stream_context(self.upload_stream):
            for group_key, bucket_indices in grouped.items():
                slab_length = max(
                    int(self.lengths[index]) for index in bucket_indices
                )
                slab = torch.zeros(
                    (
                        len(bucket_indices),
                        int(slab_length),
                        feature_dimension,
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                slabs[int(group_key)] = slab
                bundle_bytes += int(slab.numel()) * slab.element_size()
                for position, index in enumerate(bucket_indices):
                    host_embedding = self.store.get(index, self._group)
                    contiguous = np.ascontiguousarray(host_embedding)
                    host_tensor = torch.from_numpy(contiguous)
                    source_bytes = int(contiguous.nbytes)
                    use_pinned = bool(
                        self._pinned_transfers
                        and self._upload_staging_bytes
                        + self._host_pool.allocated_bytes
                        + staging_bytes
                        + source_bytes
                        <= self.host_staging_budget
                        and self._upload_staging_bytes
                        + self._host_pool.pinned_bytes
                        + staging_bytes
                        + source_bytes
                        <= self.pinned_staging_budget
                    )
                    if use_pinned:
                        try:
                            host_tensor = host_tensor.pin_memory()
                        except (RuntimeError, NotImplementedError):
                            use_pinned = False
                    true_length = int(self.lengths[index])
                    slab[position, :true_length].copy_(
                        host_tensor,
                        non_blocking=bool(
                            use_pinned and host_tensor.is_pinned()
                        ),
                    )
                    index_map[index] = (int(group_key), int(position))
                    if use_pinned and host_tensor.is_pinned():
                        staging.append(host_tensor)
                        staging_bytes += (
                            int(host_tensor.numel())
                            * host_tensor.element_size()
                        )
                    del host_embedding, contiguous
                # Normalize all true and padded residue rows in one operation.
                # Zero padding remains zero after the clamped division.
                norms = torch.linalg.vector_norm(
                    slab, ord=2, dim=-1, keepdim=True
                )
                slab.div_(
                    norms.clamp_min_(torch.finfo(torch.float32).tiny)
                )
            event = self.backend.create_event()
            self.backend.record_event(event, self.upload_stream)
        bundle = _EmbeddingSlabBundle(
            key=ordered,
            slabs=slabs,
            index_map=index_map,
            bytes=int(bundle_bytes),
        )
        capacity = max(1, int(self.plan.tile_cache_bytes))
        while (
            self._slab_cache
            and self._slab_cache_bytes + bundle.bytes > capacity
        ):
            _key, (evicted, _event) = self._slab_cache.popitem(last=False)
            self._slab_cache_bytes = max(
                0, self._slab_cache_bytes - int(evicted.bytes)
            )
        if bundle.bytes <= capacity:
            self._slab_cache[ordered] = (bundle, event)
            self._slab_cache_bytes += int(bundle.bytes)
        self._embedding_cache_bytes = int(self._slab_cache_bytes)
        if staging:
            self._upload_staging.append((event, staging, staging_bytes))
            self._upload_staging_bytes += int(staging_bytes)
            self._metrics.upload_queue_high_watermark = max(
                self._metrics.upload_queue_high_watermark,
                len(self._upload_staging),
            )
        self._metrics.upload_seconds += max(0.0, time.perf_counter() - started)
        self._metrics.upload_bytes += sum(
            int(self.store.float32_bytes[index]) for index in ordered
        )
        return bundle, event

    def _reset_workspaces(self, *, trim_allocator=False):
        if hasattr(self, "_metrics"):
            self._metrics.workspace_resets += 1
        return super()._reset_workspaces(trim_allocator=trim_allocator)

    def _prepare_packet_from_slabs(self, bundle, packet, workspace):
        """Gather one packet from two resident normalized slabs on device."""
        batch_size = len(packet.tasks)
        feature_dimension = int(self.store.feature_dimension)
        device = self.device

        def workspace_tensor(name, shape, dtype):
            tensor = workspace.get(name)
            if (
                tensor is None
                or tensor.dtype != dtype
                or len(tensor.shape) != len(shape)
                or any(
                    int(current) < int(required)
                    for current, required in zip(tensor.shape, shape)
                )
            ):
                tensor = torch.empty(shape, dtype=dtype, device=device)
                workspace[name] = tensor
            return tensor[
                tuple(slice(0, int(size)) for size in shape)
            ]

        def gather(name, indices, bucket):
            indices = tuple(int(index) for index in indices)
            group_keys = {
                int(bundle.index_map[index][0]) for index in indices
            }
            if len(group_keys) != 1:
                raise RuntimeError(
                    "One packet unexpectedly spans multiple length slabs."
                )
            group_key = next(iter(group_keys))
            positions = [
                int(bundle.index_map[index][1]) for index in indices
            ]
            position_tensor = workspace_tensor(
                f"{name}_indices", (batch_size,), torch.int64
            )
            position_tensor.copy_(
                torch.as_tensor(positions, dtype=torch.int64)
            )
            output = workspace_tensor(
                name,
                (batch_size, int(bucket), feature_dimension),
                torch.float32,
            )
            source = bundle.slabs[group_key][:, :int(bucket), :]
            if self._capabilities.get("index_select_out", True):
                try:
                    torch.index_select(
                        source, 0, position_tensor, out=output
                    )
                except (RuntimeError, NotImplementedError, TypeError):
                    self._capabilities["index_select_out"] = False
                    output.copy_(torch.index_select(source, 0, position_tensor))
            else:
                output.copy_(torch.index_select(source, 0, position_tensor))
            return output

        queries = gather(
            "queries",
            (int(task[0]) for task in packet.tasks),
            packet.query_bucket,
        )
        targets = gather(
            "targets",
            (int(task[1]) for task in packet.tasks),
            packet.target_bucket,
        )
        query_lengths = workspace_tensor(
            "query_lengths", (batch_size,), torch.int64
        )
        target_lengths = workspace_tensor(
            "target_lengths", (batch_size,), torch.int64
        )
        query_lengths.copy_(
            torch.as_tensor(
                packet.query_lengths, dtype=torch.int64
            )
        )
        target_lengths.copy_(
            torch.as_tensor(
                packet.target_lengths, dtype=torch.int64
            )
        )
        return queries, targets, query_lengths, target_lengths

    def _prepare_fixed_query_from_slabs(self, bundle, packet, workspace):
        """Return one query view and a gathered target batch."""
        query_index = int(packet.tasks[0][0])
        query_group, query_position = bundle.index_map[query_index]
        query = bundle.slabs[int(query_group)][
            int(query_position), :int(packet.query_lengths[0])
        ]
        target_indices = tuple(int(task[1]) for task in packet.tasks)
        target_groups = {
            int(bundle.index_map[index][0]) for index in target_indices
        }
        batch_size = len(target_indices)
        feature_dimension = int(self.store.feature_dimension)
        positions = workspace.get("target_indices")
        if positions is None or int(positions.numel()) < batch_size:
            positions = torch.empty(
                (batch_size,), dtype=torch.int64, device=self.device
            )
            workspace["target_indices"] = positions
        positions = positions[:batch_size]
        target_shape = (
            batch_size,
            int(packet.target_bucket),
            feature_dimension,
        )
        targets = workspace.get("targets")
        if (
            targets is None
            or len(targets.shape) != 3
            or any(
                int(current) < int(required)
                for current, required in zip(targets.shape, target_shape)
            )
        ):
            targets = torch.empty(
                target_shape, dtype=torch.float32, device=self.device
            )
            workspace["targets"] = targets
        targets = targets[
            :batch_size, :int(packet.target_bucket), :feature_dimension
        ]
        if len(target_groups) == 1:
            target_group = next(iter(target_groups))
            source = bundle.slabs[target_group][
                :, :int(packet.target_bucket), :
            ]
            host_positions = [
                int(bundle.index_map[index][1])
                for index in target_indices
            ]
            contiguous_start = int(host_positions[0])
            positions_are_contiguous = host_positions == list(
                range(contiguous_start, contiguous_start + batch_size)
            )
            if positions_are_contiguous:
                targets.copy_(
                    source[
                        contiguous_start:contiguous_start + batch_size
                    ]
                )
            else:
                positions.copy_(
                    torch.as_tensor(
                        host_positions,
                        dtype=torch.int64,
                    )
                )
            if (
                not positions_are_contiguous
                and self._capabilities.get("index_select_out", True)
            ):
                try:
                    torch.index_select(source, 0, positions, out=targets)
                except (RuntimeError, NotImplementedError, TypeError):
                    self._capabilities["index_select_out"] = False
                    targets.copy_(torch.index_select(source, 0, positions))
            elif not positions_are_contiguous:
                targets.copy_(torch.index_select(source, 0, positions))
        else:
            # Row-safe packets intentionally preserve the established
            # microbatch boundaries, which can span several normalized length
            # slabs. Gather once per slab instead of issuing one Python copy
            # per target.
            targets.zero_()
            grouped_positions = OrderedDict()
            for destination, index in enumerate(target_indices):
                group, position = bundle.index_map[index]
                grouped_positions.setdefault(int(group), []).append(
                    (int(destination), int(position))
                )
            for target_group, group_entries in grouped_positions.items():
                group_size = len(group_entries)
                source = bundle.slabs[target_group]
                source_width = min(
                    int(source.shape[1]), int(packet.target_bucket)
                )
                destinations = [entry[0] for entry in group_entries]
                destination_start = int(destinations[0])
                if destinations != list(
                    range(destination_start, destination_start + group_size)
                ):
                    raise RuntimeError(
                        "A row-safe target slab is not contiguous."
                    )
                selected = targets[
                    destination_start:destination_start + group_size,
                    :source_width,
                    :feature_dimension,
                ]
                source_positions = [entry[1] for entry in group_entries]
                source_start = int(source_positions[0])
                positions_are_contiguous = source_positions == list(
                    range(source_start, source_start + group_size)
                )
                if positions_are_contiguous:
                    selected.copy_(
                        source[
                            source_start:source_start + group_size,
                            :source_width,
                            :,
                        ]
                    )
                else:
                    positions[:group_size].copy_(
                        torch.as_tensor(
                            source_positions,
                            dtype=torch.int64,
                        )
                    )
                if (
                    not positions_are_contiguous
                    and self._capabilities.get("index_select_out", True)
                ):
                    try:
                        torch.index_select(
                            source[:, :source_width, :],
                            0,
                            positions[:group_size],
                            out=selected,
                        )
                    except (RuntimeError, NotImplementedError, TypeError):
                        self._capabilities["index_select_out"] = False
                        selected.copy_(
                            torch.index_select(
                                source[:, :source_width, :],
                                0,
                                positions[:group_size],
                            )
                        )
                elif not positions_are_contiguous:
                    selected.copy_(
                        torch.index_select(
                            source[:, :source_width, :],
                            0,
                            positions[:group_size],
                        )
                    )
        target_lengths = workspace.get("target_lengths")
        if target_lengths is None or int(target_lengths.numel()) < batch_size:
            target_lengths = torch.empty(
                (batch_size,), dtype=torch.int64, device=self.device
            )
            workspace["target_lengths"] = target_lengths
        target_lengths = target_lengths[:batch_size]
        target_lengths.copy_(
            torch.as_tensor(
                packet.target_lengths,
                dtype=torch.int64,
            )
        )
        return query, targets, target_lengths

    def _compiled_scorer(self, shape_family, *, fixed_query=False):
        cache_key = (bool(fixed_query), tuple(shape_family))
        if (
            not self._enable_compilation
            or self.execution_plan.scorer_variant
            not in {"compiled", "compiled_graph"}
            or not self.backend.supports_compilation()
            or cache_key in self._compiled_failures
        ):
            return None
        if cache_key in self._compiled_scorers:
            scorer = self._compiled_scorers.pop(cache_key)
            self._compiled_scorers[cache_key] = scorer
            return scorer
        shape_limit = (
            MAX_CUDA_GRAPH_SHAPE_FAMILIES
            if self.execution_plan.scorer_variant == "compiled_graph"
            else MAX_COMPILED_SHAPE_FAMILIES
        )
        if len(self._compiled_scorers) >= shape_limit:
            return None
        if (
            self.execution_plan.scorer_variant == "compiled_graph"
            and not self.backend.supports_graph_compilation()
        ):
            self._compiled_failures.add(cache_key)
            return None
        started = time.perf_counter()
        try:
            scorer = self.backend.compile_callable(
                (
                    _score_fixed_query_tensors
                    if fixed_query
                    else _score_padded_pair_tensors
                ),
                dynamic=False,
                mode=(
                    "reduce-overhead"
                    if self.execution_plan.scorer_variant == "compiled_graph"
                    else "default"
                ),
            )
        except (RuntimeError, NotImplementedError, TypeError, ValueError):
            self._compiled_failures.add(cache_key)
            self._metrics.compiled_fallbacks += 1
            return None
        finally:
            self._metrics.compilation_seconds += max(
                0.0, time.perf_counter() - started
            )
        self._compiled_scorers[cache_key] = scorer
        self._compiled_new_shapes.add(cache_key)
        return scorer

    def _run_alignment_chunk(self, items):
        started = time.perf_counter()
        scratch_growths = 0
        if self._alignment_chunk_callback is None:
            values = [
                (ordinal, self.alignment_callback((idx_i, idx_j, matrix)))
                for ordinal, idx_i, idx_j, matrix in items
            ]
        else:
            chunk_result = self._alignment_chunk_callback(items)
            if (
                isinstance(chunk_result, tuple)
                and len(chunk_result) == 2
                and isinstance(chunk_result[1], (int, np.integer))
            ):
                values, scratch_growths = chunk_result
            else:
                values = chunk_result
        return (
            list(values),
            max(0.0, time.perf_counter() - started),
            int(scratch_growths),
        )

    def run(
        self,
        tasks,
        *,
        progress=None,
        result_callback=None,
        result_chunk_size=65536,
        matrix_budget_override=None,
        batch_id=0,
    ):
        """Run one publication batch using bounded multi-row packets."""
        if self._closed:
            raise RuntimeError("The tiled accelerator session is closed.")
        tasks = tasks if isinstance(tasks, Sequence) else tuple(tasks)
        if not tasks:
            return []
        started = time.perf_counter()
        starting_pairs = int(self._metrics.pairs)
        matrix_budget = max(
            1,
            min(
                int(matrix_budget_override or self.matrix_budget),
                int(self.matrix_budget),
            ),
        )
        packet_budget = max(
            1,
            min(
                matrix_budget // self.lanes,
                int(self.execution_plan.microbatch_workspace_bytes),
            ),
        )
        # Length slabs use their true maximum length rather than a fixed 15%
        # pad. Actual slab bytes are enforced by the LRU and the device
        # high-watermark, so shrinking every block by the worst-case padding
        # ceiling only creates extra uploads on normally tight buckets.
        per_block = max(1, int(self.plan.tile_cache_bytes) // 2)
        block_ids = self.store.block_ids(per_block)
        tile_groups = OrderedDict()
        for ordinal, task in enumerate(tasks):
            key = (
                int(block_ids[int(task[0])]),
                int(block_ids[int(task[1])]),
            )
            tile_groups.setdefault(key, array("Q")).append(ordinal)

        inflight = deque()
        cpu_pending = {}
        ready_results = {}
        returned_results = []
        emit_buffer = []
        next_ordinal = 0
        cpu_pending_bytes = 0
        max_inflight = max(2, self.lanes * 2)
        cpu_pending_limit = max(
            1, int(self.execution_plan.active_cpu_workers) * 4
        )
        finite_buffers = deque()
        for _slot in range(max_inflight):
            try:
                finite_buffers.append(
                    torch.empty(
                        (),
                        dtype=torch.bool,
                        device="cpu",
                        pin_memory=bool(self._pinned_transfers),
                    )
                )
            except (RuntimeError, NotImplementedError):
                finite_buffers.append(
                    torch.empty((), dtype=torch.bool, device="cpu")
                )

        def emit_ready():
            nonlocal next_ordinal
            while next_ordinal in ready_results:
                result = ready_results.pop(next_ordinal)
                next_ordinal += 1
                self._metrics.pairs += 1
                if progress is not None:
                    progress.update(1)
                if result_callback is None:
                    returned_results.append(result)
                else:
                    emit_buffer.append(result)
                    if len(emit_buffer) >= max(1, int(result_chunk_size)):
                        writer_started = time.perf_counter()
                        result_callback(emit_buffer)
                        self._metrics.writer_seconds += max(
                            0.0, time.perf_counter() - writer_started
                        )
                        emit_buffer.clear()

        def collect_cpu(block=False):
            nonlocal cpu_pending_bytes
            if not cpu_pending:
                return False
            if block:
                wait_started = time.perf_counter()
                completed, _ = wait(
                    tuple(cpu_pending), return_when=FIRST_COMPLETED
                )
                self._metrics.cpu_queue_stall_seconds += max(
                    0.0, time.perf_counter() - wait_started
                )
            else:
                completed = {
                    future for future in cpu_pending if future.done()
                }
            for future in completed:
                lease = cpu_pending.pop(future)
                values, elapsed, scratch_growths = future.result()
                self._metrics.cpu_alignment_seconds += float(elapsed)
                self._metrics.scratch_growths += int(scratch_growths)
                for ordinal, result in values:
                    ready_results[int(ordinal)] = result
                if lease.release_reference():
                    cpu_pending_bytes = max(
                        0, cpu_pending_bytes - int(lease.bytes)
                    )
                emit_ready()
            return bool(completed)

        def submit_flight(flight, block):
            nonlocal cpu_pending_bytes
            if not block and not flight["event"].query():
                return False
            if block:
                flight["event"].synchronize()
            score_elapsed = self.backend.elapsed_time(
                flight["score_start"], flight["score_end"]
            )
            download_elapsed = self.backend.elapsed_time(
                flight["score_end"], flight["download_end"]
            )
            upload_wait_elapsed = self.backend.elapsed_time(
                flight["wait_start"], flight["score_start"]
            )
            if score_elapsed is not None:
                self._metrics.gpu_score_seconds += score_elapsed
            if download_elapsed is not None:
                self._metrics.download_seconds += download_elapsed
            if upload_wait_elapsed is not None:
                self._metrics.upload_wait_seconds += upload_wait_elapsed
            finite_host = flight["finite_host"]
            finite_value = bool(finite_host.item())
            finite_buffers.append(finite_host)
            if not finite_value:
                packet = flight["packet"]
                flight["lease"].release_all()
                raise FloatingPointError(
                    "Accelerator packet produced non-finite values for "
                    f"batch {packet.batch_id}, ordinals "
                    f"{packet.ordinals[0]}..{packet.ordinals[-1]}."
                )

            packet = flight["packet"]
            matrix_array = flight["lease"].tensor.numpy()
            chunks = []
            current = []
            current_bytes = 0
            for offset, (task, ordinal, query_length, target_length) in enumerate(
                zip(
                    packet.tasks,
                    packet.ordinals,
                    packet.query_lengths,
                    packet.target_lengths,
                )
            ):
                matrix = matrix_array[
                    offset,
                    :int(query_length),
                    :int(target_length),
                ]
                item = (
                    int(ordinal),
                    int(task[0]),
                    int(task[1]),
                    matrix,
                )
                item_bytes = int(matrix.nbytes)
                if current and (
                    len(current) >= int(self.execution_plan.cpu_chunk_size)
                    or current_bytes + item_bytes > MAX_CPU_CHUNK_BYTES
                ):
                    chunks.append(current)
                    current = []
                    current_bytes = 0
                current.append(item)
                current_bytes += item_bytes
            if current:
                chunks.append(current)
            lease = flight["lease"]
            lease.references = len(chunks)
            cpu_pending_bytes += int(lease.bytes)
            self._metrics.peak_cpu_pending_bytes = max(
                self._metrics.peak_cpu_pending_bytes, cpu_pending_bytes
            )
            for chunk in chunks:
                while len(cpu_pending) >= cpu_pending_limit:
                    collect_cpu(block=True)
                future = self._cpu_executor.submit(
                    self._run_alignment_chunk, chunk
                )
                cpu_pending[future] = lease
            self._metrics.peak_cpu_pending_tasks = max(
                self._metrics.peak_cpu_pending_tasks, len(cpu_pending)
            )
            return True

        def submit_oldest(block):
            if not inflight:
                return False
            flight = None
            if not block:
                for candidate in inflight:
                    if candidate["event"].query():
                        flight = candidate
                        break
                if flight is None:
                    return False
            else:
                flight = inflight[0]
            if submit_flight(flight, block):
                inflight.remove(flight)
                return True
            return False

        def acquire_host_buffer(packet):
            pin_output = bool(
                self._pinned_transfers
                and self._upload_staging_bytes
                + self._host_pool.pinned_bytes
                + int(packet.output_bytes)
                <= self.pinned_staging_budget
            )
            shape = (
                len(packet.tasks),
                int(packet.query_bucket),
                int(packet.target_bucket),
            )
            while True:
                tensor, pooled = self._host_pool.acquire(
                    shape,
                    pin_memory=pin_output,
                    capacity_bytes=max(
                        1,
                        self.host_staging_budget
                        - int(self._upload_staging_bytes),
                    ),
                )
                if tensor is not None:
                    return tensor, pooled
                if inflight:
                    submit_oldest(block=True)
                    collect_cpu(block=False)
                    continue
                if cpu_pending:
                    collect_cpu(block=True)
                    continue
                raise MemoryError(
                    "One accelerator result packet exceeds the aggregate "
                    "host-staging budget."
                )

        precision_context = (
            cuda_matmul_precision(self.precision)
            if self.backend.device_type == "cuda"
            else nullcontext()
        )
        prefetched_tiles = {}
        try:
            with precision_context, torch.inference_mode(), \
                    self.backend.device_context():
                tile_items = list(tile_groups.values())
                for tile_position, tile_ordinals in enumerate(tile_items):
                    tile_tasks = _OrdinalTaskView(tasks, tile_ordinals)
                    self._metrics.tiles += 1
                    # Bound live embedding bundles to current plus lookahead.
                    # This matters for sparse tiles that contain fewer packets
                    # than the normal in-flight depth.
                    while len(
                        {
                            id(flight["embedding_lease"][0])
                            for flight in inflight
                        }
                    ) >= 2:
                        boundary_started = time.perf_counter()
                        submit_oldest(block=True)
                        collect_cpu(block=False)
                        self._metrics.tile_boundary_stall_seconds += max(
                            0.0, time.perf_counter() - boundary_started
                        )
                    tile_indices = {int(task[0]) for task in tile_tasks}
                    tile_indices.update(int(task[1]) for task in tile_tasks)
                    if tile_position in prefetched_tiles:
                        device_embeddings, preload_event = prefetched_tiles.pop(
                            tile_position
                        )
                    else:
                        device_embeddings, preload_event = self._load_tile(
                            tile_indices
                        )

                    next_tile_indices = None
                    if tile_position + 1 < len(tile_items):
                        next_tasks = _OrdinalTaskView(
                            tasks, tile_items[tile_position + 1]
                        )
                        candidate_indices = {
                            int(task[0]) for task in next_tasks
                        }
                        candidate_indices.update(
                            int(task[1]) for task in next_tasks
                        )
                        union_bytes = sum(
                            self.store.float32_bytes[index]
                            for index in tile_indices | candidate_indices
                        )
                        if int(union_bytes * 1.15) <= int(
                            self.plan.tile_cache_bytes
                        ):
                            next_tile_indices = candidate_indices
                    prefetch_started = False

                    if (
                        self.execution_plan.length_bucket_policy
                        == "row_safe"
                    ):
                        packets = iter_row_safe_work_packets(
                            tile_tasks,
                            tile_ordinals,
                            self.lengths,
                            packet_budget,
                            self.store.feature_dimension,
                            batch_id=batch_id,
                        )
                    else:
                        packets = iter_pair_work_packets(
                            tile_tasks,
                            self.lengths,
                            packet_budget,
                            self.store.feature_dimension,
                            batch_id=batch_id,
                            length_bucket_quantum=(
                                self.execution_plan.length_bucket_quantum
                            ),
                            task_ordinals=tile_ordinals,
                            multirow=True,
                        )
                    for packet in schedule_pair_work_packets(
                        packets, self.lanes
                    ):
                        if int(packet.workspace_bytes) > packet_budget:
                            raise MemoryError(
                                "One length-bucketed pair exceeds the selected "
                                "microbatch workspace limit."
                            )
                        self._metrics.packets += 1
                        self._metrics.padded_cells += int(packet.padded_cells)
                        self._metrics.real_cells += int(packet.real_cells)
                        self._metrics.estimated_flops += int(
                            packet.estimated_flops
                        )
                        shape_key = str(packet.shape_family)
                        self._metrics.packet_shapes[shape_key] = (
                            int(self._metrics.packet_shapes.get(shape_key, 0))
                            + 1
                        )
                        while (
                            inflight
                            and sum(
                                int(item["workspace_bytes"])
                                for item in inflight
                            ) + int(packet.workspace_bytes) > matrix_budget
                        ):
                            submit_oldest(block=True)
                            collect_cpu(block=False)

                        stream_index = self._stream_cursor % self.lanes
                        stream = self.compute_streams[stream_index]
                        self._stream_cursor += 1
                        if self._workspace_requires_reset(
                            stream_index,
                            packet.workspace_bytes,
                            matrix_budget,
                        ):
                            boundary_started = time.perf_counter()
                            while inflight:
                                submit_oldest(block=True)
                            self._metrics.tile_boundary_stall_seconds += max(
                                0.0, time.perf_counter() - boundary_started
                            )
                            self._reset_workspaces(trim_allocator=True)
                        workspace = self._prepare_workspace(
                            stream_index,
                            packet.workspace_bytes,
                            matrix_budget,
                        )
                        host_tensor, pooled = acquire_host_buffer(packet)
                        while not finite_buffers:
                            submit_oldest(block=True)
                            collect_cpu(block=False)
                        finite_host = finite_buffers.popleft()
                        lease = _HostBufferLease(
                            self._host_pool, host_tensor, pooled, 0
                        )
                        with self.backend.stream_context(stream):
                            wait_start = self.backend.create_event(
                                enable_timing=True
                            )
                            self.backend.record_event(wait_start, stream)
                            if preload_event is not None:
                                self.backend.wait_event(stream, preload_event)
                            score_start = self.backend.create_event(
                                enable_timing=True
                            )
                            self.backend.record_event(score_start, stream)
                            fixed_query = len(
                                {int(task[0]) for task in packet.tasks}
                            ) == 1
                            compiled = self._compiled_scorer(
                                packet.shape_family,
                                fixed_query=fixed_query,
                            )
                            if compiled is not None:
                                prepared = (
                                    self._prepare_fixed_query_from_slabs(
                                        device_embeddings, packet, workspace
                                    )
                                    if fixed_query
                                    else self._prepare_packet_from_slabs(
                                        device_embeddings, packet, workspace
                                    )
                                )
                                compiled_key = (
                                    bool(fixed_query),
                                    tuple(packet.shape_family),
                                )
                                try:
                                    compiled_started = time.perf_counter()
                                    matrices = compiled(*prepared)
                                    if (
                                        compiled_key
                                        in self._compiled_new_shapes
                                    ):
                                        self.backend.synchronize()
                                        self._metrics.compilation_seconds += max(
                                            0.0,
                                            time.perf_counter()
                                            - compiled_started,
                                        )
                                        self._compiled_new_shapes.discard(
                                            compiled_key
                                        )
                                    self._metrics.compiled_packets += 1
                                except Exception as error:
                                    if self.backend.is_out_of_memory(error):
                                        lease.release_all()
                                        raise
                                    self._compiled_failures.add(
                                        compiled_key
                                    )
                                    self._compiled_scorers.pop(
                                        compiled_key, None
                                    )
                                    self._compiled_new_shapes.discard(
                                        compiled_key
                                    )
                                    self._metrics.compiled_fallbacks += 1
                                    if fixed_query:
                                        matrices = (
                                            _eager_fixed_query_score_tensors(
                                                *prepared,
                                                workspace=workspace,
                                                capabilities=self._capabilities,
                                            )
                                        )
                                    else:
                                        matrices = (
                                            _eager_score_padded_pair_tensors(
                                                *prepared,
                                                workspace=workspace,
                                                capabilities=self._capabilities,
                                                equal_query_lengths=(
                                                    len(set(packet.query_lengths))
                                                    == 1
                                                ),
                                            )
                                        )
                                    self._metrics.eager_packets += 1
                            else:
                                if fixed_query:
                                    fixed_prepared = (
                                        self._prepare_fixed_query_from_slabs(
                                            device_embeddings,
                                            packet,
                                            workspace,
                                        )
                                    )
                                    matrices = (
                                        _eager_fixed_query_score_tensors(
                                            *fixed_prepared,
                                            workspace=workspace,
                                            capabilities=self._capabilities,
                                        )
                                    )
                                else:
                                    prepared = (
                                        self._prepare_packet_from_slabs(
                                            device_embeddings,
                                            packet,
                                            workspace,
                                        )
                                    )
                                    matrices = (
                                        _eager_score_padded_pair_tensors(
                                            *prepared,
                                            workspace=workspace,
                                            capabilities=self._capabilities,
                                            equal_query_lengths=True,
                                        )
                                    )
                                self._metrics.eager_packets += 1
                            score_end = self.backend.create_event(
                                enable_timing=True
                            )
                            self.backend.record_event(score_end, stream)
                            host_tensor.copy_(
                                matrices,
                                non_blocking=bool(
                                    self._pinned_transfers
                                    and host_tensor.is_pinned()
                                ),
                            )
                            finite_host.copy_(
                                torch.isfinite(matrices).all(),
                                non_blocking=bool(
                                    self._pinned_transfers
                                    and finite_host.is_pinned()
                                ),
                            )
                            download_end = self.backend.create_event(
                                enable_timing=True
                            )
                            self.backend.record_event(download_end, stream)
                        inflight.append(
                            {
                                "event": download_end,
                                "wait_start": wait_start,
                                "score_start": score_start,
                                "score_end": score_end,
                                "download_end": download_end,
                                "finite_host": finite_host,
                                "packet": packet,
                                "lease": lease,
                                "workspace_bytes": int(packet.workspace_bytes),
                                # Keep embedding tensors live until all stream
                                # operations that reference them have finished.
                                "embedding_lease": (
                                    device_embeddings,
                                ),
                            }
                        )
                        self._metrics.download_bytes += int(packet.output_bytes)
                        self._metrics.peak_inflight_packets = max(
                            self._metrics.peak_inflight_packets, len(inflight)
                        )
                        self._metrics.result_queue_high_watermark = max(
                            self._metrics.result_queue_high_watermark,
                            len(inflight),
                        )
                        self._metrics.peak_host_staging_bytes = max(
                            self._metrics.peak_host_staging_bytes,
                            int(self._upload_staging_bytes)
                            + int(self._host_pool.allocated_bytes),
                        )
                        self._metrics.peak_pinned_staging_bytes = max(
                            self._metrics.peak_pinned_staging_bytes,
                            int(self._upload_staging_bytes)
                            + int(self._host_pool.pinned_bytes),
                        )

                        if (
                            not prefetch_started
                            and next_tile_indices is not None
                        ):
                            prefetched_tiles[tile_position + 1] = (
                                self._load_tile(next_tile_indices)
                            )
                            prefetch_started = True
                            self._metrics.prefetched_tiles += 1
                        while len(inflight) >= max_inflight:
                            submit_oldest(block=True)
                            collect_cpu(block=False)
                        submit_oldest(block=False)
                        collect_cpu(block=False)

                        if (
                            self.backend.memory_reserved()
                            > self.device_high_watermark
                        ):
                            while inflight:
                                submit_oldest(block=True)
                            self._trim_allocator_if_needed(force=True)
                            if (
                                self.backend.memory_allocated()
                                > self.device_high_watermark
                            ):
                                raise MemoryError(
                                    "Accelerator allocation exceeded its 80% "
                                    "device-memory high-watermark."
                                )

                    # Do not drain at tile boundaries.  Embedding leases keep
                    # current slabs alive while the next upload begins.
                    self._collect_upload_staging(block=False)

                while inflight:
                    submit_oldest(block=True)
                    collect_cpu(block=False)
                boundary_started = time.perf_counter()
                while cpu_pending:
                    collect_cpu(block=True)
                self._metrics.batch_boundary_stall_seconds += max(
                    0.0, time.perf_counter() - boundary_started
                )
                self._collect_upload_staging(block=True)
        except Exception:
            for flight in inflight:
                flight["lease"].release_all()
            inflight.clear()
            if cpu_pending:
                wait(tuple(cpu_pending))
                for future, lease in tuple(cpu_pending.items()):
                    try:
                        future.result()
                    finally:
                        lease.release_reference()
                cpu_pending.clear()
            self._metrics.pairs = starting_pairs
            self._metrics.elapsed_seconds += max(
                0.0, time.perf_counter() - started
            )
            raise

        emit_ready()
        if next_ordinal != len(tasks):
            raise RuntimeError(
                f"Accelerator pipeline completed {next_ordinal} ordered "
                f"results for {len(tasks)} requested pairs."
            )
        if result_callback is not None and emit_buffer:
            writer_started = time.perf_counter()
            result_callback(emit_buffer)
            self._metrics.writer_seconds += max(
                0.0, time.perf_counter() - writer_started
            )
            emit_buffer.clear()
        self._metrics.elapsed_seconds += max(
            0.0, time.perf_counter() - started
        )
        return returned_results

    def run_batch_stream(
        self,
        batches,
        *,
        progress=None,
        result_callbacks=None,
        result_chunk_size=65536,
    ):
        """Run at most two publication batches in each bounded lookahead window.

        ``batches`` contains ``(batch_id, tasks)`` pairs.  Optional callbacks
        may be a mapping keyed by batch ID.  Results are always routed in batch
        and ordinal order even though accelerator packets span both batches.
        """
        iterator = iter(batches)
        callbacks = result_callbacks or {}
        while True:
            window = []
            for _ in range(2):
                try:
                    window.append(next(iterator))
                except StopIteration:
                    break
            if not window:
                return
            sequences = []
            boundaries = []
            combined_count = 0
            for current_batch_id, current_tasks in window:
                current_tasks = (
                    current_tasks
                    if isinstance(current_tasks, Sequence)
                    else tuple(current_tasks)
                )
                start = combined_count
                sequences.append(current_tasks)
                combined_count += len(current_tasks)
                boundaries.append(
                    (int(current_batch_id), start, combined_count)
                )
            combined = _CombinedTaskView(sequences)
            routed = {batch_id: [] for batch_id, _start, _end in boundaries}
            emitted = 0

            def route(results):
                nonlocal emitted
                cursor = 0
                while cursor < len(results):
                    absolute = emitted + cursor
                    for current_batch_id, start, end in boundaries:
                        if start <= absolute < end:
                            take = min(len(results) - cursor, end - absolute)
                            chunk = results[cursor:cursor + take]
                            callback = callbacks.get(current_batch_id)
                            if callback is None:
                                routed[current_batch_id].extend(chunk)
                            else:
                                callback(chunk)
                            cursor += take
                            break
                    else:
                        raise RuntimeError("Batch-stream result routing overflow.")
                emitted += len(results)

            self.run(
                combined,
                progress=progress,
                result_callback=route,
                result_chunk_size=result_chunk_size,
                batch_id=boundaries[0][0],
            )
            for current_batch_id, _start, _end in boundaries:
                yield current_batch_id, routed[current_batch_id]

    def recover_from_oom(self, minimum_budget=1024 ** 2):
        self._metrics.oom_retries += 1
        self._host_pool.clear()
        self._slab_cache.clear()
        self._slab_cache_bytes = 0
        budget = super().recover_from_oom(minimum_budget=minimum_budget)
        self.execution_plan = replace(
            self.execution_plan,
            microbatch_workspace_bytes=max(
                1,
                min(
                    int(self.execution_plan.microbatch_workspace_bytes) // 2,
                    int(budget) // self.lanes,
                ),
            ),
        )
        self.microbatch_workspace_limit = int(
            self.execution_plan.microbatch_workspace_bytes
        )
        self._host_pool = _HostBufferPool(
            self.host_staging_budget,
            max_free_buffers=max(2, self.lanes),
        )
        return budget

    def metrics(self):
        values = self._metrics.as_dict()
        elapsed = max(float(values["elapsed_seconds"]), 1e-9)
        values.update(
            {
                "backend": self.backend.device_type,
                "device": self.backend.display_name,
                "execution_plan": asdict(self.execution_plan),
                "matrix_budget_bytes": int(self.matrix_budget),
                "host_staging_budget_bytes": int(self.host_staging_budget),
                "pinned_staging_budget_bytes": int(
                    self.pinned_staging_budget
                ),
                "device_high_watermark_bytes": int(
                    self.device_high_watermark
                ),
                "embedding_cache_bytes": int(self._embedding_cache_bytes),
                "peak_allocated_bytes": self.backend.max_memory_allocated(),
                "peak_reserved_bytes": self.backend.max_memory_reserved(),
                "utilization_percent": self.backend.utilization(),
                "throughput_pairs_per_second": (
                    int(values["pairs"]) / elapsed
                ),
            }
        )
        return values

    def close(self):
        if self._closed:
            return
        if (
            self._print_summary
            and self._metrics.pairs
            and not self._summary_printed
        ):
            values = self.metrics()
            print(
                f"[Accelerator] {values['pairs']} pairs at "
                f"{values['throughput_pairs_per_second']:.1f} pairs/s; "
                f"{values['bottleneck']}; "
                f"{values['packets']} packets, "
                f"{values['padding_fraction'] * 100.0:.1f}% padding, "
                f"peak device {values['peak_allocated_bytes'] / MIB:.0f} MiB."
            )
            self._summary_printed = True
        self._host_pool.clear()
        self._slab_cache.clear()
        self._slab_cache_bytes = 0
        self._compiled_scorers.clear()
        self._compiled_new_shapes.clear()
        super().close()


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
    session=None,
    execution_plan=None,
    alignment_chunk_callback=None,
    batch_id=0,
    print_summary=True,
):
    """Run tiled scoring through a persistent or one-shot accelerator session."""
    if session is not None:
        if torch.device(device) != session.device:
            raise ValueError("Tiled session device does not match the requested device.")
        return session.run(
            tasks,
            progress=progress,
            result_callback=result_callback,
            result_chunk_size=result_chunk_size,
            matrix_budget_override=matrix_budget_override,
            batch_id=batch_id,
        )
    with TiledAcceleratorSession(
        store=store,
        lengths=lengths,
        device=device,
        workers=workers,
        lanes=lanes,
        alignment_callback=alignment_callback,
        precision=precision,
        matrix_budget_override=matrix_budget_override,
        memory_plan_override=memory_plan_override,
        execution_plan=execution_plan,
        alignment_chunk_callback=alignment_chunk_callback,
        print_summary=print_summary,
    ) as owned_session:
        return owned_session.run(
            tasks,
            progress=progress,
            result_callback=result_callback,
            result_chunk_size=result_chunk_size,
            batch_id=batch_id,
        )


def run_tiled_accelerator_batch_stream(
    batches,
    *,
    session,
    writer_factory,
    progress=None,
    result_chunk_size=65536,
):
    """Process and atomically publish bounded two-batch lookahead windows.

    A failed window never publishes either batch.  Its partial writers are
    rolled back, closed, and removed so the caller can replay the earliest
    uncommitted batch with a recovered session or the next ranked plan.
    """
    iterator = iter(batches)
    while True:
        window = []
        for _ in range(2):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            return
        writers = {
            int(batch_id): writer_factory(int(batch_id))
            for batch_id, _tasks in window
        }
        try:
            list(
                session.run_batch_stream(
                    window,
                    progress=progress,
                    result_callbacks=writers,
                    result_chunk_size=result_chunk_size,
                )
            )
            for batch_id, tasks in window:
                writer = writers[int(batch_id)]
                if int(writer.count) != len(tasks):
                    raise RuntimeError(
                        f"Batch {batch_id} produced {writer.count} results "
                        f"for {len(tasks)} requested pairs."
                    )
            for batch_id, _tasks in window:
                writers[int(batch_id)].publish()
        except Exception:
            for writer in writers.values():
                try:
                    writer.rollback(0)
                except (OSError, RuntimeError, ValueError):
                    pass
                writer.close()
                partial = getattr(writer, "partial_filename", None)
                if partial and os.path.exists(partial):
                    try:
                        os.remove(partial)
                    except OSError:
                        pass
            raise
        finally:
            for writer in writers.values():
                writer.close()
        for batch_id, _tasks in window:
            yield int(batch_id)


def run_tiled_cuda_pipeline(*args, **kwargs):
    """Compatibility wrapper for the backend-neutral tiled pipeline."""
    return run_tiled_accelerator_pipeline(*args, **kwargs)


def _fixed_query_task_index(task):
    return int(task[0])


def estimate_fixed_query_accelerator_working_set(
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
    """Estimate a fixed-query tiled search without allocating device tensors."""
    plan = accelerator_memory_plan(device, lanes=lanes, memory_info=memory_info)
    plan = replace(
        plan,
        matrix_bytes=max(1, plan.matrix_pool_bytes // plan.inflight_slots),
    )
    tasks = list(tasks)
    baseline = max(0, plan.total_bytes - plan.free_bytes)
    safe_peak = max(0, plan.total_bytes - plan.reserve_bytes)
    query = np.asarray(query_embedding)
    query_length = int(query.shape[0])
    query_bytes = int(query.size) * 4
    if not tasks:
        return AcceleratorWorkloadEstimate(
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
        reason = "projected peak exceeds the reserved device-memory boundary"
    else:
        reason = "within reserved device-memory boundary"
    return AcceleratorWorkloadEstimate(
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


def estimate_fixed_query_cuda_working_set(*args, **kwargs):
    """Compatibility wrapper for the backend-neutral fixed-query estimate."""
    return estimate_fixed_query_accelerator_working_set(*args, **kwargs)


def run_fixed_query_accelerator_pipeline(
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
    backend = get_accelerator_backend(device)
    supported, reason = backend.supports_tiled(require_memory=True)
    if not supported:
        raise RuntimeError(
            f"Fixed-query tiled execution is unavailable on {device}: {reason}."
        )
    tasks = list(tasks)
    if not tasks:
        return []

    plan = accelerator_memory_plan(device, lanes=lanes)
    plan = replace(
        plan,
        matrix_bytes=max(1, plan.matrix_pool_bytes // plan.inflight_slots),
    )
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
    streams = [backend.create_stream() for _ in range(max(1, int(lanes)))]
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

    precision_context = (
        cuda_matmul_precision(precision)
        if backend.device_type == "cuda"
        else nullcontext()
    )
    with precision_context, torch.inference_mode(), \
            h5py.File(store.path, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=max(1, int(workers)),
                thread_name_prefix="search-alignment-cpu",
            ) as cpu_executor:
        preload_stream = streams[0]
        with backend.stream_context(preload_stream):
            query_tensor = _to_normalized_accelerator(query_array, device)[0]
            query_event = backend.create_event()
            backend.record_event(query_event, preload_stream)
        query_event.synchronize()
        group = hf["embeddings"]

        for block_tasks in grouped.values():
            indices = {task_index(task) for task in block_tasks}
            host_embeddings = store.load_indices(indices, group)
            preload_stream = streams[stream_cursor % len(streams)]
            with backend.stream_context(preload_stream):
                gpu_embeddings = {
                    index: _to_normalized_accelerator(array, device)[0]
                    for index, array in host_embeddings.items()
                }
                preload_event = backend.create_event()
                backend.record_event(preload_event, preload_stream)
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
                with backend.stream_context(stream):
                    matrices = _batched_score_matrices(
                        query_tensor,
                        target_tensors,
                        target_lengths,
                    )
                    host_tensor = _host_output_buffer(matrices)
                    host_tensor.copy_(matrices, non_blocking=host_tensor.is_pinned())
                    event = backend.create_event()
                    backend.record_event(event, stream)
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
            # Retain allocator reservations between tiles; release only during
            # OOM recovery or final session shutdown.

        while cpu_pending:
            collect_cpu(block=True)
    return results


def run_fixed_query_cuda_pipeline(*args, **kwargs):
    """Compatibility wrapper for the backend-neutral fixed-query pipeline."""
    return run_fixed_query_accelerator_pipeline(*args, **kwargs)


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
