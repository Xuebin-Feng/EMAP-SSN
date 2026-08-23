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

from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
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
        if self.scorer_variant not in {"eager", "compiled"}:
            raise ValueError("Scorer variant must be eager or compiled.")
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
    peak_cpu_pending_tasks: int = 0
    peak_cpu_pending_bytes: int = 0
    peak_host_staging_bytes: int = 0
    peak_pinned_staging_bytes: int = 0
    scratch_growths: int = 0
    compiled_packets: int = 0
    eager_packets: int = 0
    compiled_fallbacks: int = 0

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


def microbatch_workspace_candidates(matrix_pool_bytes, lanes):
    """Return unique bounded workspace candidates for one lane count."""
    per_lane = max(1, int(matrix_pool_bytes) // max(1, int(lanes)))
    candidates = [
        min(per_lane, int(candidate))
        for candidate in MICROBATCH_WORKSPACE_CANDIDATES
    ]
    candidates.append(per_lane)
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


def _pair_packet_workspace_bytes(
    count,
    query_bucket,
    target_bucket,
    feature_dimension,
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
    embedding_bytes = (
        count
        * (query_bucket + target_bucket)
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
):
    """Lazily pack multiple query rows into bounded length-bucketed packets."""
    workspace_budget = max(1, int(workspace_budget))
    buckets = OrderedDict()
    for ordinal, task in enumerate(tasks):
        idx_i, idx_j = int(task[0]), int(task[1])
        query_length = int(lengths[idx_i])
        target_length = int(lengths[idx_j])
        query_bucket = _length_bucket(query_length, length_bucket_quantum)
        target_bucket = _length_bucket(target_length, length_bucket_quantum)
        key = (query_bucket, target_bucket)
        entries = buckets.setdefault(key, [])
        candidate = entries + [
            (
                ordinal,
                task,
                query_length,
                target_length,
                query_bucket,
                target_bucket,
            )
        ]
        padded_cells = len(candidate) * query_bucket * target_bucket
        real_cells = sum(entry[2] * entry[3] for entry in candidate)
        padding_ok = padded_cells <= real_cells * (
            1.0 + PADDING_OVERHEAD_LIMIT
        )
        workspace_ok = _pair_packet_workspace_bytes(
            len(candidate),
            query_bucket,
            target_bucket,
            feature_dimension,
        ) <= workspace_budget
        if entries and (not padding_ok or not workspace_ok):
            yield _build_pair_work_packet(
                entries,
                batch_id=batch_id,
                feature_dimension=feature_dimension,
            )
            entries = []
            buckets[key] = entries
        entries.append(
            (
                ordinal,
                task,
                query_length,
                target_length,
                query_bucket,
                target_bucket,
            )
        )

    for entries in buckets.values():
        if entries:
            yield _build_pair_work_packet(
                entries,
                batch_id=batch_id,
                feature_dimension=feature_dimension,
            )


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


class TiledAcceleratorSession:
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
        tasks = list(tasks)
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
    ) as owned_session:
        return owned_session.run(
            tasks,
            progress=progress,
            result_callback=result_callback,
            result_chunk_size=result_chunk_size,
        )


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
