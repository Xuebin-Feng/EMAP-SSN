# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Consolidated PyTorch device discovery, hardware acceleration, and layout physics benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Optional

import numpy as np
import torch

# =====================================================================
# 1. Portable PyTorch Device Discovery & Device Benchmarks
# =====================================================================

AUTO_DEVICE = "auto"
BENCHMARK_TIE_FRACTION = 0.03
BACKEND_STATE_FILENAME = "ssn_backend.json"


@dataclass(frozen=True)
class DeviceCandidate:
    """One concrete CPU or accelerator backend available to PyTorch."""

    spec: str
    label: str
    device: Any
    backend: str
    index: Optional[int] = None
    supports_streams: bool = False

    @property
    def is_cpu(self) -> bool:
        return self.backend == "cpu"

    @property
    def display_name(self) -> str:
        return f"{self.label} [{self.spec}]"


@dataclass(frozen=True)
class BenchmarkResult:
    """Comparable timing result for one device/concurrency plan."""

    candidate: DeviceCandidate
    value: Optional[float]
    lanes: int = 1
    error: Optional[str] = None
    variant: str = "scalar"
    execution_plan: Any = None
    peak_memory_bytes: Optional[int] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.value is not None


def _safe_name(getter, fallback: str) -> str:
    try:
        value = getter()
    except Exception:
        return fallback
    return str(value) if value else fallback


def _validated_device_specs() -> Optional[set[str]]:
    """Return the installer-approved runtime devices, or ``None`` if unmanaged."""
    state_path = Path(sys.prefix) / BACKEND_STATE_FILENAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or int(state.get("schema", 0)) < 3:
        return None
    devices = state.get("validated_devices")
    if not isinstance(devices, list):
        return None
    return {
        str(device.get("spec"))
        for device in devices
        if isinstance(device, dict) and device.get("success") and device.get("spec")
    }


def get_available_devices(*, diagnostics=None) -> list[DeviceCandidate]:
    """Return CPU and every accelerator/backend visible in this process."""
    approved = _validated_device_specs()
    candidates = [
        DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
    ]

    try:
        if torch.cuda.is_available():
            backend_label = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            for index in range(torch.cuda.device_count()):
                spec = f"cuda:{index}"
                if approved is not None and spec not in approved:
                    continue
                name = _safe_name(
                    lambda i=index: torch.cuda.get_device_name(i),
                    f"{backend_label} device {index}",
                )
                candidates.append(
                    DeviceCandidate(
                        spec,
                        f"{name} ({backend_label})",
                        torch.device(f"cuda:{index}"),
                        "cuda",
                        index,
                        True,
                    )
                )
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({"field": "cuda", "reason": str(error)})

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for index in range(torch.xpu.device_count()):
                spec = f"xpu:{index}"
                if approved is not None and spec not in approved:
                    continue
                name = _safe_name(
                    lambda i=index: torch.xpu.get_device_name(i),
                    f"Intel XPU device {index}",
                )
                candidates.append(
                    DeviceCandidate(
                        spec,
                        f"{name} (XPU)",
                        torch.device(f"xpu:{index}"),
                        "xpu",
                        index,
                        True,
                    )
                )
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({"field": "xpu", "reason": str(error)})

    try:
        if (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and (approved is None or "mps" in approved)
        ):
            name = _safe_name(
                lambda: torch.backends.mps.get_name(), "Apple GPU"
            )
            candidates.append(
                DeviceCandidate("mps", f"{name} (MPS)", torch.device("mps"), "mps")
            )
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({"field": "mps", "reason": str(error)})

    return candidates


def device_selection_options() -> list[tuple[str, str]]:
    """Return ``(display label, persisted spec)`` pairs for GUI controls."""
    return [("Auto Benchmark", AUTO_DEVICE)] + [
        (candidate.display_name, candidate.spec)
        for candidate in get_available_devices()
    ]


def normalize_device_selection(selection: Any) -> str:
    """Normalize persisted specs and legacy/display labels."""
    if selection is None:
        return AUTO_DEVICE
    text = str(selection).strip()
    if not text or text.lower() in {"auto", "auto benchmark"}:
        return AUTO_DEVICE
    match = re.search(r"\[([^\[\]]+)\]\s*$", text)
    if match:
        text = match.group(1)
    normalized = text.strip().lower()
    if normalized == "directml" or re.fullmatch(r"directml:\d+", normalized):
        return AUTO_DEVICE
    if normalized == "cpu":
        return "cpu"
    if normalized in {"mps"}:
        return normalized
    if re.fullmatch(r"(?:cuda|xpu):\d+", normalized):
        return normalized
    if normalized in {"cuda", "xpu"}:
        return f"{normalized}:0"
    return normalized


def resolve_device_selection(
    selection: Any,
    candidates: Optional[Iterable[DeviceCandidate]] = None,
) -> Optional[DeviceCandidate]:
    """Resolve a manual selection; return ``None`` for automatic selection."""
    spec = normalize_device_selection(selection)
    if spec == AUTO_DEVICE:
        return None
    available = list(candidates) if candidates is not None else get_available_devices()
    for candidate in available:
        if candidate.spec == spec:
            return candidate
    labels = ", ".join(candidate.spec for candidate in available)
    raise ValueError(
        f"Selected device '{spec}' is not available. Available devices: {labels}."
    )


def synchronize_device(device_or_candidate: Any) -> None:
    """Wait for queued work on a supported accelerator backend."""
    candidate = (
        device_or_candidate
        if isinstance(device_or_candidate, DeviceCandidate)
        else None
    )
    backend = candidate.backend if candidate else getattr(
        device_or_candidate, "type", str(device_or_candidate).split(":", 1)[0]
    )
    device = candidate.device if candidate else device_or_candidate
    if backend == "cuda":
        torch.cuda.synchronize(device)
    elif backend == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize(device)
    elif backend == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def release_device_cache(device_or_candidate: Any) -> None:
    """Release unused allocations owned by the selected backend."""
    candidate = (
        device_or_candidate
        if isinstance(device_or_candidate, DeviceCandidate)
        else None
    )
    backend = candidate.backend if candidate else getattr(
        device_or_candidate, "type", str(device_or_candidate).split(":", 1)[0]
    )
    try:
        if backend == "cuda":
            torch.cuda.empty_cache()
        elif backend == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        elif backend == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    finally:
        gc.collect()


def rank_benchmark_results(
    results: Iterable[BenchmarkResult],
    *,
    higher_is_better: bool,
    tie_fraction: float = BENCHMARK_TIE_FRACTION,
) -> list[BenchmarkResult]:
    """Rank successful results, preferring CPU/fewer lanes inside a 3% tie."""
    remaining = [result for result in results if result.succeeded]
    ranked: list[BenchmarkResult] = []
    while remaining:
        values = [float(result.value) for result in remaining]
        best = max(values) if higher_is_better else min(values)
        if higher_is_better:
            threshold = best * (1.0 - tie_fraction)
            competitive = [r for r in remaining if float(r.value) >= threshold]
        else:
            threshold = best * (1.0 + tie_fraction)
            competitive = [r for r in remaining if float(r.value) <= threshold]
        selected = min(
            competitive,
            key=lambda r: (
                0 if r.candidate.is_cpu else 1,
                0 if r.variant == "scalar" else 1,
                (
                    int(r.peak_memory_bytes)
                    if r.peak_memory_bytes is not None
                    else 2 ** 63 - 1
                ),
                {
                    "eager": 0,
                    "compiled": 1,
                    "compiled_graph": 2,
                }.get(
                    getattr(
                        r.execution_plan, "scorer_variant", "eager"
                    ),
                    3,
                ),
                int(r.lanes),
                int(
                    getattr(
                        r.execution_plan,
                        "microbatch_workspace_bytes",
                        0,
                    )
                ),
                r.candidate.spec,
            ),
        )
        ranked.append(selected)
        remaining.remove(selected)
    return ranked


def get_optimal_device():
    """Return the first available accelerator, retaining the legacy API."""
    candidates = get_available_devices()
    for backend in ("cuda", "xpu", "mps"):
        for candidate in candidates:
            if candidate.backend == backend:
                return candidate.device
    return torch.device("cpu")


# =====================================================================
# 2. Physics Layout Workload Preparation & Accelerator Tuning
# =====================================================================

@dataclass
class PreparedLayoutBatch:
    global_nodes: list[int]
    edges: list[tuple[int, int]]
    scores: list[float]
    positions: np.ndarray
    component_labels: np.ndarray
    box_limits: np.ndarray
    node_count: int
    is_large_job: bool


def layout_size_class(node_count: int) -> str:
    if node_count < 500:
        return "small"
    if node_count <= 2000:
        return "medium"
    return "massive"


def benchmark_step_count(size_class: str) -> int:
    return {"small": 20, "medium": 5, "massive": 1}[size_class]


def prepare_layout_batch(
    batch_components: list[list[int]],
    node_to_component: dict[int, int],
    component_edges: dict[int, list[tuple[int, int]]],
    component_scores: dict[int, list[float]],
    params: dict[str, Any],
    *,
    add_noise: bool = True,
    verbose: bool = True,
    rng: np.random.Generator | None = None,
) -> PreparedLayoutBatch:
    """Build one vectorized physics batch without mutating source inputs."""
    node_count = sum(len(component) for component in batch_components)
    is_large_job = len(batch_components) == 1 and node_count >= 500
    global_nodes = [node for component in batch_components for node in component]
    global_to_batch = {
        global_id: local_id for local_id, global_id in enumerate(global_nodes)
    }

    batch_edges: list[tuple[int, int]] = []
    batch_scores: list[float] = []
    position_blocks = []
    label_blocks = []
    box_limit_blocks = []

    for batch_component_index, component in enumerate(batch_components):
        component_node_count = len(component)
        component_index = node_to_component[component[0]]
        edges = component_edges[component_index]
        scores = component_scores[component_index]
        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(component)
        }
        local_edges = [
            (global_to_local[source], global_to_local[target])
            for source, target in edges
        ]

        box_limit = (
            np.sqrt(component_node_count) * 2.5 + 5.0
        ) * params.get("BOX_SCALE", 1.0)
        local_positions = None
        spectral_success = False

        if component_node_count >= 4:
            if verbose and component_node_count >= 50:
                print(
                    "  > Calculating Spectral Layout for sub-component "
                    f"({component_node_count} nodes)..."
                )
            try:
                import scipy.sparse as sp
                from scipy.sparse.csgraph import laplacian
                from scipy.sparse.linalg import eigsh

                row = [edge[0] for edge in local_edges] + [
                    edge[1] for edge in local_edges
                ]
                col = [edge[1] for edge in local_edges] + [
                    edge[0] for edge in local_edges
                ]
                data = list(scores) + list(scores)
                adjacency = sp.coo_matrix(
                    (data, (row, col)),
                    shape=(component_node_count, component_node_count),
                )
                graph_laplacian = laplacian(adjacency, normed=True)
                eigsh_v0 = (
                    None
                    if rng is None
                    else rng.standard_normal(component_node_count)
                )
                _, vectors = eigsh(
                    graph_laplacian,
                    k=3,
                    which="SM",
                    tol=1e-3,
                    v0=eigsh_v0,
                )
                x_coordinates = vectors[:, 1]
                y_coordinates = vectors[:, 2]
                x_normalized = (x_coordinates - np.min(x_coordinates)) / (
                    np.ptp(x_coordinates) + 1e-9
                )
                y_normalized = (y_coordinates - np.min(y_coordinates)) / (
                    np.ptp(y_coordinates) + 1e-9
                )
                local_positions = np.column_stack(
                    (
                        (x_normalized - 0.5) * box_limit * 0.8,
                        (y_normalized - 0.5) * box_limit * 0.8,
                    )
                ).astype(np.float32)
                spectral_success = True
            except Exception as error:
                if verbose and component_node_count >= 50:
                    print(
                        f"  > Spectral solver failed: {error}. "
                        "Falling back to grid layout."
                    )

        if not spectral_success:
            side = int(np.ceil(np.sqrt(component_node_count)))
            axis = np.linspace(-box_limit * 0.5, box_limit * 0.5, side)
            x_grid, y_grid = np.meshgrid(axis, axis)
            local_positions = np.column_stack(
                (x_grid.flatten(), y_grid.flatten())
            )[:component_node_count].astype(np.float32)

        local_minimum = np.min(local_positions, axis=0)
        local_maximum = np.max(local_positions, axis=0)
        local_positions -= (local_minimum + local_maximum) / 2.0
        position_blocks.append(local_positions)
        label_blocks.append(
            np.full(component_node_count, batch_component_index, dtype=np.int32)
        )
        box_limit_blocks.append(
            np.full(component_node_count, box_limit, dtype=np.float32)
        )

        for (source, target), score in zip(edges, scores):
            batch_edges.append(
                (global_to_batch[source], global_to_batch[target])
            )
            batch_scores.append(float(score))

    positions = np.vstack(position_blocks).astype(np.float32)
    if add_noise:
        if rng is None:
            noise = np.random.normal(0, 0.1, positions.shape)
        else:
            noise = rng.normal(0, 0.1, positions.shape)
        positions += noise.astype(np.float32)

    return PreparedLayoutBatch(
        global_nodes=global_nodes,
        edges=batch_edges,
        scores=batch_scores,
        positions=positions,
        component_labels=np.concatenate(label_blocks),
        box_limits=np.concatenate(box_limit_blocks),
        node_count=node_count,
        is_large_job=is_large_job,
    )


def representative_job_indices(
    jobs: list[list[list[int]]],
    node_to_component: dict[int, int],
    component_edges: dict[int, list[tuple[int, int]]],
) -> dict[str, int]:
    """Choose the median estimated-cost job in each populated size class."""
    grouped: dict[str, list[tuple[float, int]]] = {}
    for index, job in enumerate(jobs):
        node_count = sum(len(component) for component in job)
        edge_count = sum(
            len(component_edges[node_to_component[component[0]]])
            for component in job
        )
        cost = node_count * node_count + edge_count
        grouped.setdefault(layout_size_class(node_count), []).append((cost, index))

    selected = {}
    for size_class, values in grouped.items():
        ordered = sorted(values, key=lambda value: (value[0], value[1]))
        selected[size_class] = ordered[len(ordered) // 2][1]
    return selected


def manual_layout_rankings(
    jobs: list[list[list[int]]], selection: Any
) -> dict[str, list[BenchmarkResult]] | None:
    """Return per-class manual plans, or ``None`` when Auto is selected."""
    if normalize_device_selection(selection) == AUTO_DEVICE:
        return None
    candidate = resolve_device_selection(selection)
    populated_classes = {
        layout_size_class(sum(len(component) for component in job))
        for job in jobs
    }
    print(
        f"Layout generation: manual device {candidate.display_name}; "
        "benchmark skipped."
    )
    result = BenchmarkResult(candidate=candidate, value=0.0, lanes=1)
    return {size_class: [result] for size_class in populated_classes}


def _active_mask(node_count: int, edges: np.ndarray) -> np.ndarray:
    mask = np.zeros(node_count, dtype=np.bool_)
    if len(edges):
        mask[np.unique(edges.reshape(-1))] = True
    return mask


def _snapshot_random_state() -> tuple[Any, Any, dict[str, Any]]:
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    accelerator_states = {}
    for backend_name in ("cuda", "xpu"):
        backend = getattr(torch, backend_name, None)
        try:
            if backend is not None and backend.is_available():
                accelerator_states[backend_name] = backend.get_rng_state_all()
        except Exception:
            pass
    try:
        if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
            accelerator_states["mps"] = torch.mps.get_rng_state()
    except Exception:
        pass
    return numpy_state, torch_state, accelerator_states


def _restore_random_state(state: tuple[Any, Any, dict[str, Any]]) -> None:
    numpy_state, torch_state, accelerator_states = state
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
    for backend_name in ("cuda", "xpu"):
        if backend_name not in accelerator_states:
            continue
        try:
            getattr(torch, backend_name).set_rng_state_all(
                accelerator_states[backend_name]
            )
        except Exception:
            pass
    if "mps" in accelerator_states:
        try:
            torch.mps.set_rng_state(accelerator_states["mps"])
        except Exception:
            pass


def _construct_simulation(
    candidate: DeviceCandidate,
    cpu_simulation_class: type,
    gpu_simulation_class: type | None,
    positions: np.ndarray,
    edges: np.ndarray,
    labels: np.ndarray,
    box_limits: np.ndarray,
    params: dict[str, Any],
    active_mask: np.ndarray,
):
    arguments = (
        positions.copy(),
        edges.copy(),
        labels.copy(),
        box_limits.copy(),
        params.copy(),
    )
    if candidate.is_cpu:
        return cpu_simulation_class(*arguments, active_mask=active_mask.copy())
    if gpu_simulation_class is None:
        raise RuntimeError("PyTorch accelerator simulation is unavailable")
    return gpu_simulation_class(
        *arguments,
        active_mask=active_mask.copy(),
        device=candidate.device,
    )


def benchmark_layout_devices(
    prepared: PreparedLayoutBatch,
    params: dict[str, Any],
    *,
    selection: Any,
    size_class: str,
    engine_label: str,
    cpu_simulation_class: type,
    gpu_simulation_class: type | None,
) -> list[BenchmarkResult]:
    """Benchmark final/full-connectivity work and return a ranked plan list."""
    available = get_available_devices()
    manual = resolve_device_selection(selection, available)
    if manual is not None:
        print(
            f"{engine_label} layout: manual device "
            f"{manual.display_name}; benchmark skipped."
        )
        return [
            BenchmarkResult(candidate=manual, value=0.0, lanes=1)
        ]
    candidates = available
    threshold = float(params.get("SIMILARITY_THRESHOLD", 0.0))
    final_edges = np.asarray(
        [
            edge
            for edge, score in zip(prepared.edges, prepared.scores)
            if score >= threshold
        ],
        dtype=np.int32,
    )
    if final_edges.size == 0:
        final_edges = np.zeros((0, 2), dtype=np.int32)
    active_mask = _active_mask(prepared.node_count, final_edges)
    steps = benchmark_step_count(size_class)
    results = []
    baseline_random_state = _snapshot_random_state()

    print(
        f"\n{engine_label} layout benchmark ({size_class}, "
        f"{prepared.node_count} nodes, {steps} steps)"
    )
    print("Device/backend                 Lanes   Time (s)   Status")
    try:
        for candidate in candidates:
            error = None
            elapsed = None
            try:
                _restore_random_state(baseline_random_state)
                warmup = _construct_simulation(
                    candidate,
                    cpu_simulation_class,
                    gpu_simulation_class,
                    prepared.positions,
                    final_edges,
                    prepared.component_labels,
                    prepared.box_limits,
                    params,
                    active_mask,
                )
                warmup.step(0)
                warmup.get_pos()
                synchronize_device(candidate)
                del warmup

                _restore_random_state(baseline_random_state)
                synchronize_device(candidate)
                started = time.perf_counter()
                simulation = _construct_simulation(
                    candidate,
                    cpu_simulation_class,
                    gpu_simulation_class,
                    prepared.positions,
                    final_edges,
                    prepared.component_labels,
                    prepared.box_limits,
                    params,
                    active_mask,
                )
                for step in range(steps):
                    simulation.step(step)
                simulation.get_pos()
                synchronize_device(candidate)
                elapsed = time.perf_counter() - started
                del simulation
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            finally:
                release_device_cache(candidate)

            result = BenchmarkResult(
                candidate=candidate,
                value=elapsed,
                lanes=1,
                error=error,
            )
            results.append(result)
            status = error or "ok"
            elapsed_text = f"{elapsed:.4f}" if elapsed is not None else "--"
            print(
                f"{candidate.display_name[:30]:30}  {1:>5}   "
                f"{elapsed_text:>8}   {status}"
            )
    finally:
        _restore_random_state(baseline_random_state)

    ranked = rank_benchmark_results(results, higher_is_better=False)
    if not ranked:
        failures = "; ".join(result.error or "unknown" for result in results)
        raise RuntimeError(f"Every layout benchmark candidate failed: {failures}")

    selected = ranked[0]
    fastest = min(
        (result for result in results if result.succeeded),
        key=lambda result: float(result.value),
    )
    tie_applied = selected.candidate.spec != fastest.candidate.spec
    print(
        f"Selected {selected.candidate.display_name}; "
        f"3% tie preference {'applied' if tie_applied else 'not applied'}."
    )
    return ranked


def prepare_representative_batches(
    jobs: list[list[list[int]]],
    representative_indices: dict[str, int],
    node_to_component: dict[int, int],
    component_edges: dict[int, list[tuple[int, int]]],
    component_scores: dict[int, list[float]],
    params: dict[str, Any],
) -> dict[str, PreparedLayoutBatch]:
    """Prepare benchmark copies while leaving NumPy/Torch random state intact."""
    random_state = _snapshot_random_state()
    try:
        return {
            size_class: prepare_layout_batch(
                jobs[index],
                node_to_component,
                component_edges,
                component_scores,
                params,
                add_noise=True,
                verbose=False,
            )
            for size_class, index in representative_indices.items()
        }
    finally:
        _restore_random_state(random_state)


__all__ = [
    "AUTO_DEVICE",
    "BENCHMARK_TIE_FRACTION",
    "BACKEND_STATE_FILENAME",
    "DeviceCandidate",
    "BenchmarkResult",
    "get_available_devices",
    "device_selection_options",
    "normalize_device_selection",
    "resolve_device_selection",
    "synchronize_device",
    "release_device_cache",
    "rank_benchmark_results",
    "get_optimal_device",
    "PreparedLayoutBatch",
    "layout_size_class",
    "benchmark_step_count",
    "prepare_layout_batch",
    "representative_job_indices",
    "manual_layout_rankings",
    "benchmark_layout_devices",
    "prepare_representative_batches",
]
