# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable PyTorch device discovery and workload benchmark helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Optional
import gc
import re

import torch


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


def get_available_devices() -> list[DeviceCandidate]:
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
    except Exception:
        pass

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
    except Exception:
        pass

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
    except Exception:
        pass

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
    # DirectML was used by older releases. It is no longer installed; migrate
    # persisted selections back to automatic backend discovery.
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
