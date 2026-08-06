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
from typing import Any, Iterable, Optional
import gc
import re

import torch


AUTO_DEVICE = "auto"
BENCHMARK_TIE_FRACTION = 0.03


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

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.value is not None


def _safe_name(getter, fallback: str) -> str:
    try:
        value = getter()
    except Exception:
        return fallback
    return str(value) if value else fallback


def get_available_devices() -> list[DeviceCandidate]:
    """Return CPU and every accelerator/backend visible in this process."""
    candidates = [
        DeviceCandidate("cpu", "CPU", torch.device("cpu"), "cpu")
    ]

    try:
        if torch.cuda.is_available():
            backend_label = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            for index in range(torch.cuda.device_count()):
                name = _safe_name(
                    lambda i=index: torch.cuda.get_device_name(i),
                    f"{backend_label} device {index}",
                )
                candidates.append(
                    DeviceCandidate(
                        f"cuda:{index}",
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
                name = _safe_name(
                    lambda i=index: torch.xpu.get_device_name(i),
                    f"Intel XPU device {index}",
                )
                candidates.append(
                    DeviceCandidate(
                        f"xpu:{index}",
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
        ):
            name = _safe_name(
                lambda: torch.backends.mps.get_name(), "Apple GPU"
            )
            candidates.append(
                DeviceCandidate("mps", f"{name} (MPS)", torch.device("mps"), "mps")
            )
    except Exception:
        pass

    try:
        import torch_directml

        available_getter = getattr(torch_directml, "is_available", None)
        directml_available = (
            bool(available_getter()) if callable(available_getter) else True
        )
        if directml_available:
            count_getter = getattr(torch_directml, "device_count", None)
            count = int(count_getter()) if callable(count_getter) else 1
            name_getter = getattr(torch_directml, "device_name", None)
            for index in range(max(1, count)):
                name = _safe_name(
                    (lambda i=index: name_getter(i))
                    if callable(name_getter)
                    else (lambda i=index: f"DirectML device {i}"),
                    f"DirectML device {index}",
                )
                candidates.append(
                    DeviceCandidate(
                        f"directml:{index}",
                        f"{name} (DirectML)",
                        torch_directml.device(index),
                        "directml",
                        index,
                        False,
                    )
                )
    except (ImportError, RuntimeError, TypeError, AttributeError):
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
    if normalized == "cpu":
        return "cpu"
    if normalized in {"mps"}:
        return normalized
    if re.fullmatch(r"(?:cuda|xpu|directml):\d+", normalized):
        return normalized
    if normalized in {"cuda", "xpu", "directml"}:
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
                int(r.lanes),
                r.candidate.spec,
            ),
        )
        ranked.append(selected)
        remaining.remove(selected)
    return ranked


def get_optimal_device():
    """Return the first available accelerator, retaining the legacy API."""
    candidates = get_available_devices()
    for backend in ("cuda", "xpu", "mps", "directml"):
        for candidate in candidates:
            if candidate.backend == backend:
                return candidate.device
    return torch.device("cpu")
