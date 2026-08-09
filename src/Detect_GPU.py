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

"""Detect installed graphics controllers and select a PyTorch backend."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import re
import subprocess
from typing import Any, Iterable


CUDA_13_MIN_DRIVER = (580, 0)
WINDOWS_11_MIN_BUILD = 22000

# Conservative model-name mapping from AMD's ROCm 7.14 hardware table. Generic
# names such as "AMD Radeon Graphics" are intentionally not guessed.
AMD_GFX_PATTERNS = (
    ("gfx1201", (r"rx\s*9070(?:\s*(?:xt|gre))?", r"r9700s?", r"r9600d")),
    ("gfx1200", (r"rx\s*9060(?:\s*xt(?:\s*lp)?)?",)),
    ("gfx1100", (r"rx\s*7900\s*(?:xtx|xt|gre)", r"w7900", r"w7800")),
    ("gfx1101", (r"rx\s*7800\s*xt", r"rx\s*7700(?:\s*xt)?", r"w7700", r"v710")),
    ("gfx1102", (r"rx\s*7600\b(?!\s*xt)",)),
    ("gfx1030", (r"w6800", r"v620")),
    ("gfx1103", (r"radeon\s*780m",)),
    ("gfx1150", (r"radeon\s*(?:890m|880m)", r"ryzen\s*ai\s*9\s*hx\s*(?:375|370)")),
    ("gfx1151", (r"radeon\s*(?:8060s|8050s)", r"ryzen\s*ai\s*max")),
    ("gfx1152", (r"radeon\s*(?:860m|840m)", r"ryzen\s*ai\s*[57]\s*3[45]0")),
)


def _run(command: list[str], timeout: int = 4) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _windows_build(version: str) -> int | None:
    parts = _version_tuple(version)
    if len(parts) >= 3:
        return parts[2]
    return None


def _lines(result: subprocess.CompletedProcess[str] | None) -> list[str]:
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _windows_names(class_name: str) -> list[str]:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        f"Get-CimInstance {class_name} | Select-Object -ExpandProperty Name",
    ]
    return _lines(_run(command))


def _controller_names(system: str) -> list[str]:
    if system == "windows":
        return _windows_names("Win32_VideoController")
    if system == "linux":
        result = _run(["lspci"])
        return [
            line
            for line in _lines(result)
            if re.search(r"vga|3d controller|display controller", line, re.I)
        ]
    if system == "darwin":
        return _lines(_run(["system_profiler", "SPDisplaysDataType"], timeout=8))
    return []


def _nvidia_devices() -> list[dict[str, Any]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, Any]] = []
    if result is not None and result.returncode == 0:
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) < 3:
                continue
            capability = _version_tuple(row[1].strip())
            devices.append(
                {
                    "name": row[0].strip(),
                    "compute_capability": ".".join(map(str, capability[:2])) if capability else None,
                    "driver_version": row[2].strip(),
                }
            )
        return devices

    # Older nvidia-smi versions do not expose compute_cap. Preserve discovery
    # and use the broadly compatible CUDA 12.6 build.
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is not None and result.returncode == 0:
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) >= 2:
                devices.append(
                    {
                        "name": row[0].strip(),
                        "compute_capability": None,
                        "driver_version": row[1].strip(),
                    }
                )
    return devices


def amd_gfx_target(names: Iterable[str]) -> str | None:
    text = "\n".join(names).lower()
    for target, patterns in AMD_GFX_PATTERNS:
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            return target
    return None


def _has_vendor(names: Iterable[str], *needles: str) -> bool:
    text = "\n".join(names).lower()
    return any(needle in text for needle in needles)


def _nvidia_backend(devices: list[dict[str, Any]]) -> tuple[str, str]:
    capabilities = [
        _version_tuple(str(device.get("compute_capability") or ""))
        for device in devices
    ]
    drivers = [
        _version_tuple(str(device.get("driver_version") or ""))
        for device in devices
    ]
    all_turing_or_newer = bool(capabilities) and all(
        capability >= (7, 5) for capability in capabilities if capability
    ) and all(capabilities)
    drivers_support_cuda_13 = bool(drivers) and all(
        driver >= CUDA_13_MIN_DRIVER for driver in drivers if driver
    ) and all(drivers)
    if all_turing_or_newer and drivers_support_cuda_13:
        return "cuda132", "All NVIDIA GPUs are Turing-or-newer and the driver supports CUDA 13.x."
    return "cuda126", "CUDA 12.6 was selected for older/unknown NVIDIA hardware or driver compatibility."


def detect_hardware() -> dict[str, Any]:
    system = platform.system().lower()
    version = platform.version()
    controllers = _controller_names(system)
    processors = _windows_names("Win32_Processor") if system == "windows" else []
    nvidia = _nvidia_devices()
    combined_names = controllers + processors + [device["name"] for device in nvidia]
    gfx_target = amd_gfx_target(combined_names)
    windows_build = _windows_build(version) if system == "windows" else None

    if nvidia or _has_vendor(controllers, "nvidia"):
        backend, reason = _nvidia_backend(nvidia)
        vendor = "NVIDIA"
    elif system == "windows" and _has_vendor(combined_names, "amd", "radeon"):
        if windows_build is None or windows_build < WINDOWS_11_MIN_BUILD:
            backend, vendor = "cpu", "AMD"
            reason = "Windows ROCm requires Windows 11; using CPU fallback."
        elif gfx_target:
            backend, vendor = "rocm714", "AMD"
            reason = f"Supported Windows ROCm 7.14 target detected: {gfx_target}."
        elif _has_vendor(controllers, "intel"):
            backend, vendor = "xpu", "INTEL"
            reason = "The AMD controller is not mapped to ROCm; a supported Intel GPU is available."
        else:
            backend, vendor = "cpu", "AMD"
            reason = "The AMD model is not in the maintained ROCm 7.14 allowlist; using CPU fallback."
    elif system == "linux" and _has_vendor(controllers, "amd", "radeon", "advanced micro devices"):
        backend, vendor = "rocm72", "AMD"
        reason = "AMD display controller detected on Linux; runtime validation will confirm ROCm support."
    elif _has_vendor(controllers, "intel"):
        backend, vendor = "xpu", "INTEL"
        reason = "Intel display controller detected."
    elif system == "darwin" and platform.machine().lower() == "arm64":
        backend, vendor = "mps", "APPLE"
        reason = "Apple Silicon detected."
    else:
        backend, vendor = "cpu", "CPU"
        reason = "No supported accelerator was detected."

    return {
        "platform": system,
        "os_version": version,
        "windows_build": windows_build,
        "controllers": controllers,
        "processors": processors,
        "nvidia_devices": nvidia,
        "vendor": vendor,
        "backend": backend,
        "gfx_target": gfx_target if backend == "rocm714" else None,
        "reason": reason,
    }


def detect_gpu() -> str:
    """Return the legacy vendor label used by older callers."""
    return str(detect_hardware()["vendor"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete detection report")
    args = parser.parse_args(argv)
    report = detect_hardware()
    print(json.dumps(report, sort_keys=True) if args.json else report["vendor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
