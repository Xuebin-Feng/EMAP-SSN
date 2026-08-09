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

"""Inventory graphics devices and build an ordered PyTorch backend ladder."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable


COMPATIBILITY_REVISION = 3
CUDA_13_MIN_DRIVER = (580, 0)
WINDOWS_11_MIN_BUILD = 22000
WINDOWS_11_25H2_MIN_BUILD = 26200
WINDOWS_ROCM_721_MIN_AMD_SOFTWARE = (26, 2, 2)

# This is a deliberately pinned local snapshot. A newly released GPU is not
# guessed to be compatible until its model/GFX target is added here.
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
ROCM_714_TARGETS = frozenset(target for target, _patterns in AMD_GFX_PATTERNS)
ROCM_721_TARGETS = frozenset(
    {"gfx1201", "gfx1200", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152"}
)
INTEGRATED_AMD_TARGETS = frozenset({"gfx1103", "gfx1150", "gfx1151", "gfx1152"})

INTEL_XPU_PATTERNS = (
    r"intel.*arc(?:\(tm\))?\s+[ab]-?series",
    r"intel.*arc(?:\(tm\))?\s+[ab]\d{3}",
    r"intel.*arc(?:\(tm\))?\s+graphics",
    r"intel.*data\s+center\s+gpu\s+max",
)

DEVICE_CLASS_ORDER = {"discrete": 0, "unknown": 1, "integrated": 2}
VENDOR_ORDER = {"NVIDIA": 0, "AMD": 1, "INTEL": 2, "APPLE": 3, "CPU": 4}


def _run(command: list[str], timeout: int = 5) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", value))


def _windows_build(version: str) -> int | None:
    parts = _version_tuple(version)
    return parts[2] if len(parts) >= 3 else None


def _lines(result: subprocess.CompletedProcess[str] | None) -> list[str]:
    if result is None or result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _json_result(result: subprocess.CompletedProcess[str] | None) -> list[dict[str, Any]]:
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        value = json.loads(result.stdout.lstrip("\ufeff"))
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        return [value]
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _windows_cim(class_name: str, properties: Iterable[str]) -> list[dict[str, Any]]:
    selected = ",".join(properties)
    script = (
        f"$items = @(Get-CimInstance {class_name} | Select-Object {selected}); "
        "ConvertTo-Json -InputObject $items -Compress"
    )
    return _json_result(_run(["powershell", "-NoProfile", "-Command", script], timeout=8))


def _windows_os() -> dict[str, Any]:
    values = _windows_cim("Win32_OperatingSystem", ("Caption", "Version", "BuildNumber"))
    return values[0] if values else {}


def _windows_amd_software_version() -> str | None:
    script = (
        "$keys=@('HKLM:\\SOFTWARE\\AMD\\CN','HKLM:\\SOFTWARE\\WOW6432Node\\AMD\\CN');"
        "$names=@('ProductVersion','ReleaseVersion','DriverVersion');"
        "foreach($key in $keys){if(Test-Path $key){$item=Get-ItemProperty $key;"
        "foreach($name in $names){$value=$item.$name;if($value){Write-Output $value;exit}}}}"
    )
    values = _lines(_run(["powershell", "-NoProfile", "-Command", script]))
    return values[0] if values else None


def _windows_names(class_name: str) -> list[str]:
    """Compatibility wrapper returning CIM object names."""
    return [str(row.get("Name")) for row in _windows_cim(class_name, ("Name",)) if row.get("Name")]


def _controller_names(system: str) -> list[str]:
    """Compatibility wrapper returning display-controller descriptions."""
    if system == "windows":
        return _windows_names("Win32_VideoController")
    if system == "linux":
        return [
            line
            for line in _lines(_run(["lspci"]))
            if re.search(r"VGA|3D controller|Display controller", line, re.I)
        ]
    if system == "darwin":
        return _lines(_run(["system_profiler", "SPDisplaysDataType"], timeout=8))
    return []


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.lower()] = value.strip().strip('"')
    return values


def _pci_id(value: str) -> str | None:
    match = re.search(r"(?:VEN_|\[)([0-9a-f]{4})(?:&DEV_|:)([0-9a-f]{4})", value, re.I)
    return f"{match.group(1).lower()}:{match.group(2).lower()}" if match else None


def _vendor(name: str, pci_id: str | None = None) -> str:
    text = name.lower()
    vendor_id = (pci_id or "").split(":", 1)[0]
    if "nvidia" in text or vendor_id == "10de":
        return "NVIDIA"
    if any(token in text for token in ("amd", "radeon", "advanced micro devices")) or vendor_id == "1002":
        return "AMD"
    if "intel" in text or vendor_id == "8086":
        return "INTEL"
    return "UNKNOWN"


def amd_gfx_target(names: Iterable[str]) -> str | None:
    text = "\n".join(names).lower()
    for target, patterns in AMD_GFX_PATTERNS:
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            return target
    return None


def _device_kind(vendor: str, name: str, gfx_target: str | None, processors: list[str]) -> str:
    text = name.lower()
    if vendor == "NVIDIA":
        return "discrete"
    if vendor == "AMD":
        if gfx_target in INTEGRATED_AMD_TARGETS or re.search(r"radeon\s+\d{3,4}m", text):
            return "integrated"
        if re.search(r"\b(?:rx|pro\s+w|v)\s*\d", text) or "radeon ai pro" in text:
            return "discrete"
    if vendor == "INTEL":
        if "data center gpu max" in text or re.search(r"arc.*\b[ab]\d{3}", text):
            return "discrete"
        if "arc" in text and any("core" in value.lower() and "ultra" in value.lower() for value in processors):
            return "integrated"
    return "unknown"


def _windows_inventory(processors: list[str]) -> list[dict[str, Any]]:
    rows = _windows_cim(
        "Win32_VideoController",
        ("Name", "PNPDeviceID", "DriverVersion", "AdapterCompatibility", "VideoProcessor", "AdapterRAM"),
    )
    devices: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        name = str(row.get("Name") or row.get("VideoProcessor") or "Unknown display controller")
        pnp_id = str(row.get("PNPDeviceID") or "")
        pci_id = _pci_id(pnp_id)
        vendor = _vendor(name, pci_id)
        gfx = amd_gfx_target([name, *processors]) if vendor == "AMD" else None
        devices.append(
            {
                "id": pnp_id or pci_id or f"windows:{index}",
                "name": name,
                "vendor": vendor,
                "pci_id": pci_id,
                "driver_version": str(row.get("DriverVersion") or "") or None,
                "kind": _device_kind(vendor, name, gfx, processors),
                "architecture": gfx,
                "source": "Win32_VideoController",
            }
        )
    return devices


def _inventory_from_names(names: Iterable[str], processors: list[str], source: str) -> list[dict[str, Any]]:
    """Create conservative records when only legacy name discovery is available."""
    devices: list[dict[str, Any]] = []
    for index, name_value in enumerate(names):
        name = str(name_value)
        pci_id = _pci_id(name)
        vendor = _vendor(name, pci_id)
        gfx = amd_gfx_target([name, *processors]) if vendor == "AMD" else None
        devices.append({
            "id": f"{source}:{index}", "name": name, "vendor": vendor,
            "pci_id": pci_id, "driver_version": None, "driver": None,
            "kind": _device_kind(vendor, name, gfx, processors),
            "architecture": gfx, "source": source,
        })
    return devices


def _linux_inventory(processors: list[str]) -> list[dict[str, Any]]:
    lines = _lines(_run(["lspci", "-Dnnk"], timeout=8))
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        if re.search(r"VGA compatible controller|3D controller|Display controller", line, re.I):
            match = re.match(r"(\S+)\s+[^:]+:\s*(.+)", line)
            if not match:
                continue
            address, name = match.groups()
            pci_id = _pci_id(name)
            vendor = _vendor(name, pci_id)
            gfx = amd_gfx_target([name, *processors]) if vendor == "AMD" else None
            current = {
                "id": address,
                "name": re.sub(r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]\s*$", "", name, flags=re.I),
                "vendor": vendor,
                "pci_id": pci_id,
                "driver_version": None,
                "driver": None,
                "kind": _device_kind(vendor, name, gfx, processors),
                "architecture": gfx,
                "source": "lspci",
            }
            devices.append(current)
        elif current is not None:
            driver = re.search(r"Kernel driver in use:\s*(.+)", line, re.I)
            if driver:
                current["driver"] = driver.group(1).strip()
    return devices


def _nvidia_devices() -> list[dict[str, Any]]:
    fields = "pci.bus_id,name,compute_cap,driver_version"
    result = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    devices: list[dict[str, Any]] = []
    if result is not None and result.returncode == 0:
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) < 4:
                continue
            devices.append(
                {
                    "bus_id": row[0].strip(),
                    "name": row[1].strip(),
                    "compute_capability": row[2].strip() or None,
                    "driver_version": row[3].strip() or None,
                }
            )
        return devices
    result = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"])
    if result is not None and result.returncode == 0:
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) >= 2:
                devices.append({"name": row[0].strip(), "driver_version": row[1].strip(), "compute_capability": None})
    return devices


def _merge_nvidia_inventory(devices: list[dict[str, Any]], nvidia: list[dict[str, Any]]) -> None:
    unused = list(nvidia)
    for device in devices:
        if device["vendor"] != "NVIDIA":
            continue
        match = next((item for item in unused if item["name"].lower() in device["name"].lower()), None)
        if match:
            device.update({key: value for key, value in match.items() if value is not None})
            unused.remove(match)
    for index, item in enumerate(unused):
        devices.append(
            {
                "id": item.get("bus_id") or f"nvidia:{index}",
                "name": item["name"], "vendor": "NVIDIA", "pci_id": None,
                "driver_version": item.get("driver_version"), "kind": "discrete",
                "architecture": None, "source": "nvidia-smi",
                "compute_capability": item.get("compute_capability"),
            }
        )


def _rocm_targets() -> set[str]:
    targets: set[str] = set()
    for command in (["rocm_agent_enumerator", "-name"], ["rocminfo"]):
        result = _run(command, timeout=8)
        if result is not None and result.returncode == 0:
            targets.update(re.findall(r"\bgfx\d+[a-z0-9]*\b", result.stdout, re.I))
    return {target.lower() for target in targets}


def _intel_os_supported(system: str, os_info: dict[str, Any], device_name: str) -> bool:
    if system == "windows":
        return int(os_info.get("windows_build") or 0) >= WINDOWS_11_MIN_BUILD
    if system != "linux":
        return False
    distro = str(os_info.get("id") or "").lower()
    version = str(os_info.get("version_id") or "")
    if "data center gpu max" in device_name.lower():
        return (distro == "ubuntu" and version.startswith("22.04")) or (
            distro in {"rhel", "redhat"} and version.startswith("9.2")
        ) or (distro in {"sles", "suse"} and version.startswith("15"))
    supported_ubuntu = ("24.04", "25.10", "26.04")
    if "series 3" in device_name.lower() or "panther" in device_name.lower():
        supported_ubuntu = ("25.10", "26.04")
    return distro == "ubuntu" and version.startswith(supported_ubuntu)


def _nvidia_backend(devices: list[dict[str, Any]]) -> tuple[str, str]:
    capabilities = [_version_tuple(str(device.get("compute_capability") or "")) for device in devices]
    drivers = [_version_tuple(str(device.get("driver_version") or "")) for device in devices]
    modern = bool(capabilities) and all(value >= (7, 5) for value in capabilities if value) and all(capabilities)
    driver_ready = bool(drivers) and all(value >= CUDA_13_MIN_DRIVER for value in drivers if value) and all(drivers)
    if modern and driver_ready:
        return "cuda132", "All selected NVIDIA GPUs and the installed driver support CUDA 13.2."
    return "cuda126", "CUDA 12.6 provides the common compatible NVIDIA profile."


def _rank(device: dict[str, Any]) -> tuple[int, int, str]:
    return (
        DEVICE_CLASS_ORDER.get(str(device.get("kind")), 1),
        VENDOR_ORDER.get(str(device.get("vendor")), 9),
        str(device.get("id")),
    )


def _evaluate_devices(
    devices: list[dict[str, Any]], system: str, os_info: dict[str, Any], rocm_targets: set[str]
) -> None:
    windows_build = os_info.get("windows_build")
    linux_kfd = Path("/dev/kfd")
    for device in devices:
        vendor = device["vendor"]
        profiles: list[str] = []
        reasons: list[str] = []
        status = "ineligible"
        if vendor == "NVIDIA":
            profiles = ["cuda"]
            status = "eligible" if device.get("driver_version") else "provisional"
            reasons.append("NVIDIA display device detected; runtime validation is required.")
        elif vendor == "AMD":
            target = device.get("architecture")
            if not target:
                reasons.append("AMD model is not in the pinned GFX compatibility table.")
            elif system == "windows":
                if windows_build is None or int(windows_build) < WINDOWS_11_MIN_BUILD:
                    reasons.append("Native Windows ROCm requires Windows 11.")
                else:
                    if int(windows_build) >= WINDOWS_11_25H2_MIN_BUILD and target in ROCM_714_TARGETS:
                        profiles.append("rocm714")
                        device.setdefault("profile_eligibility", {})["rocm714"] = (
                            "eligible" if device.get("driver_version") else "provisional"
                        )
                    if target in ROCM_721_TARGETS:
                        amd_release = str(os_info.get("amd_software_version") or "")
                        if amd_release and _version_tuple(amd_release) < WINDOWS_ROCM_721_MIN_AMD_SOFTWARE:
                            reasons.append(
                                f"ROCm 7.2.1 requires AMD Software 26.2.2 or newer; detected {amd_release}."
                            )
                        else:
                            profiles.append("rocm721")
                            device.setdefault("profile_eligibility", {})["rocm721"] = (
                                "eligible" if amd_release else "provisional"
                            )
                    if profiles:
                        profile_states = device.get("profile_eligibility", {}).values()
                        status = "eligible" if "eligible" in profile_states else "provisional"
                        reasons.append(f"{target} is listed for: {', '.join(profiles)}.")
                    else:
                        reasons.append(f"{target} is not supported by a Windows ROCm profile for this OS build.")
            elif system == "linux":
                distro = str(os_info.get("id") or "").lower()
                version = str(os_info.get("version_id") or "")
                os_supported = distro == "ubuntu" and version.startswith(("22.04", "24.04", "25.10", "26.04"))
                driver_ready = linux_kfd.exists() and os.access(linux_kfd, os.R_OK | os.W_OK)
                target_seen = not rocm_targets or target in rocm_targets
                if not os_supported:
                    reasons.append("Linux ROCm profile is limited to the pinned supported Ubuntu releases.")
                elif not driver_ready:
                    reasons.append("/dev/kfd is missing or inaccessible; the installer does not install system GPU drivers.")
                elif not target_seen:
                    reasons.append(f"ROCm agents do not report the mapped target {target}.")
                elif target in ROCM_714_TARGETS:
                    profiles = ["rocm72"]
                    status = "eligible" if rocm_targets else "provisional"
                    reasons.append(f"Linux ROCm preflight accepted {target}.")
            else:
                reasons.append("ROCm is not configured for this operating system.")
        elif vendor == "INTEL":
            recognized = any(re.search(pattern, device["name"], re.I) for pattern in INTEL_XPU_PATTERNS)
            if not recognized:
                reasons.append("Intel adapter is not an Arc/Core Ultra Arc/Data Center GPU Max device.")
            elif not _intel_os_supported(system, os_info, device["name"]):
                reasons.append("Intel XPU does not list this OS/device combination.")
            else:
                profiles = ["xpu"]
                status = "eligible" if device.get("driver_version") or device.get("driver") else "provisional"
                reasons.append("Intel device family and OS are in the pinned XPU compatibility table.")
        else:
            reasons.append("Display-controller vendor is not supported for compute acceleration.")
        device["eligibility"] = status
        device["eligible_profiles"] = profiles
        device.setdefault("profile_eligibility", {profile: status for profile in profiles})
        device["reasons"] = reasons


def _candidate_ladder(devices: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    ignored: list[dict[str, Any]] = []

    nvidia = sorted([d for d in devices if "cuda" in d["eligible_profiles"]], key=_rank)
    if nvidia:
        backend, reason = _nvidia_backend(nvidia)
        candidates.append(
            ((_rank(nvidia[0])[0], VENDOR_ORDER["NVIDIA"], 0), {
                "backend": backend, "profile": backend, "vendor": "NVIDIA",
                "device_ids": [d["id"] for d in nvidia], "gfx_target": None,
                "eligibility": "eligible" if any(d["eligibility"] == "eligible" for d in nvidia) else "provisional",
                "reason": reason,
            })
        )

    amd = sorted([d for d in devices if d["vendor"] == "AMD" and d["eligible_profiles"]], key=_rank)
    if amd:
        discrete = [d for d in amd if d["kind"] == "discrete"]
        primary = (discrete or amd)[0]
        selected = [d for d in amd if d["architecture"] == primary["architecture"] and d["kind"] == primary["kind"]]
        for device in amd:
            if device not in selected:
                ignored.append({
                    "id": device["id"], "name": device["name"],
                    "reason": f"Only the preferred AMD {primary['kind']} target {primary['architecture']} is installed.",
                })
        for profile_index, profile in enumerate(primary["eligible_profiles"]):
            candidates.append(
                ((_rank(primary)[0], VENDOR_ORDER["AMD"], profile_index), {
                    "backend": profile, "profile": profile, "vendor": "AMD",
                    "device_ids": [d["id"] for d in selected],
                    "gfx_target": primary["architecture"],
                    "eligibility": primary.get("profile_eligibility", {}).get(profile, primary["eligibility"]),
                    "reason": f"Preferred AMD {primary['kind']} target {primary['architecture']} via {profile}.",
                })
            )

    intel = sorted([d for d in devices if "xpu" in d["eligible_profiles"]], key=_rank)
    if intel:
        candidates.append(
            ((_rank(intel[0])[0], VENDOR_ORDER["INTEL"], 0), {
                "backend": "xpu", "profile": "xpu", "vendor": "INTEL",
                "device_ids": [d["id"] for d in intel], "gfx_target": None,
                "eligibility": "eligible" if any(d["eligibility"] == "eligible" for d in intel) else "provisional",
                "reason": "Compatible Intel XPU devices detected.",
            })
        )

    apple = [d for d in devices if "mps" in d["eligible_profiles"]]
    if apple:
        candidates.append(
            ((DEVICE_CLASS_ORDER["integrated"], VENDOR_ORDER["APPLE"], 0), {
                "backend": "mps", "profile": "mps", "vendor": "APPLE",
                "device_ids": [apple[0]["id"]], "gfx_target": None,
                "eligibility": "eligible", "reason": "Apple Silicon MPS is available.",
            })
        )

    ordered = [value for _key, value in sorted(candidates, key=lambda item: item[0])]
    ordered.append({
        "backend": "cpu", "profile": "cpu", "vendor": "CPU", "device_ids": ["cpu"],
        "gfx_target": None, "eligibility": "eligible", "reason": "Portable CPU fallback.",
    })
    return ordered, ignored


def detect_hardware() -> dict[str, Any]:
    system = platform.system().lower()
    version = platform.version()
    processors: list[str] = []
    os_info: dict[str, Any]
    controller_names = _controller_names(system)
    if system == "windows":
        processors = _windows_names("Win32_Processor")
        os_row = _windows_os()
        build = _windows_build(version)
        if build is None and str(os_row.get("BuildNumber") or "").isdigit():
            build = int(os_row["BuildNumber"])
        os_info = {
            "caption": os_row.get("Caption"), "version": os_row.get("Version") or version,
            "windows_build": build,
            "amd_software_version": _windows_amd_software_version(),
        }
        devices = _windows_inventory(processors)
    elif system == "linux":
        os_info = _read_os_release()
        os_info["kernel"] = platform.release()
        devices = _linux_inventory(processors)
    else:
        os_info = {"version": version}
        devices = []

    discovered_names = {str(device.get("name", "")).lower() for device in devices}
    requested_names = {name.lower() for name in controller_names}
    if system == "windows" and not controller_names:
        devices = []
    elif requested_names and (not devices or (system == "windows" and requested_names != discovered_names)):
        devices = _inventory_from_names(controller_names, processors, f"{system}-names")

    nvidia = _nvidia_devices()
    _merge_nvidia_inventory(devices, nvidia)
    if system == "darwin" and platform.machine().lower() == "arm64":
        devices.append({
            "id": "apple:mps", "name": "Apple Silicon GPU", "vendor": "APPLE",
            "pci_id": None, "driver_version": version, "kind": "integrated",
            "architecture": platform.machine().lower(), "source": "platform",
            "eligibility": "eligible", "eligible_profiles": ["mps"],
            "reasons": ["Apple Silicon detected."],
        })

    rocm_targets = _rocm_targets() if system == "linux" and any(d["vendor"] == "AMD" for d in devices) else set()
    non_apple = [device for device in devices if device["vendor"] != "APPLE"]
    _evaluate_devices(non_apple, system, os_info, rocm_targets)
    candidates, ignored = _candidate_ladder(devices)
    primary = candidates[0]
    primary_reason = primary["reason"]
    if primary["backend"] == "cpu":
        rejected = [reason for device in devices for reason in device.get("reasons", [])]
        if rejected:
            primary_reason = rejected[0]
    controllers = [device["name"] for device in devices]
    return {
        "schema": 2,
        "compatibility_revision": COMPATIBILITY_REVISION,
        "platform": system,
        "os_version": str(os_info.get("version") or version),
        "os": os_info,
        "windows_build": os_info.get("windows_build"),
        "controllers": controllers,
        "processors": processors,
        "nvidia_devices": nvidia,
        "rocm_agents": sorted(rocm_targets),
        "devices": devices,
        "backend_candidates": candidates,
        "ignored_devices": ignored,
        "vendor": primary["vendor"],
        "backend": primary["backend"],
        "gfx_target": primary.get("gfx_target"),
        "reason": primary_reason,
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
