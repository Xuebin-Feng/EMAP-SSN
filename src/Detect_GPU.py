# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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
from typing import Any, Iterable, NamedTuple


COMPATIBILITY_REVISION = 7
CUDA_13_MIN_DRIVER = (580, 0)
# PyTorch's CUDA 12.6 build carries kernels up to Hopper (compute capability
# 9.0). A newer GPU (Blackwell is 10.x and 12.x) runs only on the CUDA 13.2
# build, so with a driver older than 580 no build can use it.
CUDA_12_6_MAX_CAPABILITY = (9, 0)
# Its oldest kernels are for Maxwell (compute capability 5.0), and CUDA 12 runs
# on these drivers or newer (NVIDIA's minor version compatibility). Below
# either, the CUDA build would install and then fail validation.
CUDA_12_6_MIN_CAPABILITY = (5, 0)
CUDA_12_MIN_DRIVER = {"linux": (525, 60, 13), "windows": (528, 33)}
WINDOWS_11_MIN_BUILD = 22000
WINDOWS_11_25H2_MIN_BUILD = 26200

# This is a deliberately pinned local snapshot. A newly released GPU is not
# guessed to be compatible until its model/GFX target is added here.
AMD_GFX_PATTERNS = (
    ("gfx1201", (r"rx\s*9070(?:\s*(?:xt|gre))?", r"r9700s?", r"r9600d")),
    ("gfx1200", (r"rx\s*9060(?:\s*xt(?:\s*lp)?)?",)),
    ("gfx1100", (r"rx\s*7900\s*(?:xtx|xt|gre)", r"w7900", r"w7800")),
    # The trailing \b keeps the laptop RX 7700S, a gfx1102 part, out of gfx1101.
    ("gfx1101", (r"rx\s*7800\s*xt", r"rx\s*7700(?:\s*xt)?\b", r"w7700", r"v710")),
    # AMD's ROCm 7.14 list names only the RX 7600; the RX 7600 XT and RX 7650
    # GRE, the laptop RX 7600S, RX 7600M, RX 7600M XT and RX 7700S, and the
    # Radeon PRO W7600 and W7500 are the same gfx1102 silicon (Navi 33), and
    # the installer validates the backend before use. lspci names a PCI ID, not
    # a product:
    # the pci.ids in Ubuntu 24.04 (2024.03.31) call every 1002:7480 card
    # "Navi 33 [Radeon RX 7700S/7600/7600S/7600M XT/PRO W7600]", so there the
    # RX 7700S and PRO W7600 patterns are what map an RX 7600 or 7600 XT.
    ("gfx1102", (r"rx\s*7600(?:\s*xt|s|m(?:\s*xt)?)?\b", r"rx\s*7650\s*gre\b", r"rx\s*7700s\b", r"w7[56]00")),
    ("gfx1030", (r"w6800", r"v620")),
    ("gfx1103", (r"radeon\s*7[468]0m",)),
    ("gfx1150", (r"radeon\s*(?:890m|880m)", r"ryzen\s*ai\s*9\s*hx\s*(?:375|370)")),
    ("gfx1151", (r"radeon\s*(?:8060s|8050s)", r"ryzen\s*ai\s*max")),
    # AMD's 7.14.1 list has the Radeon 840M under two targets: the Ryzen AI 5
    # 430, 435 and PRO 435 and the AI 7 445 are gfx1153, the AI 7 345 and AI 5
    # 340, PRO 340 and PRO 440 are gfx1152. Only the processor name tells them
    # apart, so this row names processors and comes before gfx1152's 840M.
    ("gfx1153", (r"ryzen\s*ai\s*[57]\s*(?:pro\s*)?4(?:3[05]|45)\b",)),
    ("gfx1152", (r"radeon\s*(?:860m|840m|820m)", r"ryzen\s*ai\s*[57]\s*3[45]0")),
)
ROCM_714_TARGETS = frozenset(target for target, _patterns in AMD_GFX_PATTERNS)
# Integrated GPU names AMD gives to more than one target (see gfx1153): for
# these the processor name decides whenever it names a target.
AMD_SHARED_IGPU_PATTERNS = (r"radeon\s*840m",)

# Every GFX target the pinned ROCm PyTorch wheel (ROCM_TORCH_VERSION in
# Install_Dependencies) declares a `device-<target>` extra for; the extra pulls
# in that target's `amd-torch-device-*` kernels. Verified against the metadata
# of the Linux torch 2.12.0+rocm7.14.1 wheel from
# https://repo.amd.com/rocm/whl-multi-arch/ on 2026-09-29. The Windows wheel
# declares every AMD_GFX_PATTERNS target.
#
# Check the pinned wheel, not the channel's package list: the channel publishes
# device packages for other torch versions too (`gfx1250` for 2.11.0 only), and
# uv, like pip, merely warns about an extra the wheel does not declare and then
# installs torch without kernels for that GPU.
#
# The channel also carries `gfx11`, `gfx110x`, `gfx115x` and `gfx12`. Those are
# multi-architecture bundles, not device identities, so they are excluded: a
# device must resolve to exactly one target for `torch[device-<target>]`.
#
# This set is wider than AMD_GFX_PATTERNS on purpose. The pattern table maps a
# product NAME to a target and is the only option on Windows, where nothing
# reports the GFX target directly. On Linux the kernel reports it (_kfd_targets),
# or else the ROCm runtime (_linux_rocm_fallback_target), so the eligibility
# gate there uses this set instead.
ROCM_CHANNEL_TARGETS = frozenset({
    "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201",
})

# Linux releases for ROCm. Every target runs on these Ubuntu releases and on
# distributions built on them (Linux Mint, Pop!_OS, ...), which name their
# Ubuntu base in UBUNTU_CODENAME; their own VERSION_ID numbers differ.
LINUX_ROCM_UBUNTU_RELEASES = ("22.04", "24.04", "25.10", "26.04")
UBUNTU_CODENAME_RELEASES = {"jammy": "22.04", "noble": "24.04", "questing": "25.10", "resolute": "26.04"}
# AMD's ROCm 7.14.1 compatibility matrix also lists the Instinct MI300 (gfx942)
# and MI350 (gfx950) series on these releases, by os-release ID. Each target
# takes every release AMD lists for any of its models (MI300A, MI300X and
# MI325X differ a little); validation after installing catches the rest. A
# release given as "9" covers every 9.x; "9.4" only 9.4.
LINUX_DISTRO_NAMES = {"rhel": "RHEL", "rocky": "Rocky Linux", "ol": "Oracle Linux", "debian": "Debian", "sles": "SLES"}
INSTINCT_LINUX_RELEASES = {
    "gfx942": {
        "rhel": ("8.10", "9.4", "9.6", "9.8", "10.0", "10.2"),
        "rocky": ("9",),
        "ol": ("8", "9", "10"),
        "debian": ("12", "13"),
        "sles": ("15.7", "16.0"),
    },
    "gfx950": {
        "rhel": ("8.10", "9.4", "9.6", "9.8", "10.0", "10.2"),
        "ol": ("9", "10"),
        "debian": ("12", "13"),
        "sles": ("15.7", "16.0"),
    },
}
INTEGRATED_AMD_TARGETS = frozenset({"gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"})

INTEL_ARC_PREFIX = r"\bintel\b.*\barc(?:\(tm\)|™)?\s+"
INTEL_SERIES_3_ARC_PATTERN = INTEL_ARC_PREFIX + r"(?:pro\s+)?b3[79]0\b"
INTEL_INTEGRATED_ARC_PATTERNS = (
    INTEL_ARC_PREFIX + r"(?:pro\s+)?1[34]0[vt]\b",
    INTEL_SERIES_3_ARC_PATTERN,
)
# For an Intel GPU without a PCI device ID (plain lspci), the name decides. Arc
# Pro cards are the same Alchemist and Battlemage silicon as Arc A and B, some
# with two-digit model numbers (Pro A60, Pro B50, Pro B60).
INTEL_XPU_PATTERNS = (
    INTEL_ARC_PREFIX + r"[ab]-?series\b",
    INTEL_ARC_PREFIX + r"(?:pro\s+)?[ab]\d{2,3}",
    INTEL_ARC_PREFIX + r"(?:graphics|gpu)\b",
    *INTEL_INTEGRATED_ARC_PATTERNS,
    r"intel.*data\s+center\s+gpu\s+max",
)


class IntelPlatform(NamedTuple):
    name: str
    kind: str
    windows: bool  # the Windows wheel carries kernels for it
    linux: dict[str, tuple[str, ...]]  # distribution: releases, where the Linux wheel does
    device_ids: tuple[str, ...] = ()


INTEL_UBUNTU = {"ubuntu": ("24.04", "25.10", "26.04")}
# PyTorch lists Panther Lake for Ubuntu 25.10, whose Linux 6.17 xe driver it needs.
INTEL_UBUNTU_6_17 = {"ubuntu": ("25.10", "26.04")}
INTEL_MAX_LINUX = {"ubuntu": ("22.04",), "rhel": ("9.2",), "redhat": ("9.2",), "sles": ("15",), "suse": ("15",)}
# The Intel GPUs the pinned PyTorch 2.12.0 XPU wheels carry compiled kernels
# for, by PCI device ID. torch-xpu-ops 62b793f (PyTorch v2.12.0's
# third_party/xpu.txt) compiles ahead of time for Windows
# "mtl,mtl-h,bmg,dg2,arl-h,lnl-m,ptl" and Linux
# "pvc,bmg,dg2,arl-h,mtl-h,lnl-m,ptl-h,ptl-u" (cmake/BuildFlags.cmake), and
# Intel's compute runtime also runs dg2's kernels on ACM-G11 and G12, bmg's on
# BMG-G31 and ptl's on Panther Lake-U. Device IDs from compute-runtime
# 26.35.39758.10 (shared/source/*/device_ids_configs_*.h), less pre-production
# parts.
#
# Any other Intel GPU could only compile the wheel's SPIR-V copy on first use,
# and PyTorch lists none of them: Meteor Lake-U on Linux, Arrow Lake-U and -S,
# Wildcat Lake, Data Center GPU Flex, other Max parts, and the Xe-LP graphics
# before Meteor Lake, which lack the 64-bit floating point the kernels need.
INTEL_XPU_PLATFORMS = (
    IntelPlatform("Arc A-series (Alchemist)", "discrete", True, INTEL_UBUNTU, (
        "5690", "5691", "5692", "56a0", "56a1", "56a2", "56be", "56bf",  # ACM-G10
        "5693", "5694", "5695", "56a5", "56a6", "56b0", "56b1", "56ba", "56bb", "56bc", "56bd",  # ACM-G11
        "5696", "5697", "56a3", "56a4", "56b2", "56b3",  # ACM-G12
    )),
    IntelPlatform("Arc B-series (Battlemage)", "discrete", True, INTEL_UBUNTU, (
        "e202", "e209", "e20b", "e20c", "e20d", "e210", "e211", "e212", "e215", "e216",  # BMG-G21
        "e220", "e221", "e222", "e223",  # BMG-G31
    )),
    IntelPlatform("Meteor Lake-H", "integrated", True, INTEL_UBUNTU, ("7d55", "7dd5")),
    # The Linux wheel leaves out "mtl", Meteor Lake-U's target.
    IntelPlatform("Meteor Lake-U", "integrated", True, {}, ("7d40", "7d45")),
    IntelPlatform("Arrow Lake-H", "integrated", True, INTEL_UBUNTU, ("7d51", "7dd1")),
    IntelPlatform("Lunar Lake", "integrated", True, INTEL_UBUNTU, ("6420", "64a0", "64b0")),
    IntelPlatform("Panther Lake", "integrated", True, INTEL_UBUNTU_6_17, (
        "b080", "b081", "b082", "b083", "b084", "b085", "b086", "b087", "b08f", "b090", "b0a0", "b0b0",
    )),
    IntelPlatform("Data Center GPU Max", "discrete", False, INTEL_MAX_LINUX, (
        "0b69", "0b6e", "0bd5", "0bd6", "0bd7", "0bd8", "0bd9", "0bda", "0bdb",
    )),
)
INTEL_XPU_DEVICE_IDS = {device: row for row in INTEL_XPU_PLATFORMS for device in row.device_ids}
INTEL_XPU_PLATFORM_NAMED = {row.name: row for row in INTEL_XPU_PLATFORMS}
INTEL_ARC_BY_NAME = IntelPlatform("Intel Arc", "unknown", True, INTEL_UBUNTU)
# Intel's discrete GPUs by PCI device ID, from Linux's include/drm/intel/pciids.h
# (October 2026). Every other Intel display device is a processor's graphics
# (UHD, Iris Xe, Arc and plain "Intel Graphics" alike), so an Intel GPU
# outside INTEL_XPU_PLATFORMS is integrated unless it is listed here. Linux's
# gma500 driver gives 0be0-0bef to the 2011 Atom D2000/N2000 graphics too;
# those machines are far below what EMAP-SSN needs, so the three IDs that
# overlap count as Ponte Vecchio.
INTEL_DISCRETE_DEVICE_IDS = frozenset({
    "4905", "4906", "4907", "4908", "4909",  # DG1 (Iris Xe MAX)
    "5690", "5691", "5692", "56a0", "56a1", "56a2", "56be", "56bf",  # DG2-G10
    "5693", "5694", "5695", "56a5", "56a6", "56b0", "56b1", "56ba", "56bb", "56bc", "56bd",  # DG2-G11
    "5696", "5697", "56a3", "56a4", "56b2", "56b3",  # DG2-G12
    "56c0", "56c1", "56c2",  # ATS-M (Data Center GPU Flex)
    "0b69", "0b6e", "0bd4", "0bd5", "0bd6", "0bd7", "0bd8", "0bd9", "0bda", "0bdb",
    "0be0", "0be1", "0be5",  # Ponte Vecchio (Data Center GPU Max)
    "e202", "e209", "e20b", "e20c", "e20d", "e210", "e211", "e212", "e216",  # BMG-G21
    "e220", "e221", "e222", "e223",  # BMG-G31
    "674c", "674d", "674e", "674f", "6750",  # Crescent Island
})
# For an Intel GPU without a PCI device ID, the names Intel's driver and
# pci.ids give its discrete GPUs, then those they give processor graphics.
INTEL_DISCRETE_NAME_PATTERN = (
    r"iris(?:\(r\)|®)?\s*xe\s*max|\bdg1\b|\bats-m\b|data\s+center\s+gpu\s+(?:flex|max)"
)
INTEL_INTEGRATED_NAME_PATTERN = (
    r"\b(?:uhd|iris|hd)\b.*\bgraphics\b|integrated\s+graphics|\bintel(?:\(r\)|®)?\s+graphics\b"
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


def _windows_names(class_name: str) -> list[str]:
    """Compatibility wrapper returning CIM object names."""
    return [str(row.get("Name")) for row in _windows_cim(class_name, ("Name",)) if row.get("Name")]


def _amd_accelerator_line(line: str) -> bool:
    """Tell whether an lspci line is an AMD GPU of PCI class "Processing accelerators".

    AMD's Instinct MI300 series presents that class (12) instead of a display
    class, and amdgpu binds it by that class. NPUs, FPGAs, DPUs and other
    vendors' AI cards share the class, AMD's own Pensando DPUs among them
    (vendor 1dd8, "AMD Pensando Systems"). So it counts only for AMD graphics
    devices: vendor 1002 with `lspci -nn`, which plain `lspci` names "[AMD/ATI]".
    """
    if not re.search(r"Processing accelerators", line, re.I):
        return False
    pci_id = _pci_id(line)
    return pci_id.startswith("1002:") if pci_id else "[amd/ati]" in line.lower()


def _controller_names(system: str) -> list[str]:
    """Compatibility wrapper returning display-controller descriptions."""
    if system == "windows":
        return _windows_names("Win32_VideoController")
    if system == "linux":
        return [
            line
            for line in _lines(_run(["lspci"]))
            if re.search(r"VGA|3D controller|Display controller", line, re.I)
            or _amd_accelerator_line(line)
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


def _pci_address(value: Any) -> str | None:
    """Return a canonical PCI domain:bus:device.function address."""
    match = re.search(
        r"(?<![0-9a-f])(?:([0-9a-f]{4,8}):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])(?![0-9a-f])",
        str(value or ""),
        re.I,
    )
    if not match:
        return None
    domain, bus, device, function = match.groups()
    return f"{int(domain or '0', 16):x}:{int(bus, 16):02x}:{int(device, 16):02x}.{function.lower()}"


def _gpu_name_aliases(value: Any) -> set[str]:
    """Build conservative aliases for equivalent OS and vendor GPU names."""
    text = str(value or "")
    parts = [text, *re.findall(r"\[([^]]+)\]", text)]
    ignored = {
        "3d", "compatible", "controller", "corp", "corporation", "display",
        "inc", "limited", "ltd", "nvidia", "vga",
    }
    aliases: set[str] = set()
    for part in parts:
        tokens = [token.lower() for token in re.findall(r"[a-z0-9]+", part, re.I)]
        alias = " ".join(token for token in tokens if token not in ignored)
        if alias:
            aliases.add(alias)
    return aliases


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


def _amd_name_looks_discrete(name: str) -> bool:
    text = name.lower()
    return bool(re.search(r"\b(?:rx|pro\s+w|v)\s*\d", text)) or "radeon ai pro" in text


def _amd_device_target(name: str, processors: list[str]) -> str | None:
    """Return one AMD display device's GFX target, or None when it is unknown.

    The device's own name decides whenever the table knows it. Windows names
    an APU's iGPU generically ("AMD Radeon(TM) Graphics"), so for a name the
    table does not know, the CPU names identify the APU instead ("Ryzen AI 9
    HX 370" is gfx1150). A name with a discrete model number never takes them:
    the CPU name describes the iGPU, so an RX, PRO W or V card missing from
    the table would take the iGPU's target and be classed integrated. The one
    exception is an iGPU name AMD gives to two targets, such as the Radeon
    840M: there a CPU name that names a target decides.
    """
    target = amd_gfx_target([name])
    if target and any(re.search(pattern, name, re.I) for pattern in AMD_SHARED_IGPU_PATTERNS):
        return amd_gfx_target(processors) or target
    if target or _amd_name_looks_discrete(name):
        return target
    return amd_gfx_target(processors)


def _intel_platform(pci_id: str | None, name: str) -> IntelPlatform | None:
    """The XPU platform of an Intel GPU, or None when the pinned wheels have no kernels for it.

    The PCI device ID decides: the names in pci.ids and in Intel's driver
    follow branding, and the same GPU is "Intel Arc Graphics" or "Intel
    Graphics" depending on the memory fitted. Only without an ID does the name.
    """
    if pci_id:
        vendor, _, device = pci_id.partition(":")
        return INTEL_XPU_DEVICE_IDS.get(device) if vendor == "8086" else None
    if not any(re.search(pattern, name, re.I) for pattern in INTEL_XPU_PATTERNS):
        return None
    lowered = name.lower()
    if "data center gpu max" in lowered:
        return INTEL_XPU_PLATFORM_NAMED["Data Center GPU Max"]
    if "series 3" in lowered or "panther" in lowered or re.search(INTEL_SERIES_3_ARC_PATTERN, name, re.I):
        return INTEL_XPU_PLATFORM_NAMED["Panther Lake"]
    return INTEL_ARC_BY_NAME


def _device_kind(
    vendor: str, name: str, gfx_target: str | None, processors: list[str], pci_id: str | None = None
) -> str:
    text = name.lower()
    if vendor == "NVIDIA":
        return "discrete"
    if vendor == "AMD":
        if gfx_target in INTEGRATED_AMD_TARGETS or re.search(r"radeon\s+\d{3,4}m", text):
            return "integrated"
        if _amd_name_looks_discrete(name):
            return "discrete"
    if vendor == "INTEL":
        intel_platform = _intel_platform(pci_id, name) if pci_id else None
        if intel_platform:
            return intel_platform.kind
        vendor_id, _, device_id = (pci_id or "").partition(":")
        if vendor_id == "8086":
            return "discrete" if device_id in INTEL_DISCRETE_DEVICE_IDS else "integrated"
        # Numbered integrated Arc GPUs do not require a CPU-name probe. In
        # particular, B370/B390 are integrated despite their B-series names.
        if any(re.search(pattern, name, re.I) for pattern in INTEL_INTEGRATED_ARC_PATTERNS):
            return "integrated"
        if re.search(INTEL_DISCRETE_NAME_PATTERN, text) or re.search(r"arc.*\b[ab]\d{2,3}", text):
            return "discrete"
        if "arc" in text and any("core" in value.lower() and "ultra" in value.lower() for value in processors):
            return "integrated"
        if re.search(INTEL_INTEGRATED_NAME_PATTERN, text):
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
        gfx = _amd_device_target(name, processors) if vendor == "AMD" else None
        devices.append(
            {
                "id": pnp_id or pci_id or f"windows:{index}",
                "name": name,
                "vendor": vendor,
                "pci_id": pci_id,
                "driver_version": str(row.get("DriverVersion") or "") or None,
                "kind": _device_kind(vendor, name, gfx, processors, pci_id),
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
        gfx = _amd_device_target(name, processors) if vendor == "AMD" else None
        devices.append({
            "id": f"{source}:{index}", "name": name, "vendor": vendor,
            "pci_id": pci_id, "driver_version": None, "driver": None,
            "kind": _device_kind(vendor, name, gfx, processors, pci_id),
            "architecture": gfx, "source": source,
        })
    return devices


def _linux_inventory(processors: list[str]) -> list[dict[str, Any]]:
    lines = _lines(_run(["lspci", "-Dnnk"], timeout=8))
    devices: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        if (
            re.search(r"VGA compatible controller|3D controller|Display controller", line, re.I)
            or _amd_accelerator_line(line)
        ):
            match = re.match(r"(\S+)\s+[^:]+:\s*(.+)", line)
            if not match:
                continue
            address, name = match.groups()
            pci_id = _pci_id(name)
            vendor = _vendor(name, pci_id)
            gfx = _amd_device_target(name, processors) if vendor == "AMD" else None
            current = {
                "id": address,
                # lspci -nn ends the name with "[vendor:device]", then "(rev xx)"
                # when the revision is nonzero; pci_id keeps the numeric ID.
                "name": re.sub(
                    r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\](?:\s*\(rev [0-9a-f]{2}\))?\s*$", "", name, flags=re.I
                ),
                "vendor": vendor,
                "pci_id": pci_id,
                "driver_version": None,
                "driver": None,
                "kind": _device_kind(vendor, name, gfx, processors, pci_id),
                "architecture": gfx,
                "source": "lspci",
            }
            devices.append(current)
        elif re.match(r"(?:[0-9a-f]{4,8}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\s", line, re.I):
            # Any other device's header ends the GPU's detail lines; _lines()
            # strips their indentation, so only the leading PCI address tells
            # them apart. Otherwise the GPU's HDMI audio function, listed right
            # after it, would hand the GPU its snd_hda_intel driver.
            current = None
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
        device_address = _pci_address(device.get("bus_id")) or _pci_address(device.get("id"))
        address_matches = [
            item for item in unused
            if device_address and _pci_address(item.get("bus_id")) == device_address
        ]
        match = address_matches[0] if len(address_matches) == 1 else None
        if match is None:
            # A disagreeing address is stronger evidence than a similar model name.
            # When the OS inventory has no usable address (as on Windows), names
            # remain the only available join key. Otherwise name matching is
            # reserved for the older nvidia-smi query that omits PCI addresses.
            name_candidates = [
                item for item in unused
                if (device_address is None or _pci_address(item.get("bus_id")) is None)
                and _gpu_name_aliases(device.get("name")) & _gpu_name_aliases(item.get("name"))
            ]
            if name_candidates:
                candidate_aliases = {
                    frozenset(_gpu_name_aliases(item.get("name")))
                    for item in name_candidates
                }
                if len(name_candidates) == 1 or len(candidate_aliases) == 1:
                    match = name_candidates[0]
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


def _linux_rocm_fallback_target(rocm_targets: set[str]) -> str | None:
    """Resolve a GFX target from the ROCm runtime when name matching fails.

    On Linux the installed ROCm stack reports the real target, which is ground
    truth and does not depend on AMD_GFX_PATTERNS recognizing a product name.
    That covers hardware the pinned snapshot has no pattern for, Instinct parts
    in particular, whose lspci names never matched.

    Only applied when exactly one supported target is reported. The agent list
    is unordered and carries no PCI address, so with two distinct targets there
    is no reliable way to attribute one to a specific lspci entry; the
    conservative unmapped result is kept instead of guessing. A target outside
    ROCM_CHANNEL_TARGETS is also ignored: the pinned torch wheel declares no
    `device-<target>` extra for it, so the install would only warn and leave
    out that GPU's kernels.
    """
    supported = {target for target in rocm_targets if target in ROCM_CHANNEL_TARGETS}
    return next(iter(supported)) if len(supported) == 1 else None


KFD_TOPOLOGY_NODES = "/sys/class/kfd/kfd/topology/nodes"


def _kfd_nodes() -> list[dict[str, int]]:
    """The numeric properties of every node in the kernel's KFD topology.

    amdgpu's KFD lists a node per CPU and per GPU that can run ROCm, in sysfs,
    with no ROCm software installed. WSL, containers without /sys/class/kfd and
    kernels without the KFD list none.
    """
    try:
        folders = sorted(Path(KFD_TOPOLOGY_NODES).iterdir())
    except OSError:
        return []
    nodes = []
    for folder in folders:
        try:
            text = (folder / "properties").read_text(encoding="ascii", errors="replace")
        except OSError:
            continue
        nodes.append({
            parts[0]: int(parts[1])
            for parts in (line.split() for line in text.splitlines())
            if len(parts) == 2 and parts[1].isdigit()
        })
    return nodes


def _kfd_targets(nodes: list[dict[str, int]]) -> dict[str, str]:
    """Each GPU node's GFX target, by the PCI address _pci_address() gives.

    gfx_target_version is major * 10000 + minor * 100 + stepping, and LLVM
    names minor and stepping in hex: 90010 is gfx90a, 90402 gfx942. CPU nodes
    report 0. location_id is the bus, device and function, (bus << 8) |
    (device << 3) | function.
    """
    targets: dict[str, str] = {}
    for node in nodes:
        version = node.get("gfx_target_version", 0)
        location = node.get("location_id")
        if version <= 0 or location is None:
            continue
        address = f"{node.get('domain', 0):x}:{location >> 8:02x}:{(location >> 3) & 0x1f:02x}.{location & 7}"
        targets[address] = f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"
    return targets


def _intel_os_supported(system: str, os_info: dict[str, Any], intel_platform: IntelPlatform) -> bool:
    if system == "windows":
        return int(os_info.get("windows_build") or 0) >= WINDOWS_11_MIN_BUILD
    distro = str(os_info.get("id") or "").lower()
    return system == "linux" and str(os_info.get("version_id") or "").startswith(intel_platform.linux.get(distro, ()))


def _linux_rocm_release(os_info: dict[str, Any], target: str) -> str | None:
    """The supported Linux release this system is for ROCm on target, or None."""
    distro = str(os_info.get("id") or "").lower()
    version = str(os_info.get("version_id") or "")
    if distro == "ubuntu" and version.startswith(LINUX_ROCM_UBUNTU_RELEASES):
        return f"Ubuntu {version}"
    base = UBUNTU_CODENAME_RELEASES.get(str(os_info.get("ubuntu_codename") or "").lower())
    if base and "ubuntu" in str(os_info.get("id_like") or "").lower().split():
        return f"Ubuntu {base} ({os_info.get('name') or distro})"
    parts = version.split(".")
    for release in INSTINCT_LINUX_RELEASES.get(target, {}).get(distro, ()):
        listed = release.split(".")
        if parts[:len(listed)] == listed:
            return f"{os_info.get('name') or distro} {version}"
    return None


def _nvidia_driver_warning(device: dict[str, Any], system: str = "") -> str | None:
    """Why no installable build can run this NVIDIA GPU, or None.

    The compute capability and driver version both come from nvidia-smi, so the
    driver is NVIDIA's own version number, not the Windows driver package's.
    """
    capability = _version_tuple(str(device.get("compute_capability") or ""))
    driver = _version_tuple(str(device.get("driver_version") or ""))
    if not (capability and driver):
        return None
    name = device.get("name") or "This NVIDIA GPU"
    installed = device.get("driver_version")
    if capability < CUDA_12_6_MIN_CAPABILITY:
        return (
            f"{name} has compute capability {device.get('compute_capability')}; the CUDA builds "
            "EMAP-SSN installs need 5.0 or newer, so it uses the CPU instead."
        )
    if capability > CUDA_12_6_MAX_CAPABILITY and driver < CUDA_13_MIN_DRIVER:
        return (
            f"{name} needs NVIDIA driver 580 or newer (installed: {installed}). "
            "Update the driver to use this GPU; until then EMAP-SSN does not use it."
        )
    minimum = CUDA_12_MIN_DRIVER.get(system)
    if minimum and driver < minimum:
        return (
            f"{name} needs NVIDIA driver {'.'.join(map(str, minimum))} or newer for CUDA 12 "
            f"(installed: {installed}). Update the driver to use this GPU; until then "
            "EMAP-SSN does not use it."
        )
    return None


def _nvidia_backend(devices: list[dict[str, Any]]) -> tuple[str, str]:
    capabilities = [_version_tuple(str(device.get("compute_capability") or "")) for device in devices]
    drivers = [_version_tuple(str(device.get("driver_version") or "")) for device in devices]
    modern = bool(capabilities) and all(value >= (7, 5) for value in capabilities if value) and all(capabilities)
    driver_ready = bool(drivers) and all(value >= CUDA_13_MIN_DRIVER for value in drivers if value) and all(drivers)
    if modern and driver_ready:
        return "cuda132", "All selected NVIDIA GPUs and the installed driver support CUDA 13.2."
    return "cuda126", "CUDA 12.6 provides the common compatible NVIDIA profile."


def _leave_out_nvidia_without_kernels(devices: list[dict[str, Any]]) -> None:
    """Leave out a GPU newer than Hopper when the NVIDIA GPUs get CUDA 12.6.

    One CUDA build serves every NVIDIA GPU, and CUDA 13.2 needs all of them at
    compute capability 7.5 or newer. Beside an older GPU the CUDA 12.6 build is
    installed, which has no kernels for a GPU newer than Hopper: validation
    passed on the older GPU alone and the newer one went unused without a word.
    """
    nvidia = [device for device in devices if "cuda" in device["eligible_profiles"]]
    if not nvidia or _nvidia_backend(nvidia)[0] != "cuda126":
        return
    newer = [
        device for device in nvidia
        if _version_tuple(str(device.get("compute_capability") or "")) > CUDA_12_6_MAX_CAPABILITY
    ]
    holding_back = [device for device in nvidia if device not in newer and _nvidia_backend([device])[0] == "cuda126"]
    if not holding_back:
        return
    names = " and ".join(str(device.get("name") or "an NVIDIA GPU") for device in holding_back)
    for device in newer:
        name = device.get("name") or "This NVIDIA GPU"
        warning = (
            f"{name} runs only on the CUDA 13.2 build, which EMAP-SSN installs only when every "
            f"NVIDIA GPU has compute capability 7.5 or newer and driver 580 or newer. {names} "
            f"{'does' if len(holding_back) == 1 else 'do'} not, so the CUDA 12.6 build is installed "
            f"and EMAP-SSN does not use {name}."
        )
        device.update(warning=warning, eligibility="ineligible", eligible_profiles=[], profile_eligibility={})
        device["reasons"].append(warning)


def _rank(device: dict[str, Any]) -> tuple[int, int, str]:
    return (
        DEVICE_CLASS_ORDER.get(str(device.get("kind")), 1),
        VENDOR_ORDER.get(str(device.get("vendor")), 9),
        str(device.get("id")),
    )


def _evaluate_devices(
    devices: list[dict[str, Any]],
    system: str,
    os_info: dict[str, Any],
    rocm_targets: set[str],
    kfd_targets: dict[str, str] | None = None,
) -> None:
    windows_build = os_info.get("windows_build")
    linux_kfd = Path("/dev/kfd")
    for device in devices:
        vendor = device["vendor"]
        profiles: list[str] = []
        reasons: list[str] = []
        status = "ineligible"
        if vendor == "NVIDIA":
            warning = _nvidia_driver_warning(device, system)
            if warning:
                # A CUDA build would install and then fail validation on this
                # GPU; say why instead, and leave it out of the ladder.
                device["warning"] = warning
                reasons.append(warning)
            else:
                profiles = ["cuda"]
                status = "eligible" if device.get("driver_version") else "provisional"
                reasons.append("NVIDIA display device detected; runtime validation is required.")
        elif vendor == "AMD":
            target = device.get("architecture")
            kfd_target = None
            kfd_missing = False
            if system == "linux" and kfd_targets:
                # The kernel reports each GPU's real target by PCI address, so
                # it decides over a name match: pci.ids names a PCI ID, and one
                # name can cover GPUs of several targets.
                address = _pci_address(device.get("id")) or _pci_address(device.get("name"))
                kfd_target = kfd_targets.get(address) if address else None
                kfd_missing = address is not None and kfd_target is None
                if kfd_target:
                    if target and target != kfd_target:
                        reasons.append(f"The kernel reports {kfd_target} for this GPU; its name suggested {target}.")
                    target = kfd_target
                    device["architecture"] = target
                    device["architecture_source"] = "kfd"
                    if device.get("kind") == "unknown" and target in INTEGRATED_AMD_TARGETS:
                        device["kind"] = "integrated"
            if not target and system == "linux" and kfd_target is None and not kfd_missing:
                # The product name did not match the pinned snapshot, but the
                # ROCm runtime reports the real target. Prefer that over
                # declaring the device unsupported.
                target = _linux_rocm_fallback_target(rocm_targets or set((kfd_targets or {}).values()))
                if target:
                    device["architecture"] = target
                    device["architecture_source"] = "rocm_agent_enumerator"
                    reasons.append(
                        f"GFX target {target} resolved from the ROCm runtime; "
                        "the model is not in the pinned name table."
                    )
            if kfd_missing:
                # The kernel lists the other GPUs but none at this address, so
                # ROCm cannot use this one, whatever its name or the runtime say.
                reasons.append("The kernel's ROCm driver (KFD) lists no compute node for this GPU.")
            elif not target:
                reasons.append("AMD model is not in the pinned GFX compatibility table.")
            elif system == "windows":
                # ROCm 7.14 is the only Windows profile. AMD publishes it for
                # Windows 11 25H2 only, so an older Windows 11 build is not
                # eligible for ROCm at all and falls through to the next
                # accelerator or CPU.
                if windows_build is None or int(windows_build) < WINDOWS_11_25H2_MIN_BUILD:
                    reasons.append(
                        "Native Windows ROCm requires Windows 11 25H2 build "
                        f"{WINDOWS_11_25H2_MIN_BUILD} or newer."
                    )
                elif target in ROCM_714_TARGETS:
                    profiles.append("rocm")
                    status = "eligible" if device.get("driver_version") else "provisional"
                    device.setdefault("profile_eligibility", {})["rocm"] = status
                    reasons.append(f"{target} is listed for: rocm.")
                else:
                    reasons.append(
                        f"{target} is not in the pinned Windows GFX compatibility table."
                    )
            elif system == "linux":
                release = _linux_rocm_release(os_info, target)
                driver_ready = linux_kfd.exists() and os.access(linux_kfd, os.R_OK | os.W_OK)
                target_seen = kfd_target is not None or not rocm_targets or target in rocm_targets
                if release is None:
                    others = [LINUX_DISTRO_NAMES[distro] for distro in INSTINCT_LINUX_RELEASES.get(target, {})]
                    reasons.append(
                        "Linux ROCm needs Ubuntu 22.04, 24.04, 25.10 or 26.04, or a distribution "
                        "built on one of them"
                        + (
                            f", or a {', '.join(others[:-1])} or {others[-1]} release that AMD lists "
                            f"for {target}." if others else "."
                        )
                    )
                elif not driver_ready:
                    reasons.append("/dev/kfd is missing or inaccessible; the installer does not install system GPU drivers.")
                elif not target_seen:
                    reasons.append(f"ROCm agents do not report the mapped target {target}.")
                elif target in ROCM_CHANNEL_TARGETS:
                    # Linux gates on the targets the pinned torch wheel ships
                    # kernels for rather than on the name-pattern table, because
                    # the target here is reported by the kernel or the ROCm
                    # runtime, or name-matched (provisional).
                    profiles = ["rocm"]
                    status = "eligible" if kfd_target or rocm_targets else "provisional"
                    device["profile_eligibility"] = {
                        profile: status for profile in profiles
                    }
                    reasons.append(
                        f"Linux ROCm preflight accepted {target} for: {', '.join(profiles)}."
                    )
                else:
                    reasons.append(
                        f"{target} has no device package for the pinned PyTorch "
                        "on the ROCm 7.14 channel."
                    )
            else:
                reasons.append("ROCm is not configured for this operating system.")
        elif vendor == "INTEL":
            # Installation eligibility is about compatibility, not speed.
            # Validated integrated Arc devices participate in workload benchmarks.
            pci_id = device.get("pci_id")
            intel_platform = _intel_platform(pci_id, device["name"])
            if intel_platform is None:
                reasons.append(
                    f"The pinned PyTorch XPU build has no kernels for this Intel GPU ({pci_id})." if pci_id
                    else "Intel adapter is not an Arc/Core Ultra Arc/Data Center GPU Max device."
                )
            elif system == "windows" and "intel" not in device["name"].lower():
                # Without Intel's driver, Windows shows the GPU under its own
                # name, such as "Microsoft Basic Display Adapter".
                reasons.append(
                    f"Windows lists this Intel GPU as {device['name']}: Intel's graphics driver is not installed."
                )
            elif not (intel_platform.windows if system == "windows" else system == "linux" and intel_platform.linux):
                reasons.append(f"The pinned PyTorch XPU build for this OS has no kernels for {intel_platform.name}.")
            elif not _intel_os_supported(system, os_info, intel_platform):
                reasons.append(f"Intel XPU does not list this OS release for {intel_platform.name}.")
            else:
                profiles = ["xpu"]
                status = "eligible" if device.get("driver_version") or device.get("driver") else "provisional"
                reasons.append(f"{intel_platform.name} and this OS are in the pinned XPU compatibility table.")
        else:
            reasons.append("Display-controller vendor is not supported for compute acceleration.")
        device["eligibility"] = status
        device["eligible_profiles"] = profiles
        device.setdefault("profile_eligibility", {profile: status for profile in profiles})
        device["reasons"] = reasons
    _leave_out_nvidia_without_kernels(devices)


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


def _normalized_hardware_name(value: Any) -> str:
    """Return a comparison-safe model label without node-specific spacing."""
    return " ".join(str(value or "").strip().lower().split())


def hardware_compatibility_material(report: dict[str, Any]) -> dict[str, Any]:
    """Return stable hardware facts that determine backend compatibility.

    PCI bus addresses, enumeration order, exact driver releases, and detection
    source are intentionally excluded.  They commonly change across equivalent
    HPC nodes and after harmless desktop driver refreshes.  A driver change that
    crosses a supported CUDA profile boundary is still represented by the
    normalized backend candidate.
    """
    devices = []
    for device in report.get("devices", []):
        if not isinstance(device, dict):
            continue
        devices.append({
            "name": _normalized_hardware_name(device.get("name")),
            "vendor": str(device.get("vendor") or "").upper(),
            "pci_id": str(device.get("pci_id") or "").lower() or None,
            "kind": device.get("kind"),
            "architecture": str(device.get("architecture") or "").lower() or None,
            "compute_capability": str(device.get("compute_capability") or "") or None,
            "eligible_profiles": sorted(
                str(value) for value in device.get("eligible_profiles", [])
            ),
        })
    devices.sort(key=lambda value: json.dumps(value, sort_keys=True, default=str))

    candidates = []
    for candidate in report.get("backend_candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidates.append({
            "backend": candidate.get("backend"),
            "profile": candidate.get("profile"),
            "vendor": candidate.get("vendor"),
            "gfx_target": candidate.get("gfx_target"),
            "eligibility": candidate.get("eligibility"),
        })

    return {
        "compatibility_revision": report.get("compatibility_revision"),
        "platform": report.get("platform"),
        "devices": devices,
        "backend_candidates": candidates,
    }


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

    amd_on_linux = system == "linux" and any(d["vendor"] == "AMD" for d in devices)
    rocm_targets = _rocm_targets() if amd_on_linux else set()
    kfd_targets = _kfd_targets(_kfd_nodes()) if amd_on_linux else {}
    non_apple = [device for device in devices if device["vendor"] != "APPLE"]
    _evaluate_devices(non_apple, system, os_info, rocm_targets, kfd_targets)
    candidates, ignored = _candidate_ladder(devices)
    primary = candidates[0]
    primary_reason = primary["reason"]
    if primary["backend"] == "cpu":
        # Explain with the GPU that would have ranked first, and with its last
        # reason: each check appends its verdict after any notes about how the
        # GPU was identified (a GFX target from the runtime, say).
        rejected = [device for device in sorted(devices, key=_rank) if device.get("reasons")]
        if rejected:
            primary_reason = rejected[0]["reasons"][-1]
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
        "kfd_targets": kfd_targets,
        "devices": devices,
        "backend_candidates": candidates,
        "ignored_devices": ignored,
        "warnings": [device["warning"] for device in devices if device.get("warning")],
        "vendor": primary["vendor"],
        "backend": primary["backend"],
        "gfx_target": primary.get("gfx_target"),
        "reason": primary_reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete detection report")
    args = parser.parse_args(argv)
    report = detect_hardware()
    print(json.dumps(report, sort_keys=True) if args.json else report["vendor"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
