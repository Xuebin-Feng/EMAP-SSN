# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A second view of a CUDA card's memory: what other programs hold on it (Windows and WSL2).

On Windows (driver model WDDM) and under WSL2 the free figure of ``torch.cuda.mem_get_info``
does not go down when other programs (the desktop compositor, browsers) hold memory on the
card, and an allocation past the card is not refused. On Windows the overflow moves to shared
system memory and the run becomes several times slower, with no message. Under WSL2 the
allocation succeeds and its first use fails with a CUDA out-of-memory error from which the
process does not recover. In both, torch never gets the refusal at allocation that makes it
free its cache and retry. (Measured on 25 Nov 2026 with one card, an RTX 4070.)
``nvidia-smi`` counts the memory resident on the card for every process, so the difference
between its figure and torch's own pool is what other programs hold.

This module reads that figure (never raising, cached for 30 s, never required) so that the
GPU budgets can leave that memory to the other programs. Everything answers "unknown"
(None) on other systems, for devices that are not CUDA, and when SSN_GPU_OTHER_PROGRAMS=0.

Only the standard library is imported at module level; torch is imported inside the
functions, and only after the switch and the platform have been checked.
"""

from __future__ import annotations

import os

OTHER_PROGRAMS_SWITCH = "SSN_GPU_OTHER_PROGRAMS"
_OTHER_PROGRAMS_TTL_SECONDS = 30.0
_MARGIN_MIN_BYTES = 512 * 1024 ** 2
_OTHER_PROGRAMS_CACHE = {}  # device key -> (time.monotonic() stamp, bytes or None)
_NVIDIA_SMI_MISSING = False  # the program could not be started: never try again
_switch_on = None


def other_programs_switch_on():
    """Whether other programs' memory is looked at: always, unless SSN_GPU_OTHER_PROGRAMS=0. Read once."""
    global _switch_on
    if _switch_on is None:
        _switch_on = os.environ.get(OTHER_PROGRAMS_SWITCH, "").strip() != "0"
    return _switch_on


def limit_is_not_enforced_on(platform_name, kernel_release):
    """Whether a system lets GPU memory run past the card: Windows, and Linux under WSL.

    ``platform_name`` is ``sys.platform``; ``kernel_release`` is the Linux kernel release, which names
    Microsoft under WSL (for example ``6.18.58.1-microsoft-standard-WSL2``). WSL1 has no GPU, so
    matching it too is harmless. Pure; never raises.
    """
    try:
        if platform_name == "win32":
            return True
        return str(platform_name).startswith("linux") and "microsoft" in str(kernel_release).lower()
    except Exception:
        return False


def card_limit_is_not_enforced():
    """True where an allocation past the card's memory still succeeds: Windows and WSL2.

    On Windows the overflow moves to shared system memory and the run becomes several times
    slower, with no message. Under WSL2 the allocation succeeds and its first use fails with a CUDA
    out-of-memory error the process does not recover from. In both, torch never gets the refusal
    at allocation that makes it free its cache. On native Linux (and macOS) the driver refuses at
    the limit and torch then frees its own cache. Never raises.
    """
    import sys

    try:
        import platform

        release = platform.uname().release
    except Exception:
        release = ""  # an unreadable release counts as "not WSL"
    return limit_is_not_enforced_on(sys.platform, release)


def card_margin_bytes(total_bytes):
    """The memory left unused on a card: 512 MiB or 5 % of its total, whichever is more."""
    return max(_MARGIN_MIN_BYTES, int(int(total_bytes) * 0.05))


def normalized_gpu_uuid(value):
    text = str(value).strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def device_memory_in_use(device, total_bytes):
    """Return ``(used_bytes, total_bytes)`` of the card as nvidia-smi sees it, or None.

    ``total_bytes`` is the total CUDA reported; a row whose total differs by more than
    5 % is not this card. Never raises.
    """
    global _NVIDIA_SMI_MISSING
    if _NVIDIA_SMI_MISSING or not other_programs_switch_on():
        return None
    try:
        import subprocess
        import sys

        import torch

        # A ROCm build of torch also answers to torch.cuda; nvidia-smi says nothing about that card.
        if getattr(torch.version, "cuda", None) is None:
            return None
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                **kwargs,
            )
        except OSError:
            _NVIDIA_SMI_MISSING = True
            return None
        if proc.returncode != 0:
            return None
        rows = []
        for line in str(proc.stdout).splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                rows.append(None)
                continue
            rows.append((normalized_gpu_uuid(fields[0]), fields[1], fields[2]))
        if not rows:
            return None
        torch_uuid = None
        try:
            torch_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        except Exception:
            torch_uuid = None
        chosen = None
        if torch_uuid is not None:
            wanted = normalized_gpu_uuid(torch_uuid)
            for row in rows:
                if row is not None and row[0] == wanted:
                    chosen = row
                    break
        if chosen is None and len(rows) == 1:
            chosen = rows[0]
        if chosen is None:
            return None
        used_bytes = int(chosen[1]) * 1024 ** 2
        smi_total_bytes = int(chosen[2]) * 1024 ** 2
        if int(total_bytes) <= 0 or abs(smi_total_bytes - int(total_bytes)) > 0.05 * int(total_bytes):
            return None
        return used_bytes, smi_total_bytes
    except Exception:
        return None


def other_programs_bytes(device, reserved_bytes, total_bytes):
    """Bytes on the card that are not this process's torch pool, or None when unknown.

    Cached per device for 30 s (an unknown answer too), so a missing or slow
    nvidia-smi cannot be started again and again. None at once when the switch is off.
    """
    if not other_programs_switch_on():
        return None
    import time

    key = str(device)
    now = time.monotonic()
    cached = _OTHER_PROGRAMS_CACHE.get(key)
    if cached is not None and 0 <= now - cached[0] < _OTHER_PROGRAMS_TTL_SECONDS:
        return cached[1]
    others = None
    reading = device_memory_in_use(device, total_bytes)
    if reading is not None:
        # This process's small CUDA context lands in "others": fine and conservative.
        others = max(0, int(reading[0]) - int(reserved_bytes))
    _OTHER_PROGRAMS_CACHE[key] = (now, others)
    return others


def room_beside_other_programs(device, total_bytes):
    """Return ``(room, others)`` for a CUDA card on Windows or WSL2, or None when it is not known.

    ``others`` is what other programs hold; ``room`` is ``total - others - torch's reserved pool``
    (0 at least), the memory torch could still take. None on other systems, for a device that is not
    CUDA, with the switch off, or when the reader does not answer. Never raises.
    """
    try:
        if not other_programs_switch_on() or not card_limit_is_not_enforced():
            return None
        import torch

        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type != "cuda":
            return None
        total = int(total_bytes)
        reserved = int(torch.cuda.memory_reserved(dev))
        others = other_programs_bytes(dev, reserved, total)
        if others is None:
            return None
        return max(0, total - int(others) - reserved), int(others)
    except Exception:
        return None
