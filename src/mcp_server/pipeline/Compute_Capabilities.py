# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Isolated, metadata-only runtime discovery. Heavy imports belong to the helper."""

from datetime import datetime, timezone
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def capability(status, reason, source="runtime metadata"):
    return dict(status=status, reason=reason, source=source, computation_verified=False)


def empty_report():
    return dict(status="unavailable", observed_at=datetime.now(timezone.utc).isoformat(),
                python_version=platform.python_version(), pytorch_version=None,
                cuda_version=None, rocm_version=None, installer_filtering=None,
                cpu=dict(logical_count=None, physical_count=None),
                system_memory=dict(total_bytes=None, available_bytes=None),
                devices=[], errors=[], metadata_only=True)


def collect_metadata(torch, hardware, psutil=None):
    report = empty_report()

    def query(field, fn):
        try:
            value = fn()
            if value is None and field != "installer_filtering":
                raise ValueError("Metadata is unavailable.")
            return value
        except Exception as error:
            report["errors"].append(dict(field=field, reason=str(error)))
            return None

    def missing():
        raise RuntimeError("System memory/physical CPU metadata provider unavailable.")

    report.update(pytorch_version=str(torch.__version__),
                  cuda_version=getattr(torch.version, "cuda", None),
                  rocm_version=getattr(torch.version, "hip", None))
    approved = query("installer_filtering", hardware._validated_device_specs)
    report["installer_filtering"] = {
        "applies": approved is not None if not report["errors"] else None,
        "approved_specs": sorted(approved) if approved is not None else None,
        "reason": "Installer approval is applied by runtime discovery when managed state is present.",
    }
    report["cpu"] = dict(logical_count=query("cpu.logical_count", os.cpu_count),
                         physical_count=query("cpu.physical_count", lambda: psutil.cpu_count(logical=False) if psutil else missing()))
    memory = query("system_memory", lambda: psutil.virtual_memory() if psutil else missing())
    report["system_memory"] = dict(total_bytes=int(memory.total) if memory else None,
                                   available_bytes=int(memory.available) if memory else None)
    candidates = query("devices", lambda: hardware.get_available_devices(diagnostics=report["errors"]))
    for candidate in candidates or []:
        spec = candidate.spec
        backend = "rocm" if candidate.backend == "cuda" and report["rocm_version"] else candidate.backend
        record = dict(device_selection=spec, name=candidate.label, backend=backend,
                      memory={}, capabilities={})
        caps = record["capabilities"]
        caps["float32"] = capability("supported", "The runtime exposes this device; actual operations are unverified.")
        caps["tf32"] = capability("unsupported", "TF32 requires NVIDIA CUDA.")
        caps["bf16"] = capability("unknown", "No computation-free BF16 metadata check is used for this backend.")
        caps["tiled"] = capability("unsupported", "Tiled execution requires an accelerator.")
        mem = record["memory"]
        mem.update(kind="system" if backend == "cpu" else "unified_working_set" if backend == "mps" else "device",
                   total_bytes=None, available_bytes=None)
        if backend == "cpu":
            mem.update(report["system_memory"])
        else:
            module = getattr(torch, candidate.backend, None)
            if backend == "mps":
                mem["recommended_working_set_bytes"] = query(f"{spec}.working_set", lambda: int(module.recommended_max_memory()))
                mem["driver_allocated_bytes"] = query(f"{spec}.driver_allocated", lambda: int(module.driver_allocated_memory()))
                mem["reason"] = "Unified working-set measurements are not dedicated VRAM or free device memory."
                required = ("recommended_max_memory", "driver_allocated_memory")
            else:
                props = query(f"{spec}.properties", lambda: module.get_device_properties(candidate.index))
                if props is not None:
                    mem["total_bytes"] = query(f"{spec}.total_memory", lambda: int(props.total_memory))
                free = query(f"{spec}.available_memory", lambda: module.mem_get_info(candidate.index))
                if free is not None:
                    mem.update(available_bytes=int(free[0]), total_bytes=int(free[1]))
                required = ("Stream", "Event", "stream", "mem_get_info")
                if backend == "cuda":
                    major = query(f"{spec}.compute_capability", lambda: int(props.major))
                    for precision in ("tf32", "bf16"):
                        caps[precision] = capability(
                            "unknown" if major is None else "supported" if major >= 8 else "unsupported",
                            "NVIDIA compute capability >= 8 indicates hardware eligibility; no kernel was tested.",
                            "CUDA device properties")
                    if major is not None and major < 8:
                        caps["bf16"] = capability("unknown",
                            "Native BF16 is not indicated; possible runtime emulation has not been probed.",
                            "CUDA device properties")
            absent = [name for name in required if not callable(getattr(module, name, None))]
            caps["tiled"] = capability("unsupported" if absent else "supported",
                "Missing runtime APIs: " + ", ".join(absent) if absent else
                "Required runtime APIs exist; memory sufficiency and actual tiled operations remain unverified.",
                "runtime API inspection")
        record["errors"] = [e for e in report["errors"] if e["field"].startswith(spec + ".")]
        report["devices"].append(record)
    report["status"] = "unavailable" if candidates is None else "partial" if report["errors"] else "ok"
    return report


def discover_compute_capabilities(project_root, tool_id=None):
    # Validate identifiers before spawning; settings discovery does not import torch.
    from mcp_server.pipeline.Pipeline_Settings import get_pipeline_schema
    schema = get_pipeline_schema(tool_id, project_root) if tool_id is not None else None
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--collect"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        if completed.returncode:
            raise ValueError(f"Metadata helper exited with code {completed.returncode}: {completed.stderr[-2000:]}")
        report = json.loads(completed.stdout)
        if not isinstance(report, dict) or not empty_report().keys() <= report.keys() or report.get("status") not in {"ok", "partial", "unavailable"} or not isinstance(report.get("devices"), list):
            raise ValueError("Metadata helper returned an invalid report.")
        for device in report["devices"]:
            if not isinstance(device, dict) or not all(key in device for key in ("device_selection", "backend", "capabilities")):
                raise ValueError("Metadata helper returned an invalid device record.")
            if not isinstance(device["capabilities"], dict) or not all(key in device["capabilities"] for key in ("float32", "tf32", "bf16", "tiled")):
                raise ValueError("Metadata helper returned incomplete capabilities.")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        # subprocess.run kills and waits for its own helper on timeout.
        report = empty_report()
        report["errors"] = [dict(field="collector", reason=str(error))]
    report["notes"] = ["Metadata only; no tensor computation or benchmarking was performed.",
                       "Memory is a timestamped observation, not a reservation.",
                       "Only devices usable by this Python runtime are enumerated; auto selects at job runtime."]
    if schema is not None:
        props = schema["parameters_schema"]["properties"]
        report["tool"] = dict(tool_id=tool_id, settings={key: props[key] for key in
            ("DEVICE_SELECTION", "ACCELERATOR_PRECISION", "EXECUTION_MODE") if key in props})
        for device in report["devices"]:
            applicable = {}
            if "DEVICE_SELECTION" in props:
                applicable["DEVICE_SELECTION"] = capability("supported", "Selectable runtime device; workload suitability is unverified.")
            if "ACCELERATOR_PRECISION" in props:
                applicable["ACCELERATOR_PRECISION"] = {key: device["capabilities"][key] for key in ("float32", "tf32", "bf16")}
                if device["backend"] == "cpu":
                    applicable["ACCELERATOR_PRECISION"]["bf16"] = capability(
                        "unsupported", "These alignment tools require an accelerator for BF16 execution.", "tool restriction")
                applicable["ACCELERATOR_PRECISION"]["automatic_32bit"] = capability("supported", "Runtime selects eligible 32-bit precision; this is not a speed recommendation.")
            if "EXECUTION_MODE" in props:
                tiled = device["capabilities"]["tiled"]
                if tool_id == "network_injection" and device["backend"] == "mps":
                    tiled = capability("unsupported", "Network injection does not support MPS tiled execution.", "tool restriction")
                applicable["EXECUTION_MODE"] = dict(tiled=tiled,
                    scalar=capability("supported", "Scalar mode is selectable; no workload was tested."),
                    auto=capability("supported", "Runtime selects execution mode."))
            device["tool_settings"] = applicable
        report["tool"]["notes"] = [
            "Model-specific execution and source-network precision requirements are checked by the pipeline at runtime."
        ]
    return report


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            import torch
            from utilities import Hardware_Acceleration as Hardware_Utils
            try:
                import psutil
            except ImportError:
                psutil = None
            report = collect_metadata(torch, Hardware_Utils, psutil)
    except Exception as error:
        report = empty_report()
        report["errors"] = [dict(field="collector", reason=str(error))]
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
