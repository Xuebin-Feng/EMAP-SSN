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

"""Install project dependencies and a hardware-appropriate PyTorch build."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import Detect_GPU


TORCH_VERSION = "2.12.1"
WINDOWS_ROCM_TORCH_VERSION = "2.12.0+rocm7.14.0"
ESM_VERSION = "3.3.0"
ESM_WHEEL_SHA256 = "d5e412470877fa2e21c36b40a52cdf1bef5664234654355dc2a35bb8cd2f4d82"
STATE_FILENAME = "ssn_backend.json"
STATE_SCHEMA = 2

PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda126": "https://download.pytorch.org/whl/cu126",
    "cuda132": "https://download.pytorch.org/whl/cu132",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "rocm72": "https://download.pytorch.org/whl/rocm7.2",
    "rocm714": "https://repo.amd.com/rocm/whl-multi-arch/",
}


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    torch_requirement: str
    index_url: str | None
    gfx_target: str | None
    description: str


def backend_spec(report: dict[str, Any]) -> BackendSpec:
    backend = str(report.get("backend", "cpu"))
    gfx_target = report.get("gfx_target")
    if backend == "rocm714":
        if not isinstance(gfx_target, str) or not gfx_target.startswith("gfx"):
            raise ValueError("Windows ROCm selection requires a validated gfx target.")
        requirement = f"torch[device-{gfx_target}]=={WINDOWS_ROCM_TORCH_VERSION}"
        description = f"Windows ROCm 7.14 ({gfx_target})"
    else:
        requirement = f"torch=={TORCH_VERSION}"
        description = {
            "cuda132": "NVIDIA CUDA 13.2",
            "cuda126": "NVIDIA CUDA 12.6",
            "xpu": "Intel XPU",
            "rocm72": "Linux ROCm 7.2",
            "mps": "Apple MPS",
            "cpu": "CPU",
        }.get(backend)
        if description is None:
            raise ValueError(f"Unknown PyTorch backend: {backend}")
    return BackendSpec(
        backend=backend,
        torch_requirement=requirement,
        index_url=PYTORCH_INDEXES.get(backend),
        gfx_target=gfx_target if backend == "rocm714" else None,
        description=description,
    )


def venv_python(venv: Path) -> Path:
    candidates = (venv / "Scripts" / "python.exe", venv / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No Python executable found in virtual environment: {venv}")


def _uv_prefix(uv_executable: str, python: Path) -> list[str]:
    return [uv_executable, "pip", "install", "--python", str(python)]


def base_install_command(uv_executable: str, python: Path, requirements: Path) -> list[str]:
    return _uv_prefix(uv_executable, python) + ["-r", str(requirements)]


def torch_install_command(uv_executable: str, python: Path, spec: BackendSpec) -> list[str]:
    command = _uv_prefix(uv_executable, python) + [spec.torch_requirement]
    if spec.index_url:
        command += ["--index-url", spec.index_url]
    return command


def esm_install_command(uv_executable: str, python: Path, wheel: Path) -> list[str]:
    return _uv_prefix(uv_executable, python) + [str(wheel)]


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_esm_wheel(wheel: Path) -> None:
    if not wheel.is_file():
        raise FileNotFoundError(f"Bundled ESM wheel is missing: {wheel}")
    actual = _sha256(wheel)
    if actual != ESM_WHEEL_SHA256:
        raise ValueError(
            f"Bundled ESM wheel checksum mismatch: expected {ESM_WHEEL_SHA256}, got {actual}."
        )


def _state_profile(
    active: BackendSpec,
    requirements: Path,
    requested: BackendSpec | None = None,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "requested_backend": asdict(requested or active),
        "active_backend": asdict(active),
        "requirements_sha256": _sha256(requirements),
        "esm_version": ESM_VERSION,
        "esm_wheel_sha256": ESM_WHEEL_SHA256,
    }


def _backend_from_state(value: Any) -> BackendSpec | None:
    if not isinstance(value, dict):
        return None
    try:
        return BackendSpec(**value)
    except (TypeError, ValueError):
        return None


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_state(path: Path, profile: dict[str, Any], report: dict[str, Any]) -> None:
    payload = dict(profile)
    payload["detection"] = report
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validation_program(spec: BackendSpec) -> str:
    expected_version = "2.12.0" if spec.backend == "rocm714" else TORCH_VERSION
    checks = [
        "import torch",
        f"assert torch.__version__.split('+', 1)[0] == {expected_version!r}, torch.__version__",
    ]
    if spec.backend in {"cuda126", "cuda132"}:
        cuda_prefix = "12.6" if spec.backend == "cuda126" else "13.2"
        checks += [
            "assert torch.version.hip is None, torch.version.hip",
            f"assert str(torch.version.cuda).startswith({cuda_prefix!r}), torch.version.cuda",
            "assert torch.cuda.is_available()",
            "assert (torch.ones(1, device='cuda') + 1).item() == 2",
        ]
    elif spec.backend in {"rocm72", "rocm714"}:
        checks += [
            "assert torch.version.hip, torch.version.hip",
            "assert torch.cuda.is_available()",
            "assert (torch.ones(1, device='cuda') + 1).item() == 2",
        ]
    elif spec.backend == "xpu":
        checks += [
            "assert hasattr(torch, 'xpu') and torch.xpu.is_available()",
            "assert (torch.ones(1, device='xpu') + 1).item() == 2",
        ]
    elif spec.backend == "mps":
        checks += [
            "assert torch.backends.mps.is_available()",
            "assert (torch.ones(1, device='mps') + 1).item() == 2",
        ]
    else:
        checks += [
            "assert torch.version.cuda is None, torch.version.cuda",
            "assert torch.version.hip is None, torch.version.hip",
            "assert (torch.ones(1, device='cpu') + 1).item() == 2",
        ]
    return "; ".join(checks)


def validate_backend(python: Path, spec: BackendSpec) -> bool:
    completed = _run([str(python), "-c", _validation_program(spec)], capture=True)
    if completed.returncode == 0:
        return True
    detail = completed.stderr.strip() or completed.stdout.strip() or "unknown validation error"
    print(f"PyTorch backend validation failed: {detail}", file=sys.stderr)
    return False


def _installed_distribution_names(python: Path) -> list[str]:
    program = (
        "import importlib.metadata as m, json; "
        "print(json.dumps(sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name']})))"
    )
    completed = subprocess.run(
        [str(python), "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    try:
        values = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return [str(value) for value in values]


def _is_backend_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return (
        normalized == "torch"
        or normalized in {"mkl", "tbb", "tcmlib", "triton", "umf"}
        or normalized.startswith(
            (
                "amd-",
                "dpcpp-",
                "intel-",
                "onemkl-",
                "rocm",
                "nvidia-",
                "pytorch-triton",
                "triton-",
            )
        )
    )


def remove_backend_packages(uv_executable: str, python: Path) -> bool:
    packages = [name for name in _installed_distribution_names(python) if _is_backend_package(name)]
    if not packages:
        return True
    command = [uv_executable, "pip", "uninstall", "--python", str(python), *packages]
    return _run(command).returncode == 0


def _installed_version(python: Path, package: str) -> str | None:
    program = (
        "import importlib.metadata as m; "
        f"print(m.version({package!r}))"
    )
    completed = subprocess.run(
        [str(python), "-c", program], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def install_backend(uv_executable: str, python: Path, spec: BackendSpec) -> bool:
    if not remove_backend_packages(uv_executable, python):
        return False
    completed = _run(torch_install_command(uv_executable, python, spec))
    return completed.returncode == 0 and validate_backend(python, spec)


def install(
    *,
    project_root: Path,
    venv: Path,
    uv_executable: str,
    dry_run: bool = False,
) -> int:
    python = venv_python(venv)
    requirements = project_root / "src" / "requirements.txt"
    wheel = project_root / "src" / "resources" / "wheels" / f"esm-{ESM_VERSION}-py3-none-any.whl"
    verify_esm_wheel(wheel)

    report = Detect_GPU.detect_hardware()
    requested = backend_spec(report)
    request_profile = _state_profile(requested, requirements, requested)
    state_path = venv / STATE_FILENAME
    current_state = read_state(state_path)
    request_keys = (
        "schema",
        "requested_backend",
        "requirements_sha256",
        "esm_version",
        "esm_wheel_sha256",
    )
    state_matches = current_state is not None and all(
        current_state.get(key) == request_profile.get(key) for key in request_keys
    )
    saved_active = (
        _backend_from_state(current_state.get("active_backend"))
        if state_matches and current_state is not None
        else None
    )

    print(f"Detected: {report['reason']}")
    commands = [
        base_install_command(uv_executable, python, requirements),
        torch_install_command(uv_executable, python, requested),
        esm_install_command(uv_executable, python, wheel),
    ]
    if dry_run:
        for command in commands:
            print(f"Dry run: {shlex.join(command)}")
        return 0

    if _run(commands[0]).returncode != 0:
        print("Base dependency installation failed.", file=sys.stderr)
        return 1

    active = saved_active or requested
    if not (saved_active is not None and validate_backend(python, saved_active)):
        active = requested
        if not install_backend(uv_executable, python, requested):
            if requested.backend == "cpu":
                print("CPU PyTorch installation failed.", file=sys.stderr)
                return 1
            print(
                f"{requested.description} could not be installed or validated; falling back to CPU.",
                file=sys.stderr,
            )
            active = backend_spec({"backend": "cpu", "gfx_target": None})
            if not install_backend(uv_executable, python, active):
                print("CPU fallback installation failed.", file=sys.stderr)
                return 1
            report = dict(report)
            report["fallback_from"] = requested.backend
            report["backend"] = "cpu"
            report["gfx_target"] = None
            report["reason"] = f"{requested.description} failed validation; CPU fallback is active."

    if active.backend != requested.backend and "fallback_from" not in report:
        report = dict(report)
        report["fallback_from"] = requested.backend
        report["backend"] = active.backend
        report["gfx_target"] = None
        report["reason"] = (
            f"The previously validated {active.description} fallback remains active after "
            f"{requested.description} was unavailable."
        )

    esm_ready = state_matches and _installed_version(python, "esm") == ESM_VERSION
    if not esm_ready and _run(commands[2]).returncode != 0:
        print("Bundled ESM wheel installation failed.", file=sys.stderr)
        return 1

    profile = _state_profile(active, requirements, requested)
    write_state(state_path, profile, report)
    print(f"Dependency environment is ready ({active.description}).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--uv-executable", default="uv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        return install(
            project_root=project_root,
            venv=args.venv.resolve(),
            uv_executable=args.uv_executable,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Dependency installation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
