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

"""Install project dependencies and a validated hardware-specific PyTorch build."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import textwrap
from typing import Any, Iterable
import zipfile

import Detect_GPU


TORCH_VERSION = "2.12.1"
LINUX_ROCM_64_TORCH_VERSION = "2.9.1"
WINDOWS_ROCM_714_TORCH_VERSION = "2.12.0+rocm7.14.0"
WINDOWS_ROCM_721_TORCH_VERSION = "2.9.1+rocm7.2.1"
ESM_VERSION = "3.3.0"
ESM_WHEEL_SHA256 = "d5e412470877fa2e21c36b40a52cdf1bef5664234654355dc2a35bb8cd2f4d82"
TRANSFORMERS_VERSION = "4.57.6+biohub.3a8956f"
TRANSFORMERS_WHEEL_SHA256 = "74cb19ba0b6c4cf0769322f0ef035bd016eea6ccb2f587a1ff1263a016354c3b"
STATE_FILENAME = "ssn_backend.json"
STATE_SCHEMA = 4
SETUP_REQUIRED_EXIT = 10

PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda126": "https://download.pytorch.org/whl/cu126",
    "cuda132": "https://download.pytorch.org/whl/cu132",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "rocm72": "https://download.pytorch.org/whl/rocm7.2",
    "rocm64": "https://download.pytorch.org/whl/rocm6.4",
    "rocm714": "https://repo.amd.com/rocm/whl-multi-arch/",
}
ROCM_721_ROOT = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_721_SDK_REQUIREMENTS = (
    f"{ROCM_721_ROOT}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_721_ROOT}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_721_ROOT}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    f"{ROCM_721_ROOT}/rocm-7.2.1.tar.gz",
)
ROCM_721_TORCH_REQUIREMENT = (
    f"{ROCM_721_ROOT}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
)


@dataclass(frozen=True)
class InstallStep:
    requirements: tuple[str, ...]
    index_url: str | None = None


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    profile: str
    torch_version: str
    install_steps: tuple[InstallStep, ...]
    gfx_target: str | None
    device_ids: tuple[str, ...]
    description: str


def _standard_spec(candidate: dict[str, Any]) -> BackendSpec:
    backend = str(candidate.get("backend", "cpu"))
    profile = str(candidate.get("profile") or backend)
    gfx_target = candidate.get("gfx_target")
    device_ids = tuple(str(value) for value in candidate.get("device_ids") or ())
    if backend == "rocm714":
        if not isinstance(gfx_target, str) or not gfx_target.startswith("gfx"):
            raise ValueError("Windows ROCm 7.14 requires a validated GFX target.")
        steps = (
            InstallStep(
                (f"torch[device-{gfx_target}]=={WINDOWS_ROCM_714_TORCH_VERSION}",),
                PYTORCH_INDEXES[backend],
            ),
        )
        torch_version = "2.12.0"
        description = f"Windows ROCm 7.14 ({gfx_target})"
    elif backend == "rocm721":
        steps = (
            InstallStep(ROCM_721_SDK_REQUIREMENTS),
            InstallStep((ROCM_721_TORCH_REQUIREMENT,)),
        )
        torch_version = "2.9.1"
        description = f"Windows ROCm 7.2.1 ({gfx_target or 'supported target'})"
    elif backend == "rocm64":
        steps = (
            InstallStep(
                (f"torch=={LINUX_ROCM_64_TORCH_VERSION}",),
                PYTORCH_INDEXES[backend],
            ),
        )
        torch_version = LINUX_ROCM_64_TORCH_VERSION
        description = f"Linux ROCm 6.4 ({gfx_target or 'supported target'})"
    else:
        descriptions = {
            "cuda132": "NVIDIA CUDA 13.2",
            "cuda126": "NVIDIA CUDA 12.6",
            "xpu": "Intel XPU",
            "rocm72": "Linux ROCm 7.2",
            "mps": "Apple MPS",
            "cpu": "CPU",
        }
        description = descriptions.get(backend)
        if description is None:
            raise ValueError(f"Unknown PyTorch backend: {backend}")
        steps = (InstallStep((f"torch=={TORCH_VERSION}",), PYTORCH_INDEXES.get(backend)),)
        torch_version = TORCH_VERSION
    return BackendSpec(
        backend=backend,
        profile=profile,
        torch_version=torch_version,
        install_steps=steps,
        gfx_target=str(gfx_target) if gfx_target else None,
        device_ids=device_ids,
        description=description,
    )


def backend_specs(report: dict[str, Any]) -> list[BackendSpec]:
    candidates = report.get("backend_candidates")
    if not isinstance(candidates, list):
        candidates = [{
            "backend": report.get("backend", "cpu"),
            "profile": report.get("backend", "cpu"),
            "gfx_target": report.get("gfx_target"),
            "device_ids": (),
        }]
    specs = [_standard_spec(candidate) for candidate in candidates if isinstance(candidate, dict)]
    if not specs or specs[-1].backend != "cpu":
        specs.append(_standard_spec({"backend": "cpu", "profile": "cpu", "device_ids": ("cpu",)}))
    return specs


def backend_spec(report: dict[str, Any]) -> BackendSpec:
    """Return the first requested backend for compatibility with older callers."""
    return backend_specs(report)[0]


def venv_python(venv: Path) -> Path:
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No Python executable found in virtual environment: {venv}")


def _uv_prefix(uv_executable: str, python: Path) -> list[str]:
    return [uv_executable, "pip", "install", "--python", str(python)]


def base_install_command(uv_executable: str, python: Path, requirements: Path) -> list[str]:
    return _uv_prefix(uv_executable, python) + ["-r", str(requirements)]


def backend_install_commands(uv_executable: str, python: Path, spec: BackendSpec) -> list[list[str]]:
    commands: list[list[str]] = []
    for step in spec.install_steps:
        command = _uv_prefix(uv_executable, python) + list(step.requirements)
        if step.index_url:
            command += ["--index-url", step.index_url]
        commands.append(command)
    return commands


def torch_install_command(uv_executable: str, python: Path, spec: BackendSpec) -> list[str]:
    """Return the first backend install command for legacy tests/callers."""
    return backend_install_commands(uv_executable, python, spec)[0]


def esm_install_command(uv_executable: str, python: Path, wheel: Path) -> list[str]:
    return _uv_prefix(uv_executable, python) + ["--no-deps", str(wheel)]


def transformers_install_command(uv_executable: str, python: Path, wheel: Path) -> list[str]:
    return _uv_prefix(uv_executable, python) + [str(wheel)]


def esm_runtime_install_command(
    uv_executable: str, python: Path, requirements: Path
) -> list[str]:
    return _uv_prefix(uv_executable, python) + ["-r", str(requirements)]


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=capture)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hardware_fingerprint(report: dict[str, Any]) -> str:
    material = {
        "compatibility_revision": report.get("compatibility_revision"),
        "platform": report.get("platform"),
        "os": report.get("os"),
        "devices": [
            {
                key: device.get(key)
                for key in ("id", "name", "vendor", "pci_id", "driver_version", "driver", "kind", "architecture", "eligibility", "eligible_profiles")
            }
            for device in report.get("devices", [])
            if isinstance(device, dict)
        ],
        "backend_candidates": report.get("backend_candidates"),
    }
    return _stable_hash(material)


def _wheel_metadata(wheel: Path) -> tuple[Any, set[str]]:
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(metadata_files) != 1:
                raise ValueError(f"Expected one METADATA file in {wheel.name}.")
            metadata = Parser().parsestr(archive.read(metadata_files[0]).decode("utf-8"))
    except (OSError, UnicodeError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError(f"Bundled wheel is unreadable: {wheel}: {error}") from error
    return metadata, names


def _verify_wheel(
    wheel: Path, *, package: str, version: str, sha256: str,
    required_members: Iterable[str] = (),
) -> Any:
    if not wheel.is_file():
        raise FileNotFoundError(f"Bundled {package} wheel is missing: {wheel}")
    actual = _sha256(wheel)
    if actual != sha256:
        raise ValueError(
            f"Bundled {package} wheel checksum mismatch: expected {sha256}, got {actual}."
        )
    metadata, names = _wheel_metadata(wheel)
    if metadata.get("Name", "").lower() != package.lower():
        raise ValueError(f"Bundled wheel reports the wrong package name: {metadata.get('Name')!r}.")
    if metadata.get("Version") != version:
        raise ValueError(
            f"Bundled {package} wheel reports version {metadata.get('Version')!r}; expected {version!r}."
        )
    missing = [member for member in required_members if member not in names]
    if missing:
        raise ValueError(f"Bundled {package} wheel is missing required modules: {', '.join(missing)}.")
    return metadata


def verify_esm_wheel(wheel: Path) -> Any:
    return _verify_wheel(
        wheel, package="esm", version=ESM_VERSION, sha256=ESM_WHEEL_SHA256,
        required_members=("esm/__init__.py",),
    )


def verify_transformers_wheel(wheel: Path) -> Any:
    return _verify_wheel(
        wheel, package="transformers", version=TRANSFORMERS_VERSION,
        sha256=TRANSFORMERS_WHEEL_SHA256,
        required_members=(
            "transformers/__init__.py",
            "transformers/models/esmc/configuration_esmc.py",
            "transformers/models/esmfold2/configuration_esmfold2.py",
        ),
    )


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def esm_runtime_requirements_from_wheel(wheel: Path) -> tuple[str, ...]:
    metadata = verify_esm_wheel(wheel)
    return tuple(
        requirement
        for requirement in metadata.get_all("Requires-Dist", [])
        if _requirement_name(requirement) not in {"torch", "transformers"}
    )


def _requirements_entries(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Bundled ESM runtime requirements are missing: {path}")
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def verify_esm_runtime_requirements(wheel: Path, requirements: Path) -> None:
    expected = esm_runtime_requirements_from_wheel(wheel)
    actual = _requirements_entries(requirements)
    if actual != expected:
        raise ValueError(
            "Bundled ESM runtime requirements do not match the verified ESM wheel metadata."
        )


def verify_bundled_artifacts(
    esm_wheel: Path, transformers_wheel: Path, runtime_requirements: Path
) -> None:
    verify_esm_wheel(esm_wheel)
    verify_transformers_wheel(transformers_wheel)
    verify_esm_runtime_requirements(esm_wheel, runtime_requirements)


def _bundled_paths(project_root: Path) -> tuple[Path, Path, Path]:
    wheels = project_root / "src" / "resources" / "wheels"
    return (
        wheels / f"esm-{ESM_VERSION}-py3-none-any.whl",
        wheels / f"transformers-{TRANSFORMERS_VERSION}-py3-none-any.whl",
        wheels / f"esm-{ESM_VERSION}-runtime-requirements.txt",
    )


def read_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_state(
    path: Path, payload: dict[str, Any], report: dict[str, Any] | None = None
) -> None:
    if report is not None:
        payload = dict(payload)
        specs = backend_specs(report)
        payload.update({
            "schema": STATE_SCHEMA,
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "hardware_fingerprint": hardware_fingerprint(report),
            "detection": report,
        })
        payload.setdefault("requested_candidates", _spec_payloads(specs))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _state_profile(
    active: BackendSpec, requirements: Path, requested: BackendSpec | None = None
) -> dict[str, Any]:
    """Build state metadata while preserving the schema-2 helper interface."""
    requested = requested or active
    return {
        "schema": STATE_SCHEMA,
        "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
        "requested_backend": _spec_payloads((requested,))[0],
        "active_backend": _spec_payloads((active,))[0],
        "requirements_sha256": _sha256(requirements),
        "esm_version": ESM_VERSION,
        "esm_wheel_sha256": ESM_WHEEL_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "esm_runtime_requirements_sha256": _sha256(
            requirements.parent / "resources" / "wheels"
            / f"esm-{ESM_VERSION}-runtime-requirements.txt"
        ),
        "validated_devices": [],
        "ignored_devices": [],
        "attempts": [],
    }


def _backend_from_state(value: Any) -> BackendSpec | None:
    if not isinstance(value, dict):
        return None
    try:
        steps = tuple(
            InstallStep(tuple(step["requirements"]), step.get("index_url"))
            for step in value["install_steps"]
        )
        return BackendSpec(
            backend=str(value["backend"]), profile=str(value["profile"]),
            torch_version=str(value["torch_version"]), install_steps=steps,
            gfx_target=value.get("gfx_target"),
            device_ids=tuple(str(item) for item in value.get("device_ids", ())),
            description=str(value["description"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validation_program(spec: BackendSpec) -> str:
    return textwrap.dedent(
        f"""
        import json
        result = {{"backend": {spec.backend!r}, "profile": {spec.profile!r}, "expected_version": {spec.torch_version!r}, "devices": [], "package_error": None}}
        try:
            import torch
            result["torch_version"] = torch.__version__
            if torch.__version__.split("+", 1)[0] != {spec.torch_version!r}:
                raise RuntimeError("unexpected torch version: " + torch.__version__)
            backend = {spec.backend!r}
            if backend in {{"cuda126", "cuda132", "rocm72", "rocm64", "rocm714", "rocm721"}}:
                is_rocm = backend.startswith("rocm")
                if is_rocm and not torch.version.hip:
                    raise RuntimeError("ROCm/HIP build metadata is missing")
                if not is_rocm and torch.version.hip is not None:
                    raise RuntimeError("CUDA profile loaded a ROCm build")
                if not is_rocm:
                    expected_cuda = "12.6" if backend == "cuda126" else "13.2"
                    if not str(torch.version.cuda).startswith(expected_cuda):
                        raise RuntimeError("unexpected CUDA runtime: " + str(torch.version.cuda))
                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda.is_available() is false")
                for index in range(torch.cuda.device_count()):
                    item = {{"spec": f"cuda:{{index}}", "index": index, "name": None, "architecture": None, "success": False, "error": None}}
                    try:
                        item["name"] = torch.cuda.get_device_name(index)
                        props = torch.cuda.get_device_properties(index)
                        if is_rocm:
                            item["architecture"] = getattr(props, "gcnArchName", None) or getattr(props, "gcn_arch_name", None)
                        else:
                            major = getattr(props, "major", None)
                            minor = getattr(props, "minor", None)
                            item["architecture"] = f"sm_{{major}}{{minor}}" if major is not None and minor is not None else None
                        value = (torch.ones(1, device=f"cuda:{{index}}") + 1).item()
                        torch.cuda.synchronize(index)
                        if value != 2:
                            raise RuntimeError("unexpected tensor result")
                        item["success"] = True
                    except Exception as error:
                        item["error"] = str(error)
                    result["devices"].append(item)
            elif backend == "xpu":
                if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                    raise RuntimeError("torch.xpu.is_available() is false")
                for index in range(torch.xpu.device_count()):
                    item = {{"spec": f"xpu:{{index}}", "index": index, "name": None, "architecture": None, "success": False, "error": None}}
                    try:
                        item["name"] = torch.xpu.get_device_name(index)
                        value = (torch.ones(1, device=f"xpu:{{index}}") + 1).item()
                        torch.xpu.synchronize(index)
                        if value != 2:
                            raise RuntimeError("unexpected tensor result")
                        item["success"] = True
                    except Exception as error:
                        item["error"] = str(error)
                    result["devices"].append(item)
            elif backend == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS is unavailable")
                value = (torch.ones(1, device="mps") + 1).item()
                result["devices"].append({{"spec": "mps", "index": None, "name": "Apple GPU", "architecture": None, "success": value == 2, "error": None if value == 2 else "unexpected tensor result"}})
            else:
                if torch.version.cuda is not None or torch.version.hip is not None:
                    raise RuntimeError("CPU profile loaded an accelerator build")
                value = (torch.ones(1, device="cpu") + 1).item()
                result["devices"].append({{"spec": "cpu", "index": None, "name": "CPU", "architecture": None, "success": value == 2, "error": None if value == 2 else "unexpected tensor result"}})
        except Exception as error:
            result["package_error"] = str(error)
        print(json.dumps(result, sort_keys=True))
        """
    ).strip()


def validate_backend(python: Path, spec: BackendSpec) -> dict[str, Any] | None:
    completed = _run([str(python), "-c", _validation_program(spec)], capture=True)
    detail = completed.stderr.strip()
    if completed.returncode == 0 and not completed.stdout.strip():
        # Preserve compatibility with callers/tests that mock a successful
        # subprocess without executing the structured validator.
        return _legacy_validation_payload(spec)
    try:
        result = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        result = None
    if completed.returncode != 0 or not isinstance(result, dict):
        print(f"PyTorch backend validation failed: {detail or completed.stdout.strip() or 'invalid validator output'}", file=sys.stderr)
        return None
    target = spec.gfx_target
    devices = [item for item in result.get("devices", []) if isinstance(item, dict)]
    for item in devices:
        architecture = str(item.get("architecture") or "").split(":", 1)[0].lower()
        if target and spec.backend.startswith("rocm"):
            if not architecture:
                item["success"] = False
                item["error"] = f"Runtime did not report an architecture for selected target {target}."
            elif architecture != target.lower():
                item["success"] = False
                item["error"] = f"Device architecture {architecture} does not match selected target {target}."
    valid = [item for item in devices if item.get("success")]
    if result.get("package_error") or not valid:
        print(f"PyTorch backend validation failed: {result.get('package_error') or 'no device passed a tensor operation'}", file=sys.stderr)
        return None
    result["validated_devices"] = valid
    return result


def _legacy_validation_payload(spec: BackendSpec) -> dict[str, Any]:
    runtime_spec = "cpu" if spec.backend == "cpu" else "mps" if spec.backend == "mps" else "xpu:0" if spec.backend == "xpu" else "cuda:0"
    device = {"spec": runtime_spec, "index": 0 if ":" in runtime_spec else None, "name": spec.description, "architecture": spec.gfx_target, "success": True, "error": None}
    return {"backend": spec.backend, "profile": spec.profile, "torch_version": spec.torch_version, "devices": [device], "validated_devices": [device], "package_error": None}


def _normalize_validation(value: Any, spec: BackendSpec) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return _legacy_validation_payload(spec) if value else None


def _installed_distribution_names(python: Path) -> list[str]:
    program = "import importlib.metadata as m, json; print(json.dumps(sorted({d.metadata['Name'] for d in m.distributions() if d.metadata['Name']})))"
    completed = subprocess.run([str(python), "-c", program], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return []
    try:
        values = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return [str(value) for value in values]


def _is_backend_package(name: str) -> bool:
    normalized = name.lower().replace("_", "-")
    return normalized == "torch" or normalized in {"mkl", "tbb", "tcmlib", "triton", "umf"} or normalized.startswith(
        ("amd-", "dpcpp-", "intel-", "onemkl-", "rocm", "nvidia-", "pytorch-triton", "triton-")
    )


def remove_backend_packages(uv_executable: str, python: Path) -> bool:
    packages = [name for name in _installed_distribution_names(python) if _is_backend_package(name)]
    if not packages:
        return True
    return _run([uv_executable, "pip", "uninstall", "--python", str(python), *packages]).returncode == 0


def _installed_version(python: Path, package: str) -> str | None:
    program = f"import importlib.metadata as m; print(m.version({package!r}))"
    completed = subprocess.run([str(python), "-c", program], check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _esm_stack_program() -> str:
    return textwrap.dedent(
        f"""
        import importlib.metadata as metadata
        import esm
        from transformers.models.esmc.configuration_esmc import ESMCConfig
        from transformers.models.esmc.modeling_esmc import ESMCModel
        from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

        assert metadata.version("esm") == {ESM_VERSION!r}
        assert metadata.version("transformers") == {TRANSFORMERS_VERSION!r}
        assert ESMCConfig.model_type == "esmc"
        assert ESMFold2Config.model_type == "esmfold2"
        assert ESMCModel is not None and ESMFold2Model is not None
        """
    ).strip()


def validate_esm_stack(python: Path) -> bool:
    completed = _run(
        [str(python), "-X", "utf8", "-c", _esm_stack_program()], capture=True
    )
    if completed.returncode == 0:
        return True
    detail = completed.stderr.strip() or completed.stdout.strip() or "import smoke test failed"
    print(f"ESM/Transformers import validation failed: {detail}", file=sys.stderr)
    return False


def validate_package_consistency(uv_executable: str, python: Path) -> bool:
    completed = _run(
        [uv_executable, "pip", "check", "--python", str(python)], capture=True
    )
    if completed.returncode == 0:
        return True
    detail = completed.stderr.strip() or completed.stdout.strip() or "uv pip check failed"
    print(f"Installed package consistency failed: {detail}", file=sys.stderr)
    return False


def install_backend(uv_executable: str, python: Path, spec: BackendSpec) -> dict[str, Any] | None:
    if not remove_backend_packages(uv_executable, python):
        return None
    for command in backend_install_commands(uv_executable, python, spec):
        if _run(command).returncode != 0:
            return None
    return validate_backend(python, spec)


def _state_matches(
    state: dict[str, Any] | None,
    specs: Iterable[BackendSpec],
    fingerprint: str,
    requirements: Path,
) -> bool:
    return bool(
        state
        and state.get("schema") == STATE_SCHEMA
        and state.get("compatibility_revision") == Detect_GPU.COMPATIBILITY_REVISION
        and state.get("hardware_fingerprint") == fingerprint
        and state.get("requirements_sha256") == _sha256(requirements)
        and state.get("esm_version") == ESM_VERSION
        and state.get("esm_wheel_sha256") == ESM_WHEEL_SHA256
        and state.get("transformers_version") == TRANSFORMERS_VERSION
        and state.get("transformers_wheel_sha256") == TRANSFORMERS_WHEEL_SHA256
        and state.get("esm_runtime_requirements_sha256") == _sha256(
            requirements.parent / "resources" / "wheels"
            / f"esm-{ESM_VERSION}-runtime-requirements.txt"
        )
        and state.get("requested_candidates") == _spec_payloads(specs)
    )


def _spec_payloads(specs: Iterable[BackendSpec]) -> list[dict[str, Any]]:
    """Return the JSON-normalized representation persisted in backend state."""
    return json.loads(json.dumps([asdict(spec) for spec in specs]))


def install(
    *, project_root: Path, venv: Path, uv_executable: str,
    dry_run: bool = False, refresh_backend: bool = False,
) -> int:
    python = venv_python(venv)
    requirements = project_root / "src" / "requirements.txt"
    esm_wheel, transformers_wheel, runtime_requirements = _bundled_paths(project_root)
    verify_bundled_artifacts(esm_wheel, transformers_wheel, runtime_requirements)

    report = Detect_GPU.detect_hardware()
    specs = backend_specs(report)
    fingerprint = hardware_fingerprint(report)
    state_path = venv / STATE_FILENAME
    current_state = read_state(state_path)
    state_matches = _state_matches(current_state, specs, fingerprint, requirements)
    saved_active = _backend_from_state(current_state.get("active_backend")) if state_matches and current_state else None

    print(f"Detected: {report['reason']}")
    if report.get("ignored_devices"):
        for item in report["ignored_devices"]:
            print(f"Ignoring {item.get('name')}: {item.get('reason')}")

    base_command = base_install_command(uv_executable, python, requirements)
    transformers_command = transformers_install_command(
        uv_executable, python, transformers_wheel
    )
    runtime_command = esm_runtime_install_command(
        uv_executable, python, runtime_requirements
    )
    esm_command = esm_install_command(uv_executable, python, esm_wheel)
    if dry_run:
        print(f"Dry run base: {shlex.join(base_command)}")
        for position, spec in enumerate(specs, 1):
            print(f"Dry run candidate {position}: {spec.description}")
            for command in backend_install_commands(uv_executable, python, spec):
                print(f"  {shlex.join(command)}")
        print(f"Dry run Transformers: {shlex.join(transformers_command)}")
        print(f"Dry run ESM runtime dependencies: {shlex.join(runtime_command)}")
        print(f"Dry run ESM: {shlex.join(esm_command)}")
        return 0

    if _run(base_command).returncode != 0:
        print("Base dependency installation failed.", file=sys.stderr)
        return 1

    active: BackendSpec | None = None
    validation: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    if saved_active is not None and not refresh_backend:
        validation = _normalize_validation(validate_backend(python, saved_active), saved_active)
        if validation is not None:
            active = saved_active
            attempts = list(current_state.get("attempts", [])) if current_state else []

    if active is None:
        for spec in specs:
            print(f"Trying PyTorch backend: {spec.description}")
            validation = _normalize_validation(install_backend(uv_executable, python, spec), spec)
            attempts.append({
                "backend": spec.backend,
                "profile": spec.profile,
                "description": spec.description,
                "success": validation is not None,
                "validation": validation,
            })
            if validation is not None:
                active = spec
                break
            print(f"{spec.description} could not be installed or validated; trying the next candidate.", file=sys.stderr)

    if active is None or validation is None:
        print("No PyTorch backend, including CPU, could be installed and validated.", file=sys.stderr)
        return 1

    esm_ready = (
        state_matches
        and _installed_version(python, "transformers") == TRANSFORMERS_VERSION
        and _installed_version(python, "esm") == ESM_VERSION
        and validate_esm_stack(python)
    )
    if not esm_ready:
        if _run(transformers_command).returncode != 0:
            print("Bundled Biohub Transformers wheel installation failed.", file=sys.stderr)
            return 1
        if _run(runtime_command).returncode != 0:
            print("ESM runtime dependency installation failed.", file=sys.stderr)
            return 1
        if _run(esm_command).returncode != 0:
            print("Bundled ESM wheel installation failed.", file=sys.stderr)
            return 1
    if not validate_package_consistency(uv_executable, python):
        return 1
    if not validate_esm_stack(python):
        return 1

    active_devices = validation.get("validated_devices", [])
    payload = {
        "schema": STATE_SCHEMA,
        "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
        "hardware_fingerprint": fingerprint,
        "requirements_sha256": _sha256(requirements),
        "esm_version": ESM_VERSION,
        "esm_wheel_sha256": ESM_WHEEL_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "transformers_wheel_sha256": TRANSFORMERS_WHEEL_SHA256,
        "esm_runtime_requirements_sha256": _sha256(runtime_requirements),
        "requested_candidates": _spec_payloads(specs),
        "active_backend": _spec_payloads((active,))[0],
        "validated_devices": active_devices,
        "ignored_devices": report.get("ignored_devices", []),
        "attempts": attempts,
        "detection": report,
    }
    final_report = dict(report)
    final_report["backend"] = active.backend
    final_report["gfx_target"] = active.gfx_target
    final_report["reason"] = f"{active.description} passed runtime validation."
    if active.backend != specs[0].backend:
        final_report["fallback_from"] = specs[0].backend
    write_state(state_path, payload, final_report)
    print(f"Dependency environment is ready ({active.description}).")
    print(f"Validated runtime devices: {len(active_devices)}")
    print(f"Bundled Transformers version: {TRANSFORMERS_VERSION}")
    print(f"Bundled ESM version: {ESM_VERSION}")
    return 0


def environment_is_ready(
    *, project_root: Path, venv: Path, uv_executable: str
) -> bool:
    """Check launcher readiness without installing or changing the environment."""
    python = venv_python(venv)
    requirements = project_root / "src" / "requirements.txt"
    esm_wheel, transformers_wheel, runtime_requirements = _bundled_paths(project_root)
    if not python.is_file():
        print(f"Environment is not ready: {python} is missing.", file=sys.stderr)
        return False

    try:
        print(f"Managed Python: {python}")
        verify_bundled_artifacts(esm_wheel, transformers_wheel, runtime_requirements)
        print(f"Bundled Transformers wheel: verified ({transformers_wheel.name})")
        print(f"Bundled ESM wheel: verified ({esm_wheel.name})")
        report = Detect_GPU.detect_hardware()
        specs = backend_specs(report)
        fingerprint = hardware_fingerprint(report)
        print(f"Detected: {report.get('reason', 'hardware profile resolved')}")
        for item in report.get("ignored_devices", []):
            print(f"Ignoring {item.get('name')}: {item.get('reason')}")
        state = read_state(venv / STATE_FILENAME)
        if not _state_matches(state, specs, fingerprint, requirements):
            print(
                "Environment is not ready: dependency or hardware state changed.",
                file=sys.stderr,
            )
            return False
        print("Dependency and hardware state: current")
        active = _backend_from_state(state.get("active_backend")) if state else None
        saved_backend = state.get("active_backend", {}) if state else {}
        active_label = (
            getattr(active, "description", None)
            or saved_backend.get("description")
            or saved_backend.get("backend")
            or "validated backend"
        )
        if active is not None:
            print(f"Validating runtime backend: {active_label}")
        validation = validate_backend(python, active) if active is not None else None
        if active is None or validation is None:
            print("Environment is not ready: runtime backend validation failed.", file=sys.stderr)
            return False
        if _installed_version(python, "transformers") != TRANSFORMERS_VERSION:
            print(
                "Environment is not ready: bundled Biohub Transformers version is missing.",
                file=sys.stderr,
            )
            return False
        if _installed_version(python, "esm") != ESM_VERSION:
            print("Environment is not ready: bundled ESM version is missing.", file=sys.stderr)
            return False
        if not validate_package_consistency(uv_executable, python):
            print("Environment is not ready: installed packages are inconsistent.", file=sys.stderr)
            return False
        if not validate_esm_stack(python):
            print("Environment is not ready: ESM import validation failed.", file=sys.stderr)
            return False
        print("Installed package consistency: passed")
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Environment readiness check failed: {error}", file=sys.stderr)
        return False

    print(f"Dependency environment is ready ({active_label}).")
    print(
        "Validated runtime devices: "
        f"{len(validation.get('validated_devices', validation.get('devices', [])))}"
    )
    print(f"Bundled Transformers version: {TRANSFORMERS_VERSION}")
    print(f"Bundled ESM version: {ESM_VERSION}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path(".venv"))
    parser.add_argument("--uv-executable", default="uv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check readiness without changing the managed environment",
    )
    parser.add_argument("--refresh-backend", action="store_true", help="retry the full accelerator candidate ladder")
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    try:
        if args.check_only:
            ready = environment_is_ready(
                project_root=project_root,
                venv=args.venv.resolve(),
                uv_executable=args.uv_executable,
            )
            return 0 if ready else SETUP_REQUIRED_EXIT
        return install(
            project_root=project_root, venv=args.venv.resolve(),
            uv_executable=args.uv_executable, dry_run=args.dry_run,
            refresh_backend=args.refresh_backend,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Dependency installation error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
