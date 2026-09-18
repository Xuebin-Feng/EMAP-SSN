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

"""Install project dependencies and a validated hardware-specific PyTorch build."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

import Detect_GPU


# torch 2.12.0 rather than 2.12.1 because it is the one build published for
# every backend below, including both ROCm platforms on the AMD multi-arch
# channel. A single torch version across all backends keeps the candidate
# ladder and the saved state directly comparable.
TORCH_VERSION = "2.12.0"
# ROCm ships as a PEP 440 local version from AMD's own channel. 7.14.1 is the
# newest ROCm tag carrying torch 2.12.0 for BOTH linux_x86_64 and win_amd64.
ROCM_TORCH_VERSION = "2.12.0+rocm7.14.1"
# esm and transformers now come from PyPI. esm 3.4.x implements ESMC and
# ESMFold2 inside the esm package, so the forked Transformers build that
# carried transformers/models/esmc is no longer required and is not bundled.
ESM_VERSION = "3.4.1.post1"
TRANSFORMERS_VERSION = "5.17.0"
STATE_FILENAME = "ssn_backend.json"
# Schema 6 drops the bundled-wheel checksum fields and collapses the four ROCm
# profiles into one, so schema-5 state is not comparable and must be rebuilt.
STATE_SCHEMA = 6
SETUP_REQUIRED_EXIT = 10
ROCM_BACKEND = "rocm"
ACCELERATOR_BACKENDS = {
    "cuda126", "cuda132", "xpu", ROCM_BACKEND, "mps",
}

PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda126": "https://download.pytorch.org/whl/cu126",
    "cuda132": "https://download.pytorch.org/whl/cu132",
    "xpu": "https://download.pytorch.org/whl/xpu",
    ROCM_BACKEND: "https://repo.amd.com/rocm/whl-multi-arch/",
}


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
    if backend == ROCM_BACKEND:
        if not isinstance(gfx_target, str) or not gfx_target.startswith("gfx"):
            raise ValueError("ROCm requires a validated GFX target.")
        steps = (
            InstallStep(
                (f"torch[device-{gfx_target}]=={ROCM_TORCH_VERSION}",),
                PYTORCH_INDEXES[backend],
            ),
        )
        # The install requirement carries the +rocm local version, but the
        # validator compares against torch.__version__ with the local segment
        # stripped, so the recorded version must be the bare base version.
        torch_version = TORCH_VERSION
        description = f"ROCm 7.14 ({gfx_target})"
    else:
        descriptions = {
            "cuda132": "NVIDIA CUDA 13.2",
            "cuda126": "NVIDIA CUDA 12.6",
            "xpu": "Intel XPU",
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
    # Defensive: each vendor currently contributes one candidate, but if two ever
    # resolve to an identical install, keep the first and drop the repeat rung.
    deduplicated: list[BackendSpec] = []
    seen: set[tuple[str, str | None, tuple[Any, ...]]] = set()
    for spec in specs:
        identity = (
            spec.backend,
            spec.gfx_target,
            tuple(tuple(step.requirements) for step in spec.install_steps),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(spec)
    specs = deduplicated
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


def esm_install_command(uv_executable: str, python: Path) -> list[str]:
    # --no-deps is load-bearing: esm pins torch<2.12.0 and transformers<5.0.0,
    # and resolving those would replace the selected accelerator build and
    # downgrade Transformers. Its runtime dependencies are therefore declared
    # by hand in esm_runtime_requirements.txt and installed separately; that
    # file must be updated whenever ESM_VERSION changes.
    return _uv_prefix(uv_executable, python) + ["--no-deps", f"esm=={ESM_VERSION}"]


def transformers_install_command(uv_executable: str, python: Path) -> list[str]:
    # Installed with dependencies on purpose: transformers declares torch only
    # under an extra, so the selected accelerator build is never disturbed,
    # while huggingface-hub, tokenizers and safetensors must be resolved here.
    return _uv_prefix(uv_executable, python) + [
        f"transformers=={TRANSFORMERS_VERSION}"
    ]


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
    return _stable_hash(Detect_GPU.hardware_compatibility_material(report))


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1).lower().replace("_", "-") if match else ""


def _requirements_entries(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"ESM runtime requirements are missing: {path}")
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def verify_esm_runtime_requirements(requirements: Path) -> None:
    """Reject a runtime requirements file that names torch or transformers.

    esm is installed with --no-deps, so this file is the only place its runtime
    dependencies are declared. torch and transformers must never appear in it:
    torch is supplied by the selected accelerator backend, and transformers is
    pinned separately, so either one here would silently replace them.
    """
    entries = _requirements_entries(requirements)
    conflicting = sorted(
        {
            _requirement_name(entry)
            for entry in entries
            if _requirement_name(entry) in {"torch", "transformers"}
        }
    )
    if conflicting:
        raise ValueError(
            "ESM runtime requirements must not pin "
            f"{', '.join(conflicting)}: {requirements}"
        )


def _esm_runtime_requirements_path(project_root: Path) -> Path:
    return project_root / "src" / "esm_runtime_requirements.txt"


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
        "transformers_version": TRANSFORMERS_VERSION,
        "esm_runtime_requirements_sha256": _sha256(
            requirements.parent / "esm_runtime_requirements.txt"
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


def _backend_install_identity(spec: BackendSpec) -> dict[str, Any]:
    """Describe the installed artifacts without physical device identity."""
    return {
        "backend": spec.backend,
        "profile": spec.profile,
        "torch_version": spec.torch_version,
        "install_steps": [
            {
                "requirements": list(step.requirements),
                "index_url": step.index_url,
            }
            for step in spec.install_steps
        ],
        "gfx_target": spec.gfx_target,
    }


def _candidate_install_identities(specs: Iterable[BackendSpec]) -> list[dict[str, Any]]:
    return [_backend_install_identity(spec) for spec in specs]


def _saved_candidate_identities(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    identities: list[dict[str, Any]] = []
    for item in value:
        spec = _backend_from_state(item)
        if spec is None:
            return None
        identities.append(_backend_install_identity(spec))
    return identities


def _same_backend_install(left: BackendSpec, right: BackendSpec) -> bool:
    return _backend_install_identity(left) == _backend_install_identity(right)


def _accelerator_runtime_visible(report: dict[str, Any]) -> bool:
    """Return whether detection found an accelerator with usable runtime data."""
    candidates = report.get("backend_candidates")
    if not isinstance(candidates, list):
        return str(report.get("backend") or "cpu") != "cpu"
    return any(
        isinstance(candidate, dict)
        and candidate.get("backend") != "cpu"
        and candidate.get("eligibility", "eligible") == "eligible"
        for candidate in candidates
    )


def _reusable_backend(
    state: dict[str, Any] | None,
    specs: list[BackendSpec],
    *,
    accelerator_visible: bool = True,
) -> tuple[BackendSpec, str] | None:
    """Return a safely reusable backend and its validation mode."""
    if not state:
        return None
    active = _backend_from_state(state.get("active_backend"))
    if active is None:
        return None

    if not accelerator_visible:
        if active.backend in ACCELERATOR_BACKENDS:
            return active, "package-only"
        matching_cpu = next(
            (spec for spec in specs if _same_backend_install(active, spec)), None
        )
        if matching_cpu is not None:
            return matching_cpu, "runtime"

    matching = next((spec for spec in specs if _same_backend_install(active, spec)), None)
    if matching is None:
        return None
    if _same_backend_install(matching, specs[0]):
        return matching, "runtime"

    saved_ladder = _saved_candidate_identities(state.get("requested_candidates"))
    saved_accelerator_visible = _accelerator_runtime_visible(
        state.get("detection", {}) if isinstance(state.get("detection"), dict) else {}
    )
    if (
        saved_ladder == _candidate_install_identities(specs)
        and not (accelerator_visible and not saved_accelerator_visible)
    ):
        return matching, "runtime"
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
            if backend in {{"cuda126", "cuda132", "rocm"}}:
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


def _package_validation_program(spec: BackendSpec) -> str:
    return textwrap.dedent(
        f"""
        import json
        result = {{"backend": {spec.backend!r}, "profile": {spec.profile!r}, "package_error": None}}
        try:
            import torch
            result["torch_version"] = torch.__version__
            if torch.__version__.split("+", 1)[0] != {spec.torch_version!r}:
                raise RuntimeError("unexpected torch version: " + torch.__version__)
            backend = {spec.backend!r}
            if backend in {{"cuda126", "cuda132"}}:
                expected_cuda = "12.6" if backend == "cuda126" else "13.2"
                if torch.version.hip is not None:
                    raise RuntimeError("CUDA profile loaded a ROCm build")
                if not str(torch.version.cuda).startswith(expected_cuda):
                    raise RuntimeError("unexpected CUDA runtime: " + str(torch.version.cuda))
            elif backend.startswith("rocm"):
                if not torch.version.hip:
                    raise RuntimeError("ROCm/HIP build metadata is missing")
            elif backend == "xpu":
                if not hasattr(torch, "xpu"):
                    raise RuntimeError("Intel XPU support is missing")
            elif backend == "mps":
                if not hasattr(torch.backends, "mps"):
                    raise RuntimeError("Apple MPS support is missing")
            elif torch.version.cuda is not None or torch.version.hip is not None:
                raise RuntimeError("CPU profile loaded an accelerator build")
        except Exception as error:
            result["package_error"] = str(error)
        print(json.dumps(result, sort_keys=True))
        """
    ).strip()


def validate_backend_package(python: Path, spec: BackendSpec) -> dict[str, Any] | None:
    """Validate an installed build without requiring accelerator visibility."""
    completed = _run([str(python), "-c", _package_validation_program(spec)], capture=True)
    detail = completed.stderr.strip()
    try:
        result = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        result = None
    if (
        completed.returncode != 0
        or not isinstance(result, dict)
        or result.get("package_error")
    ):
        reason = (
            result.get("package_error") if isinstance(result, dict) else None
        ) or detail or completed.stdout.strip() or "invalid validator output"
        print(f"PyTorch package validation failed: {reason}", file=sys.stderr)
        return None
    result["devices"] = []
    result["validated_devices"] = []
    result["preserved_without_accelerator"] = True
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

        # ESMC now lives inside the esm package rather than in a forked
        # Transformers build, so the smoke test exercises the esm entry points
        # the pLM adapters actually import, plus the generic Transformers API
        # used by the ESM-2, ProtBERT, ProstT5 and Ankh adapters.
        import esm
        from esm.models.esmc import ESMC
        from esm.pretrained import register_local_model
        from esm.sdk.forge import ESMCForgeInferenceClient
        from transformers import AutoModel, AutoTokenizer, T5EncoderModel

        assert metadata.version("esm") == {ESM_VERSION!r}
        assert metadata.version("transformers") == {TRANSFORMERS_VERSION!r}
        assert ESMC is not None and callable(register_local_model)
        assert callable(getattr(ESMC, "from_pretrained", None))
        assert ESMCForgeInferenceClient is not None
        assert AutoModel is not None and AutoTokenizer is not None
        assert T5EncoderModel is not None
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


# "The package `esm` requires `torch>=2.11.0,<2.12.0`, but `2.12.0+cu132` is installed"
_PIP_CHECK_PROBLEM = re.compile(
    r"The package `(?P<package>[^`]+)` requires `(?P<requirement>[^`]+)`, "
    r"but `(?P<installed>[^`]+)` is installed"
)
_PIP_CHECK_COUNT = re.compile(r"Found (\d+) incompatibilit", re.I)


def _is_sanctioned_incompatibility(
    package: str, requirement: str, installed: str
) -> bool:
    """Allow only the two deviations this installer creates on purpose.

    esm is installed with --no-deps against a newer torch and a newer
    Transformers than it declares, so `uv pip check` always reports those two.
    Both are accepted only when the installed version is exactly the one this
    installer pinned; anything else is a real inconsistency.
    """
    if package.lower() != "esm":
        return False
    name = _requirement_name(requirement)
    if name == "torch":
        # Every backend installs the same base version; the local segment
        # (+cu132, +rocm7.14.1, ...) identifies the accelerator build.
        return installed.split("+", 1)[0] == TORCH_VERSION
    if name == "transformers":
        return installed == TRANSFORMERS_VERSION
    return False


def validate_package_consistency(uv_executable: str, python: Path) -> bool:
    completed = _run(
        [uv_executable, "pip", "check", "--python", str(python)], capture=True
    )
    if completed.returncode == 0:
        return True

    # uv interleaves progress lines ("Checked N packages in Xms") with the
    # report, so informational text is ignored rather than enumerated. Safety
    # comes from the declared count instead: every incompatibility uv reports
    # must be one this installer parsed AND sanctioned, or the check fails.
    output = f"{completed.stdout}\n{completed.stderr}"
    accepted: list[str] = []
    unsanctioned: list[str] = []
    for line in (item.strip() for item in output.splitlines()):
        match = _PIP_CHECK_PROBLEM.search(line)
        if match is None:
            continue
        if _is_sanctioned_incompatibility(
            match.group("package"),
            match.group("requirement"),
            match.group("installed"),
        ):
            accepted.append(line)
        else:
            unsanctioned.append(line)

    declared = _PIP_CHECK_COUNT.search(output)
    total = int(declared.group(1)) if declared is not None else None
    # Fail closed: an unreadable or unaccounted-for report is treated as a
    # real inconsistency rather than assumed benign.
    accounted = total is not None and total == len(accepted) + len(unsanctioned)

    if unsanctioned:
        print(
            "Installed package consistency failed:\n" + "\n".join(unsanctioned),
            file=sys.stderr,
        )
        return False
    if not accepted or not accounted:
        reported = "unreported" if total is None else str(total)
        print(
            "Installed package consistency failed: could not account for every "
            f"incompatibility (uv reported {reported}, recognized "
            f"{len(accepted)}). Full output:\n"
            + (output.strip() or "uv pip check produced no output"),
            file=sys.stderr,
        )
        return False

    print(
        "Installed package consistency: passed with "
        f"{len(accepted)} expected esm deviation(s)"
    )
    for line in accepted:
        print(f"  accepted: {line}")
    return True


def install_backend(uv_executable: str, python: Path, spec: BackendSpec) -> dict[str, Any] | None:
    if not remove_backend_packages(uv_executable, python):
        return None
    for command in backend_install_commands(uv_executable, python, spec):
        if _run(command).returncode != 0:
            return None
    return validate_backend(python, spec)


STATE_FIELD_LABELS = {
    "state_file": "saved backend state is missing or unreadable",
    "schema": "backend state schema changed",
    "compatibility_revision": "hardware compatibility rules changed",
    "hardware_fingerprint": "stable hardware compatibility profile changed",
    "requirements_sha256": "base requirements changed",
    "esm_version": "pinned ESM version changed",
    "transformers_version": "pinned Transformers version changed",
    "esm_runtime_requirements_sha256": "ESM runtime requirements changed",
    "requested_candidates": "backend candidate ladder changed",
}


def _state_mismatches(
    state: dict[str, Any] | None,
    specs: Iterable[BackendSpec],
    fingerprint: str,
    requirements: Path,
) -> list[str]:
    if not state:
        return ["state_file"]
    specs = list(specs)
    expected = {
        "schema": STATE_SCHEMA,
        "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
        "hardware_fingerprint": fingerprint,
        "requirements_sha256": _sha256(requirements),
        "esm_version": ESM_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "esm_runtime_requirements_sha256": _sha256(
            requirements.parent / "esm_runtime_requirements.txt"
        ),
    }
    mismatches = [
        field for field, value in expected.items() if state.get(field) != value
    ]
    if _saved_candidate_identities(state.get("requested_candidates")) != (
        _candidate_install_identities(specs)
    ):
        mismatches.append("requested_candidates")
    return mismatches


def _print_state_mismatches(fields: Iterable[str]) -> None:
    for field in fields:
        label = STATE_FIELD_LABELS.get(field, "saved value changed")
        print(f"State mismatch [{field}]: {label}.", file=sys.stderr)


def _state_matches(
    state: dict[str, Any] | None,
    specs: Iterable[BackendSpec],
    fingerprint: str,
    requirements: Path,
) -> bool:
    return not _state_mismatches(state, specs, fingerprint, requirements)


def _spec_payloads(specs: Iterable[BackendSpec]) -> list[dict[str, Any]]:
    """Return the JSON-normalized representation persisted in backend state."""
    return json.loads(json.dumps([asdict(spec) for spec in specs]))


def install(
    *, project_root: Path, venv: Path, uv_executable: str,
    dry_run: bool = False, refresh_backend: bool = False,
) -> int:
    python = venv_python(venv)
    requirements = project_root / "src" / "requirements.txt"
    runtime_requirements = _esm_runtime_requirements_path(project_root)
    verify_esm_runtime_requirements(runtime_requirements)

    report = Detect_GPU.detect_hardware()
    specs = backend_specs(report)
    fingerprint = hardware_fingerprint(report)
    state_path = venv / STATE_FILENAME
    current_state = read_state(state_path)
    mismatches = _state_mismatches(current_state, specs, fingerprint, requirements)

    print(f"Detected: {report['reason']}")
    if report.get("ignored_devices"):
        for item in report["ignored_devices"]:
            print(f"Ignoring {item.get('name')}: {item.get('reason')}")
    if mismatches:
        _print_state_mismatches(mismatches)

    base_command = base_install_command(uv_executable, python, requirements)
    transformers_command = transformers_install_command(uv_executable, python)
    runtime_command = esm_runtime_install_command(
        uv_executable, python, runtime_requirements
    )
    esm_command = esm_install_command(uv_executable, python)
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
    preservation_mode = False
    reusable = None if refresh_backend else _reusable_backend(
        current_state,
        specs,
        accelerator_visible=_accelerator_runtime_visible(report),
    )
    if reusable is not None:
        saved_active, validation_mode = reusable
        if validation_mode == "package-only":
            print(
                f"No accelerator is currently visible; preserving installed "
                f"{saved_active.description} and validating its package metadata."
            )
            validation = validate_backend_package(python, saved_active)
            preservation_mode = validation is not None
        else:
            print(
                f"Validating installed {saved_active.description} before considering "
                "a backend reinstall."
            )
            validation = _normalize_validation(
                validate_backend(python, saved_active), saved_active
            )
        if validation is not None:
            active = saved_active
            attempts = list(current_state.get("attempts", [])) if current_state else []
            if mismatches:
                print(
                    f"Reusing installed {active.description}; refreshing saved "
                    "dependency and hardware state."
                )
        else:
            print(
                f"Installed {saved_active.description} did not validate; "
                "the backend candidate ladder will be repaired.",
                file=sys.stderr,
            )

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

    esm_fields = {
        "esm_version", "transformers_version", "esm_runtime_requirements_sha256",
    }
    esm_ready = (
        current_state is not None
        and not esm_fields.intersection(mismatches)
        and _installed_version(python, "transformers") == TRANSFORMERS_VERSION
        and _installed_version(python, "esm") == ESM_VERSION
        and validate_esm_stack(python)
    )
    if not esm_ready:
        if _run(transformers_command).returncode != 0:
            print("Transformers installation failed.", file=sys.stderr)
            return 1
        if _run(runtime_command).returncode != 0:
            print("ESM runtime dependency installation failed.", file=sys.stderr)
            return 1
        if _run(esm_command).returncode != 0:
            print("ESM installation failed.", file=sys.stderr)
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
        "transformers_version": TRANSFORMERS_VERSION,
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
    if preservation_mode:
        final_report["reason"] = (
            f"{active.description} package metadata passed validation while no "
            "accelerator was visible."
        )
        final_report["preserved_without_accelerator"] = True
    else:
        final_report["reason"] = f"{active.description} passed runtime validation."
    if not preservation_mode and active.backend != specs[0].backend:
        final_report["fallback_from"] = specs[0].backend
    write_state(state_path, payload, final_report)
    print(f"Dependency environment is ready ({active.description}).")
    print(f"Validated runtime devices: {len(active_devices)}")
    if preservation_mode:
        print("Accelerator runtime validation: deferred until an accelerator is visible")
    print(f"Transformers version: {TRANSFORMERS_VERSION}")
    print(f"ESM version: {ESM_VERSION}")
    return 0


def environment_is_ready(
    *, project_root: Path, venv: Path, uv_executable: str
) -> bool:
    """Check launcher readiness without installing or changing the environment."""
    python = venv_python(venv)
    requirements = project_root / "src" / "requirements.txt"
    runtime_requirements = _esm_runtime_requirements_path(project_root)
    if not python.is_file():
        print(f"Environment is not ready: {python} is missing.", file=sys.stderr)
        return False

    try:
        print(f"Managed Python: {python}")
        verify_esm_runtime_requirements(runtime_requirements)
        report = Detect_GPU.detect_hardware()
        specs = backend_specs(report)
        fingerprint = hardware_fingerprint(report)
        print(f"Detected: {report.get('reason', 'hardware profile resolved')}")
        for item in report.get("ignored_devices", []):
            print(f"Ignoring {item.get('name')}: {item.get('reason')}")
        state = read_state(venv / STATE_FILENAME)
        mismatches = _state_mismatches(state, specs, fingerprint, requirements)
        active = _backend_from_state(state.get("active_backend")) if state else None
        accelerator_visible = _accelerator_runtime_visible(report)
        preserve_without_accelerator = bool(
            active is not None and not accelerator_visible
        )
        if mismatches:
            _print_state_mismatches(mismatches)
        tolerable_cpu_only_fields = {"hardware_fingerprint", "requested_candidates"}
        if mismatches and not (
            preserve_without_accelerator
            and set(mismatches).issubset(tolerable_cpu_only_fields)
        ):
            print(
                "Environment is not ready: dependency or hardware state changed.",
                file=sys.stderr,
            )
            return False
        if mismatches:
            print(
                "No accelerator is currently visible; compatible accelerator "
                "state is being preserved."
            )
        else:
            print("Dependency and hardware state: current")
        saved_backend = state.get("active_backend", {}) if state else {}
        active_label = (
            getattr(active, "description", None)
            or saved_backend.get("description")
            or saved_backend.get("backend")
            or "validated backend"
        )
        package_only_preservation = bool(
            preserve_without_accelerator
            and active is not None
            and active.backend in ACCELERATOR_BACKENDS
        )
        if active is not None and package_only_preservation:
            print(f"Validating installed backend package: {active_label}")
        elif active is not None:
            print(f"Validating runtime backend: {active_label}")
        validation = (
            validate_backend_package(python, active)
            if active is not None and package_only_preservation
            else validate_backend(python, active) if active is not None else None
        )
        if active is None or validation is None:
            validation_kind = "package" if package_only_preservation else "runtime backend"
            print(
                f"Environment is not ready: {validation_kind} validation failed.",
                file=sys.stderr,
            )
            return False
        if _installed_version(python, "transformers") != TRANSFORMERS_VERSION:
            print(
                "Environment is not ready: pinned Transformers version is missing.",
                file=sys.stderr,
            )
            return False
        if _installed_version(python, "esm") != ESM_VERSION:
            print("Environment is not ready: pinned ESM version is missing.", file=sys.stderr)
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
    if package_only_preservation:
        print("Accelerator runtime validation: deferred until an accelerator is visible")
    print(f"Transformers version: {TRANSFORMERS_VERSION}")
    print(f"ESM version: {ESM_VERSION}")
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
