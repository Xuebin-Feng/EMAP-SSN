# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import Detect_GPU  # noqa: E402
import Install_Dependencies  # noqa: E402


def detected_report(**overrides):
    report = {
        "platform": "windows",
        "os_version": "10.0.26100",
        "windows_build": 26100,
        "controllers": [],
        "processors": [],
        "nvidia_devices": [],
        "vendor": "CPU",
        "backend": "cpu",
        "gfx_target": None,
        "reason": "test",
    }
    report.update(overrides)
    return report


class GPUDetectionTests(unittest.TestCase):
    def _detect(self, *, system="Windows", version="10.0.26200", controllers=(), processors=(), nvidia=()):
        with mock.patch.object(Detect_GPU.platform, "system", return_value=system), \
                mock.patch.object(Detect_GPU.platform, "version", return_value=version), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=list(controllers)), \
                mock.patch.object(Detect_GPU, "_windows_names", return_value=list(processors)), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=list(nvidia)):
            return Detect_GPU.detect_hardware()

    def test_windows_amd_models_map_to_official_gfx_targets(self):
        # ROCm 7.14 is the single AMD profile, so every supported model now
        # resolves to the same backend and differs only in its GFX target.
        examples = {
            "AMD Radeon RX 9070 XT": "gfx1201",
            "AMD Radeon RX 9060 XT": "gfx1200",
            "AMD Radeon RX 7900 XTX": "gfx1100",
            "AMD Radeon RX 7800 XT": "gfx1101",
            "AMD Radeon RX 7600": "gfx1102",
            "AMD Radeon PRO W6800": "gfx1030",
            "AMD Radeon 780M Graphics": "gfx1103",
            "AMD Radeon 890M Graphics": "gfx1150",
            "AMD Radeon 8060S Graphics": "gfx1151",
            "AMD Radeon 860M Graphics": "gfx1152",
        }
        for name, target in examples.items():
            with self.subTest(name=name):
                report = self._detect(controllers=[name])
                self.assertEqual(report["backend"], Install_Dependencies.ROCM_BACKEND)
                self.assertEqual(report["gfx_target"], target)

    def test_unknown_windows_amd_uses_intel_if_available_otherwise_cpu(self):
        mixed = self._detect(controllers=["AMD Radeon Graphics", "Intel Arc A770"])
        self.assertEqual(mixed["backend"], "xpu")
        amd_only = self._detect(controllers=["AMD Radeon Graphics"])
        self.assertEqual(amd_only["backend"], "cpu")

    def test_windows_11_25h2_routes_supported_amd_to_rocm(self):
        report = self._detect(
            version="10.0.26200", controllers=["AMD Radeon RX 7900 XTX"]
        )
        self.assertEqual(report["backend"], "rocm")
        self.assertEqual(report["gfx_target"], "gfx1100")
        self.assertEqual(
            [candidate["backend"] for candidate in report["backend_candidates"]],
            ["rocm", "cpu"],
        )

    def test_windows_before_11_25h2_rejects_rocm(self):
        # AMD publishes native Windows ROCm for Windows 11 25H2 only, so both
        # Windows 10 and an older Windows 11 build fall through to CPU.
        for version in ("10.0.19045", "10.0.26100"):
            with self.subTest(version=version):
                report = self._detect(
                    version=version, controllers=["AMD Radeon RX 7900 XTX"]
                )
                self.assertEqual(report["backend"], "cpu")
                self.assertEqual(
                    [candidate["backend"] for candidate in report["backend_candidates"]],
                    ["cpu"],
                )
                self.assertIn("Windows 11 25H2", report["reason"])

    def test_nvidia_cuda_version_depends_on_architecture_and_driver(self):
        modern = [{"name": "RTX 5090", "compute_capability": "12.0", "driver_version": "595.10"}]
        self.assertEqual(self._detect(nvidia=modern)["backend"], "cuda132")
        old_driver = [{"name": "RTX 4090", "compute_capability": "8.9", "driver_version": "579.99"}]
        self.assertEqual(self._detect(nvidia=old_driver)["backend"], "cuda126")
        old_gpu = [{"name": "GTX 1080", "compute_capability": "6.1", "driver_version": "595.10"}]
        self.assertEqual(self._detect(nvidia=old_gpu)["backend"], "cuda126")

    def test_linux_nvidia_inventory_merges_lspci_and_smi_by_pci_address(self):
        linux_device = {
            "id": "0000:01:00.0",
            "name": "NVIDIA Corporation AD104 [GeForce RTX 4070]",
            "vendor": "NVIDIA",
            "pci_id": "10de:2786",
            "driver_version": None,
            "driver": "nvidia",
            "kind": "discrete",
            "architecture": None,
            "source": "lspci",
        }
        smi_device = {
            "bus_id": "00000000:01:00.0",
            "name": "NVIDIA GeForce RTX 4070",
            "compute_capability": "8.9",
            "driver_version": "595.84",
        }
        with mock.patch.object(Detect_GPU.platform, "system", return_value="Linux"), \
                mock.patch.object(Detect_GPU.platform, "version", return_value="6.8"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=[]), \
                mock.patch.object(Detect_GPU, "_linux_inventory", return_value=[linux_device]), \
                mock.patch.object(
                    Detect_GPU, "_read_os_release", return_value={"id": "ubuntu", "version_id": "24.04"}
                ), mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=[smi_device]):
            report = Detect_GPU.detect_hardware()

        nvidia_devices = [device for device in report["devices"] if device["vendor"] == "NVIDIA"]
        self.assertEqual(len(nvidia_devices), 1)
        self.assertEqual(nvidia_devices[0]["id"], "0000:01:00.0")
        self.assertEqual(nvidia_devices[0]["compute_capability"], "8.9")
        self.assertEqual(nvidia_devices[0]["driver_version"], "595.84")
        self.assertEqual(report["backend"], "cuda132")

    def test_linux_amd_and_apple_silicon_backends(self):
        # The inventory is supplied explicitly: on Linux the controller names
        # only seed discovery, so an unpatched host would otherwise decide this.
        amd_device = {
            "id": "0000:03:00.0",
            "name": "AMD Radeon RX 7900 XTX",
            "vendor": "AMD",
            "pci_id": "1002:744c",
            "driver_version": None,
            "driver": "amdgpu",
            "kind": "discrete",
            "architecture": "gfx1100",
            "source": "lspci",
        }
        with mock.patch.object(
                Detect_GPU, "_read_os_release", return_value={"id": "ubuntu", "version_id": "24.04"}
            ), mock.patch.object(
                Detect_GPU, "_linux_inventory", return_value=[amd_device]
            ), mock.patch.object(
                Detect_GPU, "_rocm_targets", return_value={"gfx1100"}
            ), mock.patch.object(
                Detect_GPU.Path, "exists", return_value=True
            ), mock.patch.object(
                Detect_GPU.os, "access", return_value=True
            ):
            linux = self._detect(
                system="Linux", version="6.8", controllers=["AMD Radeon RX 7900 XTX"]
            )
        # The four ROCm profiles collapsed into one, so the ladder is the single
        # ROCm candidate followed by the portable CPU fallback.
        self.assertEqual(linux["backend"], "rocm")
        self.assertEqual(linux["gfx_target"], "gfx1100")
        self.assertEqual(
            [candidate["backend"] for candidate in linux["backend_candidates"]],
            ["rocm", "cpu"],
        )
        with mock.patch.object(Detect_GPU.platform, "machine", return_value="arm64"), \
                mock.patch.object(Detect_GPU.platform, "system", return_value="Darwin"), \
                mock.patch.object(Detect_GPU.platform, "version", return_value="25.0"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=[]), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=[]):
            apple = Detect_GPU.detect_hardware()
        self.assertEqual(apple["backend"], "mps")

    def test_json_cli_contains_selection_reason(self):
        report = detected_report(reason="No accelerator")
        output = io.StringIO()
        with mock.patch.object(Detect_GPU, "detect_hardware", return_value=report), redirect_stdout(output):
            self.assertEqual(Detect_GPU.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["reason"], "No accelerator")


class DependencyInstallerTests(unittest.TestCase):
    def _cpu_report(self):
        return detected_report(
            backend_candidates=[{
                "backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]
            }],
            compatibility_revision=Detect_GPU.COMPATIBILITY_REVISION,
            platform="windows",
            os="Windows",
            devices=[],
        )

    def _cuda_report(self, backend="cuda126", device_id="0000:01:00.0"):
        return detected_report(
            backend=backend,
            backend_candidates=[
                {
                    "backend": backend, "profile": backend, "vendor": "NVIDIA",
                    "device_ids": [device_id],
                },
                {"backend": "cpu", "profile": "cpu", "vendor": "CPU", "device_ids": ["cpu"]},
            ],
            compatibility_revision=Detect_GPU.COMPATIBILITY_REVISION,
            platform="linux",
            os={"id": "rocky", "version_id": "9"},
            devices=[{
                "id": device_id, "name": "NVIDIA H100", "vendor": "NVIDIA",
                "pci_id": "10de:2330", "driver_version": "570.1", "kind": "discrete",
                "compute_capability": "9.0", "eligible_profiles": ["cuda"],
            }],
            reason="test CUDA profile",
        )

    def _rocm_report(self, gfx_target="gfx1100"):
        return detected_report(
            vendor="AMD",
            backend=Install_Dependencies.ROCM_BACKEND,
            gfx_target=gfx_target,
            reason="supported AMD test GPU",
        )

    def _pinned_versions(self):
        """Report the pinned ESM/Transformers versions as installed."""
        return lambda _python, package: (
            Install_Dependencies.TRANSFORMERS_VERSION
            if package == "transformers" else Install_Dependencies.ESM_VERSION
        )

    def test_subprocess_commands_are_not_echoed(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        with mock.patch.object(
            Install_Dependencies.subprocess, "run", return_value=success
        ) as run, redirect_stdout(output):
            completed = Install_Dependencies._run(["uv", "pip", "install", "example"])

        self.assertIs(completed, success)
        run.assert_called_once()
        self.assertEqual(output.getvalue(), "")

    def test_successful_backend_validation_is_silent(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        spec = Install_Dependencies.backend_spec({"backend": "cpu"})
        with mock.patch.object(
            Install_Dependencies, "_run", return_value=success
        ), redirect_stdout(output):
            self.assertTrue(Install_Dependencies.validate_backend(Path("python"), spec))

        self.assertEqual(output.getvalue(), "")

    def test_package_only_validation_does_not_require_a_visible_device(self):
        success = mock.Mock(
            returncode=0,
            stdout=json.dumps({
                "backend": "cuda126", "profile": "cuda126",
                "torch_version": f"{Install_Dependencies.TORCH_VERSION}+cu126",
                "package_error": None,
            }),
            stderr="",
        )
        spec = Install_Dependencies.backend_spec({"backend": "cuda126"})
        with mock.patch.object(Install_Dependencies, "_run", return_value=success) as run:
            validation = Install_Dependencies.validate_backend_package(
                Path("python"), spec
            )
        self.assertEqual(validation["validated_devices"], [])
        self.assertTrue(validation["preserved_without_accelerator"])
        program = run.call_args.args[0][-1]
        self.assertNotIn("torch.cuda.is_available", program)
        self.assertNotIn("torch.ones", program)

    def test_esm_import_smoke_test_forces_utf8_mode(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            Install_Dependencies, "_run", return_value=success
        ) as run:
            self.assertTrue(Install_Dependencies.validate_esm_stack(Path("python")))
        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["-X", "utf8"])

    def test_backend_commands_use_exact_versions_and_indexes(self):
        python = Path(".venv/Scripts/python.exe")
        index_suffixes = {
            "cpu": "/cpu",
            "cuda126": "/cu126",
            "cuda132": "/cu132",
            "xpu": "/xpu",
        }
        for backend, index_suffix in index_suffixes.items():
            with self.subTest(backend=backend):
                spec = Install_Dependencies.backend_spec({"backend": backend})
                command = Install_Dependencies.torch_install_command("uv", python, spec)
                self.assertIn(f"torch=={Install_Dependencies.TORCH_VERSION}", command)
                self.assertTrue(command[-1].endswith(index_suffix))

        mps = Install_Dependencies.backend_spec({"backend": "mps"})
        self.assertNotIn("--index-url", Install_Dependencies.torch_install_command("uv", python, mps))

    def test_rocm_is_one_architecture_specific_profile(self):
        # The per-ROCm-release profiles are gone: a single `rocm` backend
        # installs from AMD's multi-arch channel and selects the device
        # package with the validated GFX target.
        self.assertEqual(
            sorted(
                backend for backend in Install_Dependencies.ACCELERATOR_BACKENDS
                if backend.startswith("rocm")
            ),
            [Install_Dependencies.ROCM_BACKEND],
        )
        spec = Install_Dependencies.backend_spec(
            {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}
        )
        command = Install_Dependencies.torch_install_command("uv", Path("python"), spec)
        self.assertIn(
            f"torch[device-gfx1100]=={Install_Dependencies.ROCM_TORCH_VERSION}", command
        )
        self.assertEqual(command[-1], "https://repo.amd.com/rocm/whl-multi-arch/")
        # The recorded version drops the +rocm local segment so it stays
        # comparable with torch.__version__ during runtime validation.
        self.assertEqual(spec.torch_version, Install_Dependencies.TORCH_VERSION)
        with self.assertRaises(ValueError):
            Install_Dependencies.backend_spec(
                {"backend": Install_Dependencies.ROCM_BACKEND}
            )

    def test_malformed_local_state_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(Install_Dependencies.read_state(path))

    def test_state_distinguishes_requested_backend_from_cpu_fallback(self):
        requested = Install_Dependencies.backend_spec(
            {"backend": Install_Dependencies.ROCM_BACKEND, "gfx_target": "gfx1100"}
        )
        active = Install_Dependencies.backend_spec({"backend": "cpu"})
        profile = Install_Dependencies._state_profile(
            active, ROOT / "src" / "requirements.txt", requested
        )
        self.assertEqual(
            profile["requested_backend"]["backend"], Install_Dependencies.ROCM_BACKEND
        )
        self.assertEqual(profile["requested_backend"]["gfx_target"], "gfx1100")
        self.assertEqual(profile["active_backend"]["backend"], "cpu")

    def test_backend_cleanup_includes_xpu_and_rocm_runtime_packages(self):
        for name in (
            "torch",
            "triton-xpu",
            "intel-opencl-rt",
            "onemkl-sycl-blas",
            "amd-torch-device-gfx1100",
            "rocm-sdk-core",
        ):
            with self.subTest(name=name):
                self.assertTrue(Install_Dependencies._is_backend_package(name))

    def test_esm_runtime_requirements_never_pin_torch_or_transformers(self):
        # esm is installed with --no-deps, so this file is the only place its
        # runtime dependencies are declared. torch comes from the accelerator
        # index and transformers is pinned separately; either one listed here
        # would silently replace the selected build.
        path = Install_Dependencies._esm_runtime_requirements_path(ROOT)
        Install_Dependencies.verify_esm_runtime_requirements(path)
        names = {
            Install_Dependencies._requirement_name(entry)
            for entry in Install_Dependencies._requirements_entries(path)
        }
        self.assertEqual(names & {"torch", "transformers"}, set())

        with tempfile.TemporaryDirectory() as temp_dir:
            rejected = Path(temp_dir) / "rejected.txt"
            for line in ("torch==2.12.0", "Transformers >= 5.17.0", "torch"):
                with self.subTest(line=line):
                    rejected.write_text(f"# comment\neinops\n{line}\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "must not pin"):
                        Install_Dependencies.verify_esm_runtime_requirements(rejected)

            # A distribution whose name merely starts with "torch" is a
            # different package and must not be rejected.
            accepted = Path(temp_dir) / "accepted.txt"
            accepted.write_text("# comment\ntorch_geometric\neinops\n", encoding="utf-8")
            Install_Dependencies.verify_esm_runtime_requirements(accepted)
            self.assertEqual(
                Install_Dependencies._requirements_entries(accepted),
                ("torch_geometric", "einops"),
            )

            with self.assertRaises(FileNotFoundError):
                Install_Dependencies.verify_esm_runtime_requirements(
                    Path(temp_dir) / "missing.txt"
                )

    def test_runtime_requirements_are_declared_for_the_pinned_esm_version(self):
        # esm no longer ships as a bundled wheel, so its Requires-Dist metadata
        # cannot be read back here. The file records the release it was derived
        # from instead, and the installer requires it to be rewritten whenever
        # ESM_VERSION moves; this keeps the two from drifting apart silently.
        header = Install_Dependencies._esm_runtime_requirements_path(ROOT).read_text(
            encoding="utf-8"
        ).splitlines()[0]
        self.assertEqual(
            header,
            f"# ESM runtime dependencies (esm {Install_Dependencies.ESM_VERSION})",
        )

    def test_install_order_uses_transformers_dependencies_then_esm_no_deps(self):
        report = self._cpu_report()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        commands = []
        events = []

        def run(command, **_kwargs):
            commands.append(command)
            events.append(" ".join(str(part) for part in command))
            return success

        def install_backend(*_args):
            events.append("BACKEND")
            return True

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Install_Dependencies, "venv_python", return_value=Path(sys.executable)
        ), mock.patch.object(
            Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
        ), mock.patch.object(
            Install_Dependencies, "_run", side_effect=run
        ), mock.patch.object(
            Install_Dependencies, "install_backend", side_effect=install_backend
        ), mock.patch.object(
            Install_Dependencies, "_installed_version", return_value=None
        ), mock.patch.object(Install_Dependencies, "write_state"):
            code = Install_Dependencies.install(
                project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
            )

        self.assertEqual(code, 0)
        transformers_requirement = (
            f"transformers=={Install_Dependencies.TRANSFORMERS_VERSION}"
        )
        esm_requirement = f"esm=={Install_Dependencies.ESM_VERSION}"
        runtime_requirements = str(ROOT / "src" / "esm_runtime_requirements.txt")
        positions = {
            "base": next(i for i, value in enumerate(events) if str(ROOT / "src" / "requirements.txt") in value),
            "backend": events.index("BACKEND"),
            "transformers": next(i for i, value in enumerate(events) if transformers_requirement in value),
            "runtime": next(i for i, value in enumerate(events) if runtime_requirements in value),
            "esm": next(i for i, value in enumerate(events) if esm_requirement in value),
            "check": next(i for i, value in enumerate(events) if "pip check" in value),
        }
        self.assertEqual(list(positions.values()), sorted(positions.values()))
        transformers_command = next(
            command for command in commands
            if transformers_requirement in " ".join(str(part) for part in command)
        )
        esm_command = next(
            command for command in commands
            if esm_requirement in " ".join(str(part) for part in command)
        )
        # Transformers resolves its own dependencies; esm must not, because it
        # pins an older torch and Transformers than this installer selects.
        self.assertNotIn("--no-deps", transformers_command)
        self.assertIn("--no-deps", esm_command)
        self.assertLess(esm_command.index("--no-deps"), len(esm_command) - 1)
        # The hand-maintained runtime requirements replace what --no-deps skips.
        runtime_command = next(
            command for command in commands
            if runtime_requirements in " ".join(str(part) for part in command)
        )
        self.assertEqual(runtime_command[-2:], ["-r", runtime_requirements])

    def test_dry_run_names_pypi_packages_and_the_runtime_requirements_file(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            Install_Dependencies, "venv_python", return_value=Path(sys.executable)
        ), mock.patch.object(
            Install_Dependencies.Detect_GPU,
            "detect_hardware",
            return_value=self._cpu_report(),
        ), redirect_stdout(output):
            code = Install_Dependencies.install(
                project_root=ROOT,
                venv=Path(temp_dir),
                uv_executable="uv",
                dry_run=True,
            )
        text = output.getvalue()
        self.assertEqual(code, 0)
        self.assertIn(f"transformers=={Install_Dependencies.TRANSFORMERS_VERSION}", text)
        self.assertIn(f"esm=={Install_Dependencies.ESM_VERSION}", text)
        self.assertIn(str(ROOT / "src" / "esm_runtime_requirements.txt"), text)
        self.assertIn("--no-deps", text)
        self.assertIn(f"torch=={Install_Dependencies.TORCH_VERSION}", text)
        # Both packages come from PyPI now: no bundled wheel and no fork.
        self.assertNotIn(".whl", text)
        self.assertNotIn("github.com", text)

    def test_unusable_runtime_requirements_fail_before_any_install_command(self):
        errors = (
            FileNotFoundError("ESM runtime requirements are missing"),
            ValueError("ESM runtime requirements must not pin torch"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp_dir, \
                    mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                    mock.patch.object(
                        Install_Dependencies,
                        "verify_esm_runtime_requirements",
                        side_effect=error,
                    ), \
                    mock.patch.object(Install_Dependencies, "_run") as run:
                with self.assertRaises(type(error)):
                    Install_Dependencies.install(
                        project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
                    )
                run.assert_not_called()

    def test_wrong_installed_pin_makes_the_environment_not_ready(self):
        report = self._cpu_report()
        spec = Install_Dependencies.backend_specs(report)[0]
        stale = {"transformers": "4.57.6", "esm": "3.3.0"}
        for package, version in stale.items():
            with self.subTest(package=package), tempfile.TemporaryDirectory() as temp_dir:
                venv = Path(temp_dir)
                Install_Dependencies.write_state(
                    venv / Install_Dependencies.STATE_FILENAME,
                    Install_Dependencies._state_profile(
                        spec, ROOT / "src" / "requirements.txt"
                    ),
                    report,
                )
                pinned = self._pinned_versions()
                with mock.patch.object(
                    Install_Dependencies, "venv_python", return_value=Path(sys.executable)
                ), mock.patch.object(
                    Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
                ), mock.patch.object(
                    Install_Dependencies,
                    "validate_backend",
                    return_value={"validated_devices": [{"spec": "cpu", "success": True}]},
                ), mock.patch.object(
                    Install_Dependencies,
                    "_installed_version",
                    side_effect=lambda python, name: (
                        version if name == package else pinned(python, name)
                    ),
                ), mock.patch.object(
                    Install_Dependencies, "validate_package_consistency", return_value=True
                ), mock.patch.object(
                    Install_Dependencies, "validate_esm_stack", return_value=True
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    ready = Install_Dependencies.environment_is_ready(
                        project_root=ROOT, venv=venv, uv_executable="uv"
                    )
                self.assertFalse(ready)

    def test_state_schema_versions_hashes_and_runtime_requirements_invalidate(self):
        report = self._cpu_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        state = Install_Dependencies._state_profile(specs[0], requirements)
        state.update({
            "hardware_fingerprint": Install_Dependencies.hardware_fingerprint(report),
            "requested_candidates": Install_Dependencies._spec_payloads(specs),
        })
        fingerprint = Install_Dependencies.hardware_fingerprint(report)
        self.assertTrue(
            Install_Dependencies._state_matches(state, specs, fingerprint, requirements)
        )
        mutations = {
            "schema": Install_Dependencies.STATE_SCHEMA - 1,
            "esm_version": "3.3.0",
            "transformers_version": "4.57.6",
            "requirements_sha256": "0" * 64,
            "esm_runtime_requirements_sha256": "0" * 64,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = dict(state)
                changed[field] = value
                self.assertFalse(
                    Install_Dependencies._state_matches(
                        changed, specs, fingerprint, requirements
                    )
                )

    def test_state_mismatches_name_every_changed_field_in_order(self):
        report = self._cpu_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        fingerprint = Install_Dependencies.hardware_fingerprint(report)
        state = Install_Dependencies._state_profile(specs[0], requirements)
        state.update({
            "hardware_fingerprint": "old-fingerprint",
            "requested_candidates": [],
            "schema": Install_Dependencies.STATE_SCHEMA - 1,
            "requirements_sha256": "old-requirements",
            "esm_version": "old-esm",
        })
        self.assertEqual(
            Install_Dependencies._state_mismatches(
                state, specs, fingerprint, requirements
            ),
            [
                "schema", "hardware_fingerprint", "requirements_sha256",
                "esm_version", "requested_candidates",
            ],
        )

    def test_stale_metadata_reuses_compatible_cuda_without_reinstall(self):
        report = self._cuda_report()
        specs = Install_Dependencies.backend_specs(report)
        requirements = ROOT / "src" / "requirements.txt"
        success = mock.Mock(returncode=0, stdout="", stderr="")
        validation = {"validated_devices": [{"spec": "cuda:0", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            state = Install_Dependencies._state_profile(specs[0], requirements)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME, state, report
            )
            saved = Install_Dependencies.read_state(
                venv / Install_Dependencies.STATE_FILENAME
            )
            saved["schema"] = Install_Dependencies.STATE_SCHEMA - 1
            (venv / Install_Dependencies.STATE_FILENAME).write_text(
                json.dumps(saved), encoding="utf-8"
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=validation
            ) as validate, mock.patch.object(
                Install_Dependencies, "install_backend"
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cuda126")
        install_backend.assert_not_called()

    def test_cpu_only_node_preserves_installed_accelerator_package(self):
        cuda_report = self._cuda_report()
        cpu_report = self._cpu_report()
        cuda_spec = Install_Dependencies.backend_specs(cuda_report)[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            python = venv / "python"
            python.touch()
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    cuda_spec, ROOT / "src" / "requirements.txt"
                ),
                cuda_report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=python
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU,
                "detect_hardware",
                return_value=cpu_report,
            ), mock.patch.object(
                Install_Dependencies,
                "validate_backend_package",
                return_value={"validated_devices": []},
            ) as package_validation, mock.patch.object(
                Install_Dependencies, "validate_backend"
            ) as runtime_validation, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ):
                ready = Install_Dependencies.environment_is_ready(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertTrue(ready)
        package_validation.assert_called_once()
        runtime_validation.assert_not_called()

    def test_refresh_backend_on_cpu_only_node_explicitly_installs_cpu(self):
        cuda_report = self._cuda_report()
        cpu_report = self._cpu_report()
        cuda_spec = Install_Dependencies.backend_specs(cuda_report)[0]
        success = mock.Mock(returncode=0, stdout="", stderr="")
        validation = {"validated_devices": [{"spec": "cpu", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    cuda_spec, ROOT / "src" / "requirements.txt"
                ),
                cuda_report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=cpu_report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=validation
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "validate_backend_package"
            ) as package_validation, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv",
                    refresh_backend=True,
                )

        self.assertEqual(code, 0)
        self.assertEqual(install_backend.call_args.args[2].backend, "cpu")
        package_validation.assert_not_called()

    def test_changed_backend_profile_and_new_accelerator_are_not_reused(self):
        cuda126_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda126"))
        cuda132_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda132"))
        cpu_specs = Install_Dependencies.backend_specs(self._cpu_report())
        cuda_state = {
            "active_backend": Install_Dependencies._spec_payloads((cuda126_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(cuda126_specs),
        }
        cpu_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(cpu_specs),
        }
        self.assertIsNone(
            Install_Dependencies._reusable_backend(cuda_state, cuda132_specs)
        )
        self.assertIsNone(
            Install_Dependencies._reusable_backend(cpu_state, cuda126_specs)
        )

    def test_provisional_accelerator_inventory_preserves_saved_build(self):
        saved_specs = Install_Dependencies.backend_specs(self._cuda_report("cuda132"))
        provisional_report = self._cuda_report("cuda126")
        provisional_report["backend_candidates"][0]["eligibility"] = "provisional"
        current_specs = Install_Dependencies.backend_specs(provisional_report)
        state = {
            "active_backend": Install_Dependencies._spec_payloads((saved_specs[0],))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(saved_specs),
        }
        self.assertFalse(
            Install_Dependencies._accelerator_runtime_visible(provisional_report)
        )
        reusable = Install_Dependencies._reusable_backend(
            state, current_specs, accelerator_visible=False
        )
        self.assertIsNotNone(reusable)
        self.assertEqual(reusable[0].backend, "cuda132")
        self.assertEqual(reusable[1], "package-only")

    def test_provisional_gpu_keeps_cpu_but_eligible_gpu_upgrades_it(self):
        cpu_report = self._cpu_report()
        cpu_spec = Install_Dependencies.backend_specs(cpu_report)[0]
        provisional_report = self._cuda_report("cuda126")
        provisional_report["backend_candidates"][0]["eligibility"] = "provisional"
        provisional_specs = Install_Dependencies.backend_specs(provisional_report)
        cpu_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_spec,))[0],
            "requested_candidates": Install_Dependencies._spec_payloads((cpu_spec,)),
            "detection": cpu_report,
        }
        reusable = Install_Dependencies._reusable_backend(
            cpu_state, provisional_specs, accelerator_visible=False
        )
        self.assertEqual(reusable[0].backend, "cpu")

        provisional_state = {
            "active_backend": Install_Dependencies._spec_payloads((cpu_spec,))[0],
            "requested_candidates": Install_Dependencies._spec_payloads(provisional_specs),
            "detection": provisional_report,
        }
        self.assertIsNone(
            Install_Dependencies._reusable_backend(
                provisional_state, provisional_specs, accelerator_visible=True
            )
        )

    def test_failed_visible_runtime_validation_reinstalls_same_backend(self):
        report = self._cuda_report()
        specs = Install_Dependencies.backend_specs(report)
        success = mock.Mock(returncode=0, stdout="", stderr="")
        repaired = {"validated_devices": [{"spec": "cuda:0", "success": True}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    specs[0], ROOT / "src" / "requirements.txt"
                ),
                report,
            )
            with mock.patch.object(
                Install_Dependencies, "venv_python", return_value=Path(sys.executable)
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=None
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=repaired
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "_installed_version",
                side_effect=self._pinned_versions(),
            ), mock.patch.object(
                Install_Dependencies, "validate_package_consistency", return_value=True
            ), mock.patch.object(
                Install_Dependencies, "validate_esm_stack", return_value=True
            ), mock.patch.object(Install_Dependencies, "write_state"):
                code = Install_Dependencies.install(
                    project_root=ROOT, venv=venv, uv_executable="uv"
                )

        self.assertEqual(code, 0)
        self.assertEqual(install_backend.call_args.args[2].backend, "cuda126")

    def test_failed_accelerator_install_falls_back_to_cpu(self):
        report = self._rocm_report()
        success = mock.Mock(returncode=0, stdout="", stderr="")
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
                mock.patch.object(Install_Dependencies, "_run", return_value=success), \
                mock.patch.object(Install_Dependencies, "install_backend", side_effect=[False, True]) as install_backend, \
                mock.patch.object(Install_Dependencies, "_installed_version", return_value=None), \
                mock.patch.object(Install_Dependencies, "write_state") as write_state, \
                redirect_stdout(output):
            code = Install_Dependencies.install(
                project_root=ROOT,
                venv=Path(temp_dir),
                uv_executable="uv",
            )
        self.assertEqual(code, 0)
        self.assertEqual(
            install_backend.call_args_list[0].args[2].backend,
            Install_Dependencies.ROCM_BACKEND,
        )
        self.assertEqual(install_backend.call_args_list[1].args[2].backend, "cpu")
        self.assertEqual(
            write_state.call_args.args[2]["fallback_from"],
            Install_Dependencies.ROCM_BACKEND,
        )
        status = output.getvalue()
        self.assertNotIn("$ ", status)
        self.assertNotIn("Selected PyTorch backend:", status)
        self.assertNotIn("Validated PyTorch backend:", status)
        self.assertIn("Dependency environment is ready (CPU).", status)

    def test_validated_cpu_fallback_is_reused_for_the_same_hardware(self):
        report = self._rocm_report()
        requested = Install_Dependencies.backend_spec(report)
        active = Install_Dependencies.backend_spec({"backend": "cpu"})
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temp_dir:
            venv = Path(temp_dir)
            Install_Dependencies.write_state(
                venv / Install_Dependencies.STATE_FILENAME,
                Install_Dependencies._state_profile(
                    active, ROOT / "src" / "requirements.txt", requested
                ),
                report,
            )
            with mock.patch.object(
                    Install_Dependencies, "venv_python", return_value=Path(sys.executable)
                ), mock.patch.object(
                    Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
                ), mock.patch.object(
                    Install_Dependencies, "_run", return_value=success
                ), mock.patch.object(
                    Install_Dependencies, "validate_backend", return_value=True
                ) as validate, mock.patch.object(
                    Install_Dependencies, "install_backend"
                ) as install_backend, mock.patch.object(
                    Install_Dependencies, "_installed_version",
                    side_effect=self._pinned_versions(),
                ):
                code = Install_Dependencies.install(
                    project_root=ROOT,
                    venv=venv,
                    uv_executable="uv",
                )
        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cpu")
        install_backend.assert_not_called()


class PackageConsistencyTests(unittest.TestCase):
    """`uv pip check` reports the deviations this installer creates on purpose.

    esm is installed with --no-deps against a newer torch and Transformers than
    it declares, and its runtime requirements omit the ESMFold2-only
    cuequivariance kernels, so a clean install always reports four
    incompatibilities in uv's two shapes: an unsatisfied version and an absent
    dependency. Anything else must fail closed.
    """

    TORCH_LINE = (
        "The package `esm` requires `torch>=2.11.0,<2.12.0`, "
        f"but `{Install_Dependencies.TORCH_VERSION}+cu132` is installed"
    )
    TRANSFORMERS_LINE = (
        "The package `esm` requires `transformers>=4.57.6,<5.0.0`, "
        f"but `{Install_Dependencies.TRANSFORMERS_VERSION}` is installed"
    )
    MISSING_LINES = tuple(
        f"The package `esm` requires `{name}>=0.8.1 ; platform_machine == "
        "'x86_64' and sys_platform == 'linux'`, but it's not installed"
        for name in sorted(Install_Dependencies.ESM_OMITTED_REQUIREMENTS)
    )

    def _check(self, *lines):
        report = "\n".join(
            ("Checked 155 packages in 1ms", f"Found {len(lines)} incompatibilities", *lines)
        )
        completed = mock.Mock(returncode=1, stdout=report, stderr="")
        stdout = io.StringIO()
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            return Install_Dependencies.validate_package_consistency("uv", Path("python"))

    def _expected(self):
        return (self.TORCH_LINE, self.TRANSFORMERS_LINE, *self.MISSING_LINES)

    def test_clean_report_passes(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed):
            self.assertTrue(
                Install_Dependencies.validate_package_consistency("uv", Path("python"))
            )

    def test_sanctioned_version_and_missing_deviations_pass(self):
        self.assertTrue(self._check(*self._expected()))

    def test_unsanctioned_missing_dependency_fails(self):
        lines = (
            self.TORCH_LINE,
            self.TRANSFORMERS_LINE,
            "The package `esm` requires `biotite>=1.0.0`, but it's not installed",
        )
        self.assertFalse(self._check(*lines))

    def test_missing_dependency_of_another_package_fails(self):
        lines = (
            self.TORCH_LINE,
            self.TRANSFORMERS_LINE,
            "The package `scanpy` requires `cuequivariance-torch>=0.8.1`, "
            "but it's not installed",
        )
        self.assertFalse(self._check(*lines))

    def test_unpinned_torch_or_transformers_version_fails(self):
        for line in (
            self.TORCH_LINE.replace(Install_Dependencies.TORCH_VERSION, "2.9.0"),
            self.TRANSFORMERS_LINE.replace(
                Install_Dependencies.TRANSFORMERS_VERSION, "4.57.6"
            ),
        ):
            with self.subTest(line=line):
                self.assertFalse(self._check(line))

    def test_unaccounted_or_unparseable_report_fails(self):
        # The declared count is the safety net: a line uv words differently, or
        # a report shape this installer cannot parse, must not pass silently.
        completed = mock.Mock(
            returncode=1,
            stdout="Found 5 incompatibilities\n" + "\n".join(self._expected()),
            stderr="",
        )
        with mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertFalse(
                Install_Dependencies.validate_package_consistency("uv", Path("python"))
            )
        self.assertFalse(self._check("something uv started wording differently"))

    def test_omitted_requirements_are_absent_from_the_runtime_file(self):
        # The sanctioned-missing set only holds while the file really omits
        # them; listing one there would install it and silence the report.
        entries = Install_Dependencies._requirements_entries(
            Install_Dependencies._esm_runtime_requirements_path(ROOT)
        )
        names = {Install_Dependencies._requirement_name(entry) for entry in entries}
        self.assertEqual(
            names & Install_Dependencies.ESM_OMITTED_REQUIREMENTS, set()
        )


if __name__ == "__main__":
    unittest.main()
