# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
import hashlib
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
                mock.patch.object(Detect_GPU, "_windows_amd_software_version", return_value="26.2.2"), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=list(nvidia)):
            return Detect_GPU.detect_hardware()

    def test_windows_amd_models_map_to_official_gfx_targets(self):
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
                self.assertEqual(report["backend"], "rocm714")
                self.assertEqual(report["gfx_target"], target)

    def test_unknown_windows_amd_uses_intel_if_available_otherwise_cpu(self):
        mixed = self._detect(controllers=["AMD Radeon Graphics", "Intel Arc A770"])
        self.assertEqual(mixed["backend"], "xpu")
        amd_only = self._detect(controllers=["AMD Radeon Graphics"])
        self.assertEqual(amd_only["backend"], "cpu")

    def test_windows_10_rejects_rocm(self):
        report = self._detect(version="10.0.19045", controllers=["AMD Radeon RX 7900 XTX"])
        self.assertEqual(report["backend"], "cpu")
        self.assertIn("Windows 11", report["reason"])

    def test_older_windows_11_routes_supported_amd_to_rocm_721(self):
        report = self._detect(
            version="10.0.26100", controllers=["AMD Radeon RX 7900 XTX"]
        )
        self.assertEqual(report["backend"], "rocm721")
        self.assertEqual(report["gfx_target"], "gfx1100")
        self.assertEqual(
            [candidate["backend"] for candidate in report["backend_candidates"]],
            ["rocm721", "cpu"],
        )

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
        with mock.patch.object(
                Detect_GPU, "_read_os_release", return_value={"id": "ubuntu", "version_id": "24.04"}
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
        self.assertEqual(linux["backend"], "rocm72")
        self.assertEqual(
            [candidate["backend"] for candidate in linux["backend_candidates"]],
            ["rocm72", "rocm64", "cpu"],
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
                "torch_version": "2.12.1+cu126", "package_error": None,
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
        cases = {
            "cpu": ("torch==2.12.1", "/cpu"),
            "cuda126": ("torch==2.12.1", "/cu126"),
            "cuda132": ("torch==2.12.1", "/cu132"),
            "xpu": ("torch==2.12.1", "/xpu"),
            "rocm72": ("torch==2.12.1", "/rocm7.2"),
            "rocm64": ("torch==2.9.1", "/rocm6.4"),
        }
        for backend, (requirement, index_suffix) in cases.items():
            with self.subTest(backend=backend):
                spec = Install_Dependencies.backend_spec({"backend": backend})
                command = Install_Dependencies.torch_install_command("uv", python, spec)
                self.assertIn(requirement, command)
                self.assertTrue(command[-1].endswith(index_suffix))

        mps = Install_Dependencies.backend_spec({"backend": "mps"})
        self.assertNotIn("--index-url", Install_Dependencies.torch_install_command("uv", python, mps))

    def test_windows_rocm_command_is_architecture_specific(self):
        spec = Install_Dependencies.backend_spec(
            {"backend": "rocm714", "gfx_target": "gfx1100"}
        )
        command = Install_Dependencies.torch_install_command("uv", Path("python"), spec)
        self.assertIn("torch[device-gfx1100]==2.12.0+rocm7.14.0", command)
        self.assertEqual(command[-1], "https://repo.amd.com/rocm/whl-multi-arch/")

    def test_malformed_local_state_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state.json"
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(Install_Dependencies.read_state(path))

    def test_state_distinguishes_requested_backend_from_cpu_fallback(self):
        requested = Install_Dependencies.backend_spec(
            {"backend": "rocm714", "gfx_target": "gfx1100"}
        )
        active = Install_Dependencies.backend_spec({"backend": "cpu"})
        profile = Install_Dependencies._state_profile(
            active, ROOT / "src" / "requirements.txt", requested
        )
        self.assertEqual(profile["requested_backend"]["backend"], "rocm714")
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

    def test_bundled_wheels_match_manifest_hashes_metadata_licenses_and_modules(self):
        wheels = ROOT / "src" / "resources" / "wheels"
        manifest = json.loads((wheels / "manifest.json").read_text(encoding="utf-8"))
        artifacts = {item["package"]: item for item in manifest["artifacts"]}
        expected = {
            "esm": (
                "esm-3.3.0-py3-none-any.whl",
                Install_Dependencies.ESM_VERSION,
                Install_Dependencies.ESM_WHEEL_SHA256,
                "MIT",
            ),
            "transformers": (
                "transformers-4.57.6+biohub.3a8956f-py3-none-any.whl",
                Install_Dependencies.TRANSFORMERS_VERSION,
                Install_Dependencies.TRANSFORMERS_WHEEL_SHA256,
                "Apache-2.0",
            ),
        }
        for package, (filename, version, sha256, license_name) in expected.items():
            with self.subTest(package=package):
                artifact = artifacts[package]
                wheel = wheels / filename
                self.assertEqual(artifact["filename"], filename)
                self.assertEqual(artifact["version"], version)
                self.assertEqual(artifact["sha256"], sha256)
                self.assertEqual(artifact["license"], license_name)
                self.assertEqual(artifact["size"], wheel.stat().st_size)
                self.assertEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), sha256)
                license_text = (wheels / artifact["license_file"]).read_text(
                    encoding="utf-8"
                )
                self.assertIn("permission", license_text.lower())

        esm_metadata = Install_Dependencies.verify_esm_wheel(
            wheels / expected["esm"][0]
        )
        transformers_metadata = Install_Dependencies.verify_transformers_wheel(
            wheels / expected["transformers"][0]
        )
        self.assertEqual(esm_metadata["Version"], Install_Dependencies.ESM_VERSION)
        self.assertEqual(
            transformers_metadata["Version"], Install_Dependencies.TRANSFORMERS_VERSION
        )
        self.assertEqual(transformers_metadata["License"], "Apache 2.0 License")
        self.assertEqual(artifacts["transformers"]["upstream_version"], "4.57.6")
        self.assertEqual(
            artifacts["transformers"]["source_commit"],
            "3a8956fb4d4ea16b0ec8e71deef2c2909b6a5cbf",
        )

    def test_esm_runtime_requirements_exactly_follow_verified_wheel_metadata(self):
        esm_wheel, _, requirements = Install_Dependencies._bundled_paths(ROOT)
        metadata = Install_Dependencies.verify_esm_wheel(esm_wheel)
        all_requirements = tuple(metadata.get_all("Requires-Dist", []))
        expected = tuple(
            value
            for value in all_requirements
            if Install_Dependencies._requirement_name(value)
            not in {"torch", "transformers"}
        )
        actual = Install_Dependencies._requirements_entries(requirements)
        self.assertEqual(actual, expected)
        excluded = {
            Install_Dependencies._requirement_name(value)
            for value in all_requirements
            if value not in actual
        }
        self.assertEqual(excluded, {"torch", "transformers"})
        Install_Dependencies.verify_esm_runtime_requirements(esm_wheel, requirements)

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
            Install_Dependencies, "_installed_version", return_value="4.57.6"
        ), mock.patch.object(Install_Dependencies, "write_state"):
            code = Install_Dependencies.install(
                project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
            )

        self.assertEqual(code, 0)
        joined = [" ".join(str(part) for part in command) for command in commands]
        positions = {
            "base": next(i for i, value in enumerate(events) if str(ROOT / "src" / "requirements.txt") in value),
            "backend": events.index("BACKEND"),
            "transformers": next(i for i, value in enumerate(events) if "transformers-4.57.6+biohub" in value),
            "runtime": next(i for i, value in enumerate(events) if "esm-3.3.0-runtime-requirements.txt" in value),
            "esm": next(i for i, value in enumerate(events) if "esm-3.3.0-py3-none-any.whl" in value),
            "check": next(i for i, value in enumerate(events) if "pip check" in value),
        }
        self.assertEqual(list(positions.values()), sorted(positions.values()))
        transformers_command = next(
            command for command in commands
            if "transformers-4.57.6+biohub" in " ".join(str(part) for part in command)
        )
        esm_command = next(
            command for command in commands
            if "esm-3.3.0-py3-none-any.whl" in " ".join(str(part) for part in command)
        )
        self.assertNotIn("--no-deps", transformers_command)
        self.assertIn("--no-deps", esm_command)
        self.assertLess(esm_command.index("--no-deps"), len(esm_command) - 1)

    def test_dry_run_names_local_wheels_without_biohub_git_url(self):
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
        self.assertIn("transformers-4.57.6+biohub.3a8956f-py3-none-any.whl", text)
        self.assertIn("esm-3.3.0-py3-none-any.whl", text)
        self.assertIn("--no-deps", text)
        self.assertNotIn("github.com/Biohub/transformers", text)

    def test_missing_or_corrupt_bundle_fails_before_any_install_command(self):
        errors = (
            FileNotFoundError("Bundled transformers wheel is missing"),
            ValueError("Bundled transformers wheel checksum mismatch"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp_dir, \
                    mock.patch.object(Install_Dependencies, "venv_python", return_value=Path(sys.executable)), \
                    mock.patch.object(Install_Dependencies, "verify_bundled_artifacts", side_effect=error), \
                    mock.patch.object(Install_Dependencies, "_run") as run:
                with self.assertRaises(type(error)):
                    Install_Dependencies.install(
                        project_root=ROOT, venv=Path(temp_dir), uv_executable="uv"
                    )
                run.assert_not_called()

    def test_wrong_transformers_metadata_version_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wheel = Path(temp_dir) / "transformers.whl"
            wheel.touch()
            members = {
                "transformers/__init__.py",
                "transformers/models/esmc/configuration_esmc.py",
                "transformers/models/esmfold2/configuration_esmfold2.py",
            }
            metadata = {"Name": "transformers", "Version": "4.57.6"}
            with mock.patch.object(
                Install_Dependencies,
                "_sha256",
                return_value=Install_Dependencies.TRANSFORMERS_WHEEL_SHA256,
            ), mock.patch.object(
                Install_Dependencies, "_wheel_metadata", return_value=(metadata, members)
            ):
                with self.assertRaisesRegex(ValueError, "reports version"):
                    Install_Dependencies.verify_transformers_wheel(wheel)

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
            "schema": 3,
            "transformers_version": "4.57.6",
            "transformers_wheel_sha256": "0" * 64,
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
            "esm_wheel_sha256": "old-esm-wheel",
        })
        self.assertEqual(
            Install_Dependencies._state_mismatches(
                state, specs, fingerprint, requirements
            ),
            [
                "schema", "hardware_fingerprint", "requirements_sha256",
                "esm_wheel_sha256", "requested_candidates",
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
                Install_Dependencies, "verify_bundled_artifacts"
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=validation
            ) as validate, mock.patch.object(
                Install_Dependencies, "install_backend"
            ) as install_backend, mock.patch.object(
                Install_Dependencies,
                "_installed_version",
                side_effect=lambda _python, package: (
                    Install_Dependencies.TRANSFORMERS_VERSION
                    if package == "transformers" else Install_Dependencies.ESM_VERSION
                ),
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
                Install_Dependencies, "verify_bundled_artifacts"
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
                Install_Dependencies,
                "_installed_version",
                side_effect=lambda _python, package: (
                    Install_Dependencies.TRANSFORMERS_VERSION
                    if package == "transformers" else Install_Dependencies.ESM_VERSION
                ),
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
                Install_Dependencies, "verify_bundled_artifacts"
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=cpu_report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=validation
            ) as install_backend, mock.patch.object(
                Install_Dependencies, "validate_backend_package"
            ) as package_validation, mock.patch.object(
                Install_Dependencies,
                "_installed_version",
                side_effect=lambda _python, package: (
                    Install_Dependencies.TRANSFORMERS_VERSION
                    if package == "transformers" else Install_Dependencies.ESM_VERSION
                ),
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
                Install_Dependencies, "verify_bundled_artifacts"
            ), mock.patch.object(
                Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report
            ), mock.patch.object(
                Install_Dependencies, "_run", return_value=success
            ), mock.patch.object(
                Install_Dependencies, "validate_backend", return_value=None
            ), mock.patch.object(
                Install_Dependencies, "install_backend", return_value=repaired
            ) as install_backend, mock.patch.object(
                Install_Dependencies,
                "_installed_version",
                side_effect=lambda _python, package: (
                    Install_Dependencies.TRANSFORMERS_VERSION
                    if package == "transformers" else Install_Dependencies.ESM_VERSION
                ),
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
        report = detected_report(
            vendor="AMD",
            backend="rocm714",
            gfx_target="gfx1100",
            reason="supported AMD test GPU",
        )
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
        self.assertEqual(install_backend.call_args_list[0].args[2].backend, "rocm714")
        self.assertEqual(install_backend.call_args_list[1].args[2].backend, "cpu")
        self.assertEqual(write_state.call_args.args[2]["fallback_from"], "rocm714")
        status = output.getvalue()
        self.assertNotIn("$ ", status)
        self.assertNotIn("Selected PyTorch backend:", status)
        self.assertNotIn("Validated PyTorch backend:", status)
        self.assertIn("Dependency environment is ready (CPU).", status)

    def test_validated_cpu_fallback_is_reused_for_the_same_hardware(self):
        report = detected_report(
            vendor="AMD",
            backend="rocm714",
            gfx_target="gfx1100",
            reason="supported AMD test GPU",
        )
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
                    Install_Dependencies, "_installed_version", return_value=Install_Dependencies.ESM_VERSION
                ):
                code = Install_Dependencies.install(
                    project_root=ROOT,
                    venv=venv,
                    uv_executable="uv",
                )
        self.assertEqual(code, 0)
        self.assertEqual(validate.call_args.args[1].backend, "cpu")
        install_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
