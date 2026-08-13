# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from contextlib import redirect_stdout
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

    def test_backend_commands_use_exact_versions_and_indexes(self):
        python = Path(".venv/Scripts/python.exe")
        cases = {
            "cpu": ("torch==2.12.1", "/cpu"),
            "cuda126": ("torch==2.12.1", "/cu126"),
            "cuda132": ("torch==2.12.1", "/cu132"),
            "xpu": ("torch==2.12.1", "/xpu"),
            "rocm72": ("torch==2.12.1", "/rocm7.2"),
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

    def test_esm_wheel_matches_declared_hash(self):
        wheel = ROOT / "src" / "resources" / "wheels" / "esm-3.3.0-py3-none-any.whl"
        Install_Dependencies.verify_esm_wheel(wheel)

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
