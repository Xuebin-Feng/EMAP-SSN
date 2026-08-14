from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "utilities"))

import Detect_GPU
import Install_Dependencies
import Hardware_Utils


def gpu(
    name: str,
    vendor: str,
    *,
    identifier: str,
    kind: str = "unknown",
    architecture: str | None = None,
    driver: str | None = "1.0",
    profiles: list[str] | None = None,
    eligibility: str = "eligible",
    capability: str | None = None,
) -> dict:
    return {
        "id": identifier,
        "name": name,
        "vendor": vendor,
        "pci_id": None,
        "driver_version": driver,
        "kind": kind,
        "architecture": architecture,
        "eligible_profiles": list(profiles or []),
        "eligibility": eligibility,
        "reasons": [],
        "compute_capability": capability,
    }


class DetectionCompatibilityTests(unittest.TestCase):
    def test_nvidia_merge_falls_back_to_bracketed_model_name_without_bus_id(self):
        devices = [
            gpu(
                "NVIDIA Corporation AD104 [GeForce RTX 4070]",
                "NVIDIA",
                identifier="0000:01:00.0",
                kind="discrete",
                driver=None,
            )
        ]
        nvidia = [
            {
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": None,
                "driver_version": "595.84",
            }
        ]

        Detect_GPU._merge_nvidia_inventory(devices, nvidia)

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["driver_version"], "595.84")

    def test_nvidia_merge_uses_name_when_os_inventory_has_no_pci_address(self):
        devices = [
            gpu(
                "NVIDIA GeForce RTX 4070",
                "NVIDIA",
                identifier=r"PCI\VEN_10DE&DEV_2786",
                driver=None,
            )
        ]
        nvidia = [
            {
                "bus_id": "00000000:01:00.0",
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": "8.9",
                "driver_version": "595.84",
            }
        ]

        Detect_GPU._merge_nvidia_inventory(devices, nvidia)

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["compute_capability"], "8.9")
        self.assertEqual(devices[0]["driver_version"], "595.84")

    def test_nvidia_merge_uses_addresses_for_same_model_multi_gpu_inventory(self):
        devices = [
            gpu("NVIDIA Corporation AD104 [GeForce RTX 4070]", "NVIDIA", identifier=address, driver=None)
            for address in ("0000:01:00.0", "0000:02:00.0")
        ]
        nvidia = [
            {
                "bus_id": address,
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": "8.9",
                "driver_version": driver,
            }
            for address, driver in (
                ("00000000:02:00.0", "595.82"),
                ("00000000:01:00.0", "595.81"),
            )
        ]

        Detect_GPU._merge_nvidia_inventory(devices, nvidia)

        self.assertEqual(len(devices), 2)
        self.assertEqual([device["driver_version"] for device in devices], ["595.81", "595.82"])

    def test_nvidia_merge_does_not_join_disagreeing_addresses_by_name(self):
        devices = [
            gpu(
                "NVIDIA Corporation AD104 [GeForce RTX 4070]",
                "NVIDIA",
                identifier="0000:01:00.0",
                driver=None,
            )
        ]
        nvidia = [
            {
                "bus_id": "00000000:02:00.0",
                "name": "NVIDIA GeForce RTX 4070",
                "compute_capability": "8.9",
                "driver_version": "595.84",
            }
        ]

        Detect_GPU._merge_nvidia_inventory(devices, nvidia)

        self.assertEqual(len(devices), 2)
        self.assertIsNone(devices[0]["driver_version"])
        self.assertEqual(devices[1]["id"], "00000000:02:00.0")

    def test_windows_25h2_amd_gets_two_profiles(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", kind="discrete", architecture="gfx1100")
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligible_profiles"], ["rocm714", "rocm721"])
        self.assertEqual(device["eligibility"], "eligible")

    def test_older_windows_11_uses_rocm_721_only(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", kind="discrete", architecture="gfx1100")
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26100}, set())
        self.assertEqual(device["eligible_profiles"], ["rocm721"])

    def test_windows_10_rejects_native_rocm(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100")
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 19045}, set())
        self.assertEqual(device["eligible_profiles"], [])
        self.assertEqual(device["eligibility"], "ineligible")

    def test_unknown_amd_is_never_provisional(self):
        device = gpu("AMD Radeon Graphics", "AMD", identifier="amd0", architecture=None, driver=None)
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligibility"], "ineligible")

    def test_known_amd_without_driver_is_provisional(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100", driver=None)
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligibility"], "provisional")

    def test_rocm_721_applies_amd_software_version_predicate(self):
        old = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd-old", architecture="gfx1100")
        supported = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd-new", architecture="gfx1100")
        Detect_GPU._evaluate_devices(
            [old], "windows", {"windows_build": 26100, "amd_software_version": "26.2.1"}, set()
        )
        Detect_GPU._evaluate_devices(
            [supported], "windows", {"windows_build": 26100, "amd_software_version": "26.2.2"}, set()
        )
        self.assertEqual(old["eligible_profiles"], [])
        self.assertEqual(supported["eligible_profiles"], ["rocm721"])
        self.assertEqual(supported["profile_eligibility"]["rocm721"], "eligible")

    def test_intel_arc_is_xpu_but_uhd_is_not(self):
        arc = gpu("Intel(R) Arc(TM) A770 Graphics", "INTEL", identifier="intel0")
        uhd = gpu("Intel(R) UHD Graphics 770", "INTEL", identifier="intel1")
        Detect_GPU._evaluate_devices([arc, uhd], "windows", {"windows_build": 26200}, set())
        self.assertEqual(arc["eligible_profiles"], ["xpu"])
        self.assertEqual(uhd["eligible_profiles"], [])

    def test_amd_discrete_target_excludes_integrated_target(self):
        discrete = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd-d", kind="discrete", architecture="gfx1100", profiles=["rocm714", "rocm721"])
        integrated = gpu("AMD Radeon 890M", "AMD", identifier="amd-i", kind="integrated", architecture="gfx1150", profiles=["rocm714", "rocm721"])
        candidates, ignored = Detect_GPU._candidate_ladder([integrated, discrete])
        self.assertEqual(candidates[0]["device_ids"], ["amd-d"])
        self.assertEqual([item["backend"] for item in candidates], ["rocm714", "rocm721", "cpu"])
        self.assertEqual(ignored[0]["id"], "amd-i")

    def test_discrete_intel_precedes_integrated_amd(self):
        amd = gpu("AMD Radeon 890M", "AMD", identifier="amd-i", kind="integrated", architecture="gfx1150", profiles=["rocm714"])
        intel = gpu("Intel Arc B580", "INTEL", identifier="intel-d", kind="discrete", profiles=["xpu"])
        candidates, _ignored = Detect_GPU._candidate_ladder([amd, intel])
        self.assertEqual(candidates[0]["backend"], "xpu")

    def test_nvidia_precedes_integrated_intel_and_uses_common_cuda(self):
        new = gpu("NVIDIA RTX 5090", "NVIDIA", identifier="n0", kind="discrete", profiles=["cuda"], capability="12.0", driver="590.0")
        old = gpu("NVIDIA RTX 2080", "NVIDIA", identifier="n1", kind="discrete", profiles=["cuda"], capability="7.5", driver="590.0")
        intel = gpu("Intel Arc Graphics", "INTEL", identifier="i0", kind="integrated", profiles=["xpu"])
        candidates, _ignored = Detect_GPU._candidate_ladder([intel, new, old])
        self.assertEqual(candidates[0]["backend"], "cuda132")
        self.assertEqual(set(candidates[0]["device_ids"]), {"n0", "n1"})

    def test_linux_rocm_agent_mismatch_is_rejected(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100")
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=True), mock.patch.object(Detect_GPU.os, "access", return_value=True):
            Detect_GPU._evaluate_devices(
                [device], "linux", {"id": "ubuntu", "version_id": "24.04"}, {"gfx1200"}
            )
        self.assertEqual(device["eligible_profiles"], [])

    def test_linux_rocm_rejects_unsupported_distro_and_inaccessible_kfd(self):
        unsupported_distro = gpu(
            "AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100"
        )
        inaccessible_kfd = gpu(
            "AMD Radeon RX 7900 XTX", "AMD", identifier="amd1", architecture="gfx1100"
        )
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            Detect_GPU._evaluate_devices(
                [unsupported_distro],
                "linux",
                {"id": "debian", "version_id": "12"},
                {"gfx1100"},
            )
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=False), \
                mock.patch.object(Detect_GPU.os, "access", return_value=False):
            Detect_GPU._evaluate_devices(
                [inaccessible_kfd],
                "linux",
                {"id": "ubuntu", "version_id": "24.04"},
                {"gfx1100"},
            )

        self.assertEqual(unsupported_distro["eligible_profiles"], [])
        self.assertIn("Ubuntu", unsupported_distro["reasons"][0])
        self.assertEqual(inaccessible_kfd["eligible_profiles"], [])
        self.assertIn("/dev/kfd", inaccessible_kfd["reasons"][0])

    def test_linux_rocm_profiles_prefer_72_and_limit_64_targets(self):
        supported = gpu(
            "AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100"
        )
        newer_only = gpu(
            "AMD Radeon 890M", "AMD", identifier="amd1", architecture="gfx1150"
        )
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            Detect_GPU._evaluate_devices(
                [supported, newer_only],
                "linux",
                {"id": "ubuntu", "version_id": "24.04"},
                {"gfx1100", "gfx1150"},
            )

        self.assertEqual(supported["eligible_profiles"], ["rocm72", "rocm64"])
        self.assertEqual(newer_only["eligible_profiles"], ["rocm72"])

        candidates, _ignored = Detect_GPU._candidate_ladder([supported])
        self.assertEqual(
            [candidate["backend"] for candidate in candidates],
            ["rocm72", "rocm64", "cpu"],
        )


class InstallerProfileTests(unittest.TestCase):
    def test_linux_rocm_profiles_have_distinct_versions_and_indexes(self):
        report = {
            "backend_candidates": [
                {"backend": "rocm72", "profile": "rocm72", "gfx_target": "gfx1100", "device_ids": ["amd0"]},
                {"backend": "rocm64", "profile": "rocm64", "gfx_target": "gfx1100", "device_ids": ["amd0"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ]
        }
        specs = Install_Dependencies.backend_specs(report)
        self.assertEqual([spec.torch_version for spec in specs], ["2.12.1", "2.9.1", "2.12.1"])
        self.assertTrue(specs[0].install_steps[0].index_url.endswith("/rocm7.2"))
        self.assertTrue(specs[1].install_steps[0].index_url.endswith("/rocm6.4"))

    def test_windows_rocm_profiles_have_distinct_steps_and_versions(self):
        report = {
            "backend_candidates": [
                {"backend": "rocm714", "profile": "rocm714", "gfx_target": "gfx1100", "device_ids": ["amd0"]},
                {"backend": "rocm721", "profile": "rocm721", "gfx_target": "gfx1100", "device_ids": ["amd0"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ]
        }
        specs = Install_Dependencies.backend_specs(report)
        self.assertEqual([spec.torch_version for spec in specs], ["2.12.0", "2.9.1", "2.12.1"])
        self.assertEqual(len(specs[0].install_steps), 1)
        self.assertEqual(len(specs[1].install_steps), 2)
        self.assertIn("rocm_sdk_core", specs[1].install_steps[0].requirements[0])

    def test_hardware_fingerprint_changes_with_driver(self):
        report = {"compatibility_revision": 3, "platform": "windows", "os": {}, "devices": [{"id": "0", "driver_version": "1"}], "backend_candidates": []}
        first = Install_Dependencies.hardware_fingerprint(report)
        report["devices"][0]["driver_version"] = "2"
        self.assertNotEqual(first, Install_Dependencies.hardware_fingerprint(report))

    def test_state_schema_two_is_invalidated(self):
        requirements = ROOT / "src" / "requirements.txt"
        specs = Install_Dependencies.backend_specs({"backend_candidates": [{"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]}]})
        self.assertFalse(Install_Dependencies._state_matches({"schema": 2}, specs, "fingerprint", requirements))

    def test_failed_cuda_falls_through_to_xpu(self):
        report = {
            "compatibility_revision": 3,
            "platform": "windows",
            "os": {"windows_build": 26200},
            "devices": [],
            "ignored_devices": [],
            "reason": "test ladder",
            "backend_candidates": [
                {"backend": "cuda126", "profile": "cuda126", "device_ids": ["n0"]},
                {"backend": "xpu", "profile": "xpu", "device_ids": ["i0"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        validation = {"validated_devices": [{"spec": "xpu:0", "success": True}]}
        completed = subprocess.CompletedProcess([], 0, "", "")
        written: dict = {}

        def capture_state(_path, payload, _report=None):
            written.update(payload)

        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies, "verify_esm_wheel"), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
            mock.patch.object(Install_Dependencies, "install_backend", side_effect=[None, validation]) as install_backend, \
            mock.patch.object(Install_Dependencies, "write_state", side_effect=capture_state):
            result = Install_Dependencies.install(
                project_root=ROOT, venv=Path(folder), uv_executable="uv"
            )
        self.assertEqual(result, 0)
        self.assertEqual([call.args[2].backend for call in install_backend.call_args_list], ["cuda126", "xpu"])
        self.assertEqual(written["active_backend"]["backend"], "xpu")

    def test_failed_rocm_72_falls_through_to_rocm_64(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "platform": "linux",
            "os": {"id": "ubuntu", "version_id": "24.04"},
            "devices": [],
            "ignored_devices": [],
            "reason": "test ladder",
            "backend_candidates": [
                {"backend": "rocm72", "profile": "rocm72", "gfx_target": "gfx1100", "device_ids": ["a0"]},
                {"backend": "rocm64", "profile": "rocm64", "gfx_target": "gfx1100", "device_ids": ["a0"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        validation = {"validated_devices": [{"spec": "cuda:0", "architecture": "gfx1100", "success": True}]}
        completed = subprocess.CompletedProcess([], 0, "", "")
        written: dict = {}

        def capture_state(_path, payload, _report=None):
            written.update(payload)

        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies, "verify_esm_wheel"), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch.object(Install_Dependencies, "_run", return_value=completed), \
            mock.patch.object(Install_Dependencies, "install_backend", side_effect=[None, validation]) as install_backend, \
            mock.patch.object(Install_Dependencies, "write_state", side_effect=capture_state):
            result = Install_Dependencies.install(
                project_root=ROOT, venv=Path(folder), uv_executable="uv"
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.args[2].backend for call in install_backend.call_args_list],
            ["rocm72", "rocm64"],
        )
        self.assertEqual(written["active_backend"]["backend"], "rocm64")

    def test_previous_compatibility_revision_is_invalidated(self):
        requirements = ROOT / "src" / "requirements.txt"
        specs = Install_Dependencies.backend_specs(
            {"backend_candidates": [{"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]}]}
        )
        stale = {
            "schema": Install_Dependencies.STATE_SCHEMA,
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION - 1,
            "hardware_fingerprint": "fingerprint",
            "requirements_sha256": Install_Dependencies._sha256(requirements),
            "esm_version": Install_Dependencies.ESM_VERSION,
            "esm_wheel_sha256": Install_Dependencies.ESM_WHEEL_SHA256,
            "requested_candidates": Install_Dependencies._spec_payloads(specs),
        }
        self.assertFalse(
            Install_Dependencies._state_matches(stale, specs, "fingerprint", requirements)
        )

    def test_dry_run_prints_every_candidate(self):
        report = {
            "compatibility_revision": 3, "platform": "windows", "os": {}, "devices": [],
            "ignored_devices": [], "reason": "test",
            "backend_candidates": [
                {"backend": "rocm714", "profile": "rocm714", "gfx_target": "gfx1100", "device_ids": ["a"]},
                {"backend": "rocm721", "profile": "rocm721", "gfx_target": "gfx1100", "device_ids": ["a"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies, "verify_esm_wheel"), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch("builtins.print") as printer:
            result = Install_Dependencies.install(project_root=ROOT, venv=Path(folder), uv_executable="uv", dry_run=True)
        output = "\n".join(" ".join(str(value) for value in call.args) for call in printer.call_args_list)
        self.assertEqual(result, 0)
        self.assertIn("Windows ROCm 7.14", output)
        self.assertIn("Windows ROCm 7.2.1", output)
        self.assertIn("CPU", output)

    def test_linux_rocm_dry_run_prints_72_and_64_candidates(self):
        report = {
            "compatibility_revision": Detect_GPU.COMPATIBILITY_REVISION,
            "platform": "linux",
            "os": {"id": "ubuntu", "version_id": "24.04"},
            "devices": [],
            "ignored_devices": [],
            "reason": "test",
            "backend_candidates": [
                {"backend": "rocm72", "profile": "rocm72", "gfx_target": "gfx1100", "device_ids": ["a"]},
                {"backend": "rocm64", "profile": "rocm64", "gfx_target": "gfx1100", "device_ids": ["a"]},
                {"backend": "cpu", "profile": "cpu", "device_ids": ["cpu"]},
            ],
        }
        with tempfile.TemporaryDirectory() as folder, \
            mock.patch.object(Install_Dependencies, "venv_python", return_value=Path("python")), \
            mock.patch.object(Install_Dependencies, "verify_esm_wheel"), \
            mock.patch.object(Install_Dependencies.Detect_GPU, "detect_hardware", return_value=report), \
            mock.patch("builtins.print") as printer:
            result = Install_Dependencies.install(
                project_root=ROOT,
                venv=Path(folder),
                uv_executable="uv",
                dry_run=True,
            )

        output = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in printer.call_args_list
        )
        self.assertEqual(result, 0)
        self.assertIn("Linux ROCm 7.2", output)
        self.assertIn("Linux ROCm 6.4", output)
        self.assertIn("CPU", output)


class RuntimeFilteringTests(unittest.TestCase):
    def test_validated_state_is_read_from_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "ssn_backend.json").write_text(
                json.dumps({"schema": 3, "validated_devices": [{"spec": "cuda:1", "success": True}]}),
                encoding="utf-8",
            )
            with mock.patch.object(Hardware_Utils.sys, "prefix", folder):
                self.assertEqual(Hardware_Utils._validated_device_specs(), {"cuda:1"})

    def test_unvalidated_visible_gpu_is_filtered(self):
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            get_device_name=lambda index: f"GPU {index}",
        )
        fake_mps = types.SimpleNamespace(is_available=lambda: False)
        fake_torch = types.SimpleNamespace(
            cuda=fake_cuda,
            version=types.SimpleNamespace(hip=None),
            backends=types.SimpleNamespace(mps=fake_mps),
            device=lambda value: value,
        )
        with mock.patch.object(Hardware_Utils, "torch", fake_torch), mock.patch.object(
            Hardware_Utils, "_validated_device_specs", return_value={"cuda:1"}
        ):
            candidates = Hardware_Utils.get_available_devices()
        self.assertEqual([candidate.spec for candidate in candidates], ["cpu", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
