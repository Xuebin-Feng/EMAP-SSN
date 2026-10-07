# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Detect_GPU: inventory merging, per-vendor eligibility, the backend candidate
ladder and the full detection report, with every hardware probe mocked."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import Detect_GPU  # noqa: E402
import Install_Dependencies  # noqa: E402
from tests.gpu_fixtures import detected_report  # noqa: E402


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

    def test_windows_25h2_amd_gets_the_single_rocm_profile(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", kind="discrete", architecture="gfx1100")
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligible_profiles"], ["rocm"])
        self.assertEqual(device["eligibility"], "eligible")
        self.assertEqual(device["profile_eligibility"]["rocm"], "eligible")

    def test_windows_before_25h2_rejects_native_rocm(self):
        # AMD publishes native Windows ROCm for Windows 11 25H2 only, so both
        # Windows 10 and an older Windows 11 build are ineligible outright.
        for build in (19045, 26100):
            with self.subTest(build=build):
                device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100")
                Detect_GPU._evaluate_devices([device], "windows", {"windows_build": build}, set())
                self.assertEqual(device["eligible_profiles"], [])
                self.assertEqual(device["eligibility"], "ineligible")
                self.assertIn("Windows 11 25H2", device["reasons"][0])

    def test_unknown_amd_is_never_provisional(self):
        device = gpu("AMD Radeon Graphics", "AMD", identifier="amd0", architecture=None, driver=None)
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligibility"], "ineligible")

    def test_known_amd_without_driver_is_provisional(self):
        device = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100", driver=None)
        Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
        self.assertEqual(device["eligibility"], "provisional")

    def test_intel_arc_is_xpu_but_uhd_is_not(self):
        arc = gpu("Intel(R) Arc(TM) A770 Graphics", "INTEL", identifier="intel0")
        uhd = gpu("Intel(R) UHD Graphics 770", "INTEL", identifier="intel1")
        Detect_GPU._evaluate_devices([arc, uhd], "windows", {"windows_build": 26200}, set())
        self.assertEqual(arc["eligible_profiles"], ["xpu"])
        self.assertEqual(uhd["eligible_profiles"], [])

    def test_numbered_integrated_arc_selects_xpu_without_cpu_inventory(self):
        for model in ("130V", "140V", "130T", "140T", "B370", "B390", "Pro B390"):
            for brand in ("Intel(R) Arc(TM)", "Intel® Arc™", "Intel Arc"):
                name = f"{brand} {model} Graphics"
                with self.subTest(name=name):
                    kind = Detect_GPU._device_kind("INTEL", name, None, [])
                    self.assertEqual(kind, "integrated")
                    device = gpu(name, "INTEL", identifier="intel0", kind=kind)
                    Detect_GPU._evaluate_devices(
                        [device], "windows", {"windows_build": 26200}, set()
                    )
                    candidates, ignored = Detect_GPU._candidate_ladder([device])
                    self.assertEqual([item["backend"] for item in candidates], ["xpu", "cpu"])
                    self.assertEqual(candidates[0]["device_ids"], ["intel0"])
                    self.assertEqual(ignored, [])
                    specs = Install_Dependencies.backend_specs({"backend_candidates": candidates})
                    self.assertEqual(specs[0].backend, "xpu")

    def test_integrated_arc_retains_os_and_driver_checks(self):
        for system, os_info, model, expected in (
            ("windows", {"windows_build": 19045}, "140V", False),
            ("linux", {"id": "ubuntu", "version_id": "24.04"}, "140V", True),
            ("linux", {"id": "ubuntu", "version_id": "22.04"}, "140V", False),
            ("linux", {"id": "ubuntu", "version_id": "24.04"}, "B390", False),
            ("linux", {"id": "ubuntu", "version_id": "25.10"}, "B370", True),
            ("linux", {"id": "ubuntu", "version_id": "26.04"}, "B390", True),
            ("linux", {"id": "fedora", "version_id": "43"}, "140V", False),
            ("darwin", {}, "140V", False),
        ):
            with self.subTest(system=system, os_info=os_info, model=model):
                device = gpu(f"Intel Arc {model}", "INTEL", identifier="intel0", driver=None)
                Detect_GPU._evaluate_devices([device], system, os_info, set())
                self.assertEqual(device["eligible_profiles"], ["xpu"] if expected else [])
                if expected:
                    self.assertEqual(device["eligibility"], "provisional")

    def test_non_arc_intel_graphics_remain_ineligible(self):
        for name in ("Intel HD Graphics 630", "Intel UHD Graphics 770", "Intel Iris Xe Graphics", "Intel Graphics"):
            with self.subTest(name=name):
                device = gpu(name, "INTEL", identifier="intel0", kind="integrated")
                Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
                self.assertEqual(device["eligible_profiles"], [])

    def test_mixed_intel_devices_keep_integrated_arc_in_xpu_candidate(self):
        devices = []
        for index, (name, expected_kind) in enumerate((
            ("Intel Arc B390", "integrated"),
            ("Intel Arc B580", "discrete"),
            ("Intel(R) Arc(TM) A770M Graphics", "discrete"),
            ("Intel Arc Graphics", "integrated"),
        )):
            kind = Detect_GPU._device_kind("INTEL", name, None, ["Intel Core Ultra 7"])
            self.assertEqual(kind, expected_kind)
            devices.append(gpu(name, "INTEL", identifier=f"intel{index}", kind=kind))
        Detect_GPU._evaluate_devices(devices, "windows", {"windows_build": 26200}, set())
        candidates, ignored = Detect_GPU._candidate_ladder(devices)
        self.assertEqual(set(candidates[0]["device_ids"]), {"intel0", "intel1", "intel2", "intel3"})
        self.assertIn(candidates[0]["device_ids"][0], {"intel1", "intel2"})
        self.assertEqual(ignored, [])

    def test_amd_discrete_target_excludes_integrated_target(self):
        discrete = gpu("AMD Radeon RX 7900 XTX", "AMD", identifier="amd-d", kind="discrete", architecture="gfx1100", profiles=["rocm"])
        integrated = gpu("AMD Radeon 890M", "AMD", identifier="amd-i", kind="integrated", architecture="gfx1150", profiles=["rocm"])
        candidates, ignored = Detect_GPU._candidate_ladder([integrated, discrete])
        self.assertEqual(candidates[0]["device_ids"], ["amd-d"])
        self.assertEqual(candidates[0]["gfx_target"], "gfx1100")
        self.assertEqual([item["backend"] for item in candidates], ["rocm", "cpu"])
        self.assertEqual(ignored[0]["id"], "amd-i")

    def test_discrete_intel_precedes_integrated_amd(self):
        amd = gpu("AMD Radeon 890M", "AMD", identifier="amd-i", kind="integrated", architecture="gfx1150", profiles=["rocm"])
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

    def test_linux_rocm_accepts_every_channel_target_as_one_profile(self):
        # The per-release Linux profiles are gone. Eligibility is now a single
        # question: does the pinned torch wheel on the ROCm 7.14 multi-arch
        # channel ship a device package for this target? A target outside that
        # set stays ineligible.
        supported = gpu(
            "AMD Radeon RX 7900 XTX", "AMD", identifier="amd0", architecture="gfx1100"
        )
        integrated = gpu(
            "AMD Radeon 890M", "AMD", identifier="amd1", architecture="gfx1150"
        )
        off_channel = gpu(
            "AMD Radeon Test", "AMD", identifier="amd2", architecture="gfx900"
        )
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            Detect_GPU._evaluate_devices(
                [supported, integrated, off_channel],
                "linux",
                {"id": "ubuntu", "version_id": "24.04"},
                {"gfx1100", "gfx1150", "gfx900"},
            )

        self.assertEqual(supported["eligible_profiles"], ["rocm"])
        self.assertEqual(integrated["eligible_profiles"], ["rocm"])
        self.assertEqual(off_channel["eligible_profiles"], [])
        self.assertIn("ROCm 7.14 channel", off_channel["reasons"][0])

        candidates, _ignored = Detect_GPU._candidate_ladder([supported])
        self.assertEqual(
            [candidate["backend"] for candidate in candidates],
            ["rocm", "cpu"],
        )

    def test_linux_rocm_rejects_a_target_the_pinned_torch_has_no_extra_for(self):
        # AMD's channel ships gfx1250 kernels only for torch 2.11.0, so the pinned
        # 2.12.0 wheel declares no `device-gfx1250` extra. uv would only warn about
        # the unknown extra and install torch without gfx1250 kernels.
        named = gpu("AMD Test GPU", "AMD", identifier="amd0", architecture="gfx1250")
        reported = gpu("AMD Test GPU", "AMD", identifier="amd1")
        with mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            Detect_GPU._evaluate_devices(
                [named, reported],
                "linux",
                {"id": "ubuntu", "version_id": "24.04"},
                {"gfx1250"},
            )

        self.assertEqual(named["eligible_profiles"], [])
        self.assertIn("pinned PyTorch", named["reasons"][0])
        # A runtime-reported gfx1250 is not adopted as the device's target either.
        self.assertEqual(reported["eligible_profiles"], [])
        self.assertIsNone(reported["architecture"])


class GPUDetectionTests(unittest.TestCase):
    def _detect(self, *, system="Windows", version="10.0.26200", controllers=(), processors=(), nvidia=()):
        # An empty CIM (PowerShell) result leaves the controller names as the
        # whole Windows inventory; every other probe is mocked as well.
        with mock.patch.object(Detect_GPU.platform, "system", return_value=system), \
                mock.patch.object(Detect_GPU.platform, "version", return_value=version), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=list(controllers)), \
                mock.patch.object(Detect_GPU, "_windows_names", return_value=list(processors)), \
                mock.patch.object(Detect_GPU, "_windows_cim", return_value=[]), \
                mock.patch.object(
                    Detect_GPU, "_run", side_effect=AssertionError("Detect_GPU ran a real subprocess")
                ), \
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


def completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class ProbeParserTests(unittest.TestCase):
    """The parser behind each hardware probe, fed canned _run output."""

    FULL_QUERY = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    NAME_QUERY = ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"]

    def test_nvidia_smi_csv_rows_become_devices(self):
        output = (
            "00000000:01:00.0, NVIDIA GeForce RTX 4090, 8.9, 580.00\n"
            "00000000:02:00.0, Tesla V100-SXM2-16GB, 7.0, 575.51\n"
            "00000000:03:00.0, NVIDIA Graphics Device, , \n"
            "truncated, row\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)) as run:
            devices = Detect_GPU._nvidia_devices()
        run.assert_called_once_with(self.FULL_QUERY)
        self.assertEqual(
            devices,
            [
                {
                    "bus_id": "00000000:01:00.0",
                    "name": "NVIDIA GeForce RTX 4090",
                    "compute_capability": "8.9",
                    "driver_version": "580.00",
                },
                {
                    "bus_id": "00000000:02:00.0",
                    "name": "Tesla V100-SXM2-16GB",
                    "compute_capability": "7.0",
                    "driver_version": "575.51",
                },
                {
                    "bus_id": "00000000:03:00.0",
                    "name": "NVIDIA Graphics Device",
                    "compute_capability": None,
                    "driver_version": None,
                },
            ],
        )

    def test_drivers_without_compute_cap_fall_back_to_the_name_query(self):
        replies = {
            tuple(self.FULL_QUERY): completed(
                'Field "compute_cap" is not a valid field to query.\n', returncode=2
            ),
            tuple(self.NAME_QUERY): completed("NVIDIA GeForce GTX 1080, 470.82\nbroken\n"),
        }
        with mock.patch.object(
            Detect_GPU, "_run", side_effect=lambda command, **_: replies[tuple(command)]
        ) as run:
            devices = Detect_GPU._nvidia_devices()
        self.assertEqual(run.call_args_list, [mock.call(self.FULL_QUERY), mock.call(self.NAME_QUERY)])
        self.assertEqual(
            devices,
            [{"name": "NVIDIA GeForce GTX 1080", "driver_version": "470.82", "compute_capability": None}],
        )

        # Without nvidia-smi both queries come back empty.
        with mock.patch.object(Detect_GPU, "_run", return_value=None) as run:
            self.assertEqual(Detect_GPU._nvidia_devices(), [])
        self.assertEqual(run.call_args_list, [mock.call(self.FULL_QUERY), mock.call(self.NAME_QUERY)])

    def test_lspci_display_controllers_keep_their_own_kernel_driver(self):
        # Each display controller here is followed by its own detail lines.
        output = (
            "0000:00:1f.6 Ethernet controller [0200]: Intel Corporation Ethernet Connection (17) I219-LM [8086:1a1c] (rev 11)\n"
            "\tKernel driver in use: e1000e\n"
            "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation DG2 [Arc A770] [8086:56a0] (rev 08)\n"
            "\tSubsystem: Intel Corporation Device [8086:1020]\n"
            "\tKernel driver in use: i915\n"
            "\tKernel modules: i915, xe\n"
            "0000:03:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c] (rev c8)\n"
            "\tKernel driver in use: amdgpu\n"
            "0001:04:00.0 3D controller [0302]: NVIDIA Corporation GH100 [H100 PCIe] [10de:2331] (rev a1)\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)) as run:
            devices = Detect_GPU._linux_inventory([])
        run.assert_called_once_with(["lspci", "-Dnnk"], timeout=8)
        expected = [
            ("0000:00:02.0", "[Arc A770]", "INTEL", "8086:56a0", "discrete", None, "i915"),
            ("0000:03:00.0", "[Radeon RX 7900 XTX]", "AMD", "1002:744c", "discrete", "gfx1100", "amdgpu"),
            ("0001:04:00.0", "[H100 PCIe]", "NVIDIA", "10de:2331", "discrete", None, None),
        ]
        self.assertEqual(len(devices), len(expected))
        for device, (address, model, vendor, pci_id, kind, target, driver) in zip(devices, expected):
            with self.subTest(address=address):
                self.assertEqual(device["id"], address)
                self.assertIn(model, device["name"])
                self.assertEqual(
                    (device["vendor"], device["pci_id"], device["kind"]), (vendor, pci_id, kind)
                )
                self.assertEqual(device["architecture"], target)
                self.assertEqual(device["driver"], driver)
                self.assertEqual(device["source"], "lspci")

    def test_windows_cim_json_with_a_bom_becomes_devices(self):
        nvidia_id = "PCI\\VEN_10DE&DEV_2684&SUBSYS_889C1043&REV_A1\\4&2283F625&0&0019"
        amd_id = "PCI\\VEN_1002&DEV_744C&SUBSYS_0E3B1002&REV_C8\\6&1A2B3C4D&0&0008"
        rows = [
            {
                "Name": "NVIDIA GeForce RTX 4090",
                "PNPDeviceID": nvidia_id,
                "DriverVersion": "32.0.15.8097",
                "AdapterCompatibility": "NVIDIA",
                "VideoProcessor": "NVIDIA GeForce RTX 4090",
                "AdapterRAM": 4293918720,
            },
            # CIM leaves Name empty on some drivers; VideoProcessor names it.
            {
                "Name": None,
                "PNPDeviceID": amd_id,
                "DriverVersion": None,
                "AdapterCompatibility": "Advanced Micro Devices, Inc.",
                "VideoProcessor": "AMD Radeon RX 7900 XTX",
                "AdapterRAM": None,
            },
        ]
        output = "﻿" + json.dumps(rows) + "\r\n"
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)) as run:
            devices = Detect_GPU._windows_inventory([])
        run.assert_called_once_with(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "$items = @(Get-CimInstance Win32_VideoController | Select-Object "
                "Name,PNPDeviceID,DriverVersion,AdapterCompatibility,VideoProcessor,AdapterRAM); "
                "ConvertTo-Json -InputObject $items -Compress",
            ],
            timeout=8,
        )
        self.assertEqual(
            devices,
            [
                {
                    "id": nvidia_id,
                    "name": "NVIDIA GeForce RTX 4090",
                    "vendor": "NVIDIA",
                    "pci_id": "10de:2684",
                    "driver_version": "32.0.15.8097",
                    "kind": "discrete",
                    "architecture": None,
                    "source": "Win32_VideoController",
                },
                {
                    "id": amd_id,
                    "name": "AMD Radeon RX 7900 XTX",
                    "vendor": "AMD",
                    "pci_id": "1002:744c",
                    "driver_version": None,
                    "kind": "discrete",
                    "architecture": "gfx1100",
                    "source": "Win32_VideoController",
                },
            ],
        )

    def test_cim_results_accept_one_object_and_ignore_failures(self):
        cases = [
            (completed('﻿{"Name": "X"}'), [{"Name": "X"}]),
            (completed('[{"Name": "X"}, 5, "y"]'), [{"Name": "X"}]),
            (completed("not json"), []),
            (completed("  \n"), []),
            (completed('[{"Name": "X"}]', returncode=1), []),
            (None, []),
        ]
        for result, expected in cases:
            with self.subTest(stdout=getattr(result, "stdout", None)):
                self.assertEqual(Detect_GPU._json_result(result), expected)

    def test_cuda_13_needs_turing_or_newer_and_driver_580(self):
        cuda132 = ("cuda132", "All selected NVIDIA GPUs and the installed driver support CUDA 13.2.")
        cuda126 = ("cuda126", "CUDA 12.6 provides the common compatible NVIDIA profile.")

        def device(capability, driver):
            return {"compute_capability": capability, "driver_version": driver}

        cases = {
            "Ada at the minimum driver": ([device("8.9", "580.00")], cuda132),
            "Turing, the oldest CUDA 13 architecture": ([device("7.5", "580.00")], cuda132),
            "V100 (Volta)": ([device("7.0", "595.10")], cuda126),
            "driver just below 580": ([device("8.9", "579.99")], cuda126),
            "one Volta GPU among newer ones": ([device("8.9", "580.00"), device("7.0", "580.00")], cuda126),
            "capability unknown (name-only query)": ([device(None, "580.00")], cuda126),
            "driver unknown": ([device("8.9", None)], cuda126),
        }
        for label, (devices, expected) in cases.items():
            with self.subTest(label):
                self.assertEqual(Detect_GPU._nvidia_backend(devices), expected)


if __name__ == "__main__":
    unittest.main()
