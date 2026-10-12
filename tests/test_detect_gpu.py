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

    def test_intel_arc_pro_names_are_xpu_on_windows_and_linux(self):
        # Windows adapter names from Intel's driver, and lspci names from the
        # pci.ids in Ubuntu 24.04 (the Pro A-series) and 26.04 (the others).
        cases = [
            ("windows", "Intel(R) Arc(TM) Pro B60 Graphics", "discrete"),
            ("windows", "Intel(R) Arc(TM) Pro B50 Graphics", "discrete"),
            ("windows", "Intel(R) Arc(TM) Pro A60 Graphics", "discrete"),
            ("windows", "Intel(R) Arc(TM) Pro A30M Graphics", "discrete"),
            ("windows", "Intel(R) Arc(TM) Pro 140T GPU (16GB)", "integrated"),
            ("linux", "Intel Corporation Battlemage G21 [Arc Pro B60]", "discrete"),
            ("linux", "Intel Corporation DG2 [Arc Pro A40/A50]", "discrete"),
            ("linux", "Intel Corporation Arrow Lake-P [Arc Pro 130T/140T]", "integrated"),
        ]
        os_info = {"windows": {"windows_build": 26200}, "linux": {"id": "ubuntu", "version_id": "24.04"}}
        for system, name, kind in cases:
            with self.subTest(name=name):
                self.assertEqual(Detect_GPU._device_kind("INTEL", name, None, []), kind)
                device = gpu(name, "INTEL", identifier="intel0", kind=kind)
                Detect_GPU._evaluate_devices([device], system, os_info[system], set())
                self.assertEqual(device["eligible_profiles"], ["xpu"])
        npu = gpu("Intel Corporation Meteor Lake NPU", "INTEL", identifier="intel1")
        Detect_GPU._evaluate_devices([npu], "linux", os_info["linux"], set())
        self.assertEqual(npu["eligible_profiles"], [])

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

    def test_blackwell_with_a_driver_older_than_580_is_left_out_with_a_warning(self):
        # CUDA 12.6 has no Blackwell kernels: it would download and then fail
        # validation, so the GPU is left out and the user is told to update.
        old = gpu("NVIDIA GeForce RTX 5090", "NVIDIA", identifier="n0", kind="discrete", capability="12.0", driver="576.88")
        Detect_GPU._evaluate_devices([old], "windows", {"windows_build": 26200}, set())
        self.assertEqual(old["eligible_profiles"], [])
        self.assertIn("needs NVIDIA driver 580 or newer (installed: 576.88)", old["warning"])
        candidates, _ignored = Detect_GPU._candidate_ladder([old])
        self.assertEqual([candidate["backend"] for candidate in candidates], ["cpu"])

        current = gpu("NVIDIA GeForce RTX 5090", "NVIDIA", identifier="n0", kind="discrete", capability="12.0", driver="580.97")
        Detect_GPU._evaluate_devices([current], "windows", {"windows_build": 26200}, set())
        self.assertNotIn("warning", current)
        self.assertEqual(Detect_GPU._candidate_ladder([current])[0][0]["backend"], "cuda132")

    def test_pre_blackwell_gpus_keep_cuda_12_6_on_an_old_driver(self):
        for name, capability in (("NVIDIA GeForce RTX 2080", "7.5"), ("NVIDIA H100", "9.0")):
            with self.subTest(name=name):
                device = gpu(name, "NVIDIA", identifier="n0", kind="discrete", capability=capability, driver="560.94")
                Detect_GPU._evaluate_devices([device], "windows", {"windows_build": 26200}, set())
                self.assertNotIn("warning", device)
                self.assertEqual(Detect_GPU._candidate_ladder([device])[0][0]["backend"], "cuda126")

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
            "AMD Radeon RX 7700 XT": "gfx1101",
            "AMD Radeon RX 7600": "gfx1102",
            # Same silicon as the RX 7600, though AMD's list names only that.
            "AMD Radeon RX 7600 XT": "gfx1102",
            # The laptop RX 7700S is that silicon too, not an RX 7700 (gfx1101).
            "AMD Radeon RX 7700S": "gfx1102",
            # So are the laptop RX 7600S, RX 7600M and RX 7600M XT and the PRO
            # W7600 and W7500.
            "AMD Radeon RX 7600S": "gfx1102",
            "AMD Radeon RX 7600M": "gfx1102",
            "AMD Radeon RX 7600M XT": "gfx1102",
            "AMD Radeon PRO W7600": "gfx1102",
            "AMD Radeon PRO W7500": "gfx1102",
            "AMD Radeon PRO W6800": "gfx1030",
            "AMD Radeon 780M Graphics": "gfx1103",
            # AMD's 7.14.1 list has the Ryzen 200 series with these, and the
            # Ryzen AI 5 330 with a Radeon 820M.
            "AMD Radeon 760M Graphics": "gfx1103",
            "AMD Radeon 740M Graphics": "gfx1103",
            "AMD Radeon 820M Graphics": "gfx1152",
            "AMD Radeon 890M Graphics": "gfx1150",
            "AMD Radeon 8060S Graphics": "gfx1151",
            "AMD Radeon 860M Graphics": "gfx1152",
        }
        for name, target in examples.items():
            with self.subTest(name=name):
                report = self._detect(controllers=[name])
                self.assertEqual(report["backend"], Install_Dependencies.ROCM_BACKEND)
                self.assertEqual(report["gfx_target"], target)

    def test_windows_rx_7700s_beside_a_radeon_780m_keeps_its_own_target(self):
        # A Framework Laptop 16. Its CPU name names the 780M (gfx1103); the
        # 7700S's own name decides its target, and the CPU names stand in only
        # for an AMD GPU with a generic name, such as an APU's iGPU.
        report = self._detect(
            controllers=["AMD Radeon RX 7700S", "AMD Radeon 780M Graphics"],
            processors=["AMD Ryzen 9 7940HS w/ Radeon 780M Graphics"],
        )
        devices = {
            device["name"]: (device["architecture"], device["kind"]) for device in report["devices"]
        }
        self.assertEqual(devices["AMD Radeon RX 7700S"], ("gfx1102", "discrete"))
        self.assertEqual(devices["AMD Radeon 780M Graphics"], ("gfx1103", "integrated"))
        self.assertEqual(report["backend"], Install_Dependencies.ROCM_BACKEND)
        self.assertEqual(report["gfx_target"], "gfx1102")

    def test_windows_amd_dgpu_beside_an_apu_never_takes_the_igpu_target(self):
        # These CPU names name a 780M (gfx1103), which is how a generically
        # named iGPU gets its target. A discrete card used to be matched
        # against them too: one the table did not know took gfx1103, was
        # classed integrated and became a ROCm candidate with the iGPU's
        # kernels. It now keeps its own target, or none when the table does
        # not know it.
        laptop = "AMD Ryzen 9 7940HS w/ Radeon 780M Graphics"
        desktop = "AMD Ryzen 7 8700G w/ Radeon 780M Graphics"
        igpu = "AMD Radeon(TM) Graphics"
        cases = [
            ("AMD Radeon RX 7600S", laptop, "gfx1102"),
            ("AMD Radeon RX 7600M", laptop, "gfx1102"),
            ("AMD Radeon RX 7600M XT", laptop, "gfx1102"),
            ("AMD Radeon PRO W7600", desktop, "gfx1102"),
            ("AMD Radeon PRO W7500", desktop, "gfx1102"),
            # Not in the table: two RDNA 2 (gfx1032) cards.
            ("AMD Radeon RX 6600", desktop, None),
            ("AMD Radeon PRO W6600", desktop, None),
        ]
        for dgpu, cpu, target in cases:
            rows = [{"Name": name, "DriverVersion": "32.0.21013.1000"} for name in (dgpu, igpu)]
            with mock.patch.object(Detect_GPU, "_windows_cim", return_value=rows):
                cim_devices = Detect_GPU._windows_inventory([cpu])
            report = self._detect(controllers=[dgpu, igpu], processors=[cpu])
            for source, devices in (("Win32_VideoController", cim_devices), ("names", report["devices"])):
                with self.subTest(dgpu=dgpu, source=source):
                    found = {device["name"]: (device["architecture"], device["kind"]) for device in devices}
                    self.assertEqual(found, {dgpu: (target, "discrete"), igpu: ("gfx1103", "integrated")})
            with self.subTest(dgpu=dgpu, source="candidates"):
                # ROCm goes to the dGPU when it is known, else to the iGPU alone.
                ids = {device["name"]: device["id"] for device in report["devices"]}
                rocm = report["backend_candidates"][0]
                self.assertEqual(rocm["backend"], Install_Dependencies.ROCM_BACKEND)
                self.assertEqual(rocm["gfx_target"], target or "gfx1103")
                self.assertEqual(rocm["device_ids"], [ids[dgpu] if target else ids[igpu]])

    def test_windows_generic_apu_igpu_takes_its_target_from_the_cpu_name(self):
        # Windows may call an APU's iGPU just "AMD Radeon(TM) Graphics"; only
        # the CPU name tells which GPU it is.
        cases = {
            "AMD Ryzen AI 9 HX 370 w/ Radeon 890M": "gfx1150",
            "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S": "gfx1151",
            "AMD Ryzen AI 7 350 w/ Radeon 860M": "gfx1152",
            "AMD Ryzen 5 230 w/ Radeon 760M Graphics": "gfx1103",
            "AMD Ryzen 3 210 w/ Radeon 740M Graphics": "gfx1103",
            "AMD Ryzen AI 5 330 w/ Radeon 820M": "gfx1152",
            "AMD Ryzen AI 5 430 w/ Radeon 840M": "gfx1153",
        }
        for cpu, target in cases.items():
            with self.subTest(cpu=cpu):
                report = self._detect(controllers=["AMD Radeon(TM) Graphics"], processors=[cpu])
                device = report["devices"][0]
                self.assertEqual((device["architecture"], device["kind"]), (target, "integrated"))
                self.assertEqual(report["backend"], Install_Dependencies.ROCM_BACKEND)
                self.assertEqual(report["gfx_target"], target)

    def test_windows_radeon_840m_takes_its_target_from_the_cpu_name(self):
        # AMD gives the Radeon 840M to gfx1152 and gfx1153 processors, so for
        # that name the CPU decides even though the GPU's own name matches a row.
        # The 840M's name alone, with no CPU name to go by, stays gfx1152.
        cases = [
            ("AMD Ryzen AI 5 430 w/ Radeon 840M", "gfx1153"),
            ("AMD Ryzen AI 5 PRO 435 w/ Radeon 840M", "gfx1153"),
            ("AMD Ryzen AI 7 445 w/ Radeon 840M", "gfx1153"),
            ("AMD Ryzen AI 5 340 w/ Radeon 840M", "gfx1152"),
            ("AMD Ryzen AI 5 PRO 440 w/ Radeon 840M", "gfx1152"),
            ("AMD Ryzen AI 7 345 w/ Radeon 840M", "gfx1152"),
            (None, "gfx1152"),
        ]
        for cpu, target in cases:
            for igpu in ("AMD Radeon 840M Graphics", "AMD Radeon(TM) Graphics"):
                if cpu is None and igpu == "AMD Radeon(TM) Graphics":
                    continue
                with self.subTest(cpu=cpu, igpu=igpu):
                    report = self._detect(controllers=[igpu], processors=[cpu] if cpu else [])
                    device = report["devices"][0]
                    self.assertEqual((device["architecture"], device["kind"]), (target, "integrated"))
                    self.assertEqual(report["backend"], Install_Dependencies.ROCM_BACKEND)
                    self.assertEqual(report["gfx_target"], target)
        # Linux has no processor names; Krackan's lspci name keeps gfx1152.
        report = self._detect_linux_lspci(
            "Advanced Micro Devices, Inc. [AMD/ATI] Krackan [Radeon 840M / 860M Graphics] [1002:1114] (rev c1)",
            {"gfx1152"},
        )
        self.assertEqual((report["backend"], report["gfx_target"]), ("rocm", "gfx1152"))

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

    def test_a_newer_gpu_than_hopper_beside_a_pre_turing_gpu_is_left_out_with_a_warning(self):
        # One CUDA build serves every NVIDIA GPU. A GTX 1080 keeps the computer
        # on CUDA 12.6, which has no Blackwell kernels: validation passed on the
        # 1080 alone and the RTX 5090 went unused without a word.
        def smi(*gpus):
            return [
                {"bus_id": f"00000000:0{index + 1}:00.0", "name": name,
                 "compute_capability": capability, "driver_version": "595.79"}
                for index, (name, capability) in enumerate(gpus)
            ]

        report = self._detect(nvidia=smi(("NVIDIA GeForce GTX 1080", "6.1"), ("NVIDIA GeForce RTX 5090", "12.0")))
        self.assertEqual(report["backend"], "cuda126")
        self.assertEqual(report["backend_candidates"][0]["device_ids"], ["00000000:01:00.0"])
        (warning,) = report["warnings"]
        self.assertIn("NVIDIA GeForce RTX 5090 runs only on the CUDA 13.2 build", warning)
        self.assertIn("NVIDIA GeForce GTX 1080 does not", warning)
        blackwell = next(device for device in report["devices"] if "5090" in device["name"])
        self.assertEqual((blackwell["eligible_profiles"], blackwell["eligibility"]), ([], "ineligible"))
        # Every GPU keeps its build: a Turing card allows CUDA 13.2, and a V100
        # and an H100 both run on CUDA 12.6.
        for pair, backend in (
            ((("NVIDIA GeForce RTX 2080", "7.5"), ("NVIDIA GeForce RTX 5090", "12.0")), "cuda132"),
            ((("NVIDIA V100", "7.0"), ("NVIDIA H100", "9.0")), "cuda126"),
        ):
            with self.subTest(pair=pair):
                report = self._detect(nvidia=smi(*pair))
                self.assertEqual(report["backend"], backend)
                self.assertEqual(len(report["backend_candidates"][0]["device_ids"]), 2)
                self.assertEqual(report["warnings"], [])
        # Alone, a GPU with no reported driver is no other GPU's doing: it stays
        # a provisional candidate for runtime validation, with no warning.
        lone = [{"bus_id": "00000000:01:00.0", "name": "NVIDIA GeForce RTX 5090",
                 "compute_capability": "12.0", "driver_version": None}]
        report = self._detect(nvidia=lone)
        self.assertEqual((report["warnings"], report["backend_candidates"][0]["eligibility"]), ([], "provisional"))

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

    def _detect_linux_lspci(self, name: str, rocm_agents: set[str]) -> dict:
        # Ubuntu 24.04 with an accessible /dev/kfd, one AMD GPU, and the ROCm
        # runtime either reporting rocm_agents or not installed at all.
        lspci = (
            f"0000:03:00.0 VGA compatible controller [0300]: {name}\n"
            "\tKernel driver in use: amdgpu\n"
        )

        def run(command, timeout=5):
            if command == ["lspci", "-Dnnk"]:
                return completed(lspci)
            if command == ["rocm_agent_enumerator", "-name"] and rocm_agents:
                # The runtime lists its CPU agent as gfx000.
                return completed("".join(f"{agent}\n" for agent in ["gfx000", *sorted(rocm_agents)]))
            return None

        with mock.patch.object(Detect_GPU.platform, "system", return_value="Linux"), \
                mock.patch.object(Detect_GPU.platform, "version", return_value="#1 SMP"), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=[]), \
                mock.patch.object(
                    Detect_GPU, "_read_os_release", return_value={"id": "ubuntu", "version_id": "24.04"}
                ), \
                mock.patch.object(Detect_GPU, "_run", side_effect=run), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=[]), \
                mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            return Detect_GPU.detect_hardware()

    def test_linux_lspci_names_of_navi_33_map_to_gfx1102_with_or_without_rocm(self):
        # lspci names a PCI ID, not a product. The pci.ids in Ubuntu 24.04
        # (2024.03.31) start the 1002:7480 name with "RX 7700S", which the RX
        # 7700 (gfx1101) pattern used to match, so these cards never got ROCm
        # there. Newer pci.ids start it with "RX 7600". The 1002:7483 and
        # 1002:7489 names are the same in both. Navi 32 stays gfx1101.
        names = {
            "Navi 33 [Radeon RX 7700S/7600/7600S/7600M XT/PRO W7600] [1002:7480]": "gfx1102",
            "Navi 33 [Radeon RX 7600/7600 XT/7600M XT/7600S/7700S / PRO W7600] [1002:7480]": "gfx1102",
            "Navi 33 [Radeon RX 7600M/7600M XT] [1002:7483]": "gfx1102",
            "Navi 33 [Radeon Pro W7500] [1002:7489]": "gfx1102",
            "Navi 32 [Radeon RX 7700 XT / 7800 XT] [1002:747e]": "gfx1101",
        }
        for name, target in names.items():
            for agents, eligibility in (({target}, "eligible"), (set(), "provisional")):
                with self.subTest(name=name, rocm_agents=sorted(agents)):
                    report = self._detect_linux_lspci(
                        f"Advanced Micro Devices, Inc. [AMD/ATI] {name} (rev c0)", agents
                    )
                    self.assertEqual(report["devices"][0]["architecture"], target)
                    self.assertEqual(report["devices"][0]["kind"], "discrete")
                    self.assertEqual(report["backend"], "rocm")
                    self.assertEqual(report["gfx_target"], target)
                    self.assertEqual(report["backend_candidates"][0]["eligibility"], eligibility)

    def _detect_linux_node(self, *, lspci_dnnk, lspci, rocm_agents, os_release):
        # Every probe goes through _run, so both lspci readers run unpatched:
        # `lspci -Dnnk` builds the inventory and plain `lspci` names the
        # controllers, the fallback when the first yields nothing.
        replies = {("lspci", "-Dnnk"): lspci_dnnk, ("lspci",): lspci}
        if rocm_agents:
            # The runtime lists each CPU agent as gfx000, and rocminfo adds
            # target features to the ISA name.
            replies[("rocm_agent_enumerator", "-name")] = "".join(
                f"{agent}\n" for agent in ["gfx000", *rocm_agents]
            )
            replies[("rocminfo",)] = "".join(
                f"  Name:                    {agent}\n"
                f"      Name:                    amdgcn-amd-amdhsa--{agent}:sramecc+:xnack-\n"
                for agent in rocm_agents
            )

        def run(command, timeout=5):
            stdout = replies.get(tuple(command))
            return completed(stdout) if stdout is not None else None

        with mock.patch.object(Detect_GPU.platform, "system", return_value="Linux"), \
                mock.patch.object(Detect_GPU.platform, "version", return_value="#1 SMP"), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_read_os_release", return_value=dict(os_release)), \
                mock.patch.object(Detect_GPU, "_run", side_effect=run), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=[]), \
                mock.patch.object(Detect_GPU.Path, "exists", return_value=True), \
                mock.patch.object(Detect_GPU.os, "access", return_value=True):
            return Detect_GPU.detect_hardware()

    # AMD's MI300A system-acceptance guide lists a node's four APUs with plain
    # `lspci` as below (PCI 1002:74a0, class 1200). The node's BMC adds its own
    # VGA controller.
    MI300A_LSPCI = "".join(
        f"000{domain}:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] "
        "Aqua Vanjaram [Instinct MI300A]\n"
        + (
            "0000:c2:00.0 VGA compatible controller: ASPEED Technology, Inc. ASPEED Graphics Family (rev 52)\n"
            if domain == 0 else ""
        )
        for domain in range(4)
    )
    MI300A_LSPCI_DNNK = "".join(
        f"000{domain}:01:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. [AMD/ATI] "
        "Aqua Vanjaram [Instinct MI300A] [1002:74a0]\n"
        "\tKernel driver in use: amdgpu\n"
        "\tKernel modules: amdgpu\n"
        + (
            "0000:c2:00.0 VGA compatible controller [0300]: ASPEED Technology, Inc. "
            "ASPEED Graphics Family [1a03:2000] (rev 52)\n"
            "\tKernel driver in use: ast\n"
            if domain == 0 else ""
        )
        for domain in range(4)
    )

    def test_linux_instinct_mi300a_processing_accelerators_get_rocm(self):
        # The MI300 series is PCI class "Processing accelerators", not a display
        # class, and detection used to drop it: the node fell back to the CPU.
        # Plain lspci alone (no -Dnnk) is the controller-name fallback path.
        # Ubuntu 22.04's pci.ids (2022.01.22) predates the MI300A, so there
        # lspci prints only the vendor and the numeric ID.
        unnamed = {
            name: text.replace("Aqua Vanjaram [Instinct MI300A] [1002:74a0]", "Device [1002:74a0]").replace(
                "Aqua Vanjaram [Instinct MI300A]", "Device 74a0"
            )
            for name, text in (("dnnk", self.MI300A_LSPCI_DNNK), ("plain", self.MI300A_LSPCI))
        }
        for source, release, lspci_dnnk, lspci, pci_id, model in (
            ("lspci -Dnnk", "24.04", self.MI300A_LSPCI_DNNK, self.MI300A_LSPCI, "1002:74a0", "Instinct MI300A"),
            ("plain lspci", "24.04", None, self.MI300A_LSPCI, None, "Instinct MI300A"),
            ("lspci -Dnnk", "22.04", unnamed["dnnk"], unnamed["plain"], "1002:74a0", "[AMD/ATI] Device"),
            ("plain lspci", "22.04", None, unnamed["plain"], None, "[AMD/ATI] Device 74a0"),
        ):
            with self.subTest(source=source, ubuntu=release):
                report = self._detect_linux_node(
                    lspci_dnnk=lspci_dnnk,
                    lspci=lspci,
                    rocm_agents=["gfx942"] * 4,
                    os_release={"id": "ubuntu", "version_id": release},
                )
                amd = [device for device in report["devices"] if device["vendor"] == "AMD"]
                self.assertEqual(len(amd), 4)
                for device in amd:
                    self.assertIn(model, device["name"])
                    self.assertEqual(device["pci_id"], pci_id)
                    self.assertEqual(device["architecture"], "gfx942")
                    self.assertEqual(device["eligible_profiles"], ["rocm"])
                    self.assertEqual(device["eligibility"], "eligible")
                self.assertEqual(report["backend"], "rocm")
                self.assertEqual(report["gfx_target"], "gfx942")
                rocm = report["backend_candidates"][0]
                self.assertEqual(sorted(rocm["device_ids"]), sorted(device["id"] for device in amd))
                self.assertEqual(
                    [candidate["backend"] for candidate in report["backend_candidates"]], ["rocm", "cpu"]
                )

    def test_linux_instinct_mi300a_on_rhel_is_found_but_not_eligible(self):
        # Linux ROCm is limited to the supported Ubuntu releases, so the same
        # node on RHEL finds the four APUs and still uses the CPU.
        report = self._detect_linux_node(
            lspci_dnnk=self.MI300A_LSPCI_DNNK,
            lspci=self.MI300A_LSPCI,
            rocm_agents=["gfx942"] * 4,
            os_release={"id": "rhel", "version_id": "9.4"},
        )
        amd = [device for device in report["devices"] if device["vendor"] == "AMD"]
        self.assertEqual([device["pci_id"] for device in amd], ["1002:74a0"] * 4)
        for device in amd:
            self.assertEqual(device["architecture"], "gfx942")
            self.assertEqual(device["eligible_profiles"], [])
            self.assertIn(
                "Linux ROCm profile is limited to the pinned supported Ubuntu releases.", device["reasons"]
            )
        self.assertEqual(report["backend"], "cpu")
        self.assertEqual([candidate["backend"] for candidate in report["backend_candidates"]], ["cpu"])

    def test_linux_instinct_mi210_display_controller_is_unchanged(self):
        # The MI200 series is a display controller, which detection always kept.
        lspci_dnnk = (
            "0000:03:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Aldebaran/MI200 [Instinct MI210] [1002:740f] (rev 02)\n"
            "\tKernel driver in use: amdgpu\n"
        )
        lspci = (
            "03:00.0 Display controller: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Aldebaran/MI200 [Instinct MI210] (rev 02)\n"
        )
        report = self._detect_linux_node(
            lspci_dnnk=lspci_dnnk,
            lspci=lspci,
            rocm_agents=["gfx90a"],
            os_release={"id": "ubuntu", "version_id": "24.04"},
        )
        (device,) = report["devices"]
        self.assertEqual(
            (device["name"], device["pci_id"], device["architecture"], device["eligible_profiles"]),
            (
                "Advanced Micro Devices, Inc. [AMD/ATI] Aldebaran/MI200 [Instinct MI210]",
                "1002:740f",
                "gfx90a",
                ["rocm"],
            ),
        )
        self.assertEqual(report["backend"], "rocm")
        self.assertEqual(report["gfx_target"], "gfx90a")

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

    def test_lspci_gpu_keeps_its_driver_when_its_audio_function_follows(self):
        # lspci lists a GPU's HDMI audio function right after the GPU itself.
        output = (
            "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684] (rev a1)\n"
            "\tSubsystem: ASUSTeK Computer Inc. Device [1043:889c]\n"
            "\tKernel driver in use: nvidia\n"
            "\tKernel modules: nvidiafb, nouveau, nvidia_drm, nvidia\n"
            "0000:01:00.1 Audio device [0403]: NVIDIA Corporation AD102 High Definition Audio Controller [10de:22ba] (rev a1)\n"
            "\tSubsystem: ASUSTeK Computer Inc. Device [1043:889c]\n"
            "\tKernel driver in use: snd_hda_intel\n"
            "\tKernel modules: snd_hda_intel\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)):
            devices = Detect_GPU._linux_inventory([])
        self.assertEqual(
            [(device["id"], device["driver"]) for device in devices], [("0000:01:00.0", "nvidia")]
        )

    def test_lspci_gpu_without_a_driver_does_not_take_the_next_devices(self):
        # No driver is bound to this Arc GPU, so the Ethernet controller listed
        # after it must not lend it igc and make the XPU look eligible.
        output = (
            "0000:03:00.0 VGA compatible controller [0300]: Intel Corporation DG2 [Arc A770] [8086:56a0] (rev 08)\n"
            "\tSubsystem: Intel Corporation Device [8086:1020]\n"
            "\tKernel modules: i915, xe\n"
            "0000:05:00.0 Ethernet controller [0200]: Intel Corporation Ethernet Controller I225-V [8086:15f3] (rev 03)\n"
            "\tSubsystem: ASUSTeK Computer Inc. Device [1043:87d2]\n"
            "\tKernel driver in use: igc\n"
            "\tKernel modules: igc\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)):
            devices = Detect_GPU._linux_inventory([])
        self.assertEqual([(device["id"], device["driver"]) for device in devices], [("0000:03:00.0", None)])

        Detect_GPU._evaluate_devices(devices, "linux", {"id": "ubuntu", "version_id": "24.04"}, set())
        self.assertEqual(devices[0]["eligible_profiles"], ["xpu"])
        self.assertEqual(devices[0]["eligibility"], "provisional")

    def test_lspci_processing_accelerators_count_only_for_amd_gpus(self):
        # PCI class 12 also holds NPUs, FPGAs, DPUs and other vendors' AI cards,
        # as Ubuntu 24.04's pci.ids names them. AMD's Pensando DPU is among
        # them, and its vendor name contains "AMD". Only AMD graphics devices
        # (vendor 1002, "[AMD/ATI]") of that class are GPUs.
        lspci_dnnk = (
            "0000:00:0b.0 Processing accelerators [1200]: Intel Corporation Meteor Lake NPU [8086:7d1d] (rev 04)\n"
            "\tKernel driver in use: intel_vpu\n"
            "0000:01:00.0 Processing accelerators [1200]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Aqua Vanjaram [Instinct MI300X] [1002:74a1]\n"
            "\tKernel driver in use: amdgpu\n"
            "0000:02:00.0 Processing accelerators [1200]: Habana Labs Ltd. Gaudi2 AI Training Accelerator "
            "[1da3:1020] (rev 01)\n"
            "\tKernel driver in use: habanalabs\n"
            "0000:03:00.0 Processing accelerators [1200]: Xilinx Corporation Device [10ee:50b4]\n"
            "0000:04:00.0 Processing accelerators [1200]: Tenstorrent Inc Blackhole [1e52:b140]\n"
            "0000:b5:00.0 Processing accelerators [1200]: AMD Pensando Systems DSC PDS Core Management [1dd8:100c]\n"
            "\tKernel driver in use: pds_core\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(lspci_dnnk)):
            devices = Detect_GPU._linux_inventory([])
        self.assertEqual(
            [(device["id"], device["vendor"], device["pci_id"], device["driver"]) for device in devices],
            [("0000:01:00.0", "AMD", "1002:74a1", "amdgpu")],
        )

        lspci = (
            "00:0b.0 Processing accelerators: Intel Corporation Meteor Lake NPU (rev 04)\n"
            "01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X]\n"
            "02:00.0 Processing accelerators: Habana Labs Ltd. Gaudi2 AI Training Accelerator (rev 01)\n"
            "03:00.0 Processing accelerators: Xilinx Corporation Device 50b4\n"
            "04:00.0 Processing accelerators: Tenstorrent Inc Blackhole\n"
            "b5:00.0 Processing accelerators: AMD Pensando Systems DSC PDS Core Management\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(lspci)) as run:
            names = Detect_GPU._controller_names("linux")
        run.assert_called_once_with(["lspci"])
        self.assertEqual(
            names,
            ["01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300X]"],
        )

    def test_lspci_names_drop_the_trailing_pci_id_and_revision(self):
        # The numeric ID lives on in pci_id; a zero revision prints no "(rev xx)".
        output = (
            "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684] (rev a1)\n"
            "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c] (rev c8)\n"
            "0000:00:08.0 VGA compatible controller [0300]: Microsoft Corporation Hyper-V virtual VGA [1414:5353]\n"
        )
        with mock.patch.object(Detect_GPU, "_run", return_value=completed(output)):
            devices = Detect_GPU._linux_inventory([])
        self.assertEqual(
            [(device["name"], device["pci_id"]) for device in devices],
            [
                ("NVIDIA Corporation AD102 [GeForce RTX 4090]", "10de:2684"),
                (
                    "Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]",
                    "1002:744c",
                ),
                ("Microsoft Corporation Hyper-V virtual VGA", "1414:5353"),
            ],
        )

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
