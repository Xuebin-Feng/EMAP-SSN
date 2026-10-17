# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Detect_GPU: inventory merging, per-vendor eligibility, the backend candidate
ladder and the full detection report, with every hardware probe mocked."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
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

    def test_intel_xpu_operating_systems_follow_the_platform_of_the_pci_id(self):
        windows_11, windows_10 = {"windows_build": 26200}, {"windows_build": 19045}

        def ubuntu(release):
            return {"id": "ubuntu", "version_id": release}

        cases = [
            # Lunar Lake: Windows 11 and Ubuntu 24.04 or newer.
            ("64a0", "windows", windows_11, True), ("64a0", "windows", windows_10, False),
            ("64a0", "linux", ubuntu("22.04"), False), ("64a0", "linux", ubuntu("24.04"), True),
            ("64a0", "linux", {"id": "fedora", "version_id": "43"}, False),
            # PyTorch lists Panther Lake for Ubuntu 25.10, and Battlemage, the
            # Arc Pro B70 (BMG-G31) included, from 24.04.
            ("b080", "linux", ubuntu("24.04"), False), ("b080", "linux", ubuntu("25.10"), True),
            ("b090", "linux", ubuntu("26.04"), True), ("b090", "windows", windows_11, True),
            ("e223", "linux", ubuntu("24.04"), True), ("e223", "windows", windows_11, True),
            ("e20b", "linux", ubuntu("22.04"), False),
            # Meteor Lake-U has kernels in the Windows wheel only, Max in the Linux one only.
            ("7d45", "windows", windows_11, True), ("7d45", "linux", ubuntu("24.04"), False),
            ("0bd5", "windows", windows_11, False), ("0bd5", "linux", ubuntu("22.04"), True),
            ("0bd5", "linux", {"id": "rhel", "version_id": "9.2"}, True),
            ("0bd5", "linux", {"id": "sles", "version_id": "15.5"}, True),
            ("0bd5", "linux", ubuntu("24.04"), False),
        ]
        for device_id, system, os_info, expected in cases:
            with self.subTest(device_id=device_id, system=system, os_info=os_info):
                device = gpu(f"Intel Corporation Device [8086:{device_id}]", "INTEL", identifier="intel0")
                device["pci_id"] = f"8086:{device_id}"
                Detect_GPU._evaluate_devices([device], system, os_info, set())
                self.assertEqual(device["eligible_profiles"], ["xpu"] if expected else [])

    def test_intel_xpu_device_ids_are_unique_lowercase_pci_device_ids(self):
        listed = [device for platform in Detect_GPU.INTEL_XPU_PLATFORMS for device in platform.device_ids]
        self.assertEqual(len(listed), len(set(listed)))
        self.assertTrue(all(len(device) == 4 and device == device.lower() and int(device, 16) for device in listed))
        self.assertEqual(Detect_GPU._intel_platform("8086:e20b", "").name, "Arc B-series (Battlemage)")
        # Another vendor's ID with a known device number is not an Intel GPU.
        self.assertIsNone(Detect_GPU._intel_platform("1002:e20b", "Intel Arc B580"))

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

    def test_gpus_and_drivers_too_old_for_cuda_12_are_left_out_with_a_warning(self):
        # The CUDA 12.6 build's oldest kernels are for compute capability 5.0,
        # and CUDA 12 needs driver 525.60.13 on Linux, 528.33 on Windows. Below
        # either, the multi-GB build downloaded and then failed validation.
        for name, capability, driver, system, message in (
            ("NVIDIA GeForce GTX 780", "3.5", "470.256.02", "linux",
             "has compute capability 3.5; the CUDA builds EMAP-SSN installs need 5.0 or newer"),
            ("NVIDIA GeForce GTX 750 Ti", "5.0", "470.256.02", "linux",
             "needs NVIDIA driver 525.60.13 or newer for CUDA 12 (installed: 470.256.02)"),
            ("NVIDIA GeForce GTX 1080", "6.1", "527.56", "windows",
             "needs NVIDIA driver 528.33 or newer for CUDA 12 (installed: 527.56)"),
        ):
            with self.subTest(name=name, system=system):
                device = gpu(name, "NVIDIA", identifier="n0", kind="discrete", capability=capability, driver=driver)
                Detect_GPU._evaluate_devices([device], system, {"windows_build": 26200}, set())
                self.assertEqual(device["eligible_profiles"], [])
                self.assertIn(message, device["warning"])
                candidates, _ignored = Detect_GPU._candidate_ladder([device])
                self.assertEqual([candidate["backend"] for candidate in candidates], ["cpu"])
        # At the minimums the GPU is used. Without nvidia-smi there is no compute
        # capability and the driver is the Windows package's own number, which
        # is never compared.
        for name, capability, driver, system in (
            ("NVIDIA GeForce GTX 750 Ti", "5.0", "525.60.13", "linux"),
            ("NVIDIA GeForce GTX 1080", "6.1", "528.33", "windows"),
            ("NVIDIA GeForce GTX 1080", None, "32.0.15.6094", "windows"),
        ):
            with self.subTest(name=name, system=system, driver=driver):
                device = gpu(name, "NVIDIA", identifier="n0", kind="discrete", capability=capability, driver=driver)
                Detect_GPU._evaluate_devices([device], system, {"windows_build": 26200}, set())
                self.assertNotIn("warning", device)
                self.assertEqual(device["eligible_profiles"], ["cuda"])

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

    def test_linux_rocm_releases(self):
        # Every target: the pinned Ubuntu releases and distributions built on
        # them, by their UBUNTU_CODENAME. Instinct MI300 (gfx942) and MI350
        # (gfx950) also: the enterprise releases AMD's 7.14.1 matrix lists.
        mint = {"id": "linuxmint", "id_like": "ubuntu debian", "version_id": "22", "ubuntu_codename": "noble"}
        pop = {"id": "pop", "id_like": "ubuntu debian", "version_id": "24.04", "ubuntu_codename": "noble"}
        cases = [
            ({"id": "ubuntu", "version_id": "22.04"}, "gfx1100", True),
            ({"id": "ubuntu", "version_id": "20.04"}, "gfx1100", False),
            (mint, "gfx1100", True),
            (pop, "gfx1201", True),
            (dict(mint, ubuntu_codename="jammy", version_id="21.3"), "gfx1102", True),
            # An Ubuntu base older than the pinned releases, and a codename
            # without an Ubuntu ID_LIKE, are not enough.
            (dict(mint, ubuntu_codename="focal", version_id="20.3"), "gfx1100", False),
            ({"id": "other", "version_id": "1", "ubuntu_codename": "noble"}, "gfx1100", False),
            # Radeon targets stay on Ubuntu.
            ({"id": "debian", "version_id": "13"}, "gfx1100", False),
            ({"id": "rhel", "version_id": "9.4"}, "gfx1100", False),
            ({"id": "rhel", "version_id": "9.4"}, "gfx90a", False),
            ({"id": "rhel", "version_id": "9.4"}, "gfx942", True),
            ({"id": "rhel", "version_id": "8.10"}, "gfx942", True),
            ({"id": "rhel", "version_id": "10.2"}, "gfx950", True),
            ({"id": "rhel", "version_id": "9.2"}, "gfx942", False),
            ({"id": "rhel", "version_id": "10.1"}, "gfx942", False),
            ({"id": "rhel", "version_id": "8.1"}, "gfx942", False),
            ({"id": "rocky", "version_id": "9.6"}, "gfx942", True),
            ({"id": "rocky", "version_id": "9.6"}, "gfx950", False),
            ({"id": "ol", "version_id": "8.10"}, "gfx942", True),
            ({"id": "ol", "version_id": "8.10"}, "gfx950", False),
            ({"id": "ol", "version_id": "10.0"}, "gfx950", True),
            ({"id": "debian", "version_id": "12"}, "gfx950", True),
            ({"id": "debian", "version_id": "11"}, "gfx942", False),
            ({"id": "sles", "version_id": "15.7"}, "gfx942", True),
            ({"id": "sles", "version_id": "16.0"}, "gfx950", True),
            ({"id": "sles", "version_id": "15.6"}, "gfx942", False),
            ({"id": "fedora", "version_id": "42"}, "gfx942", False),
        ]
        for os_info, target, expected in cases:
            with self.subTest(os_info=os_info, target=target):
                self.assertEqual(Detect_GPU._linux_rocm_release(os_info, target) is not None, expected)

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
    def _detect(
        self, *, system="Windows", version="10.0.26200", controllers=(), processors=(), nvidia=(), adapters=()
    ):
        # Without Win32_VideoController rows (adapters), an empty CIM
        # (PowerShell) result leaves the controller names as the whole Windows
        # inventory; every other probe is mocked as well.
        def cim(class_name, _properties):
            return [dict(row) for row in adapters] if class_name == "Win32_VideoController" else []

        with mock.patch.object(Detect_GPU.platform, "system", return_value=system), \
                mock.patch.object(Detect_GPU.platform, "version", return_value=version), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=list(controllers)), \
                mock.patch.object(Detect_GPU, "_windows_names", return_value=list(processors)), \
                mock.patch.object(Detect_GPU, "_windows_cim", side_effect=cim), \
                mock.patch.object(
                    Detect_GPU, "_run", side_effect=AssertionError("Detect_GPU ran a real subprocess")
                ), \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=list(nvidia)), \
                mock.patch.object(Detect_GPU, "_kfd_nodes", return_value=[]):
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
            "AMD Radeon RX 7650 GRE": "gfx1102",
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
                mock.patch.object(Detect_GPU, "_kfd_nodes", return_value=[]), \
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

    def _detect_linux_node(self, *, lspci_dnnk, lspci, rocm_agents, os_release, kfd_nodes=()):
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
                mock.patch.object(Detect_GPU, "_kfd_nodes", return_value=[dict(node) for node in kfd_nodes]), \
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

    def test_linux_instinct_mi300a_on_enterprise_releases(self):
        # AMD lists the MI300 series on RHEL, Rocky, Oracle Linux, Debian and
        # SLES as well, so the node gets ROCm there too. A release AMD does not
        # list, or another distribution, finds the four APUs and uses the CPU.
        for os_release, backend in (
            ({"id": "rhel", "version_id": "9.4"}, "rocm"),
            ({"id": "rocky", "version_id": "9.6"}, "rocm"),
            ({"id": "debian", "version_id": "13"}, "rocm"),
            ({"id": "sles", "version_id": "15.7"}, "rocm"),
            ({"id": "rhel", "version_id": "9.2"}, "cpu"),
            ({"id": "fedora", "version_id": "42"}, "cpu"),
        ):
            with self.subTest(**os_release):
                report = self._detect_linux_node(
                    lspci_dnnk=self.MI300A_LSPCI_DNNK,
                    lspci=self.MI300A_LSPCI,
                    rocm_agents=["gfx942"] * 4,
                    os_release=os_release,
                )
                amd = [device for device in report["devices"] if device["vendor"] == "AMD"]
                self.assertEqual([device["pci_id"] for device in amd], ["1002:74a0"] * 4)
                self.assertEqual(report["backend"], backend)
                for device in amd:
                    self.assertEqual(device["architecture"], "gfx942")
                    self.assertEqual(device["eligible_profiles"], ["rocm"] if backend == "rocm" else [])
                if backend == "cpu":
                    self.assertEqual([candidate["backend"] for candidate in report["backend_candidates"]], ["cpu"])
                    # The APUs, not the BMC's VGA controller, explain the CPU
                    # fallback, with their verdict rather than how their GFX
                    # target was found.
                    self.assertEqual(
                        report["reason"],
                        "Linux ROCm needs Ubuntu 22.04, 24.04, 25.10 or 26.04, or a distribution built on "
                        "one of them, or a RHEL, Rocky Linux, Oracle Linux, Debian or SLES release that "
                        "AMD lists for gfx942.",
                    )

    def test_cpu_fallback_reason_comes_from_the_first_ranked_gpu(self):
        # An integrated Intel GPU listed first used to explain the fallback of
        # a discrete Radeon on an unsupported distribution.
        lspci_dnnk = (
            "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation AlderLake-S GT1 [8086:a780] (rev 0c)\n"
            "\tKernel driver in use: i915\n"
            "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c] (rev c8)\n"
            "\tKernel driver in use: amdgpu\n"
        )
        report = self._detect_linux_node(
            lspci_dnnk=lspci_dnnk, lspci=None, rocm_agents=["gfx1100"],
            os_release={"id": "fedora", "version_id": "42"},
        )
        self.assertEqual(report["backend"], "cpu")
        self.assertTrue(report["reason"].startswith("Linux ROCm needs Ubuntu 22.04"), report["reason"])

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

    NAVI31_AND_RAPHAEL = (
        "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
        "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] [1002:744c] (rev c8)\n"
        "\tKernel driver in use: amdgpu\n"
        "0000:7c:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
        "Raphael [1002:164e] (rev c1)\n"
        "\tKernel driver in use: amdgpu\n"
    )

    def test_linux_kfd_gives_each_gpu_its_target_without_rocm_tools(self):
        # The kernel's KFD lists each GPU ROCm can use with its target and PCI
        # address, with no ROCm software installed. Without rocminfo these
        # GPUs used to be name-matched (provisional) or, unnamed, rejected;
        # with it, three different targets could not be told apart.
        # A Phoenix laptop iGPU's lspci name carries no model, so only the
        # kernel's gfx1103 tells it is integrated graphics.
        others = (
            "0000:0a:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT] [1002:73bf] (rev c1)\n"
            "\tKernel driver in use: amdgpu\n"
            "0000:c5:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Phoenix1 [1002:15bf] (rev c4)\n"
            "\tKernel driver in use: amdgpu\n"
        )
        nodes = [
            KFD_CPU_NODE,
            kfd_node("0000:03:00.0", 110000),
            kfd_node("0000:0a:00.0", 100300),
            kfd_node("0000:7c:00.0", 100306),
            kfd_node("0000:c5:00.0", 110003),
        ]
        for agents in ([], ["gfx1100", "gfx1030", "gfx1036", "gfx1103"]):
            with self.subTest(rocm_agents=agents):
                report = self._detect_linux_node(
                    lspci_dnnk=self.NAVI31_AND_RAPHAEL + others,
                    lspci=None,
                    rocm_agents=agents,
                    os_release={"id": "ubuntu", "version_id": "24.04"},
                    kfd_nodes=nodes,
                )
                found = {
                    device["id"]: (
                        device["architecture"], device.get("architecture_source"), device["eligibility"], device["kind"]
                    )
                    for device in report["devices"]
                }
                self.assertEqual(found, {
                    "0000:03:00.0": ("gfx1100", "kfd", "eligible", "discrete"),
                    "0000:0a:00.0": ("gfx1030", "kfd", "eligible", "discrete"),
                    "0000:7c:00.0": ("gfx1036", "kfd", "eligible", "unknown"),
                    "0000:c5:00.0": ("gfx1103", "kfd", "eligible", "integrated"),
                })
                self.assertEqual(report["kfd_targets"], {
                    "0:03:00.0": "gfx1100", "0:0a:00.0": "gfx1030", "0:7c:00.0": "gfx1036", "0:c5:00.0": "gfx1103",
                })
                rocm = report["backend_candidates"][0]
                self.assertEqual(
                    (rocm["gfx_target"], rocm["device_ids"], rocm["eligibility"]),
                    ("gfx1100", ["0000:03:00.0"], "eligible"),
                )

    def test_linux_kfd_target_beats_the_gpu_name(self):
        # pci.ids names a PCI ID, and AMD gives the Radeon 840M to gfx1152 and
        # gfx1153 processors alike. With a name like this one (which a pci.ids
        # could give Krackan2), only the kernel knows the target.
        lspci_dnnk = (
            "0000:c4:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Krackan2 [Radeon 820M / 840M Graphics] [1002:1902] (rev c3)\n"
            "\tKernel driver in use: amdgpu\n"
        )
        report = self._detect_linux_node(
            lspci_dnnk=lspci_dnnk,
            lspci=None,
            rocm_agents=[],
            os_release={"id": "ubuntu", "version_id": "26.04"},
            kfd_nodes=[KFD_CPU_NODE, kfd_node("0000:c4:00.0", 110503)],
        )
        (device,) = report["devices"]
        self.assertEqual(
            (device["architecture"], device["architecture_source"], device["kind"], device["eligibility"]),
            ("gfx1153", "kfd", "integrated", "eligible"),
        )
        self.assertIn("The kernel reports gfx1153 for this GPU; its name suggested gfx1152.", device["reasons"])
        self.assertEqual((report["backend"], report["gfx_target"]), ("rocm", "gfx1153"))

    def test_linux_gpu_without_a_kfd_node_gets_no_rocm(self):
        # The ROCm tools report targets without saying which GPU has them, so
        # an unnamed GPU used to take the only one reported, here the RX 7900
        # XTX's. The kernel lists no compute node for this iGPU: ROCm cannot
        # use it.
        report = self._detect_linux_node(
            lspci_dnnk=self.NAVI31_AND_RAPHAEL,
            lspci=None,
            rocm_agents=["gfx1100"],
            os_release={"id": "ubuntu", "version_id": "24.04"},
            kfd_nodes=[KFD_CPU_NODE, kfd_node("0000:03:00.0", 110000)],
        )
        devices = {device["id"]: device for device in report["devices"]}
        igpu = devices["0000:7c:00.0"]
        self.assertEqual((igpu["architecture"], igpu["eligible_profiles"]), (None, []))
        self.assertEqual(igpu["reasons"], ["The kernel's ROCm driver (KFD) lists no compute node for this GPU."])
        self.assertEqual(devices["0000:03:00.0"]["eligible_profiles"], ["rocm"])
        self.assertEqual(report["backend_candidates"][0]["device_ids"], ["0000:03:00.0"])

    def test_linux_kfd_targets_of_instinct_parts_by_pci_domain(self):
        # The MI300A node's four APUs sit in PCI domains 0 to 3. The KFD gives
        # gfx942 as 90402 and gfx90a as 90010: minor and stepping are hex.
        report = self._detect_linux_node(
            lspci_dnnk=self.MI300A_LSPCI_DNNK,
            lspci=self.MI300A_LSPCI,
            rocm_agents=[],
            os_release={"id": "ubuntu", "version_id": "24.04"},
            kfd_nodes=[KFD_CPU_NODE, *(kfd_node(f"000{domain}:01:00.0", 90402) for domain in range(4))],
        )
        amd = [device for device in report["devices"] if device["vendor"] == "AMD"]
        self.assertEqual(
            [(device["id"], device["architecture"], device["eligibility"]) for device in amd],
            [(f"000{domain}:01:00.0", "gfx942", "eligible") for domain in range(4)],
        )
        self.assertEqual((report["backend"], len(report["backend_candidates"][0]["device_ids"])), ("rocm", 4))
        mi210 = (
            "0000:03:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] "
            "Aldebaran/MI200 [Instinct MI210] [1002:740f] (rev 02)\n"
            "\tKernel driver in use: amdgpu\n"
        )
        report = self._detect_linux_node(
            lspci_dnnk=mi210, lspci=None, rocm_agents=[], os_release={"id": "ubuntu", "version_id": "22.04"},
            kfd_nodes=[KFD_CPU_NODE, kfd_node("0000:03:00.0", 90010)],
        )
        self.assertEqual(
            (report["devices"][0]["architecture"], report["backend"], report["backend_candidates"][0]["eligibility"]),
            ("gfx90a", "rocm", "eligible"),
        )

    def _detect_linux_intel(self, line: str, driver: str, os_release: dict[str, str]) -> dict:
        lspci_dnnk = f"0000:00:02.0 {line}\n\tKernel driver in use: {driver}\n"
        return self._detect_linux_node(lspci_dnnk=lspci_dnnk, lspci=None, rocm_agents=[], os_release=os_release)

    def test_linux_intel_gpus_are_found_by_pci_id_whatever_lspci_calls_them(self):
        # Names from the pci.ids in Ubuntu 24.04 (2024.03.31). It calls these
        # GPUs "Intel Graphics", "Arc Xe Graphics" or nothing at all, so a match
        # on Arc model names missed every one; the wheels carry kernels for all.
        ubuntu_24_04 = {"id": "ubuntu", "version_id": "24.04"}
        cases = [
            ("VGA compatible controller [0300]: Intel Corporation Lunar Lake [Intel Graphics] [8086:64a0] (rev 04)",
             "xe", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation Arrow Lake-P [Intel Graphics] [8086:7dd1] (rev 03)",
             "i915", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation Meteor Lake-P [Intel Graphics] [8086:7dd5] (rev 08)",
             "i915", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation Device [8086:e20b]", "xe", "discrete"),
            ("VGA compatible controller [0300]: Intel Corporation DG2 [Arc Xe Graphics] [8086:56a3] (rev 05)",
             "i915", "discrete"),
        ]
        for line, driver, kind in cases:
            with self.subTest(line=line):
                report = self._detect_linux_intel(line, driver, ubuntu_24_04)
                device = report["devices"][0]
                self.assertEqual((device["vendor"], device["kind"], device["eligibility"]), ("INTEL", kind, "eligible"))
                self.assertEqual(report["backend"], "xpu")

    def test_linux_intel_gpus_the_xpu_wheels_have_no_kernels_for_get_none(self):
        # The wheels could run these only by compiling their SPIR-V copy on
        # first use, and PyTorch lists none of them. The Max 1550VG's name
        # alone used to make it eligible.
        cases = [
            ("VGA compatible controller [0300]: Intel Corporation Arrow Lake-U [Intel Graphics] [8086:7d41] (rev 04)",
             "24.04", "no kernels for this Intel GPU (8086:7d41)"),
            ("VGA compatible controller [0300]: Intel Corporation Meteor Lake-P [Intel Graphics] [8086:7d45] (rev 08)",
             "24.04", "build for this OS has no kernels for Meteor Lake-U"),
            ("VGA compatible controller [0300]: Intel Corporation Wildcat Lake [Intel Graphics] [8086:fd80]",
             "25.10", "no kernels for this Intel GPU (8086:fd80)"),
            ("VGA compatible controller [0300]: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] [8086:9a49] "
             "(rev 01)", "24.04", "no kernels for this Intel GPU (8086:9a49)"),
            ("Display controller [0380]: Intel Corporation ATS-M [Data Center GPU Flex 170] [8086:56c0] (rev 08)",
             "22.04", "no kernels for this Intel GPU (8086:56c0)"),
            ("Display controller [0380]: Intel Corporation Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1550VG] "
             "[8086:0bd4] (rev 2f)", "22.04", "no kernels for this Intel GPU (8086:0bd4)"),
        ]
        for line, release, reason in cases:
            with self.subTest(line=line):
                report = self._detect_linux_intel(line, "i915", {"id": "ubuntu", "version_id": release})
                device = report["devices"][0]
                self.assertEqual((device["vendor"], device["eligible_profiles"]), ("INTEL", []))
                self.assertIn(reason, device["reasons"][0])
                self.assertEqual(report["backend"], "cpu")

    def test_linux_intel_gpus_outside_the_xpu_table_are_classed_by_pci_id(self):
        # A real Linux check found the UHD 770 classed "unknown", which ranks
        # ahead of "integrated". Intel's discrete GPUs are listed by ID; every
        # other Intel GPU is a processor's graphics, whatever lspci calls it.
        ubuntu_24_04 = {"id": "ubuntu", "version_id": "24.04"}
        cases = [
            ("VGA compatible controller [0300]: Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780] "
             "(rev 04)", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation CoffeeLake-S GT2 [UHD Graphics 630] [8086:3e92]",
             "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] [8086:9a49] "
             "(rev 01)", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation Arrow Lake-U [Intel Graphics] [8086:7d41] (rev 04)",
             "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation Device [8086:fd80]", "integrated"),
            ("VGA compatible controller [0300]: Intel Corporation DG1 [Iris Xe MAX Graphics] [8086:4905] (rev 01)",
             "discrete"),
            ("Display controller [0380]: Intel Corporation ATS-M [Data Center GPU Flex 170] [8086:56c0] (rev 08)",
             "discrete"),
            ("Display controller [0380]: Intel Corporation Device [8086:0be5]", "discrete"),
        ]
        for line, kind in cases:
            with self.subTest(line=line):
                report = self._detect_linux_intel(line, "i915", ubuntu_24_04)
                device = report["devices"][0]
                self.assertEqual((device["vendor"], device["kind"], device["eligibility"]),
                                 ("INTEL", kind, "ineligible"))
                self.assertEqual(report["backend"], "cpu")

    def test_intel_gpus_without_a_pci_id_are_classed_by_name(self):
        # Legacy name discovery has no device ID; Intel's driver and pci.ids
        # names say which GPUs are discrete.
        cases = [
            ("Intel(R) UHD Graphics 770", "integrated"),
            ("Intel(R) Iris(R) Xe Graphics", "integrated"),
            ("Intel(R) HD Graphics 630", "integrated"),
            ("Intel(R) Graphics", "integrated"),
            ("Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics]", "integrated"),
            ("Intel Corporation Xeon E3-1200 v3/4th Gen Core Processor Integrated Graphics Controller", "integrated"),
            ("Intel(R) Iris(R) Xe MAX Graphics", "discrete"),
            ("Intel Corporation DG1 [Iris Xe MAX Graphics]", "discrete"),
            ("Intel(R) Data Center GPU Flex 170", "discrete"),
            ("Intel(R) Data Center GPU Max 1100", "discrete"),
            # Unchanged: a bare "Arc Graphics" is integrated only beside a Core Ultra CPU.
            ("Intel(R) Arc(TM) Graphics", "unknown"),
        ]
        for name, kind in cases:
            with self.subTest(name=name):
                self.assertEqual(Detect_GPU._device_kind("INTEL", name, None, []), kind)

    def test_intel_discrete_ids_hold_no_integrated_xpu_platform(self):
        integrated = {
            device for row in Detect_GPU.INTEL_XPU_PLATFORMS if row.kind == "integrated" for device in row.device_ids
        }
        self.assertEqual(integrated & Detect_GPU.INTEL_DISCRETE_DEVICE_IDS, set())
        self.assertTrue(all(len(device) == 4 and device == device.lower() and int(device, 16)
                            for device in Detect_GPU.INTEL_DISCRETE_DEVICE_IDS))

    def test_windows_intel_gpus_are_found_by_pci_id_whatever_the_driver_calls_them(self):
        # Intel's driver calls a Meteor Lake-H GPU "Intel(R) Graphics" rather
        # than "Intel(R) Arc(TM) Graphics" when the memory is single-channel,
        # and Meteor Lake-U always; the Windows wheel carries kernels for both.
        def adapter(name, device):
            return {
                "Name": name, "PNPDeviceID": rf"PCI\VEN_8086&DEV_{device}&SUBSYS_0C8A1028&REV_08\3&11583659&0&10",
                "DriverVersion": "32.0.101.8724",
            }

        cases = [
            (adapter("Intel(R) Graphics", "7D55"), "eligible", "Meteor Lake-H"),
            (adapter("Intel(R) Graphics", "7D45"), "eligible", "Meteor Lake-U"),
            (adapter("Intel(R) Arc(TM) B580 Graphics", "E20B"), "eligible", "Arc B-series"),
            (adapter("Intel(R) Graphics", "7D41"), "ineligible", "no kernels for this Intel GPU (8086:7d41)"),
            (adapter("Intel(R) Iris(R) Xe Graphics", "9A49"), "ineligible", "no kernels for this Intel GPU (8086:9a49)"),
            # Without Intel's driver the XPU build would install and fail validation.
            (adapter("Microsoft Basic Display Adapter", "E20B"), "ineligible", "Intel's graphics driver is not installed"),
        ]
        for row, eligibility, reason in cases:
            with self.subTest(row=row):
                report = self._detect(controllers=[row["Name"]], adapters=[row])
                device = report["devices"][0]
                self.assertEqual((device["vendor"], device["source"], device["eligibility"]),
                                 ("INTEL", "Win32_VideoController", eligibility))
                self.assertIn(reason, device["reasons"][0])
                self.assertEqual(report["backend"], "xpu" if eligibility == "eligible" else "cpu")

    def test_windows_intel_gpus_outside_the_xpu_table_are_classed_by_pci_id(self):
        def adapter(name, device):
            return {"Name": name, "PNPDeviceID": rf"PCI\VEN_8086&DEV_{device}&SUBSYS_7D251462&REV_04\3&11583659&0&10"}

        cases = [
            (adapter("Intel(R) UHD Graphics 770", "A780"), "integrated"),
            (adapter("Intel(R) Iris(R) Xe Graphics", "9A49"), "integrated"),
            # Without Intel's driver the name says nothing; the ID still does.
            (adapter("Microsoft Basic Display Adapter", "A780"), "integrated"),
            (adapter("Intel(R) Iris(R) Xe MAX Graphics", "4905"), "discrete"),
        ]
        for row, kind in cases:
            with self.subTest(row=row):
                device = self._detect(controllers=[row["Name"]], adapters=[row])["devices"][0]
                self.assertEqual((device["vendor"], device["kind"], device["eligibility"]),
                                 ("INTEL", kind, "ineligible"))

    def test_json_cli_contains_selection_reason(self):
        report = detected_report(reason="No accelerator")
        output = io.StringIO()
        with mock.patch.object(Detect_GPU, "detect_hardware", return_value=report), redirect_stdout(output):
            self.assertEqual(Detect_GPU.main(["--json"]), 0)
        self.assertEqual(json.loads(output.getvalue())["reason"], "No accelerator")


def completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def kfd_node(address: str, gfx_target_version: int) -> dict[str, int]:
    """A GPU node of the kernel's KFD topology for the PCI address DDDD:BB:DD.F."""
    domain, bus, rest = address.split(":")
    device, function = rest.split(".")
    return {
        "gfx_target_version": gfx_target_version,
        "vendor_id": 0x1002,
        "domain": int(domain, 16),
        "location_id": int(bus, 16) << 8 | int(device, 16) << 3 | int(function),
    }


# The KFD lists each CPU as a node too, with gfx_target_version 0.
KFD_CPU_NODE = {"cpu_cores_count": 16, "simd_count": 0, "gfx_target_version": 0}


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

    def test_kfd_topology_nodes_become_targets_by_pci_address(self):
        # /sys/class/kfd/kfd/topology/nodes/N/properties holds "name value"
        # lines: node 0 a CPU, then an RX 7900 XTX at 03:00.0, an MI210 at
        # c1:00.0 and an MI300A APU in PCI domain 3.
        properties = {
            "0": "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\nvendor_id 0\nlocation_id 0\ndomain 0\n",
            "1": "cpu_cores_count 0\nsimd_count 192\ngfx_target_version 110000\nvendor_id 4098\n"
                 "device_id 29772\nlocation_id 768\ndomain 0\ndrm_render_minor 128\nunique_id 9876543210123456789\n",
            "2": "simd_count 416\ngfx_target_version 90010\nvendor_id 4098\nlocation_id 49408\ndomain 0\n",
            "3": "simd_count 912\ngfx_target_version 90402\nvendor_id 4098\nlocation_id 256\ndomain 3\n",
        }
        with tempfile.TemporaryDirectory() as root:
            for node, text in properties.items():
                Path(root, node).mkdir()
                Path(root, node, "properties").write_text(text, encoding="ascii")
            Path(root, "4").mkdir()  # a node folder without properties is skipped
            with mock.patch.object(Detect_GPU, "KFD_TOPOLOGY_NODES", root):
                nodes = Detect_GPU._kfd_nodes()
            # WSL and containers have no KFD topology at all.
            with mock.patch.object(Detect_GPU, "KFD_TOPOLOGY_NODES", str(Path(root, "missing"))):
                self.assertEqual(Detect_GPU._kfd_nodes(), [])
        self.assertEqual(len(nodes), 4)
        self.assertEqual(
            Detect_GPU._kfd_targets(nodes),
            {"0:03:00.0": "gfx1100", "0:c1:00.0": "gfx90a", "3:01:00.0": "gfx942"},
        )
        self.assertEqual(Detect_GPU._pci_address("0003:01:00.0"), "3:01:00.0")

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


class WindowsProbeConcurrencyTests(unittest.TestCase):
    """Windows detection runs its four PowerShell CIM queries together.

    The queries are real subprocess calls in production; here subprocess.run is
    faked, so the real _run, _windows_cim and the parsers all execute.
    """

    NVIDIA_ID = "PCI\\VEN_10DE&DEV_2786&SUBSYS_1&REV_A1\\4&2283F625&0&0019"
    ROWS = {
        "Win32_Processor": [{"Name": "13th Gen Intel(R) Core(TM) i5-13600K"}],
        "Win32_OperatingSystem": [
            {"Caption": "Microsoft Windows 11 Education", "Version": "10.0.26300", "BuildNumber": "26300"}
        ],
        "controllers": [{"Name": "NVIDIA GeForce RTX 4070"}, {"Name": "Intel(R) UHD Graphics 770"}],
        "inventory": [
            {
                "Name": "NVIDIA GeForce RTX 4070", "PNPDeviceID": NVIDIA_ID, "DriverVersion": "32.0.15.7",
                "AdapterCompatibility": "NVIDIA", "VideoProcessor": "NVIDIA GeForce RTX 4070",
                "AdapterRAM": 4293918720,
            },
            {
                "Name": "Intel(R) UHD Graphics 770",
                "PNPDeviceID": "PCI\\VEN_8086&DEV_4680&SUBSYS_1&REV_0C\\3&11583659&0&10",
                "DriverVersion": "31.0.101.5", "AdapterCompatibility": "Intel Corporation",
                "VideoProcessor": "Intel(R) UHD Graphics Family", "AdapterRAM": 1073741824,
            },
        ],
    }

    @staticmethod
    def _query_of(command):
        script = command[-1]
        if "Win32_Processor" in script:
            return "Win32_Processor"
        if "Win32_OperatingSystem" in script:
            return "Win32_OperatingSystem"
        return "inventory" if "PNPDeviceID" in script else "controllers"

    def _fake_subprocess(self, failures=None, timeouts=None):
        """subprocess.run for the PowerShell queries, with chosen ones failing or timing out."""
        failures = failures or {}
        timeouts = list(timeouts or ())
        calls = []

        def run(command, **kwargs):
            if command[0] != "powershell":
                raise FileNotFoundError(command[0])  # no nvidia-smi on the fake host
            query = self._query_of(command)
            calls.append((query, kwargs.get("timeout")))
            if query in timeouts:
                raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))
            if query in failures:
                return completed("", returncode=failures[query])
            return completed("﻿" + json.dumps(self.ROWS[query]))

        return run, calls

    def _detect(self, run, *, sequential=False):
        patches = [
            mock.patch.object(Detect_GPU.platform, "system", return_value="Windows"),
            mock.patch.object(Detect_GPU.platform, "version", return_value="10.0.26300"),
            mock.patch.object(Detect_GPU.platform, "machine", return_value="AMD64"),
            mock.patch.object(Detect_GPU.subprocess, "run", side_effect=run),
        ]
        if sequential:
            # The behaviour before the queries were overlapped: one after another, in order.
            patches.append(mock.patch.object(
                Detect_GPU, "_call_concurrently", side_effect=lambda calls: [call() for call in calls]
            ))
        with ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            return Detect_GPU.detect_hardware()

    def test_the_four_queries_run_at_the_same_time(self):
        # Each probe waits for the other three at a barrier: run one after another,
        # the first would wait alone and the barrier would break.
        barrier = threading.Barrier(4, timeout=10)
        names = {"controllers": ["NVIDIA GeForce RTX 4070"], "processors": ["cpu"], "os": {"BuildNumber": "26300"}, "rows": []}

        def meet(value):
            def probe(*_args):
                barrier.wait()
                return value
            return probe

        with mock.patch.object(Detect_GPU, "_controller_names", side_effect=meet(names["controllers"])), \
                mock.patch.object(Detect_GPU, "_windows_names", side_effect=meet(names["processors"])), \
                mock.patch.object(Detect_GPU, "_windows_os", side_effect=meet(names["os"])), \
                mock.patch.object(Detect_GPU, "_windows_video_rows", side_effect=meet(names["rows"])):
            results = Detect_GPU._windows_probes("windows")
        self.assertEqual(results, (names["controllers"], names["processors"], names["os"], names["rows"]))

    def test_each_query_keeps_its_own_timeout_and_its_own_result(self):
        run, calls = self._fake_subprocess()
        report = self._detect(run)
        self.assertEqual(sorted(calls), sorted([
            ("Win32_Processor", 8), ("Win32_OperatingSystem", 8), ("controllers", 8), ("inventory", 8),
        ]))
        self.assertEqual(report["processors"], ["13th Gen Intel(R) Core(TM) i5-13600K"])
        self.assertEqual(report["windows_build"], 26300)
        self.assertEqual(
            [device["name"] for device in report["devices"]],
            ["NVIDIA GeForce RTX 4070", "Intel(R) UHD Graphics 770"],
        )

    def test_the_report_is_the_same_as_when_the_queries_ran_one_after_another(self):
        scenarios = {
            "all answer": {},
            "the processor query times out": {"timeouts": ["Win32_Processor"]},
            "the OS query times out": {"timeouts": ["Win32_OperatingSystem"]},
            "the controller-name query times out": {"timeouts": ["controllers"]},
            "the inventory query times out": {"timeouts": ["inventory"]},
            "every query times out": {
                "timeouts": ["Win32_Processor", "Win32_OperatingSystem", "controllers", "inventory"]
            },
            "a query exits with an error": {"failures": {"Win32_OperatingSystem": 1, "inventory": 1}},
            "two queries fail and one times out": {
                "failures": {"controllers": 1, "Win32_Processor": 5}, "timeouts": ["inventory"]
            },
        }
        for label, options in scenarios.items():
            with self.subTest(scenario=label):
                concurrent = self._detect(self._fake_subprocess(**options)[0])
                sequential = self._detect(self._fake_subprocess(**options)[0], sequential=True)
                self.assertEqual(
                    json.dumps(concurrent, sort_keys=True, default=str),
                    json.dumps(sequential, sort_keys=True, default=str),
                )
                self.assertEqual(
                    Install_Dependencies.hardware_fingerprint(concurrent),
                    Install_Dependencies.hardware_fingerprint(sequential),
                )

    def test_a_timed_out_query_degrades_as_it_always_did(self):
        report = self._detect(self._fake_subprocess(timeouts=["controllers"])[0])
        self.assertEqual(report["devices"], [])  # no controller names on Windows: no devices
        report = self._detect(self._fake_subprocess(timeouts=["Win32_OperatingSystem"])[0])
        # No CIM OS row: the build comes from platform.version().
        self.assertEqual(report["windows_build"], 26300)
        self.assertEqual(len(report["devices"]), 2)

    def test_an_unexpected_error_in_a_query_still_propagates(self):
        with mock.patch.object(Detect_GPU, "_windows_os", side_effect=RuntimeError("boom")), \
                mock.patch.object(Detect_GPU, "_windows_names", return_value=[]), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=[]), \
                mock.patch.object(Detect_GPU, "_windows_video_rows", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                Detect_GPU._windows_probes("windows")

    def test_other_systems_still_query_their_controllers_in_line(self):
        with mock.patch.object(Detect_GPU.platform, "system", return_value="Darwin"), \
                mock.patch.object(Detect_GPU.platform, "version", return_value="25.0"), \
                mock.patch.object(Detect_GPU.platform, "machine", return_value="x86_64"), \
                mock.patch.object(Detect_GPU, "_controller_names", return_value=[]) as controllers, \
                mock.patch.object(Detect_GPU, "_call_concurrently") as concurrently, \
                mock.patch.object(Detect_GPU, "_nvidia_devices", return_value=[]):
            Detect_GPU.detect_hardware()
        controllers.assert_called_once_with("darwin")
        concurrently.assert_not_called()

    def test_the_inventory_can_take_rows_already_fetched(self):
        with mock.patch.object(Detect_GPU, "_windows_cim") as cim:
            devices = Detect_GPU._windows_inventory([], self.ROWS["inventory"])
        cim.assert_not_called()
        self.assertEqual([device["vendor"] for device in devices], ["NVIDIA", "INTEL"])
        with mock.patch.object(Detect_GPU, "_windows_cim", return_value=self.ROWS["inventory"]) as cim:
            self.assertEqual(Detect_GPU._windows_inventory([]), devices)
        cim.assert_called_once()


if __name__ == "__main__":
    unittest.main()
