# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for utilities/GPU_Memory_View.py: the nvidia-smi reader, its cache and switch, and the room beside other programs.

No test starts the real nvidia-smi or reads the real card: subprocess.run and the torch memory calls are
replaced, and the module's switch (which tests/__init__.py turns off for the whole run) is patched on
only where a test exercises the reader.
"""

import os
import subprocess
import sys
import types
import unittest
import contextlib
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch

from utilities import GPU_Memory_View

# The real switch function, kept before any test replaces it.
_REAL_SWITCH = GPU_Memory_View.other_programs_switch_on
_MIB = 1024 ** 2
_CARD_UUID = "GPU-8f6c3a2e-1b2c-4d5e-9f00-112233445566"
_OTHER_CARD_UUID = "GPU-0a1b2c3d-0000-4000-8000-aabbccddeeff"
_CARD_TOTAL_MIB = 12282


def _smi_result(*rows, returncode=0):
    """A CompletedProcess whose output has one ``uuid, used, total`` row per tuple."""
    text = "".join(f"{uuid}, {used}, {total}\n" for uuid, used, total in rows)
    return subprocess.CompletedProcess([], returncode, stdout=text, stderr="")


class GPUMemoryViewTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._reset_state()
        self.addCleanup(self._reset_state)
        patcher = mock.patch.object(GPU_Memory_View, "other_programs_switch_on", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _reset_state():
        GPU_Memory_View._OTHER_PROGRAMS_CACHE.clear()
        GPU_Memory_View._NVIDIA_SMI_MISSING = False

    @contextlib.contextmanager
    def _card(self, *, run, reserved_mib=0, torch_uuid=None, windows=True):
        """Replace subprocess.run, the reserved-memory call and the platform answer for one block."""
        if torch_uuid is None:
            props = mock.Mock(side_effect=RuntimeError("no properties"))
        else:
            props = mock.Mock(return_value=types.SimpleNamespace(uuid=torch_uuid))
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(torch.cuda, "memory_reserved", return_value=reserved_mib * _MIB)
            )
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_properties", props))
            stack.enter_context(mock.patch.object(torch.version, "cuda", "12.4"))
            stack.enter_context(
                mock.patch.object(GPU_Memory_View, "card_limit_is_not_enforced", return_value=windows)
            )
            stack.enter_context(mock.patch("subprocess.run", run))
            yield

    @staticmethod
    def _smi(used_mib, total_mib=_CARD_TOTAL_MIB):
        return mock.Mock(return_value=_smi_result((_CARD_UUID, used_mib, total_mib)))

    # ---- the reader -------------------------------------------------------------------

    def _read(self, run, *, torch_uuid=None, total_mib=_CARD_TOTAL_MIB):
        with self._card(run=run, torch_uuid=torch_uuid):
            return GPU_Memory_View.device_memory_in_use(torch.device("cuda:0"), total_mib * _MIB)

    def test_reader_picks_the_row_by_uuid_ignoring_prefix_and_case(self):
        run = mock.Mock(
            return_value=_smi_result(
                (_OTHER_CARD_UUID, 1000, _CARD_TOTAL_MIB),
                (_CARD_UUID, 2913, _CARD_TOTAL_MIB),
            )
        )
        # torch reports the same id in capitals; the reader must still find the row.
        reading = self._read(run, torch_uuid=_CARD_UUID.upper())
        self.assertEqual(reading, (2913 * _MIB, _CARD_TOTAL_MIB * _MIB))
        # And without the GPU- prefix, in lower case.
        reading = self._read(run, torch_uuid=_CARD_UUID[4:].lower())
        self.assertEqual(reading, (2913 * _MIB, _CARD_TOTAL_MIB * _MIB))

    def test_reader_uses_the_only_row_when_torch_gives_no_uuid(self):
        run = self._smi(2913)
        self.assertEqual(self._read(run), (2913 * _MIB, _CARD_TOTAL_MIB * _MIB))

    def test_reader_uses_the_only_row_when_its_uuid_does_not_match(self):
        run = self._smi(2913)
        reading = self._read(run, torch_uuid=_OTHER_CARD_UUID)
        self.assertEqual(reading, (2913 * _MIB, _CARD_TOTAL_MIB * _MIB))

    def test_reader_returns_none_for_two_rows_without_a_match(self):
        run = mock.Mock(
            return_value=_smi_result(
                (_OTHER_CARD_UUID, 1000, _CARD_TOTAL_MIB),
                (_CARD_UUID, 2913, _CARD_TOTAL_MIB),
            )
        )
        self.assertIsNone(self._read(run))
        self.assertIsNone(self._read(run, torch_uuid="GPU-ffffffff-0000-4000-8000-000000000000"))

    def test_reader_returns_none_for_unreadable_fields(self):
        for label, row in (
            ("not available", (_CARD_UUID, "[N/A]", _CARD_TOTAL_MIB)),
            ("empty used", (_CARD_UUID, "", _CARD_TOTAL_MIB)),
            ("text used", (_CARD_UUID, "many", _CARD_TOTAL_MIB)),
            ("not available total", (_CARD_UUID, 2913, "[N/A]")),
        ):
            with self.subTest(case=label):
                run = mock.Mock(return_value=_smi_result(row))
                self.assertIsNone(self._read(run))
        with self.subTest(case="no output"):
            self.assertIsNone(self._read(mock.Mock(return_value=_smi_result())))
        with self.subTest(case="short row"):
            run = mock.Mock(
                return_value=subprocess.CompletedProcess([], 0, stdout="2913, 12282\n", stderr="")
            )
            self.assertIsNone(self._read(run))

    def test_reader_returns_none_for_a_non_zero_return_code(self):
        run = mock.Mock(
            return_value=_smi_result((_CARD_UUID, 2913, _CARD_TOTAL_MIB), returncode=9)
        )
        self.assertIsNone(self._read(run))

    def test_reader_returns_none_on_timeout_and_may_try_again_later(self):
        run = mock.Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 5))
        self.assertIsNone(self._read(run))
        self.assertIsNone(self._read(run))
        self.assertEqual(run.call_count, 2)

    def test_reader_never_raises_on_unexpected_errors(self):
        run = mock.Mock(side_effect=ValueError("surprise"))
        self.assertIsNone(self._read(run))

    def test_reader_returns_none_for_a_missing_program_and_never_starts_it_again(self):
        run = mock.Mock(side_effect=FileNotFoundError("nvidia-smi"))
        self.assertIsNone(self._read(run))
        self.assertTrue(GPU_Memory_View._NVIDIA_SMI_MISSING)
        self.assertIsNone(self._read(run))
        self.assertEqual(run.call_count, 1)
        # Even a working program is left alone for the life of the process.
        working = self._smi(2913)
        self.assertIsNone(self._read(working))
        working.assert_not_called()

    def test_reader_returns_none_when_the_total_differs_by_more_than_five_percent(self):
        too_big = int(_CARD_TOTAL_MIB * 1.06)
        too_small = int(_CARD_TOTAL_MIB * 0.94)
        for smi_total in (too_big, too_small, 24564):
            with self.subTest(smi_total=smi_total):
                self.assertIsNone(self._read(self._smi(2913, smi_total)))
        close_enough = int(_CARD_TOTAL_MIB * 1.04)
        self.assertEqual(
            self._read(self._smi(2913, close_enough)),
            (2913 * _MIB, close_enough * _MIB),
        )

    def test_reader_skips_rocm_builds_without_starting_anything(self):
        run = self._smi(2913)
        with self._card(run=run):
            with mock.patch.object(torch.version, "cuda", None):
                reading = GPU_Memory_View.device_memory_in_use(
                    torch.device("cuda:0"), _CARD_TOTAL_MIB * _MIB
                )
        self.assertIsNone(reading)
        run.assert_not_called()

    def test_reader_starts_nvidia_smi_with_a_timeout_and_without_a_console_window(self):
        run = self._smi(2913)
        self.assertIsNotNone(self._read(run))
        args, kwargs = run.call_args
        self.assertEqual(args[0][0], "nvidia-smi")
        self.assertIn("--query-gpu=uuid,memory.used,memory.total", args[0])
        self.assertIn("--format=csv,noheader,nounits", args[0])
        self.assertEqual(kwargs["timeout"], 5)
        self.assertIs(kwargs["capture_output"], True)
        self.assertIs(kwargs["text"], True)
        self.assertIs(kwargs["check"], False)
        if sys.platform == "win32":
            self.assertEqual(kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            self.assertNotIn("creationflags", kwargs)

    # ---- the cache --------------------------------------------------------------------

    def _others(self, run, reserved_mib=3000):
        with self._card(run=run):
            return GPU_Memory_View.other_programs_bytes(
                torch.device("cuda:0"), reserved_mib * _MIB, _CARD_TOTAL_MIB * _MIB
            )

    def test_cache_starts_nvidia_smi_once_per_thirty_seconds(self):
        clock = [1000.0]
        run = self._smi(3000 + 2950)
        with mock.patch("time.monotonic", side_effect=lambda: clock[0]):
            first = self._others(run)
            clock[0] += 29.0
            second = self._others(run)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(first, 2950 * _MIB)
            self.assertEqual(first, second)
            clock[0] += 2.0  # 31 s after the first read
            self.assertEqual(self._others(run), 2950 * _MIB)
            self.assertEqual(run.call_count, 2)

    def test_cache_gives_the_cached_value_for_a_different_reserved_size_within_thirty_seconds(self):
        clock = [1000.0]
        run = self._smi(3000 + 2950)
        with mock.patch("time.monotonic", side_effect=lambda: clock[0]):
            self.assertEqual(self._others(run, reserved_mib=3000), 2950 * _MIB)
            clock[0] += 5.0
            self.assertEqual(self._others(run, reserved_mib=4000), 2950 * _MIB)
            self.assertEqual(run.call_count, 1)
            clock[0] += 30.0
            self.assertEqual(self._others(run, reserved_mib=4000), 1950 * _MIB)
            self.assertEqual(run.call_count, 2)

    def test_cache_remembers_an_unknown_answer_for_the_same_thirty_seconds(self):
        clock = [1000.0]
        run = mock.Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 5))
        with mock.patch("time.monotonic", side_effect=lambda: clock[0]):
            self.assertIsNone(self._others(run))
            clock[0] += 10.0
            self.assertIsNone(self._others(run))
            self.assertEqual(run.call_count, 1)
            clock[0] += 25.0
            self.assertIsNone(self._others(run))
            self.assertEqual(run.call_count, 2)

    # ---- the switch -------------------------------------------------------------------

    def test_switch_is_off_only_for_a_zero_in_the_environment(self):
        for value, expected in (("0", False), (" 0 ", False), ("1", True), ("", True), ("off", True)):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ, {"SSN_GPU_OTHER_PROGRAMS": value}), \
                        mock.patch.object(GPU_Memory_View, "_switch_on", None):
                    self.assertIs(_REAL_SWITCH(), expected)
        with mock.patch.dict(os.environ):
            os.environ.pop("SSN_GPU_OTHER_PROGRAMS", None)
            with mock.patch.object(GPU_Memory_View, "_switch_on", None):
                self.assertIs(_REAL_SWITCH(), True)

    def test_switch_is_read_once(self):
        with mock.patch.dict(os.environ, {"SSN_GPU_OTHER_PROGRAMS": "0"}), \
                mock.patch.object(GPU_Memory_View, "_switch_on", None):
            self.assertIs(_REAL_SWITCH(), False)
            os.environ["SSN_GPU_OTHER_PROGRAMS"] = "1"
            self.assertIs(_REAL_SWITCH(), False)

    def test_switch_off_starts_nothing_calls_no_torch_and_answers_unknown(self):
        run = self._smi(5950)
        device = torch.device("cuda:0")
        with self._card(run=run), \
                mock.patch.object(GPU_Memory_View, "other_programs_switch_on", return_value=False), \
                mock.patch.object(torch.cuda, "memory_reserved") as reserved:
            self.assertIsNone(GPU_Memory_View.device_memory_in_use(device, _CARD_TOTAL_MIB * _MIB))
            self.assertIsNone(
                GPU_Memory_View.other_programs_bytes(device, 3000 * _MIB, _CARD_TOTAL_MIB * _MIB)
            )
            self.assertIsNone(
                GPU_Memory_View.room_beside_other_programs(device, _CARD_TOTAL_MIB * _MIB)
            )
        run.assert_not_called()
        reserved.assert_not_called()
        self.assertEqual(GPU_Memory_View._OTHER_PROGRAMS_CACHE, {})

    # ---- the platform and the margin --------------------------------------------------

    def test_the_limit_is_not_enforced_on_windows_and_under_wsl_only(self):
        decide = GPU_Memory_View.limit_is_not_enforced_on
        for platform_name, release, expected in (
            ("win32", "10", True),
            ("win32", "", True),
            ("linux", "6.18.58.1-microsoft-standard-WSL2", True),
            ("linux", "4.4.0-19041-Microsoft", True),
            ("linux", "6.8.0-45-generic", False),
            ("darwin", "23.5.0", False),
            ("darwin", "6.18.58.1-microsoft-standard-WSL2", False),
            ("linux", None, False),
            ("linux", "", False),
        ):
            with self.subTest(platform=platform_name, release=release):
                self.assertIs(decide(platform_name, release), expected)

    def test_the_public_decision_uses_this_systems_platform_and_kernel_release(self):
        import platform

        self.assertIs(
            GPU_Memory_View.card_limit_is_not_enforced(),
            GPU_Memory_View.limit_is_not_enforced_on(sys.platform, platform.uname().release),
        )

    def test_an_unreadable_kernel_release_counts_as_not_wsl_and_never_raises(self):
        with mock.patch("platform.uname", side_effect=OSError("unreadable")):
            self.assertIs(GPU_Memory_View.card_limit_is_not_enforced(), sys.platform == "win32")

    def test_card_margin_is_five_percent_of_the_total_with_a_floor_of_512_mib(self):
        total = _CARD_TOTAL_MIB * _MIB
        self.assertEqual(GPU_Memory_View.card_margin_bytes(total), int(total * 0.05))
        self.assertGreater(GPU_Memory_View.card_margin_bytes(total), 512 * _MIB)
        self.assertEqual(GPU_Memory_View.card_margin_bytes(4 * 1024 ** 3), 512 * _MIB)
        self.assertEqual(GPU_Memory_View.card_margin_bytes(10 * 1024 ** 3), int(10 * 1024 ** 3 * 0.05))

    # ---- the room beside other programs -----------------------------------------------

    def _room(self, run, *, reserved_mib=0, device="cuda:0", windows=True):
        with self._card(run=run, reserved_mib=reserved_mib, windows=windows):
            return GPU_Memory_View.room_beside_other_programs(
                torch.device(device), _CARD_TOTAL_MIB * _MIB
            )

    def test_room_is_the_total_minus_other_programs_and_the_torch_pool(self):
        # nvidia-smi: 1,000 MiB reserved by torch + 2,868 MiB held by others.
        room = self._room(self._smi(1000 + 2868), reserved_mib=1000)
        self.assertEqual(room, ((_CARD_TOTAL_MIB - 2868 - 1000) * _MIB, 2868 * _MIB))

    def test_room_is_none_off_windows_and_starts_nothing(self):
        run = self._smi(2868)
        self.assertIsNone(self._room(run, windows=False))
        run.assert_not_called()

    def test_room_is_none_for_devices_that_are_not_cuda(self):
        for device in ("cpu", "xpu:0", "mps"):
            with self.subTest(device=device):
                run = self._smi(2868)
                self.assertIsNone(self._room(run, device=device))
                run.assert_not_called()

    def test_room_is_none_when_the_reader_does_not_answer(self):
        self.assertIsNone(self._room(mock.Mock(side_effect=FileNotFoundError("nvidia-smi"))))
        GPU_Memory_View._OTHER_PROGRAMS_CACHE.clear()
        GPU_Memory_View._NVIDIA_SMI_MISSING = False
        self.assertIsNone(
            self._room(mock.Mock(return_value=_smi_result((_CARD_UUID, 2868, _CARD_TOTAL_MIB), returncode=1)))
        )

    def test_room_is_zero_when_other_programs_hold_more_than_the_card(self):
        room = self._room(self._smi(13000))
        self.assertEqual(room, (0, 13000 * _MIB))

    def test_room_is_none_without_an_exception_when_torch_fails(self):
        with self._card(run=self._smi(2868)), \
                mock.patch.object(torch.cuda, "memory_reserved", side_effect=RuntimeError("driver")):
            self.assertIsNone(
                GPU_Memory_View.room_beside_other_programs(torch.device("cuda:0"), _CARD_TOTAL_MIB * _MIB)
            )


if __name__ == "__main__":
    unittest.main()
