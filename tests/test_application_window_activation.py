# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utilities import Application_Windows  # noqa: E402


class ApplicationWindowActivationTests(unittest.TestCase):
    @staticmethod
    def _window():
        window = mock.Mock()
        window.isVisible.return_value = True
        window.isMinimized.return_value = False
        window.windowHandle.return_value = None
        return window

    def test_show_schedules_repeated_foreground_requests(self):
        window = self._window()
        callbacks = []
        with mock.patch.object(
            Application_Windows.QtCore.QTimer,
            "singleShot",
            side_effect=lambda delay, callback: callbacks.append((delay, callback)),
        ):
            Application_Windows.show_window_in_front(window)

        window.show.assert_called_once_with()
        self.assertEqual([delay for delay, _ in callbacks], [0, 100])
        with mock.patch.object(Application_Windows, "_activate_window") as activate, \
                mock.patch.object(Application_Windows, "_signal_launcher_ready") as signal:
            for _, callback in callbacks:
                callback()
        self.assertEqual(activate.call_count, 2)
        self.assertTrue(all(call.args[0] is window for call in activate.call_args_list))
        signal.assert_called_once_with(window)

    def test_activation_raises_without_setting_persistent_window_flags(self):
        window = self._window()
        with mock.patch.object(Application_Windows.sys, "platform", "linux"):
            Application_Windows._activate_window(window)

        window.raise_.assert_called_once_with()
        window.activateWindow.assert_called_once_with()
        window.setWindowFlag.assert_not_called()

    def test_ready_marker_is_atomic_and_environment_is_one_shot(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as temp_dir:
            ready_path = Path(temp_dir) / "gui.ready"
            with mock.patch.dict(
                os.environ, {"SSN_GUI_READY_FILE": str(ready_path)}, clear=False
            ):
                self.assertTrue(Application_Windows._signal_launcher_ready(window))
                self.assertNotIn("SSN_GUI_READY_FILE", os.environ)
            self.assertEqual(ready_path.read_text(encoding="utf-8"), "ready\n")
            self.assertEqual(list(ready_path.parent.glob("*.tmp")), [])

    def test_missing_ready_variable_is_a_noop(self):
        window = self._window()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(Application_Windows._signal_launcher_ready(window))

    def test_marker_failure_is_reported_and_environment_is_removed(self):
        window = self._window()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {"SSN_GUI_READY_FILE": str(Path(temp_dir) / "gui.ready")},
            clear=False,
        ), mock.patch.object(
            Application_Windows.os, "replace", side_effect=OSError("blocked")
        ), mock.patch.object(Application_Windows.sys.stderr, "write"):
            self.assertFalse(Application_Windows._signal_launcher_ready(window))
            self.assertNotIn("SSN_GUI_READY_FILE", os.environ)


if __name__ == "__main__":
    unittest.main()
