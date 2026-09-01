# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

    def test_file_manager_prefers_qt_desktop_services(self):
        with mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            self.assertTrue(Application_Windows.open_in_file_manager("output"))

        open_url.assert_called_once()

    def test_file_manager_uses_windows_shell_when_qt_declines(self):
        with mock.patch(
            "PySide6.QtGui.QDesktopServices.openUrl", return_value=False
        ), mock.patch.object(Application_Windows.os, "startfile") as startfile:
            self.assertTrue(Application_Windows.open_in_file_manager("output"))

        startfile.assert_called_once_with(os.path.abspath("output"))

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

    def test_single_instance_names_are_stable_per_user_and_distinct_per_app(self):
        config_name = Application_Windows._single_instance_server_name("SSN_Config")
        self.assertEqual(
            config_name,
            Application_Windows._single_instance_server_name("SSN_Config"),
        )
        self.assertNotEqual(
            config_name,
            Application_Windows._single_instance_server_name("SSN_Tools"),
        )

    def test_primary_instance_claims_local_server(self):
        server = mock.Mock()
        server.listen.return_value = True
        lock = mock.Mock()
        lock.tryLock.return_value = True
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer", return_value=server
        ), mock.patch.object(
            Application_Windows.QtCore, "QLockFile", return_value=lock
        ):
            controller = Application_Windows.SingleInstanceController("primary-test")
            self.assertTrue(controller.acquire_or_notify())

        self.assertTrue(controller.owns_server)
        lock.tryLock.assert_called_once_with(0)
        server.listen.assert_called_once_with(controller.server_name)
        controller.close()
        self.assertFalse(controller.owns_server)
        server.close.assert_called_once_with()
        lock.unlock.assert_called_once_with()

    def test_duplicate_instance_notifies_owner_and_does_not_claim_server(self):
        server = mock.Mock()
        server.listen.return_value = False
        socket = mock.Mock()
        socket.waitForConnected.return_value = True
        lock = mock.Mock()
        lock.tryLock.return_value = False
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer", return_value=server
        ), mock.patch.object(
            Application_Windows.QtNetwork, "QLocalSocket", return_value=socket
        ), mock.patch.object(
            Application_Windows.QtCore, "QLockFile", return_value=lock
        ):
            controller = Application_Windows.SingleInstanceController("duplicate-test")
            self.assertFalse(controller.acquire_or_notify())

        self.assertFalse(controller.owns_server)
        server.listen.assert_not_called()
        socket.connectToServer.assert_called_once_with(controller.server_name)
        socket.write.assert_called_once_with(b"activate\n")

    def test_lightweight_probe_notifies_without_claiming_an_instance(self):
        socket = mock.Mock()
        socket.waitForConnected.return_value = True
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalSocket", return_value=socket
        ), mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer"
        ) as local_server:
            self.assertTrue(
                Application_Windows.notify_existing_instance("SSN_Config")
            )

        local_server.assert_not_called()
        socket.connectToServer.assert_called_once_with(
            Application_Windows._single_instance_server_name("SSN_Config")
        )
        socket.write.assert_called_once_with(b"activate\n")

    def test_early_duplicate_activation_is_delivered_after_window_is_ready(self):
        server = mock.Mock()
        server.hasPendingConnections.side_effect = [True, False]
        connection = mock.Mock()
        server.nextPendingConnection.return_value = connection
        lock = mock.Mock()
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer", return_value=server
        ), mock.patch.object(
            Application_Windows.QtCore, "QLockFile", return_value=lock
        ):
            controller = Application_Windows.SingleInstanceController("pending-test")
            controller._accept_connections()

        callback = mock.Mock()
        with mock.patch.object(
            Application_Windows.QtCore.QTimer,
            "singleShot",
            side_effect=lambda delay, pending_callback: pending_callback(),
        ):
            controller.set_activation_callback(callback)

        callback.assert_called_once_with()
        connection.disconnectFromServer.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
