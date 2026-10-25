# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Desktop application identity and windows (desktop.Desktop_App): product
names and the Linux desktop identity, file-manager reveal, bringing a window to
the front, the launcher ready marker, and the single-instance channel shared by
the apps, their launchers and desktop.Single_Instance_Probe."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from desktop import Desktop_App as Application_Windows  # noqa: E402
from desktop import Single_Instance_Probe  # noqa: E402


class FakeApplication:
    def __init__(self):
        self.application_name = None
        self.desktop_file_name = None

    def setApplicationName(self, name):
        self.application_name = name

    def setDesktopFileName(self, name):
        self.desktop_file_name = name


class ApplicationIdentityTests(unittest.TestCase):
    def test_canonical_product_and_component_names(self):
        self.assertEqual(Application_Windows.PRODUCT_NAME, "EMAP-SSN")
        self.assertEqual(Application_Windows.APPLICATION_VERSION, "0.4.0")
        self.assertEqual(
            Application_Windows.PRODUCT_LONG_NAME,
            "EMAP-SSN: Embedding- and Multiple-Alignment-integrated Protein "
            "Sequence Similarity Network Platform",
        )
        self.assertEqual(Application_Windows.CONFIG_DISPLAY_NAME, "EMAP-SSN Configuration")
        self.assertEqual(Application_Windows.VIEWER_DISPLAY_NAME, "EMAP-SSN Viewer")
        self.assertEqual(Application_Windows.TOOLS_DISPLAY_NAME, "EMAP-SSN Tools")

    def test_linux_identity_matches_desktop_file_basename(self):
        application = FakeApplication()

        with mock.patch.object(Application_Windows.sys, "platform", "linux"):
            Application_Windows.configure_linux_qt_desktop_identity(
                application, Application_Windows.VIEWER_DESKTOP_FILE_NAME
            )

        self.assertEqual(application.application_name, "emapssn")
        self.assertEqual(application.desktop_file_name, "emapssn")

    def test_non_linux_platform_is_unchanged(self):
        application = FakeApplication()

        with mock.patch.object(Application_Windows.sys, "platform", "darwin"):
            Application_Windows.configure_linux_qt_desktop_identity(
                application, Application_Windows.VIEWER_DESKTOP_FILE_NAME
            )

        self.assertIsNone(application.application_name)
        self.assertIsNone(application.desktop_file_name)

    def test_installer_generates_matching_wm_classes(self):
        installer = (PROJECT_ROOT / "install.sh").read_text(encoding="utf-8")

        self.assertIn(
            f"StartupWMClass={Application_Windows.VIEWER_DESKTOP_FILE_NAME}\n", installer
        )
        self.assertIn(
            f"StartupWMClass={Application_Windows.TOOLS_DESKTOP_FILE_NAME}\n", installer
        )


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

    def test_file_manager_uses_the_platform_shell_when_qt_declines(self):
        # Each platform's fallback runs on any host: the module sees a
        # stand-in os, because os.startfile exists only on Windows.
        # The POSIX openers get /dev/null for every stream: the file manager they start outlives
        # the call and must not hold the pipes of the Console capture.
        target = os.path.abspath("output")
        devnull = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        for os_name, platform, expected in (
            ("nt", "win32", ("startfile", target)),
            ("posix", "darwin", ("Popen", ["open", target])),
            ("posix", "linux", ("Popen", ["xdg-open", target])),
        ):
            startfile = mock.Mock()
            stand_in_os = SimpleNamespace(name=os_name, path=os.path, startfile=startfile)
            with self.subTest(platform=platform), mock.patch(
                "PySide6.QtGui.QDesktopServices.openUrl", return_value=False
            ), mock.patch.object(Application_Windows, "os", stand_in_os), mock.patch.object(
                Application_Windows.sys, "platform", platform
            ), mock.patch("subprocess.Popen") as popen:
                self.assertTrue(Application_Windows.open_in_file_manager("output"))
                called = startfile if expected[0] == "startfile" else popen
                unused = popen if called is startfile else startfile
                if called is popen:
                    called.assert_called_once_with(expected[1], **devnull)
                else:
                    called.assert_called_once_with(expected[1])
                unused.assert_not_called()

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

    def test_single_instance_names_are_per_checkout(self):
        # A real Linux check had a second checkout's Config and Tools bring the
        # first checkout's windows forward, where they edit that checkout's
        # settings files. Each checkout now owns its windows.
        name = Application_Windows._single_instance_server_name
        self.assertEqual(Application_Windows.PROJECT_ROOT, PROJECT_ROOT)
        self.assertEqual(name("SSN_Config"), name("SSN_Config", PROJECT_ROOT))
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first, second = Path(first), Path(second)
            (first / "src").mkdir()
            self.assertNotEqual(name("SSN_Config", first), name("SSN_Config", second))
            self.assertNotEqual(name("SSN_Config", first), name("SSN_Config"))
            # One checkout, however its path is spelled, keeps one channel.
            self.assertEqual(name("SSN_Config", first), name("SSN_Config", first / "src" / ".."))
            if os.name == "nt":
                self.assertEqual(name("SSN_Config", first), name("SSN_Config", Path(str(first).upper())))
            link = second / "link"
            try:
                os.symlink(first, link, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    raise
                import _winapi  # symlinks need a privilege on Windows; junctions do not

                _winapi.CreateJunction(str(first), str(link))
            self.assertEqual(name("SSN_Config", first), name("SSN_Config", link))
            # Still per user as well.
            with mock.patch.object(Application_Windows.Path, "home", return_value=second):
                other_user = name("SSN_Config", first)
            self.assertNotEqual(other_user, name("SSN_Config", first))

    def test_handover_message_names_the_window_and_its_checkout(self):
        message = Application_Windows.instance_handover_message("EMAP-SSN Tools")
        self.assertTrue(message.startswith("EMAP-SSN Tools is already open from this checkout ("), message)
        self.assertIn(f"({PROJECT_ROOT})", message)

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


class SingleInstanceContractTests(unittest.TestCase):
    """The launcher probe and each app must agree on the instance channel."""

    def test_probe_ids_match_the_apps_their_launchers_start(self):
        self.assertEqual(
            Single_Instance_Probe.APPLICATION_IDS,
            {"viewer": "SSN_Config", "tools": "SSN_Tools"},
        )
        launchers = {
            "EMAPSSN.bat": ("viewer", "EMAPSSN_Config.py"),
            "EMAPSSN.sh": ("viewer", "EMAPSSN_Config.py"),
            "EMAPSSN_Tools.bat": ("tools", "EMAPSSN_Tools.py"),
            "EMAPSSN_Tools.sh": ("tools", "EMAPSSN_Tools.py"),
        }
        for launcher, (kind, application) in launchers.items():
            with self.subTest(launcher=launcher):
                launcher_source = (SRC_DIR / "bin" / launcher).read_text(
                    encoding="utf-8"
                )
                application_source = (SRC_DIR / application).read_text(
                    encoding="utf-8"
                )
                self.assertIn(application, launcher_source)
                self.assertEqual(
                    re.findall(
                        r'Single_Instance_Probe\.py"?\s+(\w+)', launcher_source
                    ),
                    [kind],
                )
                # The probe must knock on the channel the started app listens on.
                self.assertEqual(
                    re.findall(
                        r'SingleInstanceController\(\s*"([^"]+)"',
                        application_source,
                    ),
                    [Single_Instance_Probe.APPLICATION_IDS[kind]],
                )

    def test_probe_main_notifies_the_requested_application(self):
        for notified, exit_code in ((True, 0), (False, 1)):
            stdout = io.StringIO()
            with self.subTest(notified=notified), mock.patch.object(
                Single_Instance_Probe, "QCoreApplication"
            ), mock.patch.object(
                Single_Instance_Probe,
                "notify_existing_instance",
                return_value=notified,
            ) as notify, redirect_stdout(stdout):
                self.assertEqual(Single_Instance_Probe.main(["tools"]), exit_code)

            notify.assert_called_once_with("SSN_Tools")
            # Only a hand-over is announced, naming the window it went to.
            self.assertEqual(
                stdout.getvalue(),
                Application_Windows.instance_handover_message("EMAP-SSN Tools") + "\n" if notified else "",
            )

    def test_stale_channel_after_a_crash_is_removed_and_claimed(self):
        server = mock.Mock()
        server.listen.side_effect = [False, True]
        lock = mock.Mock()
        lock.tryLock.return_value = True
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer", return_value=server
        ) as local_server, mock.patch.object(
            Application_Windows.QtCore, "QLockFile", return_value=lock
        ):
            controller = Application_Windows.SingleInstanceController("stale-test")
            self.assertTrue(controller.acquire_or_notify())

        local_server.removeServer.assert_called_once_with(controller.server_name)
        self.assertEqual(server.listen.call_count, 2)
        self.assertTrue(controller.owns_server)
        lock.unlock.assert_not_called()

    def test_unclaimable_channel_releases_the_lock_and_raises(self):
        server = mock.Mock()
        server.listen.side_effect = [False, False]
        server.errorString.return_value = "address in use"
        lock = mock.Mock()
        lock.tryLock.return_value = True
        with mock.patch.object(
            Application_Windows.QtNetwork, "QLocalServer", return_value=server
        ) as local_server, mock.patch.object(
            Application_Windows.QtCore, "QLockFile", return_value=lock
        ):
            controller = Application_Windows.SingleInstanceController("failed-test")
            with self.assertRaisesRegex(RuntimeError, "address in use"):
                controller.acquire_or_notify()

        local_server.removeServer.assert_called_once_with(controller.server_name)
        lock.unlock.assert_called_once_with()
        self.assertFalse(controller.owns_server)


class ApplicationHandoverTests(unittest.TestCase):
    """Config and Tools say so when they hand over to an open window and exit.

    A real Linux check saw a Config exit silently with code 0 after losing the
    startup race to another window. Each app runs its real __main__ in a child
    process, with a stand-in SingleInstanceController that reports the channel
    as owned, so no real channel is touched and no window is built.
    """

    PROBE = textwrap.dedent(
        """
        import runpy
        import sys

        script = sys.argv[1]
        sys.path.insert(0, sys.argv[2])
        from desktop import Desktop_App

        class OpenAlready:
            def __init__(self, application_id, parent=None):
                print(f"channel: {application_id}", flush=True)

            def acquire_or_notify(self):
                return False

        Desktop_App.SingleInstanceController = OpenAlready
        sys.argv = [script]
        runpy.run_path(script, run_name="__main__")
        print("the app went on after handing over")
        """
    )

    def test_config_and_tools_announce_the_handover(self):
        cases = {
            "EMAPSSN_Config.py": ("SSN_Config", "EMAP-SSN Configuration"),
            "EMAPSSN_Tools.py": ("SSN_Tools", "EMAP-SSN Tools"),
        }
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(
                os.environ,
                QT_QPA_PLATFORM="offscreen",
                PYTHONDONTWRITEBYTECODE="1",
                SSN_APP_SETTINGS_PATH=str(Path(directory) / "missing.json"),
            )
            children = {
                script: subprocess.Popen(
                    [sys.executable, "-c", self.PROBE, str(SRC_DIR / script), str(SRC_DIR)],
                    cwd=str(PROJECT_ROOT), env=environment,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                for script in cases
            }
            results = {}
            for script, child in children.items():
                try:
                    stdout, stderr = child.communicate(timeout=180)
                except subprocess.TimeoutExpired:
                    child.kill()
                    stdout, stderr = child.communicate()
                results[script] = (child.returncode, stdout.decode("utf-8", "replace"),
                                   stderr.decode("utf-8", "replace"))
        for script, (application_id, window_name) in cases.items():
            returncode, stdout, stderr = results[script]
            with self.subTest(script=script):
                self.assertEqual(returncode, 0, stderr)
                lines = stdout.splitlines()
                self.assertIn(f"channel: {application_id}", lines)
                self.assertIn(Application_Windows.instance_handover_message(window_name), lines)
                self.assertNotIn("the app went on after handing over", lines)


if __name__ == "__main__":
    unittest.main()
