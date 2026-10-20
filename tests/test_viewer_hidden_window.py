"""Hidden window mode (utilities.Viewer_Display, desktop.Viewer_Window and their users).

A hidden Viewer keeps Qt's native platform with its window off screen, so it renders
(capture_view, print) and its window can be shown later. Covers: which mode a launch
gets and when a hidden request runs headless; the window controller's show, hide and
foreground reporting with the Windows and macOS calls replaced; the Viewer's one
shows_windows() question; the show/hide route; and the MCP client's launch plumbing,
fallbacks and actions. tests/test_viewer_hidden_launch.py runs a real hidden Viewer.
"""
import asyncio
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from desktop import Viewer_Window  # noqa: E402
from desktop.Viewer_Window import HEADLESS, HIDDEN, NORMAL, MacActivation, ViewerWindow, WindowUnavailable  # noqa: E402
from utilities import Viewer_Display as display  # noqa: E402
from mcp_server.viewer import Viewer_Client  # noqa: E402
from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerError  # noqa: E402
from tests.viewer_fixtures import SnapshotHTTPFixture  # noqa: E402

DISPLAY = {"DISPLAY": ":0"}
WAYLAND = {"WAYLAND_DISPLAY": "wayland-0"}


class DisplayResolverTests(unittest.TestCase):
    def test_linux_needs_an_x_server_or_a_wayland_compositor(self):
        reason = display.display_unavailable_reason({}, "linux")
        self.assertIn("no display was found", reason)
        self.assertIn("DISPLAY and WAYLAND_DISPLAY", reason)
        self.assertIsNone(display.display_unavailable_reason(DISPLAY, "linux"))
        # Qt chooses Wayland here, and the Viewer's own platform code leaves it alone.
        self.assertIsNone(display.display_unavailable_reason(WAYLAND, "linux"))
        self.assertIsNone(display.display_unavailable_reason({**DISPLAY, **WAYLAND, "XDG_SESSION_TYPE": "wayland"}, "linux"))

    def test_a_wayland_session_without_xwayland_cannot_run_the_viewers_opengl(self):
        # EMAPSSN_Viewer._configure_linux_vispy_platform moves Wayland to xcb, which needs DISPLAY.
        for environment in ({**WAYLAND, "XDG_SESSION_TYPE": "wayland"}, {**WAYLAND, "QT_QPA_PLATFORM": "wayland"}):
            with self.subTest(environment=environment):
                self.assertIn("XWayland", display.display_unavailable_reason(environment, "linux"))
        self.assertIsNone(display.display_unavailable_reason({**WAYLAND, "QT_QPA_PLATFORM": "xcb", "DISPLAY": ":1"}, "linux"))

    def test_windows_has_no_such_case_and_macos_needs_a_graphical_session(self):
        self.assertIsNone(display.display_unavailable_reason({}, "win32"))
        for session, expected in (("Aqua", None), (None, None), ("Background", "launchd session 'Background'")):
            with self.subTest(session=session):
                reason = display.display_unavailable_reason({}, "darwin", macos_session=lambda session=session: session)
                if expected is None:
                    self.assertIsNone(reason)
                else:
                    self.assertIn(expected, reason)

    def test_resolve_window_mode(self):
        self.assertEqual(display.resolve_window_mode("normal", {}, "linux"), ("normal", None))
        self.assertEqual(display.resolve_window_mode("headless", DISPLAY, "linux"), ("headless", None))
        self.assertEqual(display.resolve_window_mode("hidden", DISPLAY, "linux"), ("hidden", None))
        mode, reason = display.resolve_window_mode("hidden", {}, "linux")
        self.assertEqual(mode, "headless")
        self.assertIn("no display", reason)

    def test_a_hidden_environment_keeps_the_native_platform(self):
        environment = {"SSN_VIEWER_WINDOW": "hidden", "QT_QPA_PLATFORM": "offscreen", **DISPLAY}
        self.assertEqual(display.apply_hidden_environment(environment, "linux"), "hidden")
        # An inherited offscreen platform is dropped, as for a normal Viewer.
        self.assertNotIn("QT_QPA_PLATFORM", environment)
        self.assertNotIn(display.FALLBACK_ENV, environment)
        environment = {"SSN_VIEWER_WINDOW": "hidden", "QT_QPA_PLATFORM": "xcb", **DISPLAY}
        display.apply_hidden_environment(environment, "linux")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "xcb")

    def test_a_hidden_request_without_a_display_runs_offscreen_and_says_why(self):
        environment = {"SSN_VIEWER_WINDOW": "hidden"}
        self.assertEqual(display.apply_hidden_environment(environment, "linux"), "headless")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")
        self.assertIn("no display was found", environment[display.FALLBACK_ENV])

    def test_nothing_changes_unless_a_hidden_viewer_was_asked_for(self):
        for environment in ({}, {"SSN_VIEWER_WINDOW": "normal"}, {"SSN_VIEWER_HEADLESS": "1"},
                            {"SSN_VIEWER_WINDOW": "hidden", "SSN_VIEWER_HEADLESS": "1"},
                            {"SSN_VIEWER_WINDOW": "hidden", "SSN_VIEWER_HEADLESS": "true"}):
            with self.subTest(environment=environment):
                before = dict(environment)
                self.assertIsNone(display.apply_hidden_environment(environment, "linux"))
                self.assertEqual(environment, before)

    def test_macos_hidden_viewers_start_without_a_dock_icon(self):
        environment = {"SSN_VIEWER_WINDOW": "hidden"}
        with mock.patch.object(display, "_macos_session_name", return_value="Aqua"):
            self.assertEqual(display.apply_hidden_environment(environment, "darwin"), "hidden")
        self.assertEqual(environment[display.MAC_NO_DOCK_ENV], "1")
        # Windows and Linux never get it; a value the user set stays.
        for platform in ("win32", "linux"):
            environment = {"SSN_VIEWER_WINDOW": "hidden", **DISPLAY}
            display.apply_hidden_environment(environment, platform)
            self.assertNotIn(display.MAC_NO_DOCK_ENV, environment)
        environment = {"SSN_VIEWER_WINDOW": "hidden", display.MAC_NO_DOCK_ENV: "0"}
        with mock.patch.object(display, "_macos_session_name", return_value="Aqua"):
            display.apply_hidden_environment(environment, "darwin")
        self.assertEqual(environment[display.MAC_NO_DOCK_ENV], "0")

    def test_a_macos_ssh_login_falls_back_offscreen(self):
        environment = {"SSN_VIEWER_WINDOW": "hidden"}
        with mock.patch.object(display, "_macos_session_name", return_value="Background"):
            self.assertEqual(display.apply_hidden_environment(environment, "darwin"), "headless")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")
        self.assertNotIn(display.MAC_NO_DOCK_ENV, environment)

    def test_the_viewer_module_applies_it_before_qt_loads(self):
        import EMAPSSN_Viewer
        environment = {"SSN_VIEWER_WINDOW": "hidden"}
        self.assertEqual(EMAPSSN_Viewer._configure_hidden_platform(environment, "linux"), "headless")
        self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")
        # The headless switches of the Viewer are what they were.
        environment = {"SSN_VIEWER_HEADLESS": "1"}
        self.assertTrue(EMAPSSN_Viewer._configure_headless_platform(["EMAPSSN_Viewer.py"], environment))
        self.assertEqual(environment["QT_QPA_PLATFORM"], "offscreen")


class FakeNative:
    """Stands in for WindowsNative: records the calls, and says which window is in front."""

    def __init__(self):
        self.calls = []
        self.foreground_handle = 0
        self.topmost_result = True

    def topmost(self, hwnd):
        self.calls.append(("topmost", hwnd))
        return self.topmost_result

    def not_topmost(self, hwnd):
        self.calls.append(("not_topmost", hwnd))
        return True

    def foreground(self):
        return self.foreground_handle


class FakeMac:
    def __init__(self):
        self.calls = []

    def accessory(self):
        self.calls.append("accessory")

    def regular(self):
        self.calls.append("regular")

    def order_front(self, view):
        self.calls.append(("order_front", view))


class QtTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def pump(self, count=3):
        for _ in range(count):
            self.app.processEvents()


class WindowControllerTests(QtTestCase):
    def make(self, mode=HIDDEN, platform="win32", **extra):
        window = QtWidgets.QMainWindow()
        window.setCentralWidget(QtWidgets.QWidget())
        window.resize(300, 200)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        native = FakeNative() if platform == "win32" else None
        alerts = []
        control = ViewerWindow(window, mode, platform=platform, native=native,
                               alert=lambda target, ms: alerts.append((target, ms)), **extra)
        return control, window, native, alerts

    def dont_show(self, window):
        return window.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen)

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            ViewerWindow(QtWidgets.QMainWindow(), "invisible")

    def test_a_hidden_window_is_open_for_qt_but_not_on_screen(self):
        control, window, native, alerts = self.make()
        control.open()
        self.assertTrue(self.dont_show(window))
        self.assertTrue(window.isVisible())   # shown for Qt: the canvas gets its context
        state = control.state()
        self.assertEqual((state["mode"], state["requested_mode"], state["visible"], state["foreground"]),
                         (HIDDEN, HIDDEN, False, False))
        self.assertTrue(state["can_show"] and state["can_hide"])
        self.assertFalse(control.shows_windows())
        self.assertEqual(native.calls, [])
        self.assertEqual(alerts, [])

    def test_show_puts_the_window_on_screen_on_top_without_the_keyboard(self):
        control, window, native, alerts = self.make()
        control.open()
        hwnd = int(window.winId())
        state = control.show()
        self.assertFalse(self.dont_show(window))
        self.assertFalse(window.testAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertTrue(control.shows_windows())
        self.assertEqual((state["visible"], state["foreground"], state["on_top_until_activated"]), (True, False, True))
        self.assertEqual(native.calls, [("topmost", hwnd)])
        self.assertEqual(alerts, [(window, 0)])   # the taskbar button flashes

    def test_a_window_the_platform_made_foreground_is_reported_and_not_pinned(self):
        control, window, native, alerts = self.make()
        control.open()
        native.foreground_handle = int(window.winId())
        state = control.show()
        self.assertTrue(state["foreground"])
        self.assertFalse(state["on_top_until_activated"])
        self.assertEqual(native.calls, [])

    def test_the_pin_ends_when_the_user_takes_the_window_not_when_qt_reports_activation(self):
        control, window, native, _ = self.make()
        control.open()
        hwnd = int(window.winId())
        control.show()
        self.pump()   # Qt may report an activation although another program keeps the foreground
        QtWidgets.QApplication.sendEvent(window, QtCore.QEvent(QtCore.QEvent.Type.WindowActivate))
        self.assertTrue(control.state()["on_top_until_activated"])
        self.assertNotIn(("not_topmost", hwnd), native.calls)
        native.foreground_handle = hwnd
        QtWidgets.QApplication.sendEvent(window, QtCore.QEvent(QtCore.QEvent.Type.WindowActivate))
        self.assertFalse(control.state()["on_top_until_activated"])
        self.assertEqual(native.calls, [("topmost", hwnd), ("not_topmost", hwnd)])

    def test_a_click_in_the_window_ends_the_pin(self):
        control, window, native, _ = self.make()
        control.open()
        hwnd = int(window.winId())
        control.show()
        self.press_key(window.centralWidget())
        self.assertFalse(control.state()["on_top_until_activated"])
        self.assertEqual(native.calls[-1], ("not_topmost", hwnd))

    def press_key(self, widget):
        from PySide6.QtGui import QKeyEvent
        event = QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_A, QtCore.Qt.KeyboardModifier.NoModifier, "a")
        QtWidgets.QApplication.sendEvent(widget, event)

    def test_a_click_elsewhere_does_not_end_it(self):
        control, window, native, _ = self.make()
        control.open()
        control.show()
        other = QtWidgets.QWidget()
        self.addCleanup(other.deleteLater)
        self.press_key(other)
        self.assertTrue(control.state()["on_top_until_activated"])

    def test_a_platform_that_refuses_topmost_leaves_the_window_unpinned(self):
        control, window, native, _ = self.make()
        control.open()
        native.topmost_result = False
        self.assertFalse(control.show()["on_top_until_activated"])

    def test_hide_takes_the_window_off_screen_and_keeps_it_alive(self):
        control, window, native, _ = self.make()
        control.open()
        hwnd = int(window.winId())
        control.show()
        state = control.hide()
        self.assertTrue(self.dont_show(window))
        self.assertTrue(window.isVisible())
        self.assertEqual((state["visible"], state["on_top_until_activated"]), (False, False))
        self.assertFalse(control.shows_windows())
        self.assertEqual(native.calls, [("topmost", hwnd), ("not_topmost", hwnd)])
        # It can come back, and hiding what is hidden changes nothing.
        self.assertEqual(control.hide(), state)
        self.assertTrue(control.show()["visible"])
        self.assertEqual(control.show()["visible"], True)

    def test_a_minimized_shown_window_is_restored_by_show(self):
        control, window, native, _ = self.make(platform="linux")
        control.open()
        control.show()
        window.showMinimized()
        self.pump()
        self.assertTrue(control.state()["minimized"])
        self.assertFalse(control.show()["minimized"])

    def test_a_headless_window_cannot_be_shown_or_hidden(self):
        control, window, native, _ = self.make(mode=HEADLESS, requested=HIDDEN,
                                               fallback_reason="no display was found (DISPLAY and WAYLAND_DISPLAY are not set)")
        control.open()
        self.assertFalse(window.isVisible())
        self.assertFalse(control.shows_windows())
        for action in (control.show, control.hide):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(WindowUnavailable, "no display was found"):
                action()
        state = control.state()
        self.assertEqual((state["mode"], state["requested_mode"], state["can_show"], state["can_hide"]),
                         (HEADLESS, HIDDEN, False, False))
        self.assertIn("no display", state["fallback_reason"])
        plain, *_ = self.make(mode=HEADLESS)
        with self.assertRaisesRegex(WindowUnavailable, "Start it with mode='hidden'"):
            plain.show()

    def test_a_normal_window_shows_and_is_never_hidden(self):
        control, window, native, _ = self.make(mode=NORMAL)
        with mock.patch("desktop.Desktop_App.show_window_in_front") as shown:
            control.open()
        shown.assert_called_once_with(window)
        window.show()
        self.assertTrue(control.shows_windows())
        self.assertTrue(control.show()["visible"])
        with self.assertRaisesRegex(WindowUnavailable, "normal Viewer"):
            control.hide()

    def test_other_platforms_report_the_foreground_from_qt(self):
        control, window, native, alerts = self.make(platform="linux")
        control.open()
        state = control.show()
        self.assertIsNone(native)
        self.assertIn(state["foreground"], (True, False))
        self.assertFalse(state["on_top_until_activated"])   # only Windows pins
        self.assertEqual(alerts, [(window, 0)])

    def test_a_failing_alert_never_stops_show(self):
        control, window, native, _ = self.make()
        control._alert = mock.Mock(side_effect=RuntimeError("no taskbar"))
        control.open()
        with redirect_stdout(io.StringIO()) as output:
            self.assertTrue(control.show()["visible"])
        self.assertIn("could not flash", output.getvalue())


class MacActivationTests(QtTestCase):
    class Runtime:
        def __init__(self, policy_result=True, fail_on=None):
            self.calls = []
            self.policy_result = policy_result
            self.fail_on = fail_on

        def cls(self, name):
            self.calls.append(("cls", name))
            return 100

        def send(self, receiver, selector, *args, restype=None, argtypes=()):
            self.calls.append((receiver, selector, args))
            if selector == self.fail_on:
                raise OSError("objc failed")
            return {"sharedApplication": 200, "setActivationPolicy:": self.policy_result, "window": 300}.get(selector)

    def test_the_activation_policy_calls_in_order(self):
        runtime = self.Runtime()
        mac = MacActivation(runtime)
        self.assertTrue(mac.accessory())
        self.assertTrue(mac.regular())
        self.assertEqual(runtime.calls, [
            ("cls", "NSApplication"), (100, "sharedApplication", ()), (200, "setActivationPolicy:", (1,)),
            ("cls", "NSApplication"), (100, "sharedApplication", ()), (200, "setActivationPolicy:", (0,)),
        ])

    def test_the_window_is_ordered_front_without_being_made_key(self):
        runtime = self.Runtime()
        self.assertTrue(MacActivation(runtime).order_front(7))
        self.assertEqual(runtime.calls, [(7, "window", ()), (300, "orderFrontRegardless", ())])

    def test_every_failure_only_reports_and_never_raises(self):
        reports = []
        for runtime, call in ((self.Runtime(policy_result=False), lambda mac: mac.accessory()),
                              (self.Runtime(fail_on="setActivationPolicy:"), lambda mac: mac.regular()),
                              (self.Runtime(fail_on="sharedApplication"), lambda mac: mac.accessory()),
                              (self.Runtime(fail_on="window"), lambda mac: mac.order_front(7))):
            with self.subTest(runtime=runtime.fail_on, policy=runtime.policy_result):
                self.assertFalse(call(MacActivation(runtime, report=reports.append)))
        self.assertEqual(len(reports), 4)
        self.assertTrue(all(text.startswith("Warning:") for text in reports))

    def test_a_missing_objc_runtime_is_a_warning(self):
        reports = []
        with mock.patch.object(Viewer_Window.ctypes.util, "find_library", return_value=None):
            self.assertFalse(MacActivation(report=reports.append).accessory())
        self.assertIn("libobjc", reports[0])

    def test_the_dock_icon_follows_the_window(self):
        window = QtWidgets.QMainWindow()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        mac = FakeMac()
        control = ViewerWindow(window, HIDDEN, platform="darwin", mac=mac, alert=lambda *args: None)
        control.open()
        self.assertEqual(mac.calls, ["accessory"])
        control.show()
        self.assertEqual(mac.calls, ["accessory", "regular", ("order_front", int(window.winId()))])
        control.hide()
        self.assertEqual(mac.calls[-1], "accessory")
        # Showing a shown window again does not change the policy twice.
        control.show()
        control.show()
        self.assertEqual(mac.calls.count("regular"), 2)

    def test_a_normal_or_headless_window_never_touches_the_dock(self):
        for mode in (NORMAL, HEADLESS):
            window = QtWidgets.QMainWindow()
            self.addCleanup(window.deleteLater)
            mac = FakeMac()
            with mock.patch("desktop.Desktop_App.show_window_in_front"):
                ViewerWindow(window, mode, platform="darwin", mac=mac, alert=lambda *args: None).open()
            self.assertEqual(mac.calls, [], mode)


class ViewerQuestionTests(QtTestCase):
    """MainViewer.shows_windows() and what asks it."""

    def viewer(self, **attributes):
        import EMAPSSN_Viewer
        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        viewer.__dict__.update(attributes)
        return viewer

    def test_headless_is_never_a_user_present(self):
        control = SimpleNamespace(shows_windows=lambda: True)
        self.assertFalse(self.viewer(headless=True, window_control=control).shows_windows())
        self.assertTrue(self.viewer(headless=False, window_control=control).shows_windows())
        self.assertFalse(self.viewer(headless=False, window_control=SimpleNamespace(shows_windows=lambda: False)).shows_windows())
        # Before the window exists, a normal Viewer is a user present.
        self.assertTrue(self.viewer(headless=False).shows_windows())

    def test_a_hidden_viewer_does_not_slide_its_sidebar_until_shown(self):
        import EMAPSSN_Viewer
        from PySide6.QtWidgets import QWidget
        from tests.theme_fixture import apply_theme_for_class
        apply_theme_for_class(type(self), self.app)
        viewer = self.viewer(headless=False, window_mode=HIDDEN, requested_window_mode=HIDDEN)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(1200, 800))
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        self.assertEqual(viewer.window_control.mode, HIDDEN)
        viewer.window_control = ViewerWindow(viewer.main_window, HIDDEN, platform="linux", alert=lambda *args: None)
        viewer.window_control.open()
        self.pump()
        self.assertTrue(viewer.main_window.isVisible())
        self.assertFalse(viewer._sidebar_can_slide())
        viewer.window_control.show()
        self.assertTrue(viewer._sidebar_can_slide())
        viewer.window_control.hide()
        self.assertFalse(viewer._sidebar_can_slide())

    def test_a_viewer_built_by_hand_keeps_its_old_modes(self):
        import EMAPSSN_Viewer
        from PySide6.QtWidgets import QWidget
        from tests.theme_fixture import apply_theme_for_class
        apply_theme_for_class(type(self), self.app)
        for headless, mode in ((False, NORMAL), (True, HEADLESS)):
            viewer = self.viewer(headless=headless)
            viewer.canvas = SimpleNamespace(native=QWidget(), size=(1200, 800))
            viewer._build_main_window()
            self.addCleanup(viewer.main_window.deleteLater)
            self.assertEqual((viewer.window_control.mode, viewer.window_control.requested), (mode, mode))
            self.assertEqual(viewer.window_control.state()["can_show"], not headless)

    def test_the_command_portal_asks_the_viewer_not_the_environment(self):
        from Viewer_Command_Portal import viewer_shows_windows
        for environment in ({"QT_QPA_PLATFORM": "offscreen"}, {"SSN_VIEWER_HEADLESS": "1"}, {}):
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment):
                self.assertTrue(viewer_shows_windows(SimpleNamespace(shows_windows=lambda: True)))
                self.assertFalse(viewer_shows_windows(SimpleNamespace(shows_windows=lambda: False)))
        # A stand-in without the method is headless when the environment says so, as before.
        with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}):
            self.assertFalse(viewer_shows_windows(SimpleNamespace()))
        with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "", "SSN_VIEWER_HEADLESS": "1"}):
            self.assertFalse(viewer_shows_windows(SimpleNamespace()))
        with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "windows", "SSN_VIEWER_HEADLESS": "0"}):
            self.assertTrue(viewer_shows_windows(SimpleNamespace()))


class WindowRouteTests(SnapshotHTTPFixture, unittest.TestCase):
    """POST /api/mcp/v1/window and the window block of the session and summary."""

    class Control:
        mode, requested, fallback_reason = HIDDEN, HIDDEN, None

        def __init__(self):
            self.calls = []
            self.visible = False

        def state(self):
            return {"mode": self.mode, "visible": self.visible, "foreground": False}

        def show(self):
            self.calls.append("show")
            self.visible = True
            return self.state()

        def hide(self):
            self.calls.append("hide")
            self.visible = False
            return self.state()

    def setUp(self):
        super().setUp()
        self.control = self.Control()
        self.viewer.window_control = self.control
        self.viewer.window_mode = HIDDEN
        self.viewer.requested_window_mode = HIDDEN
        self.viewer.window_fallback_reason = None
        self.base = self.url.rsplit("/", 1)[0]
        patcher = mock.patch("web_ui.Web_Server.WINDOW_SETTLE_SECONDS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fetch(self, path, body=None, token=True):
        """GET (no body) or POST ``path`` under /api/mcp/v1 on the live server; the Qt thread keeps running."""
        import threading
        import time
        import urllib.error
        import urllib.request
        result = {}

        def work():
            headers = {"Authorization": "Bearer " + self.server.inspection_token} if token else {}
            data = json.dumps(body).encode("utf-8") if body is not None else None
            request = urllib.request.Request(self.base + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    result.update(status=response.status, payload=json.loads(response.read()))
            except urllib.error.HTTPError as error:
                result.update(status=error.code, payload=json.loads(error.read()))

        thread = threading.Thread(target=work)
        thread.start()
        deadline = time.monotonic() + 12
        while thread.is_alive() and time.monotonic() < deadline:
            self.app.processEvents()
            thread.join(0.01)
        thread.join(0.1)
        self.assertFalse(thread.is_alive())
        return result

    def window(self, action, token=True):
        return self.fetch("/window", {"action": action}, token=token)

    def test_show_and_hide_return_the_window_state(self):
        shown = self.window("show")
        self.assertEqual((shown["status"], shown["payload"]["visible"]), (200, True))
        hidden = self.window("hide")
        self.assertEqual((hidden["status"], hidden["payload"]["visible"]), (200, False))
        self.assertEqual(self.control.calls, ["show", "hide"])

    def test_the_route_is_authenticated_and_checks_its_request(self):
        self.assertEqual(self.window("show", token=False)["status"], 401)
        self.assertEqual(self.control.calls, [])
        for action in ("state", "close", ""):
            with self.subTest(action=action):
                bad = self.window(action)
                self.assertEqual(bad["status"], 400)
                self.assertIn("Unknown window action", bad["payload"]["error"])
        status, _, payload = self.exchange("POST", "/api/mcp/v1/window", {"Content-Length": "5000"}, b"x" * 5000)
        self.assertEqual(status, 401)
        self.assertEqual(self.control.calls, [])

    def test_a_window_that_cannot_be_shown_is_a_clear_400(self):
        def refuse():
            raise WindowUnavailable("This Viewer runs headless (Qt's offscreen platform), which has no window to show.")
        self.control.show = refuse
        result = self.window("show")
        self.assertEqual(result["status"], 400)
        self.assertIn("no window to show", result["payload"]["error"])

    def test_a_viewer_without_a_window_controller_says_so(self):
        del self.viewer.window_control
        result = self.window("hide")
        self.assertEqual(result["status"], 400)
        self.assertIn("no window", result["payload"]["error"])

    def test_the_session_names_the_capability_and_mode(self):
        outcome = self.fetch("/session")["payload"]
        self.assertIn("window_v1", outcome["inspection_capabilities"])
        self.assertEqual((outcome["window_mode"], outcome["window_requested_mode"], outcome["window_fallback_reason"]),
                         (HIDDEN, HIDDEN, None))

    def test_the_summary_carries_the_window_state(self):
        service = self.viewer.viewer_inspection
        expected = {"mode": HIDDEN, "visible": False, "foreground": False}
        self.assertEqual(service.window_state(), expected)
        self.assertEqual(service.get_summary()["window"], expected)
        snapshot = service.capture_snapshot()
        self.assertEqual(service.snapshots.execute("get_summary", snapshot)["window"], expected)
        del self.viewer.window_control
        self.assertIsNone(service.window_state())


class ClientLaunchTests(unittest.IsolatedAsyncioTestCase):
    """MCPViewerClient launches in mode hidden, falls back, and shows or hides."""

    def settings(self):
        return mock.patch("mcp_server.viewer.Viewer_Client.validate_viewer_document",
                          return_value={"inputs": {"TARGET_CACHE_PATH": "test"}})

    async def launch(self, mode, platform):
        """Run launch_session with Popen replaced; return (popen call, the error)."""
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(Path(temporary) / "sessions"),
                                             "QT_QPA_PLATFORM": "offscreen", "SSN_VIEWER_WINDOW": "stale",
                                             "SSN_VIEWER_WINDOW_FALLBACK": "stale"}), \
                self.settings(), \
                mock.patch.object(Viewer_Client, "_identity", return_value=None), \
                mock.patch.object(Viewer_Client, "sys", SimpleNamespace(platform=platform, executable=sys.executable)), \
                mock.patch.object(Viewer_Client.subprocess, "Popen") as popen:
            if platform == "win32":
                popen.return_value.wait.return_value = 1  # the broker fails; nothing real starts
            with self.assertRaises(MCPViewerError) as caught:
                await MCPViewerClient(Path(temporary)).launch_session(settings_document={}, mode=mode)
            return popen.call_args, str(caught.exception)

    async def test_a_hidden_launch_asks_config_for_hidden_without_a_terminal(self):
        call, _ = await self.launch("hidden", "linux")
        command, env = call.args[0], call.kwargs["env"]
        self.assertEqual(command[command.index("--viewer-mode") + 1], "hidden")
        self.assertFalse(any("Viewer_Terminal" in part for part in command))
        self.assertEqual(env["SSN_VIEWER_WINDOW"], "hidden")
        self.assertNotIn("QT_QPA_PLATFORM", env)   # an inherited offscreen platform is dropped
        self.assertNotIn("SSN_VIEWER_HEADLESS", env)
        self.assertNotIn("SSN_VIEWER_WINDOW_FALLBACK", env)
        self.assertTrue(call.kwargs["start_new_session"])

    async def test_headless_and_normal_launches_are_what_they_were(self):
        call, _ = await self.launch("headless", "linux")
        command, env = call.args[0], call.kwargs["env"]
        self.assertEqual(command[command.index("--viewer-mode") + 1], "headless")
        self.assertEqual((env["QT_QPA_PLATFORM"], env["SSN_VIEWER_HEADLESS"]), ("offscreen", "1"))
        self.assertNotIn("SSN_VIEWER_WINDOW", env)
        self.assertNotIn("SSN_VIEWER_WINDOW_FALLBACK", env)
        with mock.patch("utilities.Terminal_Launcher.launch_in_terminal") as terminal:
            terminal.return_value.poll.return_value = 1
            terminal.return_value.pid = 1
            with tempfile.TemporaryDirectory() as temporary, \
                    mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(Path(temporary) / "s"),
                                                 "QT_QPA_PLATFORM": "offscreen", "SSN_VIEWER_WINDOW": "stale"}), \
                    self.settings(), mock.patch.object(Viewer_Client, "_identity", return_value=None), \
                    mock.patch.object(Viewer_Client, "sys", SimpleNamespace(platform="linux", executable=sys.executable)):
                with self.assertRaises(MCPViewerError):
                    await MCPViewerClient(Path(temporary)).launch_session(settings_document={}, mode="normal")
            env = terminal.call_args.kwargs["env"]
            self.assertNotIn("SSN_VIEWER_WINDOW", env)
            self.assertNotIn("QT_QPA_PLATFORM", env)

    @unittest.skipUnless(sys.platform == "win32", "Windows console creation flags")
    async def test_a_hidden_launch_on_windows_uses_a_hidden_console_like_headless(self):
        for mode in ("hidden", "headless"):
            with self.subTest(mode=mode):
                call, error = await self.launch(mode, "win32")
                self.assertIn("broker failed", error)
                flags = call.kwargs["creationflags"]
                self.assertTrue(flags & Viewer_Client.subprocess.CREATE_NO_WINDOW)
                broker = call.args[0][2]
                self.assertIn("creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP", broker)
                self.assertNotIn("CREATE_NEW_CONSOLE", broker)
                self.assertNotIn("Viewer_Terminal", call.args[0][3])

    async def test_an_unknown_mode_is_refused(self):
        with self.assertRaisesRegex(MCPViewerError, "normal, headless, hidden"):
            await MCPViewerClient(Path(".")).launch_session(settings_document={}, mode="invisible")

    async def test_a_hidden_viewer_with_no_opengl_is_launched_again_headless_with_the_reason(self):
        marker = f"{display.HIDDEN_UNAVAILABLE_MARKER}: the native platform cannot create an OpenGL context"
        ready = {"status": "ready", "mode": "headless"}
        client = MCPViewerClient(Path("."))
        with mock.patch.object(MCPViewerClient, "_launch",
                               side_effect=[MCPViewerError(f"Viewer exited before readiness. Logs: x\n{marker}\n"), ready]) as launch:
            result = await client.launch_session(settings_document={"a": 1}, mode="hidden", timeout=9)
        self.assertEqual(result, ready)
        first, second = launch.call_args_list
        self.assertEqual(first.kwargs["mode"], "hidden")
        self.assertEqual((second.kwargs["mode"], second.kwargs["requested"], second.kwargs["timeout"]), ("headless", "hidden", 9))
        self.assertEqual(second.kwargs["settings_document"], {"a": 1})
        self.assertIn("cannot create an OpenGL context", second.kwargs["fallback_reason"])
        self.assertIn("native platform could not open the hidden window", second.kwargs["fallback_reason"])

    async def test_other_launch_failures_are_not_retried(self):
        marker = display.HIDDEN_UNAVAILABLE_MARKER
        for mode, message in (("hidden", "Viewer exited before readiness"), ("normal", marker), ("headless", marker)):
            with self.subTest(mode=mode, message=message), \
                    mock.patch.object(MCPViewerClient, "_launch", side_effect=MCPViewerError(message)) as launch:
                with self.assertRaises(MCPViewerError):
                    await MCPViewerClient(Path(".")).launch_session(settings_document={}, mode=mode)
            self.assertEqual(launch.call_count, 1)

    async def test_a_fallback_launch_runs_headless_and_hands_the_reason_to_the_viewer(self):
        reason = "no display was found (DISPLAY and WAYLAND_DISPLAY are not set)"
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": str(Path(temporary) / "sessions")}), \
                self.settings(), mock.patch.object(Viewer_Client, "_identity", return_value=None), \
                mock.patch.object(Viewer_Client, "sys", SimpleNamespace(platform="linux", executable=sys.executable)), \
                mock.patch.object(Viewer_Client.subprocess, "Popen") as popen:
            with self.assertRaises(MCPViewerError):
                await MCPViewerClient(Path(temporary))._launch(
                    settings_document={}, mode="headless", requested="hidden", fallback_reason=reason)
            record = json.loads(next(Path(temporary, "sessions", "launches").glob("*/launch.json")).read_text())
        self.assertEqual((record["mode"], record["requested_mode"], record["fallback_reason"]), ("headless", "hidden", reason))
        self.assertEqual(popen.call_args.kwargs["env"]["SSN_VIEWER_WINDOW_FALLBACK"], reason)
        self.assertEqual(popen.call_args.kwargs["env"]["QT_QPA_PLATFORM"], "offscreen")

    def test_a_launch_record_may_name_the_hidden_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / ("a" * 32)
            directory.mkdir(parents=True)
            record = {"launch_id": directory.name, "mode": "hidden", "started_epoch": 1.0, "cache_path": "c.h5"}
            (directory / "launch.json").write_text(json.dumps(record))
            self.assertEqual(Viewer_Client._read_launch(directory)["mode"], "hidden")
            (directory / "launch.json").write_text(json.dumps({**record, "mode": "invisible"}))
            with self.assertRaisesRegex(MCPViewerError, "Invalid launch record"):
                Viewer_Client._read_launch(directory)

    async def ready(self, window, launch=None):
        launch_id = "b" * 32
        launch = {"launch_id": launch_id, "mode": "hidden", "started_epoch": 1.0, "cache_path": "c.h5", **(launch or {})}
        session = SimpleNamespace(launch_id=launch_id, session_id="c" * 32, pid=5, base_url="http://127.0.0.1:1",
                                  started_at="now", token="t")
        client = MCPViewerClient(Path("."))
        summary = {"window": window} if window is not None else {}
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": folder.name}), \
                mock.patch.object(Viewer_Client, "discover_viewer_sessions", return_value=[session]), \
                mock.patch.object(MCPViewerClient, "_request", return_value=summary), \
                mock.patch.object(Viewer_Client, "session_alias", return_value="CCCCCCCC"):
            return await client._await_launch(Path("."), launch, lambda: True, 1)

    async def test_the_ready_result_reports_the_mode_the_viewer_really_has(self):
        result = await self.ready({"mode": "hidden", "requested_mode": "hidden", "fallback_reason": None})
        self.assertEqual(result["mode"], "hidden")
        self.assertNotIn("requested_mode", result)
        self.assertNotIn("fallback_reason", result)
        reason = "no display was found (DISPLAY and WAYLAND_DISPLAY are not set)"
        result = await self.ready({"mode": "headless", "requested_mode": "hidden", "fallback_reason": reason})
        self.assertEqual((result["mode"], result["requested_mode"], result["fallback_reason"]), ("headless", "hidden", reason))
        # A Viewer that does not report its window keeps the launch's own answer.
        result = await self.ready(None, {"mode": "headless", "requested_mode": "hidden", "fallback_reason": "x"})
        self.assertEqual((result["mode"], result["requested_mode"], result["fallback_reason"]), ("headless", "hidden", "x"))
        result = await self.ready(None, {"mode": "normal"})
        self.assertEqual(result["mode"], "normal")
        self.assertNotIn("requested_mode", result)

    async def test_window_actions_go_to_the_window_route_of_the_selected_viewer(self):
        session = SimpleNamespace(session_id="d" * 32, base_url="http://127.0.0.1:1", token="t")
        client = MCPViewerClient(Path("."))
        client.connected_session_id = session.session_id
        calls = []

        def request(self_, selected, endpoint, data=None):
            calls.append((endpoint, data))
            if endpoint.endswith("/session"):
                return {"inspection_capabilities": ["window_v1"]}
            return {"mode": "hidden", "visible": data["action"] == "show", "foreground": False}

        with mock.patch.object(Viewer_Client, "select_viewer_session", return_value=session), \
                mock.patch.object(MCPViewerClient, "_request", request), \
                mock.patch.object(Viewer_Client, "session_alias", return_value="DDDDDDDD"):
            shown = await client.window_action("show")
            hidden = await client.window_action("hide", session.session_id)
            with self.assertRaisesRegex(MCPViewerError, "Expected show or hide"):
                await client.window_action("minimize")
        self.assertEqual((shown["session_id"], shown["action"], shown["window"]["visible"]), (session.session_id, "show", True))
        self.assertFalse(hidden["window"]["visible"])
        self.assertEqual([call for call in calls if call[0].endswith("/window")],
                         [("/api/mcp/v1/window", {"action": "show"}), ("/api/mcp/v1/window", {"action": "hide"})])

    async def test_an_older_viewer_without_the_window_route_is_told_to_restart(self):
        session = SimpleNamespace(session_id="d" * 32, base_url="http://127.0.0.1:1", token="t")
        client = MCPViewerClient(Path("."))
        client.connected_session_id = session.session_id
        with mock.patch.object(Viewer_Client, "select_viewer_session", return_value=session), \
                mock.patch.object(MCPViewerClient, "_request", return_value={"inspection_capabilities": ["snapshots_v1"]}):
            with self.assertRaisesRegex(MCPViewerError, "upgrade and restart"):
                await client.window_action("show")
        with self.assertRaisesRegex(MCPViewerError, "No Viewer is connected"):
            await MCPViewerClient(Path(".")).window_action("show")


class McpActionTests(unittest.IsolatedAsyncioTestCase):
    def test_both_actions_are_registered_with_the_session_argument(self):
        from mcp_server.core.Workflow_Dispatch import REGISTRY
        for action in ("show_session", "hide_session"):
            with self.subTest(action=action):
                self.assertIn(action, REGISTRY["emapssn_viewer_control"])
                fields = REGISTRY["emapssn_viewer_control"][action].model.model_fields
                self.assertEqual(set(fields), {"session_id"})

    def test_start_session_offers_the_hidden_mode(self):
        from mcp_server.core.Workflow_Dispatch import REGISTRY
        schema = REGISTRY["emapssn_viewer_control"]["start_session"].model.model_json_schema()
        self.assertEqual(schema["properties"]["mode"]["enum"], ["normal", "headless", "hidden"])

    async def test_show_tells_the_agent_when_the_window_did_not_take_the_keyboard(self):
        from mcp_server.viewer import Viewer_Operations as operations
        from mcp.server.mcpserver.exceptions import ToolError
        client = SimpleNamespace()

        async def action(name, session_id):
            if session_id == "headless":
                raise MCPViewerError("This Viewer runs headless (Qt's offscreen platform), which has no window to show.")
            return {"session_id": "s", "action": name, "window": {"visible": True, "foreground": session_id == "front"}}

        client.window_action = action
        context = SimpleNamespace()
        with mock.patch.object(operations, "_viewer", return_value=client):
            behind = await operations.show_viewer_window(context, "behind")
            front = await operations.show_viewer_window(context, "front")
            hidden = await operations.hide_viewer_window(context, "behind")
            with self.assertRaisesRegex(ToolError, "no window to show"):
                await operations.show_viewer_window(context, "headless")
        self.assertIn("ask the user to click it", behind["message"])
        self.assertNotIn("message", front)
        self.assertEqual(hidden["action"], "hide")


if __name__ == "__main__":
    unittest.main()
