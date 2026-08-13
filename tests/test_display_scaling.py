import os
import sys
import unittest
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtCore
from SSN_Viewer import MainViewer


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        try:
            self.slots.remove(slot)
        except ValueError as exc:
            raise TypeError("slot is not connected") from exc

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class FakeGeometry:
    def __init__(self, x=0, y=0, width=1920, height=1080):
        self._values = (x, y, width, height)

    def x(self):
        return self._values[0]

    def y(self):
        return self._values[1]

    def width(self):
        return self._values[2]

    def height(self):
        return self._values[3]


class FakeScreen:
    def __init__(self, name, geometry=None):
        self._name = name
        self._geometry = geometry or FakeGeometry()
        self.logicalDotsPerInchChanged = FakeSignal()
        self.physicalDotsPerInchChanged = FakeSignal()
        self.geometryChanged = FakeSignal()

    def name(self):
        return self._name

    def geometry(self):
        return self._geometry


class FakeWindowHandle:
    def __init__(self, screen):
        self._screen = screen
        self.screenChanged = FakeSignal()

    def screen(self):
        return self._screen

    def switch_screen(self, screen):
        self._screen = screen
        self.screenChanged.emit(screen)


class FakeMainWindow:
    def __init__(self, handle):
        self._handle = handle

    def windowHandle(self):
        return self._handle


class FakeNativeCanvas:
    def __init__(self, canvas, dpr=1.0):
        self.canvas = canvas
        self.dpr = dpr
        self.screen_changed_calls = []

    def width(self):
        return self.canvas.size[0]

    def height(self):
        return self.canvas.size[1]

    def devicePixelRatioF(self):
        return self.dpr

    def screen_changed(self, screen):
        self.screen_changed_calls.append(screen)
        self.canvas.physical_size = (
            self.canvas.size[0] * self.dpr,
            self.canvas.size[1] * self.dpr,
        )


class FakeCanvas:
    def __init__(self, size=(800, 600), dpr=1.0):
        self.size = size
        self.physical_size = (size[0] * dpr, size[1] * dpr)
        self.pixel_scale = dpr
        self.update_calls = 0
        self.native = FakeNativeCanvas(self, dpr=dpr)

    def update(self):
        self.update_calls += 1


class DisplayScalingTests(unittest.TestCase):
    def setUp(self):
        self.queued_callbacks = []
        self.timer_patch = mock.patch.object(
            QtCore.QTimer,
            "singleShot",
            side_effect=lambda delay, callback: self.queued_callbacks.append(callback),
        )
        self.timer_patch.start()

    def tearDown(self):
        self.timer_patch.stop()

    def drain_callbacks(self):
        while self.queued_callbacks:
            self.queued_callbacks.pop(0)()

    @staticmethod
    def make_viewer(screen=None, handle_available=True):
        screen = screen or FakeScreen("Standard display")
        handle = FakeWindowHandle(screen) if handle_available else None
        viewer = MainViewer.__new__(MainViewer)
        viewer._display_window_handle = None
        viewer._active_screen = None
        viewer._active_screen_signal_bindings = []
        viewer._display_refresh_pending = False
        viewer._last_display_signature = None
        viewer.main_window = FakeMainWindow(handle)
        viewer.canvas = FakeCanvas()
        viewer.position_slider_overlay = mock.Mock()
        viewer.reposition_expand_btn = mock.Mock()
        viewer._update_hud_elements = mock.Mock()
        viewer._print_display_diagnostics = mock.Mock()
        return viewer, handle, screen

    def test_initial_tracking_binds_final_window_and_refreshes_layout(self):
        viewer, handle, screen = self.make_viewer()

        viewer._initialize_display_tracking()

        self.assertEqual(len(handle.screenChanged.slots), 1)
        self.assertEqual(len(screen.logicalDotsPerInchChanged.slots), 1)
        self.assertEqual(len(screen.physicalDotsPerInchChanged.slots), 1)
        self.assertEqual(len(screen.geometryChanged.slots), 1)
        self.assertTrue(viewer._display_refresh_pending)
        self.assertEqual(len(self.queued_callbacks), 1)

        self.drain_callbacks()

        self.assertEqual(viewer.canvas.native.screen_changed_calls, [screen])
        viewer.position_slider_overlay.assert_called_once_with()
        viewer.reposition_expand_btn.assert_called_once_with()
        viewer._update_hud_elements.assert_called_once_with()
        viewer._print_display_diagnostics.assert_called_once_with()
        self.assertEqual(viewer.canvas.update_calls, 1)

    def test_screen_switch_rebinds_signals_and_debounces_metric_changes(self):
        viewer, handle, old_screen = self.make_viewer()
        viewer._initialize_display_tracking()
        self.drain_callbacks()
        viewer.canvas.native.screen_changed_calls.clear()

        new_screen = FakeScreen(
            "Retina display",
            geometry=FakeGeometry(1920, 0, 1512, 982),
        )
        viewer.canvas.native.dpr = 2.0
        handle.switch_screen(new_screen)
        new_screen.logicalDotsPerInchChanged.emit(192.0)
        new_screen.geometryChanged.emit(new_screen.geometry())

        self.assertEqual(len(self.queued_callbacks), 1)
        self.assertEqual(len(old_screen.logicalDotsPerInchChanged.slots), 0)
        self.assertEqual(len(new_screen.logicalDotsPerInchChanged.slots), 1)

        self.drain_callbacks()

        self.assertEqual(viewer.canvas.native.screen_changed_calls, [new_screen])
        self.assertEqual(viewer.canvas.physical_size, (1600.0, 1200.0))

        old_screen.logicalDotsPerInchChanged.emit(96.0)
        self.assertEqual(self.queued_callbacks, [])

    def test_missing_window_handle_keeps_resize_fallback_available(self):
        viewer, handle, screen = self.make_viewer(handle_available=False)

        viewer._initialize_display_tracking()
        self.drain_callbacks()

        self.assertIsNone(handle)
        self.assertIsNone(viewer._active_screen)
        self.assertEqual(viewer.canvas.native.screen_changed_calls, [None])
        viewer._update_hud_elements.assert_called_once_with()

    def test_display_diagnostics_print_only_when_signature_changes(self):
        viewer, handle, screen = self.make_viewer()
        viewer._print_display_diagnostics = MainViewer._print_display_diagnostics.__get__(
            viewer,
            MainViewer,
        )
        viewer._active_screen = screen

        with mock.patch("builtins.print") as print_mock:
            viewer._print_display_diagnostics()
            viewer._print_display_diagnostics()
            viewer.canvas.native.dpr = 2.0
            viewer.canvas.physical_size = (1600.0, 1200.0)
            viewer._print_display_diagnostics()

        self.assertEqual(print_mock.call_count, 2)
        self.assertIn("DPR=1", print_mock.call_args_list[0].args[0])
        self.assertIn("DPR=2", print_mock.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
