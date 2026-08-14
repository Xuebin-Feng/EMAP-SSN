import os
import sys
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtCore
import SSN_Config as cfg
from SSN_Viewer import (
    HUDDisplay,
    MainViewer,
    _configure_linux_vispy_platform,
    _contiguous_line_positions,
)
from commands import meta as meta_command


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
        self.queued_delays = []
        self.timer_patch = mock.patch.object(
            QtCore.QTimer,
            "singleShot",
            side_effect=self.queue_callback,
        )
        self.timer_patch.start()

    def tearDown(self):
        self.timer_patch.stop()

    def drain_callbacks(self):
        while self.queued_callbacks:
            self.queued_callbacks.pop(0)()

    def queue_callback(self, delay, callback):
        self.queued_delays.append(delay)
        self.queued_callbacks.append(callback)

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
        viewer._apply_vispy_text_scaling = mock.Mock()
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
        self.assertEqual(self.queued_delays, [100])

        self.drain_callbacks()

        self.assertEqual(viewer.canvas.native.screen_changed_calls, [])
        viewer._apply_vispy_text_scaling.assert_called_once_with()
        viewer.position_slider_overlay.assert_called_once_with()
        viewer.reposition_expand_btn.assert_called_once_with()
        viewer._update_hud_elements.assert_called_once_with()
        viewer._print_display_diagnostics.assert_called_once_with()
        self.assertEqual(viewer.canvas.update_calls, 1)

    def test_screen_switch_rebinds_signals_and_debounces_metric_changes(self):
        viewer, handle, old_screen = self.make_viewer()
        viewer._initialize_display_tracking()
        self.drain_callbacks()
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

        self.assertEqual(viewer.canvas.native.screen_changed_calls, [])
        self.assertEqual(viewer.canvas.physical_size, (800.0, 600.0))

        old_screen.logicalDotsPerInchChanged.emit(96.0)
        self.assertEqual(self.queued_callbacks, [])

    def test_missing_window_handle_still_refreshes_logical_overlays(self):
        viewer, handle, screen = self.make_viewer(handle_available=False)

        viewer._initialize_display_tracking()
        self.drain_callbacks()

        self.assertIsNone(handle)
        self.assertIsNone(viewer._active_screen)
        self.assertEqual(viewer.canvas.native.screen_changed_calls, [])
        viewer._update_hud_elements.assert_called_once_with()

    def test_resize_refreshes_hud_synchronously_without_timer(self):
        viewer, _handle, _screen = self.make_viewer()
        viewer._hud_timer = mock.Mock()
        viewer.slider_overlay = object()

        viewer.on_resize(mock.Mock())

        viewer._update_hud_elements.assert_called_once_with()
        viewer._hud_timer.start.assert_not_called()
        viewer.position_slider_overlay.assert_called_once_with()
        viewer.reposition_expand_btn.assert_called_once_with()

    def test_bottom_right_status_positions_have_equal_line_spacing(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {
            "status_x_offset": 10.0,
            "status_bottom_offset": 30.0,
            "status_line_spacing": 25.0,
        }
        size = (1000.0, 800.0)

        hidden_pos = viewer._status_hud_position(0, size)
        view_width_pos = viewer._status_hud_position(1, size)
        property_pos = viewer._status_hud_position(2, size)

        self.assertEqual(hidden_pos[0], view_width_pos[0])
        self.assertEqual(view_width_pos[0], property_pos[0])
        self.assertEqual(hidden_pos[1] - view_width_pos[1], 25.0)
        self.assertEqual(view_width_pos[1] - property_pos[1], 25.0)

    def test_metadata_property_display_uses_status_line_above_view_width(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {
            "status_x_offset": 10.0,
            "status_bottom_offset": 30.0,
            "status_line_spacing": 25.0,
        }
        viewer.hud_displays = {}
        viewer.metadata = {"Length": {"values": [125]}}
        viewer.selected_node_idx = 0

        with (
            mock.patch.object(meta_command.os, "makedirs"),
            mock.patch.object(meta_command.Command_Engine, "print_help"),
            mock.patch.object(HUDDisplay, "show"),
        ):
            meta_command.run(viewer, ["display", "Length"])

        display = viewer.hud_displays["meta_display"]
        size = (1000.0, 800.0)
        self.assertEqual(
            display.pos_fn(size),
            viewer._status_hud_position(2, size),
        )

    def test_zero_hidden_nodes_and_view_width_use_configured_text_color(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {
            "status_x_offset": 10.0,
            "status_bottom_offset": 30.0,
            "status_line_spacing": 25.0,
        }
        viewer.canvas = FakeCanvas(size=(1000, 800))
        viewer.view = mock.Mock()
        viewer.view.camera.rect.width = 250.0
        viewer.zoom_text = mock.Mock()
        viewer.hidden_text = mock.Mock()
        viewer.visible_mask = np.ones(3, dtype=bool)
        viewer.selected_node_idx = None
        viewer.tooltip = None
        viewer.hud_displays = {}
        viewer.update_console_background = mock.Mock()

        viewer._update_hud_elements()

        self.assertEqual(viewer.zoom_text.color, cfg.TEXT_COLOR)
        self.assertEqual(viewer.hidden_text.color, cfg.TEXT_COLOR)

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
        self.assertIn("Qt platform=", print_mock.call_args_list[0].args[0])

    def test_native_wayland_is_replaced_before_vispy_import(self):
        automatic = {"XDG_SESSION_TYPE": "wayland"}
        explicit = {
            "XDG_SESSION_TYPE": "wayland",
            "QT_QPA_PLATFORM": "wayland",
        }
        headless = {
            "XDG_SESSION_TYPE": "wayland",
            "QT_QPA_PLATFORM": "offscreen",
        }

        self.assertTrue(_configure_linux_vispy_platform(automatic, "linux"))
        self.assertTrue(_configure_linux_vispy_platform(explicit, "linux"))
        self.assertFalse(_configure_linux_vispy_platform(headless, "linux"))
        self.assertEqual(automatic["QT_QPA_PLATFORM"], "xcb")
        self.assertEqual(explicit["QT_QPA_PLATFORM"], "xcb")
        self.assertEqual(headless["QT_QPA_PLATFORM"], "offscreen")

    def test_line_positions_are_finite_contiguous_float32(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)[:, ::-1]
        result = _contiguous_line_positions(source)

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, [[2.0, 1.0], [4.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            _contiguous_line_positions([[0.0, float("nan")]])


if __name__ == "__main__":
    unittest.main()
