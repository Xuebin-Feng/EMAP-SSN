import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from SSN_Viewer import (
    MainViewer,
    _apply_safe_rectangle_geometry,
    _vispy_text_line_height_pixels,
    _wrap_console_text_for_display,
)
from vispy.visuals import RectangleVisual


class FixedWidthFont:
    ratio = 4.0
    slop = 0.0
    _lowres_size = 256.0

    def __getitem__(self, character):
        return {
            "advance": 256.0,
            "kerning": {},
            "offset": (0.0, 1024.0),
            "size": (256.0, 256.0),
        }


class StrictRoundedRectangle:
    """Mirror VisPy's property-by-property mutation and radius validation."""

    def __init__(self, width, height, radius, center=(0.0, 0.0), validate=True):
        self._width = width
        self._height = height
        self._radius = radius
        self._center = center
        if validate:
            self._validate()

    def _validate(self):
        if self._radius > min(self._width, self._height) / 2.0:
            raise ValueError("radius exceeds half of min(width, height)")

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value <= 0:
            raise ValueError("width must be positive")
        self._width = value
        self._validate()

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        if value <= 0:
            raise ValueError("height must be positive")
        self._height = value
        self._validate()

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = value
        self._validate()

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._validate()


class ConsoleOverlayWrappingTests(unittest.TestCase):
    @staticmethod
    def measure_monospace(text):
        return len(text)

    @staticmethod
    def make_viewer(pixel_scale=1.0, size=(240, 100), text="Cmd: color red"):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {
            "console_bg_left_offset": 10.0,
            "console_text_x": 30.0,
            "console_bg_padding_x": 20.0,
            "console_bg_min_width": 150.0,
            "console_bg_height": 20.0,
            "console_bg_radius": 6.0,
            "console_text_y": 60.0,
            "console_text_anchor_y": "bottom",
            "background_job_status_gap": 8.0,
            "background_job_status_right_padding": 20.0,
        }
        viewer.canvas = SimpleNamespace(
            pixel_scale=pixel_scale,
            size=size,
            update=lambda: None,
        )
        viewer.console_bg = StrictRoundedRectangle(
            width=150.0,
            height=20.0,
            radius=6.0,
        )
        viewer.console_text = SimpleNamespace(
            text=text,
            _font=FixedWidthFont(),
            transforms=SimpleNamespace(dpi=1.0),
            font_size=72.0,
            _line_height=1.2,
        )
        return viewer

    @staticmethod
    def add_background_job_status(viewer, text=""):
        viewer.background_job_status_text = SimpleNamespace(
            text=text,
            visible=False,
            pos=(0.0, 0.0),
            _font=FixedWidthFont(),
            transforms=SimpleNamespace(dpi=1.0),
            font_size=72.0,
            _line_height=1.2,
        )
        viewer._background_job_status_logical_text = ""
        viewer._background_job_status_rendered_text = ""
        return viewer.background_job_status_text

    def test_short_logical_line_is_unchanged(self):
        text = "Cmd: select cluster_1"

        wrapped = _wrap_console_text_for_display(
            text,
            max_width=80,
            measure_width=self.measure_monospace,
        )

        self.assertEqual(wrapped, text)

    def test_long_line_wraps_visually_without_changing_logical_text(self):
        text = "Cmd: select a_very_long_selection_expression with more arguments"

        wrapped = _wrap_console_text_for_display(
            text,
            max_width=18,
            measure_width=self.measure_monospace,
        )

        self.assertIn("\n", wrapped)
        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertTrue(all(len(line) <= 18 for line in wrapped.splitlines()))

    def test_long_unbroken_token_uses_character_wrapping(self):
        text = "x" * 47

        wrapped = _wrap_console_text_for_display(
            text,
            max_width=10,
            measure_width=self.measure_monospace,
        )

        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertEqual([len(line) for line in wrapped.splitlines()], [10, 10, 10, 10, 7])

    def test_overlay_rewraps_on_resize_and_retains_one_logical_line(self):
        text = "Cmd: select " + ("x" * 36)
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {
            "console_bg_left_offset": 10.0,
            "console_text_x": 30.0,
            "console_bg_padding_x": 20.0,
            "console_bg_min_width": 150.0,
            "console_bg_height": 20.0,
            "console_bg_radius": 6.0,
            "console_text_y": 60.0,
            "console_text_anchor_y": "bottom",
        }
        viewer.canvas = SimpleNamespace(pixel_scale=1.0, size=(100, 100))
        viewer.console_bg = SimpleNamespace()
        viewer.console_text = SimpleNamespace(
            text=text,
            _font=FixedWidthFont(),
            transforms=SimpleNamespace(dpi=1.0),
            font_size=72.0,
            _line_height=1.2,
        )

        viewer.update_console_background()

        narrow_rendered = viewer.console_text.text
        narrow_height = viewer.console_bg.height
        narrow_center_y = viewer.console_bg.center[1]
        narrow_top = narrow_center_y - narrow_height / 2.0
        self.assertIn("\n", narrow_rendered)
        self.assertEqual(viewer._console_logical_text, text)
        self.assertEqual(narrow_rendered.replace("\n", ""), text)
        self.assertLessEqual(viewer.console_bg.width, 60.0)
        self.assertGreater(narrow_height, 20.0)

        viewer.canvas.size = (140, 100)
        viewer.update_console_background()

        self.assertEqual(viewer._console_logical_text, text)
        self.assertEqual(viewer.console_text.text.replace("\n", ""), text)
        self.assertLess(viewer.console_text.text.count("\n"), narrow_rendered.count("\n"))
        self.assertLess(viewer.console_bg.height, narrow_height)
        self.assertLess(viewer.console_bg.center[1], narrow_center_y)
        self.assertAlmostEqual(
            viewer.console_bg.center[1] - viewer.console_bg.height / 2.0,
            narrow_top,
        )

    def test_background_center_tracks_bottom_anchored_text(self):
        viewer = self.make_viewer(text="Cmd: _")
        viewer.hud_layout["console_bg_y_offset"] = 14.0

        viewer.update_console_background()

        expected_line_height = (
            (viewer.console_text.font_size / 72.0)
            * viewer.console_text.transforms.dpi
            * viewer.console_text._line_height
        )
        self.assertAlmostEqual(
            viewer.console_bg.center[1],
            viewer.hud_layout["console_text_y"]
            - expected_line_height / 2.0
            + viewer.hud_layout["console_bg_y_offset"],
        )

    def test_line_height_uses_vispy_font_metrics(self):
        text_visual = SimpleNamespace(
            _font=FixedWidthFont(),
            transforms=SimpleNamespace(dpi=72.0),
            font_size=10.0,
            _line_height=1.2,
        )

        self.assertAlmostEqual(
            _vispy_text_line_height_pixels(text_visual),
            12.0,
        )

    def test_background_job_status_wraps_below_multiline_console(self):
        viewer = self.make_viewer(
            size=(150, 160),
            text="Cmd: " + ("command" * 12),
        )
        status = self.add_background_job_status(
            viewer,
            "Background job #1 completed: " + ("long-output-path" * 12),
        )

        viewer.update_console_background()

        self.assertIn("\n", viewer.console_text.text)
        self.assertIn("\n", status.text)
        self.assertEqual(status.pos[0], viewer.hud_layout["console_text_x"])
        self.assertEqual(
            status.pos[1],
            viewer.console_bg.center[1]
            + viewer.console_bg.height / 2.0
            + viewer.hud_layout["background_job_status_gap"],
        )
        self.assertTrue(status.visible)

    def test_background_job_status_rewraps_for_sidebar_and_resize(self):
        viewer = self.make_viewer(size=(260, 160), text="Cmd: _")
        status = self.add_background_job_status(
            viewer,
            "Background job completed: " + ("result-path " * 12),
        )
        viewer.right_panel = SimpleNamespace(isVisible=lambda: False)
        viewer._panel_w = 80

        viewer.update_console_background()
        wide_text = status.text

        viewer.right_panel = SimpleNamespace(isVisible=lambda: True)
        viewer.update_console_background()
        panel_text = status.text
        self.assertGreater(panel_text.count("\n"), wide_text.count("\n"))

        viewer.canvas.size = (320, 160)
        viewer.right_panel = SimpleNamespace(isVisible=lambda: False)
        viewer.update_console_background()
        self.assertLess(status.text.count("\n"), panel_text.count("\n"))

    def test_scheduler_status_does_not_mutate_typed_command_or_box(self):
        viewer = self.make_viewer(size=(260, 160), text="Cmd: select cluster_1_")
        status = self.add_background_job_status(viewer)
        viewer.update_console_background()
        command_before = viewer.console_text.text
        geometry_before = (
            viewer.console_bg.center,
            viewer.console_bg.width,
            viewer.console_bg.height,
        )

        viewer.set_background_job_status(
            "Background job #1 completed: saved output.xlsx"
        )

        self.assertEqual(viewer.console_text.text, command_before)
        self.assertEqual(
            (
                viewer.console_bg.center,
                viewer.console_bg.width,
                viewer.console_bg.height,
            ),
            geometry_before,
        )
        self.assertIn("completed", status.text)
        self.assertEqual(
            status.pos[1],
            viewer.console_bg.center[1]
            + viewer.console_bg.height / 2.0
            + viewer.hud_layout["background_job_status_gap"],
        )

        viewer.clear_background_job_status()
        self.assertEqual(status.text, "")
        self.assertFalse(status.visible)

        viewer.set_background_job_status("Background job #2 completed")
        self.assertTrue(status.visible)
        self.assertEqual(viewer.console_text.text, command_before)

    def test_opening_command_clears_previous_scheduler_status(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.console_mode = False
        viewer.command_history = []
        viewer.console_bg = SimpleNamespace(visible=False)
        viewer.clear_background_job_status = mock.Mock()
        viewer._update_console_text = mock.Mock()
        event = SimpleNamespace(
            key="Enter",
            modifiers=[],
            text="",
            handled=False,
        )

        viewer.on_key_press(event)

        viewer.clear_background_job_status.assert_called_once_with()
        self.assertTrue(viewer.console_mode)
        self.assertEqual(viewer.input_buffer, "")
        self.assertTrue(viewer.console_bg.visible)
        self.assertTrue(event.handled)

    def test_scheduler_status_uses_downward_growing_console_anchor(self):
        viewer = self.make_viewer()
        viewer.hud_layout.update({
            "font_size_px": 16.0,
            "console_text_anchor_x": "left",
            "instr_x": 10.0,
            "instr_y": 10.0,
            "instr_anchor_x": "left",
            "instr_anchor_y": "bottom",
            "status_x_offset": 10.0,
            "status_bottom_offset": 30.0,
            "status_line_spacing": 25.0,
            "status_anchor_x": "right",
            "status_anchor_y": "bottom",
        })
        viewer.canvas.dpi = 96.0
        viewer.canvas.scene = object()
        viewer.vispy_ui_face = "Arial"
        viewer.vispy_monospace_face = "Courier New"

        def make_visual(**kwargs):
            return SimpleNamespace(visible=True, **kwargs)

        with (
            mock.patch(
                "SSN_Viewer.scene.visuals.Rectangle",
                side_effect=make_visual,
            ) as rectangle_mock,
            mock.patch(
                "SSN_Viewer.scene.visuals.Text",
                side_effect=make_visual,
            ),
        ):
            viewer.create_hud()

        self.assertEqual(
            viewer.background_job_status_text.anchor_y,
            viewer.hud_layout["console_text_anchor_y"],
        )
        self.assertEqual(rectangle_mock.call_count, 1)

    def test_safe_geometry_recovers_vispy_partial_mutation(self):
        rectangle = StrictRoundedRectangle(
            width=150.0,
            height=20.0,
            radius=12.0,
            validate=False,
        )

        applied = _apply_safe_rectangle_geometry(
            rectangle,
            center=(85.0, 35.0),
            width=150.0,
            height=20.0,
            radius=6.0,
        )

        self.assertEqual(applied, (150.0, 20.0, 6.0))
        self.assertEqual(rectangle.center, (85.0, 35.0))
        self.assertEqual(rectangle.radius, 6.0)

    def test_safe_geometry_recovers_real_vispy_rectangle(self):
        rectangle = RectangleVisual(
            center=(160.0, 35.0),
            width=300.0,
            height=40.0,
            radius=12.0,
        )
        with self.assertRaisesRegex(ValueError, "Radius of curvature"):
            rectangle.height = 20.0

        _apply_safe_rectangle_geometry(
            rectangle,
            center=(85.0, 35.0),
            width=150.0,
            height=20.0,
            radius=6.0,
        )

        self.assertEqual(rectangle.width, 150.0)
        self.assertEqual(rectangle.height, 20.0)
        self.assertEqual(rectangle.radius, 6.0)
        self.assertEqual(rectangle.center, (85.0, 35.0))

    def test_overlay_geometry_is_identical_across_device_pixel_ratios(self):
        geometries = []
        for pixel_scale in (1.0, 1.25, 1.5, 2.0):
            viewer = self.make_viewer(pixel_scale=pixel_scale)
            status = self.add_background_job_status(
                viewer,
                "Background job completed: output.xlsx",
            )
            viewer.update_console_background()
            geometries.append((
                viewer.console_bg.width,
                viewer.console_bg.height,
                viewer.console_bg.radius,
                viewer.console_bg.center,
                viewer.console_text.text,
                status.pos,
                status.text,
            ))

        self.assertTrue(all(geometry == geometries[0] for geometry in geometries[1:]))

    def test_overlay_survives_scale_down_from_retina_geometry(self):
        viewer = self.make_viewer(pixel_scale=2.0)
        viewer.console_bg = StrictRoundedRectangle(
            width=300.0,
            height=40.0,
            radius=12.0,
        )

        viewer.update_console_background()
        viewer.canvas.pixel_scale = 1.0
        viewer.update_console_background()

        self.assertEqual(viewer.console_bg.height, 20.0)
        self.assertEqual(viewer.console_bg.radius, 6.0)

    def test_tiny_canvas_clamps_corner_radius(self):
        viewer = self.make_viewer(size=(41, 100), text="_")

        viewer.update_console_background()

        self.assertEqual(viewer.console_bg.width, 1.0)
        self.assertEqual(viewer.console_bg.radius, 0.5)


if __name__ == "__main__":
    unittest.main()
