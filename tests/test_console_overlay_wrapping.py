"""Viewer console overlay (EMAPSSN_Viewer): visual wrapping of the logical
console line, the rounded background box and its safe geometry updates, the
background-job status line below it, and DPI-independent overlay geometry.
Above them, the instruction line wraps between its actions to end before the
sidebar toggle, and the console moves down by the rows it gains."""

import collections
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

import EMAPSSN_Viewer
from EMAPSSN_Viewer import (
    HUD_INSTRUCTION_SEPARATOR,
    MainViewer,
    _apply_safe_rectangle_geometry,
    _vispy_text_line_height_pixels,
    _wrap_console_text_for_display,
    _wrap_instructions_for_display,
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

    def __init__(self, width, height, radius, center=(0.0, 0.0)):
        self._width = width
        self._height = height
        self._radius = radius
        self._center = center
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
                "EMAPSSN_Viewer.scene.visuals.Rectangle",
                side_effect=make_visual,
            ) as rectangle_mock,
            mock.patch(
                "EMAPSSN_Viewer.scene.visuals.Text",
                side_effect=make_visual,
            ),
        ):
            viewer.create_hud()

        self.assertEqual(
            viewer.background_job_status_text.anchor_y,
            viewer.hud_layout["console_text_anchor_y"],
        )
        self.assertEqual(rectangle_mock.call_count, 1)

    def test_safe_geometry_recovers_real_vispy_rectangle(self):
        rectangle = RectangleVisual(
            center=(160.0, 35.0),
            width=300.0,
            height=40.0,
            radius=12.0,
        )
        with self.assertRaisesRegex(ValueError, "Radius of curvature"):
            rectangle.height = 20.0

        applied = _apply_safe_rectangle_geometry(
            rectangle,
            center=(85.0, 35.0),
            width=150.0,
            height=20.0,
            radius=6.0,
        )

        self.assertEqual(applied, (150.0, 20.0, 6.0))
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


class SidebarToggle:
    """Stands in for the sidebar's toggle button, at its left edge x."""

    def __init__(self, x):
        self._x = x

    def x(self):
        return self._x

    def isVisible(self):
        return True


class CloseButton:
    """Stands in for the instruction line's close button: where it moved, and whether shown."""

    def __init__(self, size=20):
        self.size = size
        self.pos = None
        self.shown = False

    def width(self):
        return self.size

    def height(self):
        return self.size

    def move(self, x, y):
        self.pos = (x, y)

    def show(self):
        self.shown = True

    def hide(self):
        self.shown = False


class InstructionLineWrappingTests(unittest.TestCase):
    """The instruction line lists every key and mouse action. It wraps between
    them to end before the sidebar toggle, which shares its top margin, and
    the console moves down by the rows it gains."""

    TEXT = "[A] One | [B] Two | [C] Three | [D] Four"
    # FixedWidthFont at 72 pt and 10 dpi: 10 px a character, rows 12 px apart.
    CHARACTER_WIDTH = 10.0
    ROW_HEIGHT = 12.0

    @classmethod
    def add_instructions(cls, viewer, text=TEXT):
        viewer.hud_layout.update({"instr_x": 10.0, "instr_right_padding": 10.0})
        viewer.instr_text = SimpleNamespace(
            text=text,
            _font=FixedWidthFont(),
            transforms=SimpleNamespace(dpi=10.0),
            font_size=72.0,
            _line_height=1.2,
        )
        viewer._instruction_logical_text = text
        return viewer.instr_text

    def make_viewer(self, width):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = {}
        viewer.canvas = SimpleNamespace(size=(width, 300))
        self.add_instructions(viewer)
        return viewer

    def assert_rows_end_before(self, viewer, right_edge):
        """Every row ends a padding before right_edge, and only actions break."""
        rows = viewer.instr_text.text.split("\n")
        for row in rows:
            self.assertLessEqual(
                viewer.hud_layout["instr_x"] + len(row) * self.CHARACTER_WIDTH,
                right_edge - viewer.hud_layout["instr_right_padding"],
                row,
            )
        self.assertEqual(HUD_INSTRUCTION_SEPARATOR.join(rows), self.TEXT)
        return rows

    def test_rows_break_between_actions_and_drop_the_separator(self):
        wrapped = _wrap_instructions_for_display(self.TEXT, max_width=20, measure_width=len)

        self.assertEqual(wrapped.split("\n"), ["[A] One | [B] Two", "[C] Three | [D] Four"])

    def test_a_line_that_fits_is_unchanged(self):
        self.assertEqual(_wrap_instructions_for_display(self.TEXT, 40, len), self.TEXT)
        self.assertEqual(_wrap_instructions_for_display("", 40, len), "")

    def test_an_action_wider_than_a_row_wraps_at_its_spaces(self):
        text = "[A] One | [LeftClick + Drag] Pan the view"

        rows = _wrap_instructions_for_display(text, max_width=12, measure_width=len).split("\n")

        self.assertEqual(rows[0], "[A] One")
        self.assertTrue(all(len(row) <= 12 for row in rows), rows)
        self.assertEqual("".join(rows[1:]), "[LeftClick + Drag] Pan the view")

    def test_rows_end_before_the_sidebar_toggle_and_rewrap(self):
        viewer = self.make_viewer(width=300)
        viewer._panel_w = 100
        viewer.right_panel = SimpleNamespace(isVisible=lambda: False)
        viewer.toggle_sidebar_btn = SidebarToggle(300 - 40)

        viewer.update_instruction_layout()
        hidden_rows = self.assert_rows_end_before(viewer, 260)
        self.assertEqual(len(hidden_rows), 2)

        # Shown, the sidebar puts the toggle 100 px further left.
        viewer.right_panel = SimpleNamespace(isVisible=lambda: True)
        viewer.toggle_sidebar_btn = SidebarToggle(300 - 100 - 40)
        viewer.update_instruction_layout()
        self.assertEqual(len(self.assert_rows_end_before(viewer, 160)), 4)

        viewer.canvas.size = (600, 300)
        viewer.right_panel = SimpleNamespace(isVisible=lambda: False)
        viewer.toggle_sidebar_btn = SidebarToggle(600 - 40)
        viewer.update_instruction_layout()
        self.assertEqual(viewer.instr_text.text, self.TEXT)

    def test_rows_keep_the_padding_clear_before_the_toggle(self):
        # The 400 px line fits when the toggle leaves instr_x + 400 + padding.
        viewer = self.make_viewer(width=600)
        for toggle_x, rows in ((420, 1), (419, 2)):
            viewer.toggle_sidebar_btn = SidebarToggle(toggle_x)
            viewer.update_instruction_layout()
            with self.subTest(toggle_x=toggle_x):
                self.assertEqual(len(self.assert_rows_end_before(viewer, toggle_x)), rows)

    def test_rows_end_before_the_sidebar_without_a_toggle(self):
        viewer = self.make_viewer(width=300)
        viewer._panel_w = 100
        viewer.right_panel = SimpleNamespace(isVisible=lambda: True)

        viewer.update_instruction_layout()

        self.assertEqual(len(self.assert_rows_end_before(viewer, 200)), 3)

    def test_a_new_font_size_rewraps_the_rows(self):
        # A display's DPI change resizes HUD text (_apply_vispy_text_scaling).
        viewer = self.make_viewer(width=300)
        viewer.update_instruction_layout()
        self.assertEqual(viewer.instr_text.text.count("\n"), 1)

        viewer.instr_text.font_size = 144.0
        viewer.update_instruction_layout()

        # Twice the width a character: rows fit the 280 px room in 14 characters.
        rows = viewer.instr_text.text.split("\n")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(2 * len(row) * self.CHARACTER_WIDTH <= 280.0 for row in rows), rows)

    def test_console_moves_down_by_the_rows_the_instructions_gain(self):
        viewer = ConsoleOverlayWrappingTests.make_viewer(size=(600, 300), text="Cmd: _")
        status = ConsoleOverlayWrappingTests.add_background_job_status(viewer, "Background job done")
        self.add_instructions(viewer)
        configured = (viewer.hud_layout["console_text_x"], viewer.hud_layout["console_text_y"])

        viewer.update_instruction_layout()
        viewer.update_console_background()
        self.assertEqual(viewer.instr_text.text, self.TEXT)
        self.assertEqual(viewer.console_text.pos, configured)
        box_y, status_y = viewer.console_bg.center[1], status.pos[1]

        viewer.canvas.size = (250, 300)
        viewer.update_instruction_layout()
        viewer.update_console_background()

        self.assertEqual(viewer.instr_text.text.count("\n"), 1)
        self.assertEqual(viewer.console_text.pos, (configured[0], configured[1] + self.ROW_HEIGHT))
        self.assertAlmostEqual(viewer.console_bg.center[1], box_y + self.ROW_HEIGHT)
        self.assertAlmostEqual(status.pos[1], status_y + self.ROW_HEIGHT)

    @staticmethod
    def add_close_button(viewer):
        viewer.hud_layout.update({"instr_y": 10.0, "instr_close_gap": 6.0})
        viewer.close_instructions_btn = CloseButton(20)
        return viewer.close_instructions_btn

    def test_the_close_button_follows_the_first_row_and_every_row_leaves_it_room(self):
        viewer = self.make_viewer(width=300)
        viewer.toggle_sidebar_btn = SidebarToggle(230)
        button = self.add_close_button(viewer)

        viewer.update_instruction_layout()

        # 210 px would hold "[C] Three | [D] Four" (200 px) on the second row;
        # the button and its gap take 26 of them, so it breaks once more.
        rows = self.assert_rows_end_before(viewer, 230 - 26)
        self.assertEqual(rows, ["[A] One | [B] Two", "[C] Three", "[D] Four"])
        # A gap after the first row, centred on the row (12 px rows, a 20 px button).
        self.assertEqual(button.pos, (10 + 170 + 6, 6))
        self.assertTrue(button.shown)
        self.assertLessEqual(button.pos[0] + button.size, 230 - viewer.hud_layout["instr_right_padding"])

    def test_closing_hides_the_line_and_its_button_and_moves_the_console_back_up(self):
        viewer = ConsoleOverlayWrappingTests.make_viewer(size=(250, 300), text="Cmd: _")
        viewer.hud_displays = {}
        self.add_instructions(viewer)
        button = self.add_close_button(viewer)
        configured = (viewer.hud_layout["console_text_x"], viewer.hud_layout["console_text_y"])
        viewer._update_hud_elements()
        self.assertEqual(viewer.instr_text.text.count("\n"), 1)
        self.assertEqual(viewer.console_text.pos, (configured[0], configured[1] + self.ROW_HEIGHT))

        # Nothing is saved, so the line shows again when the Viewer next starts.
        with mock.patch("builtins.open", side_effect=AssertionError("closing saved a file")):
            viewer.close_instructions()

        self.assertFalse(viewer.instr_text.visible)
        self.assertFalse(button.shown)
        self.assertEqual(viewer.console_text.pos, configured)
        # It stays closed through a resize.
        viewer.canvas.size = (600, 300)
        viewer._update_hud_elements()
        self.assertFalse(viewer.instr_text.visible)
        self.assertFalse(button.shown)
        self.assertEqual(viewer.console_text.pos, configured)

    def test_a_new_viewer_shows_the_line_and_its_button_again(self):
        closed = self.make_viewer(width=300)
        closed.canvas.update = lambda: None
        closed.hud_displays = {}
        self.add_close_button(closed)
        closed.close_instructions()

        viewer = self.make_viewer(width=300)
        button = self.add_close_button(viewer)
        viewer.update_instruction_layout()

        self.assertTrue(button.shown)
        self.assertIsNot(getattr(viewer.instr_text, "visible", True), False)

    def test_hud_update_wraps_the_instructions_before_placing_the_console(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(size=(300, 300), update=lambda: None)
        viewer.hud_displays = {}
        calls = []
        viewer.update_instruction_layout = lambda: calls.append("instructions")
        viewer.update_console_background = lambda: calls.append("console")

        viewer._update_hud_elements()

        self.assertEqual(calls, ["instructions", "console"])

    def test_resize_moves_the_sidebar_toggle_before_the_hud(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.slider_overlay = object()
        calls = []
        viewer.position_slider_overlay = lambda: calls.append("slider")
        viewer.reposition_expand_btn = lambda: calls.append("toggle")
        viewer._update_hud_elements = lambda: calls.append("hud")

        viewer.on_resize(None)

        self.assertEqual(sorted(calls), ["hud", "slider", "toggle"])
        self.assertLess(calls.index("toggle"), calls.index("hud"))


class ShippedInstructionLineTests(unittest.TestCase):
    """The instruction line as shipped, measured in the bundled Noto Sans at
    the HUD's 16 px, fits a default 1200 x 800 window once wrapped."""

    def test_the_english_line_wraps_between_actions_left_of_the_toggle(self):
        from PySide6.QtGui import QRawFont
        from PySide6.QtWidgets import QApplication

        from desktop.Desktop_App import DESKTOP_FONT_DIR, UI_REGULAR_FILE

        QApplication.instance() or QApplication([])
        font = QRawFont(str(DESKTOP_FONT_DIR / UI_REGULAR_FILE), 16)
        self.assertTrue(font.isValid())

        def measure(visual, text):
            glyphs = font.glyphIndexesForString(text)
            advances = font.advancesForGlyphIndexes(glyphs, QRawFont.LayoutFlag.KernedAdvances)
            return sum(advance.x() for advance in advances)

        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = collections.defaultdict(
            float, {"instr_x": 10.0, "instr_right_padding": 10.0, "font_size_px": 16.0}
        )
        viewer.canvas = SimpleNamespace(size=(1200, 800), scene=None)
        viewer.vispy_ui_face = viewer.vispy_monospace_face = "NotoSans"
        with mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Text", SimpleNamespace), \
                mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Rectangle", SimpleNamespace):
            viewer.create_hud()
        line = viewer._instruction_logical_text
        self.assertEqual(viewer.instr_text.text, line)

        # The Viewer opens with the sidebar hidden and its toggle at the right
        # edge; shown, the English sidebar (137 px) moves the toggle left.
        viewer._panel_w = 137
        for shown, toggle_x in ((False, 1160), (True, 1023)):
            viewer.right_panel = SimpleNamespace(isVisible=lambda shown=shown: shown)
            viewer.toggle_sidebar_btn = SidebarToggle(toggle_x)
            room = toggle_x - 10.0 - 10.0
            with self.subTest(sidebar_shown=shown), \
                    mock.patch.object(EMAPSSN_Viewer, "_vispy_text_width_pixels", measure), \
                    mock.patch.object(EMAPSSN_Viewer, "_vispy_text_line_height_pixels", lambda visual: 21.0):
                self.assertGreater(measure(None, line), room)
                viewer.update_instruction_layout()
                rows = viewer.instr_text.text.split("\n")
                self.assertGreater(len(rows), 1)
                self.assertEqual(HUD_INSTRUCTION_SEPARATOR.join(rows), line)
                for row in rows:
                    self.assertLessEqual(measure(None, row), room, row)


class InstructionCloseButtonTests(unittest.TestCase):
    """The instruction line's close button, as the Viewer's main window builds it."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        from tests.theme_fixture import apply_theme_for_class

        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def test_a_bare_cross_that_leaves_the_keyboard_with_the_canvas(self):
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QWidget

        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(1200, 800))
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer.main_window.show()
        self.app.processEvents()

        button = viewer.close_instructions_btn
        self.assertIs(button.parentWidget(), viewer.canvas.native)
        self.assertFalse(button.icon().isNull())
        self.assertEqual(button.toolTip(), "Hide these instructions")
        self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)
        # The theme's buttons are at least 28 px tall; this one stays a 20 px square.
        self.assertEqual(button.size().toTuple(), (20, 20))

        closed = []
        viewer._update_hud_elements = lambda: closed.append(viewer._instructions_closed)
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
