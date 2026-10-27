"""Viewer console overlay (EMAPSSN_Viewer): visual wrapping of the logical
console line, the rounded background box and its safe geometry updates, the
background-job status line below it, and DPI-independent overlay geometry.
As shipped, the command line sits at the top of the canvas, its box level with
the sidebar toggle: the mouse and keyboard controls open from a button by the
threshold slider (desktop.Viewer_Widgets), not from a line above it.
Also the idle-time warm-up of the glyphs those texts draw (MainViewer._warm_text_glyphs)."""

import ast
import collections
import contextlib
import importlib
import io
import math
import os
import sys
import time
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
    MainViewer,
    _apply_safe_rectangle_geometry,
    _vispy_text_cursor_span_pixels,
    _vispy_text_glyph_height_pixels,
    _vispy_text_line_height_pixels,
    _wrap_console_text_for_display,
)
from PySide6 import QtCore, QtWidgets
from utilities import Localization
from utilities.Localization import Message
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
        viewer = self.make_viewer(text="Cmd: ")
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
        viewer = self.make_viewer(size=(260, 160), text="Cmd: ")
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
        viewer = self.make_viewer(size=(260, 160), text="Cmd: select cluster_1")
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
            "status_x_offset": 10.0,
            "status_line_spacing": 38.0,
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
        viewer = self.make_viewer(size=(41, 100), text="Cmd: ")

        viewer.update_console_background()

        self.assertEqual(viewer.console_bg.width, 1.0)
        self.assertEqual(viewer.console_bg.radius, 0.5)


class FakeCaret:
    """Records what the Viewer tells the command line's caret (desktop.Viewer_Caret)."""

    def __init__(self):
        self.shown = None      # (x, top, height) while it is placed, else None
        self.steadied = 0

    def show_at(self, x, top, height):
        self.shown = (x, top, height)

    def steady(self):
        self.steadied += 1

    def away(self):
        self.shown = None


class CommandCaretTests(unittest.TestCase):
    """The command line's cursor is a caret drawn over the text, not a glyph in it.

    The test font is 1 px wide a character and 1 px tall, so a position in
    the console line reads straight off: the text starts at console_text_x
    (30) and the line at console_text_y (60), each wrapped line 1.2 px lower.
    """

    TEXT_X, TEXT_Y, LINE = 30.0, 60.0, 1.2
    PREFIX = len("Cmd: ")

    @staticmethod
    def make_typing_viewer(buffer="select cluster", cursor=None, size=(240, 100)):
        viewer = ConsoleOverlayWrappingTests.make_viewer(size=size, text="")
        viewer.console_mode = True
        viewer.console_bg.visible = True
        viewer.input_buffer = buffer
        viewer.cursor_pos = len(buffer) if cursor is None else cursor
        viewer.command_caret = FakeCaret()
        viewer.canvas.update = mock.Mock()
        return viewer

    @staticmethod
    def key(viewer, name, text=""):
        viewer.on_key_press(SimpleNamespace(key=name, text=text, modifiers=[], handled=False))

    @staticmethod
    def drawn_places(viewer):
        """Where each character of the logical text is drawn: its (line, column); then the end."""
        lines = viewer.console_text.text.split("\n")
        places = [(row, column) for row, line in enumerate(lines) for column in range(len(line))]
        places.append((len(lines) - 1, len(lines[-1])))
        return places

    def test_the_command_line_text_has_no_cursor_glyph(self):
        for cursor in (0, 1, 2, 3):
            with self.subTest(cursor=cursor):
                viewer = self.make_typing_viewer("sda", cursor)
                viewer._update_console_text()
                self.assertEqual(viewer.console_text.text, "Cmd: sda")
                self.assertEqual(str(viewer._command_line_message()), "Cmd: sda")

    def test_the_caret_sits_between_the_characters_at_the_cursor(self):
        for cursor in range(len("select") + 1):
            with self.subTest(cursor=cursor):
                viewer = self.make_typing_viewer("select", cursor)
                viewer._update_console_text()
                self.assertEqual(viewer.command_caret.shown,
                                 (self.TEXT_X + self.PREFIX + cursor, self.TEXT_Y, 1.0))

    def test_the_text_takes_no_room_for_the_cursor(self):
        # The same line, with the cursor anywhere in it, is as wide as the command and its prefix.
        widths = set()
        for cursor in (0, 3, 6):
            viewer = self.make_typing_viewer("select", cursor)
            viewer._update_console_text()
            widths.add(viewer.console_bg.width)
        self.assertEqual(len(widths), 1)

    def test_a_key_that_moves_only_the_cursor_moves_the_caret_and_draws_nothing(self):
        viewer = self.make_typing_viewer("select", 6)
        viewer._update_console_text()
        viewer.canvas.update.reset_mock()
        with mock.patch.object(EMAPSSN_Viewer.Command_Engine, "show_status",
                               wraps=EMAPSSN_Viewer.Command_Engine.show_status) as show_status:
            for name, cursor in (("Left", 5), ("Left", 4), ("Right", 5), ("Home", 0), ("End", 6)):
                with self.subTest(key=name):
                    self.key(viewer, name)
                    self.assertEqual(viewer.cursor_pos, cursor)
                    self.assertEqual(viewer.command_caret.shown[0], self.TEXT_X + self.PREFIX + cursor)
            show_status.assert_not_called()
            viewer.canvas.update.assert_not_called()
            # A key that changes the text rewrites the line and draws the canvas again.
            self.key(viewer, "a", "a")
            show_status.assert_called_once()
            viewer.canvas.update.assert_called_once_with()
        self.assertEqual(viewer.console_text.text, "Cmd: selecta")
        self.assertEqual(viewer.command_caret.shown[0], self.TEXT_X + self.PREFIX + 7)

    def test_each_key_lights_the_caret_again(self):
        viewer = self.make_typing_viewer("ab", 2)
        viewer.command_history = ["earlier"]
        viewer.history_index = 1
        viewer._update_console_text()
        for name, text in (("Left", ""), ("Right", ""), ("Home", ""), ("Delete", ""), ("End", ""), ("x", "x"),
                           ("Backspace", ""), ("Up", ""), ("Down", "")):
            with self.subTest(key=name):
                before = viewer.command_caret.steadied
                self.key(viewer, name, text)
                self.assertEqual(viewer.command_caret.steadied, before + 1)

    def test_the_caret_follows_the_cursor_through_a_wrapped_command(self):
        for width in (100, 112):
            viewer = self.make_typing_viewer(
                "alpha beta gamma delta epsilon zeta eta theta iota kappa", size=(width, 100))
            viewer._update_console_text()
            self.assertGreater(len(viewer.console_text.text.split("\n")), 2)
            for cursor in range(len(viewer.input_buffer) + 1):
                with self.subTest(width=width, cursor=cursor):
                    viewer.cursor_pos = cursor
                    viewer._update_console_text()
                    row, column = self.drawn_places(viewer)[self.PREFIX + cursor]
                    self.assertEqual(viewer.command_caret.shown,
                                     (self.TEXT_X + column, self.TEXT_Y + row * self.LINE, 1.0))

    def test_a_cursor_at_the_end_of_a_wrapped_line_is_at_the_start_of_the_next(self):
        viewer = self.make_typing_viewer("alpha beta gamma delta epsilon zeta", size=(100, 100))
        viewer._update_console_text()
        first = len(viewer.console_text.text.split("\n")[0])
        viewer.cursor_pos = first - self.PREFIX
        viewer._update_console_text()
        self.assertEqual(viewer.command_caret.shown, (self.TEXT_X, self.TEXT_Y + self.LINE, 1.0))
        viewer.cursor_pos -= 1
        viewer._update_console_text()
        self.assertEqual(viewer.command_caret.shown, (self.TEXT_X + first - 1, self.TEXT_Y, 1.0))

    def test_the_caret_follows_a_resize_that_wraps_the_line_elsewhere(self):
        viewer = self.make_typing_viewer("alpha beta gamma delta epsilon zeta eta theta", 30, size=(400, 100))
        viewer._update_console_text()
        wide = viewer.command_caret.shown
        self.assertEqual(wide, (self.TEXT_X + self.PREFIX + 30, self.TEXT_Y, 1.0))
        viewer.canvas.size = (100, 100)
        viewer.update_console_background()
        row, column = self.drawn_places(viewer)[self.PREFIX + 30]
        narrow = viewer.command_caret.shown
        self.assertEqual(narrow, (self.TEXT_X + column, self.TEXT_Y + row * self.LINE, 1.0))
        self.assertNotEqual(narrow, wide)
        viewer.canvas.size = (400, 100)
        viewer.update_console_background()
        self.assertEqual(viewer.command_caret.shown, wide)

    def test_the_caret_keeps_its_place_when_a_click_message_follows_the_command(self):
        viewer = self.make_typing_viewer("select cluster", 3)
        viewer._update_console_text()
        placed = viewer.command_caret.shown
        viewer._show_click_status(Message("Selected: {node}", node="Alpha"))
        self.assertTrue(viewer.console_text.text.startswith("Cmd: select cluster"))
        self.assertIn("Selected: Alpha", viewer.console_text.text)
        self.assertEqual(viewer.command_caret.shown, placed)
        # The next key shows the command alone again, with the caret where the key puts it.
        self.key(viewer, "Right")
        self.assertEqual(viewer.console_text.text, "Cmd: select cluster")
        self.assertEqual(viewer.command_caret.shown, (placed[0] + 1, placed[1], placed[2]))

    def test_the_caret_goes_while_a_message_has_the_console_line(self):
        viewer = self.make_typing_viewer("select", 3)
        viewer._update_console_text()
        self.assertIsNotNone(viewer.command_caret.shown)
        viewer.console_text.text = "Error: nothing is selected"    # a command's feedback, written to the line
        viewer.update_console_background()
        self.assertIsNone(viewer.command_caret.shown)
        self.key(viewer, "Right")                                   # the next key shows the command again
        self.assertIsNotNone(viewer.command_caret.shown)

    def test_the_caret_is_away_while_the_command_line_is_closed(self):
        viewer = self.make_typing_viewer("select", 3)
        viewer._update_console_text()
        viewer.console_mode = False
        viewer.update_console_background()
        self.assertIsNone(viewer.command_caret.shown)
        viewer.console_mode = True
        viewer.console_bg.visible = False
        viewer.update_console_background()
        self.assertIsNone(viewer.command_caret.shown)

    def test_enter_and_escape_take_the_caret_away(self):
        for name in ("Enter", "Escape"):
            with self.subTest(key=name):
                viewer = self.make_typing_viewer("select", 3)
                viewer._update_console_text()
                order = []
                viewer.command_caret.away = lambda order=order: order.append("away")
                viewer.process_command = lambda command, order=order: order.append("run")
                self.key(viewer, name)
                self.assertFalse(viewer.console_mode)
                # Away before the command runs: a command that takes a while leaves no caret blinking.
                self.assertEqual(order, ["away", "run"] if name == "Enter" else ["away"])

    def test_opening_the_command_line_places_the_caret_after_the_prefix(self):
        viewer = self.make_typing_viewer("", 0)
        viewer.console_mode = False
        viewer.console_bg.visible = False
        viewer.command_history = []
        viewer.clear_background_job_status = mock.Mock()
        self.key(viewer, "Enter")
        self.assertTrue(viewer.console_mode)
        self.assertEqual(viewer.console_text.text, "Cmd: ")
        self.assertEqual(viewer.command_caret.shown, (self.TEXT_X + self.PREFIX, self.TEXT_Y, 1.0))

    def test_a_translation_may_put_the_command_anywhere(self):
        for name, template, before in (("prefix", "命令：{command}", len("命令：")),
                                       ("suffix", "{command} <- cmd", 0),
                                       ("around", "[{command}]", 1)):
            with self.subTest(translation=name):
                previous = Localization.set_translator(
                    lambda text, n, template=template: template if text == "Cmd: {command}" else text)
                self.addCleanup(Localization.set_translator, previous)
                viewer = self.make_typing_viewer("select", 2)
                viewer._update_console_text()
                self.assertEqual(viewer.console_text.text, template.format(command="select"))
                self.assertEqual(viewer.command_caret.shown, (self.TEXT_X + before + 2, self.TEXT_Y, 1.0))
                Localization.set_translator(previous)

    def test_a_command_holding_the_private_mark_is_placed_right(self):
        viewer = self.make_typing_viewer("ab", 2)
        viewer._update_console_text()
        self.assertEqual(viewer.console_text.text, "Cmd: ab")
        self.assertEqual(viewer.command_caret.shown[0], self.TEXT_X + self.PREFIX + 2)

    def test_a_viewer_without_a_caret_still_lays_the_line_out(self):
        viewer = self.make_typing_viewer("select", 3)
        del viewer.command_caret
        viewer._update_console_text()
        self.key(viewer, "Left")
        self.assertEqual(viewer.console_text.text, "Cmd: select")

    def test_the_caret_spans_the_glyphs_centred_on_the_letters(self):
        class AccentedFont(FixedWidthFont):
            # An accent standing 40 units above the letters, whose own tops are 1024 * 0.25 = 256 up.
            def __getitem__(self, character):
                glyph = dict(super().__getitem__(character))
                if character in "ÅÉÑŐ":
                    glyph["offset"] = (0.0, 1184.0)
                return glyph

        visual = SimpleNamespace(_font=AccentedFont(), transforms=SimpleNamespace(dpi=72.0),
                                 font_size=100.0, _line_height=1.2)
        # The glyphs reach 296 units up of 256 to the em, and none down: the cursor is that tall...
        inset, height = _vispy_text_cursor_span_pixels(visual)
        self.assertAlmostEqual(height, _vispy_text_glyph_height_pixels(visual))
        self.assertAlmostEqual(height, 100.0 * 296 / 256)
        # ...and sits half of what the accent stands above a "d" below the top of the line.
        self.assertAlmostEqual(inset, 100.0 * 40 / 256 / 2.0)
        # Without font metrics it spans the font's size from the top of the line.
        bare = SimpleNamespace(transforms=SimpleNamespace(dpi=72.0), font_size=10.0, _line_height=1.2)
        self.assertEqual(_vispy_text_cursor_span_pixels(bare), (0.0, 10.0))


def shipped_hud_layout():
    """MainViewer's hud_layout as written in its __init__."""
    with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
        tree = ast.parse(source.read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "hud_layout"):
            return ast.literal_eval(node.value)
    raise AssertionError("MainViewer no longer sets hud_layout")


class ShippedCommandLineTests(unittest.TestCase):
    """The command line as shipped, at the HUD's 16 px in the bundled Noto Sans."""

    # Noto Sans's line advance at 16 px (VisPy's 1.1 line height).
    LINE_HEIGHT = 20.9
    # The sidebar toggle's top (MainViewer.reposition_expand_btn).
    TOGGLE_TOP = 10

    def test_the_command_box_lines_up_with_the_sidebar_toggle(self):
        viewer = ConsoleOverlayWrappingTests.make_viewer(size=(1200, 800), text="Cmd: ")
        viewer.hud_layout = shipped_hud_layout()
        layout = viewer.hud_layout
        viewer.console_bg = StrictRoundedRectangle(
            width=layout["console_bg_min_width"], height=layout["console_bg_height"],
            radius=layout["console_bg_radius"],
        )
        with mock.patch.object(EMAPSSN_Viewer, "_vispy_text_line_height_pixels",
                               lambda visual: self.LINE_HEIGHT):
            viewer.update_console_background()
        box_top = viewer.console_bg.center[1] - viewer.console_bg.height / 2.0
        self.assertAlmostEqual(box_top, self.TOGGLE_TOP, delta=1.0)
        self.assertEqual(viewer.console_text.pos, (layout["console_text_x"], layout["console_text_y"]))
        # No line of instructions sits above it any more.
        self.assertFalse([key for key in layout if key.startswith("instr")])

    def test_the_command_box_stops_short_of_the_frame_rate(self):
        """With Show Frame Rate (FPS) on, a long command wraps before the frame-rate card's room."""
        from desktop.Viewer_Fps import FPS_GAP
        long_text = "Cmd: " + "select " * 600
        edges = {}
        left = ConsoleOverlayWrappingTests.make_viewer().hud_layout["console_bg_left_offset"]
        for overlay, right in ((None, left + 1200 - 40), (SimpleNamespace(widest_width=lambda: 300),
                                                            1200 - 10 - 300 - FPS_GAP)):
            with self.subTest(fps=overlay is not None):
                viewer = ConsoleOverlayWrappingTests.make_viewer(size=(1200, 800), text=long_text)
                viewer.fps_overlay = overlay
                self.assertEqual(viewer._command_line_right(), right)
                viewer.update_console_background()
                box = viewer.console_bg
                edges[overlay is not None] = box.center[0] + box.width / 2.0
                self.assertLessEqual(edges[overlay is not None], right + 0.5)
                self.assertGreater(box.height, viewer.hud_layout["console_bg_height"])  # wrapped
        # The card's room moves the wrap.
        self.assertGreater(edges[False], 1200 - 10 - 300 - FPS_GAP)

    def test_the_command_box_takes_the_theme_colour(self):
        from vispy.color import Color

        from desktop.Studio_Theme import set_theme_color, theme_color, theme_tokens

        self.addCleanup(set_theme_color, theme_color())
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = collections.defaultdict(float, shipped_hud_layout())
        viewer.canvas = SimpleNamespace(size=(1200, 800), scene=None)
        viewer.vispy_ui_face = viewer.vispy_monospace_face = "NotoSans"
        with mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Text", SimpleNamespace), \
                mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Rectangle", SimpleNamespace):
            viewer.create_hud()
        for color in (theme_color(), "teal"):
            set_theme_color(color)
            viewer._color_command_box()
            tokens = theme_tokens(color)
            with self.subTest(color=color):
                # A light tint of the colour inside, the colour itself round it.
                self.assertEqual(Color(viewer.console_bg.color).hex, tokens["tint"])
                self.assertEqual(viewer.console_bg.border_color, tokens["accent"])
        # The HUD is drawn before the Viewer takes Config's theme colour, so the
        # box takes it again once the theme is applied.
        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                    and any(isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "create_hud"
                            for call in ast.walk(node)))

        def lines(name):
            return [call.lineno for call in ast.walk(init) if isinstance(call, ast.Call)
                    and name in (getattr(call.func, "attr", None), getattr(call.func, "id", None))]

        self.assertLess(lines("create_hud")[0], lines("apply_studio_theme")[0])
        self.assertTrue([line for line in lines("_color_command_box") if line > lines("apply_studio_theme")[0]])
        # So does the hidden-node count, in case the session opens with nodes hidden.
        self.assertTrue([line for line in lines("_color_hidden_count") if line > lines("apply_studio_theme")[0]])
        # And the right-drag selection box, drawn with the network before the HUD.
        self.assertLess(lines("draw_network")[0], lines("create_hud")[0])
        self.assertTrue([line for line in lines("_color_selection_box") if line > lines("apply_studio_theme")[0]])

    def test_the_selection_box_takes_the_theme_colour(self):
        from vispy.color import Color

        from desktop.Studio_Theme import set_theme_color, theme_color, theme_tokens

        self.addCleanup(set_theme_color, theme_color())
        viewer = MainViewer.__new__(MainViewer)
        viewer.selection_box = EMAPSSN_Viewer.scene.visuals.Line(method="gl", name="selection_box")
        for color in (theme_color(), "teal", "black"):
            set_theme_color(color)
            viewer._color_selection_box()
            with self.subTest(color=color):
                # The colour the command line's box is edged in; grey in black.
                self.assertEqual(Color(viewer.selection_box.color).hex, theme_tokens(color)["accent"])
        # draw_network colours the box as soon as it builds it.
        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        draw_network = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef) and node.name == "draw_network")
        calls = [call.func.attr for call in ast.walk(draw_network)
                 if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)]
        self.assertIn("_color_selection_box", calls)

    def test_the_hud_has_no_instruction_line(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = collections.defaultdict(float, shipped_hud_layout())
        viewer.canvas = SimpleNamespace(size=(1200, 800), scene=None)
        viewer.vispy_ui_face = viewer.vispy_monospace_face = "NotoSans"
        with mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Text", SimpleNamespace),                 mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Rectangle", SimpleNamespace):
            viewer.create_hud()
        self.assertFalse(hasattr(viewer, "instr_text"))
        self.assertFalse(hasattr(MainViewer, "update_instruction_layout"))


class HudOrderTests(unittest.TestCase):
    """The Qt overlays move before the HUD text that keeps clear of them."""

    def test_resize_places_the_frame_rate_and_jobs_badge_before_the_hud(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.slider_overlay = object()
        calls = []
        viewer.position_slider_overlay = lambda: calls.append("slider")
        # reposition_expand_btn places the sidebar and its toggle, and with them the frame rate
        # and the jobs badge, which the HUD text keeps clear of.
        viewer.reposition_expand_btn = lambda: calls.append("toggle")
        viewer._update_hud_elements = lambda: calls.append("hud")

        viewer.on_resize(None)

        self.assertEqual(sorted(calls), ["hud", "slider", "toggle"])
        self.assertLess(calls.index("toggle"), calls.index("hud"))


# Draws each (y, text) with VisPy as the status stack does, on a hidden canvas
# whose y grows downward as the Viewer's does, and prints where its ink is.
_STATUS_RENDER_SCRIPT = """
import sys
import numpy as np
from vispy import app, scene
app.use_app("pyside6")
anchor, size_px = sys.argv[1], float(sys.argv[2])
for y, text in zip(sys.argv[3::2], sys.argv[4::2]):
    canvas = scene.SceneCanvas(size=(400, 300), show=False, bgcolor="white")
    scene.visuals.Text(text, pos=(390, float(y)), anchor_x="right", anchor_y=anchor,
                       font_size=size_px * 72.0 / canvas.dpi, color="black", parent=canvas.scene)
    image = canvas.render()
    print("CONTEXT OK", flush=True)
    rows = np.where(image[:, :, :3].min(axis=(1, 2)) < 128)[0] * 300.0 / image.shape[0]
    print("INK", rows.min(), rows.max(), flush=True)
    canvas.close()
"""


class StatusStackRenderTests(unittest.TestCase):
    """The status stack as VisPy draws it: Hidden Nodes clear above the View
    Width box, the metadata property clear above Hidden Nodes, and the three
    evenly spaced."""

    def test_the_status_lines_sit_evenly_above_the_box(self):
        import subprocess

        from EMAPSSN_Viewer import _slider_row_centre
        from desktop.Viewer_Widgets import SLIDER_BOX_HEIGHT

        layout = shipped_hud_layout()
        viewer = MainViewer.__new__(MainViewer)
        viewer.hud_layout = layout
        hidden_y = viewer._status_hud_position(0, (400, 300))[1]
        meta_y = viewer._status_hud_position(1, (400, 300))[1]
        row = _slider_row_centre(300)
        box_top = row - SLIDER_BOX_HEIGHT / 2.0
        environment = dict(os.environ)
        # tests/__init__.py defaults Qt to offscreen, which has no OpenGL on
        # Windows; a hidden canvas on the native platform renders.
        environment.pop("QT_QPA_PLATFORM", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        arguments = [layout["status_anchor_y"], str(layout["font_size_px"]),
                     str(hidden_y), "Hidden Nodes: 0", str(meta_y), "Size: 512", str(meta_y), "Length: 512 (gy)"]
        try:
            result = subprocess.run([sys.executable, "-c", _STATUS_RENDER_SCRIPT, *arguments],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    env=environment, timeout=120)
        except subprocess.TimeoutExpired:
            self.skipTest("the GL render probe timed out")
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
        inks = [tuple(map(float, line.split()[1:])) for line in result.stdout.splitlines()
                if line.startswith("INK")]
        self.assertEqual(len(inks), 3, result.stdout + result.stderr[-2000:])
        (hidden_top, hidden_bottom), (size_top, size_bottom), (_meta_top, meta_bottom) = inks
        # Hidden Nodes ends above the box, with room between, but not far off.
        self.assertGreaterEqual(box_top - hidden_bottom, 10, inks)
        self.assertLessEqual(box_top - hidden_bottom, 22, inks)
        # The metadata property, descenders and all, ends clear above it.
        self.assertGreaterEqual(hidden_top - meta_bottom, 6, inks)
        # The box's middle and the two lines' (without descenders) are evenly spaced.
        hidden_middle, size_middle = (hidden_top + hidden_bottom) / 2, (size_top + size_bottom) / 2
        self.assertAlmostEqual(row - hidden_middle, hidden_middle - size_middle, delta=2.0, msg=inks)


class FakeClock:
    """The clock a warm-up slice reads (EMAPSSN_Viewer.time): it moves only when a font says so."""

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def __getattr__(self, name):
        return getattr(time, name)


class GlyphRecordingFont(FixedWidthFont):
    """A VisPy font that records each character indexed from it, as VisPy builds a glyph on first sight.

    Each build costs `cost` seconds of the clock it is given (none by default), `fails_on` is a character
    whose glyph cannot be built, and `after_first_glyph` runs once the first glyph is built.
    """

    def __init__(self, clock=None, cost=0.0, fails_on=None, after_first_glyph=None):
        self.built = []
        self.clock = clock
        self.cost = cost
        self.fails_on = fails_on
        self.after_first_glyph = after_first_glyph

    def __getitem__(self, character):
        if character == self.fails_on:
            raise RuntimeError("the glyph atlas is full")
        self.built.append(character)
        if self.clock is not None:
            self.clock.now += self.cost
        if self.after_first_glyph is not None and len(self.built) == 1:
            self.after_first_glyph()
        return super().__getitem__(character)


class GlyphWarmupTests(unittest.TestCase):
    """MainViewer._warm_text_glyphs: the glyphs of the console line and the status line, built while idle.

    The Viewer is a stand-in with the two texts and a window to own the timer, the fonts record what they
    are asked to build, and the timer is the real one, served by processing events.
    """

    MODULES = ["Viewer_Command_Portal", "Command_Declarations"]
    # What a language without a font of its own warms, in the order it warms it.
    ALL_CHARS = list(dict.fromkeys(EMAPSSN_Viewer.GLYPH_EXTENT_CHARS + EMAPSSN_Viewer.GLYPH_WARMUP_CHARS))
    EXTENT_CHARS = list(EMAPSSN_Viewer.GLYPH_EXTENT_CHARS)

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        # Registered first, so it runs last: the windows the tests made are really gone.
        self.addCleanup(QtCore.QCoreApplication.sendPostedEvents, None, QtCore.QEvent.Type.DeferredDelete)
        self.window = self.make_window()
        language = mock.patch.object(EMAPSSN_Viewer, "installed_language", return_value=None)
        self.language = language.start()
        self.addCleanup(language.stop)

    def make_window(self):
        window = QtWidgets.QWidget()  # never shown: it owns the warm-up's timer

        def discard():
            with contextlib.suppress(RuntimeError):  # a test may have deleted it already
                window.deleteLater()

        self.addCleanup(discard)
        return window

    def make_viewer(self, console_font, status_font, window=None):
        return SimpleNamespace(
            main_window=window or self.window,
            console_text=SimpleNamespace(_font=console_font),
            background_job_status_text=SimpleNamespace(_font=status_font),
        )

    @contextlib.contextmanager
    def modules_imported(self, failing=()):
        """The two modules the warm-up imports are not really imported; the attempts are recorded."""
        real = importlib.import_module
        self.imports = []

        def import_module(name, package=None):
            if name not in self.MODULES:
                return real(name, package)
            self.imports.append(name)
            if name in failing:
                raise ImportError(f"No module named {name!r}")
            return mock.sentinel.module

        with mock.patch.object(EMAPSSN_Viewer.importlib, "import_module", import_module):
            yield

    def serve_events_until(self, done, seconds=10.0):
        deadline = time.monotonic() + seconds
        while not done() and time.monotonic() < deadline:
            QtWidgets.QApplication.processEvents()
        return done()

    def warm(self, viewer, failing_imports=(), clock=None, watch=None, after_call=None):
        """Start the warm-up as the Viewer's own timer does and serve events until it ends.

        Returns what it printed. `after_call` runs as soon as the call returns, `watch` after each timer
        pass. Fails if anything escapes a slot.
        """
        printed = io.StringIO()
        escaped = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(printed))
            stack.enter_context(self.modules_imported(failing_imports))
            stack.enter_context(mock.patch.object(sys, "excepthook", lambda kind, error, trace: escaped.append(error)))
            if clock is not None:
                stack.enter_context(mock.patch.object(EMAPSSN_Viewer, "time", clock))
            MainViewer._warm_text_glyphs(viewer)
            if after_call is not None:
                after_call()
            if watch is not None:
                viewer._glyph_warmup_timer.timeout.connect(watch)
            ended =self.serve_events_until(lambda: hasattr(viewer, "_glyph_warmup_done"))
        self.assertEqual(escaped, [])
        self.assertTrue(ended, "the warm-up did not end")
        return printed.getvalue()

    def test_a_language_without_a_font_of_its_own_builds_every_glyph_of_each_font_once(self):
        for language in (None, "xx"):
            with self.subTest(language=language):
                self.language.return_value = language
                console, status = GlyphRecordingFont(), GlyphRecordingFont()
                viewer = self.make_viewer(console, status)
                printed = self.warm(viewer)
                for font in (console, status):
                    # The 'j' of the measuring glyphs and of ASCII is one glyph, built once.
                    self.assertEqual(sorted(font.built), sorted(self.ALL_CHARS))
                    self.assertEqual(max(collections.Counter(font.built).values()), 1)
                    # The accented capitals, the slowest to build, come first.
                    self.assertEqual(font.built[:len(self.EXTENT_CHARS)], self.EXTENT_CHARS)
                self.assertTrue(viewer._glyph_warmup_done)
                self.assertFalse(viewer._glyph_warmup_timer.isActive())
                self.assertEqual(self.imports, self.MODULES)
                self.assertEqual(printed, "")

    def test_a_language_with_a_font_of_its_own_builds_only_the_glyphs_a_text_is_measured_from(self):
        self.assertIn("zh_CN", EMAPSSN_Viewer.LANGUAGE_FONTS)
        self.language.return_value = "zh_CN"
        console, status = GlyphRecordingFont(), GlyphRecordingFont()
        viewer = self.make_viewer(console, status)
        self.warm(viewer)
        self.assertEqual(console.built, self.EXTENT_CHARS)
        self.assertEqual(status.built, self.EXTENT_CHARS)
        self.assertTrue(viewer._glyph_warmup_done)

    def test_the_glyphs_a_texts_height_is_measured_from_are_built_ahead(self):
        # The cursor's span also measures a "d", which the measuring glyphs do not include.
        for language, chars, measures in ((None, self.ALL_CHARS, (_vispy_text_line_height_pixels,
                                                                  _vispy_text_cursor_span_pixels)),
                                          ("zh_CN", self.EXTENT_CHARS, (_vispy_text_line_height_pixels,))):
            for measure in measures:
                with self.subTest(language=language, measure=measure.__name__):
                    self.language.return_value = language
                    font = GlyphRecordingFont()
                    self.warm(self.make_viewer(font, GlyphRecordingFont()))
                    warmed = set(font.built)
                    self.assertEqual(warmed, set(chars))
                    font.built.clear()
                    visual = SimpleNamespace(_font=font, transforms=SimpleNamespace(dpi=72.0),
                                             font_size=16.0, _line_height=1.2)
                    measure(visual)
                    self.assertTrue(font.built)
                    self.assertLessEqual(set(font.built), warmed)

    def test_visuals_sharing_a_font_build_each_glyph_once(self):
        font = GlyphRecordingFont()
        self.warm(self.make_viewer(font, font))
        self.assertEqual(sorted(font.built), sorted(self.ALL_CHARS))
        self.assertEqual(max(collections.Counter(font.built).values()), 1)

    def test_a_visual_without_a_font_is_left_out(self):
        for status in (SimpleNamespace(), SimpleNamespace(_font=None)):
            with self.subTest(status=status):
                font = GlyphRecordingFont()
                viewer = self.make_viewer(font, None)
                viewer.background_job_status_text = status
                self.warm(viewer)
                self.assertEqual(sorted(font.built), sorted(self.ALL_CHARS))
        bare = SimpleNamespace(main_window=self.window, console_text=SimpleNamespace(),
                               background_job_status_text=SimpleNamespace(_font=None))
        self.warm(bare)
        self.assertTrue(bare._glyph_warmup_done)

    def test_a_glyph_longer_than_a_slice_leaves_one_glyph_to_each_pass_and_serves_events_between(self):
        clock = FakeClock()
        served = []
        font = GlyphRecordingFont(clock, cost=EMAPSSN_Viewer.GLYPH_WARMUP_SLICE_S * 2)
        # An event queued while the first glyph is built waits for that pass to end, and runs before the next.
        font.after_first_glyph = lambda: QtCore.QTimer.singleShot(
            0, lambda: served.append((len(font.built), hasattr(viewer, "_glyph_warmup_done"))))
        viewer = self.make_viewer(font, font)
        passes = []
        # The call itself builds nothing: the timer does, in the passes.
        self.warm(viewer, clock=clock, watch=lambda: passes.append(len(font.built)),
                  after_call=lambda: self.assertEqual(font.built, []))
        total = len(self.ALL_CHARS)
        self.assertEqual(passes, list(range(1, total + 1)))
        self.assertEqual(len(served), 1)
        built_when_served, ended_when_served = served[0]
        self.assertGreater(built_when_served, 0)
        self.assertLess(built_when_served, total)
        self.assertFalse(ended_when_served)

    def test_a_slice_builds_glyphs_until_its_time_is_up(self):
        clock = FakeClock()
        cost = EMAPSSN_Viewer.GLYPH_WARMUP_SLICE_S / 2.5
        per_pass = math.ceil(EMAPSSN_Viewer.GLYPH_WARMUP_SLICE_S / cost)
        self.assertEqual(per_pass, 3)
        font = GlyphRecordingFont(clock, cost=cost)
        passes = []
        self.warm(self.make_viewer(font, font), clock=clock, watch=lambda: passes.append(len(font.built)))
        total = len(self.ALL_CHARS)
        self.assertEqual(passes, [min(built, total) for built in range(per_pass, total + per_pass, per_pass)])
        self.assertEqual(sorted(font.built), sorted(self.ALL_CHARS))

    def test_glyphs_that_build_at_once_take_one_pass(self):
        clock = FakeClock()  # a clock that does not move: no slice ever ends
        font = GlyphRecordingFont(clock)
        passes = []
        self.warm(self.make_viewer(font, font), clock=clock, watch=lambda: passes.append(len(font.built)))
        self.assertEqual(passes, [len(self.ALL_CHARS)])

    def test_a_glyph_that_cannot_be_built_ends_the_warm_up_with_a_warning(self):
        console, status = GlyphRecordingFont(fails_on="A"), GlyphRecordingFont()
        viewer = self.make_viewer(console, status)
        printed = self.warm(viewer)  # warm() also checks that nothing was raised out of the timer
        before = self.ALL_CHARS[:self.ALL_CHARS.index("A")]
        self.assertEqual(console.built, before)
        self.assertEqual(status.built, before)
        self.assertEqual(printed.splitlines(),
                         ["Warning: the glyph of 'A' could not be built ahead (the glyph atlas is full)."])
        self.assertTrue(viewer._glyph_warmup_done)
        self.assertFalse(viewer._glyph_warmup_timer.isActive())

    def test_a_module_that_cannot_be_loaded_ahead_is_a_warning_and_the_glyphs_are_still_built(self):
        for failing in (self.MODULES[:1], self.MODULES[1:], self.MODULES):
            with self.subTest(failing=failing):
                console, status = GlyphRecordingFont(), GlyphRecordingFont()
                viewer = self.make_viewer(console, status)
                printed = self.warm(viewer, failing_imports=failing)
                # Each is tried whatever became of the other, and the glyphs follow.
                self.assertEqual(self.imports, self.MODULES)
                self.assertEqual(printed.splitlines(),
                                 [f"Warning: {name} could not be loaded ahead (No module named {name!r})."
                                  for name in failing])
                for font in (console, status):
                    self.assertEqual(sorted(font.built), sorted(self.ALL_CHARS))
                self.assertTrue(viewer._glyph_warmup_done)

    def test_a_window_that_closes_between_passes_ends_the_warm_up_quietly(self):
        window = self.make_window()
        clock = FakeClock()
        font = GlyphRecordingFont(clock, cost=EMAPSSN_Viewer.GLYPH_WARMUP_SLICE_S * 2)
        viewer = self.make_viewer(font, font, window=window)
        passes = []
        escaped = []
        with contextlib.redirect_stdout(io.StringIO()) as printed, self.modules_imported(), \
                mock.patch.object(sys, "excepthook", lambda kind, error, trace: escaped.append(error)), \
                mock.patch.object(EMAPSSN_Viewer, "time", clock):
            MainViewer._warm_text_glyphs(viewer)
            viewer._glyph_warmup_timer.timeout.connect(lambda: passes.append(len(font.built)))
            self.assertTrue(self.serve_events_until(lambda: len(passes) >= 3))
            window.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            for _ in range(20):
                QtWidgets.QApplication.processEvents()
        self.assertEqual(len(font.built), 3)
        self.assertEqual(passes, [1, 2, 3])
        self.assertFalse(hasattr(viewer, "_glyph_warmup_done"))
        self.assertEqual(escaped, [])
        self.assertEqual(printed.getvalue(), "")

    def test_a_real_texts_glyphs_are_built_so_that_measuring_its_height_builds_none(self):
        self.language.return_value = "zh_CN"  # the four accented capitals and five descenders
        text = EMAPSSN_Viewer.scene.visuals.Text("Cmd: ", font_size=16.0)
        font = text._font
        self.assertEqual(len(font._glyphs), 0)
        viewer = SimpleNamespace(main_window=self.window, console_text=text,
                                 background_job_status_text=SimpleNamespace())
        self.warm(viewer)
        self.assertEqual(sorted(font._glyphs), sorted(set(self.EXTENT_CHARS)))
        measured = SimpleNamespace(_font=font, transforms=SimpleNamespace(dpi=72.0), font_size=16.0)
        with mock.patch.object(type(font), "_load_char", side_effect=AssertionError("a glyph was built")):
            self.assertGreater(_vispy_text_glyph_height_pixels(measured), 0.0)

    def test_only_a_windowed_viewer_schedules_the_warm_up(self):
        """MainViewer.__init__ is a real canvas and web server: its source is read, as the HUD order test does."""
        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        viewer_class = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.ClassDef) and node.name == "MainViewer")
        init = next(node for node in viewer_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        # One place names the method, and it is a one-shot timer after the delay...
        named = [node for node in ast.walk(viewer_class)
                 if isinstance(node, ast.Attribute) and node.attr == "_warm_text_glyphs"]
        self.assertEqual(len(named), 1)
        schedules = [call for call in ast.walk(init)
                     if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "singleShot"
                     and len(call.args) == 2 and call.args[1] is named[0]]
        self.assertEqual(len(schedules), 1)
        self.assertEqual(ast.unparse(schedules[0].args[0]), "GLYPH_WARMUP_DELAY_MS")
        # ...inside the branch taken by every mode but headless.
        windowed = [node for node in ast.walk(init)
                    if isinstance(node, ast.If) and ast.unparse(node.test) == "self.window_mode != HEADLESS"
                    and any(schedules[0] in ast.walk(statement) for statement in node.body)]
        self.assertEqual(len(windowed), 1)


if __name__ == "__main__":
    unittest.main()
