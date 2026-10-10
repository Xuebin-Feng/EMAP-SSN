"""The zoom command (commands/zoom.py): the view width it accepts and the camera
rectangle it sets, and what it leaves alone when it refuses a width."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import zoom as zoom_command
from tests.command_fixtures import reported_outcomes


ORIGINAL_RECT = SimpleNamespace(pos=(0.0, 0.0), width=100.0, height=50.0)


class FakeCamera:
    """Holds a rectangle (left, bottom, width, height) as the vispy camera does.

    With fail_next set, the next assignment stores its rectangle and then raises,
    as the real camera does when its transform cannot be built.
    """

    def __init__(self, rect=(0.0, 0.0, 100.0, 50.0), fail_next=False):
        self._rect = rect
        self.fail_next = fail_next

    @property
    def rect(self):
        left, bottom, width, height = self._rect
        return SimpleNamespace(pos=(left, bottom), width=width, height=height)

    @rect.setter
    def rect(self, value):
        self._rect = tuple(value)
        if self.fail_next:
            self.fail_next = False
            raise np.linalg.LinAlgError("Singular matrix")


def zoom_viewer(canvas_size=(200, 100), camera=None):
    """A 100 x 50 camera view centered on (50, 25)."""
    return SimpleNamespace(
        canvas=SimpleNamespace(size=canvas_size),
        view=SimpleNamespace(camera=camera if camera is not None else FakeCamera()),
        _hud_timer=SimpleNamespace(start=mock.Mock()),
        console_text=SimpleNamespace(text=""),
    )


class ZoomCommandTests(unittest.TestCase):
    def run_zoom(self, width, canvas_size=(200, 100), camera=None):
        viewer = zoom_viewer(canvas_size, camera)
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            zoom_command.run(viewer, [width])
        return viewer, succeeded, failed

    def assert_refused(self, viewer, succeeded, failed, message):
        self.assertEqual(viewer.view.camera.rect, ORIGINAL_RECT)
        viewer._hud_timer.start.assert_not_called()
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn(message, failed.call_args.args[1])

    def test_width_must_be_a_positive_finite_number(self):
        # float() parses all of these; none of them gives a usable view.
        for width in ("0", "-5", "nan", "inf", "-inf", "1e400"):
            with self.subTest(width=width):
                self.assert_refused(
                    *self.run_zoom(width), "Zoom width must be a positive, finite number."
                )

    def test_text_width_is_refused(self):
        self.assert_refused(*self.run_zoom("wide"), "Zoom width must be a valid number.")

    def test_canvas_without_area_is_refused_without_dividing_by_zero(self):
        for canvas_size in ((200, 0), (0, 100)):
            with self.subTest(canvas_size=canvas_size):
                self.assert_refused(
                    *self.run_zoom("500", canvas_size), "The canvas has no visible area"
                )

    def test_width_that_overflows_the_view_range_is_refused(self):
        # Finite, but a 1:3 canvas makes the view 3e308 high, which overflows.
        self.assert_refused(
            *self.run_zoom("1e308", canvas_size=(100, 300)), "Zoom width is too large"
        )

    def test_widths_beyond_the_float32_scale_range_are_refused(self):
        # 1e50 once made the camera's matrix singular after its rectangle had changed.
        for width in ("1e50", "1e308"):
            with self.subTest(width=width):
                self.assert_refused(*self.run_zoom(width), "Zoom width is too large")

    def test_width_too_small_for_the_view_centre_is_refused(self):
        # Centred at (50, 25), a view under about 1e-3 units wide is not drawn to within a pixel.
        self.assert_refused(*self.run_zoom("1e-14"), "Zoom width is too small")

    def test_a_narrow_width_the_centre_can_hold_is_set_exactly(self):
        viewer, succeeded, failed = self.run_zoom("0.01")

        self.assertEqual(viewer.view.camera.rect.width, 0.01)
        self.assertEqual(viewer.view.camera.rect.height, 0.01 / 2.0)
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Zoom snapped to View Width: 0.01")

    def test_width_sets_the_range_around_the_current_center(self):
        viewer, succeeded, failed = self.run_zoom("500")

        # 500 wide on a 2:1 canvas is 250 high, centered on (50, 25), with no margin added.
        self.assertEqual(
            viewer.view.camera.rect,
            SimpleNamespace(pos=(-200.0, -100.0), width=500.0, height=250.0),
        )
        viewer._hud_timer.start.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Zoom snapped to View Width: 500.0")

    def test_a_camera_that_fails_to_apply_the_view_gets_its_previous_view_back(self):
        viewer, succeeded, failed = self.run_zoom("500", camera=FakeCamera(fail_next=True))

        self.assertEqual(viewer.view.camera.rect, ORIGINAL_RECT)
        viewer._hud_timer.start.assert_not_called()
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("Zoom could not be applied: Singular matrix", failed.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
