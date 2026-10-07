"""The zoom command (commands/zoom.py): the view width it accepts and the camera
range it sets."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import zoom as zoom_command
from tests.command_fixtures import reported_outcomes


def zoom_viewer(canvas_size=(200, 100)):
    """A 100 x 50 camera view centered on (50, 25)."""
    return SimpleNamespace(
        canvas=SimpleNamespace(size=canvas_size),
        view=SimpleNamespace(camera=SimpleNamespace(
            rect=SimpleNamespace(pos=(0.0, 0.0), width=100.0, height=50.0),
            set_range=mock.Mock(),
        )),
        _hud_timer=SimpleNamespace(start=mock.Mock()),
        console_text=SimpleNamespace(text=""),
    )


class ZoomCommandTests(unittest.TestCase):
    def run_zoom(self, width, canvas_size=(200, 100)):
        viewer = zoom_viewer(canvas_size)
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            zoom_command.run(viewer, [width])
        return viewer, succeeded, failed

    def assert_refused(self, viewer, succeeded, failed, message):
        viewer.view.camera.set_range.assert_not_called()
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

    def test_width_sets_the_range_around_the_current_center(self):
        viewer, succeeded, failed = self.run_zoom("500")

        # 500 wide on a 2:1 canvas is 250 high, centered on (50, 25).
        viewer.view.camera.set_range.assert_called_once_with(
            x=(-200.0, 300.0), y=(-100.0, 150.0)
        )
        viewer._hud_timer.start.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Zoom snapped to View Width: 500.0")


if __name__ == "__main__":
    unittest.main()
