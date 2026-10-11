"""The zoom command (commands/zoom.py): the view width it accepts and the camera
rectangle it sets, and what it leaves alone when it refuses a width. The
Viewer's view width keeps to limits, by the command and by the wheel or a
pinch alike: at least 0.5, and at most the nearest 100 that shows every node,
hidden ones too, as the Viewer opens on them; that widest view follows the
nodes as they move."""

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
        # This viewer has no nodes, so no view width limits (ViewWidthLimitTests) apply.
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


# A 900 x 300 network, a node at each corner of that box and one hidden between.
NETWORK = np.array([[0.0, 0.0], [900.0, 0.0], [0.0, 300.0], [900.0, 300.0], [450.0, 150.0]])


def network_viewer(positions=NETWORK, canvas_size=(1200, 800), camera=None):
    """A viewer showing positions on a canvas, its camera on the view the Viewer opens with."""
    camera = camera or FakeCamera(zoom_command.opening_view(positions))
    viewer = zoom_viewer(canvas_size, camera)
    viewer.pos = np.array(positions, dtype=float)
    viewer.visible_mask = np.ones(len(positions), dtype=bool)
    return viewer


class ViewWidthLimitTests(unittest.TestCase):
    def test_the_viewer_opens_on_the_view_set_range_used_to_draw(self):
        # The network, 5% of its extent each side (at least 10 units), then the
        # camera's set_range margin of 5% of that range each side.
        width, height = 900 + 2 * 45, 300 + 2 * 15
        expected = (-45 - width * 0.05, -15 - height * 0.05, width * 1.1, height * 1.1)
        for got, want in zip(zoom_command.opening_view(NETWORK), expected):
            self.assertAlmostEqual(got, want)
        # A single node still gets 10 units each side.
        self.assertEqual(zoom_command.opening_view([[5.0, 5.0]]), (-6.0, -6.0, 22.0, 22.0))

    def test_the_widest_view_is_the_next_hundred_that_shows_every_node(self):
        # 1089 wide, 363 high: on a 3:2 canvas the width decides.
        self.assertEqual(zoom_command.view_width_limits(network_viewer()), (0.5, 1100.0))
        # A tall network on a wide canvas: its height decides, at the canvas's shape.
        tall = [[0.0, 0.0], [100.0, 1000.0]]
        self.assertEqual(zoom_command.view_width_limits(network_viewer(tall, (1200, 600))),
                         (0.5, 2500.0))  # 1210 high x 2
        # Exactly a multiple stays that multiple, though floating point leaves this
        # one 2900.0000000000005 wide.
        exact = [[0.0, 0.0], [2900.0 / 1.21, 0.0]]
        self.assertGreater(zoom_command.opening_view(exact)[2], 2900.0)
        self.assertEqual(zoom_command.view_width_limits(network_viewer(exact))[1], 2900.0)
        # One node: one step.
        self.assertEqual(zoom_command.view_width_limits(network_viewer([[5.0, 5.0]]))[1], 100.0)

    def test_hidden_nodes_count_and_the_widest_view_follows_moved_nodes(self):
        # A hidden node far out still sets the widest view: 2420 wide.
        far = network_viewer(np.vstack([NETWORK, [[2000.0, 0.0]]]))
        far.visible_mask[-1] = False
        self.assertEqual(zoom_command.view_width_limits(far)[1], 2500.0)
        viewer = network_viewer()
        viewer.pos[1] = [2000.0, 0.0]  # dragged further out: 2420 wide
        self.assertEqual(zoom_command.view_width_limits(viewer)[1], 2500.0)
        viewer.pos[1] = viewer.pos[3] = [100.0, 0.0]  # two dragged in: 594 wide
        self.assertEqual(zoom_command.view_width_limits(viewer)[1], 600.0)

    def test_without_nodes_there_is_no_widest_view(self):
        viewer = zoom_viewer()
        self.assertEqual(zoom_command.view_width_limits(viewer), (0.5, None))
        viewer.pos = np.empty((0, 2))
        self.assertEqual(zoom_command.view_width_limits(viewer), (0.5, None))

    def test_a_zoom_stops_at_a_limit_and_goes_no_further_past_one(self):
        viewer = network_viewer(camera=FakeCamera((0.0, 0.0, 1000.0, 666.0)))
        factor = zoom_command.limited_zoom_factor
        self.assertAlmostEqual(factor(viewer, 2.0), 1.1)        # out to 1100, no further
        self.assertEqual(factor(viewer, 1.05), 1.05)            # within: as asked
        self.assertAlmostEqual(factor(viewer, 1e-5), 0.0005)    # in to 0.5, no further
        self.assertEqual(factor(viewer, [0.5, 0.5]), 0.5)       # the camera's pair form
        viewer.view.camera.rect = (0.0, 0.0, 1500.0, 1000.0)    # past the widest, nodes moved in
        self.assertEqual(factor(viewer, 1.2), 1.0)
        self.assertEqual(factor(viewer, 0.9), 0.9)
        viewer.view.camera.rect = (0.0, 0.0, 0.4, 0.3)          # past the narrowest
        self.assertEqual(factor(viewer, 0.5), 1.0)
        self.assertEqual(factor(viewer, 1.5), 1.5)

    def test_the_command_refuses_a_width_outside_the_limits_and_takes_each_limit(self):
        for width, refused in (("1101", True), ("0.49", True), ("1100", False), ("0.5", False)):
            viewer = network_viewer()
            opened = viewer.view.camera.rect
            with self.subTest(width=width), reported_outcomes() as (succeeded, failed), \
                    redirect_stdout(io.StringIO()):
                zoom_command.run(viewer, [width])
                if refused:
                    self.assertEqual(viewer.view.camera.rect, opened)
                    succeeded.assert_not_called()
                    self.assertEqual(
                        str(failed.call_args.args[1]),
                        "Error: Zoom width must be between 0.5 and 1100, "
                        "the widest view that shows every node.",
                    )
                else:
                    failed.assert_not_called()
                    self.assertEqual(viewer.view.camera.rect.width, float(width))


class ViewerWheelZoomTests(unittest.TestCase):
    """The Viewer's camera: a wheel or pinch zoom goes through camera.zoom, which keeps to the limits."""

    def test_the_cameras_wheel_and_pinch_zoom_through_its_zoom_method(self):
        import inspect

        from vispy.scene.cameras import PanZoomCamera

        source = inspect.getsource(PanZoomCamera.viewbox_mouse_event)
        for event in ("'mouse_wheel'", "'gesture_zoom'"):
            branch = source.split(event, 1)[1].split("elif", 1)[0]
            self.assertIn("self.zoom(", branch, event)

    def test_the_viewer_opens_on_every_node_and_limits_its_cameras_zoom(self):
        import ast

        import EMAPSSN_Viewer

        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
                    and node.name == "MainViewer").body
        init = next(node for node in init if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        calls = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Call)]
        self.assertIn("self._limit_camera_zoom()", calls)
        assigned = [ast.unparse(node) for node in ast.walk(init) if isinstance(node, ast.Assign)]
        self.assertIn("self.view.camera.rect = opening_view(self.pos)", assigned)
        self.assertFalse([call for call in calls if "set_range" in call])

    def test_a_wheel_zoom_stops_at_the_widest_and_narrowest_view(self):
        from vispy.scene.cameras import PanZoomCamera

        import EMAPSSN_Viewer

        camera = PanZoomCamera(aspect=1)
        viewer = network_viewer(camera=camera)
        camera.rect = zoom_command.opening_view(NETWORK)
        EMAPSSN_Viewer.MainViewer._limit_camera_zoom(viewer)
        for _ in range(20):
            camera.zoom(1.5, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 1100.0)
        camera.zoom(0.5, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 550.0)
        for _ in range(40):
            camera.zoom(0.25, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 0.5)
        # Dragging a node out lets the view widen with it.
        viewer.pos[1] = [3000.0, 0.0]  # 3630 wide
        for _ in range(30):
            camera.zoom(2.0, (450.0, 150.0))
        self.assertAlmostEqual(camera.rect.width, 3700.0)


if __name__ == "__main__":
    unittest.main()
