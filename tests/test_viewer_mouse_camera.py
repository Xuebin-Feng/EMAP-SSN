# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's mouse movement, rotation, camera wheel, and capture gestures
(EMAPSSN_Viewer.py: on_mouse_press, on_mouse_move, on_mouse_wheel,
on_mouse_release): Shift straight-line move snapping, wheel rotation during
drag, Ctrl + right-drag turning, Ctrl + wheel turning, real-canvas camera
wheel isolation, and running print-capture invalidation."""
import importlib
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vispy import scene  # noqa: E402
from vispy.util import keys  # noqa: E402

import Command_Engine  # noqa: E402
import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from Command_Queue import CommandQueue  # noqa: E402
from desktop.Viewer_Zoom import SmoothWheelZoom  # noqa: E402
from EMAPSSN_Viewer import (  # noqa: E402
    MOVE_SNAP_DEGREES, ROTATE_DEAD_ZONE_PIXELS, ROTATE_DEGREES_PER_NOTCH,
    ROTATE_SNAP_DEGREES, WHEEL_TURN_IDLE_SECONDS,
)
from tests.mouse_gesture_fixtures import (  # noqa: E402
    CTRL, GestureTestCase, NODE, SHIFT, ViewTransform, ZoomTimer,
)
from tests.timer_fixtures import FakeTimer  # noqa: E402


# =====================================================================
# Shift holds a move of nodes to one of eight directions
# =====================================================================

class StraightMoveTests(GestureTestCase):
    def drag_to(self, viewer, x, y, modifiers=()):
        """Move the pointer to (x, y); return where the nodes moved from their places at the press."""
        self.move(viewer, x, y, modifiers=modifiers)
        return viewer.pos[[1, 2], :2] - np.array([NODE[1], NODE[2]], dtype=np.float32)

    def test_without_shift_the_nodes_follow_the_pointer(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, 70, 55), [[30, 55]] * 2)

    def test_shift_gives_eight_directions_every_45_degrees(self):
        self.assertEqual(MOVE_SNAP_DEGREES, 45.0)
        for step in range(8):
            degrees = 45.0 * step
            with self.subTest(degrees=degrees):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                offset = 30.0 * np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
                np.testing.assert_allclose(
                    self.drag_to(viewer, NODE[1][0] + offset[0], NODE[1][1] + offset[1], SHIFT),
                    [offset] * 2, atol=1e-3)

    def test_the_pointer_takes_the_nearest_direction_and_the_nodes_follow_its_distance_along_it(self):
        # (the pointer's angle, the line it is held to), and the pointer 40 px from the press.
        for angle, line in ((10, 0), (22, 0), (23, 45), (40, 45), (67, 45), (68, 90), (100, 90), (120, 135),
                            (170, 180), (-10, 0), (-30, -45), (-100, -90), (-150, -135), (-170, -180)):
            with self.subTest(angle=angle):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                x, y = 40.0 * np.cos(np.radians(angle)), 40.0 * np.sin(np.radians(angle))
                along = 40.0 * np.cos(np.radians(angle - line))
                expected = along * np.array([np.cos(np.radians(line)), np.sin(np.radians(line))])
                np.testing.assert_allclose(self.drag_to(viewer, NODE[1][0] + x, NODE[1][1] + y, SHIFT),
                                           [expected] * 2, atol=1e-3)

    def test_the_nodes_slide_along_the_line_as_the_pointer_goes_along_it(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, 70, 10, SHIFT), [[30, 0]] * 2)
        np.testing.assert_allclose(self.drag_to(viewer, 140, 5, SHIFT), [[100, 0]] * 2)
        np.testing.assert_allclose(self.drag_to(viewer, 100, 50, SHIFT), [[55, 55]] * 2, atol=1e-3)   # 45 degrees
        np.testing.assert_allclose(self.drag_to(viewer, 0, 10, SHIFT), [[-40, 0]] * 2, atol=1e-3)     # leftward

    def test_the_press_itself_gives_no_move(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, *NODE[1], modifiers=SHIFT), [[0, 0]] * 2)

    def test_the_line_goes_as_shift_is_let_go_and_pressed_again(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        # (30, 20) is nearest to 45 degrees, which it goes 35.4 px along: 25 across and 25 up.
        np.testing.assert_allclose(self.drag_to(viewer, 70, 20, SHIFT), [[25, 25]] * 2, atol=1e-3)
        np.testing.assert_allclose(self.drag_to(viewer, 71, 21), [[31, 21]] * 2)        # free again
        np.testing.assert_allclose(self.drag_to(viewer, 71, 21, SHIFT), [[26, 26]] * 2, atol=1e-3)

    def test_shift_at_the_press_adds_the_node_and_the_drag_is_held_to_a_line(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(viewer.selected_indices, [1, 2, 3])
        self.assertTrue(viewer.is_multi_dragging)
        self.move(viewer, NODE[3][0] + 5, 50, modifiers=SHIFT)
        np.testing.assert_allclose(viewer.pos[[1, 2, 3], :2] - np.array(NODE[1:4], dtype=np.float32),
                                   [[0, 50]] * 3, atol=1e-3)
        self.release(viewer, NODE[3][0] + 5, 50)
        self.assertEqual(len(viewer.position_history), 1)

    def test_the_picture_follows_the_line_in_both_modes(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.drag_to(viewer, 100, 50, SHIFT)
                np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
                order = viewer._submitted_marker_node_order
                np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])


# =====================================================================
# The wheel turns the nodes of a drag
# =====================================================================

class RotationTests(GestureTestCase):
    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 2])
        self.press(self.viewer, *NODE[1])
        self.pivot = np.array(NODE[1])

    def turned(self, degrees):
        return self.rotated([NODE[1], NODE[2]], self.pivot, degrees)

    def test_the_wheel_turns_five_degrees_a_notch(self):
        self.assertEqual(ROTATE_DEGREES_PER_NOTCH, 5.0)
        for notches in (1.0, -1.0, 3.0):
            with self.subTest(notches=notches):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, notches), *NODE[1])
                self.assert_places(viewer, [1, 2], self.turned(5.0 * notches))

    def test_there_is_no_fine_turn_and_shift_makes_no_difference(self):
        self.assertFalse(hasattr(EMAPSSN_Viewer, "ROTATE_FINE_DEGREES_PER_NOTCH"))
        for modifiers in ([], SHIFT):
            with self.subTest(modifiers=modifiers):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, 2.0), *NODE[1], modifiers=modifiers)
                self.assert_places(viewer, [1, 2], self.turned(10.0))

    def test_a_fraction_of_a_notch_turns_that_fraction(self):
        # A high-resolution wheel or a touchpad reports parts of a notch, and the turn follows.
        for modifiers in ([], SHIFT):
            with self.subTest(modifiers=modifiers):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, 0.1), *NODE[1], modifiers=modifiers)
                self.wheel(viewer, (0.0, 0.15), *NODE[1], modifiers=modifiers)   # turns add up
                self.assert_places(viewer, [1, 2], self.turned(0.25 * 5.0))

    def test_a_sideways_scroll_turns_the_same_way_whatever_keys_are_held(self):
        # macOS, and some mouse drivers on Windows, send Shift + wheel as a horizontal scroll.
        for modifiers in ([], SHIFT):
            with self.subTest(modifiers=modifiers):
                results = {}
                for name, delta in (("up", (0.0, 2.0)), ("sideways up", (2.0, 0.0)),
                                    ("down", (0.0, -2.0)), ("sideways down", (-2.0, 0.0))):
                    viewer = self.make_viewer(selected=[1, 2])
                    self.press(viewer, *NODE[1])
                    self.wheel(viewer, delta, *NODE[1], modifiers=modifiers)
                    results[name] = viewer.pos[[1, 2], :2].copy()
                np.testing.assert_allclose(results["sideways up"], results["up"])
                np.testing.assert_allclose(results["sideways down"], results["down"])
                np.testing.assert_allclose(results["up"], self.turned(10.0), atol=1e-3)
                self.assertFalse(np.allclose(results["up"], results["down"]))

    def test_the_vertical_notches_win_when_both_are_given(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.wheel(viewer, (5.0, 1.0), *NODE[1], modifiers=SHIFT)
        self.assert_places(viewer, [1, 2], self.turned(5.0))

    def test_shift_and_the_wheel_never_move_a_diagonal_move_onto_a_line(self):
        viewer = self.viewer
        self.move(viewer, 70, 20)                        # a diagonal move of (30, 20), no Shift
        for modifiers in (SHIFT, []):
            self.wheel(viewer, (0.0, 1.0), 70, 20, modifiers=modifiers)
        turned = self.turned(10.0)
        self.assert_places(viewer, [1, 2], turned + [30.0, 20.0])
        # The line applies when the pointer next moves: (35, 22) held to 45 degrees, 28.5 along each axis.
        self.move(viewer, 75, 22, modifiers=SHIFT)
        self.assert_places(viewer, [1, 2], turned + [28.5, 28.5])

    def test_the_wheel_takes_the_move_the_last_move_gave_not_the_pointer(self):
        viewer = self.viewer
        self.move(viewer, 70, 10, modifiers=SHIFT)              # held to the line: (30, 0)
        # The wheel event is at another place, and without Shift: neither counts.
        self.wheel(viewer, (0.0, 1.0), 300, 200)
        self.assert_places(viewer, [1, 2], self.turned(5.0) + [30.0, 0.0])

    def test_a_turn_before_any_move_is_about_the_press_and_moves_nothing(self):
        self.wheel(self.viewer, (0.0, 18.0), 500, 500)           # a quarter turn, the pointer elsewhere
        np.testing.assert_allclose(self.viewer.pos[2], [40.0, 40.0], atol=1e-3)
        np.testing.assert_allclose(self.viewer.pos[1], NODE[1], atol=1e-3)

    def test_the_turn_composes_with_moves_before_and_after_it(self):
        viewer = self.viewer
        self.move(viewer, 60, 0)
        self.wheel(viewer, (0.0, 18.0), 60, 0)
        np.testing.assert_allclose(viewer.pos[2], [60.0, 40.0], atol=1e-3)
        self.move(viewer, 60, 10)
        np.testing.assert_allclose(viewer.pos[2], [60.0, 50.0], atol=1e-3)
        np.testing.assert_allclose(viewer.pos[1], [60.0, 10.0], atol=1e-3)

    def test_the_drawing_follows_a_turn_in_both_modes(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, 18.0), *NODE[1], modifiers=SHIFT)
                order = viewer._submitted_marker_node_order
                np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
                np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])


# =====================================================================
# Ctrl + right-drag turns the selection about the press point
# =====================================================================

class TurnByDragTests(GestureTestCase):
    """The selection is nodes 1 and 3, at (40, 0) and (120, 0); its bounding box is centred on (80, 0).

    The press is on node 1 but off its centre, at (42.5, 1.5): that exact place is the pivot, not
    node 1's centre and not the box's. The pointer is at the pivot at the press, so it has no angle
    about it until it first leaves the dead zone; start_turn takes it out to 180 degrees, 40 px
    away, and the angle is measured from there.
    """

    PRESS = np.array([42.5, 1.5])
    PIVOT = PRESS
    BOX_CENTRE = np.array([80.0, 0.0])
    FIRST = 180.0

    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 3])
        self.start = self.places(self.viewer, [1, 3])

    def press_turn(self, viewer=None, modifiers=CTRL, where=None):
        """Press the right button, with Ctrl, on a selected node, at `where` (the press point by default)."""
        viewer = viewer or self.viewer
        self.press(viewer, *(self.PRESS if where is None else where), modifiers=modifiers)
        self.assertTrue(viewer.is_multi_dragging)

    def start_turn(self, viewer=None, modifiers=CTRL):
        """Press and take the pointer out of the dead zone, to FIRST degrees about the pivot: nothing has moved."""
        viewer = viewer or self.viewer
        self.press_turn(viewer, modifiers)
        self.at(self.FIRST, viewer=viewer)

    def at(self, angle, radius=40.0, pivot=None, viewer=None, modifiers=()):
        """Move the pointer to the place at angle degrees about the pivot, radius px out."""
        pivot = self.PIVOT if pivot is None else pivot
        radians = np.radians(angle)
        return self.move(viewer or self.viewer, pivot[0] + radius * np.cos(radians),
                         pivot[1] + radius * np.sin(radians), modifiers=modifiers)

    def round_about(self, degrees, first=FIRST, step=10.0, radius=40.0, viewer=None, modifiers=()):
        """Move the pointer round the pivot from angle `first` by `degrees` more, in steps."""
        for index in range(1, int(round(abs(degrees) / step)) + 1):
            self.at(first + np.sign(degrees) * step * index, radius, viewer=viewer, modifiers=modifiers)

    def turned(self, degrees, pivot=None):
        return self.rotated(self.start, self.PIVOT if pivot is None else pivot, degrees)

    def test_the_pivot_is_the_exact_press_point_not_the_nodes_centre_nor_the_selections(self):
        # Pressed anywhere on a selected node's disc, off its centre: that place is the pivot. The
        # offsets are within the pick (a node of size 10 takes a press about 5 px out).
        for node, offset in ((1, (2.5, 1.5)), (1, (-3.0, 2.0)), (1, (1.0, -3.5)), (1, (-2.0, -2.5)),
                              (1, (3.0, 0.0)), (3, (-2.0, 3.0)), (3, (3.5, -1.0)), (3, (0.0, 0.0))):
            with self.subTest(node=node, offset=offset):
                viewer = self.make_viewer(selected=[1, 3])
                press = np.array(NODE[node]) + offset
                self.press_turn(viewer, where=press)
                np.testing.assert_array_equal(viewer._drag_turn.pivot, press)
                self.at(0, pivot=press, viewer=viewer)                  # out to the first angle: 0 degrees
                self.at(90, pivot=press, viewer=viewer)                 # a quarter turn on
                self.assert_places(viewer, [1, 3], self.rotated(self.start, press, 90.0))
                if any(offset):      # a different pivot would have put the nodes somewhere else
                    for other in (NODE[node], self.BOX_CENTRE):
                        self.assertFalse(np.allclose(self.places(viewer, [1, 3]),
                                                     self.rotated(self.start, other, 90.0), atol=0.5))

    def test_the_pivot_is_the_press_point_whatever_the_selections_extent(self):
        # Nodes 0, 1 and 5 lie at x = 0, 40 and 200: the box's middle is 100 and the mean of the nodes 80.
        viewer = self.make_viewer(selected=[0, 1, 5])
        start = self.places(viewer, [0, 1, 5])
        press = np.array([NODE[5][0] + 2.0, NODE[5][1] + 2.0])
        self.press(viewer, *press, modifiers=CTRL)
        np.testing.assert_array_equal(viewer._drag_turn.pivot, press)
        self.at(0, pivot=press, viewer=viewer)
        for angle in range(10, 100, 10):
            self.at(angle, pivot=press, viewer=viewer)
        self.assert_places(viewer, [0, 1, 5], self.rotated(start, press, 90.0))

    def test_the_nodes_follow_the_pointer_and_only_its_direction_counts(self):
        for degrees in (30, 100, 170, -50, -120, -170):
            with self.subTest(degrees=degrees):
                viewer = self.make_viewer(selected=[1, 3])
                self.start_turn(viewer)
                self.at(self.FIRST + degrees, radius=25 + abs(degrees), viewer=viewer)    # one move, any distance
                self.assert_places(viewer, [1, 3], self.turned(degrees))

    def test_the_angle_is_measured_from_the_first_point_outside_the_dead_zone_and_nothing_jumps(self):
        for first in (0.0, 70.0, 180.0, 250.0, -135.0):
            with self.subTest(first=first):
                viewer = self.make_viewer(selected=[1, 3])
                self.press_turn(viewer)
                self.assertIsNone(viewer._drag_turn.last)             # at the pivot, the pointer has no angle
                self.at(first + 20, radius=6.0, viewer=viewer)        # past the click radius, inside the dead zone
                np.testing.assert_array_equal(self.places(viewer, [1, 3]), self.start)
                self.assertIsNone(viewer._drag_turn.last)
                self.assertEqual(self.status(viewer), [])
                self.at(first, radius=15.5, viewer=viewer)            # the first point outside: the reference
                np.testing.assert_array_equal(self.places(viewer, [1, 3]), self.start)      # exactly: no jump
                self.assertIsNotNone(viewer._drag_turn.last)
                self.assertEqual(viewer._drag_turn.degrees, 0.0)
                self.assertEqual(self.status(viewer), ["Rotate 0°"])
                self.at(first + 90, radius=60.0, viewer=viewer)       # a quarter turn on, from that reference
                self.assert_places(viewer, [1, 3], self.turned(90.0))

    def test_a_fast_drag_that_leaves_the_dead_zone_in_its_first_move_takes_that_move_as_the_reference(self):
        self.press_turn()
        self.at(33.0, radius=80.0)                            # past the click radius and the dead zone at once
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.at(63.0, radius=80.0)
        self.assert_places(self.viewer, [1, 3], self.turned(30.0))

    def test_two_circles_make_720_degrees(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                viewer = self.make_viewer(selected=[1, 3])
                self.start_turn(viewer)
                self.round_about(direction * 540, viewer=viewer)                         # one and a half circles
                self.assertAlmostEqual(viewer._drag_turn.degrees, direction * 540.0, places=6)
                self.assert_places(viewer, [1, 3], self.turned(180.0))
                self.round_about(direction * 180, first=self.FIRST + direction * 540, viewer=viewer)    # to two
                self.assertAlmostEqual(viewer._drag_turn.degrees, direction * 720.0, places=6)
                self.assert_places(viewer, [1, 3], self.start)
                self.assertEqual(self.status(viewer)[-1], "Rotate 0°")     # where they stand, not 720

    def test_one_circle_makes_360_degrees(self):
        self.start_turn()
        self.round_about(360)
        self.assertAlmostEqual(self.viewer._drag_turn.degrees, 360.0, places=6)
        self.assertEqual(self.status(self.viewer)[-1], "Rotate 0°")
        self.assert_places(self.viewer, [1, 3], self.start)

    def test_a_step_across_the_half_turn_is_taken_the_short_way(self):
        self.start_turn()
        self.at(190)                                       # 10 degrees on, past 180
        self.assertAlmostEqual(self.viewer._drag_turn.degrees, 10.0, places=6)
        self.at(170)                                       # and 20 back, across it again
        self.assertAlmostEqual(self.viewer._drag_turn.degrees, -10.0, places=6)
        self.assert_places(self.viewer, [1, 3], self.turned(-10.0))

    def test_the_angle_holds_within_the_dead_zone_and_goes_on_from_where_it_left(self):
        self.assertEqual(ROTATE_DEAD_ZONE_PIXELS, 15.0)
        self.start_turn()
        self.round_about(40)                               # 40 degrees
        before = self.viewer.pos.copy()
        calls = len(self.status(self.viewer))
        for angle in (10, 100, 200):                       # inside, whatever the direction
            self.at(angle, radius=14.9)
            np.testing.assert_array_equal(self.viewer.pos, before)
        self.assertEqual(len(self.status(self.viewer)), calls)
        self.at(270, radius=30)                            # out again: 50 degrees on from where it left (220)
        self.assert_places(self.viewer, [1, 3], self.turned(90.0))

    def test_the_dead_zone_ends_fifteen_pixels_from_the_press_point(self):
        self.press_turn()
        self.at(210, radius=14.99)
        self.assertIsNone(self.viewer._drag_turn.last)     # still no reference
        self.at(210, radius=15.01)                         # the first point outside: the reference
        self.assertIsNotNone(self.viewer._drag_turn.last)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.at(240, radius=14.99)                         # in again: holds
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.at(240, radius=15.01)
        self.assert_places(self.viewer, [1, 3], self.turned(30.0))

    def test_the_dead_zone_is_round_the_press_point_not_the_selections_centre(self):
        self.start_turn()
        # 5 px from the box's centre, 37 px from the press point: outside the dead zone, so it turns.
        self.move(self.viewer, 80.0, 5.0)
        swept = np.degrees(np.arctan2(5.0 - self.PIVOT[1], 80.0 - self.PIVOT[0])) - self.FIRST
        self.assert_places(self.viewer, [1, 3], self.turned(swept))
        # 10 px from the press point, 37 from the box's centre: inside, so the angle holds.
        before = self.viewer.pos.copy()
        self.move(self.viewer, self.PIVOT[0] - 10.0, self.PIVOT[1])
        np.testing.assert_array_equal(self.viewer.pos, before)

    def test_the_pivot_stays_where_it_was_in_the_network_when_the_view_pans_or_zooms_mid_drag(self):
        viewer = self.viewer
        self.start_turn(viewer)
        self.at(200, viewer=viewer)                        # 20 degrees on
        np.testing.assert_array_equal(viewer._drag_turn.pivot, self.PIVOT)

        def canvas_place(view, angle, radius):
            """The canvas place `radius` px from where the pivot is drawn in this view, at angle in the network."""
            radians = np.radians(angle)
            return view.imap(self.PIVOT)[:2] + radius * np.array([np.cos(radians), np.sin(radians)])

        # The view zooms in twofold and pans: the canvas place c is now c / 2 + (30, -20) in the network.
        zoomed = ViewTransform(0.5, (30.0, -20.0))
        viewer.canvas.scene = SimpleNamespace(node_transform=lambda _scene: zoomed)
        before = viewer.pos.copy()
        self.move(viewer, *canvas_place(zoomed, 250, 14.5))          # inside the dead zone of the pivot as drawn now
        np.testing.assert_array_equal(viewer.pos, before)
        self.move(viewer, *canvas_place(zoomed, 250, 15.5))          # outside: 70 degrees from the reference
        self.assert_places(viewer, [1, 3], self.turned(70.0))
        # It pans and zooms out: still the same place in the network.
        panned = ViewTransform(4.0, (-90.0, 60.0))
        viewer.canvas.scene = SimpleNamespace(node_transform=lambda _scene: panned)
        self.move(viewer, *canvas_place(panned, 300, 30.0))
        self.assert_places(viewer, [1, 3], self.turned(120.0))
        np.testing.assert_array_equal(viewer._drag_turn.pivot, self.PIVOT)
        self.release(viewer, 0, 0)
        self.assert_places(viewer, [1, 3], self.turned(120.0))

    def test_shift_rounds_the_angle_to_steps_of_15_degrees(self):
        self.assertEqual(ROTATE_SNAP_DEGREES, 15.0)
        for degrees, held in ((37, 30), (38, 45), (-37, -30), (-38, -45), (7, 0), (8, 15), (100, 105)):
            with self.subTest(degrees=degrees):
                viewer = self.make_viewer(selected=[1, 3])
                self.start_turn(viewer)
                self.at(self.FIRST + degrees, radius=100, viewer=viewer, modifiers=SHIFT)
                self.assert_places(viewer, [1, 3], self.turned(held))
                self.assertEqual(self.status(viewer)[-1], f"Rotate {held}°")
                self.at(self.FIRST + degrees, radius=101, viewer=viewer)       # Shift let go: free again
                self.assert_places(viewer, [1, 3], self.turned(degrees))

    def test_shift_held_at_the_press_as_well_still_turns_and_removes_nothing(self):
        self.start_turn(modifiers=["Control", "Shift"])
        self.at(215, modifiers=SHIFT)                      # 35 degrees, held to 30
        self.assert_places(self.viewer, [1, 3], self.turned(30.0))
        self.release(self.viewer, 0, 0)
        self.assertEqual(self.viewer.selected_indices, [1, 3])

    def test_the_nodes_wait_for_the_click_radius(self):
        self.press_turn()
        self.move(self.viewer, self.PRESS[0], self.PRESS[1] + 4.9)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.assertEqual(self.status(self.viewer), [])
        self.assertTrue(self.viewer._drag_in_click)           # still a click
        self.move(self.viewer, self.PRESS[0], self.PRESS[1] + 5.0)        # 5 px: a drag, though no turn yet
        self.assertFalse(self.viewer._drag_in_click)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)      # the dead zone holds it
        self.assertEqual(self.status(self.viewer), [])

    def test_a_ctrl_click_without_a_drag_turns_nothing(self):
        self.press_turn()
        self.release(self.viewer, *self.PRESS)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.assertEqual(self.viewer.selected_indices, [1, 3])
        self.assertEqual(self.status(self.viewer), [])
        self.assertFalse(self.viewer.is_multi_dragging)
        self.assertEqual(self.viewer.position_history, [])      # the press's step turned out to be nothing

    def test_a_drag_that_never_leaves_the_dead_zone_leaves_no_undo_step(self):
        viewer = self.viewer
        viewer._save_state()                                      # an earlier step
        earlier = list(viewer.position_history)
        undone = {"pos": "a step that was undone"}
        viewer.redo_stack = [undone]
        self.press_turn()
        self.assertEqual(len(viewer.position_history), 2)
        self.at(33.0, radius=6.0)                                 # past the click radius, inside the dead zone
        self.assertFalse(viewer._drag_in_click)
        self.release(viewer, 0, 0)
        np.testing.assert_array_equal(self.places(viewer, [1, 3]), self.start)
        self.assertEqual(len(viewer.position_history), 1)
        self.assertIs(viewer.position_history[0], earlier[0])
        self.assertEqual(viewer.redo_stack, [undone])             # Ctrl+Y still has its step

    def test_a_turn_that_comes_back_to_where_it_began_leaves_no_undo_step(self):
        self.start_turn()
        self.at(self.FIRST + 90, modifiers=SHIFT)
        self.assertFalse(np.array_equal(self.places(self.viewer, [1, 3]), self.start))
        self.at(self.FIRST, modifiers=SHIFT)                      # Shift: back to 0 exactly
        self.release(self.viewer, 0, 0)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.assertEqual(self.viewer.position_history, [])

    def test_a_turn_that_ends_elsewhere_keeps_its_undo_step(self):
        self.start_turn()
        self.at(self.FIRST + 90)
        self.release(self.viewer, 0, 0)
        self.assertEqual(len(self.viewer.position_history), 1)

    def test_the_release_is_one_undo_step(self):
        self.start_turn()
        self.round_about(90)
        self.release(self.viewer, 0, 0)
        self.assertEqual(len(self.viewer.position_history), 1)
        self.assert_places(self.viewer, [1, 3], self.turned(90.0))
        self.viewer._do_undo(announce=False)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)

    def test_esc_puts_the_nodes_back_and_leaves_no_undo_step(self):
        viewer = self.viewer
        undone = {"pos": "a step that was undone"}
        viewer.redo_stack = [undone]
        self.start_turn()
        self.round_about(90)
        self.assertEqual(len(viewer.position_history), 1)
        with mock.patch.object(Command_Engine, "show_status") as show_status:
            self.assertTrue(self.esc(viewer).handled)
        np.testing.assert_array_equal(viewer.pos[[1, 3], :2], self.start.astype(np.float32))   # exactly
        self.assertEqual(viewer.position_history, [])
        self.assertEqual(viewer.redo_stack, [undone])
        self.assertEqual(viewer.selected_indices, [1, 3])
        self.assertFalse(viewer.is_multi_dragging)
        self.assertIsNone(viewer._drag_turn)
        show_status.assert_called_once_with(viewer, "")              # no angle left on the status line
        self.assert_picture_matches(viewer)
        # The rest of the drag, and its release, do nothing.
        self.assertTrue(self.at(300).handled)
        np.testing.assert_array_equal(viewer.pos[[1, 3], :2], self.start.astype(np.float32))
        self.release(viewer, 0, 0)
        self.assertEqual(viewer.selected_indices, [1, 3])

    def test_a_cancel_before_the_click_radius_is_left_changes_nothing(self):
        self.press_turn()
        self.assertTrue(self.esc(self.viewer).handled)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.assertEqual(self.viewer.position_history, [])

    def test_the_wheel_is_ignored_during_the_turn(self):
        self.start_turn()
        self.round_about(40)
        before = self.viewer.pos.copy()
        wheel = self.wheel(self.viewer, (0.0, 3.0), 60, 20)
        self.assertTrue(wheel.handled)
        np.testing.assert_array_equal(self.viewer.pos, before)
        self.assertAlmostEqual(self.viewer._drag_turn.degrees, 40.0, places=6)
        self.round_about(10, first=self.FIRST + 40)          # and the drag goes on
        self.assertAlmostEqual(self.viewer._drag_turn.degrees, 50.0, places=6)

    def test_it_is_refused_where_moving_is_refused(self):
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            self.press(self.viewer, *self.PRESS, modifiers=CTRL)
            self.assertFalse(self.viewer.is_multi_dragging)
            self.assertIsNone(self.viewer._drag_turn)
            self.at(250)
            self.release(self.viewer, 0, 0)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 3]), self.start)
        self.assertEqual(self.viewer.selected_indices, [1, 3])

    def test_ctrl_on_an_unselected_node_is_a_plain_press_and_moves_it(self):
        self.press(self.viewer, *NODE[2], modifiers=CTRL)
        self.assertEqual(self.viewer.selected_indices, [2])
        self.assertTrue(self.viewer.is_multi_dragging)
        self.assertIsNone(self.viewer._drag_turn)
        self.move(self.viewer, NODE[2][0] + 20, 30)
        np.testing.assert_allclose(self.viewer.pos[2, :2], [100, 30])

    def test_cmd_on_macos_does_what_ctrl_does(self):
        self.start_turn(modifiers=["Meta"])
        self.at(210)
        self.assert_places(self.viewer, [1, 3], self.turned(30.0))

    def test_the_status_line_shows_the_angle_when_it_changes(self):
        self.start_turn()                                    # the turn takes hold at 0 degrees
        self.at(217)
        self.at(217.04)                                      # the same to a tenth of a degree
        self.at(217.5)
        self.at(142.0)                                       # -38
        self.assertEqual(self.status(self.viewer), ["Rotate 0°", "Rotate 37°", "Rotate 37.5°", "Rotate -38°"])

    def test_the_status_line_wraps_the_angle_into_minus_180_to_180_after_circles(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                viewer = self.make_viewer(selected=[1, 3])
                self.start_turn(viewer)
                self.round_about(direction * 1090, viewer=viewer)             # three circles and 10 degrees
                self.assertAlmostEqual(viewer._drag_turn.degrees, direction * 1090.0, places=6)   # the turn is whole
                shown = [float(text.split()[1].rstrip("°")) for text in self.status(viewer)]
                self.assertTrue(all(-180.0 < angle <= 180.0 for angle in shown), shown)
                self.assertEqual(self.status(viewer)[-1], f"Rotate {direction * 10}°")
                self.assert_places(viewer, [1, 3], self.turned(direction * 10.0))        # the nodes are as they were

    def test_shift_steps_stay_multiples_of_15_in_the_wrapped_angle(self):
        viewer = self.make_viewer(selected=[1, 3])
        self.start_turn(viewer)
        self.round_about(820, step=5.0, viewer=viewer, modifiers=SHIFT)           # past two circles
        self.assertEqual(viewer._drag_turn.degrees, 825.0)
        shown = [float(text.split()[1].rstrip("°")) for text in self.status(viewer)]
        self.assertGreater(len(shown), 40)
        for angle in shown:
            self.assertTrue(-180.0 < angle <= 180.0 and angle % ROTATE_SNAP_DEGREES == 0.0, angle)
        self.assertEqual(self.status(viewer)[-1], "Rotate 105°")

    def test_the_wrapped_angle_to_a_tenth_of_a_degree(self):
        # Rounded to a tenth first, then wrapped, so that nothing reads -180 or 180.04, and no "-0".
        cases = {0.0: "0", 2869.3: "-10.7", -1006.5: "73.5", -2.9: "-2.9", 180.0: "180", -180.0: "180",
                 179.96: "180", -179.96: "180", 180.04: "180", 180.1: "-179.9", -180.1: "179.9",
                 359.96: "0", -359.96: "0", -0.04: "0", 360.0: "0", -360.0: "0", 540.0: "180", -540.0: "180",
                 15.0: "15", 195.0: "-165", -195.0: "165"}
        for degrees, expected in cases.items():
            with self.subTest(degrees=degrees):
                viewer = self.make_viewer(selected=[1, 3])
                turn = EMAPSSN_Viewer._Turn(np.array([1, 3], dtype=np.intp), self.start.copy(), self.PIVOT)
                viewer._show_turn(turn, degrees)
                self.assertEqual(self.status(viewer), [f"Rotate {expected}°"])
                self.assertEqual(turn.degrees, degrees)                            # only the display wraps
                self.assert_places(viewer, [1, 3], self.turned(degrees))

    def test_the_turn_goes_on_the_selection_without_changing_it(self):
        self.start_turn()
        self.round_about(90)
        self.release(self.viewer, 0, 0)
        self.assertEqual(self.viewer.selected_indices, [1, 3])
        self.assertEqual(self.viewer.selection_epoch, 0)
        # A plain drag after it is a plain move, from where the node is now.
        node = self.turned(90.0)[0]
        self.press(self.viewer, *node)
        self.assertTrue(self.viewer.is_multi_dragging)
        self.assertIsNone(self.viewer._drag_turn)
        self.move(self.viewer, node[0] + 10, node[1] + 20)
        np.testing.assert_allclose(self.viewer.pos[1, :2] - node, [10, 20], atol=1e-3)

    def test_the_picture_follows_in_both_modes_and_the_web_pages_hear_nothing_meanwhile(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 3])
                self.start_turn(viewer)
                self.round_about(60, viewer=viewer)
                order = viewer._submitted_marker_node_order
                np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
                np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
                self.assertEqual(viewer.broadcast_event.call_count, 1)   # the press's, for the selection
                self.release(viewer, 0, 0)
                self.assert_picture_matches(viewer)


# =====================================================================
# Ctrl + wheel turns the selection about the cursor
# =====================================================================

class TurnByWheelTests(GestureTestCase):
    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 2])
        self.start = self.places(self.viewer, [1, 2])

    def notch(self, delta=(0.0, 1.0), x=60.0, y=30.0, modifiers=CTRL, viewer=None, buttons=()):
        return self.wheel(viewer or self.viewer, delta, x, y, buttons=buttons, modifiers=modifiers)

    def turned(self, degrees, about=(60.0, 30.0), start=None):
        return self.rotated(self.start if start is None else start, about, degrees)

    def timer(self, viewer=None):
        return (viewer or self.viewer)._wheel_turn_timer

    def test_notches_turn_five_degrees_a_notch_about_the_cursor(self):
        self.assertEqual(ROTATE_DEGREES_PER_NOTCH, 5.0)
        for notches in (1.0, -1.0, 3.0, 0.2):
            with self.subTest(notches=notches):
                viewer = self.make_viewer(selected=[1, 2])
                self.notch((0.0, notches), viewer=viewer)
                self.assert_places(viewer, [1, 2], self.turned(5.0 * notches))

    def test_the_event_is_marked_handled(self):
        # Only the canvas's own event: the camera's copy is CameraWheelTests's.
        self.assertTrue(self.notch().handled)
        self.assertTrue(self.notch((0.0, 0.0)).handled)             # even one of no notches

    def test_the_pivot_is_the_cursor_at_the_first_notch_and_stays_while_it_wobbles(self):
        self.notch(x=60, y=30)
        self.notch(x=200, y=100)
        self.notch(x=-30, y=-90)
        self.assert_places(self.viewer, [1, 2], self.turned(15.0))
        np.testing.assert_allclose(self.viewer._wheel_turn.pivot, [60, 30])

    def test_notches_back_and_forth_add_up(self):
        for _ in range(4):
            self.notch((0.0, 1.0))
        for _ in range(4):
            self.notch((0.0, -1.0), x=10, y=10)
        self.assert_places(self.viewer, [1, 2], self.start)

    def test_the_turn_goes_on_while_ctrl_is_held_and_the_timer_restarts_each_notch(self):
        self.notch()
        timer = self.timer()
        self.assertTrue(timer.running)
        self.assertEqual(timer.interval, WHEEL_TURN_IDLE_SECONDS)
        self.assertEqual(WHEEL_TURN_IDLE_SECONDS, 1.0)
        self.notch()
        self.notch()
        self.assertEqual(len(timer.starts), 3)                      # started afresh by every notch
        self.assertEqual(len(FakeTimer.created), 1)                 # the one timer

    def test_a_second_with_no_notch_ends_the_turn_and_the_next_notch_takes_a_new_pivot(self):
        self.notch(x=60, y=30)
        self.timer().fire()
        self.assertIsNone(self.viewer._wheel_turn)
        self.notch(x=10, y=-20)
        first = self.turned(5.0)
        self.assert_places(self.viewer, [1, 2], self.rotated(first, (10.0, -20.0), 5.0))
        self.assertEqual(len(self.viewer.position_history), 2)      # a second undo step

    def test_letting_go_of_ctrl_ends_the_turn(self):
        for key in ("Control", "Meta"):
            with self.subTest(key=key):
                viewer = self.make_viewer(selected=[1, 2])
                self.notch(viewer=viewer)
                self.key_release(viewer, key)
                self.assertIsNone(viewer._wheel_turn)
                self.assertFalse(self.timer(viewer).running)
                self.notch(x=10, y=10, viewer=viewer)
                self.assertEqual(len(viewer.position_history), 2)
                self.assertTrue(np.allclose(viewer._wheel_turn.pivot, [10, 10]))

    def test_another_key_let_go_does_not(self):
        self.notch()
        for key in ("Shift", "a", None):
            self.key_release(self.viewer, key)
        self.assertIsNotNone(self.viewer._wheel_turn)

    def test_a_button_pressed_ends_the_turn(self):
        for button in (1, 2):
            with self.subTest(button=button):
                viewer = self.make_viewer(selected=[1, 2])
                self.notch(viewer=viewer)
                self.press(viewer, 150, 50, button=button)
                self.assertIsNone(viewer._wheel_turn)
                self.assertFalse(self.timer(viewer).running)

    def test_a_wheel_notch_without_ctrl_ends_the_turn_and_zooms(self):
        self.notch()
        plain = self.notch(modifiers=())
        self.assertFalse(plain.handled)                              # the camera has it
        self.assertIsNone(self.viewer._wheel_turn)

    def test_a_turn_is_one_undo_step(self):
        for _ in range(3):
            self.notch()
        self.assertEqual(len(self.viewer.position_history), 1)
        self.timer().fire()
        self.viewer._do_undo(announce=False)
        np.testing.assert_array_equal(self.viewer.pos[[1, 2], :2], self.start.astype(np.float32))
        self.assertEqual(self.viewer.selected_indices, [1, 2])

    def test_esc_puts_the_nodes_back_and_ends_the_turn_with_no_undo_step(self):
        viewer = self.viewer
        undone = {"pos": "a step that was undone"}
        viewer.redo_stack = [undone]
        self.notch()
        self.notch()
        self.assertEqual(len(viewer.position_history), 1)
        with mock.patch.object(Command_Engine, "show_status") as show_status:
            self.assertTrue(self.esc(viewer).handled)
        np.testing.assert_array_equal(viewer.pos[[1, 2], :2], self.start.astype(np.float32))   # exactly
        self.assertEqual(viewer.position_history, [])
        self.assertEqual(viewer.redo_stack, [undone])
        self.assertIsNone(viewer._wheel_turn)
        self.assertFalse(self.timer().running)
        show_status.assert_called_once_with(viewer, "")
        self.assert_picture_matches(viewer)
        self.assertEqual(viewer.selected_indices, [1, 2])
        # The next notch starts a turn of its own, from the places Esc put back.
        self.notch(x=0, y=0)
        self.assert_places(viewer, [1, 2], self.turned(5.0, about=(0.0, 0.0)))
        self.assertEqual(len(viewer.position_history), 1)

    def test_nothing_selected_zooms(self):
        viewer = self.make_viewer()
        before = viewer.pos.copy()
        ev = self.notch(viewer=viewer)
        self.assertFalse(ev.handled)
        self.assertIsNone(viewer._wheel_turn)
        self.assertFalse(hasattr(viewer, "_wheel_turn_timer"))
        self.assertEqual(viewer.position_history, [])
        np.testing.assert_array_equal(viewer.pos, before)

    def test_a_plain_wheel_is_the_cameras(self):
        ev = self.notch(modifiers=())
        self.assertFalse(ev.handled)
        self.assertIsNone(self.viewer._wheel_turn)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 2]), self.start)

    def test_a_button_held_is_no_ctrl_wheel_turn(self):
        ev = self.notch(buttons=(1,))
        self.assertFalse(ev.handled)
        self.assertIsNone(self.viewer._wheel_turn)

    def test_shift_makes_no_difference_and_a_sideways_scroll_turns_the_same(self):
        results = {}
        for name, delta, modifiers in (("ctrl", (0.0, 1.0), CTRL), ("ctrl shift", (0.0, 1.0), CTRL + SHIFT),
                                       ("sideways", (2.0, 0.0), CTRL + SHIFT), ("sideways ctrl", (2.0, 0.0), CTRL),
                                       ("down", (-2.0, 0.0), CTRL + SHIFT)):
            viewer = self.make_viewer(selected=[1, 2])
            self.notch(delta, modifiers=modifiers, viewer=viewer)
            results[name] = viewer.pos[[1, 2], :2].copy()
        np.testing.assert_allclose(results["ctrl shift"], results["ctrl"])
        np.testing.assert_allclose(results["ctrl"], self.turned(5.0), atol=1e-3)
        np.testing.assert_allclose(results["sideways"], self.turned(10.0), atol=1e-3)     # two notches' worth
        np.testing.assert_allclose(results["sideways ctrl"], results["sideways"])
        np.testing.assert_allclose(results["down"], self.turned(-10.0), atol=1e-3)

    def test_cmd_on_macos_does_what_ctrl_does(self):
        self.notch(modifiers=["Meta"])
        self.assert_places(self.viewer, [1, 2], self.turned(5.0))

    def test_it_is_refused_where_moving_is_refused(self):
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            ev = self.notch()
        self.assertFalse(ev.handled)                                 # the camera zooms
        self.assertIsNone(self.viewer._wheel_turn)
        np.testing.assert_array_equal(self.places(self.viewer, [1, 2]), self.start)
        self.assertEqual(self.viewer.position_history, [])

    def test_the_status_line_shows_the_angle(self):
        for _ in range(3):
            self.notch()
        self.notch((0.0, -8.0))
        self.assertEqual(self.status(self.viewer), ["Rotate 5°", "Rotate 10°", "Rotate 15°", "Rotate -25°"])

    def test_the_status_line_wraps_the_angle_into_minus_180_to_180(self):
        self.notch((0.0, 37.0))                                      # 185 degrees
        self.notch((0.0, 35.0))                                      # 360: a whole circle
        self.notch((0.0, 1.0))
        self.notch((0.0, -146.0))                                    # -365
        self.assertEqual(self.status(self.viewer), ["Rotate -175°", "Rotate 0°", "Rotate 5°", "Rotate -5°"])
        self.assertEqual(self.viewer._wheel_turn.degrees, -365.0)    # the turn itself is not wrapped
        self.assert_places(self.viewer, [1, 2], self.turned(-5.0))

    def test_a_turn_that_ends_where_it_began_leaves_no_undo_step(self):
        for end in ("timer", "ctrl let go", "a press"):
            with self.subTest(end=end):
                viewer = self.make_viewer(selected=[1, 2])
                viewer._save_state()                                 # an earlier step
                earlier = list(viewer.position_history)
                undone = {"pos": "a step that was undone"}
                viewer.redo_stack = [undone]
                for _ in range(4):
                    self.notch(viewer=viewer)
                for _ in range(4):
                    self.notch((0.0, -1.0), x=10, y=10, viewer=viewer)
                self.assertEqual(len(viewer.position_history), 2)    # the first notch took the snapshot
                self.assertEqual(viewer.redo_stack, [])
                if end == "timer":
                    self.timer(viewer).fire()
                elif end == "ctrl let go":
                    self.key_release(viewer, "Control")
                else:
                    self.press(viewer, 150, 50, button=1)
                self.assertIsNone(viewer._wheel_turn)
                self.assertEqual(len(viewer.position_history), 1)
                self.assertIs(viewer.position_history[0], earlier[0])
                self.assertEqual(viewer.redo_stack, [undone])        # Ctrl+Y still has its step
                np.testing.assert_array_equal(viewer.pos[[1, 2], :2], self.start.astype(np.float32))

    def test_a_turn_that_moved_the_nodes_keeps_its_step_and_so_does_one_that_something_else_followed(self):
        self.notch()
        self.timer().fire()
        self.assertEqual(len(self.viewer.position_history), 1)
        viewer = self.make_viewer(selected=[1, 2])
        self.notch((0.0, 1.0), viewer=viewer)
        self.notch((0.0, -1.0), viewer=viewer)
        viewer._save_state()                                         # something else was added meanwhile
        self.timer(viewer).fire()
        self.assertEqual(len(viewer.position_history), 2)

    def test_the_picture_follows_in_both_modes_and_the_pick_is_made_again_when_the_turn_ends(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 2])
                self.notch(viewer=viewer)
                order = viewer._submitted_marker_node_order
                np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
                np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
                if drawing:
                    self.assertIsNone(viewer._marker_picker)         # under Update Colors After Mouse Release
                self.timer(viewer).fire()
                self.assert_picture_matches(viewer)
                self.assertEqual(viewer.broadcast_event.call_count, 0)    # the selection did not change

    def test_commands_wait_while_a_turn_lasts_and_go_on_when_it_ends(self):
        viewer = self.viewer
        viewer.command_queue = mock.Mock()
        viewer.command_queue.pending.return_value = [object()]
        self.assertFalse(CommandQueue._busy(None, viewer))
        self.notch()
        self.assertTrue(CommandQueue._busy(None, viewer))
        viewer.command_queue.resume.assert_not_called()
        self.timer().fire()
        self.assertFalse(CommandQueue._busy(None, viewer))
        viewer.command_queue.resume.assert_called_once_with()

    def test_a_cancel_resumes_the_command_queue_too(self):
        viewer = self.viewer
        viewer.command_queue = mock.Mock()
        viewer.command_queue.pending.return_value = [object()]
        self.notch()
        with mock.patch.object(Command_Engine, "show_status"):
            self.esc(viewer)
        viewer.command_queue.resume.assert_called_once_with()


# =====================================================================
# The camera gets the wheel too: Ctrl + wheel with a selection turns it and does not zoom
# =====================================================================

class CameraWheelTests(GestureTestCase):
    """What the camera does with a wheel notch, on a real canvas, wired as MainViewer.__init__ wires it.

    VisPy sends a wheel notch twice: the canvas's own event goes to the Viewer's on_mouse_wheel, and
    the scene sends the camera a SceneMouseEvent of its own, whose `handled` is not the canvas
    event's. So marking the canvas's event handled never kept the camera from zooming; the camera's
    wrapper (_camera_mouse_event) asks the same question as on_mouse_wheel (_wheel_turns_nodes).
    These tests send notches through the canvas, as the platform does, and look at the camera's view.
    """

    RECT = (-50.0, -40.0, 300.0, 240.0)       # what the 300 x 240 canvas shows: a pixel is a unit, y up
    FIRST = (60.0, 30.0)                      # a canvas place, and so the place in the network
    FIRST_PLACE = (10.0, 170.0)               # it is, with the view above
    KEYS = {"Control": keys.CONTROL, "Shift": keys.SHIFT, "Meta": keys.META}

    def make_canvas_viewer(self, selected=(1, 2)):
        viewer = self.make_viewer(selected=list(selected))
        canvas = scene.SceneCanvas(size=(300, 240), bgcolor="white", show=False)
        self.addCleanup(canvas.close)
        view = canvas.central_widget.add_view()
        view.camera = scene.PanZoomCamera(aspect=1)
        view.camera.rect = self.RECT
        canvas.visual_at = lambda pos: view       # VisPy picks by rendering, which this canvas has no GL for
        viewer.canvas, viewer.view = canvas, view
        # The wiring of MainViewer.__init__: the Viewer's wheel handler, the camera's wrapper, the smooth zoom.
        canvas.events.mouse_wheel.connect(viewer.on_mouse_wheel)
        viewer._original_camera_mouse_event = mock.Mock(side_effect=view.camera.viewbox_mouse_event)
        view.camera.viewbox_mouse_event = viewer._camera_mouse_event
        viewer._limit_camera_zoom()
        viewer.zoom_clock, viewer.zoom_timer = [0.0], ZoomTimer()
        viewer._wheel_zoom = SmoothWheelZoom(view.camera, view.camera.zoom, viewer._after_wheel_zoom_step,
                                             clock=lambda: viewer.zoom_clock[0], timer=viewer.zoom_timer)
        return viewer

    @staticmethod
    def animations(off):
        """Low Resource Mode's No Zoom Animation: the camera zooms at once rather than over some frames."""
        return mock.patch.object(cfg, "LOW_RESOURCE_ANIMATIONS", off)

    def scroll(self, viewer, delta=(0.0, 1.0), at=FIRST, modifiers=CTRL, buttons=()):
        """A wheel notch at a canvas place, sent as the platform sends it; the canvas's event comes back."""
        return viewer.canvas.events.mouse_wheel(
            native=None, pos=at, delta=delta, modifiers=tuple(self.KEYS[name] for name in modifiers),
            buttons=list(buttons))

    @staticmethod
    def play_out(viewer):
        """Run the smooth zoom's frames until it ends."""
        for _ in range(200):
            if not viewer.zoom_timer.active:
                return
            viewer.zoom_clock[0] += 0.015
            viewer.zoom_timer.fire()
        raise AssertionError("the smooth zoom does not end")

    @staticmethod
    def view_of(viewer):
        rect = viewer.view.camera.rect
        return (rect.left, rect.bottom, rect.width, rect.height)

    def test_the_view_below_is_the_one_the_tests_assume(self):
        viewer = self.make_canvas_viewer()
        to_network = viewer.canvas.scene.node_transform(viewer.view.scene)
        np.testing.assert_allclose(to_network.map(self.FIRST)[:2], self.FIRST_PLACE, atol=1e-6)
        self.assertEqual(self.view_of(viewer), self.RECT)

    def test_ctrl_wheel_with_a_selection_turns_the_nodes_and_the_view_does_not_zoom(self):
        for off in (False, True):
            for modifiers in (CTRL, ["Meta"], CTRL + SHIFT):
                with self.subTest(no_zoom_animation=off, modifiers=modifiers), self.animations(off):
                    viewer = self.make_canvas_viewer()
                    start = self.places(viewer, [1, 2])
                    canvas_event = self.scroll(viewer, modifiers=modifiers)
                    self.assertEqual(self.view_of(viewer), self.RECT)
                    self.play_out(viewer)
                    self.assertEqual(self.view_of(viewer), self.RECT)
                    self.assertFalse(viewer.zoom_timer.active)             # no glide was ever started
                    viewer._original_camera_mouse_event.assert_not_called()
                    self.assertTrue(canvas_event.handled)
                    self.assert_places(viewer, [1, 2], self.rotated(start, self.FIRST_PLACE, 5.0))

    def test_every_notch_of_a_turn_leaves_the_view_alone(self):
        for off in (False, True):
            with self.subTest(no_zoom_animation=off), self.animations(off):
                viewer = self.make_canvas_viewer()
                for delta in ((0.0, 1.0), (0.0, 2.0), (0.0, -3.0), (4.0, 0.0), (0.0, 0.5)):
                    self.scroll(viewer, delta)
                    self.play_out(viewer)
                    self.assertEqual(self.view_of(viewer), self.RECT, delta)
                viewer._original_camera_mouse_event.assert_not_called()

    def test_the_pivot_is_the_cursor_of_the_first_notch_and_stays_while_the_cursor_moves(self):
        cursors = [self.FIRST, (200.0, 100.0), (5.0, 220.0), (140.0, 140.0)]
        for off in (False, True):
            with self.subTest(no_zoom_animation=off), self.animations(off):
                viewer = self.make_canvas_viewer()
                start = self.places(viewer, [1, 2])
                for notches, cursor in enumerate(cursors, start=1):
                    self.scroll(viewer, at=cursor)
                    self.play_out(viewer)
                    self.assertEqual(self.view_of(viewer), self.RECT)
                    self.assert_places(viewer, [1, 2], self.rotated(start, self.FIRST_PLACE, 5.0 * notches))
                np.testing.assert_allclose(viewer._wheel_turn.pivot, self.FIRST_PLACE, atol=1e-6)
                # A second without a notch ends the turn; the next one takes the cursor's place as its pivot.
                viewer._wheel_turn_timer.fire()
                turned = self.places(viewer, [1, 2])
                self.scroll(viewer, at=(200.0, 100.0))
                self.assert_places(viewer, [1, 2], self.rotated(turned, (150.0, 100.0), 5.0))
                self.assertEqual(self.view_of(viewer), self.RECT)

    def test_a_wheel_that_turns_nothing_zooms_the_view(self):
        cases = {
            "Ctrl, nothing selected": dict(selected=[], modifiers=CTRL),
            "Ctrl in a UMAP or PaCMAP layout": dict(modifiers=CTRL, knn=True),
            "a plain wheel": dict(modifiers=()),
            "Shift": dict(modifiers=SHIFT),
            "Ctrl with the left button held": dict(modifiers=CTRL, buttons=(1,)),
        }
        for off in (False, True):
            for name, case in cases.items():
                with self.subTest(no_zoom_animation=off, case=name), self.animations(off):
                    viewer = self.make_canvas_viewer(selected=case.get("selected", (1, 2)))
                    nodes = viewer.pos.copy()
                    with mock.patch.object(cfg, "UMAP_MODE", case.get("knn", False)):
                        self.scroll(viewer, modifiers=case["modifiers"], buttons=case.get("buttons", ()))
                    if off:
                        self.assertNotEqual(self.view_of(viewer), self.RECT)      # at once
                        self.assertFalse(viewer.zoom_timer.active)
                    else:
                        self.assertEqual(self.view_of(viewer), self.RECT)         # over the next frames
                        self.assertTrue(viewer.zoom_timer.active)
                        self.play_out(viewer)
                    _left, _bottom, width, _height = self.view_of(viewer)
                    self.assertLess(width, self.RECT[2])                          # a notch up zooms in
                    np.testing.assert_array_equal(viewer.pos, nodes)
                    self.assertIsNone(viewer._wheel_turn)

    def test_a_turn_that_ended_leaves_the_next_wheel_to_the_camera(self):
        for off in (False, True):
            with self.subTest(no_zoom_animation=off), self.animations(off):
                viewer = self.make_canvas_viewer()
                self.scroll(viewer)
                self.scroll(viewer, modifiers=())             # no Ctrl: the turn ends and the view zooms
                self.play_out(viewer)
                self.assertIsNone(viewer._wheel_turn)
                self.assertLess(self.view_of(viewer)[2], self.RECT[2])

    def test_the_wrapper_still_passes_a_pinch_on_and_ignores_the_right_buttons_wheel(self):
        viewer = self.make_canvas_viewer()
        viewer._original_camera_mouse_event = mock.Mock()
        for event_type in ("gesture_zoom", "gesture_rotate"):
            pinch = SimpleNamespace(type=event_type, handled=False, modifiers=["Control"], buttons=[],
                                    button=None)
            viewer._camera_mouse_event(pinch)                 # Ctrl and a selection make no difference
            viewer._original_camera_mouse_event.assert_called_with(pinch)
        viewer._original_camera_mouse_event.reset_mock()
        right = SimpleNamespace(type="mouse_wheel", handled=False, modifiers=["Control"], buttons=[2],
                                button=None, delta=(0.0, 1.0), pos=self.FIRST)
        viewer._camera_mouse_event(right)
        viewer._original_camera_mouse_event.assert_not_called()
        self.assertFalse(viewer.zoom_timer.active)
        self.assertEqual(self.view_of(viewer), self.RECT)


# =====================================================================
# A running print capture takes a turn as it takes a move
# =====================================================================

class CaptureTests(GestureTestCase):
    """What a print ... full capture watches (commands/print.py: _CaptureEvents) is the network's drawing:
    the node markers and the edge layer. Selecting nodes, moving them and turning them all redraw it,
    and each cancels the capture; the same updates serve all three."""

    def make_capture(self, viewer):
        print_command = importlib.import_module("commands.print")
        root = scene.Node()
        viewer.markers._markers.parent = root            # the real markers, as the capture finds them in the scene
        viewer.canvas.scene = SimpleNamespace(node_transform=viewer.canvas.scene.node_transform,
                                              events=root.events, children=root.children)
        viewer.view = SimpleNamespace(scene=object(), camera=SimpleNamespace(rect=(0, 0, 100, 100), aspect=1.0))
        events = print_command._CaptureEvents(viewer, [])
        self.addCleanup(events.close)
        return print_command, events

    def gestures(self):
        """Each gesture as (what comes before the capture's pause, what happens in it)."""
        def press(modifiers=()):
            return lambda v: self.press(v, *NODE[1], modifiers=modifiers)
        return {
            "a move": (press(), lambda v: self.move(v, 70, 20)),
            "a move held to a line": (press(), lambda v: self.move(v, 70, 20, modifiers=SHIFT)),
            "the wheel during a move": (press(), lambda v: self.wheel(v, (0.0, 2.0), *NODE[1])),
            "a Ctrl + right-drag turn": (press(CTRL), lambda v: self.move(v, 70, 40, modifiers=CTRL)),
            "Ctrl + wheel": (lambda v: None,
                             lambda v: self.wheel(v, (0.0, 1.0), 60, 30, buttons=(), modifiers=CTRL)),
        }

    def test_each_of_them_spoils_the_capture_in_both_drawing_modes(self):
        for drawing in (False, True):
            for name, (before, during) in self.gestures().items():
                with self.subTest(drawing=drawing, gesture=name), \
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                    viewer = self.make_viewer(selected=[1, 2])
                    print_command, events = self.make_capture(viewer)
                    before(viewer)                       # the press, which redraws for the selection, is no part of it
                    events.release()
                    during(viewer)
                    with self.assertRaises(print_command._CaptureInterrupted):
                        events.reclaim()

    def test_a_turn_that_changes_nothing_spoils_nothing(self):
        viewer = self.make_viewer()                       # nothing is selected: Ctrl + wheel is the camera's
        _print_command, events = self.make_capture(viewer)
        events.release()
        self.wheel(viewer, (0.0, 1.0), 60, 30, buttons=(), modifiers=CTRL)
        events.reclaim()


if __name__ == "__main__":
    unittest.main()
