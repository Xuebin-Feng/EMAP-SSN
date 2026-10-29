# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's mouse and keyboard gestures on nodes (EMAPSSN_Viewer.py: on_key_press,
on_key_release, on_mouse_press, on_mouse_double_click, on_mouse_move, on_mouse_wheel,
on_mouse_release).

* Esc cancels a drag in progress, a box, a Ctrl + right-drag turn and a Ctrl + wheel turn. A drag
  of selected nodes puts every node back where it was at the press, a rotation too, keeps the
  selection the press made and leaves no step on the undo history; a selection box is hidden and
  the selection from before the press comes back. The rest of that right-button drag does
  nothing, nothing the release does, and the camera does not take it for a zoom. The command
  line stays open for the Esc that cancels a drag.
* Shift on the right button toggles: it adds an unselected node at the press, removes a selected
  one when the button is let go as a click, and a box toggles the nodes in it. Ctrl no longer
  removes anything.
* A right-click on a node that changes the selection (Shift adding or removing one, a press
  selecting a node alone) writes the status line a box selection writes: "Selected N nodes." or
  "Selection cleared."; one that changes nothing writes nothing.
* A press on a node saves an undo step; when the button is let go, or a Ctrl + wheel turn ends,
  the step stays only if the press changed the nodes' places or the selection. Otherwise it is
  taken off the history, with the redo steps it cleared given back, as Esc does.
* Shift while dragging nodes holds the move to one of eight directions, and the wheel's
  rotation never moves it.
* The wheel during a drag turns the nodes 5 degrees a notch about the press, Shift or not.
* Ctrl + right-drag from a selected node turns the selection about the exact point where the
  button went down, following the pointer, full turns included. The angle is measured from where
  the pointer first leaves 15 px of that point, so nothing jumps; Shift rounds it to 15 degrees.
  The status line shows the angle wrapped into (-180, 180], so after circles it reads where the
  nodes stand; the turn itself is whole.
* Ctrl + wheel turns the selection 5 degrees a notch about the cursor, a run of notches being
  one turn and one undo step, and the camera does not zoom meanwhile: its wrapper gets the
  scene's own copy of the event, so CameraWheelTests send notches through a real canvas.
* A Shift left-click copies the header once the double-click interval has passed, a Shift
  left double click the sequence alone, on release within 5 px of the press. Ctrl copies nothing.

The tests send synthetic VisPy-style events to a Viewer built as test_viewer_markers builds one:
real markers and pick, a recording edge layer. The Viewer's timers are tests.timer_fixtures's.
"""
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

from PySide6 import QtWidgets  # noqa: E402
from vispy import app as vispy_app, scene  # noqa: E402
from vispy.util import keys  # noqa: E402

import Command_Engine  # noqa: E402
import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from Command_Queue import CommandQueue  # noqa: E402
from desktop.Viewer_Zoom import SmoothWheelZoom  # noqa: E402
from EMAPSSN_Viewer import (  # noqa: E402
    CLICK_DISTANCE_PIXELS, MOVE_SNAP_DEGREES, ROTATE_DEAD_ZONE_PIXELS, ROTATE_DEGREES_PER_NOTCH,
    ROTATE_SNAP_DEGREES, WHEEL_TURN_IDLE_SECONDS, MainViewer,
)
from tests import test_viewer_markers as marker_tests  # noqa: E402
from tests.timer_fixtures import FakeTimer  # noqa: E402

# The nodes lie 40 px apart on y = 0, as test_viewer_markers places them, and
# a click on a node's own place picks it.
NODE = [(40.0 * index, 0.0) for index in range(6)]
SHIFT = ["Shift"]
CTRL = ["Control"]


def event(x=0.0, y=0.0, button=None, buttons=(), modifiers=(), delta=(0.0, 0.0)):
    return SimpleNamespace(pos=np.array([float(x), float(y)]), button=button, buttons=list(buttons),
                           modifiers=list(modifiers), handled=False, delta=delta)


def escape():
    return SimpleNamespace(key="Escape", text="", modifiers=[], handled=False)


class RecordingEdges:
    """An edge layer that keeps the positions it was last given."""

    def __init__(self):
        self.positions = None
        self.visible = True
        self.calls = 0

    def show_threshold(self, positions, visible_mask, threshold):
        self.positions = np.array(positions, dtype=float)
        self.calls += 1
        return 3

    def show_edges(self, positions, edges):
        return self.show_threshold(positions, None, 0.0)

    def set_opacity_by_score(self, enabled):
        return False

    def note_motion(self):
        pass


class GestureTestCase(unittest.TestCase):
    """A Viewer to drag nodes in, and the events that do it."""

    DOUBLE_CLICK_MS = 40

    def setUp(self):
        FakeTimer.created = []
        for patcher in (mock.patch.object(cfg, "UMAP_MODE", False, create=True),
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", False),
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", False),
                        marker_tests.distinct_outlines(),     # a selected node's outline differs
                        mock.patch.object(vispy_app, "Timer", FakeTimer),
                        mock.patch.object(QtWidgets.QApplication, "doubleClickInterval",
                                          return_value=self.DOUBLE_CLICK_MS),
                        mock.patch("builtins.print")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_viewer(self, selected=()):
        viewer = marker_tests.DraggedNodePickTests("test_a_dragged_node_can_be_dragged_again").make_viewer()
        del viewer._save_state          # the real one: it snapshots into position_history
        viewer.position_history = []
        viewer.redo_stack = []
        viewer.edge_layer = RecordingEdges()
        viewer.current_slider_threshold = 0.0
        viewer.update_console_background = mock.Mock()
        viewer.selected_indices = list(selected)
        viewer.selection_epoch = 0
        viewer._drag_in_click, viewer._drag_turn, viewer._wheel_turn = False, None, None   # as __init__ has them
        viewer._remove_on_click = viewer._header_copy_node = None
        viewer.update_selection_visual()
        viewer.broadcast_event.reset_mock()
        viewer.edge_layer.calls = 0
        return viewer

    @staticmethod
    def press(viewer, x, y, button=2, modifiers=()):
        ev = event(x, y, button=button, buttons=[button], modifiers=modifiers)
        viewer.on_mouse_press(ev)
        return ev

    @staticmethod
    def move(viewer, x, y, buttons=(2,), modifiers=()):
        ev = event(x, y, buttons=buttons, modifiers=modifiers)
        viewer.on_mouse_move(ev)
        return ev

    @staticmethod
    def wheel(viewer, delta, x=0.0, y=0.0, buttons=(2,), modifiers=()):
        ev = event(x, y, buttons=buttons, modifiers=modifiers, delta=delta)
        viewer.on_mouse_wheel(ev)
        return ev

    @staticmethod
    def release(viewer, x, y, button=2, modifiers=()):
        ev = event(x, y, button=button, modifiers=modifiers)
        viewer.on_mouse_release(ev)
        return ev

    @staticmethod
    def double_click(viewer, x, y, button=1, modifiers=()):
        ev = event(x, y, button=button, buttons=[button], modifiers=modifiers)
        viewer.on_mouse_double_click(ev)
        return ev

    @staticmethod
    def esc(viewer):
        ev = escape()
        viewer.on_key_press(ev)
        return ev

    @staticmethod
    def key_release(viewer, key):
        ev = SimpleNamespace(key=key, text="", modifiers=[], handled=False)
        viewer.on_key_release(ev)
        return ev

    @staticmethod
    def rotated(points, about, degrees):
        angle = np.radians(degrees)
        rel = np.asarray(points, dtype=float) - about
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.column_stack([rel[:, 0] * cos_a - rel[:, 1] * sin_a,
                                rel[:, 0] * sin_a + rel[:, 1] * cos_a]) + about

    @staticmethod
    def places(viewer, nodes):
        return viewer.pos[list(nodes), :2].astype(float)

    @staticmethod
    def status(viewer):
        return [str(call.args[0]) for call in viewer._show_click_status.call_args_list]

    def assert_picture_matches(self, viewer):
        """The markers, the edge layer and the pick show the nodes where they are."""
        order = viewer._submitted_marker_node_order
        np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
        np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
        self.assertIsNotNone(viewer._marker_picker)
        for node in range(len(viewer.pos)):
            self.assertEqual(viewer.pick_hovered_node(viewer.pos[node, :2]), node)

    def assert_places(self, viewer, nodes, expected, atol=1e-3):
        np.testing.assert_allclose(self.places(viewer, nodes), np.asarray(expected, dtype=float), atol=atol)


# =====================================================================
# Esc cancels a drag of nodes
# =====================================================================

class CancelNodeDragTests(GestureTestCase):
    def test_esc_puts_every_dragged_node_back_after_moves_and_rotations(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 2])
                start = viewer.pos.copy()
                self.press(viewer, *NODE[1])
                self.move(viewer, 70, 20)
                self.wheel(viewer, (0.0, 6.0), 70, 20)                             # 30 degrees
                self.wheel(viewer, (0.0, 3.0), 70, 20, modifiers=["Shift"])        # 15 more, Shift or not
                self.move(viewer, 90, 30)
                self.assertFalse(np.array_equal(viewer.pos, start))
                handled = self.esc(viewer)
                self.assertTrue(handled.handled)
                np.testing.assert_array_equal(viewer.pos, start)       # exactly, not rotated back
                self.assertFalse(viewer.is_multi_dragging)
                self.assertEqual(viewer.selected_indices, [1, 2])
                self.assert_picture_matches(viewer)

    def test_the_wheel_turns_a_copy_so_esc_does_not_put_back_the_turned_nodes(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.wheel(viewer, (0.0, 18.0), *NODE[1])                  # a quarter turn about the press
        self.assertFalse(np.allclose(viewer.drag_start_nodes_pos, [NODE[1], NODE[2]]))
        self.esc(viewer)
        np.testing.assert_array_equal(viewer.pos[[1, 2], :2], np.array([NODE[1], NODE[2]], dtype=np.float32))

    def test_the_selection_stays_as_the_press_left_it(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])                  # an unselected node: it becomes the selection
        self.assertEqual(viewer.selected_indices, [3])
        self.move(viewer, 150, 50)
        self.esc(viewer)
        self.assertEqual(viewer.selected_indices, [3])
        np.testing.assert_array_equal(viewer.pos[3], NODE[3])

    def test_a_cancelled_drag_leaves_no_undo_step_with_or_without_a_selection_change(self):
        for pressed, selected, changes in ((1, [1, 2], False), (3, [1, 2], True)):
            with self.subTest(selection_changes=changes):
                viewer = self.make_viewer(selected=selected)
                viewer._save_state()                                  # an earlier step
                earlier = list(viewer.position_history)
                undone = {"pos": "a step that was undone"}
                viewer.redo_stack = [undone]
                self.press(viewer, *NODE[pressed])
                self.assertEqual(len(viewer.position_history), 2)    # the press snapshots
                self.assertEqual(viewer.redo_stack, [])
                self.move(viewer, 150, 50)
                self.esc(viewer)
                self.assertEqual(len(viewer.position_history), 1)
                self.assertIs(viewer.position_history[0], earlier[0])
                self.assertEqual(viewer.redo_stack, [undone])         # Ctrl+Y still has its step

    def test_undo_does_not_restore_a_selection_so_the_press_snapshot_is_never_worth_keeping(self):
        # Why Esc never keeps the press's snapshot, even when the press changed the selection: it
        # holds the nodes' places and the colours, sizes and so on, never the selection, so undoing
        # it would change nothing.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])
        self.release(viewer, *NODE[3])
        self.assertEqual(viewer.selected_indices, [3])
        self.assertEqual(len(viewer.position_history), 1)
        self.assertNotIn("selected_indices", viewer.position_history[0])
        viewer._do_undo(announce=False)
        self.assertEqual(viewer.selected_indices, [3])
        np.testing.assert_array_equal(viewer.pos[:, 0], [x for x, _ in NODE])

    def test_the_oldest_step_the_press_pushed_off_the_cap_is_given_back(self):
        viewer = self.make_viewer(selected=[1, 2])
        for _ in range(50):
            viewer._save_state()
        before = list(viewer.position_history)
        self.press(viewer, *NODE[1])
        self.assertIsNot(viewer.position_history[0], before[0])    # the oldest fell off the cap
        self.esc(viewer)
        self.assertEqual(len(viewer.position_history), 50)
        self.assertTrue(all(a is b for a, b in zip(viewer.position_history, before)))

    def test_a_step_added_during_the_drag_is_not_taken_back(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        viewer._save_state()                                       # something else, meanwhile
        self.esc(viewer)
        self.assertEqual(len(viewer.position_history), 2)

    def test_a_release_leaves_the_press_snapshot_as_a_step(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.move(viewer, 70, 20)
        self.release(viewer, 70, 20)
        self.assertEqual(len(viewer.position_history), 1)
        viewer._do_undo(announce=False)
        np.testing.assert_array_equal(viewer.pos[1], NODE[1])

    def test_the_picture_and_the_web_pages_follow_the_restored_state(self):
        # Under Update Colors After Mouse Release the markers moved and nothing else did; the pick
        # and the colours are made again. The web pages hear of a selection the press held back
        # (Update Web Pages After Mouse Release) once, as a release tells them.
        for drawing in (False, True):
            for webserver in (False, True):
                with self.subTest(drawing=drawing, webserver=webserver), \
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing), \
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", webserver):
                    viewer = self.make_viewer(selected=[1, 2])
                    self.press(viewer, *NODE[3])                   # selects node 3
                    sent_at_press = viewer.broadcast_event.call_count
                    self.assertEqual(sent_at_press, 0 if webserver else 1)
                    outlines = []
                    real_outlines = viewer.node_outlines
                    viewer.node_outlines = lambda *a, **k: outlines.append(1) or real_outlines(*a, **k)
                    self.move(viewer, 150, 50)
                    self.move(viewer, 170, 70)
                    if drawing:
                        self.assertEqual(outlines, [])             # only the markers moved
                    self.esc(viewer)
                    self.assertEqual(len(outlines), 1 if drawing else 3)    # the colours, again
                    self.assert_picture_matches(viewer)
                    self.assertEqual(viewer.broadcast_event.call_count, 1)
                    self.assertEqual(viewer.broadcast_event.call_args.args[0],
                                     {"type": "selection_changed", "indices": [3]})

    def test_a_cancel_resumes_the_command_queue_as_a_release_does(self):
        for cancel in (True, False):
            viewer = self.make_viewer(selected=[1, 2])
            viewer.command_queue = mock.Mock()
            viewer.command_queue.pending.return_value = [object()]
            self.press(viewer, *NODE[1])
            viewer.command_queue.resume.assert_not_called()
            if cancel:
                self.esc(viewer)
            else:
                self.release(viewer, *NODE[1])
            viewer.command_queue.resume.assert_called_once_with()

    def test_the_rest_of_the_drag_does_nothing_until_the_button_is_let_go(self):
        viewer = self.make_viewer(selected=[1, 2])
        start = viewer.pos.copy()
        self.press(viewer, *NODE[1])
        self.move(viewer, 70, 20)
        self.esc(viewer)
        calls = viewer.edge_layer.calls
        for x, y in ((100, 40), (120, 60)):
            ev = self.move(viewer, x, y)
            self.assertTrue(ev.handled)                  # the camera, which zooms on a right-drag, is told
        wheel = self.wheel(viewer, (0.0, 5.0), 120, 60)
        self.assertFalse(wheel.handled)                  # not a rotation: nothing is being dragged
        np.testing.assert_array_equal(viewer.pos, start)
        self.assertEqual(viewer.edge_layer.calls, calls)
        self.assertFalse(viewer.is_multi_dragging)
        ev = self.release(viewer, 120, 60)
        self.assertTrue(ev.handled)
        np.testing.assert_array_equal(viewer.pos, start)
        self.assertEqual(viewer.selected_indices, [1, 2])
        # The release ended it: moves are the hover's again.
        self.assertFalse(viewer._drag_cancelled)
        self.move(viewer, *NODE[4], buttons=())
        self.assertEqual(viewer.hovered_node_idx, 4)

    def test_a_missed_release_ends_the_swallowing_at_the_next_move_without_the_button(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.esc(viewer)
        self.assertTrue(self.move(viewer, 100, 40).handled)
        with mock.patch.object(viewer, "on_mouse_release") as on_release:
            self.move(viewer, *NODE[4], buttons=())     # the button is up: no release was seen
        on_release.assert_not_called()
        self.assertEqual(viewer.hovered_node_idx, 4)    # the hover works again
        # And a new press starts a drag as usual.
        self.press(viewer, *NODE[1])
        self.assertTrue(viewer.is_multi_dragging)
        self.move(viewer, 70, 20)
        self.assertEqual(viewer.pos[1, 1], 20)

    def test_a_new_right_press_ends_the_swallowing_even_when_the_release_was_never_seen(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.esc(viewer)
        self.press(viewer, *NODE[1])                  # no release, and no move without the button, between
        self.assertTrue(viewer.is_multi_dragging)
        self.move(viewer, 70, 20)
        np.testing.assert_allclose(viewer.pos[1], [70, 20])
        self.release(viewer, 70, 20)
        self.assertFalse(viewer.is_multi_dragging)
        np.testing.assert_allclose(viewer.pos[1], [70, 20])      # the release ended this drag, as always

    def test_the_camera_does_not_zoom_on_the_cancelled_drag(self):
        viewer = self.make_viewer(selected=[1, 2])
        viewer._original_camera_mouse_event = mock.Mock()
        viewer.view = SimpleNamespace(scene=object(), camera=SimpleNamespace(interactive=True))
        self.press(viewer, *NODE[1])
        self.esc(viewer)
        scene_event = SimpleNamespace(type="mouse_move", button=None, buttons=[2], handled=False)
        viewer.on_mouse_move(scene_event)
        self.assertTrue(scene_event.handled)
        viewer._camera_mouse_event(scene_event)           # as VisPy sends it on, a right-drag
        viewer._original_camera_mouse_event.assert_not_called()

    def test_esc_with_no_drag_does_what_it_did(self):
        viewer = self.make_viewer()
        viewer.console_mode = True
        viewer.console_bg = SimpleNamespace(visible=True)
        viewer._hide_command_caret = mock.Mock()
        with mock.patch.object(Command_Engine, "show_status") as show_status:
            ev = self.esc(viewer)
        self.assertTrue(ev.handled)                       # never closes the window
        self.assertFalse(viewer.console_mode)
        self.assertFalse(viewer.console_bg.visible)
        show_status.assert_called_once_with(viewer, "")
        ev = self.esc(viewer)                             # nothing to close either
        self.assertTrue(ev.handled)

    def test_a_drag_in_progress_comes_before_the_command_line(self):
        for box in (False, True):
            with self.subTest(box=box):
                viewer = self.make_viewer(selected=[1, 2])
                viewer.console_mode = True
                viewer.console_bg = SimpleNamespace(visible=True)
                viewer._hide_command_caret = mock.Mock()
                self.press(viewer, 30, -10) if box else self.press(viewer, *NODE[1])
                self.assertEqual(viewer.is_box_selecting, box)
                with mock.patch.object(Command_Engine, "show_status"):
                    self.assertTrue(self.esc(viewer).handled)
                    self.assertTrue(viewer.console_mode)           # still open
                    self.assertTrue(viewer.console_bg.visible)
                    self.assertFalse(viewer.is_box_selecting or viewer.is_multi_dragging)
                    self.release(viewer, 100, 0)
                    self.esc(viewer)                               # the next Esc closes it
                self.assertFalse(viewer.console_mode)


# =====================================================================
# Esc cancels a selection box
# =====================================================================

class CancelBoxTests(GestureTestCase):
    def box(self, viewer, modifiers=()):
        """Press on empty space and draw a box over nodes 1 and 2."""
        self.press(viewer, 30, -10, modifiers=modifiers)
        self.assertTrue(viewer.is_box_selecting)
        self.move(viewer, 90, 10, modifiers=modifiers)

    def test_esc_hides_the_box_and_puts_back_the_selection_in_its_order(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[4, 0, 5])
                self.box(viewer)
                self.assertTrue(viewer.selection_box.visible)
                self.assertEqual(sorted(viewer.selected_indices), [0, 4, 5] if drawing else [1, 2])
                handled = self.esc(viewer)
                self.assertTrue(handled.handled)
                self.assertFalse(viewer.selection_box.visible)
                self.assertFalse(viewer.is_box_selecting)
                self.assertEqual(viewer.selected_indices, [4, 0, 5])
                self.assertEqual(viewer.position_history, [])

    def test_the_release_that_follows_applies_no_box_and_clears_nothing(self):
        for drawing in (False, True):
            for release_at in ((90, 10), (30, -10)):         # where it ended, and where it began
                with self.subTest(drawing=drawing, release_at=release_at), \
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                    viewer = self.make_viewer(selected=[4, 0, 5])
                    self.box(viewer)
                    self.esc(viewer)
                    viewer.broadcast_event.reset_mock()
                    viewer._show_click_status.reset_mock()
                    ev = self.release(viewer, *release_at)
                    self.assertTrue(ev.handled)
                    self.assertEqual(viewer.selected_indices, [4, 0, 5])
                    self.assertFalse(viewer.selection_box.visible)
                    viewer._show_click_status.assert_not_called()      # no "Selection cleared."
                    viewer.broadcast_event.assert_not_called()

    def test_the_rest_of_the_box_drag_does_nothing(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[4])
                self.box(viewer)
                self.esc(viewer)
                viewer.selection_box.set_data.reset_mock()
                ev = self.move(viewer, 150, 40)
                self.assertTrue(ev.handled)
                self.assertEqual(viewer.selected_indices, [4])
                viewer.selection_box.set_data.assert_not_called()
                self.release(viewer, 150, 40)
                self.assertEqual(viewer.selected_indices, [4])

    def test_the_selection_clock_is_as_before_the_press(self):
        # A command that waits in the queue applies its own selection unless one was made by hand
        # since it was given (selection_epoch): a cancelled box made none.
        viewer = self.make_viewer(selected=[4])
        viewer.selection_epoch = 7
        self.box(viewer)
        self.assertGreater(viewer.selection_epoch, 7)       # the live box counted as made by hand
        self.esc(viewer)
        self.assertEqual(viewer.selection_epoch, 7)

    def test_the_web_pages_are_told_the_restored_selection_when_they_heard_the_box(self):
        # Each move of a live box tells them; Update Web Pages After Mouse Release leaves them to
        # hear it once, on release, so they still have the selection from before the press.
        for webserver in (False, True):
            with self.subTest(webserver=webserver), mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", webserver):
                viewer = self.make_viewer(selected=[4, 0])
                self.box(viewer)
                self.esc(viewer)
                sent = [call.args[0] for call in viewer.broadcast_event.call_args_list]
                if webserver:
                    self.assertEqual(sent, [])
                else:
                    self.assertEqual(sent[-1], {"type": "selection_changed", "indices": [4, 0]})

    def test_the_markers_show_the_restored_selection(self):
        viewer = self.make_viewer(selected=[4, 0])
        colours = viewer.markers._data["a_fg_color"].copy()
        self.box(viewer)
        self.assertFalse(np.array_equal(viewer.markers._data["a_fg_color"], colours))
        self.esc(viewer)
        np.testing.assert_array_equal(viewer.markers._data["a_fg_color"], colours)

    def test_a_cancel_resumes_the_command_queue(self):
        viewer = self.make_viewer(selected=[4])
        viewer.command_queue = mock.Mock()
        viewer.command_queue.pending.return_value = [object()]
        self.box(viewer)
        self.esc(viewer)
        viewer.command_queue.resume.assert_called_once_with()

    def test_a_box_not_cancelled_ends_as_it_did(self):
        viewer = self.make_viewer(selected=[4, 0])
        self.box(viewer)
        self.release(viewer, 90, 10)
        self.assertEqual(sorted(viewer.selected_indices), [1, 2])
        self.press(viewer, 30, -10)
        self.release(viewer, 30, -10)                       # a short right-click clears
        self.assertEqual(viewer.selected_indices, [])


# =====================================================================
# Shift on the right button toggles; Ctrl removes nothing
# =====================================================================

class ShiftSelectionTests(GestureTestCase):
    def test_a_shift_press_on_an_unselected_node_adds_it_at_the_press(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(viewer.selected_indices, [1, 2, 3])
        self.assertEqual(viewer.selection_epoch, 1)
        self.assertTrue(viewer.is_multi_dragging)
        self.move(viewer, NODE[3][0] + 10, 30)                  # a drag moves the whole selection
        self.assert_places(viewer, [1, 2, 3], [(x + 10, 30) for x, _ in NODE[1:4]])
        self.release(viewer, NODE[3][0] + 10, 30)
        self.assertEqual(viewer.selected_indices, [1, 2, 3])

    def test_a_shift_click_on_a_selected_node_removes_it_on_release_not_at_the_press(self):
        viewer = self.make_viewer(selected=[1, 2])
        start = viewer.pos.copy()
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(viewer.selected_indices, [1, 2])       # the press removes nothing
        self.assertEqual(viewer.selection_epoch, 0)
        viewer.broadcast_event.reset_mock()
        self.move(viewer, NODE[2][0] + 3, 2, modifiers=SHIFT)   # a slip, inside the click's radius
        np.testing.assert_array_equal(viewer.pos, start)        # moves nothing
        ev = self.release(viewer, NODE[2][0] + 3, 2, modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.assertEqual(viewer.selected_indices, [1])
        self.assertEqual(viewer.selection_epoch, 1)             # a selection by hand
        np.testing.assert_array_equal(viewer.pos, start)
        self.assertFalse(viewer.is_multi_dragging)
        self.assertEqual(viewer.broadcast_event.call_args.args[0],
                         {"type": "selection_changed", "indices": [1]})
        self.assert_picture_matches(viewer)
        # And the drag it was is over: the next move is the hover's.
        self.move(viewer, *NODE[4], buttons=())
        self.assertEqual(viewer.hovered_node_idx, 4)

    def test_five_pixels_make_it_a_move_and_the_nodes_wait_for_them(self):
        self.assertEqual(CLICK_DISTANCE_PIXELS, 5.0)
        viewer = self.make_viewer(selected=[1, 2])
        start = viewer.pos.copy()
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.move(viewer, NODE[2][0] + 4.9, 0, modifiers=SHIFT)
        np.testing.assert_array_equal(viewer.pos, start)        # still a click: nothing moves
        self.move(viewer, NODE[2][0] + 5.0, 0, modifiers=SHIFT)
        self.assert_places(viewer, [1, 2], [(45, 0), (85, 0)])  # a move, from the press
        self.move(viewer, NODE[2][0] + 30, 5, modifiers=SHIFT)  # along a line, as a Shift move is
        self.assert_places(viewer, [1, 2], [(70, 0), (110, 0)])
        self.release(viewer, NODE[2][0] + 30, 5)
        self.assertEqual(viewer.selected_indices, [1, 2])       # nothing is removed
        self.assertEqual(viewer.selection_epoch, 0)

    def test_a_move_that_comes_back_to_the_press_is_still_a_move(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.move(viewer, NODE[2][0] + 30, 0, modifiers=SHIFT)
        self.move(viewer, NODE[2][0] + 1, 0, modifiers=SHIFT)
        self.release(viewer, NODE[2][0] + 1, 0)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assert_places(viewer, [1, 2], [(41, 0), (81, 0)])

    def test_a_turn_of_the_wheel_is_no_click_either(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.wheel(viewer, (0.0, 2.0), *NODE[2])
        self.release(viewer, *NODE[2])
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assert_places(viewer, [1, 2], self.rotated([NODE[1], NODE[2]], NODE[2], 10.0))

    def test_a_shift_click_removes_where_nothing_can_be_dragged(self):
        # The kNN layout has no moving of nodes, but selecting works.
        viewer = self.make_viewer(selected=[1, 2])
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            self.press(viewer, *NODE[2], modifiers=SHIFT)
            self.assertFalse(viewer.is_multi_dragging)
            self.assertEqual(viewer.selected_indices, [1, 2])
            ev = self.release(viewer, *NODE[2], modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.assertEqual(viewer.selected_indices, [1])
        self.assertEqual(viewer.broadcast_event.call_args.args[0], {"type": "selection_changed", "indices": [1]})
        # A release a long way off is no click there either.
        viewer = self.make_viewer(selected=[1, 2])
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            self.press(viewer, *NODE[2], modifiers=SHIFT)
            self.release(viewer, 150, 50)
        self.assertEqual(viewer.selected_indices, [1, 2])

    def test_the_web_pages_hear_of_the_removal_whether_the_press_held_its_news_back_or_not(self):
        for webserver in (False, True):
            with self.subTest(webserver=webserver), mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", webserver):
                viewer = self.make_viewer(selected=[1, 2])
                sent = []
                viewer.broadcast_event = lambda event, sent=sent: sent.append(list(event["indices"]))
                self.press(viewer, *NODE[2], modifiers=SHIFT)
                self.release(viewer, *NODE[2])
                self.assertEqual(sent, [[1]] if webserver else [[1, 2], [1]])

    def test_esc_in_a_shift_click_removes_nothing(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.assertTrue(self.esc(viewer).handled)
        self.release(viewer, *NODE[2])
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertEqual(viewer.selection_epoch, 0)

    def test_a_missed_release_ends_it_at_the_next_move_and_that_is_no_click_when_the_pointer_is_away(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)            # no release comes
        self.move(viewer, 150, 50, buttons=())                   # the next move, the button up
        self.assertFalse(viewer.is_multi_dragging)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.press(viewer, *NODE[1])
        self.release(viewer, *NODE[1])                           # a plain click on a selected node
        self.assertEqual(viewer.selected_indices, [1, 2])        # the old press forgot, nothing removed

    def test_two_quick_shift_right_clicks_on_a_node_both_count(self):
        # Qt sends the second press of two quick clicks as a double click: it is the press it replaces.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=SHIFT)
        self.release(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(viewer.selected_indices, [1, 2, 3])                    # added
        self.double_click(viewer, *NODE[3], button=2, modifiers=SHIFT)
        self.release(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(viewer.selected_indices, [1, 2])                       # and taken out again

    def test_a_removal_is_no_undo_step_of_its_own(self):
        # Undo holds the nodes' places, colours and so on, never the selection, so it does not bring
        # the node back; the press's snapshot is the one every node click takes.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.release(viewer, *NODE[2])
        self.assertEqual(len(viewer.position_history), 1)
        self.assertNotIn("selected_indices", viewer.position_history[0])
        viewer._do_undo(announce=False)
        self.assertEqual(viewer.selected_indices, [1])

    def test_ctrl_no_longer_removes_by_click(self):
        for modifier in ("Control", "Meta"):
            with self.subTest(modifier=modifier):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[2], modifiers=[modifier])
                self.assertEqual(viewer.selected_indices, [1, 2])
                self.release(viewer, *NODE[2], modifiers=[modifier])
                self.assertEqual(viewer.selected_indices, [1, 2])
                self.assertEqual(viewer.selection_epoch, 0)

    def test_a_ctrl_click_does_nothing_beyond_a_plain_right_click(self):
        results = {}
        for name, modifiers in (("plain", []), ("ctrl", CTRL), ("meta", ["Meta"])):
            viewer = self.make_viewer(selected=[1, 2])
            start = viewer.pos.copy()
            self.press(viewer, *NODE[2], modifiers=modifiers)
            self.release(viewer, *NODE[2], modifiers=modifiers)
            np.testing.assert_array_equal(viewer.pos, start)
            results[name] = (viewer.selected_indices, viewer.selection_epoch, len(viewer.position_history),
                             [call.args[0] for call in viewer.broadcast_event.call_args_list],
                             self.status(viewer))
        self.assertEqual(results["ctrl"], results["plain"])
        self.assertEqual(results["meta"], results["plain"])
        # On a node that is not selected, a Ctrl press selects it alone, as any press does.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=CTRL)
        self.assertEqual(viewer.selected_indices, [3])
        self.assertIsNone(viewer._drag_turn)


class NodeClickStatusTests(GestureTestCase):
    """A right-click that changes the selection says so on the status line, in the words a box selection uses."""

    def test_a_box_and_a_click_say_the_same(self):
        viewer = self.make_viewer(selected=[1])
        self.press(viewer, 30, -10)
        self.move(viewer, 90, 10)                                       # a box over nodes 1 and 2
        self.release(viewer, 90, 10)
        self.assertEqual(self.status(viewer), ["Selected 2 nodes."])
        viewer = self.make_viewer(selected=[1])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), ["Selected 2 nodes."])

    def test_a_shift_click_that_adds_a_node_says_how_many_are_selected(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), ["Selected 3 nodes."])
        self.release(viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), ["Selected 3 nodes."])      # the release adds nothing

    def test_a_shift_click_that_removes_a_node_says_so_when_the_button_is_let_go(self):
        viewer = self.make_viewer(selected=[1, 2, 3])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), [])                           # the press removes nothing
        self.release(viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), ["Selected 2 nodes."])

    def test_removing_the_last_node_clears_the_selection(self):
        for knn in (False, True):
            with self.subTest(knn=knn), mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=knn):
                viewer = self.make_viewer(selected=[2])
                self.press(viewer, *NODE[2], modifiers=SHIFT)
                self.release(viewer, *NODE[2], modifiers=SHIFT)
                self.assertEqual(viewer.selected_indices, [])
                self.assertEqual(self.status(viewer), ["Selection cleared."])

    def test_a_shift_click_that_removes_a_node_where_nothing_can_be_dragged_says_so_too(self):
        viewer = self.make_viewer(selected=[1, 2, 3])
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            self.press(viewer, *NODE[2], modifiers=SHIFT)
            self.release(viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), ["Selected 2 nodes."])

    def test_a_click_that_selects_a_node_alone_says_so(self):
        for name, modifiers in (("plain", []), ("ctrl", CTRL), ("meta", ["Meta"])):
            with self.subTest(modifiers=name):
                viewer = self.make_viewer(selected=[1, 2, 4, 5])
                self.press(viewer, *NODE[3], modifiers=modifiers)
                self.release(viewer, *NODE[3], modifiers=modifiers)
                self.assertEqual(viewer.selected_indices, [3])
                self.assertEqual(self.status(viewer), ["Selected 1 node."])

    def test_a_click_that_changes_nothing_says_nothing(self):
        for modifiers in ([], CTRL, ["Meta"]):
            with self.subTest(modifiers=modifiers):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[2], modifiers=modifiers)
                self.release(viewer, *NODE[2], modifiers=modifiers)
                self.assertEqual(self.status(viewer), [])
        viewer = self.make_viewer(selected=[3])                           # Ctrl on the one selected node
        self.press(viewer, *NODE[3], modifiers=CTRL)
        self.release(viewer, *NODE[3], modifiers=CTRL)
        self.assertEqual(self.status(viewer), [])

    def test_a_drag_from_a_node_that_is_not_selected_names_the_selection_it_made(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])
        self.move(viewer, 150, 50)
        self.release(viewer, 150, 50)
        self.assertEqual(self.status(viewer), ["Selected 1 node."])    # once, at the press

    def test_esc_leaves_the_message_of_the_selection_the_press_made(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])
        self.esc(viewer)                                                  # the selection stays as the press made it
        self.release(viewer, *NODE[3])
        self.assertEqual(viewer.selected_indices, [3])
        self.assertEqual(self.status(viewer), ["Selected 1 node."])

    def test_a_left_click_is_left_alone(self):
        viewer = self.make_viewer(selected=[1, 2])
        for modifiers in ([], SHIFT, CTRL):
            self.press(viewer, *NODE[3], button=1, modifiers=modifiers)
            self.release(viewer, *NODE[3], button=1, modifiers=modifiers)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertEqual(self.status(viewer), [])               # "Selected: <header>" is the focus's, mocked here


class IdlePressStepTests(GestureTestCase):
    """A press on a node saves an undo step, but only one that changes something stays when the button is let go."""

    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 2])
        self.viewer._save_state()                                    # an earlier step
        self.earlier = list(self.viewer.position_history)
        self.undone = {"pos": "a step that was undone"}
        self.viewer.redo_stack = [self.undone]

    def assert_history_as_before(self, viewer=None):
        viewer = viewer or self.viewer
        self.assertEqual(len(viewer.position_history), 1)
        self.assertIs(viewer.position_history[0], self.earlier[0])
        self.assertEqual(viewer.redo_stack, [self.undone])           # Ctrl+Y still has its step

    def test_a_right_click_on_a_selected_node_leaves_no_step(self):
        for modifiers in ([], CTRL, ["Meta"]):
            with self.subTest(modifiers=modifiers):
                self.setUp()
                self.press(self.viewer, *NODE[1], modifiers=modifiers)
                self.assertEqual(len(self.viewer.position_history), 2)    # the press saves, as ever
                self.assertEqual(self.viewer.redo_stack, [])
                self.release(self.viewer, *NODE[1], modifiers=modifiers)
                self.assert_history_as_before()
                self.assertEqual(self.viewer.selected_indices, [1, 2])

    def test_a_ctrl_right_click_where_nothing_can_move_and_the_node_is_the_selection_leaves_no_step(self):
        viewer = self.make_viewer(selected=[3])
        viewer._save_state()
        self.earlier = list(viewer.position_history)
        viewer.redo_stack = [self.undone]
        with mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=True):
            self.press(viewer, *NODE[3], modifiers=CTRL)
            self.assertFalse(viewer.is_multi_dragging)
            self.assertEqual(len(viewer.position_history), 2)
            self.release(viewer, *NODE[3], modifiers=CTRL)
        self.assertEqual(viewer.selected_indices, [3])
        self.assert_history_as_before(viewer)
        self.assertIsNone(viewer._press_history)                      # nothing is left waiting

    def test_a_press_that_changes_the_selection_keeps_its_step(self):
        # Undo restores no selection, but the step stays as it did: the press chose a node.
        for knn in (False, True):
            for name, modifiers in (("plain", []), ("shift add", SHIFT), ("ctrl", CTRL)):
                with self.subTest(knn=knn, press=name), \
                        mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=knn):
                    self.setUp()
                    self.press(self.viewer, *NODE[3], modifiers=modifiers)
                    self.release(self.viewer, *NODE[3], modifiers=modifiers)
                    self.assertEqual(len(self.viewer.position_history), 2)
                    self.assertEqual(self.viewer.redo_stack, [])

    def test_a_shift_click_that_removes_a_node_keeps_its_step(self):
        self.press(self.viewer, *NODE[2], modifiers=SHIFT)
        self.release(self.viewer, *NODE[2], modifiers=SHIFT)
        self.assertEqual(self.viewer.selected_indices, [1])
        self.assertEqual(len(self.viewer.position_history), 2)

    def test_a_press_that_moved_the_nodes_keeps_its_step(self):
        self.press(self.viewer, *NODE[1])
        self.move(self.viewer, 70, 20)
        self.release(self.viewer, 70, 20)
        self.assertEqual(len(self.viewer.position_history), 2)
        self.viewer._do_undo(announce=False)
        np.testing.assert_array_equal(self.viewer.pos[1], NODE[1])

    def test_a_press_that_turned_the_nodes_keeps_its_step(self):
        self.press(self.viewer, *NODE[1])
        self.wheel(self.viewer, (0.0, 6.0), *NODE[1])                 # the wheel during a drag turns them
        self.release(self.viewer, *NODE[1])
        self.assertEqual(len(self.viewer.position_history), 2)

    def test_a_step_added_meanwhile_is_not_taken_back(self):
        self.press(self.viewer, *NODE[1])
        self.viewer._save_state()                                    # something else, meanwhile
        self.release(self.viewer, *NODE[1])
        self.assertEqual(len(self.viewer.position_history), 3)

    def test_the_buttons_follow_the_history(self):
        self.viewer._refresh_history_buttons = mock.Mock()
        self.press(self.viewer, *NODE[1])
        self.viewer._refresh_history_buttons.reset_mock()
        self.release(self.viewer, *NODE[1])
        self.viewer._refresh_history_buttons.assert_called_once_with()

    def test_the_release_after_esc_leaves_the_history_alone(self):
        self.press(self.viewer, *NODE[1])
        self.esc(self.viewer)
        self.assert_history_as_before()
        self.release(self.viewer, *NODE[1])
        self.assert_history_as_before()

    def test_a_left_release_with_no_press_waiting_leaves_the_history_alone(self):
        self.viewer._save_state()
        self.release(self.viewer, *NODE[1], button=1)
        self.assertEqual(len(self.viewer.position_history), 2)

    def test_a_press_after_a_missed_release_starts_afresh(self):
        self.press(self.viewer, *NODE[1])                            # no release comes
        self.press(self.viewer, *NODE[1])
        self.release(self.viewer, *NODE[1])
        self.assertEqual(len(self.viewer.position_history), 2)       # the first press's step, and no more
        self.assertIs(self.viewer.position_history[0], self.earlier[0])

    def test_a_press_on_empty_space_forgets_an_earlier_press(self):
        self.press(self.viewer, *NODE[1])
        self.press(self.viewer, 30, -10)                             # a box
        self.assertIsNone(self.viewer._press_selection)
        self.release(self.viewer, 30, -10)
        self.assertEqual(len(self.viewer.position_history), 2)       # nothing was taken back by the box's release


class BoxToggleTests(GestureTestCase):
    def box(self, viewer, modifiers=(), end=(90, 10), release=True):
        """Press on empty space and draw a box over nodes 1 and 2."""
        self.press(viewer, 30, -10, modifiers=modifiers)
        self.move(viewer, *end, modifiers=modifiers)
        if release:
            self.release(viewer, *end, modifiers=modifiers)

    def test_the_shift_box_toggles_the_nodes_in_it_live_and_on_release(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[2, 4])
                self.box(viewer, SHIFT, release=False)
                if not drawing:       # the live box, which Update Colors After Mouse Release leaves to the release
                    self.assertEqual(sorted(viewer.selected_indices), [1, 4])
                self.release(viewer, 90, 10, modifiers=SHIFT)
                self.assertEqual(sorted(viewer.selected_indices), [1, 4])   # 1 added, 2 taken out, 4 left

    def test_the_preview_is_the_result_at_every_step(self):
        viewer = self.make_viewer(selected=[0, 2, 5])
        self.press(viewer, -10, -10, modifiers=SHIFT)
        previews = []
        for x in (10, 50, 90, 130, 170, 210):                   # the box takes in one more node each time
            self.move(viewer, x, 10, modifiers=SHIFT)
            previews.append(sorted(viewer.selected_indices))
        self.release(viewer, 210, 10, modifiers=SHIFT)
        self.assertEqual(previews, [[2, 5], [1, 2, 5], [1, 5], [1, 3, 5], [1, 3, 4, 5], [1, 3, 4]])
        self.assertEqual(sorted(viewer.selected_indices), previews[-1])

    def test_a_shift_click_on_empty_space_changes_nothing(self):
        viewer = self.make_viewer(selected=[4, 0])
        self.press(viewer, 30, -10, modifiers=SHIFT)
        self.release(viewer, 30, -10, modifiers=SHIFT)
        self.assertEqual(sorted(viewer.selected_indices), [0, 4])

    def test_ctrl_no_longer_removes_by_box_and_the_box_is_a_plain_one(self):
        for modifier in ("Control", "Meta"):
            with self.subTest(modifier=modifier):
                viewer = self.make_viewer(selected=[1, 4])
                self.box(viewer, [modifier])
                self.assertEqual(sorted(viewer.selected_indices), [1, 2])    # replaced, as without a key

    def test_a_ctrl_click_on_empty_space_clears_as_a_plain_click_does(self):
        for modifier in ("Control", "Meta"):
            with self.subTest(modifier=modifier):
                viewer = self.make_viewer(selected=[1, 4])
                self.press(viewer, 30, -10, modifiers=[modifier])
                self.release(viewer, 30, -10, modifiers=[modifier])
                self.assertEqual(viewer.selected_indices, [])

    def test_a_toggling_box_is_no_undo_step_and_counts_as_a_selection_by_hand(self):
        viewer = self.make_viewer(selected=[2, 4])
        self.box(viewer, SHIFT)
        self.assertEqual(viewer.position_history, [])
        self.assertGreater(viewer.selection_epoch, 0)

    def test_esc_puts_back_the_selection_a_shift_box_toggled(self):
        viewer = self.make_viewer(selected=[2, 4, 0])
        self.box(viewer, SHIFT, release=False)
        self.assertNotEqual(viewer.selected_indices, [2, 4, 0])
        self.esc(viewer)
        self.assertEqual(viewer.selected_indices, [2, 4, 0])


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

class ViewTransform:
    """A node transform for a view that has panned and zoomed: the canvas place c is the place c * scale + offset
    in the network (a vector of four, as VisPy's transforms and IdentityTransforms give)."""

    def __init__(self, scale, offset):
        self.scale, self.offset = float(scale), np.asarray(offset, dtype=float)

    def map(self, point):
        place = np.asarray(point, dtype=float)[:2] * self.scale + self.offset
        return np.array([place[0], place[1], 0.0, 1.0])

    def imap(self, point):
        place = (np.asarray(point, dtype=float)[:2] - self.offset) / self.scale
        return np.array([place[0], place[1], 0.0, 1.0])


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

class ZoomTimer:
    """A QTimer's start, stop and timeout, for the smooth wheel zoom, fired by the test."""

    def __init__(self):
        self.active = False
        self.timeout = SimpleNamespace(connect=self._connect)
        self._slot = None

    def _connect(self, slot):
        self._slot = slot

    def isActive(self):
        return self.active

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def fire(self):
        self._slot()


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


# =====================================================================
# Copy on click
# =====================================================================

class CopyOnClickTests(GestureTestCase):
    def setUp(self):
        super().setUp()
        self.clipboard = mock.Mock()
        native = SimpleNamespace(clipboard=lambda: self.clipboard)
        patcher = mock.patch("vispy.app.use_app", return_value=SimpleNamespace(native=native))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = self.make_viewer()
        self.viewer.sequences_map = {f"N{index}": f"SEQ{index}" for index in range(6)}
        self.viewer._show_click_status.reset_mock()

    def copied(self):
        return [call.args[0] for call in self.clipboard.setText.call_args_list]

    def timer(self):
        return getattr(self.viewer, "_header_copy_timer", None)

    def shift_click(self, node=1, **offset):
        x, y = NODE[node]
        self.press(self.viewer, x, y, button=1, modifiers=SHIFT)
        self.release(self.viewer, x + offset.get("dx", 0.0), y + offset.get("dy", 0.0), button=1)

    def test_a_shift_click_copies_the_header_once_the_double_click_interval_has_passed(self):
        ev = self.press(self.viewer, *NODE[1], button=1, modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.assertIsNone(self.timer())
        ev = self.release(self.viewer, *NODE[1], button=1)
        self.assertTrue(ev.handled)
        self.assertEqual(self.copied(), [])                       # not on press, nor on release
        self.assertEqual(self.status(self.viewer), [])
        self.assertTrue(self.timer().running)
        self.assertEqual(self.timer().starts, [self.DOUBLE_CLICK_MS / 1000.0])      # Qt's interval
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1"])
        self.assertEqual(self.status(self.viewer), ["Copied: N1"])

    def test_the_wait_is_the_one_qt_gives(self):
        with mock.patch.object(QtWidgets.QApplication, "doubleClickInterval", return_value=500):
            self.shift_click()
        self.assertEqual(self.timer().starts, [0.5])

    def test_a_double_click_copies_the_sequence_alone_and_drops_the_header(self):
        self.shift_click()
        self.assertTrue(self.timer().running)
        ev = self.double_click(self.viewer, *NODE[1], modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.assertFalse(self.timer().running)                    # the header's wait is over, with no copy
        self.assertEqual(self.copied(), [])                       # the sequence waits for the release
        ev = self.release(self.viewer, *NODE[1], button=1)
        self.assertTrue(ev.handled)
        self.assertEqual(self.copied(), ["SEQ1"])                 # one entry for the clipboard's history
        self.assertEqual(self.clipboard.setText.call_count, 1)
        self.assertEqual(self.status(self.viewer), ["Copied sequence of: N1"])
        self.assertIsNone(self.viewer._header_copy_node)

    def test_a_slipped_double_click_copies_nothing_at_all(self):
        self.shift_click()
        self.double_click(self.viewer, *NODE[1], modifiers=SHIFT)
        self.release(self.viewer, NODE[1][0] + 6, NODE[1][1], button=1)
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(self.viewer), [])
        self.assertFalse(self.timer().running)

    def test_a_slipped_click_starts_no_wait_and_copies_nothing(self):
        for dx, dy in ((5.0, 0.0), (0.0, -5.0), (3.0, 4.0), (30.0, 0.0), (3.6, 3.6)):
            with self.subTest(moved=(dx, dy)):
                self.shift_click(dx=dx, dy=dy)
                self.assertTrue(self.timer() is None or not self.timer().running)
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(self.viewer), [])

    def test_a_release_within_five_pixels_is_a_click(self):
        for shift in (4.9, 3.0, 0.0):
            with self.subTest(moved=shift):
                self.shift_click(dx=shift)
                self.assertTrue(self.timer().running)
                self.timer().fire()
        self.assertEqual(self.copied(), ["N1"] * 3)

    def test_a_shift_drag_copies_nothing_and_does_nothing_else(self):
        viewer = self.viewer
        viewer.selected_indices = [2]
        start = viewer.pos.copy()
        self.press(viewer, *NODE[1], button=1, modifiers=SHIFT)
        for x in (50, 80, 120):
            self.move(viewer, x, 30, buttons=(1,), modifiers=SHIFT)
        self.release(viewer, 120, 30, button=1)
        self.assertIsNone(self.timer())
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(viewer), [])
        self.assertEqual(viewer.selected_indices, [2])
        np.testing.assert_array_equal(viewer.pos, start)
        self.assertFalse(viewer.is_multi_dragging or viewer.is_box_selecting)
        viewer.apply_left_click_focus.assert_not_called()
        viewer.clear_left_click_focus.assert_not_called()

    def test_ctrl_copies_nothing_and_is_a_plain_left_click_or_drag(self):
        for modifier in ("Control", "Meta"):
            with self.subTest(modifier=modifier):
                viewer = self.make_viewer()
                viewer.sequences_map = {f"N{index}": f"SEQ{index}" for index in range(6)}
                self.press(viewer, *NODE[1], button=1, modifiers=[modifier])
                viewer.apply_left_click_focus.assert_called_once_with(1)       # highlighted, as a plain click does
                self.release(viewer, *NODE[1], button=1, modifiers=[modifier])
                self.press(viewer, 100, 60, button=1, modifiers=[modifier])
                viewer.clear_left_click_focus.assert_called_once_with()        # and on empty space
                self.move(viewer, 140, 90, buttons=(1,), modifiers=[modifier])
                self.release(viewer, 140, 90, button=1, modifiers=[modifier])
                self.double_click(viewer, 100, 60, modifiers=[modifier])
                self.assertIsNone(getattr(viewer, "_header_copy_timer", None))
                self.assertEqual(self.copied(), [])
                self.assertEqual(self.status(viewer), [])

    def test_the_node_is_the_one_at_the_press(self):
        viewer = self.viewer
        with mock.patch.object(viewer, "pick_hovered_node", return_value=1) as pick:
            self.press(viewer, *NODE[1], button=1, modifiers=SHIFT)
            pick.assert_called_once()
            np.testing.assert_array_equal(pick.call_args.args[0], NODE[1])
            pick.return_value = 3                       # the pointer is over another node at the release
            self.release(viewer, NODE[1][0] + 2, NODE[1][1], button=1, modifiers=CTRL)
            pick.assert_called_once()                   # the release asked nothing
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1"])

    def test_a_press_on_no_node_copies_nothing(self):
        ev = self.press(self.viewer, 100, 60, button=1, modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.release(self.viewer, 100, 60, button=1)
        ev = self.double_click(self.viewer, 100, 60, modifiers=SHIFT)
        self.release(self.viewer, 100, 60, button=1)
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(self.viewer), [])
        self.assertTrue(self.timer() is None or not self.timer().running)

    def test_a_plain_left_click_still_highlights_on_the_press(self):
        viewer = self.viewer
        self.press(viewer, *NODE[2], button=1)
        viewer.apply_left_click_focus.assert_called_once_with(2)
        self.release(viewer, *NODE[2], button=1)
        viewer.apply_left_click_focus.assert_called_once_with(2)
        self.assertEqual(self.copied(), [])

    def test_a_later_press_forgets_a_press_whose_release_was_missed(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1], button=1, modifiers=SHIFT)      # no release comes
        self.press(viewer, NODE[1][0] + 2, NODE[1][1], button=1)     # a plain click, nearby
        self.release(viewer, NODE[1][0] + 2, NODE[1][1], button=1)
        self.assertIsNone(self.timer())
        self.assertEqual(self.copied(), [])

    def test_the_release_of_another_button_copies_nothing_and_keeps_the_press(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1], button=1, modifiers=SHIFT)
        self.release(viewer, *NODE[1], button=2)
        self.assertIsNone(self.timer())
        self.release(viewer, *NODE[1], button=1)
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1"])

    def test_one_press_copies_once(self):
        self.shift_click()
        self.release(self.viewer, *NODE[1], button=1)
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1"])
        self.assertFalse(self.timer().running)

    def test_a_second_click_on_another_node_is_a_click_of_its_own_and_both_are_copied(self):
        self.shift_click(1)
        # Qt sends a second press as a double click only close to the first: this one is on node 4.
        self.double_click(self.viewer, *NODE[4], modifiers=SHIFT)
        self.release(self.viewer, *NODE[4], button=1)
        self.assertEqual(self.copied(), ["N1"])                    # the first one's wait was over at once
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1", "N4"])
        self.assertEqual(self.status(self.viewer), ["Copied: N1", "Copied: N4"])

    def test_two_clicks_on_different_nodes_are_both_copied(self):
        self.shift_click(1)
        self.shift_click(2)                                        # the press of a node nearer than Qt's distance, say
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1", "N2"])

    def test_a_shift_click_after_a_plain_one_is_a_click_not_a_double_click(self):
        # Qt calls it a double click, though nothing waited for it; it is the Shift click the user made.
        ev = self.double_click(self.viewer, *NODE[1], modifiers=SHIFT)
        self.assertTrue(ev.handled)
        self.release(self.viewer, *NODE[1], button=1)
        self.assertEqual(self.copied(), [])
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1"])

    def test_a_click_after_the_wait_has_nothing_to_do_with_the_one_before(self):
        self.shift_click()
        self.timer().fire()
        self.double_click(self.viewer, *NODE[1], modifiers=SHIFT)  # nothing waits now: a new click
        self.release(self.viewer, *NODE[1], button=1)
        self.timer().fire()
        self.assertEqual(self.copied(), ["N1", "N1"])

    def test_any_other_double_click_is_the_press_it_replaces(self):
        # A plain left click highlights, a Ctrl one does the same, a right one selects: as the press would.
        for button, modifiers in ((1, []), (1, CTRL)):
            self.double_click(self.viewer, *NODE[1], button=button, modifiers=modifiers)
        self.assertEqual(self.viewer.apply_left_click_focus.call_args_list, [mock.call(1), mock.call(1)])
        self.assertIsNone(self.timer())
        self.assertEqual(self.viewer.selected_indices, [])
        ev = self.double_click(self.viewer, *NODE[2], button=2)
        self.assertTrue(ev.handled)
        self.assertEqual(self.viewer.selected_indices, [2])
        self.assertTrue(self.viewer.is_multi_dragging)
        self.assertEqual(self.copied(), [])

    def test_a_sequence_that_is_not_there_says_so_on_release(self):
        viewer = self.viewer
        viewer.sequences_map = {}
        self.shift_click()
        self.double_click(viewer, *NODE[1], modifiers=SHIFT)
        self.assertEqual(self.status(viewer), [])
        self.release(viewer, *NODE[1], button=1)
        self.assertEqual(self.status(viewer), ["Sequence not found for: N1"])
        self.assertEqual(self.copied(), [])

    def test_a_failed_copy_says_so(self):
        viewer = self.viewer
        self.clipboard.setText.side_effect = RuntimeError("no clipboard")
        self.shift_click()
        self.timer().fire()
        self.assertEqual(self.status(viewer), ["Copy Failed: N1"])

    def test_the_canvas_sends_double_clicks_and_key_releases_to_the_viewer(self):
        import inspect
        source = inspect.getsource(MainViewer.__init__)
        self.assertIn("self.canvas.events.mouse_double_click.connect(self.on_mouse_double_click)", source)
        self.assertIn("self.canvas.events.key_release.connect(self.on_key_release)", source)


class QtDoubleClickTests(unittest.TestCase):
    SCRIPT = """
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6 import QtCore, QtGui, QtWidgets
from vispy import app, scene
app.use_app("pyside6")
canvas = scene.SceneCanvas(keys=None, show=False, size=(300, 200))
seen = []
canvas.events.mouse_press.connect(lambda ev: seen.append(["press", ev.button]))
canvas.events.mouse_release.connect(lambda ev: seen.append(["release", ev.button]))
canvas.events.mouse_double_click.connect(lambda ev: seen.append(["double click", ev.button, "Shift" in ev.modifiers]))
point = QtCore.QPointF(50, 60)
for kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease,
             QtCore.QEvent.Type.MouseButtonDblClick, QtCore.QEvent.Type.MouseButtonRelease):
    event = QtGui.QMouseEvent(kind, point, canvas.native.mapToGlobal(point.toPoint()),
                              QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton,
                              QtCore.Qt.KeyboardModifier.ShiftModifier)
    QtWidgets.QApplication.sendEvent(canvas.native, event)
print("SEEN", json.dumps(seen))
"""

    def test_qt_sends_the_second_press_of_a_double_click_as_a_double_click_instead_of_a_press(self):
        # Why the Viewer reads mouse_double_click: the second click of a Shift double click never
        # arrives as a press. A VisPy canvas is made in a process of its own, as the GL probes are.
        import json
        import subprocess
        result = subprocess.run([sys.executable, "-c", self.SCRIPT], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=120,
                                env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
        line = next((line for line in result.stdout.splitlines() if line.startswith("SEEN ")), None)
        if line is None:
            self.skipTest(f"no VisPy canvas to send events to (exit {result.returncode}): {result.stderr[-500:]}")
        self.assertEqual(json.loads(line[len("SEEN "):]),
                         [["press", 1], ["release", 1], ["double click", 1, True], ["release", 1]])


if __name__ == "__main__":
    unittest.main()
