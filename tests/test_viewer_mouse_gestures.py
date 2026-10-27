# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's mouse and keyboard gestures on a node drag, and the copy click
(EMAPSSN_Viewer.py: on_key_press, on_mouse_press, on_mouse_move,
on_mouse_wheel, on_mouse_release).

* Esc cancels a drag in progress. A drag of selected nodes puts every node back
  where it was at the press, a rotation too, keeps the selection the press made
  and leaves no step on the undo history; a selection box is hidden and the
  selection from before the press comes back. The rest of that right-button drag
  does nothing, nothing the release does, and the camera does not take it for a
  zoom. The command line stays open for the Esc that cancels a drag.
* Shift while dragging nodes holds the move to the larger of the offsets from
  the press, horizontal or vertical, and the wheel's rotation never moves it.
* The wheel turns the dragged nodes 5 degrees a notch, and with Shift 0.5.
* A Shift or Ctrl left-click copies on release, within 5 px of the press.

The tests send synthetic VisPy-style events to a Viewer built as
test_viewer_markers builds one: real markers and pick, a recording edge layer.
"""
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

import Command_Engine  # noqa: E402
import EMAPSSN_Config as cfg  # noqa: E402
from EMAPSSN_Viewer import CLICK_DISTANCE_PIXELS, MainViewer  # noqa: E402
from tests import test_viewer_markers as marker_tests  # noqa: E402

# The nodes lie 40 px apart on y = 0, as test_viewer_markers places them, and
# a click on a node's own place picks it.
NODE = [(40.0 * index, 0.0) for index in range(6)]


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

    def note_motion(self):
        pass


class GestureTestCase(unittest.TestCase):
    """A Viewer to drag nodes in, and the events that do it."""

    def setUp(self):
        for patcher in (mock.patch.object(cfg, "UMAP_MODE", False, create=True),
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", False),
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", False),
                        marker_tests.distinct_outlines(),     # a selected node's outline differs
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
    def esc(viewer):
        ev = escape()
        viewer.on_key_press(ev)
        return ev

    @staticmethod
    def rotated(points, about, degrees):
        angle = np.radians(degrees)
        rel = np.asarray(points, dtype=float) - about
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.column_stack([rel[:, 0] * cos_a - rel[:, 1] * sin_a,
                                rel[:, 0] * sin_a + rel[:, 1] * cos_a]) + about

    def assert_picture_matches(self, viewer):
        """The markers, the edge layer and the pick show the nodes where they are."""
        order = viewer._submitted_marker_node_order
        np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
        np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
        self.assertIsNotNone(viewer._marker_picker)
        for node in range(len(viewer.pos)):
            self.assertEqual(viewer.pick_hovered_node(viewer.pos[node, :2]), node)


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
                self.wheel(viewer, (0.0, 3.0), 70, 20, modifiers=["Shift"])        # 1.5 more
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
# Shift holds a drag of nodes to a line
# =====================================================================

class AxisLockTests(GestureTestCase):
    def drag_to(self, viewer, x, y, modifiers=()):
        """Move the pointer to (x, y); return where the nodes moved from their places at the press."""
        self.move(viewer, x, y, modifiers=modifiers)
        return viewer.pos[[1, 2], :2] - np.array([NODE[1], NODE[2]], dtype=np.float32)

    def test_without_shift_the_nodes_follow_the_pointer(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, 70, 55), [[30, 55]] * 2)

    def test_shift_keeps_the_larger_offset_and_switches_with_it(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                shift = ["Shift"]
                # The offsets are from the press, (40, 0): horizontal while |dx| >= |dy| ...
                np.testing.assert_allclose(self.drag_to(viewer, 70, 10, shift), [[30, 0]] * 2)
                np.testing.assert_allclose(self.drag_to(viewer, 65, 25, shift), [[25, 0]] * 2)
                np.testing.assert_allclose(self.drag_to(viewer, 65, 25.5, shift), [[0, 25.5]] * 2)
                # ... vertical once |dy| is larger, whichever way it goes ...
                np.testing.assert_allclose(self.drag_to(viewer, 50, -60, shift), [[0, -60]] * 2)
                # ... and back again.
                np.testing.assert_allclose(self.drag_to(viewer, 130, -60, shift), [[90, 0]] * 2)
                np.testing.assert_allclose(self.drag_to(viewer, 0, 20, shift), [[-40, 0]] * 2)   # leftward
                self.assert_picture_matches_after_move(viewer)

    def assert_picture_matches_after_move(self, viewer):
        np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])

    def test_the_line_goes_as_shift_is_let_go_and_pressed_again(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, 70, 20, ["Shift"]), [[30, 0]] * 2)
        np.testing.assert_allclose(self.drag_to(viewer, 71, 21), [[31, 21]] * 2)        # free again
        np.testing.assert_allclose(self.drag_to(viewer, 71, 21, ["Shift"]), [[31, 0]] * 2)

    def test_a_tie_and_the_press_itself_move_along_x(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        np.testing.assert_allclose(self.drag_to(viewer, 60, 20, ["Shift"]), [[20, 0]] * 2)
        np.testing.assert_allclose(self.drag_to(viewer, *NODE[1], modifiers=["Shift"]), [[0, 0]] * 2)

    def test_shift_at_the_press_adds_the_node_and_the_drag_is_held_to_a_line(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3], modifiers=["Shift"])
        self.assertEqual(viewer.selected_indices, [1, 2, 3])
        self.assertTrue(viewer.is_multi_dragging)
        self.move(viewer, NODE[3][0] + 5, 50, modifiers=["Shift"])
        np.testing.assert_allclose(viewer.pos[[1, 2, 3], :2] - np.array(NODE[1:4], dtype=np.float32),
                                   [[0, 50]] * 3)
        self.release(viewer, NODE[3][0] + 5, 50)
        self.assertEqual(len(viewer.position_history), 1)

    def test_shift_still_adds_a_box_to_the_selection(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer = self.make_viewer(selected=[4])
                self.press(viewer, 30, -10, modifiers=["Shift"])
                self.move(viewer, 90, 10, modifiers=["Shift"])
                self.release(viewer, 90, 10, modifiers=["Shift"])
                self.assertEqual(sorted(viewer.selected_indices), [1, 2, 4])


# =====================================================================
# The wheel turns the nodes
# =====================================================================

class RotationTests(GestureTestCase):
    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 2])
        self.press(self.viewer, *NODE[1])
        self.pivot = np.array(NODE[1])

    def test_the_wheel_turns_five_degrees_a_notch(self):
        for notches in (1.0, -1.0, 3.0):
            with self.subTest(notches=notches):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, notches), *NODE[1])
                np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                           self.rotated([NODE[1], NODE[2]], self.pivot, 5.0 * notches),
                                           atol=1e-3)

    def test_shift_turns_half_a_degree_a_notch(self):
        for notches in (1.0, -1.0, 4.0):
            with self.subTest(notches=notches):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, notches), *NODE[1], modifiers=["Shift"])
                np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                           self.rotated([NODE[1], NODE[2]], self.pivot, 0.5 * notches),
                                           atol=1e-3)

    def test_a_fraction_of_a_notch_turns_that_fraction(self):
        # A high-resolution wheel or a touchpad reports parts of a notch, and the turn follows.
        for modifiers, degrees in (([], 5.0), (["Shift"], 0.5)):
            with self.subTest(modifiers=modifiers):
                viewer = self.make_viewer(selected=[1, 2])
                self.press(viewer, *NODE[1])
                self.wheel(viewer, (0.0, 0.1), *NODE[1], modifiers=modifiers)
                self.wheel(viewer, (0.0, 0.15), *NODE[1], modifiers=modifiers)   # turns add up
                np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                           self.rotated([NODE[1], NODE[2]], self.pivot, 0.25 * degrees),
                                           atol=1e-3)

    def test_shift_and_a_sideways_scroll_turns_the_same_way_as_a_scroll_up(self):
        # macOS, and some mouse drivers on Windows, send Shift + wheel as a horizontal scroll.
        results = {}
        for name, delta in (("up", (0.0, 2.0)), ("sideways up", (2.0, 0.0)),
                            ("down", (0.0, -2.0)), ("sideways down", (-2.0, 0.0))):
            viewer = self.make_viewer(selected=[1, 2])
            self.press(viewer, *NODE[1])
            self.wheel(viewer, delta, *NODE[1], modifiers=["Shift"])
            results[name] = viewer.pos[[1, 2], :2].copy()
        np.testing.assert_allclose(results["sideways up"], results["up"])
        np.testing.assert_allclose(results["sideways down"], results["down"])
        np.testing.assert_allclose(results["up"],
                                   self.rotated([NODE[1], NODE[2]], self.pivot, 1.0), atol=1e-3)
        self.assertFalse(np.allclose(results["up"], results["down"]))

    def test_the_vertical_notches_win_when_both_are_given(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[1])
        self.wheel(viewer, (5.0, 1.0), *NODE[1], modifiers=["Shift"])
        np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                   self.rotated([NODE[1], NODE[2]], self.pivot, 0.5), atol=1e-3)

    def test_without_shift_a_sideways_scroll_turns_nothing(self):
        before = self.viewer.pos.copy()
        self.wheel(self.viewer, (3.0, 0.0), *NODE[1])
        np.testing.assert_array_equal(self.viewer.pos, before)

    def test_shift_and_the_wheel_never_move_a_diagonal_move_onto_an_axis(self):
        viewer = self.viewer
        self.move(viewer, 70, 20)                        # a diagonal move of (30, 20), no Shift
        for fine in (["Shift"], []):
            self.wheel(viewer, (0.0, 1.0), 70, 20, modifiers=fine)
        turned = self.rotated([NODE[1], NODE[2]], self.pivot, 5.5)
        np.testing.assert_allclose(viewer.pos[[1, 2], :2], turned + [30.0, 20.0], atol=1e-3)
        # The line applies when the pointer next moves.
        self.move(viewer, 75, 22, modifiers=["Shift"])
        np.testing.assert_allclose(viewer.pos[[1, 2], :2], turned + [35.0, 0.0], atol=1e-3)

    def test_the_wheel_takes_the_move_the_last_move_gave_not_the_pointer(self):
        viewer = self.viewer
        self.move(viewer, 70, 10, modifiers=["Shift"])          # held to the line: (30, 0)
        # The wheel event is at another place, and without Shift: neither counts.
        self.wheel(viewer, (0.0, 1.0), 300, 200)
        np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                   self.rotated([NODE[1], NODE[2]], self.pivot, 5.0) + [30.0, 0.0],
                                   atol=1e-3)

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
                self.wheel(viewer, (0.0, 18.0), *NODE[1], modifiers=["Shift"])
                order = viewer._submitted_marker_node_order
                np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
                np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])


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

    def status(self):
        return [str(call.args[0]) for call in self.viewer._show_click_status.call_args_list]

    def test_the_copy_is_made_on_release_not_on_press(self):
        for modifier, text, line in (("Shift", "N1", "Copied: N1"),
                                     ("Control", "SEQ1", "Copied sequence of: N1"),
                                     ("Meta", "SEQ1", "Copied sequence of: N1")):
            with self.subTest(modifier=modifier):
                self.clipboard.reset_mock()
                self.viewer._show_click_status.reset_mock()
                ev = self.press(self.viewer, *NODE[1], button=1, modifiers=[modifier])
                self.assertTrue(ev.handled)
                self.assertEqual(self.copied(), [])
                self.assertEqual(self.status(), [])
                ev = self.release(self.viewer, *NODE[1], button=1)
                self.assertTrue(ev.handled)
                self.assertEqual(self.copied(), [text])
                self.assertEqual(self.status(), [line])

    def test_a_release_within_five_pixels_copies_and_five_or_more_does_not(self):
        for shift in (4.9, 3.0, 0.0):
            with self.subTest(moved=shift):
                self.clipboard.reset_mock()
                self.press(self.viewer, *NODE[1], button=1, modifiers=["Shift"])
                self.release(self.viewer, NODE[1][0] + shift, NODE[1][1], button=1)
                self.assertEqual(self.copied(), ["N1"])
        self.assertEqual(CLICK_DISTANCE_PIXELS, 5.0)
        for dx, dy in ((5.0, 0.0), (0.0, -5.0), (3.0, 4.0), (30.0, 0.0), (3.6, 3.6)):
            for modifier in ("Shift", "Control"):
                with self.subTest(moved=(dx, dy), modifier=modifier):
                    self.clipboard.reset_mock()
                    self.viewer._show_click_status.reset_mock()
                    self.press(self.viewer, *NODE[1], button=1, modifiers=[modifier])
                    self.release(self.viewer, NODE[1][0] + dx, NODE[1][1] + dy, button=1)
                    self.assertEqual(self.copied(), [])
                    self.assertEqual(self.status(), [])

    def test_a_shift_or_ctrl_drag_copies_nothing_and_does_nothing_else(self):
        viewer = self.viewer
        viewer.selected_indices = [2]
        start = viewer.pos.copy()
        for modifier in ("Shift", "Control"):
            self.press(viewer, *NODE[1], button=1, modifiers=[modifier])
            for x in (50, 80, 120):
                self.move(viewer, x, 30, buttons=(1,), modifiers=[modifier])
            self.release(viewer, 120, 30, button=1)
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(), [])
        self.assertEqual(viewer.selected_indices, [2])
        np.testing.assert_array_equal(viewer.pos, start)
        self.assertFalse(viewer.is_multi_dragging or viewer.is_box_selecting)
        viewer.apply_left_click_focus.assert_not_called()
        viewer.clear_left_click_focus.assert_not_called()

    def test_the_node_and_the_key_are_those_at_the_press(self):
        viewer = self.viewer
        with mock.patch.object(viewer, "pick_hovered_node", return_value=1) as pick:
            self.press(viewer, *NODE[1], button=1, modifiers=["Shift"])
            pick.assert_called_once()
            np.testing.assert_array_equal(pick.call_args.args[0], NODE[1])
            pick.return_value = 3                       # the pointer is over another node at the release
            self.release(viewer, NODE[1][0] + 2, NODE[1][1], button=1, modifiers=["Control"])
            pick.assert_called_once()                   # the release asked nothing
        self.assertEqual(self.copied(), ["N1"])         # the header: Shift was held at the press
        self.press(viewer, *NODE[1], button=1, modifiers=["Control"])
        self.release(viewer, *NODE[1], button=1, modifiers=["Shift"])
        self.assertEqual(self.copied(), ["N1", "SEQ1"])  # the sequence: Ctrl at the press

    def test_a_press_on_no_node_copies_nothing(self):
        for modifier in ("Shift", "Control"):
            ev = self.press(self.viewer, 100, 60, button=1, modifiers=[modifier])
            self.assertTrue(ev.handled)
            self.release(self.viewer, 100, 60, button=1)
        self.assertEqual(self.copied(), [])
        self.assertEqual(self.status(), [])

    def test_a_plain_left_click_still_highlights_on_the_press(self):
        viewer = self.viewer
        self.press(viewer, *NODE[2], button=1)
        viewer.apply_left_click_focus.assert_called_once_with(2)
        self.release(viewer, *NODE[2], button=1)
        viewer.apply_left_click_focus.assert_called_once_with(2)
        self.assertEqual(self.copied(), [])

    def test_a_later_press_forgets_a_press_whose_release_was_missed(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1], button=1, modifiers=["Shift"])      # no release comes
        self.press(viewer, NODE[1][0] + 2, NODE[1][1], button=1)         # a plain click, nearby
        self.release(viewer, NODE[1][0] + 2, NODE[1][1], button=1)
        self.assertEqual(self.copied(), [])

    def test_the_release_of_another_button_copies_nothing_and_keeps_the_press(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1], button=1, modifiers=["Shift"])
        self.release(viewer, *NODE[1], button=2)
        self.assertEqual(self.copied(), [])
        self.release(viewer, *NODE[1], button=1)
        self.assertEqual(self.copied(), ["N1"])

    def test_one_press_copies_once(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1], button=1, modifiers=["Shift"])
        self.release(viewer, *NODE[1], button=1)
        self.release(viewer, *NODE[1], button=1)
        self.assertEqual(self.copied(), ["N1"])

    def test_a_sequence_that_is_not_there_says_so_on_release(self):
        viewer = self.viewer
        viewer.sequences_map = {}
        self.press(viewer, *NODE[1], button=1, modifiers=["Control"])
        self.assertEqual(self.status(), [])
        self.release(viewer, *NODE[1], button=1)
        self.assertEqual(self.status(), ["Sequence not found for: N1"])
        self.assertEqual(self.copied(), [])

    def test_a_failed_copy_says_so_on_release(self):
        viewer = self.viewer
        self.clipboard.setText.side_effect = RuntimeError("no clipboard")
        self.press(viewer, *NODE[1], button=1, modifiers=["Shift"])
        self.release(viewer, *NODE[1], button=1)
        self.assertEqual(self.status(), ["Copy Failed: N1"])


if __name__ == "__main__":
    unittest.main()
