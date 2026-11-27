# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's mouse selection, click, cancel, and copy-on-click gestures
(EMAPSSN_Viewer.py: on_key_press, on_mouse_press, on_mouse_double_click,
on_mouse_move, on_mouse_release): Esc cancelling drags and boxes, Shift
selection toggles, click status reporting, right-press focus clearing, idle
undo-step cleanup and button state, box toggling, Shift copy-on-click, and
Qt double-click delivery."""
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

import Command_Engine  # noqa: E402
import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from EMAPSSN_Viewer import CLICK_DISTANCE_PIXELS, MainViewer  # noqa: E402
from tests.mouse_gesture_fixtures import CTRL, GestureTestCase, NODE, SHIFT, event  # noqa: E402


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
        # Why neither Esc nor a release keeps the press's snapshot when the press only changed the
        # selection: it holds the nodes' places and the colours, sizes and so on, never the
        # selection, so undoing it would change nothing.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])
        self.release(viewer, *NODE[3])
        self.assertEqual(viewer.selected_indices, [3])
        self.assertEqual(viewer.position_history, [])                 # a step with nothing to undo is no step
        self.assertFalse(viewer._do_undo(announce=False))
        # Moved, the same press keeps its step, which holds no selection and brings none back.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[3])
        self.move(viewer, 150, 50)
        self.release(viewer, 150, 50)
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
        # Undo holds the nodes' places, colours and so on, never the selection, so it would not
        # bring the node back: the press's snapshot, which every node click takes, is taken off.
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, *NODE[2], modifiers=SHIFT)
        self.release(viewer, *NODE[2])
        self.assertEqual(viewer.selected_indices, [1])
        self.assertEqual(viewer.position_history, [])
        self.assertFalse(viewer._do_undo(announce=False))
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
        self.assertEqual(self.status(viewer), [])               # "Clicked: <header>" is the focus's, mocked here


class RightPressEmptySelectionAndFocusTests(GestureTestCase):
    """A right click over an empty selection says and changes nothing about it; a right press takes the click focus off."""

    EMPTY = (30, -10)          # no node here

    def make_viewer(self, selected=(), focus=None):
        viewer = super().make_viewer(selected=selected)
        del viewer.clear_left_click_focus          # the real one: the press takes the focus off through it
        viewer.update_nodes = mock.Mock()
        viewer.update_selection_visual = mock.Mock()
        viewer.selected_node_idx = focus
        return viewer

    def click(self, viewer, at):
        self.press(viewer, *at)
        self.release(viewer, *at)

    def selection_news(self, viewer):
        return [call for call in viewer.broadcast_event.call_args_list if call.args[0].get("type") == "selection_changed"]

    def focus_news(self, viewer):
        return [call.args[0] for call in viewer.broadcast_event.call_args_list if call.args[0].get("type") == "highlight_row"]

    def test_an_empty_selection_that_stays_empty_says_and_does_nothing(self):
        viewer = self.make_viewer(selected=[])
        self.click(viewer, self.EMPTY)
        self.assertEqual(self.status(viewer), [])                        # no "Selection cleared."
        self.assertEqual(viewer.selection_epoch, 0)                      # the commands' selection timeline is as it was
        self.assertEqual(self.selection_news(viewer), [])
        viewer.update_selection_visual.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_a_shift_click_on_empty_space_over_an_empty_selection_says_nothing_either(self):
        viewer = self.make_viewer(selected=[])
        self.press(viewer, *self.EMPTY, modifiers=SHIFT)
        self.release(viewer, *self.EMPTY, modifiers=SHIFT)
        self.assertEqual(self.status(viewer), [])
        self.assertEqual(viewer.selection_epoch, 0)

    def test_a_selection_that_is_cleared_says_so(self):
        viewer = self.make_viewer(selected=[1, 2])
        self.click(viewer, self.EMPTY)
        self.assertEqual(viewer.selected_indices, [])
        self.assertEqual(self.status(viewer), ["Selection cleared."])
        self.assertEqual(viewer.selection_epoch, 1)
        viewer.update_selection_visual.assert_called_once_with()

    def test_a_box_that_selects_nothing_from_nothing_says_nothing_and_one_that_selects_says_so(self):
        viewer = self.make_viewer(selected=[])
        self.press(viewer, 100, 100)
        self.move(viewer, 140, 140)                                      # a box over no node
        self.release(viewer, 140, 140)
        self.assertEqual(self.status(viewer), [])
        viewer = self.make_viewer(selected=[1, 2])
        self.press(viewer, 30, -10)
        self.move(viewer, 90, 10)                                        # the same two nodes again
        self.release(viewer, 90, 10)
        self.assertEqual(self.status(viewer), ["Selected 2 nodes."])

    def test_a_right_press_with_a_focus_on_takes_it_off_and_tells_the_pages(self):
        for where, name in ((self.EMPTY, "empty space"), (NODE[3], "a node")):
            with self.subTest(over=name):
                viewer = self.make_viewer(selected=[], focus=2)
                viewer.left_click_highlight_indices = None
                self.press(viewer, *where)
                self.assertIsNone(viewer.selected_node_idx)
                self.assertIsNone(viewer.left_click_highlight_indices)
                self.assertEqual(self.focus_news(viewer), [{"type": "highlight_row", "index": None}])

    def test_a_right_press_with_a_highlight_set_takes_that_off_too(self):
        viewer = self.make_viewer(selected=[], focus=None)
        viewer.left_click_highlight_indices = [1, 2]
        self.press(viewer, *self.EMPTY)
        self.assertIsNone(viewer.left_click_highlight_indices)
        self.assertEqual(self.focus_news(viewer), [{"type": "highlight_row", "index": None}])

    def test_a_right_press_with_no_focus_on_tells_the_pages_nothing(self):
        for where in (self.EMPTY, NODE[3]):
            with self.subTest(where=where):
                viewer = self.make_viewer(selected=[], focus=None)
                self.press(viewer, *where)
                self.release(viewer, *where)
                self.assertEqual(self.focus_news(viewer), [])

    def test_the_press_redraws_the_markers_once_at_most(self):
        # Over a node update_selection_visual does it; over empty space the press does, for the halo it took off.
        viewer = self.make_viewer(selected=[], focus=2)
        self.press(viewer, *NODE[3])
        viewer.update_nodes.assert_not_called()
        viewer.update_selection_visual.assert_called_once()
        viewer = self.make_viewer(selected=[], focus=2)
        self.press(viewer, *self.EMPTY)
        viewer.update_nodes.assert_called_once_with()
        viewer = self.make_viewer(selected=[], focus=None)
        self.press(viewer, *self.EMPTY)
        viewer.update_nodes.assert_not_called()


class IdlePressStepTests(GestureTestCase):
    """A press on a node saves an undo step, but only one that moved the nodes stays when the button is let go."""

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

    def test_a_press_that_only_changes_the_selection_leaves_no_step_either(self):
        # Undo restores no selection, so the step would undo nothing: a quick Shift + right
        # double-click, which adds a node and takes it out again, left two of them.
        for knn in (False, True):
            for name, modifiers, selection in (("plain", [], [3]), ("shift add", SHIFT, [1, 2, 3]),
                                               ("ctrl", CTRL, [3]), ("meta", ["Meta"], [3])):
                with self.subTest(knn=knn, press=name), \
                        mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=knn):
                    self.setUp()
                    self.press(self.viewer, *NODE[3], modifiers=modifiers)
                    self.release(self.viewer, *NODE[3], modifiers=modifiers)
                    self.assertEqual(self.viewer.selected_indices, selection)
                    self.assert_history_as_before()

    def test_a_shift_click_that_removes_a_node_leaves_no_step(self):
        for knn in (False, True):
            with self.subTest(knn=knn), mock.patch.object(EMAPSSN_Viewer, "knn_topology_layout", return_value=knn):
                self.setUp()
                self.press(self.viewer, *NODE[2], modifiers=SHIFT)
                self.release(self.viewer, *NODE[2], modifiers=SHIFT)
                self.assertEqual(self.viewer.selected_indices, [1])
                self.assert_history_as_before()

    def test_two_quick_shift_right_clicks_that_add_a_node_and_take_it_out_leave_no_step(self):
        self.press(self.viewer, *NODE[3], modifiers=SHIFT)
        self.release(self.viewer, *NODE[3], modifiers=SHIFT)
        self.double_click(self.viewer, *NODE[3], button=2, modifiers=SHIFT)     # Qt sends the second press so
        self.release(self.viewer, *NODE[3], modifiers=SHIFT)
        self.assertEqual(self.viewer.selected_indices, [1, 2])
        self.assert_history_as_before()

    def test_a_press_that_changed_the_selection_and_moved_the_nodes_keeps_its_step(self):
        self.press(self.viewer, *NODE[3])
        self.move(self.viewer, 150, 50)
        self.release(self.viewer, 150, 50)
        self.assertEqual(self.viewer.selected_indices, [3])
        self.assertEqual(len(self.viewer.position_history), 2)
        self.assertEqual(self.viewer.redo_stack, [])

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

    def test_a_release_during_a_wheel_turn_leaves_the_turns_step_for_esc(self):
        self.wheel(self.viewer, (0.0, 1.0), 60, 30, buttons=(), modifiers=CTRL)
        self.assertEqual(len(self.viewer.position_history), 2)
        self.release(self.viewer, 150, 50, button=1)                  # a stray release, no press before it
        self.assertEqual(len(self.viewer.position_history), 2)
        with mock.patch.object(Command_Engine, "show_status"):
            self.assertTrue(self.esc(self.viewer).handled)
        self.assert_history_as_before()

    def test_a_press_after_a_missed_release_starts_afresh(self):
        self.press(self.viewer, *NODE[1])                            # no release comes
        self.press(self.viewer, *NODE[1])
        self.release(self.viewer, *NODE[1])
        self.assertEqual(len(self.viewer.position_history), 2)       # the first press's step, and no more
        self.assertIs(self.viewer.position_history[0], self.earlier[0])

    def test_a_press_on_empty_space_forgets_an_earlier_press(self):
        self.press(self.viewer, *NODE[1])
        self.press(self.viewer, 30, -10)                             # a box
        self.assertIsNone(self.viewer._press_history)
        self.release(self.viewer, 30, -10)
        self.assertEqual(len(self.viewer.position_history), 2)       # nothing was taken back by the box's release


class PressStepButtonTests(GestureTestCase):
    """The Undo and Redo buttons show a press's step only once it is one: when the nodes move.

    A press on a node saves a snapshot at once, in case a drag follows, and a release that
    moved nothing takes it back. The buttons must not light for it and dim again at the
    release, a flash on every click, so they stay as they were until the nodes move.
    """

    def setUp(self):
        super().setUp()
        self.shown = []

    def make_viewer(self, selected=(), earlier=1, undone=1):
        """A viewer with ``earlier`` undo steps and ``undone`` redo steps, whose buttons record what they are told."""
        viewer = super().make_viewer(selected=selected)
        for _ in range(earlier):
            viewer._save_state()
        viewer.redo_stack = [{"pos": "a step that was undone"} for _ in range(undone)]
        viewer.history_buttons = SimpleNamespace(show_state=lambda *state: self.shown.append(state))
        viewer._refresh_history_buttons()
        self.shown.clear()
        return viewer

    def test_a_click_that_leaves_no_step_never_changes_the_buttons(self):
        for earlier, undone, before in ((1, 1, (True, True)), (0, 0, (False, False)),
                                        (0, 1, (False, True)), (2, 0, (True, False))):
            with self.subTest(earlier=earlier, undone=undone):
                viewer = self.make_viewer(selected=[1, 2], earlier=earlier, undone=undone)
                self.press(viewer, *NODE[1])
                viewer._refresh_history_buttons()               # whatever else asks for them, meanwhile
                self.assertEqual(len(viewer.position_history), earlier + 1)   # the snapshot is there
                self.release(viewer, *NODE[1])
                self.assertEqual(len(viewer.position_history), earlier)
                self.assertEqual(set(self.shown), {before})

    def test_a_click_that_changes_only_the_selection_never_changes_the_buttons(self):
        for modifiers in ([], SHIFT):
            with self.subTest(modifiers=modifiers):
                self.shown.clear()
                viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
                self.press(viewer, *NODE[3], modifiers=modifiers)
                self.release(viewer, *NODE[3], modifiers=modifiers)
                self.assertEqual(set(self.shown), {(False, True)})

    def test_the_buttons_show_the_step_when_the_nodes_move(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.press(viewer, *NODE[1])
        self.assertEqual(self.shown, [])
        self.move(viewer, 70, 20)
        self.assertEqual(self.shown, [(True, False)])           # Undo has the step, and the press cleared Redo
        self.move(viewer, 90, 30)
        self.release(viewer, 90, 30)
        self.assertEqual(self.shown[-1], (True, False))
        self.assertEqual(len(viewer.position_history), 1)

    def test_a_turn_by_the_wheel_during_a_drag_shows_the_step_too(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.press(viewer, *NODE[1])
        self.wheel(viewer, (0.0, 6.0), *NODE[1])
        self.assertEqual(self.shown, [(True, False)])
        self.release(viewer, *NODE[1])
        self.assertEqual(self.shown[-1], (True, False))

    def test_a_cancelled_drag_puts_the_buttons_back(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.press(viewer, *NODE[1])
        self.move(viewer, 70, 20)
        self.esc(viewer)
        self.assertEqual(self.shown[-1], (False, True))
        self.release(viewer, 70, 20)
        self.assertEqual(self.shown[-1], (False, True))

    def test_esc_before_the_nodes_move_leaves_the_buttons_as_they_were(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.press(viewer, *NODE[1])
        self.esc(viewer)
        self.assertEqual(set(self.shown), {(False, True)})

    def test_a_waiting_command_still_lights_undo_during_a_press(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=0)
        viewer.command_queue = SimpleNamespace(pending=lambda: ["a command"])
        self.press(viewer, *NODE[1])
        viewer._refresh_history_buttons()
        self.assertEqual(set(self.shown), {(True, False)})

    def test_ctrl_z_during_a_press_cancels_it_and_ctrl_y_does_nothing(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=1, undone=1)
        self.press(viewer, *NODE[1])
        self.assertEqual(len(viewer.position_history), 2)
        self.assertEqual(viewer.redo_stack, [])
        with mock.patch.object(Command_Engine, "show_status"), mock.patch.object(viewer, "_refresh_after_history"):
            self.assertFalse(viewer.redo_by_hand())              # nothing while the press is open
            self.assertEqual(len(viewer.position_history), 2)
            self.assertTrue(viewer.undo_by_hand())               # cancels the press, as Esc does
        self.assertEqual(len(viewer.position_history), 1)        # the press's snapshot is off, the earlier step stays
        self.assertEqual(len(viewer.redo_stack), 1)              # and the step it cleared from redo is given back
        self.assertEqual(self.shown[-1], (True, True))

    def test_the_first_notch_of_a_ctrl_wheel_turn_turns_the_nodes_and_lights_undo_at_once(self):
        # The turn's snapshot is no pending one: its first notch has turned the nodes by the time it is back.
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.wheel(viewer, (0.0, 1.0), 60, 30, buttons=(), modifiers=CTRL)
        self.assertEqual(self.shown, [(True, False)])
        self.assertEqual(len(viewer.position_history), 1)

    def test_a_ctrl_wheel_turn_that_ends_where_it_began_puts_the_buttons_back(self):
        viewer = self.make_viewer(selected=[1, 2], earlier=0, undone=1)
        self.wheel(viewer, (0.0, 1.0), 60, 30, buttons=(), modifiers=CTRL)
        self.wheel(viewer, (0.0, -1.0), 60, 30, buttons=(), modifiers=CTRL)
        self.key_release(viewer, "Control")
        self.assertEqual(self.shown[-1], (False, True))


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

    @unittest.skipIf(sys.platform == "darwin",
                     "off screen on macOS the child process would crash (the press reaches OpenGL with no GL context) "
                     "and leave a crash report in ~/Library/Logs/DiagnosticReports")
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
