# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's undo/redo and web-page selection interactions during mouse
gestures (EMAPSSN_Viewer.py): undo/redo during wheel turns and drags, and
web-page selection or box-gesture updates while a drag or turn is in progress."""
import os
from pathlib import Path
import sys
import threading
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
from Command_Queue import CommandQueue  # noqa: E402
from tests.mouse_gesture_fixtures import CTRL, GestureTestCase, NODE  # noqa: E402


class UndoDuringWheelTurnTests(GestureTestCase):
    """Ctrl+Z or Ctrl+Y while a Ctrl + wheel turn is open ends the turn first, so the turn is its own step.

    Undo used to take the turn's own step while the turn stayed open: the next notch turned the nodes again from
    where the turn began, with no step of its own, and the undone turn's state stayed on the redo stack.
    """

    def setUp(self):
        super().setUp()
        self.viewer = self.make_viewer(selected=[1, 2])
        self.start = self.places(self.viewer, [1, 2])
        patcher = mock.patch.object(Command_Engine, "show_status")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer._refresh_after_history = mock.Mock()
        self.viewer._save_state()                  # an earlier step, as a command leaves one
        self.earlier_steps = len(self.viewer.position_history)

    def notch(self):
        return self.wheel(self.viewer, (0.0, 1.0), 60.0, 30.0, buttons=(), modifiers=CTRL)

    def turned(self, degrees):
        return self.rotated(self.start, (60.0, 30.0), degrees)

    def timer(self):
        return self.viewer._wheel_turn_timer

    def test_undo_during_a_turn_ends_it_and_takes_the_turn_back(self):
        viewer = self.viewer
        self.notch()
        self.notch()
        self.assertTrue(viewer.undo_by_hand())
        self.assertIsNone(viewer._wheel_turn)
        self.assert_places(viewer, [1, 2], self.start)
        self.assertEqual(len(viewer.position_history), self.earlier_steps)
        self.assertEqual(len(viewer.redo_stack), 1)

    def test_a_notch_after_that_undo_is_a_new_turn_with_a_step_of_its_own(self):
        viewer = self.viewer
        self.notch()
        self.notch()
        viewer.undo_by_hand()
        self.notch()
        self.assert_places(viewer, [1, 2], self.turned(5.0))        # from where the nodes are, not from the old turn
        self.assertEqual(len(viewer.position_history), self.earlier_steps + 1)
        self.assertEqual(viewer.redo_stack, [])
        self.timer().fire()
        self.assertEqual(len(viewer.position_history), self.earlier_steps + 1)

    def test_redo_after_that_undo_brings_the_whole_turn_back(self):
        viewer = self.viewer
        self.notch()
        self.notch()
        viewer.undo_by_hand()
        self.assertTrue(viewer.redo_by_hand())
        self.assert_places(viewer, [1, 2], self.turned(10.0))

    def test_redo_during_a_turn_ends_it_and_keeps_its_step(self):
        viewer = self.viewer
        self.notch()
        viewer.redo_by_hand()
        self.assertIsNone(viewer._wheel_turn)
        self.assert_places(viewer, [1, 2], self.turned(5.0))
        self.assertEqual(len(viewer.position_history), self.earlier_steps + 1)


class UndoDuringDragTests(GestureTestCase):
    """Ctrl+Z while a right-button drag is open cancels the drag exactly as Esc does, and does nothing else.

    It used to pop the snapshot the press took and leave the drag open, so the next move carried the nodes on from
    the press positions (a node drag), or the box went on with its selection while the step before it was undone.
    """

    CASES = ("drag", "turn", "box")

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(Command_Engine, "show_status")
        patcher.start()
        self.addCleanup(patcher.stop)

    def setup_gesture(self, case, redo=False):
        """A Viewer with an earlier step in its history and a gesture of `case` open (the button down)."""
        viewer = self.make_viewer(selected=[1, 2])
        viewer.console_mode = False
        viewer._refresh_after_history = mock.Mock()
        viewer._save_state()                          # an earlier step: node 3 was here
        viewer.pos[3, 0] = 400.0                      # ... and has moved since, which only that step undoes
        if redo:
            viewer._save_state()
            viewer.pos[3, 0] = 500.0
            viewer.undo_by_hand()                     # a real step to redo, and node 3 back at 400
        self.earlier = list(viewer.position_history)
        if case == "drag":
            self.press(viewer, *NODE[1])
            self.move(viewer, 70, 20)
        elif case == "turn":                          # Ctrl + right-drag: turns the selection about the press
            self.press(viewer, 42.5, 1.5, modifiers=CTRL)
            self.move(viewer, 2.5, 1.5)
            self.move(viewer, 42.5, 41.5)
            self.assertFalse(np.array_equal(viewer.pos[1, :2], NODE[1]))
        else:
            self.press(viewer, 30, -10)
            self.move(viewer, 90, 10)
        self.assertEqual(case == "box", viewer.is_box_selecting)
        self.assertEqual(case != "box", viewer.is_multi_dragging)
        return viewer

    @staticmethod
    def ctrl_key(name, modifiers=CTRL):
        return SimpleNamespace(key=SimpleNamespace(name=name), text="", modifiers=list(modifiers), handled=False)

    def ctrl_z(self, viewer):
        ev = self.ctrl_key("Z")
        viewer.on_key_press(ev)
        return ev

    @staticmethod
    def state(viewer):
        return {"pos": viewer.pos.copy(), "selection": list(viewer.selected_indices),
                "epoch": viewer.selection_epoch, "dragging": viewer.is_multi_dragging,
                "box": viewer.is_box_selecting, "box_visible": viewer.selection_box.visible,
                "cancelled": getattr(viewer, "_drag_cancelled", False),
                "history": [id(step) for step in viewer.position_history],
                "redo": [id(step) for step in viewer.redo_stack]}

    def assert_same_state(self, one, other):
        for name in one:
            if name == "pos":
                np.testing.assert_array_equal(one[name], other[name])
            else:
                self.assertEqual(one[name], other[name], name)

    def cancelled_by(self, case, how):
        """The Viewer, and its state after the cancel, after a later move, and after the release."""
        viewer = self.setup_gesture(case)
        if how == "esc":
            handled = self.esc(viewer)
        else:
            handled = self.ctrl_z(viewer)
        self.assertTrue(handled.handled)
        states = [self.state(viewer)]
        self.move(viewer, 120, 50)
        states.append(self.state(viewer))
        self.release(viewer, 120, 50)
        states.append(self.state(viewer))
        return viewer, states

    @staticmethod
    def shape(states):
        """A state with the history as lengths, so that two Viewers' steps compare."""
        out = []
        for state in states:
            state = dict(state)
            state["history"], state["redo"] = len(state["history"]), len(state["redo"])
            out.append(state)
        return out

    def test_ctrl_z_leaves_the_state_esc_leaves_and_the_rest_of_the_drag_does_nothing(self):
        for case in self.CASES:
            with self.subTest(case=case):
                _, by_esc = self.cancelled_by(case, "esc")
                viewer, by_undo = self.cancelled_by(case, "undo")
                for after_esc, after_undo in zip(self.shape(by_esc), self.shape(by_undo)):
                    self.assert_same_state(after_esc, after_undo)
                cancelled, moved, released = by_undo
                self.assertFalse(cancelled["dragging"] or cancelled["box"])
                self.assertEqual(cancelled["selection"], [1, 2])
                self.assertEqual(len(cancelled["history"]), 1)       # the earlier step, none of the press's
                self.assertEqual(len(cancelled["redo"]), 0)          # and nothing was undone
                np.testing.assert_array_equal(moved["pos"], cancelled["pos"])    # a later move moves no node
                self.assertEqual(released["history"], cancelled["history"])      # the release adds no step
                self.assertEqual(released["selection"], [1, 2])

    def undone_viewer(self):
        """A Viewer with no gesture open and one step undone (node 3 back at 400, 500 waits in the redo stack)."""
        viewer = self.make_viewer(selected=[1, 2])
        viewer.console_mode = False
        viewer._refresh_after_history = mock.Mock()
        viewer._save_state()
        viewer.pos[3, 0] = 400.0
        viewer._save_state()
        viewer.pos[3, 0] = 500.0
        viewer.undo_by_hand()
        return viewer

    def test_shift_z_cancels_a_gesture_and_redoes_outside_one(self):
        """Shift is the 8-direction lock in a gesture, so there Ctrl/Cmd+Shift+Z cancels as Ctrl+Z does; elsewhere it redoes."""
        for modifier in ("Control", "Meta"):
            for case in self.CASES:
                with self.subTest(modifier=modifier, case=case):
                    _, by_esc = self.cancelled_by(case, "esc")
                    viewer = self.setup_gesture(case)
                    viewer.redo_by_hand = mock.Mock()
                    ev = self.ctrl_key("Z", [modifier, "Shift"])
                    viewer.on_key_press(ev)
                    self.assertTrue(ev.handled)
                    viewer.redo_by_hand.assert_not_called()
                    after = self.shape([self.state(viewer)])[0]
                    self.assert_same_state(self.shape(by_esc)[0], after)
            with self.subTest(modifier=modifier, case="none"):
                viewer = self.undone_viewer()
                self.assertEqual(len(viewer.redo_stack), 1)
                viewer.on_key_press(self.ctrl_key("Z", [modifier, "Shift"]))
                self.assertEqual(len(viewer.redo_stack), 0)
                self.assertEqual(viewer.pos[3, 0], 500.0)
            with self.subTest(modifier=modifier, case="plain z"):
                viewer = self.undone_viewer()
                viewer.redo_by_hand = mock.Mock()
                viewer.on_key_press(self.ctrl_key("Z", [modifier]))
                viewer.redo_by_hand.assert_not_called()

    def test_ctrl_z_puts_the_nodes_back(self):
        for case in ("drag", "turn"):
            with self.subTest(case=case):
                viewer = self.setup_gesture(case)
                self.ctrl_z(viewer)
                self.assert_places(viewer, [1, 2], [NODE[1], NODE[2]], atol=0)
                self.assertEqual(viewer.pos[3, 0], 400.0)

    def test_the_step_before_is_undone_by_the_next_ctrl_z_while_the_button_is_down_or_after_the_release(self):
        for case in self.CASES:
            for released in (False, True):
                with self.subTest(case=case, released=released):
                    viewer = self.setup_gesture(case)
                    self.ctrl_z(viewer)
                    if released:
                        self.release(viewer, 90, 10)
                    self.ctrl_z(viewer)
                    self.assertEqual(viewer.pos[3, 0], 120.0)         # the earlier step is undone
                    self.assertEqual(viewer.position_history, [])
                    self.assertEqual(len(viewer.redo_stack), 1)
                    self.assertFalse(viewer.is_multi_dragging or viewer.is_box_selecting)

    def test_the_undo_button_and_undo_by_hand_cancel_the_drag_too(self):
        for case in self.CASES:
            with self.subTest(case=case):
                viewer = self.setup_gesture(case)
                self.assertTrue(viewer.undo_by_hand())
                self.assertFalse(viewer.is_multi_dragging or viewer.is_box_selecting)
                self.assertEqual(len(viewer.position_history), 1)
                self.assertEqual(viewer.redo_stack, [])

    def test_a_command_that_waited_for_the_drag_goes_on_and_is_not_taken_back(self):
        for case in self.CASES:
            with self.subTest(case=case):
                viewer = self.setup_gesture(case)
                queue = mock.Mock()
                queue.pending.return_value = ["a waiting command"]
                viewer.command_queue = queue
                self.assertTrue(viewer.undo_by_hand())
                queue.resume.assert_called_once_with()
                queue.take_back.assert_not_called()

    def test_ctrl_z_with_no_drag_still_takes_back_a_waiting_command_or_undoes(self):
        viewer = self.make_viewer(selected=[1, 2])
        viewer.console_mode = False
        viewer._refresh_after_history = mock.Mock()
        viewer._save_state()
        queue = mock.Mock()
        queue.pending.return_value = ["a waiting command"]
        queue.take_back.return_value = (1, "taken back")
        viewer.command_queue = queue
        self.assertTrue(self.ctrl_z(viewer).handled)
        queue.take_back.assert_called_once_with(1)
        self.assertEqual(len(viewer.position_history), 1)
        queue.pending.return_value = []
        self.ctrl_z(viewer)
        self.assertEqual(viewer.position_history, [])

    def test_a_wheel_turn_is_still_its_own_step_that_ctrl_z_takes_back(self):
        viewer = self.make_viewer(selected=[1, 2])
        viewer.console_mode = False
        viewer._refresh_after_history = mock.Mock()
        viewer._save_state()
        self.wheel(viewer, (0.0, 1.0), 60.0, 30.0, buttons=(), modifiers=CTRL)
        self.assertIsNotNone(viewer._wheel_turn)
        self.ctrl_z(viewer)
        self.assertIsNone(viewer._wheel_turn)
        self.assertEqual(len(viewer.position_history), 1)
        self.assertEqual(len(viewer.redo_stack), 1)
        self.assert_places(viewer, [1, 2], [NODE[1], NODE[2]])

    def test_a_wheel_turn_cannot_be_open_with_a_drag(self):
        # A button press ends the turn, and a Ctrl + wheel notch with a button held turns nothing,
        # so the drag branch of undo_by_hand never has a turn to deal with.
        viewer = self.setup_gesture("drag")
        self.assertIsNone(viewer._wheel_turn)
        self.wheel(viewer, (0.0, 1.0), 60.0, 30.0, buttons=(2,), modifiers=CTRL)
        self.assertIsNone(viewer._wheel_turn)

    def test_ctrl_y_during_a_node_drag_changes_nothing(self):
        # The press emptied the redo steps, so there is nothing for Ctrl+Y to bring back.
        for case in ("drag", "turn"):
            with self.subTest(case=case):
                viewer = self.setup_gesture(case, redo=True)
                self.assertEqual(viewer.redo_stack, [])
                before = self.state(viewer)
                viewer.on_key_press(self.ctrl_key("Y"))
                after = self.state(viewer)
                self.assertTrue(after["dragging"])
                self.assert_same_state(before, after)
                self.assertEqual(len(after["history"]), 2)

    def test_ctrl_y_during_a_box_does_nothing_and_the_box_goes_on(self):
        # A box press leaves the redo steps in place; redoing one under the box used to change the nodes and the
        # history while the gesture was open.
        viewer = self.setup_gesture("box", redo=True)
        self.assertEqual(len(viewer.redo_stack), 1)
        before = self.state(viewer)
        ev = self.ctrl_key("Y")
        viewer.on_key_press(ev)
        self.assertTrue(ev.handled)
        self.assertFalse(viewer.redo_by_hand())
        self.assert_same_state(before, self.state(viewer))
        self.assertTrue(viewer.is_box_selecting)
        self.move(viewer, 120, 10)                             # the box goes on
        self.assertTrue(viewer.is_box_selecting and viewer.selection_box.visible)
        self.release(viewer, 120, 10)
        self.assertEqual(len(viewer.redo_stack), 1)            # and Ctrl+Y works again once it is over
        self.assertTrue(viewer.redo_by_hand())
        self.assertEqual(viewer.redo_stack, [])


class PageSelectionDuringGestureTests(GestureTestCase):
    """A web page's action (Tree View's tree_select) that arrives in the middle of a mouse gesture waits for it to end.

    Replacing the selection under a drag moved the new nodes by the old ones' offsets, left the old ones where the
    drag had put them, or raised in every mouse move when the counts differed.
    """

    def setUp(self):
        super().setUp()
        from PySide6 import QtCore
        from web_ui import tree_backend
        from web_ui.Web_Server import QtCommunicator
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.QtCore = QtCore
        self.tree_backend = tree_backend
        self.viewer = self.make_viewer(selected=[1, 2])
        self.viewer.command_queue = CommandQueue(self.viewer)
        self.addCleanup(self.viewer.command_queue.shutdown)
        self.viewer.handle_web_action = lambda data: tree_backend.handle_tree_select(self.viewer, data)
        self.communicator = QtCommunicator(self.viewer)

    def page_selects(self, indices):
        """The page's POST, as the web server hands it to the GUI thread."""
        self.communicator.handle_action({"action": "tree_select", "indices": indices, "mode": "replace"})
        self.app.processEvents()

    def settle(self, milliseconds=100):
        deadline = self.QtCore.QDeadlineTimer(milliseconds)
        while not deadline.hasExpired():
            self.app.processEvents()

    def test_a_selection_from_a_page_waits_for_a_node_drag_to_end(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1])
        self.move(viewer, 70, 20)
        self.page_selects([4, 5])
        self.move(viewer, 90, 30)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assert_places(viewer, [1, 2], [(90, 30), (130, 30)])
        self.assert_places(viewer, [4, 5], [NODE[4], NODE[5]])      # the nodes that were dragged moved, no others
        self.release(viewer, 90, 30)
        self.settle()
        self.assertEqual(viewer.selected_indices, [4, 5])
        self.assert_places(viewer, [4, 5], [NODE[4], NODE[5]])

    def test_a_selection_of_another_size_from_a_page_does_not_break_the_drag(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1])
        self.move(viewer, 70, 20)
        self.page_selects([4])
        self.move(viewer, 90, 30)                                    # raised ValueError when the sizes differed
        self.assertTrue(viewer.is_multi_dragging)
        self.release(viewer, 90, 30)
        self.settle()
        self.assertEqual(viewer.selected_indices, [4])

    def test_a_selection_from_a_page_waits_for_a_box_to_end(self):
        viewer = self.viewer
        self.press(viewer, 300, 300)
        self.move(viewer, 330, 330)
        self.page_selects([4, 5])
        self.assertTrue(viewer.is_box_selecting)
        self.assertEqual(viewer.selected_indices, [])
        self.release(viewer, 330, 330)
        self.settle()
        self.assertEqual(viewer.selected_indices, [4, 5])

    def test_a_selection_from_a_page_waits_for_a_ctrl_wheel_turn_to_end(self):
        viewer = self.viewer
        self.wheel(viewer, (0.0, 1.0), 60.0, 30.0, buttons=(), modifiers=CTRL)
        self.assertIsNotNone(viewer._wheel_turn)
        self.page_selects([4, 5])
        self.assertEqual(viewer.selected_indices, [1, 2])
        viewer._wheel_turn_timer.fire()
        self.settle()
        self.assertEqual(viewer.selected_indices, [4, 5])

    def test_with_no_gesture_open_the_selection_applies_at_once(self):
        self.page_selects([4, 5])
        self.assertEqual(self.viewer.selected_indices, [4, 5])

    def test_a_handler_that_raises_costs_only_its_own_action(self):
        viewer = self.viewer
        delivered = []

        def handle(data):
            if data["name"] == "bad":
                raise RuntimeError("handler broke")
            delivered.append(data["name"])

        viewer.handle_web_action = handle
        reported = []
        with mock.patch.object(sys, "excepthook", lambda *info: reported.append(info[1])):
            self.press(viewer, *NODE[1])                  # a gesture is open: both actions are held
            for name in ("bad", "second"):
                self.communicator.handle_action({"name": name})
            self.app.processEvents()
            self.assertEqual(delivered, [])
            self.release(viewer, *NODE[1])
            self.settle()
            self.assertEqual(delivered, ["second"])
            self.assertEqual([str(error) for error in reported], ["handler broke"])   # still reported
            self.communicator.handle_action({"name": "third"})   # no gesture, nothing held: at once
            self.app.processEvents()
            self.assertEqual(delivered, ["second", "third"])

    def test_a_handler_that_raises_with_no_gesture_open_is_reported_as_before(self):
        reported = []

        def handle(data):
            raise RuntimeError("direct")

        self.viewer.handle_web_action = handle
        with mock.patch.object(sys, "excepthook", lambda *info: reported.append(info[1])):
            self.communicator.handle_action({"name": "x"})
            self.app.processEvents()
        self.assertEqual([str(error) for error in reported], ["direct"])
        self.assertEqual(len(self.communicator._held), 0)

    def test_actions_that_waited_run_in_the_order_they_came(self):
        viewer = self.viewer
        self.press(viewer, *NODE[1])
        self.page_selects([3])
        self.page_selects([4, 5])
        self.release(viewer, *NODE[1])
        self.settle()
        self.assertEqual(viewer.selected_indices, [4, 5])


class PageBoxGestureTests(GestureTestCase):
    """A box that a web page draws (Tree View's) makes the selection follow it, and the Viewer treats it as it treats its own.

    The messages go the way the web server sends them: through the QtCommunicator, which answers before the Viewer
    runs the action, so the Viewer is the one that skips an update that a newer one has overtaken.
    """

    def setUp(self):
        super().setUp()
        from PySide6 import QtCore
        from web_ui import tree_backend
        from web_ui.Web_Server import QtCommunicator
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.QtCore = QtCore
        self.tree_backend = tree_backend
        self.viewer = self.make_viewer(selected=[1, 2])
        self.viewer.command_queue = CommandQueue(self.viewer)
        self.addCleanup(self.viewer.command_queue.shutdown)
        self.viewer.n_nodes = len(self.viewer.pos)
        self.viewer._show_selection_status = mock.Mock()
        self.viewer.handle_web_action = lambda data: tree_backend.handle_tree_select(self.viewer, data)
        self.communicator = QtCommunicator(self.viewer)

    def settle(self, milliseconds=100):
        deadline = self.QtCore.QDeadlineTimer(milliseconds)
        while not deadline.hasExpired():
            self.app.processEvents()

    def post(self, phase, nodes=None, seq=0, mode="replace", gesture_id="g1", wait=True):
        """The page's POST, as the web server hands it to the GUI thread."""
        import base64
        data = {"action": "tree_select", "gesture": {"id": gesture_id, "phase": phase, "seq": seq}}
        if nodes is not None:
            raw = bytearray((self.viewer.n_nodes + 7) // 8)
            for node in nodes:
                raw[node >> 3] |= 1 << (node & 7)
            data["nodes"] = base64.b64encode(bytes(raw)).decode("ascii")
            data["mode"] = mode
        # From another thread, as the web server's: the signal is then queued to the GUI thread.
        thread = threading.Thread(target=self.communicator.handle_action, args=(data,))
        thread.start()
        thread.join()
        if wait:
            self.app.processEvents()

    def test_the_selection_follows_the_box_and_it_counts_as_a_gesture_while_it_is_open(self):
        from Command_Queue import gesture_open
        viewer = self.viewer
        self.post("update", [3, 4], seq=1)
        self.assertEqual(viewer.selected_indices, [3, 4])
        self.assertTrue(gesture_open(viewer))
        self.assertFalse(viewer.is_box_selecting)
        self.post("update", [3], seq=2)
        self.assertEqual(viewer.selected_indices, [3])
        self.post("end", [3], seq=3)
        self.assertEqual(viewer.selected_indices, [3])
        self.assertFalse(gesture_open(viewer))
        viewer._show_selection_status.assert_called_once_with()

    def test_the_updates_run_in_one_go_are_one_the_newest(self):
        viewer = self.viewer
        calls = []
        original = viewer.update_selection_visual
        viewer.update_selection_visual = lambda broadcast=True: (calls.append(list(viewer.selected_indices)), original(broadcast))[1]
        for seq, nodes in enumerate(([3], [3, 4], [4, 5], [5]), start=1):
            self.post("update", nodes, seq=seq, wait=False)
        self.app.processEvents()
        self.assertEqual(calls, [[5]])                         # the older three were skipped, never run
        self.post("end", [5], seq=5)
        self.assertEqual(viewer.selected_indices, [5])

    def test_the_end_after_updates_that_are_still_waiting_leaves_the_final_set_whatever_arrived(self):
        viewer = self.viewer
        self.post("update", [3], seq=1, wait=False)
        self.post("update", [3, 4], seq=2, wait=False)
        self.post("end", [4, 5], seq=3, wait=False)           # the end overtakes the updates that were waiting
        self.app.processEvents()
        self.assertEqual(sorted(viewer.selected_indices), [4, 5])
        self.assertEqual(viewer.selection_epoch, 1)
        self.post("update", [0], seq=4)                       # a late one
        self.post("end", [0], seq=5)
        self.assertEqual(sorted(viewer.selected_indices), [4, 5])

    def test_esc_in_the_viewer_puts_the_selection_back_and_the_pages_later_messages_change_nothing(self):
        viewer = self.viewer
        self.post("update", [4, 5], seq=1)
        viewer.broadcast_event.reset_mock()
        self.assertTrue(self.esc(viewer).handled)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertEqual(viewer.selection_epoch, 0)
        self.assertFalse(viewer.outside_selection_gesture.is_open)
        self.assertEqual(viewer.broadcast_event.call_args.args[0]["gesture"], {"id": "g1", "final": True})
        self.post("update", [0], seq=2)
        self.post("end", [0], seq=3)
        self.assertEqual(viewer.selected_indices, [1, 2])

    def test_ctrl_z_cancels_the_gesture_as_it_cancels_a_box_and_redo_does_nothing_meanwhile(self):
        viewer = self.viewer
        viewer.console_mode = False
        with mock.patch.object(Command_Engine, "show_status"):
            self.post("update", [4, 5], seq=1)
            self.assertFalse(viewer.redo_by_hand())
            self.assertEqual(viewer.selected_indices, [4, 5])
            self.assertTrue(viewer.undo_by_hand())
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertFalse(viewer.outside_selection_gesture.is_open)

    def test_a_press_on_the_canvas_ends_the_gesture_by_putting_the_selection_back(self):
        viewer = self.viewer
        self.post("update", [4, 5], seq=1)
        self.press(viewer, 30, -10)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertFalse(viewer.outside_selection_gesture.is_open)
        self.release(viewer, 30, -10)

    def test_a_plain_page_action_and_a_command_wait_for_the_gesture_as_they_wait_for_a_box(self):
        viewer = self.viewer
        self.post("update", [4], seq=1)
        self.communicator.handle_action({"action": "tree_select", "indices": [0], "mode": "replace"})
        self.settle(60)
        self.assertEqual(viewer.selected_indices, [4])         # held
        self.post("update", [5], seq=2)                        # the gesture's own messages are not held
        self.assertEqual(viewer.selected_indices, [5])
        self.post("end", [5], seq=3)
        self.settle(100)
        self.assertEqual(viewer.selected_indices, [0])         # the held action ran after the end, in its turn

    def test_a_page_that_goes_away_in_the_middle_of_a_drag_is_taken_as_a_cancel_after_a_few_seconds(self):
        from desktop.Viewer_Selection_Gesture import SelectionGesture
        viewer = self.viewer
        viewer.outside_selection_gesture = SelectionGesture(viewer, silence_ms=60)
        self.post("update", [4, 5], seq=1)
        self.assertEqual(viewer.selected_indices, [4, 5])
        self.settle(30)
        self.post("alive")                                      # the page is still there: the wait starts again
        self.settle(40)
        self.assertEqual(viewer.selected_indices, [4, 5])
        self.settle(150)
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertFalse(viewer.outside_selection_gesture.is_open)
        self.assertEqual(viewer.selection_epoch, 0)

    def test_the_selection_the_gesture_started_with_is_the_one_when_its_first_update_ran(self):
        viewer = self.viewer
        self.post("update", [4], seq=1, mode="toggle")
        self.assertEqual(sorted(viewer.selected_indices), [1, 2, 4])
        self.post("update", [1, 4], seq=2, mode="toggle")
        self.assertEqual(sorted(viewer.selected_indices), [2, 4])
        self.post("end", [1, 4], seq=3, mode="toggle")
        self.assertEqual(sorted(viewer.selected_indices), [2, 4])


if __name__ == "__main__":
    unittest.main()
