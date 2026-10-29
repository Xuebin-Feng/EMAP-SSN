# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's Undo and Redo buttons (desktop/Viewer_History.py and MainViewer's wiring of them).

Two icon buttons sit in the canvas's top-right row: at its right end, with the
frame rate (when shown) to their left and the sidebar's toggle (when shown) to
their right, the jobs badge under the row, and the command line short of all
of them. A click is Ctrl+Z or Ctrl+Y; a button is enabled while its key could
do something, and the Viewer refreshes that wherever the history or the
command queue changes. The buttons take no keyboard focus, never move on a
state change (Qt draws the whole network again whenever a widget over the canvas
moves or resizes, as test_viewer_caret describes), and are never in a picture.
The GL test counts the scene draws and takes a picture on a real canvas.
"""
import ast
import os
import random
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(PROJECT_ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from Command_Queue import CommandQueue, ManualExecutor  # noqa: E402
from EMAPSSN_Viewer import CLICK_STATUS_SEPARATOR, COMMAND_LINE_MIN_WIDTH, MainViewer  # noqa: E402
from PySide6.QtCore import QEvent, QObject, QRect, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402
from desktop.Desktop_App import PSEUDO_LANGUAGE, install_translations  # noqa: E402
from desktop.Viewer_Caret import CommandCaret  # noqa: E402
from desktop.Viewer_Fps import FPS_GAP  # noqa: E402
from desktop.Viewer_History import HISTORY_BUTTON_SIZE, HistoryButtons, history_availability  # noqa: E402
from desktop.Viewer_Jobs import BADGE_GAP  # noqa: E402
from tests.test_viewer_undo_redo import make_viewer, quietly  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402
from vispy.util.keys import Key  # noqa: E402

SIZES = ((1400, 900), (563, 300))   # a roomy window, and the Viewer's smallest


class FakeHeavyCommand:
    """A command module whose compute step runs on the queue's worker (a ManualExecutor here) and changes nothing."""

    @staticmethod
    def plan(viewer, args):
        return SimpleNamespace(heavy=True)

    @staticmethod
    def start(viewer, plan):
        return {}

    @staticmethod
    def compute(inputs, progress):
        return 1

    @staticmethod
    def apply(viewer, plan, result):
        pass


class ViewerCase(unittest.TestCase):
    """A Viewer built as the Viewer builds its top-right row, around a stand-in canvas, with a real command queue."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def build(self, size=(1400, 900), *, fps=False, toggle=False):
        viewer = make_viewer()
        viewer.canvas = SimpleNamespace(native=QWidget(), size=size, update=mock.Mock())
        viewer.update_console_background = mock.Mock()
        viewer._update_hud_elements = lambda: None
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer.command_queue = CommandQueue(viewer, executor=ManualExecutor())
        self.addCleanup(viewer.command_queue.shutdown)
        viewer._build_history_buttons()
        with mock.patch.object(cfg, "SHOW_FPS", fps):
            viewer._build_fps_overlay()
        viewer._build_jobs_badge()
        with mock.patch.object(EMAPSSN_Viewer, "SIDEBAR_ENABLED", toggle):
            if toggle:
                viewer.add_sidebar_button("metaDataBtn", "Meta Data", lambda: None, icon="chart-column")
            viewer.set_sidebar_visible(False, animate=False)
        viewer.main_window.resize(*size)
        viewer.main_window.show()
        self.app.processEvents()
        native = viewer.canvas.native
        viewer.canvas.size = (native.width(), native.height())   # the window's size, as the canvas reports it
        viewer.reposition_expand_btn()
        self.app.processEvents()
        return viewer

    def click(self, button):
        """A real mouse click on the button, with the console line's print kept off the test log."""
        quietly(lambda: QTest.mouseClick(button, Qt.MouseButton.LeftButton))
        self.app.processEvents()

    @staticmethod
    def state(viewer):
        """(Undo enabled, Redo enabled)."""
        buttons = viewer.history_buttons
        return buttons.undo.isEnabled(), buttons.redo.isEnabled()

    def move_node(self, viewer, x):
        """What a move of nodes does: the press saves the state, the drag changes it."""
        viewer._save_state()
        viewer.pos[0] = [x, x]

    def give_command(self, viewer, line="fake"):
        """Give the queue a heavy command: it computes (or waits behind one that does) until its executor finishes it."""
        return quietly(lambda: viewer.command_queue.dispatch(FakeHeavyCommand, line, [], line))

    def finish_command(self, viewer, timeout=10):
        """Let the oldest compute step run and its result apply."""
        job = quietly(lambda: viewer.command_queue.executor.finish())
        deadline = time.monotonic() + timeout
        while not job.finished and time.monotonic() < deadline:
            quietly(self.app.processEvents)
            time.sleep(0.002)
        self.assertTrue(job.finished)


class PlacementTests(ViewerCase):
    """The row: the card, then Undo, then Redo, then the toggle; the badge under it; the command line short of all."""

    def row(self, viewer):
        buttons = viewer.history_buttons
        return buttons.undo.geometry(), buttons.redo.geometry()

    def check(self, viewer, *, fps, toggle):
        width, height = viewer.canvas.size
        undo, redo = self.row(viewer)
        toggle_button = viewer.toggle_sidebar_btn
        self.assertEqual(not toggle_button.isHidden(), toggle)
        # The pair keeps the row's right end: the corner, or FPS_GAP left of the toggle.
        row_right = toggle_button.x() - FPS_GAP if toggle else width - 10
        self.assertEqual(redo.x() + redo.width(), row_right)
        self.assertEqual(redo.x() - (undo.x() + undo.width()), 4)       # a small gap between the two
        for rectangle in (undo, redo):
            # As big as the sidebar's toggle, in the row it makes.
            self.assertEqual((rectangle.width(), rectangle.height()), (30, 30))
        # In the row of the toggle and the frame rate: the same top, the same height, the same middle.
        self.assertEqual((undo.y(), redo.y()), (10, 10))
        self.assertEqual(undo.y() * 2 + undo.height(), 50)
        occupied = {"undo": undo, "redo": redo}
        card_left = undo.x()
        if fps:
            card = viewer.fps_overlay.geometry()
            self.assertEqual(card.x() + card.width() + FPS_GAP, undo.x())
            self.assertEqual(2 * card.y() + card.height(), 2 * undo.y() + undo.height())
            occupied["card"] = card
            card_left = card.x()
            self.assertLessEqual(viewer._command_line_right(),
                                 card.x() + card.width() - viewer.fps_overlay.widest_width() - FPS_GAP)
        else:
            self.assertIsNone(viewer.fps_overlay)
            # The command line stops FPS_GAP short of the pair.
            self.assertEqual(viewer._command_line_right(), undo.x() - FPS_GAP)
        if toggle:
            occupied["toggle"] = toggle_button.geometry()
        badge = viewer.jobs_badge
        badge.show_jobs(["#1 running: a file", "#2 waiting: a fold"])
        # Under the row, right-aligned with it.
        self.assertEqual(badge.x() + badge.width(), redo.x() + redo.width())
        self.assertGreaterEqual(badge.y(), redo.y() + redo.height() + BADGE_GAP)
        occupied["badge"] = badge.geometry()
        # Nothing overlaps, and nothing is cut off by the canvas's edge.
        names = list(occupied)
        for index, first in enumerate(names):
            self.assertTrue(QRect(0, 0, width, height).contains(occupied[first]), first)
            for second in names[index + 1:]:
                self.assertFalse(occupied[first].intersects(occupied[second]), (first, second))
        self.assertLessEqual(viewer._command_line_right(), card_left - FPS_GAP)
        badge.show_jobs([])

    def test_the_pair_takes_the_rows_right_end_whatever_else_shows(self):
        for width, height in SIZES:
            for fps in (False, True):
                for toggle in (False, True):
                    with self.subTest(size=(width, height), fps=fps, toggle=toggle):
                        viewer = self.build((width, height), fps=fps, toggle=toggle)
                        self.assertEqual(viewer.canvas.size, (width, height))
                        self.check(viewer, fps=fps, toggle=toggle)

    def test_a_longer_frame_rate_text_keeps_clear_of_the_pair(self):
        for width, height in SIZES:
            with self.subTest(size=(width, height)):
                viewer = self.build((width, height), fps=True)
                # The longest text it shows at this size: the rate alone in the smallest window.
                viewer.fps_overlay.show_text(viewer.fps_overlay.text_for(123.0, 45.6, 64))
                self.check(viewer, fps=True, toggle=False)
                undo, _redo = self.row(viewer)
                card = viewer.fps_overlay.geometry()
                self.assertEqual(card.x() + card.width() + FPS_GAP, undo.x())

    def test_the_command_line_never_overlaps_the_row_even_when_it_is_long(self):
        # The box runs from its left edge to _command_line_right(); at the smallest window it keeps a usable width.
        for width, height in SIZES:
            for fps in (False, True):
                with self.subTest(size=(width, height), fps=fps):
                    viewer = self.build((width, height), fps=fps)
                    undo, _redo = self.row(viewer)
                    left = viewer.hud_layout["console_bg_left_offset"] if hasattr(viewer, "hud_layout") else 20.0
                    right = viewer._command_line_right()
                    self.assertGreaterEqual(right - left, COMMAND_LINE_MIN_WIDTH)
                    self.assertLessEqual(right, undo.x() - FPS_GAP)

    def test_a_narrow_window_shows_the_rate_alone_and_keeps_the_command_line_wide(self):
        # The full text's room would leave the command line ~110 px in the smallest window; the rate alone leaves it more.
        viewer = self.build((1400, 900), fps=True)
        overlay = viewer.fps_overlay
        overlay.frames.gpu_ms, overlay.frames.detail = 2.5, 16
        left = viewer.hud_layout["console_bg_left_offset"] if hasattr(viewer, "hud_layout") else 20.0
        with mock.patch.object(overlay.frames, "rate", return_value=40.0):
            for size, rate_only in (((1400, 900), False), ((563, 300), True), ((1400, 900), False)):
                with self.subTest(size=size):
                    viewer.main_window.resize(*size)
                    self.app.processEvents()
                    native = viewer.canvas.native
                    viewer.canvas.size = (native.width(), native.height())
                    viewer.reposition_expand_btn()
                    overlay.refresh()                 # its next tick
                    self.assertEqual(overlay.rate_only, rate_only)
                    self.assertEqual(overlay.text(), "40 fps" if rate_only else "40 fps · GPU 2.5 ms · edges 1/16")
                    self.assertGreaterEqual(viewer._command_line_right() - left, COMMAND_LINE_MIN_WIDTH)
                    self.check(viewer, fps=True, toggle=False)

    def test_it_keeps_its_place_as_the_window_is_resized(self):
        viewer = self.build((1400, 900), fps=True)
        for size in ((900, 600), (563, 300), (1400, 900)):
            viewer.main_window.resize(*size)
            self.app.processEvents()
            native = viewer.canvas.native
            viewer.canvas.size = (native.width(), native.height())
            viewer.reposition_expand_btn()
            self.check(viewer, fps=True, toggle=False)

    def test_the_pair_and_the_card_follow_the_sidebar_open_and_closed(self):
        with mock.patch.object(cfg, "LOW_RESOURCE_ANIMATIONS", True):    # no slide: it opens at once
            viewer = self.build(fps=True, toggle=True)
            with mock.patch.object(EMAPSSN_Viewer, "SIDEBAR_ENABLED", True):
                for visible in (True, False):
                    viewer.set_sidebar_visible(visible)
                    self.app.processEvents()
                    viewer.canvas.size = (viewer.canvas.native.width(), viewer.canvas.native.height())
                    undo, redo = self.row(viewer)
                    toggle = viewer.toggle_sidebar_btn.geometry()
                    room = viewer._panel_w if visible else 0
                    self.assertEqual(toggle.x() + toggle.width() + 10, viewer.canvas.size[0] - room)
                    self.assertEqual(redo.x() + redo.width() + FPS_GAP, toggle.x())
                    self.assertEqual(viewer.fps_overlay.geometry().right() + FPS_GAP + 1, undo.x())

    def test_the_pair_is_placed_as_soon_as_it_is_built(self):
        viewer = make_viewer()
        viewer.canvas = SimpleNamespace(native=QWidget(), size=(900, 600), update=mock.Mock())
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        viewer.command_queue = CommandQueue(viewer, executor=ManualExecutor())
        self.addCleanup(viewer.command_queue.shutdown)
        viewer._build_history_buttons()
        undo, redo = self.row(viewer)
        self.assertEqual((redo.x() + redo.width(), redo.y()), (900 - 10, 10))
        self.assertEqual((undo.y(), undo.width()), (10, 30))

    def test_a_viewer_without_the_buttons_places_the_card_and_badge_as_before(self):
        viewer = self.build(fps=True)
        del viewer.history_buttons
        viewer.reposition_expand_btn()
        card = viewer.fps_overlay.geometry()
        self.assertEqual(card.x() + card.width(), viewer.canvas.size[0] - 10)
        self.assertEqual(viewer._command_line_right(), card.x() + card.width() - viewer.fps_overlay.widest_width() - FPS_GAP)


class EnabledStateTests(ViewerCase):
    """Undo is enabled while Ctrl+Z could act, Redo while Ctrl+Y could; both follow the history and the queue."""

    def test_both_are_dimmed_in_a_new_viewer(self):
        self.assertEqual(self.state(self.build()), (False, False))

    def test_a_move_then_undo_to_the_bottom_then_redo_to_the_top(self):
        viewer = self.build()
        self.move_node(viewer, 7.0)
        self.assertEqual(self.state(viewer), (True, False))
        self.click(viewer.history_buttons.undo)
        np.testing.assert_array_equal(viewer.pos[0], [0.0, 0.0])                       # the node is back
        self.assertEqual(self.state(viewer), (False, True))                             # the bottom
        self.assertEqual(viewer.console_text.text, "Undo successful.")
        self.click(viewer.history_buttons.redo)
        np.testing.assert_array_equal(viewer.pos[0], [7.0, 7.0])                       # moved again
        self.assertEqual(self.state(viewer), (True, False))                             # the top
        self.assertEqual(viewer.console_text.text, "Redo successful.")

    def test_a_walk_through_several_steps(self):
        viewer = self.build()
        for x in (1.0, 2.0, 3.0):
            self.move_node(viewer, x)
        self.assertEqual(self.state(viewer), (True, False))
        self.click(viewer.history_buttons.undo)
        self.click(viewer.history_buttons.undo)
        np.testing.assert_array_equal(viewer.pos[0], [1.0, 1.0])
        self.assertEqual(self.state(viewer), (True, True))                              # in the middle
        self.click(viewer.history_buttons.redo)
        self.assertEqual(self.state(viewer), (True, True))
        self.click(viewer.history_buttons.undo)
        self.click(viewer.history_buttons.undo)
        self.assertEqual(self.state(viewer), (False, True))
        self.assertEqual(len(viewer.redo_stack), 3)

    def test_a_new_action_clears_the_redo_steps_and_dims_redo(self):
        viewer = self.build()
        self.move_node(viewer, 1.0)
        self.move_node(viewer, 2.0)
        self.click(viewer.history_buttons.undo)
        self.assertEqual(self.state(viewer), (True, True))
        self.move_node(viewer, 5.0)                                                     # the new action
        self.assertEqual(viewer.redo_stack, [])
        self.assertEqual(self.state(viewer), (True, False))

    def test_a_metadata_edit_is_an_undo_step_too(self):
        viewer = self.build()
        viewer._save_metadata_cell_history("Length", 0, 10, 99)
        self.assertEqual(self.state(viewer), (True, False))

    def test_the_history_limit_does_not_dim_undo(self):
        viewer = self.build()
        for index in range(60):
            viewer._save_metadata_cell_history("Length", 0, index, index + 1)
        self.assertEqual(len(viewer.position_history), 50)
        self.assertEqual(self.state(viewer), (True, False))

    def test_typed_undo_and_redo_steps_follow_too(self):
        # `undo N` and `redo N` take their steps with announce False: the buttons still follow each step.
        viewer = self.build()
        for x in (1.0, 2.0):
            self.move_node(viewer, x)
        quietly(lambda: viewer._do_undo(announce=False))
        self.assertEqual(self.state(viewer), (True, True))
        quietly(lambda: viewer._do_undo(announce=False))
        self.assertEqual(self.state(viewer), (False, True))
        quietly(lambda: viewer._do_redo(announce=False))
        quietly(lambda: viewer._do_redo(announce=False))
        self.assertEqual(self.state(viewer), (True, False))

    def test_cancelling_a_drag_gives_the_redo_steps_back(self):
        # A press saves the state and clears redo; Esc takes the press's step off and gives redo back.
        viewer = self.build()
        self.move_node(viewer, 1.0)
        self.click(viewer.history_buttons.undo)
        self.assertEqual(self.state(viewer), (False, True))
        history = viewer.position_history
        kept = (list(history), list(viewer.redo_stack))
        viewer._save_state()
        viewer._press_history = (kept, history[-1])
        self.assertEqual(self.state(viewer), (True, False))
        viewer._take_back_press_history()
        self.assertEqual(self.state(viewer), (False, True))

    def test_a_waiting_command_enables_undo_and_dims_redo(self):
        viewer = self.build()
        self.move_node(viewer, 1.0)
        self.click(viewer.history_buttons.undo)
        self.assertEqual(self.state(viewer), (False, True))                              # nothing to undo, one to redo
        self.give_command(viewer)
        self.assertEqual(len(viewer.command_queue.pending()), 1)
        self.assertEqual(len(viewer.redo_stack), 1)
        self.assertEqual(self.state(viewer), (True, False))                              # a command runs: undo takes it back
        self.finish_command(viewer)                                                      # it finishes, saving nothing
        self.assertEqual(viewer.command_queue.pending(), [])
        self.assertEqual(self.state(viewer), (False, True))

    def test_a_click_on_undo_takes_back_the_waiting_command_not_the_history(self):
        viewer = self.build()
        self.move_node(viewer, 1.0)
        queue = viewer.command_queue
        self.give_command(viewer, "first")
        self.give_command(viewer, "second")
        self.assertEqual(len(queue.pending()), 2)
        history = list(viewer.position_history)
        self.click(viewer.history_buttons.undo)
        self.assertEqual([job.line for job in queue.pending()], ["first"])               # the newest is taken back
        self.assertTrue(viewer.console_text.text.startswith("Undo: cancelled"), viewer.console_text.text)
        self.assertIn("second", viewer.console_text.text)
        self.assertEqual(viewer.position_history, history)                               # the history is untouched
        np.testing.assert_array_equal(viewer.pos[0], [1.0, 1.0])
        self.assertEqual(self.state(viewer), (True, False))                              # one still waits
        self.click(viewer.history_buttons.undo)
        self.assertEqual(queue.pending(), [])
        self.assertEqual(self.state(viewer), (True, False))                              # the history step is left
        self.click(viewer.history_buttons.undo)
        np.testing.assert_array_equal(viewer.pos[0], [0.0, 0.0])
        self.assertEqual(self.state(viewer), (False, True))

    def test_the_state_always_says_what_the_keys_would_do(self):
        # A random walk over the actions that change the history and the queue: before each undo or redo,
        # the button is enabled exactly when the key's own method then does something.
        viewer = self.build()
        rng = random.Random(42)
        saw = set()
        for step in range(300):
            action = rng.choice(("move", "undo", "undo", "redo", "redo", "give", "finish", "cancel"))
            queue = viewer.command_queue
            if action == "move":
                self.move_node(viewer, float(step))
            elif action == "give" and len(queue.pending()) < 3:
                self.give_command(viewer)
            elif action == "finish" and queue.executor.tasks:
                self.finish_command(viewer)
            elif action == "cancel" and queue.pending():
                quietly(lambda: queue.cancel(queue.pending()[-1]))
            elif action in ("undo", "redo"):
                undo_enabled, redo_enabled = self.state(viewer)
                enabled = undo_enabled if action == "undo" else redo_enabled
                method = viewer.undo_by_hand if action == "undo" else viewer.redo_by_hand
                acted = quietly(method)
                self.assertEqual(bool(acted), enabled, (step, action, saw))
                saw.add((action, enabled))
            self.assertEqual(self.state(viewer), history_availability(viewer), (step, action))
        self.assertEqual(saw, {("undo", True), ("undo", False), ("redo", True), ("redo", False)})

    def test_the_availability_of_a_viewer_with_no_history_or_queue(self):
        self.assertEqual(history_availability(SimpleNamespace()), (False, False))
        queue = SimpleNamespace(pending=lambda: ["a command"])
        self.assertEqual(history_availability(SimpleNamespace(command_queue=queue)), (True, False))
        self.assertEqual(history_availability(
            SimpleNamespace(command_queue=queue, position_history=[], redo_stack=[{}])), (True, False))
        idle = SimpleNamespace(pending=lambda: [])
        self.assertEqual(history_availability(
            SimpleNamespace(command_queue=idle, position_history=[{}], redo_stack=[{}])), (True, True))


class ClickTests(ViewerCase):
    """A click is Ctrl+Z or Ctrl+Y: it calls undo_by_hand() or redo_by_hand(), and does what they do."""

    def test_each_button_calls_its_method_once_with_no_arguments(self):
        viewer = self.build()
        viewer.position_history.append({})
        viewer.redo_stack.append({})
        viewer._refresh_history_buttons()
        with mock.patch.object(viewer, "undo_by_hand") as undo, mock.patch.object(viewer, "redo_by_hand") as redo:
            self.click(viewer.history_buttons.undo)
            undo.assert_called_once_with()
            redo.assert_not_called()
            self.click(viewer.history_buttons.redo)
            undo.assert_called_once_with()
            redo.assert_called_once_with()

    def test_a_dimmed_button_does_nothing_when_clicked(self):
        viewer = self.build()
        self.assertEqual(self.state(viewer), (False, False))
        with mock.patch.object(viewer, "undo_by_hand") as undo, mock.patch.object(viewer, "redo_by_hand") as redo:
            self.click(viewer.history_buttons.undo)
            self.click(viewer.history_buttons.redo)
        undo.assert_not_called()
        redo.assert_not_called()

    def test_a_click_does_what_the_keys_do(self):
        # Two Viewers in the same state: one is clicked, the other is sent the key; they end as one.
        def key(viewer, name):
            viewer.console_mode = False
            viewer.on_key_press(SimpleNamespace(key=Key(name), modifiers=("Control",), text="", handled=False))

        clicked, pressed = self.build(), self.build()
        for viewer in (clicked, pressed):
            for x in (1.0, 2.0):
                self.move_node(viewer, x)
        for name, button in (("Z", "undo"), ("Z", "undo"), ("Y", "redo"), ("Z", "undo")):
            self.click(getattr(clicked.history_buttons, button))
            quietly(lambda: key(pressed, name))
            np.testing.assert_array_equal(clicked.pos, pressed.pos)
            self.assertEqual(clicked.console_text.text, pressed.console_text.text)
            self.assertEqual(len(clicked.position_history), len(pressed.position_history))
            self.assertEqual(len(clicked.redo_stack), len(pressed.redo_stack))
            self.assertEqual(self.state(clicked), self.state(pressed))
        # At the bottom the key says so; the button, dimmed, does nothing.
        self.assertEqual(self.state(clicked)[0], False)
        quietly(lambda: key(pressed, "Z"))
        self.assertEqual(pressed.console_text.text, "Nothing to undo.")

    def test_the_keys_the_tooltips_name_are_the_keys_the_handler_reads(self):
        # Ctrl+Z undoes and Ctrl+Y redoes (Cmd on macOS is Control to Qt, and Meta also counts); Ctrl+Shift+Z
        # undoes too, and nothing else redoes, so the tooltips name Ctrl+Z and Ctrl+Y alone.
        viewer = self.build()
        viewer.console_mode = False
        cases = (("Z", ("Control",), "undo_by_hand"), ("Y", ("Control",), "redo_by_hand"),
                 ("Z", ("Control", "Shift"), "undo_by_hand"), ("Z", ("Meta",), "undo_by_hand"),
                 ("Z", (), None), ("Y", ("Shift",), None))
        for name, modifiers, expected in cases:
            with self.subTest(key=name, modifiers=modifiers), \
                    mock.patch.object(viewer, "undo_by_hand") as undo, mock.patch.object(viewer, "redo_by_hand") as redo:
                viewer.on_key_press(SimpleNamespace(key=Key(name), modifiers=modifiers, text=name.lower(), handled=False))
                called = [method for method, mocked in (("undo_by_hand", undo), ("redo_by_hand", redo)) if mocked.called]
                self.assertEqual(called, [expected] if expected else [])

    def test_while_a_command_is_typed_the_message_follows_it(self):
        # The keys do nothing with the command line open, but the buttons can be clicked: the message
        # follows the command being typed, as a click on the network's does, and does not replace it.
        viewer = self.build()
        self.move_node(viewer, 1.0)
        viewer.console_mode = True
        viewer.input_buffer = "select 1-3"
        viewer.cursor_pos = len(viewer.input_buffer)
        viewer.update_console_background.reset_mock()
        self.click(viewer.history_buttons.undo)
        self.assertEqual(viewer.console_text.text, f"Cmd: select 1-3{CLICK_STATUS_SEPARATOR}Undo successful.")
        viewer.update_console_background.assert_called_once_with()
        self.assertEqual(viewer.input_buffer, "select 1-3")
        np.testing.assert_array_equal(viewer.pos[0], [0.0, 0.0])

    def test_with_the_command_line_closed_the_message_is_alone_on_the_line(self):
        viewer = self.build()
        self.move_node(viewer, 1.0)
        viewer.console_mode = False
        self.click(viewer.history_buttons.undo)
        self.assertEqual(viewer.console_text.text, "Undo successful.")
        viewer.update_console_background.assert_not_called()


class _KeyRecorder(QObject):
    """Records the keys a widget receives."""

    def __init__(self, widget):
        super().__init__(widget)
        self.keys = []
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress:
            self.keys.append(event.text())
        return False


class FocusTests(ViewerCase):
    """The buttons take no keyboard focus: the canvas keeps the keys, and the command line's caret stays lit."""

    def canvas_with_focus(self, viewer):
        native = viewer.canvas.native
        native.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        viewer.main_window.activateWindow()
        native.setFocus()
        self.app.processEvents()
        self.assertIs(QApplication.focusWidget(), native)
        return native

    def test_the_buttons_never_take_focus(self):
        viewer = self.build()
        for button in viewer.history_buttons.buttons():
            self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_a_click_leaves_the_keyboard_on_the_canvas_and_typing_goes_on(self):
        viewer = self.build()
        native = self.canvas_with_focus(viewer)
        recorder = _KeyRecorder(native)
        self.move_node(viewer, 1.0)
        self.click(viewer.history_buttons.undo)
        self.click(viewer.history_buttons.redo)
        self.assertIs(QApplication.focusWidget(), native)
        QTest.keyClicks(QApplication.focusWidget(), "abc")
        self.assertEqual(recorder.keys, ["a", "b", "c"])

    def test_the_command_lines_caret_goes_on_blinking_after_a_click(self):
        viewer = self.build()
        native = self.canvas_with_focus(viewer)
        caret = CommandCaret(native)
        caret.show_at(30.0, 12.0, 20.0)
        self.assertTrue(caret.blinking)
        self.move_node(viewer, 1.0)
        self.click(viewer.history_buttons.undo)
        self.assertIs(QApplication.focusWidget(), native)
        self.assertTrue(caret.blinking)


class NeverMovesTests(ViewerCase):
    """A state change repaints a button and nothing more: no move, resize, show or hide, and no canvas update."""

    GUARDED = ("setGeometry", "move", "resize", "setVisible", "show", "hide", "raise_", "lower", "setFixedSize")

    def guard(self, viewer):
        guards = []
        for button in viewer.history_buttons.buttons():
            for name in self.GUARDED:
                patcher = mock.patch.object(button, name)
                guards.append(patcher.start())
                self.addCleanup(patcher.stop)
        return guards

    def geometries(self, viewer):
        widgets = [*viewer.history_buttons.buttons(), viewer.jobs_badge, viewer.toggle_sidebar_btn]
        if viewer.fps_overlay is not None:
            widgets.append(viewer.fps_overlay)
        return [(widget.geometry(), widget.isHidden()) for widget in widgets]

    def test_changing_the_enabled_state_moves_nothing(self):
        viewer = self.build(fps=True)
        before = self.geometries(viewer)
        guards = self.guard(viewer)
        for undo, redo in ((True, False), (True, True), (False, True), (False, False), (True, True), (True, True)):
            viewer.history_buttons.show_state(undo, redo)
            self.assertEqual(self.state(viewer), (undo, redo))
        for method in guards:
            method.assert_not_called()
        viewer.canvas.update.assert_not_called()
        self.assertEqual(self.geometries(viewer), before)

    def test_every_place_that_changes_the_history_moves_nothing(self):
        viewer = self.build(fps=True)
        before = self.geometries(viewer)
        guards = self.guard(viewer)
        self.move_node(viewer, 1.0)
        self.move_node(viewer, 2.0)
        self.click(viewer.history_buttons.undo)
        self.click(viewer.history_buttons.redo)
        quietly(lambda: viewer._do_undo(announce=False))
        quietly(lambda: viewer._do_redo(announce=False))
        self.give_command(viewer)
        self.click(viewer.history_buttons.undo)
        viewer._save_metadata_cell_history("Length", 0, 1, 2)
        for method in guards:
            method.assert_not_called()
        self.assertEqual(self.geometries(viewer), before)

    def test_placing_the_pair_where_it_already_is_moves_nothing(self):
        viewer = self.build()
        guards = self.guard(viewer)
        viewer.reposition_expand_btn()
        viewer.reposition_expand_btn()
        for method in guards:
            method.assert_not_called()

    def test_a_dimmed_button_keeps_its_size_and_icon_and_has_the_arrow_cursor(self):
        viewer = self.build()
        for button in viewer.history_buttons.buttons():
            self.assertFalse(button.isEnabled())
            self.assertEqual(button.size().toTuple(), (HISTORY_BUTTON_SIZE, HISTORY_BUTTON_SIZE))
            self.assertFalse(button.icon().isNull())
            self.assertEqual(button.cursor().shape(), Qt.CursorShape.ArrowCursor)
        self.move_node(viewer, 1.0)
        undo = viewer.history_buttons.undo
        self.assertEqual(undo.cursor().shape(), Qt.CursorShape.PointingHandCursor)


class LookTests(ViewerCase):
    def test_the_buttons_are_widgets_over_the_canvas_with_lucide_icons_and_a_dimmed_look(self):
        viewer = self.build()
        undo, redo = viewer.history_buttons.buttons()
        for button in (undo, redo):
            self.assertIs(button.parentWidget(), viewer.canvas.native)
            self.assertTrue(button.isVisible())
        self.assertEqual((undo.objectName(), redo.objectName()), ("viewerUndoButton", "viewerRedoButton"))
        # An enabled button and a dimmed one draw different pixels: the icon takes the disabled colour.
        viewer.history_buttons.show_state(True, False)
        enabled = undo.grab().toImage()
        viewer.history_buttons.show_state(False, False)
        self.assertNotEqual(enabled, undo.grab().toImage())
        for button in (undo, redo):
            self.assertIn(":disabled", button.styleSheet())


class BuiltLikeTheOtherOverlays(unittest.TestCase):
    def test_the_buttons_are_built_in_every_window_mode_as_the_other_overlays_are(self):
        # __init__ builds the frame rate, the jobs badge, the command line's caret and the buttons as
        # plain statements, whatever the window mode: a headless or hidden Viewer has the widgets
        # too (nothing of them shows where no window is on screen), so a test can drive them offscreen.
        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        init = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                    and any(isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "_build_fps_overlay"
                            for call in ast.walk(node)))
        built = [statement.value.func.attr for statement in init.body
                 if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                 and isinstance(statement.value.func, ast.Attribute)
                 and statement.value.func.attr.startswith(("_build_fps", "_build_jobs", "_build_command_caret",
                                                           "_build_history"))]
        self.assertEqual(sorted(built), ["_build_command_caret", "_build_fps_overlay", "_build_history_buttons",
                                         "_build_jobs_badge"])
        # The pair is made before the frame rate, so that the card is placed beside it at once.
        self.assertLess(built.index("_build_history_buttons"), built.index("_build_fps_overlay"))


class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def tooltips(self, language):
        self.addCleanup(install_translations, self.app, None)
        install_translations(self.app, language)
        canvas = SimpleNamespace(native=QWidget())
        buttons = HistoryButtons(canvas, lambda: None, lambda: None)
        undo, redo = buttons.buttons()
        self.assertEqual((undo.accessibleName(), redo.accessibleName()), (undo.toolTip(), redo.toolTip()))
        return undo.toolTip(), redo.toolTip()

    def test_the_tooltips_name_the_real_shortcuts_in_english(self):
        self.assertEqual(self.tooltips(None), ("Undo (Ctrl+Z)", "Redo (Ctrl+Y)"))

    def test_the_tooltips_are_translated(self):
        self.assertEqual(self.tooltips("zh_CN"), ("撤销（Ctrl+Z）", "重做（Ctrl+Y）"))
        pseudo = self.tooltips(PSEUDO_LANGUAGE)
        self.assertNotEqual(pseudo, ("Undo (Ctrl+Z)", "Redo (Ctrl+Y)"))


_GL_SCRIPT = textwrap.dedent(
    """
    import sys
    import time
    sys.path.insert(0, sys.argv[1])
    from vispy import app, scene
    app.use_app("pyside6")
    from PySide6 import QtCore
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
    from desktop.Viewer_History import HistoryButtons

    qapp = QApplication.instance() or QApplication([])
    window = QWidget()
    # On a screen, so that it is drawn, but invisible and out of the way of the mouse and keys.
    window.setWindowFlags(QtCore.Qt.WindowType.Tool | QtCore.Qt.WindowType.FramelessWindowHint
                          | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
                          | QtCore.Qt.WindowType.WindowTransparentForInput)
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
    window.setWindowOpacity(0.0)
    canvas = scene.SceneCanvas(size=(480, 200), bgcolor="white", show=False)
    # A red scene under the buttons' place, so that a button drawn into a picture would show.
    scene.visuals.Rectangle(center=(240, 100), width=480, height=200, color=(1, 0, 0, 1), parent=canvas.scene)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(canvas.native)
    window.resize(480, 200)
    window.show()
    draws = [0]
    draw_scene = canvas._draw_scene

    def counted(*args, **kwargs):
        draws[0] += 1
        return draw_scene(*args, **kwargs)

    canvas._draw_scene = counted

    def exposed():
        handle = window.windowHandle()
        return handle is not None and handle.isExposed()

    # Whether the desktop covered, hid or exposed the window since this was last cleared: it draws a window it
    # covers or hides not at all, and Qt draws the whole scene again each time the desktop exposes it, so a
    # count taken then says nothing about the buttons.
    disturbed = [False]

    class ExposeWatcher(QtCore.QObject):
        def eventFilter(self, watched, event):
            if event.type() == QtCore.QEvent.Type.Expose:
                disturbed[0] = True
            return False

    watcher = ExposeWatcher()
    qapp.installEventFilter(watcher)

    def wait(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            qapp.processEvents()
            disturbed[0] = disturbed[0] or not exposed()
            time.sleep(0.005)

    wait(1.0)
    if not canvas.native.isValid():
        print("NO CONTEXT", flush=True)
        sys.exit(0)
    print("CONTEXT OK", flush=True)
    wait(1.0)
    before = canvas.render()
    buttons = HistoryButtons(canvas, lambda: None, lambda: None)
    buttons.place(470, 25)
    wait(0.5)
    after = canvas.render()
    out = {
        "shown": int(buttons.undo.isVisible() and buttons.redo.isVisible()),
        "identical": int(bool((before == after).all())),
        # Where the redo button sits (x 440 to 470, y 10 to 40), the picture is still the scene's red.
        "scene_under_button": int(tuple(after[25, 455, :3]) == (255, 0, 0)),
        "scene_is_red": int(tuple(before[100, 240, :3]) == (255, 0, 0)),
    }

    def phase(name, seconds, action):
        wait(0.3)
        draws[0] = 0
        action()
        wait(seconds)
        out[name] = draws[0]

    def toggles():
        for index in range(24):
            buttons.show_state(index % 2 == 0, index % 3 == 0)
            wait(0.02)

    def measure():
        # Run every phase; return whether the desktop left the window alone all the while, as the counts need.
        disturbed[0] = False
        for name, seconds, action in (
            ("toggles", 0.3, toggles),
            ("placed_again", 0.3, lambda: buttons.place(470, 25)),
            ("control", 0.5, canvas.update),
        ):
            phase(name, seconds, action)
            if disturbed[0]:
                return False
        return True

    give_up = time.monotonic() + 30
    while not (exposed() and measure()):
        if time.monotonic() > give_up:
            print("DISTURBED", flush=True)
            sys.exit(0)
        wait(0.1)
    print("DRAWS", " ".join(f"{key}={value}" for key, value in out.items()), flush=True)
    sys.exit(0)
    """
)


class SceneDrawTests(unittest.TestCase):
    def test_enabling_the_buttons_draws_no_scene_and_a_picture_holds_none_of_them(self):
        environment = dict(os.environ)
        environment.pop("QT_QPA_PLATFORM", None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", QT_ENABLE_HIGHDPI_SCALING="0")
        try:
            result = subprocess.run(
                [sys.executable, "-c", _GL_SCRIPT, SRC_DIR],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=120,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("the GL probe timed out")
        output = result.stdout + result.stderr[-2000:]
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to draw with (exit {result.returncode})")
        if "DISTURBED" in result.stdout:
            self.skipTest("the desktop kept covering or exposing the test window, so its draws cannot be counted")
        self.assertEqual(result.returncode, 0, output)
        line = next(line for line in result.stdout.splitlines() if line.startswith("DRAWS "))
        values = {key: int(value) for key, value in (item.split("=") for item in line.split()[1:])}
        # The control shows the count works: a canvas.update() is a draw.
        self.assertGreaterEqual(values["control"], 1, output)
        for phase in ("toggles", "placed_again"):
            self.assertEqual(values[phase], 0, f"{phase}: {output}")
        # The scene was drawn (red), the buttons were up, and the picture is the same with them as without.
        self.assertEqual(values["scene_is_red"], 1, output)
        self.assertEqual(values["shown"], 1, output)
        self.assertEqual(values["scene_under_button"], 1, output)
        self.assertEqual(values["identical"], 1, output)


if __name__ == "__main__":
    unittest.main()
