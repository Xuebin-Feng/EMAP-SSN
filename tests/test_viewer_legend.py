# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's legend card (desktop/Viewer_Legend.py and MainViewer's wiring of it).

The card shows what the colors of the network mean, as the color key says: its layers newest first, counting
shown (not hidden) nodes, then the colors the key does not cover. A click on a row selects its nodes the way a
box selection does, Ctrl adds them, the same row again clears; a press or drag on a gradient bar selects the
nodes whose value is in its range, the selection moving as the drag goes and the pages and the status line
hearing of it once, when it is let go. A Legend button in the top-right row, left of Undo, opens and closes it
(a card like the session card, on white); it hangs under the row, follows the sidebar, and shares the room with the
session card without overlapping it. Offscreen Qt only: how it looks has not been checked on a screen here.
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from desktop import Studio_Theme  # noqa: E402
from commands import print as print_command  # noqa: E402
from desktop import Node_Colors, Viewer_Legend  # noqa: E402
from desktop.Desktop_App import PSEUDO_LANGUAGE, install_translations  # noqa: E402
from desktop.Viewer_Legend import LegendCard, _GradientBar, _ElidedLabel, _RowList, legend_model  # noqa: E402
from desktop.Viewer_History import HISTORY_BUTTON_GAP, HISTORY_BUTTON_SIZE  # noqa: E402
from desktop.Viewer_Fps import FPS_GAP  # noqa: E402
from EMAPSSN_Viewer import SLIDER_ROW_BOTTOM, SLIDER_ROW_HEIGHT  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QToolTip  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from tests.test_viewer_history_buttons import ViewerCase  # noqa: E402
from tests.test_viewer_undo_redo import quietly  # noqa: E402

RED = (1.0, 0.0, 0.0, 1.0)
GREEN = (0.0, 1.0, 0.0, 1.0)
GRAY = (0.5, 0.5, 0.5, 1.0)
ODD = (0.2, 0.4, 0.8, 1.0)          # #3366cc
N = 12
# Nodes 0-2 are cluster A, 3-4 cluster B; 5-8 a gradient of the values 0 to 3; 9 is unpainted;
# 10 is the gradient's "No value" extra; 11 has a color no layer declared.
CTRL = Qt.KeyboardModifier.ControlModifier
SIZES_UNDER_TEST = ((1400, 900), (563, 300))   # a roomy window, and the Viewer's smallest


class LegendCase(ViewerCase):
    """A Viewer with the legend, around the stand-in canvas of the history buttons' tests, and a key to show."""

    def build_legend(self, size=(1400, 900), *, toggle=False, with_key=True, n=N, opened=True, fps=False):
        viewer = self.build(size, toggle=toggle, fps=fps)
        viewer.n_nodes = n
        viewer.pos = np.zeros((n, 2), dtype=np.float32)
        viewer.visible_mask = np.ones(n, dtype=bool)
        viewer.node_render_order = np.arange(n, dtype=np.int32)
        viewer.metadata = {}
        viewer.full_headers = [f"n{i}" for i in range(n)]
        viewer.custom_scores = np.zeros(n)
        from matplotlib import colors as mcolors
        plain = np.array(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), dtype=np.float32)
        viewer.current_colors = np.tile(plain, (n, 1)).astype(np.float32)
        viewer.selected_indices = []
        for name in Node_Colors.KEY_ATTRIBUTES:
            setattr(viewer, name, None)
        viewer._cacheable_attrs = set(viewer._cacheable_attrs) | set(Node_Colors.KEY_ATTRIBUTES)
        viewer.broadcast_event = mock.Mock()
        viewer._show_click_status = mock.Mock()

        def update_selection_visual(broadcast=True):
            viewer._notify_node_update_listeners()
            if broadcast:
                viewer.broadcast_event({"type": "selection_changed", "indices": viewer.selected_indices})
        viewer.update_selection_visual = update_selection_visual
        viewer._build_legend()
        self.assertIsNotNone(viewer.legend_card)
        if with_key:
            self.declare(viewer)
        viewer.reposition_expand_btn()
        self.settle()
        if opened:
            self.click_legend_button(viewer)
        return viewer

    def click_legend_button(self, viewer):
        """A real click on the Legend button, which opens the card, or closes it if it is open."""
        QTest.mouseClick(viewer.history_buttons.legend, Qt.MouseButton.LeftButton)
        self.settle()

    @staticmethod
    def shown(card):
        """The card's rectangle in canvas px (inside its shadow)."""
        return card.visible_rect()

    def settle(self):
        self.app.processEvents()
        self.app.processEvents()

    def declare(self, viewer):
        """Two layers, as commands declare them: clusters, then a gradient with an extra, then an odd color."""
        colors = viewer.current_colors
        colors[0:3] = RED
        colors[3:5] = GREEN
        Node_Colors.declare_categories(viewer, "Clusters", [("A", RED), ("B", GREEN)],
                                       [0, 0, 0, 1, 1], nodes=np.arange(0, 5))
        viridis = colormaps["viridis"]
        values = np.array([0.0, 1.0, 2.0, 3.0])
        colors[5:9] = viridis(values / 3.0)
        colors[10] = GRAY
        Node_Colors.declare_gradient(viewer, "Spectrum", "viridis", viridis, 0.0, 3.0, np.arange(5, 9), values,
                                     extras=[("No value", GRAY, [10])])
        colors[11] = ODD
        viewer._notify_node_update_listeners()
        self.settle()

    @staticmethod
    def lists(card):
        return card.body.findChildren(_RowList)

    @staticmethod
    def titles(card):
        return [label.text() for label in card.body.findChildren(_ElidedLabel) if label.objectName() == "legendTitle"]

    def click_row(self, rowlist, index, modifiers=Qt.KeyboardModifier.NoModifier):
        QTest.mouseClick(rowlist, Qt.MouseButton.LeftButton, modifiers, QPoint(20, Viewer_Legend.ROW_HEIGHT * index + 5))
        self.settle()

    def selection_events(self, viewer):
        return [call.args[0] for call in viewer.broadcast_event.call_args_list
                if call.args and call.args[0].get("type") == "selection_changed"]


class ShowingTests(LegendCase):
    def test_a_key_that_appears_does_not_open_the_card_and_the_open_card_follows_it(self):
        viewer = self.build_legend(with_key=False, opened=False)
        card = viewer.legend_card
        self.assertTrue(card.isHidden())
        self.assertIsNone(legend_model(viewer))
        self.declare(viewer)
        self.assertTrue(card.isHidden())                   # no auto-open
        self.assertFalse(card.is_open())
        self.click_legend_button(viewer)
        self.assertFalse(card.isHidden())
        self.assertEqual(self.titles(card), ["Spectrum", "Clusters", "Unlabelled colours"])
        self.assertTrue(card.scroll.isVisible() or not card.scroll.isHidden())

    def test_without_a_key_the_open_card_says_so_in_one_plain_line_naming_no_command(self):
        viewer = self.build_legend(with_key=False)
        card = viewer.legend_card
        self.assertFalse(card.isHidden())
        lines = [label for label in card.body.findChildren(QLabel) if label.objectName() == "legendEmpty"]
        self.assertEqual([line.text() for line in lines], ["No colours have been described yet."])
        self.assertEqual(self.titles(card), [])
        self.assertEqual(self.lists(card), [])
        commands = {path.stem.lower() for path in (Path(SRC_DIR) / "commands").glob("*.py") if path.stem != "__init__"}
        words = {word.strip(".,").lower() for word in lines[0].text().split()}
        self.assertEqual(words & commands, set())
        # A key that arrives replaces the line as it is told.
        self.declare(viewer)
        self.assertEqual([label for label in card.body.findChildren(QLabel) if label.objectName() == "legendEmpty"], [])

    def test_layers_are_listed_newest_first_with_titles_rows_and_counts(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        self.assertEqual(self.titles(card), ["Spectrum", "Clusters", "Unlabelled colours"])
        rows = [rowlist.rows() for rowlist in self.lists(card)]
        self.assertEqual([[(label, count) for _, label, count in group] for group in rows],
                         [[("No value", "1")], [("A", "3"), ("B", "2")], [("#3366cc", "1")]])
        self.assertEqual(rows[1][0][0], (255, 0, 0))

    def test_counts_are_of_shown_nodes_and_an_empty_row_or_layer_is_left_out(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        viewer.visible_mask[[0, 3, 4]] = False
        viewer._notify_node_update_listeners()
        self.settle()
        groups = {title: rowlist.rows() for title, rowlist in zip(self.titles(card), self.lists(card))}
        self.assertEqual([(label, count) for _, label, count in groups["Clusters"]], [("A", "2")])  # B is gone
        viewer.visible_mask[[1, 2]] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertNotIn("Clusters", self.titles(card))                # no shown node: the layer is gone
        viewer.visible_mask[:] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIsNone(card.model)                                  # nothing shown to describe: the plain line

    def test_a_gradient_shows_its_title_scheme_bar_and_ends(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        layer = legend_model(viewer)["layers"][0]
        self.assertEqual((layer["kind"], layer["title"], layer["scheme"], layer["min"], layer["max"], layer["bar"]),
                         ("gradient", "Spectrum", "viridis", 0.0, 3.0, 4))
        self.assertEqual(len(layer["stops"]), Node_Colors.GRADIENT_STOPS)
        self.assertEqual(len(card.body.findChildren(_GradientBar)), 1)
        texts = {label.text() for label in card.body.findChildren(_ElidedLabel)}
        self.assertTrue({"viridis", "0", "3"} <= texts)

    def test_unlabelled_rows_group_by_color_and_leave_the_unpainted_color_out(self):
        viewer = self.build_legend()
        model = legend_model(viewer)
        self.assertEqual([(entry["hex"], entry["count"]) for entry in model["unlabelled"]], [("#3366cc", 1)])
        # A node painted after its layer declared it is no longer what the key says: it joins the unlabelled.
        viewer.current_colors[0] = (0.2, 0.4, 0.8, 1.0)
        viewer.current_colors[1] = (0.21, 0.41, 0.81, 1.0)          # the same class: 4 bits per channel
        self.assertEqual([(e["hex"], e["count"]) for e in legend_model(viewer)["unlabelled"]], [("#3366cc", 3)])
        counts = {row["label"]: row["count"] for row in legend_model(viewer)["layers"][1]["rows"]}
        self.assertEqual(counts, {"A": 1, "B": 2})

    def test_labels_are_plain_text(self):
        viewer = self.build_legend(with_key=False)
        viewer.current_colors[0:2] = RED
        Node_Colors.declare_categories(viewer, "<b>Bold</b> title", [("<i>x</i>", RED)], [0, 0], nodes=[0, 1])
        viewer._notify_node_update_listeners()
        self.settle()
        card = viewer.legend_card
        label = next(label for label in card.body.findChildren(_ElidedLabel) if label.objectName() == "legendTitle")
        self.assertEqual(label.text(), "<b>Bold</b> title")
        self.assertEqual(label.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(self.lists(card)[0].rows()[0][1], "<i>x</i>")
        # What is painted: no rich text anywhere in the card's widgets.
        for widget in card.findChildren(_ElidedLabel):
            self.assertEqual(widget.textFormat(), Qt.TextFormat.PlainText)


class SelectionTests(LegendCase):
    def test_a_click_replaces_the_selection_ctrl_adds_and_the_same_row_clears(self):
        viewer = self.build_legend()
        clusters = self.lists(viewer.legend_card)[1]
        viewer.selected_indices = [11]
        self.click_row(clusters, 0)
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2])
        self.click_row(clusters, 1, CTRL)
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2, 3, 4])
        self.click_row(clusters, 1)                                     # replaces with B alone
        self.assertEqual(sorted(viewer.selected_indices), [3, 4])
        self.click_row(clusters, 1)                                     # exactly the selection: clears
        self.assertEqual(viewer.selected_indices, [])
        unlabelled = self.lists(viewer.legend_card)[2]
        self.click_row(unlabelled, 0)
        self.assertEqual(viewer.selected_indices, [11])

    def test_hidden_nodes_are_never_selected(self):
        viewer = self.build_legend()
        viewer.visible_mask[1] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.click_row(self.lists(viewer.legend_card)[1], 0)
        self.assertEqual(sorted(viewer.selected_indices), [0, 2])
        viewer.visible_mask[:] = True
        viewer.visible_mask[[6, 7]] = False
        self.drag_bar(viewer, 0.0, 1.0)
        self.assertEqual(sorted(viewer.selected_indices), [5, 8])

    def drag_bar(self, viewer, first, second, modifiers=Qt.KeyboardModifier.NoModifier, steps=5):
        bar = viewer.legend_card.body.findChildren(_GradientBar)[0]
        width = bar.width()
        self.assertGreater(width, 50)
        QTest.mousePress(bar, Qt.MouseButton.LeftButton, modifiers, QPoint(int(first * (width - 1)), 6))
        for step in range(1, steps + 1):
            share = first + (second - first) * step / steps
            QTest.mouseMove(bar, QPoint(int(share * (width - 1)), 6))
        QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, modifiers, QPoint(int(second * (width - 1)), 6))
        self.settle()

    def test_a_drag_on_the_bar_selects_the_range_and_shows_it(self):
        viewer = self.build_legend()
        self.drag_bar(viewer, 0.3, 0.7)           # 0.9 to 2.1 of 0 to 3: the values 1 and 2
        self.assertEqual(sorted(viewer.selected_indices), [6, 7])
        card = viewer.legend_card
        label = next(label for label in card.body.findChildren(_ElidedLabel) if label.objectName() == "legendRange")
        self.assertFalse(label.isHidden())
        low, high = (float(part) for part in label.text().split(" to "))
        self.assertAlmostEqual(low, 0.9, delta=0.05)
        self.assertAlmostEqual(high, 2.1, delta=0.05)
        # Ctrl adds the range to what is selected; a press without Ctrl replaces it.
        self.drag_bar(viewer, 0.0, 0.2, CTRL)
        self.assertEqual(sorted(viewer.selected_indices), [5, 6, 7])
        self.drag_bar(viewer, 0.9, 1.0)
        self.assertEqual(sorted(viewer.selected_indices), [8])
        # A selection made some other way takes the range line away.
        viewer.selected_indices = [0]
        card.refresh()
        self.assertTrue(label.isHidden())

    def test_the_selection_is_told_once_per_gesture(self):
        viewer = self.build_legend()
        self.drag_bar(viewer, 0.1, 0.9, steps=8)
        self.assertEqual(len(self.selection_events(viewer)), 1)
        self.assertEqual(self.selection_events(viewer)[0]["indices"], viewer.selected_indices)
        self.assertEqual(viewer._show_click_status.call_count, 1)
        viewer.broadcast_event.reset_mock()
        viewer._show_click_status.reset_mock()
        self.click_row(self.lists(viewer.legend_card)[1], 0)
        self.assertEqual(len(self.selection_events(viewer)), 1)
        self.assertEqual(viewer._show_click_status.call_count, 1)
        # A gesture that leaves the selection as it was tells nobody.
        viewer.broadcast_event.reset_mock()
        viewer._show_click_status.reset_mock()
        viewer.selected_indices = []
        self.drag_bar(viewer, 0.4, 0.45)           # between 1.2 and 1.35: no node
        self.assertEqual((self.selection_events(viewer), viewer._show_click_status.call_count), ([], 0))

    def test_a_selection_goes_through_the_box_selections_path(self):
        viewer = self.build_legend()
        epoch = getattr(viewer, "selection_epoch", 0)
        self.click_row(self.lists(viewer.legend_card)[1], 0)
        self.assertEqual(viewer.selection_epoch, epoch + 1)       # the commands hear of it (_note_hand_selection)


class PlacementTests(LegendCase):
    def test_it_hangs_under_the_top_right_row_right_edges_level(self):
        viewer = self.build_legend()
        card = self.shown(viewer.legend_card)
        redo = viewer.history_buttons.redo.geometry()
        self.assertEqual(card.right(), redo.right() + 1)
        self.assertEqual(card.top() - redo.bottom() - 1, EMAPSSN_Viewer.BADGE_GAP)
        self.assertLessEqual(card.left(), viewer.history_buttons.legend.geometry().left())

    def test_it_moves_down_while_the_jobs_badge_shows(self):
        viewer = self.build_legend()
        before = self.shown(viewer.legend_card).top()
        viewer.jobs_badge.show_jobs(["a job"])
        viewer._place_legend()
        self.assertGreater(self.shown(viewer.legend_card).top(), viewer.jobs_badge.geometry().bottom())
        viewer.jobs_badge.show_jobs([])
        viewer._place_legend()
        self.assertEqual(self.shown(viewer.legend_card).top(), before)

    def test_it_follows_the_sidebar_and_never_sits_under_it(self):
        patch = mock.patch.object(EMAPSSN_Viewer, "SIDEBAR_ENABLED", True)
        patch.start()
        self.addCleanup(patch.stop)
        viewer = self.build_legend(toggle=True)
        closed = viewer.legend_card.geometry()
        viewer.set_sidebar_visible(True, animate=False)
        self.settle()
        opened = viewer.legend_card.geometry()
        self.assertLess(opened.right(), closed.right())
        self.assertLess(self.shown(viewer.legend_card).right(), viewer.right_panel.geometry().left())
        self.assertLess(self.shown(viewer.legend_card).right(), viewer.toggle_sidebar_btn.geometry().left())
        self.assertEqual(self.shown(viewer.legend_card).right(), viewer.history_buttons.redo.geometry().right() + 1)
        # Mid-slide too, as the slide width is what the row follows.
        for width in (30.0, 90.0, 150.0):
            viewer._sidebar_slide_width = width
            viewer.reposition_expand_btn()
            self.assertLess(self.shown(viewer.legend_card).right(), viewer.right_panel.geometry().left())
        viewer._sidebar_slide_width = None
        viewer.set_sidebar_visible(False, animate=False)
        self.settle()
        self.assertEqual(viewer.legend_card.geometry(), closed)

    def test_its_height_is_capped_above_the_status_lines_and_the_slider_row(self):
        for size in ((1400, 900), (563, 300)):
            with self.subTest(size=size):
                viewer = self.build_legend(size, n=4000)
                # Many categories: more rows than any window holds.
                rows = [(f"cat {i}", (1.0, i / 400.0, 0.0, 1.0)) for i in range(300)]
                viewer.current_colors[:300] = [row[1] for row in rows]
                Node_Colors.declare_categories(viewer, "Many", rows, np.arange(300), nodes=np.arange(300))
                viewer._notify_node_update_listeners()
                self.settle()
                card = viewer.legend_card
                self.assertGreater(card.natural_height(), 2000)
                height = viewer.canvas.size[1]
                slider_top = height - SLIDER_ROW_BOTTOM - SLIDER_ROW_HEIGHT
                self.assertLess(self.shown(card).bottom(), slider_top - 2 * 34 + 17 + 1)
                self.assertLess(self.shown(card).bottom(), slider_top)
                self.assertGreater(self.shown(card).height(), 40)
                self.assertFalse(card.scroll.verticalScrollBar().maximum() == 0)   # the content scrolls

    def test_the_open_card_is_never_shorter_than_its_rows_after_the_key_changes(self):
        # A rebuilt body's new widgets stay hidden until a queued event, so its size hint was 0 when the
        # card was placed right after a refresh: the card kept its heading's height (rows cut off).
        from desktop.Viewer_Widgets import ControlsCard
        for opened_first in (True, False):
            with self.subTest(opened_first=opened_first):
                viewer = self.build_legend((1400, 900), with_key=False, opened=opened_first)
                card = viewer.legend_card
                for step, name in enumerate(("one", "two")):
                    rows = [(f"{name} {i}", RED if i else GREEN) for i in range(3)]
                    first = step * 4
                    viewer.current_colors[first:first + 3] = RED
                    Node_Colors.declare_categories(viewer, "Colors", rows, [0, 1, 2], nodes=np.arange(first, first + 3))
                    viewer._notify_node_update_listeners()
                    self.settle()
                    if not card.is_open():
                        self.click_legend_button(viewer)
                    wanted = card.natural_height() - 2 * ControlsCard.SHADOW
                    self.assertGreater(wanted, 100)
                    self.assertGreaterEqual(self.shown(card).height(), wanted - 1)
                    for rowlist in self.lists(card):
                        self.assertTrue(card.scroll.viewport().rect().contains(
                            rowlist.mapTo(card.scroll.viewport(), rowlist.rect().bottomLeft())))

    def test_mouse_events_on_the_card_stop_at_it_but_not_in_its_shadow(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        seen = PressCounter(card.parentWidget())
        body = card.card_rect()
        # The card's padding, where no widget of it takes a press: it must stop at the card, not reach the canvas.
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(body.left()) + 3, int(body.bottom()) - 3))
        self.assertEqual(seen.reached_parent, 0)
        # The shadow's margin is not the card: a click there is the network's, as at the session card.
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(2, card.height() - 2))
        self.assertEqual(seen.reached_parent, 1)


class ButtonTests(LegendCase):
    def test_the_button_is_in_the_row_left_of_undo_and_counts_in_the_rows_width(self):
        for size in SIZES_UNDER_TEST:
            with self.subTest(size=size):
                viewer = self.build_legend(size, opened=False, fps=True)
                buttons = viewer.history_buttons
                legend, undo, redo = buttons.legend.geometry(), buttons.undo.geometry(), buttons.redo.geometry()
                self.assertEqual(buttons.buttons(), (buttons.legend, buttons.undo, buttons.redo))
                self.assertEqual(legend.size(), undo.size())
                self.assertEqual((legend.width(), legend.height()), (HISTORY_BUTTON_SIZE, HISTORY_BUTTON_SIZE))
                self.assertEqual(legend.top(), undo.top())
                self.assertEqual(undo.left() - legend.right() - 1, HISTORY_BUTTON_GAP)
                self.assertEqual(buttons.width(), 3 * HISTORY_BUTTON_SIZE + 2 * HISTORY_BUTTON_GAP)
                self.assertEqual(redo.right() - legend.left() + 1, buttons.width())
                # The frame rate sits FPS_GAP left of the whole row, and the command line left of that.
                overlay = viewer.fps_overlay
                self.assertEqual(overlay.geometry().right() + 1 + FPS_GAP, legend.left())
                self.assertLessEqual(viewer._command_line_right(), overlay.geometry().left() - FPS_GAP)
                self.assertLess(viewer._command_line_right(), legend.left())

    def test_it_is_always_there_with_the_icon_a_tooltip_and_the_buttons_look(self):
        viewer = self.build_legend(with_key=False, opened=False)
        button = viewer.history_buttons.legend
        undo = viewer.history_buttons.undo
        self.assertFalse(button.isHidden())
        self.assertTrue(button.isEnabled())
        self.assertEqual(button.toolTip(), "Legend")
        self.assertEqual(button.accessibleName(), "Legend")
        self.assertFalse(button.icon().isNull())
        self.assertEqual(button.iconSize(), undo.iconSize())
        self.assertEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)
        # The same sheet as the others, with a checked look added.
        self.assertIn(":checked", button.styleSheet())
        self.assertTrue(button.isCheckable())
        self.assertFalse(undo.isCheckable())

    def test_a_click_opens_the_card_and_a_second_click_or_the_x_closes_it(self):
        viewer = self.build_legend(opened=False)
        card, button = viewer.legend_card, viewer.history_buttons.legend
        self.assertTrue(card.isHidden())
        self.assertFalse(button.isChecked())
        self.click_legend_button(viewer)
        self.assertFalse(card.isHidden())
        self.assertTrue(button.isChecked())
        self.click_legend_button(viewer)
        self.assertTrue(card.isHidden())
        self.assertFalse(button.isChecked())
        self.click_legend_button(viewer)
        self.assertTrue(button.isChecked())
        QTest.mouseClick(card.close_button, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(card.isHidden())
        self.assertFalse(button.isChecked())              # the x un-checks the button
        self.assertEqual(card.close_button.toolTip(), "Close")
        # The card reads the key as it is when it opens.
        viewer.visible_mask[[3, 4]] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.click_legend_button(viewer)
        self.assertEqual([(label, count) for _, label, count in self.lists(card)[1].rows()], [("A", "3")])

    def test_it_is_not_a_popup_and_stays_open_through_clicks_on_the_canvas_and_its_rows(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        self.assertFalse(card.windowFlags() & Qt.WindowType.Popup)
        self.assertIs(card.parentWidget(), viewer.canvas.native)
        QTest.mouseClick(viewer.canvas.native, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(300, 400))
        self.settle()
        self.assertFalse(card.isHidden())
        self.click_row(self.lists(card)[1], 0)
        self.assertFalse(card.isHidden())
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2])

    def test_the_card_is_white_and_shaped_like_the_session_cards(self):
        from desktop.Viewer_Widgets import ControlsCard
        viewer = self.build_legend()
        card = viewer.legend_card
        card.resize(card.size())
        image = card.grab().toImage()
        centre = self.shown(card).center().toPoint()
        self.assertEqual(image.pixelColor(centre.x() - card.x(), centre.y() - card.y()).name(), "#ffffff")
        self.assertEqual(Viewer_Legend.WHITE, "#ffffff")
        self.assertNotEqual(Studio_Theme.TOKENS["tint"].lower(), "#ffffff")
        # Its margins, spacing and close button are the session card's.
        self.assertEqual(card.close_button.width(), ControlsCard.CLOSE_SIZE)
        self.assertEqual(card.layout().spacing(), 10)
        self.assertIn("QFrame#controlsSeparator", card.styleSheet())
        self.assertEqual(len([f for f in card.body.findChildren(QFrame) if f.objectName() == "controlsSeparator"]),
                         len(self.titles(card)) - 1)


class TooltipTests(LegendCase):
    def test_a_tooltip_box_fits_its_text(self):
        viewer = self.build_legend(opened=False)
        widgets = (viewer.history_buttons.legend, viewer.history_buttons.undo, viewer.history_buttons.redo)
        for widget in widgets:
            with self.subTest(widget=widget.objectName()):
                QToolTip.hideText()
                QToolTip.showText(widget.mapToGlobal(QPoint(5, 5)), widget.toolTip(), widget)
                self.settle()
                tip = next(w for w in QApplication.topLevelWidgets() if w.metaObject().className() == "QTipLabel")
                metrics = tip.fontMetrics()
                text = metrics.boundingRect(tip.text())
                extra_x, extra_y = tip.width() - text.width(), tip.height() - text.height()
                # About 6 px round the text at each side and 5 above and below; it was 18 and 14 (Qt adds
                # the style sheet's padding to a margin that is already that padding).
                self.assertLessEqual(extra_x, 2 * 8 + 4)
                self.assertLessEqual(extra_y, 2 * 7 + 4)
                self.assertGreaterEqual(extra_x, 2 * 4)
                self.assertLessEqual(tip.margin(), 3)
                QToolTip.hideText()

    def test_the_style_sheet_has_no_padding_for_qt_to_double(self):
        sheet = Studio_Theme.studio_stylesheet()
        rule = sheet[sheet.index("QToolTip {"):]
        rule = rule[:rule.index("}")]
        self.assertIn("padding: 0px", rule)


class PressCounter(QObject):
    """Counts the mouse presses that reach ``widget`` from a widget in it (an event filter sees propagated events)."""

    def __init__(self, widget):
        super().__init__(widget)
        self.reached_parent = 0
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            self.reached_parent += 1
        return False


INFO = {"node_count": 44000, "visible_node_count": 43000, "edge_count": 5000000, "shown_edge_count": 100000,
        "inputs": {key: f"C:/Users/someone/EMAP-SSN/Cache/{key}_file_name.fasta"
                   for key in ("sequence_set", "network", "layout_cache", "alignment", "embedding")},
        "directories": {}, "project_root": "C:/Users/someone/EMAP-SSN"}


class SessionCardTests(LegendCase):
    def open_info_card(self, viewer):
        from desktop.Viewer_Widgets import ControlsButton
        native = viewer.canvas.native
        button = ControlsButton("Session and controls", "[ENTER] Command | [LeftClick] Highlight", parent=native, host=native,
                                session=lambda: INFO, controls_title="Mouse and keyboard controls",
                                area=viewer._session_card_area, card_changed=viewer._place_legend)
        button.setGeometry(15, native.height() - 60, 28, 28)
        button.show()
        viewer.controls_button = button
        return button

    def both_open(self, size):
        viewer = self.build_legend(size)
        button = self.open_info_card(viewer)
        button.toggle_card()
        self.settle()
        return viewer, button

    @staticmethod
    def session_rect(button):
        info = button.card
        return info.card_rect().translated(info.pos().x(), info.pos().y())

    def assert_apart(self, viewer, button):
        legend, session = self.shown(viewer.legend_card), self.session_rect(button)
        self.assertFalse(legend.intersects(session), (legend, session))
        # Neither goes out of the canvas.
        canvas = QRectF(0, 0, *viewer.canvas.size)
        self.assertTrue(canvas.contains(legend), (legend, canvas))

    def test_both_cards_open_at_once_and_never_overlap(self):
        for size in SIZES_UNDER_TEST:
            with self.subTest(size=size):
                viewer, button = self.both_open(size)
                self.assertFalse(viewer.legend_card.isHidden())
                self.assertFalse(button.card.isHidden())
                self.assert_apart(viewer, button)
                # Closing either leaves the other where it belongs; opening it again is apart again.
                button.card.close()
                self.settle()
                self.assertFalse(viewer.legend_card.isHidden())
                button.toggle_card()
                self.settle()
                self.assert_apart(viewer, button)
                self.click_legend_button(viewer)
                self.assertTrue(viewer.legend_card.isHidden())
                self.assertFalse(button.card.isHidden())
                self.click_legend_button(viewer)
                self.assert_apart(viewer, button)

    def test_in_a_wide_window_the_session_card_stays_whole_beside_the_legend(self):
        viewer, button = self.both_open((1400, 900))
        legend, session = self.shown(viewer.legend_card), self.session_rect(button)
        self.assertLess(session.right() + EMAPSSN_Viewer.LEGEND_CARD_GAP, legend.left() + 1)
        self.assertEqual(legend.top(), 48)                                  # not cut short: the full height
        self.assertGreater(legend.height(), 150)

    def test_in_a_middling_window_the_session_card_is_narrowed_to_end_left_of_the_legend(self):
        viewer, button = self.both_open((1000, 900))
        legend, session = self.shown(viewer.legend_card), self.session_rect(button)
        self.assertLessEqual(session.right() + EMAPSSN_Viewer.LEGEND_CARD_GAP, legend.left())
        self.assertGreaterEqual(session.width(), EMAPSSN_Viewer.SESSION_CARD_MIN_WIDTH)
        self.assertLess(session.width(), button.card.natural_size().width() - 16)    # narrower than it likes
        self.assert_apart(viewer, button)

    def test_in_a_narrow_window_the_legends_height_is_cut_to_end_above_the_session_card(self):
        viewer, button = self.both_open((700, 700))
        legend, session = self.shown(viewer.legend_card), self.session_rect(button)
        self.assertGreater(session.right() + EMAPSSN_Viewer.LEGEND_CARD_GAP, legend.left())   # side by side is no room
        self.assertLessEqual(legend.bottom() + EMAPSSN_Viewer.LEGEND_CARD_GAP, session.top() + 0.5)
        self.assert_apart(viewer, button)

    def test_opening_closing_resizing_and_the_sidebar_lay_the_cards_out_again(self):
        patch = mock.patch.object(EMAPSSN_Viewer, "SIDEBAR_ENABLED", True)
        patch.start()
        self.addCleanup(patch.stop)
        viewer = self.build_legend(toggle=True)
        button = self.open_info_card(viewer)
        button.toggle_card()
        self.settle()
        self.assert_apart(viewer, button)
        before = self.shown(viewer.legend_card)
        # A resize.
        for size in ((1100, 800), (900, 700), (1400, 900)):
            viewer.canvas.size = size
            viewer.reposition_expand_btn()
            self.settle()
            self.assert_apart(viewer, button)
            self.assertEqual(self.shown(viewer.legend_card).right(), viewer.history_buttons.redo.geometry().right() + 1)
        self.assertEqual(self.shown(viewer.legend_card), before)
        # The sidebar slides open: the legend moves left with the row and the cards still keep apart.
        viewer.set_sidebar_visible(True, animate=False)
        self.settle()
        self.assertLess(self.shown(viewer.legend_card).right(), before.right())
        self.assert_apart(viewer, button)
        for width in (30.0, 90.0, 150.0):
            viewer._sidebar_slide_width = width
            viewer.reposition_expand_btn()
            self.assert_apart(viewer, button)
        viewer._sidebar_slide_width = None
        viewer.set_sidebar_visible(False, animate=False)
        self.settle()
        self.assertEqual(self.shown(viewer.legend_card), before)

    def test_row_clicks_still_select_while_the_session_card_is_open(self):
        viewer, button = self.both_open((1400, 900))
        self.click_row(self.lists(viewer.legend_card)[1], 0)
        self.assertEqual(sorted(viewer.selected_indices), [0, 1, 2])
        self.assertFalse(button.card.isHidden())
        self.assertFalse(viewer.legend_card.isHidden())
        self.drag_row_bar(viewer)

    def drag_row_bar(self, viewer):
        bar = viewer.legend_card.body.findChildren(_GradientBar)[0]
        QTest.mouseClick(bar, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(bar.width() - 2, 6))
        self.settle()
        self.assertEqual(sorted(viewer.selected_indices), [8])

    def test_the_minimum_window_width_covers_the_top_row(self):
        viewer = self.build_legend(fps=True)
        buttons = viewer.history_buttons
        need = 20 + EMAPSSN_Viewer.COMMAND_LINE_MIN_WIDTH + FPS_GAP + buttons.width() + FPS_GAP \
            + viewer.fps_overlay.widest_width(rate_only=True) + 10
        self.assertEqual(viewer._top_row_width(), need)
        self.assertGreaterEqual(viewer.main_window.minimumWidth(), need)
        # At the least the window may be, the command line keeps its least width beside the row.
        viewer.canvas.size = (viewer.main_window.minimumWidth(), 300)
        viewer.reposition_expand_btn()
        self.assertGreaterEqual(viewer._command_line_right() - 20, EMAPSSN_Viewer.COMMAND_LINE_MIN_WIDTH)


class KeyRuleTests(LegendCase):
    def test_no_key_means_no_rows_even_when_nodes_are_colored(self):
        viewer = self.build_legend(with_key=False)
        viewer.current_colors[0:5] = RED               # a cache's own colors: no key declared
        viewer.current_colors[5] = ODD
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIsNone(viewer.color_key_layers)
        self.assertIsNone(viewer.legend_card.model)                    # the card has its one plain line, and no rows
        self.assertEqual(self.lists(viewer.legend_card), [])
        self.assertIsNone(legend_model(viewer))

    def test_a_key_that_covers_nothing_shown_keeps_the_card_for_the_unlabelled_colors(self):
        viewer = self.build_legend()
        viewer.visible_mask[:] = False
        viewer.visible_mask[[11]] = True             # only the node no layer covers is shown
        viewer._notify_node_update_listeners()
        self.settle()
        model = legend_model(viewer)
        self.assertEqual(model["layers"], [])
        self.assertEqual([entry["hex"] for entry in model["unlabelled"]], ["#3366cc"])
        self.assertFalse(viewer.legend_card.isHidden())
        self.assertEqual(self.titles(viewer.legend_card), ["Unlabelled colours"])


class CostTests(LegendCase):
    def test_an_update_that_changed_nothing_the_legend_shows_counts_nothing(self):
        viewer = self.build_legend()
        with mock.patch.object(Viewer_Legend, "legend_model", wraps=legend_model) as spy:
            for _ in range(5):
                viewer.pos[0] += 1.0                                   # a node drag: positions move, colors do not
                viewer._notify_node_update_listeners()
                self.settle()
            self.assertEqual(spy.call_count, 0)
            viewer.current_colors[9] = ODD                             # a recolor, in place
            viewer._notify_node_update_listeners()
            self.settle()
            self.assertEqual(spy.call_count, 1)
            viewer.visible_mask[0] = False                             # a hide, in place
            viewer._notify_node_update_listeners()
            self.settle()
            self.assertEqual(spy.call_count, 2)
            viewer.selected_indices = [1]                              # a selection alone is not what it shows
            viewer._notify_node_update_listeners()
            self.settle()
            self.assertEqual(spy.call_count, 2)

    def test_a_new_key_with_the_same_colors_is_noticed(self):
        viewer = self.build_legend()
        first = self.titles(viewer.legend_card)
        viewer.color_key_layers = [dict(layer, title="Renamed") if layer["title"] == "Clusters" else layer
                                   for layer in viewer.color_key_layers]
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIn("Renamed", self.titles(viewer.legend_card))
        self.assertNotEqual(first, self.titles(viewer.legend_card))


class KeepingCurrentTests(LegendCase):
    def test_undo_updates_it(self):
        viewer = self.build_legend(with_key=False)
        card = viewer.legend_card
        viewer._save_state()
        viewer.current_colors[0:2] = RED
        Node_Colors.declare_categories(viewer, "Clusters", [("A", RED)], [0, 0], nodes=[0, 1])
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIsNotNone(card.model)
        self.assertEqual(self.titles(card), ["Clusters"])
        self.assertTrue(quietly(viewer._do_undo))
        self.settle()
        self.assertIsNone(card.model)
        self.assertEqual(self.titles(card), [])
        self.assertTrue(quietly(viewer._do_redo))
        self.settle()
        self.assertEqual(self.titles(card), ["Clusters"])

    def test_a_burst_of_updates_is_one_refresh(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        with mock.patch.object(Viewer_Legend, "legend_model", wraps=legend_model) as spy:
            viewer.current_colors[9] = ODD
            for _ in range(6):
                viewer._notify_node_update_listeners()
            self.assertEqual(spy.call_count, 0)           # nothing until the turn ends
            self.settle()
            self.assertEqual(spy.call_count, 1)
        del card

    def test_nothing_is_rebuilt_while_what_it_says_is_the_same(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        first = self.lists(card)[0]
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIs(self.lists(card)[0], first)
        viewer.visible_mask[3] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIsNot(self.lists(card)[0], first)

    def test_it_is_colored_from_the_theme_it_was_built_under_and_follows_a_new_one(self):
        from desktop import Studio_Theme
        viewer = self.build_legend()
        card = viewer.legend_card
        self.assertIn(Studio_Theme.TOKENS["accent_text"], card.styleSheet())
        previous = Studio_Theme.theme_color()
        self.addCleanup(Studio_Theme.set_theme_color, previous)
        other = next(name for name in Studio_Theme.THEME_COLORS if name != previous and
                     Studio_Theme.theme_tokens(name)["accent_text"] != Studio_Theme.TOKENS["accent_text"])
        Studio_Theme.set_theme_color(other)
        card.refresh()
        self.assertIn(Studio_Theme.TOKENS["accent_text"], card.styleSheet())

    def test_it_follows_the_language(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        button = viewer.history_buttons.legend
        self.assertEqual((card.heading.text(), button.toolTip()), ("Legend", "Legend"))
        self.addCleanup(install_translations, self.app, None)
        install_translations(self.app, PSEUDO_LANGUAGE)
        self.settle()
        self.assertNotEqual(card.heading.text(), "Legend")
        self.assertNotIn("Unlabelled colours", self.titles(card))


class WindowlessTests(LegendCase):
    def hidden_viewer(self, flags):
        viewer = self.build(toggle=False)
        for name, value in flags.items():
            setattr(viewer, name, value)
        viewer.node_update_listeners = []
        viewer._build_legend()
        return viewer

    def test_a_viewer_with_no_window_creates_no_card(self):
        for flags in ({"headless": True}, {"window_control": SimpleNamespace(shows_windows=lambda: False)}):
            with self.subTest(flags=flags):
                viewer = self.hidden_viewer(flags)
                viewer._legend_listener()
                self.assertIsNone(getattr(viewer, "legend_card", None))
                viewer._place_legend()      # and placing it is harmless

    def test_a_hidden_viewer_gets_its_card_when_it_is_shown(self):
        shown = {"yes": False}
        control = SimpleNamespace(shows_windows=lambda: shown["yes"], on_shown=None)
        viewer = self.hidden_viewer({"window_control": control})
        self.assertIsNone(getattr(viewer, "legend_card", None))
        viewer._notify_node_update_listeners()
        self.assertIsNone(getattr(viewer, "legend_card", None))
        shown["yes"] = True
        self.assertEqual(control.on_shown, viewer._legend_listener)     # what ViewerWindow.show() calls
        viewer._notify_node_update_listeners()                        # the next refresh makes it
        self.assertIsNotNone(viewer.legend_card)
        self.settle()

    def test_showing_the_window_makes_the_card(self):
        from desktop.Viewer_Window import HIDDEN, ViewerWindow
        viewer = self.build(toggle=False)
        control = ViewerWindow(viewer.main_window, HIDDEN, platform="linux", alert=lambda *args: None)
        viewer.window_control = control
        viewer.node_update_listeners = []
        viewer._build_legend()
        self.assertIsNone(getattr(viewer, "legend_card", None))
        control.show()
        self.assertIsNotNone(viewer.legend_card)


class PrintTests(LegendCase):
    def test_a_printed_picture_does_not_hold_the_card(self):
        # A picture is the canvas's own render (print._render_tile), which draws the scene alone: the card is
        # a widget over the canvas, like the Undo and Redo buttons, so it is never in it, and it is no HUD visual.
        viewer = self.build_legend()
        card = viewer.legend_card
        viewer.hud_displays = {}
        for name in print_command.CAPTURE_HIDDEN_OVERLAYS:
            setattr(viewer, name, None)
        self.assertNotIn(card, print_command._capture_overlays(viewer))
        self.assertIs(card.parentWidget(), viewer.canvas.native)
        canvas = mock.Mock()
        canvas.render.return_value = np.zeros((2, 2, 4), dtype=np.uint8)
        viewer.canvas.render = canvas.render
        print_command._render_tile(viewer, False)
        canvas.render.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
