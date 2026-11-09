# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's legend card (desktop/Viewer_Legend.py and MainViewer's wiring of it).

The card shows what the colors of the network mean, as the color key says: its layers newest first, counting
shown (not hidden) nodes, then the colors the key does not cover. A click on a row selects its nodes the way a
box selection does, Ctrl adds them, the same row again clears; a press or drag on a gradient bar selects the
nodes whose value is in its range, the selection moving as the drag goes and the pages and the status line
hearing of it once, when it is let go. It sits under the Undo and Redo buttons and follows the sidebar. Offscreen
Qt only: how it looks has not been checked on a screen here.
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
from commands import print as print_command  # noqa: E402
from desktop import Node_Colors, Viewer_Legend  # noqa: E402
from desktop.Desktop_App import PSEUDO_LANGUAGE, install_translations  # noqa: E402
from desktop.Viewer_Legend import LegendCard, _GradientBar, _ElidedLabel, _RowList, legend_model  # noqa: E402
from EMAPSSN_Viewer import SLIDER_ROW_BOTTOM, SLIDER_ROW_HEIGHT  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from PySide6.QtCore import QEvent, QObject, QPoint, Qt  # noqa: E402
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


class LegendCase(ViewerCase):
    """A Viewer with the legend, around the stand-in canvas of the history buttons' tests, and a key to show."""

    def build_legend(self, size=(1400, 900), *, toggle=False, with_key=True, n=N):
        viewer = self.build(size, toggle=toggle)
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
        return viewer

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
    def test_no_key_gives_no_card_and_the_first_key_shows_it_expanded(self):
        viewer = self.build_legend(with_key=False)
        card = viewer.legend_card
        self.assertTrue(card.isHidden())
        self.assertIsNone(legend_model(viewer))
        self.declare(viewer)
        self.assertFalse(card.isHidden())
        self.assertTrue(card.expanded)
        self.assertFalse(card.scroll.isHidden())

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
        self.assertTrue(card.isHidden())                               # nothing shown to describe: no card

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
    def test_it_sits_under_the_undo_and_redo_buttons_right_edges_level(self):
        viewer = self.build_legend()
        card = viewer.legend_card.geometry()
        redo = viewer.history_buttons.redo.geometry()
        self.assertEqual(card.right(), redo.right())
        self.assertEqual(card.top() - redo.bottom() - 1, EMAPSSN_Viewer.BADGE_GAP)
        self.assertLessEqual(card.left(), redo.left())

    def test_it_moves_down_while_the_jobs_badge_shows(self):
        viewer = self.build_legend()
        before = viewer.legend_card.geometry().top()
        viewer.jobs_badge.show_jobs(["a job"])
        viewer._place_legend()
        self.assertGreater(viewer.legend_card.geometry().top(), viewer.jobs_badge.geometry().bottom())
        viewer.jobs_badge.show_jobs([])
        viewer._place_legend()
        self.assertEqual(viewer.legend_card.geometry().top(), before)

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
        self.assertLess(opened.right(), viewer.right_panel.geometry().left())
        self.assertLess(opened.right(), viewer.toggle_sidebar_btn.geometry().left())
        self.assertEqual(opened.right(), viewer.history_buttons.redo.geometry().right())
        # Mid-slide too, as the slide width is what the row follows.
        for width in (30.0, 90.0, 150.0):
            viewer._sidebar_slide_width = width
            viewer.reposition_expand_btn()
            self.assertLess(viewer.legend_card.geometry().right(), viewer.right_panel.geometry().left())
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
                self.assertLess(card.geometry().bottom(), slider_top - 2 * 34 + 17 + 1)
                self.assertLess(card.geometry().bottom(), slider_top)
                self.assertGreater(card.geometry().height(), 40)
                self.assertFalse(card.scroll.verticalScrollBar().maximum() == 0)   # the content scrolls

    def test_collapse_and_expand(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        full = card.geometry()
        QTest.mouseClick(card.header, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertFalse(card.expanded)
        self.assertTrue(card.scroll.isHidden())
        self.assertFalse(card.isHidden())
        tab = card.geometry()
        self.assertLess(tab.height(), full.height())
        self.assertLess(tab.width(), full.width())
        self.assertEqual(tab.right(), full.right())
        # The state is kept while the key changes under it, even through a time with no card.
        viewer.visible_mask[:] = False
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertTrue(card.isHidden())
        viewer.visible_mask[:] = True
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertFalse(card.isHidden())
        self.assertTrue(card.scroll.isHidden())
        QTest.mouseClick(card.header, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(card.expanded)
        self.assertEqual(card.geometry(), full)

    def test_mouse_events_on_the_card_stop_at_it(self):
        viewer = self.build_legend()
        card = viewer.legend_card
        seen = PressCounter(card.parentWidget())
        # The card's padding, where no widget of it takes a press: it must stop at the card, not reach the canvas.
        QTest.mouseClick(card, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(3, card.height() - 3))
        self.assertEqual(seen.reached_parent, 0)


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


class SessionCardTests(LegendCase):
    def open_info_card(self, viewer):
        from PySide6.QtCore import QRect
        from desktop.Viewer_Widgets import ControlsButton
        native = viewer.canvas.native
        button = ControlsButton("Session and controls", "[ENTER] Command | [LeftClick] Highlight", parent=native, host=native,
                                area=lambda: QRect(8, 96, native.width() - 16, native.height() - 104),
                                card_changed=viewer._place_legend)
        button.setGeometry(15, native.height() - 60, 28, 28)
        button.show()
        viewer.controls_button = button
        return button

    def test_the_legend_gives_way_while_the_session_card_is_open(self):
        for size in ((1400, 900), (563, 300)):
            with self.subTest(size=size):
                viewer = self.build_legend(size)
                legend = viewer.legend_card
                button = self.open_info_card(viewer)
                self.assertFalse(legend.isHidden())
                button.toggle_card()
                self.settle()
                info = button.card
                self.assertFalse(info.isHidden())
                self.assertTrue(legend.isHidden())
                self.assertFalse(legend.isVisible() and legend.geometry().intersects(info.geometry()))
                # Closed by its x (not the button): the legend comes back where it was.
                info.close()
                self.settle()
                self.assertFalse(legend.isHidden())
                self.assertEqual(legend.geometry().right(), viewer.history_buttons.redo.geometry().right())
                button.toggle_card()
                self.settle()
                self.assertTrue(legend.isHidden())
                button.toggle_card()
                self.settle()
                self.assertFalse(legend.isHidden())


class KeyRuleTests(LegendCase):
    def test_no_key_means_no_card_even_when_nodes_are_colored(self):
        viewer = self.build_legend(with_key=False)
        viewer.current_colors[0:5] = RED               # a cache's own colors: no key declared
        viewer.current_colors[5] = ODD
        viewer._notify_node_update_listeners()
        self.settle()
        self.assertIsNone(viewer.color_key_layers)
        self.assertTrue(viewer.legend_card.isHidden())
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
        self.assertFalse(card.isHidden())
        self.assertTrue(quietly(viewer._do_undo))
        self.settle()
        self.assertTrue(card.isHidden())
        self.assertTrue(quietly(viewer._do_redo))
        self.settle()
        self.assertFalse(card.isHidden())
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
        self.assertEqual(card.header.text(), "Legend")
        self.addCleanup(install_translations, self.app, None)
        install_translations(self.app, PSEUDO_LANGUAGE)
        self.settle()
        self.assertNotEqual(card.header.text(), "Legend")
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
