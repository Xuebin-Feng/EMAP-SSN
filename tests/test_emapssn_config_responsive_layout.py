# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.config_gui_loader import load_config_namespace, open_config_window


class ResponsiveConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load GUI definitions without the launcher, IPC, or application event loop.
        cls.namespace = load_config_namespace()
        cls.gui_class = cls.namespace["ConfigGUI"]
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])
        cls.namespace["configure_qt_application_fonts"](cls.app)
        cls.namespace["force_light_palette"](cls.app)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.window = open_config_window(self.gui_class, self.directory.name)
        self.window.show()
        self.flush()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.flush()

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def resize_panel(self, width, window=None):
        window = self.window if window is None else window
        window.resize(width + 360, 850)
        window.main_split.setSizes([width + 4, 326])
        self.flush()
        delta = width - window.tabs.currentWidget().width()
        left, right = window.main_split.sizes()
        window.main_split.setSizes([left + delta, right - delta])
        self.flush()

    def natural_width(self, label):
        """Width a label's text needs.

        QLabel expands its size hint to its minimum width, so a label in the
        fixed-width column is measured on a fresh copy.
        """
        if isinstance(label, self.namespace["QStackedWidget"]):
            return max(self.natural_width(label.widget(i)) for i in range(label.count()))
        probe = self.namespace["QLabel"](label.text())
        probe.setFont(label.font())
        return probe.sizeHint().width()

    def assert_fields_follow_the_label_column(self, window):
        """Each label at a tab's left edge shows its whole text, and its field starts
        a field spacing after the label column. The column's own labels span it;
        a stacked row's later labels keep their width. Returns how many labels
        were checked."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QLabel, QWidget
        margin = self.namespace["CONFIG_TAB_CONTENT_MARGIN"]
        field_x = (margin + window.label_column_width
                   + self.namespace["CONFIG_FIELD_HORIZONTAL_SPACING"])
        column = set(window._label_column)
        checked = 0
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            self.flush()
            page = window.tabs.currentWidget().widget()
            visible = [widget for widget in page.findChildren(QWidget)
                       if widget.isVisibleTo(page)]
            corner = {widget: widget.mapTo(page, QPoint()) for widget in visible}
            for label in visible:
                if not isinstance(label, QLabel) or not label.text():
                    continue
                if corner[label].x() != margin:
                    continue
                top, bottom = corner[label].y(), corner[label].y() + label.height()
                beside = [corner[other].x() for other in visible
                          if not other.isAncestorOf(label)
                          and corner[other].x() >= margin + label.width()
                          and corner[other].y() < bottom
                          and corner[other].y() + other.height() > top]
                with self.subTest(tab=index, label=label.text()):
                    if column & {label, label.parentWidget()}:
                        self.assertEqual(label.width(), window.label_column_width)
                    self.assertGreaterEqual(label.width(), self.natural_width(label))
                    self.assertEqual(min(beside), field_x)
                checked += 1
        return checked

    def assert_geometry(self, parent):
        QWidget = self.namespace["QWidget"]
        for widget in [parent] + parent.findChildren(QWidget):
            layout = widget.layout()
            if layout is None:
                continue
            children = [layout.itemAt(i).widget() for i in range(layout.count())]
            children = [child for child in children
                        if child is not None and child.isVisibleTo(parent)]
            for index, child in enumerate(children):
                name = f"{widget.objectName()}/{child.objectName()} {type(child).__name__}"
                self.assertTrue(widget.rect().contains(child.geometry()), name)
                for other in children[index + 1:]:
                    self.assertFalse(child.geometry().intersects(other.geometry()), name)

    def show_optional_names(self):
        for key, selector in self.window.profile_selectors.items():
            selector.setCurrentText("(new)")
            self.window.profile_name_inputs[key].setText("A long profile name " * 8)
        combo = self.window.cb_cache_file
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("A long existing cache filename " * 8, "existing.h5")
        combo.addItem("(New Layout Cache)", None)
        combo.setCurrentIndex(1)
        combo.setEnabled(True)
        combo.blockSignals(False)
        self.window._toggle_new_cache_input()
        self.window.line_new_cache.setText("new-layout-name")
        self.flush()

    def test_all_tabs_and_optional_fields_fit_repeated_resizes(self):
        self.show_optional_names()
        stacking_rows = {0: {"filterRow"}, 1: {"visualRow5"},
                         2: {"physicsSlidersRow1", "convergenceRow0"}}
        for width in (600, 800, 1000, 1400, 600, 1400):
            self.resize_panel(width)
            for index in range(self.window.tabs.count()):
                with self.subTest(width=width, tab=index):
                    self.window.tabs.setCurrentIndex(index)
                    self.flush()
                    scroll = self.window.tabs.currentWidget()
                    self.assertEqual(scroll.width(), width)
                    self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                    self.assert_geometry(scroll.widget())
                    if width in (600, 1400):
                        expected = stacking_rows.get(index, set())
                        groups = {widget.objectName(): widget
                                  for widget in scroll.widget().findChildren(
                                      self.namespace["QWidget"]
                                  ) if widget.objectName() in expected}
                        self.assertEqual(set(groups), expected)
                        for group in groups.values():
                            self.assertEqual(group.property("stacked"), width == 600)
            self.assert_geometry(self.window.left_bottom_widget)

    def test_optional_name_wrap_and_hidden_space(self):
        self.show_optional_names()
        self.window.tabs.setCurrentIndex(0)
        self.resize_panel(600)
        selector = self.window.profile_selectors["inputs_outputs"]
        name = self.window.profile_name_inputs["inputs_outputs"]
        folder = self.window.profile_folder_buttons["inputs_outputs"]
        # Explicit wider editor minima exercise the second-row case.
        selector.setMinimumWidth(180)
        name.setMinimumWidth(180)
        self.flush()
        self.assertGreater(name.y(), selector.geometry().bottom())
        self.assertEqual(folder.y(), selector.y())
        before = name.parentWidget().height()
        selector.setCurrentText("(custom)")
        self.flush()
        self.assertTrue(name.isHidden())
        self.assertLess(name.parentWidget().height(), before)
        self.assert_geometry(self.window.tabs.currentWidget().widget())
        self.resize_panel(1400)
        selector.setCurrentText("(new)")
        self.flush()
        self.assertEqual(name.y(), selector.y())

    def test_resize_preserves_values_focus_signals_and_help(self):
        from PySide6.QtCore import QEvent, QPoint
        from PySide6.QtGui import QHelpEvent
        self.show_optional_names()
        self.window.tabs.setCurrentIndex(0)
        self.window.line_ref.setText("reference-id")
        self.window.spin_alignment_offset.setValue(17)
        self.window.check_umap.setChecked(True)
        self.window.spin_umap_k.setValue(23)
        self.window.inputs["TEXT_COLOR"].setText("#123456")
        self.window.inputs["NODE_SIZE"].setValue(12)
        self.flush()
        self.window.line_ref.setFocus()
        self.flush()
        snapshot = self.window.collect_data()
        for width in (600, 1400, 800, 600):
            self.resize_panel(width)
            self.assertIs(self.app.focusWidget(), self.window.line_ref)
            self.assertEqual(self.window.collect_data(), snapshot)
            self.assertTrue(self.window.spin_alignment_offset.isEnabled())
            self.assertTrue(self.window.spin_umap_k.isEnabled())
        self.window.check_umap.setChecked(False)
        self.assertFalse(self.window.spin_umap_k.isEnabled())
        self.window.check_umap.setChecked(True)
        self.window.line_ref.clear()
        self.assertFalse(self.window.spin_alignment_offset.isEnabled())
        self.window.line_ref.setText("reference-id")
        self.app.sendEvent(self.window.labels["UMAP_NEIGHBORS"],
                           QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1)))
        self.assertIn("UMAP", self.window.tip_panel.text())
        self.assertIn("#123456", self.window.color_swatches["TEXT_COLOR"].styleSheet())
        for key in ("visual_effects", "simulation_physics"):
            self.window.profile_selectors[key].setCurrentText("(default)")
        self.resize_panel(1400)
        self.assertFalse(self.window.inputs["NODE_SIZE"].isEnabled())
        self.assertFalse(self.window.inputs["DAMPING"].isEnabled())
        self.window.profile_selectors["visual_effects"].setCurrentText("(new)")
        self.assertTrue(self.window.inputs["NODE_SIZE"].isEnabled())

    def test_labels_stay_beside_long_file_selectors_and_stacked_fields(self):
        from PySide6.QtCore import QPoint
        self.resize_panel(600)
        self.window.tabs.setCurrentIndex(0)
        for key in ("NODE_FASTA_FILE", "MSA_FILE", "INPUT_HDF5"):
            combo = self.window.inputs[key]
            combo.addItem("Very_long_sequence_or_alignment_filename_" * 15)
            combo.setCurrentIndex(combo.count() - 1)
        self.flush()
        page = self.window.tabs.currentWidget().widget()
        for key in ("NODE_FASTA_FILE", "MSA_FILE", "INPUT_HDF5", "UMAP_NEIGHBORS",
                    "UMAP_MIN_DIST", "TOP_EDGE_PERCENT"):
            self.window.check_umap.setChecked(key.startswith("UMAP_"))
            self.flush()
            label = self.window.labels[key]
            field = self.window.inputs[key]
            label_pos = label.mapTo(page, QPoint(0, 0))
            field_pos = field.mapTo(page, QPoint(0, 0))
            self.assertGreater(field_pos.x(), label_pos.x())
            self.assertLess(abs((label_pos.y() + label.height() / 2)
                                - (field_pos.y() + field.height() / 2)), 3)
        self.assertEqual(self.window.tabs.currentWidget().horizontalScrollBar().maximum(), 0)

    def test_input_order_and_shared_rows(self):
        from PySide6.QtCore import QPoint
        self.resize_panel(1400)
        page = self.window.tabs.currentWidget().widget()
        def pos(key):
            return self.window.labels[key].mapTo(page, QPoint())
        order = [pos(key).y() for key in (
            "NODE_FASTA_FILE", "INPUT_HDF5", "MSA_FILE", "ALIGNMENT_SCORE",
            "ALIGNMENT_REFERENCE", "SIMILARITY_THRESHOLD")]
        self.assertEqual(order, sorted(set(order)))
        self.assertEqual(pos("ALIGNMENT_SCORE").y(), pos("NORM_MODE").y())
        keys = ("ALIGNMENT_REFERENCE", "FILTER_MIN_OCCUPANCY", "ALIGNMENT_OFFSET")
        self.assertEqual(len({pos(key).y() for key in keys}), 1)
        self.assertLess(pos(keys[0]).x(), pos(keys[1]).x())
        self.assertLess(pos(keys[1]).x(), pos(keys[2]).x())
        row = self.window.line_ref.parentWidget()
        reference_width = self.window.line_ref.geometry().right() + 1
        self.assertAlmostEqual(reference_width / row.width(), 0.5, delta=0.04)
        offset = self.window.spin_alignment_offset
        self.assertEqual(offset.mapTo(row, QPoint(offset.width(), 0)).x(), row.width())
        switch = self.window.check_umap
        self.assertEqual(switch.geometry().right(), switch.parentWidget().width() - 1)

    def test_umap_swaps_fields_preserving_both_modes(self):
        window = self.window
        values = {"SIMILARITY_THRESHOLD": "0.75", "TOP_EDGE_PERCENT": "0.0",
                  "UMAP_NEIGHBORS": "37", "UMAP_MIN_DIST": "0.42", "UMAP_MODE": True}
        profile = window._normalize_profile_data("inputs_outputs", {
            **window._collect_tab_profile_data("inputs_outputs"), **values})
        window._apply_profile_data("inputs_outputs", profile)
        self.flush()
        self.assertTrue(window.spin_umap_k.isVisible())
        for width in (1400, 600, 800, 1400):
            self.resize_panel(width)
            for umap in (True, False, True, False):
                window.check_umap.setChecked(umap)
                self.flush()
                for key in ("UMAP_NEIGHBORS", "UMAP_MIN_DIST"):
                    self.assertEqual(window.inputs[key].isVisible(), umap)
                    self.assertEqual(window.labels[key].isVisible(), umap)
                for key in ("SIMILARITY_THRESHOLD", "TOP_EDGE_PERCENT"):
                    self.assertEqual(window.inputs[key].isVisible(), not umap)
                    self.assertEqual(window.labels[key].isVisible(), not umap)
                self.assertEqual(window.btn_clear_top_edge.isVisible(), not umap)
                data = window.collect_data()
                self.assertEqual(data["UMAP_MODE"], umap)
                for key, value in values.items():
                    if key != "UMAP_MODE":
                        self.assertEqual(float(data[key]), float(value))
                self.assert_geometry(window.tabs.currentWidget().widget())

    def test_input_columns_align_across_rows_and_modes(self):
        from PySide6.QtCore import QPoint
        page = self.window.tabs.currentWidget().widget()
        for width in (1400, 1000, 600, 1400):
            self.resize_panel(width)
            for umap in (False, True):
                self.window.check_umap.setChecked(umap)
                self.flush()
                first = "UMAP_NEIGHBORS" if umap else "SIMILARITY_THRESHOLD"
                second = "UMAP_MIN_DIST" if umap else "TOP_EDGE_PERCENT"
                for keys in (("ALIGNMENT_SCORE", "ALIGNMENT_REFERENCE", first),
                             ("NORM_MODE", "FILTER_MIN_OCCUPANCY", second)):
                    for widgets in (self.window.labels, self.window.inputs):
                        starts = [widgets[key].mapTo(page, QPoint()).x() for key in keys]
                        self.assertEqual(len(set(starts)), 1, (width, umap, keys, starts))
                starts = [self.window.labels[key].mapTo(page, QPoint()).x()
                          for key in ("ALIGNMENT_OFFSET", "UMAP_MODE")]
                self.assertEqual(starts[0], starts[1])
                self.assert_geometry(page)

    def test_alignment_offset_spans_stacked_rows_and_keeps_its_wide_width(self):
        from PySide6.QtCore import QPoint
        page = self.window.tabs.currentWidget().widget()
        offset = self.window.spin_alignment_offset
        label = self.window.lbl_alignment_offset

        def right(widget):
            return widget.mapTo(page, QPoint(widget.width(), 0)).x()

        for width in (1400, 600, 800, 1400):
            self.resize_panel(width)
            with self.subTest(width=width):
                stacked = width < 1400
                self.assertEqual(offset.parentWidget().property("stacked"), stacked)
                self.assertGreaterEqual(label.width(), label.sizeHint().width())
                if stacked:
                    self.assertEqual(right(offset), right(self.window.line_ref))
                    self.assertEqual(right(offset), right(self.window.spin_min_occ))
                else:
                    self.assertEqual(offset.width(), 100)

    def test_alignment_offset_ignores_hover_wheel_but_accepts_keyboard(self):
        from PySide6.QtCore import QPoint, QPointF, Qt
        from PySide6.QtGui import QWheelEvent
        from PySide6.QtTest import QTest
        self.window.line_ref.setText("reference")
        offset = self.window.spin_alignment_offset
        offset.setValue(17)
        self.window.line_ref.setFocus()
        self.flush()
        self.assertFalse(offset.hasFocus())
        event = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(), QPoint(0, 120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        self.app.sendEvent(offset, event)
        self.assertEqual(offset.value(), 17)
        self.assertFalse(event.isAccepted())
        offset.setFocus()
        QTest.keyClick(offset, Qt.Key.Key_Up)
        self.assertEqual(offset.value(), 18)

    def test_clear_top_edge_unsets_value_and_restores_threshold(self):
        self.window.check_umap.setChecked(False)
        self.window.spin_thresh.setOptionalValue(0.75)
        for value in (5.0, 0.0):
            self.window.spin_top.setOptionalValue(value)
            self.assertFalse(self.window.spin_thresh.isEnabled())
            self.assertTrue(self.window.btn_clear_top_edge.isEnabled())
            self.window.btn_clear_top_edge.click()
            self.assertIsNone(self.window.spin_top.optionalValue())
            self.assertEqual(self.window.spin_top.text().strip(), "")
            self.assertEqual(self.window.collect_data()["TOP_EDGE_PERCENT"], "None")
            self.assertTrue(self.window.spin_thresh.isEnabled())
            self.assertEqual(self.window.spin_thresh.optionalValue(), 0.75)
            self.assertFalse(self.window.btn_clear_top_edge.isEnabled())
        self.window.spin_top.setOptionalValue(5.0)
        self.window.check_umap.setChecked(True)
        self.assertFalse(self.window.btn_clear_top_edge.isEnabled())
        self.window.spin_top.setOptionalValue(None)
        self.assertFalse(self.window.spin_thresh.isEnabled())
        self.window.check_umap.setChecked(False)
        self.assertTrue(self.window.spin_thresh.isEnabled())

    def test_action_row_fills_width_and_stays_at_panel_bottom(self):
        from PySide6.QtWidgets import QWidget
        row = self.window.findChild(QWidget, "configActionButtons")
        layout = row.layout()
        for width in (600, 1400, 600):
            self.resize_panel(width)
            for bottom_height in (200, 350):
                self.window.left_split.setSizes([450, bottom_height])
                self.flush()
                panel = self.window.left_bottom_widget
                self.assertEqual(panel.height() - row.geometry().bottom() - 1, 6)
                self.assertEqual(row.height(), layout.heightForWidth(row.width()))
                self.assertEqual(layout.itemAt(0).widget().x(), 0)
                self.assertEqual(layout.itemAt(4).widget().geometry().right(), row.width() - 1)
                self.assert_geometry(panel)

    def test_action_buttons_wrap_in_order_and_restore(self):
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QPushButton, QWidget
        from desktop.Desktop_App import ResponsiveFlowLayout
        container = QWidget()
        layout = ResponsiveFlowLayout(container)
        buttons = [QPushButton(text) for text in (
            "Save & Run", "Export Layout Settings", "Consistency Check", "Save", "Exit"
        )]
        for button in buttons:
            layout.addWidget(button)
        container.show()
        for width in (300, 1000, 300):
            height = layout.heightForWidth(width)
            container.resize(width, height)
            layout.setGeometry(QRect(0, 0, width, height))
            self.flush()
            self.assert_geometry(container)
            positions = [(button.y(), button.x()) for button in buttons]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len({button.y() for button in buttons}) > 1, width == 300)
        container.close()
        container.deleteLater()

    def test_explicit_minimum_retains_horizontal_scroll_fallback(self):
        self.window.tabs.setCurrentIndex(0)
        self.window.line_ref.setMinimumWidth(700)
        self.resize_panel(600)
        self.assertGreater(self.window.tabs.currentWidget().horizontalScrollBar().maximum(), 0)
        self.assertGreaterEqual(self.window.line_ref.width(), 700)
        self.assert_geometry(self.window.tabs.currentWidget().widget())

    def test_label_column_is_the_longest_label_on_every_tab(self):
        window = self.window
        column = list(dict.fromkeys(window._label_column))
        self.assertGreater(len(column), 30)
        self.assertEqual(window.label_column_width,
                         max(self.natural_width(label) for label in column))
        # Stacked rows move their later labels into the column (600 px).
        for width in (1400, 600):
            self.resize_panel(width)
            with self.subTest(width=width):
                self.assertGreater(self.assert_fields_follow_the_label_column(window), 30)

    def open_window_with(self, **classes):
        """A window built while the module uses ``classes`` in place of its own."""
        with patch.dict(self.namespace, classes):
            window = open_config_window(self.gui_class, self.directory.name)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.show()
        self.flush()
        return window

    def test_a_later_label_longer_than_the_first_column_widens_it(self):
        """A stacked row moves its later labels into the column, so the longest
        of those counts too, or that row's fields would start further right."""
        QLabel = self.namespace["QLabel"]

        class LongLabel(QLabel):
            def __init__(self, text="", *args, **kwargs):
                if text == "Connected Node Color:":
                    text = f"{text} {'~' * 30}"
                super().__init__(text, *args, **kwargs)

        window = self.open_window_with(QLabel=LongLabel)
        longest = window.labels["CONNECTED_NODE_COLOR"]
        self.assertEqual(window.label_column_width, self.natural_width(longest))
        self.assertGreater(window.label_column_width, self.window.label_column_width)
        for width in (1600, 700):
            self.resize_panel(width, window)
            with self.subTest(width=width):
                self.assertGreater(self.assert_fields_follow_the_label_column(window), 30)

    def test_longer_labels_widen_the_column_without_clipping(self):
        """Labels and switch texts half again as long, as a translation may make
        them, move the fields and widen the switches; nothing overlaps or clips."""
        ResponsiveFieldLayout = self.namespace["ResponsiveFieldLayout"]
        QLabel = self.namespace["QLabel"]
        ToggleSwitch = self.namespace["ToggleSwitch"]

        def longer(text):
            return f"{text} {'~' * (len(text) // 2)}" if isinstance(text, str) and text else text

        class LongLabel(QLabel):
            def __init__(self, text="", *args, **kwargs):
                super().__init__(longer(text), *args, **kwargs)

        class LongToggle(ToggleSwitch):
            def __init__(self, on_text="ON", off_text="OFF", parent=None):
                super().__init__(longer(on_text), longer(off_text), parent)

        from desktop.Desktop_App import BUTTON_TEXT_PADDING

        window = self.open_window_with(QLabel=LongLabel, ToggleSwitch=LongToggle)
        self.assertGreater(window.label_column_width, self.window.label_column_width + 40)
        switches = window.findChildren(LongToggle)
        self.assertEqual(len(switches), 4)
        for switch in switches:
            text_width = switch.fontMetrics().horizontalAdvance(switch.text())
            self.assertGreaterEqual(switch.width() - text_width, 2 * BUTTON_TEXT_PADDING)
        for width in (1600, 700):
            self.resize_panel(width, window)
            with self.subTest(width=width):
                self.assertGreater(self.assert_fields_follow_the_label_column(window), 30)
            for index in range(window.tabs.count()):
                window.tabs.setCurrentIndex(index)
                self.flush()
                scroll = window.tabs.currentWidget()
                page = scroll.widget()
                self.assert_geometry(page)
                if width == 1600:
                    self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                for group in page.findChildren(self.namespace["QWidget"]):
                    if not isinstance(group.layout(), ResponsiveFieldLayout):
                        continue
                    for label, _control in group.layout().pairs:
                        if label.isVisibleTo(page):
                            with self.subTest(width=width, label=group.objectName()):
                                self.assertGreaterEqual(label.width(),
                                                        self.natural_width(label))


if __name__ == "__main__":
    unittest.main()
