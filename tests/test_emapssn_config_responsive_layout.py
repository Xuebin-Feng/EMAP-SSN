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


class ResponsiveConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load GUI definitions without the launcher, IPC, or application event loop.
        path = SRC / "EMAPSSN_Config.py"
        source = path.read_text(encoding="utf-8").split(
            "    existing_qt_application = QApplication.instance()"
        )[0]
        cls.namespace = {"__name__": "__main__", "__file__": str(path)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SSN_VIEWER_SETTINGS_PATH": str(Path(directory) / "missing.json")}
        ):
            exec(compile(source, str(path), "exec"), cls.namespace)
        cls.gui_class = cls.namespace["ConfigGUI"]
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])
        cls.namespace["configure_qt_application_fonts"](cls.app)
        cls.namespace["force_light_palette"](cls.app)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        settings = {"INPUT_FILE_DIR": str(root / "inputs"),
                    "CACHE_FILE_DIR": str(root / "cache"),
                    "ANALYSIS_RESULT_DIR": str(root / "results")}
        with patch.object(self.gui_class, "_read_custom_settings", return_value=settings):
            self.window = self.gui_class()
        self.window.show()
        self.flush()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.flush()

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def resize_panel(self, width):
        self.window.resize(width + 360, 850)
        self.window.main_split.setSizes([width + 4, 326])
        self.flush()
        delta = width - self.window.tabs.currentWidget().width()
        left, right = self.window.main_split.sizes()
        self.window.main_split.setSizes([left + delta, right - delta])
        self.flush()

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
        self.window._toggle_new_cache_input(combo.currentText())
        self.window.line_new_cache.setText("new-layout-name")
        self.flush()

    def test_all_tabs_and_optional_fields_fit_repeated_resizes(self):
        self.show_optional_names()
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
                        groups = [widget for widget in scroll.widget().findChildren(
                            self.namespace["QWidget"]
                        ) if widget.objectName() in {"umapRow", "visualRow5", "physicsSlidersRow0"}]
                        for group in groups:
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
            label = self.window.labels[key]
            field = self.window.inputs[key]
            label_pos = label.mapTo(page, QPoint(0, 0))
            field_pos = field.mapTo(page, QPoint(0, 0))
            self.assertGreater(field_pos.x(), label_pos.x())
            self.assertLess(abs((label_pos.y() + label.height() / 2)
                                - (field_pos.y() + field.height() / 2)), 3)
        self.assertEqual(self.window.tabs.currentWidget().horizontalScrollBar().maximum(), 0)

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
        from utilities.Responsive_Layouts import ResponsiveFlowLayout
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


if __name__ == "__main__":
    unittest.main()
