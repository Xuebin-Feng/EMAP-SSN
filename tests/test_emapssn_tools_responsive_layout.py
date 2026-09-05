# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QTextBrowser, QWidget
from EMAPSSN_Tools import ToolsGUI, ResponsiveFieldLayout, configure_qt_application_fonts


class ResponsiveToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_qt_application_fonts(cls.app)

    def setUp(self):
        # The independent Chromium description panel is not part of form geometry.
        with patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            self.window = ToolsGUI()
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
        # QTabWidget adds four pixels outside its scroll page.
        self.window.splitter.setSizes([width + 4, 326])
        self.flush()
        delta = width - self.window.tabs.currentWidget().width()
        left, right = self.window.splitter.sizes()
        self.window.splitter.setSizes([left + delta, right - delta])
        self.flush()

    def assert_layout_geometry(self, parent):
        """Check real layout children, including compound controls and headers."""
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
                if isinstance(layout, ResponsiveFieldLayout):
                    self.assertGreaterEqual(child.width(), child.minimumSizeHint().width(), name)

    def test_all_tabs_fit_and_restore_rows_across_repeated_resizes(self):
        # Long text must not change the page minimum or resize breakpoints.
        for combo in self.window.findChildren(QComboBox):
            combo.addItem("A very long filename or model option " * 12, "layout-test")
        for editor in self.window.dir_inputs.values():
            editor.setText("C:/long-directory/" * 20)
        for width in (600, 800, 1000, 1400, 600, 1400):
            self.resize_panel(width)
            for index in range(self.window.tabs.count()):
                with self.subTest(width=width, tab=index):
                    self.window.tabs.setCurrentIndex(index)
                    self.flush()
                    scroll = self.window.tabs.widget(index)
                    self.assertEqual(scroll.width(), width)
                    self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
                    self.assertEqual(scroll.widget().minimumWidth(), 0)
                    self.assert_layout_geometry(scroll.widget())
            group = self.window.findChild(
                QWidget, "compactRow_MODEL_NAME_SAVING_MODE_DEVICE_SELECTION"
            )
            if width in (600, 1400):
                self.assertEqual(group.property("stacked"), width == 600)

    def test_exceptional_control_minimum_keeps_scroll_fallback_and_stacks_label(self):
        self.window.tabs.setCurrentIndex(0)
        group = self.window.findChild(
            QWidget, "compactRow_MODEL_NAME_SAVING_MODE_DEVICE_SELECTION"
        )
        label, control = group.layout().pairs[0]
        control.setMinimumWidth(700)
        self.resize_panel(600)
        scroll = self.window.tabs.currentWidget()
        self.assertGreater(scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreaterEqual(control.width(), 700)
        self.assertGreater(control.y(), label.geometry().bottom())
        self.assert_layout_geometry(scroll.widget())

    def test_resize_preserves_settings_focus_and_manual_control_connections(self):
        path = next(path for path in self.window.script_data
                    if path.endswith("Embedding_PWA.py"))
        inputs = self.window.script_data[path]["inputs"]
        toggle = inputs["MANUAL_REF_SEQ"]["widget"]
        editor = inputs["REF_SEQUENCE"]["widget"]
        self.assertIsInstance(editor, QLineEdit)
        tab_index = next(index for index in range(self.window.tabs.count())
                         if self.window.tabs.tabText(index) == "Manual Tools")
        self.window.tabs.setCurrentIndex(tab_index)
        toggle.setChecked(True)
        editor.setText("ACDEFGHIKLMNPQRSTVWY")
        self.flush()
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        before = self.window._collect_tool_settings(path)
        for width in (600, 1400, 800, 600):
            self.resize_panel(width)
            self.assertIs(self.app.focusWidget(), editor)
            self.assertEqual(self.window._collect_tool_settings(path), before)
            self.assertTrue(editor.isEnabled())
        toggle.setChecked(False)
        self.assertFalse(editor.isEnabled())
        toggle.setChecked(True)
        self.assertTrue(editor.isEnabled())
        self.assertEqual(editor.text(), "ACDEFGHIKLMNPQRSTVWY")
        self.window.tabs.setCurrentIndex(0)
        self.window.tabs.setCurrentIndex(tab_index)
        self.flush()
        self.assertEqual(self.window._collect_tool_settings(path), before)


if __name__ == "__main__":
    unittest.main()
