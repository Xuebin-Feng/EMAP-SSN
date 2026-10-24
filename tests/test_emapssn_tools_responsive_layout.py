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

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QTextBrowser, QWidget
from EMAPSSN_Tools import ToolsGUI, ResponsiveFieldLayout, configure_qt_application_fonts
from tests.tools_gui_fixtures import isolated_tools_project


class ResponsiveToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        configure_qt_application_fonts(cls.app)

    def setUp(self):
        # Saved values and directories come from a temporary tools_settings.json.
        isolated_tools_project(self)
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
        # The side panel is as wide as its bottom row (the Exit button, the Console button and
        # the language selector) allows, at least 326 px.
        side = max(326, self.window.right_widget.minimumSizeHint().width())
        self.window.resize(width + side + 34, 850)
        # QTabWidget adds four pixels outside its scroll page.
        self.window.splitter.setSizes([width + 4, side])
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

    def test_sanitize_puts_the_length_filter_on_its_bounds_row(self):
        # Row 2: Show Length Histogram, then Remove Header Substring. Row 3:
        # Enable Length Filter, then Min and Max Seq Length, which it turns on.
        path, data = next((path, data) for path, data in self.window.script_data.items()
                          if path.endswith("Sanitize_Sequences.py"))
        inputs = data["inputs"]
        self.window.tabs.setCurrentIndex(0)
        self.resize_panel(1400)
        rows = [self.window.findChild(QWidget, "compactRow_" + name) for name in (
            "INPUT_FASTA_OVER_WRITE",
            "SHOW_LENGTH_HISTOGRAM_REMOVE_BY_HEADER_STRING",
            "ENABLE_LENGTH_FILTER_MIN_SEQ_LENGTH_MAX_SEQ_LENGTH",
        )]
        self.assertNotIn(None, rows)
        self.assertIsNone(self.window.findChild(QWidget, "compactRow_MIN_SEQ_LENGTH_MAX_SEQ_LENGTH"))
        card = rows[0].parentWidget()
        tops = [row.mapTo(card, QPoint(0, 0)).y() for row in rows]
        self.assertEqual(tops, sorted(tops))
        for row in rows:
            self.assertFalse(row.property("stacked"), row.objectName())
        filter_row = rows[2].layout().pairs
        self.assertEqual([control for _, control in filter_row],
                         [inputs[name]["widget"] for name in
                          ("ENABLE_LENGTH_FILTER", "MIN_SEQ_LENGTH", "MAX_SEQ_LENGTH")])
        xs = [control.mapTo(card, QPoint(0, 0)).x() for _, control in filter_row]
        self.assertEqual(xs, sorted(xs))
        # The second column starts at the same x on both switch rows.
        second_labels = [row.layout().pairs[1][0] for row in rows[1:]]
        self.assertEqual(len({label.mapTo(card, QPoint(0, 0)).x() for label in second_labels}), 1)
        # The histogram is on by default, and the filter still turns its bounds on and off.
        self.assertTrue(inputs["SHOW_LENGTH_HISTOGRAM"]["widget"].isChecked())
        switch = inputs["ENABLE_LENGTH_FILTER"]["widget"]
        for checked in (True, False):
            switch.setChecked(checked)
            for name in ("MIN_SEQ_LENGTH", "MAX_SEQ_LENGTH"):
                self.assertEqual(inputs[name]["widget"].isEnabled(), checked, name)
        self.assert_layout_geometry(self.window.tabs.currentWidget().widget())

    def test_exit_fills_the_row_under_the_help_and_the_tip_panel_reaches_the_bottom(self):
        window = self.window

        def box(widget):
            return widget.rect().translated(widget.mapTo(window, QPoint(0, 0)))

        # The help panel at its narrowest, then wider.
        for help_width in (None, 560):
            self.resize_panel(1000)
            if help_width:
                left_width, right_width = window.splitter.sizes()
                window.splitter.setSizes([left_width + right_width - help_width, help_width])
                self.flush()
            exit_button, language, help_card = (
                box(widget) for widget in (window.btn_exit, window.language_selector, window.script_desc_text)
            )
            tip, left = box(window.tip_panel), box(window.left_widget)
            globe = box(window.findChild(QWidget, "languageGlobe"))
            console = box(window.findChild(QWidget, "consoleButton"))
            with self.subTest(help_width=help_width):
                self.assertTrue(window.right_widget.isAncestorOf(window.btn_exit))
                self.assertEqual(exit_button.left(), help_card.left())
                self.assertGreater(exit_button.top(), help_card.bottom())
                # Exit takes all the room up to the Console button, which sits before the globe and the dropdown.
                self.assertLess(exit_button.right(), console.left())
                self.assertLessEqual(console.left() - exit_button.right(), 12)
                self.assertLess(console.right(), globe.left())
                self.assertLessEqual(globe.left() - console.right(), 12)
                if help_width:
                    self.assertGreater(exit_button.width(), window.btn_exit.minimumSizeHint().width() + 100)
                self.assertAlmostEqual(exit_button.center().y(), language.center().y(), delta=1)
                # Nothing below the tip panel: it ends where the left side does, level with the row.
                self.assertEqual(tip.bottom(), left.bottom())
                self.assertGreaterEqual(tip.bottom(), exit_button.bottom())
        window.btn_exit.click()
        self.flush()
        self.assertFalse(window.isVisible())


if __name__ == "__main__":
    unittest.main()
