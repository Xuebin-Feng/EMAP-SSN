# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QHelpEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QFormLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from SSN_Tools import (  # noqa: E402
    SpacedTipLabel,
    ToolsGUI,
    _configure_linux_qtwebengine_rendering,
    bind_custom_blast_column_controls,
)


class TooltipRoutingWindow(ToolsGUI):
    def __init__(self):
        QMainWindow.__init__(self)
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.native_tip_button = QPushButton("Native tooltip", central_widget)
        self.native_tip_button.setToolTip("Help from the widget tooltip.")
        self.shared_tip_label = QLabel("Shared tooltip", central_widget)
        layout.addWidget(self.native_tip_button)
        layout.addWidget(self.shared_tip_label)
        self.setCentralWidget(central_widget)

        self.tip_panel = SpacedTipLabel("Initial help")
        self.tip_db = {self.shared_tip_label: "Help from the shared database."}
        self.shared_tip_label.installEventFilter(self)
        self._route_native_tooltips_to_tip_panel()


class TooltipRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = TooltipRoutingWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_linux_webengine_software_rendering_preserves_existing_flags(self):
        environment = {"QTWEBENGINE_CHROMIUM_FLAGS": "--disable-logging"}

        changed = _configure_linux_qtwebengine_rendering(environment, "linux")

        self.assertTrue(changed)
        self.assertEqual(
            environment["QTWEBENGINE_CHROMIUM_FLAGS"],
            "--disable-logging --disable-gpu",
        )
        self.assertFalse(
            _configure_linux_qtwebengine_rendering(environment, "linux")
        )

    def test_webengine_rendering_is_unchanged_outside_linux(self):
        for platform_name in ("darwin", "win32"):
            with self.subTest(platform_name=platform_name):
                environment = {
                    "QTWEBENGINE_CHROMIUM_FLAGS": "--disable-logging"
                }

                changed = _configure_linux_qtwebengine_rendering(
                    environment, platform_name
                )

                self.assertFalse(changed)
                self.assertEqual(
                    environment["QTWEBENGINE_CHROMIUM_FLAGS"],
                    "--disable-logging",
                )

    def test_native_tooltip_event_updates_bottom_panel_and_is_suppressed(self):
        event = QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1))

        QApplication.sendEvent(self.window.native_tip_button, event)

        self.assertIn(
            "Help from the widget tooltip.",
            self.window.tip_panel.text(),
        )
        self.assertTrue(
            self.window.eventFilter(
                self.window.native_tip_button,
                QHelpEvent(QEvent.Type.ToolTip, QPoint(1, 1), QPoint(1, 1)),
            )
        )

    def test_shared_database_tip_still_updates_bottom_panel(self):
        event = QEvent(QEvent.Type.Enter)

        handled = self.window.eventFilter(self.window.shared_tip_label, event)

        self.assertFalse(handled)
        self.assertIn(
            "Help from the shared database.",
            self.window.tip_panel.text(),
        )

    def test_custom_blast_columns_follow_layout_selection(self):
        layout_combo = QComboBox()
        layout_combo.addItem("standard_outfmt6", "standard_outfmt6")
        layout_combo.addItem("outfmt7_fields", "outfmt7_fields")
        layout_combo.addItem(
            "Custom Columns (1-based indexing)", "custom_columns"
        )
        layout_combo.setProperty("persistItemData", True)
        inputs = {"BLAST_LAYOUT": {"widget": layout_combo}}
        row_widgets = {}
        for name in ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"):
            label = QLabel(name)
            widget = QSpinBox()
            inputs[name] = {"widget": widget}
            row_widgets[name] = (label, widget)

        bind_custom_blast_column_controls(inputs, row_widgets)

        self.assertTrue(
            all(not inputs[name]["widget"].isEnabled() for name in row_widgets)
        )
        self.assertTrue(all(not label.isEnabled() for label, _ in row_widgets.values()))
        layout_combo.setCurrentText("Custom Columns (1-based indexing)")
        self.app.processEvents()
        self.assertEqual(layout_combo.currentData(), "custom_columns")
        self.assertTrue(
            all(inputs[name]["widget"].isEnabled() for name in row_widgets)
        )
        self.assertTrue(all(label.isEnabled() for label, _ in row_widgets.values()))

        self.window.script_data = {
            "parse": {
                "inputs": {
                    "BLAST_LAYOUT": {
                        "widget": layout_combo,
                        "type": "dropdown",
                    }
                },
                "settings": [{"name": "BLAST_LAYOUT", "def": {}}],
            }
        }
        self.assertEqual(
            self.window._collect_tool_settings("parse")["BLAST_LAYOUT"],
            "custom_columns",
        )

    def test_custom_blast_column_controls_are_merged_into_one_row(self):
        container = QWidget()
        form = QFormLayout(container)
        row_widgets = {}
        for name in ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"):
            label = QLabel(name)
            widget = QSpinBox()
            form.addRow(label, widget)
            row_widgets[name] = (label, widget)

        ToolsGUI._merge_inline_field_rows(
            form, "Parse_BLAST_Output.py", row_widgets
        )

        self.assertEqual(form.rowCount(), 1)
        field = form.itemAt(0, QFormLayout.ItemRole.FieldRole).widget()
        self.assertEqual(field.property("compactColumnRatio"), "1:1:1")


if __name__ == "__main__":
    unittest.main()
