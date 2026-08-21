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
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from SSN_Tools import SpacedTipLabel, ToolsGUI  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
