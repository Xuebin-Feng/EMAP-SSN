# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Text not yet marked for translation, found with the pseudo-language.

Each window opens under the pseudo-language made from the real catalog
(src/resources/languages/emapssn.ts). Text that came from the catalog shows
bracketed, so whatever shows unbracketed was never marked for translation.
Until a window's text is marked, all of it shows that way. Once none is
left, the window joins FULLY_MARKED, and from then on a new unmarked text
fails its test. Text that is translated must also show whole.

The Viewer is covered by its side panel, which its web plugins fill.
Its canvas text (the HUD and the console line) isn't Qt widgets; the
console line's Message texts are checked in test_translation_loading.
"""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from desktop.Desktop_App import fit_buttons_to_text
from utilities.Localization import is_pseudo_translated, pseudo_translate
from tests.translation_fixtures import cut_off_texts, pseudo_language, visible_texts

# Windows whose every text comes from a catalog. Marking a window's text
# (language step 6) moves it here.
FULLY_MARKED = frozenset()


def flush(app):
    for _ in range(3):
        app.processEvents()


class WindowTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def show(self, window):
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        window.resize(1800, 1000)
        window.show()
        flush(self.app)
        return window


class VisibleTextTests(WindowTestCase):
    def test_lists_every_kind_of_text_and_leaves_out_the_users_data(self):
        window = QWidget()
        window.setWindowTitle("Window title")
        layout = QVBoxLayout(window)
        label = QLabel("A label")
        label.setToolTip("A tooltip")
        group = QGroupBox("A group")
        tabs = QTabWidget()
        tabs.addTab(QWidget(), "A tab")
        choices = QComboBox()
        choices.addItems(["A choice", "42"])
        edit = QLineEdit("typed by the user")
        edit.setPlaceholderText("A placeholder")
        spin = QSpinBox()
        spin.setSuffix(" nodes")
        for widget in (label, QPushButton("A button"), QCheckBox("A check box"), group, tabs,
                       choices, edit, spin, QLabel("12.5"), QPushButton("📁"), QPushButton(">>")):
            layout.addWidget(widget)
        self.assertEqual(sorted(text for _, text in visible_texts(window)), sorted([
            "Window title", "A label", "A tooltip", "A button", "A check box", "A group",
            "A tab", "A choice", "A placeholder", " nodes",
        ]))


class CutOffTextTests(WindowTestCase):
    def test_finds_translated_text_too_wide_for_its_button_or_label(self):
        window = QWidget()
        layout = QVBoxLayout(window)
        text = pseudo_translate("Consistency Check")
        narrow_button, fitted_button = QPushButton(text), QPushButton(text)
        narrow_button.setFixedWidth(30)
        fit_buttons_to_text(fitted_button)
        narrow_label, natural_label = QLabel(text), QLabel(text)
        narrow_label.setFixedWidth(30)
        english = QPushButton("Consistency Check")
        english.setFixedWidth(30)
        for widget in (narrow_button, fitted_button, narrow_label, natural_label, english):
            layout.addWidget(widget)
        self.show(window)
        self.assertEqual(cut_off_texts(window), [("QPushButton text", text), ("QLabel text", text)])


class WindowTextTests(WindowTestCase):
    def setUp(self):
        pseudo_language(self, self.app)

    def check(self, name, window):
        texts = visible_texts(window)
        self.assertTrue(texts, f"{name} shows no text at all")
        unmarked = [(where, text) for where, text in texts if not is_pseudo_translated(text)]
        if name in FULLY_MARKED:
            self.assertEqual(unmarked, [], f"{name} shows text not marked for translation")
        else:
            self.assertTrue(unmarked, f"Every text in {name} is marked now: add it to FULLY_MARKED.")
        self.assertEqual(cut_off_texts(window), [], f"{name} cuts translated text off")

    def test_config_window(self):
        from tests.config_gui_loader import load_config_namespace, open_config_window

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        window = open_config_window(load_config_namespace()["ConfigGUI"], folder.name)
        self.check("Config", self.show(window))

    def test_tools_window(self):
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        isolated_tools_project(self)
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = EMAPSSN_Tools.ToolsGUI()
        self.check("Tools", self.show(window))

    def test_viewer_side_panel(self):
        from EMAPSSN_Viewer import MainViewer
        from web_ui import agent_backend, esmfold_backend, meta_backend

        viewer = MainViewer.__new__(MainViewer)
        viewer.main_window = QMainWindow()
        viewer.right_panel = QWidget()
        viewer.right_panel.setObjectName("rightPanel")
        viewer.main_window.setCentralWidget(viewer.right_panel)
        viewer.right_panel_layout = QVBoxLayout(viewer.right_panel)
        viewer.right_panel_layout.addStretch()
        viewer.set_sidebar_visible = lambda visible: None
        viewer.open_agent_ui = viewer.open_metadata_ui = lambda: None
        with contextlib.redirect_stdout(io.StringIO()):
            for plugin in (agent_backend, esmfold_backend, meta_backend):
                plugin.activate(viewer)
        self.assertEqual(len(viewer.sidebar_buttons), 3)
        self.check("Viewer side panel", self.show(viewer.main_window))


if __name__ == "__main__":
    unittest.main()
