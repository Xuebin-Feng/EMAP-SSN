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

from PySide6.QtGui import QTextDocumentFragment

from desktop.Desktop_App import ToggleSwitch, fit_buttons_to_text, mark_name_item
from utilities.Localization import display_text, is_pseudo_translated, pseudo_translate
from tests.translation_fixtures import cut_off_texts, outside_the_catalog, pseudo_language, visible_texts

# Windows whose every text comes from a catalog. Marking a window's text
# (language step 6) moves it here.
FULLY_MARKED = frozenset({"Config", "Tools"})


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
        choices.addItems(["A choice", "42", "NVIDIA GeForce [cuda:0]"])
        mark_name_item(choices, 2)
        edit = QLineEdit("typed by the user")
        edit.setPlaceholderText("A placeholder")
        spin = QSpinBox()
        spin.setSuffix(" nodes")
        rich = QLabel('<div style="line-height: 120%;">A rich label &amp; more</div>')
        switch = ToggleSwitch("Shown when on", "Shown when off")
        switch.setAccessibleName("A spoken name")
        switch.setAccessibleDescription("A spoken description")
        for widget in (label, QPushButton("A button"), QCheckBox("A check box"), group, tabs,
                       choices, edit, spin, QLabel("12.5"), QPushButton("📁"), QPushButton(">>"), rich, switch):
            layout.addWidget(widget)
        self.assertEqual(sorted(text for _, text in visible_texts(window)), sorted([
            "Window title", "A label", "A tooltip", "A button", "A check box", "A group",
            "A tab", "A choice", "A placeholder", " nodes", "A rich label & more",
            "Shown when off", "Shown when on", "A spoken name", "A spoken description",
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
        self.assertEqual(self.cut_off_on_every_tab(window), [], f"{name} cuts translated text off")

    def cut_off_on_every_tab(self, window):
        """cut_off_texts(window), with each tab of each tab widget shown in turn."""
        found = cut_off_texts(window)
        for tabs in window.findChildren(QTabWidget):
            for index in range(tabs.count()):
                if tabs.isTabVisible(index):
                    tabs.setCurrentIndex(index)
                    flush(self.app)
                    found += [text for text in cut_off_texts(window) if text not in found]
        return found

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


class ConfigRunTimeTextTests(WindowTestCase):
    """Text the Config window shows after it opens: tips, reports, messages and errors.

    Each is checked under the pseudo-language: what is left once every
    bracketed, translated piece is taken out came from no catalog. A value
    filled into a translated text, such as a file name, is inside its piece.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tests.config_gui_loader import load_config_namespace

        cls.namespace = load_config_namespace()

    def setUp(self):
        from tests.config_gui_loader import open_config_window

        pseudo_language(self, self.app)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.window = open_config_window(self.namespace["ConfigGUI"], folder.name)
        self.addCleanup(self.window.deleteLater)

    def assert_translated(self, text):
        self.assertTrue(text)
        self.assertEqual(outside_the_catalog(text), "", text)

    def test_every_tip(self):
        self.assertGreater(len(self.window.tip_db_keys), 50)
        for key, tip in self.window.tip_db_keys.items():
            with self.subTest(key=key):
                self.assertTrue(is_pseudo_translated(tip), tip)

    def test_the_save_message(self):
        self.assert_translated(self.window._save_success_message(
            [("visual_effects", "mine"), ("directories", "shared")], ["simulation_physics"]
        ))

    def test_profile_errors(self):
        validate = self.namespace["_validate_profile_name"]
        for name, existing in (("", ()), ("(new)", ()), ("name.", ()), ("a/b", ()), ("con", ()),
                               ("taken", ("taken",))):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                validate(name, existing)
            self.assert_translated(display_text(caught.exception))
        for tab_id, data in (("visual_effects", []), ("visual_effects", {"BOGUS": 1}),
                             ("visual_effects", {"NODE_SIZE": 10.5}), ("visual_effects", {"TEXT_COLOR": "nocolor"}),
                             ("visual_effects", {"NODE_SIZE": 1e999}), ("simulation_physics", {"PACKING_GEOMETRY": "Hex"}),
                             ("inputs_outputs", {"ALIGNMENT_SCORE": "local", "NORM_MODE": "alignment_length"})):
            with self.subTest(data=data), self.assertRaises(ValueError) as caught:
                self.window._normalize_profile_data(tab_id, data)
            self.assert_translated(display_text(caught.exception))

    def write_inputs(self, network_headers):
        import h5py
        import numpy

        (self.folder / "subset.fasta").write_text(
            ">WP_1_alpha\nMKTA\n>WP_2_beta\nMSEQ\n>Other\nMAAA\n", encoding="utf-8"
        )
        with h5py.File(self.folder / "network.h5", "w") as hf:
            hf.create_dataset("headers", data=[header.encode() for header in network_headers])
            hf.create_dataset("score", data=numpy.asarray([4.0], dtype=numpy.float32))
            hf.create_dataset("i", data=numpy.asarray([0], dtype=numpy.int64))
            hf.create_dataset("j", data=numpy.asarray([1], dtype=numpy.int64))
        (self.folder / "alignment.fasta").write_text(">WP_1_alpha\nMKTA\n", encoding="utf-8")
        for key in ("FASTA_DIR", "HDF5_DIR", "MSA_DIR"):
            self.window.inputs[key].blockSignals(True)
            self.window.inputs[key].setText(str(self.folder))
            self.window.inputs[key].blockSignals(False)
        for combo, name in ((self.window.cb_fasta, "subset.fasta"), (self.window.cb_hdf5, "network.h5"),
                            (self.window.cb_msa, "alignment.fasta")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(name)
            combo.blockSignals(False)

    def test_the_consistency_check_report(self):
        self.write_inputs(["WP_1_alpha", "WP_2_beta"])
        self.window.line_ref.setText("WP_*")
        self.window.run_consistency_check()
        report = QTextDocumentFragment.fromHtml(self.window.tip_panel.text()).toPlainText()
        # Every part of the report: the network and MSA checks with their
        # missing headers, then the reference and the headers it ties with.
        self.assertEqual(report.count("\n\n"), 2, report)
        for header in ("Other", "WP_2_beta", "WP_1_alpha"):
            self.assertIn(header, report)
        self.assert_translated(report)

    def test_the_statistics_report(self):
        from types import SimpleNamespace

        self.write_inputs(["WP_1_alpha", "WP_2_beta", "Other"])
        cache_manifest = self.window.run_statistics.__globals__["cache_manifest"]
        blast = SimpleNamespace(network_type="blast", model_name="BLAST")
        with mock.patch.object(cache_manifest, "validate_network_schema", return_value=blast):
            self.window.run_statistics()
        report = self.window.stat_display.toPlainText()
        self.assertGreater(len(report.splitlines()), 10, report)
        self.assert_translated(report)
        self.assert_translated(QTextDocumentFragment.fromHtml(self.window.tip_panel.text()).toPlainText())


class ToolsRunTimeTextTests(WindowTestCase):
    """Text the Tools window shows after it opens: tips, help, messages and errors."""

    def setUp(self):
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        pseudo_language(self, self.app)
        isolated_tools_project(self)
        patcher = mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tools = EMAPSSN_Tools
        self.window = EMAPSSN_Tools.ToolsGUI()
        self.addCleanup(self.window.deleteLater)

    def assert_translated(self, text):
        self.assertTrue(text)
        self.assertEqual(outside_the_catalog(text), "", text)

    def test_every_tool_tip(self):
        tips = [tip for script in self.window.SCRIPT_TIPS.values() for tip in script.values()]
        self.assertGreater(len(tips), 90)
        for tip in tips:
            with self.subTest(tip=tip[:40]):
                self.assertTrue(is_pseudo_translated(tip), tip)

    def test_the_network_switch_tips_in_every_state(self):
        from Cache_Manifest import NetworkCompletenessInfo

        states = [
            None,
            NetworkCompletenessInfo("unknown"),
            NetworkCompletenessInfo("complete", 4, 6, 6),
            NetworkCompletenessInfo("incomplete", 4000, 1500, 7998000),
        ]
        for info in states:
            for noise, checked in ((True, True), (True, False), (False, False)):
                with self.subTest(info=info, noise=noise, checked=checked):
                    self.assert_translated(self.tools.imputed_consensus_switch_state(info, noise, checked)[1])
            for blast in (True, False):
                with self.subTest(info=info, blast=blast):
                    self.assert_translated(self.tools.isotonic_regression_switch_state(info, blast)[1])

    def test_export_name_errors(self):
        for name in ("", ".", "a/b", "name.", "con"):
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                self.window._normalized_export_filename(name)
            self.assert_translated(display_text(caught.exception))

    def test_the_help_shown_for_the_directories_and_without_a_help_file(self):
        directories = self.window.tab_paths.index("DIRECTORIES_TAB")
        self.window.on_tab_changed(directories)
        help_text = self.window.script_desc_text.toPlainText()
        self.assert_translated(help_text)
        self.assertNotIn("##", help_text, "the Markdown heading is shown as a heading")
        # Without a help file, the tool's docstring shows under a heading, or,
        # without one, where a help file would go.
        self.window.tabs.widget(0).setProperty("descriptionKey", "No_Such_Help")
        self.window.on_tab_changed(0)
        heading = self.window.script_desc_text.toPlainText().splitlines()[0]
        self.assertIn("📄", heading)
        self.assert_translated(heading)
        self.window.script_data[self.window.tab_paths[0]]["docstring"] = ""
        self.window.on_tab_changed(0)
        help_text = self.window.script_desc_text.toPlainText()
        missing_file = os.path.join("src", "tools", "tool_descriptions", "No_Such_Help.md")
        self.assertIn(missing_file, help_text)
        self.assert_translated(help_text.replace(missing_file, ""))

    def test_the_license_dialog(self):
        from PySide6.QtWidgets import QMessageBox

        class RecordingBox:
            """Stands in for the modal QMessageBox and keeps the texts it was given."""

            Icon, ButtonRole, StandardButton = QMessageBox.Icon, QMessageBox.ButtonRole, QMessageBox.StandardButton
            texts = []

            def __init__(self, parent):
                pass

            def setWindowTitle(self, text):
                self.texts.append(text)

            setText = setInformativeText = setWindowTitle

            def addButton(self, button, role=None):
                if isinstance(button, str):
                    self.texts.append(button)
                return object()

            def setIcon(self, icon):
                pass

            setDefaultButton = setIcon

            def exec(self):
                return 0

            def clickedButton(self):
                return None  # Cancelled.

        terms = {"license_id": "X-1", "restriction": "Research use", "source_url": "https://example.org/m",
                 "license_url": "https://example.org/l"}
        with mock.patch.object(self.tools, "QMessageBox", RecordingBox), \
                mock.patch.object(self.tools, "is_model_license_accepted", return_value=False):
            self.assertFalse(self.tools.confirm_model_usage_terms(None, "model_a", terms))
        self.assertEqual(len(RecordingBox.texts), 5, RecordingBox.texts)
        for text in RecordingBox.texts:
            self.assert_translated(text)
        self.assertIn("Weights license: X-1", str(self.tools.model_usage_terms_message("model_a", terms)))

    def test_file_names_in_folder_dropdowns_show_as_names(self):
        fasta_dir = Path(self.window.dir_inputs["FASTA_DIR"].text())
        fasta_dir.mkdir(parents=True, exist_ok=True)
        (fasta_dir / "my_sequences.fasta").write_text(">a\nMK\n", encoding="utf-8")
        # Opening a folder's dropdown lists it again from the Directories tab.
        for combo in self.window.findChildren(self.tools.DynamicComboBox):
            combo.populate()
        combos = [combo for combo in self.window.findChildren(QComboBox) if combo.findText("my_sequences.fasta") >= 0]
        self.assertTrue(combos)
        self.assertNotIn("my_sequences.fasta", [text for _, text in visible_texts(self.window)])

    def test_card_titles_are_the_headings_of_the_tool_help_files(self):
        headings = self.tools.get_tool_titles()
        self.assertEqual(len(headings), 14)
        self.assertEqual(self.tools.TOOL_TITLES, headings)


if __name__ == "__main__":
    unittest.main()
