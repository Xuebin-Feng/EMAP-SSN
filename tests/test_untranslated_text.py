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

The Viewer is covered by its window: the title, the sidebar its web
plugins fill, and the sidebar's toggle. Its canvas text (the HUD, the
console line and the background-job status) isn't Qt widgets. Each file
that writes it is read instead, and a file whose canvas text is all marked
joins CANVAS_MARKED; ViewerCanvasTextTests shows the texts put together
from several parts under the pseudo-language.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import re
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
from tests.translation_fixtures import (
    cut_off_texts,
    outside_the_catalog,
    pseudo_language,
    shows_canvas_text,
    unmarked_canvas_texts,
    unmarked_dialog_texts,
    visible_texts,
)

# Windows whose every text comes from a catalog. Marking a window's text
# (language step 6) moves it here.
FULLY_MARKED = frozenset({"Config", "Tools", "Viewer window"})

# Files under src whose every text on the Viewer's canvas comes from a
# catalog. Marking a file's console messages moves it here.
CANVAS_MARKED = frozenset({
    "Background_Job_Scheduler.py",
    "Command_Engine.py",
    "EMAPSSN_Viewer.py",
    "Metadata_Core.py",
    "commands/agent.py",
    "commands/alignment.py",
    "commands/cluster.py",
    "commands/color.py",
    "commands/esmfold.py",
    "commands/export.py",
    "commands/group.py",
    "commands/hide.py",
    "commands/label.py",
    "commands/logo.py",
    "commands/meta.py",
    "commands/offset.py",
    "commands/print.py",
    "commands/query.py",
    "commands/redo.py",
    "commands/reference.py",
    "commands/reset.py",
    "commands/run.py",
    "commands/save.py",
    "commands/select.py",
    "commands/spectrum.py",
    "commands/subcluster.py",
    "commands/undo.py",
    "commands/zoom.py",
    "web_ui/Browser_Page.py",
    "web_ui/agent_backend.py",
    "web_ui/meta_backend.py",
})


def flush(app):
    for _ in range(3):
        app.processEvents()


def open_viewer_window(test_case):
    """A Viewer's main window, built as the Viewer builds it, around a stand-in canvas.

    Its sidebar holds the buttons the web plugins add. No network is loaded,
    so the canvas draws nothing.
    """
    from types import SimpleNamespace

    from EMAPSSN_Viewer import MainViewer
    from web_ui import agent_backend, esmfold_backend, meta_backend

    viewer = MainViewer.__new__(MainViewer)
    viewer.canvas = SimpleNamespace(native=QWidget(), size=(1800, 1000))
    viewer._build_main_window()
    test_case.addCleanup(viewer.main_window.deleteLater)
    # The real one also moves the slider and the HUD, which need a network.
    viewer.set_sidebar_visible = lambda visible: None
    with contextlib.redirect_stdout(io.StringIO()):
        for plugin in (agent_backend, esmfold_backend, meta_backend):
            plugin.activate(viewer)
    test_case.assertEqual(len(viewer.sidebar_buttons), 3)
    viewer.reposition_expand_btn()
    return viewer


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

    def test_viewer_window(self):
        viewer = open_viewer_window(self)
        self.check("Viewer window", self.show(viewer.main_window))


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

    def test_a_saved_file_missing_from_its_folder_and_its_refusal(self):
        settings = Path(self.tools._PROJECT_ROOT) / "tools_settings.json"
        # A file the tools find by its full path shows as a name, though no folder lists it.
        full_path = settings.parent / "elsewhere" / "by_full_path.fasta"
        full_path.parent.mkdir()
        full_path.write_text(">a\nMK\n", encoding="utf-8")
        document = json.loads(settings.read_text(encoding="utf-8"))
        document["Sanitize_Sequences.py"] = {"INPUT_FASTA": "gone.fasta"}
        document["Generate_Embeddings.py"] = {"INPUT_FASTA": str(full_path)}
        settings.write_text(json.dumps(document), encoding="utf-8")
        window = self.tools.ToolsGUI()
        self.addCleanup(window.deleteLater)
        path, data = next(
            (path, data) for path, data in window.script_data.items() if Path(path).name == "Sanitize_Sequences.py"
        )

        shown = data["inputs"]["INPUT_FASTA"]["widget"].combo.currentText()
        self.assertIn("gone.fasta", shown)
        self.assert_translated(shown)
        self.assertNotIn(str(full_path), [text for _, text in visible_texts(window)])
        with mock.patch.object(self.tools.QMessageBox, "critical") as critical:
            window.save_and_run(path)
        title, message = critical.call_args.args[1:]
        self.assert_translated(title)
        self.assertIn("gone.fasta", message)
        self.assert_translated(message)

    def test_card_titles_are_the_headings_of_the_tool_help_files(self):
        headings = self.tools.get_tool_titles()
        self.assertEqual(len(headings), 14)
        self.assertEqual(self.tools.TOOL_TITLES, headings)


class ViewerRunTimeTextTests(WindowTestCase):
    """Text the Viewer's window shows after it opens: its dialogs and the
    messages that opening a browser page puts on the console line."""

    def setUp(self):
        pseudo_language(self, self.app)

    def assert_translated(self, text):
        self.assertTrue(text)
        self.assertEqual(outside_the_catalog(text), "", text)

    def test_the_title_shows_translated_and_mcp_reads_it_in_english(self):
        from desktop.Viewer_Inspection import english_window_title

        viewer = open_viewer_window(self)
        alias = viewer.inspection_session_alias
        title = viewer.main_window.windowTitle()
        self.assertTrue(title.endswith(f" [{alias}]"), title)
        self.assertNotIn("Viewer", title)
        self.assert_translated(title)
        self.assertEqual(english_window_title(viewer), f"EMAP-SSN Viewer [{alias}]")

    def test_the_browser_page_messages(self):
        from types import SimpleNamespace

        from EMAPSSN_Viewer import MainViewer
        from web_ui import Browser_Page, Web_Server, esmfold_backend

        class WebServer:
            def __init__(self, connected):
                self.connected = connected

            def has_event_client(self, client_id):
                return client_id in self.connected

        def viewer_with(connected=(), url="http://127.0.0.1:49123"):
            viewer = MainViewer.__new__(MainViewer)
            viewer.console_text = SimpleNamespace(text="")
            viewer.main_window = None
            viewer.update_console_background = lambda: None
            viewer.web_server = WebServer(set(connected))
            viewer.web_server_url = url
            return viewer

        dialogs = []
        shown = []

        def show(viewer):
            shown.append(viewer.console_text.text)

        with mock.patch.object(Browser_Page.QMessageBox, "information",
                               side_effect=lambda parent, title, text: dialogs.extend([title, text])), \
                mock.patch.object(Web_Server, "is_running", return_value=True):
            for open_page in (MainViewer.open_agent_ui, MainViewer.open_metadata_ui, esmfold_backend.open_esmfold_ui):
                viewer = viewer_with()
                with mock.patch.object(Browser_Page.webbrowser, "open", return_value=True):
                    self.assertTrue(open_page(viewer))
                    show(viewer)  # Opened.
                    self.assertFalse(open_page(viewer))
                    show(viewer)  # Still being opened.
            viewer = viewer_with(connected={"agent"})
            self.assertFalse(viewer.open_agent_ui())
            show(viewer)  # Already open.
            viewer = viewer_with(url=None)
            self.assertFalse(viewer.open_agent_ui())
            show(viewer)  # The web server is unavailable.
            for outcome in ({"return_value": False}, {"side_effect": OSError(5)}):
                viewer = viewer_with()
                with mock.patch.object(Browser_Page.webbrowser, "open", **outcome):
                    self.assertFalse(viewer.open_agent_ui())
                show(viewer)  # Could not open it.
        with mock.patch.object(Web_Server, "is_running", return_value=False), \
                mock.patch.object(Web_Server, "ensure_server", side_effect=OSError(98)):
            viewer = viewer_with()
            self.assertFalse(viewer.open_agent_ui())
            show(viewer)  # The web server could not start again.
        self.assertEqual(len(shown), 11, shown)
        self.assertEqual(len(dialogs), 8, dialogs)
        for text in shown + dialogs:
            with self.subTest(text=text):
                self.assert_translated(text)

    def test_the_parts_filled_into_the_browser_page_messages(self):
        # A value filled into a translated text shows inside its brackets, so
        # the page's name and the web server's errors are checked on their own.
        from types import SimpleNamespace

        from EMAPSSN_Viewer import MainViewer
        from web_ui import Web_Server, esmfold_backend

        for open_page in (MainViewer.open_agent_ui, MainViewer.open_metadata_ui, esmfold_backend.open_esmfold_ui):
            viewer = SimpleNamespace(_open_web_ui=mock.Mock(return_value=False))
            open_page(viewer)
            name = display_text(viewer._open_web_ui.call_args.args[1])
            with self.subTest(name=name):
                self.assertTrue(is_pseudo_translated(name), name)
        viewer = MainViewer.__new__(MainViewer)
        viewer.web_server, viewer.web_server_url = object(), None
        for running in (True, False):
            with mock.patch.object(Web_Server, "is_running", return_value=running), \
                    mock.patch.object(Web_Server, "ensure_server", side_effect=OSError(98)), \
                    self.assertRaises(RuntimeError) as caught:
                viewer.get_web_url("/agent.html")
            with self.subTest(running=running):
                self.assert_translated(display_text(caught.exception))

    def test_the_metadata_file_dialogs(self):
        from types import SimpleNamespace

        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog, QFileDialog

        from web_ui import meta_backend

        texts = []

        class RecordingFileDialog:
            """Stands in for the modal QFileDialog and keeps the texts it was given."""

            FileMode, AcceptMode = QFileDialog.FileMode, QFileDialog.AcceptMode

            def __init__(self, parent):
                pass

            def setWindowTitle(self, text):
                texts.append(text)

            setNameFilter = setWindowTitle

            def setNameFilters(self, filters):
                texts.extend(filters)

            def windowFlags(self):
                return Qt.WindowType.Dialog

            def exec(self):
                return QDialog.DialogCode.Rejected  # Cancelled.

            def __getattr__(self, name):
                return lambda *args: None

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        viewer = SimpleNamespace(main_window=mock.Mock())
        with mock.patch.object(meta_backend.QtWidgets, "QFileDialog", RecordingFileDialog), \
                mock.patch.object(meta_backend.cfg, "METADATA_DIR", folder.name, create=True):
            meta_backend.handle_import_metadata(viewer, {})
            meta_backend.handle_export_metadata(viewer, {})
        self.assertEqual(len(texts), 5, texts)
        for text in texts:
            with self.subTest(text=text):
                self.assert_translated(text)


class CanvasTextTests(unittest.TestCase):
    """What each file puts on the Viewer's canvas, read from its code
    (translation_fixtures.unmarked_canvas_texts)."""

    def test_every_canvas_text_of_a_marked_file_comes_from_a_catalog(self):
        shown_by = set()
        for path in sorted(SRC.rglob("*.py")):
            relative = path.relative_to(SRC).as_posix()
            source = path.read_text(encoding="utf-8")
            if "__pycache__" in path.parts or not shows_canvas_text(source):
                continue
            shown_by.add(relative)
            unmarked = unmarked_canvas_texts(source)
            with self.subTest(file=relative):
                if relative in CANVAS_MARKED:
                    self.assertEqual(unmarked, [], f"{relative} shows canvas text not marked for translation")
                else:
                    self.assertTrue(unmarked, f"Every canvas text in {relative} is marked now: add it to CANVAS_MARKED.")
        self.assertLessEqual(CANVAS_MARKED, shown_by, "CANVAS_MARKED names a file that shows no canvas text")

    def test_a_marked_file_never_reads_the_console_line_back(self):
        # The console line shows the translation, so a report or print of what it
        # shows would give the terminal and MCP clients translated text.
        import ast

        for relative in sorted(CANVAS_MARKED - {"EMAPSSN_Viewer.py"}):  # The Viewer draws the line.
            tree = ast.parse((SRC / relative).read_text(encoding="utf-8"))
            reads = [
                node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "text" and isinstance(node.ctx, ast.Load)
                and "console_text" in (getattr(node.value, "attr", None), getattr(node.value, "id", None))
            ]
            with self.subTest(file=relative):
                self.assertEqual(reads, [], f"{relative} reads the console line back")

    def test_a_marked_file_opens_no_dialog_with_unmarked_text(self):
        for relative in sorted(CANVAS_MARKED):
            with self.subTest(file=relative):
                self.assertEqual(unmarked_dialog_texts((SRC / relative).read_text(encoding="utf-8")), [])

    def test_the_dialog_check_finds_unmarked_text(self):
        source = '''
def run(viewer, error):
    QMessageBox.critical(None, "Load Error", f"Could not load: {error}")
    path, _ = QFileDialog.getOpenFileName(None, "Select File", "", "Text Files (*.txt)")
    QMessageBox.critical(None, translate("Viewer", "Load Error"), Message("Could not load: {error}", error=error).display())
    path, _ = QFileDialog.getOpenFileName(None, translate("Viewer", "Select File"), "",
                                          ";;".join([translate("Viewer", "Text Files (*.txt)")]))
    path, _ = QFileDialog.getOpenFileName(None, translate("Viewer", "Select File"), "",
                                          ";;".join([translate("Viewer", "Text Files (*.txt)"), "All Files (*)"]))
'''
        self.assertEqual([line for line, _ in unmarked_dialog_texts(source)], [3, 4, 8])

    def test_the_check_finds_each_kind_of_unmarked_text(self):
        source = '''
HELP = "Usage: zoom <width>"

def run(viewer, args, error):
    Command_Engine.show_status(viewer, "Running.")
    Command_Engine.print_help(viewer, f"Zoomed to {args[0]}.")
    Command_Engine.print_help(viewer, HELP)
    message = "Saved " + args[0]
    Command_Engine.show_status(viewer, message)
    Command_Engine.show_status(viewer, str(error))
    Command_Engine.print_help(viewer, "; ".join(args))
    viewer.zoom_text.text = f"View Width: {args[0]}"
    Command_Engine.report_selection_error(viewer, args[0], error, "Hide")
    Command_Engine.print_help(viewer, msg="Usage: hide <expression>")
    Command_Engine.show_status(viewer, Message("Running {name}.", name=args[0]))
    Command_Engine.show_status(viewer, translate("Viewer", "View Width: {width}").format(width=1))
    Command_Engine.report_selection_error(viewer, args[0], error, Message("Hide"))
    Command_Engine.show_status(viewer, "")
    Command_Engine.show_status(viewer, args[0])
    _report_error(viewer, "Error: bad width.")
    _report_twice(viewer, "Error: again.")
    _report_error(viewer, Message("Error: bad width."))

def _report_error(viewer, msg):
    Command_Engine.command_failed(viewer, msg)
    Command_Engine.print_help(viewer, msg)

def _report_twice(viewer, text):
    _report_error(viewer, text)
'''
        self.assertEqual([line for line, _ in unmarked_canvas_texts(source)], [5, 6, 7, 9, 10, 11, 12, 13, 14, 20, 21])


class ViewerCanvasTextTests(WindowTestCase):
    """The Viewer's canvas text under the pseudo-language: the HUD, and the
    messages put together from several parts, which the code alone can't show.

    Every letter of the pseudo-language takes an accent, so once a test takes
    out the values it filled in, a plain letter left came from no catalog.
    Brackets can't tell here, since some texts hold their own, as
    "[Cluster {cluster}] {node}" does.
    """

    def setUp(self):
        pseudo_language(self, self.app)

    def assert_translated(self, text, *values):
        self.assertTrue(text)
        left = str(text)
        for value in values:
            left = left.replace(str(value), "")
        self.assertEqual(re.findall("[A-Za-z]", left), [], text)

    def test_the_hud(self):
        import collections
        from types import SimpleNamespace

        import numpy

        import EMAPSSN_Viewer

        class Visual(SimpleNamespace):
            """Stands in for a VisPy visual and keeps what it was given."""

        viewer = EMAPSSN_Viewer.MainViewer.__new__(EMAPSSN_Viewer.MainViewer)
        viewer.hud_layout = collections.defaultdict(float)
        viewer.canvas = SimpleNamespace(size=(1200, 800), scene=None, update=lambda: None)
        viewer.vispy_ui_face = viewer.vispy_monospace_face = "Noto Sans"
        viewer.view = SimpleNamespace(camera=SimpleNamespace(rect=SimpleNamespace(width=1234.5)))
        viewer.visible_mask = numpy.array([True, False, False])
        viewer.update_console_background = lambda: None
        viewer.hud_displays = {}
        with mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Text", Visual), \
                mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Rectangle", Visual):
            viewer.create_hud()
        viewer._update_hud_elements()
        self.assert_translated(viewer.instr_text.text)
        self.assert_translated(viewer.zoom_text.text, "1234.5")
        self.assert_translated(viewer.hidden_text.text, "2")

    def test_the_selected_nodes_cluster_and_groups(self):
        from types import SimpleNamespace

        from EMAPSSN_Viewer import MainViewer

        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = ["101", "102"]
        viewer.cluster_labels = [3, -1]
        viewer.group_labels = [{"7", "8"}, set()]
        viewer.console_text = SimpleNamespace(text="")
        viewer.hud_displays = {}
        viewer.update_nodes = viewer.broadcast_event = lambda *args: None
        for node in (0, 1):
            with contextlib.redirect_stdout(io.StringIO()):
                viewer.apply_left_click_focus(node)
            with self.subTest(node=node):
                self.assert_translated(viewer.console_text.text, "101", "102", "7", "8", "3")

    def test_a_selection_errors_first_line_and_what_failed(self):
        # The console line shows the error's first line; the terminal gets every line, in English.
        from types import SimpleNamespace

        import Command_Engine

        failing = (
            lambda: Command_Engine.resolve_label_target(None, None, "noise"),
            lambda: Command_Engine.resolve_label_target([1, 2], [set(), set()], "123"),
            lambda: Command_Engine.parse_selection_expression('"12'),
            lambda: Command_Engine.parse_selection_expression("12 &"),
            lambda: Command_Engine.parse_metadata_range("-1-0"),
        )
        viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
        for call in failing:
            with self.assertRaises(Command_Engine.SelectionExpressionError) as caught:
                call()
            error = caught.exception
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                Command_Engine.report_selection_error(viewer, "12", error)
            with self.subTest(error=str(error)):
                self.assert_translated(viewer.console_text.text, '"12', "12 &", "123", "-1-0", "12")
                self.assertIn(f"Selection error: {str(error).splitlines()[0]}", terminal.getvalue())
                self.assertNotIn("Šéļéçţîöñ", terminal.getvalue())

    def test_the_help_of_every_marked_command(self):
        # A help text's first line, such as "Usage: zoom <width>", is in a
        # JoinedMessage, whose parts the code alone can't show.
        import importlib

        import EMAPSSN_Config

        folder = tempfile.mkdtemp()  # meta and print make their folders first.
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        commands = sorted(name for name in CANVAS_MARKED if name.startswith("commands/"))
        self.assertTrue(commands)
        for relative in commands:
            module = importlib.import_module(relative[:-3].replace("/", "."))
            viewer = mock.MagicMock()
            viewer.console_text.text = ""
            with self.subTest(command=relative), contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(module, "register", create=True), \
                    mock.patch.object(EMAPSSN_Config, "METADATA_DIR", folder, create=True), \
                    mock.patch.object(EMAPSSN_Config, "resolve_directory_path", lambda value, *rest: folder):
                module.run(viewer, ["help"])
                shown = viewer.console_text.text
                self.assertTrue(shown)
                self.assertEqual(outside_the_catalog(shown), "", shown)

    def test_command_messages_put_together_from_parts(self):
        # An error raised with a message, a message a helper returns, a first
        # line in a JoinedMessage: what the console line shows, the code alone can't.
        from types import SimpleNamespace

        from commands import esmfold, offset, reference, reset

        cases = (
            (reset, []),
            (reset, ["123"]),
            (offset, ["1", "2"]),
            (esmfold, ["123"]),
            (reference, []),
        )
        for module, args in cases:
            viewer = SimpleNamespace(
                console_text=SimpleNamespace(text=""), alignment=None, active_reference=None, alignment_offset=0,
            )
            with self.subTest(command=module.__name__, args=args), contextlib.redirect_stdout(io.StringIO()):
                module.run(viewer, args)
                shown = viewer.console_text.text
                self.assertTrue(shown)
                self.assert_translated(shown, "123")

    def command_viewer(self, **attributes):
        """Two nodes, with headers "101" and "102", and what the selection commands use."""
        from types import SimpleNamespace

        import numpy

        viewer = SimpleNamespace(
            full_headers=["101", "102"], n_nodes=2, alignment=None, metadata=None,
            cluster_labels=None, group_labels=None, selected_indices=[],
            visible_mask=numpy.array([True, True]), current_colors=numpy.ones((2, 4)),
            current_sizes=numpy.ones(2), console_text=SimpleNamespace(text=""),
            _save_state=lambda: None, update_nodes=lambda: None, promote_nodes=lambda mask: None,
            update_selection_visual=lambda: None,
        )
        vars(viewer).update(attributes)
        return viewer

    def test_command_reports_put_together_from_counted_parts(self):
        # "Applied: 1 node (red)" joins counted parts, a group report may hold
        # a skipped warning, and select counts each of its two numbers.
        from commands import color, group, select

        cases = (
            (color, ['"101"', "red"], {}, ("red",)),
            (group, ['"101"', "7"], {}, ("7",)),
            (group, ['"101"', "7", '"102"', "noise"], {}, ("7",)),
            (group, ['"101"', "noise"], {}, ("noise",)),
            (group, ["remove", "8"], {"group_labels": [{"7"}, set()]}, ("8",)),
            (group, ["remove", "7"], {"group_labels": [{"7"}, set()]}, ()),
            (select, ["invert"], {"selected_indices": [0]}, ()),
            (select, ['"101"'], {"selected_indices": [1]}, ()),
            (select, ["add", '"101"'], {"selected_indices": [1]}, ()),
            (select, ["keep", '"101"'], {"selected_indices": [0, 1]}, ()),
        )
        for module, args, attributes, values in cases:
            viewer = self.command_viewer(**attributes)
            with self.subTest(command=module.__name__, args=args), contextlib.redirect_stdout(io.StringIO()):
                module.run(viewer, args)
                self.assert_translated(viewer.console_text.text, *values)

    def test_command_errors_that_hold_another_message(self):
        # An error filled into an error, such as the output-name check's, and a
        # first line in a JoinedMessage.
        import EMAPSSN_Config
        from commands import color, export, logo, meta, query, select
        from utilities.Localization import Message

        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        viewer = self.command_viewer()
        with contextlib.redirect_stdout(io.StringIO()):
            export._refused_output_name(viewer, "../1", Message("group label '{group}'", group="1"))
        self.assert_translated(viewer.console_text.text)
        cases = (
            (color, [], {}, ()),
            (logo, [], {}, ()),
            (query, [], {}, ()),
            (select, [], {}, ()),
            (select, ["9"], {}, ("9",)),
            (select, ["save", "../1.txt"], {"selected_indices": [0]}, ()),
            (select, ["save", "1.fasta"], {"selected_indices": [0]}, ()),
            (meta, ["download", "../1.csv"], {}, ()),
            (meta, ["delete", "9"], {"metadata": {"1": {}}}, ("9",)),
            (meta, ["show"], {}, ("meta show <property_name>", "meta show clear/off")),
        )
        for module, args, attributes, values in cases:
            viewer = self.command_viewer(**attributes)
            with self.subTest(command=module.__name__, args=args), contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(EMAPSSN_Config, "METADATA_DIR", folder, create=True), \
                    mock.patch.object(EMAPSSN_Config, "HEADER_LIST_DIR", folder, create=True):
                module.run(viewer, args)
                self.assert_translated(viewer.console_text.text, *values)
        self.assertEqual(os.listdir(folder), [])

    def test_export_refusing_a_name_that_is_a_path(self):
        # What export refused, a group label or the clustering parameters, is a message of its own.
        from types import SimpleNamespace

        import numpy

        import EMAPSSN_Config
        from commands import export

        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        network = SimpleNamespace(model_name="1", network_type="blast")
        with mock.patch.object(EMAPSSN_Config, "NODE_FASTA_FILE", os.path.join(folder, "1.fasta"), create=True), \
                mock.patch.object(EMAPSSN_Config, "INPUT_HDF5", "1.h5", create=True), \
                mock.patch.object(EMAPSSN_Config, "TOP_EDGE_PERCENT", None, create=True), \
                mock.patch.object(EMAPSSN_Config, "SIMILARITY_THRESHOLD", 0.5, create=True), \
                mock.patch.object(export, "SEQUENCE_EXPORT_DIRECTORY", folder), \
                mock.patch.object(export.cache_manifest, "validate_network_schema", return_value=network), \
                mock.patch.object(export, "open_in_file_manager"):
            for args, attributes in (
                (["groups"], {"group_labels": [{"../1"}, set()]}),
                (["clusters"], {"last_cluster_params": ("../1", 2)}),
            ):
                viewer = self.command_viewer(**{
                    "cluster_labels": numpy.array([0, 0]), "group_labels": [set(), set()],
                    "_selected_fasta_records": [("101", "ACD"), ("102", "EFG")], **attributes,
                })
                with self.subTest(args=args), contextlib.redirect_stdout(io.StringIO()):
                    export.run(viewer, args)
                    self.assertIn("../1", viewer.console_text.text)
                    self.assert_translated(viewer.console_text.text, "../1")
        self.assertEqual(os.listdir(folder), [])

    def test_the_metadata_display_on_the_hud(self):
        # A node without a value shows "N/A" on a HUD visual, not on the console line.
        from types import SimpleNamespace

        import EMAPSSN_Config
        import EMAPSSN_Viewer
        from commands import meta

        class Visual(SimpleNamespace):
            """Stands in for a VisPy visual and keeps what it was given."""

        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        viewer = self.command_viewer(
            metadata={"9": {"type": "number", "values": [None, 5.0]}}, hud_displays={}, selected_node_idx=0,
            canvas=SimpleNamespace(size=(1200, 800), scene=None), vispy_ui_face="Noto Sans",
            _hud_font_size_points=lambda: 10, _status_hud_position=lambda index, size: (0, 0),
        )
        with mock.patch.object(EMAPSSN_Config, "METADATA_DIR", folder, create=True), \
                mock.patch.object(EMAPSSN_Viewer.scene.visuals, "Text", Visual), \
                contextlib.redirect_stdout(io.StringIO()):
            meta.run(viewer, ["show", "9"])
            display = viewer.hud_displays["meta_display"]
            self.assert_translated(display.text_visual.text, "9")
            self.assert_translated(viewer.console_text.text, "9")
            display.on_node_clicked(0)
            self.assert_translated(display.text_visual.text, "9")
            viewer.meta_display_prop = "8"  # Not in the metadata.
            display.on_node_clicked(0)
            self.assert_translated(display.text_visual.text, "8")

    def test_the_spectrum_report(self):
        # The report joins counted sentences around the range and scheme it fills in.
        import EMAPSSN_Config
        from commands import spectrum
        from utilities.Localization import Message

        report = spectrum._applied_message(1, "9", Message("(min: {min}, max: {max})", min=1.5, max=2.5), "8", 2, False)
        self.assert_translated(display_text(report), "9", "8", "1.5", "2.5")
        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        viewer = self.command_viewer(metadata={"9": {"type": "number", "values": [1.5, float("nan")]}})
        with mock.patch.object(EMAPSSN_Config, "METADATA_DIR", folder, create=True), \
                contextlib.redirect_stdout(io.StringIO()):
            spectrum.run(viewer, ["{9}"])
        self.assert_translated(viewer.console_text.text, "9", "1.5", "coolwarm")

    def test_errors_of_position_and_frequency_arguments(self):
        # query and logo fill these errors into their own messages, as cluster
        # and subcluster do the MCL inflation error.
        from commands import cluster, query
        from utilities import Sequence_Utils

        failing = (
            lambda: Sequence_Utils.reject_bare_negative_positions("-1"),
            lambda: Sequence_Utils.normalize_displayed_position_atom("1-"),
            lambda: Sequence_Utils.normalize_displayed_position_atom("1-", allow_end=True),
            lambda: query.parse_query_positions("[-1]", []),
            lambda: query._normalize_frequency_target("(K)"),
            lambda: query._FrequencyLogicParser("M_0 &", {"M_0": True}).parse(),
        )
        for call in failing:
            with self.assertRaises(ValueError) as caught:
                call()
            with self.subTest(error=str(caught.exception)):
                self.assert_translated(display_text(caught.exception), "(K)")
        self.assert_translated(display_text(cluster.mcl_inflation_error(99)))

    def test_print_failures(self):
        # A capture the window spoiled says why, inside "Failed to save {format}: {error}".
        import importlib
        from types import SimpleNamespace

        import EMAPSSN_Config

        print_command = importlib.import_module("commands.print")
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(
                size=(10, 10), physical_size=(10, 10), scene=None, bgcolor="white", update=lambda: None,
            ),
            view=SimpleNamespace(camera=SimpleNamespace(rect=(0, 0, 1, 1), aspect=None)),
            console_text=SimpleNamespace(text=""), hud_displays={},
        )
        events = print_command._CaptureEvents(viewer, [])
        failures = []
        events._next_pause = 0
        viewer.canvas.size = (20, 20)  # The window was resized.
        with self.assertRaises(print_command._CaptureInterrupted) as caught:
            events.pause_if_due()
        failures.append(caught.exception)
        viewer.canvas.size = (10, 10)
        events._next_pause = 0
        with mock.patch.object(print_command, "_process_events_without_input", side_effect=events._mark_changed), \
                self.assertRaises(print_command._CaptureInterrupted) as caught:
            events.pause_if_due()  # Something redrew the network.
        failures.append(caught.exception)
        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        for error in failures + [RuntimeError("9")]:
            with self.subTest(error=str(error)), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(EMAPSSN_Config, "resolve_directory_path", lambda value, *rest: folder), \
                    mock.patch.object(print_command, "_render_capture", side_effect=error):
                print_command.run(viewer, ["1"])
                self.assert_translated(viewer.console_text.text, "PNG", "9")
        self.assertEqual(os.listdir(folder), [])

    def test_label_errors(self):
        # label checks its inputs before its background job and reports them
        # through one helper, whose callers the code check can't follow; a
        # failed job raises the Message its worker showed.
        from types import SimpleNamespace

        import numpy

        import EMAPSSN_Config
        from commands import label
        from utilities.Localization import Message

        class Scheduler:
            def __init__(self, reserved=False, error=None):
                self.reserved, self.error = reserved, error

            def is_output_path_reserved(self, path):
                return self.reserved

            def enqueue(self, **job):
                raise self.error

        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        reserved_path = os.path.abspath(os.path.join(folder, "1.xlsx"))
        clustered = {"cluster_labels": [1, 1]}
        shutting_down = RuntimeError(Message("The background job scheduler is shutting down."))
        cases = (
            ([], {"alignment": None}),
            ([], {"alignment": SimpleNamespace(aln=[], has_reference=True)}),
            ([], {"alignment": SimpleNamespace(aln=[1], has_reference=False)}),
            (["gmin"], {}),
            (["clusters"], {"group_labels": [set(), set()]}),
            (["groups"], clustered),
            ([], {}),
            ([], clustered),
            (["1.xlsx"], {**clustered, "background_job_scheduler": Scheduler(reserved=True)}),
            ([], {**clustered, "background_job_scheduler": Scheduler(error=shutting_down)}),
        )
        for args, attributes in cases:
            viewer = SimpleNamespace(**{
                "alignment": SimpleNamespace(aln=[1], has_reference=True), "cluster_labels": None,
                "group_labels": None, "full_headers": ["101", "102"], "n_nodes": 2,
                "console_text": SimpleNamespace(text=""), **attributes,
            })
            with self.subTest(args=args, attributes=sorted(attributes)), contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(EMAPSSN_Config, "resolve_directory_path", lambda value, *rest: folder), \
                    mock.patch.object(label.Command_Engine, "get_alignment_mapping",
                                      return_value=(numpy.array([0, 1]), numpy.array([0, 1]))), \
                    mock.patch.object(label, "_FrozenAlignmentManager", lambda alignment, viewer_to_aln: None):
                label.run(viewer, args)
                # The pseudo-language leaves markup such as the syntax <ID> as it is.
                self.assert_translated(viewer.console_text.text, reserved_path, "<ID>")
        viewer = SimpleNamespace(**{**clustered, "alignment": SimpleNamespace(aln=[1], has_reference=True),
                                    "group_labels": None, "full_headers": [], "n_nodes": 0,
                                    "console_text": SimpleNamespace(text=""),
                                    "background_job_scheduler": Scheduler()})
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(EMAPSSN_Config, "resolve_directory_path", lambda value, *rest: folder), \
                mock.patch.object(label.Command_Engine, "get_alignment_mapping", side_effect=RuntimeError("9")):
            label.run(viewer, [])
        self.assert_translated(viewer.console_text.text, "9")
        snapshot = SimpleNamespace(alignment=None, console_text=SimpleNamespace(text=""))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError) as caught:
            label._execute_label_envelope(label._LabelJobEnvelope(snapshot, ()))
        self.assertEqual(str(caught.exception), "Error: Global Alignment not loaded.")
        self.assert_translated(display_text(caught.exception))
        self.assertEqual(os.listdir(folder), [])

    def test_logo_errors_and_report(self):
        # The errors logo fills into "Error: {error}", and the report of its background job.
        from commands import logo

        failing = (
            lambda: logo.parse_identity_threshold(""),
            lambda: logo.parse_identity_threshold("x"),
            lambda: logo.parse_identity_threshold("200%"),
            lambda: logo.extract_identity_threshold(["90%", "80%"]),
            lambda: logo.parse_logo_positions("[1.0]"),
            lambda: logo.parse_logo_positions("[1.1-2]"),
            lambda: logo.parse_logo_positions("[5-2]"),
        )
        for call in failing:
            with self.assertRaises(ValueError) as caught:
                call()
            with self.subTest(error=str(caught.exception)):
                self.assert_translated(display_text(caught.exception), "x")
        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        for threshold, filename in ((None, "1.svg"), (0.9, "2.svg")):
            payload = {
                "selected_seqs": ("AAAA", "AAAA", "GGGG"), "valid_cols": (0,), "plot_positions": (1,),
                "mode": "pcts", "gap_mode": "no_gap", "identity_threshold": threshold, "filename": filename,
                "color_scheme": "chemistry", "logo_dir": folder, "allow_overwrite": False, "ref_id": "1",
            }
            with self.subTest(threshold=threshold), contextlib.redirect_stdout(io.StringIO()):
                result = logo._generate_logo_artifact(payload)
                self.assert_translated(display_text(result["message"]), "no_gap", "pcts", filename)

    def test_background_job_errors(self):
        # The scheduler's refusals, and a failed job's error, show translated.
        import threading
        from types import SimpleNamespace

        from Background_Job_Scheduler import BackgroundJob, BackgroundJobScheduler
        from utilities.Localization import Message

        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        existing = os.path.abspath(os.path.join(folder, "1.svg"))
        Path(existing).write_text("", encoding="utf-8")
        reserved = os.path.abspath(os.path.join(folder, "2.svg"))

        def scheduler(accepting=True):
            return SimpleNamespace(
                _lock=threading.Lock(), _accepting=accepting, _output_key=BackgroundJobScheduler._output_key,
                _reserved_output_paths={BackgroundJobScheduler._output_key(reserved)},
            )

        for state, path in ((scheduler(accepting=False), reserved), (scheduler(), existing), (scheduler(), reserved)):
            with self.subTest(path=path, accepting=state._accepting), \
                    self.assertRaises((RuntimeError, FileExistsError)) as caught:
                BackgroundJobScheduler.enqueue(state, "1", "1", None, None, path)
            self.assert_translated(display_text(caught.exception), path)
        shown = []
        state = SimpleNamespace(_set_viewer_status=lambda message: shown.append(display_text(message)))
        job = BackgroundJob(7, "label", "label -> 1.xlsx", None, None, "1.xlsx")
        with contextlib.redirect_stdout(io.StringIO()):
            BackgroundJobScheduler._report_failed(
                state, job, RuntimeError(Message("Error: No groups defined.")), "", 2.5
            )
        self.assert_translated(shown[0], "label", "2.5", "7")

        # From the worker thread to the console line, the failed job's error stays a Message.
        import time

        class Viewer:
            """The scheduler keeps a weak reference to its viewer."""

            def __init__(self):
                self.statuses = []

            def set_background_job_status(self, message):
                self.statuses.append(display_text(message))

        viewer = Viewer()
        statuses = viewer.statuses
        jobs = BackgroundJobScheduler(viewer)
        self.addCleanup(jobs.shutdown)

        def worker(payload):
            raise RuntimeError(Message("Error: No groups defined."))

        with contextlib.redirect_stdout(io.StringIO()):
            jobs.enqueue("label", "1", None, worker, os.path.join(folder, "3.svg"))
            deadline = time.monotonic() + 10
            while len(statuses) < 3 and time.monotonic() < deadline:  # Queued, running, failed.
                self.app.processEvents()
                time.sleep(0.005)
        self.assertEqual(len(statuses), 3, statuses)
        self.assert_translated(statuses[-1], "label")

    def test_the_metadata_upload_summary_and_its_errors(self):
        from types import SimpleNamespace

        import Metadata_Core

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        (folder / "1.csv").write_text(",1\n,number\n101,5\n102,6\n", encoding="utf-8")
        (folder / "2.csv").write_text(",1\n", encoding="utf-8")
        viewer = SimpleNamespace(
            full_headers=["101", "102"], n_nodes=2, metadata={},
            console_text=SimpleNamespace(text=""),
            _save_state=lambda: None, broadcast_metadata_state=lambda: None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            Metadata_Core.upload_metadata(viewer, [str(folder / name) for name in ("1.csv", "2.csv", "3.csv")])
        self.assertIn("1.csv", viewer.console_text.text)
        self.assert_translated(viewer.console_text.text, "1.csv", "2.csv", "3.csv")
        for requested, metadata in ((["9"], {"1": {}}), (["9"], {}), (["all"], {}), (["Node ID"], {})):
            viewer.metadata = metadata
            with self.subTest(requested=requested, metadata=metadata), \
                    self.assertRaises(Metadata_Core.MetadataColumnDeleteError) as caught:
                Metadata_Core._resolve_metadata_column_names(viewer, requested)
            self.assert_translated(display_text(caught.exception), "9", "1", "Node ID")

    def test_the_model_card_errors(self):
        from web_ui import agent_backend

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        for content in ("{", '{"cards": 1}'):
            cards = folder / "cards.json"
            cards.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(agent_backend.ModelCardsError) as caught:
                agent_backend._read_model_card_document(str(cards))
            cause = caught.exception.__cause__
            self.assert_translated(display_text(caught.exception), str(cards), *([str(cause)] if cause else []))

    def test_the_background_job_status(self):
        from types import SimpleNamespace

        from Background_Job_Scheduler import BackgroundJob, BackgroundJobScheduler

        shown = []
        scheduler = SimpleNamespace(_set_viewer_status=lambda message: shown.append(display_text(message)))
        job = BackgroundJob(7, "label", "label -> 1.svg", None, None, "1.svg")
        with contextlib.redirect_stdout(io.StringIO()):
            BackgroundJobScheduler._report_started(scheduler, job)
            BackgroundJobScheduler._report_succeeded(scheduler, job, {}, 2.5)
            BackgroundJobScheduler._report_failed(scheduler, job, "98", "", 2.5)
        self.assertEqual(len(shown), 3)
        for text in shown:
            with self.subTest(text=text):
                self.assert_translated(text, "label -> 1.svg", "1.svg", "label", "2.5", "98", "7")


if __name__ == "__main__":
    unittest.main()
