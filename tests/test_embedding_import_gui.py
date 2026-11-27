"""GUI tests for the Embedding Import Tools card in EMAPSSN_Tools.

Backend, matching, array validation, source reader, output/resume, CLI, and MCP
validation tests live in test_embedding_import.py (Qt-free).
"""

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, SRC / "tools"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def _generate_embeddings_default():
    """The SAVING_MODE literal assigned in Generate_Embeddings.py."""
    tree = ast.parse((SRC / "tools" / "Generate_Embeddings.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "SAVING_MODE" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Generate_Embeddings.py assigns no SAVING_MODE")


class ToolsCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def window(self, sections=None):
        from tests.tools_gui_fixtures import isolated_tools_project
        root = isolated_tools_project(self, sections=sections)
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        path = next(p for p in window.script_data if Path(p).name == "Embedding_Import.py")
        return window, path, root

    def test_the_card_is_last_on_the_first_tab_and_offers_the_contracts_fields(self):
        import EMAPSSN_Tools
        from mcp_server.pipeline.Pipeline_Settings import _CONTRACTS
        window, path, _ = self.window()
        first_tab = list(window.MANUAL_SETTINGS["Sequence_and_Embedding_Preparation"]["scripts"])
        self.assertEqual(first_tab[-1], "Embedding_Import.py")
        self.assertEqual(EMAPSSN_Tools.TOOL_TITLES["Embedding_Import.py"], "Embedding Import")
        self.assertEqual(EMAPSSN_Tools.TOOL_ICONS["Embedding_Import.py"], "file-input")
        inputs = window.script_data[path]["inputs"]
        self.assertLessEqual(set(_CONTRACTS["Embedding_Import.py"]["properties"]), set(inputs))
        self.assertEqual(window._collect_tool_settings(path)["SAVING_MODE"], _generate_embeddings_default())

    def test_the_source_row_picks_a_file_or_a_folder_and_the_mapping_row_a_file(self):
        window, path, _ = self.window()
        inputs = window.script_data[path]["inputs"]
        source, id_map = inputs["SOURCE_PATH"]["widget"], inputs["ID_MAP_FILE"]["widget"]
        self.assertEqual((source.browse_button.text(), source.folder_button.text()), ("File...", "Folder..."))
        self.assertIsNone(id_map.folder_button)
        folder = tempfile.gettempdir()
        with mock.patch("EMAPSSN_Tools.QFileDialog.getExistingDirectory", return_value=folder) as dialog:
            source.folder_button.click()
        self.assertEqual(dialog.call_args.args[1], "Select Embedding Source")
        self.assertEqual(source.line_edit.text(), os.path.normpath(folder))
        chosen = os.path.join(folder, "per-residue.h5")
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=(chosen, "")) as dialog:
            source.browse_button.click()
        self.assertIn("*.npz", dialog.call_args.args[3])
        self.assertEqual(source.line_edit.text(), os.path.normpath(chosen))
        with mock.patch("EMAPSSN_Tools.QFileDialog.getOpenFileName", return_value=("", "")) as dialog:
            id_map.browse_button.click()
        self.assertEqual(dialog.call_args.args[1], "Select ID Mapping File")
        self.assertIn("*.tsv *.csv *.txt", dialog.call_args.args[3])

    def test_save_and_run_hands_both_paths_to_the_run_but_never_saves_them(self):
        window, path, root = self.window()
        inputs = window.script_data[path]["inputs"]
        source = str(Path(tempfile.gettempdir()) / "per-residue.h5")
        id_map = str(Path(tempfile.gettempdir()) / "map.tsv")
        inputs["SOURCE_PATH"]["widget"].line_edit.setText(source)
        inputs["ID_MAP_FILE"]["widget"].line_edit.setText(id_map)
        inputs["MODEL_LABEL"]["widget"].setText("uniprot_prot_t5")
        launched = []
        with mock.patch("EMAPSSN_Tools.launch_in_terminal", side_effect=lambda argv, **kw: launched.append(argv)):
            window.save_and_run(path)
        saved_text = (root / "tools_settings.json").read_text(encoding="utf-8")
        section = json.loads(saved_text)["Embedding_Import.py"]
        self.assertNotIn("SOURCE_PATH", section)
        self.assertNotIn("ID_MAP_FILE", section)
        self.assertEqual(section["MODEL_LABEL"], "uniprot_prot_t5")
        self.assertNotIn("per-residue.h5", saved_text)
        self.assertEqual(len(launched), 1)
        from tools.tool_helpers.Tool_Pipeline import forget_one_run_snapshot
        snapshot = Path(launched[0][-1])
        self.addCleanup(lambda: forget_one_run_snapshot(str(snapshot)))
        document = json.loads(snapshot.read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertEqual((document["SOURCE_PATH"], document["ID_MAP_FILE"]),
                         (os.path.normpath(source), os.path.normpath(id_map)))

    def test_an_untouched_card_shows_an_empty_label_and_saves_no_text_none(self):
        window, path, root = self.window()
        label = window.script_data[path]["inputs"]["MODEL_LABEL"]["widget"]
        self.assertEqual(label.text(), "")
        self.assertEqual(window._collect_tool_settings(path)["MODEL_LABEL"], "")
        launched = []
        with mock.patch("EMAPSSN_Tools.launch_in_terminal", side_effect=lambda argv, **kw: launched.append(argv)):
            window.save_and_run(path)
        saved = json.loads((root / "tools_settings.json").read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertEqual(saved.get("MODEL_LABEL", ""), "")
        self.assertNotIn("None", saved.values())
        self.assertEqual(len(launched), 1)
        from tools.tool_helpers.Tool_Pipeline import forget_one_run_snapshot
        snapshot = Path(launched[0][-1])
        self.addCleanup(lambda: forget_one_run_snapshot(str(snapshot)))
        document = json.loads(snapshot.read_text(encoding="utf-8"))["Embedding_Import.py"]
        self.assertNotIn("None", document.values())

    def test_every_text_field_whose_default_is_none_shows_empty_even_after_an_older_save(self):
        from PySide6.QtWidgets import QLineEdit
        saved = {"Embedding_Import.py": {"MODEL_LABEL": "None"},
                 "Embedding_PWA.py": {"REF_HEADER": "None", "TAR_HEADER": "None"},
                 "Embedding_SSEARCH.py": {"NORM_THRESHOLD": "None"}}
        for sections in ({}, saved):
            window, _, _ = self.window(sections=sections)
            for script, names in (("Embedding_Import.py", ("MODEL_LABEL",)),
                                  ("Embedding_PWA.py", ("REF_HEADER", "TAR_HEADER")),
                                  ("Embedding_SSEARCH.py", ("NORM_THRESHOLD",))):
                path = next(p for p in window.script_data if Path(p).name == script)
                for name in names:
                    with self.subTest(script=script, name=name, saved=bool(sections)):
                        field = window.script_data[path]["inputs"][name]["widget"]
                        self.assertIsInstance(field, QLineEdit)
                        self.assertEqual(field.text(), "")
                        self.assertEqual(window._collect_tool_settings(path)[name], "")

    # --- the card's rows ---

    def shown_card(self):
        from PySide6.QtWidgets import QLabel, QWidget
        from EMAPSSN_Tools import configure_qt_application_fonts
        configure_qt_application_fonts(self.app)
        window, path, _ = self.window()
        window.show()
        self.flush()
        inputs = window.script_data[path]["inputs"]
        field = inputs["MODEL_LABEL"]["widget"]
        tab = next(index for index in range(window.tabs.count()) if window.tabs.widget(index).isAncestorOf(field))
        window.tabs.setCurrentIndex(tab)
        self.flush()
        scroll = window.tabs.widget(tab)
        self.assertTrue(scroll.isAncestorOf(inputs["SOURCE_PATH"]["widget"]))
        # The source row is on its own, so its parent is the card's form.
        card = inputs["SOURCE_PATH"]["widget"].parentWidget()

        def label_x(text):
            labels = [label for label in card.findChildren(QLabel) if label.text() == text]
            self.assertEqual(len(labels), 1, text)
            return labels[0].mapTo(scroll.widget(), labels[0].rect().topLeft()).x()

        def box(widget):
            return widget.rect().translated(widget.mapTo(scroll.widget(), widget.rect().topLeft()))

        return window, path, scroll, card, label_x, box, QWidget

    def flush(self):
        for _ in range(4):
            self.app.processEvents()

    def resize_panel(self, window, width):
        right = window.splitter.widget(1).minimumSizeHint().width() + 4
        window.resize(width + right + 34, 850)
        window.splitter.setSizes([width + 4, right])
        self.flush()
        delta = width - window.tabs.currentWidget().width()
        left, right = window.splitter.sizes()
        window.splitter.setSizes([left + delta, right - delta])
        self.flush()

    def test_the_rows_hold_the_set_and_the_mapping_then_the_source_then_the_label_and_the_mode(self):
        window, path, scroll, card, label_x, box, _ = self.shown_card()
        self.resize_panel(window, 1400)
        inputs = window.script_data[path]["inputs"]
        fasta, id_map, source, label, mode = (box(inputs[name]["widget"]) for name in (
            "INPUT_FASTA", "ID_MAP_FILE", "SOURCE_PATH", "MODEL_LABEL", "SAVING_MODE"))
        self.assertLessEqual(abs(fasta.center().y() - id_map.center().y()), 3)
        self.assertLess(fasta.right(), id_map.left())
        self.assertLess(fasta.bottom(), source.top())
        self.assertLess(source.bottom(), label.top())
        self.assertLessEqual(abs(label.center().y() - mode.center().y()), 3)
        self.assertLess(label.right(), mode.left())
        self.assertEqual({fasta.left(), source.left(), label.left()}, {fasta.left()})
        self.assertGreater(source.width(), fasta.width())
        self.assertGreater(label_x("ID Mapping (.tsv/.csv):"), label_x("Sequence Set (.fasta):"))

    def test_the_saving_mode_label_starts_where_the_id_mapping_label_does_at_every_width(self):
        # Measured (offscreen, the window's fonts): the two rows switch together, stacked at the default
        # window (a 600 px panel) up to 1060 px, and side by side from 1080 px.
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        # The equal columns line the labels up by themselves while both rows are side by side; the
        # declaration keeps them lined up if a row's columns ever differ, so it is tested as declared.
        first = card.findChild(QWidget, "compactRow_INPUT_FASTA_ID_MAP_FILE").layout()
        self.assertIs(card.findChild(QWidget, "compactRow_MODEL_LABEL_SAVING_MODE").layout().align_to, first)
        found = lambda: self.label_positions(card, scroll)
        for width in (None, 1000, 1400, 1900, 2000):
            if width:
                self.resize_panel(window, width)
            with self.subTest(width=width):
                positions = found()
                (id_x, id_left, id_stacked), (mode_x, mode_left, mode_stacked) = positions["ID Mapping (.tsv/.csv):"], positions["Saving Mode:"]
                if width in (None, 1000):
                    # Stacked together: each label is at its row's left edge.
                    self.assertTrue(id_stacked and mode_stacked, positions)
                    self.assertEqual((id_x, mode_x), (id_left, mode_left), positions)
                else:
                    self.assertFalse(id_stacked or mode_stacked, positions)
                    self.assertGreater(id_x, id_left + 100, positions)
                    self.assertEqual(mode_x, id_x, positions)

    def test_the_two_rows_are_never_in_different_states_at_any_width(self):
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        seen = set()
        for width in range(600, 1301, 50):
            self.resize_panel(window, width)
            positions = self.label_positions(card, scroll)
            (id_x, _, id_stacked), (mode_x, _, mode_stacked) = (positions["ID Mapping (.tsv/.csv):"],
                                                                 positions["Saving Mode:"])
            with self.subTest(width=width):
                self.assertEqual(id_stacked, mode_stacked, positions)
                if not id_stacked:
                    self.assertEqual(id_x, mode_x, positions)
            seen.add(id_stacked)
        self.assertEqual(seen, {True, False})  # the scan covers both states

    def label_positions(self, card, scroll):
        """For each second label: (its x, the x of its row's left edge, whether its row is stacked)."""
        from PySide6.QtWidgets import QLabel, QWidget
        found = {}
        for group_name, text in (("compactRow_INPUT_FASTA_ID_MAP_FILE", "ID Mapping (.tsv/.csv):"),
                                 ("compactRow_MODEL_LABEL_SAVING_MODE", "Saving Mode:")):
            group = card.findChild(QWidget, group_name)
            self.assertIsNotNone(group, group_name)
            labels = [label for label in group.findChildren(QLabel) if label.text() == text]
            self.assertEqual(len(labels), 1, text)
            x = labels[0].mapTo(scroll.widget(), labels[0].rect().topLeft()).x()
            left = group.mapTo(scroll.widget(), group.rect().topLeft()).x()
            found[text] = (x, left, bool(group.property("stacked")))
        return found

    def test_the_tab_key_follows_the_new_rows(self):
        from PySide6.QtCore import Qt
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        inputs = window.script_data[path]["inputs"]
        order = ("INPUT_FASTA", "ID_MAP_FILE", "SOURCE_PATH", "MODEL_LABEL", "SAVING_MODE")
        owner = {}
        for name in order:
            widget = inputs[name]["widget"]
            for member in [widget] + widget.findChildren(QWidget):
                owner[member] = name
        seen, current = [], inputs[order[0]]["widget"]
        for _ in range(800):
            name = owner.get(current)
            if name and current.focusPolicy() != Qt.FocusPolicy.NoFocus and name not in seen[-1:]:
                seen.append(name)
            current = current.nextInFocusChain()
            if len(seen) >= len(order):
                break
        self.assertEqual(seen, list(order))

    def test_the_rows_stack_without_overlap_below_the_width_they_need(self):
        window, path, scroll, card, label_x, box, QWidget = self.shown_card()
        # Measured: both rows stack, together, at 600 and at 1000 px.
        both = ("compactRow_INPUT_FASTA_ID_MAP_FILE", "compactRow_MODEL_LABEL_SAVING_MODE")
        for width, stacked_rows in ((600, both), (1000, both)):
            self.resize_panel(window, width)
            for name in stacked_rows:
                self.assertTrue(card.findChild(QWidget, name).property("stacked"), (width, name))
            for widget in [card] + card.findChildren(QWidget):
                layout = widget.layout()
                if layout is None:
                    continue
                children = [layout.itemAt(index).widget() for index in range(layout.count())]
                children = [child for child in children if child is not None and child.isVisibleTo(card)]
                for index, child in enumerate(children):
                    self.assertTrue(widget.rect().contains(child.geometry()), (width, child.objectName()))
                    for other in children[index + 1:]:
                        self.assertFalse(child.geometry().intersects(other.geometry()), (width, type(child).__name__))


if __name__ == "__main__":
    unittest.main()
