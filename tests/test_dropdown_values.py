# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Dropdowns keep each option's stored value apart from its displayed label.

Most tests relabel the options first (a pseudo-translation), so code that still
reads or matches the displayed text fails here rather than after a real
translation ships.
"""

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

from tests.config_gui_loader import load_config_namespace, open_config_window
from tests.tools_gui_fixtures import isolated_tools_project


def pseudo_translate(combo):
    """Give every option a label that differs from its stored value."""
    for index in range(combo.count()):
        combo.setItemText(index, f"[{combo.itemText(index)[::-1]}]")


def stored_values(combo):
    return [combo.itemData(index) for index in range(combo.count())]


class ComboHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_values_are_read_and_selected_apart_from_labels(self):
        from PySide6.QtWidgets import QComboBox
        from desktop.Desktop_App import (
            add_combo_options,
            combo_value,
            select_combo_value,
        )

        combo = QComboBox()
        add_combo_options(combo, ["global", "local"], ["Global", "Lokal"])
        self.assertTrue(combo.property("persistItemData"))
        self.assertEqual(combo.itemText(1), "Lokal")
        self.assertTrue(select_combo_value(combo, "local"))
        self.assertEqual(combo_value(combo), "local")
        # A label is not a value, and an unknown value keeps the selection.
        self.assertFalse(select_combo_value(combo, "Lokal"))
        self.assertFalse(select_combo_value(combo, None))
        self.assertEqual(combo_value(combo), "local")
        combo.setCurrentIndex(-1)
        self.assertEqual(combo_value(combo), "")
        with self.assertRaises(ValueError):
            add_combo_options(combo, ["a", "b"], ["only one label"])


class ConfigDropdownValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = load_config_namespace()
        cls.gui_class = cls.namespace["ConfigGUI"]
        cls.app = (
            cls.namespace["QApplication"].instance()
            or cls.namespace["QApplication"]([])
        )
        cls.combo_value = staticmethod(cls.namespace["combo_value"])
        cls.select_combo_value = staticmethod(cls.namespace["select_combo_value"])

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.window = open_config_window(self.gui_class, directory.name)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_relabelled_modes_save_their_values(self):
        window = self.window
        geometry = window.inputs["PACKING_GEOMETRY"]
        for combo in (window.cb_score_mode, geometry):
            pseudo_translate(combo)
        self.select_combo_value(window.cb_score_mode, "local")
        # Choosing "local" rebuilt the normalization options.
        pseudo_translate(window.cb_norm_mode)
        self.assertNotIn("alignment_length", stored_values(window.cb_norm_mode))
        self.select_combo_value(window.cb_norm_mode, "average_sequence")
        self.select_combo_value(geometry, "Circle")

        data = window.collect_data()
        self.assertEqual(data["ALIGNMENT_SCORE"], "local")
        self.assertEqual(data["NORM_MODE"], "average_sequence")
        self.assertEqual(data["PACKING_GEOMETRY"], "Circle")
        profile = window._collect_tab_profile_data("inputs_outputs")
        self.assertEqual(profile["ALIGNMENT_SCORE"], "local")
        self.assertEqual(profile["NORM_MODE"], "average_sequence")
        cache_settings = window._cache_setting_values()
        self.assertEqual(cache_settings["alignment_score"], "local")
        self.assertEqual(cache_settings["normalization"], "average_sequence")

    def test_profile_values_are_restored_by_value(self):
        window = self.window
        geometry = window.inputs["PACKING_GEOMETRY"]
        pseudo_translate(geometry)
        window._set_widget_profile_value("PACKING_GEOMETRY", "Circle")
        self.assertEqual(self.combo_value(geometry), "Circle")
        pseudo_translate(window.cb_score_mode)
        window._set_widget_profile_value("ALIGNMENT_SCORE", "local")
        self.assertEqual(self.combo_value(window.cb_score_mode), "local")

    def test_new_layout_cache_entry_is_found_without_its_label(self):
        window = self.window
        combo = window.cb_cache_file
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("version_00.h5", "folder/version_00.h5")
        combo.addItem("(New Layout Cache)", None)
        combo.setEnabled(True)
        combo.blockSignals(False)
        window._cache_launch_allowed = True
        pseudo_translate(combo)

        combo.setCurrentIndex(1)
        self.assertTrue(window._new_cache_selected())
        self.assertTrue(window.line_new_cache.isEnabled())
        self.assertTrue(window.btn_export_layout.isEnabled())
        window.line_new_cache.setText("named")
        self.assertEqual(window._selected_new_cache_filename(), "named.h5")

        combo.setCurrentIndex(0)
        self.assertFalse(window._new_cache_selected())
        self.assertFalse(window.line_new_cache.isEnabled())
        self.assertFalse(window.btn_export_layout.isEnabled())
        with self.assertRaises(ValueError):
            window._selected_new_cache_filename()

    def test_saved_config_entries_are_found_without_their_labels(self):
        window = self.window
        selector = window.profile_selectors["visual_effects"]
        pseudo_translate(selector)
        with mock.patch.object(self.namespace["QMessageBox"], "critical") as critical:
            self.select_combo_value(selector, "(default)")
        critical.assert_not_called()
        self.assertEqual(
            window._profile_previous_selection["visual_effects"], "(default)"
        )
        _writes, _custom, _created, default_tabs = window._prepare_profile_writes()
        self.assertIn("visual_effects", default_tabs)

    def test_bracketed_profile_names_are_reserved(self):
        validate = self.namespace["_validate_profile_name"]
        discover = self.namespace["_discover_profile_names"]
        for name in ("(custom)", "(Default)", "(new)"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate(name)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory, "visual_effects")
            folder.mkdir()
            for filename in ("(default).json", "theme.json"):
                (folder / filename).touch()
            self.assertEqual(discover(Path(directory), "visual_effects"), ["theme"])


class ToolsDropdownValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        isolated_tools_project(self)
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            self.window = ToolsGUI()
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_relabelled_dropdowns_collect_the_same_settings(self):
        checked = 0
        for script_path, data in self.window.script_data.items():
            dropdowns = [
                entry["widget"]
                for entry in data["inputs"].values()
                if entry["type"] == "dropdown"
            ]
            if not dropdowns:
                continue
            checked += 1
            with self.subTest(script=Path(script_path).name):
                before = self.window._collect_tool_settings(script_path)
                for combo in dropdowns:
                    self.assertTrue(combo.property("persistItemData"))
                    pseudo_translate(combo)
                self.assertEqual(self.window._collect_tool_settings(script_path), before)
        self.assertGreater(checked, 0)

    def script(self, name):
        return next(path for path in self.window.script_data if Path(path).name == name)

    def widget(self, script, variable):
        return self.window.script_data[self.script(script)]["inputs"][variable]["widget"]

    def test_normalization_modes_read_as_words_and_store_their_values(self):
        from desktop.Desktop_App import combo_value, select_combo_value
        from desktop.Tip_Panel import resolve

        values = ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"]
        words = ["Alignment Length", "Shorter Sequence", "Longer Sequence", "Average Sequence"]
        combo = self.widget("Embedding_MSA.py", "NORMALIZATION_MODE")
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], words)
        self.assertEqual(stored_values(combo), values)
        self.assertFalse(any("_" in combo.itemText(i) for i in range(combo.count())))
        # The tip's rows name the options as the dropdown does.
        model = resolve(self.window.tip_db[combo])
        self.assertEqual([label for label, _text in model.options], words)
        self.assertEqual(dict(model.facts)["Default"], "Alignment Length")
        # Each value is selected and collected as itself, whatever its label says.
        path = self.script("Embedding_MSA.py")
        for value, label in zip(values, words):
            with self.subTest(value=value):
                self.assertTrue(select_combo_value(combo, value))
                self.assertEqual(combo.currentText(), label)
                self.assertEqual(combo_value(combo), value)
                self.assertEqual(self.window._collect_tool_settings(path)["NORMALIZATION_MODE"], value)

    def test_a_local_score_leaves_alignment_length_out_of_the_dropdown(self):
        from types import SimpleNamespace

        import EMAPSSN_Tools
        from Cache_Manifest import NetworkCompletenessInfo
        from desktop.Desktop_App import combo_value, select_combo_value

        score = self.widget("Embedding_MSA.py", "ALIGNMENT_SCORE")
        norm = self.widget("Embedding_MSA.py", "NORMALIZATION_MODE")
        network = self.widget("Embedding_MSA.py", "INPUT_NETWORK").combo
        complete = NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
        )
        with mock.patch.object(
            EMAPSSN_Tools, "validate_network_schema", return_value=SimpleNamespace(network_type="embedding")
        ), mock.patch.object(EMAPSSN_Tools, "inspect_network_completeness", return_value=complete):
            network.addItem("embedding.h5")
            network.setCurrentText("embedding.h5")
            select_combo_value(norm, "average_sequence")
            select_combo_value(score, "local")
            self.assertEqual(
                [norm.itemText(i) for i in range(norm.count())],
                ["Shorter Sequence", "Longer Sequence", "Average Sequence"],
            )
            self.assertEqual(
                stored_values(norm), ["shorter_sequence", "longer_sequence", "average_sequence"]
            )
            self.assertEqual(norm.currentText(), "Average Sequence")
            self.assertEqual(combo_value(norm), "average_sequence")
            # Its tip no longer gives the hidden mode's stored value as the default.
            from desktop.Tip_Panel import resolve

            model = resolve(self.window.tip_db[norm])
            self.assertNotIn("Default", dict(model.facts))
            self.assertNotIn("alignment_length", [value for _label, value in model.facts])
            # Back to global, the first mode is offered again and the choice stays.
            select_combo_value(score, "global")
            self.assertEqual(norm.itemText(0), "Alignment Length")
            self.assertEqual(combo_value(norm), "average_sequence")

    def test_a_saved_value_selects_its_words(self):
        from desktop.Desktop_App import combo_value

        # Embedding_SSEARCH has no network to read, so its dropdown shows the saved value at once.
        isolated_tools_project(
            self, sections={"Embedding_SSEARCH.py": {"NORM_MODE": "average_sequence"}}
        )
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        path = next(p for p in window.script_data if Path(p).name == "Embedding_SSEARCH.py")
        combo = window.script_data[path]["inputs"]["NORM_MODE"]["widget"]
        self.assertEqual(combo_value(combo), "average_sequence")
        self.assertEqual(combo.currentText(), "Average Sequence")
        self.assertEqual(window._collect_tool_settings(path)["NORM_MODE"], "average_sequence")

    def test_neighbor_joining_is_recognised_by_value(self):
        from desktop.Desktop_App import select_combo_value

        inputs = next(
            data["inputs"]
            for path, data in self.window.script_data.items()
            if Path(path).name == "Embedding_MSA.py"
        )
        tree_method = inputs["TREE_METHOD"]["widget"]
        bootstrap = inputs["BOOTSTRAP_TREE"]["widget"]
        pseudo_translate(tree_method)
        select_combo_value(tree_method, "UPGMA (Fast)")
        bootstrap.setChecked(True)
        select_combo_value(tree_method, "Neighbor-joining (Slow)")
        self.assertFalse(bootstrap.isEnabled())
        self.assertFalse(bootstrap.isChecked())
        select_combo_value(tree_method, "UPGMA (Fast)")
        self.assertTrue(bootstrap.isEnabled())


class ToolsOptionLabelTests(unittest.TestCase):
    """Tools dropdowns that showed machine names read as words; the stored values stay.

    Execution Mode (Auto, Scalar, Tiled), Precision (FP32 (Full Precision)),
    Saving Mode, BLAST Layout, and the model names. Only the labels changed:
    tools_settings.json, exported settings and the plugins keep the identifiers.
    """

    EXECUTION = {"auto": "Auto", "scalar": "Scalar", "tiled": "Tiled"}
    PRECISION = {
        "automatic_32bit": "Automatic 32-bit",
        "float32": "FP32 (Full Precision)",
        "tf32": "TF32 (Nvidia GPU Only)",
        "bf16": "BF16 (Low Precision)",
    }
    SAVING = {"float32": "Float32 (Full Precision)", "float16": "Float16 (Half Size)"}
    BLAST = {
        "standard_outfmt6": "Standard (outfmt 6)",
        "outfmt7_fields": "Fields Header (outfmt 7)",
        "custom_columns": "Custom Columns (1-based indexing)",
    }
    MODELS = {
        "ankh_base": "Ankh Base",
        "ankh_large": "Ankh Large",
        "esm2_t6_8m": "ESM-2 8M",
        "esm2_t12_35m": "ESM-2 35M",
        "esm2_t30_150m": "ESM-2 150M",
        "esm2_t33_650m": "ESM-2 650M",
        "esmc_300m": "ESM C 300M",
        "esmc_600m": "ESM C 600M",
        "esmc_6b": "ESM C 6B",
        "prost_t5": "ProstT5",
        "prot_bert": "ProtBERT",
    }
    NON_COMMERCIAL = {"ankh_base", "ankh_large"}
    # (script, setting, label by stored value, whether every value is always offered)
    CHANGED = (
        ("Align_Similarity_Matrix.py", "EXECUTION_MODE", EXECUTION, False),
        ("Network_Injection.py", "EXECUTION_MODE", EXECUTION, False),
        ("Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION", PRECISION, False),
        ("Embedding_SSEARCH.py", "ACCELERATOR_PRECISION", PRECISION, False),
        ("Generate_Embeddings.py", "SAVING_MODE", SAVING, True),
        ("Parse_BLAST_Output.py", "BLAST_LAYOUT", BLAST, True),
        ("Generate_Embeddings.py", "MODEL_NAME", MODELS, True),
        ("Embedding_PWA.py", "EMBEDDING_MODEL", MODELS, True),
    )

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def open_tools(self, sections=None):
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        root = isolated_tools_project(self, sections)
        with mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI()
        self.addCleanup(self.app.processEvents)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        return window, root

    @staticmethod
    def script(window, name):
        return next(path for path in window.script_data if Path(path).name == name)

    def combo(self, window, script, setting):
        return window.script_data[self.script(window, script)]["inputs"][setting]["widget"]

    def label_for(self, value, labels):
        """The label a value shows as, with the non-commercial marker the Ankh models carry."""
        if value in self.NON_COMMERCIAL:
            return f"{labels[value]} [non-commercial]"
        return labels[value]

    def test_each_dropdown_offers_its_options_in_words_over_the_stored_values(self):
        from desktop.Desktop_App import NAME_ITEM_ROLE

        window, _ = self.open_tools()
        for script, setting, labels, complete in self.CHANGED:
            with self.subTest(script=script, setting=setting):
                combo = self.combo(window, script, setting)
                values = stored_values(combo)
                # A device decides whether TF32, BF16 and tiled are offered; the rest always is.
                self.assertTrue(set(values) <= set(labels), values)
                if complete:
                    self.assertEqual(set(values), set(labels))
                for index, value in enumerate(values):
                    text = combo.itemText(index)
                    self.assertEqual(text, self.label_for(value, labels))
                    self.assertNotIn("_", text)
                    # Words a language translates, not outside names such as a file's.
                    if setting not in ("MODEL_NAME", "EMBEDDING_MODEL"):
                        self.assertFalse(combo.itemData(index, NAME_ITEM_ROLE), text)
                self.assertIn(values[0], labels)

    def test_models_without_separate_terms_are_names_no_language_translates(self):
        from desktop.Desktop_App import NAME_ITEM_ROLE

        window, _ = self.open_tools()
        combo = self.combo(window, "Generate_Embeddings.py", "MODEL_NAME")
        for index, value in enumerate(stored_values(combo)):
            with self.subTest(model=value):
                self.assertEqual(
                    bool(combo.itemData(index, NAME_ITEM_ROLE)), value not in self.NON_COMMERCIAL
                )

    def test_selecting_by_value_collects_the_value_and_loads_that_plugin(self):
        from desktop.Desktop_App import select_combo_value
        from tools.tool_helpers.Model_Plugins import find_model_plugin

        window, _ = self.open_tools()
        for script, setting, labels, _complete in self.CHANGED:
            path = self.script(window, script)
            combo = self.combo(window, script, setting)
            for value in stored_values(combo):
                with self.subTest(script=script, setting=setting, value=value):
                    self.assertTrue(select_combo_value(combo, value))
                    self.assertEqual(combo.currentText(), self.label_for(value, labels))
                    self.assertEqual(window._collect_tool_settings(path)[setting], value)
                    if setting in ("MODEL_NAME", "EMBEDDING_MODEL"):
                        # The identifier, not the name, finds the plugin that runs the model.
                        self.assertIn(value, find_model_plugin(value).SUPPORTED_MODELS)

    def test_longer_labels_do_not_widen_their_dropdowns(self):
        """A dropdown keeps the minimum width its old options gave it, so no row grows.

        The minimum is what a row needs before it stacks. The reference is a
        dropdown holding the old texts (the stored values and identifiers),
        measured the same way: Float32 (Full Precision) and the model names
        are longer, so those dropdowns cap their minimum (minimum_characters).
        """
        from PySide6.QtWidgets import QComboBox
        from EMAPSSN_Tools import NoScrollComboBox

        window, _ = self.open_tools()
        old_texts = {
            ("Generate_Embeddings.py", "SAVING_MODE"): ["float32", "float16"],
            ("Parse_BLAST_Output.py", "BLAST_LAYOUT"): [
                "standard_outfmt6", "outfmt7_fields", "Custom Columns (1-based indexing)",
            ],
            ("Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION"): [
                "Automatic 32-bit", "float32", "TF32 (Nvidia GPU Only)", "BF16 (Low Precision)",
            ],
            **{
                (script, setting): [
                    f"{model} [non-commercial]" if model in self.NON_COMMERCIAL else model
                    for model in self.MODELS
                ]
                for script, setting in (
                    ("Generate_Embeddings.py", "MODEL_NAME"), ("Embedding_PWA.py", "EMBEDDING_MODEL"),
                )
            },
        }
        for (script, setting), texts in old_texts.items():
            with self.subTest(script=script, setting=setting):
                reference = NoScrollComboBox()
                reference.addItems(texts)
                combo = self.combo(window, script, setting)
                self.assertIsInstance(combo, QComboBox)
                self.assertLessEqual(
                    combo.minimumSizeHint().width(), reference.minimumSizeHint().width()
                )
        # The three whose labels outgrew the old texts cap their minimum at so many characters,
        # which the responsive setup keeps. (Where a window measures a dropdown first decides
        # whether the measure alone shows it, so the cap is pinned too.)
        for script, setting, characters in (
            ("Generate_Embeddings.py", "SAVING_MODE", 2),
            ("Generate_Embeddings.py", "MODEL_NAME", 20),
            ("Embedding_PWA.py", "EMBEDDING_MODEL", 20),
        ):
            with self.subTest(capped=setting):
                combo = self.combo(window, script, setting)
                self.assertEqual(combo.property("minimumCharacters"), characters)
                self.assertEqual(combo.minimumContentsLength(), characters)

    def test_a_label_is_not_a_value(self):
        from desktop.Desktop_App import select_combo_value

        window, _ = self.open_tools()
        combo = self.combo(window, "Generate_Embeddings.py", "MODEL_NAME")
        self.assertFalse(select_combo_value(combo, "ESM-2 650M"))
        combo = self.combo(window, "Generate_Embeddings.py", "SAVING_MODE")
        self.assertFalse(select_combo_value(combo, "Float16 (Half Size)"))

    def test_tips_name_the_options_and_defaults_as_the_dropdowns_do(self):
        from desktop.Tip_Panel import resolve

        window, _ = self.open_tools()
        for script, setting, labels, _complete in self.CHANGED:
            combo = self.combo(window, script, setting)
            entry = window.tip_db[combo]
            if not entry.tip.options:
                continue
            with self.subTest(script=script, setting=setting):
                model = resolve(entry)
                shown = {value: label for value, label in zip(
                    [option.value for option in entry.tip.options if combo.findData(option.value) >= 0],
                    [label for label, _text in model.options],
                )}
                self.assertEqual(
                    shown, {value: combo.itemText(combo.findData(value)) for value in shown}
                )
                self.assertTrue(all(label == self.label_for(value, labels) for value, label in shown.items()))
                default = dict(model.facts).get("Default")
                if default is not None:
                    self.assertIn(default, [combo.itemText(i) for i in range(combo.count())])

    def test_saved_tools_settings_select_the_words_and_export_the_values(self):
        import json

        from PySide6.QtWidgets import QInputDialog, QMessageBox

        saved = {
            "Generate_Embeddings.py": {"MODEL_NAME": "esm2_t12_35m", "SAVING_MODE": "float16"},
            "Embedding_PWA.py": {"EMBEDDING_MODEL": "ankh_large"},
            "Parse_BLAST_Output.py": {"BLAST_LAYOUT": "outfmt7_fields"},
            "Align_Similarity_Matrix.py": {"EXECUTION_MODE": "scalar", "ACCELERATOR_PRECISION": "float32"},
            "Network_Injection.py": {"EXECUTION_MODE": "auto"},
        }
        window, root = self.open_tools(saved)
        for script, section in saved.items():
            path = self.script(window, script)
            for setting, value in section.items():
                with self.subTest(script=script, setting=setting):
                    combo = self.combo(window, script, setting)
                    self.assertEqual(combo.currentData(), value)
                    labels = next(
                        labels for name, key, labels, _ in self.CHANGED if (name, key) == (script, setting)
                    )
                    self.assertEqual(combo.currentText(), self.label_for(value, labels))
                    self.assertEqual(window._collect_tool_settings(path)[setting], value)
        # What an export writes is the value, never the words. (Generate_Embeddings' model is
        # checked against the plugins of the project root, which the isolated project lacks.)
        for script in ("Parse_BLAST_Output.py", "Align_Similarity_Matrix.py"):
            path = self.script(window, script)
            with mock.patch.object(QInputDialog, "getText", return_value=(f"export_{script[:5]}", True)), \
                    mock.patch.object(QMessageBox, "information"), \
                    mock.patch.object(QMessageBox, "critical") as critical:
                window.export_settings(path)
            critical.assert_not_called()
            exported = [p for p in root.rglob(f"export_{script[:5]}*.json")]
            self.assertEqual(len(exported), 1, script)
            document = json.loads(exported[0].read_text(encoding="utf-8"))
            for setting, value in saved[script].items():
                self.assertEqual(document[script][setting], value)
            text = exported[0].read_text(encoding="utf-8")
            for words in (*self.SAVING.values(), *self.BLAST.values(), *self.EXECUTION.values(),
                          *self.MODELS.values()):
                self.assertNotIn(f'"{words}"', text)

    def test_simplified_chinese_translates_the_words_and_leaves_model_names_alone(self):
        from desktop import Desktop_App
        from desktop.Desktop_App import install_translations
        from utilities.Localization import LANGUAGES_DIR, read_catalog

        # The labels need the catalog, not the bundled Chinese font, whose registering and
        # removing would make Qt look fonts up again for every window after this test.
        with mock.patch.dict(Desktop_App.LANGUAGE_FONTS, clear=True):
            installed = install_translations(self.app, "zh_CN")
        self.addCleanup(installed.remove)
        window, _ = self.open_tools()
        zh = {
            (message.context, message.source, message.comment): message.translations[0]
            for message in read_catalog(LANGUAGES_DIR / "zh_CN" / "emapssn_zh_CN.ts")
            if message.translations and message.translations[0]
        }
        wanted = (
            ("Align_Similarity_Matrix.py", "EXECUTION_MODE", self.EXECUTION, "execution mode"),
            ("Generate_Embeddings.py", "SAVING_MODE", self.SAVING, "saving mode"),
            ("Parse_BLAST_Output.py", "BLAST_LAYOUT", {k: v for k, v in self.BLAST.items() if k != "custom_columns"}, "BLAST layout"),
        )
        for script, setting, labels, comment in wanted:
            combo = self.combo(window, script, setting)
            for index, value in enumerate(stored_values(combo)):
                if value not in labels:
                    continue
                with self.subTest(setting=setting, value=value):
                    chinese = zh[("Tools", labels[value], comment)]
                    self.assertEqual(combo.itemText(index), chinese)
                    self.assertNotEqual(chinese, labels[value])
                    self.assertTrue(any("一" <= c <= "鿿" for c in chinese), chinese)
        precision = self.combo(window, "Align_Similarity_Matrix.py", "ACCELERATOR_PRECISION")
        fp32 = precision.itemText(precision.findData("float32"))
        self.assertEqual(fp32, zh[("Tools", "FP32 (Full Precision)", "")])
        self.assertNotEqual(fp32, "FP32 (Full Precision)")
        # A model's name is a name: the same in every language. Only the terms marker translates.
        models = self.combo(window, "Generate_Embeddings.py", "MODEL_NAME")
        for index, value in enumerate(stored_values(models)):
            text = models.itemText(index)
            with self.subTest(model=value):
                self.assertTrue(text.startswith(self.MODELS[value]), text)
                if value not in self.NON_COMMERCIAL:
                    self.assertEqual(text, self.MODELS[value])


if __name__ == "__main__":
    unittest.main()
