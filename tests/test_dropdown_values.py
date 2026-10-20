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


if __name__ == "__main__":
    unittest.main()
