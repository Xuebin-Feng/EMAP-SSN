# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window: score-mode and normalization controls follow the network type."""

import json
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Desktop_App import combo_value, select_combo_value  # noqa: E402
from desktop.Viewer_State import decode_document  # noqa: E402
from tests.config_gui_loader import load_config_namespace, open_config_window  # noqa: E402

_NAMESPACE = None
_WINDOW = None
_APP = None
_DIRECTORY = None


def setUpModule():
    # The window reads temporary directories, never viewer_settings.json, so it
    # selects no inputs and starts no input-hashing worker.
    global _NAMESPACE, _WINDOW, _APP, _DIRECTORY
    _NAMESPACE = load_config_namespace()
    _APP = _NAMESPACE["QApplication"].instance() or _NAMESPACE["QApplication"]([])
    _DIRECTORY = tempfile.TemporaryDirectory()
    _WINDOW = open_config_window(_NAMESPACE["ConfigGUI"], _DIRECTORY.name)


def _close_window(window):
    for worker in window._cache_hash_workers.values():
        worker.requestInterruption()
        worker.wait(10000)
    window.close()
    window.deleteLater()
    _APP.processEvents()


def tearDownModule():
    _close_window(_WINDOW)
    _DIRECTORY.cleanup()


def _items(combo):
    return [combo.itemData(index) for index in range(combo.count())]


def _choose_alignment_modes(window, score, norm):
    """Select these modes for an alignment network, as the user would."""
    window._set_network_type_controls("alignment")
    select_combo_value(window.cb_score_mode, score)
    select_combo_value(window.cb_norm_mode, norm)


def _select_blast_network(window, root):
    """Select a real BLAST network and wait for cache discovery to see it."""
    inputs = pathlib.Path(root, "blast_inputs")
    inputs.mkdir()
    (inputs / "set.fasta").write_text(">a\nAAAA\n>b\nAAAT\n", encoding="utf-8")
    with h5py.File(inputs / "blast.h5", "w") as network:
        network.attrs["model_name"] = "BLAST"
        network.create_dataset("headers", data=[b"a", b"b"])
        network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
        network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
        network.create_dataset("score", data=np.asarray([1e-30]))
    window.inputs["SAVED_LAYOUT_DIR"].setText(str(pathlib.Path(root, "layouts")))
    window.inputs["FASTA_DIR"].setText(str(inputs))
    window.inputs["HDF5_DIR"].setText(str(inputs))
    window.cb_fasta.clear()
    window.cb_fasta.addItem("set.fasta")
    window.cb_hdf5.clear()
    window.cb_hdf5.addItem("blast.h5")
    # UMAP mode skips Save & Run's layout-device check, which reads the hardware.
    window.check_umap.setChecked(True)
    # The inputs are hashed in the background; discovery runs when that ends.
    deadline = time.monotonic() + 60
    while window._cache_hash_pending_keys is not None:
        if time.monotonic() > deadline:
            raise AssertionError("cache discovery did not finish")
        _APP.processEvents()
        time.sleep(0.01)


class NormalizationModeRefreshTests(unittest.TestCase):
    def test_leaving_blast_restores_the_alignment_choice_and_its_options(self):
        """Signals stay blocked while the controls change, so the normalization
        list is refreshed explicitly to match the restored score mode."""
        window = _WINDOW
        _choose_alignment_modes(window, "local", "average_sequence")
        self.assertNotIn("alignment_length", _items(window.cb_norm_mode))

        window._set_network_type_controls("blast")
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)
        self.assertEqual(window.cb_norm_mode.currentIndex(), -1)
        self.assertFalse(window.cb_score_mode.isEnabled())
        self.assertFalse(window.cb_norm_mode.isEnabled())

        window._set_network_type_controls("alignment")
        self.assertTrue(window.cb_score_mode.isEnabled())
        self.assertEqual(combo_value(window.cb_score_mode), "local")
        self.assertNotIn("alignment_length", _items(window.cb_norm_mode))
        self.assertEqual(combo_value(window.cb_norm_mode), "average_sequence")
        self.assertFalse(window.cb_score_mode.signalsBlocked())
        self.assertFalse(window.cb_norm_mode.signalsBlocked())


NORMALIZATION_VALUES = ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"]
NORMALIZATION_WORDS = ["Alignment Length", "Shorter Sequence", "Longer Sequence", "Average Sequence"]


class NormalizationModeLabelTests(unittest.TestCase):
    """The dropdown names each mode in words; what is stored stays as it was saved.

    Only the labels read as words (Alignment Length). The stored values
    (alignment_length, ...) are what settings files, profiles, caches and the
    MCP documents hold, so they must not change with the labels.
    """

    def tearDown(self):
        _choose_alignment_modes(_WINDOW, "global", "alignment_length")

    def test_a_global_score_offers_the_four_modes_in_words(self):
        _choose_alignment_modes(_WINDOW, "global", "alignment_length")
        combo = _WINDOW.cb_norm_mode
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], NORMALIZATION_WORDS)
        self.assertEqual(_items(combo), NORMALIZATION_VALUES)
        self.assertEqual(combo.currentText(), "Alignment Length")
        self.assertFalse(any("_" in combo.itemText(i) for i in range(combo.count())))

    def test_a_local_score_leaves_alignment_length_out(self):
        _choose_alignment_modes(_WINDOW, "local", "average_sequence")
        combo = _WINDOW.cb_norm_mode
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], NORMALIZATION_WORDS[1:])
        self.assertEqual(_items(combo), NORMALIZATION_VALUES[1:])
        self.assertEqual(combo.currentText(), "Average Sequence")
        # A choice Alignment Length cannot keep becomes the longer sequence.
        _choose_alignment_modes(_WINDOW, "global", "alignment_length")
        select_combo_value(_WINDOW.cb_score_mode, "local")
        self.assertEqual(combo_value(combo), "longer_sequence")
        self.assertEqual(combo.currentText(), "Longer Sequence")

    def test_the_stored_value_is_read_saved_and_restored_whatever_the_label(self):
        window = _WINDOW
        for value, words in zip(NORMALIZATION_VALUES, NORMALIZATION_WORDS):
            with self.subTest(value=value):
                _choose_alignment_modes(window, "global", value)
                self.assertEqual(window.cb_norm_mode.currentText(), words)
                self.assertEqual(combo_value(window.cb_norm_mode), value)
                self.assertEqual(window.collect_data()["NORM_MODE"], value)
                self.assertEqual(window._cache_setting_values()["normalization"], value)
                saved = window._collect_tab_profile_data("inputs_outputs")
                self.assertEqual(saved["NORM_MODE"], value)
                # A profile (saved by an earlier version too) restores by the stored value.
                profile = dict(_NAMESPACE["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
                profile.update(ALIGNMENT_SCORE="global", NORM_MODE=value)
                _choose_alignment_modes(window, "global", "shorter_sequence")
                window._apply_profile_data("inputs_outputs", profile)
                self.assertEqual(combo_value(window.cb_norm_mode), value)
                self.assertEqual(window.cb_norm_mode.currentText(), words)
                self.assertEqual(window._collect_tab_profile_data("inputs_outputs")["NORM_MODE"], value)

    def test_saved_settings_hold_the_value_not_the_label(self):
        window = _WINDOW
        _choose_alignment_modes(window, "global", "average_sequence")
        original_custom = dict(window._custom_settings)
        with tempfile.TemporaryDirectory() as directory:
            settings_file = pathlib.Path(directory, "viewer_settings.json")
            try:
                with mock.patch.dict(
                    window.save_settings.__globals__, {"DEFAULT_SETTINGS_FILE": str(settings_file)}
                ):
                    self.assertTrue(window.save_settings(), window.tip_panel.text())
            finally:
                window._custom_settings = original_custom
            text = settings_file.read_text(encoding="utf-8")
        self.assertEqual(json.loads(text)["NORM_MODE"], "average_sequence")
        self.assertNotIn("Average Sequence", text)

    def test_the_tip_and_the_statistics_report_name_the_modes_as_the_dropdown_does(self):
        from desktop.Tip_Panel import resolve

        window = _WINDOW
        _choose_alignment_modes(window, "global", "alignment_length")
        names = _NAMESPACE["normalization_names"]()
        self.assertEqual(list(names), NORMALIZATION_VALUES)
        self.assertEqual(list(names.values()), NORMALIZATION_WORDS)
        model = resolve(window.tip_db[window.cb_norm_mode])
        self.assertEqual([label for label, _text in model.options], NORMALIZATION_WORDS)
        # The facts footer keeps naming the setting by its key, and the default by its words.
        self.assertIn(("Setting", "NORM_MODE"), model.facts)
        self.assertIn(("Default", "Alignment Length"), model.facts)
        # The statistics report's metric line quotes the same words.
        for value, words in zip(NORMALIZATION_VALUES, NORMALIZATION_WORDS):
            with self.subTest(value=value):
                summary, _headings = _NAMESPACE["statistics_report_text"]({
                    "blast": False, "score_mode": "global", "normalization": value,
                    "model": "E1_RA", "fasta": "a.fasta", "nodes": 3, "stored_edges": 3,
                    "possible_edges": 3, "maximum": 1.0, "minimum": 0.0, "mean": 0.5,
                })
                self.assertIn(f"Metric: Global Alignment Score with {words} Normalization", summary)
        _choose_alignment_modes(window, "local", "longer_sequence")
        model = resolve(window.tip_db[window.cb_norm_mode])
        self.assertEqual([label for label, _text in model.options], NORMALIZATION_WORDS[1:])
        # Alignment Length is not offered now, and its stored value is not what the tip calls it.
        self.assertNotIn("Default", dict(model.facts))
        self.assertNotIn("alignment_length", [value for _label, value in model.facts])

    def test_simplified_chinese_shows_the_reviewed_wording_of_the_report(self):
        from desktop.Desktop_App import install_translations

        window = _WINDOW
        # Cleanups run last in, first out: the language goes, then the options are English again.
        self.addCleanup(window.update_norm_mode_options)
        installed = install_translations(_APP, "zh_CN")
        self.addCleanup(installed.remove)
        _choose_alignment_modes(window, "global", "alignment_length")
        window.update_norm_mode_options()
        combo = window.cb_norm_mode
        shown = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(_items(combo), NORMALIZATION_VALUES)
        # The statistics report's translations are the dropdown's.
        self.assertEqual(shown, list(_NAMESPACE["normalization_names"]().values()))
        for words, chinese in zip(NORMALIZATION_WORDS, shown):
            with self.subTest(words=words):
                self.assertNotEqual(chinese, words)
                self.assertNotIn("_", chinese)
                self.assertTrue(any("一" <= character <= "鿿" for character in chinese))


class BlastNetworkSaveTests(unittest.TestCase):
    """A BLAST network blanks both modes; Save keeps the alignment choice they hide.

    Saving used to fail with "invalid value for ALIGNMENT_SCORE" because the
    blank controls were saved as empty text.
    """

    def tearDown(self):
        _WINDOW._set_network_type_controls("alignment")

    def test_save_writes_the_choice_the_blank_controls_hide(self):
        window = _WINDOW
        _choose_alignment_modes(window, "local", "average_sequence")
        window._set_network_type_controls("blast")
        original_custom = dict(window._custom_settings)
        with tempfile.TemporaryDirectory() as directory:
            settings_file = pathlib.Path(directory, "viewer_settings.json")
            try:
                with mock.patch.dict(
                    window.save_settings.__globals__,
                    {"DEFAULT_SETTINGS_FILE": str(settings_file)},
                ):
                    self.assertTrue(window.save_settings(), window.tip_panel.text())
            finally:
                window._custom_settings = original_custom
            saved = json.loads(settings_file.read_text(encoding="utf-8"))

        self.assertEqual(saved["ALIGNMENT_SCORE"], "local")
        self.assertEqual(saved["NORM_MODE"], "average_sequence")
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)
        self.assertEqual(window.cb_norm_mode.currentIndex(), -1)

    def test_a_profile_loaded_during_blast_becomes_the_kept_choice(self):
        window = _WINDOW
        _choose_alignment_modes(window, "local", "average_sequence")
        window._set_network_type_controls("blast")
        profile = dict(_NAMESPACE["TAB_PROFILE_SPECS"]["inputs_outputs"]["defaults"])
        profile.update(ALIGNMENT_SCORE="global", NORM_MODE="longer_sequence")

        window._apply_profile_data("inputs_outputs", profile)
        # Until discovery blanks them again, the controls show the profile.
        saved = window._collect_tab_profile_data("inputs_outputs")
        self.assertEqual(
            (saved["ALIGNMENT_SCORE"], saved["NORM_MODE"]), ("global", "longer_sequence")
        )
        window._set_network_type_controls("blast")
        saved = window._collect_tab_profile_data("inputs_outputs")
        self.assertEqual(
            (saved["ALIGNMENT_SCORE"], saved["NORM_MODE"]), ("global", "longer_sequence")
        )

        window._set_network_type_controls("alignment")
        self.assertEqual(combo_value(window.cb_score_mode), "global")
        self.assertEqual(combo_value(window.cb_norm_mode), "longer_sequence")


class BlastSaveAndRunTests(unittest.TestCase):
    """Save & Run saves, then launches, while a real BLAST network is selected."""

    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.TemporaryDirectory()
        cls.window = open_config_window(_NAMESPACE["ConfigGUI"], cls.home.name)

    @classmethod
    def tearDownClass(cls):
        _close_window(cls.window)
        cls.home.cleanup()

    def test_save_and_run_saves_the_hidden_choice_and_launches(self):
        window = self.window
        _choose_alignment_modes(window, "local", "average_sequence")
        _select_blast_network(window, self.home.name)
        self.assertEqual(window.cb_score_mode.currentIndex(), -1)
        self.assertFalse(window.cb_score_mode.isEnabled())

        settings_file = pathlib.Path(self.home.name, "viewer_settings.json")
        handoff = mock.Mock(return_value=object())
        with mock.patch.dict(
            window.save_and_run.__globals__,
            {
                "DEFAULT_SETTINGS_FILE": str(settings_file),
                "_handoff_to_layout_generator": handoff,
            },
        ), mock.patch.object(_NAMESPACE["QMessageBox"], "critical") as critical:
            window.save_and_run()

        self.assertFalse(critical.called, critical.call_args)
        self.assertEqual(handoff.call_count, 1, window.tip_panel.text())
        viewer_snapshot = pathlib.Path(handoff.call_args.args[2]["SSN_VIEWER_SETTINGS_PATH"])
        # The generator would delete its layout snapshot; the mock leaves it.
        layout_snapshot = pathlib.Path(handoff.call_args.args[1])
        try:
            # The Viewer's document has no score fields; it reads them from the cache.
            decode_document(json.loads(viewer_snapshot.read_text(encoding="utf-8")), "viewer")
            layout = decode_document(
                json.loads(layout_snapshot.read_text(encoding="utf-8")), "layout"
            )
        finally:
            viewer_snapshot.unlink()
            layout_snapshot.unlink()

        saved = json.loads(settings_file.read_text(encoding="utf-8"))
        self.assertEqual(
            (saved["ALIGNMENT_SCORE"], saved["NORM_MODE"]), ("local", "average_sequence")
        )
        # A BLAST cache records neither mode, so the generator gets null for both.
        self.assertEqual((layout["ALIGNMENT_SCORE"], layout["NORM_MODE"]), (None, None))


if __name__ == "__main__":
    unittest.main()
