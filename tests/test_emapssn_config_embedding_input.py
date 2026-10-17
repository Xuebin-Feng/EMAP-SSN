# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Configuration window: the optional embedding input.

Like the MSA, the embedding is a dropdown of the files in a directory with a
blank first entry, which selects none. It sits below the MSA row, its folder
on the Directories tab with the reusable roots (the fourth, below the Analysis
Results Directory), and it is saved, restored, profiled and handed to the
Viewer as EMBED_FILE.
"""

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from tests.config_gui_loader import load_config_namespace, open_config_window


class EmbeddingInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.namespace = load_config_namespace()
        cls.app = cls.namespace["QApplication"].instance() or cls.namespace["QApplication"]([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.embeddings = self.root / "Embeddings"
        self.embeddings.mkdir()
        for name in ("set_[esmc_6b]_embeddings.h5", "set_[ankh_base]_embeddings.h5", "notes.txt"):
            (self.embeddings / name).write_bytes(b"")
        with redirect_stdout(io.StringIO()):
            self.window = open_config_window(self.namespace["ConfigGUI"], self.root)
        self.addCleanup(self.close_window)

    def close_window(self):
        self.window._cache_recheck_timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def items(self, combo):
        return [combo.itemText(index) for index in range(combo.count())]

    def form_row(self, key):
        label = self.window.labels[key]
        layout = label.parentWidget().layout()
        row, _role = layout.getWidgetPosition(label)
        return layout, row

    def test_the_row_lists_the_embedding_folders_h5_files_below_the_msa_row(self):
        combo = self.window.cb_embed
        self.assertIs(self.window.inputs["EMBED_FILE"], combo)
        self.assertEqual(self.window.labels["EMBED_FILE"].text(), "Embedding Input (.h5):")
        combo.refresh_callback()
        self.assertEqual(self.items(combo)[0], "")
        self.assertEqual(
            sorted(self.items(combo)[1:]),
            ["set_[ankh_base]_embeddings.h5", "set_[esmc_6b]_embeddings.h5"],
        )
        self.assertEqual(combo.currentText(), "")
        msa_layout, msa_row = self.form_row("MSA_FILE")
        embed_layout, embed_row = self.form_row("EMBED_FILE")
        self.assertIs(embed_layout, msa_layout)
        self.assertEqual(embed_row, msa_row + 1)

    def test_the_folder_row_follows_the_analysis_results_folder_and_refreshes_the_list(self):
        self.assertEqual(self.window.labels["EMBED_DIR"].text(), "Embedding Directory:")
        self.assertEqual(Path(self.window.inputs["EMBED_DIR"].text()), self.embeddings)
        _, results_dir_row = self.form_row("ANALYSIS_RESULT_DIR")
        _, embed_dir_row = self.form_row("EMBED_DIR")
        self.assertEqual(embed_dir_row, results_dir_row + 1)
        other = self.root / "other_embeddings"
        other.mkdir()
        (other / "subset_[esmc_6b]_embeddings.h5").write_bytes(b"")
        self.window.inputs["EMBED_DIR"].setText(str(other))
        self.assertEqual(self.items(self.window.cb_embed), ["", "subset_[esmc_6b]_embeddings.h5"])

    def test_the_choice_is_saved_as_a_path_in_its_folder_and_blank_as_empty(self):
        window = self.window
        window.cb_embed.refresh_callback()
        window.cb_embed.setCurrentText("set_[esmc_6b]_embeddings.h5")
        expected = os.path.join(window.inputs["EMBED_DIR"].text(),
                                "set_[esmc_6b]_embeddings.h5").replace("\\", "/")
        self.assertEqual(window.collect_data()["EMBED_FILE"], expected)
        self.assertEqual(window._collect_tab_profile_data("inputs_outputs")["EMBED_FILE"], expected)
        window.cb_embed.setCurrentText("")
        self.assertEqual(window.collect_data()["EMBED_FILE"], "")
        self.assertEqual(window._collect_tab_profile_data("inputs_outputs")["EMBED_FILE"], "")

    def test_a_profile_restores_the_choice_by_file_name(self):
        window = self.window
        data = window._normalize_profile_data(
            "inputs_outputs", window._collect_tab_profile_data("inputs_outputs")
        )
        data["EMBED_FILE"] = "$input_file$/elsewhere/set_[ankh_base]_embeddings.h5"
        window._apply_profile_data("inputs_outputs", data)
        self.assertEqual(window.cb_embed.currentText(), "set_[ankh_base]_embeddings.h5")
        data["EMBED_FILE"] = ""
        window._apply_profile_data("inputs_outputs", data)
        self.assertEqual(window.cb_embed.currentText(), "")

    def test_the_viewer_snapshot_carries_the_embedding(self):
        window = self.window
        window.cb_embed.refresh_callback()
        window.cb_embed.setCurrentText("set_[esmc_6b]_embeddings.h5")
        snapshot = self.namespace["_create_viewer_settings_snapshot"](window.collect_data())
        self.addCleanup(os.unlink, snapshot)
        with open(snapshot, encoding="utf-8") as handle:
            document = json.load(handle)
        self.assertTrue(document["embedding"]["EMBED_FILE"].endswith("set_[esmc_6b]_embeddings.h5"))
        self.assertEqual(Path(document["directories"]["EMBED_DIR"]), self.embeddings)

    def test_saved_settings_apply_an_empty_embedding(self):
        namespace = self.namespace
        old = namespace["EMBED_FILE"]
        self.addCleanup(namespace.__setitem__, "EMBED_FILE", old)
        namespace["apply_viewer_settings"]({"EMBED_FILE": "x_embeddings.h5"})
        self.assertEqual(namespace["EMBED_FILE"], "x_embeddings.h5")
        namespace["apply_viewer_settings"]({"EMBED_FILE": ""})
        self.assertEqual(namespace["EMBED_FILE"], "")

    def test_both_rows_have_tips(self):
        for key in ("EMBED_FILE", "EMBED_DIR"):
            with self.subTest(key=key):
                self.assertIn(self.window.inputs[key], self.window.tip_db)
                self.assertIn(self.window.labels[key], self.window.tip_db)


if __name__ == "__main__":
    unittest.main()
