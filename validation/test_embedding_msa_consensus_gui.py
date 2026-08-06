# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offscreen GUI tests for network-aware cophenetic consensus controls."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# SSN_Tools imports hardware discovery for unrelated device dropdowns.  Stub that
# narrow API so this control-state test does not initialize Torch/CUDA DLLs.
hardware_utils = types.ModuleType("utilities.Hardware_Utils")
hardware_utils.device_selection_options = lambda: [("CPU", "cpu")]
hardware_utils.normalize_device_selection = lambda _value: "cpu"
hardware_utils.resolve_device_selection = lambda _value: "cpu"
hardware_utils.get_optimal_device = lambda: "cpu"
sys.modules["utilities.Hardware_Utils"] = hardware_utils

from Cache_Manifest import NetworkCompletenessInfo
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QTextBrowser
import SSN_Tools as tools_gui
import utilities as utilities_package

# Leave later validation modules free to import the real hardware helper.
sys.modules.pop("utilities.Hardware_Utils", None)
if getattr(utilities_package, "Hardware_Utils", None) is hardware_utils:
    delattr(utilities_package, "Hardware_Utils")

# The production description pane uses Chromium.  A text browser is sufficient
# here and keeps the Windows offscreen test independent of WebEngine rendering.
tools_gui.ResponsiveTextBrowser = QTextBrowser


class ConsensusSwitchPresentationTests(unittest.TestCase):
    def test_complete_network_is_automatic_full_consensus(self):
        info = NetworkCompletenessInfo("complete", 4, 6, 6)
        enabled, tip = tools_gui.imputed_consensus_switch_state(info, True, False)
        self.assertFalse(enabled)
        self.assertIn("full cophenetic consensus is automatic", tip)

    def test_incomplete_network_describes_both_modes(self):
        info = NetworkCompletenessInfo("incomplete", 4, 3, 6)

        enabled, partial_tip = tools_gui.imputed_consensus_switch_state(
            info, True, False
        )
        self.assertTrue(enabled)
        self.assertIn("participate in every replicate tree", partial_tip)
        self.assertIn("retain their baseline imputed distances", partial_tip)

        enabled, full_tip = tools_gui.imputed_consensus_switch_state(info, True, True)
        self.assertTrue(enabled)
        self.assertIn("replicate-averaged cophenetic distances", full_tip)

    def test_incomplete_network_requires_active_noise_trees(self):
        info = NetworkCompletenessInfo("incomplete", 4, 3, 6)
        enabled, tip = tools_gui.imputed_consensus_switch_state(info, False, False)
        self.assertFalse(enabled)
        self.assertIn("Enable Noise-Perturbed Trees with UPGMA", tip)

    def test_unknown_and_unselected_networks_show_reasons(self):
        enabled, tip = tools_gui.imputed_consensus_switch_state(None, True, False)
        self.assertFalse(enabled)
        self.assertIn("No network is selected", tip)

        info = NetworkCompletenessInfo("unknown", reason="missing j")
        enabled, tip = tools_gui.imputed_consensus_switch_state(info, True, False)
        self.assertFalse(enabled)
        self.assertIn("missing j", tip)


class ConsensusSwitchOffscreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _write_network(path, sequence_count, edge_count, malformed=False):
        with h5py.File(path, "w") as network:
            network.attrs["model_name"] = "BLAST"
            network.create_dataset(
                "headers",
                data=np.asarray(
                    [f"seq_{index}" for index in range(sequence_count)], dtype="S"
                ),
            )
            network.create_dataset(
                "i", data=np.arange(edge_count, dtype=np.uint32) % sequence_count
            )
            if not malformed:
                network.create_dataset(
                    "j",
                    data=(np.arange(edge_count, dtype=np.uint32) + 1) % sequence_count,
                )
            network.create_dataset(
                "score", data=np.ones(edge_count, dtype=np.float32)
            )

    @staticmethod
    def _write_alignment_network(path):
        with h5py.File(path, "w") as network:
            network.attrs["model_name"] = "esmc_6b"
            network.create_dataset("headers", data=np.asarray(["seq_0", "seq_1"], dtype="S"))
            network.create_dataset("seq_lens", data=np.asarray([10, 12], dtype=np.uint16))
            network.create_dataset("i", data=np.asarray([0], dtype=np.uint32))
            network.create_dataset("j", data=np.asarray([1], dtype=np.uint32))
            for name in ("l_score", "g_score"):
                network.create_dataset(name, data=np.asarray([1.0], dtype=np.float32))
            for name in ("l_len", "g_len"):
                network.create_dataset(name, data=np.asarray([1], dtype=np.uint16))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self._write_network(os.path.join(self.directory.name, "complete.h5"), 3, 3)
        self._write_network(os.path.join(self.directory.name, "incomplete.h5"), 3, 2)
        self._write_network(
            os.path.join(self.directory.name, "invalid.h5"), 3, 2, malformed=True
        )
        self._write_alignment_network(
            os.path.join(self.directory.name, "misleading_[BLAST]_EValue.h5")
        )

        self.window = tools_gui.ToolsGUI()
        self.addCleanup(self.window.close)
        msa_data = next(
            data
            for path, data in self.window.script_data.items()
            if os.path.basename(path) == "Embedding_MSA.py"
        )
        self.inputs = msa_data["inputs"]
        self.network_combo = self.inputs["INPUT_NETWORK"]["widget"].combo
        self.bootstrap = self.inputs["BOOTSTRAP_TREE"]["widget"]
        self.score_mode = self.inputs["ALIGNMENT_SCORE"]["widget"]
        self.norm_mode = self.inputs["NORMALIZATION_MODE"]["widget"]
        self.show_plot = self.inputs["SHOW_REGRESSION_PLOT"]["widget"]
        self.consensus = self.inputs[
            "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"
        ]["widget"]
        self.tree_method = self.inputs["TREE_METHOD"]["widget"]

        self.window.dir_inputs["NETWORK_DIR"].setText(self.directory.name)
        self.network_combo.populate()
        self.tree_method.setCurrentText("UPGMA (Fast)")
        self.bootstrap.setChecked(True)
        QApplication.processEvents()

    def _select(self, filename):
        self.network_combo.setCurrentText(filename)
        QApplication.processEvents()

    def _inputs_for(self, script_name):
        return next(
            data["inputs"]
            for path, data in self.window.script_data.items()
            if os.path.basename(path) == script_name
        )

    def test_requested_tool_rows_have_expected_membership_and_widths(self):
        dynamic_inputs = self._inputs_for("Align_Similarity_Matrix.py")
        embedding = dynamic_inputs["INPUT_HDF5"]["widget"]
        device = dynamic_inputs["DEVICE_SELECTION"]["widget"]
        embedding_row = embedding.parentWidget().parentWidget()
        self.assertIs(embedding_row, device.parentWidget().parentWidget())
        self.assertEqual(embedding_row.property("compactColumnRatio"), "2:1")
        self.assertEqual(embedding_row.layout().stretch(0), 2)
        self.assertEqual(embedding_row.layout().stretch(1), 1)
        self.assertNotIn("ACCELERATOR_LANES", dynamic_inputs)

        substitution_inputs = self._inputs_for("Align_Substitution_Matrix.py")
        sequence_set = substitution_inputs["INPUT_FASTA"]["widget"]
        matrix = substitution_inputs["MATRIX"]["widget"]
        sequence_row = sequence_set.parentWidget().parentWidget()
        self.assertIs(sequence_row, matrix.parentWidget().parentWidget())
        self.assertEqual(sequence_row.property("compactColumnRatio"), "2:1")
        self.assertEqual(sequence_row.layout().stretch(0), 2)
        self.assertEqual(sequence_row.layout().stretch(1), 1)

        similarity_tab = next(
            index
            for index in range(self.window.tabs.count())
            if "Sequence Similarity Calculations"
            in self.window.tabs.tabText(index)
        )
        self.window.tabs.setCurrentIndex(similarity_tab)
        self.window.resize(1600, 1000)
        self.window.show()
        QApplication.processEvents()
        device_x = device.parentWidget().mapTo(self.window, QPoint(0, 0)).x()
        matrix_x = matrix.parentWidget().mapTo(self.window, QPoint(0, 0)).x()
        self.assertEqual(device_x, matrix_x)

        tree = self.inputs["TREE_METHOD"]["widget"]
        bootstrap = self.inputs["BOOTSTRAP_TREE"]["widget"]
        num_trees = self.inputs["NUM_TREES"]["widget"]
        tree_row = tree.parentWidget()
        self.assertIs(tree_row, bootstrap.parentWidget())
        self.assertIs(tree_row, num_trees.parentWidget())
        tree_index = tree_row.layout().indexOf(tree)
        self.assertEqual(tree_row.layout().stretch(tree_index), 1)
        self.assertLess(
            tree_row.layout().indexOf(num_trees),
            tree_row.layout().indexOf(bootstrap),
        )
        self.assertEqual(
            tree_row.layout().indexOf(bootstrap),
            tree_row.layout().count() - 1,
        )

        noise_scale = self.inputs["NOISE_SCALE"]["widget"]
        self.assertFalse(
            noise_scale.parentWidget().objectName().startswith("compactRow_")
        )

        score = self.inputs["ALIGNMENT_SCORE"]["widget"]
        show_plot = self.inputs["SHOW_REGRESSION_PLOT"]["widget"]
        score_row = score.parentWidget()
        self.assertIs(score_row, show_plot.parentWidget())
        self.assertEqual(
            score_row.layout().stretch(score_row.layout().indexOf(score)), 1
        )
        self.assertEqual(
            score_row.layout().stretch(score_row.layout().indexOf(show_plot)), 0
        )
        self.assertIsNot(self.inputs["INPUT_NETWORK"]["widget"].parentWidget(), score_row)

        normalization = self.inputs["NORMALIZATION_MODE"]["widget"]
        imputed = self.inputs[
            "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"
        ]["widget"]
        normalization_row = normalization.parentWidget()
        self.assertIs(normalization_row, imputed.parentWidget())
        self.assertEqual(
            normalization_row.layout().stretch(
                normalization_row.layout().indexOf(normalization)
            ),
            1,
        )
        self.assertEqual(
            normalization_row.layout().stretch(
                normalization_row.layout().indexOf(imputed)
            ),
            0,
        )

        msa_tab = next(
            index
            for index in range(self.window.tabs.count())
            if "Embedding MSA" in self.window.tabs.tabText(index)
        )
        self.window.tabs.setCurrentIndex(msa_tab)
        QApplication.processEvents()
        aligned_x_positions = {
            widget.mapTo(self.window, QPoint(0, 0)).x()
            for widget in (tree, noise_scale, score, normalization)
        }
        self.assertEqual(len(aligned_x_positions), 1)

        for row, button in (
            (tree_row, bootstrap),
            (score_row, show_plot),
            (normalization_row, imputed),
        ):
            button_index = row.layout().indexOf(button)
            label = row.layout().itemAt(button_index - 2).widget()
            gap = button.geometry().left() - label.geometry().right() - 1
            self.assertGreaterEqual(gap, 10)

    def test_widget_tracks_network_bootstrap_tree_method_and_preference(self):
        self._select("incomplete.h5")
        self.assertTrue(self.consensus.isEnabled())
        self.consensus.setChecked(True)
        self.assertIn("replicate-averaged", self.consensus.toolTip())

        self._select("complete.h5")
        self.assertFalse(self.consensus.isEnabled())
        self.assertTrue(self.consensus.isChecked())
        self.assertIn("automatic", self.consensus.toolTip())

        self._select("incomplete.h5")
        self.bootstrap.setChecked(False)
        self.assertFalse(self.consensus.isEnabled())
        self.assertTrue(self.consensus.isChecked())

        self.bootstrap.setChecked(True)
        self.assertTrue(self.consensus.isEnabled())
        self.tree_method.setCurrentText("Neighbor-joining (Slow)")
        self.assertFalse(self.consensus.isEnabled())
        self.assertTrue(self.consensus.isChecked())

        self.tree_method.setCurrentText("UPGMA (Fast)")
        self.bootstrap.setChecked(True)
        self._select("invalid.h5")
        self.assertFalse(self.consensus.isEnabled())
        self.assertIn("Missing required dataset", self.consensus.toolTip())

    def test_probe_cache_uses_path_size_and_mtime(self):
        self.window.network_completeness_cache.clear()
        original_probe = tools_gui.inspect_network_completeness
        with mock.patch.object(
            tools_gui, "inspect_network_completeness", wraps=original_probe
        ) as probe:
            self._select("incomplete.h5")
            self._select("invalid.h5")
            self._select("incomplete.h5")
            self.assertEqual(probe.call_count, 2)

            incomplete_path = os.path.join(self.directory.name, "incomplete.h5")
            with h5py.File(incomplete_path, "a") as network:
                network.attrs["cachebuster"] = "changed"
            self._select("invalid.h5")
            self._select("incomplete.h5")
            self.assertEqual(probe.call_count, 3)

    def test_alignment_controls_follow_metadata_not_filename(self):
        self._select("misleading_[BLAST]_EValue.h5")
        self.assertTrue(self.score_mode.isEnabled())
        self.assertTrue(self.norm_mode.isEnabled())
        self.assertTrue(self.show_plot.isEnabled())
        self.assertEqual(self.score_mode.currentText(), "global")
        self.assertEqual(self.norm_mode.currentText(), "alignment_length")
        self.show_plot.setChecked(True)

        self._select("incomplete.h5")
        self.assertFalse(self.score_mode.isEnabled())
        self.assertFalse(self.norm_mode.isEnabled())
        self.assertFalse(self.show_plot.isChecked())
        self.assertFalse(self.show_plot.isEnabled())
        self.assertIn("unavailable for BLAST", self.show_plot.toolTip())

        self._select("misleading_[BLAST]_EValue.h5")
        self.assertTrue(self.show_plot.isEnabled())
        self.assertFalse(self.show_plot.isChecked())


if __name__ == "__main__":
    unittest.main()
