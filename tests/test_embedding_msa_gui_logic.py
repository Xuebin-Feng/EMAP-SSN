import pathlib
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Cache_Manifest import NetworkCompletenessInfo
from EMAPSSN_Tools import (
    imputed_consensus_switch_state,
    isotonic_regression_switch_state,
)


class TestEmbeddingMsaGuiLogic(unittest.TestCase):
    def test_complete_network_display_preserves_incomplete_preference(self):
        from PySide6.QtWidgets import QApplication, QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        app = QApplication.instance() or QApplication([])
        incomplete = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20,
            expected_edge_count=45,
        )
        complete = NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45,
            expected_edge_count=45,
        )
        with patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser), patch(
            "EMAPSSN_Tools.inspect_network_completeness", return_value=incomplete
        ) as inspect:
            window = ToolsGUI()
            try:
                data = next(value for key, value in window.script_data.items()
                            if pathlib.Path(key).name == "Embedding_MSA.py")
                inputs = data["inputs"]
                network = inputs["INPUT_NETWORK"]["widget"].combo
                switch = inputs["INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"]["widget"]
                inputs["TREE_METHOD"]["widget"].setCurrentIndex(0)
                inputs["BOOTSTRAP_TREE"]["widget"].setChecked(True)
                network.addItem("test-missing-incomplete.h5")
                network.setCurrentText("test-missing-incomplete.h5")
                for preference in (True, False):
                    switch.setChecked(preference)
                    inspect.return_value = complete
                    network.addItem(f"test-missing-complete-{preference}.h5")
                    network.setCurrentText(f"test-missing-complete-{preference}.h5")
                    self.assertFalse(switch.isEnabled())
                    self.assertFalse(switch.isChecked())
                    self.assertEqual(switch.text(), "OFF")
                    self.assertEqual(switch.property("incomplete_network_preference"), preference)
                    self.assertIn("Not applicable", switch.toolTip())
                    inspect.return_value = incomplete
                    network.setCurrentText("test-missing-incomplete.h5")
                    self.assertTrue(switch.isEnabled())
                    self.assertEqual(switch.isChecked(), preference)
                    self.assertEqual(switch.text(), "ON" if preference else "OFF")
                    plot = inputs["SHOW_REGRESSION_PLOT"]["widget"]
                    plot.setChecked(True)
                    network.setCurrentIndex(-1)
                    for control in (switch, plot):
                        self.assertFalse(control.isEnabled())
                        self.assertFalse(control.isChecked())
                        self.assertEqual(control.text(), "OFF")
                    self.assertEqual(switch.property("incomplete_network_preference"), preference)
                    network.setCurrentText("test-missing-incomplete.h5")
                    self.assertTrue(switch.isEnabled())
                    self.assertEqual(switch.isChecked(), preference)
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_isotonic_regression_disabled_for_blast(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete",
            sequence_count=10,
            edge_count=20,
            expected_edge_count=45,
        )
        enabled, tip = isotonic_regression_switch_state(sparse_info, is_blast=True)
        self.assertFalse(enabled)
        self.assertIn("unavailable for BLAST networks", tip)

    def test_isotonic_regression_disabled_for_no_network(self):
        enabled, tip = isotonic_regression_switch_state(None, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("No network is selected", tip)

    def test_isotonic_regression_disabled_for_unknown_network(self):
        unknown_info = NetworkCompletenessInfo(
            status="unknown",
            reason="Missing required dataset(s): i, j.",
        )
        enabled, tip = isotonic_regression_switch_state(unknown_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("Missing required dataset(s): i, j", tip)

    def test_isotonic_regression_disabled_for_complete_dense_network(self):
        complete_info = NetworkCompletenessInfo(
            status="complete",
            sequence_count=10,
            edge_count=45,
            expected_edge_count=45,
        )
        enabled, tip = isotonic_regression_switch_state(complete_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("Complete network", tip)
        self.assertIn("only available for sparse networks", tip)
        self.assertIn("45/45 observed pairs", tip)

    def test_isotonic_regression_enabled_for_sparse_network(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete",
            sequence_count=10,
            edge_count=20,
            expected_edge_count=45,
        )
        default_tip = "Custom default plot tooltip."
        enabled, tip = isotonic_regression_switch_state(
            sparse_info, is_blast=False, default_tip=default_tip
        )
        self.assertTrue(enabled)
        self.assertIn("Incomplete network", tip)
        self.assertIn("20/45 observed pairs", tip)
        self.assertIn(default_tip, tip)

    def test_imputed_consensus_disabled_for_complete_network(self):
        complete_info = NetworkCompletenessInfo(
            status="complete",
            sequence_count=10,
            edge_count=45,
            expected_edge_count=45,
        )
        enabled, tip = imputed_consensus_switch_state(
            complete_info, noise_trees_active=True, checked=False
        )
        self.assertFalse(enabled)
        self.assertIn("Complete network", tip)
        self.assertIn("full cophenetic consensus is automatic", tip)

    def test_imputed_consensus_enabled_for_sparse_network_with_noise_trees(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete",
            sequence_count=10,
            edge_count=20,
            expected_edge_count=45,
        )
        enabled, tip = imputed_consensus_switch_state(
            sparse_info, noise_trees_active=True, checked=True
        )
        self.assertTrue(enabled)
        self.assertIn("Incomplete network", tip)
        self.assertIn("replicate-averaged cophenetic distances", tip)

    def test_imputed_consensus_disabled_when_noise_trees_inactive(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete",
            sequence_count=10,
            edge_count=20,
            expected_edge_count=45,
        )
        enabled, tip = imputed_consensus_switch_state(
            sparse_info, noise_trees_active=False, checked=False
        )
        self.assertFalse(enabled)
        self.assertIn("Enable Noise-Perturbed Trees with UPGMA", tip)


if __name__ == "__main__":
    unittest.main()
