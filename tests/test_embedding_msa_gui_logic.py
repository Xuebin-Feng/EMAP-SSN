import pathlib
import sys
import unittest

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
