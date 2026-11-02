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
from tests.tools_gui_fixtures import isolated_tools_project


class TestEmbeddingMsaGuiLogic(unittest.TestCase):
    def test_complete_network_display_preserves_incomplete_preference(self):
        from PySide6.QtWidgets import QApplication, QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        app = QApplication.instance() or QApplication([])
        # No saved network or directories from the developer's tools_settings.json.
        isolated_tools_project(self)
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


class PerturbedTreeSwitchTests(unittest.TestCase):
    """Neighbor-joining turns Noise-Perturbed Trees off; UPGMA brings back its state.

    Save & Run collects OFF while Neighbor-joining shows it off, since the
    tool would run perturbed Neighbor-joining trees for ON.
    """

    UPGMA = "UPGMA (Fast)"
    NEIGHBOR_JOINING = "Neighbor-joining (Slow)"

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def open_tools(self, tree_method=None, perturbed=None, carried=None):
        """Open the Tools window, saving the tree settings first if they are given."""
        from PySide6.QtWidgets import QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        if carried is None:
            section = {}
            if tree_method is not None:
                section["TREE_METHOD"] = tree_method
            if perturbed is not None:
                section["BOOTSTRAP_TREE"] = perturbed
            isolated_tools_project(self, sections={"Embedding_MSA.py": section})
        with patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser):
            window = ToolsGUI(carried=carried)

        def close():
            window.close()
            window.deleteLater()
            self.app.processEvents()

        self.addCleanup(close)
        path = next(path for path in window.script_data
                    if pathlib.Path(path).name == "Embedding_MSA.py")
        return window, path, window.script_data[path]["inputs"]

    def select(self, inputs, tree_method):
        from desktop.Desktop_App import select_combo_value

        select_combo_value(inputs["TREE_METHOD"]["widget"], tree_method)

    def assert_perturbed(self, window, path, inputs, checked, enabled):
        switch = inputs["BOOTSTRAP_TREE"]["widget"]
        self.assertEqual(switch.isChecked(), checked)
        self.assertEqual(switch.isEnabled(), enabled)
        self.assertEqual(switch.text(), "ON" if checked else "OFF")
        # Its replicate count follows the switch as shown.
        self.assertEqual(inputs["NUM_TREES"]["widget"].isEnabled(), checked)
        # What Save & Run and Export collect.
        self.assertEqual(window._collect_tool_settings(path)["BOOTSTRAP_TREE"], checked)

    def test_upgma_gets_back_the_state_neighbor_joining_turned_off(self):
        for perturbed in (True, False):
            with self.subTest(perturbed=perturbed):
                window, path, inputs = self.open_tools(self.UPGMA, perturbed)
                self.assert_perturbed(window, path, inputs, perturbed, True)
                for _ in range(2):
                    self.select(inputs, self.NEIGHBOR_JOINING)
                    self.assert_perturbed(window, path, inputs, False, False)
                    self.select(inputs, self.UPGMA)
                    self.assert_perturbed(window, path, inputs, perturbed, True)

    def test_the_last_choice_made_with_upgma_is_restored(self):
        window, path, inputs = self.open_tools(self.UPGMA, True)
        switch = inputs["BOOTSTRAP_TREE"]["widget"]
        for choice in (False, True):
            switch.setChecked(choice)
            self.select(inputs, self.NEIGHBOR_JOINING)
            self.select(inputs, self.UPGMA)
            self.assert_perturbed(window, path, inputs, choice, True)

    def test_saved_neighbor_joining_shows_perturbed_trees_off_until_upgma(self):
        # Exported or hand-edited settings can pair them; the tool itself would honour both.
        window, path, inputs = self.open_tools(self.NEIGHBOR_JOINING, True)
        self.assert_perturbed(window, path, inputs, False, False)
        self.select(inputs, self.UPGMA)
        self.assert_perturbed(window, path, inputs, True, True)

    def test_imputed_pairs_follow_the_restored_perturbed_trees(self):
        incomplete = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20,
            expected_edge_count=45,
        )
        with patch("EMAPSSN_Tools.inspect_network_completeness", return_value=incomplete):
            window, path, inputs = self.open_tools(self.UPGMA, True)
            network = inputs["INPUT_NETWORK"]["widget"].combo
            network.addItem("incomplete.h5")
            network.setCurrentText("incomplete.h5")
            imputed = inputs["INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"]["widget"]
            self.assertTrue(imputed.isEnabled())
            self.select(inputs, self.NEIGHBOR_JOINING)
            self.assertFalse(imputed.isEnabled())
            self.select(inputs, self.UPGMA)
            self.assertTrue(imputed.isEnabled())

    def test_a_redraw_keeps_the_choice_neighbor_joining_turned_off(self):
        window, path, inputs = self.open_tools(self.UPGMA, True)
        self.select(inputs, self.NEIGHBOR_JOINING)
        carried = window.language_carry_over()
        self.assertTrue(carried["document"]["Embedding_MSA.py"]["BOOTSTRAP_TREE"])

        redrawn, path, inputs = self.open_tools(carried=carried)
        self.assert_perturbed(redrawn, path, inputs, False, False)
        self.assertEqual(redrawn.language_carry_over()["document"], carried["document"])
        self.select(inputs, self.UPGMA)
        self.assert_perturbed(redrawn, path, inputs, True, True)


if __name__ == "__main__":
    unittest.main()
