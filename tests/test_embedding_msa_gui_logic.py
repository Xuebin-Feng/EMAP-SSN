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
    calibrated_imputation_switch_state,
    isotonic_regression_switch_state,
)
from tests.tools_gui_fixtures import isolated_tools_project


class TestEmbeddingMsaGuiLogic(unittest.TestCase):
    def test_the_calibration_switch_follows_the_network_and_keeps_its_state(self):
        from PySide6.QtWidgets import QApplication, QTextBrowser
        from EMAPSSN_Tools import ToolsGUI

        app = QApplication.instance() or QApplication([])
        # No saved network or directories from the developer's tools_settings.json.
        isolated_tools_project(self)
        states = {
            "sampled": NetworkCompletenessInfo(
                status="incomplete", sequence_count=10, edge_count=20,
                expected_edge_count=45, calibration_pairs=7,
            ),
            "unsampled": NetworkCompletenessInfo(
                status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
            ),
            "complete": NetworkCompletenessInfo(
                status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
                calibration_pairs=7,
            ),
            "unknown": NetworkCompletenessInfo(status="unknown", reason="Missing required dataset(s): i."),
        }
        with patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser), patch(
            "EMAPSSN_Tools.inspect_network_completeness", return_value=states["sampled"]
        ) as inspect:
            window = ToolsGUI()
            try:
                data = next(value for key, value in window.script_data.items()
                            if pathlib.Path(key).name == "Embedding_MSA.py")
                inputs = data["inputs"]
                network = inputs["INPUT_NETWORK"]["widget"].combo
                switch = inputs["CALIBRATED_IMPUTATION"]["widget"]
                path = next(key for key in window.script_data if pathlib.Path(key).name == "Embedding_MSA.py")
                # The switch is ON by default and has no say in the perturbed trees or the tree method.
                self.assertTrue(switch.isChecked())
                for method in (0, 1):
                    inputs["TREE_METHOD"]["widget"].setCurrentIndex(method)
                    for perturbed in (True, False):
                        inputs["BOOTSTRAP_TREE"]["widget"].setChecked(perturbed)
                        network.addItem("sampled.h5")
                        network.setCurrentText("sampled.h5")
                        self.assertTrue(switch.isEnabled(), (method, perturbed))
                for checked in (True, False):
                    for name, info in states.items():
                        with self.subTest(checked=checked, network=name):
                            inspect.return_value = states["sampled"]
                            network.addItem("sampled.h5")
                            network.setCurrentText("sampled.h5")
                            switch.setChecked(checked)
                            inspect.return_value = info
                            network.addItem(f"{name}.h5")
                            network.setCurrentText(f"{name}.h5")
                            self.assertEqual(switch.isEnabled(), name == "sampled")
                            # Greyed out or not, the state stays and is what a run collects.
                            self.assertEqual(switch.isChecked(), checked)
                            self.assertEqual(window._collect_tool_settings(path)["CALIBRATED_IMPUTATION"], checked)
                            if name == "unsampled":
                                self.assertIn("stores no calibration sample, so its missing pairs get the uniform fill",
                                              switch.toolTip())
                            if name == "complete":
                                self.assertIn("Not applicable", switch.toolTip())
                    inspect.return_value = states["sampled"]
                    network.setCurrentText("sampled.h5")
                    self.assertTrue(switch.isEnabled())
                    network.setCurrentIndex(-1)
                    self.assertFalse(switch.isEnabled())
                    self.assertEqual(switch.isChecked(), checked)
                    self.assertIn("No network is selected", switch.toolTip())
                    plot = inputs["SHOW_REGRESSION_PLOT"]["widget"]
                    self.assertFalse(plot.isEnabled())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_a_blast_network_greys_the_calibration_switch(self):
        from PySide6.QtWidgets import QApplication, QTextBrowser
        from EMAPSSN_Tools import ToolsGUI
        from Cache_Manifest import NetworkMetadata

        app = QApplication.instance() or QApplication([])
        isolated_tools_project(self)
        sampled = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
            calibration_pairs=7,
        )
        with patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser), patch(
            "EMAPSSN_Tools.inspect_network_completeness", return_value=sampled
        ), patch("EMAPSSN_Tools.validate_network_schema") as validate:
            window = ToolsGUI()
            try:
                data = next(value for key, value in window.script_data.items()
                            if pathlib.Path(key).name == "Embedding_MSA.py")
                network = data["inputs"]["INPUT_NETWORK"]["widget"].combo
                switch = data["inputs"]["CALIBRATED_IMPUTATION"]["widget"]
                for kind, enabled in (("alignment", True), ("blast", False), ("alignment", True)):
                    validate.return_value = NetworkMetadata("model", kind)
                    network.addItem(f"{kind}-{enabled}.h5")
                    network.setCurrentText(f"{kind}-{enabled}.h5")
                    self.assertEqual(switch.isEnabled(), enabled, kind)
                    self.assertTrue(switch.isChecked())
                    if not enabled:
                        self.assertIn("BLAST networks store no calibration sample", switch.toolTip())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_the_completeness_check_counts_the_stored_calibration_sample(self):
        import tempfile
        import h5py
        import numpy as np
        from Cache_Manifest import inspect_network_completeness

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "network.h5"
            with h5py.File(path, "w") as network:
                network.create_dataset("headers", data=np.array([b"a", b"b", b"c", b"d"]))
                network.create_dataset("i", data=np.array([0, 1], dtype=np.uint16))
                network.create_dataset("j", data=np.array([1, 2], dtype=np.uint16))
            self.assertEqual(inspect_network_completeness(path).calibration_pairs, 0)
            with h5py.File(path, "a") as network:
                network.create_dataset("calibration_i", data=np.array([0, 0, 2], dtype=np.uint16))
                network.create_dataset("calibration_j", data=np.array([2, 3, 3], dtype=np.uint16))
                # A sample missing its weights is no sample.
                self.assertEqual(inspect_network_completeness(path).calibration_pairs, 0)
                network.create_dataset("calibration_weight", data=np.ones(3))
            info = inspect_network_completeness(path)
            self.assertEqual((info.status, info.calibration_pairs), ("incomplete", 3))

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

    def test_calibrated_imputation_disabled_for_complete_network(self):
        complete_info = NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
            calibration_pairs=5,
        )
        enabled, tip = calibrated_imputation_switch_state(complete_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("Complete network", tip)
        self.assertIn("no pair is missing, so nothing is imputed", tip)

    def test_calibrated_imputation_enabled_for_a_sparse_network_with_a_sample(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
            calibration_pairs=2_000,
        )
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=False)
        self.assertTrue(enabled)
        self.assertIn("Incomplete network", tip)
        self.assertIn("20/45 observed pairs", tip)
        self.assertIn("It stores a calibration sample of 2,000 pairs.", tip)

    def test_calibrated_imputation_disabled_without_a_sample_for_blast_no_network_and_unknown(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
        )
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("This network stores no calibration sample, so its missing pairs get the uniform fill.", tip)
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=True)
        self.assertFalse(enabled)
        self.assertIn("BLAST networks store no calibration sample", tip)
        enabled, tip = calibrated_imputation_switch_state(None, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("No network is selected", tip)
        enabled, tip = calibrated_imputation_switch_state(
            NetworkCompletenessInfo(status="unknown", reason="Missing required dataset(s): i, j."), is_blast=False
        )
        self.assertFalse(enabled)
        self.assertIn("Missing required dataset(s): i, j", tip)


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

    def test_the_calibration_switch_ignores_the_tree_method_and_perturbed_trees(self):
        sampled = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
            calibration_pairs=9,
        )
        with patch("EMAPSSN_Tools.inspect_network_completeness", return_value=sampled):
            window, path, inputs = self.open_tools(self.UPGMA, True)
            network = inputs["INPUT_NETWORK"]["widget"].combo
            network.addItem("sampled.h5")
            network.setCurrentText("sampled.h5")
            calibrated = inputs["CALIBRATED_IMPUTATION"]["widget"]
            self.assertTrue(calibrated.isEnabled())
            self.select(inputs, self.NEIGHBOR_JOINING)
            self.assertTrue(calibrated.isEnabled())
            inputs["BOOTSTRAP_TREE"]["widget"].setChecked(False)
            self.select(inputs, self.UPGMA)
            self.assertTrue(calibrated.isEnabled())

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
