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
                expected_edge_count=45, records_scoring=True, model_name="m",
            ),
            "unsampled": NetworkCompletenessInfo(
                status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
                model_name="m",
            ),
            "complete": NetworkCompletenessInfo(
                status="complete", sequence_count=10, edge_count=45, expected_edge_count=45,
                records_scoring=True, model_name="m",
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
                                self.assertIn("does not record how its scores were made", switch.toolTip())
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
            records_scoring=True, model_name="m",
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
                        self.assertIn("BLAST network: its scores were not made by this program's alignment", switch.toolTip())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_the_completeness_check_says_whether_the_network_records_its_scoring_and_which_model(self):
        import tempfile
        import h5py
        import numpy as np
        from Cache_Manifest import inspect_embedding_model, inspect_network_completeness

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "network.h5"
            with h5py.File(path, "w") as network:
                network.create_dataset("headers", data=np.array([b"a", b"b", b"c", b"d"]))
                network.create_dataset("i", data=np.array([0, 1], dtype=np.uint16))
                network.create_dataset("j", data=np.array([1, 2], dtype=np.uint16))
            info = inspect_network_completeness(path)
            self.assertEqual((info.status, info.records_scoring, info.model_name, info.sequence_count, info.edge_count),
                             ("incomplete", False, "", 4, 2))
            with h5py.File(path, "a") as network:
                network.attrs["model_name"] = "ankh_base"
                network.attrs["gap_penalties"] = np.array([-2.0, 0.0], dtype=np.float32)
            info = inspect_network_completeness(path)
            self.assertEqual((info.records_scoring, info.model_name), (True, "ankh_base"))
            with h5py.File(path, "a") as network:
                network.attrs["model_name"] = b"esm2 "
            self.assertEqual(inspect_network_completeness(path).model_name, "esm2")
            # An embeddings file: the model it records, from the root attribute alone.
            embeddings = pathlib.Path(directory) / "embeddings.h5"
            with h5py.File(embeddings, "w") as handle:
                handle.attrs["model_name"] = "ankh_base"
            self.assertEqual(inspect_embedding_model(embeddings).model_name, "ankh_base")
            with h5py.File(embeddings, "w") as handle:
                handle.attrs["other"] = 1
            info = inspect_embedding_model(embeddings)
            self.assertEqual((info.status, info.model_name), ("ok", ""))
            self.assertEqual(inspect_embedding_model(pathlib.Path(directory) / "missing.h5").status, "unknown")

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
            records_scoring=True, model_name="m",
        )
        enabled, tip = calibrated_imputation_switch_state(complete_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("Complete network", tip)
        self.assertIn("no pair is missing, so nothing is imputed", tip)

    def test_calibrated_imputation_enabled_for_a_sparse_network_that_records_its_scoring(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
            records_scoring=True, model_name="m",
        )
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=False)
        self.assertTrue(enabled)
        self.assertIn("Incomplete network", tip)
        self.assertIn("20/45 observed pairs", tip)
        self.assertIn("a sample of the missing pairs is aligned with the network's own scoring", tip)

    def test_the_limit_is_ten_percent_of_the_pairs_missing_and_exactly_ten_is_enabled(self):
        # 1,000 sequences have 499,500 pairs: 10.0% missing is 49,950 of them.
        expected = 499_500
        for present_percent, edges, enabled in (
            ("89.9%", 449_050, True),   # 10.1% missing
            ("90.0%", 449_550, True),   # exactly 10.0% missing
            ("90.1%", 450_050, False),  # 9.9% missing
        ):
            with self.subTest(present=present_percent):
                info = NetworkCompletenessInfo(
                    status="incomplete", sequence_count=1_000, edge_count=edges, expected_edge_count=expected,
                    records_scoring=True, model_name="m",
                )
                self.assertEqual(calibrated_imputation_switch_state(info, is_blast=False)[0], enabled)
        info = NetworkCompletenessInfo(
            status="incomplete", sequence_count=1_000, edge_count=450_050, expected_edge_count=expected,
            records_scoring=True, model_name="m",
        )
        tip = calibrated_imputation_switch_state(info, is_blast=False)[1]
        self.assertIn("9.9% of its pairs are missing; calibration is used from 10%.", tip)
        self.assertIn("the run counts the sequences it uses and has the last word", tip)
        # The same constant as Embedding MSA's, the run's own limit.
        from utilities import Prefilter_Calibration
        self.assertEqual(Prefilter_Calibration.MIN_MISSING_FRACTION, 0.10)

    def test_calibrated_imputation_disabled_without_scoring_for_blast_no_network_and_unknown(self):
        sparse_info = NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
        )
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("does not record how its scores were made (its gap penalties), so its missing pairs get the uniform fill.", tip)
        enabled, tip = calibrated_imputation_switch_state(sparse_info, is_blast=True)
        self.assertFalse(enabled)
        self.assertIn("BLAST network: its scores were not made by this program's alignment", tip)
        enabled, tip = calibrated_imputation_switch_state(None, is_blast=False)
        self.assertFalse(enabled)
        self.assertIn("No network is selected", tip)
        enabled, tip = calibrated_imputation_switch_state(
            NetworkCompletenessInfo(status="unknown", reason="Missing required dataset(s): i, j."), is_blast=False
        )
        self.assertFalse(enabled)
        self.assertIn("Missing required dataset(s): i, j", tip)


class NetworkEmbeddingsFieldTests(unittest.TestCase):
    """The Network Embeddings field: usable with the Calibrated Imputation switch, and its warnings about models."""

    NETWORK = NetworkCompletenessInfo(
        status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45,
        records_scoring=True, model_name="net_model",
    )

    def setUp(self):
        import h5py
        from PySide6.QtWidgets import QApplication, QTextBrowser
        from EMAPSSN_Tools import ToolsGUI
        from tools.tool_helpers.Tool_Pipeline import DEFAULT_DIRECTORY_PATHS
        from types import SimpleNamespace

        self.app = QApplication.instance() or QApplication([])
        root = isolated_tools_project(self)
        embeddings = root / DEFAULT_DIRECTORY_PATHS["EMBED_DIR"]
        embeddings.mkdir(parents=True, exist_ok=True)
        for name, model in (("own.h5", "own_model"), ("match.h5", "net_model"), ("other.h5", "other_model"), ("none.h5", None)):
            with h5py.File(embeddings / name, "w") as handle:
                if model is not None:
                    handle.attrs["model_name"] = model
        self.info = self.NETWORK
        for patcher in (
            patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser),
            patch("EMAPSSN_Tools.inspect_network_completeness", side_effect=lambda path: self.info),
            patch("EMAPSSN_Tools.validate_network_schema", return_value=SimpleNamespace(network_type="embedding")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.window = ToolsGUI()
        self.addCleanup(self.close_window)
        self.path = next(key for key in self.window.script_data if pathlib.Path(key).name == "Embedding_MSA.py")
        self.inputs = self.window.script_data[self.path]["inputs"]
        self.network = self.inputs["INPUT_NETWORK"]["widget"].combo
        self.switch = self.inputs["CALIBRATED_IMPUTATION"]["widget"]
        self.field = self.inputs["INPUT_NETWORK_EMBED"]["widget"]
        self.second = self.field.combo
        self.own = self.inputs["INPUT_EMBED"]["widget"].combo
        self.choose(self.own, "own.h5")
        self.use_network("sparse.h5")

    def close_window(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    @staticmethod
    def choose(combo, name):
        index = combo.findText(name)
        combo.setCurrentIndex(index)
        assert combo.currentText() == name, (name, [combo.itemText(i) for i in range(combo.count())])

    def use_network(self, name, info=None):
        if info is not None:
            self.info = info
        self.network.addItem(name)
        self.network.setCurrentText(name)

    def choose_none(self):
        self.second.setCurrentIndex(0)
        self.assertEqual(self.second.currentData(), "")

    def test_the_field_lists_the_embeddings_with_an_empty_choice_first_and_collects_it(self):
        items = [self.second.itemText(i) for i in range(self.second.count())]
        self.assertEqual(items[0], "")
        self.assertEqual(self.second.itemData(0), "")
        for label in items:
            self.assertNotIn(label.strip().lower(), ("(none)", "none"))
        self.assertEqual(sorted(items[1:]), ["match.h5", "none.h5", "other.h5", "own.h5"])
        self.assertEqual(self.window._collect_tool_settings(self.path)["INPUT_NETWORK_EMBED"], "")
        self.choose(self.second, "match.h5")
        self.assertEqual(self.window._collect_tool_settings(self.path)["INPUT_NETWORK_EMBED"], "match.h5")
        self.choose_none()
        self.assertEqual(self.window._collect_tool_settings(self.path)["INPUT_NETWORK_EMBED"], "")

    def test_a_blank_choice_stays_selected_when_the_dropdown_refills_itself_and_does_not_jump_to_a_file(self):
        self.choose_none()
        self.second.populate()
        self.assertEqual(self.second.currentIndex(), 0)
        self.assertEqual(self.second.currentText(), "")
        self.assertEqual(self.second.chosen(), "")
        self.choose(self.second, "match.h5")
        self.second.populate()
        self.assertEqual(self.second.currentText(), "match.h5")

    def test_a_saved_empty_value_loads_as_the_blank_entry_and_a_saved_file_loads_as_that_file(self):
        self.assertEqual(self.second.currentIndex(), 0)
        self.assertEqual(self.second.currentText(), "")
        self.assertEqual(self.window._collect_tool_settings(self.path)["INPUT_NETWORK_EMBED"], "")
        self.second.select_file("match.h5")
        self.assertEqual(self.second.currentText(), "match.h5")
        self.assertEqual(self.second.findText(""), 0)

    def test_the_field_is_usable_only_while_the_switch_is_enabled_and_on(self):
        self.switch.setChecked(True)
        self.assertTrue(self.switch.isEnabled())
        self.assertTrue(self.field.isEnabled())
        self.switch.setChecked(False)
        self.assertFalse(self.field.isEnabled())
        self.assertIn("Not used while Calibrated Imputation is off.", self.field.toolTip())
        self.switch.setChecked(True)
        self.assertTrue(self.field.isEnabled())
        # A network the switch is greyed for: the field is greyed with that reason, and keeps its choice.
        self.choose(self.second, "match.h5")
        self.use_network("complete.h5", NetworkCompletenessInfo(
            status="complete", sequence_count=10, edge_count=45, expected_edge_count=45, records_scoring=True,
            model_name="net_model"))
        self.assertFalse(self.switch.isEnabled())
        self.assertFalse(self.field.isEnabled())
        self.assertIn("Not used: Complete network", self.field.toolTip())
        self.assertEqual(self.second.currentText(), "match.h5")
        # Only 5% missing: greyed, with the share.
        self.use_network("nearly.h5", NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=43, expected_edge_count=45, records_scoring=True,
            model_name="net_model"))
        self.assertFalse(self.field.isEnabled())
        self.assertIn("4.4% of its pairs are missing; calibration is used from 10%.", self.field.toolTip())
        self.use_network("sparse_again.h5", self.NETWORK)
        self.assertTrue(self.field.isEnabled())

    def test_a_second_file_of_the_networks_model_gets_no_warning(self):
        self.choose(self.second, "match.h5")
        self.assertNotIn("The run itself decides", self.field.toolTip())
        self.assertIn("The embeddings the network was made from", self.field.toolTip())

    def test_a_second_file_of_another_model_or_of_none_warns_but_never_blocks(self):
        for name, text in (
            ("other.h5", "other.h5 is of model other_model, but the network was made with net_model."),
            ("none.h5", "none.h5 records no embedding model, so it cannot be checked against the network's."),
        ):
            with self.subTest(file=name):
                self.choose(self.second, name)
                tip = self.field.toolTip()
                self.assertIn(text, tip)
                self.assertIn("The run itself decides: the model, a check on 32 of the network's own edges, and every sequence being present.", tip)
                self.assertTrue(self.field.isEnabled())
                self.assertEqual(self.window._collect_tool_settings(self.path)["INPUT_NETWORK_EMBED"], name)

    def test_with_the_field_empty_the_embedding_sets_model_is_checked_instead(self):
        self.choose_none()
        # own_model against the network's net_model: a warning on this field.
        tip = self.field.toolTip()
        self.assertIn("With this field empty the Embedding Set own.h5 (model own_model) serves, but the network was made with net_model: choose the network's embeddings here.", tip)
        self.assertIn("The run itself decides", tip)
        self.choose(self.own, "match.h5")
        self.assertNotIn("The run itself decides", self.field.toolTip())
        self.choose(self.own, "none.h5")
        self.assertIn("it records no embedding model", self.field.toolTip())

    def test_the_models_are_read_once_per_file(self):
        import EMAPSSN_Tools

        with patch.object(EMAPSSN_Tools, "inspect_embedding_model", wraps=EMAPSSN_Tools.inspect_embedding_model) as inspect:
            for _ in range(3):
                self.choose(self.second, "other.h5")
                self.choose(self.second, "match.h5")
                self.choose_none()
            asked = {pathlib.Path(call.args[0]).name for call in inspect.call_args_list}
            self.assertLessEqual(len(inspect.call_args_list), 3)  # other.h5, match.h5 and own.h5 at most, then the cache
            self.assertTrue(asked <= {"other.h5", "match.h5", "own.h5"})

    def test_a_network_that_records_no_model_cannot_be_checked(self):
        self.use_network("unlabelled.h5", NetworkCompletenessInfo(
            status="incomplete", sequence_count=10, edge_count=20, expected_edge_count=45, records_scoring=True))
        self.choose(self.second, "match.h5")
        self.assertIn("The network records no model, so match.h5 (model net_model) cannot be checked against it.", self.field.toolTip())


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
            records_scoring=True, model_name="m",
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
