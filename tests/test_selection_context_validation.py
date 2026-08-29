"""Regression coverage for strict, context-aware Boolean selections."""

import io
import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Command_Engine
import SSN_Config as cfg
from commands import color as color_command
from commands import export as export_command
from commands import group as group_command
from commands import hide as hide_command
from commands import select as select_command
from web_ui import meta_backend


class FakeAlignmentRows:
    def __init__(self, columns):
        self.columns = {
            (int(column), str(residue).upper()): np.asarray(mask, dtype=bool)
            for (column, residue), mask in columns.items()
        }

    def bulk_residue_check(self, column, residue):
        return self.columns.get(
            (int(column), str(residue).upper()),
            np.zeros(3, dtype=bool),
        )


class StrictExpressionTests(unittest.TestCase):
    def setUp(self):
        self.headers = ["node_a", "node_b", "node_c"]
        self.viewer_to_aln = np.array([0, 1, 2], dtype=int)
        self.valid_indices = np.array([0, 1, 2], dtype=int)
        self.alignment = SimpleNamespace(
            aln=FakeAlignmentRows(
                {
                    (0, "A"): [False, False, False],
                    (2, "K"): [True, False, True],
                    (3, "K"): [False, True, False],
                    (4, "A"): [True, True, False],
                }
            ),
            label_to_col={"10": 0, "10.1": 1, "-1": 2, "-1.1": 3, "0": 4},
            col_to_label={0: "10", 1: "10.1", 2: "-1", 3: "-1.1", 4: "0"},
        )
        self.metadata = {
            "Length": {
                "type": "number",
                "values": np.array([100.0, 200.0, 300.0]),
            },
            "Organism": {
                "type": "text",
                "values": np.array(["alpha", "beta", "gamma"], dtype=object),
            },
        }

    def evaluate(self, expression, **overrides):
        arguments = {
            "cluster_labels": np.array([1, 2, -1]),
            "group_labels": [{"alpha"}, set(), {"omega"}],
            "alignment": self.alignment,
            "metadata": self.metadata,
        }
        arguments.update(overrides)
        return Command_Engine.parse_advanced_expression(
            expression,
            self.viewer_to_aln,
            self.valid_indices,
            self.headers,
            **arguments,
        )

    def test_existing_cluster_noise_and_group_targets_evaluate(self):
        np.testing.assert_array_equal(self.evaluate("#cluster_1#"), [True, False, False])
        np.testing.assert_array_equal(self.evaluate("#noise#"), [False, False, True])
        np.testing.assert_array_equal(self.evaluate("#ALPHA#"), [True, False, False])

    def test_missing_cluster_group_and_noise_raise_context_errors(self):
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "cluster_99"):
            self.evaluate("#cluster_99#")
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "Group 'missing'"):
            self.evaluate("#missing#")
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "Noise does not exist"):
            self.evaluate("#noise#", cluster_labels=np.array([1, 2, 2]))

    def test_available_target_suggestions_are_capped_at_ten(self):
        labels = np.arange(1, 13, dtype=int)
        with self.assertRaises(Command_Engine.SelectionContextError) as raised:
            Command_Engine.parse_advanced_expression(
                "#cluster_99#",
                np.full(12, -1, dtype=int),
                np.array([], dtype=int),
                [f"node_{index}" for index in range(12)],
                cluster_labels=labels,
                group_labels=[set() for _ in range(12)],
            )
        self.assertIn("(+2 more)", str(raised.exception))

    def test_alignment_context_and_displayed_position_are_required(self):
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "no alignment"):
            self.evaluate("A10", alignment=None)
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "position '11'"):
            self.evaluate("A11")
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "not a valid"):
            self.evaluate("A10..1")
        np.testing.assert_array_equal(self.evaluate("A10"), [False, False, False])

    def test_negative_positions_require_parentheses_and_support_boolean_logic(self):
        np.testing.assert_array_equal(
            self.evaluate("K(-1)&!A0"),
            [False, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("K(-1.1)"),
            [False, True, False],
        )

        for expression, replacement in (("K-1", "K(-1)"), ("K-1.1", "K(-1.1)")):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(
                    Command_Engine.SelectionExpressionError,
                    re.escape(replacement),
                ):
                    self.evaluate(expression)

    def test_empty_or_malformed_target_does_not_produce_scalar_mask(self):
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "malformed targets"):
            self.evaluate("{}")

    def test_metadata_property_format_type_and_value_are_validated(self):
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "property 'Missing'"):
            self.evaluate("{Missing=1}")
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "missing a comparison value"):
            self.evaluate("{Length>}")
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "not numeric"):
            self.evaluate("{Length=large}")
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "not supported for text"):
            self.evaluate("{Organism>alpha}")

        np.testing.assert_array_equal(
            self.evaluate("{length>999}"),
            [False, False, False],
        )

    def test_missing_file_raises_but_existing_empty_file_is_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                with self.assertRaisesRegex(Command_Engine.SelectionContextError, "missing.txt"):
                    self.evaluate("@missing.txt@")

                open(os.path.join(temp_dir, "empty.txt"), "w", encoding="utf-8").close()
                np.testing.assert_array_equal(
                    self.evaluate("@empty.txt@"),
                    [False, False, False],
                )

    def test_invalid_branch_aborts_combined_expression(self):
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "cluster_99"):
            self.evaluate('"node"|#cluster_99#')

    def test_error_report_uses_overlay_summary_and_terminal_details(self):
        viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
        error = Command_Engine.SelectionContextError(
            "Group 'missing' does not exist.\nAvailable groups: alpha"
        )
        output = io.StringIO()
        with redirect_stdout(output):
            Command_Engine.report_selection_error(
                viewer,
                "#missing#",
                error,
                "Selection",
            )

        self.assertEqual(
            viewer.console_text.text,
            "Selection error: Group 'missing' does not exist.",
        )
        terminal_text = output.getvalue()
        self.assertIn("Available groups: alpha", terminal_text)
        self.assertIn("Expression: #missing#", terminal_text)
        self.assertIn("no changes were applied", terminal_text)


class AtomicCommandTests(unittest.TestCase):
    def make_viewer(self):
        return SimpleNamespace(
            n_nodes=1,
            full_headers=["node"],
            cluster_labels=np.array([1]),
            group_labels=[set()],
            alignment=None,
            metadata={
                "Length": {
                    "type": "number",
                    "values": np.array([100.0]),
                }
            },
            current_colors=np.array([[0.1, 0.2, 0.3, 1.0]]),
            current_sizes=np.array([5.0]),
            visible_mask=np.array([True]),
            selected_indices=[],
            console_text=SimpleNamespace(text=""),
            _save_state=mock.Mock(),
            promote_nodes=mock.Mock(),
            update_nodes=mock.Mock(),
            update_selection_visual=mock.Mock(),
            update_edges=mock.Mock(),
        )

    def test_color_does_not_apply_earlier_pair_when_later_pair_is_invalid(self):
        viewer = self.make_viewer()
        original_colors = viewer.current_colors.copy()
        original_sizes = viewer.current_sizes.copy()
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                color_command.run(
                    viewer,
                    ['"node"', "red", "#cluster_99#", "blue"],
                )

        np.testing.assert_array_equal(viewer.current_colors, original_colors)
        np.testing.assert_array_equal(viewer.current_sizes, original_sizes)
        self.assertFalse(hasattr(viewer, "current_shapes"))
        viewer._save_state.assert_not_called()
        viewer.promote_nodes.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_group_does_not_apply_earlier_pair_when_later_pair_is_invalid(self):
        viewer = self.make_viewer()
        group_command.run(
            viewer,
            ['"node"', "first", "#missing#", "second"],
        )

        self.assertEqual(viewer.group_labels, [set()])
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_group_can_define_then_reference_group_atomically(self):
        viewer = self.make_viewer()
        group_command.run(
            viewer,
            ['"node"', "first", "#first#", "second"],
        )

        self.assertEqual(viewer.group_labels, [{"first", "second"}])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()

    def test_group_rejects_reserved_words_and_canonical_generated_labels(self):
        prohibited_names = (
            "noise",
            "reset",
            "remove",
            "delete",
            "list",
            "help",
            "cluster",
            "group",
            "groups",
            "clusters",
            "cluster_1",
            "cluster_27",
            "subcluster_1_1",
            "subcluster_12_34",
            "GROUP",
            "CLUSTER",
            "Cluster_1",
            "Subcluster_1_1",
        )

        for name in prohibited_names:
            with self.subTest(name=name):
                viewer = self.make_viewer()
                with mock.patch("builtins.print"):
                    group_command.run(viewer, ['"node"', name])

                self.assertEqual(viewer.group_labels, [set()])
                self.assertTrue(viewer.console_text.text.startswith("Skipped:"))
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()

    def test_group_allows_noncanonical_generated_label_lookalikes(self):
        allowed_names = (
            "cluster_0",
            "cluster_001",
            "subcluster_0_1",
            "subcluster_001_2",
            "subcluster_1_002",
        )

        for name in allowed_names:
            with self.subTest(name=name):
                viewer = self.make_viewer()
                with mock.patch("builtins.print"):
                    group_command.run(viewer, ['"node"', name])

                self.assertEqual(viewer.group_labels, [{name}])
                viewer._save_state.assert_called_once_with()
                viewer.update_nodes.assert_called_once_with()

    def test_invalid_select_does_not_replace_current_selection(self):
        viewer = self.make_viewer()
        viewer.selected_indices = [0]
        select_command.run(viewer, ["#cluster_99#"])

        self.assertEqual(viewer.selected_indices, [0])
        viewer.update_selection_visual.assert_not_called()

    def test_select_save_uses_configured_header_list_directory(self):
        viewer = self.make_viewer()
        viewer.selected_indices = [0]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                select_command.run(viewer, ["save", "selected.txt"])

            output_path = os.path.join(temp_dir, "selected.txt")
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "node\n")

    def test_export_uses_configured_sequence_export_directory(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = np.array([0])
        viewer._selected_fasta_records = [("node", "CCCC")]

        with tempfile.TemporaryDirectory() as temp_dir:
            fasta_path = os.path.join(temp_dir, "source.fasta")
            with open(fasta_path, "w", encoding="utf-8") as handle:
                handle.write(">node\nAAAA\n")

            export_root = os.path.join(temp_dir, "exports")
            metadata = SimpleNamespace(model_name="test", network_type="blast")
            patches = (
                mock.patch.object(cfg, "NODE_FASTA_FILE", fasta_path),
                mock.patch.object(cfg, "INPUT_HDF5", "network.h5"),
                mock.patch.object(
                    export_command,
                    "SEQUENCE_EXPORT_DIRECTORY",
                    export_root,
                ),
                mock.patch.object(cfg, "TOP_EDGE_PERCENT", None),
                mock.patch.object(cfg, "SIMILARITY_THRESHOLD", 0.5),
                mock.patch.object(
                    export_command.cache_manifest,
                    "validate_network_schema",
                    return_value=metadata,
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                with redirect_stdout(io.StringIO()):
                    export_command.run(viewer, ["clusters"])

            output_path = os.path.join(
                export_root,
                "source_[test]",
                "Score0.5",
                "Cluster_0.fasta",
            )
            self.assertTrue(os.path.isfile(output_path))
            with open(output_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read(), ">node\nCCCC\n")

    def test_invalid_hide_does_not_change_visibility_or_undo_state(self):
        viewer = self.make_viewer()
        hide_command.run(viewer, ["#cluster_99#"])

        np.testing.assert_array_equal(viewer.visible_mask, [True])
        viewer._save_state.assert_not_called()
        viewer.update_edges.assert_not_called()

    def test_invalid_metadata_export_does_not_create_output_file(self):
        viewer = self.make_viewer()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "metadata.csv")
            with mock.patch.object(cfg, "HEADER_LIST_DIR", temp_dir, create=True):
                result = meta_backend.download_metadata(
                    viewer,
                    output_path,
                    expr="{Missing=1}",
                )

            self.assertFalse(result)
            self.assertFalse(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
