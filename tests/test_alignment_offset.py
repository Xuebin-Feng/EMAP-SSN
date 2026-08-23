import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
import Command_Engine
from utilities.FASTA_Sanitization import simplify_node_label
from utilities.Position_Parsing import (
    format_alignment_offset_display,
    sort_alignment_labels,
)
from commands import label as label_command
from commands import logo as logo_command
from commands import offset as offset_command
from commands import query as query_command


def make_reference_manager():
    manager = Alignment_Manager.Alignment_Manager.__new__(
        Alignment_Manager.Alignment_Manager
    )
    manager.aln = MultipleSeqAlignment(
        [
            SeqRecord(Seq("ACG"), id="ref"),
            SeqRecord(Seq("ATG"), id="other"),
        ]
    )
    manager.has_reference = True
    manager.offset = 0
    manager._base_col_to_label = {0: "1", 1: "1.1", 2: "2"}
    manager.col_to_label = dict(manager._base_col_to_label)
    manager.label_to_col = {
        label: col_idx for col_idx, label in manager.col_to_label.items()
    }
    manager.seq_map = {"ref": 0, "other": 1}
    return manager


class AlignmentOffsetMappingTests(unittest.TestCase):
    def test_alignment_labels_sort_by_major_and_insertion_position(self):
        self.assertEqual(
            sort_alignment_labels(["188.10", "2", "188.6", "-1.2", "label"]),
            ["-1.2", "label", "2", "188.6", "188.10"],
        )

    def test_node_label_simplification_preserves_accession_rules(self):
        self.assertEqual(simplify_node_label("protein WP_012345678.1 detail"), "WP_012345678.1")
        self.assertEqual(simplify_node_label("gi|1|gb|legacy_id|desc"), "legacy_id")
        self.assertEqual(simplify_node_label("plain_id description"), "plain_id")

    def test_constructor_applies_configured_offset_after_reference_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            msa_path = os.path.join(temp_dir, "alignment.fasta")
            with open(msa_path, "w", encoding="utf-8") as handle:
                handle.write(">reference_sequence\nA-C\n>other_sequence\nATC\n")

            manager = Alignment_Manager.Alignment_Manager(
                msa_path,
                active_reference="reference",
                alignment_offset=10,
            )

        self.assertTrue(manager.has_reference)
        self.assertEqual(manager.offset, 10)
        self.assertEqual(
            manager.col_to_label,
            {0: "11", 1: "11.1", 2: "12"},
        )

    def test_constructor_falls_back_to_occupancy_when_reference_resolution_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            msa_path = os.path.join(temp_dir, "alignment.fasta")
            with open(msa_path, "w", encoding="utf-8") as handle:
                handle.write(">reference_sequence\nAC\n>other_sequence\nAT\n")

            manager = Alignment_Manager.Alignment_Manager(
                msa_path,
                active_reference="missing",
                alignment_offset=10,
            )

        self.assertFalse(manager.has_reference)
        self.assertIsNotNone(manager.aln)
        self.assertEqual(len(manager.aln), 2)
        self.assertEqual(manager.offset, 0)
        self.assertTrue(manager.col_to_label)

    def test_offset_shifts_major_position_and_preserves_insertion_suffix(self):
        manager = make_reference_manager()

        self.assertTrue(manager.set_offset(10))

        self.assertEqual(manager.offset, 10)
        self.assertEqual(
            manager.col_to_label,
            {0: "11", 1: "11.1", 2: "12"},
        )
        self.assertEqual(
            manager.label_to_col,
            {"11": 0, "11.1": 1, "12": 2},
        )

    def test_repeated_changes_are_based_on_original_reference_numbering(self):
        manager = make_reference_manager()

        manager.set_offset(10)
        manager.set_offset(-2)

        self.assertEqual(
            manager.col_to_label,
            {0: "-1", 1: "-1.1", 2: "0"},
        )

    def test_offset_is_rejected_without_a_resolved_reference(self):
        manager = make_reference_manager()
        manager.has_reference = False

        self.assertFalse(manager.set_offset(10))
        self.assertEqual(manager.col_to_label[0], "1")

    def test_shifted_mapping_drives_query_lookup_and_label_statistics(self):
        manager = make_reference_manager()
        manager.set_offset(10)

        query_mask = Command_Engine.evaluate_aa_mask(
            ["ref", "other"],
            manager,
            "A",
            "11",
            np.array([0, 1]),
            np.array([0, 1]),
        )
        old_position_mask = Command_Engine.evaluate_aa_mask(
            ["ref", "other"],
            manager,
            "A",
            "1",
            np.array([0, 1]),
            np.array([0, 1]),
        )
        label_stats = manager.calculate_frequencies(manager.col_to_label)

        np.testing.assert_array_equal(query_mask, np.array([True, True]))
        np.testing.assert_array_equal(old_position_mask, np.array([False, False]))
        self.assertEqual(set(label_stats), {"11", "11.1", "12"})

    def test_shifted_mapping_drives_logo_position_resolution(self):
        manager = make_reference_manager()
        manager.set_offset(10)

        columns, positions, missing = logo_command.resolve_reference_columns(
            manager,
            [1, 11, "11.1", 12],
            "ACG",
        )

        self.assertEqual(columns, [0, 1, 2])
        self.assertEqual(positions, [11, "11.1", 12])
        self.assertEqual(missing, [1])

    def test_query_output_reports_current_offset(self):
        manager = make_reference_manager()
        manager.set_offset(10)
        manager.aln = list(manager.aln)
        viewer = SimpleNamespace(
            alignment=manager,
            alignment_offset=10,
            full_headers=["ref", "other"],
            selected_indices=[],
            cluster_labels=None,
            group_labels=None,
            metadata=None,
            console_text=SimpleNamespace(text=""),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            query_command.run(viewer, ["[11]"])

        self.assertIn("Alignment Offset: 10", output.getvalue())
        self.assertIn("Pos 11", output.getvalue())

    def test_inactive_offset_display_is_explicit(self):
        manager = make_reference_manager()
        manager.has_reference = False
        viewer = SimpleNamespace(alignment=manager, alignment_offset=10)

        self.assertEqual(
            format_alignment_offset_display(
                viewer.alignment, viewer.alignment_offset
            ),
            "10 (inactive)",
        )

    def test_label_workbook_metadata_includes_offset(self):
        class Worksheet:
            def __init__(self):
                self.rows = []

            def append(self, row):
                self.rows.append(row)

        worksheet = Worksheet()
        label_command._append_workbook_metadata(
            worksheet,
            "labels.xlsx",
            "reference_sequence",
            "10",
            ["A11"],
        )

        self.assertEqual(worksheet.rows[2], ["Alignment Offset: 10"])
        self.assertIn(["Global Conserved (>97%)"], worksheet.rows)


class OffsetCommandTests(unittest.TestCase):
    def setUp(self):
        self.manager = make_reference_manager()
        self.viewer = mock.Mock()
        self.viewer.alignment = self.manager
        self.viewer.alignment_offset = 0

    def test_command_without_argument_reports_current_offset(self):
        self.manager.set_offset(7)

        with mock.patch.object(offset_command.Command_Engine, "print_help") as output:
            offset_command.run(self.viewer, [])

        output.assert_called_once_with(
            self.viewer,
            "Current Alignment Offset: 7",
        )

    def test_help_describes_numbering_rules_and_affected_commands(self):
        output = io.StringIO()

        with redirect_stdout(output):
            offset_command.print_help()

        help_text = output.getvalue()
        self.assertIn("Alignment Numbering Offset Tool", help_text)
        self.assertIn("displayed position = reference position + offset", help_text)
        self.assertIn("insertion", help_text)
        self.assertIn("query, label, logo", help_text)
        self.assertIn("K(-1)", help_text)
        self.assertIn("never K-1", help_text)
        self.assertIn("offset 0", help_text)

    def test_command_sets_offset_and_updates_runtime_state(self):
        old_cfg_offset = getattr(offset_command.cfg, "ALIGNMENT_OFFSET", 0)
        try:
            with mock.patch.object(offset_command.Command_Engine, "print_help") as output:
                offset_command.run(self.viewer, ["10"])

            self.assertEqual(self.manager.offset, 10)
            self.assertEqual(self.viewer.alignment_offset, 10)
            self.assertEqual(offset_command.cfg.ALIGNMENT_OFFSET, 10)
            self.assertEqual(self.manager.label_to_col["11.1"], 1)
            output.assert_called_once_with(
                self.viewer,
                "Alignment Offset set to 10. Position numbering updated.",
            )
        finally:
            offset_command.cfg.ALIGNMENT_OFFSET = old_cfg_offset

    def test_command_rejects_non_integer(self):
        with mock.patch.object(offset_command.Command_Engine, "print_help") as output:
            offset_command.run(self.viewer, ["1.5"])

        self.assertEqual(self.manager.offset, 0)
        self.assertIn(
            "must be an integer",
            output.call_args.args[1],
        )

    def test_command_cannot_set_offset_without_valid_reference(self):
        self.manager.has_reference = False

        with mock.patch.object(offset_command.Command_Engine, "print_help") as output:
            offset_command.run(self.viewer, ["10"])

        self.assertEqual(self.manager.offset, 0)
        self.assertIn(
            "requires a correctly loaded reference",
            output.call_args.args[1],
        )


if __name__ == "__main__":
    unittest.main()
