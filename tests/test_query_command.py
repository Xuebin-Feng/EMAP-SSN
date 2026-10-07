"""The `query` command (commands/query.py).

Covers frequency-logic expressions (single, grouped and compound),
position parsing, help text, and the per-position breakdown `query` prints.
"""

import os
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Alignment_Manager
from commands.query import evaluate_frequency_logic, parse_query_positions, print_help, run
from tests.sparse_alignment import load_manager, sparse_alignment, write_fasta


class QueryFrequencyLogicTests(unittest.TestCase):
    def setUp(self):
        self.gaps = np.array([0.25, 0.50, 0.00])
        self.amino_acids = {
            "R": np.array([0.25, 0.25, 0.00]),
            "H": np.array([0.25, 0.00, 0.25]),
            "K": np.array([0.25, 0.00, 0.25]),
            "D": np.array([0.00, 0.25, 0.25]),
            "E": np.array([0.00, 0.25, 0.00]),
        }

    def evaluate(self, expression):
        return evaluate_frequency_logic(expression, self.gaps, self.amino_acids)

    def test_single_group_sums_unique_residue_frequencies(self):
        np.testing.assert_array_equal(
            self.evaluate("(RHK)>50%"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("(RRHK)>=0.5"),
            [True, False, True],
        )

    def test_comparison_boundaries_and_numeric_percent_convention(self):
        np.testing.assert_array_equal(
            self.evaluate("(RHK)>=50%"),
            [True, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("(RHK)>=50"),
            [True, False, True],
        )

    def test_compound_grouped_and_mixed_conditions_require_outer_parentheses(self):
        np.testing.assert_array_equal(
            self.evaluate("((RHK)>=50%)&((DE)>20%)"),
            [False, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("((RHK)>=50%)&(GAP<30%)"),
            [True, False, True],
        )

        for expression in (
            "(RHK)>50%&(DE)>20%",
            "((RHK)>50%)&(DE)>20%",
        ):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(
                    ValueError,
                    re.escape("must be enclosed in parentheses"),
                ):
                    self.evaluate(expression)

    def test_existing_single_residue_gap_and_boolean_syntax_remain_valid(self):
        np.testing.assert_array_equal(
            self.evaluate("K>20%"),
            [True, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("(K>20%)|(R>20%)"),
            [True, True, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("(GAP>=50%)"),
            [False, True, False],
        )

    def test_boolean_precedence_negation_and_complete_consumption(self):
        np.testing.assert_array_equal(
            self.evaluate("(K>20%)|(R>20%)&(GAP>=50%)"),
            [True, True, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("!(K>20%)"),
            [False, True, False],
        )
        with self.assertRaisesRegex(ValueError, "Invalid frequency Boolean expression"):
            self.evaluate("(K>20%) trailing")

    def test_absent_residues_and_malformed_groups(self):
        np.testing.assert_array_equal(
            self.evaluate("(WY)>0"),
            [False, False, False],
        )
        with self.assertRaisesRegex(
            ValueError,
            "at least two one-letter residue symbols",
        ):
            self.evaluate("(R)>10%")

    def test_help_documents_grouped_single_and_compound_syntax(self):
        output = StringIO()
        with redirect_stdout(output):
            print_help()
        help_text = output.getvalue()
        self.assertIn("[(RHK)>50%]", help_text)
        self.assertIn("[((RHK)>50%) & ((DE)>20%)]", help_text)
        self.assertIn("(RHK)(-1)", help_text)

    def test_command_uses_all_mapped_sequences_as_grouped_frequency_denominator(self):
        headers = ["arginine", "histidine", "gap", "alanine"]
        alignment = SimpleNamespace(
            aln=sparse_alignment(zip(headers, ["R", "H", "-", "A"])),
            label_to_col={"1": 0},
            col_to_label={0: "1"},
            seq_map={header: index for index, header in enumerate(headers)},
            has_reference=False,
            offset=0,
            msa_file="grouped-frequency.fasta",
        )
        viewer = SimpleNamespace(
            alignment=alignment,
            alignment_offset=0,
            full_headers=headers,
            selected_indices=[],
            cluster_labels=None,
            group_labels=None,
            metadata=None,
            console_text=SimpleNamespace(text=""),
        )

        strict_output = StringIO()
        with redirect_stdout(strict_output):
            run(viewer, ["[(RH)>50%]"])
        self.assertIn("Matching Positions (0 found)", strict_output.getvalue())

        inclusive_output = StringIO()
        with redirect_stdout(inclusive_output):
            run(viewer, ["[(RH)>=50%]"])
        self.assertIn("Matching Positions (1 found)", inclusive_output.getvalue())
        self.assertIn("Pos 1", inclusive_output.getvalue())


class QueryPositionBreakdownTests(unittest.TestCase):
    """The per-position residue breakdown `query` prints for a node subset."""

    def run_query(self, records, headers, args, cluster_labels=None):
        """Run `query` with ARGS on an MSA loaded as the Viewer loads it."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        msa_path = os.path.join(directory.name, "breakdown.fasta")
        write_fasta(msa_path, records)
        with mock.patch.object(Alignment_Manager.cfg, "FILTER_MIN_OCCUPANCY", 50):
            alignment = load_manager(msa_path, headers)
        viewer = SimpleNamespace(
            alignment=alignment,
            alignment_offset=0,
            full_headers=list(headers),
            selected_indices=[],
            cluster_labels=None if cluster_labels is None else np.asarray(cluster_labels),
            group_labels=None,
            metadata=None,
            console_text=SimpleNamespace(text=""),
        )
        output = StringIO()
        with redirect_stdout(output):
            run(viewer, args)
        return viewer, output.getvalue()

    @staticmethod
    def position_lines(output):
        return [line for line in output.splitlines() if line.startswith("Pos ")]

    def test_subset_breakdown_uses_the_subsets_aligned_rows(self):
        # The A row is in cluster 2; "unaligned" is in cluster 1 but the MSA
        # lacks it, so cluster 1 maps to the R, H and gap rows.
        headers = ["arginine", "histidine", "gap", "alanine", "unaligned"]
        records = list(zip(headers[:4], ["R", "H", "-", "A"]))

        viewer, output = self.run_query(
            records, headers, ["#cluster_1#", "[1,7]"], cluster_labels=[1, 1, 1, 2, 1]
        )

        self.assertIn("QUERY SUBSET: '#cluster_1#' (3 sequences mapped)", output)
        self.assertEqual(
            self.position_lines(output),
            [
                "Pos 1       \tGap  33.3% | R  33.3% | H  33.3%",
                "Pos     7: [Not found in active alignment mapping]",
            ],
        )
        self.assertEqual(viewer.console_text.text, "Queried 1 position(s). Check terminal.")

    def test_residues_under_one_percent_are_omitted(self):
        headers = [f"s{index}" for index in range(101)]
        records = [(header, "A") for header in headers[:100]] + [(headers[100], "C")]

        _, output = self.run_query(records, headers, ['"*"', "[1]"])

        # C is 1 of 101 sequences (0.99%).
        self.assertEqual(self.position_lines(output), ["Pos 1       \tGap   0.0% | A  99.0%"])

    def test_expression_without_aligned_matches_aborts(self):
        headers = ["arginine", "histidine"]

        viewer, output = self.run_query(
            list(zip(headers, ["R", "H"])), headers, ['"zzz"', "[1]"]
        )

        message = "No sequences matched the expression '\"zzz\"'. Aborting query."
        self.assertIn(message, output)
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(self.position_lines(output), [])


VALID_LABELS = [
    ((-3, 0), "-3"),
    ((-2, 0), "-2"),
    ((-1, 0), "-1"),
    ((-1, 1), "-1.1"),
    ((0, 0), "0"),
    ((1, 0), "1"),
    ((1, 1), "1.1"),
    ((2, 0), "2"),
]


class QueryPositionParsingTests(unittest.TestCase):
    def test_parenthesized_negative_positions_are_normalized(self):
        self.assertEqual(
            parse_query_positions("[(-1),(-1.1),0]", VALID_LABELS),
            ["-1", "-1.1", "0"],
        )

    def test_negative_ranges_expand_over_mapped_insertion_labels(self):
        self.assertEqual(
            parse_query_positions("[(-3)-(-1)]", VALID_LABELS),
            ["-3", "-2", "-1"],
        )
        self.assertEqual(
            parse_query_positions("[(-2)-1]", VALID_LABELS),
            ["-2", "-1", "-1.1", "0", "1"],
        )

    def test_descending_ranges_and_end_alias_remain_supported(self):
        self.assertEqual(
            parse_query_positions("[(-1)-(-3)]", VALID_LABELS),
            ["-3", "-2", "-1"],
        )
        self.assertEqual(
            parse_query_positions("[END-1]", VALID_LABELS),
            ["1", "1.1", "2"],
        )
        self.assertEqual(
            parse_query_positions("[E,0]", VALID_LABELS),
            ["2", "0"],
        )

    def test_bare_negative_positions_are_rejected_with_correction(self):
        for position_spec, correction in (
            ("[-1]", "(-1)"),
            ("[-1.1,0]", "(-1.1)"),
            ("[-3--1]", "(-3)"),
            ("[1--1]", "(-1)"),
        ):
            with self.subTest(position_spec=position_spec):
                with self.assertRaisesRegex(ValueError, re.escape(correction)):
                    parse_query_positions(position_spec, VALID_LABELS)


if __name__ == "__main__":
    unittest.main()
