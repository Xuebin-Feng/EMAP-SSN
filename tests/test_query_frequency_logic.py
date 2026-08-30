"""Focused tests for grouped amino-acid frequency-query logic."""

import os
import re
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands.query import evaluate_frequency_logic, print_help, run


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
        records = [
            SeqRecord(Seq(residue), id=header)
            for header, residue in zip(headers, ["R", "H", "-", "A"])
        ]
        alignment = SimpleNamespace(
            aln=records,
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


if __name__ == "__main__":
    unittest.main()
