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
import Command_Engine
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

    def test_grouped_comparisons_match_single_residues_at_exact_boundaries(self):
        # 10 sequences: A is 1 (0.1) and C is 2 (0.2), so (AC) is exactly 30%,
        # as a single residue with 3 of 10 is. Adding the fractions gives
        # 0.30000000000000004.
        gaps = np.array([0.7])
        fractions = {"A": np.array([0.1]), "C": np.array([0.2]), "T": np.array([0.3])}

        def grouped(expression):
            return evaluate_frequency_logic(expression, gaps, fractions, 10)[0]

        self.assertTrue(grouped("(AC)<=30%"))
        self.assertFalse(grouped("(AC)>30%"))
        self.assertTrue(grouped("(AC)>=30%"))
        self.assertFalse(grouped("(AC)<30%"))
        self.assertTrue(grouped("(AC)>=0.3"))
        self.assertFalse(grouped("(AC)>30"))
        # The single residue with the same count agrees on every comparison.
        for operator in ("<=", ">", ">=", "<"):
            with self.subTest(operator=operator):
                self.assertEqual(
                    grouped(f"(AC){operator}30%"),
                    grouped(f"T{operator}30%"),
                )

    def test_grouped_boundaries_are_exact_for_every_split_of_a_count(self):
        # Every a + b of n sequences: the group agrees with one residue of a + b.
        for n in (3, 7, 10, 30, 101):
            for total in range(n + 1):
                for first in range(total + 1):
                    gaps = np.array([(n - total) / n])
                    fractions = {
                        "A": np.array([first / n]),
                        "C": np.array([(total - first) / n]),
                        "T": np.array([total / n]),
                    }
                    for percent in (10, 20, 30, 50, 70):
                        for operator in ("<=", ">", ">=", "<"):
                            expression = f"{operator}{percent}%"
                            grouped = evaluate_frequency_logic(
                                f"(AC){expression}", gaps, fractions, n
                            )
                            single = evaluate_frequency_logic(
                                f"T{expression}", gaps, fractions, n
                            )
                            self.assertEqual(
                                grouped.tolist(), single.tolist(),
                                (n, total, first, expression),
                            )

    def test_grouped_sums_without_a_sequence_count_are_unchanged(self):
        np.testing.assert_array_equal(
            self.evaluate("(RHK)>=50%"),
            [True, False, True],
        )

    def test_unparenthesized_multi_letter_targets_are_rejected(self):
        for expression in ("KR>20%", "LYS>10%", "(KR>20%)|(R>20%)", "GAPS<10%", "K_>5%"):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(ValueError, "Unknown frequency target"):
                    self.evaluate(expression)

    def test_valid_single_letter_gap_and_group_targets_remain_accepted(self):
        for expression in ("K>20%", "k>20%", "_>=50%", "GAP>=50%", "gap>=50%", "(rhk)>50%", "W>10%"):
            with self.subTest(expression=expression):
                self.evaluate(expression)

    def test_help_documents_threshold_numbers_hidden_nodes_and_group_syntax(self):
        output = StringIO()
        with redirect_stdout(output):
            print_help()
        text = " ".join(output.getvalue().split())
        self.assertIn("1 means 100%", text)
        self.assertIn("5 means 5%", text)
        self.assertIn("Hidden nodes are included", text)
        self.assertIn("write several residues as a group, (KR)", text)


class QueryPositionBreakdownTests(unittest.TestCase):
    """The per-position residue breakdown `query` prints for a node subset."""

    def run_query(self, records, headers, args, cluster_labels=None, metadata=None, console=True):
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
            metadata=metadata,
            console_text=SimpleNamespace(text=""),
        )
        if not console:
            del viewer.console_text
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
        self.assertEqual(viewer.console_text.text, "Queried 1 position. Check terminal.")

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

    def run_reporting(self, *arguments, **options):
        """run_query with the command's outcome: (viewer, output, failed, succeeded)."""
        with mock.patch.object(Command_Engine, "command_failed") as failed, \
                mock.patch.object(Command_Engine, "command_succeeded") as succeeded:
            viewer, output = self.run_query(*arguments, **options)
        return viewer, output, failed, succeeded

    def test_expression_tokens_are_joined_with_spaces(self):
        headers = ["Escherichia_a", "Escherichia_b", "Bacillus_c"]
        metadata = {
            "Organism": {
                "type": "text",
                "values": ["Escherichia coli", "Escherichia coli", "Bacillus subtilis"],
            }
        }

        _, output = self.run_query(
            list(zip(headers, ["A", "A", "C"])),
            headers,
            ["{Organism=Escherichia", "coli}", "[1]"],
            metadata=metadata,
        )

        # "Escherichiacoli" matches no node, so the old join queried nothing.
        self.assertIn("(2 sequences mapped)", output)
        self.assertEqual(
            self.position_lines(output), ["Pos 1       \tGap   0.0% | A 100.0%"]
        )

    def test_operators_between_separate_tokens_parse_as_before(self):
        headers = ["arginine", "histidine", "gap", "alanine"]
        records = list(zip(headers, ["R", "H", "-", "A"]))

        for tokens in (
            ["#cluster_1#", "|", "#cluster_2#"],
            ["#cluster_1#", "|", "#cluster_2#", "&", "!", "#cluster_3#"],
            ["(#cluster_1#", "|", "#cluster_2#)"],
        ):
            with self.subTest(tokens=tokens):
                _, output = self.run_query(
                    records, headers, tokens + ["[1]"], cluster_labels=[1, 1, 2, 3]
                )
                self.assertIn("(3 sequences mapped)", output)

    def test_single_positions_are_spelled_as_the_alignment_labels_them(self):
        headers = ["arginine", "histidine"]
        records = list(zip(headers, ["R", "H"]))

        for position in ("01", "+1", "001"):
            with self.subTest(position=position):
                _, output = self.run_query(records, headers, ['"*"', f"[{position}]"])
                self.assertEqual(
                    self.position_lines(output),
                    ["Pos 1       \tGap   0.0% | R  50.0% | H  50.0%"],
                )

    def test_nothing_queried_is_a_failure_not_a_success(self):
        headers = ["arginine", "histidine"]
        records = list(zip(headers, ["R", "H"]))

        for position_list in ("[99]", "[]", "[,]", "[99-100]"):
            with self.subTest(position_list=position_list):
                viewer, output, failed, succeeded = self.run_reporting(
                    records, headers, ['"*"', position_list]
                )
                failed.assert_called_once()
                self.assertEqual(str(failed.call_args.args[1]), "No valid positions queried.")
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, "No valid positions queried.")

    def test_a_position_that_is_found_still_succeeds(self):
        headers = ["arginine", "histidine"]
        records = list(zip(headers, ["R", "H"]))

        _, _, failed, succeeded = self.run_reporting(records, headers, ['"*"', "[1, 99]"])

        failed.assert_not_called()
        succeeded.assert_called_once()
        self.assertEqual(str(succeeded.call_args.args[1]), "Queried 1 position. Check terminal.")

    def test_matched_nodes_that_are_not_aligned_say_so(self):
        # Node 2 is the only member of cluster 2 and the MSA lacks it.
        headers = ["arginine", "histidine", "unaligned"]
        records = list(zip(headers[:2], ["R", "H"]))

        viewer, output, _, _ = self.run_reporting(
            records, headers, ["#cluster_2#", "[1]"], cluster_labels=[1, 1, 2]
        )

        message = (
            "The expression '#cluster_2#' matched 1 node, but none is in the "
            "alignment. Aborting query."
        )
        self.assertIn(message, output)
        self.assertEqual(viewer.console_text.text, message)
        self.assertNotIn("No sequences matched", output)

    def test_grouped_frequency_search_uses_exact_counts_at_the_boundary(self):
        # Ten sequences: one A, two C and seven G, so (AC) is exactly 30%.
        headers = [f"s{index}" for index in range(10)]
        records = list(zip(headers, ["A", "C", "C"] + ["G"] * 7))

        _, at_most = self.run_query(records, headers, ['"*"', "[(AC)<=30%]"])
        _, more_than = self.run_query(records, headers, ['"*"', "[(AC)>30%]"])

        self.assertIn("Matching Positions (1 found)", at_most)
        self.assertIn("Matching Positions (0 found)", more_than)

    def test_unparenthesized_multi_letter_target_is_an_error_not_zero_matches(self):
        headers = ["arginine", "histidine"]
        records = list(zip(headers, ["R", "H"]))

        for logic in ("[KR>20%]", "[LYS>10%]"):
            with self.subTest(logic=logic):
                viewer, output, failed, _ = self.run_reporting(records, headers, ['"*"', logic])
                failed.assert_called_once()
                self.assertIn("Unknown frequency target", str(failed.call_args.args[1]))
                self.assertIn("Unknown frequency target", viewer.console_text.text)
                self.assertNotIn("Matching Positions", output)

    def test_missing_parentheses_error_is_a_failure_without_a_console_line(self):
        headers = ["arginine", "histidine"]
        records = list(zip(headers, ["R", "H"]))

        for console in (True, False):
            with self.subTest(console=console):
                viewer, output, failed, succeeded = self.run_reporting(
                    records, headers, ['"*"', "[K>10% & R>5%]"], console=console
                )
                failed.assert_called_once()
                self.assertEqual(
                    str(failed.call_args.args[1]),
                    "Error: Individual frequency arguments must be enclosed in ()",
                )
                succeeded.assert_not_called()
                self.assertIn("must be enclosed in parentheses", output)


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

    def test_single_positions_are_canonicalized_as_logo_does(self):
        from commands.logo import parse_logo_positions

        for position_spec, expected in (
            ("[01]", ["1"]),
            ("[+1]", ["1"]),
            ("[(-0)]", ["0"]),
            ("[(-01)]", ["-1"]),
            ("[001.01]", ["1.1"]),
            ("[(-1.01)]", ["-1.1"]),
            ("[(-0.1)]", ["0.1"]),
            ("[1, 01, +1]", ["1"]),
            ("[(-1),(-01)]", ["-1"]),
        ):
            with self.subTest(position_spec=position_spec):
                self.assertEqual(
                    parse_query_positions(position_spec, VALID_LABELS), expected
                )
                # logo spells each of them the same way.
                self.assertEqual(
                    [str(position) for position in parse_logo_positions(position_spec)],
                    expected,
                )

    def test_labels_already_spelled_canonically_are_unchanged(self):
        self.assertEqual(
            parse_query_positions("[2,1.1,(-1.1),0]", VALID_LABELS),
            ["2", "1.1", "-1.1", "0"],
        )
        # An insertion of 0 names no label; it is reported as not found, as before.
        self.assertEqual(parse_query_positions("[1.0]", VALID_LABELS), ["1.0"])
        self.assertEqual(parse_query_positions("[E]", []), ["E"])
        self.assertEqual(parse_query_positions("[END, 01]", VALID_LABELS), ["2", "1"])

    def test_reversed_ranges_are_still_read_in_either_order(self):
        self.assertEqual(
            parse_query_positions("[(-01)-(-3)]", VALID_LABELS),
            ["-3", "-2", "-1"],
        )


if __name__ == "__main__":
    unittest.main()
