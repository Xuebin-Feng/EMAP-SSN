"""Boolean selection expressions (Command_Engine): syntax, context validation and
what each predicate matches."""

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
import EMAPSSN_Config as cfg


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
                    (0, "R"): [True, False, False],
                    (0, "H"): [False, True, False],
                    (0, "K"): [False, False, True],
                    (1, "R"): [True, False, False],
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

    def test_classifier_distinguishes_valid_plain_and_malformed_arguments(self):
        valid = (
            '"node"',
            "@missing.txt@",
            "#missing#",
            "{Missing>1}",
            "P106",
            "(RHK)71",
            "$sele$",
            "(#alpha#|#omega#)&!A10",
        )
        for text in valid:
            with self.subTest(text=text):
                result = Command_Engine.classify_selection_expression(text)
                self.assertEqual(
                    result.kind,
                    Command_Engine.SelectionClassificationKind.VALID_EXPRESSION,
                )
                self.assertIsNotNone(result.expression)

        for text in ("virids", "target_logo.svg", "active_site"):
            with self.subTest(text=text):
                result = Command_Engine.classify_selection_expression(text)
                self.assertEqual(
                    result.kind,
                    Command_Engine.SelectionClassificationKind.NOT_EXPRESSION,
                )

        for text in ("#missing", "{Length}", "P106&", "K-1", "(RHK)-1"):
            with self.subTest(text=text):
                result = Command_Engine.classify_selection_expression(text)
                self.assertEqual(
                    result.kind,
                    Command_Engine.SelectionClassificationKind.MALFORMED_EXPRESSION,
                )
                self.assertIsInstance(result.error, Command_Engine.SelectionExpressionError)

    def test_native_selection_atom_and_operator_precedence(self):
        np.testing.assert_array_equal(
            self.evaluate("$sele$", selection_mask=np.array([False, True, False])),
            [False, True, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("#alpha#|#omega#&#cluster_2#"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("(#alpha#|#omega#)^#cluster_2#"),
            [True, True, True],
        )

    def test_canonical_cluster_spelling_and_ambiguity(self):
        np.testing.assert_array_equal(
            self.evaluate(
                "#cluster_0#",
                cluster_labels=np.array([0, 1, -1]),
                group_labels=[set(), set(), set()],
            ),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate(
                "#cluster_001#",
                cluster_labels=np.array([1, 2, -1]),
                group_labels=[{"cluster_001"}, set(), set()],
            ),
            [True, False, False],
        )
        with self.assertRaisesRegex(Command_Engine.SelectionContextError, "ambiguous"):
            self.evaluate(
                "#cluster_0#",
                cluster_labels=np.array([0, 1, -1]),
                group_labels=[{"cluster_0"}, set(), set()],
            )

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

    def test_grouped_amino_acids_match_explicit_or_expression(self):
        expected = self.evaluate("R10|H10|K10")
        np.testing.assert_array_equal(self.evaluate("(RHK)10"), expected)
        np.testing.assert_array_equal(self.evaluate("(rrhk)10"), expected)
        np.testing.assert_array_equal(
            self.evaluate("(RHK)10&!A0"),
            [False, False, True],
        )

    def test_grouped_amino_acids_support_insertion_and_negative_positions(self):
        np.testing.assert_array_equal(
            self.evaluate("(RHK)10.1"),
            [True, False, False],
        )
        np.testing.assert_array_equal(
            self.evaluate("(RHK)(-1)"),
            [True, False, True],
        )
        np.testing.assert_array_equal(
            self.evaluate("(RHK)(-1.1)"),
            [False, True, False],
        )

    def test_grouped_amino_acids_deduplicate_and_validate_targets(self):
        np.testing.assert_array_equal(
            self.evaluate("(RRHK)10"),
            self.evaluate("(RHK)10"),
        )
        with self.assertRaisesRegex(
            Command_Engine.SelectionExpressionError,
            "at least two one-letter residue symbols",
        ):
            self.evaluate("(R)10")
        with self.assertRaisesRegex(
            Command_Engine.SelectionExpressionError,
            re.escape("(RHK)(-1)"),
        ):
            self.evaluate("(RHK)-1")

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

    def gravy(self, expression):
        metadata = dict(self.metadata)
        metadata["GRAVY"] = {"type": "number", "values": np.array([-1.2, -0.4, 0.3])}
        return self.evaluate(expression, metadata=metadata)

    def test_metadata_ranges_take_parenthesised_negative_bounds(self):
        """GRAVY is usually negative, so its ranges need negative bounds."""
        np.testing.assert_array_equal(self.gravy("{GRAVY=(-1)-0}"), [False, True, False])
        np.testing.assert_array_equal(self.gravy("{GRAVY=(-1.5)-(-0.5)}"), [True, False, False])
        np.testing.assert_array_equal(self.gravy("{GRAVY==(-0.5)-1}"), [False, True, True])
        np.testing.assert_array_equal(self.gravy("{GRAVY=(-1)-0}|#cluster_2#"), [False, True, False])
        np.testing.assert_array_equal(self.gravy("{Length=150-300}"), [False, True, True])
        # Single values never needed parentheses, and may have them.
        np.testing.assert_array_equal(self.gravy("{GRAVY>=-1}"), [False, True, True])
        np.testing.assert_array_equal(self.gravy("{GRAVY>=(-1)}"), [False, True, True])

    def test_an_unparenthesised_negative_bound_is_explained(self):
        for expression in ("{GRAVY=-1-0}", "{GRAVY=-1.5--0.5}", "{GRAVY=0--1}"):
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(Command_Engine.SelectionExpressionError,
                                            r"must be written in parentheses.*\(-1\)-0"):
                    self.gravy(expression)
        with self.assertRaisesRegex(Command_Engine.SelectionExpressionError, "not numeric"):
            self.gravy("{GRAVY>(-1)-0}")

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


class MetadataPredicateTests(unittest.TestCase):
    """Which nodes a {PROPERTY OPERATOR VALUE} predicate matches."""

    def evaluate(self, expression):
        metadata = {
            "Organism": {
                "type": "text",
                "values": np.array(["Escherichia coli", "Bacillus", "coliform"], dtype=object),
            },
            "Length": {"type": "number", "values": np.array([100.0, 200.0, 300.0])},
            # NaN where a sequence has no hydropathy-scored residue.
            "GRAVY": {"type": "number", "values": np.array([-0.5, np.nan, 0.25])},
        }
        return Command_Engine.parse_advanced_expression(
            expression,
            np.full(3, -1, dtype=int),
            np.array([], dtype=int),
            ["node_a", "node_b", "node_c"],
            metadata=metadata,
        )

    def assert_matches(self, cases):
        for expression, expected in cases:
            with self.subTest(expression=expression):
                np.testing.assert_array_equal(self.evaluate(expression), expected)

    def test_text_equality_is_a_case_insensitive_substring_match(self):
        self.assert_matches((
            ("{Organism=coli}", [True, False, True]),
            ("{Organism==coli}", [True, False, True]),
            ("{Organism=COLI}", [True, False, True]),
            ("{organism=Bacil}", [False, True, False]),
        ))

    def test_wildcards_make_text_equality_a_whole_value_pattern(self):
        self.assert_matches((
            ("{Organism=*coli}", [True, False, False]),
            ("{Organism=coli*}", [False, False, True]),
            ("{Organism=*COLI*}", [True, False, True]),
            ("{Organism=?acillus}", [False, True, False]),
            # A substring match would find "bacill" in "bacillus".
            ("{Organism=bacill?}", [False, False, False]),
        ))

    def test_not_equal_negates_the_text_match(self):
        self.assert_matches((
            ("{Organism!=coli}", [False, True, False]),
            ("{Organism!=*coli}", [False, True, True]),
        ))

    def test_quotes_around_a_text_value_are_ignored(self):
        self.assert_matches((
            ('{Organism="coli"}', [True, False, True]),
            ("{Organism='coli'}", [True, False, True]),
            ('{Organism!="coli"}', [False, True, False]),
            # Quotes delimit wildcard patterns too: the quoted pattern matched
            # nothing, and its != form selected every node.
            ('{Organism="*coli"}', [True, False, False]),
            ("{Organism='coli*'}", [False, False, True]),
            ('{Organism!="*coli"}', [False, True, True]),
        ))

    def test_numeric_comparisons(self):
        self.assert_matches((
            ("{Length=200}", [False, True, False]),
            ("{Length==200.0}", [False, True, False]),
            ("{Length!=200}", [True, False, True]),
            ("{Length<=200}", [True, True, False]),
            ("{Length<200}", [True, False, False]),
            ("{Length>=200}", [False, True, True]),
            ("{Length>200}", [False, False, True]),
        ))

    def test_numeric_comparisons_skip_missing_values(self):
        # != skips a missing value like every other comparison (it used to
        # select it); the ! operator still selects the complement.
        self.assert_matches((
            ("{GRAVY!=-0.5}", [False, False, True]),
            ("{GRAVY=-0.5}", [True, False, False]),
            ("{GRAVY>=-0.5}", [True, False, True]),
            ("{GRAVY<1}", [True, False, True]),
            ("{GRAVY=(-1)-1}", [True, False, True]),
            ("!{GRAVY=-0.5}", [False, True, True]),
        ))


class HeaderListFileTests(unittest.TestCase):
    """Which nodes an @file@ header list in cfg.HEADER_LIST_DIR selects."""

    HEADERS = ["Alpha", "Alpha_2", "beta_7", "MKVL"]

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.header_dir = temp_dir.name
        patcher = mock.patch.object(
            cfg, "HEADER_LIST_DIR", self.header_dir, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_list(self, name, text):
        path = os.path.join(self.header_dir, name)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def test_plain_list_matches_whole_headers_ignoring_case_and_outer_spaces(self):
        self.write_list("picked.txt", "ALPHA\n  beta_7  \n\nAlp\n")
        # A name without .txt or .fasta reads <name>.txt; "Alp" is no
        # substring match and "Alpha" does not select "Alpha_2".
        np.testing.assert_array_equal(
            Command_Engine.evaluate_file_mask(self.HEADERS, "picked"),
            [True, False, True, False],
        )

    def test_fasta_list_reads_only_header_lines(self):
        self.write_list("picked.fasta", ">beta_7\nMKVL\n>alpha\nMKVLAAA\n")
        np.testing.assert_array_equal(
            Command_Engine.evaluate_file_mask(self.HEADERS, "picked.fasta"),
            [True, False, True, False],
        )

    def test_ncbi_list_matches_the_accessions_in_headers(self):
        headers = [
            "WP_012345678.1",
            "XP_000111222.3|desc",
            "AAB12345.2|x",
            "WP_999999999.1",
        ]
        self.write_list(
            "accessions.txt",
            "wp_012345678.1\nXP_000111222.3 some description\nAAB12345.2\nnot listed\n",
        )
        np.testing.assert_array_equal(
            Command_Engine.evaluate_file_mask(headers, "[NCBI]accessions.txt"),
            [True, True, True, False],
        )

    def test_pdb_list_matches_the_ids_in_headers(self):
        headers = ["1ABC_A", "pdb|2XYZ|B", "3DEF_C", "9QQQ"]
        self.write_list("structures.txt", "1abc\n2XYZ_B\n9QQQ chain\n")
        np.testing.assert_array_equal(
            Command_Engine.evaluate_file_mask(headers, "[PDB]structures"),
            [True, True, False, True],
        )

    def test_expression_atoms_read_the_list_with_its_prefix(self):
        self.write_list("structures.txt", "3DEF\n")
        self.write_list("picked.txt", "MKVL\n")
        headers = ["1ABC_A", "3DEF_C", "MKVL"]
        np.testing.assert_array_equal(
            Command_Engine.parse_advanced_expression(
                "@[PDB]structures@|@picked@",
                np.full(3, -1, dtype=int),
                np.array([], dtype=int),
                headers,
            ),
            [False, True, True],
        )

    def test_missing_list_selects_nothing_and_warns(self):
        output = io.StringIO()
        with redirect_stdout(output):
            mask = Command_Engine.evaluate_file_mask(self.HEADERS, "absent")
        np.testing.assert_array_equal(mask, [False, False, False, False])
        self.assertEqual(
            output.getvalue(),
            f"Warning: Could not find file 'absent.txt' in {self.header_dir}\n",
        )


if __name__ == "__main__":
    unittest.main()
