"""Label analysis helpers in commands/label.py.

Covers residue counts and the Global Stats profile cell, weighted column
statistics, the gmax outside background, subset tasks for each target mode,
and argument and threshold parsing. The command itself, end to end, is in
test_label_command.
"""

import os
import sys
import unittest
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

from commands import label
from commands.label import (
    _calculate_outside_frequency,
    _calculate_weighted_frequencies,
    _format_global_amino_acid_profile,
    _get_amino_acid_counts,
    _get_indexed_amino_acid_count,
)
from tests.sparse_alignment import sparse_alignment


GAP_CHARS = frozenset(("-", "."))


def frozen_alignment(*sequences):
    """Rows as label analyses the global alignment: a frozen sparse copy."""
    return label._FrozenSparseAlignment(
        sparse_alignment(
            (f"row{index}", sequence) for index, sequence in enumerate(sequences)
        )
    )


class LabelGlobalProfileTests(unittest.TestCase):
    @staticmethod
    def global_profile(alignment, col_idx=0):
        """Format a Global Stats cell as an unweighted label run does.

        label counts the column over the whole frozen alignment, divides by
        every aligned row (gap rows included) and passes those frequencies in.
        """
        counts = _get_amino_acid_counts(alignment, col_idx, gap_chars=GAP_CHARS)
        frequencies = {aa: count / len(alignment) for aa, count in counts.items()}
        return _format_global_amino_acid_profile(frequencies)

    def test_profile_matches_query_format_and_omits_gaps(self):
        profile = self.global_profile(frozen_alignment("A", "A", "C", "-"))

        self.assertEqual(profile, "A  50.0% | C  25.0%")
        self.assertNotIn("-", profile)

    def test_profile_lists_residues_by_descending_frequency(self):
        profile = self.global_profile(frozen_alignment("A", "C", "C", "-"))

        self.assertEqual(profile, "C  50.0% | A  25.0%")

    def test_all_gap_column_is_reported_as_empty(self):
        alignment = frozen_alignment("A-", "C-", "A-")

        self.assertEqual(self.global_profile(alignment, col_idx=1), "-")

    def test_weighted_frequency_stats_match_for_dense_and_sparse_alignments(self):
        dense_alignment = MultipleSeqAlignment(
            [
                SeqRecord(Seq("A-"), id="a_gap"),
                SeqRecord(Seq("AA"), id="a_full"),
                SeqRecord(Seq("CA"), id="c_full"),
            ]
        )
        sparse_alignment = frozen_alignment("A-", "AA", "CA")
        mapping = {0: "1", 1: "2"}
        weights = np.array([0.5, 0.5, 1.0])

        dense_stats, dense_counts = _calculate_weighted_frequencies(
            dense_alignment,
            mapping,
            weights,
        )
        sparse_stats, sparse_counts = _calculate_weighted_frequencies(
            sparse_alignment,
            mapping,
            weights,
        )

        self.assertEqual(dense_stats.keys(), sparse_stats.keys())
        for label in mapping.values():
            self.assertEqual(dense_stats[label][0], sparse_stats[label][0])
            np.testing.assert_allclose(dense_stats[label][1:], sparse_stats[label][1:])
            self.assertEqual(dense_counts[label].keys(), sparse_counts[label].keys())
            for amino_acid in dense_counts[label]:
                self.assertAlmostEqual(
                    dense_counts[label][amino_acid],
                    sparse_counts[label][amino_acid],
                )

        self.assertEqual(dense_stats["1"][0], "A")
        self.assertAlmostEqual(dense_stats["1"][1], 0.5)
        self.assertAlmostEqual(dense_stats["1"][2], 1.0)
        self.assertAlmostEqual(dense_stats["2"][1], 0.75)
        self.assertAlmostEqual(dense_stats["2"][2], 0.75)

    def test_shared_outside_frequency_uses_union_counts(self):
        self.assertEqual(
            _calculate_outside_frequency(
                "Y",
                global_counts={"Y": 4},
                global_size=6,
                excluded_count=4,
                excluded_size=4,
            ),
            0.0,
        )

    def test_shared_outside_frequency_rejects_empty_background(self):
        self.assertIsNone(
            _calculate_outside_frequency(
                "Y",
                global_counts={"Y": 4},
                global_size=4,
                excluded_count=4,
                excluded_size=4,
            )
        )

    def test_indexed_counts_deduplicate_rows_and_preserve_weights(self):
        alignment = frozen_alignment("Y", "Y", "A")
        count = _get_indexed_amino_acid_count(
            alignment,
            0,
            "Y",
            [0, 1, 1],
            weights=np.array([0.5, 0.25, 1.0]),
        )
        self.assertAlmostEqual(count, 0.75)

    def test_indexed_counts_skip_selected_rows_with_other_residues(self):
        alignment = frozen_alignment("Y", "A", "Y")

        count = _get_indexed_amino_acid_count(
            alignment, 0, "Y", [0, 1, 2], weights=np.array([0.5, 1.0, 0.25])
        )

        self.assertAlmostEqual(count, 0.75)


class LabelAmbiguityCodeTests(unittest.TestCase):
    """X, B, Z and J occupy a position but are never its consensus residue."""

    MAPPING = {0: "1"}

    @staticmethod
    def dense_alignment(*sequences):
        return MultipleSeqAlignment(
            [SeqRecord(Seq(sequence), id=f"row{index}") for index, sequence in enumerate(sequences)]
        )

    def stats_for_every_reading_path(self, sequences, weights):
        """The Global Stats of one column from each way label reads columns."""
        weights = np.asarray(weights, dtype=float)
        results = {
            "dense": _calculate_weighted_frequencies(
                self.dense_alignment(*sequences), self.MAPPING, weights
            ),
            "row-major": _calculate_weighted_frequencies(
                frozen_alignment(*sequences), self.MAPPING, weights
            ),
        }
        with mock.patch.object(label, "_ROW_MAJOR_ENTRY_LIMIT", 0):
            results["column-block"] = _calculate_weighted_frequencies(
                frozen_alignment(*sequences), self.MAPPING, weights
            )
        return results

    def test_ambiguity_codes_count_toward_occupancy_but_never_as_consensus(self):
        for code in "XBZJ":
            with self.subTest(code=code):
                results = self.stats_for_every_reading_path(
                    [code, code, code, "A", "-"], np.ones(5)
                )
                for path, (stats, counts) in results.items():
                    with self.subTest(path=path):
                        consensus, frequency, occupancy = stats["1"]
                        self.assertEqual(consensus, "A")
                        self.assertAlmostEqual(frequency, 0.2)
                        self.assertAlmostEqual(occupancy, 0.8)
                        self.assertEqual(counts["1"], {code: 3.0, "A": 1.0})

    def test_a_column_of_only_ambiguity_codes_has_no_consensus_but_full_occupancy(self):
        for code in "XBZJ":
            with self.subTest(code=code):
                results = self.stats_for_every_reading_path(
                    [code, code, "-", code], np.ones(4)
                )
                for path, (stats, counts) in results.items():
                    with self.subTest(path=path):
                        self.assertEqual(stats["1"][0], "-")
                        self.assertEqual(stats["1"][1], 0.0)
                        self.assertAlmostEqual(stats["1"][2], 0.75)
                        self.assertEqual(counts["1"], {code: 3.0})

    def test_weighted_consensus_skips_an_ambiguity_code_that_outweighs_it(self):
        # X holds 3 of the 4.5 weight, A 1.5: A is the consensus at 1.5/4.5.
        results = self.stats_for_every_reading_path(
            ["X", "X", "A", "A"], [1.0, 2.0, 0.5, 1.0]
        )
        for path, (stats, _counts) in results.items():
            with self.subTest(path=path):
                self.assertEqual(stats["1"][0], "A")
                self.assertAlmostEqual(stats["1"][1], 1.5 / 4.5)
                self.assertAlmostEqual(stats["1"][2], 1.0)

    def test_selenocysteine_and_pyrrolysine_can_be_the_consensus(self):
        for residue in "UO":
            with self.subTest(residue=residue):
                stats, _counts = _calculate_weighted_frequencies(
                    frozen_alignment(residue, residue, "X"), self.MAPPING, np.ones(3)
                )
                self.assertEqual(stats["1"][0], residue)
                self.assertAlmostEqual(stats["1"][1], 2 / 3)


class LabelTargetModeTests(unittest.TestCase):
    def setUp(self):
        alignment = SimpleNamespace(
            aln=sparse_alignment([("node0", "A"), ("node1", "C"), ("node2", "D")]),
            col_to_label={0: "1"},
        )
        self.viewer = SimpleNamespace(
            alignment=alignment,
            full_headers=["node0", "node1", "node2"],
            cluster_labels=np.asarray([1, 2, -1], dtype=int),
            group_labels=[{"alpha"}, {"beta"}, set()],
        )
        self.viewer_to_aln = np.asarray([0, 1, 2], dtype=int)

    def test_singular_and_plural_targets_are_aliases(self):
        self.assertEqual(label._parse_label_arguments([])["forced_target"], "all")
        for token in ("cluster", "clusters"):
            with self.subTest(token=token):
                self.assertEqual(
                    label._parse_label_arguments([token])["forced_target"],
                    "clusters",
                )
        for token in ("group", "groups"):
            with self.subTest(token=token):
                self.assertEqual(
                    label._parse_label_arguments([token])["forced_target"],
                    "groups",
                )

    def test_default_mode_builds_cluster_and_group_results(self):
        tasks = label._build_label_tasks(self.viewer, "all", self.viewer_to_aln)

        self.assertEqual(
            [(task[0], task[1]) for task in tasks],
            [("cluster", 1), ("cluster", 2), ("group", "alpha"), ("group", "beta")],
        )

    def test_cluster_modes_retain_only_cluster_results(self):
        for token in ("cluster", "clusters"):
            with self.subTest(token=token):
                mode = label._parse_label_arguments([token])["forced_target"]
                tasks = label._build_label_tasks(
                    self.viewer,
                    mode,
                    self.viewer_to_aln,
                )
                self.assertEqual(
                    [(task[0], task[1]) for task in tasks],
                    [("cluster", 1), ("cluster", 2)],
                )

    def test_group_modes_retain_only_group_results(self):
        for token in ("group", "groups"):
            with self.subTest(token=token):
                mode = label._parse_label_arguments([token])["forced_target"]
                tasks = label._build_label_tasks(
                    self.viewer,
                    mode,
                    self.viewer_to_aln,
                )
                self.assertEqual(
                    [(task[0], task[1]) for task in tasks],
                    [("group", "alpha"), ("group", "beta")],
                )

    def test_default_mode_uses_whichever_result_types_are_available(self):
        self.viewer.cluster_labels = None
        group_tasks = label._build_label_tasks(
            self.viewer,
            "all",
            self.viewer_to_aln,
        )
        self.assertEqual(
            [(task[0], task[1]) for task in group_tasks],
            [("group", "alpha"), ("group", "beta")],
        )

        self.viewer.cluster_labels = np.asarray([1, 2, -1], dtype=int)
        self.viewer.group_labels = None
        cluster_tasks = label._build_label_tasks(
            self.viewer,
            "all",
            self.viewer_to_aln,
        )
        self.assertEqual(
            [(task[0], task[1]) for task in cluster_tasks],
            [("cluster", 1), ("cluster", 2)],
        )


class LabelThresholdParsingTests(unittest.TestCase):
    """gmax and cmin accept fractions or percentages, as logo's identity does."""

    def test_a_trailing_percent_sign_always_means_percent(self):
        for text, expected in (("0.5%", 0.005), ("1%", 0.01), ("40%", 0.40), ("100%", 1.0)):
            with self.subTest(text=text):
                self.assertAlmostEqual(label.parse_percentage(text), expected)

    def test_bare_values_keep_their_meaning(self):
        for text, expected in (("0.4", 0.4), ("1", 1.0), ("40", 0.40), ("98", 0.98)):
            with self.subTest(text=text):
                self.assertAlmostEqual(label.parse_percentage(text), expected)

    def test_non_numbers_are_not_thresholds(self):
        for text in ("report", "nan", "inf", "-inf", "%", "40%.xlsx"):
            with self.subTest(text=text):
                self.assertIsNone(label.parse_percentage(text))

    def test_small_percentages_reach_the_analysis(self):
        keyword = label._parse_label_arguments(["gmax", "0.5%", "cmin", "1%"])
        self.assertAlmostEqual(keyword["global_max"], 0.005)
        self.assertAlmostEqual(keyword["cluster_min"], 0.01)
        positional = label._parse_label_arguments(["0.5%", "1%"])
        self.assertAlmostEqual(positional["global_max"], 0.005)
        self.assertAlmostEqual(positional["cluster_min"], 0.01)

    def test_non_finite_thresholds_are_rejected_in_every_position(self):
        """A positional nan or inf was once taken as the report filename."""
        for args in (["nan"], ["0.4", "inf"], ["0.4", "0.9", "-inf"], ["INF%"],
                     ["1e400"], ["gmax", "nan"], ["cmin", "inf"]):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "Invalid percentage"):
                    label._parse_label_arguments(args)

    def test_a_non_finite_name_can_still_be_the_report_filename(self):
        parsed = label._parse_label_arguments(["0.4", "nan.xlsx"])
        self.assertAlmostEqual(parsed["global_max"], 0.4)
        self.assertEqual(parsed["requested_filename"], "nan.xlsx")

    def test_a_percentage_is_exactly_its_decimal_fraction(self):
        # 99.9 / 100 is 0.9990000000000001, which 999 of 1000 rows fall short of.
        for text, expected in (("99.9%", 0.999), ("29%", 0.29), ("57", 0.57), ("0.5%", 0.005)):
            with self.subTest(text=text):
                self.assertEqual(label.parse_percentage(text), expected)

    def test_only_plain_numbers_are_thresholds(self):
        for text in ("2026_01_01", "1_0", "1_000%", "0x10", "1e", "4 0", "٤٠"):
            with self.subTest(text=text):
                self.assertIsNone(label.parse_percentage(text))
        for text, expected in (("+40", 0.4), ("-5%", -0.05), ("-5", -5.0), (".5", 0.5), ("5.", 0.05),
                               ("1e-1", 0.1), (" 40 % ", 0.4)):
            with self.subTest(text=text):
                self.assertAlmostEqual(label.parse_percentage(text), expected)

    def test_an_underscored_number_or_trailing_dot_is_a_filename(self):
        for args, expected in (
            (["2026_01_01"], "2026_01_01.xlsx"),
            (["0.4", "1_000"], "1_000.xlsx"),
            (["report.xlsx."], "report.xlsx"),
            (["report."], "report.xlsx"),
            (["report.xlsx"], "report.xlsx"),
        ):
            with self.subTest(args=args):
                parsed = label._parse_label_arguments(args)
                self.assertEqual(parsed["requested_filename"], expected)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            label._parse_label_arguments(["..."])

    def test_assignments_and_flags_are_not_filenames(self):
        for args in (["gmax=0.4"], ["cmin=90%"], ["id=90"], ["-x"], ["--help"],
                     ["0.4", "-report"], ["report=1.xlsx"]):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "Unrecognized argument"):
                    label._parse_label_arguments(args)

    def test_gmax_and_cmin_must_lie_between_zero_and_one_hundred_percent(self):
        for args in (["cmin", "-1"], ["gmax", "-5"], ["cmin", "150%"], ["gmax", "101"],
                     ["-5"], ["0.4", "-1%"], ["0.4", "100.1"]):
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "outside the supported range"):
                    label._parse_label_arguments(args)
        parsed = label._parse_label_arguments(["gmax", "0", "cmin", "100%"])
        self.assertEqual((parsed["global_max"], parsed["cluster_min"]), (0.0, 1.0))
        parsed = label._parse_label_arguments(["0%", "1"])
        self.assertEqual((parsed["global_max"], parsed["cluster_min"]), (0.0, 1.0))


class LabelSubsetDefinitionTests(unittest.TestCase):
    def test_clusters_are_defined_only_by_a_label_other_than_noise(self):
        self.assertFalse(label._has_clusters(None))
        self.assertFalse(label._has_clusters([]))
        self.assertFalse(label._has_clusters(np.array([-1, -1])))
        self.assertTrue(label._has_clusters((-1, 0)))
        self.assertTrue(label._has_clusters(np.array([-1, 3])))

    def test_groups_are_defined_only_by_a_non_empty_set(self):
        self.assertFalse(label._has_groups(None))
        self.assertFalse(label._has_groups([]))
        self.assertFalse(label._has_groups([set(), None, frozenset()]))
        self.assertTrue(label._has_groups([set(), {"a"}]))

    def test_subsets_without_an_aligned_row_are_listed(self):
        viewer = SimpleNamespace(
            cluster_labels=np.array([0, 1, 2, -1]),
            group_labels=[{"a"}, {"b"}, set(), {"a", "c"}],
        )
        tasks = [("cluster", 0, None, None, None), ("group", "a", None, None, None)]

        self.assertEqual(
            label._subsets_without_aligned_members(viewer, "all", tasks),
            ["Cluster 1", "Cluster 2", "Group b", "Group c"],
        )
        self.assertEqual(
            label._subsets_without_aligned_members(viewer, "clusters", tasks),
            ["Cluster 1", "Cluster 2"],
        )
        self.assertEqual(
            label._subsets_without_aligned_members(viewer, "groups", tasks),
            ["Group b", "Group c"],
        )
        viewer.cluster_labels = None
        viewer.group_labels = None
        self.assertEqual(label._subsets_without_aligned_members(viewer, "all", []), [])


if __name__ == "__main__":
    unittest.main()
