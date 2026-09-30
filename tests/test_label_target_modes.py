"""Focused coverage for label target aliases and result filtering."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import label
from tests.sparse_alignment import sparse_alignment


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


if __name__ == "__main__":
    unittest.main()
