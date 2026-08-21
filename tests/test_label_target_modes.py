"""Focused coverage for label target aliases and result filtering."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import label


class LabelTargetModeTests(unittest.TestCase):
    def setUp(self):
        alignment = SimpleNamespace(
            aln=MultipleSeqAlignment([
                SeqRecord(Seq("A"), id="node0"),
                SeqRecord(Seq("C"), id="node1"),
                SeqRecord(Seq("D"), id="node2"),
            ]),
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


if __name__ == "__main__":
    unittest.main()
