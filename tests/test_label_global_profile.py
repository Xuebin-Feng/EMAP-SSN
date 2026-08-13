import os
import sys
import unittest

import numpy as np
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy import sparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands.label import (
    _format_global_amino_acid_profile,
    _get_amino_acid_counts,
    _get_amino_acid_frequencies,
    _is_subset_specific_residue,
)


class SparseAlignmentStub:
    def __init__(self, matrix, int_to_aa):
        self.matrix = sparse.csr_matrix(matrix)
        self.int_to_aa = int_to_aa

    def __len__(self):
        return self.matrix.shape[0]


class LabelGlobalProfileTests(unittest.TestCase):
    def test_dense_profile_matches_query_format_and_omits_gaps(self):
        alignment = MultipleSeqAlignment(
            [
                SeqRecord(Seq("A"), id="a1"),
                SeqRecord(Seq("A"), id="a2"),
                SeqRecord(Seq("C"), id="c1"),
                SeqRecord(Seq("-"), id="gap"),
            ]
        )

        profile = _format_global_amino_acid_profile(alignment, 0)

        self.assertEqual(profile, "A  50.0% | C  25.0%")
        self.assertNotIn("-", profile)

    def test_sparse_profile_uses_all_sequences_as_denominator(self):
        alignment = SparseAlignmentStub(
            np.array([[1], [2], [2], [0]], dtype=np.uint8),
            {1: "A", 2: "C"},
        )

        profile = _format_global_amino_acid_profile(alignment, 0)

        self.assertEqual(profile, "C  50.0% | A  25.0%")

    def test_all_gap_column_is_reported_as_empty(self):
        alignment = SparseAlignmentStub(
            np.zeros((3, 1), dtype=np.uint8),
            {1: "A"},
        )

        self.assertEqual(_format_global_amino_acid_profile(alignment, 0), "-")

    def test_small_subset_uses_the_outside_background_for_gmax(self):
        self.assertFalse(
            _is_subset_specific_residue(
                "R",
                subset_frequency=1.0,
                subset_size=10,
                global_counts={"R": 2400},
                global_size=12000,
                cluster_min=0.98,
                global_max=0.05,
            )
        )

    def test_large_subset_no_longer_contaminates_its_gmax_background(self):
        self.assertTrue(
            _is_subset_specific_residue(
                "R",
                subset_frequency=1.0,
                subset_size=1800,
                global_counts={"R": 1800},
                global_size=12000,
                cluster_min=0.98,
                global_max=0.02,
            )
        )

    def test_cmin_still_filters_before_gmax(self):
        self.assertFalse(
            _is_subset_specific_residue(
                "R",
                subset_frequency=0.90,
                subset_size=100,
                global_counts={"R": 100},
                global_size=1000,
                cluster_min=0.98,
                global_max=0.30,
            )
        )

    def test_subset_covering_entire_alignment_is_not_assigned_zero_background(self):
        self.assertFalse(
            _is_subset_specific_residue(
                "R",
                subset_frequency=1.0,
                subset_size=100,
                global_counts={"R": 100},
                global_size=100,
                cluster_min=0.98,
                global_max=0.30,
            )
        )

    def test_frequency_map_is_gap_diluted(self):
        alignment = MultipleSeqAlignment(
            [
                SeqRecord(Seq("R"), id="r1"),
                SeqRecord(Seq("R"), id="r2"),
                SeqRecord(Seq("H"), id="h1"),
                SeqRecord(Seq("-"), id="gap"),
            ]
        )

        self.assertEqual(
            _get_amino_acid_frequencies(alignment, 0),
            {"R": 0.5, "H": 0.25},
        )
        self.assertEqual(
            _get_amino_acid_counts(alignment, 0),
            {"R": 2, "H": 1},
        )


if __name__ == "__main__":
    unittest.main()
