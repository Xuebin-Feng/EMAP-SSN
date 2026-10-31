"""The substitution matrices that count similar residue pairs (utilities.Similarity_Matrices).

Covers the list of matrices, the refusal of any other name, and the uint8 table the
SSEARCH kernels read: which pairs of ASCII residue codes are similar, that identical
standard pairs always are, that no other code is, and that it is built once and
read-only. The pairwise alignment's own marks are checked against it in
test_embedding_pwa, and the kernels' counts in test_network_kernels.
"""

import os
import re
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Similarity_Matrices  # noqa: E402
from utilities.Sequence_Utils import IDENTITY_RESIDUES  # noqa: E402
from utilities.Similarity_Matrices import (  # noqa: E402
    SIMILARITY_MATRICES,
    load_similarity_matrix,
    similar_pair_table,
    validate_similarity_matrix,
)


class SimilarityMatrixListTests(unittest.TestCase):
    def test_the_blast_tools_eight_matrices_in_their_order(self):
        self.assertEqual(
            SIMILARITY_MATRICES,
            ("BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
             "PAM30", "PAM70", "PAM250"),
        )

    def test_every_listed_matrix_loads_and_scores_the_standard_residues(self):
        for name in SIMILARITY_MATRICES:
            with self.subTest(matrix=name):
                matrix = load_similarity_matrix(name)
                for a in IDENTITY_RESIDUES:
                    self.assertGreater(matrix[a, a], 0)
                    for b in IDENTITY_RESIDUES:
                        self.assertEqual(matrix[a, b], matrix[b, a])

    def test_an_unknown_matrix_is_refused_naming_the_choices(self):
        names = ("BLOSUM", "blosum62", " BLOSUM62", "BLOSUM62.txt", "../BLOSUM62", "", None, 62, ["BLOSUM62"])
        for refuse in (validate_similarity_matrix, load_similarity_matrix, similar_pair_table):
            for name in names:
                with self.subTest(function=refuse.__name__, name=name):
                    with self.assertRaisesRegex(
                        ValueError,
                        re.escape(f"Unknown SIMILARITY_MATRIX {name!r}; choose BLOSUM45, ")
                        + r".*PAM250\.$",
                    ):
                        refuse(name)

    def test_a_listed_name_is_accepted(self):
        for name in SIMILARITY_MATRICES:
            self.assertIsNone(validate_similarity_matrix(name))


class SimilarPairTableTests(unittest.TestCase):
    def test_the_table_is_a_read_only_uint8_grid_built_once_per_matrix(self):
        for name in SIMILARITY_MATRICES:
            with self.subTest(matrix=name):
                table = similar_pair_table(name)
                self.assertEqual((table.dtype, table.shape), (np.uint8, (256, 256)))
                self.assertTrue(table.flags.c_contiguous)
                self.assertFalse(table.flags.writeable)
                with self.assertRaises(ValueError):
                    table[65, 65] = 0
                self.assertIs(similar_pair_table(name), table)
        self.assertIsNot(similar_pair_table("BLOSUM62"), similar_pair_table("PAM250"))

    def test_a_pair_is_similar_when_identical_or_scored_above_zero(self):
        for name in SIMILARITY_MATRICES:
            with self.subTest(matrix=name):
                matrix = load_similarity_matrix(name)
                table = similar_pair_table(name)
                for a in IDENTITY_RESIDUES:
                    for b in IDENTITY_RESIDUES:
                        self.assertEqual(
                            int(table[ord(a), ord(b)]), int(a == b or matrix[a, b] > 0), (a, b)
                        )

    def test_every_identical_standard_pair_is_similar_and_the_table_is_symmetric(self):
        for name in SIMILARITY_MATRICES:
            table = similar_pair_table(name)
            with self.subTest(matrix=name):
                for residue in IDENTITY_RESIDUES:
                    self.assertEqual(table[ord(residue), ord(residue)], 1)
                np.testing.assert_array_equal(table, table.T)

    def test_no_other_code_is_ever_similar(self):
        standard = {ord(residue) for residue in IDENTITY_RESIDUES}
        outside = [code for code in range(256) if code not in standard]
        for name in SIMILARITY_MATRICES:
            table = similar_pair_table(name)
            with self.subTest(matrix=name):
                # X, B, Z, J, U and O, lower case, the unmatched codes 0 and 1, and the rest.
                self.assertEqual(int(table[outside].sum()), 0)
                self.assertEqual(int(table[:, outside].sum()), 0)
                for residue in "XBZJUO":
                    self.assertEqual(int(table[ord(residue)].sum()), 0, residue)
                    self.assertEqual(int(table[:, ord(residue)].sum()), 0, residue)

    def test_the_matrix_decides_which_pairs_are_similar(self):
        # K/R scores 0 in PAM30 and +3 in PAM250; S/T is BLOSUM62's lowest positive pair.
        k, r, s, t = (ord(letter) for letter in "KRST")
        self.assertEqual(int(similar_pair_table("PAM30")[k, r]), 0)
        self.assertEqual(int(similar_pair_table("PAM250")[k, r]), 1)
        self.assertEqual(int(similar_pair_table("BLOSUM62")[s, t]), 1)
        a, g = ord("A"), ord("G")
        self.assertEqual(int(similar_pair_table("BLOSUM62")[a, g]), 0)  # scores 0: not similar

    def test_blosum62_makes_the_twenty_identical_pairs_and_forty_two_others_similar(self):
        table = similar_pair_table("BLOSUM62")
        self.assertEqual(int(table.sum()), 62)
        self.assertEqual(int(np.trace(table)), 20)

    def test_biopython_loads_only_when_a_matrix_is_asked_for(self):
        import subprocess

        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from utilities import Similarity_Matrices as m;"
            "before = 'Bio.Align' in sys.modules;"
            "m.similar_pair_table('BLOSUM62');"
            "print(before, 'Bio.Align' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, SRC_DIR], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.stdout.split(), ["False", "True"], result.stderr)

    def test_the_module_offers_only_what_it_lists(self):
        self.assertEqual(
            sorted(Similarity_Matrices.__all__),
            ["SIMILARITY_MATRICES", "load_similarity_matrix", "similar_pair_table", "validate_similarity_matrix"],
        )


if __name__ == "__main__":
    unittest.main()
