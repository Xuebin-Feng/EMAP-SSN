"""Pairwise identity within groups of aligned rows (utilities/Alignment_Profiles.py):
group_identities and its kernels, against a direct count of each pair."""

import itertools
import os
import sys
import unittest
from unittest import mock

import numpy as np
from scipy import sparse


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Alignment_Profiles  # noqa: E402
from utilities.Alignment_Profiles import (  # noqa: E402
    group_identities,
    residue_column_table,
    standard_residue_table,
)

# Sparse codes as the loaders store them: 0 a gap, then letters; 'X' and 'B'
# are not standard amino acids, and code 30 is in no table.
LETTERS = "ACDEFGHIKLMNPQRSTVWYXB"
INT_TO_AA = {code: letter for code, letter in enumerate(LETTERS, start=1)}


def code_matrix(rows):
    """A CSR code matrix from aligned strings ('-' is a gap)."""
    codes = [[0 if ch == "-" else LETTERS.index(ch) + 1 for ch in row] for row in rows]
    return sparse.csr_matrix(np.asarray(codes, dtype=np.uint8))


def pair_identity(left, right):
    """The tools' local identity of two aligned rows, or None without an aligned residue pair.

    Identical standard residues / the columns from the first to the last
    column where both rows hold a letter, counting those where either does.
    """
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    both = [column for column, (a, b) in enumerate(zip(left, right)) if a != "-" and b != "-"]
    if not both:
        return None
    length = sum(left[c] != "-" or right[c] != "-" for c in range(both[0], both[-1] + 1))
    matches = sum(a in standard and a == b for a, b in zip(left, right))
    return 100.0 * matches / length


def expected_identities(rows, node_rows, labels):
    """{label: (mean, minimum, pairs, members with a row)} by a direct count."""
    expected = {}
    for label in sorted(set(labels) - {-1}):
        members = [node_rows[i] for i, value in enumerate(labels) if value == label and node_rows[i] >= 0]
        values = [pair_identity(rows[a], rows[b]) for a, b in itertools.combinations(members, 2)]
        values = [value for value in values if value is not None]
        if values:
            expected[label] = (sum(values) / len(values), min(values), len(values), len(members))
        else:
            expected[label] = (None, None, 0, len(members))
    return expected


def random_rows(rng, count, length, gap_share=0.3):
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWYXB"))
    # A few ancestors keep pairs from all being near 5% identity.
    ancestors = rng.choice(alphabet[:20], size=(3, length))
    rows = []
    for _ in range(count):
        row = ancestors[rng.integers(3)].copy()
        mutate = rng.random(length) < rng.uniform(0.1, 0.8)
        row[mutate] = rng.choice(alphabet, size=mutate.sum())
        row[rng.random(length) < gap_share] = "-"
        rows.append("".join(row))
    return rows


class GroupIdentityTests(unittest.TestCase):
    def assert_matches(self, identities, expected):
        self.assertEqual(set(identities), set(expected))
        for label, (mean, minimum, pairs, members) in expected.items():
            identity = identities[label]
            self.assertEqual((identity.pairs, identity.rows, identity.sampled), (pairs, members, False), label)
            if mean is None:
                self.assertTrue(np.isnan(identity.mean) and np.isnan(identity.minimum), label)
            else:
                self.assertAlmostEqual(identity.mean, mean, places=9, msg=label)
                self.assertAlmostEqual(identity.minimum, minimum, places=9, msg=label)

    def test_random_groups_match_a_direct_count_on_numba_and_on_numpy(self):
        rng = np.random.default_rng(42)
        for trial in range(12):
            count = int(rng.integers(2, 40))
            rows = random_rows(rng, count, int(rng.integers(1, 150)), gap_share=rng.uniform(0, 0.9))
            n_nodes = count + 3
            # Three nodes have no row; rows are reached in a shuffled order.
            node_rows = np.concatenate([rng.permutation(count), [-1, -1, -1]])
            labels = rng.integers(-1, 4, n_nodes)
            expected = expected_identities(rows, node_rows, labels.tolist())
            for numba in (True, False):
                with self.subTest(trial=trial, numba=numba), \
                        mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                    self.assert_matches(
                        group_identities(code_matrix(rows), INT_TO_AA, node_rows, labels), expected)

    def test_a_column_region_counts_only_its_columns_on_numba_and_on_numpy(self):
        rng = np.random.default_rng(23)
        for trial in range(6):
            count, length = int(rng.integers(2, 30)), int(rng.integers(2, 140))
            rows = random_rows(rng, count, length, gap_share=rng.uniform(0, 0.9))
            labels = rng.integers(-1, 3, count)
            columns = sorted(rng.choice(length, size=max(1, length // 3), replace=False).tolist())
            for numba in (True, False):
                with self.subTest(trial=trial, numba=numba), \
                        mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                    identities = group_identities(code_matrix(rows), INT_TO_AA, np.arange(count), labels,
                                                  columns=columns)
                    for label in set(labels.tolist()) - {-1}:
                        members = np.flatnonzero(labels == label)
                        values = [region_identity(rows[a], rows[b], columns)
                                  for i, a in enumerate(members) for b in members[i + 1:]]
                        values = [value for value in values if value is not None]
                        self.assertEqual(identities[label].pairs, len(values))
                        if values:
                            self.assertAlmostEqual(identities[label].mean, sum(values) / len(values), places=9)
                            self.assertAlmostEqual(identities[label].minimum, min(values), places=9)

    def test_internal_gaps_count_and_overhanging_ends_do_not(self):
        cases = [
            # The overhang before and after the shared span is not counted.
            ("--AAAA--", "AAAAAAAA", 100.0),
            # An internal gap in either row lengthens the alignment.
            ("AA--AA", "AAAAAA", 400 / 6),
            ("AA-AA", "A-AAA", 300 / 5),
            # X counts toward the length, and X against X is no identity.
            ("ACDE-X", "ACDKAA", 50.0),
            ("XXAA", "XXAA", 50.0),
            # Columns where both rows are gaps are no part of the alignment.
            ("A--A", "A--A", 100.0),
        ]
        for left, right, expected in cases:
            for numba in (True, False):
                with self.subTest(left=left, right=right, numba=numba), \
                        mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                    identity = group_identities(code_matrix([left, right]), INT_TO_AA, [0, 1], [1, 1])[1]
                    self.assertEqual(identity.pairs, 1)
                    self.assertAlmostEqual(identity.mean, expected)

    def test_rows_without_an_aligned_residue_pair_have_no_value(self):
        rows = ["AA----", "----AA", "------", "XXXXXX"]
        identities = group_identities(code_matrix(rows), INT_TO_AA, [0, 1, 2, 3], [1, 1, 2, 2])
        for group in (1, 2):
            self.assertEqual((identities[group].pairs, identities[group].rows), (0, 2))
            self.assertTrue(np.isnan(identities[group].mean))

    def test_a_long_row_spans_word_boundaries(self):
        # Aligned pairs only at columns 70 and 200, three words apart: the
        # length counts the 131 columns between them that either row fills.
        left = ["-"] * 260
        right = ["-"] * 260
        left[70] = right[70] = left[200] = right[200] = "A"
        for column in range(71, 200, 2):
            left[column] = "C"
        for column in (5, 250):
            right[column] = "W"
        rows = ["".join(left), "".join(right)]
        for numba in (True, False):
            with self.subTest(numba=numba), mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                identity = group_identities(code_matrix(rows), INT_TO_AA, [0, 1], [1, 1])[1]
                self.assertAlmostEqual(identity.mean, 100.0 * 2 / (2 + 65))
                self.assertAlmostEqual(identity.mean, pair_identity(*rows))

    def test_lower_case_codes_count_and_unknown_codes_do_not(self):
        codes = {1: "a", 2: "W", 3: "X", 4: "-", 5: "AB", 6: ".", 7: 5}
        np.testing.assert_array_equal(
            standard_residue_table(codes, 9), [-1, 0, 18, -1, -1, -1, -1, -1, -1])
        # Every letter holds a residue; gaps, unnamed codes and code 0 do not.
        np.testing.assert_array_equal(
            residue_column_table(codes, 9), [False, True, True, True, False, False, False, False, False])

    def test_noise_and_groups_without_two_rows(self):
        rows = ["AAAA", "AAAC", "CCCC"]
        identities = group_identities(code_matrix(rows), INT_TO_AA, [0, 1, 2, -1], [1, 1, -1, 2])
        self.assertEqual(set(identities), {1, 2})
        self.assertNotIn(-1, identities)
        self.assertAlmostEqual(identities[1].mean, 75.0)
        self.assertEqual((identities[2].pairs, identities[2].rows), (0, 0))

    def test_a_group_over_the_exact_limit_is_sampled_with_the_fixed_seed(self):
        rng = np.random.default_rng(7)
        rows = random_rows(rng, 60, 80)
        labels = np.ones(60, dtype=int)
        exact = group_identities(code_matrix(rows), INT_TO_AA, np.arange(60), labels)[1]
        self.assertFalse(exact.sampled)
        for numba in (True, False):
            with self.subTest(numba=numba), mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                sampled = group_identities(code_matrix(rows), INT_TO_AA, np.arange(60), labels,
                                           exact_pairs=1000, sample_pairs=20000)[1]
                self.assertTrue(sampled.sampled)
                self.assertLessEqual(sampled.pairs, 20000)
                self.assertGreater(sampled.pairs, 19000)
                self.assertAlmostEqual(sampled.mean, exact.mean, delta=0.5)
                self.assertGreaterEqual(sampled.minimum, exact.minimum)
        # The same members give the same sample, whatever else is labelled.
        again = group_identities(code_matrix(rows), INT_TO_AA, np.concatenate([[-1], np.arange(60)]),
                                 np.concatenate([[5], labels]), exact_pairs=1000, sample_pairs=20000)[1]
        self.assertEqual((again.mean, again.minimum, again.pairs), (sampled.mean, sampled.minimum, sampled.pairs))

    def test_exact_sums_do_not_depend_on_the_thread_count(self):
        if not Alignment_Profiles.NUMBA_AVAILABLE:
            self.skipTest("Numba is not installed")
        rng = np.random.default_rng(3)
        codes = code_matrix(random_rows(rng, 301, 200)).toarray()
        planes = Alignment_Profiles._pack_local_identity_planes(
            standard_residue_table(INT_TO_AA, 23)[codes], residue_column_table(INT_TO_AA, 23)[codes])
        previous = Alignment_Profiles.get_num_threads()
        results = []
        try:
            for threads in sorted({1, 2, min(previous, 7)}):
                Alignment_Profiles.set_num_threads(threads)
                results.append(Alignment_Profiles._all_pairs_identity_kernel(planes, True))
        finally:
            Alignment_Profiles.set_num_threads(previous)
        for sums, counts, mins in results[1:]:
            np.testing.assert_array_equal(sums, results[0][0])
            np.testing.assert_array_equal(counts, results[0][1])
            np.testing.assert_array_equal(mins, results[0][2])

    def test_the_thread_count_is_restored(self):
        if not Alignment_Profiles.NUMBA_AVAILABLE:
            self.skipTest("Numba is not installed")
        before = Alignment_Profiles.get_num_threads()
        group_identities(code_matrix(["AAAA", "AAAC"]), INT_TO_AA, [0, 1], [1, 1])
        self.assertEqual(Alignment_Profiles.get_num_threads(), before)


def region_identity(left, right, columns):
    """Identity over the given columns only: identical standard residues / those of them either row fills."""
    standard = set("ACDEFGHIKLMNPQRSTVWY")
    length = sum(left[c] != "-" or right[c] != "-" for c in columns)
    if not length:
        return None
    return 100.0 * sum(left[c] in standard and left[c] == right[c] for c in columns) / length


class IdentityMatrixTests(unittest.TestCase):
    def assert_cells(self, values, expected):
        self.assertEqual(values.shape, (len(expected), len(expected[0])))
        for i, row in enumerate(expected):
            for j, value in enumerate(row):
                if value is None:
                    self.assertTrue(np.isnan(values[i, j]), (i, j))
                else:
                    self.assertAlmostEqual(values[i, j], value, places=9, msg=(i, j))

    def test_matrices_match_a_direct_count_on_numba_and_on_numpy(self):
        rng = np.random.default_rng(11)
        for trial in range(6):
            rows = random_rows(rng, int(rng.integers(2, 30)), int(rng.integers(1, 140)), gap_share=rng.uniform(0, 0.9))
            matrix = code_matrix(rows)
            lefts = rng.permutation(len(rows))[: max(1, len(rows) // 2)]
            rights = rng.permutation(len(rows))[: max(1, len(rows) // 3)]
            columns = sorted(rng.choice(len(rows[0]), size=max(1, len(rows[0]) // 3), replace=False).tolist())
            for numba in (True, False):
                with self.subTest(trial=trial, numba=numba), \
                        mock.patch.object(Alignment_Profiles, "NUMBA_AVAILABLE", numba):
                    full = Alignment_Profiles.identity_matrix(matrix, INT_TO_AA, range(len(rows)))
                    self.assert_cells(full, [[pair_identity(a, b) for b in rows] for a in rows])
                    np.testing.assert_array_equal(full, full.T)
                    cross = Alignment_Profiles.identity_matrix(matrix, INT_TO_AA, lefts, rights)
                    self.assert_cells(cross, [[pair_identity(rows[a], rows[b]) for b in rights] for a in lefts])
                    region = Alignment_Profiles.identity_matrix(matrix, INT_TO_AA, lefts, rights, columns)
                    self.assert_cells(region, [[region_identity(rows[a], rows[b], columns) for b in rights]
                                               for a in lefts])
                    square_region = Alignment_Profiles.identity_matrix(matrix, INT_TO_AA, lefts, None, columns)
                    self.assert_cells(square_region, [[region_identity(rows[a], rows[b], columns) for b in lefts]
                                                      for a in lefts])

    def test_the_symmetric_matrix_does_not_depend_on_the_thread_count(self):
        if not Alignment_Profiles.NUMBA_AVAILABLE:
            self.skipTest("Numba is not installed")
        matrix = code_matrix(random_rows(np.random.default_rng(5), 151, 90))
        previous = Alignment_Profiles.get_num_threads()
        results = []
        try:
            for threads in sorted({1, 2, min(previous, 7)}):
                Alignment_Profiles.set_num_threads(threads)
                with mock.patch.object(Alignment_Profiles, "choose_balanced_thread_count", lambda count: count):
                    results.append(Alignment_Profiles.identity_matrix(matrix, INT_TO_AA, range(151)))
        finally:
            Alignment_Profiles.set_num_threads(previous)
        for values in results[1:]:
            np.testing.assert_array_equal(values, results[0])


if __name__ == "__main__":
    unittest.main()
