import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for directory in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from utilities.Network_Kernels import (  # noqa: E402
    global_local_scores,
    global_score_length_identity,
    local_score_length_identity,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_SSEARCH as embedding_ssearch
    import Network_Injection as network_injection


def reference_global_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
):
    num_rows, num_cols = score_matrix.shape
    scores = np.zeros((num_rows + 1, num_cols + 1), dtype=np.float32)
    pointers = np.zeros((num_rows + 1, num_cols + 1), dtype=np.int8)

    for col in range(1, num_cols + 1):
        scores[0, col] = col * gap_penalty
        pointers[0, col] = 3
    for row in range(1, num_rows + 1):
        scores[row, 0] = row * gap_penalty
        pointers[row, 0] = 2

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            match = (
                scores[row - 1, col - 1]
                + score_matrix[row - 1, col - 1]
            )
            delete = scores[row - 1, col] + gap_penalty
            insert = scores[row, col - 1] + gap_penalty
            best_score = match
            pointer = 1
            if delete > best_score:
                best_score = delete
                pointer = 2
            if insert > best_score:
                best_score = insert
                pointer = 3
            scores[row, col] = best_score
            pointers[row, col] = pointer

    row, col = num_rows, num_cols
    path_length = 0
    identities = 0
    while row > 0 or col > 0:
        pointer = pointers[row, col]
        path_length += 1
        if pointer == 1:
            identities += int(query_codes[row - 1] == target_codes[col - 1])
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return scores[num_rows, num_cols], path_length, identities


def reference_local_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
    score_shift=2.0,
):
    shifted_matrix = score_matrix.copy()
    shifted_matrix -= score_shift
    num_rows, num_cols = shifted_matrix.shape
    scores = np.zeros((num_rows + 1, num_cols + 1), dtype=np.float32)
    pointers = np.zeros((num_rows + 1, num_cols + 1), dtype=np.int8)
    max_score = 0.0
    max_position = (0, 0)

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            match = (
                scores[row - 1, col - 1]
                + shifted_matrix[row - 1, col - 1]
            )
            delete = scores[row - 1, col] + gap_penalty
            insert = scores[row, col - 1] + gap_penalty
            best_score = 0.0
            pointer = 0
            if match > best_score:
                best_score = match
                pointer = 1
            if delete > best_score:
                best_score = delete
                pointer = 2
            if insert > best_score:
                best_score = insert
                pointer = 3
            scores[row, col] = best_score
            pointers[row, col] = pointer
            if best_score > max_score:
                max_score = best_score
                max_position = (row, col)

    row, col = max_position
    path_length = 0
    identities = 0
    while row > 0 and col > 0 and scores[row, col] != 0:
        pointer = pointers[row, col]
        if pointer == 0:
            break
        path_length += 1
        if pointer == 1:
            identities += int(query_codes[row - 1] == target_codes[col - 1])
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return max_score, path_length, identities


def residue_codes(rng, length):
    # A three-letter alphabet makes identical pairs common on random paths.
    return rng.integers(0, 3, size=length).astype(np.uint8)


class AlignmentScoreKernelTests(unittest.TestCase):
    def test_shared_kernels_match_traceback_references(self):
        rng = np.random.default_rng(4815)
        cases = [
            np.zeros((1, 1), dtype=np.float32),
            np.full((2, 3), 3.0, dtype=np.float32),
            rng.normal(size=(3, 5)).astype(np.float32),
            rng.normal(size=(7, 2)).astype(np.float32),
            rng.normal(size=(8, 9)).astype(np.float32),
        ]
        # More shapes exercise gaps on both sides of many selected paths.
        for rows, cols in rng.integers(1, 13, size=(20, 2)):
            cases.append(rng.normal(size=(rows, cols)).astype(np.float32))

        for matrix in cases:
            query_codes = residue_codes(rng, matrix.shape[0])
            target_codes = residue_codes(rng, matrix.shape[1])
            for global_gap in (0.0, -1.0):
                expected_global = reference_global_score_length_identity(
                    matrix,
                    global_gap,
                    query_codes,
                    target_codes,
                )
                actual_global = global_score_length_identity(
                    matrix,
                    global_gap,
                    query_codes,
                    target_codes,
                )
                self.assertAlmostEqual(
                    float(actual_global[0]),
                    float(expected_global[0]),
                    places=5,
                )
                self.assertEqual(
                    [int(value) for value in actual_global[1:]],
                    list(expected_global[1:]),
                )

            for local_gap in (-1.0, -2.0, -4.0):
                expected_local = reference_local_score_length_identity(
                    matrix,
                    local_gap,
                    query_codes,
                    target_codes,
                )
                actual_local = local_score_length_identity(
                    matrix,
                    local_gap,
                    query_codes,
                    target_codes,
                )
                self.assertAlmostEqual(
                    float(actual_local[0]),
                    float(expected_local[0]),
                    places=5,
                )
                self.assertEqual(
                    [int(value) for value in actual_local[1:]],
                    list(expected_local[1:]),
                )

    def test_fused_kernel_matches_individual_kernels(self):
        rng = np.random.default_rng(20260729)
        matrix = rng.normal(size=(11, 13)).astype(np.float32)
        query_codes = residue_codes(rng, 11)
        target_codes = residue_codes(rng, 13)

        expected_global = global_score_length_identity(
            matrix, 0.0, query_codes, target_codes
        )
        expected_local = local_score_length_identity(
            matrix, -2.0, query_codes, target_codes
        )
        actual = global_local_scores(matrix, 0.0, -2.0)

        self.assertAlmostEqual(
            float(actual[0]),
            float(expected_global[0]),
            places=5,
        )
        self.assertEqual(int(actual[1]), int(expected_global[1]))
        self.assertAlmostEqual(
            float(actual[2]),
            float(expected_local[0]),
            places=5,
        )
        self.assertEqual(int(actual[3]), int(expected_local[1]))

    def test_callers_share_the_same_kernel_objects(self):
        self.assertIs(
            similarity_matrix.global_local_scores,
            global_local_scores,
        )
        self.assertIs(
            network_injection.global_local_scores,
            global_local_scores,
        )
        self.assertIs(
            embedding_ssearch.global_score_length_identity,
            global_score_length_identity,
        )
        self.assertIs(
            embedding_ssearch.local_score_length_identity,
            local_score_length_identity,
        )

    def test_ssearch_uses_only_the_selected_mode_and_preserves_matrix(self):
        matrix = np.array(
            [
                [3.0, -1.0],
                [-1.0, 3.0],
            ],
            dtype=np.float32,
        )
        original = matrix.copy()

        local_result = embedding_ssearch.finish_search(
            (0, "local", 2, 2, "local", -2.0, "alignment_length", matrix,
             "AC", "AD")
        )
        np.testing.assert_array_equal(matrix, original)
        self.assertEqual(float(local_result["raw_score"]), 2.0)
        self.assertEqual(int(local_result["aln_len"]), 2)
        self.assertEqual(local_result["identity"], 50.0)

        global_result = embedding_ssearch.finish_search(
            (1, "global", 2, 2, "global", 0.0, "alignment_length", matrix,
             "AC", "AD")
        )
        np.testing.assert_array_equal(matrix, original)
        self.assertEqual(float(global_result["raw_score"]), 6.0)
        self.assertEqual(int(global_result["aln_len"]), 2)
        self.assertEqual(global_result["identity"], 50.0)

    def test_ssearch_identity_denominators_follow_alignment_mode(self):
        # Only the C/C pair aligns well, so a global path pays two end gaps
        # while the local path is that single pair. Transposing the case
        # moves the gaps from the target side to the query side.
        end_gap_matrix = np.array([[-5.0], [3.0], [-5.0]], dtype=np.float32)
        for matrix, query, target in (
            (end_gap_matrix, "ACD", "C"),
            (end_gap_matrix.T.copy(), "C", "ACD"),
        ):
            with self.subTest(query=query, target=target):
                rows, cols = matrix.shape
                local_result = embedding_ssearch.finish_search(
                    (0, "t", rows, cols, "local", -2.0, "alignment_length",
                     matrix, query, target)
                )
                global_result = embedding_ssearch.finish_search(
                    (0, "t", rows, cols, "global", 0.0, "alignment_length",
                     matrix, query, target)
                )
                self.assertEqual(int(local_result["aln_len"]), 1)
                self.assertEqual(local_result["identity"], 100.0)
                self.assertEqual(int(global_result["aln_len"]), 3)
                self.assertAlmostEqual(global_result["identity"], 100.0 / 3.0)

        # A/A and W/W flank one unpaired residue, so the local path keeps the
        # internal gap in its denominator on either side of the alignment.
        internal_gap_matrix = np.full((3, 2), -10.0, dtype=np.float32)
        internal_gap_matrix[0, 0] = 4.0
        internal_gap_matrix[2, 1] = 4.0
        for matrix, query, target in (
            (internal_gap_matrix, "ACW", "AW"),
            (internal_gap_matrix.T.copy(), "AW", "ACW"),
        ):
            with self.subTest(query=query, target=target):
                rows, cols = matrix.shape
                result = embedding_ssearch.finish_search(
                    (0, "t", rows, cols, "local", -1.0, "alignment_length",
                     matrix, query, target)
                )
                self.assertEqual(int(result["aln_len"]), 3)
                self.assertAlmostEqual(result["identity"], 200.0 / 3.0)

    def test_ssearch_identity_counts_only_standard_amino_acids(self):
        residues = "AXBZJUOC"
        matrix = np.where(
            np.eye(len(residues), dtype=bool), 4.0, -10.0
        ).astype(np.float32)
        result = embedding_ssearch.finish_search(
            (0, "t", len(residues), len(residues), "local", -2.0,
             "alignment_length", matrix, residues, residues)
        )
        self.assertEqual(int(result["aln_len"]), len(residues))
        self.assertEqual(result["identity"], 25.0)

    def test_ssearch_rejects_sequences_that_do_not_match_the_matrix(self):
        matrix = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "has shape"):
            embedding_ssearch.finish_search(
                (0, "t", 2, 3, "local", -2.0, "alignment_length", matrix,
                 "AC", "ACDE")
            )


if __name__ == "__main__":
    unittest.main()
