import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
if UTILITIES_DIR not in sys.path:
    sys.path.insert(0, UTILITIES_DIR)

from Alignment_Score_Kernels import (  # noqa: E402
    global_local_scores,
    global_score_length,
    local_score_length,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_SSEARCH as embedding_ssearch
    import Network_Injection as network_injection


def reference_global_score_length(score_matrix, gap_penalty):
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
    while row > 0 or col > 0:
        pointer = pointers[row, col]
        path_length += 1
        if pointer == 1:
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return scores[num_rows, num_cols], path_length


def reference_local_score_length(
    score_matrix,
    gap_penalty,
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
    while row > 0 and col > 0 and scores[row, col] != 0:
        pointer = pointers[row, col]
        if pointer == 0:
            break
        path_length += 1
        if pointer == 1:
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return max_score, path_length


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

        for matrix in cases:
            for global_gap in (0.0, -1.0):
                expected_global = reference_global_score_length(
                    matrix,
                    global_gap,
                )
                actual_global = global_score_length(matrix, global_gap)
                self.assertAlmostEqual(
                    float(actual_global[0]),
                    float(expected_global[0]),
                    places=5,
                )
                self.assertEqual(
                    int(actual_global[1]),
                    int(expected_global[1]),
                )

            for local_gap in (-1.0, -2.0, -4.0):
                expected_local = reference_local_score_length(
                    matrix,
                    local_gap,
                )
                actual_local = local_score_length(matrix, local_gap)
                self.assertAlmostEqual(
                    float(actual_local[0]),
                    float(expected_local[0]),
                    places=5,
                )
                self.assertEqual(
                    int(actual_local[1]),
                    int(expected_local[1]),
                )

    def test_fused_kernel_matches_individual_kernels(self):
        rng = np.random.default_rng(20260729)
        matrix = rng.normal(size=(11, 13)).astype(np.float32)

        expected_global = global_score_length(matrix, 0.0)
        expected_local = local_score_length(matrix, -2.0)
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
            embedding_ssearch.global_score_length,
            global_score_length,
        )
        self.assertIs(
            embedding_ssearch.local_score_length,
            local_score_length,
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
            (0, "local", 2, 2, "local", -2.0, "alignment_length", matrix)
        )
        np.testing.assert_array_equal(matrix, original)
        self.assertEqual(float(local_result["raw_score"]), 2.0)
        self.assertEqual(int(local_result["aln_len"]), 2)

        global_result = embedding_ssearch.finish_search(
            (1, "global", 2, 2, "global", 0.0, "alignment_length", matrix)
        )
        np.testing.assert_array_equal(matrix, original)
        self.assertEqual(float(global_result["raw_score"]), 6.0)
        self.assertEqual(int(global_result["aln_len"]), 2)


if __name__ == "__main__":
    unittest.main()
