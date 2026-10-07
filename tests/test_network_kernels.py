"""Alignment score kernels shared through utilities.Network_Kernels.

The traceback references below restate the score, path length and identity
definitions of the global and local kernels. The shared kernels must reproduce
them, and every tool that aligns embeddings must call those same kernel
objects. The batched microbatch kernels are covered in
test_network_kernels_batch.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for directory in (SRC_DIR, TOOLS_DIR):
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


if __name__ == "__main__":
    unittest.main()
