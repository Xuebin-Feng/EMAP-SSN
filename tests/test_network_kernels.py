"""Alignment score kernels shared through utilities.Network_Kernels.

The traceback references below restate the score, path length and identity
definitions of the global and local kernels. The shared kernels must reproduce
them, and every tool that aligns embeddings must call those same kernel
objects. The batched microbatch kernels are covered in
test_network_kernels_batch.
"""

import builtins
import importlib.util
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for directory in (SRC_DIR, TOOLS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from utilities import Network_Kernels as network_kernels  # noqa: E402
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


BARBELL = np.asarray(
    [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)], dtype=np.int32
)


class PanicLikeError(BaseException):
    """Stands in for pyo3's PanicException, which is no Exception."""


def fake_graspologic(leiden):
    return SimpleNamespace(leiden=leiden)


class LeidenPartitionTests(unittest.TestCase):
    """leiden_partition keeps graspologic_native's panics away from its callers."""

    PATH = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    def partition(self, weights, resolution, native):
        with mock.patch.dict(sys.modules, {"graspologic_native": native}):
            return network_kernels.leiden_partition(
                3, self.PATH, weights, resolution, 1
            )

    def test_a_nan_or_infinite_resolution_is_refused_before_the_native_call(self):
        for resolution in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(resolution=resolution):
                native = fake_graspologic(mock.Mock())
                with self.assertRaises(ValueError) as caught:
                    self.partition(None, resolution, native)
                self.assertEqual(
                    str(caught.exception),
                    f"Leiden resolution must be a finite number; got {resolution}.",
                )
                native.leiden.assert_not_called()

    def test_nan_or_infinite_weights_are_refused_before_the_native_call(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(weight=bad):
                native = fake_graspologic(mock.Mock())
                with self.assertRaises(ValueError) as caught:
                    self.partition(np.asarray([1.0, bad]), 1.0, native)
                self.assertEqual(
                    str(caught.exception), "Leiden edge weights must be finite numbers."
                )
                native.leiden.assert_not_called()

    def test_a_network_without_edges_returns_noise(self):
        with mock.patch.dict(sys.modules, {"graspologic_native": fake_graspologic(mock.Mock())}):
            labels = network_kernels.leiden_partition(
                3, np.zeros((0, 2), dtype=np.int32), None, 1.0, 1
            )
        np.testing.assert_array_equal(labels, [-1, -1, -1])

    def test_a_panic_becomes_a_runtime_error(self):
        native = fake_graspologic(mock.Mock(side_effect=PanicLikeError("boom")))
        with self.assertRaises(RuntimeError) as caught:
            self.partition(None, 1.0, native)
        self.assertEqual(str(caught.exception), "Leiden clustering failed: boom")
        self.assertIsInstance(caught.exception.__cause__, PanicLikeError)

    def test_exceptions_and_interrupts_pass_through_unchanged(self):
        for error in (
            ValueError("range"),
            KeyboardInterrupt(),
            SystemExit(2),
            GeneratorExit(),
        ):
            with self.subTest(error=type(error).__name__):
                native = fake_graspologic(mock.Mock(side_effect=error))
                with self.assertRaises(type(error)) as caught:
                    self.partition(None, 1.0, native)
                self.assertIs(caught.exception, error)

    def test_the_csr_call_is_guarded_too(self):
        csr = mock.Mock(side_effect=PanicLikeError("csr boom"))
        native = SimpleNamespace(leiden=mock.Mock(), leiden_csr=csr)
        with self.assertRaises(RuntimeError):
            self.partition(np.asarray([1.0, 2.0]), 1.0, native)
        csr.assert_called_once()

    def test_valid_input_gives_the_same_partitions(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")
        cases = (
            (None, 1.0, 1, [1, 1, 1, 2, 2, 2, -1]),
            (np.arange(1.0, 8.0), 1.0, 1, [1, 1, 1, 2, 2, 2, -1]),
            (np.arange(1.0, 8.0), 0.3, 1, [1, 1, 1, 1, 1, 1, -1]),
            (None, 2.0, 3, [1, 1, 1, 2, 2, 2, -1]),
        )
        for weights, resolution, min_size, labels in cases:
            with self.subTest(resolution=resolution, min_size=min_size):
                actual = network_kernels.leiden_partition(
                    7, BARBELL, weights, resolution, min_size
                )
                np.testing.assert_array_equal(actual, labels)

    def test_the_real_library_panic_on_zero_weights_becomes_a_runtime_error(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")
        with self.assertRaises(RuntimeError) as caught:
            network_kernels.leiden_partition(
                3, self.PATH, np.zeros(2), 1.0, 1
            )
        self.assertTrue(str(caught.exception).startswith("Leiden clustering failed: "))


def load_without_numba():
    """A copy of Network_Kernels imported as it is when Numba is missing."""
    path = os.path.join(SRC_DIR, "utilities", "Network_Kernels.py")
    spec = importlib.util.spec_from_file_location("Network_Kernels_without_numba", path)
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def refuse_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba is blocked for this test")
        return real_import(name, *args, **kwargs)

    sys.modules[spec.name] = module
    try:
        with mock.patch("builtins.__import__", refuse_numba):
            spec.loader.exec_module(module)
    finally:
        # The copy registers itself under the module's other names too.
        for name in [name for name, loaded in sys.modules.items() if loaded is module]:
            del sys.modules[name]
    return module


class JaccardPartitionWithoutNumbaTests(unittest.TestCase):
    def test_the_pure_python_fallback_matches_the_numba_kernel(self):
        plain = load_without_numba()
        self.assertFalse(plain.NUMBA_AVAILABLE)
        for threshold in (0.0, 0.2, 0.26, 0.5, 1.0):
            for min_size in (1, 3):
                with self.subTest(threshold=threshold, min_size=min_size):
                    np.testing.assert_array_equal(
                        plain.jaccard_partition(7, BARBELL, threshold, min_size),
                        network_kernels.jaccard_partition(7, BARBELL, threshold, min_size),
                    )


if __name__ == "__main__":
    unittest.main()
