"""Batched alignment kernels must reproduce the per-pair DP bit for bit.

The tiled pipeline aligns a whole padded microbatch per CPU task through
``align_microbatch``. Gap penalties for which float32 is exact run a faster
float32 kernel; every other gap runs the reference kernel. All comparisons
below are against ``global_local_scores`` on the unpadded matrix.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for directory in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from utilities import Network_Kernels as kernels  # noqa: E402

# Point each tool at a missing settings file so the developer's
# tools_settings.json cannot change module globals for later test modules.
MISSING_SETTINGS = os.path.join(PROJECT_ROOT, "tests", "nonexistent-settings.json")
with mock.patch.dict(os.environ, {
    "SSN_TOOL_SETTINGS_SCRIPT": "Align_Similarity_Matrix.py",
    "SSN_TOOL_SETTINGS_FILE": MISSING_SETTINGS,
}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
with mock.patch.dict(os.environ, {
    "SSN_TOOL_SETTINGS_SCRIPT": "Network_Injection.py",
    "SSN_TOOL_SETTINGS_FILE": MISSING_SETTINGS,
}), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Network_Injection as network_injection


EXACT_GAPS = (
    (0.0, -2.0),
    (0.0, -1.0),
    (0.0, -0.5),
    (0.0, -4.0),
    (-0.0, -2.0),
    (0.0, 0.0),
    (np.float32(0.0), np.float32(-2.0)),
)
REFERENCE_GAPS = (
    (-1.0, -2.0),
    (0.0, -0.7),
    (-0.7, -0.3),
    (np.float32(-0.7), np.float32(-0.3)),
    (-2.5, -0.1),
    (0.0, -3.0),
)


def padded_batches(seed, count):
    """Padded microbatches: exact ties, tiny values, and wider scores.

    Padding columns hold finite noise that the kernels must ignore.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for index in range(count):
        rows = int(rng.integers(1, 30))
        lengths = rng.integers(1, 35, size=int(rng.integers(1, 40)))
        shape = (len(lengths), rows, int(lengths.max()))
        kind = index % 3
        if kind == 0:
            matrices = rng.integers(-3, 4, size=shape).astype(np.float32)
        elif kind == 1:
            # Local values below 1 make value + gap inexact in float32.
            matrices = (rng.standard_normal(shape) * 1e-3).astype(np.float32)
        else:
            matrices = (rng.standard_normal(shape) * 2.0).astype(np.float32)
        for pair, columns in enumerate(lengths):
            matrices[pair, :, columns:] = rng.standard_normal(
                (rows, shape[2] - columns)
            )
        batches.append((matrices, lengths.astype(np.int64)))
    return batches


def reference_scores(matrices, lengths, global_gap, local_gap):
    results = [
        kernels.global_local_scores(
            np.ascontiguousarray(matrices[pair, :, :columns]),
            global_gap,
            local_gap,
        )
        for pair, columns in enumerate(lengths)
    ]
    return (
        np.array([result[0] for result in results], dtype=np.float32),
        np.array([result[1] for result in results], dtype=np.uint32),
        np.array([result[2] for result in results], dtype=np.float64),
        np.array([result[3] for result in results], dtype=np.uint32),
    )


class BatchKernelIdentityTests(unittest.TestCase):
    def assert_bitwise_equal(self, expected, actual, context):
        names = ("global score", "global length", "local score", "local length")
        for name, want, got in zip(names, expected, actual):
            self.assertEqual(want.dtype, got.dtype, f"{context}: {name} dtype")
            self.assertTrue(
                np.array_equal(want.view(np.uint8), got.view(np.uint8)),
                f"{context}: {name} differs",
            )

    def test_reference_batch_kernel_matches_per_pair_kernel(self):
        for global_gap, local_gap in EXACT_GAPS + REFERENCE_GAPS:
            for number, (matrices, lengths) in enumerate(padded_batches(11, 30)):
                batch = len(lengths)
                actual = (
                    np.empty(batch, np.float32),
                    np.empty(batch, np.uint32),
                    np.empty(batch, np.float64),
                    np.empty(batch, np.uint32),
                )
                kernels.global_local_scores_batch(
                    matrices, lengths, global_gap, local_gap, *actual
                )
                self.assert_bitwise_equal(
                    reference_scores(matrices, lengths, global_gap, local_gap),
                    actual,
                    f"gaps ({global_gap!r}, {local_gap!r}) batch {number}",
                )

    def test_float32_kernel_matches_reference_for_exact_gaps(self):
        batches = padded_batches(23, 90)
        for global_gap, local_gap in EXACT_GAPS:
            with mock.patch.object(
                kernels,
                "global_local_scores_batch",
                side_effect=AssertionError("reference kernel used"),
            ):
                for number, (matrices, lengths) in enumerate(batches):
                    self.assert_bitwise_equal(
                        reference_scores(matrices, lengths, global_gap, local_gap),
                        kernels.alignment_batch_scores(
                            matrices, lengths, global_gap, local_gap
                        ),
                        f"gaps ({global_gap!r}, {local_gap!r}) batch {number}",
                    )

    def test_other_gaps_never_use_the_float32_kernel(self):
        batches = padded_batches(31, 30)
        for global_gap, local_gap in REFERENCE_GAPS:
            with mock.patch.object(
                kernels,
                "_float32_global_local_scores_batch",
                side_effect=AssertionError("float32 kernel used"),
            ):
                for number, (matrices, lengths) in enumerate(batches):
                    self.assert_bitwise_equal(
                        reference_scores(matrices, lengths, global_gap, local_gap),
                        kernels.alignment_batch_scores(
                            matrices, lengths, global_gap, local_gap
                        ),
                        f"gaps ({global_gap!r}, {local_gap!r}) batch {number}",
                    )

    def test_float32_bound_accepts_only_provably_exact_gaps(self):
        self.assertEqual(kernels.float32_exact_local_bound(0.0, -2.0), 2.0 ** 23)
        self.assertEqual(kernels.float32_exact_local_bound(-0.0, -1.0), 2.0 ** 22)
        self.assertEqual(
            kernels.float32_exact_local_bound(np.float32(0.0), np.float32(-0.5)),
            2.0 ** 21,
        )
        self.assertEqual(
            kernels.float32_exact_local_bound(0.0, 0.0),
            float(np.finfo(np.float32).max),
        )
        for global_gap, local_gap in REFERENCE_GAPS + ((0.0, 2.0), (1.0, 0.0)):
            self.assertIsNone(
                kernels.float32_exact_local_bound(global_gap, local_gap),
                (global_gap, local_gap),
            )

    def test_pairs_above_the_exact_bound_fall_back_to_reference(self):
        original = kernels.float32_exact_local_bound

        def low_bound(global_gap, local_gap):
            return None if original(global_gap, local_gap) is None else 1.0

        fallbacks = 0
        with mock.patch.object(kernels, "float32_exact_local_bound", low_bound):
            for global_gap, local_gap in ((0.0, -2.0), (0.0, 0.0)):
                for matrices, lengths in padded_batches(47, 45):
                    expected = reference_scores(
                        matrices, lengths, global_gap, local_gap
                    )
                    fallbacks += int(np.count_nonzero(expected[2] >= 1.0))
                    self.assert_bitwise_equal(
                        expected,
                        kernels.alignment_batch_scores(
                            matrices, lengths, global_gap, local_gap
                        ),
                        f"gaps ({global_gap!r}, {local_gap!r}) forced fallback",
                    )
        self.assertGreater(fallbacks, 100)

    def test_fallback_recomputes_with_the_reference_gaps(self):
        # Within the proven range both kernels agree, so give the fallback
        # different gaps to see that every pair at the bound is recomputed.
        for matrices, lengths in padded_batches(53, 12):
            batch = len(lengths)
            actual = (
                np.empty(batch, np.float32),
                np.empty(batch, np.uint32),
                np.empty(batch, np.float64),
                np.empty(batch, np.uint32),
            )
            kernels._float32_global_local_scores_batch(
                matrices,
                lengths,
                np.float32(0.0),
                np.float32(-2.0),
                np.float32(0.0),
                -1.0,
                -0.7,
                *actual,
            )
            self.assert_bitwise_equal(
                reference_scores(matrices, lengths, -1.0, -0.7),
                actual,
                "fallback with reference gaps",
            )

    def test_align_microbatch_rejects_non_finite_scores(self):
        matrices, lengths = padded_batches(5, 1)[0]
        for bad in (np.nan, np.inf):
            broken = matrices.copy()
            broken[0, 0, 0] = bad
            with self.assertRaises(FloatingPointError):
                kernels.align_microbatch(
                    broken, 0, list(range(len(lengths))), lengths, 0.0, -2.0
                )


def residue_codes(rng, rows, batch, columns):
    """Query codes and padded target codes, with the unmatched codes 0 and 1 among the residues."""
    query = rng.integers(0, 6, size=rows).astype(np.uint8)
    targets = rng.integers(0, 6, size=(batch, columns)).astype(np.uint8)
    return query, targets


def reference_search(mode, matrices, lengths, gap, query, targets):
    kernel = (
        kernels.local_score_length_identity if mode == "local"
        else kernels.global_score_length_identity
    )
    results = [
        kernel(
            np.ascontiguousarray(matrices[pair, :, :columns]),
            gap,
            query,
            np.ascontiguousarray(targets[pair, :columns]),
        )
        for pair, columns in enumerate(lengths)
    ]
    return (
        [result[0] for result in results],
        [int(result[1]) for result in results],
        [int(result[2]) for result in results],
    )


class SearchKernelTests(unittest.TestCase):
    """The database search's kernels: score, path length and identities, bit for bit."""

    def batches(self, seed=7, count=12):
        rng = np.random.default_rng(seed)
        for matrices, lengths in padded_batches(seed, count):
            query, targets = residue_codes(rng, matrices.shape[1], len(lengths), matrices.shape[2])
            yield matrices, lengths, query, targets
        # Dense, positive-leaning matrices like real alignments: long paths, many identities.
        for _ in range(3):
            shape = (int(rng.integers(2, 9)), int(rng.integers(5, 60)), int(rng.integers(5, 60)))
            matrices = (rng.random(shape) * 3.0).astype(np.float32)
            lengths = rng.integers(1, shape[2] + 1, size=shape[0]).astype(np.int64)
            query, targets = residue_codes(rng, shape[1], shape[0], shape[2])
            yield matrices, lengths, query, targets

    def assert_search_equals_reference(self, mode, gap):
        for matrices, lengths, query, targets in self.batches():
            scores, path_lengths, identities = kernels.search_batch_scores(
                mode, matrices, lengths, gap, query, targets
            )
            expected = reference_search(mode, matrices, lengths, gap, query, targets)
            self.assertEqual(scores.dtype, np.float64 if mode == "local" else np.float32)
            self.assertEqual(path_lengths.dtype, np.uint32)
            self.assertEqual(identities.dtype, np.uint32)
            # Equal as Python numbers, which is what the result rows hold.
            self.assertEqual(scores.tolist(), expected[0], (mode, gap))
            self.assertEqual(path_lengths.tolist(), expected[1], (mode, gap))
            self.assertEqual(identities.tolist(), expected[2], (mode, gap))

    def test_global_float32_kernel_matches_the_reference_for_a_zero_gap(self):
        for gap in (0.0, -0.0, np.float32(0.0), 0):
            with self.subTest(gap=gap), mock.patch.object(
                kernels,
                "_float32_global_length_identity_batch",
                wraps=kernels._float32_global_length_identity_batch,
            ) as fast, mock.patch.object(
                kernels,
                "_reference_global_length_identity_batch",
                side_effect=AssertionError("the reference kernel ran"),
            ):
                self.assert_search_equals_reference("global", gap)
            self.assertTrue(fast.called)

    def test_any_other_global_gap_runs_the_reference_kernel(self):
        for gap in (-1.0, -0.7, -2.5, 0.1):
            with self.subTest(gap=gap), mock.patch.object(
                kernels,
                "_float32_global_length_identity_batch",
                side_effect=AssertionError("the float32 kernel ran"),
            ):
                self.assert_search_equals_reference("global", gap)

    def test_local_float32_kernel_matches_the_reference_for_exact_gaps(self):
        for gap in (-2.0, 0.0, -1.0, -0.5, -4.0, -0.25):
            with self.subTest(gap=gap), mock.patch.object(
                kernels,
                "_float32_local_length_identity_batch",
                wraps=kernels._float32_local_length_identity_batch,
            ) as fast, mock.patch.object(
                kernels,
                "_reference_local_length_identity_batch",
                side_effect=AssertionError("the reference kernel ran"),
            ):
                self.assert_search_equals_reference("local", gap)
            self.assertTrue(fast.called)

    def test_any_other_local_gap_runs_the_reference_kernel(self):
        for gap in (-0.7, -3.0, -2.5, 0.5, -1.5):
            with self.subTest(gap=gap), mock.patch.object(
                kernels,
                "_float32_local_length_identity_batch",
                side_effect=AssertionError("the float32 kernel ran"),
            ):
                self.assert_search_equals_reference("local", gap)

    def test_a_pair_reaching_the_score_bound_is_run_again_with_the_reference(self):
        # With a gap of -2 float32 is proven exact below 2**23. This pair scores 8e7: there a float32
        # candidate that the reference keeps unrounded can tie, and the path's identities differ.
        bound = kernels.float32_exact_local_bound(0.0, -2.0)
        self.assertEqual(bound, 2.0 ** 23)
        rng = np.random.default_rng(204)
        rows, columns = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        matrix = (
            np.round(rng.random((rows, columns)) * 3) * 1.6e7 / 3 + rng.integers(0, 9, size=(rows, columns))
        ).astype(np.float32)
        query = rng.integers(0, 4, size=rows).astype(np.uint8)
        target = rng.integers(0, 4, size=columns).astype(np.uint8)
        want = kernels.local_score_length_identity(matrix, -2.0, query, target)
        scratch = [np.empty(columns + 1, dtype) for dtype in (np.float32,) * 2 + (np.uint32,) * 4]
        fast = kernels._float32_local_length_identity(
            matrix, np.float32(-2.0), np.float32(2.0), query, target, *scratch
        )
        self.assertGreaterEqual(fast[0], bound)
        self.assertNotEqual((float(fast[0]), int(fast[1]), int(fast[2])), (want[0], int(want[1]), int(want[2])))
        # A batch holding it and a small pair: both equal the reference.
        small = (rng.standard_normal((rows, columns)) * 2).astype(np.float32)
        matrices = np.stack([matrix, small])
        targets = np.stack([target, target])
        scores, path_lengths, identities = kernels.search_batch_scores(
            "local", matrices, [columns, columns], -2.0, query, targets
        )
        expected = reference_search("local", matrices, [columns, columns], -2.0, query, targets)
        self.assertEqual(scores.tolist(), expected[0])
        self.assertEqual(path_lengths.tolist(), expected[1])
        self.assertEqual(identities.tolist(), expected[2])

    def test_which_global_gaps_float32_is_exact_for(self):
        for gap in (0.0, -0.0, np.float32(0.0), 0):
            self.assertTrue(kernels.float32_exact_global_gap(gap), gap)
        for gap in (-1.0, 0.5, -1e-30, np.float32(-0.25)):
            self.assertFalse(kernels.float32_exact_global_gap(gap), gap)

    def test_padding_columns_and_codes_are_ignored(self):
        rng = np.random.default_rng(3)
        matrices = (rng.standard_normal((2, 6, 9)) * 2).astype(np.float32)
        lengths = np.array([9, 4])
        query, targets = residue_codes(rng, 6, 2, 9)
        for mode, gap in (("global", 0.0), ("local", -2.0)):
            first = kernels.search_batch_scores(mode, matrices, lengths, gap, query, targets)
            noisy = matrices.copy()
            noisy[1, :, 4:] = 1e6
            other = targets.copy()
            other[1, 4:] = 5
            second = kernels.search_batch_scores(mode, noisy, lengths, gap, query, other)
            for want, got in zip(first, second):
                self.assertEqual(want.tolist(), got.tolist())

    def test_one_pair_gives_the_python_numbers_the_reference_gives(self):
        rng = np.random.default_rng(11)
        matrix = (rng.standard_normal((20, 31)) * 2).astype(np.float32)
        query, targets = residue_codes(rng, 20, 1, 31)
        for mode, gap in (("local", -2.0), ("global", 0.0), ("global", -0.5)):
            kernel = (
                kernels.local_score_length_identity if mode == "local"
                else kernels.global_score_length_identity
            )
            want = kernel(matrix, gap, query, targets[0])
            got = kernels.search_pair_scores(mode, matrix, gap, query, targets[0])
            self.assertEqual(tuple(got), tuple(want))
            self.assertEqual([type(value) for value in got], [type(value) for value in want])

    def test_an_unknown_mode_is_refused(self):
        matrices = np.zeros((1, 2, 2), np.float32)
        codes = np.zeros(2, np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown alignment mode"):
            kernels.search_batch_scores("semi", matrices, [2], 0.0, codes, codes[np.newaxis])


class GlobalLocalPairScoresTests(unittest.TestCase):
    """The per-pair path (scalar accelerator and CPU pool) gives the reference kernel's numbers exactly."""

    def matrices(self):
        for batch, lengths in padded_batches(23, 15):
            for pair, columns in enumerate(lengths):
                yield np.ascontiguousarray(batch[pair, :, :columns])
        rng = np.random.default_rng(31)
        for _ in range(6):  # dense and positive-leaning, like real alignments
            rows, columns = int(rng.integers(2, 60)), int(rng.integers(2, 60))
            yield (rng.random((rows, columns)) * 3.0).astype(np.float32)

    def assert_same(self, matrix, global_gap, local_gap):
        want = kernels.global_local_scores(matrix, global_gap, local_gap)
        got = kernels.global_local_pair_scores(matrix, global_gap, local_gap)
        self.assertEqual(got, want, (global_gap, local_gap))
        self.assertEqual([type(value) for value in got], [type(value) for value in want])

    def test_every_gap_gives_the_reference_numbers_and_python_types(self):
        for global_gap, local_gap in EXACT_GAPS + REFERENCE_GAPS:
            with self.subTest(gaps=(global_gap, local_gap)):
                for matrix in self.matrices():
                    self.assert_same(matrix, global_gap, local_gap)

    def test_float32_kernels_run_only_for_exact_gaps(self):
        matrix = next(iter(self.matrices()))
        with mock.patch.object(
            kernels, "_float32_global_local_scores_batch",
            wraps=kernels._float32_global_local_scores_batch,
        ) as fast, mock.patch.object(
            kernels, "global_local_scores_batch",
            side_effect=AssertionError("the reference batch kernel ran"),
        ):
            kernels.global_local_pair_scores(matrix, 0.0, -2.0)
        self.assertEqual(fast.call_count, 1)
        with mock.patch.object(
            kernels, "_float32_global_local_scores_batch",
            side_effect=AssertionError("the float32 kernel ran"),
        ):
            for gaps in ((-1.0, -2.0), (0.0, -0.7), (-0.5, -0.5)):
                self.assert_same(matrix, *gaps)

    def test_a_matrix_that_is_not_float32_runs_the_reference_kernel(self):
        matrix = next(iter(self.matrices())).astype(np.float64)
        with mock.patch.object(
            kernels, "alignment_batch_scores", side_effect=AssertionError("the batch dispatcher ran")
        ):
            got = kernels.global_local_pair_scores(matrix, 0.0, -2.0)
        self.assertEqual(got, kernels.global_local_scores(matrix, 0.0, -2.0))

    def test_a_matrix_that_is_a_view_of_a_padded_batch_is_scored_like_its_copy(self):
        batch = padded_batches(5, 1)[0][0]
        view = batch[0, :, : max(1, batch.shape[2] - 1)]
        self.assertEqual(
            kernels.global_local_pair_scores(view, 0.0, -2.0),
            kernels.global_local_scores(np.ascontiguousarray(view), 0.0, -2.0),
        )

    def test_a_score_past_the_float32_bound_is_run_again_with_the_reference(self):
        rng = np.random.default_rng(204)
        rows, columns = int(rng.integers(3, 12)), int(rng.integers(3, 12))
        matrix = (
            np.round(rng.random((rows, columns)) * 3) * 1.6e7 / 3 + rng.integers(0, 9, size=(rows, columns))
        ).astype(np.float32)
        self.assert_same(matrix, 0.0, -2.0)
        small = (rng.standard_normal((rows, columns)) * 2).astype(np.float32)
        self.assert_same(small, 0.0, -2.0)


class ToolBatchCallbackTests(unittest.TestCase):
    def test_tool_batch_callbacks_match_their_per_pair_callbacks(self):
        batches = padded_batches(59, 24)
        for tool in (similarity_matrix, network_injection):
            for global_gap, local_gap in ((0.0, -2.0), (-1.0, -0.7)):
                with self.subTest(tool=tool.__name__, gaps=(global_gap, local_gap)), \
                        mock.patch.object(tool, "GLOBAL_GAP_P", global_gap), \
                        mock.patch.object(tool, "LOCAL_GAP_P", local_gap):
                    for matrices, lengths in batches:
                        # Production matrices are zero-padded on the device.
                        padded = matrices.copy()
                        for pair, columns in enumerate(lengths):
                            padded[pair, :, columns:] = 0.0
                        targets = [7 + pair for pair in range(len(lengths))]
                        batched = tool.calculate_alignment_batch(
                            (3, targets, [int(v) for v in lengths], padded)
                        )
                        per_pair = [
                            tool.calculate_alignment_data(
                                (3, target, padded[pair, :, :columns])
                            )
                            for pair, (target, columns) in enumerate(
                                zip(targets, lengths)
                            )
                        ]
                        self.assertEqual(len(batched), len(per_pair))
                        for got, want in zip(batched, per_pair):
                            self.assertEqual(got[:2], (want[0], want[1]))
                            self.assertEqual(got[3], int(want[3]))
                            self.assertEqual(got[5], int(want[5]))
                            for index in (2, 4):
                                self.assertEqual(
                                    np.float64(got[index]).tobytes(),
                                    np.float64(want[index]).tobytes(),
                                )

    def test_per_pair_callbacks_give_the_reference_kernels_numbers(self):
        # The scalar accelerator and the CPU pool call calculate_alignment_data once per pair.
        rng = np.random.default_rng(3)
        matrices = [
            (rng.integers(-3, 4, size=(9, 14)) * 0.5).astype(np.float32),
            (rng.random((20, 17)) * 3.0).astype(np.float32),
            (rng.standard_normal((6, 30)) * 1e-3).astype(np.float32),
        ]
        for tool in (similarity_matrix, network_injection):
            for global_gap, local_gap in ((0.0, -2.0), (0.0, 0.0), (0.0, -4.0), (-1.0, -0.7), (-0.3, -2.0)):
                with self.subTest(tool=tool.__name__, gaps=(global_gap, local_gap)),                         mock.patch.object(tool, "GLOBAL_GAP_P", global_gap),                         mock.patch.object(tool, "LOCAL_GAP_P", local_gap):
                    for matrix in matrices:
                        g_raw, g_len, l_raw, l_len = kernels.global_local_scores(matrix, global_gap, local_gap)
                        got = tool.calculate_alignment_data((4, 9, matrix))
                        self.assertEqual(got, (4, 9, l_raw, l_len, g_raw, g_len))
                        self.assertEqual(
                            [type(value) for value in got], [type(value) for value in (4, 9, l_raw, l_len, g_raw, g_len)]
                        )

    def test_per_pair_callbacks_use_the_float32_kernel_where_it_is_exact(self):
        matrix = (np.random.default_rng(8).random((12, 15)) * 3.0).astype(np.float32)
        for tool in (similarity_matrix, network_injection):
            with self.subTest(tool=tool.__name__),                     mock.patch.object(tool, "GLOBAL_GAP_P", 0.0), mock.patch.object(tool, "LOCAL_GAP_P", -2.0),                     mock.patch.object(
                        kernels, "_float32_global_local_scores_batch",
                        wraps=kernels._float32_global_local_scores_batch,
                    ) as fast:
                tool.calculate_alignment_data((0, 1, matrix))
            self.assertEqual(fast.call_count, 1)


if __name__ == "__main__":
    unittest.main()
