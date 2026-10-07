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


if __name__ == "__main__":
    unittest.main()
