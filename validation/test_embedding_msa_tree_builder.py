"""Regression tests for complete-baseline replicate tree construction."""

import os
import multiprocessing as mp
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools import Embedding_MSA as msa


class TreeBuilderTests(unittest.TestCase):
    @staticmethod
    def _write_baseline(directory, values):
        path = os.path.join(directory, "baseline_dist.dat")
        baseline_map = np.memmap(
            path,
            dtype=np.float32,
            mode="w+",
            shape=values.shape,
        )
        baseline_map[:] = values
        baseline_map.flush()
        del baseline_map
        return path

    def test_worker_receives_every_observed_and_imputed_distance(self):
        baseline = np.array(
            [0.10, 0.25, 0.80, 0.35, 0.70, 0.45],
            dtype=np.float32,
        )
        captured = {}

        def capture_linkage(distances, method):
            captured["distances"] = np.array(distances, copy=True)
            captured["method"] = method
            return np.zeros((3, 4), dtype=np.float64)

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = self._write_baseline(directory, baseline)
            with mock.patch.object(msa.sch, "linkage", side_effect=capture_linkage):
                msa.compute_single_tree_worker(
                    seed=17,
                    num_seqs=4,
                    baseline_dist_path=baseline_path,
                    max_dist=1.0,
                    noise_scale=0.0,
                    tree_method="UPGMA (Fast)",
                )

        np.testing.assert_array_equal(captured["distances"], baseline)
        self.assertEqual(captured["method"], "average")

    def test_worker_adds_clipped_noise_to_the_complete_baseline(self):
        baseline = np.array(
            [0.05, 0.20, 0.40, 0.60, 0.80, 0.95],
            dtype=np.float32,
        )
        seed = 23
        max_dist = 1.0
        noise_scale = 0.10
        captured = {}

        def capture_linkage(distances, method):
            captured["distances"] = np.array(distances, copy=True)
            return np.zeros((3, 4), dtype=np.float64)

        with tempfile.TemporaryDirectory() as directory:
            baseline_path = self._write_baseline(directory, baseline)
            with mock.patch.object(msa.sch, "linkage", side_effect=capture_linkage):
                msa.compute_single_tree_worker(
                    seed=seed,
                    num_seqs=4,
                    baseline_dist_path=baseline_path,
                    max_dist=max_dist,
                    noise_scale=noise_scale,
                    tree_method="UPGMA (Fast)",
                )

        rng = np.random.default_rng(seed)
        expected_noise = rng.normal(
            0.0,
            noise_scale * max_dist,
            size=baseline.size,
        ).astype(np.float32)
        expected = np.clip(baseline + expected_noise, 0.0, max_dist)

        np.testing.assert_allclose(captured["distances"], expected, atol=1e-7)
        self.assertTrue(np.all(captured["distances"] >= 0.0))
        self.assertTrue(np.all(captured["distances"] <= max_dist))

    def test_worker_is_compatible_with_windows_spawn(self):
        baseline = np.array(
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60], dtype=np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = self._write_baseline(directory, baseline)
            context = mp.get_context("spawn")
            with context.Pool(processes=1) as pool:
                linkage = pool.apply(
                    msa.compute_single_tree_worker,
                    (
                        11,
                        4,
                        baseline_path,
                        1.0,
                        0.0,
                        "UPGMA (Fast)",
                    ),
                )

        self.assertEqual(linkage.shape, (3, 4))

    def test_full_cophenetic_matches_scipy(self):
        hand_built = np.array(
            [
                [0, 1, 0.20, 2],
                [2, 3, 0.40, 2],
                [4, 5, 0.90, 4],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            msa.compute_full_cophenetic(hand_built, 4),
            msa.sch.cophenet(hand_built).astype(np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

        rng = np.random.default_rng(41)
        for num_seqs in (2, 4, 8):
            points = rng.normal(size=(num_seqs, 3))
            linkage = msa.sch.linkage(points, method="average")
            expected = msa.sch.cophenet(linkage).astype(np.float32)

            actual = msa.compute_full_cophenetic(linkage, num_seqs)

            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_complete_network_always_uses_full_consensus(self):
        self.assertTrue(msa.use_full_cophenetic_consensus(False, False))
        self.assertTrue(msa.use_full_cophenetic_consensus(False, True))
        self.assertFalse(msa.use_full_cophenetic_consensus(True, False))
        self.assertTrue(msa.use_full_cophenetic_consensus(True, True))

    def test_partial_consensus_retains_imputed_baseline_values(self):
        baseline = np.array([0.20, 0.70, 0.80], dtype=np.float32)
        edge_i = np.array([0], dtype=np.int32)
        edge_j = np.array([1], dtype=np.int32)
        accumulated = np.array([0.60], dtype=np.float32)

        result = msa.finalize_cophenetic_consensus(
            baseline,
            num_seqs=3,
            edge_i=edge_i,
            edge_j=edge_j,
            cophenetic_accumulator=accumulated,
            num_trees=2,
            full_consensus=False,
        )

        np.testing.assert_allclose(result, [0.30, 0.70, 0.80])

    def test_full_consensus_replaces_every_pair(self):
        accumulated = np.array([0.60, 1.00, 1.40], dtype=np.float32)

        result = msa.finalize_cophenetic_consensus(
            accumulated,
            num_seqs=3,
            edge_i=None,
            edge_j=None,
            cophenetic_accumulator=None,
            num_trees=2,
            full_consensus=True,
        )

        np.testing.assert_allclose(result, [0.30, 0.50, 0.70])


if __name__ == "__main__":
    unittest.main()
