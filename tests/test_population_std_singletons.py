"""Regression tests for population Z-scores and singleton sequences."""

import io
import pathlib
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TOOLS_DIR = SRC_DIR / "tools"
UTILITIES_DIR = SRC_DIR / "utilities"
for module_dir in (SRC_DIR, TOOLS_DIR, UTILITIES_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Cache_Manifest
    import Embedding_MSA as embedding_msa
    import Embedding_PWA as embedding_pwa
    import Embedding_SSEARCH as embedding_ssearch
    with mock.patch.object(
        Cache_Manifest,
        "validate_network_schema",
        return_value=SimpleNamespace(model_name="test_model"),
    ):
        import Network_Injection as network_injection


class PopulationStandardDeviationTests(unittest.TestCase):
    def setUp(self):
        self.singleton = np.array(
            [[1.0, 2.0, 3.0, 4.0]],
            dtype=np.float32,
        )
        self.multiple = np.array(
            [
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def assert_supports_singletons(self, score_builder):
        for emb_i, emb_j in (
            (self.singleton, self.multiple),
            (self.multiple, self.singleton),
            (self.singleton, self.singleton),
        ):
            score_matrix = score_builder(emb_i, emb_j)
            self.assertTrue(np.isfinite(score_matrix).all())

        np.testing.assert_array_equal(
            score_builder(self.singleton, self.singleton),
            np.zeros((1, 1), dtype=np.float32),
        )

    def test_pairwise_score_builders_support_singletons(self):
        builders = (
            lambda emb_i, emb_j: similarity_matrix.compute_score_matrix_torch(
                emb_i,
                emb_j,
                "cpu",
            ),
            lambda emb_i, emb_j: network_injection.compute_score_matrix_torch(
                emb_i,
                emb_j,
                "cpu",
            ),
            lambda emb_i, emb_j: embedding_ssearch.compute_score_matrix_torch(
                emb_i,
                emb_j,
                "cpu",
            ),
            lambda emb_i, emb_j: embedding_pwa.compute_score_matrix_torch(
                emb_i,
                emb_j,
                "cpu",
            ),
        )
        for builder in builders:
            self.assert_supports_singletons(builder)

    def test_msa_score_builder_supports_singletons(self):
        self.assert_supports_singletons(
            lambda emb_i, emb_j: embedding_msa.compute_score_matrix_torch(
                emb_i,
                emb_j,
                "cpu",
            )
        )

    def test_primary_score_builder_uses_population_standard_deviation(self):
        t_i = torch.nn.functional.normalize(
            torch.as_tensor(self.multiple),
            dim=-1,
        )
        t_j = torch.nn.functional.normalize(
            torch.as_tensor(self.multiple[::-1].copy()),
            dim=-1,
        )
        cosine = torch.mm(t_i, t_j.T).clamp(-1.0, 1.0)
        similarity = torch.exp(-(1.0 - cosine))
        row_mean = similarity.mean(dim=1, keepdim=True)
        row_std = similarity.std(dim=1, keepdim=True, correction=0)
        col_mean = similarity.mean(dim=0, keepdim=True)
        col_std = similarity.std(dim=0, keepdim=True, correction=0)
        expected = (
            (similarity - row_mean) / (row_std + 1e-8)
            + (similarity - col_mean) / (col_std + 1e-8)
        ) / 2.0

        actual = similarity_matrix.compute_score_matrix_torch(
            self.multiple,
            self.multiple[::-1].copy(),
            "cpu",
        )
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6)

    def test_primary_score_builder_clamps_cosine_bounds(self):
        forced_cosine = torch.tensor(
            [[1.25, -1.25]],
            dtype=torch.float32,
        )
        with mock.patch.object(torch, "mm", return_value=forced_cosine):
            actual = similarity_matrix.compute_score_matrix_torch(
                np.ones((1, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                "cpu",
            )

        bounded_similarity = torch.exp(
            -(1.0 - forced_cosine.clamp(-1.0, 1.0))
        )
        row_mean = bounded_similarity.mean(dim=1, keepdim=True)
        row_std = bounded_similarity.std(
            dim=1,
            keepdim=True,
            correction=0,
        )
        expected = (bounded_similarity - row_mean) / (row_std + 1e-8)
        expected /= 2.0
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6)

    def test_pairwise_score_builders_match_primary_float32_scores(self):
        emb_i = self.multiple
        emb_j = self.multiple[::-1].copy()
        expected = similarity_matrix.compute_score_matrix_torch(
            emb_i,
            emb_j,
            "cpu",
        )

        for score_builder in (
            network_injection.compute_score_matrix_torch,
            embedding_ssearch.compute_score_matrix_torch,
            embedding_pwa.compute_score_matrix_torch,
        ):
            actual = score_builder(emb_i, emb_j, "cpu")
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_pairwise_dynamic_programming_uses_float32_scores(self):
        score_matrix = similarity_matrix.compute_score_matrix_torch(
            self.multiple,
            self.multiple[::-1].copy(),
            "cpu",
        )
        expected = similarity_matrix.global_local_scores(
            score_matrix,
            0.0,
            -2.0,
        )
        pairwise_global = embedding_pwa.needleman_wunsch_custom(
            score_matrix,
            0.0,
        )
        pairwise_local = embedding_pwa.smith_waterman_custom(
            score_matrix,
            -2.0,
        )

        self.assertEqual(np.asarray(pairwise_global[2]).dtype, np.float32)
        self.assertEqual(np.asarray(pairwise_local[2]).dtype, np.float32)
        self.assertAlmostEqual(
            float(pairwise_global[2]),
            float(expected[0]),
            places=5,
        )
        self.assertAlmostEqual(
            float(pairwise_local[2]),
            float(expected[2]),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
