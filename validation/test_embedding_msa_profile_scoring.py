# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for occupancy-aware embedding profile scoring."""

import os
import sys
import unittest

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools import Embedding_MSA as msa


def legacy_leaf_score(emb_i, emb_j):
    """Previous score calculation for unit-confidence leaf vectors."""
    t_i = torch.as_tensor(emb_i, dtype=torch.float32)
    t_j = torch.as_tensor(emb_j, dtype=torch.float32)
    t_i = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    sim_mat = torch.exp(-(1.0 - torch.mm(t_i, t_j.T)))

    epsilon = 1e-8
    z_r = (sim_mat - sim_mat.mean(dim=1, keepdim=True)) / (
        sim_mat.std(dim=1, keepdim=True) + epsilon
    )
    z_c = (sim_mat - sim_mat.mean(dim=0, keepdim=True)) / (
        sim_mat.std(dim=0, keepdim=True) + epsilon
    )
    return ((z_r + z_c) / 2.0).numpy()


class ProfileScoringTests(unittest.TestCase):
    def test_leaf_normalization_preserves_zero_vectors(self):
        raw = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

        normalized = msa._normalize_residue_embeddings(raw)

        np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=5e-4)
        np.testing.assert_array_equal(normalized[1], [0.0, 0.0])

    def test_profile_support_scales_scores(self):
        profile = np.eye(3, dtype=np.float32)
        target = msa._normalize_residue_embeddings(
            np.array(
                [
                    [1.0, 0.2, 0.0],
                    [0.1, 1.0, 0.2],
                    [0.0, 0.1, 1.0],
                    [0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
        )
        sparse_profile = profile.copy()
        sparse_profile[0] *= 0.01

        full_scores = msa.compute_score_matrix_torch(profile, target)
        sparse_scores = msa.compute_score_matrix_torch(sparse_profile, target)

        np.testing.assert_allclose(
            sparse_scores[0], full_scores[0] * 0.01, rtol=2e-4, atol=2e-5
        )
        np.testing.assert_allclose(
            sparse_scores[1:], full_scores[1:], rtol=2e-4, atol=2e-5
        )

    def test_disagreement_reduces_merged_column_confidence(self):
        cluster_a = msa.MSACluster(0, ["A"], [0], np.array([[1.0, 0.0]]))
        cluster_b = msa.MSACluster(1, ["B"], [1], np.array([[0.0, 1.0]]))

        merged = msa.merge_clusters(
            cluster_a,
            cluster_b,
            np.array([1], dtype=np.int8),
            cluster_a.embedding,
            cluster_b.embedding,
        )

        np.testing.assert_allclose(merged.embedding[0], [0.5, 0.5], atol=5e-4)
        self.assertAlmostEqual(
            float(np.linalg.norm(merged.embedding[0])), np.sqrt(0.5), places=3
        )

    def test_full_support_leaf_scores_match_previous_calculation(self):
        raw_i = np.array(
            [[3.0, 4.0, 0.0], [1.0, 0.0, 2.0], [0.2, 1.0, 0.5]],
            dtype=np.float32,
        )
        raw_j = np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
                [1.0, 1.0, 1.0],
                [0.5, 2.0, 0.2],
            ],
            dtype=np.float32,
        )
        leaf_i = msa._normalize_residue_embeddings(raw_i)
        leaf_j = msa._normalize_residue_embeddings(raw_j)

        expected = legacy_leaf_score(leaf_i, leaf_j)
        actual = msa.compute_score_matrix_torch(leaf_i, leaf_j)

        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
