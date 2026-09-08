# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

import pathlib
import sys
import unittest
from unittest import mock
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Layout_Engine_UMAP


class TestLayoutEngineUMAP(unittest.TestCase):
    def capture_layout(self, edges, n, k, live=False):
        captured = []
        original = Layout_Engine_UMAP.umap.UMAP

        def construct(**kwargs):
            captured.append(kwargs)
            if live:
                reducer = original(**kwargs)
                captured.append(reducer)
                return reducer
            return mock.Mock(fit_transform=lambda x: np.column_stack((np.arange(n), np.arange(n))).astype(np.float32))

        with mock.patch.object(Layout_Engine_UMAP.umap, 'UMAP', side_effect=construct):
            Layout_Engine_UMAP.calculate_layout(np.array(edges, dtype=np.float32), n, {'UMAP_NEIGHBORS': k})
        return captured

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, 'umap-learn is required')
    def test_k_excludes_self_and_preserves_zero_distance_matches(self):
        edges = [(i, j, 100) for i in range(20) for j in range(i + 1, 20)]
        args = self.capture_layout(edges, 20, 15)[0]
        indices, distances, _ = args['precomputed_knn']
        self.assertEqual(args['n_neighbors'], 16)
        self.assertEqual(indices.shape, (20, 16))
        np.testing.assert_array_equal(indices[:, 0], np.arange(20))
        np.testing.assert_array_equal(distances, 0)
        for i, row in enumerate(indices):
            self.assertEqual(len(set(row)), 16)
            self.assertNotIn(i, row[1:])

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, 'umap-learn is required')
    def test_duplicate_edges_self_loops_and_missing_neighbors(self):
        args = self.capture_layout([(0, 0, 100), (0, 1, 100), (1, 0, 90), (0, 2, 80)], 4, 3)[0]
        indices, distances, _ = args['precomputed_knn']
        np.testing.assert_array_equal(indices, [[0, 1, 2, -1], [1, 0, -1, -1], [2, 0, -1, -1], [3, -1, -1, -1]])
        np.testing.assert_array_equal(distances, [[0, 0, 20, np.inf], [0, 0, np.inf, np.inf], [0, 20, np.inf, np.inf], [0, np.inf, np.inf, np.inf]])

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, 'umap-learn is required')
    def test_live_small_graphs_retain_all_real_neighbors(self):
        for n in (2, 3, 5):
            with self.subTest(n=n):
                edges = [(i, j, 100 - abs(i-j)) for i in range(n) for j in range(i+1, n)]
                args, reducer = self.capture_layout(edges, n, 15, live=True)
                self.assertEqual(args['n_neighbors'], n)
                self.assertEqual(reducer._knn_indices.shape, (n, n))
                self.assertEqual(reducer.graph_.nnz, n*(n-1))
                np.testing.assert_array_equal(reducer.graph_.diagonal(), 0)
                self.assertTrue(np.isfinite(reducer.embedding_).all())

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, 'umap-learn is required')
    def test_graph_matches_independent_neighbor_table(self):
        from umap import UMAP
        n, k = 8, 3
        edges = [(i, j, 100 - abs(i-j)) for i in range(n) for j in range(i+1, n)]
        _, actual = self.capture_layout(edges, n, k, live=True)
        indices = np.array([[i] + sorted((j for j in range(n) if j != i), key=lambda j: (abs(i-j), j))[:k] for i in range(n)], dtype=np.int32)
        distances = np.array([[0] + [abs(i-j)-1 for j in row[1:]] for i, row in enumerate(indices)], dtype=np.float32)
        expected = UMAP(n_neighbors=k+1, precomputed_knn=(indices, distances, None), random_state=42)
        expected.fit(np.zeros((n, 1), dtype=np.float32))
        np.testing.assert_allclose(actual.graph_.toarray(), expected.graph_.toarray())
        np.testing.assert_array_equal(actual.graph_.diagonal(), 0)

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, 'umap-learn is required')
    def test_calibration_includes_all_real_neighbors(self):
        from umap.umap_ import smooth_knn_dist, compute_membership_strengths
        args = self.capture_layout([(0, 0, 100), (0, 1, 99), (0, 2, 98), (0, 3, 97), (0, 4, 96)], 5, 4)[0]
        indices, distances, _ = args['precomputed_knn']
        sigma, rho = smooth_knn_dist(distances[:1].copy(), 5.0)
        _, _, weights, _ = compute_membership_strengths(indices[:1].copy(), distances[:1].copy(), sigma, rho)
        np.testing.assert_allclose(weights, [0, 1, .642896, .413315, .265719], atol=1e-5)

    def test_empty_connectivity_generates_random_layout(self):
        connectivity = np.zeros((0, 3), dtype=np.float32)
        params = {"BOX_SCALE": 1.5}
        pos, box_limit = Layout_Engine_UMAP.calculate_layout(
            connectivity, n_nodes=10, params=params
        )
        self.assertEqual(pos.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(pos)))
        expected_limit = (np.sqrt(10) * 2.5 + 5.0) * 1.5
        self.assertAlmostEqual(box_limit, expected_limit, places=5)
        self.assertTrue(np.all(np.abs(pos) <= box_limit / 2.0))

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_identical_scores_preserve_closest_homologues(self):
        # 4 nodes, fully connected, all sharing maximal similarity score
        # Previously, max_score - scores == 0 was wiped by eliminate_zeros().
        sources = [0, 0, 0, 1, 1, 2]
        targets = [1, 2, 3, 2, 3, 3]
        scores = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        connectivity = np.column_stack((sources, targets, scores)).astype(np.float32)
        params = {"UMAP_NEIGHBORS": 3, "UMAP_MIN_DIST": 0.1, "BOX_SCALE": 1.0}

        pos, box_limit = Layout_Engine_UMAP.calculate_layout(
            connectivity, n_nodes=4, params=params
        )
        self.assertEqual(pos.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(pos)))
        self.assertGreater(box_limit, 0.0)

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_graded_similarity_layout(self):
        # 5 nodes with graded similarities
        edges = [
            (0, 1, 95.0),
            (0, 2, 90.0),
            (1, 2, 85.0),
            (1, 3, 70.0),
            (2, 4, 60.0),
            (3, 4, 75.0),
            (3, 0, 65.0),
            (4, 0, 55.0),
        ]
        connectivity = np.array(edges, dtype=np.float32)
        params = {"UMAP_NEIGHBORS": 2, "UMAP_MIN_DIST": 0.1, "BOX_SCALE": 1.0}

        pos, box_limit = Layout_Engine_UMAP.calculate_layout(
            connectivity, n_nodes=5, params=params
        )
        self.assertEqual(pos.shape, (5, 2))
        self.assertTrue(np.all(np.isfinite(pos)))

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_low_degree_and_isolated_nodes_do_not_crash(self):
        # 6 nodes:
        # 0, 1, 2 form a triangle
        # 3 has only 1 edge (degree 1 < k=3)
        # 4 and 5 have 0 edges (completely isolated)
        edges = [
            (0, 1, 100.0),
            (1, 2, 100.0),
            (0, 2, 100.0),
            (0, 3, 50.0),
        ]
        connectivity = np.array(edges, dtype=np.float32)
        params = {"UMAP_NEIGHBORS": 3, "UMAP_MIN_DIST": 0.1, "BOX_SCALE": 1.0}

        pos, box_limit = Layout_Engine_UMAP.calculate_layout(
            connectivity, n_nodes=6, params=params
        )
        self.assertEqual(pos.shape, (6, 2))
        self.assertTrue(np.all(np.isfinite(pos)))
        self.assertGreater(box_limit, 0.0)


if __name__ == "__main__":
    unittest.main()
