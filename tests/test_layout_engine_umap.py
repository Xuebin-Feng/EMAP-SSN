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
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Layout_Engine_UMAP


class TestLayoutEngineUMAP(unittest.TestCase):
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
