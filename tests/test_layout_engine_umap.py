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

import contextlib
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from unittest import mock
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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


class UMAPSeedTests(unittest.TestCase):
    """LAYOUT_SEED must drive the UMAP engine, replacing its hardcoded 42."""

    EDGES = [(i, j, 100.0) for i in range(12) for j in range(i + 1, 12)]

    def _captured_kwargs(self, params):
        captured = []

        def construct(**kwargs):
            captured.append(kwargs)
            return mock.Mock(
                fit_transform=lambda x: np.column_stack(
                    (np.arange(12), np.arange(12))
                ).astype(np.float32)
            )

        with mock.patch.object(
            Layout_Engine_UMAP.umap, "UMAP", side_effect=construct
        ):
            Layout_Engine_UMAP.calculate_layout(
                np.array(self.EDGES, dtype=np.float32),
                12,
                {"UMAP_NEIGHBORS": 5, **params},
            )
        return captured[0]

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_default_seed_is_forty_two(self):
        self.assertEqual(self._captured_kwargs({})["random_state"], 42)

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_explicit_seed_reaches_the_reducer(self):
        self.assertEqual(
            self._captured_kwargs({"LAYOUT_SEED": 7})["random_state"], 7
        )

    @unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
    def test_null_seed_opts_out_of_determinism(self):
        self.assertIsNone(
            self._captured_kwargs({"LAYOUT_SEED": None})["random_state"]
        )

    def _empty_layout(self, params):
        positions, _ = Layout_Engine_UMAP.calculate_layout(
            np.zeros((0, 3), dtype=np.float32), n_nodes=10, params=params
        )
        return positions

    def test_empty_network_fallback_is_seeded(self):
        first = self._empty_layout({"BOX_SCALE": 1.5})
        second = self._empty_layout({"BOX_SCALE": 1.5})
        np.testing.assert_array_equal(first, second)

    def test_empty_network_fallback_varies_with_seed(self):
        first = self._empty_layout({"BOX_SCALE": 1.5, "LAYOUT_SEED": 1})
        second = self._empty_layout({"BOX_SCALE": 1.5, "LAYOUT_SEED": 2})
        self.assertFalse(np.array_equal(first, second))

    def test_negative_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            self._empty_layout({"BOX_SCALE": 1.5, "LAYOUT_SEED": -1})


def clique_ring_network(cliques=8, clique_size=6, ring=40):
    """(connectivity, n_nodes) of cliques whose scores are all equal, next to one graded ring.

    Every node of a clique of equal scores has the same weights to the others, so the
    fuzzy graph's component for it has a Laplacian with one repeated eigenvalue, and
    ARPACK runs out of Krylov space on it (six or more nodes) and asks for a random
    vector. A graph like that is where an unseeded eigsh made the layouts of one seed
    differ from run to run. The ring is one ordinary connected component.
    """
    edges = []
    node = 0
    for _ in range(cliques):
        members = list(range(node, node + clique_size))
        edges += [(a, b, 100.0) for i, a in enumerate(members) for b in members[i + 1:]]
        node += clique_size
    ring_nodes = list(range(node, node + ring))
    edges += [(ring_nodes[i], ring_nodes[(i + 1) % ring], 90.0 - i * 0.5) for i in range(ring)]
    edges += [(ring_nodes[i], ring_nodes[(i + 2) % ring], 60.0 - i * 0.3) for i in range(ring)]
    return np.array(edges, dtype=np.float32), node + ring


def uniform_clique_laplacian(size):
    """The normalised Laplacian umap hands eigsh for a clique of equal scores."""
    graph = scipy.sparse.coo_matrix(
        np.ones((size, size), dtype=np.float32) - np.eye(size, dtype=np.float32)
    )
    root_degree = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    scale = scipy.sparse.spdiags(1.0 / root_degree, 0, size, size)
    return scipy.sparse.identity(size, dtype=np.float64) - scale * graph * scale


def digest(array):
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


COLD_WARM_SCRIPT = """
import contextlib, io, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import Layout_Engine_UMAP as engine
import umap.layouts
connectivity = np.load(sys.argv[2])
n_nodes = int(connectivity[:, :2].max()) + 1
with contextlib.redirect_stdout(io.StringIO()):
    positions, _ = engine.calculate_layout(
        connectivity, n_nodes, {"UMAP_NEIGHBORS": 15, "UMAP_MIN_DIST": 0.1, "LAYOUT_SEED": 42})
np.save(sys.argv[3], positions)
print(json.dumps({"rdist_cache_hits": sum(umap.layouts.rdist.stats.cache_hits.values())}))
"""


@unittest.skipUnless(Layout_Engine_UMAP.UMAP_AVAILABLE, "umap-learn is required")
class SeededSpectralStartTests(unittest.TestCase):
    """A seeded UMAP layout is the same layout every time, whatever ran before it.

    The unseeded draw lay in SciPy's ARPACK, not in the Numba cache: a cold and a warm
    cache only looked different because two runs of the same seed disagreed by chance.
    """

    def layout(self, connectivity, n_nodes, seed=42, dimensions=2):
        with contextlib.redirect_stdout(io.StringIO()):
            positions, _ = Layout_Engine_UMAP.calculate_layout(
                connectivity, n_nodes,
                {"UMAP_NEIGHBORS": 15, "UMAP_MIN_DIST": 0.1, "LAYOUT_SEED": seed,
                 "LAYOUT_DIMENSIONS": dimensions},
            )
        return positions

    def eigenvectors(self, size=6, **kwargs):
        values, vectors = scipy.sparse.linalg.eigsh(
            uniform_clique_laplacian(size), 3, which="SM", ncv=max(7, int(np.sqrt(size))),
            tol=1e-4, v0=np.ones(size), maxiter=size * 5, **kwargs,
        )
        return vectors

    def test_control_arpack_draws_an_unseeded_vector_on_equal_score_cliques(self):
        # Without this the fixture below would prove nothing: eigsh must really differ
        # between calls when unseeded, and agree when given the same rng.
        unseeded = {digest(self.eigenvectors()) for _ in range(8)}
        self.assertGreater(len(unseeded), 1, "this SciPy's eigsh is deterministic; the seeding may be unnecessary")
        self.assertEqual(len({digest(self.eigenvectors(rng=42)) for _ in range(8)}), 1)

    def test_control_unseeded_layouts_of_equal_score_cliques_differ(self):
        connectivity, n_nodes = clique_ring_network()
        with mock.patch.object(Layout_Engine_UMAP, "_seeded_arpack",
                               lambda seed: contextlib.nullcontext()):
            layouts = {digest(self.layout(connectivity, n_nodes)) for _ in range(6)}
        self.assertGreater(len(layouts), 1)

    def test_seeded_layouts_of_equal_score_cliques_repeat_exactly(self):
        connectivity, n_nodes = clique_ring_network()
        first = self.layout(connectivity, n_nodes)
        for _ in range(4):
            np.testing.assert_array_equal(self.layout(connectivity, n_nodes), first)

    def test_seeded_3d_layouts_of_equal_score_cliques_repeat_exactly(self):
        connectivity, n_nodes = clique_ring_network()
        first = self.layout(connectivity, n_nodes, dimensions=3)
        self.assertEqual(first.shape, (n_nodes, 3))
        for _ in range(2):
            np.testing.assert_array_equal(self.layout(connectivity, n_nodes, dimensions=3), first)

    def test_seed_still_chooses_the_layout(self):
        connectivity, n_nodes = clique_ring_network()
        self.assertFalse(np.array_equal(
            self.layout(connectivity, n_nodes, seed=1), self.layout(connectivity, n_nodes, seed=2)))

    def test_layout_is_unchanged_when_arpack_needs_no_random_vector(self):
        # One ordinary component: ARPACK never asks for a vector, so the seeded run must
        # be bit-identical to umap run without the wrapper.
        connectivity, n_nodes = clique_ring_network(cliques=0)
        captured = []
        original = Layout_Engine_UMAP.umap.UMAP

        def construct(**kwargs):
            captured.append(kwargs)
            reducer = original(**kwargs)
            captured.append(reducer)
            return reducer

        with mock.patch.object(Layout_Engine_UMAP.umap, "UMAP", side_effect=construct):
            self.layout(connectivity, n_nodes)
        kwargs, seeded = captured
        dummy = np.zeros((n_nodes, 1), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = [original(**kwargs).fit_transform(dummy) for _ in range(2)]
        np.testing.assert_array_equal(plain[0], plain[1])
        np.testing.assert_array_equal(seeded.embedding_, plain[0])

    def test_seeding_is_scoped_to_the_layout_and_to_its_thread(self):
        calls = []

        def eigsh(matrix, k=6, rng=None):
            calls.append((threading.current_thread().name, rng))

        library = scipy.sparse.linalg
        with mock.patch.object(library, "eigsh", eigsh):
            with Layout_Engine_UMAP._seeded_arpack(7):
                self.assertIsNot(library.eigsh, eigsh)
                library.eigsh(None)
                library.eigsh(None, rng=3)
                other = threading.Thread(target=lambda: library.eigsh(None), name="other")
                other.start()
                other.join()
                with Layout_Engine_UMAP._seeded_arpack(9):
                    library.eigsh(None)
                library.eigsh(None)
            self.assertIs(library.eigsh, eigsh)
            library.eigsh(None)
            with Layout_Engine_UMAP._seeded_arpack(None):
                self.assertIs(library.eigsh, eigsh)
        main = threading.current_thread().name
        self.assertEqual(calls, [(main, 7), (main, 3), ("other", None), (main, 9), (main, 7), (main, None)])

    def test_seeding_covers_scikit_learn_and_is_undone(self):
        import sklearn.manifold._spectral_embedding as embedding
        before = scipy.sparse.linalg.eigsh, embedding.eigsh
        with Layout_Engine_UMAP._seeded_arpack(1):
            self.assertIs(embedding.eigsh, scipy.sparse.linalg.eigsh)
            self.assertIsNot(embedding.eigsh, before[1])
        self.assertEqual((scipy.sparse.linalg.eigsh, embedding.eigsh), before)
        with self.assertRaises(RuntimeError), Layout_Engine_UMAP._seeded_arpack(1):
            raise RuntimeError("a failing layout still restores eigsh")
        self.assertEqual((scipy.sparse.linalg.eigsh, embedding.eigsh), before)

    def test_cold_and_warm_numba_caches_give_the_same_bits(self):
        connectivity, _ = clique_ring_network()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            folder = pathlib.Path(folder)
            np.save(folder / "connectivity.npy", connectivity)

            def start(cache, name):
                environment = {**os.environ, "NUMBA_CACHE_DIR": str(folder / cache)}
                return subprocess.Popen(
                    [sys.executable, "-I", "-c", COLD_WARM_SCRIPT, str(SRC),
                     str(folder / "connectivity.npy"), str(folder / (name + ".npy"))],
                    env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            def finish(process):
                out, err = process.communicate(timeout=600)
                self.assertEqual(process.returncode, 0, err[-2000:])
                return json.loads(out.strip().splitlines()[-1])["rdist_cache_hits"]

            # two cold processes with empty caches of their own, then one on the first cache
            cold = [start("cache_a", "cold_a"), start("cache_b", "cold_b")]
            try:
                hits = [finish(process) for process in cold]
            finally:
                for process in cold:
                    if process.poll() is None:
                        process.kill()
            hits.append(finish(start("cache_a", "warm_a")))
            self.assertEqual(hits[:2], [0, 0], "the first two runs must start with empty caches")
            self.assertGreater(hits[2], 0, "the third run must load umap's kernel from the cache")
            layouts = [np.load(folder / (name + ".npy")) for name in ("cold_a", "cold_b", "warm_a")]
            np.testing.assert_array_equal(layouts[0], layouts[1])
            np.testing.assert_array_equal(layouts[0], layouts[2])


if __name__ == "__main__":
    unittest.main()
