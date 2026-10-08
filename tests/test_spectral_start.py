"""utilities.Spectral_Start: the spectral start's axes (the eigenvectors after
the trivial one, from LOBPCG or ARPACK), its fallbacks, its determinism, the
asinh spread of each axis, and the threaded Laplacian product."""
import io
import os
import platform
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Hardware_Acceleration as Layout_Hardware  # noqa: E402
from utilities import Spectral_Start  # noqa: E402


def clustered_graph(sizes, seed=3, links=3, chords=2):
    """Clusters of distinct sizes (rings plus random chords) joined in a chain by
    a few links, so the smallest non-trivial eigenvalues are well separated."""
    rng = np.random.default_rng(seed)
    blocks, groups, start = [], [], 0
    for size in sizes:
        nodes = np.arange(start, start + size)
        groups.append(nodes)
        start += size
        blocks.append(np.column_stack((nodes, np.roll(nodes, -1))))
        blocks.append(rng.choice(nodes, size=(chords * size, 2)))
    for left, right in zip(groups, groups[1:]):
        blocks.append(np.column_stack((rng.choice(left, links), rng.choice(right, links))))
    edges = np.vstack(blocks)
    edges = edges[edges[:, 0] != edges[:, 1]].astype(np.int32)
    return edges, rng.uniform(0.5, 1.0, len(edges)), start


GRAPH = clustered_graph((150, 110, 80, 60, 40))


def dense_pairs(edges, scores, nodes, dimensions):
    """Exact eigenpairs 0 .. dimensions, and sqrt(degree)."""
    laplacian, root_degree = Spectral_Start.normalized_laplacian(edges, scores, nodes)
    values, vectors = scipy.linalg.eigh(laplacian.toarray(), subset_by_index=[0, dimensions + 1])
    return values, vectors, root_degree


def cosines(found, expected):
    return [abs(found[:, i] @ expected[:, i])
            / np.linalg.norm(found[:, i]) / np.linalg.norm(expected[:, i])
            for i in range(expected.shape[1])]


def solve(dimensions, seed=7, graph=GRAPH, verbose=False):
    edges, scores, nodes = graph
    rng = np.random.default_rng(seed)
    output = io.StringIO()
    with redirect_stdout(output):
        axes = Spectral_Start.spectral_axes(
            edges, scores, nodes, dimensions, draw=rng.standard_normal, verbose=verbose)
    return axes, output.getvalue()


class IntendedAxesTests(unittest.TestCase):
    def test_the_trivial_vector_is_dropped_wherever_the_solver_returns_it(self):
        for dimensions in (2, 3):
            values, vectors, root_degree = dense_pairs(*GRAPH, dimensions)
            pairs = values[:dimensions + 1], vectors[:, :dimensions + 1]
            for order in (np.arange(dimensions + 1), np.arange(dimensions + 1)[::-1]):
                axes = Spectral_Start.intended_axes(
                    pairs[0][order], pairs[1][:, order], root_degree, dimensions)
                np.testing.assert_array_equal(axes, vectors[:, 1:dimensions + 1])

    def test_without_the_trivial_vector_the_first_axes_are_kept(self):
        # ARPACK's which="SM" often returns lambda2 upward without lambda1 = 0.
        for dimensions in (2, 3):
            values, vectors, root_degree = dense_pairs(*GRAPH, dimensions)
            axes = Spectral_Start.intended_axes(
                values[1:][::-1], vectors[:, 1:][:, ::-1], root_degree, dimensions)
            np.testing.assert_array_equal(axes, vectors[:, 1:dimensions + 1])

    def test_unusable_solver_output_is_refused(self):
        values, vectors, root_degree = dense_pairs(*GRAPH, 2)
        broken = vectors.copy()
        broken[0, 1] = np.nan
        with self.assertRaises(ValueError):
            Spectral_Start.intended_axes(values, broken, root_degree, 2)
        with self.assertRaises(ValueError):    # the trivial vector and one more
            Spectral_Start.intended_axes(values[:2], vectors[:, :2], root_degree, 2)


class SolverAgreementTests(unittest.TestCase):
    """Both solvers find the eigenvectors after the trivial one, in 2D and 3D."""

    def assert_matches_dense(self, axes, dimensions, minimum):
        _, vectors, _ = dense_pairs(*GRAPH, dimensions)
        self.assertEqual(axes.shape, (GRAPH[2], dimensions))
        for value in cosines(axes, vectors[:, 1:dimensions + 1]):
            self.assertGreater(value, minimum)

    def test_lobpcg_matches_dense_eigh(self):
        for dimensions in (2, 3):
            with mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
                    mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                      wraps=scipy.sparse.linalg.lobpcg) as lobpcg, \
                    mock.patch.object(scipy.sparse.linalg, "eigsh",
                                      side_effect=AssertionError("ARPACK ran")):
                axes, _ = solve(dimensions)
            start = lobpcg.call_args.args[1]
            self.assertEqual(start.shape, (GRAPH[2], dimensions + 4))
            constraint = lobpcg.call_args.kwargs["Y"]
            _, _, root_degree = dense_pairs(*GRAPH, dimensions)
            np.testing.assert_allclose(
                constraint[:, 0], root_degree / np.linalg.norm(root_degree))
            self.assert_matches_dense(axes, dimensions, 1 - 1e-8)

    def test_arpack_matches_dense_eigh_below_the_lobpcg_size(self):
        for dimensions in (2, 3):
            with mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                   side_effect=AssertionError("LOBPCG ran")):
                axes, _ = solve(dimensions)
            self.assert_matches_dense(axes, dimensions, 1 - 1e-4)

    def test_a_large_components_batch_starts_from_lobpcg(self):
        edges, scores, nodes = GRAPH
        with mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                  wraps=scipy.sparse.linalg.lobpcg) as lobpcg, \
                redirect_stdout(io.StringIO()):
            Layout_Hardware.prepare_layout_batch(
                [np.arange(nodes)], np.zeros(nodes, dtype=np.int64), {0: edges}, {0: scores},
                {"BOX_SCALE": 1.0}, rng=np.random.default_rng(1))
        self.assertEqual(lobpcg.call_count, 1)

    def test_a_component_of_exactly_the_lobpcg_size_uses_lobpcg(self):
        with mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", GRAPH[2]), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                  wraps=scipy.sparse.linalg.lobpcg) as lobpcg:
            solve(2)
        self.assertEqual(lobpcg.call_count, 1)


class FallbackTests(unittest.TestCase):
    """A LOBPCG result that fails or can't be trusted falls back to ARPACK,
    which still yields the axes after the trivial one, reproducibly."""

    def fake_lobpcg(self, broken):
        def fake(operator, start, **kwargs):
            values, vectors, root_degree = dense_pairs(*GRAPH, 2)
            values, vectors = values[1:], vectors[:, 1:].copy()
            if broken == "residual":
                vectors[:, 0] += np.random.default_rng(0).normal(0.0, 1e-3, len(vectors))
            else:
                # The trivial pair itself: its residual is zero, so only the
                # check against sqrt(degree) can reject it.
                values = values.copy()
                values[1] = 0.0
                vectors[:, 1] = root_degree / np.linalg.norm(root_degree)
            return values, vectors
        return fake

    def test_every_rejected_lobpcg_result_still_yields_the_intended_axes(self):
        def missing(*args, **kwargs):
            raise AssertionError("unreachable")

        cases = {
            "exception": mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                           side_effect=RuntimeError("diverged")),
            "maximum iterations": mock.patch.object(Spectral_Start, "LOBPCG_MAX_ITERATIONS", 1),
            "bad residual": mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                              side_effect=self.fake_lobpcg("residual")),
            "vector parallel to sqrt(d)": mock.patch.object(
                scipy.sparse.linalg, "lobpcg", side_effect=self.fake_lobpcg("trivial")),
            "import error": mock.patch.dict(scipy.sparse.linalg.__dict__, {"lobpcg": missing}),
        }
        _, vectors, _ = dense_pairs(*GRAPH, 2)
        for name, patch in cases.items():
            with self.subTest(name), patch, \
                    mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
                    mock.patch.object(scipy.sparse.linalg, "eigsh",
                                      wraps=scipy.sparse.linalg.eigsh) as eigsh:
                if name == "import error":
                    del scipy.sparse.linalg.__dict__["lobpcg"]
                first, log = solve(2, verbose=True)
                second, _ = solve(2)
                self.assertEqual(eigsh.call_count, 2)
                self.assertIn("using ARPACK instead", log)
                np.testing.assert_array_equal(first, second)
                for value in cosines(first, vectors[:, 1:3]):
                    self.assertGreater(value, 1 - 1e-4)


class DeterminismTests(unittest.TestCase):
    def test_a_seeded_start_ignores_numpys_global_generator(self):
        for minimum in (100, 10 ** 9):          # the LOBPCG path, then ARPACK
            with mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", minimum):
                np.random.seed(1)
                first, _ = solve(2)
                np.random.seed(2)
                second, _ = solve(2)
            np.testing.assert_array_equal(first, second)

    def test_the_fallback_continues_the_same_stream(self):
        with mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                  side_effect=RuntimeError("diverged")), \
                mock.patch.object(scipy.sparse.linalg, "eigsh",
                                  wraps=scipy.sparse.linalg.eigsh) as eigsh:
            solve(2, seed=11)
        rng = np.random.default_rng(11)
        rng.standard_normal((GRAPH[2], 6))              # LOBPCG's start block
        np.testing.assert_array_equal(eigsh.call_args.kwargs["v0"],
                                      rng.standard_normal(GRAPH[2]))


class AsinhScalingTests(unittest.TestCase):
    def test_the_central_ninety_percent_sets_the_scale(self):
        rng = np.random.default_rng(5)
        coordinates = np.concatenate((rng.normal(0.0, 0.01, 970), rng.normal(5.0, 0.5, 30)))
        low, middle, high = np.percentile(coordinates, [5, 50, 95])
        expected = np.arcsinh((coordinates - middle) / ((high - low) / 2.0))
        scaled = Spectral_Start.asinh_scaled(coordinates)
        np.testing.assert_allclose(scaled, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(np.argsort(scaled, kind="stable"),
                                      np.argsort(coordinates, kind="stable"))
        # Min-max alone would leave the 970-node bulk about 1% of the range.
        self.assertGreater(np.ptp(scaled[:970]) / np.ptp(scaled), 0.25)

    def test_an_empty_central_range_uses_the_full_range(self):
        star_axis = np.zeros(40)
        star_axis[:2] = (0.7, -0.7)              # a star's eigenvector: leaves only
        nearly_empty = star_axis.copy()
        nearly_empty[2:] = np.linspace(-1e-12, 1e-12, 38)
        for coordinates in (star_axis, nearly_empty):
            middle = np.median(coordinates)
            expected = np.arcsinh((coordinates - middle) / (np.ptp(coordinates) / 2.0))
            scaled = Spectral_Start.asinh_scaled(coordinates)
            self.assertTrue(np.isfinite(scaled).all())
            np.testing.assert_allclose(scaled, expected, rtol=0, atol=1e-12)
        constant = np.full(5, 0.25)
        np.testing.assert_array_equal(Spectral_Start.asinh_scaled(constant), constant)

    def test_symmetric_tiny_and_clustered_components_get_finite_spread_starts(self):
        def ring(n):
            return [(i, (i + 1) % n) for i in range(n)]

        def clique(nodes):
            return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]

        components = {
            "star": (9, [(0, leaf) for leaf in range(1, 9)]),
            "path": (10, [(i, i + 1) for i in range(9)]),
            "clique": (6, clique(list(range(6)))),
            "two cliques and a bridge": (12, clique(list(range(6)))
                                         + clique(list(range(6, 12))) + [(5, 6)]),
            "4 nodes": (4, [(0, 1), (1, 2), (2, 3)]),
            "5 nodes": (5, ring(5)),
            "6 nodes": (6, ring(6)),
        }
        for dimensions in (2, 3):
            for name, (nodes, edges) in components.items():
                if nodes < dimensions + 2:
                    continue
                with self.subTest(name=name, dimensions=dimensions):
                    batches = [self.prepare(nodes, edges, dimensions) for _ in range(2)]
                    positions, limit = batches[0].positions, batches[0].box_limits[0]
                    np.testing.assert_array_equal(positions, batches[1].positions)
                    self.assertTrue(np.isfinite(positions).all())
                    self.assertTrue((np.ptp(positions, axis=0) > 0.1 * limit).all())
                    self.assertTrue((np.abs(positions) <= 0.4 * limit + 1.0).all())

    def test_an_outlying_group_no_longer_squeezes_the_component(self):
        # A 600-node cluster with a 6-node group hanging off one weak link: the
        # group dominates the second eigenvector's range.
        edges, scores, _ = clustered_graph((600,), seed=9)
        tail = np.array([(600 + i, 601 + i) for i in range(5)] + [(0, 600)], dtype=np.int32)
        edges = np.vstack((edges, tail))
        scores = np.concatenate((scores, np.full(5, 1.0), [0.01]))
        batch = self.prepare(606, edges, 2, scores=scores)
        bulk = batch.positions[:600]
        share = (np.percentile(bulk, 95, axis=0) - np.percentile(bulk, 5, axis=0)) / (
            0.8 * batch.box_limits[0])
        # About 0.21 and 0.09; min-max scaling alone gives 0.003 and 0.0003.
        self.assertTrue((share > 0.05).all(), share)

    @staticmethod
    def prepare(nodes, edges, dimensions, scores=None):
        edges = np.asarray(edges, dtype=np.int32)
        scores = np.ones(len(edges)) if scores is None else scores
        with redirect_stdout(io.StringIO()):
            return Layout_Hardware.prepare_layout_batch(
                [np.arange(nodes)], np.zeros(nodes, dtype=np.int64), {0: edges}, {0: scores},
                {"BOX_SCALE": 1.0, "LAYOUT_DIMENSIONS": dimensions},
                rng=np.random.default_rng(4))


class ProductTests(unittest.TestCase):
    def test_threaded_products_match_scipys(self):
        """The same bits on x86-64. Elsewhere only within rounding: an ARM
        compiler may fuse SciPy's multiply-add into one instruction, and Numba
        never fuses without fastmath."""
        edges, scores, nodes = GRAPH
        doubled = np.vstack((edges, edges[:50], [[3, 3]]))      # duplicates and a self-loop
        laplacian, _ = Spectral_Start.normalized_laplacian(
            doubled, np.concatenate((scores, scores[:50], [1.0])), nodes)
        csr = laplacian.tocsr()
        operator = Spectral_Start.laplacian_operator(csr)
        block = np.random.default_rng(2).standard_normal((nodes, 6))
        if platform.machine().lower() in {"x86_64", "amd64"}:
            check = np.testing.assert_array_equal
        else:
            def check(found, expected):
                np.testing.assert_allclose(found, expected, rtol=1e-12, atol=1e-15)
        check(operator @ block, csr @ block)
        check(operator @ block[:, 0], csr @ block[:, 0])

    def test_without_numba_scipy_does_the_products(self):
        # Without Numba the row loop would run as plain Python: hours on a large component.
        edges, scores, nodes = GRAPH
        laplacian, _ = Spectral_Start.normalized_laplacian(edges, scores, nodes)
        csr = laplacian.tocsr()
        block = np.random.default_rng(2).standard_normal((nodes, 3))
        with mock.patch.object(Spectral_Start, "NUMBA_AVAILABLE", False), \
                mock.patch.object(Spectral_Start, "_csr_rows_product",
                                  side_effect=AssertionError("the Numba loop ran")):
            operator = Spectral_Start.laplacian_operator(csr)
            np.testing.assert_array_equal(operator @ block, csr @ block)


if __name__ == "__main__":
    unittest.main()
