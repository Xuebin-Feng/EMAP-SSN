"""utilities.Spectral_Start: the spectral start's axes (the eigenvectors after
the trivial one, from LOBPCG, AMG-preconditioned LOBPCG or ARPACK), its
fallbacks, its determinism, the asinh spread of each axis, and the threaded
Laplacian product.

pyamg is optional. Tests marked as needing it skip without it; the others fake
it, or hide it the way a missing package is hidden (sys.modules["pyamg"] = None).
"""
import contextlib
import gc
import io
import os
import platform
import sys
import types
import unittest
import weakref
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Hardware_Acceleration as Layout_Hardware  # noqa: E402
from utilities import Spectral_Start  # noqa: E402

try:
    import pyamg
except ImportError:
    pyamg = None

_ABSENT = object()


@contextlib.contextmanager
def pyamg_as(module):
    """Make ``import pyamg`` give ``module``; None fails like a missing package.

    Only the one entry is swapped and restored: mock.patch.dict(sys.modules)
    would also drop every module first imported inside it."""
    saved = sys.modules.get("pyamg", _ABSENT)
    sys.modules["pyamg"] = module
    try:
        yield
    finally:
        if saved is _ABSENT:
            sys.modules.pop("pyamg", None)
        else:
            sys.modules["pyamg"] = saved


def amg_switched_off():
    return mock.patch.dict(os.environ, {Spectral_Start.AMG_SWITCH_VARIABLE: "0"})


def fake_pyamg(failure=None):
    """A stand-in pyamg whose hierarchy setup raises ``failure``."""
    module = types.ModuleType("pyamg")
    module.__version__ = "0.0-test"

    def smoothed_aggregation_solver(*args, **kwargs):
        raise failure or AssertionError("the fake pyamg was asked for a hierarchy")

    module.smoothed_aggregation_solver = smoothed_aggregation_solver
    return module


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


def first_block(seed=5, nodes=GRAPH[2], width=6):
    """The start block a solve draws first from ``default_rng(seed)``."""
    return np.random.default_rng(seed).standard_normal((nodes, width))


class LobpcgRecorder:
    """Stands in for scipy's lobpcg: records each call's start block (LOBPCG
    overwrites it) and whether it was preconditioned, then runs the real
    solver. With ``reject_preconditioned``, a preconditioned solve's result is
    spoiled so that the residual check rejects it."""

    def __init__(self, reject_preconditioned=False):
        self.calls = []
        self.reject_preconditioned = reject_preconditioned
        self.real = scipy.sparse.linalg.lobpcg

    def __call__(self, operator, start, **kwargs):
        preconditioned = kwargs.get("M") is not None
        self.calls.append((start.copy(), preconditioned))
        values, vectors = self.real(operator, start, **kwargs)
        if preconditioned and self.reject_preconditioned:
            vectors = vectors.copy()
            vectors[:, 0] += np.random.default_rng(0).normal(0.0, 1e-3, len(vectors))
        return values, vectors

    def preconditioned(self):
        return [preconditioned for _, preconditioned in self.calls]


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
            with amg_switched_off(), mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
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
        with amg_switched_off(), mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                  wraps=scipy.sparse.linalg.lobpcg) as lobpcg, \
                redirect_stdout(io.StringIO()):
            Layout_Hardware.prepare_layout_batch(
                [np.arange(nodes)], np.zeros(nodes, dtype=np.int64), {0: edges}, {0: scores},
                {"BOX_SCALE": 1.0}, rng=np.random.default_rng(1))
        self.assertEqual(lobpcg.call_count, 1)

    def test_a_component_of_exactly_the_lobpcg_size_uses_lobpcg(self):
        with amg_switched_off(), mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", GRAPH[2]), \
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
            with self.subTest(name), patch, amg_switched_off(), \
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

    def test_the_axes_do_not_depend_on_the_thread_count(self):
        # Every threaded step (the Laplacian products, the l1 row sums) is a
        # row-wise gather: one thread adds each output row in storage order.
        modes = [("plain", amg_switched_off())]
        if pyamg is not None:
            modes.append(("AMG", mock.patch.object(Spectral_Start, "AMG_MAX_COARSE", 40)))
        for name, mode in modes:
            with self.subTest(name), mode, \
                    mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100):
                default, _ = solve(2)
                with mock.patch.object(Spectral_Start.Numba_Threads, "default_thread_count",
                                       return_value=1):
                    single, _ = solve(2)
                np.testing.assert_array_equal(single, default)

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


class AmgFallbackTests(unittest.TestCase):
    """Without pyamg, with SSN_SPECTRAL_AMG=0, or when the AMG-preconditioned
    solve fails or is rejected, plain LOBPCG runs from the same start block:
    the same bits as a machine without pyamg. Runs with or without pyamg."""

    def setUp(self):
        patcher = mock.patch.object(Spectral_Start, "LOBPCG_MIN_NODES", 100)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def without_pyamg(seed=5):
        with pyamg_as(None):
            return solve(2, seed=seed)[0]

    def test_without_pyamg_lobpcg_runs_unpreconditioned_from_the_first_block(self):
        recorder = LobpcgRecorder()
        with pyamg_as(None), mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder):
            axes, log = solve(2, seed=5, verbose=True)
        self.assertEqual(recorder.preconditioned(), [False])
        np.testing.assert_array_equal(recorder.calls[0][0], first_block())
        self.assertIn("LOBPCG without a preconditioner (pyamg is not installed).", log)
        laplacian, root_degree = Spectral_Start.normalized_laplacian(*GRAPH)
        expected, _ = Spectral_Start._lobpcg_axes(laplacian.tocsr(), root_degree, 2, first_block())
        np.testing.assert_array_equal(axes, expected)

    def test_the_switch_turns_amg_off_where_pyamg_is_installed(self):
        stand_in = fake_pyamg()      # fails the test if asked for a hierarchy
        for value in ("0", "off", "False", " no "):
            with self.subTest(value=value), pyamg_as(stand_in), \
                    mock.patch.dict(os.environ, {Spectral_Start.AMG_SWITCH_VARIABLE: value}):
                self.assertEqual(Spectral_Start.amg_module(),
                                 (None, "disabled by SSN_SPECTRAL_AMG"))
                axes, log = solve(2, seed=5, verbose=True)
                self.assertIn("LOBPCG without a preconditioner (disabled by SSN_SPECTRAL_AMG).", log)
                np.testing.assert_array_equal(axes, self.without_pyamg())
        for value in ("", "1", "on"):
            with self.subTest(value=value), pyamg_as(stand_in), \
                    mock.patch.dict(os.environ, {Spectral_Start.AMG_SWITCH_VARIABLE: value}):
                self.assertEqual(Spectral_Start.amg_module(), (stand_in, None))

    def test_a_broken_pyamg_install_is_reported_not_raised(self):
        real_import = __import__

        def broken(name, *args, **kwargs):
            if name == "pyamg":
                raise OSError("DLL load failed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=broken):
            self.assertEqual(Spectral_Start.amg_module(),
                             (None, "pyamg failed to import (OSError: DLL load failed)"))

    def test_a_failing_amg_setup_retries_plain_lobpcg_with_the_same_block(self):
        recorder = LobpcgRecorder()
        with pyamg_as(fake_pyamg(MemoryError("hierarchy"))), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder):
            axes, log = solve(2, seed=5, verbose=True)
        self.assertEqual(recorder.preconditioned(), [False])
        np.testing.assert_array_equal(recorder.calls[0][0], first_block())
        self.assertIn("LOBPCG with the AMG preconditioner failed (MemoryError: hierarchy); "
                      "retrying without it.", log)
        self.assertIn("LOBPCG without a preconditioner.", log)
        np.testing.assert_array_equal(axes, self.without_pyamg())

    def test_a_rejected_amg_result_retries_plain_lobpcg_with_the_same_block(self):
        recorder = LobpcgRecorder(reject_preconditioned=True)
        identity = scipy.sparse.linalg.aslinearoperator(sp.identity(GRAPH[2]))
        with pyamg_as(fake_pyamg()), \
                mock.patch.object(Spectral_Start, "amg_preconditioner", return_value=identity), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder):
            axes, log = solve(2, seed=5, verbose=True)
        self.assertEqual(recorder.preconditioned(), [True, False])
        for start, _ in recorder.calls:     # the retry gets the block untouched
            np.testing.assert_array_equal(start, first_block())
        self.assertIn("LOBPCG with the AMG preconditioner did not converge", log)
        np.testing.assert_array_equal(axes, self.without_pyamg())

    def test_arpack_still_comes_last_and_continues_the_stream(self):
        with pyamg_as(fake_pyamg(RuntimeError("no hierarchy"))), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg",
                                  side_effect=RuntimeError("diverged")), \
                mock.patch.object(scipy.sparse.linalg, "eigsh",
                                  wraps=scipy.sparse.linalg.eigsh) as eigsh:
            _, log = solve(2, seed=11, verbose=True)
        rng = np.random.default_rng(11)
        rng.standard_normal((GRAPH[2], 6))              # the one start block
        self.assertEqual(eigsh.call_count, 1)
        np.testing.assert_array_equal(eigsh.call_args.kwargs["v0"], rng.standard_normal(GRAPH[2]))
        self.assertLess(log.index("retrying without it"), log.index("using ARPACK instead"))


@unittest.skipIf(pyamg is None, "pyamg is not installed")
class AmgSolveTests(unittest.TestCase):
    """LOBPCG preconditioned with the V-cycle on pyamg's hierarchy. The test
    graph is coarsened over several levels, as a large component would be."""

    def setUp(self):
        for name, value in (("LOBPCG_MIN_NODES", 100), ("AMG_MAX_COARSE", 40)):
            patcher = mock.patch.object(Spectral_Start, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_amg_lobpcg_matches_dense_eigh(self):
        depths = []
        real = pyamg.smoothed_aggregation_solver

        def counted(*args, **kwargs):
            hierarchy = real(*args, **kwargs)
            depths.append(len(hierarchy.levels))
            return hierarchy

        for dimensions in (2, 3):
            recorder = LobpcgRecorder()
            with mock.patch.object(pyamg, "smoothed_aggregation_solver", side_effect=counted), \
                    mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder), \
                    mock.patch.object(scipy.sparse.linalg, "eigsh",
                                      side_effect=AssertionError("ARPACK ran")):
                axes, log = solve(dimensions, verbose=True)
            self.assertEqual(recorder.preconditioned(), [True])
            self.assertIn(f"LOBPCG with an AMG preconditioner (pyamg {pyamg.__version__}).", log)
            _, vectors, _ = dense_pairs(*GRAPH, dimensions)
            for value in cosines(axes, vectors[:, 1:dimensions + 1]):
                self.assertGreater(value, 1 - 1e-8)
        self.assertTrue(all(depth >= 3 for depth in depths), depths)

    def test_the_v_cycle_contracts_the_error_off_sqrt_degree(self):
        # The error after one cycle, I - M L, shrinks every direction but
        # sqrt(degree): by about 0.91 here, against 0.999 for the smoothing
        # alone. With a one-row coarsest level, that row is sqrt(degree)'s
        # image, so its inverse is all rounding unless that direction is removed.
        laplacian, root_degree = Spectral_Start.normalized_laplacian(*GRAPH)
        csr = laplacian.tocsr()
        unit = root_degree / np.linalg.norm(root_degree)
        outside = np.eye(GRAPH[2]) - np.outer(unit, unit)
        for coarse_rows in (40, 1):
            with self.subTest(coarse_rows=coarse_rows), \
                    mock.patch.object(Spectral_Start, "AMG_MAX_COARSE", coarse_rows):
                preconditioner = Spectral_Start.amg_preconditioner(pyamg, csr, root_degree)
                error = outside @ (np.eye(GRAPH[2]) - preconditioner @ csr.toarray()) @ outside
                self.assertLess(max(abs(np.linalg.eigvals(error))), 0.95)

    def test_a_single_level_is_the_pseudo_inverse_off_sqrt_degree(self):
        # Checked against its definition: numpy.linalg.pinv is no reference here,
        # as its cutoff keeps sqrt(degree)'s rounding-level singular value
        # (|L x - b| about 0.02 and x . sqrt(degree) about 9 on this graph).
        laplacian, root_degree = Spectral_Start.normalized_laplacian(*GRAPH)
        with mock.patch.object(Spectral_Start, "AMG_MAX_COARSE", GRAPH[2]):
            preconditioner = Spectral_Start.amg_preconditioner(pyamg, laplacian.tocsr(), root_degree)
        unit = root_degree / np.linalg.norm(root_degree)
        block = np.random.default_rng(8).standard_normal((GRAPH[2], 3))
        block -= np.outer(unit, unit @ block)
        solved = preconditioner @ block
        np.testing.assert_allclose(laplacian @ solved, block, rtol=0, atol=1e-9)
        self.assertLess(np.abs(unit @ solved).max(), 1e-9)

    def test_a_matrix_small_enough_to_be_the_coarsest_level_is_solved_densely(self):
        recorder = LobpcgRecorder()
        with mock.patch.object(Spectral_Start, "AMG_MAX_COARSE", GRAPH[2]), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder):
            axes, _ = solve(2)
        self.assertEqual(recorder.preconditioned(), [True])
        _, vectors, _ = dense_pairs(*GRAPH, 2)
        for value in cosines(axes, vectors[:, 1:3]):
            self.assertGreater(value, 1 - 1e-8)

    def test_the_hierarchy_is_built_without_random_numbers_or_matrix_copies(self):
        with mock.patch.object(pyamg, "smoothed_aggregation_solver",
                               wraps=pyamg.smoothed_aggregation_solver) as built:
            solve(2)
        options = built.call_args.kwargs
        self.assertEqual({key: options[key] for key in
                          ("strength", "smooth", "improve_candidates", "max_coarse")},
                         {"strength": None, "smooth": None, "improve_candidates": None,
                          "max_coarse": 40})
        _, _, root_degree = dense_pairs(*GRAPH, 2)
        np.testing.assert_array_equal(options["B"][:, 0], root_degree)

    def test_the_v_cycle_is_symmetric_and_positive_off_sqrt_degree(self):
        laplacian, root_degree = Spectral_Start.normalized_laplacian(*GRAPH)
        preconditioner = Spectral_Start.amg_preconditioner(pyamg, laplacian.tocsr(), root_degree)
        rng = np.random.default_rng(3)
        left, right = rng.standard_normal((GRAPH[2], 3)), rng.standard_normal((GRAPH[2], 3))
        np.testing.assert_allclose(left.T @ (preconditioner @ right),
                                   (preconditioner @ left).T @ right, rtol=1e-10, atol=1e-12)
        unit = root_degree / np.linalg.norm(root_degree)
        block = right - np.outer(unit, unit @ right)
        self.assertTrue((np.einsum("ij,ij->j", block, preconditioner @ block) > 0).all())
        # One vector and a block take the same path (BLAS may round them differently).
        np.testing.assert_allclose(preconditioner @ right[:, 0], (preconditioner @ right)[:, 0],
                                   rtol=1e-12, atol=1e-14)

    def test_amg_starts_are_reproducible_and_leave_numpys_generator_alone(self):
        np.random.seed(1)
        before = np.random.get_state()
        first, _ = solve(2)
        after = np.random.get_state()
        self.assertEqual((before[0],) + before[2:], (after[0],) + after[2:])
        np.testing.assert_array_equal(before[1], after[1])
        np.random.seed(2)
        second, _ = solve(2)
        np.testing.assert_array_equal(first, second)

    def test_stalled_coarsening_falls_back_to_plain_lobpcg(self):
        recorder = LobpcgRecorder()
        with mock.patch.object(Spectral_Start, "AMG_COARSEST_LIMIT", 1), \
                mock.patch.object(scipy.sparse.linalg, "lobpcg", side_effect=recorder):
            axes, log = solve(2, seed=5, verbose=True)
        self.assertEqual(recorder.preconditioned(), [False])
        self.assertIn("coarsening stalled", log)
        np.testing.assert_array_equal(axes, AmgFallbackTests.without_pyamg())

    def test_the_hierarchy_and_preconditioner_are_released_before_returning(self):
        hierarchies, preconditioners = [], []
        real_hierarchy, real_preconditioner = (pyamg.smoothed_aggregation_solver,
                                               Spectral_Start.amg_preconditioner)

        def tracked_hierarchy(*args, **kwargs):
            hierarchy = real_hierarchy(*args, **kwargs)
            hierarchies.append(weakref.ref(hierarchy))
            return hierarchy

        def tracked_preconditioner(*args, **kwargs):
            operator = real_preconditioner(*args, **kwargs)
            preconditioners.append(weakref.ref(operator))
            return operator

        with mock.patch.object(pyamg, "smoothed_aggregation_solver", side_effect=tracked_hierarchy), \
                mock.patch.object(Spectral_Start, "amg_preconditioner",
                                  side_effect=tracked_preconditioner):
            solve(2)
        # Released by reference counting as the solve returns: no gc.collect().
        self.assertEqual([len(hierarchies), len(preconditioners)], [1, 1])
        self.assertIsNone(hierarchies[0]())
        self.assertIsNone(preconditioners[0]())


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

    def test_l1_jacobi_weights_are_inverse_absolute_row_sums(self):
        laplacian, _ = Spectral_Start.normalized_laplacian(*GRAPH)
        csr = laplacian.tocsr()
        expected = 1.0 / np.asarray(abs(csr).sum(axis=1)).ravel()
        np.testing.assert_allclose(Spectral_Start._inverse_absolute_row_sums(csr), expected,
                                   rtol=1e-15, atol=0)
        with mock.patch.object(Spectral_Start, "NUMBA_AVAILABLE", False), \
                mock.patch.object(Spectral_Start, "_absolute_row_sums",
                                  side_effect=AssertionError("the Numba loop ran")):
            np.testing.assert_allclose(Spectral_Start._inverse_absolute_row_sums(csr), expected,
                                       rtol=1e-15, atol=0)


if __name__ == "__main__":
    unittest.main()
