"""Alignment score kernels shared through utilities.Network_Kernels.

The traceback references below restate the score, path length and identity
definitions of the global and local kernels. The shared kernels must reproduce
them, and every tool that aligns embeddings must call those same kernel
objects. The batched microbatch kernels are covered in
test_network_kernels_batch.
"""

import builtins
import importlib.util
import io
import os
import sys
import unittest
import warnings
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
for directory in (SRC_DIR, TOOLS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from utilities import Network_Kernels as network_kernels  # noqa: E402
from utilities.Network_Kernels import (  # noqa: E402
    global_local_scores,
    global_score_length_identity,
    local_score_length_identity,
)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_SSEARCH as embedding_ssearch
    import Network_Injection as network_injection


def reference_global_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
):
    num_rows, num_cols = score_matrix.shape
    scores = np.zeros((num_rows + 1, num_cols + 1), dtype=np.float32)
    pointers = np.zeros((num_rows + 1, num_cols + 1), dtype=np.int8)

    for col in range(1, num_cols + 1):
        scores[0, col] = col * gap_penalty
        pointers[0, col] = 3
    for row in range(1, num_rows + 1):
        scores[row, 0] = row * gap_penalty
        pointers[row, 0] = 2

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            match = (
                scores[row - 1, col - 1]
                + score_matrix[row - 1, col - 1]
            )
            delete = scores[row - 1, col] + gap_penalty
            insert = scores[row, col - 1] + gap_penalty
            best_score = match
            pointer = 1
            if delete > best_score:
                best_score = delete
                pointer = 2
            if insert > best_score:
                best_score = insert
                pointer = 3
            scores[row, col] = best_score
            pointers[row, col] = pointer

    row, col = num_rows, num_cols
    path_length = 0
    identities = 0
    while row > 0 or col > 0:
        pointer = pointers[row, col]
        path_length += 1
        if pointer == 1:
            identities += int(query_codes[row - 1] == target_codes[col - 1])
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return scores[num_rows, num_cols], path_length, identities


def reference_local_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
    score_shift=2.0,
):
    shifted_matrix = score_matrix.copy()
    shifted_matrix -= score_shift
    num_rows, num_cols = shifted_matrix.shape
    scores = np.zeros((num_rows + 1, num_cols + 1), dtype=np.float32)
    pointers = np.zeros((num_rows + 1, num_cols + 1), dtype=np.int8)
    max_score = 0.0
    max_position = (0, 0)

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            match = (
                scores[row - 1, col - 1]
                + shifted_matrix[row - 1, col - 1]
            )
            delete = scores[row - 1, col] + gap_penalty
            insert = scores[row, col - 1] + gap_penalty
            best_score = 0.0
            pointer = 0
            if match > best_score:
                best_score = match
                pointer = 1
            if delete > best_score:
                best_score = delete
                pointer = 2
            if insert > best_score:
                best_score = insert
                pointer = 3
            scores[row, col] = best_score
            pointers[row, col] = pointer
            if best_score > max_score:
                max_score = best_score
                max_position = (row, col)

    row, col = max_position
    path_length = 0
    identities = 0
    while row > 0 and col > 0 and scores[row, col] != 0:
        pointer = pointers[row, col]
        if pointer == 0:
            break
        path_length += 1
        if pointer == 1:
            identities += int(query_codes[row - 1] == target_codes[col - 1])
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return max_score, path_length, identities


def residue_codes(rng, length):
    # A three-letter alphabet makes identical pairs common on random paths.
    return rng.integers(0, 3, size=length).astype(np.uint8)


class AlignmentScoreKernelTests(unittest.TestCase):
    def test_shared_kernels_match_traceback_references(self):
        rng = np.random.default_rng(4815)
        cases = [
            np.zeros((1, 1), dtype=np.float32),
            np.full((2, 3), 3.0, dtype=np.float32),
            rng.normal(size=(3, 5)).astype(np.float32),
            rng.normal(size=(7, 2)).astype(np.float32),
            rng.normal(size=(8, 9)).astype(np.float32),
        ]
        # More shapes exercise gaps on both sides of many selected paths.
        for rows, cols in rng.integers(1, 13, size=(20, 2)):
            cases.append(rng.normal(size=(rows, cols)).astype(np.float32))

        for matrix in cases:
            query_codes = residue_codes(rng, matrix.shape[0])
            target_codes = residue_codes(rng, matrix.shape[1])
            for global_gap in (0.0, -1.0):
                expected_global = reference_global_score_length_identity(
                    matrix,
                    global_gap,
                    query_codes,
                    target_codes,
                )
                actual_global = global_score_length_identity(
                    matrix,
                    global_gap,
                    query_codes,
                    target_codes,
                )
                self.assertAlmostEqual(
                    float(actual_global[0]),
                    float(expected_global[0]),
                    places=5,
                )
                self.assertEqual(
                    [int(value) for value in actual_global[1:]],
                    list(expected_global[1:]),
                )

            for local_gap in (-1.0, -2.0, -4.0):
                expected_local = reference_local_score_length_identity(
                    matrix,
                    local_gap,
                    query_codes,
                    target_codes,
                )
                actual_local = local_score_length_identity(
                    matrix,
                    local_gap,
                    query_codes,
                    target_codes,
                )
                self.assertAlmostEqual(
                    float(actual_local[0]),
                    float(expected_local[0]),
                    places=5,
                )
                self.assertEqual(
                    [int(value) for value in actual_local[1:]],
                    list(expected_local[1:]),
                )

    def test_fused_kernel_matches_individual_kernels(self):
        rng = np.random.default_rng(20260729)
        matrix = rng.normal(size=(11, 13)).astype(np.float32)
        query_codes = residue_codes(rng, 11)
        target_codes = residue_codes(rng, 13)

        expected_global = global_score_length_identity(
            matrix, 0.0, query_codes, target_codes
        )
        expected_local = local_score_length_identity(
            matrix, -2.0, query_codes, target_codes
        )
        actual = global_local_scores(matrix, 0.0, -2.0)

        self.assertAlmostEqual(
            float(actual[0]),
            float(expected_global[0]),
            places=5,
        )
        self.assertEqual(int(actual[1]), int(expected_global[1]))
        self.assertAlmostEqual(
            float(actual[2]),
            float(expected_local[0]),
            places=5,
        )
        self.assertEqual(int(actual[3]), int(expected_local[1]))

    def test_callers_share_the_same_kernel_objects(self):
        self.assertIs(
            similarity_matrix.global_local_scores,
            global_local_scores,
        )
        self.assertIs(
            network_injection.global_local_scores,
            global_local_scores,
        )
        self.assertIs(
            embedding_ssearch.global_score_length_identity,
            global_score_length_identity,
        )
        self.assertIs(
            embedding_ssearch.local_score_length_identity,
            local_score_length_identity,
        )


BARBELL = np.asarray(
    [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)], dtype=np.int32
)


class PanicLikeError(BaseException):
    """Stands in for pyo3's PanicException, which is no Exception."""


def fake_graspologic(leiden):
    return SimpleNamespace(leiden=leiden)


class LeidenPartitionTests(unittest.TestCase):
    """leiden_partition keeps graspologic_native's panics away from its callers."""

    PATH = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    def partition(self, weights, resolution, native):
        with mock.patch.dict(sys.modules, {"graspologic_native": native}):
            return network_kernels.leiden_partition(
                3, self.PATH, weights, resolution, 1
            )

    def test_a_nan_or_infinite_resolution_is_refused_before_the_native_call(self):
        for resolution in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(resolution=resolution):
                native = fake_graspologic(mock.Mock())
                with self.assertRaises(ValueError) as caught:
                    self.partition(None, resolution, native)
                self.assertEqual(
                    str(caught.exception),
                    f"Leiden resolution must be a finite number; got {resolution}.",
                )
                native.leiden.assert_not_called()

    def test_nan_or_infinite_weights_are_refused_before_the_native_call(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(weight=bad):
                native = fake_graspologic(mock.Mock())
                with self.assertRaises(ValueError) as caught:
                    self.partition(np.asarray([1.0, bad]), 1.0, native)
                self.assertEqual(
                    str(caught.exception), "Leiden edge weights must be finite numbers."
                )
                native.leiden.assert_not_called()

    def test_a_network_without_edges_has_singletons_for_a_min_size_of_one_else_noise(self):
        native = fake_graspologic(mock.Mock())
        for min_size, expected in ((1, [1, 2, 3]), (2, [-1, -1, -1]), (10, [-1, -1, -1])):
            with self.subTest(min_size=min_size):
                with mock.patch.dict(sys.modules, {"graspologic_native": native}):
                    labels = network_kernels.leiden_partition(
                        3, np.zeros((0, 2), dtype=np.int32), None, 1.0, min_size
                    )
                np.testing.assert_array_equal(labels, expected)
        native.leiden.assert_not_called()

    def test_isolated_nodes_are_singleton_clusters_only_when_min_size_is_one(self):
        # Nodes 0 and 4 have no edge; nodes 1-3 form one community. The native
        # library returns only the nodes that appear in an edge.
        native = fake_graspologic(mock.Mock(return_value=(0.5, {"1": 0, "2": 0, "3": 0})))
        edges = np.asarray([[1, 2], [2, 3]], dtype=np.int32)
        for min_size, expected in (
            # Singletons follow the clusters the library found, in node order.
            (1, [2, 1, 1, 1, 3]),
            (2, [-1, 1, 1, 1, -1]),
            (3, [-1, 1, 1, 1, -1]),
            # The community itself is below min_size too.
            (4, [-1, -1, -1, -1, -1]),
        ):
            with self.subTest(min_size=min_size):
                with mock.patch.dict(sys.modules, {"graspologic_native": native}):
                    labels = network_kernels.leiden_partition(5, edges, None, 1.0, min_size)
                np.testing.assert_array_equal(labels, expected)

    def test_a_node_with_only_a_self_loop_is_not_isolated(self):
        # The library gives it a community of its own, so min_size decides as
        # for any community of one.
        native = fake_graspologic(mock.Mock(return_value=(0.1, {"0": 0, "1": 0, "2": 1})))
        edges = np.asarray([[0, 1], [2, 2]], dtype=np.int32)
        for min_size, expected in ((1, [1, 1, 2, 3]), (2, [1, 1, -1, -1])):
            with self.subTest(min_size=min_size):
                with mock.patch.dict(sys.modules, {"graspologic_native": native}):
                    labels = network_kernels.leiden_partition(4, edges, None, 1.0, min_size)
                np.testing.assert_array_equal(labels, expected)

    def test_a_panic_becomes_a_runtime_error(self):
        native = fake_graspologic(mock.Mock(side_effect=PanicLikeError("boom")))
        with self.assertRaises(RuntimeError) as caught:
            self.partition(None, 1.0, native)
        self.assertEqual(str(caught.exception), "Leiden clustering failed: boom")
        self.assertIsInstance(caught.exception.__cause__, PanicLikeError)

    def test_exceptions_and_interrupts_pass_through_unchanged(self):
        for error in (
            ValueError("range"),
            KeyboardInterrupt(),
            SystemExit(2),
            GeneratorExit(),
        ):
            with self.subTest(error=type(error).__name__):
                native = fake_graspologic(mock.Mock(side_effect=error))
                with self.assertRaises(type(error)) as caught:
                    self.partition(None, 1.0, native)
                self.assertIs(caught.exception, error)

    def test_the_csr_call_is_guarded_too(self):
        csr = mock.Mock(side_effect=PanicLikeError("csr boom"))
        native = SimpleNamespace(leiden=mock.Mock(), leiden_csr=csr)
        with self.assertRaises(RuntimeError):
            self.partition(np.asarray([1.0, 2.0]), 1.0, native)
        csr.assert_called_once()

    def test_valid_input_gives_the_same_partitions(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")
        cases = (
            # Node 6 has no edge: a singleton with min_size 1, Noise above it.
            (None, 1.0, 1, [1, 1, 1, 2, 2, 2, 3]),
            (np.arange(1.0, 8.0), 1.0, 1, [1, 1, 1, 2, 2, 2, 3]),
            (np.arange(1.0, 8.0), 0.3, 1, [1, 1, 1, 1, 1, 1, 2]),
            (None, 1.0, 2, [1, 1, 1, 2, 2, 2, -1]),
            (None, 2.0, 3, [1, 1, 1, 2, 2, 2, -1]),
        )
        for weights, resolution, min_size, labels in cases:
            with self.subTest(resolution=resolution, min_size=min_size):
                actual = network_kernels.leiden_partition(
                    7, BARBELL, weights, resolution, min_size
                )
                np.testing.assert_array_equal(actual, labels)

    def test_the_real_library_panic_on_zero_weights_becomes_a_runtime_error(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")
        with self.assertRaises(RuntimeError) as caught:
            network_kernels.leiden_partition(
                3, self.PATH, np.zeros(2), 1.0, 1
            )
        self.assertTrue(str(caught.exception).startswith("Leiden clustering failed: "))


def load_without_numba():
    """A copy of Network_Kernels imported as it is when Numba is missing."""
    path = os.path.join(SRC_DIR, "utilities", "Network_Kernels.py")
    spec = importlib.util.spec_from_file_location("Network_Kernels_without_numba", path)
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def refuse_numba(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("numba is blocked for this test")
        return real_import(name, *args, **kwargs)

    sys.modules[spec.name] = module
    try:
        with mock.patch("builtins.__import__", refuse_numba):
            spec.loader.exec_module(module)
    finally:
        # The copy registers itself under the module's other names too.
        for name in [name for name, loaded in sys.modules.items() if loaded is module]:
            del sys.modules[name]
    return module


class JaccardPartitionWithoutNumbaTests(unittest.TestCase):
    def test_the_pure_python_fallback_matches_the_numba_kernel(self):
        plain = load_without_numba()
        self.assertFalse(plain.NUMBA_AVAILABLE)
        for threshold in (0.0, 0.2, 0.34, 0.5, 0.76, 1.0):
            for min_size in (1, 3):
                with self.subTest(threshold=threshold, min_size=min_size):
                    np.testing.assert_array_equal(
                        plain.jaccard_partition(7, BARBELL, threshold, min_size),
                        network_kernels.jaccard_partition(7, BARBELL, threshold, min_size),
                    )


def closed_jaccard_counts(n_nodes, edges, multiset=False):
    """(shared, union) sizes of the closed neighbourhoods at each edge's ends.

    Plain Python from the definition: N[x] is x's neighbours and x itself.
    The pure-Python kernel treats a neighbourhood as a set. The Numba kernel
    reads a CSR row as stored, so a repeated pair or a self-loop (two
    half-edges) repeats an entry; ``multiset=True`` counts those, and a node
    is added to its own neighbourhood only when its row lacks it.
    """
    rows = [[] for _ in range(n_nodes)]
    for u, v in edges.tolist():
        rows[u].append(v)
        rows[v].append(u)
    counts = []
    for u, v in edges.tolist():
        if multiset:
            closed_u, closed_v = Counter(rows[u]), Counter(rows[v])
            closed_u[u] = max(closed_u[u], 1)
            closed_v[v] = max(closed_v[v], 1)
            shared = sum((closed_u & closed_v).values())
            union = sum(closed_u.values()) + sum(closed_v.values()) - shared
        else:
            closed_u, closed_v = set(rows[u]) | {u}, set(rows[v]) | {v}
            shared, union = len(closed_u & closed_v), len(closed_u | closed_v)
        counts.append((shared, union))
    return counts


def random_edges(rng, n_nodes, n_pairs, simple):
    """n_pairs random pairs; simple=True drops self-loops and repeated pairs."""
    pairs = rng.integers(0, n_nodes, (n_pairs, 2))
    if simple:
        pairs = sorted({tuple(sorted(pair)) for pair in pairs.tolist() if pair[0] != pair[1]})
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


class ClosedNeighbourhoodJaccardTests(unittest.TestCase):
    """The Jaccard filter compares closed neighbourhoods N[x] = N(x) + {x}, so an
    edge scores (c + 2) / (a + b - c) for a = |N(u)|, b = |N(v)| and c shared
    neighbours: an edge outside every triangle scores above 0, and an isolated
    pair and every edge of a clique score 1."""

    @classmethod
    def setUpClass(cls):
        # (name, module, whether it counts a CSR row as a multiset)
        cls.kernels = (
            ("default", network_kernels, network_kernels.NUMBA_AVAILABLE),
            ("without numba", load_without_numba(), False),
        )

    def keep(self, kernel, n_nodes, edges, threshold):
        edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        indptr, indices = kernel.sorted_neighbour_csr(edges, n_nodes)
        return kernel.fast_jaccard_filter(edges, indptr, indices, threshold)

    def assert_matches(self, kernel, n_nodes, edges, counts):
        """The kernel keeps each edge exactly at its index and above: at every
        index, just above every index, and at 0 and 1."""
        indices = {shared / union for shared, union in counts}
        thresholds = sorted(
            indices | {float(np.nextafter(index, 2)) for index in indices} | {0.0, 1.0}
        )
        for threshold in thresholds:
            expected = [shared / union >= threshold for shared, union in counts]
            np.testing.assert_array_equal(
                self.keep(kernel, n_nodes, edges, threshold), expected,
                err_msg=f"threshold {threshold}",
            )

    def test_edges_outside_every_triangle_score_above_zero(self):
        # n_nodes, edges, and each edge's (shared, union) closed neighbours.
        cases = {
            # Both closed neighbourhoods are {0, 1}.
            "isolated pair": (2, [(0, 1)], [(2, 2)]),
            # A leaf {leaf, hub} against its hub of degree 3 {hub, 3 leaves}.
            "star": (4, [(0, 1), (0, 2), (0, 3)], [(2, 4)] * 3),
            # The end edges of 0-1-2-3 share 2 of 3 nodes, the middle one 2 of 4.
            "path": (4, [(0, 1), (1, 2), (2, 3)], [(2, 3), (2, 4), (2, 3)]),
            "triangle with a tail": (
                4, [(0, 1), (0, 2), (1, 2), (2, 3)], [(3, 3), (3, 4), (3, 4), (2, 4)],
            ),
            "4-cycle": (4, [(0, 1), (1, 2), (2, 3), (3, 0)], [(2, 4)] * 4),
        }
        for label, (n_nodes, edges, expected) in cases.items():
            edges = np.asarray(edges, dtype=np.int32)
            for name, kernel, _multiset in self.kernels:
                with self.subTest(graph=label, kernel=name):
                    self.assertEqual(closed_jaccard_counts(n_nodes, edges), expected)
                    self.assert_matches(kernel, n_nodes, edges, expected)

    def test_a_clique_scores_one_at_every_size(self):
        for size in range(2, 8):
            edges = [(u, v) for u in range(size) for v in range(u + 1, size)]
            for name, kernel, _multiset in self.kernels:
                with self.subTest(size=size, kernel=name):
                    np.testing.assert_array_equal(
                        self.keep(kernel, size, edges, 1.0), [True] * len(edges)
                    )

    def test_the_formula_holds_for_simple_graphs(self):
        rng = np.random.default_rng(5)
        for _trial in range(30):
            n_nodes = int(rng.integers(2, 25))
            edges = random_edges(rng, n_nodes, int(rng.integers(1, 70)), simple=True)
            neighbours = [set() for _ in range(n_nodes)]
            for u, v in edges.tolist():
                neighbours[u].add(v)
                neighbours[v].add(u)
            expected = []
            for u, v in edges.tolist():
                shared = len(neighbours[u] & neighbours[v])
                expected.append(
                    (shared + 2, len(neighbours[u]) + len(neighbours[v]) - shared)
                )
            self.assertEqual(closed_jaccard_counts(n_nodes, edges), expected)

    def test_both_kernels_match_a_brute_force_on_random_graphs(self):
        rng = np.random.default_rng(11)
        for trial in range(40):
            n_nodes = int(rng.integers(2, 30))
            n_pairs = int(rng.integers(1, 90))
            # Simple graphs, and graphs with self-loops and repeated pairs
            # (an edge in both orientations among them).
            for simple in (True, False):
                edges = random_edges(rng, n_nodes, n_pairs, simple)
                if not len(edges):
                    continue
                for name, kernel, multiset in self.kernels:
                    counts = closed_jaccard_counts(n_nodes, edges, multiset)
                    with self.subTest(trial=trial, simple=simple, kernel=name):
                        self.assert_matches(kernel, n_nodes, edges, counts)

    def test_a_self_loop_does_not_count_the_node_twice(self):
        # Node 0 has a loop beside the triangle's three edges.
        edges = np.asarray([(0, 0), (0, 1), (0, 2), (1, 2)], dtype=np.int32)
        for name, kernel, multiset in self.kernels:
            if multiset:
                # The loop is the two entries of 0 its CSR row holds, and no
                # third is added: 0 and 1 share 3 of the 4 entries.
                expected = [(4, 4), (3, 4), (3, 4), (3, 3)]
            else:
                # N[0] = N[1] = N[2] = {0, 1, 2}, with the loop or without.
                expected = [(3, 3)] * 4
            with self.subTest(kernel=name):
                self.assertEqual(closed_jaccard_counts(3, edges, multiset), expected)
                self.assert_matches(kernel, 3, edges, expected)


class ClosedNeighbourhoodPartitionTests(unittest.TestCase):
    """jaccard_partition keeps pairs, stars and chains: they are not Noise."""

    # An isolated pair, a star with three leaves, a chain of four, a lone node.
    EDGES = np.asarray(
        [(0, 1), (2, 3), (2, 4), (2, 5), (6, 7), (7, 8), (8, 9)], dtype=np.int32
    )

    def groups(self, labels):
        """The labelled nodes' groups, noise (-1) set aside."""
        by_label = {}
        for node, label in enumerate(labels.tolist()):
            by_label.setdefault(label, set()).add(node)
        by_label.pop(-1, None)
        return list(by_label.values())

    def test_pairs_stars_and_chains_are_not_noise(self):
        # The pair scores 1, the star's spokes 1/2, the chain's end edges 2/3
        # and its middle edge 1/2.
        for threshold in (0.0, 0.2, 0.5):
            labels = network_kernels.jaccard_partition(11, self.EDGES, threshold, 2)
            with self.subTest(threshold=threshold):
                self.assertCountEqual(
                    self.groups(labels), [{0, 1}, {2, 3, 4, 5}, {6, 7, 8, 9}]
                )
                self.assertEqual(labels[10], -1)

    def test_a_threshold_above_the_scores_cuts_the_edges(self):
        cases = [
            (0.6, [{0, 1}, {6, 7}, {8, 9}]),
            (0.7, [{0, 1}]),
            (1.0, [{0, 1}]),
        ]
        for threshold, groups in cases:
            labels = network_kernels.jaccard_partition(11, self.EDGES, threshold, 2)
            with self.subTest(threshold=threshold):
                self.assertCountEqual(self.groups(labels), groups)

    def test_both_kernels_label_alike(self):
        plain = load_without_numba()
        for threshold in (0.0, 0.5, 0.6, 0.7, 1.0):
            for min_size in (1, 2):
                with self.subTest(threshold=threshold, min_size=min_size):
                    np.testing.assert_array_equal(
                        plain.jaccard_partition(11, self.EDGES, threshold, min_size),
                        network_kernels.jaccard_partition(11, self.EDGES, threshold, min_size),
                    )


def mcl_adjacency(n_nodes, edges, weights):
    """The canonical CSR the cluster commands build from an edge list."""
    import scipy.sparse as sp

    edges = np.asarray(edges).reshape(-1, 2)
    weights = np.asarray(weights, dtype=float)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    return sp.csr_matrix(
        (np.concatenate([weights, weights]), (rows, columns)), shape=(n_nodes, n_nodes)
    )


class MCLSelfLoopTests(unittest.TestCase):
    """MCL gives every node a self-loop as heavy as its strongest edge, so the
    clusters do not depend on the scale of the scores."""

    EDGES = [(0, 1), (0, 2), (1, 2), (2, 3)]
    WEIGHTS = [1.5, 2.0, 3.0, 0.5]

    def test_each_loop_takes_the_nodes_strongest_edge(self):
        # Node 4 has no edge: its loop weighs 1.
        matrix = mcl_adjacency(5, self.EDGES, self.WEIGHTS)
        looped = network_kernels._mcl_add_self_loops(matrix)
        dense = looped.toarray()
        np.testing.assert_array_equal(np.diag(dense), [2.0, 3.0, 3.0, 0.5, 1.0])
        # Every other entry is as it was, and each loop is one entry.
        np.fill_diagonal(dense, 0)
        np.testing.assert_array_equal(dense, matrix.toarray())
        self.assertEqual(looped.nnz, matrix.nnz + 5)

    def test_unweighted_loops_weigh_one_as_before(self):
        matrix = mcl_adjacency(4, self.EDGES, np.ones(4))
        looped = network_kernels._mcl_add_self_loops(matrix)
        np.testing.assert_array_equal(np.diag(looped.toarray()), [1.0] * 4)

    def test_an_existing_loop_is_replaced_and_is_not_the_maximum(self):
        import scipy.sparse as sp

        dense = np.array(
            [
                [7.0, 2.0, 0.0, 0.0],
                [2.0, 9.0, 3.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0],  # a loop and no edge
            ]
        )
        looped = network_kernels._mcl_add_self_loops(sp.csr_matrix(dense))
        # The loops 7 and 9 are neither kept nor added to; node 3 has no edge.
        np.testing.assert_array_equal(np.diag(looped.toarray()), [2.0, 3.0, 3.0, 1.0])
        # Four edge entries and one loop per node.
        self.assertEqual(looped.nnz, 8)

    def test_the_loops_keep_the_weight_dtype(self):
        matrix = mcl_adjacency(5, self.EDGES, self.WEIGHTS).astype(np.float32)
        looped = network_kernels._mcl_add_self_loops(matrix)
        self.assertEqual(looped.dtype, np.float32)
        np.testing.assert_array_equal(np.diag(looped.toarray()), [2.0, 3.0, 3.0, 0.5, 1.0])

    def clusters(self, n_nodes, edges, weights, inflation=2.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return network_kernels.markov_clusters(
                mcl_adjacency(n_nodes, edges, weights), inflation
            )

    def test_multiplying_every_weight_by_a_constant_keeps_the_clusters(self):
        rng = np.random.default_rng(3)
        for trial in range(15):
            n_nodes = int(rng.integers(5, 40))
            n_pairs = int(rng.integers(n_nodes, 4 * n_nodes))
            edges = random_edges(rng, n_nodes, n_pairs, simple=True)
            weights = rng.uniform(0.2, 5.0, len(edges))
            for inflation in (1.5, 2.0, 4.0):
                expected = self.clusters(n_nodes, edges, weights, inflation)
                for scale in (0.01, 0.5, 3.0, 250.0):
                    with self.subTest(trial=trial, inflation=inflation, scale=scale):
                        self.assertEqual(
                            self.clusters(n_nodes, edges, weights * scale, inflation),
                            expected,
                        )

    def test_identity_scores_and_log_e_value_scores_cluster_alike(self):
        # The path 0-...-5 scored on the 0-1 scale of identities and on a
        # scale 300 times larger, like -log10 of E-values. Loops fixed at 1
        # barely mattered on the second and let the walk swing between the
        # path's even and odd nodes.
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        identity = np.array([0.9, 0.8, 0.7, 0.8, 0.9])
        expected = [(0, 1, 2), (3, 4, 5)]
        self.assertEqual(self.clusters(6, edges, identity), expected)
        self.assertEqual(self.clusters(6, edges, identity * 300.0), expected)


if __name__ == "__main__":
    unittest.main()
