"""Chain fill (utilities/Chain_Fill.py): the missing pairs of a sparse network estimated from chains of computed
pairs, and its use by Embedding_MSA when Calibrated Imputation is on and a calibration sample was used.

The references are written here in plain numpy from the method's formulas: the chain value of a pair is the
maximum over every k with both (i, k) and (k, j) computed of min(y_ik, y_kj); the fill is
c0 + c1 * base + c2 * (chain if covered else base) + c3 * [uncovered].
"""

import contextlib
import glob
import io
import pathlib
import re
import tempfile
import tracemalloc
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.distance import squareform
from sklearn.isotonic import IsotonicRegression

from tests.msa_fixtures import (
    GuideTreeBuilt,
    InProcessPool,
    NetworkFixture,
    builder_settings,
    fake_calibration_alignment,
)
import Embedding_MSA  # noqa: E402  msa_fixtures puts src/tools on the path
from utilities import Chain_Fill  # noqa: E402
from utilities import Numba_Threads  # noqa: E402
from utilities import Prefilter_Calibration  # noqa: E402

NEG = Chain_Fill.NEG
HALF = Chain_Fill.HALF_NEG
GIB = 1024 ** 3


def bits(values):
    return np.ascontiguousarray(values, dtype=np.float32).view(np.int32)


def brute_force_chain(square, rows, cols):
    """max over k of min(square[i, k], square[k, j]) by a full scan, NEG where no k has both computed."""
    out = np.full(len(rows), NEG, dtype=np.float32)
    for t, (i, j) in enumerate(zip(rows, cols)):
        best = NEG
        for k in range(square.shape[0]):
            a, b = square[i, k], square[k, j]
            value = a if a < b else b
            if value > best:
                best = value
        out[t] = best
    return out


def random_network(n, density, seed, isolated=()):
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, 1)
    keep = rng.random(rows.size) < density
    for node in isolated:
        keep &= (rows != node) & (cols != node)
    edge_i = rows[keep].astype(np.int32)
    edge_j = cols[keep].astype(np.int32)
    # Scores with repeats, so ties exercise the walk's stop rule.
    y = rng.integers(0, 12, edge_i.size).astype(np.float32) / 4.0
    return edge_i, edge_j, y


def one_thread():
    return mock.patch.object(Chain_Fill.Numba_Threads, "default_thread_count", return_value=1)


def many_threads():
    import numba

    return mock.patch.object(
        Chain_Fill.Numba_Threads, "default_thread_count", return_value=numba.config.NUMBA_NUM_THREADS
    )


class ChainKernelTests(unittest.TestCase):
    def test_every_pair_matches_a_full_scan_bit_for_bit_at_several_densities_and_thread_counts(self):
        rows, cols = np.triu_indices(45, 1)
        for density in (0.0, 0.03, 0.1, 0.3, 0.7, 1.0):
            edge_i, edge_j, y = random_network(45, density, seed=int(density * 100), isolated=(7, 20))
            square = Chain_Fill.score_matrix(45, edge_i, edge_j, y)
            expected = brute_force_chain(square, rows, cols)
            index = Chain_Fill.build_index(square)
            for threads in (one_thread, many_threads):
                with self.subTest(density=density, threads=threads.__name__):
                    with threads():
                        found = Chain_Fill.chain_values(square, index, rows, cols)
                    self.assertEqual(found.dtype, np.float32)
                    self.assertTrue(np.array_equal(bits(found), bits(expected)))

    def test_an_isolated_sequence_and_a_pair_without_a_common_neighbour_are_uncovered(self):
        # 0-1, 1-2 and 3-4; sequence 5 has no pair at all.
        edge_i = np.array([0, 1, 3], dtype=np.int32)
        edge_j = np.array([1, 2, 4], dtype=np.int32)
        y = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        square = Chain_Fill.score_matrix(6, edge_i, edge_j, y)
        index = Chain_Fill.build_index(square)
        found = Chain_Fill.chain_values(square, index, [0, 0, 0, 3, 5, 2], [2, 3, 5, 5, 4, 5])
        self.assertEqual(found[0], np.float32(2.0))  # 0 - 1 - 2: min(2, 3)
        for position in range(1, 6):
            self.assertEqual(found[position], NEG, position)
        self.assertEqual(index.start.tolist(), [0, 1, 3, 4, 5, 6, 6])  # sequence 5 has an empty list

    def test_a_computed_pairs_own_score_never_routes_through_its_own_ends(self):
        # The only computed pair is (0, 1): k = 0 and k = 1 give "not computed", so it has no chain.
        square = Chain_Fill.score_matrix(3, np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
                                         np.array([5.0], dtype=np.float32))
        index = Chain_Fill.build_index(square)
        self.assertEqual(Chain_Fill.chain_values(square, index, [0], [1])[0], NEG)
        self.assertTrue(np.all(np.diag(square) == NEG))

    def test_a_pair_listed_twice_keeps_the_last_score_as_the_trees_baseline_does(self):
        n = 6
        edge_i = np.array([0, 1, 2, 3, 1, 3, 4], dtype=np.int32)
        edge_j = np.array([1, 2, 3, 4, 0, 2, 5], dtype=np.int32)  # (1, 0) repeats (0, 1); (3, 2) repeats (2, 3)
        y = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 7.5, 6.0], dtype=np.float32)
        square = Chain_Fill.score_matrix(n, edge_i, edge_j, y)
        baseline = np.full(n * (n - 1) // 2, -1.0, dtype=np.float32)
        Embedding_MSA.populate_condensed_matrix(baseline, n, edge_i, edge_j, y)
        rows, cols = np.triu_indices(n, 1)
        listed = baseline != -1.0
        self.assertTrue(np.array_equal(bits(square[rows, cols][listed]), bits(baseline[listed])))
        self.assertEqual(square[0, 1], np.float32(9.0))
        self.assertEqual(square[1, 0], np.float32(9.0))
        self.assertEqual(square[2, 3], np.float32(7.5))
        self.assertEqual(int(np.count_nonzero(square[rows, cols] > HALF)), int(listed.sum()))


def hand_network(seed=0, n=12, observed=36, calibration=14, asymmetric=False):
    """A network of n sequences written by hand: observed pairs, a calibration sample, a similarity matrix."""
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, 1)
    order = rng.permutation(rows.size)
    obs, cal = order[:observed], order[observed:observed + calibration]
    similarity = rng.random((n, n))
    if not asymmetric:
        similarity = np.triu(similarity, 1)
        similarity = similarity + similarity.T
    edge_i = np.concatenate([rows[obs], rows[cal]]).astype(np.int32)
    edge_j = np.concatenate([cols[obs], cols[cal]]).astype(np.int32)
    # Scores follow the similarity loosely, with noise.
    y = (3.0 * similarity[edge_i, edge_j] + rng.normal(0.0, 0.5, edge_i.size)).astype(np.float32)
    positions = np.arange(observed, observed + calibration)
    weights = rng.uniform(1.0, 6.0, calibration)
    x = similarity[edge_i, edge_j]
    imputation = Embedding_MSA.fit_imputation(x, y, positions, weights)
    return dict(n=n, edge_i=edge_i, edge_j=edge_j, y=y, similarity=similarity, positions=positions,
                weights=weights, x=x, imputation=imputation, max_norm=float(y.max()))


def reference_out_of_fold(case):
    """Each calibration pair predicted by an isotonic fit without its half (even or odd positions), by the rules of
    fit_imputation: the other observed pairs at their count weight, the half's pairs at weight * size / half."""
    imputation = case["imputation"]
    x, y = case["x"], case["y"]
    positions = case["positions"]
    mask = np.ones(x.size, dtype=bool)
    mask[positions] = False
    fit_x, fit_y = x[mask], y[mask]
    other_weight = fit_x.size / fit_x.size  # every other pair is taken: count / taken = 1
    cal_x, cal_y, weights = x[positions], y[positions], case["weights"]
    out = np.empty(cal_x.size)
    for chosen, predicted in ((np.arange(0, cal_x.size, 2), np.arange(1, cal_x.size, 2)),
                              (np.arange(1, cal_x.size, 2), np.arange(0, cal_x.size, 2))):
        scale = len(weights) / len(chosen)
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.concatenate([fit_x, cal_x[chosen]]), np.concatenate([fit_y, cal_y[chosen]]),
            sample_weight=np.concatenate([np.full(fit_x.size, other_weight), weights[chosen] * scale]),
        )
        out[predicted] = iso.predict(cal_x[predicted])
    return out


def expected_missing_distances(case, out_of_fold, coefficients=None):
    """The dense layout the tool's own inversion and symmetrising give, in plain numpy; returns (condensed, coefficients, covered)."""
    n = case["n"]
    square = np.full((n, n), NEG, dtype=np.float32)
    for i, j, value in zip(case["edge_i"], case["edge_j"], case["y"]):  # edge order: the last listing wins
        square[i, j] = square[j, i] = value
    chain = np.full((n, n), NEG, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                if square[i, k] > HALF and square[k, j] > HALF:
                    chain[i, j] = max(chain[i, j], min(square[i, k], square[k, j]))
    positions = case["positions"]
    cal_chain = chain[case["edge_i"][positions], case["edge_j"][positions]].astype(np.float64)
    covered_cal = cal_chain > float(HALF)
    design = np.column_stack([np.ones(positions.size), out_of_fold, np.where(covered_cal, cal_chain, out_of_fold),
                              (~covered_cal).astype(np.float64)])
    root = np.sqrt(case["weights"])
    if coefficients is None:
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], case["y"][positions].astype(np.float64) * root,
                                           rcond=None)
    base = case["imputation"].iso.predict(case["similarity"].ravel()).reshape(n, n)
    covered = chain > HALF
    predicted = (coefficients[0] + coefficients[1] * base
                 + coefficients[2] * np.where(covered, chain.astype(np.float64), base)
                 + coefficients[3] * (~covered))
    dense = np.maximum(0.0, case["max_norm"] - predicted).astype(np.float32)
    np.fill_diagonal(dense, 0.0)
    dense = 0.5 * (dense + dense.T)
    return squareform(dense), coefficients, covered


def condensed_missing(case):
    n = case["n"]
    present = np.zeros((n, n), dtype=bool)
    present[case["edge_i"], case["edge_j"]] = True
    present[case["edge_j"], case["edge_i"]] = True
    rows, cols = np.triu_indices(n, 1)
    return ~present[rows, cols]


class ChainFillTests(unittest.TestCase):
    def check_case(self, case, block_pairs=Chain_Fill.BLOCK_PAIRS):
        out_of_fold = reference_out_of_fold(case)
        self.assertTrue(np.array_equal(case["imputation"].out_of_fold, out_of_fold))
        plan = Chain_Fill.prepare(case["n"], case["edge_i"], case["edge_j"], case["y"], case["positions"],
                                  case["imputation"].out_of_fold, case["imputation"].calibration_weights)
        expected, coefficients, covered = expected_missing_distances(case, out_of_fold)
        self.assertTrue(np.array_equal(plan.coefficients, coefficients))
        n = case["n"]
        condensed = np.full(n * (n - 1) // 2, -5.0, dtype=np.float32)
        found_covered, found_missing = Chain_Fill.fill_distances(
            plan, condensed, case["similarity"], case["imputation"].iso, case["max_norm"], block_pairs=block_pairs)
        missing = condensed_missing(case)
        self.assertEqual(found_missing, int(missing.sum()))
        rows, cols = np.triu_indices(n, 1)
        self.assertEqual(found_covered, int(covered[rows, cols][missing].sum()))
        self.assertTrue(np.array_equal(bits(condensed[missing]), bits(expected[missing])))
        return plan, condensed, missing, covered[rows, cols]

    def test_the_fill_of_a_dozen_sequences_equals_the_formulas_in_plain_numpy_bit_for_bit(self):
        for seed in (0, 1, 2, 3):
            case = hand_network(seed)
            with self.subTest(seed=seed):
                _, _, missing, covered = self.check_case(case)
                self.assertGreater(int(missing.sum()), 10)
                self.assertTrue(covered[missing].any())  # some pairs have a chain

    def test_row_blocks_of_any_size_give_the_same_bits(self):
        case = hand_network(5)
        _, whole, _, _ = self.check_case(case)
        for block_pairs in (1, 7, 25):
            with self.subTest(block_pairs=block_pairs):
                _, blocked, missing, _ = self.check_case(case, block_pairs=block_pairs)
                self.assertTrue(np.array_equal(bits(blocked[missing]), bits(whole[missing])))

    def test_an_asymmetric_similarity_matrix_gives_the_dense_layouts_average_of_both_sides(self):
        for seed in (0, 1, 2):
            case = hand_network(seed, asymmetric=True)
            self.assertFalse(Chain_Fill.is_symmetric(case["similarity"]))
            with self.subTest(seed=seed):
                self.check_case(case)
                self.check_case(case, block_pairs=9)

    def test_symmetry_is_detected_exactly(self):
        matrix = np.random.default_rng(0).random((150, 150))
        matrix = matrix + matrix.T
        self.assertTrue(Chain_Fill.is_symmetric(matrix))
        matrix[149, 3] = np.nextafter(matrix[149, 3], 2.0)
        self.assertFalse(Chain_Fill.is_symmetric(matrix))

    def test_the_fill_does_not_depend_on_the_number_of_threads(self):
        case = hand_network(9, n=40, observed=300, calibration=60)
        results = []
        for threads in (one_thread, many_threads):
            with threads():
                _, condensed, missing, _ = self.check_case(case, block_pairs=100)
            results.append(condensed[missing].tobytes())
        self.assertEqual(results[0], results[1])

    def test_a_calibration_pair_that_is_also_listed_before_it_is_still_resolved_by_the_last_listing(self):
        case = hand_network(4)
        # List one observed pair again with another score, after everything else; the last score is the one used.
        extra = 3
        case["edge_i"] = np.concatenate([case["edge_i"], case["edge_i"][extra:extra + 1]])
        case["edge_j"] = np.concatenate([case["edge_j"], case["edge_j"][extra:extra + 1]])
        case["y"] = np.concatenate([case["y"], np.array([case["y"][extra] + 1.5], dtype=np.float32)])
        plan = Chain_Fill.prepare(case["n"], case["edge_i"], case["edge_j"], case["y"], case["positions"],
                                  case["imputation"].out_of_fold, case["imputation"].calibration_weights)
        i, j = case["edge_i"][extra], case["edge_j"][extra]
        self.assertEqual(plan.square[i, j], case["y"][-1])
        self.assertEqual(plan.square[j, i], case["y"][-1])


class CoefficientTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.count = 40
        self.out_of_fold = rng.normal(2.0, 1.0, self.count)
        self.y = self.out_of_fold * 0.8 + rng.normal(0.0, 0.2, self.count)
        self.weights = rng.uniform(1.0, 5.0, self.count)
        self.chain = (self.y + rng.normal(0.0, 0.1, self.count)).astype(np.float32)

    def reference(self, design):
        root = np.sqrt(self.weights)
        return np.linalg.lstsq(design * root[:, None], self.y * root, rcond=None)[0]

    def test_the_four_columns_are_a_constant_the_base_the_chain_or_the_base_and_the_uncovered_indicator(self):
        chain = self.chain.copy()
        chain[[3, 17]] = NEG
        covered = chain > HALF
        design = np.column_stack([np.ones(self.count), self.out_of_fold,
                                  np.where(covered, chain.astype(np.float64), self.out_of_fold),
                                  (~covered).astype(np.float64)])
        found = Chain_Fill.fit_coefficients(self.out_of_fold, chain, self.y, self.weights)
        self.assertTrue(np.array_equal(found, self.reference(design)))

    def test_when_every_calibration_pair_is_covered_the_indicator_column_is_zero_and_its_coefficient_is_zero(self):
        found = Chain_Fill.fit_coefficients(self.out_of_fold, self.chain, self.y, self.weights)
        design = np.column_stack([np.ones(self.count), self.out_of_fold, self.chain.astype(np.float64)])
        self.assertTrue(np.all(np.isfinite(found)))
        self.assertEqual(found[3], 0.0)  # lstsq's minimum-norm answer, which normal equations would not give
        np.testing.assert_allclose(found[:3], self.reference(design), rtol=1e-9)

    def test_when_no_calibration_pair_is_covered_the_fill_still_predicts_what_a_two_column_fit_predicts(self):
        chain = np.full(self.count, NEG, dtype=np.float32)
        found = Chain_Fill.fit_coefficients(self.out_of_fold, chain, self.y, self.weights)
        self.assertTrue(np.all(np.isfinite(found)))
        design = np.column_stack([np.ones(self.count), self.out_of_fold])
        plain = self.reference(design)
        predicted = Chain_Fill.combine(found, self.out_of_fold, chain)
        np.testing.assert_allclose(predicted, plain[0] + plain[1] * self.out_of_fold, rtol=1e-9, atol=1e-9)

    def test_the_held_out_check_fits_the_even_half_and_scores_the_odd_half(self):
        chain = self.chain.copy()
        chain[[1, 5]] = NEG  # both uncovered pairs at odd positions: the even half's fourth column is all zero
        predicted = Chain_Fill.held_out_predictions(self.out_of_fold, chain, self.y, self.weights)
        even = np.arange(0, self.count, 2)
        odd = np.arange(1, self.count, 2)
        coefficients = Chain_Fill.fit_coefficients(
            self.out_of_fold[even], chain[even], self.y[even], self.weights[even])
        self.assertEqual(coefficients[3], 0.0)
        self.assertEqual(predicted.size, odd.size)
        self.assertTrue(np.array_equal(
            predicted, Chain_Fill.combine(coefficients, self.out_of_fold[odd], chain[odd])))
        # The odd pairs' uncovered ones fall back on the base (c_oof), plus the indicator's zero coefficient.
        self.assertEqual(predicted[0], np.float64(
            coefficients[0] + coefficients[1] * self.out_of_fold[1] + coefficients[2] * self.out_of_fold[1]
            + coefficients[3] * True))

    def test_the_combination_uses_the_chain_where_there_is_one_and_the_base_where_there_is_not(self):
        coefficients = np.array([0.5, 2.0, 3.0, 7.0])
        base = np.array([1.0, 1.0])
        chain = np.array([4.0, NEG], dtype=np.float32)
        np.testing.assert_array_equal(
            Chain_Fill.combine(coefficients, base, chain),
            [0.5 + 2.0 * 1.0 + 3.0 * 4.0 + 0.0, 0.5 + 2.0 * 1.0 + 3.0 * 1.0 + 7.0])


class SmallNetworkTests(unittest.TestCase):
    def fill(self, n, pairs, scores, coefficients=(0.0, 1.0, 1.0, 0.0)):
        edge_i = np.array([p[0] for p in pairs], dtype=np.int32)
        edge_j = np.array([p[1] for p in pairs], dtype=np.int32)
        square = Chain_Fill.score_matrix(n, edge_i, edge_j, np.array(scores, dtype=np.float32))
        plan = Chain_Fill.ChainPlan(n, square, Chain_Fill.build_index(square), np.array(coefficients),
                                    np.zeros(0, dtype=np.float32), 1.0, np.zeros(0))
        similarity = np.full((n, n), 0.5)
        iso = IsotonicRegression(out_of_bounds="clip").fit([0.0, 1.0], [0.0, 1.0])
        condensed = np.full(n * (n - 1) // 2, -1.0, dtype=np.float32)
        counts = Chain_Fill.fill_distances(plan, condensed, similarity, iso, 10.0)
        return condensed, counts

    def test_two_sequences_have_one_pair_and_nothing_to_fill(self):
        condensed, counts = self.fill(2, [(0, 1)], [3.0])
        self.assertEqual(counts, (0, 0))
        self.assertEqual(condensed.tolist(), [0.0])  # the computed pair's own distance is written by the caller

    def test_three_sequences_with_two_computed_pairs_fill_the_third_from_their_chain(self):
        # (0, 1) = 2 and (1, 2) = 3: the pair (0, 2) has the chain min(2, 3) = 2; base 0.5, coefficients (0, 1, 1, 0).
        condensed, counts = self.fill(3, [(0, 1), (1, 2)], [2.0, 3.0])
        self.assertEqual(counts, (1, 1))
        self.assertEqual(condensed[1], np.float32(10.0 - (0.5 + 2.0)))

    def test_three_sequences_with_one_computed_pair_have_no_chain(self):
        condensed, counts = self.fill(3, [(0, 1)], [2.0], coefficients=(0.25, 1.0, 1.0, 0.5))
        self.assertEqual(counts, (0, 2))
        # uncovered: 0.25 + 1 * 0.5 + 1 * 0.5 + 0.5 * 1
        self.assertEqual(condensed[1], np.float32(10.0 - 1.75))
        self.assertEqual(condensed[2], np.float32(10.0 - 1.75))

    def test_no_missing_pair_covered_and_an_isolated_sequence(self):
        # 0-1 and 2-3 only, and sequence 4 isolated: no missing pair has a chain.
        condensed, counts = self.fill(5, [(0, 1), (2, 3)], [1.0, 2.0])
        self.assertEqual(counts, (0, 8))
        self.assertEqual(int(np.count_nonzero(condensed == np.float32(10.0 - (0.5 + 0.5)))), 8)

    def test_a_network_in_which_the_chain_fill_has_no_sequence_pairs_at_all_is_handled(self):
        condensed, counts = self.fill(4, [], [])
        self.assertEqual(counts, (0, 6))


class MemoryEstimateTests(unittest.TestCase):
    def test_the_estimate_is_the_sum_of_the_arrays_the_fill_holds(self):
        n, edges, block = 300, 9000, 5000
        pairs = n * (n - 1) // 2
        words = (n + 63) // 64
        expected = (4 * n * n + 16 * edges + 8 * (n + 1) + 8 * n * words
                    + Chain_Fill.BLOCK_BYTES_PER_PAIR * block + 4 * pairs)
        self.assertEqual(Chain_Fill.estimate_bytes(n, edges, block), expected)
        # one block is never more than all the pairs
        self.assertEqual(Chain_Fill.estimate_bytes(3, 1, 10**9),
                         4 * 9 + 16 + 32 + 8 * 3 + Chain_Fill.BLOCK_BYTES_PER_PAIR * 3 + 12)

    def test_the_estimate_covers_the_measured_peak_and_is_not_far_above_it(self):
        n, block = 400, 5000
        edge_i, edge_j, y = random_network(n, 0.15, seed=1)
        edges = edge_i.size
        similarity = np.random.default_rng(0).random((n, n))
        similarity = similarity + similarity.T
        iso = IsotonicRegression(out_of_bounds="clip").fit(np.linspace(0.0, 2.0, 50), np.linspace(0.0, 3.0, 50))
        plan = Chain_Fill.ChainPlan(n, None, None, np.array([0.0, 1.0, 1.0, 0.0]), np.zeros(0, dtype=np.float32), 1.0,
                                    np.zeros(0))
        tracemalloc.start()
        try:
            condensed = np.empty(n * (n - 1) // 2, dtype=np.float32)
            plan.square = Chain_Fill.score_matrix(n, edge_i, edge_j, y)
            plan.index = Chain_Fill.build_index(plan.square)
            Chain_Fill.fill_distances(plan, condensed, similarity, iso, 3.0, block_pairs=block)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        estimate = Chain_Fill.estimate_bytes(n, edges, block)
        self.assertLessEqual(peak, estimate)
        self.assertGreater(peak, 0.6 * estimate)
        arrays = (plan.square.nbytes + plan.index.idx.nbytes + plan.index.val.nbytes + plan.index.start.nbytes
                  + plan.index.bits.nbytes + condensed.nbytes)
        self.assertEqual(plan.index.idx.nbytes + plan.index.val.nbytes, 16 * np.count_nonzero(
            plan.square[np.triu_indices(n, 1)] > HALF))
        self.assertLessEqual(arrays, estimate)

    def test_the_verdict_compares_the_estimate_with_the_free_memory_minus_the_reserve(self):
        need = Chain_Fill.estimate_bytes(1000, 50_000)
        with mock.patch.object(Chain_Fill, "available_memory_bytes", return_value=(64 * GIB, 40 * GIB)):
            verdict = Chain_Fill.memory_verdict(1000, 50_000)
        self.assertEqual(verdict.reserve, 16 * GIB)  # a quarter of the total, above the 8 GiB floor
        self.assertEqual(verdict.usable, 24 * GIB)
        self.assertEqual(verdict.need, need)
        self.assertTrue(verdict.fits)
        with mock.patch.object(Chain_Fill, "available_memory_bytes", return_value=(16 * GIB, 8 * GIB)):
            verdict = Chain_Fill.memory_verdict(1000, 50_000)
        self.assertEqual((verdict.reserve, verdict.usable, verdict.fits), (8 * GIB, 0, False))
        with mock.patch.object(Chain_Fill, "available_memory_bytes", return_value=(0, 0)):
            self.assertFalse(Chain_Fill.memory_verdict(10, 5).fits)


class RunPathTests(unittest.TestCase):
    """The tool's own code path on a fixture, with the sample's alignment replaced as the other MSA tests do."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = pathlib.Path(temp_dir.name)

    def fixture(self, name="run", **options):
        options.setdefault("sequences", 27)
        options.setdefault("ghosts", 0)
        options.setdefault("seed", 4)
        options.setdefault("complete", False)
        fixture = NetworkFixture(self.directory / name, **options)
        fixture.record_scoring((-3.0, -0.5), "ieee_fp32")
        return fixture

    def run_builder(self, fixture, sample=12, memory=None, **settings):
        """Run to the guide tree; return a dict of the text printed, the notes file and what the run computed."""
        output = io.StringIO()
        seen = {}
        real_pair_similarity = Prefilter_Calibration.pair_similarity
        real_kernel = Embedding_MSA.calculate_normalized_scores_kernel
        real_fit = Embedding_MSA.fit_imputation
        real_linkage = Embedding_MSA.sch.linkage

        def similarity(*args, **kwargs):
            seen["similarity"] = real_pair_similarity(*args, **kwargs)
            return seen["similarity"]

        def kernel(**arguments):
            result = real_kernel(**arguments)
            seen["edge_i"], seen["edge_j"] = np.array(arguments["edge_i"]), np.array(arguments["edge_j"])
            seen["norm"], seen["max_norm"] = np.array(result[0]), float(result[1])
            return result

        def fit(*args, **kwargs):
            seen["fit"] = real_fit(*args, **kwargs)
            seen["fit_args"] = args
            return seen["fit"]

        def linkage(condensed, *args, **kwargs):
            seen["baseline"] = np.array(condensed)
            return real_linkage(condensed, *args, **kwargs)

        patches, _ = fake_calibration_alignment()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.multiple(
                Embedding_MSA,
                benchmark_msa_devices=mock.Mock(side_effect=GuideTreeBuilt),
                calculate_normalized_scores_kernel=kernel,
                fit_imputation=fit,
                **builder_settings(fixture, **settings),
            ))
            stack.enter_context(mock.patch.object(Embedding_MSA.sch, "linkage", linkage))
            stack.enter_context(mock.patch.object(Prefilter_Calibration, "pair_similarity", similarity))
            stack.enter_context(patches)
            stack.enter_context(mock.patch.object(Embedding_MSA.mp, "set_start_method"))
            stack.enter_context(mock.patch.object(Embedding_MSA.mp, "Pool", InProcessPool))
            stack.enter_context(mock.patch.object(Prefilter_Calibration, "DRAW_MIN_PAIRS", sample))
            stack.enter_context(mock.patch.object(Prefilter_Calibration, "DRAW_MAX_PAIRS", sample))
            if memory is not None:
                stack.enter_context(mock.patch.object(Chain_Fill, "available_memory_bytes", return_value=memory))
            stack.enter_context(contextlib.redirect_stdout(output))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            with self.assertRaises(GuideTreeBuilt):
                Embedding_MSA.run_msa_builder()
        (path,) = glob.glob(str(fixture.msa_dir / "**" / "*guide_tree.nex"), recursive=True)
        text = pathlib.Path(path).read_text(encoding="utf-8")
        seen["output"] = output.getvalue()
        seen["file"] = text
        seen["notes"] = {
            name: value.replace("''", "'")
            for name, value in re.findall(r"^    (\w+) = '((?:[^']|'')*)';$", text, re.MULTILINE)
        }
        return seen

    @staticmethod
    def baseline_from(seen, fill):
        """The run's complete distance baseline as the tool builds it, with `fill` giving the missing pairs'
        score estimates as an (n, n) matrix (the dense layout, in plain numpy)."""
        n = seen["similarity"].shape[0]
        max_norm = seen["max_norm"]
        dense = np.maximum(0.0, max_norm - fill).astype(np.float32)
        np.fill_diagonal(dense, 0.0)
        dense = 0.5 * (dense + dense.T)
        baseline = squareform(dense)
        edge_dists = np.maximum(0.0, max_norm - seen["norm"]).astype(np.float32)
        for i, j, distance in zip(seen["edge_i"], seen["edge_j"], edge_dists):  # the last listing wins
            a, b = min(i, j), max(i, j)
            baseline[n * a - a * (a + 1) // 2 + b - a - 1] = distance
        return np.clip(baseline, 0.0, max_norm + 0.1)

    @staticmethod
    def chain_estimate(seen):
        """The chain + calibration score estimate of every pair, plain numpy, from what the run computed."""
        n = seen["similarity"].shape[0]
        imputation = seen["fit"]
        square = np.full((n, n), NEG, dtype=np.float32)
        for i, j, value in zip(seen["edge_i"], seen["edge_j"], seen["norm"]):
            square[i, j] = square[j, i] = value
        chain = np.full((n, n), NEG, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and k not in (i, j) and square[i, k] > HALF and square[k, j] > HALF:
                        chain[i, j] = max(chain[i, j], min(square[i, k], square[k, j]))
        positions = seen["fit_args"][2]
        cal_chain = chain[seen["edge_i"][positions], seen["edge_j"][positions]].astype(np.float64)
        covered_cal = cal_chain > float(HALF)
        c_oof = imputation.out_of_fold
        design = np.column_stack([np.ones(c_oof.size), c_oof, np.where(covered_cal, cal_chain, c_oof),
                                  (~covered_cal).astype(np.float64)])
        root = np.sqrt(imputation.calibration_weights)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], imputation.calibration_y.astype(np.float64) * root,
                                           rcond=None)
        base = imputation.iso.predict(seen["similarity"].ravel()).reshape(n, n)
        covered = chain > HALF
        return (coefficients[0] + coefficients[1] * base
                + coefficients[2] * np.where(covered, chain.astype(np.float64), base)
                + coefficients[3] * (~covered)), coefficients, covered

    def old_estimate(self, seen):
        n = seen["similarity"].shape[0]
        return seen["fit"].iso.predict(seen["similarity"].ravel()).reshape(n, n)

    def test_a_run_with_a_calibration_sample_fills_by_the_chain_and_its_baseline_equals_the_formulas(self):
        seen = self.run_builder(self.fixture())
        estimate, coefficients, covered = self.chain_estimate(seen)
        expected = self.baseline_from(seen, estimate)
        self.assertTrue(np.array_equal(seen["baseline"], expected.astype(np.float64)))
        output = seen["output"]
        self.assertIn("Chain fill: the missing pairs are estimated from the calibrated isotonic fit and from "
                      "chains of computed pairs.", output)
        self.assertIn("Applying chain + calibration imputation for final master tree...", output)
        self.assertIn(
            "Coefficients (constant, isotonic fit, chain, no chain): " + ", ".join(f"{c:.6f}" for c in coefficients),
            output)
        n = covered.shape[0]
        present = np.zeros((n, n), dtype=bool)
        present[seen["edge_i"], seen["edge_j"]] = present[seen["edge_j"], seen["edge_i"]] = True
        rows, cols = np.triu_indices(n, 1)
        missing = ~present[rows, cols]
        share = covered[rows, cols][missing].mean()
        self.assertIn(f"Missing pairs with a chain: {100.0 * share:.1f}% of {int(missing.sum()):,}", output)

    def test_the_isotonic_check_and_the_chain_check_score_the_same_odd_pairs(self):
        seen = self.run_builder(self.fixture())
        output = seen["output"]
        check = seen["fit"].check
        self.assertEqual(check.pairs, 6)
        match = re.search(r"checked on the same (\d+) pairs: Spearman rho (\S+), MAE (\S+), R\^2 (\S+)", output)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), check.pairs)
        note = seen["notes"]["MISSING_PAIRS"]
        number = r"(-?\d+\.\d{4}|nan)"
        self.assertRegex(
            note,
            rf"^Imputed from embedding similarity and chains of computed pairs \(calibrated\); checked on 6 held-out "
            rf"calibration pairs: Spearman {number}, MAE {number}, R\^2 {number}; \d+\.\d% of missing pairs had a "
            rf"chain; coefficients \(constant, similarity fit, chain, no chain\) {number}, {number}, {number}, "
            rf"{number}; \d+\.\d% of missing pairs outside the fit's range$",
        )

    def test_a_run_too_short_of_memory_falls_back_to_todays_isotonic_fill_and_says_why(self):
        seen = self.run_builder(self.fixture(), memory=(16 * GIB, 8 * GIB))
        expected = self.baseline_from(seen, self.old_estimate(seen))
        self.assertTrue(np.array_equal(seen["baseline"], expected.astype(np.float64)))
        output = seen["output"]
        self.assertRegex(
            output,
            r"⚠️ Chain fill not used: it needs about [\d.]+ MB and 0\.0 MB of memory is free beyond a reserve of "
            r"8\.0 GB\n  > The missing pairs are filled from the calibrated isotonic fit alone\.",
        )
        self.assertIn("Applying hybrid adjusted similarity-alignment imputation for final master tree...", output)
        self.assertNotIn("Applying chain + calibration imputation", output)
        note = seen["notes"]["MISSING_PAIRS"]
        self.assertTrue(note.startswith("Regression-imputed from embedding distances; checked on 6 held-out calibration "
                                        "pairs: Spearman "))
        self.assertIn("; chain fill not used: it needs about ", note)
        self.assertTrue(note.endswith("% of missing pairs outside the fit's range"))

    def test_every_run_without_a_calibration_sample_keeps_the_uniform_fill_and_prints_no_chain_line(self):
        cases = {
            "switch off": dict(settings=dict(CALIBRATED_IMPUTATION=False)),
            "too small a sample": dict(sample=5),
            "blast": dict(options=dict(blast=True), settings=dict(CALIBRATED_IMPUTATION=True)),
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                fixture = self.fixture(name.replace(" ", "_"), **case.get("options", {}))
                seen = self.run_builder(fixture, sample=case.get("sample", 12), **case.get("settings", {}))
                self.assertNotIn("Chain fill", seen["output"], name)
                self.assertNotIn("chain + calibration", seen["output"], name)
                self.assertNotIn("chain fill", seen["notes"]["MISSING_PAIRS"])
                self.assertNotIn("chains of computed pairs", seen["notes"]["MISSING_PAIRS"])
                self.assertTrue(seen["notes"]["MISSING_PAIRS"].startswith("Regression-imputed from embedding distances"))
                expected = self.baseline_from(seen, self.old_estimate(seen))
                self.assertTrue(np.array_equal(seen["baseline"], expected.astype(np.float64)))

    # Made by Embedding_MSA at fcc6c6f (before the chain fill) on this fixture: 27 sequences, seed 4, 60% of the
    # pairs listed, a sample of 12 calibration pairs; TREE lines of the guide tree files.
    TREE_BEFORE_SWITCH_OFF = (
        "(((seq0011:146.688,seq0020:146.688):43.121,((seq0014:178.697,seq0023:178.697):10.048,"
        "(seq0013:179.681,seq0022:179.681):9.06408):1.06394):1.23969,((seq0001:189.232,(seq0015:188.89,"
        "(((seq0003:151.278,seq0012:151.278):33.3894,(seq0009:175.924,(seq0021:0,seq0026:0):175.924):8.743):3.30151,"
        "(seq0024:184.693,(seq0000:178.35,seq0002:178.35):6.34296):3.27588):0.920948):0.342204):0.930033,"
        "((seq0018:187.874,((seq0006:182.586,(seq0016:165.628,seq0019:165.628):16.9576):4.53639,"
        "(seq0004:183.102,(seq0008:174.856,seq0010:174.856):8.24553):4.02075):0.751809):2.005,"
        "(seq0007:187.874,seq0025:187.874):2.00468):0.282845):0.886822);"
    )
    TREE_BEFORE_SWITCH_ON = (
        "(((seq0011:146.688,seq0020:146.688):43.3922,((seq0014:178.697,seq0023:178.697):10.048,"
        "(seq0013:179.681,seq0022:179.681):9.06408):1.33507):1.23219,((seq0007:187.874,seq0025:187.874):2.79614,"
        "(((seq0024:184.965,(seq0000:178.35,seq0002:178.35):6.61467):3.32962,((seq0003:151.278,seq0012:151.278):33.7509,"
        "(seq0009:175.924,(seq0021:0,seq0026:0):175.924):9.10452):3.26544):2.12165,((seq0004:183.188,"
        "seq0015:183.188):6.15303,((seq0018:185.552,(seq0008:174.856,seq0010:174.856):10.6959):3.23841,"
        "(seq0001:187.876,(seq0006:182.586,(seq0016:165.628,seq0019:165.628):16.9576):5.29037):0.914092):0.550872):"
        "1.07461):0.254712):0.641842);"
    )
    NOTES_BEFORE_SWITCH_OFF = {
        "MISSING_PAIRS": "Regression-imputed from embedding distances; the calibration switch is off, so the "
                         "estimates are unchecked; 2.4% of missing pairs outside the fit's range",
        "CALIBRATED_IMPUTATION": "false",
        "CALIBRATION": "Calibration switched off: no sample is drawn or aligned, so the fit takes the observed pairs only",
    }

    @staticmethod
    def tree_line(text):
        return re.search(r"^    TREE guide_tree = \[&R\] (.*)$", text, re.MULTILINE).group(1)

    def test_the_guide_tree_of_a_run_with_the_switch_off_is_what_it_was_before_the_chain_fill(self):
        seen = self.run_builder(self.fixture(), CALIBRATED_IMPUTATION=False)
        self.assertEqual(self.tree_line(seen["file"]), self.TREE_BEFORE_SWITCH_OFF)
        for name, value in self.NOTES_BEFORE_SWITCH_OFF.items():
            self.assertEqual(seen["notes"][name], value, name)

    def test_the_guide_tree_of_the_memory_fall_back_is_what_a_calibrated_run_gave_before_the_chain_fill(self):
        seen = self.run_builder(self.fixture(), memory=(16 * GIB, 8 * GIB))
        self.assertEqual(self.tree_line(seen["file"]), self.TREE_BEFORE_SWITCH_ON)

    def test_the_chain_fill_gives_another_tree_than_the_isotonic_fill_on_the_same_run(self):
        seen = self.run_builder(self.fixture())
        self.assertNotEqual(self.tree_line(seen["file"]), self.TREE_BEFORE_SWITCH_ON)

    def test_a_complete_network_imputes_nothing_and_prints_no_chain_line(self):
        seen = self.run_builder(self.fixture(complete=True, duplicates=0, out_of_range=False))
        self.assertNotIn("Chain fill", seen["output"])
        self.assertEqual(seen["notes"]["MISSING_PAIRS"], "None. Every pair is in the network.")


if __name__ == "__main__":
    unittest.main()
