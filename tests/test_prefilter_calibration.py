"""utilities.Prefilter_Calibration: the prefilter's similarity, the calibration
sample of the pairs it skips, the network file's record of both, and finding
the sample among a run's edges."""

import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Prefilter_Calibration as calibration  # noqa: E402


def upper_values(matrix):
    """A square matrix's upper triangle in row-major order."""
    return matrix[np.triu(np.ones(matrix.shape, dtype=bool), k=1)]


def random_similarity(n, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.random((n, n))
    return (matrix + matrix.T) / 2


class SampleSizeTests(unittest.TestCase):
    def test_one_percent_between_the_floor_and_the_cap(self):
        cases = {
            0: 0,
            150: 150,                    # fewer skipped than the floor: all of them
            2_000: 2_000,
            150_000: 2_000,              # 1 % is 1,500, under the floor
            450_049: 4_500,              # rounded
            450_050: 4_500,              # round half to even: 4500.5 -> 4500
            450_051: 4_501,
            5_000_000: 50_000,
            60_000_000: 50_000,          # the cap
        }
        for skipped, size in cases.items():
            with self.subTest(skipped=skipped):
                self.assertEqual(calibration.sample_size(skipped), size)


class CondensedPairTests(unittest.TestCase):
    def test_positions_map_back_to_their_pairs(self):
        for n in (2, 3, 7, 100, 12_927, 70_000):
            with self.subTest(n=n):
                total = n * (n - 1) // 2
                rng = np.random.default_rng(n)
                positions = np.unique(np.concatenate([
                    np.arange(min(total, 3_000)),
                    rng.integers(0, total, 5_000),
                    [total - 1],
                ]))
                rows, cols = calibration.condensed_pairs(positions, n)
                self.assertTrue(np.all(rows >= 0))
                self.assertTrue(np.all(cols > rows))
                self.assertTrue(np.all(cols < n))
                back = rows * n - rows * (rows + 1) // 2 + (cols - rows - 1)
                np.testing.assert_array_equal(back, positions)

    def test_the_order_is_numpys_upper_triangle(self):
        rows, cols = np.triu_indices(9, k=1)
        got_rows, got_cols = calibration.condensed_pairs(np.arange(rows.size), 9)
        np.testing.assert_array_equal(got_rows, rows)
        np.testing.assert_array_equal(got_cols, cols)

    def test_no_positions(self):
        rows, cols = calibration.condensed_pairs([], 5)
        self.assertEqual(rows.size, 0)
        self.assertEqual(cols.size, 0)


class SelectCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.n = 240
        self.similarity = random_similarity(self.n)
        self.values = upper_values(self.similarity)
        self.threshold = float(np.percentile(self.values, 60))
        self.skipped = int(np.count_nonzero(self.values < self.threshold))

    def select(self, **options):
        return calibration.select_calibration(self.values, self.threshold, self.n, **options)

    def test_every_pair_is_a_distinct_skipped_pair(self):
        sample = self.select(size=2_000)
        self.assertEqual(len(sample), 2_000)
        self.assertTrue(np.all(sample.rows < sample.cols))
        self.assertTrue(np.all(self.similarity[sample.rows, sample.cols] < self.threshold))
        self.assertEqual(len(set(zip(sample.rows.tolist(), sample.cols.tolist()))), 2_000)

    def test_each_band_gives_the_same_count_the_lowest_included(self):
        sample = self.select(size=2_000, bins=20)
        skipped = np.sort(self.values[self.values < self.threshold])
        edges = np.quantile(skipped, np.linspace(0, 1, 21))
        drawn = self.similarity[sample.rows, sample.cols]
        bands = np.clip(np.searchsorted(edges, drawn, side="right") - 1, 0, 19)
        np.testing.assert_array_equal(np.bincount(bands, minlength=20), np.full(20, 100))
        # The lowest band reaches the lowest skipped pairs.
        self.assertLess(drawn.min(), edges[1])

    def test_weights_stand_for_every_skipped_pair(self):
        sample = self.select(size=2_000)
        self.assertAlmostEqual(float(sample.weights.sum()), self.skipped, places=6)
        # Equal-count bands drawn equally: every weight is about skipped / size.
        np.testing.assert_allclose(sample.weights, self.skipped / 2_000, rtol=0.01)

    def test_the_same_seed_draws_the_same_pairs_another_does_not(self):
        first = self.select(size=1_000)
        again = self.select(size=1_000)
        other = self.select(size=1_000, seed=7)
        np.testing.assert_array_equal(first.rows, again.rows)
        np.testing.assert_array_equal(first.cols, again.cols)
        self.assertFalse(np.array_equal(first.rows, other.rows))

    def test_the_default_size_follows_sample_size(self):
        values = np.linspace(0.0, 1.0, 1_000_001)
        threshold = float(np.percentile(values, 25))
        sample = calibration.select_calibration(values, threshold, 1415)
        self.assertEqual(len(sample), calibration.sample_size(np.count_nonzero(values < threshold)))

    def test_few_skipped_pairs_are_all_taken_with_weight_one(self):
        threshold = float(np.sort(self.values)[30])
        sample = calibration.select_calibration(self.values, threshold, self.n, size=2_000)
        self.assertEqual(len(sample), 30)
        np.testing.assert_array_equal(sample.weights, np.ones(30))
        self.assertTrue(np.all(self.similarity[sample.rows, sample.cols] < threshold))

    def test_nothing_skipped_gives_an_empty_sample(self):
        sample = calibration.select_calibration(self.values, -1.0, self.n)
        self.assertEqual(len(sample), 0)
        self.assertEqual(sample.weights.size, 0)

    def test_ties_on_a_band_edge_keep_every_pair_in_one_band(self):
        values = np.repeat(np.arange(10, dtype=np.float64), 500)  # 5,000 pairs, ten values
        n = 101  # 5,050 pairs
        values = np.concatenate([values, np.full(50, 99.0)])
        sample = calibration.select_calibration(values, 50.0, n, size=400, bins=20)
        self.assertLessEqual(len(sample), 400)
        self.assertAlmostEqual(float(sample.weights.sum()), 5_000, places=6)


class SimilarityTests(unittest.TestCase):
    def test_cosine_times_the_length_ratio_to_a_power(self):
        rng = np.random.default_rng(1)
        pooled = rng.standard_normal((6, 5)).astype(np.float32)
        lengths = [10, 20, 40, 40, 5, 33]
        for power in (0.0, 1.0, 2.0):
            with self.subTest(power=power):
                got = calibration.length_adjusted_similarity(pooled, lengths, power)
                self.assertEqual(got.dtype, np.float64)
                for a in range(6):
                    for b in range(6):
                        cosine = np.dot(pooled[a], pooled[b]) / (
                            np.linalg.norm(pooled[a]) * np.linalg.norm(pooled[b])
                        )
                        ratio = min(lengths[a], lengths[b]) / max(lengths[a], lengths[b])
                        self.assertAlmostEqual(got[a, b], cosine * ratio ** power, places=5)

    def test_a_zero_vector_gives_zero_not_nan(self):
        pooled = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        got = calibration.length_adjusted_similarity(pooled, [4, 4], 2.0)
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertEqual(got[0, 1], 0.0)

    def test_pooled_vectors_take_the_max_or_the_mean(self):
        embeddings = {
            "a": np.array([[1, 5], [3, 1]], dtype=np.float16),
            "b": np.array([[2, 2]], dtype=np.float16),
        }
        np.testing.assert_array_equal(
            calibration.pooled_vectors(embeddings, ["a", "b"], "max"), [[3, 5], [2, 2]]
        )
        np.testing.assert_array_equal(
            calibration.pooled_vectors(embeddings, ["b", "a"], "mean"), [[2, 2], [2, 3]]
        )
        seen = []
        calibration.pooled_vectors(embeddings, ["a"], "max", progress=lambda names: seen.append(names) or names)
        self.assertEqual(seen, [["a"]])
        with self.assertRaisesRegex(ValueError, "pooling"):
            calibration.pooled_vectors(embeddings, ["a"], "median")


class RecordTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.path = pathlib.Path(temp_dir.name) / "network.h5"
        self.record = calibration.PrefilterRecord(
            strength=25.0, threshold=0.4321, pooling="max", length_power=2.0,
            kept=900, skipped=300,
            sample=calibration.CalibrationSample(
                np.array([0, 3, 7]), np.array([5, 9, 8]), np.array([100.0, 100.0, 100.0])
            ),
        )

    def test_a_record_reads_back_as_written(self):
        with h5py.File(self.path, "w") as hf:
            calibration.write_record(hf, self.record, np.uint16)
        with h5py.File(self.path, "r") as hf:
            self.assertEqual(hf[calibration.DATA_I].dtype, np.uint16)
            got = calibration.read_record(hf)
        self.assertEqual(
            (got.strength, got.threshold, got.pooling, got.length_power, got.kept, got.skipped,
             got.bins, got.seed),
            (25.0, 0.4321, "max", 2.0, 900, 300, calibration.BINS, calibration.SEED),
        )
        np.testing.assert_array_equal(got.sample.rows, [0, 3, 7])
        np.testing.assert_array_equal(got.sample.cols, [5, 9, 8])
        np.testing.assert_array_equal(got.sample.weights, [100.0, 100.0, 100.0])

    def test_a_file_without_a_whole_record_has_none(self):
        with h5py.File(self.path, "w") as hf:
            hf.attrs["model_name"] = "m"
        with h5py.File(self.path, "r") as hf:
            self.assertIsNone(calibration.read_record(hf))
        for missing in (calibration.ATTR_POOLING, calibration.DATA_WEIGHT):
            with self.subTest(missing=missing):
                with h5py.File(self.path, "w") as hf:
                    calibration.write_record(hf, self.record, np.uint16)
                    if missing in hf.attrs:
                        del hf.attrs[missing]
                    else:
                        del hf[missing]
                with h5py.File(self.path, "r") as hf:
                    self.assertIsNone(calibration.read_record(hf))

    def test_a_pooling_stored_as_bytes_reads_as_text(self):
        with h5py.File(self.path, "w") as hf:
            calibration.write_record(hf, self.record, np.uint32)
            hf.attrs[calibration.ATTR_POOLING] = np.bytes_(b"mean")
        with h5py.File(self.path, "r") as hf:
            self.assertEqual(calibration.read_record(hf).pooling, "mean")


class LocateSampleTests(unittest.TestCase):
    def sample(self, rows, cols):
        return calibration.CalibrationSample(
            np.asarray(rows), np.asarray(cols), np.ones(len(rows))
        )

    def test_pairs_are_found_either_way_round_in_sample_order(self):
        edge_i = np.array([4, 1, 0, 3, 2], dtype=np.int32)
        edge_j = np.array([2, 0, 3, 4, 1], dtype=np.int32)
        identity = np.arange(5, dtype=np.int32)
        located = calibration.locate_sample(
            self.sample([2, 0, 0], [4, 3, 1]), identity, edge_i, edge_j, 5
        )
        np.testing.assert_array_equal(located, [0, 2, 1])

    def test_a_pair_not_among_the_edges_or_the_sequences_is_minus_one(self):
        edge_i = np.array([0, 1], dtype=np.int32)
        edge_j = np.array([1, 2], dtype=np.int32)
        # Network sequence 3 was dropped; network 4 is the run's 2.
        index_map = np.array([0, 1, 2, -1, 2], dtype=np.int32)
        located = calibration.locate_sample(
            self.sample([0, 3, 1, 0], [2, 4, 4, 1]), index_map, edge_i, edge_j, 3
        )
        np.testing.assert_array_equal(located, [-1, -1, 1, 0])

    def test_a_pair_stored_twice_is_found_at_its_first_edge(self):
        edge_i = np.array([5, 0, 1, 1], dtype=np.int32)
        edge_j = np.array([6, 1, 0, 2], dtype=np.int32)
        located = calibration.locate_sample(
            self.sample([1, 1], [0, 2]), np.arange(7), edge_i, edge_j, 7
        )
        np.testing.assert_array_equal(located, [1, 3])

    def test_chunks_find_what_one_pass_finds(self):
        rng = np.random.default_rng(4)
        n = 60
        rows, cols = np.triu_indices(n, k=1)
        keep = rng.random(rows.size) < 0.7
        edge_i, edge_j = rows[keep], cols[keep]
        picked = rng.choice(rows.size, 80, replace=False)
        sample = self.sample(rows[picked], cols[picked])
        whole = calibration.locate_sample(sample, np.arange(n), edge_i, edge_j, n)
        with mock.patch.object(calibration, "_MATCH_CHUNK", 37):
            chunked = calibration.locate_sample(sample, np.arange(n), edge_i, edge_j, n)
        np.testing.assert_array_equal(whole, chunked)
        np.testing.assert_array_equal(whole >= 0, keep[picked])
        found = whole >= 0
        np.testing.assert_array_equal(edge_i[whole[found]], rows[picked][found])
        np.testing.assert_array_equal(edge_j[whole[found]], cols[picked][found])

    def test_an_empty_sample(self):
        located = calibration.locate_sample(
            self.sample([], []), np.arange(3), np.array([0]), np.array([1]), 3
        )
        self.assertEqual(located.size, 0)


class RunSampleTests(unittest.TestCase):
    """The sample Embedding_MSA draws from a run's own missing pairs."""

    def setUp(self):
        self.n = 240
        self.similarity = random_similarity(self.n, seed=3)
        rows, cols = np.triu_indices(self.n, k=1)
        rng = np.random.default_rng(5)
        has_edge = rng.random(rows.size) < 0.6
        self.edge_i, self.edge_j = rows[has_edge], cols[has_edge]
        self.present = calibration.pair_presence(self.edge_i, self.edge_j, self.n)
        self.values, self.locate = calibration.missing_pairs(self.similarity, self.present)
        self.rows, self.cols = self.locate(np.arange(self.values.size))

    def test_the_bounds_of_a_run_sample_are_5000_and_100000(self):
        self.assertEqual((calibration.DRAW_MIN_PAIRS, calibration.DRAW_MAX_PAIRS), (5_000, 100_000))
        cases = {0: 0, 3_000: 3_000, 5_000: 5_000, 200_000: 5_000, 1_000_000: 10_000,
                 10_000_000: 100_000, 90_000_000: 100_000}
        for missing, size in cases.items():
            with self.subTest(missing=missing):
                self.assertEqual(
                    calibration.sample_size(missing, calibration.DRAW_MIN_PAIRS, calibration.DRAW_MAX_PAIRS), size)
        # Without bounds the stored sample's own bounds still apply.
        self.assertEqual(calibration.sample_size(150_000), 2_000)

    def test_presence_ignores_the_order_of_a_pair_duplicates_and_self_edges(self):
        present = calibration.pair_presence([3, 1, 2, 4], [1, 3, 2, 0], 5)
        self.assertTrue(present[1, 3] and present[0, 4])
        self.assertEqual(int(np.count_nonzero(present)), 2)
        self.assertEqual(calibration.missing_count(present), 5 * 4 // 2 - 2)

    def test_the_missing_pairs_are_exactly_the_pairs_without_an_edge_in_row_major_order(self):
        all_rows, all_cols = np.triu_indices(self.n, k=1)
        expected = ~np.isin(all_rows * self.n + all_cols, self.edge_i * self.n + self.edge_j)
        np.testing.assert_array_equal(self.rows, all_rows[expected])
        np.testing.assert_array_equal(self.cols, all_cols[expected])
        np.testing.assert_array_equal(self.values, self.similarity[all_rows[expected], all_cols[expected]])

    def test_the_sample_is_drawn_only_among_the_missing_pairs(self):
        sample = calibration.select_among(self.values, self.locate, size=2_000)
        self.assertEqual(len(sample), 2_000)
        self.assertFalse(self.present[sample.rows, sample.cols].any())
        self.assertTrue(np.all(sample.rows < sample.cols))
        self.assertEqual(len(set(zip(sample.rows.tolist(), sample.cols.tolist()))), 2_000)
        self.assertAlmostEqual(float(sample.weights.sum()), self.values.size, places=6)

    def test_with_the_stored_bounds_it_draws_what_the_prefilter_drew(self):
        upper = upper_values(self.similarity)
        threshold = float(np.percentile(upper, 40))
        old = calibration.select_calibration(upper, threshold, self.n)
        rows_all, cols_all = np.triu_indices(self.n, k=1)
        keep = upper >= threshold
        present = calibration.pair_presence(rows_all[keep], cols_all[keep], self.n)
        values, locate = calibration.missing_pairs(self.similarity, present)
        new = calibration.select_among(
            values, locate, minimum=calibration.MIN_PAIRS, maximum=calibration.MAX_PAIRS)
        np.testing.assert_array_equal(new.rows, old.rows)
        np.testing.assert_array_equal(new.cols, old.cols)
        np.testing.assert_array_equal(new.weights, old.weights)

    def test_no_more_missing_pairs_than_the_size_are_all_taken_with_weight_one(self):
        sample = calibration.select_among(self.values, self.locate, size=10 ** 6)
        np.testing.assert_array_equal(sample.rows, self.rows)
        np.testing.assert_array_equal(sample.cols, self.cols)
        np.testing.assert_array_equal(sample.weights, np.ones(self.values.size))
        none = calibration.select_among(self.values[:0], self.locate)
        self.assertEqual(len(none), 0)

    def test_locate_gives_the_rows_and_columns_of_any_positions_in_any_order(self):
        all_rows, all_cols = np.triu_indices(self.n, k=1)
        expected = ~np.isin(all_rows * self.n + all_cols, self.edge_i * self.n + self.edge_j)
        want_rows, want_cols = all_rows[expected], all_cols[expected]
        rng = np.random.default_rng(9)
        positions = np.concatenate([[0, want_rows.size - 1], rng.integers(0, want_rows.size, 500)])
        rng.shuffle(positions)
        rows, cols = self.locate(positions)
        np.testing.assert_array_equal(rows, want_rows[positions])
        np.testing.assert_array_equal(cols, want_cols[positions])
        self.assertEqual(rows.dtype, np.int64)
        # Rows that are full and rows that are empty of edges: every pair of an edgeless network, none of a complete one.
        for edges in (np.zeros(0, dtype=int), None):
            n = 9
            r, c = np.triu_indices(n, k=1)
            present = calibration.pair_presence(r if edges is None else edges, c if edges is None else edges, n)
            values, locate = calibration.missing_pairs(np.ones((n, n)), present)
            self.assertEqual(values.size, 0 if edges is None else n * (n - 1) // 2)
            if edges is not None:
                got = locate(np.arange(values.size))
                np.testing.assert_array_equal(got[0], r)
                np.testing.assert_array_equal(got[1], c)

    def test_the_draw_adds_at_most_about_ten_bytes_per_candidate_to_the_values(self):
        import tracemalloc

        count = 400_000
        values = np.random.default_rng(4).random(count)
        with mock.patch.object(calibration, "_BAND_CHUNK", 10_000):
            tracemalloc.start()
            try:
                base = tracemalloc.get_traced_memory()[0]
                picked, weights = calibration._draw_by_bands(values, 1_000, calibration.BINS, calibration.SEED)
                peak = tracemalloc.get_traced_memory()[1] - base
            finally:
                tracemalloc.stop()
        # The quantile's one copy (8), the one-byte bands (1), a band's mask (1) and its members (8 / 20): about 10.4.
        self.assertLess(peak, 12 * count + 1_000_000)
        self.assertEqual(len(picked), 1_000)

    def test_the_missing_pairs_hold_the_values_alone_eight_bytes_each(self):
        import tracemalloc

        n = 600
        similarity = random_similarity(n, seed=1)
        rows, cols = np.triu_indices(n, k=1)
        keep = np.random.default_rng(2).random(rows.size) < 0.5
        present = calibration.pair_presence(rows[keep], cols[keep], n)
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            values, locate = calibration.missing_pairs(similarity, present)
            held = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
        # The values (8 bytes per missing pair) plus the packed presence (n * n / 8) and a few vectors of n.
        self.assertLess(held, values.size * 8 + n * n // 8 + 100_000)
        self.assertGreater(held, values.size * 8)


class SharedSimilarityRuleTests(unittest.TestCase):
    """One function owns the similarity rule; no tool keeps its own copy of it."""

    def embeddings(self):
        rng = np.random.default_rng(3)
        return {
            "a_b": rng.standard_normal((5, 6)).astype(np.float32),
            "c_d": rng.standard_normal((9, 6)).astype(np.float32),
            "e": rng.standard_normal((7, 6)).astype(np.float32),
        }

    def test_the_rule_is_the_shared_pooling_and_power_over_the_dataset_names_of_the_headers(self):
        embeddings = self.embeddings()
        headers, lengths = ["a/b", "c\\d", "e"], [5, 9, 7]
        got = calibration.pair_similarity(embeddings, headers, lengths)
        pooled = calibration.pooled_vectors(embeddings, ["a_b", "c_d", "e"], calibration.POOLING)
        want = calibration.length_adjusted_similarity(pooled, lengths, calibration.LENGTH_RATIO_POWER)
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(got.tobytes(), want.tobytes())
        # Headers already mapped to dataset names give the same matrix.
        self.assertEqual(calibration.pair_similarity(embeddings, ["a_b", "c_d", "e"], lengths).tobytes(), got.tobytes())
        self.assertEqual((calibration.POOLING, calibration.LENGTH_RATIO_POWER), ("max", 2.0))

    def test_the_constants_are_what_the_rule_uses_and_the_progress_hook_wraps_the_names(self):
        embeddings = self.embeddings()
        with mock.patch.multiple(calibration, POOLING="mean", LENGTH_RATIO_POWER=1.0):
            seen = []
            got = calibration.pair_similarity(
                embeddings, ["a/b", "c\\d", "e"], [5, 9, 7], progress=lambda names: seen.append(list(names)) or names)
            pooled = calibration.pooled_vectors(embeddings, ["a_b", "c_d", "e"], "mean")
            want = calibration.length_adjusted_similarity(pooled, [5, 9, 7], 1.0)
        self.assertEqual(got.tobytes(), want.tobytes())
        self.assertEqual(seen, [["a_b", "c_d", "e"]])
        import inspect

        self.assertEqual(list(inspect.signature(calibration.pair_similarity).parameters),
                         ["embeddings", "headers", "lengths", "progress"])

    @staticmethod
    def whole_matrix_reference(pooled, lengths, power):
        """The arithmetic as it was before the row blocks: n x n temporaries, the one-thread product."""
        from threadpoolctl import threadpool_limits

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        unit = pooled / norms
        with threadpool_limits(limits=1, user_api="blas"):
            similarity = np.clip(np.dot(unit, unit.T), -1.0, 1.0)
        lengths = np.asarray(lengths, dtype=np.int32)
        shorter = np.minimum(lengths[:, None], lengths[None, :])
        longer = np.maximum(np.maximum(lengths[:, None], lengths[None, :]), 1)
        ratio = shorter / longer
        if power != 1.0:
            ratio = ratio ** power
        return similarity * ratio

    def test_the_row_blocks_give_the_bytes_of_the_whole_matrix_arithmetic(self):
        rng = np.random.default_rng(11)
        pooled = rng.standard_normal((257, 24)).astype(np.float32)
        pooled[5] = 0.0  # a zero vector counts as length 1e-8
        lengths = rng.integers(0, 400, 257)  # unequal, with zeros
        for power in (2.0, 1.0, 1.5, 3.0):
            for block in (4_000_000, 257 * 7, 257, 1):  # one block, a few rows, one row, less than a row
                with self.subTest(power=power, block=block), mock.patch.object(calibration, "_SIMILARITY_BLOCK", block):
                    got = calibration.length_adjusted_similarity(pooled, lengths, power)
                    want = self.whole_matrix_reference(pooled, lengths, power)
                    self.assertEqual(got.dtype, np.float64)
                    self.assertEqual(got.tobytes(), want.tobytes())

    def test_the_similarity_holds_the_product_and_the_result_and_one_block(self):
        import tracemalloc

        rng = np.random.default_rng(12)
        n = 1200
        pooled = rng.standard_normal((n, 16)).astype(np.float32)
        lengths = rng.integers(20, 300, n)
        block = 50_000
        with mock.patch.object(calibration, "_SIMILARITY_BLOCK", block):
            tracemalloc.start()
            try:
                result = calibration.length_adjusted_similarity(pooled, lengths, 2.0)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        # 4 bytes (product) + 8 bytes (result) per pair of sequences, plus one block's few temporaries.
        self.assertLess(peak, 12 * n * n + 40 * block + 1_000_000)
        self.assertEqual(result.nbytes, 8 * n * n)

    def test_the_bytes_are_the_same_whatever_number_of_blas_threads_the_process_has(self):
        from threadpoolctl import threadpool_info, threadpool_limits

        many = max((p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"), default=1)
        if many < 2:
            self.skipTest("this machine's BLAS has one thread, so a threaded product cannot be compared with a single one")
        rng = np.random.default_rng(1)
        pooled = rng.standard_normal((200, 64)).astype(np.float32)
        lengths = rng.integers(50, 90, 200)
        unit = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        with threadpool_limits(limits=1, user_api="blas"):
            raw_one = np.dot(unit, unit.T).tobytes()
        with threadpool_limits(limits=many, user_api="blas"):
            raw_many = np.dot(unit, unit.T).tobytes()
        if raw_one == raw_many:
            self.skipTest("this BLAS gives the same bits at every thread count here, so the limit cannot be seen")
        results = {}
        for threads in (1, 2, many):
            with threadpool_limits(limits=threads, user_api="blas"):
                results[threads] = calibration.length_adjusted_similarity(pooled, lengths, 2.0).tobytes()
        self.assertEqual(len(set(results.values())), 1, "the similarity follows the process's BLAS thread count")
        # And it is the one-thread bits: what a process with one BLAS thread (Align's own) computes.
        with threadpool_limits(limits=1, user_api="blas"):
            shorter = np.minimum(lengths[:, None], lengths[None, :])
            longer = np.maximum(lengths[:, None], lengths[None, :])
            expected = np.clip(np.dot(unit, unit.T), -1.0, 1.0) * ((shorter / longer) ** 2.0)
        self.assertEqual(results[many], expected.tobytes())

    def test_no_tool_script_keeps_its_own_pooling_or_power_or_calls_the_pieces_of_the_rule(self):
        import ast

        tools = pathlib.Path(__file__).resolve().parent.parent / "src" / "tools"
        problems = []
        for path in sorted(tools.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = [t for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
                    targets = [node.target]
                for target in targets:
                    if target.id in ("POOLING_METHOD", "LENGTH_RATIO_POWER"):
                        problems.append(f"{path.name}:{node.lineno} defines {target.id}")
                if isinstance(node, ast.Call):
                    function = node.func
                    name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
                    if name in ("pooled_vectors", "length_adjusted_similarity"):
                        problems.append(f"{path.name}:{node.lineno} calls {name}")
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
