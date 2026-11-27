"""utilities.Prefilter_Calibration: the prefilter's similarity (dense and streamed),
the calibration sample of a run's missing pairs, and the network file's record."""

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
        # The quantile's one copy and what np.quantile needs besides (measured 10.65 in all).
        self.assertLess(peak, 12 * count + 1_000_000)
        self.assertEqual(len(picked), 1_000)
        # Without the quantile, what is left is the bands: one byte each (measured 2.4 with the mask and the members),
        # not the eight of an int64 array.
        edges = np.quantile(values, np.linspace(0.0, 1.0, calibration.BINS + 1))
        with mock.patch.object(calibration, "_BAND_CHUNK", 10_000), mock.patch.object(
            calibration.np, "quantile", lambda *args, **kwargs: edges
        ):
            tracemalloc.start()
            try:
                base = tracemalloc.get_traced_memory()[0]
                again, _ = calibration._draw_by_bands(values, 1_000, calibration.BINS, calibration.SEED)
                bands_peak = tracemalloc.get_traced_memory()[1] - base
            finally:
                tracemalloc.stop()
        np.testing.assert_array_equal(again, picked)
        self.assertLess(bands_peak, 4 * count + 1_000_000)

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


class StreamedPairSimilarityTests(unittest.TestCase):
    """Memory-bounded StreamedPairSimilarity exactness, caching, and edge cases."""

    def test_streamed_pair_similarity_exact_percentile_keys_and_edge_cases(self):
        import tracemalloc

        rng = np.random.default_rng(2026)
        n = 140
        pooled = rng.standard_normal((n, 32)).astype(np.float32)
        pooled[7] = 0.0  # zero vector
        lengths = rng.integers(30, 300, n)
        lengths[11] = 0  # zero length

        # Test normal bracket, tiny bracket forcing refinement, and multiple block sizes.
        for block, max_br in ((4_000_000, 4_000_000), (140 * 17, 25), (140 * 9, 5)):
            with self.subTest(block=block, max_br=max_br), mock.patch.multiple(
                calibration,
                _SIMILARITY_BLOCK=block,
                _MAX_BRACKET_VALUES=max_br,
            ):
                sim = calibration.StreamedPairSimilarity(pooled, lengths, 2.0)
                # Materialize exact streamed upper triangle to verify 100% bit-exact percentile & keys.
                upper_vals = []
                upper_keys = []
                for s, e, blk in sim.iter_upper_blocks():
                    m = sim._upper_mask(s, e, blk.shape[1])
                    rr, cc = np.nonzero(m)
                    upper_vals.append(blk[rr, cc])
                    upper_keys.append((rr.astype(np.int64) + s) * n + (cc.astype(np.int64) + s))
                all_vals = np.concatenate(upper_vals)
                all_keys = np.concatenate(upper_keys)

                for q in (0.0, 1.5, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0):
                    want_t = float(np.percentile(all_vals, q))
                    got_t = sim.upper_percentile(q)
                    self.assertEqual(got_t, want_t, f"q={q}")
                    got_t2, got_keys = sim.upper_percentile_and_keys(q)
                    self.assertEqual(got_t2, want_t, f"q={q}")
                    want_keys = all_keys[all_vals >= want_t]
                    np.testing.assert_array_equal(got_keys, want_keys)
                    self.assertEqual(sim.count_upper_at_or_above(want_t), int(want_keys.size))

                # Also check proximity to dense full-matrix length_adjusted_similarity (<= 1e-6 float32 ULP).
                dense = calibration.length_adjusted_similarity(pooled, lengths, 2.0)
                rr_sample = rng.integers(0, n, size=200)
                cc_sample = rng.integers(0, n, size=200)
                np.testing.assert_allclose(
                    sim.at(rr_sample, cc_sample), dense[rr_sample, cc_sample], atol=1e-6
                )

        # All-identical vectors (100% ties) and tiny n < 2 edge cases.
        tied_pooled = np.ones((80, 16), dtype=np.float32)
        tied_lens = np.full(80, 100, dtype=np.int32)
        with mock.patch.object(calibration, "_MAX_BRACKET_VALUES", 10):
            tied_sim = calibration.StreamedPairSimilarity(tied_pooled, tied_lens, 2.0)
            t_tied, k_tied = tied_sim.upper_percentile_and_keys(50.0)
            self.assertAlmostEqual(t_tied, 1.0, places=6)
            self.assertEqual(k_tied.size, 80 * 79 // 2)

        single = calibration.StreamedPairSimilarity(pooled[:1], lengths[:1], 2.0)
        self.assertIsNone(single.upper_percentile(50.0))
        self.assertEqual(single.upper_percentile_and_keys(50.0)[1].size, 0)

        # Empty pooled inputs.
        empty_dense = calibration.length_adjusted_similarity(np.zeros((0,), dtype=np.float32), [], 2.0)
        self.assertEqual(empty_dense.shape, (0, 0))
        empty_stream = calibration.StreamedPairSimilarity(np.zeros((0,), dtype=np.float32), [], 2.0)
        self.assertEqual(empty_stream.shape, (0, 0))
        self.assertIsNone(empty_stream.upper_percentile(50.0))
        self.assertEqual(empty_stream.upper_percentile_and_keys(50.0)[1].size, 0)

        # Memory check: N=1500 dense would allocate ~27 MB (12*n^2); streamed stays well below dense.
        n_mem = 1500
        p_mem = rng.standard_normal((n_mem, 16)).astype(np.float32)
        l_mem = rng.integers(50, 200, n_mem)
        with mock.patch.object(calibration, "_SIMILARITY_BLOCK", 60_000):
            tracemalloc.start()
            try:
                s_mem = calibration.StreamedPairSimilarity(p_mem, l_mem, 2.0)
                _t_m, _k_m = s_mem.upper_percentile_and_keys(99.0)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        # Dense 12*n^2 would be ~27 MB; streamed peak should stay under 6 MB.
        self.assertLess(peak, 6 * 1024 * 1024)

    def test_streamed_at_matches_upper_strip_rows_lerp_0ulp_and_threshold_cache(self):
        rng = np.random.default_rng(31415)
        for _ in range(5):
            n = 60
            pooled = rng.standard_normal((n, 128)).astype(np.float32)
            lengths = rng.integers(40, 350, n)
            sim = calibration.StreamedPairSimilarity(pooled, lengths, 2.0)

            rr, cc = np.triu_indices(n, k=1)
            at_vals = sim.at(rr, cc)
            strip_vals = []
            for s, e, blk in sim.iter_upper_blocks():
                m = sim._upper_mask(s, e, blk.shape[1])
                strip_vals.append(blk[m])
            strip_all = np.concatenate(strip_vals)
            np.testing.assert_array_equal(at_vals, strip_all)
            full_rows = sim.rows(0, n)
            np.testing.assert_array_equal(at_vals, full_rows[rr, cc])

        # _finish_bracket / _lerp matches NumPy 2.x np.percentile to 0 ULP (incl. frac >= 0.5).
        vals = np.array(
            [0.10000000000000009, 0.30000000000000004, 0.7000000000000001, 0.9999999999999999],
            dtype=np.float64,
        )
        for q in (10.0, 20.0, 33.0, 60.0, 66.6666666667, 75.0, 83.3333333333, 90.0):
            want = float(np.percentile(vals, q))
            h = (q / 100.0) * (vals.size - 1)
            k_lo = int(np.floor(h))
            k_hi = int(np.ceil(h))
            frac = h - k_lo
            got = calibration.StreamedPairSimilarity._finish_bracket(
                vals.copy(), 0, k_lo, k_hi, frac
            )
            self.assertEqual(got, want, f"q={q}")

        # upper_percentile_and_keys caches _last_threshold_count so count_upper_at_or_above hits cache.
        sim_c = calibration.StreamedPairSimilarity(
            rng.standard_normal((50, 32)).astype(np.float32),
            rng.integers(50, 200, 50),
            2.0,
        )
        t_val, k_arr = sim_c.upper_percentile_and_keys(75.0)
        self.assertEqual(sim_c._last_threshold_count, (t_val, int(k_arr.size)))
        with mock.patch.object(sim_c, "iter_upper_blocks", side_effect=AssertionError("should use cache")):
            self.assertEqual(sim_c.count_upper_at_or_above(t_val), int(k_arr.size))


if __name__ == "__main__":
    unittest.main()
