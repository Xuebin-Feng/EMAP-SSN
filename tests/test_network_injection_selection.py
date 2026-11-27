"""Tests of Network_Injection pair selection (cut rule), 64-bit pair keys,
final merge equivalence, and end-to-end cut-rule execution against reference oracles.
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(SRC_DIR, "utilities")
TOOLS_DIR = os.path.join(SRC_DIR, "tools")
for _path in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    from utilities import Prefilter_Calibration
    import Network_Injection as network_injection

from tests.embedding_fixtures import write_embedding_arrays  # noqa: E402
from tests.network_injection_fixtures import (  # noqa: E402
    oracle_cut_rule,
    oracle_merge,
    oracle_pending,
    random_world,
    world_similarity,
    write_batch_file,
)


class NetworkInjectionSelectionTests(unittest.TestCase):
    """The numpy pair selection and merge equal the original per-pair loops."""

    def _glue(self, old_headers, edges, new_headers, random_orientation=None):
        """old_to_new maps and the two ends of every old edge, as run_injection builds them."""
        new_idx = {h: idx for idx, h in enumerate(new_headers)}
        old_to_new = np.array([new_idx.get(h, -1) for h in old_headers], dtype=np.int64)
        old_i = np.array([e[0] for e in edges], dtype=np.int64)
        old_j = np.array([e[1] for e in edges], dtype=np.int64)
        if random_orientation is not None:
            flip = random_orientation.random(len(edges)) < 0.5
            old_i, old_j = np.where(flip, old_j, old_i), np.where(flip, old_i, old_j)
        return old_i, old_j, old_to_new[old_i], old_to_new[old_j]

    KINDS = ("dense", "sparse_record", "sparse_plain", "sparse_all_dropped", "sparse_no_edge")

    def test_cut_rule_helpers_equal_the_oracle(self):
        set_by_seen = {"s_min": 0, "t'": 0}
        seen_kinds = set()
        for seed, large in [(s, False) for s in range(150)] + [(s, True) for s in range(300, 310)]:
            rng = np.random.default_rng(seed)
            kind = self.KINDS[seed % len(self.KINDS)]
            embeddings, old_headers, new_headers, edges, calibration = random_world(rng, kind, large)
            new_n = len(new_headers)
            old_n = len(old_headers)
            if large:
                self.assertGreater(new_n, 300)  # keys pass 65,535
            similarity = world_similarity(embeddings, new_headers)
            old_i, old_j, edge_u, edge_v = self._glue(old_headers, edges, new_headers, rng)
            skip = np.zeros(len(edges), dtype=bool)
            skip[list(calibration)] = True
            expected = oracle_cut_rule(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers, similarity, calibration
            )
            block_cells = int(rng.choice([1, new_n, 2 * new_n + 1, 7, 10 ** 6]))
            chunk = int(rng.choice([977, 1 << 22] if large else [1, 2, 3, 1 << 22]))
            with self.subTest(seed=seed, kind=kind, block_cells=block_cells, chunk=chunk), \
                    mock.patch.object(network_injection, "_PAIR_BLOCK_CELLS", block_cells), \
                    mock.patch.object(network_injection, "_EDGE_CHUNK", chunk), \
                    mock.patch.object(network_injection, "_PENDING_CHUNK", chunk):
                cached = network_injection._cached_pair_keys(edge_u, edge_v, new_n)
                self.assertEqual(cached.dtype, np.int64)
                self.assertEqual(cached.tolist(), sorted(expected["cached"]))
                old_set = set(old_headers)
                is_old = np.array([h in old_set for h in new_headers])
                total_pairs = new_n * (new_n - 1) // 2
                if expected["sparse"] and len(edges) > 0:
                    kept_share = (len(edges) - len(calibration)) / (old_n * (old_n - 1) // 2)
                    cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
                        similarity, edge_u, edge_v, skip, kept_share
                    )
                    self.assertEqual(lowest, expected["lowest"])
                    self.assertEqual(s_min, expected["s_min"])
                    self.assertEqual(t_prime, expected["t_prime"])
                    self.assertEqual(cut, expected["cut"])
                    self.assertEqual(set_by, expected["set_by"])
                    if set_by:
                        set_by_seen[set_by] += 1
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(
                        new_n, is_old, cached, similarity, cut
                    )
                    kept_count = network_injection._pairs_at_or_above(similarity, cut)
                    self.assertEqual(kept_count / total_pairs if total_pairs else 0.0, expected["kept_after"])
                    # The properties of the rule, checked on the helpers' own output.
                    self.assertEqual(cut, min(v for v in (s_min, t_prime) if v is not None))
                    self.assertGreater(kept_count, kept_share * total_pairs - 1)  # >= K up to rounding
                    first_keys = set(required.tolist())
                    self.assertTrue(set(cached.tolist()) <= first_keys)  # no old edge is dropped
                    for key in first_keys - set(cached.tolist()):
                        self.assertGreaterEqual(similarity[key // new_n, key % new_n], cut)
                    pair_rows, pair_columns = np.triu_indices(new_n, 1)
                    at_or_above = pair_rows[similarity[pair_rows, pair_columns] >= cut] * new_n \
                        + pair_columns[similarity[pair_rows, pair_columns] >= cut]
                    self.assertTrue(set(at_or_above.tolist()) <= first_keys)  # every pair at or above the cut
                elif expected["sparse"]:
                    # A sparse input without an edge: every pair with a new sequence, no old-old pair.
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(new_n, is_old, cached, None, None)
                else:
                    seen_kinds.add(kind)
                    required = network_injection._required_pair_keys(new_n, is_old, cached, None, None)
                self.assertEqual(required.dtype, np.int64)
                self.assertEqual(required.tolist(), sorted(expected["required"]))

                computed = {
                    int(k) for k in rng.choice(
                        new_n * new_n, int(rng.integers(0, 10)), replace=True
                    )
                } | set(sorted(expected["required"])[: int(rng.integers(0, 3))])
                headers = [f"h{k}" for k in range(new_n)]
                pending = oracle_pending(new_n, expected["required"], expected["cached"], computed)
                self.assertEqual(
                    list(network_injection._iter_pending_pairs(
                        headers, expected["required"], expected["cached"], computed
                    )),
                    [(p // new_n, p % new_n, headers[p // new_n], headers[p % new_n]) for p in pending],
                )
                keys = network_injection._pending_keys(
                    required, cached, np.array(sorted(computed), dtype=np.int64), new_n
                )
                self.assertEqual(keys.tolist(), pending)
                np.testing.assert_array_equal(
                    np.bincount(keys // new_n, minlength=new_n),
                    np.bincount([p // new_n for p in pending], minlength=new_n),
                )
                kinds = [0, 0, 0]
                for p in pending:
                    kinds[2 - int(is_old[p // new_n]) - int(is_old[p % new_n])] += 1
                self.assertEqual(list(network_injection._pending_kinds(keys, is_old, new_n)), kinds)
        # Both ways the rule can go, and every kind of input, were exercised.
        self.assertGreater(set_by_seen["s_min"], 5)
        self.assertGreater(set_by_seen["t'"], 5)
        self.assertEqual(seen_kinds, set(self.KINDS))

    def test_share_cut_is_the_percentile_aligns_prefilter_takes(self):
        rng = np.random.default_rng(3)
        for n in (2, 3, 7, 40):
            similarity = rng.random((n, n))
            for share in (0.0, 0.1, 0.5, 0.62, 0.999, 1.0):
                with self.subTest(n=n, share=share):
                    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
                    expected = float(np.percentile(similarity[upper], 100.0 * (1.0 - share)))
                    self.assertEqual(network_injection._share_cut(similarity, share), expected)
        self.assertIsNone(network_injection._share_cut(np.ones((1, 1)), 0.5))

    def test_a_network_align_made_keeps_exactly_its_own_pairs(self):
        # Align's prefilter keeps the pairs at or above the strength percentile; the share of those
        # pairs, fed back into the cut, must select the same pairs again (a no-op injection queues nothing).
        rng = np.random.default_rng(11)
        for n in (5, 12, 60):
            similarity = rng.random((n, n))
            upper = np.triu(np.ones((n, n), dtype=bool), k=1)
            total = n * (n - 1) // 2
            for strength in (10.0, 25.0, 50.0, 75.0, 90.0):
                with self.subTest(n=n, strength=strength):
                    threshold = float(np.percentile(similarity[upper], strength))
                    kept = (similarity >= threshold) & upper
                    share = np.count_nonzero(kept) / total
                    cut = network_injection._share_cut(similarity, share)
                    again = (similarity >= cut) & upper
                    np.testing.assert_array_equal(again, kept)

    def test_lowest_edge_similarity_is_the_first_edge_among_equal_values(self):
        similarity = np.array([[1.0, 0.5, 0.5, 0.9], [0.5, 1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.7], [0.9, 0.5, 0.7, 1.0]])
        edge_u = np.array([0, 2, 0, 1, 1, 0])
        edge_v = np.array([3, 3, 1, 2, 3, 2])
        skip = np.zeros(6, dtype=bool)
        for chunk in (1, 2, 4, 100):
            with mock.patch.object(network_injection, "_EDGE_CHUNK", chunk):
                self.assertEqual(
                    network_injection._lowest_edge_similarity(similarity, edge_u, edge_v, skip),
                    (0.5, 2),
                )
                skip[2] = True
                self.assertEqual(
                    network_injection._lowest_edge_similarity(similarity, edge_u, edge_v, skip),
                    (0.5, 3),
                )
                skip[2] = False
                # An edge that lost an end (-1) never counts.
                self.assertEqual(
                    network_injection._lowest_edge_similarity(
                        similarity, np.array([0, -1]), np.array([3, 1]), np.zeros(2, bool)
                    ),
                    (0.9, 0),
                )
        # An edge at similarity 1.0 is a lowest edge like any other; a NaN never is.
        ones = np.ones((3, 3))
        self.assertEqual(
            network_injection._lowest_edge_similarity(ones, np.array([0]), np.array([1]), np.zeros(1, bool)),
            (1.0, 0),
        )
        with_nan = np.array([[1.0, np.nan, 0.8], [np.nan, 1.0, 0.7], [0.8, 0.7, 1.0]])
        self.assertEqual(
            network_injection._lowest_edge_similarity(
                with_nan, np.array([0, 0, 1]), np.array([1, 2, 2]), np.zeros(3, bool)
            ),
            (0.7, 2),
        )
        # No counted edge: nothing.
        self.assertEqual(
            network_injection._lowest_edge_similarity(
                similarity, np.array([0]), np.array([1]), np.ones(1, bool)
            ),
            (None, -1),
        )

    def test_the_cut_is_the_lower_of_s_min_and_t_prime_in_both_cases(self):
        # Six sequences, 15 pairs. The off-diagonal similarities are 0.0, 0.1, ... 1.4 / 1.5 (distinct).
        n = 6
        similarity = np.zeros((n, n))
        values = iter(np.arange(15) / 10.0)
        for i in range(n):
            similarity[i, i] = 1.0
            for j in range(i + 1, n):
                similarity[i, j] = similarity[j, i] = next(values)
        # Pair (0, 1) has 0.0 ... pair (4, 5) 1.4.
        edge_u = np.array([2, 3])
        edge_v = np.array([4, 5])
        skip = np.zeros(2, dtype=bool)
        low_edge = similarity[2, 4]
        high_edge = similarity[3, 5]
        # A tiny kept share: t' is high; the lowest edge lies below it, so it sets the cut.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(similarity, edge_u, edge_v, skip, 2 / 15)
        self.assertEqual((set_by, cut, s_min, lowest), ("s_min", low_edge, low_edge, 0))
        self.assertGreater(t_prime, low_edge)
        # A large kept share: t' is low, below both edges: it sets the cut and keeps K.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(similarity, edge_u, edge_v, skip, 14 / 15)
        self.assertEqual(set_by, "t'")
        self.assertEqual(cut, t_prime)
        self.assertLess(t_prime, low_edge)
        self.assertLessEqual(t_prime, min(low_edge, high_edge))
        # Equal: the lowest edge at exactly t' does not set the cut (t' does), same value.
        exact = float(network_injection._share_cut(similarity, 0.5))
        edge_exact_u, edge_exact_v = np.nonzero(np.triu(similarity == exact, 1))
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
            similarity, edge_exact_u, edge_exact_v, np.zeros(edge_exact_u.size, bool), 0.5
        ) if edge_exact_u.size else (exact, exact, 0, exact, "t'")
        self.assertEqual((cut, set_by), (exact, "t'"))
        # No edge in the new set: the cut is t'.
        cut, s_min, lowest, t_prime, set_by = network_injection._similarity_cut(
            similarity, np.array([-1]), np.array([4]), skip[:1], 0.5
        )
        self.assertEqual((set_by, s_min, lowest, cut), ("t'", None, -1, t_prime))
        # A new set with fewer than two sequences has no pair: the cut is infinite.
        cut, _, _, t_prime, set_by = network_injection._similarity_cut(
            np.ones((1, 1)), np.array([0]), np.array([0]), skip[:1], 0.5
        )
        self.assertEqual((cut, t_prime, set_by), (float("inf"), None, None))

    def test_a_new_pair_at_exactly_the_cut_is_required(self):
        # Pair (0, 1) is the lowest old edge; the pair (2, 3) has exactly the same similarity.
        similarity = np.array([
            [1.0, 0.6, 0.9, 0.9],
            [0.6, 1.0, 0.9, 0.9],
            [0.9, 0.9, 1.0, 0.6],
            [0.9, 0.9, 0.6, 1.0],
        ])
        is_old = np.array([True, True, False, False])
        cached = np.array([0 * 4 + 1], dtype=np.int64)
        required = network_injection._required_pair_keys(4, is_old, cached, similarity, 0.6)
        self.assertEqual(required.tolist(), [1, 2, 3, 6, 7, 11])
        # Just above the cut the cached edge stays (never dropped), the pair at 0.6 does not.
        self.assertEqual(
            network_injection._required_pair_keys(4, is_old, cached, similarity, 0.61).tolist(),
            [1, 2, 3, 6, 7],
        )
        # Old-old pairs without an edge are required too when they reach the cut.
        both_old = np.array([True, True, True, True])
        self.assertEqual(
            network_injection._required_pair_keys(4, both_old, cached, similarity, 0.9).tolist(),
            [1, 2, 3, 6, 7],
        )

    def test_pair_keys_stay_exact_far_beyond_32_bits(self):
        # N = 100,000: keys reach 10 ** 10, past both 2 ** 31 and 2 ** 32.
        n = 100_000
        edge_u = np.array([99998, 5, 70000, -1, 3, 5], dtype=np.int64)
        edge_v = np.array([99999, 99990, 3, 7, -1, 99990], dtype=np.int64)
        cached = network_injection._cached_pair_keys(edge_u, edge_v, n)
        expected_cached = sorted({99998 * n + 99999, 5 * n + 99990, 3 * n + 70000})
        self.assertEqual(cached.dtype, np.int64)
        self.assertEqual(cached.tolist(), expected_cached)
        self.assertGreater(int(cached.max()), 2 ** 33)

        extra = [10 * n + 11, 99990 * n + 99991, 40_000 * n + 80_000]
        # A wrong-orientation key and one whose row is past the sequence count never queue.
        invalid = [50 * n + 50, 70_000 * n + 60_000, n * n + 1]
        required = network_injection._as_sorted_keys(set(expected_cached + extra + invalid))
        self.assertEqual(required.tolist(), sorted(expected_cached + extra + invalid))
        computed = network_injection._as_sorted_keys([extra[0], 123])
        pending = network_injection._pending_keys(required, cached, computed, n)
        self.assertEqual(pending.tolist(), sorted(extra[1:]))

        position, found = network_injection._locate(required, np.array(extra + [7, 10 ** 10 + 5], dtype=np.int64))
        self.assertEqual(found.tolist(), [True, True, True, False, False])
        np.testing.assert_array_equal(required[position[:3]], extra)
        self.assertEqual(
            network_injection._members(cached, np.array(expected_cached[::-1] + [1], dtype=np.int64)).tolist(),
            [True, True, True, False],
        )

        headers = [f"h{k}" for k in range(n)]
        queued = list(network_injection._iter_pending_keys(headers, pending, n))
        self.assertEqual(
            queued,
            [(k // n, k % n, headers[k // n], headers[k % n]) for k in sorted(extra[1:])],
        )
        self.assertEqual(
            list(network_injection._iter_pending_pairs(headers, set(required.tolist()), set(cached.tolist()), set(computed.tolist()))),
            queued,
        )

        # Flat condensed indices of an old network of 100,000 sequences pass 4.9 * 10 ** 9.
        flat = np.array([5_000_000_000, 7, 4_999_000_000, 7], dtype=np.int64)
        index = network_injection._OldEdgeIndex(flat)
        self.assertEqual(
            index[np.array([5_000_000_000, 7, 4_999_000_000, 8, 5_000_000_001])].tolist(),
            [0, 3, 2, -1, -1],
        )
        self.assertEqual(network_injection._sorted_unique(np.array([10 ** 10, 3, 10 ** 10, 3], dtype=np.int64)).tolist(), [3, 10 ** 10])

    def test_compile_equals_the_original_merge_on_random_inputs(self):
        # The large cases have keys past 65,535 and store the batch rows as uint16.
        for seed, large in [(s, False) for s in range(40)] + [(s, True) for s in range(100, 104)]:
            rng = np.random.default_rng(1000 + seed)
            embeddings, old_headers, new_headers, edges, _calibration = random_world(
                rng, ("dense", "sparse_plain")[seed % 2], large
            )
            batch_index_dtype = np.uint16 if large else np.uint32
            new_n = len(new_headers)
            old_n = len(old_headers)
            old_i = np.array([e[0] for e in edges], dtype=np.int64)
            old_j = np.array([e[1] for e in edges], dtype=np.int64)
            count = len(edges)
            old_arrays = (
                (np.arange(count) + 10.5).astype(np.float32),
                (np.arange(count) + 3).astype(np.uint16),
                (np.arange(count) + 500.25).astype(np.float32),
                (np.arange(count) + 4).astype(np.uint16),
            )
            flat = old_i * old_n - old_i * (old_i + 1) // 2 + (old_j - old_i - 1)
            legacy_map = np.full(old_n * (old_n - 1) // 2, -1, dtype=np.int64)
            legacy_map[flat] = np.arange(count)
            old_index = {h: k for k, h in enumerate(old_headers)}
            decided = oracle_cut_rule(
                old_headers, old_i.tolist(), old_j.tolist(), new_headers,
                world_similarity(embeddings, new_headers), set(),
            )
            cached, required = decided["cached"], decided["required"]
            # Batches: pairs to compute, a pair written twice, pairs that are not
            # required, a cached pair with a wrong score, and reversed (j, i) rows.
            everything = [(i, j) for i in range(new_n) for j in range(i + 1, new_n)]
            batch_pairs = [
                [everything[k] for k in rng.permutation(len(everything))[: int(rng.integers(0, len(everything) + 1))]]
                for _ in range(int(rng.integers(1, 4)))
            ]
            batches = []
            with tempfile.TemporaryDirectory() as temp_dir:
                results_dir = os.path.join(temp_dir, "batches")
                os.makedirs(results_dir)
                for number, pairs in enumerate(batch_pairs):
                    pairs = [p[::-1] if rng.random() < 0.3 else p for p in pairs]
                    scores = [(float(100 * number + n + 0.5), n % 50 + 1, float(7 * number + n + 0.25), n % 40 + 2)
                              for n in range(len(pairs))]
                    write_batch_file(results_dir, f"batch_{number:05d}.h5", pairs, scores, batch_index_dtype)
                    batches.append((
                        np.array([p[0] for p in pairs], dtype=np.int64),
                        np.array([p[1] for p in pairs], dtype=np.int64),
                        np.array([s[0] for s in scores], np.float32),
                        np.array([s[1] for s in scores], np.uint16),
                        np.array([s[2] for s in scores], np.float32),
                        np.array([s[3] for s in scores], np.uint16),
                    ))
                expected = oracle_merge(
                    new_n, required, cached, new_headers, old_index, old_arrays, legacy_map,
                    batches if batch_pairs else [],
                )
                kinds = {
                    "sets and legacy map": (set(required), set(cached), legacy_map),
                    "arrays and edge index": (
                        np.array(sorted(required), dtype=np.int64),
                        np.array(sorted(cached), dtype=np.int64),
                        network_injection._OldEdgeIndex(flat),
                    ),
                }
                for label, (req, cac, index_map) in kinds.items():
                    output = os.path.join(temp_dir, label.split()[0], "net_[model]_network.h5")
                    with self.subTest(seed=seed, inputs=label), mock.patch.object(
                        network_injection, "RESULTS_DIR", results_dir
                    ), mock.patch.object(
                        network_injection, "FINAL_OUTPUT_NET", output
                    ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        network_injection.compile_final_output(
                            new_headers=new_headers, seq_lens=[5] * new_n,
                            required_pairs=req, cached_old_pairs=cac, new_N=new_n,
                            old_header_to_idx=old_index,
                            old_l_score=old_arrays[0], old_l_len=old_arrays[1],
                            old_g_score=old_arrays[2], old_g_len=old_arrays[3],
                            actual_idx_map=index_map, current_checksum="c",
                            model_name="model", saving_mode="float32",
                            gap_penalties=[-2.0, 0.0],
                        )
                        with h5py.File(output, "r") as hf:
                            for name in ("i", "j", "l_score", "l_len", "g_score", "g_len"):
                                np.testing.assert_array_equal(hf[name][:], expected[name], err_msg=name)
                                self.assertEqual(hf[name].dtype, expected[name].dtype)

    def _run_injection(self, temp_dir, rng, kind, batch_size, large=False):
        """Run run_injection on a random world with stubbed alignment; return what it did.

        large: 300 to 380 sequences, so pair keys pass 65,535; the old network's
        i/j are uint16 as always and the left-over batch stores its rows as uint16.
        """
        world, old_headers, new_headers, edges, calibration = random_world(rng, kind, large)
        embed_dir = os.path.join(temp_dir, "embed")
        network_dir = os.path.join(temp_dir, "network")
        os.makedirs(embed_dir)
        os.makedirs(network_dir)
        embeddings = os.path.join(embed_dir, "newset_[esm2_t6_8m]_embeddings.h5")
        write_embedding_arrays(
            embeddings,
            {header: world[header].astype(np.float32) for header in new_headers},
            sequences=["K" * len(world[h]) for h in new_headers],
            model_name="esm2_t6_8m",
            saving_mode="float32",
        )
        old_n = len(old_headers)
        count = len(edges)
        with h5py.File(os.path.join(network_dir, "oldset_[esm2_t6_8m]_network.h5"), "w") as hf:
            hf.attrs["model_name"] = "esm2_t6_8m"
            hf.attrs["saving_mode"] = "float32"
            hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
            hf.attrs["matmul_precision"] = "ieee_fp32"
            hf.create_dataset("headers", data=np.array(old_headers, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            hf.create_dataset("seq_lens", data=np.ones(old_n, np.uint16))
            for name, data in (
                ("i", [e[0] for e in edges]), ("j", [e[1] for e in edges]),
                ("l_score", np.arange(count) + 10.5), ("l_len", np.arange(count) + 1),
                ("g_score", np.arange(count) + 500.25), ("g_len", np.arange(count) + 2),
            ):
                dtype = np.float32 if name.endswith("score") else np.uint16
                hf.create_dataset(name, data=np.asarray(data, dtype))
            if calibration:
                ordered = sorted(calibration)
                Prefilter_Calibration.write_record(hf, Prefilter_Calibration.PrefilterRecord(
                    strength=50.0, threshold=0.9, pooling="max", length_power=2.0, kept=count, skipped=count,
                    sample=Prefilter_Calibration.CalibrationSample(
                        np.array([edges[k][0] for k in ordered]),
                        np.array([edges[k][1] for k in ordered]),
                        np.ones(len(ordered)),
                    ),
                ), np.uint16)
        similarity = world_similarity(world, new_headers)
        old_i = [e[0] for e in edges]
        old_j = [e[1] for e in edges]
        decided = oracle_cut_rule(old_headers, old_i, old_j, new_headers, similarity, calibration)
        cached, required = decided["cached"], decided["required"]
        sparse = decided["sparse"]
        new_n = len(new_headers)
        number = lambda header: int(header[1:])
        stub = lambda i, j: (
            1000.5 + 10 * number(new_headers[i]) + number(new_headers[j]),
            number(new_headers[i]) + number(new_headers[j]) + 1,
            2000.25 + 7 * number(new_headers[i]) + number(new_headers[j]),
            number(new_headers[i]) + number(new_headers[j]) + 2,
        )
        # A batch left by an earlier run: some pending pairs, a pair no longer
        # required, and a cached pair with a wrong score; all must be handled.
        pending = oracle_pending(new_n, required, cached, set())
        resumed = {}
        computed_all = set()
        if rng.random() < 0.85 and pending:
            results_dir = os.path.join(network_dir, "newset_[esm2_t6_8m]_network_temp")
            os.makedirs(results_dir)
            chosen = [pending[k] for k in rng.permutation(len(pending))[: int(rng.integers(1, len(pending) + 1))]]
            rows = [(p // new_n, p % new_n) for p in chosen]
            scores = [(7000.5 + n, n + 1, 8000.25 + n, n + 2) for n in range(len(rows))]
            resumed = {p: scores[n] for n, p in enumerate(chosen)}
            extra = [p for p in range(new_n * new_n)
                     if p // new_n < p % new_n and p not in required][:1] + sorted(cached)[:1]
            for p in extra:
                rows.append((p // new_n, p % new_n))
                scores.append((9999.5, 9, 9999.25, 9))
            computed_all = set(resumed) | set(extra)
            write_batch_file(
                results_dir, "batch_00000.h5", rows, scores, np.uint16 if large else np.uint32
            )
            with h5py.File(os.path.join(results_dir, "batch_00000.h5"), "a") as hf:
                hf.attrs["embedding_checksum"] = network_injection.calculate_file_hash(embeddings)
                hf.attrs["model_name"] = "esm2_t6_8m"
                hf.attrs["saving_mode"] = "float32"
                hf.attrs["gap_penalties"] = np.asarray([-2.0, 0.0], np.float32)
                hf.attrs["matmul_precision"] = "ieee_fp32"
        queue = oracle_pending(new_n, required, cached, set(resumed))

        def compute(tasks, *_args, **_kwargs):
            return [(i, j, *stub(i, j)) for i, j, _h_i, _h_j in tasks]

        cpu_tasks = mock.Mock(side_effect=compute)
        plan = mock.Mock(variant="scalar", lanes=1)
        plan.candidate.is_cpu = True
        plan.candidate.device = torch.device("cpu")
        assigned = (
            "_old_seq_set", "_new_seq_set", "_model_name", "RESULTS_DIR",
            "FINAL_OUTPUT_NET", "CONFIG_FILE", "LOCAL_GAP_P", "GLOBAL_GAP_P",
        )
        printed = io.StringIO()
        with mock.patch.multiple(
            network_injection,
            **{name: getattr(network_injection, name) for name in assigned},
            OLD_NETWORK="oldset_[esm2_t6_8m]_network.h5",
            NEW_EMBEDDINGS="newset_[esm2_t6_8m]_embeddings.h5",
            EMBED_DIR=embed_dir, NETWORK_DIR=network_dir,
            EXECUTION_MODE="auto", DEVICE_SELECTION="auto", BATCH_SIZE=batch_size,
            set_start_method=mock.DEFAULT,
            _benchmark_injection_plans=mock.Mock(return_value=[plan]),
            process_cpu_tasks=cpu_tasks,
        ), redirect_stdout(printed), redirect_stderr(io.StringIO()), mock.patch.object(
            Prefilter_Calibration, "pair_similarity", wraps=Prefilter_Calibration.pair_similarity
        ) as similarity_probe:
            network_injection.run_injection()
        with h5py.File(os.path.join(network_dir, "newset_[esm2_t6_8m]_network.h5"), "r") as hf:
            actual = {name: hf[name][:] for name in ("i", "j", "l_score", "l_len", "g_score", "g_len")}
        expected_rows = []
        old_position = {h: n for n, h in enumerate(old_headers)}
        edge_row = {(min(e), max(e)): n for n, e in enumerate(edges)}
        for p in sorted(required):
            if p in cached:
                u = old_position[new_headers[p // new_n]]
                v = old_position[new_headers[p % new_n]]
                k = edge_row[(min(u, v), max(u, v))]
                row = (k + 10.5, k + 1, k + 500.25, k + 2)
            elif p in resumed:
                row = resumed[p]
            else:
                row = stub(p // new_n, p % new_n)
            expected_rows.append((p // new_n, p % new_n, *row))
        return {
            "tasks": [list(call.args[0]) for call in cpu_tasks.call_args_list],
            "queue": queue, "new_n": new_n, "printed": printed.getvalue(),
            "actual": actual, "expected_rows": expected_rows,
            "decided": decided, "required": required, "cached": cached,
            "resumed": resumed, "computed_all": computed_all,
            "calibration": calibration, "sparse": sparse,
            "old_headers": old_headers, "new_headers": new_headers, "edges": edges,
            "similarity": similarity, "similarity_calls": similarity_probe.call_count,
        }

    def test_run_injection_follows_the_cut_rule_end_to_end(self):
        resumed_seen = 0
        kinds_seen = set()
        for seed, large in [(s, False) for s in range(60)] + [(s, True) for s in range(200, 210)]:
            kind = self.KINDS[seed % len(self.KINDS)]
            rng = np.random.default_rng(5000 + seed)
            batch_size = int(rng.choice([997, 500000] if large else [1, 3, 7, 500000]))
            with self.subTest(seed=seed, kind=kind, batch_size=batch_size, large=large), \
                    tempfile.TemporaryDirectory() as temp_dir:
                result = self._run_injection(temp_dir, rng, kind, batch_size, large)
                new_n = result["new_n"]
                if large:
                    self.assertGreater(new_n, 300)  # keys pass 65,535
                queue = result["queue"]
                flat = [task for batch in result["tasks"] for task in batch]
                self.assertEqual([(t[0], t[1]) for t in flat], [(p // new_n, p % new_n) for p in queue])
                self.assertEqual(
                    [len(batch) for batch in result["tasks"]],
                    [len(queue[k:k + batch_size]) for k in range(0, len(queue), batch_size)],
                )
                rows = list(zip(*(result["actual"][n].tolist() for n in ("i", "j", "l_score", "l_len", "g_score", "g_len"))))
                self.assertEqual(rows, result["expected_rows"])
                text = result["printed"]
                self.assertIn(f"Total Required pairs: {len(result['required'])}\n", text)
                self.assertIn(f"Pre-existing cached pairs: {len(result['cached'])}\n", text)
                # Every pair the left-over batch holds that the network requires counts,
                # including its cached extra row.
                self.assertIn(
                    f"Already computed pairs: {len(result['computed_all'] & result['required'])}\n", text
                )
                self.assertIn(f"Pairs queued for calculation: {len(queue)}\n", text)
                decided = result["decided"]
                # The similarity is computed once for a sparse input with an edge, never otherwise.
                self.assertEqual(
                    result["similarity_calls"],
                    1 if decided["sparse"] and decided["K"] is not None else 0,
                )
                if kind == "sparse_no_edge":
                    # Nothing to compare with: every pair with a new sequence is aligned, as on
                    # main, and no pair of two old sequences.
                    new_old_set = set(result["old_headers"])
                    self.assertEqual(
                        set(queue) | set(result["resumed"]),
                        {
                            p for p in range(new_n * new_n)
                            if p // new_n < p % new_n
                            and not (result["new_headers"][p // new_n] in new_old_set
                                     and result["new_headers"][p % new_n] in new_old_set)
                        },
                    )
                    self.assertIn(
                        "every pair with a new sequence is aligned and no pair of two old sequences.", text
                    )
                if kind == "dense":
                    # A complete input needs no similarity: every pair with a new sequence is aligned.
                    self.assertEqual(
                        set(queue) | set(result["resumed"]),
                        {
                            p for p in range(new_n * new_n)
                            if p // new_n < p % new_n
                            and not (result["new_headers"][p // new_n] in set(result["old_headers"])
                                     and result["new_headers"][p % new_n] in set(result["old_headers"]))
                        },
                    )
                old_set = set(result["old_headers"])
                new_headers = result["new_headers"]
                old_ends = [
                    int(new_headers[p // new_n] in old_set) + int(new_headers[p % new_n] in old_set)
                    for p in queue
                ]
                self.assertIn(
                    f"  > Queued by kind: {old_ends.count(2):,} old-old (both sequences were in the input "
                    f"network, but not joined by an edge), {old_ends.count(1):,} old-new, "
                    f"{old_ends.count(0):,} new-new\n",
                    text,
                )
                if decided["sparse"] and decided["K"] is not None:
                    self.assertIn(f"Kept share of the input network, K: {decided['K']:.6f} (", text)
                    if decided["lowest"] != -1:
                        edge = result["edges"][decided["lowest"]]
                        self.assertIn(
                            f"s_min: {decided['s_min']:.6f} (between '{result['old_headers'][edge[0]]}' "
                            f"and '{result['old_headers'][edge[1]]}', score: {decided['lowest'] + 10.5:.4f})\n",
                            text,
                        )
                    else:
                        self.assertIn("s_min does not apply.\n", text)
                    if decided["set_by"] is None:
                        self.assertIn("fewer than two sequences", text)
                    else:
                        self.assertIn(f"t': {decided['t_prime']:.6f}\n", text)
                        self.assertIn(f"Cut: {decided['cut']:.6f}, set by {decided['set_by']}.", text)
                        self.assertIn(
                            f"Kept share of the new set's pairs at or above the cut: "
                            f"{decided['kept_after']:.6f} (input network: {decided['K']:.6f})\n",
                            text,
                        )
                    # The old scan and its lines are gone.
                    self.assertNotIn("below the threshold", text)
                elif decided["sparse"]:
                    self.assertIn("The input network has no edge", text)
                else:
                    self.assertNotIn("Kept share", text)
                kinds_seen.add(kind)
                resumed_seen += bool(result["resumed"])
        self.assertGreater(resumed_seen, 5)
        self.assertEqual(kinds_seen, set(self.KINDS))


if __name__ == "__main__":
    unittest.main()
