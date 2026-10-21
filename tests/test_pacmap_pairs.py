"""utilities.PaCMAP_Pairs: the near, mid-near and further pairs of a PaCMAP layout.

The references are plain Python restatements of the rules in the module's docstring: a node's
near partners are its strongest edges (ties by the earlier edge, NaN last, a repeated partner at
its best rank); a mid-near partner is the second best of six sampled nodes by the best stored
score of the pair, a missing pair counting as the farthest; further partners are distinct nodes
that are neither the node nor a near partner. The pair lookup is compared with a dictionary over
pairs stored in no order, with repeats and NaN scores, for the chunk sizes, worker counts and hash
table sizes that must not change the answer.
"""

import pathlib
import sys
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.network_fixtures import ArraySource
from utilities import Network_Score_Statistics as scan
from utilities import Numba_Threads
from utilities import PaCMAP_Pairs as pairs


def reference_near(u, v, scores, count, neighbors):
    """Each node's strongest partners: higher score, then the earlier edge; NaN after every number."""
    incident = defaultdict(list)
    for edge, (a, b, score) in enumerate(zip(u, v, scores)):
        if a == b:
            continue
        key = (1, 0.0, edge) if score != score else (0, -float(score), edge)
        incident[a].append((key, b))
        incident[b].append((key, a))
    table = np.full((count, neighbors), -1, dtype=np.int32)
    for node in range(count):
        seen = []
        for _, other in sorted(incident[node]):
            if other not in seen:
                seen.append(other)
        table[node, :min(neighbors, len(seen))] = seen[:neighbors]
    return table


def random_edges(rng, count, edge_total, *, ties=False, nan=False):
    u = rng.integers(0, count, edge_total)
    v = rng.integers(0, count, edge_total)
    scores = rng.integers(0, 4, edge_total).astype(np.float64) if ties else rng.random(edge_total)
    if nan:
        scores[rng.random(edge_total) < 0.2] = np.nan
    return u, v, scores


MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


def python_mix(value):
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def python_stream(seed, kind, node):
    """SplitMix64 on the state of one node in one stream, in plain Python integers."""
    state = python_mix((seed + kind * GOLDEN) & MASK)
    state = python_mix((state + (node + 1) * GOLDEN) & MASK)
    while True:
        state = (state + GOLDEN) & MASK
        yield python_mix(state)


# The first draws of seed 42, the mid-near stream, node 0, as written down once from the
# reference above.
FIRST_DRAWS = [7138415436909018950, 16995303060395781456, 1533225871787718299, 11132436533567955750]


def jit_draws(seed, kind, node, count, bound=None):
    """Draws of the kernels' generator. The state is passed back as np.uint64, as it is inside a kernel:
    a bare Python integer would be typed int64 or uint64 by its size and change the arithmetic."""
    state = np.uint64(pairs.node_state(seed, kind, node))
    out = []
    for _ in range(count):
        if bound is None:
            state, value = pairs.next_value(state)
        else:
            state, value = pairs.below(state, bound)
        state = np.uint64(state)
        out.append(int(value))
    return out


class SplitMixTests(unittest.TestCase):
    def test_the_kernels_generator_is_splitmix64(self):
        reference = python_stream(42, pairs.STREAM_MID_NEAR, 0)
        self.assertEqual(jit_draws(42, pairs.STREAM_MID_NEAR, 0, 4), [next(reference) for _ in range(4)])
        self.assertEqual(jit_draws(42, pairs.STREAM_MID_NEAR, 0, 4), FIRST_DRAWS)
        for seed, kind, node in ((0, 1, 0), (7, 2, 123456), (2 ** 62, 3, 5)):
            reference = python_stream(seed, kind, node)
            self.assertEqual(jit_draws(seed, kind, node, 6), [next(reference) for _ in range(6)])

    def test_below_is_uniform_and_in_range(self):
        for bound in (1, 2, 3, 7, 1000, 2 ** 31 - 1):
            draws = jit_draws(42, pairs.STREAM_FURTHER, 3, 2000, bound)
            self.assertTrue(0 <= min(draws) and max(draws) < bound, bound)
        counts = np.bincount(jit_draws(42, 1, 5, 28000, 7), minlength=7)
        self.assertTrue(np.all(np.abs(counts - 4000) < 300), counts)

    def test_below_masks_the_draw_without_a_modulo(self):
        # The kernel takes the low bits of each draw and redraws what is out of range.
        reference = python_stream(42, 1, 5)
        expected = []
        while len(expected) < 50:
            value = next(reference) & 7
            if value < 6:
                expected.append(value)
        self.assertEqual(jit_draws(42, 1, 5, 50, 6), expected)

    def test_streams_depend_on_seed_kind_and_node_only(self):
        first = pairs.node_state(42, 1, 5)
        self.assertEqual(first, pairs.node_state(42, 1, 5))
        for other in (pairs.node_state(43, 1, 5), pairs.node_state(42, 2, 5), pairs.node_state(42, 1, 6)):
            self.assertNotEqual(first, other)


class ConstantsTests(unittest.TestCase):
    def test_counts_follow_the_neighbours(self):
        self.assertEqual([pairs.mid_near_count(k) for k in (1, 2, 3, 10, 15, 50)], [1, 1, 2, 5, 8, 25])
        self.assertEqual([pairs.further_count(k) for k in (2, 10, 50)], [4, 20, 100])
        self.assertEqual(pairs.MID_NEAR_CANDIDATES, 6)

    def test_a_small_network_gets_fewer_mid_near_partners(self):
        # Six distinct sampled nodes other than the node itself, per partner.
        self.assertEqual(pairs.effective_mid_near_count(100, 10), 5)
        self.assertEqual(pairs.effective_mid_near_count(31, 10), 5)
        self.assertEqual(pairs.effective_mid_near_count(30, 10), 4)
        self.assertEqual(pairs.effective_mid_near_count(7, 10), 1)
        for count in (0, 1, 6):
            self.assertEqual(pairs.effective_mid_near_count(count, 10), 0)


class NearPairsTests(unittest.TestCase):
    def test_connected_nodes_ignore_self_loops_and_number_the_rest(self):
        ids, compact = pairs.connected_nodes([0, 2, 5, 5], [1, 2, 4, 4], 8)
        np.testing.assert_array_equal(ids, [0, 1, 4, 5])
        np.testing.assert_array_equal(compact, [0, 1, -1, -1, 2, 3, -1, -1])
        ids, _ = pairs.connected_nodes([3], [3], 5)
        self.assertEqual(len(ids), 0)

    def test_partners_are_the_strongest_edges_for_every_kind_of_input(self):
        rng = np.random.default_rng(0)
        for kind in ({}, {"ties": True}, {"nan": True}, {"ties": True, "nan": True}):
            for count, edge_total, neighbors in ((30, 200, 5), (12, 400, 10), (200, 150, 3), (40, 40, 15)):
                with self.subTest(kind=kind, count=count, edges=edge_total, neighbors=neighbors):
                    u, v, scores = random_edges(rng, count, edge_total, **kind)
                    ids, compact = pairs.connected_nodes(u, v, count)
                    got = pairs.near_partners(compact[u], compact[v], scores, len(ids), neighbors)
                    want = reference_near(compact[u], compact[v], scores, len(ids), neighbors)
                    np.testing.assert_array_equal(got, want)

    def test_a_hub_with_thousands_of_edges_keeps_only_its_best(self):
        count = 3000
        u = np.zeros(count - 1, dtype=np.int64)
        v = np.arange(1, count)
        scores = np.random.default_rng(1).random(count - 1)
        ids, compact = pairs.connected_nodes(u, v, count)
        table = pairs.near_partners(compact[u], compact[v], scores, count, 10)
        np.testing.assert_array_equal(table[0], 1 + np.argsort(-scores, kind="stable")[:10])
        for node in range(1, count):
            self.assertEqual(list(table[node]), [0] + [-1] * 9)

    def test_a_pair_stored_twice_counts_once_at_its_best_edge(self):
        table = pairs.near_partners([0, 0, 0, 1], [1, 2, 1, 2], [0.5, 0.4, 0.9, 0.1], 3, 3)
        np.testing.assert_array_equal(table, [[1, 2, -1], [0, 2, -1], [0, 1, -1]])

    def test_equal_scores_go_to_the_earlier_edge(self):
        table = pairs.near_partners([0, 0, 0, 0], [3, 1, 2, 4], [1.0, 1.0, 1.0, 1.0], 5, 2)
        np.testing.assert_array_equal(table[0], [3, 1])

    def test_no_edges_no_nodes(self):
        self.assertEqual(pairs.near_partners([], [], [], 0, 5).shape, (0, 5))

    def test_the_threads_do_not_change_the_partners(self):
        rng = np.random.default_rng(2)
        u, v, scores = random_edges(rng, 500, 4000, ties=True)
        ids, compact = pairs.connected_nodes(u, v, 500)
        results = []
        for threads in (1, Numba_Threads.default_thread_count()):
            with Numba_Threads.limited_threads(threads):
                results.append(pairs.near_partners(compact[u], compact[v], scores, len(ids), 10))
        np.testing.assert_array_equal(results[0], results[1])


class FurtherPairsTests(unittest.TestCase):
    def check(self, near, count, neighbors, seed=42):
        table = pairs.further_partners(near, count, neighbors, seed)
        self.assertEqual(table.shape, (count, pairs.further_count(neighbors)))
        for node in range(count):
            row = [int(x) for x in table[node] if x >= 0]
            held = {int(x) for x in near[node] if x >= 0}
            self.assertEqual(len(row), len(set(row)), node)
            self.assertNotIn(node, row)
            self.assertFalse(held & set(row), node)
            wanted = min(pairs.further_count(neighbors), count - 1 - len(held))
            self.assertEqual(len(row), wanted, node)
            self.assertEqual(list(table[node][len(row):]), [-1] * (table.shape[1] - len(row)))
        return table

    def test_partners_avoid_the_node_its_neighbours_and_repeats(self):
        rng = np.random.default_rng(3)
        for count, neighbors in ((500, 10), (60, 10), (25, 10), (12, 5), (3, 10), (2, 4)):
            with self.subTest(count=count, neighbors=neighbors):
                u, v, scores = random_edges(rng, count, count * 4)
                ids, compact = pairs.connected_nodes(u, v, count)
                m = len(ids)
                near = pairs.near_partners(compact[u], compact[v], scores, m, neighbors)
                self.check(near, m, neighbors)

    def test_a_complete_graph_leaves_nothing_further(self):
        count = 6
        near = np.array([[j for j in range(count) if j != i] for i in range(count)], dtype=np.int32)
        table = pairs.further_partners(near, count, 5, 42)
        self.assertTrue((table < 0).all())

    def test_the_seed_decides_and_the_threads_do_not(self):
        rng = np.random.default_rng(4)
        u, v, scores = random_edges(rng, 400, 1600)
        ids, compact = pairs.connected_nodes(u, v, 400)
        near = pairs.near_partners(compact[u], compact[v], scores, len(ids), 10)
        base = pairs.further_partners(near, len(ids), 10, 42)
        np.testing.assert_array_equal(base, pairs.further_partners(near, len(ids), 10, 42))
        self.assertFalse(np.array_equal(base, pairs.further_partners(near, len(ids), 10, 43)))
        with Numba_Threads.limited_threads(1):
            np.testing.assert_array_equal(base, pairs.further_partners(near, len(ids), 10, 42))

    def test_the_partners_are_spread_over_every_node(self):
        count = 300
        near = np.full((count, 4), -1, dtype=np.int32)
        table = pairs.further_partners(near, count, 5, 42)
        chosen = np.bincount(table[table >= 0], minlength=count)
        self.assertTrue(chosen.min() > 0 and chosen.max() < 40, (chosen.min(), chosen.max()))


class CandidateTests(unittest.TestCase):
    def test_candidates_are_distinct_other_nodes(self):
        for count, neighbors in ((100, 10), (31, 10), (13, 4)):
            cand = pairs.mid_near_candidates(count, neighbors, 42)
            slots = pairs.effective_mid_near_count(count, neighbors)
            self.assertEqual(cand.shape, (count, slots, 6))
            for node in range(count):
                row = cand[node].ravel()
                self.assertEqual(len(set(row.tolist())), len(row))
                self.assertNotIn(node, row)
                self.assertTrue(0 <= row.min() and row.max() < count)

    def test_too_few_nodes_give_no_slots(self):
        for count in (0, 1, 6):
            self.assertEqual(pairs.mid_near_candidates(count, 10, 42).shape, (count, 0, 6))

    def test_seed_and_threads(self):
        a = pairs.mid_near_candidates(200, 10, 42)
        np.testing.assert_array_equal(a, pairs.mid_near_candidates(200, 10, 42))
        self.assertFalse(np.array_equal(a, pairs.mid_near_candidates(200, 10, 7)))
        with Numba_Threads.limited_threads(1):
            np.testing.assert_array_equal(a, pairs.mid_near_candidates(200, 10, 42))


class PickTests(unittest.TestCase):
    def test_the_second_best_wins_and_ties_go_to_the_earlier(self):
        candidates = np.arange(24, dtype=np.int32).reshape(2, 2, 6)
        scores = np.array([
            [[1, 5, 3, 2, 4, 0], [9, 9, 9, 9, 9, 9]],
            [[-np.inf] * 6, [3, 5, 5, 1, 0, 2]],
        ], dtype=np.float64)
        picked = pairs.pick_mid_near(candidates, scores)
        # slot (0,0): scores 5 then 4 -> candidate 4; (0,1): all equal -> the second; (1,0): all missing -> the second;
        # (1,1): 5 at 1 and 5 at 2 -> the second of them, 2.
        np.testing.assert_array_equal(picked, [[4, 7], [13, 20]])

    def test_matches_a_stable_sort(self):
        rng = np.random.default_rng(5)
        candidates = rng.integers(0, 1000, (50, 5, 6)).astype(np.int32)
        scores = rng.integers(0, 4, (50, 5, 6)).astype(np.float64)
        scores[rng.random(scores.shape) < 0.3] = -np.inf
        order = np.argsort(-scores, axis=2, kind="stable")
        want = np.take_along_axis(candidates, order[:, :, 1:2], axis=2)[:, :, 0]
        np.testing.assert_array_equal(pairs.pick_mid_near(candidates, scores), want)


def brute_force_best(u, v, scores):
    best = {}
    for a, b, score in zip(u, v, scores):
        if score != score:
            continue
        key = (min(a, b), max(a, b))
        best[key] = max(best.get(key, -np.inf), float(score))
    return best


class PairLookupTests(unittest.TestCase):
    def run_lookup(self, source, first, second, node_count, index_map=None, *, chunk=None, workers=None, **options):
        lookup = pairs.PairLookup(first, second, node_count, **options)
        with mock.patch.object(scan, "CHUNK_EDGES", chunk or scan.CHUNK_EDGES):
            pairs.stream_pair_scores(source, index_map, lookup, workers=workers)
        return lookup.scores_of(first, second)

    def stored(self, rng, count, total, **kind):
        u, v, scores = random_edges(rng, count, total, **kind)
        return u, v, scores

    def check(self, u, v, scores, count, sample_size, rng, **options):
        best = brute_force_best(u, v, scores)
        first = rng.integers(0, count, sample_size)
        second = rng.integers(0, count, sample_size)
        got = self.run_lookup(ArraySource(u.astype(np.uint16), v.astype(np.uint16), scores), first, second, count,
                              **options)
        want = np.array([best.get((min(a, b), max(a, b)), -np.inf) for a, b in zip(first, second)])
        np.testing.assert_array_equal(got, want)
        return got

    def test_scores_of_sampled_pairs_from_pairs_stored_in_no_order(self):
        rng = np.random.default_rng(6)
        u, v, scores = self.stored(rng, 300, 5000)
        got = self.check(u, v, scores, 300, 2000, rng, chunk=317)
        self.assertTrue(np.isfinite(got).any() and np.isinf(got).any())

    def test_a_pair_that_is_not_stored_is_the_farthest(self):
        got = self.run_lookup(ArraySource(np.array([0], dtype=np.uint16), np.array([1], dtype=np.uint16),
                                          np.array([5.0])), [0, 1, 2], [1, 0, 3], 4)
        np.testing.assert_array_equal(got, [5.0, 5.0, -np.inf])

    def test_a_pair_stored_twice_keeps_its_best_score_and_nan_counts_as_missing(self):
        source = ArraySource(np.array([0, 1, 0, 2], dtype=np.uint16), np.array([1, 0, 1, 3], dtype=np.uint16),
                             np.array([2.0, 7.0, 4.0, np.nan]))
        for chunk in (1, 2, 100):
            got = self.run_lookup(source, [0, 1, 2], [1, 0, 3], 4, chunk=chunk)
            np.testing.assert_array_equal(got, [7.0, 7.0, -np.inf])

    def test_the_answer_does_not_depend_on_chunks_workers_or_the_hash_table(self):
        rng = np.random.default_rng(7)
        u, v, scores = self.stored(rng, 400, 6000, ties=True, nan=True)
        # A table as small as the sampled pairs allow, and a one-bit-wide filter, make nearly every
        # probe collide: the result is the exact one all the same.
        for options in ({}, {"table_bits": 12, "filter_bits": 3}, {"table_bits": 13, "filter_bits": 16}):
            for chunk, workers in ((6000, 1), (257, 1), (257, 3), (1, 2)):
                with self.subTest(options=options, chunk=chunk, workers=workers):
                    self.check(u, v, scores, 400, 1500, np.random.default_rng(8), chunk=chunk, workers=workers,
                               **options)

    def test_a_full_table_is_refused(self):
        with self.assertRaises(ValueError):
            pairs.PairLookup(np.arange(10), np.arange(10, 20), 100, table_bits=3)

    def test_stored_node_columns_of_every_integer_width_probe_alike(self):
        rng = np.random.default_rng(9)
        u, v, scores = self.stored(rng, 200, 3000)
        first = rng.integers(0, 200, 400)
        second = rng.integers(0, 200, 400)
        results = []
        for dtype in (np.uint16, np.uint32, np.int32, np.int64):
            results.append(self.run_lookup(ArraySource(u.astype(dtype), v.astype(dtype), scores), first, second, 200))
        for other in results[1:]:
            np.testing.assert_array_equal(results[0], other)

    def test_an_index_map_takes_stored_numbers_to_layout_numbers(self):
        # Stored nodes 10..14 are layout nodes 0..4; the pair file also holds pairs outside the layout.
        index_map = np.zeros(20, dtype=np.int32)
        index_map[10:15] = np.arange(5)
        source = ArraySource(np.array([10, 11, 12], dtype=np.uint16), np.array([11, 14, 13], dtype=np.uint16),
                             np.array([1.0, 2.0, 3.0]))
        got = self.run_lookup(source, [0, 1, 2, 3], [1, 4, 3, 0], 5, index_map)
        np.testing.assert_array_equal(got, [1.0, 2.0, 3.0, -np.inf])

    def test_the_memory_does_not_grow_with_the_square_of_the_nodes(self):
        lookup = pairs.PairLookup(np.arange(0, 1000), np.arange(1000, 2000), 2_000_000_000)
        self.assertLess(lookup.table_keys.nbytes + lookup.bit_array.nbytes, 10_000_000)
        with self.assertRaises(ValueError):
            pairs.PairLookup([0], [1], 3_000_000_001)


class MidNearPairsTests(unittest.TestCase):
    def complete_network(self, count, score):
        u, v = np.triu_indices(count, k=1)
        rng = np.random.default_rng(10)
        order = rng.permutation(len(u))
        u, v = u[order], v[order]
        return ArraySource(u.astype(np.uint16), v.astype(np.uint16), score(u, v).astype(np.float32))

    def test_each_partner_is_the_second_best_of_its_six_sampled_nodes(self):
        count = 120
        source = self.complete_network(count, lambda u, v: -np.abs(u - v).astype(np.float64))
        ids = np.arange(count)
        got = pairs.mid_near_pairs(source, None, count, ids, 10, 42)
        self.assertEqual(got.shape, (count * 5, 2))
        np.testing.assert_array_equal(got[:, 0], np.repeat(ids, 5))
        cand = pairs.mid_near_candidates(count, 10, 42)
        for node in range(count):
            for slot in range(5):
                sampled = cand[node, slot]
                nearness = sorted(range(6), key=lambda i: (abs(int(sampled[i]) - node), i))
                self.assertEqual(got[node * 5 + slot, 1], sampled[nearness[1]])

    def test_scores_decide_so_a_nearer_node_is_chosen_more_often_than_chance(self):
        count = 400
        source = self.complete_network(count, lambda u, v: -np.abs(u - v).astype(np.float64))
        ids = np.arange(count)
        scored = pairs.mid_near_pairs(source, None, count, ids, 10, 42)
        blind = pairs.random_mid_near_pairs(ids, 10, 42)
        near_scored = np.abs(scored[:, 0].astype(int) - scored[:, 1]).mean()
        near_blind = np.abs(blind[:, 0].astype(int) - blind[:, 1]).mean()
        self.assertLess(near_scored, 0.6 * near_blind)

    def test_without_stored_pairs_the_partners_are_the_blind_ones(self):
        count = 50
        empty = ArraySource(np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float32))
        got = pairs.mid_near_pairs(empty, None, count, np.arange(count), 10, 42)
        np.testing.assert_array_equal(got, pairs.random_mid_near_pairs(np.arange(count), 10, 42))

    def test_the_chunks_the_workers_and_the_seed(self):
        count = 150
        source = self.complete_network(count, lambda u, v: ((u * 7 + v * 13) % 11).astype(np.float64))
        ids = np.arange(count)
        base = pairs.mid_near_pairs(source, None, count, ids, 10, 42, workers=1)
        for chunk, workers in ((97, 1), (97, 4), (1000, 3)):
            with mock.patch.object(scan, "CHUNK_EDGES", chunk):
                np.testing.assert_array_equal(
                    base, pairs.mid_near_pairs(source, None, count, ids, 10, 42, workers=workers))
        self.assertFalse(np.array_equal(base, pairs.mid_near_pairs(source, None, count, ids, 10, 7, workers=1)))
        np.testing.assert_array_equal(
            base, pairs.mid_near_pairs(source, None, count, ids, 10, 42, workers=1,
                                       lookup_options={"table_bits": 14, "filter_bits": 4}))

    def test_only_connected_nodes_take_part(self):
        # Layout nodes 0..99 of which 10..19 are isolated; the pair file has pairs among all of them.
        count = 100
        source = self.complete_network(count, lambda u, v: -np.abs(u - v).astype(np.float64))
        ids = np.array([i for i in range(count) if not 10 <= i < 20])
        got = pairs.mid_near_pairs(source, None, count, ids, 10, 42)
        used = set(got.ravel().tolist())
        self.assertFalse(used & set(range(10, 20)))
        self.assertEqual(set(got[:, 0].tolist()), set(ids.tolist()))

    def test_a_tiny_network_has_no_mid_near_pairs(self):
        source = self.complete_network(5, lambda u, v: u.astype(np.float64))
        got = pairs.mid_near_pairs(source, None, 5, np.arange(5), 10, 42)
        self.assertEqual(got.shape, (0, 2))


class IncidenceTests(unittest.TestCase):
    def test_every_pair_is_in_both_its_nodes_lists_in_row_order(self):
        first = np.array([0, 0, 2, 1], dtype=np.int32)
        second = np.array([1, 2, 1, 3], dtype=np.int32)
        pointer, partners = pairs.incidence_lists(first, second, 4)
        lists = [partners[pointer[i]:pointer[i + 1]].tolist() for i in range(4)]
        self.assertEqual(lists, [[1, 2], [0, 2, 3], [0, 1], [1]])

    def test_rows_of_partners_leave_out_the_padding(self):
        owner, other = pairs.rows_of_partners(np.array([[1, 2, -1], [-1, -1, -1], [0, -1, -1]], dtype=np.int32))
        self.assertEqual(list(zip(owner.tolist(), other.tolist())), [(0, 1), (0, 2), (2, 0)])

    def test_empty(self):
        pointer, partners = pairs.incidence_lists([], [], 3)
        self.assertEqual(pointer.tolist(), [0, 0, 0, 0])
        self.assertEqual(len(partners), 0)


class StartTests(unittest.TestCase):
    def test_the_start_is_a_normal_cloud_of_the_given_scale(self):
        start = pairs.normal_start(20000, 2, 42, 1e-4)
        self.assertEqual(start.dtype, np.float32)
        self.assertLess(abs(float(start.mean())), 3e-6)
        self.assertLess(abs(float(start.std()) - 1e-4), 3e-6)
        np.testing.assert_array_equal(start, pairs.normal_start(20000, 2, 42, 1e-4))
        self.assertFalse(np.array_equal(start, pairs.normal_start(20000, 2, 43, 1e-4)))
        self.assertFalse(np.array_equal(start[:, 0], start[:, 1]))
        # The first nodes do not move when more are asked for.
        np.testing.assert_array_equal(start[:100], pairs.normal_start(100, 2, 42, 1e-4))


if __name__ == "__main__":
    unittest.main()
