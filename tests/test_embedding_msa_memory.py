"""Embedding_MSA builds the same guide tree while holding far less memory.

The reference functions restate the code before the memory changes (commit
17a1030): a per-edge Python filter, float32 bootstrap replicates and one
cophenetic array per tree. The current code must reproduce them bit for bit.
The guide-tree kernels (score normalization, sparse cophenetic distances and
the consensus finalization) are also checked against hand-computed values and
SciPy.
"""

import copy
import gc
import io
import os
import pathlib
import sys
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA
from tests.msa_fixtures import (
    UPGMA,
    ArrayAssertions,
    GuideTreeBuilt,
    InProcessPool,
    NetworkFixture,
    builder_settings,
)


NEIGHBOR_JOINING = "Neighbor-joining (Slow)"
MODE_INTS = {
    "alignment_length": 0,
    "shorter_sequence": 1,
    "longer_sequence": 2,
    "average_sequence": 3,
}


def reference_edges(fixture, normalization_mode="alignment_length"):
    """Return the edges and distances the per-edge filter of 17a1030 built."""
    common = set(fixture.net_headers) & set(fixture.emb_headers)
    valid_headers = [header for header in fixture.net_headers if header in common]
    header_to_new_idx = {header: index for index, header in enumerate(valid_headers)}
    net_old_to_new = {}
    for index, header in enumerate(fixture.net_headers):
        if header in header_to_new_idx:
            net_old_to_new[index] = header_to_new_idx[header]

    processed_edges = []
    for k in range(len(fixture.arr_i)):
        u_old, v_old = int(fixture.arr_i[k]), int(fixture.arr_j[k])
        if u_old in net_old_to_new and v_old in net_old_to_new:
            processed_edges.append((
                net_old_to_new[u_old],
                net_old_to_new[v_old],
                fixture.score[k],
                fixture.length[k],
            ))
    edge_count = len(processed_edges)
    edge_i = np.zeros(edge_count, dtype=np.int32)
    edge_j = np.zeros(edge_count, dtype=np.int32)
    raw_scores = np.zeros(edge_count, dtype=np.float32)
    align_lens = np.zeros(edge_count, dtype=np.float32)
    for k, edge in enumerate(processed_edges):
        edge_i[k] = edge[0]
        edge_j[k] = edge[1]
        raw_scores[k] = edge[2]
        align_lens[k] = edge[3]

    sequence_by_header = dict(zip(fixture.emb_headers, fixture.emb_sequences))
    seq_lens = np.asarray(
        [len(sequence_by_header[header]) for header in valid_headers],
        dtype=np.int32,
    )
    norm_scores, max_norm_score = Embedding_MSA.calculate_normalized_scores_kernel(
        edge_i,
        edge_j,
        raw_scores,
        align_lens,
        seq_lens,
        fixture.blast,
        MODE_INTS[normalization_mode],
    )
    edge_dists = np.maximum(0.0, max_norm_score - norm_scores).astype(np.float32)
    return edge_i, edge_j, edge_dists, max_norm_score + 0.1, len(valid_headers)


def reference_baseline(edge_i, edge_j, edge_dists, max_distance, num_seqs):
    """Return the complete network's condensed distance baseline."""
    baseline = np.full(num_seqs * (num_seqs - 1) // 2, max_distance, dtype=np.float32)
    Embedding_MSA.populate_condensed_matrix(baseline, num_seqs, edge_i, edge_j, edge_dists)
    np.clip(baseline, 0.0, max_distance, out=baseline)
    return baseline


def reference_replicate(seed, baseline, max_dist, noise_scale):
    """Return the float32 bootstrap replicate that 17a1030 handed to SciPy."""
    rng = np.random.default_rng(seed)
    sigma = float(noise_scale) * float(max_dist)
    replicate = np.empty(baseline.size, dtype=np.float32)
    if sigma == 0.0:
        replicate[:] = baseline
    else:
        for start in range(0, baseline.size, 1_000_000):
            end = min(start + 1_000_000, baseline.size)
            noise = rng.normal(0.0, sigma, size=end - start).astype(np.float32)
            replicate[start:end] = baseline[start:end] + noise
    np.clip(replicate, 0.0, max_dist, out=replicate)
    return replicate


def reference_consensus_tree(baseline, max_dist, noise_scale, num_trees, random_seed):
    """Return the master tree built from one float32 cophenetic array per tree."""
    seeds = np.random.default_rng(random_seed).integers(0, int(1e9), size=num_trees)
    consensus = np.zeros_like(baseline)
    for seed in seeds:
        tree = sch.linkage(
            reference_replicate(seed, baseline, max_dist, noise_scale),
            method="average",
        )
        consensus += sch.cophenet(tree).astype(np.float32)
    consensus /= num_trees
    return sch.linkage(consensus, method="average")


class HeldMemoryPool(InProcessPool):
    """Records the traced memory still held when bootstrapping starts."""

    held_bytes = None

    def imap_unordered(self, func, iterable):
        HeldMemoryPool.held_bytes = tracemalloc.get_traced_memory()[0]
        raise GuideTreeBuilt


def run_until_guide_tree(fixture, *, record_edges=True, pool=InProcessPool, **settings):
    """Run run_msa_builder on the fixture and report what the tree stage saw.

    Recording copies the edge arrays, so memory measurements turn it off.
    """
    values = builder_settings(fixture, **settings)
    seen = {"populate": [], "linkage_dtypes": []}
    real_populate = Embedding_MSA.populate_condensed_matrix
    real_linkage = sch.linkage

    def record_populate(D_condensed, num_seqs, edge_i, edge_j, edge_dists):
        seen["populate"].append(
            (np.array(edge_i), np.array(edge_j), np.array(edge_dists))
        )
        return real_populate(D_condensed, num_seqs, edge_i, edge_j, edge_dists)

    def record_linkage(y, *args, **kwargs):
        seen["linkage_dtypes"].append(np.asarray(y).dtype)
        return real_linkage(y, *args, **kwargs)

    def stop_at_guide_tree(linkage_matrix, *args, **kwargs):
        seen["guide_tree"] = np.array(linkage_matrix)
        raise GuideTreeBuilt

    if record_edges:
        values["populate_condensed_matrix"] = record_populate

    # set_start_method is patched because the builder switches the whole
    # process to spawn, which would leak into later tests on Linux.
    with mock.patch.multiple(
        Embedding_MSA,
        benchmark_msa_devices=stop_at_guide_tree,
        **values,
    ), mock.patch.object(Embedding_MSA.mp, "set_start_method"), mock.patch.object(
        Embedding_MSA.mp, "Pool", pool
    ), mock.patch.object(
        Embedding_MSA.sch, "linkage", record_linkage
    ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            Embedding_MSA.run_msa_builder()
        except GuideTreeBuilt:
            pass
    # Release the builder's HDF5 handles before the folder is removed.
    gc.collect()
    return seen


class EdgeIndexMappingTests(unittest.TestCase):
    def test_unknown_and_out_of_range_endpoints_map_to_minus_one(self):
        table = np.asarray([2, -1, 0, 1], dtype=np.int32)
        cases = {
            "uint16 in range": (np.asarray([0, 1, 2, 3, 0], np.uint16), [2, -1, 0, 1, 2]),
            "uint16 too large": (np.asarray([3, 4, 65535], np.uint16), [1, -1, -1]),
            "int32 negative": (np.asarray([-1, 2, -4], np.int32), [-1, 0, -1]),
            "uint32": (np.asarray([2, 7], np.uint32), [0, -1]),
            "empty": (np.asarray([], np.uint16), []),
        }
        for name, (indices, expected) in cases.items():
            with self.subTest(name):
                mapped = Embedding_MSA.map_network_indices(indices, table)
                self.assertEqual(mapped.dtype, np.int32)
                self.assertEqual(mapped.tolist(), expected)


class CopheneticAccumulationTests(ArrayAssertions, unittest.TestCase):
    def test_in_place_sum_matches_per_tree_cophenetic_arrays(self):
        rng = np.random.default_rng(3)
        num_seqs = 40
        expected = np.zeros(num_seqs * (num_seqs - 1) // 2, dtype=np.float32)
        actual = np.zeros_like(expected)
        for _ in range(4):
            tree = sch.linkage(rng.random(expected.size), method="average")
            expected += sch.cophenet(tree).astype(np.float32)
            Embedding_MSA.accumulate_full_cophenetic(tree, num_seqs, actual)
        self.assertSameArray(actual, expected)

    def test_single_sequence_adds_nothing(self):
        accumulator = np.zeros(0, dtype=np.float32)
        Embedding_MSA.accumulate_full_cophenetic(np.zeros((0, 4)), 1, accumulator)
        self.assertEqual(accumulator.size, 0)


class SparseCopheneticTests(ArrayAssertions, unittest.TestCase):
    """Sparse networks average cophenetic distances over their own edges only."""

    def test_listed_pairs_match_scipy_cophenetic_distances(self):
        rng = np.random.default_rng(5)
        num_seqs = 40
        tree = sch.linkage(rng.random(num_seqs * (num_seqs - 1) // 2), method="average")
        # Every ordered pair, so half of them list the larger index first.
        edge_i, edge_j = np.nonzero(~np.eye(num_seqs, dtype=bool))
        edge_i = edge_i.astype(np.int32)
        edge_j = edge_j.astype(np.int32)
        expected = squareform(sch.cophenet(tree))[edge_i, edge_j].astype(np.float32)

        actual = Embedding_MSA.compute_sparse_cophenetic(tree, num_seqs, edge_i, edge_j)

        self.assertSameArray(actual, expected)
        self.assertTrue(np.any(edge_i > edge_j))

    def test_partial_consensus_replaces_only_the_listed_pairs(self):
        baseline = np.arange(1, 11, dtype=np.float32)  # 5 sequences, 10 pairs
        # Pairs (0, 1), (4, 3) and (1, 3) are condensed entries 0, 9 and 5.
        edge_i = np.asarray([0, 4, 1], dtype=np.int32)
        edge_j = np.asarray([1, 3, 3], dtype=np.int32)
        accumulated = np.asarray([8.0, 12.0, 2.0], dtype=np.float32)
        expected = baseline.copy()
        expected[[0, 9, 5]] = [2.0, 3.0, 0.5]

        result = Embedding_MSA.finalize_cophenetic_consensus(
            baseline, 5, edge_i, edge_j, accumulated, 4, False
        )

        self.assertIs(result, baseline)
        self.assertSameArray(result, expected)

    def test_full_consensus_divides_every_pair(self):
        accumulated = np.arange(1, 11, dtype=np.float32)
        expected = accumulated / np.float32(4)

        result = Embedding_MSA.finalize_cophenetic_consensus(
            accumulated, 5, None, None, None, 4, True
        )

        self.assertIs(result, accumulated)
        self.assertSameArray(result, expected)

    def test_zero_trees_are_rejected(self):
        for full_consensus in (False, True):
            with self.subTest(full_consensus=full_consensus):
                with self.assertRaisesRegex(ValueError, "NUM_TREES must be greater than zero"):
                    Embedding_MSA.finalize_cophenetic_consensus(
                        np.ones(3, dtype=np.float32),
                        3,
                        np.asarray([0], dtype=np.int32),
                        np.asarray([1], dtype=np.int32),
                        np.ones(1, dtype=np.float32),
                        0,
                        full_consensus,
                    )


class NormalizedScoreKernelTests(unittest.TestCase):
    """One edge between sequences of 4 and 8 residues, raw score 12, length 0."""

    def normalize(self, mode_int, is_evalue=False):
        return Embedding_MSA.calculate_normalized_scores_kernel(
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([12.0], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([4, 8], dtype=np.int32),
            is_evalue,
            mode_int,
        )

    def test_each_mode_divides_by_its_own_length(self):
        # A zero alignment length counts as 1.
        for mode, expected in (
            ("alignment_length", 12.0),
            ("shorter_sequence", 3.0),
            ("longer_sequence", 1.5),
            ("average_sequence", 2.0),
        ):
            with self.subTest(mode=mode):
                norm_scores, max_norm_score = self.normalize(MODE_INTS[mode])
                self.assertEqual(norm_scores.dtype, np.float32)
                self.assertEqual(norm_scores.tolist(), [expected])
                self.assertEqual(max_norm_score, expected)

    def test_evalue_scores_are_not_normalized(self):
        norm_scores, max_norm_score = self.normalize(
            MODE_INTS["shorter_sequence"], is_evalue=True
        )
        self.assertEqual(norm_scores.tolist(), [12.0])
        self.assertEqual(max_norm_score, 12.0)


class BootstrapWorkerTests(ArrayAssertions, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def write_baseline(self, num_seqs, seed):
        rng = np.random.default_rng(seed)
        baseline = rng.random(num_seqs * (num_seqs - 1) // 2, dtype=np.float32) * 3.0
        # A ceiling below the largest distance makes the upper clip matter
        # even without noise.
        max_dist = float(baseline.max()) - 0.05
        path = os.path.join(self.directory, f"baseline_{num_seqs}_{seed}.dat")
        baseline.tofile(path)
        return baseline, max_dist, path

    def run_worker(self, seed, num_seqs, path, max_dist, noise_scale, tree_method):
        dtypes = []
        real_linkage = sch.linkage

        def record_linkage(y, *args, **kwargs):
            dtypes.append(np.asarray(y).dtype)
            return real_linkage(y, *args, **kwargs)

        with mock.patch.object(Embedding_MSA.sch, "linkage", record_linkage):
            tree = Embedding_MSA.compute_single_tree_worker(
                seed, num_seqs, path, max_dist, noise_scale, tree_method
            )
        gc.collect()
        return tree, dtypes

    def test_upgma_replicate_matches_float32_reference(self):
        # 1,123,250 pairs span two noise chunks.
        for num_seqs, noise_scale, seed in ((1500, 0.02, 11), (300, 0.0, 12), (300, 0.5, 13)):
            with self.subTest(num_seqs=num_seqs, noise_scale=noise_scale):
                baseline, max_dist, path = self.write_baseline(num_seqs, seed)
                expected = sch.linkage(
                    reference_replicate(seed, baseline, max_dist, noise_scale),
                    method="average",
                )
                tree, dtypes = self.run_worker(
                    seed, num_seqs, path, max_dist, noise_scale, UPGMA
                )
                self.assertSameArray(tree, expected)
                # SciPy receives float64, so it makes no conversion copy.
                self.assertEqual(dtypes, [np.dtype(np.float64)])

    def test_neighbor_joining_replicate_matches_float32_reference(self):
        baseline, max_dist, path = self.write_baseline(120, 14)
        expected = Embedding_MSA.neighbor_joining_condensed(
            reference_replicate(14, baseline, max_dist, 0.02), 120
        )
        tree, dtypes = self.run_worker(14, 120, path, max_dist, 0.02, NEIGHBOR_JOINING)
        self.assertSameArray(tree, expected)
        self.assertEqual(dtypes, [])


class GuideTreeEquivalenceTests(ArrayAssertions, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def assert_reference_edges(self, seen, expected_edges):
        edge_i, edge_j, edge_dists = seen["populate"][0]
        self.assertSameArray(edge_i, expected_edges[0])
        self.assertSameArray(edge_j, expected_edges[1])
        self.assertSameArray(edge_dists, expected_edges[2])

    def test_complete_network_matches_reference(self):
        fixture = NetworkFixture(self.directory)
        # Sequence-length denominators make the largest normalized score a
        # float64 that float32 cannot hold, so inverting it with float64
        # arithmetic would change distances.
        for normalization_mode in ("alignment_length", "average_sequence"):
            with self.subTest(normalization_mode=normalization_mode):
                seen = run_until_guide_tree(
                    fixture, NORMALIZATION_MODE=normalization_mode
                )

                expected_edges = reference_edges(fixture, normalization_mode)
                self.assert_reference_edges(seen, expected_edges)
                baseline = reference_baseline(*expected_edges)
                self.assertSameArray(
                    seen["guide_tree"], sch.linkage(baseline, method="average")
                )
                self.assertEqual(seen["linkage_dtypes"], [np.dtype(np.float64)])

    def test_complete_network_bootstrap_consensus_matches_reference(self):
        fixture = NetworkFixture(self.directory, seed=1)
        seen = run_until_guide_tree(fixture, BOOTSTRAP_TREE=True, NUM_TREES=3)

        expected_edges = reference_edges(fixture)
        self.assert_reference_edges(seen, expected_edges)
        baseline = reference_baseline(*expected_edges)
        expected_tree = reference_consensus_tree(
            baseline, expected_edges[3], 0.02, num_trees=3, random_seed=42
        )
        self.assertSameArray(seen["guide_tree"], expected_tree)
        # Three replicates and the master tree all reach SciPy as float64.
        self.assertEqual(seen["linkage_dtypes"], [np.dtype(np.float64)] * 4)

    def test_sparse_network_edges_match_reference(self):
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        expected_edges = reference_edges(fixture, "average_sequence")
        for include_imputed in (False, True):
            with self.subTest(include_imputed=include_imputed):
                seen = run_until_guide_tree(
                    fixture,
                    NORMALIZATION_MODE="average_sequence",
                    BOOTSTRAP_TREE=True,
                    NUM_TREES=2,
                    INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS=include_imputed,
                )
                self.assert_reference_edges(seen, expected_edges)
                self.assertEqual(seen["guide_tree"].shape, (expected_edges[4] - 1, 4))
                self.assertEqual(seen["linkage_dtypes"], [np.dtype(np.float64)] * 3)

    def test_blast_network_matches_reference(self):
        fixture = NetworkFixture(self.directory, blast=True, seed=3)
        seen = run_until_guide_tree(fixture)

        expected_edges = reference_edges(fixture)
        self.assert_reference_edges(seen, expected_edges)
        baseline = reference_baseline(*expected_edges)
        self.assertSameArray(seen["guide_tree"], sch.linkage(baseline, method="average"))

    def test_local_score_reads_the_local_datasets(self):
        fixture = NetworkFixture(self.directory, seed=5)
        seen = run_until_guide_tree(
            fixture, ALIGNMENT_SCORE="local", NORMALIZATION_MODE="shorter_sequence"
        )

        # The fixture stores half of every global score as the local score.
        local = copy.copy(fixture)
        local.score = fixture.score / 2
        expected_edges = reference_edges(local, "shorter_sequence")
        self.assert_reference_edges(seen, expected_edges)
        global_dists = reference_edges(fixture, "shorter_sequence")[2]
        self.assertNotEqual(global_dists.tobytes(), expected_edges[2].tobytes())


class MemoryBudgetTests(unittest.TestCase):
    # Bytes per edge for a complete network with uint16 indices, a float32
    # score and a uint16 length, every sequence kept. Per-sequence headers,
    # sets and dicts add about 1 byte per edge at this size.
    #
    # The peak is the score normalization: int32 indices (4 + 4), the float32
    # scores and lengths (4 + 4) and the normalized scores (4) make 20 bytes.
    # One Python tuple per edge took about 190.
    PEAK_BUDGET = 22.0
    # When bootstrapping starts, the main process holds only the float32
    # consensus accumulator (4 bytes per pair). Keeping the normalized scores
    # would add 4, and keeping the raw scores and lengths 8 more.
    HELD_BUDGET = 6.0

    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.fixture = NetworkFixture(
            temporary.name,
            sequences=1415,
            missing=(),
            ghosts=0,
            duplicates=0,
            out_of_range=False,
            seed=4,
        )
        cls.edge_count = cls.fixture.arr_i.size
        # This run compiles the numba kernels, which tracemalloc would
        # otherwise count.
        run_until_guide_tree(cls.fixture, record_edges=False)

    def test_edge_filtering_peak_stays_within_budget(self):
        tracemalloc.start()
        try:
            run_until_guide_tree(self.fixture, record_edges=False)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak / self.edge_count, self.PEAK_BUDGET)

    def test_bootstrap_starts_holding_only_the_consensus(self):
        HeldMemoryPool.held_bytes = None
        tracemalloc.start()
        try:
            run_until_guide_tree(
                self.fixture,
                record_edges=False,
                pool=HeldMemoryPool,
                BOOTSTRAP_TREE=True,
            )
        finally:
            tracemalloc.stop()
        self.assertIsNotNone(HeldMemoryPool.held_bytes)
        self.assertLess(HeldMemoryPool.held_bytes / self.edge_count, self.HELD_BUDGET)


if __name__ == "__main__":
    unittest.main()
