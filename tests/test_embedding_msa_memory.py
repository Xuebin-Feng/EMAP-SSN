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

import h5py
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


def array_locals(frame):
    """{name: shape} of the NumPy arrays that a frame's local variables name."""
    return {
        name: value.shape for name, value in frame.f_locals.items() if isinstance(value, np.ndarray)
    }


class HeldMemoryPool(InProcessPool):
    """Records the traced memory still held, and the arrays run_msa_builder still names, when bootstrapping starts."""

    held_bytes = None
    arrays = None

    def imap(self, func, iterable):
        HeldMemoryPool.held_bytes = tracemalloc.get_traced_memory()[0]
        HeldMemoryPool.arrays = array_locals(sys._getframe(1))
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
        result = real_populate(D_condensed, num_seqs, edge_i, edge_j, edge_dists)
        # The complete baseline the replicates start from, before the builder clips it.
        seen["baseline"] = np.array(D_condensed)
        return result

    def record_linkage(y, *args, **kwargs):
        seen["linkage_dtypes"].append(np.asarray(y).dtype)
        # The single tree of a run without perturbed trees is built here, from run_msa_builder.
        seen.setdefault("linkage_arrays", array_locals(sys._getframe(1)))
        if record_edges:
            # Every tree's input; the last is the consensus the master tree is built from.
            seen.setdefault("linkage_inputs", []).append(np.array(y))
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


class StreamedEdgeFilterTests(ArrayAssertions, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)

    def _reference_filter(self, arr_i, arr_j, scores, lens, table):
        ei = Embedding_MSA.map_network_indices(arr_i, table)
        ej = Embedding_MSA.map_network_indices(arr_j, table)
        keep = (ei >= 0) & (ej >= 0) & (ei != ej)
        ei = ei[keep]
        ej = ej[keep]
        raw_scores = scores[keep].astype(np.float32, copy=False)
        if lens is None:
            align_lens = np.ones(len(ei), dtype=np.float32)
        else:
            align_lens = lens[keep].astype(np.float32, copy=False)
        return ei, ej, raw_scores, align_lens

    def test_chunked_filter_matches_whole_array_across_chunk_sizes_and_dtypes(self):
        rng = np.random.default_rng(19)
        table = np.asarray([0, -1, 1, -1, 2, 3, -1, 4], dtype=np.int32)
        for name, idx_dtype, score_dtype, blast in (
            ("uint16_f32_alignment", np.uint16, np.float32, False),
            ("uint32_f64_alignment", np.uint32, np.float64, False),
            ("uint16_f32_blast", np.uint16, np.float32, True),
        ):
            with self.subTest(name):
                # Mix kept edges, dropped endpoints, self-loops, duplicates, and out-of-range indices.
                arr_i = rng.integers(0, len(table) + 3, size=85, dtype=idx_dtype)
                arr_j = rng.integers(0, len(table) + 3, size=85, dtype=idx_dtype)
                arr_i[10], arr_j[10] = 0, 0  # explicit self-loop on a kept node
                arr_i[20], arr_j[20] = 0, 2  # explicit duplicate pair
                arr_i[60], arr_j[60] = 0, 2
                scores = rng.uniform(1.0, 200.0, size=85).astype(score_dtype)
                lens = None if blast else rng.integers(10, 400, size=85, dtype=np.uint16)

                expected = self._reference_filter(arr_i, arr_j, scores, lens, table)
                h5_path = self.directory / f"{name}.h5"
                with h5py.File(h5_path, "w") as hf:
                    hf.create_dataset("i", data=arr_i)
                    hf.create_dataset("j", data=arr_j)
                    hf.create_dataset("score", data=scores)
                    if lens is not None:
                        hf.create_dataset("len", data=lens)

                with h5py.File(h5_path, "r") as hf:
                    ds_len = None if blast else hf["len"]
                    for chunk_size in (1, 7, 16, 85, 200):
                        got = Embedding_MSA.filter_network_edges(
                            hf["i"], hf["j"], hf["score"], ds_len, table, chunk_size=chunk_size
                        )
                        for actual, want in zip(got, expected):
                            self.assertSameArray(actual, want)

    def test_empty_all_dropped_and_all_kept_networks(self):
        table = np.asarray([0, 1, 2], dtype=np.int32)
        # Empty network
        empty = np.empty(0, dtype=np.uint16)
        got = Embedding_MSA.filter_network_edges(
            empty, empty, np.empty(0, dtype=np.float32), empty, table, chunk_size=4
        )
        for arr, dtype in zip(got, (np.int32, np.int32, np.float32, np.float32)):
            self.assertEqual(arr.shape, (0,))
            self.assertEqual(arr.dtype, dtype)

        # All dropped (self-loops + out-of-range)
        arr_i = np.asarray([0, 1, 2, 5, 6], dtype=np.uint16)
        arr_j = np.asarray([0, 1, 2, 0, 1], dtype=np.uint16)
        scores = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        lens = np.asarray([12, 13, 14, 15, 16], dtype=np.uint16)
        got_dropped = Embedding_MSA.filter_network_edges(
            arr_i, arr_j, scores, lens, table, chunk_size=2
        )
        for arr in got_dropped:
            self.assertEqual(arr.size, 0)

        # All kept across multiple chunks (hits the all_valid fast path)
        valid_i = np.asarray([0, 0, 1, 2, 1, 2], dtype=np.uint16)
        valid_j = np.asarray([1, 2, 2, 0, 0, 1], dtype=np.uint16)
        valid_scores = np.asarray([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=np.float32)
        valid_lens = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.uint16)
        for chunk_size in (2, 10):
            got_kept = Embedding_MSA.filter_network_edges(
                valid_i, valid_j, valid_scores, valid_lens, table, chunk_size=chunk_size
            )
            self.assertSameArray(got_kept[0], valid_i.astype(np.int32))
            self.assertSameArray(got_kept[1], valid_j.astype(np.int32))
            self.assertSameArray(got_kept[2], valid_scores)
            self.assertSameArray(got_kept[3], valid_lens.astype(np.float32))

    def test_skips_unneeded_columns_on_empty_chunks(self):
        class SpyArray:
            def __init__(self, data):
                self._data = data
                self.slices = []

            def __len__(self):
                return len(self._data)

            def __getitem__(self, key):
                self.slices.append((key.start, key.stop))
                return self._data[key]

        table = np.asarray([0, 1, -1, -1], dtype=np.int32)
        # 4 chunks of 3 edges:
        # [0:3]  all i outside subset -> skip j, score, len
        # [3:6]  i hits, all j outside subset -> skip score, len
        # [6:9]  i and j hit, but all self-loops -> skip score, len
        # [9:12] real valid edges -> read all 4 columns
        ds_i = SpyArray(np.asarray([2, 3, 2,  0, 1, 0,  0, 1, 0,  0, 1, 2], dtype=np.uint16))
        ds_j = SpyArray(np.asarray([0, 1, 0,  2, 3, 2,  0, 1, 0,  1, 0, 1], dtype=np.uint16))
        ds_s = SpyArray(np.arange(12, dtype=np.float32))
        ds_l = SpyArray(np.arange(10, 22, dtype=np.uint16))

        ei, ej, scores, lens = Embedding_MSA.filter_network_edges(
            ds_i, ds_j, ds_s, ds_l, table, chunk_size=3
        )
        self.assertEqual(ds_i.slices, [(0, 3), (3, 6), (6, 9), (9, 12)])
        self.assertEqual(ds_j.slices, [(3, 6), (6, 9), (9, 12)])
        self.assertEqual(ds_s.slices, [(9, 12)])
        self.assertEqual(ds_l.slices, [(9, 12)])
        self.assertEqual(ei.tolist(), [0, 1])
        self.assertEqual(ej.tolist(), [1, 0])
        self.assertEqual(scores.tolist(), [9.0, 10.0])
        self.assertEqual(lens.tolist(), [19.0, 20.0])

    def test_small_subset_of_large_network_uses_bounded_memory(self):
        # 2,000,000 edges with uint32 indices (4 + 4 + 4 + 2 = 14 B/edge = 28 MB raw;
        # eager load + int32 mapping would peak at ~48 MB). Only 60 edges match.
        total_edges = 2_000_000
        h5_path = self.directory / "large_subset_net.h5"
        with h5py.File(h5_path, "w") as hf:
            ds_i = hf.create_dataset("i", shape=(total_edges,), dtype=np.uint32)
            ds_j = hf.create_dataset("j", shape=(total_edges,), dtype=np.uint32)
            ds_s = hf.create_dataset("g_score", shape=(total_edges,), dtype=np.float32)
            ds_l = hf.create_dataset("g_len", shape=(total_edges,), dtype=np.uint16)
            fill_i = np.full(200_000, 999, dtype=np.uint32)
            fill_j = np.full(200_000, 998, dtype=np.uint32)
            fill_s = np.full(200_000, 5.0, dtype=np.float32)
            fill_l = np.full(200_000, 50, dtype=np.uint16)
            for start in range(0, total_edges, 200_000):
                stop = start + 200_000
                ds_i[start:stop] = fill_i
                ds_j[start:stop] = fill_j
                ds_s[start:stop] = fill_s
                ds_l[start:stop] = fill_l
            # Sprinkle 60 matching edges across the file.
            hit_positions = np.linspace(100, total_edges - 100, 60, dtype=np.int64)
            ds_i[hit_positions] = np.arange(60, dtype=np.uint32) % 10
            ds_j[hit_positions] = (np.arange(60, dtype=np.uint32) % 10 + 1) % 10
            ds_s[hit_positions] = np.arange(60, dtype=np.float32) + 100.0
            ds_l[hit_positions] = np.arange(60, dtype=np.uint16) + 20

        table = np.full(1000, -1, dtype=np.int32)
        table[:10] = np.arange(10, dtype=np.int32)

        gc.collect()
        with h5py.File(h5_path, "r") as hf:
            tracemalloc.start()
            try:
                ei, ej, scores, lens = Embedding_MSA.filter_network_edges(
                    hf["i"], hf["j"], hf["g_score"], hf["g_len"], table, chunk_size=100_000
                )
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        self.assertEqual(len(ei), 60)
        self.assertEqual(scores[0], 100.0)
        self.assertEqual(scores[-1], 159.0)
        self.assertEqual(lens[0], 20.0)
        # Peak must stay well below the 28 MB raw file size (~1-3 MB for 100k-edge chunks).
        self.assertLess(peak, 8 * 1024 * 1024)

    def test_run_msa_builder_with_small_chunks_matches_reference(self):
        for name, options in (
            ("complete", dict(seed=11)),
            ("sparse_calibrated", dict(seed=12, complete=False)),
            ("blast", dict(seed=13, blast=True)),
        ):
            with self.subTest(name):
                fixture = NetworkFixture(self.directory / name, **options)
                if not options.get("blast"):
                    fixture.record_scoring((-3.0, -0.5), "ieee_fp32")
                baseline_run = run_until_guide_tree(fixture, CALIBRATED_IMPUTATION=False)
                with mock.patch.object(Embedding_MSA, "NETWORK_EDGE_CHUNK_SIZE", 13):
                    chunked_run = run_until_guide_tree(fixture, CALIBRATED_IMPUTATION=False)
                self.assertSameArray(chunked_run["guide_tree"], baseline_run["guide_tree"])
                for got, want in zip(chunked_run["populate"][0], baseline_run["populate"][0]):
                    self.assertSameArray(got, want)


class CopheneticAccumulationTests(ArrayAssertions, unittest.TestCase):
    def test_in_place_sum_matches_per_tree_cophenetic_arrays(self):
        rng = np.random.default_rng(3)
        num_seqs = 40
        expected = np.zeros(num_seqs * (num_seqs - 1) // 2, dtype=np.float32)
        actual_single = np.zeros_like(expected)
        replicates = []
        for _ in range(4):
            tree = sch.linkage(rng.random(expected.size), method="average")
            replicates.append((tree, None))
            expected += sch.cophenet(tree).astype(np.float32)
            Embedding_MSA.add_replicates_batch_to_consensus([(tree, None)], num_seqs, actual_single)
        self.assertSameArray(actual_single, expected)

        actual_batch = np.zeros_like(expected)
        Embedding_MSA.add_replicates_batch_to_consensus(replicates, num_seqs, actual_batch)
        self.assertSameArray(actual_batch, expected)

    def test_single_sequence_and_empty_batch_add_nothing(self):
        accumulator = np.zeros(0, dtype=np.float32)
        Embedding_MSA.add_replicates_batch_to_consensus(
            [(np.zeros((0, 4)), None)], 1, accumulator
        )
        self.assertEqual(accumulator.size, 0)

        nonempty = np.ones(3, dtype=np.float32)
        Embedding_MSA.add_replicates_batch_to_consensus([], 3, nonempty)
        self.assertSameArray(nonempty, np.ones(3, dtype=np.float32))


class ConsensusFinalizeTests(ArrayAssertions, unittest.TestCase):
    """Every network, sparse or complete, averages the replicate distances of all its pairs."""

    def test_the_consensus_divides_every_pair(self):
        accumulated = np.arange(1, 11, dtype=np.float32)
        expected = accumulated / np.float32(4)

        result = Embedding_MSA.finalize_cophenetic_consensus(accumulated, 4)

        self.assertIs(result, accumulated)
        self.assertSameArray(result, expected)

    def test_zero_trees_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "NUM_TREES must be greater than zero"):
            Embedding_MSA.finalize_cophenetic_consensus(np.ones(3, dtype=np.float32), 0)

    def test_no_partial_consensus_code_is_left(self):
        for name in ("compute_sparse_cophenetic", "accumulate_sparse_path_lengths",
                     "use_full_cophenetic_consensus", "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(Embedding_MSA, name))


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
            tree, branch_lengths = Embedding_MSA.compute_single_tree_worker(
                seed, num_seqs, path, max_dist, noise_scale, tree_method
            )
        gc.collect()
        return tree, branch_lengths, dtypes

    def test_upgma_replicate_matches_float32_reference(self):
        # 1,123,250 pairs span two noise chunks.
        for num_seqs, noise_scale, seed in ((1500, 0.02, 11), (300, 0.0, 12), (300, 0.5, 13)):
            with self.subTest(num_seqs=num_seqs, noise_scale=noise_scale):
                baseline, max_dist, path = self.write_baseline(num_seqs, seed)
                expected = sch.linkage(
                    reference_replicate(seed, baseline, max_dist, noise_scale),
                    method="average",
                )
                tree, branch_lengths, dtypes = self.run_worker(
                    seed, num_seqs, path, max_dist, noise_scale, UPGMA
                )
                self.assertSameArray(tree, expected)
                self.assertIsNone(branch_lengths)
                # SciPy receives float64, so it makes no conversion copy.
                self.assertEqual(dtypes, [np.dtype(np.float64)])

    def test_neighbor_joining_replicate_matches_float32_reference(self):
        baseline, max_dist, path = self.write_baseline(120, 14)
        expected, expected_lengths = Embedding_MSA.neighbor_joining_condensed(
            reference_replicate(14, baseline, max_dist, 0.02),
            120,
            return_branch_lengths=True,
        )
        tree, branch_lengths, dtypes = self.run_worker(
            14, 120, path, max_dist, 0.02, NEIGHBOR_JOINING
        )
        self.assertSameArray(tree, expected)
        self.assertSameArray(branch_lengths, expected_lengths)
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
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                seen = run_until_guide_tree(
                    fixture,
                    NORMALIZATION_MODE="average_sequence",
                    BOOTSTRAP_TREE=True,
                    NUM_TREES=2,
                    CALIBRATED_IMPUTATION=calibrated,
                )
                self.assert_reference_edges(seen, expected_edges)
                self.assertEqual(seen["guide_tree"].shape, (expected_edges[4] - 1, 4))
                self.assertEqual(seen["linkage_dtypes"], [np.dtype(np.float64)] * 3)

    def test_sparse_network_consensus_averages_every_pair(self):
        # A sparse network gets the all-pairs consensus a complete one does, so the
        # imputed pairs end up as replicate-averaged tree distances too.
        fixture = NetworkFixture(self.directory, complete=False, seed=2)
        expected_edges = reference_edges(fixture, "average_sequence")
        seen = run_until_guide_tree(
            fixture, NORMALIZATION_MODE="average_sequence", BOOTSTRAP_TREE=True, NUM_TREES=3
        )
        max_distance = expected_edges[3]
        baseline = np.clip(seen["baseline"], 0.0, max_distance).astype(np.float32)
        expected_tree = reference_consensus_tree(
            baseline, max_distance, 0.02, num_trees=3, random_seed=42
        )
        self.assertSameArray(seen["guide_tree"], expected_tree)

    def test_the_consensus_is_summed_in_seed_order_whatever_order_a_pool_finishes_in(self):
        # The float32 sums depend on the order of the replicates, so the builder has to ask
        # for them in seed order (imap), not in the order they finish (imap_unordered).
        class OrderedPool(InProcessPool):
            def imap_unordered(self, func, iterable):
                return self.imap(func, iterable)

        class ReversedPool(InProcessPool):
            def imap(self, func, iterable):
                return self.imap_unordered(func, iterable)

        fixture = NetworkFixture(self.directory, seed=7)

        def consensus(pool):
            seen = run_until_guide_tree(fixture, BOOTSTRAP_TREE=True, NUM_TREES=8, pool=pool)
            return seen["linkage_inputs"][-1]

        reference = consensus(OrderedPool)
        # The other order does give other bytes here; otherwise this test would prove nothing.
        self.assertNotEqual(consensus(ReversedPool).tobytes(), reference.tobytes())
        # InProcessPool gives seed order through imap and the reverse through imap_unordered.
        self.assertSameArray(consensus(InProcessPool), reference)

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


class SparseImputationReleaseTests(unittest.TestCase):
    """A sparse run frees its imputation arrays before the guide tree is built.

    The n x n similarity and dense matrices and the per-edge similarity and score
    arrays are read only by the imputation, so none may be alive, or named by
    run_msa_builder, when the replicate trees start (or, without perturbed trees,
    when the one tree is built). The network has far more than the 100,000 pairs
    a fit takes: below that, a fit without the calibration sample keeps the
    similarity and score arrays themselves as its own, and they would stay.
    """

    RELEASED = ("cos_sim_mat", "dense_scores", "dense_dists", "X_cos", "Y_align", "norm_scores")
    # Held when the replicates start, in bytes per n x n pair: the float32 accumulator
    # (2) and a few small arrays: 3.3 on this network. Before the arrays were freed it was 26.9.
    HELD_BUDGET = 6.0

    @classmethod
    def setUpClass(cls):
        cls.fixtures = {}
        for name, sample in (("no sample", False), ("sample", True), ("complete", None)):
            temporary = tempfile.TemporaryDirectory()
            cls.addClassCleanup(temporary.cleanup)
            fixture = NetworkFixture(
                temporary.name, sequences=1415 if sample is not None else 120, missing=(), ghosts=0,
                duplicates=0, out_of_range=False, complete=sample is None, seed=4,
            )
            if sample:
                fixture.add_calibration_sample(count=12)
            cls.fixtures[name] = fixture
        # The first run compiles the numba kernels, which tracemalloc would otherwise count.
        for fixture in cls.fixtures.values():
            run_until_guide_tree(fixture, record_edges=False)

    def assert_released(self, arrays, fixture):
        count, edges = len(fixture.net_headers), fixture.arr_i.size
        self.assertEqual(sorted(set(arrays) & set(self.RELEASED)), [])
        for name, shape in arrays.items():
            # (A complete network has as many edges as the condensed matrix and its memory map have pairs.)
            if name not in ("D_final_cond", "mm_baseline"):
                self.assertNotIn(shape, ((count, count), (edges,)), name)

    def cases(self):
        for name in ("no sample", "sample"):
            for switch in (True, False):
                if name == "no sample" and not switch:
                    continue
                yield name, switch
        yield "complete", True

    def test_the_arrays_are_gone_when_the_replicate_trees_start(self):
        for name, switch in self.cases():
            with self.subTest(network=name, calibrated=switch):
                fixture = self.fixtures[name]
                self.assertGreater(fixture.arr_i.size, 100_000 if name != "complete" else 1_000)
                HeldMemoryPool.arrays = None
                run_until_guide_tree(
                    fixture, record_edges=False, pool=HeldMemoryPool, BOOTSTRAP_TREE=True,
                    CALIBRATED_IMPUTATION=switch,
                )
                self.assertIn("D_final_cond", HeldMemoryPool.arrays)
                self.assert_released(HeldMemoryPool.arrays, fixture)

    def test_the_arrays_are_gone_when_the_one_tree_is_built(self):
        for name, switch in self.cases():
            with self.subTest(network=name, calibrated=switch):
                fixture = self.fixtures[name]
                seen = run_until_guide_tree(
                    fixture, record_edges=False, BOOTSTRAP_TREE=False, CALIBRATED_IMPUTATION=switch
                )
                self.assertIn("D_final_cond", seen["linkage_arrays"])
                self.assert_released(seen["linkage_arrays"], fixture)

    def test_a_sparse_run_starts_its_replicates_holding_little(self):
        for name in ("no sample", "sample"):
            with self.subTest(network=name):
                fixture = self.fixtures[name]
                count = len(fixture.net_headers)
                HeldMemoryPool.held_bytes = None
                tracemalloc.start()
                try:
                    run_until_guide_tree(
                        fixture, record_edges=False, pool=HeldMemoryPool, BOOTSTRAP_TREE=True
                    )
                finally:
                    tracemalloc.stop()
                self.assertIsNotNone(HeldMemoryPool.held_bytes)
                self.assertLess(HeldMemoryPool.held_bytes / (count * count), self.HELD_BUDGET)


GIB = 1024 ** 3
MIB = 1024 ** 2


class GuideTreeWorkerPlanTests(unittest.TestCase):
    """The guide-tree workers are capped by the memory that is free (pure arithmetic, no processes)."""

    LARGE_SET = 44127
    LARGE_PAIRS = LARGE_SET * (LARGE_SET - 1) // 2  # 973,574,001

    def plan(self, num_seqs, method, requested, total_gib, available_gib):
        return Embedding_MSA.guide_tree_worker_plan(
            num_seqs, method, requested, int(total_gib * GIB), int(available_gib * GIB)
        )

    def test_the_constants_are_the_measured_figures(self):
        self.assertEqual(Embedding_MSA.UPGMA_BYTES_PER_PAIR, 16)
        self.assertEqual(Embedding_MSA.NEIGHBOR_JOINING_BYTES_PER_PAIR, 28)
        self.assertEqual(Embedding_MSA.WORKER_START_BYTES, 768 * MIB)
        self.assertEqual(Embedding_MSA.GUIDE_TREE_MIN_HEADROOM_BYTES, 2 * GIB)
        self.assertEqual(Embedding_MSA.GUIDE_TREE_HEADROOM_FRACTION, 0.10)

    def test_large_set_with_upgma_is_capped_by_the_free_memory(self):
        # need = 973,574,001 * 16 + 768 MiB = 16,382,490,384 bytes (15.3 GiB)
        # room = 80 GiB - int(96 GiB * 0.10) = 85,899,345,920 - 10,307,921,510 = 75,591,424,410
        # fit  = 75,591,424,410 // 16,382,490,384 = 4
        need = self.LARGE_PAIRS * 16 + 768 * MIB
        room = 80 * GIB - max(2 * GIB, int(96 * GIB * 0.10))
        self.assertEqual(need, 16382490384)
        self.assertEqual(room, 75591424410)
        plan = self.plan(self.LARGE_SET, "UPGMA (Fast)", 8, 96, 80)
        self.assertEqual((plan.workers, plan.need_bytes, plan.room_bytes, plan.fits_one), (4, need, room, True))
        # A request the memory holds is left as it is.
        self.assertEqual(self.plan(self.LARGE_SET, "UPGMA (Fast)", 3, 96, 80).workers, 3)

    def test_large_set_with_neighbor_joining_needs_more_per_worker(self):
        # need = 973,574,001 * 28 + 768 MiB = 28,065,378,396 bytes; fit = 75,591,424,410 // 28,065,378,396 = 2
        need = self.LARGE_PAIRS * 28 + 768 * MIB
        self.assertEqual(need, 28065378396)
        plan = self.plan(self.LARGE_SET, "Neighbor-joining (Slow)", 8, 96, 80)
        self.assertEqual((plan.workers, plan.need_bytes, plan.fits_one), (2, need, True))

    def test_a_small_set_is_decided_by_the_start_cost(self):
        # 1,000 sequences: 499,500 pairs, need = 499,500 * 16 + 768 MiB = 813,298,368 bytes.
        # room = 8 GiB - max(2 GiB, int(16 GiB * 0.10)) = 6 GiB = 6,442,450,944; fit = 7 (8 * need is 6,506,386,944).
        pairs = 1000 * 999 // 2
        need = pairs * 16 + 768 * MIB
        self.assertEqual(need, 813298368)
        self.assertGreater(8 * need, 6 * GIB)
        plan = self.plan(1000, "UPGMA (Fast)", 16, 16, 8)
        self.assertEqual((plan.workers, plan.room_bytes), (7, 6 * GIB))

    def test_no_room_for_one_worker_gives_one_worker_and_says_it_does_not_fit(self):
        # Available below the headroom: room is 0.
        below = self.plan(1000, "UPGMA (Fast)", 8, 16, 1)
        self.assertEqual((below.workers, below.room_bytes, below.fits_one), (1, 0, False))
        # 2.5 GiB available: room 0.5 GiB = 536,870,912 is below one worker's need of 813,298,368.
        short = self.plan(1000, "UPGMA (Fast)", 8, 16, 2.5)
        self.assertEqual((short.workers, short.room_bytes, short.fits_one), (1, 512 * MIB, False))

    def test_unknown_memory_leaves_the_request_alone(self):
        for total, available in ((0, 0), (16 * GIB, 0), (0, 8 * GIB), (-1, -1)):
            with self.subTest(total=total, available=available):
                plan = Embedding_MSA.guide_tree_worker_plan(
                    self.LARGE_SET, "UPGMA (Fast)", 16, total, available
                )
                self.assertEqual(plan.workers, 16)
                self.assertTrue(plan.fits_one)
                self.assertIsNone(plan.room_bytes)

    def test_one_worker_is_never_raised_and_never_lowered_below_one(self):
        self.assertEqual(self.plan(1000, "UPGMA (Fast)", 1, 256, 200).workers, 1)
        self.assertEqual(self.plan(self.LARGE_SET, "UPGMA (Fast)", 1, 16, 1).workers, 1)
        self.assertEqual(self.plan(1000, "UPGMA (Fast)", 0, 256, 200).workers, 1)

    def test_an_unknown_tree_method_uses_the_upgma_figure(self):
        # compute_single_tree_worker treats every method but Neighbor-joining as UPGMA; so does the plan.
        upgma = self.plan(self.LARGE_SET, "UPGMA (Fast)", 8, 96, 80)
        other = self.plan(self.LARGE_SET, "something else", 8, 96, 80)
        self.assertEqual(other, upgma)


class GuideTreeMemoryReaderTests(unittest.TestCase):
    def test_it_returns_the_engines_quota_aware_figures(self):
        with mock.patch("Embedding_Alignment_Engine.system_memory_bytes", return_value=(96 * GIB, 80 * GIB)):
            self.assertEqual(Embedding_MSA.read_guide_tree_memory(), (96 * GIB, 80 * GIB))

    def test_it_returns_zeros_when_the_engines_reader_raises(self):
        with mock.patch("Embedding_Alignment_Engine.system_memory_bytes", side_effect=RuntimeError("no memory info")):
            self.assertEqual(Embedding_MSA.read_guide_tree_memory(), (0, 0))


class RecordingPool(InProcessPool):
    """Runs the replicates in this process and records how many workers the builder asked for."""

    created = []

    def __init__(self, processes=None):
        super().__init__(processes)
        RecordingPool.created.append(processes)


class GuideTreeWorkerStageTests(unittest.TestCase):
    """Step 8 of run_msa_builder starts its pool with the capped number of workers (no real processes)."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name

    def run_stage(self, memory, workers=3):
        RecordingPool.created = []
        lines = []
        fixture = NetworkFixture(self.directory, seed=7)
        with mock.patch.object(Embedding_MSA, "read_guide_tree_memory", return_value=memory), \
                mock.patch.object(
                    Embedding_MSA, "print", side_effect=lambda *args, **kw: lines.append(" ".join(map(str, args))),
                    create=True,
                ):
            run_until_guide_tree(
                fixture, record_edges=False, pool=RecordingPool,
                BOOTSTRAP_TREE=True, NUM_TREES=3, WORKERS=workers,
            )
        return RecordingPool.created, [line for line in lines if line.startswith("[Hardware] ") and "uide-tree" in line]

    def test_a_small_machine_lowers_the_workers_and_says_so_once(self):
        # 4 GiB machine with 3 GiB free: room 3 - 2 = 1 GiB, one worker needs about 768 MiB, so fit is 1.
        created, lines = self.run_stage((4 * GIB, 3 * GIB), workers=3)
        self.assertEqual(created, [1])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("[Hardware] Guide-tree workers lowered from 3 to 1: each needs about 0.8 GB for "))
        self.assertTrue(lines[0].endswith("sequences with UPGMA (Fast), and 3.0 GB of memory is free."))

    def test_plenty_of_memory_gives_the_requested_workers_and_prints_nothing(self):
        created, lines = self.run_stage((64 * GIB, 60 * GIB), workers=3)
        self.assertEqual(created, [3])
        self.assertEqual(lines, [])

    def test_unknown_memory_gives_the_requested_workers_and_prints_nothing(self):
        created, lines = self.run_stage((0, 0), workers=3)
        self.assertEqual(created, [3])
        self.assertEqual(lines, [])

    def test_memory_that_does_not_hold_one_worker_says_so(self):
        created, lines = self.run_stage((4 * GIB, 2 * GIB + 100 * MIB), workers=1)
        self.assertEqual(created, [1])
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("[Hardware] One guide-tree worker needs about 0.8 GB and 2.1 GB of memory is free; "))
        self.assertTrue(lines[0].endswith("the run may run out of memory."))


if __name__ == "__main__":
    unittest.main()
