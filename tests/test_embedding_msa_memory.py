"""Embedding_MSA builds the same guide tree while holding far less memory.

The reference functions restate the code before the memory changes (commit
17a1030): a per-edge Python filter, float32 bootstrap replicates and one
cophenetic array per tree. The current code must reproduce them bit for bit.
"""

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


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA
from utilities.HDF5_Storage import create_metadata_first_file, mark_generation_complete


RESIDUES = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
MODEL = "test_model"
EMBED_NAME = f"Fixture_[{MODEL}]_embeddings.h5"
NETWORK_NAME = f"Fixture_[{MODEL}]_network.h5"
UPGMA = "UPGMA (Fast)"
NEIGHBOR_JOINING = "Neighbor-joining (Slow)"
MODE_INTS = {
    "alignment_length": 0,
    "shorter_sequence": 1,
    "longer_sequence": 2,
    "average_sequence": 3,
}


class NetworkFixture:
    """A small embedding database and network written to a folder.

    By default two embedded sequences are absent from the network, two
    network sequences have no embedding, some pairs are listed twice with new
    scores, and three edges point outside the header table, so every filter
    path runs.
    """

    def __init__(
        self,
        root,
        *,
        sequences=24,
        missing=(5, 17),
        ghosts=2,
        complete=True,
        duplicates=4,
        out_of_range=True,
        blast=False,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        self.root = pathlib.Path(root)
        self.embed_dir = self.root / "Embeddings"
        self.network_dir = self.root / "Networks"
        self.msa_dir = self.root / "Multiple_Alignments"
        for directory in (self.embed_dir, self.network_dir, self.msa_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.blast = blast

        self.emb_headers = [f"seq{index:04d}" for index in range(sequences)]
        self.emb_sequences = [
            "".join(rng.choice(RESIDUES, int(rng.integers(10, 31))))
            for _ in self.emb_headers
        ]
        self.net_headers = [
            header for index, header in enumerate(self.emb_headers)
            if index not in missing
        ]
        for ghost in range(ghosts):
            self.net_headers.insert(1 + 7 * ghost, f"ghost{ghost}")

        node_count = len(self.net_headers)
        sources, targets = np.triu_indices(node_count, k=1)
        if not complete:
            listed = rng.random(sources.size) < 0.6
            sources, targets = sources[listed], targets[listed]
        order = rng.permutation(sources.size)
        sources, targets = sources[order], targets[order]
        flipped = rng.random(sources.size) < 0.5
        sources, targets = (
            np.where(flipped, targets, sources),
            np.where(flipped, sources, targets),
        )
        if duplicates:
            again = rng.choice(sources.size, duplicates, replace=False)
            sources = np.concatenate([sources, targets[again]])
            targets = np.concatenate([targets, sources[again]])
        if out_of_range:
            sources = np.concatenate([sources, [node_count + 2, 1, node_count]])
            targets = np.concatenate([targets, [0, 65535, node_count + 1]])
        self.arr_i = sources.astype(np.uint16)
        self.arr_j = targets.astype(np.uint16)

        edge_count = self.arr_i.size
        if blast:
            self.score = rng.uniform(0.0, 180.0, edge_count).astype(np.float32)
            self.length = np.ones_like(self.score)
        else:
            self.score = rng.uniform(-20.0, 400.0, edge_count).astype(np.float32)
            # Zero lengths exercise the max(length, 1) guard.
            self.length = rng.integers(0, 80, edge_count).astype(np.uint16)

        with h5py.File(self.embed_dir / EMBED_NAME, "w") as embeddings:
            group = create_metadata_first_file(
                embeddings, self.emb_headers, self.emb_sequences, MODEL, "float16"
            )
            for header, sequence in zip(self.emb_headers, self.emb_sequences):
                group.create_dataset(
                    header,
                    data=rng.standard_normal((len(sequence), 8)).astype(np.float16),
                )
            mark_generation_complete(embeddings)

        with h5py.File(self.network_dir / NETWORK_NAME, "w") as network:
            network.attrs["model_name"] = "BLAST" if blast else MODEL
            network.create_dataset(
                "headers",
                data=np.asarray(self.net_headers, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            network.create_dataset("i", data=self.arr_i)
            network.create_dataset("j", data=self.arr_j)
            if blast:
                network.create_dataset("score", data=self.score)
            else:
                network.create_dataset(
                    "seq_lens", data=np.full(node_count, 20, dtype=np.uint16)
                )
                network.create_dataset("g_score", data=self.score)
                network.create_dataset("g_len", data=self.length)
                network.create_dataset("l_score", data=self.score / 2)
                network.create_dataset("l_len", data=self.length)


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


class InProcessPool:
    """Runs bootstrap trees in this process, in seed order.

    The consensus sum is then reproducible, and test patches reach the
    workers.
    """

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap_unordered(self, func, iterable):
        return map(func, iterable)


class GuideTreeBuilt(Exception):
    """Stops run_msa_builder once the guide tree exists."""


class HeldMemoryPool(InProcessPool):
    """Records the traced memory still held when bootstrapping starts."""

    held_bytes = None

    def imap_unordered(self, func, iterable):
        HeldMemoryPool.held_bytes = tracemalloc.get_traced_memory()[0]
        raise GuideTreeBuilt


def builder_settings(fixture, **settings):
    """Return the module globals that point run_msa_builder at the fixture."""
    values = {
        "USE_SEQUENCE_FILTER": False,
        "INPUT_FASTA": "",
        "INPUT_EMBED": EMBED_NAME,
        "INPUT_NETWORK": NETWORK_NAME,
        "ALIGNMENT_SCORE": "global",
        "NORMALIZATION_MODE": "alignment_length",
        "TREE_METHOD": UPGMA,
        "BOOTSTRAP_TREE": False,
        "NUM_TREES": 3,
        "NOISE_SCALE": 0.02,
        "RANDOM_SEED": 42,
        "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS": False,
        "WORKERS": 1,
        "DEVICE_SELECTION": "cpu",
        "SHOW_REGRESSION_PLOT": False,
        "POOLING_METHOD": "max",
        "LENGTH_RATIO_POWER": 2.0,
        "FASTA_DIR": str(fixture.root),
        "EMBED_DIR": str(fixture.embed_dir),
        "NETWORK_DIR": str(fixture.network_dir),
        "MSA_DIR": str(fixture.msa_dir),
        "SAFE_TEMP_DIR": str(fixture.msa_dir),
        # run_msa_builder assigns these globals; patching restores them.
        "FULL_INPUT_FASTA": Embedding_MSA.FULL_INPUT_FASTA,
        "FULL_INPUT_EMBED": Embedding_MSA.FULL_INPUT_EMBED,
        "FULL_INPUT_NETWORK": Embedding_MSA.FULL_INPUT_NETWORK,
        "OUTPUT_FASTA": Embedding_MSA.OUTPUT_FASTA,
        "_seq_set": Embedding_MSA._seq_set,
        "_model_name": Embedding_MSA._model_name,
    }
    values.update(settings)
    return values


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


class ArrayAssertions:
    def assertSameArray(self, actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())


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
