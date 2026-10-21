"""utilities.Network_Top_Edges and the UMAP mode of desktop.Viewer_State.prepare_network:
each node's strongest edges, found while the network's pairs are streamed.

The reference is the pandas code prepare_network used before (a full sort of every
pair and two group-bys). Its default sort is not stable, so which of several equal
scores it keeps at a node's last place is arbitrary; the kernel's rule is score,
then the pair stored first, which is what the same code gives with a stable sort.
Without equal scores at a last place the two are the same edges in the same order.
"""

import io
import pathlib
import sys
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_State import _read_pair_scores, prepare_network
from utilities import Network_Score_Statistics as scan
from utilities import Network_Top_Edges as top


def reference_positions(sources, targets, scores, keep, kind):
    """The stored positions the former pandas code kept (kind: its sort algorithm)."""
    frame = pd.DataFrame(
        {"u": sources, "v": targets, "score": scores, "idx": np.arange(len(scores))}
    )
    ordered = frame.sort_values("score", ascending=False, kind=kind)
    best_as_source = ordered.groupby("u").head(keep)["idx"]
    best_as_target = ordered.groupby("v").head(keep)["idx"]
    return np.sort(pd.concat([best_as_source, best_as_target]).unique())


class ArraySource:
    """The part of a ScoreSource that strongest_edges uses, over arrays in memory."""

    def __init__(self, sources, targets, scores):
        self.sources, self.targets, self.scores = sources, targets, scores
        self.edge_count = len(scores)

    def read(self, start, stop, nodes=False, check_nodes=False, positions=False):
        stop = min(stop, self.edge_count)
        return scan.Slice(
            self.scores[start:stop].copy(),
            self.sources[start:stop].copy(),
            self.targets[start:stop].copy(),
            np.arange(start, stop, dtype=np.int64),
        )


def random_network(rng, node_count, pair_count, kind):
    """Pairs in no particular order (as stored networks are), with repeats and self-pairs."""
    sources = rng.integers(0, node_count, pair_count).astype(np.uint16)
    targets = rng.integers(0, node_count, pair_count).astype(np.uint16)
    if kind == "distinct":
        scores = rng.permutation(pair_count).astype(np.float32)
    elif kind == "ties":
        scores = rng.integers(0, 4, pair_count).astype(np.float32)
    elif kind == "float64":
        scores = rng.integers(0, 6, pair_count).astype(np.float64) / 3.0
    elif kind == "integers":
        scores = rng.integers(-3, 4, pair_count).astype(np.int64)
    elif kind == "nan":
        scores = rng.integers(0, 5, pair_count).astype(np.float32)
        scores[rng.random(pair_count) < 0.15] = np.nan
    else:
        raise ValueError(kind)
    return sources, targets, scores


class StrongestEdgesTests(unittest.TestCase):
    def run_kernel(self, network, node_count, keep, *, chunk, workers):
        with mock.patch.object(scan, "CHUNK_EDGES", chunk):
            return top.strongest_edges(
                ArraySource(*network), node_count, keep, workers=workers
            )

    def assert_matches(self, network, node_count, keep, kind, *, chunk, workers):
        sources, targets, scores = network
        positions, found_sources, found_targets, found_scores = self.run_kernel(
            network, node_count, keep, chunk=chunk, workers=workers
        )
        expected = reference_positions(sources, targets, scores, keep, kind)
        np.testing.assert_array_equal(positions, expected)
        np.testing.assert_array_equal(found_sources, sources[expected])
        np.testing.assert_array_equal(found_targets, targets[expected])
        self.assertEqual(found_scores.dtype, scores.dtype)
        np.testing.assert_array_equal(found_scores, scores[expected])

    def test_matches_a_stable_sort_whatever_the_scores_chunks_and_threads(self):
        # Equal scores, NaN scores, other dtypes, k above a node's degree, k = 1, nodes
        # without pairs, and a pass cut into many chunks run by several threads.
        cases = 0
        for seed in range(6):
            rng = np.random.default_rng(seed)
            for score_kind in ("distinct", "ties", "float64", "integers", "nan"):
                for keep in (1, 2, 5, 40):
                    node_count = int(rng.integers(2, 45))
                    network = random_network(rng, node_count, int(rng.integers(1, 700)), score_kind)
                    for chunk, workers in ((1_000, 1), (37, 1), (11, 3)):
                        with self.subTest(seed=seed, scores=score_kind, keep=keep, chunk=chunk, workers=workers):
                            self.assert_matches(
                                network, node_count, keep, "stable", chunk=chunk, workers=workers
                            )
                        cases += 1
        self.assertGreater(cases, 300)

    def test_without_equal_scores_it_keeps_what_the_former_sort_kept(self):
        for seed in range(12):
            rng = np.random.default_rng(100 + seed)
            node_count = int(rng.integers(3, 60))
            network = random_network(rng, node_count, int(rng.integers(50, 900)), "distinct")
            for keep in (1, 3, 15):
                with self.subTest(seed=seed, keep=keep):
                    self.assert_matches(network, node_count, keep, "quicksort", chunk=64, workers=2)

    def kept_from_node_zero(self, targets, scores, keep, *, chunk, workers):
        """Which of node 0's pairs (to ``targets``, in this stored order) its source group keeps.

        Every target also gets ``keep`` better pairs from other nodes, stored after, so
        no pair of node 0 is kept as a target: only node 0's own ranking decides.
        """
        count = len(targets)
        rival_sources = np.repeat(np.arange(100, 100 + keep), count).astype(np.uint16)
        rival_targets = np.tile(targets, keep)
        sources = np.concatenate([np.zeros(count, dtype=np.uint16), rival_sources])
        all_targets = np.concatenate([targets, rival_targets]).astype(np.uint16)
        all_scores = np.concatenate([scores, np.full(len(rival_sources), 1e6)]).astype(np.float32)
        positions, *_ = self.run_kernel(
            (sources, all_targets, all_scores), 100 + keep, keep, chunk=chunk, workers=workers
        )
        return positions[positions < count]

    def test_equal_scores_keep_the_pairs_stored_first_whatever_the_thread_order(self):
        # Six pairs with one score, stored out of target order: keep=2 keeps the first two stored.
        targets = np.array([4, 2, 6, 1, 5, 3])
        for chunk, workers in ((100, 1), (2, 1), (1, 3), (1, 8)):
            with self.subTest(chunk=chunk, workers=workers):
                kept = self.kept_from_node_zero(
                    targets, np.full(6, 2.0), 2, chunk=chunk, workers=workers
                )
                np.testing.assert_array_equal(kept, [0, 1])

    def test_higher_scores_replace_earlier_equal_ones(self):
        cases = (
            ([1, 2, 2, 3], 1, [3]),      # the highest
            ([1, 2, 2], 2, [1, 2]),      # both equal scores beat the lower one
            ([2, 2, 3], 2, [0, 2]),      # the 3, then the first of the equal scores
            ([2, 2, 2, 5, 2], 3, [0, 1, 3]),
        )
        for scores, keep, expected in cases:
            for chunk, workers in ((100, 1), (1, 1), (1, 3)):
                with self.subTest(scores=scores, keep=keep, chunk=chunk, workers=workers):
                    kept = self.kept_from_node_zero(
                        np.arange(1, len(scores) + 1), np.array(scores, dtype=float), keep,
                        chunk=chunk, workers=workers,
                    )
                    np.testing.assert_array_equal(kept, expected)

    def test_nan_scores_rank_after_every_other_score(self):
        nan = np.nan
        cases = (
            ([nan, 1.0, nan], 2, [0, 1]),           # the score, then the first NaN
            ([nan, nan, 1.0, nan], 3, [0, 1, 2]),   # the score, then the NaNs stored first
            ([nan, 1.0, 2.0], 2, [1, 2]),           # no NaN while real scores fill the place
        )
        for scores, keep, expected in cases:
            for chunk, workers in ((100, 1), (1, 1), (1, 3)):
                with self.subTest(scores=scores, keep=keep, chunk=chunk, workers=workers):
                    kept = self.kept_from_node_zero(
                        np.arange(1, len(scores) + 1), np.array(scores), keep,
                        chunk=chunk, workers=workers,
                    )
                    np.testing.assert_array_equal(kept, expected)

    def test_a_node_with_fewer_pairs_than_k_keeps_them_all(self):
        rng = np.random.default_rng(3)
        network = random_network(rng, 20, 60, "ties")
        positions, *_ = self.run_kernel(network, 20, 500, chunk=7, workers=2)
        np.testing.assert_array_equal(positions, np.arange(60))

    def test_nothing_is_kept_without_pairs_or_without_a_limit(self):
        empty = (np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float32))
        for keep in (15, 0):
            positions, sources, targets, scores = self.run_kernel(empty, 5, keep, chunk=10, workers=2)
            self.assertEqual((len(positions), len(sources), len(targets), len(scores)), (0, 0, 0, 0))
            self.assertEqual(scores.dtype, np.float32)
        rng = np.random.default_rng(4)
        positions, *_ = self.run_kernel(random_network(rng, 8, 30, "ties"), 8, 0, chunk=10, workers=1)
        self.assertEqual(len(positions), 0)

    def test_the_result_does_not_depend_on_chunks_or_threads(self):
        rng = np.random.default_rng(5)
        network = random_network(rng, 40, 1500, "ties")
        results = [
            self.run_kernel(network, 40, 4, chunk=chunk, workers=workers)
            for chunk, workers in ((10_000, 1), (97, 1), (97, 4), (13, 8), (1, 2))
        ]
        for other in results[1:]:
            for first, second in zip(results[0], other):
                np.testing.assert_array_equal(first, second)

    def test_the_buffer_budget_limits_the_threads_not_the_result(self):
        # 44,127 nodes: 16 threads at k=15, 7 at k=100, one at k=500 (1 GiB of buffers).
        self.assertEqual(top.buffer_thread_count(44127, 15, 4, 16), 16)
        self.assertEqual(top.buffer_thread_count(44127, 100, 4, 16), 7)
        self.assertEqual(top.buffer_thread_count(44127, 500, 4, 16), 1)
        self.assertEqual(top.buffer_thread_count(44127, 15, 4, 1), 1)
        rng = np.random.default_rng(6)
        network = random_network(rng, 30, 800, "ties")
        with mock.patch.object(top, "BUFFER_BUDGET_BYTES", 1):
            capped = self.run_kernel(network, 30, 5, chunk=50, workers=8)
        for first, second in zip(capped, self.run_kernel(network, 30, 5, chunk=50, workers=8)):
            np.testing.assert_array_equal(first, second)

    def test_scores_that_cannot_be_ranked_are_refused(self):
        with self.assertRaises(TypeError):
            top._lowest_value(np.dtype("U3"))


def write_network(path, node_count, rng, *, blast=False, ties=False, with_nodes=True):
    """A complete network stored in no particular order, in the dtypes real files use."""
    sources, targets = np.triu_indices(node_count, k=1)
    order = rng.permutation(len(sources))
    sources, targets = sources[order], targets[order]
    pair_count = len(sources)
    columns = {"i": sources.astype(np.uint16), "j": targets.astype(np.uint16)}
    if blast:
        if ties:
            columns["score"] = rng.integers(0, 5, pair_count).astype(np.float32) * 25.0
        else:
            columns["score"] = rng.permutation(pair_count).astype(np.float32)
    else:
        columns["seq_lens"] = rng.integers(1, 9, node_count).astype(np.uint16)
        for kind in ("g", "l"):
            if ties:
                columns[f"{kind}_score"] = rng.integers(0, 6, pair_count).astype(np.float32)
                columns[f"{kind}_len"] = rng.integers(1, 3, pair_count).astype(np.uint16)
            else:
                # Length 1: the normalized scores are the distinct raw scores.
                columns[f"{kind}_score"] = rng.permutation(pair_count).astype(np.float32)
                columns[f"{kind}_len"] = np.ones(pair_count, dtype=np.uint16)
    with h5py.File(path, "w") as network:
        network.attrs["model_name"] = "BLAST" if blast else "model"
        network.create_dataset(
            "headers", data=[f"N{index}".encode() for index in range(node_count)]
        )
        for name, values in columns.items():
            network.create_dataset(name, data=values)
    return columns


def settings_for(**overrides):
    values = {
        "NODE_FASTA_FILE": "",
        "ALIGNMENT_SCORE": "global",
        "NORM_MODE": "alignment_length",
        "SIMILARITY_THRESHOLD": 0.0,
        "TOP_EDGE_PERCENT": None,
        "UMAP_MODE": True,
        "UMAP_NEIGHBORS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def former_umap_edges(path, settings, selected, kind):
    """What prepare_network returned in UMAP mode before the streaming kernel."""
    with h5py.File(path, "r") as network:
        headers = [name.decode() for name in network["headers"][:]]
        settings = SimpleNamespace(**{**vars(settings), "INPUT_IS_EVALUE": "score" in network})
        sources, targets = network["i"][:], network["j"][:]
        kept = np.array([index for index, name in enumerate(headers) if name in selected])
        kept_mask = np.zeros(len(headers), dtype=bool)
        kept_mask[kept] = True
        valid = kept_mask[sources] & kept_mask[targets]
        valid_sources, valid_targets = sources[valid], targets[valid]
        scores = _read_pair_scores(network, settings, valid, valid_sources, valid_targets)
    keep = int(settings.UMAP_NEIGHBORS)
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[reference_positions(valid_sources, valid_targets, scores, keep, kind)] = True
    index_map = np.zeros(len(headers), dtype=np.int32)
    index_map[kept] = np.arange(len(kept))
    edges = np.column_stack((index_map[valid_sources[chosen]], index_map[valid_targets[chosen]]))
    return [headers[index] for index in kept], edges.astype(np.int32), scores[chosen]


class PrepareNetworkUmapTests(unittest.TestCase):
    def prepared(self, path, settings, selected):
        with h5py.File(path, "r") as network, redirect_stdout(io.StringIO()):
            return prepare_network(network, settings=settings, selected_fasta_headers=selected)

    def assert_same_output(self, found, expected):
        self.assertEqual(found[0], expected[0])
        self.assertEqual(found[1].dtype, expected[1].dtype)
        np.testing.assert_array_equal(found[1], expected[1])
        self.assertEqual(found[2].dtype, expected[2].dtype)
        np.testing.assert_array_equal(found[2], expected[2])

    def test_every_alignment_setting_matches_the_former_output(self):
        node_count = 40
        everyone = [f"N{index}" for index in range(node_count)]
        subset = [f"N{index}" for index in range(0, node_count, 3)]
        with tempfile.TemporaryDirectory() as folder:
            distinct = pathlib.Path(folder) / "distinct.h5"
            tied = pathlib.Path(folder) / "tied.h5"
            write_network(distinct, node_count, np.random.default_rng(1))
            write_network(tied, node_count, np.random.default_rng(2), ties=True)
            for score, norm in (
                ("global", "alignment_length"), ("local", "alignment_length"),
                ("global", "shorter_sequence"), ("local", "longer_sequence"),
                ("global", "average_sequence"),
            ):
                for selected_label, selected in (("every node", everyone), ("a subset", subset)):
                    for keep in (2, 15, 60):
                        settings = settings_for(
                            ALIGNMENT_SCORE=score, NORM_MODE=norm, UMAP_NEIGHBORS=keep
                        )
                        with self.subTest(score=score, norm=norm, nodes=selected_label, keep=keep):
                            found = self.prepared(tied, settings, selected)
                            self.assert_same_output(
                                found, former_umap_edges(tied, settings, set(selected), "stable")
                            )
                            if norm == "alignment_length" and score == "global":
                                # No equal scores: the former default sort kept the same edges.
                                found = self.prepared(distinct, settings, selected)
                                self.assert_same_output(
                                    found,
                                    former_umap_edges(distinct, settings, set(selected), "quicksort"),
                                )

    def test_e_value_networks_match_the_former_output(self):
        node_count = 40
        everyone = [f"N{index}" for index in range(node_count)]
        with tempfile.TemporaryDirectory() as folder:
            for ties in (False, True):
                path = pathlib.Path(folder) / f"blast_{ties}.h5"
                write_network(path, node_count, np.random.default_rng(3), blast=True, ties=ties)
                for selected in (everyone, everyone[1::2]):
                    settings = settings_for(UMAP_NEIGHBORS=7)
                    with self.subTest(ties=ties, nodes=len(selected)):
                        found = self.prepared(path, settings, selected)
                        self.assertTrue(settings.INPUT_IS_EVALUE)
                        self.assert_same_output(
                            found,
                            former_umap_edges(path, settings, set(selected), "stable"),
                        )
                        if not ties:
                            self.assert_same_output(
                                found, former_umap_edges(path, settings, set(selected), "quicksort")
                            )

    def test_an_e_value_network_needs_no_alignment_settings(self):
        # E-value scores are used as stored, so a settings object without an alignment
        # score or normalization (the BLAST case) still prepares the network.
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "blast.h5"
            write_network(path, 12, np.random.default_rng(8), blast=True)
            settings = SimpleNamespace(
                NODE_FASTA_FILE="", SIMILARITY_THRESHOLD=None, TOP_EDGE_PERCENT=None,
                UMAP_MODE=True, UMAP_NEIGHBORS=3,
            )
            headers = [f"N{index}" for index in range(12)]
            found = self.prepared(path, settings, headers)
            expected = former_umap_edges(
                path, settings_for(UMAP_NEIGHBORS=3), set(headers), "stable"
            )
        self.assert_same_output(found, expected)

    def test_the_kept_edges_keep_the_stored_pair_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            columns = write_network(path, 30, np.random.default_rng(4), blast=True)
            headers = [f"N{index}" for index in range(30)]
            _, edges, scores = self.prepared(path, settings_for(UMAP_NEIGHBORS=3), headers)
        stored = {
            (int(i), int(j)): index for index, (i, j) in enumerate(zip(columns["i"], columns["j"]))
        }
        positions = [stored[(int(i), int(j))] for i, j in edges]
        self.assertEqual(positions, sorted(positions))
        np.testing.assert_array_equal(scores, columns["score"][positions])

    def test_an_empty_network_has_no_edges(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "empty.h5"
            with h5py.File(path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([], dtype=np.int32))
                network.create_dataset("j", data=np.asarray([], dtype=np.int32))
                network.create_dataset("score", data=np.asarray([], dtype=np.float32))
            headers, edges, scores = self.prepared(path, settings_for(), ["A", "B"])
        self.assertEqual(headers, ["A", "B"])
        self.assertEqual(edges.shape, (0, 2))
        self.assertEqual(edges.dtype, np.int32)
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(scores.dtype, np.float32)

    def test_isolated_nodes_and_a_subset_that_removes_every_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            with h5py.File(path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B", b"C", b"D"])
                network.create_dataset("i", data=np.asarray([0], dtype=np.uint16))
                network.create_dataset("j", data=np.asarray([1], dtype=np.uint16))
                network.create_dataset("score", data=np.asarray([5.0], dtype=np.float32))
            headers, edges, scores = self.prepared(path, settings_for(), ["A", "B", "C", "D"])
            np.testing.assert_array_equal(edges, [[0, 1]])
            headers, edges, scores = self.prepared(path, settings_for(), ["B", "C"])
        self.assertEqual(headers, ["B", "C"])
        self.assertEqual(edges.shape, (0, 2))
        self.assertEqual(scores.dtype, np.float32)

    def test_a_pair_index_past_the_header_count_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            with h5py.File(path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([0, 0], dtype=np.uint16))
                network.create_dataset("j", data=np.asarray([1, 2], dtype=np.uint16))
                network.create_dataset("score", data=np.asarray([8.0, 2.0], dtype=np.float32))
            for selected in (["A", "B"], ["A"]):
                with self.subTest(selected=selected), self.assertRaises(IndexError):
                    self.prepared(path, settings_for(UMAP_NEIGHBORS=1), selected)

    def test_it_says_what_it_keeps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            write_network(path, 12, np.random.default_rng(5), blast=True)
            output = io.StringIO()
            with h5py.File(path, "r") as network, redirect_stdout(output):
                _, edges, _ = prepare_network(
                    network, settings=settings_for(UMAP_NEIGHBORS=2),
                    selected_fasta_headers=[f"N{index}" for index in range(12)],
                )
        text = output.getvalue()
        self.assertIn("UMAP Mode enabled: Bypassing global threshold.", text)
        self.assertIn(f"Kept {len(edges)} edges for UMAP topology (max 2 per node direction).", text)

    def test_a_threshold_network_is_not_changed_by_the_umap_branch(self):
        # Threshold mode still filters by score and ignores UMAP_NEIGHBORS.
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            columns = write_network(path, 15, np.random.default_rng(6), blast=True)
            settings = settings_for(UMAP_MODE=False, SIMILARITY_THRESHOLD=80.0, UMAP_NEIGHBORS=1)
            _, edges, scores = self.prepared(path, settings, [f"N{index}" for index in range(15)])
        keep = columns["score"] >= 80.0
        np.testing.assert_array_equal(scores, columns["score"][keep])
        np.testing.assert_array_equal(edges[:, 0], columns["i"][keep])


class MemoryTests(unittest.TestCase):
    def test_the_peak_does_not_grow_with_the_pair_count(self):
        # The pandas code held about 90 bytes per pair (a copy of every column, the
        # sorted frame and the group codes). The stream holds a few slices per thread
        # and node_count * k entries per thread, so its peak stays far below one byte of
        # the former kind per pair.
        node_count = 2000
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            write_network(path, node_count, np.random.default_rng(7), blast=True)
            pair_count = node_count * (node_count - 1) // 2
            headers = [f"N{index}" for index in range(node_count)]
            with h5py.File(path, "r") as network, redirect_stdout(io.StringIO()):
                with mock.patch.object(scan, "CHUNK_EDGES", 10_000), mock.patch.object(
                    top, "DEFAULT_MAX_WORKERS", 2
                ):
                    # A first run loads the compiled kernels, whose Python-side set-up
                    # would otherwise count towards the peak.
                    prepare_network(
                        network, settings=settings_for(), selected_fasta_headers=headers
                    )
                    tracemalloc.start()
                    try:
                        prepare_network(
                            network, settings=settings_for(), selected_fasta_headers=headers
                        )
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
        self.assertLess(peak / pair_count, 6.0)


if __name__ == "__main__":
    unittest.main()
