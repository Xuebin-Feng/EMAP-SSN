"""utilities.Network_Threshold_Edges and the threshold mode of desktop.Viewer_State.prepare_network:
the edges at or above a threshold, and the threshold of the best TOP_EDGE_PERCENT edges, found
while the network's pairs are streamed.

The reference is the code prepare_network had before (tests/network_fixtures.py): it loaded
every column, scored all the pairs, took ``np.partition`` of all the scores for a top-percent
cutoff, and masked them with ``scores >= threshold``. The streamed output must be the same
edges, scores, order, dtypes, threshold and printed lines.
"""

import io
import itertools
import math
import pathlib
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_State import prepare_network
from tests.network_fixtures import ArraySource, former_threshold_prepare, write_network
from utilities import Network_Score_Statistics as scan
from utilities import Network_Threshold_Edges as threshold_edges


def same_number(first, second):
    return (math.isnan(first) and math.isnan(second)) or first == second


def reference_cutoff(scores, requested):
    """float(np.partition(scores, n - e)[n - e]) for e = requested limited to 1 .. n; 0.0 without scores."""
    if len(scores) == 0:
        return 0.0
    best = max(1, min(requested, len(scores)))
    kth = len(scores) - best
    return float(np.partition(scores, kth)[kth])


def random_scores(rng, count, kind):
    """Scores of a few kinds; none is -0.0 (np.partition may leave either zero at a zero cutoff)."""
    if kind == "float32":
        scores = (rng.standard_normal(count) * 10.0 ** rng.integers(-6, 6, count)).astype(np.float32)
    elif kind == "float32 ties":
        scores = rng.integers(-3, 4, count).astype(np.float32) / np.float32(2)
    elif kind == "float64":
        scores = rng.standard_normal(count) * 10.0 ** rng.integers(-30, 30, count)
    elif kind == "float64 ties":
        scores = rng.integers(0, 5, count) / 3.0
    elif kind == "int64":
        scores = rng.integers(-5, 6, count).astype(np.int64)
    elif kind == "uint16":
        scores = rng.integers(0, 4, count).astype(np.uint16)
    elif kind == "infinite":
        scores = rng.integers(0, 4, count).astype(np.float32)
        scores[rng.random(count) < 0.2] = np.inf
        scores[rng.random(count) < 0.2] = -np.inf
    else:
        raise ValueError(kind)
    scores[scores == 0] = 0  # drop any -0.0
    return scores


class CutoffTests(unittest.TestCase):
    def cutoff(self, scores, requested, *, chunk, workers):
        source = ArraySource(
            np.zeros(len(scores), dtype=np.uint16), np.zeros(len(scores), dtype=np.uint16), scores
        )
        asked = []
        with mock.patch.object(scan, "CHUNK_EDGES", chunk):
            pairs, cutoff = threshold_edges.cutoff_of_best_edges(
                source, lambda count: asked.append(count) or requested, workers=workers
            )
        self.assertEqual(asked, [len(scores)])
        self.assertEqual(pairs, len(scores))
        return cutoff, source

    def test_the_cutoff_is_what_np_partition_gave_whatever_the_scores(self):
        cases = 0
        for seed in range(5):
            rng = np.random.default_rng(seed)
            for kind in ("float32", "float32 ties", "float64", "float64 ties", "int64", "uint16", "infinite"):
                for nan_share in (0.0, 0.05, 0.7):
                    count = int(rng.integers(1, 400))
                    scores = random_scores(rng, count, kind)
                    if nan_share and scores.dtype.kind == "f":
                        scores[rng.random(count) < nan_share] = np.nan
                        # x86 arithmetic makes NaN with the sign bit set (inf - inf), which a
                        # sort key of the bits would put below every number.
                        scores[rng.random(count) < nan_share / 2] = np.copysign(np.nan, -1.0)
                    for requested in (0, 1, 3, count // 2, count, count + 5, int(rng.integers(0, count + 3))):
                        for chunk, workers in ((1_000, 1), (37, 1), (11, 3)):
                            with self.subTest(seed=seed, kind=kind, nan=nan_share, requested=requested,
                                              chunk=chunk, workers=workers):
                                found, _ = self.cutoff(scores, requested, chunk=chunk, workers=workers)
                                self.assertTrue(same_number(found, reference_cutoff(scores, requested)))
                            cases += 1
        self.assertGreater(cases, 2000)

    def test_numbers_pile_up_at_the_cutoff(self):
        # Many scores equal to the cutoff, and to its neighbours in the float32 order.
        base = np.float32(1.5)
        neighbours = np.array(
            [np.nextafter(base, np.float32(0)), base, np.nextafter(base, np.float32(9))], dtype=np.float32
        )
        rng = np.random.default_rng(1)
        scores = rng.choice(neighbours, 600)
        for requested in (1, 100, 300, 599, 600):
            for chunk in (600, 29):
                with self.subTest(requested=requested, chunk=chunk):
                    found, _ = self.cutoff(scores, requested, chunk=chunk, workers=2)
                    self.assertEqual(found, reference_cutoff(scores, requested))

    def test_nan_scores_take_the_highest_ranks(self):
        scores = np.array([3, np.nan, 1, np.nan, 2, 5], dtype=np.float32)
        # Sorted: 1 2 3 5 nan nan. The 1st and 2nd largest are NaN, the 3rd is 5.
        for requested, expected in ((1, math.nan), (2, math.nan), (3, 5.0), (4, 3.0), (6, 1.0), (60, 1.0)):
            with self.subTest(requested=requested):
                found, _ = self.cutoff(scores, requested, chunk=2, workers=1)
                self.assertTrue(same_number(found, expected))
                self.assertTrue(same_number(found, reference_cutoff(scores, requested)))
        found, _ = self.cutoff(np.full(7, np.nan, dtype=np.float64), 3, chunk=3, workers=1)
        self.assertTrue(math.isnan(found))
        # NaN with the sign bit set (what inf - inf gives on x86) is a NaN like any other.
        negative = np.copysign(np.nan, -1.0)
        scores = np.array([3, negative, 1, negative, 2, 5, np.nan], dtype=np.float32)
        for requested, expected in ((1, math.nan), (3, math.nan), (4, 5.0), (5, 3.0), (7, 1.0)):
            with self.subTest(requested=requested, signed=True):
                found, _ = self.cutoff(scores, requested, chunk=3, workers=2)
                self.assertTrue(same_number(found, expected))
                self.assertTrue(same_number(found, reference_cutoff(scores, requested)))

    def test_no_pairs_have_a_cutoff_of_zero_but_still_ask_for_the_count(self):
        found, source = self.cutoff(np.zeros(0, dtype=np.float32), 5, chunk=10, workers=2)
        self.assertEqual(found, 0.0)
        self.assertEqual(source.reads, [])

    def test_a_float32_cutoff_takes_two_passes_a_float64_one_four(self):
        # Never more slices read than the digits of the keys need (a load of the whole
        # column would show as a read of more than one slice per pass).
        rng = np.random.default_rng(2)
        for kind, passes in (("float32", 2), ("float64", 4), ("int64", 4)):
            scores = random_scores(rng, 1000, kind)
            with self.subTest(kind=kind):
                _, source = self.cutoff(scores, 100, chunk=100, workers=1)
                self.assertEqual(len(source.reads), passes * 10)
                self.assertTrue(all(stop - start <= 100 for start, stop in source.reads))

    def test_workers_do_not_change_the_cutoff(self):
        scores = random_scores(np.random.default_rng(3), 5000, "float32 ties")
        found = {self.cutoff(scores, 700, chunk=101, workers=workers)[0] for workers in (1, 2, 5, 8)}
        self.assertEqual(found, {reference_cutoff(scores, 700)})


class EdgesAtLeastTests(unittest.TestCase):
    def edges(self, network, threshold, index_map, *, chunk, workers):
        with mock.patch.object(scan, "CHUNK_EDGES", chunk):
            return threshold_edges.edges_at_least(
                ArraySource(*network), threshold, index_map, workers=workers
            )

    def test_the_edges_are_those_the_whole_column_comparison_kept(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            node_count = 30
            index_map = rng.permutation(node_count).astype(np.int32)
            for kind in ("float32", "float32 ties", "float64", "int64", "uint16"):
                count = int(rng.integers(1, 500))
                scores = random_scores(rng, count, kind)
                sources = rng.integers(0, node_count, count).astype(np.uint16)
                targets = rng.integers(0, node_count, count).astype(np.uint16)
                for threshold in (-np.inf, 0.0, 0.5, 1.5, 2.0, 3.0, 1e9, np.float64(0.5), np.float32(1.5), 1, np.nan):
                    keep = scores >= threshold
                    expected_edges = np.column_stack((index_map[sources[keep]], index_map[targets[keep]])).astype(np.int32)
                    for chunk, workers in ((1_000, 1), (37, 1), (11, 3)):
                        with self.subTest(seed=seed, kind=kind, threshold=threshold, chunk=chunk, workers=workers):
                            edges, kept_scores = self.edges(
                                (sources, targets, scores), threshold, index_map, chunk=chunk, workers=workers
                            )
                            self.assertEqual(edges.dtype, np.int32)
                            self.assertEqual(edges.shape, expected_edges.shape)
                            np.testing.assert_array_equal(edges, expected_edges)
                            self.assertEqual(kept_scores.dtype, scores.dtype)
                            np.testing.assert_array_equal(kept_scores, scores[keep])

    def test_nothing_kept_leaves_empty_arrays_of_the_right_types(self):
        sources = targets = np.zeros(5, dtype=np.uint16)
        scores = np.arange(5, dtype=np.float32)
        edges, kept = self.edges((sources, targets, scores), 99.0, np.arange(3, dtype=np.int32), chunk=2, workers=2)
        self.assertEqual((edges.shape, edges.dtype, kept.shape, kept.dtype), ((0, 2), np.int32, (0,), np.float32))
        empty = (np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.uint16), np.zeros(0, dtype=np.float64))
        edges, kept = self.edges(empty, 0.0, np.arange(3, dtype=np.int32), chunk=2, workers=1)
        self.assertEqual((edges.shape, edges.dtype, kept.shape, kept.dtype), ((0, 2), np.int32, (0,), np.float64))

    def test_a_threshold_that_cannot_be_compared_fails_as_it_did(self):
        scores = np.arange(5, dtype=np.float32)
        full = (np.zeros(5, dtype=np.uint16), np.zeros(5, dtype=np.uint16), scores)
        with self.assertRaises(TypeError):
            self.edges(full, None, np.arange(3, dtype=np.int32), chunk=2, workers=1)
        # Compared with None, no scores raise nothing, as before.
        empty = tuple(array[:0] for array in full)
        edges, _ = self.edges(empty, None, np.arange(3, dtype=np.int32), chunk=2, workers=1)
        self.assertEqual(edges.shape, (0, 2))

    def test_one_pass_and_no_slice_larger_than_the_chunk(self):
        rng = np.random.default_rng(4)
        scores = random_scores(rng, 1000, "float32")
        source = ArraySource(np.zeros(1000, dtype=np.uint16), np.zeros(1000, dtype=np.uint16), scores)
        with mock.patch.object(scan, "CHUNK_EDGES", 100):
            threshold_edges.edges_at_least(source, 0.0, np.zeros(2, dtype=np.int32), workers=1)
        self.assertEqual(len(source.reads), 10)
        self.assertTrue(all(stop - start <= 100 for start, stop in source.reads))


def thresholds_and_percents():
    thresholds = (-np.inf, 0.0, 2.5, 3.0, 1e9, np.nan)
    percents = (0.0, 0.0001, 5.0, 37.5, 100.0, 250.0)
    return [{"SIMILARITY_THRESHOLD": value, "TOP_EDGE_PERCENT": None} for value in thresholds] + [
        {"SIMILARITY_THRESHOLD": 0.0, "TOP_EDGE_PERCENT": value} for value in percents
    ]


class PrepareNetworkThresholdTests(unittest.TestCase):
    def settings(self, **overrides):
        values = {
            "NODE_FASTA_FILE": "", "ALIGNMENT_SCORE": "global", "NORM_MODE": "alignment_length",
            "SIMILARITY_THRESHOLD": 0.0, "TOP_EDGE_PERCENT": None, "UMAP_MODE": False,
            "UMAP_NEIGHBORS": 15,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def prepared(self, path, settings, selected):
        settings = SimpleNamespace(**vars(settings))
        output = io.StringIO()
        with h5py.File(path, "r") as network, redirect_stdout(output):
            result = prepare_network(network, settings=settings, selected_fasta_headers=selected)
        lines = [line for line in output.getvalue().splitlines() if line.startswith(("Top ", "Calculated "))]
        return result, settings.SIMILARITY_THRESHOLD, lines

    def assert_same_as_former(self, path, settings, selected, label):
        (headers, edges, scores), threshold, lines = self.prepared(path, settings, selected)
        expected_headers, expected_edges, expected_scores, expected_threshold, expected_lines = (
            former_threshold_prepare(path, settings, set(selected))
        )
        self.assertEqual(headers, expected_headers, label)
        self.assertEqual(edges.dtype, expected_edges.dtype, label)
        self.assertEqual(edges.shape, expected_edges.shape, label)
        np.testing.assert_array_equal(edges, expected_edges, label)
        self.assertEqual(scores.dtype, expected_scores.dtype, label)
        np.testing.assert_array_equal(scores, expected_scores, label)
        if expected_threshold is None:
            self.assertIsNone(threshold, label)
        else:
            self.assertTrue(same_number(float(threshold), float(expected_threshold)), label)
        self.assertEqual(lines, expected_lines, label)

    def test_every_network_setting_and_filter_matches_the_former_output(self):
        node_count = 24
        everyone = [f"N{index}" for index in range(node_count)]
        subset = [f"N{index}" for index in range(1, node_count, 2)]
        filters = thresholds_and_percents()
        with tempfile.TemporaryDirectory() as folder:
            networks = {}
            for label, builder in {
                "distinct": dict(),
                "ties": dict(ties=True),
                "ties and NaN, random pairs": dict(ties=True, nan=True, pair_count=500),
                "distinct and NaN": dict(nan=True),
            }.items():
                path = pathlib.Path(folder) / f"{label}.h5"
                write_network(path, node_count, np.random.default_rng(len(networks)), **builder)
                networks[label] = (path, [("global", "alignment_length"), ("local", "average_sequence")])
            for label, builder in {
                "e-value": dict(ties=True), "e-value float64 NaN": dict(ties=True, nan=True, score_dtype=np.float64),
                "e-value int64": dict(ties=True, score_dtype=np.int64),
                "e-value distinct": dict(),
            }.items():
                path = pathlib.Path(folder) / f"{label}.h5"
                write_network(path, node_count, np.random.default_rng(10 + len(networks)), blast=True, **builder)
                networks[label] = (path, [("global", "alignment_length")])
            cases = 0
            for label, (path, modes) in networks.items():
                for (score, norm), (selection, selected) in itertools.product(
                    modes, (("every node", everyone), ("a subset", subset))
                ):
                    for chosen in filters:
                        settings = self.settings(ALIGNMENT_SCORE=score, NORM_MODE=norm, **chosen)
                        for chunk, workers in ((10_000, 1), (13, 3)):
                            description = (f"{label}, {score}/{norm}, {selection}, {chosen}, "
                                           f"chunk {chunk}, workers {workers}")
                            with mock.patch.object(scan, "CHUNK_EDGES", chunk), mock.patch.object(
                                threshold_edges, "DEFAULT_MAX_WORKERS", workers
                            ):
                                self.assert_same_as_former(path, settings, selected, description)
                            cases += 1
            self.assertGreater(cases, 500)

    def test_many_random_networks_match_the_former_output(self):
        for seed in range(40):
            rng = np.random.default_rng(1000 + seed)
            node_count = int(rng.integers(4, 30))
            blast = bool(rng.integers(0, 2))
            with tempfile.TemporaryDirectory() as folder:
                path = pathlib.Path(folder) / "network.h5"
                write_network(
                    path, node_count, rng, blast=blast, ties=bool(rng.integers(0, 2)),
                    nan=bool(rng.integers(0, 4) == 0),
                    pair_count=None if rng.integers(0, 2) else int(rng.integers(1, 300)),
                    score_dtype=np.float64 if blast and rng.integers(0, 2) else np.float32,
                )
                names = [f"N{index}" for index in range(node_count)]
                selected = [name for name in names if rng.random() < 0.7] or names[:1]
                for _ in range(6):
                    if rng.integers(0, 2):
                        chosen = {"SIMILARITY_THRESHOLD": float(rng.choice([-1.0, 0.0, 1.0, 2.0, 10.0, 25.0, 50.0])),
                                  "TOP_EDGE_PERCENT": None}
                    else:
                        chosen = {"SIMILARITY_THRESHOLD": 0.0,
                                  "TOP_EDGE_PERCENT": float(rng.choice([0.01, 1.0, 12.5, 50.0, 100.0, 400.0]))}
                    settings = self.settings(
                        ALIGNMENT_SCORE=str(rng.choice(["global", "local"])),
                        NORM_MODE=str(rng.choice(["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"])),
                        **chosen,
                    )
                    chunk, workers = int(rng.choice([7, 50, 10_000])), int(rng.choice([1, 2, 4]))
                    with self.subTest(seed=seed, blast=blast, settings=vars(settings), chunk=chunk):
                        with mock.patch.object(scan, "CHUNK_EDGES", chunk), mock.patch.object(
                            threshold_edges, "DEFAULT_MAX_WORKERS", workers
                        ):
                            self.assert_same_as_former(path, settings, selected, f"seed {seed}")

    def test_the_kept_edges_keep_the_stored_pair_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            columns = write_network(path, 20, np.random.default_rng(5), blast=True)
            names = [f"N{index}" for index in range(20)]
            with mock.patch.object(scan, "CHUNK_EDGES", 17):
                (_, edges, scores), _, _ = self.prepared(
                    path, self.settings(SIMILARITY_THRESHOLD=100.0), names
                )
        stored = {(int(i), int(j)): index for index, (i, j) in enumerate(zip(columns["i"], columns["j"]))}
        positions = [stored[(int(i), int(j))] for i, j in edges]
        self.assertEqual(positions, sorted(positions))
        np.testing.assert_array_equal(scores, columns["score"][positions])
        self.assertTrue(np.all(scores >= 100.0))
        self.assertEqual(len(scores), int(np.count_nonzero(columns["score"] >= 100.0)))

    def test_ties_at_the_cutoff_are_all_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            columns = write_network(path, 20, np.random.default_rng(6), blast=True, ties=True)
            names = [f"N{index}" for index in range(20)]
            settings = self.settings(TOP_EDGE_PERCENT=10.0)
            (_, edges, scores), threshold, lines = self.prepared(path, settings, names)
        requested = int(190 * 0.10)
        self.assertEqual(threshold, float(np.sort(columns["score"])[::-1][requested - 1]))
        self.assertGreater(len(scores), requested)
        self.assertEqual(lines[0], "Top 10.0% Edges Requested (based on max possible 190 edges).")
        self.assertEqual(lines[1], f"Calculated E-Value Cutoff: {threshold:.5f}")

    def test_a_network_without_pairs_or_with_none_left_has_a_zero_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            empty = pathlib.Path(folder) / "empty.h5"
            with h5py.File(empty, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B", b"C"])
                for name, dtype in (("i", np.int32), ("j", np.int32), ("score", np.float32)):
                    network.create_dataset(name, data=np.asarray([], dtype=dtype))
            for selected in (["A", "B", "C"], ["A"]):
                for chosen in ({"TOP_EDGE_PERCENT": 50.0}, {"SIMILARITY_THRESHOLD": 1.0}):
                    with self.subTest(selected=selected, chosen=chosen):
                        settings = self.settings(**chosen)
                        self.assert_same_as_former(empty, settings, selected, "empty")
                        (headers, edges, scores), threshold, lines = self.prepared(empty, settings, selected)
                        self.assertEqual((edges.shape, edges.dtype, scores.dtype), ((0, 2), np.int32, np.float32))
                        if "TOP_EDGE_PERCENT" in chosen:
                            self.assertEqual(threshold, 0.0)
                            self.assertEqual(lines[1], "Calculated E-Value Cutoff: 0.00000")
            # Pairs exist, but the subset removes them all.
            path = pathlib.Path(folder) / "pairs.h5"
            write_network(path, 6, np.random.default_rng(7), blast=True)
            for chosen in ({"TOP_EDGE_PERCENT": 50.0}, {"SIMILARITY_THRESHOLD": -1.0}):
                settings = self.settings(**chosen)
                self.assert_same_as_former(path, settings, ["N0"], "no pair left")

    def test_a_pair_index_past_the_header_count_is_refused_before_anything_is_printed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            with h5py.File(path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([0, 0], dtype=np.uint16))
                network.create_dataset("j", data=np.asarray([1, 2], dtype=np.uint16))
                network.create_dataset("score", data=np.asarray([8.0, 2.0], dtype=np.float32))
            for selected in (["A", "B"], ["A"]):
                for chosen in ({"SIMILARITY_THRESHOLD": 3.0}, {"TOP_EDGE_PERCENT": 50.0}):
                    with self.subTest(selected=selected, chosen=chosen):
                        output = io.StringIO()
                        with h5py.File(path, "r") as network, redirect_stdout(output):
                            with self.assertRaises(IndexError) as raised:
                                prepare_network(
                                    network, settings=self.settings(**chosen), selected_fasta_headers=selected
                                )
                        if selected == ["A", "B"]:
                            self.assertEqual(str(raised.exception), "Network pair indices exceed the header count.")
                        self.assertNotIn("Top 50", output.getvalue())
                        self.assertNotIn("Calculated", output.getvalue())

    def test_a_threshold_that_is_none_fails_as_before_unless_there_is_nothing_to_compare(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            write_network(path, 6, np.random.default_rng(8), blast=True)
            names = [f"N{index}" for index in range(6)]
            settings = self.settings(SIMILARITY_THRESHOLD=None)
            with self.assertRaises(TypeError):
                former_threshold_prepare(path, settings, set(names))
            with self.assertRaises(TypeError):
                self.prepared(path, settings, names)
            # A subset that leaves no pair compares nothing, with no error either way.
            self.assert_same_as_former(path, settings, ["N0"], "no pair left")

    def test_the_streamed_module_needs_neither_numba_nor_qt(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from utilities import Network_Threshold_Edges; "
            "assert 'numba' not in sys.modules and 'PySide6' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(SRC)], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class MemoryTests(unittest.TestCase):
    def test_the_peak_stays_a_few_slices_for_a_threshold_and_a_top_percent(self):
        # Loading the columns took 15 to 28 bytes per pair. 20,000-pair slices on two
        # threads hold a few bytes per pair of this 1M-pair network, the 1% of its edges
        # that are kept included (those are the result, 12 bytes each).
        node_count = 1415
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "network.h5"
            columns = write_network(path, node_count, np.random.default_rng(9), blast=True)
            pair_count = len(columns["i"])
            names = [f"N{index}" for index in range(node_count)]
            for chosen in ({"SIMILARITY_THRESHOLD": float(pair_count * 0.99)},
                           {"TOP_EDGE_PERCENT": 1.0}):
                values = {
                    "NODE_FASTA_FILE": "", "ALIGNMENT_SCORE": "global", "NORM_MODE": "alignment_length",
                    "SIMILARITY_THRESHOLD": 0.0, "TOP_EDGE_PERCENT": None, "UMAP_MODE": False, **chosen,
                }
                with self.subTest(**chosen):
                    with h5py.File(path, "r") as network, redirect_stdout(io.StringIO()):
                        with mock.patch.object(scan, "CHUNK_EDGES", 20_000), mock.patch.object(
                            threshold_edges, "DEFAULT_MAX_WORKERS", 2
                        ):
                            tracemalloc.start()
                            try:
                                _, edges, _ = prepare_network(
                                    network, settings=SimpleNamespace(**values),
                                    selected_fasta_headers=names,
                                )
                                _, peak = tracemalloc.get_traced_memory()
                            finally:
                                tracemalloc.stop()
                    self.assertGreater(len(edges), 0)
                    self.assertLess(peak / pair_count, 6.0)


if __name__ == "__main__":
    unittest.main()
