"""desktop.Viewer_State.prepare_network: scores, normalization, FASTA subsets and
the columns it reads, with its memory peak kept low."""

import io
import pathlib
import sys
import tempfile
import tracemalloc
import unittest
import warnings
from contextlib import redirect_stdout
from types import SimpleNamespace

import h5py
import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from desktop.Viewer_State import prepare_network


NORMALIZATIONS = (
    "alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"
)
PAIR_COLUMNS = {"i", "j", "score", "g_score", "g_len", "l_score", "l_len"}


def _every_header(network):
    """Every node's header: a FASTA selection that keeps the whole network."""
    return [
        header.decode("utf-8") if isinstance(header, bytes) else header
        for header in network["headers"][:]
    ]


def _settings(**overrides):
    values = {
        "NODE_FASTA_FILE": "",
        "ALIGNMENT_SCORE": "global",
        "NORM_MODE": "alignment_length",
        "SIMILARITY_THRESHOLD": -np.inf,
        "TOP_EDGE_PERCENT": None,
        "UMAP_MODE": False,
        "UMAP_NEIGHBORS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_alignment_network(path, node_count, rng):
    """Write a complete network with the dtypes real alignment networks use.

    Each column holds different values, so reading the wrong score type's
    column changes the result.
    """
    sources, targets = np.triu_indices(node_count, k=1)
    pair_count = len(sources)
    columns = {
        "i": sources.astype(np.uint16),
        "j": targets.astype(np.uint16),
        "seq_lens": rng.integers(0, 6, node_count).astype(np.uint16),
    }
    for kind in ("g", "l"):
        columns[f"{kind}_score"] = rng.normal(5.0, 4.0, pair_count).astype(np.float32)
        # Zero lengths exercise the pairs that score 0 instead of dividing.
        columns[f"{kind}_len"] = rng.integers(0, 6, pair_count).astype(np.uint16)
        columns[f"{kind}_len"][0] = 0
    columns["seq_lens"][0] = 0
    with h5py.File(path, "w") as network:
        network.attrs["model_name"] = "model"
        network.create_dataset(
            "headers", data=[f"N{index}".encode() for index in range(node_count)]
        )
        for name, values in columns.items():
            network.create_dataset(name, data=values)
    return columns


def _former_scores(columns, settings):
    """Normalize every pair with the formula prepare_network used to apply."""
    kind = "g" if settings.ALIGNMENT_SCORE == "global" else "l"
    raw = columns[f"{kind}_score"]
    source_lengths = columns["seq_lens"][columns["i"]]
    target_lengths = columns["seq_lens"][columns["j"]]
    denominator = {
        "alignment_length": columns[f"{kind}_len"],
        "shorter_sequence": np.minimum(source_lengths, target_lengths),
        "longer_sequence": np.maximum(source_lengths, target_lengths),
        "average_sequence": (source_lengths + target_lengths) / 2.0,
    }[settings.NORM_MODE]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, raw / denominator, 0.0)


class _ReadRecorder:
    """An open network that records which datasets prepare_network indexes."""

    def __init__(self, network):
        self.attrs = network.attrs
        self._network = network
        self.indexed = set()

    def __contains__(self, name):
        return name in self._network

    def __getitem__(self, name):
        self.indexed.add(name)
        return self._network[name]


class ScoreEquivalenceTests(unittest.TestCase):
    def test_scores_and_dtypes_match_the_former_formula(self):
        rng = np.random.default_rng(11)
        node_count = 30
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            columns = _write_alignment_network(network_path, node_count, rng)
            kept_nodes = np.arange(0, node_count, 2)
            in_subset = np.isin(columns["i"], kept_nodes) & np.isin(columns["j"], kept_nodes)
            for score in ("global", "local"):
                for normalization in NORMALIZATIONS:
                    for subset in (False, True):
                        settings = _settings(ALIGNMENT_SCORE=score, NORM_MODE=normalization)
                        expected = _former_scores(columns, settings)
                        selected = [f"N{index}" for index in range(node_count)]
                        if subset:
                            selected = [f"N{index}" for index in kept_nodes]
                            expected = expected[in_subset]
                        with self.subTest(score=score, normalization=normalization, subset=subset):
                            with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
                                _, edges, scores = prepare_network(
                                    network, settings=settings, selected_fasta_headers=selected
                                )
                            self.assertEqual(scores.dtype, expected.dtype)
                            np.testing.assert_array_equal(scores, expected)
                            self.assertEqual(len(edges), len(expected))

    def test_pairs_without_a_positive_denominator_warn_nothing(self):
        # Those pairs are never divided, so neither a divide-by-zero
        # RuntimeWarning nor NumPy's warning about where= leaving output
        # uninitialized reaches the Viewer console.
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            _write_alignment_network(network_path, 12, np.random.default_rng(13))
            for normalization in NORMALIZATIONS:
                settings = _settings(NORM_MODE=normalization)
                with self.subTest(normalization=normalization), h5py.File(
                    network_path, "r"
                ) as network, redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("error")
                    prepare_network(
                        network, settings=settings, selected_fasta_headers=_every_header(network)
                    )


class ColumnReadTests(unittest.TestCase):
    def assert_pair_columns_read(self, network_path, settings, expected):
        with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
            recorder = _ReadRecorder(network)
            prepare_network(
                recorder, settings=settings, selected_fasta_headers=_every_header(network)
            )
        self.assertEqual(recorder.indexed & PAIR_COLUMNS, expected)

    def test_alignment_networks_skip_the_unused_score_and_length_columns(self):
        expected = {
            ("global", "alignment_length"): {"i", "j", "g_score", "g_len"},
            ("local", "alignment_length"): {"i", "j", "l_score", "l_len"},
        }
        for score, kind in (("global", "g"), ("local", "l")):
            for normalization in NORMALIZATIONS[1:]:
                expected[score, normalization] = {"i", "j", f"{kind}_score"}
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            _write_alignment_network(network_path, 6, np.random.default_rng(5))
            for (score, normalization), columns in expected.items():
                with self.subTest(score=score, normalization=normalization):
                    self.assert_pair_columns_read(
                        network_path,
                        _settings(ALIGNMENT_SCORE=score, NORM_MODE=normalization),
                        columns,
                    )

    def test_blast_networks_read_pairs_and_scores_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "blast.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([0], dtype=np.uint32))
                network.create_dataset("j", data=np.asarray([1], dtype=np.uint32))
                network.create_dataset("score", data=np.asarray([7.0], dtype=np.float32))
            self.assert_pair_columns_read(network_path, _settings(), {"i", "j", "score"})


class PeakMemoryTests(unittest.TestCase):
    # Bytes per pair for a complete network (uint16 indices, float32 scores,
    # uint16 lengths) with every node kept. Normalization is the peak: i and
    # j (2 + 2), the raw scores (4), the denominator (the alignment length, 2,
    # or both sequence lengths and their minimum, 2 + 2 + 2), the positive
    # mask (1) and the scores (4) make 15 and 19 bytes. The former np.where
    # also held a second score array (4) and, for sequence normalizations,
    # the unused length column (2): 19 and 25 bytes.
    BUDGETS = {
        ("global", "alignment_length"): 16.0,
        ("local", "shorter_sequence"): 20.0,
    }

    def test_normalization_peak_stays_within_budget(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            columns = _write_alignment_network(network_path, 1415, np.random.default_rng(7))
            pair_count = len(columns["i"])
            for (score, normalization), budget in self.BUDGETS.items():
                settings = _settings(
                    ALIGNMENT_SCORE=score, NORM_MODE=normalization, TOP_EDGE_PERCENT=5.0
                )
                with self.subTest(score=score, normalization=normalization):
                    with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
                        every_header = _every_header(network)
                        tracemalloc.start()
                        try:
                            prepare_network(
                                network, settings=settings, selected_fasta_headers=every_header
                            )
                            _, peak = tracemalloc.get_traced_memory()
                        finally:
                            tracemalloc.stop()
                    self.assertLess(peak / pair_count, budget)


def _preparation_settings(**overrides):
    values = {
        "NODE_FASTA_FILE": "",
        "ALIGNMENT_SCORE": "global",
        "NORM_MODE": "alignment_length",
        "SIMILARITY_THRESHOLD": 0.0,
        "TOP_EDGE_PERCENT": None,
        "UMAP_MODE": False,
        "UMAP_NEIGHBORS": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class NetworkPreparationTests(unittest.TestCase):
    def test_alignment_normalization_modes_and_fasta_subset(self):
        expected_scores = {
            "alignment_length": 4.0,
            "shorter_sequence": 4.0,
            "longer_sequence": 2.0,
            "average_sequence": 8.0 / 3.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A", b"B", b"C"])
                network.create_dataset("i", data=[0, 0, 1])
                network.create_dataset("j", data=[1, 2, 2])
                network.create_dataset("seq_lens", data=[2, 4, 8])
                for name in ("g_score", "l_score"):
                    network.create_dataset(name, data=[8.0, 8.0, 8.0])
                for name in ("g_len", "l_len"):
                    network.create_dataset(name, data=[2, 2, 2])

            for normalization, expected_score in expected_scores.items():
                with self.subTest(normalization=normalization), h5py.File(
                    network_path, "r"
                ) as network:
                    settings = _preparation_settings(NORM_MODE=normalization)
                    headers, edges, scores = prepare_network(
                        network,
                        settings=settings,
                        selected_fasta_headers=["A", "B"],
                    )
                    self.assertEqual(headers, ["A", "B"])
                    np.testing.assert_array_equal(edges, [[0, 1]])
                    np.testing.assert_allclose(scores, [expected_score])
                    self.assertFalse(settings.INPUT_IS_EVALUE)

    def test_top_percent_updates_effective_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A", b"B", b"C"])
                network.create_dataset("i", data=[0, 0, 1])
                network.create_dataset("j", data=[1, 2, 2])
                network.create_dataset("seq_lens", data=[2, 2, 2])
                for name in ("g_score", "l_score"):
                    network.create_dataset(name, data=[2.0, 6.0, 4.0])
                for name in ("g_len", "l_len"):
                    network.create_dataset(name, data=[2, 2, 2])

            settings = _preparation_settings(TOP_EDGE_PERCENT=34.0)
            with h5py.File(network_path, "r") as network:
                headers, edges, scores = prepare_network(
                    network,
                    settings=settings,
                    selected_fasta_headers=["A", "B", "C"],
                )

            self.assertEqual(headers, ["A", "B", "C"])
            self.assertEqual(settings.SIMILARITY_THRESHOLD, 3.0)
            np.testing.assert_array_equal(edges, [[0, 2]])
            np.testing.assert_allclose(scores, [3.0])

    def test_top_percent_cutoff_matches_descending_sort(self):
        # The cutoff selects the edge_count-th largest score without sorting;
        # it must equal the former np.sort(scores)[::-1][edge_count - 1],
        # including ties and NaN scores, with every node kept or a subset.
        rng = np.random.default_rng(3)
        node_count = 40
        pairs = np.array([(i, j) for i in range(node_count) for j in range(i + 1, node_count)])
        headers = [f"N{index}".encode() for index in range(node_count)]
        for trial in range(16):
            if trial % 4 == 3:  # Distinct scores expose an off-by-one rank.
                raw = rng.permutation(len(pairs)).astype(np.float32)
            else:
                raw = rng.integers(0, 7, len(pairs)).astype(np.float32)
            if trial % 3 == 0:
                raw[rng.integers(0, len(raw), 25)] = np.nan
            selected = [f"N{index}" for index in range(0, node_count, 1 if trial % 2 else 3)]
            percent = float(rng.choice([0.5, 5.0, 37.5, 100.0]))
            with self.subTest(trial=trial), tempfile.TemporaryDirectory() as temp_dir:
                network_path = pathlib.Path(temp_dir) / "alignment.h5"
                with h5py.File(network_path, "w") as network:
                    network.attrs["model_name"] = "model"
                    network.create_dataset("headers", data=headers)
                    network.create_dataset("i", data=pairs[:, 0].astype(np.uint16))
                    network.create_dataset("j", data=pairs[:, 1].astype(np.uint16))
                    network.create_dataset("seq_lens", data=np.full(node_count, 9, np.uint16))
                    for name in ("g_score", "l_score"):
                        network.create_dataset(name, data=raw)
                    for name in ("g_len", "l_len"):
                        network.create_dataset(name, data=np.full(len(pairs), 2, np.uint16))
                settings = _preparation_settings(TOP_EDGE_PERCENT=percent)
                with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
                    prepare_network(network, settings=settings, selected_fasta_headers=selected)
                kept = np.arange(0, node_count, 1 if trial % 2 else 3)
                in_subset = np.isin(pairs[:, 0], kept) & np.isin(pairs[:, 1], kept)
                normalized = raw[in_subset] / np.float32(2)
                edge_count = int(len(kept) * (len(kept) - 1) / 2.0 * (percent / 100.0))
                edge_count = max(1, min(edge_count, len(normalized)))
                expected = float(np.sort(normalized)[::-1][edge_count - 1])
                if np.isnan(expected):
                    self.assertTrue(np.isnan(settings.SIMILARITY_THRESHOLD))
                else:
                    self.assertEqual(settings.SIMILARITY_THRESHOLD, expected)

    def test_pair_indices_beyond_headers_still_fail_when_every_node_is_kept(self):
        # The out-of-range pair scores below the threshold, so only an
        # up-front bounds check can reject it; filtering would drop it silently.
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "alignment.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "model"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=[0, 0])
                network.create_dataset("j", data=[1, 2])
                network.create_dataset("seq_lens", data=[2, 2])
                for name in ("g_score", "l_score"):
                    network.create_dataset(name, data=[8.0, 2.0])
                for name in ("g_len", "l_len"):
                    network.create_dataset(name, data=[2, 2])
            settings = _preparation_settings(SIMILARITY_THRESHOLD=3.0)
            with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
                with self.assertRaises(IndexError):
                    prepare_network(
                        network, settings=settings, selected_fasta_headers=_every_header(network)
                    )

    def test_empty_blast_network_returns_empty_connectivity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            network_path = pathlib.Path(temp_dir) / "blast.h5"
            with h5py.File(network_path, "w") as network:
                network.attrs["model_name"] = "blast"
                network.create_dataset("headers", data=[b"A", b"B"])
                network.create_dataset("i", data=np.asarray([], dtype=np.int32))
                network.create_dataset("j", data=np.asarray([], dtype=np.int32))
                network.create_dataset("score", data=np.asarray([], dtype=np.float32))

            settings = _preparation_settings()
            with h5py.File(network_path, "r") as network:
                headers, edges, scores = prepare_network(
                    network,
                    settings=settings,
                    selected_fasta_headers=_every_header(network),
                )

            self.assertEqual(headers, ["A", "B"])
            self.assertEqual(edges.shape, (0, 2))
            self.assertEqual(scores.shape, (0,))
            self.assertTrue(settings.INPUT_IS_EVALUE)


if __name__ == "__main__":
    unittest.main()
