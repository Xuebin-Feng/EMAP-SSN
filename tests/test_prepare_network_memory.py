"""prepare_network reads only the columns it uses and keeps its peak low."""

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
                        selected = None
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
                    prepare_network(network, settings=settings, selected_fasta_headers=None)


class ColumnReadTests(unittest.TestCase):
    def assert_pair_columns_read(self, network_path, settings, expected):
        with h5py.File(network_path, "r") as network, redirect_stdout(io.StringIO()):
            recorder = _ReadRecorder(network)
            prepare_network(recorder, settings=settings, selected_fasta_headers=None)
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
                        tracemalloc.start()
                        try:
                            prepare_network(network, settings=settings, selected_fasta_headers=None)
                            _, peak = tracemalloc.get_traced_memory()
                        finally:
                            tracemalloc.stop()
                    self.assertLess(peak / pair_count, budget)


if __name__ == "__main__":
    unittest.main()
