"""Shared Embedding_MSA fixtures: a small embedding database and network on disk.

The Embedding_MSA test modules (memory, merge, configuration) build networks
with NetworkFixture and point run_msa_builder at them with builder_settings.
InProcessPool replaces multiprocessing.Pool so bootstrap trees run in this
process, GuideTreeBuilt stops a run once its guide tree exists, and
ArrayAssertions compares arrays bit for bit.
"""

import pathlib
import sys

import h5py
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA  # noqa: E402
from utilities.HDF5_Storage import create_metadata_first_file, mark_generation_complete  # noqa: E402


RESIDUES = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
MODEL = "test_model"
EMBED_NAME = f"Fixture_[{MODEL}]_embeddings.h5"
NETWORK_NAME = f"Fixture_[{MODEL}]_network.h5"
UPGMA = "UPGMA (Fast)"


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

    def add_calibration_sample(self, count=12, pooling="max", length_power=2.0, weight=3.0):
        """Record count of the network's listed pairs as an Edge Pre-filtering calibration sample.

        The pairs are distinct, in range and between sequences that have
        embeddings, so a run finds every one. Returns the record written.
        """
        from utilities import Prefilter_Calibration

        node_count = len(self.net_headers)
        embedded = set(self.emb_headers)
        rows, cols, seen = [], [], set()
        for a, b in zip(self.arr_i.tolist(), self.arr_j.tolist()):
            if a == b or max(a, b) >= node_count:
                continue
            if self.net_headers[a] not in embedded or self.net_headers[b] not in embedded:
                continue
            pair = (min(a, b), max(a, b))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append(pair[0])
            cols.append(pair[1])
            if len(rows) == count:
                break
        record = Prefilter_Calibration.PrefilterRecord(
            strength=20.0, threshold=0.5, pooling=pooling, length_power=length_power,
            kept=int(self.arr_i.size) - len(rows), skipped=len(rows) * int(weight),
            sample=Prefilter_Calibration.CalibrationSample(
                np.array(rows), np.array(cols), np.full(len(rows), float(weight))
            ),
        )
        with h5py.File(self.network_dir / NETWORK_NAME, "a") as network:
            Prefilter_Calibration.write_record(network, record, np.uint16)
        return record


class InProcessPool:
    """Runs bootstrap trees in this process.

    imap gives them in seed order, as the real pool's ordered call does, so the
    consensus sum is reproducible, and test patches reach the workers.
    imap_unordered gives them in the reverse order, as a real pool may.
    """

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def imap(self, func, iterable):
        return map(func, iterable)

    def imap_unordered(self, func, iterable):
        # The other order a real pool may give, so a caller that needs seed order must not use this.
        return map(func, list(iterable)[::-1])


class GuideTreeBuilt(Exception):
    """Stops run_msa_builder once the guide tree exists."""


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
        "CALIBRATED_IMPUTATION": True,
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


class ArrayAssertions:
    def assertSameArray(self, actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(), expected.tobytes())
