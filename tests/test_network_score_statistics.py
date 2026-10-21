# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The chunked network statistics give the numbers the whole-column code gave, in bounded memory.

legacy_statistics_report and legacy_summarize_network below restate, frozen, what the Config
window's "Compute Network Statistics" and the MCP network_statistics action computed when they
read whole columns (commit 85bb43e). Every figure the chunked code reports must equal theirs,
whatever the chunk size and the number of worker threads, on networks built to trip the
differences between the two: scores sitting on the 0.1 threshold grid and one float32 step
either side of it, negative scores, uint32 indices, FASTA subsets and BLAST networks.
The mean is the one figure allowed to differ, and only in its last digits (a float64 running
sum against numpy's float32 pairwise sum).
"""

import contextlib
import io
import itertools
import math
import os
import pathlib
import sys
import tempfile
import time
import tracemalloc
import unittest
from unittest import mock

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest as cache_manifest  # noqa: E402
from mcp_server.pipeline import Network_Statistics as mcp_statistics  # noqa: E402
from utilities import Network_Score_Statistics as stats  # noqa: E402


# --- The code as it was, frozen ---------------------------------------------------------


def legacy_statistics_report(network_path, fasta_path, score_mode, norm_mode, keep_scores=False):
    """ConfigGUI.run_statistics' numbers, from whole columns, as at 85bb43e. None: no edges.

    keep_scores adds the sorted scores under "_sorted_scores", for tests of rows the old code made but the new one caps.
    """
    from Bio import SeqIO

    with h5py.File(network_path, "r") as hf:
        metadata = cache_manifest.validate_network_schema(hf)
        is_blast = metadata.network_type == "blast"
        kept_mask = None
        if fasta_path and os.path.exists(fasta_path):
            fasta_ids = set()
            fasta_headers = set()
            with open(fasta_path) as handle:
                for rec in SeqIO.parse(handle, "fasta"):
                    fasta_ids.add(rec.id)
                    fasta_headers.add(rec.description)
            raw_headers = hf["headers"][:]
            headers = [h.decode("utf-8") if isinstance(h, bytes) else h for h in raw_headers]
            net_headers_set = set(headers)
            net_id_set = {h.split()[0] for h in headers}
            missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
            if missing_nodes:
                raise ValueError(f"missing {len(missing_nodes)}")
            valid_indices = []
            for i, h in enumerate(headers):
                rec_id = h.split()[0]
                if h in fasta_headers or rec_id in fasta_ids:
                    valid_indices.append(i)
            if len(valid_indices) < len(headers):
                kept_mask = np.zeros(len(headers), dtype=bool)
                kept_mask[valid_indices] = True

        sources = hf["i"][:]
        targets = hf["j"][:]
        if is_blast:
            raw_scores = hf["score"][:]
        else:
            if score_mode == "local":
                raw_scores = hf["l_score"][:].astype(np.float32, copy=False)
                length_name = "l_len"
            else:
                raw_scores = hf["g_score"][:].astype(np.float32, copy=False)
                length_name = "g_len"
            align_lens = None
            if norm_mode not in ("shorter_sequence", "longer_sequence", "average_sequence"):
                align_lens = hf[length_name][:].astype(np.float32)
            if "seq_lens" in hf:
                seq_lens = hf["seq_lens"][:]
            else:
                seq_lens = np.ones(int(max(sources.max(), targets.max())) + 1)

        if kept_mask is not None:
            valid_edges_mask = kept_mask[sources] & kept_mask[targets]
            raw_scores = raw_scores[valid_edges_mask]
            sources = sources[valid_edges_mask]
            targets = targets[valid_edges_mask]
            if not is_blast and align_lens is not None:
                align_lens = align_lens[valid_edges_mask]

        if is_blast:
            scores = raw_scores.astype(np.float32)
        else:
            epsilon = 1e-6
            if norm_mode == "alignment_length":
                denom = align_lens
            else:
                len_src = seq_lens[sources].astype(np.float32)
                len_dst = seq_lens[targets].astype(np.float32)
                if norm_mode == "shorter_sequence": denom = np.minimum(len_src, len_dst)
                elif norm_mode == "longer_sequence": denom = np.maximum(len_src, len_dst)
                elif norm_mode == "average_sequence": denom = (len_src + len_dst) / 2.0
                else: denom = align_lens
            denom = np.maximum(denom, epsilon)
            scores = (raw_scores / denom).astype(np.float32)

    if len(scores) == 0:
        return None
    max_score = np.max(scores)
    min_score = np.min(scores)
    avg_score = np.mean(scores)
    sorted_scores = np.sort(scores)
    stored_edges = len(sorted_scores)
    total_nodes = np.sum(kept_mask) if kept_mask is not None else len(headers)
    theoretical_max_edges = (total_nodes * (total_nodes - 1)) / 2.0
    if is_blast:
        start_val = int(math.floor(min_score))
        limit_step_1 = min(100, int(math.ceil(max_score)))
        thresh_low = np.arange(start_val, limit_step_1 + 1, 1)
        if max_score > 100:
            thresh_high = np.arange(105, int(math.ceil(max_score)) + 5, 5)
            thresholds = np.concatenate([thresh_low, thresh_high])
        else:
            thresholds = thresh_low
    else:
        start_val = math.floor(min_score * 10) / 10.0
        thresholds = np.arange(start_val, max_score + 0.1, 0.1)
    indices = np.searchsorted(sorted_scores, thresholds, side="left")
    counts = stored_edges - indices
    rows = []
    for thresh, count in zip(thresholds, counts):
        pct = (count / theoretical_max_edges) * 100.0 if theoretical_max_edges > 0 else 0
        value = f"{int(thresh)}" if is_blast else f"{thresh:.1f}"
        rows.append((value, f"{count}", f"{pct:.2f}%"))
    report = {
        "model": metadata.model_name,
        "fasta": os.path.basename(fasta_path),
        "nodes": int(total_nodes),
        "blast": is_blast,
        "score_mode": score_mode,
        "normalization": norm_mode,
        "stored_edges": int(stored_edges),
        "possible_edges": int(theoretical_max_edges),
        "maximum": float(max_score),
        "minimum": float(min_score),
        "mean": float(avg_score),
        "rows": rows,
    }
    if keep_scores:
        report["_sorted_scores"] = sorted_scores
    return report


def new_statistics_report(network_path, fasta_path, score_mode, norm_mode, *, workers, chunk_edges):
    """The same report from the chunked code, assembled as the Config window assembles it."""
    with h5py.File(network_path, "r") as hf:
        metadata = cache_manifest.validate_network_schema(hf)
        is_blast = metadata.network_type == "blast"
        headers = [h.decode("utf-8") if isinstance(h, bytes) else h for h in hf["headers"][:]]
        ids, names = stats.read_fasta_names(fasta_path)
        mask, missing = stats.kept_node_mask(headers, ids, names)
        if missing:
            raise ValueError(f"missing {missing}")
        kept_mask = None if mask.all() else mask
        source = stats.ScoreSource(hf, is_blast=is_blast, score_mode=score_mode,
                                   norm_mode=norm_mode, kept_mask=kept_mask)
        result = stats.compute_score_statistics(source, workers=workers, chunk_edges=chunk_edges)
        if result.stored_edges == 0:
            return None
        return stats.statistics_report(
            result, model_name=metadata.model_name, fasta_name=os.path.basename(fasta_path),
            node_count=int(mask.sum()), is_blast=is_blast, score_mode=score_mode, norm_mode=norm_mode,
        )


# The MCP network_statistics report as it was: every pair read, scored, sorted and filtered at once
# (Network_Statistics.summarize_network at 85bb43e, which scored through prepare_network).


def legacy_top_percent_cutoff(scores_descending, node_count, percent):
    """The layout job's TOP_EDGE_PERCENT cutoff (prepare_network's rule)."""
    possible = node_count * (node_count - 1) / 2.0
    count = int(possible * (percent / 100.0))
    if len(scores_descending) == 0:
        return 0.0
    count = max(1, min(count, len(scores_descending)))
    return float(scores_descending[count - 1])


def legacy_summarize_network(network_path, fasta_path=None, alignment_score="global",
                      norm_mode="alignment_length", thresholds=(), top_edge_percents=None,
                      budget=40.0):
    """Score distribution and edge-filter table for the nodes a layout would keep."""
    import h5py
    import numpy as np
    from types import SimpleNamespace

    started = time.monotonic()
    from Cache_Manifest import validate_network_schema
    from desktop.Viewer_State import prepare_network
    from utilities.Sequence_Utils import load_sanitized_fasta

    network_path = os.fspath(network_path)
    if not os.path.isfile(network_path):
        raise mcp_statistics.NetworkStatisticsError(f"Network file was not found: {network_path}")
    metadata = validate_network_schema(network_path)
    is_blast = metadata.network_type == "blast"
    if not is_blast:
        if alignment_score not in ("global", "local"):
            raise mcp_statistics.NetworkStatisticsError("alignment_score must be global or local.")
        if norm_mode not in ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"):
            raise mcp_statistics.NetworkStatisticsError("norm_mode is not supported.")
        if alignment_score == "local" and norm_mode == "alignment_length":
            raise mcp_statistics.NetworkStatisticsError(
                "norm_mode alignment_length is unavailable for local scores; choose "
                "shorter_sequence, longer_sequence or average_sequence."
            )

    with h5py.File(network_path, "r") as data:
        raw_headers = data["headers"][:]
        network_headers = [h.decode("utf-8") if isinstance(h, bytes) else h for h in raw_headers]
        if fasta_path:
            if not os.path.isfile(fasta_path):
                raise mcp_statistics.NetworkStatisticsError(f"FASTA file was not found: {fasta_path}")
            selected, _sequences, _stats = load_sanitized_fasta(fasta_path)
        else:
            selected = network_headers
        settings = SimpleNamespace(
            ALIGNMENT_SCORE=alignment_score, NORM_MODE=norm_mode,
            SIMILARITY_THRESHOLD=-math.inf, TOP_EDGE_PERCENT=None, UMAP_MODE=False,
            NODE_FASTA_FILE=fasta_path or network_path,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            headers, edges, scores = prepare_network(
                data, settings=settings, selected_fasta_headers=selected,
            )
    node_count = len(headers)
    kept_set = set(headers)
    kept_ids = {header.split()[0] for header in headers if header.split()}
    fasta_missing = None
    if fasta_path:
        fasta_missing = sum(
            1 for header in selected
            if header not in kept_set and (not header.split() or header.split()[0] not in kept_ids)
        )
    scores = np.asarray(scores, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    possible = node_count * (node_count - 1) // 2
    metric = ("-log10(E-value); larger is more similar, capped at 300" if is_blast else
              f"{alignment_score} alignment score / {norm_mode.replace('_', ' ')}; larger is more similar")
    report = {
        "input_hdf5": network_path,
        "node_fasta_file": fasta_path,
        "network_type": metadata.network_type,
        "model_name": metadata.model_name,
        "metric": metric,
        "alignment_score": None if is_blast else alignment_score,
        "norm_mode": None if is_blast else norm_mode,
        "nodes": {
            "in_network": len(network_headers),
            "kept_for_layout": node_count,
            "fasta_records": len(selected) if fasta_path else None,
            "fasta_records_missing_from_network": fasta_missing,
        },
        "pairs": {
            "stored": int(len(scores)),
            "possible": int(possible),
            "stored_percent_of_possible": mcp_statistics._rounded(100.0 * len(scores) / possible) if possible else None,
        },
        "score_summary": None,
        "histogram": None,
        "edge_filters": [],
        "notes": [
            "Scores are computed exactly as start_layout_job computes them for these settings and nodes.",
            "SIMILARITY_THRESHOLD keeps edges with score >= threshold; TOP_EDGE_PERCENT keeps the top N percent "
            "of all possible pairs among the kept nodes (not of stored pairs), so its cutoff is the Nth-best score.",
            "clusters counts connected groups of two or more nodes; isolated_nodes have no kept edge and are packed "
            "around the clusters as single points.",
            "fasta_records counts the node FASTA after the standard cleanup, which merges identical sequences "
            "(inspect_file counts raw records and duplicate_sequences).",
            "UMAP and PaCMAP layouts ignore both filters and keep each node's UMAP_NEIGHBORS or "
            "PACMAP_NEIGHBORS strongest edges.",
        ],
        "complete": True,
    }
    if len(scores) == 0:
        report["notes"].append("The network has no edges between the kept nodes.")
        return report

    quantiles = np.quantile(scores, mcp_statistics.QUANTILES)
    report["score_summary"] = {
        "min": mcp_statistics._rounded(scores.min()), "max": mcp_statistics._rounded(scores.max()),
        "mean": mcp_statistics._rounded(scores.mean()),
        "quantiles": {f"p{int(q * 100)}": mcp_statistics._rounded(v) for q, v in zip(mcp_statistics.QUANTILES, quantiles)},
    }
    counts, bin_edges = np.histogram(scores, bins=20)
    report["histogram"] = {"bin_edges": [mcp_statistics._rounded(v) for v in bin_edges],
                           "counts": [int(v) for v in counts]}

    order = np.argsort(-scores, kind="stable")
    descending = scores[order]
    rows = []
    percents = list(mcp_statistics.DEFAULT_TOP_EDGE_PERCENTS if top_edge_percents is None else top_edge_percents)
    for percent in percents[:len(mcp_statistics.DEFAULT_TOP_EDGE_PERCENTS) + mcp_statistics.MAX_EXTRA_ROWS]:
        rows.append(("top_edge_percent", float(percent),
                     legacy_top_percent_cutoff(descending, node_count, float(percent))))
    for threshold in list(thresholds)[:mcp_statistics.MAX_EXTRA_ROWS]:
        rows.append(("similarity_threshold", float(threshold), float(threshold)))
    rows.sort(key=lambda row: -row[2])
    for mode, value, cutoff in rows:
        kept = int(np.searchsorted(-descending, -cutoff, side="right"))
        entry = {
            "filter": mode, "value": mcp_statistics._rounded(value),
            "similarity_threshold": mcp_statistics._rounded(cutoff),
            "kept_edges": kept,
            "kept_percent_of_possible": mcp_statistics._rounded(100.0 * kept / possible) if possible else None,
            "mean_degree": mcp_statistics._rounded(2.0 * kept / node_count) if node_count else None,
        }
        if kept > mcp_statistics.MAX_COMPONENT_EDGES:
            entry["connectivity"] = None
            entry["connectivity_skipped"] = f"more than {mcp_statistics.MAX_COMPONENT_EDGES} edges"
        elif time.monotonic() - started > budget:
            entry["connectivity"] = None
            entry["connectivity_skipped"] = "time budget reached"
            report["complete"] = False
        else:
            chosen = order[:kept]
            entry["connectivity"] = mcp_statistics.component_summary(node_count, edges[chosen, 0], edges[chosen, 1])
        report["edge_filters"].append(entry)
    return report


# --- Networks to compare on --------------------------------------------------------------


def write_network(path, headers, pairs, columns, *, blast=False, seq_lens=None, index_type=np.uint16):
    with h5py.File(path, "w") as hf:
        hf.attrs["model_name"] = "BLAST" if blast else "synthetic"
        hf.create_dataset("headers", data=np.asarray(headers, dtype=object), dtype=h5py.string_dtype("utf-8"))
        hf.create_dataset("i", data=np.asarray(pairs[0], dtype=index_type))
        hf.create_dataset("j", data=np.asarray(pairs[1], dtype=index_type))
        if blast:
            hf.create_dataset("score", data=np.asarray(columns["score"], dtype=np.float32))
            return
        hf.create_dataset("seq_lens", data=np.asarray(seq_lens, dtype=np.uint16))
        for name, kind in (("l_score", np.float32), ("l_len", np.uint16),
                           ("g_score", np.float32), ("g_len", np.uint16)):
            hf.create_dataset(name, data=np.asarray(columns[name], dtype=kind))


def unique_sequence(index):
    residues = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(residues[(index // 20 ** place) % 20] for place in range(6)) + "ACDE"


def write_fasta(path, headers, keep):
    with open(path, "w", encoding="utf-8") as handle:
        for index in keep:
            handle.write(f">{headers[index]}\n{unique_sequence(index)}\n")


def tie_scores(rng, low, high, copies):
    """float32 scores on, and one step either side of, j/10 for j in [low, high], shuffled."""
    values = []
    for j in range(low, high + 1):
        v = np.float32(j / 10.0)
        values += [v, np.nextafter(v, np.float32(np.inf)), np.nextafter(v, np.float32(-np.inf))]
    scores = np.repeat(np.array(values, dtype=np.float32), copies)
    rng.shuffle(scores)
    return scores


class Networks:
    """A folder of small networks, each with a FASTA of all its nodes and one of every other."""

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.cases = []
        rng = np.random.default_rng(42)
        # Alignment networks whose scores land on the threshold grid: g_len is 1, so score == ratio.
        for label, low, high in (("ties from 0.3", 3, 25), ("ties from 0.5", 5, 18), ("ties from 0.0", 0, 12),
                                 ("ties from -1.0", -10, 12), ("ties from -0.05", -1, 5)):
            ratios = tie_scores(rng, low, high, 5)
            self.add_alignment(label, rng, len(ratios), 60, ratios, np.ones(len(ratios)),
                               ratios[::-1].copy(), np.ones(len(ratios)), np.ones(60))
        count, nodes = 12_000, 150
        lengths = rng.integers(30, 1500, size=count)
        self.add_alignment(
            "random", rng, count, nodes, (rng.beta(2, 5, size=count) * 0.9 + 0.05) * lengths, lengths,
            rng.beta(3, 2.5, size=count) * 90, rng.integers(8, 800, size=count), rng.integers(30, 900, size=nodes),
        )
        count = 3_000
        lengths = rng.integers(30, 1500, size=count)
        self.add_alignment("spaced headers", rng, count, 80, rng.random(count) * lengths, lengths,
                           rng.random(count) * 90, rng.integers(8, 800, size=count), rng.integers(30, 900, size=80),
                           spaced=True)
        # Zero lengths: the Config statistics divide by at least 1e-6, the layout's scores are 0 there.
        # The raw scores are tiny, so that ratio stays within a few dozen and the report a few hundred rows.
        count = 2_000
        self.add_alignment("zero lengths", rng, count, 40, rng.random(count) * 1e-5, rng.integers(0, 40, size=count),
                           rng.random(count) * 1e-5, rng.integers(0, 30, size=count), rng.integers(0, 50, size=40))
        # Long sequences: the uint16 sums the layout's own average normalization overflows on.
        self.add_alignment(
            "long sequences", rng, 2_000, 40, rng.random(2_000) * 40000, rng.integers(1, 60000, size=2_000),
            rng.random(2_000) * 900, rng.integers(1, 800, size=2_000), rng.integers(20000, 60000, size=40),
        )
        # E-value networks: integers on the step grid, fractions, values above 100, negative values.
        score = np.concatenate([rng.gamma(1.2, 8.0, size=9_000).clip(0, 300),
                                rng.integers(0, 300, size=1_000).astype(float)]).astype(np.float32)
        rng.shuffle(score)
        self.add_blast("blast", rng, score, 120)
        self.add_blast("blast negative", rng, (score - 20.0).astype(np.float32), 120)
        self.add_blast("blast uint32", rng, score, 120, index_type=np.uint32)
        # Identical scores only: every count equals the stored edges at one threshold and none above.
        self.add_blast("blast constant", rng, np.full(500, 7.0, dtype=np.float32), 30)

    def pairs(self, rng, count, nodes):
        first = rng.integers(0, nodes - 1, size=count)
        return first, rng.integers(first + 1, nodes)

    def fastas(self, name, headers, names=None):
        everything = self.folder / f"{name}.all.fasta"
        subset = self.folder / f"{name}.subset.fasta"
        names = names or headers
        write_fasta(everything, names, range(len(headers)))
        write_fasta(subset, names, range(0, len(headers), 2))
        return everything, subset

    def add_alignment(self, name, rng, count, nodes, g_score, g_len, l_score, l_len, seq_lens, spaced=False):
        # spaced: network headers of several words, which the FASTA names by their first word only.
        headers = [f"n{index} description" if spaced else f"n{index}_description" for index in range(nodes)]
        path = self.folder / f"{name}.h5"
        write_network(path, headers, self.pairs(rng, count, nodes),
                      {"g_score": g_score, "g_len": g_len, "l_score": l_score, "l_len": l_len}, seq_lens=seq_lens)
        for fasta in self.fastas(name, headers, [f"n{index}" for index in range(nodes)] if spaced else None):
            for score, norm in itertools.product(("global", "local"), ("alignment_length", "shorter_sequence",
                                                                       "longer_sequence", "average_sequence")):
                if score == "local" and norm == "alignment_length":
                    continue
                self.cases.append((f"{name} {fasta.name.split('.')[1]} {score}/{norm}", path, fasta, score, norm))

    def add_blast(self, name, rng, score, nodes, index_type=np.uint16):
        headers = [f"n{index}_description" for index in range(nodes)]
        path = self.folder / f"{name}.h5"
        write_network(path, headers, self.pairs(rng, len(score), nodes), {"score": score},
                      blast=True, index_type=index_type)
        for fasta in self.fastas(name, headers):
            self.cases.append((f"{name} {fasta.name.split('.')[1]}", path, fasta, "global", "alignment_length"))


class ChunkedStatisticsAreExact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.networks = Networks(cls.directory.name)

    def assert_same(self, label, old, new, *, mean_exact=False):
        self.assertEqual(old is None, new is None, label)
        if old is None:
            return
        for key in old:
            if key in ("mean", "_sorted_scores"):
                continue
            if key == "rows" and len(old[key]) > ROW_CAP:
                self.assert_capped_rows(label, old["_sorted_scores"], new[key])
                continue
            self.assertEqual(old[key], new[key], f"{label}: {key}")
        if mean_exact:
            self.assertEqual(old["mean"].hex(), new["mean"].hex(), label)
        else:
            self.assertLess(abs(old["mean"] - new["mean"]), 1e-6 * max(1.0, abs(old["mean"])), label)
            self.assertEqual(f"{old['mean']:.4f}", f"{new['mean']:.4f}", label)

    def assert_capped_rows(self, label, sorted_scores, rows):
        """The old code made more rows than the table now has: the new rows are fewer, and every count is right."""
        self.assertLessEqual(len(rows), ROW_CAP, label)
        scores = sorted_scores.astype(np.float64)
        for value, shown, _percent in rows:
            self.assertEqual(int(shown), int(np.count_nonzero(scores >= float(value))), f"{label}: row {value}")

    def test_every_case_at_every_chunk_size_and_worker_count(self):
        # 1_000_000 reads each network in one chunk; 997 and 64 cut it in many odd pieces.
        capped = 0
        for label, path, fasta, score, norm in self.networks.cases:
            old = legacy_statistics_report(path, str(fasta), score, norm, keep_scores=True)
            capped += old is not None and len(old["rows"]) > ROW_CAP
            for chunk, workers in ((1_000_000, 1), (997, 1), (64, 4), (997, 3)):
                with self.subTest(case=label, chunk=chunk, workers=workers):
                    new = new_statistics_report(path, str(fasta), score, norm, workers=workers, chunk_edges=chunk)
                    self.assert_same(label, old, new)
        # Some cases make more rows than the table now has, so the capped branch is exercised.
        self.assertGreater(capped, 3)

    def test_the_cases_really_put_scores_on_the_threshold_grid(self):
        # The tie networks must be able to tell a threshold rounded to the nearest float32 from one
        # rounded up: with nearest rounding some count differs from the whole-column count.
        path = self.networks.folder / "ties from 0.3.h5"
        fasta = self.networks.folder / "ties from 0.3.all.fasta"
        old = legacy_statistics_report(path, str(fasta), "global", "alignment_length")
        nearest = lambda t: np.asarray(t, dtype=np.float64).astype(np.float32)
        with mock.patch.object(stats, "ceil_float32", nearest):
            new = new_statistics_report(path, str(fasta), "global", "alignment_length", workers=1, chunk_edges=997)
        self.assertNotEqual(old["rows"], new["rows"])

    def test_ceil_float32_is_the_smallest_float32_not_below(self):
        rng = np.random.default_rng(1)
        thresholds = np.concatenate([np.arange(-3.0, 40.0, 0.1), rng.normal(size=500), [0.0, 1.0, 0.5, 300.0]])
        cutoffs = stats.ceil_float32(thresholds)
        self.assertEqual(cutoffs.dtype, np.float32)
        self.assertTrue(np.all(cutoffs.astype(np.float64) >= thresholds))
        below = np.nextafter(cutoffs, np.float32(-np.inf))
        self.assertTrue(np.all(below.astype(np.float64) < thresholds))

    def test_only_the_columns_a_setting_needs_are_read(self):
        class Recorder:
            """An open network that records the datasets whose values are read (not the shape asked of one)."""

            def __init__(self, network, read):
                self._network, self._read = network, read

            def __contains__(self, name):
                return name in self._network

            def __getitem__(self, name):
                dataset, read = self._network[name], self._read

                class Dataset:
                    shape, dtype = dataset.shape, dataset.dtype

                    def read_direct(self, *arguments, **options):
                        read.add(name)
                        return dataset.read_direct(*arguments, **options)

                    def __getitem__(self, selection):
                        read.add(name)
                        return dataset[selection]

                return Dataset()

        cases = (
            (("random", False, "global", "alignment_length"), None, {"g_score", "g_len"}),
            (("random", False, "local", "shorter_sequence"), None, {"l_score", "i", "j", "seq_lens"}),
            (("random", False, "global", "average_sequence"), "mask", {"g_score", "i", "j", "seq_lens"}),
            (("random", False, "global", "alignment_length"), "mask", {"g_score", "g_len", "i", "j"}),
            (("blast", True, "global", "alignment_length"), None, {"score"}),
            (("blast", True, "global", "alignment_length"), "mask", {"score", "i", "j"}),
        )
        for (name, blast, score, norm), mask, expected in cases:
            with self.subTest(network=name, score=score, norm=norm, subset=bool(mask)):
                read = set()
                with h5py.File(self.networks.folder / f"{name}.h5", "r") as hf:
                    kept = None if mask is None else np.arange(len(hf["headers"])) % 2 == 0
                    source = stats.ScoreSource(Recorder(hf, read), is_blast=blast, score_mode=score,
                                               norm_mode=norm, kept_mask=kept)
                    stats.compute_score_statistics(source, workers=1, chunk_edges=1000)
                self.assertEqual(read, expected)

    def test_a_slice_survives_the_next_read_on_the_same_thread(self):
        # Slices are read into buffers each thread keeps; what read() returns must not be one of them.
        for name, blast, normalize, cast in (("random", False, stats.statistics_normalize, True),
                                             ("blast", True, stats.statistics_normalize, False),
                                             ("blast", True, stats.statistics_normalize, True)):
            with self.subTest(network=name, cast=cast), h5py.File(self.networks.folder / f"{name}.h5", "r") as hf:
                source = stats.ScoreSource(hf, is_blast=blast, normalize=normalize, cast_float32=cast)
                first = source.read(0, 500, nodes=True)
                kept = [array.copy() for array in (first.scores, first.sources, first.targets)]
                source.read(500, 1000, nodes=True)
                for before, after in zip(kept, (first.scores, first.sources, first.targets)):
                    self.assertEqual(before.tolist(), after.tolist())

    def test_pair_positions_are_the_stored_offsets_of_the_kept_pairs(self):
        # read(positions=True) says where each kept pair sits among the stored pairs, whether or
        # not a node subset removed some of its slice, and costs nothing when not asked for.
        with h5py.File(self.networks.folder / "random.h5", "r") as hf:
            sources, targets = hf["i"][:], hf["j"][:]
            kept = np.arange(len(hf["headers"])) % 2 == 0
            valid = kept[sources] & kept[targets]
            for mask in (None, kept):
                source = stats.ScoreSource(hf, is_blast=False, kept_mask=mask)
                for start, stop in ((0, 1), (0, 997), (997, 2500), (11_000, 12_000)):
                    with self.subTest(subset=mask is not None, start=start, stop=stop):
                        piece = source.read(start, stop, nodes=True, positions=True)
                        expected = np.arange(start, stop)
                        if mask is not None:
                            expected = expected[valid[start:stop]]
                        self.assertEqual(piece.positions.dtype, np.int64)
                        np.testing.assert_array_equal(piece.positions, expected)
                        self.assertEqual(len(piece.positions), len(piece.scores))
                        self.assertIsNone(source.read(start, stop, nodes=True).positions)
            empty = stats.ScoreSource(hf, is_blast=False).read(0, 0, nodes=True, positions=True)
            self.assertEqual(len(empty.positions), 0)

    def test_a_cancelled_run_stops_and_says_so(self):
        path, fasta = self.networks.folder / "random.h5", self.networks.folder / "random.all.fasta"
        calls = []
        with h5py.File(path, "r") as hf:
            source = stats.ScoreSource(hf, is_blast=False)
            with self.assertRaises(stats.StatisticsCancelled):
                stats.compute_score_statistics(
                    source, workers=2, chunk_edges=500,
                    cancelled=lambda: calls.append(1) or len(calls) > 3,
                )

    def test_progress_runs_from_zero_to_one(self):
        path = self.networks.folder / "random.h5"
        seen = []
        with h5py.File(path, "r") as hf:
            source = stats.ScoreSource(hf, is_blast=False)
            stats.compute_score_statistics(source, workers=2, chunk_edges=500, progress=seen.append)
        self.assertEqual(seen, sorted(seen))
        self.assertGreater(seen[0], 0.0)
        self.assertAlmostEqual(seen[-1], 1.0)

    def test_the_worker_count_leaves_two_cpus_free(self):
        from utilities import Numba_Threads

        self.assertEqual(stats.RESERVED_CPUS, Numba_Threads.RESERVED_CPUS)
        self.assertEqual(stats.usable_cpu_count(), Numba_Threads.usable_cpu_count())
        self.assertEqual(stats.worker_count(), max(1, Numba_Threads.usable_cpu_count() - Numba_Threads.RESERVED_CPUS))


class OrderStatisticsAreExact(unittest.TestCase):
    """RankSelector and linear_quantiles give what np.sort and np.quantile give."""

    def select(self, scores, ranks, chunk, probes=()):
        selector = stats.RankSelector(ranks, scores.dtype, probes=probes)
        for level in range(selector.levels):
            total = selector.empty_histogram(level)
            for start in range(0, len(scores), chunk):
                selector.merge(total, selector.count(scores[start:start + chunk], level), level)
            selector.advance(total)
        return selector

    def arrays(self):
        rng = np.random.default_rng(3)
        yield "float32 uniform", rng.random(20_000).astype(np.float32)
        yield "float32 ties", rng.integers(0, 40, size=20_000).astype(np.float32) / 4
        yield "float32 negative", (rng.normal(size=20_000) * 50).astype(np.float32)
        yield "float32 constant", np.full(5_000, 0.3, dtype=np.float32)
        yield "float64 uniform", rng.random(20_000)
        yield "float64 clustered", 1.0 + rng.random(20_000) * 1e-9
        yield "float64 ties and negatives", rng.integers(-30, 30, size=20_000) / 8.0

    def test_values_and_the_number_below_each_rank(self):
        for label, scores in self.arrays():
            ordered = np.sort(scores)
            count = len(scores)
            ranks = [0, 1, count // 3, count // 2, count - 2, count - 1]
            for chunk in (count, 997):
                with self.subTest(case=label, chunk=chunk):
                    selector = self.select(scores, ranks, chunk)
                    for rank in ranks:
                        self.assertEqual(selector.values[rank], ordered[rank])
                        self.assertEqual(selector.below[rank], int(np.searchsorted(ordered, ordered[rank], side="left")))

    def test_the_number_of_scores_below_a_probe(self):
        for label, scores in self.arrays():
            ordered = np.sort(scores)
            rng = np.random.default_rng(4)
            probes = np.concatenate([
                rng.choice(scores, 8), ordered[[0, -1]], [ordered[0] - 1, ordered[-1] + 1],
                np.nextafter(rng.choice(scores, 3), np.array(np.inf, dtype=scores.dtype)),
            ]).astype(scores.dtype)
            for chunk in (len(scores), 997):
                with self.subTest(case=label, chunk=chunk):
                    selector = self.select(scores, [len(scores) // 2], chunk, probes=probes)
                    expected = [int(np.searchsorted(ordered, probe, side="left")) for probe in probes]
                    self.assertEqual(selector.probe_below, expected)

    def test_linear_quantiles_are_numpys(self):
        quantiles = mcp_statistics.QUANTILES
        for label, scores in self.arrays():
            for count in (len(scores), 7, 2, 1):
                part = scores[:count]
                ordered = np.sort(part)
                with self.subTest(case=label, count=count):
                    expected = np.quantile(part.astype(np.float64), quantiles)
                    got = stats.linear_quantiles(count, quantiles, lambda rank: ordered[rank])
                    self.assertEqual(expected.tolist(), got.tolist())

    def test_float_keys_sort_like_floats(self):
        for kind in (np.float32, np.float64):
            values = np.array([-np.inf, -1e30, -2.5, -1.0, -1e-30, 0.0, 1e-30, 0.5, 1.0, 2.5, 1e30, np.inf], dtype=kind)
            keys = stats.float_keys(values)
            self.assertTrue(np.all(np.diff(keys.astype(object)) > 0))
            self.assertEqual([stats.float_from_key(k, kind) for k in keys], list(values))

    def test_top_digit_keys_are_the_top_bits_of_the_keys(self):
        rng = np.random.default_rng(8)
        for kind, bits in ((np.float32, 32), (np.float64, 64)):
            values = np.concatenate([(rng.normal(size=5_000) * 30).astype(kind), np.array([0.0, -0.0, 1.0, -1.0], dtype=kind)])
            keys = stats.float_keys(values).astype(np.uint64)
            self.assertEqual(stats.top_digit_keys(values).tolist(), (keys >> np.uint64(bits - 16)).tolist())


class McpSummaryIsUnchanged(unittest.TestCase):
    """The MCP network_statistics report is the same, figure for figure, as the whole-column code made."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.networks = Networks(cls.directory.name)

    def settings(self, path):
        blast = path.name.startswith("blast")
        # 0.7, 0.9 and 1.3 are below the float32 nearest them; 0.5 and 5.0 are float32 exactly.
        yield {"thresholds": [5.0, 100.0, -3.0] if blast else [0.7, 0.9, 1.3, 0.5, 5.0]}
        yield {"top_edge_percents": [0.5, 30.0, 100.0], "thresholds": [0.0]}
        yield {}
        yield {"budget": 0.0}

    def compare(self, path, fasta, score, norm, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            old = legacy_summarize_network(str(path), fasta_path=fasta, alignment_score=score, norm_mode=norm, **options)
        with contextlib.redirect_stdout(io.StringIO()):
            new = mcp_statistics.summarize_network(str(path), fasta_path=fasta, alignment_score=score, norm_mode=norm, **options)
        self.assertEqual(old, new)

    WIDE = ("random", "spaced headers", "long sequences", "zero lengths", "blast", "blast negative")

    def distinct_cases(self):
        """(label, path, fasta, score, norm): the full set of modes on the wide networks, two on the others."""
        seen = set()
        for label, path, fasta, score, norm in self.networks.cases:
            wide = path.stem in self.WIDE
            if not wide and (score, norm) not in (("global", "alignment_length"), ("local", "shorter_sequence"))                     and not path.stem.startswith("blast"):
                continue
            key = (path, fasta, score, norm)
            if key not in seen:
                seen.add(key)
                yield label, path, fasta, score, norm

    def test_every_network_setting_and_chunk_size(self):
        # The default chunk takes every network in one slice; 97 cuts them into many.
        for chunk, settings in ((stats.CHUNK_EDGES, 3), (97, 1)):
            with mock.patch.object(stats, "CHUNK_EDGES", chunk):
                for label, path, fasta, score, norm in self.distinct_cases():
                    wide = path.stem in self.WIDE
                    for options in list(self.settings(path))[:settings if wide else 1]:
                        with self.subTest(case=label, chunk=chunk, options=options):
                            self.compare(path, str(fasta), score, norm, **options)

    def test_without_a_fasta_every_node_is_kept(self):
        for path in sorted(self.networks.folder.glob("*.h5")):
            with self.subTest(network=path.name):
                self.compare(path, None, "global", "alignment_length", thresholds=[0.3])

    def test_a_blank_header_keeps_every_node_as_prepare_network_does(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "blank.h5"
            write_network(path, ["a", "", "c", "d"], ([0, 0, 2], [1, 2, 3]), {"score": [1.0, 2.0, 3.0]}, blast=True)
            self.compare(path, None, "global", "alignment_length", thresholds=[1.5])

    def test_rows_over_the_edge_limit_skip_their_clusters_as_before(self):
        path = self.networks.folder / "random.h5"
        with mock.patch.object(mcp_statistics, "MAX_COMPONENT_EDGES", 300):
            for chunk in (stats.CHUNK_EDGES, 97):
                with mock.patch.object(stats, "CHUNK_EDGES", chunk):
                    self.compare(path, None, "global", "alignment_length", thresholds=[0.2, 0.3])
                    self.compare(path, None, "local", "average_sequence")

    def test_a_pair_index_past_the_header_count_is_refused_as_before(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "bad indices.h5"
            write_network(path, ["A", "B", "C"], ([0, 1], [2, 7]), {"score": [1.0, 2.0]}, blast=True)
            with self.assertRaisesRegex(IndexError, "exceed the header count"):
                mcp_statistics.summarize_network(str(path))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(IndexError, "exceed the header count"):
                legacy_summarize_network(str(path))


def legacy_thresholds(is_blast, min_score, max_score):
    """The threshold list the report always made, with no limit on its length."""
    if is_blast:
        start_val = int(math.floor(min_score))
        limit_step_1 = min(100, int(math.ceil(max_score)))
        thresh_low = np.arange(start_val, limit_step_1 + 1, 1)
        if max_score > 100:
            thresh_high = np.arange(105, int(math.ceil(max_score)) + 5, 5)
            return np.concatenate([thresh_low, thresh_high])
        return thresh_low
    start_val = math.floor(min_score * 10) / 10.0
    return np.arange(start_val, max_score + 0.1, 0.1)


# The table's limit, written out so a change to it is a decision: 200 keeps every usual table
# (a few dozen rows for alignment scores, about 140 for E-values) and fits the report panel.
ROW_CAP = 200


def is_nice(step):
    """Whether step is 1, 2 or 5 times a power of ten."""
    digit = step / 10.0 ** math.floor(math.log10(step))
    return any(abs(digit - wanted) < 1e-6 for wanted in (1.0, 2.0, 5.0))


class ThresholdTableTests(unittest.TestCase):
    """The report's threshold table has at most MAX_REPORT_ROWS rows, and its usual rows when that is enough."""

    def test_the_limit_is_the_two_hundred_rows_the_report_panel_shows_well(self):
        self.assertEqual(stats.MAX_REPORT_ROWS, ROW_CAP)

    def table(self, is_blast, low, high):
        return stats.report_thresholds(is_blast, np.float32(low), np.float32(high))

    def test_tables_that_fit_are_exactly_what_they_always_were(self):
        rng = np.random.default_rng(6)
        cases = [(False, 0.05, 0.936), (False, 0.0, 1.0), (False, -1.2, 0.95), (False, 0.3, 19.7), (True, 0.0, 300.0),
                 (True, 3.7, 120.2), (True, -5.0, 80.0), (True, 0.0, 100.0), (True, 0.0, 100.5), (True, 50.0, 594.0)]
        cases += [(False, float(lo), float(lo + width)) for lo, width in zip(rng.uniform(-3, 3, 60), rng.uniform(0.0, 19.9, 60))]
        cases += [(True, float(lo), float(lo + width)) for lo, width in zip(rng.uniform(-10, 80, 60), rng.uniform(0.0, 520, 60))]
        fitted = 0
        for is_blast, low, high in cases:
            expected = legacy_thresholds(is_blast, np.float32(low), np.float32(high))
            if len(expected) > ROW_CAP:
                continue
            fitted += 1
            with self.subTest(blast=is_blast, low=low, high=high):
                got = self.table(is_blast, low, high)
                self.assertEqual(got.dtype, expected.dtype)
                self.assertEqual(got.tolist(), expected.tolist())
        self.assertGreater(fitted, 100)

    def test_the_usual_steps_hold_up_to_the_limit_and_no_further(self):
        # 200 rows of 0.1 reach 19.9 above the lowest; a table to 20.0 would have 201.
        self.assertEqual(len(self.table(False, 0.0, 19.8)), 199)
        self.assertEqual(len(legacy_thresholds(False, np.float32(0.0), np.float32(20.0))), 201)
        self.assertLessEqual(len(self.table(False, 0.0, 20.0)), ROW_CAP)
        self.assertEqual(len(self.table(True, 0.0, 595.0)), ROW_CAP)
        self.assertLessEqual(len(self.table(True, 0.0, 600.0)), ROW_CAP)

    def test_a_wide_range_gets_the_smallest_nice_step_that_fits(self):
        for is_blast, low, high in ((False, 0.0, 5000.0), (False, -3.0, 2.5e4), (False, 0.05, 3.0e7), (False, -5.0e5, 1.2e6),
                                    (True, 0.0, 2000.0), (True, 2.0, 1.0e9), (True, -40.0, 9.0e3)):
            with self.subTest(blast=is_blast, low=low, high=high):
                table = self.table(is_blast, low, high)
                self.assertLessEqual(len(table), ROW_CAP)
                steps = np.diff(table.astype(np.float64))
                step = float(steps[0])
                self.assertTrue(np.allclose(steps, step, rtol=1e-9))
                self.assertTrue(is_nice(step), step)
                # from just below the minimum to just above the maximum
                low32, high32 = float(np.float32(low)), float(np.float32(high))
                self.assertLessEqual(float(table[0]), low32)
                self.assertGreater(float(table[0]) + step, low32 - 1e-9)
                # the last threshold reaches the maximum, as the usual table's does (its count may be 0)
                self.assertGreaterEqual(float(table[-1]), high32 - 1e-9 * max(1.0, abs(high32)))
                self.assertLess(float(table[-1]), high32 + step)
                # the next smaller nice step would not fit
                for candidate in (step / 2.5, step / 2.0, step / 5.0 * 2.0):
                    if candidate > 0.1 * (1 + 1e-9) and is_nice(candidate) and not (is_blast and candidate < 1):
                        rows = math.ceil((high32 + candidate) / candidate) - math.floor(low32 / candidate)
                        self.assertGreater(rows, ROW_CAP)
                if is_blast:
                    self.assertEqual(table.dtype, np.int64)

    def test_the_decimal_steps_are_the_floats_nearest_the_decimals(self):
        table = self.table(False, 0.0, 30.0)  # 0.2 apart
        self.assertEqual(table.tolist(), [round(index * 0.2, 1) for index in range(len(table))])
        self.assertIn(0.6, table.tolist())  # not 0.6000000000000001

    def test_scores_that_are_not_finite_are_refused(self):
        for bad in (np.inf, -np.inf, np.nan):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "not all finite"):
                stats.report_thresholds(False, np.float32(0.0), np.float32(bad))

    def test_the_report_of_a_network_with_huge_scores_stays_short_and_counts_right(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as folder:
            for blast, top in ((False, 2.0e4), (True, 5000.0)):
                count, nodes = 3_000, 80
                first = rng.integers(0, nodes - 1, size=count)
                second = rng.integers(first + 1, nodes)
                scores = (rng.random(count) ** 3 * top).astype(np.float32)
                scores[0] = top
                headers = [f"n{index}_x" for index in range(nodes)]
                path = pathlib.Path(folder) / ("blast.h5" if blast else "aln.h5")
                if blast:
                    write_network(path, headers, (first, second), {"score": scores}, blast=True)
                else:
                    write_network(path, headers, (first, second),
                                  {"g_score": scores, "g_len": np.ones(count), "l_score": scores, "l_len": np.ones(count)},
                                  seq_lens=np.ones(nodes))
                fasta = path.with_suffix(".fasta")
                write_fasta(fasta, headers, range(nodes))
                tracemalloc.start()
                try:
                    report = new_statistics_report(path, str(fasta), "global", "alignment_length", workers=2, chunk_edges=700)
                    peak = tracemalloc.get_traced_memory()[1]
                finally:
                    tracemalloc.stop()
                with self.subTest(blast=blast):
                    rows = report["rows"]
                    self.assertLessEqual(len(rows), ROW_CAP)
                    self.assertLess(peak, 20_000_000)
                    counts = [int(row[1]) for row in rows]
                    self.assertEqual(counts[0], count)
                    self.assertEqual(counts, sorted(counts, reverse=True))
                    for value, shown, _percent in rows:
                        self.assertEqual(int(shown), int(np.count_nonzero(scores.astype(np.float64) >= float(value))))


class MemoryBudgets(unittest.TestCase):
    """A few million pairs are read in a few megabytes, where the whole-column code needed 30 to 100 bytes each."""

    NODES = 2_450  # 3,000,025 pairs
    # Bytes per pair at the peak. The chunked code holds a few slices per worker (here 2 workers,
    # 50,000 pairs each, a few columns at about 30 bytes a pair); the Config code read 32 bytes per
    # pair and the MCP code more than 100.
    CONFIG_BUDGET = 6.0
    MCP_BUDGET = 25.0

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        folder = pathlib.Path(cls.directory.name)
        rng = np.random.default_rng(5)
        first, second = np.triu_indices(cls.NODES, 1)
        cls.edges = len(first)
        lengths = rng.integers(30, 1500, size=cls.edges)
        headers = [f"n{index} description" for index in range(cls.NODES)]
        cls.path = folder / "big.h5"
        write_network(
            cls.path, headers, (first, second),
            {"g_score": (rng.beta(2, 5, size=cls.edges) * 0.9 + 0.05) * lengths, "g_len": lengths,
             "l_score": rng.random(cls.edges), "l_len": lengths},
            seq_lens=rng.integers(30, 900, size=cls.NODES),
        )
        cls.fasta = folder / "big.fasta"
        write_fasta(cls.fasta, headers, range(cls.NODES))
        # numba and the first import of every library would otherwise count against the first run
        new_statistics_report(cls.path, str(cls.fasta), "global", "alignment_length", workers=2, chunk_edges=50_000)

    def peak(self, function):
        tracemalloc.start()
        try:
            function()
            return tracemalloc.get_traced_memory()[1] / self.edges
        finally:
            tracemalloc.stop()

    def test_the_config_statistics_stay_within_budget(self):
        for score, norm in (("global", "alignment_length"), ("local", "average_sequence")):
            with self.subTest(score=score, norm=norm):
                peak = self.peak(lambda: new_statistics_report(
                    self.path, str(self.fasta), score, norm, workers=2, chunk_edges=50_000))
                self.assertLess(peak, self.CONFIG_BUDGET)

    def test_the_whole_column_code_exceeds_that_budget(self):
        # Positive control: the measurement notices code that holds the columns.
        peak = self.peak(lambda: legacy_statistics_report(self.path, str(self.fasta), "global", "alignment_length"))
        self.assertGreater(peak, 4 * self.CONFIG_BUDGET)

    def test_the_mcp_summary_stays_within_budget(self):
        with mock.patch.object(stats, "CHUNK_EDGES", 50_000), mock.patch.object(stats, "worker_count", lambda: 2):
            peak = self.peak(lambda: mcp_statistics.summarize_network(str(self.path), fasta_path=None))
        self.assertLess(peak, self.MCP_BUDGET)

    def test_the_whole_column_mcp_code_exceeds_that_budget(self):
        def old():
            with contextlib.redirect_stdout(io.StringIO()):
                legacy_summarize_network(str(self.path))
        self.assertGreater(self.peak(old), 2 * self.MCP_BUDGET)


if __name__ == "__main__":
    unittest.main()
