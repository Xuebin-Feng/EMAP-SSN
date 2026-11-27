"""Networks and former-code references shared by the edge-filter and statistics tests.

write_network() writes a network in the dtypes real files use, its pairs in no
particular order (as stored networks are); ArraySource stands in for a ScoreSource
over arrays in memory. former_pair_scores(), former_threshold_prepare(), and
legacy_statistics_report() restate what the whole-column readers computed before
streaming the pairs: the references the streamed output is compared with.
"""

import math
import os
import pathlib
import sys
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Cache_Manifest as cache_manifest
from desktop.Viewer_State import _normalize_score
from utilities import Network_Score_Statistics as scan

SEQUENCE_LENGTH_NORMALIZATIONS = frozenset(
    {"shorter_sequence", "longer_sequence", "average_sequence"}
)


class ArraySource:
    """The part of a ScoreSource the edge filters use, over arrays in memory."""

    def __init__(self, sources, targets, scores):
        self.sources, self.targets, self.scores = sources, targets, scores
        self.edge_count = len(scores)
        self.reads = []

    def read(self, start, stop, nodes=False, check_nodes=False, positions=False):
        stop = min(stop, self.edge_count)
        self.reads.append((start, stop))
        return scan.Slice(
            self.scores[start:stop].copy(),
            self.sources[start:stop].copy(),
            self.targets[start:stop].copy(),
            np.arange(start, stop, dtype=np.int64),
        )


def write_network(
    path, node_count, rng, *, blast=False, ties=False, nan=False, pair_count=None,
    score_dtype=np.float32,
):
    """Write a network and return its columns.

    A complete network (every pair once) unless ``pair_count`` asks for that many random
    pairs, repeats and self-pairs included. ``ties`` makes scores take a few values;
    ``nan`` makes some raw scores NaN; ``score_dtype`` is the stored dtype of an E-value
    network's scores.
    """
    if pair_count is None:
        sources, targets = np.triu_indices(node_count, k=1)
        order = rng.permutation(len(sources))
        sources, targets = sources[order], targets[order]
    else:
        sources = rng.integers(0, node_count, pair_count)
        targets = rng.integers(0, node_count, pair_count)
    pair_count = len(sources)
    columns = {"i": sources.astype(np.uint16), "j": targets.astype(np.uint16)}

    def raw_scores(distinct_kind):
        if ties:
            scores = rng.integers(0, 6, pair_count).astype(np.float64) * distinct_kind
        else:
            scores = rng.permutation(pair_count).astype(np.float64)
        if nan:
            scores[rng.random(pair_count) < 0.15] = np.nan
            # half of them with the sign bit set, as inf - inf makes them on x86
            scores[rng.random(pair_count) < 0.07] = np.copysign(np.nan, -1.0)
        return scores

    if blast:
        columns["score"] = raw_scores(25.0).astype(score_dtype)
    else:
        columns["seq_lens"] = rng.integers(1, 9, node_count).astype(np.uint16)
        for kind in ("g", "l"):
            columns[f"{kind}_score"] = raw_scores(1.0).astype(np.float32)
            if ties:
                columns[f"{kind}_len"] = rng.integers(1, 3, pair_count).astype(np.uint16)
            else:
                # Length 1: the normalized scores are the distinct raw scores.
                columns[f"{kind}_len"] = np.ones(pair_count, dtype=np.uint16)
    with h5py.File(path, "w") as network:
        network.attrs["model_name"] = "BLAST" if blast else "model"
        network.create_dataset(
            "headers", data=[f"N{index}".encode() for index in range(node_count)]
        )
        for name, values in columns.items():
            network.create_dataset(name, data=values)
    return columns


def former_pair_scores(data, settings, pair_mask, valid_sources, valid_targets):
    """The normalized score of each pair ``pair_mask`` keeps, from whole columns (the former reader)."""
    if settings.INPUT_IS_EVALUE:
        return data["score"][:][pair_mask]

    if settings.ALIGNMENT_SCORE == "global":
        score_name, length_name = "g_score", "g_len"
    else:
        score_name, length_name = "l_score", "l_len"
    raw_scores = data[score_name][:][pair_mask]
    alignment_lengths = source_lengths = target_lengths = None
    if settings.NORM_MODE in SEQUENCE_LENGTH_NORMALIZATIONS:
        sequence_lengths = data["seq_lens"][:]
        source_lengths = sequence_lengths[valid_sources]
        target_lengths = sequence_lengths[valid_targets]
    else:
        alignment_lengths = data[length_name][:][pair_mask]
    return _normalize_score(
        raw_scores, alignment_lengths, source_lengths, target_lengths, settings.NORM_MODE
    )


def former_threshold_prepare(path, settings, selected):
    """prepare_network's former threshold branch on a network file.

    ``selected`` is the set of kept node names. Returns the kept headers, edges and
    edge scores, the threshold the settings end with, and the two lines a top-percent
    cutoff printed (none otherwise). ``settings`` is not changed.
    """
    settings = SimpleNamespace(**vars(settings))
    printed = []
    with h5py.File(path, "r") as network:
        headers = [name.decode() for name in network["headers"][:]]
        settings.INPUT_IS_EVALUE = "score" in network
        total_nodes = len(headers)
        sources = network["i"][:]
        targets = network["j"][:]
        kept_indices = np.array(
            [index for index, name in enumerate(headers) if name in selected], dtype=np.int64
        )
        kept_mask = np.zeros(total_nodes, dtype=bool)
        kept_mask[kept_indices] = True
        index_map = np.zeros(total_nodes, dtype=np.int32)
        index_map[kept_indices] = np.arange(len(kept_indices))
        if kept_mask.all():
            if len(sources) and max(int(sources.max()), int(targets.max())) >= total_nodes:
                raise IndexError("Network pair indices exceed the header count.")
            valid_edges_mask = slice(None)
        else:
            valid_edges_mask = kept_mask[sources] & kept_mask[targets]
        valid_sources = sources[valid_edges_mask]
        valid_targets = targets[valid_edges_mask]
        valid_scores = former_pair_scores(
            network, settings, valid_edges_mask, valid_sources, valid_targets
        )

    top_percent = getattr(settings, "TOP_EDGE_PERCENT", None)
    if top_percent is not None:
        active_nodes = len(kept_indices)
        theoretical_max_edges = (active_nodes * (active_nodes - 1)) / 2.0
        edge_count = int(theoretical_max_edges * (top_percent / 100.0))
        if len(valid_scores) == 0:
            calculated_cutoff = 0.0
        else:
            edge_count = max(1, min(edge_count, len(valid_scores)))
            kth = len(valid_scores) - edge_count
            calculated_cutoff = float(np.partition(valid_scores, kth)[kth])
        mode_label = "E-Value" if settings.INPUT_IS_EVALUE else "Similarity"
        printed.append(
            f"Top {top_percent}% Edges Requested (based on max possible "
            f"{int(theoretical_max_edges)} edges)."
        )
        printed.append(f"Calculated {mode_label} Cutoff: {calculated_cutoff:.5f}")
        settings.SIMILARITY_THRESHOLD = calculated_cutoff

    threshold_mask = valid_scores >= settings.SIMILARITY_THRESHOLD
    final_sources = index_map[valid_sources[threshold_mask]]
    final_targets = index_map[valid_targets[threshold_mask]]
    edges = np.column_stack((final_sources, final_targets)).astype(np.int32)
    edge_scores = valid_scores[threshold_mask]
    return (
        [headers[index] for index in kept_indices], edges, edge_scores,
        settings.SIMILARITY_THRESHOLD, printed,
    )


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
