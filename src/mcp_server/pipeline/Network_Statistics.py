# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Edge-filter preview for one network, scored exactly as a layout job scores it.

The MCP server runs this module as a child process (like file inspection), so
reading a large network never blocks or bloats the server. Heavy imports stay
inside functions.

The pairs are read a slice at a time (utilities.Network_Score_Statistics), never as
whole columns, so a network of a billion pairs needs a few hundred megabytes besides
the edges of the rows whose connectivity is asked for (at most
MAX_COMPONENT_EDGES of them). Everything reported is what scoring and sorting every
pair gave before: the quantiles and the cutoffs are exact order statistics of the
scores, found by a radix selection over a few passes, and the counts are exact.
"""

import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT_TOP_EDGE_PERCENTS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
MAX_EXTRA_ROWS = 10
# Connectivity is skipped for rows that keep more edges than this.
MAX_COMPONENT_EDGES = 25_000_000
HELPER_TIMEOUT_SECONDS = 50
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
HISTOGRAM_BINS = 20


class NetworkStatisticsError(ValueError):
    pass


def component_summary(node_count, sources, targets):
    """Clusters (components of two or more nodes), isolated nodes and the largest cluster."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    node_count = int(node_count)
    if node_count == 0:
        return {"clusters": 0, "isolated_nodes": 0, "largest_cluster_nodes": 0}
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(len(sources), dtype=np.int8), (sources, targets)),
        shape=(node_count, node_count),
    ).tocsr()
    _count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    return {
        "clusters": int(np.count_nonzero(sizes >= 2)),
        "isolated_nodes": int(np.count_nonzero(sizes == 1)),
        "largest_cluster_nodes": int(sizes.max()) if sizes.max() >= 2 else 0,
    }


def _rounded(value):
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _top_percent_rank(stored, node_count, percent):
    """How many best scores the layout job's TOP_EDGE_PERCENT keeps (prepare_network's rule).

    The cutoff is the score of that rank from the top.
    """
    possible = node_count * (node_count - 1) / 2.0
    count = int(possible * (percent / 100.0))
    return max(1, min(count, stored))


def _layout_node_mask(network_headers, selected):
    """Which network nodes a layout job keeps for these FASTA headers (prepare_network's rule)."""
    import numpy as np
    from utilities import Network_Score_Statistics as scan

    fasta_ids, fasta_headers = set(), set()
    try:
        for header in selected:
            fasta_headers.add(header)
            parts = header.split()
            if parts:
                fasta_ids.add(parts[0])
        mask, _missing = scan.kept_node_mask(network_headers, fasta_ids, fasta_headers)
    except Exception:
        # prepare_network keeps every node when it cannot read the filter.
        mask = np.ones(len(network_headers), dtype=bool)
    return mask


def summarize_network(network_path, fasta_path=None, alignment_score="global",
                      norm_mode="alignment_length", thresholds=(), top_edge_percents=None,
                      budget=40.0):
    """Score distribution and edge-filter table for the nodes a layout would keep."""
    import h5py
    import numpy as np

    started = time.monotonic()
    from Cache_Manifest import validate_network_schema
    from desktop.Viewer_State import _normalize_score
    from utilities import Network_Score_Statistics as scan
    from utilities.Sequence_Utils import load_sanitized_fasta

    network_path = os.fspath(network_path)
    if not os.path.isfile(network_path):
        raise NetworkStatisticsError(f"Network file was not found: {network_path}")
    metadata = validate_network_schema(network_path)
    is_blast = metadata.network_type == "blast"
    if not is_blast:
        if alignment_score not in ("global", "local"):
            raise NetworkStatisticsError("alignment_score must be global or local.")
        if norm_mode not in ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"):
            raise NetworkStatisticsError("norm_mode is not supported.")
        if alignment_score == "local" and norm_mode == "alignment_length":
            raise NetworkStatisticsError(
                "norm_mode alignment_length is unavailable for local scores; choose "
                "shorter_sequence, longer_sequence or average_sequence."
            )

    with h5py.File(network_path, "r") as data:
        raw_headers = data["headers"][:]
        network_headers = [h.decode("utf-8") if isinstance(h, bytes) else h for h in raw_headers]
        if fasta_path:
            if not os.path.isfile(fasta_path):
                raise NetworkStatisticsError(f"FASTA file was not found: {fasta_path}")
            selected, _sequences, _stats = load_sanitized_fasta(fasta_path)
        else:
            selected = network_headers
        kept_mask = _layout_node_mask(network_headers, selected)
        kept_indices = np.flatnonzero(kept_mask)
        headers = [network_headers[index] for index in kept_indices]
        all_kept = bool(kept_mask.all())
        source = scan.ScoreSource(
            data, is_blast=is_blast, score_mode=alignment_score, norm_mode=norm_mode,
            kept_mask=None if all_kept else kept_mask,
            # The layout's own normalization, on the columns as stored.
            normalize=_normalize_score, cast_float32=False,
            # A layout job refuses pair indices past the header count when every node is kept.
            node_count=len(network_headers) if all_kept else None,
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
                "stored": 0,
                "possible": int(possible),
                "stored_percent_of_possible": None,
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
        percents = list(DEFAULT_TOP_EDGE_PERCENTS if top_edge_percents is None else top_edge_percents)
        percents = [float(percent) for percent in percents[:len(DEFAULT_TOP_EDGE_PERCENTS) + MAX_EXTRA_ROWS]]
        extra_thresholds = [float(threshold) for threshold in list(thresholds)[:MAX_EXTRA_ROWS]]
        scores = _score_passes(scan, source, node_count, percents, extra_thresholds)
        report["pairs"]["stored"] = int(scores.count)
        report["pairs"]["stored_percent_of_possible"] = (
            _rounded(100.0 * scores.count / possible) if possible else None
        )
        if scores.count == 0:
            report["notes"].append("The network has no edges between the kept nodes.")
            return report

        report["score_summary"] = {
            "min": _rounded(scores.minimum), "max": _rounded(scores.maximum),
            "mean": _rounded(scores.total / scores.count),
            "quantiles": {f"p{int(q * 100)}": _rounded(v) for q, v in zip(QUANTILES, scores.quantiles)},
        }
        report["histogram"] = {"bin_edges": [_rounded(v) for v in scores.bin_edges],
                               "counts": [int(v) for v in scores.histogram]}

        rows = []
        for percent in percents:
            rows.append(("top_edge_percent", percent, scores.top_cutoffs[percent], scores.top_kept[percent]))
        for threshold, kept in zip(extra_thresholds, scores.threshold_kept):
            rows.append(("similarity_threshold", threshold, threshold, kept))
        rows.sort(key=lambda row: -row[2])
        # The rows whose clusters will be counted need the edges they keep: read once, below.
        wanted = [row for row in rows if row[3] <= MAX_COMPONENT_EDGES]
        lowest = min(wanted, key=lambda row: row[2]) if wanted else None
        edges = None
        for mode, value, cutoff, kept in rows:
            entry = {
                "filter": mode, "value": _rounded(value),
                "similarity_threshold": _rounded(cutoff),
                "kept_edges": int(kept),
                "kept_percent_of_possible": _rounded(100.0 * kept / possible) if possible else None,
                "mean_degree": _rounded(2.0 * kept / node_count) if node_count else None,
            }
            if kept > MAX_COMPONENT_EDGES:
                entry["connectivity"] = None
                entry["connectivity_skipped"] = f"more than {MAX_COMPONENT_EDGES} edges"
            elif time.monotonic() - started > budget:
                entry["connectivity"] = None
                entry["connectivity_skipped"] = "time budget reached"
                report["complete"] = False
            else:
                if edges is None:
                    edges = _collect_edges(scan, source, scores.dtype, kept_indices, lowest[2], lowest[3])
                chosen = edges.scores >= scan.cutoffs_for([cutoff], scores.dtype)[0]
                entry["connectivity"] = component_summary(
                    node_count, edges.sources[chosen], edges.targets[chosen]
                )
            report["edge_filters"].append(entry)
    return report


class _Scores:
    """What the passes over the scores found."""


def _score_passes(scan, source, node_count, percents, extra_thresholds):
    """The count, extremes, sum, quantiles, histogram, cutoffs and kept counts of the scores."""
    import numpy as np

    found = _Scores()
    ranges = scan.chunk_ranges(source.edge_count)

    def first_pass(start, stop):
        # The first look at every pair also checks that the pair indices fit the headers.
        part = source.read(start, stop, check_nodes=True).scores
        if len(part) == 0:
            return None
        kept = None
        if extra_thresholds:
            ordered = np.sort(part)
            kept = scan.count_at_least(ordered, scan.cutoffs_for(extra_thresholds, part.dtype))
        return (len(part), part.min(), part.max(), float(part.sum(dtype=np.float64)),
                scan.RankSelector.first_digit_counts(part), kept, part.dtype)

    count, total = 0, 0.0
    low = high = dtype = None
    digits = np.zeros(scan.RankSelector.SIZE, dtype=np.int64)
    threshold_kept = np.zeros(len(extra_thresholds), dtype=np.int64)
    for result in scan.run_pass(first_pass, ranges):
        if result is None:
            continue
        part_count, part_low, part_high, part_total, part_digits, part_kept, dtype = result
        count += part_count
        total += part_total
        low = part_low if low is None else min(low, part_low)
        high = part_high if high is None else max(high, part_high)
        digits += part_digits
        if part_kept is not None:
            threshold_kept += part_kept
    found.count, found.total, found.minimum, found.maximum, found.dtype = count, total, low, high, dtype
    found.threshold_kept = [int(value) for value in threshold_kept]
    if count == 0:
        return found

    # Order statistics: numpy's quantiles interpolate between two neighbouring ranks, and a
    # TOP_EDGE_PERCENT cutoff is the score of one rank counted from the top.
    previous, following = scan.quantile_ranks(count, QUANTILES)
    top_ranks = {
        percent: count - _top_percent_rank(count, node_count, percent) for percent in percents
    }
    # The histogram's equal-width bins, as np.histogram makes them. Bin i holds the scores from
    # edge i up to edge i + 1 (the last bin includes its upper edge), so the bin counts are the
    # differences of how many scores lie below each edge, which the selector counts as it narrows
    # the ranks: no slice is histogrammed.
    low64, high64 = np.float64(low), np.float64(high)
    found.bin_edges = np.histogram(np.empty(0, dtype=np.float64), bins=HISTOGRAM_BINS, range=(low64, high64))[1]
    selector = scan.RankSelector(
        [*previous.tolist(), *following.tolist(), *top_ranks.values()], dtype,
        probes=scan.cutoffs_for(found.bin_edges, dtype),
    )
    selector.advance(digits)
    for level in range(1, selector.levels):
        def pass_at_level(start, stop, level=level):
            part = source.read(start, stop).scores
            return selector.count(part, level) if len(part) else None

        level_counts = selector.empty_histogram(level)
        for result in scan.run_pass(pass_at_level, ranges):
            if result is not None:
                selector.merge(level_counts, result, level)
        selector.advance(level_counts)
    below_edge = selector.probe_below
    found.histogram = np.array(
        [below_edge[index + 1] - below_edge[index] for index in range(HISTOGRAM_BINS - 1)]
        + [count - below_edge[HISTOGRAM_BINS - 1]], dtype=np.int64)
    found.quantiles = scan.linear_quantiles(count, QUANTILES, lambda rank: selector.values[rank])
    found.top_cutoffs = {percent: float(selector.values[rank]) for percent, rank in top_ranks.items()}
    found.top_kept = {percent: int(count - selector.below[rank]) for percent, rank in top_ranks.items()}
    return found


class _Edges:
    pass


def _collect_edges(scan, source, dtype, kept_indices, cutoff, expected):
    """The kept nodes' numbering of every edge scoring at least ``cutoff``, with its score.

    ``expected`` is how many edges that is, known from the passes before, so the arrays
    are made once and filled slice by slice.
    """
    import numpy as np

    index_map = np.zeros(source.network["headers"].shape[0], dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices), dtype=np.int32)
    bar = scan.cutoffs_for([cutoff], dtype)[0]

    def pick(start, stop):
        part = source.read(start, stop, nodes=True)
        wanted = part.scores >= bar
        return (index_map[part.sources[wanted]], index_map[part.targets[wanted]], part.scores[wanted])

    found = _Edges()
    found.sources = np.empty(expected, dtype=np.int32)
    found.targets = np.empty(expected, dtype=np.int32)
    found.scores = np.empty(expected, dtype=dtype)
    filled = 0
    for sources, targets, scores in scan.run_pass(pick, scan.chunk_ranges(source.edge_count)):
        found.sources[filled:filled + len(scores)] = sources
        found.targets[filled:filled + len(scores)] = targets
        found.scores[filled:filled + len(scores)] = scores
        filled += len(scores)
    if filled != expected:
        raise RuntimeError(f"Found {filled} edges at or above the cutoff, where the counts said {expected}.")
    return found


def network_statistics(request, timeout=HELPER_TIMEOUT_SECONDS):
    """Run summarize_network in a child process and return its report."""
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())], input=json.dumps(request),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired as error:
        raise NetworkStatisticsError(
            f"Network statistics did not finish within {timeout} s. The network is too large for a "
            "synchronous preview; choose TOP_EDGE_PERCENT for the layout instead."
        ) from error
    try:
        output = json.loads(completed.stdout)
    except ValueError as error:
        detail = completed.stderr.strip()[-1500:]
        raise NetworkStatisticsError(
            f"Network statistics helper failed (exit {completed.returncode}): {detail}"
        ) from error
    if not isinstance(output, dict):
        raise NetworkStatisticsError("Network statistics helper returned an invalid report.")
    if "error" in output:
        raise NetworkStatisticsError(output["error"])
    return output


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    request = json.load(sys.stdin)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            report = summarize_network(**request)
    except (NetworkStatisticsError, OSError, KeyError, ValueError, TypeError) as error:
        report = {"error": str(error)}
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
