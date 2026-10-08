# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Edge-filter preview for one network, scored exactly as a layout job scores it.

The MCP server runs this module as a child process (like file inspection), so
reading a large network never blocks or bloats the server. Heavy imports stay
inside functions.
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


def _top_percent_cutoff(scores_descending, node_count, percent):
    """The layout job's TOP_EDGE_PERCENT cutoff (prepare_network's rule)."""
    possible = node_count * (node_count - 1) / 2.0
    count = int(possible * (percent / 100.0))
    if len(scores_descending) == 0:
        return 0.0
    count = max(1, min(count, len(scores_descending)))
    return float(scores_descending[count - 1])


def summarize_network(network_path, fasta_path=None, alignment_score="global",
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
            "stored_percent_of_possible": _rounded(100.0 * len(scores) / possible) if possible else None,
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
            "UMAP layouts ignore both filters and keep each node's UMAP_NEIGHBORS strongest edges.",
        ],
        "complete": True,
    }
    if len(scores) == 0:
        report["notes"].append("The network has no edges between the kept nodes.")
        return report

    quantiles = np.quantile(scores, QUANTILES)
    report["score_summary"] = {
        "min": _rounded(scores.min()), "max": _rounded(scores.max()),
        "mean": _rounded(scores.mean()),
        "quantiles": {f"p{int(q * 100)}": _rounded(v) for q, v in zip(QUANTILES, quantiles)},
    }
    counts, bin_edges = np.histogram(scores, bins=20)
    report["histogram"] = {"bin_edges": [_rounded(v) for v in bin_edges],
                           "counts": [int(v) for v in counts]}

    order = np.argsort(-scores, kind="stable")
    descending = scores[order]
    rows = []
    percents = list(DEFAULT_TOP_EDGE_PERCENTS if top_edge_percents is None else top_edge_percents)
    for percent in percents[:len(DEFAULT_TOP_EDGE_PERCENTS) + MAX_EXTRA_ROWS]:
        rows.append(("top_edge_percent", float(percent),
                     _top_percent_cutoff(descending, node_count, float(percent))))
    for threshold in list(thresholds)[:MAX_EXTRA_ROWS]:
        rows.append(("similarity_threshold", float(threshold), float(threshold)))
    rows.sort(key=lambda row: -row[2])
    for mode, value, cutoff in rows:
        kept = int(np.searchsorted(-descending, -cutoff, side="right"))
        entry = {
            "filter": mode, "value": _rounded(value),
            "similarity_threshold": _rounded(cutoff),
            "kept_edges": kept,
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
            chosen = order[:kept]
            entry["connectivity"] = component_summary(node_count, edges[chosen, 0], edges[chosen, 1])
        report["edge_filters"].append(entry)
    return report


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
