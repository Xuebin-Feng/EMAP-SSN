# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare validated SSN connectivity from raw network HDF5 data."""

import os

import numpy as np

import Cache_Manifest as cache_manifest
from utilities.FASTA_Sanitization import load_sanitized_fasta


def _normalize_score(raw_score, align_len, len_i, len_j, mode):
    if mode == "alignment_length":
        denominator = align_len
    elif mode == "shorter_sequence":
        denominator = np.minimum(len_i, len_j)
    elif mode == "longer_sequence":
        denominator = np.maximum(len_i, len_j)
    elif mode == "average_sequence":
        denominator = (len_i + len_j) / 2.0
    else:
        denominator = align_len

    return np.where(denominator > 0, raw_score / denominator, 0.0)


def prepare_network(data, *, settings, selected_fasta_headers=None):
    """Return filtered ``(headers, edges, scores)`` for an open network file.

    ``settings`` is explicit because preparation is shared by the viewer and
    the headless layout generator. The effective network type and any computed
    top-edge threshold are written back to that namespace for their downstream
    consumers.
    """
    metadata = cache_manifest.validate_network_schema(data)
    settings.INPUT_IS_EVALUE = metadata.network_type == "blast"

    raw_headers = data["headers"][:]
    headers = [
        header.decode("utf-8") if isinstance(header, bytes) else header
        for header in raw_headers
    ]
    total_nodes = len(headers)

    sources = data["i"][:]
    targets = data["j"][:]
    if settings.INPUT_IS_EVALUE:
        scores = data["score"][:]
    else:
        sequence_lengths = data["seq_lens"][:]
        if settings.ALIGNMENT_SCORE == "global":
            alignment_scores = data["g_score"][:]
            alignment_lengths = data["g_len"][:]
        else:
            alignment_scores = data["l_score"][:]
            alignment_lengths = data["l_len"][:]

    print(f"Raw Data: {total_nodes} sequences.")
    if not settings.INPUT_IS_EVALUE:
        print(
            f"Metric: {settings.ALIGNMENT_SCORE.upper()} Alignment with "
            f"{settings.NORM_MODE} Normalization"
        )

    fasta_path = getattr(settings, "NODE_FASTA_FILE", "")
    kept_indices = []
    if selected_fasta_headers is not None or os.path.exists(fasta_path):
        clean_fasta_path = os.path.normpath(fasta_path)
        print(f"Scanning FASTA file for node filter: {clean_fasta_path}")
        fasta_ids = set()
        fasta_headers = set()
        try:
            if selected_fasta_headers is None:
                selected_fasta_headers, _, _ = load_sanitized_fasta(fasta_path)

            for header in selected_fasta_headers:
                fasta_headers.add(header)
                header_parts = header.split()
                if header_parts:
                    fasta_ids.add(header_parts[0])

            network_headers = set(headers)
            network_ids = {header.split()[0] for header in headers}
            missing_nodes = [
                identifier
                for identifier in fasta_ids
                if identifier not in network_ids and identifier not in network_headers
            ]
            if missing_nodes:
                print(
                    "CRITICAL WARNING: The passed FASTA file is NOT a strict "
                    f"subset of the network file. {len(missing_nodes)} FASTA "
                    "sequences are missing from the network."
                )

            for index, header in enumerate(headers):
                record_id = header.split()[0]
                if header in fasta_headers or record_id in fasta_ids:
                    kept_indices.append(index)

            kept_indices = np.asarray(kept_indices, dtype=np.int64)
            print(
                f"Filtered {total_nodes} down to {len(kept_indices)} valid "
                "FASTA subsets."
            )
        except Exception as error:
            print(f"Error reading FASTA filter: {error}. Retaining all sequences.")
            kept_indices = np.arange(total_nodes)
    else:
        print(
            f"No FASTA file found at {fasta_path}. Retaining all "
            f"{total_nodes} sequences."
        )
        kept_indices = np.arange(total_nodes)

    kept_mask = np.zeros(total_nodes, dtype=bool)
    kept_mask[kept_indices] = True
    filtered_headers = [headers[index] for index in kept_indices]

    index_map = np.zeros(total_nodes, dtype=np.int32)
    index_map[kept_indices] = np.arange(len(kept_indices))

    valid_edges_mask = kept_mask[sources] & kept_mask[targets]
    valid_sources = sources[valid_edges_mask]
    valid_targets = targets[valid_edges_mask]
    if settings.INPUT_IS_EVALUE:
        valid_scores = scores[valid_edges_mask]
    else:
        valid_raw_scores = alignment_scores[valid_edges_mask]
        valid_alignment_lengths = alignment_lengths[valid_edges_mask]
        valid_scores = _normalize_score(
            valid_raw_scores,
            valid_alignment_lengths,
            sequence_lengths[valid_sources],
            sequence_lengths[valid_targets],
            settings.NORM_MODE,
        )

    top_percent = getattr(settings, "TOP_EDGE_PERCENT", None)
    if top_percent is not None and not getattr(settings, "UMAP_MODE", False):
        active_nodes = len(kept_indices)
        theoretical_max_edges = (active_nodes * (active_nodes - 1)) / 2.0
        edge_count = int(theoretical_max_edges * (top_percent / 100.0))
        if len(valid_scores) == 0:
            calculated_cutoff = 0.0
        else:
            edge_count = max(1, min(edge_count, len(valid_scores)))
            calculated_cutoff = np.sort(valid_scores)[::-1][edge_count - 1]

        mode_label = "E-Value" if settings.INPUT_IS_EVALUE else "Similarity"
        print(
            f"Top {top_percent}% Edges Requested (based on max possible "
            f"{int(theoretical_max_edges)} edges)."
        )
        print(f"Calculated {mode_label} Cutoff: {calculated_cutoff:.5f}")
        settings.SIMILARITY_THRESHOLD = calculated_cutoff

    if getattr(settings, "UMAP_MODE", False):
        print(
            "UMAP Mode enabled: Bypassing global threshold. Filtering top k "
            "edges per node..."
        )
        keep_limit = int(getattr(settings, "UMAP_NEIGHBORS", 15))
        import pandas as pd

        frame = pd.DataFrame(
            {
                "u": valid_sources,
                "v": valid_targets,
                "score": valid_scores,
                "idx": np.arange(len(valid_scores)),
            }
        )
        sorted_frame = frame.sort_values("score", ascending=False)
        top_sources = sorted_frame.groupby("u").head(keep_limit)["idx"]
        top_targets = sorted_frame.groupby("v").head(keep_limit)["idx"]
        kept_edge_indices = pd.concat([top_sources, top_targets]).unique()
        threshold_mask = np.zeros(len(valid_scores), dtype=bool)
        threshold_mask[kept_edge_indices] = True
        print(
            f"Kept {len(kept_edge_indices)} edges for UMAP topology "
            f"(max {keep_limit} per node direction)."
        )
    else:
        threshold_mask = valid_scores >= settings.SIMILARITY_THRESHOLD

    final_sources = index_map[valid_sources[threshold_mask]]
    final_targets = index_map[valid_targets[threshold_mask]]
    edges = np.column_stack((final_sources, final_targets)).astype(np.int32)
    edge_scores = valid_scores[threshold_mask]
    return filtered_headers, edges, edge_scores


__all__ = ["prepare_network"]
