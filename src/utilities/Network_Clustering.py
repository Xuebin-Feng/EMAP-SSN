# Copyright 2026 Xuebin Feng
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

"""Shared topology filtering and community-detection helpers."""

import numpy as np

try:
    from numba import jit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def leiden_partition(n_nodes, edges, weights, resolution, min_size, seed=42):
    """Partition a network with Leiden and return 1-based cluster labels."""
    import graspologic_native as gn

    labels = np.full(n_nodes, -1, dtype=int)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return labels

    if weights is not None and len(weights) != edges.shape[0]:
        print(
            "Warning: edge weight count does not match edge count; "
            "treating the network as unweighted."
        )
        weights = None

    if weights is None:
        weights = np.ones(edges.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).ravel()

    edge_list = [
        (str(int(u)), str(int(v)), float(weight))
        for (u, v), weight in zip(edges, weights)
    ]
    _, membership = gn.leiden(
        edges=edge_list,
        resolution=float(resolution),
        use_modularity=True,
        seed=int(seed),
    )

    communities = {}
    for node_string, community in membership.items():
        communities.setdefault(community, []).append(int(node_string))

    cluster_id = 1
    for community in sorted(communities):
        members = communities[community]
        if len(members) >= min_size:
            for node in members:
                labels[node] = cluster_id
            cluster_id += 1

    # Isolated nodes are always Noise, even when min_size == 1.
    connected = np.zeros(n_nodes, dtype=bool)
    connected[edges[:, 0]] = True
    connected[edges[:, 1]] = True
    labels[~connected] = -1
    return labels


if NUMBA_AVAILABLE:

    @jit(nopython=True)
    def fast_jaccard_filter(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=np.bool_)
        for edge_index in range(n_edges):
            u, v = edges[edge_index, 0], edges[edge_index, 1]
            start_u, end_u = indptr[u], indptr[u + 1]
            start_v, end_v = indptr[v], indptr[v + 1]
            size_u, size_v = end_u - start_u, end_v - start_v

            intersection = 0
            pointer_u, pointer_v = start_u, start_v
            while pointer_u < end_u and pointer_v < end_v:
                value_u, value_v = indices[pointer_u], indices[pointer_v]
                if value_u == value_v:
                    intersection += 1
                    pointer_u += 1
                    pointer_v += 1
                elif value_u < value_v:
                    pointer_u += 1
                else:
                    pointer_v += 1

            union = size_u + size_v - intersection
            if union > 0 and (intersection / union) >= threshold:
                keep_mask[edge_index] = True
        return keep_mask

else:

    def fast_jaccard_filter(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=bool)
        for edge_index in range(n_edges):
            u, v = edges[edge_index]
            neighbours_u = set(indices[indptr[u] : indptr[u + 1]])
            neighbours_v = set(indices[indptr[v] : indptr[v + 1]])
            intersection = len(neighbours_u.intersection(neighbours_v))
            union = len(neighbours_u.union(neighbours_v))
            if union > 0 and (intersection / union) >= threshold:
                keep_mask[edge_index] = True
        return keep_mask


__all__ = ["NUMBA_AVAILABLE", "fast_jaccard_filter", "leiden_partition"]
