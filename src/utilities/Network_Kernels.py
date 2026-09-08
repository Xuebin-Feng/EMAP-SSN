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

"""Shared dynamic-programming kernels, topology filtering, and network clustering."""

import sys
import numpy as np

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Module aliasing to prevent duplicate JIT caches in long-running processes
_THIS_MODULE = sys.modules[__name__]
if __name__ == "Network_Kernels":
    sys.modules.setdefault("utilities.Network_Kernels", _THIS_MODULE)
else:
    sys.modules.setdefault("Network_Kernels", _THIS_MODULE)


# ---------------------------------------------------------------------------
# Alignment Score and Length Dynamic Programming Kernels
# ---------------------------------------------------------------------------

@njit(nogil=True, fastmath=True, cache=True)
def global_score_length(score_matrix, gap_penalty):
    """Return linear-gap global alignment score and selected path length."""
    num_rows, num_cols = score_matrix.shape
    previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    previous_lengths = np.arange(num_cols + 1, dtype=np.uint32)
    current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)

    for col in range(1, num_cols + 1):
        previous_scores[col] = col * gap_penalty

    for row in range(1, num_rows + 1):
        current_scores[0] = row * gap_penalty
        current_lengths[0] = row

        for col in range(1, num_cols + 1):
            match = (
                previous_scores[col - 1]
                + score_matrix[row - 1, col - 1]
            )
            delete = previous_scores[col] + gap_penalty
            insert = current_scores[col - 1] + gap_penalty

            best_score = match
            best_length = previous_lengths[col - 1] + 1
            if delete > best_score:
                best_score = delete
                best_length = previous_lengths[col] + 1
            if insert > best_score:
                best_score = insert
                best_length = current_lengths[col - 1] + 1

            current_scores[col] = best_score
            current_lengths[col] = best_length

        previous_scores, current_scores = current_scores, previous_scores
        previous_lengths, current_lengths = (
            current_lengths,
            previous_lengths,
        )

    return previous_scores[num_cols], previous_lengths[num_cols]


@njit(nogil=True, fastmath=True, cache=True)
def local_score_length(score_matrix, gap_penalty, score_shift=2.0):
    """Return shifted linear-gap local alignment score and path length."""
    num_rows, num_cols = score_matrix.shape
    previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    previous_lengths = np.zeros(num_cols + 1, dtype=np.uint32)
    current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)

    max_score = 0.0
    max_length = np.uint32(0)

    for row in range(1, num_rows + 1):
        current_scores[0] = 0.0
        current_lengths[0] = 0

        for col in range(1, num_cols + 1):
            shifted_score = np.float32(
                score_matrix[row - 1, col - 1] - score_shift
            )
            match = previous_scores[col - 1] + shifted_score
            delete = previous_scores[col] + gap_penalty
            insert = current_scores[col - 1] + gap_penalty

            best_score = 0.0
            best_length = np.uint32(0)
            if match > best_score:
                best_score = match
                best_length = previous_lengths[col - 1] + 1
            if delete > best_score:
                best_score = delete
                best_length = previous_lengths[col] + 1
            if insert > best_score:
                best_score = insert
                best_length = current_lengths[col - 1] + 1

            current_scores[col] = best_score
            current_lengths[col] = best_length

            if best_score > max_score:
                max_score = best_score
                max_length = best_length

        previous_scores, current_scores = current_scores, previous_scores
        previous_lengths, current_lengths = (
            current_lengths,
            previous_lengths,
        )

    return max_score, max_length


@njit(nogil=True, fastmath=True, cache=True)
def global_local_scores_reuse(
    score_matrix,
    global_gap_penalty,
    local_gap_penalty,
    global_previous_scores,
    global_current_scores,
    global_previous_lengths,
    global_current_lengths,
    local_previous_scores,
    local_current_scores,
    local_previous_lengths,
    local_current_lengths,
    local_score_shift=2.0,
):
    """
    Return global and local scores and lengths in one matrix traversal.

    The result order is ``global_score, global_length, local_score, local_length``.
    """
    num_rows, num_cols = score_matrix.shape

    # Every used cell is initialized so a thread can safely reuse arrays that
    # contain results from an earlier, longer alignment.
    for col in range(num_cols + 1):
        global_previous_scores[col] = col * global_gap_penalty
        global_current_scores[col] = 0.0
        global_previous_lengths[col] = col
        global_current_lengths[col] = 0
        local_previous_scores[col] = 0.0
        local_current_scores[col] = 0.0
        local_previous_lengths[col] = 0
        local_current_lengths[col] = 0

    max_local_score = 0.0
    max_local_length = np.uint32(0)

    for row in range(1, num_rows + 1):
        global_current_scores[0] = row * global_gap_penalty
        global_current_lengths[0] = row
        local_current_scores[0] = 0.0
        local_current_lengths[0] = 0

        for col in range(1, num_cols + 1):
            cell_score = score_matrix[row - 1, col - 1]

            global_match = (
                global_previous_scores[col - 1] + cell_score
            )
            global_delete = (
                global_previous_scores[col] + global_gap_penalty
            )
            global_insert = (
                global_current_scores[col - 1] + global_gap_penalty
            )

            best_global_score = global_match
            best_global_length = global_previous_lengths[col - 1] + 1
            if global_delete > best_global_score:
                best_global_score = global_delete
                best_global_length = global_previous_lengths[col] + 1
            if global_insert > best_global_score:
                best_global_score = global_insert
                best_global_length = global_current_lengths[col - 1] + 1

            global_current_scores[col] = best_global_score
            global_current_lengths[col] = best_global_length

            shifted_score = np.float32(cell_score - local_score_shift)
            local_match = (
                local_previous_scores[col - 1] + shifted_score
            )
            local_delete = (
                local_previous_scores[col] + local_gap_penalty
            )
            local_insert = (
                local_current_scores[col - 1] + local_gap_penalty
            )

            best_local_score = 0.0
            best_local_length = np.uint32(0)
            if local_match > best_local_score:
                best_local_score = local_match
                best_local_length = local_previous_lengths[col - 1] + 1
            if local_delete > best_local_score:
                best_local_score = local_delete
                best_local_length = local_previous_lengths[col] + 1
            if local_insert > best_local_score:
                best_local_score = local_insert
                best_local_length = local_current_lengths[col - 1] + 1

            local_current_scores[col] = best_local_score
            local_current_lengths[col] = best_local_length

            if best_local_score > max_local_score:
                max_local_score = best_local_score
                max_local_length = best_local_length

        global_previous_scores, global_current_scores = (
            global_current_scores,
            global_previous_scores,
        )
        global_previous_lengths, global_current_lengths = (
            global_current_lengths,
            global_previous_lengths,
        )
        local_previous_scores, local_current_scores = (
            local_current_scores,
            local_previous_scores,
        )
        local_previous_lengths, local_current_lengths = (
            local_current_lengths,
            local_previous_lengths,
        )

    return (
        global_previous_scores[num_cols],
        global_previous_lengths[num_cols],
        max_local_score,
        max_local_length,
    )


@njit(nogil=True, fastmath=True, cache=True)
def global_local_scores(
    score_matrix,
    global_gap_penalty,
    local_gap_penalty,
    local_score_shift=2.0,
):
    """Compatibility wrapper using one-call dynamic-programming scratch."""
    num_cols = score_matrix.shape[1]
    global_previous_scores = np.empty(num_cols + 1, dtype=np.float32)
    global_current_scores = np.empty(num_cols + 1, dtype=np.float32)
    global_previous_lengths = np.empty(num_cols + 1, dtype=np.uint32)
    global_current_lengths = np.empty(num_cols + 1, dtype=np.uint32)
    local_previous_scores = np.empty(num_cols + 1, dtype=np.float32)
    local_current_scores = np.empty(num_cols + 1, dtype=np.float32)
    local_previous_lengths = np.empty(num_cols + 1, dtype=np.uint32)
    local_current_lengths = np.empty(num_cols + 1, dtype=np.uint32)
    return global_local_scores_reuse(
        score_matrix,
        global_gap_penalty,
        local_gap_penalty,
        global_previous_scores,
        global_current_scores,
        global_previous_lengths,
        global_current_lengths,
        local_previous_scores,
        local_current_scores,
        local_previous_lengths,
        local_current_lengths,
        local_score_shift,
    )


class GlobalLocalScratch:
    """Grow-only scratch owned by one CPU alignment worker thread."""

    def __init__(self):
        self.capacity = 0
        self.growths = 0
        self.arrays = None

    def _ensure(self, columns):
        required = max(1, int(columns) + 1)
        if required <= self.capacity:
            return False
        # Geometric growth avoids repeatedly reallocating for gradually
        # increasing sequence lengths while keeping each worker bounded.
        self.capacity = max(required, max(64, self.capacity * 2))
        self.arrays = (
            np.empty(self.capacity, dtype=np.float32),
            np.empty(self.capacity, dtype=np.float32),
            np.empty(self.capacity, dtype=np.uint32),
            np.empty(self.capacity, dtype=np.uint32),
            np.empty(self.capacity, dtype=np.float32),
            np.empty(self.capacity, dtype=np.float32),
            np.empty(self.capacity, dtype=np.uint32),
            np.empty(self.capacity, dtype=np.uint32),
        )
        self.growths += 1
        return True

    def score(
        self,
        score_matrix,
        global_gap_penalty,
        local_gap_penalty,
        local_score_shift=2.0,
    ):
        grew = self._ensure(score_matrix.shape[1])
        result = global_local_scores_reuse(
            score_matrix,
            global_gap_penalty,
            local_gap_penalty,
            *self.arrays,
            local_score_shift,
        )
        return result, grew


# ---------------------------------------------------------------------------
# Network Topology Filtering and Community Detection (Clustering)
# ---------------------------------------------------------------------------

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


__all__ = [
    "NUMBA_AVAILABLE",
    "global_score_length",
    "local_score_length",
    "global_local_scores_reuse",
    "global_local_scores",
    "GlobalLocalScratch",
    "leiden_partition",
    "fast_jaccard_filter",
]
