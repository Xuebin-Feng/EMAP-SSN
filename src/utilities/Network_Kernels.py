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
    from numba import jit, njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


try:
    from utilities import Numba_Threads
except ImportError:
    import Numba_Threads


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
def global_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
):
    """
    Return linear-gap global score, path length and identities on that path.

    The length counts every column of the selected path, end gaps included.
    Identities count its aligned pairs whose row and column codes are equal.
    """
    num_rows, num_cols = score_matrix.shape
    previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    previous_lengths = np.arange(num_cols + 1, dtype=np.uint32)
    current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)
    previous_identities = np.zeros(num_cols + 1, dtype=np.uint32)
    current_identities = np.zeros(num_cols + 1, dtype=np.uint32)

    for col in range(1, num_cols + 1):
        previous_scores[col] = col * gap_penalty

    for row in range(1, num_rows + 1):
        current_scores[0] = row * gap_penalty
        current_lengths[0] = row
        current_identities[0] = 0
        query_code = query_codes[row - 1]

        for col in range(1, num_cols + 1):
            match = (
                previous_scores[col - 1]
                + score_matrix[row - 1, col - 1]
            )
            delete = previous_scores[col] + gap_penalty
            insert = current_scores[col - 1] + gap_penalty

            best_score = match
            best_length = previous_lengths[col - 1] + 1
            best_identities = previous_identities[col - 1] + (
                query_code == target_codes[col - 1]
            )
            if delete > best_score:
                best_score = delete
                best_length = previous_lengths[col] + 1
                best_identities = previous_identities[col]
            if insert > best_score:
                best_score = insert
                best_length = current_lengths[col - 1] + 1
                best_identities = current_identities[col - 1]

            current_scores[col] = best_score
            current_lengths[col] = best_length
            current_identities[col] = best_identities

        previous_scores, current_scores = current_scores, previous_scores
        previous_lengths, current_lengths = (
            current_lengths,
            previous_lengths,
        )
        previous_identities, current_identities = (
            current_identities,
            previous_identities,
        )

    return (
        previous_scores[num_cols],
        previous_lengths[num_cols],
        previous_identities[num_cols],
    )


@njit(nogil=True, fastmath=True, cache=True)
def local_score_length_identity(
    score_matrix,
    gap_penalty,
    query_codes,
    target_codes,
    score_shift=2.0,
):
    """
    Return shifted linear-gap local score, path length and path identities.

    With a non-positive gap penalty the path starts and ends on aligned pairs,
    so its length counts internal gaps only. Identities count its aligned
    pairs whose row and column codes are equal.
    """
    num_rows, num_cols = score_matrix.shape
    previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    previous_lengths = np.zeros(num_cols + 1, dtype=np.uint32)
    current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)
    previous_identities = np.zeros(num_cols + 1, dtype=np.uint32)
    current_identities = np.zeros(num_cols + 1, dtype=np.uint32)

    max_score = 0.0
    max_length = np.uint32(0)
    max_identities = np.uint32(0)

    for row in range(1, num_rows + 1):
        current_scores[0] = 0.0
        current_lengths[0] = 0
        current_identities[0] = 0
        query_code = query_codes[row - 1]

        for col in range(1, num_cols + 1):
            shifted_score = np.float32(
                score_matrix[row - 1, col - 1] - score_shift
            )
            match = previous_scores[col - 1] + shifted_score
            delete = previous_scores[col] + gap_penalty
            insert = current_scores[col - 1] + gap_penalty

            best_score = 0.0
            best_length = np.uint32(0)
            best_identities = np.uint32(0)
            if match > best_score:
                best_score = match
                best_length = previous_lengths[col - 1] + 1
                best_identities = previous_identities[col - 1] + (
                    query_code == target_codes[col - 1]
                )
            if delete > best_score:
                best_score = delete
                best_length = previous_lengths[col] + 1
                best_identities = previous_identities[col]
            if insert > best_score:
                best_score = insert
                best_length = current_lengths[col - 1] + 1
                best_identities = current_identities[col - 1]

            current_scores[col] = best_score
            current_lengths[col] = best_length
            current_identities[col] = best_identities

            if best_score > max_score:
                max_score = best_score
                max_length = best_length
                max_identities = best_identities

        previous_scores, current_scores = current_scores, previous_scores
        previous_lengths, current_lengths = (
            current_lengths,
            previous_lengths,
        )
        previous_identities, current_identities = (
            current_identities,
            previous_identities,
        )

    return max_score, max_length, max_identities


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


@njit(nogil=True, fastmath=True, cache=True)
def global_local_scores_batch(
    matrices,
    column_lengths,
    global_gap_penalty,
    local_gap_penalty,
    global_scores,
    global_lengths,
    local_scores,
    local_lengths,
    local_score_shift=2.0,
):
    """
    Score every pair of one padded microbatch with the per-pair kernel.

    ``matrices`` is ``(batch, rows, padded_columns)``; pair ``b`` uses its
    first ``column_lengths[b]`` columns. Results equal ``global_local_scores``
    on each unpadded matrix, and the scratch is allocated once per batch.
    """
    batch = matrices.shape[0]
    max_columns = matrices.shape[2]
    global_previous_scores = np.empty(max_columns + 1, dtype=np.float32)
    global_current_scores = np.empty(max_columns + 1, dtype=np.float32)
    global_previous_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    global_current_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    local_previous_scores = np.empty(max_columns + 1, dtype=np.float32)
    local_current_scores = np.empty(max_columns + 1, dtype=np.float32)
    local_previous_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    local_current_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    for index in range(batch):
        global_score, global_length, local_score, local_length = (
            global_local_scores_reuse(
                matrices[index, :, :column_lengths[index]],
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
        )
        global_scores[index] = global_score
        global_lengths[index] = global_length
        local_scores[index] = local_score
        local_lengths[index] = local_length


# Without 'nsz', LLVM may not exchange +0.0 and -0.0 when it turns the
# strict-greater selections into max instructions.
@njit(nogil=True, fastmath={"nnan", "ninf"}, cache=True)
def _float32_global_local_scores(
    score_matrix,
    global_gap,
    local_gap,
    local_score_shift,
    previous_global_scores,
    current_global_scores,
    previous_global_lengths,
    current_global_lengths,
    previous_local_scores,
    current_local_scores,
    previous_local_lengths,
    current_local_lengths,
):
    """
    Float32 ``global_local_scores_reuse`` for gaps accepted by
    ``float32_exact_local_bound``.

    Each cell evaluates the same candidates in the same order with the same
    strict ``>`` ties. Neighbours stay in registers and the selections are
    branch-free, which removes the float32/float64 conversion chain that
    limits the reference kernel.
    """
    num_rows, num_cols = score_matrix.shape
    zero = np.float32(0.0)
    for col in range(num_cols + 1):
        previous_global_scores[col] = np.float32(col) * global_gap
        previous_global_lengths[col] = col
        previous_local_scores[col] = zero
        previous_local_lengths[col] = 0

    max_local_score = zero
    max_local_length = np.int64(0)
    for row in range(1, num_rows + 1):
        left_global = np.float32(row) * global_gap
        left_global_length = np.int64(row)
        left_local = zero
        left_local_length = np.int64(0)
        current_global_scores[0] = left_global
        current_global_lengths[0] = row
        current_local_scores[0] = zero
        current_local_lengths[0] = 0
        diagonal_global = previous_global_scores[0]
        diagonal_global_length = np.int64(previous_global_lengths[0])
        diagonal_local = previous_local_scores[0]
        diagonal_local_length = np.int64(previous_local_lengths[0])

        for col in range(1, num_cols + 1):
            cell = score_matrix[row - 1, col - 1]
            up_global = previous_global_scores[col]
            up_global_length = np.int64(previous_global_lengths[col])
            up_local = previous_local_scores[col]
            up_local_length = np.int64(previous_local_lengths[col])

            global_delete = up_global + global_gap
            global_insert = left_global + global_gap
            best_global = diagonal_global + cell
            best_global_length = diagonal_global_length + 1
            take = global_delete > best_global
            best_global = global_delete if take else best_global
            best_global_length = (
                up_global_length + 1 if take else best_global_length
            )
            take = global_insert > best_global
            best_global = global_insert if take else best_global
            best_global_length = (
                left_global_length + 1 if take else best_global_length
            )
            current_global_scores[col] = best_global
            current_global_lengths[col] = best_global_length

            local_match = diagonal_local + (cell - local_score_shift)
            local_delete = up_local + local_gap
            local_insert = left_local + local_gap
            take = local_match > zero
            best_local = local_match if take else zero
            best_local_length = diagonal_local_length + 1 if take else 0
            take = local_delete > best_local
            best_local = local_delete if take else best_local
            best_local_length = (
                up_local_length + 1 if take else best_local_length
            )
            take = local_insert > best_local
            best_local = local_insert if take else best_local
            best_local_length = (
                left_local_length + 1 if take else best_local_length
            )
            current_local_scores[col] = best_local
            current_local_lengths[col] = best_local_length

            take = best_local > max_local_score
            max_local_score = best_local if take else max_local_score
            max_local_length = best_local_length if take else max_local_length

            diagonal_global = up_global
            diagonal_global_length = up_global_length
            diagonal_local = up_local
            diagonal_local_length = up_local_length
            left_global = best_global
            left_global_length = best_global_length
            left_local = best_local
            left_local_length = best_local_length

        previous_global_scores, current_global_scores = (
            current_global_scores,
            previous_global_scores,
        )
        previous_global_lengths, current_global_lengths = (
            current_global_lengths,
            previous_global_lengths,
        )
        previous_local_scores, current_local_scores = (
            current_local_scores,
            previous_local_scores,
        )
        previous_local_lengths, current_local_lengths = (
            current_local_lengths,
            previous_local_lengths,
        )

    return (
        previous_global_scores[num_cols],
        previous_global_lengths[num_cols],
        max_local_score,
        max_local_length,
    )


@njit(nogil=True, cache=True)
def _float32_global_local_scores_batch(
    matrices,
    column_lengths,
    global_gap,
    local_gap,
    local_bound,
    reference_global_gap,
    reference_local_gap,
    global_scores,
    global_lengths,
    local_scores,
    local_lengths,
):
    batch = matrices.shape[0]
    max_columns = matrices.shape[2]
    previous_global_scores = np.empty(max_columns + 1, dtype=np.float32)
    current_global_scores = np.empty(max_columns + 1, dtype=np.float32)
    previous_global_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    current_global_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    previous_local_scores = np.empty(max_columns + 1, dtype=np.float32)
    current_local_scores = np.empty(max_columns + 1, dtype=np.float32)
    previous_local_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    current_local_lengths = np.empty(max_columns + 1, dtype=np.uint32)
    local_score_shift = np.float32(2.0)
    for index in range(batch):
        matrix = matrices[index, :, :column_lengths[index]]
        global_score, global_length, local_score, local_length = (
            _float32_global_local_scores(
                matrix,
                global_gap,
                local_gap,
                local_score_shift,
                previous_global_scores,
                current_global_scores,
                previous_global_lengths,
                current_global_lengths,
                previous_local_scores,
                current_local_scores,
                previous_local_lengths,
                current_local_lengths,
            )
        )
        if local_score >= local_bound:
            # Outside the range where float32 is proven exact.
            global_score, global_length, reference_score, local_length = (
                global_local_scores_reuse(
                    matrix,
                    reference_global_gap,
                    reference_local_gap,
                    previous_global_scores,
                    current_global_scores,
                    previous_global_lengths,
                    current_global_lengths,
                    previous_local_scores,
                    current_local_scores,
                    previous_local_lengths,
                    current_local_lengths,
                    2.0,
                )
            )
            local_scores[index] = reference_score
        else:
            local_scores[index] = local_score
        global_scores[index] = global_score
        global_lengths[index] = global_length
        local_lengths[index] = local_length


def float32_exact_local_bound(global_gap_penalty, local_gap_penalty):
    """
    Return the local score below which float32 reproduces the reference DP,
    or ``None`` when these gaps need the reference kernel.

    The reference kernel adds float gap penalties in float64. With a zero
    global gap, every compared global value is already a float32. Local
    values are never negative; with a zero local gap, or a gap of -2**k,
    ``value + gap`` is exact in float32 whenever it is >= 0 (Sterbenz below
    2**(k+1), ulp multiples above, up to 2**(k+23)). An inexact sum is
    negative and loses to the zero floor in both precisions.
    """
    global_gap = float(global_gap_penalty)
    local_gap = float(local_gap_penalty)
    if global_gap != 0.0 or local_gap > 0.0:
        return None
    if local_gap == 0.0:
        return float(np.finfo(np.float32).max)
    mantissa, exponent = np.frexp(-local_gap)
    if mantissa != 0.5:
        return None
    # -local_gap == 2**(exponent - 1); keep a factor-two margin.
    return float(2.0 ** (exponent - 1 + 22))


def alignment_batch_scores(
    matrices,
    column_lengths,
    global_gap_penalty,
    local_gap_penalty,
):
    """
    Return global and local scores and lengths for one padded microbatch.

    The arrays are ``global_score`` (float32), ``global_length`` (uint32),
    ``local_score`` (float64) and ``local_length`` (uint32), equal to
    ``global_local_scores`` on each pair. Gaps that make float32 exact use the
    faster float32 kernel.
    """
    column_lengths = np.asarray(column_lengths, dtype=np.int64)
    batch = len(column_lengths)
    global_scores = np.empty(batch, dtype=np.float32)
    global_lengths = np.empty(batch, dtype=np.uint32)
    local_scores = np.empty(batch, dtype=np.float64)
    local_lengths = np.empty(batch, dtype=np.uint32)
    bound = float32_exact_local_bound(global_gap_penalty, local_gap_penalty)
    if bound is None:
        global_local_scores_batch(
            matrices,
            column_lengths,
            global_gap_penalty,
            local_gap_penalty,
            global_scores,
            global_lengths,
            local_scores,
            local_lengths,
        )
    else:
        _float32_global_local_scores_batch(
            matrices,
            column_lengths,
            np.float32(global_gap_penalty),
            np.float32(local_gap_penalty),
            np.float32(bound),
            global_gap_penalty,
            local_gap_penalty,
            global_scores,
            global_lengths,
            local_scores,
            local_lengths,
        )
    return global_scores, global_lengths, local_scores, local_lengths


def align_microbatch(
    matrices,
    row_index,
    target_indices,
    target_lengths,
    global_gap_penalty,
    local_gap_penalty,
):
    """
    Align one padded accelerator microbatch on the calling CPU thread.

    Returns ``(row, target, local_score, local_length, global_score,
    global_length)`` per pair, the tuple produced by the tools' per-pair
    alignment callbacks.
    """
    if not np.isfinite(matrices).all():
        raise FloatingPointError(
            "Batched accelerator scoring produced non-finite values."
        )
    global_scores, global_lengths, local_scores, local_lengths = (
        alignment_batch_scores(
            matrices,
            target_lengths,
            global_gap_penalty,
            local_gap_penalty,
        )
    )
    row_index = int(row_index)
    return list(
        zip(
            [row_index] * len(global_scores),
            [int(index) for index in target_indices],
            local_scores.tolist(),
            local_lengths.tolist(),
            global_scores.tolist(),
            global_lengths.tolist(),
        )
    )


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

    def fast_jaccard_filter(edges, indptr, indices, threshold):
        """Keep edges whose endpoint neighbourhoods meet the Jaccard threshold."""
        with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
            return _jaccard_keep_mask(edges, indptr, indices, threshold)

    # Every edge writes only its own mask entry, so edges run in parallel.
    @jit(nopython=True, parallel=True)
    def _jaccard_keep_mask(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=np.bool_)
        for edge_index in prange(n_edges):
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
    "global_score_length_identity",
    "local_score_length_identity",
    "global_local_scores_reuse",
    "global_local_scores",
    "global_local_scores_batch",
    "float32_exact_local_bound",
    "alignment_batch_scores",
    "align_microbatch",
    "leiden_partition",
    "fast_jaccard_filter",
]
