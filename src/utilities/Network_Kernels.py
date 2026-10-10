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
    import numba
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

try:
    from utilities.Localization import Message
except ImportError:
    from Localization import Message


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

def _first_appearance_order(edges, n_nodes):
    """Return the nodes of ``edges`` in the order they first appear in
    ``(u0, v0, u1, v1, ...)``."""
    flat = edges.ravel()
    first = np.full(n_nodes, flat.size, dtype=np.int64)
    np.minimum.at(first, flat, np.arange(flat.size, dtype=np.int64))
    present = np.flatnonzero(first < flat.size)
    # First positions are distinct, so any sort gives the same order.
    return present[np.argsort(first[present])]


def _leiden_csr_input(edges, weights, n_nodes):
    """Return ``(nodes, indptr, indices, data)`` describing to
    ``graspologic_native.leiden_csr`` the network that ``leiden`` builds
    from the string edge list, or None when the edges repeat a pair or hold
    a self-loop (whose handling the list path keeps).

    ``leiden`` numbers nodes by first appearance in ``(u0, v0, u1, v1, ...)``
    and orders each node's neighbours by that number. Given that numbering
    and order, ``leiden_csr`` with the same seed returns the same partition
    and community ids, without building a Python string per endpoint.
    """
    import scipy.sparse as sp

    nodes = _first_appearance_order(edges, n_nodes)
    local = np.empty(n_nodes, dtype=np.int64)
    local[nodes] = np.arange(nodes.size, dtype=np.int64)
    sources = local[edges[:, 0]]
    targets = local[edges[:, 1]]
    # csr_matrix sorts each row and merges repeated pairs (a self-loop's two
    # half-edges as well), which the count check below detects.
    graph = sp.csr_matrix(
        (
            np.concatenate((weights, weights)),
            (np.concatenate((sources, targets)), np.concatenate((targets, sources))),
        ),
        shape=(nodes.size, nodes.size),
    )
    if graph.nnz != 2 * edges.shape[0]:
        return None
    return (
        nodes,
        graph.indptr.astype(np.int64),
        graph.indices.astype(np.int32, copy=False),
        graph.data,
    )


def _call_graspologic(function, *args, **kwargs):
    """Call a graspologic_native function, turning its panics into RuntimeError.

    pyo3 raises a Rust panic as PanicException, a BaseException that escapes
    every ``except Exception`` of the callers (the command dispatcher, the MCP
    portal, the VR terminal). Ordinary exceptions, KeyboardInterrupt,
    SystemExit and GeneratorExit pass through unchanged.
    """
    try:
        return function(*args, **kwargs)
    except Exception:
        raise
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except BaseException as error:
        raise RuntimeError(
            Message("Leiden clustering failed: {error}", error=str(error))
        ) from error


def _label_isolated_nodes(labels, isolated, min_size):
    """Label each node of the ``isolated`` mask as a singleton cluster, in node
    order after the clusters ``labels`` already holds, when ``min_size`` is 1;
    with a larger ``min_size`` a lone node is too small and stays Noise."""
    if min_size <= 1 and isolated.any():
        first = int(labels.max(initial=0)) + 1
        labels[isolated] = np.arange(first, first + int(isolated.sum()))
    return labels


def leiden_partition(n_nodes, edges, weights, resolution, min_size, seed=42):
    """Partition a network with Leiden and return 1-based cluster labels.

    A node with no edge is a community of one: a singleton cluster when
    ``min_size`` is 1, like the Jaccard and MCL partitions, and Noise (-1)
    otherwise. Such nodes are numbered after the other clusters, in node order.

    Raises ValueError for a NaN or infinite resolution or edge weight, which
    would panic graspologic_native, and RuntimeError when it panics anyway.
    """
    import graspologic_native as gn

    if not np.isfinite(float(resolution)):
        raise ValueError(
            Message(
                "Leiden resolution must be a finite number; got {resolution}.",
                resolution=resolution,
            )
        )

    labels = np.full(n_nodes, -1, dtype=int)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if edges.shape[0] == 0:
        return _label_isolated_nodes(labels, np.ones(n_nodes, dtype=bool), min_size)

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

    if not np.isfinite(weights).all():
        raise ValueError(Message("Leiden edge weights must be finite numbers."))

    # leiden_csr refuses negative and non-finite weights, which leiden takes.
    csr_input = None
    if hasattr(gn, "leiden_csr") and np.all((weights >= 0) & (weights < np.inf)):
        csr_input = _leiden_csr_input(edges, weights, n_nodes)
    if csr_input is not None:
        nodes, indptr, indices, data = csr_input
        _, membership = _call_graspologic(
            gn.leiden_csr,
            indptr,
            indices,
            data,
            int(nodes.size),
            resolution=float(resolution),
            use_modularity=True,
            seed=int(seed),
        )
        members = nodes[
            np.fromiter(membership.keys(), dtype=np.int64, count=len(membership))
        ]
    else:
        edge_list = list(
            zip(
                map(str, edges[:, 0].tolist()),
                map(str, edges[:, 1].tolist()),
                weights.tolist(),
            )
        )
        _, membership = _call_graspologic(
            gn.leiden,
            edges=edge_list,
            resolution=float(resolution),
            use_modularity=True,
            seed=int(seed),
        )
        members = np.fromiter(
            map(int, membership.keys()), dtype=np.int64, count=len(membership)
        )
    communities = np.fromiter(
        membership.values(), dtype=np.int64, count=len(membership)
    )

    # Communities of at least min_size members are numbered 1, 2, ... in
    # community-id order; smaller ones are Noise.
    sizes = np.bincount(communities)
    kept = (sizes > 0) & (sizes >= min_size)
    cluster_ids = np.where(kept, np.cumsum(kept), -1)
    labels[members] = cluster_ids[communities]

    # graspologic_native returns only the nodes that appear in an edge. The
    # others are isolated: singleton clusters when min_size is 1, else Noise.
    connected = np.zeros(n_nodes, dtype=bool)
    connected[edges[:, 0]] = True
    connected[edges[:, 1]] = True
    return _label_isolated_nodes(labels, ~connected, min_size)


def sorted_neighbour_csr(edges, n_nodes):
    """Return int32 ``(indptr, indices)`` listing each edge at both of its
    endpoints, every row's neighbours in ascending order (repeated pairs and
    self-loops are kept, as repeated entries)."""
    edges = np.asarray(edges).reshape(-1, 2)
    sources = np.concatenate((edges[:, 0], edges[:, 1])).astype(np.int64)
    targets = np.concatenate((edges[:, 1], edges[:, 0])).astype(np.int64)
    keys = sources * n_nodes + targets
    keys.sort()
    indices = (keys % max(n_nodes, 1)).astype(np.int32)
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(sources, minlength=n_nodes))
    return indptr, indices


def jaccard_partition(n_nodes, edges, threshold, min_size):
    """Label the connected components left by ``fast_jaccard_filter``.

    Edges are kept by the Jaccard index of their endpoints' closed
    neighbourhoods. Components with fewer than ``min_size`` nodes are Noise
    (-1); the others get distinct positive labels in no particular order, so
    callers number them with ``renumber_clusters_by_size``.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    indptr, indices = sorted_neighbour_csr(edges, n_nodes)
    keep_mask = _jaccard_filter_own_csr(edges, indptr, indices, threshold)
    kept = edges[keep_mask]
    graph = coo_matrix(
        (np.ones(kept.shape[0], dtype=bool), (kept[:, 0], kept[:, 1])),
        shape=(n_nodes, n_nodes),
    )
    _count, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    return np.where(sizes[components] >= min_size, components + 1, -1)


def _mcl_add_self_loops(matrix):
    """Give every node a self-loop as heavy as its strongest edge, so the
    loops scale with the edge weights (``markov_clustering.add_self_loops``
    gives every node the same fixed weight, whatever the scale of the scores),
    for the canonical CSR adjacency the commands build and without the
    library's DOK round trip.

    A node's loop weighs the largest entry of its column outside the
    diagonal, or 1 for a node with no edge, so that its column normalises to
    itself. A diagonal entry the matrix already holds is replaced, not added
    to, and plays no part in the maximum.

    Returns the CSC arrays ``add_self_loops`` returns for a sparse matrix:
    each column's rows ascending, its diagonal entry replaced, or appended
    last when the column had none.
    """
    import scipy.sparse as sp

    csc = matrix.tocsc()
    n_nodes = csc.shape[0]
    indptr, indices = csc.indptr, csc.indices
    data = csc.data.copy()
    columns = np.repeat(
        np.arange(n_nodes, dtype=indices.dtype), np.diff(indptr)
    )
    diagonal = indices == columns

    # A column's entries are contiguous, so the maximum of every column that
    # has an off-diagonal entry is a reduction over that column's run.
    others = ~diagonal
    other_columns = columns[others]
    other_data = data[others]
    loops = np.ones(n_nodes, dtype=data.dtype)
    if other_data.size:
        run_starts = np.flatnonzero(
            np.concatenate(([True], other_columns[1:] != other_columns[:-1]))
        )
        loops[other_columns[run_starts]] = np.maximum.reduceat(
            other_data, run_starts
        )

    data[diagonal] = loops[columns[diagonal]]
    missing = np.ones(n_nodes, dtype=bool)
    missing[columns[diagonal]] = False
    positions = indptr[1:][missing]
    added = np.flatnonzero(missing).astype(indices.dtype)
    new_indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    new_indptr[1:] = indptr[1:] + np.cumsum(missing)
    return sp.csc_matrix(
        (
            np.insert(data, positions, loops[added]),
            np.insert(indices, positions, added),
            new_indptr,
        ),
        shape=csc.shape,
    )


def _mcl_prune(matrix, threshold):
    """``markov_clustering.prune`` for the canonical CSC matrices MCL
    iterates on, without its DOK round trip: the entries >= threshold in
    their order, as float64 (the DOK's dtype), with each column's maximum
    restored by the library's own statements."""
    import scipy.sparse as sp

    # A no-op for the inflated matrices MCL prunes; it guarantees the sorted
    # rows the DOK round trip produced.
    matrix.sum_duplicates()
    keep = matrix.data >= threshold
    kept_before = np.zeros(matrix.data.size + 1, dtype=np.int64)
    np.cumsum(keep, out=kept_before[1:])
    pruned = sp.csc_matrix(
        (
            matrix.data[keep].astype(np.float64),
            matrix.indices[keep],
            kept_before[matrix.indptr],
        ),
        shape=matrix.shape,
    )
    num_cols = matrix.shape[1]
    row_indices = _mcl_column_argmax(matrix)
    if row_indices is None:
        row_indices = matrix.argmax(axis=0).reshape((num_cols,))
    col_indices = np.arange(num_cols)
    pruned[row_indices, col_indices] = matrix[row_indices, col_indices]
    return pruned


def _mcl_column_argmax(matrix):
    """``matrix.argmax(axis=0)`` as an array for a canonical CSC matrix whose
    columns all hold a positive maximum (MCL's do: the maximum is never
    pruned), else None. Like SciPy, it takes each column's first maximum in
    storage order."""
    indptr, data = matrix.indptr, matrix.data
    counts = np.diff(indptr)
    if data.size == 0 or not counts.all():
        return None
    column_max = np.maximum.reduceat(data, indptr[:-1])
    if not (column_max > 0).all():  # also refuses NaN
        return None
    max_positions = np.flatnonzero(data == np.repeat(column_max, counts))
    first = max_positions[np.searchsorted(max_positions, indptr[:-1])]
    return matrix.indices[first]


def _mcl_get_clusters(matrix):
    """``markov_clustering.get_clusters`` with one CSR conversion instead of
    a row extraction per attractor."""
    attractors = matrix.diagonal().nonzero()[0]
    rows = matrix.tocsr()
    indptr, indices, data = rows.indptr, rows.indices, rows.data
    clusters = set()
    for attractor in attractors.tolist():
        start, end = indptr[attractor], indptr[attractor + 1]
        members = indices[start:end][data[start:end] != 0]
        clusters.add(tuple(members.tolist()))
    return sorted(clusters)


def markov_clusters(matrix, inflation):
    """Return ``get_clusters(run_mcl(matrix, inflation=inflation))`` of
    ``markov_clustering``, with the library's defaults and arithmetic, except
    for the self-loops: each node's loop weighs its strongest edge (1 without
    an edge) instead of the library's fixed 1, so scaling every weight by a
    constant does not change the clusters.

    Its DOK-based pruning and cluster-extraction steps are replaced by array
    code producing identical matrices, which ``matrix`` (canonical CSR from
    ``csr_matrix((data, (row, col)))``) guarantees.
    """
    from markov_clustering import mcl

    matrix = mcl.normalize(_mcl_add_self_loops(matrix))
    for _iteration in range(100):
        last_mat = matrix.copy()
        matrix = mcl.iterate(matrix, 2, inflation)
        matrix = _mcl_prune(matrix, 0.001)
        if mcl.converged(matrix, last_mat):
            break
    return _mcl_get_clusters(matrix)


if NUMBA_AVAILABLE:

    def fast_jaccard_filter(edges, indptr, indices, threshold):
        """Keep the edges whose endpoints' closed neighbourhoods have a
        Jaccard index of at least ``threshold``.

        The closed neighbourhood N[x] of node x is its neighbours in the
        CSR rows (``indptr``, ``indices``, each row ascending) and x itself.
        For an edge (u, v) of a simple graph the index is
        (c + 2) / (|N(u)| + |N(v)| - c), c being the neighbours u and v share,
        so an isolated pair and a clique both score 1. A node already in its
        own row, through a self-loop, is not counted twice.
        """
        with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
            return _jaccard_keep_mask(edges, indptr, indices, threshold)

    def _jaccard_filter_own_csr(edges, indptr, indices, threshold):
        """``fast_jaccard_filter`` for the CSR ``sorted_neighbour_csr`` built
        from these very edges, faster on the simple graphs networks are.

        When no row lists a node twice, there is no repeated pair and no
        self-loop (each puts an entry in a row twice), so every edge (u, v) has
        v in N(u), u in N(v) and neither in its own row. Its closed index is
        then (c + 2) / (|N(u)| + |N(v)| - c), c being the neighbours u and v
        share: the same integers, so the same decision, as
        ``_jaccard_keep_mask``, which takes every other graph.
        """
        with Numba_Threads.limited_threads(Numba_Threads.default_thread_count()):
            if _rows_hold_no_repeats(indptr, indices):
                keep_half = _jaccard_half_edge_keep(
                    indptr, indices, threshold, numba.config.NUMBA_NUM_THREADS
                )
                return _jaccard_edge_keep(edges, indptr, indices, keep_half)
            return _jaccard_keep_mask(edges, indptr, indices, threshold)

    def _rows_hold_no_repeats(indptr, indices):
        """True when no row of the CSR (each row ascending) lists a node twice."""
        if indices.size < 2:
            return True
        repeats = indices[1:] == indices[:-1]
        # Two equal entries either side of a row boundary are in different rows.
        starts = np.asarray(indptr[1:-1], dtype=np.int64)
        repeats[starts[(starts > 0) & (starts < indices.size)] - 1] = False
        return not repeats.any()

    # Each node marks its neighbours once and, for every edge it anchors (the
    # endpoint with more neighbours, the lower index on a tie), counts the
    # other endpoint's neighbours it marked. The work is the smaller row per
    # edge, sum(min(du, dv)), where merging both rows costs sum(du + dv). The
    # decision lands on the anchor's half of the edge, its row's entry for
    # the other endpoint. Every node writes only its own row's entries, and
    # every thread marks in its own array.
    @njit(parallel=True, cache=True)
    def _jaccard_half_edge_keep(indptr, indices, threshold, n_threads):
        n_nodes = indptr.size - 1
        keep_half = np.zeros(indices.size, dtype=np.bool_)
        # A row per thread the pool may run, whatever the current limit.
        markers = np.zeros((n_threads, n_nodes), dtype=np.int32)
        for a in prange(n_nodes):
            start_a, end_a = indptr[a], indptr[a + 1]
            size_a = end_a - start_a
            marker = markers[numba.get_thread_id()]
            stamp = a + 1
            marked = False
            for p in range(start_a, end_a):
                b = indices[p]
                size_b = indptr[b + 1] - indptr[b]
                if size_a > size_b or (size_a == size_b and a < b):
                    if not marked:
                        for q in range(start_a, end_a):
                            marker[indices[q]] = stamp
                        marked = True
                    shared = 0
                    for q in range(indptr[b], indptr[b + 1]):
                        if marker[indices[q]] == stamp:
                            shared += 1
                    # a and b each add themselves, and each is in the other's row.
                    if ((shared + 2) / (size_a + size_b - shared)) >= threshold:
                        keep_half[p] = True
        return keep_half

    @njit(parallel=True, cache=True)
    def _jaccard_edge_keep(edges, indptr, indices, keep_half):
        """Each edge's decision, read from its anchor's half of it."""
        keep_mask = np.zeros(edges.shape[0], dtype=np.bool_)
        for edge_index in prange(edges.shape[0]):
            u, v = edges[edge_index, 0], edges[edge_index, 1]
            size_u = indptr[u + 1] - indptr[u]
            size_v = indptr[v + 1] - indptr[v]
            if size_u > size_v or (size_u == size_v and u < v):
                a, b = u, v
            else:
                a, b = v, u
            low, high = indptr[a], indptr[a + 1]
            while low < high:  # the position of b in a's ascending row
                middle = (low + high) // 2
                if indices[middle] < b:
                    low = middle + 1
                else:
                    high = middle
            keep_mask[edge_index] = keep_half[low]
        return keep_mask

    @njit(cache=True)
    def _sorted_row_count(indices, start, end, value):
        """How many entries of the ascending run indices[start:end] equal value."""
        row = indices[start:end]
        return np.searchsorted(row, value, side="right") - np.searchsorted(
            row, value, side="left"
        )

    # Every edge writes only its own mask entry, so edges run in parallel.
    @jit(nopython=True, parallel=True, cache=True)
    def _jaccard_keep_mask(edges, indptr, indices, threshold):
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=np.bool_)
        for edge_index in prange(n_edges):
            u, v = edges[edge_index, 0], edges[edge_index, 1]
            start_u, end_u = indptr[u], indptr[u + 1]
            start_v, end_v = indptr[v], indptr[v + 1]
            size_u, size_v = end_u - start_u, end_v - start_v

            # The rows' common entries: the open neighbourhoods' intersection.
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

            # Close the neighbourhoods: add each node to its own row unless a
            # self-loop has put it there already, then count what that adds to
            # the intersection (u in v's row, v in u's row).
            own_u = _sorted_row_count(indices, start_u, end_u, u)
            own_v = _sorted_row_count(indices, start_v, end_v, v)
            add_u = 1 if own_u == 0 else 0
            add_v = 1 if own_v == 0 else 0
            size_u += add_u
            size_v += add_v
            if u == v:
                intersection = size_u
            else:
                v_in_u = _sorted_row_count(indices, start_u, end_u, v)
                u_in_v = _sorted_row_count(indices, start_v, end_v, u)
                intersection += min(own_u + add_u, u_in_v) - min(own_u, u_in_v)
                intersection += min(v_in_u, own_v + add_v) - min(v_in_u, own_v)

            # Each closed neighbourhood holds its own node, so union >= 1.
            union = size_u + size_v - intersection
            if (intersection / union) >= threshold:
                keep_mask[edge_index] = True
        return keep_mask

else:

    def fast_jaccard_filter(edges, indptr, indices, threshold):
        """Keep the edges whose endpoints' closed neighbourhoods have a
        Jaccard index of at least ``threshold`` (see the Numba version)."""
        n_edges = edges.shape[0]
        keep_mask = np.zeros(n_edges, dtype=bool)
        for edge_index in range(n_edges):
            u, v = edges[edge_index]
            closed_u = set(indices[indptr[u] : indptr[u + 1]].tolist())
            closed_u.add(int(u))
            closed_v = set(indices[indptr[v] : indptr[v + 1]].tolist())
            closed_v.add(int(v))
            intersection = len(closed_u.intersection(closed_v))
            union = len(closed_u.union(closed_v))
            if (intersection / union) >= threshold:
                keep_mask[edge_index] = True
        return keep_mask

    _jaccard_filter_own_csr = fast_jaccard_filter


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
    "sorted_neighbour_csr",
    "jaccard_partition",
    "markov_clusters",
    "fast_jaccard_filter",
]
