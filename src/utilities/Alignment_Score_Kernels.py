"""
Shared score-and-length dynamic-programming kernels.

These kernels intentionally do not retain traceback paths. They use rolling
score and length rows so their scratch memory grows linearly with the second
sequence length.
"""

import numpy as np
from numba import njit


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
def global_local_scores(
    score_matrix,
    global_gap_penalty,
    local_gap_penalty,
    local_score_shift=2.0,
):
    """
    Return global and local scores and lengths in one matrix traversal.

    The result order is ``global_score, global_length, local_score,
    local_length``.
    """
    num_rows, num_cols = score_matrix.shape

    global_previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    global_current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    global_previous_lengths = np.arange(
        num_cols + 1,
        dtype=np.uint32,
    )
    global_current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)

    local_previous_scores = np.zeros(num_cols + 1, dtype=np.float32)
    local_current_scores = np.zeros(num_cols + 1, dtype=np.float32)
    local_previous_lengths = np.zeros(num_cols + 1, dtype=np.uint32)
    local_current_lengths = np.zeros(num_cols + 1, dtype=np.uint32)

    for col in range(1, num_cols + 1):
        global_previous_scores[col] = col * global_gap_penalty

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
