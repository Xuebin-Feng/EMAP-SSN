import os
import statistics
import sys
import time

import numpy as np
from numba import njit


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if UTILITIES_DIR not in sys.path:
    sys.path.insert(0, UTILITIES_DIR)

from Alignment_Score_Kernels import global_local_scores
from commands.logo import (
    _calculate_identity_neighbour_counts_numpy,
    _encode_standard_amino_acids,
    run_identity_neighbour_counts,
)


@njit(nogil=True, fastmath=True)
def baseline_global_score_length(score_matrix, gap_penalty):
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


@njit(nogil=True, fastmath=True)
def legacy_local_traceback(score_matrix, gap_penalty):
    num_rows, num_cols = score_matrix.shape
    scores = np.zeros(
        (num_rows + 1, num_cols + 1),
        dtype=np.float32,
    )
    pointers = np.zeros(
        (num_rows + 1, num_cols + 1),
        dtype=np.int8,
    )
    max_score = 0.0
    max_row, max_col = 0, 0

    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            match = (
                scores[row - 1, col - 1]
                + score_matrix[row - 1, col - 1]
            )
            delete = scores[row - 1, col] + gap_penalty
            insert = scores[row, col - 1] + gap_penalty
            best_score = 0.0
            pointer = 0
            if match > best_score:
                best_score = match
                pointer = 1
            if delete > best_score:
                best_score = delete
                pointer = 2
            if insert > best_score:
                best_score = insert
                pointer = 3
            scores[row, col] = best_score
            pointers[row, col] = pointer
            if best_score > max_score:
                max_score = best_score
                max_row = row
                max_col = col

    path_length = 0
    row, col = max_row, max_col
    while row > 0 and col > 0:
        if scores[row, col] == 0:
            break
        pointer = pointers[row, col]
        if pointer == 0:
            break
        path_length += 1
        if pointer == 1:
            row -= 1
            col -= 1
        elif pointer == 2:
            row -= 1
        elif pointer == 3:
            col -= 1
        else:
            break

    return max_score, path_length


def baseline_alignment(matrix):
    working_matrix = matrix.copy()
    global_score, global_length = baseline_global_score_length(
        working_matrix,
        0.0,
    )
    working_matrix -= 2.0
    local_score, local_length = legacy_local_traceback(
        working_matrix,
        -2.0,
    )
    return global_score, global_length, local_score, local_length


def benchmark(function, matrix, repeats=5):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(matrix)
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


rng = np.random.default_rng(20260729)
warmup = rng.normal(size=(8, 8)).astype(np.float32)
baseline_alignment(warmup)
global_local_scores(warmup, 0.0, -2.0)

for size in (256, 512, 1024):
    matrix = rng.normal(size=(size, size)).astype(np.float32)
    baseline_seconds = benchmark(baseline_alignment, matrix)
    shared_seconds = benchmark(
        lambda values: global_local_scores(values, 0.0, -2.0),
        matrix,
    )
    speedup = baseline_seconds / shared_seconds
    print(
        f"{size}x{size}: baseline={baseline_seconds:.6f}s "
        f"shared={shared_seconds:.6f}s speedup={speedup:.2f}x"
    )


print("\nLogo identity-neighbour kernel (non-gating benchmark)")
amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
logo_rows = rng.choice(amino_acids, size=(2000, 300))
logo_sequences = ["".join(row) for row in logo_rows]
logo_encoded = _encode_standard_amino_acids(logo_sequences)
logo_multiplicities = np.ones(len(logo_sequences), dtype=np.int64)

# Exclude JIT compilation from the timed accelerated measurement.
run_identity_neighbour_counts(
    logo_encoded[:2],
    logo_multiplicities[:2],
    0.9,
)
started = time.perf_counter()
numpy_counts = _calculate_identity_neighbour_counts_numpy(
    logo_encoded,
    logo_multiplicities,
    0.9,
)
numpy_seconds = time.perf_counter() - started
started = time.perf_counter()
numba_counts, logo_threads = run_identity_neighbour_counts(
    logo_encoded,
    logo_multiplicities,
    0.9,
)
numba_seconds = time.perf_counter() - started
np.testing.assert_array_equal(numba_counts, numpy_counts)
logo_speedup = numpy_seconds / numba_seconds
print(
    f"2000x300: numpy={numpy_seconds:.6f}s "
    f"numba={numba_seconds:.6f}s speedup={logo_speedup:.2f}x "
    f"threads={logo_threads}"
)
if logo_speedup < 10.0:
    raise RuntimeError(
        f"Logo identity kernel speedup {logo_speedup:.2f}x is below 10x"
    )
