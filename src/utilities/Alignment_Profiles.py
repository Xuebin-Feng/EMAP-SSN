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

"""What an alignment's columns say, computed from its aligned rows.

The sequence-logo letter heights, the identity-neighbour weights that
reweight redundant sequences, and the helpers that encode and decode aligned
rows for them. They were part of the `logo` command; `logo` and `label` use
them from here, and so does the residue data `esmfold` writes
(utilities.Residue_Data). The module is Qt-free and reads no settings, so the
Viewer, the VR viewer and the command scripts can all import it.
"""

import os

import numpy as np

from utilities.Job_Cancellation import raise_if_cancelled

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    get_num_threads = None
    set_num_threads = None


STANDARD_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")


def choose_balanced_thread_count(configured_threads, logical_cpus=None):
    """Reserve two logical CPUs when possible without exceeding Numba's limit."""
    configured_threads = max(1, int(configured_threads))
    if logical_cpus is None:
        logical_cpus = os.cpu_count() or 1
    logical_cpus = max(1, int(logical_cpus))
    available_for_kernel = max(1, logical_cpus - 2)
    return max(1, min(configured_threads, available_for_kernel))


def get_balanced_thread_count():
    """Return the balanced count for the calling thread's Numba configuration."""
    if not NUMBA_AVAILABLE:
        return 0
    return choose_balanced_thread_count(get_num_threads())


_IDENTITY_TILE_ROWS = 32
_IDENTITY_CHUNKS_PER_THREAD = 4

if NUMBA_AVAILABLE:

    @njit(inline="always")
    def _popcount64(word):
        # LLVM recognises this SWAR idiom and emits popcnt / vpshufb.
        word = word - ((word >> np.uint64(1)) & np.uint64(0x5555555555555555))
        word = (word & np.uint64(0x3333333333333333)) + (
            (word >> np.uint64(2)) & np.uint64(0x3333333333333333)
        )
        word = (word + (word >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        return np.int64((word * np.uint64(0x0101010101010101)) >> np.uint64(56))

    @njit(parallel=True, nogil=True, cache=True)
    def _identity_neighbour_counts_kernel(
        planes, valid_counts, multiplicities, threshold, chunk_count
    ):
        """Count threshold neighbours for every unique encoded sequence.

        planes[u, 0] is a bit set of u's valid (standard residue) columns and
        planes[u, 1:6] hold the five bits of its 0-19 residue code. Rows are
        sorted by valid count, so a pair whose shorter row cannot reach the
        threshold even with every overlapping column matching the longer one
        is skipped: match <= valid_i and union >= valid_j. The exact union
        comes from the validity planes alone, and since match <= overlap a
        pair whose overlap already fails the threshold skips the residue
        planes. Each unordered pair is evaluated once and credited to both
        rows in a per-chunk accumulator, so the integer counts equal the full
        n x n scan.
        """
        unique_count = planes.shape[0]
        word_count = planes.shape[2]
        tile_count = (unique_count + _IDENTITY_TILE_ROWS - 1) // _IDENTITY_TILE_ROWS
        partial = np.zeros((chunk_count, unique_count), dtype=np.int64)

        for chunk in prange(chunk_count):
            accumulator = partial[chunk]
            for tile in range(chunk, tile_count, chunk_count):
                tile_start = tile * _IDENTITY_TILE_ROWS
                tile_end = min(tile_start + _IDENTITY_TILE_ROWS, unique_count)
                longest_in_tile = valid_counts[tile_end - 1]
                right_end = tile_end
                while right_end < unique_count and not (
                    longest_in_tile
                    < threshold * valid_counts[right_end] - 1e-12
                ):
                    right_end += 1

                for right in range(tile_start, right_end):
                    right_planes = planes[right]
                    right_valid = valid_counts[right]
                    for left in range(tile_start, min(tile_end, right + 1)):
                        left_valid = valid_counts[left]
                        if left_valid < threshold * right_valid - 1e-12:
                            continue
                        left_planes = planes[left]
                        overlap_count = 0
                        for word in range(word_count):
                            overlap_count += _popcount64(
                                left_planes[0, word] & right_planes[0, word]
                            )
                        union_count = left_valid + right_valid - overlap_count
                        # match <= overlap: skip pairs that fail even if every
                        # overlapping column matched.
                        if (
                            union_count <= 0
                            or overlap_count < threshold * union_count - 1e-12
                        ):
                            continue
                        match_count = 0
                        for word in range(word_count):
                            both_valid = left_planes[0, word] & right_planes[0, word]
                            different = (
                                (left_planes[1, word] ^ right_planes[1, word])
                                | (left_planes[2, word] ^ right_planes[2, word])
                                | (left_planes[3, word] ^ right_planes[3, word])
                                | (left_planes[4, word] ^ right_planes[4, word])
                                | (left_planes[5, word] ^ right_planes[5, word])
                            )
                            match_count += _popcount64(both_valid & ~different)
                        if match_count >= threshold * union_count - 1e-12:
                            accumulator[left] += multiplicities[right]
                            if left != right:
                                accumulator[right] += multiplicities[left]

        neighbour_counts = np.zeros(unique_count, dtype=np.int64)
        for chunk in range(chunk_count):
            for index in range(unique_count):
                neighbour_counts[index] += partial[chunk, index]
        # An all-invalid row has undefined identity. It remains one
        # independent observation, matching the historical NumPy path.
        for index in range(unique_count):
            if neighbour_counts[index] == 0:
                neighbour_counts[index] = 1
        return neighbour_counts


def _pack_identity_planes(encoded, block_rows=4096):
    """Pack (n, L) codes (-1 invalid, 0-19 residue) into (n, 6, W) uint64 bit planes.

    Plane 0 marks valid columns; planes 1-5 hold the code's bits (0 where
    invalid). Column c is bit c % 64 of word c // 64. Rows are packed in blocks
    to bound temporary memory.
    """
    sequence_count, alignment_length = encoded.shape
    word_count = max(1, (alignment_length + 63) // 64)
    planes = np.empty((sequence_count, 6, word_count), dtype=np.uint64)
    valid_counts = np.empty(sequence_count, dtype=np.int64)
    for start in range(0, sequence_count, block_rows):
        raise_if_cancelled()
        stop = min(sequence_count, start + block_rows)
        padded = np.full((stop - start, word_count * 64), -1, dtype=np.int8)
        padded[:, :alignment_length] = encoded[start:stop]
        valid = padded >= 0
        valid_counts[start:stop] = valid.sum(axis=1, dtype=np.int64)
        codes = np.where(valid, padded, 0).astype(np.uint8)
        bit_rows = [valid] + [((codes >> bit) & 1).astype(bool) for bit in range(5)]
        for plane_index, bit_row in enumerate(bit_rows):
            packed = np.packbits(
                bit_row.reshape(stop - start, word_count, 64),
                axis=2,
                bitorder="little",
            )
            planes[start:stop, plane_index, :] = packed.view("<u8").reshape(
                stop - start, word_count
            )
    return planes, valid_counts


def run_identity_neighbour_counts(encoded, multiplicities, threshold):
    """Run the exact kernel with balanced threads and restore thread settings."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    encoded = np.ascontiguousarray(encoded, dtype=np.int8)
    multiplicities = np.ascontiguousarray(multiplicities, dtype=np.int64)
    planes, valid_counts = _pack_identity_planes(encoded)
    order = np.argsort(valid_counts, kind="stable")

    previous_threads = get_num_threads()
    selected_threads = choose_balanced_thread_count(previous_threads)
    if selected_threads != previous_threads:
        set_num_threads(selected_threads)

    try:
        sorted_counts = _identity_neighbour_counts_kernel(
            np.ascontiguousarray(planes[order]),
            np.ascontiguousarray(valid_counts[order]),
            np.ascontiguousarray(multiplicities[order]),
            float(threshold),
            _IDENTITY_CHUNKS_PER_THREAD * selected_threads,
        )
    finally:
        if selected_threads != previous_threads:
            set_num_threads(previous_threads)

    counts = np.empty(len(order), dtype=np.int64)
    counts[order] = sorted_counts
    return counts, selected_threads


_ASCII_TO_STANDARD_CODE = np.full(256, -1, dtype=np.int8)
for _code, _amino_acid in enumerate(STANDARD_AAS):
    _ASCII_TO_STANDARD_CODE[ord(_amino_acid)] = _code
    _ASCII_TO_STANDARD_CODE[ord(_amino_acid.lower())] = _code


def encode_standard_amino_acids(sequences):
    """Encode aligned sequences as 0-19 and all other symbols as -1."""
    max_length = max((len(sequence) for sequence in sequences), default=0)
    encoded = np.full((len(sequences), max_length), -1, dtype=np.int8)

    if all(isinstance(sequence, str) and sequence.isascii() for sequence in sequences):
        # ASCII upper() never changes length, so a byte table is exact.
        if all(len(sequence) == max_length for sequence in sequences):
            if max_length:
                text = "".join(sequences).encode("ascii")
                encoded[:] = _ASCII_TO_STANDARD_CODE[
                    np.frombuffer(text, dtype=np.uint8)
                ].reshape(len(sequences), max_length)
            return encoded
        for row, sequence in enumerate(sequences):
            if sequence:
                encoded[row, :len(sequence)] = _ASCII_TO_STANDARD_CODE[
                    np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
                ]
        return encoded

    aa_codes = {aa: index for index, aa in enumerate(STANDARD_AAS)}
    for row, sequence in enumerate(sequences):
        values = [aa_codes.get(char, -1) for char in sequence.upper()]
        if values:
            encoded[row, :len(values)] = values
    return encoded


def _unique_encoded_rows(encoded):
    """Return first-seen unique rows, their multiplicities and the row inverse.

    Rows with equal encodings have equal identity to every other row, so
    merging them (even when their raw text differed only in gaps versus
    nonstandard symbols) leaves every neighbour count unchanged.
    """
    encoded = np.ascontiguousarray(encoded, dtype=np.int8)
    row_to_unique = {}
    first_rows = []
    multiplicities = []
    inverse = np.empty(encoded.shape[0], dtype=np.int64)
    for index in range(encoded.shape[0]):
        key = encoded[index].tobytes()
        unique_index = row_to_unique.get(key)
        if unique_index is None:
            unique_index = len(first_rows)
            row_to_unique[key] = unique_index
            first_rows.append(index)
            multiplicities.append(0)
        multiplicities[unique_index] += 1
        inverse[index] = unique_index
    return (
        encoded[first_rows],
        np.asarray(multiplicities, dtype=np.int64),
        inverse,
    )


def _calculate_identity_neighbour_counts_numpy(
    encoded,
    multiplicities,
    threshold,
    block_size=128,
):
    """Exact NumPy fallback on the same bit planes as the Numba kernel."""
    planes, valid_counts = _pack_identity_planes(
        np.ascontiguousarray(encoded, dtype=np.int8)
    )
    neighbour_counts = np.zeros(len(encoded), dtype=np.int64)
    block_size = max(1, int(block_size))

    for left_start in range(0, len(encoded), block_size):
        raise_if_cancelled()
        left_end = min(left_start + block_size, len(encoded))
        left_planes = planes[left_start:left_end, :, None, :]

        for right_start in range(left_start, len(encoded), block_size):
            right_end = min(right_start + block_size, len(encoded))
            right_planes = planes[None, right_start:right_end]
            right_planes = np.moveaxis(right_planes, 2, 1)

            both_valid = left_planes[:, 0] & right_planes[:, 0]
            different = left_planes[:, 1] ^ right_planes[:, 1]
            for plane in range(2, 6):
                different |= left_planes[:, plane] ^ right_planes[:, plane]
            overlap = np.bitwise_count(both_valid).sum(axis=2, dtype=np.int64)
            union = (
                valid_counts[left_start:left_end, None]
                + valid_counts[None, right_start:right_end]
                - overlap
            )
            matches = np.bitwise_count(both_valid & ~different).sum(
                axis=2, dtype=np.int64
            )
            similar = np.logical_and(
                union > 0,
                matches >= (threshold * union - 1e-12),
            )

            neighbour_counts[left_start:left_end] += (
                similar @ multiplicities[right_start:right_end]
            )
            if right_start != left_start:
                neighbour_counts[right_start:right_end] += (
                    similar.T @ multiplicities[left_start:left_end]
                )

    neighbour_counts[neighbour_counts == 0] = 1
    return neighbour_counts


def calculate_identity_weights(
    sequences,
    threshold,
    block_size=128,
    return_metadata=False,
    report_backend=False,
):
    """Return inverse-neighbour weights for aligned protein sequences.

    Identity is the fraction of matching standard amino acids over positions
    where either sequence contains a standard amino acid. Thus gaps and
    nonstandard symbols never count as matches, while missing coverage lowers
    the identity rather than creating a spuriously perfect fragment match.

    SEQUENCES may also be a 2-D integer array already encoded as 0-19 for
    STANDARD_AAS and -1 for every other symbol (label passes its sparse
    alignment this way instead of rebuilding a string per row).
    """
    if (
        isinstance(sequences, np.ndarray)
        and sequences.ndim == 2
        and np.issubdtype(sequences.dtype, np.integer)
    ):
        encoded_rows = sequences
    else:
        encoded_rows = encode_standard_amino_acids(
            [str(sequence).upper() for sequence in sequences]
        )
    if encoded_rows.shape[0] == 0:
        empty_weights = np.zeros(0, dtype=float)
        metadata = {"backend": "disabled", "threads": 0, "fallback_reason": None}
        return (empty_weights, metadata) if return_metadata else empty_weights

    encoded, multiplicities, inverse = _unique_encoded_rows(encoded_rows)

    metadata = {"backend": "numpy", "threads": 1, "fallback_reason": None}
    if NUMBA_AVAILABLE:
        planned_threads = get_balanced_thread_count()
        if report_backend:
            print(
                "Redundancy backend: Numba "
                f"({planned_threads} balanced worker threads; "
                "first use may compile)"
            )
        try:
            neighbour_counts, selected_threads = run_identity_neighbour_counts(
                encoded,
                multiplicities,
                threshold,
            )
            metadata.update(backend="numba", threads=selected_threads)
        except Exception as exc:
            metadata["fallback_reason"] = str(exc)
            if report_backend:
                print(f"Numba redundancy kernel failed; using NumPy fallback ({exc})")
            neighbour_counts = _calculate_identity_neighbour_counts_numpy(
                encoded,
                multiplicities,
                threshold,
                block_size=block_size,
            )
    else:
        metadata["fallback_reason"] = "Numba is not available"
        if report_backend:
            print("Redundancy backend: NumPy fallback (Numba is not available)")
        neighbour_counts = _calculate_identity_neighbour_counts_numpy(
            encoded,
            multiplicities,
            threshold,
            block_size=block_size,
        )

    unique_weights = 1.0 / neighbour_counts.astype(float)
    weights = unique_weights[inverse]
    return (weights, metadata) if return_metadata else weights


def calculate_logo_matrix(
    selected_seqs,
    valid_cols,
    mode="bits",
    gap_mode="with_gap",
    identity_threshold=None,
    return_weighting_metadata=False,
    report_weighting_backend=False,
):
    """Calculate logo letter heights and per-sequence redundancy weights."""
    amino_acids = list(STANDARD_AAS)
    aa_to_index = {aa: index for index, aa in enumerate(amino_acids)}
    matrix = np.zeros((len(valid_cols), len(amino_acids)), dtype=float)
    raw_sequence_count = len(selected_seqs)

    if identity_threshold is None:
        weights = np.ones(raw_sequence_count, dtype=float)
        weighting_metadata = {
            "backend": "disabled",
            "threads": 0,
            "fallback_reason": None,
        }
    else:
        weights, weighting_metadata = calculate_identity_weights(
            selected_seqs,
            identity_threshold,
            return_metadata=True,
            report_backend=report_weighting_backend,
        )

    total_weight = float(weights.sum())
    if raw_sequence_count == 0 or total_weight <= 0.0:
        result = (matrix, weights, weighting_metadata)
        return result if return_weighting_metadata else result[:2]

    weights = np.asarray(weights, dtype=float)
    ascii_sequences = all(
        isinstance(sequence, str) and sequence.isascii() for sequence in selected_seqs
    )
    shortest_length = min(len(sequence) for sequence in selected_seqs)

    for row, col in enumerate(valid_cols):
        raise_if_cancelled()
        if ascii_sequences and 0 <= col < shortest_length:
            # bincount and cumsum add in sequence order, exactly like the loop.
            codes = _ASCII_TO_STANDARD_CODE[np.frombuffer(
                "".join([sequence[col] for sequence in selected_seqs]).encode("ascii"),
                dtype=np.uint8,
            )]
            valid = codes >= 0
            if not valid.any():
                continue
            valid_weights = weights[valid]
            weighted_counts = np.bincount(
                codes[valid], weights=valid_weights, minlength=len(amino_acids)
            )
            valid_weight = np.cumsum(valid_weights)[-1]
        else:
            weighted_counts = np.zeros(len(amino_acids), dtype=float)
            valid_weight = 0.0

            for sequence, weight in zip(selected_seqs, weights):
                if col >= len(sequence):
                    continue
                aa_index = aa_to_index.get(sequence[col].upper())
                if aa_index is None:
                    continue
                weighted_counts[aa_index] += weight
                valid_weight += weight

        if valid_weight <= 0.0:
            continue

        occupancy = valid_weight / total_weight
        probabilities = weighted_counts / valid_weight

        if mode == "pcts":
            heights = probabilities
        else:
            positive = probabilities > 0.0
            entropy = -np.sum(
                probabilities[positive] * np.log2(probabilities[positive])
            )
            # The small-sample correction counts the observations at this
            # column, as WebLogo's n does: the sequences with a residue here,
            # or with reweighting their summed weight. Gaps are not counted,
            # with or without reweighting. (with_gap scaling by occupancy is a
            # separate step, below.)
            correction = 19.0 / (2.0 * np.log(2) * valid_weight)
            information = max(0.0, np.log2(20) - (entropy + correction))
            heights = probabilities * information

        if gap_mode == "with_gap":
            heights = heights * occupancy
        matrix[row, :] = heights

    result = (matrix, weights, weighting_metadata)
    return result if return_weighting_metadata else result[:2]


def logo_column_heights(counts, total_rows):
    """The height in bits of each column's stack of letters, as `logo` draws it by default.

    COUNTS is (columns, 20): how many of the TOTAL_ROWS aligned sequences have
    each standard amino acid in the column, gaps and every other symbol not
    counted. Each height is that of calculate_logo_matrix with mode "bits",
    gap_mode "with_gap" and no identity threshold, summed over the letters:
    the information log2(20) - (entropy + 19 / (2 ln 2 n)) of the n sequences
    that have a residue in the column, floored at 0, times n / TOTAL_ROWS. A
    column with no residue is 0 tall.

    The same formula, worked out from the counts for every column at once; the
    matrix function cuts each column out of a list of strings, which costs
    seconds on a 44,000-row alignment. The two agree to rounding (the matrix
    sums its letters one by one), which tests hold them to.
    """
    counts = np.asarray(counts, dtype=np.float64)
    observed = counts.sum(axis=1)
    has_residues = observed > 0
    probabilities = np.divide(
        counts, observed[:, None], out=np.zeros_like(counts), where=has_residues[:, None]
    )
    entropy = -np.sum(
        probabilities
        * np.log2(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0),
        axis=1,
    )
    correction = np.divide(
        19.0, 2.0 * np.log(2) * observed, out=np.zeros_like(observed), where=has_residues
    )
    information = np.maximum(0.0, np.log2(20) - (entropy + correction))
    return np.where(has_residues, information * (observed / total_rows), 0.0)


_DECODE_BLOCK_ROWS = 2048


def aligned_row_strings(alignment, rows):
    """Return str(alignment[row].seq) for each row without per-residue Python work.

    Decodes blocks of sparse rows through a byte table (0 and unmapped codes
    become '-'), the same spelling the sparse loaders' __getitem__ produces.
    Falls back to __getitem__ for anything the table cannot express.
    """
    matrix = getattr(alignment, "matrix", None)
    int_to_aa = getattr(alignment, "int_to_aa", None)
    table = None
    if (
        matrix is not None
        and int_to_aa is not None
        and getattr(matrix, "dtype", None) == np.uint8
        and getattr(matrix, "ndim", 0) == 2
        and matrix.shape[1] > 0
    ):
        table = np.full(256, ord("-"), dtype=np.uint8)
        for code, residue in int_to_aa.items():
            code = int(code)
            if code == 0 or not 0 < code < 256:
                continue
            if not (isinstance(residue, str) and len(residue) == 1 and residue.isascii()):
                table = None
                break
            table[code] = ord(residue)
    if table is None or not rows or min(rows) < 0 or max(rows) >= matrix.shape[0]:
        return [str(alignment[row].seq) for row in rows]

    column_count = matrix.shape[1]
    row_array = np.asarray(rows, dtype=np.intp)
    strings = []
    for start in range(0, len(row_array), _DECODE_BLOCK_ROWS):
        block = matrix[row_array[start:start + _DECODE_BLOCK_ROWS]].toarray()
        text = table[block].tobytes().decode("ascii")
        strings.extend(
            text[offset:offset + column_count]
            for offset in range(0, len(text), column_count)
        )
    return strings
