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

import os
import re
import tempfile
import numpy as np
from datetime import datetime  # <--- NEW IMPORT
import EMAPSSN_Config as cfg
import Command_Engine
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import validate_output_basename
from utilities.Sequence_Utils import (
    DISPLAYED_POSITION_ATOM_PATTERN,
    normalize_displayed_position_atom,
    reject_bare_negative_positions,
)

LOGO_DIRECTORY = os.path.join("$analysis_result$", "Sequence_Logos")

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    get_num_threads = None
    set_num_threads = None


STANDARD_AAS = tuple("ACDEFGHIKLMNPQRSTVWY")
_BARE_IDENTITY_THRESHOLD = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$"
)
_POSITION_LABEL_PATTERN = DISPLAYED_POSITION_ATOM_PATTERN
_POSITION_RANGE_RE = re.compile(
    rf"^({_POSITION_LABEL_PATTERN})\s*-\s*({_POSITION_LABEL_PATTERN})$"
)
# Ranges expand to one entry per position before the alignment is consulted,
# so a range, and the whole list, is capped far above any alignment's length.
_MAX_LOGO_RANGE_POSITIONS = 1_000_000


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


def parse_identity_threshold(value):
    """Normalize an identity threshold written as a fraction or percentage."""
    text = str(value).strip()
    if not text:
        raise ValueError(Message("Identity threshold cannot be empty."))

    is_percent = text.endswith('%')
    numeric_text = text[:-1].strip() if is_percent else text
    try:
        threshold = float(numeric_text)
    except ValueError as exc:
        raise ValueError(Message(
            "Invalid identity threshold '{value}'. Use 0.9, 90, or 90%.", value=value
        )) from exc

    if is_percent or threshold > 1.0:
        threshold /= 100.0

    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError(Message(
            "Identity threshold '{value}' is outside the supported range (0, 100%].", value=value
        ))
    return threshold


def extract_identity_threshold(args):
    """Remove and parse an optional identity-reweighting argument."""
    threshold = None
    remaining_args = []

    for arg in args:
        text = str(arg).strip()
        threshold_value = text if _BARE_IDENTITY_THRESHOLD.fullmatch(text) else None

        if threshold_value is None:
            remaining_args.append(arg)
            continue

        if threshold is not None:
            raise ValueError(Message("Provide only one identity threshold for logo reweighting."))
        threshold = parse_identity_threshold(threshold_value)

    return threshold, remaining_args


_ASCII_TO_STANDARD_CODE = np.full(256, -1, dtype=np.int8)
for _code, _amino_acid in enumerate(STANDARD_AAS):
    _ASCII_TO_STANDARD_CODE[ord(_amino_acid)] = _code
    _ASCII_TO_STANDARD_CODE[ord(_amino_acid.lower())] = _code


def _encode_standard_amino_acids(sequences):
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
        encoded_rows = _encode_standard_amino_acids(
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


def _configure_logo_y_axis(ax, mode, gap_mode):
    """Use a fixed theoretical scale so separate logos are comparable."""
    if mode == "bits":
        maximum_bits = float(np.log2(len(STANDARD_AAS)))
        ax.set_ylim(0.0, maximum_bits)
        ylabel = "Information Content (Bits)" if gap_mode == "with_gap" else "Bits"
    else:
        from matplotlib.ticker import PercentFormatter

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ylabel = "Percentage"
    ax.set_ylabel(ylabel)


def _generate_logo_artifact(payload):
    """Calculate and render one logo without accessing live viewer state."""
    import logomaker
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.transforms import Affine2D

    selected_seqs = payload["selected_seqs"]
    valid_cols = payload["valid_cols"]
    plot_positions = payload["plot_positions"]
    mode = payload["mode"]
    gap_mode = payload["gap_mode"]
    identity_threshold = payload["identity_threshold"]
    filename = payload["filename"]

    amino_acids = list(STANDARD_AAS)
    plot_coordinates = get_compact_logo_coordinates(plot_positions)
    matrix, sequence_weights, weighting_metadata = calculate_logo_matrix(
        selected_seqs,
        valid_cols,
        mode=mode,
        gap_mode=gap_mode,
        identity_threshold=identity_threshold,
        return_weighting_metadata=True,
        report_weighting_backend=True,
    )
    dataframe = pd.DataFrame(matrix, index=plot_coordinates, columns=amino_acids)
    effective_sequence_count = float(sequence_weights.sum())

    if identity_threshold is not None:
        print(
            "Identity reweighting enabled: "
            f"threshold={identity_threshold * 100:g}%, "
            f"raw N={len(selected_seqs)}, "
            f"effective N={effective_sequence_count:.2f}"
        )

    logo_dir = payload["logo_dir"]
    os.makedirs(logo_dir, exist_ok=True)
    save_path = os.path.abspath(
        payload.get("output_path") or os.path.join(logo_dir, filename)
    )
    allow_overwrite = bool(payload.get("allow_overwrite", False))
    if not allow_overwrite and os.path.exists(save_path):
        raise FileExistsError(Message("Output file already exists: {path}", path=save_path))

    fig_width = max(6, len(plot_positions) * 0.5 + 1)
    fig = Figure(figsize=(fig_width, 4))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    partial_path = None
    try:
        logo = logomaker.Logo(dataframe, ax=ax, color_scheme=payload["color_scheme"])

        base_gap = 0.01
        max_gap = base_gap if mode == "pcts" else base_gap * np.log2(20)

        for patch in ax.patches:
            local_bbox = patch.get_path().get_extents()
            local_height = local_bbox.height
            if local_height <= 0.001:
                continue

            local_ymax = local_bbox.ymax
            gap = min(local_height * 0.05, max_gap)
            scale_factor = (local_height - gap) / local_height
            local_shrink = (
                Affine2D()
                .translate(0, -local_ymax)
                .scale(1.0, scale_factor)
                .translate(0, local_ymax)
            )
            patch.set_transform(local_shrink + patch.get_transform())

        logo.style_spines(visible=False)
        logo.style_spines(spines=['left', 'bottom'], visible=True)
        ax.set_xticks(plot_coordinates)
        ax.set_xticklabels(plot_positions)
        ax.set_xlim(-0.5, len(plot_coordinates) - 0.5)
        ax.set_xlabel(_position_axis_label(payload))

        _configure_logo_y_axis(ax, mode, gap_mode)

        fig.tight_layout()
        suffix = os.path.splitext(filename)[1].lower()
        file_descriptor, partial_path = tempfile.mkstemp(
            prefix=f".{os.path.splitext(filename)[0]}.",
            suffix=f".partial{suffix}",
            dir=logo_dir,
        )
        os.close(file_descriptor)
        fig.savefig(
            partial_path,
            format=suffix.lstrip("."),
            transparent=filename.lower().endswith('.png'),
            dpi=600,
            bbox_inches='tight',
        )
        if not allow_overwrite and os.path.exists(save_path):
            raise FileExistsError(Message("Output file already exists: {path}", path=save_path))
        os.replace(partial_path, save_path)
        partial_path = None
    finally:
        fig.clear()
        if partial_path and os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass

    # The background job's report: the console line shows it translated.
    if identity_threshold is None:
        message = Message(
            "Saved {gap_mode} {mode} logo for %n aligned node(s) to {file}",
            n=len(selected_seqs), gap_mode=gap_mode, mode=mode, file=filename,
        )
    else:
        message = Message(
            "Saved {gap_mode} {mode} logo for %n aligned node(s) "
            "(identity {identity}%, effective N {effective}) to {file}",
            n=len(selected_seqs), gap_mode=gap_mode, mode=mode, file=filename,
            identity=f"{identity_threshold * 100:g}", effective=f"{effective_sequence_count:.2f}",
        )
    return {
        "message": message,
        "save_path": save_path,
        "effective_sequence_count": effective_sequence_count,
    }


_DECODE_BLOCK_ROWS = 2048


def _aligned_row_strings(alignment, rows):
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


def _normalize_logo_filename(filename):
    """Return a safe SVG/PNG basename for the configured logo directory."""
    filename = validate_output_basename(filename)
    if not filename.lower().endswith((".png", ".svg")):
        filename += ".svg"
    # A name that is only an extension (".svg") has no extension to os.path:
    # the job would fail later on a format of "".
    if not os.path.splitext(filename)[1]:
        raise ValueError(Message(
            "Filename '{file}' needs a name before its extension.", file=filename
        ))
    return filename


def _available_automatic_filename(scheduler, directory, filename):
    """Add a stable numeric suffix when a generated timestamp is occupied."""
    stem, suffix = os.path.splitext(filename)
    candidate = filename
    index = 2
    while True:
        path = os.path.abspath(os.path.join(directory, candidate))
        if not os.path.exists(path) and not scheduler.is_output_path_reserved(path):
            return candidate, path
        candidate = f"{stem}_{index}{suffix}"
        index += 1


def _position_axis_label(payload):
    """Describe the numbering the plotted position labels actually use."""
    if payload.get("numbering") == "occupancy":
        return "Position (occupancy numbering)"
    return f"Position (relative to {payload['ref_id']})"


def resolve_reference_columns(alignment, requested_positions):
    """Resolve displayed positions to alignment columns.

    Uses the alignment's displayed-label mapping in both reference mode
    (reference numbering plus offset) and occupancy mode (retained columns
    numbered from 1), so logo positions match `query` and residue predicates.
    A position without a label is missing, as it is for `query`; with no
    retained columns, every position is.
    """
    valid_cols = []
    plot_positions = []
    missing_positions = []

    label_to_col = getattr(alignment, 'label_to_col', None) or {}
    for position in requested_positions:
        col_idx = label_to_col.get(str(position))
        if col_idx is None:
            missing_positions.append(position)
        else:
            valid_cols.append(col_idx)
            plot_positions.append(position)
    return valid_cols, plot_positions, missing_positions


def get_compact_logo_coordinates(plot_positions):
    """Return evenly spaced plot coordinates for arbitrary residue labels."""
    return list(range(len(plot_positions)))


def _normalize_logo_position_label(value):
    """Return an integer or canonical hierarchical insertion label."""
    text = normalize_displayed_position_atom(value)

    major_text, separator, insertion_text = text.partition('.')
    major = int(major_text)
    if not separator:
        return major

    insertion = int(insertion_text)
    if insertion <= 0:
        raise ValueError(Message(
            "Invalid insertion position '{value}'; the fractional suffix must be positive.", value=value
        ))
    return f"{major}.{insertion}"


def _logo_position_sort_key(position):
    """Sort hierarchical labels in reference-alignment order, not as floats."""
    major_text, separator, insertion_text = str(position).partition('.')
    return int(major_text), int(insertion_text) if separator else 0


def parse_logo_positions(position_spec):
    """Parse integer reference positions and explicit insertion labels.

    Integer ranges retain their historical meaning and expand to integer
    reference positions only. Insertion positions must be listed explicitly.
    """
    text = str(position_spec).strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    reject_bare_negative_positions(text)

    positions = {}
    for raw_part in text.split(','):
        part = raw_part.strip()
        if not part:
            continue

        range_match = _POSITION_RANGE_RE.fullmatch(part)
        if range_match:
            start = _normalize_logo_position_label(range_match.group(1))
            end = _normalize_logo_position_label(range_match.group(2))
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(Message(
                    "Fractional range '{range}' is not supported; list insertion "
                    "positions explicitly.",
                    range=part,
                ))
            if start > end:
                raise ValueError(Message(
                    "Position range '{range}' must be written from lower to higher.", range=part
                ))
            if end - start + 1 > _MAX_LOGO_RANGE_POSITIONS:
                raise ValueError(Message(
                    "Position range '{range}' is too large; a range may span at most "
                    "%n position(s).",
                    n=_MAX_LOGO_RANGE_POSITIONS, range=part,
                ))
            for position in range(start, end + 1):
                positions[str(position)] = position
            if len(positions) > _MAX_LOGO_RANGE_POSITIONS:
                raise ValueError(Message(
                    "The position list is too large; it may name at most %n position(s).",
                    n=_MAX_LOGO_RANGE_POSITIONS,
                ))
            continue

        position = _normalize_logo_position_label(part)
        positions[str(position)] = position

    return sorted(positions.values(), key=_logo_position_sort_key)


def print_help():
    print("""
    Sequence Logo Generator
    =======================
    Usage: logo [EXPRESSION] [POSITIONS] [FILENAME] [MODE] [GAP_MODE] [COLOR_SCHEME] [IDENTITY]
           logo help

    Description:
      Generates a high-resolution SVG or PNG sequence logo for a targeted subset of nodes.
      Output is saved beneath the configured Analysis Results directory
      (default: 'Analysis_Results/Sequence_Logos/').
      Label and logo jobs share one sequential background queue. Selection,
      aligned sequences, mapped positions, and rendering options are captured
      when the command is submitted.

      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer. If no nodes are selected, it 
        defaults to analyzing ALL nodes in the entire network.

      * Hidden nodes are included: they are used whenever the selection or
        expression covers them.

    Arguments (Can be provided in almost any order):
      1. [POSITIONS] : (Required) Comma-separated displayed positions or integer
                       ranges enclosed in brackets: reference numbering when a
                       reference is active, otherwise occupancy numbering, the same
                       labels 'query' uses. Fractional insertion positions
                       (alignment columns where the reference has a gap) are accepted
                       when listed explicitly. Negative positions must be enclosed
                       individually in parentheses. Spaces inside the brackets
                       are allowed, and only one bracketed argument is accepted.
                       Examples: [1,2,9-12], [10,10.1,10.2,11],
                       or [(-3)-(-1),0]
                       Non-contiguous positions are plotted adjacently while retaining
                       their original position labels.
      2. EXPRESSION  : Boolean logic target (e.g., #cluster_1#, "ATA", or $sele$).
      3. FILENAME    : Output name. Defaults to logo_YYYYMMDD_HHMMSS.svg.
                       (Note: A filename must end in .svg or .png. If the LAST
                       remaining string does, it is the filename and any others form
                       the expression; otherwise all remaining strings form the
                       expression, so a mistyped keyword such as 'nogap' is an error
                       and is never used as a filename. Strings of an expression are
                       joined with spaces, as in '#c1# & #c2#'.)
      4. MODE        : 'bits' (Default, Information Content) or 'pcts' (Percentages).
                       Bits mode subtracts a small-sample correction computed from the
                       number of sequences with a residue at the position (gaps are
                       not counted), or with IDENTITY from their summed weight.
      5. GAP_MODE    : 'with_gap' (Default, scales total height by occupancy) or 'no_gap'.
      6. COLOR_SCHEME: Preset color scheme name. (Default: chemistry)
                       Can be provided standalone or as key-value (e.g. color=classic).
                       Presets: chemistry, classic, grays, base_pairing, colorblind_safe,
                       weblogo_protein, skylign_protein, dmslogo_charge, dmslogo_funcgroup,
                       hydrophobicity, charge, NajafabadiEtAl2017.
                       Preset names are not case-sensitive. Any matplotlib color,
                       such as color=red or color=#ff0000, colors every letter.
      7. IDENTITY    : Optional sequence-redundancy threshold. Reweighting is OFF
                       unless supplied. Equivalent forms: 0.9, 90, or 90%.
                       A bare number up to 1 is a fraction (1 means 100%, 0.5 means
                       50%); a larger bare number is a percentage (5 means 5%).
                       Write 1% for 1 percent.
                       Applies weighted frequencies to both modes
                       and effective-sample correction to bits mode.

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort before a logo job is submitted.
      A valid expression may match zero nodes.

    Examples:
      logo [10-20]                        (Logos pos 10-20 for selected or all nodes)
      logo [10,10.1,10.2,11]             (Includes explicit insertion positions)
      logo [(-1),0,1]                     (Includes a parenthesized negative position)
      logo #cluster_1# [(-3)-(-1)] pcts   (Percentage logo across a negative range)
      logo #cluster_1# [1,5] pcts no_gap  (Percentage logo ignoring gaps for pos 1 and 5)
      logo [10-20] color=charge           (Generates bits logo using the charge color scheme)
      logo #cluster_1# [1,5] 90%           (Reweights sequences at 90% identity)
      logo #cluster_1# [1,5] classic      (Generates bits logo using classic scheme)
      logo K10 [1] target_logo.png        (Logos pos 1 for K10 expr, saves as target_logo.png)
    """)

def _rejoin_split_brackets(args):
    """Join the tokens of a bracketed argument that spaces split, e.g. "[1," "5]".

    Tokens whose brackets never balance are returned as they were.
    """
    rejoined = []
    pending = []
    in_bracket = False

    for arg in args:
        if '[' in arg and not in_bracket:
            if arg.count('[') > arg.count(']'):
                in_bracket = True
                pending.append(arg)
            else:
                rejoined.append(arg)
        elif in_bracket:
            pending.append(arg)
            if ']' in arg:
                joined = " ".join(pending)
                if joined.count('[') <= joined.count(']'):
                    rejoined.append(joined)
                    pending = []
                    in_bracket = False
        else:
            rejoined.append(arg)

    rejoined.extend(pending)
    return rejoined


def _resolve_color_scheme(value, known_schemes):
    """Return the color scheme logomaker is given for VALUE, or raise ValueError.

    logomaker takes a preset's exact name, or else any matplotlib color, so a
    preset is matched case-insensitively here and anything else must be a
    color matplotlib reads (red, #ff0000, 0.5).
    """
    for scheme in known_schemes:
        if scheme.lower() == value.lower():
            return scheme

    from matplotlib.colors import to_rgb

    try:
        to_rgb(value)
    except ValueError:
        raise ValueError(Message(
            "Unknown color scheme '{value}'. Use a preset from the help, "
            "or a color such as red or #ff0000.",
            value=value,
        )) from None
    return value


def run(viewer, args):
    if not args:
        # The console line shows the first line; the usage is for the terminal.
        msg = JoinedMessage([
            Message("Error: Logo command requires a POSITIONS parameter."),
            "Usage: logo [POSITIONS]",
        ], separator="\n")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # 0.5. Join positions that spaces split, e.g. [1, 5], before any token is
    # read on its own (a lone "5" would otherwise be an identity threshold).
    args = _rejoin_split_brackets(args)

    # 1. Extract Mode Keywords (Aggressively filter to prevent filename confusion)
    mode = "bits"
    gap_mode = "with_gap"
    filtered_args = []
    
    for arg in args:
        a_lower = arg.lower()
        if a_lower in ["pcts", "pct", "percentage", "percentages"]:
            mode = "pcts"
        elif a_lower in ["bits", "bit"]:
            mode = "bits"
        elif a_lower in ["with_gap", "with_gaps", "gaps", "gap"]:
            gap_mode = "with_gap"
        elif a_lower in ["no_gap", "no_gaps"]:
            gap_mode = "no_gap"
        else:
            filtered_args.append(arg)
            
    args = filtered_args

    # 1.5. Extract Color Scheme preset
    KNOWN_SCHEMES = [
        'classic', 'grays', 'base_pairing', 'colorblind_safe',
        'weblogo_protein', 'skylign_protein', 'dmslogo_charge',
        'dmslogo_funcgroup', 'hydrophobicity', 'chemistry', 'charge',
        'NajafabadiEtAl2017'
    ]
    color_scheme = "chemistry"  # Default
    
    remaining_args = []
    for arg in args:
        match = re.match(r'^(color_scheme|colors|color|scheme)=(.*)$', arg, re.IGNORECASE)
        if match:
            # A preset (any case) or a color matplotlib reads, as logomaker does;
            # anything else would only fail later in the background job.
            try:
                color_scheme = _resolve_color_scheme(match.group(2), KNOWN_SCHEMES)
            except ValueError as exc:
                msg = Message("Error: {error}", error=exc)
                Command_Engine.command_failed(viewer, msg)
                Command_Engine.print_help(viewer, msg)
                return
        elif arg.lower() in [s.lower() for s in KNOWN_SCHEMES]:
            # Case-insensitive standalone known preset matched
            color_scheme = [s for s in KNOWN_SCHEMES if s.lower() == arg.lower()][0]
        else:
            remaining_args.append(arg)
    args = remaining_args

    # 1.6. Extract optional sequence-redundancy threshold
    try:
        identity_threshold, args = extract_identity_threshold(args)
    except ValueError as exc:
        msg = Message("Error: {error}", error=exc)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    # 2. Extract Positions Argument (First argument containing brackets)
    bracket_indices = [i for i, a in enumerate(args) if a.startswith('[') and a.endswith(']')]
    
    if not bracket_indices:
        msg = Message("Error: No positions provided. Use [...] syntax.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return
        
    pos_idx = bracket_indices[0]
    pos_str = args.pop(pos_idx)
    
    # A second bracket argument is neither a filename nor part of an expression.
    for arg in args:
        if arg.startswith('[') and not arg.lower().endswith(('.png', '.svg')):
            msg = Message(
                "Error: Give the positions in one [...] argument; found a second one: '{argument}'.",
                argument=arg,
            )
            Command_Engine.command_failed(viewer, msg)
            Command_Engine.print_help(viewer, msg)
            return

    # 3. Handle Ambiguity & Assign Filename/Expression
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logo_{timestamp}.svg"
    automatic_filename = True
    expr = "$sele$" 
    
    # A filename ends in .svg or .png. If the last remaining string does, it is
    # the filename; everything else, and every string when it does not, is the
    # expression, so a mistyped keyword fails validation instead of naming a file.
    if args and args[-1].lower().endswith(('.png', '.svg')):
        filename = args.pop(-1)
        automatic_filename = False
    if args:
        expr = " ".join(args)

    if expr != "$sele$":
        classification = Command_Engine.classify_selection_expression(expr)
        if classification.kind != Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
            # With no .svg/.png name given, a mistyped keyword or an extension-less
            # filename is the last word of the expression: name it. A last string
            # that is itself part of an expression keeps the selection error.
            if (
                automatic_filename
                and len(args) >= 2
                and Command_Engine.classify_selection_expression(args[-1]).kind
                == Command_Engine.SelectionClassificationKind.NOT_EXPRESSION
            ):
                msg = Message(
                    "Error: Unrecognized logo argument '{argument}'. "
                    "A filename must end in .svg or .png.",
                    argument=args[-1],
                )
                Command_Engine.command_failed(viewer, msg)
                Command_Engine.print_help(viewer, msg)
                return
            error = classification.error or Command_Engine.SelectionExpressionError(
                Message("'{expression}' is not a Boolean selection expression.", expression=expr)
            )
            Command_Engine.report_selection_error(viewer, expr, error, Message("Logo"))
            return

    try:
        filename = _normalize_logo_filename(filename)
    except ValueError as exc:
        msg = Message("Error: {error}", error=exc)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    # ---> NEW LOGIC: Smart Fallback to ALL Nodes <---
    if expr == "$sele$" and not getattr(viewer, 'selected_indices', []):
        expr = '"*"'  # The wildcard string matches all headers
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("No selection found. Defaulting to ALL nodes."))
        print("No nodes selected. Defaulting to ALL nodes in the network.")

    # 5. Parse Position Array
    try:
        requested_positions = parse_logo_positions(pos_str)
    except ValueError as exc:
        msg = Message("Error: {error}", error=exc)
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
        return

    if not requested_positions:
        msg = Message("Error: Could not parse positions from brackets.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        msg = Message("Error: MSA not loaded in viewer. Please check inputs.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        return
    if len(alignment.aln) == 0:
        msg = Message(
            "Error: The selected MSA contains no aligned rows for the current network."
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        return

    # 6. Apply Boolean Logic to get matching sequences
    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    
    try:
        mask = Command_Engine.parse_advanced_expression(
            expr,
            viewer_to_aln,
            valid_indices,
            viewer.full_headers,
            getattr(viewer, 'cluster_labels', None),
            getattr(viewer, 'group_labels', None),
            getattr(viewer, 'alignment', None),
            metadata=getattr(viewer, 'metadata', None),
            selection_mask=Command_Engine.get_selected_mask(viewer),
        )
        selected_nodes = np.where(mask)[0]
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, Message("Logo"))
        return
        
    if len(selected_nodes) == 0:
        msg = Message("No nodes matched the criteria for logo generation.")
        Command_Engine.show_status(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    # 7. Map positions through the alignment's displayed labels, the mapping
    # `query` and residue predicates use. The axis names the header the
    # alignment anchored on, not the text the reference was requested by.
    has_reference = getattr(viewer.alignment, 'has_reference', False)
    numbering = "reference" if has_reference else "occupancy"
    ref_id = getattr(viewer.alignment, 'resolved_ref_full', None) if has_reference else ""

    valid_cols, plot_positions, missing_positions = resolve_reference_columns(
        viewer.alignment,
        requested_positions,
    )
    for position in missing_positions:
        print(f"Warning: Position {position} was not found in the active alignment mapping.")

    if not valid_cols:
        msg = Message("Error: Requested positions are outside the sequence bounds.")
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        return

    # 8. Extract Sequences for Selected Nodes
    selected_rows = [int(viewer_to_aln[idx]) for idx in selected_nodes]
    selected_seqs = _aligned_row_strings(
        viewer.alignment.aln,
        [row_idx for row_idx in selected_rows if row_idx != -1],
    )

    if not selected_seqs:
        msg = Message(
            "Error: No aligned nodes matched the logo selection criteria."
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.show_status(viewer, msg)
        return

    # 9. Freeze the selected data and submit one background artifact job.
    logo_dir = cfg.resolve_directory_path(LOGO_DIRECTORY)
    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        msg = Message("Logo generation failed: the background job scheduler is unavailable.")
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
        return

    if automatic_filename:
        filename, output_path = _available_automatic_filename(
            scheduler,
            logo_dir,
            filename,
        )
    else:
        output_path = os.path.abspath(os.path.join(logo_dir, filename))
        if scheduler.is_output_path_reserved(output_path):
            msg = Message(
                "Logo generation failed: {error}",
                error=Message("Output file is already reserved by a background job: {path}", path=output_path),
            )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return
    allow_overwrite = not automatic_filename

    payload = {
        "selected_seqs": tuple(selected_seqs),
        "valid_cols": tuple(valid_cols),
        "plot_positions": tuple(plot_positions),
        "mode": mode,
        "gap_mode": gap_mode,
        "identity_threshold": identity_threshold,
        "filename": filename,
        "color_scheme": color_scheme,
        "logo_dir": logo_dir,
        "output_path": output_path,
        "allow_overwrite": allow_overwrite,
        "ref_id": ref_id,
        "numbering": numbering,
    }
    try:
        scheduler.enqueue(
            command_name="logo",
            description=f"logo -> {filename}",
            payload=payload,
            worker=_generate_logo_artifact,
            output_path=output_path,
            allow_overwrite=allow_overwrite,
        )
    except (FileExistsError, RuntimeError) as exc:
        msg = Message("Logo generation failed: {error}", error=exc)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_failed(viewer, msg)
    else:
        Command_Engine.command_succeeded(viewer, f"Queued logo generation for {output_path}; waiting for the background job.")
