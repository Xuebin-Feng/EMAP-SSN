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

import os
import re
import tempfile
import numpy as np
from datetime import datetime  # <--- NEW IMPORT
import SSN_Config as cfg
import SSN_Utils as utils
import Command_Engine

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
_POSITION_LABEL_PATTERN = r"[+-]?\d+(?:\.\d+)?"
_POSITION_LABEL_RE = re.compile(rf"^{_POSITION_LABEL_PATTERN}$")
_POSITION_RANGE_RE = re.compile(
    rf"^({_POSITION_LABEL_PATTERN})\s*-\s*({_POSITION_LABEL_PATTERN})$"
)


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


if NUMBA_AVAILABLE:

    @njit(parallel=True, nogil=True, cache=True)
    def _identity_neighbour_counts_kernel(encoded, multiplicities, threshold):
        """Count threshold neighbours for every unique encoded sequence."""
        sequence_count, alignment_length = encoded.shape
        neighbour_counts = np.empty(sequence_count, dtype=np.int64)

        for left_index in prange(sequence_count):
            neighbour_count = 0
            for right_index in range(sequence_count):
                union_count = 0
                match_count = 0

                for column in range(alignment_length):
                    left_residue = encoded[left_index, column]
                    right_residue = encoded[right_index, column]

                    if left_residue >= 0 or right_residue >= 0:
                        union_count += 1
                    if left_residue >= 0 and left_residue == right_residue:
                        match_count += 1

                if (
                    union_count > 0
                    and match_count >= threshold * union_count - 1e-12
                ):
                    neighbour_count += multiplicities[right_index]

            # An all-invalid row has undefined identity. It remains one
            # independent observation, matching the historical NumPy path.
            neighbour_counts[left_index] = (
                neighbour_count if neighbour_count > 0 else 1
            )

        return neighbour_counts


def run_identity_neighbour_counts(encoded, multiplicities, threshold):
    """Run the exact kernel with balanced threads and restore thread settings."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    previous_threads = get_num_threads()
    selected_threads = choose_balanced_thread_count(previous_threads)
    if selected_threads != previous_threads:
        set_num_threads(selected_threads)

    try:
        counts = _identity_neighbour_counts_kernel(
            np.ascontiguousarray(encoded, dtype=np.int8),
            np.ascontiguousarray(multiplicities, dtype=np.int64),
            float(threshold),
        )
    finally:
        if selected_threads != previous_threads:
            set_num_threads(previous_threads)

    return counts, selected_threads


def parse_identity_threshold(value):
    """Normalize an identity threshold written as a fraction or percentage."""
    text = str(value).strip()
    if not text:
        raise ValueError("Identity threshold cannot be empty.")

    is_percent = text.endswith('%')
    numeric_text = text[:-1].strip() if is_percent else text
    try:
        threshold = float(numeric_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid identity threshold '{value}'. Use 0.9, 90, or 90%."
        ) from exc

    if is_percent or threshold > 1.0:
        threshold /= 100.0

    if not np.isfinite(threshold) or threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            f"Identity threshold '{value}' is outside the supported range (0, 100%]."
        )
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
            raise ValueError("Provide only one identity threshold for logo reweighting.")
        threshold = parse_identity_threshold(threshold_value)

    return threshold, remaining_args


def _encode_standard_amino_acids(sequences):
    """Encode aligned sequences as 0-19 and all other symbols as -1."""
    max_length = max((len(sequence) for sequence in sequences), default=0)
    encoded = np.full((len(sequences), max_length), -1, dtype=np.int8)
    aa_codes = {aa: index for index, aa in enumerate(STANDARD_AAS)}

    for row, sequence in enumerate(sequences):
        values = [aa_codes.get(char, -1) for char in sequence.upper()]
        if values:
            encoded[row, :len(values)] = values
    return encoded


def _calculate_identity_neighbour_counts_numpy(
    encoded,
    multiplicities,
    threshold,
    block_size=128,
):
    """Historical exact NumPy implementation retained as a fallback."""
    valid = encoded >= 0
    valid_counts = valid.sum(axis=1, dtype=np.int64)
    neighbour_counts = np.zeros(len(encoded), dtype=np.int64)
    block_size = max(1, int(block_size))

    for left_start in range(0, len(encoded), block_size):
        left_end = min(left_start + block_size, len(encoded))
        left_encoded = encoded[left_start:left_end]
        left_valid = valid[left_start:left_end]

        for right_start in range(left_start, len(encoded), block_size):
            right_end = min(right_start + block_size, len(encoded))
            right_encoded = encoded[right_start:right_end]
            right_valid = valid[right_start:right_end]

            overlap = np.logical_and(
                left_valid[:, None, :], right_valid[None, :, :]
            ).sum(axis=2, dtype=np.int64)
            union = (
                valid_counts[left_start:left_end, None]
                + valid_counts[None, right_start:right_end]
                - overlap
            )
            matches = np.logical_and(
                left_encoded[:, None, :] == right_encoded[None, :, :],
                np.logical_and(left_valid[:, None, :], right_valid[None, :, :]),
            ).sum(axis=2, dtype=np.int64)
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
    """
    normalized_sequences = [str(sequence).upper() for sequence in sequences]
    if not normalized_sequences:
        empty_weights = np.zeros(0, dtype=float)
        metadata = {"backend": "disabled", "threads": 0, "fallback_reason": None}
        return (empty_weights, metadata) if return_metadata else empty_weights

    unique_sequences = []
    sequence_to_unique = {}
    inverse = np.empty(len(normalized_sequences), dtype=np.int64)
    multiplicities = []
    for index, sequence in enumerate(normalized_sequences):
        unique_index = sequence_to_unique.get(sequence)
        if unique_index is None:
            unique_index = len(unique_sequences)
            sequence_to_unique[sequence] = unique_index
            unique_sequences.append(sequence)
            multiplicities.append(0)
        multiplicities[unique_index] += 1
        inverse[index] = unique_index

    encoded = _encode_standard_amino_acids(unique_sequences)
    multiplicities = np.asarray(multiplicities, dtype=np.int64)

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

    for row, col in enumerate(valid_cols):
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
            # Preserve the historical calculation when reweighting is off.
            # With reweighting enabled, use the effective non-gap observations
            # available at this specific alignment column.
            correction_count = (
                valid_weight if identity_threshold is not None else raw_sequence_count
            )
            correction = 19.0 / (2.0 * np.log(2) * correction_count)
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
        raise FileExistsError(f"Output file already exists: {save_path}")

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
        ax.set_xlabel(
            f"Position (relative to {payload['ref_id'] or 'first sequence'})"
        )

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
            raise FileExistsError(f"Output file already exists: {save_path}")
        os.replace(partial_path, save_path)
        partial_path = None
    finally:
        fig.clear()
        if partial_path and os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass

    message = f"Saved {gap_mode} {mode} logo for {len(selected_seqs)} aligned nodes"
    if identity_threshold is not None:
        message += (
            f" (identity {identity_threshold * 100:g}%, "
            f"effective N {effective_sequence_count:.2f})"
        )
    message += f" to {filename}"
    return {
        "message": message,
        "save_path": save_path,
        "effective_sequence_count": effective_sequence_count,
    }


def _normalize_logo_filename(filename):
    """Return a safe SVG/PNG basename for the configured logo directory."""
    filename = str(filename).strip()
    if not filename:
        raise ValueError("Filename cannot be empty.")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("Filename must not include a directory or path separators.")
    if re.search(r'[<>:"|?*\x00-\x1f]', filename):
        raise ValueError(f"Filename contains unsupported characters: '{filename}'.")
    if not filename.lower().endswith((".png", ".svg")):
        filename += ".svg"
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


def resolve_reference_columns(alignment, requested_positions, ref_seq_str):
    """Resolve displayed integer positions to alignment columns."""
    valid_cols = []
    plot_positions = []
    missing_positions = []

    if getattr(alignment, 'has_reference', False) and getattr(alignment, 'label_to_col', None):
        for position in requested_positions:
            col_idx = alignment.label_to_col.get(str(position))
            if col_idx is None:
                missing_positions.append(position)
            else:
                valid_cols.append(col_idx)
                plot_positions.append(position)
        return valid_cols, plot_positions, missing_positions

    ref_pos_to_col = {}
    curr_pos = 1
    for col_idx, char in enumerate(ref_seq_str):
        if char not in getattr(cfg, 'GAP_CHARS', ['-', '.']):
            ref_pos_to_col[curr_pos] = col_idx
            curr_pos += 1

    for position in requested_positions:
        if position in ref_pos_to_col:
            valid_cols.append(ref_pos_to_col[position])
            plot_positions.append(position)
        else:
            missing_positions.append(position)

    return valid_cols, plot_positions, missing_positions


def get_compact_logo_coordinates(plot_positions):
    """Return evenly spaced plot coordinates for arbitrary residue labels."""
    return list(range(len(plot_positions)))


def _normalize_logo_position_label(value):
    """Return an integer or canonical hierarchical insertion label."""
    text = str(value).strip()
    if not _POSITION_LABEL_RE.fullmatch(text):
        raise ValueError(f"Invalid position label '{value}'.")

    major_text, separator, insertion_text = text.partition('.')
    major = int(major_text)
    if not separator:
        return major

    insertion = int(insertion_text)
    if insertion <= 0:
        raise ValueError(
            f"Invalid insertion position '{value}'; the fractional suffix must be positive."
        )
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
                raise ValueError(
                    f"Fractional range '{part}' is not supported; list insertion "
                    "positions explicitly."
                )
            if start > end:
                raise ValueError(
                    f"Position range '{part}' must be written from lower to higher."
                )
            for position in range(start, end + 1):
                positions[str(position)] = position
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
      Output is automatically saved to your 'Analysis_Results/Sequence_Logos/' directory.
      Label and logo jobs share one sequential background queue. Selection,
      aligned sequences, mapped positions, and rendering options are captured
      when the command is submitted.

      * QUICK USE: If no expression is provided, the command automatically targets 
        the nodes currently selected in the viewer. If no nodes are selected, it 
        defaults to analyzing ALL nodes in the entire network.

    Arguments (Can be provided in almost any order):
      1. [POSITIONS] : (Required) Comma-separated reference positions or integer
                       ranges enclosed in brackets. Fractional insertion positions
                       (alignment columns where the reference has a gap) are accepted
                       when listed explicitly.
                       Examples: [1, 2, 9-12] or [10, 10.1, 10.2, 11]
                       Non-contiguous positions are plotted adjacently while retaining
                       their original reference-position labels.
      2. EXPRESSION  : Boolean logic target (e.g., #cluster_1#, "ATA", or $sele$).
      3. FILENAME    : Output name. Defaults to logo_YYYYMMDD_HHMMSS.svg.
                       (Note: The LAST unrecognized string is treated as the filename).
      4. MODE        : 'bits' (Default, Information Content) or 'pcts' (Percentages).
      5. GAP_MODE    : 'with_gap' (Default, scales total height by occupancy) or 'no_gap'.
      6. COLOR_SCHEME: Preset color scheme name. (Default: chemistry)
                       Can be provided standalone or as key-value (e.g. color=classic).
                       Presets: chemistry, classic, grays, base_pairing, colorblind_safe,
                       weblogo_protein, skylign_protein, dmslogo_charge, dmslogo_funcgroup,
                       hydrophobicity, charge, NajafabadiEtAl2017.
      7. IDENTITY    : Optional sequence-redundancy threshold. Reweighting is OFF
                       unless supplied. Equivalent forms: 0.9, 90, or 90%.
                       Applies weighted frequencies to both modes
                       and effective-sample correction to bits mode.

    Selection Validation:
      Referenced clusters, groups, alignment positions, metadata properties, and
      files must exist. Invalid references abort before a logo job is submitted.
      A valid expression may match zero nodes.

    Examples:
      logo [10-20]                        (Logos pos 10-20 for selected or all nodes)
      logo [10,10.1,10.2,11]             (Includes explicit insertion positions)
      logo #cluster_1# [1,5] pcts no_gap  (Percentage logo ignoring gaps for pos 1 and 5)
      logo [10-20] color=charge           (Generates bits logo using the charge color scheme)
      logo #cluster_1# [1,5] 90%           (Reweights sequences at 90% identity)
      logo #cluster_1# [1,5] classic      (Generates bits logo using classic scheme)
      logo K10 [1] target_logo.png        (Logos pos 1 for K10 expr, saves as target_logo.png)
    """)

def run(viewer, args):
    if not args:
        msg = "Error: Logo command requires a POSITIONS parameter.\nUsage: logo [POSITIONS]"
        Command_Engine.print_help(viewer, msg)
        return

    if args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

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
            # Direct streaming to logomaker to support future presets/updates
            color_scheme = match.group(2)
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
        msg = f"Error: {exc}"
        Command_Engine.print_help(viewer, msg)
        return

    # 2. Extract Positions Argument (First argument containing brackets)
    bracket_indices = [i for i, a in enumerate(args) if a.startswith('[') and a.endswith(']')]
    
    if not bracket_indices:
        msg = "Error: No positions provided. Use [...] syntax."
        Command_Engine.print_help(viewer, msg)
        return
        
    pos_idx = bracket_indices[0]
    pos_str = args.pop(pos_idx)
    
    # 3. Handle Ambiguity & Assign Filename/Expression
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logo_{timestamp}.svg"
    automatic_filename = True
    expr = "$sele$" 
    
    if len(args) == 1:
        if args[0].lower().endswith(('.png', '.svg')) or args[0].startswith('['): 
            filename = args[0]
            automatic_filename = False
        else:
            expr = args[0]
    elif len(args) >= 2:
        filename = args.pop(-1)
        automatic_filename = False
        expr = "".join(args)

    try:
        filename = _normalize_logo_filename(filename)
    except ValueError as exc:
        Command_Engine.print_help(viewer, f"Error: {exc}")
        return

    # ---> NEW LOGIC: Smart Fallback to ALL Nodes <---
    if expr == "$sele$" and not getattr(viewer, 'selected_indices', []):
        expr = '"*"'  # The wildcard string matches all headers
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "No selection found. Defaulting to ALL nodes."
        print("No nodes selected. Defaulting to ALL nodes in the network.")

    # 4. Handle UI Selection dynamically
    if "$sele$" in expr.lower(): 
        header_dir = getattr(cfg, 'HEADER_LIST_DIR', os.path.join("Input_Files", "Header_Lists"))
        os.makedirs(header_dir, exist_ok=True)
        sele_path = os.path.join(header_dir, "_sele.txt")
        
        if hasattr(viewer, 'selected_indices') and viewer.selected_indices:
            with open(sele_path, "w", encoding="utf-8", newline="\n") as f:
                for idx in viewer.selected_indices:
                    f.write(viewer.full_headers[idx] + "\n")
        else:
            if os.path.exists(sele_path):
                open(sele_path, "w", encoding="utf-8").close()
                    
        # <--- UPDATED REGEX TO \$sele\$
        expr = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)

    # 5. Parse Position Array
    try:
        requested_positions = parse_logo_positions(pos_str)
    except ValueError as exc:
        msg = f"Error: {exc}"
        Command_Engine.print_help(viewer, msg)
        return

    if not requested_positions:
        msg = "Error: Could not parse positions from brackets."
        viewer.console_text.text = msg
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        msg = "Error: MSA not loaded in viewer. Please check inputs."
        viewer.console_text.text = msg
        return
    if len(alignment.aln) == 0:
        msg = (
            "Error: The selected MSA contains no aligned rows for the current network."
        )
        viewer.console_text.text = msg
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
        )
        selected_nodes = np.where(mask)[0]
    except Exception as e:
        Command_Engine.report_selection_error(viewer, expr, e, "Logo")
        return
        
    if len(selected_nodes) == 0:
        msg = "No nodes matched the criteria for logo generation."
        viewer.console_text.text = msg
        return

    # 7. Map Reference Sequence
    ref_id = getattr(viewer, 'active_reference', None) or getattr(cfg, 'ALIGNMENT_REFERENCE', '')
    ref_seq_str = None

    if ref_id:
        if hasattr(viewer.alignment.aln, 'header_map'): # Sparse mode
            for k, idx in viewer.alignment.aln.header_map.items():
                if ref_id in k:
                    ref_seq_str = str(viewer.alignment.aln[idx].seq)
                    break
        if not ref_seq_str: # Fallback / Legacy mode
            for r in viewer.alignment.aln:
                if ref_id in r.id or ref_id in r.description:
                    ref_seq_str = str(r.seq)
                    break
                
    if not ref_seq_str:
        print(f"Warning: Reference ID '{ref_id}' not found. Using the first sequence as reference.")
        ref_seq_str = str(viewer.alignment.aln[0].seq)

    valid_cols, plot_positions, missing_positions = resolve_reference_columns(
        viewer.alignment,
        requested_positions,
        ref_seq_str,
    )
    for position in missing_positions:
        print(f"Warning: Position {position} was not found in the active alignment mapping.")

    if not valid_cols:
        msg = "Error: Requested positions are outside the sequence bounds."
        viewer.console_text.text = msg
        return

    # 8. Extract Sequences for Selected Nodes
    selected_seqs = []
    for idx in selected_nodes:
        row_idx = int(viewer_to_aln[idx])
        if row_idx != -1:
            seq = str(viewer.alignment.aln[row_idx].seq)
            selected_seqs.append(seq)

    if not selected_seqs:
        msg = (
            "Error: No aligned nodes matched the logo selection criteria."
        )
        viewer.console_text.text = msg
        return

    # 9. Freeze the selected data and submit one background artifact job.
    logo_dir = getattr(
        cfg,
        'LOGO_DIR',
        os.path.join("Analysis_Results", "Sequence_Logos"),
    )
    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        Command_Engine.print_help(
            viewer,
            "Logo generation failed: the background job scheduler is unavailable.",
        )
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
            Command_Engine.print_help(
                viewer,
                "Logo generation failed: Output file is already reserved by a "
                f"background job: {output_path}",
            )
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
        Command_Engine.print_help(viewer, f"Logo generation failed: {exc}")
