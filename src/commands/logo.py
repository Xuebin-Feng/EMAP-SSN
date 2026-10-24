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

import math
import os
import re
import tempfile
import numpy as np
from datetime import datetime  # <--- NEW IMPORT
import EMAPSSN_Config as cfg
import Command_Engine
from utilities import Alignment_Profiles as _profiles
from utilities.Alignment_Profiles import (
    NUMBA_AVAILABLE,
    STANDARD_AAS,
    calculate_identity_weights,
    calculate_logo_matrix,
    choose_balanced_thread_count,
    get_balanced_thread_count,
    get_num_threads,
    run_identity_neighbour_counts,
    set_num_threads,
)
from utilities.Job_Cancellation import CancellableFile, forget_partial_path, note_partial_path, raise_if_cancelled
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import validate_output_basename
from utilities.Sequence_Utils import (
    DISPLAYED_POSITION_ATOM_PATTERN,
    normalize_displayed_position_atom,
    reject_bare_negative_positions,
)

LOGO_DIRECTORY = os.path.join("$analysis_result$", "Sequence_Logos")

# The computation these names stand for is in utilities.Alignment_Profiles, which
# `label` and the residue data of `esmfold` use too. The names imported above
# and assigned below are the ones this module and its tests have always had,
# and they are the same objects. To change what the computation does (a Numba
# switch, a stand-in kernel), patch the utility, not these.
_ASCII_TO_STANDARD_CODE = _profiles._ASCII_TO_STANDARD_CODE
_DECODE_BLOCK_ROWS = _profiles._DECODE_BLOCK_ROWS
_IDENTITY_CHUNKS_PER_THREAD = _profiles._IDENTITY_CHUNKS_PER_THREAD
_IDENTITY_TILE_ROWS = _profiles._IDENTITY_TILE_ROWS
_aligned_row_strings = _profiles.aligned_row_strings
_calculate_identity_neighbour_counts_numpy = _profiles._calculate_identity_neighbour_counts_numpy
_encode_standard_amino_acids = _profiles.encode_standard_amino_acids
_pack_identity_planes = _profiles._pack_identity_planes
_unique_encoded_rows = _profiles._unique_encoded_rows
if NUMBA_AVAILABLE:
    _identity_neighbour_counts_kernel = _profiles._identity_neighbour_counts_kernel
    _popcount64 = _profiles._popcount64

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

# The logo figure: it is wide enough for every position (never narrower than
# the minimum), and is saved at one dpi. The renderer and the PNG size check
# both read these, so the check measures what the renderer draws.
_LOGO_FIGURE_HEIGHT_INCHES = 4
_LOGO_MIN_FIGURE_WIDTH_INCHES = 6
_LOGO_WIDTH_PER_POSITION_INCHES = 0.5
_LOGO_FIGURE_PADDING_INCHES = 1
_LOGO_DPI = 600

# A PNG logo is drawn as one raster of 4 bytes (RGBA) per pixel. Agg, the
# renderer, refuses a side of 2^23 pixels or more (matplotlib 3.11, the pinned
# version; older versions stopped at 2^16), and the raster of a logo of a few
# hundred positions already takes about a gigabyte, with about as much again
# while it is saved. A test reads the Agg limit from matplotlib itself.
_AGG_MAX_SIDE_PIXELS = 2 ** 23
_MAX_PNG_RASTER_BYTES = 1 << 30


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


def _logo_figure_width(position_count):
    """The logo figure's width in inches for that many plotted positions."""
    return max(
        _LOGO_MIN_FIGURE_WIDTH_INCHES,
        position_count * _LOGO_WIDTH_PER_POSITION_INCHES + _LOGO_FIGURE_PADDING_INCHES,
    )


def _png_logo_raster(position_count):
    """The (width, height, RGBA bytes) of the raster a PNG logo of that many positions is drawn on.

    Agg draws the whole figure at the save dpi before bbox_inches='tight' crops
    it by about 0.1 inch on a side, so the full figure is the size that counts.
    """
    width = math.ceil(_logo_figure_width(position_count) * _LOGO_DPI)
    height = math.ceil(_LOGO_FIGURE_HEIGHT_INCHES * _LOGO_DPI)
    return width, height, width * height * 4


def _png_logo_fits(position_count):
    """Whether a PNG logo of that many positions is within Agg's size limit and the memory budget."""
    width, height, raster_bytes = _png_logo_raster(position_count)
    return (
        max(width, height) < _AGG_MAX_SIDE_PIXELS
        and raster_bytes <= _MAX_PNG_RASTER_BYTES
    )


def _max_png_logo_positions():
    """The most positions a PNG logo can have and still fit, or 0 if none can."""
    fitting, too_many = 0, _MAX_LOGO_RANGE_POSITIONS + 1
    while too_many - fitting > 1:
        middle = (fitting + too_many) // 2
        if _png_logo_fits(middle):
            fitting = middle
        else:
            too_many = middle
    return fitting


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

    raise_if_cancelled()
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
    raise_if_cancelled()
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

    fig_width = _logo_figure_width(len(plot_positions))
    fig = Figure(figsize=(fig_width, _LOGO_FIGURE_HEIGHT_INCHES))
    FigureCanvasAgg(fig)
    ax = fig.subplots()

    partial_path = named_partial = None
    try:
        raise_if_cancelled()
        logo = logomaker.Logo(dataframe, ax=ax, color_scheme=payload["color_scheme"])
        raise_if_cancelled()

        base_gap = 0.01
        max_gap = base_gap if mode == "pcts" else base_gap * np.log2(20)

        for patch in ax.patches:
            raise_if_cancelled()
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

        _configure_logo_y_axis(ax, mode, gap_mode)

        # The x label is fitted to the axes, whose width the layout settles
        # first (the label sits below them and does not narrow them).
        raise_if_cancelled()
        fig.tight_layout()
        _fit_position_axis_label(ax, payload)
        fig.tight_layout()
        raise_if_cancelled()
        suffix = os.path.splitext(filename)[1].lower()
        file_descriptor, partial_path = tempfile.mkstemp(
            prefix=f".{os.path.splitext(filename)[0]}.",
            suffix=f".partial{suffix}",
            dir=logo_dir,
        )
        os.close(file_descriptor)
        # Named to the scheduler, which deletes it if the Viewer closes while this job is stuck.
        named_partial = partial_path
        note_partial_path(named_partial)
        # Every write of the image is a check point (CancellableFile): an SVG goes out as
        # the figure is drawn and a PNG as it is encoded, so a big one stops while it is saved.
        # The file is opened as matplotlib opens a path for the format (an SVG as text, with
        # the platform's line ends), so the image is the same as before; and only at its
        # first write, so nothing holds it open while the figure is drawn.
        with (CancellableFile.at(partial_path, "w", "utf-8") if suffix == ".svg"
              else CancellableFile.at(partial_path, "wb")) as target:
            fig.savefig(
                target,
                format=suffix.lstrip("."),
                transparent=filename.lower().endswith('.png'),
                dpi=_LOGO_DPI,
                bbox_inches='tight',
            )
        raise_if_cancelled()
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
        if named_partial is not None:
            forget_partial_path(named_partial)

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


def _position_axis_label(payload, header=None):
    """Describe the numbering the plotted position labels actually use.

    HEADER is the reference header the label names, in place of the payload's.
    """
    if payload.get("numbering") == "occupancy":
        return "Position (occupancy numbering)"
    if header is None:
        header = payload["ref_id"]
    return f"Position (relative to {header})"


# The label is fitted a little narrower than the axes: its width is measured at
# the figure's dpi and drawn at the save dpi, where the glyph widths differ a little.
_AXIS_LABEL_FIT = 0.97


def _fit_position_axis_label(ax, payload):
    """Label the x axis, shortening a long reference header so the label is no wider than the axes.

    The label is centered under the axes, so a wider one would make the saved
    image wider than the logo. The header is cut at the longest length that
    fits and ends in an ellipsis; a label that fits is left as it is.
    """
    label = ax.set_xlabel(_position_axis_label(payload))
    if payload.get("numbering") == "occupancy":
        return
    renderer = ax.figure.canvas.get_renderer()
    available = ax.get_window_extent(renderer).width * _AXIS_LABEL_FIT
    header = str(payload["ref_id"])

    def fits(header_text):
        label.set_text(_position_axis_label(payload, header_text))
        return label.get_window_extent(renderer).width <= available

    def shortened(kept):
        return header[:kept].rstrip() + "\u2026"

    if fits(header):
        return
    # Binary search for the longest cut that fits; the whole header does not,
    # and a cut of nothing but the ellipsis is the least the label can say.
    longest, too_long = 0, len(header)
    while too_long - longest > 1:
        middle = (longest + too_long) // 2
        if fits(shortened(middle)):
            longest = middle
        else:
            too_long = middle
    label.set_text(_position_axis_label(payload, shortened(longest)))


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
    print(f"""
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

      * The x axis names the reference header the positions are numbered
        against. A header too long for the axis is cut short with an ellipsis,
        so it never widens the image.

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
                       A PNG is drawn as one 600 dpi image, so a PNG logo may have at
                       most {_max_png_logo_positions()} positions (about 1 GB of image) and is refused
                       beyond that, before any job is queued. An SVG has no such limit.
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
      A valid expression may match zero nodes: the command reports that none
      matched and queues no job. Nodes that match but are not in the alignment
      leave nothing to analyse, and the command fails.

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

    # A PNG is one raster at 600 dpi, so a logo of enough positions cannot be
    # drawn; SVG has no such limit. Refuse it here, not in the background job.
    if os.path.splitext(filename)[1].lower() == ".png" and not _png_logo_fits(len(plot_positions)):
        msg = Message(
            "Error: A PNG logo of %n position(s) is too large to draw; at most {limit} "
            "positions fit. Use an .svg filename, which has no such limit, or fewer positions.",
            n=len(plot_positions), limit=_max_png_logo_positions(),
        )
        Command_Engine.command_failed(viewer, msg)
        Command_Engine.print_help(viewer, msg)
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
