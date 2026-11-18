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

"""The identity command: the pairwise % sequence identity of one or two node selections, read from the loaded MSA.

With one expression every matched sequence is compared with every other; with two, each sequence the first matches
is compared with each the second matches. The first expression's sequences are the rows and the second's the
columns, always. The matrix is written to a file (.xlsx by default, .csv or .tsv by its extension), and the Console
shows a summary and at most PREVIEW x PREVIEW cells of it.
"""

import csv
import io
import os
from datetime import datetime

import numpy as np

import EMAPSSN_Config as cfg
import Command_Engine
from utilities.Job_Cancellation import (
    CancellableFile, forget_partial_path, note_partial_path, raise_if_cancelled, remove_partial, replace_file,
)
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import new_partial_file, validate_output_basename
from utilities.Sequence_Utils import rejoin_split_brackets
from desktop.Command_Metadata import argument, entry, EXPRESSION

# --- What this command declares about itself ---
# Writes a file into the analysis results folder.
AGENT_POLICY = 'approve'

CATALOG_ENTRY = entry('Write the pairwise % sequence identity of one or two node selections, read from the loaded MSA, to an .xlsx, .csv or .tsv file; requires a nonempty MSA.',
    argument('expressions', EXPRESSION + ' One or two expressions, separated by a space. With one, every matched sequence is compared with every other; with two, each sequence the first matches with each the second matches. The first expression\'s sequences are the rows of the matrix and the second\'s the columns. Without an expression the selected nodes are used. Hidden nodes count; nodes without an MSA row are left out and counted. An expression that matches no node succeeds with a note and writes nothing.'),
    argument('positions', 'Optional brackets of displayed alignment positions, numbered as in query and logo: [45-120], [10,12,15.1]. Ranges expand to whole reference positions; insertion positions are listed explicitly. Without them a pair\'s identity is identical standard residues / alignment length incl. internal gaps, the length from the pair\'s first to its last aligned residue pair; with them, only those columns count and the length is every one of them where either sequence has a residue.'),
    argument('filename', 'Optional output name ending in .xlsx (the default), .csv or .tsv, written to the analysis results folder\'s Sequence_Identity folder; an existing file of that name is replaced. Without one, identity_<time>.xlsx. An .xlsx holds at most 4,000,000 cells and 16,383 columns, a .csv or .tsv 16,000,000 cells; a larger matrix is refused before anything runs.'),
    requires=('alignment',), writes_files=True)

# The Agent's instructions for this command, numbered in the order given (desktop.Command_Metadata).
AGENT_PROMPT_ORDER = 215
AGENT_PROMPT = """
`identity [EXPRESSION] [EXPRESSION] [POSITIONS] [FILENAME]`
- Requires a loaded MSA. Writes the pairwise % sequence identity of the sequences one expression matches (all against all) or of the first expression's sequences against the second's, as a matrix whose rows are the first expression's sequences and whose columns are the second's, to FILENAME in the Sequence_Identity results folder: `.xlsx` (the default, with Summary and mean sheets), `.csv` or `.tsv`. Without an expression, the selected nodes are used. Hidden nodes count; nodes without an MSA row are left out.
- A pair's identity is identical standard residues / alignment length incl. internal gaps, as the tools count a local alignment. With bracketed POSITIONS such as `[45-120]`, only those columns count and the length is every one of them where either sequence has a residue.
- The Console shows the mean, median, minimum and maximum, the sequences with the highest and lowest mean identity, and at most 20 x 20 cells of the matrix. An `.xlsx` holds at most 4,000,000 cells, a `.csv`/`.tsv` 16,000,000; a larger matrix is refused.
"""

IDENTITY_DIRECTORY = os.path.join("$analysis_result$", "Sequence_Identity")
FILE_FORMATS = ('.xlsx', '.csv', '.tsv')
# A larger matrix is refused before anything runs: an .xlsx writes about 0.15 million cells a second, a .csv about
# 2.4 million, and the matrix and its statistics hold a few copies of 8 bytes a cell.
MAX_CELLS = {'.xlsx': 4_000_000, '.csv': 16_000_000, '.tsv': 16_000_000}
XLSX_MAX_COLUMNS = 16_383  # Excel's 16,384 columns, one of them holding the row names
XLSX_MAX_ROWS = 1_048_575  # Excel's 1,048,576 rows, one of them holding the column names
PREVIEW = 20
# Rows of the matrix per write: each write is a check point where a stop request is honoured.
_ROWS_PER_WRITE = 200

LOCAL_DEFINITION = "identical standard residues / alignment length incl. internal gaps (as the tools count a local alignment)"
REGION_DEFINITION = ("identical standard residues / the selected columns where either sequence has a residue "
                     "(the global length over the region)")


def print_help():
    print("""
    Pairwise Identity Tool
    ======================
    Usage: identity [EXPRESSION] [EXPRESSION] [POSITIONS] [FILENAME]
           identity help

    Description:
      Computes the pairwise % sequence identity of sequences from the loaded MSA
      and writes the matrix to a file. The Console shows a summary and at most
      20 x 20 cells of the matrix.

      * One expression: every matched sequence against every other. The matrix
        is square and symmetric, and its diagonal is each sequence against itself.
      * Two expressions: each sequence the first matches against each the second
        matches. A node both match is not compared with itself in the summary.
      * The first expression's sequences are always the rows, and the second's
        the columns, in node order.
      * Without an expression, the selected nodes are used.
      * Hidden nodes count. Matched nodes without an MSA row are left out, and
        the summary counts them.

    How identity is counted:
      * Without positions, as the Pairwise Alignment and Embedding Search tools
        count a local alignment: identical standard residues / alignment length
        incl. internal gaps. For two MSA rows the length is the columns from
        their first to their last aligned residue pair (both have a residue)
        where either has a residue; overhanging ends are not counted.
      * With positions, only those columns count, and the length is every one
        of them where either sequence has a residue: a sequence missing part of
        the region counts as different there.
      * X and other non-standard letters count toward the length but never as
        identical. A pair with no length has no value (an empty cell, '-' in
        the Console). The tools align each pair themselves, so their value for
        a pair can differ from the MSA's.

    Positions:
      [POSITIONS] - optional displayed alignment positions in brackets, numbered
      as in query and logo: [45-120], [10,12,15.1] or [(-3)-(-1),0]. Ranges
      expand to whole reference positions; insertion positions (15.1) are listed
      explicitly. Positions missing from the mapping are named and skipped.

    Output file:
      [FILENAME] - a name ending in .xlsx (the default), .csv or .tsv, written to
      the Sequence_Identity folder of the analysis results. A file of that name is
      replaced. Without a name: identity_<date>_<time>.xlsx.
      * .xlsx: the matrix (sheet Identity), a Summary sheet, and each sequence's
        mean identity to the others (Row means; Column means with two
        expressions). Values have two decimals.
      * .csv / .tsv: the matrix alone, UTF-8.
      * An .xlsx holds at most 4,000,000 cells and 16,383 columns; a .csv or
        .tsv at most 16,000,000 cells. A larger matrix is refused before
        anything runs: narrow an expression, or use .csv/.tsv.

    Sequence Selection Expression Targets (Do NOT use spaces inside expressions!):
      The same as in query and color: AA positions such as P106, "header_text",
      @file@, #cluster_1#, $sele$ and {Length>500} metadata, joined with
      & (AND), | (OR), ! (NOT) and ^ (XOR).

    Examples:
      identity #cluster_3#                        (All against all in cluster 3)
      identity #cluster_3# #cluster_5#            (Cluster 3's sequences as rows, cluster 5's as columns)
      identity #cluster_3# [45-120] c3_core.csv   (Over positions 45-120 only, as CSV)
      identity #cluster_3# "P00350"               (Each member against one sequence)
    """)


def _fail(viewer, message):
    Command_Engine.print_help(viewer, message)
    Command_Engine.command_failed(viewer, message)


def _note(viewer, message):
    """A valid request that leaves nothing to compare: it succeeds, and says why nothing was written."""
    Command_Engine.show_status(viewer, message)
    print(message)
    Command_Engine.command_succeeded(viewer, message)


def _available_automatic_filename(scheduler, directory, filename):
    """The generated name, with a numeric suffix when a file or a queued job already has it."""
    stem, suffix = os.path.splitext(filename)
    candidate, index = filename, 2
    while True:
        path = os.path.abspath(os.path.join(directory, candidate))
        if not os.path.exists(path) and not scheduler.is_output_path_reserved(path):
            return candidate, path
        candidate = f"{stem}_{index}{suffix}"
        index += 1


def _split_arguments(args):
    """(expressions, position text or None, filename or None) from the arguments, or raise ValueError(Message)."""
    args = rejoin_split_brackets(args)
    brackets = [arg for arg in args if arg.strip().startswith('[') and arg.strip().endswith(']')]
    files = [arg for arg in args if arg.lower().endswith(FILE_FORMATS) and arg not in brackets]
    expressions = [arg for arg in args if arg not in brackets and arg not in files]
    if len(brackets) > 1:
        raise ValueError(Message(
            "Error: Give the positions in one [...] argument; found a second one: '{argument}'.",
            argument=brackets[1],
        ))
    if len(files) > 1:
        raise ValueError(Message(
            "Error: Give one output filename; found a second one: '{argument}'.", argument=files[1],
        ))
    if len(expressions) > 2:
        raise ValueError(Message(
            "Error: identity takes one or two expressions; found %n expression(s): {expressions}. "
            "Expressions have no spaces inside.",
            n=len(expressions), expressions=" ".join(expressions),
        ))
    positions = brackets[0].strip()[1:-1].strip() if brackets else None
    return expressions, positions, (files[0] if files else None)


def _check_expression(viewer, expression):
    """True when expression is a Boolean selection expression; otherwise report why and return False."""
    classification = Command_Engine.classify_selection_expression(expression)
    if classification.kind == Command_Engine.SelectionClassificationKind.VALID_EXPRESSION:
        return True
    if classification.kind == Command_Engine.SelectionClassificationKind.NOT_EXPRESSION:
        _fail(viewer, Message(
            "Error: Unrecognized identity argument '{argument}'. "
            "A filename must end in .xlsx, .csv or .tsv.",
            argument=expression,
        ))
        return False
    error = classification.error or Command_Engine.SelectionExpressionError(
        Message("'{expression}' is not a Boolean selection expression.", expression=expression)
    )
    Command_Engine.report_selection_error(viewer, expression, error, Message("Identity"))
    return False


def _matched_nodes(viewer, expression, viewer_to_aln, valid_indices):
    """The nodes expression matches, in node order, hidden ones included."""
    mask = Command_Engine.parse_advanced_expression(
        expression,
        viewer_to_aln,
        valid_indices,
        viewer.full_headers,
        getattr(viewer, 'cluster_labels', None),
        getattr(viewer, 'group_labels', None),
        viewer.alignment,
        metadata=getattr(viewer, 'metadata', None),
        selection_mask=Command_Engine.get_selected_mask(viewer),
        viewer=viewer,
    )
    return np.flatnonzero(mask)


def _region_columns(viewer, positions):
    """(MSA column indices, labels found, labels missing) for the bracketed positions; raise ValueError(Message)."""
    # logo holds the position grammar query and residue predicates share; it loads only when positions are given.
    from commands.logo import parse_logo_positions, resolve_reference_columns
    try:
        requested = parse_logo_positions(positions)
    except ValueError as exc:
        raise ValueError(Message("Error: {error}", error=exc))
    if not requested:
        raise ValueError(Message("Error: Could not parse positions from brackets."))
    columns, found, missing = resolve_reference_columns(viewer.alignment, requested)
    if not columns:
        raise ValueError(Message("Error: None of the requested positions is in the active alignment mapping."))
    # A position listed twice, or two labels on one column, counts its column once.
    unique_columns = sorted(set(int(column) for column in columns))
    return unique_columns, found, missing


def run(viewer, args):
    if args and args[0].lower() in ['help', '-h', '--help']:
        print_help()
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the Console"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the Console.')
        return

    alignment = getattr(viewer, 'alignment', None)
    if alignment is None or alignment.aln is None:
        _fail(viewer, Message("Error: No alignment loaded in the viewer."))
        return
    if len(alignment.aln) == 0:
        _fail(viewer, Message(
            "Error: The selected MSA contains no aligned rows for the current network."
        ))
        return

    # --- 1. The arguments: one or two expressions, positions and a filename, in any order ---
    try:
        expressions, positions, filename = _split_arguments(args)
    except ValueError as exc:
        _fail(viewer, exc.args[0])
        return
    if not expressions:
        expressions = ["$sele$"]
    for expression in expressions:
        if expression != "$sele$" and not _check_expression(viewer, expression):
            return

    automatic_filename = filename is None
    if automatic_filename:
        filename = f"identity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    try:
        filename = validate_output_basename(filename)
        file_format = next(suffix for suffix in FILE_FORMATS if filename.lower().endswith(suffix))
        if not filename[:-len(file_format)].strip():
            raise ValueError(Message("Filename '{file}' needs a name before its extension.", file=filename))
    except ValueError as exc:
        _fail(viewer, Message("Error: {error}", error=exc))
        return

    columns = None
    if positions is not None:
        try:
            columns, _found_positions, missing_positions = _region_columns(viewer, positions)
        except ValueError as exc:
            _fail(viewer, exc.args[0])
            return
        for position in missing_positions:
            print(f"Warning: Position {position} was not found in the active alignment mapping.")

    # --- 2. The sequences each expression matches ---
    viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
    viewer_to_aln = np.asarray(viewer_to_aln)
    sides = []
    for expression in expressions:
        if expression == "$sele$" and not Command_Engine.command_selection(viewer):
            _note(viewer, Message("No nodes are selected; give an expression or select nodes. Nothing was written."))
            return
        try:
            matched = _matched_nodes(viewer, expression, viewer_to_aln, valid_indices)
        except Exception as error:
            Command_Engine.report_selection_error(viewer, expression, error, Message("Identity"))
            return
        if len(matched) == 0:
            _note(viewer, Message(
                "No sequences matched the expression '{expression}'. Nothing was written.", expression=expression,
            ))
            return
        aligned = matched[viewer_to_aln[matched] >= 0]
        if len(aligned) == 0:
            _fail(viewer, Message(
                "Error: The expression '{expression}' matched %n node(s), but none is in the alignment.",
                n=len(matched), expression=expression,
            ))
            return
        sides.append((expression, aligned, matched[viewer_to_aln[matched] < 0]))

    rows, columns_nodes = sides[0][1], sides[-1][1]
    n_rows, n_columns = len(rows), len(columns_nodes)

    # --- 3. A matrix the file can hold, refused before anything runs ---
    cells = n_rows * n_columns
    if cells > MAX_CELLS[file_format] or (
            file_format == '.xlsx' and (n_columns > XLSX_MAX_COLUMNS or n_rows > XLSX_MAX_ROWS)):
        limit = Message("at most {cells} cells", cells=f"{MAX_CELLS[file_format]:,}")
        if file_format == '.xlsx':
            limit = Message("at most {cells} cells and {columns} columns",
                            cells=f"{MAX_CELLS[file_format]:,}", columns=f"{XLSX_MAX_COLUMNS:,}")
        _fail(viewer, Message(
            "Error: A {rows} x {columns} matrix ({cells} cells) is too large for a {format} file, which holds "
            "{limit}. Narrow an expression, or use .csv or .tsv.",
            rows=f"{n_rows:,}", columns=f"{n_columns:,}", cells=f"{cells:,}", format=file_format, limit=limit,
        ))
        return

    # --- 4. The data the job reads: its own copy of the rows it compares, nothing live ---
    msa_rows = np.unique(viewer_to_aln[np.concatenate([rows, columns_nodes])])
    frozen = alignment.aln.matrix[msa_rows].tocsr(copy=True)
    local_row = {int(row): index for index, row in enumerate(msa_rows.tolist())}
    headers = viewer.full_headers
    has_reference = bool(getattr(alignment, 'has_reference', False))
    payload = {
        "matrix": frozen,
        "int_to_aa": dict(alignment.aln.int_to_aa),
        "row_rows": np.asarray([local_row[int(viewer_to_aln[node])] for node in rows], dtype=np.int64),
        "col_rows": (None if len(sides) == 1 else
                     np.asarray([local_row[int(viewer_to_aln[node])] for node in columns_nodes], dtype=np.int64)),
        "row_nodes": rows,
        "col_nodes": columns_nodes,
        "row_headers": [str(headers[node]) for node in rows],
        "col_headers": [str(headers[node]) for node in columns_nodes],
        "expressions": tuple(expressions),
        "unaligned": tuple((side[0], [str(headers[node]) for node in side[2]]) for side in sides),
        "columns": columns,
        "positions": None if positions is None else f"[{positions}]",
        "numbering": (f"reference {getattr(alignment, 'resolved_ref_full', '')}" if has_reference else "occupancy"),
        "msa_file": str(getattr(alignment, 'msa_file', '') or ''),
        "filename": filename,
        "file_format": file_format,
    }

    # --- 5. One background job computes the matrix and writes the file ---
    scheduler = getattr(viewer, "background_job_scheduler", None)
    if scheduler is None:
        _fail(viewer, Message("Identity failed: the background job scheduler is unavailable."))
        return
    out_dir = cfg.resolve_directory_path(IDENTITY_DIRECTORY)
    if automatic_filename:
        filename, output_path = _available_automatic_filename(scheduler, out_dir, filename)
        payload["filename"] = filename
    else:
        output_path = os.path.abspath(os.path.join(out_dir, filename))
        if scheduler.is_output_path_reserved(output_path):
            _fail(viewer, Message(
                "Identity failed: {error}",
                error=Message("Output file is already reserved by a background job: {path}", path=output_path),
            ))
            return
    payload["out_dir"] = out_dir
    payload["output_path"] = output_path
    payload["allow_overwrite"] = not automatic_filename
    try:
        scheduler.enqueue(
            command_name="identity",
            description=f"identity {n_rows} x {n_columns} -> {filename}",
            payload=payload,
            worker=_write_identity_artifact,
            output_path=output_path,
            allow_overwrite=not automatic_filename,
        )
    except (FileExistsError, RuntimeError) as exc:
        _fail(viewer, Message("Identity failed: {error}", error=exc))
    else:
        Command_Engine.command_succeeded(
            viewer, f"Queued pairwise identity of {n_rows} x {n_columns} sequences for {output_path}; "
                    "waiting for the background job.")


# --- The background job ---

def _means(values, counted, axis):
    """Each row's (axis 1) or column's (axis 0) mean over the cells counted, and how many it has."""
    sums = np.where(counted, values, 0.0).sum(axis=axis)
    counts = counted.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan), counts


def _identity_summary(payload, values):
    """The statistics the Console and the Summary sheet give, as a dict.

    The pairs are two different sequences: with one expression each pair once
    (the cells above the diagonal), and a row's pairs every cell of it but its
    own; with two, every cell but a node against itself.
    """
    n_rows, n_columns = values.shape
    if payload["col_rows"] is None:
        pairs = np.triu(np.ones((n_rows, n_columns), dtype=bool), 1)
        row_pairs = ~np.eye(n_rows, dtype=bool)
    else:
        pairs = row_pairs = np.asarray(payload["row_nodes"])[:, None] != np.asarray(payload["col_nodes"])[None, :]
    has_value = ~np.isnan(values)
    present = values[pairs & has_value]
    row_means, row_counts = _means(values, row_pairs & has_value, 1)
    col_means = col_counts = None
    if payload["col_rows"] is not None:
        col_means, col_counts = _means(values, row_pairs & has_value, 0)
    summary = {
        "pairs": int(pairs.sum()),
        "with_value": int(present.size),
        "self_pairs": (0 if payload["col_rows"] is None else int((~pairs).sum())),
        "row_means": row_means, "row_counts": row_counts,
        "col_means": col_means, "col_counts": col_counts,
    }
    if present.size:
        summary.update(mean=float(present.mean()), median=float(np.median(present)),
                       minimum=float(present.min()), maximum=float(present.max()))
    return summary


def _summary_lines(payload, values, summary):
    """The Console's summary, line by line (also the Summary sheet's first rows)."""
    n_rows, n_columns = values.shape
    expressions = payload["expressions"]
    if len(expressions) == 1:
        lines = [f"Pairwise % identity of {n_rows:,} sequences ({expressions[0]}), all against all: "
                 f"{summary['pairs']:,} pairs."]
    else:
        lines = [f"Pairwise % identity: rows {expressions[0]} ({n_rows:,} sequences), columns {expressions[1]} "
                 f"({n_columns:,} sequences): {summary['pairs']:,} pairs."]
        if summary["self_pairs"]:
            lines.append(f"{summary['self_pairs']:,} node(s) both expressions match are not compared with "
                         "themselves in these figures.")
    if payload["columns"] is None:
        lines.append(f"Identity: {LOCAL_DEFINITION}, over all MSA columns.")
    else:
        lines.append(f"Identity: {REGION_DEFINITION}, over positions {payload['positions']} "
                     f"({len(payload['columns']):,} columns).")
    if "mean" in summary:
        lines.append(f"Mean {summary['mean']:.1f}%, median {summary['median']:.1f}%, "
                     f"minimum {summary['minimum']:.1f}%, maximum {summary['maximum']:.1f}%.")
        means = summary["row_means"]
        if np.any(~np.isnan(means)):
            highest, lowest = int(np.nanargmax(means)), int(np.nanargmin(means))
            other = "the others" if payload["col_rows"] is None else "the columns"
            lines.append(f"Highest mean identity to {other}: {payload['row_headers'][highest]} "
                         f"({means[highest]:.1f}%).")
            lines.append(f"Lowest mean identity to {other}: {payload['row_headers'][lowest]} "
                         f"({means[lowest]:.1f}%).")
    else:
        lines.append("No pair has a value to summarize.")
    missing = summary["pairs"] - summary["with_value"]
    if missing:
        lines.append(f"{missing:,} pair(s) share no aligned residue and have no value.")
    for expression, unaligned in payload["unaligned"]:
        if unaligned:
            lines.append(f"{len(unaligned):,} node(s) '{expression}' matched have no MSA row and are left out.")
    return lines


def _preview_lines(payload, values):
    """At most PREVIEW x PREVIEW cells of the matrix, with what each row and column is."""
    n_rows, n_columns = values.shape
    shown_rows, shown_columns = min(n_rows, PREVIEW), min(n_columns, PREVIEW)
    symmetric = payload["col_rows"] is None
    row_names = [f"{i + 1}" if symmetric else f"r{i + 1}" for i in range(shown_rows)]
    col_names = [f"{j + 1}" if symmetric else f"c{j + 1}" for j in range(shown_columns)]
    width = max(len(name) for name in row_names)
    lines = [" " * width + "".join(f"{name:>7}" for name in col_names)]
    for i, name in enumerate(row_names):
        cells = "".join("      -" if np.isnan(value) else f"{value:7.1f}" for value in values[i, :shown_columns])
        lines.append(f"{name:<{width}}{cells}")
    if shown_rows < n_rows or shown_columns < n_columns:
        lines.append(f"(The first {shown_rows} of {n_rows:,} rows and {shown_columns} of {n_columns:,} columns; "
                     "the file holds them all.)")
    if symmetric:
        lines.extend(f"{name:>{width}} = {header}" for name, header in zip(row_names, payload["row_headers"]))
    else:
        lines.extend(f"{name:>{width}} = {header}" for name, header in zip(row_names, payload["row_headers"]))
        col_width = max(len(name) for name in col_names)
        lines.extend(f"{name:>{col_width}} = {header}" for name, header in zip(col_names, payload["col_headers"]))
    return lines


def _cell_values(row):
    """A matrix row for a file: two decimals, None where a pair has no value."""
    return [None if value != value else value for value in np.round(row, 2).tolist()]


def _write_xlsx(target, payload, values, summary, summary_lines):
    import openpyxl
    workbook = openpyxl.Workbook(write_only=True)
    sheet = workbook.create_sheet("Identity")
    sheet.freeze_panes = "B2"
    sheet.append(["% identity"] + payload["col_headers"])
    for index, header in enumerate(payload["row_headers"]):
        if index % _ROWS_PER_WRITE == 0:
            raise_if_cancelled()
        sheet.append([header] + _cell_values(values[index]))

    info = workbook.create_sheet("Summary")
    for line in summary_lines:
        info.append([line])
    info.append([])
    expressions = payload["expressions"]
    rows = [
        ("Rows", expressions[0]),
        ("Columns", expressions[1] if len(expressions) > 1 else "the rows (all against all)"),
        ("Row sequences", values.shape[0]),
        ("Column sequences", values.shape[1]),
        ("Positions", payload["positions"] or "all MSA columns"),
        ("Columns counted", len(payload["columns"]) if payload["columns"] is not None else "all"),
        ("Identity", REGION_DEFINITION if payload["columns"] is not None else LOCAL_DEFINITION),
        ("Numbering", payload["numbering"]),
        ("MSA", payload["msa_file"]),
        ("Pairs", summary["pairs"]),
        ("Pairs with a value", summary["with_value"]),
    ]
    for key in ("mean", "median", "minimum", "maximum"):
        if key in summary:
            rows.append((f"{key.capitalize()} % identity", round(summary[key], 2)))
    for item in rows:
        info.append(list(item))
    for expression, unaligned in payload["unaligned"]:
        if unaligned:
            info.append([])
            info.append([f"Nodes '{expression}' matched without an MSA row"])
            for header in unaligned:
                info.append([header])

    def mean_sheet(title, headers, means, counts):
        sheet = workbook.create_sheet(title)
        sheet.freeze_panes = "A2"
        sheet.append(["Sequence", "Mean % identity", "Pairs with a value"])
        for header, mean, count in zip(headers, means.tolist(), counts.tolist()):
            sheet.append([header, None if mean != mean else round(mean, 2), int(count)])

    mean_sheet("Row means", payload["row_headers"], summary["row_means"], summary["row_counts"])
    if summary["col_means"] is not None:
        mean_sheet("Column means", payload["col_headers"], summary["col_means"], summary["col_counts"])
    raise_if_cancelled()
    workbook.save(target)


def _write_delimited(target, payload, values, delimiter):
    """The matrix alone, as UTF-8 bytes, a block of rows per write."""
    def encoded(rows):
        text = io.StringIO()
        writer = csv.writer(text, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)
        return text.getvalue().encode("utf-8")

    target.write(encoded([["% identity"] + payload["col_headers"]]))
    for start in range(0, values.shape[0], _ROWS_PER_WRITE):
        raise_if_cancelled()
        block = []
        for index in range(start, min(values.shape[0], start + _ROWS_PER_WRITE)):
            block.append([payload["row_headers"][index]]
                         + ["" if value is None else f"{value:.2f}" for value in _cell_values(values[index])])
        target.write(encoded(block))


def _write_identity_artifact(payload):
    """Compute the matrix, print its summary and preview, and write the file, without the live viewer."""
    from utilities.Alignment_Profiles import identity_matrix

    raise_if_cancelled()
    values = identity_matrix(payload["matrix"], payload["int_to_aa"], payload["row_rows"],
                             payload["col_rows"], payload["columns"])
    raise_if_cancelled()
    summary = _identity_summary(payload, values)
    summary_lines = _summary_lines(payload, values, summary)
    print("\n".join(["", *summary_lines, *_preview_lines(payload, values)]))

    out_dir = payload["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_path = payload["output_path"]
    allow_overwrite = payload["allow_overwrite"]
    if not allow_overwrite and os.path.exists(save_path):
        raise FileExistsError(Message("Output file already exists: {path}", path=save_path))
    suffix = payload["file_format"]
    partial_path = new_partial_file(out_dir, f".{os.path.splitext(payload['filename'])[0]}.", f".partial{suffix}")
    # Named to the scheduler, which deletes it if the Viewer closes while this job is stuck.
    named_partial = partial_path
    note_partial_path(named_partial)
    try:
        # Every write is a check point (CancellableFile), so a large file stops while it is saved.
        with CancellableFile.at(partial_path, "wb") as target:
            if suffix == ".xlsx":
                _write_xlsx(target, payload, values, summary, summary_lines)
            else:
                _write_delimited(target, payload, values, "," if suffix == ".csv" else "\t")
        raise_if_cancelled()
        if not allow_overwrite and os.path.exists(save_path):
            raise FileExistsError(Message("Output file already exists: {path}", path=save_path))
        replace_file(partial_path, save_path)
        partial_path = None
    finally:
        if partial_path:
            remove_partial(partial_path)
        forget_partial_path(named_partial)

    message = Message(
        "Saved the pairwise identity of {rows} x {columns} sequences to {file}",
        rows=f"{values.shape[0]:,}", columns=f"{values.shape[1]:,}", file=payload["filename"],
    )
    if "mean" in summary:
        message = JoinedMessage([message, Message("(mean {mean}%).", mean=f"{summary['mean']:.1f}")], separator=" ")
    return {"message": message, "save_path": save_path}
