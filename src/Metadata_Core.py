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

"""Metadata operations shared by the desktop viewer and the VR front end.

These functions are the metadata *data model*: reading a spreadsheet into
``viewer.metadata``, writing it back out, and deleting columns atomically.
They contain no user interface. Every viewer-specific hook they use - history
recording, view refresh, state broadcast, the HUD property - is probed with
``getattr``/``hasattr`` and skipped when absent, which is what lets one
implementation serve a Qt viewer with a web spreadsheet and a headless viewer
with neither.

They used to live in ``web_ui/meta_backend.py``, whose module-level
``from PySide6 import ...`` made them unreachable from a headless process even
though nothing in them touches Qt. ``opt_vr`` therefore could not offer
``meta`` upload, download or delete without copying some four hundred lines -
the kind of duplication that had already caused one round of drift. Moving
them here costs the desktop viewer nothing: ``meta_backend`` re-exports every
name, so its own callers and ``commands/meta.py`` are unchanged.

``Command_Engine`` resolves to the desktop engine in one process and to the VR
engine in the other; both expose the reporting helpers and the selection
grammar used below.
"""

import math
import os
import re
import uuid

import numpy as np
import pandas as pd

import Command_Engine
from utilities.Localization import JoinedMessage, Message
from utilities.Output_Names import validate_output_basename

# Spreadsheet formats `meta download` writes, matched in any case.
METADATA_DOWNLOAD_EXTENSIONS = (".csv", ".xlsx")

# Row keys the metadata web table adds to every row; a metadata column of the
# same name would overwrite them, so an upload refuses these names.
RESERVED_METADATA_COLUMN_NAMES = ("id", "Node ID")

# Type-row values `meta upload` reads as a number or as text; any other value
# in the type row is read as text, with a warning.
NUMBER_TYPE_NAMES = ("number", "num", "numerical")
TEXT_TYPE_NAMES = ("text", "string", "str")

# read_csv / read_excel options that keep pandas' default NA words as text.
# Only a cell with nothing in it is blank; see `_metadata_cell_is_blank`.
_READ_CELLS_AS_WRITTEN = {"keep_default_na": False, "na_filter": False}


class MetadataColumnDeleteError(ValueError):
    """Raised when a metadata-column deletion request is not atomic and valid."""

def _is_integral_number(value):
    """Report a floating metadata value that carries no fractional part."""
    if isinstance(value, (float, np.floating)):
        return float(value).is_integer()
    return False

def format_metadata_value(value):
    """Render one metadata value for display, without a decimal point when integral."""
    if _is_integral_number(value):
        return f"{float(value):.0f}"
    return str(value)

def export_metadata_value(value):
    """Return one metadata value for file export, keeping spreadsheet cells numeric."""
    if _is_integral_number(value):
        return int(value)
    return value

def _metadata_cell_is_blank(value):
    """True for a spreadsheet cell with nothing in it: empty, whitespace or missing.

    Words such as "NA" or "None" are values, not blanks; whether a number
    column can read them is decided where the column is converted.
    """
    return bool(pd.isna(value)) or str(value).strip() == ""

def _parse_metadata_value(prop_type, value):
    if prop_type == "number":
        if str(value).strip() == "":
            return np.nan
        return float(value)
    return str(value)

def _metadata_values_equal(left, right):
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False

def _record_metadata_cell_history(viewer, column, row, before, after):
    save_compact = getattr(viewer, "_save_metadata_cell_history", None)
    if callable(save_compact):
        save_compact(column, row, before, after)
    elif hasattr(viewer, "_save_state"):
        viewer._save_state()

def metadata_state_event(viewer):
    return {
        "type": "state_updated",
        "visible_mask": viewer.visible_mask.tolist(),
        "selected_indices": viewer.selected_indices,
        "metadata": viewer.get_serializable_metadata(),
        "columns": ["Node ID"] + list(viewer.metadata.keys()),
        "types": {
            key: entry["type"] for key, entry in viewer.metadata.items()
        },
    }

def broadcast_metadata_state(viewer):
    broadcast = getattr(viewer, "broadcast_metadata_state", None)
    if callable(broadcast):
        broadcast()
    elif hasattr(viewer, "broadcast_event"):
        viewer.broadcast_event(metadata_state_event(viewer))

def _resolve_metadata_column_names(viewer, requested_names):
    if not isinstance(requested_names, (list, tuple)):
        raise MetadataColumnDeleteError(Message(
            "Metadata columns must be supplied as a list of property names."
        ))

    normalized = [str(name).strip() for name in requested_names]
    if not normalized or any(not name for name in normalized):
        raise MetadataColumnDeleteError(Message(
            "Specify at least one metadata property to delete."
        ))
    if any(name.casefold() == "all" for name in normalized):
        raise MetadataColumnDeleteError(Message(
            "Deleting all metadata columns at once is not supported."
        ))

    protected = {"node id", "sequence header"}
    protected_requested = [
        name for name in normalized if name.casefold() in protected
    ]
    if protected_requested:
        raise MetadataColumnDeleteError(Message(
            "Protected columns cannot be deleted: {names}.",
            names=", ".join(protected_requested),
        ))

    available = list(getattr(viewer, "metadata", {}).keys())
    folded = {}
    for available_name in available:
        folded.setdefault(available_name.casefold(), []).append(available_name)

    resolved = []
    missing = []
    ambiguous = []
    for requested in normalized:
        if requested in viewer.metadata:
            actual = requested
        else:
            matches = folded.get(requested.casefold(), [])
            if not matches:
                missing.append(requested)
                continue
            if len(matches) > 1:
                ambiguous.append(requested)
                continue
            actual = matches[0]
        if actual not in resolved:
            resolved.append(actual)

    problems = []
    if missing:
        problems.append(Message("not found: {names}", names=", ".join(missing)))
    if ambiguous:
        problems.append(Message("case-ambiguous: {names}", names=", ".join(ambiguous)))
    if problems:
        available_text = ", ".join(available) if available else Message("none")
        raise MetadataColumnDeleteError(Message(
            "Cannot delete metadata columns ({problems}). Available properties: {available}.",
            problems=JoinedMessage(problems, separator="; "),
            available=available_text,
        ))
    return resolved

def delete_metadata_columns(viewer, requested_names, broadcast=True):
    """Atomically delete resolved metadata columns and record compact history."""
    resolved = _resolve_metadata_column_names(viewer, requested_names)
    metadata_names = list(viewer.metadata.keys())
    removed_columns = []
    for name in resolved:
        entry = viewer.metadata[name]
        removed_columns.append({
            "name": name,
            "index": metadata_names.index(name),
            "type": entry["type"],
            "values": entry["values"].copy(),
        })

    active_hud = getattr(viewer, "meta_display_prop", None)
    hud_property = active_hud if active_hud in resolved else None
    save_compact = getattr(viewer, "_save_metadata_column_history", None)
    if callable(save_compact):
        save_compact(removed_columns, hud_property=hud_property)
    elif hasattr(viewer, "_save_state"):
        viewer._save_state()

    for name in resolved:
        del viewer.metadata[name]

    if hud_property:
        set_hud = getattr(viewer, "_set_metadata_hud_property", None)
        if callable(set_hud):
            set_hud(None)
        else:
            viewer.meta_display_prop = None
            display = getattr(viewer, "hud_displays", {}).get("meta_display")
            if display is not None:
                display.hide()

    if broadcast:
        broadcast_metadata_state(viewer)
    return resolved

def upload_metadata(viewer, file_paths):
    """Parses and merges Excel/CSV metadata into viewer.metadata."""
    # One undo entry for the whole upload, taken just before the first file
    # changes metadata, so an upload that fails validation leaves history alone.
    state_saved = False

    successful_files = []
    failed_files = []
    matched_nodes = set()
    total_unmatched = 0
    all_merged_props = set()

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        if not os.path.exists(filepath):
            failed_files.append((filename, Message("File not found.")))
            continue

        try:
            # Cells are read as written: pandas' default NA words ("NA",
            # "N/A", "None", "null", "NaN", ...) stay text instead of turning
            # into blanks, since in a text column they can be real values
            # (NA for North America or Namibia). An empty cell comes back as
            # "" (a spreadsheet's error cell as NaN) and is blank below; a
            # number column reads these words as missing when converting.
            if ext.lower() == ".csv":
                try:
                    df = pd.read_csv(filepath, header=None, dtype=str, **_READ_CELLS_AS_WRITTEN)
                except UnicodeDecodeError:
                    # Excel saves a plain "CSV" in the system's Windows code
                    # page, not in UTF-8.
                    print(f"Note: {filename} is not valid UTF-8; reading it as Windows-1252 (cp1252).")
                    df = pd.read_csv(
                        filepath, header=None, dtype=str, encoding="cp1252", **_READ_CELLS_AS_WRITTEN
                    )
            else:
                df = pd.read_excel(filepath, header=None, **_READ_CELLS_AS_WRITTEN)

            if df.shape[0] < 3 or df.shape[1] < 2:
                raise ValueError(Message(
                    "Invalid file format. Must contain at least sequence headers and one property column."
                ))

            prop_names = []
            valid_cols = []
            for col_idx in range(1, df.shape[1]):
                val = df.iloc[0, col_idx]
                if pd.notna(val) and str(val).strip():
                    prop_names.append(str(val).strip())
                    valid_cols.append(col_idx)

            if not prop_names:
                raise ValueError(Message("No valid property names found in the first row."))

            illegal_props = [prop for prop in prop_names if not re.match(r'^[a-zA-Z0-9_\-\.]+$', prop)]
            if illegal_props:
                raise ValueError(Message(
                    "Property names {names} contain illegal characters. "
                    "Allowed characters are: letters, numbers, underscores (_), hyphens (-), and periods (.)",
                    names=", ".join([repr(p) for p in illegal_props]),
                ))

            reserved_props = [prop for prop in prop_names if prop in RESERVED_METADATA_COLUMN_NAMES]
            if reserved_props:
                raise ValueError(Message(
                    "Property names {names} are reserved for the metadata table's row index. "
                    "Rename the column and upload the file again.",
                    names=", ".join([repr(p) for p in reserved_props]),
                ))

            seen_props = set()
            for prop in prop_names:
                if prop in seen_props:
                    print(f"Warning: Property '{prop}' appears in more than one column of {filename}; "
                          "the columns are merged, and a later column's values replace an earlier one's.")
                seen_props.add(prop)

            prop_types = []
            for prop, col_idx in zip(prop_names, valid_cols):
                val = df.iloc[1, col_idx]
                if pd.notna(val) and str(val).strip():
                    t = str(val).strip().lower()
                    if t in NUMBER_TYPE_NAMES:
                        prop_types.append('number')
                    else:
                        if t not in TEXT_TYPE_NAMES:
                            print(f"Warning: Property '{prop}' in {filename} has the unrecognized type "
                                  f"'{str(val).strip()}'; treating it as text. "
                                  "Recognized types are: number, num, numerical, text.")
                        prop_types.append('text')
                else:
                    prop_types.append('text')

            header_to_idx = {h: idx for idx, h in enumerate(viewer.full_headers)}
            node_updates = {}
            matched_count = 0
            unmatched_count = 0

            # Each column is read out of the table once, rather than cell by
            # cell, which costs pandas far more than the cell itself.
            first_column = df.iloc[:, 0].to_numpy(dtype=object).tolist()
            for df_row_idx in range(2, df.shape[0]):
                header_val = first_column[df_row_idx]
                if _metadata_cell_is_blank(header_val):
                    unmatched_count += 1
                    continue
                header_str = str(header_val).strip()
                
                if header_str in header_to_idx:
                    node_idx = header_to_idx[header_str]
                    node_updates[node_idx] = df_row_idx
                    matched_count += 1
                else:
                    unmatched_count += 1

            if matched_count == 0:
                raise ValueError(Message(
                    "No matching sequence headers found. Enforced strict exact matching against full headers."
                ))

            if not state_saved:
                viewer._save_state()
                state_saved = True

            for p_idx, prop_name in enumerate(prop_names):
                prop_type = prop_types[p_idx]
                col_idx = valid_cols[p_idx]

                if prop_name not in viewer.metadata:
                    if prop_type == 'number':
                        values = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                    else:
                        values = np.full(viewer.n_nodes, "", dtype=object)
                    viewer.metadata[prop_name] = {
                        "type": prop_type,
                        "values": values
                    }
                else:
                    old_type = viewer.metadata[prop_name]["type"]
                    viewer.metadata[prop_name]["type"] = prop_type
                    
                    if old_type != prop_type:
                        old_vals = viewer.metadata[prop_name]["values"]
                        if prop_type == 'number':
                            new_vals = np.full(viewer.n_nodes, np.nan, dtype=np.float64)
                            for i in range(viewer.n_nodes):
                                try:
                                    if str(old_vals[i]).strip():
                                        new_vals[i] = float(old_vals[i])
                                except ValueError:
                                    pass
                            viewer.metadata[prop_name]["values"] = new_vals
                        else:
                            new_vals = np.full(viewer.n_nodes, "", dtype=object)
                            for i in range(viewer.n_nodes):
                                if pd.notna(old_vals[i]):
                                    new_vals[i] = str(old_vals[i])
                            viewer.metadata[prop_name]["values"] = new_vals

                values_arr = viewer.metadata[prop_name]["values"]
                column = df.iloc[:, col_idx].to_numpy(dtype=object)
                # A blank cell is missing (pd.isna, as _metadata_cell_is_blank
                # tests it, but for the whole column at once) or holds only
                # whitespace.
                missing = pd.isna(column).tolist()
                column = column.tolist()
                for node_idx, df_row_idx in node_updates.items():
                    if missing[df_row_idx]:
                        continue
                    cell_val = column[df_row_idx]
                    if str(cell_val).strip() == "":
                        continue

                    if prop_type == 'number':
                        # Anything that is not a number is missing: "NA",
                        # "N/A", "None" and "null" as much as "abc". "nan"
                        # parses but is missing too.
                        try:
                            number = float(cell_val)
                        except (ValueError, TypeError):
                            continue
                        if not math.isnan(number):
                            values_arr[node_idx] = number
                    else:
                        # A text column keeps every word as written, "NA" and
                        # "NaN" included; only an empty cell is blank.
                        values_arr[node_idx] = str(cell_val)

            successful_files.append(filename)
            matched_nodes.update(node_updates.keys())
            total_unmatched += unmatched_count
            all_merged_props.update(prop_names)

        except Exception as e:
            failed_files.append((filename, e))
            print(f"Error uploading metadata from {filename}: {e}")
            Command_Engine.command_failed(viewer, f'Error uploading metadata from {filename}: {e}')

    msg_parts = []
    if successful_files:
        msg_parts += [
            Message(
                "Successfully uploaded metadata from %n file(s): {files}.",
                n=len(successful_files),
                files=", ".join(successful_files),
            ),
            Message("Matched %n unique node(s).", n=len(matched_nodes)),
            Message("Ignored %n row(s).", n=total_unmatched),
            Message("Merged properties: {properties}.", properties=", ".join(sorted(all_merged_props))),
        ]
        broadcast_metadata_state(viewer)
    if failed_files:
        fail_details = JoinedMessage(
            [Message("{file}: {error}", file=f, error=err) for f, err in failed_files], separator="; "
        )
        msg_parts.append(Message("Failed to upload from %n file(s): {details}", n=len(failed_files), details=fail_details))

    msg = JoinedMessage(msg_parts)
    Command_Engine.print_help(viewer, msg)
    if successful_files and not failed_files:
        Command_Engine.command_succeeded(viewer, msg)

def metadata_download_path(meta_dir, filename=""):
    """Return the file `meta download [filename]` writes in ``meta_dir``.

    A requested name must be a plain file name ending in .csv or .xlsx, in any
    case; a name without an extension gets .csv. Without a name, the next free
    metadata.csv, metadata1.csv, ... is chosen. A refused name raises
    ValueError before anything is written.
    """
    if filename:
        filename = validate_output_basename(filename)
        _, ext = os.path.splitext(filename)
        if not ext:
            filename += ".csv"
        elif ext.lower() not in METADATA_DOWNLOAD_EXTENSIONS:
            raise ValueError(Message(
                "Metadata can only be downloaded as .csv or .xlsx, not '{extension}'.", extension=ext
            ))
        return os.path.join(meta_dir, filename)

    base_name = "metadata"
    ext = ".csv"
    candidate = f"{base_name}{ext}"
    filepath = os.path.join(meta_dir, candidate)
    counter = 1
    while os.path.exists(filepath):
        candidate = f"{base_name}{counter}{ext}"
        filepath = os.path.join(meta_dir, candidate)
        counter += 1
    return os.path.abspath(filepath)

def _download_column(entry, nodes):
    """One property's cells for NODES as `meta download` writes them.

    Returns (cells, valid): a number is written as export_metadata_value
    gives it and a missing one as ""; a text value as stored, and "" for
    None or whitespace. VALID marks the nodes whose cell holds a value.
    """
    stored = entry["values"]
    if isinstance(stored, np.ndarray):
        values = stored[nodes]
    else:
        values = [stored[i] for i in nodes.tolist()]
    if entry["type"] == "number":
        if isinstance(values, np.ndarray) and values.dtype.kind == "f":
            valid = ~np.isnan(values)
            # Integral values are written as ints and the rest as the stored
            # float scalars, as export_metadata_value writes them one by one.
            integral = valid & np.isfinite(values) & (np.floor(values) == values)
            cells = list(values)
            for position in np.flatnonzero(~valid).tolist():
                cells[position] = ""
            for position in np.flatnonzero(integral).tolist():
                cells[position] = int(cells[position])
            return cells, valid
        valid = np.fromiter((bool(pd.notna(value)) for value in values), dtype=bool, count=len(values))
        cells = [export_metadata_value(value) if ok else "" for value, ok in zip(values, valid.tolist())]
        return cells, valid
    cells = list(values)
    valid = np.fromiter(
        (value is not None and str(value).strip() != "" for value in cells), dtype=bool, count=len(cells)
    )
    for position in np.flatnonzero(~valid).tolist():
        cells[position] = ""
    return cells, valid


def download_metadata(viewer, filepath, expr=None):
    """Downloads network metadata to a file, applying optional logic filters."""
    if not getattr(viewer, 'metadata', None):
        message = Message("Error: No metadata available in the viewer to download.")
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return False

    try:
        mask = np.ones(viewer.n_nodes, dtype=bool)
        if expr:
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
                    metadata=viewer.metadata,
                    selection_mask=Command_Engine.get_selected_mask(viewer),
                )
            except Command_Engine.SelectionExpressionError as error:
                Command_Engine.report_selection_error(
                    viewer,
                    expr,
                    error,
                    Message("Metadata export"),
                )
                return False
            
            if np.sum(mask) == 0:
                message = Message("Error: No nodes matched the expression '{expression}'.", expression=expr)
                Command_Engine.print_help(viewer, message)
                Command_Engine.command_failed(viewer, message)
                return False

        prop_names = list(viewer.metadata.keys())

        row_0 = [""] + prop_names
        row_1 = [""] + [viewer.metadata[p]["type"] for p in prop_names]

        # The table is built a column at a time; each row then holds the same
        # cells, in the same order, as one built node by node.
        nodes = np.flatnonzero(mask)
        headers = viewer.full_headers
        columns = [[headers[i] for i in nodes.tolist()]]
        has_valid_prop = np.zeros(len(nodes), dtype=bool)
        for p in prop_names:
            cells, valid = _download_column(viewer.metadata[p], nodes)
            columns.append(cells)
            has_valid_prop |= valid
        kept = range(len(nodes)) if expr else np.flatnonzero(has_valid_prop).tolist()
        rows = [row_0, row_1]
        rows.extend([column[k] for column in columns] for k in kept)

        df = pd.DataFrame(rows)

        # Write beside the target and move the finished file over it, so a
        # failure part-way (a control character in a text value, say) leaves
        # an earlier export in place instead of a partial file.
        folder, name = os.path.split(filepath)
        stem, ext = os.path.splitext(name)
        temp_path = os.path.join(folder, f"{stem}.{uuid.uuid4().hex[:8]}.partial{ext}")
        try:
            if ext.lower() == ".csv":
                df.to_csv(temp_path, header=False, index=False)
            elif ext.lower() == ".xlsx":
                # pandas checks a path's extension case-sensitively and refuses
                # ".XLSX"; a file handle has none to check and gets the same
                # default writer as ".xlsx".
                with open(temp_path, "wb") as handle:
                    df.to_excel(handle, header=False, index=False)
            else:
                df.to_excel(temp_path, header=False, index=False)
            os.replace(temp_path, filepath)
        except BaseException:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise

        if expr:
            msg = Message(
                "Metadata successfully downloaded to {path} (filtered by: {expression})",
                path=filepath,
                expression=expr,
            )
        else:
            msg = Message("Metadata successfully downloaded to {path}", path=filepath)
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_artifact(viewer, filepath)
        Command_Engine.command_succeeded(viewer, msg)
        return True
    except Exception as e:
        message = Message("Error downloading metadata: {error}", error=e)
        Command_Engine.print_help(viewer, message)
        Command_Engine.command_failed(viewer, message)
        return False
