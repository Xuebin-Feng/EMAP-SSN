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

import os
import re

import numpy as np
import pandas as pd

import Command_Engine


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

def refresh_metadata_views(viewer):
    refresh = getattr(viewer, "_refresh_metadata_views", None)
    if callable(refresh):
        refresh()
        return

    source_model = getattr(viewer, "metadata_source_model", None)
    if source_model is not None and hasattr(source_model, "refresh_columns"):
        source_model.refresh_columns()

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
        raise MetadataColumnDeleteError(
            "Metadata columns must be supplied as a list of property names."
        )

    normalized = [str(name).strip() for name in requested_names]
    if not normalized or any(not name for name in normalized):
        raise MetadataColumnDeleteError(
            "Specify at least one metadata property to delete."
        )
    if any(name.casefold() == "all" for name in normalized):
        raise MetadataColumnDeleteError(
            "Deleting all metadata columns at once is not supported."
        )

    protected = {"node id", "sequence header"}
    protected_requested = [
        name for name in normalized if name.casefold() in protected
    ]
    if protected_requested:
        raise MetadataColumnDeleteError(
            "Protected columns cannot be deleted: "
            + ", ".join(protected_requested)
            + "."
        )

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
        problems.append("not found: " + ", ".join(missing))
    if ambiguous:
        problems.append("case-ambiguous: " + ", ".join(ambiguous))
    if problems:
        available_text = ", ".join(available) if available else "none"
        raise MetadataColumnDeleteError(
            "Cannot delete metadata columns ("
            + "; ".join(problems)
            + f"). Available properties: {available_text}."
        )
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

    refresh_metadata_views(viewer)
    if broadcast:
        broadcast_metadata_state(viewer)
    return resolved

def upload_metadata(viewer, file_paths):
    """Parses and merges Excel/CSV metadata into viewer.metadata."""
    viewer._save_state()

    successful_files = []
    failed_files = []
    matched_nodes = set()
    total_unmatched = 0
    all_merged_props = set()

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        _, ext = os.path.splitext(filename)
        if not os.path.exists(filepath):
            failed_files.append((filename, "File not found."))
            continue

        try:
            if ext.lower() == ".csv":
                df = pd.read_csv(filepath, header=None, dtype=str)
            else:
                df = pd.read_excel(filepath, header=None)

            if df.shape[0] < 3 or df.shape[1] < 2:
                raise ValueError("Invalid file format. Must contain at least sequence headers and one property column.")

            prop_names = []
            valid_cols = []
            for col_idx in range(1, df.shape[1]):
                val = df.iloc[0, col_idx]
                if pd.notna(val) and str(val).strip():
                    prop_names.append(str(val).strip())
                    valid_cols.append(col_idx)

            if not prop_names:
                raise ValueError("No valid property names found in the first row.")

            illegal_props = [prop for prop in prop_names if not re.match(r'^[a-zA-Z0-9_\-\.]+$', prop)]
            if illegal_props:
                raise ValueError(
                    f"Property names {', '.join([repr(p) for p in illegal_props])} contain illegal characters. "
                    "Allowed characters are: letters, numbers, underscores (_), hyphens (-), and periods (.)"
                )

            prop_types = []
            for col_idx in valid_cols:
                val = df.iloc[1, col_idx]
                if pd.notna(val) and str(val).strip():
                    t = str(val).strip().lower()
                    if t in ['number', 'num', 'numerical']:
                        prop_types.append('number')
                    else:
                        prop_types.append('text')
                else:
                    prop_types.append('text')

            header_to_idx = {h: idx for idx, h in enumerate(viewer.full_headers)}
            node_updates = {}
            matched_count = 0
            unmatched_count = 0

            for df_row_idx in range(2, df.shape[0]):
                header_val = df.iloc[df_row_idx, 0]
                if pd.isna(header_val):
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
                raise ValueError("No matching sequence headers found. Enforced strict exact matching against full headers.")

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
                for node_idx, df_row_idx in node_updates.items():
                    cell_val = df.iloc[df_row_idx, col_idx]
                    if pd.isna(cell_val) or str(cell_val).strip() == "" or str(cell_val).strip().lower() == "nan":
                        continue
                    
                    if prop_type == 'number':
                        try:
                            values_arr[node_idx] = float(cell_val)
                        except (ValueError, TypeError):
                            pass
                    else:
                        values_arr[node_idx] = str(cell_val)

            successful_files.append(filename)
            matched_nodes.update(node_updates.keys())
            total_unmatched += unmatched_count
            all_merged_props.update(prop_names)

        except Exception as e:
            failed_files.append((filename, str(e)))
            print(f"Error uploading metadata from {filename}: {e}")
            Command_Engine.command_failed(viewer, f'Error uploading metadata from {filename}: {e}')

    msg_parts = []
    if successful_files:
        msg_parts.append(
            f"Successfully uploaded metadata from {len(successful_files)} file(s): {', '.join(successful_files)}. "
            f"Matched {len(matched_nodes)} unique nodes, ignored {total_unmatched} rows. "
            f"Merged properties: {', '.join(sorted(all_merged_props))}."
        )
        broadcast_metadata_state(viewer)
    if failed_files:
        fail_details = "; ".join([f"{f}: {err}" for f, err in failed_files])
        msg_parts.append(f"Failed to upload from {len(failed_files)} file(s): {fail_details}")

    msg = " ".join(msg_parts)
    Command_Engine.print_help(viewer, msg)
    if successful_files and not failed_files:
        Command_Engine.command_succeeded(viewer, msg)

def download_metadata(viewer, filepath, expr=None):
    """Downloads network metadata to a file, applying optional logic filters."""
    if not getattr(viewer, 'metadata', None):
        Command_Engine.print_help(viewer, "Error: No metadata available in the viewer to download.")
        Command_Engine.command_failed(viewer, 'Error: No metadata available in the viewer to download.')
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
                    "Metadata export",
                )
                return False
            
            if np.sum(mask) == 0:
                Command_Engine.print_help(viewer, f"Error: No nodes matched the expression '{expr}'.")
                Command_Engine.command_failed(viewer, f"Error: No nodes matched the expression '{expr}'.")
                return False

        prop_names = list(viewer.metadata.keys())
        
        row_0 = [""] + prop_names
        row_1 = [""] + [viewer.metadata[p]["type"] for p in prop_names]
        
        rows = [row_0, row_1]
        for i in range(viewer.n_nodes):
            if not mask[i]:
                continue
            has_valid_prop = False
            row_val = [viewer.full_headers[i]]
            for p in prop_names:
                val = viewer.metadata[p]["values"][i]
                if viewer.metadata[p]["type"] == "number":
                    if pd.notna(val):
                        has_valid_prop = True
                        row_val.append(export_metadata_value(val))
                    else:
                        row_val.append("")
                else:
                    if val is not None and str(val).strip() != "":
                        has_valid_prop = True
                        row_val.append(val)
                    else:
                        row_val.append("")
            
            if has_valid_prop or expr:
                rows.append(row_val)

        df = pd.DataFrame(rows)

        _, ext = os.path.splitext(filepath)
        if ext.lower() == ".csv":
            df.to_csv(filepath, header=False, index=False)
        else:
            df.to_excel(filepath, header=False, index=False)

        msg = f"Metadata successfully downloaded to {filepath}"
        if expr:
            msg += f" (filtered by: {expr})"
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_artifact(viewer, filepath)
        Command_Engine.command_succeeded(viewer, msg)
        return True
    except Exception as e:
        Command_Engine.print_help(viewer, f"Error downloading metadata: {e}")
        Command_Engine.command_failed(viewer, f'Error downloading metadata: {e}')
        return False
