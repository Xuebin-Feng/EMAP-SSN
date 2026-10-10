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

"""
commands/meta.py — Thin CLI portal stub for the EMAP-SSN Viewer metadata spreadsheet.

Delegates core backend models, widgets, uploads, and downloads to web_ui/meta_backend.py.
"""

import os
import re
import pandas as pd
import Command_Engine
import EMAPSSN_Config as cfg
from Viewer_Command_Portal import CURRENT
from desktop.Desktop_App import translate
from utilities.Localization import Message
from web_ui.Browser_Page import page_is_open
from web_ui.meta_backend import (
    MetadataColumnDeleteError,
    delete_metadata_columns,
    format_metadata_value,
    register,
    upload_metadata,
    metadata_download_path,
    download_metadata,
)

def print_help(meta_dir):
    print(f"""
    Node Metadata Manager CLI Portal
    ================================
    Usage:
      meta
          Opens the HTML5 Metadata Spreadsheet in your web browser and registers 
          the "📊 Meta Data" sidebar shortcut button.
      meta [upload|import] <filename> [filename ...]
          Uploads and merges one or more metadata files (.xlsx, .xls, .csv; the
          extension may be omitted) into the current viewer session. Each path can
          be absolute, relative, or located inside the metadata directory: {meta_dir}
          A single path may contain spaces, with or without quotes around it.
      meta download
          Downloads the current session metadata to a generic file (e.g. metadata.csv, 
          or metadata1.csv if already taken) in {meta_dir}.
      meta download <filename>
          Downloads the metadata to a file of that name in {meta_dir}. The
          name must be a plain file name, not a path, ending in .csv or .xlsx
          (.csv is added when no extension is given). Overwrites the file if
          it already exists.
      meta show/display <property_name>
          Displays the selected property above the bottom-right status indicators
          whenever a node is clicked.
      meta show/display clear/off
          Clears and removes the metadata property display.
      meta delete/remove/clear <property_name> [property_name ...]
          Atomically deletes one or more metadata columns from the current session.
          Property matching is case-insensitive. Node ID/Sequence Header cannot be
          deleted; the generated Length, kDa, pI, and GRAVY columns are deletable
          metadata. Deleting every column with "all" is not supported.
      meta help
          Displays this help message.

    Examples:
      meta
      meta my_data.xlsx
      meta upload traits.xlsx taxonomy.csv
      meta download
      meta download my_exported_data
      meta show Organism
      meta show clear
      meta delete Organism Taxonomy
      meta remove Host
      meta clear Source
    """)

def _strip_matching_quotes(text):
    """Remove one pair of matching single or double quotes around text."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    return text

def _find_metadata_file(path, meta_dir):
    """Return the absolute path a metadata file name refers to, or None."""
    if os.path.exists(path):
        return os.path.abspath(path)
    path_in_dir = os.path.join(meta_dir, path)
    if os.path.exists(path_in_dir):
        return os.path.abspath(path_in_dir)
    for ext in ['.xlsx', '.xls', '.csv']:
        if os.path.exists(path + ext):
            return os.path.abspath(path + ext)
        elif os.path.exists(os.path.join(meta_dir, path + ext)):
            return os.path.abspath(os.path.join(meta_dir, path + ext))
    return None

def run(viewer, args):
    # Retrieve configuration directory for metadata files
    meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Input_Files", "Meta_Data"))

    # 1. Registration callback support
    # Register sidebar button when called alone, or with upload, or via startup flag
    should_register = (not args or 
                       (args and args[0].lower() not in ['help', '-h', '--help', 'show', 'display', 'download', 'retrieve', 'export', 'delete', 'remove', 'clear', 'off', 'deactivate']) or
                       (args and args[0] == '--register-only'))

    if should_register:
        register(viewer)

    if args and args[0] == '--register-only':
        Command_Engine.command_succeeded(viewer, 'Registered the metadata interface.')
        return

    # 2. No arguments: Open spreadsheet browser page
    if not args:
        # A command from the MCP or the agent has nobody at the Viewer to
        # dismiss a dialog, so an already-open page is reported on the status
        # line only.
        options = {"show_existing_dialog": False} if CURRENT.get() is not None else {}
        if viewer.open_metadata_ui(**options):
            Command_Engine.command_succeeded(viewer, "Opened the metadata interface.")
        elif page_is_open(viewer, "meta"):
            Command_Engine.command_failed(viewer, "The metadata interface is already open in your browser.")
        else:
            Command_Engine.command_failed(viewer, "Could not open the metadata interface.")
        return

    first_arg = args[0].lower()

    # 3. Help & Usage Check
    if first_arg in ['help', '-h', '--help']:
        print_help(meta_dir)
        if hasattr(viewer, 'console_text'):
            Command_Engine.show_status(viewer, Message("Help information printed to the terminal"))
        Command_Engine.command_succeeded(viewer, 'Help information printed to the terminal.')
        return

    # 4. Delete Metadata Columns
    if first_arg in ['delete', 'remove', 'clear']:
        if len(args) < 2:
            Command_Engine.command_failed(viewer, "Missing metadata command arguments")
            Command_Engine.print_help(
                viewer,
                Message("Usage: {syntax}", syntax=f"meta {first_arg} <property_name> [property_name ...]"),
            )
            return
        try:
            deleted = delete_metadata_columns(
                viewer, args[1:], broadcast=False
            )
        except MetadataColumnDeleteError as error:
            msg = Message("Error: {error}", error=error)
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return
        msg = Message("Deleted metadata columns: {columns}.", columns=", ".join(deleted))
        Command_Engine.print_help(viewer, msg)
        Command_Engine.command_succeeded(viewer, msg)
        return

    # 5. Display/Show Property Check
    if first_arg in ['display', 'show']:
        if len(args) < 2:
            Command_Engine.command_failed(viewer, "Missing metadata command arguments")
            Command_Engine.print_help(
                viewer,
                Message(
                    "Usage: {syntax} OR {other_syntax}",
                    syntax="meta show <property_name>",
                    other_syntax="meta show clear/off",
                ),
            )
            return

        prop_name = " ".join(args[1:]).strip()
        if prop_name.lower() in ['clear', 'off']:
            if 'meta_display' in viewer.hud_displays:
                viewer.hud_displays['meta_display'].hide()
            viewer.meta_display_prop = None
            msg = Message("Metadata display cleared.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_succeeded(viewer, msg)
            return

        if not prop_name:
            msg = Message("Error: Please specify a valid property name.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

        available_props = list(viewer.metadata.keys()) if getattr(viewer, 'metadata', None) else []
        # An exact name wins over a name that differs only in case.
        resolved_prop = prop_name if prop_name in available_props else None
        if resolved_prop is None:
            for p in available_props:
                if p.lower() == prop_name.lower():
                    resolved_prop = p
                    break

        if not resolved_prop:
            msg = Message(
                "Error: Property '{property}' not found in current metadata. Available properties: {available}.",
                property=prop_name,
                available=", ".join(available_props) if available_props else Message("none"),
            )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

        viewer.meta_display_prop = resolved_prop

        if 'meta_display' not in viewer.hud_displays:
            from EMAPSSN_Viewer import HUDDisplay
            
            class MetaHUDDisplay(HUDDisplay):
                def __init__(self, main_viewer):
                    super().__init__(
                        viewer=main_viewer,
                        name='meta_display',
                        pos_fn=lambda size: main_viewer._status_hud_position(2, size),
                        anchor_x='right',
                        anchor_y='bottom'
                    )
                
                def on_node_clicked(self, node_idx):
                    p_name = getattr(self.viewer, 'meta_display_prop', None)
                    if p_name and getattr(self.viewer, 'metadata', None) and p_name in self.viewer.metadata:
                        val = self.viewer.metadata[p_name]["values"][node_idx]
                        val_str = (
                            format_metadata_value(val).strip()
                            if pd.notna(val) and val is not None
                            else translate("Viewer", "N/A")
                        )
                        self.show(f"{p_name}: {val_str}")
                    else:
                        not_available = translate("Viewer", "N/A")
                        self.show(f"{p_name}: {not_available}")

            viewer.hud_displays['meta_display'] = MetaHUDDisplay(viewer)

        display = viewer.hud_displays['meta_display']
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None and getattr(viewer, 'metadata', None) and resolved_prop in viewer.metadata:
            val = viewer.metadata[resolved_prop]["values"][node_idx]
            val_str = (
                format_metadata_value(val).strip()
                if pd.notna(val) and val is not None
                else translate("Viewer", "N/A")
            )
            display.show(f"{resolved_prop}: {val_str}")
        else:
            display.show(f"{resolved_prop}: -")

        Command_Engine.print_help(
            viewer, Message("Metadata display enabled for property: '{property}'", property=resolved_prop)
        )
        Command_Engine.command_succeeded(viewer, f"Metadata display enabled for property: {resolved_prop!r}")
        return

    # 6. Download Check
    if first_arg in ['download', 'retrieve', 'export']:
        os.makedirs(meta_dir, exist_ok=True)
        try:
            filepath = metadata_download_path(meta_dir, " ".join(args[1:]).strip())
        except ValueError as error:
            msg = Message("Error: {error}", error=error)
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

        download_metadata(viewer, filepath)
        return

    # 7. Upload Check (Treat first argument as filename to upload)
    upload_args = list(args)
    if first_arg in ['upload', 'import']:
        upload_args = args[1:]
        if not upload_args:
            msg = Message("Error: Please specify a file path or filename to upload.")
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

    # A path with spaces reaches here as several arguments: try the whole
    # remainder as one file first, and treat the arguments as separate files
    # only when no such file exists.
    file_paths = []
    whole_path = None
    if len(upload_args) > 1:
        whole_path = _find_metadata_file(_strip_matching_quotes(" ".join(upload_args).strip()), meta_dir)
    if whole_path:
        file_paths.append(whole_path)
    else:
        for arg in upload_args:
            path = _strip_matching_quotes(arg.strip())
            found = _find_metadata_file(path, meta_dir)
            if found:
                file_paths.append(found)
                continue
            if first_arg in ['off', 'deactivate'] and not file_paths:
                # These belong to the agent command and are not metadata
                # options; say so instead of looking for a file of that name
                # (one that exists is still uploaded, as it always was).
                msg = Message(
                    "Error: '{option}' is not a metadata option. To clear the metadata display, use {syntax}.",
                    option=path,
                    syntax="meta show clear",
                )
            else:
                msg = Message(
                    "Error: Metadata file '{file}' not found (checked absolute, relative, and {folder}).",
                    file=path,
                    folder=meta_dir,
                )
            Command_Engine.print_help(viewer, msg)
            Command_Engine.command_failed(viewer, msg)
            return

    upload_metadata(viewer, file_paths)
