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

"""
commands/meta.py — Thin CLI portal stub for the SSN Viewer metadata spreadsheet.

Delegates core backend models, widgets, uploads, and downloads to web_ui/meta_backend.py.
"""

import os
import re
import pandas as pd
import Command_Engine
import SSN_Config as cfg
from web_ui.meta_backend import (
    MetadataColumnDeleteError,
    delete_metadata_columns,
    register,
    upload_metadata,
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
      meta <filename>
          Uploads and merges the specified metadata file (.xlsx, .xls, .csv) into the 
          current viewer session. The path can be absolute, relative, or located 
          inside the metadata directory: {meta_dir}
      meta download
          Downloads the current session metadata to a generic file (e.g. metadata.csv, 
          or metadata1.csv if already taken) in {meta_dir}.
      meta download <filename>
          Downloads the metadata using the specified filename (defaults to .csv if 
          no extension is provided). Overwrites the file if it already exists.
      meta show/display <property_name>
          Displays the selected property above the bottom-right status indicators
          whenever a node is clicked.
      meta show/display clear/off
          Clears and removes the metadata property display.
      meta delete/remove/clear <property_name> [property_name ...]
          Atomically deletes one or more metadata columns from the current session.
          Property matching is case-insensitive. Node ID/Sequence Header cannot be
          deleted; Length is deletable metadata. Deleting every column with "all"
          is not supported.
      meta help
          Displays this help message.

    Examples:
      meta
      meta my_data.xlsx
      meta download
      meta download my_exported_data
      meta show Organism
      meta show clear
      meta delete Organism Taxonomy
      meta remove Host
      meta clear Source
    """)

def run(viewer, args):
    # Retrieve configuration directory for metadata files
    meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Input_Files", "Meta_Data"))
    os.makedirs(meta_dir, exist_ok=True)

    # 1. Registration callback support
    # Register sidebar button when called alone, or with upload, or via startup flag
    should_register = (not args or 
                       (args and args[0].lower() not in ['help', '-h', '--help', 'show', 'display', 'download', 'retrieve', 'export', 'delete', 'remove', 'clear', 'off', 'deactivate']) or
                       (args and args[0] == '--register-only'))

    if should_register:
        register(viewer)

    if args and args[0] == '--register-only':
        return

    # 2. No arguments: Open spreadsheet browser page
    if not args:
        viewer.open_metadata_ui()
        return

    first_arg = args[0].lower()

    # 3. Help & Usage Check
    if first_arg in ['help', '-h', '--help']:
        print_help(meta_dir)
        if hasattr(viewer, 'console_text'):
            viewer.console_text.text = "Help information printed to the terminal"
        return

    # 4. Delete Metadata Columns
    if first_arg in ['delete', 'remove', 'clear']:
        if len(args) < 2:
            Command_Engine.print_help(
                viewer,
                f"Usage: meta {first_arg} <property_name> [property_name ...]",
            )
            return
        try:
            deleted = delete_metadata_columns(
                viewer, args[1:], broadcast=False
            )
        except MetadataColumnDeleteError as error:
            Command_Engine.print_help(viewer, f"Error: {error}")
            return
        Command_Engine.print_help(
            viewer, "Deleted metadata columns: " + ", ".join(deleted) + "."
        )
        return

    # 5. Display/Show Property Check
    if first_arg in ['display', 'show']:
        if len(args) < 2:
            Command_Engine.print_help(viewer, "Usage: meta show <property_name> OR meta show clear/off")
            return

        prop_name = " ".join(args[1:]).strip()
        if prop_name.lower() in ['clear', 'off']:
            if 'meta_display' in viewer.hud_displays:
                viewer.hud_displays['meta_display'].hide()
            viewer.meta_display_prop = None
            Command_Engine.print_help(viewer, "Metadata display cleared.")
            return

        if not prop_name:
            Command_Engine.print_help(viewer, "Error: Please specify a valid property name.")
            return

        available_props = list(viewer.metadata.keys()) if getattr(viewer, 'metadata', None) else []
        resolved_prop = None
        for p in available_props:
            if p.lower() == prop_name.lower():
                resolved_prop = p
                break

        if not resolved_prop:
            resolved_prop = prop_name
            if available_props:
                print(f"Warning: Property '{prop_name}' not found in current metadata. Available properties: {', '.join(available_props)}")

        viewer.meta_display_prop = resolved_prop

        if 'meta_display' not in viewer.hud_displays:
            from SSN_Viewer import HUDDisplay
            
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
                        val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
                        self.show(f"{p_name}: {val_str}")
                    else:
                        self.show(f"{p_name}: N/A")

            viewer.hud_displays['meta_display'] = MetaHUDDisplay(viewer)

        display = viewer.hud_displays['meta_display']
        node_idx = getattr(viewer, 'selected_node_idx', None)
        if node_idx is not None and getattr(viewer, 'metadata', None) and resolved_prop in viewer.metadata:
            val = viewer.metadata[resolved_prop]["values"][node_idx]
            val_str = str(val).strip() if pd.notna(val) and val is not None else "N/A"
            display.show(f"{resolved_prop}: {val_str}")
        else:
            display.show(f"{resolved_prop}: -")

        Command_Engine.print_help(viewer, f"Metadata display enabled for property: '{resolved_prop}'")
        return

    # 6. Download Check
    if first_arg in ['download', 'retrieve', 'export']:
        filename = ""
        if len(args) >= 2:
            filename = " ".join(args[1:]).strip()

        if filename:
            _, ext = os.path.splitext(filename)
            if not ext:
                filename += ".csv"
            filepath = os.path.join(meta_dir, filename)
        else:
            base_name = "metadata"
            ext = ".csv"
            candidate = f"{base_name}{ext}"
            filepath = os.path.join(meta_dir, candidate)
            counter = 1
            while os.path.exists(filepath):
                candidate = f"{base_name}{counter}{ext}"
                filepath = os.path.join(meta_dir, candidate)
                counter += 1
            filepath = os.path.abspath(filepath)

        download_metadata(viewer, filepath)
        return

    # 7. Upload Check (Treat first argument as filename to upload)
    upload_args = list(args)
    if first_arg in ['upload', 'import']:
        upload_args = args[1:]
        if not upload_args:
            Command_Engine.print_help(viewer, "Error: Please specify a file path or filename to upload.")
            return

    file_paths = []
    for arg in upload_args:
        path = arg.strip()
        if os.path.exists(path):
            file_paths.append(os.path.abspath(path))
        else:
            path_in_dir = os.path.join(meta_dir, path)
            if os.path.exists(path_in_dir):
                file_paths.append(os.path.abspath(path_in_dir))
            else:
                found = False
                for ext in ['.xlsx', '.xls', '.csv']:
                    if os.path.exists(path + ext):
                        file_paths.append(os.path.abspath(path + ext))
                        found = True
                        break
                    elif os.path.exists(os.path.join(meta_dir, path + ext)):
                        file_paths.append(os.path.abspath(os.path.join(meta_dir, path + ext)))
                        found = True
                        break
                if not found:
                    Command_Engine.print_help(viewer, f"Error: Metadata file '{path}' not found (checked absolute, relative, and {meta_dir}).")
                    return

    upload_metadata(viewer, file_paths)
