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
import numpy as np
from PySide6 import QtWidgets, QtCore
import Command_Engine
import EMAPSSN_Config as cfg
from desktop.Desktop_App import translate
from web_ui.Plugin_Manager import ensure_registry
# The metadata data model lives in Metadata_Core so the headless VR front
# end can use it too; re-exported here so this module's own callers and
# commands/meta.py keep importing them from where they always have.
from Metadata_Core import (
    MetadataColumnDeleteError,
    _is_integral_number,
    format_metadata_value,
    export_metadata_value,
    _parse_metadata_value,
    _metadata_values_equal,
    _record_metadata_cell_history,
    metadata_state_event,
    broadcast_metadata_state,
    _resolve_metadata_column_names,
    delete_metadata_columns,
    upload_metadata,
    metadata_download_path,
    download_metadata,
)


def handle_delete_columns(viewer, data):
    try:
        deleted = delete_metadata_columns(viewer, data.get("columns"))
    except MetadataColumnDeleteError as error:
        message = str(error)
        if hasattr(viewer, "broadcast_event"):
            viewer.broadcast_event({
                "type": "metadata_error",
                "message": message,
            })
        Command_Engine.print_help(viewer, f"Error: {message}")
        Command_Engine.command_failed(viewer, f'Error: {message}')
        return False

    Command_Engine.print_help(
        viewer, "Deleted metadata columns: " + ", ".join(deleted) + "."
    )
    return True


def handle_metadata_undo(viewer, _data):
    return bool(viewer._do_undo())


def handle_metadata_redo(viewer, _data):
    return bool(viewer._do_redo())


def handle_edit_cell(viewer, data):
    row = data.get("row")
    col = data.get("column")
    value = data.get("value")

    if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
        return False
    row = int(row)
    if row < 0 or row >= getattr(viewer, "n_nodes", 0):
        return False

    meta_entry = viewer.metadata.get(col)
    if not meta_entry:
        return False
    try:
        parsed_val = _parse_metadata_value(meta_entry["type"], value)
    except (TypeError, ValueError):
        return False

    old_value = meta_entry["values"][row]
    if _metadata_values_equal(old_value, parsed_val):
        return False
    _record_metadata_cell_history(viewer, col, row, old_value, parsed_val)
    meta_entry["values"][row] = parsed_val
    viewer.update_nodes()
    viewer.canvas.update()
    return True

def handle_import_metadata(viewer, data):
    try:
        parent_widget = getattr(viewer, 'main_window', None)
        if not parent_widget and hasattr(viewer, 'canvas'):
            parent_widget = viewer.canvas.native
        if not parent_widget:
            parent_widget = QtWidgets.QApplication.activeWindow()
            
        if parent_widget:
            parent_widget.raise_()
            parent_widget.activateWindow()
            
        meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Input_Files", "Meta_Data"))
        abs_meta_dir = os.path.abspath(meta_dir)
        os.makedirs(abs_meta_dir, exist_ok=True)
        
        dialog = QtWidgets.QFileDialog(parent_widget)
        dialog.setWindowTitle(translate("Viewer", "Import Metadata Spreadsheet"))
        dialog.setDirectory(abs_meta_dir)
        dialog.setNameFilter(translate("Viewer", "Excel/CSV Files (*.xlsx *.xls *.csv)"))
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        
        # Bring to front of browser window
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        dialog.raise_()
        dialog.activateWindow()
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                filepath = selected[0]
                upload_metadata(viewer, [filepath])
    except Exception as e:
        print(f"Error picking file for metadata import: {e}")
        Command_Engine.command_failed(viewer, f'Error picking file for metadata import: {e}')

def handle_export_metadata(viewer, data):
    try:
        parent_widget = getattr(viewer, 'main_window', None)
        if not parent_widget and hasattr(viewer, 'canvas'):
            parent_widget = viewer.canvas.native
        if not parent_widget:
            parent_widget = QtWidgets.QApplication.activeWindow()
            
        if parent_widget:
            parent_widget.raise_()
            parent_widget.activateWindow()
            
        meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Input_Files", "Meta_Data"))
        abs_meta_dir = os.path.abspath(meta_dir)
        os.makedirs(abs_meta_dir, exist_ok=True)
        
        default_path = os.path.join(abs_meta_dir, "metadata_export.csv")
        
        dialog = QtWidgets.QFileDialog(parent_widget)
        dialog.setWindowTitle(translate("Viewer", "Export Metadata Spreadsheet"))
        dialog.setDirectory(abs_meta_dir)
        dialog.selectFile(default_path)
        dialog.setNameFilters([
            translate("Viewer", "CSV Files (*.csv)"),
            translate("Viewer", "Excel Files (*.xlsx)"),
        ])
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        dialog.raise_()
        dialog.activateWindow()
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                filepath = selected[0]
                # A filter's pattern stays as written in every language; its name may not.
                selected_filter = dialog.selectedNameFilter()
                if "*.xlsx" in selected_filter and not filepath.lower().endswith(('.xlsx', '.xls')):
                    filepath += ".xlsx"
                elif "*.csv" in selected_filter and not filepath.lower().endswith('.csv'):
                    filepath += ".csv"
                download_metadata(viewer, filepath)
    except Exception as e:
        print(f"Error picking file for metadata export: {e}")
        Command_Engine.command_failed(viewer, f'Error picking file for metadata export: {e}')

def _extend_initial_web_state(viewer, state):
    state["columns"] = ["Node ID"] + list(viewer.metadata.keys())
    state["types"] = {
        key: entry["type"] for key, entry in viewer.metadata.items()
    }
    return state


def handle_select_node(viewer, data):
    """Apply SSN left-click focus for a validated metadata row index."""
    node_idx = data.get("index")
    if isinstance(node_idx, (bool, np.bool_)) or not isinstance(
        node_idx, (int, np.integer)
    ):
        return False

    node_idx = int(node_idx)
    node_count = int(getattr(viewer, "n_nodes", 0))
    if node_idx < 0 or node_idx >= node_count:
        return False

    visible_mask = getattr(viewer, "visible_mask", None)
    if visible_mask is None or not bool(visible_mask[node_idx]):
        return False

    apply_focus = getattr(viewer, "apply_left_click_focus", None)
    if not callable(apply_focus):
        return False
    apply_focus(node_idx)
    return True


def handle_clear_selection(viewer, _data):
    """Clear only temporary click focus, leaving command selection intact."""
    clear_focus = getattr(viewer, "clear_left_click_focus", None)
    if not callable(clear_focus):
        return False
    clear_focus()
    return True


def register_backend(registry, viewer):
    """Register Metadata web capabilities without changing sidebar state."""
    registry.register_action(
        "meta", "select", lambda data: handle_select_node(viewer, data)
    )
    registry.register_action(
        "meta", "clear_selection", lambda data: handle_clear_selection(viewer, data)
    )
    registry.register_action(
        "meta", "edit_cell", lambda data: handle_edit_cell(viewer, data)
    )
    registry.register_action(
        "meta", "import_metadata", lambda data: handle_import_metadata(viewer, data)
    )
    registry.register_action(
        "meta", "export_metadata", lambda data: handle_export_metadata(viewer, data)
    )
    registry.register_action(
        "meta", "delete_columns", lambda data: handle_delete_columns(viewer, data)
    )
    registry.register_action(
        "meta", "metadata_undo", lambda data: handle_metadata_undo(viewer, data)
    )
    registry.register_action(
        "meta", "metadata_redo", lambda data: handle_metadata_redo(viewer, data)
    )
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry.register_static_route(
        "meta", "/meta_resource/", os.path.join(src_dir, "resources", "meta")
    )
    registry.register_state_provider(
        "meta", "metadata_columns", _extend_initial_web_state
    )


def activate(viewer):
    """Show and persist the Metadata UI while preserving current behavior."""
    # Preserve the existing rule that any SSN mouse press clears metadata
    # multi-highlights. Shared viewer helpers now own row broadcasts.
    if hasattr(viewer, 'on_mouse_press') and not hasattr(viewer, "_on_mouse_press_patched_meta"):
        orig_on_mouse_press = viewer.on_mouse_press
        def wrapped_on_mouse_press(event):
            if hasattr(viewer, 'left_click_highlight_indices'):
                viewer.left_click_highlight_indices = None
            orig_on_mouse_press(event)
        viewer.on_mouse_press = wrapped_on_mouse_press
        viewer._on_mouse_press_patched_meta = True

    if hasattr(viewer, 'add_sidebar_button'):
        viewer.add_sidebar_button(
            name="metaDataBtn",
            label=translate("Viewer", "📊 Meta Data"),
            callback=viewer.open_metadata_ui,
            tooltip=translate("Viewer", "Open Metadata Spreadsheet in browser")
        )
        if not hasattr(viewer, 'sidebar_buttons_to_persist'):
            viewer.sidebar_buttons_to_persist = []
        if "meta" not in viewer.sidebar_buttons_to_persist:
            viewer.sidebar_buttons_to_persist.append("meta")


def register(viewer):
    """Compatibility wrapper: ensure backend registration, then activate its UI."""
    registry = ensure_registry(viewer)
    register_backend(registry, viewer)
    registry.registered_plugins.add("meta")
    return activate(viewer)


