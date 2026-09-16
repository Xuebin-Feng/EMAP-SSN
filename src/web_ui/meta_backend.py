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
import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtCore, QtGui
import Command_Engine
import EMAPSSN_Config as cfg
from desktop.Desktop_App import UI_QSS_FONT_STACK
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
    refresh_metadata_views,
    metadata_state_event,
    broadcast_metadata_state,
    _resolve_metadata_column_names,
    delete_metadata_columns,
    upload_metadata,
    download_metadata,
)

class MetadataTableModel(QtCore.QAbstractTableModel):
    """High-performance table model backed directly by viewer.metadata arrays."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.columns = []
        self.refresh_columns()

    def refresh_columns(self):
        self.beginResetModel()
        self.columns = ["Node ID"]
        if hasattr(self.viewer, 'metadata'):
            self.columns.extend(list(self.viewer.metadata.keys()))
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return getattr(self.viewer, 'n_nodes', 0)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if pd.isna(val):
                        return ""
                    return format_metadata_value(val)
        elif role == QtCore.Qt.ItemDataRole.EditRole:
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if isinstance(val, (float, np.floating)):
                        if pd.isna(val):
                            return ""
                        return str(val)
                    return str(val) if pd.notna(val) else ""
        elif role == QtCore.Qt.ItemDataRole.UserRole:
            if col_name == "Node ID":
                return str(self.viewer.full_headers[row])
            else:
                meta_entry = self.viewer.metadata.get(col_name)
                if meta_entry is not None:
                    val = meta_entry["values"][row]
                    if isinstance(val, (float, np.floating)) and pd.isna(val):
                        return None
                    return val
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags
        base_flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        col_name = self.columns[index.column()]
        if col_name != "Node ID":
            return base_flags | QtCore.Qt.ItemFlag.ItemIsEditable
        return base_flags

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.EditRole:
            return False
        row = index.row()
        col = index.column()
        col_name = self.columns[col]
        
        if col_name == "Node ID":
            return False
            
        meta_entry = self.viewer.metadata.get(col_name)
        if meta_entry is None:
            return False
            
        try:
            parsed_val = _parse_metadata_value(meta_entry["type"], value)
        except (TypeError, ValueError):
            return False

        old_value = meta_entry["values"][row]
        if _metadata_values_equal(old_value, parsed_val):
            return False
        _record_metadata_cell_history(
            self.viewer, col_name, row, old_value, parsed_val
        )
        meta_entry["values"][row] = parsed_val
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
        if hasattr(self.viewer, "update_nodes"):
            self.viewer.update_nodes()
        if hasattr(self.viewer, "canvas"):
            self.viewer.canvas.update()
        broadcast_metadata_state(self.viewer)
        return True

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Horizontal:
                return self.columns[section]
            else:
                return str(section + 1)
        return None


class MultiColumnFilterProxyModel(QtCore.QSortFilterProxyModel):
    """Proxy model that filters by visibility mask AND per-column text filters."""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.column_filters = {}  # col_index -> filter_text

    def set_column_filter(self, col, text):
        self.column_filters[col] = text.lower().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if hasattr(self.viewer, 'visible_mask'):
            if not bool(self.viewer.visible_mask[source_row]):
                return False
        for col, text in self.column_filters.items():
            if not text:
                continue
            idx = self.sourceModel().index(source_row, col)
            val = str(self.sourceModel().data(idx, QtCore.Qt.ItemDataRole.DisplayRole) or "")
            if text not in val.lower():
                return False
        return True

    def lessThan(self, left, right):
        left_val = self.sourceModel().data(left, QtCore.Qt.ItemDataRole.UserRole)
        right_val = self.sourceModel().data(right, QtCore.Qt.ItemDataRole.UserRole)
        if left_val is None and right_val is None:
            return False
        if left_val is None:
            return True
        if right_val is None:
            return False
        if isinstance(left_val, (int, float, np.integer, np.floating)) and isinstance(right_val, (int, float, np.integer, np.floating)):
            return float(left_val) < float(right_val)
        return str(left_val).lower() < str(right_val).lower()

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Orientation.Vertical:
                return str(section + 1)
        return super().headerData(section, orientation, role)


class FilterHeaderView(QtWidgets.QHeaderView):
    """Custom header with filter QLineEdit widgets embedded below each column header."""
    filterChanged = QtCore.Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self._editors = []
        self._padding = 4
        self.setSectionsClickable(True)
        self.setSortIndicatorShown(True)
        self.sectionResized.connect(self._adjust_positions)
        self.sectionMoved.connect(self._adjust_positions)

    def setFilterBoxes(self, count):
        for ed in self._editors:
            ed.deleteLater()
        self._editors = []
        for i in range(count):
            editor = QtWidgets.QLineEdit(self)
            editor.setPlaceholderText("Filter...")
            editor.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #d0d7de;
                    border-radius: 3px;
                    padding: 1px 4px;
                    font-size: 8.5pt;
                    background-color: #ffffff;
                }
                QLineEdit:focus {
                    border-color: #0969da;
                }
            """)
            editor.textChanged.connect(lambda text, col=i: self.filterChanged.emit(col, text))
            self._editors.append(editor)
        self._adjust_positions()

    def _adjust_positions(self):
        for i, editor in enumerate(self._editors):
            h = self.sectionSize(i)
            px = self.sectionPosition(i) - self.offset()
            filter_h = 22
            editor.setGeometry(px + self._padding, 2,
                               h - 2 * self._padding, filter_h)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.fillRect(rect, QtGui.QColor("#f0f0f0"))
        
        painter.setPen(QtGui.QColor("#e2e2e2"))
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.right(), rect.bottom())
        
        filter_h = 22
        offset_y = filter_h + 4
        text_rect = QtCore.QRect(rect.x() + 6, rect.y() + offset_y, rect.width() - 12, rect.height() - offset_y)
        
        text = str(self.model().headerData(logicalIndex, QtCore.Qt.Orientation.Horizontal, QtCore.Qt.ItemDataRole.DisplayRole) or "")
        
        if self.isSortIndicatorShown() and self.sortIndicatorSection() == logicalIndex:
            order = self.sortIndicatorOrder()
            text += "  \u25B2" if order == QtCore.Qt.SortOrder.AscendingOrder else "  \u25BC"
        
        painter.setPen(QtGui.QColor("#1f2328"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        
        painter.restore()

    def sizeHint(self):
        s = super().sizeHint()
        s.setHeight(s.height() + 26)
        return s

    def updateGeometries(self):
        super().updateGeometries()
        self._adjust_positions()

    def showEvent(self, event):
        super().showEvent(event)
        self._adjust_positions()


def is_logic_expression(arg):
    if any(c in arg for c in '{}#@&|!^"'):
        return True
    if arg.lower() == '$sele$':
        return True
    if re.match(r'^[a-zA-Z_][\d\.]+$', arg):
        return True
    return False


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


def inject_spreadsheet_panel(viewer, show_sidebar=True):
    if not getattr(viewer, 'metadata', None):
        return
    real_keys = list(viewer.metadata.keys())
    if not real_keys:
        return

    if hasattr(viewer, 'tab_widget'):
        tab_idx = -1
        for idx in range(viewer.tab_widget.count()):
            if viewer.tab_widget.tabText(idx) == "Metadata":
                tab_idx = idx
                break

        if tab_idx == -1:
            table_view = QtWidgets.QTableView()
            source_model = MetadataTableModel(viewer)
            proxy_model = MultiColumnFilterProxyModel(viewer)
            proxy_model.setSourceModel(source_model)
            table_view.setModel(proxy_model)

            table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            table_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
            table_view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
            table_view.setSortingEnabled(False)
            table_view.setAlternatingRowColors(True)

            table_view.setStyleSheet("""
                QTableView {
                    gridline-color: #e2e2e2;
                    background-color: #ffffff;
                    alternate-background-color: #f8f9fa;
                    font-family: %(font)s;
                    font-size: 9.5pt;
                    border: none;
                }
                QTableView::item:selected {
                    background-color: #e6f7ff;
                    color: #1f2328;
                }
                QHeaderView::section {
                    background-color: #f0f0f0;
                    padding: 4px;
                    border: none;
                    border-right: 1px solid #e2e2e2;
                    border-bottom: 2px solid #e2e2e2;
                    font-weight: bold;
                    font-size: 9pt;
                }
            """ % {"font": UI_QSS_FONT_STACK})

            filter_header = FilterHeaderView(table_view)
            table_view.setHorizontalHeader(filter_header)
            filter_header.setFilterBoxes(source_model.columnCount())
            filter_header.filterChanged.connect(proxy_model.set_column_filter)

            table_view._current_sort_col = -1
            table_view._current_sort_order = QtCore.Qt.SortOrder.AscendingOrder

            def handle_header_click(logical_index):
                header = table_view.horizontalHeader()
                if table_view._current_sort_col == logical_index:
                    if table_view._current_sort_order == QtCore.Qt.SortOrder.AscendingOrder:
                        table_view._current_sort_order = QtCore.Qt.SortOrder.DescendingOrder
                        proxy_model.sort(logical_index, QtCore.Qt.SortOrder.DescendingOrder)
                        header.setSortIndicator(logical_index, QtCore.Qt.SortOrder.DescendingOrder)
                        header.setSortIndicatorShown(True)
                    else:
                        table_view._current_sort_col = -1
                        proxy_model.sort(-1, QtCore.Qt.SortOrder.AscendingOrder)
                        header.setSortIndicatorShown(False)
                else:
                    table_view._current_sort_col = logical_index
                    table_view._current_sort_order = QtCore.Qt.SortOrder.AscendingOrder
                    proxy_model.sort(logical_index, QtCore.Qt.SortOrder.AscendingOrder)
                    header.setSortIndicator(logical_index, QtCore.Qt.SortOrder.AscendingOrder)
                    header.setSortIndicatorShown(True)

            filter_header.sectionClicked.connect(handle_header_click)

            def on_table_double_clicked(index):
                col_name = source_model.columns[index.column()]
                if col_name == "Node ID":
                    source_index = proxy_model.mapToSource(index)
                    row = source_index.row()
                    if hasattr(viewer, 'pos') and row < len(viewer.pos):
                        viewer.view.camera.center = tuple(viewer.pos[row][:2])
                        viewer.selected_indices = [row]
                        viewer.selected_node_idx = row
                        viewer.update_selection_visual()
                        if hasattr(viewer, '_hud_timer'):
                            viewer._hud_timer.start()
            table_view.doubleClicked.connect(on_table_double_clicked)

            def on_table_selection_changed():
                if getattr(viewer, '_syncing_selection', False):
                    return
                viewer._syncing_selection = True
                try:
                    selected_rows = table_view.selectionModel().selectedRows()
                    new_selected = []
                    for index in selected_rows:
                        source_index = proxy_model.mapToSource(index)
                        new_selected.append(source_index.row())
                    viewer.selected_indices = new_selected
                    viewer.update_selection_visual()
                finally:
                    viewer._syncing_selection = False
            table_view.selectionModel().selectionChanged.connect(on_table_selection_changed)

            viewer.metadata_table_view = table_view
            viewer.metadata_source_model = source_model
            viewer.metadata_proxy_model = proxy_model

            def sync_selection_to_table(node_idx=None):
                if getattr(viewer, '_syncing_selection', False):
                    return
                viewer._syncing_selection = True
                try:
                    selection_model = table_view.selectionModel()
                    selection_model.clearSelection()
                    if node_idx is not None:
                        source_idx = source_model.index(node_idx, 0)
                        proxy_idx = proxy_model.mapFromSource(source_idx)
                        if proxy_idx.isValid():
                            row_start = proxy_model.index(proxy_idx.row(), 0)
                            row_end = proxy_model.index(proxy_idx.row(), source_model.columnCount() - 1)
                            selection_model.select(
                                QtCore.QItemSelection(row_start, row_end),
                                QtCore.QItemSelectionModel.SelectionFlag.Select
                            )
                            table_view.scrollTo(proxy_idx, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
                finally:
                    viewer._syncing_selection = False
            viewer.sync_metadata_table_selection = sync_selection_to_table

            def sync_visibility_to_table():
                proxy_model.invalidateFilter()
            viewer.sync_metadata_table_visibility = sync_visibility_to_table

            viewer.tab_widget.addTab(table_view, "Metadata")
            tab_idx = viewer.tab_widget.count() - 1

            sel_idx = getattr(viewer, 'selected_node_idx', None)
            if sel_idx is not None:
                sync_selection_to_table(sel_idx)
        else:
            if hasattr(viewer, 'metadata_source_model'):
                viewer.metadata_source_model.refresh_columns()
                if hasattr(viewer, 'metadata_table_view'):
                    hdr = viewer.metadata_table_view.horizontalHeader()
                    if isinstance(hdr, FilterHeaderView):
                        hdr.setFilterBoxes(viewer.metadata_source_model.columnCount())

        viewer.tab_widget.setCurrentIndex(tab_idx)
        if show_sidebar:
            viewer.set_sidebar_visible(True)
        else:
            viewer.set_sidebar_visible(False)


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
        dialog.setWindowTitle("Import Metadata Spreadsheet")
        dialog.setDirectory(abs_meta_dir)
        dialog.setNameFilter("Excel/CSV Files (*.xlsx *.xls *.csv)")
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
        dialog.setWindowTitle("Export Metadata Spreadsheet")
        dialog.setDirectory(abs_meta_dir)
        dialog.selectFile(default_path)
        dialog.setNameFilter("CSV Files (*.csv);;Excel Files (*.xlsx)")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        dialog.raise_()
        dialog.activateWindow()
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                filepath = selected[0]
                selected_filter = dialog.selectedNameFilter()
                if "Excel" in selected_filter and not filepath.lower().endswith(('.xlsx', '.xls')):
                    filepath += ".xlsx"
                elif "CSV" in selected_filter and not filepath.lower().endswith('.csv'):
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
            label="📊 Meta Data",
            callback=viewer.open_metadata_ui,
            tooltip="Open Metadata Spreadsheet in browser"
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


