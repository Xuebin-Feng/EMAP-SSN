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
import numpy as np
import pandas as pd
from PySide6 import QtWidgets, QtCore, QtGui
import Command_Engine
import SSN_Utils as utils
import SSN_Config as cfg
from web_ui.Plugin_Manager import ensure_registry

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
                    if isinstance(val, (float, np.floating)):
                        if pd.isna(val):
                            return ""
                        return f"{val:g}"
                    return str(val) if pd.notna(val) else ""
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
            
        prop_type = meta_entry["type"]
        if prop_type == "number":
            try:
                if not str(value).strip():
                    parsed_val = np.nan
                else:
                    parsed_val = float(value)
            except ValueError:
                return False
        else:
            parsed_val = str(value)
            
        meta_entry["values"][row] = parsed_val
        self.dataChanged.emit(index, index, [QtCore.Qt.ItemDataRole.DisplayRole, QtCore.Qt.ItemDataRole.EditRole])
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
            """ % {"font": utils.UI_FONT_STACK})

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
    
    meta_entry = viewer.metadata.get(col)
    if meta_entry:
        prop_type = meta_entry["type"]
        if prop_type == "number":
            try:
                if str(value).strip() == "":
                    parsed_val = np.nan
                else:
                    parsed_val = float(value)
            except ValueError:
                return
        else:
            parsed_val = str(value)
        
        meta_entry["values"][row] = parsed_val
        viewer.update_nodes()
        viewer.canvas.update()

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
            
        meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Cache_Files", "Meta_Data"))
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
            
        meta_dir = getattr(cfg, 'METADATA_DIR', os.path.join("Cache_Files", "Meta_Data"))
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

def _extend_initial_web_state(viewer, state):
    state["columns"] = ["Node ID"] + list(viewer.metadata.keys())
    state["types"] = {
        key: entry["type"] for key, entry in viewer.metadata.items()
    }
    return state


def register_backend(registry, viewer):
    """Register Metadata web capabilities without changing sidebar state."""
    registry.register_action("meta", "select", lambda data: None)
    registry.register_action(
        "meta", "edit_cell", lambda data: handle_edit_cell(viewer, data)
    )
    registry.register_action(
        "meta", "import_metadata", lambda data: handle_import_metadata(viewer, data)
    )
    registry.register_action(
        "meta", "export_metadata", lambda data: handle_export_metadata(viewer, data)
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
    # Intercept mouse press to broadcast left-click node highlights & clear multi-highlights
    if hasattr(viewer, 'on_mouse_press') and not hasattr(viewer, "_on_mouse_press_patched_meta"):
        orig_on_mouse_press = viewer.on_mouse_press
        def wrapped_on_mouse_press(event):
            if hasattr(viewer, 'left_click_highlight_indices'):
                viewer.left_click_highlight_indices = None
            orig_on_mouse_press(event)
            if event.button == 1 and 'Shift' not in event.modifiers and not viewer.console_mode:
                idx = getattr(viewer, 'selected_node_idx', None)
                if idx is not None:
                    viewer.broadcast_event({"type": "highlight_row", "index": int(idx)})
                else:
                    viewer.broadcast_event({"type": "highlight_row", "index": None})
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

    msg_parts = []
    if successful_files:
        msg_parts.append(
            f"Successfully uploaded metadata from {len(successful_files)} file(s): {', '.join(successful_files)}. "
            f"Matched {len(matched_nodes)} unique nodes, ignored {total_unmatched} rows. "
            f"Merged properties: {', '.join(sorted(all_merged_props))}."
        )
        viewer.broadcast_event({
            "type": "state_updated",
            "visible_mask": viewer.visible_mask.tolist(),
            "selected_indices": viewer.selected_indices,
            "metadata": viewer.get_serializable_metadata(),
            "columns": ["Node ID"] + list(viewer.metadata.keys()),
            "types": {k: entry["type"] for k, entry in viewer.metadata.items()}
        })
    if failed_files:
        fail_details = "; ".join([f"{f}: {err}" for f, err in failed_files])
        msg_parts.append(f"Failed to upload from {len(failed_files)} file(s): {fail_details}")

    msg = " ".join(msg_parts)
    Command_Engine.print_help(viewer, msg)


def download_metadata(viewer, filepath, expr=None):
    """Downloads network metadata to a file, applying optional logic filters."""
    if not getattr(viewer, 'metadata', None):
        Command_Engine.print_help(viewer, "Error: No metadata available in the viewer to download.")
        return False

    try:
        mask = np.ones(viewer.n_nodes, dtype=bool)
        if expr:
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

            expr_cleaned = re.sub(r'["\']?\$sele\$["\']?', '@_sele.txt@', expr, flags=re.IGNORECASE)
            expr_cleaned = re.sub(r'\{([^}]+)\}', lambda m: '{' + m.group(1).replace(' ', '') + '}', expr_cleaned)
            
            viewer_to_aln, valid_indices = Command_Engine.get_alignment_mapping(viewer)
            
            try:
                mask = Command_Engine.parse_advanced_expression(
                    expr_cleaned,
                    viewer_to_aln,
                    valid_indices,
                    viewer.full_headers,
                    getattr(viewer, 'cluster_labels', None),
                    getattr(viewer, 'group_labels', None),
                    getattr(viewer, 'alignment', None),
                    metadata=viewer.metadata
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
                        row_val.append(val)
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
        return True
    except Exception as e:
        Command_Engine.print_help(viewer, f"Error downloading metadata: {e}")
        return False
