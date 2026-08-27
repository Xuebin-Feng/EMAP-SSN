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
import sys
import json


def _configure_linux_vispy_platform(
    environment=None,
    platform_name=sys.platform,
):
    """Route VisPy through XWayland before Qt/OpenGL is imported.

    On affected Linux systems Qt's native Wayland plugin creates an OpenGL ES
    context while VisPy 0.16 compiles desktop GLSL 1.20 shaders.  XCB creates
    the compatible desktop OpenGL context.  Preserve non-window platforms
    such as ``offscreen`` for tests and headless use.
    """
    environment = os.environ if environment is None else environment
    if not str(platform_name).startswith('linux'):
        return False

    selected_platform = environment.get("QT_QPA_PLATFORM", "").lower()
    wayland_session = environment.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    native_wayland_selected = selected_platform.startswith("wayland")
    if native_wayland_selected or (wayland_session and not selected_platform):
        environment["QT_QPA_PLATFORM"] = "xcb"
        return True
    return False


# This must run before torch, VisPy, or PySide6 can initialize Qt/OpenGL.
_configure_linux_vispy_platform()

import unicodedata  # Pre-load to prevent Windows DLL search path conflicts with Qt/OpenGL
try:
    import torch  # Pre-load to prevent DLL initialization conflicts between PyTorch and PySide6/OpenGL
except Exception:
    pass
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# Ensure src/ (the directory containing all project modules) is on sys.path.
# This is needed when the script is launched as a subprocess (e.g. from SSN_Config.py).
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import h5py
import numpy as np
import importlib
from collections import deque
import math
import queue
from vispy import scene, app
from PySide6 import QtWidgets, QtCore, QtGui

import SSN_Config as cfg
import Command_Engine
import Cache_Manifest as cache_manifest
from Layout_Cache_Generator import (
    LayoutGenerationSettings,
    generate_layout_cache,
)
from Background_Job_Scheduler import BackgroundJobScheduler
from utilities.FASTA_Sanitization import (
    load_sanitized_fasta,
)
from utilities.Application_Fonts import (
    UI_QSS_FONT_STACK,
    VISPY_FALLBACK_FACE,
    configure_qt_application_fonts,
    force_light_palette,
    register_vispy_application_fonts,
    vispy_points_at_reference_dpi,
    vispy_points_for_logical_pixels,
)
from utilities.Application_Windows import show_window_in_front
from utilities.Cache_Selection import resolve_selected_cache
from utilities.Network_Preparation import prepare_network
from utilities.Application_Identity import (
    VIEWER_DESKTOP_FILE_NAME,
    configure_linux_qt_desktop_identity,
)
from web_ui.Browser_Page import open_browser_page


def _remove_consumed_settings_snapshot():
    """Remove a per-launch settings snapshot after SSN_Config imported it."""
    snapshot_path = os.environ.pop("SSN_VIEWER_SETTINGS_PATH", None)
    if not snapshot_path:
        return
    try:
        os.unlink(snapshot_path)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(f"Warning: Could not remove settings snapshot {snapshot_path}: {error}")


_remove_consumed_settings_snapshot()


def _load_selected_fasta_records(fasta_path):
    """Load the same canonical FASTA records used by embedding generation."""
    headers, sequences, _ = load_sanitized_fasta(fasta_path)
    return list(zip(headers, sequences))


def _build_sequence_lookup(records):
    """Index canonical FASTA records by both full header and first token."""
    sequence_lookup = {}
    for header, sequence in records:
        sequence_lookup[header] = sequence
        header_parts = header.split()
        if header_parts:
            sequence_lookup[header_parts[0]] = sequence
    return sequence_lookup


def _wrap_console_text_for_display(text, max_width, measure_width):
    """Wrap one logical console line without removing or changing its text."""
    if not text or max_width <= 0 or measure_width(text) <= max_width:
        return text

    lines = []
    remaining = text
    while remaining:
        if measure_width(remaining) <= max_width:
            lines.append(remaining)
            break

        # Find the longest prefix that fits. Glyph advances are monotonic, so a
        # binary search avoids repeatedly measuring every possible substring.
        low, high = 1, len(remaining)
        while low < high:
            midpoint = (low + high + 1) // 2
            if measure_width(remaining[:midpoint]) <= max_width:
                low = midpoint
            else:
                high = midpoint - 1

        split_at = max(1, low)

        # Prefer a natural whitespace boundary, while retaining that whitespace
        # so removing the visual newlines reconstructs the exact logical text.
        whitespace_break = -1
        for index in range(split_at - 1, 0, -1):
            if remaining[index].isspace():
                whitespace_break = index + 1
                break
        if whitespace_break > 0:
            split_at = whitespace_break

        lines.append(remaining[:split_at])
        remaining = remaining[split_at:]

    return "\n".join(lines)


def _vispy_text_line_height_pixels(text_visual):
    """Return VisPy's rendered multiline advance in logical pixels."""
    n_pix = (text_visual.font_size / 72.0) * text_visual.transforms.dpi
    line_height = getattr(text_visual, '_line_height', 1.2)
    font = getattr(text_visual, '_font', None)
    if font is None:
        return n_pix * line_height

    try:
        ratio = 1.0 / font.ratio
        slop = font.slop
        ascender = 0.0
        descender = 0.0
        for char in 'ÅÉÑŐjgpqy':
            glyph = font[char]
            y0 = glyph['offset'][1] * ratio + slop
            y1 = y0 - glyph['size'][1]
            ascender = max(ascender, y0 - slop)
            descender = min(descender, y1 + slop)
        glyph_height = ascender - descender
        lowres_size = float(font._lowres_size)
        if glyph_height > 0.0 and lowres_size > 0.0:
            return (glyph_height / lowres_size) * n_pix * line_height
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        pass

    return n_pix * line_height


def _apply_safe_rectangle_geometry(rectangle, center, width, height, radius):
    """Apply rounded-rectangle geometry without exposing invalid interim state."""
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    radius = min(
        max(float(radius), 0.0),
        width / 2.0,
        height / 2.0,
    )

    # VisPy regenerates and validates vertices after every property assignment.
    # Resetting the radius first also repairs rectangles left in an invalid
    # partially-mutated state by an earlier failed width/height assignment.
    rectangle.radius = 0.0
    rectangle.width = width
    rectangle.height = height
    rectangle.center = (float(center[0]), float(center[1]))
    rectangle.radius = radius

    return width, height, radius


def _contiguous_line_positions(positions):
    """Normalize VisPy line vertices before uploading them to the GPU."""
    result = np.ascontiguousarray(positions, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] not in (2, 3):
        raise ValueError(
            "Line positions must have shape (N, 2) or (N, 3); "
            f"received {result.shape}."
        )
    if not np.isfinite(result).all():
        raise ValueError("Line positions contain NaN or infinite values.")
    return result


def _topmost_nearest_visible_node_index(
    positions,
    visible_mask,
    point,
    visible_draw_order=None,
):
    """Return the nearest visible node, preferring the later-drawn node on ties."""
    if visible_draw_order is None:
        visible_indices = np.flatnonzero(visible_mask)
    else:
        candidate_order = np.asarray(visible_draw_order, dtype=np.int64)
        valid = (candidate_order >= 0) & (candidate_order < len(positions))
        visible_indices = candidate_order[valid]
        visible_indices = visible_indices[visible_mask[visible_indices]]
    if visible_indices.size == 0:
        return None

    # With depth testing disabled, later entries are drawn over earlier entries,
    # so search the submitted order in reverse to make exact distance ties match
    # the visual stacking order.
    topmost_first = visible_indices[::-1]
    distances = np.linalg.norm(
        positions[topmost_first, :2] - np.asarray(point)[:2],
        axis=1,
    )
    return int(topmost_first[np.argmin(distances)])

# =========================================================================
# MANUAL CUSTOM ATTRIBUTES INITIALIZATION SECTION
# Users can add custom attributes to be initialized on the viewer at startup
# and registered for layout cache saving/loading.
# Format: "attribute_name": default_value (e.g. "my_scores": None)
# =========================================================================
CUSTOM_ATTRIBUTES_INIT = {
    # Add your custom attributes here:
    "sidebar_buttons_to_persist": []
}

# Fix High-DPI scaling
class HUDDisplay:
    def __init__(self, viewer, name, pos_fn, anchor_x='right', anchor_y='bottom'):
        self.viewer = viewer
        self.name = name
        self.pos_fn = pos_fn  # lambda size: (x, y)
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y
        self.text_visual = None
        self.visible = False

    def show(self, text):
        w, h = self.viewer.canvas.size
        panel_visible = hasattr(self.viewer, 'right_panel') and self.viewer.right_panel.isVisible()
        panel_w = getattr(self.viewer, '_panel_w', 120) if panel_visible else 0
        pos = self.pos_fn((w - panel_w, h))

        if self.text_visual is None:
            self.text_visual = scene.visuals.Text(
                text=text,
                bold=True,
                face=self.viewer.vispy_ui_face,
                font_size=self.viewer._hud_font_size_points(),
                color=cfg.TEXT_COLOR,
                pos=pos,
                anchor_x=self.anchor_x,
                anchor_y=self.anchor_y,
                parent=self.viewer.canvas.scene
            )
        else:
            self.text_visual.text = text
            self.text_visual.pos = pos
            self.text_visual.visible = True
        self.visible = True

    def hide(self):
        if self.text_visual is not None:
            self.text_visual.visible = False
            self.text_visual.text = ""
        self.visible = False

    def update_position(self):
        if self.text_visual is not None and self.visible:
            w, h = self.viewer.canvas.size
            panel_visible = hasattr(self.viewer, 'right_panel') and self.viewer.right_panel.isVisible()
            panel_w = getattr(self.viewer, '_panel_w', 120) if panel_visible else 0
            self.text_visual.pos = self.pos_fn((w - panel_w, h))

    def on_node_clicked(self, node_idx):
        """Override in subclasses to handle left-click updates."""
        pass

    def on_right_click(self):
        """Handle right-click event. Hides by default, override if custom behavior is needed."""
        self.hide()


class MainViewer:
    def __init__(self):
        # --- 1. Viewer State ---
        self.console_mode = False
        from web_ui.Plugin_Manager import WebPluginRegistry
        self.web_plugin_registry = WebPluginRegistry(self)
        self.web_action_handlers = self.web_plugin_registry.actions
        self.vispy_ui_face = VISPY_FALLBACK_FACE
        self.vispy_monospace_face = VISPY_FALLBACK_FACE
        self._display_window_handle = None
        self._active_screen = None
        self._active_screen_signal_bindings = []
        self._display_refresh_pending = False
        self._last_display_signature = None
        
        # =========================================================================
        # HUD & CONSOLE LAYOUT CONFIGURATION SECTION
        # Users can adjust the positions, sizes, and padding of the text and 
        # background elements below. Adjust these coordinates if elements do not
        # align correctly on your screen or High-DPI display.
        # Note: All coordinates are defined in logical pixels and are automatically
        # scaled by the canvas pixel scale (DPI factor) at runtime.
        # =========================================================================
        self.hud_layout = {
            # 1. Top-left Instructions (" [ENTER] Command | [LeftClick] Highlight | ... ")
            "instr_x": 10.0,             # Horizontal coordinate from left edge
            "instr_y": 10.0,             # Vertical coordinate from top edge (baseline)
            "instr_anchor_x": "left",    # Horizontal text alignment: 'left', 'center', 'right'
            "instr_anchor_y": "bottom",  # Vertical text alignment: 'top', 'middle', 'bottom'
            
            # 2. Command Line Text (" Cmd: <input> ")
            "console_text_x": 30.0,      # Horizontal coordinate from left edge
            "console_text_y": 60.0,      # Vertical coordinate from top edge (baseline)
            "console_text_anchor_x": "left",
            "console_text_anchor_y": "bottom",
            "font_size_px": 16.0,        # Logical-pixel size shared by all HUD text
            
            # 3. Command Line Background Box
            "console_bg_height": 40.0,   # Vertical height of the background box
            "console_bg_min_width": 250.0, # Minimum width of the background box when empty/short
            "console_bg_left_offset": 20.0, # Fixed horizontal left position of the box
            "console_bg_y_offset": 20.0, # Positive values move only the box downward
            "console_bg_radius": 10.0,    # Radius for the rounded corners (0.0 for sharp corners)
            "console_bg_padding_x": 20.0, # Fixed logical padding added to the end of the command box
            # 4. Background-job status text below the command box
            "background_job_status_gap": 8.0,
            "background_job_status_right_padding": 20.0,
            
            # 5. Bottom-right status stack, from bottom to top:
            #    Hidden Nodes, View Width, selected metadata property.
            "status_x_offset": 10.0,       # Distance from the right edge of the window
            "status_bottom_offset": 30.0,  # Distance from the bottom edge to Hidden Nodes
            "status_line_spacing": 25.0,
            "status_anchor_x": "right",
            "status_anchor_y": "bottom"
        }
        
        self.input_buffer = ""
        self.cursor_pos = 0       # Tracks cursor position
        
        # ---> Persistent Command History (Per Layout) <---
        self.command_history = []
        try:
            cache_path, _ = resolve_selected_cache(cfg)
            cache_dir = os.path.dirname(cache_path)
            self.history_file = os.path.join(cache_dir, "cli_history.txt")
            
            # Migration check: if old history exists in Cache_Files/History/folder_name.txt, copy to new location
            folder_name = os.path.basename(cache_dir)
            old_history_file = os.path.join("Cache_Files", "History", f"{folder_name}.txt")
            if not os.path.exists(self.history_file) and os.path.exists(old_history_file):
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    import shutil
                    shutil.copy2(old_history_file, self.history_file)
                except Exception as e:
                    print(f"Warning: Could not migrate old CLI history: {e}")
            
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.command_history = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Could not bind specific history file ({e}). Defaulting format.")
            saved_layout_dir = getattr(cfg, 'SAVED_LAYOUT_DIR', os.path.join("Cache_Files", "Saved_Layouts"))
            self.history_file = os.path.join(saved_layout_dir, "cli_history.txt")

        self.history_index = len(self.command_history)
        
        self.cluster_labels = None
        self.label_visuals = []
        

        self.full_headers = [] # Original headers (for caching/integrity)

        
        self.original_seqs = None       
        self.last_cluster_params = None 
        
        # Alignment Data 
        self.active_reference = cfg.ALIGNMENT_REFERENCE
        try:
            self.alignment_offset = int(getattr(cfg, 'ALIGNMENT_OFFSET', 0))
        except (TypeError, ValueError):
            self.alignment_offset = 0
        self.alignment = None
        self.col_to_label = None  
        self.label_to_col = None  
        
        # ---> NEW: Selection & Drag State <---
        self.selected_indices = []
        self.is_box_selecting = False
        self.is_multi_dragging = False
        self._drag_edges_hidden = False
        self.drag_start_mouse = None
        self.drag_start_screen = None
        self.drag_start_nodes_pos = None
        self.position_history = []  # Tracks states for Undo
        self.hud_displays = {}

        # --- 2. Data Loading & Simulation ---
        self.load_and_simulate()
        self.original_pos = self.pos.copy()  # <--- NEW: Backup original layout
        self.load_global_alignment()
        
        # --- 3. Setup Window & Canvas ---
        self.canvas = scene.SceneCanvas(keys=None, show=False, title="SSN Viewer (Live)", bgcolor='white')
        # Keep the complete traceback available if a future VisPy draw
        # callback fails; the default logarithmic reminders hide the cause
        # after the first occurrence.
        self.canvas.events.draw.print_callback_errors = 'first'
        qapp = QtWidgets.QApplication.instance()
        if qapp:
            try:
                qt_font_status = configure_qt_application_fonts(qapp)
                vispy_font_status = register_vispy_application_fonts(
                    qt_font_status
                )
                self.vispy_ui_face = vispy_font_status.ui_face
                self.vispy_monospace_face = vispy_font_status.monospace_face
            except Exception as e:
                print(f"Warning: Could not configure bundled application fonts: {e}")
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.resize.connect(self.on_resize)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        
        # --- NEW: Hook mouse wheel and move for dynamic tooltips and HUD ---
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        
        self.selected_node_idx = None
        # Rename the timer so it handles all dynamic HUD elements
        self._hud_timer = app.Timer(0.001, connect=self._update_hud_elements, iterations=1)
        
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom' 
        self.view.camera.aspect = 1
        
        # --- Disable default Vispy Backspace reset ---
        # Save the original bound method
        self._original_camera_key_event = self.view.camera.viewbox_key_event
        
        # Create a custom wrapper that filters out Backspace
        def safe_camera_key_event(event):
            if event.key == 'Backspace':
                return  # Block the event from reaching the camera
            self._original_camera_key_event(event)
            
        # Replace the camera's handler with our custom wrapper
        self.view.camera.viewbox_key_event = safe_camera_key_event

        # ---> NEW: Disable default Vispy Right-Click Zoom <---
        self._original_camera_mouse_event = self.view.camera.viewbox_mouse_event
        
        def safe_camera_mouse_event(event):
            # If the event involves the right mouse button (button 2), block it from the camera
            if getattr(event, 'button', None) == 2 or 2 in getattr(event, 'buttons', []):
                return 
            self._original_camera_mouse_event(event)
            
        self.view.camera.viewbox_mouse_event = safe_camera_mouse_event

        # --- 4. Draw Initial State ---
        self.draw_network()
        self.create_hud()
        self.background_job_scheduler = BackgroundJobScheduler(self)
        if hasattr(self.canvas.events, "close"):
            self.canvas.events.close.connect(
                lambda _event: self.background_job_scheduler.shutdown()
            )
        if qapp:
            qapp.aboutToQuit.connect(self.background_job_scheduler.shutdown)
        
        # 1. Find the extreme coordinates of the final grid
        min_x, min_y = np.min(self.pos[:, :2], axis=0)
        max_x, max_y = np.max(self.pos[:, :2], axis=0)
        
        # 2. Calculate dimensions
        width = max_x - min_x
        height = max_y - min_y
        
        # 3. Add a 5% padding margin (or at least 10 units) so nodes don't touch the window edge
        margin_x = max(width * 0.05, 10.0)
        margin_y = max(height * 0.05, 10.0)
        
        # 4. Snap the camera to this precise rectangle
        self.view.camera.set_range(
            x=(min_x - margin_x, max_x + margin_x), 
            y=(min_y - margin_y, max_y + margin_y)
        )
        
        # --- 5. Setup Similarity / E-value Slider Bar ---
        self.is_evalue = getattr(cfg, 'INPUT_IS_EVALUE', False)
        
        min_val = getattr(cfg, 'SIMILARITY_THRESHOLD', None)
        if min_val is None or min_val == "None":
            if hasattr(self, 'edge_scores') and len(self.edge_scores) > 0:
                min_val = float(np.min(self.edge_scores))
            else:
                min_val = 0.0
        self.min_threshold = float(min_val)
        
        if hasattr(self, 'edge_scores') and len(self.edge_scores) > 0:
            self.max_threshold = float(np.max(self.edge_scores))
        else:
            self.max_threshold = self.min_threshold + 1.0
            
        if self.min_threshold >= self.max_threshold:
            self.max_threshold = self.min_threshold + 1.0
            
        self.current_slider_threshold = self.min_threshold

        # Force light theme on the QApplication managed by Vispy
        if qapp:
            try:
                force_light_palette(qapp)
            except Exception as e:
                print(f"Warning: Could not force light palette: {e}")
            
            # Set Application-wide Icon (covers Vispy and all spawned windows/dialogs)
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
            if os.path.exists(icon_path):
                qapp.setWindowIcon(QtGui.QIcon(icon_path))

        # Create overlay container widget as a child of the native canvas
        self.slider_overlay = QtWidgets.QWidget(self.canvas.native)
        self.slider_overlay.setObjectName("sliderOverlay")
        
        overlay_layout = QtWidgets.QHBoxLayout(self.slider_overlay)
        overlay_layout.setContentsMargins(5, 5, 5, 5)
        overlay_layout.setSpacing(10)
        
        self.slider_label = QtWidgets.QLabel()
        self.slider_label.setObjectName("sliderLabel")
        self.slider_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.update_slider_label_text(self.current_slider_threshold)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        
        overlay_layout.addWidget(self.slider_label)
        overlay_layout.addWidget(self.slider)
        
        # Style sheet matching the third image
        self.slider_overlay.setStyleSheet("""
            QWidget#sliderOverlay {
                background: transparent;
            }
            QLabel#sliderLabel {
                font-family: %(font)s;
                font-size: 12pt;
                font-weight: normal;
                color: %(text_color)s;
                background: transparent;
                min-width: 45px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bcbcbc;
                height: 4px;
                background: #d8d8d8;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3A96A6;
                border: 1px solid #2E8B9A;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #bcbcbc;
                width: 14px;
                height: 16px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal:hover {
                background: #f5f5f5;
                border-color: #a0a0a0;
            }
            QSlider::handle:horizontal:pressed {
                background: #e5e5e5;
                border-color: #888888;
            }
        """ % {"font": UI_QSS_FONT_STACK, "text_color": cfg.TEXT_COLOR})
        
        self.position_slider_overlay()
        self.slider_overlay.show()
        
        # --- 6. Set up MainWindow & WebServer ---
        self._panel_w = 180
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("Sequence Similarity Network Viewer")
        
        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
        if os.path.exists(icon_path):
            self.main_window.setWindowIcon(QtGui.QIcon(icon_path))
            
        self.main_window.resize(1200, 800)
        self.main_window.setMinimumWidth(self._panel_w)
        
        # Set the Vispy canvas directly as the central widget
        self.main_window.setCentralWidget(self.canvas.native)
        
        # Collapsible Right Panel Container (overlay on the canvas.native)
        self.right_panel = QtWidgets.QWidget(self.canvas.native)
        self.right_panel.setObjectName("rightPanel")
        right_panel_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_panel_layout.setContentsMargins(10, 20, 10, 20)
        right_panel_layout.setSpacing(15)
        right_panel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        # Stretch at the bottom to keep dynamic buttons at the top
        self.right_panel_layout = right_panel_layout
        self.right_panel_layout.addStretch()
        
        # Single floating toggle button on the canvas.native to collapse/expand sidebar
        self.toggle_sidebar_btn = QtWidgets.QPushButton(">>", self.canvas.native)
        self.toggle_sidebar_btn.setObjectName("toggleSidebarBtn")
        self.toggle_sidebar_btn.setToolTip("Toggle sidebar panel")
        self.toggle_sidebar_btn.setFixedWidth(30)
        self.toggle_sidebar_btn.setFixedHeight(30)
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        self.toggle_sidebar_btn.hide()
        
        # Apply modern premium stylesheet
        self.main_window.setStyleSheet("""
            QMainWindow {
                background-color: #f7f7f7;
            }
            QWidget#rightPanel {
                background-color: rgba(255, 255, 255, 0.95);
                border-left: 1px solid #dcdcdc;
            }
            QPushButton#toggleSidebarBtn {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                font-weight: bold;
                color: %(text_color)s;
            }
            QPushButton#toggleSidebarBtn:hover {
                background-color: #f0f0f0;
                border-color: #c0c0c0;
            }
            QPushButton#toggleSidebarBtn:pressed {
                background-color: #e5e5e5;
            }
            QWidget#rightPanel QPushButton {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
                font-weight: bold;
                color: %(text_color)s;
                font-family: %(font)s;
                font-size: 10pt;
                padding-left: 10px;
                padding-right: 10px;
            }
            QWidget#rightPanel QPushButton:hover {
                background-color: #f0f8ff;
                border-color: #0969da;
            }
            QWidget#rightPanel QPushButton:pressed {
                background-color: #e2f0fe;
            }
        """ % {"font": UI_QSS_FONT_STACK, "text_color": cfg.TEXT_COLOR})
        
        # Initialize thread-safe QtCommunicator for server commands
        from web_ui import Web_Server
        self.communicator = Web_Server.QtCommunicator(self)
        
        # Discover bundled web plugins before the server begins accepting requests.
        self.web_server = None
        self.web_server_url = None
        self.web_plugin_manager = None
        try:
            from web_ui.Plugin_Manager import WebPluginManager
            self.web_plugin_manager = WebPluginManager(self)
            self.web_plugin_manager.discover_and_register()
        except Exception as error:
            print(f"Web plugin discovery failed: {error}")

        # Initialize background WebServer
        self.start_web_server()
        
        # Run persistent sidebar button registration commands
        if getattr(self, 'sidebar_buttons_to_persist', None):
            for cmd_name in self.sidebar_buttons_to_persist:
                self.process_command(f"{cmd_name} --register-only", record_history=False, silent=True)
        
        # Ensure the side panel is hidden at startup
        self.set_sidebar_visible(False)
        
        show_window_in_front(self.main_window)
        QtCore.QTimer.singleShot(0, self._initialize_display_tracking)
        
        self._hud_timer.start()
        print("\nViewer Ready. Press [ENTER] to type commands.")

    def _update_console_text(self):
        """Helper to render the command line with a visible cursor."""
        buf = self.input_buffer
        c = self.cursor_pos
        self.console_text.text = f"Cmd: {buf[:c]}_{buf[c:]}"
        self.update_console_background()
        self.canvas.update()

    def _canvas_dpi(self):
        """Return the DPI value VisPy uses to convert text points to pixels."""
        return float(getattr(self.canvas, 'dpi', 96.0))

    def _hud_font_size_points(self):
        """Return a cross-platform VisPy point size for the logical HUD size."""
        logical_pixels = self.hud_layout.get("font_size_px", 16.0)
        return vispy_points_for_logical_pixels(logical_pixels, self._canvas_dpi())

    def _tooltip_font_size_points(self):
        """Normalize the configurable tooltip size to its 96-DPI appearance."""
        return vispy_points_at_reference_dpi(cfg.TEXT_SIZE, self._canvas_dpi())

    def _status_hud_position(self, line_index, size=None):
        """Return one evenly-spaced position in the bottom-right HUD stack."""
        if size is None:
            size = self.canvas.size
        cfg_hud = self.hud_layout
        return (
            size[0] - cfg_hud["status_x_offset"],
            size[1] - (
                cfg_hud["status_bottom_offset"]
                + line_index * cfg_hud["status_line_spacing"]
            ),
        )

    def _apply_vispy_text_scaling(self):
        """Apply logical-pixel sizing to static and dynamically-created text."""
        hud_font_size = self._hud_font_size_points()
        for attribute_name in (
            'instr_text',
            'console_text',
            'background_job_status_text',
            'zoom_text',
            'hidden_text',
        ):
            visual = getattr(self, attribute_name, None)
            if visual is not None:
                visual.font_size = hud_font_size

        tooltip = getattr(self, 'tooltip', None)
        if tooltip is not None:
            tooltip.font_size = self._tooltip_font_size_points()

        for display in getattr(self, 'hud_displays', {}).values():
            visual = getattr(display, 'text_visual', None)
            if visual is not None:
                visual.font_size = hud_font_size

    def update_console_background(self):
        """Wrap the console visually and size its background to the rendered rows."""
        if not hasattr(self, 'console_bg') or not hasattr(self, 'console_text'):
            return
        
        cfg_hud = self.hud_layout
        text_visual = self.console_text

        def measure_text_width(text):
            """Measure one rendered row using VisPy's own glyph metrics."""
            if not text or not hasattr(text_visual, '_font'):
                return 0.0

            font = text_visual._font
            dpi = text_visual.transforms.dpi
            font_size = text_visual.font_size
            n_pix = (font_size / 72.0) * dpi
            ratio = 1.0 / getattr(font, 'ratio', 4.0)
            width_val = 0.0
            prev = None
            for char in text:
                glyph = font[char]
                kerning = glyph['kerning'].get(prev, 0.0) * ratio
                x_move = glyph['advance'] * ratio + kerning
                width_val += x_move
                prev = char
            return (width_val / 64.0) * n_pix

        left_edge = cfg_hud["console_bg_left_offset"]
        left_gap = cfg_hud["console_text_x"] - cfg_hud["console_bg_left_offset"]
        padding_x = cfg_hud.get("console_bg_padding_x", 20.0)

        panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
        panel_width = getattr(self, '_panel_w', 120) if panel_visible else 0
        effective_canvas_width = self.canvas.size[0] - panel_width
        max_width = max(1.0, effective_canvas_width - 40.0)
        available_text_width = max(
            1.0,
            max_width - left_gap - padding_x,
        )

        current_text = text_visual.text or ""
        previous_rendered_text = getattr(self, '_console_rendered_text', None)
        if previous_rendered_text is None or current_text != previous_rendered_text:
            # Overlay output remains one logical line. Newlines supplied by a
            # command are normalized; only the rendered copy receives wrapping.
            logical_text = current_text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
            self._console_logical_text = logical_text
        else:
            logical_text = getattr(self, '_console_logical_text', current_text)

        rendered_text = _wrap_console_text_for_display(
            logical_text,
            available_text_width,
            measure_text_width,
        )
        self._console_rendered_text = rendered_text
        if current_text != rendered_text:
            text_visual.text = rendered_text

        rendered_lines = rendered_text.split('\n') if rendered_text else ['']
        text_width = max(
            (measure_text_width(line) for line in rendered_lines),
            default=0.0,
        )

        min_width = min(
            cfg_hud["console_bg_min_width"],
            max_width,
        )
        desired_width = left_gap + text_width + padding_x
        width = min(max_width, max(min_width, desired_width))

        base_height = cfg_hud["console_bg_height"]
        font_line_height = _vispy_text_line_height_pixels(text_visual)
        extra_height = max(0, len(rendered_lines) - 1) * font_line_height
        height = base_height + extra_height
        radius = cfg_hud["console_bg_radius"]
        center_x = left_edge + width / 2.0
        text_y = cfg_hud["console_text_y"]
        text_anchor_y = cfg_hud.get("console_text_anchor_y", "bottom")
        if text_anchor_y == "top":
            single_line_center_y = text_y + font_line_height / 2.0
        elif text_anchor_y in ("center", "middle"):
            single_line_center_y = text_y
        else:
            single_line_center_y = text_y - font_line_height / 2.0
        # Keep the single-line top edge fixed so wrapped command lines and the
        # background both extend downward together.
        center_y = (
            single_line_center_y
            + extra_height / 2.0
            + cfg_hud.get("console_bg_y_offset", 0.0)
        )

        _apply_safe_rectangle_geometry(
            self.console_bg,
            center=(center_x, center_y),
            width=width,
            height=height,
            radius=radius,
        )

        self.update_background_job_status_layout()

    def update_background_job_status_layout(self):
        """Wrap and position scheduler status beneath the current command box."""
        if (
            not hasattr(self, 'background_job_status_text')
            or not hasattr(self, 'console_bg')
        ):
            return

        cfg_hud = self.hud_layout
        text_visual = self.background_job_status_text

        def measure_text_width(text):
            if not text or not hasattr(text_visual, '_font'):
                return 0.0

            font = text_visual._font
            dpi = text_visual.transforms.dpi
            font_size = text_visual.font_size
            n_pix = (font_size / 72.0) * dpi
            ratio = 1.0 / getattr(font, 'ratio', 4.0)
            width_val = 0.0
            previous = None
            for char in text:
                glyph = font[char]
                kerning = glyph['kerning'].get(previous, 0.0) * ratio
                width_val += glyph['advance'] * ratio + kerning
                previous = char
            return (width_val / 64.0) * n_pix

        panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
        panel_width = getattr(self, '_panel_w', 120) if panel_visible else 0
        effective_canvas_width = self.canvas.size[0] - panel_width
        right_padding = cfg_hud.get("background_job_status_right_padding", 20.0)
        available_width = max(
            1.0,
            effective_canvas_width - cfg_hud["console_text_x"] - right_padding,
        )

        current_text = text_visual.text or ""
        previous_rendered_text = getattr(
            self,
            '_background_job_status_rendered_text',
            None,
        )
        if previous_rendered_text is None or current_text != previous_rendered_text:
            logical_text = current_text.replace(
                '\r\n', ' '
            ).replace('\r', ' ').replace('\n', ' ')
            self._background_job_status_logical_text = logical_text
        else:
            logical_text = getattr(
                self,
                '_background_job_status_logical_text',
                current_text,
            )

        rendered_text = _wrap_console_text_for_display(
            logical_text,
            available_width,
            measure_text_width,
        )
        self._background_job_status_rendered_text = rendered_text
        if current_text != rendered_text:
            text_visual.text = rendered_text

        box_bottom = self.console_bg.center[1] + self.console_bg.height / 2.0
        text_visual.pos = (
            cfg_hud["console_text_x"],
            box_bottom + cfg_hud.get("background_job_status_gap", 8.0),
        )
        text_visual.visible = bool(logical_text)

    def set_background_job_status(self, message):
        """Show a scheduler lifecycle message without altering command input."""
        if not hasattr(self, 'background_job_status_text'):
            return
        self.background_job_status_text.text = str(message)
        self.update_background_job_status_layout()
        self.canvas.update()

    def clear_background_job_status(self):
        """Hide the previous scheduler status when a new command is opened."""
        if not hasattr(self, 'background_job_status_text'):
            return
        self._background_job_status_logical_text = ""
        self._background_job_status_rendered_text = ""
        self.background_job_status_text.text = ""
        self.background_job_status_text.visible = False
        self.canvas.update()

    def load_and_simulate(self):
            """
            Loads layout. Saves original headers to cache, but uses simplified headers in memory.
            """
            if not os.path.exists(cfg.SAVED_LAYOUT_DIR):
                os.makedirs(cfg.SAVED_LAYOUT_DIR)
                
            # --- Resolve Path and Header ---
            cache_path, self.resolved_ref_full = resolve_selected_cache(cfg)
            print(f"Target Cache File: {cache_path}")
            cache_mode = getattr(cfg, 'TARGET_CACHE_MODE', None)
            if cache_mode not in {'existing', 'new'}:
                cache_mode = 'existing' if os.path.exists(cache_path) else 'new'

            selected_fasta_path = (
                getattr(cfg, 'NODE_FASTA_FILE', None)
                or getattr(cfg, 'SEQUENCES_FILE', '')
            )
            self._selected_fasta_records = _load_selected_fasta_records(
                selected_fasta_path
            )
            self.sequences_map = _build_sequence_lookup(
                self._selected_fasta_records
            )
            selected_fasta_headers = [
                header for header, _ in self._selected_fasta_records
            ]

            manifest_settings = {
                'alignment_score': getattr(cfg, 'ALIGNMENT_SCORE', None),
                'normalization': getattr(cfg, 'NORM_MODE', None),
                'umap_mode': getattr(cfg, 'UMAP_MODE', False),
                'umap_neighbors': getattr(cfg, 'UMAP_NEIGHBORS', 15),
                'top_edge_percent': getattr(cfg, 'TOP_EDGE_PERCENT', None),
                'similarity_threshold': getattr(cfg, 'SIMILARITY_THRESHOLD', None),
            }
            try:
                current_manifest = cache_manifest.build_manifest_for_files(
                    selected_fasta_path,
                    cfg.INPUT_HDF5,
                    **manifest_settings,
                )
            except Exception as error:
                raise RuntimeError(f"Unable to fingerprint cache inputs: {error}") from error
            self.cache_manifest = current_manifest
            self.cache_manifest_id = current_manifest['manifest_id']
            cfg.CACHE_MANIFEST_ID = self.cache_manifest_id

            raw_loaded = False
            self._metadata_loaded_from_cache = False

            # --- Try Loading Cache ---
            if cache_mode == 'existing':
                if not os.path.exists(cache_path):
                    raise RuntimeError(f"Selected cache file does not exist: {cache_path}")
                print(f"--- Found Cached Layout! ---")
                try:
                    import json
                    stored_manifest = cache_manifest.read_manifest(
                        os.path.dirname(cache_path),
                        current_manifest['compatibility'],
                    )
                    if stored_manifest['manifest_id'] != self.cache_manifest_id:
                        raise cache_manifest.CacheManifestError(
                            "Selected cache folder manifest does not match current inputs."
                        )

                    with h5py.File(cfg.INPUT_HDF5, "r") as raw_data:
                        expected_headers, expected_edges, expected_edge_scores = prepare_network(
                            raw_data,
                            settings=cfg,
                            selected_fasta_headers=selected_fasta_headers,
                        )
                    with h5py.File(cache_path, "r") as hf:
                        cache_manifest.validate_cache_hdf5(
                            hf, expected_headers, self.cache_manifest_id
                        )
                        raw_headers = hf["headers"][:]
                        self.full_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        self.pos = hf["positions"][:].astype(np.float32)
                        
                        self.n_nodes = len(self.full_headers)
                        
                        if "colors" in hf: self.current_colors = hf["colors"][:]
                        if "sizes" in hf:
                            self.current_sizes = hf["sizes"][:].astype(np.float32)
                            # Sync/Rescale cached sizes with current cfg.NODE_SIZE
                            old_base = hf.attrs.get("base_node_size", None)
                            if old_base is not None:
                                try:
                                    old_base = float(old_base)
                                    new_base = float(cfg.NODE_SIZE)
                                    if old_base > 0 and old_base != new_base:
                                        self.current_sizes = self.current_sizes * (new_base / old_base)
                                except Exception as e:
                                    print(f"Warning: Failed to rescale cached node sizes: {e}")
                            else:
                                # Backward compatibility: if all sizes are uniform, overwrite with current cfg.NODE_SIZE
                                if len(self.current_sizes) > 0:
                                    first_val = self.current_sizes[0]
                                    if np.allclose(self.current_sizes, first_val):
                                        self.current_sizes.fill(cfg.NODE_SIZE)
                                    else:
                                        # Rescale assuming old base was 10.0
                                        new_base = float(cfg.NODE_SIZE)
                                        self.current_sizes = self.current_sizes * (new_base / 10.0)
                        if "shapes" in hf:
                            raw_shapes = hf["shapes"][:]
                            self.current_shapes = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in raw_shapes], dtype=object)
                        if "visible_mask" in hf: self.visible_mask = hf["visible_mask"][:]
                        if "node_render_order" in hf:
                            self.node_render_order = cache_manifest.validate_node_render_order(
                                hf["node_render_order"][:], self.n_nodes
                            ).copy()
                        if "cluster_labels" in hf: self.cluster_labels = hf["cluster_labels"][:]
                        
                        # --- Load Metadata from Cache ---
                        self.metadata = {}
                        if "metadata" in hf:
                            self._metadata_loaded_from_cache = True
                            meta_group = hf["metadata"]
                            for prop_name in meta_group.keys():
                                ds = meta_group[prop_name]
                                prop_type = ds.attrs.get("type", "text")
                                raw_vals = ds[:]
                                if prop_name == "Length":
                                    values = raw_vals.astype(np.int32)
                                elif prop_type == "number":
                                    values = raw_vals.astype(np.float64)
                                else:
                                    values = np.array([v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in raw_vals], dtype=object)
                                self.metadata[prop_name] = {
                                    "type": prop_type,
                                    "values": values
                                }
                                
                        # --- Load Custom Dynamic Attributes from Cache (Root Level) ---
                        if not hasattr(self, '_cacheable_attrs'):
                            self._cacheable_attrs = set()
                        
                        # Register manually configured attributes
                        for attr_name in CUSTOM_ATTRIBUTES_INIT.keys():
                            self._cacheable_attrs.add(attr_name)
                            
                        # Scan root-level keys for any non-core custom datasets
                        CORE_DATASETS = {
                            "headers", "positions", "colors", "sizes", "shapes", 
                            "visible_mask", "cluster_labels", "group_labels", "metadata",
                            "connectivity", "edge_scores", "node_render_order"
                        }
                        for key in hf.keys():
                            if key not in CORE_DATASETS:
                                ds = hf[key]
                                if "is_json" in ds.attrs and ds.attrs["is_json"]:
                                    import json
                                    raw_val = ds[()]
                                    if isinstance(raw_val, bytes):
                                        raw_val = raw_val.decode('utf-8')
                                    setattr(self, key, json.loads(raw_val))
                                else:
                                    setattr(self, key, ds[:])
                                self._cacheable_attrs.add(key)
                        
                        # --- Safely decode strings/bytes ---
                        if "group_labels" in hf:
                            gl_data = hf["group_labels"][()]
                            if isinstance(gl_data, bytes):
                                gl_data = gl_data.decode('utf-8')
                            self.group_labels = [set(g) for g in json.loads(gl_data)]
                            
                        if "last_cluster_params" in hf.attrs: 
                            val = hf.attrs["last_cluster_params"]
                            if isinstance(val, bytes): val = val.decode('utf-8')
                            if isinstance(val, str) and val.startswith('['):
                                self.last_cluster_params = tuple(json.loads(val))
                            else:
                                self.last_cluster_params = tuple(val)
                                
                    # Headers were validated in exact order, so fresh edges can be used directly.
                    print("Using fresh connectivity and edge scores from raw network file...")
                    self.edges = expected_edges.astype(np.int32, copy=False)
                    self.edge_scores = expected_edge_scores.astype(np.float32, copy=False)
                    
                    base_box = np.sqrt(self.n_nodes) * 2.5 + 5.0
                    self.box_limit = base_box * cfg.BOX_SCALE
                    
                    raw_loaded = True

                except Exception as e:
                    raise RuntimeError(
                        f"Selected cache is incompatible or invalid: {e}. "
                        "Choose '(New Layout Cache)' in SSN Config."
                    ) from e

            # --- Calculate from Scratch (if cache failed or missing) ---
            if not raw_loaded:
                if cache_mode != 'new':
                    raise RuntimeError("Existing cache validation did not complete.")
                clean_hdf5_path = os.path.normpath(cfg.INPUT_HDF5)
                print(f"--- Calculating New Layout (Raw: {clean_hdf5_path}) ---")
                try:
                    generation_settings = LayoutGenerationSettings.from_namespace(
                        cfg,
                        cache_filename=os.path.basename(cache_path),
                        project_root=os.path.dirname(_SRC_DIR),
                        target_cache_path=cache_path,
                    )
                    result = generate_layout_cache(generation_settings)
                except Exception as error:
                    raise RuntimeError(
                        f"Could not generate layout cache: {error}"
                    ) from error

                self.cache_manifest = result.manifest
                self.cache_manifest_id = result.manifest["manifest_id"]
                cfg.CACHE_MANIFEST_ID = self.cache_manifest_id
                cfg.SIMILARITY_THRESHOLD = result.effective_similarity_threshold
                self._selected_fasta_records = result.fasta_records
                self.sequences_map = _build_sequence_lookup(result.fasta_records)
                self.full_headers = result.full_headers
                self.edges = result.edges
                self.edge_scores = result.edge_scores
                self.pos = result.positions
                self.box_limit = result.box_limit
                self.n_nodes = len(self.full_headers)
                print(
                    f"Network Built: {self.n_nodes} Nodes, {len(self.edges)} Edges."
                )

            self._init_colors()

    def _init_colors(self):
        import matplotlib.colors as mcolors
        
        # Only initialize if they weren't loaded from the cache
        if not hasattr(self, 'current_colors'):
            n_rgba = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
            self.current_colors = np.tile(n_rgba, (self.n_nodes, 1)).astype(np.float32)
        if not hasattr(self, 'current_sizes'): self.current_sizes = np.full(self.n_nodes, cfg.NODE_SIZE, dtype=np.float32)
        if not hasattr(self, 'current_shapes'): self.current_shapes = np.full(self.n_nodes, 'disc', dtype=object)
        if not hasattr(self, 'visible_mask'): self.visible_mask = np.ones(self.n_nodes, dtype=bool)
        if not hasattr(self, 'node_render_order'):
            self.node_render_order = np.arange(self.n_nodes, dtype=np.int32)
        if not hasattr(self, 'redo_stack'): self.redo_stack = []
        if not hasattr(self, 'selected_indices'): self.selected_indices = []
        if not hasattr(self, 'cluster_labels'): self.cluster_labels = None
        if not hasattr(self, 'group_labels'): self.group_labels = [set() for _ in range(self.n_nodes)]
        if not hasattr(self, 'metadata'): self.metadata = {}
        
        # New layouts start with generated Length metadata.  A saved metadata
        # group without Length represents an intentional column deletion.
        if (
            "Length" not in self.metadata
            and not getattr(self, "_metadata_loaded_from_cache", False)
        ):
            lengths_map = {
                header: len(sequence)
                for header, sequence in self.sequences_map.items()
            }
            
            length_values = np.zeros(self.n_nodes, dtype=np.int32)
            for i, h in enumerate(self.full_headers):
                rec_id = h.split()[0]
                if h in lengths_map:
                    length_values[i] = lengths_map[h]
                elif rec_id in lengths_map:
                    length_values[i] = lengths_map[rec_id]
            
            self.metadata["Length"] = {
                "type": "number",
                "values": length_values
            }
            
        # Reorder metadata dictionary so "Length" is the first property
        if self.metadata and "Length" in self.metadata:
            ordered_metadata = {"Length": self.metadata["Length"]}
            for k, v in self.metadata.items():
                if k != "Length":
                    ordered_metadata[k] = v
            self.metadata = ordered_metadata
            
        # Initialize Dynamic Registry
        if not hasattr(self, '_cacheable_attrs'):
            self._cacheable_attrs = set()
            
        # Initialize manual custom attributes from the top-level section
        for attr_name, default_val in CUSTOM_ATTRIBUTES_INIT.items():
            if not hasattr(self, attr_name):
                setattr(self, attr_name, default_val)
            self._cacheable_attrs.add(attr_name)

    def _get_current_state(self):
        """Helper to package the entire visual and spatial state."""
        return {
            'pos': self.pos.copy() if hasattr(self, 'pos') else None,
            'visible_mask': self.visible_mask.copy() if hasattr(self, 'visible_mask') else None,
            'colors': self.current_colors.copy() if hasattr(self, 'current_colors') else None,
            'sizes': self.current_sizes.copy() if hasattr(self, 'current_sizes') else None,
            'shapes': self.current_shapes.copy() if hasattr(self, 'current_shapes') else None,
            'node_render_order': self.node_render_order.copy() if hasattr(self, 'node_render_order') else None,
            'clusters': self.cluster_labels.copy() if getattr(self, 'cluster_labels', None) is not None else None,
            'groups': [g.copy() for g in self.group_labels] if getattr(self, 'group_labels', None) is not None else None,
            'last_cluster_params': self.last_cluster_params if getattr(self, 'last_cluster_params', None) is not None else None,
            'metadata': {k: {'type': v['type'], 'values': v['values'].copy()} for k, v in self.metadata.items()} if getattr(self, 'metadata', None) else {},
            '_custom_data': self._get_custom_attributes_snapshot()
        }

    def _apply_state(self, state):
        """Helper to unpack a state dictionary and apply it to the viewer."""
        if state['pos'] is not None: self.pos = state['pos'].copy()
        if state['visible_mask'] is not None: self.visible_mask = state['visible_mask'].copy()
        if state['colors'] is not None: self.current_colors = state['colors'].copy()
        if state['sizes'] is not None: self.current_sizes = state['sizes'].copy()
        if state['shapes'] is not None: self.current_shapes = state['shapes'].copy()
        if state.get('node_render_order') is not None:
            self.node_render_order = cache_manifest.validate_node_render_order(
                state['node_render_order'], self.n_nodes
            ).copy()
        else:
            self.node_render_order = np.arange(self.n_nodes, dtype=np.int32)
        
        if state['clusters'] is not None: 
            self.cluster_labels = state['clusters'].copy()
            self.last_cluster_params = state.get('last_cluster_params')
        else:
            self.cluster_labels = None
            self.last_cluster_params = None

        if state.get('groups') is not None:
            self.group_labels = [g.copy() for g in state['groups']]
        else:
            self.group_labels = [set() for _ in range(self.n_nodes)]
            
        if state.get('metadata') is not None:
            self.metadata = {k: {'type': v['type'], 'values': v['values'].copy()} for k, v in state['metadata'].items()}
        else:
            self.metadata = {}
            
        if '_custom_data' in state and state['_custom_data'] is not None:
            self._apply_custom_attributes_snapshot(state['_custom_data'])
            
        # Clean up any active selections if those nodes are now hidden in this restored state
        if hasattr(self, 'selected_indices'):
            self.selected_indices = [i for i in self.selected_indices if self.visible_mask[i]]
            
        self.update_selection_visual()
        self.update_edges()
        self._refresh_metadata_views()

    def _get_custom_attributes_snapshot(self):
        if not getattr(self, '_cacheable_attrs', None):
            return {}
        snapshot = {}
        for attr_name in self._cacheable_attrs:
            val = getattr(self, attr_name, None)
            if isinstance(val, np.ndarray):
                snapshot[attr_name] = val.copy()
            else:
                import copy
                snapshot[attr_name] = copy.deepcopy(val)
        return snapshot

    def _apply_custom_attributes_snapshot(self, snapshot):
        if not hasattr(self, '_cacheable_attrs'):
            self._cacheable_attrs = set()
        for attr_name, val in snapshot.items():
            if isinstance(val, np.ndarray):
                setattr(self, attr_name, val.copy())
            else:
                import copy
                setattr(self, attr_name, copy.deepcopy(val))
            self._cacheable_attrs.add(attr_name)

    def _append_history_entry(self, entry):
        """Append a full-state or compact history entry and clear redo state."""
        self.position_history.append(entry)
        if len(self.position_history) > 50:
            self.position_history.pop(0)
        self.redo_stack.clear()

    def _save_metadata_cell_history(self, column, row, before, after):
        """Record one metadata cell edit without copying the full network state."""
        self._append_history_entry({
            "_history_kind": "metadata_cell",
            "column": column,
            "row": int(row),
            "before": before,
            "after": after,
        })

    def _save_metadata_column_history(self, removed_columns, hud_property=None):
        """Record deleted metadata columns and their original ordering."""
        self._append_history_entry({
            "_history_kind": "metadata_columns",
            "columns": removed_columns,
            "hud_property": hud_property,
        })

    def _refresh_metadata_views(self):
        """Refresh native metadata widgets after metadata shape/value changes."""
        source_model = getattr(self, "metadata_source_model", None)
        if source_model is not None and hasattr(source_model, "refresh_columns"):
            source_model.refresh_columns()

        table_view = getattr(self, "metadata_table_view", None)
        if table_view is not None:
            header = table_view.horizontalHeader()
            if hasattr(header, "setFilterBoxes") and source_model is not None:
                header.setFilterBoxes(source_model.columnCount())

        proxy_model = getattr(self, "metadata_proxy_model", None)
        if proxy_model is not None:
            proxy_model.invalidateFilter()

    def _set_metadata_hud_property(self, property_name):
        """Synchronize the optional clicked-node metadata HUD with history."""
        self.meta_display_prop = property_name
        display = getattr(self, "hud_displays", {}).get("meta_display")
        if display is None:
            return
        if property_name is None:
            display.hide()
            return

        node_idx = getattr(self, "selected_node_idx", None)
        entry = getattr(self, "metadata", {}).get(property_name)
        if entry is None or node_idx is None:
            display.show(f"{property_name}: -")
            return

        value = entry["values"][node_idx]
        if value is None or (
            isinstance(value, (float, np.floating)) and np.isnan(value)
        ):
            value_text = "N/A"
        else:
            value_text = str(value).strip() or "N/A"
        display.show(f"{property_name}: {value_text}")

    def _apply_metadata_history_entry(self, entry, undo):
        """Apply one compact metadata history entry in either direction."""
        kind = entry.get("_history_kind")
        if kind == "metadata_cell":
            column = entry["column"]
            metadata_entry = self.metadata.get(column)
            if metadata_entry is not None:
                value = entry["before"] if undo else entry["after"]
                metadata_entry["values"][entry["row"]] = value
        elif kind == "metadata_columns":
            removed_columns = entry["columns"]
            if undo:
                restored_items = list(self.metadata.items())
                for removed in sorted(
                    removed_columns, key=lambda item: item["index"]
                ):
                    restored_entry = {
                        "type": removed["type"],
                        "values": removed["values"].copy(),
                    }
                    restored_items.insert(
                        min(removed["index"], len(restored_items)),
                        (removed["name"], restored_entry),
                    )
                self.metadata = dict(restored_items)
                if entry.get("hud_property"):
                    self._set_metadata_hud_property(entry["hud_property"])
            else:
                for removed in removed_columns:
                    self.metadata.pop(removed["name"], None)
                if entry.get("hud_property"):
                    self._set_metadata_hud_property(None)
        else:
            raise ValueError(f"Unsupported history entry kind: {kind}")

        self._refresh_metadata_views()

    def _save_state(self):
        """Saves current state to history and clears redo stack."""
        self._append_history_entry(self._get_current_state())
        
    def _do_undo(self):
        if len(self.position_history) > 0:
            state = self.position_history.pop()
            if state.get("_history_kind"):
                self.redo_stack.append(state)
                self._apply_metadata_history_entry(state, undo=True)
            else:
                self.redo_stack.append(self._get_current_state())
                self._apply_state(state)
            msg = "Undo successful."
            changed = True
        else:
            msg = "Nothing to undo."
            changed = False
        self.console_text.text = msg
        print(msg)
        if changed:
            self.broadcast_metadata_state()
        return changed

    def _do_redo(self):
        if len(self.redo_stack) > 0:
            state = self.redo_stack.pop()
            if state.get("_history_kind"):
                self.position_history.append(state)
                self._apply_metadata_history_entry(state, undo=False)
            else:
                self.position_history.append(self._get_current_state())
                self._apply_state(state)
            msg = "Redo successful."
            changed = True
        else:
            msg = "Nothing to redo."
            changed = False
        self.console_text.text = msg
        print(msg)
        if changed:
            self.broadcast_metadata_state()
        return changed

    def load_global_alignment(self):
        """
        Loads alignment using the new standalone Alignment_Manager.
        """
        import Alignment_Manager
        self.alignment = Alignment_Manager.Alignment_Manager(
            cfg.MSA_FILE,
            full_headers=self.full_headers,
            active_reference=self.active_reference,
            alignment_offset=self.alignment_offset,
        )


    def draw_network(self):
        edge_coords = []
        if len(self.edges) > 0:
            for u, v in self.edges:
                edge_coords.append(self.pos[u])
                edge_coords.append(self.pos[v])
            if edge_coords:
                import matplotlib.colors as mcolors
                # Fetch custom edge color, fallback to black
                edge_rgba = list(mcolors.to_rgba(getattr(cfg, 'EDGE_COLOR', '#000000')))
                edge_rgba[3] = cfg.EDGE_ALPHA # Apply transparency

                edge_positions = _contiguous_line_positions(edge_coords)
                self.line_visual = scene.visuals.Line(
                    pos=edge_positions, connect='segments',
                    color=tuple(edge_rgba), width=cfg.EDGE_WIDTH,
                    parent=self.view.scene, name='network_edges',
                )
                self.line_visual.set_gl_state('translucent', depth_test=False)
                if getattr(cfg, 'UMAP_MODE', False):
                    self.line_visual.visible = False
        else:
            self.line_visual = None
            
        # Left-click rings are interleaved into this marker visual immediately
        # before their nodes so their layer relationship remains exact.
        self.left_click_highlight = None
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.update_nodes()
        
        # --- MODIFIED: Parent changed to canvas.scene ---
        self.tooltip = scene.visuals.Text(
            text="",
            color=cfg.TEXT_COLOR,
            pos=(0, 0),
            anchor_x='left',
            face=self.vispy_ui_face,
            font_size=self._tooltip_font_size_points(),
            parent=self.canvas.scene,
        )

        # ---> NEW: Visuals for Selection Feedback <---
        self.selection_box = scene.visuals.Line(
            color='black', method='gl',
            parent=self.view.scene, name='selection_box',
        )
        self.selection_box.visible = False
        
        self.selection_highlight = scene.visuals.Markers(parent=self.view.scene)
        # Initialize with a single dummy point so Vispy builds the internal vertex buffers
        self.selection_highlight.set_data(pos=np.array([[0.0, 0.0]], dtype=np.float32))
        self.selection_highlight.set_gl_state('translucent', depth_test=False)
        self.selection_highlight.visible = False

    def update_selection_visual(self):
        """Triggers a node update to draw selection edges and ensures the old highlight is hidden."""
        if hasattr(self, 'selection_highlight'):
            self.selection_highlight.visible = False
            
        self.update_nodes()
        self.update_edges()
        
        # Broadcast the selection change to SSE clients
        self.broadcast_event({"type": "selection_changed", "indices": self.selected_indices})
    
    def format_sig_figs(self, val):
        if val == 0:
            return "0.00"
        try:
            decimals = 2 - int(math.floor(math.log10(abs(val))))
            if decimals < 0:
                return f"{round(val, decimals):g}"
            else:
                return f"{val:.{decimals}f}"
        except:
            return f"{val:.3g}"

    def update_slider_label_text(self, threshold):
        formatted = self.format_sig_figs(threshold)
        self.slider_label.setText(formatted)

    def position_slider_overlay(self):
        if hasattr(self, 'slider_overlay') and hasattr(self, 'canvas'):
            canvas_w, canvas_h = self.canvas.size
            panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
            panel_w = getattr(self, '_panel_w', 120) if panel_visible else 0
            effective_w = canvas_w - panel_w
            overlay_w = max(100, effective_w - 220)
            overlay_h = 45
            overlay_x = 20
            overlay_y = canvas_h - overlay_h - 15
            self.slider_overlay.setGeometry(overlay_x, overlay_y, overlay_w, overlay_h)

    def on_slider_value_changed(self, value):
        self.current_slider_threshold = self.min_threshold + (value / 1000.0) * (self.max_threshold - self.min_threshold)
        self.update_slider_label_text(self.current_slider_threshold)
        self.update_edges()

    def update_edges(self):
        """Updates the line visuals to follow nodes dynamically using fast vectorization."""
        if getattr(self, 'line_visual', None) is not None and len(self.edges) > 0:
            # ---> NEW: Only draw edges where BOTH connected nodes are visible and above the active slider threshold <---
            current_slider_val = getattr(self, 'current_slider_threshold', getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0))
            current_vis_hash = (self.visible_mask.tobytes(), current_slider_val)
            
            if getattr(self, '_last_vis_mask_hash', None) != current_vis_hash:
                self._last_vis_mask_hash = current_vis_hash
                if hasattr(self, 'sync_metadata_table_visibility'):
                    self.sync_metadata_table_visibility()
                nodes_visible_mask = self.visible_mask[self.edges[:, 0]] & self.visible_mask[self.edges[:, 1]]
                
                if hasattr(self, 'edge_scores') and len(self.edge_scores) > 0:
                    threshold_visible_mask = self.edge_scores >= current_slider_val
                    valid_edges_mask = nodes_visible_mask & threshold_visible_mask
                else:
                    valid_edges_mask = nodes_visible_mask
                    
                self._cached_active_edges = self.edges[valid_edges_mask]
                
            active_edges = self._cached_active_edges
            
            # --- Low Resource Mode: Hide edges of dragged nodes ---
            if getattr(cfg, 'LOW_RESOURCE_MODE', False) and getattr(self, 'is_multi_dragging', False):
                if getattr(self, 'selected_indices', None) and len(self.selected_indices) > 0:
                    selected_set_arr = np.array(self.selected_indices)
                    mask_u_moved = np.isin(active_edges[:, 0], selected_set_arr)
                    mask_v_moved = np.isin(active_edges[:, 1], selected_set_arr)
                    active_edges = active_edges[~(mask_u_moved | mask_v_moved)]
            
            # ---> NEW: In UMAP mode, only show edges connected to selected nodes <---
            if getattr(cfg, 'UMAP_MODE', False):
                if getattr(self, 'selected_indices', None) and len(self.selected_indices) > 0:
                    mask_u = np.isin(active_edges[:, 0], self.selected_indices)
                    mask_v = np.isin(active_edges[:, 1], self.selected_indices)
                    active_edges = active_edges[mask_u | mask_v]
                else:
                    active_edges = np.zeros((0, 2), dtype=np.int32)
            
            if len(active_edges) > 0:
                self.line_visual.visible = True
                edge_coords = _contiguous_line_positions(
                    self.pos[active_edges].reshape(-1, 2)
                )
                self.line_visual.set_data(pos=edge_coords)
            else:
                self.line_visual.visible = False # Prevents Vispy crash on empty arrays

    def promote_nodes(self, indices):
        """Move one node group to the top of the persistent render order."""
        values = np.asarray(indices)
        if values.dtype == np.bool_:
            if values.ndim != 1 or len(values) != self.n_nodes:
                raise ValueError("Node promotion mask must match the node count.")
            promoted = np.flatnonzero(values)
        else:
            promoted = values.astype(np.int64, copy=False).reshape(-1)

        promoted = np.unique(promoted)
        if promoted.size == 0:
            return False
        if np.any(promoted < 0) or np.any(promoted >= self.n_nodes):
            raise ValueError("Node promotion indices are outside the network.")

        current_order = getattr(self, 'node_render_order', None)
        if current_order is None:
            current_order = np.arange(self.n_nodes, dtype=np.int32)
        else:
            current_order = cache_manifest.validate_node_render_order(
                current_order, self.n_nodes
            )

        keep_mask = ~np.isin(current_order, promoted)
        new_order = np.concatenate(
            (current_order[keep_mask], np.sort(promoted))
        ).astype(np.int32, copy=False)
        changed = not np.array_equal(new_order, current_order)
        self.node_render_order = new_order
        return changed

    def _valid_node_indices(self, values):
        """Return unique in-range node indices as an ascending array."""
        if values is None:
            return np.empty(0, dtype=np.int32)
        array = np.asarray(list(values) if isinstance(values, set) else values)
        if array.size == 0:
            return np.empty(0, dtype=np.int32)
        indices = np.unique(array.astype(np.int64, copy=False).reshape(-1))
        valid = (indices >= 0) & (indices < self.n_nodes)
        return indices[valid].astype(np.int32, copy=False)

    def _selected_node_indices(self):
        """Return command and box-selected node indices."""
        return self._valid_node_indices(getattr(self, 'selected_indices', None))

    def _left_click_node_indices(self):
        """Return ordinary and metadata-driven left-click highlights."""
        highlight_parts = []
        highlighted = self._valid_node_indices(
            getattr(self, 'left_click_highlight_indices', None)
        )
        if len(highlighted):
            highlight_parts.append(highlighted)

        clicked_index = getattr(self, 'selected_node_idx', None)
        if clicked_index is not None:
            highlight_parts.append(np.array([clicked_index], dtype=np.int64))

        if not highlight_parts:
            return np.empty(0, dtype=np.int32)
        return self._valid_node_indices(np.concatenate(highlight_parts))

    def _connected_node_identification_enabled(self, boundary_rgba=None, connected_rgba=None):
        """Return whether connected-node border and ordering work is enabled."""
        import matplotlib.colors as mcolors

        if boundary_rgba is None:
            boundary_rgba = mcolors.to_rgba(
                getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000')
            )
        if connected_rgba is None:
            connected_rgba = mcolors.to_rgba(
                getattr(cfg, 'CONNECTED_NODE_COLOR', '#ff0000')
            )
        return tuple(boundary_rgba) != tuple(connected_rgba)

    def _connected_to_selected_indices(self, selected):
        """Return nodes adjacent to selected nodes in the complete topology."""
        cache_key = (id(self.edges), tuple(np.asarray(selected).tolist()))
        if getattr(self, '_render_order_neighbor_cache_key', None) == cache_key:
            return self._render_order_neighbor_cache.copy()
        if len(selected) == 0 or len(self.edges) == 0:
            connected = np.empty(0, dtype=np.int32)
        else:
            touches_selection = np.isin(self.edges[:, 0], selected) | np.isin(
                self.edges[:, 1], selected
            )
            connected = np.unique(self.edges[touches_selection].reshape(-1))
            connected = connected[~np.isin(connected, selected)]
            connected = connected.astype(np.int32, copy=False)
        self._render_order_neighbor_cache_key = cache_key
        self._render_order_neighbor_cache = connected.copy()
        return connected

    def visible_node_render_order(self, identify_connected=None):
        """Return the effective low-to-high order for currently visible nodes."""
        base_order = getattr(self, 'node_render_order', None)
        if base_order is None:
            base_order = np.arange(self.n_nodes, dtype=np.int32)
        else:
            base_order = np.asarray(base_order, dtype=np.int32)
            if base_order.ndim != 1 or len(base_order) != self.n_nodes:
                raise ValueError("Node render order does not match the node count.")

        visible = np.asarray(self.visible_mask, dtype=bool)
        selected = self._selected_node_indices()
        left_clicked = self._left_click_node_indices()

        selected = selected[visible[selected]]
        left_clicked = left_clicked[visible[left_clicked]]
        selected_for_connections = selected
        if len(left_clicked):
            selected = selected[~np.isin(selected, left_clicked)]

        if identify_connected is None:
            identify_connected = self._connected_node_identification_enabled()
        if identify_connected and len(selected_for_connections):
            connected = self._connected_to_selected_indices(
                selected_for_connections
            )
            connected = connected[visible[connected]]
            if len(left_clicked):
                connected = connected[~np.isin(connected, left_clicked)]
        else:
            connected = np.empty(0, dtype=np.int32)

        elevated = np.zeros(self.n_nodes, dtype=bool)
        elevated[connected] = True
        elevated[selected] = True
        elevated[left_clicked] = True
        remaining = base_order[visible[base_order] & ~elevated[base_order]]
        return np.concatenate(
            (
                remaining,
                np.sort(connected),
                np.sort(selected),
                np.sort(left_clicked),
            )
        ).astype(np.int32, copy=False)

    def update_nodes(self):
        colors = self.current_colors.copy()
        import matplotlib.colors as mcolors
        if getattr(self, 'hovered_node_idx', None) is not None:
            colors[self.hovered_node_idx] = mcolors.to_rgba(cfg.HOVER_COLOR)
            
        sizes = getattr(self, 'current_sizes', cfg.NODE_SIZE)
        shapes = getattr(self, 'current_shapes', np.full(self.n_nodes, 'disc', dtype=object))
        
        # ---> FIXED: Fetch custom boundary color, fallback to black
        bound_rgba = mcolors.to_rgba(getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000'))
        conn_rgba = mcolors.to_rgba(getattr(cfg, 'CONNECTED_NODE_COLOR', '#ff0000'))
        identify_connected = self._connected_node_identification_enabled(
            bound_rgba, conn_rgba
        )
        
        edge_colors = np.zeros((self.n_nodes, 4), dtype=np.float32)
        edge_colors[:] = bound_rgba 
        
        # ---> NEW: Fetch custom boundary width, fallback to 0.5
        b_width = getattr(cfg, 'NODE_BOUNDARY_WIDTH', 0.5)
        edge_widths = np.full(self.n_nodes, b_width, dtype=np.float32)
        
        if getattr(self, 'selected_indices', None) is not None and len(self.selected_indices) > 0:
            hover_rgba = mcolors.to_rgba(cfg.HOVER_COLOR)

            if identify_connected:
                selected = self._selected_node_indices()
                neighbor_indices = self._connected_to_selected_indices(selected)
                if len(neighbor_indices) > 0:
                    edge_colors[neighbor_indices] = conn_rgba
                    edge_widths[neighbor_indices] = 2.0
                
            edge_colors[self.selected_indices] = hover_rgba
            edge_widths[self.selected_indices] = 2.0
        
        # Submit every node attribute in the same effective low-to-high order.
        draw_order = self.visible_node_render_order(
            identify_connected=identify_connected
        )
        self._submitted_visible_node_order = draw_order.copy()
        if len(draw_order) == 0:
            self._submitted_marker_node_order = np.empty(0, dtype=np.int32)
            self._submitted_marker_ring_mask = np.empty(0, dtype=bool)
            self.markers.visible = False
        else:
            self.markers.visible = True

            # A clicked node's enlarged translucent ring and the node itself
            # must share one draw call. Interleave each ring directly before
            # its node; separate VisPy visuals can only be layered as wholes.
            left_clicked = self._left_click_node_indices()
            if len(left_clicked):
                ring_before = np.isin(draw_order, left_clicked)
            else:
                ring_before = np.zeros(len(draw_order), dtype=bool)
            marker_count = len(draw_order) + int(np.count_nonzero(ring_before))
            marker_node_order = np.empty(marker_count, dtype=np.int32)
            marker_ring_mask = np.zeros(marker_count, dtype=bool)

            node_slots = np.arange(len(draw_order)) + np.cumsum(ring_before)
            marker_node_order[node_slots] = draw_order
            ring_slots = node_slots[ring_before] - 1
            marker_node_order[ring_slots] = draw_order[ring_before]
            marker_ring_mask[ring_slots] = True

            self._submitted_marker_node_order = marker_node_order.copy()
            self._submitted_marker_ring_mask = marker_ring_mask.copy()

            marker_face_colors = colors[marker_node_order].copy()
            marker_edge_colors = edge_colors[marker_node_order].copy()
            marker_edge_widths = edge_widths[marker_node_order].copy()
            if isinstance(sizes, np.ndarray):
                marker_sizes = sizes[marker_node_order].copy()
            else:
                marker_sizes = np.full(marker_count, sizes, dtype=np.float32)

            if len(ring_slots):
                marker_face_colors[ring_slots, 3] *= 0.5
                marker_edge_colors[ring_slots] = 0.0
                marker_edge_widths[ring_slots] = 0.0
                marker_sizes[ring_slots] *= 2.0
            
            self.markers.set_data(
                pos=self.pos[marker_node_order],
                face_color=marker_face_colors,
                edge_color=marker_edge_colors,
                size=marker_sizes,
                edge_width=marker_edge_widths,
                symbol=shapes[marker_node_order].tolist()
            )
        self.markers.set_gl_state('translucent', depth_test=False)

        self.canvas.update()
        
        # ---> NEW: Force HUD to instantly sync whenever visual state changes
        self._update_hud_elements()

    def create_hud(self):
        cfg_hud = self.hud_layout
        hud_font_size = self._hud_font_size_points()
        
        self.instr_text = scene.visuals.Text(
            text="[ENTER] Command | [LeftClick] Highlight | [RightClick] Select/Clear | [Scroll] Zoom | [LeftClick + Shift/Ctrl] Copy Node Header/Sequence | [LeftClick + Drag] Pan | [RightClick + Drag] GroupSelect/MoveNodes",
            bold=False, 
            face=self.vispy_ui_face,
            font_size=hud_font_size,
            color='gray', 
            pos=(cfg_hud["instr_x"], cfg_hud["instr_y"]), 
            anchor_y=cfg_hud["instr_anchor_y"], 
            anchor_x=cfg_hud["instr_anchor_x"], 
            parent=self.canvas.scene
        )
        
        self.console_bg = scene.visuals.Rectangle(
            center=(
                cfg_hud["console_bg_left_offset"]
                + cfg_hud["console_bg_min_width"] / 2.0,
                cfg_hud["console_text_y"]
                + cfg_hud.get("console_bg_y_offset", 0.0),
            ),
            width=cfg_hud["console_bg_min_width"],
            height=cfg_hud["console_bg_height"],
            radius=min(
                max(cfg_hud["console_bg_radius"], 0.0),
                cfg_hud["console_bg_min_width"] / 2.0,
                cfg_hud["console_bg_height"] / 2.0,
            ),
            color=(0.95, 0.95, 0.95, 0.95), 
            border_color='black', 
            parent=self.canvas.scene
        )
        self.console_bg.visible = False

        self._console_logical_text = ""
        self._console_rendered_text = ""
        self.console_text = scene.visuals.Text(
            text="", 
            bold=True, 
            face=self.vispy_monospace_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR, 
            pos=(cfg_hud["console_text_x"], cfg_hud["console_text_y"]), 
            anchor_y=cfg_hud["console_text_anchor_y"], 
            anchor_x=cfg_hud["console_text_anchor_x"], 
            parent=self.canvas.scene
        )

        self._background_job_status_logical_text = ""
        self._background_job_status_rendered_text = ""
        self.background_job_status_text = scene.visuals.Text(
            text="",
            bold=True,
            face=self.vispy_monospace_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR,
            pos=(cfg_hud["console_text_x"], cfg_hud["console_text_y"]),
            anchor_y=cfg_hud["console_text_anchor_y"],
            anchor_x=cfg_hud["console_text_anchor_x"],
            parent=self.canvas.scene,
        )
        self.background_job_status_text.visible = False
        
        self.zoom_text = scene.visuals.Text(
            text="", 
            bold=False, 
            face=self.vispy_ui_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR,
            pos=self._status_hud_position(1),
            anchor_y=cfg_hud["status_anchor_y"],
            anchor_x=cfg_hud["status_anchor_x"],
            parent=self.canvas.scene
        )
        
        self.hidden_text = scene.visuals.Text(
            text="", 
            bold=False, 
            face=self.vispy_ui_face,
            font_size=hud_font_size,
            color=cfg.TEXT_COLOR,
            pos=self._status_hud_position(0),
            anchor_y=cfg_hud["status_anchor_y"],
            anchor_x=cfg_hud["status_anchor_x"],
            parent=self.canvas.scene
        )

    def process_command(self, cmd_str, record_history=True, silent=False):
        cmd_str = cmd_str.strip()
        if not cmd_str: return

        # Normalize reverse commands (e.g., 'color reset' -> 'reset color', 'help color' -> 'color help')
        

        # --- 0. FILE-BACKED HISTORY ---
        # Only record if it's different from the very last command typed
        if record_history:
            if not self.command_history or self.command_history[-1] != cmd_str:
                self.command_history.append(cmd_str)
                try:
                    os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
                    with open(self.history_file, "a", encoding="utf-8") as f:
                        f.write(cmd_str + "\n")
                    
                    # Truncate if file exceeds 1 MB (1,048,576 bytes)
                    if os.path.getsize(self.history_file) > 1048576:
                        # Keep latest ~2000 lines (safely under 1MB limit for string paths)
                        self.command_history = self.command_history[-2000:]
                        with open(self.history_file, "w", encoding="utf-8") as f:
                            for line in self.command_history:
                                f.write(line + "\n")
                except Exception as e:
                    print(f"Warning: Failed to save history to {self.history_file} ({e})")

        # --- 3. PARSE COMMAND ---
        parts = cmd_str.split()
        if not parts: return
        
        command_name = parts[0].lower()
        args = parts[1:]

        # --- 6. DYNAMIC EXTERNAL COMMANDS ---
        try:
            module = importlib.import_module(f"commands.{command_name}")
            importlib.reload(module) 
            
            if hasattr(module, 'run'):
                if not silent and hasattr(self, 'console_text'):
                    self.console_text.text = f"Running {command_name}..."
                if not silent and hasattr(self, 'update_console_background'):
                    self.update_console_background()
                if hasattr(app, 'process_events'):
                    app.process_events() 
                module.run(self, args)
                if not silent and hasattr(self, 'update_console_background'):
                    self.update_console_background()
                # Broadcast a complete browser state, including metadata shape.
                self.broadcast_metadata_state()
            else:
                if not silent and hasattr(self, 'console_text'):
                    self.console_text.text = f"Error: No 'run' in {command_name}"
                if not silent and hasattr(self, 'update_console_background'):
                    self.update_console_background()
                
        except ModuleNotFoundError:
            if not silent and hasattr(self, 'console_text'):
                self.console_text.text = f"Unknown command: {command_name}"
            if not silent and hasattr(self, 'update_console_background'):
                self.update_console_background()
        except Exception as e:
            if not silent and hasattr(self, 'console_text'):
                self.console_text.text = f"Error: {e}"
            if not silent and hasattr(self, 'update_console_background'):
                self.update_console_background()
            print(f"Command Error: {e}")
            import traceback
            traceback.print_exc()

    def on_key_press(self, event):
        # --- SAFE KEY DETECTION ---
        # Vispy sometimes fails to map hardware toggles (like CapsLock, NumLock) 
        # and passes None. We must ignore these to prevent attribute errors.
        if getattr(event, 'key', None) is None:
            return

        # --- NEW: Global Escape Interceptor ---
        if event.key == 'Escape':
            event.handled = True  # Block Vispy from closing the window
            if self.console_mode:
                self.console_mode = False
                self.console_bg.visible = False
                self.console_text.text = ""
                self.canvas.update()
            return

        # --- Console Typing Mode ---
        if self.console_mode:
            event.handled = True 
            
            # Safely extract the key name as a string
            key_name = getattr(event.key, 'name', '') or ''

            # Detect modifiers for Copy/Paste safely
            is_modifier_active = 'Control' in event.modifiers or 'Meta' in event.modifiers
            is_paste = (key_name.lower() == 'v') and is_modifier_active
            is_copy = (key_name.lower() == 'c') and is_modifier_active

            if event.key in ['Enter', 'Return']:
                self.process_command(self.input_buffer)
                self.console_mode = False; self.console_bg.visible = False; self.canvas.update()
                
            elif event.key == 'Backspace':
                if self.cursor_pos > 0:
                    self.input_buffer = self.input_buffer[:self.cursor_pos-1] + self.input_buffer[self.cursor_pos:]
                    self.cursor_pos -= 1
                    self._update_console_text()
                    
            elif event.key == 'Delete':
                if self.cursor_pos < len(self.input_buffer):
                    self.input_buffer = self.input_buffer[:self.cursor_pos] + self.input_buffer[self.cursor_pos+1:]
                    self._update_console_text()
                    
            elif event.key == 'Left':
                self.cursor_pos = max(0, self.cursor_pos - 1)
                self._update_console_text()
                
            elif event.key == 'Right':
                self.cursor_pos = min(len(self.input_buffer), self.cursor_pos + 1)
                self._update_console_text()
                
            elif event.key == 'Up':
                if hasattr(self, 'command_history') and self.command_history:
                    self.history_index = max(0, self.history_index - 1)
                    self.input_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.input_buffer)
                    self._update_console_text()
                    
            elif event.key == 'Down':
                if hasattr(self, 'command_history') and self.command_history:
                    self.history_index = min(len(self.command_history), self.history_index + 1)
                    if self.history_index == len(self.command_history):
                        self.input_buffer = ""
                    else:
                        self.input_buffer = self.command_history[self.history_index]
                    self.cursor_pos = len(self.input_buffer)
                    self._update_console_text()
                    
            elif is_paste:
                try:
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    
                    cb_text = native_app.clipboard().text()
                    if cb_text:
                        cb_text = cb_text.replace('\n', ' ').replace('\r', '') 
                        self.input_buffer = self.input_buffer[:self.cursor_pos] + cb_text + self.input_buffer[self.cursor_pos:]
                        self.cursor_pos += len(cb_text)
                        self._update_console_text()
                except Exception as e:
                    print(f"Paste failed: {e}")
                    
            elif is_copy:
                try:
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    
                    native_app.clipboard().setText(self.input_buffer)
                    print("Copied command to clipboard.")
                except Exception as e:
                    print(f"Copy failed: {e}")
                    
            elif len(event.text) > 0 and not is_modifier_active and event.key not in ['Shift', 'Alt']:
                self.input_buffer = self.input_buffer[:self.cursor_pos] + event.text + self.input_buffer[self.cursor_pos:]
                self.cursor_pos += len(event.text)
                self._update_console_text()
                
        # --- Opening the Console (and hotkeys) ---
        else:
            key_name = getattr(event.key, 'name', '') or ''
            is_modifier_active = 'Control' in event.modifiers or 'Meta' in event.modifiers
            
            # Handle Undo / Redo
            if (key_name.lower() == 'z') and is_modifier_active:
                self._do_undo()
                event.handled = True
                return
            if (key_name.lower() == 'y') and is_modifier_active:
                self._do_redo()
                event.handled = True
                return

            if event.key in ['Enter', 'Return']:
                self.clear_background_job_status()
                self.console_mode = True
                self.input_buffer = ""
                self.cursor_pos = 0
                
                if hasattr(self, 'command_history'):
                    self.history_index = len(self.command_history)
                else:
                    self.history_index = 0
                    
                self.console_bg.visible = True
                self._update_console_text()
                event.handled = True

    def on_mouse_press(self, event):
        # ---> 1. RIGHT-CLICK LOGIC (Drag & Select) <---
        if event.button == 2 and not self.console_mode:
            self.tooltip.text = "" 
            self.selected_node_idx = None
            
            # Clear or hide registered HUD displays
            for display in self.hud_displays.values():
                if getattr(display, 'on_right_click', None):
                    display.on_right_click()
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            
            nearest_idx = _topmost_nearest_visible_node_index(
                self.pos,
                self.visible_mask,
                mouse_world,
                getattr(self, '_submitted_visible_node_order', None),
            )
            screen_dist = np.inf
            if nearest_idx is not None:
                node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
                screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            is_node_clicked = screen_dist < cfg.NODE_SIZE
            self.drag_start_mouse = mouse_world
            self.drag_start_screen = event.pos
            
            if is_node_clicked:
                self._save_state()

                # Modifier logic for individual node clicking
                if 'Shift' in event.modifiers:
                    if nearest_idx not in self.selected_indices:
                        self.selected_indices.append(nearest_idx)
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    if nearest_idx in self.selected_indices:
                        self.selected_indices.remove(nearest_idx)
                else:
                    if nearest_idx not in self.selected_indices:
                        self.selected_indices = [nearest_idx]
                        
                self.update_selection_visual()
                
                if nearest_idx in self.selected_indices:
                    if not getattr(cfg, 'UMAP_MODE', False):
                        self.is_multi_dragging = True
                        self._drag_edges_hidden = False
                        self.drag_start_nodes_pos = self.pos[self.selected_indices, :2].copy()
            else:
                # Clicked empty space: Store the current state, DO NOT clear yet
                self._pre_drag_selection = set(self.selected_indices)
                self.is_box_selecting = True
                self.selection_box.set_data(
                    pos=np.zeros((5, 2), dtype=np.float32),
                )
                self.selection_box.visible = True
                
            event.handled = True 
            return
            
        # ---> 2. SHIFT + LEFT-CLICK LOGIC (Copy to Clipboard) <---
        # Vispy Button 1 = Left Click
        if event.button == 1 and 'Shift' in event.modifiers and not self.console_mode:
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            
            nearest_idx = _topmost_nearest_visible_node_index(
                self.pos,
                self.visible_mask,
                mouse_world,
                getattr(self, '_submitted_visible_node_order', None),
            )
            screen_dist = np.inf
            if nearest_idx is not None:
                node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
                screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            if screen_dist < (cfg.NODE_SIZE / 1.5):
                full_header = self.full_headers[nearest_idx]
                try:
                    # Use Vispy's native app instance, just like in on_key_press
                    from vispy import app as vispy_app
                    native_app = vispy_app.use_app().native
                    native_app.clipboard().setText(full_header)
                    
                    self.console_text.text = f"Copied: {full_header}"
                    print(f"Copied to clipboard: {full_header}")
                except Exception as e:
                    self.console_text.text = f"Copy Failed: {full_header}"
                    print(f"Clipboard Error: {e}")

                self.update_console_background()
                
            event.handled = True
            return
        
        # ---> 2.5 CONTROL + LEFT-CLICK LOGIC (Copy Sequence to Clipboard) <---
        # Vispy Button 1 = Left Click
        if event.button == 1 and ('Control' in event.modifiers or 'Meta' in event.modifiers) and not self.console_mode:
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            
            nearest_idx = _topmost_nearest_visible_node_index(
                self.pos,
                self.visible_mask,
                mouse_world,
                getattr(self, '_submitted_visible_node_order', None),
            )
            screen_dist = np.inf
            if nearest_idx is not None:
                node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
                screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            if screen_dist < (cfg.NODE_SIZE / 1.5):
                full_header = self.full_headers[nearest_idx]
                rec_id = full_header.split()[0]
                
                # Lazy-load sequence map if not already loaded
                if not hasattr(self, 'sequences_map'):
                    fasta_path = getattr(cfg, 'NODE_FASTA_FILE', None) or getattr(cfg, 'SEQUENCES_FILE', '')
                    if fasta_path and os.path.exists(fasta_path):
                        try:
                            records = _load_selected_fasta_records(fasta_path)
                            self.sequences_map = _build_sequence_lookup(records)
                        except Exception as e:
                            self.sequences_map = {}
                            print(f"Warning: Failed to parse FASTA for sequences: {e}")
                    else:
                        self.sequences_map = {}
                
                # Look up sequence
                sequence = None
                if full_header in self.sequences_map:
                    sequence = self.sequences_map[full_header]
                elif rec_id in self.sequences_map:
                    sequence = self.sequences_map[rec_id]
                
                if sequence:
                    try:
                        from vispy import app as vispy_app
                        native_app = vispy_app.use_app().native
                        native_app.clipboard().setText(sequence)
                        self.console_text.text = f"Copied sequence of: {rec_id}"
                        print(f"Copied sequence to clipboard: {rec_id} ({len(sequence)} aa)")
                    except Exception as e:
                        self.console_text.text = f"Copy Failed: {rec_id}"
                        print(f"Clipboard Error: {e}")
                else:
                    self.console_text.text = f"Sequence not found for: {rec_id}"
                    print(f"Sequence not found in FASTA for: {rec_id}")

                self.update_console_background()
                    
            event.handled = True
            return
        
        # ---> 3. PLAIN LEFT-CLICK LOGIC (Show Highlight) <---
        if event.button == 1 and 'Shift' not in event.modifiers and not self.console_mode:
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            
            nearest_idx = _topmost_nearest_visible_node_index(
                self.pos,
                self.visible_mask,
                mouse_world,
                getattr(self, '_submitted_visible_node_order', None),
            )
            screen_dist = np.inf
            if nearest_idx is not None:
                node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
                screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            # If clicked within the node's radius, apply the shared left-click focus.
            if screen_dist < cfg.NODE_SIZE:
                self.apply_left_click_focus(nearest_idx)
            else:
                # Clicking empty space clears the left-click highlight
                self.clear_left_click_focus()
                
            event.handled = True
            return

    def apply_left_click_focus(self, node_idx):
        """Apply the shared temporary focus used by SSN and metadata clicks."""
        node_idx = int(node_idx)
        self.selected_node_idx = node_idx
        self.left_click_highlight_indices = None
        if getattr(self, 'tooltip', None) is not None:
            self.tooltip.text = ""

        label = ""
        if getattr(self, 'cluster_labels', None) is not None:
            cluster_id = self.cluster_labels[node_idx]
            label = "[Noise] " if cluster_id == -1 else f"[Cluster {cluster_id}] "
        label += self.full_headers[node_idx]

        group_suffix = ""
        if getattr(self, 'group_labels', None) and self.group_labels[node_idx]:
            group_text = ", ".join(sorted(self.group_labels[node_idx]))
            group_suffix = f" [Groups: {group_text}]"

        print(f"Node Selected: {self.full_headers[node_idx]}")
        if getattr(self, 'console_text', None) is not None:
            self.console_text.text = f"Selected: {label}{group_suffix}"

        if hasattr(self, 'sync_metadata_table_selection'):
            self.sync_metadata_table_selection(node_idx)

        for display in getattr(self, 'hud_displays', {}).values():
            if getattr(display, 'on_node_clicked', None):
                display.on_node_clicked(node_idx)

        self.update_nodes()
        self.broadcast_event({"type": "highlight_row", "index": node_idx})

    def clear_left_click_focus(self):
        """Clear temporary click focus without changing command selection."""
        self.selected_node_idx = None
        self.left_click_highlight_indices = None
        if getattr(self, 'tooltip', None) is not None:
            self.tooltip.text = ""
        self.update_nodes()
        self.broadcast_event({"type": "highlight_row", "index": None})
    
    def _update_hud_elements(self, event=None):
        """Updates the zoom indicator, hidden nodes count, and maintains the tooltip pixel gap."""
        panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
        panel_w = getattr(self, '_panel_w', 120) if panel_visible else 0
        effective_canvas_w = self.canvas.size[0] - panel_w
        effective_size = (effective_canvas_w, self.canvas.size[1])
        
        # 1. Update Zoom Indicator (Visible World Width)
        if hasattr(self, 'zoom_text'):
            visible_width = self.view.camera.rect.width
            self.zoom_text.text = f"View Width: {visible_width:.1f}"
            self.zoom_text.color = cfg.TEXT_COLOR
            self.zoom_text.pos = self._status_hud_position(1, effective_size)

        # 2. Update Hidden Nodes Indicator
        if hasattr(self, 'hidden_text') and hasattr(self, 'visible_mask'):
            hidden_count = int(np.sum(~self.visible_mask))
            self.hidden_text.text = f"Hidden Nodes: {hidden_count}"
            if hidden_count > 0:
                self.hidden_text.color = 'red'
            else:
                self.hidden_text.color = cfg.TEXT_COLOR
            self.hidden_text.pos = self._status_hud_position(0, effective_size)

        # 3. Update Tooltip Distance
        if getattr(self, 'selected_node_idx', None) is not None and getattr(self, 'tooltip', None) and self.tooltip.text != "":
            tr = self.canvas.scene.node_transform(self.view.scene)
            screen_pos = tr.inverse.map(self.pos[self.selected_node_idx])

            self.tooltip.pos = screen_pos[:2] + [15, -15]

        # Keep the one-line console value visually wrapped as the canvas or
        # sidebar width changes.
        self.update_console_background()

        # 4. Update any registered HUD displays
        for display in self.hud_displays.values():
            if getattr(display, 'update_position', None):
                display.update_position()
            
        self.canvas.update()

    def _initialize_display_tracking(self):
        """Bind DPI notifications after the canvas has its final top-level window."""
        main_window = getattr(self, 'main_window', None)
        window_handle = main_window.windowHandle() if main_window is not None else None

        if window_handle is not getattr(self, '_display_window_handle', None):
            previous_handle = getattr(self, '_display_window_handle', None)
            if previous_handle is not None:
                try:
                    previous_handle.screenChanged.disconnect(self._on_display_screen_changed)
                except (RuntimeError, TypeError):
                    pass

            self._display_window_handle = window_handle
            if window_handle is not None:
                window_handle.screenChanged.connect(self._on_display_screen_changed)

        screen = window_handle.screen() if window_handle is not None else None
        self._bind_active_screen(screen)
        self._schedule_display_refresh()

    def _bind_active_screen(self, screen):
        """Follow DPI and geometry changes from only the window's active screen."""
        signal_bindings = getattr(self, '_active_screen_signal_bindings', [])
        if screen is getattr(self, '_active_screen', None) and signal_bindings:
            return

        for signal, callback in signal_bindings:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass

        self._active_screen_signal_bindings = []
        self._active_screen = screen
        if screen is None:
            return

        callback = self._on_active_screen_metrics_changed
        for signal_name in (
            'logicalDotsPerInchChanged',
            'physicalDotsPerInchChanged',
            'geometryChanged',
        ):
            signal = getattr(screen, signal_name, None)
            if signal is None:
                continue
            signal.connect(callback)
            self._active_screen_signal_bindings.append((signal, callback))

    def _on_display_screen_changed(self, screen):
        self._bind_active_screen(screen)
        self._schedule_display_refresh()

    def _on_active_screen_metrics_changed(self, *args):
        self._schedule_display_refresh()

    def _schedule_display_refresh(self):
        """Debounce display notifications until Qt's GL transition settles."""
        if getattr(self, '_display_refresh_pending', False):
            return
        self._display_refresh_pending = True
        QtCore.QTimer.singleShot(100, self._refresh_display_layout)

    def _refresh_display_layout(self):
        """Refresh logical overlays after Qt/VisPy update the framebuffer."""
        self._display_refresh_pending = False
        canvas = getattr(self, 'canvas', None)

        # QOpenGLWidget owns resizeGL and invokes it with a current context.
        # VisPy's Qt backend is already connected to the native window's
        # screenChanged signal. Calling either method manually here happens
        # outside paintGL and can race context migration between monitors.

        for method_name in (
            '_apply_vispy_text_scaling',
            'position_slider_overlay',
            'reposition_expand_btn',
            '_update_hud_elements',
        ):
            method = getattr(self, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception as e:
                print(f"Warning: Could not refresh {method_name}: {e}")

        if canvas is not None:
            try:
                canvas.update()
            except Exception as e:
                print(f"Warning: Could not redraw canvas after display refresh: {e}")

        self._print_display_diagnostics()

    def _print_display_diagnostics(self):
        """Print current logical/physical display state only when it changes."""
        canvas = getattr(self, 'canvas', None)
        if canvas is None:
            return

        screen = getattr(self, '_active_screen', None)
        native_canvas = getattr(canvas, 'native', None)
        try:
            screen_name = screen.name() if screen is not None else "Unknown display"
            geometry = screen.geometry() if screen is not None else None
            screen_geometry = (
                geometry.x(),
                geometry.y(),
                geometry.width(),
                geometry.height(),
            ) if geometry is not None else (0, 0, 0, 0)
            dpr = (
                float(native_canvas.devicePixelRatioF())
                if native_canvas is not None and hasattr(native_canvas, 'devicePixelRatioF')
                else float(getattr(canvas, 'pixel_scale', 1.0))
            )
            logical_size = tuple(canvas.size)
            physical_size = tuple(canvas.physical_size)
            qapp = QtWidgets.QApplication.instance()
            qt_platform = qapp.platformName() if qapp is not None else "unknown"
        except Exception as e:
            print(f"Warning: Could not read display information: {e}")
            return

        signature = (
            screen_name,
            screen_geometry,
            round(dpr, 6),
            logical_size,
            physical_size,
            qt_platform,
        )
        if signature == getattr(self, '_last_display_signature', None):
            return
        self._last_display_signature = signature

        print(
            "Display: "
            f"{screen_name} | geometry={screen_geometry[2]}x{screen_geometry[3]} "
            f"at ({screen_geometry[0]}, {screen_geometry[1]}) | DPR={dpr:g} | "
            f"Qt platform={qt_platform} | "
            f"canvas logical={logical_size[0]}x{logical_size[1]} | "
            f"physical={physical_size[0]}x{physical_size[1]}"
        )

    def on_resize(self, event): 
        # Resize events already run after VisPy has updated the canvas size and
        # scene transforms. Refresh the HUD synchronously so pixel-anchored
        # text follows the window instead of waiting for a Qt-backed timer,
        # which can be starved during interactive window resizing.
        self._update_hud_elements()
        if hasattr(self, 'slider_overlay'):
            self.position_slider_overlay()
        if hasattr(self, 'reposition_expand_btn'):
            self.reposition_expand_btn()

    def reposition_expand_btn(self):
        if hasattr(self, 'canvas'):
            w, h = self.canvas.size
            panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
            panel_w = getattr(self, '_panel_w', 120)
            if hasattr(self, 'right_panel'):
                self.right_panel.setGeometry(w - panel_w, 0, panel_w, h)
            if hasattr(self, 'toggle_sidebar_btn'):
                if panel_visible:
                    self.toggle_sidebar_btn.setGeometry(w - panel_w - 40, 10, 30, 30)
                else:
                    self.toggle_sidebar_btn.setGeometry(w - 40, 10, 30, 30)

    def toggle_sidebar(self):
        if hasattr(self, 'right_panel'):
            visible = not self.right_panel.isVisible()
            self.set_sidebar_visible(visible)

    def set_sidebar_visible(self, visible):
        has_buttons = bool(getattr(self, 'sidebar_buttons', {}))
        if hasattr(self, 'right_panel'):
            if not has_buttons:
                self.right_panel.hide()
                if hasattr(self, 'toggle_sidebar_btn'):
                    self.toggle_sidebar_btn.hide()
                self.reposition_expand_btn()
                return

            self.right_panel.setVisible(visible)
            if hasattr(self, 'toggle_sidebar_btn'):
                self.toggle_sidebar_btn.setVisible(True)
                self.toggle_sidebar_btn.setText(">>" if visible else "<<")
            self.reposition_expand_btn()

            # Immediately update the positions of HUD labels and slider
            if hasattr(self, 'position_slider_overlay'):
                self.position_slider_overlay()
            if hasattr(self, '_update_hud_elements'):
                self._update_hud_elements()

    def open_metadata_ui(self):
        return self._open_web_ui("/meta.html", "Metadata UI", "meta")

    def open_agent_ui(self):
        return self._open_web_ui("/agent.html", "Agent UI", "agent")

    def _open_web_ui(
        self,
        path,
        label,
        client_id,
        *,
        show_existing_dialog=True,
    ):
        return open_browser_page(
            self,
            path,
            label,
            client_id,
            show_existing_dialog=show_existing_dialog,
        )

    def get_web_url(self, path="/"):
        """Return a URL served by this Viewer instance."""
        if not self.web_server_url:
            raise RuntimeError("This Viewer instance's web server is unavailable.")
        return f"{self.web_server_url}/{str(path).lstrip('/')}"

    def add_sidebar_button(self, name, label, callback, tooltip=None):
        if not hasattr(self, 'sidebar_buttons'):
            self.sidebar_buttons = {}
        
        # If button already exists, just show it and expand the sidebar
        if name in self.sidebar_buttons:
            self.sidebar_buttons[name].show()
            self.set_sidebar_visible(True)
            return self.sidebar_buttons[name]
        
        btn = QtWidgets.QPushButton(label, self.right_panel)
        btn.setObjectName(name)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setFixedWidth(150)
        btn.setFixedHeight(35)
        btn.clicked.connect(callback)
        
        # Insert button in layout right before the bottom stretch spacer
        layout = self.right_panel_layout
        layout.insertWidget(layout.count() - 1, btn)
        
        self.sidebar_buttons[name] = btn
        self.set_sidebar_visible(True)
        return btn


    def start_web_server(self):
        try:
            from web_ui import Web_Server
            self.web_server = Web_Server.start_server(self)
            port = int(self.web_server.server_address[1])
            self.web_server_url = f"http://localhost:{port}"
            print(f"WebServer started at {self.web_server_url}")
        except Exception as e:
            self.web_server = None
            self.web_server_url = None
            print(f"Error starting WebServer: {e}")

    def broadcast_event(self, event):
        if hasattr(self, 'web_server') and self.web_server:
            with self.web_server.queues_lock:
                queues = list(self.web_server.event_queues)
            for q in queues:
                q.put(event)

    def broadcast_metadata_state(self):
        """Broadcast metadata rows and schema as one authoritative state."""
        self.broadcast_event({
            "type": "state_updated",
            "visible_mask": self.visible_mask.tolist(),
            "selected_indices": self.selected_indices,
            "metadata": self.get_serializable_metadata(),
            "columns": ["Node ID"] + list(self.metadata.keys()),
            "types": {
                key: entry["type"] for key, entry in self.metadata.items()
            },
        })

    def handle_web_action(self, data):
        action = data.get("action")
        if action in self.web_action_handlers:
            self.web_action_handlers[action](data)
        else:
            print(f"Warning: No handler registered for web action '{action}'")

    def get_serializable_metadata(self):
        rows = []
        for row_idx in range(self.n_nodes):
            row_dict = {
                "id": row_idx,
                "Node ID": str(self.full_headers[row_idx])
            }
            for key, entry in self.metadata.items():
                val = entry["values"][row_idx]
                if isinstance(val, (float, np.floating)) and np.isnan(val):
                    val = ""
                else:
                    val = val.item() if hasattr(val, 'item') else val
                row_dict[key] = val
            rows.append(row_dict)
        return rows

    def get_initial_web_state(self):
        state = {
            "rows": self.get_serializable_metadata(),
            "columns": ["Node ID"] + [k for k in self.metadata.keys() if k.lower() != "length"],
            "selected_indices": self.selected_indices,
            "visible_mask": self.visible_mask.tolist(),
            "llm_loaded": getattr(self, 'llm_loaded', False),
            "llm_backend": getattr(self, 'llm_backend', None),
            "llm_model_name": getattr(self, 'llm_model_name', "Unknown")
        }
        registry = getattr(self, "web_plugin_registry", None)
        if registry is not None:
            return registry.apply_state_providers(state)
        return state

    def on_mouse_wheel(self, event):
        self._hud_timer.start()

        # ---> Rotate selected nodes if right-click dragging <---
        if getattr(self, 'is_multi_dragging', False) and 2 in event.buttons:
            # event.delta[1] is > 0 for scroll up, < 0 for scroll down
            angle = event.delta[1] * (np.pi / 36) # 5 degrees per tick
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            center = self.drag_start_mouse
            
            pts = self.drag_start_nodes_pos - center
            rotated_x = pts[:, 0] * cos_a - pts[:, 1] * sin_a
            rotated_y = pts[:, 0] * sin_a + pts[:, 1] * cos_a
            self.drag_start_nodes_pos[:, 0] = rotated_x + center[0]
            self.drag_start_nodes_pos[:, 1] = rotated_y + center[1]
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            delta = mouse_world - self.drag_start_mouse
            self.pos[self.selected_indices, :2] = self.drag_start_nodes_pos + delta
            
            if getattr(cfg, 'LOW_RESOURCE_MODE', False):
                if not getattr(self, '_drag_edges_hidden', False):
                    self._drag_edges_hidden = True
                    self.update_edges()
                self.update_nodes()
            else:
                self.update_selection_visual()
            
            event.handled = True
            return

    def on_mouse_move(self, event):
        tr = self.canvas.scene.node_transform(self.view.scene)
        mouse_world = tr.map(event.pos)[:2]

        # ---> 0. FAILSAFE: CATCH MISSED MOUSE RELEASES <---
        # If we are dragging or boxing, but the right button is NO LONGER held down:
        if getattr(self, 'is_multi_dragging', False) or getattr(self, 'is_box_selecting', False):
            if 2 not in event.buttons:
                self.on_mouse_release(event)
                return

        # ---> 1. MULTI-NODE DRAGGING <---
        if getattr(self, 'is_multi_dragging', False):
            delta = mouse_world - self.drag_start_mouse
            self.pos[self.selected_indices, :2] = self.drag_start_nodes_pos + delta
            
            if getattr(cfg, 'LOW_RESOURCE_MODE', False):
                if not getattr(self, '_drag_edges_hidden', False):
                    self._drag_edges_hidden = True
                    self.update_edges()
                self.update_nodes()
            else:
                self.update_selection_visual()
            
            event.handled = True
            return
        # ---> 2. BOX SELECTION DRAWING <---
        if getattr(self, 'is_box_selecting', False):
            x0, y0 = self.drag_start_mouse
            x1, y1 = mouse_world
            
            rect_pts = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]
            ], dtype=np.float32)
            self.selection_box.set_data(
                pos=rect_pts,
            )
            
            # --- Dynamic Highlighting Math ---
            if not getattr(cfg, 'LOW_RESOURCE_MODE', False):
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                xs = self.pos[:, 0]
                ys = self.pos[:, 1]
                mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y) & self.visible_mask
                box_indices = set(np.where(mask)[0].tolist())
                
                pre_drag = getattr(self, '_pre_drag_selection', set())
                
                if 'Shift' in event.modifiers:
                    current_selection = pre_drag.union(box_indices)     # Add
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    current_selection = pre_drag.difference(box_indices)# Remove
                else:
                    current_selection = box_indices                     # Replace
                    
                self.selected_indices = list(current_selection)
                self.update_selection_visual()
            
            event.handled = True
            return
        # ---> 3. PANNING HUD OVERLAY <---
        # Update if we are panning (Left Click Drag only)
        if 1 in event.buttons:
            self._hud_timer.start()

        # ---> 4. HOVER EFFECT <---
        if not self.console_mode and not event.buttons: 
            nearest_idx = _topmost_nearest_visible_node_index(
                self.pos,
                self.visible_mask,
                mouse_world,
                getattr(self, '_submitted_visible_node_order', None),
            )
            screen_dist = np.inf
            if nearest_idx is not None:
                node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
                screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            if screen_dist < (cfg.NODE_SIZE / 1.5):
                if getattr(self, 'hovered_node_idx', None) != nearest_idx:
                    self.hovered_node_idx = nearest_idx
                    self.update_nodes()
            else:
                if getattr(self, 'hovered_node_idx', None) is not None:
                    self.hovered_node_idx = None
                    self.update_nodes()
    
    def on_mouse_release(self, event):
        # ---> 1. MULTI-DRAG RELEASE <---
        if getattr(self, 'is_multi_dragging', False):
            self.is_multi_dragging = False
            self._drag_edges_hidden = False
            self.update_selection_visual()
            event.handled = True
            return
            
        # ---> 2. FINALIZE BOX SELECTION <---
        if getattr(self, 'is_box_selecting', False):
            self.is_box_selecting = False
            self.selection_box.visible = False
            
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            x0, y0 = self.drag_start_mouse
            x1, y1 = mouse_world
            
            # Calculate drag distance in screen coordinates (pixels)
            drag_start_screen = getattr(self, 'drag_start_screen', None)
            if drag_start_screen is None:
                drag_start_screen = event.pos
            dx = event.pos[0] - drag_start_screen[0]
            dy = event.pos[1] - drag_start_screen[1]
            drag_dist_screen = np.hypot(dx, dy)
            
            if getattr(cfg, 'LOW_RESOURCE_MODE', False) and drag_dist_screen >= 5.0:
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                xs = self.pos[:, 0]
                ys = self.pos[:, 1]
                mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y) & self.visible_mask
                box_indices = set(np.where(mask)[0].tolist())
                
                pre_drag = getattr(self, '_pre_drag_selection', set())
                
                if 'Shift' in event.modifiers:
                    current_selection = pre_drag.union(box_indices)     # Add
                elif 'Control' in event.modifiers or 'Meta' in event.modifiers:
                    current_selection = pre_drag.difference(box_indices)# Remove
                else:
                    current_selection = box_indices                     # Replace
                    
                self.selected_indices = list(current_selection)
                self.update_selection_visual()
                
            # Single click detection (no drag distance in screen pixels)
            if drag_dist_screen < 5.0:
                if 'Shift' not in event.modifiers and 'Control' not in event.modifiers and 'Meta' not in event.modifiers:
                    self.selected_indices = []
                    self.update_selection_visual()

            # Output final count to console
            if len(self.selected_indices) > 0:
                self.console_text.text = f"Selected {len(self.selected_indices)} nodes."
            else:
                self.console_text.text = "Selection cleared."
                
            event.handled = True
            return
        
if __name__ == '__main__':
    # Set the identity before VisPy creates QApplication or any native window.
    configure_linux_qt_desktop_identity(
        QtWidgets.QApplication, VIEWER_DESKTOP_FILE_NAME
    )
    viewer = MainViewer()
    app.run()
