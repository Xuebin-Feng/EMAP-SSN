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

import unicodedata  # Pre-load to prevent Windows DLL search path conflicts with Qt/OpenGL
try:
    import torch  # Pre-load to prevent DLL initialization conflicts between PyTorch and PySide6/OpenGL
except Exception:
    pass
import sys
import os
import json
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
import SSN_Utils as utils
import Command_Engine
import Cache_Manifest as cache_manifest
from utilities.FASTA_Sanitization import (
    load_sanitized_fasta,
    write_fasta_atomic,
)
from utilities.Application_Fonts import (
    VISPY_FALLBACK_FACE,
    configure_qt_application_fonts,
    register_vispy_application_fonts,
)


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
                font_size=8,
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
        self.web_action_handlers = {}
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
            "console_font_family": "Open Sans", # Font family used to measure text width
            "console_font_size": 8,      # Font size used to measure text width
            
            # 3. Command Line Background Box
            "console_bg_height": 40.0,   # Vertical height of the background box
            "console_bg_min_width": 250.0, # Minimum width of the background box when empty/short
            "console_bg_left_offset": 20.0, # Fixed horizontal left position of the box
            "console_bg_y_offset": 22.0, # Positive values move only the box downward
            "console_bg_radius": 10.0,    # Radius for the rounded corners (0.0 for sharp corners)
            "console_bg_padding_x": 20.0, # Fixed logical padding added to the end of the command box
            
            # 4. Zoom Indicator Text (" View Width: XXX ") at bottom-right
            "zoom_x_offset": 10.0,       # Distance from the right edge of the window
            "zoom_y_offset": 55.0,       # Distance from the bottom edge of the window (original default)
            "zoom_anchor_x": "right",
            "zoom_anchor_y": "bottom",
            
            # 5. Hidden Nodes Indicator Text (" Hidden Nodes: X ") at bottom-right
            "hidden_x_offset": 10.0,     # Distance from the right edge of the window
            "hidden_y_offset": 30.0,     # Distance from the bottom edge of the window (original default)
            "hidden_anchor_x": "right",
            "hidden_anchor_y": "bottom"
        }
        
        self.input_buffer = ""
        self.cursor_pos = 0       # Tracks cursor position
        
        # ---> Persistent Command History (Per Layout) <---
        self.command_history = []
        try:
            cache_path, _ = utils.get_cache_filename()
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
                utils.force_light_palette(qapp)
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
                color: gray;
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
        """ % {"font": utils.UI_FONT_STACK})
        
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
                color: #555555;
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
                color: #0969da;
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
        """ % {"font": utils.UI_FONT_STACK})
        
        # Initialize thread-safe QtCommunicator for server commands
        from web_ui import Web_Server
        self.communicator = Web_Server.QtCommunicator(self)
        
        # Initialize background WebServer
        self.web_server = None
        self.start_web_server()
        
        # Run persistent sidebar button registration commands
        if getattr(self, 'sidebar_buttons_to_persist', None):
            for cmd_name in self.sidebar_buttons_to_persist:
                self.process_command(f"{cmd_name} --register-only", record_history=False, silent=True)
        
        # Ensure the side panel is hidden at startup
        self.set_sidebar_visible(False)
        
        self.main_window.show()
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
        font_line_height = (
            (text_visual.font_size / 72.0)
            * text_visual.transforms.dpi
            * getattr(text_visual, '_line_height', 1.2)
        )
        extra_height = max(0, len(rendered_lines) - 1) * font_line_height
        height = base_height + extra_height
        radius = cfg_hud["console_bg_radius"]
        center_x = left_edge + width / 2.0
        text_block_height = max(1, len(rendered_lines)) * font_line_height
        text_y = cfg_hud["console_text_y"]
        text_anchor_y = cfg_hud.get("console_text_anchor_y", "bottom")
        if text_anchor_y == "top":
            center_y = text_y + text_block_height / 2.0
        elif text_anchor_y in ("center", "middle"):
            center_y = text_y
        else:
            # Bottom and baseline anchored text extends upward from its anchor.
            center_y = text_y - text_block_height / 2.0
        center_y += cfg_hud.get("console_bg_y_offset", 0.0)

        _apply_safe_rectangle_geometry(
            self.console_bg,
            center=(center_x, center_y),
            width=width,
            height=height,
            radius=radius,
        )

    def load_and_simulate(self):
            """
            Loads layout. Saves original headers to cache, but uses simplified headers in memory.
            """
            if not os.path.exists(cfg.SAVED_LAYOUT_DIR):
                os.makedirs(cfg.SAVED_LAYOUT_DIR)
                
            # --- Resolve Path and Header ---
            cache_path, self.resolved_ref_full = utils.get_cache_filename()
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
                        expected_headers, expected_edges, expected_edge_scores, _, _ = utils.build_network_from_raw(
                            raw_data,
                            forced_ref_header=self.resolved_ref_full,
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
                        if "cluster_labels" in hf: self.cluster_labels = hf["cluster_labels"][:]
                        
                        # --- Load Metadata from Cache ---
                        self.metadata = {}
                        if "metadata" in hf:
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
                            "connectivity", "edge_scores"
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
                if os.path.exists(cache_path):
                    raise RuntimeError(
                        f"New cache target already exists and will not be overwritten: {cache_path}"
                    )
                # ---> FIX: Normalize the slash direction for the console output <---
                clean_hdf5_path = os.path.normpath(cfg.INPUT_HDF5)
                print(f"--- Calculating New Layout (Raw: {clean_hdf5_path}) ---")
                try:
                    with h5py.File(cfg.INPUT_HDF5, "r") as raw_data: 
                        self.full_headers, self.edges, self.edge_scores, initial_pos, self.box_limit = utils.build_network_from_raw(
                            raw_data, 
                            forced_ref_header=self.resolved_ref_full,
                            selected_fasta_headers=selected_fasta_headers,
                        )
                except Exception as e:
                    sys.exit(f"Error loading HDF5 file: {e}")
                
                self.n_nodes = len(self.full_headers)
                print(f"Network Built: {self.n_nodes} Nodes, {len(self.edges)} Edges.")
                
                # --- Layout Engine Calculation ---
                if getattr(cfg, 'UMAP_MODE', False):
                    import Layout_Engine_UMAP as Layout_Engine
                else:
                    engine_style = getattr(cfg, 'PHYSICS_ENGINE', 'Molecular Dynamics (Style)')
                    if engine_style == 'Monte Carlo (Style)':
                        import Layout_Engine_SSN_MonteCarlo as Layout_Engine
                    else:
                        import Layout_Engine_SSN_MolecularDynamics as Layout_Engine
                
                # Construct params dictionary from cfg
                params = {
                    'PHYSICS_ENGINE': getattr(cfg, 'PHYSICS_ENGINE', 'Molecular Dynamics (Style)'),
                    'LAYOUT_DEVICE_SELECTION': getattr(
                        cfg, 'LAYOUT_DEVICE_SELECTION', 'auto'
                    ),
                    'BOX_SCALE': getattr(cfg, 'BOX_SCALE', 1.0),
                    'SIMILARITY_THRESHOLD': getattr(cfg, 'SIMILARITY_THRESHOLD', 0.0),
                    'ENABLE_PROGRESSIVE_SIMULATION': getattr(cfg, 'ENABLE_PROGRESSIVE_SIMULATION', True),
                    'RMSD_WINDOW': getattr(cfg, 'RMSD_WINDOW', 50),
                    'MAX_STEPS': getattr(cfg, 'MAX_STEPS', 2000),
                    'RMSD_THRESHOLD': getattr(cfg, 'RMSD_THRESHOLD', 0.005),
                    'PERCENTAGE_DROP_THRESHOLD': getattr(cfg, 'PERCENTAGE_DROP_THRESHOLD', 0.0),
                    'PACKING_GRID_SIZE': getattr(cfg, 'PACKING_GRID_SIZE', 200.0),
                    'PACKING_PADDING': getattr(cfg, 'PACKING_PADDING', 50.0),
                    'PACKING_GEOMETRY': getattr(cfg, 'PACKING_GEOMETRY', 'Square'),
                    'COULOMB_CUTOFF': getattr(cfg, 'COULOMB_CUTOFF', 15.0),
                    'COULOMB_K': getattr(cfg, 'COULOMB_K', 50.0),
                    'MAX_FORCE_LIMIT': getattr(cfg, 'MAX_FORCE_LIMIT', 20.0),
                    'MAX_TOTAL_REPULSION_FORCE': getattr(
                        cfg, 'MAX_TOTAL_REPULSION_FORCE', 0.0
                    ),
                    'SPRING_K': getattr(cfg, 'SPRING_K', 0.1),
                    'DAMPING': getattr(cfg, 'DAMPING', 0.5),
                    'DT': getattr(cfg, 'DT', 0.1),
                    'UMAP_NEIGHBORS': getattr(cfg, 'UMAP_NEIGHBORS', 15),
                    'UMAP_MIN_DIST': getattr(cfg, 'UMAP_MIN_DIST', 0.1),
                    'SGLD_MIN_K': getattr(cfg, 'SGLD_MIN_K', 20),
                    'SGLD_K_PERCENT': getattr(cfg, 'SGLD_K_PERCENT', 0.01),
                    'SGLD_START_TEMP': getattr(cfg, 'SGLD_START_TEMP', 1.5),
                    'SGLD_NOISE_SCALE': getattr(cfg, 'SGLD_NOISE_SCALE', 1.0)
                }
                
                # Construct N x 3 connectivity table
                if len(self.edges) > 0:
                    connectivity = np.column_stack((self.edges, self.edge_scores))
                else:
                    connectivity = np.zeros((0, 3), dtype=np.float32)
                    
                self.pos, self.box_limit = Layout_Engine.calculate_layout(connectivity, self.n_nodes, params)
                
                # --- Save to Cache (Using FULL headers) ---
                try:
                    cache_folder = os.path.dirname(cache_path)
                    os.makedirs(cache_folder, exist_ok=True)

                    fasta_backup_name = os.path.basename(
                        os.path.normpath(selected_fasta_path)
                    )
                    fasta_backup_path = os.path.join(
                        cache_folder, fasta_backup_name
                    )
                    if os.path.normcase(os.path.abspath(fasta_backup_path)) == (
                        os.path.normcase(os.path.abspath(selected_fasta_path))
                    ):
                        raise RuntimeError(
                            "The cache FASTA backup path resolves to the selected "
                            "input FASTA. Choose a separate saved-layout folder."
                        )
                    sanitized_headers = [
                        header for header, _ in self._selected_fasta_records
                    ]
                    sanitized_sequences = [
                        sequence for _, sequence in self._selected_fasta_records
                    ]
                    write_fasta_atomic(
                        fasta_backup_path,
                        sanitized_headers,
                        sanitized_sequences,
                    )
                    print(
                        f"Sanitized FASTA backup saved to: "
                        f"{fasta_backup_path}"
                    )

                    partial_cache_path = cache_path + ".partial"
                    if os.path.exists(partial_cache_path):
                        os.remove(partial_cache_path)

                    with h5py.File(partial_cache_path, "w") as hf:
                        dt_str = h5py.string_dtype(encoding='utf-8')
                        hf.attrs["cache_manifest_id"] = self.cache_manifest_id
                        hf.create_dataset("headers", data=np.array(self.full_headers, dtype=object), dtype=dt_str, compression="gzip")
                        hf.create_dataset("positions", data=self.pos, compression="gzip")
                        
                        if getattr(self, 'last_cluster_params', None) is not None: hf.attrs["last_cluster_params"] = json.dumps(self.last_cluster_params)

                    os.replace(partial_cache_path, cache_path)
                    cache_manifest.write_manifest_atomic(
                        cache_folder, self.cache_manifest
                    )
                    print(f"Layout saved to: {cache_path}")
                except Exception as e:
                    if 'partial_cache_path' in locals() and os.path.exists(partial_cache_path):
                        os.remove(partial_cache_path)
                    raise RuntimeError(f"Could not save layout cache: {e}") from e

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
        if not hasattr(self, 'redo_stack'): self.redo_stack = []
        if not hasattr(self, 'selected_indices'): self.selected_indices = []
        if not hasattr(self, 'cluster_labels'): self.cluster_labels = None
        if not hasattr(self, 'group_labels'): self.group_labels = [set() for _ in range(self.n_nodes)]
        if not hasattr(self, 'metadata'): self.metadata = {}
        
        # Initialize Length metadata if not already loaded from cache
        if "Length" not in self.metadata:
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

    def _save_state(self):
        """Saves current state to history and clears redo stack."""
        self.position_history.append(self._get_current_state())
        if len(self.position_history) > 50:
            self.position_history.pop(0)
        self.redo_stack.clear()
        
    def _do_undo(self):
        if len(self.position_history) > 0:
            self.redo_stack.append(self._get_current_state())
            state = self.position_history.pop()
            self._apply_state(state)
            msg = "Undo successful."
        else:
            msg = "Nothing to undo."
        self.console_text.text = msg
        print(msg)

    def _do_redo(self):
        if len(self.redo_stack) > 0:
            self.position_history.append(self._get_current_state())
            state = self.redo_stack.pop()
            self._apply_state(state)
            msg = "Redo successful."
        else:
            msg = "Nothing to redo."
        self.console_text.text = msg
        print(msg)

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
                
                self.line_visual = scene.visuals.Line(
                    pos=np.array(edge_coords), connect='segments', 
                    color=tuple(edge_rgba), width=cfg.EDGE_WIDTH, parent=self.view.scene
                )
                self.line_visual.set_gl_state('translucent', depth_test=False)
                if getattr(cfg, 'UMAP_MODE', False):
                    self.line_visual.visible = False
        else:
            self.line_visual = None
            
        self.left_click_highlight = scene.visuals.Markers(parent=self.view.scene)
        self.left_click_highlight.set_data(pos=np.array([[0.0, 0.0]], dtype=np.float32))
        self.left_click_highlight.set_gl_state('translucent', depth_test=False)
        self.left_click_highlight.visible = False

        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.update_nodes()
        
        # --- MODIFIED: Parent changed to canvas.scene ---
        self.tooltip = scene.visuals.Text(
            text="",
            color=cfg.TEXT_COLOR,
            pos=(0, 0),
            anchor_x='left',
            face=self.vispy_ui_face,
            font_size=cfg.TEXT_SIZE,
            parent=self.canvas.scene,
        )

        # ---> NEW: Visuals for Selection Feedback <---
        self.selection_box = scene.visuals.Line(color='black', method='gl', parent=self.view.scene)
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
                edge_coords = self.pos[active_edges].reshape(-1, 2)
                self.line_visual.set_data(pos=edge_coords)
            else:
                self.line_visual.visible = False # Prevents Vispy crash on empty arrays

    def update_nodes(self):
        colors = self.current_colors.copy()
        import matplotlib.colors as mcolors
        if getattr(self, 'hovered_node_idx', None) is not None:
            colors[self.hovered_node_idx] = mcolors.to_rgba(cfg.HOVER_COLOR)
            
        sizes = getattr(self, 'current_sizes', cfg.NODE_SIZE)
        shapes = getattr(self, 'current_shapes', np.full(self.n_nodes, 'disc', dtype=object))
        
        # ---> FIXED: Fetch custom boundary color, fallback to black
        bound_rgba = mcolors.to_rgba(getattr(cfg, 'NODE_BOUNDARY_COLOR', '#000000'))
        
        edge_colors = np.zeros((self.n_nodes, 4), dtype=np.float32)
        edge_colors[:] = bound_rgba 
        
        # ---> NEW: Fetch custom boundary width, fallback to 0.5
        b_width = getattr(cfg, 'NODE_BOUNDARY_WIDTH', 0.5)
        edge_widths = np.full(self.n_nodes, b_width, dtype=np.float32)
        
        if getattr(self, 'selected_indices', None) is not None and len(self.selected_indices) > 0:
            hover_rgba = mcolors.to_rgba(cfg.HOVER_COLOR)
            conn_rgba = mcolors.to_rgba(getattr(cfg, 'CONNECTED_NODE_COLOR', '#ff0000'))
            
            # Cache the neighbor computation because np.isin is expensive and selected_indices don't change during drag
            current_sel_tuple = tuple(self.selected_indices)
            if getattr(self, '_last_selected_tuple', None) != current_sel_tuple:
                self._last_selected_tuple = current_sel_tuple
                
                # Find neighbors using fast NumPy masking
                selected_arr = np.array(self.selected_indices)
                mask_u = np.isin(self.edges[:, 0], selected_arr)
                mask_v = np.isin(self.edges[:, 1], selected_arr)
                
                # XOR mask correctly isolates edges where exactly one side is selected
                # meaning the other side must be a neighbor.
                valid_edge_mask = mask_u ^ mask_v
                connected_edges = self.edges[valid_edge_mask]
                
                # Extract unique neighbors that aren't themselves selected
                self._cached_neighbors = np.unique(connected_edges[~np.isin(connected_edges, selected_arr)])
                
            neighbor_indices = self._cached_neighbors
            
            if len(neighbor_indices) > 0:
                edge_colors[neighbor_indices] = conn_rgba
                edge_widths[neighbor_indices] = 2.0
                
            edge_colors[self.selected_indices] = hover_rgba
            edge_widths[self.selected_indices] = 2.0
        
       # ---> NEW: Filter data by visibility mask <---
        vis = self.visible_mask
        if not np.any(vis):
            self.markers.visible = False
        else:
            self.markers.visible = True
            
            # Safely extract array and handle single float cases
            safe_sizes = sizes[vis] if isinstance(sizes, np.ndarray) else sizes
            
            self.markers.set_data(
                pos=self.pos[vis], 
                face_color=colors[vis], 
                edge_color=edge_colors[vis], 
                size=safe_sizes, 
                edge_width=edge_widths[vis],
                symbol=shapes[vis].tolist() 
            )
        self.markers.set_gl_state('translucent', depth_test=False)
        
        # ---> NEW: Update left-click background highlight if active <---
        if getattr(self, 'left_click_highlight', None) is not None:
            indices_to_highlight = []
            if getattr(self, 'left_click_highlight_indices', None):
                indices_to_highlight = [idx for idx in self.left_click_highlight_indices if self.visible_mask[idx]]
            elif getattr(self, 'selected_node_idx', None) is not None:
                sel_idx = self.selected_node_idx
                if self.visible_mask[sel_idx]:
                    indices_to_highlight = [sel_idx]
            
            if len(indices_to_highlight) > 0:
                pos_list = []
                color_list = []
                size_list = []
                symbol_list = []
                for idx in indices_to_highlight:
                    shape = shapes[idx] if isinstance(shapes, np.ndarray) else shapes
                    node_size = sizes[idx] if isinstance(sizes, np.ndarray) else sizes
                    enlarged_size = node_size * 2.0
                    
                    node_color = colors[idx]
                    highlight_color = node_color.copy()
                    highlight_color[3] = highlight_color[3] * 0.5
                    
                    pos_list.append(self.pos[idx])
                    color_list.append(highlight_color)
                    size_list.append(enlarged_size)
                    symbol_list.append(shape)
                
                self.left_click_highlight.set_data(
                    pos=np.array(pos_list, dtype=np.float32),
                    face_color=np.array(color_list, dtype=np.float32),
                    edge_color=np.zeros((len(pos_list), 4), dtype=np.float32),
                    size=np.array(size_list, dtype=np.float32),
                    edge_width=np.zeros(len(pos_list), dtype=np.float32),
                    symbol=symbol_list
                )
                self.left_click_highlight.visible = True
            else:
                self.left_click_highlight.visible = False
                
        self.canvas.update()
        
        # ---> NEW: Force HUD to instantly sync whenever visual state changes
        self._update_hud_elements()

    def create_hud(self):
        cfg_hud = self.hud_layout
        
        self.instr_text = scene.visuals.Text(
            text="[ENTER] Command | [LeftClick] Highlight | [RightClick] Select/Clear | [Scroll] Zoom | [LeftClick + Shift/Ctrl] Copy Node Header/Sequence | [LeftClick + Drag] Pan | [RightClick + Drag] GroupSelect/MoveNodes",
            bold=False, 
            face=self.vispy_ui_face,
            font_size=8, 
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
            font_size=8, 
            color=cfg.TEXT_COLOR, 
            pos=(cfg_hud["console_text_x"], cfg_hud["console_text_y"]), 
            anchor_y=cfg_hud["console_text_anchor_y"], 
            anchor_x=cfg_hud["console_text_anchor_x"], 
            parent=self.canvas.scene
        )
        
        self.zoom_text = scene.visuals.Text(
            text="", 
            bold=False, 
            face=self.vispy_ui_face,
            font_size=8, 
            color='gray', 
            pos=(self.canvas.size[0] - cfg_hud["zoom_x_offset"], self.canvas.size[1] - cfg_hud["zoom_y_offset"]), 
            anchor_y=cfg_hud["zoom_anchor_y"], 
            anchor_x=cfg_hud["zoom_anchor_x"], 
            parent=self.canvas.scene
        )
        
        self.hidden_text = scene.visuals.Text(
            text="", 
            bold=False, 
            face=self.vispy_ui_face,
            font_size=8, 
            color='gray', 
            pos=(self.canvas.size[0] - cfg_hud["hidden_x_offset"], self.canvas.size[1] - cfg_hud["hidden_y_offset"]), 
            anchor_y=cfg_hud["hidden_anchor_y"], 
            anchor_x=cfg_hud["hidden_anchor_x"], 
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
                # Broadcast state update to HTML5 browser!
                self.broadcast_event({
                    "type": "state_updated",
                    "visible_mask": self.visible_mask.tolist(),
                    "selected_indices": self.selected_indices,
                    "metadata": self.get_serializable_metadata()
                })
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
            
            dists = np.linalg.norm(self.pos[:, :2] - mouse_world, axis=1)
            dists[~self.visible_mask] = np.inf
            nearest_idx = np.argmin(dists)
            
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
                self.selection_box.set_data(pos=np.zeros((5, 2)))
                self.selection_box.visible = True
                
            event.handled = True 
            return
            
        # ---> 2. SHIFT + LEFT-CLICK LOGIC (Copy to Clipboard) <---
        # Vispy Button 1 = Left Click
        if event.button == 1 and 'Shift' in event.modifiers and not self.console_mode:
            tr = self.canvas.scene.node_transform(self.view.scene)
            mouse_world = tr.map(event.pos)[:2]
            
            dists = np.linalg.norm(self.pos[:, :2] - mouse_world, axis=1)
            dists[~self.visible_mask] = np.inf
            nearest_idx = np.argmin(dists)
            
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
            
            dists = np.linalg.norm(self.pos[:, :2] - mouse_world, axis=1)
            dists[~self.visible_mask] = np.inf
            nearest_idx = np.argmin(dists)
            
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
            
            dists = np.linalg.norm(self.pos[:, :2] - mouse_world, axis=1)
            dists[~self.visible_mask] = np.inf
            nearest_idx = np.argmin(dists)
            
            node_screen_pos = tr.inverse.map(self.pos[nearest_idx])[:2]
            screen_dist = np.linalg.norm(node_screen_pos - event.pos)
            
            # If clicked within the node's radius, highlight the node and print full header
            if screen_dist < cfg.NODE_SIZE:
                self.selected_node_idx = nearest_idx
                self.tooltip.text = ""
                
                # Row 1: Cluster + Header
                lbl_line1 = ""
                if getattr(self, 'cluster_labels', None) is not None:
                    cid = self.cluster_labels[nearest_idx]
                    lbl_line1 = "[Noise] " if cid == -1 else f"[Cluster {cid}] "
                
                lbl_line1 += self.full_headers[nearest_idx]
                
                # Row 2: Groups (if they exist)
                group_line = ""
                group_print = ""
                if getattr(self, 'group_labels', None) and self.group_labels[nearest_idx]:
                    group_str = ", ".join(sorted(self.group_labels[nearest_idx]))
                    group_line = f"\n[Groups: {group_str}]"
                    group_print = f" [Groups: {group_str}]"
                
                # Fetch and print the full FASTA header (keep print statement on one line)
                full_header = self.full_headers[nearest_idx]
                print(f"Node Selected: {full_header}")
                
                # Optionally update the HUD console text with a brief confirmation
                self.console_text.text = f"Selected: {lbl_line1}{group_print}"
                
                # Sync selection to metadata spreadsheet
                if hasattr(self, 'sync_metadata_table_selection'):
                    self.sync_metadata_table_selection(nearest_idx)

                # Update any registered HUD displays
                for display in self.hud_displays.values():
                    if getattr(display, 'on_node_clicked', None):
                        display.on_node_clicked(nearest_idx)
                
                self.update_nodes()
            else:
                # Clicking empty space clears the left-click highlight
                self.selected_node_idx = None
                self.tooltip.text = ""
                self.update_nodes()
                
            event.handled = True
            return
    
    def _update_hud_elements(self, event=None):
        """Updates the zoom indicator, hidden nodes count, and maintains the tooltip pixel gap."""
        cfg_hud = self.hud_layout
        
        panel_visible = hasattr(self, 'right_panel') and self.right_panel.isVisible()
        panel_w = getattr(self, '_panel_w', 120) if panel_visible else 0
        effective_canvas_w = self.canvas.size[0] - panel_w
        
        # 1. Update Zoom Indicator (Visible World Width)
        if hasattr(self, 'zoom_text'):
            visible_width = self.view.camera.rect.width
            self.zoom_text.text = f"View Width: {visible_width:.1f}"
            self.zoom_text.pos = (effective_canvas_w - cfg_hud["zoom_x_offset"], self.canvas.size[1] - cfg_hud["zoom_y_offset"]) 

        # 2. Update Hidden Nodes Indicator
        if hasattr(self, 'hidden_text') and hasattr(self, 'visible_mask'):
            hidden_count = int(np.sum(~self.visible_mask))
            self.hidden_text.text = f"Hidden Nodes: {hidden_count}"
            if hidden_count > 0:
                self.hidden_text.color = 'red'
            else:
                self.hidden_text.color = 'gray'
            self.hidden_text.pos = (effective_canvas_w - cfg_hud["hidden_x_offset"], self.canvas.size[1] - cfg_hud["hidden_y_offset"])

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
        """Collapse related Qt display notifications into one layout pass."""
        if getattr(self, '_display_refresh_pending', False):
            return
        self._display_refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refresh_display_layout)

    def _refresh_display_layout(self):
        """Refresh VisPy's framebuffer and all logical-pixel overlays."""
        self._display_refresh_pending = False
        canvas = getattr(self, 'canvas', None)
        native_canvas = getattr(canvas, 'native', None) if canvas is not None else None

        if native_canvas is not None:
            try:
                screen_changed = getattr(native_canvas, 'screen_changed', None)
                if callable(screen_changed):
                    screen_changed(getattr(self, '_active_screen', None))
                else:
                    resize_gl = getattr(native_canvas, 'resizeGL', None)
                    width = native_canvas.width()
                    height = native_canvas.height()
                    if callable(resize_gl) and width > 0 and height > 0:
                        resize_gl(width, height)
            except Exception as e:
                print(f"Warning: Could not refresh VisPy display scaling: {e}")

        for method_name in (
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
        except Exception as e:
            print(f"Warning: Could not read display information: {e}")
            return

        signature = (
            screen_name,
            screen_geometry,
            round(dpr, 6),
            logical_size,
            physical_size,
        )
        if signature == getattr(self, '_last_display_signature', None):
            return
        self._last_display_signature = signature

        print(
            "Display: "
            f"{screen_name} | geometry={screen_geometry[2]}x{screen_geometry[3]} "
            f"at ({screen_geometry[0]}, {screen_geometry[1]}) | DPR={dpr:g} | "
            f"canvas logical={logical_size[0]}x{logical_size[1]} | "
            f"physical={physical_size[0]}x{physical_size[1]}"
        )

    def on_resize(self, event): 
        self._hud_timer.start()
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
        import webbrowser
        webbrowser.open("http://localhost:8000/meta.html")

    def open_agent_ui(self):
        import webbrowser
        webbrowser.open("http://localhost:8000/agent.html")

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
            print(f"WebServer started at http://localhost:8000")
        except Exception as e:
            print(f"Error starting WebServer: {e}")

    def broadcast_event(self, event):
        if hasattr(self, 'web_server') and self.web_server:
            with self.web_server.queues_lock:
                queues = list(self.web_server.event_queues)
            for q in queues:
                q.put(event)

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
        return {
            "rows": self.get_serializable_metadata(),
            "columns": ["Node ID"] + [k for k in self.metadata.keys() if k.lower() != "length"],
            "selected_indices": self.selected_indices,
            "visible_mask": self.visible_mask.tolist(),
            "llm_loaded": getattr(self, 'llm_loaded', False),
            "llm_backend": getattr(self, 'llm_backend', None),
            "llm_model_name": getattr(self, 'llm_model_name', "Unknown")
        }

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
            ])
            self.selection_box.set_data(pos=rect_pts)
            
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
            dists = np.linalg.norm(self.pos[:, :2] - mouse_world, axis=1)
            dists[~self.visible_mask] = np.inf
            nearest_idx = np.argmin(dists)
            
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
    viewer = MainViewer()
    app.run()


