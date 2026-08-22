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
# Import Libraries
import ast
import html
import json
import math
import os
import re
import sys
import tempfile
from types import SimpleNamespace
import traceback
from pathlib import Path
from utilities.Terminal_Launcher import HoldMode, launch_in_terminal
from utilities.Application_Identity import (
    VIEWER_DESKTOP_FILE_NAME,
    configure_linux_qt_desktop_identity,
)
from Layout_Cache_Generator import LayoutGenerationSettings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

# --- Placeholder Parameters ---
SEQUENCE_SET = ""
ALIGNMENT_REFERENCE = ""
ALIGNMENT_OFFSET = 0
SIMILARITY_THRESHOLD = None
TOP_EDGE_PERCENT = None    
ALIGNMENT_SCORE = None
NORM_MODE = None
UMAP_MODE = None
UMAP_NEIGHBORS = None
UMAP_MIN_DIST = None
TARGET_CACHE_FILE = os.environ.get("SSN_TARGET_CACHE", None)
TARGET_CACHE_PATH = os.environ.get("SSN_TARGET_CACHE_PATH", None)
TARGET_CACHE_MODE = os.environ.get("SSN_TARGET_CACHE_MODE", None)

# --- Directory & File Paths ---
FASTA_DIR = os.path.join("Input_Files", "Sequence_Sets")
MSA_DIR = os.path.join("Input_Files", "Multiple_Alignments")
HDF5_DIR = os.path.join("Input_Files", "Networks_EValues")
SAVED_LAYOUT_DIR = os.path.join("Cache_Files", "Saved_Layouts")
SETTING_EXPORT_DIR = os.path.join("Cache_Files", "Exported_Settings")
METADATA_DIR = os.path.join("Input_Files", "Meta_Data")
PRINT_SAVE_DIR = os.path.join("Analysis_Results", "Saved_Images")
SEQUENCE_EXPORT_DIR = os.path.join("Analysis_Results", "Sequence_Export")
CLUSTER_LABEL_DIR = os.path.join("Analysis_Results", "Cluster_Label")
HEADER_LIST_DIR = os.path.join("Input_Files", "Header_Lists")
LOGO_DIR = os.path.join("Analysis_Results", "Sequence_Logos")
STRUCTURES_DIR = os.path.join("Cache_Files", "Predicted_Structures")
DEFAULT_SAVED_CONFIG_DIR = os.path.join("Cache_Files", "Saved_Config")
SAVED_CONFIG_DIR = DEFAULT_SAVED_CONFIG_DIR

# --- Explicit Input File Paths ---
# You can manually replace these string paths to decouple file logic:
NODE_FASTA_FILE = ""

# Default values if scanning directories fails:
MSA_FILE = ""
INPUT_HDF5 = ""

# Sequences File points to NODE_FASTA_FILE for backward compatibility
SEQUENCES_FILE = ""

# --- Command Settings ---
GAP_CHARS = ['-', '.']
FILTER_MIN_OCCUPANCY = 10.0

# --- Visual Defaults ---
NODE_SIZE = 10
EDGE_WIDTH = 1.0
EDGE_ALPHA = 0.1     
NODE_BOUNDARY_WIDTH = 0.5
TEXT_SIZE = 8
TEXT_COLOR = 'grey'
INITIAL_NODE_COLOR = '#4488ff'
HOVER_COLOR = '#ffaa00'
CONNECTED_NODE_COLOR = '#ff0000'
EDGE_COLOR = '#000000'
LOW_RESOURCE_MODE = False
NODE_BOUNDARY_COLOR = '#000000'

# --- Grid Packing Settings ---
PACKING_GRID_SIZE = 20.0  # The base size of one grid square
PACKING_PADDING = 10.0     # Extra padding applied to the bounding box of each cluster

# --- Simulation & Physics Settings ---
PHYSICS_ENGINE = "Molecular Dynamics (Style)"
LAYOUT_DEVICE_SELECTION = "auto"
SPRING_K = 5.0             
COULOMB_K = 10.0            
COULOMB_CUTOFF = 30.0      
DAMPING = 0.9              
MAX_FORCE_LIMIT = 20.0      
MAX_TOTAL_REPULSION_FORCE = 0.0

DT = 0.005
BOX_SCALE = 2.0
MAX_STEPS = 10000           
RMSD_THRESHOLD = 0.005 
PERCENTAGE_DROP_THRESHOLD = 0.1    
RMSD_WINDOW = 50
ENABLE_PROGRESSIVE_SIMULATION = False
PACKING_GEOMETRY = "Square"

# --- Monte Carlo / SGLD Settings ---
SGLD_MIN_K = 20
SGLD_K_PERCENT = 0.01
SGLD_START_TEMP = 1.5
SGLD_NOISE_SCALE = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PROFILE_DEFAULTS = {
    "NODE_FASTA_FILE": "",
    "MSA_FILE": "",
    "INPUT_HDF5": "",
    "ALIGNMENT_SCORE": "global",
    "NORM_MODE": "alignment_length",
    "ALIGNMENT_REFERENCE": "",
    "ALIGNMENT_OFFSET": 0,
    "UMAP_MODE": False,
    "UMAP_NEIGHBORS": 15,
    "UMAP_MIN_DIST": 0.1,
    "SIMILARITY_THRESHOLD": None,
    "TOP_EDGE_PERCENT": None,
    "FILTER_MIN_OCCUPANCY": 10.0,
}

VISUAL_PROFILE_DEFAULTS = {
    "NODE_SIZE": 10,
    "EDGE_WIDTH": 1.0,
    "NODE_BOUNDARY_WIDTH": 0.5,
    "EDGE_ALPHA": 0.1,
    "TEXT_SIZE": 8,
    "TEXT_COLOR": "grey",
    "INITIAL_NODE_COLOR": "#4488ff",
    "HOVER_COLOR": "#ffaa00",
    "CONNECTED_NODE_COLOR": "#ff0000",
    "EDGE_COLOR": "#000000",
    "NODE_BOUNDARY_COLOR": "#000000",
    "LOW_RESOURCE_MODE": False,
}

PHYSICS_PROFILE_DEFAULTS = {
    "PHYSICS_ENGINE": "Molecular Dynamics (Style)",
    "LAYOUT_DEVICE_SELECTION": "auto",
    "SPRING_K": 5.0,
    "COULOMB_K": 10.0,
    "COULOMB_CUTOFF": 30.0,
    "DAMPING": 0.9,
    "DT": 0.005,
    "MAX_STEPS": 10000,
    "RMSD_THRESHOLD": 0.005,
    "PERCENTAGE_DROP_THRESHOLD": 0.1,
    "RMSD_WINDOW": 50,
    "ENABLE_PROGRESSIVE_SIMULATION": False,
    "PACKING_GEOMETRY": "Square",
    "PACKING_GRID_SIZE": 20.0,
    "SGLD_MIN_K": 20,
    "SGLD_K_PERCENT": 0.01,
    "SGLD_START_TEMP": 1.5,
    "SGLD_NOISE_SCALE": 1.0,
}

DIRECTORY_PROFILE_DEFAULTS = {
    "FASTA_DIR": os.path.join("Input_Files", "Sequence_Sets"),
    "MSA_DIR": os.path.join("Input_Files", "Multiple_Alignments"),
    "HDF5_DIR": os.path.join("Input_Files", "Networks_EValues"),
    "SAVED_LAYOUT_DIR": os.path.join("Cache_Files", "Saved_Layouts"),
    "SETTING_EXPORT_DIR": os.path.join("Cache_Files", "Exported_Settings"),
    "METADATA_DIR": os.path.join("Input_Files", "Meta_Data"),
    "SEQUENCE_EXPORT_DIR": os.path.join("Analysis_Results", "Sequence_Export"),
    "HEADER_LIST_DIR": os.path.join("Input_Files", "Header_Lists"),
    "STRUCTURES_DIR": os.path.join("Cache_Files", "Predicted_Structures"),
    "PRINT_SAVE_DIR": os.path.join("Analysis_Results", "Saved_Images"),
    "CLUSTER_LABEL_DIR": os.path.join("Analysis_Results", "Cluster_Label"),
    "LOGO_DIR": os.path.join("Analysis_Results", "Sequence_Logos"),
}

DIRECTORY_DISPLAY_NAMES = {
    "FASTA_DIR": "Input FASTA Directory",
    "SAVED_LAYOUT_DIR": "Layout Directory",
    "SETTING_EXPORT_DIR": "Setting Export Directory",
    "SEQUENCE_EXPORT_DIR": "Sequence Export Directory",
    "PRINT_SAVE_DIR": "Print Directory",
}

TAB_PROFILE_SPECS = {
    "inputs_outputs": {
        "defaults": INPUT_PROFILE_DEFAULTS,
        "allow_default": False,
    },
    "visual_effects": {
        "defaults": VISUAL_PROFILE_DEFAULTS,
        "allow_default": True,
    },
    "simulation_physics": {
        "defaults": PHYSICS_PROFILE_DEFAULTS,
        "allow_default": True,
    },
    "directories": {
        "defaults": DIRECTORY_PROFILE_DEFAULTS,
        "allow_default": True,
    },
}

PROFILE_TAB_DISPLAY_NAMES = {
    "inputs_outputs": "Inputs & Outputs",
    "visual_effects": "Visual Effects",
    "simulation_physics": "Simulation & Physics",
    "directories": "Directories",
}

RESERVED_PROFILE_NAMES = {"custom", "default", "new"}
CONFIG_TAB_CONTENT_MARGIN = 18
CONFIG_TAB_ROW_SPACING = 12
CONFIG_FIELD_LABEL_WIDTH = 180
CONFIG_FIELD_HORIZONTAL_SPACING = 12
CONFIG_SEPARATOR_THICKNESS = 2
CONFIG_SEPARATOR_PADDING = 24
PROFILE_DISABLED_LABEL_STYLESHEET = "QLabel:disabled { color: #888; }"
PROFILE_DISABLED_SPINBOX_STYLESHEET = (
    "QSpinBox:disabled, QDoubleSpinBox:disabled { "
    "background-color: #f0f0f0; color: #888; }"
)
PROFILE_DISABLED_TOGGLE_STYLESHEET = (
    "QPushButton:disabled { background-color: #e0e0e0; color: #888; "
    "border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }"
)

PROFILE_ENUM_VALUES = {
    "ALIGNMENT_SCORE": {"global", "local"},
    "NORM_MODE": {
        "alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"
    },
    "PHYSICS_ENGINE": {"Molecular Dynamics (Style)", "Monte Carlo (Style)"},
    "PACKING_GEOMETRY": {"Square", "Circle"},
}

PROFILE_RANGES = {
    "ALIGNMENT_OFFSET": (-1000000, 1000000),
    "UMAP_NEIGHBORS": (2, 500),
    "UMAP_MIN_DIST": (0.0, 1.0),
    "TOP_EDGE_PERCENT": (0.0, 100.0),
    "FILTER_MIN_OCCUPANCY": (0.0, 100.0),
    "NODE_SIZE": (1, 20),
    "EDGE_WIDTH": (0.1, 3.0),
    "NODE_BOUNDARY_WIDTH": (0.0, 2.0),
    "EDGE_ALPHA": (0.0, 1.0),
    "TEXT_SIZE": (1, 24),
    "SPRING_K": (1.0, 20.0),
    "COULOMB_K": (1.0, 30.0),
    "COULOMB_CUTOFF": (1.0, 100.0),
    "DAMPING": (0.1, 2.0),
    "RMSD_WINDOW": (10, 1000),
    "PACKING_GRID_SIZE": (1.0, 200.0),
}


def _resolved_saved_config_root(value):
    path = Path(str(value or DEFAULT_SAVED_CONFIG_DIR)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _migrate_saved_config_dir(value):
    normalized = os.path.normpath(str(value or "")).replace("\\", "/").rstrip("/")
    if normalized.casefold() == "saved_config":
        return DEFAULT_SAVED_CONFIG_DIR
    return value


def _discover_profile_names(root, tab_id):
    folder = Path(root) / tab_id
    try:
        names = sorted((
            entry.stem
            for entry in folder.iterdir()
            if entry.is_file()
            and entry.suffix.lower() == ".json"
            and entry.stem.casefold() not in RESERVED_PROFILE_NAMES
        ), key=str.casefold)
    except (FileNotFoundError, NotADirectoryError, OSError):
        return []
    unique_names = {}
    for name in names:
        unique_names.setdefault(name.casefold(), name)
    return list(unique_names.values())


def _validate_profile_name(name, existing_names=()):
    normalized = str(name).strip()
    if normalized.lower().endswith(".json"):
        normalized = normalized[:-5].rstrip()
    if not normalized:
        raise ValueError("Enter a profile name.")
    if normalized.casefold() in RESERVED_PROFILE_NAMES:
        raise ValueError(f"'{normalized}' is a reserved profile name.")
    if normalized in {".", ".."} or normalized.endswith((" ", ".")):
        raise ValueError("Profile names cannot end with a space or period.")
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', normalized):
        raise ValueError("Profile names cannot contain path separators or Windows-reserved characters.")
    windows_stem = normalized.split(".", 1)[0].casefold()
    windows_reserved = {"con", "prn", "aux", "nul"}
    windows_reserved.update(f"com{i}" for i in range(1, 10))
    windows_reserved.update(f"lpt{i}" for i in range(1, 10))
    if windows_stem in windows_reserved:
        raise ValueError(f"'{normalized}' is reserved by Windows.")
    if normalized.casefold() in {str(item).casefold() for item in existing_names}:
        raise ValueError(f"A profile named '{normalized}' already exists.")
    return normalized


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, indent=4)
            handle.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise

# --- JSON Settings Override ---
import Cache_Manifest as cache_manifest

DEFAULT_SETTINGS_FILE = str(PROJECT_ROOT / "Input_Files" / "viewer_settings.json")
SETTINGS_FILE = os.environ.get("SSN_VIEWER_SETTINGS_PATH") or DEFAULT_SETTINGS_FILE
viewer_settings = {}
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            viewer_settings = json.load(f)
            # Map legacy settings keys to new keys
            LEGACY_KEYS_MAPPING = {
                "NEIGHBOR_COLOR": "INITIAL_NODE_COLOR"
            }
            for k, v in viewer_settings.items():
                if k in LEGACY_KEYS_MAPPING:
                    k = LEGACY_KEYS_MAPPING[k]
                if k == "SAVED_CONFIG_DIR":
                    v = _migrate_saved_config_dir(v)
                if k in globals() and v is not None and (str(v).strip() != "" or k in ["MSA_FILE", "ALIGNMENT_REFERENCE"]):
                    orig = globals()[k]
                    if isinstance(orig, int) and not isinstance(orig, bool):
                        try: v = int(v)
                        except: pass
                    elif isinstance(orig, float):
                        try: v = float(v)
                        except: pass
                    elif isinstance(orig, bool):                                  # <--- ADD THIS
                        v = str(v).lower() in ['true', '1', 't', 'y', 'yes']      # <--- ADD THIS
                    elif isinstance(orig, list):
                        try: v = ast.literal_eval(v) if isinstance(v, str) else v
                        except: pass
                    elif orig is None:
                        if v == "None": v = None
                        elif str(v).replace('.', '', 1).isdigit():
                            v = float(v) if '.' in str(v) else int(v)
                    globals()[k] = v
                        
                # ---> NEW: SYNC LEGACY VARIABLES <---
                if "NODE_FASTA_FILE" in globals() and globals()["NODE_FASTA_FILE"]:
                    globals()["SEQUENCES_FILE"] = globals()["NODE_FASTA_FILE"]
                    globals()["SEQUENCE_SET"] = os.path.splitext(os.path.basename(globals()["NODE_FASTA_FILE"]))[0]
                    # Also update default fallback paths if they still contain "None" or are empty after SEQUENCE_SET is resolved
                    if globals().get("MSA_FILE") and "none_[e1_ra]_alignment.fasta" in globals()["MSA_FILE"].lower():
                        globals()["MSA_FILE"] = os.path.join(globals()["MSA_DIR"], f"{globals()['SEQUENCE_SET']}_[E1_RA]_alignment.fasta")
                    if not globals().get("INPUT_HDF5") or "none_[e1_ra]_network.h5" in globals()["INPUT_HDF5"].lower():
                        globals()["INPUT_HDF5"] = os.path.join(globals()["HDF5_DIR"], f"{globals()['SEQUENCE_SET']}_[E1_RA]_network.h5")
                    
    except Exception as e:
        print(f"Failed to load viewer settings: {e}")


def _handoff_to_viewer(
    project_root,
    env,
    *,
    platform_name=None,
    executable=None,
):
    """Launch the viewer while preserving each platform's terminal contract."""
    executable = executable or sys.executable
    project_root = os.path.abspath(project_root)
    viewer_script = os.path.join(project_root, "src", "SSN_Viewer.py")
    return launch_in_terminal(
        [executable, "-u", viewer_script],
        cwd=project_root,
        env=env,
        hold=HoldMode.ON_ERROR,
        title="SSN Viewer",
        platform_name=platform_name,
    )


def _create_viewer_settings_snapshot(settings):
    """Write one private settings file for a single Viewer process."""
    descriptor, path = tempfile.mkstemp(prefix="ssn_viewer_", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings, handle, indent=4)
            handle.write("\n")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _load_consistency_fasta_headers(fasta_path):
    """Return the canonical FASTA headers used by downstream SSN workflows."""
    from utilities.FASTA_Sanitization import load_sanitized_fasta

    headers, _, _ = load_sanitized_fasta(fasta_path)
    return headers


def _load_consistency_msa_headers(msa_path):
    """Return canonical MSA headers using the same rules as the viewer."""
    from utilities.MSA_Sanitization import (
        load_sanitized_msa_fasta,
        sanitize_msa_headers,
    )

    if os.path.splitext(msa_path)[1].lower() == ".h5":
        import h5py

        with h5py.File(msa_path, "r") as hf:
            raw_headers = hf["headers"][:]
        decoded_headers = [
            header.decode("utf-8") if isinstance(header, bytes) else header
            for header in raw_headers
        ]
        return sanitize_msa_headers(decoded_headers)

    headers, _, _ = load_sanitized_msa_fasta(msa_path)
    return headers

# =============================================================================
# GUI APPLICATION
# =============================================================================
if __name__ == "__main__":
    # Hardware_Utils imports PyTorch.  On Windows this must happen before
    # PySide6/OpenGL initializes, otherwise torch's c10.dll can fail to load.
    try:
        from utilities import Hardware_Utils
    except ImportError:
        import Hardware_Utils

    os.environ["QT_API"] = "pyside6"
    os.environ["QT_MAC_WANTS_LIGHT_THEME"] = "1"
    from PySide6.QtWidgets import (
        QApplication, QDialog, QMainWindow, QWidget, QVBoxLayout,
        QHBoxLayout, QGridLayout, QTabWidget, QFormLayout, QLineEdit,
        QComboBox, QPushButton, QMessageBox, QTextEdit,
        QLabel, QSplitter, QSlider, QSpinBox, QDoubleSpinBox,
        QStyle, QStyleOptionSlider, QFileDialog, QColorDialog, QSizePolicy,
        QFrame,
    )
    from PySide6.QtCore import Qt, QUrl, QThread, Signal
    from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPalette
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
    from utilities.Application_Fonts import (
        configure_qt_application_fonts,
        qt_monospace_font,
    )
    from utilities.Application_Windows import (
        SingleInstanceController,
        show_window_in_front,
    )

    def apply_gated_input_palette(widget):
        """Grey disabled inputs while retaining their native Fusion controls."""
        palette = widget.palette()
        disabled = QPalette.ColorGroup.Disabled
        for role in (
            QPalette.ColorRole.Base,
            QPalette.ColorRole.AlternateBase,
            QPalette.ColorRole.Button,
            QPalette.ColorRole.Window,
        ):
            palette.setColor(disabled, role, QColor("#f0f0f0"))
        for role in (
            QPalette.ColorRole.Text,
            QPalette.ColorRole.ButtonText,
            QPalette.ColorRole.WindowText,
            QPalette.ColorRole.PlaceholderText,
        ):
            palette.setColor(disabled, role, QColor("#888888"))
        widget.setPalette(palette)

    # --- Custom Widget Classes ---
    class ScoreHistogramDialog(QDialog):
        """Qt-owned modal container for a Matplotlib score histogram."""

        def __init__(self, figure, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Score Histogram")
            self.resize(1000, 650)

            self.figure = figure
            self.canvas = FigureCanvasQTAgg(figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)

            close_button = QPushButton("Close")
            close_button.clicked.connect(self.accept)

            button_layout = QHBoxLayout()
            button_layout.addStretch(1)
            button_layout.addWidget(close_button)

            layout = QVBoxLayout(self)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, 1)
            layout.addLayout(button_layout)

        def release_figure(self):
            """Release Matplotlib resources after the modal dialog closes."""
            self.figure.clear()
            self.canvas.deleteLater()

    class SpacedTipLabel(QLabel):
        """Help-panel label with proportional multiline spacing."""

        def __init__(self, text="", parent=None):
            super().__init__(parent)
            self.setTextFormat(Qt.TextFormat.RichText)
            self.setText(text)

        def setText(self, text):
            escaped_text = html.escape(str(text)).replace("\r\n", "\n").replace("\r", "\n")
            escaped_text = escaped_text.replace("\n", "<br>")
            super().setText(f'<div style="line-height: 120%;">{escaped_text}</div>')

    class NoScrollComboBox(QComboBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class DynamicComboBox(NoScrollComboBox):
        def __init__(self, refresh_callback, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.refresh_callback = refresh_callback
            
        def showPopup(self):
            if self.refresh_callback:
                self.refresh_callback()
            super().showPopup()
            
    class NoScrollSpinBox(QSpinBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMinimumHeight(28)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class NoScrollDoubleSpinBox(QDoubleSpinBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMinimumHeight(28)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)

    class OptionalNoScrollDoubleSpinBox(NoScrollDoubleSpinBox):
        """Double spin box that keeps the existing blank/unset setting state."""

        def __init__(self, unset_value, first_value, *args, **kwargs):
            self._unset_value = float(unset_value)
            self._first_value = float(first_value)
            super().__init__(*args, **kwargs)

        def setOptionalValue(self, value):
            if value in [None, "", "None"]:
                self.setValue(self._unset_value)
            else:
                self.setValue(float(value))

        def optionalValue(self):
            if self.value() <= self.minimum():
                return None
            return self.value()

        def stepBy(self, steps):
            if self.optionalValue() is None and steps > 0:
                self.setValue(
                    self._first_value + (steps - 1) * self.singleStep()
                )
                return
            super().stepBy(steps)
                
    class NoScrollSlider(QSlider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                opt = QStyleOptionSlider()
                self.initStyleOption(opt)
                sr = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self)
                if not sr.contains(event.position().toPoint()):
                    val = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), int(event.position().x()), self.width())
                    self.setValue(val)
                    event.accept()
                    return
            super().mousePressEvent(event)

    class CacheHashWorker(QThread):
        completed = Signal(int, object, str)

        def __init__(self, request_id, sequence_path, network_path, cached_records=None):
            super().__init__()
            self.request_id = request_id
            self.sequence_path = sequence_path
            self.network_path = network_path
            self.cached_records = cached_records or {}

        def run(self):
            try:
                records = {}
                for kind, path in (
                    ("sequence", self.sequence_path),
                    ("network", self.network_path),
                ):
                    records[kind] = self.cached_records.get(kind)
                    if records[kind] is None:
                        records[kind] = cache_manifest.fingerprint_file(
                            path,
                            cancellation_requested=self.isInterruptionRequested,
                        )
                records["network_type"] = cache_manifest.validate_network_schema(
                    self.network_path
                ).network_type
                self.completed.emit(self.request_id, records, "")
            except Exception as error:
                self.completed.emit(self.request_id, {}, str(error))

    class ConfigGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("SSN Configuration Editor")
            
            # Set Window Icon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                
            self.resize(1000, 650)
            
            self.central_widget = QWidget()
            self.setCentralWidget(self.central_widget)
            
            # --- SPLIT LAYOUT ---
            self.main_layout = QHBoxLayout(self.central_widget)
            
            # Retain native Fusion theme graphics but increase the invisible grab area width
            self.main_split = QSplitter(Qt.Orientation.Horizontal)
            self.main_split.setHandleWidth(12) 
            self.main_layout.addWidget(self.main_split)
            
            # Left Panel Splitter (Vertical)
            self.left_split = QSplitter(Qt.Orientation.Vertical)
            self.left_split.setHandleWidth(12)
            self.main_split.addWidget(self.left_split)
            
            # Left Top: Tabs Only
            self.left_top_widget = QWidget()
            self.left_top_layout = QVBoxLayout(self.left_top_widget)
            self.left_top_layout.setContentsMargins(0, 0, 0, 0)
            self.tabs = QTabWidget()
            self.left_top_layout.addWidget(self.tabs)
            
            # Left Bottom: Tool Tip Box + Action Buttons
            self.left_bottom_widget = QWidget()
            self.left_bottom_layout = QVBoxLayout(self.left_bottom_widget)
            self.left_bottom_layout.setContentsMargins(0, 0, 0, 0)
            
            self.tip_panel = SpacedTipLabel("Click or tab to an input or its label to see helpful tips here.")
            self.tip_panel.setWordWrap(True)
            self.tip_panel.setStyleSheet("color: #444; font-style: normal; background-color: #e8eaed; padding: 10px; border-radius: 5px;")
            self.left_bottom_layout.addWidget(self.tip_panel)
            
            btn_layout = QHBoxLayout()
            self.btn_check = QPushButton("Consistency Check")
            self.btn_check.setStyleSheet("background-color: #f39c12; color: white; font-weight: bold; padding: 5px;")
            self.btn_check.clicked.connect(self.run_consistency_check)
            
            self.btn_save_run = QPushButton("Save && Run")
            self.btn_save_run.clicked.connect(self.save_and_run)
            self.btn_save_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")

            self.btn_export_layout = QPushButton("Export Layout Settings")
            self.btn_export_layout.clicked.connect(self.export_layout_settings)
            self.btn_export_layout.setEnabled(False)
            self.btn_export_layout.setStyleSheet(
                "background-color: #2196F3; color: white; "
                "font-weight: bold; padding: 5px;"
            )
            self.btn_export_layout.setToolTip(
                "Export a generation-only JSON file for the selected new layout cache."
            )
            
            btn_save = QPushButton("Save")
            btn_save.clicked.connect(self.save_only)
            
            btn_exit = QPushButton("Exit")
            btn_exit.clicked.connect(self.close)
            
            btn_layout.addWidget(self.btn_save_run)
            btn_layout.addWidget(self.btn_export_layout)
            btn_layout.addWidget(self.btn_check)
            btn_layout.addWidget(btn_save)
            btn_layout.addWidget(btn_exit)
            self.left_bottom_layout.addLayout(btn_layout)
            
            self.left_split.addWidget(self.left_top_widget)
            self.left_split.addWidget(self.left_bottom_widget)
            
            # Explicitly force the initial pixel heights (tabs get 550px, bottom gets 100px)
            self.left_split.setSizes([450, 200])
            
            # Ensure that if the user resizes the window, extra space goes to the tabs, not the bottom
            self.left_split.setStretchFactor(0, 1)
            self.left_split.setStretchFactor(1, 0)
            
            # Right Panel: Statistics
            self.right_panel = QWidget()
            self.right_layout = QVBoxLayout(self.right_panel)
            self.right_layout.setContentsMargins(0, 0, 0, 0)
            self.stat_label = QLabel("Network Statistics Report")
            self.stat_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            self.stat_display = QTextEdit()
            self.stat_display.setReadOnly(True)
            self.stat_display.setPlaceholderText("Select Fasta subset and HDF5 Network file, then click compute.")
            self.stat_display.setFont(qt_monospace_font(self.stat_display.font()))
            self.stat_display.setStyleSheet("background-color: #f5f5f5;")
            self.right_layout.addWidget(self.stat_label)
            self.right_layout.addWidget(self.stat_display)
            
            self.main_split.addWidget(self.right_panel)
            self.main_split.setStretchFactor(0, 6) 
            self.main_split.setStretchFactor(1, 4) 
            
            # Data Containers
            self.inputs = {}
            self.labels = {} 
            self.color_swatches = {} 
            self.profile_selectors = {}
            self.profile_name_inputs = {}
            self.profile_folder_buttons = {}
            self.profile_labels = {}
            self.profile_separators = {}
            self.profile_content_widgets = {}
            self._profile_previous_selection = {}
            self._profile_loading = False
            self._initializing_profiles = True
            self._custom_settings = self._read_custom_settings()
            self._cache_hash_cache = {}
            self._cache_hash_request_id = 0
            self._cache_hash_workers = {}
            self._cache_hash_pending_keys = None
            self._cache_launch_allowed = False
            self._last_duplicate_signature = None
            self.current_cache_folder = None
            
            self.create_inputs_tab()
            self.create_visuals_tab()
            self.create_physics_tab()
            self.create_directories_tab()

            self._initializing_profiles = False
            self._load_all_custom_profiles()
            
            self.cb_fasta.currentTextChanged.connect(self.update_live_validators)
            self.cb_hdf5.currentTextChanged.connect(self.update_live_validators)
            self.cb_score_mode.currentTextChanged.connect(self.update_live_validators)
            self.cb_norm_mode.currentTextChanged.connect(self.update_live_validators)
            self.line_ref.textChanged.connect(self.update_live_validators)
            self.spin_thresh.valueChanged.connect(self.update_live_validators)
            self.spin_top.valueChanged.connect(self.update_live_validators)
            self.cb_msa.currentTextChanged.connect(self.update_live_validators)
            
            self.update_live_validators()
            self.setup_tips()

        def _read_custom_settings(self):
            try:
                with open(DEFAULT_SETTINGS_FILE, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    raise ValueError("the JSON root must be an object")
                if "SAVED_CONFIG_DIR" in data:
                    data["SAVED_CONFIG_DIR"] = _migrate_saved_config_dir(
                        data["SAVED_CONFIG_DIR"]
                    )
                return dict(data)
            except FileNotFoundError:
                return {}
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"Failed to load custom settings from {DEFAULT_SETTINGS_FILE}: {error}")
                return {}

        def _saved_config_root(self):
            widget = self.inputs.get("SAVED_CONFIG_DIR")
            value = widget.text() if widget is not None else globals().get(
                "SAVED_CONFIG_DIR", SAVED_CONFIG_DIR
            )
            return _resolved_saved_config_root(value)

        def _add_profile_selector(
            self,
            tab_id,
            form_layout,
            *,
            label_width=CONFIG_FIELD_LABEL_WIDTH,
        ):
            container = QWidget()
            row_layout = QHBoxLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)

            selector = DynamicComboBox(
                lambda selected_tab=tab_id: self._refresh_profile_combo(selected_tab)
            )
            name_input = QLineEdit()
            name_input.setPlaceholderText("New profile name")
            name_input.setVisible(False)
            folder_button = QPushButton("📂")
            folder_button.setFixedWidth(30)
            folder_button.setToolTip("Open this tab's saved config folder")
            row_layout.addWidget(selector, 1)
            row_layout.addWidget(name_input, 1)
            row_layout.addWidget(folder_button)

            label = QLabel("Saved Config:")
            label.setFixedWidth(label_width)
            form_layout.addRow(label, container)

            def open_profile_folder(checked=False, selected_tab=tab_id):
                folder = self._saved_config_root() / selected_tab
                os.makedirs(folder, exist_ok=True)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

            folder_button.clicked.connect(open_profile_folder)

            self.profile_selectors[tab_id] = selector
            self.profile_name_inputs[tab_id] = name_input
            self.profile_folder_buttons[tab_id] = folder_button
            self.profile_labels[tab_id] = label
            self._profile_previous_selection[tab_id] = "(custom)"
            self._refresh_profile_combo(tab_id)
            selector.currentTextChanged.connect(
                lambda text, selected_tab=tab_id: self._profile_selection_changed(
                    selected_tab, text
                )
            )

        def _profile_special_items(self, tab_id):
            items = ["(custom)"]
            if TAB_PROFILE_SPECS[tab_id]["allow_default"]:
                items.append("(default)")
            items.append("(new)")
            return items

        def _add_padded_separator(self, layout, object_name):
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            extra_padding = CONFIG_SEPARATOR_PADDING - CONFIG_TAB_ROW_SPACING
            wrapper_layout.setContentsMargins(0, extra_padding, 0, extra_padding)
            wrapper_layout.setSpacing(0)

            separator = QFrame()
            separator.setObjectName(object_name)
            separator.setFrameShape(QFrame.Shape.NoFrame)
            separator.setFixedHeight(CONFIG_SEPARATOR_THICKNESS)
            separator.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            separator.setStyleSheet("background-color: #9e9e9e; border: none;")
            wrapper_layout.addWidget(separator)
            if isinstance(layout, QFormLayout):
                layout.addRow(wrapper)
            else:
                layout.addWidget(wrapper)
            return separator

        def _add_profile_separator(self, tab_id, layout):
            separator = self._add_padded_separator(
                layout, f"{tab_id}_saved_config_separator"
            )
            self.profile_separators[tab_id] = separator

        def _refresh_profile_combo(self, tab_id):
            selector = self.profile_selectors[tab_id]
            current = selector.currentText() or "(custom)"
            items = _discover_profile_names(self._saved_config_root(), tab_id)
            items.extend(self._profile_special_items(tab_id))
            disappeared = current not in items

            selector.blockSignals(True)
            try:
                selector.clear()
                selector.addItems(items)
                selector.setCurrentText("(custom)" if disappeared else current)
            finally:
                selector.blockSignals(False)

            if (
                disappeared
                and current != "(custom)"
                and not self._initializing_profiles
                and not self._profile_loading
            ):
                self._profile_selection_changed(tab_id, "(custom)")

        def _profile_path(self, tab_id, profile_name):
            folder = self._saved_config_root() / tab_id
            candidate = folder / f"{profile_name}.json"
            if candidate.exists():
                return candidate
            try:
                for entry in folder.iterdir():
                    if (
                        entry.is_file()
                        and entry.suffix.lower() == ".json"
                        and entry.stem.casefold() == str(profile_name).casefold()
                    ):
                        return entry
            except (FileNotFoundError, NotADirectoryError, OSError):
                pass
            return candidate

        def _normalize_profile_data(self, tab_id, raw_data, *, allow_extra=False):
            if not isinstance(raw_data, dict):
                raise ValueError("the JSON root must be an object")

            defaults = TAB_PROFILE_SPECS[tab_id]["defaults"]
            unknown = set(raw_data) - set(defaults)
            if unknown and not allow_extra:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unknown setting key(s): {names}")

            normalized = dict(defaults)
            for key in defaults:
                if key not in raw_data:
                    continue
                value = raw_data[key]
                default = defaults[key]
                try:
                    if default is None:
                        if value is None or str(value).strip().lower() in {"", "none"}:
                            value = None
                        else:
                            value = float(value)
                    elif isinstance(default, bool):
                        if isinstance(value, bool):
                            pass
                        elif str(value).strip().lower() in {"true", "1", "t", "y", "yes"}:
                            value = True
                        elif str(value).strip().lower() in {"false", "0", "f", "n", "no"}:
                            value = False
                        else:
                            raise ValueError("expected true or false")
                    elif isinstance(default, int):
                        if isinstance(value, bool) or float(value) != int(float(value)):
                            raise ValueError("expected an integer")
                        value = int(float(value))
                    elif isinstance(default, float):
                        if isinstance(value, bool):
                            raise ValueError("expected a number")
                        value = float(value)
                    else:
                        if not isinstance(value, str):
                            raise ValueError("expected text")
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"invalid value for {key}: {error}") from error

                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"invalid value for {key}: expected a finite number")
                if key in PROFILE_ENUM_VALUES and value not in PROFILE_ENUM_VALUES[key]:
                    allowed = ", ".join(sorted(PROFILE_ENUM_VALUES[key]))
                    raise ValueError(f"invalid value for {key}; expected one of: {allowed}")
                if value is not None and key in PROFILE_RANGES:
                    minimum, maximum = PROFILE_RANGES[key]
                    if value < minimum or value > maximum:
                        raise ValueError(
                            f"invalid value for {key}; expected {minimum} to {maximum}"
                        )
                normalized[key] = value

            if (
                normalized.get("ALIGNMENT_SCORE") == "local"
                and normalized.get("NORM_MODE") == "alignment_length"
            ):
                raise ValueError(
                    "NORM_MODE alignment_length is unavailable for local alignment scores"
                )

            color_keys = {
                "TEXT_COLOR", "INITIAL_NODE_COLOR", "HOVER_COLOR",
                "CONNECTED_NODE_COLOR", "EDGE_COLOR", "NODE_BOUNDARY_COLOR",
            }
            for key in color_keys.intersection(normalized):
                if not QColor(normalized[key]).isValid():
                    raise ValueError(f"invalid color value for {key}: {normalized[key]}")
            return normalized

        def _custom_profile_data(self, tab_id):
            defaults = TAB_PROFILE_SPECS[tab_id]["defaults"]
            raw_data = {
                key: self._custom_settings[key]
                for key in defaults
                if key in self._custom_settings
            }
            try:
                return self._normalize_profile_data(tab_id, raw_data)
            except ValueError as error:
                print(f"Invalid custom settings for {tab_id}; using defaults: {error}")
                return dict(defaults)

        def _named_profile_data(self, tab_id, profile_name):
            path = self._profile_path(tab_id, profile_name)
            with open(path, "r", encoding="utf-8") as handle:
                raw_data = json.load(handle)
            return self._normalize_profile_data(tab_id, raw_data)

        def _set_widget_profile_value(self, key, value):
            widget = self.inputs[key]
            if isinstance(widget, OptionalNoScrollDoubleSpinBox):
                widget.setOptionalValue(value)
            elif isinstance(widget, QComboBox):
                if key == "LAYOUT_DEVICE_SELECTION":
                    index = widget.findData(value)
                    if index < 0:
                        widget.addItem(f"Unavailable saved device [{value}]", value)
                        index = widget.count() - 1
                    widget.setCurrentIndex(index)
                else:
                    text_value = str(value)
                    if key in {"NODE_FASTA_FILE", "MSA_FILE", "INPUT_HDF5"}:
                        text_value = os.path.basename(text_value)
                        if text_value and widget.findText(text_value) < 0:
                            widget.addItem(text_value)
                    widget.setCurrentText(text_value)
            elif isinstance(widget, QPushButton) and widget.isCheckable():
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText("" if value is None else str(value))
            elif hasattr(widget, "setChecked"):
                widget.setChecked(bool(value))
            else:
                raise TypeError(f"Unsupported settings widget for {key}")

        def _apply_profile_data(self, tab_id, data, *, read_only=False):
            self._set_profile_content_enabled(tab_id, True)
            self._profile_loading = True
            try:
                for key in TAB_PROFILE_SPECS[tab_id]["defaults"]:
                    self._set_widget_profile_value(key, data[key])
            finally:
                self._profile_loading = False

            if tab_id == "inputs_outputs":
                self.update_norm_mode_options()
            if hasattr(self, "update_live_validators"):
                self.update_live_validators()
            self._set_profile_content_enabled(tab_id, not read_only)

        def _set_profile_content_enabled(self, tab_id, enabled):
            content = self.profile_content_widgets[tab_id]
            roots = list(content) if isinstance(content, (list, tuple)) else [content]
            seen = set()
            color_swatches = set(self.color_swatches.values())
            for root in roots:
                candidates = [root, *root.findChildren(QWidget)]
                for widget in candidates:
                    identity = id(widget)
                    if identity in seen or widget in color_swatches:
                        continue
                    seen.add(identity)
                    if isinstance(widget, QLabel):
                        current_style = widget.styleSheet()
                        if PROFILE_DISABLED_LABEL_STYLESHEET not in current_style:
                            widget.setStyleSheet(
                                f"{current_style.rstrip()}\n"
                                f"{PROFILE_DISABLED_LABEL_STYLESHEET}".strip()
                            )
                    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                        current_style = widget.styleSheet()
                        if PROFILE_DISABLED_SPINBOX_STYLESHEET not in current_style:
                            widget.setStyleSheet(
                                f"{current_style.rstrip()}\n"
                                f"{PROFILE_DISABLED_SPINBOX_STYLESHEET}".strip()
                            )
                    elif isinstance(widget, QPushButton) and widget.isCheckable():
                        current_style = widget.styleSheet()
                        if PROFILE_DISABLED_TOGGLE_STYLESHEET not in current_style:
                            widget.setStyleSheet(
                                f"{current_style.rstrip()}\n"
                                f"{PROFILE_DISABLED_TOGGLE_STYLESHEET}".strip()
                            )
                    elif isinstance(widget, (QLineEdit, QComboBox, QPushButton)):
                        apply_gated_input_palette(widget)

            for root in roots:
                root.setEnabled(enabled)

        def _set_new_profile_field_visible(self, tab_id, visible):
            name_input = self.profile_name_inputs[tab_id]
            name_input.setVisible(visible)
            if visible:
                name_input.setFocus()
            else:
                name_input.clear()

        def _set_profile_selection(self, tab_id, text):
            selector = self.profile_selectors[tab_id]
            selector.blockSignals(True)
            try:
                selector.setCurrentText(text)
            finally:
                selector.blockSignals(False)

        def _profile_selection_changed(self, tab_id, text):
            if self._initializing_profiles or self._profile_loading or not text:
                return
            previous = self._profile_previous_selection.get(tab_id, "(custom)")

            if text == "(new)":
                self._set_new_profile_field_visible(tab_id, True)
                self._set_profile_content_enabled(tab_id, True)
                self._profile_previous_selection[tab_id] = text
                return

            self._set_new_profile_field_visible(tab_id, False)
            try:
                if text == "(custom)":
                    data = self._custom_profile_data(tab_id)
                    read_only = False
                elif text == "(default)":
                    if not TAB_PROFILE_SPECS[tab_id]["allow_default"]:
                        raise ValueError("this tab does not provide a default profile")
                    data = dict(TAB_PROFILE_SPECS[tab_id]["defaults"])
                    read_only = True
                else:
                    data = self._named_profile_data(tab_id, text)
                    read_only = False
                self._apply_profile_data(tab_id, data, read_only=read_only)
                self._profile_previous_selection[tab_id] = text
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                self._set_profile_selection(tab_id, previous)
                self._set_new_profile_field_visible(tab_id, previous == "(new)")
                QMessageBox.critical(
                    self,
                    "Saved Config Error",
                    f"Could not load profile '{text}':\n{error}",
                )

        def _load_all_custom_profiles(self):
            for tab_id in (
                "directories", "inputs_outputs", "visual_effects", "simulation_physics"
            ):
                self._set_profile_selection(tab_id, "(custom)")
                self._profile_previous_selection[tab_id] = "(custom)"
                self._set_new_profile_field_visible(tab_id, False)
                self._apply_profile_data(tab_id, self._custom_profile_data(tab_id))

        def _saved_config_directory_committed(self):
            if self._initializing_profiles or self._profile_loading:
                return
            for tab_id in TAB_PROFILE_SPECS:
                self._set_profile_selection(tab_id, "(custom)")
                self._profile_previous_selection[tab_id] = "(custom)"
                self._set_new_profile_field_visible(tab_id, False)
                self._refresh_profile_combo(tab_id)
            self._load_all_custom_profiles()

        def closeEvent(self, event):
            self._cache_hash_request_id += 1
            for worker in self._cache_hash_workers.values():
                worker.requestInterruption()
            for worker in self._cache_hash_workers.values():
                worker.wait(5000)
            super().closeEvent(event)
            
        def run_consistency_check(self):
            import h5py
            from utilities.FASTA_Sanitization import sanitize_header
            
            # 1. Define the file names by grabbing them from the UI dropdowns
            fasta_file = self.cb_fasta.currentText()
            hdf5_file = self.cb_hdf5.currentText()
            msa_file = self.cb_msa.currentText()
            
            # 2. Halt if the user hasn't selected the required files
            if not fasta_file or not hdf5_file:
                return
            
            # 3. Build the paths safely
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), fasta_file)
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), hdf5_file)
            msa_path = os.path.join(self.inputs["MSA_DIR"].text(), msa_file) if msa_file else None
                
            self.tip_panel.setText("Running Consistency Check...")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            
            try:
                fasta_headers = _load_consistency_fasta_headers(fasta_path)
                
                with h5py.File(hdf5_path, "r") as hf:
                    raw_headers = hf['headers'][:]
                    headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                    
                net_headers_set = set(headers)
                missing_nodes = [
                    header for header in fasta_headers
                    if header not in net_headers_set
                ]
                
                num_matched = len(fasta_headers) - len(missing_nodes)
                num_missing = len(missing_nodes)
                
                msg = f"FASTA vs HDF5:\nMatched: {num_matched} of {len(fasta_headers)} | Missing: {num_missing}"
                
                if missing_nodes:
                    msg = f"ERROR: FASTA is NOT a subset of HDF5.\n{msg}\nMissing examples: {', '.join(missing_nodes[:5])}"
                else:
                    msg = f"SUCCESS: FASTA is a strict subset of HDF5.\n{msg}"
                
                if msa_path and os.path.exists(msa_path):
                    msa_headers = set(_load_consistency_msa_headers(msa_path))

                    msa_missing = [
                        header for header in fasta_headers if header not in msa_headers
                    ]
                    msa_matched = len(fasta_headers) - len(msa_missing)
                    
                    msa_msg = (
                        f"FASTA vs MSA:\nMatched: {msa_matched} of "
                        f"{len(fasta_headers)} network headers | Missing: {len(msa_missing)}"
                    )
                    
                    if msa_missing:
                        msg += (
                            f"\n\nWARNING: MSA coverage is incomplete.\n{msa_msg}\n"
                            f"Missing examples: {', '.join(msa_missing[:5])}\n"
                            "Missing nodes remain plotted but are excluded from "
                            "alignment-dependent analyses."
                        )
                    else:
                        msg += f"\n\nSUCCESS: MSA covers all FASTA nodes.\n{msa_msg}"
                
                # Check Reference ID if provided
                ref_id = self.line_ref.text().strip()
                if ref_id:
                    # Proceed with normal matching only (Case-Insensitive)
                    ref_id_lower = sanitize_header(ref_id)[0].lower()
                    matched_refs = [h for h in fasta_headers if ref_id_lower in h.lower()]
                    if matched_refs:
                        msg += f"\n\nSUCCESS: Reference ID '{ref_id}' matched {len(matched_refs)} header(s) in FASTA."
                        for h in matched_refs[:5]:
                            msg += f"\n  - {h}"
                    else:
                        msg += f"\n\nWARNING: Reference ID '{ref_id}' NOT found in FASTA headers."
                
                self.tip_panel.setText(msg)
            
            except Exception as e:
                self.tip_panel.setText(f"Error during consistency check: {e}")
            
        def eventFilter(self, obj, event):
            from PySide6.QtCore import QEvent
            event_type = event.type()
            if event_type in (
                QEvent.Type.FocusIn,
                QEvent.Type.MouseButtonPress,
                QEvent.Type.ToolTip,
            ):
                if hasattr(self, 'tip_db'):
                    tip = self.tip_db.get(obj, None)
                    if not tip and isinstance(obj, QWidget):
                        tip = obj.toolTip()
                    if tip:
                        self.tip_panel.setText(tip)
                        if event_type == QEvent.Type.ToolTip:
                            return True
            return super().eventFilter(obj, event)

        def _register_tip_targets(self, widget, tip, *, overwrite=True):
            if widget is None:
                return
            targets = [widget]
            targets.extend(widget.findChildren(QWidget))
            for target in targets:
                if overwrite or target not in self.tip_db:
                    self.tip_db[target] = tip
                    target.installEventFilter(self)

        def setup_tips(self):
            self.tip_db_keys = {
                "SAVED_CONFIG": "Saved Config: Selects the settings profile used for this tab.\n(custom) uses the current viewer_settings.json values; (default) uses read-only built-in defaults; (new) creates a named profile; named entries load profiles from the Saved Config Directory.",
                "NODE_FASTA_FILE": "Primary FASTA file containing sequences visualized as nodes in the SSN.\nMust match sequences present in the selected network edges and multiple alignments.",
                "MSA_FILE": "Multiple sequence alignment file (.fasta, .h5, or _sparse.h5) for the sequence set.\nUsed to calculate positional conservation, gaps, and occupancy thresholds during analysis.",
                "INPUT_HDF5": "Network or similarity matrix file (.h5) containing pairwise sequence similarity scores and edge coordinates.\nMust contain alignment metrics for at least all sequences present in the active sequence set.",
                "ALIGNMENT_SCORE": "(For embedding SSNs) Specifies whether to use global (Needleman-Wunsch) or local (Smith-Waterman) scores.\nLocal alignment is recommended for multi-domain proteins; global alignment is best for full-length comparisons.",
                "NORM_MODE": "(For embedding SSNs) Normalization strategy for pairwise sequence alignment scores.\nNormalizes by alignment length, shorter sequence, longer sequence, or average sequence length to reduce length bias.",
                "ALIGNMENT_REFERENCE": "Substring or ID from a sequence header to identify the reference sequence in the alignment.\nUsed to anchor absolute relative residue numbering and mapping offsets across the entire network.",
                "ALIGNMENT_OFFSET": "Integer offset added to reference-anchored alignment residue positions (e.g. +10 shifts position 1 to 11).\nApplied only when the Alignment Reference ID resolves successfully in the alignment.",
                "SIMILARITY_THRESHOLD": "Minimum similarity score threshold (identity fraction, normalized score, or -Log10 E-Value) to retain an edge.\nEdges below this cutoff are filtered out and excluded from physics simulation and rendering.",
                "TOP_EDGE_PERCENT": "Alternative edge filter that retains only the top N% highest-scoring edges in the network.\nMaintains consistent network connectivity and density without manually tuning raw score cutoffs (overrides threshold).",
                "FILTER_MIN_OCCUPANCY": "Minimum percentage of non-gap characters required at an alignment column to retain it in residue analyses.\nColumns with occupancy below this percentage are excluded from logo and conservation calculations.",
                "TARGET_CACHE": "Target Cache: Compatibility-specific folder for layout cache files matching the selected sequence set and network.\nThe status reports whether a compatible target folder is available; the folder button opens the layout-cache root.",
                "TARGET_CACHE_FILE": "Selects a pre-computed 2D layout coordinate cache file (.h5) from the cache directory.\nInstantly restores previously computed node positions to bypass physics simulation.",
                "NEW_CACHE_NAME": "Specifies a custom filename when saving a new layout configuration iteration.\nOnly editable when Selected Cache File is set to '(New Layout Cache)'.",
                "NODE_SIZE": "Visual rendering diameter (in pixels) for each sequence node in the network plot.\nAdjust to optimize visual density; smaller node sizes are recommended for large networks.",
                "EDGE_WIDTH": "Line thickness (in pixels) of connection lines drawn between related sequence nodes.\nThinner lines reduce visual clutter in dense networks; thicker lines highlight strong relationships.",
                "EDGE_ALPHA": "Opacity of network edge lines, ranging from 0.0 (fully transparent) to 1.0 (opaque).\nLower opacity reveals underlying node clustering and density in highly connected graphs.",
                "TEXT_SIZE": "Font size used for rendering cluster annotations, node labels, and sequence IDs in the visualizer.\nAdjust to ensure labels remain legible without obstructing structural network features.",
                "TEXT_COLOR": "Color of viewer HUD text, slider values, and control labels. The top interaction instruction remains gray, and a nonzero Hidden Nodes count remains red.\nCan be specified as a standard color name or hex code (e.g. 'grey', '#333333').",
                "INITIAL_NODE_COLOR": "Baseline fill color applied to all nodes when the network is first loaded.\nServes as the default background color before custom cluster or metadata coloring is applied.",
                "HOVER_COLOR": "Highlight color applied to a node and its adjacent connections on hover or selection.\nProvides high-contrast interactive visual feedback in the viewer.",
                "CONNECTED_NODE_COLOR": "Border highlight color applied to neighboring nodes directly connected to the currently selected node.\nSet it to the same rendered RGBA color as Node Boundary Color to disable connected-node identification, border highlighting, and render promotion.",
                "EDGE_COLOR": "Color of connection lines (edges) drawn between similar nodes in the network.\nLighter or neutral colors reduce edge dominance in dense network clusters.",
                "NODE_BOUNDARY_COLOR": "Color of the outer border ring outline drawn around each sequence node.\nProvides visual contrast to cleanly separate adjacent and overlapping nodes.",
                "NODE_BOUNDARY_WIDTH": "Stroke width (in pixels) of the outer border ring outline drawn around each node.\nSetting a non-zero width helps distinguish overlapping nodes in dense clusters.",
                "LOW_RESOURCE_MODE": "Performance mode that simplifies graphics and hides edge lines during pan/zoom/drag interactions.\nSignificantly improves responsiveness and reduces rendering latency for large networks.",
                "PHYSICS_ENGINE": "Selects the simulation engine for 2D layout: Molecular Dynamics or Monte Carlo (SGLD).\nMolecular Dynamics uses deterministic force integration; Monte Carlo uses stochastic Langevin sampling.",
                "LAYOUT_DEVICE_SELECTION": "Selects the compute device used for physics layout generation (CPU, CUDA, XPU, MPS).\nAuto Benchmark tests CPU and available accelerators separately for each layout size class.",
                "SPRING_K": "Attractive Hookean spring constant pulling connected sequence nodes closer together.\nLarger values draw highly similar sequences into tighter, more compact clusters.",
                "COULOMB_K": "Repulsive constant controlling the electrostatic-like force pushing all nodes apart.\nLarger values push unrelated nodes and clusters apart, increasing family separation.",
                "COULOMB_CUTOFF": "Maximum spatial distance threshold beyond which node repulsive forces drop to zero.\nLower cutoffs accelerate computation and prevent distant clusters from exerting unnecessary forces.",
                "DAMPING": "Frictional resistance coefficient applied to node velocities to dissipate kinetic energy.\nValues near 1.0 allow smooth movement; lower values freeze oscillations quickly.",
                "DT": "Timestep size for each numerical integration step of the physics simulation.\nSmaller timesteps increase stability and precision; larger timesteps speed up convergence but may jitter.",
                "MAX_STEPS": "Maximum number of physics iterations the simulation engine will run before terminating.\nEnsure this is large enough to allow node positions to settle into a stable configuration.",
                "RMSD_THRESHOLD": "Root-Mean-Square Deviation convergence threshold for early simulation termination.\nIf average node displacement between consecutive steps falls below this value, layout halts as converged.",
                "PERCENTAGE_DROP_THRESHOLD": "Early termination threshold based on the rate of RMSD change over the moving window.\nTerminates simulation when layout change plateaus (set to 0 to disable).",
                "RMSD_WINDOW": "Number of simulation steps over which moving-average RMSD is calculated for plateau detection.\nSmoothes transient velocity spikes to ensure early termination triggers only on true convergence.",
                "ENABLE_PROGRESSIVE_SIMULATION": "Progressively lowers the similarity threshold in stages for massive connected components.\nHelps resolve fine-grained sub-clusters and prevents gridlock in large, dense components.",
                "PACKING_GEOMETRY": "Macro-level boundary packing geometry (Square or Circle) used to arrange disconnected components.\nControls how independent clusters are organized in the overall visualization window.",
                "PACKING_GRID_SIZE": "Base grid square unit size used for macro-grid component packing.\nControls spacing and separation between packed independent clusters in the final layout.",
                "SGLD_MIN_K": "Minimum number of nearest neighbors (K) retained per node in Monte Carlo / SGLD simulation.\nPrevents small or disconnected clusters from collapsing by maintaining a baseline neighborhood.",
                "SGLD_K_PERCENT": "Fraction of component nodes used to set dynamic negative sampling size (K) in SGLD mode.\nCalculated as max(SGLD_MIN_K, Fraction * total_nodes) to balance global topology and memory.",
                "SGLD_START_TEMP": "Starting temperature for Simulated Annealing in Monte Carlo / SGLD mode.\nHigher initial temperatures inject stochastic noise to help nodes escape local energy minima.",
                "SGLD_NOISE_SCALE": "Scaling factor for stochastic Brownian noise added to node velocities in SGLD simulation.\nAdjusts random thermal fluctuations to prevent premature layout freezing.",
                "UMAP_MODE": "Uses UMAP manifold learning to compute 2D coordinates directly from sequence distances.\nProvides fast non-linear dimensionality reduction as an alternative to iterative physics simulations.",
                "UMAP_NEIGHBORS": "Size of the local neighborhood (n_neighbors) used by UMAP to learn manifold topology.\nSmaller values emphasize local sub-clusters; larger values preserve broad global relationships.",
                "UMAP_MIN_DIST": "Minimum distance between points in low-dimensional UMAP space (0.0 to 1.0).\nLower values produce tight, dense point clusters; larger values distribute nodes more evenly.",
                "FASTA_DIR": "Directory containing input FASTA files for sequence sets and subsets.\nPopulates the Sequence Set dropdown in the Inputs tab.",
                "MSA_DIR": "Directory containing multiple sequence alignment files (.fasta, .h5, or _sparse.h5).\nPopulates the MSA dropdown in the Inputs tab.",
                "HDF5_DIR": "Directory containing HDF5 pairwise sequence similarity scores and network edge files.\nPopulates the Network Edges dropdown in the Inputs tab.",
                "SAVED_LAYOUT_DIR": "Directory where calculated 2D layout coordinate files and network metadata (.h5) are saved and loaded.\nServes as the layout cache to avoid recalculating layouts when reopening networks.",
                "SETTING_EXPORT_DIR": "Directory where command-line layout generation JSON settings are exported.",
                "METADATA_DIR": "Directory where uploaded node metadata spreadsheets and CSV files are stored and loaded.\nUsed for custom node coloring, categorization, and annotation in the visualizer.",
                "PRINT_SAVE_DIR": "Directory where high-resolution image snapshots and vector graphics (PDF, PNG, SVG) are exported.\nEnsure this path is writable with sufficient disk space for graphic outputs.",
                "SEQUENCE_EXPORT_DIR": "Directory where dynamically split or extracted sequence subset FASTA files are saved.\nUsed by sub-cluster extraction and downstream sequence analyses.",
                "CLUSTER_LABEL_DIR": "Directory where exported cluster metadata, sequence IDs, and automated cluster labels are saved.\nUseful for downstream annotation pipelines and external inspection.",
                "HEADER_LIST_DIR": "Directory containing text files with lists of sequence headers matching network query criteria.\nUsed to store and track sequence cohorts identified in the visualizer.",
                "LOGO_DIR": "Directory where exported sequence logos representing consensus conservation are saved.\nOutputs PNG or vector graphics for publication and presentation.",
                "STRUCTURES_DIR": "Directory where predicted 3D structures and PDB/mmCIF files are stored and loaded.\nUsed by the 3D structure viewer module for structural superposition and inspection.",
                "SAVED_CONFIG_DIR": "Directory containing named per-tab configuration profiles.\nRelative paths are resolved from the project root; the location itself is always stored in viewer_settings.json."
            }
            
            self.tip_db = {}
            for key, tip in self.tip_db_keys.items():
                if key in self.labels:
                    self._register_tip_targets(self.labels[key], tip)
                
                if key in self.inputs:
                    self._register_tip_targets(self.inputs[key], tip)
                    
            for key, tip in self.tip_db_keys.items():
                if key in self.inputs:
                    widget = self.inputs[key]
                    parent = widget.parentWidget()
                    if parent and parent.objectName() == "wrapper":
                        self._register_tip_targets(parent, tip, overwrite=False)

            saved_config_tip = self.tip_db_keys["SAVED_CONFIG"]
            for tab_id, selector in self.profile_selectors.items():
                self._register_tip_targets(
                    self.profile_labels.get(tab_id), saved_config_tip
                )
                self._register_tip_targets(
                    selector.parentWidget(), saved_config_tip
                )

            target_cache_tip = self.tip_db_keys["TARGET_CACHE"]
            self._register_tip_targets(
                self.labels.get("TARGET_CACHE"), target_cache_tip
            )
            self._register_tip_targets(
                self.lbl_cache_tracker.parentWidget(), target_cache_tip
            )

            # Any remaining native Qt tooltip is also displayed in the shared
            # panel, and its native popup is suppressed by eventFilter().
            for widget in self.findChildren(QWidget):
                if widget.toolTip():
                    self._register_tip_targets(
                        widget, widget.toolTip(), overwrite=False
                    )
        
        def _toggle_new_cache_input(self, text):
            is_new_layout = text == "(New Layout Cache)"
            self.line_new_cache.setVisible(is_new_layout)
            self.line_new_cache.setEnabled(is_new_layout)
            self.btn_export_layout.setEnabled(
                is_new_layout and self._cache_launch_allowed
            )
            if is_new_layout:
                self.line_new_cache.setFocus()
            else:
                self.line_new_cache.clear()

        def _refresh_cache_file_combo(self):
            folder_path = self.current_cache_folder
            if not folder_path or not os.path.isdir(folder_path):
                return

            try:
                cache_files = [
                    (entry.name, entry.stat().st_mtime)
                    for entry in os.scandir(folder_path)
                    if entry.is_file() and entry.name.lower().endswith(".h5")
                ]
                cache_files.sort(key=lambda item: item[1], reverse=True)
                saved_layout_dir = os.path.abspath(
                    self.inputs["SAVED_LAYOUT_DIR"].text()
                )
                cache_items = [
                    (
                        filename,
                        cache_manifest.relative_cache_path(
                            saved_layout_dir, folder_path, filename
                        ),
                    )
                    for filename, _mtime in cache_files
                ]
            except (OSError, cache_manifest.CacheManifestError):
                return

            current_text = self.cb_cache_file.currentText()
            current_data = self.cb_cache_file.currentData()
            self.cb_cache_file.blockSignals(True)
            try:
                self.cb_cache_file.clear()
                for filename, relative_path in cache_items:
                    self.cb_cache_file.addItem(filename, relative_path)
                self.cb_cache_file.addItem("(New Layout Cache)", None)

                if current_text == "(New Layout Cache)":
                    selected_index = self.cb_cache_file.count() - 1
                else:
                    selected_index = self.cb_cache_file.findData(current_data)
                    if selected_index < 0:
                        selected_index = 0
                self.cb_cache_file.setCurrentIndex(selected_index)
            finally:
                self.cb_cache_file.blockSignals(False)

            self.line_new_cache.setPlaceholderText(
                self._default_new_cache_name(folder_path)
            )
            self._toggle_new_cache_input(self.cb_cache_file.currentText())

        def refresh_combo(self, combo, dir_key, ext_list):
            import os  # Moved here to ensure it's loaded before use
            if dir_key not in self.inputs: 
                return
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItem("")
            dir_path = self.inputs[dir_key].text()
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if any(f.endswith(ext) for ext in ext_list)]
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            combo.blockSignals(False)
            self.update_live_validators()

        def _set_cache_unavailable(self, message, color="gray"):
            self._cache_launch_allowed = False
            self.btn_save_run.setEnabled(False)
            self.current_cache_folder = None
            self.lbl_cache_tracker.setText(message)
            self.lbl_cache_tracker.setStyleSheet(f"color: {color};")
            self.cb_cache_file.blockSignals(True)
            self.cb_cache_file.clear()
            self.cb_cache_file.setEnabled(False)
            self.cb_cache_file.blockSignals(False)
            self._toggle_new_cache_input("")
            self.btn_open_target_folder.setEnabled(False)

        def _cache_paths_from_inputs(self):
            fasta_name = self.cb_fasta.currentText().strip()
            network_name = self.cb_hdf5.currentText().strip()
            if not fasta_name or not network_name:
                return None, None
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), fasta_name)
            network_path = os.path.join(self.inputs["HDF5_DIR"].text(), network_name)
            return os.path.abspath(fasta_path), os.path.abspath(network_path)

        def _cache_setting_values(self):
            top_value = self.spin_top.optionalValue()
            threshold_value = self.spin_thresh.optionalValue()
            return {
                "alignment_score": self.cb_score_mode.currentText() or None,
                "normalization": self.cb_norm_mode.currentText() or None,
                "umap_mode": self.check_umap.isChecked(),
                "umap_neighbors": self.spin_umap_k.value(),
                "top_edge_percent": top_value,
                "similarity_threshold": threshold_value,
            }

        def _set_network_type_controls(self, network_type):
            is_blast = network_type == "blast"
            self.cb_score_mode.blockSignals(True)
            self.cb_norm_mode.blockSignals(True)
            self.cb_score_mode.setEnabled(not is_blast)
            self.cb_norm_mode.setEnabled(not is_blast)
            if is_blast:
                self.cb_score_mode.setCurrentIndex(-1)
                self.cb_norm_mode.setCurrentIndex(-1)
            else:
                if self.cb_score_mode.currentIndex() == -1:
                    self.cb_score_mode.setCurrentText("global")
                if self.cb_norm_mode.currentIndex() == -1:
                    self.cb_norm_mode.setCurrentText("alignment_length")
            self.cb_score_mode.blockSignals(False)
            self.cb_norm_mode.blockSignals(False)

        def _default_new_cache_name(self, folder_path):
            return cache_manifest.next_cache_version_filename(folder_path)

        def _apply_cache_discovery(self, records):
            sequence_path, network_path = self._cache_paths_from_inputs()
            if not sequence_path or not network_path:
                self._set_cache_unavailable("Target Cache: Missing input files")
                return

            network_type = records["network_type"]
            self._set_network_type_controls(network_type)
            settings = self._cache_setting_values()
            try:
                compatibility = cache_manifest.build_compatibility(
                    records["sequence"]["sha256"],
                    records["network"]["sha256"],
                    network_type,
                    **settings,
                )
                saved_layout_dir = os.path.abspath(
                    self.inputs["SAVED_LAYOUT_DIR"].text()
                )
                canonical_name = cache_manifest.build_canonical_cache_name(
                    sequence_path,
                    network_path,
                    network_type,
                    **settings,
                )
                canonical_folder = os.path.join(saved_layout_dir, canonical_name)
                matches = cache_manifest.find_matching_manifest_folders(
                    saved_layout_dir, compatibility
                )
            except Exception as error:
                self._set_cache_unavailable(
                    f"Cache compatibility error: {error}", "#d32f2f"
                )
                return

            self.cb_cache_file.blockSignals(True)
            self.cb_cache_file.clear()
            self.line_new_cache.clear()

            if len(matches) > 1:
                folders = tuple(item["folder"] for item in matches)
                self.current_cache_folder = None
                self._cache_launch_allowed = False
                self.btn_save_run.setEnabled(False)
                self.cb_cache_file.setEnabled(False)
                self._toggle_new_cache_input("")
                self.btn_open_target_folder.setEnabled(False)
                self.lbl_cache_tracker.setText(
                    f"Error: {len(folders)} compatible cache folders found"
                )
                self.lbl_cache_tracker.setStyleSheet(
                    "color: #d32f2f; font-weight: bold;"
                )
                if folders != self._last_duplicate_signature:
                    print("ERROR: Multiple compatible cache folders were found:")
                    for folder in folders:
                        print(f"  - {folder}")
                self._last_duplicate_signature = folders
                self.cb_cache_file.blockSignals(False)
                return

            self._last_duplicate_signature = None
            if matches:
                active_folder = matches[0]["folder"]
                self.current_cache_folder = active_folder
                cache_files = [
                    entry.name
                    for entry in os.scandir(active_folder)
                    if entry.is_file() and entry.name.lower().endswith(".h5")
                ]
                cache_files.sort(
                    key=lambda name: os.path.getmtime(os.path.join(active_folder, name)),
                    reverse=True,
                )
                for filename in cache_files:
                    relative_path = cache_manifest.relative_cache_path(
                        saved_layout_dir, active_folder, filename
                    )
                    self.cb_cache_file.addItem(filename, relative_path)
                self.lbl_cache_tracker.setText(
                    f"Compatible Folder: {os.path.basename(active_folder)}"
                )
                self.lbl_cache_tracker.setStyleSheet(
                    "color: green; font-weight: bold;"
                )
                self.btn_open_target_folder.setEnabled(True)
            else:
                active_folder = canonical_folder
                self.current_cache_folder = active_folder
                self.lbl_cache_tracker.setText(
                    f"Target Folder: {canonical_name} [Needs Computing]"
                )
                self.lbl_cache_tracker.setStyleSheet("color: #d32f2f;")
                self.btn_open_target_folder.setEnabled(False)

            self.line_new_cache.setPlaceholderText(
                self._default_new_cache_name(active_folder)
            )
            self.cb_cache_file.addItem("(New Layout Cache)", None)
            self.cb_cache_file.setEnabled(True)
            self.cb_cache_file.setCurrentIndex(0)
            self.cb_cache_file.blockSignals(False)
            self._cache_launch_allowed = True
            self.btn_save_run.setEnabled(True)
            self._toggle_new_cache_input(self.cb_cache_file.currentText())

        def _cache_hash_completed(self, request_id, records, error):
            worker = self._cache_hash_workers.pop(request_id, None)
            if worker is not None:
                worker.deleteLater()
            if request_id != self._cache_hash_request_id:
                return
            self._cache_hash_pending_keys = None
            if error:
                self._set_cache_unavailable(f"Cache hashing failed: {error}", "#d32f2f")
                return
            sequence_path, network_path = self._cache_paths_from_inputs()
            try:
                self._cache_hash_cache[cache_manifest.file_cache_key(sequence_path)] = records["sequence"]
                self._cache_hash_cache[cache_manifest.file_cache_key(network_path)] = records["network"]
            except (OSError, TypeError):
                self.update_live_validators()
                return
            self._apply_cache_discovery(records)

        def _request_cache_discovery(self):
            sequence_path, network_path = self._cache_paths_from_inputs()
            if not sequence_path or not network_path:
                self._cache_hash_request_id += 1
                self._cache_hash_pending_keys = None
                self._set_cache_unavailable("Target Cache: Missing FASTA or HDF5")
                return
            if not os.path.isfile(sequence_path) or not os.path.isfile(network_path):
                self._cache_hash_request_id += 1
                self._cache_hash_pending_keys = None
                self._set_cache_unavailable("Target Cache: Selected input file is missing", "#d32f2f")
                return

            try:
                sequence_key = cache_manifest.file_cache_key(sequence_path)
                network_key = cache_manifest.file_cache_key(network_path)
            except OSError as error:
                self._cache_hash_request_id += 1
                self._cache_hash_pending_keys = None
                self._set_cache_unavailable(f"Cache input error: {error}", "#d32f2f")
                return

            cached_records = {
                "sequence": self._cache_hash_cache.get(sequence_key),
                "network": self._cache_hash_cache.get(network_key),
            }
            if all(cached_records.values()):
                try:
                    cached_records["network_type"] = (
                        cache_manifest.validate_network_schema(network_path).network_type
                    )
                    self._apply_cache_discovery(cached_records)
                except Exception as error:
                    self._set_cache_unavailable(
                        f"Cache compatibility error: {error}", "#d32f2f"
                    )
                return

            pending_keys = (sequence_key, network_key)
            if self._cache_hash_pending_keys == pending_keys:
                return

            for active_worker in self._cache_hash_workers.values():
                active_worker.requestInterruption()
            self._cache_hash_request_id += 1
            request_id = self._cache_hash_request_id
            self._cache_hash_pending_keys = pending_keys
            self._set_cache_unavailable("Checking input files…")
            worker = CacheHashWorker(
                request_id,
                sequence_path,
                network_path,
                cached_records=cached_records,
            )
            self._cache_hash_workers[request_id] = worker
            worker.completed.connect(self._cache_hash_completed)
            worker.start()

        def create_inputs_tab(self):
            tab = QWidget()
            layout = QFormLayout(tab)
            layout.setContentsMargins(
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
            )
            layout.setHorizontalSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            layout.setVerticalSpacing(CONFIG_TAB_ROW_SPACING)
            self._add_profile_selector("inputs_outputs", layout)
            self._add_profile_separator("inputs_outputs", layout)
            self.profile_content_widgets["inputs_outputs"] = []
            
            def add_row(key, label_text, widget):
                lbl = QLabel(label_text)
                lbl.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
                layout.addRow(lbl, widget)
                self.labels[key] = lbl
                self.inputs[key] = widget

            # Helper for adding a combobox + dynamic folder button
            def add_row_with_dynamic_btn(key, label_text, combo, dir_key, default_dir):
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                btn = QPushButton("📂")
                btn.setFixedWidth(30)
                btn.setToolTip("Open Folder")
                
                def open_folder(checked):
                    import os
                    # Read from the directory input if it exists, otherwise use the default global
                    path = self.inputs[dir_key].text() if dir_key in self.inputs else globals().get(dir_key, default_dir)
                    abs_path = os.path.abspath(path)
                    os.makedirs(abs_path, exist_ok=True)
                    from PySide6.QtGui import QDesktopServices
                    from PySide6.QtCore import QUrl
                    QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                    
                btn.clicked.connect(open_folder)
                
                h_lay.addWidget(combo)
                h_lay.addWidget(btn)
                
                lbl = QLabel(label_text)
                lbl.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
                layout.addRow(lbl, container)
                self.labels[key] = lbl
                self.inputs[key] = combo 
            
            # --- Sequence Set Input ---
            # Use DynamicComboBox and bind the refresh function directly to the click event!
            self.cb_fasta = DynamicComboBox(lambda: self.refresh_combo(self.cb_fasta, "FASTA_DIR", ['.fasta']))
            seq_dir = globals().get("FASTA_DIR", os.path.join("Input_Files", "Sequence_Sets"))
            fasta_files = [f for f in os.listdir(seq_dir) if f.endswith('.fasta')] if os.path.exists(seq_dir) else []
            self.cb_fasta.addItems([""] + fasta_files)
            fasta_val = globals().get("NODE_FASTA_FILE") or ""
            if os.path.basename(fasta_val) in fasta_files:
                self.cb_fasta.setCurrentText(os.path.basename(fasta_val))
            add_row_with_dynamic_btn("NODE_FASTA_FILE", "Sequence Set / Subset (.fasta):", self.cb_fasta, "FASTA_DIR", seq_dir)
            
            # --- MSA Input ---
            self.cb_msa = DynamicComboBox(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            msa_dir_path = globals().get("MSA_DIR", os.path.join("Input_Files", "Multiple_Alignments"))
            msa_files = [f for f in os.listdir(msa_dir_path) if f.endswith('.fasta') or f.endswith('.h5')] if os.path.exists(msa_dir_path) else []
            self.cb_msa.addItems([""] + msa_files)
            msa_val = globals().get("MSA_FILE") or ""
            if os.path.basename(msa_val) in msa_files:
                self.cb_msa.setCurrentText(os.path.basename(msa_val))
            add_row_with_dynamic_btn("MSA_FILE", "MSA Input (.fasta / _sparse.h5):", self.cb_msa, "MSA_DIR", msa_dir_path)

            # --- HDF5 Input ---
            self.cb_hdf5 = DynamicComboBox(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
            hdf5_dir = globals().get("HDF5_DIR", os.path.join("Input_Files", "Networks_EValues"))
            hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')] if os.path.exists(hdf5_dir) else []
            self.cb_hdf5.addItems([""] + hdf5_files)
            hdf5_val = globals().get("INPUT_HDF5") or ""
            if os.path.basename(hdf5_val) in hdf5_files:
                self.cb_hdf5.setCurrentText(os.path.basename(hdf5_val))
            add_row_with_dynamic_btn("INPUT_HDF5", "Network Edges Input (.h5):", self.cb_hdf5, "HDF5_DIR", hdf5_dir)
            
            # --- Rest of Inputs ---
            # Use NoScrollComboBox here to prevent accidental scroll wheel changes
            self.cb_score_mode = NoScrollComboBox()
            self.cb_score_mode.addItems(["global", "local"])
            self.cb_score_mode.setCurrentText(str(globals().get("ALIGNMENT_SCORE", "global")))
            add_row("ALIGNMENT_SCORE", "Alignment Score Mode:", self.cb_score_mode)
            
            self.cb_norm_mode = NoScrollComboBox()
            add_row("NORM_MODE", "Normalization Mode:", self.cb_norm_mode)
            
            self.cb_score_mode.currentTextChanged.connect(self.update_norm_mode_options)
            self.update_norm_mode_options()
            
            initial_norm = str(globals().get("NORM_MODE", "alignment_length"))
            if self.cb_score_mode.currentText() == "local" and initial_norm == "alignment_length":
                initial_norm = "longer_sequence"
            self.cb_norm_mode.setCurrentText(initial_norm)
            
            ref_val = globals().get("ALIGNMENT_REFERENCE", "")
            self.line_ref = QLineEdit("" if ref_val in [None, "None"] else str(ref_val))

            ref_container = QWidget()
            ref_container.setObjectName("alignment_reference_wrapper")
            ref_layout = QHBoxLayout(ref_container)
            ref_layout.setContentsMargins(0, 0, 0, 0)
            ref_layout.addWidget(self.line_ref, 1)

            self.lbl_alignment_offset = QLabel("Alignment Offset:")
            self.spin_alignment_offset = QSpinBox()
            self.spin_alignment_offset.setRange(-1000000, 1000000)
            try:
                offset_value = int(globals().get("ALIGNMENT_OFFSET", 0) or 0)
            except (TypeError, ValueError):
                offset_value = 0
            self.spin_alignment_offset.setValue(offset_value)
            self.spin_alignment_offset.setAccelerated(True)
            self.spin_alignment_offset.setFixedWidth(100)
            self.spin_alignment_offset.setStyleSheet(
                "QSpinBox:disabled { background-color: #f0f0f0; color: #888; }"
            )
            ref_layout.addSpacing(12)
            ref_layout.addWidget(self.lbl_alignment_offset)
            ref_layout.addWidget(self.spin_alignment_offset)

            ref_label = QLabel("Alignment Reference ID:")
            ref_label.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            layout.addRow(ref_label, ref_container)
            self.labels["ALIGNMENT_REFERENCE"] = ref_label
            self.inputs["ALIGNMENT_REFERENCE"] = self.line_ref
            self.labels["ALIGNMENT_OFFSET"] = self.lbl_alignment_offset
            self.inputs["ALIGNMENT_OFFSET"] = self.spin_alignment_offset
            
            # --- UMAP Controls ---
            umap_container = QWidget()
            umap_container.setObjectName("wrapper")
            umap_layout = QHBoxLayout(umap_container)
            umap_layout.setContentsMargins(0, 0, 0, 0)
            
            self.check_umap = QPushButton()
            self.check_umap.setCheckable(True)
            self.check_umap.setFixedSize(60, 28)
            def switch_umap_style(checked, btn=self.check_umap):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            self.check_umap.toggled.connect(switch_umap_style)
            
            umap_mode_val = globals().get("UMAP_MODE", False)
            if isinstance(umap_mode_val, str):
                umap_mode_val = umap_mode_val.lower() in ['true', '1', 't', 'y', 'yes']
            self.check_umap.setChecked(bool(umap_mode_val))
            switch_umap_style(bool(umap_mode_val))
            
            lbl_k = QLabel("   UMAP Nearest Neighbors (k):")
            lbl_md = QLabel("   UMAP Min Distance:")
            
            self.spin_umap_k = NoScrollSpinBox()
            self.spin_umap_k.setRange(2, 500)
            self.spin_umap_k.setValue(int(globals().get("UMAP_NEIGHBORS") or 15))
            self.spin_umap_k.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            
            self.spin_umap_md = NoScrollDoubleSpinBox()
            self.spin_umap_md.setRange(0.0, 1.0)
            self.spin_umap_md.setSingleStep(0.1)
            self.spin_umap_md.setDecimals(2)
            self.spin_umap_md.setValue(float(globals().get("UMAP_MIN_DIST") or 0.1))
            self.spin_umap_md.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            # Match the taller spinboxes used beside sliders on the Visuals/Physics tabs.
            input_spinbox_height = max(
                self.spin_umap_k.sizeHint().height(),
                self.spin_umap_md.sizeHint().height(),
            )
            for spinbox in (
                self.spin_alignment_offset,
                self.spin_umap_k,
                self.spin_umap_md,
            ):
                spinbox.setFixedHeight(input_spinbox_height)
            
            # Apply styling for disabled states (grayed out)
            disabled_spinbox_style = "QSpinBox:disabled, QDoubleSpinBox:disabled { background-color: #f0f0f0; color: #888; }"
            disabled_label_style = "QLabel:disabled { color: #888; }"
            self.spin_umap_k.setStyleSheet(disabled_spinbox_style)
            self.spin_umap_md.setStyleSheet(disabled_spinbox_style)
            lbl_k.setStyleSheet(disabled_label_style)
            lbl_md.setStyleSheet(disabled_label_style)
            
            # Set initial enabled/disabled states based on current toggle value
            umap_enabled = self.check_umap.isChecked()
            self.spin_umap_k.setEnabled(umap_enabled)
            self.spin_umap_md.setEnabled(umap_enabled)
            lbl_k.setEnabled(umap_enabled)
            lbl_md.setEnabled(umap_enabled)
            
            def toggle_umap(state):
                self.spin_umap_k.setEnabled(state)
                self.spin_umap_md.setEnabled(state)
                lbl_k.setEnabled(state)
                lbl_md.setEnabled(state)
                self.update_live_validators()
                
            self.check_umap.toggled.connect(toggle_umap)
            self.spin_umap_k.valueChanged.connect(self.update_live_validators)
            
            umap_layout.addWidget(self.check_umap)
            umap_layout.addWidget(lbl_k)
            umap_layout.addWidget(self.spin_umap_k)
            umap_layout.addWidget(lbl_md)
            umap_layout.addWidget(self.spin_umap_md)
            
            layout.addRow("Enable UMAP Layout:", umap_container)
            self.labels["UMAP_MODE"] = layout.labelForField(umap_container)
            self.labels["UMAP_MODE"].setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            self.inputs["UMAP_MODE"] = self.check_umap
            self.inputs["UMAP_NEIGHBORS"] = self.spin_umap_k
            self.inputs["UMAP_MIN_DIST"] = self.spin_umap_md
            self.labels["UMAP_NEIGHBORS"] = lbl_k
            self.labels["UMAP_MIN_DIST"] = lbl_md
            
            self.spin_thresh = OptionalNoScrollDoubleSpinBox(-1000000000.0, 0.0)
            self.spin_thresh.setDecimals(5)
            self.spin_thresh.setRange(-1000000000.0, 1000000000.0)
            self.spin_thresh.setSingleStep(0.1)
            self.spin_thresh.setSpecialValueText(" ")
            self.spin_thresh.setOptionalValue(globals().get("SIMILARITY_THRESHOLD"))

            self.spin_top = OptionalNoScrollDoubleSpinBox(-0.01, 1.0)
            self.spin_top.setDecimals(2)
            self.spin_top.setRange(-0.01, 100.0)
            self.spin_top.setSingleStep(0.1)
            self.spin_top.setSpecialValueText(" ")
            self.spin_top.setOptionalValue(globals().get("TOP_EDGE_PERCENT"))

            self.spin_min_occ = NoScrollDoubleSpinBox()
            self.spin_min_occ.setDecimals(2)
            self.spin_min_occ.setRange(0.0, 100.0)
            self.spin_min_occ.setSingleStep(1.0)
            self.spin_min_occ.setValue(float(globals().get("FILTER_MIN_OCCUPANCY") or 10.0))

            for spinbox in (
                self.spin_thresh,
                self.spin_top,
                self.spin_min_occ,
            ):
                spinbox.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                spinbox.setFixedHeight(input_spinbox_height)

            filter_container = QWidget()
            filter_container.setObjectName("wrapper")
            filter_layout = QHBoxLayout(filter_container)
            filter_layout.setContentsMargins(0, 0, 0, 0)

            lbl_thresh = QLabel("Similarity Threshold:")
            lbl_thresh.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            lbl_top = QLabel("   Top Edge %:")
            lbl_min_occ = QLabel("   Min Occupancy %:")

            filter_layout.addWidget(self.spin_thresh, 1)
            filter_layout.addWidget(lbl_top)
            filter_layout.addWidget(self.spin_top, 1)
            filter_layout.addWidget(lbl_min_occ)
            filter_layout.addWidget(self.spin_min_occ, 1)

            layout.addRow(lbl_thresh, filter_container)

            self.labels["SIMILARITY_THRESHOLD"] = lbl_thresh
            self.inputs["SIMILARITY_THRESHOLD"] = self.spin_thresh
            self.labels["TOP_EDGE_PERCENT"] = lbl_top
            self.inputs["TOP_EDGE_PERCENT"] = self.spin_top
            self.labels["FILTER_MIN_OCCUPANCY"] = lbl_min_occ
            self.inputs["FILTER_MIN_OCCUPANCY"] = self.spin_min_occ

            # Retain the previous internal attribute names for compatibility.
            self.line_thresh = self.spin_thresh
            self.line_top = self.spin_top
            self.line_min_occ = self.spin_min_occ
            
            # Horizontal layout for both statistics and histogram buttons
            btn_container = QWidget()
            btn_lay = QHBoxLayout(btn_container)
            btn_lay.setContentsMargins(0, 0, 0, 0)
            
            self.btn_stats = QPushButton("Compute Network Statistics")
            self.btn_stats.setStyleSheet("background-color: #2196F3; color: white;")
            self.btn_stats.clicked.connect(self.run_statistics)

            self.btn_hist = QPushButton("Histogram")
            self.btn_hist.setStyleSheet("background-color: #9C27B0; color: white;")
            self.btn_hist.clicked.connect(self.run_histogram)
            
            btn_lay.addWidget(self.btn_stats, 3)
            btn_lay.addWidget(self.btn_hist, 1)
            layout.addRow("", btn_container)

            self.cache_file_separator = self._add_padded_separator(
                layout, "cache_file_separator"
            )

            # --- Target Cache Tracker & Folder Button ---
            cache_container = QWidget()
            cache_lay = QHBoxLayout(cache_container)
            cache_lay.setContentsMargins(0, 0, 0, 0)
            
            self.lbl_cache_tracker = QLabel("Target Folder: None")
            self.lbl_cache_tracker.setStyleSheet("color: gray;")
            self.lbl_cache_tracker.setWordWrap(False)
            self.lbl_cache_tracker.setMinimumWidth(0)
            self.lbl_cache_tracker.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
            )
            
            self.btn_open_cache = QPushButton("📂")
            self.btn_open_cache.setFixedWidth(30)
            self.btn_open_cache.setToolTip("Open Target Cache Folder")
            
            def open_cache_folder(checked):
                import os
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                
                # Exclusively open the parent directory
                dir_input = self.inputs.get("SAVED_LAYOUT_DIR")
                path = dir_input.text() if dir_input else globals().get("SAVED_LAYOUT_DIR", os.path.join("Cache_Files", "Saved_Layouts"))
                
                abs_path = os.path.abspath(path)
                os.makedirs(abs_path, exist_ok=True)
                QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                
            self.btn_open_cache.clicked.connect(open_cache_folder)
            
            cache_lay.addWidget(self.lbl_cache_tracker, 1)
            cache_lay.addWidget(self.btn_open_cache)
            target_cache_label = QLabel("Target Cache:")
            target_cache_label.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            layout.addRow(target_cache_label, cache_container)
            self.labels["TARGET_CACHE"] = target_cache_label

            # Cache dropdown, conditional new-cache name, and folder button.
            target_container = QWidget()
            target_container.setObjectName("wrapper")
            target_lay = QHBoxLayout(target_container)
            target_lay.setContentsMargins(0, 0, 0, 0)

            self.cb_cache_file = DynamicComboBox(self._refresh_cache_file_combo)
            self.cb_cache_file.setEnabled(False)

            self.line_new_cache = QLineEdit()
            self.line_new_cache.setVisible(False)
            self.line_new_cache.setEnabled(False)
            
            self.btn_open_target_folder = QPushButton("📂")
            self.btn_open_target_folder.setFixedWidth(30)
            self.btn_open_target_folder.setToolTip("Open Specific Target Folder")
            self.btn_open_target_folder.setEnabled(False)  # Greyed out by default
            
            def open_target_folder(checked):
                import os
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                
                path = getattr(self, 'current_cache_folder', None)
                if path and os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                    
            self.btn_open_target_folder.clicked.connect(open_target_folder)

            target_lay.addWidget(self.cb_cache_file, 1)
            target_lay.addWidget(self.line_new_cache, 1)
            target_lay.addWidget(self.btn_open_target_folder)
            
            layout.addRow("Selected Cache File:", target_container)
            self.labels["TARGET_CACHE_FILE"] = layout.labelForField(target_container)
            self.labels["TARGET_CACHE_FILE"].setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            self.inputs["TARGET_CACHE_FILE"] = self.cb_cache_file
            self.inputs["NEW_CACHE_NAME"] = self.line_new_cache
            
            # Hook up the toggle switch
            self.cb_cache_file.currentTextChanged.connect(self._toggle_new_cache_input)

            self.tabs.addTab(tab, "Inputs && Outputs")
            
        def update_norm_mode_options(self):
            if not hasattr(self, 'cb_score_mode') or not hasattr(self, 'cb_norm_mode'):
                return
            current_norm = self.cb_norm_mode.currentText()
            self.cb_norm_mode.blockSignals(True)
            self.cb_norm_mode.clear()
            
            is_local = self.cb_score_mode.currentText() == "local"
            if is_local:
                self.cb_norm_mode.addItems(["shorter_sequence", "longer_sequence", "average_sequence"])
                if current_norm == "alignment_length":
                    current_norm = "longer_sequence"
            else:
                self.cb_norm_mode.addItems(["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"])
                
            self.cb_norm_mode.setCurrentText(current_norm)
            self.cb_norm_mode.blockSignals(False)

        def update_live_validators(self):
            has_fasta = bool(self.cb_fasta.currentText().strip())
            has_hdf5 = bool(self.cb_hdf5.currentText().strip())
            has_reference = bool(self.line_ref.text().strip())
            if hasattr(self, 'spin_alignment_offset'):
                self.spin_alignment_offset.setEnabled(has_reference)
            if hasattr(self, 'lbl_alignment_offset'):
                self.lbl_alignment_offset.setEnabled(has_reference)
            self.btn_stats.setEnabled(has_fasta and has_hdf5)
            if hasattr(self, 'btn_hist'):
                self.btn_hist.setEnabled(has_fasta and has_hdf5)
            
            is_umap = hasattr(self, 'check_umap') and self.check_umap.isChecked()
            if hasattr(self, 'cb_layout_device'):
                self.cb_layout_device.setEnabled(not is_umap)
            if "LAYOUT_DEVICE_SELECTION" in self.labels:
                self.labels["LAYOUT_DEVICE_SELECTION"].setEnabled(not is_umap)
            
            if hasattr(self, 'spin_thresh') and hasattr(self, 'spin_top'):
                has_top_edge = self.spin_top.optionalValue() is not None
                self.spin_thresh.setEnabled(not is_umap and not has_top_edge)
                
                if not self.spin_thresh.isEnabled():
                    self.spin_thresh.setStyleSheet("QDoubleSpinBox:disabled { background-color: #f0f0f0; color: #888; }")
                else:
                    self.spin_thresh.setStyleSheet("")
                    
                self.spin_top.setEnabled(not is_umap)
                
            if hasattr(self, 'tabs') and self.tabs.count() > 2:
                self.tabs.setTabEnabled(2, not is_umap)
            
            if hasattr(self, 'btn_check'):
                self.btn_check.setEnabled(has_fasta and has_hdf5)
            
            # Keep cache status on one compact line and refresh manifest discovery.
            self.lbl_cache_tracker.setWordWrap(False) 
            self.lbl_cache_tracker.setMinimumHeight(0)
            self.lbl_cache_tracker.setMaximumHeight(30)
            self._request_cache_discovery()

        def run_statistics(self):
            import h5py
            import numpy as np
            import math
            import sys
            
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), self.cb_fasta.currentText())
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), self.cb_hdf5.currentText())
            
            self.tip_panel.setText("Computing network statistics... This may take a moment for large HDF5 networks.")
            QApplication.processEvents()
            
            try:
                score_mode = self.cb_score_mode.currentText()
                norm_mode = self.cb_norm_mode.currentText()
                
                with h5py.File(hdf5_path, "r") as hf:
                    metadata = cache_manifest.validate_network_schema(hf)
                    is_blast = metadata.network_type == "blast"
                    from Bio import SeqIO
                    kept_mask = None
                    if fasta_path and os.path.exists(fasta_path):
                        fasta_ids = set()
                        fasta_headers = set()
                        for rec in SeqIO.parse(fasta_path, "fasta"):
                            fasta_ids.add(rec.id)
                            fasta_headers.add(rec.description)
                            
                        raw_headers = hf['headers'][:]
                        headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        
                        net_headers_set = set(headers)
                        net_id_set = {h.split()[0] for h in headers}
                        missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
                        if missing_nodes:
                            raise ValueError(f"FASTA file is NOT a strict subset of the network file. {len(missing_nodes)} sequences are missing from the network.")
                        
                        valid_indices = []
                        for i, h in enumerate(headers):
                            rec_id = h.split()[0]
                            if h in fasta_headers or rec_id in fasta_ids:
                                valid_indices.append(i)
                                
                        if len(valid_indices) < len(headers):
                            kept_mask = np.zeros(len(headers), dtype=bool)
                            kept_mask[valid_indices] = True
                    
                    if is_blast:
                        raw_scores = hf['score'][:]
                        sources = hf['i'][:]
                        targets = hf['j'][:]
                    else:
                        sources = hf['i'][:].astype(np.int64)
                        targets = hf['j'][:].astype(np.int64)
                        if score_mode == "local":
                            raw_scores = hf['l_score'][:].astype(np.float32)
                            align_lens = hf['l_len'][:].astype(np.float32)
                        else:
                            raw_scores = hf['g_score'][:].astype(np.float32)
                            align_lens = hf['g_len'][:].astype(np.float32)
                            
                        if 'seq_lens' in hf:
                            seq_lens = hf['seq_lens'][:]
                        else:
                            seq_lens = np.ones(np.max([np.max(sources), np.max(targets)]) + 1)
                        
                    if kept_mask is not None:
                        valid_edges_mask = kept_mask[sources] & kept_mask[targets]
                        raw_scores = raw_scores[valid_edges_mask]
                        sources = sources[valid_edges_mask]
                        targets = targets[valid_edges_mask]
                        if not is_blast:
                            align_lens = align_lens[valid_edges_mask]
                            
                    if is_blast:
                        scores = raw_scores.astype(np.float32)
                    else:
                        epsilon = 1e-6
                        if norm_mode == "alignment_length":
                            denom = align_lens
                        else:
                            len_src = seq_lens[sources].astype(np.float32)
                            len_dst = seq_lens[targets].astype(np.float32)
                            if norm_mode == "shorter_sequence": denom = np.minimum(len_src, len_dst)
                            elif norm_mode == "longer_sequence": denom = np.maximum(len_src, len_dst)
                            elif norm_mode == "average_sequence": denom = (len_src + len_dst) / 2.0
                            else: denom = align_lens
                            
                        denom = np.maximum(denom, epsilon)
                        scores = (raw_scores / denom).astype(np.float32)
                
                if len(scores) == 0:
                    self.tip_panel.setText("Warning: No valid edges were found in the selected FASTA subset.")
                    return
                    
                max_score = np.max(scores)
                min_score = np.min(scores)
                avg_score = np.mean(scores)
                
                sorted_scores = np.sort(scores)
                stored_edges = len(sorted_scores)
                
                total_nodes = np.sum(kept_mask) if kept_mask is not None else len(headers)
                theoretical_max_edges = (total_nodes * (total_nodes - 1)) / 2.0
                
                if is_blast:
                    start_val = int(math.floor(min_score))
                    limit_step_1 = min(100, int(math.ceil(max_score)))
                    thresh_low = np.arange(start_val, limit_step_1 + 1, 1)
                    if max_score > 100:
                        thresh_high = np.arange(105, int(math.ceil(max_score)) + 5, 5)
                        thresholds = np.concatenate([thresh_low, thresh_high])
                    else:
                        thresholds = thresh_low
                else:
                    start_val = math.floor(min_score * 10) / 10.0
                    thresholds = np.arange(start_val, max_score + 0.1, 0.1)
                    
                indices = np.searchsorted(sorted_scores, thresholds, side='left')
                counts = stored_edges - indices
                
                model_name = metadata.model_name
                
                lines = []
                lines.append(f"====== Network Statistics ======")
                lines.append(f"Network Model: {model_name}")
                lines.append(f"Fasta Node Subset: {os.path.basename(fasta_path)}")
                lines.append(f"Total Nodes Processed: {total_nodes}")
                display_norm = norm_mode.replace('_', ' ').title()
                lines.append(f"Metric: {'Log10(E-Value)' if is_blast else f'{score_mode.title()} Alignment Score with {display_norm} Normalization'}")
                lines.append(f"Stored Edges: {stored_edges} (Max possible: {int(theoretical_max_edges)})")
                lines.append(f"Max: {max_score:.4f} | Min: {min_score:.4f} | Avg: {avg_score:.4f}")
                lines.append("-" * 40)
                lines.append(f"{'Threshold':<10} | {'Count':<10} | {'Percentage':<10}")
                lines.append("-" * 40)
                
                for thresh, count in zip(thresholds, counts):
                    pct = (count / theoretical_max_edges) * 100.0 if theoretical_max_edges > 0 else 0
                    if is_blast:
                         lines.append(f"{int(thresh):<10} | {count:<10} | {pct:<9.2f}%")
                    else:
                         lines.append(f"{thresh:<10.1f} | {count:<10} | {pct:<9.2f}%")
                         
                self.stat_display.setText("\n".join(lines))
                self.tip_panel.setText("Network statistics computed successfully.")

            except Exception as e:
                self.tip_panel.setText(f"Error during network statistics calculation: {e}")

        def run_histogram(self):
            import h5py
            import numpy as np
            import math
            
            fasta_path = os.path.join(self.inputs["FASTA_DIR"].text(), self.cb_fasta.currentText())
            hdf5_path = os.path.join(self.inputs["HDF5_DIR"].text(), self.cb_hdf5.currentText())
            
            self.tip_panel.setText("Computing score distribution... This may take a moment.")
            QApplication.processEvents()
            
            try:
                score_mode = self.cb_score_mode.currentText()
                norm_mode = self.cb_norm_mode.currentText()
                
                with h5py.File(hdf5_path, "r") as hf:
                    metadata = cache_manifest.validate_network_schema(hf)
                    is_blast = metadata.network_type == "blast"
                    from Bio import SeqIO
                    kept_mask = None
                    if fasta_path and os.path.exists(fasta_path):
                        fasta_ids = set()
                        fasta_headers = set()
                        for rec in SeqIO.parse(fasta_path, "fasta"):
                            fasta_ids.add(rec.id)
                            fasta_headers.add(rec.description)
                            
                        raw_headers = hf['headers'][:]
                        headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
                        
                        net_headers_set = set(headers)
                        net_id_set = {h.split()[0] for h in headers}
                        missing_nodes = [hid for hid in fasta_ids if hid not in net_id_set and hid not in net_headers_set]
                        if missing_nodes:
                            raise ValueError(f"FASTA file is NOT a strict subset of the network file. {len(missing_nodes)} sequences are missing from the network.")
                        
                        valid_indices = []
                        for i, h in enumerate(headers):
                            rec_id = h.split()[0]
                            if h in fasta_headers or rec_id in fasta_ids:
                                valid_indices.append(i)
                                
                        if len(valid_indices) < len(headers):
                            kept_mask = np.zeros(len(headers), dtype=bool)
                            kept_mask[valid_indices] = True
                    
                    if is_blast:
                        raw_scores = hf['score'][:]
                        sources = hf['i'][:]
                        targets = hf['j'][:]
                    else:
                        sources = hf['i'][:].astype(np.int64)
                        targets = hf['j'][:].astype(np.int64)
                        if score_mode == "local":
                            raw_scores = hf['l_score'][:].astype(np.float32)
                            align_lens = hf['l_len'][:].astype(np.float32)
                        else:
                            raw_scores = hf['g_score'][:].astype(np.float32)
                            align_lens = hf['g_len'][:].astype(np.float32)
                            
                        if 'seq_lens' in hf:
                            seq_lens = hf['seq_lens'][:]
                        else:
                            seq_lens = np.ones(np.max([np.max(sources), np.max(targets)]) + 1)
                        
                    if kept_mask is not None:
                        valid_edges_mask = kept_mask[sources] & kept_mask[targets]
                        raw_scores = raw_scores[valid_edges_mask]
                        sources = sources[valid_edges_mask]
                        targets = targets[valid_edges_mask]
                        if not is_blast:
                            align_lens = align_lens[valid_edges_mask]
                            
                    if is_blast:
                        scores = raw_scores.astype(np.float32)
                    else:
                        epsilon = 1e-6
                        if norm_mode == "alignment_length":
                            denom = align_lens
                        else:
                            len_src = seq_lens[sources].astype(np.float32)
                            len_dst = seq_lens[targets].astype(np.float32)
                            if norm_mode == "shorter_sequence": denom = np.minimum(len_src, len_dst)
                            elif norm_mode == "longer_sequence": denom = np.maximum(len_src, len_dst)
                            elif norm_mode == "average_sequence": denom = (len_src + len_dst) / 2.0
                            else: denom = align_lens
                            
                        denom = np.maximum(denom, epsilon)
                        scores = (raw_scores / denom).astype(np.float32)
                
                if len(scores) == 0:
                    self.tip_panel.setText("Warning: No valid edges were found in the selected FASTA subset.")
                    return
                
                # Determine threshold based on top edge % override
                is_umap = hasattr(self, 'check_umap') and self.check_umap.isChecked()
                top_percent = self.spin_top.optionalValue()
                    
                if top_percent is not None and not is_umap:
                    total_active_nodes = np.sum(kept_mask) if kept_mask is not None else len(headers)
                    theoretical_max_edges = (total_active_nodes * (total_active_nodes - 1)) / 2.0
                    k = int(theoretical_max_edges * (top_percent / 100.0))
                    if len(scores) == 0:
                        threshold = 0.0
                    else:
                        k = max(1, min(k, len(scores)))
                        sorted_all = np.sort(scores)[::-1]
                        threshold = sorted_all[k - 1]
                else:
                    threshold = self.spin_thresh.optionalValue()
                    if threshold is None:
                        threshold = 0.0
                
                self.tip_panel.setText("Displaying score histogram...")
                QApplication.processEvents()
                
                from SSN_Utils import build_score_histogram_figure

                print("Displaying Score Histogram... (Close histogram to continue)")
                figure = build_score_histogram_figure(
                    scores,
                    threshold,
                    is_evalue=is_blast,
                    norm_mode=norm_mode,
                )
                dialog = ScoreHistogramDialog(figure, self)
                try:
                    dialog.exec()
                finally:
                    dialog.release_figure()
                    dialog.deleteLater()
                
                self.tip_panel.setText("Histogram displayed successfully.")

            except Exception as e:
                self.tip_panel.setText(f"Error during histogram generation: {e}")

        def create_visuals_tab(self):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
            )
            tab_layout.setSpacing(CONFIG_TAB_ROW_SPACING)
            profile_widget = QWidget()
            profile_widget.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
            )
            profile_layout = QFormLayout(profile_widget)
            profile_layout.setContentsMargins(0, 0, 0, 0)
            profile_layout.setHorizontalSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            self._add_profile_selector("visual_effects", profile_layout)
            tab_layout.addWidget(profile_widget)
            self._add_profile_separator("visual_effects", tab_layout)
            content = QWidget()
            main_layout = QVBoxLayout(content)
            main_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.addWidget(content, 1)
            self.profile_content_widgets["visual_effects"] = content
            visual_grid = QGridLayout()
            visual_grid.setHorizontalSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            visual_grid.setVerticalSpacing(12)
            visual_grid.setColumnMinimumWidth(0, CONFIG_FIELD_LABEL_WIDTH)
            visual_grid.setColumnStretch(1, 1)
            visual_grid.setColumnMinimumWidth(2, 16)
            visual_grid.setColumnStretch(4, 1)
            visual_row = 0
            
            # 1. Sliders Setup
            slider_settings = [
                {"key": "NODE_SIZE", "type": "int", "min": 1, "max": 20, "default": 10},
                {"key": "EDGE_WIDTH", "type": "float", "min": 0.1, "max": 3.0, "scale": 10.0, "decimals": 1, "default": 1.0},
                {"key": "NODE_BOUNDARY_WIDTH", "type": "float", "min": 0.0, "max": 2.0, "scale": 10.0, "decimals": 1, "default": 0.5},
                {"key": "EDGE_ALPHA", "type": "float", "min": 0.0, "max": 1.0, "scale": 100.0, "decimals": 2, "default": 0.1},
                {"key": "TEXT_SIZE", "type": "int", "min": 1, "max": 24, "default": 8}
            ]
            
            for s in slider_settings:
                key = s["key"]
                display_name = key.replace('_', ' ').title()
                if key == "EDGE_ALPHA": display_name = "Edge Opacity"
                
                val_raw = globals().get(key, s["default"])
                
                ui_element = QWidget()
                ui_element.setObjectName("wrapper")
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                
                if s["type"] == "int":
                    try: val = int(val_raw)
                    except: val = s["default"]
                    
                    sl.setMinimum(s["min"])
                    sl.setMaximum(s["max"])
                    
                    box = NoScrollSpinBox()
                    box.setRange(-999999, 999999)
                    # Match the double spinbox width so every slider row lines up.
                    box.setFixedWidth(70)
                    
                    sl.setValue(val)
                    box.setValue(val)
                    
                    sl.valueChanged.connect(box.setValue)
                    box.valueChanged.connect(sl.setValue)
                    
                else: 
                    try: val = float(val_raw)
                    except: val = s["default"]
                    
                    sc = s["scale"]
                    sl.setMinimum(int(s["min"] * sc))
                    sl.setMaximum(int(s["max"] * sc))
                    
                    box = NoScrollDoubleSpinBox()
                    box.setRange(-999999.0, 999999.0)
                    box.setDecimals(s["decimals"])
                    box.setFixedWidth(70)
                    
                    sl.setValue(int(val * sc))
                    box.setValue(val)
                    
                    sl.valueChanged.connect(lambda v, b=box, scale=sc: b.setValue(v / scale))
                    box.valueChanged.connect(lambda v, s=sl, scale=sc: s.setValue(int(v * scale)))

                sl.setTickPosition(QSlider.TickPosition.TicksBelow)
                sl.setTickInterval(1 if s["type"] == "int" else 10)

                h_lay.addWidget(sl)
                h_lay.addWidget(box)

                lbl = QLabel(f"{display_name}:")
                visual_grid.addWidget(lbl, visual_row, 0)
                visual_grid.addWidget(ui_element, visual_row, 1, 1, 4)
                visual_row += 1
                self.labels[key] = lbl
                self.inputs[key] = box

            # 2. Colors Setup
            color_keys = ["TEXT_COLOR", "INITIAL_NODE_COLOR", "HOVER_COLOR", "CONNECTED_NODE_COLOR", "EDGE_COLOR", "NODE_BOUNDARY_COLOR"]
            self.visual_defaults = VISUAL_PROFILE_DEFAULTS

            color_row_start = visual_row

            for index, key in enumerate(color_keys):
                if key == "HOVER_COLOR": display_name = "Highlight Color"
                else: display_name = key.replace('_', ' ').title()
                
                color_container = QWidget()
                color_container.setObjectName("wrapper")
                h_layout = QHBoxLayout(color_container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                
                val = globals().get(key, self.visual_defaults[key])
                
                swatch = QLabel()
                swatch.setFixedSize(20, 20)
                swatch.setStyleSheet(f"background-color: {val}; border: 1px solid gray; border-radius: 3px;")
                
                le = QLineEdit("" if val in [None, "None"] else str(val))

                def update_color_swatch(value, color_swatch=swatch):
                    color = QColor(str(value).strip())
                    if color.isValid():
                        color_swatch.setStyleSheet(
                            f"background-color: {color.name()}; border: 1px solid gray; border-radius: 3px;"
                        )

                le.textChanged.connect(update_color_swatch)
                
                btn = QPushButton("Pick")
                btn.setFixedWidth(50)
                
                def pick_color(checked, line_edit=le, color_swatch=swatch):
                    initial = line_edit.text()
                    color = QColorDialog.getColor(QColor(initial) if initial else QColor("white"), self, "Select Color")
                    if color.isValid():
                        hex_val = color.name()
                        line_edit.setText(hex_val)
                        color_swatch.setStyleSheet(f"background-color: {hex_val}; border: 1px solid gray; border-radius: 3px;")
                
                btn.clicked.connect(pick_color)
                
                h_layout.addWidget(swatch)
                h_layout.addWidget(le)
                h_layout.addWidget(btn)
                
                lbl = QLabel(f"{display_name}:")
                row = color_row_start + index // 2
                column = 0 if index % 2 == 0 else 3
                visual_grid.addWidget(lbl, row, column)
                visual_grid.addWidget(color_container, row, column + 1)
                self.labels[key] = lbl
                self.inputs[key] = le
                self.color_swatches[key] = swatch

            visual_row = color_row_start + (len(color_keys) + 1) // 2

            # --- Low Resource Mode Toggle ---
            lbl_low_res = QLabel("Low Resource Mode:")
            lbl_low_res.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            cb_low_res = QPushButton()
            cb_low_res.setCheckable(True)
            cb_low_res.setFixedSize(60, 28)
            
            def switch_toggle_style_low_res(checked, btn=cb_low_res):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            
            cb_low_res.toggled.connect(switch_toggle_style_low_res)
            initial_low_res = bool(globals().get("LOW_RESOURCE_MODE", False))
            cb_low_res.setChecked(initial_low_res)
            switch_toggle_style_low_res(initial_low_res)

            low_resource_row = QWidget()
            low_resource_row.setObjectName("lowResourceModeRow")
            low_resource_layout = QHBoxLayout(low_resource_row)
            low_resource_layout.setContentsMargins(0, 8, 0, 0)
            low_resource_layout.setSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            low_resource_layout.addWidget(lbl_low_res)
            low_resource_layout.addWidget(cb_low_res)
            low_resource_layout.addStretch()
            visual_grid.addWidget(low_resource_row, visual_row, 0, 1, 5)
            self.labels["LOW_RESOURCE_MODE"] = lbl_low_res
            self.inputs["LOW_RESOURCE_MODE"] = cb_low_res
            
            main_layout.addLayout(visual_grid)
            main_layout.addStretch()

            self.profile_labels["visual_effects"].setFixedWidth(
                CONFIG_FIELD_LABEL_WIDTH
            )
            
            self.tabs.addTab(tab, "Visual Effects")
            
        def create_physics_tab(self):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
            )
            tab_layout.setSpacing(CONFIG_TAB_ROW_SPACING)
            profile_widget = QWidget()
            profile_widget.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
            )
            profile_layout = QFormLayout(profile_widget)
            profile_layout.setContentsMargins(0, 0, 0, 0)
            profile_layout.setHorizontalSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            self._add_profile_selector("simulation_physics", profile_layout)
            tab_layout.addWidget(profile_widget)
            self._add_profile_separator("simulation_physics", tab_layout)
            content = QWidget()
            main_layout = QVBoxLayout(content)
            main_layout.setContentsMargins(0, 0, 0, 0)
            tab_layout.addWidget(content, 1)
            self.profile_content_widgets["simulation_physics"] = content
            form_layout = QFormLayout()
            field_label_gap = CONFIG_FIELD_HORIZONTAL_SPACING
            paired_group_padding = 24
            form_layout.setHorizontalSpacing(field_label_gap)
            form_layout.setVerticalSpacing(12)
            
            self.physics_defaults = PHYSICS_PROFILE_DEFAULTS
            
            # --- 1. Physics Engine Choice ---
            cb_engine = NoScrollComboBox()
            cb_engine.addItems(["Molecular Dynamics (Style)", "Monte Carlo (Style)"])
            initial_engine = globals().get("PHYSICS_ENGINE", "Molecular Dynamics (Style)")
            cb_engine.setCurrentText(initial_engine)
            lbl_engine = QLabel("Physics Engine:")
            lbl_engine.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            form_layout.addRow(lbl_engine, cb_engine)
            self.inputs["PHYSICS_ENGINE"] = cb_engine
            self.labels["PHYSICS_ENGINE"] = lbl_engine

            cb_layout_device = NoScrollComboBox()
            for display_name, specification in Hardware_Utils.device_selection_options():
                cb_layout_device.addItem(display_name, specification)
            saved_device = Hardware_Utils.normalize_device_selection(
                globals().get("LAYOUT_DEVICE_SELECTION", "auto")
            )
            saved_index = cb_layout_device.findData(saved_device)
            if saved_index < 0:
                cb_layout_device.addItem(
                    f"Unavailable saved device [{saved_device}]", saved_device
                )
                saved_index = cb_layout_device.count() - 1
            cb_layout_device.setCurrentIndex(saved_index)
            lbl_layout_device = QLabel("Layout Device:")
            lbl_layout_device.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            form_layout.addRow(lbl_layout_device, cb_layout_device)
            self.inputs["LAYOUT_DEVICE_SELECTION"] = cb_layout_device
            self.labels["LAYOUT_DEVICE_SELECTION"] = lbl_layout_device
            self.cb_layout_device = cb_layout_device
            
            # --- 2. Existing Physics Sliders ---
            slider_settings = [
                {"key": "SPRING_K", "type": "float", "min": 1.0, "max": 20.0, "scale": 10.0, "decimals": 1, "default": 5.0, "tick": 10},
                {"key": "COULOMB_K", "type": "float", "min": 1.0, "max": 30.0, "scale": 10.0, "decimals": 1, "default": 10.0, "tick": 10},
                {"key": "COULOMB_CUTOFF", "type": "float", "min": 1.0, "max": 100.0, "scale": 10.0, "decimals": 1, "default": 30.0, "tick": 100},
                {"key": "DAMPING", "type": "float", "min": 0.1, "max": 2.0, "scale": 100.0, "decimals": 2, "default": 0.9, "tick": 10}
            ]

            physics_slider_controls = {}
            for s in slider_settings:
                key = s["key"]
                display_name = key.replace('_', ' ').title()
                if key == "COULOMB_K": display_name = "Repulsion Constant"
                elif key == "COULOMB_CUTOFF": display_name = "Max Repulsion Cutoff"
                elif key == "SPRING_K": display_name = "Spring Constant"
                elif key == "DAMPING": display_name = "Damping Coefficient"
                    
                val_raw = globals().get(key, s["default"])
                
                ui_element = QWidget()
                ui_element.setObjectName("wrapper")
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                
                try: val = float(val_raw)
                except: val = s["default"]
                
                sc = s["scale"]
                sl.setMinimum(int(s["min"] * sc))
                sl.setMaximum(int(s["max"] * sc))
                
                box = NoScrollDoubleSpinBox()
                box.setRange(-999999.0, 999999.0)
                box.setDecimals(s["decimals"])
                box.setFixedWidth(70)
                
                sl.setValue(int(val * sc))
                box.setValue(val)
                
                sl.valueChanged.connect(lambda v, b=box, scale=sc: b.setValue(v / scale))
                box.valueChanged.connect(lambda v, s=sl, scale=sc: s.setValue(int(v * scale)))

                sl.setTickPosition(QSlider.TickPosition.TicksBelow)
                sl.setTickInterval(s["tick"])

                h_lay.addWidget(sl)
                h_lay.addWidget(box)

                lbl = QLabel(f"{display_name}:")
                physics_slider_controls[key] = (lbl, ui_element)
                self.labels[key] = lbl
                self.inputs[key] = box

            # All paired slider rows share one grid. A grid per row sizes its columns
            # independently, so rows whose right-hand labels differ in width end up a
            # few pixels out of step with each other.
            slider_pair_grid = QGridLayout()
            slider_pair_grid.setHorizontalSpacing(0)
            slider_pair_grid.setVerticalSpacing(12)
            slider_pair_grid.setColumnMinimumWidth(0, CONFIG_FIELD_LABEL_WIDTH)
            slider_pair_grid.setColumnMinimumWidth(1, field_label_gap)
            slider_pair_grid.setColumnMinimumWidth(3, paired_group_padding)
            slider_pair_grid.setColumnMinimumWidth(5, field_label_gap)
            slider_pair_grid.setColumnStretch(2, 1)
            slider_pair_grid.setColumnStretch(6, 1)

            def add_paired_slider_row(left_key, right_key):
                row = slider_pair_grid.rowCount()
                left_label, left_control = physics_slider_controls[left_key]
                right_label, right_control = physics_slider_controls[right_key]
                left_label.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)

                slider_pair_grid.addWidget(left_label, row, 0)
                slider_pair_grid.addWidget(left_control, row, 2)
                slider_pair_grid.addWidget(right_label, row, 4)
                slider_pair_grid.addWidget(right_control, row, 6)

            add_paired_slider_row("SPRING_K", "COULOMB_K")
            add_paired_slider_row("COULOMB_CUTOFF", "DAMPING")
            form_layout.addRow(slider_pair_grid)

            # --- 3. Integration and convergence settings (two columns) ---
            convergence_grid = QGridLayout()
            convergence_grid.setHorizontalSpacing(0)
            convergence_grid.setVerticalSpacing(12)
            convergence_grid.setColumnMinimumWidth(0, CONFIG_FIELD_LABEL_WIDTH)
            convergence_grid.setColumnMinimumWidth(1, field_label_gap)
            convergence_grid.setColumnMinimumWidth(3, paired_group_padding)
            convergence_grid.setColumnMinimumWidth(5, field_label_gap)
            convergence_grid.setColumnStretch(2, 1)
            convergence_grid.setColumnStretch(6, 1)

            lbl_dt = QLabel("Step Size:")
            le_dt = QLineEdit(str(globals().get("DT", 0.005)))
            self.inputs["DT"] = le_dt
            self.labels["DT"] = lbl_dt

            lbl_steps = QLabel("Max Steps:")
            le_steps = QLineEdit(str(globals().get("MAX_STEPS", 10000)))
            self.inputs["MAX_STEPS"] = le_steps
            self.labels["MAX_STEPS"] = lbl_steps

            convergence_grid.addWidget(lbl_dt, 0, 0)
            convergence_grid.addWidget(le_dt, 0, 2)
            convergence_grid.addWidget(lbl_steps, 0, 4)
            convergence_grid.addWidget(le_steps, 0, 6)

            lbl_rmsd = QLabel("RMSD Threshold:")
            le_rmsd = QLineEdit(str(globals().get("RMSD_THRESHOLD", 0.005)))
            self.inputs["RMSD_THRESHOLD"] = le_rmsd
            self.labels["RMSD_THRESHOLD"] = lbl_rmsd

            lbl_drop = QLabel("Min % Drop Threshold:")
            le_drop = QLineEdit(str(globals().get("PERCENTAGE_DROP_THRESHOLD", 0.1)))
            self.inputs["PERCENTAGE_DROP_THRESHOLD"] = le_drop
            self.labels["PERCENTAGE_DROP_THRESHOLD"] = lbl_drop

            convergence_grid.addWidget(lbl_rmsd, 1, 0)
            convergence_grid.addWidget(le_rmsd, 1, 2)
            convergence_grid.addWidget(lbl_drop, 1, 4)
            convergence_grid.addWidget(le_drop, 1, 6)
            form_layout.addRow(convergence_grid)

            # --- 4. RMSD Window logscale slider + spinbox (10 to 1000) ---
            import math
            sl_window = NoScrollSlider(Qt.Orientation.Horizontal)
            sl_window.setMinimum(0)
            sl_window.setMaximum(100)
            sl_window.setTickPosition(QSlider.TickPosition.TicksBelow)
            sl_window.setTickInterval(10)

            box_window = NoScrollSpinBox()
            box_window.setRange(10, 1000)
            box_window.setFixedWidth(70)
            
            # Mapping functions
            def slider_to_val(x):
                return int(round(10.0 ** (1.0 + 2.0 * x / 100.0)))
                
            def val_to_slider(v):
                if v < 10: v = 10
                if v > 1000: v = 1000
                return int(round(100.0 * (math.log10(v) - 1.0) / 2.0))
                
            def on_slider_changed(x):
                val = slider_to_val(x)
                box_window.blockSignals(True)
                box_window.setValue(val)
                box_window.blockSignals(False)
                
            def on_box_changed(val):
                x = val_to_slider(val)
                sl_window.blockSignals(True)
                sl_window.setValue(x)
                sl_window.blockSignals(False)
                
            sl_window.valueChanged.connect(on_slider_changed)
            box_window.valueChanged.connect(on_box_changed)
            
            initial_window = int(globals().get("RMSD_WINDOW", 50))
            box_window.setValue(initial_window)
            sl_window.setValue(val_to_slider(initial_window))
            
            ui_window = QWidget()
            ui_window.setObjectName("wrapper")
            h_lay_window = QHBoxLayout(ui_window)
            h_lay_window.setContentsMargins(0, 0, 0, 0)
            h_lay_window.addWidget(sl_window)
            h_lay_window.addWidget(box_window)
            
            lbl_window = QLabel("RMSD Window:")
            lbl_window.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            form_layout.addRow(lbl_window, ui_window)
            self.inputs["RMSD_WINDOW"] = box_window
            self.labels["RMSD_WINDOW"] = lbl_window
            
            # --- 5. Packing controls ---
            lbl_prog = QLabel("Progressive Annealing:")
            lbl_prog.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            cb_prog = QPushButton()
            cb_prog.setCheckable(True)
            cb_prog.setFixedSize(60, 28)
            prog_field = QWidget()
            prog_field.setObjectName("wrapper")
            prog_field.setMinimumHeight(cb_prog.height())
            prog_field_layout = QHBoxLayout(prog_field)
            prog_field_layout.setContentsMargins(0, 0, 0, 0)
            prog_field_layout.addWidget(cb_prog)
            prog_field_layout.addStretch()
            prog_field.setFixedWidth(cb_prog.width())
            self.inputs["ENABLE_PROGRESSIVE_SIMULATION"] = cb_prog
            self.labels["ENABLE_PROGRESSIVE_SIMULATION"] = lbl_prog
            
            def switch_toggle_style(checked, btn=cb_prog):
                if checked:
                    btn.setText("ON")
                    btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                else:
                    btn.setText("OFF")
                    btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
            
            cb_prog.toggled.connect(switch_toggle_style)
            initial_state = bool(globals().get("ENABLE_PROGRESSIVE_SIMULATION", False))
            cb_prog.setChecked(initial_state)
            switch_toggle_style(initial_state)
            
            # Packing Geometry Dropdown
            lbl_geom = QLabel("Packing Geometry:")
            cb_geom = NoScrollComboBox()
            cb_geom.addItems(["Square", "Circle"])

            cb_geom.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            cb_geom.setMinimumWidth(120)

            initial_geom = globals().get("PACKING_GEOMETRY", "Square")
            if initial_geom not in ["Square", "Circle"]:
                initial_geom = "Square"
            cb_geom.setCurrentText(initial_geom)
            self.inputs["PACKING_GEOMETRY"] = cb_geom
            self.labels["PACKING_GEOMETRY"] = lbl_geom

            # Packing Grid Size Slider (Logarithmic Scale)
            pgs_widget = QWidget()
            pgs_widget.setObjectName("wrapper")
            pgs_layout = QHBoxLayout(pgs_widget)
            pgs_layout.setContentsMargins(0, 0, 0, 0)
            
            sl_pgs = NoScrollSlider(Qt.Orientation.Horizontal)
            sl_pgs.setMinimum(0)
            sl_pgs.setMaximum(1000)
            sl_pgs.setTickPosition(QSlider.TickPosition.TicksBelow)
            sl_pgs.setTickInterval(100)

            box_pgs = NoScrollDoubleSpinBox()
            box_pgs.setRange(1.0, 200.0)
            box_pgs.setDecimals(1)
            box_pgs.setFixedWidth(70)
            pgs_widget.setMinimumHeight(box_pgs.minimumHeight())
            
            import math
            val_pgs = globals().get("PACKING_GRID_SIZE", 20.0)
            try: val_pgs = float(val_pgs)
            except: val_pgs = 20.0
            
            box_pgs.setValue(val_pgs)
            pct = math.log(max(1.0, min(val_pgs, 200.0))) / math.log(200.0)
            sl_pgs.setValue(int(pct * 1000.0))
            
            def update_from_slider(x):
                box_pgs.blockSignals(True)
                pct_val = x / 1000.0
                val = math.exp(pct_val * math.log(200.0))
                box_pgs.setValue(round(val, 1))
                box_pgs.blockSignals(False)
                
            def update_from_box(v):
                sl_pgs.blockSignals(True)
                v_clamped = max(1.0, min(v, 200.0))
                pct_val = math.log(v_clamped) / math.log(200.0)
                sl_pgs.setValue(int(pct_val * 1000.0))
                sl_pgs.blockSignals(False)
                
            sl_pgs.valueChanged.connect(update_from_slider)
            box_pgs.valueChanged.connect(update_from_box)
            
            pgs_layout.addWidget(sl_pgs)
            pgs_layout.addWidget(box_pgs)
            
            lbl_pgs = QLabel("Packing Grid Size:")
            self.inputs["PACKING_GRID_SIZE"] = box_pgs
            self.labels["PACKING_GRID_SIZE"] = lbl_pgs

            packing_controls_grid = QGridLayout()
            packing_controls_grid.setHorizontalSpacing(0)
            packing_controls_grid.setVerticalSpacing(12)
            packing_controls_grid.setColumnMinimumWidth(1, field_label_gap)
            packing_controls_grid.setColumnMinimumWidth(3, paired_group_padding)
            packing_controls_grid.setColumnMinimumWidth(5, field_label_gap)
            packing_controls_grid.setColumnMinimumWidth(7, paired_group_padding)
            packing_controls_grid.setColumnMinimumWidth(9, field_label_gap)
            packing_controls_grid.setColumnStretch(10, 1)
            packing_controls_grid.addWidget(lbl_prog, 0, 0)
            packing_controls_grid.addWidget(prog_field, 0, 2)
            packing_controls_grid.addWidget(lbl_geom, 0, 4)
            packing_controls_grid.addWidget(cb_geom, 0, 6)
            packing_controls_grid.addWidget(lbl_pgs, 0, 8)
            packing_controls_grid.addWidget(pgs_widget, 0, 10)
            packing_controls_grid.setRowMinimumHeight(0, cb_prog.height())
            form_layout.addRow(packing_controls_grid)

            # --- 6. Monte Carlo settings (two columns) ---
            monte_carlo_grid = QGridLayout()
            monte_carlo_grid.setObjectName("monteCarloGrid")
            monte_carlo_grid.setContentsMargins(0, 8, 0, 0)
            monte_carlo_grid.setHorizontalSpacing(0)
            monte_carlo_grid.setVerticalSpacing(12)
            monte_carlo_grid.setColumnMinimumWidth(0, CONFIG_FIELD_LABEL_WIDTH)
            monte_carlo_grid.setColumnMinimumWidth(1, field_label_gap)
            monte_carlo_grid.setColumnMinimumWidth(3, paired_group_padding)
            monte_carlo_grid.setColumnMinimumWidth(5, field_label_gap)
            monte_carlo_grid.setColumnStretch(2, 1)
            monte_carlo_grid.setColumnStretch(6, 1)

            lbl_min_k = QLabel("Minimum K:")
            le_min_k = QLineEdit(str(globals().get("SGLD_MIN_K", 20)))
            self.inputs["SGLD_MIN_K"] = le_min_k
            self.labels["SGLD_MIN_K"] = lbl_min_k

            lbl_pct_k = QLabel("Fraction K:")
            le_pct_k = QLineEdit(str(globals().get("SGLD_K_PERCENT", 0.01)))
            self.inputs["SGLD_K_PERCENT"] = le_pct_k
            self.labels["SGLD_K_PERCENT"] = lbl_pct_k

            monte_carlo_grid.addWidget(lbl_min_k, 0, 0)
            monte_carlo_grid.addWidget(le_min_k, 0, 2)
            monte_carlo_grid.addWidget(lbl_pct_k, 0, 4)
            monte_carlo_grid.addWidget(le_pct_k, 0, 6)

            lbl_start_temp = QLabel("Starting Temp:")
            le_start_temp = QLineEdit(str(globals().get("SGLD_START_TEMP", 1.5)))
            self.inputs["SGLD_START_TEMP"] = le_start_temp
            self.labels["SGLD_START_TEMP"] = lbl_start_temp

            lbl_noise_scale = QLabel("Thermal Noise Scale:")
            le_noise_scale = QLineEdit(str(globals().get("SGLD_NOISE_SCALE", 1.0)))
            self.inputs["SGLD_NOISE_SCALE"] = le_noise_scale
            self.labels["SGLD_NOISE_SCALE"] = lbl_noise_scale

            paired_left_labels = (
                lbl_dt, lbl_rmsd, lbl_min_k, lbl_start_temp,
            )
            paired_right_labels = (
                physics_slider_controls["COULOMB_K"][0],
                physics_slider_controls["DAMPING"][0],
                lbl_steps, lbl_drop, lbl_pct_k, lbl_noise_scale,
            )
            for paired_label in paired_left_labels:
                paired_label.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
            right_label_width = max(
                label.fontMetrics().horizontalAdvance(label.text())
                for label in paired_right_labels
            )
            for paired_label in paired_right_labels:
                paired_label.setFixedWidth(right_label_width)

            monte_carlo_grid.addWidget(lbl_start_temp, 1, 0)
            monte_carlo_grid.addWidget(le_start_temp, 1, 2)
            monte_carlo_grid.addWidget(lbl_noise_scale, 1, 4)
            monte_carlo_grid.addWidget(le_noise_scale, 1, 6)
            form_layout.addRow(monte_carlo_grid)

            # Apply styling for disabled states to match other tabs
            disabled_lineedit_style = "QLineEdit:disabled { background-color: #f0f0f0; color: #888; }"
            disabled_label_style = "QLabel:disabled { color: #888; }"
            le_min_k.setStyleSheet(disabled_lineedit_style)
            le_pct_k.setStyleSheet(disabled_lineedit_style)
            le_start_temp.setStyleSheet(disabled_lineedit_style)
            le_noise_scale.setStyleSheet(disabled_lineedit_style)
            lbl_min_k.setStyleSheet(disabled_label_style)
            lbl_pct_k.setStyleSheet(disabled_label_style)
            lbl_start_temp.setStyleSheet(disabled_label_style)
            lbl_noise_scale.setStyleSheet(disabled_label_style)
            
            # --- Toggle Dependencies Function ---
            def update_engine_ui():
                is_mc = cb_engine.currentText() == "Monte Carlo (Style)"
                le_min_k.setEnabled(is_mc)
                le_pct_k.setEnabled(is_mc)
                lbl_min_k.setEnabled(is_mc)
                lbl_pct_k.setEnabled(is_mc)
                
                le_start_temp.setEnabled(is_mc)
                le_noise_scale.setEnabled(is_mc)
                lbl_start_temp.setEnabled(is_mc)
                lbl_noise_scale.setEnabled(is_mc)
                
            cb_engine.currentTextChanged.connect(update_engine_ui)
            update_engine_ui()
            
            main_layout.addLayout(form_layout)
            main_layout.addStretch()
 
            self.tabs.addTab(tab, "Simulation && Physics")
            
        def create_directories_tab(self):
            tab = QWidget()
            layout = QFormLayout(tab)
            layout.setContentsMargins(
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
                CONFIG_TAB_CONTENT_MARGIN,
            )
            layout.setHorizontalSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            layout.setVerticalSpacing(CONFIG_TAB_ROW_SPACING)
            self._add_profile_selector("directories", layout)

            self.directory_open_buttons = {}

            def add_open_folder_button(line_edit, key):
                button = QPushButton("📂")
                button.setFixedWidth(30)
                button.setToolTip("Open Folder")
                button.setEnabled(bool(line_edit.text().strip()))

                def open_selected_folder(checked=False):
                    raw_path = line_edit.text().strip()
                    if not raw_path:
                        return
                    folder = Path(raw_path).expanduser()
                    if not folder.is_absolute():
                        folder = PROJECT_ROOT / folder
                    folder = folder.resolve()
                    folder.mkdir(parents=True, exist_ok=True)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

                button.clicked.connect(open_selected_folder)
                line_edit.textChanged.connect(
                    lambda text, target=button: target.setEnabled(bool(text.strip()))
                )
                self.directory_open_buttons[key] = button
                return button

            saved_config_container = QWidget()
            saved_config_container.setObjectName("wrapper")
            saved_config_layout = QHBoxLayout(saved_config_container)
            saved_config_layout.setContentsMargins(0, 0, 0, 0)
            saved_config_input = QLineEdit(str(globals().get("SAVED_CONFIG_DIR", SAVED_CONFIG_DIR)))
            saved_config_open_button = add_open_folder_button(
                saved_config_input, "SAVED_CONFIG_DIR"
            )
            saved_config_button = QPushButton("Browse...")
            saved_config_layout.addWidget(saved_config_input)
            saved_config_layout.addWidget(saved_config_open_button)
            saved_config_layout.addWidget(saved_config_button)
            layout.addRow("Saved Config Directory:", saved_config_container)
            self.inputs["SAVED_CONFIG_DIR"] = saved_config_input
            self.labels["SAVED_CONFIG_DIR"] = layout.labelForField(saved_config_container)
            self.labels["SAVED_CONFIG_DIR"].setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)

            def browse_saved_config_directory(checked=False):
                folder = QFileDialog.getExistingDirectory(
                    self, "Select Saved Config Directory", saved_config_input.text() or ""
                )
                if folder:
                    saved_config_input.setText(os.path.normpath(folder))
                    self._saved_config_directory_committed()

            saved_config_button.clicked.connect(browse_saved_config_directory)
            saved_config_input.editingFinished.connect(self._saved_config_directory_committed)
            self._add_profile_separator("directories", layout)

            directory_profile_controls = []
            self.profile_content_widgets["directories"] = directory_profile_controls
            
            keys = [
                # Input_Files
                "FASTA_DIR", "MSA_DIR", "HDF5_DIR", "METADATA_DIR",
                "HEADER_LIST_DIR",
                # Cache_Files
                "SAVED_LAYOUT_DIR", "SETTING_EXPORT_DIR", "STRUCTURES_DIR",
                # Analysis_Results
                "PRINT_SAVE_DIR", "SEQUENCE_EXPORT_DIR", "CLUSTER_LABEL_DIR",
                "LOGO_DIR",
            ]
            
            for key in keys:
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                val = globals().get(key, "")
                le = QLineEdit("" if val in [None, "None"] else str(val))
                open_button = add_open_folder_button(le, key)
                btn = QPushButton("Browse...")
                
                def browse_dir(checked, line_edit=le):
                    folder = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or "")
                    if folder:
                        import os
                        line_edit.setText(os.path.normpath(folder))
                        
                btn.clicked.connect(browse_dir)
                
                h_lay.addWidget(le)
                h_lay.addWidget(open_button)
                h_lay.addWidget(btn)
                
                display_name = DIRECTORY_DISPLAY_NAMES.get(key)
                if display_name is None:
                    display_name = key.replace('_', ' ').title()
                    display_name = display_name.replace('Msa', 'MSA')
                    display_name = display_name.replace('Hdf5', 'Network')
                    display_name = display_name.replace('Dir', 'Directory')
                
                lbl = QLabel(f"{display_name}:")
                lbl.setFixedWidth(CONFIG_FIELD_LABEL_WIDTH)
                layout.addRow(lbl, container)
                directory_profile_controls.extend((lbl, container))
                self.labels[key] = lbl
                self.inputs[key] = le

            # Bind the text changes to dynamically refresh the dropdowns in Tab 1
            self.inputs["FASTA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_fasta, "FASTA_DIR", ['.fasta']))
            self.inputs["MSA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            self.inputs["HDF5_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
            self.inputs["SAVED_LAYOUT_DIR"].textChanged.connect(self.update_live_validators)
                
            self.tabs.addTab(tab, "Directories")

        def collect_data(self):
            data = {}
            from PySide6.QtWidgets import QComboBox, QPushButton, QLineEdit
            for key, widget in self.inputs.items():
                
                # ---> NEW: Completely skip saving the target cache selection to JSON
                if key in {"TARGET_CACHE_FILE", "NEW_CACHE_NAME"}:
                    continue
                    
                if isinstance(widget, QComboBox):
                    if key == "LAYOUT_DEVICE_SELECTION":
                        val = widget.currentData()
                    else:
                        val = widget.currentText()
                elif isinstance(widget, OptionalNoScrollDoubleSpinBox):
                    optional_value = widget.optionalValue()
                    val = "" if optional_value is None else str(optional_value)
                elif hasattr(widget, 'value'): 
                    val = str(widget.value())
                elif isinstance(widget, QPushButton) and widget.isCheckable(): 
                    val = widget.isChecked()                                   
                elif hasattr(widget, 'isChecked'): 
                    val = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    val = widget.text()
                else: 
                    val = str(widget)
                
                if not str(val).strip(): 
                    if key == "TOP_EDGE_PERCENT": val = "None"
                    elif key in ["MSA_FILE", "ALIGNMENT_REFERENCE"]: val = ""
                    else: continue
                
                if key == "NODE_FASTA_FILE": val = os.path.join(self.inputs["FASTA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "MSA_FILE": val = os.path.join(self.inputs["MSA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "INPUT_HDF5": val = os.path.join(self.inputs["HDF5_DIR"].text(), val).replace("\\", "/") if val else ""
                
                data[key] = val
            return data

        def _widget_profile_value(self, key):
            widget = self.inputs[key]
            if isinstance(widget, QComboBox):
                value = (
                    widget.currentData()
                    if key == "LAYOUT_DEVICE_SELECTION"
                    else widget.currentText()
                )
            elif isinstance(widget, OptionalNoScrollDoubleSpinBox):
                optional_value = widget.optionalValue()
                value = "None" if optional_value is None else str(optional_value)
            elif isinstance(widget, QPushButton) and widget.isCheckable():
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = str(widget.value())
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            elif hasattr(widget, "isChecked"):
                value = widget.isChecked()
            else:
                value = str(widget)

            if key == "NODE_FASTA_FILE":
                value = (
                    os.path.join(self.inputs["FASTA_DIR"].text(), value).replace("\\", "/")
                    if value else ""
                )
            elif key == "MSA_FILE":
                value = (
                    os.path.join(self.inputs["MSA_DIR"].text(), value).replace("\\", "/")
                    if value else ""
                )
            elif key == "INPUT_HDF5":
                value = (
                    os.path.join(self.inputs["HDF5_DIR"].text(), value).replace("\\", "/")
                    if value else ""
                )
            return value

        def _collect_tab_profile_data(self, tab_id):
            data = {
                key: self._widget_profile_value(key)
                for key in TAB_PROFILE_SPECS[tab_id]["defaults"]
            }
            self._normalize_profile_data(tab_id, data)
            return data

        def _prepare_profile_writes(self):
            tab_data = {
                tab_id: self._collect_tab_profile_data(tab_id)
                for tab_id in TAB_PROFILE_SPECS
            }
            custom_settings = dict(self._custom_settings)
            custom_settings["SAVED_CONFIG_DIR"] = self.inputs["SAVED_CONFIG_DIR"].text()
            writes = []
            created_profiles = []
            default_tabs = []

            for tab_id in TAB_PROFILE_SPECS:
                selection = self.profile_selectors[tab_id].currentText()
                if selection == "(custom)":
                    custom_settings.update(tab_data[tab_id])
                elif selection == "(default)":
                    default_tabs.append(tab_id)
                    continue
                elif selection == "(new)":
                    existing = _discover_profile_names(self._saved_config_root(), tab_id)
                    name = _validate_profile_name(
                        self.profile_name_inputs[tab_id].text(), existing
                    )
                    writes.append((self._profile_path(tab_id, name), tab_data[tab_id]))
                    created_profiles.append((tab_id, name))
                else:
                    writes.append(
                        (self._profile_path(tab_id, selection), tab_data[tab_id])
                    )

            writes.append((Path(DEFAULT_SETTINGS_FILE), custom_settings))
            return writes, custom_settings, created_profiles, default_tabs

        def _save_success_message(self, created_profiles, default_tabs):
            messages = ["Settings saved successfully."]
            if created_profiles:
                created = ", ".join(
                    f"{PROFILE_TAB_DISPLAY_NAMES[tab_id]}: '{name}'"
                    for tab_id, name in created_profiles
                )
                messages.append(f"Created profile(s): {created}.")
            if default_tabs:
                unchanged = ", ".join(
                    PROFILE_TAB_DISPLAY_NAMES[tab_id] for tab_id in default_tabs
                )
                messages.append(
                    "Built-in default settings were left unchanged for: "
                    f"{unchanged}."
                )
            return " ".join(messages)

        def save_settings(self):
            try:
                (
                    writes,
                    custom_settings,
                    created_profiles,
                    default_tabs,
                ) = self._prepare_profile_writes()
                for path, profile_data in writes:
                    _atomic_write_json(path, profile_data)
                self._custom_settings = custom_settings
                for tab_id, name in created_profiles:
                    self._refresh_profile_combo(tab_id)
                    self._set_profile_selection(tab_id, name)
                    self._profile_previous_selection[tab_id] = name
                    self._set_new_profile_field_visible(tab_id, False)
                self.tip_panel.setText(
                    self._save_success_message(created_profiles, default_tabs)
                )
                return True
            except Exception as e:
                self.tip_panel.setText(f"Failed to save settings: {e}")
                return False

        def _selected_new_cache_filename(self):
            if self.cb_cache_file.currentText() != "(New Layout Cache)":
                raise ValueError(
                    "Layout settings can only be exported for (New Layout Cache)."
                )
            cache_name = self.line_new_cache.text().strip()
            if not cache_name:
                cache_name = self.line_new_cache.placeholderText()
            if not cache_name.lower().endswith(".h5"):
                cache_name += ".h5"
            cache_manifest.validate_cache_filename(cache_name)
            return cache_name

        def _collect_layout_generation_settings(self):
            if not self._cache_launch_allowed or not self.current_cache_folder:
                raise ValueError("A unique compatible cache folder has not been resolved.")

            cache_name = self._selected_new_cache_filename()
            collected = self.collect_data()
            collected["ALIGNMENT_SCORE"] = self.cb_score_mode.currentText() or None
            collected["NORM_MODE"] = self.cb_norm_mode.currentText() or None
            collected["SIMILARITY_THRESHOLD"] = self.spin_thresh.optionalValue()
            collected["TOP_EDGE_PERCENT"] = self.spin_top.optionalValue()
            collected["LAYOUT_DEVICE_SELECTION"] = self.cb_layout_device.currentData()
            collected["SAVED_LAYOUT_DIR"] = self.inputs["SAVED_LAYOUT_DIR"].text()
            for hidden_key in (
                "BOX_SCALE",
                "PACKING_PADDING",
                "MAX_FORCE_LIMIT",
                "MAX_TOTAL_REPULSION_FORCE",
            ):
                collected[hidden_key] = globals()[hidden_key]

            return LayoutGenerationSettings.from_namespace(
                SimpleNamespace(**collected),
                cache_filename=cache_name,
                project_root=PROJECT_ROOT,
            )

        def export_layout_settings(self):
            try:
                settings = self._collect_layout_generation_settings()
                export_directory = Path(
                    self.inputs["SETTING_EXPORT_DIR"].text().strip()
                    or DIRECTORY_PROFILE_DEFAULTS["SETTING_EXPORT_DIR"]
                ).expanduser()
                if not export_directory.is_absolute():
                    export_directory = PROJECT_ROOT / export_directory
                export_directory = Path(
                    os.path.abspath(os.path.normpath(export_directory))
                )
                export_directory.mkdir(parents=True, exist_ok=True)
                suggested_name = f"{Path(settings.CACHE_FILENAME).stem}_layout.json"
                selected_path, _selected_filter = QFileDialog.getSaveFileName(
                    self,
                    "Export Layout Settings",
                    str(export_directory / suggested_name),
                    "JSON Files (*.json)",
                )
                if not selected_path:
                    return
                if not selected_path.lower().endswith(".json"):
                    selected_path += ".json"
                target_path = Path(selected_path)
                _atomic_write_json(
                    target_path,
                    settings.to_document(project_root=PROJECT_ROOT),
                )
                command = (
                    f'"{sys.executable}" -u '
                    f'"{PROJECT_ROOT / "src" / "Layout_Cache_Generator.py"}" '
                    f'"{target_path.resolve()}"'
                )
                QMessageBox.information(
                    self,
                    "Layout Settings Exported",
                    f"Settings exported to:\n{target_path.resolve()}\n\n"
                    f"Command-line usage:\n{command}",
                )
            except Exception as error:
                QMessageBox.critical(
                    self, "Export Layout Settings Error", str(error)
                )

        def save_and_run(self):
            if not self.check_umap.isChecked():
                try:
                    Hardware_Utils.resolve_device_selection(
                        self.cb_layout_device.currentData()
                    )
                except ValueError as error:
                    QMessageBox.critical(
                        self, "Layout Device Unavailable", str(error)
                    )
                    return

            if not self._cache_launch_allowed or not self.current_cache_folder:
                QMessageBox.critical(
                    self,
                    "Cache Selection Error",
                    "A unique compatible cache folder has not been resolved.",
                )
                return

            selected_cache = self.cb_cache_file.currentText()
            saved_layout_dir = os.path.abspath(self.inputs["SAVED_LAYOUT_DIR"].text())
            try:
                if selected_cache == "(New Layout Cache)":
                    cache_name = self._selected_new_cache_filename()
                    relative_path = cache_manifest.relative_cache_path(
                        saved_layout_dir, self.current_cache_folder, cache_name
                    )
                    cache_mode = "new"
                else:
                    relative_path = self.cb_cache_file.currentData()
                    cache_manifest.resolve_relative_cache_path(
                        saved_layout_dir, relative_path
                    )
                    cache_mode = "existing"
            except Exception as error:
                QMessageBox.critical(self, "Cache Selection Error", str(error))
                return

            settings_data = self.collect_data()
            if not self.save_settings():
                return

            try:
                settings_snapshot = _create_viewer_settings_snapshot(settings_data)
            except OSError as error:
                QMessageBox.critical(
                    self,
                    "Viewer Launch Error",
                    f"Failed to create the per-launch settings snapshot:\n{error}",
                )
                return

            print("Launching SSN_Viewer.py...")
            env = os.environ.copy()
            env.pop("SSN_TARGET_CACHE", None)
            env["SSN_TARGET_CACHE_PATH"] = relative_path.replace("\\", "/")
            env["SSN_TARGET_CACHE_MODE"] = cache_mode
            env["SSN_VIEWER_SETTINGS_PATH"] = settings_snapshot

            # Use the project root (parent of src/) as cwd so all relative data paths resolve correctly
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            try:
                _handoff_to_viewer(script_dir, env)
            except (OSError, RuntimeError) as error:
                try:
                    os.unlink(settings_snapshot)
                except OSError:
                    pass
                QMessageBox.critical(
                    self,
                    "Viewer Launch Error",
                    f"Failed to launch SSN_Viewer.py:\n{error}",
                )
                return

        def save_only(self):
            self.save_settings()

    existing_qt_application = QApplication.instance()
    app = existing_qt_application or QApplication(sys.argv)
    single_instance = None
    if existing_qt_application is None:
        single_instance = SingleInstanceController("SSN_Config", app)
        try:
            is_primary_instance = single_instance.acquire_or_notify()
        except RuntimeError as error:
            QMessageBox.critical(None, "SSN Config Startup Error", str(error))
            raise SystemExit(1)
        if not is_primary_instance:
            raise SystemExit(0)
        app.aboutToQuit.connect(single_instance.close)

    configure_linux_qt_desktop_identity(app, VIEWER_DESKTOP_FILE_NAME)
    def _exit_on_uncaught_exception(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        app.exit(1)

    sys.excepthook = _exit_on_uncaught_exception
    try:
        configure_qt_application_fonts(app)
    except Exception as e:
        print(f"Warning: Could not configure bundled application fonts: {e}")
    
    # Set Application-wide Icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "viewer_logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    try:
        from SSN_Utils import force_light_palette
        force_light_palette(app)
    except Exception as e:
        print(f"Warning: Could not force light palette: {e}")
        app.setStyle("Fusion")
    window = ConfigGUI()
    if single_instance is not None:
        single_instance.set_activation_callback(
            lambda active_window=window: show_window_in_front(active_window)
        )
    show_window_in_front(window)
    sys.exit(app.exec())
