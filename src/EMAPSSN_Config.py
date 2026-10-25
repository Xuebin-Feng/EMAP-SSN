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

import unicodedata  # Pre-load to prevent Windows DLL search path conflicts with Qt/OpenGL
import sys
if __name__ == "__main__":
    # Before any output: the desktop launcher gives this process a hidden console, MCP a pipe.
    # A headless run, or --console, only gets UTF-8 streams (utilities.Console_Capture).
    from utilities.Console_Capture import start as start_console_capture
    start_console_capture("config")
if __name__ == "__main__" and "--headless" in sys.argv:
    from utilities.Headless_Settings import main as headless_main
    raise SystemExit(headless_main("config"))
# Import Libraries
import ast
import html
import json
import math
import os
import re
import tempfile
import time
from types import SimpleNamespace
import traceback
from pathlib import Path
from utilities.Localization import QT_TRANSLATE_NOOP, Message, display_text
from utilities.Console_Capture import install_qt_message_handler
from utilities import Desktop_Open, Startup_Status
from utilities.Terminal_Launcher import HoldMode, launch_in_terminal
from desktop.Desktop_App import (
    APP_USER_MODEL_IDS,
    APPLICATION_VERSION,
    CONFIG_DISPLAY_NAME,
    VIEWER_DISPLAY_NAME,
    VIEWER_DESKTOP_FILE_NAME,
    configure_linux_qt_desktop_identity,
    set_app_user_model_id,
)
from desktop.Desktop_Launcher_Monitor import launch_viewer_session
from Layout_Cache_Generator import LayoutGenerationSettings
from desktop.Viewer_State import (
    INPUT_PROFILE_DEFAULTS, VISUAL_PROFILE_DEFAULTS, PHYSICS_PROFILE_DEFAULTS, DIRECTORY_PROFILE_DEFAULTS, LEGACY_DEFAULT_DIRECTORY_PATHS, PROFILE_ENUM_VALUES, PROFILE_RANGES,
    migrate_low_resource_mode,
)
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
LAYOUT_METHOD = None
PACMAP_NEIGHBORS = None
TARGET_CACHE_FILE = os.environ.get("SSN_TARGET_CACHE", None)
TARGET_CACHE_PATH = os.environ.get("SSN_TARGET_CACHE_PATH", None)
TARGET_CACHE_MODE = os.environ.get("SSN_TARGET_CACHE_MODE", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Directory & File Paths ---
INPUT_FILE_ALIAS = "$input_file$"
CACHE_FILE_ALIAS = "$cache_file$"
ANALYSIS_RESULT_ALIAS = "$analysis_result$"
EMBEDDING_ALIAS = "$embedding$"

INPUT_FILE_DIR = "Input_Files"
CACHE_FILE_DIR = "Cache_Files"
ANALYSIS_RESULT_DIR = "Analysis_Results"
EMBED_DIR = DIRECTORY_PROFILE_DEFAULTS["EMBED_DIR"]

FASTA_DIR = os.path.join(INPUT_FILE_ALIAS, "Sequence_Sets")
MSA_DIR = os.path.join(INPUT_FILE_ALIAS, "Multiple_Alignments")
HDF5_DIR = os.path.join(INPUT_FILE_ALIAS, "Networks_EValues")
SAVED_LAYOUT_DIR = os.path.join(CACHE_FILE_ALIAS, "Saved_Layouts")
SETTING_EXPORT_DIR = os.path.join(CACHE_FILE_ALIAS, "Exported_Settings")
METADATA_DIR = os.path.join(INPUT_FILE_ALIAS, "Meta_Data")
HEADER_LIST_DIR = os.path.join(INPUT_FILE_ALIAS, "Header_Lists")
DEFAULT_SAVED_CONFIG_DIR = os.path.join(CACHE_FILE_ALIAS, "Saved_Config")
SAVED_CONFIG_DIR = DEFAULT_SAVED_CONFIG_DIR

DIRECTORY_PATH_ALIASES = {
    INPUT_FILE_ALIAS: "INPUT_FILE_DIR",
    CACHE_FILE_ALIAS: "CACHE_FILE_DIR",
    ANALYSIS_RESULT_ALIAS: "ANALYSIS_RESULT_DIR",
    EMBEDDING_ALIAS: "EMBED_DIR",
}
DEPRECATED_DIRECTORY_KEYS = frozenset({
    "LOGO_DIR",
    "CLUSTER_LABEL_DIR",
    "SEQUENCE_EXPORT_DIR",
    "PRINT_SAVE_DIR",
    "STRUCTURES_DIR",
})
OBSOLETE_LAYOUT_ENGINE_KEYS = frozenset({
    "PHYSICS_ENGINE",
    "MC_SWEEPS",
    "MC_QUENCH_SWEEPS",
    "MC_TELEPORT_PROBABILITY",
    "MC_RANDOM_SEED",
    "SGLD_MIN_K",
    "SGLD_K_PERCENT",
    "SGLD_START_TEMP",
    "SGLD_NOISE_SCALE",
})

# --- Explicit Input File Paths ---
# You can manually replace these string paths to decouple file logic:
NODE_FASTA_FILE = ""

# Default values if scanning directories fails:
MSA_FILE = ""
EMBED_FILE = ""
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
NODE_BOUNDARY_COLOR = '#000000'
# Low Resource Mode, by category (desktop.Viewer_State.LOW_RESOURCE_DEFAULTS).
FEWER_EDGES_WHILE_MOVING = False
LOW_RESOURCE_ANIMATIONS = False
LOW_RESOURCE_DRAWING = False
LOW_RESOURCE_WEBSERVER = False
SHOW_FPS = False

# --- Grid Packing Settings ---
PACKING_GRID_SIZE = 10.0  # The base size of one grid square
PACKING_PADDING = 10.0     # Extra padding applied to the bounding box of each cluster

# --- Simulation & Physics Settings ---
LAYOUT_DEVICE_SELECTION = "auto"
SPRING_K = 5.0             
COULOMB_K = 10.0            
COULOMB_CUTOFF = 30.0      
DAMPING = 0.9              
MAX_FORCE_LIMIT = 20.0      
MAX_TOTAL_REPULSION_FORCE = 0.0

DT = 0.005
AUTO_DT = False
BOX_SCALE = 2.0
MAX_STEPS = 10000           
RMSD_THRESHOLD = 0.005 
PERCENTAGE_DROP_THRESHOLD = 0.1    
RMSD_WINDOW = 50
ENABLE_PROGRESSIVE_SIMULATION = False
PACKING_GEOMETRY = "Square"






# Each directory's label on the Directories tab, in the order the tab lists them.
DIRECTORY_LABELS = {
    "INPUT_FILE_DIR": QT_TRANSLATE_NOOP("Config", "Input File Directory:"),
    "CACHE_FILE_DIR": QT_TRANSLATE_NOOP("Config", "Cache File Directory:"),
    "ANALYSIS_RESULT_DIR": QT_TRANSLATE_NOOP("Config", "Analysis Results Directory:"),
    "EMBED_DIR": QT_TRANSLATE_NOOP("Config", "Embedding Directory:"),
    "FASTA_DIR": QT_TRANSLATE_NOOP("Config", "Input FASTA Directory:"),
    "MSA_DIR": QT_TRANSLATE_NOOP("Config", "MSA Directory:"),
    "HDF5_DIR": QT_TRANSLATE_NOOP("Config", "Network Directory:"),
    "METADATA_DIR": QT_TRANSLATE_NOOP("Config", "Metadata Directory:"),
    "HEADER_LIST_DIR": QT_TRANSLATE_NOOP("Config", "Header List Directory:"),
    "SAVED_LAYOUT_DIR": QT_TRANSLATE_NOOP("Config", "Layout Directory:"),
    "SETTING_EXPORT_DIR": QT_TRANSLATE_NOOP("Config", "Setting Export Directory:"),
}

# The Visual Effects tab's Low Resource Mode section: a switch per category,
# then the frame rate, in this order.
LOW_RESOURCE_SWITCH_LABELS = {
    "FEWER_EDGES_WHILE_MOVING": QT_TRANSLATE_NOOP("Config", "Fewer Edges Drawn While Moving:"),
    "LOW_RESOURCE_ANIMATIONS": QT_TRANSLATE_NOOP("Config", "No Zoom or Sidebar Animation:"),
    "LOW_RESOURCE_DRAWING": QT_TRANSLATE_NOOP("Config", "Update Colors After Mouse Release:"),
    "LOW_RESOURCE_WEBSERVER": QT_TRANSLATE_NOOP("Config", "Update Web Pages After Mouse Release:"),
    "SHOW_FPS": QT_TRANSLATE_NOOP("Config", "Show Frame Rate (FPS):"),
}
# Their rows in the section's own two columns, two to a row in the order
# above (the user's choice). Each column's labels are as wide as its longest,
# and take no part in the window's shared label column.
LOW_RESOURCE_SWITCH_ROWS = (
    ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS"),
    ("LOW_RESOURCE_DRAWING", "LOW_RESOURCE_WEBSERVER"),
    ("SHOW_FPS",),
)

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
    "inputs_outputs": QT_TRANSLATE_NOOP("Config", "Inputs & Outputs"),
    "visual_effects": QT_TRANSLATE_NOOP("Config", "Visual Effects"),
    "simulation_physics": QT_TRANSLATE_NOOP("Config", "Simulation & Physics"),
    "directories": QT_TRANSLATE_NOOP("Config", "Directories"),
}

# The bracketed names are the Saved Config selector's own entries, whose stored
# values a profile of the same name would duplicate.
RESERVED_PROFILE_NAMES = {"custom", "default", "new", "(custom)", "(default)", "(new)"}
# What the selector shows for its own entries; a named profile shows its name.
PROFILE_ITEM_LABELS = {
    "(custom)": QT_TRANSLATE_NOOP("Config", "(custom)"),
    "(default)": QT_TRANSLATE_NOOP("Config", "(default)"),
    "(new)": QT_TRANSLATE_NOOP("Config", "(new)"),
}
CONFIG_TAB_CONTENT_MARGIN = 18
CONFIG_TAB_ROW_SPACING = 12
CONFIG_FIELD_HORIZONTAL_SPACING = 12
CONFIG_SEPARATOR_THICKNESS = 2  # the theme's separator_thickness (desktop.Studio_Theme.GEOMETRY)
CONFIG_SEPARATOR_PADDING = 24
# Between a Low Resource Mode label and its switch, in px: twice the field
# spacing (the Studio theme has no spacing token of this size). The switches
# stand apart from their labels, not in line with the colour fields above.
LOW_RESOURCE_SWITCH_GAP = 2 * CONFIG_FIELD_HORIZONTAL_SPACING
# How long Save & Run stays greyed out once clicked, in ms, so a double click
# launches once; as a tool's Save & Run (EMAPSSN_Tools.RUN_BUTTON_PAUSE_MS).
CONFIG_RUN_BUTTON_PAUSE_MS = 1000
# How often, in ms, a shown window looks again for layout caches, so a cache
# published after the inputs were checked shows by itself (_recheck_saved_caches).
CONFIG_CACHE_RECHECK_MS = 1000
# A colour swatch: its colour, framed by a hairline that shows on white and on dark colours.
SWATCH_STYLESHEET = (
    "background-color: {color}; border: 1px solid rgba(9, 9, 11, 0.16); border-radius: 5px;"
)




def resolve_directory_path(value, base_directories=None):
    """Expand one exact leading directory alias without changing other paths."""
    if value is None:
        return value
    raw_path = os.fspath(value)
    bases = base_directories or {}
    for alias, setting_key in DIRECTORY_PATH_ALIASES.items():
        if raw_path == alias:
            suffix = ""
        elif raw_path.startswith(alias) and raw_path[len(alias):len(alias) + 1] in {"/", "\\"}:
            suffix = raw_path[len(alias):].lstrip("/\\")
        else:
            continue

        base_path = os.fspath(
            bases.get(setting_key, globals().get(setting_key, ""))
        )
        if not suffix:
            return os.path.normpath(base_path)
        suffix_parts = [part for part in re.split(r"[/\\]+", suffix) if part]
        return os.path.normpath(os.path.join(base_path, *suffix_parts))
    return raw_path


def _migrate_default_directory_path(key, value):
    """Link an old canonical child default to its configurable base alias."""
    legacy_default = LEGACY_DEFAULT_DIRECTORY_PATHS.get(key)
    if legacy_default is None or not isinstance(value, str):
        return value
    portable_value = value.replace("\\", "/").rstrip("/")
    portable_legacy = legacy_default.replace("\\", "/").rstrip("/")
    if portable_value == portable_legacy:
        return DIRECTORY_PROFILE_DEFAULTS[key]
    return value


def _resolve_runtime_path_settings():
    """Resolve aliases after settings overrides, before Viewer code consumes them."""
    for key in (
        "FASTA_DIR",
        "MSA_DIR",
        "EMBED_DIR",
        "HDF5_DIR",
        "METADATA_DIR",
        "HEADER_LIST_DIR",
        "SAVED_LAYOUT_DIR",
        "SETTING_EXPORT_DIR",
        "NODE_FASTA_FILE",
        "MSA_FILE",
        "EMBED_FILE",
        "INPUT_HDF5",
    ):
        value = globals().get(key)
        if value:
            globals()[key] = resolve_directory_path(value)


def _resolved_saved_config_root(value, base_directories=None):
    resolved_value = resolve_directory_path(
        value or DEFAULT_SAVED_CONFIG_DIR,
        base_directories=base_directories,
    )
    path = Path(str(resolved_value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _migrate_saved_config_dir(value):
    normalized = os.path.normpath(str(value or "")).replace("\\", "/").rstrip("/")
    if normalized.casefold() in {"saved_config", "cache_files/saved_config"}:
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
        raise ValueError(Message("Enter a profile name."))
    if normalized.casefold() in RESERVED_PROFILE_NAMES:
        raise ValueError(Message("'{name}' is a reserved profile name.", name=normalized))
    if normalized in {".", ".."} or normalized.endswith((" ", ".")):
        raise ValueError(Message("Profile names cannot end with a space or period."))
    if re.search(r'[<>:"/\\|?*\x00-\x1f]', normalized):
        raise ValueError(Message(
            "Profile names cannot contain path separators or Windows-reserved characters."
        ))
    windows_stem = normalized.split(".", 1)[0].casefold()
    windows_reserved = {"con", "prn", "aux", "nul"}
    windows_reserved.update(f"com{i}" for i in range(1, 10))
    windows_reserved.update(f"lpt{i}" for i in range(1, 10))
    if windows_stem in windows_reserved:
        raise ValueError(Message("'{name}' is reserved by Windows.", name=normalized))
    if normalized.casefold() in {str(item).casefold() for item in existing_names}:
        raise ValueError(Message("A profile named '{name}' already exists.", name=normalized))
    return normalized


# A save writes a temporary sibling, .<name>.<random>.partial, and renames it
# into place. Only a hard kill in between leaves the sibling behind; a save
# takes milliseconds, so one older than this belongs to no save in progress.
_STALE_PARTIAL_SECONDS = 60


def _remove_stale_partials(path):
    """Delete temporary siblings of path that an interrupted save left behind."""
    prefix, cutoff = f".{path.name}.", time.time() - _STALE_PARTIAL_SECONDS
    try:
        entries = list(os.scandir(path.parent))
    except OSError:
        return
    for entry in entries:
        if not (entry.name.startswith(prefix) and entry.name.endswith(".partial")):
            continue
        try:
            if entry.is_file(follow_symlinks=False) and entry.stat(follow_symlinks=False).st_mtime < cutoff:
                os.unlink(entry.path)
        except OSError:
            pass


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_stale_partials(path)
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

def apply_viewer_settings(settings_dict):
    """Apply a dictionary of viewer settings to EMAPSSN_Config globals."""
    if not isinstance(settings_dict, dict):
        return
    settings_dict = migrate_low_resource_mode(settings_dict)
    LEGACY_KEYS_MAPPING = {
        "NEIGHBOR_COLOR": "INITIAL_NODE_COLOR"
    }
    for k, v in settings_dict.items():
        if k in LEGACY_KEYS_MAPPING:
            k = LEGACY_KEYS_MAPPING[k]
        if k == "SAVED_CONFIG_DIR":
            v = _migrate_saved_config_dir(v)
        if k in DIRECTORY_PROFILE_DEFAULTS:
            v = _migrate_default_directory_path(k, v)
        if k in globals() and v is not None and (str(v).strip() != "" or k in ["MSA_FILE", "EMBED_FILE", "ALIGNMENT_REFERENCE"]):
            orig = globals()[k]
            if isinstance(orig, int) and not isinstance(orig, bool):
                try: v = int(v)
                except: pass
            elif isinstance(orig, float):
                try: v = float(v)
                except: pass
            elif isinstance(orig, bool):
                v = str(v).lower() in ['true', '1', 't', 'y', 'yes']
            elif isinstance(orig, list):
                try: v = ast.literal_eval(v) if isinstance(v, str) else v
                except: pass
            elif orig is None:
                if v == "None": v = None
                elif str(v).replace('.', '', 1).isdigit():
                    v = float(v) if '.' in str(v) else int(v)
            globals()[k] = v
    _resolve_runtime_path_settings()


DEFAULT_SETTINGS_FILE = str(PROJECT_ROOT / "viewer_settings.json")
SETTINGS_FILE = os.environ.get("SSN_VIEWER_SETTINGS_PATH") or DEFAULT_SETTINGS_FILE
viewer_settings = {}
if not os.environ.get("SSN_VIEWER_EXPLICIT_SETTINGS") and os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            viewer_settings = json.load(f)
            apply_viewer_settings(viewer_settings)
    except Exception as e:
        print(f"Failed to load viewer settings: {e}")

_resolve_runtime_path_settings()
if NODE_FASTA_FILE:
    SEQUENCES_FILE = NODE_FASTA_FILE
    SEQUENCE_SET = os.path.splitext(os.path.basename(NODE_FASTA_FILE))[0]
    if MSA_FILE and "none_[e1_ra]_alignment.fasta" in MSA_FILE.lower():
        MSA_FILE = os.path.join(MSA_DIR, f"{SEQUENCE_SET}_[E1_RA]_alignment.fasta")
    if not INPUT_HDF5 or "none_[e1_ra]_network.h5" in INPUT_HDF5.lower():
        INPUT_HDF5 = os.path.join(HDF5_DIR, f"{SEQUENCE_SET}_[E1_RA]_network.h5")


def _handoff_to_viewer(
    project_root,
    env,
    *,
    settings_path=None,
    platform_name=None,
    executable=None,
):
    """Launch the viewer: on Windows without a console, elsewhere in the platform's terminal.

    A Windows Viewer starts through desktop/Desktop_Launcher.py, which shows a splash and runs
    it with a hidden console; the Viewer keeps what it prints for its Console window
    (utilities/Console_Capture.py), so no terminal is needed. The other platforms keep the
    terminal contract (utilities/Terminal_Launcher.py).
    """
    executable = executable or sys.executable
    project_root = os.path.abspath(project_root)
    if (platform_name or sys.platform) == "win32":
        return launch_viewer_session(project_root, env, settings_path=settings_path, executable=executable)
    viewer_script = os.path.join(project_root, "src", "EMAPSSN_Viewer.py")
    argv = [executable, "-u", viewer_script]
    if settings_path:
        argv.extend(["--settings", os.fspath(settings_path), "--delete-settings"])
    return launch_in_terminal(
        argv,
        cwd=project_root,
        env=env,
        hold=HoldMode.ON_ERROR,
        title=VIEWER_DISPLAY_NAME,
        platform_name=platform_name,
    )


def _handoff_to_layout_generator(
    project_root,
    layout_settings_path,
    env,
    *,
    launch_viewer=True,
    platform_name=None,
    executable=None,
):
    """Launch the layout generator in a terminal, optionally handing off to viewer."""
    executable = executable or sys.executable
    project_root = os.path.abspath(project_root)
    generator_script = os.path.join(project_root, "src", "Layout_Cache_Generator.py")
    argv = [executable, "-u", generator_script, layout_settings_path]
    if launch_viewer:
        argv.append("--launch-viewer")
    argv.append("--delete-settings")
    return launch_in_terminal(
        argv,
        cwd=project_root,
        env=env,
        hold=HoldMode.ON_ERROR,
        title="EMAP-SSN Layout Generator",
        platform_name=platform_name,
    )


def _create_layout_settings_snapshot(settings_doc):
    """Write one private layout settings file for a single Layout Generator process."""
    descriptor, path = tempfile.mkstemp(prefix="ssn_layout_", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings_doc, handle, indent=4)
            handle.write("\n")
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _create_viewer_settings_snapshot(settings):
    """Write one private settings file for a single Viewer process."""
    from desktop.Viewer_State import DEFAULTS, encode_document
    settings = encode_document("viewer", {**DEFAULTS, **settings})
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
    from utilities.Sequence_Utils import load_sanitized_fasta

    headers, _, _ = load_sanitized_fasta(fasta_path)
    return headers


def _load_consistency_msa_headers(msa_path):
    """Return canonical MSA headers using the same rules as the viewer."""
    from utilities.Sequence_Utils import (
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


def _load_consistency_fasta_sequences(fasta_path):
    """Return the canonical FASTA sequence of each canonical header."""
    from utilities.Sequence_Utils import load_sanitized_fasta

    headers, sequences, _ = load_sanitized_fasta(fasta_path, report=False)
    return dict(zip(headers, sequences))


def _load_consistency_embedding_manifest(embed_path):
    """Return (manifest, width) of an embedding file, as the Viewer reads it."""
    from Embedding_Manager import read_embedding_summary

    return read_embedding_summary(embed_path)


def _consistency_network_metadata(network_path):
    """The network's model name and type (Cache_Manifest), or None when unreadable."""
    try:
        return cache_manifest.read_network_metadata(network_path)
    except (OSError, ValueError):
        return None


# =============================================================================
# GUI APPLICATION
# =============================================================================
if __name__ == "__main__":
    # Hardware_Utils imports PyTorch.  On Windows this must happen before
    # PySide6/OpenGL initializes, otherwise torch's c10.dll can fail to load.
    Startup_Status.report(Startup_Status.LOADING_PYTORCH, 0.45)
    try:
        from utilities import Hardware_Acceleration as Hardware_Utils
    except ImportError:
        import Hardware_Acceleration as Hardware_Utils

    os.environ["QT_API"] = "pyside6"
    os.environ["QT_MAC_WANTS_LIGHT_THEME"] = "1"
    from PySide6.QtWidgets import (
        QStackedWidget,
        QApplication, QMainWindow, QWidget, QVBoxLayout,
        QHBoxLayout, QGridLayout, QTabWidget, QFormLayout, QLineEdit,
        QComboBox, QPushButton, QMessageBox,
        QLabel, QSplitter, QSlider, QSpinBox, QDoubleSpinBox,
        QStyle, QStyleOptionSlider, QStyleOptionSpinBox, QFileDialog,
        QSizePolicy, QFrame, QScrollArea,
    )
    from desktop.Desktop_App import (
        ResponsiveFieldLayout,
        ResponsiveFlowLayout,
        ResponsiveSelectorLayout,
        SingleInstanceController,
        StatisticsReport,
        SteadyPopupComboBox,
        ToggleSwitch,
        add_combo_options,
        combo_value,
        configure_qt_application_fonts,
        configured_theme_color,
        fit_buttons_to_text,
        apply_studio_theme,
        choose_language,
        choose_theme_color,
        install_translations,
        installed_language,
        instance_handover_message,
        language_selector_row,
        LanguageFollower,
        LanguageSelector,
        NAME_ITEM_ROLE,
        mark_name_item,
        open_color_picker,
        qt_monospace_font,
        redraw_in_language,
        redraw_in_theme_color,
        CONFIG_THEME_SETTING,
        ThemeColorButton,
        select_combo_value,
        show_window_in_front,
        startup_language,
        # Config and VR Config file their texts under one context, "Config",
        # so a sentence the two share is translated once.
        translate,
    )
    from desktop.Studio_Theme import (
        TOKENS,
        CardShadows,
        add_hint_icon,
        icon_button,
        prepare_tab_widget,
        set_role,
        status_color,
        theme_color,
    )
    from desktop.Tip_Panel import (
        Code,
        CodeList,
        Field,
        FieldTip,
        LiveNote,
        MISSING,
        PanelHeightAtFirstLayout,
        SpacedTipLabel,
        Tip,
        TipNote,
        TipOption,
        change_signal,
        free_bytes,
        status_mark,
        wrappable,
    )
    from PySide6.QtCore import Qt, QSize, QThread, QTimer, Signal
    from PySide6.QtGui import (
        QColor,
        QIcon,
    )

    def statistics_report_html(summary, headings, rows):
        """The statistics report: its summary lines, which wrap, then the threshold table.

        A table keeps each threshold's figures on one row however narrow the
        report is, which spaced-out plain text couldn't: it wrapped or ran off
        the side. The figures align right under their headings.
        """
        muted = TOKENS["text_muted"]
        lines = "".join(
            f'<p style="margin-top: 0px; margin-bottom: 4px;">{html.escape(line)}</p>'
            for line in summary
        )
        def cell(index):
            # The first column starts flush with the summary; the others keep a gap.
            return f"padding: 2px 0px 2px {0 if index == 0 else 12}px; white-space: nowrap;"

        head = "".join(
            f'<th align="{"left" if index == 0 else "right"}" style="{cell(index)} color: {muted}; '
            f'font-weight: 600; border-bottom: 1px solid {TOKENS["border"]};">'
            f'{html.escape(heading)}</th>'
            for index, heading in enumerate(headings)
        )
        body = "".join(
            "<tr>" + "".join(
                f'<td align="{"left" if index == 0 else "right"}" style="{cell(index)}">{html.escape(value)}</td>'
                for index, value in enumerate(row)
            ) + "</tr>"
            for row in rows
        )
        return (
            f"{lines}<table width=\"100%\" cellspacing=\"0\" style=\"margin-top: 8px;\">"
            f"<tr>{head}</tr>{body}</table>"
        )

    def normalization_names():
        """What the window calls each normalization mode, by its stored value.

        The Normalization Mode dropdown shows these and the statistics report
        quotes them, so a mode reads the same in both. Only the labels differ
        from the stored values (NORM_MODE keeps alignment_length and the rest).
        """
        return {
            "alignment_length": translate("Config", "Alignment Length", "normalization"),
            "shorter_sequence": translate("Config", "Shorter Sequence", "normalization"),
            "longer_sequence": translate("Config", "Longer Sequence", "normalization"),
            "average_sequence": translate("Config", "Average Sequence", "normalization"),
        }

    # The Layout Method dropdown offers SSN's physics simulation and the two manifold methods.
    # It stands for two saved settings: UMAP_MODE ("a manifold layout instead of physics") and
    # LAYOUT_METHOD (which manifold method). SSN is UMAP_MODE off, and then LAYOUT_METHOD keeps
    # whichever method was saved, so choosing SSN and back loses nothing.
    MANIFOLD_METHODS = ("umap", "pacmap")
    LAYOUT_CHOICES = ("ssn",) + MANIFOLD_METHODS
    LAYOUT_SETTINGS = ("UMAP_MODE", "LAYOUT_METHOD")
    # The dropdown's narrowest width, in characters: PaCMAP, its longest choice, in full.
    LAYOUT_METHOD_MINIMUM_CHARACTERS = 7

    def layout_choice(umap_mode, method):
        """The dropdown's choice for the saved UMAP_MODE and LAYOUT_METHOD."""
        if not umap_mode:
            return "ssn"
        return method if method in MANIFOLD_METHODS else "umap"

    def layout_method_names():
        """What the window calls each Layout Method choice, by its stored value."""
        return {
            "ssn": translate("Config", "SSN", "layout method"),
            "umap": translate("Config", "UMAP", "layout method"),
            "pacmap": translate("Config", "PaCMAP", "layout method"),
        }

    def statistics_report_text(report):
        """The statistics report's summary lines and column headings, in the window's language.

        report is what run_statistics found, in no language (ConfigGUI.show_statistics),
        so a window redrawn in another language shows the same report in it.
        """
        if report["blast"]:
            metric = translate("Config", "Metric: Log10(E-Value)")
        else:
            score_mode, norm_mode = report["score_mode"], report["normalization"]
            score_names = {
                "global": translate("Config", "Global", "alignment score"),
                "local": translate("Config", "Local", "alignment score"),
            }
            norm_names = normalization_names()
            metric = translate(
                "Config", "Metric: {score} Alignment Score with {normalization} Normalization"
            ).format(
                score=score_names.get(score_mode, score_mode.title()),
                normalization=norm_names.get(norm_mode, norm_mode.replace('_', ' ').title()),
            )
        summary = [
            translate("Config", "Network Model: {model}").format(model=report["model"]),
            translate("Config", "Fasta Node Subset: {name}").format(name=report["fasta"]),
            translate("Config", "Total Nodes Processed: {count}").format(count=report["nodes"]),
            metric,
            translate("Config", "Stored Edges: {stored} (Max possible: {possible})").format(
                stored=report["stored_edges"], possible=report["possible_edges"]
            ),
            translate("Config", "Max: {maximum:.4f} | Min: {minimum:.4f} | Avg: {mean:.4f}").format(
                maximum=report["maximum"], minimum=report["minimum"], mean=report["mean"]
            ),
        ]
        headings = (
            translate("Config", "Threshold", "statistics column"),
            translate("Config", "Count", "statistics column"),
            translate("Config", "Percentage", "statistics column"),
        )
        return summary, headings

    def _profile_default(key):
        """The value a settings key starts with in a profile (the program's own default), or MISSING."""
        for defaults in (
            INPUT_PROFILE_DEFAULTS, VISUAL_PROFILE_DEFAULTS, PHYSICS_PROFILE_DEFAULTS,
            DIRECTORY_PROFILE_DEFAULTS,
        ):
            if key in defaults:
                return defaults[key]
        return MISSING

    def _current_color_note(field):
        """The live note of a colour field: a swatch of the colour now typed, and its name."""
        value = field.widget.text().strip()
        if not value or not QColor(value).isValid():
            return None
        return LiveNote(
            translate("Config", "Current color: {value}"), {"value": Code(value)}, swatch=value
        )

    def _detected_devices_note(field):
        """The live note of a device dropdown: the devices it found on this computer, by name."""
        combo = field.widget
        names = [
            combo.itemText(index) for index in range(combo.count())
            if combo.itemData(index, NAME_ITEM_ROLE)
        ]
        if not names:
            return None
        return LiveNote(translate("Config", "Detected devices: {devices}"), {"devices": CodeList(names)})

    # The Consistency Check's marks, which tell a check's status by shape as
    # well as by colour (Studio_Theme.status_color kinds): Lucide icons, the
    # tip panel's own (desktop.Tip_Panel).
    consistency_mark = status_mark

    # How many headers a check names as examples of the nodes it misses.
    CONSISTENCY_EXAMPLES = 5

    class ConsistencyCheck:
        """One finding of the Consistency Check: a row of its table, and what follows the table.

        status is "success", "warning" or "danger", the colour and mark of the
        row. result and details are the row's two text cells, in the window's
        language: details is a list of lines (the first states the verdict),
        and a row without any lets result take both cells. notes are the
        sentences that follow the table, each as (kind, text), kind being
        "warning", "danger" or "subtle" (a remark that changes no status).
        """

        def __init__(self, status, name, result, details=(), notes=()):
            self.status = status
            self.name = name
            self.result = result
            self.details = list(details)
            self.notes = list(notes)

    def _consistency_examples(headers):
        return ", ".join(headers[:CONSISTENCY_EXAMPLES])

    def _consistency_matched(matched, total):
        return translate("Config", "{matched} of {total} matched").format(
            matched=matched, total=total
        )

    def network_consistency_check(fasta_headers, network_headers):
        """The Consistency Check's network row: every FASTA header must be a node of the network."""
        known = set(network_headers)
        missing = [header for header in fasta_headers if header not in known]
        name = translate("Config", "FASTA vs HDF5")
        result = _consistency_matched(len(fasta_headers) - len(missing), len(fasta_headers))
        if not missing:
            return ConsistencyCheck(
                "success", name, result, [translate("Config", "strict subset")]
            )
        return ConsistencyCheck(
            "danger", name, result,
            [translate("Config", "NOT a subset: {missing} missing").format(missing=len(missing))],
            [("danger", translate("Config", "Missing examples: {examples}").format(
                examples=_consistency_examples(missing)
            ))],
        )

    def msa_consistency_check(fasta_headers, msa_headers):
        """The Consistency Check's MSA row: how many FASTA headers the alignment holds."""
        missing = [header for header in fasta_headers if header not in msa_headers]
        name = translate("Config", "FASTA vs MSA")
        result = _consistency_matched(len(fasta_headers) - len(missing), len(fasta_headers))
        if not missing:
            return ConsistencyCheck(
                "success", name, result, [translate("Config", "covers all FASTA nodes")]
            )
        return ConsistencyCheck(
            "warning", name, result,
            [translate("Config", "coverage incomplete: {missing} missing").format(
                missing=len(missing)
            )],
            [
                ("warning", translate("Config", "Missing examples: {examples}").format(
                    examples=_consistency_examples(missing)
                )),
                ("warning", translate(
                    "Config",
                    "Missing nodes remain plotted but are excluded from "
                    "alignment-dependent analyses.",
                )),
            ],
        )

    def embedding_consistency_check(embed_path, fasta_path, fasta_headers, network_path):
        """The Consistency Check's embedding row, matched as the Viewer matches it.

        A FASTA node is covered when the embedding file holds its header with
        the same sanitized sequence (Embedding_Manager).
        """
        name = translate("Config", "FASTA vs embedding")
        try:
            manifest, width = _load_consistency_embedding_manifest(embed_path)
        except (OSError, ValueError, KeyError) as error:
            return ConsistencyCheck(
                "danger", name, translate("Config", "the file cannot be used"),
                notes=[("danger", translate(
                    "Config",
                    "The embedding file cannot be used, so the Viewer will not "
                    "open with it: {error}",
                ).format(error=display_text(error)))],
            )
        stored = manifest.sequence_by_header
        sequences = _load_consistency_fasta_sequences(fasta_path)
        missing = [header for header in fasta_headers if header not in stored]
        different = [
            header for header in fasta_headers
            if header in stored and sequences.get(header, stored[header]) != stored[header]
        ]
        matched = len(fasta_headers) - len(missing) - len(different)
        result = _consistency_matched(matched, len(fasta_headers))
        model = translate(
            "Config", "Embedding model: {model} (width {width}, {saving_mode})."
        ).format(model=manifest.model_name, width=width, saving_mode=manifest.saving_mode)
        notes = []
        if missing or different:
            status = "warning"
            verdict = translate(
                "Config",
                "coverage incomplete: {missing} missing, %n different sequence(s)",
                None, len(different),
            ).format(missing=len(missing))
            if missing:
                notes.append(("warning", translate("Config", "Missing examples: {examples}").format(
                    examples=_consistency_examples(missing)
                )))
            if different:
                notes.append(("warning", translate(
                    "Config", "Examples with a different sequence: {examples}"
                ).format(examples=_consistency_examples(different))))
            notes.append(("warning", translate(
                "Config",
                "Missing nodes, and nodes whose sequence differs, remain plotted "
                "but are excluded from embedding-dependent analyses.",
            )))
        else:
            status = "success"
            verdict = translate("Config", "covers all FASTA nodes")
        network = _consistency_network_metadata(network_path)
        if (
            network is not None
            and network.network_type == "alignment"
            and network.model_name != manifest.model_name
        ):
            notes.append(("subtle", translate(
                "Config",
                "NOTE: The network was built with {network}, not {model}. The "
                "Viewer accepts this; embedding commands use the embedding's model.",
            ).format(network=network.model_name, model=manifest.model_name)))
        return ConsistencyCheck(status, name, result, [verdict, model], notes)

    def reference_consistency_check(reference, candidates, msa_headers):
        """The Consistency Check's reference row: the header the Viewer resolves reference to.

        It resolves it exactly as the Viewer will: the same rules over the
        same candidates, the network headers the FASTA keeps, in network
        order. msa_headers is None when no MSA was checked.
        """
        from utilities.Sequence_Utils import reference_header_matches

        name = translate("Config", "Reference ID")
        matches = reference_header_matches(candidates, reference)
        if not matches:
            return ConsistencyCheck(
                "warning", name,
                translate(
                    "Config",
                    "'{reference}' matches no network header, so reference "
                    "numbering will be inactive.",
                ).format(reference=reference),
            )
        resolved = matches[0]
        status, notes = "success", []
        if msa_headers is not None and resolved not in msa_headers:
            status = "warning"
            notes.append(("warning", translate(
                "Config",
                "The MSA lacks this header, so positions will be numbered by "
                "occupancy and the offset ignored.",
            )))
        if len(matches) > 1:
            others = ", ".join(matches[1:4])
            if len(matches) > 4:
                others += ", ..."
            notes.append(("subtle", translate(
                "Config",
                "It matches %n header(s) equally; the first in network order "
                "is used. Others: {others}",
                None, len(matches),
            ).format(others=others)))
        return ConsistencyCheck(
            status, name,
            translate("Config", "'{reference}' resolves to {header}").format(
                reference=reference, header=resolved
            ),
            notes=notes,
        )

    def consistency_report_html(checks):
        """The Consistency Check's report: one table row per check, then each problem's detail.

        A row reads: a mark in the status colour, the check, its result and
        its details. The mark, the check and the result stay on one line; the
        details take the rest of the width and wrap. Below the table, a
        check's notes (the headers it missed, what that costs, errors) follow
        in a paragraph that opens with its mark and name. Rich text, like
        statistics_report_html, is what a QLabel can lay out as a table.
        """
        strong, soft, muted, border = (
            TOKENS[key] for key in ("text", "text_soft", "text_muted", "border")
        )

        def shown(text):
            return html.escape(wrappable(text))

        def cell(content, style, span=1, width=None):
            attributes = (f' colspan="{span}"' if span > 1 else "") + (
                f' width="{width}"' if width else ""
            )
            return f'<td valign="top"{attributes} style="{style}">{content}</td>'

        rows = []
        for index, check in enumerate(checks):
            line = f" border-bottom: 1px solid {border};" if index + 1 < len(checks) else ""
            gap = f"padding: 2px 10px 2px 0px; white-space: nowrap;{line}"
            flush = f"padding: 2px 0px 2px 0px;{line}"
            cells = [
                cell(consistency_mark(check.status), gap),
                cell(html.escape(check.name), f"{gap} font-weight: 600; color: {strong};"),
            ]
            if check.details:
                verdict, *more = check.details
                verdict_color = muted if check.status == "success" else status_color(check.status)
                lines = [f'<span style="color: {verdict_color};">{shown(verdict)}</span>']
                lines += [f'<span style="color: {muted};">{shown(text)}</span>' for text in more]
                cells.append(cell(shown(check.result), f"{gap} color: {soft};"))
                cells.append(cell("<br>".join(lines), flush, width="100%"))
            else:
                cells.append(cell(shown(check.result), f"{flush} color: {soft};", span=2))
            rows.append(f"<tr>{''.join(cells)}</tr>")
        parts = [
            '<table width="100%" cellspacing="0" cellpadding="0" style="margin: 0px;">'
            f"{''.join(rows)}</table>"
        ]
        for check in checks:
            # A check's consecutive notes of one kind share a paragraph.
            runs = []
            for kind, sentence in check.notes:
                if runs and runs[-1][0] == kind:
                    runs[-1][1].append(sentence)
                else:
                    runs.append((kind, [sentence]))
            for kind, sentences in runs:
                parts.append(
                    '<p style="margin: 5px 0px 0px 0px; line-height: 115%; font-size: smaller;">'
                    f'{consistency_mark(kind)} <span style="font-weight: 600; color: {strong};">'
                    f'{html.escape(check.name)}</span> '
                    f'{"<br>".join(shown(text) for text in sentences)}</p>'
                )
        return "".join(parts)

    def consistency_verdict(checks):
        """The Consistency Check's overall status, which the panel's corner icon shows: its worst row's."""
        worst = {"success": 0, "warning": 1, "danger": 2}
        return max((check.status for check in checks), key=lambda status: worst[status])

    # --- Custom Widget Classes ---
    # SpacedTipLabel, the tip panel's label, is in desktop.Tip_Panel.
    class NoScrollComboBox(SteadyPopupComboBox):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class DynamicComboBox(NoScrollComboBox):
        """A combo box that lists the files or profiles on disk again as it opens."""

        def __init__(self, refresh_callback, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.refresh_callback = refresh_callback
            # Twelve characters at its narrowest, however long the names it
            # lists. Set before anything measures it: a combo box keeps the
            # minimum width it first works out, and a tab's scroll area works
            # it out as it takes the tab, before _prepare_responsive_layouts.
            # A long file name would otherwise widen the window for good.
            self.setMinimumContentsLength(12)
            self.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )

        def showPopup(self):
            if self.refresh_callback:
                self.refresh_callback()
            super().showPopup()
            
    def spin_box_minimum_size(spinbox, hint):
        """``hint``, made as wide as ``spinbox`` needs for its minimum_text.

        Qt makes a spin box wide enough for the longest value in its range, plus
        a space and room for the cursor. A box whose range is far wider than the
        values it usually holds names a usual one in minimum_text, which is
        measured the same way. A longer value is cut off only while the box is
        at that minimum width.
        """
        if spinbox.minimum_text is None:
            return hint
        option = QStyleOptionSpinBox()
        spinbox.initStyleOption(option)
        text = spinbox.prefix() + spinbox.minimum_text + spinbox.suffix() + " "
        content = QSize(spinbox.fontMetrics().horizontalAdvance(text) + 2, hint.height())
        width = spinbox.style().sizeFromContents(
            QStyle.ContentsType.CT_SpinBox, option, content, spinbox
        ).width()
        return QSize(width, hint.height())

    class NoScrollSpinBox(QSpinBox):
        # The text its minimum width shows; None shows its whole range.
        minimum_text = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMinimumHeight(28)

        def minimumSizeHint(self):
            return spin_box_minimum_size(self, super().minimumSizeHint())
            
        def wheelEvent(self, event):
            if not self.hasFocus():
                event.ignore()
            else:
                super().wheelEvent(event)
                
    class NoScrollDoubleSpinBox(QDoubleSpinBox):
        # The text its minimum width shows; None shows its whole range.
        minimum_text = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMinimumHeight(28)

        def minimumSizeHint(self):
            return spin_box_minimum_size(self, super().minimumSizeHint())
            
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
        completed = Signal(int, object, object, str)

        def __init__(
            self, request_id, sequence_path, network_path, cache_keys, cached_records=None
        ):
            super().__init__()
            self.request_id = request_id
            self.sequence_path = sequence_path
            self.network_path = network_path
            # The inputs' file keys when hashing was requested. The result is
            # cached under them, whatever is selected or on disk when it ends.
            self.cache_keys = cache_keys
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
                self.completed.emit(self.request_id, self.cache_keys, records, "")
            except Exception as error:
                self.completed.emit(self.request_id, self.cache_keys, {}, str(error))

    class StatisticsWorker(QThread):
        """Computes a network's statistics off the window's thread.

        The pairs are read and scored a slice at a time by worker threads of their own
        (utilities.Network_Score_Statistics), so a network of any size takes a few
        hundred megabytes. completed carries the request id, the report (None when no
        edge is left) and the error, if any; a stopped run reports a StatisticsCancelled.
        """

        progress = Signal(int, float)
        completed = Signal(int, object, object)

        def __init__(self, request_id, network_path, fasta_path, score_mode, norm_mode):
            super().__init__()
            self.request_id = request_id
            self.network_path = network_path
            self.fasta_path = fasta_path
            self.score_mode = score_mode
            self.norm_mode = norm_mode

        def run(self):
            try:
                report = self._report()
            except Exception as error:
                self.completed.emit(self.request_id, None, error)
            else:
                self.completed.emit(self.request_id, report, None)

        def _report(self):
            import h5py
            from utilities import Network_Score_Statistics as scores

            with h5py.File(self.network_path, "r") as hf:
                metadata = cache_manifest.validate_network_schema(hf)
                is_blast = metadata.network_type == "blast"
                headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in hf['headers'][:]]
                kept_mask = None
                node_count = len(headers)
                if self.fasta_path and os.path.exists(self.fasta_path):
                    fasta_ids, fasta_headers = scores.read_fasta_names(self.fasta_path)
                    mask, missing = scores.kept_node_mask(headers, fasta_ids, fasta_headers)
                    if missing:
                        raise ValueError(translate(
                            "Config",
                            "FASTA file is NOT a strict subset of the network file. "
                            "The network lacks %n sequence(s).",
                            None, missing,
                        ))
                    if not mask.all():
                        kept_mask, node_count = mask, int(mask.sum())
                source = scores.ScoreSource(
                    hf, is_blast=is_blast, score_mode=self.score_mode,
                    norm_mode=self.norm_mode, kept_mask=kept_mask,
                )
                result = scores.compute_score_statistics(
                    source,
                    progress=lambda fraction: self.progress.emit(self.request_id, fraction),
                    cancelled=self.isInterruptionRequested,
                )
            if result.stored_edges == 0:
                return None
            return scores.statistics_report(
                result, model_name=metadata.model_name,
                fasta_name=os.path.basename(self.fasta_path), node_count=node_count,
                is_blast=is_blast, score_mode=self.score_mode, norm_mode=self.norm_mode,
            )

    class ConfigGUI(QMainWindow):
        def __init__(self, carried=None):
            # carried: what a redraw in another language keeps (language_carry_over).
            super().__init__()
            self.setWindowTitle(translate("Config", CONFIG_DISPLAY_NAME))
            
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
            self.tabs = prepare_tab_widget(QTabWidget())
            self.left_top_layout.addWidget(self.tabs)
            
            # Left Bottom: Tool Tip Box + Action Buttons
            self.left_bottom_widget = QWidget()
            self.left_bottom_layout = QVBoxLayout(self.left_bottom_widget)
            self.left_bottom_layout.setContentsMargins(0, 0, 0, 0)
            self.left_bottom_layout.setSpacing(TOKENS["card_gap"])
            
            self.tip_panel = SpacedTipLabel(translate(
                "Config", "Click or tab to an input or its label to see helpful tips here."
            ))
            add_hint_icon(self.tip_panel)
            self.left_bottom_layout.addWidget(self.tip_panel, 1)
            
            action_row = QWidget()
            action_row.setObjectName("configActionButtons")
            action_row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            btn_layout = ResponsiveFlowLayout(action_row)
            self.btn_check = QPushButton(translate("Config", "Consistency Check"))
            self.btn_check.clicked.connect(self.run_consistency_check)
            
            self.btn_save_run = QPushButton(translate("Config", "Save && Run"))
            self.btn_save_run.clicked.connect(self._save_and_run_from_button)
            set_role(self.btn_save_run, "primary")

            self.btn_export_layout = QPushButton(translate("Config", "Export Layout Settings"))
            self.btn_export_layout.clicked.connect(self.export_layout_settings)
            self.btn_export_layout.setEnabled(False)
            self.btn_export_layout.setToolTip(translate(
                "Config", "Export a generation-only JSON file for the selected new layout cache."
            ))
            
            btn_save = QPushButton(translate("Config", "Save"))
            btn_save.clicked.connect(self.save_only)
            
            btn_exit = QPushButton(translate("Config", "Exit"))
            btn_exit.clicked.connect(self.close)
            
            btn_layout.addWidget(self.btn_save_run, 1)
            btn_layout.addWidget(self.btn_export_layout, 1)
            btn_layout.addWidget(self.btn_check, 1)
            btn_layout.addWidget(btn_save, 1)
            btn_layout.addWidget(btn_exit, 1)
            self.left_bottom_layout.addWidget(action_row)
            
            self.left_split.addWidget(self.left_top_widget)
            self.left_split.addWidget(self.left_bottom_widget)
            
            # The initial heights: the tabs get most of the room, the tip panel its
            # fixed height (TIP_PANEL_LINES) and the buttons under it, as many rows of
            # them as the width makes. What the panel shows never changes its height;
            # only the splitter does.
            self.left_split.setSizes([500, self.left_bottom_widget.sizeHint().height()])
            PanelHeightAtFirstLayout(self.left_split)
            
            # Ensure that if the user resizes the window, extra space goes to the tabs, not the bottom
            self.left_split.setStretchFactor(0, 1)
            self.left_split.setStretchFactor(1, 0)
            
            # Right Panel: Statistics
            self.right_panel = QWidget()
            self.right_layout = QVBoxLayout(self.right_panel)
            self.right_layout.setContentsMargins(0, 0, 0, 0)
            self.right_layout.setSpacing(TOKENS["card_gap"])
            self.stat_label = set_role(QLabel(translate("Config", "Network Statistics Report")), "title")
            self.stat_display = StatisticsReport()
            self.stat_display.setReadOnly(True)
            self.stat_display.setPlaceholderText(translate(
                "Config", "Select Fasta subset and HDF5 Network file, then click compute."
            ))
            self.stat_display.setFont(qt_monospace_font(self.stat_display.font()))
            set_role(self.stat_display, "report")
            self.right_layout.addWidget(self.stat_label)
            self.right_layout.addWidget(self.stat_display)
            self.language_selector = LanguageSelector()
            self.language_selector.language_chosen.connect(lambda setting: choose_language(self, setting))
            # A language chosen in another window comes here when this one is next activated.
            self.language_follower = LanguageFollower(self)
            self.theme_switch = ThemeColorButton(CONFIG_THEME_SETTING, color=theme_color())
            self.theme_switch.color_chosen.connect(lambda color: choose_theme_color(self, color))
            self.right_layout.addLayout(language_selector_row(self.language_selector, self.theme_switch))
            
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
            self._custom_settings = (
                dict(carried["custom_settings"]) if carried else self._read_custom_settings()
            )
            self._cache_hash_cache = dict(carried["cache_hashes"]) if carried else {}
            self._carried_cache_choice = None
            self._cache_hash_request_id = 0
            self._cache_hash_workers = {}
            self._cache_hash_pending_keys = None
            # The statistics run in progress (run_statistics): its request id, the inputs it
            # was started on (None while none runs), and the progress line it last wrote.
            self._statistics_request_id = 0
            self._statistics_workers = {}
            self._statistics_request = None
            self._statistics_percent = -1
            self._statistics_status_html = None
            self._cache_launch_allowed = False
            self._save_run_paused = False  # in the pause after a click (_save_and_run_from_button)
            self._last_duplicate_signature = None
            self._scoring_hidden_by_blast = None
            self._statistics = None  # what the statistics report shows (show_statistics)
            self.current_cache_folder = None
            # What the shown cache search looked at and found (_apply_cache_discovery);
            # None while no search result is shown.
            self._saved_cache_look = None
            self._cache_listing_shown = None
            self._label_column = []

            self.create_inputs_tab()
            self.create_visuals_tab()
            self.create_physics_tab()
            self.create_directories_tab()

            self._align_label_column()
            self._prepare_responsive_layouts()
            self._align_report_with_tabs()
            self._initializing_profiles = False
            if carried is None:
                self._load_all_custom_profiles()
            else:
                self._restore_carried(carried)
            
            self.cb_fasta.currentTextChanged.connect(self.update_live_validators)
            self.cb_hdf5.currentTextChanged.connect(self.update_live_validators)
            self.cb_score_mode.currentTextChanged.connect(self.update_live_validators)
            self.cb_norm_mode.currentTextChanged.connect(self.update_live_validators)
            self.line_ref.textChanged.connect(self._update_reference_controls)
            self.spin_thresh.valueChanged.connect(self.update_live_validators)
            self.spin_top.valueChanged.connect(self.update_live_validators)
            self.cb_msa.currentTextChanged.connect(self.update_live_validators)
            self.cb_embed.currentTextChanged.connect(self.update_live_validators)
            
            self.update_live_validators()
            # A cache published later, by a Layout Generator this window
            # launched or by anything else, shows without the inputs being chosen again.
            self._cache_recheck_timer = QTimer(self)
            self._cache_recheck_timer.setInterval(CONFIG_CACHE_RECHECK_MS)
            self._cache_recheck_timer.timeout.connect(self._recheck_saved_caches_while_shown)
            self._cache_recheck_timer.start()
            self.setup_tips()

        def _make_field_group(self, pairs, *, parent=None, name="", ratios=None, **options):
            group = parent if parent is not None else QWidget()
            group.setObjectName(name)
            self._label_column.append(pairs[0][0])
            ResponsiveFieldLayout(
                group, pairs, ratios or tuple(1 for _ in pairs),
                spacing=CONFIG_FIELD_HORIZONTAL_SPACING, wrap_labels=False, **options,
            )
            return group

        def _align_label_column(self):
            """Start every field one field spacing after the longest field label.

            The first label of every row, on every tab, shares one column, so
            fields line up across the tabs. A row of paired fields that stacks
            in a narrow window moves its other labels into that column too, so
            they count toward its width, unless the row gives its own labels
            a stacked width (the Low Resource Mode section does).
            """
            labels = list(dict.fromkeys(self._label_column))
            width = max(label.sizeHint().width() for label in labels)
            for group in self.findChildren(QWidget):
                layout = group.layout()
                if isinstance(layout, ResponsiveFieldLayout) and layout.stacked_label_width is None:
                    width = max([width] + [layout._label_width(label)
                                           for label, _ in layout.pairs[1:]])
            for label in labels:
                label.setFixedWidth(width)
            self.label_column_width = width

        def _responsive_grid_rows(self, grid, name, *, trailing=False):
            rows = {}
            for index in range(grid.count()):
                row, column, _, _ = grid.getItemPosition(index)
                rows.setdefault(row, []).append((column, grid.itemAt(index).widget()))
            while grid.count():
                grid.takeAt(0)
            result = QVBoxLayout()
            result.setContentsMargins(0, 0, 0, 0)
            result.setSpacing(12)
            right_labels = [sorted(cells)[2][1] for cells in rows.values()
                            if len(cells) == 4]
            if right_labels:
                shared_width = max(label.sizeHint().width() for label in right_labels)
                for label in right_labels:
                    label.setFixedWidth(max(label.minimumWidth(), shared_width))
            for row, cells in sorted(rows.items()):
                widgets = [widget for _, widget in sorted(cells)]
                if len(widgets) == 1:
                    result.addWidget(widgets[0])
                    continue
                pairs = list(zip(widgets[::2], widgets[1::2]))
                options = {"equal_fields": not trailing, "column_spacing": 24}
                if trailing:
                    options.update(trailing=True,
                                   control_stretches=(0,) * (len(pairs) - 1) + (1,))
                result.addWidget(self._make_field_group(
                    pairs, name=f"{name}Row{row}", **options,
                ))
            return result

        def _add_scroll_tab(self, content, title):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setMinimumWidth(600)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, title)

        def _align_report_with_tabs(self):
            """Start the report card level with the settings card, and shade both cards."""
            bar = self.tabs.tabBar()
            bar.ensurePolished()
            self.stat_label.setFixedHeight(bar.sizeHint().height())
            self._card_shadows = CardShadows(
                self.central_widget, [self.tabs.findChild(QStackedWidget), self.stat_display]
            )

        def _prepare_responsive_layouts(self):
            for combo in self.findChildren(QComboBox):
                if isinstance(combo, LanguageSelector):
                    continue  # It fits its entries, beside the row's stretch.
                # Twelve characters, unless the combo was given its own length.
                combo.setMinimumContentsLength(combo.minimumContentsLength() or 12)
                combo.setSizeAdjustPolicy(
                    QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
                )
                combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            for editor in self.findChildren(QLineEdit):
                if isinstance(editor.parentWidget(), (QSpinBox, QDoubleSpinBox, QComboBox)):
                    continue
                # A field sharing its cell with a button leaves room for it.
                # The button fits its text, so a long one can take it all.
                shared_width = int(editor.property("sharedCellWidth") or 0)
                # Twelve characters, unless the field asks for fewer.
                characters = int(editor.property("minimumCharacters") or 12)
                editor.setMinimumWidth(max(
                    0, editor.fontMetrics().horizontalAdvance("M" * characters) - shared_width
                ))
            for slider in self.findChildren(QSlider):
                slider.setMinimumWidth(slider.fontMetrics().horizontalAdvance("M" * 8))
            for form in self.findChildren(QFormLayout):
                form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
                form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

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
                for key in DEPRECATED_DIRECTORY_KEYS:
                    data.pop(key, None)
                for key in DIRECTORY_PROFILE_DEFAULTS:
                    if key in data:
                        data[key] = _migrate_default_directory_path(key, data[key])
                # The old single Low Resource Mode switch, as the categories
                # it stood for; a save then writes those, not the old key.
                return dict(migrate_low_resource_mode(data))
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
            return _resolved_saved_config_root(
                value,
                base_directories=self._directory_base_values(),
            )

        def _add_profile_selector(self, tab_id, form_layout):
            container = QWidget()
            row_layout = ResponsiveSelectorLayout(container)
            row_layout.setContentsMargins(0, 0, 0, 0)

            selector = DynamicComboBox(
                lambda selected_tab=tab_id: self._refresh_profile_combo(selected_tab)
            )
            name_input = QLineEdit()
            name_input.setPlaceholderText(translate("Config", "New profile name"))
            name_input.setVisible(False)
            folder_button = icon_button("folder-open", translate("Config", "Open this tab's saved config folder"))
            folder_button.setFixedWidth(30)
            row_layout.addWidget(selector, 1)
            row_layout.addWidget(name_input, 1)
            row_layout.addWidget(folder_button)

            label = QLabel(translate("Config", "Saved Config:"))
            self._label_column.append(label)
            form_layout.addRow(label, container)

            def open_profile_folder(checked=False, selected_tab=tab_id):
                folder = self._saved_config_root() / selected_tab
                os.makedirs(folder, exist_ok=True)
                Desktop_Open.open_path(folder)

            folder_button.clicked.connect(open_profile_folder)

            self.profile_selectors[tab_id] = selector
            self.profile_name_inputs[tab_id] = name_input
            self.profile_folder_buttons[tab_id] = folder_button
            self.profile_labels[tab_id] = label
            self._profile_previous_selection[tab_id] = "(custom)"
            self._refresh_profile_combo(tab_id)
            selector.currentIndexChanged.connect(
                lambda index, selected_tab=tab_id, combo=selector: (
                    self._profile_selection_changed(selected_tab, combo.itemData(index))
                )
            )

        def _profile_special_items(self, tab_id):
            items = ["(custom)"]
            if TAB_PROFILE_SPECS[tab_id]["allow_default"]:
                items.append("(default)")
            items.append("(new)")
            return items

        def _add_padded_separator(self, layout, object_name):
            """The theme's separator line, padded, added to layout (or to none,
            for a grid to place: its wrapper is the line's parent widget)."""
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
            set_role(separator, "separator")
            wrapper_layout.addWidget(separator)
            if isinstance(layout, QFormLayout):
                layout.addRow(wrapper)
            elif layout is not None:
                layout.addWidget(wrapper)
            else:
                # Kept alive until a layout takes it: a widget nothing owns is deleted.
                separator.wrapper = wrapper
            return separator

        def _add_profile_separator(self, tab_id, layout):
            separator = self._add_padded_separator(
                layout, f"{tab_id}_saved_config_separator"
            )
            self.profile_separators[tab_id] = separator

        def _refresh_profile_combo(self, tab_id):
            selector = self.profile_selectors[tab_id]
            current = combo_value(selector) or "(custom)"
            items = _discover_profile_names(self._saved_config_root(), tab_id)
            items.extend(self._profile_special_items(tab_id))
            disappeared = current not in items

            selector.blockSignals(True)
            try:
                selector.clear()
                add_combo_options(selector, items, [
                    translate("Config", PROFILE_ITEM_LABELS[item])
                    if item in PROFILE_ITEM_LABELS else item
                    for item in items
                ])
                select_combo_value(selector, "(custom)" if disappeared else current)
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
            # The errors print in English and show translated (display_text).
            if not isinstance(raw_data, dict):
                raise ValueError(Message("the JSON root must be an object"))

            defaults = TAB_PROFILE_SPECS[tab_id]["defaults"]
            if tab_id == "directories":
                ignored = DEPRECATED_DIRECTORY_KEYS
            elif tab_id == "simulation_physics":
                ignored = OBSOLETE_LAYOUT_ENGINE_KEYS
            else:
                ignored = set()
            unknown = set(raw_data) - set(defaults) - ignored
            if unknown and not allow_extra:
                names = ", ".join(sorted(unknown))
                raise ValueError(Message("unknown setting key(s): {names}", names=names))

            normalized = dict(defaults)
            for key in defaults:
                if key not in raw_data:
                    continue
                value = raw_data[key]
                if tab_id == "directories":
                    value = _migrate_default_directory_path(key, value)
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
                            raise ValueError(Message("expected true or false"))
                    elif isinstance(default, int):
                        if isinstance(value, bool) or float(value) != int(float(value)):
                            raise ValueError(Message("expected an integer"))
                        value = int(float(value))
                    elif isinstance(default, float):
                        if isinstance(value, bool):
                            raise ValueError(Message("expected a number"))
                        value = float(value)
                    else:
                        if not isinstance(value, str):
                            raise ValueError(Message("expected text"))
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        Message("invalid value for {key}: {error}", key=key, error=error)
                    ) from error

                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(Message("invalid value for {key}: expected a finite number", key=key))
                if key in PROFILE_ENUM_VALUES and value not in PROFILE_ENUM_VALUES[key]:
                    allowed = ", ".join(sorted(PROFILE_ENUM_VALUES[key]))
                    raise ValueError(Message(
                        "invalid value for {key}; expected one of: {allowed}", key=key, allowed=allowed
                    ))
                if value is not None and key in PROFILE_RANGES:
                    minimum, maximum = PROFILE_RANGES[key]
                    if value < minimum or value > maximum:
                        raise ValueError(Message(
                            "invalid value for {key}; expected {minimum} to {maximum}",
                            key=key, minimum=minimum, maximum=maximum,
                        ))
                normalized[key] = value

            if (
                normalized.get("ALIGNMENT_SCORE") == "local"
                and normalized.get("NORM_MODE") == "alignment_length"
            ):
                raise ValueError(Message(
                    "NORM_MODE alignment_length is unavailable for local alignment scores"
                ))

            color_keys = {
                "TEXT_COLOR", "INITIAL_NODE_COLOR", "HOVER_COLOR",
                "CONNECTED_NODE_COLOR", "EDGE_COLOR", "NODE_BOUNDARY_COLOR",
            }
            for key in color_keys.intersection(normalized):
                if not QColor(normalized[key]).isValid():
                    raise ValueError(Message(
                        "invalid color value for {key}: {value}", key=key, value=normalized[key]
                    ))
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
            if tab_id == "visual_effects":
                raw_data = migrate_low_resource_mode(raw_data)
            return self._normalize_profile_data(tab_id, raw_data)

        def _set_widget_profile_value(self, key, value):
            if key in LAYOUT_SETTINGS:
                self._set_layout_setting(key, value)
                return
            widget = self.inputs[key]
            if isinstance(widget, OptionalNoScrollDoubleSpinBox):
                widget.setOptionalValue(value)
            elif isinstance(widget, QComboBox):
                if key == "LAYOUT_DEVICE_SELECTION":
                    index = widget.findData(value)
                    if index < 0:
                        widget.addItem(
                            translate("Config", "Unavailable saved device [{device}]").format(device=value),
                            value,
                        )
                        index = widget.count() - 1
                    widget.setCurrentIndex(index)
                elif widget.property("persistItemData"):
                    select_combo_value(widget, str(value))
                else:
                    text_value = str(value)
                    if key in {"NODE_FASTA_FILE", "MSA_FILE", "EMBED_FILE", "INPUT_HDF5"}:
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
            self._set_profile_content_enabled(tab_id, not read_only)
            if hasattr(self, "update_live_validators"):
                self.update_live_validators()

        def _set_profile_content_enabled(self, tab_id, enabled):
            # A locked profile's fields look disabled through the theme's :disabled rules.
            content = self.profile_content_widgets[tab_id]
            roots = list(content) if isinstance(content, (list, tuple)) else [content]
            for root in roots:
                root.setEnabled(enabled)

        def _set_new_profile_field_visible(self, tab_id, visible):
            name_input = self.profile_name_inputs[tab_id]
            name_input.setVisible(visible)
            if visible:
                name_input.setFocus()
            else:
                name_input.clear()

        def _set_profile_selection(self, tab_id, value):
            selector = self.profile_selectors[tab_id]
            selector.blockSignals(True)
            try:
                select_combo_value(selector, value)
            finally:
                selector.blockSignals(False)

        def _profile_selection_changed(self, tab_id, value):
            if self._initializing_profiles or self._profile_loading or not value:
                return
            previous = self._profile_previous_selection.get(tab_id, "(custom)")

            if value == "(new)":
                self._set_new_profile_field_visible(tab_id, True)
                self._set_profile_content_enabled(tab_id, True)
                self._profile_previous_selection[tab_id] = value
                return

            self._set_new_profile_field_visible(tab_id, False)
            try:
                if value == "(custom)":
                    data = self._custom_profile_data(tab_id)
                    read_only = False
                elif value == "(default)":
                    if not TAB_PROFILE_SPECS[tab_id]["allow_default"]:
                        raise ValueError(translate("Config", "this tab does not provide a default profile"))
                    data = dict(TAB_PROFILE_SPECS[tab_id]["defaults"])
                    read_only = True
                else:
                    data = self._named_profile_data(tab_id, value)
                    read_only = False
                self._apply_profile_data(tab_id, data, read_only=read_only)
                self._profile_previous_selection[tab_id] = value
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                self._set_profile_selection(tab_id, previous)
                self._set_new_profile_field_visible(tab_id, previous == "(new)")
                QMessageBox.critical(
                    self,
                    translate("Config", "Saved Config Error"),
                    translate("Config", "Could not load profile '{name}':\n{error}").format(
                        name=value, error=display_text(error)
                    ),
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

        def switch_language(self, language, **options):
            """Redraw this window in language (None for English), keeping all it shows.

            The new window carries over every field as shown, unsaved edits
            included, and appears at this one's size, position, splitter
            positions, tabs and scroll positions. Returns the window that
            shows the language.
            """
            if language == installed_language():
                return self
            carried = self.language_carry_over()
            return redraw_in_language(self, language, lambda: type(self)(carried=carried), **options)

        def switch_theme_color(self, color):
            """Redraw this window in the theme colour named color, keeping all it shows.

            The new window carries over what a language redraw does, at this
            one's place and view. Returns the window that shows the colour.
            """
            if color == theme_color():
                return self
            carried = self.language_carry_over()
            return redraw_in_theme_color(self, color, lambda: type(self)(carried=carried))

        def show_statistics(self, report):
            """Show report, what run_statistics found, in the statistics panel, in the window's language.

            The window keeps the report, so a redraw in another language
            (language_carry_over) shows it again, in that language.
            """
            self._statistics = report
            summary, headings = statistics_report_text(report)
            self.stat_display.setHtml(statistics_report_html(summary, headings, report["rows"]))

        def language_carry_over(self):
            """What a redraw in another language keeps beyond the view."""
            hidden = self._scoring_hidden_by_blast
            blanked = bool(hidden) and self.cb_score_mode.currentIndex() == -1
            return {
                "custom_settings": dict(self._custom_settings),
                "cache_hashes": dict(self._cache_hash_cache),
                # A BLAST network blanks both scoring modes; carry the choice they hide.
                "values": {
                    **{
                        key: hidden[key] if blanked and key in hidden else self._carried_widget_value(key, widget)
                        for key, widget in self.inputs.items()
                    },
                    # The Layout Method dropdown holds two settings: UMAP_MODE and LAYOUT_METHOD.
                    **self._layout_settings(),
                },
                "profiles": {
                    tab_id: (combo_value(self.profile_selectors[tab_id]), self.profile_name_inputs[tab_id].text())
                    for tab_id in TAB_PROFILE_SPECS
                },
                "cache_choice": self._current_cache_choice(),
                # Its figures, not its text, so it comes back in the new language.
                "statistics": self._statistics,
            }

        def _carried_widget_value(self, key, widget):
            """A field's value as _set_widget_profile_value takes it back."""
            if isinstance(widget, OptionalNoScrollDoubleSpinBox):
                return widget.optionalValue()
            if isinstance(widget, QComboBox):
                if key == "LAYOUT_DEVICE_SELECTION":
                    return widget.currentData()
                if widget.property("persistItemData"):
                    return combo_value(widget)
                return widget.currentText()
            if isinstance(widget, QPushButton) and widget.isCheckable():
                return widget.isChecked()
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                return widget.value()
            if isinstance(widget, QLineEdit):
                return widget.text()
            if hasattr(widget, "isChecked"):
                return widget.isChecked()
            return None

        def _restore_carried(self, carried):
            """Show what language_carry_over kept, in place of the saved custom profile."""
            values = carried["values"]
            # Named profiles come from the saved-config folder, itself perhaps unsaved.
            self.inputs["SAVED_CONFIG_DIR"].setText(values["SAVED_CONFIG_DIR"])
            for tab_id in TAB_PROFILE_SPECS:
                self._refresh_profile_combo(tab_id)
            for tab_id in ("directories", "inputs_outputs", "visual_effects", "simulation_physics"):
                selection, new_name = carried["profiles"][tab_id]
                self._set_profile_selection(tab_id, selection)
                self._profile_previous_selection[tab_id] = selection
                # Shown as before, without taking the keyboard focus.
                name_input = self.profile_name_inputs[tab_id]
                name_input.setVisible(selection == "(new)")
                name_input.setText(new_name if selection == "(new)" else "")
                data = {key: values[key] for key in TAB_PROFILE_SPECS[tab_id]["defaults"]}
                self._apply_profile_data(tab_id, data, read_only=selection == "(default)")
            # The cache list fills in once discovery knows the inputs' hashes,
            # at once if they are carried over, else when hashing finishes.
            self._carried_cache_choice = carried["cache_choice"]
            self.update_live_validators()
            if carried["statistics"]:
                self.show_statistics(carried["statistics"])

        def _apply_carried_cache_choice(self):
            choice, self._carried_cache_choice = self._carried_cache_choice, None
            if not choice:
                return
            if choice["new"]:
                index = self.cb_cache_file.count() - 1
            else:
                index = self.cb_cache_file.findData(choice["data"])
            if index >= 0:
                self.cb_cache_file.setCurrentIndex(index)
            if self._new_cache_selected():
                self.line_new_cache.setText(choice["name"])
            self._toggle_new_cache_input()

        def closeEvent(self, event):
            self._cache_hash_request_id += 1
            for worker in self._cache_hash_workers.values():
                worker.requestInterruption()
            self._cancel_statistics()
            for worker in self._cache_hash_workers.values():
                worker.wait(5000)
            for worker in list(self._statistics_workers.values()):
                worker.wait(5000)
            super().closeEvent(event)
            
        def run_consistency_check(self):
            import h5py

            # 1. Define the file names by grabbing them from the UI dropdowns
            fasta_file = self.cb_fasta.currentText()
            hdf5_file = self.cb_hdf5.currentText()
            msa_file = self.cb_msa.currentText()

            # 2. Halt if the user hasn't selected the required files
            if not fasta_file or not hdf5_file:
                return

            # 3. Build the paths safely
            fasta_path = os.path.join(
                self._resolved_directory_input("FASTA_DIR"), fasta_file
            )
            hdf5_path = os.path.join(
                self._resolved_directory_input("HDF5_DIR"), hdf5_file
            )
            msa_path = (
                os.path.join(self._resolved_directory_input("MSA_DIR"), msa_file)
                if msa_file else None
            )

            self.tip_panel.setStatus("info", translate("Config", "Running Consistency Check..."))
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()

            try:
                fasta_headers = _load_consistency_fasta_headers(fasta_path)

                with h5py.File(hdf5_path, "r") as hf:
                    raw_headers = hf['headers'][:]
                    headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]

                # One row per check, in the order the Viewer needs them.
                checks = [network_consistency_check(fasta_headers, headers)]

                msa_headers = None
                if msa_path and os.path.exists(msa_path):
                    msa_headers = set(_load_consistency_msa_headers(msa_path))
                    checks.append(msa_consistency_check(fasta_headers, msa_headers))

                embed_file = self.cb_embed.currentText()
                if embed_file:
                    embed_path = os.path.join(
                        self._resolved_directory_input("EMBED_DIR"), embed_file
                    )
                    checks.append(embedding_consistency_check(
                        embed_path, fasta_path, fasta_headers, hdf5_path
                    ))

                # Check Reference ID if provided. Resolve it exactly as the
                # Viewer will: the same rules over the same candidates, the
                # network headers kept by the FASTA, in network order.
                ref_id = self.line_ref.text().strip()
                if ref_id:
                    fasta_set = set(fasta_headers)
                    candidates = [header for header in headers if header in fasta_set]
                    checks.append(reference_consistency_check(ref_id, candidates, msa_headers))

                # The corner icon shows the worst row's status; each row keeps its own.
                self.tip_panel.setReport(consistency_report_html(checks), consistency_verdict(checks))

            except Exception as e:
                self.tip_panel.setStatus("danger", translate(
                    "Config", "Error during consistency check: {error}"
                ).format(error=display_text(e)))
            
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
                        # Hovering, and focus that Qt moves by itself, only
                        # replace another tip. A status message stays until
                        # the user picks an element with a tip by a left click
                        # or the keyboard; then hovering works again.
                        if self.tip_panel.shows_tip() or self._picks_tip_target(event):
                            self.tip_panel.setTip(tip)
                        if event_type == QEvent.Type.ToolTip:
                            return True
            return super().eventFilter(obj, event)

        @staticmethod
        def _picks_tip_target(event):
            """Whether event is a left click, or focus by Tab, Shift+Tab or a shortcut."""
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Type.MouseButtonPress:
                return event.button() == Qt.MouseButton.LeftButton
            if event.type() == QEvent.Type.FocusIn:
                return event.reason() in (
                    Qt.FocusReason.TabFocusReason,
                    Qt.FocusReason.BacktabFocusReason,
                    Qt.FocusReason.ShortcutFocusReason,
                )
            return False

        def _register_tip_targets(self, widget, tip, *, overwrite=True):
            if widget is None:
                return
            targets = [widget]
            targets.extend(widget.findChildren(QWidget))
            for target in targets:
                if overwrite or target not in self.tip_db:
                    self.tip_db[target] = tip
                    target.installEventFilter(self)

        def _folder_note(self, key, *, space=False):
            """The live notes of a directory field: the folder it resolves to now, and with space the room free there."""
            def notes(field):
                folder = self._resolved_directory_input(key)
                if not folder:
                    return None
                found = [LiveNote(translate("Config", "Resolves to {folder}."), {"folder": Code(folder)})]
                free = free_bytes(folder) if space else None
                if free is not None:
                    found.append(LiveNote(
                        translate("Config", "{free} free on its drive."),
                        {"free": Code(f"{free / 1024 ** 3:.1f} GiB")},
                    ))
                return found

            return notes

        def _field_tip(self, key, tip):
            """What tip_db holds for the field key: plain text as it is, a Tip with its field.

            The panel reads the title, the options, the range and the default
            from the field when the tip is shown.
            """
            if not isinstance(tip, Tip):
                return tip
            entry = self._field_tips.get(key)
            if entry is None:
                shown_key, default = key, _profile_default(key)
                if key == "LAYOUT_METHOD":
                    # The dropdown stands for both its settings, and starts as SSN.
                    shown_key = ", ".join(LAYOUT_SETTINGS)
                    default = layout_choice(_profile_default("UMAP_MODE"), default)
                entry = self._field_tips[key] = FieldTip(tip, Field(
                    key=shown_key,
                    label=self.labels.get(key),
                    widget=self.inputs.get(key),
                    default=default,
                    limits=PROFILE_RANGES.get(key),
                ))
                if tip.live is not None:
                    signal = change_signal(entry.field.widget)
                    if signal is not None:
                        signal.connect(self._live_tip_changed)
            return entry

        def _live_tip_changed(self, *_):
            """A field with a live note changed: the tip on display shows what holds now."""
            self.tip_panel.refresh()

        def setup_tips(self):
            self._field_tips = {}
            self.tip_db_keys = {
                "SAVED_CONFIG": Tip(
                    translate("Config", "Selects the settings profile used for this tab."),
                    details=(translate("Config", "A named entry loads that profile from the Saved Config Directory."),),
                    options=(
                        TipOption("(custom)", translate("Config", "Uses the current values in `{file}`.").format(
                            file=os.path.basename(DEFAULT_SETTINGS_FILE)
                        )),
                        TipOption("(default)", translate("Config", "Uses the built-in defaults, which are read-only.")),
                        TipOption("(new)", translate("Config", "Creates a named profile.")),
                    ),
                    facts=False,
                ),
                "NODE_FASTA_FILE": Tip(
                    translate("Config", "Primary FASTA file containing sequences visualized as nodes in the SSN."),
                    details=(translate("Config", "Must match sequences present in the selected network edges and multiple alignments."),),
                ),
                "MSA_FILE": Tip(
                    translate("Config", "Multiple sequence alignment file (`.fasta`, `.h5`, or `_sparse.h5`) for the sequence set."),
                    details=(translate("Config", "Used to calculate positional conservation, gaps, and occupancy thresholds during analysis."),),
                ),
                "EMBED_FILE": Tip(
                    translate("Config", "Optional protein language model embedding file (`.h5`) from Generate Embeddings, for commands that analyse embeddings."),
                    details=(translate("Config", "Matched to nodes by header and sequence; the per-residue data stays on disk until a command reads it."),),
                    notes=(
                        TipNote("info", translate("Config", "Keep the file on this device.")),
                    ),
                ),
                "INPUT_HDF5": Tip(
                    translate("Config", "Network or similarity matrix file (`.h5`) containing pairwise sequence similarity scores and edge coordinates."),
                    details=(translate("Config", "Must contain alignment metrics for at least all sequences present in the active sequence set."),),
                ),
                "ALIGNMENT_SCORE": Tip(
                    translate("Config", "(For embedding SSNs) Specifies whether to use global (Needleman-Wunsch) or local (Smith-Waterman) scores."),
                    options=(
                        TipOption("global", translate("Config", "Best for full-length comparisons.")),
                        TipOption("local", translate("Config", "Recommended for multi-domain proteins.")),
                    ),
                ),
                "NORM_MODE": Tip(
                    translate("Config", "(For embedding SSNs) Normalization strategy for pairwise sequence alignment scores."),
                    details=(translate("Config", "Reduces length bias."),),
                    options=(
                        TipOption("alignment_length", translate("Config", "Divides the score by the alignment length.")),
                        TipOption("shorter_sequence", translate("Config", "Divides the score by the length of the shorter sequence.")),
                        TipOption("longer_sequence", translate("Config", "Divides the score by the length of the longer sequence.")),
                        TipOption("average_sequence", translate("Config", "Divides the score by the average length of the two sequences.")),
                    ),
                ),
                "ALIGNMENT_REFERENCE": Tip(
                    translate("Config", "Full header, leading identifier (e.g. `WP_0123.1`), substring, or wildcard (e.g. `WP_01*`) naming the reference sequence; an exact header or identifier takes priority."),
                    details=(translate("Config", "Used to anchor absolute relative residue numbering and mapping offsets across the entire network."),),
                ),
                "ALIGNMENT_OFFSET": Tip(
                    translate("Config", "Integer offset added to reference-anchored alignment residue positions (e.g. `+10` shifts position 1 to 11)."),
                    notes=(
                        TipNote("info", translate("Config", "Applied only when the Alignment Reference ID resolves successfully in the alignment.")),
                    ),
                ),
                "SIMILARITY_THRESHOLD": Tip(
                    translate("Config", "Minimum similarity score threshold (identity fraction, normalized score, or -Log10 E-Value) to retain an edge."),
                    details=(translate("Config", "Edges below this cutoff are filtered out and excluded from physics simulation and rendering."),),
                ),
                "TOP_EDGE_PERCENT": Tip(
                    translate("Config", "Alternative edge filter that retains only the top N% highest-scoring edges in the network."),
                    details=(translate("Config", "Maintains consistent network connectivity and density without manually tuning raw score cutoffs (overrides threshold)."),),
                ),
                "FILTER_MIN_OCCUPANCY": Tip(
                    translate("Config", "Minimum percentage of non-gap characters required at an alignment column to retain it in residue analyses."),
                    details=(translate("Config", "Columns with occupancy below this percentage are excluded from logo and conservation calculations."),),
                ),
                "TARGET_CACHE": Tip(
                    translate("Config", "Compatibility-specific folder for layout cache files matching the selected sequence set and network."),
                    details=(translate("Config", "The status reports whether a compatible target folder is available; the folder button opens the layout-cache root."),),
                    facts=False,
                ),
                "TARGET_CACHE_FILE": Tip(
                    translate("Config", "Selects a pre-computed 2D layout coordinate cache file (`.h5`) from the cache directory."),
                    details=(translate("Config", "Instantly restores previously computed node positions to bypass physics simulation."),),
                ),
                "NEW_CACHE_NAME": Tip(
                    translate("Config", "Specifies a custom filename when saving a new layout configuration iteration."),
                    details=(translate("Config", "Only editable when Selected Cache File is set to `{entry}`.").format(
                        entry=translate("Config", "(New Layout Cache)")
                    ),),
                ),
                "NODE_SIZE": Tip(
                    translate("Config", "Visual rendering diameter (in pixels) for each sequence node in the network plot."),
                    details=(translate("Config", "Adjust to optimize visual density; smaller node sizes are recommended for large networks."),),
                    unit="px",
                ),
                "EDGE_WIDTH": Tip(
                    translate("Config", "Line thickness (in pixels) of connection lines drawn between related sequence nodes."),
                    details=(translate("Config", "Thinner lines reduce visual clutter in dense networks; thicker lines highlight strong relationships."),),
                    unit="px",
                ),
                "EDGE_ALPHA": Tip(
                    translate("Config", "Opacity of network edge lines, ranging from `0.0` (fully transparent) to `1.0` (opaque)."),
                    details=(translate("Config", "Lower opacity reveals underlying node clustering and density in highly connected graphs."),),
                ),
                "TEXT_SIZE": Tip(
                    translate("Config", "Font size used for rendering cluster annotations, node labels, and sequence IDs in the visualizer."),
                    details=(translate("Config", "Adjust to ensure labels remain legible without obstructing structural network features."),),
                ),
                "TEXT_COLOR": Tip(
                    translate("Config", "Color of viewer HUD text and control labels."),
                    details=(
                        translate("Config", "A nonzero Hidden Nodes count is shown in a dark shade of the theme color instead."),
                        translate("Config", "Can be specified as a standard color name or hex code (e.g. `grey`, `#333333`)."),
                    ),
                    live=_current_color_note,
                ),
                "INITIAL_NODE_COLOR": Tip(
                    translate("Config", "Baseline fill color applied to all nodes when the network is first loaded."),
                    details=(translate("Config", "Serves as the default background color before custom cluster or metadata coloring is applied."),),
                    live=_current_color_note,
                ),
                "HOVER_COLOR": Tip(
                    translate("Config", "Highlight color applied to a node and its adjacent connections on hover or selection."),
                    details=(translate("Config", "Provides high-contrast interactive visual feedback in the viewer."),),
                    live=_current_color_note,
                ),
                "CONNECTED_NODE_COLOR": Tip(
                    translate("Config", "Border highlight color applied to neighboring nodes directly connected to the currently selected node."),
                    notes=(
                        TipNote("info", translate("Config", "Set it to the same rendered RGBA color as Node Boundary Color to disable connected-node identification, border highlighting, and render promotion.")),
                    ),
                    live=_current_color_note,
                ),
                "EDGE_COLOR": Tip(
                    translate("Config", "Color of connection lines (edges) drawn between similar nodes in the network."),
                    details=(translate("Config", "Lighter or neutral colors reduce edge dominance in dense network clusters."),),
                    live=_current_color_note,
                ),
                "NODE_BOUNDARY_COLOR": Tip(
                    translate("Config", "Color of the outer border ring outline drawn around each sequence node."),
                    details=(translate("Config", "Provides visual contrast to cleanly separate adjacent and overlapping nodes."),),
                    live=_current_color_note,
                ),
                "NODE_BOUNDARY_WIDTH": Tip(
                    translate("Config", "Stroke width (in pixels) of the outer border ring outline drawn around each node."),
                    details=(translate("Config", "Setting a non-zero width helps distinguish overlapping nodes in dense clusters."),),
                    unit="px",
                ),
                "FEWER_EDGES_WHILE_MOVING": Tip(
                    translate("Config", "While you pan, zoom or drag nodes, draws every 4th, 16th or 64th edge in score order, and every edge again a moment after the view stops."),
                    details=(
                        translate("Config", "The step follows how long the edges take to draw on this computer. The edges drawn are made more opaque, so dense regions keep their shade."),
                        translate("Config", "Fewer than 250,000 edges are always drawn in full."),
                    ),
                    notes=(
                        TipNote("info", translate("Config", "Prints and captures always draw every edge.")),
                        TipNote("info", translate("Config", "The environment variable `SSN_EDGE_MOTION_LOD=0` turns it off whatever this switch says.")),
                        TipNote("info", translate("Config", "With Layout Method set to UMAP or PaCMAP, only the selected nodes' edges are drawn, so edges are left out only while the selection has 250,000 or more of them, and nodes cannot be dragged.")),
                        TipNote("warning", translate("Config", "While it is on, the graphics card holds thinned copies of the drawn edges: about a third more memory for them.")),
                    ),
                ),
                "LOW_RESOURCE_ANIMATIONS": Tip(
                    translate("Config", "Skips the Viewer's animations: the wheel zoom without the glide, and the sidebar without the slide."),
                    details=(translate("Config", "Each wheel step zooms at once, and the sidebar opens and closes at once, sparing the frames in between."),),
                ),
                "LOW_RESOURCE_DRAWING": Tip(
                    translate("Config", "Redraws less of the network while you drag, and catches up when you let go."),
                    details=(
                        translate("Config", "The threshold slider redraws the edges only when you let go; the threshold box still follows it."),
                        translate("Config", "A right-button drag of selected nodes moves them and their edges as they are drawn; the selection and neighbour colouring is worked out again when you let go."),
                        translate("Config", "A right-button selection box colours the selected nodes and their neighbours only when you let go."),
                    ),
                    notes=(
                        TipNote("info", translate("Config", "With Layout Method set to UMAP or PaCMAP, nodes cannot be dragged, so only the threshold slider and the selection box apply.")),
                    ),
                ),
                "LOW_RESOURCE_WEBSERVER": Tip(
                    translate("Config", "Sends the Viewer's web pages fewer updates while you work in the network."),
                    details=(
                        translate("Config", "The Meta Data sheet updates only when a node drag ends, instead of when it starts."),
                        translate("Config", "It updates once a right-button selection box is let go, instead of at each move of the box."),
                    ),
                    notes=(
                        TipNote("info", translate("Config", "With Layout Method set to UMAP or PaCMAP, nodes cannot be dragged, so only the selection box applies.")),
                    ),
                ),
                "SHOW_FPS": Tip(
                    translate("Config", "Shows the Viewer's frame rate at its top right, left of the sidebar button."),
                    details=(translate("Config", "It counts the frames drawn to the screen over the last second, with the graphics card's time for a recent frame and, while the edges are thinned, the share drawn. It reads Idle while nothing is drawn."),),
                    notes=(
                        TipNote("info", translate("Config", "It is never part of prints or captures.")),
                    ),
                ),
                "LAYOUT_DEVICE_SELECTION": Tip(
                    translate("Config", "Selects the compute device used for SSN layout generation (CPU, CUDA, XPU, MPS)."),
                    options=(
                        TipOption("auto", translate("Config", "Measures representative workloads and may choose a different backend for each component-size class.")),
                    ),
                    live=_detected_devices_note,
                ),
                "SPRING_K": Tip(
                    translate("Config", "Attractive Hookean spring constant pulling nodes joined by retained network edges closer together."),
                    details=(translate("Config", "Every retained edge has the same spring strength; its similarity score determines filtering, not attraction strength."),),
                ),
                "COULOMB_K": Tip(
                    translate("Config", "Repulsive constant controlling the electrostatic-like force between nodes in the same connected component."),
                    details=(translate("Config", "Larger values spread nearby nodes apart; disconnected components are positioned later by component packing."),),
                ),
                "COULOMB_CUTOFF": Tip(
                    translate("Config", "Maximum spatial distance threshold beyond which node repulsive forces drop to zero."),
                    details=(translate("Config", "Lower cutoffs accelerate computation and prevent distant clusters from exerting unnecessary forces."),),
                ),
                "DAMPING": Tip(
                    translate("Config", "Frictional resistance coefficient applied to node velocities during layout simulation."),
                    details=(translate("Config", "Higher values dissipate kinetic energy and suppress oscillatory motion more quickly."),),
                ),
                "DT": Tip(
                    translate("Config", "Timestep size for each numerical integration step of the physics simulation."),
                    details=(translate("Config", "Smaller timesteps increase stability and precision; larger timesteps speed up convergence but may jitter."),),
                ),
                "AUTO_DT": Tip(
                    translate("Config", "When ON, each simulation stage uses the largest step size that keeps it stable, worked out from its springs and repulsion, and the Step Size field is ignored."),
                    details=(translate("Config", "Max Steps, RMSD Threshold and RMSD Window still count steps of whatever size each stage uses."),),
                ),
                "MAX_STEPS": Tip(
                    translate("Config", "Maximum number of physics iterations the simulation engine will run before terminating."),
                    details=(translate("Config", "Ensure this is large enough to allow node positions to settle into a stable configuration."),),
                ),
                "RMSD_THRESHOLD": Tip(
                    translate("Config", "Root-Mean-Square Deviation convergence threshold for early simulation termination."),
                    details=(translate("Config", "If average node displacement between consecutive steps falls below this value, layout halts as converged."),),
                ),
                "PERCENTAGE_DROP_THRESHOLD": Tip(
                    translate("Config", "Early termination threshold based on the rate of RMSD change over the moving window."),
                    details=(translate("Config", "Terminates simulation when layout change plateaus (set to `0` to disable)."),),
                ),
                "RMSD_WINDOW": Tip(
                    translate("Config", "Number of simulation steps over which moving-average RMSD is calculated for plateau detection."),
                    details=(translate("Config", "Smoothes transient velocity spikes to ensure early termination triggers only on true convergence."),),
                ),
                "ENABLE_PROGRESSIVE_SIMULATION": Tip(
                    translate("Config", "Progressively lowers the similarity threshold in stages for massive connected components."),
                    details=(translate("Config", "Helps resolve fine-grained sub-clusters and prevents gridlock in large, dense components."),),
                ),
                "PACKING_GEOMETRY": Tip(
                    translate("Config", "Macro-level boundary packing geometry (Square or Circle) used to arrange disconnected components."),
                    details=(translate("Config", "Controls how independent clusters are organized in the overall visualization window."),),
                ),
                "PACKING_GRID_SIZE": Tip(
                    translate("Config", "Cell size of the grid on which disconnected components are packed around the largest one."),
                    details=(translate("Config", "Smaller cells pack components more tightly but take longer; the clearance kept between components does not change."),),
                ),
                "LAYOUT_METHOD": Tip(
                    translate("Config", "How the 2D coordinates are computed: SSN simulates physics, while UMAP and PaCMAP compute them directly from each node's strongest edges."),
                    details=(translate("Config", "UMAP and PaCMAP give fast non-linear dimensionality reduction as an alternative to iterative physics simulations."),),
                    options=(
                        TipOption("ssn", translate("Config", "Runs a force-directed network simulation: edges pull their nodes together and nodes push each other apart, as set on the Simulation & Physics tab.")),
                        TipOption("umap", translate("Config", "Keeps each node close to its nearest neighbors; clusters land wherever the optimizer leaves them.")),
                        TipOption("pacmap", translate("Config", "Also looks up the stored scores of random pairs of nodes, so clusters end up in a meaningful order relative to each other.")),
                    ),
                    notes=(
                        TipNote("info", translate("Config", "UMAP and PaCMAP use only each node's strongest edges, so neither uses the Similarity Threshold or Top Edge %, and the Simulation & Physics tab is hidden while one is chosen.")),
                    ),
                ),
"UMAP_NEIGHBORS": Tip(
                    translate("Config", "Maximum number of other nodes in each UMAP neighborhood (K excludes self)."),
                    details=(translate("Config", "`K=15` supplies up to 15 neighbors plus self to UMAP."), translate("Config", "Smaller values emphasize local sub-clusters; larger values preserve broad global relationships.")),
                ),
                "UMAP_MIN_DIST": Tip(
                    translate("Config", "Minimum distance between points in low-dimensional UMAP space (`0.0` to `1.0`)."),
                    details=(translate("Config", "Lower values produce tight, dense point clusters; larger values distribute nodes more evenly."),),
                    notes=(
                        TipNote("info", translate("Config", "PaCMAP does not use it.")),
                    ),
                ),
                "PACMAP_NEIGHBORS": Tip(
                    translate("Config", "Number of each node's strongest edges that PaCMAP keeps close together (`2` to `50`)."),
                    details=(
                        translate("Config", "These are also the edges kept and drawn for each node, as a source and as a target."),
                        translate("Config", "Smaller values emphasize local sub-clusters; larger values hold clusters together more firmly."),
                    ),
                ),
"FASTA_DIR": Tip(
                    translate("Config", "Directory containing input FASTA files for sequence sets and subsets."),
                    details=(translate("Config", "Populates the Sequence Set dropdown in the Inputs tab."),),
                    live=self._folder_note("FASTA_DIR"),
                ),
                "MSA_DIR": Tip(
                    translate("Config", "Directory containing multiple sequence alignment files (`.fasta`, `.h5`, or `_sparse.h5`)."),
                    details=(translate("Config", "Populates the MSA dropdown in the Inputs tab."),),
                    live=self._folder_note("MSA_DIR"),
                ),
                "HDF5_DIR": Tip(
                    translate("Config", "Directory containing HDF5 pairwise sequence similarity scores and network edge files."),
                    details=(translate("Config", "Populates the Network Edges dropdown in the Inputs tab."),),
                    live=self._folder_note("HDF5_DIR"),
                ),
                "SAVED_LAYOUT_DIR": Tip(
                    translate("Config", "Directory where calculated 2D layout coordinate files and network metadata (`.h5`) are saved and loaded."),
                    details=(translate("Config", "Serves as the layout cache to avoid recalculating layouts when reopening networks."),),
                    live=self._folder_note("SAVED_LAYOUT_DIR"),
                ),
                "SETTING_EXPORT_DIR": Tip(
                    translate("Config", "Directory where command-line layout generation JSON settings are exported."),
                    live=self._folder_note("SETTING_EXPORT_DIR", space=True),
                ),
                "METADATA_DIR": Tip(
                    translate("Config", "Directory where uploaded node metadata spreadsheets and CSV files are stored and loaded."),
                    details=(translate("Config", "Used for custom node coloring, categorization, and annotation in the visualizer."),),
                    live=self._folder_note("METADATA_DIR"),
                ),
                "HEADER_LIST_DIR": Tip(
                    translate("Config", "Directory containing text files with lists of sequence headers matching network query criteria."),
                    details=(translate("Config", "Used to store and track sequence cohorts identified in the visualizer."),),
                    live=self._folder_note("HEADER_LIST_DIR"),
                ),
                "INPUT_FILE_DIR": Tip(
                    translate("Config", "Base directory represented by `$input_file$`."),
                    details=(translate("Config", "Relative paths are resolved from the project root; absolute paths remain absolute."),),
                    live=self._folder_note("INPUT_FILE_DIR"),
                ),
                "CACHE_FILE_DIR": Tip(
                    translate("Config", "Base directory represented by `$cache_file$`."),
                    details=(translate("Config", "Relative paths are resolved from the project root; absolute paths remain absolute."),),
                    live=self._folder_note("CACHE_FILE_DIR"),
                ),
                "ANALYSIS_RESULT_DIR": Tip(
                    translate("Config", "Base directory represented by `$analysis_result$`."),
                    details=(translate("Config", "Viewer commands place their output subdirectories beneath this location."),),
                    live=self._folder_note("ANALYSIS_RESULT_DIR"),
                ),
                "EMBED_DIR": Tip(
                    translate("Config", "Base directory represented by `$embedding$`."),
                    details=(translate("Config", "Holds the protein language model embedding files (`.h5`) the Tools write; the Embedding Input dropdown lists the `.h5` files in it."),),
                    live=self._folder_note("EMBED_DIR"),
                ),
                "SAVED_CONFIG_DIR": Tip(
                    translate("Config", "Directory containing named per-tab configuration profiles."),
                    details=(translate("Config", "The default uses `$cache_file$` and follows the Cache File Directory; custom relative and absolute paths remain supported."),),
                    live=self._folder_note("SAVED_CONFIG_DIR"),
                ),
            }
            
            self.tip_db = {}
            for key, tip in self.tip_db_keys.items():
                entry = self._field_tip(key, tip)
                if key in self.labels:
                    self._register_tip_targets(self.labels[key], entry)
                
                if key in self.inputs:
                    self._register_tip_targets(self.inputs[key], entry)
                    
            for key, tip in self.tip_db_keys.items():
                if key in self.inputs:
                    widget = self.inputs[key]
                    parent = widget.parentWidget()
                    if parent and parent.objectName() == "wrapper":
                        self._register_tip_targets(
                            parent, self._field_tip(key, tip), overwrite=False
                        )

            saved_config_tip = self.tip_db_keys["SAVED_CONFIG"]
            for tab_id, selector in self.profile_selectors.items():
                # Each tab's selector lists its own entries ((default) is not on every tab).
                entry = FieldTip(saved_config_tip, Field(
                    label=self.profile_labels.get(tab_id), widget=selector
                ))
                self._register_tip_targets(self.profile_labels.get(tab_id), entry)
                self._register_tip_targets(selector.parentWidget(), entry)

            target_cache_tip = self._field_tip("TARGET_CACHE", self.tip_db_keys["TARGET_CACHE"])
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
        
        def _new_cache_selected(self, index=None):
            # "(New Layout Cache)" is the only entry without a cache path.
            if index is None:
                index = self.cb_cache_file.currentIndex()
            return index >= 0 and self.cb_cache_file.itemData(index) is None

        def _toggle_new_cache_input(self):
            is_new_layout = self._new_cache_selected()
            self.line_new_cache.setVisible(is_new_layout)
            self.line_new_cache.setEnabled(is_new_layout)
            self.btn_export_layout.setEnabled(
                is_new_layout and self._cache_launch_allowed
            )
            if not is_new_layout:
                self.line_new_cache.clear()

        def _cache_file_activated(self, index):
            if self._new_cache_selected(index):
                self.line_new_cache.setFocus()

        def _directory_base_values(self):
            return {
                key: (
                    self.inputs[key].text()
                    if key in self.inputs
                    else globals().get(key, DIRECTORY_PROFILE_DEFAULTS[key])
                )
                for key in (
                    "INPUT_FILE_DIR",
                    "CACHE_FILE_DIR",
                    "ANALYSIS_RESULT_DIR",
                    "EMBED_DIR",
                )
            }

        def _resolved_directory_value(self, value):
            return resolve_directory_path(
                value,
                base_directories=self._directory_base_values(),
            )

        def _resolved_directory_input(self, key):
            widget = self.inputs.get(key)
            value = widget.text() if widget is not None else globals().get(key, "")
            return self._resolved_directory_value(value)

        def _base_directory_changed(self, base_key):
            if base_key == "INPUT_FILE_DIR":
                for combo, key, extensions in (
                    (self.cb_fasta, "FASTA_DIR", [".fasta"]),
                    (self.cb_msa, "MSA_DIR", [".fasta", ".h5"]),
                    (self.cb_embed, "EMBED_DIR", [".h5"]),
                    (self.cb_hdf5, "HDF5_DIR", [".h5"]),
                ):
                    self.refresh_combo(combo, key, extensions)
                self._cancel_statistics_if_inputs_changed()
            elif base_key == "CACHE_FILE_DIR":
                self.update_live_validators()
            elif base_key == "EMBED_DIR":
                self.refresh_combo(self.cb_embed, "EMBED_DIR", [".h5"])

        def _absolute_project_path(self, value):
            path = Path(self._resolved_directory_value(value)).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            return path.resolve()

        def _refresh_cache_file_combo(self):
            # The list opens on what is there now: a compatible folder, too.
            self._recheck_saved_caches()
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
                    self._resolved_directory_input("SAVED_LAYOUT_DIR")
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

            new_cache_was_selected = self._new_cache_selected()
            current_data = self.cb_cache_file.currentData()
            self.cb_cache_file.blockSignals(True)
            try:
                self.cb_cache_file.clear()
                for filename, relative_path in cache_items:
                    self.cb_cache_file.addItem(filename, relative_path)
                self.cb_cache_file.addItem(translate("Config", "(New Layout Cache)"), None)

                if new_cache_was_selected:
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
            self._toggle_new_cache_input()

        def refresh_combo(self, combo, dir_key, ext_list):
            import os  # Moved here to ensure it's loaded before use
            if dir_key not in self.inputs: 
                return
            combo.blockSignals(True)
            current = combo.currentText()
            combo.clear()
            combo.addItem("")
            dir_path = self._resolved_directory_input(dir_key)
            
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if any(f.endswith(ext) for ext in ext_list)]
                combo.addItems(files)
                if current in files:
                    combo.setCurrentText(current)
            combo.blockSignals(False)
            self.update_live_validators()

        def _show_cache_status(self, text, status, bold=False):
            """Show the target cache line in a status colour ("success", "danger" or "subtle")."""
            self.lbl_cache_tracker.setText(text)
            weight = " font-weight: 600;" if bold else ""
            self.lbl_cache_tracker.setStyleSheet(f"color: {status_color(status)};{weight}")

        def _set_cache_unavailable(self, message, status="subtle"):
            self._carried_cache_choice = None
            self._saved_cache_look = None
            self._cache_listing_shown = None
            self._cache_launch_allowed = False
            self._update_save_run_button()
            self.current_cache_folder = None
            self._show_cache_status(message, status)
            self.cb_cache_file.blockSignals(True)
            self.cb_cache_file.clear()
            self.cb_cache_file.setEnabled(False)
            self.cb_cache_file.blockSignals(False)
            self._toggle_new_cache_input()
            self.btn_open_target_folder.setEnabled(False)

        def _cache_paths_from_inputs(self):
            fasta_name = self.cb_fasta.currentText().strip()
            network_name = self.cb_hdf5.currentText().strip()
            if not fasta_name or not network_name:
                return None, None
            fasta_path = os.path.join(
                self._resolved_directory_input("FASTA_DIR"), fasta_name
            )
            network_path = os.path.join(
                self._resolved_directory_input("HDF5_DIR"), network_name
            )
            return os.path.abspath(fasta_path), os.path.abspath(network_path)

        def _cache_setting_values(self):
            top_value = self.spin_top.optionalValue()
            threshold_value = self.spin_thresh.optionalValue()
            return {
                "alignment_score": combo_value(self.cb_score_mode) or None,
                "normalization": combo_value(self.cb_norm_mode) or None,
                "umap_mode": self.uses_manifold_layout(),
                "umap_neighbors": self.spin_umap_k.value(),
                "layout_method": self._layout_settings()["LAYOUT_METHOD"],
                "pacmap_neighbors": self.spin_pacmap_k.value(),
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
                # BLAST scores use neither mode, so both controls go blank.
                # Keep the alignment choice they showed: Save writes it and
                # leaving BLAST restores it.
                if self.cb_score_mode.currentIndex() != -1:
                    self._scoring_hidden_by_blast = {
                        "ALIGNMENT_SCORE": combo_value(self.cb_score_mode),
                        "NORM_MODE": combo_value(self.cb_norm_mode),
                    }
                self.cb_score_mode.setCurrentIndex(-1)
                self.cb_norm_mode.setCurrentIndex(-1)
            else:
                hidden = self._scoring_hidden_by_blast or {
                    "ALIGNMENT_SCORE": "global",
                    "NORM_MODE": "alignment_length",
                }
                self._scoring_hidden_by_blast = None
                restore = self.cb_score_mode.currentIndex() == -1
                if restore:
                    select_combo_value(self.cb_score_mode, hidden["ALIGNMENT_SCORE"])
                # Signals are blocked above, so the currentTextChanged-driven
                # refresh of cb_norm_mode's item list never fires here. Refresh
                # it explicitly or a stale item list (e.g. left over from
                # "local" mode) can silently reject the value restored below.
                self.update_norm_mode_options()
                if restore:
                    select_combo_value(self.cb_norm_mode, hidden["NORM_MODE"])
            self.cb_score_mode.blockSignals(False)
            self.cb_norm_mode.blockSignals(False)

        def _default_new_cache_name(self, folder_path):
            return cache_manifest.next_cache_version_filename(folder_path)

        def _apply_cache_discovery(self, records, refresh=False):
            """Show the cache folder these inputs and settings match, and its caches.

            refresh=True is another look for the same inputs (_recheck_saved_caches).
            Finding what is shown, it leaves the window alone; otherwise a
            cache that has appeared, most likely one just generated, is
            selected, or else the cache chosen stays chosen where it can.
            """
            sequence_path, network_path = self._cache_paths_from_inputs()
            if not sequence_path or not network_path:
                self._set_cache_unavailable(translate("Config", "Target Cache: Missing input files"))
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
                    self._resolved_directory_input("SAVED_LAYOUT_DIR")
                )
                canonical_name = cache_manifest.build_canonical_cache_name(
                    sequence_path,
                    network_path,
                    network_type,
                    **settings,
                )
                default_folder = cache_manifest.resolve_default_cache_folder(
                    saved_layout_dir, canonical_name, compatibility
                )
                # Read before the search, so a cache published during it
                # still differs at the next look (_recheck_saved_caches).
                looked_at = [default_folder]
                state = [
                    self._saved_cache_names(saved_layout_dir, subfolders=True),
                    self._saved_cache_names(default_folder),
                ]
                matches = cache_manifest.find_matching_manifest_folders(
                    saved_layout_dir, compatibility
                )
                for item in matches:
                    if os.path.normcase(item["folder"]) != os.path.normcase(default_folder):
                        looked_at.append(item["folder"])
                        state.append(self._saved_cache_names(item["folder"]))
                cache_items = []
                if len(matches) == 1:
                    # Listed here, so a folder removed since the search shows as an error.
                    active_folder = matches[0]["folder"]
                    cache_files = [
                        entry.name
                        for entry in os.scandir(active_folder)
                        if entry.is_file() and entry.name.lower().endswith(".h5")
                    ]
                    cache_files.sort(
                        key=lambda name: os.path.getmtime(os.path.join(active_folder, name)),
                        reverse=True,
                    )
                    cache_items = [
                        (
                            filename,
                            cache_manifest.relative_cache_path(
                                saved_layout_dir, active_folder, filename
                            ),
                        )
                        for filename in cache_files
                    ]
            except Exception as error:
                self._set_cache_unavailable(
                    translate("Config", "Cache compatibility error: {error}").format(error=error),
                    "danger",
                )
                return

            listing = (tuple(item["folder"] for item in matches), default_folder, tuple(cache_items))
            shown, self._cache_listing_shown = self._cache_listing_shown, listing
            self._saved_cache_look = (saved_layout_dir, tuple(looked_at), tuple(state))
            if refresh:
                if listing == shown:
                    return
                shown_items = shown[2] if shown else ()
                appeared = [data for name, data in cache_items if (name, data) not in shown_items]
                self._carried_cache_choice = (
                    {"new": False, "data": appeared[0], "name": ""}
                    if appeared else self._current_cache_choice()
                )

            self.cb_cache_file.blockSignals(True)
            self.cb_cache_file.clear()
            self.line_new_cache.clear()

            if len(matches) > 1:
                folders = tuple(item["folder"] for item in matches)
                self.current_cache_folder = None
                self._cache_launch_allowed = False
                self._update_save_run_button()
                self.cb_cache_file.setEnabled(False)
                self._toggle_new_cache_input()
                self.btn_open_target_folder.setEnabled(False)
                self._show_cache_status(translate(
                    "Config", "Error: %n compatible cache folder(s) found", None, len(folders)
                ), "danger", bold=True)
                if folders != self._last_duplicate_signature:
                    print("ERROR: Multiple compatible cache folders were found:")
                    for folder in folders:
                        print(f"  - {folder}")
                self._last_duplicate_signature = folders
                self.cb_cache_file.blockSignals(False)
                self._carried_cache_choice = None
                return

            self._last_duplicate_signature = None
            if matches:
                active_folder = matches[0]["folder"]
                self.current_cache_folder = active_folder
                for filename, relative_path in cache_items:
                    self.cb_cache_file.addItem(filename, relative_path)
                self._show_cache_status(translate(
                    "Config", "Compatible Folder: {folder}"
                ).format(folder=os.path.basename(active_folder)), "success", bold=True)
                self.btn_open_target_folder.setEnabled(True)
            else:
                active_folder = default_folder
                default_name = os.path.basename(default_folder)
                self.current_cache_folder = active_folder
                self._show_cache_status(translate(
                    "Config", "Target Folder: {folder} [Needs Computing]"
                ).format(folder=default_name), "danger")
                self.btn_open_target_folder.setEnabled(False)

            self.line_new_cache.setPlaceholderText(
                self._default_new_cache_name(active_folder)
            )
            self.cb_cache_file.addItem(translate("Config", "(New Layout Cache)"), None)
            self.cb_cache_file.setEnabled(True)
            self.cb_cache_file.setCurrentIndex(0)
            self.cb_cache_file.blockSignals(False)
            self._cache_launch_allowed = True
            self._update_save_run_button()
            self._toggle_new_cache_input()
            self._apply_carried_cache_choice()

        def _cache_hash_completed(self, request_id, cache_keys, records, error):
            worker = self._cache_hash_workers.pop(request_id, None)
            if worker is not None:
                worker.deleteLater()
            if request_id != self._cache_hash_request_id:
                return
            self._cache_hash_pending_keys = None
            if error:
                self._set_cache_unavailable(
                    translate("Config", "Cache hashing failed: {error}").format(error=error), "danger"
                )
                return
            sequence_key, network_key = cache_keys
            self._cache_hash_cache[sequence_key] = records["sequence"]
            self._cache_hash_cache[network_key] = records["network"]
            self._apply_cache_discovery(records)

        def _request_cache_discovery(self, refresh=False):
            """Find the cache for the selected inputs, hashing them first unless their hashes are cached.

            refresh=True looks again for the same inputs (_apply_cache_discovery);
            inputs that must be hashed again count as new ones.
            """
            sequence_path, network_path = self._cache_paths_from_inputs()
            if not sequence_path or not network_path:
                self._cancel_cache_hashing()
                self._set_cache_unavailable(translate("Config", "Target Cache: Missing FASTA or HDF5"))
                return
            if not os.path.isfile(sequence_path) or not os.path.isfile(network_path):
                self._cancel_cache_hashing()
                self._set_cache_unavailable(
                    translate("Config", "Target Cache: Selected input file is missing"), "danger"
                )
                return

            try:
                sequence_key = cache_manifest.file_cache_key(sequence_path)
                network_key = cache_manifest.file_cache_key(network_path)
            except OSError as error:
                self._cancel_cache_hashing()
                self._set_cache_unavailable(
                    translate("Config", "Cache input error: {error}").format(error=error), "danger"
                )
                return

            cached_records = {
                "sequence": self._cache_hash_cache.get(sequence_key),
                "network": self._cache_hash_cache.get(network_key),
            }
            if all(cached_records.values()):
                # A worker still hashing earlier inputs would otherwise report
                # as current and replace what is shown for these.
                self._cancel_cache_hashing()
                try:
                    cached_records["network_type"] = (
                        cache_manifest.validate_network_schema(network_path).network_type
                    )
                    self._apply_cache_discovery(cached_records, refresh=refresh)
                except Exception as error:
                    self._set_cache_unavailable(
                        translate("Config", "Cache compatibility error: {error}").format(error=error),
                        "danger",
                    )
                return

            pending_keys = (sequence_key, network_key)
            if self._cache_hash_pending_keys == pending_keys:
                return

            self._cancel_cache_hashing()
            request_id = self._cache_hash_request_id
            self._cache_hash_pending_keys = pending_keys
            self._set_cache_unavailable(translate("Config", "Checking input files…"))
            worker = CacheHashWorker(
                request_id,
                sequence_path,
                network_path,
                pending_keys,
                cached_records=cached_records,
            )
            self._cache_hash_workers[request_id] = worker
            worker.completed.connect(self._cache_hash_completed)
            worker.start()

        def _cancel_cache_hashing(self):
            """Stop any input hashing still running and ignore its result."""
            self._cache_hash_request_id += 1
            self._cache_hash_pending_keys = None
            for worker in self._cache_hash_workers.values():
                worker.requestInterruption()

        def _current_cache_choice(self):
            """The cache chosen, as _apply_carried_cache_choice takes it back."""
            return {
                "new": self._new_cache_selected(),
                "data": self.cb_cache_file.currentData(),
                "name": self.line_new_cache.text(),
            }

        def _saved_cache_names(self, folder, subfolders=False):
            """The names in folder that the cache search depends on, None if it cannot be read.

            Its subfolders for the layout folder (subfolders=True), else its
            manifest and .h5 files; a file the Layout Generator is still
            writing has a .partial name, so it shows only once published.
            """
            try:
                with os.scandir(folder) as entries:
                    if subfolders:
                        names = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
                    else:
                        names = [
                            entry.name for entry in entries
                            if entry.name.lower() == cache_manifest.MANIFEST_FILENAME
                            or entry.name.lower().endswith(".h5")
                        ]
            except OSError:
                return None
            return tuple(sorted(names))

        def _recheck_saved_caches(self):
            """Search for layout caches again if a folder the shown search looked at has changed.

            A cache published since, by a Layout Generator this window launched
            or by anything else, then shows without the inputs being chosen
            again. The inputs' hashes are cached, so nothing is hashed again
            unless an input file changed too.
            """
            if self._saved_cache_look is None:
                return
            saved_layout_dir, folders, state = self._saved_cache_look
            now = (self._saved_cache_names(saved_layout_dir, subfolders=True),) + tuple(
                self._saved_cache_names(folder) for folder in folders
            )
            if now != state:
                self._request_cache_discovery(refresh=True)

        def _recheck_saved_caches_while_shown(self):
            """_recheck_saved_caches on the window's timer, while it shows and its cache list is closed."""
            if not self.isVisible() or self.isMinimized() or self.cb_cache_file.view().isVisible():
                return
            started = time.perf_counter()
            self._recheck_saved_caches()
            # A slow layout folder (on a network share, say) is looked at less
            # often, so looking takes at most a twentieth of the window's time.
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._cache_recheck_timer.setInterval(max(CONFIG_CACHE_RECHECK_MS, round(elapsed_ms * 20)))

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
                self._label_column.append(lbl)
                layout.addRow(lbl, widget)
                self.labels[key] = lbl
                self.inputs[key] = widget

            # Helper for adding a combobox + dynamic folder button
            def add_row_with_dynamic_btn(key, label_text, combo, dir_key, default_dir):
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                btn = icon_button("folder-open", translate("Config", "Open Folder"))
                btn.setFixedWidth(30)
                
                def open_folder(checked):
                    import os
                    # Read from the directory input if it exists, otherwise use the default global
                    path = self.inputs[dir_key].text() if dir_key in self.inputs else globals().get(dir_key, default_dir)
                    abs_path = str(self._absolute_project_path(path))
                    os.makedirs(abs_path, exist_ok=True)
                    Desktop_Open.open_path(abs_path)
                    
                btn.clicked.connect(open_folder)
                
                h_lay.addWidget(combo)
                h_lay.addWidget(btn)
                
                lbl = QLabel(label_text)
                self._label_column.append(lbl)
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
            add_row_with_dynamic_btn(
                "NODE_FASTA_FILE", translate("Config", "Sequence Set / Subset (.fasta):"),
                self.cb_fasta, "FASTA_DIR", seq_dir,
            )
            
            # --- HDF5 Input ---
            self.cb_hdf5 = DynamicComboBox(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
            hdf5_dir = globals().get("HDF5_DIR", os.path.join("Input_Files", "Networks_EValues"))
            hdf5_files = [f for f in os.listdir(hdf5_dir) if f.endswith('.h5')] if os.path.exists(hdf5_dir) else []
            self.cb_hdf5.addItems([""] + hdf5_files)
            hdf5_val = globals().get("INPUT_HDF5") or ""
            if os.path.basename(hdf5_val) in hdf5_files:
                self.cb_hdf5.setCurrentText(os.path.basename(hdf5_val))
            add_row_with_dynamic_btn(
                "INPUT_HDF5", translate("Config", "Network Edges Input (.h5):"),
                self.cb_hdf5, "HDF5_DIR", hdf5_dir,
            )
            
            # --- MSA Input ---
            self.cb_msa = DynamicComboBox(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            msa_dir_path = globals().get("MSA_DIR", os.path.join("Input_Files", "Multiple_Alignments"))
            msa_files = [f for f in os.listdir(msa_dir_path) if f.endswith('.fasta') or f.endswith('.h5')] if os.path.exists(msa_dir_path) else []
            self.cb_msa.addItems([""] + msa_files)
            msa_val = globals().get("MSA_FILE") or ""
            if os.path.basename(msa_val) in msa_files:
                self.cb_msa.setCurrentText(os.path.basename(msa_val))
            add_row_with_dynamic_btn(
                "MSA_FILE", translate("Config", "MSA Input (.fasta / _sparse.h5):"),
                self.cb_msa, "MSA_DIR", msa_dir_path,
            )

            # --- Embedding Input (optional, like the MSA) ---
            self.cb_embed = DynamicComboBox(lambda: self.refresh_combo(self.cb_embed, "EMBED_DIR", ['.h5']))
            embed_dir_path = globals().get("EMBED_DIR", DIRECTORY_PROFILE_DEFAULTS["EMBED_DIR"])
            embed_files = [f for f in os.listdir(embed_dir_path) if f.endswith('.h5')] if os.path.exists(embed_dir_path) else []
            self.cb_embed.addItems([""] + embed_files)
            embed_val = globals().get("EMBED_FILE") or ""
            if os.path.basename(embed_val) in embed_files:
                self.cb_embed.setCurrentText(os.path.basename(embed_val))
            add_row_with_dynamic_btn(
                "EMBED_FILE", translate("Config", "Embedding Input (.h5):"),
                self.cb_embed, "EMBED_DIR", embed_dir_path,
            )

            # --- Rest of Inputs ---
            # Use NoScrollComboBox here to prevent accidental scroll wheel changes
            self.cb_score_mode = NoScrollComboBox()
            # Its options name their algorithms. At its narrowest it keeps
            # about four characters, so the row is no wider than when they
            # read just "global" and "local". Set before anything measures
            # it: a combo box keeps the minimum width it first works out.
            self.cb_score_mode.setMinimumContentsLength(4)
            add_combo_options(self.cb_score_mode, ["global", "local"], [
                translate("Config", "global (Needleman-Wunsch)", "alignment score mode"),
                translate("Config", "local (Smith-Waterman)", "alignment score mode"),
            ])
            select_combo_value(self.cb_score_mode, str(globals().get("ALIGNMENT_SCORE", "global")))
            
            self.cb_norm_mode = NoScrollComboBox()
            # Its modes have long names; at its narrowest it keeps about six
            # characters. Set before anything measures it: a combo box keeps
            # the minimum width it first works out.
            self.cb_norm_mode.setMinimumContentsLength(6)
            score_label = QLabel(translate("Config", "Alignment Score Mode:"))
            norm_label = QLabel(translate("Config", "Normalization Mode:"))
            mode_container = self._make_field_group(
                [(score_label, self.cb_score_mode), (norm_label, self.cb_norm_mode)],
                name="alignmentModeRow", equal_fields=True,
            )
            layout.addRow(mode_container)
            self.labels["ALIGNMENT_SCORE"] = score_label
            self.inputs["ALIGNMENT_SCORE"] = self.cb_score_mode
            self.labels["NORM_MODE"] = norm_label
            self.inputs["NORM_MODE"] = self.cb_norm_mode
            
            self.cb_score_mode.currentTextChanged.connect(self.update_norm_mode_options)
            self.update_norm_mode_options()
            
            initial_norm = str(globals().get("NORM_MODE", "alignment_length"))
            if combo_value(self.cb_score_mode) == "local" and initial_norm == "alignment_length":
                initial_norm = "longer_sequence"
            select_combo_value(self.cb_norm_mode, initial_norm)
            
            ref_val = globals().get("ALIGNMENT_REFERENCE", "")
            self.line_ref = QLineEdit("" if ref_val in [None, "None"] else str(ref_val))
            # The first column narrows before the others (aligned_column_widths).
            self.line_ref.setProperty("minimumCharacters", 8)

            ref_container = QWidget()
            ref_container.setObjectName("alignment_reference_wrapper")

            self.lbl_alignment_offset = QLabel(translate("Config", "Alignment Offset:"))
            self.spin_alignment_offset = NoScrollSpinBox()
            self.spin_alignment_offset.setRange(-1000000, 1000000)
            try:
                offset_value = int(globals().get("ALIGNMENT_OFFSET", 0) or 0)
            except (TypeError, ValueError):
                offset_value = 0
            self.spin_alignment_offset.setValue(offset_value)
            self.spin_alignment_offset.setAccelerated(True)
            # Room for a sign and a few digits, not the whole range. A minimum
            # rather than a fixed width, so it spans the row once the fields stack.
            self.spin_alignment_offset.minimum_text = "-000"

            self.spin_min_occ = NoScrollDoubleSpinBox()
            self.spin_min_occ.setDecimals(2)
            self.spin_min_occ.setRange(0.0, 100.0)
            self.spin_min_occ.setSingleStep(1.0)
            self.spin_min_occ.setValue(float(globals().get("FILTER_MIN_OCCUPANCY") or 10.0))
            self.spin_min_occ.minimum_text = "10.00"

            lbl_min_occ = QLabel(translate("Config", "Min Occupancy %:"))

            ref_label = QLabel(translate("Config", "Alignment Reference ID:"))
            self._make_field_group(
                [(ref_label, self.line_ref), (lbl_min_occ, self.spin_min_occ),
                 (self.lbl_alignment_offset, self.spin_alignment_offset)],
                parent=ref_container, name="alignmentReferenceRow",
                trailing=True, control_stretches=(3, 2, 0),
            )
            layout.addRow(ref_container)
            self.labels["ALIGNMENT_REFERENCE"] = ref_label
            self.inputs["ALIGNMENT_REFERENCE"] = self.line_ref
            self.labels["ALIGNMENT_OFFSET"] = self.lbl_alignment_offset
            self.inputs["ALIGNMENT_OFFSET"] = self.spin_alignment_offset
            
            # --- Layout Method, UMAP and PaCMAP Controls ---
            # The Layout Method dropdown ends the filter row. SSN is the physics layout; UMAP and
            # PaCMAP are the manifold ones. It stands for UMAP_MODE and LAYOUT_METHOD together
            # (see layout_choice), and remembers the manifold method for while SSN is chosen.

            umap_mode_val = globals().get("UMAP_MODE", False)
            if isinstance(umap_mode_val, str):
                umap_mode_val = umap_mode_val.lower() in ['true', '1', 't', 'y', 'yes']
            saved_method = str(globals().get("LAYOUT_METHOD") or "umap")
            self._manifold_method = saved_method if saved_method in MANIFOLD_METHODS else "umap"
            
            lbl_k = QLabel(translate("Config", "UMAP Nearest Neighbors:"))
            lbl_md = QLabel(translate("Config", "UMAP Min Distance:"))
            lbl_pk = QLabel(translate("Config", "PaCMAP Nearest Neighbors:"))

            self.spin_umap_k = NoScrollSpinBox()
            self.spin_umap_k.setRange(2, 500)
            self.spin_umap_k.setValue(int(globals().get("UMAP_NEIGHBORS") or 15))
            self.spin_umap_k.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self.spin_pacmap_k = NoScrollSpinBox()
            self.spin_pacmap_k.setRange(*PROFILE_RANGES["PACMAP_NEIGHBORS"])
            self.spin_pacmap_k.setValue(int(globals().get("PACMAP_NEIGHBORS") or INPUT_PROFILE_DEFAULTS["PACMAP_NEIGHBORS"]))
            self.spin_pacmap_k.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
                self.spin_pacmap_k,
            ):
                spinbox.setFixedHeight(input_spinbox_height)

            self.cb_layout_method = NoScrollComboBox()
            # Its longest choice is PaCMAP, and it keeps room for all of it however narrow the row
            # gets. Set before anything measures it: a combo box keeps the minimum width it first
            # works out.
            self.cb_layout_method.setMinimumContentsLength(LAYOUT_METHOD_MINIMUM_CHARACTERS)
            add_combo_options(self.cb_layout_method, list(LAYOUT_CHOICES), [
                layout_method_names()[choice] for choice in LAYOUT_CHOICES
            ])
            select_combo_value(self.cb_layout_method, layout_choice(umap_mode_val, self._manifold_method))

            self.cb_layout_method.currentIndexChanged.connect(self._layout_choice_changed)
            self.spin_umap_k.valueChanged.connect(self.update_live_validators)
            self.spin_pacmap_k.valueChanged.connect(self.update_live_validators)
            self.labels["LAYOUT_METHOD"] = QLabel(translate("Config", "Layout Method:"))

            # One widget, two settings: the dropdown is "LAYOUT_METHOD" here, and the
            # settings readers and writers (collect_data, the profile and carry-over
            # methods) give UMAP_MODE and LAYOUT_METHOD their values from it.
            self.inputs["LAYOUT_METHOD"] = self.cb_layout_method
            self.inputs["UMAP_NEIGHBORS"] = self.spin_umap_k
            self.inputs["UMAP_MIN_DIST"] = self.spin_umap_md
            self.inputs["PACMAP_NEIGHBORS"] = self.spin_pacmap_k
            self.labels["UMAP_NEIGHBORS"] = lbl_k
            self.labels["UMAP_MIN_DIST"] = lbl_md
            self.labels["PACMAP_NEIGHBORS"] = lbl_pk

            self.spin_thresh = OptionalNoScrollDoubleSpinBox(-1000000000.0, 0.0)
            self.spin_thresh.setDecimals(5)
            self.spin_thresh.setRange(-1000000000.0, 1000000000.0)
            self.spin_thresh.setSingleStep(0.1)
            self.spin_thresh.setSpecialValueText(" ")
            self.spin_thresh.setOptionalValue(globals().get("SIMILARITY_THRESHOLD"))
            # The first column narrows before the others (aligned_column_widths);
            # at its narrowest this shows a threshold to three decimals.
            self.spin_thresh.minimum_text = "0.000"

            self.spin_top = OptionalNoScrollDoubleSpinBox(-0.01, 1.0)
            self.spin_top.setDecimals(2)
            self.spin_top.setRange(-0.01, 100.0)
            self.spin_top.setSingleStep(0.1)
            self.spin_top.setSpecialValueText(" ")
            self.spin_top.setOptionalValue(globals().get("TOP_EDGE_PERCENT"))
            self.spin_top.minimum_text = "10.00"


            for spinbox in (
                self.spin_thresh,
                self.spin_top,
                self.spin_min_occ,
            ):
                spinbox.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
                )
                spinbox.setFixedHeight(input_spinbox_height)

            top_edge_control = QWidget()
            top_edge_control.setObjectName("topEdgeControl")
            top_edge_layout = QHBoxLayout(top_edge_control)
            top_edge_layout.setContentsMargins(0, 0, 0, 0)
            top_edge_layout.setSpacing(6)
            self.btn_clear_top_edge = QPushButton(translate("Config", "Clear"))
            self.btn_clear_top_edge.setObjectName("clearTopEdgeButton")
            self.btn_clear_top_edge.setToolTip(translate(
                "Config", "Unset Top Edge % to use Similarity Threshold when Layout Method is SSN."
            ))
            self.btn_clear_top_edge.clicked.connect(
                lambda: self.spin_top.setOptionalValue(None)
            )
            top_edge_layout.addWidget(self.spin_top, 1)
            top_edge_layout.addWidget(self.btn_clear_top_edge)

            filter_container = QWidget()
            filter_container.setObjectName("wrapper")

            lbl_thresh = QLabel(translate("Config", "Similarity Threshold:"))
            lbl_top = QLabel(translate("Config", "Top Edge %:"))

            # Keep separate widgets and settings for each mode while sharing two slots.
            self._input_mode_stacks = []
            def mode_stack(ssn_widget, *manifold_widgets):
                stack = QStackedWidget()
                stack.addWidget(ssn_widget)
                for widget in manifold_widgets:
                    stack.addWidget(widget)
                self._input_mode_stacks.append(stack)
                return stack

            # The first pair also has a page for PaCMAP's neighbour count; PaCMAP has no
            # minimum distance, so the second pair keeps UMAP's page, greyed out.
            threshold_label = mode_stack(lbl_thresh, lbl_k, lbl_pk)
            threshold_control = mode_stack(self.spin_thresh, self.spin_umap_k, self.spin_pacmap_k)
            top_label = mode_stack(lbl_top, lbl_md)
            top_control = mode_stack(top_edge_control, self.spin_umap_md)
            self._make_field_group(
                [(threshold_label, threshold_control), (top_label, top_control),
                 (self.labels["LAYOUT_METHOD"], self.cb_layout_method)],
                parent=filter_container, name="filterRow",
                trailing=True, control_stretches=(1, 1, 0),
            )
            layout.addRow(filter_container)

            # Coordinate all three rows.Normalization spans columns two and
            # three; the two other rows retain a compact, right-aligned last field.
            aligned_rows = [mode_container.layout(), ref_container.layout(),
                            filter_container.layout()]
            # The first column's labels join the window's label column
            # (_align_label_column); the second column's labels match here.
            label_width = max(row._label_width(row.pairs[1][0]) for row in aligned_rows)
            for row in aligned_rows:
                row.pairs[1][0].setFixedWidth(label_width)
            # The last column holds Alignment Offset and Layout Method. Their labels share one
            # width and their fields share one width, so the labels start at one x and the
            # fields share a left edge: each field is as wide as the wider minimum, which is
            # the dropdown's, the room PaCMAP needs.
            last_label_width = max(row._label_width(row.pairs[2][0]) for row in aligned_rows[1:])
            last_control_width = max(row._minimum(row.pairs[2][1]).width() for row in aligned_rows[1:])
            for row in aligned_rows[1:]:
                row.pairs[2][0].setFixedWidth(last_label_width)

            def aligned_column_widths(width, *, spanning=False):
                gap = CONFIG_FIELD_HORIZONTAL_SPACING
                first_min = max(row._column_minima()[0] for row in aligned_rows)
                middle_min = max(row._column_minima()[1] for row in aligned_rows[1:])
                last = last_label_width + gap + last_control_width
                # The right part holds occupancy and the fixed final column,
                # or the normalization mode spanning both.
                right_min = max(middle_min + gap + last,
                                aligned_rows[0]._column_minima()[1])
                if width is None:
                    return first_min + gap + right_min
                # Give the reference half the usable width while both halves
                # fit. Narrower, the first column gives way, down to its own
                # minimum, and the right part keeps its minimum.
                first = max(first_min, min((width - gap) // 2, width - gap - right_min))
                remainder = width - gap - first
                return ([first, remainder] if spanning else
                        [first, remainder - gap - last, last])

            for index, row in enumerate(aligned_rows):
                row.column_width_provider = (
                    lambda width, spanning=index == 0:
                    aligned_column_widths(width, spanning=spanning)
                )
                row.invalidate()

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
            
            # The statistics button spans the field column under the edge filters.
            btn_container = QWidget()
            btn_lay = QHBoxLayout(btn_container)
            btn_lay.setContentsMargins(0, 0, 0, 0)

            self.btn_stats = QPushButton(translate("Config", "Compute Network Statistics"))
            set_role(self.btn_stats, "muted")
            self.btn_stats.clicked.connect(self.run_statistics)

            btn_lay.addWidget(self.btn_stats, 1)
            layout.addRow("", btn_container)

            self.cache_file_separator = self._add_padded_separator(
                layout, "cache_file_separator"
            )

            # --- Target Cache Tracker & Folder Button ---
            cache_container = QWidget()
            cache_lay = QHBoxLayout(cache_container)
            cache_lay.setContentsMargins(0, 0, 0, 0)
            
            self.lbl_cache_tracker = QLabel(translate("Config", "Target Folder: None"))
            set_role(self.lbl_cache_tracker, "subtle")
            self.lbl_cache_tracker.setWordWrap(False)
            self.lbl_cache_tracker.setMinimumWidth(0)
            self.lbl_cache_tracker.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
            )
            
            self.btn_open_cache = icon_button("folder-open", translate("Config", "Open Target Cache Folder"))
            self.btn_open_cache.setFixedWidth(30)
            
            def open_cache_folder(checked):
                import os

                # Exclusively open the parent directory
                dir_input = self.inputs.get("SAVED_LAYOUT_DIR")
                path = dir_input.text() if dir_input else globals().get("SAVED_LAYOUT_DIR", os.path.join("Cache_Files", "Saved_Layouts"))

                abs_path = str(self._absolute_project_path(path))
                os.makedirs(abs_path, exist_ok=True)
                Desktop_Open.open_path(abs_path)
                
            self.btn_open_cache.clicked.connect(open_cache_folder)
            
            cache_lay.addWidget(self.lbl_cache_tracker, 1)
            cache_lay.addWidget(self.btn_open_cache)
            target_cache_label = QLabel(translate("Config", "Target Cache:"))
            self._label_column.append(target_cache_label)
            layout.addRow(target_cache_label, cache_container)
            self.labels["TARGET_CACHE"] = target_cache_label

            # Cache dropdown, conditional new-cache name, and folder button.
            target_container = QWidget()
            target_container.setObjectName("wrapper")
            target_lay = ResponsiveSelectorLayout(target_container)
            target_lay.setContentsMargins(0, 0, 0, 0)

            self.cb_cache_file = DynamicComboBox(self._refresh_cache_file_combo)
            self.cb_cache_file.setEnabled(False)

            self.line_new_cache = QLineEdit()
            self.line_new_cache.setVisible(False)
            self.line_new_cache.setEnabled(False)
            
            self.btn_open_target_folder = icon_button("folder-open", translate("Config", "Open Specific Target Folder"))
            self.btn_open_target_folder.setFixedWidth(30)
            self.btn_open_target_folder.setEnabled(False)  # Greyed out by default
            
            def open_target_folder(checked):
                import os

                path = getattr(self, 'current_cache_folder', None)
                if path and os.path.exists(path):
                    abs_path = os.path.abspath(path)
                    Desktop_Open.open_path(abs_path)
                    
            self.btn_open_target_folder.clicked.connect(open_target_folder)

            target_lay.addWidget(self.cb_cache_file, 1)
            target_lay.addWidget(self.line_new_cache, 1)
            target_lay.addWidget(self.btn_open_target_folder)
            
            layout.addRow(translate("Config", "Selected Cache File:"), target_container)
            self.labels["TARGET_CACHE_FILE"] = layout.labelForField(target_container)
            self._label_column.append(self.labels["TARGET_CACHE_FILE"])
            self.inputs["TARGET_CACHE_FILE"] = self.cb_cache_file
            self.inputs["NEW_CACHE_NAME"] = self.line_new_cache
            
            # Hook up the toggle switch
            self.cb_cache_file.currentIndexChanged.connect(self._toggle_new_cache_input)
            self.cb_cache_file.activated.connect(self._cache_file_activated)

            self._add_scroll_tab(tab, translate("Config", "Inputs && Outputs"))
            
        def uses_manifold_layout(self):
            """True while the Layout Method dropdown says UMAP or PaCMAP, false for SSN (UMAP_MODE)."""
            return combo_value(self.cb_layout_method) in MANIFOLD_METHODS

        def _layout_settings(self):
            """The two settings the Layout Method dropdown stands for, as saved.

            UMAP_MODE is whether a manifold method is chosen. LAYOUT_METHOD is that
            method, or, while SSN is chosen, the one last shown or loaded, so SSN
            leaves it as saved.
            """
            choice = combo_value(self.cb_layout_method)
            return {
                "UMAP_MODE": choice in MANIFOLD_METHODS,
                "LAYOUT_METHOD": choice if choice in MANIFOLD_METHODS else self._manifold_method,
            }

        def _set_layout_setting(self, key, value):
            """Show a saved UMAP_MODE or LAYOUT_METHOD in the dropdown, whichever order they arrive in."""
            if key == "LAYOUT_METHOD":
                self._manifold_method = value if value in MANIFOLD_METHODS else "umap"
                if self.uses_manifold_layout():
                    select_combo_value(self.cb_layout_method, self._manifold_method)
            else:
                select_combo_value(self.cb_layout_method, layout_choice(value, self._manifold_method))

        def _layout_choice_changed(self, _index):
            """A choice in the dropdown, by the user or a load: UMAP and PaCMAP are remembered for SSN."""
            choice = combo_value(self.cb_layout_method)
            if choice in MANIFOLD_METHODS:
                self._manifold_method = choice
            self.update_live_validators()

        def update_norm_mode_options(self):
            if not hasattr(self, 'cb_score_mode') or not hasattr(self, 'cb_norm_mode'):
                return
            current_norm = combo_value(self.cb_norm_mode)
            self.cb_norm_mode.blockSignals(True)
            self.cb_norm_mode.clear()
            
            labels = normalization_names()
            is_local = combo_value(self.cb_score_mode) == "local"
            if is_local:
                modes = ["shorter_sequence", "longer_sequence", "average_sequence"]
                if current_norm == "alignment_length":
                    current_norm = "longer_sequence"
            else:
                modes = ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"]
            add_combo_options(self.cb_norm_mode, modes, [labels[mode] for mode in modes])
                
            select_combo_value(self.cb_norm_mode, current_norm)
            self.cb_norm_mode.blockSignals(False)

        def _update_reference_controls(self, text=None):
            reference_text = self.line_ref.text() if text is None else text
            has_reference = bool(reference_text.strip())
            if hasattr(self, 'spin_alignment_offset'):
                self.spin_alignment_offset.setEnabled(has_reference)
            if hasattr(self, 'lbl_alignment_offset'):
                self.lbl_alignment_offset.setEnabled(has_reference)

        def update_live_validators(self):
            has_fasta = bool(self.cb_fasta.currentText().strip())
            has_hdf5 = bool(self.cb_hdf5.currentText().strip())
            self._update_reference_controls()
            self._cancel_statistics_if_inputs_changed()
            self._update_statistics_button()

            # "is_umap" means a manifold layout (UMAP or PaCMAP) instead of SSN's physics.
            is_umap = hasattr(self, 'cb_layout_method') and self.uses_manifold_layout()
            is_pacmap = is_umap and combo_value(self.cb_layout_method) == "pacmap"
            for stack in getattr(self, "_input_mode_stacks", ()):
                # SSN, then UMAP, then PaCMAP where the pair has a page for it.
                stack.setCurrentIndex(min(stack.count() - 1, int(is_umap) + int(is_pacmap)))
            if hasattr(self, "spin_umap_k"):
                enabled = {
                    "UMAP_NEIGHBORS": is_umap and not is_pacmap,
                    "UMAP_MIN_DIST": is_umap and not is_pacmap,
                    "PACMAP_NEIGHBORS": is_pacmap,
                }
                for key, state in enabled.items():
                    self.inputs[key].setEnabled(state)
                    self.labels[key].setEnabled(state)
            physics_selector = self.profile_selectors.get("simulation_physics")
            physics_profile_editable = (
                physics_selector is None
                or combo_value(physics_selector) != "(default)"
            )
            if hasattr(self, 'cb_layout_device'):
                self.cb_layout_device.setEnabled(
                    physics_profile_editable
                    and not is_umap
                )
            if "LAYOUT_DEVICE_SELECTION" in self.labels:
                self.labels["LAYOUT_DEVICE_SELECTION"].setEnabled(
                    physics_profile_editable
                    and not is_umap
                )
            
            if hasattr(self, 'spin_thresh') and hasattr(self, 'spin_top'):
                has_top_edge = self.spin_top.optionalValue() is not None
                top_edge_enabled = not is_umap
                thresh_enabled = not is_umap and not has_top_edge

                self.spin_top.setEnabled(top_edge_enabled)
                self.spin_thresh.setEnabled(thresh_enabled)

                if hasattr(self, 'btn_clear_top_edge'):
                    self.btn_clear_top_edge.setEnabled(
                        top_edge_enabled and has_top_edge
                    )

                if "SIMILARITY_THRESHOLD" in self.labels:
                    self.labels["SIMILARITY_THRESHOLD"].setEnabled(thresh_enabled)
                if "TOP_EDGE_PERCENT" in self.labels:
                    self.labels["TOP_EDGE_PERCENT"].setEnabled(top_edge_enabled)

            if hasattr(self, 'tabs') and self.tabs.count() > 2:
                if is_umap and self.tabs.currentIndex() == 2:
                    self.tabs.setCurrentIndex(0)
                self.tabs.setTabVisible(2, not is_umap)
                self.tabs.setTabEnabled(2, True)
            
            if hasattr(self, 'btn_check'):
                self.btn_check.setEnabled(has_fasta and has_hdf5)
            
            # Keep cache status on one compact line and refresh manifest discovery.
            self.lbl_cache_tracker.setWordWrap(False) 
            self.lbl_cache_tracker.setMinimumHeight(0)
            self.lbl_cache_tracker.setMaximumHeight(30)
            self._request_cache_discovery()

        def _statistics_inputs(self):
            """What the statistics depend on: the two files, how they stand on disk, and the modes."""
            fasta_path = os.path.join(
                self._resolved_directory_input("FASTA_DIR"),
                self.cb_fasta.currentText(),
            )
            hdf5_path = os.path.join(
                self._resolved_directory_input("HDF5_DIR"),
                self.cb_hdf5.currentText(),
            )
            states = []
            for path in (fasta_path, hdf5_path):
                try:
                    states.append(cache_manifest.file_cache_key(path))
                except OSError:
                    states.append(None)
            return (
                fasta_path, hdf5_path,
                combo_value(self.cb_score_mode), combo_value(self.cb_norm_mode),
                tuple(states),
            )

        def _update_statistics_button(self):
            """The button works while both inputs are chosen and no run is in progress."""
            self.btn_stats.setEnabled(
                bool(self.cb_fasta.currentText().strip())
                and bool(self.cb_hdf5.currentText().strip())
                and self._statistics_request is None
            )

        def _show_statistics_progress(self, fraction):
            """Write the progress line, unless something else has taken the panel since the last one."""
            percent = min(100, int(fraction * 100))
            if percent == self._statistics_percent:
                return
            self._statistics_percent = percent
            if self._statistics_status_html not in (None, self.tip_panel.text()):
                return
            self.tip_panel.setStatus("info", translate(
                "Config", "Computing network statistics... {percent} %"
            ).format(percent=percent))
            self._statistics_status_html = self.tip_panel.text()

        def _show_statistics_stopped(self):
            """Replace the progress line of a run that was stopped, if it is still on show."""
            if self._statistics_status_html == self.tip_panel.text():
                self.tip_panel.setStatus("info", translate(
                    "Config", "Network statistics stopped: the inputs changed."
                ))
            self._statistics_status_html = None

        def _cancel_statistics(self):
            """Stop the statistics run in progress and ignore whatever it still reports."""
            self._statistics_request_id += 1
            self._statistics_request = None
            for worker in self._statistics_workers.values():
                worker.requestInterruption()
            if hasattr(self, "btn_stats"):
                self._update_statistics_button()

        def _cancel_statistics_if_inputs_changed(self, *_ignored):
            if self._statistics_request is None:
                return
            if self._statistics_inputs() != self._statistics_request:
                self._cancel_statistics()
                self._show_statistics_stopped()

        def _statistics_progress(self, request_id, fraction):
            if request_id == self._statistics_request_id and self._statistics_request is not None:
                self._show_statistics_progress(fraction)

        def _statistics_completed(self, request_id, report, error):
            worker = self._statistics_workers.pop(request_id, None)
            if worker is not None:
                worker.deleteLater()
            if request_id != self._statistics_request_id:
                return
            started_on, self._statistics_request = self._statistics_request, None
            self._update_statistics_button()
            if started_on != self._statistics_inputs():
                # The files or the modes changed while it ran: its figures describe other inputs.
                self._show_statistics_stopped()
                return
            self._statistics_status_html = None
            if error is not None:
                self.tip_panel.setStatus("danger", translate(
                    "Config", "Error during network statistics calculation: {error}"
                ).format(error=display_text(error)))
            elif report is None:
                self.tip_panel.setStatus("warning", translate(
                    "Config", "Warning: No valid edges were found in the selected FASTA subset."
                ))
            else:
                self.show_statistics(report)
                self.tip_panel.setStatus("success", translate("Config", "Network statistics computed successfully."))

        def run_statistics(self):
            """Compute the network statistics on a worker thread and show them when it is done.

            A run in progress is stopped first. The window stays usable: the tip panel
            shows the progress, the button waits, and a change of the inputs, a second
            run or closing the window stops the run, so a report never lands for inputs
            that are no longer selected.
            """
            self._cancel_statistics()
            request_id = self._statistics_request_id
            inputs = self._statistics_inputs()
            fasta_path, hdf5_path, score_mode, norm_mode, _states = inputs
            self._statistics_request = inputs
            self._statistics_percent = -1
            self._statistics_status_html = None
            self._update_statistics_button()
            self._show_statistics_progress(0.0)
            worker = StatisticsWorker(request_id, hdf5_path, fasta_path, score_mode, norm_mode)
            self._statistics_workers[request_id] = worker
            worker.progress.connect(self._statistics_progress)
            worker.completed.connect(self._statistics_completed)
            worker.start()

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
            visual_grid.setColumnStretch(1, 1)
            visual_grid.setColumnMinimumWidth(2, 16)
            visual_grid.setColumnStretch(4, 1)
            visual_row = 0
            
            # 1. Sliders Setup
            slider_settings = [
                {"key": "NODE_SIZE", "label": translate("Config", "Node Size:"), "type": "int", "min": 1, "max": 20, "default": 10},
                {"key": "EDGE_WIDTH", "label": translate("Config", "Edge Width:"), "type": "float", "min": 0.1, "max": 3.0, "scale": 10.0, "decimals": 1, "default": 1.0},
                {"key": "NODE_BOUNDARY_WIDTH", "label": translate("Config", "Node Boundary Width:"), "type": "float", "min": 0.0, "max": 2.0, "scale": 10.0, "decimals": 1, "default": 0.5},
                {"key": "EDGE_ALPHA", "label": translate("Config", "Edge Opacity:"), "type": "float", "min": 0.0, "max": 1.0, "scale": 100.0, "decimals": 2, "default": 0.1},
                {"key": "TEXT_SIZE", "label": translate("Config", "Text Size:"), "type": "int", "min": 1, "max": 24, "default": 8}
            ]
            
            for s in slider_settings:
                key = s["key"]
                
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

                lbl = QLabel(s["label"])
                visual_grid.addWidget(lbl, visual_row, 0)
                visual_grid.addWidget(ui_element, visual_row, 1, 1, 4)
                visual_row += 1
                self.labels[key] = lbl
                self.inputs[key] = box

            # 2. Colors Setup
            color_labels = {
                "TEXT_COLOR": translate("Config", "Text Color:"),
                "INITIAL_NODE_COLOR": translate("Config", "Initial Node Color:"),
                "HOVER_COLOR": translate("Config", "Highlight Color:"),
                "CONNECTED_NODE_COLOR": translate("Config", "Connected Node Color:"),
                "EDGE_COLOR": translate("Config", "Edge Color:"),
                "NODE_BOUNDARY_COLOR": translate("Config", "Node Boundary Color:"),
            }
            color_keys = list(color_labels)
            self.visual_defaults = VISUAL_PROFILE_DEFAULTS

            color_row_start = visual_row

            for index, key in enumerate(color_keys):
                
                color_container = QWidget()
                color_container.setObjectName("wrapper")
                h_layout = QHBoxLayout(color_container)
                h_layout.setContentsMargins(0, 0, 0, 0)
                
                val = globals().get(key, self.visual_defaults[key])
                
                swatch = QLabel()
                swatch.setFixedSize(20, 20)
                swatch.setStyleSheet(SWATCH_STYLESHEET.format(color=val))
                
                le = QLineEdit("" if val in [None, "None"] else str(val))

                def update_color_swatch(value, color_swatch=swatch):
                    color = QColor(str(value).strip())
                    if color.isValid():
                        color_swatch.setStyleSheet(SWATCH_STYLESHEET.format(color=color.name()))

                le.textChanged.connect(update_color_swatch)
                
                btn = set_role(QPushButton(translate("Config", "Pick")), "compact")
                fit_buttons_to_text(btn)
                
                def pick_color(checked, line_edit=le, button=btn):
                    # The field, and so its swatch, follows the picker at once.
                    self.color_picker = open_color_picker(button, line_edit)
                
                btn.clicked.connect(pick_color)
                
                h_layout.addWidget(swatch)
                h_layout.addWidget(le)
                h_layout.addWidget(btn)
                
                lbl = QLabel(color_labels[key])
                row = color_row_start + index // 2
                column = 0 if index % 2 == 0 else 3
                visual_grid.addWidget(lbl, row, column)
                visual_grid.addWidget(color_container, row, column + 1)
                self.labels[key] = lbl
                self.inputs[key] = le
                self.color_swatches[key] = swatch

            visual_rows = self._responsive_grid_rows(visual_grid, "visual")
            main_layout.addLayout(visual_rows)

            # Low Resource Mode, under a line like the saved-config row's: a
            # switch per category of work the Viewer can be spared, in a
            # section of its own below the grid, on the same spacing.
            self.low_resource_separator = self._add_padded_separator(None, "lowResourceSeparator")
            visual_rows.addWidget(self.low_resource_separator.wrapper)
            visual_rows.addWidget(self._low_resource_section())
            main_layout.addStretch()

            self._add_scroll_tab(tab, translate("Config", "Visual Effects"))

        def _low_resource_section(self):
            """The Low Resource Mode switches, two to a row in the order of
            LOW_RESOURCE_SWITCH_ROWS, stacking to one a line, in that order,
            when the page is too narrow for two.

            The section's columns start where the colour grid's do: the first
            column's labels with Text Color's, the second's with Initial Node
            Color's, at every width, because the section asks the colour row
            for its column widths (ResponsiveFieldLayout.column_widths). It
            stacks together with the colour rows. Each column's labels are as
            wide as its longest, and its switches share one x, a
            LOW_RESOURCE_SWITCH_GAP after them, not in line with the colour
            fields. Stacked, every label is as wide as the longest of all, so
            the switches still line up. Its labels take no part in the
            window's shared label column (_align_label_column).
            """
            section = QWidget()
            section.setObjectName("lowResourceSection")
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(CONFIG_TAB_ROW_SPACING)
            reference = self.labels["INITIAL_NODE_COLOR"].parentWidget().layout()
            rows = [[self._low_resource_switch(key) for key in keys]
                    for keys in LOW_RESOURCE_SWITCH_ROWS]
            column_widths = [
                max(row[column][0].sizeHint().width() for row in rows if len(row) > column)
                for column in range(max(len(row) for row in rows))
            ]
            for row in rows:
                for column, (label, _field) in enumerate(row):
                    label.setFixedWidth(column_widths[column])
            # A column needs its label, a gap and a switch's field.
            field_width = max(ResponsiveFieldLayout._minimum(field).width()
                              for row in rows for _label, field in row)
            needs = [width + LOW_RESOURCE_SWITCH_GAP + field_width for width in column_widths]
            fitted = {}

            def wide_width():
                """The width from which the section is side by side: the colour rows' own
                and, should the section's columns need more than the colour rows give
                them there (long labels), the first width that holds them. Every row,
                the single one too, stacks at it, so all stack together."""
                wide = reference._wide_width()
                if wide not in fitted:
                    width = wide
                    while width - wide < 2000 and any(
                            column < need for column, need in zip(reference.column_widths(width), needs)):
                        width += 1
                    fitted[wide] = width
                return fitted[wide]

            for index, row in enumerate(rows):
                group = QWidget()
                group.setObjectName(f"lowResourceRow{index}")
                ResponsiveFieldLayout(
                    group, row, tuple(1 for _ in row),
                    spacing=LOW_RESOURCE_SWITCH_GAP,
                    column_spacing=reference.column_spacing,
                    wrap_labels=False,
                    column_width_provider=lambda width, count=len(row): (
                        wide_width() if width is None else reference.column_widths(width)[:count]),
                    stacked_label_width=max(column_widths),
                )
                section_layout.addWidget(group)
            return section

        def _low_resource_switch(self, key):
            """A Low Resource Mode switch and its label, the switch in a field
            of its own width, as the Simulation & Physics tab's switches are."""
            label = QLabel(translate("Config", LOW_RESOURCE_SWITCH_LABELS[key]))
            switch = ToggleSwitch()
            switch.setChecked(bool(globals().get(key, VISUAL_PROFILE_DEFAULTS[key])))
            field = QWidget()
            field.setObjectName("wrapper")
            field.setMinimumHeight(switch.height())
            field_layout = QHBoxLayout(field)
            field_layout.setContentsMargins(0, 0, 0, 0)
            field_layout.addWidget(switch)
            field_layout.addStretch()
            field.setMinimumWidth(switch.width())
            self.labels[key] = label
            self.inputs[key] = switch
            return label, field

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
            
            cb_layout_device = NoScrollComboBox()
            for display_name, specification in Hardware_Utils.device_selection_options():
                if specification == Hardware_Utils.AUTO_DEVICE:
                    cb_layout_device.addItem(translate("Config", "Auto Benchmark"), specification)
                else:
                    cb_layout_device.addItem(display_name, specification)
                    mark_name_item(cb_layout_device, cb_layout_device.count() - 1)
            saved_device = Hardware_Utils.normalize_device_selection(
                globals().get("LAYOUT_DEVICE_SELECTION", "auto")
            )
            saved_index = cb_layout_device.findData(saved_device)
            if saved_index < 0:
                cb_layout_device.addItem(
                    translate("Config", "Unavailable saved device [{device}]").format(device=saved_device),
                    saved_device,
                )
                saved_index = cb_layout_device.count() - 1
            cb_layout_device.setCurrentIndex(saved_index)
            lbl_layout_device = QLabel(translate("Config", "Layout Device:"))
            self._label_column.append(lbl_layout_device)
            form_layout.addRow(lbl_layout_device, cb_layout_device)
            self.inputs["LAYOUT_DEVICE_SELECTION"] = cb_layout_device
            self.labels["LAYOUT_DEVICE_SELECTION"] = lbl_layout_device
            self.cb_layout_device = cb_layout_device
            
            # --- Physics sliders ---
            slider_settings = [
                {"key": "SPRING_K", "label": translate("Config", "Spring Constant:"), "type": "float", "min": 1.0, "max": 20.0, "scale": 10.0, "decimals": 1, "default": 5.0, "tick": 10},
                {"key": "COULOMB_K", "label": translate("Config", "Repulsion Constant:"), "type": "float", "min": 1.0, "max": 30.0, "scale": 10.0, "decimals": 1, "default": 10.0, "tick": 10},
                {"key": "COULOMB_CUTOFF", "label": translate("Config", "Max Repulsion Cutoff:"), "type": "float", "min": 1.0, "max": 100.0, "scale": 10.0, "decimals": 1, "default": 30.0, "tick": 100},
                {"key": "DAMPING", "label": translate("Config", "Damping Coefficient:"), "type": "float", "min": 0.1, "max": 2.0, "scale": 100.0, "decimals": 2, "default": 0.9, "tick": 10}
            ]

            physics_slider_controls = {}
            for s in slider_settings:
                key = s["key"]
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

                lbl = QLabel(s["label"])
                physics_slider_controls[key] = (lbl, ui_element)
                self.labels[key] = lbl
                self.inputs[key] = box

            # All paired slider rows share one grid. A grid per row sizes its columns
            # independently, so rows whose right-hand labels differ in width end up a
            # few pixels out of step with each other.
            slider_pair_grid = QGridLayout()
            slider_pair_grid.setHorizontalSpacing(0)
            slider_pair_grid.setVerticalSpacing(12)
            slider_pair_grid.setColumnMinimumWidth(1, field_label_gap)
            slider_pair_grid.setColumnMinimumWidth(3, paired_group_padding)
            slider_pair_grid.setColumnMinimumWidth(5, field_label_gap)
            slider_pair_grid.setColumnStretch(2, 1)
            slider_pair_grid.setColumnStretch(6, 1)

            def add_paired_slider_row(left_key, right_key):
                row = slider_pair_grid.rowCount()
                left_label, left_control = physics_slider_controls[left_key]
                right_label, right_control = physics_slider_controls[right_key]

                slider_pair_grid.addWidget(left_label, row, 0)
                slider_pair_grid.addWidget(left_control, row, 2)
                slider_pair_grid.addWidget(right_label, row, 4)
                slider_pair_grid.addWidget(right_control, row, 6)

            add_paired_slider_row("SPRING_K", "COULOMB_K")
            add_paired_slider_row("COULOMB_CUTOFF", "DAMPING")
            form_layout.addRow(self._responsive_grid_rows(slider_pair_grid, "physicsSliders"))

            # --- 3. Integration and convergence settings (two columns) ---
            convergence_grid = QGridLayout()
            convergence_grid.setHorizontalSpacing(0)
            convergence_grid.setVerticalSpacing(12)
            convergence_grid.setColumnMinimumWidth(1, field_label_gap)
            convergence_grid.setColumnMinimumWidth(3, paired_group_padding)
            convergence_grid.setColumnMinimumWidth(5, field_label_gap)
            convergence_grid.setColumnStretch(2, 1)
            convergence_grid.setColumnStretch(6, 1)

            lbl_dt = QLabel(translate("Config", "Step Size:"))
            le_dt = QLineEdit(str(globals().get("DT", 0.005)))
            self.inputs["DT"] = le_dt
            self.labels["DT"] = lbl_dt

            # Auto lets each simulation stage pick its own fastest stable step,
            # so the Step Size field is greyed out while it is on.
            btn_auto_dt = ToggleSwitch(translate("Config", "Auto ON"), translate("Config", "Auto OFF"))
            self.inputs["AUTO_DT"] = btn_auto_dt
            btn_auto_dt.toggled.connect(
                lambda checked, field=le_dt: field.setEnabled(not checked)
            )
            auto_dt_state = bool(globals().get("AUTO_DT", False))
            btn_auto_dt.setChecked(auto_dt_state)
            le_dt.setEnabled(not auto_dt_state)

            # The button starts the Step Size cell, in line with the fields below
            # it. The field gives up the button's width (see
            # _prepare_responsive_layouts), so the cell needs no more room than
            # a plain field and the row wraps no sooner than the others.
            dt_field = QWidget()
            dt_field.setObjectName("wrapper")
            dt_field_layout = QHBoxLayout(dt_field)
            dt_field_layout.setContentsMargins(0, 0, 0, 0)
            dt_field_layout.setSpacing(CONFIG_FIELD_HORIZONTAL_SPACING)
            dt_field_layout.addWidget(btn_auto_dt)
            dt_field_layout.addWidget(le_dt)
            le_dt.setProperty(
                "sharedCellWidth", btn_auto_dt.width() + dt_field_layout.spacing()
            )

            lbl_steps = QLabel(translate("Config", "Max Steps:"))
            le_steps = QLineEdit(str(globals().get("MAX_STEPS", 10000)))
            self.inputs["MAX_STEPS"] = le_steps
            self.labels["MAX_STEPS"] = lbl_steps

            convergence_grid.addWidget(lbl_dt, 0, 0)
            convergence_grid.addWidget(dt_field, 0, 2)
            convergence_grid.addWidget(lbl_steps, 0, 4)
            convergence_grid.addWidget(le_steps, 0, 6)

            lbl_rmsd = QLabel(translate("Config", "RMSD Threshold:"))
            le_rmsd = QLineEdit(str(globals().get("RMSD_THRESHOLD", 0.005)))
            self.inputs["RMSD_THRESHOLD"] = le_rmsd
            self.labels["RMSD_THRESHOLD"] = lbl_rmsd

            lbl_drop = QLabel(translate("Config", "Min % Drop Threshold:"))
            le_drop = QLineEdit(str(globals().get("PERCENTAGE_DROP_THRESHOLD", 0.1)))
            self.inputs["PERCENTAGE_DROP_THRESHOLD"] = le_drop
            self.labels["PERCENTAGE_DROP_THRESHOLD"] = lbl_drop

            convergence_grid.addWidget(lbl_rmsd, 1, 0)
            convergence_grid.addWidget(le_rmsd, 1, 2)
            convergence_grid.addWidget(lbl_drop, 1, 4)
            convergence_grid.addWidget(le_drop, 1, 6)
            form_layout.addRow(self._responsive_grid_rows(convergence_grid, "convergence"))

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
            
            lbl_window = QLabel(translate("Config", "RMSD Window:"))
            self._label_column.append(lbl_window)
            form_layout.addRow(lbl_window, ui_window)
            self.inputs["RMSD_WINDOW"] = box_window
            self.labels["RMSD_WINDOW"] = lbl_window
            
            # --- 5. Packing controls ---
            lbl_prog = QLabel(translate("Config", "Progressive Annealing:"))
            cb_prog = ToggleSwitch()
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
            
            initial_state = bool(globals().get("ENABLE_PROGRESSIVE_SIMULATION", False))
            cb_prog.setChecked(initial_state)
            
            # Packing Geometry Dropdown
            lbl_geom = QLabel(translate("Config", "Packing Geometry:"))
            cb_geom = NoScrollComboBox()
            add_combo_options(cb_geom, ["Square", "Circle"], [
                translate("Config", "Square", "packing geometry"),
                translate("Config", "Circle", "packing geometry"),
            ])

            cb_geom.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            cb_geom.setMinimumWidth(120)

            initial_geom = globals().get("PACKING_GEOMETRY", "Square")
            if initial_geom not in ["Square", "Circle"]:
                initial_geom = "Square"
            select_combo_value(cb_geom, initial_geom)
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
            val_pgs = globals().get("PACKING_GRID_SIZE", 10.0)
            try: val_pgs = float(val_pgs)
            except: val_pgs = 10.0
            
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
            
            lbl_pgs = QLabel(translate("Config", "Packing Grid Size:"))
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
            form_layout.addRow(self._responsive_grid_rows(
                packing_controls_grid, "packing", trailing=True
            ))

            paired_right_labels = (
                physics_slider_controls["COULOMB_K"][0],
                physics_slider_controls["DAMPING"][0],
                lbl_steps,
                lbl_drop,
            )
            # The size hint, not the text's advance: Linux font hinting can
            # draw a text a pixel past its advance.
            right_label_width = max(label.sizeHint().width() for label in paired_right_labels)
            for paired_label in paired_right_labels:
                paired_label.setFixedWidth(right_label_width)
            
            main_layout.addLayout(form_layout)
            main_layout.addStretch()
 
            self._add_scroll_tab(tab, translate("Config", "Simulation && Physics"))
            
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
                button = icon_button("folder-open", translate("Config", "Open Folder"))
                button.setFixedWidth(30)
                button.setEnabled(bool(line_edit.text().strip()))

                def open_selected_folder(checked=False):
                    raw_path = line_edit.text().strip()
                    if not raw_path:
                        return
                    folder = self._absolute_project_path(raw_path)
                    folder.mkdir(parents=True, exist_ok=True)
                    Desktop_Open.open_path(folder)

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
            saved_config_button = QPushButton(translate("Config", "Browse..."))
            saved_config_layout.addWidget(saved_config_input)
            saved_config_layout.addWidget(saved_config_open_button)
            saved_config_layout.addWidget(saved_config_button)
            layout.addRow(translate("Config", "Saved Config Directory:"), saved_config_container)
            self.inputs["SAVED_CONFIG_DIR"] = saved_config_input
            self.labels["SAVED_CONFIG_DIR"] = layout.labelForField(saved_config_container)
            self._label_column.append(self.labels["SAVED_CONFIG_DIR"])

            def browse_saved_config_directory(checked=False):
                folder = QFileDialog.getExistingDirectory(
                    self,
                    translate("Config", "Select Saved Config Directory"),
                    str(self._saved_config_root()),
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
                # Reusable roots
                "INPUT_FILE_DIR", "CACHE_FILE_DIR", "ANALYSIS_RESULT_DIR",
                "EMBED_DIR",
                # Input files
                "FASTA_DIR", "MSA_DIR", "HDF5_DIR", "METADATA_DIR",
                "HEADER_LIST_DIR",
                # Cache files
                "SAVED_LAYOUT_DIR", "SETTING_EXPORT_DIR",
            ]
            
            for key in keys:
                container = QWidget()
                container.setObjectName("wrapper")
                h_lay = QHBoxLayout(container)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                val = DIRECTORY_PROFILE_DEFAULTS.get(key, globals().get(key, ""))
                le = QLineEdit("" if val in [None, "None"] else str(val))
                open_button = add_open_folder_button(le, key)
                btn = QPushButton(translate("Config", "Browse..."))
                
                def browse_dir(checked, line_edit=le):
                    starting_path = (
                        str(self._absolute_project_path(line_edit.text()))
                        if line_edit.text().strip() else ""
                    )
                    folder = QFileDialog.getExistingDirectory(
                        self, translate("Config", "Select Directory"), starting_path
                    )
                    if folder:
                        import os
                        line_edit.setText(os.path.normpath(folder))
                        
                btn.clicked.connect(browse_dir)
                
                h_lay.addWidget(le)
                h_lay.addWidget(open_button)
                h_lay.addWidget(btn)
                
                lbl = QLabel(translate("Config", DIRECTORY_LABELS[key]))
                self._label_column.append(lbl)
                layout.addRow(lbl, container)
                directory_profile_controls.extend((lbl, container))
                self.labels[key] = lbl
                self.inputs[key] = le
                if key == "EMBED_DIR":
                    self.directory_base_separator = self._add_padded_separator(
                        layout, "directory_base_separator"
                    )

            # Bind the text changes to dynamically refresh the dropdowns in Tab 1
            self.inputs["FASTA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_fasta, "FASTA_DIR", ['.fasta']))
            self.inputs["MSA_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_msa, "MSA_DIR", ['.fasta', '.h5']))
            self.inputs["HDF5_DIR"].textChanged.connect(lambda: self.refresh_combo(self.cb_hdf5, "HDF5_DIR", ['.h5']))
            # The statistics read these folders' files, so a run on the old ones stops.
            self.inputs["FASTA_DIR"].textChanged.connect(self._cancel_statistics_if_inputs_changed)
            self.inputs["HDF5_DIR"].textChanged.connect(self._cancel_statistics_if_inputs_changed)
            self.inputs["SAVED_LAYOUT_DIR"].textChanged.connect(self.update_live_validators)
            for base_key in (
                "INPUT_FILE_DIR",
                "CACHE_FILE_DIR",
                "ANALYSIS_RESULT_DIR",
                "EMBED_DIR",
            ):
                self.inputs[base_key].textChanged.connect(
                    lambda _text, key=base_key: self._base_directory_changed(key)
                )
                
            self._add_scroll_tab(tab, translate("Config", "Directories"))

        def collect_data(self):
            data = {}
            from PySide6.QtWidgets import QComboBox, QPushButton, QLineEdit
            for key, widget in self.inputs.items():
                
                # ---> NEW: Completely skip saving the target cache selection to JSON
                if key in {"TARGET_CACHE_FILE", "NEW_CACHE_NAME"}:
                    continue
                # The Layout Method dropdown holds two settings.
                if key == "LAYOUT_METHOD":
                    data.update(self._layout_settings())
                    continue
                    
                if isinstance(widget, QComboBox):
                    if key == "LAYOUT_DEVICE_SELECTION":
                        val = widget.currentData()
                    elif widget.property("persistItemData"):
                        val = combo_value(widget)
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
                    elif key in ["MSA_FILE", "EMBED_FILE", "ALIGNMENT_REFERENCE"]: val = ""
                    else: continue
                
                if key == "NODE_FASTA_FILE": val = os.path.join(self.inputs["FASTA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "MSA_FILE": val = os.path.join(self.inputs["MSA_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "EMBED_FILE": val = os.path.join(self.inputs["EMBED_DIR"].text(), val).replace("\\", "/") if val else ""
                elif key == "INPUT_HDF5": val = os.path.join(self.inputs["HDF5_DIR"].text(), val).replace("\\", "/") if val else ""
                
                data[key] = val
            return data

        def _widget_profile_value(self, key):
            if key in LAYOUT_SETTINGS:
                return self._layout_settings()[key]
            hidden = self._scoring_hidden_by_blast
            if hidden and key in hidden and self.cb_score_mode.currentIndex() == -1:
                # A BLAST network blanks both modes; save the choice they hide.
                return hidden[key]
            widget = self.inputs[key]
            if isinstance(widget, QComboBox):
                if key == "LAYOUT_DEVICE_SELECTION":
                    value = widget.currentData()
                elif widget.property("persistItemData"):
                    value = combo_value(widget)
                else:
                    value = widget.currentText()
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
            elif key == "EMBED_FILE":
                value = (
                    os.path.join(self.inputs["EMBED_DIR"].text(), value).replace("\\", "/")
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
                selection = combo_value(self.profile_selectors[tab_id])
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
            messages = [translate("Config", "Settings saved successfully.")]
            if created_profiles:
                created = ", ".join(
                    translate("Config", "{tab}: '{name}'").format(
                        tab=translate("Config", PROFILE_TAB_DISPLAY_NAMES[tab_id]), name=name
                    )
                    for tab_id, name in created_profiles
                )
                messages.append(translate(
                    "Config", "Created profile(s): {profiles}.", None, len(created_profiles)
                ).format(profiles=created))
            if default_tabs:
                unchanged = ", ".join(
                    translate("Config", PROFILE_TAB_DISPLAY_NAMES[tab_id]) for tab_id in default_tabs
                )
                messages.append(translate(
                    "Config", "Built-in default settings were left unchanged for: {tabs}."
                ).format(tabs=unchanged))
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
                self.tip_panel.setStatus(
                    "success", self._save_success_message(created_profiles, default_tabs)
                )
                return True
            except Exception as e:
                self.tip_panel.setStatus(
                    "danger",
                    translate("Config", "Failed to save settings: {error}").format(error=display_text(e)),
                )
                return False

        def _selected_new_cache_filename(self):
            if not self._new_cache_selected():
                raise ValueError(translate(
                    "Config", "Layout settings can only be exported for (New Layout Cache)."
                ))
            cache_name = self.line_new_cache.text().strip()
            if not cache_name:
                cache_name = self.line_new_cache.placeholderText()
            if not cache_name.lower().endswith(".h5"):
                cache_name += ".h5"
            cache_manifest.validate_cache_filename(cache_name)
            return cache_name

        def _collect_layout_generation_settings(self):
            if not self._cache_launch_allowed or not self.current_cache_folder:
                raise ValueError(translate("Config", "A unique compatible cache folder has not been resolved."))

            cache_name = self._selected_new_cache_filename()
            collected = self.collect_data()
            collected["ALIGNMENT_SCORE"] = combo_value(self.cb_score_mode) or None
            collected["NORM_MODE"] = combo_value(self.cb_norm_mode) or None
            collected["SIMILARITY_THRESHOLD"] = self.spin_thresh.optionalValue()
            collected["TOP_EDGE_PERCENT"] = self.spin_top.optionalValue()
            collected["LAYOUT_DEVICE_SELECTION"] = self.cb_layout_device.currentData()
            collected["SAVED_LAYOUT_DIR"] = self.inputs["SAVED_LAYOUT_DIR"].text()
            for path_key in (
                "NODE_FASTA_FILE",
                "INPUT_HDF5",
                "SAVED_LAYOUT_DIR",
            ):
                collected[path_key] = self._resolved_directory_value(
                    collected.get(path_key, "")
                )
            for hidden_key in (
                "BOX_SCALE",
                "PACKING_PADDING",
                "MAX_FORCE_LIMIT",
                "MAX_TOTAL_REPULSION_FORCE",
            ):
                collected[hidden_key] = globals()[hidden_key]

            from utilities.Headless_Settings import build_layout_export
            document = build_layout_export(
                collected, PROJECT_ROOT,
                cache_filename=cache_name,
                target_cache_path=os.path.join(self.current_cache_folder, cache_name),
                automatic=not self.line_new_cache.text().strip(), selection_resolved=True,
            )
            return LayoutGenerationSettings.from_document(document, project_root=PROJECT_ROOT)

        def export_layout_settings(self):
            try:
                settings = self._collect_layout_generation_settings()
                export_directory = Path(
                    self._resolved_directory_value(
                        self.inputs["SETTING_EXPORT_DIR"].text().strip()
                        or DIRECTORY_PROFILE_DEFAULTS["SETTING_EXPORT_DIR"]
                    )
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
                    translate("Config", "Export Layout Settings"),
                    str(export_directory / suggested_name),
                    translate("Config", "JSON Files (*.json)"),
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
                    translate("Config", "Layout Settings Exported"),
                    translate(
                        "Config", "Settings exported to:\n{path}\n\nCommand-line usage:\n{command}"
                    ).format(path=target_path.resolve(), command=command),
                )
            except Exception as error:
                QMessageBox.critical(
                    self, translate("Config", "Export Layout Settings Error"), display_text(error)
                )

        def _update_save_run_button(self):
            """Save & Run is on while a cache can be launched, except in the pause after a click."""
            self.btn_save_run.setEnabled(self._cache_launch_allowed and not self._save_run_paused)

        def _save_and_run_from_button(self):
            """Save & Run from its button, greyed out for CONFIG_RUN_BUTTON_PAUSE_MS.

            It greys out before the Viewer starts, so the second click of a double
            click, queued meanwhile, reaches a disabled button and is dropped. A
            cache check that ends in the pause leaves the button grey until it ends.
            """
            self._save_run_paused = True
            self._update_save_run_button()
            # Timed on the button, so the timer goes with it if the window closes first.
            QTimer.singleShot(CONFIG_RUN_BUTTON_PAUSE_MS, self.btn_save_run, self._end_save_run_pause)
            self.save_and_run()

        def _end_save_run_pause(self):
            self._save_run_paused = False
            self._update_save_run_button()

        def save_and_run(self):
            if not self.uses_manifold_layout():
                try:
                    Hardware_Utils.resolve_device_selection(
                        self.cb_layout_device.currentData()
                    )
                except ValueError as error:
                    QMessageBox.critical(
                        self, translate("Config", "Layout Device Unavailable"), display_text(error)
                    )
                    return

            if not self._cache_launch_allowed or not self.current_cache_folder:
                QMessageBox.critical(
                    self,
                    translate("Config", "Cache Selection Error"),
                    translate("Config", "A unique compatible cache folder has not been resolved."),
                )
                return

            saved_layout_dir = os.path.abspath(
                self._resolved_directory_input("SAVED_LAYOUT_DIR")
            )
            try:
                if self._new_cache_selected():
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
                QMessageBox.critical(self, translate("Config", "Cache Selection Error"), display_text(error))
                return

            settings_data = self.collect_data()
            settings_data["TARGET_CACHE_PATH"] = cache_manifest.resolve_relative_cache_path(saved_layout_dir, relative_path)
            settings_data["TARGET_CACHE_MODE"] = "existing"
            settings_data["SIMILARITY_THRESHOLD"] = self.spin_thresh.optionalValue()
            settings_data["TOP_EDGE_PERCENT"] = self.spin_top.optionalValue()
            if self.uses_manifold_layout():
                settings_data["SIMILARITY_THRESHOLD"] = None
                settings_data["TOP_EDGE_PERCENT"] = None
            elif settings_data["TOP_EDGE_PERCENT"] is not None:
                settings_data["SIMILARITY_THRESHOLD"] = None
            if not self.save_settings():
                return

            try:
                settings_snapshot = _create_viewer_settings_snapshot(settings_data)
            except OSError as error:
                QMessageBox.critical(
                    self,
                    translate("Config", "Viewer Launch Error"),
                    translate("Config", "Failed to create the per-launch settings snapshot:\n{error}").format(
                        error=error
                    ),
                )
                return

            env = os.environ.copy()
            env.pop("SSN_TARGET_CACHE", None)
            env["SSN_TARGET_CACHE_PATH"] = relative_path.replace("\\", "/")
            env["SSN_TARGET_CACHE_MODE"] = cache_mode
            env["SSN_VIEWER_SETTINGS_PATH"] = settings_snapshot

            # Use the project root (parent of src/) as cwd so all relative data paths resolve correctly
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            if cache_mode == "new":
                try:
                    layout_settings = self._collect_layout_generation_settings()
                    layout_doc = layout_settings.to_document(project_root=PROJECT_ROOT)
                    layout_snapshot = _create_layout_settings_snapshot(layout_doc)
                except Exception as error:
                    try:
                        os.unlink(settings_snapshot)
                    except OSError:
                        pass
                    QMessageBox.critical(
                        self,
                        translate("Config", "Layout Settings Error"),
                        translate("Config", "Failed to prepare layout generation settings:\n{error}").format(
                            error=display_text(error)
                        ),
                    )
                    return

                print("Launching Layout_Cache_Generator.py...")
                try:
                    _handoff_to_layout_generator(
                        script_dir, layout_snapshot, env, launch_viewer=True
                    )
                except (OSError, RuntimeError) as error:
                    try:
                        os.unlink(settings_snapshot)
                    except OSError:
                        pass
                    try:
                        os.unlink(layout_snapshot)
                    except OSError:
                        pass
                    QMessageBox.critical(
                        self,
                        translate("Config", "Layout Generator Launch Error"),
                        translate("Config", "Failed to launch Layout_Cache_Generator.py:\n{error}").format(
                            error=error
                        ),
                    )
                    return
            else:
                print("Launching EMAPSSN_Viewer.py...")
                try:
                    _handoff_to_viewer(
                        script_dir,
                        env,
                        settings_path=settings_snapshot,
                    )
                except (OSError, RuntimeError) as error:
                    try:
                        os.unlink(settings_snapshot)
                    except OSError:
                        pass
                    QMessageBox.critical(
                        self,
                        translate("Config", "Viewer Launch Error"),
                        translate("Config", "Failed to launch EMAPSSN_Viewer.py:\n{error}").format(error=error),
                    )
                    return

        def save_only(self):
            self.save_settings()

    existing_qt_application = QApplication.instance()
    app = existing_qt_application or QApplication(sys.argv)
    install_qt_message_handler()  # Qt's warnings reach the Console tagged; a no-op without a capture
    set_app_user_model_id(APP_USER_MODEL_IDS["config"])  # before the first window: the taskbar says EMAP-SSN
    app.setApplicationVersion(APPLICATION_VERSION)
    try:
        install_translations(app, startup_language())
    except Exception as e:
        print(f"Warning: Could not load translations, so text stays English: {e}")
    single_instance = None
    if existing_qt_application is None:
        single_instance = SingleInstanceController("SSN_Config", app)
        try:
            is_primary_instance = single_instance.acquire_or_notify()
        except RuntimeError as error:
            QMessageBox.critical(
                None,
                translate("Config", "{window} Startup Error").format(
                    window=translate("Config", CONFIG_DISPLAY_NAME)
                ),
                str(error),
            )
            raise SystemExit(1)
        if not is_primary_instance:
            print(instance_handover_message(CONFIG_DISPLAY_NAME))
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
        apply_studio_theme(app, configured_theme_color(CONFIG_THEME_SETTING))
    except Exception as e:
        print(f"Warning: Could not apply the window theme: {e}")
        app.setStyle("Fusion")
    Startup_Status.report(Startup_Status.BUILDING_WINDOW, 0.8)
    window = ConfigGUI()
    window.single_instance = single_instance  # a language redraw hands it on
    if single_instance is not None:
        single_instance.set_activation_callback(
            lambda active_window=window: show_window_in_front(active_window)
        )
    show_window_in_front(window)
    sys.exit(app.exec())
