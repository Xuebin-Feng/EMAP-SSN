# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
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
import html
import sys
import os
import ntpath
import posixpath
import ast
import json
import math
import markdown
import re
import traceback

from utilities import Hardware_Utils
from utilities.Embedding_Alignment_Engine import (
    DEFAULT_HOST_CACHE_CAP,
    GIB,
    is_nvidia_cuda,
    normalize_execution_mode,
    tiled_accelerator_support,
)
from utilities.Terminal_Launcher import HoldMode, launch_in_terminal
from utilities.PLM_Plugin_Utils import (
    discover_model_execution_modes,
    discover_model_usage_terms,
)
from utilities.Model_License_Utils import (
    format_model_selector_label,
    format_model_usage_terms,
    is_model_license_accepted,
    record_model_license_acceptance,
)
from utilities.Tool_Directories import (
    DEFAULT_DIRECTORY_PATHS,
    TOOL_DIRECTORY_KEYS,
    fill_missing_directory_defaults,
)
from utilities.Application_Windows import (
    SingleInstanceController,
    show_window_in_front,
)
from utilities.Application_Identity import (
    TOOLS_DESKTOP_FILE_NAME,
    configure_linux_qt_desktop_identity,
)
from Cache_Manifest import (
    file_cache_key,
    inspect_network_completeness,
    validate_network_schema,
)

MAX_CORES = os.cpu_count() or 16
HOST_CACHE_MAX_GB = DEFAULT_HOST_CACHE_CAP / GIB
HOST_CACHE_SLIDER_SCALE = 10
HOST_CACHE_SLIDER_STEPS = round(HOST_CACHE_MAX_GB * HOST_CACHE_SLIDER_SCALE)

SECTION_CARD_STYLE = (
    "QFrame#toolSectionCard { "
    "  border: none; "
    "  border-radius: 8px; "
    "  background-color: #f4f6f8; "
    "  padding: 16px; "
    "  margin-bottom: 20px; "
    "}"
)
PRIMARY_TITLE_STYLE = (
    "font-weight: bold; font-size: 18px; margin-top: 5px; margin-bottom: 5px; "
    "color: #2C3E50; border-bottom: 1px solid #3498DB; padding-bottom: 8px;"
)
COMPACT_ROW_GROUPS = {
    "Sanitize_Sequences.py": [
        ("MIN_SEQ_LENGTH", "MAX_SEQ_LENGTH"),
    ],
    "Generate_Embeddings.py": [
        ("MODEL_NAME", "SAVING_MODE", "DEVICE_SELECTION"),
    ],
    "Align_Similarity_Matrix.py": [
        ("LOCAL_GAP_P", "GLOBAL_GAP_P"),
        ("BATCH_SIZE", "WORKERS"),
        ("ACCELERATOR_PRECISION", "EXECUTION_MODE"),
    ],
    "Align_Substitution_Matrix.py": [
        ("BATCH_SIZE", "NUM_THREADS"),
    ],
    "Network_Injection.py": [
        ("BATCH_SIZE", "WORKERS"),
        ("EXECUTION_MODE", "DEVICE_SELECTION"),
    ],
    "Embedding_MSA.py": [
        ("GAP_OPEN", "GAP_EXTEND"),
    ],
    "Embedding_PWA.py": [
        ("HIGHLIGHT_POSITIONS", "EMBEDDING_MODEL"),
        ("LOCAL_GAP_P", "GLOBAL_GAP_P"),
    ],
    "Embedding_SSEARCH.py": [
        ("OUTPUT_NAME", "TOP_K", "NORM_THRESHOLD"),
        ("ALIGNMENT_MODE", "NORM_MODE"),
        ("LOCAL_GAP_P", "GLOBAL_GAP_P"),
        ("DEVICE_SELECTION", "ACCELERATOR_PRECISION"),
    ],
}
INLINE_FIELD_GROUPS = {
    "Align_Similarity_Matrix.py": [
        ("INPUT_HDF5", "DEVICE_SELECTION"),
    ],
    "Align_Substitution_Matrix.py": [
        ("INPUT_FASTA", "MATRIX"),
    ],
    "Parse_BLAST_Output.py": [
        ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"),
    ],
    "Sanitize_Sequences.py": [
        ("INPUT_FASTA", "OVER_WRITE"),
        ("ENABLE_LENGTH_FILTER", "REMOVE_BY_HEADER_STRING"),
    ],
    "Sparse_MSA_Converter.py": [
        ("CONVERT_ALL", "INPUT_FASTA"),
    ],
    "Embedding_MSA.py": [
        ("USE_SEQUENCE_FILTER", "INPUT_FASTA"),
        ("TREE_METHOD", "NUM_TREES", "BOOTSTRAP_TREE"),
        ("ALIGNMENT_SCORE", "SHOW_REGRESSION_PLOT"),
        ("NORMALIZATION_MODE", "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"),
    ],
    "Embedding_PWA.py": [
        ("REF_HEADER", "MANUAL_REF_SEQ"),
        ("TAR_HEADER", "MANUAL_TAR_SEQ"),
        ("ALIGNMENT_MODE", "GENERATE_REPORT"),
    ],
    "Embedding_SSEARCH.py": [
        ("QUERY_HEADER", "MANUAL_QUERY_SEQ"),
        ("GENERATE_FASTA", "WORKERS"),
    ],
}
INLINE_FIELD_RATIOS = {
    ("INPUT_HDF5", "DEVICE_SELECTION"): (2, 1),
    ("INPUT_FASTA", "MATRIX"): (2, 1),
    ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN"): (1, 1, 1),
}
INLINE_TRAILING_CONTROL_GROUPS = {
    ("INPUT_FASTA", "OVER_WRITE"),
    ("TREE_METHOD", "NUM_TREES", "BOOTSTRAP_TREE"),
    ("ALIGNMENT_SCORE", "SHOW_REGRESSION_PLOT"),
    ("NORMALIZATION_MODE", "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"),
    ("REF_HEADER", "MANUAL_REF_SEQ"),
    ("TAR_HEADER", "MANUAL_TAR_SEQ"),
    ("ALIGNMENT_MODE", "GENERATE_REPORT"),
    ("QUERY_HEADER", "MANUAL_QUERY_SEQ"),
}
SPANNING_TRAILING_CONTROL_GROUPS = {
    ("TREE_METHOD", "NUM_TREES", "BOOTSTRAP_TREE"),
    ("ALIGNMENT_SCORE", "SHOW_REGRESSION_PLOT"),
    ("NORMALIZATION_MODE", "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"),
}
MATCHED_TRAILING_LABEL_VARS = {
    "SHOW_REGRESSION_PLOT",
    "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS",
}
TAB_DISPLAY_NAMES = {
    "Sequence_and_Embedding_Preparation": "Sequence && Embedding Preparation",
    "Embedding_and_Network_Tools": "Embedding && Network Tools",
    "Others": "Manual Tools",
    "Sequence_Similarity_Calculations": "Sequence Similarity Calculations",
}


def imputed_consensus_switch_state(network_info, noise_trees_active, checked):
    """Return the enabled state and explanatory tooltip for the MSA switch."""
    if network_info is None:
        return False, (
            "No network is selected. Select a valid network to determine whether "
            "imputed pairs can be included in the final consensus."
        )

    if network_info.status == "unknown":
        reason = network_info.reason or "The network metadata could not be validated."
        return False, f"Network completeness is unknown: {reason}"

    observed = network_info.edge_count
    expected = network_info.expected_edge_count
    sequences = network_info.sequence_count
    if network_info.status == "complete":
        return False, (
            f"Complete network: {sequences:,} sequences and {observed:,}/{expected:,} "
            "observed pairs. All pairs are already observed, so full cophenetic "
            "consensus is automatic."
        )

    coverage = 100.0 if expected == 0 else 100.0 * observed / expected
    prefix = (
        f"Incomplete network: {sequences:,} sequences and {observed:,}/{expected:,} "
        f"observed pairs ({coverage:.2f}% coverage). "
    )
    if checked:
        behavior = (
            "Imputed pairs participate in every replicate tree and are also replaced "
            "by replicate-averaged cophenetic distances in the final matrix."
        )
    else:
        behavior = (
            "Imputed pairs participate in every replicate tree but retain their "
            "baseline imputed distances in the final matrix."
        )
    if noise_trees_active:
        return True, prefix + behavior
    return False, (
        prefix
        + behavior
        + " Enable Noise-Perturbed Trees with UPGMA to change this setting."
    )

def get_tool_titles():
    """Map tool script filenames to their display titles in the Markdown descriptions."""
    descriptions_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tools",
        "tool_descriptions",
    )
    heading_pattern = re.compile(
        r"^#\s+(.+?)\s+\(`([^`]+\.py)`\)\s*$"
    )
    titles = {}
    if not os.path.isdir(descriptions_dir):
        return titles

    for filename in os.listdir(descriptions_dir):
        if not filename.endswith(".md"):
            continue
        description_path = os.path.join(descriptions_dir, filename)
        try:
            with open(description_path, "r", encoding="utf-8") as description_file:
                for line in description_file:
                    match = heading_pattern.match(line.strip())
                    if match:
                        title, script_name = match.groups()
                        titles[script_name] = title
        except OSError:
            continue
    return titles

def get_supported_embedding_models():
    """
    Scans the src/resources/pLM_models/ folder and parses all scripts
    statically via AST to dynamically build the list of supported embedding models.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(current_dir, "resources", "pLM_models")
    try:
        return list(discover_model_execution_modes(plugin_dir))
    except Exception as error:
        print(f"Failed to discover pLM plugin metadata: {error}")
        return []


def get_embedding_model_execution_modes():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(current_dir, "resources", "pLM_models")
    return discover_model_execution_modes(plugin_dir)


def get_embedding_model_usage_terms():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.join(current_dir, "resources", "pLM_models")
    try:
        return discover_model_usage_terms(plugin_dir)
    except Exception as error:
        print(f"Failed to discover pLM model usage terms: {error}")
        return {}


# Ensure src/ (the directory containing all project modules) is on sys.path.
# This is needed when the script is run as a subprocess or from a different working directory.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Fix High-DPI scaling
os.environ["QT_API"] = "pyside6"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_MAC_WANTS_LIGHT_THEME"] = "1"



from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QFormLayout, QLineEdit, 
                             QPushButton, QMessageBox, QLabel, QScrollArea, QTextEdit,
                             QTextBrowser, QSplitter, QComboBox, QSlider, QDoubleSpinBox, 
                             QSpinBox, QFileDialog, QStyle, QStyleOptionSlider,
                             QSizePolicy, QFrame, QInputDialog)
from PySide6.QtCore import QEvent, Qt

# QtWebEngine ships inside the PySide6-Addons wheel, but its bundled Chromium
# links against system libraries that pip cannot install. On a stock Linux
# desktop the import below is the first thing that fails. Keep the module
# importable on that path so __main__ can report which package is missing
# instead of dumping a traceback; ResponsiveTextBrowser is never instantiated
# when QTWEBENGINE_IMPORT_ERROR is set.
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    QTWEBENGINE_IMPORT_ERROR = None
except ImportError as exc:
    QWebEngineView = object
    QTWEBENGINE_IMPORT_ERROR = str(exc)

from PySide6.QtGui import QColor, QIcon, QPalette
from utilities.Application_Fonts import (
    MONOSPACE_QSS_FONT_STACK,
    UI_QSS_FONT_STACK,
    configure_qt_application_fonts,
    force_light_palette,
)


def confirm_model_usage_terms(parent, model_name, terms):
    """Prompt once per model-and-license fingerprint and persist an opt-in."""
    if is_model_license_accepted(model_name, terms):
        return True

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    while True:
        dialog = QMessageBox(parent)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("External Model License")
        dialog.setText(
            f"{model_name} weights require separate publisher terms."
        )
        dialog.setInformativeText(format_model_usage_terms(model_name, terms))
        accept_button = dialog.addButton(
            "I Accept These Terms",
            QMessageBox.ButtonRole.AcceptRole,
        )
        view_button = dialog.addButton(
            "View License",
            QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = dialog.addButton(
            QMessageBox.StandardButton.Cancel,
        )
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is view_button:
            QDesktopServices.openUrl(QUrl(terms["license_url"]))
            continue
        if clicked is not accept_button:
            return False
        record_model_license_acceptance(model_name, terms)
        return True


def apply_gated_input_palette(widget):
    """Grey disabled inputs without replacing their native control theme."""
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


def _selection_supports_tf32(device_selection, candidates=None):
    """Return whether the effective device selection can execute TF32."""
    candidates = (
        Hardware_Utils.get_available_devices()
        if candidates is None
        else list(candidates)
    )
    normalized = Hardware_Utils.normalize_device_selection(device_selection)
    if normalized == "auto":
        eligible = candidates
    else:
        eligible = [
            candidate for candidate in candidates
            if candidate.spec == normalized
        ]
    return any(
        candidate.backend == "cuda" and is_nvidia_cuda(candidate.device)
        for candidate in eligible
    )


def _sync_tf32_precision_option(device_combo, precision_combo, candidates=None):
    """Show TF32 only when the current hardware selection supports it."""
    selection = device_combo.currentData()
    if selection is None:
        selection = device_combo.currentText()
    available = _selection_supports_tf32(selection, candidates)
    tf32_index = precision_combo.findText("tf32")
    if not available:
        if precision_combo.currentText() == "tf32":
            auto_index = precision_combo.findText("auto")
            precision_combo.setCurrentIndex(max(0, auto_index))
        tf32_index = precision_combo.findText("tf32")
        if tf32_index >= 0:
            precision_combo.removeItem(tf32_index)
    elif tf32_index < 0:
        precision_combo.addItem("tf32")
    precision_combo.setProperty("tf32Available", available)
    return available


QTWEBENGINE_MISSING_MESSAGE = """\
SSN Tools could not load QtWebEngine, which renders the documentation panel.

  {error}

QtWebEngine is installed with PySide6, but its bundled Chromium needs system
libraries that pip cannot provide. Install them, then start SSN Tools again.

Ubuntu / Debian:
  sudo apt install libnss3 libnspr4 libxcomposite1 libxdamage1 libxrandr2 \\
                   libxkbcommon-x11-0 libxtst6 libgbm1 libegl1 libxslt1.1 \\
                   libasound2t64 libcups2t64

  On Ubuntu 22.04 and older, use libasound2 and libcups2 instead -- the t64
  suffix only exists on 24.04 and newer.

Fedora / RHEL:
  sudo dnf install nss nspr libXcomposite libXdamage libXrandr \\
                   libxkbcommon-x11 libXtst mesa-libgbm mesa-libEGL libxslt \\
                   alsa-lib cups-libs

The first line above names the exact library that failed to load; if it is not
covered by these commands, install the package that provides it.\
"""

class ResponsiveTextBrowser(QWebEngineView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setZoomFactor(1.0)
        # Set a white background on the widget itself to prevent black flash during Chromium init
        self.setStyleSheet("background-color: #ffffff;")
        self.page().setBackgroundColor(QColor(255, 255, 255))
        # Warm up the Chromium renderer with a blank white page
        super().setHtml("<html><body style='background:#fff'></body></html>")
        
    def setReadOnly(self, read_only):
        pass
        
    def font(self):
        from PySide6.QtGui import QFont
        return QFont()
        
    def setFont(self, font):
        pass
        
    def setHtml(self, html_content, baseUrl=None):
        github_style = """
        body {
            font-family: __UI_FONT_STACK__;
            font-size: 13.5px;
            line-height: 1.5;
            color: #24292e;
            background-color: #ffffff;
            padding: 24px;
            max-width: 800px;
            min-width: 600px;
            margin: 0 auto;
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
            color: #1f2328;
        }
        h1 {
            font-size: 1.8em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #d0d7de;
        }
        h2 {
            font-size: 1.4em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #d0d7de;
        }
        h3 {
            font-size: 1.15em;
        }
        p, ul, ol {
            margin-top: 0;
            margin-bottom: 16px;
        }
        li {
            margin-top: 0.25em;
        }
        code {
            font-family: __MONOSPACE_FONT_STACK__;
            font-size: 85%;
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 4px;
            color: #1f2328;
        }
        pre {
            font-family: __MONOSPACE_FONT_STACK__;
            font-size: 85%;
            padding: 16px;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            margin-bottom: 16px;
            overflow: auto;
        }
        pre code {
            padding: 0;
            background-color: transparent;
        }
        table {
            border-collapse: collapse;
            border: 1px solid #d0d7de;
            width: 100%;
            margin-top: 0;
            margin-bottom: 16px;
        }
        table th {
            font-weight: 600;
            background-color: #f6f8fa;
            border: 1px solid #d0d7de;
            padding: 6px 10px;
            text-align: left;
        }
        table td {
            border: 1px solid #d0d7de;
            padding: 6px 10px;
            text-align: left;
        }
        details {
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 12px 16px;
            margin-top: 15px;
            margin-bottom: 15px;
            background-color: #f6f8fa;
        }
        summary {
            font-weight: bold;
            font-size: 110%;
            cursor: pointer;
            color: #0969da;
            outline: none;
        }
        details[open] {
            background-color: #ffffff;
        }
        details[open] summary {
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        """.replace("__UI_FONT_STACK__", UI_QSS_FONT_STACK).replace(
            "__MONOSPACE_FONT_STACK__", MONOSPACE_QSS_FONT_STACK
        )
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <link rel="stylesheet" href="fonts/fonts.css">
            <style>
                {github_style}
            </style>
            <!-- KaTeX is vendored under src/resources so math renders offline.
                 These paths are relative to the baseUrl set below. -->
            <link rel="stylesheet" href="katex.min.css">
            <script defer src="katex.min.js"></script>
            <script defer src="katex-auto-render.min.js"
                    onload="renderMathInElement(document.body, {{
                        delimiters: [
                            {{left: '$$', right: '$$', display: true}},
                            {{left: '$', right: '$', display: false}},
                            {{left: '\\\\(', right: '\\\\)', display: false}},
                            {{left: '\\\\[', right: '\\\\]', display: true}}
                        ],
                        throwOnError : false
                    }});"></script>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        # The vendored KaTeX assets are referenced relatively, so the page needs a
        # baseUrl pointing at src/resources/. Without one, setHtml() resolves
        # against about:blank and the relative paths silently fail to load.
        if not baseUrl:
            from PySide6.QtCore import QUrl
            resources_dir = os.path.join(_SRC_DIR, "resources")
            baseUrl = QUrl.fromLocalFile(resources_dir + os.sep)
        super().setHtml(full_html, baseUrl)

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
    def wheelEvent(self, e):
        e.ignore()

class NoScrollSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def wheelEvent(self, e):
        e.ignore()
        
    def mousePressEvent(self, event):
        from PySide6.QtCore import Qt
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            sr = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self)
            
            # If the user clicked the track (not the handle itself), calculate the jump
            if not sr.contains(event.pos()):
                val = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), int(event.position().x()), self.width())
                self.setValue(val)
                event.accept()
                return
        super().mousePressEvent(event)

class NoScrollSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(28)
        self.setStyleSheet("QSpinBox:disabled { background-color: #f0f0f0; color: #888; }")
    def wheelEvent(self, e):
        e.ignore()

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(28)
        self.setStyleSheet("QDoubleSpinBox:disabled { background-color: #f0f0f0; color: #888; }")
    def wheelEvent(self, e):
        e.ignore()


class HostCacheControl(QWidget):
    """Auto/manual host-cache selector with a linear GiB slider."""

    def __init__(self, value="auto", parent=None):
        super().__init__(parent)
        control_layout = QHBoxLayout(self)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(12)

        self.auto_button = QPushButton()
        self.auto_button.setObjectName("hostCacheAutoButton")
        self.auto_button.setAccessibleName("Automatic host cache")
        self.auto_button.setCheckable(True)
        self.auto_button.setFixedSize(82, 28)

        self.slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("hostCacheSlider")
        self.slider.setAccessibleName("Host cache size linear slider")
        self.slider.setRange(0, HOST_CACHE_SLIDER_STEPS)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(HOST_CACHE_SLIDER_STEPS // 4)

        self.spinbox = NoScrollDoubleSpinBox()
        self.spinbox.setObjectName("hostCacheSpinBox")
        self.spinbox.setAccessibleName("Host cache size in GiB")
        self.spinbox.setRange(0.0, HOST_CACHE_MAX_GB)
        self.spinbox.setDecimals(1)
        self.spinbox.setSingleStep(1.0)
        self.spinbox.setFixedWidth(78)
        apply_gated_input_palette(self.spinbox)

        control_layout.addWidget(self.auto_button)
        control_layout.addWidget(self.slider, 1)
        control_layout.addWidget(self.spinbox)

        is_auto = str(value).strip().lower() == "auto"
        try:
            manual_value = float(value) if not is_auto else HOST_CACHE_MAX_GB
        except (TypeError, ValueError):
            is_auto = True
            manual_value = HOST_CACHE_MAX_GB
        if not math.isfinite(manual_value):
            is_auto = True
            manual_value = HOST_CACHE_MAX_GB
        manual_value = min(HOST_CACHE_MAX_GB, max(0.0, manual_value))

        self.slider.valueChanged.connect(self._sync_spinbox_from_slider)
        self.spinbox.valueChanged.connect(self._sync_slider_from_spinbox)
        self.auto_button.toggled.connect(self._apply_auto_state)

        self.spinbox.setValue(manual_value)
        self._sync_slider_from_spinbox(manual_value)
        self.auto_button.setChecked(is_auto)
        self._apply_auto_state(is_auto)

    @staticmethod
    def slider_position_for_gb(value):
        value = min(HOST_CACHE_MAX_GB, max(0.0, float(value)))
        return round(value * HOST_CACHE_SLIDER_SCALE)

    @staticmethod
    def gb_for_slider_position(position):
        position = min(
            HOST_CACHE_SLIDER_STEPS,
            max(0, int(position)),
        )
        return position / HOST_CACHE_SLIDER_SCALE

    def _sync_spinbox_from_slider(self, position):
        value = round(self.gb_for_slider_position(position), 1)
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)

    def _sync_slider_from_spinbox(self, value):
        position = self.slider_position_for_gb(value)
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)

    def _apply_auto_state(self, enabled):
        self.auto_button.setText("AUTO ON" if enabled else "AUTO OFF")
        if enabled:
            self.auto_button.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; "
                "border-radius: 14px; font-weight: bold; "
                "border: 1px solid #388E3C; }"
            )
        else:
            self.auto_button.setStyleSheet(
                "QPushButton { background-color: #e0e0e0; color: #333; "
                "border-radius: 14px; font-weight: bold; "
                "border: 1px solid #bdbdbd; }"
            )
        self.slider.setEnabled(not enabled)
        self.spinbox.setEnabled(not enabled)

    def setting_value(self):
        if self.auto_button.isChecked():
            return "auto"
        value = float(self.spinbox.value())
        return int(value) if value.is_integer() else value

class DynamicComboBox(QComboBox):
    def __init__(self, folder, ext, include_ext=False, exclude_str=None, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.ext = ext
        self.include_ext = include_ext
        self.exclude_str = exclude_str

    def wheelEvent(self, e):
        e.ignore()

    def populate(self):
        current_text = self.currentText()
        current_index = self.currentIndex()
        signals_were_blocked = self.blockSignals(True)
        try:
            self.clear()
            options = []
            if os.path.exists(self.folder):
                for f in os.listdir(self.folder):
                    if f.endswith(self.ext):
                        # --- NEW: Skip files containing the exclusion string ---
                        if self.exclude_str and self.exclude_str in f:
                            continue
                        # -------------------------------------------------------
                        if self.include_ext:
                            options.append(f)
                        else:
                            options.append(f.replace(self.ext, ""))
            self.addItems(options)
            if current_text:
                idx = self.findText(current_text)
                if idx >= 0:
                    self.setCurrentIndex(idx)
        finally:
            self.blockSignals(signals_were_blocked)

        if not signals_were_blocked:
            refreshed_index = self.currentIndex()
            refreshed_text = self.currentText()
            if refreshed_index != current_index:
                self.currentIndexChanged.emit(refreshed_index)
            if refreshed_text != current_text:
                self.currentTextChanged.emit(refreshed_text)

    def showPopup(self):
        self.populate()
        super().showPopup()


def bind_custom_blast_column_controls(inputs, row_widgets):
    """Enable custom BLAST column controls only for the custom layout."""
    layout_input = inputs.get("BLAST_LAYOUT")
    custom_names = ("QUERY_COLUMN", "SUBJECT_COLUMN", "EVALUE_COLUMN")
    if not layout_input or any(name not in inputs for name in custom_names):
        return

    layout_combo = layout_input["widget"]

    def sync_custom_blast_columns(current_layout=None):
        selected_layout = (
            layout_combo.currentData()
            if layout_combo.property("persistItemData")
            else layout_combo.currentText()
        )
        enabled = selected_layout == "custom_columns"
        for name in custom_names:
            inputs[name]["widget"].setEnabled(enabled)
            label = row_widgets.get(name, (None, None))[0]
            if label is not None:
                label.setEnabled(enabled)

    layout_combo.currentTextChanged.connect(sync_custom_blast_columns)
    sync_custom_blast_columns()

def render_markdown_with_math(text):
    # Temporarily hide display math ($$ ... $$) and inline math ($ ... $) from the markdown parser
    block_math = []
    inline_math = []
    
    # Replace display math
    def block_repl(match):
        placeholder = f"<!--BLOCK_MATH_{len(block_math)}-->"
        block_math.append(match.group(0))
        return placeholder
    
    # We use re.DOTALL to handle multi-line display math blocks
    text = re.sub(r"\$\$(.*?)\$\$", block_repl, text, flags=re.DOTALL)
    
    # Replace inline math
    def inline_repl(match):
        placeholder = f"<!--INLINE_MATH_{len(inline_math)}-->"
        inline_math.append(match.group(0))
        return placeholder
        
    text = re.sub(r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$", inline_repl, text)
    
    # Compile markdown to HTML
    html = markdown.markdown(text, extensions=['tables', 'fenced_code', 'md_in_html'])
    
    # Restore inline math
    for i, math_str in enumerate(inline_math):
        html = html.replace(f"<!--INLINE_MATH_{i}-->", math_str)
        
    # Restore display math
    for i, math_str in enumerate(block_math):
        html = html.replace(f"<!--BLOCK_MATH_{i}-->", math_str)
        
    return html

class ToolsGUI(QMainWindow):
    COMMON_TAB_VIEWPORT_MINIMUM_WIDTH = 600

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SSN Utilities Tools")
        self.tool_titles = get_tool_titles()
        
        # Set Window Icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "tool_logo.ico")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "tool_logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.resize(850, 650)
        
        # --- CENTRALIZED SCRIPT TIPS DICTIONARY ---
        self.SCRIPT_TIPS = {
            "Sanitize_Sequences.py": {
                "INPUT_FASTA": "Sequence Set (.fasta): The raw FASTA sequence database to clean.\nUppercases residues, trims terminal non-residues, masks invalid characters with 'X', and deduplicates headers.",
                "ENABLE_LENGTH_FILTER": "Enable Length Filter: Toggle to filter sequences by amino acid length.\nWhen enabled, only sequences within the minimum and maximum length bounds will be retained.",
                "OVER_WRITE": "Overwrite Original File: If ON, replaces the input FASTA file with sanitized sequences.\nIf OFF, creates a new file named <input_name>_sanitized.fasta to preserve the original file.",
                "REMOVE_BY_HEADER_STRING": "Remove Header Substring: Case-sensitive substring filter on raw headers.\nSequences containing this exact text (e.g. 'fragment', 'partial') are discarded. Leave blank to disable.",
                "MIN_SEQ_LENGTH": "Minimum Sequence Length: Lower length bound (inclusive) in amino acids.\nSequences shorter than this threshold will be discarded during sanitization.",
                "MAX_SEQ_LENGTH": "Maximum Sequence Length: Upper length bound (inclusive) in amino acids.\nSequences longer than this threshold will be discarded during sanitization."
            },
            "Generate_Embeddings.py": {
                "INPUT_FASTA": "Sequence Set (.fasta): FASTA sequence database to embed.\nRecords are sanitized in memory (uppercased, invalid residues masked, duplicates merged) before model inference.",
                "MODEL_NAME": "Model Name: Protein language model (pLM) architecture used to generate residue embeddings.\nSupports local ESM-2/ESM-C/ProtBERT/ProstT5/Ankh and remote API models (e.g. esmc_6b with API key).",
                "SAVING_MODE": "Saving Mode: Floating-point precision for storing embedding tensors in HDF5.\nFloat16 saves 50% disk space and RAM with minimal precision loss; float32 retains full precision.",
                "DEVICE_SELECTION": "Device: Hardware compute device used for neural network inference.\nAuto Benchmark profiles CPU and available local accelerators (CUDA, XPU, MPS) on representative sequences."
            },
            "Embedding_Cropping.py": {
                "INPUT_EMBED": "Full Embedding Set (.h5): Pre-computed HDF5 database containing embeddings of full-length sequences.\nContextual residue embeddings for cropped segments are sliced directly from these full-context tensors.",
                "CROPPED_FASTA": "Cropped Sequence Set (.fasta): FASTA file containing partial/cropped sequence segments.\nHeaders and sequences must match exact contiguous substrings within the full embedding database."
            },
            "Align_Similarity_Matrix.py": {
                "INPUT_HDF5": "Embedding Set (.h5): HDF5 database containing dense residue embeddings.\nVectors are used to compute pairwise residue similarity matrices and dynamic programming alignment scores.",
                "EDGE_PREFILTERING": "Edge Prefiltering: Pre-filters sequence pairs using cosine similarity of global pooled embeddings.\nSkips full residue-level dynamic programming for highly dissimilar pairs to accelerate calculation.",
                "PREFILTER_STRENGTH": "Strength (%): Percentage of candidate sequence pairs with lowest cosine similarity to discard.\nHigher percentages speed up calculations by performing residue alignments on only the most promising pairs.",
                "WORKERS": "CPU Workers: Number of parallel CPU worker processes allocated for sequence alignment calculations.\nIncreasing workers speeds up alignment of large datasets across multiple CPU cores.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: Gap penalty applied in Smith-Waterman local alignment.\nMore negative values penalize gap insertions and extensions, resulting in fewer gaps.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: Gap penalty applied in Needleman-Wunsch global alignment.\nControls gap insertion penalties across end-to-end full-length alignments.",
                "BATCH_SIZE": "Batch Size: Number of sequence pairs processed in a single chunk before writing to HDF5.\nLarger values improve throughput but require more RAM. Enter an integer or 'auto'.",
                "DEVICE_SELECTION": "Device: Hardware compute device used for pairwise residue score matrix calculation.\nAuto benchmarks CPU and accelerators; dynamic programming alignment scoring always runs on CPU.",
                "EXECUTION_MODE": "Execution Mode: 'auto' benchmarks scalar and tiled plans where supported.\n'scalar' processes one pairwise score matrix at a time; 'tiled' uses CUDA embedding tiles and padded microbatches and requires a CUDA device.",
                "HOST_CACHE_GB": f"Host Cache (GiB): Maximum RAM used to retain packed embeddings and reduce repeated HDF5 reads.\nAUTO ON selects a safe system-memory budget up to {HOST_CACHE_MAX_GB:g} GiB. Turn AUTO OFF to choose 0 to {HOST_CACHE_MAX_GB:g} GiB with the linear slider or spinbox; 0 disables persistent caching.",
                "ACCELERATOR_PRECISION": "Accelerator Precision: 'auto' tests FP32 and TF32 with every CUDA plan allowed by Execution Mode, validates alignment lengths and scores, and requires at least a 10% best-plan speedup before enabling TF32.\n'float32' preserves IEEE FP32 matmul. 'tf32' is shown only when Auto can use NVIDIA CUDA or an NVIDIA CUDA device is selected explicitly."
            },
            "Align_Substitution_Matrix.py": {
                "INPUT_FASTA": "Sequence Set (.fasta): FASTA sequence database to align with BLASTP.\nRecords undergo canonical header sanitization, residue masking, and duplicate deduplication before alignment.",
                "MATRIX": "Substitution Matrix: Amino acid substitution matrix (e.g. BLOSUM62, PAM250) used for scoring.\nSelect based on the expected evolutionary distance of the sequence set.",
                "NUM_THREADS": "CPU Workers: Number of parallel CPU threads allocated for BLASTP execution and parsing.\nIncreasing threads accelerates all-vs-all search across multi-core systems.",
                "BATCH_SIZE": "Batch Size: Maximum number of parsed alignment edges buffered per chunk during HDF5 writing.\nTuning this parameter controls RAM usage and optimizes disk write performance.",
                "BLASTP_DIR": "BLASTP Directory: Directory containing local blastp and makeblastdb executable binaries.\nIf left blank, standard system PATH and default platform installation locations are searched."
            },
            "Embedding_MSA.py": {
                "USE_SEQUENCE_FILTER": "Use Sequence Filter: Toggle to restrict alignment to sequences in an explicit FASTA file.\nWhen OFF, aligns all sequences present in the intersection of the embedding and network databases.",
                "INPUT_FASTA": "Sequence Set (.fasta): FASTA file used to filter sequences when Sequence Filter is ON.\nIgnored and blanked out when Use Sequence Filter is disabled.",
                "INPUT_EMBED": "Embedding Set (.h5): HDF5 embedding database containing dense residue embeddings.\nUsed to weight progressive profile-profile alignments along evolutionary guide tree nodes.",
                "INPUT_NETWORK": "Network File (.h5): Pairwise similarity network (.h5) used to construct the guide tree.\nFor sparse networks, missing edge scores are automatically imputed using isotonic regression.",
                "SHOW_REGRESSION_PLOT": "Show Isotonic Regression Plot: Displays a diagnostic scatter plot for sparse networks.\nVisualizes the isotonic regression fit between mean embedding cosine distances and network scores.",
                "TREE_METHOD": "Tree Building Method: Algorithm used to construct the evolutionary guide tree.\nUPGMA (Fast) uses average linkage; Neighbor-joining (Slow) accounts for unequal evolutionary rates.",
                "ALIGNMENT_SCORE": "Score Mode: Selects whether to weight guide tree branches using 'global' or 'local' network scores.\nDetermines the hierarchical branching and progressive alignment order.",
                "NORMALIZATION_MODE": "Normalization Mode: Formula used to normalize network scores by sequence or alignment length.\nCorrects for sequence length discrepancies before distance matrix conversion (disabled for BLAST).",
                "BOOTSTRAP_TREE": "Noise-Perturbed Trees: Toggle to average guide trees across randomly perturbed distance replicates.\nAssesses tree sensitivity to distance fluctuations; disabling this runs a single deterministic tree.",
                "NUM_TREES": "Number of Perturbed Trees: Number of noise-perturbed replicate trees used to build consensus.\nHigher values yield a more stable consensus guide tree but increase computation time.",
                "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS": "Include Imputed Pairs in Final Consensus: For incomplete networks, OFF retains baseline imputed distances;\nON replaces all pairs with replicate-averaged cophenetic distances. Imputed pairs participate in all replicate trees.",
                "NOISE_SCALE": "Normalized Noise Scale: Gaussian standard deviation expressed as a fraction of max distance (e.g. 0.02 = 2%).\nApplies additive noise across all observed and regression-imputed distances, clamped to valid bounds.",
                "GAP_OPEN": "Gap Open Penalty: Penalty score applied for opening a new gap in profile alignments.\nMore negative values penalize gap initiation, yielding fewer overall gap regions.",
                "GAP_EXTEND": "Gap Extend Penalty: Penalty score applied for extending an existing gap in profile alignments.\nMore negative values shorten gap lengths.",
                "WORKERS": "CPU Workers: Number of CPU worker processes allocated for parallel guide-tree replicate calculations.\nIncreasing workers accelerates consensus tree generation on multi-core systems.",
                "SAFE_TEMP_DIR": "Temporary Working Directory: Directory for caching intermediate files and memory-mapped matrices.\nEnsures large guide tree and distance matrix calculations do not exceed system RAM."
            },
            "Sparse_MSA_Converter.py": {
                "CONVERT_ALL": "Convert All Alignments: If ON, converts all FASTA MSA files in the alignment directory to sparse HDF5 format.\nIf OFF, converts only the selected FASTA alignment file.",
                "INPUT_FASTA": "Input MSA (.fasta): Standard FASTA multiple sequence alignment file to convert.\nCompresses alignment residues into a SciPy CSR sparse matrix (.h5), reducing file size by up to 95%."
            },
            "Parse_BLAST_Output.py": {
                "INPUT_BLAST_TABULAR": "BLAST Results: Tab-delimited BLASTP output in standard outfmt 6, metadata-bearing outfmt 7, or an explicitly mapped custom layout.",
                "INPUT_FASTA": "Sequence Set (.fasta): Original FASTA used for the BLAST search. Full headers are sanitized without changing or deduplicating sequences and become the viewer node headers.",
                "BLAST_LAYOUT": "BLAST Layout: Standard outfmt 6 requires exactly 12 columns. Outfmt 7 reads the full query from # Query and subject/E-value positions from # Fields. Custom Columns uses the three one-based column settings below.",
                "QUERY_COLUMN": "Query Column: One-based full query-header column used only for Custom Columns.",
                "SUBJECT_COLUMN": "Subject Column: One-based full subject-header column used only for Custom Columns.",
                "EVALUE_COLUMN": "E-Value Column: One-based E-value column used only for Custom Columns."
            },
            "Embedding_Injection.py": {
                "INPUT_EMBED": "Input Embedding Set (.h5): Master HDF5 embedding database to receive new sequences.\nExisting sequence embeddings are preserved and reused without recalculation.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): FASTA file containing existing sequences plus newly added targets.\nEmbeddings are computed only for the newly introduced sequences to optimize compute time."
            },
            "Embedding_Extraction.py": {
                "INPUT_EMBED": "Input Embedding Set (.h5): Master HDF5 embedding database from which subset embeddings are extracted.\nExtracts matching residue embedding datasets without re-running language model inference.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): FASTA file or text list defining the whitelist of sequence headers to extract.\nOnly embeddings matching these headers are saved to the new HDF5 database."
            },
            "Network_Injection.py": {
                "OLD_NETWORK": "Input Network Edges (.h5): Pre-existing HDF5 similarity network file.\nPre-computed alignment scores between existing sequence pairs are reused directly.",
                "NEW_EMBEDDINGS": "Input Embedding Set (.h5): Updated HDF5 embedding database containing all sequence embeddings.\nNewly introduced sequence pairs are aligned and injected into the updated network file.",
                "WORKERS": "CPU Workers: Number of parallel CPU worker processes allocated for dynamic programming alignments.\nDistributes alignment of newly added sequence pairs across CPU cores.",
                "BATCH_SIZE": "Batch Size: Number of sequence alignments calculated and buffered per write block.\nTuning this parameter controls RAM usage and optimizes file write performance.",
                "DEVICE_SELECTION": "Device: Hardware used for new residue score matrices. TF32 source networks require NVIDIA CUDA; dynamic programming remains on CPU.",
                "EXECUTION_MODE": "Execution Mode: 'auto' benchmarks scalar and tiled plans where supported.\n'scalar' processes one pairwise score matrix at a time; 'tiled' uses accelerator embedding tiles and padded microbatches on CUDA/ROCm or XPU.",
                "HOST_CACHE_GB": f"Host Cache (GiB): RAM cap for retaining packed embeddings across injection batches. AUTO ON selects a safe budget up to {HOST_CACHE_MAX_GB:g} GiB; turn it OFF to choose 0 to {HOST_CACHE_MAX_GB:g} GiB with the linear slider or spinbox."
            },
            "Network_Extraction.py": {
                "INPUT_NET": "Input Network Edges (.h5): Master HDF5 network file containing pairwise similarity scores or E-values.\nEdges connecting sequences outside the whitelist are filtered out.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): Whitelist FASTA file defining the subset of sequence nodes to retain.\nOnly edges connecting two whitelist sequences are extracted and re-indexed into the sub-network."
            },
            "Embedding_PWA.py": {
                "INPUT_EMBED": "Embedding Set (.h5): HDF5 database containing pre-computed sequences and residue embeddings.\nSupplies stored sequences and models whenever a manual sequence switch is OFF.",
                "REF_HEADER": "Reference Header: Header of the reference sequence in the embedding database.\nTyped text is sanitized before lookup; if left blank, the first database sequence is used.",
                "MANUAL_REF_SEQ": "Manual Ref Seq: Toggle to enter a raw reference sequence manually.\nWhen OFF, the reference sequence and embedding are loaded from the database by header.",
                "REF_SEQUENCE": "Ref Sequence (Optional): Raw amino acid sequence for the reference protein.\nUsed only when Manual Ref Seq is ON; sanitized and embedded on the fly.",
                "TAR_HEADER": "Target Header: Header of the target sequence in the embedding database.\nTyped text is sanitized before lookup; if left blank, the second database sequence is used.",
                "MANUAL_TAR_SEQ": "Manual Tar Seq: Toggle to enter a raw target sequence manually.\nWhen OFF, the target sequence and embedding are loaded from the database by header.",
                "TAR_SEQUENCE": "Tar Sequence (Optional): Raw amino acid sequence for the target protein.\nUsed only when Manual Tar Seq is ON; sanitized and embedded on the fly.",
                "HIGHLIGHT_POSITIONS": "Highlight Pos (e.g. 1, 4-6): Comma-separated 1-indexed residue positions or ranges in the reference.\nTracked through the alignment and highlighted directly on target sequence positions.",
                "EMBEDDING_MODEL": "Embedding Model: Protein language model used when both sequences are entered manually.\nWhen either sequence is selected from an embedding database, that database's model is used instead.",
                "ALIGNMENT_MODE": "Alignment Mode: Selects global (Needleman-Wunsch) or local (Smith-Waterman) alignment.\nCalculates dynamic programming alignment based on residue embedding cosine similarities.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: Gap penalty applied in Smith-Waterman local alignment.\nMore negative values penalize gap insertions within local alignments.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: Gap penalty applied in Needleman-Wunsch global alignment.\nMore negative values penalize gap insertions across full-length alignments.",
                "GENERATE_REPORT": "Generate Report: Toggle to save an interactive, color-coded HTML alignment report.\nOutputs a formatted report with highlighted residue mappings to the report directory."
            },
            "Embedding_SSEARCH.py": {
                "INPUT_EMBED": "Embedding Set (.h5): Master HDF5 database containing pre-computed sequence embeddings.\nDatabase sequences are scanned against the query sequence using parallelized dynamic programming.",
                "QUERY_HEADER": "Query Header: Header of a sequence in the embedding database used as the search query.\nSanitized before lookup when Manual Query Seq is OFF.",
                "MANUAL_QUERY_SEQ": "Manual Query Seq: Toggle to provide a custom query sequence manually.\nWhen OFF, the query sequence and embedding are loaded from the database by header.",
                "QUERY_SEQUENCE": "Query Sequence (Optional): Raw amino acid sequence for the query protein.\nUsed only when Manual Query Seq is ON; embedded on the fly using the database model.",
                "OUTPUT_NAME": "Output Name: Custom prefix for exported report files (.txt, .xlsx, .fasta).\nIf left blank, defaults to the sanitized query header.",
                "TOP_K": "Top K Hits: Maximum number of highest-scoring database matches to export in results.\nControls the output hit list size in summary tables and reports.",
                "NORM_THRESHOLD": "Norm Score Cutoff: Minimum normalized similarity score threshold for database hits.\nHits scoring below this cutoff are excluded from results. Set to 'None' to disable.",
                "ALIGNMENT_MODE": "Alignment Mode: Selects global (Needleman-Wunsch) or local (Smith-Waterman) alignment.\nCompares database sequence embeddings against the query using dynamic programming.",
                "NORM_MODE": "Normalization Mode: Formula used to normalize alignment scores by sequence or alignment length.\nPrevents score bias toward longer or shorter sequences.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: Gap penalty applied during local (Smith-Waterman) database search.\nMore negative values penalize gap insertions.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: Gap penalty applied during global (Needleman-Wunsch) database search.\nMore negative values penalize gap insertions across full sequences.",
                "WORKERS": "CPU Workers: Number of parallel CPU worker processes allocated for database search.\nRunning with more workers speeds up database scanning on multi-core systems.",
                "GENERATE_FASTA": "Generate FASTA File: Toggle to export a FASTA file containing top hit sequences.\nOutputs the query sequence followed by ranked matching sequences.",
                "DEVICE_SELECTION": "Device: Hardware used for residue score matrices. Searches below 512 targets retain the scalar path; larger CUDA searches may batch targets.",
                "ACCELERATOR_PRECISION": "Accelerator Precision: auto considers validated TF32 only for at least 4,096 targets. The tf32 option is shown only when Auto can use NVIDIA CUDA or an NVIDIA CUDA device is selected explicitly."
            }
        }
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.MANUAL_SETTINGS = {
            "Sequence_and_Embedding_Preparation": {
                "is_combined": True,
                "scripts": {
                    "Sanitize_Sequences.py": [
                        {
                            "var_name": "title_sanitize",
                            "type": "title",
                            "display": "Sequence Sanitization Settings:"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,    # <-- Added
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta):"
                        },
                        {
                            "var_name": "ENABLE_LENGTH_FILTER", # <--- NEW
                            "type": "switch",
                            "display": "Enable Length Filter:"
                        },
                        {
                            "var_name": "OVER_WRITE",
                            "type": "switch",
                            "display": "Overwrite Original File:"
                        },
                        {
                            "var_name": "REMOVE_BY_HEADER_STRING",
                            "type": "text",
                            "display": "Remove Header Substring:"
                        },
                        {
                            "var_name": "MIN_SEQ_LENGTH",
                            "type": "number",
                            "display": "Min Seq Length:"
                        },
                        {
                            "var_name": "MAX_SEQ_LENGTH",
                            "type": "number",
                            "display": "Max Seq Length:"
                        }
                    ],
                    "Generate_Embeddings.py": [
                        {
                            "var_name": "title_embed",
                            "type": "title",
                            "display": "Embedding Generation Settings:"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,    # <-- Added
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta):"
                        },
                        {
                            "var_name": "MODEL_NAME",
                            "type": "dropdown",
                            "options": get_supported_embedding_models(),
                            "model_license_labels": True,
                            "display": "Model Name:"
                        },
                        {
                            "var_name": "SAVING_MODE",
                            "type": "dropdown",
                            "options": ["float16", "float32"],
                            "display": "Saving Mode:"
                        },
                        {
                            "var_name": "DEVICE_SELECTION",
                            "type": "device_dropdown",
                            "display": "Device:"
                        }
                    ],
                    "Embedding_Cropping.py": [
                        {
                            "var_name": "title_crop",
                            "type": "title",
                            "display": "Embedding Cropping Settings:"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",
                            "display": "Full Embedding Set (.h5):"
                        },
                        {
                            "var_name": "CROPPED_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",
                            "display": "Cropped Sequence Set (.fasta):"
                        }
                    ]
                }
            },
            "Sequence_Similarity_Calculations": {
                "is_combined": True,
                "scripts": {
                    "Align_Similarity_Matrix.py": [
                {
                    "var_name": "title_asm_io",
                    "type": "title",
                    "display": "Input & Output Settings:"
                },
                {
                    "var_name": "INPUT_HDF5",
                    "type": "dropdown_from_folder",
                    "folder": "Embeddings",
                    "extension": ".h5",
                    "include_ext": True,
                    "dir_key": "EMBED_DIR",
                    "display": "Embedding Set (.h5):"
                },
                {
                    "var_name": "EDGE_PREFILTERING",
                    "type": "switch",
                    "display": "Edge Prefiltering:"
                },
                {
                    "var_name": "PREFILTER_STRENGTH",
                    "type": "slider",
                    "min": 0,
                    "max": 80,
                    "display": "Strength (%):"
                },
                {
                    "var_name": "title_asm_align",
                    "type": "title",
                    "display": "Alignment Settings:"
                },
                {
                    "var_name": "LOCAL_GAP_P",
                    "type": "negative_number",
                    "display": "Local Align Gap Penalty:"
                },
                {
                    "var_name": "GLOBAL_GAP_P",
                    "type": "negative_number",
                    "display": "Global Align Gap Penalty:"
                },
                {
                    "var_name": "title_asm_hw",
                    "type": "title",
                    "display": "Hardware Settings:"
                },
                {
                    "var_name": "WORKERS",
                    "type": "slider",
                    "min": 1,
                    "max": MAX_CORES,
                    "display": "CPU Workers:"
                },
                {
                    "var_name": "BATCH_SIZE",
                    "type": "text",
                    "display": "Batch Size:"
                },
                {
                    "var_name": "DEVICE_SELECTION",
                    "type": "device_dropdown",
                    "display": "Device:"
                },
                {
                    "var_name": "ACCELERATOR_PRECISION",
                    "type": "dropdown",
                    "options": ["auto", "float32", "tf32"],
                    "display": "Precision:"
                },
                {
                    "var_name": "EXECUTION_MODE",
                    "type": "dropdown",
                    "options": ["auto", "scalar", "tiled"],
                    "display": "Execution Mode:"
                },
                {
                    "var_name": "HOST_CACHE_GB",
                    "type": "host_cache",
                    "display": "Host Cache (GiB):"
                }
                    ],
                    "Align_Substitution_Matrix.py": [
                        {
                            "var_name": "title_sub_io",
                            "type": "title",
                            "display": "Input Settings:"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": (".fasta", ".fa", ".faa"),
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta/.fa/.faa):"
                        },
                        {
                            "var_name": "MATRIX",
                            "type": "dropdown",
                            "options": ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90", "PAM30", "PAM70", "PAM250"],
                            "display": "Substitution Matrix:"
                        },
                        {
                            "var_name": "title_sub_hw",
                            "type": "title",
                            "display": "Hardware & Workspace Settings:"
                        },
                        {
                            "var_name": "NUM_THREADS",
                            "type": "slider",
                            "min": 1,
                            "max": MAX_CORES,
                            "display": "CPU Workers:"
                        },
                        {
                            "var_name": "BATCH_SIZE",
                            "type": "text",
                            "display": "Batch Size:"
                        },
                        {
                            "var_name": "BLASTP_DIR",
                            "type": "folder_browser",
                            "display": "BLASTP Directory:"
                        }
                    ],
                    "Parse_BLAST_Output.py": [
                        {
                            "var_name": "title_parse",
                            "type": "title",
                            "display": "Parse External BLAST Output:"
                        },
                        {
                            "var_name": "INPUT_BLAST_TABULAR",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Networks_EValues"),
                            "extension": (".tabular", ".txt", ".tab", ".tsv"),
                            "include_ext": True,
                            "dir_key": "NETWORK_DIR",
                            "display": "BLAST Results (.tabular/.txt/.tab/.tsv):"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta):"
                        },
                        {
                            "var_name": "BLAST_LAYOUT",
                            "type": "dropdown",
                            "options": [
                                "standard_outfmt6",
                                "outfmt7_fields",
                                "Custom Columns (1-based indexing)"
                            ],
                            "option_values": [
                                "standard_outfmt6",
                                "outfmt7_fields",
                                "custom_columns"
                            ],
                            "display": "BLAST Layout:"
                        },
                        {
                            "var_name": "QUERY_COLUMN",
                            "type": "number",
                            "display": "Query Column:"
                        },
                        {
                            "var_name": "SUBJECT_COLUMN",
                            "type": "number",
                            "display": "Subject Column:"
                        },
                        {
                            "var_name": "EVALUE_COLUMN",
                            "type": "number",
                            "display": "EValue Column:"
                        }
                    ]
                }
            },
            "Embedding_MSA": {
                "is_combined": True,
                "scripts": {
                    "Embedding_MSA.py": [
                        {
                            "var_name": "title_io",
                            "type": "title",
                            "display": "Input & Output Settings:"
                        },
                        {
                            "var_name": "USE_SEQUENCE_FILTER",
                            "type": "switch",
                            "display": "Use Sequence Filter:"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta):"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",
                            "display": "Embedding Set (.h5):"
                        },
                        {
                            "var_name": "INPUT_NETWORK",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Networks_EValues"),
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "NETWORK_DIR",
                            "display": "Network File (.h5):"
                        },
                        {
                            "var_name": "title_guide",
                            "type": "title",
                            "display": "Guide Tree Settings:"
                        },
                        {
                            "var_name": "TREE_METHOD",
                            "type": "dropdown",
                            "options": ["UPGMA (Fast)", "Neighbor-joining (Slow)"],
                            "display": "Tree Building Method:"
                        },
                        {
                            "var_name": "BOOTSTRAP_TREE",
                            "type": "switch",
                            "display": "Noise-Perturbed Trees:"
                        },
                        {
                            "var_name": "NUM_TREES",
                            "type": "number",
                            "display": "Number of Perturbed Trees:"
                        },
                        {
                            "var_name": "NOISE_SCALE",
                            "type": "slider_float",
                            "min": 0,
                            "max": 100,
                            "scale": 1000.0,
                            "display": "Normalized Noise Scale (0 to 0.1):"
                        },
                        {
                            "var_name": "ALIGNMENT_SCORE",
                            "type": "dropdown",
                            "options": ["global", "local"],
                            "display": "Score Mode:"
                        },
                        {
                            "var_name": "SHOW_REGRESSION_PLOT",
                            "type": "switch",
                            "display": "Show Isotonic Regression Plot:"
                        },
                        {
                            "var_name": "NORMALIZATION_MODE",
                            "type": "dropdown",
                            "options": ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"],
                            "display": "Normalization Mode:"
                        },
                        {
                            "var_name": "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS",
                            "type": "switch",
                            "display": "Include Imputed Pairs in Final Consensus:"
                        },
                        {
                            "var_name": "title_align",
                            "type": "title",
                            "display": "Alignment Settings:"
                        },
                        {
                            "var_name": "GAP_OPEN",
                            "type": "negative_number",
                            "display": "Gap Open Penalty:"
                        },
                        {
                            "var_name": "GAP_EXTEND",
                            "type": "negative_number",
                            "display": "Gap Extend Penalty:"
                        },
                        {
                            "var_name": "WORKERS",
                            "type": "slider",
                            "min": 1,
                            "max": MAX_CORES,
                            "display": "CPU Workers:"
                        }
                    ],
                    "Sparse_MSA_Converter.py": [
                        {
                            "var_name": "title_sparse",
                            "type": "title",
                            "display": "Sparse MSA Converter Settings:"
                        },
                        {
                            "var_name": "CONVERT_ALL",
                            "type": "switch",
                            "display": "Convert All Alignments:"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Multiple_Alignments"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "MSA_DIR",
                            "display": "Input MSA (.fasta):"
                        }
                    ]
                }
            },
            "Embedding_and_Network_Tools": {
                "is_combined": True,
                "scripts": {
                    "Embedding_Injection.py": [
                        {
                            "var_name": "title_inj",
                            "type": "title",
                            "display": "Embedding Injection Settings:"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",  # <-- Added
                            "display": "Input Embedding Set (.h5):"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",  # <-- Added
                            "display": "Input Sequence Set (.fasta):"
                        }
                    ],
                    "Embedding_Extraction.py": [
                        {
                            "var_name": "title_ext",
                            "type": "title",
                            "display": "Embedding Extraction Settings:"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",  # <-- Added
                            "display": "Input Embedding Set (.h5):"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",  # <-- Added
                            "display": "Input Sequence Set (.fasta):"
                        }
                    ],
                    "Network_Injection.py": [
                        {
                            "var_name": "title_net_inj",
                            "type": "title",
                            "display": "Network Injection Settings:"
                        },
                        {
                            "var_name": "OLD_NETWORK",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Networks_EValues"),
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "NETWORK_DIR",
                            "display": "Input Network Edges (.h5):"
                        },
                        {
                            "var_name": "NEW_EMBEDDINGS",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",  # <-- Added
                            "display": "Input Embedding Set (.h5):"
                        },
                        {
                            "var_name": "WORKERS",
                            "type": "slider",
                            "min": 1,
                            "max": MAX_CORES,
                            "display": "CPU Workers:"
                        },
                        {
                            "var_name": "BATCH_SIZE",
                            "type": "text",
                            "display": "Batch Size:"
                        },
                        {
                            "var_name": "DEVICE_SELECTION",
                            "type": "device_dropdown",
                            "display": "Device:"
                        },
                        {
                            "var_name": "EXECUTION_MODE",
                            "type": "dropdown",
                            "options": ["auto", "scalar", "tiled"],
                            "display": "Execution Mode:"
                        },
                        {
                            "var_name": "HOST_CACHE_GB",
                            "type": "host_cache",
                            "display": "Host Cache (GiB):"
                        }
                    ],
                    "Network_Extraction.py": [
                        {
                            "var_name": "title_net_ext",
                            "type": "title",
                            "display": "Network Extraction Settings:"
                        },
                        {
                            "var_name": "INPUT_NET",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Networks_EValues"),
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "NETWORK_DIR",  # <-- Added
                            "display": "Input Network Edges (.h5):"
                        },
                        {
                            "var_name": "INPUT_FASTA",
                            "type": "dropdown_from_folder",
                            "folder": os.path.join("Input_Files", "Sequence_Sets"),
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",  # <-- Added
                            "display": "Input Sequence Set (.fasta):"
                        }
                    ]
                }
            },
            "Others": {
                "is_combined": True,
                "scripts": {
                    "Embedding_PWA.py": [
                        {
                            "var_name": "title_pwa_io",
                            "type": "title",
                            "display": "Embedding Input:"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",
                            "display": "Embedding Set (.h5):"
                        },
                        {
                            "var_name": "title_pwa_headers",
                            "type": "title",
                            "display": "Target Sequences:"
                        },
                        {
                            "var_name": "REF_HEADER",
                            "type": "text",
                            "display": "Reference Header:"
                        },
                        {
                            "var_name": "MANUAL_REF_SEQ",
                            "type": "switch",
                            "display": "Manual Ref Seq:"
                        },
                        {
                            "var_name": "REF_SEQUENCE",
                            "type": "text",
                            "display": "Ref Sequence (Optional):"
                        },
                        {
                            "var_name": "TAR_HEADER",
                            "type": "text",
                            "display": "Target Header:"
                        },
                        {
                            "var_name": "MANUAL_TAR_SEQ",
                            "type": "switch",
                            "display": "Manual Tar Seq:"
                        },
                        {
                            "var_name": "TAR_SEQUENCE",
                            "type": "text",
                            "display": "Tar Sequence (Optional):"
                        },
                        {
                            "var_name": "HIGHLIGHT_POSITIONS",
                            "type": "text",
                            "display": "Highlight Pos (e.g., 1, 4-6):"
                        },
                        {
                            "var_name": "EMBEDDING_MODEL",
                            "type": "dropdown",
                            "options": get_supported_embedding_models(),
                            "model_license_labels": True,
                            "display": "Embedding Model:"
                        },
                        {
                            "var_name": "title_pwa_params",
                            "type": "title",
                            "display": "Alignment Parameters:"
                        },
                        {
                            "var_name": "ALIGNMENT_MODE",
                            "type": "dropdown",
                            "options": ["global", "local"],
                            "display": "Alignment Mode:"
                        },
                        {
                            "var_name": "LOCAL_GAP_P",
                            "type": "negative_number",
                            "display": "Local Align Gap Penalty:"
                        },
                        {
                            "var_name": "GLOBAL_GAP_P",
                            "type": "negative_number",
                            "display": "Global Align Gap Penalty:"
                        },
                        {
                            "var_name": "GENERATE_REPORT",
                            "type": "switch",
                            "display": "Generate Report:"
                        }
                    ],
                    "Embedding_SSEARCH.py": [
                        {
                            "var_name": "title_ss_io",
                            "type": "title",
                            "display": "Input Files:"
                        },
                        {
                            "var_name": "INPUT_EMBED",
                            "type": "dropdown_from_folder",
                            "folder": "Embeddings",
                            "extension": ".h5",
                            "include_ext": True,
                            "dir_key": "EMBED_DIR",
                            "display": "Embedding Set (.h5):"
                        },
                        {
                            "var_name": "title_ss_query",
                            "type": "title",
                            "display": "Query Parameters:"
                        },
                        {
                            "var_name": "QUERY_HEADER",
                            "type": "text",
                            "display": "Query Header:"
                        },
                        {
                            "var_name": "MANUAL_QUERY_SEQ",
                            "type": "switch",
                            "display": "Manual Query Seq:"
                        },
                        {
                            "var_name": "QUERY_SEQUENCE",
                            "type": "text",
                            "display": "Query Sequence (Optional):"
                        },
                        {
                            "var_name": "OUTPUT_NAME",
                            "type": "text",
                            "display": "Output Name:"
                        },
                        {
                            "var_name": "TOP_K",
                            "type": "number",
                            "display": "Top K Hits:"
                        },
                        {
                            "var_name": "NORM_THRESHOLD",
                            "type": "text",
                            "display": "Norm Score Cutoff (Optional):"
                        },
                        {
                            "var_name": "title_ss_params",
                            "type": "title",
                            "display": "Alignment Parameters:"
                        },
                        {
                            "var_name": "ALIGNMENT_MODE",
                            "type": "dropdown",
                            "options": ["global", "local"],
                            "display": "Alignment Mode:"
                        },
                        {
                            "var_name": "NORM_MODE",
                            "type": "dropdown",
                            "options": ["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"],
                            "display": "Normalization Mode:"
                        },
                        {
                            "var_name": "LOCAL_GAP_P",
                            "type": "negative_number",
                            "display": "Local Align Gap Penalty:"
                        },
                        {
                            "var_name": "GLOBAL_GAP_P",
                            "type": "negative_number",
                            "display": "Global Align Gap Penalty:"
                        },
                        {
                            "var_name": "WORKERS",
                            "type": "slider",
                            "min": 1,
                            "max": MAX_CORES,
                            "display": "CPU Workers:"
                        },
                        {
                            "var_name": "DEVICE_SELECTION",
                            "type": "device_dropdown",
                            "display": "Device:"
                        },
                        {
                            "var_name": "ACCELERATOR_PRECISION",
                            "type": "dropdown",
                            "options": ["auto", "float32", "tf32"],
                            "display": "Precision:"
                        },
                        {
                            "var_name": "GENERATE_FASTA",
                            "type": "switch",
                            "display": "Generate FASTA File:"
                        }
                    ]
                }
            }
        }
        
        # --- SPLIT LAYOUT ---
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(12)
        self.main_layout.addWidget(self.splitter)
        
        # --- LEFT SIDE SETUP ---
        self.left_widget = QWidget()
        self.left_panel = QVBoxLayout(self.left_widget)
        self.left_panel.setContentsMargins(0, 0, 0, 0)
        self.splitter.addWidget(self.left_widget)
        
        self.left_split = QSplitter(Qt.Orientation.Vertical)
        self.left_split.setHandleWidth(12)
        self.left_panel.addWidget(self.left_split)
        
        # Left Top: Tabs
        self.left_top_widget = QWidget()
        self.left_top_layout = QVBoxLayout(self.left_top_widget)
        self.left_top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.left_top_layout.addWidget(self.tabs)
        
        # Left Bottom: Tip Panel & Action Buttons
        self.left_bottom_widget = QWidget()
        self.left_bottom_layout = QVBoxLayout(self.left_bottom_widget)
        self.left_bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tip_panel = SpacedTipLabel("Hover or focus on an input to see its description.")
        self.tip_panel.setWordWrap(True)
        self.tip_panel.setMinimumHeight(20)
        self.tip_panel.setStyleSheet("color: #444; font-style: normal; background-color: #e8eaed; padding: 10px; border-radius: 5px;")
        self.left_bottom_layout.addWidget(self.tip_panel)
        
        btn_layout = QHBoxLayout()
        btn_exit = QPushButton("Exit")
        btn_exit.clicked.connect(self.close)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_exit)
        self.left_bottom_layout.addLayout(btn_layout)
        
        self.left_split.addWidget(self.left_top_widget)
        self.left_split.addWidget(self.left_bottom_widget)
        
        # Explicitly force the initial pixel heights (tabs get 450px, bottom gets 200px)
        self.left_split.setSizes([450, 200])
        
        # Ensure that if the user resizes the window, extra space goes to the tabs
        self.left_split.setStretchFactor(0, 1)
        self.left_split.setStretchFactor(1, 0)
        
        # --- RIGHT SIDE SETUP ---
        self.right_widget = QWidget()
        self.right_panel = QVBoxLayout(self.right_widget)
        self.splitter.addWidget(self.right_widget)
        
        # Set initial partition to 70% main panel (left) and 30% side panel (right)
        self.splitter.setSizes([70, 30])
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        
        self.desc_title = QLabel("Script Description")
        self.desc_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        self.desc_title.setFixedHeight(25)
        self.right_panel.addWidget(self.desc_title, 0)
        
        self.script_desc_text = ResponsiveTextBrowser()
        self.script_desc_text.setReadOnly(True)
        self.script_desc_text.setStyleSheet("background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px;")
        font = self.script_desc_text.font()
        font.setPointSize(10)
        self.script_desc_text.setFont(font)
        self.right_panel.addWidget(self.script_desc_text, 1)
        
        self.script_data = {} 
        self.tab_paths = [] 
        self._tool_form_layouts = []

        self.tip_db = {}
        self.network_completeness_cache = {}
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.load_tools()
        self.create_directories_tab()
        self._align_all_tool_cards()
        self._harmonize_tab_page_widths()
        self._route_native_tooltips_to_tip_panel()

    def _route_native_tooltips_to_tip_panel(self):
        """Route every native widget tooltip through the shared help panel."""
        for widget in self.findChildren(QWidget):
            if widget.toolTip():
                widget.installEventFilter(self)
    
    def create_directories_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        main_layout = QVBoxLayout(tab)
        
        form_widget = QFrame()
        form_widget.setObjectName("toolSectionCard")
        form_widget.setStyleSheet(SECTION_CARD_STYLE)
        layout = QFormLayout(form_widget)
        layout.setHorizontalSpacing(30)
        layout.setVerticalSpacing(12)
        
        header = QWidget()
        header.setObjectName("toolHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        desc_label = QLabel("📂 Global Directory Settings")
        desc_label.setObjectName("toolTitle")
        desc_label.setStyleSheet(PRIMARY_TITLE_STYLE)

        btn_save = QPushButton("Save Directories")
        btn_save.setObjectName("saveDirectoriesButton")
        btn_save.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px 16px;"
        )
        btn_save.clicked.connect(self.save_directories)

        directory_actions = QWidget()
        directory_actions.setObjectName("directoryActionButtons")
        directory_actions.setProperty(
            "originalSingleButtonHeight", btn_save.sizeHint().height()
        )
        directory_action_layout = QHBoxLayout(directory_actions)
        directory_action_layout.setContentsMargins(0, 0, 0, 0)
        directory_action_layout.setSpacing(0)
        directory_action_layout.addWidget(btn_save)
        directory_action_layout.addStretch()

        header_layout.addWidget(
            directory_actions,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        header_layout.addWidget(desc_label, 1)
        layout.addRow(header)
        
        self.dir_inputs = {}
        self.directory_open_buttons = {}
        dir_defaults = dict(DEFAULT_DIRECTORY_PATHS)
        
        # Load existing paths from JSON if available
        import json
        settings_file = os.path.join(_PROJECT_ROOT, "Input_Files", "tools_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    j_data = json.load(f)
                    saved_directories = j_data.get("DIRECTORIES", {})
                    if isinstance(saved_directories, dict):
                        for key in dir_defaults:
                            if key in saved_directories:
                                dir_defaults[key] = saved_directories[key]
            except: pass
            
        dir_tips = {
            "FASTA_DIR": "Directory containing unaligned FASTA sequence files (.fasta) for sequence sets and subsets.",
            "MSA_DIR": "Directory containing multiple sequence alignment files (.fasta, .h5, or _sparse.h5).",
            "EMBED_DIR": "Directory containing pre-computed protein language model embedding databases (.h5).",
            "NETWORK_DIR": "Directory containing pairwise similarity networks, E-value matrices, and BLAST tabular files (.h5, .tabular).",
            "REPORT_DIR": "Directory where generated pairwise alignment HTML reports and SSEARCH result files are saved.",
            "SETTING_EXPORT_DIR": "Directory where per-tool JSON settings files are exported for command-line execution."
        }
        
        for key, current_val in dir_defaults.items():
            ui_element = QWidget()
            h_lay = QHBoxLayout(ui_element)
            h_lay.setContentsMargins(0, 0, 0, 0)
            
            clean_val_str = str(current_val).replace('r"', '"').replace("r'", "'").strip("\"'")
            le = QLineEdit(clean_val_str)
            open_button = QPushButton("📂")
            open_button.setFixedWidth(30)
            open_button.setToolTip("Open Folder")
            open_button.setEnabled(bool(le.text().strip()))
            btn = QPushButton("Browse...")

            def open_selected_folder(checked=False, line_edit=le):
                raw_path = line_edit.text().strip()
                if not raw_path:
                    return
                folder = os.path.expanduser(raw_path)
                if not os.path.isabs(folder):
                    folder = os.path.join(_PROJECT_ROOT, folder)
                folder = os.path.abspath(folder)
                os.makedirs(folder, exist_ok=True)
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

            open_button.clicked.connect(open_selected_folder)
            le.textChanged.connect(
                lambda text, target=open_button: target.setEnabled(bool(text.strip()))
            )
            self.directory_open_buttons[key] = open_button
            
            def open_folder_dialog(checked=False, line_edit=le):
                folder = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() if line_edit.text() else "")
                if folder:
                    import os
                    line_edit.setText(os.path.normpath(folder))
                    
            btn.clicked.connect(open_folder_dialog)
            h_lay.addWidget(le)
            h_lay.addWidget(open_button)
            h_lay.addWidget(btn)
            
            display_name = key.replace('_', ' ').title()
            display_name = display_name.replace('Msa', 'MSA').replace('Dir', 'Directory')
            display_name = display_name.replace('Fasta', 'FASTA')
            display_name = display_name.replace('Embed', 'Embedding')
            display_name = display_name.replace('Report Directory', 'Alignment Report Directory')
            display_name = display_name.replace('Blastp', 'BLASTP')
            
            lbl = QLabel(f"{display_name}:")
            layout.addRow(lbl, ui_element)
            self.dir_inputs[key] = le
            
            tip = dir_tips.get(key, "")
            ui_element.setToolTip(tip)
            self.tip_db[ui_element] = tip
            self.tip_db[lbl] = tip
            self.tip_db[le] = tip
            ui_element.installEventFilter(self)
            lbl.installEventFilter(self)
            le.installEventFilter(self)
            
        main_layout.addWidget(form_widget)
        main_layout.addStretch() # Pushes the form strictly to the top
        self._tool_form_layouts.append(layout)
        
        self.tabs.addTab(scroll, "Directories")
        self.tab_paths.append("DIRECTORIES_TAB")

    def save_directories(self):
        import json
        import os
        
        # Determine the absolute path of the project root (where SSN_Tools.py lives)
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        new_settings = {}
        for key, le in self.dir_inputs.items():
            raw_path = le.text().strip()
            # Save the path exactly as written
            new_settings[key] = os.path.normpath(raw_path) if raw_path else ""
            
        settings_file = os.path.join(_PROJECT_ROOT, "Input_Files", "tools_settings.json")
        combined_settings = {}
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    combined_settings = json.load(f)
            except: pass
            
        combined_settings["DIRECTORIES"] = new_settings
        
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(combined_settings, f, indent=4)
            QMessageBox.information(self, "Success", "Global directories saved to JSON successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save directories:\n{e}")

    def load_tools(self):
        tools_dir = os.path.join(_SRC_DIR, "tools")
        if not os.path.exists(tools_dir):
            QMessageBox.critical(self, "Error", f"Could not find '{tools_dir}' directory.")
            return
            
        for tab_key, settings_def in self.MANUAL_SETTINGS.items():
            if isinstance(settings_def, dict) and settings_def.get("is_combined"):
                self.create_combined_tab(
                    tools_dir,
                    tab_key,
                    settings_def["scripts"],
                )
            else:
                script_path = os.path.join(tools_dir, tab_key)
                if os.path.exists(script_path):
                    self.create_script_tab(script_path, tab_key, settings_def)
            
        if self.tabs.count() > 0:
            self.on_tab_changed(0)
            
    def on_tab_changed(self, index):
        if index >= 0 and index < len(self.tab_paths):
            path = self.tab_paths[index]
            if path == "DIRECTORIES_TAB":
                dir_md = (
                    "## 📂 Global Directory Settings\n\n"
                    "Define paths to folders used globally across the SSN tool scripts. "
                    "These configurations are automatically saved, validated, and loaded at runtime by all scripts."
                )
                dir_html = render_markdown_with_math(dir_md)
                dir_html = dir_html.replace("<table>", '<table border="1" cellpadding="6" style="border-collapse: collapse;">')
                self.script_desc_text.setHtml(dir_html)
                return
                
            # Get the exact name of the current tab
            tab_name = self.tabs.tabText(index)
            tab_widget = self.tabs.widget(index)
            description_key = (
                tab_widget.property("descriptionKey")
                if tab_widget is not None
                else None
            ) or tab_name
            
            # Formulate the target Markdown file paths (checking both exact match and underscore match)
            md_name = f"{description_key}.md"
            alt_md_name = f"{description_key.replace(' ', '_')}.md"
            
            md_path = os.path.join(_SRC_DIR, "tools", "tool_descriptions", md_name)
            alt_md_path = os.path.join(_SRC_DIR, "tools", "tool_descriptions", alt_md_name)
            
            markdown_content = ""
            
            # 1. Try to load the exact Markdown file
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            # 2. Try the underscore version if the exact one fails
            elif os.path.exists(alt_md_path):
                with open(alt_md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
            
            # 3. Fallback to the Python script's internal docstring if no MD file exists
            if not markdown_content.strip():
                s_data = self.script_data.get(path, {})
                docstring = s_data.get('docstring', '')
                
                if docstring.strip():
                    markdown_content = (
                        f"## 📄 Internal Documentation\n\n"
                        f"```text\n{docstring.strip()}\n```"
                    )
                else:
                    # Final placeholder if absolutely nothing is found
                    markdown_content = (
                        f"## ⚠️ Documentation Missing\n\n"
                        f"No documentation file found for this tab.\n\n"
                        f"To add one, create a Markdown document at:\n\n"
                        f"`{os.path.join('src', 'tools', 'tool_descriptions', md_name)}`"
                    )
            
            html_content = render_markdown_with_math(markdown_content.strip())
            html_content = html_content.replace("<table>", '<table border="1" cellpadding="6" style="border-collapse: collapse;">')
            self.script_desc_text.setHtml(html_content)
            
    def _populate_script_layout(
        self,
        layout,
        script_name,
        script_path,
        script_settings_def,
        source,
        tree,
    ):
        defined_vars = {item["var_name"]: item for item in script_settings_def}
        settings = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in defined_vars:
                        val_str = ast.get_source_segment(source, node.value)
                        
                        # First get the hardcoded default via AST
                        try:
                            default_val = ast.literal_eval(node.value)
                        except Exception:
                            default_val = val_str.strip("\"'")
                            
                        actual_val = default_val
                        
                        # Then, try to overwrite it with the nested JSON value if it exists
                        settings_path = os.path.join(
                            _PROJECT_ROOT, "Input_Files", "tools_settings.json"
                        )
                        if os.path.exists(settings_path):
                            try:
                                with open(settings_path, "r", encoding="utf-8") as f:
                                    j_data = json.load(f)
                                    if script_name in j_data and target.id in j_data[script_name]:
                                        actual_val = j_data[script_name][target.id]
                            except: pass
                            
                        # Dynamic default fallbacks for GUI fields if empty or containing expressions
                        if target.id == "SAFE_TEMP_DIR" and (actual_val is None or str(actual_val).strip() == "" or "os.path" in str(actual_val)):
                            actual_val = os.path.normpath(os.path.join(os.path.expanduser("~"), "Alignment_TEMP"))
                            
                        if target.id == "BLASTP_DIR" and (actual_val is None or str(actual_val).strip() == ""):
                            import shutil
                            default_blastp_dir = ""
                            blastp_path = shutil.which("blastp")
                            if blastp_path:
                                default_blastp_dir = os.path.dirname(os.path.abspath(blastp_path))
                            else:
                                if os.name == 'nt':
                                    ncbi_dir = r"C:\Program Files\NCBI"
                                    if os.path.exists(ncbi_dir):
                                        try:
                                            valid_dirs = []
                                            for d in os.listdir(ncbi_dir):
                                                bin_path = os.path.join(ncbi_dir, d, "bin")
                                                if os.path.exists(os.path.join(bin_path, "blastp.exe")):
                                                    valid_dirs.append(bin_path)
                                            if valid_dirs:
                                                valid_dirs.sort(reverse=True)
                                                default_blastp_dir = valid_dirs[0]
                                        except:
                                            pass
                                else:
                                    unix_fallbacks = [
                                        "/usr/local/ncbi/blast/bin",
                                        "/usr/local/bin",
                                        "/usr/bin",
                                        "/opt/homebrew/bin"
                                    ]
                                    for path in unix_fallbacks:
                                        if os.path.exists(os.path.join(path, "blastp")):
                                            default_blastp_dir = path
                                            break
                            actual_val = default_blastp_dir
                            
                        settings.append({
                            'name': target.id, 'value': val_str, 'actual_val': actual_val,
                            'lineno': node.lineno, 'node': node, 'def': defined_vars[target.id]
                        })
                        
        if len(settings) == 0:
            return
            
        inputs = {}
        row_widgets = {}
        skip_vars = set()
        for s_def in script_settings_def:
            if s_def['type'] == "title":
                continue
                
            var_name = s_def['var_name']
            if var_name in skip_vars:
                continue
                
            # Look up the actual value for the current variable from the parsed settings list
            setting = next((s for s in settings if s['name'] == var_name), None)
            actual_val = setting['actual_val'] if setting else None
            
            if var_name == "EDGE_PREFILTERING":
                # Create the switch button
                switch_btn = QPushButton()
                switch_btn.setCheckable(True)
                switch_btn.setFixedSize(60, 28)
                
                def switch_toggle_style(checked, btn=switch_btn):
                    if checked:
                        btn.setText("ON")
                        btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                    else:
                        btn.setText("OFF")
                        btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
                
                switch_btn.toggled.connect(switch_toggle_style)
                switch_btn.setChecked(bool(actual_val))
                switch_toggle_style(bool(actual_val))
                
                # Get tooltip for prefiltering
                prefilter_tip = self.SCRIPT_TIPS.get(script_name, {}).get("EDGE_PREFILTERING", "Edge Prefiltering")
                switch_btn.setToolTip(prefilter_tip)
                self.tip_db[switch_btn] = prefilter_tip
                switch_btn.installEventFilter(self)
                
                # Create the slider + spinbox
                strength_def = next((d for d in script_settings_def if d.get('var_name') == 'PREFILTER_STRENGTH'), None)
                strength_setting = next((s for s in settings if s['name'] == 'PREFILTER_STRENGTH'), None)
                
                if strength_def and strength_setting:
                    skip_vars.add("PREFILTER_STRENGTH")
                    strength_actual_val = strength_setting['actual_val']
                    
                    strength_widget = QWidget()
                    strength_lay = QHBoxLayout(strength_widget)
                    strength_lay.setContentsMargins(0, 0, 0, 0)
                    
                    sl = NoScrollSlider(Qt.Orientation.Horizontal)
                    sl.setMinimum(strength_def['min'])
                    sl.setMaximum(strength_def['max'])
                    
                    box = NoScrollSpinBox()
                    box.setRange(strength_def['min'], strength_def['max'])
                    box.setFixedWidth(60)
                    
                    try: 
                        val = int(strength_actual_val)
                        sl.setValue(val)
                        box.setValue(val)
                    except: 
                        pass
                    
                    sl.setTickPosition(QSlider.TickPosition.TicksBelow)
                    sl.setTickInterval(10)
                    
                    sl.valueChanged.connect(box.setValue)
                    box.valueChanged.connect(sl.setValue)
                    
                    strength_lay.addWidget(sl)
                    strength_lay.addWidget(box)
                    strength_widget.slider = sl
                    
                    strength_tip = self.SCRIPT_TIPS.get(script_name, {}).get("PREFILTER_STRENGTH", "Strength (%)")
                    strength_widget.setToolTip(strength_tip)
                    self.tip_db[strength_widget] = strength_tip
                    strength_widget.installEventFilter(self)
                    
                    sl.setToolTip(strength_tip)
                    self.tip_db[sl] = strength_tip
                    sl.installEventFilter(self)
                    
                    box.setToolTip(strength_tip)
                    self.tip_db[box] = strength_tip
                    box.installEventFilter(self)
                else:
                    strength_widget = None
                
                # Assemble in compound widget
                compound_widget = QWidget()
                compound_lay = QHBoxLayout(compound_widget)
                compound_lay.setContentsMargins(0, 0, 0, 0)
                compound_lay.setSpacing(12)
                compound_lay.addWidget(switch_btn)
                
                if strength_widget:
                    strength_lbl = QLabel("  Strength (%):")
                    compound_lay.addWidget(strength_lbl)
                    compound_lay.addWidget(strength_widget)
                    
                    # Tooltip for the label
                    strength_lbl.setToolTip(strength_tip)
                    self.tip_db[strength_lbl] = strength_tip
                    strength_lbl.installEventFilter(self)
                    
                    # Connect switch to enable/disable the strength widget
                    def update_strength_state(checked, sl_ref=sl, w_ref=strength_widget):
                        w_ref.setEnabled(checked)
                        if checked:
                            if sl_ref.value() == 0:
                                sl_ref.setValue(20)
                        else:
                            sl_ref.setValue(0)
                    
                    switch_btn.toggled.connect(update_strength_state)
                    update_strength_state(switch_btn.isChecked())
                    
                    # Connect slider changes to automatically toggle switch based on value
                    def on_strength_changed(val, btn_ref=switch_btn):
                        if val == 0:
                            btn_ref.setChecked(False)
                        else:
                            btn_ref.setChecked(True)
                    sl.valueChanged.connect(on_strength_changed)
                    
                    # Register strength in inputs
                    inputs["PREFILTER_STRENGTH"] = {'widget': strength_widget, 'type': 'slider'}
                
                # Create label for the row
                label = QLabel(s_def['display'])
                label.setToolTip(prefilter_tip)
                self.tip_db[label] = prefilter_tip
                label.installEventFilter(self)
                
                layout.addRow(label, compound_widget)
                inputs["EDGE_PREFILTERING"] = {'widget': switch_btn, 'type': 'switch'}
                continue

            if var_name == "ENABLE_LENGTH_FILTER":
                # Create the switch button for length filter
                filter_btn = QPushButton()
                filter_btn.setCheckable(True)
                filter_btn.setFixedSize(60, 28)
                
                def switch_toggle_style_filter(checked, btn=filter_btn):
                    if checked:
                        btn.setText("ON")
                        btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                    else:
                        btn.setText("OFF")
                        btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
                
                filter_btn.toggled.connect(switch_toggle_style_filter)
                filter_btn.setChecked(bool(actual_val))
                switch_toggle_style_filter(bool(actual_val))
                
                filter_tip = self.SCRIPT_TIPS.get(script_name, {}).get("ENABLE_LENGTH_FILTER", "Enable Length Filter")
                filter_btn.setToolTip(filter_tip)
                self.tip_db[filter_btn] = filter_tip
                filter_btn.installEventFilter(self)
                
                # Create label for the row
                label = QLabel(s_def['display'])
                label.setToolTip(filter_tip)
                self.tip_db[label] = filter_tip
                label.installEventFilter(self)
                
                layout.addRow(label, filter_btn)
                row_widgets["ENABLE_LENGTH_FILTER"] = (label, filter_btn)
                inputs["ENABLE_LENGTH_FILTER"] = {'widget': filter_btn, 'type': 'switch'}
                continue
                
            setting = next((s for s in settings if s['name'] == var_name), None)
            if not setting: continue
            
            actual_val = setting['actual_val']
            ui_element = None
            
            if s_def['type'] == "dropdown":
                ui_element = NoScrollComboBox()
                if s_def.get("option_values") is not None:
                    option_values = s_def["option_values"]
                    if len(option_values) != len(s_def["options"]):
                        raise ValueError(
                            f"Dropdown {var_name} has mismatched options and values."
                        )
                    for display_value, stored_value in zip(
                        s_def["options"], option_values
                    ):
                        ui_element.addItem(display_value, stored_value)
                    ui_element.setProperty("persistItemData", True)
                    idx = ui_element.findData(str(actual_val))
                elif s_def.get("model_license_labels", False):
                    usage_terms = get_embedding_model_usage_terms()
                    for model_name in s_def['options']:
                        ui_element.addItem(
                            format_model_selector_label(
                                model_name,
                                usage_terms.get(model_name),
                            ),
                            model_name,
                        )
                    ui_element.setProperty("persistItemData", True)
                    idx = ui_element.findData(str(actual_val))
                else:
                    ui_element.addItems(s_def['options'])
                    idx = ui_element.findText(str(actual_val))
                if idx >= 0: ui_element.setCurrentIndex(idx)

            elif s_def['type'] == "device_dropdown":
                ui_element = NoScrollComboBox()
                for display, spec in Hardware_Utils.device_selection_options():
                    ui_element.addItem(display, spec)
                normalized = Hardware_Utils.normalize_device_selection(actual_val)
                idx = ui_element.findData(normalized)
                if idx < 0 and normalized != "auto":
                    ui_element.addItem(
                        f"Unavailable saved device [{normalized}]", normalized
                    )
                    idx = ui_element.count() - 1
                ui_element.setCurrentIndex(max(0, idx))
                
            elif s_def['type'] == "dropdown_from_folder":
                ui_element = QWidget()
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)

                folder = s_def['folder']
                ext = s_def['extension']
                include_ext = s_def.get('include_ext', False)
                dir_key = s_def.get('dir_key')
                exclude_str = s_def.get('exclude_str', None) # <-- Fetch exclusion
                
                combo = DynamicComboBox(folder, ext, include_ext, exclude_str) # <-- Pass it in
                
                # Override the populate method to fetch the live directory path right before opening
                original_populate = combo.populate
                
                # By passing them as default arguments (c=combo, dk=dir_key, op=original_populate), 
                # Python locks in their values instantly during the loop!
                def dynamic_populate(c=combo, dk=dir_key, op=original_populate):
                    if dk and hasattr(self, 'dir_inputs') and dk in self.dir_inputs:
                        c.folder = self.dir_inputs[dk].text()
                    op()
                    
                combo.populate = dynamic_populate
                
                # Initial population
                combo.populate()
                clean_val = str(actual_val).replace('"','')
                idx = combo.findText(clean_val)
                if idx >= 0: combo.setCurrentIndex(idx)
                
                # Add the folder button
                btn = QPushButton("📂")
                btn.setFixedWidth(30)
                btn.setToolTip("Open Folder")
                def open_folder(checked, dk=dir_key, df=folder):
                    import os
                    from PySide6.QtGui import QDesktopServices
                    from PySide6.QtCore import QUrl
                    path = self.dir_inputs[dk].text() if dk and hasattr(self, 'dir_inputs') and dk in self.dir_inputs else df
                    abs_path = os.path.abspath(path)
                    os.makedirs(abs_path, exist_ok=True)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
                btn.clicked.connect(open_folder)
                
                h_lay.addWidget(combo)
                h_lay.addWidget(btn)
                
                ui_element.combo = combo # Save a reference so save_and_run can extract the text
                
            elif s_def['type'] == "switch":
                ui_element = QPushButton()
                ui_element.setCheckable(True)
                ui_element.setFixedSize(60, 28)
                
                def switch_toggle_style(checked, btn=ui_element):
                    if checked:
                        btn.setText("ON")
                        btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 14px; font-weight: bold; border: 1px solid #388E3C; }")
                    else:
                        btn.setText("OFF")
                        btn.setStyleSheet("QPushButton { background-color: #e0e0e0; color: #333; border-radius: 14px; font-weight: bold; border: 1px solid #bdbdbd; }")
                
                ui_element.toggled.connect(switch_toggle_style)
                ui_element.setChecked(bool(actual_val))
                switch_toggle_style(bool(actual_val)) 
                
            elif s_def['type'] == "slider":
                ui_element = QWidget()
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                sl.setMinimum(s_def['min'])
                sl.setMaximum(s_def['max'])
                
                # Replace QLabel with NoScrollSpinBox
                box = NoScrollSpinBox()
                box.setRange(s_def['min'], s_def['max'])
                box.setFixedWidth(60)
                
                try: 
                    val = int(actual_val)
                    sl.setValue(val)
                    box.setValue(val)
                except: 
                    pass
                
                sl.setTickPosition(QSlider.TickPosition.TicksBelow)
                sl.setTickInterval(1)
                
                # Two-way signal binding
                sl.valueChanged.connect(box.setValue)
                box.valueChanged.connect(sl.setValue)
                
                h_lay.addWidget(sl)
                h_lay.addWidget(box)
                ui_element.slider = sl
                
            elif s_def['type'] == "slider_float":
                ui_element = QWidget()
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                sl = NoScrollSlider(Qt.Orientation.Horizontal)
                vmin = s_def.get('min', 0)
                vmax = s_def.get('max', 100)
                scale = s_def.get('scale', 1000.0) 
                
                sl.setMinimum(vmin)
                sl.setMaximum(vmax)
                
                # Replace QLabel with NoScrollDoubleSpinBox
                box = NoScrollDoubleSpinBox()
                box.setRange(vmin / scale, vmax / scale)
                box.setDecimals(3) # Set to 3 to accommodate a scale of 1000.0 safely
                box.setSingleStep(1.0 / scale)
                box.setFixedWidth(70)
                
                try: 
                    sl_val = int(float(actual_val) * scale)
                    sl.setValue(sl_val)
                    box.setValue(float(actual_val))
                except: 
                    pass
                
                sl.setTickPosition(QSlider.TickPosition.TicksBelow)
                sl.setTickInterval(10)
                
                # Two-way signal binding with scaling math
                sl.valueChanged.connect(lambda v, b=box, sc=scale: b.setValue(v / sc))
                box.valueChanged.connect(lambda v, s=sl, sc=scale: s.setValue(int(v * sc)))
                
                h_lay.addWidget(sl)
                h_lay.addWidget(box)
                ui_element.slider = sl
                ui_element.scale = scale

            elif s_def['type'] == "host_cache":
                ui_element = HostCacheControl(actual_val)

            elif s_def['type'] == "negative_number":
                ui_element = NoScrollDoubleSpinBox()
                ui_element.setMinimum(-1000.0)
                ui_element.setMaximum(0.0)
                ui_element.setDecimals(1)
                ui_element.setSingleStep(0.5)
                try: ui_element.setValue(float(actual_val))
                except: pass
                
            elif s_def['type'] == "number":
                ui_element = NoScrollSpinBox()
                ui_element.setRange(0, 999999)
                try: ui_element.setValue(int(actual_val))
                except: pass

            elif s_def['type'] == "folder_browser":
                ui_element = QWidget()
                h_lay = QHBoxLayout(ui_element)
                h_lay.setContentsMargins(0, 0, 0, 0)
                
                clean_val_str = str(actual_val).replace('r"', '"').replace("r'", "'").strip("\"'")
                
                le = QLineEdit(clean_val_str)
                btn = QPushButton("Browse...")
                
                def open_folder_dialog(checked=False, line_edit=le):
                    folder = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() if line_edit.text() else "")
                    if folder:
                        import os
                        folder = os.path.normpath(folder)
                        line_edit.setText(folder)
                        
                btn.clicked.connect(open_folder_dialog)
                h_lay.addWidget(le)
                h_lay.addWidget(btn)
                ui_element.line_edit = le
                
            else:
                ui_element = QLineEdit(str(actual_val))
            
            script_dict = self.SCRIPT_TIPS.get(script_name, {})
            tip = script_dict.get(var_name, f"Setting: {var_name}")
                
            ui_element.setToolTip(tip)
            self.tip_db[ui_element] = tip
            ui_element.installEventFilter(self)
            
            label = QLabel(s_def['display'])
            label.setStyleSheet("QLabel:disabled { color: #888; }")
            self.tip_db[label] = tip
            label.installEventFilter(self)
            
            if hasattr(ui_element, 'layout') and ui_element.layout() is not None:
                 for child in ui_element.children():
                     if child.isWidgetType():
                         self.tip_db[child] = tip
                         child.installEventFilter(self)

            if (
                script_name == "Embedding_MSA.py"
                and var_name in {"INPUT_EMBED", "INPUT_NETWORK"}
            ):
                responsive_policy = ui_element.sizePolicy()
                responsive_policy.setHorizontalPolicy(
                    QSizePolicy.Policy.Ignored
                )
                ui_element.setSizePolicy(responsive_policy)
                ui_element.setMinimumWidth(0)
            
            layout.addRow(label, ui_element)
            row_widgets[var_name] = (label, ui_element)
            inputs[var_name] = {'widget': ui_element, 'type': s_def['type']}

        self._merge_compact_rows(layout, script_name, row_widgets)
        self._merge_inline_field_rows(layout, script_name, row_widgets)
        self.script_data[script_path] = {'inputs': inputs, 'settings': settings}

        if script_name == "Parse_BLAST_Output.py":
            bind_custom_blast_column_controls(inputs, row_widgets)

        if script_name in {
            "Align_Similarity_Matrix.py",
            "Embedding_SSEARCH.py",
        }:
            device_input = inputs.get("DEVICE_SELECTION")
            precision_input = inputs.get("ACCELERATOR_PRECISION")
            if device_input and precision_input:
                device_combo = device_input["widget"]
                precision_combo = precision_input["widget"]

                def update_precision_options(index=None):
                    _sync_tf32_precision_option(
                        device_combo,
                        precision_combo,
                    )

                device_combo.currentIndexChanged.connect(
                    update_precision_options
                )
                update_precision_options()

        if script_name == "Generate_Embeddings.py":
            model_input = inputs.get("MODEL_NAME")
            device_input = inputs.get("DEVICE_SELECTION")
            if model_input and device_input:
                model_combo = model_input['widget']
                device_combo = device_input['widget']
                execution_modes = get_embedding_model_execution_modes()

                def selected_model_name():
                    return (
                        model_combo.currentData()
                        if model_combo.property("persistItemData")
                        else model_combo.currentText()
                    )

                def update_embedding_device(model_name):
                    is_remote = execution_modes.get(model_name) == "remote_api"
                    if is_remote:
                        if device_combo.currentData() != "__remote_api__":
                            device_combo.setProperty(
                                "localDeviceSelection", device_combo.currentData()
                            )
                        remote_index = device_combo.findData("__remote_api__")
                        if remote_index < 0:
                            device_combo.addItem(
                                "Remote API — local device not applicable",
                                "__remote_api__",
                            )
                            remote_index = device_combo.count() - 1
                        device_combo.setCurrentIndex(remote_index)
                        device_combo.setEnabled(False)
                        device_combo.setToolTip(
                            "Remote API — local device not applicable"
                        )
                    else:
                        local_selection = device_combo.property(
                            "localDeviceSelection"
                        )
                        if device_combo.currentData() == "__remote_api__":
                            restore_index = device_combo.findData(
                                local_selection or "auto"
                            )
                            device_combo.setCurrentIndex(max(0, restore_index))
                        device_combo.setEnabled(True)
                        tip = self.SCRIPT_TIPS[script_name]["DEVICE_SELECTION"]
                        device_combo.setToolTip(tip)

                model_combo.currentIndexChanged.connect(
                    lambda index: update_embedding_device(selected_model_name())
                )
                update_embedding_device(selected_model_name())

        if script_name == "Embedding_MSA.py":
            use_filter_input = inputs.get("USE_SEQUENCE_FILTER")
            fasta_input = inputs.get("INPUT_FASTA")

            if use_filter_input and fasta_input:
                filter_switch = use_filter_input['widget']
                fasta_widget = fasta_input['widget']
                fasta_combo = getattr(fasta_widget, 'combo', None)

                def update_seq_filter_toggle(checked):
                    fasta_widget.setEnabled(checked)
                    if fasta_combo is not None:
                        fasta_combo.blockSignals(True)
                        if not checked:
                            fasta_combo.setCurrentIndex(-1)
                        else:
                            if fasta_combo.currentIndex() == -1 and fasta_combo.count() > 0:
                                fasta_combo.setCurrentIndex(0)
                        fasta_combo.blockSignals(False)

                filter_switch.toggled.connect(update_seq_filter_toggle)
                update_seq_filter_toggle(filter_switch.isChecked())

            net_input = inputs.get("INPUT_NETWORK")
            score_input = inputs.get("ALIGNMENT_SCORE")
            norm_input = inputs.get("NORMALIZATION_MODE")
            bootstrap_input = inputs.get("BOOTSTRAP_TREE")
            num_trees_input = inputs.get("NUM_TREES")
            imputed_consensus_input = inputs.get(
                "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS"
            )
            noise_scale_input = inputs.get("NOISE_SCALE")
            tree_method_input = inputs.get("TREE_METHOD")
            show_plot_input = inputs.get("SHOW_REGRESSION_PLOT")

            if net_input:
                net_combo = net_input['widget'].combo

                def selected_network_path():
                    filename = net_combo.currentText().strip()
                    if not filename:
                        return None
                    if hasattr(self, 'dir_inputs') and "NETWORK_DIR" in self.dir_inputs:
                        network_dir = self.dir_inputs["NETWORK_DIR"].text()
                    else:
                        network_dir = net_combo.folder
                    return os.path.abspath(os.path.join(network_dir, filename))
            
            if net_input and score_input and norm_input and show_plot_input:
                score_combo = score_input['widget']
                norm_combo = norm_input['widget']
                show_plot_switch = show_plot_input['widget']
                show_plot_label = row_widgets.get(
                    "SHOW_REGRESSION_PLOT", (None, None)
                )[0]
                score_default_tip = score_combo.toolTip()
                norm_default_tip = norm_combo.toolTip()
                show_plot_default_tip = show_plot_switch.toolTip()

                def update_show_plot_control(enabled, tip):
                    if not enabled:
                        show_plot_switch.setChecked(False)
                    show_plot_switch.setEnabled(enabled)
                    show_plot_switch.setToolTip(tip)
                    self.tip_db[show_plot_switch] = tip
                    if show_plot_label is not None:
                        show_plot_label.setEnabled(enabled)
                        show_plot_label.setToolTip(tip)
                        self.tip_db[show_plot_label] = tip
                
                def sync_local_norm_mode():
                    if not score_combo.isEnabled():
                        norm_combo.blockSignals(True)
                        norm_combo.clear()
                        norm_combo.setCurrentIndex(-1)
                        norm_combo.blockSignals(False)
                        return
                    
                    current_norm = norm_combo.currentText()
                    norm_combo.blockSignals(True)
                    norm_combo.clear()
                    
                    is_local = score_combo.currentText() == "local"
                    if is_local:
                        norm_combo.addItems(["shorter_sequence", "longer_sequence", "average_sequence"])
                        if current_norm == "alignment_length":
                            current_norm = "longer_sequence"
                    else:
                        norm_combo.addItems(["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"])
                        
                    norm_combo.setCurrentText(current_norm)
                    norm_combo.blockSignals(False)
                
                def update_msa_toggles(_text):
                    network_path = selected_network_path()
                    try:
                        if network_path is None:
                            raise ValueError("No network is selected.")
                        network_metadata = validate_network_schema(network_path)
                    except (OSError, ValueError) as error:
                        error_tip = f"Unable to determine network type: {error}"
                        score_combo.setEnabled(False)
                        norm_combo.setEnabled(False)
                        score_combo.setToolTip(error_tip)
                        norm_combo.setToolTip(error_tip)
                        update_show_plot_control(False, error_tip)
                        score_combo.blockSignals(True)
                        norm_combo.blockSignals(True)
                        score_combo.setCurrentIndex(-1)
                        norm_combo.setCurrentIndex(-1)
                        score_combo.blockSignals(False)
                        norm_combo.blockSignals(False)
                        return

                    is_blast = network_metadata.network_type == "blast"
                    score_combo.setToolTip(score_default_tip)
                    norm_combo.setToolTip(norm_default_tip)
                    if is_blast:
                        update_show_plot_control(
                            False,
                            "Isotonic regression plots are unavailable for "
                            "BLAST networks.",
                        )
                    else:
                        update_show_plot_control(True, show_plot_default_tip)
                    
                    score_combo.setEnabled(not is_blast)
                    norm_combo.setEnabled(not is_blast)
                    
                    score_combo.blockSignals(True)
                    norm_combo.blockSignals(True)
                    if is_blast:
                        score_combo.setCurrentIndex(-1)
                        norm_combo.setCurrentIndex(-1)
                    else:
                        if score_combo.currentIndex() == -1: score_combo.setCurrentText("global")
                    score_combo.blockSignals(False)
                    norm_combo.blockSignals(False)
                    
                    sync_local_norm_mode()
                    if not is_blast and norm_combo.currentIndex() == -1:
                        norm_combo.setCurrentText("alignment_length")
                    
                score_combo.currentTextChanged.connect(lambda text: sync_local_norm_mode())
                net_combo.currentTextChanged.connect(update_msa_toggles)
                update_msa_toggles(net_combo.currentText()) # Trigger once on load
                
            if bootstrap_input and num_trees_input and noise_scale_input:
                bootstrap_switch = bootstrap_input['widget']
                num_trees_widget = num_trees_input['widget']
                noise_scale_widget = noise_scale_input['widget']
                
                def update_bootstrap_toggles(checked):
                    num_trees_widget.setEnabled(checked)
                    noise_scale_widget.setEnabled(checked)
                    
                bootstrap_switch.toggled.connect(update_bootstrap_toggles)
                update_bootstrap_toggles(bootstrap_switch.isChecked())
                
            if tree_method_input and bootstrap_input:
                tree_method_combo = tree_method_input['widget']
                bootstrap_switch = bootstrap_input['widget']
                
                def update_tree_method_toggles(text):
                    is_nj = "neighbor-joining" in text.lower()
                    if is_nj:
                        bootstrap_switch.setChecked(False)
                        bootstrap_switch.setEnabled(False)
                    else:
                        bootstrap_switch.setEnabled(True)
                        
                tree_method_combo.currentTextChanged.connect(update_tree_method_toggles)
                update_tree_method_toggles(tree_method_combo.currentText()) # Trigger once on load

            if net_input and bootstrap_input and imputed_consensus_input:
                bootstrap_switch = bootstrap_input['widget']
                imputed_consensus_switch = imputed_consensus_input['widget']
                imputed_consensus_label = row_widgets.get(
                    "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS", (None, None)
                )[0]
                tree_method_combo = (
                    tree_method_input['widget'] if tree_method_input else None
                )

                def cached_network_completeness():
                    network_path = selected_network_path()
                    if network_path is None:
                        return None
                    try:
                        cache_key = file_cache_key(network_path)
                    except OSError:
                        return inspect_network_completeness(network_path)
                    if cache_key not in self.network_completeness_cache:
                        self.network_completeness_cache[cache_key] = (
                            inspect_network_completeness(network_path)
                        )
                    return self.network_completeness_cache[cache_key]

                def update_imputed_consensus_toggle(*_):
                    noise_trees_active = (
                        bootstrap_switch.isChecked()
                        and bootstrap_switch.isEnabled()
                        and (
                            tree_method_combo is None
                            or "neighbor-joining"
                            not in tree_method_combo.currentText().lower()
                        )
                    )
                    enabled, tip = imputed_consensus_switch_state(
                        cached_network_completeness(),
                        noise_trees_active,
                        imputed_consensus_switch.isChecked(),
                    )
                    imputed_consensus_switch.setEnabled(enabled)
                    imputed_consensus_switch.setToolTip(tip)
                    self.tip_db[imputed_consensus_switch] = tip
                    if imputed_consensus_label is not None:
                        imputed_consensus_label.setToolTip(tip)
                        self.tip_db[imputed_consensus_label] = tip

                net_combo.currentTextChanged.connect(update_imputed_consensus_toggle)
                bootstrap_switch.toggled.connect(update_imputed_consensus_toggle)
                imputed_consensus_switch.toggled.connect(
                    update_imputed_consensus_toggle
                )
                if tree_method_combo is not None:
                    tree_method_combo.currentTextChanged.connect(
                        update_imputed_consensus_toggle
                    )
                update_imputed_consensus_toggle()
        
        if script_name == "Sparse_MSA_Converter.py":
            conv_all_input = inputs.get("CONVERT_ALL")
            fasta_input = inputs.get("INPUT_FASTA")
            
            if conv_all_input and fasta_input:
                conv_all_switch = conv_all_input['widget'] 
                fasta_combo = fasta_input['widget'].combo # Extract the underlying combobox
                
                def update_convert_all(checked):
                    # Grey out the dropdown if Convert All is ON
                    fasta_combo.setEnabled(not checked)
                    
                    if checked:
                        # Clear the selection to indicate it's empty/inactive
                        fasta_combo.setCurrentIndex(-1)
                        
                conv_all_switch.toggled.connect(update_convert_all)
                update_convert_all(conv_all_switch.isChecked()) # Trigger once on load
        
        if script_name == "Sanitize_Sequences.py":
            filter_input = inputs.get("ENABLE_LENGTH_FILTER")
            min_input = inputs.get("MIN_SEQ_LENGTH")
            max_input = inputs.get("MAX_SEQ_LENGTH")
            
            if filter_input and min_input and max_input:
                filter_switch = filter_input['widget'] 
                min_spinbox = min_input['widget'] 
                max_spinbox = max_input['widget']
                
                min_label = row_widgets.get("MIN_SEQ_LENGTH", (None, None))[0]
                max_label = row_widgets.get("MAX_SEQ_LENGTH", (None, None))[0]
                
                def update_length_filters(checked):
                    min_spinbox.setEnabled(checked)
                    max_spinbox.setEnabled(checked)
                    if min_label: min_label.setEnabled(checked)
                    if max_label: max_label.setEnabled(checked)
                        
                filter_switch.toggled.connect(update_length_filters)
                update_length_filters(filter_switch.isChecked()) # Trigger once on load

        if script_name == "Embedding_PWA.py":
            embedding_set_input = inputs.get("INPUT_EMBED")
            ref_toggle_input = inputs.get("MANUAL_REF_SEQ")
            ref_sequence_input = inputs.get("REF_SEQUENCE")
            tar_toggle_input = inputs.get("MANUAL_TAR_SEQ")
            tar_sequence_input = inputs.get("TAR_SEQUENCE")
            model_input = inputs.get("EMBEDDING_MODEL")

            if all((
                embedding_set_input,
                ref_toggle_input,
                ref_sequence_input,
                tar_toggle_input,
                tar_sequence_input,
                model_input,
            )):
                embedding_set_widget = embedding_set_input['widget']
                embedding_set_combo = embedding_set_widget.combo
                ref_toggle = ref_toggle_input['widget']
                ref_sequence = ref_sequence_input['widget']
                tar_toggle = tar_toggle_input['widget']
                tar_sequence = tar_sequence_input['widget']
                model_combo = model_input['widget']
                embedding_set_label = row_widgets.get(
                    "INPUT_EMBED", (None, None)
                )[0]
                ref_sequence_label = row_widgets.get(
                    "REF_SEQUENCE", (None, None)
                )[0]
                tar_sequence_label = row_widgets.get(
                    "TAR_SEQUENCE", (None, None)
                )[0]
                model_label = row_widgets.get(
                    "EMBEDDING_MODEL", (None, None)
                )[0]
                apply_gated_input_palette(embedding_set_combo)
                for gated_widget in (
                    ref_sequence,
                    tar_sequence,
                    model_combo,
                ):
                    apply_gated_input_palette(gated_widget)
                previous_embedding_set = embedding_set_combo.currentText()

                def sync_manual_pairwise_controls():
                    nonlocal previous_embedding_set
                    ref_enabled = ref_toggle.isChecked()
                    tar_enabled = tar_toggle.isChecked()
                    both_manual = ref_enabled and tar_enabled
                    if both_manual:
                        current_embedding_set = embedding_set_combo.currentText()
                        if current_embedding_set:
                            previous_embedding_set = current_embedding_set
                        embedding_set_combo.setCurrentIndex(-1)
                    elif embedding_set_combo.currentIndex() < 0:
                        restore_index = embedding_set_combo.findText(
                            previous_embedding_set
                        )
                        if restore_index >= 0:
                            embedding_set_combo.setCurrentIndex(restore_index)

                    embedding_set_widget.setEnabled(not both_manual)
                    if embedding_set_label is not None:
                        embedding_set_label.setEnabled(not both_manual)
                    model_enabled = (
                        both_manual
                        and model_combo.count() > 0
                    )
                    for widget, label, enabled in (
                        (ref_sequence, ref_sequence_label, ref_enabled),
                        (tar_sequence, tar_sequence_label, tar_enabled),
                        (model_combo, model_label, model_enabled),
                    ):
                        widget.setEnabled(enabled)
                        if label is not None:
                            label.setEnabled(enabled)

                ref_toggle.toggled.connect(
                    lambda checked: sync_manual_pairwise_controls()
                )
                tar_toggle.toggled.connect(
                    lambda checked: sync_manual_pairwise_controls()
                )
                sync_manual_pairwise_controls()

        if script_name == "Embedding_SSEARCH.py":
            score_input = inputs.get("ALIGNMENT_MODE")
            norm_input = inputs.get("NORM_MODE")
            query_toggle_input = inputs.get("MANUAL_QUERY_SEQ")
            query_sequence_input = inputs.get("QUERY_SEQUENCE")

            if query_toggle_input and query_sequence_input:
                query_toggle = query_toggle_input['widget']
                query_sequence = query_sequence_input['widget']
                query_sequence_label = row_widgets.get(
                    "QUERY_SEQUENCE", (None, None)
                )[0]
                apply_gated_input_palette(query_sequence)

                def sync_manual_query_control(checked):
                    query_sequence.setEnabled(checked)
                    if query_sequence_label is not None:
                        query_sequence_label.setEnabled(checked)

                query_toggle.toggled.connect(sync_manual_query_control)
                sync_manual_query_control(query_toggle.isChecked())
            
            if score_input and norm_input:
                score_combo = score_input['widget']
                norm_combo = norm_input['widget']
                
                def sync_local_norm_mode_ssearch():
                    current_norm = norm_combo.currentText()
                    norm_combo.blockSignals(True)
                    norm_combo.clear()
                    
                    is_local = score_combo.currentText() == "local"
                    if is_local:
                        norm_combo.addItems(["shorter_sequence", "longer_sequence", "average_sequence"])
                        if current_norm == "alignment_length":
                            current_norm = "longer_sequence"
                    else:
                        norm_combo.addItems(["alignment_length", "shorter_sequence", "longer_sequence", "average_sequence"])
                        
                    norm_combo.setCurrentText(current_norm)
                    norm_combo.blockSignals(False)
                    
                score_combo.currentTextChanged.connect(lambda text: sync_local_norm_mode_ssearch())
                sync_local_norm_mode_ssearch() # Trigger once on load

    @staticmethod
    def _merge_compact_rows(layout, script_name, row_widgets):
        for variable_group in COMPACT_ROW_GROUPS.get(script_name, []):
            if any(var_name not in row_widgets for var_name in variable_group):
                continue

            group_widgets = [
                (var_name, *row_widgets[var_name])
                for var_name in variable_group
            ]
            row_positions = [
                layout.getWidgetPosition(label_widget)[0]
                for _, label_widget, _ in group_widgets
            ]
            if any(row < 0 for row in row_positions):
                continue

            insertion_row = min(row_positions)
            for row in sorted(row_positions, reverse=True):
                layout.takeRow(row)

            is_batch_worker_pair = (
                len(variable_group) == 2
                and variable_group[0] == "BATCH_SIZE"
                and variable_group[1] in {"WORKERS", "NUM_THREADS"}
            )
            if is_batch_worker_pair:
                first_var, first_label, first_input = group_widgets[0]
                second_var, second_label, second_input = group_widgets[1]
                field_row = QWidget()
                field_row.setObjectName(
                    f"compactRow_{first_var}_{second_var}"
                )
                field_row.setProperty("compactColumnRatio", "1:3")
                field_layout = QHBoxLayout(field_row)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.setSpacing(layout.horizontalSpacing())

                first_column = QWidget()
                first_column.setObjectName(
                    f"compactColumn_left_{first_var}_{second_var}"
                )
                first_column_layout = QHBoxLayout(first_column)
                first_column_layout.setContentsMargins(0, 0, 0, 0)
                first_column_layout.addWidget(first_input)
                first_input.setMinimumWidth(110)

                second_column = QWidget()
                second_column.setObjectName(
                    f"compactColumn_right_{first_var}_{second_var}"
                )
                second_column_layout = QHBoxLayout(second_column)
                second_column_layout.setContentsMargins(0, 0, 0, 0)
                second_column_layout.setSpacing(layout.horizontalSpacing())
                second_column_layout.addWidget(second_label)
                second_column_layout.addWidget(second_input, 1)

                for column in (first_column, second_column):
                    column.setSizePolicy(
                        QSizePolicy.Policy.Ignored,
                        QSizePolicy.Policy.Preferred,
                    )

                field_layout.addWidget(first_column, 1)
                field_layout.addWidget(second_column, 3)
                layout.insertRow(insertion_row, first_label, field_row)
                continue

            compact_row = QWidget()
            group_name = "_".join(variable_group)
            compact_row.setObjectName(
                f"compactRow_{group_name}"
            )
            compact_layout = QHBoxLayout(compact_row)
            compact_layout.setContentsMargins(0, 0, 0, 0)
            compact_layout.setSpacing(30)

            columns = []
            column_names = (
                ("left", "right")
                if len(group_widgets) == 2
                else tuple(str(index) for index in range(len(group_widgets)))
            )
            for column_name, (_, label_widget, input_widget) in zip(
                column_names,
                group_widgets,
            ):
                if label_widget is group_widgets[0][1]:
                    label_widget.setProperty("compactColumnLabel", True)
                column = QWidget()
                column.setObjectName(
                    f"compactColumn_{column_name}_{group_name}"
                )
                column_layout = QHBoxLayout(column)
                column_layout.setContentsMargins(0, 0, 0, 0)
                column_layout.setSpacing(layout.horizontalSpacing())
                column_layout.addWidget(label_widget)
                column_layout.addWidget(input_widget, 1)
                columns.append(column)

            shared_column_width = max(
                column.sizeHint().width() for column in columns
            )
            compact_row.setProperty(
                "compactColumnRatio",
                ":".join("1" for _ in group_widgets),
            )
            for column in columns:
                column.setMinimumWidth(shared_column_width)

            for column in columns:
                compact_layout.addWidget(column, 1)

            layout.insertRow(insertion_row, compact_row)

    @staticmethod
    def _merge_inline_field_rows(layout, script_name, row_widgets):
        for variable_group in INLINE_FIELD_GROUPS.get(script_name, []):
            if any(var_name not in row_widgets for var_name in variable_group):
                continue

            group_widgets = [
                (var_name, *row_widgets[var_name])
                for var_name in variable_group
            ]
            row_positions = [
                layout.getWidgetPosition(label_widget)[0]
                for _, label_widget, _ in group_widgets
            ]
            if any(row < 0 for row in row_positions):
                continue

            insertion_row = min(row_positions)
            for row in sorted(row_positions, reverse=True):
                layout.takeRow(row)

            group_name = "_".join(variable_group)
            field_row = QWidget()
            field_row.setObjectName(f"compactRow_{group_name}")
            column_ratios = INLINE_FIELD_RATIOS.get(
                variable_group,
                tuple(1 for _ in variable_group),
            )
            field_row.setProperty(
                "compactColumnRatio",
                ":".join(str(ratio) for ratio in column_ratios),
            )
            field_layout = QHBoxLayout(field_row)
            field_layout.setContentsMargins(0, 0, 0, 0)

            first_label = group_widgets[0][1]
            first_input = group_widgets[0][2]
            if variable_group in INLINE_TRAILING_CONTROL_GROUPS:
                field_row.setProperty("compactColumnRatio", "inline")
                field_row.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                field_layout.setSpacing(layout.horizontalSpacing())
                is_spanning = (
                    variable_group in SPANNING_TRAILING_CONTROL_GROUPS
                )
                if is_spanning:
                    form_spacing = max(0, layout.horizontalSpacing())
                    field_layout.setSpacing(0)
                    first_input_policy = first_input.sizePolicy()
                    first_input_policy.setHorizontalPolicy(
                        QSizePolicy.Policy.Ignored
                    )
                    first_input.setSizePolicy(first_input_policy)
                    first_input.setMinimumWidth(0)
                    field_layout.addWidget(first_input, 1)
                else:
                    first_input_policy = first_input.sizePolicy()
                    first_input_policy.setHorizontalPolicy(
                        QSizePolicy.Policy.Ignored
                    )
                    first_input.setSizePolicy(first_input_policy)
                    first_input.setMinimumWidth(0)
                    field_layout.addWidget(first_input, 1)
                for var_name, label_widget, input_widget in group_widgets[1:]:
                    if is_spanning:
                        field_layout.addSpacing(
                            30 if variable_group == (
                                "TREE_METHOD",
                                "NUM_TREES",
                                "BOOTSTRAP_TREE",
                            ) else 12
                        )
                    else:
                        label_widget.setText(
                            f"   {label_widget.text().lstrip()}"
                        )
                    if var_name in MATCHED_TRAILING_LABEL_VARS:
                        label_widget.setProperty("matchedTrailingLabel", True)
                    field_layout.addWidget(label_widget)
                    if isinstance(input_widget, QPushButton):
                        field_layout.addSpacing(
                            10 + (2 * form_spacing if is_spanning else 0)
                        )
                    elif is_spanning:
                        field_layout.addSpacing(6)
                    if var_name == "NUM_TREES":
                        input_policy = input_widget.sizePolicy()
                        input_policy.setHorizontalPolicy(
                            QSizePolicy.Policy.Ignored
                        )
                        input_widget.setSizePolicy(input_policy)
                        input_widget.setMinimumWidth(0)
                        field_layout.addWidget(input_widget, 1)
                    else:
                        field_layout.addWidget(input_widget)
                layout.insertRow(insertion_row, first_label, field_row)
                continue

            if isinstance(first_input, QPushButton):
                field_row.setProperty("compactColumnRatio", "inline")
                field_row.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                field_layout.addWidget(first_input)
                for _, label_widget, input_widget in group_widgets[1:]:
                    label_widget.setText(
                        f"   {label_widget.text().lstrip()}"
                    )
                    field_layout.addWidget(label_widget)
                    input_policy = input_widget.sizePolicy()
                    input_policy.setHorizontalPolicy(
                        QSizePolicy.Policy.Ignored
                    )
                    input_widget.setSizePolicy(input_policy)
                    input_widget.setMinimumWidth(0)
                    field_layout.addWidget(input_widget, 1)
                layout.insertRow(insertion_row, first_label, field_row)
                continue

            field_layout.setSpacing(layout.horizontalSpacing())
            for column_index, (_, label_widget, input_widget) in enumerate(
                group_widgets
            ):
                column = QWidget()
                column.setObjectName(
                    f"compactColumn_{column_index}_{group_name}"
                )
                column_layout = QHBoxLayout(column)
                column_layout.setContentsMargins(0, 0, 0, 0)
                column_layout.setSpacing(layout.horizontalSpacing())
                if column_index > 0:
                    column_layout.addWidget(label_widget)
                column_layout.addWidget(input_widget, 1)
                column.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                field_layout.addWidget(column, column_ratios[column_index])

            layout.insertRow(insertion_row, first_label, field_row)

    def _create_tool_header(self, script_name, script_path):
        header = QWidget()
        header.setObjectName("toolHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        tool_title = self.tool_titles.get(
            script_name,
            script_name.removesuffix(".py").replace("_", " "),
        )
        title_label = QLabel(tool_title)
        title_label.setObjectName("toolTitle")
        title_label.setStyleSheet(PRIMARY_TITLE_STYLE)

        btn_run = QPushButton("Save && Run")
        btn_run.setObjectName("saveRunButton")
        btn_run.setToolTip(
            "Save the current tool settings to the shared settings file "
            "and run this tool."
        )
        btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px 16px;"
        )
        original_button_height = btn_run.sizeHint().height()
        btn_run.clicked.connect(
            lambda checked, sp=script_path: self.save_and_run(sp)
        )

        btn_export = QPushButton("Export\nSetting")
        btn_export.setObjectName("exportSettingButton")
        btn_export.setAccessibleName("Export Settings")
        btn_export.setToolTip(
            "Export the current tool settings to a standalone JSON file "
            "for command-line execution."
        )
        btn_export.setStyleSheet(
            "background-color: #3498DB; color: white; "
            "font-weight: bold; font-size: 10px; padding: 1px 8px;"
        )
        btn_export.clicked.connect(
            lambda checked, sp=script_path: self.export_settings(sp)
        )

        button_height = max(original_button_height, btn_export.sizeHint().height())
        for button in (btn_run, btn_export):
            button.setFixedHeight(button_height)

        button_row = QWidget()
        button_row.setObjectName("toolActionButtons")
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        button_layout.addWidget(btn_run)
        button_layout.addWidget(btn_export)
        button_row.setProperty("originalSingleButtonHeight", button_height)

        header_layout.addWidget(
            button_row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        header_layout.addWidget(title_label, 1)
        return header

    @staticmethod
    def _align_form_label_columns(form_layouts):
        label_widgets = []
        matched_trailing_labels = []
        compact_rows = []
        shared_width = 0

        for form_layout in form_layouts:
            for row in range(form_layout.rowCount()):
                label_item = form_layout.itemAt(
                    row,
                    QFormLayout.ItemRole.LabelRole,
                )
                if label_item is None or label_item.widget() is None:
                    continue
                label_widget = label_item.widget()
                label_widgets.append(label_widget)
                shared_width = max(shared_width, label_widget.sizeHint().width())

            form_widget = form_layout.parentWidget()
            compact_labels = [
                label
                for label in form_widget.findChildren(QLabel)
                if label.property("compactColumnLabel")
            ]
            label_widgets.extend(compact_labels)
            for label_widget in compact_labels:
                shared_width = max(
                    shared_width,
                    label_widget.sizeHint().width(),
                )
            compact_rows.extend(
                row
                for row in form_widget.findChildren(QWidget)
                if row.objectName().startswith("compactRow_")
            )
            matched_trailing_labels.extend(
                label
                for label in form_widget.findChildren(QLabel)
                if label.property("matchedTrailingLabel")
            )
        for label_widget in label_widgets:
            label_widget.setFixedWidth(shared_width)

        if matched_trailing_labels:
            matched_width = max(
                label.sizeHint().width() for label in matched_trailing_labels
            )
            for label_widget in matched_trailing_labels:
                label_widget.setFixedWidth(matched_width)

        for compact_row in compact_rows:
            column_ratio = str(compact_row.property("compactColumnRatio"))
            if not column_ratio or any(
                ratio != "1" for ratio in column_ratio.split(":")
            ):
                continue
            columns = [
                compact_row.layout().itemAt(index).widget()
                for index in range(compact_row.layout().count())
            ]
            shared_column_width = max(
                column.sizeHint().width() for column in columns
            )
            for column in columns:
                column.setMinimumWidth(shared_column_width)

        return shared_width

    @staticmethod
    def _align_tool_card_headers(form_layouts, shared_label_width):
        if not form_layouts:
            return 0

        horizontal_spacing = max(
            max(0, form_layout.horizontalSpacing())
            for form_layout in form_layouts
        )
        title_start_x = shared_label_width + horizontal_spacing
        action_gap = 10
        full_button_width = max(1, (title_start_x - action_gap) // 2)
        run_button_width = full_button_width
        export_button_width = max(1, round(full_button_width * 0.6))
        trailing_space = max(
            0,
            title_start_x
            - (run_button_width + export_button_width + action_gap),
        )

        for form_layout in form_layouts:
            form_layout.setHorizontalSpacing(horizontal_spacing)
            header = None
            for row in range(form_layout.rowCount()):
                spanning_item = form_layout.itemAt(
                    row,
                    QFormLayout.ItemRole.SpanningRole,
                )
                if (
                    spanning_item is not None
                    and spanning_item.widget() is not None
                    and spanning_item.widget().objectName() == "toolHeader"
                ):
                    header = spanning_item.widget()
                    break
            if header is None:
                continue

            header_layout = header.layout()
            header_layout.setSpacing(0)
            action_widget = header_layout.itemAt(0).widget()
            if action_widget is None:
                continue

            button_height = int(
                action_widget.property("originalSingleButtonHeight") or 0
            )
            action_widget.setFixedSize(title_start_x, button_height)

            if action_widget.objectName() == "toolActionButtons":
                action_layout = action_widget.layout()
                action_layout.setContentsMargins(0, 0, trailing_space, 0)
                action_layout.setSpacing(action_gap)
                for button in action_widget.findChildren(
                    QPushButton,
                    options=Qt.FindChildOption.FindDirectChildrenOnly,
                ):
                    width = (
                        run_button_width
                        if button.objectName() == "saveRunButton"
                        else export_button_width
                    )
                    button.setFixedSize(width, button_height)
            elif action_widget.objectName() == "directoryActionButtons":
                save_button = action_widget.findChild(
                    QPushButton,
                    "saveDirectoriesButton",
                    Qt.FindChildOption.FindDirectChildrenOnly,
                )
                if save_button is not None:
                    save_button.setFixedSize(
                        round(full_button_width * 1.5),
                        button_height,
                    )

            header.setProperty("sharedTitleStartX", title_start_x)

        return title_start_x

    def _align_all_tool_cards(self):
        shared_label_width = self._align_form_label_columns(
            self._tool_form_layouts
        )
        self._align_tool_card_headers(
            self._tool_form_layouts,
            shared_label_width,
        )

    def _harmonize_tab_page_widths(self):
        scroll_pages = [
            self.tabs.widget(index)
            for index in range(self.tabs.count())
            if isinstance(self.tabs.widget(index), QScrollArea)
        ]
        content_pages = [
            scroll_page.widget()
            for scroll_page in scroll_pages
            if scroll_page.widget() is not None
        ]
        if not content_pages:
            return 0

        common_content_width = max(
            max(page.minimumWidth(), page.minimumSizeHint().width())
            for page in content_pages
        )
        viewport_minimum_width = self.COMMON_TAB_VIEWPORT_MINIMUM_WIDTH

        for scroll_page in scroll_pages:
            scroll_page.setMinimumWidth(viewport_minimum_width)
            scroll_page.setProperty(
                "commonViewportMinimumWidth",
                viewport_minimum_width,
            )
        for content_page in content_pages:
            content_page.setMinimumWidth(common_content_width)
            content_page.setProperty(
                "commonContentMinimumWidth",
                common_content_width,
            )

        self.tabs.setProperty(
            "commonContentMinimumWidth",
            common_content_width,
        )
        self.tabs.setProperty(
            "commonViewportMinimumWidth",
            viewport_minimum_width,
        )
        return common_content_width
            
    def create_combined_tab(
        self,
        tools_dir,
        tab_key,
        scripts_dict,
    ):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        main_layout = QVBoxLayout(tab)
        
        combined_docstring = ""
        script_idx = 0
        
        for script_name, script_settings_def in scripts_dict.items():
            script_path = os.path.join(tools_dir, script_name)
            if not os.path.exists(script_path): continue
                
            with open(script_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            try: tree = ast.parse(source)
            except SyntaxError: continue
                
            docstring = ast.get_docstring(tree) or ""
            if not combined_docstring: combined_docstring = docstring
                
            form_widget = QFrame()
            form_widget.setObjectName("toolSectionCard")
            form_widget.setStyleSheet(SECTION_CARD_STYLE)
            layout = QFormLayout(form_widget)
            layout.setHorizontalSpacing(30)
            layout.setVerticalSpacing(12)
            
            layout.addRow(self._create_tool_header(script_name, script_path))
            
            self._populate_script_layout(
                layout,
                script_name,
                script_path,
                script_settings_def,
                source,
                tree,
            )
            main_layout.addWidget(form_widget)
            self._tool_form_layouts.append(layout)
            script_idx += 1

        main_layout.addStretch()

        pseudo_path = os.path.join(tools_dir, tab_key) + "_GUI_tab"
        self.script_data[pseudo_path] = {'inputs': {}, 'settings': [], 'docstring': combined_docstring}
        self.tab_paths.append(pseudo_path)
        
        scroll.setProperty("descriptionKey", tab_key)
        tab_name = TAB_DISPLAY_NAMES.get(tab_key, tab_key.replace("_", " "))
        self.tabs.addTab(scroll, tab_name)

    def create_script_tab(self, script_path, script_name, script_settings_def=None):
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
        except SyntaxError:
            print(f"Syntax error in {script_name}, skipping.")
            return
            
        docstring = ast.get_docstring(tree) or ""
            
        if script_name not in self.MANUAL_SETTINGS:
            return
            
        script_settings_def = self.MANUAL_SETTINGS[script_name]
        
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        main_layout = QVBoxLayout(tab)
        
        form_widget = QFrame()
        form_widget.setObjectName("toolSectionCard")
        form_widget.setStyleSheet(SECTION_CARD_STYLE)
        layout = QFormLayout(form_widget)
        layout.setHorizontalSpacing(30)
        layout.setVerticalSpacing(12)
        
        layout.addRow(self._create_tool_header(script_name, script_path))

        self._populate_script_layout(layout, script_name, script_path, script_settings_def, source, tree)
        self._tool_form_layouts.append(layout)
        
        main_layout.addWidget(form_widget)
        main_layout.addStretch() # Pushes the form strictly to the top
        
        self.tab_paths.append(script_path)
        self.script_data[script_path]['docstring'] = docstring
        
        scroll.setProperty(
            "descriptionKey",
            script_name.removesuffix(".py"),
        )
        tab_name = script_name.replace(".py", "").replace("_", " ")
        self.tabs.addTab(scroll, tab_name)

    def eventFilter(self, obj, event):
        event_type = event.type()
        routed_events = (
            QEvent.Type.FocusIn,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Enter,
            QEvent.Type.ToolTip,
        )
        if event_type in routed_events:
            tip = obj.toolTip() if isinstance(obj, QWidget) else ""
            tip = tip or self.tip_db.get(obj, "")
            if tip:
                self.tip_panel.setText(tip)
                if event_type == QEvent.Type.ToolTip:
                    return True
        return super().eventFilter(obj, event)

    def _collect_tool_settings(self, script_path):
        data = self.script_data[script_path]
        inputs = data['inputs']
        settings = data['settings']

        new_settings = {}
        for s in settings:
            var_name = s['name']
            input_data = inputs[var_name]
            widget = input_data['widget']
            w_type = input_data['type']
            
            if w_type in ["dropdown", "dropdown_from_folder", "device_dropdown"]:
                if w_type == "dropdown_from_folder":
                    val = widget.combo.currentText()
                elif w_type == "device_dropdown":
                    val = widget.currentData()
                else:
                    val = (
                        widget.currentData()
                        if widget.property("persistItemData")
                        else widget.currentText()
                    )
                    
                if w_type == "dropdown_from_folder" and s['def'].get('include_ext', False):
                    extensions = s['def']['extension']
                    if val and not val.endswith(extensions):
                        val += extensions[0] if isinstance(extensions, tuple) else extensions
                new_settings[var_name] = val
            elif w_type == "switch":
                new_settings[var_name] = widget.isChecked()
            elif w_type == "slider":
                new_settings[var_name] = int(widget.slider.value())
            elif w_type == "slider_float":
                new_settings[var_name] = float(widget.slider.value() / widget.scale)
            elif w_type == "host_cache":
                new_settings[var_name] = widget.setting_value()
            elif w_type == "negative_number":
                new_settings[var_name] = float(widget.value())
            elif w_type == "number":
                new_settings[var_name] = int(widget.value())
            elif w_type == "folder_browser":
                raw_path = widget.line_edit.text().strip()
                new_settings[var_name] = os.path.normpath(raw_path) if raw_path else ""
            else:
                new_settings[var_name] = widget.text()

        return new_settings

    def _current_directory_settings(self):
        directories = {}
        for key, line_edit in self.dir_inputs.items():
            raw_path = line_edit.text().strip()
            directories[key] = os.path.normpath(raw_path) if raw_path else ""
        return directories

    @staticmethod
    def _portable_export_directory_path(path):
        """Use portable separators for relative directories in exported JSON."""
        if not path:
            return ""
        path = os.fspath(path)
        if ntpath.isabs(path) or posixpath.isabs(path):
            return path
        return path.replace("\\", "/")

    @staticmethod
    def _normalized_export_filename(raw_name):
        name = raw_name.strip()
        if not name:
            raise ValueError("Enter a name for the exported settings file.")
        if name.lower().endswith(".json"):
            stem = name[:-5]
        else:
            stem = name
            name += ".json"
        if not stem or stem in {".", ".."}:
            raise ValueError("Enter a valid settings filename.")
        if any(character in name for character in '<>:"/\\|?*'):
            raise ValueError("The settings name cannot contain path separators or <>:\"|?*.")
        if stem[-1] in {" ", "."}:
            raise ValueError("The settings name cannot end with a space or period.")
        reserved = {"CON", "PRN", "AUX", "NUL"}
        reserved.update(f"COM{index}" for index in range(1, 10))
        reserved.update(f"LPT{index}" for index in range(1, 10))
        if stem.split(".", 1)[0].upper() in reserved:
            raise ValueError(f"'{stem}' is a reserved filename.")
        return name

    def export_settings(self, script_path):
        script_name = os.path.basename(script_path)
        suggested_name = script_name.removesuffix(".py")
        raw_name, accepted = QInputDialog.getText(
            self,
            "Export Tool Settings",
            "Settings name:",
            QLineEdit.EchoMode.Normal,
            suggested_name,
        )
        if not accepted:
            return

        try:
            filename = self._normalized_export_filename(raw_name)
            current_directories = self._current_directory_settings()
            export_directory = current_directories.get("SETTING_EXPORT_DIR", "")
            if not export_directory:
                export_directory = DEFAULT_DIRECTORY_PATHS["SETTING_EXPORT_DIR"]
            if not os.path.isabs(export_directory):
                export_directory = os.path.join(_PROJECT_ROOT, export_directory)
            export_directory = os.path.normpath(export_directory)
            target_path = os.path.join(export_directory, filename)

            if os.path.exists(target_path):
                answer = QMessageBox.question(
                    self,
                    "Replace Exported Settings?",
                    f"'{target_path}' already exists. Replace it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

            directory_keys = TOOL_DIRECTORY_KEYS.get(script_name)
            if directory_keys is None:
                raise ValueError(f"No directory export contract is registered for {script_name}.")
            tool_settings = self._collect_tool_settings(script_path)
            tool_settings = {
                key: (
                    self._portable_export_directory_path(value)
                    if key.endswith("_DIR") and isinstance(value, (str, os.PathLike))
                    else value
                )
                for key, value in tool_settings.items()
            }
            payload = {
                "DIRECTORIES": {
                    key: self._portable_export_directory_path(
                        current_directories.get(key, "")
                    )
                    for key in directory_keys
                },
                script_name: tool_settings,
            }

            os.makedirs(export_directory, exist_ok=True)
            partial_path = f"{target_path}.{os.getpid()}.partial"
            try:
                with open(partial_path, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, indent=4)
                    handle.write("\n")
                os.replace(partial_path, target_path)
            finally:
                if os.path.exists(partial_path):
                    os.unlink(partial_path)

            command = (
                f'"{sys.executable}" -u "{os.path.abspath(script_path)}" '
                f'"{os.path.abspath(target_path)}"'
            )
            QMessageBox.information(
                self,
                "Settings Exported",
                f"Settings exported to:\n{os.path.abspath(target_path)}\n\n"
                f"Command-line usage:\n{command}",
            )
        except Exception as error:
            QMessageBox.critical(self, "Export Settings Error", str(error))

    def save_and_run(self, script_path):
        import json

        # 1. Collect current values from GUI
        new_settings = self._collect_tool_settings(script_path)

        # 2. Load existing JSON to avoid overwriting unrelated settings
        settings_file = os.path.join(_PROJECT_ROOT, "Input_Files", "tools_settings.json")
        combined_settings = {}
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    combined_settings = json.load(f)
            except: pass

        # A first-run Save and Run must persist the same usable directory
        # defaults shown in the Directories tab. Existing nonblank custom paths
        # remain authoritative.
        fill_missing_directory_defaults(combined_settings)

        script_name = os.path.basename(script_path)
        selected_model = None
        if script_name == "Generate_Embeddings.py":
            selected_model = new_settings.get("MODEL_NAME")
        elif (
            script_name == "Embedding_PWA.py"
            and new_settings.get("MANUAL_REF_SEQ")
            and new_settings.get("MANUAL_TAR_SEQ")
        ):
            selected_model = new_settings.get("EMBEDDING_MODEL")
        if selected_model:
            usage_terms = get_embedding_model_usage_terms().get(selected_model)
            if (
                usage_terms
                and usage_terms.get("requires_acknowledgement", False)
                and not confirm_model_usage_terms(
                    self,
                    selected_model,
                    usage_terms,
                )
            ):
                return

        if script_name == "Generate_Embeddings.py":
            execution_modes = get_embedding_model_execution_modes()
            execution_mode = execution_modes.get(new_settings.get("MODEL_NAME"))
            if execution_mode == "remote_api":
                previous_device = combined_settings.get(script_name, {}).get(
                    "DEVICE_SELECTION"
                )
                if previous_device is None:
                    new_settings.pop("DEVICE_SELECTION", None)
                else:
                    new_settings["DEVICE_SELECTION"] = previous_device
            else:
                try:
                    Hardware_Utils.resolve_device_selection(
                        new_settings.get("DEVICE_SELECTION", "auto")
                    )
                except ValueError as error:
                    QMessageBox.critical(
                        self, "Invalid Hardware Selection", str(error)
                    )
                    return

        if script_name in {
            "Align_Similarity_Matrix.py",
            "Network_Injection.py",
            "Embedding_SSEARCH.py",
        }:
            available_devices = Hardware_Utils.get_available_devices()
            try:
                Hardware_Utils.resolve_device_selection(
                    new_settings.get("DEVICE_SELECTION", "auto"),
                    available_devices,
                )
            except ValueError as error:
                QMessageBox.critical(self, "Invalid Hardware Selection", str(error))
                return
            if script_name in {
                "Align_Similarity_Matrix.py", "Embedding_SSEARCH.py"
            }:
                precision = str(
                    new_settings.get("ACCELERATOR_PRECISION", "auto")
                ).strip().lower()
                if precision == "tf32" and not _selection_supports_tf32(
                    new_settings.get("DEVICE_SELECTION", "auto"),
                    available_devices,
                ):
                    precision = "auto"
                    new_settings["ACCELERATOR_PRECISION"] = "auto"
                if precision not in {"auto", "float32", "tf32"}:
                    QMessageBox.critical(
                        self,
                        "Invalid Accelerator Precision",
                        "Precision must be auto, float32, or tf32.",
                    )
                    return
            if script_name in {
                "Align_Similarity_Matrix.py", "Network_Injection.py"
            }:
                try:
                    execution_mode = normalize_execution_mode(
                        new_settings.get("EXECUTION_MODE", "auto")
                    )
                except ValueError as error:
                    QMessageBox.critical(
                        self, "Invalid Execution Mode", str(error)
                    )
                    return
                if execution_mode == "tiled":
                    available = Hardware_Utils.get_available_devices()
                    selected = Hardware_Utils.resolve_device_selection(
                        new_settings.get("DEVICE_SELECTION", "auto"), available
                    )
                    eligible_backends = (
                        {"cuda"}
                        if script_name == "Align_Similarity_Matrix.py"
                        else {"cuda", "xpu"}
                    )
                    eligible = [
                        candidate for candidate in available
                        if candidate.backend in eligible_backends
                        and tiled_accelerator_support(
                            candidate.device, require_memory=False
                        )[0]
                    ]
                    if selected is not None:
                        eligible = (
                            [selected]
                            if selected.backend in eligible_backends
                            and tiled_accelerator_support(
                                selected.device, require_memory=False
                            )[0]
                            else []
                        )
                    if not eligible:
                        QMessageBox.critical(
                            self,
                            "Invalid Execution Mode",
                            (
                                "Tiled alignment requires an available "
                                "CUDA/ROCm accelerator."
                                if script_name == "Align_Similarity_Matrix.py"
                                else "Tiled execution requires an available "
                                "CUDA/ROCm or XPU accelerator."
                            ),
                        )
                        return
                host_cache = str(new_settings.get("HOST_CACHE_GB", "auto")).strip()
                if host_cache.lower() != "auto":
                    try:
                        host_cache_value = float(host_cache)
                    except ValueError:
                        host_cache_value = -1.0
                    if not math.isfinite(host_cache_value) or host_cache_value < 0:
                        QMessageBox.critical(
                            self,
                            "Invalid Host Cache",
                            "Host Cache must be 'auto' or a non-negative GiB value.",
                        )
                        return

        # 3. Replace and save only the selected script's complete settings section.
        # This removes stale keys that are no longer represented in the GUI while
        # preserving DIRECTORIES and every other script section.
        combined_settings[script_name] = new_settings
        
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(combined_settings, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save JSON settings:\n{e}")
            return
            
        # 4. Run the script
        try:
            # Resolve script path, folder directory, and name to absolute values to avoid execution context errors
            abs_script_path = os.path.abspath(script_path)
            script_dir = os.path.dirname(abs_script_path)
            script_name = os.path.basename(abs_script_path)
            
            print(f"Executing: {script_name} in {script_dir}")
            
            launch_in_terminal(
                [sys.executable, "-u", abs_script_path],
                cwd=script_dir,
                hold=HoldMode.ALWAYS,
                title=script_name,
            )
            
            QMessageBox.information(self, "Success", f"Saved configuration to JSON and launched {script_name}.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run {script_path}:\n{e}")

if __name__ == "__main__":
    existing_qt_application = QApplication.instance()
    app = existing_qt_application or QApplication(sys.argv)
    single_instance = None
    if existing_qt_application is None:
        single_instance = SingleInstanceController("SSN_Tools", app)
        try:
            is_primary_instance = single_instance.acquire_or_notify()
        except RuntimeError as error:
            QMessageBox.critical(None, "SSN Tools Startup Error", str(error))
            raise SystemExit(1)
        if not is_primary_instance:
            raise SystemExit(0)
        app.aboutToQuit.connect(single_instance.close)

    configure_linux_qt_desktop_identity(app, TOOLS_DESKTOP_FILE_NAME)
    def _exit_on_uncaught_exception(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        app.exit(1)

    sys.excepthook = _exit_on_uncaught_exception
    try:
        configure_qt_application_fonts(app)
    except Exception as e:
        print(f"Warning: Could not configure bundled application fonts: {e}")

    # Report a missing QtWebEngine before building the GUI, since ToolsGUI
    # constructs a ResponsiveTextBrowser. Print as well as show a dialog: if
    # the Qt platform plugin itself is broken no window can appear, and the
    # launchers run in a terminal where the printed copy is still readable.
    if QTWEBENGINE_IMPORT_ERROR is not None:
        message = QTWEBENGINE_MISSING_MESSAGE.format(error=QTWEBENGINE_IMPORT_ERROR)
        print(message, file=sys.stderr)
        try:
            QMessageBox.critical(None, "Missing QtWebEngine libraries", message)
        except Exception:
            pass
        sys.exit(1)

    # Set Application-wide Icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "tool_logo.ico")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "logos", "tool_logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    try:
        force_light_palette(app)
    except Exception as e:
        print(f"Warning: Could not force light palette: {e}")
        app.setStyle("Fusion")
    window = ToolsGUI()
    if single_instance is not None:
        single_instance.set_activation_callback(
            lambda active_window=window: show_window_in_front(active_window)
        )
    show_window_in_front(window)
    sys.exit(app.exec())
