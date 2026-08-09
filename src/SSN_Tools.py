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
import sys
import os
import ast
import json
import subprocess
import markdown
import re

from utilities import Hardware_Utils
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
from Cache_Manifest import (
    file_cache_key,
    inspect_network_completeness,
    validate_network_schema,
)

MAX_CORES = os.cpu_count() or 16

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
SECONDARY_TITLE_STYLE = (
    "font-weight: bold; font-size: 15px; margin-bottom: 5px; "
    "border-bottom: 1px solid #95A5A6; padding-bottom: 2px;"
)
SECONDARY_TITLE_WITH_TOP_PADDING_STYLE = (
    SECONDARY_TITLE_STYLE + " padding-top: 18px;"
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
    ],
    "Align_Substitution_Matrix.py": [
        ("BATCH_SIZE", "NUM_THREADS"),
    ],
    "Network_Injection.py": [
        ("BATCH_SIZE", "WORKERS"),
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
    ],
}
INLINE_FIELD_GROUPS = {
    "Align_Similarity_Matrix.py": [
        ("INPUT_HDF5", "DEVICE_SELECTION"),
    ],
    "Align_Substitution_Matrix.py": [
        ("INPUT_FASTA", "MATRIX"),
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
                             QSizePolicy, QFrame)
from PySide6.QtCore import Qt

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

# --- Re-enabled Qt log filter for window state transitions ---
import Qt_Log_Filter
Qt_Log_Filter.install()

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
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
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
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 85%;
            background-color: #f6f8fa;
            padding: 2px 4px;
            border-radius: 4px;
            color: #1f2328;
        }
        pre {
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
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
        """
        
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
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
                "INPUT_FASTA": "Sequence Set (.fasta): The raw sequence file to be cleaned. Sanitization standardizes characters to uppercase, strips trailing/leading invalid symbols, and filters out non-standard elements.",
                "ENABLE_LENGTH_FILTER": "Enable Length Filter: Toggle to filter sequences based on their amino acid length. When enabled, only sequences within the specified minimum and maximum length bounds will be retained.",
                "OVER_WRITE": "Overwrite Original File: If enabled, the sanitized sequences will overwrite the input file. If disabled, a new file named <input_name>_sanitized.fasta will be created to preserve the original raw file.",
                "REMOVE_BY_HEADER_STRING": "Remove Header Substring: If specified, sequences with headers containing this exact case-sensitive text will be discarded. Leave empty to disable filtering.",
                "MIN_SEQ_LENGTH": "Minimum Sequence Length: The lower limit (inclusive) for filtering sequences by length. Sequences shorter than this number of residues will be discarded during sanitization.",
                "MAX_SEQ_LENGTH": "Maximum Sequence Length: The upper limit (inclusive) for filtering sequences by length. Sequences longer than this number of residues will be discarded to remove outliers."
            },
            "Generate_Embeddings.py": {
                "INPUT_FASTA": "Sequence Set (.fasta): The sanitized input sequence file to generate embeddings for. Each sequence is parsed and fed through the neural network to produce high-dimensional dense representations.",
                "MODEL_NAME": "Model Name: The protein language model (pLM) used to calculate sequence embeddings and label the output filename. ESMC 300M/600M run locally; esmc_6b maps internally to Biohub's esmc-6b-2024-12 API model and reads ESM_API_TOKEN from src/resources/pLM_models/esmc_6b_api_key.json. Rostlab models (prot_bert/prost_t5) are also supported. All model identifiers are lower case.",
                "SAVING_MODE": "Saving Mode: The floating-point precision for storing embedding tensors in the HDF5 file. Float16 is highly recommended to save up to 50% disk space and RAM, while float32 retains full uncompressed precision.",
                "DEVICE_SELECTION": "Device: Auto Benchmark compares CPU and every available local accelerator using representative sequences. Remote API models do not use this setting."
            },
            "Embedding_Cropping.py": {
                "INPUT_EMBED": "Full Embedding Set (.h5): The pre-computed embedding database for the full-length sequences, generated by Generate_Embeddings.py. Cropped embeddings are sliced directly out of these arrays.",
                "CROPPED_FASTA": "Cropped Sequence Set (.fasta): The partial sequences to produce contextual embeddings for. Headers and sequences are sanitized exactly as in embedding generation; full sequences are read from the embedding file metadata."
            },
            "Align_Similarity_Matrix.py": {
                "INPUT_HDF5": "Embedding Set (.h5): The HDF5 database containing dense embedding vectors for each sequence in the network. These vectors are used to compute residue-level alignment scores.",
                "EDGE_PREFILTERING": "Edge Prefiltering: Pre-filter sequence pairs by evaluating the cosine similarity of their global mean embedding vectors. This avoids running full alignments on highly dissimilar pairs, saving computation.",
                "PREFILTER_STRENGTH": "Strength (%): The percentage of candidate edges with the lowest cosine similarity to discard. Higher percentages speed up calculations by performing sequence alignment on only the most promising pairs.",
                "WORKERS": "CPU Workers: The number of CPU threads allocated for parallel processing. Running with more threads speeds up the alignment of large embedding matrices by distributing pairs across multiple cores.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: The penalty score applied for initiating or extending gaps in local alignment. More negative values enforce stricter local alignments with fewer gaps.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: The penalty score applied for initiating or extending gaps in global alignment. Adjust this to control how alignment length matches are forced.",
                "BATCH_SIZE": "Batch Size: The number of sequence pairs processed in a single chunk. Larger values maximize CPU utilization but require more system memory. Set to 'auto' or specify a number.",
                "DEVICE_SELECTION": "Device: Auto Benchmark compares the complete CPU pipeline with every available accelerator. Select a concrete device to bypass device comparison.",
            },
            "Align_Substitution_Matrix.py": {
                "INPUT_FASTA": "Sequence Set (.fasta): The sequence file to align with BLASTP. Before alignment, records undergo the same canonical header, residue, empty-record, and duplicate-sequence sanitization used by Generate Embeddings.",
                "MATRIX": "Substitution Matrix: The amino acid substitution matrix (e.g., BLOSUM62, PAM250) used to score matches/mismatches during pairwise alignment. Select based on the evolutionary distance of the sequences.",
                "NUM_THREADS": "CPU Workers: The number of CPU threads allocated for parallel sequence alignments. Increasing threads speeds up computations on multi-core systems.",
                "BATCH_SIZE": "Batch Size: The number of sequence pairs aligned per block. Tuning this controls memory consumption and parallel execution batch sizes.",
                "BLASTP_DIR": "BLASTP Directory: The folder containing your local blastp and makeblastdb binaries (usually named 'bin'). If left blank, standard system PATH directories are searched."
            },
            "Embedding_MSA.py": {
                "USE_SEQUENCE_FILTER": "Use Sequence Filter: Toggle whether to filter the alignment by an explicit FASTA sequence set. If OFF (default), the sequence set field is blanked out and alignment uses all sequences in the embedding database.",
                "INPUT_FASTA": "Sequence Set (.fasta): The raw sequence file to be aligned. These letters are aligned, padded with gaps, and output as the final Multiple Sequence Alignment (MSA).",
                "INPUT_EMBED": "Embedding Set (.h5): The HDF5 database containing dense sequence embedding tensors. These embeddings drive the progressive profile alignments along the guide tree nodes.",
                "INPUT_NETWORK": "Network File (.h5): The pairwise similarity network used to build the evolutionary guide tree. For sparse networks, missing edge scores are predicted using regression.",
                "SHOW_REGRESSION_PLOT": "Show Isotonic Regression Plot: Toggle whether to display a regression plot when a sparse network is loaded. This visualizes the fit between embedding distances and pairwise connectivity.",
                "TREE_METHOD": "Tree Building Method: The method used to construct the guide tree. 'UPGMA' groups by average proximity. 'Neighbor-joining' adjusts for rate variations (slower but more biologically standard).",
                "ALIGNMENT_SCORE": "Score Mode: Specifies whether to weight guide tree branches based on 'global' or 'local' connectivity scores. This determines the progressive alignment order.",
                "NORMALIZATION_MODE": "Normalization Mode: Normalization method for pairwise embedding scores (e.g., alignment length, shorter sequence length). Not active when using raw BLAST E-values.",
                "BOOTSTRAP_TREE": "Noise-Perturbed Consensus Guide Tree: Toggle whether to average guide-tree distances across randomly perturbed replicate trees (ON) or build one deterministic tree (OFF). This is a sensitivity ensemble, not classical bootstrap support. Disabling it significantly speeds up the run.",
                "NUM_TREES": "Number of Perturbed Trees: The number of noise-perturbed replicate trees used to construct the consensus guide tree. Higher values produce a more stable average but increase calculation time.",
                "INCLUDE_IMPUTED_PAIRS_IN_CONSENSUS": "Include Imputed Pairs in Final Consensus: For an incomplete network, OFF retains baseline imputed distances for missing pairs after they participate in every replicate tree; ON replaces them with replicate-averaged cophenetic distances.",
                "NOISE_SCALE": "Normalized Additive Noise Scale: Gaussian standard deviation expressed as a fraction of the valid distance range. For example, 0.02 means 2% of the maximum guide-tree distance. Every observed and regression-imputed distance is perturbed, then clamped between zero (closest) and the maximum distance (weakest or unconnected).",
                "GAP_OPEN": "Gap Open Penalty: The penalty score applied for initiating a new gap within progressive profile alignments. More negative values result in fewer gaps.",
                "GAP_EXTEND": "Gap Extend Penalty: The penalty score applied for extending an existing gap. More negative values yield shorter, more compact gap regions.",
                "WORKERS": "CPU Workers: The number of CPU processes allocated to parallel noise-perturbed guide-tree replicates.",
                "SAFE_TEMP_DIR": "Temporary Working Directory: The directory for caching intermediate files and memory-mapped matrices. Ensures that massive guide tree calculations do not overflow RAM."
            },
            "Sparse_MSA_Converter.py": {
                "CONVERT_ALL": "Convert All Alignments: If ON, converts all standard MSAs in the input folder to the space-saving sparse format. If OFF, only the selected MSA file is converted.",
                "INPUT_FASTA": "Input MSA (.fasta): The standard multiple sequence alignment file to convert. The script extracts consensus positions to save files in the sparse representation."
            },
            "Parse_BLAST_Output.py": {
                "INPUT_BLAST_TABULAR": "BLAST Results (.tabular): The outfmt 6 formatted BLAST text file to parse. The script extracts e-values, alignment identities, and headers to construct a compatible HDF5 network file."
            },
            "Embedding_Injection.py": {
                "INPUT_EMBED": "Input Embedding Set (.h5): The HDF5 embedding database to receive the injected sequences and metadata. This updates files with correct headers.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): The fasta file containing sequences to be injected into the embedding file. Re-aligns sequence indexes and updates corresponding metadata."
            },
            "Embedding_Extraction.py": {
                "INPUT_EMBED": "Input Embedding Set (.h5): The source HDF5 embedding file from which subset embeddings will be extracted based on matching sequence headers.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): The fasta file defining the subset of sequences to extract. Only embeddings matching these headers will be written to the output."
            },
            "Network_Injection.py": {
                "OLD_NETWORK": "Input Network Edges (.h5): The pre-existing HDF5 network file. The script will inject newly calculated embedding alignments into this file to expand its edge details.",
                "NEW_EMBEDDINGS": "Input Embedding Set (.h5): The HDF5 embedding set containing the dense representations to align and inject into the targeted network file.",
                "WORKERS": "CPU Workers: The number of CPU threads allocated for parallel embedding alignment calculation and network writing.",
                "BATCH_SIZE": "Batch Size: The number of sequence alignments calculated per write block. Tuning this controls memory consumption and optimizes file write performance."
            },
            "Network_Extraction.py": {
                "INPUT_NET": "Input Network Edges (.h5): The source network file containing pairwise connectivity data from which a subset will be extracted.",
                "INPUT_FASTA": "Input Sequence Set (.fasta): The FASTA file defining the subset of sequences. Only network edges between these sequences will be extracted."
            },
            "Embedding_PWA.py": {
                "INPUT_EMBED": "Embedding Set (.h5): A metadata-first HDF5 database containing sanitized headers, sequences, model metadata, and residue-level embeddings. It supplies stored sequences whenever a manual-sequence switch is OFF.",
                "REF_HEADER": "Reference Header: The header of the reference sequence in the embedding database. Typed text is canonically sanitized before lookup. If left empty, the first stored sequence is used.",
                "MANUAL_REF_SEQ": "Manual Ref Seq: Enable the optional reference-sequence field. When OFF, the reference is loaded from the selected embedding set by header.",
                "REF_SEQUENCE": "Ref Sequence (Optional): Manually provide an amino acid sequence for the reference. This field is used only while Manual Ref Seq is ON and is canonically sanitized before embedding generation.",
                "TAR_HEADER": "Target Header: The header of the target sequence in the embedding database. Typed text is canonically sanitized before lookup. If left empty, the second stored sequence is used.",
                "MANUAL_TAR_SEQ": "Manual Tar Seq: Enable the optional target-sequence field. When OFF, the target is loaded from the selected embedding set by header.",
                "TAR_SEQUENCE": "Tar Sequence (Optional): Manually provide an amino acid sequence for the target. This field is used only while Manual Tar Seq is ON and is canonically sanitized before embedding generation.",
                "HIGHLIGHT_POSITIONS": "Highlight Pos (e.g., 1, 4-6): A comma-separated list of 1-indexed residue positions or ranges to highlight in the alignment visualization.",
                "EMBEDDING_MODEL": "Embedding Model: Protein language model used to generate embeddings when both sequences are entered manually. The choices are discovered from src/resources/pLM_models. When either sequence comes from the embedding set, its stored model is used instead.",
                "ALIGNMENT_MODE": "Alignment Mode: Select whether to compute a global (Needleman-Wunsch) or local (Smith-Waterman) alignment based on embedding similarities.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: Gap penalty applied when using local alignment. Adjusts the frequency and size of gap insertions within local alignments.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: Gap penalty applied when using global alignment. Adjusts the frequency and size of gap insertions across entire sequences.",
                "GENERATE_REPORT": "Generate Report: Toggle whether to save the pairwise alignment visualization and score into a color-coded HTML report in the report directory."
            },
            "Embedding_SSEARCH.py": {
                "INPUT_EMBED": "Embedding Set (.h5): A complete metadata-first HDF5 database containing sanitized headers, sequences, and pre-computed tensors for database search.",
                "QUERY_HEADER": "Query Header: The header of a sequence stored in the embedding database to use as the query. It is sanitized before lookup unless Manual Query Seq is enabled.",
                "MANUAL_QUERY_SEQ": "Manual Query Seq: Enable the optional raw query-sequence field. When OFF, the query is loaded from the embedding database by header.",
                "QUERY_SEQUENCE": "Query Sequence (Optional): A raw amino acid sequence used only while Manual Query Seq is ON.",
                "OUTPUT_NAME": "Output Name: Custom prefix for search output files. If left blank, the query header (sanitized) is used as the default name.",
                "TOP_K": "Top K Hits: The maximum number of highest-scoring database hits to include in the output results. Set to control list size.",
                "NORM_THRESHOLD": "Norm Score Cutoff: The minimum normalized similarity score threshold for hits. Sequences scoring below this are excluded.",
                "ALIGNMENT_MODE": "Alignment Mode: Whether to perform global or local alignment when scanning database sequence embeddings against the query.",
                "NORM_MODE": "Normalization Mode: Length-normalization formula for alignment scores to prevent bias toward longer or shorter alignments.",
                "LOCAL_GAP_P": "Local Align Gap Penalty: Gap penalty applied when using local alignment (Smith-Waterman) to scan database embeddings.",
                "GLOBAL_GAP_P": "Global Align Gap Penalty: Gap penalty applied when using global alignment (Needleman-Wunsch) to scan database embeddings.",
                "WORKERS": "CPU Workers: The number of CPU threads allocated for parallel scanning. Running with more threads reduces search time.",
                "GENERATE_FASTA": "Generate FASTA File: Toggle whether to generate a FASTA file containing all the top hit sequences aligned with the query."
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
                "hide_secondary_titles": True,
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
                            "extension": ".fasta",
                            "include_ext": True,
                            "dir_key": "FASTA_DIR",
                            "display": "Sequence Set (.fasta):"
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
                            "extension": ".tabular",
                            "include_ext": True,
                            "dir_key": "NETWORK_DIR",
                            "display": "BLAST Results (.tabular):"
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
        
        self.tip_panel = QLabel("Hover or focus on an input to see its description.")
        self.tip_panel.setWordWrap(True)
        self.tip_panel.setMinimumHeight(20)
        self.tip_panel.setStyleSheet("color: #444; font-style: italic; background-color: #e8eaed; padding: 10px; border-radius: 5px;")
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

        self.tip_db = {}
        self.network_completeness_cache = {}
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        self.load_tools()
        self.create_directories_tab()
    
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
        header_layout.setSpacing(12)

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

        header_layout.addWidget(
            btn_save,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        header_layout.addWidget(desc_label, 1)
        layout.addRow(header)
        
        self.dir_inputs = {}
        dir_defaults = {
            "EMBED_DIR": os.path.join("Embeddings"),
            "FASTA_DIR": os.path.join("Input_Files","Sequence_Sets"),
            "MSA_DIR": os.path.join("Input_Files","Multiple_Alignments"),
            "NETWORK_DIR": os.path.join("Input_Files","Networks_EValues"),
            "PATH_DIR": os.path.join("Cache_Files","Global_Path"),
            "REPORT_DIR": os.path.join("Cache_Files","Align_Report")
        }
        
        # Load existing paths from JSON if available
        import json
        settings_file = os.path.join("Input_Files", "tools_settings.json")
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    j_data = json.load(f)
                    if "DIRECTORIES" in j_data:
                        dir_defaults.update(j_data["DIRECTORIES"])
            except: pass
            
        dir_tips = {
            "FASTA_DIR": "Directory containing unaligned sequence sets (.fasta).",
            "MSA_DIR": "Directory containing multiple sequence alignments (.fasta, .pkl).",
            "EMBED_DIR": "Directory containing language model embeddings (.h5).",
            "NETWORK_DIR": "Directory containing SSN edge networks and E-value matrices (.h5).",
            "PATH_DIR": "Directory for caching global paths (.h5).",
            "REPORT_DIR": "Directory for storing alignment reports and generated files."
        }
        
        for key, current_val in dir_defaults.items():
            ui_element = QWidget()
            h_lay = QHBoxLayout(ui_element)
            h_lay.setContentsMargins(0, 0, 0, 0)
            
            clean_val_str = str(current_val).replace('r"', '"').replace("r'", "'").strip("\"'")
            le = QLineEdit(clean_val_str)
            btn = QPushButton("Browse...")
            
            def open_folder_dialog(checked=False, line_edit=le):
                folder = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() if line_edit.text() else "")
                if folder:
                    import os
                    line_edit.setText(os.path.normpath(folder))
                    
            btn.clicked.connect(open_folder_dialog)
            h_lay.addWidget(le)
            h_lay.addWidget(btn)
            
            display_name = key.replace('_', ' ').title()
            display_name = display_name.replace('Msa', 'MSA').replace('Dir', 'Directory')
            display_name = display_name.replace('Fasta', 'FASTA')
            display_name = display_name.replace('Embed', 'Embedding')
            display_name = display_name.replace('Path Directory', 'Alignment Path Directory')
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
            
        settings_file = os.path.join("Input_Files", "tools_settings.json")
        combined_settings = {}
        os.makedirs("Input_Files", exist_ok=True)
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
                    show_secondary_titles=not settings_def.get(
                        "hide_secondary_titles",
                        False,
                    ),
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
        show_secondary_titles=True,
    ):
        defined_vars = {item["var_name"]: item for item in script_settings_def}
        section_title_count = sum(item["type"] == "title" for item in script_settings_def)
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
                        settings_path = os.path.join("Input_Files", "tools_settings.json")
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
        section_title_index = 0
        for s_def in script_settings_def:
            if s_def['type'] == "title":
                if section_title_count > 1 and show_secondary_titles:
                    title_lbl = QLabel(s_def['display'])
                    title_style = (
                        SECONDARY_TITLE_STYLE
                        if section_title_index == 0
                        else SECONDARY_TITLE_WITH_TOP_PADDING_STYLE
                    )
                    title_lbl.setStyleSheet(title_style)
                    layout.addRow(title_lbl)
                section_title_index += 1
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
                if s_def.get("model_license_labels", False):
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
        header_layout.setSpacing(12)

        tool_title = self.tool_titles.get(
            script_name,
            script_name.removesuffix(".py").replace("_", " "),
        )
        title_label = QLabel(tool_title)
        title_label.setObjectName("toolTitle")
        title_label.setStyleSheet(PRIMARY_TITLE_STYLE)

        btn_run = QPushButton("Save && Run")
        btn_run.setObjectName("saveRunButton")
        btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 10px 16px;"
        )
        btn_run.clicked.connect(
            lambda checked, sp=script_path: self.save_and_run(sp)
        )

        header_layout.addWidget(
            btn_run,
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
            
    def create_combined_tab(
        self,
        tools_dir,
        tab_key,
        scripts_dict,
        show_secondary_titles=True,
    ):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        
        main_layout = QVBoxLayout(tab)
        
        combined_docstring = ""
        script_idx = 0
        script_form_layouts = []
        
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
                show_secondary_titles=show_secondary_titles,
            )
            main_layout.addWidget(form_widget)
            script_form_layouts.append(layout)
            script_idx += 1

        self._align_form_label_columns(script_form_layouts)
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
        self._align_form_label_columns([layout])
        
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
        from PySide6.QtCore import QEvent
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress, QEvent.Type.Enter):
            tip = self.tip_db.get(obj, None)
            if tip:
                self.tip_panel.setText(tip)
        return super().eventFilter(obj, event)

    def save_and_run(self, script_path):
        import json
        data = self.script_data[script_path]
        inputs = data['inputs']
        settings = data['settings']
        
        # 1. Collect current values from GUI
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
                    if val and not val.endswith(s['def']['extension']):
                        val += s['def']['extension']
                new_settings[var_name] = val
            elif w_type == "switch":
                new_settings[var_name] = widget.isChecked()
            elif w_type == "slider":
                new_settings[var_name] = int(widget.slider.value())
            elif w_type == "slider_float":
                new_settings[var_name] = float(widget.slider.value() / widget.scale)
            elif w_type == "negative_number":
                new_settings[var_name] = float(widget.value())
            elif w_type == "number":
                new_settings[var_name] = int(widget.value())
            elif w_type == "folder_browser":
                raw_path = widget.line_edit.text().strip()
                new_settings[var_name] = os.path.normpath(raw_path) if raw_path else ""
            else:
                new_settings[var_name] = widget.text()
                
        # 2. Load existing JSON to avoid overwriting unrelated settings
        settings_file = os.path.join("Input_Files", "tools_settings.json")
        combined_settings = {}
        os.makedirs("Input_Files", exist_ok=True)
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    combined_settings = json.load(f)
            except: pass

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

        if script_name == "Align_Similarity_Matrix.py":
            try:
                Hardware_Utils.resolve_device_selection(
                    new_settings.get("DEVICE_SELECTION", "auto")
                )
            except ValueError as error:
                QMessageBox.critical(self, "Invalid Hardware Selection", str(error))
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
            
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_CONSOLE
                subprocess.Popen(
                    ["cmd.exe", "/k", sys.executable, "-u", script_name],
                    creationflags=creationflags,
                    cwd=script_dir
                )
            elif sys.platform == "darwin":
                # macOS: AppleScript to activate Terminal.app and execute the script in a new window/tab.
                # Running terminal commands interactively naturally leaves the session open at the end.
                escaped_dir = script_dir.replace('"', '\\"')
                cmd_str = f'cd "{escaped_dir}" && "{sys.executable}" -u "{script_name}"'
                escaped_cmd = cmd_str.replace('"', '\\"')
                
                subprocess.Popen([
                    "osascript",
                    "-e", 'tell application "Terminal"',
                    "-e", 'activate',
                    "-e", f'do script "{escaped_cmd}"',
                    "-e", 'end tell'
                ])
            else:
                # Linux: Detect available terminal emulator and run command
                import shutil
                terminals = [
                    "gnome-terminal", "konsole", "xfce4-terminal", 
                    "mate-terminal", "lxterminal", "kitty", 
                    "alacritty", "xterm", "x-terminal-emulator"
                ]
                chosen_terminal = None
                for term in terminals:
                    if shutil.which(term):
                        chosen_terminal = term
                        break
                
                # Change directory explicitly inside the shell command to ensure it runs from the correct directory
                escaped_dir = script_dir.replace('"', '\\"')
                cmd_str = f'cd "{escaped_dir}" && "{sys.executable}" -u "{script_name}"; exec bash'
                
                if chosen_terminal:
                    if chosen_terminal in ["gnome-terminal", "kitty", "alacritty"]:
                        subprocess.Popen([chosen_terminal, "--", "bash", "-c", cmd_str], cwd=script_dir)
                    elif chosen_terminal == "konsole":
                        subprocess.Popen(["konsole", "--hold", "-e", "bash", "-c", cmd_str], cwd=script_dir)
                    else:
                        # Fallback for terminals that support the -e option with a single string command
                        subprocess.Popen([chosen_terminal, "-e", f"bash -c '{cmd_str}'"], cwd=script_dir)
                else:
                    # If absolutely no terminal emulator is found, run in the background as a fallback
                    subprocess.Popen([sys.executable, "-u", script_name], cwd=script_dir)
                    QMessageBox.warning(
                        self, "No Terminal Emulator Found",
                        "Could not locate a terminal emulator (e.g. gnome-terminal, xterm). "
                        "The script has been launched in the background, but console progress output will not be visible."
                    )
            
            QMessageBox.information(self, "Success", f"Saved configuration to JSON and launched {script_name}.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run {script_path}:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

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
        from SSN_Utils import force_light_palette
        force_light_palette(app)
    except Exception as e:
        print(f"Warning: Could not force light palette: {e}")
        app.setStyle("Fusion")
    window = ToolsGUI()
    window.show()
    sys.exit(app.exec())
