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

"""
File: Sanitize_Sequences.py
===================================
Description:
Welcome to the Sequence Sanitizer! This script is designed to clean, standardize, and filter 
protein sequence datasets in FASTA format. It ensures your data is perfectly formatted and free 
of redundant duplicates before running complex downstream tasks.

What this script does step-by-step:

1. Header Cleaning: 
   - Replaces problematic characters (like brackets, quotes, and slashes) with safe 
     alternatives (parentheses or underscores) to prevent tool or file system errors.
   - Removes leading and trailing whitespace, collapses repeated whitespace, and
     replaces the remaining spaces with underscores.

2. Sequence Cleaning:
   - Converts all amino acid letters to UPPERCASE.
   - Strips away any invalid characters, numbers, or formatting artifacts from the 
     very beginning and very end of the sequence.
   - Replaces any invalid characters or gaps (like '-') found *inside* the sequence 
     with the standard 'X' mask token. This keeps the sequence length intact.

3. Smart Deduplication & Conflict Resolution:
   - Exact Duplicates: If multiple entries have the exact same header AND sequence, 
     it keeps only one copy.
    - Merging (Same Sequence, Different Headers): If the exact same sequence appears 
      multiple times under different names, it keeps only one copy of the sequence and 
      assigns it the longest, most descriptive header name from the group.
   - Renaming (Different Sequences, Same Header): If different sequences share the 
     exact same name, it prevents data loss by keeping all sequences and renaming the 
     headers (e.g., Header_1, Header_2).

4. Length Filtering (Optional):
   - Removes sequences that fall outside of a specified length range.

5. Outputs:
   - Saves a new, clean `.fasta` file.
   - Prints a highly detailed diagnostic report to your console showing exactly what
     was changed, removed, or merged.
   - Prints the length statistics of the kept sequences: their number and the
     minimum, median, mean, standard deviation and maximum length.
   - Displays a pop-up length histogram in the Tools window's theme colour and
     language, or prints the same bins and counts as a table when invoked with a
     JSON settings file (including MCP runs). Setting SHOW_LENGTH_HISTOGRAM to
     False skips both. Everything printed stays English.

Input:
- A raw text-based FASTA file (`INPUT_FASTA`).

Output:
- A new sanitized FASTA file containing the cleaned sequences (`OUTPUT_FASTA`).
"""

import os
try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap
import tempfile
from tqdm import tqdm
from collections import Counter
import numpy as np

from utilities.Localization import Message, set_translator
from utilities.Sequence_Utils import (
    allocate_unique_headers,
    read_fasta,
    sanitize_header,
    sanitize_sequence,
    select_preferred_header,
)
from tools.tool_helpers.Tool_Pipeline import (
    inherited_settings_path,
    load_tool_settings,
    project_directory_defaults,
    select_settings_path,
)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FASTA = None  
OVER_WRITE = False
ENABLE_LENGTH_FILTER = False
MIN_SEQ_LENGTH = 0
MAX_SEQ_LENGTH = 0
REMOVE_BY_HEADER_STRING = ""
SHOW_LENGTH_HISTOGRAM = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]

# --- JSON Settings Override ---
import json
import ast

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "tools_settings.json")

if __name__ != "__main__" and os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            all_settings = json.load(f)
            
            # 1. Load GLOBAL directories and convert relative paths to absolute paths
            if "DIRECTORIES" in all_settings:
                for k, v in all_settings["DIRECTORIES"].items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        # Expand relative paths dynamically based on the current PC
                        if not os.path.isabs(str(v)):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, str(v)))
                        globals()[k] = v
                        
            # 2. Load script-specific settings
            script_name = os.path.basename(__file__)
            if script_name in all_settings:
                user_settings = all_settings[script_name]
                for k, v in user_settings.items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        orig = globals()[k]
                        
                        # Type casting to match the original Python variable type
                        if isinstance(orig, int) and not isinstance(orig, bool):
                            try: v = int(v)
                            except: pass
                        elif isinstance(orig, float):
                            try: v = float(v)
                            except: pass
                        elif isinstance(orig, list):
                            try: v = ast.literal_eval(v) if isinstance(v, str) else v
                            except: pass
                        elif orig is None:
                            if v == "None": v = None
                            elif str(v).replace('.', '', 1).isdigit():
                                v = float(v) if '.' in str(v) else int(v)
                                
                        # Convert any script-specific directory paths to absolute paths
                        if isinstance(v, str) and k.endswith("_DIR") and not os.path.isabs(v):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, v))
                            
                        globals()[k] = v
    except Exception as e:
        print(f"Failed to load user settings: {e}")

FULL_INPUT_FASTA = None
SEQUENCE_SET = ""
OUTPUT_FASTA = None


def configure_runtime_paths():
    """Resolve selected FASTA paths immediately before sanitization."""
    global FULL_INPUT_FASTA, SEQUENCE_SET, OUTPUT_FASTA

    if FASTA_DIR is None or not str(FASTA_DIR).strip():
        raise ValueError("No FASTA output directory has been configured.")
    if INPUT_FASTA is None or not str(INPUT_FASTA).strip():
        raise ValueError("No input FASTA file has been selected.")

    selected_path = os.fspath(INPUT_FASTA)
    FULL_INPUT_FASTA = (
        os.path.normpath(selected_path)
        if os.path.isabs(selected_path)
        else os.path.normpath(os.path.join(FASTA_DIR, selected_path))
    )
    SEQUENCE_SET = os.path.splitext(os.path.basename(selected_path))[0]
    OUTPUT_FASTA = (
        FULL_INPUT_FASTA
        if OVER_WRITE
        else os.path.join(FASTA_DIR, f"{SEQUENCE_SET}_sanitized.fasta")
    )

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def should_remove_by_header(header, filter_string):
    """
    Checks if the header contains the specified filter string (case-sensitive).
    """
    normalized_filter = str(filter_string).strip()
    if not normalized_filter:
        return False

    return normalized_filter in header


def validate_configuration(fasta_dir, input_fasta, output_fasta):
    """Validate paths that must be selected before processing starts."""
    if fasta_dir is None or not str(fasta_dir).strip():
        raise ValueError(
            "No FASTA output directory is configured. Select FASTA_DIR in "
            "the utility settings before running sequence sanitization."
        )
    if input_fasta is None or not str(input_fasta).strip():
        raise ValueError(
            "No input FASTA is configured. Select INPUT_FASTA before running "
            "sequence sanitization."
        )
    if output_fasta is None or not str(output_fasta).strip():
        raise ValueError("Unable to determine the sanitized FASTA output path.")

def write_fasta_atomic(file_path, headers, sequences, refuse_empty=False):
    """Atomically write UTF-8 FASTA without a BOM via a same-directory temp file."""
    if len(headers) != len(sequences):
        raise ValueError("FASTA header and sequence counts do not match.")
    if refuse_empty and not headers:
        raise RuntimeError(
            "Refusing to overwrite the input FASTA because sanitization produced "
            "zero output sequences."
        )

    output_path = os.path.abspath(file_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            for header, seq in zip(headers, sequences):
                temporary_file.write(f">{header}\n{seq}\n")

        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

def length_statistics(lengths):
    """One line on the lengths: their number, minimum, median, mean, SD and maximum.

    A Message: str() is the English line the terminal prints, and display()
    the chart's, in the Tools window's language. The numbers come formatted,
    with thousands separators, so the count isn't Qt's %n, which has none;
    one sequence and several get a line each.

    The standard deviation is that of these sequences themselves (numpy's
    default, ddof=0), not an estimate for a larger population.
    """
    values = np.asarray(lengths)
    median = float(np.median(values))
    numbers = {
        "count": f"{values.size:,}",
        "min": f"{values.min():,}",
        "median": f"{median:,.0f}" if median.is_integer() else f"{median:,.1f}",
        "mean": f"{values.mean():,.1f}",
        "sd": f"{values.std():,.1f}",
        "max": f"{values.max():,}",
    }
    if values.size == 1:
        return Message(
            "{count} sequence · min {min} · median {median} · mean {mean} · SD {sd} · max {max} aa",
            **numbers,
        )
    return Message(
        "{count} sequences · min {min} · median {median} · mean {mean} · SD {sd} · max {max} aa",
        **numbers,
    )


# The histogram's bins: one per length while the lengths span this many
# values or fewer, otherwise this many bins of equal width.
HISTOGRAM_BINS = 50


def length_bins(lengths):
    """The histogram's (counts, edges, one_per_length), the chart's and the table's alike.

    Lengths spanning HISTOGRAM_BINS values or fewer get one bin per length,
    from half a residue below it to half above, so each bar stands over its
    length; wider spans get HISTOGRAM_BINS bins of equal width.
    """
    values = np.asarray(lengths)
    low, high = int(values.min()), int(values.max())
    if high - low + 1 <= HISTOGRAM_BINS:
        counts, edges = np.histogram(values, bins=np.arange(low, high + 2) - 0.5)
        return counts, edges, True
    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS)
    return counts, edges, False


def length_filter_bounds():
    """The length filter's bounds above 0 as (label, length) pairs; none while it is off."""
    if not ENABLE_LENGTH_FILTER:
        return []
    return [
        (label, int(value))
        for label, value in (("min", MIN_SEQ_LENGTH), ("max", MAX_SEQ_LENGTH))
        if value is not None and value > 0
    ]


# The Tools window keeps its theme colour in app_settings.json under this
# setting (Desktop_App's TOOLS_THEME_SETTING). It launches this tool, so the
# pop-up histogram takes that colour.
_TOOLS_THEME_SETTING = "TOOLS_THEME_COLOR"
# The histogram's bars, after Config's Network Statistics Report: a theme
# shade with a darker edge in the theme colour, a gap between bars (a
# fraction of a bin's width) and corners 2.25 pt round, the report's 3 px.
_BAR_FILL = "separator"
_BAR_EDGE = "accent"
_BAR_GAP = 0.16
_BAR_RADIUS = 2.25
# The chart's title, which names its window too.
_HISTOGRAM_TITLE = Message("Sequence Length Distribution")


def tools_theme_color():
    """The Tools window's theme colour, or the default one when it can't be read."""
    from desktop.Studio_Theme import DEFAULT_THEME_COLOR, THEME_COLORS
    from utilities.App_Settings import AppSettingsError, read_app_settings

    try:
        color = read_app_settings().get(_TOOLS_THEME_SETTING)
    except AppSettingsError:
        color = None
    return color if isinstance(color, str) and color in THEME_COLORS else DEFAULT_THEME_COLOR


def chart_translation():
    """(language, translator) for the chart: the Tools window's language, or (None, None).

    The language is the one the windows show: the saved LANGUAGE setting,
    else the system's, or the pseudo-language for tests
    (Desktop_App.startup_language). The translator answers from that
    language's catalog (Desktop_App.message_translator). Anything that
    can't be read leaves the chart English: the run never fails over its
    language. The terminal stays English either way.
    """
    try:
        from desktop.Desktop_App import message_translator, startup_language

        language = startup_language()
        return language, message_translator(language)
    except Exception as error:  # A chart in English beats a failed run.
        print(f"Note: the length histogram stays English ({error}).")
        return None, None


def _bound_label(label, value):
    """The text beside a length filter bound's line, as "min 50" or "max 1,000"."""
    if label == "min":
        return Message("min {length}", length=f"{value:,}")
    return Message("max {length}", length=f"{value:,}")


def _plot_font_families(language=None):
    """The bundled Noto Sans, registered with matplotlib, else matplotlib's own font.

    A language whose script Noto Sans lacks adds its language pack's font
    (Noto Sans SC for Simplified Chinese), which matplotlib draws those
    characters in. Otherwise only one family is named: matplotlib looks up
    every listed family for each text, and warns when the fallback lacks the
    title's semibold face.
    """
    from matplotlib import font_manager

    fonts = os.path.join(PROJECT_ROOT, "src", "resources", "fonts", "desktop")
    known = {entry.fname for entry in font_manager.fontManager.ttflist}

    def registered(paths):
        found = False
        for path in paths:
            try:
                if path not in known:
                    font_manager.fontManager.addfont(path)
                found = True
            except (OSError, RuntimeError, ValueError):
                pass
        return found

    core = registered(
        os.path.join(fonts, "noto", "NotoSans", face)
        for face in ("NotoSans-Regular.ttf", "NotoSans-SemiBold.ttf", "NotoSans-Bold.ttf")
    )
    families = ["Noto Sans"] if core else ["DejaVu Sans"]
    font = None
    if language is not None:
        try:
            from utilities.Language_Packs import read_pack

            font = read_pack(language).font
        except Exception:  # No usable pack: the core font, as for English.
            font = None
    if font is not None and registered(os.path.join(fonts, path) for path in font.files):
        families.append(font.family)
    return families


def plot_length_distribution(lengths, bounds=()):
    """Show the lengths' bins (length_bins) as a pop-up chart in the Tools window's theme colour.

    bounds holds (label, length) pairs, the length filter's, drawn as dashed lines.
    Its texts show in the Tools window's language (chart_translation).
    """
    language, translator = chart_translation()
    previous = set_translator(translator)
    try:
        _plot_length_distribution(lengths, bounds, language)
    finally:
        set_translator(previous)


def _plot_length_distribution(lengths, bounds, language):
    """plot_length_distribution's chart, its texts in language with the translator installed."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import BoxStyle, FancyBboxPatch
    from matplotlib.path import Path
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    from desktop.Studio_Theme import theme_tokens

    class RoundedBar(FancyBboxPatch):
        """A bar whose corners stay _BAR_RADIUS points round however the axes scale."""

        def get_path(self):
            x, y = self.get_x(), self.get_y()
            width, height = self.get_width(), self.get_height()
            (x0, y0), (x1, y1) = self.axes.transData.transform([(0, 0), (1, 1)])
            pixels = _BAR_RADIUS * self.figure.dpi / 72
            radius_x = pixels / max(abs(x1 - x0), 1e-12)
            aspect = (pixels / max(abs(y1 - y0), 1e-12)) / radius_x
            # A short or narrow bar gets corners no larger than half its side.
            size = min(radius_x, width / 2, height / aspect / 2)
            path = BoxStyle.Round(pad=0, rounding_size=size)(
                x, y / aspect, width, height / aspect, 1
            )
            return Path(path.vertices * [1, aspect], path.codes)

    tokens = theme_tokens(tools_theme_color())
    counts, edges, _one_per_length = length_bins(lengths)
    families = _plot_font_families(language)
    # A language font has regular and bold faces only (Noto Sans SC): asked for
    # semibold, matplotlib logs a fallback and draws its characters bold anyway.
    # So its title is bold, which Noto Sans has too; English keeps semibold.
    title_weight = "semibold" if len(families) == 1 else "bold"
    style = {
        "font.family": families,
        "font.size": 10,
        "axes.edgecolor": tokens["border"],
        "axes.labelcolor": tokens["text_muted"],
        "xtick.color": tokens["border"],
        "ytick.color": tokens["border"],
        "xtick.labelcolor": tokens["text_muted"],
        "ytick.labelcolor": tokens["text_muted"],
    }
    with plt.rc_context(style):
        # The figure's label names the pop-up window.
        figure, axes = plt.subplots(
            num=_HISTOGRAM_TITLE.display(), figsize=(10, 6), facecolor=tokens["surface"]
        )
        axes.set_facecolor(tokens["surface"])
        bin_width = edges[1] - edges[0]
        for left, count in zip(edges[:-1], counts):
            if count:
                axes.add_patch(RoundedBar(
                    (left + bin_width * _BAR_GAP / 2, 0), bin_width * (1 - _BAR_GAP), count,
                    boxstyle="round,pad=0", facecolor=tokens[_BAR_FILL],
                    edgecolor=tokens[_BAR_EDGE], linewidth=0.8,
                ))

        # The x axis spans the bins and any filter bound, with a little room.
        low = min([edges[0], *(value for _, value in bounds)])
        high = max([edges[-1], *(value for _, value in bounds)])
        margin = (high - low) * 0.02
        axes.set_xlim(low - margin, high + margin)
        axes.set_ylim(0, counts.max() * (1.14 if bounds else 1.06))
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.yaxis.grid(True, color=tokens["border"], linewidth=0.8)
        axes.set_axisbelow(True)
        axes.tick_params(axis="y", length=0, pad=6)
        axes.tick_params(axis="x", length=4)
        axes.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes.set_xlabel(Message("Sequence length (amino acids)").display(), labelpad=8)
        # Not "Sequences", whose catalog entry names the benchmark report's row:
        # a Message has no disambiguation, and a language may word a count apart.
        axes.set_ylabel(Message("Number of sequences").display(), labelpad=8)

        axes.set_title(
            _HISTOGRAM_TITLE.display(), loc="left", pad=30, fontsize=15,
            fontweight=title_weight, color=tokens["text"],
        )
        axes.annotate(
            length_statistics(lengths).display(), (0, 1), xycoords="axes fraction",
            xytext=(0, 10), textcoords="offset points", va="bottom",
            color=tokens["text_muted"],
        )
        for label, value in bounds:
            axes.axvline(
                value, color=tokens["text_subtle"], linewidth=1,
                linestyle=(0, (4, 3)), zorder=3,
            )
            # The label sits beside its line, on the side of the kept lengths.
            inward = 1 if label == "min" else -1
            axes.annotate(
                _bound_label(label, value).display(), (value, 1),
                xycoords=("data", "axes fraction"),
                xytext=(5 * inward, -4), textcoords="offset points",
                ha="left" if inward > 0 else "right", va="top", fontsize=9,
                color=tokens["text_muted"],
            )

        figure.tight_layout()
        plt.show()  # Display as a pop-up window
        plt.close(figure)


def print_length_distribution(lengths):
    """Print the bins of the interactive histogram (length_bins), without plotting."""
    counts, edges, one_per_length = length_bins(lengths)
    print("\nSequence Length Distribution (Post-Sanitization & Filtering)")
    if one_per_length:
        print("Sequence length in amino acids, one row per length.")
        heading = "Length (aa)"
    else:
        print("Sequence length in amino acids; bins include the lower bound and")
        print("exclude the upper bound, except the final bin includes both bounds.")
        heading = "Length bin (aa)"
    print(f"{heading:<32} | {'Frequency (sequences)':>21}")
    print("-" * 57)
    for index, count in enumerate(counts):
        if one_per_length:
            row = f"{edges[index] + 0.5:.12g}"
        else:
            closing = "]" if index == len(counts) - 1 else ")"
            row = f"[{edges[index]:.12g}, {edges[index + 1]:.12g}{closing}"
        print(f"{row:<32} | {int(count):>21}")
    print(f"{'Total':<32} | {int(counts.sum()):>21}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(argv=None):
    # Explicit JSON settings are the shared CLI/MCP headless invocation contract.
    _, headless = select_settings_path(os.path.basename(__file__), PROJECT_ROOT, argv)
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    print(f"--- 🧬 Sequence Sanitization ---")
    print(f"Reading from: {INPUT_FASTA}")

    try:
        configure_runtime_paths()
    except ValueError as error:
        raise SystemExit(f"❌ Error: {error}") from error

    validate_configuration(FASTA_DIR, INPUT_FASTA, OUTPUT_FASTA)
    
    headers, raw_seqs = read_fasta(FULL_INPUT_FASTA)
    print(f"Loaded {len(headers)} raw sequences.")
    
    empty_count = 0
    short_count = 0
    long_count = 0
    exact_duplicates_removed = 0
    different_headers_merged = 0  # <--- NEW COUNTER for your feature
    headers_renamed = 0
    headers_sanitized_count = 0  
    header_filtered_count = 0
    
    removed_exact_duplicate_headers = set() 
    renamed_duplicate_headers = []
    merged_header_logs = []       # <--- NEW TRACKER for your feature
    removed_by_header_logs = []

    stripped_counter = Counter()
    converted_counter = Counter()
    
    # --- PHASE 1: Group Headers by Sequence ---
    seq_to_headers = {}
    
    for header, seq in tqdm(zip(headers, raw_seqs), total=len(headers), desc="Sanitizing & Deduplicating"):
        # Check if we should remove by header substring
        if should_remove_by_header(header, REMOVE_BY_HEADER_STRING):
            header_filtered_count += 1
            removed_by_header_logs.append(header)
            continue
            
        # 1. Clean the header
        safe_header, was_modified = sanitize_header(header)
        if was_modified:
            headers_sanitized_count += 1
            
        # 2. Clean the sequence
        cleaned, stripped, converted = sanitize_sequence(seq)
        
        if stripped:
            stripped_counter.update(stripped)
        if converted:
            converted_counter.update(converted)
            
        if cleaned:
            if cleaned not in seq_to_headers:
                seq_to_headers[cleaned] = []
            seq_to_headers[cleaned].append(safe_header)
        else:
            empty_count += 1
            
    # --- PHASE 2: Select Longest Header & Resolve Collisions ---
    header_to_seqs = {} 
    
    for seq, current_headers in seq_to_headers.items():
        (
            best_header,
            discarded_headers,
            duplicate_headers,
            duplicates_count,
        ) = select_preferred_header(current_headers)

        if duplicates_count > 0:
            exact_duplicates_removed += duplicates_count
            removed_exact_duplicate_headers.update(duplicate_headers)
            
        # Keep the longest header if there are different headers for the same sequence.
        if discarded_headers:
            different_headers_merged += len(discarded_headers)
            merged_header_logs.append((best_header, discarded_headers))
            
        # Re-group by the chosen header. (This catches edge cases where two completely different 
        # sequences happen to end up with the exact same chosen header name).
        if best_header not in header_to_seqs:
            header_to_seqs[best_header] = []
        header_to_seqs[best_header].append(seq)
            
    # --- PHASE 3: Rename and Apply Length Filters ---
    clean_headers, clean_seqs, clean_lengths = [], [], []
    assigned_headers_by_base = allocate_unique_headers(header_to_seqs)
    
    for header, unique_seqs in header_to_seqs.items():
        # Rename headers if multiple different sequences ended up with the same header
        assigned_headers = assigned_headers_by_base[header]
        if len(unique_seqs) > 1:
            headers_renamed += len(unique_seqs)
            renamed_duplicate_headers.append(header) 
            
        for h, s in zip(assigned_headers, unique_seqs):
            seq_len = len(s)
            
            # Apply Length Filtering
            if ENABLE_LENGTH_FILTER:
                if MIN_SEQ_LENGTH is not None and MIN_SEQ_LENGTH > 0 and seq_len < MIN_SEQ_LENGTH:
                    short_count += 1
                elif MAX_SEQ_LENGTH is not None and MAX_SEQ_LENGTH > 0 and seq_len > MAX_SEQ_LENGTH:
                    long_count += 1
                else:
                    clean_headers.append(h)
                    clean_seqs.append(s)
                    clean_lengths.append(seq_len)
            else:
                clean_headers.append(h)
                clean_seqs.append(s)
                clean_lengths.append(seq_len)
            
    # Write FASTA
    print(f"\nWriting clean sequences to {OUTPUT_FASTA}...")
    write_fasta_atomic(
        OUTPUT_FASTA,
        clean_headers,
        clean_seqs,
        refuse_empty=bool(OVER_WRITE),
    )
            
    # Final Diagnostics
    print("\n" + "="*50)
    print("SANITIZATION DIAGNOSTICS")
    print("="*50)
    print(f"Original Sequences:       {len(headers)}")
    print(f"Final Sequences:          {len(clean_headers)}")
    
    if clean_lengths:
        print(f"Max Length:               {max(clean_lengths)}")
        print(f"Min Length:               {min(clean_lengths)}")
        
    print("-" * 50)
    print("REMOVED & RENAMED SEQUENCES")
    print("-" * 50)
    print(f"Empty After Clean:         {empty_count}")
    print(f"Exact Duplicates Removed:  {exact_duplicates_removed}")
    print(f"Diff Headers Merged:       {different_headers_merged}")
    print(f"Headers Sanitized (Chars): {headers_sanitized_count}") 
    print(f"Headers Renamed (_N):      {headers_renamed}")
    print(f"Removed by Header String:  {header_filtered_count}")
    if ENABLE_LENGTH_FILTER:
        print(f"Too Short (< {MIN_SEQ_LENGTH}):         {short_count}")
        print(f"Too Long  (> {MAX_SEQ_LENGTH}):         {long_count}")
    else:
        print(f"Length Filtering:             [DISABLED]")
    
    if removed_exact_duplicate_headers:
        print("\n  [Exact Duplicates Removed - Headers]")
        for h in sorted(list(removed_exact_duplicate_headers))[:20]:
            print(f"    - {h}")
        if len(removed_exact_duplicate_headers) > 20:
            print(f"    ... and {len(removed_exact_duplicate_headers) - 20} more.")
            
    if removed_by_header_logs:
        print("\n  [Removed by Header String - Headers]")
        for h in sorted(list(set(removed_by_header_logs)))[:20]:
            print(f"    - {h}")
        if len(set(removed_by_header_logs)) > 20:
            print(f"    ... and {len(set(removed_by_header_logs)) - 20} more.")
            
    # --- NEW PRINTOUT LOGIC FOR MERGED HEADERS ---
    if merged_header_logs:
        print("\n  [Different Headers Merged (Kept Preferred/Longest)]")
        for kept, discarded in merged_header_logs[:15]:
            print(f"    - Kept: '{kept}' (Discarded: {', '.join(discarded)})")
        if len(merged_header_logs) > 15:
            print(f"    ... and {len(merged_header_logs) - 15} more.")
            
    if renamed_duplicate_headers:
        print("\n  [Headers Renamed (Duplicate Name, Different Seq)]")
        for h in renamed_duplicate_headers[:20]:
            print(f"    - {h}")
        if len(renamed_duplicate_headers) > 20:
            print(f"    ... and {len(renamed_duplicate_headers) - 20} more.")
        
    print("-" * 50)
    print("STRIPPED CHARACTERS (Leading/Trailing)")
    print("-" * 50)
    if not stripped_counter:
        print("  None.")
    else:
        for char, count in stripped_counter.most_common():
            display_char = f"'{char}'" if char.strip() else f"Whitespace/Invisible"
            print(f"  {display_char:<20} : {count}")
            
    print("-" * 50)
    print("CONVERTED TO 'X' (Internal Invalid/Gaps)")
    print("-" * 50)
    if not converted_counter:
        print("  None. All internal characters were valid.")
    else:
        for char, count in converted_counter.most_common():
            display_char = f"'{char}'" if char.strip() else f"Whitespace/Invisible"
            print(f"  {display_char:<20} : {count} occurrences")
            
    print("="*50 + "\n")
    print(f"✅ Done! Sanitized FASTA saved.")

    # Length Statistics & Histogram
    if not clean_lengths:
        print(f"⚠️ No sequences passed the length filters. Histogram skipped.")
        return 0
    print(f"\nLength statistics: {length_statistics(clean_lengths)}")
    if not SHOW_LENGTH_HISTOGRAM:
        print("Length histogram skipped (SHOW_LENGTH_HISTOGRAM is off).")
    elif headless:
        print_length_distribution(clean_lengths)
    else:
        print(f"Opening length distribution histogram...")
        plot_length_distribution(clean_lengths, length_filter_bounds())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
