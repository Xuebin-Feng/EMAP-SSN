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
   - Displays a pop-up histogram visualizing the final sequence length distribution.

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
import matplotlib.pyplot as plt

from utilities.FASTA_Sanitization import (
    allocate_unique_headers,
    sanitize_header,
    sanitize_sequence,
    select_preferred_header,
)
from utilities.Tool_Directories import project_directory_defaults
from utilities.Tool_Settings import inherited_settings_path, load_tool_settings

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FASTA = None  
OVER_WRITE = False
ENABLE_LENGTH_FILTER = False
MIN_SEQ_LENGTH = 0
MAX_SEQ_LENGTH = 0
REMOVE_BY_HEADER_STRING = ""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]

# --- JSON Settings Override ---
import json
import ast

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

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

def read_fasta(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
        
    headers, sequences = [], []
    current_header, current_sequence = None, []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue  
            
            if line.startswith(">"):
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_sequence))
                current_header = line[1:]
                current_sequence = []
            else:
                current_sequence.append(line)
        
        if current_header is not None:
            headers.append(current_header)
            sequences.append("".join(current_sequence))
            
    return headers, sequences

def write_fasta_atomic(file_path, headers, sequences, refuse_empty=False):
    """Write a FASTA through a same-directory temporary file and replace atomically."""
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

def plot_length_distribution(lengths):
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='royalblue', edgecolor='black', alpha=0.8)
    plt.title('Sequence Length Distribution (Post-Sanitization & Filtering)')
    plt.xlabel('Sequence Length (Number of Amino Acids)')
    plt.ylabel('Frequency (Number of Sequences)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()  # Display as a pop-up window
    plt.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(argv=None):
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

    # Plot Histogram
    if clean_lengths:
        print(f"Opening length distribution histogram...")
        plot_length_distribution(clean_lengths)
    else:
        print(f"⚠️ No sequences passed the length filters. Histogram skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
