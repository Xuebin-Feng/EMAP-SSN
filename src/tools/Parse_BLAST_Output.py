"""
File: Parse_BLAST_Output.py
===================================
Description:
This script converts an all-vs-all tabular BLAST result file (e.g., from an online server like Galaxy) 
into an HDF5 network file strictly formatted to match the output of Align_Substitution_Matrix.py.
It dynamically builds the sequence index list based on the order of appearance in the BLAST output, 
meaning sequences with zero hits (orphans) will be excluded from the final network.

Input:
- A text-based tabular BLAST output file (`INPUT_BLAST_TABULAR`).

Output:
- An HDF5 file containing the ordered headers, unique pairwise edges, and their Log10(E-Value) scores.
"""

import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import sys
import numpy as np
import math
import h5py
import re
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_BLAST_TABULAR = None 
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")

# --- JSON Settings Override ---
import json
import ast
import os

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
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

# --- DYNAMIC INFERENCE ---
FULL_INPUT_BLAST_TABULAR = os.path.join(NETWORK_DIR, INPUT_BLAST_TABULAR) if NETWORK_DIR else ""

_base_name = INPUT_BLAST_TABULAR.replace(".tabular", "")
OUTPUT_HDF5 = os.path.join(NETWORK_DIR, f"{_base_name}_[BLAST]_EValue.h5") if NETWORK_DIR else ""

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def detect_evalue_column(filepath, num_lines_to_check=1000):
    """
    Scans the first N valid lines of a tabular BLAST file to find the column 
    that most consistently contains scientific notation or '0.0' (E-values).
    """
    col_scores = {}
    
    # Regex to catch scientific notation (e.g., 1.36e-89, 2e-04) or exactly 0.0
    evalue_pattern = re.compile(r'^-?\d+(\.\d+)?e[+-]?\d+$|^0\.0$', re.IGNORECASE)
    
    lines_checked = 0
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            cols = line.split()
            if len(cols) < 3: continue 
            
            for i in range(2, len(cols)):
                if evalue_pattern.match(cols[i]):
                    col_scores[i] = col_scores.get(i, 0) + 1
                    
            lines_checked += 1
            if lines_checked >= num_lines_to_check:
                break
                
    if not col_scores:
        print("⚠️ Warning: Could not explicitly detect an E-value column using scientific notation.")
        print("Assuming standard BLAST outfmt 6 format (E-value is at column index 10).")
        return 10
        
    # Get the column index with the most matches
    best_col = max(col_scores, key=col_scores.get)
    return best_col

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"--- 🔄 Converting External BLAST Results to HDF5 ---")
    
    # 1. Verify Input
    if not os.path.exists(FULL_INPUT_BLAST_TABULAR):
        sys.exit(f"❌ Error: BLAST tabular file not found at {FULL_INPUT_BLAST_TABULAR}")

    # 2. Detect Columns
    print(f"Scanning tabular file to detect column formats...")
    evalue_col_idx = detect_evalue_column(FULL_INPUT_BLAST_TABULAR)
    
    print("-" * 40)
    print("COLUMN MAPPING CONFIGURATION:")
    print(f"  Query Header Column  : 0")
    print(f"  Target Header Column : 1")
    print(f"  E-Value Column       : {evalue_col_idx} (Auto-detected)")
    print("-" * 40 + "\n")

    # 3. Parse Results and Build Dictionary On-The-Fly
    headers = []
    id_to_index = {}
    best_edges = {}
    
    total_lines = 0
    invalid_evalue_count = 0
    
    file_size = os.path.getsize(FULL_INPUT_BLAST_TABULAR)
    
    print("Parsing BLAST alignments...")
    with open(FULL_INPUT_BLAST_TABULAR, 'r') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing") as pbar:
        for line in f:
            pbar.update(len(line))
            
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            cols = line.split()
            if len(cols) <= evalue_col_idx: continue
            
            total_lines += 1
            
            q_id = cols[0]
            s_id = cols[1]
            eval_str = cols[evalue_col_idx]
            
            # Dynamically assign indices to new headers
            if q_id not in id_to_index:
                id_to_index[q_id] = len(headers)
                headers.append(q_id)
            if s_id not in id_to_index:
                id_to_index[s_id] = len(headers)
                headers.append(s_id)
                
            u = id_to_index[q_id]
            v = id_to_index[s_id]
            
            # Ignore self-loops
            if u == v: continue
                
            # Parse E-value
            try:
                raw_e = float(eval_str)
            except ValueError:
                invalid_evalue_count += 1
                continue
                
            # Enforce undirected graph (u < v)
            if u > v:
                u, v = v, u
                
            # Convert to Log10 score exactly like Align_Substitution_Matrix.py
            log_score = -math.log10(raw_e + 1e-300)
            
            # Prevent duplicates by keeping only the best score for this pair
            pair = (u, v)
            if pair not in best_edges or log_score > best_edges[pair]:
                best_edges[pair] = log_score

    # 4. Extract Final Arrays
    arr_i_list = []
    arr_j_list = []
    arr_score_list = []
    
    for (u, v), score in best_edges.items():
        arr_i_list.append(u)
        arr_j_list.append(v)
        arr_score_list.append(score)

    num_seqs = len(headers)

    # 5. Diagnostics
    print("\n" + "="*40)
    print("PARSING DIAGNOSTICS")
    print("="*40)
    print(f"Total Tabular Rows:   {total_lines}")
    print(f"Unique Headers Found: {num_seqs}")
    print(f"Unique Edges Saved:   {len(arr_i_list)}")
    if invalid_evalue_count > 0:
        print(f"Invalid E-Values:     {invalid_evalue_count}")
    print("="*40 + "\n")

    if not arr_i_list:
        sys.exit("❌ Error: No valid alignments were parsed. Check the column indices or header formats.")

    # 6. Save to HDF5
    idx_dtype = np.uint16 if num_seqs <= 65535 else np.uint32
    print(f"Packing into numpy arrays (Topology dtype: {idx_dtype.__name__})...")
    
    arr_i = np.array(arr_i_list, dtype=idx_dtype)
    arr_j = np.array(arr_j_list, dtype=idx_dtype)
    arr_score = np.array(arr_score_list, dtype=np.float32)

    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    
    print(f"Writing to {OUTPUT_HDF5}...")
    with h5py.File(OUTPUT_HDF5, "w") as hf:
        hf.attrs["model_name"] = "BLAST"
        hf.attrs["matrix"] = "BLAST"
        dt_str = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt_str)
        hf.create_dataset("i", data=arr_i)
        hf.create_dataset("j", data=arr_j)
        hf.create_dataset("score", data=arr_score)
        
    print(f"✅ Conversion Complete!")
