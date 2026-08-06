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
File: Sparse_MSA_Converter.py
===================================
Description:
Massive multiple sequence alignments (MSAs) padded heavily with gap characters ("-") are extraordinarily 
inefficient to store in system RAM as contiguous string arrays. This script parses standard aligned FASTA 
text files and converts them into mathematically compressed SciPy Sparse CSR matrices.

Input:
- A completed multiple sequence alignment in FASTA format (`INPUT_FASTA`).

Output:
- A compressed HDF5 database (.h5) containing the structural SciPy matrix arrays, a specialized integer-to-amino-acid 
  character map, and header dictionaries for O(1) instantaneous lookup (`output_h5`).

Settings:
- SEQUENCE_SET / MSA_METHOD: File path parameters used to locate the input FASTA file.

Algorithm:
1. Iterates through the input alignment FASTA one record at a time.
2. Initializes coordinate lists for Rows (Sequence Index) and Columns (Amino Acid position).
3. If an explicit amino acid is found, it looks up the character in `AA_MAP` (e.g. 'A' -> 1) and records 
   the coordinate geometry. If a gap (`-`) is found, it skips the coordinate entirely (treated as a 0).
4. After fully encoding the array, it constructs a SciPy Compressed Sparse Row (CSR) matrix assigning 
   each coordinate an unsigned 8-bit integer (shrinking the memory footprint of massive alignments by >95%).
5. Serializes the complex object into an HDF5 file ready to be instantly mounted into the Python backend 
   of the viewer GUI.
"""
# %% --- Imports ---
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import numpy as np
import h5py
import json
from Bio import SeqIO

# Check for Scipy
try:
    from scipy import sparse
except ImportError:
    raise ImportError("Error: 'scipy' is missing. Please install it: !pip install scipy")

# --- Configuration ---
INPUT_FASTA = None
CONVERT_ALL = False

# --- DIRECTORY DEFAULTS ---
MSA_DIR = os.path.join("..", "Input_Files", "Multiple_Alignments")

# --- JSON Settings Override ---
import ast

# Automatically calculate the root directory of the SSN project for the current PC
# (Tool scripts are located in the /tools/ folder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
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

# --- DYNAMIC INFERENCE ---
# Built AFTER JSON loading so it uses the updated MSA_DIR
FULL_INPUT_FASTA = os.path.join(MSA_DIR, INPUT_FASTA) if INPUT_FASTA else ""

# --- Constants & Mapping ---
# Maps Amino Acids to Integers (1-21). 0 is reserved for Gaps.
AA_MAP = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 
    'W': 18, 'Y': 19, 'V': 20, 
    'X': 21, 'B': 3, 'Z': 6, 'J': 10, 'U': 5, 'O': 12
}
INT_TO_AA = {v: k for k, v in AA_MAP.items() if k not in ['B', 'Z', 'J', 'U', 'O']}

def build_sparse_alignment(input_path):
    if not input_path or not os.path.exists(input_path):
        print(f"❌ Error: File not found: {input_path}")
        return

    # Auto-generate output filename in the same directory
    output_h5 = os.path.splitext(input_path)[0] + "_sparse.h5"
    
    print(f"--- Building Sparse Alignment ---")
    print(f"📂 Input:  {input_path}")
    print(f"💾 Output: {output_h5}")

    # Data containers for Sparse Matrix construction
    row_ind = []
    col_ind = []
    data_vals = []
    
    headers = []
    header_map = {} 
    
    max_col = 0
    row_idx = 0

    print("Processing sequences...", end="")
    
    try:
        for record in SeqIO.parse(input_path, "fasta"):
            # 1. Store Header Info
            headers.append(record.description)
            header_map[record.id] = row_idx
            header_map[record.description] = row_idx
            
            # 2. Encode Sequence
            seq_str = str(record.seq).upper()
            length = len(seq_str)
            if length > max_col: max_col = length
            
            # Identify non-gap indices
            for col_idx, char in enumerate(seq_str):
                if char in AA_MAP:
                    val = AA_MAP[char]
                    row_ind.append(row_idx)
                    col_ind.append(col_idx)
                    data_vals.append(val)
            
            row_idx += 1
            if row_idx % 5000 == 0:
                print(f".", end="") # Print a dot every 5000 seqs to show life
                
    except Exception as e:
        print(f"\n❌ Error parsing FASTA: {e}")
        return

    print(f"\n✅ Parsed {row_idx} sequences.")
    print(f"Finalizing Matrix ({row_idx} sequences x {max_col} columns)...")

    # 3. Create CSR Matrix
    # uint8 saves massive memory (1 byte per AA instead of 8 bytes)
    sparse_matrix = sparse.csr_matrix(
        (data_vals, (row_ind, col_ind)), 
        shape=(row_idx, max_col),
        dtype=np.uint8 
    )

    # 4. Save to HDF5
    try:
        with h5py.File(output_h5, "w") as hf:
            # 4a. Store Matrix Components
            mat_group = hf.create_group("matrix")
            mat_group.create_dataset("data", data=sparse_matrix.data, compression="gzip")
            mat_group.create_dataset("indices", data=sparse_matrix.indices, compression="gzip")
            mat_group.create_dataset("indptr", data=sparse_matrix.indptr, compression="gzip")
            mat_group.attrs["shape"] = sparse_matrix.shape
            
            # 4b. Store Headers (Variable Length Strings)
            dt_str = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset("headers", data=np.array(headers, dtype=object), dtype=dt_str, compression="gzip")
            
            # 4c. Store Dictionaries as JSON strings
            hf.create_dataset("header_map", data=json.dumps(header_map))
            hf.create_dataset("aa_map", data=json.dumps(AA_MAP))
            hf.create_dataset("int_to_aa", data=json.dumps(INT_TO_AA))
            
            # Global Metadata
            hf.attrs["shape"] = (row_idx, max_col)
            
        print(f"🎉 Success! Sparse alignment saved to:\n   {output_h5}")

        # ---> NEW LOGIC: Move the original FASTA <---
        base_dir = os.path.dirname(input_path)
        full_alignments_dir = os.path.join(base_dir, "Full_Alignments")
        os.makedirs(full_alignments_dir, exist_ok=True)
        
        dest_fasta = os.path.join(full_alignments_dir, os.path.basename(input_path))
        
        # os.replace handles cross-platform atomic overwriting natively
        os.replace(input_path, dest_fasta)
        
        print(f"📁 Original FASTA safely moved to:\n   {dest_fasta}")

    except Exception as e:
        print(f"❌ Error during HDF5 save or file transfer: {e}")

# --- Execution ---
if __name__ == "__main__":
    if CONVERT_ALL:
        import glob
        # Find all .fasta files in the MSA directory
        search_pattern = os.path.join(MSA_DIR, "*.fasta")
        fasta_files = glob.glob(search_pattern)
        
        if not fasta_files:
            print(f"⚠️ No FASTA files found in {MSA_DIR} to convert.")
        else:
            print(f"🚀 Starting batch conversion of {len(fasta_files)} alignments...")
            for f in fasta_files:
                build_sparse_alignment(f)
                print("-" * 40)
            print("✅ Batch conversion complete.")
    else:
        # Standard single-file execution
        build_sparse_alignment(FULL_INPUT_FASTA)
