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
File: Network_Extraction.py
===================================
Description:
This script acts as a filter to extract a specific sub-network from a massive master HDF5 network file.
Instead of recalculating thousands of pairwise alignment scores on a parsed-down dataset, this script simply scans 
the existing pre-computed connections database and slices out any edges that do not belong to the target whitelist.

Input:
- A large master network file containing pairwise connectivity combinations (`INPUT_NET`).
- A target text or FASTA file containing the sequence headers you wish to retain/extract (`INPUT_FASTA`).
- (Optional) A path database file associated with the embedding sequence traceback (`INPUT_PATHS`).

Output:
- A compact HDF5 file containing only the connectivity scores/edges between the sequences on your whitelist (`OUTPUT_NET`).
- (Optional) A similarly truncated traceback paths file if extracting from an embedding network (`OUTPUT_PATHS`).

Settings:
- INPUT_NET: The path to the massive network database you want to filter down.
- INPUT_FASTA: A FASTA file dictating the exact sub-population of headers that should survive the filter.
- OUTPUT_NET: The location to save the newly filtered subset network.
- INPUT_PATHS / OUTPUT_PATHS: Matching parameters for truncating traceback arrays (set to None if processing BLAST networks).

Algorithm:
1. Loads the target whitelist of sequences into RAM from the parsed FASTA headers.
2. Interrogates the HDF5 network metadata to auto-detect its file structure (BLAST vs Embedding vs Embedding+Path).
3. Constructs an integer mapping array bridging the old global node indices to the newly contiguous subset indices.
4. Loads the source/target topology vectors into RAM and masks them utilizing boolean logic (dropping connections where either participant is omitted).
5. Commits the newly filtered vectors back into a fresh HDF5 database structure.
6. If applicable, iterates over variable-length sequence traceback arrays, dropping omitting rows and updating binary structures dynamically.
"""
# %% Imports
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import sys
import h5py
import numpy as np
from tqdm import tqdm
from Cache_Manifest import validate_network_schema

# ==========================================
# USER CONFIGURATION
# ==========================================
# INPUTS (Filenames only, provided by GUI)
INPUT_NET = None 
INPUT_FASTA = None

# DIRECTORIES
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")
FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
PATH_DIR = os.path.join("..", "Cache_Files", "Global_Path")

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
import re

# 1. Resolve inputs and read the authoritative network metadata.
_fasta_base = INPUT_FASTA.replace(".fasta", "")
FULL_INPUT_NET = os.path.join(NETWORK_DIR, INPUT_NET) if NETWORK_DIR else ""
FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR else ""
_network_metadata = validate_network_schema(FULL_INPUT_NET)
is_blast = _network_metadata.network_type == "blast"
_model_name = re.sub(r'[<>:"/\\|?*]', "_", _network_metadata.model_name)

# 2. Build output paths from the metadata-selected network type.
if is_blast:
    INPUT_PATHS = None
    OUTPUT_PATHS = None
    OUTPUT_NET = os.path.join(NETWORK_DIR, f"{_fasta_base}_[{_model_name}]_EValue.h5") if NETWORK_DIR else ""
else:
    _old_net_base = INPUT_NET.replace("_network.h5", "")
    INPUT_PATHS = os.path.join(PATH_DIR, f"{_old_net_base}_paths.h5") if PATH_DIR else ""
    OUTPUT_PATHS = os.path.join(PATH_DIR, f"{_fasta_base}_[{_model_name}]_paths.h5") if PATH_DIR else ""
    OUTPUT_NET = os.path.join(NETWORK_DIR, f"{_fasta_base}_[{_model_name}]_network.h5") if NETWORK_DIR else ""

def load_fasta_headers(fasta_path):
    print(f"Loading filtered headers from: {fasta_path}")
    headers = set()
    try:
        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    headers.add(line[1:]) 
    except FileNotFoundError:
        sys.exit(f"❌ Error: FASTA file not found at {fasta_path}")
    print(f"-> Found {len(headers)} sequences in filtered FASTA.")
    return headers

def filter_network(input_net, input_fasta, output_net, input_paths, output_paths):
    # 1. Load Whitelist
    keep_headers_set = load_fasta_headers(input_fasta)

    # 2. Verify Input
    print(f"\nLoading master network: {input_net} ...")
    if not os.path.exists(input_net):
        sys.exit(f"❌ Error: Network file not found at {input_net}")

    # 3. Read HDF5 Metadata & Index Mapping
    print("Building index mapping...")
    with h5py.File(input_net, "r") as hf_in:
        
        # --- METADATA-ONLY FORMAT DETECTION ---
        network_metadata = validate_network_schema(
            hf_in,
            expected_network_type=_network_metadata.network_type,
        )
        is_blast_network = network_metadata.network_type == "blast"
        if is_blast_network:
            print("-> Format Detected: BLAST/SSEARCH E-Value Network")
        else:
            print("-> Format Detected: Embedding Similarity Network")

        raw_headers = hf_in['headers'][:]
        original_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_headers]
        
        if not is_blast_network:
            original_seq_lens = hf_in['seq_lens'][:]
        
        kept_indices_original = []
        new_headers = []
        
        # Use int32 for the map to handle -1 (unmapped/discarded)
        old_to_new = np.full(len(original_headers), -1, dtype=np.int32)
        new_idx_counter = 0

        for i, header in enumerate(original_headers):
            if header in keep_headers_set:
                old_to_new[i] = new_idx_counter
                kept_indices_original.append(i)
                new_headers.append(header)
                new_idx_counter += 1
                
        num_kept = len(new_headers)
        missing_headers = keep_headers_set - set(new_headers)
        
        print(f"-> Retaining {num_kept} / {len(original_headers)} sequences.")
        print(f"-> Missing   {len(missing_headers)} sequences from FASTA.")

        if num_kept == 0:
            sys.exit("❌ Error: No headers matched. Check your FASTA IDs vs Network IDs.")

        # Determine best index type based on new count
        idx_dtype = np.uint16 if num_kept <= 65535 else np.uint32
        print(f"-> Optimization: Using {idx_dtype.__name__} for new indices.")

        # 4. Filter and Remap Edges
        print("Loading and filtering edges...")
        src_arr = hf_in['i'][:]
        tgt_arr = hf_in['j'][:]
        
        # Create Boolean Mask: Only keep edges where BOTH nodes are in our keep list
        valid_mask = (old_to_new[src_arr] != -1) & (old_to_new[tgt_arr] != -1)
        
        n_edges_orig = len(src_arr)
        n_edges_new = np.sum(valid_mask)
        print(f"-> Edges reduced from {n_edges_orig} to {n_edges_new}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_net), exist_ok=True)

        # 5. Save to New HDF5 Network
        print(f"Saving filtered network to {output_net}...")
        with h5py.File(output_net, "w") as hf_out:
            for k, v in hf_in.attrs.items():
                hf_out.attrs[k] = v
            dt_str = h5py.string_dtype(encoding='utf-8')
            hf_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
            
            new_i = old_to_new[src_arr[valid_mask]].astype(idx_dtype)
            new_j = old_to_new[tgt_arr[valid_mask]].astype(idx_dtype)

            hf_out.create_dataset("i", data=new_i)
            hf_out.create_dataset("j", data=new_j)
            
            if is_blast_network:
                # BLAST Format
                hf_out.create_dataset("score", data=hf_in['score'][valid_mask])
            else:
                # Embedding Format
                new_seq_lens = original_seq_lens[kept_indices_original]
                hf_out.create_dataset("seq_lens", data=new_seq_lens, dtype=np.uint16)
                hf_out.create_dataset("l_score", data=hf_in['l_score'][valid_mask])
                hf_out.create_dataset("l_len", data=hf_in['l_len'][valid_mask])
                hf_out.create_dataset("g_score", data=hf_in['g_score'][valid_mask])
                hf_out.create_dataset("g_len", data=hf_in['g_len'][valid_mask])

    # 6. Extract Paths (If available and applicable)
    if is_blast_network:
        print("\nBLAST E-Value network detected. Skipping path file extraction.")
    elif input_paths is None or output_paths is None:
        print("\nPath configurations set to None. Skipping path extraction.")
    elif not os.path.exists(input_paths):
        print(f"\n⚠️  Warning: Paths file not found at {input_paths}.")
        print("   Proceeding without extracting traceback paths.")
    else:
        print(f"\nExtracting corresponding traceback paths from {input_paths}...")
        os.makedirs(os.path.dirname(output_paths), exist_ok=True)

        with h5py.File(input_paths, "r") as hf_p_in, h5py.File(output_paths, "w") as hf_p_out:
            # Recreate headers for the paths file
            dt_str = h5py.string_dtype(encoding='utf-8')
            hf_p_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)

            # Get the explicit integer indices of the rows we are keeping
            valid_indices = np.where(valid_mask)[0]

            # Setup the variable-length structure in the new file
            dt_vlen = h5py.vlen_dtype(np.uint8)
            paths_out_ds = hf_p_out.create_dataset("paths", shape=(len(valid_indices),), dtype=dt_vlen)
            master_paths = hf_p_in['paths']
            
            # Stream the filtered paths directly from disk to disk
            for new_idx, old_idx in tqdm(enumerate(valid_indices), total=len(valid_indices), desc="-> Saving Paths"):
                paths_out_ds[new_idx] = master_paths[old_idx]
                
        print(f"✅ Filtered paths saved to {output_paths}")

    print("\n✅ Sub-Network Extraction Complete!")

    if missing_headers:
        print("\n⚠️  The following FASTA headers were not found in the network:")
        for missing in list(missing_headers)[:10]:
            print(f"    - {missing}")
        if len(missing_headers) > 10:
            print(f"    ... and {len(missing_headers) - 10} more.")

if __name__ == "__main__":
    filter_network(FULL_INPUT_NET, FULL_INPUT_FASTA, OUTPUT_NET, INPUT_PATHS, OUTPUT_PATHS)
