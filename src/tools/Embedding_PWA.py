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
File: Embedding_PWA.py
===================================
Description:
This script performs a Pairwise Sequence Alignment (PWA) between exactly two specific sequences, utilizing their 
structural protein language model (pLM) embeddings instead of traditional amino acid substitution matrices. 
It represents a "sandbox" or debugging version of the core algorithm used in the massive all-vs-all array script.

Input:
- A metadata-first embedding HDF5 file containing sanitized headers, sequences,
  model metadata, and residue-level embedding tensors.
- Optional manually entered reference and target sequences.

Output:
- Prints a text-based visual alignment of the two sequences directly to the terminal, highlighting matching vs mismatched residues along with the final similarity score.
"""
# %% Import
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import numpy as np
import torch
import h5py
from utilities import Hardware_Utils
from utilities.FASTA_Sanitization import sanitize_header, sanitize_sequence
from utilities.Embedding_HDF5 import (
    read_embedding_manifest,
    validate_embedding_array,
)
from tools.Generate_Embeddings import find_model_plugin

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_EMBED = None

REF_HEADER = None
TAR_HEADER = None

MANUAL_REF_SEQ = False
REF_SEQUENCE = ""
MANUAL_TAR_SEQ = False
TAR_SEQUENCE = ""

HIGHLIGHT_POSITIONS = ""
EMBEDDING_MODEL = "esmc_300m"

ALIGNMENT_MODE = "global"
LOCAL_GAP_P = -2.0
GLOBAL_GAP_P = 0.0

from utilities.Tool_Directories import project_directory_defaults
from utilities.Tool_Settings import inherited_settings_path, load_tool_settings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
REPORT_DIR = _DEFAULT_DIRECTORIES["REPORT_DIR"]
GENERATE_REPORT = False

# --- JSON Settings Override ---
import json
import ast

# Automatically calculate the root directory of the SSN project for the current PC
SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if __name__ != "__main__" and os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            all_settings = json.load(f)
            if "DIRECTORIES" in all_settings:
                for k, v in all_settings["DIRECTORIES"].items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        if not os.path.isabs(str(v)):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, str(v)))
                        globals()[k] = v
                        
            script_name = os.path.basename(__file__)
            if script_name in all_settings:
                user_settings = all_settings[script_name]
                for k, v in user_settings.items():
                    if k in globals() and v is not None and str(v).strip() != "":
                        orig = globals()[k]
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
                        if isinstance(v, str) and k.endswith("_DIR") and not os.path.isabs(v):
                            v = os.path.normpath(os.path.join(PROJECT_ROOT, v))
                        globals()[k] = v
    except Exception as e:
        print(f"Failed to load user settings: {e}")

# --- DYNAMIC INFERENCE ---
FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR and INPUT_EMBED else ""

def resolve_manual_alignment_inputs(
    manual_ref_sequence,
    ref_sequence,
    manual_tar_sequence,
    tar_sequence,
    inferred_model_name,
    selected_model_name,
):
    """Apply manual switches, canonical sequence cleaning, and model selection."""

    def resolve_sequence(enabled, sequence, label):
        if not enabled:
            return ""
        cleaned_sequence, _, _ = sanitize_sequence(str(sequence or ""))
        if not cleaned_sequence:
            raise ValueError(
                f"{label} manual sequence is empty after sanitization."
            )
        return cleaned_sequence

    resolved_ref = resolve_sequence(
        manual_ref_sequence,
        ref_sequence,
        "Reference",
    )
    resolved_tar = resolve_sequence(
        manual_tar_sequence,
        tar_sequence,
        "Target",
    )
    selected_model = str(selected_model_name or "").strip()
    model_name = (
        selected_model
        if manual_ref_sequence and manual_tar_sequence and selected_model
        else inferred_model_name
    )
    if not model_name:
        raise ValueError(
            "An embedding model is required for manual sequence generation."
        )
    return resolved_ref, resolved_tar, model_name

# ==========================================
# 1. HELPER FUNCTIONS (Data Loading & Gen)
# ==========================================

def prepare_embedding_database(h5_path):
    """Load and validate the metadata-first embedding database manifest."""
    if not h5_path or not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Embedding database not found: {h5_path or '(not selected)'}."
        )
    with h5py.File(h5_path, "r") as hf:
        return read_embedding_manifest(hf, require_complete=True)


def sanitize_alignment_header(header):
    """Apply the canonical Generate_Embeddings header rules."""
    return sanitize_header(str(header or ""))[0]

def fetch_embedding(h5_path, header):
    if not header or not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as hf:
        if "embeddings" in hf and header in hf["embeddings"]:
            emb_array = hf["embeddings"][header][:]
            return torch.from_numpy(emb_array).float()
    return None

def load_model_integrated(model_name):
    device = Hardware_Utils.get_optimal_device()
    print(f"[System] Loading model '{model_name}' on {device}...")
    plugin = find_model_plugin(model_name)
    if plugin is None:
        raise ValueError(
            f"Model '{model_name}' is not supported by an available pLM plugin."
        )
    return plugin.load_model(model_name, device), device, plugin

def get_embedding_integrated(seq, model_obj, device, model_type, target_dtype):
    return model_type.get_embedding(seq, model_obj, device, target_dtype)
def map_highlight_positions(highlight_str, ref_to_tar_map):
    if not highlight_str:
        return []
        
    mapped_items = []
    for item in highlight_str.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' in item:
            try:
                start_str, end_str = item.split('-')
                start_ref = int(start_str.strip())
                end_ref = int(end_str.strip())
                
                # Find all target positions corresponding to the reference range [start_ref, end_ref]
                target_positions = []
                for pos in range(start_ref, end_ref + 1):
                    tar_pos = ref_to_tar_map.get(pos)
                    if tar_pos is not None:
                        target_positions.append(tar_pos)
                
                if not target_positions:
                    mapped_items.append("*")
                else:
                    target_positions.sort()
                    start_tar = target_positions[0]
                    end_tar = target_positions[-1]
                    if start_tar == end_tar:
                        mapped_items.append(str(start_tar))
                    else:
                        mapped_items.append(f"{start_tar}-{end_tar}")
            except Exception:
                mapped_items.append("*")
        elif item.isdigit():
            try:
                pos_ref = int(item)
                tar_pos = ref_to_tar_map.get(pos_ref)
                if tar_pos is not None:
                    mapped_items.append(str(tar_pos))
                else:
                    mapped_items.append("*")
            except Exception:
                mapped_items.append("*")
        else:
            mapped_items.append("*")
            
    return mapped_items

# ==========================================
# 2. ALIGNMENT ALGORITHMS (Core Logic)
# ==========================================

def needleman_wunsch_custom(score_matrix, gap_penalty):
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    gap_penalty = np.float32(gap_penalty)
    N, M = score_matrix.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float32)
    dp[0, :] = np.arange(M + 1, dtype=np.float32) * gap_penalty
    dp[:, 0] = np.arange(N + 1, dtype=np.float32) * gap_penalty

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp[i-1, j-1] + score_matrix[i-1, j-1]
            delete = dp[i-1, j] + gap_penalty
            insert = dp[i, j-1] + gap_penalty
            dp[i, j] = max(match, delete, insert)

    i, j = N, M
    idx_1, idx_2 = [], []
    while i > 0 or j > 0:
        curr = dp[i, j]
        if i > 0 and j > 0 and np.isclose(curr, dp[i-1, j-1] + score_matrix[i-1, j-1]):
            idx_1.append(i - 1); idx_2.append(j - 1); i -= 1; j -= 1
        elif i > 0 and np.isclose(curr, dp[i-1, j] + gap_penalty):
            idx_1.append(i - 1); idx_2.append(-1); i -= 1
        else:
            idx_1.append(-1); idx_2.append(j - 1); j -= 1

    return idx_1[::-1], idx_2[::-1], dp[N, M]

def smith_waterman_custom(score_matrix, gap_penalty):
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    score_matrix = score_matrix - np.float32(2.0)
    gap_penalty = np.float32(gap_penalty)
    N, M = score_matrix.shape
    dp = np.zeros((N + 1, M + 1), dtype=np.float32)
    pointer = np.zeros((N + 1, M + 1), dtype=np.int8)
    max_score, max_pos = np.float32(0.0), (0, 0)

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = dp[i-1, j-1] + score_matrix[i-1, j-1]
            delete = dp[i-1, j] + gap_penalty
            insert = dp[i, j-1] + gap_penalty
            score = max(0, match, delete, insert) 
            dp[i, j] = score
            if score > max_score: max_score = score; max_pos = (i, j)
            if score == 0: pointer[i, j] = 0
            elif score == match: pointer[i, j] = 1
            elif score == delete: pointer[i, j] = 2
            else: pointer[i, j] = 3

    i, j = max_pos
    idx_1, idx_2 = [], []
    while i > 0 and j > 0 and dp[i, j] > 0:
        p = pointer[i, j]
        if p == 1:
            idx_1.append(i - 1); idx_2.append(j - 1); i -= 1; j -= 1
        elif p == 2:
            idx_1.append(i - 1); idx_2.append(-1); i -= 1
        elif p == 3:
            idx_1.append(-1); idx_2.append(j - 1); j -= 1
        else: break

    return idx_1[::-1], idx_2[::-1], max_score

# ==========================================
# 3. MAIN RUNNER
# ==========================================

def compute_score_matrix_torch(emb_i, emb_j, device):
    """Build the population-normalized residue score matrix."""
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float32)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float32)
    t_i_norm = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j_norm = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    cos_sim = torch.mm(t_i_norm, t_j_norm.T).clamp(-1.0, 1.0)
    sim_mat = torch.exp(-(1.0 - cos_sim))

    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True)
    row_std = sim_mat.std(dim=1, keepdim=True, correction=0)
    col_mean = sim_mat.mean(dim=0, keepdim=True)
    col_std = sim_mat.std(dim=0, keepdim=True, correction=0)
    z_row = (sim_mat - row_mean) / (row_std + epsilon)
    z_col = (sim_mat - col_mean) / (col_std + epsilon)
    return ((z_row + z_col) / 2.0).to(
        dtype=torch.float32,
        device="cpu",
    ).numpy()

def run_alignment(
    header_ref,
    header_tar,
    seq_ref_manual,
    seq_tar_manual,
    h5_path,
    seq_db,
    mode,
    gap_p_local,
    gap_p_global,
    highlight_str,
    model_name=None,
    *,
    manual_ref_enabled=None,
    manual_tar_enabled=None,
):
    if manual_ref_enabled is None:
        manual_ref_enabled = bool(seq_ref_manual)
    else:
        manual_ref_enabled = bool(manual_ref_enabled)
    if manual_tar_enabled is None:
        manual_tar_enabled = bool(seq_tar_manual)
    else:
        manual_tar_enabled = bool(manual_tar_enabled)

    header_ref = sanitize_alignment_header(header_ref)
    header_tar = sanitize_alignment_header(header_tar)
    stored_headers = list(seq_db)

    # 1. Determine Sequences and Check for Pre-calculated Embeddings
    seq_ref = (
        sanitize_sequence(str(seq_ref_manual or ""))[0]
        if manual_ref_enabled
        else ""
    )
    emb_ref = None

    if manual_ref_enabled:
        if not seq_ref:
            raise ValueError(
                "Reference manual sequence is empty after sanitization."
            )
        print("[Input] Using manually provided sanitized Reference sequence (Forcing Generation).")
    else:
        if not header_ref and stored_headers:
            header_ref = stored_headers[0]
        if header_ref and header_ref in seq_db:
            seq_ref = seq_db[header_ref]
            emb_ref = fetch_embedding(h5_path, header_ref)
            if emb_ref is not None:
                print(f"[Input] Found pre-calculated embedding for Reference '{header_ref}'.")
        else:
            raise ValueError(
                f"CRITICAL: Reference sequence not provided and sanitized "
                f"header '{header_ref}' not found in the embedding database."
            )

    seq_tar = (
        sanitize_sequence(str(seq_tar_manual or ""))[0]
        if manual_tar_enabled
        else ""
    )
    emb_tar = None

    if manual_tar_enabled:
        if not seq_tar:
            raise ValueError(
                "Target manual sequence is empty after sanitization."
            )
        print("[Input] Using manually provided sanitized Target sequence (Forcing Generation).")
    else:
        if not header_tar and stored_headers:
            header_tar = stored_headers[1] if len(stored_headers) > 1 else stored_headers[0]
        if header_tar and header_tar in seq_db:
            seq_tar = seq_db[header_tar]
            emb_tar = fetch_embedding(h5_path, header_tar)
            if emb_tar is not None:
                print(f"[Input] Found pre-calculated embedding for Target '{header_tar}'.")
        else:
            raise ValueError(
                f"CRITICAL: Target sequence not provided and sanitized "
                f"header '{header_tar}' not found in the embedding database."
            )

    # 2. Generate Missing Embeddings dynamically
    if emb_ref is None or emb_tar is None:
        runtime_model_name = str(model_name or "").strip()
        if not runtime_model_name:
            raise ValueError(
                "An embedding model is required to generate missing embeddings."
            )
        print(f"\n[Input] Generating missing embeddings using model: {runtime_model_name}")
        model_obj, device, model_type = load_model_integrated(runtime_model_name)
        target_dtype = np.float32

        def generate_validated_embedding(sequence, label, feature_dimension=None):
            embedding = get_embedding_integrated(
                sequence,
                model_obj,
                device,
                model_type,
                target_dtype,
            )
            validate_embedding_array(
                embedding,
                sequence,
                "float32",
                feature_dimension=feature_dimension,
                require_finite=True,
                header=label,
            )
            return torch.from_numpy(embedding).float()
        
        if emb_ref is None:
            print("        -> Generating Reference Embedding...")
            expected_dimension = emb_tar.shape[1] if emb_tar is not None else None
            emb_ref = generate_validated_embedding(
                seq_ref,
                header_ref or "Manual Reference",
                expected_dimension,
            )
            
        if emb_tar is None:
            print("        -> Generating Target Embedding...")
            emb_tar = generate_validated_embedding(
                seq_tar,
                header_tar or "Manual Target",
                emb_ref.shape[1],
            )

    # 3. Process Highlighting Positions
    highlight_set = set()
    if highlight_str:
        for p in highlight_str.split(','):
            p = p.strip()
            if not p: continue
            if '-' in p:
                try:
                    start, end = map(int, p.split('-'))
                    highlight_set.update(range(start, end + 1))
                except: pass
            elif p.isdigit():
                highlight_set.add(int(p))

    # 4. Calculate Similarity Matrix
    print(f"\n[Compute] Calculating similarity matrix...")
    device = Hardware_Utils.get_optimal_device()
    score_mat = compute_score_matrix_torch(emb_ref, emb_tar, device)

    # 5. Run Alignment
    print(f"[Compute] Running {mode.upper()} alignment...")
    if mode == "global":
        idx_1, idx_2, score = needleman_wunsch_custom(score_mat, gap_p_global)
    else:
        idx_1, idx_2, score = smith_waterman_custom(score_mat, gap_p_local)

    # 6. Visualize
    len_ref = len(seq_ref)
    len_tar = len(seq_tar)
    align_len = len(idx_1)

    print("\n" + "="*80)
    print(f"ALIGNMENT RESULT (Mode: {mode.upper()} | Score: {score:.4f})")
    print("="*80)
    print(f"Reference : {header_ref if not manual_ref_enabled else 'Manual Input'} (Length: {len_ref})")
    print(f"Target    : {header_tar if not manual_tar_enabled else 'Manual Input'} (Length: {len_tar})")
    print(f"Align Len : {align_len}")
    print("-" * 80)

    # Map and print highlight positions
    mapped_positions_lines = []
    if highlight_str and highlight_str.strip():
        ref_to_tar_map = {}
        for i, j in zip(idx_1, idx_2):
            if i != -1:
                ref_to_tar_map[i + 1] = j + 1 if j != -1 else None

        mapped_items = map_highlight_positions(highlight_str, ref_to_tar_map)
        if mapped_items:
            print("Highlight Position Mapping (Reference -> Target):")
            orig_items = [item.strip() for item in highlight_str.split(',') if item.strip()]
            for orig, mapped in zip(orig_items, mapped_items):
                mapping_line = f"  {orig} -> {mapped}"
                print(mapping_line)
                mapped_positions_lines.append(mapping_line)
            print("-" * 80)

    # Output Parsing with ANSI Colors
    GREEN = "\033[1;32m"
    RESET = "\033[0m"

    alignment_data = []
    for i, j in zip(idx_1, idx_2):
        c1 = seq_ref[i] if i != -1 else "-"
        c2 = seq_tar[j] if j != -1 else "-"
        marker = "|" if (i != -1 and j != -1 and c1 == c2) else ("." if i!=-1 and j!=-1 else " ")
        is_hl = (i != -1 and (i + 1) in highlight_set)
        alignment_data.append((c1, c2, marker, is_hl))

    chunk = 80
    for k in range(0, len(alignment_data), chunk):
        chunk_data = alignment_data[k:k+chunk]
        ref_str = ""
        mark_str = ""
        tar_str = ""
        for c1, c2, m, is_hl in chunk_data:
            if is_hl:
                ref_str += f"{GREEN}{c1}{RESET}"
                tar_str += f"{GREEN}{c2}{RESET}"
            else:
                ref_str += c1
                tar_str += c2
            mark_str += m
        print(f"Ref: {ref_str}")
        print(f"     {mark_str}")
        print(f"Tar: {tar_str}\n")

    if GENERATE_REPORT:
        import datetime
        html_lines = []
        html_lines.append("<!DOCTYPE html>")
        html_lines.append("<html>")
        html_lines.append("<head>")
        html_lines.append("<meta charset='utf-8'>")
        html_lines.append("<title>Pairwise Sequence Alignment Report</title>")
        html_lines.append("<style>")
        html_lines.append("body { font-family: monospace; background-color: #ffffff; color: #1e293b; padding: 20px; font-size: 14px; line-height: 1.5; }")
        html_lines.append("pre { margin: 0; white-space: pre-wrap; word-wrap: break-word; }")
        html_lines.append(".highlight { color: #ef4444; font-weight: bold; }")
        html_lines.append(".title { color: #1e40af; font-weight: bold; }")
        html_lines.append(".header { color: #b45309; }")
        html_lines.append(".score { color: #15803d; }")
        html_lines.append("hr { border: 0; border-top: 1px solid #e2e8f0; margin: 20px 0; }")
        html_lines.append("</style>")
        html_lines.append("</head>")
        html_lines.append("<body>")
        html_lines.append("<pre>")
        
        html_lines.append("="*80)
        html_lines.append(f"<span class='title'>ALIGNMENT RESULT (Mode: {mode.upper()} | Score: {score:.4f})</span>")
        html_lines.append("="*80)
        html_lines.append(f"Reference : <span class='header'>{header_ref if not manual_ref_enabled else 'Manual Input'}</span> (Length: {len_ref})")
        html_lines.append(f"Target    : <span class='header'>{header_tar if not manual_tar_enabled else 'Manual Input'}</span> (Length: {len_tar})")
        html_lines.append(f"Align Len : {align_len}")
        html_lines.append("-" * 80)
        
        if mapped_positions_lines:
            html_lines.append("Highlight Position Mapping (Reference -> Target):")
            html_lines.extend(mapped_positions_lines)
            html_lines.append("-" * 80)
            
        for k in range(0, len(alignment_data), chunk):
            chunk_data = alignment_data[k:k+chunk]
            html_ref = ""
            html_mark = ""
            html_tar = ""
            for c1, c2, m, is_hl in chunk_data:
                if is_hl:
                    html_ref += f"<span class='highlight'>{c1}</span>"
                    html_tar += f"<span class='highlight'>{c2}</span>"
                else:
                    html_ref += c1
                    html_tar += c2
                html_mark += m
            html_lines.append(f"Ref: {html_ref}")
            html_lines.append(f"     {html_mark}")
            html_lines.append(f"Tar: {html_tar}\n")
            
        html_lines.append("</pre>")
        html_lines.append("</body>")
        html_lines.append("</html>")
        
        os.makedirs(REPORT_DIR, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"PWA_Report_{current_time}.html"
        report_path = os.path.join(REPORT_DIR, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))
        print(f"[Export] Alignment report saved to: {report_path}")


# ==========================================
# 4. USER CONFIGURATION
# ==========================================
def main(argv=None):
    global FULL_INPUT_EMBED
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    FULL_INPUT_EMBED = (
        os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR and INPUT_EMBED else ""
    )
    print(f"--- 🧬 Embedding Pairwise Alignment ---")
    try:
        database = None
        if not MANUAL_REF_SEQ or not MANUAL_TAR_SEQ:
            database = prepare_embedding_database(FULL_INPUT_EMBED)
        seq_database = database.sequence_by_header if database else {}
        database_model_name = database.model_name if database else None

        manual_ref_sequence, manual_tar_sequence, runtime_model_name = (
            resolve_manual_alignment_inputs(
                MANUAL_REF_SEQ,
                REF_SEQUENCE,
                MANUAL_TAR_SEQ,
                TAR_SEQUENCE,
                database_model_name,
                EMBEDDING_MODEL,
            )
        )
        
        run_alignment(REF_HEADER, TAR_HEADER, manual_ref_sequence, manual_tar_sequence,
                      FULL_INPUT_EMBED, seq_database, ALIGNMENT_MODE, 
                      LOCAL_GAP_P, GLOBAL_GAP_P, HIGHLIGHT_POSITIONS,
                      runtime_model_name,
                      manual_ref_enabled=MANUAL_REF_SEQ,
                      manual_tar_enabled=MANUAL_TAR_SEQ)
        
    except Exception as e:
        print(f"\n❌ {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
