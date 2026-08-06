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
File: Embedding_SSEARCH.py
===================================
Similar to the NCBI's traditional SSEARCH program, this tool performs a rigorous one-vs-all database search. However, instead of using a standard amino acid substitution matrix (like BLOSUM62), it aligns sequences based on the high-dimensional structural similarity of their Protein Language Model (pLM) embeddings.

This allows you to find remote homologs that share structural similarities even if their literal sequence identity has degraded entirely.

How it Works:
1. Target Database: You select a complete metadata-first embedding database (.h5) to search against.
2. Query Input: You can either type the header of a stored sequence, OR paste a brand new raw amino acid sequence into the 'Query Sequence' box.
3. Inference: The script calculates the structural embedding for your query.
4. Scanning: Using parallel CPU workers, it scans your query against every sequence in the database using either Local (Smith-Waterman) or Global (Needleman-Wunsch) dynamic programming.
5. Scoring: The raw alignment scores are normalized (to prevent bias toward excessively long sequences) and ranked.

Outputs:
The script generates report files in the configured report directory:
- Report_<Query>.txt: A human-readable text file showing the ranked hits, their normalized scores, raw scores, and effective alignment lengths.
- Hits_<Query>.fasta: A clean FASTA file containing the sequences of all your top hits, ordered strictly by rank, with your query sequence pinned to the very top. This file is perfectly formatted to be immediately dropped into an MSA tool!

Key Parameters:
- Query Sequence (Optional): If populated, the script ignores stored-header lookup and sanitizes the supplied sequence in memory.
- Norm Score Cutoff: Filters out any hits that fall below a specific normalized similarity score.
- Alignment Mode: Use 'local' if you are searching for a specific structural domain within larger proteins. Use 'global' if you are comparing overall holistic similarity.
"""
# %%
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import h5py
import numpy as np
import pandas as pd
import torch
import re
import sys
import gc
import threading
import time
from utilities import Hardware_Utils
from utilities.Alignment_Score_Kernels import global_score_length, local_score_length
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from multiprocessing import Pool, set_start_method
from tqdm import tqdm

from utilities.FASTA_Sanitization import sanitize_header, sanitize_sequence
from utilities.Embedding_HDF5 import (
    dtype_for_saving_mode,
    read_embedding_manifest,
    validate_embedding_array,
)
from tools.Generate_Embeddings import find_model_plugin

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_EMBED = "Sample_[E1_RA]_embeddings.h5"

# QUERY SETTINGS
QUERY_HEADER = "Query_Header"
QUERY_SEQUENCE = "" # Optional: if blank, fetch from INPUT_EMBED using QUERY_HEADER.
OUTPUT_NAME = "" # Optional: Custom base name for the generated output files.

# SEARCH PARAMETERS
TOP_K = 2500 
NORM_THRESHOLD = None
ALIGNMENT_MODE = "local"
LOCAL_GAP_P = -2.0
GLOBAL_GAP_P = 0.0
NORM_MODE = "longer_sequence"

# HARDWARE & CACHE
WORKERS = 8                  
ACCELERATOR_LANES = "auto"
ACCELERATOR_TUNE_PAIRS = 256
EMBED_DIR = os.path.join("..", "Embeddings")
REPORT_DIR = os.path.join("..", "Cache_Files", "Align_Report")
GENERATE_FASTA = False

# --- JSON Settings Override ---
import json
import ast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r") as f:
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
FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR else ""

# --- 3. DATA & MODEL LOADING --------------------------------------------------
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


def release_accelerator_cache(device):
    """Release cached model allocations before the alignment search begins."""
    if device is None:
        return
    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    try:
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
        elif device_type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except (AttributeError, RuntimeError):
        pass


def prepare_database_embeddings():
    if not os.path.exists(FULL_INPUT_EMBED):
        raise FileNotFoundError(
            f"Embedding database not found: {FULL_INPUT_EMBED}. Generate it "
            "with Generate_Embeddings.py before running SSEARCH."
        )
    print(f"[Init] Loading HDF5 embeddings from:\n       {FULL_INPUT_EMBED}")
    with h5py.File(FULL_INPUT_EMBED, "r") as hf:
        return read_embedding_manifest(hf, require_complete=True)

# --- 4. ALIGNMENT & NORMALIZATION LOGIC ---------------------------------------
def compute_score_matrix_torch(emb_i, emb_j, device):
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float16)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float16)
    t_i_norm = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j_norm = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    cos_sim = torch.mm(t_i_norm, t_j_norm.T)
    dist_mat = 1.0 - cos_sim
    sim_mat = torch.exp(-dist_mat)
    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True); row_std = sim_mat.std(dim=1, keepdim=True)
    col_mean = sim_mat.mean(dim=0, keepdim=True); col_std = sim_mat.std(dim=0, keepdim=True)
    z_r = (sim_mat - row_mean) / (row_std + epsilon)
    z_c = (sim_mat - col_mean) / (col_std + epsilon)
    return ((z_r + z_c) / 2.0).to(dtype=torch.float32, device="cpu").numpy()

def normalize_score(raw_score, align_len, len_i, len_j, mode):
    if mode == "alignment_length": return raw_score / align_len if align_len > 0 else 0.0
    elif mode == "shorter_sequence": denom = min(len_i, len_j); return raw_score / denom if denom > 0 else 0.0
    elif mode == "longer_sequence": denom = max(len_i, len_j); return raw_score / denom if denom > 0 else 0.0
    elif mode == "average_sequence": denom = (len_i + len_j) / 2.0; return raw_score / denom if denom > 0 else 0.0
    else: return raw_score / align_len if align_len > 0 else 0.0

# --- HDF5 MULTIPROCESSING INITIALIZATION ---
worker_hf = None
worker_device = None

def init_worker(h5_path):
    global worker_hf, worker_device
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)
    worker_device = torch.device("cpu")


def finish_search(args):
    idx, header, len_q, len_t, mode, gap, norm_mode, mat = args
    is_local = (mode == "local")
    if is_local:
        raw, path_len = local_score_length(mat, gap)
    else:
        raw, path_len = global_score_length(mat, gap)
    norm = normalize_score(raw, path_len, len_q, len_t, norm_mode)

    if norm_mode == "alignment_length":
        eff_len = path_len
    elif norm_mode == "shorter_sequence":
        eff_len = min(len_q, len_t)
    elif norm_mode == "longer_sequence":
        eff_len = max(len_q, len_t)
    elif norm_mode == "average_sequence":
        eff_len = (len_q + len_t) / 2.0
    else:
        eff_len = path_len

    return {
        "index": idx,
        "header": header,
        "raw_score": raw,
        "norm_score": norm,
        "length": eff_len,
        "seq_len": len_t,
        "aln_len": path_len,
    }


def search_cpu_worker(args):
    idx, header, safe_h, q_emb, mode, gap, norm_mode = args
    global worker_hf, worker_device
    t_emb = worker_hf["embeddings"][safe_h][:]
    mat = compute_score_matrix_torch(q_emb, t_emb, worker_device)
    return finish_search(
        (
            idx,
            header,
            q_emb.shape[0],
            t_emb.shape[0],
            mode,
            gap,
            norm_mode,
            mat,
        )
    )


accelerator_thread_state = threading.local()
accelerator_lane_cache = {}


def _device_type(device):
    return getattr(device, "type", str(device).split(":", 1)[0])


def _uses_accelerator(device):
    return _device_type(device) != "cpu"


def _supports_explicit_streams(device):
    return _device_type(device) in {"cuda", "xpu"}


def _lane_candidates(device, cpu_workers):
    device_type = _device_type(device)
    max_lanes = max(1, int(cpu_workers))
    if device_type == "cuda":
        base, backend_cap = [1, 2, 4, 8, 16], 16
    elif device_type == "xpu":
        base, backend_cap = [1, 2, 4], 4
    else:
        return [1]

    limit = min(max_lanes, backend_cap)
    candidates = [count for count in base if count <= limit]
    if limit not in candidates:
        candidates.append(limit)
    return sorted(set(candidates))


def _configured_lanes(device, cpu_workers):
    configured = ACCELERATOR_LANES
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {"", "auto"}:
            return None
        try:
            configured = int(normalized)
        except ValueError:
            print(
                f"⚠️ Invalid ACCELERATOR_LANES={configured!r}; using auto."
            )
            return None
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        return None
    if configured < 1:
        return None
    if not _supports_explicit_streams(device):
        return 1
    return min(configured, max(1, int(cpu_workers)))


def _accelerator_name(device):
    device_type = _device_type(device)
    try:
        if device_type == "cuda":
            return torch.cuda.get_device_name(device)
        if device_type == "xpu":
            return torch.xpu.get_device_name(device)
        if device_type == "mps" and hasattr(torch.backends.mps, "get_name"):
            return torch.backends.mps.get_name()
    except (AttributeError, RuntimeError):
        pass
    return str(device)


class _DeviceStreamContext:
    def __init__(self, backend, device, stream):
        self.device_context = backend.device(device)
        self.stream_context = backend.stream(stream)

    def __enter__(self):
        self.device_context.__enter__()
        try:
            return self.stream_context.__enter__()
        except BaseException:
            self.device_context.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.stream_context.__exit__(
                exc_type,
                exc_value,
                traceback,
            )
        finally:
            self.device_context.__exit__(exc_type, exc_value, traceback)


def _stream_context(device):
    device_type = _device_type(device)
    if device_type == "cuda":
        backend = torch.cuda
    elif device_type == "xpu":
        backend = torch.xpu
    else:
        return nullcontext()

    key = (device_type, str(device))
    stream = getattr(accelerator_thread_state, "stream", None)
    if (
        stream is None
        or getattr(accelerator_thread_state, "stream_key", None) != key
    ):
        stream = backend.Stream(device=device)
        accelerator_thread_state.stream = stream
        accelerator_thread_state.stream_key = key
    return _DeviceStreamContext(backend, device, stream)


def _compute_accelerated_search(args):
    idx, header, q_emb, t_emb, mode, gap, norm_mode, device = args
    with torch.inference_mode():
        with _stream_context(device):
            mat = compute_score_matrix_torch(q_emb, t_emb, device)
    return (
        idx,
        header,
        q_emb.shape[0],
        t_emb.shape[0],
        mode,
        gap,
        norm_mode,
        mat,
    )


def _run_accelerated_search(
    tasks,
    workers,
    input_h5,
    device,
    lanes,
    show_progress,
):
    results = []
    accelerator_pending = set()
    cpu_pending = set()
    ready_for_cpu = deque()
    ready_limit = max(lanes, min(workers, 8))
    task_iterator = iter(tasks)
    tasks_exhausted = False
    progress_context = (
        tqdm(
            total=len(tasks),
            desc=f"Search ({lanes} accelerator lane"
                 f"{'s' if lanes != 1 else ''})",
        )
        if show_progress
        else nullcontext(None)
    )

    with h5py.File(input_h5, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=lanes,
                thread_name_prefix="search-accelerator",
            ) as accelerator_executor, \
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="search-cpu",
            ) as cpu_executor, \
            progress_context as progress:
        while (
            not tasks_exhausted
            or accelerator_pending
            or ready_for_cpu
            or cpu_pending
        ):
            while ready_for_cpu and len(cpu_pending) < workers:
                cpu_pending.add(
                    cpu_executor.submit(
                        finish_search,
                        ready_for_cpu.popleft(),
                    )
                )

            while (
                not tasks_exhausted
                and len(accelerator_pending) < lanes
                and len(ready_for_cpu) + len(accelerator_pending) < ready_limit
            ):
                try:
                    (
                        idx,
                        header,
                        safe_h,
                        q_emb,
                        mode,
                        gap,
                        norm_mode,
                    ) = next(task_iterator)
                except StopIteration:
                    tasks_exhausted = True
                    break

                t_emb = hf["embeddings"][safe_h][:]
                accelerator_pending.add(
                    accelerator_executor.submit(
                        _compute_accelerated_search,
                        (
                            idx,
                            header,
                            q_emb,
                            t_emb,
                            mode,
                            gap,
                            norm_mode,
                            device,
                        ),
                    )
                )

            completed_accelerator = {
                future
                for future in accelerator_pending
                if future.done()
            }
            completed_cpu = {
                future for future in cpu_pending if future.done()
            }
            if not completed_accelerator and not completed_cpu:
                pending = accelerator_pending | cpu_pending
                if not pending:
                    continue
                completed, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                completed_accelerator = completed & accelerator_pending
                completed_cpu = completed & cpu_pending

            for future in completed_accelerator:
                accelerator_pending.remove(future)
                ready_for_cpu.append(future.result())
            for future in completed_cpu:
                cpu_pending.remove(future)
                results.append(future.result())
                if progress is not None:
                    progress.update(1)

    return results


def _select_lanes(tasks, workers, input_h5, device):
    manual = _configured_lanes(device, workers)
    if manual is not None:
        return manual
    candidates = _lane_candidates(device, workers)
    if len(candidates) == 1 or len(tasks) < 2:
        return 1

    cache_key = (_device_type(device), _accelerator_name(device), int(workers))
    if cache_key in accelerator_lane_cache:
        return accelerator_lane_cache[cache_key]

    count = max(2, min(int(ACCELERATOR_TUNE_PAIRS), len(tasks)))
    sample = list(tasks[:count])
    print(
        f"[Search] Auto-tuning accelerator lanes on {cache_key[1]} "
        f"using {count} representative targets..."
    )
    _run_accelerated_search(
        sample, workers, input_h5, device, 1, False
    )

    rates = {}
    for lanes in candidates:
        started = time.perf_counter()
        try:
            _run_accelerated_search(
                sample, workers, input_h5, device, lanes, False
            )
        except (RuntimeError, NotImplementedError) as error:
            if lanes == 1:
                raise
            print(f"    {lanes} lanes unavailable: {error}")
            continue
        rates[lanes] = count / max(time.perf_counter() - started, 1e-9)

    fastest = max(rates.values())
    selected = min(
        lanes
        for lanes, rate in rates.items()
        if rate >= fastest * 0.97
    )
    accelerator_lane_cache[cache_key] = selected
    print(
        "    "
        + ", ".join(
            f"{lanes}: {rate:.1f} targets/s"
            for lanes, rate in sorted(rates.items())
        )
    )
    print(f"[Search] Selected {selected} accelerator lane(s).")
    return selected


def process_search_tasks(tasks, workers, input_h5):
    device = Hardware_Utils.get_optimal_device()
    if _uses_accelerator(device):
        lanes = _select_lanes(tasks, workers, input_h5, device)
        return _run_accelerated_search(
            tasks, workers, input_h5, device, lanes, True
        )

    results = []
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(input_h5,),
    ) as pool:
        iterator = pool.imap_unordered(
            search_cpu_worker,
            tasks,
            chunksize=50,
        )
        for result in tqdm(iterator, total=len(tasks)):
            results.append(result)
    return results

# --- 5. REPORTING -------------------------------------------------------------
def save_results(df, query_meta, db_size, seq_lookup, base_filename, query_seq, norm_mode, gap_p):
    q_head, q_len = query_meta
    
    col_header = "ALN-LEN"

    # Base metadata block
    meta_lines = []
    meta_lines.append("="*80); meta_lines.append(f"{'PROTEIN EMBEDDING SEARCH REPORT':^80}"); meta_lines.append("="*80)
    meta_lines.append(f" Query:       {q_head}")
    meta_lines.append(f" Query Len:   {q_len} residues")
    meta_lines.append(f" Database:    {INPUT_EMBED} ({db_size} sequences)")
    meta_lines.append(f" Mode:        {ALIGNMENT_MODE.upper()} Alignment")
    meta_lines.append("-" * 80)
    meta_lines.append(f" Parameters:  Gap Penalty = {gap_p}")
    meta_lines.append(f" Metric:      Raw Score / {norm_mode}")
    meta_lines.append(f" Filters:     Top_K={TOP_K} | Norm_Threshold={NORM_THRESHOLD}")
    meta_lines.append("-" * 80 + "\n")
    
    report_lines = list(meta_lines)
    onscreen_lines = list(meta_lines)
    
    if df.empty:
        report_lines.append("  [No hits found satisfying the criteria]")
        onscreen_lines.append("  [No hits found satisfying the criteria]")
        xlsx_data = []
    else:
        table_hdr = [
            f" {'RANK':<6} | {'NORM-SCR':<9} | {'RAW':<9} | {'SEQ-LEN':<8} | {col_header:<8} | {'HEADER'}",
            f"{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*35}"
        ]
        report_lines.extend(table_hdr)
        onscreen_lines.extend(table_hdr)
        
        printed_hits = 0
        rank_counter = 1
        xlsx_data = []
        for i, row in df.iterrows():
            if row['index'] == -1: continue 
            head = row['header']
            seq_len_val = int(row['seq_len'])
            aln_len_val = int(row['aln_len'])
            norm_score = row['norm_score']
            raw_score = row['raw_score']
            
            row_line = f" {rank_counter:<6} | {norm_score:<9.3f} | {raw_score:<9.1f} | {seq_len_val:<8} | {aln_len_val:<8} | {head}"
            report_lines.append(row_line)
            if printed_hits < 100:
                onscreen_lines.append(row_line)
                printed_hits += 1
                
            xlsx_data.append({
                "Rank": rank_counter,
                "Norm Score": float(norm_score),
                "Raw Score": float(raw_score),
                "Seq Len": int(seq_len_val),
                "Aln Len": int(aln_len_val),
                "Header": str(head)
            })
            
            rank_counter += 1
            
        if len(df) - 1 > 100:
            onscreen_lines.append(f"\n  [Onscreen display limited to first 100 hits. The full list of {len(df) - 1} hits is stored in the report file.]")
            
    print("\n".join(onscreen_lines))
    
    output_dir = REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, f"Report_{base_filename}.txt"), "w") as f: f.write(report_text)
    
    # Generate and save Excel Report
    xlsx_path = os.path.join(output_dir, f"Report_{base_filename}.xlsx")
    if not xlsx_data:
        xlsx_df = pd.DataFrame(columns=["Rank", "Norm Score", "Raw Score", "Seq Len", "Aln Len", "Header"])
    else:
        xlsx_df = pd.DataFrame(xlsx_data)
        
    meta_data = [
        {"Parameter": "Query Header", "Value": q_head},
        {"Parameter": "Query Length (residues)", "Value": q_len},
        {"Parameter": "Database", "Value": INPUT_EMBED},
        {"Parameter": "Database Size (sequences)", "Value": db_size},
        {"Parameter": "Alignment Mode", "Value": ALIGNMENT_MODE},
        {"Parameter": "Gap Penalty", "Value": gap_p},
        {"Parameter": "Normalization Mode", "Value": norm_mode},
        {"Parameter": "Norm Score Cutoff", "Value": NORM_THRESHOLD if NORM_THRESHOLD is not None else "None"},
        {"Parameter": "Top K", "Value": TOP_K if TOP_K is not None else "None"}
    ]
    meta_df = pd.DataFrame(meta_data)
    
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            xlsx_df.to_excel(writer, sheet_name="Search Results", index=False)
            meta_df.to_excel(writer, sheet_name="Search Parameters", index=False)
        print(f"[Export] Excel report saved to: {xlsx_path}")
    except Exception as e:
        print(f"[Warning] Failed to save Excel report: {e}")
    
    if GENERATE_FASTA:
        # OUTPUT FASTA (Ranked with query at the top)
        count = 1 
        with open(os.path.join(output_dir, f"Hits_{base_filename}.fasta"), "w") as f:
            f.write(f">{q_head}\n{query_seq}\n")
            if not df.empty and seq_lookup:
                for i, row in df.iterrows():
                    if row['index'] == -1: continue
                    head = row['header']
                    if head in seq_lookup:
                        f.write(f">{head}\n{seq_lookup[head]}\n")
                        count += 1
        print(f"[Export] {count} sequences exported to Hits_{base_filename}.fasta (Query is #1)")

# --- 6. MAIN ------------------------------------------------------------------
if __name__ == "__main__":
    database = prepare_database_embeddings()
    db_headers = database.headers
    seq_lookup = database.sequence_by_header

    # Process Query Input
    raw_query_name = QUERY_HEADER if QUERY_HEADER else "Manual_Query"
    query_name = sanitize_header(str(raw_query_name))[0] or "Manual_Query"
    raw_query_sequence = QUERY_SEQUENCE.strip() if QUERY_SEQUENCE else ""
    query_seq_str = ""
    query_emb = None
    target_dtype = dtype_for_saving_mode(database.saving_mode)
    cleanup_device = None

    if raw_query_sequence:
        query_seq_str, _, _ = sanitize_sequence(raw_query_sequence)
        if not query_seq_str:
            raise ValueError("QUERY_SEQUENCE is empty after sanitization.")
        print("[Input] Using manually provided sanitized query sequence.")
    else:
        if query_name not in seq_lookup:
            raise ValueError(
                f"QUERY_SEQUENCE is empty and sanitized QUERY_HEADER "
                f"'{query_name}' is not stored in the embedding database."
            )
        query_seq_str = seq_lookup[query_name]
        with h5py.File(FULL_INPUT_EMBED, "r") as hf:
            query_emb = hf["embeddings"][query_name][:]
        print(
            f"[Input] Reusing the stored sequence and embedding for "
            f"'{query_name}'."
        )

    if query_emb is None:
        print("[Input] Generating a new embedding for the query...")
        model_obj, device, model_type = load_model_integrated(database.model_name)
        cleanup_device = device
        query_emb = get_embedding_integrated(
            query_seq_str,
            model_obj,
            device,
            model_type,
            target_dtype,
        )
        validate_embedding_array(
            query_emb,
            query_seq_str,
            database.saving_mode,
            feature_dimension=database.feature_dimension,
            require_finite=True,
            header=query_name,
        )

    # The embedding model is no longer needed. Release it before the
    # multi-lane search so smaller accelerators retain maximum working memory.
    if "model_obj" in locals():
        del model_obj
    gc.collect()
    release_accelerator_cache(cleanup_device)

    # 3. Search
    gap_p = LOCAL_GAP_P if ALIGNMENT_MODE == "local" else GLOBAL_GAP_P
    tasks = []
    
    for i, header in enumerate(db_headers):
        tasks.append((i, header, header, query_emb, ALIGNMENT_MODE, gap_p, NORM_MODE))
        
    print(
        f"[Search] Scanning {len(tasks)} sequences against "
        f"{database.model_name} using '{NORM_MODE}' normalization..."
    )
    results = []
    
    try: set_start_method('spawn')
    except RuntimeError: pass

    results = process_search_tasks(tasks, WORKERS, FULL_INPUT_EMBED)
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="norm_score", ascending=False)
    
    # 4. Filter
    if NORM_THRESHOLD is not None: 
        df = df[df['norm_score'] >= float(NORM_THRESHOLD)]
        
    # Check if the query already exists in the database
    query_in_db = query_name in db_headers
    
    # Remove the query from the database hits to prevent duplicates in the FASTA
    df = df[df['header'] != query_name]
    
    if TOP_K is not None:
        # If in DB, we want TOP_K total sequences in the FASTA (1 pinned query + TOP_K-1 hits)
        # If not in DB, we want TOP_K+1 total sequences (1 pinned query + TOP_K hits)
        limit = int(TOP_K) - 1 if query_in_db else int(TOP_K)
        limit = max(0, limit)
        
        if len(df) > limit:
            df = df.head(limit)
    
    # Add dummy row for Query (Ensures it appears at the top of the text report)
    q_row = pd.DataFrame([{"index": -1, "header": f"(Query) {query_name}", "raw_score": 0.0, "norm_score": 99.9, "length": len(query_emb), "seq_len": len(query_emb), "aln_len": len(query_emb)}])
    df = pd.concat([q_row, df], ignore_index=True)
    
    # Use custom name if provided, otherwise fallback to sanitized query name
    if OUTPUT_NAME and str(OUTPUT_NAME).strip():
        base_filename = str(OUTPUT_NAME).strip()
    else:
        base_filename = re.sub(r'[^a-zA-Z0-9]', '_', query_name)[:20]
        
    save_results(df, (query_name, len(query_emb)), len(db_headers), seq_lookup, base_filename, query_seq_str, NORM_MODE, gap_p)
