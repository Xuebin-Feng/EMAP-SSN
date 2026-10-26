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
File: Embedding_SSEARCH.py
===================================
Similar to the NCBI's traditional SSEARCH program, this tool performs a rigorous one-vs-all database search. However, instead of using a standard amino acid substitution matrix (like BLOSUM62), it aligns sequences based on the high-dimensional structural similarity of their Protein Language Model (pLM) embeddings.

This allows you to find remote homologs that share structural similarities even if their literal sequence identity has degraded entirely.

How it Works:
1. Target Database: You select a complete metadata-first embedding database (.h5) to search against.
2. Query Input: You can either type the header of a stored sequence, OR enable 'Manual Query Seq' and paste a new raw amino acid sequence into the 'Query Sequence' box.
3. Inference: The script calculates the structural embedding for your query.
4. Scanning: Using parallel CPU workers, it scans your query against every sequence in the database using either Local (Smith-Waterman) or Global (Needleman-Wunsch) dynamic programming.
5. Scoring: The raw alignment scores are normalized (to prevent bias toward excessively long sequences) and ranked.
6. Identity: Along the same alignment path, identical standard amino acids are counted and divided by the alignment length: internal gaps included for local alignments, and the full length including end gaps for global alignments.

Outputs:
The script generates report files in the configured report directory:
- Report_<Query>.txt: A human-readable text file showing the ranked hits, their normalized scores, raw scores, alignment lengths, and percent identities.
- Hits_<Query>.fasta: A clean FASTA file containing the sequences of all your top hits, ordered strictly by rank, with your query sequence pinned to the very top. This file is perfectly formatted to be immediately dropped into an MSA tool!

Key Parameters:
- Manual Query Seq: When enabled, the script ignores stored-header lookup and sanitizes the supplied Query Sequence in memory.
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
import math
from tools.tool_helpers.Tool_Pipeline import SearchPlan, SearchSelector, SearchTiming, stratified
from utilities import Benchmark_Record
from utilities import Hardware_Acceleration as Hardware_Utils
from utilities.Network_Kernels import search_batch_scores, search_pair_scores
from Embedding_Alignment_Engine import (
    BF16ValidationIntegrityError,
    EmbeddingTileStore,
    bf16_accelerator_support,
    bf16_validation_notice,
    compare_bf16_metric_results,
    compute_score_matrix_torch as _shared_score_matrix,
    cuda_matmul_precision,
    cuda_memory_plan,
    estimate_fixed_query_cuda_working_set,
    fixed_query_block_ids,
    format_bf16_validation_report,
    is_nvidia_cuda,
    normalize_precision_setting,
    precision_element_bytes,
    run_fixed_query_cuda_pipeline,
)
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from multiprocessing import set_start_method, get_context
from tqdm import tqdm

from utilities.Sequence_Utils import (
    IDENTITY_RESIDUES,
    identity_definition,
    sanitize_header,
    sanitize_sequence,
)
from utilities.HDF5_Storage import (
    dtype_for_saving_mode,
    read_embedding_manifest,
    validate_embedding_array,
)
from tools.tool_helpers.Model_Plugins import find_model_plugin

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_EMBED = "Sample_[E1_RA]_embeddings.h5"

# QUERY SETTINGS
QUERY_HEADER = "Query_Header"
MANUAL_QUERY_SEQ = False
QUERY_SEQUENCE = "" # Used only when MANUAL_QUERY_SEQ is enabled.
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
DEVICE_SELECTION = "auto"
ACCELERATOR_PRECISION = "automatic_32bit"
ACCELERATOR_LANES = "auto"
TF32_SEARCH_MIN_TARGETS = 4096
from tools.tool_helpers.Tool_Pipeline import (
    inherited_settings_path,
    load_tool_settings,
    project_directory_defaults,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
REPORT_DIR = _DEFAULT_DIRECTORIES["REPORT_DIR"]
GENERATE_FASTA = False

# --- JSON Settings Override ---
import json
import ast

SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "tools_settings.json")

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


def resolve_manual_query_sequence(enabled, sequence):
    """Return a canonical manual query only when its switch is enabled."""
    if not enabled:
        return ""
    cleaned_sequence, _, _ = sanitize_sequence(str(sequence or ""))
    if not cleaned_sequence:
        raise ValueError("QUERY_SEQUENCE is empty after sanitization.")
    return cleaned_sequence


ALIGNMENT_MODES = ("global", "local")
NORMALIZATION_MODES = ("alignment_length", "shorter_sequence", "longer_sequence", "average_sequence")


def _unknown_norm_mode(norm_mode):
    return ValueError(
        f"Unknown NORM_MODE {norm_mode!r}; choose alignment_length, "
        "shorter_sequence, longer_sequence, or average_sequence."
    )


def validate_score_normalization(alignment_mode, norm_mode):
    """Reject unknown modes, and normalizing local search scores by their own
    alignment length.

    The Tools window removes this option for local searches: dividing a local
    score by its own path length rewards short local matches however little of
    either sequence they cover.
    """
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(
            f"Unknown ALIGNMENT_MODE {alignment_mode!r}; choose global or local."
        )
    if norm_mode not in NORMALIZATION_MODES:
        raise _unknown_norm_mode(norm_mode)
    if alignment_mode == "local" and norm_mode == "alignment_length":
        raise ValueError(
            "NORM_MODE alignment_length is unavailable for local alignments; "
            "choose shorter_sequence, longer_sequence, or average_sequence."
        )


def filter_ranked_hits(df, query_name, manual_query_enabled, top_k):
    """Apply self-hit removal and hit limits using the explicit query source."""
    query_from_database = not bool(manual_query_enabled)
    if query_from_database:
        df = df[df['header'] != query_name]

    if top_k is not None:
        # Stored queries occupy one of TOP_K output slots. Manual queries are
        # pinned separately and retain TOP_K database hits, including a
        # same-header record.
        limit = int(top_k) - 1 if query_from_database else int(top_k)
        df = df.head(max(0, limit))

    return df


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
    return _shared_score_matrix(emb_i, emb_j, device, precision="float32")

# Residues outside IDENTITY_RESIDUES receive different query and target codes,
# so X, B, Z, J, U and O never pair as identical in the identity kernels.
QUERY_UNMATCHED_CODE = 0
TARGET_UNMATCHED_CODE = 1
_IDENTITY_RESIDUE_MASK = np.zeros(256, dtype=bool)
_IDENTITY_RESIDUE_MASK[
    np.frombuffer(IDENTITY_RESIDUES.encode("ascii"), dtype=np.uint8)
] = True


def encode_identity_residues(sequence, unmatched_code):
    """Return one residue code per position for the identity kernels."""
    codes = np.frombuffer(
        str(sequence).encode("ascii", errors="replace"),
        dtype=np.uint8,
    ).copy()
    codes[~_IDENTITY_RESIDUE_MASK[codes]] = unmatched_code
    return codes


def normalization_length(align_len, len_i, len_j, mode):
    """Return the length a NORM_MODE divides raw scores by (the reported length)."""
    if mode == "alignment_length": return align_len
    elif mode == "shorter_sequence": return min(len_i, len_j)
    elif mode == "longer_sequence": return max(len_i, len_j)
    elif mode == "average_sequence": return (len_i + len_j) / 2.0
    raise _unknown_norm_mode(mode)


def normalize_score(raw_score, align_len, len_i, len_j, mode):
    denom = normalization_length(align_len, len_i, len_j, mode)
    return raw_score / denom if denom > 0 else 0.0

# --- HDF5 MULTIPROCESSING INITIALIZATION ---
worker_hf = None
worker_device = None

def init_worker(h5_path):
    global worker_hf, worker_device
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)
    worker_device = torch.device("cpu")


def _search_row(index, header, len_q, len_t, norm_mode, raw, path_len, identities):
    """One target's result row from its DP outputs (Python numbers) and lengths."""
    norm = normalize_score(raw, path_len, len_q, len_t, norm_mode)
    # Both kernels report the length of the selected path: internal gaps only
    # for local alignments and every column, end gaps included, for global.
    identity = 100.0 * int(identities) / path_len if path_len > 0 else 0.0
    eff_len = normalization_length(path_len, len_q, len_t, norm_mode)

    return {
        "index": index,
        "header": header,
        "raw_score": raw,
        "norm_score": norm,
        "length": eff_len,
        "seq_len": len_t,
        "aln_len": path_len,
        "identity": identity,
    }


def _mismatched_matrix_error(header, shape, query_residues, target_residues):
    return ValueError(
        f"Score matrix for '{header}' has shape {shape}, but the query "
        f"and target sequences have {query_residues} and "
        f"{target_residues} residues."
    )


def finish_search(args):
    (
        idx,
        header,
        len_q,
        len_t,
        mode,
        gap,
        norm_mode,
        mat,
        query_seq,
        target_seq,
    ) = args
    query_codes = encode_identity_residues(query_seq, QUERY_UNMATCHED_CODE)
    target_codes = encode_identity_residues(target_seq, TARGET_UNMATCHED_CODE)
    if mat.shape != (len(query_codes), len(target_codes)):
        raise _mismatched_matrix_error(
            header, mat.shape, len(query_codes), len(target_codes)
        )
    raw, path_len, identities = search_pair_scores(
        "local" if mode == "local" else "global", mat, gap, query_codes, target_codes
    )
    return _search_row(idx, header, len_q, len_t, norm_mode, raw, path_len, identities)


def finish_search_batch(tasks, target_lengths, matrices):
    """Result rows for one padded microbatch, equal to ``finish_search`` on each target.

    ``tasks`` are the search tasks of the microbatch's targets, ``target_lengths`` their
    embedding lengths and ``matrices`` the ``(batch, query rows, padded columns)`` score
    matrices. One call replaces a call per target: the finite check, the residue codes
    and the dynamic programming run once for the whole microbatch.
    """
    if not np.isfinite(matrices).all():
        raise FloatingPointError(
            "Fixed-query accelerator scoring produced non-finite values."
        )
    first = tasks[0]
    mode, gap, norm_mode, query_seq = first[4], first[5], first[6], first[7]
    if any(
        (task[4], task[5], task[6]) != (mode, gap, norm_mode) or task[7] != query_seq
        for task in tasks
    ):
        # A microbatch of searches that differ in query or settings: one at a time.
        return [
            finish_search((
                task[0], task[1], matrices.shape[1], int(length), task[4], task[5], task[6],
                matrices[offset, :, :int(length)], task[7], task[8],
            ))
            for offset, (task, length) in enumerate(zip(tasks, target_lengths))
        ]
    query_codes = encode_identity_residues(query_seq, QUERY_UNMATCHED_CODE)
    target_codes = np.zeros((len(tasks), matrices.shape[2]), dtype=np.uint8)
    for offset, (task, length) in enumerate(zip(tasks, target_lengths)):
        codes = encode_identity_residues(task[8], TARGET_UNMATCHED_CODE)
        if matrices.shape[1] != len(query_codes) or int(length) != len(codes):
            raise _mismatched_matrix_error(
                task[1], (matrices.shape[1], int(length)), len(query_codes), len(codes)
            )
        target_codes[offset, :len(codes)] = codes
    scores, path_lengths, identities = search_batch_scores(
        "local" if mode == "local" else "global",
        matrices,
        target_lengths,
        gap,
        query_codes,
        target_codes,
    )
    return [
        _search_row(
            task[0], task[1], matrices.shape[1], int(length), norm_mode, raw, path_len, identity
        )
        for task, length, raw, path_len, identity in zip(
            tasks, target_lengths, scores.tolist(), path_lengths.tolist(), identities.tolist()
        )
    ]


def search_cpu_worker(args):
    idx, header, safe_h, q_emb, mode, gap, norm_mode, q_seq, t_seq = args
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
            q_seq,
            t_seq,
        )
    )


accelerator_thread_state = threading.local()


def _device_type(device):
    return getattr(device, "type", str(device).split(":", 1)[0])


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
    (
        idx,
        header,
        q_emb,
        t_emb,
        mode,
        gap,
        norm_mode,
        device,
        precision,
        q_seq,
        t_seq,
    ) = args
    with torch.inference_mode():
        with _stream_context(device):
            mat = _shared_score_matrix(
                q_emb, t_emb, device, precision=precision
            )
    return (
        idx,
        header,
        q_emb.shape[0],
        t_emb.shape[0],
        mode,
        gap,
        norm_mode,
        mat,
        q_seq,
        t_seq,
    )


def _run_accelerated_search(
    tasks,
    workers,
    input_h5,
    device,
    lanes,
    show_progress,
    precision="float32",
    timing=None,
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

    with cuda_matmul_precision(precision), \
            h5py.File(input_h5, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=lanes,
                thread_name_prefix="search-accelerator",
            ) as accelerator_executor, \
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="search-cpu",
            ) as cpu_executor, \
            progress_context as progress:
        if timing is not None:
            task = tasks[0]
            def warm_lane():
                with _stream_context(device):
                    _warm_search_kernel(task, device, precision)
            timing.warm_executor(accelerator_executor, lanes, warm_lane)
            timing.warm_executor(cpu_executor, workers)
            Hardware_Utils.synchronize_device(device)
            timing.start()
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
                        q_seq,
                        t_seq,
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
                            precision,
                            q_seq,
                            t_seq,
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

        if timing is not None:
            Hardware_Utils.synchronize_device(device)
            timing.stop()
    if timing is not None:
        timing.finish()
    return results


def _cpu_search_with_handle(task, hf):
    idx, header, safe_h, query, mode, gap, norm_mode, q_seq, t_seq = task
    target = hf["embeddings"][safe_h][:]
    matrix = compute_score_matrix_torch(query, target, torch.device("cpu"))
    return finish_search((idx, header, query.shape[0], target.shape[0],
                          mode, gap, norm_mode, matrix, q_seq, t_seq))


def _warm_search_kernel(task, device, precision="float32"):
    _idx, _header, _safe_h, query, mode, gap, norm_mode, q_seq, _t_seq = task
    small = np.asarray(query[:64])
    residues = q_seq[:len(small)]
    matrix = _shared_score_matrix(small, small, device, precision=precision)
    finish_search((0, "warmup", len(small), len(small), mode, gap, norm_mode,
                   matrix, residues, residues))


_WARM_ROWS = 64


def _warm_spec(task):
    """What a pool worker needs to warm the search kernels, in a few bytes.

    The timing trial's pool once received the whole first task, query embedding
    included (hundreds of KiB), in its initargs. A spawned worker reads that
    pickle only after it has imported this module (about 2.3 s), and the
    parent cannot start the next worker until the write completes, so the
    workers started one after another and the trial reported a start-up of
    seconds per worker. A real run's pool starts in parallel.
    """
    query = np.asarray(task[3])
    return (int(query.shape[1]), query.dtype.str, task[4], float(task[5]), task[6])


def _warm_task(spec):
    """A small stand-in task with the dtypes, mode, gap and normalization of ``spec``."""
    features, dtype, mode, gap, norm_mode = spec
    rng = np.random.default_rng(42)
    query = rng.standard_normal((_WARM_ROWS, features)).astype(np.dtype(dtype))
    residues = ("ACDEFGHIKLMNPQRSTVWY" * _WARM_ROWS)[:_WARM_ROWS]
    return (0, "warmup", "warmup", query, mode, gap, norm_mode, residues, residues)


# The CPU search's worker pool always starts its processes with spawn, as the script itself asks for and as
# Windows and macOS do anyway. A program that imports this module on Linux would otherwise fork, and a fork of
# a process that has run torch's CPU kernels (an OpenMP thread pool) or used CUDA hangs in the child's first
# parallel kernel: the workers never reported ready.
_POOL_CONTEXT = get_context("spawn")
Pool = _POOL_CONTEXT.Pool


def _init_search_pool(input_h5, warm_spec, ready):
    try:
        init_worker(input_h5)
        _warm_search_kernel(_warm_task(warm_spec), torch.device("cpu"))
        ready.put(None)
    except Exception as error:
        ready.put(f"{type(error).__name__}: {error}")


def _run_serial_search(tasks, input_h5, show_progress=False, timing=None):
    results = []
    with h5py.File(input_h5, "r", libver="latest", swmr=True) as hf:
        if timing is not None:
            _warm_search_kernel(tasks[0], torch.device("cpu"))
            timing.start()
        for task in tqdm(tasks, desc="Search (serial CPU)", disable=not show_progress):
            results.append(_cpu_search_with_handle(task, hf))
        if timing is not None:
            timing.stop()
    if timing is not None:
        timing.finish()
    return results


def _run_cpu_search(tasks, workers, input_h5, show_progress=True, timing=None):
    results = []
    workers = max(1, int(workers))
    ready = _POOL_CONTEXT.Queue() if timing is not None else None
    try:
        with Pool(
            processes=workers,
            initializer=_init_search_pool if timing is not None else init_worker,
            initargs=(input_h5, _warm_spec(tasks[0]), ready) if timing is not None else (input_h5,),
        ) as pool:
            if timing is not None:
                for _ in range(workers):
                    error = ready.get(timeout=120)
                    if error is not None:
                        raise RuntimeError(error)
                timing.start()
            chunksize = max(1, min(50, math.ceil(len(tasks) / (4 * workers))))
            iterator = pool.imap_unordered(search_cpu_worker, tasks, chunksize=chunksize)
            for result in tqdm(iterator, total=len(tasks), disable=not show_progress):
                results.append(result)
            if timing is not None:
                timing.stop()
        if timing is not None:
            timing.finish()
    finally:
        if ready is not None:
            ready.close()
            ready.join_thread()
    return results


def _bf16_search_validation_sample(tasks, lengths):
    return stratified(tasks, lengths, 2048)


def _search_results_equivalent(
    fp32_results,
    candidate_results,
    tolerance=1e-3,
    candidate_label="candidate",
):
    fp32 = {int(result["index"]): result for result in fp32_results}
    candidate_map = {
        int(result["index"]): result for result in candidate_results
    }
    if fp32.keys() != candidate_map.keys():
        return False, "target identities differ"
    for index, baseline in fp32.items():
        candidate = candidate_map[index]
        if int(baseline["aln_len"]) != int(candidate["aln_len"]):
            return False, f"alignment length changed for target {index}"
        values = (
            float(candidate["raw_score"]),
            float(candidate["norm_score"]),
        )
        baseline_values = (
            float(baseline["raw_score"]),
            float(baseline["norm_score"]),
        )
        if not all(np.isfinite(value) for value in baseline_values + values):
            return False, f"non-finite {candidate_label} result for target {index}"
        denominator = max(1, int(baseline["aln_len"]))
        drift = abs(
            float(baseline["raw_score"]) - float(candidate["raw_score"])
        ) / denominator
        if drift > tolerance:
            return False, f"per-aligned-residue drift exceeded {tolerance:g}"
    return True, "alignment lengths and per-residue scores passed"


def _search_bf16_validation(
    fp32_results,
    candidate_results,
):
    """Build an informational BF16 report for SSEARCH's selected mode."""
    fp32_results = list(fp32_results)
    candidate_results = list(candidate_results)
    try:
        fp32 = {int(result["index"]): result for result in fp32_results}
        candidate_map = {
            int(result["index"]): result for result in candidate_results
        }
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise BF16ValidationIntegrityError(
            f"invalid SSEARCH identity: {error}"
        ) from error
    if (
        len(fp32) != len(fp32_results)
        or len(candidate_map) != len(candidate_results)
    ):
        raise BF16ValidationIntegrityError(
            "duplicate target identities in validation results"
        )
    if fp32.keys() != candidate_map.keys():
        raise BF16ValidationIntegrityError("target identities differ")

    for index, baseline in fp32.items():
        candidate = candidate_map[index]
        try:
            values = (
                float(baseline["raw_score"]),
                float(baseline["norm_score"]),
                float(candidate["raw_score"]),
                float(candidate["norm_score"]),
            )
        except (KeyError, OverflowError, TypeError, ValueError) as error:
            raise BF16ValidationIntegrityError(
                f"invalid SSEARCH result for target {index}: {error}"
            ) from error
        if not all(np.isfinite(value) for value in values):
            raise BF16ValidationIntegrityError(
                f"non-finite BF16 result for target {index}"
            )

    try:
        baseline_metrics = {
            index: {"selected": (result["raw_score"], result["aln_len"])}
            for index, result in fp32.items()
        }
        candidate_metrics = {
            index: {"selected": (result["raw_score"], result["aln_len"])}
            for index, result in candidate_map.items()
        }
    except (KeyError, TypeError) as error:
        raise BF16ValidationIntegrityError(
            f"invalid SSEARCH result: {error}"
        ) from error
    return compare_bf16_metric_results(
        baseline_metrics,
        candidate_metrics,
        identity_label="target",
        candidate_label="BF16",
    )


active_search_hardware = {
    "device": "cpu",
    "plan": "scalar",
    "precision": "ieee_fp32",
    "lanes": 1,
    "microbatch_mib": None,
    "tuning_seconds": 0.0, "validation_seconds": 0.0,
    "sampled_targets": 0, "reused_targets": 0,
    "estimated_remaining": 0.0, "benchmark_notes": "",
}


def _execute_search_plan(
    plan,
    tasks,
    workers,
    input_h5,
    store,
    lengths,
    query_embedding,
    show_progress,
    timing=None,
):
    candidate, variant, precision, lanes = plan
    if not tasks:
        return []
    if candidate.is_cpu:
        if variant == "serial":
            return _run_serial_search(tasks, input_h5, show_progress, timing)
        return _run_cpu_search(tasks, workers, input_h5, show_progress, timing)
    if timing is not None and variant == "tiled":
        _warm_search_kernel(tasks[0], candidate.device, precision)
        Hardware_Utils.synchronize_device(candidate)
    if variant == "tiled":
        progress = tqdm(total=len(tasks), desc="Search (tiled CUDA)") if show_progress else None
        try:
            return run_fixed_query_cuda_pipeline(
                tasks,
                query_embedding=query_embedding,
                store=store,
                lengths=lengths,
                device=candidate.device,
                workers=workers,
                lanes=lanes,
                batch_alignment_callback=finish_search_batch,
                precision=precision,
                progress=progress,
                timing=timing,
            )
        finally:
            if progress is not None:
                progress.close()
    return _run_accelerated_search(
        tasks,
        workers,
        input_h5,
        candidate.device,
        lanes,
        show_progress,
        precision=precision,
        timing=timing,
    )


# Where a backend is expected to search fastest, as (variant, lanes) in the order to time them. The
# tuner times these right after a device's first scalar trial and before the slow-to-start CPU pool;
# the first of a variant it has not measured yet is guaranteed a measurement whatever the budget says.
# Tiled searches exist only on CUDA/ROCm here; MPS has one lane and nothing else to try.
EXPECTED_BEST_VARIANTS = {
    "cuda": (("tiled", 1), ("tiled", 2)),
    "xpu": (("scalar", 2),),
}
# A tiled plan's timing sample (see _tiled_timing_sample) holds TILED_SAMPLE_TARGETS targets (at most a
# TILED_SAMPLE_JOB_SHARE-th of the search, at least TILED_SAMPLE_MINIMUM), in TILED_SAMPLE_WINDOWS windows.
# The pipeline overlaps reading, copying up and scoring only once a pass is long enough: a sample of 300
# targets runs at about 1.7 times the cost per target of a whole search, one of 1,500 at about 1.2 times.
# It is timed after an untimed pass over TILED_WARMUP_TARGETS of them, in TILED_TIMED_PASSES passes, and
# again on every TILED_THIN_STEP-th target, so that the fixed cost of a pass can be taken out.
TILED_SAMPLE_TARGETS = 1500
TILED_SAMPLE_JOB_SHARE = 4
TILED_SAMPLE_MINIMUM = 64
TILED_SAMPLE_WINDOWS = 4
TILED_WARMUP_TARGETS = 48
TILED_TIMED_PASSES = 2
TILED_THIN_STEP = 3


def _expected_best_ladders(candidate, workers):
    """``{variant: [lanes, ...]}`` that ``candidate`` is expected to be fastest with."""
    usable = set(_lane_candidates(candidate.device, workers))
    manual_lanes = _configured_lanes(candidate.device, workers)
    ladders = {}
    for variant, lanes in EXPECTED_BEST_VARIANTS.get(candidate.backend, ()):
        if manual_lanes:
            lanes = manual_lanes
        elif lanes not in usable:
            continue
        ladder = ladders.setdefault(variant, [])
        if lanes not in ladder:
            ladder.append(lanes)
    return ladders


def _tiled_timing_sample(tasks, lengths, store, query_embedding, device, lanes, precision, count):
    """Targets that stand for a whole tiled search when it is timed on only a few.

    Two costs of a tiled search do not scale with the targets timed, so the shared
    timing sample, spread over the whole job, made tiled look 2-3 times slower than it is:
    - Each tile block is read, uploaded, drained and its cache cleared once. On a
      44,127-target job the 39-target sample touched all 16 blocks (one per 2.4 targets).
    - A microbatch holds consecutive targets in length order (at most 15% padding)
      and costs a launch and a wait. Spread over the lengths, a 200-target sample made
      11 microbatches of 18 targets where a real block makes 13 of about 220.
    So the sample comes from one block, the one whose mean length is nearest the job's
    (with at least ``count`` targets, else the largest), as a few windows of consecutive
    targets in length order at evenly spaced length quantiles: they still cover the
    lengths, and each window fills a microbatch. None when the block layout cannot be
    read: the shared sample is used.
    """
    try:
        element_bytes = precision_element_bytes(precision)
        plan = cuda_memory_plan(device, lanes=lanes, compute_element_bytes=element_bytes)
        block_ids = fixed_query_block_ids(store, query_embedding, plan, element_bytes)
        blocks = {}
        for task in tasks:
            blocks.setdefault(int(block_ids[int(task[0])]), []).append(task)
    except Exception:  # The sample is advisory: trouble reading the device only costs it its own sample.
        return None
    mean_length = sum(lengths[int(task[0])] for task in tasks) / len(tasks)

    def distance(block):
        members = blocks[block]
        mean = sum(lengths[int(task[0])] for task in members) / len(members)
        return abs(mean - mean_length), block

    roomy = [block for block, members in blocks.items() if len(members) >= count]
    block = min(roomy, key=distance) if roomy else max(blocks, key=lambda b: (len(blocks[b]), -b))
    members = sorted(blocks[block], key=lambda task: (lengths[int(task[0])], int(task[0])))
    count = min(count, len(members))
    windows = max(1, min(TILED_SAMPLE_WINDOWS, count))
    size, extra = divmod(count, windows)
    widths = [size + (1 if window < extra else 0) for window in range(windows)]
    sample, cursor = [], 0
    for window, width in enumerate(widths):
        centre = (window + 0.5) * len(members) / windows
        # Centred on its quantile, never overlapping the window before, always leaving room for the rest.
        first = min(max(int(centre - width / 2), cursor), len(members) - sum(widths[window:]))
        sample.extend(members[first:first + width])
        cursor = first + width
    return sample


def _select_search_plans(tasks, workers, input_h5, store, lengths, query_embedding, selection_started=None):
    selection_started = time.perf_counter() if selection_started is None else selection_started
    precision_setting = normalize_precision_setting(ACCELERATOR_PRECISION)
    candidates = Hardware_Utils.get_available_devices()
    manual = Hardware_Utils.resolve_device_selection(DEVICE_SELECTION, candidates)
    if manual is not None:
        candidates = [manual]
    if precision_setting == "tf32":
        candidates = [c for c in candidates if c.backend == "cuda" and is_nvidia_cuda(c.device)]
    elif precision_setting == "bf16":
        candidates = [c for c in candidates if not c.is_cpu and bf16_accelerator_support(c.device)[0]]
    if not candidates:
        raise RuntimeError(f"No selected device supports SSEARCH {precision_setting}.")

    def execute(plan, sample, timing=None):
        if timing is None or plan.variant != "tiled":
            return _execute_search_plan(plan.execution, sample, workers, input_h5,
                                        store, lengths, query_embedding, False, timing=timing)
        # A tiled pipeline's first pass pays one-time costs (pinned buffers, batched-matmul
        # shapes, streams) that a real run pays once for the whole job. They would be
        # extrapolated over every target, so an untimed pass over some of the same targets
        # takes them out of the timed ones. Its results are dropped. Interference from other
        # programs only ever slows a pass, so of two timed passes the faster counts.
        _execute_search_plan(plan.execution, stratified(sample, lengths, TILED_WARMUP_TARGETS),
                             workers, input_h5, store, lengths, query_embedding, False)
        passes = []
        for _ in range(TILED_TIMED_PASSES):
            attempt = SearchTiming()
            passes.append((attempt, _execute_search_plan(plan.execution, sample, workers, input_h5, store,
                                                         lengths, query_embedding, False, timing=attempt)))
        best, payload = min(passes, key=lambda entry: entry[0].processing)
        # Every pass also pays a fixed cost (the reader starting and the pipeline filling and draining) that
        # a real run pays once, not per target. A pass over every third target of the sample times it: the
        # difference between the two passes, over the cells they differ by, is the cost per cell without it.
        thin = sample[::TILED_THIN_STEP]
        if len(thin) >= 8:
            thin_best = min(
                (_timed_tiled_pass(plan, thin) for _ in range(TILED_TIMED_PASSES)),
                key=lambda attempt: attempt.processing,
            )
            cells = sum(selector.costs[int(task[0])] for task in sample)
            thin_cells = sum(selector.costs[int(task[0])] for task in thin)
            if cells > thin_cells and best.processing > thin_best.processing:
                per_cell = (best.processing - thin_best.processing) / (cells - thin_cells)
                best.processing = min(best.processing, max(0.5 * best.processing, per_cell * cells))
        timing.setup, timing.processing, timing.shutdown = best.setup, best.processing, best.shutdown
        return payload

    def _timed_tiled_pass(plan, subset):
        attempt = SearchTiming()
        _execute_search_plan(plan.execution, subset, workers, input_h5, store, lengths,
                             query_embedding, False, timing=attempt)
        return attempt

    selector = SearchSelector(tasks, lengths, len(query_embedding), execute)
    selector.started = selection_started
    tiled_samples = {}

    def own_sample(plan):
        """A tiled plan's timing sample (cached per device, lanes and compute width)."""
        if plan.variant != "tiled":
            return None
        element_bytes = precision_element_bytes(plan.precision)
        key = (plan.candidate.spec, plan.lanes, element_bytes)
        if key not in tiled_samples:
            count = min(TILED_SAMPLE_TARGETS, max(TILED_SAMPLE_MINIMUM, len(tasks) // TILED_SAMPLE_JOB_SHARE))
            tiled_samples[key] = _tiled_timing_sample(
                tasks, lengths, store, query_embedding, plan.candidate.device,
                plan.lanes, plan.precision, count)
        return tiled_samples[key]

    selector.sampler = own_sample
    precision = precision_setting if precision_setting in {"tf32", "bf16"} else "float32"
    cpu = next((c for c in candidates if c.is_cpu), None)
    serial = SearchPlan(cpu, "serial", "float32") if cpu is not None else None
    tiny = False
    if serial is not None:
        tiny = selector.baseline(serial)
    if serial is not None and not tiny and serial.observations:
        selector.trial(serial)
    print(f"[Hardware] Storage dtypes: {', '.join(sorted({str(dtype) for dtype in store.dtypes}))}")
    print(f"[Hardware] SSEARCH: {len(tasks)} targets; query={len(query_embedding)}; "
          f"features={store.feature_dimension}; sample={len(selector.sample)}; "
          f"soft tuning budget={selector.budget:.2f}s (initialization may exceed it).")

    validated = set()
    bf16_baselines = {}
    validation_sample = _bf16_search_validation_sample(tasks, lengths) if precision == "bf16" else []

    def feasible(plan):
        if plan.variant != "tiled":
            return True
        estimate = estimate_fixed_query_cuda_working_set(
            tasks, query_embedding=query_embedding, store=store, lengths=lengths,
            device=plan.candidate.device, lanes=plan.lanes, precision=plan.precision,
        )
        if not estimate.feasible:
            selector.skip(plan, f"unmeasured: {estimate.reason}")
        return estimate.feasible

    def validate_bf16(plan):
        key = (plan.candidate.spec, plan.variant)
        if precision != "bf16" or key in validated:
            return True
        started = time.perf_counter()
        try:
            candidate = plan.candidate
            if candidate.spec not in bf16_baselines:
                print(bf16_validation_notice(tool_name="Embedding SSEARCH",
                                             sample_count=len(validation_sample)))
                baseline = SearchPlan(candidate, "scalar", "float32", 1)
                bf16_baselines[candidate.spec] = execute(baseline, validation_sample)
            validation_plan = SearchPlan(candidate, plan.variant, "bf16", plan.lanes)
            payload = execute(validation_plan, validation_sample)
            report = _search_bf16_validation(bf16_baselines[candidate.spec], payload)
            print(format_bf16_validation_report(
                report, context=f"tool=Embedding SSEARCH; device={candidate.display_name}; "
                f"backend={candidate.backend}; variant={plan.variant}", identity_label="target"))
            validated.add(key)
            # Validation results belong to this exact plan and can also be reused.
            plan.results.update({int(row["index"]): row for row in payload})
            return True
        except Exception as error:
            selector.skip(plan, f"BF16 validation failed: {type(error).__name__}: {error}")
            return False
        finally:
            selector.excluded += time.perf_counter() - started

    def trial(plan, required=False, guaranteed=False):
        if not required and not selector.allowed(plan, guaranteed):
            return False
        try:
            if not feasible(plan) or not validate_bf16(plan):
                return False
            return selector.trial(plan, required=True)
        except Exception as error:
            selector.skip(plan, f"failed: {type(error).__name__}: {error}")
            return False
        finally:
            Hardware_Utils.release_device_cache(plan.candidate)

    def measured(candidate, variant):
        return sorted((p for p in selector.plans if p.candidate.spec == candidate.spec
                       and p.variant == variant and p.precision == precision), key=lambda p: p.lanes)

    def climb(candidate, variant, ladder, guarantee=False):
        """Time ``variant`` at each lane count of ``ladder`` while doubling the lanes still gains 10%."""
        previous = None
        for lanes in ladder:
            plan = next((p for p in measured(candidate, variant) if p.lanes == lanes), None)
            if plan is None:
                plan = SearchPlan(candidate, variant, precision, lanes)
                if not trial(plan, guaranteed=guarantee and not measured(candidate, variant)):
                    break
            if previous is not None and plan.predicted(selector.costs) > previous.predicted(selector.costs) * 0.90:
                break
            previous = plan

    if not tiny:
        accelerators = [candidate for candidate in candidates if not candidate.is_cpu]
        # 1. Each accelerator's plain scalar plan, one lane.
        for candidate in accelerators:
            lanes = _configured_lanes(candidate.device, workers) or 1
            trial(SearchPlan(candidate, "scalar", precision, lanes), required=not selector.ranked())
        # 2. The variants each is expected to be fastest with, before the CPU pool, whose start-up
        #    can use up the budget. The first measurement of a variant is guaranteed.
        for candidate in accelerators:
            for variant, ladder in _expected_best_ladders(candidate, workers).items():
                climb(candidate, variant, ladder, guarantee=True)
        # 3. The CPU pool.
        if cpu is not None and workers > 1:
            trial(SearchPlan(cpu, "pool", "float32"), required=not selector.ranked())

        # 4. The rest of each accelerator's lane counts, while the budget lasts.
        for candidate in accelerators:
            device_plans = [p for p in selector.plans if p.candidate.spec == candidate.spec]
            best = selector.ranked()
            if not device_plans or not best:
                continue
            if min(p.predicted(selector.costs) for p in device_plans) > best[0].predicted(selector.costs) * 1.20:
                selector.skip(device_plans[0], "refinement skipped: more than 20% behind")
                continue
            manual_lanes = _configured_lanes(candidate.device, workers)
            lanes_to_try = [manual_lanes] if manual_lanes else _lane_candidates(candidate.device, workers)
            for variant in (["scalar", "tiled"] if candidate.backend == "cuda" else ["scalar"]):
                climb(candidate, variant, lanes_to_try)

        # Automatic reduced precision is considered only after FP32 screening.
        if precision_setting == "automatic_32bit" and len(tasks) >= TF32_SEARCH_MIN_TARGETS:
            for baseline in list(selector.ranked()):
                if not is_nvidia_cuda(baseline.candidate.device):
                    continue
                plan = SearchPlan(baseline.candidate, baseline.variant, "tf32", baseline.lanes)
                if not trial(plan):
                    continue
                reference = [row for index, row in baseline.results.items() if index in plan.results]
                equivalent, reason = _search_results_equivalent(reference, list(plan.results.values()))
                fp32_best = min(p.predicted(selector.costs) for p in selector.plans if p.precision == "float32")
                if not equivalent or plan.predicted(selector.costs) * 1.10 > fp32_best:
                    selector.plans.remove(plan)
                    selector.skip(plan, f"TF32 rejected: {reason}; requires 10% advantage")

        finalists = selector.ranked()[:2]
        for _ in range(2):
            for plan in finalists:
                if len(plan.observations) < 3:
                    trial(plan)

    ranked = selector.ranked()
    if not ranked:
        raise RuntimeError("No SSEARCH hardware plan completed successfully.")
    # An automatic IEEE search always has an explicitly labelled last-resort CPU plan.
    if manual is None and precision == "float32" and serial is not None and serial not in ranked:
        ranked.append(serial)
    active_search_hardware.update(
        tuning_seconds=selector.elapsed, validation_seconds=selector.excluded,
        sampled_targets=len(selector.sampled_ids),
        total_targets=len(tasks), benchmark_notes="; ".join(selector.messages),
        benchmark_timings="; ".join(
            f"{p.label}: " + ", ".join(
                f"setup={t.setup:.4f}s processing={t.processing:.4f}s shutdown={t.shutdown:.4f}s"
                for t, _rate in p.observations) for p in selector.plans),
        estimated_remaining=ranked[0].predicted(selector.costs),
    )
    Benchmark_Record.record_search_plans(
        selector.plans, ranked, selector.costs, notes=selector.messages,
        targets=len(tasks), query_length=len(query_embedding),
        tuning_seconds=selector.elapsed, validation_seconds=selector.excluded,
        budget_seconds=selector.budget,
    )
    print(f"[Hardware] Selected {ranked[0].label}; tuning={selector.elapsed:.3f}s; "
          f"required precision checks={selector.excluded:.3f}s; "
          f"reusing {len(ranked[0].results)} targets.")
    return ranked


def process_search_tasks(tasks, workers, input_h5):
    tasks = list(tasks)
    if not tasks:
        return []
    workers = max(1, int(workers))
    selection_started = time.perf_counter()
    store = EmbeddingTileStore(input_h5, [task[2] for task in tasks], host_cache_setting=0)
    lengths = [shape[0] for shape in store.shapes]
    query_embedding = tasks[0][3]
    plans = _select_search_plans(tasks, workers, input_h5, store, lengths, query_embedding,
                                 selection_started=selection_started)
    manual = Hardware_Utils.normalize_device_selection(DEVICE_SELECTION) != "auto"
    last_error = None
    expected = {int(task[0]) for task in tasks}
    for plan in plans:
        candidate, variant, precision, lanes = plan.execution
        try:
            retained = dict(plan.results)
            pending = [task for task in tasks if int(task[0]) not in retained]
            payload = _execute_search_plan(plan.execution, pending, workers, input_h5,
                                           store, lengths, query_embedding, True)
            if len(payload) != len({int(row["index"]) for row in payload}):
                raise RuntimeError("Duplicate SSEARCH result identities")
            if {int(row["index"]) for row in payload} != {int(task[0]) for task in pending}:
                raise RuntimeError("Missing or unexpected SSEARCH result identities")
            retained.update({int(row["index"]): row for row in payload})
            if retained.keys() != expected:
                raise RuntimeError("Incomplete SSEARCH results")
            active_search_hardware.update(
                device=candidate.display_name, plan=variant,
                precision=precision if precision in {"tf32", "bf16"} else "ieee_fp32",
                lanes=lanes, reused_targets=len(plan.results),
                estimated_remaining=plan.predicted({int(t[0]): max(1, len(query_embedding) * lengths[int(t[0])]) for t in tasks}),
                microbatch_mib=(cuda_memory_plan(candidate.device, lanes=lanes,
                    compute_element_bytes=precision_element_bytes(precision)).matrix_bytes / (1024 ** 2)
                    if variant == "tiled" else None),
            )
            return [retained[int(task[0])] for task in tasks]
        except (RuntimeError, NotImplementedError, MemoryError) as error:
            last_error = error
            if manual:
                raise
            print(f"[Hardware] {plan.label} failed: {error}; trying next plan.")
        finally:
            Hardware_Utils.release_device_cache(candidate)
    raise RuntimeError("Every benchmarked SSEARCH plan failed.") from last_error


# --- 5. REPORTING -------------------------------------------------------------
METADATA_REPORT_COLUMNS = [
    "Node ID",
    "Rank",
    "Norm_Score",
    "Raw_Score",
    "Sequence_Length",
    "Alignment_Length",
    "Percent_Identity",
]
METADATA_REPORT_TYPES = [
    "Data Type",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
]


def build_metadata_report_dataframe(rows):
    """Return search hits in the metadata viewer's two-header-row schema."""
    data_rows = [METADATA_REPORT_TYPES]
    data_rows.extend(
        [
            row["Node ID"],
            row["Rank"],
            row["Norm_Score"],
            row["Raw_Score"],
            row["Sequence_Length"],
            row["Alignment_Length"],
            row["Percent_Identity"],
        ]
        for row in rows
    )
    return pd.DataFrame(data_rows, columns=METADATA_REPORT_COLUMNS)


def format_metadata_report_sheet(worksheet):
    """Match docs/metadata_template.xlsx while retaining typed cell values."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    header_fill = PatternFill("solid", fgColor="2C3E50")
    type_fill = PatternFill("solid", fgColor="D5D8DC")
    alternate_fill = PatternFill("solid", fgColor="F7F8F9")
    white_fill = PatternFill("solid", fgColor="FFFFFF")
    border_side = Side(style="thin", color="E2E2E2")
    cell_border = Border(
        left=border_side,
        right=border_side,
        top=border_side,
        bottom=border_side,
    )
    header_font = Font(
        name="Segoe UI",
        size=11,
        bold=True,
        color="FFFFFF",
    )
    type_font = Font(
        name="Segoe UI",
        size=10,
        bold=True,
        italic=True,
        color="1F2328",
    )
    data_font = Font(name="Segoe UI", size=10, color="000000")

    worksheet.row_dimensions[1].height = 28.05
    worksheet.row_dimensions[2].height = 22.05
    worksheet.freeze_panes = "A3"
    worksheet.sheet_view.showGridLines = True

    widths = {
        "A": 80,
        "B": 12,
        "C": 16,
        "D": 16,
        "E": 20,
        "F": 20,
        "G": 20,
    }
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width

    for column_index, cell in enumerate(worksheet[1], start=1):
        cell.fill = header_fill
        cell.font = header_font
        cell.border = cell_border
        cell.alignment = Alignment(
            horizontal="left" if column_index == 1 else "center",
            vertical="center",
        )

    for column_index, cell in enumerate(worksheet[2], start=1):
        cell.fill = type_fill
        cell.font = type_font
        cell.border = cell_border
        cell.alignment = Alignment(
            horizontal="left" if column_index == 1 else "center",
            vertical="center",
        )

    number_formats = {
        2: "0",
        3: "0.000",
        4: "0.0",
        5: "0",
        6: "0",
        7: "0.0",
    }
    for row_index, row in enumerate(
        worksheet.iter_rows(
            min_row=3,
            max_row=worksheet.max_row,
            min_col=1,
            max_col=len(METADATA_REPORT_COLUMNS),
        ),
        start=3,
    ):
        worksheet.row_dimensions[row_index].height = 20
        row_fill = alternate_fill if row_index % 2 == 0 else white_fill
        for column_index, cell in enumerate(row, start=1):
            cell.fill = row_fill
            cell.font = data_font
            cell.border = cell_border
            cell.alignment = Alignment(
                horizontal="left" if column_index == 1 else "right",
                vertical="center",
            )
            if column_index in number_formats:
                cell.number_format = number_formats[column_index]


def format_search_parameters_sheet(worksheet):
    """Keep the secondary parameter sheet readable without changing its data."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    header_fill = PatternFill("solid", fgColor="2C3E50")
    alternate_fill = PatternFill("solid", fgColor="F7F8F9")
    white_fill = PatternFill("solid", fgColor="FFFFFF")
    border_side = Side(style="thin", color="E2E2E2")
    cell_border = Border(
        left=border_side,
        right=border_side,
        top=border_side,
        bottom=border_side,
    )

    worksheet.column_dimensions["A"].width = 32
    worksheet.column_dimensions["B"].width = 80
    worksheet.row_dimensions[1].height = 28.05
    worksheet.freeze_panes = "A2"

    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(
            name="Segoe UI",
            size=11,
            bold=True,
            color="FFFFFF",
        )
        cell.border = cell_border
        cell.alignment = Alignment(horizontal="left", vertical="center")

    for row_index, row in enumerate(
        worksheet.iter_rows(
            min_row=2,
            max_row=worksheet.max_row,
            min_col=1,
            max_col=2,
        ),
        start=2,
    ):
        worksheet.row_dimensions[row_index].height = 20
        row_fill = alternate_fill if row_index % 2 == 0 else white_fill
        for cell in row:
            cell.fill = row_fill
            cell.font = Font(name="Segoe UI", size=10, color="000000")
            cell.border = cell_border
            cell.alignment = Alignment(horizontal="left", vertical="center")


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
    meta_lines.append(f" Identity:    {identity_definition(ALIGNMENT_MODE)}")
    meta_lines.append(f" Filters:     Top_K={TOP_K} | Norm_Threshold={NORM_THRESHOLD}")
    hardware_line = (
        f" Hardware:    {active_search_hardware['device']} | "
        f"{active_search_hardware['plan']} | "
        f"{active_search_hardware['precision']} | "
        f"lanes={active_search_hardware['lanes']}"
    )
    if active_search_hardware["microbatch_mib"] is not None:
        hardware_line += (
            f" | microbatch="
            f"{active_search_hardware['microbatch_mib']:.1f} MiB"
        )
    meta_lines.append(hardware_line)
    meta_lines.append("-" * 80 + "\n")
    
    report_lines = list(meta_lines)
    onscreen_lines = list(meta_lines)
    
    if df.empty:
        report_lines.append("  [No hits found satisfying the criteria]")
        onscreen_lines.append("  [No hits found satisfying the criteria]")
        xlsx_data = []
    else:
        table_hdr = [
            f" {'RANK':<6} | {'NORM-SCR':<9} | {'RAW':<9} | {'SEQ-LEN':<8} | {col_header:<8} | {'IDENT%':<6} | {'HEADER'}",
            f"{'-'*7}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*35}"
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
            identity = float(row['identity'])

            row_line = f" {rank_counter:<6} | {norm_score:<9.3f} | {raw_score:<9.1f} | {seq_len_val:<8} | {aln_len_val:<8} | {identity:<6.1f} | {head}"
            report_lines.append(row_line)
            if printed_hits < 100:
                onscreen_lines.append(row_line)
                printed_hits += 1
                
            xlsx_data.append({
                "Node ID": str(head),
                "Rank": rank_counter,
                "Norm_Score": float(norm_score),
                "Raw_Score": float(raw_score),
                "Sequence_Length": int(seq_len_val),
                "Alignment_Length": int(aln_len_val),
                "Percent_Identity": identity,
            })
            
            rank_counter += 1
            
        if len(df) - 1 > 100:
            onscreen_lines.append(f"\n  [Onscreen display limited to first 100 hits. The full list of {len(df) - 1} hits is stored in the report file.]")
            
    print("\n".join(onscreen_lines))
    
    output_dir = REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    report_text = "\n".join(report_lines)
    with open(os.path.join(output_dir, f"Report_{base_filename}.txt"), "w", encoding="utf-8") as f: f.write(report_text)
    
    # Generate and save Excel Report
    xlsx_path = os.path.join(output_dir, f"Report_{base_filename}.xlsx")
    xlsx_df = build_metadata_report_dataframe(xlsx_data)
        
    meta_data = [
        {"Parameter": "Query Header", "Value": q_head},
        {"Parameter": "Query Length (residues)", "Value": q_len},
        {"Parameter": "Database", "Value": INPUT_EMBED},
        {"Parameter": "Database Size (sequences)", "Value": db_size},
        {"Parameter": "Alignment Mode", "Value": ALIGNMENT_MODE},
        {"Parameter": "Gap Penalty", "Value": gap_p},
        {"Parameter": "Normalization Mode", "Value": norm_mode},
        {"Parameter": "Percent Identity", "Value": identity_definition(ALIGNMENT_MODE)},
        {"Parameter": "Norm Score Cutoff", "Value": NORM_THRESHOLD if NORM_THRESHOLD is not None else "None"},
        {"Parameter": "Top K", "Value": TOP_K if TOP_K is not None else "None"},
        {"Parameter": "Compute Device", "Value": active_search_hardware["device"]},
        {"Parameter": "Compute Plan", "Value": active_search_hardware["plan"]},
        {"Parameter": "Matmul Precision", "Value": active_search_hardware["precision"]},
        {"Parameter": "Accelerator Lanes", "Value": active_search_hardware["lanes"]},
        {"Parameter": "Microbatch Budget (MiB)", "Value": active_search_hardware["microbatch_mib"] if active_search_hardware["microbatch_mib"] is not None else "N/A"},
    ]
    for label, key in (
        ("Tuning Time (s)", "tuning_seconds"),
        ("Precision Validation Time (s)", "validation_seconds"),
        ("Benchmark Sample Targets", "sampled_targets"),
        ("Reused Targets", "reused_targets"),
        ("Estimated Remaining Search Time (s)", "estimated_remaining"),
        ("Benchmark Timings", "benchmark_timings"),
        ("Benchmark Notes", "benchmark_notes"),
    ):
        meta_data.append({"Parameter": label, "Value": active_search_hardware.get(key, "N/A")})
    meta_df = pd.DataFrame(meta_data)
    
    try:
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            xlsx_df.to_excel(writer, sheet_name="Search Results", index=False)
            meta_df.to_excel(writer, sheet_name="Search Parameters", index=False)
            format_metadata_report_sheet(writer.sheets["Search Results"])
            format_search_parameters_sheet(writer.sheets["Search Parameters"])
        print(f"[Export] Excel report saved to: {xlsx_path}")
    except Exception as e:
        print(f"[Warning] Failed to save Excel report: {e}")
    
    if GENERATE_FASTA:
        # OUTPUT FASTA (Ranked with query at the top)
        count = 1 
        with open(os.path.join(output_dir, f"Hits_{base_filename}.fasta"), "w", encoding="utf-8") as f:
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
def main(argv=None):
    global FULL_INPUT_EMBED
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    validate_score_normalization(ALIGNMENT_MODE, NORM_MODE)
    FULL_INPUT_EMBED = os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR else ""
    database = prepare_database_embeddings()
    db_headers = database.headers
    seq_lookup = database.sequence_by_header

    # Process Query Input
    raw_query_name = QUERY_HEADER if QUERY_HEADER else "Manual_Query"
    query_name = sanitize_header(str(raw_query_name))[0] or "Manual_Query"
    manual_query_enabled = bool(MANUAL_QUERY_SEQ)
    query_seq_str = resolve_manual_query_sequence(
        manual_query_enabled,
        QUERY_SEQUENCE,
    )
    query_emb = None
    target_dtype = dtype_for_saving_mode(database.saving_mode)
    cleanup_device = None

    if manual_query_enabled:
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
        # Sequences travel with each task so the alignment DP can count the
        # identical residues on the same path that yields its score and length.
        tasks.append((i, header, header, query_emb, ALIGNMENT_MODE, gap_p,
                      NORM_MODE, query_seq_str, seq_lookup[header]))
        
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
        
    df = filter_ranked_hits(
        df,
        query_name,
        manual_query_enabled,
        TOP_K,
    )
    
    # Add dummy row for Query (Ensures it appears at the top of the text report)
    q_row = pd.DataFrame([{"index": -1, "header": f"(Query) {query_name}", "raw_score": 0.0, "norm_score": 99.9, "length": len(query_emb), "seq_len": len(query_emb), "aln_len": len(query_emb), "identity": 100.0}])
    df = pd.concat([q_row, df], ignore_index=True)
    
    # Use custom name if provided, otherwise fallback to sanitized query name
    if OUTPUT_NAME and str(OUTPUT_NAME).strip():
        base_filename = str(OUTPUT_NAME).strip()
    else:
        base_filename = re.sub(r'[^a-zA-Z0-9]', '_', query_name)[:20]
        
    save_results(df, (query_name, len(query_emb)), len(db_headers), seq_lookup, base_filename, query_seq_str, NORM_MODE, gap_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
