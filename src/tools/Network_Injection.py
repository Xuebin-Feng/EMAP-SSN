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
File: Network_Injection.py
===================================
Description:
After adding new sequences to a precomputed embedding database (via `Embedding_Injection.py`), this script efficiently 
updates the massive all-vs-all similarity network without recalculating the pre-existing pairwise combinations.
It dynamically computes combinations involving the *new* sequences while rapidly copying existing connections from cache.

Input:
- The old existing HDF5 network containing pre-calculated alignment scores (`OLD_NETWORK`).
- The newly updated embeddings database containing both old and new tensors (`NEW_EMBEDDINGS`).

Output:
- A new master HDF5 network containing all completed old and new edges (`FINAL_OUTPUT_NET`).

Settings:
- OLD_SEQUENCE_SET / NEW_SEQUENCE_SET: Filename parameters used to locate the input databases and save the updated output.
- MODEL_NAME: The model identifier matching the embeddings used.
- BATCH_SIZE: Number of pairs to process between writing to intermediate temp files (RAM protection).
- WORKERS: Threads used for multiprocessing new alignment combinations.
- EXECUTION_MODE: Automatic, scalar, or tiled pairwise score execution.

Algorithm:
1. Loads the headers of both the OLD network and the NEW embedding database.
2. Mathematically calculates the exact number of pairs required for an all-vs-all grid `(N*(N-1))/2` for the new sequence length.
3. Pre-allocates massive, correctly sized blank contiguous 1D numpy arrays into system memory.
4. Iterates linearly over every theoretical combination pair in the new index space.
5. If both headers previously existed, it converts their 2D coordinates back into a 1D flat index and instantly copies 
   the raw alignment score bytes straight into the new arrays in RAM.
6. If the combination involves a new header, the required index coordinates are tossed into a "To-Do" queue.
7. Multiprocessing CPU workers iterate over the to-do queue, plucking the required embeddings from disk and directly evaluating their Needleman-Wunsch/Smith-Waterman scores.
8. Calculated results are merged into the massive contiguous array space and structurally streamed back to disk as HDF5 arrays.
"""
# %% Import Necessary Libraries
# Limit threads to prevent CPU thrashing
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap
from utilities import Tool_Supervisor
Tool_Supervisor.supervise(__name__, __file__)  # a run of this script starts under a light parent; nothing else here

from Cache_Manifest import validate_network_schema

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
import h5py
import torch
import glob
import math
import shutil
import sys
import hashlib
import threading
import time
from collections import deque
from itertools import islice, chain
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
from utilities import Benchmark_Record
from utilities import Hardware_Acceleration as Hardware_Utils
# global_local_scores is the reference kernel the per-pair path is checked against.
from utilities.Network_Kernels import (
    align_microbatch,
    global_local_pair_scores,
    global_local_scores,
)
from Embedding_Alignment_Engine import (
    BenchmarkTrial,
    BENCHMARK_TRIAL_SECONDS,
    run_bounded_cpu_trial,
    EmbeddingTileStore,
    bf16_accelerator_support,
    bf16_validation_notice,
    compare_bf16_precision_results,
    compute_score_matrix_torch as _shared_score_matrix,
    cuda_matmul_precision,
    cuda_memory_plan,
    evenly_spaced_task_subset,
    estimate_cuda_working_set,
    format_bf16_validation_report,
    get_accelerator_backend,
    is_nvidia_cuda,
    matched_benchmark_task_halves,
    normalize_execution_mode,
    precision_element_bytes,
    run_tiled_accelerator_pipeline,
    tiled_accelerator_support,
)
from utilities.HDF5_Storage import read_embedding_manifest
from utilities import Prefilter_Calibration

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS (Filenames only, provided by GUI)
OLD_NETWORK = None 
NEW_EMBEDDINGS = None 

# SETTINGS
BATCH_SIZE = 500000 
WORKERS = 12 
DEVICE_SELECTION = "auto"
EXECUTION_MODE = "auto"
HOST_CACHE_GB = "auto"
ACCELERATOR_BENCHMARK_HALF_PAIRS = 4096
CPU_BENCHMARK_HALF_PAIRS = 256
ACCELERATOR_TUNE_PAIRS = None
ACCELERATOR_CONFIRM_PAIRS = None
LOCAL_GAP_P = None
GLOBAL_GAP_P = None 

from tools.tool_helpers.Tool_Pipeline import (
    inherited_settings_path,
    load_tool_settings,
    project_directory_defaults,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
NETWORK_DIR = _DEFAULT_DIRECTORIES["NETWORK_DIR"]
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]

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

import re

_old_seq_set = "UnknownOld"
_new_seq_set = "UnknownNew"
_model_name = "unknown"
RESULTS_DIR = None
FINAL_OUTPUT_NET = None
CONFIG_FILE = None


def _resolve_selected_path(value, directory, description):
    if value is None or not str(value).strip():
        raise ValueError(f"No {description} has been selected.")

    selected_path = os.fspath(value)
    if os.path.isabs(selected_path):
        return os.path.normpath(selected_path)
    return os.path.normpath(os.path.join(directory, selected_path))


def configure_input_paths():
    """Resolve GUI-selected inputs without opening either file."""
    global OLD_NETWORK, NEW_EMBEDDINGS, _old_seq_set, _new_seq_set

    old_filename = os.path.basename(os.fspath(OLD_NETWORK)) if OLD_NETWORK else ""
    new_filename = os.path.basename(os.fspath(NEW_EMBEDDINGS)) if NEW_EMBEDDINGS else ""

    _old_seq_set = "UnknownOld"
    match_old = re.search(r"^(.*)_\[(.*)\]_network\.h5$", old_filename)
    if match_old:
        _old_seq_set = match_old.group(1)

    _new_seq_set = "UnknownNew"
    match_new = re.search(r"^(.*)_\[(.*)\]_embeddings\.h5$", new_filename)
    if match_new:
        _new_seq_set = match_new.group(1)

    OLD_NETWORK = _resolve_selected_path(
        OLD_NETWORK,
        NETWORK_DIR,
        "existing network file",
    )
    NEW_EMBEDDINGS = _resolve_selected_path(
        NEW_EMBEDDINGS,
        EMBED_DIR,
        "new embeddings file",
    )


def configure_output_paths(model_name):
    """Build output paths after the existing network has been validated."""
    global _model_name, RESULTS_DIR, FINAL_OUTPUT_NET, CONFIG_FILE

    _model_name = re.sub(r'[<>:"/\\|?*]', "_", model_name)
    RESULTS_DIR = os.path.join(
        NETWORK_DIR,
        f"{_new_seq_set}_[{_model_name}]_network_temp",
    )
    FINAL_OUTPUT_NET = os.path.join(
        NETWORK_DIR,
        f"{_new_seq_set}_[{_model_name}]_network.h5",
    )
    CONFIG_FILE = os.path.join(RESULTS_DIR, "injection_config.json")

# %% =======================================
# KERNELS 
# ==========================================
def calculate_file_hash(file_path):
    """
    Computes the MD5 checksum of a file in chunks to verify consistency.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class EmbeddingFileError(RuntimeError):
    """Raised when an embedding HDF5 file is incomplete or incompatible."""


class NetworkCompilationError(RuntimeError):
    """Raised when batch files cannot be merged into a complete network."""


def load_embedding_metadata(file_path):
    """
    Validate an embedding database using Embedding_HDF5.read_embedding_manifest
    and return ordered headers, safe headers, sequence lengths, and manifest.
    """
    try:
        with h5py.File(file_path, "r") as hf:
            manifest = read_embedding_manifest(
                hf,
                require_complete=True,
                validate_embeddings=True,
            )
            headers = manifest.headers
            safe_headers = [
                header.replace("/", "_").replace("\\", "_")
                for header in headers
            ]
            seq_lens = [len(seq) for seq in manifest.sequences]
    except EmbeddingFileError:
        raise
    except Exception as error:
        raise EmbeddingFileError(
            f"Embedding file '{file_path}' validation failed: {error}"
        ) from error

    return headers, safe_headers, seq_lens, manifest


def compute_score_matrix_torch(emb_i, emb_j, device):
    return _shared_score_matrix(emb_i, emb_j, device, precision="float32")

# %% =======================================
# HDF5 WORKER INITIALIZATION
# ==========================================
worker_hf = None
worker_device = None

def init_worker(h5_path, gap_penalties):
    global worker_hf, worker_device, LOCAL_GAP_P, GLOBAL_GAP_P
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)
    worker_device = torch.device("cpu")
    # Spawned workers re-import this module, so they never see the gap
    # penalties run_injection inherits from the input network.
    LOCAL_GAP_P, GLOBAL_GAP_P = gap_penalties


# %% =======================================
# PORTABLE ACCELERATOR/CPU PIPELINE
# ==========================================
def calculate_alignment_data(args):
    idx_i, idx_j, matrix = args
    g_raw, g_len, l_raw, l_len = global_local_pair_scores(
        matrix,
        GLOBAL_GAP_P,
        LOCAL_GAP_P,
    )

    return (idx_i, idx_j, l_raw, l_len, g_raw, g_len)


def calculate_alignment_batch(args):
    """
    Run the dynamic programming for one padded accelerator microbatch.

    One CPU task aligns every pair of the microbatch, so the thread pool
    handles one future per microbatch instead of one per pair.
    """
    idx_i, target_indices, target_lengths, matrices = args
    return align_microbatch(
        matrices,
        idx_i,
        target_indices,
        target_lengths,
        GLOBAL_GAP_P,
        LOCAL_GAP_P,
    )


def calculate_cpu_pair(args):
    idx_i, idx_j, safe_h_i, safe_h_j = args
    global worker_hf, worker_device

    emb_i = worker_hf["embeddings"][safe_h_i][:]
    emb_j = worker_hf["embeddings"][safe_h_j][:]
    matrix = compute_score_matrix_torch(emb_i, emb_j, worker_device)
    return calculate_alignment_data((idx_i, idx_j, matrix))


accelerator_thread_state = threading.local()


def _benchmark_half_sizes():
    accelerator_half = (
        ACCELERATOR_BENCHMARK_HALF_PAIRS
        if ACCELERATOR_CONFIRM_PAIRS is None
        else ACCELERATOR_CONFIRM_PAIRS
    )
    cpu_half = (
        CPU_BENCHMARK_HALF_PAIRS
        if ACCELERATOR_TUNE_PAIRS is None
        else ACCELERATOR_TUNE_PAIRS
    )
    return max(1, int(accelerator_half)), max(1, int(cpu_half))


def _device_type(device):
    return getattr(device, "type", str(device).split(":", 1)[0])


def _uses_accelerator(device):
    return _device_type(device) != "cpu"


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


def _compute_accelerated_matrix(args):
    idx_i, idx_j, emb_i, emb_j, device, precision = args
    with torch.inference_mode():
        with _stream_context(device):
            matrix = _shared_score_matrix(
                emb_i, emb_j, device, precision=precision
            )
    return idx_i, idx_j, matrix


def _run_accelerated_pipeline(
    tasks,
    workers,
    input_h5,
    device,
    batch_id,
    lanes,
    show_progress,
    matmul_precision="float32",
    result_callback=None,
    benchmark_trial=None,
):
    tasks = list(tasks)
    results = []
    ready_limit = max(lanes, min(workers, 8))
    cached_row_header = None
    cached_row_embedding = None
    progress_context = (
        tqdm(
            total=len(tasks),
            desc=f"  Batch {batch_id} ({lanes} accelerator lane"
                 f"{'s' if lanes != 1 else ''})",
            leave=False,
        )
        if show_progress
        else nullcontext(None)
    )

    with cuda_matmul_precision(matmul_precision), \
            h5py.File(input_h5, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=lanes,
                thread_name_prefix="injection-accelerator",
            ) as accelerator_executor, \
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="injection-cpu",
            ) as cpu_executor, \
            progress_context as progress:
        def run_phase(phase_tasks, trial=None):
            nonlocal cached_row_header, cached_row_embedding
            gpu_pending = set()
            cpu_pending = set()
            ready_for_cpu = deque()
            task_iterator = iter(phase_tasks)
            tasks_exhausted = False
            while (
                not tasks_exhausted
                or gpu_pending
                or ready_for_cpu
                or cpu_pending
            ):
                while ready_for_cpu and len(cpu_pending) < workers:
                    cpu_pending.add(
                        cpu_executor.submit(
                            calculate_alignment_data,
                            ready_for_cpu.popleft(),
                        )
                    )
                while (
                    not tasks_exhausted
                    and len(gpu_pending) < lanes
                    and len(ready_for_cpu) + len(gpu_pending) < ready_limit
                ):
                    if trial is not None and not trial.can_submit():
                        tasks_exhausted = True
                        break
                    try:
                        idx_i, idx_j, safe_h_i, safe_h_j = next(task_iterator)
                    except StopIteration:
                        tasks_exhausted = True
                        break
                    if safe_h_i != cached_row_header:
                        cached_row_embedding = hf["embeddings"][safe_h_i][:]
                        cached_row_header = safe_h_i
                    emb_j = hf["embeddings"][safe_h_j][:]
                    gpu_pending.add(
                        accelerator_executor.submit(
                            _compute_accelerated_matrix,
                            (
                                idx_i,
                                idx_j,
                                cached_row_embedding,
                                emb_j,
                                device,
                                matmul_precision,
                            ),
                        )
                    )
                    if trial is not None:
                        trial.submitted += 1

                completed_gpu = {
                    future for future in gpu_pending if future.done()
                }
                completed_cpu = {
                    future for future in cpu_pending if future.done()
                }
                if not completed_gpu and not completed_cpu:
                    pending = gpu_pending | cpu_pending
                    if not pending:
                        continue
                    completed, _ = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )
                    completed_gpu = completed & gpu_pending
                    completed_cpu = completed & cpu_pending

                for future in completed_gpu:
                    gpu_pending.remove(future)
                    ready_for_cpu.append(future.result())
                for future in completed_cpu:
                    cpu_pending.remove(future)
                    results.append(future.result())
                    if trial is not None:
                        trial.completed += 1
                    if progress is not None:
                        progress.update(1)
                if result_callback is not None and len(results) >= 65536:
                    result_callback(results)
                    results.clear()

        if benchmark_trial is not None:
            run_phase(tasks[:benchmark_trial.warmup_count(tasks, workers, lanes)])
            results.clear()
            cached_row_header = None
            cached_row_embedding = None
            benchmark_trial.start()
            run_phase(tasks, benchmark_trial)
            benchmark_trial.stop(len(tasks))
        else:
            run_phase(tasks)

    if result_callback is not None and results:
        result_callback(results)
        results.clear()
    return results


def process_cpu_tasks(
    tasks,
    workers,
    input_h5,
    batch_id,
    show_progress=True,
    result_callback=None,
    benchmark_trial=None,
):
    tasks = list(tasks)
    results = []
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(input_h5, (LOCAL_GAP_P, GLOBAL_GAP_P)),
    ) as pool:
        progress = tqdm(
            total=len(tasks),
            desc=f"  Batch {batch_id}",
            leave=False,
            disable=not show_progress,
        )

        def run_phase(phase_tasks):
            iterator = pool.imap_unordered(
                calculate_cpu_pair,
                phase_tasks,
                chunksize=10,
            )
            for result in iterator:
                results.append(result)
                progress.update(1)
                if result_callback is not None and len(results) >= 65536:
                    result_callback(results)
                    results.clear()

        try:
            if benchmark_trial is not None:
                results = run_bounded_cpu_trial(
                    pool, calculate_cpu_pair, tasks, workers, benchmark_trial
                )
            else:
                run_phase(tasks)
        finally:
            progress.close()
    if result_callback is not None and results:
        result_callback(results)
        results.clear()
    return results


def _execution_variants(candidate):
    """Return execution variants allowed for one hardware candidate."""
    mode = normalize_execution_mode(EXECUTION_MODE)
    tiled_supported = (
        candidate.backend in {"cuda", "xpu"}
        and tiled_accelerator_support(
            candidate.device, require_memory=False
        )[0]
    )
    if mode == "scalar":
        return ["scalar"]
    if mode == "tiled":
        return ["tiled"] if tiled_supported else []
    variants = ["scalar"]
    if tiled_supported:
        variants.append("tiled")
    return variants


def _validate_execution_mode_hardware():
    """Fail early when forced tiled execution cannot use selected hardware."""
    mode = normalize_execution_mode(EXECUTION_MODE)
    if mode != "tiled":
        return mode
    candidates = Hardware_Utils.get_available_devices()
    manual = Hardware_Utils.resolve_device_selection(
        DEVICE_SELECTION, candidates
    )
    if manual is not None and not _execution_variants(manual):
        raise ValueError(
            "Tiled execution requires a CUDA/ROCm or XPU accelerator; "
            "the selected device "
            f"is '{manual.spec}'."
        )
    eligible = [
        candidate for candidate in candidates if _execution_variants(candidate)
    ]
    if manual is not None:
        eligible = [manual]
    if not eligible:
        raise ValueError(
            "Tiled execution was requested, but no compatible CUDA/ROCm or "
            "XPU accelerator is available."
        )
    return mode


def _execute_injection_plan(
    plan, tasks, workers, input_h5, batch_id, store, lengths,
    matmul_precision, show_progress=False, benchmark_trial=None,
):
    candidate, variant, lanes = plan
    if candidate.is_cpu:
        return process_cpu_tasks(
            tasks, workers, input_h5, batch_id,
            show_progress=show_progress,
            benchmark_trial=benchmark_trial,
        )
    if variant == "tiled":
        return run_tiled_accelerator_pipeline(
            tasks,
            store=store,
            lengths=lengths,
            device=candidate.device,
            workers=workers,
            lanes=lanes,
            alignment_callback=calculate_alignment_data,
            batch_alignment_callback=calculate_alignment_batch,
            precision=matmul_precision,
            benchmark_trial=benchmark_trial,
        )
    return _run_accelerated_pipeline(
        tasks, workers, input_h5, candidate.device, batch_id, lanes,
        show_progress, matmul_precision=matmul_precision,
        benchmark_trial=benchmark_trial,
    )


# Pair selection works on int64 pair keys, min * N + max (row * N + column with
# row < column): sorted arrays, tested with searchsorted. For N = 50,000 the
# largest key is 2,500,049,999.
_PAIR_BLOCK_CELLS = 1 << 22   # cells of one row block of the required-pair mask
_EDGE_CHUNK = 1 << 22         # old edges handled at once by the cosine scans
_PENDING_CHUNK = 1 << 16      # queued pairs turned into tuples at once


def _sorted_unique(keys):
    """Sorted distinct values (np.unique, which is several times slower on millions of int64 keys)."""
    keys = np.sort(keys)
    if keys.size > 1:
        keys = keys[np.concatenate(([True], keys[1:] != keys[:-1]))]
    return keys


def _as_sorted_keys(keys):
    """Pair keys as a sorted, unique int64 array; a set, list or array is accepted."""
    if isinstance(keys, np.ndarray):
        array = keys.astype(np.int64, copy=False)
    else:
        array = np.fromiter(keys, dtype=np.int64, count=len(keys))
    if array.size > 1 and not bool(np.all(array[1:] > array[:-1])):
        array = _sorted_unique(array)
    return array


def _locate(sorted_keys, needles):
    """Position of each needle in sorted_keys (clipped) and whether it is there."""
    if sorted_keys.size == 0:
        return np.zeros(needles.shape, dtype=np.int64), np.zeros(needles.shape, dtype=bool)
    position = np.searchsorted(sorted_keys, needles)
    np.minimum(position, sorted_keys.size - 1, out=position)
    return position, sorted_keys[position] == needles


def _members(sorted_keys, needles):
    return _locate(sorted_keys, needles)[1]


def _pair_similarity_rows(cos_sim, start, stop):
    """The similarity of rows start..stop-1 to every sequence."""
    return cos_sim[start:stop]


def _pair_similarity_at(cos_sim, rows, columns):
    """The similarity of each (row, column) pair."""
    return cos_sim[rows, columns]


def _lowest_edge_cosine(cos_sim, edge_u, edge_v, skip):
    """(value, edge) of the FIRST old edge with the lowest similarity below 1.0.

    Only edges that keep both sequences (edge_u, edge_v >= 0) and are not
    skipped (calibration edges) count; (1.0, -1) when none is below 1.0.
    """
    best_value = 1.0
    best_edge = -1
    for start in range(0, edge_u.size, _EDGE_CHUNK):
        stop = min(edge_u.size, start + _EDGE_CHUNK)
        u = edge_u[start:stop]
        v = edge_v[start:stop]
        positions = np.flatnonzero((u >= 0) & (v >= 0) & ~skip[start:stop])
        if not positions.size:
            continue
        sims = _pair_similarity_at(cos_sim, u[positions], v[positions])
        below = np.flatnonzero(sims < best_value)
        if below.size:
            first = below[np.argmin(sims[below])]
            best_value = float(sims[first])
            best_edge = start + int(positions[first])
    return best_value, best_edge


def _edges_below_cosine(cos_sim, edge_u, edge_v, skip, limit, keep_first=10):
    """How many counted old edges have a similarity below limit, and the first few as (edge, value)."""
    count = 0
    first = []
    for start in range(0, edge_u.size, _EDGE_CHUNK):
        stop = min(edge_u.size, start + _EDGE_CHUNK)
        u = edge_u[start:stop]
        v = edge_v[start:stop]
        positions = np.flatnonzero((u >= 0) & (v >= 0) & ~skip[start:stop])
        if not positions.size:
            continue
        sims = _pair_similarity_at(cos_sim, u[positions], v[positions])
        hits = np.flatnonzero(sims < limit)
        count += int(hits.size)
        for hit in hits[: keep_first - len(first)]:
            first.append((start + int(positions[hit]), sims[hit]))
    return count, first


def _cached_pair_keys(edge_u, edge_v, new_N):
    """Sorted keys of the old edges whose two sequences are both in the new set."""
    kept = (edge_u >= 0) & (edge_v >= 0)
    u = edge_u[kept]
    v = edge_v[kept]
    return _sorted_unique(np.minimum(u, v) * new_N + np.maximum(u, v))


def _required_pair_keys(new_N, is_old, cached, cos_sim, threshold):
    """Sorted keys of the pairs the new network needs.

    A pair of two old sequences is needed when the old network has that edge
    (cached). Any other pair is needed when cos_sim is None (a dense input) or
    reaches the threshold. Built in row blocks, so no N x N array is held
    beyond one block of at most _PAIR_BLOCK_CELLS cells.
    """
    rows = cached // new_N if new_N else cached
    columns = cached % new_N if new_N else cached
    old_cached = cached[(rows < columns) & is_old[rows] & is_old[columns]] if cached.size else cached
    column_index = np.arange(new_N)
    block_rows = max(1, _PAIR_BLOCK_CELLS // max(new_N, 1))
    pieces = []
    for start in range(0, new_N, block_rows):
        stop = min(new_N, start + block_rows)
        keep = column_index[None, :] > np.arange(start, stop)[:, None]
        keep &= ~(is_old[start:stop, None] & is_old[None, :])
        if cos_sim is not None:
            keep &= _pair_similarity_rows(cos_sim, start, stop) >= threshold
        block_row, block_column = np.nonzero(keep)
        mixed = (block_row + start).astype(np.int64) * new_N + block_column
        low = np.searchsorted(old_cached, start * new_N)
        high = np.searchsorted(old_cached, stop * new_N)
        piece = np.concatenate((mixed, old_cached[low:high]))
        piece.sort()
        pieces.append(piece)
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)


def _pending_keys(required_pairs, cached_old_pairs, computed_pairs, count):
    """Sorted keys of the required pairs that are neither cached nor computed."""
    required = _as_sorted_keys(required_pairs)
    cached = _as_sorted_keys(cached_old_pairs)
    computed = _as_sorted_keys(computed_pairs)
    pending = required[~_members(cached, required) & ~_members(computed, required)]
    if not count:
        return pending
    rows = pending // count
    return pending[(pending % count > rows) & (rows < count)]


def _iter_pending_keys(headers, pending, count):
    """Yield (row, column, header, header) for sorted pending keys: production order."""
    for start in range(0, pending.size, _PENDING_CHUNK):
        chunk = pending[start:start + _PENDING_CHUNK]
        for row, column in zip((chunk // count).tolist(), (chunk % count).tolist()):
            yield row, column, headers[row], headers[column]


def _iter_pending_pairs(headers, required_pairs, cached_old_pairs, computed_pairs):
    """Yield injection pairs in production order, excluding cached work."""
    count = len(headers)
    pending = _pending_keys(required_pairs, cached_old_pairs, computed_pairs, count)
    yield from _iter_pending_keys(headers, pending, count)


class _OldEdgeIndex:
    """File row of an old edge, looked up by its flat (condensed) pair index; -1 when absent."""

    def __init__(self, flat_indices):
        flat_indices = np.asarray(flat_indices, dtype=np.int64)
        self._order = np.argsort(flat_indices, kind="stable")
        self._sorted = flat_indices[self._order]

    def __getitem__(self, flat):
        flat = np.asarray(flat, dtype=np.int64)
        if self._sorted.size == 0:
            return np.full(flat.shape, -1, dtype=np.int64)
        # side="right" - 1 finds the last duplicate, as a later assignment did.
        position = np.searchsorted(self._sorted, flat, side="right") - 1
        position = np.maximum(position, 0)
        found = self._sorted[position] == flat
        return np.where(found, self._order[position], -1)


def _benchmark_injection_plans(
    tasks, workers, input_h5, store, lengths, matmul_precision,
    precision_tasks=None,
):
    execution_mode = normalize_execution_mode(EXECUTION_MODE)
    candidates = Hardware_Utils.get_available_devices()
    if matmul_precision == "tf32":
        candidates = [
            candidate for candidate in candidates
            if candidate.backend == "cuda" and is_nvidia_cuda(candidate.device)
        ]
        if not candidates:
            raise RuntimeError(
                "The input network was calculated with TF32 and can only be "
                "extended on an NVIDIA CUDA device."
            )
    elif matmul_precision == "bf16":
        candidates = [
            candidate for candidate in candidates
            if not candidate.is_cpu
            and bf16_accelerator_support(candidate.device)[0]
        ]
        if not candidates:
            raise RuntimeError(
                "The input network uses BF16, but no accelerator passed the "
                "BF16 runtime capability probe."
            )
    manual = Hardware_Utils.resolve_device_selection(DEVICE_SELECTION, candidates)
    if execution_mode == "tiled":
        if manual is not None and not _execution_variants(manual):
            raise ValueError(
                "Tiled execution requires a CUDA/ROCm or XPU accelerator; "
                "the selected device "
                f"is '{manual.spec}'."
            )
        candidates = [
            candidate for candidate in candidates
            if _execution_variants(candidate)
        ]
        if not candidates:
            raise ValueError(
                "Tiled execution was requested, but no compatible CUDA/ROCm "
                "or XPU accelerator is available."
            )
    if manual is not None:
        candidates = [manual]
        if manual.is_cpu:
            print(f"[Hardware] Using manually selected {manual.display_name}.")
            return [Hardware_Utils.BenchmarkResult(manual, 0.0, variant="scalar")]
    if matmul_precision == "tf32" and (
        not candidates or any(
            candidate.backend != "cuda" or not is_nvidia_cuda(candidate.device)
            for candidate in candidates
        )
    ):
        raise RuntimeError(
            "The input network uses TF32, but the selected device is not "
            "an NVIDIA CUDA device."
        )

    if matmul_precision == "bf16":
        validation_source = tasks if precision_tasks is None else precision_tasks
        validation_tasks = evenly_spaced_task_subset(validation_source, min(2048, len(validation_source)))
        print(
            bf16_validation_notice(
                tool_name="Network Injection",
                sample_count=len(validation_tasks),
            )
        )
        validated = []
        validated_variants_by_spec = {}
        failures = []
        for candidate in candidates:
            if candidate.is_cpu:
                failures.append(f"{candidate.display_name}: CPU BF16 is unsupported")
                continue
            try:
                baseline = _run_accelerated_pipeline(
                    validation_tasks,
                    workers,
                    input_h5,
                    candidate.device,
                    -1,
                    1,
                    False,
                    matmul_precision="float32",
                )
            except Exception as error:
                Hardware_Utils.note_gpu_failure(candidate, error, where="precision validation")
                failure = (
                    f"{candidate.display_name} FP32 baseline: "
                    f"{type(error).__name__}: {error}"
                )
                failures.append(failure)
                print(f"[Precision] Excluding BF16 on {failure}.")
                Hardware_Utils.release_device_cache(candidate)
                continue

            completed_variants = []
            for variant in _execution_variants(candidate):
                try:
                    candidate_results = _execute_injection_plan(
                        (candidate, variant, 1),
                        validation_tasks,
                        workers,
                        input_h5,
                        -1,
                        store,
                        lengths,
                        "bf16",
                    )
                    report = compare_bf16_precision_results(
                        baseline,
                        candidate_results,
                        candidate_label="BF16",
                    )
                    print(
                        format_bf16_validation_report(
                            report,
                            context=(
                                f"tool=Network Injection; "
                                f"device={candidate.display_name}; "
                                f"backend={candidate.backend}; variant={variant}"
                            ),
                            identity_label="pair",
                        )
                    )
                except Exception as error:
                    Hardware_Utils.note_gpu_failure(candidate, error, where="precision validation")
                    failure = (
                        f"{candidate.display_name} {variant}: "
                        f"{type(error).__name__}: {error}"
                    )
                    failures.append(failure)
                    print(f"[Precision] Excluding BF16 plan {failure}.")
                    continue
                completed_variants.append(variant)
            if completed_variants:
                validated.append(candidate)
                validated_variants_by_spec[candidate.spec] = set(
                    completed_variants
                )
                print(
                    f"[Precision] Completed informational BF16 validation for "
                    f"{', '.join(completed_variants)} on {candidate.display_name}."
                )
            Hardware_Utils.release_device_cache(candidate)
        candidates = validated
        if not candidates:
            raise RuntimeError(
                "No selected accelerator completed BF16 validation ("
                + "; ".join(failures)
                + ")."
            )

    sample = list(tasks)
    if not sample:
        raise ValueError("The hardware benchmark requires at least one pending pair.")
    print(
        f"[Hardware] First pending batch: {len(sample)} pairs; "
        f"{BENCHMARK_TRIAL_SECONDS:g}-second soft submission limit per configuration; "
        "reported time includes draining."
    )
    print("Device/backend                 Plan      Lanes   Est. VRAM peak/safe   Pairs/s   Status")
    results = []
    for candidate in candidates:
        variants = _execution_variants(candidate)
        if matmul_precision == "bf16":
            variants = [
                variant
                for variant in variants
                if variant in validated_variants_by_spec.get(candidate.spec, set())
            ]
        lane_candidates = [1] if candidate.is_cpu else _lane_candidates(
            candidate.device, workers
        )
        memory_info = None
        if candidate.backend in {"cuda", "xpu"}:
            Hardware_Utils.release_device_cache(candidate)
            memory = cuda_memory_plan(
                candidate.device,
                lanes=1,
                compute_element_bytes=precision_element_bytes(matmul_precision),
            )
            memory_info = (memory.free_bytes, memory.total_bytes)
        estimates = {}
        for variant in variants:
            for lanes in lane_candidates:
                estimates[variant, lanes] = (
                    estimate_cuda_working_set(
                        sample, store=store, lengths=lengths,
                        device=candidate.device, lanes=lanes, variant=variant,
                        memory_info=memory_info,
                        compute_element_bytes=precision_element_bytes(matmul_precision),
                    ) if memory_info is not None else None
                )
        feasible_count = sum(e is None or e.feasible for e in estimates.values())
        print(f"[Hardware] {candidate.display_name}: {feasible_count} feasible configurations.")
        for variant in variants:
            for lanes in lane_candidates:
                vram = "--"
                if memory_info is not None:
                    estimate = estimates[variant, lanes]
                    vram = (
                        f"{estimate.projected_peak_bytes / (1024 ** 3):.1f}/"
                        f"{estimate.safe_peak_bytes / (1024 ** 3):.1f}G"
                    )
                    if not estimate.feasible:
                        print(
                            f"{candidate.display_name[:30]:30}  {variant:8}  "
                            f"{lanes:>5}   {vram:>14}   {'--':>9}   "
                            f"skipped: {estimate.reason}"
                        )
                        continue
                free_memory = Hardware_Utils.begin_gpu_trial(candidate)
                trial = BenchmarkTrial()
                try:
                    _execute_injection_plan(
                        (candidate, variant, lanes), sample, workers, input_h5,
                        -1, store, lengths, matmul_precision,
                        benchmark_trial=trial,
                    )
                    rate = trial.rate
                    results.append(
                        Hardware_Utils.BenchmarkResult(
                            candidate, rate, lanes=lanes, variant=variant
                        )
                    )
                    print(
                        f"{candidate.display_name[:30]:30}  {variant:8}  "
                        f"{lanes:>5}   {vram:>14}   {rate:>9.2f}   "
                        f"ok ({trial.status(tiled=variant == 'tiled')})"
                    )
                except Exception as error:
                    memory_text = Hardware_Utils.note_gpu_failure(
                        candidate, error, free_memory,
                        where=f"plan test: {variant}, {lanes} lane(s)",
                    )
                    status, *detail_lines = Hardware_Utils.failure_lines(error, memory_text)
                    print(
                        f"{candidate.display_name[:30]:30}  {variant:8}  "
                        f"{lanes:>5}   {vram:>14}   {'--':>9}   {status}"
                    )
                    for line in detail_lines:
                        print(line)
        Hardware_Utils.release_device_cache(candidate)

    ranked = Hardware_Utils.rank_benchmark_results(
        results,
        higher_is_better=True,
        decision={
            "kind": "injection_plan",
            "unit": "pairs/s",
            "pairs": len(sample),
            "precision": matmul_precision,
            "trial_seconds": BENCHMARK_TRIAL_SECONDS,
        },
    )
    if not ranked:
        raise RuntimeError("No injection processing plan completed successfully.")
    winner = ranked[0]
    fastest = max(results, key=lambda result: float(result.value))
    print(
        f"[Hardware] Selected {winner.candidate.display_name}, "
        f"{winner.variant} plan, {winner.lanes} lane(s); 3% tie preference "
        f"{'applied' if winner is not fastest else 'not applied'}."
    )
    if winner.variant == "tiled":
        budget = cuda_memory_plan(
            winner.candidate.device, lanes=winner.lanes
        ).matrix_bytes
        print(
            f"[Hardware] Accelerator microbatch budget: "
            f"{budget / (1024 ** 2):.1f} MiB."
        )
    return ranked


class _PartialBatchWriter:
    DATASET_DTYPES = {
        "i": np.uint32,
        "j": np.uint32,
        "l_score": np.float32,
        "l_len": np.uint16,
        "g_score": np.float32,
        "g_len": np.uint16,
    }

    def __init__(
        self, output_filename, embedding_checksum, model_name, saving_mode,
        gap_penalties, matmul_precision,
    ):
        self.output_filename = output_filename
        self.partial_filename = output_filename + ".partial"
        if os.path.exists(self.partial_filename):
            os.remove(self.partial_filename)
        self.hf = h5py.File(self.partial_filename, "w")
        if embedding_checksum is not None:
            self.hf.attrs["embedding_checksum"] = embedding_checksum
        if model_name is not None:
            self.hf.attrs["model_name"] = model_name
        if saving_mode is not None:
            self.hf.attrs["saving_mode"] = saving_mode
        if gap_penalties is not None:
            self.hf.attrs["gap_penalties"] = np.asarray(
                gap_penalties, dtype=np.float32
            )
        self.hf.attrs["matmul_precision"] = matmul_precision
        self.datasets = {
            name: self.hf.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dtype
            )
            for name, dtype in self.DATASET_DTYPES.items()
        }
        self.count = 0
        self.closed = False

    def __call__(self, results):
        if not results:
            return
        start = self.count
        end = start + len(results)
        for dataset in self.datasets.values():
            dataset.resize((end,))
        columns = tuple(zip(*results))
        for column, (name, dtype) in zip(columns, self.DATASET_DTYPES.items()):
            self.datasets[name][start:end] = np.asarray(column, dtype=dtype)
        self.count = end

    def rollback(self, count=0):
        count = int(count)
        for dataset in self.datasets.values():
            dataset.resize((count,))
        self.count = count
        self.hf.flush()

    def publish(self):
        self.hf.flush()
        self.hf.close()
        self.closed = True
        os.replace(self.partial_filename, self.output_filename)

    def close(self):
        if not self.closed:
            self.hf.close()
            self.closed = True


def process_batch(
    batch_tasks,
    batch_id,
    workers,
    new_emb_path,
    embedding_checksum,
    model_name,
    saving_mode,
    gap_penalties,
    device=None,
    accelerator_workers=None,
    execution_variant="scalar",
    matmul_precision="ieee_fp32",
    embedding_store=None,
    sequence_lengths=None,
):
    output_filename = os.path.join(RESULTS_DIR, f"batch_{batch_id:05d}.h5")
    if device is None:
        device = Hardware_Utils.get_optimal_device()
    lanes = max(1, int(accelerator_workers or 1))
    writer = _PartialBatchWriter(
        output_filename, embedding_checksum, model_name, saving_mode,
        gap_penalties, matmul_precision,
    )
    try:
        if _uses_accelerator(device) and execution_variant == "tiled":
            backend = get_accelerator_backend(device)
            backend_label = backend.device_type.upper()
            budget = cuda_memory_plan(device, lanes=lanes).matrix_bytes
            while True:
                try:
                    results = run_tiled_accelerator_pipeline(
                        batch_tasks,
                        store=embedding_store,
                        lengths=sequence_lengths,
                        device=device,
                        workers=workers,
                        lanes=lanes,
                        alignment_callback=calculate_alignment_data,
                        batch_alignment_callback=calculate_alignment_batch,
                        precision=matmul_precision,
                        result_callback=writer,
                        matrix_budget_override=budget,
                    )
                    break
                except Exception as error:
                    if not backend.is_out_of_memory(error):
                        raise
                    writer.rollback(0)
                    backend.empty_cache()
                    if budget <= 1024 ** 2:
                        Hardware_Utils.note_gpu_failure(
                            device, error, where="batch: injection tile out of memory"
                        )
                        raise
                    next_budget = max(1024 ** 2, budget // 2)
                    print(
                        f"[{backend_label}] Injection tile OOM; retrying uncommitted "
                        f"batch with {next_budget / (1024 ** 2):.1f} MiB "
                        f"per microbatch."
                    )
                    budget = next_budget
        elif _uses_accelerator(device):
            results = _run_accelerated_pipeline(
                batch_tasks, workers, new_emb_path, device, batch_id, lanes,
                True, matmul_precision=matmul_precision,
                result_callback=writer,
            )
        else:
            results = process_cpu_tasks(
                batch_tasks, workers, new_emb_path, batch_id,
                result_callback=writer,
            )
        writer(results)
        if writer.count != len(batch_tasks):
            raise RuntimeError(
                f"Batch produced {writer.count} results for "
                f"{len(batch_tasks)} requested pairs."
            )
        writer.publish()
    finally:
        writer.close()

def _decode_attr(val):
    if val is None:
        return None
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)

def _compare_gap_penalties(cached_gap, current_gap):
    if cached_gap is None:
        return False
    try:
        cached_arr = np.array(cached_gap, dtype=np.float32).flatten()
        current_arr = np.array(current_gap, dtype=np.float32).flatten()
        return len(cached_arr) == len(current_arr) and np.allclose(cached_arr, current_arr, atol=1e-5)
    except Exception:
        return False

def scan_existing_batches(
    new_N,
    current_checksum,
    model_name,
    saving_mode,
    gap_penalties,
    requested_matmul_precision="ieee_fp32",
    as_array=False,
):
    """The pair keys already computed in RESULTS_DIR: a set, or with as_array a sorted unique int64 array."""
    computed_pairs = set()
    computed_chunks = []
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return np.zeros(0, dtype=np.int64) if as_array else computed_pairs

    batch_files = glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5"))
    if not batch_files:
        return np.zeros(0, dtype=np.int64) if as_array else computed_pairs

    mismatches = []
    found_attrs = {
        "model_name": "Unknown/Legacy",
        "saving_mode": "Unknown/Legacy",
        "gap_penalties": "Unknown/Legacy",
        "embedding_checksum": "Unknown/Legacy",
        "matmul_precision": "ieee_fp32",
    }

    for bf in batch_files:
        try:
            with h5py.File(bf, "r") as hf:
                cached_checksum = _decode_attr(hf.attrs.get("embedding_checksum"))
                cached_model = _decode_attr(hf.attrs.get("model_name"))
                cached_saving_mode = _decode_attr(hf.attrs.get("saving_mode"))
                cached_gaps = hf.attrs.get("gap_penalties")
                cached_precision = _decode_attr(
                    hf.attrs.get("matmul_precision", "ieee_fp32")
                ).strip().lower()
                cached_precision = {
                    "float32": "ieee_fp32",
                    "fp32": "ieee_fp32",
                    "ieee": "ieee_fp32",
                }.get(cached_precision, cached_precision)

                if cached_checksum: found_attrs["embedding_checksum"] = cached_checksum
                if cached_model: found_attrs["model_name"] = cached_model
                if cached_saving_mode: found_attrs["saving_mode"] = cached_saving_mode
                if cached_gaps is not None: found_attrs["gap_penalties"] = list(np.array(cached_gaps, dtype=np.float32).flatten())
                found_attrs["matmul_precision"] = cached_precision

                required_keys = ["i", "j", "l_score", "l_len", "g_score", "g_len"]
                if not all(k in hf for k in required_keys):
                    mismatches.append(f"Missing required datasets in batch file '{os.path.basename(bf)}'")
                    break

                len_i = len(hf["i"])
                if any(len(hf[k]) != len_i for k in ["j", "l_score", "l_len", "g_score", "g_len"]):
                    mismatches.append(f"Dataset length mismatch in batch file '{os.path.basename(bf)}'")
                    break

                if cached_checksum != current_checksum:
                    mismatches.append(f"Checksum mismatch in '{os.path.basename(bf)}' ('{cached_checksum}' vs current '{current_checksum}')")
                    break
                if cached_model != model_name:
                    mismatches.append(f"Model name mismatch in '{os.path.basename(bf)}' ('{cached_model}' vs current '{model_name}')")
                    break
                if cached_saving_mode != saving_mode:
                    mismatches.append(f"Saving mode mismatch in '{os.path.basename(bf)}' ('{cached_saving_mode}' vs current '{saving_mode}')")
                    break
                if not _compare_gap_penalties(cached_gaps, gap_penalties):
                    mismatches.append(f"Gap penalties mismatch in '{os.path.basename(bf)}' ({found_attrs['gap_penalties']} vs current {gap_penalties})")
                    break
                if cached_precision != requested_matmul_precision:
                    mismatches.append(
                        f"Matmul precision mismatch in '{os.path.basename(bf)}' "
                        f"('{cached_precision}' vs '{requested_matmul_precision}')"
                    )
                    break
        except Exception as e:
            mismatches.append(f"Error reading batch file '{os.path.basename(bf)}': {e}")
            break

    if mismatches:
        backup_dir = f"{RESULTS_DIR}_BackUp"
        counter = 1
        while os.path.exists(backup_dir):
            backup_dir = f"{RESULTS_DIR}_BackUp_{counter}"
            counter += 1

        print(f"\n⚠️ WARNING: Existing batch cache directory '{RESULTS_DIR}' contains mismatched or incomplete batches.")
        print(f"  > Mismatch reason: {mismatches[0]}")
        print(f"  > Renaming existing batch folder to: '{backup_dir}'")

        try:
            shutil.move(RESULTS_DIR, backup_dir)
        except Exception as err:
            print(f"⚠️ Error renaming directory {RESULTS_DIR} to {backup_dir}: {err}")

        info_file = os.path.join(backup_dir, "batch_attributes_info.txt")
        try:
            os.makedirs(backup_dir, exist_ok=True)
            with open(info_file, "w", encoding="utf-8", newline="\n") as info_f:
                info_f.write("==========================================================\n")
                info_f.write("NETWORK INJECTION BATCH DIRECTORY BACKUP REPORT\n")
                info_f.write("==========================================================\n")
                info_f.write(f"Backup Timestamp:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                info_f.write(f"Original Directory:  {RESULTS_DIR}\n")
                info_f.write(f"Backup Directory:    {backup_dir}\n\n")
                info_f.write("REASON(S) FOR BACKUP:\n")
                for m in mismatches:
                    info_f.write(f"  - {m}\n")
                info_f.write("\nATTRIBUTES DETECTED IN BACKED-UP BATCHES:\n")
                info_f.write(f"  - Model Name:         {found_attrs['model_name']}\n")
                info_f.write(f"  - Saving Mode:        {found_attrs['saving_mode']}\n")
                info_f.write(f"  - Gap Penalties:      {found_attrs['gap_penalties']}\n")
                info_f.write(f"  - Embedding Checksum: {found_attrs['embedding_checksum']}\n\n")
                info_f.write(f"  - Matmul Precision:   {found_attrs['matmul_precision']}\n\n")
                info_f.write("CURRENT EXECUTION SETTINGS:\n")
                info_f.write(f"  - Model Name:         {model_name}\n")
                info_f.write(f"  - Saving Mode:        {saving_mode}\n")
                info_f.write(f"  - Gap Penalties:      {gap_penalties}\n")
                info_f.write(f"  - Embedding Checksum: {current_checksum}\n")
                info_f.write(f"  - Matmul Precision:   {requested_matmul_precision}\n")
                info_f.write("==========================================================\n")
            print(f"  > Created attribute report: '{info_file}'")
        except Exception as err:
            print(f"⚠️ Error creating info file '{info_file}': {err}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"  > Initialized fresh calculation directory: '{RESULTS_DIR}'\n")
        return np.zeros(0, dtype=np.int64) if as_array else set()

    for bf in batch_files:
        try:
            with h5py.File(bf, "r") as hf:
                if "i" in hf and "j" in hf:
                    # The stored indices are uint32: widen them before the
                    # multiplication, or a pair above 65,535 nodes wraps.
                    arr_i = hf["i"][:].astype(np.int64)
                    arr_j = hf["j"][:].astype(np.int64)
                    pair_ids = np.minimum(arr_i, arr_j) * new_N + np.maximum(arr_i, arr_j)
                    if as_array:
                        computed_chunks.append(pair_ids)
                    else:
                        computed_pairs.update(pair_ids.tolist())
        except Exception as e:
            print(f"Warning: Could not read batch file {bf}: {e}")
    if as_array:
        return _sorted_unique(np.concatenate(computed_chunks)) if computed_chunks else np.zeros(0, dtype=np.int64)
    return computed_pairs

def compile_final_output(
    new_headers,
    seq_lens,
    required_pairs,
    cached_old_pairs,
    new_N,
    old_header_to_idx,
    old_l_score,
    old_l_len,
    old_g_score,
    old_g_len,
    actual_idx_map,
    current_checksum,
    model_name,
    saving_mode,
    gap_penalties,
    matmul_precision="ieee_fp32",
):
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "Cannot publish an alignment network without nonempty model_name metadata."
        )
    model_name = model_name.strip()
    sorted_pairs = _as_sorted_keys(required_pairs)
    cached_pairs = _as_sorted_keys(cached_old_pairs)
    print(f"\n--- Compiling Final HDF5 Output (Keep count: {len(sorted_pairs)}) ---")

    num_kept = len(sorted_pairs)
    idx_dtype = np.uint16 if new_N <= 65535 else np.uint32

    final_i = (sorted_pairs // new_N).astype(idx_dtype)
    final_j = (sorted_pairs % new_N).astype(idx_dtype)
    final_l_score = np.zeros(num_kept, dtype=np.float32)
    final_l_len = np.zeros(num_kept, dtype=np.uint16)
    final_g_score = np.zeros(num_kept, dtype=np.float32)
    final_g_len = np.zeros(num_kept, dtype=np.uint16)

    old_count = len(old_header_to_idx)
    reused = cached_pairs[_members(sorted_pairs, cached_pairs)]
    if reused.size:
        position = _locate(sorted_pairs, reused)[0]
        # Old index of each new sequence (-1 if it is not an old one).
        new_to_old = np.fromiter(
            (old_header_to_idx.get(header, -1) for header in new_headers),
            dtype=np.int64, count=len(new_headers),
        )
        old_u = new_to_old[reused // new_N]
        old_v = new_to_old[reused % new_N]
        if (old_u < 0).any() or (old_v < 0).any():
            raise KeyError("a reused pair has a sequence the old network does not have")
        low = np.minimum(old_u, old_v)
        high = np.maximum(old_u, old_v)
        theoretical_idx = low * old_count - low * (low + 1) // 2 + (high - low - 1)
        actual_idx = actual_idx_map[theoretical_idx]
        final_l_score[position] = old_l_score[actual_idx]
        final_l_len[position] = old_l_len[actual_idx]
        final_g_score[position] = old_g_score[actual_idx]
        final_g_len[position] = old_g_len[actual_idx]

    batch_files = sorted(glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5")))
    for bf in tqdm(batch_files, desc="Filtering computed batches"):
        try:
            with h5py.File(bf, "r") as hf:
                arr_i = hf["i"][:].astype(np.int64)
                arr_j = hf["j"][:].astype(np.int64)
                arr_l_score = hf["l_score"][:]
                arr_l_len = hf["l_len"][:]
                arr_g_score = hf["g_score"][:]
                arr_g_len = hf["g_len"][:]

                pair_ids = np.minimum(arr_i, arr_j) * new_N + np.maximum(arr_i, arr_j)
                position, wanted = _locate(sorted_pairs, pair_ids)
                wanted &= ~_members(cached_pairs, pair_ids)
                # A pair written twice keeps its last row, as the row-by-row loop did.
                position = position[wanted]
                final_l_score[position] = arr_l_score[wanted]
                final_l_len[position] = arr_l_len[wanted]
                final_g_score[position] = arr_g_score[wanted]
                final_g_len[position] = arr_g_len[wanted]
        except Exception as e:
            # Skipping the batch would publish its pairs with zero scores.
            raise NetworkCompilationError(
                f"Error reading {bf} during compilation: {e}"
            ) from e

    print(f"Saving Combined Scores to {FINAL_OUTPUT_NET}...")
    os.makedirs(os.path.dirname(FINAL_OUTPUT_NET), exist_ok=True)

    # Publish only a fully written network; a failed write must not leave a
    # truncated file at the path the Viewer and later jobs load.
    partial_output = FINAL_OUTPUT_NET + ".partial"
    try:
        with h5py.File(partial_output, "w") as hf_out:
            if current_checksum is not None:
                hf_out.attrs["embedding_checksum"] = current_checksum
            hf_out.attrs["model_name"] = model_name
            if saving_mode is not None:
                hf_out.attrs["saving_mode"] = saving_mode
            if gap_penalties is not None:
                hf_out.attrs["gap_penalties"] = np.array(gap_penalties, dtype=np.float32)
            hf_out.attrs["matmul_precision"] = matmul_precision

            dt_str = h5py.string_dtype(encoding='utf-8')
            hf_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
            hf_out.create_dataset("seq_lens", data=np.array(seq_lens, dtype=np.uint16))

            hf_out.create_dataset("i", data=final_i)
            hf_out.create_dataset("j", data=final_j)
            hf_out.create_dataset("l_score", data=final_l_score)
            hf_out.create_dataset("l_len", data=final_l_len)
            hf_out.create_dataset("g_score", data=final_g_score)
            hf_out.create_dataset("g_len", data=final_g_len)
        os.replace(partial_output, FINAL_OUTPUT_NET)
    except BaseException:
        try:
            os.remove(partial_output)
        except OSError:
            pass
        raise

    print("✅ Compilation complete!")

def run_injection():
    """Run or resume the injection; return 1 if it cannot start, None otherwise."""
    try:
        _validate_execution_mode_hardware()
        configure_input_paths()
    except ValueError as error:
        print(f"\n❌ Cannot start Network Injection:\n{error}")
        return 1

    try: set_start_method('spawn')
    except RuntimeError: pass

    print("Loading New Embeddings Metadata...")
    if not os.path.exists(NEW_EMBEDDINGS):
        sys.exit(f"❌ Error: New embeddings file not found at {NEW_EMBEDDINGS}")

    try:
        new_headers, safe_new_headers, seq_lens, manifest = load_embedding_metadata(
            NEW_EMBEDDINGS
        )
    except EmbeddingFileError as error:
        print(f"\n❌ Cannot start Network Injection:\n{error}")
        return 1

    current_model_name = manifest.model_name
    current_saving_mode = manifest.saving_mode
    current_gap_penalties = [LOCAL_GAP_P, GLOBAL_GAP_P]

    new_N = len(new_headers)

    print("Loading Existing Network...")
    if not os.path.exists(OLD_NETWORK):
        sys.exit(f"❌ Error: Old network file not found at {OLD_NETWORK}")
        
    with h5py.File(OLD_NETWORK, "r") as hf_old_net:
        old_network_metadata = validate_network_schema(
            hf_old_net,
            expected_network_type="alignment",
        )
        raw_old_headers = hf_old_net['headers'][:]
        old_headers = [h.decode('utf-8') if isinstance(h, bytes) else h for h in raw_old_headers]
        old_N = len(old_headers)
        old_header_to_idx = {h: i for i, h in enumerate(old_headers)}
        
        old_i = hf_old_net['i'][:]
        old_j = hf_old_net['j'][:]
        
        old_l_score = hf_old_net['l_score'][:]
        old_l_len = hf_old_net['l_len'][:]
        old_g_score = hf_old_net['g_score'][:]
        old_g_len = hf_old_net['g_len'][:]
        # An Edge Pre-filtering network's calibration sample lies below its
        # cut on purpose, so it must not set the cut for the new pairs.
        old_prefilter = Prefilter_Calibration.read_record(hf_old_net)
        inherited_precision = _decode_attr(
            hf_old_net.attrs.get("matmul_precision", "ieee_fp32")
        ).strip().lower()
        inherited_precision = {
            "float32": "ieee_fp32",
            "fp32": "ieee_fp32",
            "ieee": "ieee_fp32",
        }.get(inherited_precision, inherited_precision)
        if inherited_precision not in {"ieee_fp32", "tf32", "bf16"}:
            raise EmbeddingFileError(
                f"Input network has unsupported matmul_precision="
                f"'{inherited_precision}'."
            )

        if "gap_penalties" not in hf_old_net.attrs:
            raise EmbeddingFileError(
                f"Input network '{os.path.basename(OLD_NETWORK)}' is missing the required "
                "'gap_penalties' attribute. Legacy network files without embedded gap penalty "
                "metadata are unsupported. Re-run Align_Similarity_Matrix to generate a compatible network file."
            )

        old_gaps = hf_old_net.attrs["gap_penalties"]
        inherited_local = float(old_gaps[0])
        inherited_global = float(old_gaps[1])
        globals()["LOCAL_GAP_P"] = inherited_local
        globals()["GLOBAL_GAP_P"] = inherited_global
        current_gap_penalties = [inherited_local, inherited_global]
        print(f"  > Inherited gap penalties from input network: Local={inherited_local}, Global={inherited_global}")

    if inherited_precision == "tf32":
        available = Hardware_Utils.get_available_devices()
        selected = Hardware_Utils.resolve_device_selection(
            DEVICE_SELECTION, available
        )
        eligible = [
            candidate for candidate in available
            if candidate.backend == "cuda" and is_nvidia_cuda(candidate.device)
        ]
        if selected is not None:
            eligible = [
                selected
            ] if selected.backend == "cuda" and is_nvidia_cuda(selected.device) else []
        if not eligible:
            raise EmbeddingFileError(
                "The input network uses TF32. Network Injection requires an "
                "NVIDIA CUDA device to calculate numerically compatible new edges."
            )
    print(f"  > Inherited matmul precision: {inherited_precision}")

    configure_output_paths(old_network_metadata.model_name)
    if os.path.exists(FINAL_OUTPUT_NET):
        try:
            with h5py.File(FINAL_OUTPUT_NET, "r") as hf_existing:
                completed_precision = _decode_attr(
                    hf_existing.attrs.get("matmul_precision", "ieee_fp32")
                ).strip().lower()
                completed_precision = {
                    "float32": "ieee_fp32",
                    "fp32": "ieee_fp32",
                    "ieee": "ieee_fp32",
                }.get(completed_precision, completed_precision)
        except Exception as error:
            raise EmbeddingFileError(
                f"Cannot inspect existing output precision: {error}"
            ) from error
        if completed_precision != inherited_precision:
            raise EmbeddingFileError(
                "Completed output precision conflict: input network uses "
                f"'{inherited_precision}', but the existing output uses "
                f"'{completed_precision}'. The output was not changed."
            )

    print("Calculating checksum of input embedding file...")
    current_checksum = calculate_file_hash(NEW_EMBEDDINGS)
    print(f"  > Checksum: {current_checksum}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    theoretical_old_total = (old_N * (old_N - 1)) // 2
    total_old_pairs = len(old_i)
    
    old_i_64 = old_i.astype(np.int64)
    old_j_64 = old_j.astype(np.int64)
    theoretical_flat_indices = (old_i_64 * old_N) - (old_i_64 * (old_i_64 + 1) // 2) + (old_j_64 - old_i_64 - 1)

    actual_idx_map = _OldEdgeIndex(theoretical_flat_indices)
    del theoretical_flat_indices

    # New index of each old sequence (-1 if it was dropped), and of each old edge's two ends.
    new_header_to_idx = {h: idx for idx, h in enumerate(new_headers)}
    old_to_new = np.fromiter(
        (new_header_to_idx.get(h, -1) for h in old_headers), dtype=np.int64, count=old_N
    )
    edge_new_u = old_to_new[old_i_64]
    edge_new_v = old_to_new[old_j_64]
    del old_to_new, old_i_64, old_j_64

    is_sparse_old = total_old_pairs < theoretical_old_total
    cos_sim = None
    threshold_cos = -1.0
    
    if is_sparse_old:
        percent_exists = (total_old_pairs / theoretical_old_total) * 100
        print(f"\n--- 🛠️ Sparse Input Network Detected: {total_old_pairs} / {theoretical_old_total} edges exist ({percent_exists:.2f}%) ---")
        
        print("Loading mean embeddings for new sequence set...")
        mean_embs = []
        with h5py.File(NEW_EMBEDDINGS, "r") as hf_new_emb:
            for h in tqdm(safe_new_headers, desc="Calculating mean embeddings"):
                emb = hf_new_emb["embeddings"][h][:]
                mean_emb = np.mean(emb, axis=0)
                mean_embs.append(mean_emb)
        mean_embs = np.array(mean_embs, dtype=np.float32)
        
        norms = np.linalg.norm(mean_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        norm_embs = mean_embs / norms
        print("Computing all-vs-all cosine similarities...")
        cos_sim = np.clip(np.dot(norm_embs, norm_embs.T), -1.0, 1.0)
        
        calibration_edges = np.zeros(total_old_pairs, dtype=bool)
        if old_prefilter is not None and len(old_prefilter.sample):
            sample_keys = Prefilter_Calibration.pair_keys(
                old_prefilter.sample.rows, old_prefilter.sample.cols, old_N
            )
            edge_keys = Prefilter_Calibration.pair_keys(old_i, old_j, old_N)
            calibration_edges = np.isin(edge_keys, sample_keys)
            del edge_keys
            print(
                f"  > Leaving out the input network's {int(np.count_nonzero(calibration_edges)):,} calibration "
                "pairs, which its prefilter aligned below its cut."
            )

        if total_old_pairs > 0:
            min_cos_val, min_cos_k = _lowest_edge_cosine(
                cos_sim, edge_new_u, edge_new_v, calibration_edges
            )

            if min_cos_k != -1:
                threshold_cos = min_cos_val
                h_i_min = old_headers[old_i[min_cos_k]]
                h_j_min = old_headers[old_j[min_cos_k]]
                print(f"  > Lowest cosine similarity in input network: {threshold_cos:.6f} (between '{h_i_min}' and '{h_j_min}', score: {old_l_score[min_cos_k]:.4f})")
            else:
                threshold_cos = 0.0
                print("  > Warning: No old network edges could be matched in new headers.")
                
            print(f"  > Setting cosine similarity prefiltering threshold to: {threshold_cos:.6f}")
            
            print("  > Scanning original network for edges below the threshold...")
            reported_count, reported_edges = _edges_below_cosine(
                cos_sim, edge_new_u, edge_new_v, calibration_edges, threshold_cos - 1e-5
            )
            for k, sim_val in reported_edges:
                u_h = old_headers[old_i[k]]
                v_h = old_headers[old_j[k]]
                print(f"    * Warning: Edge '{u_h}' - '{v_h}' has cosine similarity {sim_val:.6f} < threshold {threshold_cos:.6f} (score: {old_l_score[k]:.4f})")
            if reported_count > 0:
                print(f"  > Total edges in original network below threshold: {reported_count}")
            else:
                print("  > No edges in original network are below the threshold. (As expected!)")
        else:
            print("  > Warning: Old network is empty.")
        print("---------------------------------------------------------------------------\n")

    total_new_pairs = (new_N * (new_N - 1)) // 2
    old_headers_set = set(old_headers)
    
    cached_old_pairs = _cached_pair_keys(edge_new_u, edge_new_v, new_N)
    del edge_new_u, edge_new_v
    new_is_old = np.fromiter(
        (h in old_headers_set for h in new_headers), dtype=bool, count=new_N
    )
    required_pairs = _required_pair_keys(
        new_N, new_is_old, cached_old_pairs, cos_sim, threshold_cos
    )
    del new_is_old

    effective_total_pairs = len(required_pairs)

    computed_pairs = scan_existing_batches(
        new_N,
        current_checksum,
        current_model_name,
        current_saving_mode,
        current_gap_penalties,
        inherited_precision,
        as_array=True,
    )

    existing_batches = glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5"))
    batch_ids = []
    for f in existing_batches:
        base = os.path.basename(f)
        try:
            num = int(base.split("_")[1].split(".")[0])
            batch_ids.append(num)
        except:
            pass
    next_batch_id = max(batch_ids) + 1 if batch_ids else 0

    pending_keys = _pending_keys(
        required_pairs, cached_old_pairs, computed_pairs, len(safe_new_headers)
    )

    def iter_pending_tasks():
        return _iter_pending_keys(safe_new_headers, pending_keys, len(safe_new_headers))

    pending_counts = np.bincount(pending_keys // new_N, minlength=new_N).astype(np.int64) if new_N else np.zeros(0, dtype=np.int64)
    num_tasks = int(pending_counts.sum())
    print(f"Total Required pairs: {effective_total_pairs}")
    print(f"Pre-existing cached pairs: {int(np.count_nonzero(_members(required_pairs, cached_old_pairs)))}")
    print(f"Already computed pairs: {int(np.count_nonzero(_members(required_pairs, computed_pairs)))}")
    print(f"Pairs queued for calculation: {num_tasks}")

    if num_tasks > 0:
        embedding_store = EmbeddingTileStore(
            NEW_EMBEDDINGS, safe_new_headers, HOST_CACHE_GB
        )
        if embedding_store.fully_cached:
            print(
                f"[Memory] Using packed host cache "
                f"({embedding_store.cached_bytes / (1024 ** 3):.2f} GiB)."
            )
        else:
            print("[Memory] Using byte-bounded embedding tiles.")
        Benchmark_Record.record_host_cache(embedding_store, HOST_CACHE_GB)
        sequence_lengths = [shape[0] for shape in embedding_store.shapes]

        def pending_columns_for_row(row):
            base = int(row) * new_N
            low, high = np.searchsorted(pending_keys, (base, base + new_N))
            return pending_keys[low:high] - base

        precision_tasks = None
        if inherited_precision == "bf16":
            manual_candidate = Hardware_Utils.resolve_device_selection(
                DEVICE_SELECTION,
                Hardware_Utils.get_available_devices(),
            )
            if manual_candidate is not None and manual_candidate.is_cpu:
                precision_warmup = []
                precision_timed = [next(iter_pending_tasks())]
            else:
                accelerator_half, _cpu_half = _benchmark_half_sizes()
                precision_warmup, precision_timed = matched_benchmark_task_halves(
                    safe_new_headers,
                    sequence_lengths,
                    pending_counts,
                    pending_columns_for_row,
                    accelerator_half,
                )
            precision_tasks = precision_warmup + precision_timed
        pending_iterator = iter_pending_tasks()
        first_batch = list(islice(pending_iterator, int(BATCH_SIZE)))
        ranked_plans = _benchmark_injection_plans(
            first_batch,
            WORKERS,
            NEW_EMBEDDINGS,
            embedding_store,
            sequence_lengths,
            inherited_precision,
            precision_tasks=precision_tasks,
        )
        active_plan_index = 0

        def publish_batch(batch, current_id):
            nonlocal active_plan_index
            last_error = None
            while active_plan_index < len(ranked_plans):
                selected_plan = ranked_plans[active_plan_index]
                try:
                    process_batch(
                        batch,
                        current_id,
                        WORKERS,
                        NEW_EMBEDDINGS,
                        current_checksum,
                        current_model_name,
                        current_saving_mode,
                        current_gap_penalties,
                        device=selected_plan.candidate.device,
                        accelerator_workers=(
                            None if selected_plan.candidate.is_cpu
                            else selected_plan.lanes
                        ),
                        execution_variant=selected_plan.variant,
                        matmul_precision=inherited_precision,
                        embedding_store=embedding_store,
                        sequence_lengths=sequence_lengths,
                    )
                    return
                except (RuntimeError, NotImplementedError, MemoryError) as error:
                    last_error = error
                    active_plan_index += 1
                    if active_plan_index < len(ranked_plans):
                        fallback = ranked_plans[active_plan_index]
                        print(
                            f"[Hardware] {type(error).__name__}; retrying "
                            f"uncommitted batch with {fallback.candidate.display_name}, "
                            f"{fallback.variant}, lanes={fallback.lanes}."
                        )
            raise RuntimeError(
                "Every benchmarked Network Injection plan failed."
            ) from last_error

        current_batch = []
        batch_id = next_batch_id
        
        pbar = tqdm(total=math.ceil(num_tasks / BATCH_SIZE), desc="Progress", unit="batch")
        for t in chain(first_batch, pending_iterator):
            current_batch.append(t)
            if len(current_batch) >= BATCH_SIZE:
                publish_batch(current_batch, batch_id)
                batch_id += 1; pbar.update(1)
                current_batch = []
                
        if current_batch:
            publish_batch(current_batch, batch_id)
            pbar.update(1)
        pbar.close()
        
    compile_final_output(
        new_headers=new_headers,
        seq_lens=seq_lens,
        required_pairs=required_pairs,
        cached_old_pairs=cached_old_pairs,
        new_N=new_N,
        old_header_to_idx=old_header_to_idx,
        old_l_score=old_l_score,
        old_l_len=old_l_len,
        old_g_score=old_g_score,
        old_g_len=old_g_len,
        actual_idx_map=actual_idx_map,
        current_checksum=current_checksum,
        model_name=current_model_name,
        saving_mode=current_saving_mode,
        gap_penalties=current_gap_penalties,
        matmul_precision=inherited_precision,
    )

def main(argv=None):
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    try:
        exit_code = run_injection()
    except NetworkCompilationError as error:
        print(
            f"\n❌ Cannot compile the final network:\n{error}\n"
            "No network file was written. Delete that batch file and run "
            "the job again to recompute its pairs."
        )
        return 1
    # The MCP job runner reports exit code 0 as a successful job.
    return exit_code or 0


if __name__ == "__main__":
    raise SystemExit(main())
