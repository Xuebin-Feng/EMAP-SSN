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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import Hardware_Utils
from Alignment_Score_Kernels import global_local_scores

try:
    from Embedding_HDF5 import read_embedding_manifest
except ModuleNotFoundError:
    from src.utilities.Embedding_HDF5 import read_embedding_manifest

# ==========================================
# CONFIGURATION
# ==========================================
# INPUTS (Filenames only, provided by GUI)
OLD_NETWORK = None 
NEW_EMBEDDINGS = None 

# SETTINGS
BATCH_SIZE = 500000 
WORKERS = 12 
ACCELERATOR_LANES = "auto"
ACCELERATOR_TUNE_PAIRS = 256
LOCAL_GAP_P = None
GLOBAL_GAP_P = None 

# DIRECTORIES
NETWORK_DIR = os.path.join("..", "Input_Files", "Networks_EValues")
EMBED_DIR = os.path.join("..", "Embeddings")

# --- JSON Settings Override ---
import json
import ast

# Automatically calculate the root directory of the SSN project for the current PC
# (Assuming utility scripts are located in the /utilities/ folder)
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

# --- DYNAMIC INFERENCE ---
import re

# 1. Dynamically parse names from the input files provided by the GUI
_old_seq_set = "UnknownOld"
_model_name = "UnknownModel"
match_old = re.search(r"^(.*)_\[(.*)\]_network\.h5$", OLD_NETWORK)
if match_old:
    _old_seq_set = match_old.group(1)
    _model_name = match_old.group(2)

_new_seq_set = "UnknownNew"
match_new = re.search(r"^(.*)_\[(.*)\]_embeddings\.h5$", NEW_EMBEDDINGS)
if match_new:
    _new_seq_set = match_new.group(1)

# 2. Build Full Input Paths
OLD_NETWORK = os.path.join(NETWORK_DIR, OLD_NETWORK) if NETWORK_DIR else ""
NEW_EMBEDDINGS = os.path.join(EMBED_DIR, NEW_EMBEDDINGS) if EMBED_DIR else ""

# 3. Build Full Output Paths
RESULTS_DIR = os.path.join(NETWORK_DIR, f"{_new_seq_set}_[{_model_name}]_network_temp") if NETWORK_DIR else ""
FINAL_OUTPUT_NET = os.path.join(NETWORK_DIR, f"{_new_seq_set}_[{_model_name}]_network.h5") if NETWORK_DIR else ""
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
    t_i = torch.as_tensor(emb_i, device=device, dtype=torch.float16)
    t_j = torch.as_tensor(emb_j, device=device, dtype=torch.float16)
    t_i_norm = torch.nn.functional.normalize(t_i, p=2, dim=-1)
    t_j_norm = torch.nn.functional.normalize(t_j, p=2, dim=-1)
    cos_sim = torch.mm(t_i_norm, t_j_norm.T)
    dist_mat = 1.0 - cos_sim
    sim_mat = torch.exp(-dist_mat)
    
    epsilon = 1e-8
    row_mean = sim_mat.mean(dim=1, keepdim=True)
    row_std = sim_mat.std(dim=1, keepdim=True)
    col_mean = sim_mat.mean(dim=0, keepdim=True)
    col_std = sim_mat.std(dim=0, keepdim=True)
    
    z_r = (sim_mat - row_mean) / (row_std + epsilon)
    z_c = (sim_mat - col_mean) / (col_std + epsilon)
    final_score = (z_r + z_c) / 2.0
    
    return final_score.to(dtype=torch.float32, device="cpu").numpy()

# %% =======================================
# HDF5 WORKER INITIALIZATION
# ==========================================
worker_hf = None
worker_device = None

def init_worker(h5_path):
    global worker_hf, worker_device
    worker_hf = h5py.File(h5_path, "r", libver='latest', swmr=True)
    worker_device = torch.device("cpu")


# %% =======================================
# PORTABLE ACCELERATOR/CPU PIPELINE
# ==========================================
def calculate_alignment_data(args):
    idx_i, idx_j, matrix = args
    g_raw, g_len, l_raw, l_len = global_local_scores(
        matrix,
        GLOBAL_GAP_P,
        LOCAL_GAP_P,
    )

    return (idx_i, idx_j, l_raw, l_len, g_raw, g_len)


def calculate_cpu_pair(args):
    idx_i, idx_j, safe_h_i, safe_h_j = args
    global worker_hf, worker_device

    emb_i = worker_hf["embeddings"][safe_h_i][:]
    emb_j = worker_hf["embeddings"][safe_h_j][:]
    matrix = compute_score_matrix_torch(emb_i, emb_j, worker_device)
    return calculate_alignment_data((idx_i, idx_j, matrix))


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


def _compute_accelerated_matrix(args):
    idx_i, idx_j, emb_i, emb_j, device = args
    with torch.inference_mode():
        with _stream_context(device):
            matrix = compute_score_matrix_torch(emb_i, emb_j, device)
    return idx_i, idx_j, matrix


def _run_accelerated_pipeline(
    tasks,
    workers,
    input_h5,
    device,
    batch_id,
    lanes,
    show_progress,
):
    results = []
    gpu_pending = set()
    cpu_pending = set()
    ready_for_cpu = deque()
    ready_limit = max(lanes, min(workers, 8))
    task_iterator = iter(tasks)
    tasks_exhausted = False
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

    with h5py.File(input_h5, "r", libver="latest", swmr=True) as hf, \
            ThreadPoolExecutor(
                max_workers=lanes,
                thread_name_prefix="injection-accelerator",
            ) as accelerator_executor, \
            ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="injection-cpu",
            ) as cpu_executor, \
            progress_context as progress:
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
                try:
                    (
                        idx_i,
                        idx_j,
                        safe_h_i,
                        safe_h_j,
                    ) = next(task_iterator)
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
                        ),
                    )
                )

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
                if progress is not None:
                    progress.update(1)

    return results


def _select_lanes(
    tasks,
    workers,
    input_h5,
    device,
    batch_id,
):
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
        f"  > Auto-tuning accelerator lanes on {cache_key[1]} "
        f"using {count} representative pairs..."
    )
    _run_accelerated_pipeline(
        sample,
        workers,
        input_h5,
        device,
        batch_id,
        1,
        False,
    )

    rates = {}
    for lanes in candidates:
        started = time.perf_counter()
        try:
            _run_accelerated_pipeline(
                sample,
                workers,
                input_h5,
                device,
                batch_id,
                lanes,
                False,
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
            f"{lanes}: {rate:.1f} pairs/s"
            for lanes, rate in sorted(rates.items())
        )
    )
    print(f"  > Selected {selected} accelerator lane(s).")
    return selected


def process_accelerated_tasks(
    tasks,
    workers,
    input_h5,
    device,
    batch_id,
):
    lanes = _select_lanes(
        tasks,
        workers,
        input_h5,
        device,
        batch_id,
    )
    return _run_accelerated_pipeline(
        tasks,
        workers,
        input_h5,
        device,
        batch_id,
        lanes,
        True,
    )


def process_cpu_tasks(
    tasks,
    workers,
    input_h5,
    batch_id,
):
    results = []
    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(input_h5,),
    ) as pool:
        iterator = pool.imap_unordered(
            calculate_cpu_pair,
            tasks,
            chunksize=10,
        )
        for result in tqdm(
            iterator,
            total=len(tasks),
            desc=f"  Batch {batch_id}",
            leave=False,
        ):
            results.append(result)
    return results


def process_batch(
    batch_tasks,
    batch_id,
    workers,
    new_emb_path,
    embedding_checksum,
    model_name,
    saving_mode,
    gap_penalties,
):
    output_filename = os.path.join(RESULTS_DIR, f"batch_{batch_id:05d}.h5")
    device = Hardware_Utils.get_optimal_device()
    if _uses_accelerator(device):
        results = process_accelerated_tasks(
            batch_tasks,
            workers,
            new_emb_path,
            device,
            batch_id,
        )
    else:
        results = process_cpu_tasks(
            batch_tasks,
            workers,
            new_emb_path,
            batch_id,
        )
            
    # Save as uncompressed HDF5 for easy access
    rows_i = [r[0] for r in results]
    rows_j = [r[1] for r in results]
    l_scores = [r[2] for r in results]
    l_lens = [r[3] for r in results]
    g_scores = [r[4] for r in results]
    g_lens = [r[5] for r in results]
    
    with h5py.File(output_filename, "w") as hf:
        if embedding_checksum is not None:
            hf.attrs["embedding_checksum"] = embedding_checksum
        if model_name is not None:
            hf.attrs["model_name"] = model_name
        if saving_mode is not None:
            hf.attrs["saving_mode"] = saving_mode
        if gap_penalties is not None:
            hf.attrs["gap_penalties"] = np.array(gap_penalties, dtype=np.float32)

        hf.create_dataset("i", data=np.array(rows_i, dtype=np.uint32))
        hf.create_dataset("j", data=np.array(rows_j, dtype=np.uint32))
        hf.create_dataset("l_score", data=np.array(l_scores, dtype=np.float32))
        hf.create_dataset("l_len", data=np.array(l_lens, dtype=np.uint16))
        hf.create_dataset("g_score", data=np.array(g_scores, dtype=np.float32))
        hf.create_dataset("g_len", data=np.array(g_lens, dtype=np.uint16))

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
):
    computed_pairs = set()
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return computed_pairs
    
    batch_files = glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5"))
    if not batch_files:
        return computed_pairs

    mismatches = []
    found_attrs = {
        "model_name": "Unknown/Legacy",
        "saving_mode": "Unknown/Legacy",
        "gap_penalties": "Unknown/Legacy",
        "embedding_checksum": "Unknown/Legacy",
    }

    for bf in batch_files:
        try:
            with h5py.File(bf, "r") as hf:
                cached_checksum = _decode_attr(hf.attrs.get("embedding_checksum"))
                cached_model = _decode_attr(hf.attrs.get("model_name"))
                cached_saving_mode = _decode_attr(hf.attrs.get("saving_mode"))
                cached_gaps = hf.attrs.get("gap_penalties")

                if cached_checksum: found_attrs["embedding_checksum"] = cached_checksum
                if cached_model: found_attrs["model_name"] = cached_model
                if cached_saving_mode: found_attrs["saving_mode"] = cached_saving_mode
                if cached_gaps is not None: found_attrs["gap_penalties"] = list(np.array(cached_gaps, dtype=np.float32).flatten())

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
            with open(info_file, "w", encoding="utf-8") as info_f:
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
                info_f.write("CURRENT EXECUTION SETTINGS:\n")
                info_f.write(f"  - Model Name:         {model_name}\n")
                info_f.write(f"  - Saving Mode:        {saving_mode}\n")
                info_f.write(f"  - Gap Penalties:      {gap_penalties}\n")
                info_f.write(f"  - Embedding Checksum: {current_checksum}\n")
                info_f.write("==========================================================\n")
            print(f"  > Created attribute report: '{info_file}'")
        except Exception as err:
            print(f"⚠️ Error creating info file '{info_file}': {err}")

        os.makedirs(RESULTS_DIR, exist_ok=True)
        print(f"  > Initialized fresh calculation directory: '{RESULTS_DIR}'\n")
        return set()

    for bf in batch_files:
        try:
            with h5py.File(bf, "r") as hf:
                if "i" in hf and "j" in hf:
                    arr_i = hf["i"][:]
                    arr_j = hf["j"][:]
                    for i_val, j_val in zip(arr_i, arr_j):
                        computed_pairs.add(min(i_val, j_val) * new_N + max(i_val, j_val))
        except Exception as e:
            print(f"Warning: Could not read batch file {bf}: {e}")
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
):
    print(f"\n--- Compiling Final HDF5 Output (Keep count: {len(required_pairs)}) ---")
    
    num_kept = len(required_pairs)
    idx_dtype = np.uint16 if new_N <= 65535 else np.uint32
    
    final_i = np.zeros(num_kept, dtype=idx_dtype)
    final_j = np.zeros(num_kept, dtype=idx_dtype)
    final_l_score = np.zeros(num_kept, dtype=np.float32)
    final_l_len = np.zeros(num_kept, dtype=np.uint16)
    final_g_score = np.zeros(num_kept, dtype=np.float32)
    final_g_len = np.zeros(num_kept, dtype=np.uint16)
            
    sorted_pairs = sorted(list(required_pairs))
    pair_id_to_final_idx = {}
    for idx, pair_id in enumerate(sorted_pairs):
        i = pair_id // new_N
        j = pair_id % new_N
        final_i[idx] = i
        final_j[idx] = j
        pair_id_to_final_idx[pair_id] = idx
        
    old_headers_set = set(old_header_to_idx.keys())
    for pair_id in cached_old_pairs:
        if pair_id in pair_id_to_final_idx:
            idx = pair_id_to_final_idx[pair_id]
            i = pair_id // new_N
            j = pair_id % new_N
            h_i = new_headers[i]
            h_j = new_headers[j]
            u = old_header_to_idx[h_i]
            v = old_header_to_idx[h_j]
            theoretical_idx = int(u * len(old_headers_set) - u * (u + 1) // 2 + (v - u - 1)) if u < v else int(v * len(old_headers_set) - v * (v + 1) // 2 + (u - v - 1))
            actual_idx = actual_idx_map[theoretical_idx]
            
            final_l_score[idx] = old_l_score[actual_idx]
            final_l_len[idx] = old_l_len[actual_idx]
            final_g_score[idx] = old_g_score[actual_idx]
            final_g_len[idx] = old_g_len[actual_idx]
                
    batch_files = sorted(glob.glob(os.path.join(glob.escape(RESULTS_DIR), "batch_*.h5")))
    for bf in tqdm(batch_files, desc="Filtering computed batches"):
        try:
            with h5py.File(bf, "r") as hf:
                if "i" not in hf or "j" not in hf:
                    continue
                arr_i = hf["i"][:]
                arr_j = hf["j"][:]
                arr_l_score = hf["l_score"][:]
                arr_l_len = hf["l_len"][:]
                arr_g_score = hf["g_score"][:]
                arr_g_len = hf["g_len"][:]
                    
                for k in range(len(arr_i)):
                    i_val = arr_i[k]
                    j_val = arr_j[k]
                    pair_id = min(i_val, j_val) * new_N + max(i_val, j_val)
                    if pair_id in pair_id_to_final_idx and pair_id not in cached_old_pairs:
                        idx = pair_id_to_final_idx[pair_id]
                        final_l_score[idx] = arr_l_score[k]
                        final_l_len[idx] = arr_l_len[k]
                        final_g_score[idx] = arr_g_score[k]
                        final_g_len[idx] = arr_g_len[k]
        except Exception as e:
            print(f"Warning: Error reading {bf} during compilation: {e}")
            
    print(f"Saving Combined Scores to {FINAL_OUTPUT_NET}...")
    os.makedirs(os.path.dirname(FINAL_OUTPUT_NET), exist_ok=True)
    with h5py.File(FINAL_OUTPUT_NET, "w") as hf_out:
        if current_checksum is not None:
            hf_out.attrs["embedding_checksum"] = current_checksum
        if model_name is not None:
            hf_out.attrs["model_name"] = model_name
        if saving_mode is not None:
            hf_out.attrs["saving_mode"] = saving_mode
        if gap_penalties is not None:
            hf_out.attrs["gap_penalties"] = np.array(gap_penalties, dtype=np.float32)

        dt_str = h5py.string_dtype(encoding='utf-8')
        hf_out.create_dataset("headers", data=np.array(new_headers, dtype=object), dtype=dt_str)
        hf_out.create_dataset("seq_lens", data=np.array(seq_lens, dtype=np.uint16))
        
        hf_out.create_dataset("i", data=final_i)
        hf_out.create_dataset("j", data=final_j)
        hf_out.create_dataset("l_score", data=final_l_score)
        hf_out.create_dataset("l_len", data=final_l_len)
        hf_out.create_dataset("g_score", data=final_g_score)
        hf_out.create_dataset("g_len", data=final_g_len)
            
    print("✅ Compilation complete!")

def run_injection():
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
        return

    current_model_name = manifest.model_name
    current_saving_mode = manifest.saving_mode
    current_gap_penalties = [LOCAL_GAP_P, GLOBAL_GAP_P]

    new_N = len(new_headers)

    print("Calculating checksum of input embedding file...")
    current_checksum = calculate_file_hash(NEW_EMBEDDINGS)
    print(f"  > Checksum: {current_checksum}")
            
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Existing Network...")
    if not os.path.exists(OLD_NETWORK):
        sys.exit(f"❌ Error: Old network file not found at {OLD_NETWORK}")
        
    with h5py.File(OLD_NETWORK, "r") as hf_old_net:
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

    theoretical_old_total = (old_N * (old_N - 1)) // 2
    total_old_pairs = len(old_i)
    
    actual_idx_map = np.full(theoretical_old_total, -1, dtype=np.int64)
    old_i_64 = old_i.astype(np.int64)
    old_j_64 = old_j.astype(np.int64)
    theoretical_flat_indices = (old_i_64 * old_N) - (old_i_64 * (old_i_64 + 1) // 2) + (old_j_64 - old_i_64 - 1)
    
    actual_idx_map[theoretical_flat_indices] = np.arange(total_old_pairs)
    
    exists_in_old = np.zeros(theoretical_old_total, dtype=bool)
    exists_in_old[theoretical_flat_indices] = True

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
        cos_sim = np.dot(norm_embs, norm_embs.T)
        
        new_header_to_idx = {h: idx for idx, h in enumerate(new_headers)}
        
        if total_old_pairs > 0:
            min_cos_val = 1.0
            min_cos_k = -1
            
            for k in range(total_old_pairs):
                u_h = old_headers[old_i[k]]
                v_h = old_headers[old_j[k]]
                if u_h in new_header_to_idx and v_h in new_header_to_idx:
                    u_new_idx = new_header_to_idx[u_h]
                    v_new_idx = new_header_to_idx[v_h]
                    sim_val = float(cos_sim[u_new_idx, v_new_idx])
                    if sim_val < min_cos_val:
                        min_cos_val = sim_val
                        min_cos_k = k
            
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
            reported_count = 0
            for k in range(total_old_pairs):
                u_h = old_headers[old_i[k]]
                v_h = old_headers[old_j[k]]
                if u_h in new_header_to_idx and v_h in new_header_to_idx:
                    u_new_idx = new_header_to_idx[u_h]
                    v_new_idx = new_header_to_idx[v_h]
                    sim_val = cos_sim[u_new_idx, v_new_idx]
                    if sim_val < threshold_cos - 1e-5:
                        if reported_count < 10:
                            print(f"    * Warning: Edge '{u_h}' - '{v_h}' has cosine similarity {sim_val:.6f} < threshold {threshold_cos:.6f} (score: {old_l_score[k]:.4f})")
                        reported_count += 1
            if reported_count > 0:
                print(f"  > Total edges in original network below threshold: {reported_count}")
            else:
                print("  > No edges in original network are below the threshold. (As expected!)")
        else:
            print("  > Warning: Old network is empty.")
        print("---------------------------------------------------------------------------\n")

    total_new_pairs = (new_N * (new_N - 1)) // 2
    old_headers_set = set(old_headers)
    
    cached_old_pairs = set()
    old_header_to_new_idx = {h: idx for idx, h in enumerate(new_headers)}
    for k in range(total_old_pairs):
        u_h = old_headers[old_i[k]]
        v_h = old_headers[old_j[k]]
        if u_h in old_header_to_new_idx and v_h in old_header_to_new_idx:
            u_idx = old_header_to_new_idx[u_h]
            v_idx = old_header_to_new_idx[v_h]
            cached_old_pairs.add(min(u_idx, v_idx) * new_N + max(u_idx, v_idx))
            
    required_pairs = set()
    for i in range(new_N):
        h_i = new_headers[i]
        is_old_i = h_i in old_headers_set
        for j in range(i + 1, new_N):
            h_j = new_headers[j]
            is_old_j = h_j in old_headers_set
            
            keep = False
            if is_old_i and is_old_j:
                pair_id = i * new_N + j
                if pair_id in cached_old_pairs:
                    keep = True
            else:
                if not is_sparse_old or (cos_sim is not None and cos_sim[i, j] >= threshold_cos):
                    keep = True
                    
            if keep:
                required_pairs.add(i * new_N + j)
                
    effective_total_pairs = len(required_pairs)

    computed_pairs = scan_existing_batches(
        new_N,
        current_checksum,
        current_model_name,
        current_saving_mode,
        current_gap_penalties,
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

    tasks_to_compute = []
    for i in range(new_N):
        h_i = new_headers[i]
        for j in range(i + 1, new_N):
            pair_id = i * new_N + j
            if pair_id in required_pairs and pair_id not in cached_old_pairs and pair_id not in computed_pairs:
                tasks_to_compute.append((i, j, safe_new_headers[i], safe_new_headers[j]))

    num_tasks = len(tasks_to_compute)
    print(f"Total Required pairs: {effective_total_pairs}")
    print(f"Pre-existing cached pairs: {len(cached_old_pairs & required_pairs)}")
    print(f"Already computed pairs: {len(computed_pairs & required_pairs)}")
    print(f"Pairs queued for calculation: {num_tasks}")

    if num_tasks > 0:
        current_batch = []
        batch_id = next_batch_id
        
        pbar = tqdm(total=math.ceil(num_tasks / BATCH_SIZE), desc="Progress", unit="batch")
        for t in tasks_to_compute:
            current_batch.append(t)
            if len(current_batch) >= BATCH_SIZE:
                process_batch(
                    current_batch,
                    batch_id,
                    WORKERS,
                    NEW_EMBEDDINGS,
                    current_checksum,
                    current_model_name,
                    current_saving_mode,
                    current_gap_penalties,
                )
                batch_id += 1; pbar.update(1)
                current_batch = []
                
        if current_batch:
            process_batch(
                current_batch,
                batch_id,
                WORKERS,
                NEW_EMBEDDINGS,
                current_checksum,
                current_model_name,
                current_saving_mode,
                current_gap_penalties,
            )
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
    )

if __name__ == "__main__":
    run_injection()
