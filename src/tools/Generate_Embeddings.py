"""
File: Generate_Embeddings.py
===================================
Description:
This script acts as the foundation for the structural similarity pipeline. It ingests a standard FASTA file 
containing raw amino acid sequences and utilizes large language models (pLM) to convert each sequence into 
a mathematically dense, high-dimensional floating-point representation (embedding).

Input:
- A text-based FASTA file containing raw protein sequence strings (`INPUT_FASTA`).

Output:
- A comprehensive, serialized HDF5 database containing the structural embedding arrays for every sequence, metadata, and 
precisely matched order arrays (`OUTPUT_HDF5`).

Settings:
- SEQUENCE_SET: Defines the input FASTA file to target.
- MODEL_NAME: The protein language model identifier to download from HuggingFace and load into VRAM. Supported models include 
  the local Evolutionary Scale Modeling families (`esmc_300m`, `esmc_600m`), the remote API-backed
  ESMC 6B model (`esmc_6b`), and the Rostlab families (`prot_bert`, `ProstT5`).
- SAVING_MODE: Determines data precision. `float16` halves HDF5 file size and RAM requirements by slightly reducing gradient precision, 
  which is recommended for massive datasets. `float32` uses standard uncompressed precision.

Algorithm:
1. Sequentially parses and sanitizes the target FASTA records in RAM.
2. Identifies PyTorch hardware acceleration (CUDA/XPU/CPU) and allocates the massive neural networks accordingly.
3. Initializes a new HDF5 file stream in append mode ("a"), checking for existing embeddings to allow seamless resuming.
4. Iterates linearly over the sanitized sequences and passes them into the loaded neural network.
5. The model strips start/stop tokens internally and isolates the output matrices characterizing every residue in the sequence.
6. The resultant PyTorch tensor is demoted to a Numpy array, cast to the selected precision (`float16/float32`), and streamed 
   directly to disk under a sanitized header name key to prevent RAM overflow.
"""
# %% Import Necessary Libraries
import os
try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap
import time
from tqdm import tqdm
import numpy as np
import torch
import h5py
from utilities import Hardware_Utils

from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Embedding_HDF5 import (
    create_metadata_first_file,
    dtype_for_saving_mode,
    mark_generation_complete,
    read_embedding_manifest,
    validate_embedding_array,
    validate_manifest_records,
    write_embedding_manifest,
)
from utilities.PLM_Plugin_Utils import (
    read_plugin_metadata,
    validate_loaded_plugin,
)

# Script configuration
INPUT_FASTA = None
MODEL_NAME = None
SAVING_MODE = "float16" 
DEVICE_SELECTION = "auto"
                  
FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
EMBED_DIR = os.path.join("..", "Embeddings")

# --- JSON Settings Override ---
import json
import ast

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

# --- DYNAMIC PATH INFERENCE ---
FULL_INPUT_FASTA = os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR and INPUT_FASTA else ""

# Derive the base name for saving
SEQUENCE_SET = INPUT_FASTA.replace(".fasta", "") if INPUT_FASTA else "Unknown_Set"
OUTPUT_HDF5 = os.path.join(EMBED_DIR, f"{SEQUENCE_SET}_[{MODEL_NAME}]_embeddings.h5") if EMBED_DIR else ""

# Embedding model imports are deferred to dynamic plugin scripts under src/resources/pLM_models/

# %% =======================================
# OPTIMIZED EMBEDDING (GPU/CPU)
# ==========================================

def find_model_plugin(model_name):
    """
    Dynamically locates and loads the plugin script supporting the selected model_name.
    Uses AST to inspect supported models statically to avoid running code of non-matching plugins.
    """
    import ast
    import glob
    import importlib.util

    current_dir = os.path.dirname(os.path.abspath(__file__))
    plugin_dir = os.path.abspath(os.path.join(current_dir, "..", "resources", "pLM_models"))

    if not os.path.exists(plugin_dir):
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    for filepath in glob.glob(os.path.join(plugin_dir, "*.py")):
        if os.path.basename(filepath) == "__init__.py":
            continue
        try:
            supported_models, _ = read_plugin_metadata(filepath)
            
            if model_name in supported_models:
                module_name = os.path.splitext(os.path.basename(filepath))[0]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                validate_loaded_plugin(module, model_name)
                return module
        except Exception as error:
            raise ValueError(
                f"Invalid pLM plugin '{os.path.basename(filepath)}': {error}"
            ) from error
    return None


def _move_model_object(model_obj, device):
    """Move tensors/modules inside the supported plugin container shapes."""
    if isinstance(model_obj, torch.nn.Module):
        return model_obj.to(device)
    if torch.is_tensor(model_obj):
        return model_obj.to(device)
    if isinstance(model_obj, tuple):
        return tuple(_move_model_object(value, device) for value in model_obj)
    if isinstance(model_obj, list):
        return [_move_model_object(value, device) for value in model_obj]
    if isinstance(model_obj, dict):
        return {
            key: _move_model_object(value, device)
            for key, value in model_obj.items()
        }
    return model_obj


def _representative_sequences(pending):
    """Choose actual sequences nearest the 25th, 50th, and 90th percentiles."""
    ordered = sorted((len(sequence), sequence) for _, sequence in pending)
    if not ordered:
        return []
    selected = []
    for fraction in (0.25, 0.50, 0.90):
        index = round((len(ordered) - 1) * fraction)
        sequence = ordered[index][1]
        if sequence not in selected:
            selected.append(sequence)
    return selected


def _predicted_embedding_seconds(pending, samples, sample_times, move_seconds):
    predicted = float(move_seconds)
    for _, sequence in pending:
        nearest = min(samples, key=lambda sample: abs(len(sample) - len(sequence)))
        predicted += sample_times[nearest]
    return predicted


def _recover_model_on_cpu(plugin, model_name, model_obj):
    """Recover after a partial accelerator move, reloading only if necessary."""
    cpu = torch.device("cpu")
    try:
        return _move_model_object(model_obj, cpu)
    except Exception:
        return plugin.load_model(model_name, cpu)


def _benchmark_embedding_devices(
    plugin,
    model_name,
    model_obj,
    pending,
    target_dtype,
    feature_dimension,
    candidates,
):
    samples = _representative_sequences(pending)
    results = []
    print(
        "[Hardware] Benchmarking local embedding inference on "
        f"{len(candidates)} available device(s) using lengths "
        f"{[len(sequence) for sequence in samples]}..."
    )
    print("Device/backend                 Move (s)   Predicted job (s)   Status")

    for candidate in candidates:
        move_seconds = 0.0
        try:
            model_obj = _recover_model_on_cpu(plugin, model_name, model_obj)
            Hardware_Utils.release_device_cache(candidate)
            move_started = time.perf_counter()
            model_obj = _move_model_object(model_obj, candidate.device)
            Hardware_Utils.synchronize_device(candidate)
            move_seconds = time.perf_counter() - move_started

            warmup = samples[0][: min(64, len(samples[0]))]
            plugin.get_embedding(
                warmup, model_obj, candidate.device, target_dtype
            )

            sample_times = {}
            for sequence in samples:
                Hardware_Utils.synchronize_device(candidate)
                started = time.perf_counter()
                embedding = plugin.get_embedding(
                    sequence, model_obj, candidate.device, target_dtype
                )
                Hardware_Utils.synchronize_device(candidate)
                sample_times[sequence] = max(
                    time.perf_counter() - started, 1e-9
                )
                validate_embedding_array(
                    embedding,
                    sequence,
                    "float16" if target_dtype == np.dtype(np.float16) else "float32",
                    feature_dimension=feature_dimension,
                    require_finite=True,
                    header="hardware benchmark",
                )

            predicted = _predicted_embedding_seconds(
                pending, samples, sample_times, move_seconds
            )
            result = Hardware_Utils.BenchmarkResult(candidate, predicted)
            print(
                f"{candidate.display_name[:30]:30}  {move_seconds:>8.3f}   "
                f"{predicted:>17.3f}   ok"
            )
        except Exception as error:
            result = Hardware_Utils.BenchmarkResult(
                candidate,
                None,
                error=f"{type(error).__name__}: {error}",
            )
            print(
                f"{candidate.display_name[:30]:30}  {move_seconds:>8.3f}   "
                f"{'--':>17}   {result.error}"
            )
            model_obj = _recover_model_on_cpu(plugin, model_name, model_obj)
        results.append(result)

    ranked = Hardware_Utils.rank_benchmark_results(
        results, higher_is_better=False
    )
    if not ranked:
        failures = "; ".join(result.error or "unknown" for result in results)
        raise RuntimeError(f"No device completed the embedding benchmark: {failures}")
    winner = ranked[0]
    fastest = min(
        (result for result in results if result.succeeded),
        key=lambda result: float(result.value),
    )
    tie_applied = winner.candidate.spec != fastest.candidate.spec
    model_obj = _recover_model_on_cpu(plugin, model_name, model_obj)
    model_obj = _move_model_object(model_obj, winner.candidate.device)
    Hardware_Utils.synchronize_device(winner.candidate)
    print(
        f"[Hardware] Selected {winner.candidate.display_name} for embeddings; "
        f"3% tie preference {'applied' if tie_applied else 'not applied'}."
    )
    return model_obj, ranked

def generate_embeddings(
    input_fasta,
    output_hdf5,
    model_name,
    saving_mode,
    *,
    plugin_loader=find_model_plugin,
):
    """Generate or safely resume one metadata-first embedding database."""
    target_dtype = dtype_for_saving_mode(saving_mode)
    headers, sequences, _ = load_sanitized_fasta(input_fasta)
    validate_manifest_records(headers, sequences)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")
    print(f"Loaded {len(headers)} sequences from FASTA.")

    output_directory = os.path.dirname(output_hdf5)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    print(f"Opening {output_hdf5} for evaluation...")

    with h5py.File(output_hdf5, "a") as hf:
        is_new_file = len(hf) == 0 and len(hf.attrs) == 0
        if is_new_file:
            emb_group = create_metadata_first_file(
                hf,
                headers,
                sequences,
                model_name,
                saving_mode,
            )
            feature_dimension = None
        else:
            existing = read_embedding_manifest(
                hf,
                require_complete=False,
                validate_embeddings=True,
            )
            if existing.model_name != model_name:
                raise ValueError(
                    f"Cannot resume model '{model_name}' from a file created "
                    f"with model '{existing.model_name}'."
                )
            if existing.saving_mode != saving_mode:
                raise ValueError(
                    f"Cannot resume {saving_mode} generation from a "
                    f"{existing.saving_mode} file."
                )

            incoming_by_header = dict(zip(headers, sequences))
            existing_by_header = existing.sequence_by_header
            removed = sorted(set(existing.headers) - set(headers))
            if removed:
                raise ValueError(
                    f"Cannot resume because existing header '{removed[0]}' "
                    "was removed from the sanitized FASTA manifest."
                )
            changed = sorted(
                header
                for header in set(existing.headers) & set(headers)
                if existing_by_header[header] != incoming_by_header[header]
            )
            if changed:
                raise ValueError(
                    f"Cannot resume because the sanitized sequence for "
                    f"'{changed[0]}' changed."
                )

            if existing.headers != headers or existing.sequences != sequences:
                write_embedding_manifest(
                    hf,
                    headers,
                    sequences,
                    model_name,
                    saving_mode,
                    replace=True,
                )
            emb_group = hf["embeddings"]
            feature_dimension = existing.feature_dimension

        existing_keys = set(emb_group.keys())
        pending = [
            (header, sequence)
            for header, sequence in zip(headers, sequences)
            if header not in existing_keys
        ]

        if not pending:
            if not bool(hf.attrs["generation_complete"]):
                mark_generation_complete(hf)
            print(
                f"HDF5 database already complete ({len(headers)} embeddings). "
                "Skipping generation."
            )
            return 0

        hf.attrs["generation_complete"] = False
        hf.flush()
        if existing_keys:
            print(
                f"Resuming from interruption: {len(existing_keys)} found, "
                f"{len(pending)} remaining."
            )
        else:
            print(f"Starting fresh embedding generation for {len(pending)} sequences.")

        plugin = plugin_loader(model_name)
        if plugin is None:
            raise ValueError(
                f"Model '{model_name}' is not supported by any available "
                "plugin in 'pLM_models'."
            )
        execution_mode = validate_loaded_plugin(plugin, model_name)

        ranked_devices = []
        manual_candidate = None
        if execution_mode == "remote_api":
            print(
                f"[Hardware] Model '{model_name}' uses remote API inference; "
                "local CPU/GPU benchmarking is not applicable."
            )
            device = None
            model_obj = plugin.load_model(model_name, device)
        else:
            available_devices = Hardware_Utils.get_available_devices()
            manual_candidate = Hardware_Utils.resolve_device_selection(
                DEVICE_SELECTION,
                available_devices,
            )

        if execution_mode == "remote_api":
            pass
        elif manual_candidate is not None:
            device = manual_candidate.device
            print(f"[Hardware] Using manually selected {manual_candidate.display_name}.")
            model_obj = plugin.load_model(model_name, device)
            ranked_devices = [
                Hardware_Utils.BenchmarkResult(manual_candidate, 0.0)
            ]
        elif len(available_devices) == 1:
            only_candidate = available_devices[0]
            device = only_candidate.device
            print(f"[Hardware] Only {only_candidate.display_name} is available.")
            model_obj = plugin.load_model(model_name, device)
            ranked_devices = [
                Hardware_Utils.BenchmarkResult(only_candidate, 0.0)
            ]
        else:
            model_obj = plugin.load_model(model_name, torch.device("cpu"))
            model_obj, ranked_devices = _benchmark_embedding_devices(
                plugin,
                model_name,
                model_obj,
                pending,
                target_dtype,
                feature_dimension,
                available_devices,
            )
            device = ranked_devices[0].candidate.device

        active_rank = 0
        for header, sequence in tqdm(pending, desc="Embedding"):
            while True:
                try:
                    embedding = plugin.get_embedding(
                        sequence,
                        model_obj,
                        device,
                        target_dtype,
                    )
                    break
                except (RuntimeError, NotImplementedError, MemoryError) as error:
                    if execution_mode != "local":
                        raise
                    if manual_candidate is not None:
                        raise RuntimeError(
                            f"Embedding '{header}' failed on manually selected "
                            f"device '{manual_candidate.spec}': {error}"
                        ) from error
                    active_rank += 1
                    if active_rank >= len(ranked_devices):
                        raise RuntimeError(
                            f"Every benchmarked device failed while embedding '{header}'."
                        ) from error
                    fallback = ranked_devices[active_rank].candidate
                    print(
                        f"[Hardware] {type(error).__name__} on {device}; "
                        f"retrying '{header}' on {fallback.display_name}."
                    )
                    model_obj = _recover_model_on_cpu(plugin, model_name, model_obj)
                    model_obj = _move_model_object(model_obj, fallback.device)
                    device = fallback.device
            feature_dimension = validate_embedding_array(
                embedding,
                sequence,
                saving_mode,
                feature_dimension=feature_dimension,
                require_finite=True,
                header=header,
            )
            emb_group.create_dataset(header, data=embedding)
            hf.flush()

        mark_generation_complete(hf)
        return len(pending)


# %% =======================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Step 1: Embedding Generation ---")
    if SAVING_MODE == "float16":
        print("--> Mode: Saving as float16 (Compact)")
    elif SAVING_MODE == "float32":
        print("--> Mode: Saving as float32 (High Precision)")

    generated_count = generate_embeddings(
        FULL_INPUT_FASTA,
        OUTPUT_HDF5,
        MODEL_NAME,
        SAVING_MODE,
    )
    if generated_count:
        print(f"\nDone! All embeddings generated and saved to {OUTPUT_HDF5}")
