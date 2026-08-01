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
from tqdm import tqdm
import numpy as np
import torch
import h5py
import Hardware_Utils

try:
    from FASTA_Sanitization import load_sanitized_fasta
    from Embedding_HDF5 import (
        create_metadata_first_file,
        dtype_for_saving_mode,
        mark_generation_complete,
        read_embedding_manifest,
        validate_embedding_array,
        validate_manifest_records,
        write_embedding_manifest,
    )
except ModuleNotFoundError:
    from src.utilities.FASTA_Sanitization import load_sanitized_fasta
    from src.utilities.Embedding_HDF5 import (
        create_metadata_first_file,
        dtype_for_saving_mode,
        mark_generation_complete,
        read_embedding_manifest,
        validate_embedding_array,
        validate_manifest_records,
        write_embedding_manifest,
    )

# Script configuration
INPUT_FASTA = None
MODEL_NAME = None
SAVING_MODE = "float16" 
                  
FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
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
            with open(filepath, "r", encoding="utf-8") as f:
                node = ast.parse(f.read(), filename=filepath)
            
            supported_models = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "SUPPORTED_MODELS":
                            supported_models = ast.literal_eval(item.value)
                            break
            
            if model_name in supported_models:
                module_name = os.path.splitext(os.path.basename(filepath))[0]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            print(f"Warning: Failed to parse/load plugin {filepath}: {e}")
    return None

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
        device = Hardware_Utils.get_optimal_device()
        model_obj = plugin.load_model(model_name, device)

        for header, sequence in tqdm(pending, desc="Embedding"):
            embedding = plugin.get_embedding(
                sequence,
                model_obj,
                device,
                target_dtype,
            )
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
