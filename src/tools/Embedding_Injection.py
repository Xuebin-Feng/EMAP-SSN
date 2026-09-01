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
File: Embedding_Injection.py
===================================
Description:
Once a network has been run, it is often useful to add a few more sequences (like a newly discovered wild-type or a 
specific reference structure) into the existing network without having to recalculate embeddings for all 50,000 original sequences.
This script takes an existing embedding HDF5 database and a NEW FASTA file containing both the old and new sequences. 
It identifies the newly added sequences, dynamically boots up the required language model, computes embeddings ONLY for the 
new additions, and synthesizes a new combined database.

Input:
- An existing HDF5 file containing pre-calculated embeddings (`OLD_HDF5`).
- A new FASTA file containing all the old sequences + any new ones you want to add (`NEW_FASTA`).

Output:
- A new, complete HDF5 file containing all embeddings, properly ordered to match the new FASTA file (`OUTPUT_HDF5`).

Settings:
- OLD_HDF5: Path to your original workspace embedding database.
- NEW_FASTA: Path to the FASTA file that contains your original sequences plus your manually added additions.
- OUTPUT_HDF5: The filename to save the newly merged database as so it does not overwrite your original.

Algorithm:
1. Loads the metadata from the old database to identify the pLM plugin originally used.
2. Checks the datatypes of the arrays to ensure precision matching (FP16 vs FP32).
3. Parses and sanitizes the new FASTA file, then identifies genuinely new headers.
4. Validates that ALL original sequences are still present in the new FASTA (throwing an error if the user accidentally deleted some).
5. If new sequences are detected, it initializes the matching local or remote pLM plugin.
6. Streams the final output database sequentially: if a sequence is old, it blitz-copies the binary array from the old file. If it is new, it routes it through the GPU for inference and streams the result directly to disk.
"""
# %% Import Necessary Libraries
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import numpy as np
import h5py
from tqdm import tqdm
from utilities import Hardware_Utils
from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Embedding_HDF5 import (
    create_metadata_first_file,
    dtype_for_saving_mode,
    mark_generation_complete,
    read_embedding_manifest,
    validate_embedding_array,
    validate_manifest_records,
)

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_EMBED = None
INPUT_FASTA = None

from utilities.Tool_Directories import project_directory_defaults
from utilities.Tool_Settings import inherited_settings_path, load_tool_settings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]

# --- JSON Settings Override ---
import json
import ast
import os

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

FULL_INPUT_EMBED = None
FULL_INPUT_FASTA = None


def _resolve_selected_path(value, directory, description):
    if value is None or not str(value).strip():
        raise ValueError(f"No {description} has been selected.")

    selected_path = os.fspath(value)
    if os.path.isabs(selected_path):
        return os.path.normpath(selected_path)
    return os.path.normpath(os.path.join(directory, selected_path))


def configure_runtime_paths():
    """Resolve GUI-selected inputs immediately before injection starts."""
    global FULL_INPUT_EMBED, FULL_INPUT_FASTA

    FULL_INPUT_EMBED = _resolve_selected_path(
        INPUT_EMBED,
        EMBED_DIR,
        "existing embeddings file",
    )
    FULL_INPUT_FASTA = _resolve_selected_path(
        INPUT_FASTA,
        FASTA_DIR,
        "replacement FASTA file",
    )

# %% Helper Functions
def find_model_plugin(model_name):
    """
    Locates the pLM plugin that declares support for model_name.

    SUPPORTED_MODELS is inspected with AST so unrelated plugins are not imported.
    """
    import ast
    import glob
    import importlib.util

    plugin_dir = os.path.abspath(
        os.path.join(PROJECT_ROOT, "src", "resources", "pLM_models")
    )

    if not os.path.exists(plugin_dir):
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")

    for filepath in glob.glob(os.path.join(plugin_dir, "*.py")):
        if os.path.basename(filepath) == "__init__.py":
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                node = ast.parse(handle.read(), filename=filepath)

            supported_models = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "SUPPORTED_MODELS"
                        ):
                            supported_models = ast.literal_eval(item.value)
                            break

            if model_name in supported_models:
                module_name = os.path.splitext(os.path.basename(filepath))[0]
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception as exc:
            print(f"Warning: Failed to parse/load plugin {filepath}: {exc}")

    return None


def load_model(model_name):
    """
    Loads the model plugin identified by the HDF5 model_name metadata.
    """
    plugin = find_model_plugin(model_name)
    if plugin is None:
        raise ValueError(
            f"Model '{model_name}' is not supported by any available plugin "
            "in 'pLM_models'."
        )

    missing_functions = [
        function_name
        for function_name in ("load_model", "get_embedding")
        if not callable(getattr(plugin, function_name, None))
    ]
    if missing_functions:
        raise AttributeError(
            f"Model plugin '{plugin.__name__}' is missing required function(s): "
            f"{', '.join(missing_functions)}"
        )

    device = Hardware_Utils.get_optimal_device()
    model_obj = plugin.load_model(model_name, device)
    return model_obj, device, plugin


def get_embedding(seq, model_obj, device, model_plugin, target_dtype):
    """
    Generates an embedding through the selected pLM plugin.
    """
    return model_plugin.get_embedding(seq, model_obj, device, target_dtype)

def inject_embeddings(input_hdf5, input_fasta, output_hdf5=None):
    """Copy unchanged embeddings and generate only sanitized additions."""
    if not os.path.exists(input_hdf5):
        raise FileNotFoundError(f"Original embedding file not found: {input_hdf5}")

    new_headers, new_sequences, _ = load_sanitized_fasta(input_fasta)
    validate_manifest_records(new_headers, new_sequences)
    with h5py.File(input_hdf5, "r") as hf_in:
        source_manifest = read_embedding_manifest(hf_in, require_complete=True)

    old_sequences = source_manifest.sequence_by_header
    new_by_header = dict(zip(new_headers, new_sequences))
    missing_from_new = sorted(set(source_manifest.headers) - set(new_headers))
    if missing_from_new:
        raise ValueError(
            f"Cannot inject because original header '{missing_from_new[0]}' "
            "is absent from the sanitized replacement FASTA."
        )
    changed = sorted(
        header
        for header in source_manifest.headers
        if old_sequences[header] != new_by_header[header]
    )
    if changed:
        raise ValueError(
            f"Cannot reuse the embedding for '{changed[0]}' because its "
            "sanitized sequence changed."
        )

    additions = [header for header in new_headers if header not in old_sequences]
    if output_hdf5 is None:
        fasta_base = os.path.splitext(os.path.basename(input_fasta))[0]
        output_hdf5 = os.path.join(
            EMBED_DIR,
            f"{fasta_base}_[{source_manifest.model_name}]_embeddings.h5",
        )
    if os.path.abspath(output_hdf5) == os.path.abspath(input_hdf5):
        raise ValueError("Injection output must not overwrite its source file.")
    output_directory = os.path.dirname(output_hdf5)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    model_obj = device = model_plugin = None
    if additions:
        model_obj, device, model_plugin = load_model(source_manifest.model_name)
    target_dtype = dtype_for_saving_mode(source_manifest.saving_mode)

    copied_count = 0
    generated_count = 0
    feature_dimension = source_manifest.feature_dimension
    with h5py.File(input_hdf5, "r") as hf_in, h5py.File(output_hdf5, "w") as hf_out:
        emb_group_out = create_metadata_first_file(
            hf_out,
            new_headers,
            new_sequences,
            source_manifest.model_name,
            source_manifest.saving_mode,
        )
        for header, sequence in tqdm(
            zip(new_headers, new_sequences),
            total=len(new_headers),
            desc="Writing",
        ):
            is_new = header not in old_sequences
            if is_new:
                embedding = get_embedding(
                    sequence,
                    model_obj,
                    device,
                    model_plugin,
                    target_dtype,
                )
            else:
                embedding = hf_in["embeddings"][header][:]
            feature_dimension = validate_embedding_array(
                embedding,
                sequence,
                source_manifest.saving_mode,
                feature_dimension=feature_dimension,
                require_finite=is_new,
                header=header,
            )
            emb_group_out.create_dataset(header, data=embedding)
            hf_out.flush()
            if is_new:
                generated_count += 1
            else:
                copied_count += 1
        mark_generation_complete(hf_out)

    return output_hdf5, generated_count, copied_count


# %% Main Execution
def main(argv=None):
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    print("--- Embedding Injection ---")
    try:
        configure_runtime_paths()
    except ValueError as error:
        raise SystemExit(f"❌ Error: {error}") from error

    output_path, generated_count, copied_count = inject_embeddings(
        FULL_INPUT_EMBED,
        FULL_INPUT_FASTA,
    )
    print(
        f"\n✅ Done! Injected {generated_count} new embeddings and copied "
        f"{copied_count} existing ones."
    )
    print(f"Saved directly to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
