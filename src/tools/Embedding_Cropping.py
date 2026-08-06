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

"""Slice contextual cropped embeddings using sequences stored in the source HDF5."""

import ast
import json
import os
try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import h5py
from tqdm import tqdm

from utilities.FASTA_Sanitization import load_sanitized_fasta
from utilities.Embedding_HDF5 import (
    create_metadata_first_file,
    mark_generation_complete,
    read_embedding_manifest,
    validate_embedding_array,
    validate_manifest_records,
)


INPUT_EMBED = None
CROPPED_FASTA = None

FASTA_DIR = os.path.join("..", "Input_Files", "Sequence_Sets")
EMBED_DIR = os.path.join("..", "Embeddings")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as settings_handle:
            all_settings = json.load(settings_handle)
        for key, value in all_settings.get("DIRECTORIES", {}).items():
            if key in globals() and value is not None and str(value).strip():
                if not os.path.isabs(str(value)):
                    value = os.path.normpath(os.path.join(PROJECT_ROOT, str(value)))
                globals()[key] = value
        for key, value in all_settings.get(os.path.basename(__file__), {}).items():
            if key in globals() and value is not None and str(value).strip():
                original = globals()[key]
                if isinstance(original, list) and isinstance(value, str):
                    try:
                        value = ast.literal_eval(value)
                    except (SyntaxError, ValueError):
                        pass
                globals()[key] = value
    except Exception as exc:
        print(f"Failed to load user settings: {exc}")

FULL_INPUT_EMBED = (
    os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR and INPUT_EMBED else ""
)
CROPPED_FASTA_PATH = (
    os.path.join(FASTA_DIR, CROPPED_FASTA) if FASTA_DIR and CROPPED_FASTA else ""
)


def resolve_crops(source_manifest, cropped_headers, cropped_sequences):
    """Resolve sanitized crops against the source file's stored sequences."""
    source_sequences = source_manifest.sequence_by_header
    missing_from_source = []
    substring_not_found = []
    ambiguous_resolved = []
    resolved = []

    for header, cropped_sequence in zip(cropped_headers, cropped_sequences):
        full_sequence = source_sequences.get(header)
        if full_sequence is None:
            missing_from_source.append(header)
            continue
        offset = full_sequence.find(cropped_sequence)
        if offset < 0:
            substring_not_found.append(header)
            continue
        if full_sequence.count(cropped_sequence) > 1:
            ambiguous_resolved.append(header)
        resolved.append((header, cropped_sequence, offset))

    return resolved, missing_from_source, substring_not_found, ambiguous_resolved


def crop_embeddings(input_hdf5, cropped_fasta, output_hdf5=None):
    """Create cropped embeddings without requiring a separate full FASTA."""
    if not os.path.exists(input_hdf5):
        raise FileNotFoundError(f"Input embedding database not found: {input_hdf5}")

    cropped_headers, cropped_sequences, _ = load_sanitized_fasta(cropped_fasta)
    validate_manifest_records(cropped_headers, cropped_sequences)
    with h5py.File(input_hdf5, "r") as hf_in:
        source_manifest = read_embedding_manifest(hf_in, require_complete=True)
        (
            resolved,
            missing_from_source,
            substring_not_found,
            ambiguous_resolved,
        ) = resolve_crops(source_manifest, cropped_headers, cropped_sequences)

    if output_hdf5 is None:
        cropped_base = os.path.splitext(os.path.basename(cropped_fasta))[0]
        output_hdf5 = os.path.join(
            EMBED_DIR,
            f"{cropped_base}_[{source_manifest.model_name}]_embeddings.h5",
        )
    if os.path.abspath(output_hdf5) == os.path.abspath(input_hdf5):
        raise ValueError("Cropping output must not overwrite its source embedding file.")

    resolved_headers = [record[0] for record in resolved]
    resolved_sequences = [record[1] for record in resolved]
    output_directory = os.path.dirname(output_hdf5)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    feature_dimension = None
    with h5py.File(input_hdf5, "r") as hf_in, h5py.File(output_hdf5, "w") as hf_out:
        emb_group_out = create_metadata_first_file(
            hf_out,
            resolved_headers,
            resolved_sequences,
            source_manifest.model_name,
            source_manifest.saving_mode,
        )
        for header, cropped_sequence, offset in tqdm(resolved, desc="Cropping"):
            cropped_embedding = hf_in["embeddings"][header][
                offset : offset + len(cropped_sequence)
            ]
            feature_dimension = validate_embedding_array(
                cropped_embedding,
                cropped_sequence,
                source_manifest.saving_mode,
                feature_dimension=feature_dimension,
                header=header,
            )
            emb_group_out.create_dataset(header, data=cropped_embedding)
            hf_out.flush()
        mark_generation_complete(hf_out)

    return {
        "output_hdf5": output_hdf5,
        "resolved_headers": resolved_headers,
        "missing_from_source": missing_from_source,
        "substring_not_found": substring_not_found,
        "ambiguous_resolved": ambiguous_resolved,
    }


def _report(label, items):
    if not items:
        return
    print(f"\n⚠️  {len(items)} header(s) skipped ({label}):")
    for header in items[:10]:
        print(f"    - {header}")
    if len(items) > 10:
        print(f"    ... and {len(items) - 10} more.")


if __name__ == "__main__":
    print("--- Embedding Cropping ---")
    result = crop_embeddings(FULL_INPUT_EMBED, CROPPED_FASTA_PATH)
    print(f"\n✅ Done! Cropped embeddings saved to {result['output_hdf5']}")
    print(f"  > Resolved: {len(result['resolved_headers'])}")
    if result["ambiguous_resolved"]:
        print(
            f"\n⚠️  {len(result['ambiguous_resolved'])} header(s) had an "
            "ambiguous repeated crop match; the first occurrence was used:"
        )
        for header in result["ambiguous_resolved"][:10]:
            print(f"    - {header}")
    _report("missing from source embedding metadata", result["missing_from_source"])
    _report("crop substring not found in stored full sequence", result["substring_not_found"])
