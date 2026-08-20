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

"""Extract a sanitized subset from a metadata-first embedding database."""

import ast
import json
import os

try:
    from tools import _bootstrap
except ModuleNotFoundError:
    import _bootstrap

import h5py
from tqdm import tqdm

from utilities.FASTA_Sanitization import load_sanitized_fasta, sanitize_header
from utilities.Embedding_HDF5 import (
    create_metadata_first_file,
    mark_generation_complete,
    read_embedding_manifest,
    validate_embedding_array,
    validate_manifest_records,
)


INPUT_EMBED = None
INPUT_FASTA = None

from utilities.Tool_Directories import project_directory_defaults
from utilities.Tool_Settings import inherited_settings_path, load_tool_settings

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_DIRECTORIES = project_directory_defaults(PROJECT_ROOT)
EMBED_DIR = _DEFAULT_DIRECTORIES["EMBED_DIR"]
FASTA_DIR = _DEFAULT_DIRECTORIES["FASTA_DIR"]
SETTINGS_FILE = inherited_settings_path(__file__) or os.path.join(PROJECT_ROOT, "Input_Files", "tools_settings.json")

if __name__ != "__main__" and os.path.exists(SETTINGS_FILE):
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
FULL_INPUT_FASTA = (
    os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR and INPUT_FASTA else ""
)


def load_target_records(file_path):
    """Load a sanitized FASTA selection or a sanitized header-only list."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension in {".fasta", ".fa", ".fna"}:
        headers, sequences, _ = load_sanitized_fasta(file_path)
        validate_manifest_records(headers, sequences)
        return headers, dict(zip(headers, sequences))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target selection file not found: {file_path}")
    headers = []
    with open(file_path, "r", encoding="utf-8-sig") as selection_handle:
        for line in selection_handle:
            if line.strip():
                headers.append(sanitize_header(line.strip())[0])
    if len(headers) != len(set(headers)):
        raise ValueError("Target header list contains duplicates after sanitization.")
    validate_manifest_records(headers, ["X"] * len(headers))
    return headers, None


def extract_subset(input_hdf5, selection_file, output_hdf5=None):
    """Copy selected embeddings and their stored sanitized sequences."""
    if not os.path.exists(input_hdf5):
        raise FileNotFoundError(f"Source HDF5 not found: {input_hdf5}")
    target_headers, supplied_sequences = load_target_records(selection_file)

    with h5py.File(input_hdf5, "r") as hf_in:
        source_manifest = read_embedding_manifest(hf_in, require_complete=True)
    source_sequences = source_manifest.sequence_by_header

    found_headers = []
    found_sequences = []
    missing_headers = []
    for header in target_headers:
        if header not in source_sequences:
            missing_headers.append(header)
            continue
        stored_sequence = source_sequences[header]
        if supplied_sequences is not None and supplied_sequences[header] != stored_sequence:
            raise ValueError(
                f"Selection FASTA sequence for '{header}' does not match the "
                "sanitized sequence stored in the source embedding file."
            )
        found_headers.append(header)
        found_sequences.append(stored_sequence)

    validate_manifest_records(found_headers, found_sequences)
    if output_hdf5 is None:
        selection_base = os.path.splitext(os.path.basename(selection_file))[0]
        output_hdf5 = os.path.join(
            EMBED_DIR,
            f"{selection_base}_[{source_manifest.model_name}]_embeddings.h5",
        )
    if os.path.abspath(output_hdf5) == os.path.abspath(input_hdf5):
        raise ValueError("Extraction output must not overwrite its source file.")
    output_directory = os.path.dirname(output_hdf5)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    feature_dimension = None
    with h5py.File(input_hdf5, "r") as hf_in, h5py.File(output_hdf5, "w") as hf_out:
        emb_group_out = create_metadata_first_file(
            hf_out,
            found_headers,
            found_sequences,
            source_manifest.model_name,
            source_manifest.saving_mode,
        )
        for header, sequence in tqdm(
            zip(found_headers, found_sequences),
            total=len(found_headers),
            desc="Extracting",
        ):
            embedding = hf_in["embeddings"][header][:]
            feature_dimension = validate_embedding_array(
                embedding,
                sequence,
                source_manifest.saving_mode,
                feature_dimension=feature_dimension,
                header=header,
            )
            emb_group_out.create_dataset(header, data=embedding)
            hf_out.flush()
        mark_generation_complete(hf_out)

    return output_hdf5, found_headers, missing_headers


def main(argv=None):
    global FULL_INPUT_EMBED, FULL_INPUT_FASTA
    load_tool_settings(globals(), __file__, PROJECT_ROOT, argv)
    FULL_INPUT_EMBED = (
        os.path.join(EMBED_DIR, INPUT_EMBED) if EMBED_DIR and INPUT_EMBED else ""
    )
    FULL_INPUT_FASTA = (
        os.path.join(FASTA_DIR, INPUT_FASTA) if FASTA_DIR and INPUT_FASTA else ""
    )
    print("--- HDF5 Embedding Extractor ---")
    output_path, found, missing = extract_subset(
        FULL_INPUT_EMBED,
        FULL_INPUT_FASTA,
    )
    print("\n✅ Extraction Complete!")
    print(f"  > Saved to: {output_path}")
    print(f"  > Extracted: {len(found)}")
    print(f"  > Missing:   {len(missing)}")
    if missing:
        print("\n⚠️  The following headers were not found in the source:")
        for header in missing[:10]:
            print(f"    - {header}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
