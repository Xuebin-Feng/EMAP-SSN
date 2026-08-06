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

"""Shared metadata-first HDF5 helpers for protein embeddings."""

from dataclasses import dataclass

import h5py
import numpy as np


SAVING_MODE_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}
REQUIRED_ATTRIBUTES = (
    "model_name",
    "saving_mode",
    "num_sequences",
    "generation_complete",
)
REQUIRED_OBJECTS = ("headers", "sequences", "embeddings")
SANITIZED_RESIDUE_CODES = frozenset("ACDEFGHIKLMNPQRSTVWYBZJXUO")


@dataclass(frozen=True)
class EmbeddingManifest:
    headers: list
    sequences: list
    model_name: str
    saving_mode: str
    generation_complete: bool
    feature_dimension: int | None

    @property
    def sequence_by_header(self):
        return dict(zip(self.headers, self.sequences))


def _decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def decode_string_dataset(dataset):
    """Decode one HDF5 string dataset into ordinary Python strings."""
    return [_decode_text(value) for value in dataset[:]]


def dtype_for_saving_mode(saving_mode):
    """Return the required NumPy dtype for a persisted saving mode."""
    try:
        return SAVING_MODE_DTYPES[saving_mode]
    except KeyError as exc:
        raise ValueError(
            "saving_mode must be exactly 'float16' or 'float32'."
        ) from exc


def saving_mode_for_dtype(dtype):
    """Return the saving-mode label for a supported floating-point dtype."""
    normalized = np.dtype(dtype)
    for saving_mode, expected_dtype in SAVING_MODE_DTYPES.items():
        if normalized == expected_dtype:
            return saving_mode
    raise ValueError(
        f"Unsupported embedding dtype '{normalized}'; expected float16 or float32."
    )


def validate_manifest_records(headers, sequences):
    """Validate the one-to-one sanitized header/sequence manifest."""
    if len(headers) != len(sequences):
        raise ValueError("Embedding headers and sequences must have equal lengths.")

    seen = set()
    for index, (header, sequence) in enumerate(zip(headers, sequences)):
        if not isinstance(header, str) or not header:
            raise ValueError(f"Embedding header at index {index} is empty or invalid.")
        if header in {".", ".."}:
            raise ValueError(f"Embedding header '{header}' is not a safe HDF5 key.")
        if "/" in header or "\\" in header or "\x00" in header:
            raise ValueError(
                f"Embedding header '{header}' contains a path or null character."
            )
        if header in seen:
            raise ValueError(f"Duplicate embedding header '{header}'.")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(
                f"Sanitized sequence for header '{header}' is empty or invalid."
            )
        invalid_codes = sorted(set(sequence) - SANITIZED_RESIDUE_CODES)
        if invalid_codes:
            raise ValueError(
                f"Sequence for header '{header}' is not sanitized; unsupported "
                f"code '{invalid_codes[0]}' remains."
            )
        seen.add(header)


def validate_embedding_array(
    embedding,
    sequence,
    saving_mode,
    feature_dimension=None,
    require_finite=False,
    header=None,
):
    """Validate one generated array or HDF5 embedding dataset."""
    label = f" for '{header}'" if header is not None else ""
    if getattr(embedding, "ndim", None) != 2:
        raise ValueError(f"Embedding{label} must be two-dimensional.")
    if embedding.shape[0] != len(sequence):
        raise ValueError(
            f"Embedding{label} has {embedding.shape[0]} rows but its stored "
            f"sequence has {len(sequence)} residues."
        )
    if np.dtype(embedding.dtype) != dtype_for_saving_mode(saving_mode):
        raise ValueError(
            f"Embedding{label} has dtype {embedding.dtype}, not {saving_mode}."
        )
    current_dimension = int(embedding.shape[1])
    if current_dimension <= 0:
        raise ValueError(f"Embedding{label} has an empty feature dimension.")
    if feature_dimension is not None and current_dimension != feature_dimension:
        raise ValueError(
            f"Embedding{label} has feature dimension {current_dimension}; "
            f"expected {feature_dimension}."
        )
    if require_finite and not np.isfinite(np.asarray(embedding)).all():
        raise ValueError(f"Embedding{label} contains NaN or infinite values.")
    return current_dimension


def _validate_embedding_group(
    hf,
    headers,
    sequences,
    saving_mode,
    require_all,
):
    emb_group = hf["embeddings"]
    if not isinstance(emb_group, h5py.Group):
        raise ValueError("'/embeddings' must be an HDF5 group.")

    expected = set(headers)
    actual = set(emb_group.keys())
    orphaned = sorted(actual - expected)
    if orphaned:
        raise ValueError(
            f"Embedding database contains unexpected dataset '{orphaned[0]}'."
        )
    missing = sorted(expected - actual)
    if require_all and missing:
        raise ValueError(
            f"Embedding database is missing dataset '{missing[0]}'."
        )

    sequence_by_header = dict(zip(headers, sequences))
    feature_dimension = None
    for header in headers:
        if header not in emb_group:
            continue
        dataset = emb_group[header]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"Embedding object '{header}' is not a dataset.")
        feature_dimension = validate_embedding_array(
            dataset,
            sequence_by_header[header],
            saving_mode,
            feature_dimension=feature_dimension,
            header=header,
        )
    return feature_dimension


def read_embedding_manifest(
    hf,
    *,
    require_complete=True,
    validate_embeddings=True,
):
    """Read and validate the required embedding-file interface."""
    missing_attrs = [name for name in REQUIRED_ATTRIBUTES if name not in hf.attrs]
    if missing_attrs:
        raise ValueError(
            "Embedding file is missing required attribute(s): "
            + ", ".join(missing_attrs)
            + ". Legacy embedding files are unsupported."
        )
    missing_objects = [name for name in REQUIRED_OBJECTS if name not in hf]
    if missing_objects:
        raise ValueError(
            "Embedding file is missing required object(s): "
            + ", ".join(f"/{name}" for name in missing_objects)
            + ". Legacy embedding files are unsupported."
        )
    if not isinstance(hf["headers"], h5py.Dataset) or not isinstance(
        hf["sequences"], h5py.Dataset
    ):
        raise ValueError("'/headers' and '/sequences' must be HDF5 datasets.")
    if hf["headers"].ndim != 1 or hf["sequences"].ndim != 1:
        raise ValueError("'/headers' and '/sequences' must be one-dimensional.")

    headers = decode_string_dataset(hf["headers"])
    sequences = decode_string_dataset(hf["sequences"])
    validate_manifest_records(headers, sequences)

    model_name = _decode_text(hf.attrs["model_name"])
    if not model_name:
        raise ValueError("Embedding attribute 'model_name' cannot be empty.")
    saving_mode = _decode_text(hf.attrs["saving_mode"])
    dtype_for_saving_mode(saving_mode)
    try:
        num_sequences = int(hf.attrs["num_sequences"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding attribute 'num_sequences' is invalid.") from exc
    if num_sequences != len(headers):
        raise ValueError(
            f"num_sequences is {num_sequences}, but /headers contains "
            f"{len(headers)} records."
        )

    complete_value = hf.attrs["generation_complete"]
    if not isinstance(complete_value, (bool, np.bool_)):
        raise ValueError("Embedding attribute 'generation_complete' must be boolean.")
    generation_complete = bool(complete_value)
    if require_complete and not generation_complete:
        raise ValueError("Embedding generation is incomplete.")

    feature_dimension = None
    if validate_embeddings:
        feature_dimension = _validate_embedding_group(
            hf,
            headers,
            sequences,
            saving_mode,
            require_all=require_complete or generation_complete,
        )

    return EmbeddingManifest(
        headers=headers,
        sequences=sequences,
        model_name=model_name,
        saving_mode=saving_mode,
        generation_complete=generation_complete,
        feature_dimension=feature_dimension,
    )


def write_embedding_manifest(
    hf,
    headers,
    sequences,
    model_name,
    saving_mode,
    *,
    replace=False,
):
    """Write and flush metadata before any embedding datasets are created."""
    headers = list(headers)
    sequences = list(sequences)
    validate_manifest_records(headers, sequences)
    dtype_for_saving_mode(saving_mode)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a non-empty string.")

    existing = [name for name in ("headers", "sequences") if name in hf]
    if existing and not replace:
        raise ValueError("Embedding manifest already exists.")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    next_headers = "_headers_next"
    next_sequences = "_sequences_next"
    for temporary_name in (next_headers, next_sequences):
        if temporary_name in hf:
            del hf[temporary_name]

    hf.create_dataset(
        next_headers,
        data=np.asarray(headers, dtype=object),
        dtype=string_dtype,
    )
    hf.create_dataset(
        next_sequences,
        data=np.asarray(sequences, dtype=object),
        dtype=string_dtype,
    )
    hf.attrs["model_name"] = model_name
    hf.attrs["saving_mode"] = saving_mode
    hf.attrs["num_sequences"] = len(headers)
    hf.attrs["generation_complete"] = False
    hf.flush()

    for final_name in ("headers", "sequences"):
        if final_name in hf:
            del hf[final_name]
    hf.move(next_headers, "headers")
    hf.move(next_sequences, "sequences")
    hf.flush()


def create_metadata_first_file(hf, headers, sequences, model_name, saving_mode):
    """Initialize a new output file in the required metadata-first order."""
    if len(hf) or len(hf.attrs):
        raise ValueError("Output HDF5 file must be empty before initialization.")
    write_embedding_manifest(
        hf,
        headers,
        sequences,
        model_name,
        saving_mode,
    )
    emb_group = hf.create_group("embeddings")
    hf.flush()
    return emb_group


def mark_generation_complete(hf):
    """Validate the finished database, mark it complete, and flush it."""
    manifest = read_embedding_manifest(
        hf,
        require_complete=False,
        validate_embeddings=True,
    )
    expected = set(manifest.headers)
    actual = set(hf["embeddings"].keys())
    if expected != actual:
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"Embedding database is missing dataset '{missing[0]}'.")
        orphaned = sorted(actual - expected)
        raise ValueError(
            f"Embedding database contains unexpected dataset '{orphaned[0]}'."
        )
    hf.attrs["generation_complete"] = True
    hf.flush()
    return read_embedding_manifest(hf, require_complete=True)
