# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""HDF5 storage helpers for protein embedding files."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Protein Embedding HDF5 Specifications and Utilities
# ---------------------------------------------------------------------------

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
# Where a reader's refusal of a file in another format sends the user.
FOREIGN_FILE_ADVICE = (
    "EMAP-SSN reads only its own embedding format. Convert per-residue "
    "embeddings from elsewhere with Embedding Import (Tools window, Sequence & "
    "Embedding Preparation tab); its help describes the format."
)
# A model label names the model in an embedding file and in the names of the
# files made from it, such as <set>_[<label>]_embeddings.h5.
MODEL_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


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


def open_embeddings_shared(path):
    """Open an embeddings file read-only the way the alignment code opens it: latest format, SWMR read.

    HDF5 refuses a second open of a file in the same process when the SWMR flag differs from the open
    one. The alignment path (the tile store, the accelerator pipelines, Align_Similarity_Matrix's
    align_pairs_in_memory) opens embeddings this way, so a program that keeps an embeddings file open
    while it calls that path (Embedding_MSA does, during the calibration alignment) opens it with this
    function too, whichever embeddings file it is.
    """
    return h5py.File(path, "r", libver="latest", swmr=True)


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
            + ". "
            + FOREIGN_FILE_ADVICE
        )
    missing_objects = [name for name in REQUIRED_OBJECTS if name not in hf]
    if missing_objects:
        raise ValueError(
            "Embedding file is missing required object(s): "
            + ", ".join(f"/{name}" for name in missing_objects)
            + ". "
            + FOREIGN_FILE_ADVICE
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


def validate_model_label(label):
    """Return ``label`` if it can name a model in a file and in file names."""
    if not isinstance(label, str) or not MODEL_LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            f"Model label {label!r} cannot be used: give 1 to 64 lower-case "
            "letters, digits, '_' or '-', starting with a letter or digit."
        )
    return label


def embedding_shape_problem(shape, sequence_length):
    """Say why an array of ``shape`` cannot be one sequence's embedding, or None.

    Names the usual causes: a per-protein vector where one vector per residue
    is needed, and the rows of a model's start or end tokens left in.
    """
    shape = tuple(int(size) for size in shape)
    if len(shape) == 1:
        return (
            f"is a single vector of {shape[0]} values, a per-protein embedding; "
            "alignment needs one vector per residue (a residues-by-features matrix)"
        )
    if len(shape) != 2:
        return f"has {len(shape)} dimensions; expected residues by features"
    rows, width = shape
    if width == 0:
        return "has no features"
    if rows == sequence_length:
        return None
    hint = ""
    if rows == sequence_length + 2:
        hint = ": the model's start and end tokens were probably left in"
    elif rows == sequence_length + 1:
        hint = ": a start or end token was probably left in"
    return f"has {rows} rows for a sequence of {sequence_length} residues{hint}"


class EmbeddingWriter:
    """Write an EMAP-SSN embedding file from any model, one sequence at a time.

    For a model EMAP-SSN has no plugin for::

        with EmbeddingWriter("Embeddings/my_set_[my_model]_embeddings.h5",
                             "my_set.fasta", "my_model", "float16") as writer:
            for header, sequence in writer.pending:
                writer.add(header, my_model(sequence))  # residues x features

    ``fasta`` is a FASTA path, which is sanitized as every EMAP-SSN tool
    sanitizes it, or a ``(headers, sequences)`` pair already sanitized.
    ``pending`` lists the sanitized (header, sequence) pairs that still lack an
    array. Embed that sanitized sequence: readers match nodes by header and
    identical sequence. ``add`` checks the array (one row per residue, one
    feature width for the whole file, floating-point and finite values) and
    stores it at the saving mode's precision.

    Opening an incomplete file resumes it, as Generate Embeddings does: the
    model label and saving mode must be the same, and no sequence may have been
    removed or changed. Leaving the ``with`` block normally checks the file and
    marks it complete when every sequence has its array. Otherwise, or after an
    error, it stays incomplete, and every reader refuses an incomplete file.
    """

    def __init__(self, path, fasta, model_name, saving_mode):
        validate_model_label(model_name)
        self.dtype = dtype_for_saving_mode(saving_mode)
        if isinstance(fasta, (str, os.PathLike)):
            from utilities.Sequence_Utils import load_sanitized_fasta

            headers, sequences, _ = load_sanitized_fasta(os.fspath(fasta))
        else:
            headers, sequences = (list(part) for part in fasta)
        validate_manifest_records(headers, sequences)
        if not headers:
            raise ValueError("The FASTA holds no sequences.")
        self.path = os.fspath(path)
        self.model_name = model_name
        self.saving_mode = saving_mode
        self.headers = headers
        self.manifest = None
        self._sequence_by_header = dict(zip(headers, sequences))
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = h5py.File(self.path, "a")
        try:
            self.feature_dimension = self._open_manifest(headers, sequences)
            self._written = set(self._file["embeddings"].keys())
            if self.pending:
                self._file.attrs["generation_complete"] = False
                self._file.flush()
        except BaseException:
            self._file.close()
            self._file = None
            raise

    def _open_manifest(self, headers, sequences):
        hf = self._file
        if len(hf) == 0 and len(hf.attrs) == 0:
            create_metadata_first_file(
                hf, headers, sequences, self.model_name, self.saving_mode
            )
            return None
        existing = read_embedding_manifest(
            hf, require_complete=False, validate_embeddings=True
        )
        if existing.model_name != self.model_name:
            raise ValueError(
                f"Cannot resume model '{self.model_name}' in a file of model "
                f"'{existing.model_name}'."
            )
        if existing.saving_mode != self.saving_mode:
            raise ValueError(
                f"Cannot resume {self.saving_mode} writing in a "
                f"{existing.saving_mode} file."
            )
        removed = sorted(set(existing.headers) - set(headers))
        if removed:
            raise ValueError(
                f"Cannot resume because existing header '{removed[0]}' was "
                "removed from the sanitized FASTA."
            )
        changed = sorted(
            header
            for header in set(existing.headers) & set(headers)
            if existing.sequence_by_header[header] != self._sequence_by_header[header]
        )
        if changed:
            raise ValueError(
                f"Cannot resume because the sanitized sequence for '{changed[0]}' "
                "changed."
            )
        if existing.headers != headers or existing.sequences != sequences:
            write_embedding_manifest(
                hf, headers, sequences, self.model_name, self.saving_mode, replace=True
            )
        return existing.feature_dimension

    @property
    def pending(self):
        """The sanitized (header, sequence) pairs that still lack an array."""
        return [
            (header, self._sequence_by_header[header])
            for header in self.headers
            if header not in self._written
        ]

    @property
    def written(self):
        """How many sequences have their array in the file."""
        return len(self._written)

    def add(self, header, embedding):
        """Check one sequence's residues-by-features array and store it."""
        if self._file is None:
            raise ValueError("The embedding file is closed.")
        sequence = self._sequence_by_header.get(header)
        if sequence is None:
            raise KeyError(f"'{header}' is not a sanitized header of this FASTA.")
        if header in self._written:
            raise ValueError(f"'{header}' already has an embedding in this file.")
        array = np.asarray(embedding)
        problem = embedding_shape_problem(array.shape, len(sequence))
        if problem:
            raise ValueError(f"The embedding for '{header}' {problem}.")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(
                f"The embedding for '{header}' holds {array.dtype} values, not "
                "floating-point ones."
            )
        finite_before = bool(np.isfinite(array).all())
        with np.errstate(over="ignore"):
            stored = array.astype(self.dtype)
        if finite_before and not np.isfinite(stored).all():
            raise ValueError(
                f"The embedding for '{header}' has values beyond the range of "
                f"{self.saving_mode}; save it as float32."
            )
        self.feature_dimension = validate_embedding_array(
            stored,
            sequence,
            self.saving_mode,
            feature_dimension=self.feature_dimension,
            require_finite=True,
            header=header,
        )
        self._file["embeddings"].create_dataset(header, data=stored)
        self._file.flush()
        self._written.add(header)

    def set_attributes(self, **attributes):
        """Store extra file attributes, such as where the embeddings came from."""
        if self._file is None:
            raise ValueError("The embedding file is closed.")
        reserved = sorted(set(attributes) & set(REQUIRED_ATTRIBUTES))
        if reserved:
            raise ValueError(f"Attribute '{reserved[0]}' is set by the writer itself.")
        for name, value in attributes.items():
            self._file.attrs[name] = value
        self._file.flush()

    def close(self):
        """Mark the file complete if every sequence has its array, then close it.

        Returns the complete file's manifest, or None while sequences are missing.
        """
        if self._file is None:
            return self.manifest
        try:
            if not self.pending:
                self.manifest = mark_generation_complete(self._file)
        finally:
            self._file.close()
            self._file = None
        return self.manifest

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        elif self._file is not None:
            self._file.close()
            self._file = None
        return False


__all__ = [
    "SAVING_MODE_DTYPES",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_OBJECTS",
    "SANITIZED_RESIDUE_CODES",
    "EmbeddingManifest",
    "decode_string_dataset",
    "dtype_for_saving_mode",
    "saving_mode_for_dtype",
    "validate_manifest_records",
    "validate_embedding_array",
    "read_embedding_manifest",
    "write_embedding_manifest",
    "create_metadata_first_file",
    "mark_generation_complete",
    "FOREIGN_FILE_ADVICE",
    "MODEL_LABEL_PATTERN",
    "validate_model_label",
    "embedding_shape_problem",
    "EmbeddingWriter",
]
