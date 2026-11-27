# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Small embedding files in the format the Tools write (utilities.HDF5_Storage)."""

from collections.abc import Mapping
import os
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import HDF5_Storage  # noqa: E402


def residue_values(record_index, length, width):
    """The matrix written for a record: exact in float16, distinct per record."""
    rows = np.arange(length, dtype=np.float32)[:, None] * 0.5
    columns = np.arange(width, dtype=np.float32)[None, :] * 0.25
    return record_index + rows + columns


def write_embedding_arrays(
    path,
    embeddings_by_header,
    *,
    sequences=None,
    model_name="test_model",
    saving_mode=None,
    complete=True,
    attrs=None,
):
    """Write explicit per-header embedding arrays as a standard EMAP-SSN HDF5 file.

    ``embeddings_by_header`` may be a mapping ``{header: array}`` or an iterable
    of ``(header, array)`` pairs. When ``saving_mode`` is omitted, it is inferred
    from the first array's dtype (defaulting to ``"float32"`` when empty). When
    ``sequences`` is omitted, placeholder ``"A" * rows`` sequences are written.
    """
    items = (
        list(embeddings_by_header.items())
        if isinstance(embeddings_by_header, Mapping)
        else list(embeddings_by_header)
    )
    headers = [header for header, _ in items]
    arrays = [np.asarray(array) for _, array in items]

    if saving_mode is None:
        saving_mode = (
            HDF5_Storage.saving_mode_for_dtype(arrays[0].dtype)
            if arrays
            else "float32"
        )
    dtype = HDF5_Storage.dtype_for_saving_mode(saving_mode)

    if sequences is None:
        resolved_sequences = ["A" * int(array.shape[0]) for array in arrays]
    elif isinstance(sequences, Mapping):
        resolved_sequences = [sequences[header] for header in headers]
    else:
        resolved_sequences = list(sequences)

    with h5py.File(path, "w") as handle:
        group = HDF5_Storage.create_metadata_first_file(
            handle, headers, resolved_sequences, model_name, saving_mode
        )
        for header, array in zip(headers, arrays):
            group.create_dataset(header, data=array.astype(dtype, copy=False))
        if complete:
            HDF5_Storage.mark_generation_complete(handle)
        if attrs:
            for key, value in attrs.items():
                handle.attrs[key] = value
            handle.flush()
    return path


def write_embedding_file(path, records, *, model_name="test_model", saving_mode="float16",
                         width=3, complete=True):
    """Write RECORDS, (header, sanitized sequence) pairs, as an embedding database."""
    dtype = HDF5_Storage.dtype_for_saving_mode(saving_mode)
    arrays = [
        (header, residue_values(index, len(sequence), width).astype(dtype))
        for index, (header, sequence) in enumerate(records)
    ]
    sequences = [sequence for _, sequence in records]
    return write_embedding_arrays(
        path,
        arrays,
        sequences=sequences,
        model_name=model_name,
        saving_mode=saving_mode,
        complete=complete,
    )
