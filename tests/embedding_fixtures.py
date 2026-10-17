# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Small embedding files in the format the Tools write (utilities.HDF5_Storage)."""

import h5py
import numpy as np

from utilities import HDF5_Storage


def residue_values(record_index, length, width):
    """The matrix written for a record: exact in float16, distinct per record."""
    rows = np.arange(length, dtype=np.float32)[:, None] * 0.5
    columns = np.arange(width, dtype=np.float32)[None, :] * 0.25
    return record_index + rows + columns


def write_embedding_file(path, records, *, model_name="test_model", saving_mode="float16",
                         width=3, complete=True):
    """Write RECORDS, (header, sanitized sequence) pairs, as an embedding database."""
    headers = [header for header, _ in records]
    sequences = [sequence for _, sequence in records]
    dtype = HDF5_Storage.dtype_for_saving_mode(saving_mode)
    with h5py.File(path, "w") as handle:
        group = HDF5_Storage.create_metadata_first_file(
            handle, headers, sequences, model_name, saving_mode
        )
        for index, (header, sequence) in enumerate(records):
            group.create_dataset(
                header, data=residue_values(index, len(sequence), width).astype(dtype)
            )
        if complete:
            HDF5_Storage.mark_generation_complete(handle)
    return path
