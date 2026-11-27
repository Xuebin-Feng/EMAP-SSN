# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Align_Similarity_Matrix pipeline, tiled, and resume tests."""

import io
import os
import sys
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
UTILITIES_DIR = os.path.join(SRC_DIR, "utilities")
TOOLS_DIR = os.path.join(SRC_DIR, "tools")
for _folder in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if _folder not in sys.path:
        sys.path.insert(0, _folder)

# The tests package points tool imports at a missing settings file.
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Align_Similarity_Matrix as similarity_matrix
    import Embedding_Alignment_Engine as alignment_engine

from tests.embedding_fixtures import write_embedding_arrays  # noqa: E402


COMPILE_BATCHES = {
    "batch_00000.h5": [(0, 1), (0, 2), (0, 3)],
    "batch_00001.h5": [(1, 2), (1, 3), (2, 3)],
    "batch_00002.h5": [(3, 4), (4, 5)],
}
OLD_ATTRIBUTES = {"embedding_checksum", "model_name", "gap_penalties", "matmul_precision"}
OLD_DATASETS = {"headers", "seq_lens", "i", "j", "l_score", "l_len", "g_score", "g_len"}


class FakeStreamBackend:
    """Stream-capable backend stand-in that runs the tiled math on the CPU."""

    device_type = "cuda"

    def __init__(self, log):
        self.log = log

    def supports_tiled(self, require_memory=True):
        return True, "fake streams"

    def create_stream(self):
        return object()

    def stream_context(self, stream):
        return nullcontext()

    def create_event(self):
        return SimpleNamespace(
            record=lambda stream: None,
            query=lambda: True,
            synchronize=lambda: None,
        )

    def empty_cache(self):
        self.log.append("empty-cache")


def write_random_embeddings(path, lengths, features, seed):
    rng = np.random.default_rng(seed)
    headers = [f"seq{index}" for index in range(len(lengths))]
    write_embedding_arrays(
        path,
        {
            header: rng.normal(size=(length, features)).astype(np.float32)
            for header, length in zip(headers, lengths)
        },
        saving_mode="float32",
    )
    return headers


def small_tile_plan(features, lanes):
    """About a dozen embedding rows per block, so tiles change often."""
    return alignment_engine.CudaMemoryPlan(
        free_bytes=1 << 30,
        total_bytes=1 << 30,
        usable_bytes=1 << 30,
        tile_cache_bytes=2 * 12 * features * 4,
        matrix_pool_bytes=1 << 22,
        matrix_bytes=1 << 20,
        reserve_bytes=0,
        lanes=lanes,
        inflight_slots=max(2, 2 * lanes),
    )


def write_compile_batches(results_dir, corrupt=None, corruption=None):
    os.makedirs(results_dir)
    for name, pairs in COMPILE_BATCHES.items():
        path = os.path.join(results_dir, name)
        if name == corrupt and corruption == "not_hdf5":
            with open(path, "wb") as handle:
                handle.write(b"truncated batch")
            continue
        arr_i = np.array([pair[0] for pair in pairs], dtype=np.uint16)
        arr_j = np.array([pair[1] for pair in pairs], dtype=np.uint16)
        columns = {
            "i": arr_i,
            "j": arr_j,
            "l_score": (arr_i * 10 + arr_j + 0.5).astype(np.float32),
            "l_len": (arr_i + arr_j + 1).astype(np.uint16),
            "g_score": (arr_i * 10 + arr_j + 0.25).astype(np.float32),
            "g_len": (arr_i + arr_j + 2).astype(np.uint16),
        }
        if name == corrupt and corruption == "missing_l_score":
            del columns["l_score"]
        with h5py.File(path, "w") as hf:
            for column, data in columns.items():
                hf.create_dataset(column, data=data)


def compile_batches(temp_dir, required_keys=None):
    output = os.path.join(temp_dir, "network", "set_[model]_network.h5")
    stdout = io.StringIO()
    with mock.patch.object(
        similarity_matrix,
        "RESULTS_DIR",
        os.path.join(temp_dir, "batches"),
    ), mock.patch.object(
        similarity_matrix,
        "FINAL_OUTPUT_NET",
        output,
    ), redirect_stdout(stdout), redirect_stderr(io.StringIO()):
        similarity_matrix.compile_final_output(
            [f"s{index}" for index in range(6)],
            [5] * 6,
            6,
            "checksum",
            "model",
            "float32",
            [-2.0, 0.0],
            "ieee_fp32",
            required_keys=required_keys,
        )
    return output, stdout.getvalue()


def run_align_main(temp_dir, **patches):
    """Run Align_Similarity_Matrix.main() on temporary paths with settings and compile stubbed."""
    values = {
        "INPUT_HDF5": os.path.join(temp_dir, "set_[model]_embeddings.h5"),
        "NETWORK_DIR": os.path.join(temp_dir, "network"),
        "EXECUTION_MODE": "auto",
        "ACCELERATOR_PRECISION": "automatic_32bit",
        "EDGE_PREFILTERING": False,
        **patches,
    }
    # The job assigns these globals; patching them restores the originals.
    assigned = (
        "MODEL_NAME",
        "SEQUENCE_SET",
        "FULL_INPUT_HDF5",
        "RESULTS_DIR",
        "FINAL_OUTPUT_NET",
        "active_matmul_precision",
        "active_embedding_store",
        "active_sequence_lengths",
    )
    stdout = io.StringIO()
    with ExitStack() as stack:
        for name in assigned:
            stack.enter_context(mock.patch.object(
                similarity_matrix, name, getattr(similarity_matrix, name)
            ))
        for name, value in values.items():
            stack.enter_context(mock.patch.object(similarity_matrix, name, value))
        stack.enter_context(mock.patch.object(similarity_matrix, "load_tool_settings"))
        stack.enter_context(mock.patch.object(similarity_matrix, "set_start_method"))
        compile_output = stack.enter_context(
            mock.patch.object(similarity_matrix, "compile_final_output")
        )
        stack.enter_context(redirect_stdout(stdout))
        stack.enter_context(redirect_stderr(io.StringIO()))
        exit_code = similarity_matrix.main([])
    return exit_code, stdout.getvalue(), compile_output
