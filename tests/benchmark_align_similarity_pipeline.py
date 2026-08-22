"""Non-gating scalar-versus-tiled CUDA alignment benchmark.

Run from the repository root with the project environment::

    python tests/benchmark_align_similarity_pipeline.py
"""

import os
import sys
import tempfile
import time

import h5py
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "src", "tools")
UTILITIES_DIR = os.path.join(PROJECT_ROOT, "src", "utilities")
for path in (TOOLS_DIR, UTILITIES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import Align_Similarity_Matrix as alignment
from utilities.Embedding_Alignment_Engine import (
    EmbeddingTileStore,
    cuda_memory_plan,
    run_tiled_cuda_pipeline,
)


if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this benchmark.")

rng = np.random.default_rng(20260821)
sequence_count = 92
feature_dimension = 256
lengths = [96 + (index * 17) % 65 for index in range(sequence_count)]
headers = [f"sequence_{index:04d}" for index in range(sequence_count)]
tasks = [
    (row, column, headers[row], headers[column])
    for row in range(sequence_count)
    for column in range(row + 1, sequence_count)
][:4096]
device = torch.device("cuda:0")
workers = min(12, os.cpu_count() or 1)
lanes = min(4, workers)

with tempfile.TemporaryDirectory() as temp_dir:
    input_h5 = os.path.join(temp_dir, "benchmark_embeddings.h5")
    with h5py.File(input_h5, "w") as hf:
        group = hf.create_group("embeddings")
        for header, length in zip(headers, lengths):
            group.create_dataset(
                header,
                data=rng.normal(size=(length, feature_dimension)).astype(np.float32),
            )

    store = EmbeddingTileStore(input_h5, headers, "auto")
    warmup = tasks[:32]
    alignment._run_accelerated_pipeline(
        warmup,
        workers,
        input_h5,
        device,
        0,
        accelerator_workers=lanes,
        show_progress=False,
    )
    run_tiled_cuda_pipeline(
        warmup,
        store=store,
        lengths=lengths,
        device=device,
        workers=workers,
        lanes=lanes,
        alignment_callback=alignment.calculate_alignment_data,
        precision="float32",
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    scalar = alignment._run_accelerated_pipeline(
        tasks,
        workers,
        input_h5,
        device,
        0,
        accelerator_workers=lanes,
        show_progress=False,
    )
    scalar_seconds = time.perf_counter() - started
    scalar_peak = torch.cuda.max_memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    tiled = run_tiled_cuda_pipeline(
        tasks,
        store=store,
        lengths=lengths,
        device=device,
        workers=workers,
        lanes=lanes,
        alignment_callback=alignment.calculate_alignment_data,
        precision="float32",
    )
    tiled_seconds = time.perf_counter() - started
    tiled_peak = torch.cuda.max_memory_allocated(device)

plan = cuda_memory_plan(device, lanes=lanes)
print(f"Device: {torch.cuda.get_device_name(device)}")
print(f"Pairs: {len(tasks)}, workers: {workers}, lanes: {lanes}")
print(f"Host cache: {store.cached_bytes / (1024 ** 2):.1f} MiB")
print(f"CUDA tile budget: {plan.tile_cache_bytes / (1024 ** 2):.1f} MiB")
print(f"CUDA matrix pool: {plan.matrix_pool_bytes / (1024 ** 2):.1f} MiB")
print(
    f"CUDA per-microbatch budget: "
    f"{plan.matrix_bytes / (1024 ** 2):.1f} MiB "
    f"across {plan.inflight_slots} in-flight slots"
)
print(
    f"Scalar: {len(tasks) / scalar_seconds:.1f} pairs/s, "
    f"peak {scalar_peak / (1024 ** 2):.1f} MiB"
)
print(
    f"Tiled:  {len(tasks) / tiled_seconds:.1f} pairs/s, "
    f"peak {tiled_peak / (1024 ** 2):.1f} MiB"
)
print(f"Speedup: {scalar_seconds / tiled_seconds:.2f}x")
if {result[:2] for result in scalar} != {result[:2] for result in tiled}:
    raise RuntimeError("Scalar and tiled pair sets differ.")
