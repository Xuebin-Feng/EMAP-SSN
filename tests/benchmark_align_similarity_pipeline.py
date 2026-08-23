"""Non-gating scalar-versus-persistent-tiled CUDA alignment benchmark.

Run from the repository root with the project environment::

    python tests/benchmark_align_similarity_pipeline.py
"""

import os
import sys
import tempfile
import time
from statistics import median

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
    TiledAcceleratorSession,
    EmbeddingTileStore,
    accelerator_memory_plan,
    compare_precision_results,
    get_accelerator_backend,
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
backend = get_accelerator_backend(device)
workers = min(12, os.cpu_count() or 1)
lanes = min(4, workers)
repetitions = 3
memory_profiles = [
    ("microbatch-heavy", 0.20, 0.60),
    ("balanced-default", 0.30, 0.50),
    ("tile-heavy", 0.40, 0.40),
]

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
    scalar_times = []
    scalar_peaks = []
    scalar_reference = None
    for _ in range(repetitions):
        torch.cuda.empty_cache()
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
            matmul_precision="float32",
        )
        scalar_times.append(time.perf_counter() - started)
        scalar_peaks.append(torch.cuda.max_memory_allocated(device))
        if scalar_reference is None:
            scalar_reference = scalar

    scalar_seconds = median(scalar_times)
    scalar_peak = median(scalar_peaks)
    torch.cuda.empty_cache()
    memory_info = backend.memory_info()
    profile_results = []
    for name, tile_fraction, matrix_fraction in memory_profiles:
        plan = accelerator_memory_plan(
            device,
            lanes=lanes,
            memory_info=memory_info,
            tile_fraction=tile_fraction,
            matrix_fraction=matrix_fraction,
        )
        session = TiledAcceleratorSession(
            store=store,
            lengths=lengths,
            device=device,
            workers=workers,
            lanes=lanes,
            alignment_callback=alignment.calculate_alignment_data,
            precision="float32",
            memory_plan_override=plan,
        )
        try:
            session.run(warmup)
            elapsed_runs = []
            peak_runs = []
            for _ in range(repetitions):
                backend.reset_peak_memory_stats()
                started = time.perf_counter()
                tiled = session.run(tasks)
                elapsed_runs.append(time.perf_counter() - started)
                peak_runs.append(backend.max_memory_allocated())
                equivalent, reason = compare_precision_results(
                    scalar_reference,
                    tiled,
                )
                if not equivalent:
                    raise RuntimeError(
                        f"Profile {name} failed validation: {reason}"
                    )
            metrics = session.metrics()
        finally:
            session.close()
        profile_results.append(
            (name, plan, median(elapsed_runs), median(peak_runs), metrics)
        )

print(f"Device: {torch.cuda.get_device_name(device)}")
print(f"Pairs: {len(tasks)}, workers: {workers}, lanes: {lanes}")
print(f"Host cache: {store.cached_bytes / (1024 ** 2):.1f} MiB")
print(
    f"Scalar median ({repetitions} runs): "
    f"{len(tasks) / scalar_seconds:.1f} pairs/s, "
    f"peak {scalar_peak / (1024 ** 2):.1f} MiB"
)
print(
    "Profile             Tile MiB   Transient MiB   Matrix MiB   "
    "Pairs/s   GPU peak   Host stage   Cache hit   vs scalar"
)
for name, plan, tiled_seconds, tiled_peak, metrics in profile_results:
    cache_requests = (
        metrics["embedding_cache_hits"]
        + metrics["embedding_cache_misses"]
    )
    cache_hit_rate = (
        metrics["embedding_cache_hits"] / max(1, cache_requests)
    )
    print(
        f"{name:19} "
        f"{plan.tile_cache_bytes / (1024 ** 2):>8.1f}   "
        f"{plan.transient_pool_bytes / (1024 ** 2):>13.1f}   "
        f"{plan.matrix_pool_bytes / (1024 ** 2):>10.1f}   "
        f"{len(tasks) / tiled_seconds:>7.1f}   "
        f"{tiled_peak / (1024 ** 2):>8.1f}   "
        f"{metrics['peak_host_staging_bytes'] / (1024 ** 2):>10.1f}   "
        f"{cache_hit_rate:>8.1%}   "
        f"{scalar_seconds / tiled_seconds:>8.2f}x"
    )
