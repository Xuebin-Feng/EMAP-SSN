"""Non-gating CUDA benchmark for Injection and fixed-query SSEARCH.

Usage:
    python tests/benchmark_injection_ssearch_cuda.py embeddings.h5
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from utilities.Alignment_Score_Kernels import global_local_scores, local_score_length
from utilities.Embedding_Alignment_Engine import (
    EmbeddingTileStore,
    compute_score_matrix_torch,
    run_fixed_query_cuda_pipeline,
    run_tiled_cuda_pipeline,
)


def injection_finish(args):
    left, right, matrix = args
    global_score, global_length, local_score, local_length = global_local_scores(
        matrix, 0.0, -2.0
    )
    return left, right, local_score, local_length, global_score, global_length


def search_finish(task, query_length, target_length, matrix):
    score, length = local_score_length(matrix, -2.0)
    return task[0], float(score), int(length), query_length, target_length


parser = argparse.ArgumentParser()
parser.add_argument("embedding_h5")
parser.add_argument("--pairs", type=int, default=4096)
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this benchmark.")

with h5py.File(args.embedding_h5, "r") as hf:
    headers = list(hf["embeddings"].keys())
store = EmbeddingTileStore(args.embedding_h5, headers, 0)
lengths = [shape[0] for shape in store.shapes]
device = torch.device("cuda:0")

pair_tasks = []
for left in range(max(0, len(headers) - 1)):
    for right in range(left + 1, len(headers)):
        pair_tasks.append((left, right, headers[left], headers[right]))
        if len(pair_tasks) >= args.pairs:
            break
    if len(pair_tasks) >= args.pairs:
        break

with h5py.File(args.embedding_h5, "r") as hf:
    group = hf["embeddings"]
    started = time.perf_counter()
    scalar_results = []
    for left, right, _left_header, _right_header in pair_tasks:
        matrix = compute_score_matrix_torch(
            group[headers[left]][:], group[headers[right]][:], device
        )
        scalar_results.append(injection_finish((left, right, matrix)))
    scalar_seconds = time.perf_counter() - started

torch.cuda.reset_peak_memory_stats(device)
started = time.perf_counter()
tiled_results = run_tiled_cuda_pipeline(
    pair_tasks,
    store=store,
    lengths=lengths,
    device=device,
    workers=args.workers,
    lanes=min(args.workers, 4),
    alignment_callback=injection_finish,
)
tiled_seconds = time.perf_counter() - started
print(
    f"Injection: scalar={len(pair_tasks) / scalar_seconds:.1f} pairs/s, "
    f"tiled={len(pair_tasks) / tiled_seconds:.1f} pairs/s, "
    f"speedup={scalar_seconds / tiled_seconds:.2f}x, "
    f"peak={torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f} MiB"
)
if {result[:2] for result in scalar_results} != {result[:2] for result in tiled_results}:
    raise RuntimeError("Injection scalar/tiled pair identities differ.")

with h5py.File(args.embedding_h5, "r") as hf:
    query = hf["embeddings"][headers[0]][:]
search_tasks = [(index, headers[index]) for index in range(min(args.pairs, len(headers)))]
torch.cuda.reset_peak_memory_stats(device)
started = time.perf_counter()
search_results = run_fixed_query_cuda_pipeline(
    search_tasks,
    query_embedding=query,
    store=store,
    lengths=lengths,
    device=device,
    workers=args.workers,
    lanes=min(args.workers, 4),
    alignment_callback=search_finish,
)
search_seconds = time.perf_counter() - started
print(
    f"SSEARCH tiled: {len(search_results) / search_seconds:.1f} targets/s, "
    f"peak={torch.cuda.max_memory_allocated(device) / (1024 ** 2):.1f} MiB"
)
