"""End-to-end stable-baseline-versus-current tiled alignment benchmark.

Examples::

    python tests/benchmark_align_similarity_pipeline.py --pairs 65536
    python tests/benchmark_align_similarity_pipeline.py \
        --embeddings Cache_Files/my_embeddings.h5 --duration 60 \
        --json metrics.json

With a real embeddings file, pairs are sampled in production row-major
``i < j`` order. The benchmark never changes scientific output files.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
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
    _LegacyTiledAcceleratorSession,
    AcceleratorExecutionPlan,
    EmbeddingTileStore,
    TiledAcceleratorSession,
    accelerator_memory_plan,
    benchmark_accelerator_execution_plans,
    compare_precision_results,
    get_accelerator_backend,
    is_nvidia_cuda,
    measure_accelerator_session,
)
from utilities.Alignment_Score_Kernels import global_local_scores


def legacy_alignment_callback(args):
    """Pre-refactor per-pair scratch-allocation behavior for the baseline."""
    idx_i, idx_j, matrix = args
    g_raw, g_len, l_raw, l_len = global_local_scores(
        matrix,
        alignment.GLOBAL_GAP_P,
        alignment.LOCAL_GAP_P,
    )
    return idx_i, idx_j, l_raw, l_len, g_raw, g_len


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", help="Real embeddings HDF5 file.")
    parser.add_argument("--device", default="auto", help="auto, cuda:N, or xpu:N")
    parser.add_argument("--pairs", type=int, default=65536)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--lanes", type=int, nargs="*", default=None)
    parser.add_argument("--active-cpu-workers", type=int)
    parser.add_argument(
        "--cpu-chunk-size", type=int, choices=(1, 2, 4, 8)
    )
    parser.add_argument("--microbatch-mib", type=int)
    parser.add_argument(
        "--precision", choices=("float32", "tf32"), default="float32"
    )
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--profile", help="Optional PyTorch profiler trace path.")
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--synthetic-sequences", type=int, default=370)
    parser.add_argument("--synthetic-dimension", type=int, default=256)
    args = parser.parse_args()
    if args.pairs < 1 or args.duration < 0 or args.repetitions < 1:
        parser.error("pairs/repetitions must be positive and duration nonnegative")
    if args.active_cpu_workers is not None and args.active_cpu_workers < 1:
        parser.error("active CPU workers must be positive")
    if args.microbatch_mib is not None and args.microbatch_mib < 1:
        parser.error("microbatch MiB must be positive")
    return args


def choose_device(spec):
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")
    raise SystemExit("A CUDA/ROCm or Intel XPU accelerator is required.")


def canonical_tasks(headers, limit):
    tasks = []
    for row in range(len(headers)):
        for column in range(row + 1, len(headers)):
            tasks.append((row, column, headers[row], headers[column]))
            if len(tasks) >= limit:
                return tasks
    return tasks


def create_synthetic(path, sequence_count, feature_dimension):
    rng = np.random.default_rng(20260823)
    headers = [f"sequence_{index:05d}" for index in range(sequence_count)]
    with h5py.File(path, "w") as hf:
        group = hf.create_group("embeddings", track_order=True)
        for index, header in enumerate(headers):
            length = 96 + (index * 17) % 257
            group.create_dataset(
                header,
                data=rng.normal(size=(length, feature_dimension)).astype(np.float32),
            )
    return headers


def read_headers(path):
    with h5py.File(path, "r", libver="latest", swmr=True) as hf:
        if "embeddings" not in hf:
            raise ValueError("Embeddings HDF5 lacks an 'embeddings' group.")
        return list(hf["embeddings"].keys())


def profiler_context(device, trace_path):
    if not trace_path:
        return nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif device.type == "xpu" and hasattr(torch.profiler.ProfilerActivity, "XPU"):
        activities.append(torch.profiler.ProfilerActivity.XPU)
    return torch.profiler.profile(
        activities=activities,
        on_trace_ready=lambda profile: profile.export_chrome_trace(trace_path),
        record_shapes=True,
        profile_memory=True,
    )


def run_repetitions(
    factory, tasks, repetitions, minimum_duration, profile_path=None
):
    times = []
    peaks = []
    reference = None
    final_metrics = None
    total = 0.0
    iteration = 0
    while iteration < repetitions or total < minimum_duration:
        session = factory()
        try:
            session.run(tasks[:min(32, len(tasks))])
            session.backend.reset_peak_memory_stats()
            context = profiler_context(
                session.device,
                profile_path if iteration == 0 else None,
            )
            started = time.perf_counter()
            with context:
                results = session.run(tasks)
            elapsed = max(time.perf_counter() - started, 1e-9)
            times.append(elapsed)
            peaks.append(session.backend.max_memory_allocated())
            final_metrics = session.metrics()
            if reference is None:
                reference = results
        finally:
            session.close()
        total += elapsed
        iteration += 1
    return {
        "seconds": median(times),
        "pairs_per_second": len(tasks) / median(times),
        "peak_allocated_bytes": int(median(peaks)),
        "repetitions": len(times),
        "results": reference,
        "metrics": final_metrics,
    }


def run_paired_repetitions(
    baseline_factory,
    current_factory,
    tasks,
    repetitions,
    minimum_duration,
    profile_path=None,
):
    """Alternate stable/current order to suppress cache and thermal drift."""
    records = {
        "baseline": {"times": [], "peaks": [], "results": None, "metrics": None},
        "current": {"times": [], "peaks": [], "results": None, "metrics": None},
    }
    totals = {"baseline": 0.0, "current": 0.0}
    iteration = 0
    while iteration < repetitions or min(totals.values()) < minimum_duration:
        order = (
            ("baseline", baseline_factory),
            ("current", current_factory),
        )
        if iteration % 2:
            order = tuple(reversed(order))
        for label, factory in order:
            session = factory()
            try:
                session.run(tasks[:min(32, len(tasks))])
                session.backend.reset_peak_memory_stats()
                context = profiler_context(
                    session.device,
                    (
                        profile_path
                        if label == "current" and iteration == 0
                        else None
                    ),
                )
                started = time.perf_counter()
                with context:
                    values = session.run(tasks)
                elapsed = max(time.perf_counter() - started, 1e-9)
                record = records[label]
                record["times"].append(elapsed)
                record["peaks"].append(
                    session.backend.max_memory_allocated()
                )
                record["metrics"] = session.metrics()
                if record["results"] is None:
                    record["results"] = values
            finally:
                session.close()
            totals[label] += elapsed
        iteration += 1

    reports = {}
    for label, record in records.items():
        seconds = median(record["times"])
        reports[label] = {
            "seconds": seconds,
            "pairs_per_second": len(tasks) / seconds,
            "peak_allocated_bytes": int(median(record["peaks"])),
            "repetitions": len(record["times"]),
            "results": record["results"],
            "metrics": record["metrics"],
        }
    return reports["baseline"], reports["current"]


def main():
    args = parse_arguments()
    device = choose_device(args.device)
    backend = get_accelerator_backend(device)
    temp = None
    if args.embeddings:
        input_h5 = os.path.abspath(args.embeddings)
        headers = read_headers(input_h5)
    else:
        temp = tempfile.TemporaryDirectory()
        input_h5 = os.path.join(temp.name, "benchmark_embeddings.h5")
        headers = create_synthetic(
            input_h5,
            args.synthetic_sequences,
            args.synthetic_dimension,
        )
    try:
        tasks = canonical_tasks(headers, args.pairs)
        if not tasks:
            raise ValueError("At least two embeddings are required.")
        store = EmbeddingTileStore(input_h5, headers, "auto")
        lengths = [int(shape[0]) for shape in store.shapes]
        lane_candidates = args.lanes or (
            [1, 2, 4] if device.type == "xpu" else [1, 2, 4, 8]
        )
        lane_candidates = [
            lane for lane in lane_candidates if 1 <= lane <= args.workers
        ] or [1]
        memory_info = backend.memory_info()

        def memory_factory(lanes):
            return accelerator_memory_plan(
                device, lanes=lanes, memory_info=memory_info
            )

        selected_plan = None
        tuning_observations = ()
        if not args.no_autotune:
            short = tasks[:min(256, len(tasks))]
            confirmation = tasks[:min(2048, len(tasks))]

            def measure(plan, plan_tasks):
                with TiledAcceleratorSession(
                    store=store,
                    lengths=lengths,
                    device=device,
                    workers=args.workers,
                    lanes=plan.lanes,
                    alignment_callback=alignment.calculate_alignment_data,
                    alignment_chunk_callback=alignment.calculate_alignment_chunk,
                    precision=args.precision,
                    memory_plan_override=memory_factory(plan.lanes),
                    execution_plan=plan,
                    print_summary=False,
                ) as session:
                    values, rate, measured_pairs = (
                        measure_accelerator_session(session, plan_tasks)
                    )
                    metrics = session.metrics()
                return {
                    "rate": rate,
                    "measured_pairs": measured_pairs,
                    "peak_memory_bytes": metrics["peak_allocated_bytes"],
                    "compilation_seconds": metrics["compilation_seconds"],
                    "results": values,
                }

            warm_plan = AcceleratorExecutionPlan(
                lanes=lane_candidates[0],
                microbatch_workspace_bytes=min(
                    int(memory_factory(lane_candidates[0]).matrix_pool_bytes)
                    // max(1, lane_candidates[0]),
                    512 * 1024 ** 2,
                ),
                active_cpu_workers=args.workers,
                cpu_chunk_size=1,
            )
            measure(warm_plan, short[:min(32, len(short))])

            ranked, tuning_observations = benchmark_accelerator_execution_plans(
                lane_candidates=lane_candidates,
                memory_plan_factory=memory_factory,
                workers=args.workers,
                short_tasks=short,
                confirmation_tasks=confirmation,
                remaining_pairs=len(tasks),
                measure=measure,
                allow_compilation=True,
                allow_graph_compilation=is_nvidia_cuda(device),
            )
            if not ranked:
                raise RuntimeError("No tiled execution plan completed tuning.")
            selected_plan = ranked[0].execution_plan

        lanes = (
            selected_plan.lanes
            if selected_plan is not None
            else lane_candidates[0]
        )
        plan = memory_factory(lanes)
        if selected_plan is None:
            selected_plan = AcceleratorExecutionPlan(
                lanes=lanes,
                microbatch_workspace_bytes=min(
                    int(plan.matrix_pool_bytes) // max(1, lanes),
                    int(args.microbatch_mib or 512) * 1024 ** 2,
                ),
                active_cpu_workers=min(
                    args.workers,
                    int(args.active_cpu_workers or args.workers),
                ),
                cpu_chunk_size=int(args.cpu_chunk_size or 1),
            )
        print(
            "Selected plan: "
            f"lanes={selected_plan.lanes}, "
            f"cap={selected_plan.microbatch_workspace_bytes / 2**20:.0f} MiB, "
            f"workers={selected_plan.active_cpu_workers}, "
            f"chunk={selected_plan.cpu_chunk_size}, "
            f"policy={selected_plan.length_bucket_policy}, "
            f"scorer={selected_plan.scorer_variant}"
        )
        common = dict(
            store=store,
            lengths=lengths,
            device=device,
            workers=args.workers,
            lanes=lanes,
            precision=args.precision,
            memory_plan_override=plan,
        )
        baseline, current = run_paired_repetitions(
            lambda: _LegacyTiledAcceleratorSession(
                **common,
                alignment_callback=legacy_alignment_callback,
            ),
            lambda: TiledAcceleratorSession(
                **common,
                alignment_callback=alignment.calculate_alignment_data,
                alignment_chunk_callback=alignment.calculate_alignment_chunk,
                execution_plan=selected_plan,
                print_summary=False,
            ),
            tasks,
            args.repetitions,
            args.duration,
            args.profile,
        )
        equivalent, reason = compare_precision_results(
            baseline["results"], current["results"]
        )
        if not equivalent:
            raise RuntimeError(f"Scientific parity failed: {reason}")
        speedup = (
            current["pairs_per_second"] / baseline["pairs_per_second"]
        )
        report = {
            "device": backend.display_name,
            "device_type": device.type,
            "embeddings": input_h5 if args.embeddings else "synthetic",
            "pairs": len(tasks),
            "workers": args.workers,
            "precision": args.precision,
            "selected_execution_plan": (
                None if selected_plan is None else selected_plan.__dict__
            ),
            "baseline": {
                key: value for key, value in baseline.items() if key != "results"
            },
            "current": {
                key: value for key, value in current.items() if key != "results"
            },
            "speedup": speedup,
            "scientific_parity": reason,
            "acceptance_10_percent": speedup >= 1.10,
            "tuning": [
                {
                    "execution_plan": item.execution_plan.__dict__,
                    "pairs_per_second": item.pairs_per_second,
                    "peak_memory_bytes": item.peak_memory_bytes,
                    "compilation_seconds": item.compilation_seconds,
                    "error": item.error,
                }
                for item in tuning_observations
            ],
        }
        print(f"Device: {report['device']}")
        print(f"Pairs: {len(tasks)}, workers: {args.workers}, lanes: {lanes}")
        print(
            f"Stable tiled baseline: {baseline['pairs_per_second']:.1f} pairs/s; "
            f"current: {current['pairs_per_second']:.1f} pairs/s; "
            f"speedup {speedup:.2f}x"
        )
        print(
            f"Peak allocated: baseline "
            f"{baseline['peak_allocated_bytes'] / 2**20:.1f} MiB; current "
            f"{current['peak_allocated_bytes'] / 2**20:.1f} MiB; parity {reason}"
        )
        if args.json_path:
            with open(args.json_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, default=str)
        return 0 if speedup >= 1.10 else 2
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
