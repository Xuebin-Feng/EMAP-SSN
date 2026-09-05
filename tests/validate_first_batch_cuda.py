"""Read-only production-shaped trial and full-batch timing on configured input."""
import json
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/tools")]
os.environ["SSN_TOOL_SETTINGS_SCRIPT"] = "Align_Similarity_Matrix.py"
os.environ["SSN_TOOL_SETTINGS_FILE"] = str(ROOT / "tests/nonexistent-settings.json")
import Align_Similarity_Matrix as alignment
import torch
from utilities.Embedding_Alignment_Engine import EmbeddingTileStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-tiles", action="store_true", help="Disable host packing to reproduce the reported HDF5-tile workload.")
    parser.add_argument("--skip-full-scalar", action="store_true", help="Still benchmark scalar, but skip its long full-batch run.")
    args = parser.parse_args()
    settings = json.loads((ROOT / "tools_settings.json").read_text())
    config = settings["Align_Similarity_Matrix.py"]
    path = Path(settings["DIRECTORIES"]["EMBED_DIR"]) / config["INPUT_HDF5"]
    headers, safe, lengths, _ = alignment.load_embedding_metadata(str(path))
    from itertools import islice
    tasks = list(islice(
        ((i, j, safe[i], safe[j]) for i in range(len(safe)) for j in range(i + 1, len(safe))),
        int(config["BATCH_SIZE"]),
    ))
    alignment.EXECUTION_MODE = "auto"
    alignment.DEVICE_SELECTION = "cuda:0"
    alignment.LOCAL_GAP_P = config["LOCAL_GAP_P"]
    alignment.GLOBAL_GAP_P = config["GLOBAL_GAP_P"]
    store = EmbeddingTileStore(str(path), safe, 0 if args.hdf5_tiles else config["HOST_CACHE_GB"])
    print(f"Input: {path}; {len(tasks)} first-batch pairs; host cached={store.fully_cached}", flush=True)
    plans = alignment._benchmark_processing_plans(
        tasks, config["WORKERS"], str(path), -1,
        embedding_store=store, sequence_lengths=lengths, matmul_precision="tf32",
    )
    for variant in ("tiled", "scalar"):
        if variant == "scalar" and args.skip_full_scalar:
            continue
        selected = next(p for p in plans if p.variant == variant)
        print(f"FULL BATCH START {variant} lanes={selected.lanes}", flush=True)
        started = time.perf_counter()
        if variant == "tiled":
            results = alignment.run_tiled_accelerator_pipeline(
                tasks, store=store, lengths=lengths, device=selected.candidate.device,
                workers=config["WORKERS"], lanes=selected.lanes,
                alignment_callback=alignment.calculate_alignment_data, precision="tf32",
                memory_plan_override=selected.execution_plan.memory_plan,
            )
        else:
            results = alignment._run_accelerated_pipeline(
                tasks, config["WORKERS"], str(path), selected.candidate.device, -1,
                selected.lanes, False, matmul_precision="tf32",
            )
        elapsed = time.perf_counter() - started
        assert len(results) == len(tasks)
        assert len({(r[0], r[1]) for r in results}) == len(tasks)
        print(f"FULL BATCH {variant}: {len(results)} pairs / {elapsed:.3f}s = {len(results)/elapsed:.2f} pairs/s; trial={selected.value:.2f}", flush=True)
        del results
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
