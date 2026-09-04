"""CPU scaling probe for the component-energy Monte Carlo layout engine."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN_MonteCarlo as monte_carlo


def benchmark(component_size: int, sweeps: int) -> tuple[float, float, int]:
    side = int(np.ceil(np.sqrt(component_size)))
    node_ids = np.arange(component_size)
    positions = np.column_stack((node_ids % side, node_ids // side)).astype(
        np.float64
    )
    positions -= positions.mean(axis=0)
    positions *= 3.5
    edges = np.column_stack((node_ids[:-1], node_ids[1:])).astype(np.int32)
    params = {
        "SPRING_K": 5.0,
        "COULOMB_K": 10.0,
        "COULOMB_CUTOFF": 30.0,
        "MAX_FORCE_LIMIT": 20.0,
        "MC_SWEEPS": sweeps,
        "MC_QUENCH_SWEEPS": 0,
        "MC_TELEPORT_PROBABILITY": 0.10,
    }

    tracemalloc.start()
    started = time.perf_counter()
    optimizer = monte_carlo.ComponentEnergyMonteCarlo(
        positions,
        edges,
        float(side * 4.0),
        params,
        np.random.default_rng(42),
    )
    optimizer.optimize()
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    spatial_entries = sum(len(nodes) for nodes in optimizer.spatial.cells.values())
    return elapsed, peak_bytes / (1024.0 * 1024.0), spatial_entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=(500, 2000, 5000)
    )
    arguments = parser.parse_args()
    monte_carlo.repulsive_energy_sum(
        np.zeros(2, dtype=np.float64),
        np.ones((1, 2), dtype=np.float64),
        np.asarray([0], dtype=np.int32),
        10.0,
        20.0,
        30.0,
    )
    for component_size in arguments.sizes:
        elapsed, peak_mib, spatial_entries = benchmark(
            component_size, arguments.sweeps
        )
        print(
            f"nodes={component_size} sweeps={arguments.sweeps} "
            f"seconds={elapsed:.3f} peak_traced_mib={peak_mib:.2f} "
            f"spatial_entries={spatial_entries}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
