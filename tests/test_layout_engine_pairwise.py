import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN_MolecularDynamics as molecular_dynamics


def brute_force_repulsion_step(
    positions,
    component_labels,
    active_mask,
    box_limits,
    params,
):
    positions = positions.astype(np.float32, copy=True)
    repulsion = np.zeros_like(positions)
    active_nodes = np.flatnonzero(active_mask)
    cutoff = float(params["COULOMB_CUTOFF"])
    taper_start = cutoff * 0.8
    taper_width = max(cutoff * 0.2, 1e-9)

    for left_index, node_a in enumerate(active_nodes):
        for node_b in active_nodes[left_index + 1:]:
            if component_labels[node_a] != component_labels[node_b]:
                continue

            delta = positions[node_a] - positions[node_b]
            distance_sq = float(np.dot(delta, delta))
            if distance_sq == 0.0 or distance_sq > cutoff * cutoff:
                continue

            distance = np.sqrt(distance_sq)
            force_magnitude = params["COULOMB_K"] / max(distance, 0.5) ** 2
            pair_cap = params.get("MAX_FORCE_LIMIT", 20.0)
            if pair_cap > 0.0:
                force_magnitude = min(force_magnitude, pair_cap)
            if distance > taper_start:
                force_magnitude *= max(
                    0.0,
                    (cutoff - distance) / taper_width,
                )

            force = force_magnitude * delta / distance
            repulsion[node_a] += force
            repulsion[node_b] -= force

    total_cap = params.get("MAX_TOTAL_REPULSION_FORCE", 0.0)
    if total_cap > 0.0:
        for node in active_nodes:
            force_norm = np.linalg.norm(repulsion[node])
            if force_norm > total_cap:
                repulsion[node] *= total_cap / force_norm

    velocity = np.zeros_like(positions)
    velocity[active_mask] += repulsion[active_mask] * params["DT"]
    positions[active_mask] += velocity[active_mask] * params["DT"]
    positions[active_mask] = np.maximum(
        np.minimum(positions[active_mask], box_limits[active_mask, None]),
        -box_limits[active_mask, None],
    )
    return positions


class MolecularDynamicsPairwiseTests(unittest.TestCase):
    def test_pair_moved_inside_cutoff_is_used_on_next_step(self):
        positions = np.array([[-6.1, 0.0], [6.1, 0.0]], dtype=np.float32)
        simulation = molecular_dynamics.SSNSimulationCPU(
            positions,
            np.zeros((0, 2), dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            100.0,
            {
                "DT": 1.0,
                "DAMPING": 0.0,
                "COULOMB_K": 9.8**2,
                "COULOMB_CUTOFF": 10.0,
                "MAX_FORCE_LIMIT": 1000.0,
            },
        )

        simulation.pos[:] = np.array(
            [[-4.9, 0.0], [4.9, 0.0]], dtype=np.float32
        )
        before = simulation.pos.copy()
        simulation.step(0)

        self.assertFalse(hasattr(simulation, "neighbor_pairs"))
        self.assertGreater(abs(simulation.pos[0, 0]), abs(before[0, 0]))
        self.assertGreater(abs(simulation.pos[1, 0]), abs(before[1, 0]))

    def test_cpu_matches_independent_all_pairs_reference(self):
        positions = np.array(
            [
                [-2.0, 0.0],
                [2.0, 0.0],
                [8.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [50.0, 50.0],
            ],
            dtype=np.float32,
        )
        component_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        active_mask = np.array([True, True, True, True, True, False])
        box_limits = np.full(len(positions), 100.0, dtype=np.float32)
        params = {
            "DT": 0.05,
            "DAMPING": 0.25,
            "COULOMB_K": 20.0,
            "COULOMB_CUTOFF": 5.0,
            "MAX_FORCE_LIMIT": 50.0,
            "MAX_TOTAL_REPULSION_FORCE": 10.0,
        }
        expected = brute_force_repulsion_step(
            positions,
            component_labels,
            active_mask,
            box_limits,
            params,
        )
        simulation = molecular_dynamics.SSNSimulationCPU(
            positions.copy(),
            np.zeros((0, 2), dtype=np.int32),
            component_labels,
            box_limits,
            params,
            active_mask=active_mask,
        )

        simulation.step(0)

        np.testing.assert_allclose(
            simulation.get_pos(), expected, rtol=0.0, atol=1e-6
        )

    @unittest.skipUnless(molecular_dynamics.HAS_TORCH, "PyTorch is unavailable")
    def test_gpu_all_pairs_forces_match_cpu(self):
        positions = np.array(
            [
                [-2.0, 0.0],
                [2.0, 0.0],
                [20.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [50.0, 50.0],
            ],
            dtype=np.float32,
        )
        component_labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        active_mask = np.array([True, True, True, True, True, False])
        params = {
            "DT": 0.05,
            "DAMPING": 0.25,
            "COULOMB_K": 20.0,
            "COULOMB_CUTOFF": 5.0,
            "MAX_FORCE_LIMIT": 50.0,
            "MAX_TOTAL_REPULSION_FORCE": 10.0,
        }
        no_springs = np.zeros((0, 2), dtype=np.int32)
        cpu_simulation = molecular_dynamics.SSNSimulationCPU(
            positions.copy(),
            no_springs,
            component_labels,
            100.0,
            params,
            active_mask=active_mask,
        )
        gpu_simulation = molecular_dynamics.SSNSimulationGPU(
            positions.copy(),
            no_springs,
            component_labels,
            100.0,
            params,
            active_mask=active_mask,
        )

        cpu_simulation.step(0)
        gpu_simulation.step(0)

        self.assertFalse(hasattr(gpu_simulation, "neighbor_pairs"))
        np.testing.assert_allclose(
            gpu_simulation.get_pos(),
            cpu_simulation.get_pos(),
            rtol=0.0,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
