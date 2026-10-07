"""Layout_Engine_SSN: the SSNSimulationCPU/GPU physics step (all-pairs Coulomb
repulsion against an independent reference, the cutoff taper, the pair and
total force caps, per-component boundaries), component batching, progressive
placement, CPU/torch/CUDA agreement, and LAYOUT_SEED determinism of
calculate_layout."""
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
    import Layout_Engine_SSN as ssn_engine


CUDA_AVAILABLE = (
    ssn_engine.HAS_TORCH
    and ssn_engine.torch.cuda.is_available()
)


# CPU explicitly: the default "auto" benchmarks every device (and initialises
# CUDA) before the first layout.
BASE_LAYOUT_PARAMS = {
    "MAX_STEPS": 1,
    "BOX_SCALE": 1.0,
    "SIMILARITY_THRESHOLD": 0.0,
    "PACKING_GRID_SIZE": 200.0,
    "PACKING_PADDING": 50.0,
    "LAYOUT_DEVICE_SELECTION": "cpu",
}


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


class SegmentedBatchTests(unittest.TestCase):
    def _run_md_repulsive_pair(
        self,
        distance,
        coulomb_k,
        total_cap=0.0,
        pair_cap=1000.0,
        simulation_class=ssn_engine.SSNSimulationCPU,
    ):
        positions = np.array(
            [[-distance / 2.0, 0.0], [distance / 2.0, 0.0]],
            dtype=np.float32,
        )
        simulation = simulation_class(
            positions,
            np.zeros((0, 2), dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            100.0,
            {
                "DT": 1.0,
                "DAMPING": 0.0,
                "COULOMB_K": coulomb_k,
                "COULOMB_CUTOFF": 15.0,
                "MAX_FORCE_LIMIT": pair_cap,
                "MAX_TOTAL_REPULSION_FORCE": total_cap,
            },
        )
        before = simulation.get_pos().copy()
        simulation.step(0)
        return np.abs(simulation.get_pos()[:, 0] - before[:, 0])

    def test_ssn_engine_batch_matches_individual_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 15.0,
        }

        individual = ssn_engine.SSNSimulationCPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = ssn_engine.SSNSimulationCPU(
            batch,
            no_springs,
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.full(4, 100.0, dtype=np.float32),
            params,
        )

        individual.step(0)
        batched.step(0)

        expected = np.vstack((individual.get_pos(), individual.get_pos()))
        np.testing.assert_allclose(batched.get_pos(), expected, rtol=0.0, atol=1e-6)

    @unittest.skipUnless(ssn_engine.HAS_TORCH, "PyTorch is unavailable")
    def test_ssn_engine_torch_batch_matches_individual_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 15.0,
        }

        individual = ssn_engine.SSNSimulationGPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = ssn_engine.SSNSimulationGPU(
            batch,
            no_springs,
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.full(4, 100.0, dtype=np.float32),
            params,
        )

        individual.step(0)
        batched.step(0)

        expected = np.vstack((individual.get_pos(), individual.get_pos()))
        np.testing.assert_allclose(batched.get_pos(), expected, rtol=0.0, atol=1e-6)

    def test_ssn_engine_two_node_layout_does_not_collapse(self):
        connectivity = np.array([[0, 1, 1.0]], dtype=np.float64)

        with redirect_stdout(io.StringIO()):
            positions, box_limit = ssn_engine.calculate_layout(
                connectivity,
                2,
                dict(BASE_LAYOUT_PARAMS),
            )

        self.assertGreater(np.linalg.norm(positions[0] - positions[1]), 1.0)
        self.assertGreater(box_limit, 0.0)

    def test_component_specific_boundaries_are_applied(self):
        positions = np.array(
            [
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        simulation = ssn_engine.SSNSimulationCPU(
            positions,
            np.zeros((0, 2), dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([1.0, 10.0], dtype=np.float32),
            {"DT": 0.1, "DAMPING": 0.5, "COULOMB_K": 0.0},
        )

        simulation.step(0)

        self.assertEqual(simulation.get_pos()[0, 0], 1.0)
        self.assertEqual(simulation.get_pos()[1, 0], 0.0)

    def test_full_cutoff_is_active_from_first_step(self):
        positions = np.array([[-7.0, 0.0], [7.0, 0.0]], dtype=np.float32)
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 1.0,
            "DAMPING": 0.0,
            "COULOMB_K": 196.0,
            "COULOMB_CUTOFF": 15.0,
            "MAX_FORCE_LIMIT": 1000.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        simulation = ssn_engine.SSNSimulationCPU(
            positions.copy(), no_springs, np.array([0, 0]), 100.0, params
        )
        simulation.step(0)

        self.assertGreater(abs(simulation.get_pos()[0, 0]), 7.0)

    def test_coulomb_force_tapers_linearly_over_outer_twenty_percent(self):
        at_taper_start = self._run_md_repulsive_pair(12.0, 12.0**2)
        halfway_through_taper = self._run_md_repulsive_pair(13.5, 13.5**2)
        at_cutoff = self._run_md_repulsive_pair(15.0, 15.0**2)

        np.testing.assert_allclose(at_taper_start, [1.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(
            halfway_through_taper, [0.5, 0.5], atol=1e-5
        )
        np.testing.assert_allclose(at_cutoff, [0.0, 0.0], atol=1e-5)

    def test_accumulated_repulsion_is_capped_before_integration(self):
        displacement = self._run_md_repulsive_pair(
            distance=1.0,
            coulomb_k=100.0,
            total_cap=3.0,
        )

        np.testing.assert_allclose(displacement, [3.0, 3.0], atol=1e-5)

    def test_zero_pair_cap_disables_cap_without_disabling_repulsion(self):
        displacement = self._run_md_repulsive_pair(
            distance=1.0,
            coulomb_k=25.0,
            pair_cap=0.0,
        )
        np.testing.assert_allclose(displacement, [25.0, 25.0], atol=1e-5)

    @unittest.skipUnless(ssn_engine.HAS_TORCH, "PyTorch is unavailable")
    def test_torch_force_taper_and_caps_match_cpu(self):
        md_tapered = self._run_md_repulsive_pair(
            13.5,
            13.5**2,
            simulation_class=ssn_engine.SSNSimulationGPU,
        )
        md_capped = self._run_md_repulsive_pair(
            1.0,
            100.0,
            total_cap=3.0,
            simulation_class=ssn_engine.SSNSimulationGPU,
        )
        md_pair_uncapped = self._run_md_repulsive_pair(
            1.0,
            25.0,
            pair_cap=0.0,
            simulation_class=ssn_engine.SSNSimulationGPU,
        )
        np.testing.assert_allclose(md_tapered, [0.5, 0.5], atol=1e-5)
        np.testing.assert_allclose(md_capped, [3.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(md_pair_uncapped, [25.0, 25.0], atol=1e-5)

    def test_inactive_progressive_nodes_are_frozen_and_nonrepulsive(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        with_inactive = np.vstack((pair, np.array([[0.0, 0.0]], dtype=np.float32)))
        params = {
            "DT": 0.1,
            "DAMPING": 0.0,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 15.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        no_springs = np.zeros((0, 2), dtype=np.int32)
        individual = ssn_engine.SSNSimulationCPU(
            pair.copy(), no_springs, np.array([0, 0]), 100.0, params
        )
        staged = ssn_engine.SSNSimulationCPU(
            with_inactive,
            no_springs,
            np.array([0, 0, 0]),
            100.0,
            params,
            active_mask=np.array([True, True, False]),
        )

        individual.step(0)
        staged.step(0)

        np.testing.assert_allclose(
            staged.get_pos()[:2], individual.get_pos(), atol=1e-6
        )
        np.testing.assert_allclose(staged.get_pos()[2], [0.0, 0.0], atol=0.0)

    def test_new_progressive_node_is_placed_near_existing_neighbor(self):
        positions = np.array(
            [[-1.0, 0.0], [1.0, 0.0], [50.0, 50.0]],
            dtype=np.float32,
        )
        previous_active = np.array([True, True, False])
        stage_edges = [(0, 1), (1, 2)]
        stage_scores = [1.0, 0.8]

        active = ssn_engine._prepare_progressive_stage(
            positions,
            stage_edges,
            stage_scores,
            previous_active,
        )

        np.testing.assert_array_equal(active, [True, True, True])
        self.assertLess(np.linalg.norm(positions[2] - positions[1]), 0.25)


class SSNLayoutPairwiseTests(unittest.TestCase):
    def _assert_torch_matches_cpu(
        self,
        positions,
        component_labels,
        params,
        *,
        active_mask=None,
        box_limits=100.0,
        device=None,
        atol=1e-5,
    ):
        no_springs = np.zeros((0, 2), dtype=np.int32)
        cpu_simulation = ssn_engine.SSNSimulationCPU(
            positions.copy(),
            no_springs,
            component_labels,
            box_limits,
            params,
            active_mask=active_mask,
        )
        torch_simulation = ssn_engine.SSNSimulationGPU(
            positions.copy(),
            no_springs,
            component_labels,
            box_limits,
            params,
            active_mask=active_mask,
            device=device,
        )

        cpu_rmsd = cpu_simulation.step(0)
        torch_rmsd = torch_simulation.step(0)

        np.testing.assert_allclose(
            torch_simulation.get_pos(),
            cpu_simulation.get_pos(),
            rtol=0.0,
            atol=atol,
        )
        self.assertAlmostEqual(cpu_rmsd, torch_rmsd, delta=atol)
        return torch_simulation

    def _assert_all_pairs_forces_match_cpu(self, device=None):
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

        self._assert_torch_matches_cpu(
            positions,
            component_labels,
            params,
            active_mask=active_mask,
            device=device,
        )

    def _assert_near_coincident_pair_matches_cpu(self, device=None):
        positions = np.array(
            [[0.0, 0.0], [1e-12, 0.0]],
            dtype=np.float32,
        )
        params = {
            "DT": 1.0,
            "DAMPING": 0.0,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 5.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        self._assert_torch_matches_cpu(
            positions,
            np.array([0, 0], dtype=np.int32),
            params,
            device=device,
        )

    def _assert_coincident_pair_matches_cpu(self, device=None):
        positions = np.zeros((2, 2), dtype=np.float32)
        params = {
            "DT": 1.0,
            "DAMPING": 0.0,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 5.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        simulation = self._assert_torch_matches_cpu(
            positions,
            np.array([0, 0], dtype=np.int32),
            params,
            device=device,
        )
        self.assertTrue(np.isfinite(simulation.get_pos()).all())

    def test_pair_moved_inside_cutoff_is_used_on_next_step(self):
        positions = np.array([[-6.1, 0.0], [6.1, 0.0]], dtype=np.float32)
        simulation = ssn_engine.SSNSimulationCPU(
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
        simulation = ssn_engine.SSNSimulationCPU(
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

    @unittest.skipUnless(ssn_engine.HAS_TORCH, "PyTorch is unavailable")
    def test_torch_all_pairs_forces_match_cpu(self):
        self._assert_all_pairs_forces_match_cpu()

    @unittest.skipUnless(ssn_engine.HAS_TORCH, "PyTorch is unavailable")
    def test_torch_near_coincident_pair_matches_cpu(self):
        self._assert_near_coincident_pair_matches_cpu()

    @unittest.skipUnless(ssn_engine.HAS_TORCH, "PyTorch is unavailable")
    def test_torch_coincident_pair_matches_cpu(self):
        self._assert_coincident_pair_matches_cpu()

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cuda_all_pairs_forces_match_cpu(self):
        self._assert_all_pairs_forces_match_cpu(
            device=ssn_engine.torch.device("cuda")
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cuda_near_coincident_pair_matches_cpu(self):
        self._assert_near_coincident_pair_matches_cpu(
            device=ssn_engine.torch.device("cuda")
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cuda_coincident_pair_matches_cpu(self):
        self._assert_coincident_pair_matches_cpu(
            device=ssn_engine.torch.device("cuda")
        )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
    def test_cuda_boundary_collision_matches_cpu(self):
        positions = np.array([[0.9, 0.0], [-0.9, 0.0]], dtype=np.float32)
        velocities = np.array([[1.0, 2.0], [-1.0, -2.0]], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int32)
        params = {
            "DT": 0.2,
            "DAMPING": 0.0,
            "COULOMB_K": 0.0,
            "COULOMB_CUTOFF": 5.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        no_springs = np.zeros((0, 2), dtype=np.int32)
        cpu_simulation = ssn_engine.SSNSimulationCPU(
            positions.copy(), no_springs, labels, 1.0, params
        )
        cuda_simulation = ssn_engine.SSNSimulationGPU(
            positions.copy(),
            no_springs,
            labels,
            1.0,
            params,
            device=ssn_engine.torch.device("cuda"),
        )
        cpu_simulation.vel[:] = velocities
        cuda_simulation.vel.copy_(
            ssn_engine.torch.tensor(
                velocities,
                dtype=ssn_engine.torch.float32,
                device=cuda_simulation.device,
            )
        )

        cpu_rmsd = cpu_simulation.step(0)
        cuda_rmsd = cuda_simulation.step(0)

        np.testing.assert_allclose(
            cuda_simulation.get_pos(),
            cpu_simulation.get_pos(),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            cuda_simulation.vel.cpu().numpy(),
            cpu_simulation.vel,
            rtol=0.0,
            atol=1e-6,
        )
        self.assertAlmostEqual(cpu_rmsd, cuda_rmsd, delta=1e-6)


class DeterministicLayoutTests(unittest.TestCase):
    """LAYOUT_SEED must make calculate_layout reproducible run to run."""

    NODES = 24

    def _connectivity(self):
        edges = []
        for base, size in ((0, 12), (12, 12)):
            for offset in range(size):
                edges.append((base + offset, base + (offset + 1) % size))
        return np.array([(u, v, 1.0) for u, v in edges], dtype=np.float64)

    def _layout(self, *, seed=0, dimensions=2):
        params = {
            **BASE_LAYOUT_PARAMS,
            "MAX_STEPS": 25,
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "LAYOUT_SEED": seed,
        }
        if dimensions != 2:
            params["LAYOUT_DIMENSIONS"] = dimensions
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            positions, _ = ssn_engine.calculate_layout(
                self._connectivity(), self.NODES, params
            )
        return np.asarray(positions)

    def test_fixed_seed_is_reproducible_in_2d(self):
        first = self._layout(seed=0)
        second = self._layout(seed=0)
        self.assertEqual(first.shape, (self.NODES, 2))
        np.testing.assert_array_equal(first, second)

    def test_fixed_seed_is_reproducible_in_3d(self):
        first = self._layout(seed=0, dimensions=3)
        second = self._layout(seed=0, dimensions=3)
        self.assertEqual(first.shape, (self.NODES, 3))
        np.testing.assert_array_equal(first, second)

    def test_distinct_seeds_produce_distinct_layouts(self):
        self.assertFalse(
            np.array_equal(self._layout(seed=0), self._layout(seed=7))
        )

    def test_null_seed_opts_out_of_reproducibility(self):
        first = self._layout(seed=None)
        second = self._layout(seed=None)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.array_equal(first, second))

    def test_negative_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            self._layout(seed=-1)

    def test_three_dimensional_kernel_reduces_to_the_two_dimensional_one(self):
        """With z pinned to zero the 3D kernel must match the 2D kernel."""
        rng = np.random.default_rng(7)
        count = 16
        positions_2d = (rng.random((count, 2)).astype(np.float32) - 0.5) * 20.0
        positions_3d = np.zeros((count, 3), dtype=np.float32)
        positions_3d[:, :2] = positions_2d
        velocity_2d = np.zeros((count, 2), dtype=np.float32)
        velocity_3d = np.zeros((count, 3), dtype=np.float32)
        springs = np.array(
            [(index, (index * 3 + 1) % count) for index in range(count)],
            dtype=np.int32,
        )
        labels = np.zeros(count, dtype=np.int32)
        mask = np.ones(count, dtype=np.bool_)
        active = np.flatnonzero(mask).astype(np.int32)
        limits = np.full(count, 20.0, dtype=np.float32)
        arguments = (0.005, 0.9, 5.0, 10.0, 20.0, 0.0, 30.0)

        kernel_2d = ssn_engine._get_physics_kernel(2)
        kernel_3d = ssn_engine._get_physics_kernel(3)
        for _ in range(10):
            rmsd_2d = kernel_2d(
                positions_2d, velocity_2d, springs, labels, mask, active,
                limits, *arguments,
            )
            rmsd_3d = kernel_3d(
                positions_3d, velocity_3d, springs, labels, mask, active,
                limits, *arguments,
            )

        np.testing.assert_array_equal(positions_3d[:, :2], positions_2d)
        np.testing.assert_array_equal(
            positions_3d[:, 2], np.zeros(count, dtype=np.float32)
        )
        self.assertEqual(rmsd_2d, rmsd_3d)


if __name__ == "__main__":
    unittest.main()
