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
    import Layout_Engine_SSN_MonteCarlo as monte_carlo


BASE_LAYOUT_PARAMS = {
    "MAX_STEPS": 1,
    "BOX_SCALE": 1.0,
    "SIMILARITY_THRESHOLD": 0.0,
    "PACKING_GRID_SIZE": 200.0,
    "PACKING_PADDING": 50.0,
}


class SegmentedBatchTests(unittest.TestCase):
    def _run_md_repulsive_pair(
        self,
        distance,
        coulomb_k,
        total_cap=0.0,
        pair_cap=1000.0,
        simulation_class=molecular_dynamics.SSNSimulationCPU,
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

    def _run_mc_repulsive_pair(
        self,
        distance,
        coulomb_k,
        total_cap=0.0,
        pair_cap=1000.0,
        simulation_class=monte_carlo.SSNSimulationCPU,
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
                "SGLD_K": 1,
                "SGLD_START_TEMP": 0.0,
                "MAX_STEPS": 1,
            },
        )
        before = simulation.get_pos().copy()
        simulation.step(0)
        return np.abs(simulation.get_pos()[:, 0] - before[:, 0])

    def test_molecular_dynamics_batch_matches_individual_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 15.0,
        }

        individual = molecular_dynamics.SSNSimulationCPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = molecular_dynamics.SSNSimulationCPU(
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

    @unittest.skipUnless(molecular_dynamics.HAS_TORCH, "PyTorch is unavailable")
    def test_molecular_dynamics_torch_batch_matches_individual_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 15.0,
        }

        individual = molecular_dynamics.SSNSimulationGPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = molecular_dynamics.SSNSimulationGPU(
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

    def test_molecular_dynamics_two_node_layout_does_not_collapse(self):
        connectivity = np.array([[0, 1, 1.0]], dtype=np.float64)

        with redirect_stdout(io.StringIO()):
            positions, box_limit = molecular_dynamics.calculate_layout(
                connectivity,
                2,
                dict(BASE_LAYOUT_PARAMS),
            )

        self.assertGreater(np.linalg.norm(positions[0] - positions[1]), 1.0)
        self.assertGreater(box_limit, 0.0)

    def test_monte_carlo_batch_matches_individual_two_node_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 30.0,
            "SGLD_K": 1,
            "SGLD_START_TEMP": 0.0,
            "MAX_STEPS": 1,
        }

        individual = monte_carlo.SSNSimulationCPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = monte_carlo.SSNSimulationCPU(
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

    @unittest.skipUnless(monte_carlo.HAS_TORCH, "PyTorch is unavailable")
    def test_monte_carlo_torch_batch_matches_individual_two_node_components(self):
        pair = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        batch = np.vstack((pair, pair))
        no_springs = np.zeros((0, 2), dtype=np.int32)
        params = {
            "DT": 0.1,
            "DAMPING": 0.5,
            "COULOMB_K": 50.0,
            "COULOMB_CUTOFF": 30.0,
            "SGLD_K": 1,
            "SGLD_START_TEMP": 0.0,
            "MAX_STEPS": 1,
        }

        individual = monte_carlo.SSNSimulationGPU(
            pair.copy(),
            no_springs,
            np.array([0, 0], dtype=np.int32),
            100.0,
            params,
        )
        batched = monte_carlo.SSNSimulationGPU(
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

    def test_monte_carlo_two_node_layout_does_not_collapse(self):
        connectivity = np.array([[0, 1, 1.0]], dtype=np.float64)
        params = dict(BASE_LAYOUT_PARAMS)
        params["SGLD_START_TEMP"] = 0.0

        with redirect_stdout(io.StringIO()):
            positions, box_limit = monte_carlo.calculate_layout(
                connectivity,
                2,
                params,
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
        simulation = molecular_dynamics.SSNSimulationCPU(
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
        simulation = molecular_dynamics.SSNSimulationCPU(
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
        md_displacement = self._run_md_repulsive_pair(
            distance=1.0,
            coulomb_k=25.0,
            pair_cap=0.0,
        )
        mc_displacement = self._run_mc_repulsive_pair(
            distance=1.0,
            coulomb_k=25.0,
            pair_cap=0.0,
        )

        np.testing.assert_allclose(md_displacement, [25.0, 25.0], atol=1e-5)
        np.testing.assert_allclose(mc_displacement, [25.0, 25.0], atol=1e-5)

    def test_monte_carlo_uses_same_taper_and_total_repulsion_cap(self):
        tapered = self._run_mc_repulsive_pair(13.5, 13.5**2)
        capped = self._run_mc_repulsive_pair(1.0, 100.0, total_cap=3.0)

        np.testing.assert_allclose(tapered, [0.5, 0.5], atol=1e-5)
        np.testing.assert_allclose(capped, [3.0, 3.0], atol=1e-5)

    @unittest.skipUnless(molecular_dynamics.HAS_TORCH, "PyTorch is unavailable")
    def test_torch_force_taper_and_caps_match_cpu(self):
        md_tapered = self._run_md_repulsive_pair(
            13.5,
            13.5**2,
            simulation_class=molecular_dynamics.SSNSimulationGPU,
        )
        md_capped = self._run_md_repulsive_pair(
            1.0,
            100.0,
            total_cap=3.0,
            simulation_class=molecular_dynamics.SSNSimulationGPU,
        )
        mc_tapered = self._run_mc_repulsive_pair(
            13.5,
            13.5**2,
            simulation_class=monte_carlo.SSNSimulationGPU,
        )
        mc_capped = self._run_mc_repulsive_pair(
            1.0,
            100.0,
            total_cap=3.0,
            simulation_class=monte_carlo.SSNSimulationGPU,
        )
        md_pair_uncapped = self._run_md_repulsive_pair(
            1.0,
            25.0,
            pair_cap=0.0,
            simulation_class=molecular_dynamics.SSNSimulationGPU,
        )
        mc_pair_uncapped = self._run_mc_repulsive_pair(
            1.0,
            25.0,
            pair_cap=0.0,
            simulation_class=monte_carlo.SSNSimulationGPU,
        )

        np.testing.assert_allclose(md_tapered, [0.5, 0.5], atol=1e-5)
        np.testing.assert_allclose(md_capped, [3.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(mc_tapered, [0.5, 0.5], atol=1e-5)
        np.testing.assert_allclose(mc_capped, [3.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(md_pair_uncapped, [25.0, 25.0], atol=1e-5)
        np.testing.assert_allclose(mc_pair_uncapped, [25.0, 25.0], atol=1e-5)

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
        individual = molecular_dynamics.SSNSimulationCPU(
            pair.copy(), no_springs, np.array([0, 0]), 100.0, params
        )
        staged = molecular_dynamics.SSNSimulationCPU(
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

        active = molecular_dynamics._prepare_progressive_stage(
            positions,
            stage_edges,
            stage_scores,
            previous_active,
        )

        np.testing.assert_array_equal(active, [True, True, True])
        self.assertLess(np.linalg.norm(positions[2] - positions[1]), 0.25)


if __name__ == "__main__":
    unittest.main()
