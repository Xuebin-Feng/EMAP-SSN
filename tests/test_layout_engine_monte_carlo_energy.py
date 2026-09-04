import io
import math
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN_MolecularDynamics as molecular_dynamics
    import Layout_Engine_SSN_MonteCarlo as monte_carlo


def _force_magnitude(distance, k_coul, max_force, cutoff):
    if distance >= cutoff:
        return 0.0
    force = k_coul / max(distance, 0.5) ** 2
    if max_force > 0.0:
        force = min(force, max_force)
    taper_start = 0.8 * cutoff
    if distance > taper_start:
        force *= max(0.0, (cutoff - distance) / (0.2 * cutoff))
    return force


def _optimizer(positions, edges, **overrides):
    params = {
        "SPRING_K": 1.25,
        "COULOMB_K": 8.0,
        "COULOMB_CUTOFF": 10.0,
        "MAX_FORCE_LIMIT": 2.0,
        "MC_SWEEPS": 2,
        "MC_QUENCH_SWEEPS": 0,
        "MC_TELEPORT_PROBABILITY": 0.10,
    }
    params.update(overrides)
    return monte_carlo.ComponentEnergyMonteCarlo(
        np.asarray(positions, dtype=np.float64),
        np.asarray(edges, dtype=np.int32),
        100.0,
        params,
        np.random.default_rng(42),
    )


class RepulsivePotentialTests(unittest.TestCase):
    def test_analytic_potential_matches_numerical_integration_and_force(self):
        k_coul = 8.0
        max_force = 2.0
        cutoff = 10.0
        for distance in (0.25, 1.0, 2.5, 6.0, 8.5, 9.75):
            with self.subTest(distance=distance):
                analytic = monte_carlo.repulsive_pair_energy(
                    distance, k_coul, max_force, cutoff
                )
                grid = np.linspace(distance, cutoff, 100001)
                forces = np.asarray([
                    _force_magnitude(value, k_coul, max_force, cutoff)
                    for value in grid
                ])
                numerical = np.trapezoid(forces, grid)
                self.assertAlmostEqual(analytic, numerical, places=6)

                step = 1.0e-5
                derivative_force = -(
                    monte_carlo.repulsive_pair_energy(
                        distance + step, k_coul, max_force, cutoff
                    )
                    - monte_carlo.repulsive_pair_energy(
                        distance - step, k_coul, max_force, cutoff
                    )
                ) / (2.0 * step)
                self.assertAlmostEqual(
                    derivative_force,
                    _force_magnitude(distance, k_coul, max_force, cutoff),
                    places=5,
                )
        self.assertEqual(
            monte_carlo.repulsive_pair_energy(cutoff, k_coul, max_force, cutoff),
            0.0,
        )


class ComponentEnergyTests(unittest.TestCase):
    def test_incremental_delta_matches_full_recompute_across_cutoff(self):
        simulation = _optimizer(
            [[0.0, 0.0], [4.0, 0.0], [9.5, 1.0], [18.0, 0.0]],
            [[0, 1], [1, 2], [2, 3]],
        )
        for node, proposed in (
            (0, np.array([-3.0, 2.0])),
            (1, np.array([11.0, -1.0])),
            (2, np.array([7.0, 4.0])),
        ):
            with self.subTest(node=node, proposed=proposed.tolist()):
                candidate = simulation.pos.copy()
                candidate[node] = proposed
                incremental = simulation.current_energy + simulation.energy_delta(
                    node, proposed
                )
                exact = simulation.total_energy_for(candidate)
                self.assertAlmostEqual(incremental, exact, places=11)

    def test_pair_is_counted_once_and_spatial_storage_is_linear(self):
        positions = np.column_stack((np.arange(100, dtype=float), np.zeros(100)))
        simulation = _optimizer(
            positions,
            np.zeros((0, 2), dtype=np.int32),
            SPRING_K=0.0,
            COULOMB_CUTOFF=1.1,
        )
        expected = 99 * monte_carlo.repulsive_pair_energy(1.0, 8.0, 2.0, 1.1)
        self.assertAlmostEqual(simulation.current_energy, expected, places=10)
        self.assertEqual(
            sum(len(nodes) for nodes in simulation.spatial.cells.values()),
            len(positions),
        )
        self.assertEqual(len(simulation.spatial.node_cells), len(positions))

    def test_spatial_candidates_are_sorted_before_every_repulsive_sum(self):
        simulation = _optimizer(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            np.zeros((0, 2), dtype=np.int32),
        )
        real_sum = monte_carlo.repulsive_energy_sum
        candidate_orders = []

        def recording_sum(position, positions, candidate_indices, *args):
            candidate_orders.append(candidate_indices.tolist())
            return real_sum(position, positions, candidate_indices, *args)

        with mock.patch.object(
            monte_carlo, "repulsive_energy_sum", side_effect=recording_sum
        ):
            simulation.total_energy()
            simulation.energy_delta(1, np.asarray([1.5, 0.0]))

        self.assertTrue(candidate_orders)
        for order in candidate_orders:
            self.assertEqual(order, sorted(order))

    def test_spring_energy_is_a_uniform_sum_over_retained_edges(self):
        positions = np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]], dtype=np.float64
        )
        edges = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        simulation = _optimizer(
            positions,
            edges,
            SPRING_K=2.5,
            COULOMB_K=0.0,
            COULOMB_CUTOFF=0.0,
        )
        expected = 0.5 * 2.5 * (2.0**2 + 3.0**2)
        self.assertEqual(simulation.current_energy, expected)

    def test_accepted_and_rejected_moves_update_state_atomically(self):
        simulation = _optimizer(
            [[-15.0, 0.0], [0.0, 0.0]],
            [[0, 1]],
            COULOMB_K=0.0,
            COULOMB_CUTOFF=10.0,
        )
        old_energy = simulation.current_energy
        simulation._proposal = lambda node, local_only=False: (
            np.array([-5.0, 0.0]), 0.0, "local"
        )
        accepted, _ = simulation._attempt(0, 0.0)
        self.assertTrue(accepted)
        np.testing.assert_array_equal(simulation.pos[0], [-5.0, 0.0])
        self.assertLess(simulation.current_energy, old_energy)
        self.assertIn(0, simulation.spatial.candidates(simulation.pos[0]))

        accepted_position = simulation.pos.copy()
        accepted_energy = simulation.current_energy
        accepted_cells = [set(nodes) for nodes in simulation.spatial.cells.values()]
        simulation._proposal = lambda node, local_only=False: (
            np.array([101.0, 0.0]), 0.0, "uniform"
        )
        accepted, _ = simulation._attempt(0, 1.0)
        self.assertFalse(accepted)
        np.testing.assert_array_equal(simulation.pos, accepted_position)
        self.assertEqual(simulation.current_energy, accepted_energy)
        self.assertEqual(
            [set(nodes) for nodes in simulation.spatial.cells.values()],
            accepted_cells,
        )

    def test_neighbor_proposal_has_hastings_density_correction(self):
        class NeighborRng:
            def random(self):
                return 0.0

            def normal(self, mean, deviation, size):
                return np.zeros(size)

        simulation = _optimizer(
            [[8.0, 0.0], [0.0, 0.0]],
            [[0, 1]],
            MC_TELEPORT_PROBABILITY=1.0,
        )
        simulation.rng = NeighborRng()
        proposed, log_hastings, kind = simulation._proposal(0)
        np.testing.assert_array_equal(proposed, [0.0, 0.0])
        self.assertEqual(kind, "neighbor")
        expected = -0.5 * 8.0**2 / simulation.teleport_sigma**2
        self.assertAlmostEqual(log_hastings, expected, places=12)


class AcceptanceAndAnnealingTests(unittest.TestCase):
    class FixedRng:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    def _controlled_attempt(self, delta_energy, random_value, temperature, log_hastings=0.0):
        simulation = _optimizer(
            [[0.0, 0.0]],
            np.zeros((0, 2), dtype=np.int32),
            SPRING_K=0.0,
            COULOMB_K=0.0,
        )
        simulation.rng = self.FixedRng(random_value)
        simulation._proposal = lambda node, local_only=False: (
            np.array([1.0, 0.0]), log_hastings, "local"
        )
        simulation.energy_delta = lambda node, proposed: delta_energy
        return simulation._attempt(0, temperature)[0]

    def test_log_space_metropolis_and_zero_temperature_rules(self):
        self.assertTrue(self._controlled_attempt(-1.0, 0.999, 0.0))
        self.assertFalse(self._controlled_attempt(1.0, 0.001, 0.0))
        self.assertTrue(self._controlled_attempt(1.0, 0.1, 1.0))
        self.assertFalse(self._controlled_attempt(1.0, 0.5, 1.0))
        self.assertFalse(
            self._controlled_attempt(0.2, 0.5, 1.0, log_hastings=-1.0)
        )

    def test_best_state_is_returned_when_current_state_is_worse(self):
        simulation = _optimizer(
            [[-2.0, 0.0], [2.0, 0.0]],
            [[0, 1]],
            COULOMB_K=0.0,
            MC_SWEEPS=1,
            MC_QUENCH_SWEEPS=0,
        )
        initial = simulation.pos.copy()

        moved = False

        def scripted_attempt(node, temperature, local_only=False):
            nonlocal moved
            if not moved:
                simulation.pos[node, 0] += 10.0
                moved = True
            simulation.current_energy = simulation.total_energy()
            return True, "local"

        simulation._calibrate_temperature = lambda: 1.0
        simulation._attempt = scripted_attempt
        best, best_energy = simulation.optimize()
        np.testing.assert_allclose(best, initial)
        self.assertAlmostEqual(
            best_energy, simulation.total_energy_for(initial), places=12
        )
        self.assertGreater(simulation.current_energy, best_energy)

    def test_quench_is_downhill_only_and_stops_after_five_inactive_sweeps(self):
        simulation = _optimizer(
            [[0.0, 0.0]],
            np.zeros((0, 2), dtype=np.int32),
            SPRING_K=0.0,
            COULOMB_K=0.0,
            MC_SWEEPS=1,
            MC_QUENCH_SWEEPS=25,
        )
        simulation.optimize()
        self.assertEqual(simulation.quench_sweeps_completed, 5)
        self.assertTrue(all(
            later <= earlier + 1.0e-12
            for earlier, later in zip(
                simulation.quench_energy_history,
                simulation.quench_energy_history[1:],
            )
        ))

    def test_seed_42_repeats_positions_and_energy_history(self):
        positions = [[-4.0, 0.0], [0.0, 2.0], [4.0, 0.0]]
        edges = [[0, 1], [1, 2]]
        results = []
        for _ in range(2):
            simulation = _optimizer(
                positions,
                edges,
                MC_SWEEPS=12,
                MC_QUENCH_SWEEPS=3,
            )
            best, energy = simulation.optimize()
            results.append((best, energy, simulation.energy_history))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][2], results[1][2])

    def test_incremental_drift_and_stale_best_energy_are_reconciled_exactly(self):
        simulation = _optimizer(
            [[-2.0, 0.0], [2.0000003, 0.0]],
            [[0, 1]],
            COULOMB_K=0.0,
            MC_SWEEPS=1,
            MC_QUENCH_SWEEPS=0,
        )
        proposed = np.asarray([-1.123456789, 0.0])
        exact_delta = simulation.energy_delta(0, proposed)
        simulation._proposal = lambda node, local_only=False: (
            proposed.copy(), 0.0, "local"
        )
        simulation.energy_delta = lambda node, position: exact_delta + 0.125

        accepted, _ = simulation._attempt(0, 0.0)
        self.assertTrue(accepted)
        self.assertGreaterEqual(simulation.max_energy_drift, 0.125 - 1.0e-12)
        self.assertAlmostEqual(
            simulation.current_energy, simulation.total_energy(), places=12
        )
        np.testing.assert_array_equal(simulation.best_pos, simulation.pos)
        self.assertAlmostEqual(simulation.best_energy, simulation.total_energy(), places=12)

        simulation.best_energy -= 10.0
        simulation._reconcile_energy()
        self.assertAlmostEqual(
            simulation.best_energy,
            simulation.total_energy_for(simulation.best_pos),
            places=12,
        )

        simulation._calibrate_temperature = lambda: 0.0
        simulation._attempt = lambda node, temperature, local_only=False: (
            False, "local"
        )
        returned, reported_energy = simulation.optimize()
        self.assertEqual(returned.dtype, np.float32)
        np.testing.assert_array_equal(
            simulation.best_pos, returned.astype(np.float64)
        )
        self.assertEqual(
            reported_energy, simulation.total_energy_for(returned.astype(np.float64))
        )
        self.assertEqual(simulation.best_energy, reported_energy)


class EngineBehaviorTests(unittest.TestCase):
    def test_public_layout_is_repeatable_with_seed_42_and_float32(self):
        connectivity = np.asarray(
            [[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0]], dtype=np.float64
        )
        params = {
            "LAYOUT_DEVICE_SELECTION": "auto",
            "SPRING_K": 1.0,
            "COULOMB_K": 4.0,
            "COULOMB_CUTOFF": 10.0,
            "MAX_FORCE_LIMIT": 20.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
            "BOX_SCALE": 2.0,
            "PACKING_GRID_SIZE": 20.0,
            "PACKING_PADDING": 5.0,
            "PACKING_GEOMETRY": "Square",
            "MC_SWEEPS": 3,
            "MC_QUENCH_SWEEPS": 1,
            "MC_TELEPORT_PROBABILITY": 0.10,
            "MC_RANDOM_SEED": 42,
        }
        with redirect_stdout(io.StringIO()):
            first, first_limit = monte_carlo.calculate_layout(
                connectivity, 4, params
            )
            second, second_limit = monte_carlo.calculate_layout(
                connectivity, 4, params
            )
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_limit, second_limit)

    def test_explicit_none_seed_uses_fresh_entropy(self):
        real_seed_sequence = np.random.SeedSequence
        with mock.patch.object(
            monte_carlo.np.random,
            "SeedSequence",
            wraps=real_seed_sequence,
        ) as seed_sequence, redirect_stdout(io.StringIO()):
            positions, _ = monte_carlo.calculate_layout(
                np.zeros((0, 3), dtype=np.float64),
                0,
                {
                    "LAYOUT_DEVICE_SELECTION": "cpu",
                    "MC_RANDOM_SEED": None,
                    "MC_SWEEPS": 1,
                    "MC_QUENCH_SWEEPS": 0,
                },
            )
        seed_sequence.assert_called_once_with(None)
        self.assertEqual(positions.shape, (0, 2))
        first = real_seed_sequence(None)
        second = real_seed_sequence(None)
        self.assertNotEqual(first.entropy, second.entropy)

    def test_balanced_md_node_can_cross_barrier_via_lower_energy_teleport(self):
        positions = np.array(
            [[10.0, 0.0], [-10.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        params = {
            "DT": 1.0,
            "DAMPING": 0.0,
            "SPRING_K": 0.0625,
            "COULOMB_K": 100.0,
            "COULOMB_CUTOFF": 30.0,
            "MAX_FORCE_LIMIT": 1000.0,
            "MAX_TOTAL_REPULSION_FORCE": 0.0,
        }
        md = molecular_dynamics.SSNSimulationCPU(
            positions.copy(), edges, np.zeros(3, dtype=np.int32), 100.0, params
        )
        md.step(0)
        self.assertAlmostEqual(float(md.get_pos()[0, 0]), 10.0, places=6)

        mc = monte_carlo.ComponentEnergyMonteCarlo(
            positions, edges, 100.0, params, np.random.default_rng(42)
        )
        old_energy = mc.current_energy
        mc._proposal = lambda node, local_only=False: (
            np.array([-23.0, 0.0]), 0.0, "neighbor"
        )
        accepted, kind = mc._attempt(0, 0.0)
        self.assertEqual(kind, "neighbor")
        self.assertTrue(accepted)
        self.assertLess(mc.current_energy, old_energy)
        self.assertLess(mc.pos[0, 0], mc.pos[2, 0])

    def test_cpu_only_and_energy_model_validation(self):
        base = {
            "MC_SWEEPS": 1,
            "MC_QUENCH_SWEEPS": 0,
            "MC_TELEPORT_PROBABILITY": 0.10,
            "MC_RANDOM_SEED": 42,
        }
        for selection in ("auto", "cpu"):
            monte_carlo._validate_energy_monte_carlo_params(
                dict(base, LAYOUT_DEVICE_SELECTION=selection)
            )
        with self.assertRaisesRegex(ValueError, "requires CPU"):
            monte_carlo._validate_energy_monte_carlo_params(
                dict(base, LAYOUT_DEVICE_SELECTION="cuda:0")
            )
        with self.assertRaisesRegex(ValueError, "MAX_TOTAL_REPULSION_FORCE"):
            monte_carlo._validate_energy_monte_carlo_params(
                dict(base, MAX_TOTAL_REPULSION_FORCE=1.0)
            )
        with self.assertRaisesRegex(ValueError, "MAX_FORCE_LIMIT"):
            monte_carlo._validate_energy_monte_carlo_params(
                dict(base, MAX_FORCE_LIMIT=-1.0)
            )
        with self.assertRaisesRegex(ValueError, "Progressive simulation"):
            monte_carlo._validate_energy_monte_carlo_params(
                dict(base, ENABLE_PROGRESSIVE_SIMULATION=True)
            )

    def test_parameter_validation_rejects_nonfinite_and_invalid_ranges(self):
        base = {
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "MC_SWEEPS": 1,
            "MC_QUENCH_SWEEPS": 0,
            "MC_TELEPORT_PROBABILITY": 0.10,
            "MC_RANDOM_SEED": 42,
        }
        invalid_cases = (
            ("SPRING_K", float("nan")),
            ("COULOMB_K", -1.0),
            ("COULOMB_CUTOFF", float("inf")),
            ("MAX_FORCE_LIMIT", -1.0),
            ("PACKING_PADDING", -1.0),
            ("BOX_SCALE", 0.0),
            ("PACKING_GRID_SIZE", 0.0),
            ("MC_SWEEPS", 1.0),
            ("MC_SWEEPS", 1_000_001),
            ("MC_QUENCH_SWEEPS", -1),
            ("MC_TELEPORT_PROBABILITY", 1.01),
            ("MC_RANDOM_SEED", -1),
        )
        for name, value in invalid_cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                monte_carlo._validate_energy_monte_carlo_params(
                    dict(base, **{name: value})
                )

    def test_layout_input_validation_rejects_malformed_connectivity(self):
        params = {
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "MC_SWEEPS": 1,
            "MC_QUENCH_SWEEPS": 0,
        }
        invalid_cases = (
            (np.zeros((1, 2)), 2),
            (np.asarray([[0.5, 1.0, 1.0]]), 2),
            (np.asarray([[0.0, 2.0, 1.0]]), 2),
            (np.asarray([[0.0, 1.0, np.nan]]), 2),
            (np.asarray([[0, 1, "score"]]), 2),
            (np.asarray([[0 + 1j, 1, 1]]), 2),
            (np.zeros((0, 3)), -1),
            (np.zeros((0, 3)), 2.0),
        )
        for connectivity, node_count in invalid_cases:
            with self.subTest(
                shape=connectivity.shape, node_count=node_count
            ), self.assertRaises(ValueError):
                monte_carlo.calculate_layout(connectivity, node_count, params)

    def test_component_input_validation_rejects_invalid_initial_state(self):
        params = {
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "MC_SWEEPS": 1,
            "MC_QUENCH_SWEEPS": 0,
        }
        cases = (
            (np.asarray([0.0, 0.0]), np.zeros((0, 2)), 10.0),
            (np.asarray([[np.nan, 0.0]]), np.zeros((0, 2)), 10.0),
            (np.asarray([[0 + 1j, 0.0]]), np.zeros((0, 2)), 10.0),
            (np.asarray([[11.0, 0.0]]), np.zeros((0, 2)), 10.0),
            (np.asarray([[0.0, 0.0]]), np.asarray([[0.0, 0.5]]), 10.0),
            (np.asarray([[0.0, 0.0]]), np.asarray([[0, 1]]), 10.0),
            (np.asarray([[0.0, 0.0]]), np.zeros((0, 2)), 0.0),
        )
        for positions, edges, box_limit in cases:
            with self.subTest(
                positions_shape=positions.shape, edges_shape=edges.shape
            ), self.assertRaises(ValueError):
                monte_carlo.ComponentEnergyMonteCarlo(
                    positions,
                    edges,
                    box_limit,
                    params,
                    np.random.default_rng(42),
                )


if __name__ == "__main__":
    unittest.main()
