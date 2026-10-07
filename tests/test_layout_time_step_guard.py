"""Time-step guard, automatic time step and stage-end reports of the SSN layout engine."""
import ast
import io
import math
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN as ssn_engine


CPU = SimpleNamespace(is_cpu=True, type="cpu")


def complete_graph(n):
    i, j = np.triu_indices(n, 1)
    return np.column_stack((i, j)).astype(np.int32)


def complete_bipartite_graph(a):
    i, j = np.meshgrid(np.arange(a), np.arange(a, 2 * a), indexing="ij")
    return np.column_stack((i.ravel(), j.ravel())).astype(np.int32)


def star_graph(leaves):
    return np.array([(0, leaf) for leaf in range(1, leaves + 1)], dtype=np.int32)


def largest_laplacian_eigenvalue(n, edges):
    laplacian = np.zeros((n, n))
    for u, v in edges:
        if u != v:
            laplacian[u, u] += 1.0
            laplacian[v, v] += 1.0
            laplacian[u, v] -= 1.0
            laplacian[v, u] -= 1.0
    return float(np.linalg.eigvalsh(laplacian)[-1])


class SpringDegreeBoundTests(unittest.TestCase):
    def test_bound_lies_between_the_largest_eigenvalue_and_twice_it(self):
        graphs = {
            "path": (10, np.array([(k, k + 1) for k in range(9)], dtype=np.int32)),
            "star": (10, star_graph(9)),
            "cycle": (12, np.array([(k, (k + 1) % 12) for k in range(12)], dtype=np.int32)),
            "K_8": (8, complete_graph(8)),
            "K_4,4": (8, complete_bipartite_graph(4)),
        }
        for seed in range(5):
            pairs = complete_graph(30)
            keep = np.random.default_rng(seed).random(len(pairs)) < 0.3
            graphs[f"random_{seed}"] = (30, pairs[keep])
        for name, (n, edges) in graphs.items():
            with self.subTest(graph=name):
                bound = ssn_engine._spring_degree_bound(n, edges)
                eigenvalue = largest_laplacian_eigenvalue(n, edges)
                self.assertGreaterEqual(bound, eigenvalue - 1e-9)
                self.assertLessEqual(bound, 2.0 * eigenvalue + 1e-9)

    def test_bound_is_exact_for_stars_and_complete_bipartite_graphs(self):
        self.assertEqual(ssn_engine._spring_degree_bound(10, star_graph(9)), 10)
        self.assertEqual(
            ssn_engine._spring_degree_bound(8, complete_bipartite_graph(4)), 8
        )
        # A clique's bound, 2(n - 1), is nearly twice its eigenvalue n.
        self.assertEqual(ssn_engine._spring_degree_bound(8, complete_graph(8)), 14)

    def test_self_loops_are_ignored_and_parallel_springs_count(self):
        clique = complete_graph(4)
        with_loop = np.vstack((clique, [[2, 2]])).astype(np.int32)
        self.assertEqual(ssn_engine._spring_degree_bound(4, with_loop), 6)
        doubled = np.vstack((clique, [[0, 1]])).astype(np.int32)
        self.assertEqual(ssn_engine._spring_degree_bound(4, doubled), 8)

    def test_no_springs_give_zero(self):
        self.assertEqual(
            ssn_engine._spring_degree_bound(5, np.zeros((0, 2), dtype=np.int32)), 0
        )


class StageParamsTests(unittest.TestCase):
    PHYSICS = {
        "SPRING_K": 5.0,
        "DAMPING": 0.9,
        "MAX_STEPS": 1000,
        "RMSD_WINDOW": 50,
        "RMSD_THRESHOLD": 0.005,
    }

    def _stage(self, n, edges, params):
        output = io.StringIO()
        with redirect_stdout(output):
            stage = ssn_engine._stage_params(n, edges, params)
        return stage, output.getvalue()

    def test_safe_stage_keeps_the_same_params_object(self):
        params = {**self.PHYSICS, "DT": 0.005}
        stage, log = self._stage(100, complete_graph(100), params)
        self.assertIs(stage, params)
        self.assertEqual(log, "")

    def test_unsafe_stage_lowers_dt_and_keeps_its_simulated_time(self):
        params = {**self.PHYSICS, "DT": 0.1}
        original = dict(params)
        stage, log = self._stage(100, complete_graph(100), params)
        limit = ssn_engine._euler_dt_limit(5.0 * 198, 0.9)
        expected_dt = float(f"{0.85 * limit:.3g}")
        scale = 0.1 / expected_dt
        self.assertEqual(stage["DT"], expected_dt)
        self.assertEqual(stage["MAX_STEPS"], math.ceil(1000 * scale))
        self.assertEqual(stage["RMSD_WINDOW"], math.ceil(50 * scale))
        self.assertAlmostEqual(stage["RMSD_THRESHOLD"], 0.005 / scale)
        self.assertEqual(params, original)
        self.assertIn(f"DT 0.1 -> {expected_dt:g}", log)

    def test_lowered_dt_stays_inside_the_true_stability_limit(self):
        # Both graphs have largest eigenvalue 200. The bound is 398 for the
        # clique but exactly 200 for the complete bipartite graph.
        for name, edges in (
            ("K_200", complete_graph(200)),
            ("K_100,100", complete_bipartite_graph(100)),
        ):
            with self.subTest(graph=name):
                stage, _ = self._stage(200, edges, {**self.PHYSICS, "DT": 1.0})
                fraction = stage["DT"] / ssn_engine._euler_dt_limit(5.0 * 200, 0.9)
                self.assertGreater(fraction, 0.55)
                self.assertLess(fraction, 0.86)

    def test_without_spring_stiffness_params_are_unchanged(self):
        cases = (
            ({**self.PHYSICS, "DT": 1.0, "SPRING_K": 0.0}, complete_graph(50)),
            ({**self.PHYSICS, "DT": 1.0}, np.zeros((0, 2), dtype=np.int32)),
        )
        for params, edges in cases:
            stage, _ = self._stage(50, edges, params)
            self.assertIs(stage, params)

    def test_missing_settings_use_the_kernel_defaults(self):
        # The kernels fall back to DT 0.1, SPRING_K 0.1 and DAMPING 0.5.
        star = star_graph(4000)
        implicit, _ = self._stage(4001, star, {})
        explicit, _ = self._stage(
            4001, star, {"DT": 0.1, "SPRING_K": 0.1, "DAMPING": 0.5}
        )
        self.assertLess(implicit["DT"], 0.1)
        for key in ("DT", "MAX_STEPS", "RMSD_WINDOW", "RMSD_THRESHOLD"):
            self.assertEqual(implicit[key], explicit[key])


class AutoTimeStepTests(unittest.TestCase):
    PHYSICS = {
        "AUTO_DT": True,
        "SPRING_K": 5.0,
        "DAMPING": 0.9,
        "COULOMB_K": 10.0,
        "MAX_FORCE_LIMIT": 20.0,
        "COULOMB_CUTOFF": 30.0,
        "MAX_STEPS": 1000,
        "RMSD_WINDOW": 50,
        "RMSD_THRESHOLD": 0.005,
    }

    def _stage(self, n, edges, params):
        output = io.StringIO()
        with redirect_stdout(output):
            stage = ssn_engine._stage_params(n, edges, params)
        return stage, output.getvalue()

    def test_steepest_repulsion_is_where_the_force_cap_starts(self):
        steepest = ssn_engine._steepest_repulsion
        self.assertAlmostEqual(
            steepest({"COULOMB_K": 10.0, "MAX_FORCE_LIMIT": 20.0}),
            2.0 * 10.0 / math.sqrt(10.0 / 20.0) ** 3,
        )
        # Without a cap, or with one that never binds above the kernels'
        # 0.5 distance floor, the floor is the steepest point.
        self.assertAlmostEqual(steepest({"COULOMB_K": 10.0, "MAX_FORCE_LIMIT": 0.0}), 160.0)
        self.assertAlmostEqual(steepest({"COULOMB_K": 10.0, "MAX_FORCE_LIMIT": 100.0}), 160.0)
        self.assertEqual(steepest({"COULOMB_K": 0.0}), 0.0)
        # Missing settings use the kernels' fallbacks, COULOMB_K 50 and MAX_FORCE_LIMIT 20.
        self.assertAlmostEqual(steepest({}), 2.0 * 50.0 / math.sqrt(2.5) ** 3)

    def test_auto_dt_is_half_the_limit_of_springs_plus_one_stiff_contact(self):
        # The configured DT is ignored whether it is smaller or larger.
        for name, n, edges, configured in (
            ("K_100", 100, complete_graph(100), 0.001),
            ("pair", 2, complete_graph(2), 5.0),
        ):
            with self.subTest(graph=name):
                params = {**self.PHYSICS, "DT": configured}
                stage, log = self._stage(n, edges, params)
                stiffness = (
                    5.0 * ssn_engine._spring_degree_bound(n, edges)
                    + 2.0 * ssn_engine._steepest_repulsion(params)
                )
                expected = float(f"{0.5 * ssn_engine._euler_dt_limit(stiffness, 0.9):.3g}")
                self.assertEqual(stage["DT"], expected)
                self.assertIn(f"Auto DT {expected:g} for this stage", log)

    def test_auto_dt_applies_the_stopping_rules_as_configured(self):
        params = {**self.PHYSICS, "DT": 0.001}
        original = dict(params)
        stage, _ = self._stage(100, complete_graph(100), params)
        self.assertIsNot(stage, params)
        for key in ("MAX_STEPS", "RMSD_WINDOW", "RMSD_THRESHOLD"):
            self.assertEqual(stage[key], params[key])
        self.assertEqual(params, original)

    def test_without_any_stiffness_the_configured_dt_runs(self):
        params = {**self.PHYSICS, "DT": 0.02, "SPRING_K": 0.0, "COULOMB_K": 0.0}
        stage, _ = self._stage(50, complete_graph(50), params)
        self.assertIs(stage, params)

    def test_auto_dt_settles_pairs_triangles_and_small_cliques(self):
        # Steps sized for the springs alone let pairs and triangles keep
        # passing through each other's repulsive cores.
        graphs = (
            ("pair", 2, complete_graph(2)),
            ("triangle", 3, complete_graph(3)),
            ("K_4", 4, complete_graph(4)),
            ("path P_3", 3, np.array([(0, 1), (1, 2)], dtype=np.int32)),
        )
        for name, n, edges in graphs:
            with redirect_stdout(io.StringIO()):
                params = ssn_engine._stage_params(n, edges, self.PHYSICS)
            box = (math.sqrt(n) * 2.5 + 5.0) * 2.0
            steps = int(round(100.0 / params["DT"]))
            for seed in range(8):
                with self.subTest(graph=name, seed=seed):
                    start = np.random.default_rng(seed).uniform(
                        -0.4 * box, 0.4 * box, (n, 2)
                    ).astype(np.float32)
                    simulation = ssn_engine.SSNSimulationCPU(
                        start, edges, np.zeros(n, dtype=np.int32), box, params
                    )
                    tail = [simulation.step(step) for step in range(steps)][-200:]
                    self.assertLess(np.mean(tail) / params["DT"], 1e-3)

    def test_calculate_layout_ignores_the_configured_dt_in_auto_mode(self):
        edges = np.vstack((
            complete_graph(2), complete_graph(3) + 2, complete_graph(30) + 5,
        ))
        connectivity = np.column_stack((edges, np.ones(len(edges)))).astype(np.float64)
        params = {
            **self.PHYSICS,
            "DT": 5.0,
            "MAX_STEPS": 3000,
            "PERCENTAGE_DROP_THRESHOLD": 0.1,
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "LAYOUT_SEED": 0,
            "SIMILARITY_THRESHOLD": 0.0,
            "BOX_SCALE": 2.0,
            "PACKING_GRID_SIZE": 10.0,
            "PACKING_PADDING": 10.0,
        }
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            positions, _ = ssn_engine.calculate_layout(connectivity, 35, params)
        log = output.getvalue()
        self.assertIn("Auto DT", log)
        self.assertNotIn("WARNING", log)
        self.assertNotIn("Not settling", log)
        self.assertTrue(np.isfinite(positions).all())


class AutoTimeStepDefaultTests(unittest.TestCase):
    FILES = (
        "EMAPSSN_Config.py",
        os.path.join("desktop", "Viewer_State.py"),
        "Layout_Cache_Generator.py",
        os.path.join("mcp_server", "pipeline", "Pipeline_Operations.py"),
        "Layout_Engine_SSN.py",
    )

    def test_auto_dt_is_off_by_default_everywhere(self):
        found = []
        for relative in self.FILES:
            with open(os.path.join(SRC_DIR, relative), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if any(isinstance(t, ast.Name) and t.id == "AUTO_DT" for t in targets):
                        found.append((relative, node.value))
                elif isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == "AUTO_DT":
                            found.append((relative, value))
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "AUTO_DT"
                ):
                    found.append((relative, node.args[1]))
        defaults = [
            (relative, value.value) for relative, value in found
            if isinstance(value, ast.Constant) and isinstance(value.value, bool)
        ]
        self.assertGreaterEqual(len(defaults), 8, defaults)
        self.assertEqual({value for _, value in defaults}, {False}, defaults)
        self.assertEqual(
            {relative for relative, _ in defaults}, set(self.FILES), defaults
        )


class GuardedSimulationTests(unittest.TestCase):
    PHYSICS = {
        "SPRING_K": 5.0,
        "DAMPING": 0.9,
        "COULOMB_K": 0.0,
        "MAX_STEPS": 1500,
        "RMSD_WINDOW": 50,
        "RMSD_THRESHOLD": 1e-12,
        "PERCENTAGE_DROP_THRESHOLD": 0.0,
    }

    def _run_stage(self, n, edges, params):
        start = np.random.default_rng(0).uniform(-20.0, 20.0, (n, 2)).astype(np.float32)
        output = io.StringIO()
        with redirect_stdout(output):
            positions = ssn_engine._run_layout_stage(
                CPU, start, edges, np.zeros(n, dtype=np.int32),
                np.full(n, 500.0, dtype=np.float32), params,
                np.ones(n, dtype=np.bool_),
            )
        return np.asarray(positions), output.getvalue()

    def test_guard_keeps_a_clique_from_blowing_up(self):
        n = 100
        edges = complete_graph(n)
        params = {
            **self.PHYSICS,
            "DT": 1.1 * ssn_engine._euler_dt_limit(5.0 * n, 0.9),
        }
        _, log = self._run_stage(n, edges, params)
        self.assertIn(
            f"WARNING: {n} of {n} active nodes ended on the layout boundary", log
        )

        with redirect_stdout(io.StringIO()):
            guarded = ssn_engine._stage_params(n, edges, params)
        positions, log = self._run_stage(n, edges, guarded)
        self.assertNotIn("WARNING", log)
        # Without repulsion the springs pull the clique onto one point.
        self.assertLess(np.abs(positions - positions.mean(axis=0)).max(), 1e-3)

    def test_calculate_layout_runs_each_stage_with_its_guarded_params(self):
        n = 60
        edges = complete_graph(n)
        connectivity = np.column_stack((edges, np.ones(len(edges)))).astype(np.float64)
        params = {
            **self.PHYSICS,
            "DT": 1.33 * ssn_engine._euler_dt_limit(5.0 * n, 0.9),
            "MAX_STEPS": 600,
            "LAYOUT_DEVICE_SELECTION": "cpu",
            "LAYOUT_SEED": 0,
            "SIMILARITY_THRESHOLD": 0.0,
            "BOX_SCALE": 1.0,
            "PACKING_GRID_SIZE": 10.0,
            "PACKING_PADDING": 10.0,
        }

        def layout_log():
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                ssn_engine.calculate_layout(connectivity, n, dict(params))
            return output.getvalue()

        guarded = layout_log()
        self.assertIn("for this stage so its springs stay stable", guarded)
        self.assertNotIn("WARNING", guarded)

        with mock.patch.object(
            ssn_engine, "_stage_params", side_effect=lambda nodes, springs, p: p
        ):
            unguarded = layout_log()
        self.assertIn(
            f"WARNING: {n} of {n} active nodes ended on the layout boundary", unguarded
        )


class StageEndReportTests(unittest.TestCase):
    PARAMS = {
        "MAX_STEPS": 400,
        "RMSD_WINDOW": 10,
        "RMSD_THRESHOLD": 1e-6,
        "PERCENTAGE_DROP_THRESHOLD": 0.1,
    }

    def _report(self, rmsd_values, positions=None, active_mask=None, box=10.0):
        if positions is None:
            positions = np.zeros((3, 2), dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        script = list(rmsd_values)

        class ScriptedSimulation:
            """Replays RMSD values and returns fixed positions."""

            def __init__(self, *args, **kwargs):
                self._rmsd = iter(script)

            def step(self, current_step):
                return next(self._rmsd)

            def get_pos(self):
                return positions

        output = io.StringIO()
        with mock.patch.object(
            ssn_engine, "SSNSimulationCPU", ScriptedSimulation
        ), redirect_stdout(output):
            ssn_engine._run_layout_stage(
                CPU, positions, np.zeros((0, 2), dtype=np.int32),
                np.zeros(len(positions), dtype=np.int32),
                np.full(len(positions), box, dtype=np.float32),
                dict(self.PARAMS), active_mask,
            )
        return output.getvalue()

    def test_converged_stage(self):
        self.assertIn("Converged at Step 9", self._report([1e-7] * 400))

    def test_flat_rmsd_is_a_plateau(self):
        self.assertIn("Plateau Reached at Step 101", self._report([1.0] * 400))

    def test_rising_rmsd_stops_at_the_same_step_as_not_settling(self):
        log = self._report([1.01 ** step for step in range(400)])
        self.assertIn("Not settling at Step 101", log)
        self.assertNotIn("Plateau", log)

    def test_slight_rise_is_still_a_plateau(self):
        # About 0.1% per window: noise around a plateau, not growth.
        log = self._report([1.0001 ** step for step in range(400)])
        self.assertIn("Plateau Reached at Step 101", log)
        self.assertNotIn("Not settling", log)

    def test_exhausted_step_budget_is_reported(self):
        log = self._report([0.99 ** step for step in range(400)])
        self.assertIn("Step limit reached after 400 steps", log)

    def test_nodes_on_the_box_wall_are_reported(self):
        on_walls = [[10.0, 0.0], [0.0, -10.0], [3.0, 3.0]]
        settled = [1e-7] * 400
        self.assertIn(
            "WARNING: 2 of 3 active nodes ended on the layout boundary",
            self._report(settled, on_walls),
        )
        self.assertIn(
            "WARNING: 1 of 2 active nodes ended on the layout boundary",
            self._report(settled, on_walls, np.array([False, True, True])),
        )
        inside = [[9.9, 0.0], [0.0, -9.9], [3.0, 3.0]]
        self.assertNotIn("WARNING", self._report(settled, inside))


if __name__ == "__main__":
    unittest.main()
