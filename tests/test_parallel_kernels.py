"""Numba-parallel kernels match their serial definitions on any thread count.

The references below restate the serial layout physics and neighbor-joining
kernels from before the parallel rewrite (commit 5d3cd91). The parallel
kernels must reproduce them bit for bit, so layouts and guide trees do not
change with the thread count or the upgrade.
"""

import gc
import io
import math
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import squareform


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
UTILITIES_DIR = SRC_DIR / "utilities"
TOOLS_DIR = SRC_DIR / "tools"
for directory in (SRC_DIR, UTILITIES_DIR, TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Embedding_MSA
    import Layout_Engine_SSN as ssn_engine
    from utilities import Network_Kernels
    from utilities import Numba_Threads

# Numba is a pinned requirement, and Embedding_MSA above imports it
# unconditionally, so these tests never run without it.
import numba  # noqa: E402
from numba import jit  # noqa: E402

POOL_SIZE = numba.config.NUMBA_NUM_THREADS
needs_threads = unittest.skipUnless(
    POOL_SIZE >= 2, "Numba has fewer than two threads"
)


def _serial_kernel_2d(pos, vel, springs, comp_labels, active_mask, active_nodes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist):
    n_balls = pos.shape[0]
    acc = np.zeros_like(pos)
    repulsion = np.zeros_like(pos)
    cutoff_sq = cutoff_dist * cutoff_dist
    taper_start = cutoff_dist * 0.8
    taper_width = max(cutoff_dist * 0.2, 1e-9)
    for i in range(springs.shape[0]):
        idx_a, idx_b = springs[i, 0], springs[i, 1]
        if not active_mask[idx_a] or not active_mask[idx_b]:
            continue
        dx, dy = pos[idx_a, 0] - pos[idx_b, 0], pos[idx_a, 1] - pos[idx_b, 1]
        dist = math.sqrt(dx*dx + dy*dy) + 1e-9
        f = -k_spr * dist
        acc[idx_a, 0] += f * (dx/dist); acc[idx_a, 1] += f * (dy/dist)
        acc[idx_b, 0] -= f * (dx/dist); acc[idx_b, 1] -= f * (dy/dist)
    for active_i in range(active_nodes.shape[0]):
        i = active_nodes[active_i]
        for active_j in range(active_i + 1, active_nodes.shape[0]):
            j = active_nodes[active_j]
            if comp_labels[i] != comp_labels[j]:
                continue
            dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > cutoff_sq: continue
            if dist_sq == 0.0: continue
            dist = math.sqrt(dist_sq)
            safe_dist = max(dist, 0.5)
            f = k_coul / (safe_dist**2)
            if max_f > 0.0 and f > max_f:
                f = max_f
            if dist > taper_start:
                f *= max(0.0, (cutoff_dist - dist) / taper_width)
            repulsion[i, 0] += f*(dx/dist); repulsion[i, 1] += f*(dy/dist)
            repulsion[j, 0] -= f*(dx/dist); repulsion[j, 1] -= f*(dy/dist)
    for active_idx in range(active_nodes.shape[0]):
        i = active_nodes[active_idx]
        rep_norm = math.sqrt(repulsion[i, 0] * repulsion[i, 0] + repulsion[i, 1] * repulsion[i, 1])
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[i, 0] *= rep_scale
            repulsion[i, 1] *= rep_scale
        acc[i, 0] += repulsion[i, 0]
        acc[i, 1] += repulsion[i, 1]
    rmsd = 0.0
    n_active = active_nodes.shape[0]
    for i in range(n_balls):
        if not active_mask[i]:
            continue
        box_limit = box_limits[i]
        acc[i] -= damping * vel[i]
        vel[i] += acc[i] * dt
        old_p = pos[i].copy()
        pos[i] += vel[i] * dt
        if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
        elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
        if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
        elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5
        diff = pos[i] - old_p
        rmsd += diff[0]**2 + diff[1]**2
    if n_active == 0:
        return 0.0
    return math.sqrt(rmsd / n_active)


def _serial_kernel_3d(pos, vel, springs, comp_labels, active_mask, active_nodes, box_limits, dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist):
    n_balls = pos.shape[0]
    acc = np.zeros_like(pos)
    repulsion = np.zeros_like(pos)
    cutoff_sq = cutoff_dist * cutoff_dist
    taper_start = cutoff_dist * 0.8
    taper_width = max(cutoff_dist * 0.2, 1e-9)
    for i in range(springs.shape[0]):
        idx_a, idx_b = springs[i, 0], springs[i, 1]
        if not active_mask[idx_a] or not active_mask[idx_b]:
            continue
        dx = pos[idx_a, 0] - pos[idx_b, 0]
        dy = pos[idx_a, 1] - pos[idx_b, 1]
        dz = pos[idx_a, 2] - pos[idx_b, 2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
        f = -k_spr * dist
        acc[idx_a, 0] += f * (dx/dist); acc[idx_a, 1] += f * (dy/dist); acc[idx_a, 2] += f * (dz/dist)
        acc[idx_b, 0] -= f * (dx/dist); acc[idx_b, 1] -= f * (dy/dist); acc[idx_b, 2] -= f * (dz/dist)
    for active_i in range(active_nodes.shape[0]):
        i = active_nodes[active_i]
        for active_j in range(active_i + 1, active_nodes.shape[0]):
            j = active_nodes[active_j]
            if comp_labels[i] != comp_labels[j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            if dist_sq > cutoff_sq: continue
            if dist_sq == 0.0: continue
            dist = math.sqrt(dist_sq)
            safe_dist = max(dist, 0.5)
            f = k_coul / (safe_dist**2)
            if max_f > 0.0 and f > max_f:
                f = max_f
            if dist > taper_start:
                f *= max(0.0, (cutoff_dist - dist) / taper_width)
            repulsion[i, 0] += f*(dx/dist); repulsion[i, 1] += f*(dy/dist); repulsion[i, 2] += f*(dz/dist)
            repulsion[j, 0] -= f*(dx/dist); repulsion[j, 1] -= f*(dy/dist); repulsion[j, 2] -= f*(dz/dist)
    for active_idx in range(active_nodes.shape[0]):
        i = active_nodes[active_idx]
        rep_norm = math.sqrt(
            repulsion[i, 0] * repulsion[i, 0]
            + repulsion[i, 1] * repulsion[i, 1]
            + repulsion[i, 2] * repulsion[i, 2]
        )
        if max_total_repulsion > 0.0 and rep_norm > max_total_repulsion:
            rep_scale = max_total_repulsion / rep_norm
            repulsion[i, 0] *= rep_scale
            repulsion[i, 1] *= rep_scale
            repulsion[i, 2] *= rep_scale
        acc[i, 0] += repulsion[i, 0]
        acc[i, 1] += repulsion[i, 1]
        acc[i, 2] += repulsion[i, 2]
    rmsd = 0.0
    n_active = active_nodes.shape[0]
    for i in range(n_balls):
        if not active_mask[i]:
            continue
        box_limit = box_limits[i]
        acc[i] -= damping * vel[i]
        vel[i] += acc[i] * dt
        old_p = pos[i].copy()
        pos[i] += vel[i] * dt
        if pos[i,0] > box_limit: pos[i,0]=box_limit; vel[i,0]*=-0.5
        elif pos[i,0] < -box_limit: pos[i,0]=-box_limit; vel[i,0]*=-0.5
        if pos[i,1] > box_limit: pos[i,1]=box_limit; vel[i,1]*=-0.5
        elif pos[i,1] < -box_limit: pos[i,1]=-box_limit; vel[i,1]*=-0.5
        if pos[i,2] > box_limit: pos[i,2]=box_limit; vel[i,2]*=-0.5
        elif pos[i,2] < -box_limit: pos[i,2]=-box_limit; vel[i,2]*=-0.5
        diff = pos[i] - old_p
        rmsd += diff[0]**2 + diff[1]**2 + diff[2]**2
    if n_active == 0:
        return 0.0
    return math.sqrt(rmsd / n_active)


def _serial_neighbor_joining(D, N):
    Z = np.zeros((N - 1, 4), dtype=np.float64)
    active_list = np.arange(N, dtype=np.int32)
    k = N
    R = np.zeros(2 * N - 1, dtype=np.float64)
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += D[i, j]
        R[i] = s
    node_height = np.zeros(2 * N - 1, dtype=np.float64)
    num_leaves = np.ones(2 * N - 1, dtype=np.float64)
    r_list = np.zeros(2 * N - 1, dtype=np.float64)
    for step in range(N - 1):
        if k > 2:
            inv_k_minus_2 = 1.0 / (k - 2)
            min_Q = 1e15
            idx_u = -1
            idx_v = -1
            for i in range(k):
                r_list[i] = R[active_list[i]] * inv_k_minus_2
            for i in range(k):
                u = active_list[i]
                r_u = r_list[i]
                for j in range(i + 1, k):
                    v = active_list[j]
                    q = D[u, v] - (r_u + r_list[j])
                    if q < min_Q:
                        min_Q = q
                        idx_u = i
                        idx_v = j
        else:
            idx_u = 0
            idx_v = 1
        u = active_list[idx_u]
        v = active_list[idx_v]
        w = N + step
        dist_uv = D[u, v]
        child_max = max(node_height[u], node_height[v])
        node_height[w] = max(dist_uv, child_max)
        sum_d_wm = 0.0
        for p in range(k):
            if p != idx_u and p != idx_v:
                m = active_list[p]
                d_wm = 0.5 * (D[u, m] + D[v, m] - dist_uv)
                D[w, m] = d_wm
                D[m, w] = d_wm
                sum_d_wm += d_wm
                R[m] = R[m] - D[u, m] - D[v, m] + d_wm
        R[w] = sum_d_wm
        c1 = min(u, v)
        c2 = max(u, v)
        Z[step, 0] = float(c1)
        Z[step, 1] = float(c2)
        Z[step, 2] = node_height[w]
        Z[step, 3] = num_leaves[u] + num_leaves[v]
        num_leaves[w] = num_leaves[u] + num_leaves[v]
        write_idx = 0
        for p in range(k):
            if p != idx_u and p != idx_v:
                active_list[write_idx] = active_list[p]
                write_idx += 1
        active_list[write_idx] = w
        k -= 1
    return Z


# dt, damping, k_spr, k_coul, max_f, max_total_repulsion, cutoff_dist
DEFAULT_FORCES = (0.005, 0.9, 5.0, 10.0, 20.0, 0.0, 30.0)
CAPPED_FORCES = (0.05, 0.5, 0.1, 50.0, 20.0, 3.0, 15.0)


def _layout_problem(count, dimensions, labels, *, active_fraction=1.0, seed=0):
    rng = np.random.default_rng(seed)
    box = np.sqrt(count) * 2.5 + 5.0
    positions = ((rng.random((count, dimensions)) - 0.5) * box).astype(np.float32)
    springs = []
    for label in np.unique(labels):
        nodes = np.flatnonzero(labels == label)
        for _ in range(3 * len(nodes)):
            source, target = rng.choice(nodes, 2)
            if source != target:
                springs.append((source, target))
    mask = rng.random(count) < active_fraction
    # Some nodes start outside their box, so the wall clamps are exercised.
    limits = np.full(count, box * 0.45, dtype=np.float32)
    return (
        positions,
        np.asarray(springs, dtype=np.int32).reshape(-1, 2),
        np.asarray(labels, dtype=np.int32),
        mask,
        np.flatnonzero(mask).astype(np.int32),
        limits,
    )


def _run_kernel(kernel, problem, forces, steps=8):
    positions, springs, labels, mask, active, limits = problem
    positions = positions.copy()
    velocity = np.zeros_like(positions)
    rmsd = [
        kernel(positions, velocity, springs, labels, mask, active, limits, *forces)
        for _ in range(steps)
    ]
    return positions, velocity, np.asarray(rmsd)


class ThreadBudgetTests(unittest.TestCase):
    def test_leaves_two_usable_cpus_free(self):
        self.assertEqual(Numba_Threads.choose_thread_count(64, logical_cpus=20), 18)
        self.assertEqual(Numba_Threads.choose_thread_count(64, logical_cpus=4), 2)

    def test_keeps_at_least_one_thread(self):
        for logical_cpus in (1, 2, 3):
            with self.subTest(logical_cpus=logical_cpus):
                self.assertEqual(
                    Numba_Threads.choose_thread_count(64, logical_cpus=logical_cpus),
                    1,
                )

    def test_stays_within_numba_pool(self):
        self.assertEqual(Numba_Threads.choose_thread_count(8, logical_cpus=20), 8)

    def test_explicit_numba_num_threads_is_used_as_given(self):
        self.assertEqual(
            Numba_Threads.choose_thread_count(
                20, logical_cpus=20, explicit_limit=True
            ),
            20,
        )

    def test_default_follows_numba_num_threads_only_when_set(self):
        with mock.patch.dict(os.environ, {"NUMBA_NUM_THREADS": str(POOL_SIZE)}):
            self.assertEqual(Numba_Threads.default_thread_count(), POOL_SIZE)
        with mock.patch.dict(os.environ), mock.patch.object(
            Numba_Threads, "usable_cpu_count", return_value=20
        ):
            os.environ.pop("NUMBA_NUM_THREADS", None)
            self.assertEqual(
                Numba_Threads.default_thread_count(), min(POOL_SIZE, 18)
            )

    @needs_threads
    def test_limited_threads_restores_the_previous_count(self):
        previous = numba.get_num_threads()
        with Numba_Threads.limited_threads(1):
            self.assertEqual(numba.get_num_threads(), 1)
        self.assertEqual(numba.get_num_threads(), previous)
        with Numba_Threads.limited_threads(POOL_SIZE + 100):
            self.assertEqual(numba.get_num_threads(), POOL_SIZE)
        self.assertEqual(numba.get_num_threads(), previous)


class ParallelLayoutKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.serial = {
            2: jit(nopython=True, fastmath=True)(_serial_kernel_2d),
            3: jit(nopython=True, fastmath=True)(_serial_kernel_3d),
        }

    def problems(self, dimensions):
        rng = np.random.default_rng(3)
        return {
            "one component": _layout_problem(300, dimensions, np.zeros(300, int)),
            "interleaved components": _layout_problem(
                300, dimensions, rng.integers(0, 4, 300), seed=1
            ),
            "partly active batch": _layout_problem(
                300, dimensions, np.repeat(np.arange(3), 100),
                active_fraction=0.6, seed=2,
            ),
        }

    def assert_same_run(self, expected, actual):
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_array_equal(actual_array, expected_array)

    def test_matches_serial_all_pairs_kernel_bit_for_bit(self):
        for dimensions in (2, 3):
            kernel = ssn_engine._get_physics_kernel(dimensions)
            for name, problem in self.problems(dimensions).items():
                for forces in (DEFAULT_FORCES, CAPPED_FORCES):
                    with self.subTest(dimensions=dimensions, case=name, cap=forces[5]):
                        self.assert_same_run(
                            _run_kernel(self.serial[dimensions], problem, forces),
                            _run_kernel(kernel, problem, forces),
                        )

    def test_euler_update_rounds_each_product_like_the_serial_kernel(self):
        # Fusing a product into its sum (FMA) changes only a few float32
        # results in 10^5, so this case moves many independent nodes at once.
        rng = np.random.default_rng(0)
        count = 20000
        problem = (
            ((rng.random((count, 2)) - 0.5) * 200).astype(np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.arange(count, dtype=np.int32),
            np.ones(count, dtype=np.bool_),
            np.arange(count, dtype=np.int32),
            np.full(count, 90.0, dtype=np.float32),
        )
        velocity = rng.normal(0.0, 30.0, (count, 2)).astype(np.float32)
        forces = (0.1, 0.9, 5.0, 10.0, 20.0, 0.0, 30.0)
        runs = []
        for kernel in (self.serial[2], ssn_engine._get_physics_kernel(2)):
            positions, moving = problem[0].copy(), velocity.copy()
            rmsd = [
                kernel(positions, moving, *problem[1:], *forces)
                for _ in range(4)
            ]
            runs.append((positions, moving, np.asarray(rmsd)))
        self.assert_same_run(*runs)

    @needs_threads
    def test_result_does_not_depend_on_thread_count(self):
        for dimensions in (2, 3):
            kernel = ssn_engine._get_physics_kernel(dimensions)
            problem = self.problems(dimensions)["interleaved components"]
            with Numba_Threads.limited_threads(1):
                one_thread = _run_kernel(kernel, problem, CAPPED_FORCES)
            with Numba_Threads.limited_threads(POOL_SIZE):
                all_threads = _run_kernel(kernel, problem, CAPPED_FORCES)
            with self.subTest(dimensions=dimensions):
                self.assert_same_run(one_thread, all_threads)

    @needs_threads
    def test_simulation_step_runs_on_its_thread_budget(self):
        simulation = ssn_engine.SSNSimulationCPU(
            np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(2, dtype=np.int32),
            100.0,
            {"DT": 0.1, "DAMPING": 0.5, "COULOMB_K": 50.0},
        )
        simulation.threads = 1
        kernel = simulation._kernel
        seen = []

        def recording_kernel(*arguments):
            seen.append(numba.get_num_threads())
            return kernel(*arguments)

        simulation._kernel = recording_kernel
        previous = numba.get_num_threads()
        with Numba_Threads.limited_threads(POOL_SIZE):
            simulation.step(0)
            self.assertEqual(numba.get_num_threads(), POOL_SIZE)
        self.assertEqual(seen, [1])
        self.assertEqual(numba.get_num_threads(), previous)


class ParallelNeighborJoiningTests(unittest.TestCase):
    def test_ties_keep_the_first_minimum_in_row_major_order(self):
        # Q(0,1), Q(0,2), Q(1,3) and Q(2,3) are all exactly -4.5. The serial
        # scan merged (0, 1): the first tie in its row and the first row.
        distances = np.array(
            [[0, 1, 1, 3], [1, 0, 3, 2], [1, 3, 0, 2], [3, 2, 2, 0]],
            dtype=np.float32,
        )
        for threads in (1, 2):
            with self.subTest(threads=threads):
                tree = Embedding_MSA.neighbor_joining_condensed(
                    squareform(distances), 4, threads=threads
                )
                np.testing.assert_array_equal(tree[0, :2], [0.0, 1.0])

    def test_matches_serial_scan_bit_for_bit_on_any_thread_count(self):
        serial = jit(nopython=True, fastmath=True)(_serial_neighbor_joining)
        rng = np.random.default_rng(5)
        # Odd and even sizes: the folded row pairing has a middle row when
        # the active count is odd.
        for count in (5, 64, 151):
            pair_count = count * (count - 1) // 2
            for name, condensed in (
                ("continuous", rng.random(pair_count).astype(np.float32)),
                ("many ties", rng.integers(1, 4, pair_count).astype(np.float32)),
            ):
                square = np.zeros((2 * count - 1, 2 * count - 1))
                square[:count, :count] = squareform(condensed)
                expected = serial(square, count)
                for threads in sorted({1, POOL_SIZE}):
                    with self.subTest(count=count, case=name, threads=threads):
                        np.testing.assert_array_equal(
                            Embedding_MSA.neighbor_joining_condensed(
                                condensed, count, threads=threads
                            ),
                            expected,
                        )

    def test_bootstrap_worker_passes_its_thread_share(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "baseline.dat")
            np.linspace(0.1, 1.0, 10, dtype=np.float32).tofile(path)
            with mock.patch.object(
                Embedding_MSA, "neighbor_joining_condensed",
                return_value=np.zeros((4, 4)),
            ) as neighbor_joining:
                Embedding_MSA.compute_single_tree_worker(
                    7, 5, path, 1.0, 0.0, "Neighbor-joining (Slow)",
                    kernel_threads=3,
                )
            gc.collect()  # Windows keeps the memmap's file open until collected.
        self.assertEqual(neighbor_joining.call_args.kwargs["threads"], 3)


class ParallelJaccardFilterTests(unittest.TestCase):
    def test_matches_closed_neighbourhood_set_definition_on_any_thread_count(self):
        rng = np.random.default_rng(9)
        count, pairs = 200, 900
        adjacency = sp.coo_matrix(
            (np.ones(pairs), (rng.integers(0, count, pairs), rng.integers(0, count, pairs))),
            shape=(count, count),
        )
        adjacency = ((adjacency + adjacency.T) > 0).tocsr()
        adjacency.sort_indices()
        edges = np.column_stack(sp.triu(adjacency, 1).nonzero()).astype(np.int64)
        # A node counts as its own neighbour.
        neighbours = [
            set(adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]) | {node}
            for node in range(count)
        ]
        for threshold in (0.0, 0.1, 0.3):
            expected = [
                len(neighbours[u] & neighbours[v]) / len(neighbours[u] | neighbours[v])
                >= threshold
                for u, v in edges
            ]
            for threads in (1, POOL_SIZE):
                with self.subTest(threshold=threshold, threads=threads), \
                        mock.patch.object(
                            Numba_Threads, "default_thread_count",
                            return_value=threads,
                        ):
                    np.testing.assert_array_equal(
                        Network_Kernels.fast_jaccard_filter(
                            edges, adjacency.indptr, adjacency.indices, threshold
                        ),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
