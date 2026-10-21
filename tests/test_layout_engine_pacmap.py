"""Layout_Engine_PaCMAP: the optimiser, the pairs it is given and the layout it returns.

* The kernel is checked against the loss written out in NumPy: its gradient is the derivative of the
  sum over pair rows of w * d/(10+d), w * d/(10000+d) and w/(1+d), d = |y_i - y_j|^2 + 1, in 2D and 3D.
* It is a port of the pacmap 0.9.1 package, which is not installed here. REFERENCE below holds what the package
  gave on the same network, pairs and start (written down once, in a scratch environment that has it); the
  port must give the same metrics within the spread of its own starts.
* The coordinates must not depend on the number of threads (compared in processes of 1 and 3 threads), and
  they follow LAYOUT_SEED. Isolated nodes, tiny and empty networks, 2D and 3D, and the mid-near pairs'
  share of the layout's global arrangement (a layout with random mid-near partners is as local as UMAP's) are
  covered too.
"""

import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Layout_Engine_PaCMAP as engine
from tests.network_fixtures import ArraySource
from utilities import Numba_Threads
from utilities import PaCMAP_Pairs as pairs

# The synthetic network: 8 clusters of 40 around a circle of radius 10, each node 2D latent coordinates
# plus unit noise; a pair's score is the negated latent distance; the strongest 10 edges per node are the
# network's edges. Its metrics are the share of a node's 10 nearest nodes in the embedding that are among
# its 10 nearest in the latent plane, and the Spearman correlation of all pairwise distances.
CLUSTERS, PER_CLUSTER, NEIGHBORS = 8, 40, 10
COUNT = CLUSTERS * PER_CLUSTER

# pacmap 0.9.1 on this network with this package's pairs (the near and further pairs of build_pairs, the mid-near
# pairs of mid_near_pairs from the stored scores, seed 42) and the start of normal_start(seed 42), 450
# iterations: (neighbour share, global rho) in 2D and 3D. Different starts spread by about 0.02 around it.
REFERENCE = {2: (0.7738, 0.918), 3: (0.7747, 0.8449)}


def synthetic_network():
    rng = np.random.default_rng(3)
    cluster = np.repeat(np.arange(CLUSTERS), PER_CLUSTER)
    angle = cluster * 2 * np.pi / CLUSTERS
    centre = np.column_stack((np.cos(angle), np.sin(angle))) * 10.0
    points = centre + rng.normal(size=(COUNT, 2))
    distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return points, distance


def strongest_edges(distance, neighbors=NEIGHBORS):
    """The edges the strongest-edge pass keeps: every node's nearest, as a source and as a target."""
    shown = distance + np.diag(np.full(len(distance), np.inf))
    nearest = np.argsort(shown, axis=1, kind="stable")[:, :neighbors]
    edges = sorted({(min(i, j), max(i, j)) for i in range(len(distance)) for j in nearest[i]})
    u = np.array([edge[0] for edge in edges])
    v = np.array([edge[1] for edge in edges])
    return u, v, -distance[u, v]


def stored_pairs(distance):
    """The network's pair file: every pair once, in no order, scored by negated latent distance."""
    u, v = np.triu_indices(len(distance), k=1)
    order = np.random.default_rng(10).permutation(len(u))
    u, v = u[order], v[order]
    return ArraySource(u.astype(np.uint16), v.astype(np.uint16), (-distance[u, v]).astype(np.float32))


def measures(embedding, points, distance, neighbors=NEIGHBORS):
    n = len(points)
    true_d = distance + np.diag(np.full(n, np.inf))
    embedded = np.sqrt(((embedding[:, None, :] - embedding[None, :, :]) ** 2).sum(-1))
    shown = embedded + np.diag(np.full(n, np.inf))
    true_nn = np.argsort(true_d, axis=1)[:, :neighbors]
    embedded_nn = np.argsort(shown, axis=1)[:, :neighbors]
    share = np.mean([len(set(true_nn[i]) & set(embedded_nn[i])) / neighbors for i in range(n)])
    upper = np.triu_indices(n, 1)
    return float(share), float(spearmanr(embedded[upper], distance[upper]).correlation)


def connectivity_of(u, v, scores):
    return np.column_stack((u, v, scores)).astype(np.float32)


def fit(u, v, scores, count, neighbors, mid, dims, seed):
    """The embedding the engine's optimiser gives (no box scaling), from the pairs of build_pairs."""
    ids, near, mid_lists, far = engine.build_pairs(connectivity_of(u, v, scores), count, neighbors, seed, mid)
    assert len(ids) == count
    return engine.optimise(count, dims, near, mid_lists, far, seed)


def clique_ring(cliques=8, clique_size=6, ring=40):
    """(connectivity, node count) of cliques whose scores are all equal, next to one graded ring.

    Every node of a clique of equal scores has the same weights to the others: the case where an unseeded
    eigsh made another method's layouts of one seed differ from run to run.
    """
    edges, node = [], 0
    for _ in range(cliques):
        members = list(range(node, node + clique_size))
        edges += [(a, b, 100.0) for i, a in enumerate(members) for b in members[i + 1:]]
        node += clique_size
    ring_nodes = list(range(node, node + ring))
    edges += [(ring_nodes[i], ring_nodes[(i + 1) % ring], 90.0 - i * 0.5) for i in range(ring)]
    edges += [(ring_nodes[i], ring_nodes[(i + 2) % ring], 60.0 - i * 0.3) for i in range(ring)]
    return np.array(edges, dtype=np.float32), node + ring


def numpy_loss(position, near, mid, far, weights):
    """The loss of the pairs in incidence lists, each pair row once (the lists hold a row at both its nodes)."""
    near_weight, mid_weight, far_weight = weights
    total = 0.0
    for (pointer, partners), kind in ((near, "near"), (mid, "mid"), (far, "far")):
        for node in range(len(pointer) - 1):
            for q in range(pointer[node], pointer[node + 1]):
                other = partners[q]
                if other < node:
                    continue          # the row (node, other) is listed at both; count it from its lower end
                d = 1.0 + float(((position[node] - position[other]) ** 2).sum())
                if kind == "near":
                    total += near_weight * d / (10.0 + d)
                elif kind == "mid":
                    total += mid_weight * d / (10000.0 + d)
                else:
                    total += far_weight / (1.0 + d)
    return total


class ScheduleTests(unittest.TestCase):
    def test_the_three_phases(self):
        self.assertEqual(sum(engine.ITERATIONS), 450)
        self.assertEqual(engine.phase_weights(0), (1000.0, 2.0, 1.0))
        self.assertAlmostEqual(engine.phase_weights(50)[0], 501.5)
        self.assertAlmostEqual(engine.phase_weights(99)[0], 0.01 * 1000 + 0.99 * 3)
        self.assertEqual(engine.phase_weights(100), (3.0, 3.0, 1.0))
        self.assertEqual(engine.phase_weights(199), (3.0, 3.0, 1.0))
        self.assertEqual(engine.phase_weights(200), (0.0, 1.0, 1.0))
        self.assertEqual(engine.phase_weights(449), (0.0, 1.0, 1.0))

    def test_adam_and_the_start_follow_the_package(self):
        self.assertEqual((engine.LEARNING_RATE, engine.BETA_1, engine.BETA_2), (1.0, 0.9, 0.999))
        self.assertEqual(engine.ADAM_EPSILON, 1e-7)
        self.assertEqual(engine.START_SCALE, 1e-4)


class KernelTests(unittest.TestCase):
    def build(self, dims):
        rng = np.random.default_rng(0)
        count = 40
        u = rng.integers(0, count, 300)
        v = rng.integers(0, count, 300)
        scores = rng.random(300)
        ids, near, mid, far = engine.build_pairs(connectivity_of(u, v, scores), count, 5, 42)
        position = (rng.normal(size=(len(ids), dims)) * 3).astype(np.float32)
        return position, near, mid, far

    def test_the_gradient_is_the_derivative_of_the_loss(self):
        for dims in (2, 3):
            for weights in ((2.0, 500.0, 1.0), (1.0, 0.0, 1.0), (3.0, 3.0, 0.5)):
                with self.subTest(dims=dims, weights=weights):
                    position, near, mid, far = self.build(dims)
                    gradient = np.zeros_like(position)
                    engine._gradient(position, near[0], near[1], mid[0], mid[1], far[0], far[1],
                                     weights[0], weights[1], weights[2], gradient)
                    base = position.astype(np.float64)
                    numeric = np.zeros_like(base)
                    step = 1e-5
                    for node in range(len(base)):
                        for axis in range(dims):
                            up, down = base.copy(), base.copy()
                            up[node, axis] += step
                            down[node, axis] -= step
                            numeric[node, axis] = (numpy_loss(up, near, mid, far, weights)
                                                   - numpy_loss(down, near, mid, far, weights)) / (2 * step)
                    self.assertLess(np.abs(numeric - gradient).max() / np.abs(numeric).max(), 1e-5)
                    # Every force has an equal and opposite partner.
                    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=1e-4 * np.abs(gradient).max())

    def test_adam_steps_as_the_package_does(self):
        rng = np.random.default_rng(1)
        position = rng.normal(size=(30, 3)).astype(np.float32)
        expected = position.astype(np.float64)
        first = np.zeros_like(position)
        second = np.zeros_like(position)
        want_first = np.zeros(position.shape)
        want_second = np.zeros(position.shape)
        for iteration in range(5):
            gradient = rng.normal(size=position.shape).astype(np.float32)
            step_size = math.sqrt(1 - 0.999 ** (iteration + 1)) / (1 - 0.9 ** (iteration + 1))
            engine._adam_step(position, gradient, first, second, step_size)
            g = gradient.astype(np.float64)
            want_first += 0.1 * (g - want_first)
            want_second += 0.001 * (g * g - want_second)
            expected -= step_size * want_first / (np.sqrt(want_second) + 1e-7)
        np.testing.assert_allclose(position, expected, rtol=0, atol=2e-5)


class ReferenceTests(unittest.TestCase):
    """The port gives the package's layouts' quality, on the package's own network and pairs."""

    @classmethod
    def setUpClass(cls):
        cls.points, cls.distance = synthetic_network()
        cls.u, cls.v, cls.scores = strongest_edges(cls.distance)
        cls.source = stored_pairs(cls.distance)
        ids = np.arange(COUNT)
        cls.mid = pairs.mid_near_pairs(cls.source, None, COUNT, ids, NEIGHBORS, 42)

    def embedding(self, dims, seed, mid=None):
        return fit(self.u, self.v, self.scores, COUNT, NEIGHBORS, self.mid if mid is None else mid, dims, seed)

    def test_the_start_of_the_reference_gives_the_references_metrics(self):
        for dims, (share, rho) in REFERENCE.items():
            with self.subTest(dims=dims):
                got = measures(self.embedding(dims, 42), self.points, self.distance)
                self.assertAlmostEqual(got[0], share, delta=0.01)
                self.assertAlmostEqual(got[1], rho, delta=0.03)

    def test_other_starts_stay_within_their_spread(self):
        for dims, (share, rho) in REFERENCE.items():
            with self.subTest(dims=dims):
                got = np.array([measures(self.embedding(dims, seed), self.points, self.distance)
                                for seed in (43, 44, 45)])
                self.assertAlmostEqual(got[:, 0].mean(), share, delta=0.03)
                self.assertAlmostEqual(got[:, 1].mean(), rho, delta=0.07)

    def test_mid_near_partners_from_the_stored_scores_carry_the_global_arrangement(self):
        # With every near pair inside a cluster, only the mid-near pairs know where the clusters lie. A layout
        # whose mid-near partners are random is as local as UMAP's (rho 0.05 to 0.23 on the real networks, 0.5
        # to 0.6 with the stored scores).
        blind = pairs.random_mid_near_pairs(np.arange(COUNT), NEIGHBORS, 42)
        scored = [measures(self.embedding(2, seed), self.points, self.distance)[1] for seed in (42, 43, 44)]
        random = [measures(self.embedding(2, seed, blind), self.points, self.distance)[1] for seed in (42, 43, 44)]
        self.assertGreater(min(scored), 0.8)
        self.assertLess(max(random), 0.6)
        self.assertGreater(np.mean(scored) - np.mean(random), 0.35)

    def test_local_structure_does_not_need_the_scores(self):
        share = [measures(self.embedding(2, seed, pairs.random_mid_near_pairs(np.arange(COUNT), NEIGHBORS, 42)),
                          self.points, self.distance)[0] for seed in (42, 43)]
        self.assertGreater(min(share), 0.7)


def layout(u, v, scores, count, params=None, mid=None):
    params = dict({"PACMAP_NEIGHBORS": NEIGHBORS, "BOX_SCALE": 2.0}, **(params or {}))
    if mid is not None:
        params["PACMAP_MID_NEAR"] = mid
    with redirect_stdout(StringIO()):
        return engine.calculate_layout(connectivity_of(u, v, scores), count, params)


class ThreadTests(unittest.TestCase):
    def test_the_coordinates_do_not_depend_on_the_number_of_threads(self):
        points, distance = synthetic_network()
        u, v, scores = strongest_edges(distance)
        base, _ = layout(u, v, scores, COUNT)
        with Numba_Threads.limited_threads(1):
            single, _ = layout(u, v, scores, COUNT)
        np.testing.assert_array_equal(base, single)

    def test_processes_of_one_and_three_threads_give_identical_bits(self):
        script = (
            "import hashlib, sys, numpy as np\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import numba\n"
            "import Layout_Engine_PaCMAP as engine\n"
            "from utilities import Numba_Threads\n"
            "rng = np.random.default_rng(5)\n"
            "count = 2500\n"
            "u = rng.integers(0, count, 25000); v = rng.integers(0, count, 25000); s = rng.random(25000)\n"
            "mid = np.column_stack((np.repeat(np.arange(count), 5), rng.integers(0, count, count * 5)))\n"
            "digests = []\n"
            "for dims in (2, 3):\n"
            "    pos, box = engine.calculate_layout(np.column_stack((u, v, s)).astype(np.float32), count,\n"
            "        {'PACMAP_NEIGHBORS': 10, 'LAYOUT_DIMENSIONS': dims, 'PACMAP_MID_NEAR': mid})\n"
            "    digests.append(hashlib.sha256(pos.tobytes()).hexdigest())\n"
            # Cliques of equal scores next to a ring: ties everywhere, and a graph with repeated eigenvalues.
            "ring = np.load(sys.argv[2])\n"
            "for dims in (2, 3):\n"
            "    pos, box = engine.calculate_layout(ring, int(ring[:, :2].max()) + 1,\n"
            "        {'PACMAP_NEIGHBORS': 5, 'LAYOUT_DIMENSIONS': dims})\n"
            "    digests.append(hashlib.sha256(pos.tobytes()).hexdigest())\n"
            "print('THREADS', Numba_Threads.default_thread_count(), numba.get_num_threads())\n"
            "print('DIGESTS', *digests)\n"
        )
        results = {}
        with tempfile.TemporaryDirectory() as folder:
            ring_file = os.path.join(folder, "ring.npy")
            np.save(ring_file, clique_ring()[0])
            for threads in ("1", "3"):
                environment = {**os.environ, "NUMBA_NUM_THREADS": threads, "PYTHONHASHSEED": "0"}
                environment.pop("SSN_LAYOUT_GPU_KERNELS", None)
                done = subprocess.run([sys.executable, "-c", script, str(SRC), ring_file], env=environment,
                                      capture_output=True, text=True, timeout=600)
                self.assertEqual(done.returncode, 0, done.stderr[-2000:])
                lines = {line.split()[0]: line.split()[1:] for line in done.stdout.splitlines()
                         if line.startswith(("THREADS", "DIGESTS"))}
                self.assertEqual(lines["THREADS"], [threads, threads])
                results[threads] = lines["DIGESTS"]
        self.assertEqual(results["1"], results["3"])
        self.assertEqual(len(set(results["1"])), 4)       # 2D and 3D, and the two networks, differ, of course

    def test_cliques_of_equal_scores_lay_out_the_same_every_time(self):
        # Nothing the engine draws comes from a generator that LAYOUT_SEED does not seed (it calls neither
        # eigsh nor any other routine with a random restart), so ties and repeated weights change nothing.
        connectivity, count = clique_ring()
        for dims in (2, 3):
            with self.subTest(dims=dims):
                params = {"PACMAP_NEIGHBORS": 5, "LAYOUT_DIMENSIONS": dims}
                with redirect_stdout(StringIO()):
                    first, _ = engine.calculate_layout(connectivity, count, params)
                    again, _ = engine.calculate_layout(connectivity, count, params)
                    with Numba_Threads.limited_threads(1):
                        single, _ = engine.calculate_layout(connectivity, count, params)
                np.testing.assert_array_equal(first, again)
                np.testing.assert_array_equal(first, single)
                self.assertTrue(np.isfinite(first).all())


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.distance = synthetic_network()
        cls.u, cls.v, cls.scores = strongest_edges(cls.distance)

    def test_the_layout_is_centred_and_scaled_into_the_box_like_the_umap_engine(self):
        for dims in (2, 3):
            with self.subTest(dims=dims):
                pos, box = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_DIMENSIONS": dims})
                self.assertEqual(pos.shape, (COUNT, dims))
                self.assertEqual(pos.dtype, np.float32)
                self.assertTrue(np.isfinite(pos).all())
                self.assertAlmostEqual(box, (math.sqrt(COUNT) * 2.5 + 5.0) * 2.0)
                self.assertAlmostEqual(float(np.ptp(pos, axis=0).max()), box * 1.6, delta=box * 1.6 * 1e-5)
                np.testing.assert_allclose((pos.max(axis=0) + pos.min(axis=0)) / 2, 0.0, atol=box * 1e-5)

    def test_the_box_scale_scales_the_box(self):
        _, narrow = layout(self.u, self.v, self.scores, COUNT, {"BOX_SCALE": 1.0})
        _, wide = layout(self.u, self.v, self.scores, COUNT, {"BOX_SCALE": 3.0})
        self.assertAlmostEqual(wide / narrow, 3.0)

    def test_seeds(self):
        a, _ = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": 42})
        b, _ = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": 42})
        c, _ = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": 43})
        d, _ = layout(self.u, self.v, self.scores, COUNT)       # the default is 42
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, d)
        self.assertFalse(np.array_equal(a, c))
        first, _ = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": None})
        second, _ = layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": None})
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.array_equal(first, second))          # null opts out of reproducibility
        with self.assertRaises(ValueError):
            layout(self.u, self.v, self.scores, COUNT, {"LAYOUT_SEED": -1})

    def test_the_mid_near_pairs_the_caller_found_are_used(self):
        blind = pairs.random_mid_near_pairs(np.arange(COUNT), NEIGHBORS, 42)
        scored = pairs.mid_near_pairs(stored_pairs(self.distance), None, COUNT, np.arange(COUNT), NEIGHBORS, 42)
        pos_scored, _ = layout(self.u, self.v, self.scores, COUNT, mid=scored)
        pos_blind, _ = layout(self.u, self.v, self.scores, COUNT, mid=blind)
        pos_default, _ = layout(self.u, self.v, self.scores, COUNT)
        self.assertFalse(np.array_equal(pos_scored, pos_blind))
        np.testing.assert_array_equal(pos_blind, pos_default)    # without them the partners are the blind ones

    def test_a_missing_pair_list_is_said_so(self):
        output = StringIO()
        with redirect_stdout(output):
            engine.calculate_layout(connectivity_of(self.u, self.v, self.scores), COUNT, {"PACMAP_NEIGHBORS": 10})
        self.assertIn("no stored pair scores were supplied", output.getvalue())
        output = StringIO()
        with redirect_stdout(output):
            engine.calculate_layout(connectivity_of(self.u, self.v, self.scores), COUNT,
                                    {"PACMAP_NEIGHBORS": 10, "PACMAP_MID_NEAR": np.zeros((0, 2), dtype=np.int32)})
        self.assertNotIn("no stored pair scores", output.getvalue())

    def test_invalid_settings(self):
        for params in ({"LAYOUT_DIMENSIONS": 4}, {"PACMAP_NEIGHBORS": -3}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                layout(self.u, self.v, self.scores, COUNT, params)


class SmallAndOddNetworkTests(unittest.TestCase):
    def test_an_empty_network_gets_a_seeded_random_layout_in_the_box(self):
        for dims in (2, 3):
            pos, box = layout(np.zeros(0), np.zeros(0), np.zeros(0), 50, {"LAYOUT_DIMENSIONS": dims})
            self.assertEqual(pos.shape, (50, dims))
            self.assertTrue(np.abs(pos).max() <= box / 2 + 1e-4)
            again, _ = layout(np.zeros(0), np.zeros(0), np.zeros(0), 50, {"LAYOUT_DIMENSIONS": dims})
            np.testing.assert_array_equal(pos, again)
            other, _ = layout(np.zeros(0), np.zeros(0), np.zeros(0), 50, {"LAYOUT_DIMENSIONS": dims, "LAYOUT_SEED": 1})
            self.assertFalse(np.array_equal(pos, other))

    def test_the_umap_engines_empty_layout_is_the_same(self):
        try:
            import Layout_Engine_UMAP as umap_engine
        except ImportError:
            self.skipTest("umap-learn is not installed")
        pos, box = layout(np.zeros(0), np.zeros(0), np.zeros(0), 30)
        with redirect_stdout(StringIO()):
            other, other_box = umap_engine.calculate_layout(np.zeros((0, 3), dtype=np.float32), 30, {"BOX_SCALE": 2.0})
        self.assertEqual(box, other_box)
        np.testing.assert_array_equal(pos, other)

    def test_a_network_of_self_loops_has_no_edges(self):
        pos, _ = layout(np.array([1, 2]), np.array([1, 2]), np.array([5.0, 6.0]), 4)
        self.assertEqual(pos.shape, (4, 2))
        self.assertTrue(np.isfinite(pos).all())

    def test_one_node_and_two_nodes(self):
        pos, _ = layout(np.zeros(0), np.zeros(0), np.zeros(0), 1)
        self.assertEqual(pos.shape, (1, 2))
        # Two nodes joined by one near pair only: the loss is least where they meet, so they end on top of each other.
        pos, box = layout(np.array([0]), np.array([1]), np.array([3.0]), 2)
        self.assertEqual(pos.shape, (2, 2))
        self.assertTrue(np.isfinite(pos).all())
        self.assertLessEqual(float(np.ptp(pos, axis=0).max()), box * 1.6 + 1e-3)

    def test_low_degree_nodes_and_networks_smaller_than_the_neighbour_count(self):
        # A path, a star and a triangle: nodes with one near partner, and fewer nodes than K.
        for u, v, count in (
            (np.arange(9), np.arange(1, 10), 10),
            (np.zeros(8, dtype=int), np.arange(1, 9), 9),
            (np.array([0, 1, 0]), np.array([1, 2, 2]), 3),
        ):
            for dims in (2, 3):
                with self.subTest(count=count, dims=dims):
                    pos, _ = layout(u, v, np.linspace(1, 2, len(u)), count, {"LAYOUT_DIMENSIONS": dims})
                    self.assertEqual(pos.shape, (count, dims))
                    self.assertTrue(np.isfinite(pos).all())

    def test_isolated_nodes_are_placed_like_the_umap_engines_and_stay_out_of_the_scaling(self):
        points, distance = synthetic_network()
        u, v, scores = strongest_edges(distance)
        count = COUNT + 6                          # six nodes with no edge at all
        pos, box = layout(u, v, scores, count)
        connected, isolated = pos[:COUNT], pos[COUNT:]
        self.assertAlmostEqual(float(np.ptp(connected, axis=0).max()), box * 1.6, delta=box * 1.6 * 1e-5)
        # The rule: a seeded uniform draw in the unit box around the middle, centred and scaled with the rest.
        draw = np.random.RandomState(42).uniform(-0.5, 0.5, (6, 2)).astype(np.float32)
        scale = (isolated[1:] - isolated[0]) / (draw[1:] - draw[0])
        np.testing.assert_allclose(scale, scale[0, 0], rtol=1e-3)
        self.assertGreater(scale[0, 0], 0)
        self.assertLess(np.abs(isolated).max(), 0.1 * box)
        # No pair involves them: with them added, the others lie where they would without them.
        again, _ = layout(u, v, scores, count, {"LAYOUT_SEED": 42})
        np.testing.assert_array_equal(pos, again)
        other, _ = layout(u, v, scores, count, {"LAYOUT_SEED": 43})
        self.assertFalse(np.array_equal(pos[COUNT:], other[COUNT:]))

    def test_isolated_nodes_do_not_enter_the_pair_sets(self):
        connectivity = connectivity_of(np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]), np.ones(4))
        ids, near, mid, far = engine.build_pairs(connectivity, 20, 3, 42, np.array([[0, 7], [3, 15], [1, 2]]))
        np.testing.assert_array_equal(ids, [0, 1, 2, 3, 4])
        self.assertEqual(len(near[0]), 6)
        # Only the mid-near row between two connected nodes is kept.
        self.assertEqual(mid[1].tolist(), [2, 1])
        self.assertTrue(all(0 <= other < 5 for other in far[1]))


if __name__ == "__main__":
    unittest.main()
