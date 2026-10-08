"""Component search, edge grouping and grid packing in the SSN layout engine.

The reference functions restate the per-edge Python code these replaced (as of
commit 0704907), with the spectral start's later axis rule and asinh spread.
The simulation must keep seeing exactly the same inputs, so the component
order, the batch preparation and the progressive-stage anchors are compared
bit for bit. Packing is new, so it is checked by what it
guarantees: the clearance between components, rigid moves and tight cells.
"""

import ast
import io
import math
import os
import sys
import tracemalloc
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    import Layout_Engine_SSN as ssn_engine
    from utilities import Hardware_Acceleration as Layout_Hardware


# ---------------------------------------------------------------- references

def _reference_components(n_nodes, edges):
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = np.zeros(n_nodes, dtype=bool)
    components = []

    for i in range(n_nodes):
        if not visited[i]:
            comp = []
            q = [i]
            visited[i] = True
            while q:
                curr = q.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        q.append(neighbor)
            components.append(comp)
    return components


def _reference_progressive_stage(
    pos, stage_edges, stage_scores, previous_active, rng=None
):
    active_mask = np.zeros(len(pos), dtype=np.bool_)
    for u, v in stage_edges:
        active_mask[u] = True
        active_mask[v] = True

    newly_active = active_mask & (~previous_active)
    if not np.any(newly_active) or not np.any(previous_active):
        return active_mask

    reference_pos = pos.copy()
    weighted_sum = np.zeros_like(pos, dtype=np.float64)
    weight_sum = np.zeros(len(pos), dtype=np.float64)

    for (u, v), score in zip(stage_edges, stage_scores):
        weight = max(float(score), 1e-9)
        if newly_active[u] and previous_active[v]:
            weighted_sum[u] += reference_pos[v] * weight
            weight_sum[u] += weight
        if newly_active[v] and previous_active[u]:
            weighted_sum[v] += reference_pos[u] * weight
            weight_sum[v] += weight

    anchored_nodes = np.flatnonzero(newly_active & (weight_sum > 0.0))
    if len(anchored_nodes) > 0:
        pos[anchored_nodes] = (
            weighted_sum[anchored_nodes]
            / weight_sum[anchored_nodes, None]
        ).astype(np.float32)
        draw = np.random.normal if rng is None else rng.normal
        pos[anchored_nodes] += draw(
            0.0, 0.05, (len(anchored_nodes), pos.shape[1])
        ).astype(np.float32)

    return active_mask


def _reference_layout_batch(
    batch_components, node_to_component, component_edges, component_scores,
    params, *, add_noise=True, rng=None,
):
    """The per-edge preparation, returning (global_nodes, edges, scores, positions, labels, limits)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    dimensions = int(params.get("LAYOUT_DIMENSIONS", 2) or 2)
    global_nodes = [node for component in batch_components for node in component]
    global_to_batch = {
        global_id: local_id for local_id, global_id in enumerate(global_nodes)
    }
    batch_edges, batch_scores = [], []
    position_blocks, label_blocks, box_limit_blocks = [], [], []

    for batch_component_index, component in enumerate(batch_components):
        component_node_count = len(component)
        component_index = node_to_component[component[0]]
        edges = component_edges[component_index]
        scores = component_scores[component_index]
        global_to_local = {
            global_id: local_id for local_id, global_id in enumerate(component)
        }
        local_edges = [
            (global_to_local[source], global_to_local[target])
            for source, target in edges
        ]
        box_limit = (
            np.sqrt(component_node_count) * 2.5 + 5.0
        ) * params.get("BOX_SCALE", 1.0)
        local_positions = None
        spectral_success = False
        if component_node_count >= dimensions + 2:
            try:
                row = [edge[0] for edge in local_edges] + [edge[1] for edge in local_edges]
                col = [edge[1] for edge in local_edges] + [edge[0] for edge in local_edges]
                data = list(scores) + list(scores)
                adjacency = sp.coo_matrix(
                    (data, (row, col)),
                    shape=(component_node_count, component_node_count),
                )
                graph_laplacian = laplacian(adjacency, normed=True)
                eigsh_v0 = None if rng is None else rng.standard_normal(component_node_count)
                values, vectors = eigsh(
                    graph_laplacian, k=dimensions + 1, which="SM", tol=1e-3, v0=eigsh_v0,
                )
                # The axes skip the trivial eigenvector, sqrt(degree), wherever
                # ARPACK put it, and take the next ones by eigenvalue.
                degree = [0.0] * component_node_count
                for (source, target), score in zip(local_edges, scores):
                    if source != target:
                        degree[source] += score
                        degree[target] += score
                root_degree = np.sqrt(degree)
                axes = [
                    vectors[:, index] for index in np.argsort(values, kind="stable")
                    if abs(root_degree @ vectors[:, index])
                    <= 0.9 * np.linalg.norm(root_degree) * np.linalg.norm(vectors[:, index])
                ][:dimensions]
                axis_blocks = []
                for coordinates in axes:
                    # asinh around the median, in units of half the 5-95% range
                    # (half the full range when that is numerically empty).
                    low, middle, high = np.percentile(coordinates, [5.0, 50.0, 95.0])
                    spread = (high - low) / 2.0
                    if not spread > np.ptp(coordinates) / 2.0 * 1e-6:
                        spread = float(np.ptp(coordinates)) / 2.0
                    scaled = np.arcsinh((coordinates - middle) / spread)
                    normalized = (scaled - np.min(scaled)) / (np.ptp(scaled) + 1e-9)
                    axis_blocks.append((normalized - 0.5) * box_limit * 0.8)
                local_positions = np.column_stack(axis_blocks).astype(np.float32)
                spectral_success = True
            except Exception:
                pass
        if not spectral_success:
            side = max(int(np.ceil(component_node_count ** (1.0 / dimensions))), 1)
            axis = np.linspace(-box_limit * 0.5, box_limit * 0.5, side)
            mesh = np.meshgrid(*([axis] * dimensions))
            grid = np.column_stack([block.flatten() for block in mesh])
            if grid.shape[0] < component_node_count:
                repeats = int(np.ceil(component_node_count / grid.shape[0]))
                grid = np.tile(grid, (repeats, 1))
            local_positions = grid[:component_node_count].astype(np.float32)

        local_minimum = np.min(local_positions, axis=0)
        local_maximum = np.max(local_positions, axis=0)
        local_positions -= (local_minimum + local_maximum) / 2.0
        position_blocks.append(local_positions)
        label_blocks.append(
            np.full(component_node_count, batch_component_index, dtype=np.int32)
        )
        box_limit_blocks.append(
            np.full(component_node_count, box_limit, dtype=np.float32)
        )
        for (source, target), score in zip(edges, scores):
            batch_edges.append((global_to_batch[source], global_to_batch[target]))
            batch_scores.append(float(score))

    positions = np.vstack(position_blocks).astype(np.float32)
    if add_noise:
        noise = np.random.normal(0, 0.1, positions.shape) if rng is None else rng.normal(0, 0.1, positions.shape)
        positions += noise.astype(np.float32)
    return (
        global_nodes, batch_edges, batch_scores, positions,
        np.concatenate(label_blocks), np.concatenate(box_limit_blocks),
    )


# ---------------------------------------------------------------- fixtures

def _clustered_graph(rng, sizes, extra_edges=3, isolated=5):
    """Shuffled nodes split into clusters, with self-loops, duplicates and isolated nodes."""
    n_nodes = sum(sizes) + isolated
    labels = rng.permutation(n_nodes)
    edges, start = [], 0
    for size in sizes:
        members = labels[start:start + size]
        start += size
        for k in range(1, size):                       # a random spanning tree
            edges.append((members[k], members[rng.integers(0, k)]))
        for _ in range(extra_edges * size):
            edges.append(tuple(rng.choice(members, 2)))
        if size:
            edges.append((members[0], members[0]))     # self-loop
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    edges = np.vstack((edges, edges[: len(edges) // 10]))  # duplicates
    return n_nodes, edges[rng.permutation(len(edges))]


def _scattered_layout(rng, n_components=25, singletons=15, spread=200.0):
    """Random components drawn on top of one another, as a simulation leaves them."""
    blocks, edges, offset = [], [], 0
    for _ in range(n_components):
        size = int(rng.integers(2, 40))
        centre = rng.uniform(-spread, spread, 2)
        radius = rng.uniform(1.0, 60.0)
        blocks.append(centre + rng.uniform(-radius, radius, (size, 2)))
        for k in range(1, size):
            edges.append((offset + k, offset + int(rng.integers(0, k))))
        for _ in range(size):
            edges.append(tuple(offset + rng.integers(0, size, 2)))
        offset += size
    blocks.append(rng.uniform(-spread, spread, (singletons, 2)))
    pos = np.vstack(blocks).astype(np.float32)
    return pos, np.asarray(edges, dtype=np.int32)


def _pack(pos, edges, grid_size, padding, geometry="Square"):
    with redirect_stdout(io.StringIO()):
        return ssn_engine.pack_components_to_grid(
            pos, edges, len(pos), grid_size, padding, geometry
        )


def _drawing_samples(pos, edges, per_edge=24):
    """Points along every drawn edge, plus the nodes, with the node each one came from."""
    t = np.linspace(0.0, 1.0, per_edge)[None, :, None]
    a = pos[edges[:, 0]][:, None, :]
    b = pos[edges[:, 1]][:, None, :]
    along = (a + t * (b - a)).reshape(-1, 2)
    owners = np.repeat(edges[:, 0], per_edge)
    return (
        np.vstack((pos.astype(np.float64), along)),
        np.concatenate((np.arange(len(pos)), owners)),
    )


# ---------------------------------------------------------------- tests

class ComponentSearchTests(unittest.TestCase):
    def test_order_matches_the_original_search(self):
        for seed in range(6):
            rng = np.random.default_rng(seed)
            sizes = [int(s) for s in rng.integers(1, 60, size=12)]
            n_nodes, edges = _clustered_graph(rng, sizes)
            expected = _reference_components(n_nodes, edges)
            for given in (edges, [tuple(edge) for edge in edges.tolist()]):
                found = ssn_engine.find_connected_components(n_nodes, given)
                self.assertEqual([component.tolist() for component in found], expected)

    def test_graph_without_edges(self):
        self.assertEqual(ssn_engine.find_connected_components(0, []), [])
        found = ssn_engine.find_connected_components(3, np.zeros((0, 2), dtype=np.int32))
        self.assertEqual([component.tolist() for component in found], [[0], [1], [2]])

    def test_edges_outside_the_node_range_are_rejected(self):
        for edges in ([[0, 3]], [[-1, 0]]):
            with self.assertRaises(IndexError):
                ssn_engine.find_connected_components(3, np.asarray(edges))

    def test_component_labels_follow_component_order(self):
        labels = ssn_engine.get_component_labels(5, np.array([[3, 4], [1, 0]]))
        np.testing.assert_array_equal(labels, [0, 0, 1, 2, 2])

    def test_grouping_matches_a_stable_sort(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(-1, 6, size=5000)
        order, bounds = ssn_engine._stable_group_order(labels, 6)
        kept = np.flatnonzero(labels >= 0)
        expected = kept[np.argsort(labels[kept], kind="stable")]
        np.testing.assert_array_equal(order, expected)
        np.testing.assert_array_equal(np.diff(bounds), np.bincount(labels[kept], minlength=6))


class ProgressiveStageTests(unittest.TestCase):
    def test_anchors_match_the_original_loop_bit_for_bit(self):
        for dtype in (np.float32, np.float64):
            rng = np.random.default_rng(11)
            positions = (rng.standard_normal((800, 2)) * 40).astype(dtype)
            edges = rng.integers(0, 800, size=(6000, 2))
            edges[:50, 1] = edges[:50, 0]
            scores = rng.random(6000) * 2.0 - 0.3       # some fall to the 1e-9 floor
            previous = rng.random(800) < 0.5

            expected_pos = positions.copy()
            expected = _reference_progressive_stage(
                expected_pos, [tuple(edge) for edge in edges], list(scores), previous,
                rng=np.random.default_rng(4),
            )
            found_pos = positions.copy()
            found = ssn_engine._prepare_progressive_stage(
                found_pos, edges.astype(np.int32), scores, previous,
                rng=np.random.default_rng(4),
            )
            np.testing.assert_array_equal(found, expected)
            self.assertEqual(found_pos.dtype, expected_pos.dtype)
            np.testing.assert_array_equal(found_pos, expected_pos)

    def test_score_count_must_match_the_edges(self):
        positions = np.zeros((3, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            ssn_engine._prepare_progressive_stage(
                positions, [(0, 1), (1, 2)], [1.0], np.array([True, False, False])
            )


class LayoutBatchTests(unittest.TestCase):
    def _inputs(self, rng):
        sizes = [3, 2, 7, 40, 1, 5, 120]
        n_nodes, edges = _clustered_graph(rng, sizes, isolated=0)
        scores = rng.random(len(edges)).astype(np.float32) + 0.5
        components = [
            component for component in ssn_engine.find_connected_components(n_nodes, edges)
            if len(component) > 1
        ]
        node_to_component = np.full(n_nodes, -1, dtype=np.int64)
        for index, component in enumerate(components):
            node_to_component[component] = index
        member = node_to_component[edges[:, 0]]
        array_edges = {i: edges[member == i] for i in range(len(components))}
        array_scores = {i: scores[member == i] for i in range(len(components))}
        list_edges = {i: [tuple(edge) for edge in array_edges[i]] for i in array_edges}
        list_scores = {i: list(array_scores[i]) for i in array_scores}
        dict_map = {int(node): int(index) for node, index in enumerate(node_to_component) if index >= 0}
        list_components = [component.tolist() for component in components]
        return (
            (list_components, dict_map, list_edges, list_scores),
            (components, node_to_component, array_edges, array_scores),
        )

    def test_batch_matches_the_original_preparation(self):
        for dimensions in (2, 3):
            params = {"BOX_SCALE": 2.0, "LAYOUT_DIMENSIONS": dimensions}
            legacy_inputs, array_inputs = self._inputs(np.random.default_rng(dimensions))
            expected = _reference_layout_batch(
                *legacy_inputs, params, rng=np.random.default_rng(8)
            )
            for inputs in (legacy_inputs, array_inputs):
                with redirect_stdout(io.StringIO()):
                    found = Layout_Hardware.prepare_layout_batch(
                        *inputs, params, rng=np.random.default_rng(8)
                    )
                np.testing.assert_array_equal(found.global_nodes, expected[0])
                np.testing.assert_array_equal(found.edges, np.asarray(expected[1]))
                self.assertEqual(found.scores.dtype, np.float64)
                np.testing.assert_array_equal(found.scores, np.asarray(expected[2]))
                np.testing.assert_array_equal(found.positions, expected[3])
                np.testing.assert_array_equal(found.component_labels, expected[4])
                np.testing.assert_array_equal(found.box_limits, expected[5])
                self.assertEqual(found.node_count, len(expected[0]))

    def test_an_edge_outside_its_component_is_rejected(self):
        with self.assertRaises(KeyError):
            Layout_Hardware.prepare_layout_batch(
                [[0, 1], [2, 3]], {0: 0, 1: 0, 2: 1, 3: 1},
                {0: [(0, 2)], 1: [(2, 3)]}, {0: [1.0], 1: [1.0]},
                {"BOX_SCALE": 1.0}, add_noise=False, verbose=False,
            )


class FootprintTests(unittest.TestCase):
    def test_a_reach_ending_on_a_cell_border_claims_one_cell(self):
        mask = ssn_engine._component_footprint(
            np.zeros((1, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int64), 5.0, 1.0
        )
        np.testing.assert_array_equal(mask, [[True]])

    def test_a_diagonal_edge_claims_only_the_cells_it_crosses(self):
        shifted = np.array([[0.0, 0.0], [30.0, -30.0]], dtype=np.float32)
        mask = ssn_engine._component_footprint(shifted, np.array([[0, 1]]), 10.0, 0.0)
        np.testing.assert_array_equal(mask, np.eye(4, dtype=bool))

    def test_footprint_covers_every_drawn_point_and_its_clearance(self):
        rng = np.random.default_rng(21)
        cell = 10.0
        for padding in (0.0, 7.0, 10.0, 20.0):
            reach = padding / 2.0 / cell
            for _ in range(40):
                size = int(rng.integers(1, 30))
                shifted = rng.uniform(0.0, 300.0, (size, 2)).astype(np.float32)
                shifted -= [shifted[:, 0].min(), shifted[:, 1].max()]
                edges = rng.integers(0, size, (size * 2, 2))
                mask = ssn_engine._component_footprint(shifted, edges, cell, reach)
                points, _ = _drawing_samples(shifted.astype(np.float64), edges, per_edge=40)
                u = points[:, 0] / cell
                v = -points[:, 1] / cell
                # Every corner of each point's clearance box, pulled just inside.
                inset = max(reach - 1e-6, 0.0)
                for du in (-inset, 0.0, inset):
                    for dv in (-inset, 0.0, inset):
                        cols = np.floor(u + du).astype(int)
                        rows = np.floor(v + dv).astype(int)
                        inside = (cols >= 0) & (rows >= 0)
                        self.assertTrue(
                            mask[np.minimum(rows[inside], mask.shape[0] - 1),
                                 np.minimum(cols[inside], mask.shape[1] - 1)].all()
                        )

    def test_long_edges_need_memory_for_the_raster_only(self):
        rng = np.random.default_rng(2)
        shifted = rng.uniform(0.0, 2000.0, (2000, 2)).astype(np.float32)
        shifted -= [shifted[:, 0].min(), shifted[:, 1].max()]
        edges = rng.integers(0, 2000, (5000, 2))
        ssn_engine._component_footprint(shifted[:3], edges[:1] % 3, 10.0, 0.5)
        tracemalloc.start()
        try:
            mask = ssn_engine._component_footprint(shifted, edges, 10.0, 0.5)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(mask.shape, (201, 201))
        # Sampling a point every 2.5 units would hold about 4 million points here.
        self.assertLess(peak, 2 * 1024 * 1024)


class GridPackingTests(unittest.TestCase):
    def test_components_keep_their_clearance(self):
        for geometry in ("Square", "Circle"):
            for seed, (grid_size, padding) in enumerate(((10.0, 10.0), (4.0, 10.0), (10.0, 3.0))):
                rng = np.random.default_rng(seed)
                pos, edges = _scattered_layout(rng)
                packed, _ = _pack(pos, edges, grid_size, padding, geometry)
                labels = ssn_engine.get_component_labels(len(pos), edges)
                points, owners = _drawing_samples(packed.astype(np.float64), edges)
                pairs = cKDTree(points).query_pairs(
                    padding / 2.0 - 1e-3, p=np.inf, output_type="ndarray"
                )
                crossing = labels[owners[pairs[:, 0]]] != labels[owners[pairs[:, 1]]]
                self.assertEqual(int(crossing.sum()), 0, (geometry, grid_size, padding))

    def test_components_move_without_turning(self):
        pos, edges = _scattered_layout(np.random.default_rng(7))
        packed, _ = _pack(pos, edges, 10.0, 10.0)
        for component in ssn_engine.find_connected_components(len(pos), edges):
            shift = packed[component] - pos[component]
            np.testing.assert_allclose(shift, shift[:1].repeat(len(component), 0), atol=1e-3)

    def test_single_nodes_sit_one_clearance_apart(self):
        pos = np.random.default_rng(1).uniform(-100, 100, (9, 2)).astype(np.float32)
        for geometry in ("Square", "Circle"):
            packed, box_limit = _pack(pos, np.zeros((0, 2), dtype=np.int32), 5.0, 10.0, geometry)
            np.testing.assert_allclose(packed.max(axis=0) - packed.min(axis=0), [10.0, 10.0])
            self.assertAlmostEqual(float(box_limit), 5.5, places=5)

    def test_circle_fills_a_disc(self):
        pos = np.random.default_rng(1).uniform(-100, 100, (300, 2)).astype(np.float32)
        no_edges = np.zeros((0, 2), dtype=np.int32)
        square, _ = _pack(pos, no_edges, 5.0, 10.0, "Square")
        circle, _ = _pack(pos, no_edges, 5.0, 10.0, "Circle")
        disc_radius = (math.sqrt(300 / math.pi) + 1.0) * 5.0
        self.assertLess(float(np.linalg.norm(circle, axis=1).max()), disc_radius)
        self.assertGreater(float(np.linalg.norm(square, axis=1).max()), disc_radius)

    def test_repeated_packing_is_identical(self):
        pos, edges = _scattered_layout(np.random.default_rng(5))
        first, first_limit = _pack(pos, edges, 10.0, 10.0)
        second, second_limit = _pack(pos.copy(), edges.copy(), 10.0, 10.0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_limit, second_limit)

    def test_non_finite_positions_are_rejected(self):
        pos = np.array([[0.0, 0.0], [np.nan, 1.0]], dtype=np.float32)
        with self.assertRaises(ValueError):
            _pack(pos, np.array([[0, 1]], dtype=np.int32), 10.0, 10.0)

    def test_scattered_component_packs_in_raster_sized_memory(self):
        rng = np.random.default_rng(9)
        pos = rng.uniform(-1000.0, 1000.0, (3000, 2)).astype(np.float32)
        edges = np.column_stack((np.arange(1, 3000), rng.integers(0, 3000, 2999)))
        edges = np.vstack((edges, rng.integers(0, 3000, (3000, 2)))).astype(np.int32)
        _pack(pos[:3], edges[:1] % 3, 10.0, 10.0)
        tracemalloc.start()
        try:
            _pack(pos, edges, 10.0, 10.0)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        # The edges average about 1,000 units: a sample every 2.5 units would
        # hold about 2.4 million points.
        self.assertLess(peak, 16 * 1024 * 1024)


class PackingDefaultTests(unittest.TestCase):
    FILES = (
        "EMAPSSN_Config.py",
        os.path.join("desktop", "Viewer_State.py"),
        "Layout_Cache_Generator.py",
        os.path.join("mcp_server", "pipeline", "Pipeline_Operations.py"),
        "Layout_Engine_SSN.py",
    )

    def test_every_packing_grid_default_is_ten(self):
        found = []
        for relative in self.FILES:
            with open(os.path.join(SRC_DIR, relative), encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "PACKING_GRID_SIZE"
                    for target in node.targets
                ):
                    found.append((relative, node.value))
                elif isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value == "PACKING_GRID_SIZE":
                            found.append((relative, value))
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "PACKING_GRID_SIZE"
                ):
                    found.append((relative, node.args[1]))
        defaults = [
            (relative, value.value) for relative, value in found
            if isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
        ]
        self.assertGreaterEqual(len(defaults), 7)
        self.assertEqual({value for _, value in defaults}, {10.0}, defaults)


if __name__ == "__main__":
    unittest.main()
