"""MainViewer's neighbour lookup for the selection (adjacency index), the
neighbour counts it keeps between selections, and selection validation."""
import os
import sys
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Viewer
from EMAPSSN_Viewer import MainViewer


def scan_every_edge_for_neighbors(edges, selected, node_count):
    """Neighbour discovery by full edge scan, kept as the reference result."""
    selected = np.asarray(selected)
    if len(selected) == 0 or len(edges) == 0:
        return np.empty(0, dtype=np.int32)
    touches_selection = np.isin(edges[:, 0], selected) | np.isin(
        edges[:, 1], selected
    )
    connected = np.unique(edges[touches_selection].reshape(-1))
    connected = connected[~np.isin(connected, selected)]
    return connected.astype(np.int32, copy=False)


class AdjacencyIndexTests(unittest.TestCase):
    def make_viewer(self, node_count, edges):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = node_count
        viewer.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        return viewer

    def test_matches_full_edge_scan_on_random_graphs(self):
        for node_count, edge_count, seed in (
            (40, 120, 0),
            (200, 900, 1),
            (500, 4000, 2),
        ):
            rng = np.random.default_rng(seed)
            edges = rng.integers(
                0, node_count, size=(edge_count, 2)
            ).astype(np.int32)
            viewer = self.make_viewer(node_count, edges)
            for selection_size in (1, 3, node_count // 4, node_count):
                selected = np.sort(
                    rng.choice(node_count, size=selection_size, replace=False)
                ).astype(np.int32)
                expected = scan_every_edge_for_neighbors(
                    edges, selected, node_count
                )
                with self.subTest(nodes=node_count, selected=selection_size):
                    np.testing.assert_array_equal(
                        viewer._connected_to_selected_indices(selected),
                        expected,
                    )
                    # The repeat query is served from the neighbour cache.
                    np.testing.assert_array_equal(
                        viewer._connected_to_selected_indices(selected),
                        expected,
                    )

    def test_duplicate_and_self_edges_do_not_change_the_result(self):
        viewer = self.make_viewer(5, [[0, 1], [0, 1], [1, 0], [2, 2], [1, 3]])
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([0], dtype=np.int32)),
            [1],
        )
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([2], dtype=np.int32)),
            [],
        )

    def test_selected_nodes_are_never_returned_as_neighbours(self):
        viewer = self.make_viewer(4, [[0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(
                np.array([1, 2], dtype=np.int32)
            ),
            [0, 3],
        )

    def test_empty_topology_and_empty_selection_return_no_neighbours(self):
        viewer = self.make_viewer(4, np.empty((0, 2), dtype=np.int32))
        self.assertEqual(
            len(viewer._connected_to_selected_indices(np.array([0, 1]))), 0
        )
        viewer = self.make_viewer(4, [[0, 1]])
        self.assertEqual(
            len(
                viewer._connected_to_selected_indices(
                    np.empty(0, dtype=np.int32)
                )
            ),
            0,
        )

    def test_out_of_range_endpoints_are_ignored(self):
        viewer = self.make_viewer(3, [[0, 7], [0, 1], [-2, 0]])
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([0], dtype=np.int32)),
            [1],
        )

    def test_index_is_built_once_and_rebuilt_when_the_topology_changes(self):
        viewer = self.make_viewer(4, [[0, 1], [1, 2]])
        builder = mock.Mock(wraps=EMAPSSN_Viewer._build_adjacency_index)
        with mock.patch.object(
            EMAPSSN_Viewer, "_build_adjacency_index", builder
        ):
            viewer._connected_to_selected_indices(np.array([0], dtype=np.int32))
            viewer._connected_to_selected_indices(np.array([2], dtype=np.int32))
            self.assertEqual(builder.call_count, 1)

            viewer.edges = np.array([[0, 3]], dtype=np.int32)
            np.testing.assert_array_equal(
                viewer._connected_to_selected_indices(
                    np.array([0], dtype=np.int32)
                ),
                [3],
            )
            self.assertEqual(builder.call_count, 2)


class IncrementalNeighbourTests(unittest.TestCase):
    """The neighbour counts follow the selection from one call to the next,
    as a box selection changes it on every mouse move."""

    def make_viewer(self, node_count, edges):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = node_count
        viewer.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
        return viewer

    def counted_nodes(self, viewer, selected):
        """Query once; return the result and the nodes each count step gathered."""
        original = EMAPSSN_Viewer._SelectionNeighbours._count
        with mock.patch.object(
            EMAPSSN_Viewer._SelectionNeighbours, "_count", autospec=True,
            side_effect=original,
        ) as count:
            result = viewer._connected_to_selected_indices(
                np.asarray(selected, dtype=np.int32)
            )
        gathered = [
            (sorted(call.args[1].tolist()), call.args[2])
            for call in count.call_args_list
            if len(call.args[1])
        ]
        return result, gathered

    def test_a_selection_walk_matches_the_full_edge_scan_at_every_step(self):
        rng = np.random.default_rng(42)
        node_count = 300
        edges = rng.integers(0, node_count, size=(3000, 2)).astype(np.int32)
        edges[:100, 1] = edges[:100, 0]  # Self edges.
        edges[100:200] = edges[200:300]  # Duplicates.
        viewer = self.make_viewer(node_count, edges)
        selection = set()
        for step in range(300):
            draw = rng.random()
            if draw < 0.1:
                # A new selection, unrelated to the last one (or empty).
                size = int(rng.integers(0, node_count + 1))
                selection = set(rng.choice(node_count, size, replace=False).tolist())
            elif draw < 0.55:
                selection |= set(rng.integers(0, node_count, 5).tolist())
            else:
                selection -= set(rng.integers(0, node_count, 5).tolist())
            selected = np.array(sorted(selection), dtype=np.int32)
            with self.subTest(step=step):
                np.testing.assert_array_equal(
                    viewer._connected_to_selected_indices(selected),
                    scan_every_edge_for_neighbors(edges, selected, node_count),
                )

    def test_a_grown_or_shrunk_selection_counts_only_the_changed_nodes(self):
        # A path 0-1-2-...-19: every node has at most two neighbours.
        viewer = self.make_viewer(20, [[i, i + 1] for i in range(19)])
        viewer._connected_to_selected_indices(np.arange(10, dtype=np.int32))

        result, gathered = self.counted_nodes(viewer, np.arange(11))
        np.testing.assert_array_equal(result, [11])
        self.assertEqual(gathered, [([10], 1)])

        result, gathered = self.counted_nodes(viewer, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, [0, 11])
        self.assertEqual(gathered, [([0], -1)])

        result, gathered = self.counted_nodes(viewer, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_array_equal(result, [0, 11])
        self.assertEqual(gathered, [])

    def test_a_new_selection_is_counted_afresh(self):
        viewer = self.make_viewer(20, [[i, i + 1] for i in range(19)])
        viewer._connected_to_selected_indices(np.arange(10, dtype=np.int32))

        result, gathered = self.counted_nodes(viewer, [15])
        np.testing.assert_array_equal(result, [14, 16])
        self.assertEqual(gathered, [([15], 1)])

        result, gathered = self.counted_nodes(viewer, [])
        self.assertEqual(len(result), 0)
        self.assertEqual(gathered, [])

    def test_a_replaced_topology_starts_the_counts_afresh(self):
        viewer = self.make_viewer(4, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([1], dtype=np.int32)),
            [0, 2],
        )
        viewer.edges = np.array([[1, 3]], dtype=np.int32)
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([1], dtype=np.int32)),
            [3],
        )

    def test_returned_arrays_are_copies(self):
        viewer = self.make_viewer(4, [[0, 1], [1, 2]])
        first = viewer._connected_to_selected_indices(np.array([1], dtype=np.int32))
        first[:] = 3
        np.testing.assert_array_equal(
            viewer._connected_to_selected_indices(np.array([1], dtype=np.int32)),
            [0, 2],
        )


class SelectionValidationTests(unittest.TestCase):
    def test_indices_come_back_unique_ascending_in_range_int32(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = 6
        for values, expected in (
            (None, []),
            ([], []),
            ([4, 1, 4, 0, 1], [0, 1, 4]),
            # -1 and -6 would wrap to nodes 5 and 0 if let through.
            ([3, -1, 6, 2, 99, -6], [2, 3]),
            ({3, 1}, [1, 3]),
            (np.array([[2, 2], [0, 5]], dtype=np.int32), [0, 2, 5]),
            (np.array([True, False]), [0, 1]),
        ):
            with self.subTest(values=values):
                result = viewer._valid_node_indices(values)
                self.assertEqual(result.dtype, np.int32)
                np.testing.assert_array_equal(result, expected)

    def test_an_empty_network_has_no_valid_indices(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = 0
        self.assertEqual(len(viewer._valid_node_indices([0, 1])), 0)


if __name__ == "__main__":
    unittest.main()
