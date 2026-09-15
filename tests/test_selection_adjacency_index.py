import os
import sys
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg
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


class FakeLine:
    def __init__(self):
        self.visible = True
        self.upload_count = 0
        self.last_pos = None

    def set_data(self, pos=None, **kwargs):
        self.upload_count += 1
        self.last_pos = pos


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


class EdgeGeometryUploadTests(unittest.TestCase):
    def make_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = 4
        viewer.edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        viewer.edge_scores = np.array([0.9, 0.9, 0.1], dtype=np.float32)
        viewer.pos = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
        )
        viewer.visible_mask = np.ones(4, dtype=bool)
        viewer.current_slider_threshold = 0.5
        viewer.selected_indices = []
        viewer.line_visual = FakeLine()
        viewer._uploaded_active_edges = None
        viewer._uploaded_edge_positions = None
        return viewer

    def test_repeated_updates_upload_edge_geometry_once(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        self.assertEqual(viewer.line_visual.upload_count, 1)

        # A selection change alters neither the drawn segments nor the node
        # positions, so the vertex buffer must not be resent.
        for selection in ([0], [0, 2], [1], []):
            viewer.selected_indices = selection
            viewer.update_edges()
        self.assertEqual(viewer.line_visual.upload_count, 1)

    def test_moving_nodes_uploads_edge_geometry(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        viewer.pos[1, 1] = 5.0
        viewer.update_edges()
        self.assertEqual(viewer.line_visual.upload_count, 2)
        np.testing.assert_allclose(viewer.line_visual.last_pos[1], [1.0, 5.0])

    def test_threshold_change_uploads_edge_geometry(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        self.assertEqual(len(viewer.line_visual.last_pos), 4)

        viewer.current_slider_threshold = 0.0
        viewer.update_edges()
        self.assertEqual(viewer.line_visual.upload_count, 2)
        self.assertEqual(len(viewer.line_visual.last_pos), 6)

    def test_hiding_nodes_uploads_edge_geometry(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        viewer.visible_mask[0] = False
        viewer.update_edges()
        self.assertEqual(viewer.line_visual.upload_count, 2)
        self.assertEqual(len(viewer.line_visual.last_pos), 2)

    def test_selection_filtered_mode_still_uploads_on_selection_change(self):
        viewer = self.make_viewer()
        viewer.selected_indices = [0]
        with mock.patch.object(cfg, "UMAP_MODE", True):
            viewer.update_edges()
            first = viewer.line_visual.upload_count
            viewer.selected_indices = [2]
            viewer.update_edges()
        self.assertGreater(viewer.line_visual.upload_count, first)


if __name__ == "__main__":
    unittest.main()
