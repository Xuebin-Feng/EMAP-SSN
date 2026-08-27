import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Command_Engine
import SSN_Config as cfg
from SSN_Viewer import MainViewer
from commands import color as color_command
from commands import spectrum as spectrum_command


class FakeMarkers:
    def __init__(self):
        self.visible = True
        self.data = None

    def set_data(self, **kwargs):
        self.data = kwargs

    def set_gl_state(self, *args, **kwargs):
        pass


class IdentityTransform:
    inverse = None

    def __init__(self):
        self.inverse = self

    def map(self, value):
        array = np.asarray(value, dtype=float)
        if array.size == 2:
            return np.array([array[0], array[1], 0.0])
        return array


class NodeRenderOrderTests(unittest.TestCase):
    def make_viewer(self, node_count=7):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = node_count
        viewer.node_render_order = np.arange(node_count, dtype=np.int32)
        viewer.visible_mask = np.ones(node_count, dtype=bool)
        viewer.selected_indices = []
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        viewer.edges = np.empty((0, 2), dtype=np.int32)
        return viewer

    def make_renderable_viewer(self, node_count=3):
        viewer = self.make_viewer(node_count)
        viewer.pos = np.column_stack(
            (np.arange(node_count), np.zeros(node_count))
        ).astype(np.float32)
        viewer.current_colors = np.zeros((node_count, 4), dtype=np.float32)
        viewer.current_sizes = np.ones(node_count, dtype=np.float32)
        viewer.current_shapes = np.full(node_count, "disc", dtype=object)
        viewer.markers = FakeMarkers()
        viewer.canvas = SimpleNamespace(update=mock.Mock())
        viewer._update_hud_elements = mock.Mock()
        return viewer

    def test_promotions_create_latest_index_sorted_group(self):
        viewer = self.make_viewer(6)

        viewer.promote_nodes([4, 1])
        np.testing.assert_array_equal(
            viewer.node_render_order, [0, 2, 3, 5, 1, 4]
        )

        viewer.promote_nodes(np.array([True, False, True, False, False, False]))
        np.testing.assert_array_equal(
            viewer.node_render_order, [3, 5, 1, 4, 0, 2]
        )

        viewer.promote_nodes([2, 1])
        np.testing.assert_array_equal(
            viewer.node_render_order, [3, 5, 4, 0, 1, 2]
        )

    def test_effective_order_separates_selected_and_left_click_tiers(self):
        viewer = self.make_viewer(7)
        viewer.node_render_order = np.array([6, 5, 4, 3, 2, 1, 0])
        viewer.selected_indices = [0]
        viewer.selected_node_idx = 2
        viewer.left_click_highlight_indices = [5]
        viewer.edges = np.array([[0, 3], [2, 4], [5, 1]], dtype=np.int32)

        np.testing.assert_array_equal(
            viewer.visible_node_render_order(),
            [6, 4, 1, 3, 0, 2, 5],
        )

        viewer.selected_indices = []
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        np.testing.assert_array_equal(
            viewer.visible_node_render_order(),
            [6, 5, 4, 3, 2, 1, 0],
        )

    def test_left_clicked_selected_node_is_unique_top_tier(self):
        viewer = self.make_viewer(6)
        viewer.node_render_order = np.array([5, 4, 3, 2, 1, 0])
        viewer.selected_indices = [0, 3]
        viewer.selected_node_idx = 0
        viewer.left_click_highlight_indices = [2]
        viewer.edges = np.array([[0, 4], [3, 1], [2, 5]], dtype=np.int32)

        np.testing.assert_array_equal(
            viewer.visible_node_render_order(),
            [5, 1, 4, 3, 0, 2],
        )

        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        np.testing.assert_array_equal(
            viewer.visible_node_render_order(),
            [5, 2, 1, 4, 0, 3],
        )

    def test_rgba_equivalent_colors_disable_neighbor_promotion(self):
        viewer = self.make_viewer(3)
        viewer.node_render_order = np.array([2, 1, 0])
        viewer.selected_indices = [0]
        viewer.edges = np.array([[0, 2]], dtype=np.int32)
        viewer._connected_to_selected_indices = mock.Mock(
            side_effect=AssertionError("neighbor discovery should be skipped")
        )

        with mock.patch.object(cfg, "CONNECTED_NODE_COLOR", "black"), mock.patch.object(
            cfg, "NODE_BOUNDARY_COLOR", "#000000ff"
        ):
            self.assertFalse(viewer._connected_node_identification_enabled())
            np.testing.assert_array_equal(
                viewer.visible_node_render_order(), [2, 1, 0]
            )

        viewer._connected_to_selected_indices.assert_not_called()

    def test_different_colors_enable_neighbor_promotion(self):
        viewer = self.make_viewer(3)
        viewer.node_render_order = np.array([2, 1, 0])
        viewer.selected_indices = [0]
        viewer.edges = np.array([[0, 2]], dtype=np.int32)

        with mock.patch.object(cfg, "CONNECTED_NODE_COLOR", "red"), mock.patch.object(
            cfg, "NODE_BOUNDARY_COLOR", "black"
        ):
            self.assertTrue(viewer._connected_node_identification_enabled())
            np.testing.assert_array_equal(
                viewer.visible_node_render_order(), [1, 2, 0]
            )

    def test_update_nodes_submits_every_attribute_in_effective_order(self):
        viewer = self.make_renderable_viewer(4)
        viewer.node_render_order = np.array([2, 0, 3, 1], dtype=np.int32)
        viewer.visible_mask[0] = False
        viewer.current_colors = np.column_stack(
            (np.arange(4), np.zeros((4, 2)), np.ones(4))
        ).astype(np.float32)
        viewer.current_sizes = np.arange(10, 14, dtype=np.float32)
        viewer.current_shapes = np.array(
            ["disc", "square", "triangle_up", "star"], dtype=object
        )
        viewer.update_nodes()

        np.testing.assert_array_equal(viewer._submitted_visible_node_order, [2, 3, 1])
        np.testing.assert_array_equal(viewer.markers.data["pos"][:, 0], [2, 3, 1])
        np.testing.assert_array_equal(
            viewer.markers.data["face_color"][:, 0], [2, 3, 1]
        )
        np.testing.assert_array_equal(viewer.markers.data["size"], [12, 13, 11])
        self.assertEqual(
            viewer.markers.data["symbol"], ["triangle_up", "star", "square"]
        )

    def test_left_click_rings_are_immediately_below_each_clicked_node(self):
        viewer = self.make_renderable_viewer(5)
        viewer.node_render_order = np.array([2, 4, 3, 1, 0], dtype=np.int32)
        viewer.selected_indices = [1]
        viewer.selected_node_idx = 3
        viewer.left_click_highlight_indices = [0]
        viewer.edges = np.array([[1, 2], [3, 4]], dtype=np.int32)
        viewer.current_colors = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 0.8],
                [0.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        viewer.current_sizes = np.arange(10, 15, dtype=np.float32)

        viewer.update_nodes()

        np.testing.assert_array_equal(
            viewer._submitted_visible_node_order, [4, 2, 1, 0, 3]
        )
        np.testing.assert_array_equal(
            viewer._submitted_marker_node_order, [4, 2, 1, 0, 0, 3, 3]
        )
        np.testing.assert_array_equal(
            viewer._submitted_marker_ring_mask,
            [False, False, False, True, False, True, False],
        )

        marker_data = viewer.markers.data
        for ring_slot, node_slot in ((3, 4), (5, 6)):
            self.assertEqual(
                viewer._submitted_marker_node_order[ring_slot],
                viewer._submitted_marker_node_order[node_slot],
            )
            np.testing.assert_allclose(
                marker_data["pos"][ring_slot], marker_data["pos"][node_slot]
            )
            self.assertEqual(
                marker_data["size"][ring_slot],
                marker_data["size"][node_slot] * 2.0,
            )
            np.testing.assert_allclose(marker_data["edge_color"][ring_slot], 0.0)
            self.assertEqual(marker_data["edge_width"][ring_slot], 0.0)
            np.testing.assert_allclose(
                marker_data["face_color"][ring_slot, :3],
                marker_data["face_color"][node_slot, :3],
            )
            self.assertEqual(
                marker_data["face_color"][ring_slot, 3],
                marker_data["face_color"][node_slot, 3] * 0.5,
            )

    def test_equal_colors_skip_neighbor_border_identification(self):
        viewer = self.make_renderable_viewer(3)
        viewer.selected_indices = [0]
        viewer.edges = np.array([[0, 1]], dtype=np.int32)

        with mock.patch.object(cfg, "CONNECTED_NODE_COLOR", "black"), mock.patch.object(
            cfg, "NODE_BOUNDARY_COLOR", "#000000ff"
        ), mock.patch.object(
            np, "isin", side_effect=AssertionError("neighbor discovery should be skipped")
        ):
            viewer.update_nodes()

        np.testing.assert_array_equal(viewer._submitted_visible_node_order, [1, 2, 0])
        node_edges = dict(
            zip(viewer._submitted_visible_node_order, viewer.markers.data["edge_color"])
        )
        np.testing.assert_allclose(node_edges[1], [0.0, 0.0, 0.0, 1.0])
        self.assertFalse(np.allclose(node_edges[0], node_edges[1]))

    def test_different_colors_highlight_and_promote_selected_neighbors(self):
        viewer = self.make_renderable_viewer(3)
        viewer.node_render_order = np.array([1, 2, 0], dtype=np.int32)
        viewer.selected_indices = [0]
        viewer.edges = np.array([[0, 1]], dtype=np.int32)

        with mock.patch.object(cfg, "CONNECTED_NODE_COLOR", "red"), mock.patch.object(
            cfg, "NODE_BOUNDARY_COLOR", "black"
        ):
            viewer.update_nodes()

        np.testing.assert_array_equal(viewer._submitted_visible_node_order, [2, 1, 0])
        node_edges = dict(
            zip(viewer._submitted_visible_node_order, viewer.markers.data["edge_color"])
        )
        np.testing.assert_allclose(node_edges[1], [1.0, 0.0, 0.0, 1.0])

    def test_undo_state_restores_persistent_order(self):
        viewer = self.make_viewer(4)
        viewer.node_render_order = np.array([2, 0, 3, 1], dtype=np.int32)
        viewer.update_selection_visual = mock.Mock()
        viewer.update_edges = mock.Mock()

        state = viewer._get_current_state()
        viewer.node_render_order = np.arange(4, dtype=np.int32)
        viewer._apply_state(state)

        np.testing.assert_array_equal(viewer.node_render_order, [2, 0, 3, 1])


class LeftClickFocusTests(unittest.TestCase):
    def make_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = 3
        viewer.full_headers = ["A", "B", "C"]
        viewer.cluster_labels = np.array([1, 7, -1], dtype=np.int32)
        viewer.group_labels = [set(), {"beta", "alpha"}, set()]
        viewer.selected_indices = [0, 2]
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = [0, 2]
        viewer.tooltip = SimpleNamespace(text="old tooltip")
        viewer.console_text = SimpleNamespace(text="")
        display = SimpleNamespace(on_node_clicked=mock.Mock())
        viewer.hud_displays = {"probe": display}
        viewer.sync_metadata_table_selection = mock.Mock()
        viewer.update_nodes = mock.Mock()
        viewer.broadcast_event = mock.Mock()
        return viewer, display

    def test_apply_left_click_focus_runs_full_shared_behavior(self):
        viewer, display = self.make_viewer()

        viewer.apply_left_click_focus(1)

        self.assertEqual(viewer.selected_node_idx, 1)
        self.assertIsNone(viewer.left_click_highlight_indices)
        self.assertEqual(viewer.selected_indices, [0, 2])
        self.assertEqual(viewer.tooltip.text, "")
        self.assertEqual(
            viewer.console_text.text,
            "Selected: [Cluster 7] B [Groups: alpha, beta]",
        )
        viewer.sync_metadata_table_selection.assert_called_once_with(1)
        display.on_node_clicked.assert_called_once_with(1)
        viewer.update_nodes.assert_called_once_with()
        viewer.broadcast_event.assert_called_once_with(
            {"type": "highlight_row", "index": 1}
        )

    def test_clear_left_click_focus_preserves_command_selection(self):
        viewer, _display = self.make_viewer()
        viewer.selected_node_idx = 1

        viewer.clear_left_click_focus()

        self.assertIsNone(viewer.selected_node_idx)
        self.assertIsNone(viewer.left_click_highlight_indices)
        self.assertEqual(viewer.selected_indices, [0, 2])
        self.assertEqual(viewer.tooltip.text, "")
        viewer.update_nodes.assert_called_once_with()
        viewer.broadcast_event.assert_called_once_with(
            {"type": "highlight_row", "index": None}
        )


class CommandPromotionTests(unittest.TestCase):
    def make_viewer(self):
        return SimpleNamespace(
            n_nodes=3,
            full_headers=["A", "B", "C"],
            cluster_labels=None,
            group_labels=[set(), set(), set()],
            alignment=None,
            metadata={
                "Length": {
                    "type": "number",
                    "values": np.array([1.0, np.nan, 3.0]),
                }
            },
            current_colors=np.zeros((3, 4), dtype=float),
            current_sizes=np.ones(3, dtype=float),
            current_shapes=np.full(3, "disc", dtype=object),
            visible_mask=np.ones(3, dtype=bool),
            selected_indices=[0, 2],
            console_text=SimpleNamespace(text=""),
            _save_state=mock.Mock(),
            promote_nodes=mock.Mock(),
            update_nodes=mock.Mock(),
        )

    def test_color_promotes_one_union_for_color_size_and_shape(self):
        viewer = self.make_viewer()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cfg, "HEADER_LIST_DIR", temp_dir, create=True
        ):
            color_command.run(viewer, ["red", "x2", "triangle"])

        viewer._save_state.assert_called_once_with()
        viewer.promote_nodes.assert_called_once()
        np.testing.assert_array_equal(
            viewer.promote_nodes.call_args.args[0], [True, False, True]
        )
        viewer.update_nodes.assert_called_once_with()

    def test_chained_color_assignments_promote_one_combined_group(self):
        viewer = self.make_viewer()
        viewer.selected_indices = []
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cfg, "HEADER_LIST_DIR", temp_dir, create=True
        ):
            color_command.run(viewer, ['"A"', "red", '"C"', "blue"])

        viewer.promote_nodes.assert_called_once()
        np.testing.assert_array_equal(
            viewer.promote_nodes.call_args.args[0], [True, False, True]
        )

    def test_color_excludes_hidden_nodes_from_every_assignment(self):
        viewer = self.make_viewer()
        viewer.visible_mask = np.array([False, True, False])
        viewer.selected_indices = []
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cfg, "HEADER_LIST_DIR", temp_dir, create=True
        ):
            color_command.run(
                viewer,
                ['"A"|"B"|"C"', "red", "x2", "triangle"],
            )

        np.testing.assert_array_equal(
            viewer.current_colors[:, 0], [0.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            viewer.current_sizes, [1.0, cfg.NODE_SIZE * 2.0, 1.0]
        )
        np.testing.assert_array_equal(
            viewer.current_shapes, ["disc", "triangle_up", "disc"]
        )
        np.testing.assert_array_equal(
            viewer.promote_nodes.call_args.args[0], [False, True, False]
        )
        self.assertEqual(
            viewer.console_text.text,
            "Applied: 1 nodes (red, x2.0, triangle_up)",
        )

    def test_spectrum_promotes_gradient_and_gray_nodes_together(self):
        viewer = self.make_viewer()
        viewer.selected_indices = []
        fake_meta = SimpleNamespace(run=mock.Mock())
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            cfg, "HEADER_LIST_DIR", temp_dir, create=True
        ), mock.patch("importlib.import_module", return_value=fake_meta):
            spectrum_command.run(viewer, ["prop:Length"])

        viewer._save_state.assert_called_once_with()
        viewer.promote_nodes.assert_called_once()
        np.testing.assert_array_equal(
            viewer.promote_nodes.call_args.args[0], [True, True, True]
        )
        viewer.update_nodes.assert_called_once_with()


class ResetRenderOrderTests(unittest.TestCase):
    def make_viewer(self):
        return SimpleNamespace(
            n_nodes=3,
            node_render_order=np.array([2, 0, 1], dtype=np.int32),
            selected_indices=[2],
            current_colors=np.ones((3, 4), dtype=float),
            console_text=SimpleNamespace(text=""),
            _save_state=mock.Mock(),
            update_nodes=mock.Mock(),
            update_console_background=mock.Mock(),
        )

    def test_order_and_layer_aliases_reset_only_persistent_order(self):
        for target in ("order", "orders", "layer", "layers"):
            with self.subTest(target=target):
                viewer = self.make_viewer()
                Command_Engine.execute_reset(viewer, [target])

                np.testing.assert_array_equal(viewer.node_render_order, [0, 1, 2])
                self.assertEqual(viewer.selected_indices, [2])
                viewer._save_state.assert_called_once_with()
                viewer.update_nodes.assert_called_once_with()

    def test_color_reset_does_not_change_render_order(self):
        viewer = self.make_viewer()

        Command_Engine.execute_reset(viewer, ["colors"])

        np.testing.assert_array_equal(viewer.node_render_order, [2, 0, 1])

    def test_combined_reset_uses_one_snapshot_and_redraw(self):
        viewer = self.make_viewer()

        Command_Engine.execute_reset(viewer, ["colors", "order"])

        np.testing.assert_array_equal(viewer.node_render_order, [0, 1, 2])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()


class MouseSelectionTimingTests(unittest.TestCase):
    def make_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.console_mode = False
        viewer.is_multi_dragging = False
        viewer.is_box_selecting = True
        viewer.drag_start_mouse = np.array([0.0, 0.0])
        viewer.drag_start_screen = np.array([0.0, 0.0])
        viewer._pre_drag_selection = set()
        viewer.pos = np.array([[2.0, 2.0], [8.0, 8.0], [20.0, 20.0]])
        viewer.visible_mask = np.ones(3, dtype=bool)
        viewer.selected_indices = []
        viewer.selection_box = SimpleNamespace(set_data=mock.Mock(), visible=True)
        transform = IdentityTransform()
        viewer.canvas = SimpleNamespace(
            scene=SimpleNamespace(node_transform=lambda _scene: transform)
        )
        viewer.view = SimpleNamespace(scene=object())
        viewer.update_selection_visual = mock.Mock()
        viewer.console_text = SimpleNamespace(text="")
        return viewer

    @staticmethod
    def mouse_event():
        return SimpleNamespace(
            pos=np.array([10.0, 10.0]),
            buttons=[2],
            modifiers=[],
            handled=False,
        )

    def test_normal_box_selection_updates_live(self):
        viewer = self.make_viewer()
        event = self.mouse_event()

        with mock.patch.object(cfg, "LOW_RESOURCE_MODE", False):
            viewer.on_mouse_move(event)

        self.assertEqual(viewer.selected_indices, [0, 1])
        viewer.update_selection_visual.assert_called_once_with()

    def test_low_resource_box_selection_defers_until_release(self):
        viewer = self.make_viewer()
        event = self.mouse_event()

        with mock.patch.object(cfg, "LOW_RESOURCE_MODE", True):
            viewer.on_mouse_move(event)
            viewer.update_selection_visual.assert_not_called()
            viewer.on_mouse_release(event)

        self.assertEqual(viewer.selected_indices, [0, 1])
        viewer.update_selection_visual.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
