# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Viewer full-state undo/redo (EMAPSSN_Viewer.MainViewer): _save_state,
_do_undo, _do_redo, _apply_state and the 50-entry history. Compact metadata
history entries are covered in test_metadata, the restored render order in
test_node_render_order."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from EMAPSSN_Viewer import MainViewer  # noqa: E402


BLUE = [0.0, 0.0, 1.0, 1.0]
RED = [1.0, 0.0, 0.0, 1.0]


def make_viewer():
    """A four-node viewer holding every field a full history state captures."""
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 4
    viewer.full_headers = ["n0", "n1", "n2", "n3"]
    viewer.pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    viewer.visible_mask = np.ones(4, dtype=bool)
    viewer.current_colors = np.tile(BLUE, (4, 1)).astype(np.float32)
    viewer.current_sizes = np.full(4, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(4, "disc", dtype=object)
    viewer.node_render_order = np.arange(4, dtype=np.int32)
    viewer.cluster_labels = None
    viewer.last_cluster_params = None
    viewer.group_labels = [set(), set(), set(), set()]
    viewer.metadata = {
        "Length": {"type": "number", "values": np.array([10, 20, 30, 40])}
    }
    viewer._cacheable_attrs = {"sidebar_buttons_to_persist"}
    viewer.sidebar_buttons_to_persist = []
    viewer.selected_indices = []
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.update_selection_visual = mock.Mock()
    viewer.update_edges = mock.Mock()
    viewer.broadcast_event = mock.Mock()
    return viewer


def change_every_field(viewer):
    """Change what colour, hide, group, cluster, order, layout, metadata and
    custom-attribute commands change, in place where those commands do."""
    viewer.current_colors[1] = RED
    viewer.current_sizes[1] = 20.0
    viewer.current_shapes[1] = "star"
    viewer.visible_mask[2] = False
    viewer.group_labels[3].add("kinases")
    viewer.cluster_labels = np.array([0, 0, 1, -1], dtype=np.int32)
    viewer.last_cluster_params = {"method": "leiden"}
    viewer.node_render_order = np.array([0, 2, 3, 1], dtype=np.int32)
    viewer.pos[0] = [5.0, 5.0]
    viewer.metadata["Length"]["values"][0] = 99
    viewer.metadata["Note"] = {
        "type": "text",
        "values": np.array(["a", "b", "c", "d"], dtype=object),
    }
    viewer.sidebar_buttons_to_persist.append("Kinase view")


def quietly(action):
    """Run an undo/redo step without its console print reaching the test log."""
    with redirect_stdout(io.StringIO()):
        return action()


class ViewerUndoRedoTests(unittest.TestCase):
    def assert_original_state(self, viewer):
        np.testing.assert_array_equal(viewer.current_colors, np.tile(BLUE, (4, 1)))
        np.testing.assert_array_equal(viewer.current_sizes, [10.0, 10.0, 10.0, 10.0])
        self.assertEqual(viewer.current_shapes.tolist(), ["disc"] * 4)
        np.testing.assert_array_equal(viewer.visible_mask, [True, True, True, True])
        self.assertEqual(viewer.group_labels, [set(), set(), set(), set()])
        self.assertIsNone(viewer.cluster_labels)
        self.assertIsNone(viewer.last_cluster_params)
        np.testing.assert_array_equal(viewer.node_render_order, [0, 1, 2, 3])
        np.testing.assert_array_equal(viewer.pos[0], [0.0, 0.0])
        self.assertEqual(list(viewer.metadata), ["Length"])
        np.testing.assert_array_equal(
            viewer.metadata["Length"]["values"], [10, 20, 30, 40]
        )
        self.assertEqual(viewer.sidebar_buttons_to_persist, [])

    def assert_changed_state(self, viewer):
        np.testing.assert_array_equal(
            viewer.current_colors, [BLUE, RED, BLUE, BLUE]
        )
        np.testing.assert_array_equal(viewer.current_sizes, [10.0, 20.0, 10.0, 10.0])
        self.assertEqual(
            viewer.current_shapes.tolist(), ["disc", "star", "disc", "disc"]
        )
        np.testing.assert_array_equal(viewer.visible_mask, [True, True, False, True])
        self.assertEqual(viewer.group_labels, [set(), set(), set(), {"kinases"}])
        np.testing.assert_array_equal(viewer.cluster_labels, [0, 0, 1, -1])
        self.assertEqual(viewer.last_cluster_params, {"method": "leiden"})
        np.testing.assert_array_equal(viewer.node_render_order, [0, 2, 3, 1])
        np.testing.assert_array_equal(viewer.pos[0], [5.0, 5.0])
        self.assertEqual(list(viewer.metadata), ["Length", "Note"])
        np.testing.assert_array_equal(
            viewer.metadata["Length"]["values"], [99, 20, 30, 40]
        )
        self.assertEqual(
            viewer.metadata["Note"]["values"].tolist(), ["a", "b", "c", "d"]
        )
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["Kinase view"])

    def test_undo_restores_every_saved_field_and_redo_reapplies_the_change(self):
        viewer = make_viewer()
        viewer._save_state()
        change_every_field(viewer)

        self.assertTrue(quietly(viewer._do_undo))
        self.assert_original_state(viewer)
        self.assertEqual(viewer.console_text.text, "Undo successful.")
        self.assertEqual(
            (len(viewer.position_history), len(viewer.redo_stack)), (0, 1)
        )

        self.assertTrue(quietly(viewer._do_redo))
        self.assert_changed_state(viewer)
        self.assertEqual(viewer.console_text.text, "Redo successful.")
        self.assertEqual(
            (len(viewer.position_history), len(viewer.redo_stack)), (1, 0)
        )

        # Redo saved the state it replaced, so undo goes back again.
        self.assertTrue(quietly(viewer._do_undo))
        self.assert_original_state(viewer)

        self.assertEqual(viewer.update_selection_visual.call_count, 3)
        self.assertEqual(viewer.update_edges.call_count, 3)
        self.assertEqual(viewer.broadcast_event.call_count, 3)
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["type"], "state_updated")
        self.assertEqual(event["visible_mask"], [True, True, True, True])
        self.assertEqual(event["columns"], ["Node ID", "Length"])

    def test_new_change_after_undo_discards_the_redo_branch(self):
        viewer = make_viewer()
        viewer._save_state()
        viewer.current_colors[0] = RED
        quietly(viewer._do_undo)
        self.assertEqual(len(viewer.redo_stack), 1)

        viewer._save_state()
        viewer.current_sizes[0] = 30.0

        self.assertEqual(viewer.redo_stack, [])
        self.assertEqual(len(viewer.position_history), 1)
        self.assertFalse(quietly(viewer._do_redo))
        self.assertEqual(viewer.console_text.text, "Nothing to redo.")
        np.testing.assert_array_equal(viewer.current_colors[0], BLUE)
        self.assertEqual(viewer.current_sizes[0], 30.0)

    def test_history_keeps_only_the_fifty_most_recent_states(self):
        viewer = make_viewer()
        for size in range(1, 52):
            viewer.current_sizes[0] = float(size)
            viewer._save_state()

        self.assertEqual(len(viewer.position_history), 50)
        self.assertEqual(
            [state["sizes"][0] for state in viewer.position_history],
            list(range(2, 52)),
        )
        # Fifty undos walk back to the oldest kept state; nothing is left after.
        for _ in range(50):
            self.assertTrue(quietly(viewer._do_undo))
        self.assertEqual(viewer.current_sizes[0], 2.0)
        self.assertFalse(quietly(viewer._do_undo))

    def test_undo_and_redo_with_empty_history_change_nothing(self):
        viewer = make_viewer()
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertFalse(viewer._do_undo())
            undo_message = viewer.console_text.text
            self.assertFalse(viewer._do_redo())

        self.assertEqual(undo_message, "Nothing to undo.")
        self.assertEqual(viewer.console_text.text, "Nothing to redo.")
        self.assertEqual(
            output.getvalue().splitlines(), ["Nothing to undo.", "Nothing to redo."]
        )
        self.assertEqual((viewer.position_history, viewer.redo_stack), ([], []))
        self.assert_original_state(viewer)
        viewer.update_selection_visual.assert_not_called()
        viewer.broadcast_event.assert_not_called()

    def test_restored_state_drops_selected_nodes_it_hides(self):
        viewer = make_viewer()
        viewer.visible_mask[1] = False
        viewer._save_state()
        viewer.visible_mask[1] = True
        viewer.selected_indices = [0, 1, 3]

        self.assertTrue(quietly(viewer._do_undo))

        np.testing.assert_array_equal(viewer.visible_mask, [True, False, True, True])
        self.assertEqual(viewer.selected_indices, [0, 3])
        viewer.update_selection_visual.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()
        event = viewer.broadcast_event.call_args.args[0]
        self.assertEqual(event["visible_mask"], [True, False, True, True])
        self.assertEqual(event["selected_indices"], [0, 3])


if __name__ == "__main__":
    unittest.main()
