"""The hide command (commands/hide.py)."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import hide as hide_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


def graph_viewer(visible, edges, scores, selected=()):
    """A viewer with one node per visible flag and the given scored edges."""
    viewer = one_node_viewer()
    viewer.n_nodes = len(visible)
    viewer.full_headers = [f"node{index}" for index in range(len(visible))]
    viewer.visible_mask = np.array(visible)
    viewer.edges = np.array(edges, dtype=int).reshape(-1, 2)
    viewer.edge_scores = np.array(scores, dtype=float)
    viewer.selected_indices = list(selected)
    viewer.hovered_node_idx = 0
    viewer.selected_node_idx = 0
    viewer.tooltip = SimpleNamespace(text="node0")
    return viewer


def run_hide(viewer, args):
    """Run hide and return the (succeeded, failed) outcome mocks."""
    with reported_outcomes() as outcomes, redirect_stdout(io.StringIO()):
        hide_command.run(viewer, args)
    return outcomes


class HideCommandTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def test_hide_single_uses_configured_threshold(self):
        viewer = self.make_viewer()
        viewer.edges = np.empty((0, 2), dtype=int)
        viewer.edge_scores = np.array([], dtype=float)
        viewer.current_slider_threshold = 0.5
        viewer.hovered_node_idx = 0
        viewer.selected_node_idx = 0
        viewer.tooltip = SimpleNamespace(text="node")

        hide_command.run(viewer, ["single"])

        np.testing.assert_array_equal(viewer.visible_mask, np.array([False]))
        viewer._save_state.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()

    def test_invalid_hide_does_not_change_visibility_or_undo_state(self):
        viewer = self.make_viewer()
        hide_command.run(viewer, ["#cluster_99#"])

        np.testing.assert_array_equal(viewer.visible_mask, [True])
        viewer._save_state.assert_not_called()
        viewer.update_edges.assert_not_called()

    def test_expression_with_no_visible_match_succeeds_without_an_undo_step(self):
        # '"node"' matches the only node, which is already hidden; '"absent"'
        # matches nothing. A valid expression may match zero nodes, so this
        # is an informational success, but there is nothing to undo.
        for visible, expression in (([False], '"node"'), ([True], '"absent"')):
            with self.subTest(expression=expression, visible=visible):
                viewer = self.make_viewer()
                viewer.visible_mask = np.array(visible)
                with reported_outcomes() as (succeeded, failed), \
                        redirect_stdout(io.StringIO()):
                    hide_command.run(viewer, [expression])

                np.testing.assert_array_equal(viewer.visible_mask, visible)
                viewer._save_state.assert_not_called()
                viewer.update_selection_visual.assert_not_called()
                viewer.update_edges.assert_not_called()
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertIn("No visible nodes matched", succeeded.call_args.args[1])

    def test_expression_hides_visible_matches_as_one_undo_step(self):
        viewer = self.make_viewer()
        viewer.tooltip = SimpleNamespace(text="node")
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            hide_command.run(viewer, ['"node"'])

        np.testing.assert_array_equal(viewer.visible_mask, [False])
        viewer._save_state.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Hidden 1 node matching expression.")

    def test_hide_without_arguments_hides_the_selection(self):
        viewer = graph_viewer([True, True, True], [], [], selected=[0, 2])
        succeeded, failed = run_hide(viewer, [])

        np.testing.assert_array_equal(viewer.visible_mask, [False, True, False])
        self.assertEqual(viewer.selected_indices, [])
        self.assertIsNone(viewer.hovered_node_idx)
        self.assertIsNone(viewer.selected_node_idx)
        self.assertEqual(viewer.tooltip.text, "")
        viewer._save_state.assert_called_once_with()
        viewer.update_selection_visual.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()
        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Hidden 2 selected nodes.")

    def test_hide_without_arguments_or_selection_fails(self):
        viewer = graph_viewer([True, True], [], [])
        succeeded, failed = run_hide(viewer, [])

        failed.assert_called_once_with(viewer, "Error: No nodes currently selected.")
        succeeded.assert_not_called()
        np.testing.assert_array_equal(viewer.visible_mask, [True, True])
        viewer._save_state.assert_not_called()
        viewer.update_edges.assert_not_called()

    def test_hide_single_hides_visible_nodes_without_an_active_edge(self):
        # Edges 0-1 (score 0.9) and 2-3 (score 0.3); node 4 has no edge.
        edges, scores = [(0, 1), (2, 3)], [0.9, 0.3]
        cases = [
            (0.5, [True] * 5, [True, True, False, False, False], "Hidden 3 single/free nodes."),
            # An edge scoring exactly the threshold stays active.
            (0.3, [True] * 5, [True, True, True, True, False], "Hidden 1 single/free node."),
            # An edge to a hidden node is not active.
            (0.3, [True, False, True, True, True], [False, False, True, True, False], "Hidden 2 single/free nodes."),
        ]
        for threshold, visible, expected, message in cases:
            with self.subTest(threshold=threshold, visible=visible):
                viewer = graph_viewer(visible, edges, scores, selected=[0, 4])
                viewer.current_slider_threshold = threshold
                succeeded, failed = run_hide(viewer, ["single"])

                np.testing.assert_array_equal(viewer.visible_mask, expected)
                self.assertEqual(
                    viewer.selected_indices,
                    [index for index in (0, 4) if expected[index]],
                )
                self.assertEqual(viewer.tooltip.text, "")
                viewer._save_state.assert_called_once_with()
                viewer.update_selection_visual.assert_called_once_with()
                viewer.update_edges.assert_called_once_with()
                failed.assert_not_called()
                succeeded.assert_called_once_with(viewer, message)

    def test_hide_free_without_a_slider_uses_the_configured_threshold(self):
        viewer = graph_viewer([True] * 5, [(0, 1), (2, 3)], [0.9, 0.3])
        with mock.patch.object(hide_command.cfg, "SIMILARITY_THRESHOLD", 0.5):
            succeeded, _failed = run_hide(viewer, ["FREE"])

        np.testing.assert_array_equal(
            viewer.visible_mask, [True, True, False, False, False]
        )
        succeeded.assert_called_once_with(viewer, "Hidden 3 single/free nodes.")

    def test_hide_single_without_edge_scores_counts_every_visible_edge(self):
        viewer = graph_viewer([True] * 3, [(0, 1)], [])
        viewer.current_slider_threshold = 0.99
        succeeded, _failed = run_hide(viewer, ["single"])

        np.testing.assert_array_equal(viewer.visible_mask, [True, True, False])
        succeeded.assert_called_once_with(viewer, "Hidden 1 single/free node.")

    def test_hide_single_with_nothing_to_hide_adds_no_undo_step(self):
        # Node 2 is single but already hidden.
        viewer = graph_viewer([True, True, False], [(0, 1)], [0.9])
        viewer.current_slider_threshold = 0.5
        succeeded, failed = run_hide(viewer, ["single"])

        np.testing.assert_array_equal(viewer.visible_mask, [True, True, False])
        viewer._save_state.assert_not_called()
        viewer.update_edges.assert_not_called()
        failed.assert_not_called()
        succeeded.assert_called_once_with(
            viewer, "No single/free nodes found to hide at the current edge threshold."
        )

    def test_hide_reset_shows_every_node_again(self):
        viewer = graph_viewer([False, True, False], [], [])
        succeeded, failed = run_hide(viewer, ["RESET"])

        np.testing.assert_array_equal(viewer.visible_mask, [True, True, True])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()
        viewer.update_edges.assert_called_once_with()
        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Reset successful: hidden.")
        self.assertEqual(viewer.console_text.text, "Reset successful: hidden.")

    def test_hide_reset_with_nothing_hidden_adds_no_undo_step(self):
        # An empty undo step would push a real one out of the 50 the Viewer keeps.
        viewer = graph_viewer([True, True], [], [])
        succeeded, failed = run_hide(viewer, ["reset"])

        np.testing.assert_array_equal(viewer.visible_mask, [True, True])
        viewer._save_state.assert_not_called()
        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Reset successful: hidden.")


if __name__ == "__main__":
    unittest.main()
