"""The hide command (commands/hide.py)."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import hide as hide_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


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
        self.assertEqual(succeeded.call_args.args[1], "Hidden 1 nodes matching expression.")


if __name__ == "__main__":
    unittest.main()
