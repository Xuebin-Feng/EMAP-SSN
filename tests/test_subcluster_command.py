"""The subcluster command (commands/subcluster.py): the subcluster_N_M group
labels it generates, replaces and clears."""

import os
import sys
import unittest
from types import ModuleType
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Command_Engine.command_succeeded() imports Viewer_Command_Portal (and with it
# PySide6) on first use. Imported here, that first import cannot happen inside
# the sys.modules window below, whose exit would drop the modules again.
import Viewer_Command_Portal  # noqa: F401
from commands import subcluster as subcluster_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


class SubclusterCommandTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def test_subcluster_clear_keeps_custom_lookalike_groups(self):
        viewer = self.make_viewer()
        viewer.group_labels = [
            {
                "subcluster_1_1",
                "subcluster_12_34",
                "subcluster_0_1",
                "subcluster_001_2",
                "subcluster_1_002",
                "subcluster_1_0",
                "alpha",
            }
        ]

        with mock.patch("builtins.print"):
            subcluster_command.run(viewer, ["clear"])

        self.assertEqual(
            viewer.group_labels,
            [
                {
                    "subcluster_0_1",
                    "subcluster_001_2",
                    "subcluster_1_002",
                    "subcluster_1_0",
                    "alpha",
                }
            ],
        )
        self.assertIn("removed 2 label instances", viewer.console_text.text)
        viewer._save_state.assert_called_once_with()

    def test_subcluster_clear_with_nothing_to_clear_adds_no_undo_step(self):
        viewer = self.make_viewer()
        # Only custom lookalikes, which clear keeps.
        viewer.group_labels = [{"subcluster_0_1", "alpha"}]

        with reported_outcomes() as (succeeded, failed), mock.patch("builtins.print"):
            subcluster_command.run(viewer, ["clear"])

        self.assertEqual(viewer.group_labels, [{"subcluster_0_1", "alpha"}])
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "No subcluster groups to clear.")

    def test_subclustering_again_replaces_only_that_clusters_generated_labels(self):
        viewer = self.make_viewer()
        viewer.n_nodes = 3
        viewer.full_headers = ["node_a", "node_b", "node_c"]
        viewer.cluster_labels = np.array([1, 1, 2])
        viewer.edges = [[0, 1]]
        viewer.current_colors = np.ones((3, 4))
        viewer.group_labels = [
            {"subcluster_1_1", "subcluster_1_002", "subcluster_1_0"},
            {"subcluster_1_2"},
            {"subcluster_2_1"},
        ]

        with mock.patch.dict(
            sys.modules, {"graspologic_native": ModuleType("graspologic_native")}
        ), mock.patch.object(
            subcluster_command.network_clustering,
            "leiden_partition",
            return_value=np.array([1, 1]),
        ), mock.patch("builtins.print"):
            subcluster_command.run(viewer, ["cluster_1"])

        self.assertEqual(
            viewer.group_labels,
            [
                {"subcluster_1_1", "subcluster_1_002", "subcluster_1_0"},
                {"subcluster_1_1"},
                {"subcluster_2_1"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
