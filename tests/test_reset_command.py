"""The reset command (commands/reset.py) and Command_Engine.execute_reset: which
targets they accept, that a refused reset changes nothing and saves no undo
state, and that a reset with nothing to reset saves none either."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import matplotlib.colors as mcolors
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import Command_Engine
import EMAPSSN_Config as cfg
from commands import reset as reset_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


ORIGINAL_COLORS = one_node_viewer().current_colors.copy()


class ResetCommandTests(unittest.TestCase):
    def make_viewer(self):
        viewer = one_node_viewer()
        viewer.tooltip = SimpleNamespace(text="")
        viewer.current_shapes = np.array(["star"], dtype=object)
        return viewer

    def run_reset(self, arguments):
        viewer = self.make_viewer()
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, arguments)
        return viewer, succeeded, failed

    def test_unknown_or_missing_targets_fail_and_reset_nothing(self):
        for arguments, expected in (
            (["bogus"], "Unknown reset target(s): bogus."),
            # One unknown target refuses the whole command.
            (["colors", "bogus"], "Unknown reset target(s): bogus."),
            ([], "Specify at least one reset target."),
        ):
            with self.subTest(arguments=arguments):
                viewer, succeeded, failed = self.run_reset(arguments)

                np.testing.assert_array_equal(viewer.current_colors, ORIGINAL_COLORS)
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn(expected, failed.call_args.args[1])

    def test_known_targets_reset_as_one_undo_step(self):
        viewer, succeeded, failed = self.run_reset(["Colors", "shape"])

        np.testing.assert_allclose(
            viewer.current_colors, [mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)]
        )
        np.testing.assert_array_equal(viewer.current_shapes, ["disc"])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Reset successful: colors, shapes.")

    def test_a_target_named_twice_is_reported_once(self):
        for arguments, expected in (
            (["colors", "colors"], "Reset successful: colors."),
            (["Colors", "color", "sizes"], "Reset successful: colors, sizes."),
            (["hide", "hidden"], "Reset successful: hidden."),
            (["order", "layer", "orders"], "Reset successful: node order."),
        ):
            with self.subTest(arguments=arguments):
                viewer, succeeded, failed = self.run_reset(arguments)

                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], expected)

    def test_reset_saves_undo_state_only_when_something_changes(self):
        def at_defaults():
            viewer = self.make_viewer()
            viewer.current_colors[:] = mcolors.to_rgba(cfg.INITIAL_NODE_COLOR)
            viewer.current_sizes.fill(cfg.NODE_SIZE)
            viewer.current_shapes[:] = "disc"
            viewer.cluster_labels = None
            viewer.last_cluster_params = None
            viewer.n_nodes = 1
            viewer.node_render_order = np.arange(1, dtype=np.int32)
            return viewer

        everything = ["colors", "sizes", "shapes", "clusters", "groups", "hide", "network", "order", "layer"]
        for arguments in (everything, ["hidden"], ["hide", "hidden"]):
            with self.subTest(nothing_to_reset=arguments):
                viewer = at_defaults()
                with reported_outcomes(), redirect_stdout(io.StringIO()):
                    reset_command.run(viewer, arguments)

                viewer._save_state.assert_not_called()

        # Each target alone, and then with a no-op target beside it.
        changes = {
            "colors": lambda v: v.current_colors.__setitem__(0, [1, 0, 0, 1]),
            "sizes": lambda v: v.current_sizes.fill(cfg.NODE_SIZE + 1),
            "shapes": lambda v: v.current_shapes.__setitem__(0, "star"),
            "clusters": lambda v: setattr(v, "cluster_labels", np.array([1])),
            "groups": lambda v: v.group_labels[0].add("kinases"),
            "hide": lambda v: v.visible_mask.__setitem__(0, False),
            "network": lambda v: (setattr(v, "original_pos", np.zeros((1, 2))), setattr(v, "pos", np.ones((1, 2)))),
            # A one-node order is always the identity, so this viewer gets a second node.
            "order": lambda v: (setattr(v, "n_nodes", 2), setattr(v, "node_render_order", np.array([1, 0], dtype=np.int32))),
        }
        for target, change in changes.items():
            for others in ([], ["hidden"]):
                with self.subTest(target=target, with_no_op_targets=others):
                    viewer = at_defaults()
                    change(viewer)
                    with reported_outcomes(), redirect_stdout(io.StringIO()):
                        reset_command.run(viewer, others + [target])

                    viewer._save_state.assert_called_once_with()

    def test_clusters_reset_clears_the_cluster_parameters_too(self):
        # Label, save and MCP inspection read last_cluster_params, which
        # describes the clusters that this reset just cleared.
        viewer = self.make_viewer()
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        with reported_outcomes(), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertIsNone(viewer.cluster_labels)
        self.assertIsNone(viewer.last_cluster_params)
        viewer._save_state.assert_called_once_with()

    def test_clusters_reset_with_only_parameters_left_still_saves_undo_state(self):
        viewer = self.make_viewer()
        viewer.cluster_labels = None
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        with reported_outcomes(), redirect_stdout(io.StringIO()):
            reset_command.run(viewer, ["clusters"])

        self.assertIsNone(viewer.last_cluster_params)
        viewer._save_state.assert_called_once_with()

    def test_execute_reset_refuses_unknown_targets_before_saving_state(self):
        # The other commands' "<command> reset" forms call execute_reset
        # directly, so it keeps its own guard.
        viewer = self.make_viewer()
        with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
            Command_Engine.execute_reset(viewer, ["colors", "bogus"])

        np.testing.assert_array_equal(viewer.current_colors, ORIGINAL_COLORS)
        viewer._save_state.assert_not_called()

    def test_every_accepted_target_name_is_handled(self):
        # TARGET_NAMES, which reset accepts, has to match the targets
        # execute_reset handles; a name only in the first would be accepted
        # and then silently do nothing.
        reported = {
            "color": "colors", "size": "sizes", "shape": "shapes",
            "cluster": "clusters", "group": "groups", "hide": "hidden",
            "hidden": "hidden", "network": "network", "order": "node order",
            "layer": "node order",
        }
        self.assertEqual(set(reset_command.TARGET_NAMES), set(reported))
        for name, target in reported.items():
            with self.subTest(name=name), redirect_stdout(io.StringIO()):
                message = Command_Engine.execute_reset(self.make_viewer(), [name])
                self.assertEqual(message, f"Reset successful: {target}.")


if __name__ == "__main__":
    unittest.main()
