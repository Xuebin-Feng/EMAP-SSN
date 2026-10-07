"""The reset command (commands/reset.py) and Command_Engine.execute_reset: which
targets they accept, and that a refused reset changes nothing and saves no undo
state."""

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
