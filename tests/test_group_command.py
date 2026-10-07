"""The group command (commands/group.py): atomic multi-pair assignment and the
group names it accepts."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import group as group_command
from tests.command_fixtures import one_node_viewer, reported_outcomes


class GroupCommandTests(unittest.TestCase):
    def make_viewer(self):
        return one_node_viewer()

    def test_group_does_not_apply_earlier_pair_when_later_pair_is_invalid(self):
        viewer = self.make_viewer()
        group_command.run(
            viewer,
            ['"node"', "first", "#missing#", "second"],
        )

        self.assertEqual(viewer.group_labels, [set()])
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_group_can_define_then_reference_group_atomically(self):
        viewer = self.make_viewer()
        group_command.run(
            viewer,
            ['"node"', "first", "#first#", "second"],
        )

        self.assertEqual(viewer.group_labels, [{"first", "second"}])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()

    def test_group_name_position_overrides_expression_classification(self):
        viewer = self.make_viewer()

        group_command.run(viewer, ['"node"', "P106"])

        self.assertEqual(viewer.group_labels, [{"p106"}])
        viewer._save_state.assert_called_once_with()

    def test_group_rejects_reserved_words_and_canonical_generated_labels(self):
        prohibited_names = (
            "noise",
            "reset",
            "remove",
            "delete",
            "list",
            "help",
            "cluster",
            "group",
            "groups",
            "clusters",
            "cluster_1",
            "subcluster_1_1",
            "subcluster_12_34",
            "GROUP",
            "CLUSTER",
            "Cluster_1",
            "Subcluster_1_1",
        )

        for name in prohibited_names:
            with self.subTest(name=name):
                viewer = self.make_viewer()
                with mock.patch("builtins.print"):
                    group_command.run(viewer, ['"node"', name])

                self.assertEqual(viewer.group_labels, [set()])
                self.assertTrue(viewer.console_text.text.startswith("Skipped:"))
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()

    def test_group_allows_noncanonical_generated_label_lookalikes(self):
        allowed_names = (
            "cluster_0",
            "cluster_27",
            "cluster_001",
            "subcluster_0_1",
            "subcluster_001_2",
            "subcluster_1_002",
        )

        for name in allowed_names:
            with self.subTest(name=name):
                viewer = self.make_viewer()
                with mock.patch("builtins.print"):
                    group_command.run(viewer, ['"node"', name])

                self.assertEqual(viewer.group_labels, [{name}])
                viewer._save_state.assert_called_once_with()
                viewer.update_nodes.assert_called_once_with()

    def test_group_rejects_any_loaded_canonical_cluster_name(self):
        for cluster_id in (0, 27):
            with self.subTest(cluster_id=cluster_id):
                viewer = self.make_viewer()
                viewer.cluster_labels = np.array([cluster_id])

                with mock.patch("builtins.print"):
                    group_command.run(
                        viewer,
                        ['"node"', f"cluster_{cluster_id}"],
                    )

                self.assertEqual(viewer.group_labels, [set()])
                self.assertIn(
                    "conflicts with an existing topology cluster",
                    viewer.console_text.text,
                )
                viewer._save_state.assert_not_called()

    def run_remove(self, group_labels, arguments):
        viewer = self.make_viewer()
        viewer.group_labels = group_labels
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            group_command.run(viewer, arguments)
        return viewer, succeeded, failed

    def test_remove_with_a_missing_group_fails_and_removes_nothing(self):
        for group_labels, arguments in (
            ([{"alpha"}], ["remove", "absent"]),
            # All or nothing: the existing group survives a typo next to it.
            ([{"alpha"}], ["remove", "alpha", "absent"]),
            ([{"alpha"}], ["delete", "ABSENT"]),
            (None, ["remove", "absent"]),
        ):
            with self.subTest(group_labels=group_labels, arguments=arguments):
                expected_labels = None if group_labels is None else [set(group_labels[0])]
                viewer, succeeded, failed = self.run_remove(group_labels, arguments)

                self.assertEqual(viewer.group_labels, expected_labels)
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn("not found: absent.", failed.call_args.args[1])

    def test_remove_deletes_existing_groups_as_one_undo_step(self):
        viewer, succeeded, failed = self.run_remove([{"alpha", "beta"}], ["remove", "Alpha"])

        self.assertEqual(viewer.group_labels, [{"beta"}])
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()
        failed.assert_not_called()
        self.assertEqual(
            succeeded.call_args.args[1],
            "Removed 1 group(s) from 1 total node instances.",
        )


if __name__ == "__main__":
    unittest.main()
