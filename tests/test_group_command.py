"""The group command (commands/group.py): atomic multi-pair assignment and the
group names it accepts."""

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

from Viewer_Command_Portal import ExecutionContext, bind
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
            "Removed 1 group from 1 total node instance.",
        )

    def record(self, viewer, arguments):
        """Run group under a real execution context and return what it recorded."""
        record = {"messages": [], "outcome": None, "artifacts": [], "jobs": []}
        context = ExecutionContext(SimpleNamespace(viewer=viewer), "test", record)
        with bind(context), redirect_stdout(io.StringIO()):
            group_command.run(viewer, arguments)
        return record

    def test_skipped_names_do_not_fail_a_command_that_grouped_nodes(self):
        viewer = self.make_viewer()
        record = self.record(viewer, ['"node"', "mutant", '"node"', "noise"])

        self.assertEqual(viewer.group_labels, [{"mutant"}])
        viewer._save_state.assert_called_once_with()
        self.assertEqual(record["outcome"], "succeeded")
        self.assertEqual(
            record["messages"],
            [
                {
                    "status": "succeeded",
                    "text": (
                        "Groups Applied: 1 node -> 'mutant' (1 skipped)\n"
                        "Skipped group names: noise."
                    ),
                    "truncated": False,
                }
            ],
        )
        # The console line shows the first line; the report names what was skipped.
        self.assertEqual(
            viewer.console_text.text,
            "Groups Applied: 1 node -> 'mutant' (1 skipped)",
        )

    def test_every_skipped_name_is_listed_once(self):
        viewer = self.make_viewer()
        record = self.record(
            viewer,
            ['"node"', "ok", '"node"', "noise", '"node"', "bad!", '"node"', "Noise"],
        )

        self.assertEqual(viewer.group_labels, [{"ok"}])
        self.assertEqual(record["outcome"], "succeeded")
        self.assertEqual(
            record["messages"][-1]["text"],
            "Groups Applied: 1 node -> 'ok' (3 skipped)\n"
            "Skipped group names: noise, bad!, Noise.",
        )

    def test_a_command_that_grouped_nothing_because_of_skipped_names_fails(self):
        viewer = self.make_viewer()
        record = self.record(viewer, ['"node"', "noise"])

        self.assertEqual(viewer.group_labels, [set()])
        viewer._save_state.assert_not_called()
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(
            [(m["status"], m["text"]) for m in record["messages"]],
            [("failed", "Skipped: Group name 'noise' is a reserved keyword. Skipping.")],
        )
        self.assertTrue(viewer.console_text.text.startswith("Skipped:"))

    def test_a_command_without_skips_still_succeeds(self):
        viewer = self.make_viewer()
        record = self.record(viewer, ['"node"', "ok"])

        self.assertEqual(record["outcome"], "succeeded")
        self.assertEqual(
            [m["text"] for m in record["messages"]],
            ["Groups Applied: 1 node -> 'ok'"],
        )


class GroupHiddenNodeTests(unittest.TestCase):
    """group skips hidden nodes, as color, spectrum and select do."""

    def make_viewer(self, hidden=(1,), selected=()):
        """Four nodes with Length 100-400; the nodes in hidden are not visible."""
        viewer = one_node_viewer()
        viewer.n_nodes = 4
        viewer.full_headers = [f"node{index}" for index in range(4)]
        viewer.cluster_labels = np.array([1, 1, 2, 2])
        viewer.group_labels = [set() for _ in range(4)]
        viewer.metadata = {
            "Length": {"type": "number", "values": np.array([100.0, 200.0, 300.0, 400.0])}
        }
        viewer.visible_mask = np.array([index not in hidden for index in range(4)])
        viewer.selected_indices = list(selected)
        return viewer

    def run_group(self, viewer, arguments):
        with reported_outcomes() as (succeeded, failed), redirect_stdout(io.StringIO()):
            group_command.run(viewer, arguments)
        failed.assert_not_called()
        return succeeded.call_args.args[1]

    def test_hidden_matches_are_not_labelled_and_visible_ones_are(self):
        viewer = self.make_viewer(hidden=(1,))

        message = self.run_group(viewer, ["{Length>150}", "big"])

        # Nodes 1, 2 and 3 match; node 1 is hidden.
        self.assertEqual(viewer.group_labels, [set(), set(), {"big"}, {"big"}])
        self.assertEqual(message, "Groups Applied: 2 nodes -> 'big'")
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()

    def test_every_expression_target_skips_hidden_nodes(self):
        for expression in ('"node1"', "#cluster_1#", "{Length=200-200}", "!#cluster_2#&!$sele$"):
            with self.subTest(expression=expression):
                viewer = self.make_viewer(hidden=(1,))
                self.run_group(viewer, [expression, "name"])
                self.assertFalse(viewer.group_labels[1])

        viewer = self.make_viewer(hidden=(1,))
        self.run_group(viewer, ["#cluster_1#", "name"])
        self.assertEqual(viewer.group_labels, [{"name"}, set(), set(), set()])

    def test_the_selection_labels_only_its_visible_nodes(self):
        # Node 1 was selected, then hidden.
        viewer = self.make_viewer(hidden=(1,), selected=(0, 1, 2))

        message = self.run_group(viewer, ["picked"])

        self.assertEqual(viewer.group_labels, [{"picked"}, set(), {"picked"}, set()])
        self.assertEqual(message, "Groups Applied: 2 nodes -> 'picked'")

        viewer = self.make_viewer(hidden=(1,), selected=(0, 1, 2))
        self.run_group(viewer, ["$sele$", "picked"])
        self.assertEqual(viewer.group_labels, [{"picked"}, set(), {"picked"}, set()])

    def test_an_expression_matching_only_hidden_nodes_matches_nothing(self):
        viewer = self.make_viewer(hidden=(1,))

        message = self.run_group(viewer, ['"node1"', "ghost"])

        # The message and the state are those of an expression matching no node.
        self.assertEqual(message, "No nodes matched criteria for grouping.")
        self.assertEqual(viewer.group_labels, [set()] * 4)
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

        empty = self.make_viewer(hidden=(1,))
        self.assertEqual(
            self.run_group(empty, ['"absent"', "ghost"]),
            "No nodes matched criteria for grouping.",
        )

    def test_a_selection_of_only_hidden_nodes_matches_nothing(self):
        viewer = self.make_viewer(hidden=(1,), selected=(1,))

        message = self.run_group(viewer, ["picked"])

        self.assertEqual(message, "No nodes matched criteria for grouping.")
        self.assertEqual(viewer.group_labels, [set()] * 4)
        viewer._save_state.assert_not_called()

    def test_chained_pairs_see_only_the_visible_nodes_labelled_before(self):
        viewer = self.make_viewer(hidden=(1,))

        message = self.run_group(
            viewer, ["{Length>150}", "big", "#big#&{Length<350}", "middle"]
        )

        # big is nodes 2 and 3; hidden node 1 is in neither group.
        self.assertEqual(viewer.group_labels, [set(), set(), {"big", "middle"}, {"big"}])
        self.assertEqual(message, "Groups Applied: 2 nodes -> 'big'; 1 node -> 'middle'")

    def test_labels_hidden_nodes_already_carry_are_kept_but_not_matched(self):
        viewer = self.make_viewer(hidden=(1,))
        viewer.group_labels[1] = {"old"}
        viewer.group_labels[2] = {"old"}

        self.run_group(viewer, ["#old#", "again"])

        self.assertEqual(viewer.group_labels, [set(), {"old"}, {"old", "again"}, set()])

    def test_all_visible_nodes_are_labelled_as_before(self):
        viewer = self.make_viewer(hidden=())

        message = self.run_group(viewer, ["{Length>150}", "big"])

        self.assertEqual(viewer.group_labels, [set(), {"big"}, {"big"}, {"big"}])
        self.assertEqual(message, "Groups Applied: 3 nodes -> 'big'")

    def test_remove_list_and_reset_still_cover_hidden_nodes(self):
        viewer = self.make_viewer(hidden=(1,))
        viewer.group_labels = [{"keep"}, {"keep", "drop"}, set(), {"drop"}]

        self.run_group(viewer, ["remove", "drop"])
        self.assertEqual(viewer.group_labels, [{"keep"}, {"keep"}, set(), set()])

        output = io.StringIO()
        with reported_outcomes(), redirect_stdout(output):
            group_command.run(viewer, ["list"])
        self.assertRegex(output.getvalue(), r"keep\s+\|\s+2 ")

    def test_help_says_hidden_nodes_are_skipped(self):
        output = io.StringIO()
        with redirect_stdout(output):
            group_command.print_help()
        text = " ".join(output.getvalue().split())
        self.assertIn(
            "Hidden nodes are skipped, as by color, spectrum, and select: only "
            "visible nodes receive a group label",
            text,
        )
        self.assertIn("one that matches only hidden nodes labels none", text)
        self.assertNotIn("Hidden nodes are included", text)


if __name__ == "__main__":
    unittest.main()
