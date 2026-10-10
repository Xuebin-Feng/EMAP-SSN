"""The subcluster command (commands/subcluster.py): the subcluster_N_M group
labels it generates, replaces and clears, and its Jaccard and MCL modes on one
cluster's own edges."""

import importlib
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
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
from Viewer_Command_Portal import ExecutionContext, bind
from commands import subcluster as subcluster_command
from tests.command_fixtures import (
    BARBELL_EDGES,
    network_viewer,
    one_node_viewer,
    reported_outcomes,
    run_command,
)


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


# Cluster 1 holds the barbell on nodes 2-7 and node 8, whose only edge leads
# out of the cluster; nodes 0 and 1 form cluster 2, node 9 alone is cluster 3.
# Edges between clusters are listed among the barbell's own.
NETWORK_EDGES = (
    (0, 1),
    *((u + 2, v + 2) for u, v in BARBELL_EDGES[:3]),
    (1, 2),
    (8, 0),
    *((u + 2, v + 2) for u, v in BARBELL_EDGES[3:]),
)
NETWORK_CLUSTERS = [2, 2, 1, 1, 1, 1, 1, 1, 1, 3]
GREY = (0.8, 0.8, 0.8, 0.4)


class JaccardAndMCLSubclusterTests(unittest.TestCase):
    def subcluster(self, args, viewer=None):
        if viewer is None:
            viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        succeeded, failed, output = run_command(subcluster_command, viewer, args)
        return viewer, succeeded, failed, output

    def assert_groups(self, viewer, members):
        """Each node's groups equal the names that members gives it, if any."""
        expected = [set() for _ in range(viewer.n_nodes)]
        for name, nodes in members.items():
            for node in nodes:
                expected[node].add(name)
        self.assertEqual(viewer.group_labels, expected)

    def assert_subclustered(self, args, members, mode):
        viewer, succeeded, failed, _output = self.subcluster(["cluster_1", *args])
        failed.assert_not_called()
        self.assert_groups(viewer, members)
        noun = "subcluster" if len(members) == 1 else "subclusters"
        succeeded.assert_called_once_with(
            viewer,
            f"Done! Found {len(members)} {noun} in cluster_1 via {mode}.",
        )
        viewer._save_state.assert_called_once_with()
        grouped = set().union(*members.values()) if members else set()
        for node in range(viewer.n_nodes):
            color = viewer.current_colors[node]
            if NETWORK_CLUSTERS[node] != 1:
                # Nodes outside the target cluster keep their colors.
                np.testing.assert_array_equal(color, (0.0, 0.0, 0.0, 0.0))
            elif node in grouped:
                self.assertEqual(color[3], 1.0)
            else:
                np.testing.assert_array_equal(color, GREY)

    def test_jaccard_splits_the_cluster_by_its_own_edges(self):
        # The barbell's closed-neighbourhood Jaccard indices: 1 for 0-1 and
        # 4-5, 3/4 for the other triangle edges and 1/3 for the bridge.
        triangles = {"subcluster_1_1": (2, 3, 4), "subcluster_1_2": (5, 6, 7)}
        everything = {"subcluster_1_1": (2, 3, 4, 5, 6, 7)}
        cases = [
            # At or below 1/3 the bridge is kept too.
            (["0.2", "3"], everything),
            (["0", "3"], everything),
            (["0.34", "3"], triangles),
            (["0.5", "3"], triangles),
            (["0.76", "2"], {"subcluster_1_1": (2, 3), "subcluster_1_2": (6, 7)}),
            # Node 8 has no edge inside the cluster: a singleton.
            (["0.2", "1"], {**everything, "subcluster_1_2": (8,)}),
            (["0.5", "1"], {**triangles, "subcluster_1_3": (8,)}),
        ]
        for params, members in cases:
            with self.subTest(params=params):
                self.assert_subclustered(["jaccard", *params], members, "JACCARD")

    def test_mcl_splits_the_cluster_by_its_own_edges(self):
        cases = [
            (["2", "3"], {"subcluster_1_1": (2, 3, 4), "subcluster_1_2": (5, 6, 7)}),
            (
                ["2", "1"],
                {
                    "subcluster_1_1": (2, 3, 4),
                    "subcluster_1_2": (5, 6, 7),
                    "subcluster_1_3": (8,),
                },
            ),
            (["1.2", "1"], {"subcluster_1_1": (2, 3, 4, 5, 6, 7), "subcluster_1_2": (8,)}),
        ]
        for params, members in cases:
            with self.subTest(params=params):
                self.assert_subclustered(["mcl", *params], members, "MCL")

    def test_subclusters_are_numbered_from_largest(self):
        # Cluster 1 holds a triangle on nodes 1-3 and a K4 on 4-7: the K4 is
        # found second but is the larger subcluster, and takes the first color.
        edges = (
            (0, 1), (1, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7),
        )
        first_color = subcluster_command.get_subcluster_colors(2)[0]
        for args in (["jaccard", "0.2", "3"], ["mcl", "2", "3"]):
            with self.subTest(args=args):
                viewer = network_viewer(8, edges, cluster_labels=[2, 1, 1, 1, 1, 1, 1, 1])
                self.subcluster(["cluster_1", *args], viewer)
                self.assert_groups(
                    viewer,
                    {"subcluster_1_1": (4, 5, 6, 7), "subcluster_1_2": (1, 2, 3)},
                )
                for node in (4, 5, 6, 7):
                    np.testing.assert_allclose(
                        viewer.current_colors[node], (*first_color, 1.0)
                    )

    def test_leiden_subclusters_are_numbered_by_size_then_lowest_member(self):
        # Local nodes 0-5 are cluster 1's nodes 1-6. The partition's own IDs
        # are replaced: size first (3 > 2), then the lowest member node.
        cases = [
            (
                [7, 7, 3, 3, 3, -1],
                {"subcluster_1_1": (3, 4, 5), "subcluster_1_2": (1, 2)},
            ),
            (
                [5, 5, 2, 2, 9, 9],
                {
                    "subcluster_1_1": (1, 2),
                    "subcluster_1_2": (3, 4),
                    "subcluster_1_3": (5, 6),
                },
            ),
        ]
        edges = tuple((node, node + 1) for node in range(1, 6))
        for partition, members in cases:
            with self.subTest(partition=partition):
                viewer = network_viewer(7, edges, cluster_labels=[2, 1, 1, 1, 1, 1, 1])
                with mock.patch.dict(
                    sys.modules, {"graspologic_native": ModuleType("graspologic_native")}
                ), mock.patch.object(
                    subcluster_command.network_clustering,
                    "leiden_partition",
                    return_value=np.array(partition),
                ):
                    self.subcluster(["cluster_1", "leiden", "1.0", "2"], viewer)
                self.assert_groups(viewer, members)

    def test_mcl_inflation_must_lie_in_the_documented_range(self):
        for inflation in ("1.1", "10.0"):
            with self.subTest(inflation=inflation):
                _viewer, succeeded, failed, _output = self.subcluster(
                    ["cluster_1", "mcl", inflation, "1"]
                )
                failed.assert_not_called()
                succeeded.assert_called_once()

        for inflation in ("1.0", "10.01", "nan"):
            with self.subTest(inflation=inflation):
                viewer, succeeded, failed, output = self.subcluster(
                    ["cluster_1", "mcl", inflation, "1"]
                )
                message = (
                    "Error: MCL inflation must be between 1.1 and 10.0; "
                    f"got {float(inflation)}."
                )
                failed.assert_called_once_with(viewer, message)
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, message)
                self.assertNotIn("Running MCL", output)
                viewer._save_state.assert_not_called()
                self.assert_groups(viewer, {})

    def test_edge_orientation_does_not_matter(self):
        reversed_edges = tuple((v, u) for u, v in NETWORK_EDGES)
        for args in (["jaccard", "0.5", "3"], ["mcl", "2", "3"]):
            with self.subTest(args=args):
                viewer = network_viewer(
                    10, reversed_edges, cluster_labels=NETWORK_CLUSTERS
                )
                self.subcluster(["cluster_1", *args], viewer)
                self.assert_groups(
                    viewer,
                    {"subcluster_1_1": (2, 3, 4), "subcluster_1_2": (5, 6, 7)},
                )

    def test_mcl_weights_the_clusters_edges_by_their_own_scores(self):
        # Cluster 1 is the 4-cycle 1-2-3-4-1; the edges of cluster 2 and the
        # edges between clusters sit among its edges with scores of their own.
        edges = ((0, 5), (1, 2), (0, 1), (2, 3), (3, 4), (4, 5), (4, 1))
        cases = [
            (
                (0.9, 1.0, 0.5, 0.01, 1.0, 0.7, 0.01),
                {"subcluster_1_1": (1, 2), "subcluster_1_2": (3, 4)},
            ),
            (
                (0.9, 0.01, 0.5, 1.0, 0.01, 0.7, 1.0),
                {"subcluster_1_1": (1, 4), "subcluster_1_2": (2, 3)},
            ),
        ]
        for scores, members in cases:
            with self.subTest(scores=scores):
                viewer = network_viewer(6, edges, scores, cluster_labels=[2, 1, 1, 1, 1, 2])
                _viewer, _succeeded, failed, _output = self.subcluster(
                    ["cluster_1", "mcl", "2", "1"], viewer
                )
                failed.assert_not_called()
                self.assert_groups(viewer, members)

    def test_defaults_are_threshold_0_2_inflation_2_and_min_size_10(self):
        for mode, param in (("jaccard", "0.2"), ("MCL", "2.0")):
            with self.subTest(mode=mode):
                viewer, succeeded, _failed, output = self.subcluster(["cluster_1", mode])
                self.assert_groups(viewer, {})
                self.assertIn(
                    f"Running {mode.upper()} Subclustering for cluster_1 "
                    f"(Param={param}, MinSize=10)...",
                    output,
                )
                succeeded.assert_called_once_with(
                    viewer, f"Done! Found 0 subclusters in cluster_1 via {mode.upper()}."
                )

    def test_cluster_without_internal_edges_fails_unchanged(self):
        for mode in ("jaccard", "mcl"):
            with self.subTest(mode=mode):
                viewer, succeeded, failed, _output = self.subcluster(["cluster_3", mode])
                failed.assert_called_once_with(
                    viewer,
                    "Error: No edges exist within cluster_3 to perform subclustering.",
                )
                succeeded.assert_not_called()
                viewer._save_state.assert_not_called()
                self.assert_groups(viewer, {})

    def test_parameter_errors_leave_the_groups_unchanged(self):
        cases = [
            (["mcl", "two"], "Error: Parameter must be a number."),
            (["jaccard", "0.2", "big"], "Error: Min Size must be an integer."),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                viewer, succeeded, failed, _output = self.subcluster(["cluster_1", *args])
                failed.assert_called_once_with(viewer, message)
                succeeded.assert_not_called()
                viewer._save_state.assert_not_called()
                self.assert_groups(viewer, {})


class SubclusterParameterTests(unittest.TestCase):
    """Parameters the algorithms cannot use are refused before any change."""

    def assert_refused(self, args, message):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        with mock.patch.object(
            subcluster_command.network_clustering, "leiden_partition"
        ) as leiden, mock.patch.object(
            subcluster_command.network_clustering, "jaccard_partition"
        ) as jaccard, mock.patch.object(
            subcluster_command.network_clustering, "markov_clusters"
        ) as mcl:
            succeeded, failed, output = run_command(
                subcluster_command, viewer, ["cluster_1", *args]
            )
        failed.assert_called_once_with(viewer, message)
        succeeded.assert_not_called()
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(output.count(message), 1)
        for kernel in (leiden, jaccard, mcl):
            kernel.assert_not_called()
        viewer._save_state.assert_not_called()
        self.assertEqual(viewer.group_labels, [set() for _ in range(10)])

    def test_leiden_resolution_must_be_finite_and_above_zero(self):
        for resolution in ("nan", "inf", "0", "-1"):
            message = (
                "Error: Leiden resolution must be a finite number above 0; "
                f"got {float(resolution)}."
            )
            for args in (["leiden", resolution, "1"], [resolution, "1"]):
                with self.subTest(args=args):
                    self.assert_refused(args, message)

    def test_jaccard_threshold_must_lie_in_zero_to_one(self):
        for threshold in ("nan", "inf", "1.5", "-0.1"):
            with self.subTest(threshold=threshold):
                self.assert_refused(
                    ["jaccard", threshold, "1"],
                    "Error: Jaccard threshold must be between 0.0 and 1.0; "
                    f"got {float(threshold)}.",
                )

    def test_min_size_must_be_at_least_one(self):
        for min_size in ("0", "-3"):
            message = f"Error: Min Size must be at least 1; got {int(min_size)}."
            for args in (
                ["leiden", "1", min_size],
                ["jaccard", "0.2", min_size],
                ["mcl", "2", min_size],
            ):
                with self.subTest(args=args):
                    self.assert_refused(args, message)


class SubclusterMCLScoreTests(unittest.TestCase):
    # Cluster 1 is the 4-cycle 1-2-3-4-1; edges (0, 5) lie in cluster 2.
    EDGES = ((0, 5), (1, 2), (2, 3), (3, 4), (4, 1))
    CLUSTERS = [2, 1, 1, 1, 1, 2]

    def subcluster(self, scores):
        viewer = network_viewer(6, self.EDGES, scores, cluster_labels=self.CLUSTERS)
        succeeded, failed, output = run_command(
            subcluster_command, viewer, ["cluster_1", "mcl", "2", "1"]
        )
        return viewer, succeeded, failed, output

    def test_unusable_scores_inside_the_cluster_are_refused(self):
        for scores in (
            (1.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, -0.2, 1.0, 1.0),
            (1.0, 1.0, 1.0, float("nan"), 1.0),
            (1.0, 1.0, 1.0, 1.0, float("inf")),
        ):
            with self.subTest(scores=scores):
                viewer, succeeded, failed, output = self.subcluster(scores)
                first_line = (
                    "Error: MCL needs every edge score to be finite and above 0, "
                    "but found 1 edge with a score that is not."
                )
                failed.assert_called_once()
                self.assertEqual(failed.call_args.args[1].splitlines()[0], first_line)
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, first_line)
                self.assertEqual(output.count(first_line), 1)
                viewer._save_state.assert_not_called()
                self.assertEqual(viewer.group_labels, [set() for _ in range(6)])

    def test_scores_outside_the_cluster_do_not_matter(self):
        # Only the edge (0, 5) of cluster 2 has an unusable score.
        viewer, _succeeded, failed, _output = self.subcluster(
            (0.0, 1.0, 1.0, 1.0, 1.0)
        )
        failed.assert_not_called()
        self.assertEqual(sum(bool(groups) for groups in viewer.group_labels), 4)


class OverlappingMCLSubclusterTests(unittest.TestCase):
    def test_a_subcluster_left_below_min_size_by_an_overlap_is_noise(self):
        # MCL gives the path 0-1-2-3-4 the overlapping clusters (0, 1, 2) and
        # (2, 3, 4); the later one takes node 2, leaving (0, 1) below 3.
        edges = ((0, 1), (1, 2), (2, 3), (3, 4))
        viewer = network_viewer(5, edges, cluster_labels=[1] * 5)
        _succeeded, failed, _output = run_command(
            subcluster_command, viewer, ["cluster_1", "mcl", "2.0", "3"]
        )
        failed.assert_not_called()
        self.assertEqual(
            viewer.group_labels,
            [set(), set(), {"subcluster_1_1"}, {"subcluster_1_1"}, {"subcluster_1_1"}],
        )
        for node in (0, 1):
            np.testing.assert_array_equal(viewer.current_colors[node], GREY)


class SubclusterWithoutNumbaTests(unittest.TestCase):
    def test_jaccard_does_not_need_numba_to_run(self):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        with mock.patch.object(
            subcluster_command.network_clustering, "NUMBA_AVAILABLE", False
        ):
            succeeded, failed, output = run_command(
                subcluster_command, viewer, ["cluster_1", "jaccard", "0.5", "3"]
            )
        failed.assert_not_called()
        self.assertNotIn("Numba", output)
        self.assertEqual(
            viewer.group_labels[2:8],
            [
                {"subcluster_1_1"},
                {"subcluster_1_1"},
                {"subcluster_1_1"},
                {"subcluster_1_2"},
                {"subcluster_1_2"},
                {"subcluster_1_2"},
            ],
        )


class SubclusterSingleReportTests(unittest.TestCase):
    """A failure is recorded once, as the real command portal records it."""

    def record(self, viewer, args):
        record = {"messages": [], "outcome": None, "artifacts": [], "jobs": []}
        context = ExecutionContext(SimpleNamespace(viewer=viewer), "test", record)
        with bind(context), redirect_stdout(io.StringIO()):
            subcluster_command.run(viewer, args)
        return record

    def test_refusals_are_recorded_once(self):
        unclustered = network_viewer(10, NETWORK_EDGES)
        clustered = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        for viewer, args, text in (
            (
                clustered,
                ["cluster_x"],
                "Error: First argument must be 'clear' or a cluster name like 'cluster_N' (got 'cluster_x').",
            ),
            (unclustered, ["cluster_1"], "Error: No clusters are currently defined. Run 'cluster' first."),
            (clustered, ["cluster_9"], "Error: Cluster 9 is empty or does not exist."),
            (
                clustered,
                ["cluster_3"],
                "Error: No edges exist within cluster_3 to perform subclustering.",
            ),
            (
                clustered,
                ["cluster_1", "mcl", "0.5"],
                "Error: MCL inflation must be between 1.1 and 10.0; got 0.5.",
            ),
        ):
            with self.subTest(args=args):
                record = self.record(viewer, args)
                self.assertEqual(record["outcome"], "failed")
                self.assertEqual([m["text"] for m in record["messages"]], [text])
                self.assertEqual(record["messages"][0]["status"], "failed")


class SubclusterModuleTests(unittest.TestCase):
    def test_colour_codes_are_printed_only_for_a_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        escape = chr(27) + "["
        color_map = {1: (1.0, 0.0, 0.0)}
        with redirect_stdout(io.StringIO()):
            plain = subcluster_command.get_colored_subcluster_name(1, "name", color_map)
        with redirect_stdout(Terminal()):
            colored = subcluster_command.get_colored_subcluster_name(1, "name", color_map)
        self.assertEqual(plain, "● name")
        self.assertEqual(colored, f"{escape}38;2;255;0;0m●{escape}0m name")

    def test_the_windows_ansi_switch_is_made_once_per_process(self):
        flag = "_WINDOWS_ANSI_ENABLED"
        had_flag = flag in vars(subcluster_command)
        try:
            vars(subcluster_command).pop(flag, None)
            with mock.patch.object(sys, "platform", "win32"), mock.patch(
                "os.system"
            ) as system:
                importlib.reload(subcluster_command)
                importlib.reload(subcluster_command)
            system.assert_called_once_with("")
        finally:
            if had_flag:
                vars(subcluster_command)[flag] = True
            else:
                vars(subcluster_command).pop(flag, None)

    def test_help_states_what_is_clustered(self):
        output = io.StringIO()
        with redirect_stdout(output):
            subcluster_command.print_help()
        text = " ".join(output.getvalue().split())
        for line in (
            "regardless of the similarity slider and hidden nodes",
            "Jaccard compares closed neighbourhoods",
            "a node counts as its own neighbour",
            "an edge outside every triangle still scores above 0",
            "MCL gives every node a self-loop as heavy as its strongest edge",
            "Isolated nodes are always Noise in Leiden",
            "singleton subclusters in MCL and Jaccard when MIN_SIZE is 1",
        ):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
