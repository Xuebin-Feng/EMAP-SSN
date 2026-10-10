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
NOTHING_REACHED = (
    "No subcluster of cluster_1 reached the minimum size {size}; nothing was changed."
)


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
                # The cluster has 7 nodes, so no subcluster reaches 10.
                succeeded.assert_called_once_with(viewer, NOTHING_REACHED.format(size=10))

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


class LeidenSubclusterIsolateTests(unittest.TestCase):
    """A node with no edge inside the cluster is a singleton subcluster with a
    MIN_SIZE of 1 and Noise above it, in Leiden as in MCL and Jaccard."""

    def setUp(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")

    def subcluster(self, args):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        succeeded, failed, _output = run_command(
            subcluster_command, viewer, ["cluster_1", *args]
        )
        failed.assert_not_called()
        return viewer, succeeded

    def members(self, viewer):
        names = {}
        for node, groups in enumerate(viewer.group_labels):
            for name in groups:
                names.setdefault(name, []).append(node)
        return names

    def test_all_three_modes_treat_the_isolated_node_alike(self):
        # Node 8 has no edge inside cluster 1; its only edge leads out.
        triangles = {"subcluster_1_1": [2, 3, 4], "subcluster_1_2": [5, 6, 7]}
        modes = (("leiden", "1.0"), ("mcl", "2"), ("jaccard", "0.5"))
        for min_size, expected in (
            ("1", {**triangles, "subcluster_1_3": [8]}),
            ("2", triangles),
            ("3", triangles),
        ):
            for mode, param in modes:
                with self.subTest(mode=mode, min_size=min_size):
                    viewer, _succeeded = self.subcluster([mode, param, min_size])
                    self.assertEqual(self.members(viewer), expected)

    def test_a_leiden_singleton_is_coloured_and_noise_is_grey(self):
        viewer, succeeded = self.subcluster(["leiden", "1.0", "1"])
        succeeded.assert_called_once_with(
            viewer, "Done! Found 3 subclusters in cluster_1 via LEIDEN."
        )
        self.assertEqual(viewer.current_colors[8][3], 1.0)
        viewer, _succeeded = self.subcluster(["leiden", "1.0", "3"])
        np.testing.assert_array_equal(viewer.current_colors[8], GREY)


class SubclusterNameTests(unittest.TestCase):
    """The cluster is named as expressions name it, without leading zeros."""

    def assert_refused(self, viewer, args, message):
        with mock.patch.object(
            subcluster_command.network_clustering, "leiden_partition"
        ) as leiden, mock.patch.object(
            subcluster_command.network_clustering, "jaccard_partition"
        ) as jaccard, mock.patch.object(
            subcluster_command.network_clustering, "markov_clusters"
        ) as mcl:
            succeeded, failed, output = run_command(subcluster_command, viewer, args)
        failed.assert_called_once_with(viewer, message)
        succeeded.assert_not_called()
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(output.count(message), 1)
        for kernel in (leiden, jaccard, mcl):
            kernel.assert_not_called()
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_a_name_with_leading_zeros_is_refused_with_the_canonical_name(self):
        for name, canonical in (
            ("cluster_01", "cluster_1"),
            ("cluster_001", "cluster_1"),
            ("CLUSTER_01", "cluster_1"),
            ("cluster_010", "cluster_10"),
            ("cluster_00", "cluster_0"),
            ("cluster_0000", "cluster_0"),
        ):
            message = f"Error: Write the cluster as {canonical}, without leading zeros."
            for rest in ([], ["jaccard", "0.5", "3"], ["mcl", "2", "1"], ["leiden", "1", "1"]):
                with self.subTest(name=name, rest=rest):
                    viewer = network_viewer(
                        10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS
                    )
                    self.assert_refused(viewer, [name, *rest], message)
                    self.assertEqual(viewer.group_labels, [set() for _ in range(10)])

    def test_the_name_is_checked_before_the_clusters_are(self):
        viewer = network_viewer(10, NETWORK_EDGES)
        self.assert_refused(
            viewer, ["cluster_01"], "Error: Write the cluster as cluster_1, without leading zeros."
        )

    def test_the_canonical_name_is_accepted_in_any_case(self):
        for name in ("cluster_1", "Cluster_1", "CLUSTER_1"):
            with self.subTest(name=name):
                viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
                succeeded, failed, _output = run_command(
                    subcluster_command, viewer, [name, "jaccard", "0.5", "3"]
                )
                failed.assert_not_called()
                succeeded.assert_called_once_with(
                    viewer, "Done! Found 2 subclusters in cluster_1 via JACCARD."
                )

    def test_cluster_zero_is_still_an_empty_cluster(self):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        succeeded, failed, _output = run_command(subcluster_command, viewer, ["cluster_0"])
        failed.assert_called_once_with(viewer, "Error: Cluster 0 is empty or does not exist.")
        succeeded.assert_not_called()

    def test_other_malformed_names_keep_their_error(self):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        for name in ("cluster_", "cluster_x", "cluster-1", "#cluster_1#", "1"):
            with self.subTest(name=name):
                self.assert_refused(
                    viewer,
                    [name],
                    "Error: First argument must be 'clear' or a cluster name like "
                    f"'cluster_N' (got '{name}').",
                )


class NoSubclusterReachesMinSizeTests(unittest.TestCase):
    """A run in which no subcluster reaches MIN_SIZE changes nothing."""

    FIRST = ["cluster_1", "jaccard", "0.5", "3"]
    MEMBERS = {"subcluster_1_1": (2, 3, 4), "subcluster_1_2": (5, 6, 7)}

    def subclustered_viewer(self):
        """A viewer already subclustered into the barbell's two triangles."""
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        for node in (0, 1, 9):
            viewer.group_labels[node].add("mine")
        run_command(subcluster_command, viewer, self.FIRST)
        for name, nodes in self.MEMBERS.items():
            for node in nodes:
                self.assertIn(name, viewer.group_labels[node])
        viewer._save_state.reset_mock()
        viewer.update_nodes.reset_mock()
        return viewer

    def assert_unchanged_run(self, viewer, args, size):
        groups = [set(names) for names in viewer.group_labels]
        colors = viewer.current_colors.copy()
        labels = viewer.cluster_labels.copy()
        message = NOTHING_REACHED.format(size=size)

        succeeded, failed, output = run_command(subcluster_command, viewer, args)

        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, message)
        self.assertEqual(viewer.console_text.text, message)
        self.assertIn(message, output)
        self.assertNotIn("Stats for Cluster", output)
        self.assertEqual(viewer.group_labels, groups)
        np.testing.assert_array_equal(viewer.current_colors, colors)
        np.testing.assert_array_equal(viewer.cluster_labels, labels)
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_a_rerun_keeps_the_earlier_subclusters_and_colours(self):
        for mode, params in (("jaccard", ["0.5", "4"]), ("mcl", ["2", "4"])):
            with self.subTest(mode=mode):
                viewer = self.subclustered_viewer()
                self.assert_unchanged_run(viewer, ["cluster_1", mode, *params], 4)
                # The triangles keep their subcluster colours, none of them grey.
                for node in range(2, 8):
                    self.assertEqual(viewer.current_colors[node][3], 1.0)

    def test_a_leiden_rerun_keeps_them_too(self):
        viewer = self.subclustered_viewer()
        with mock.patch.dict(
            sys.modules, {"graspologic_native": ModuleType("graspologic_native")}
        ), mock.patch.object(
            subcluster_command.network_clustering,
            "leiden_partition",
            return_value=np.full(7, -1),
        ) as partition:
            self.assert_unchanged_run(viewer, ["cluster_1", "leiden", "1.0", "50"], 50)
        partition.assert_called_once()

    def test_a_first_run_changes_nothing_either(self):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        self.assert_unchanged_run(viewer, ["cluster_1", "jaccard", "0.5", "4"], 4)
        # The cluster's nodes are not greyed.
        np.testing.assert_array_equal(viewer.current_colors[2:9], np.zeros((7, 4)))

    def test_the_run_leaves_the_undo_history_alone(self):
        from EMAPSSN_Viewer import MainViewer

        n_nodes = 10
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = n_nodes
        viewer.full_headers = [f"n{node}" for node in range(n_nodes)]
        viewer.edges = np.array(NETWORK_EDGES, dtype=np.int32)
        viewer.pos = np.zeros((n_nodes, 2), dtype=np.float32)
        viewer.visible_mask = np.ones(n_nodes, dtype=bool)
        viewer.current_colors = np.zeros((n_nodes, 4))
        viewer.current_sizes = np.full(n_nodes, 10.0, dtype=np.float32)
        viewer.current_shapes = np.full(n_nodes, "disc", dtype=object)
        viewer.node_render_order = np.arange(n_nodes, dtype=np.int32)
        viewer.cluster_labels = np.array(NETWORK_CLUSTERS)
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        viewer.group_labels = [set() for _ in range(n_nodes)]
        viewer.metadata = {}
        viewer._cacheable_attrs = set()
        viewer.selected_indices = []
        viewer.position_history = []
        viewer.redo_stack = []
        viewer.console_text = SimpleNamespace(text="")
        viewer.update_nodes = mock.Mock()
        viewer.update_selection_visual = mock.Mock()
        viewer.update_edges = mock.Mock()
        viewer.broadcast_event = mock.Mock()

        run_command(subcluster_command, viewer, self.FIRST)
        self.assertEqual(len(viewer.position_history), 1)
        groups = [set(names) for names in viewer.group_labels]

        succeeded, failed, _output = run_command(
            subcluster_command, viewer, ["cluster_1", "jaccard", "0.5", "4"]
        )

        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, NOTHING_REACHED.format(size=4))
        self.assertEqual(len(viewer.position_history), 1)
        self.assertEqual(viewer.group_labels, groups)

    def test_a_run_that_does_reach_min_size_still_replaces_the_labels(self):
        viewer = self.subclustered_viewer()
        succeeded, failed, _output = run_command(
            subcluster_command, viewer, ["cluster_1", "jaccard", "0.2", "6"]
        )
        failed.assert_not_called()
        succeeded.assert_called_once_with(
            viewer, "Done! Found 1 subcluster in cluster_1 via JACCARD."
        )
        viewer._save_state.assert_called_once_with()
        self.assertEqual(
            [sorted(names) for names in viewer.group_labels[2:8]],
            [["subcluster_1_1"]] * 6,
        )
        self.assertEqual(viewer.group_labels[0], {"mine"})


class SubclusterExtraArgumentTests(unittest.TestCase):
    """An argument after those the command takes is refused, naming the first
    one, before any change; the arguments it does take are read as before."""

    USAGE = "Usage: subcluster <CLUSTER_NAME> [MODE] [PARAM_1] [MIN_SIZE]"

    def viewer(self):
        viewer = network_viewer(10, NETWORK_EDGES, cluster_labels=NETWORK_CLUSTERS)
        viewer.group_labels[2].update({"subcluster_1_1", "mine"})
        return viewer

    def assert_refused(self, args, argument, usage=USAGE):
        viewer = self.viewer()
        groups = [set(names) for names in viewer.group_labels]
        with mock.patch.object(
            subcluster_command.network_clustering, "leiden_partition"
        ) as leiden, mock.patch.object(
            subcluster_command.network_clustering, "jaccard_partition"
        ) as jaccard, mock.patch.object(
            subcluster_command.network_clustering, "markov_clusters"
        ) as mcl:
            succeeded, failed, output = run_command(subcluster_command, viewer, args)
        first_line = f"Error: Unrecognized subcluster argument '{argument}'."
        failed.assert_called_once_with(viewer, f"{first_line}\n{usage}")
        succeeded.assert_not_called()
        # The console line shows the first line; the terminal all of it.
        self.assertEqual(viewer.console_text.text, first_line)
        self.assertEqual(output.count(first_line), 1)
        self.assertIn(usage, output)
        for kernel in (leiden, jaccard, mcl):
            kernel.assert_not_called()
        self.assertEqual(viewer.group_labels, groups)
        np.testing.assert_array_equal(viewer.current_colors, np.zeros((10, 4)))
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_extra_tokens_after_each_form_are_refused_naming_the_first(self):
        cases = [
            (["cluster_1", "leiden", "1.0", "10", "foo"], "foo"),
            (["cluster_1", "leiden", "1.0", "10", "foo", "bar"], "foo"),
            (["cluster_1", "Leiden", "1.0", "10", "Foo"], "Foo"),
            (["cluster_1", "mcl", "2", "3", "x"], "x"),
            (["cluster_1", "MCL", "2", "3", "x", "y", "z"], "x"),
            (["cluster_1", "jaccard", "0.5", "3", "extra", "tokens"], "extra"),
            (["cluster_1", "jaccard", "0.5", "3", "4"], "4"),
            # MODE omitted: PARAM_1 and MIN_SIZE take two arguments.
            (["cluster_1", "1.0", "10", "foo"], "foo"),
            (["cluster_1", "1.0", "10", "foo", "bar"], "foo"),
            (["cluster_1", "1.5", "3", "leiden"], "leiden"),
        ]
        for args, argument in cases:
            with self.subTest(args=args):
                self.assert_refused(args, argument)

    def test_extra_tokens_after_clear_are_refused(self):
        for args, argument in ((["clear", "foo"], "foo"), (["CLEAR", "1", "2"], "1")):
            with self.subTest(args=args):
                self.assert_refused(args, argument, "Usage: subcluster clear")

    def test_a_parameter_error_before_the_extra_token_is_reported_first(self):
        viewer = self.viewer()
        succeeded, failed, _output = run_command(
            subcluster_command, viewer, ["cluster_1", "jaccard", "many", "3", "extra"]
        )
        failed.assert_called_once_with(viewer, "Error: Parameter must be a number.")
        succeeded.assert_not_called()

    def test_every_form_without_extras_is_read_as_before(self):
        cases = [
            (["cluster_1"], 1.0, 10),
            (["cluster_1", "leiden"], 1.0, 10),
            (["cluster_1", "leiden", "1.5"], 1.5, 10),
            (["cluster_1", "leiden", "1.5", "4"], 1.5, 4),
            (["cluster_1", "1.5"], 1.5, 10),
            (["cluster_1", "1.5", "4"], 1.5, 4),
            (["cluster_1", "LEIDEN", "0.5", "1"], 0.5, 1),
        ]
        for args, resolution, min_size in cases:
            with self.subTest(args=args):
                viewer = self.viewer()
                with mock.patch.dict(
                    sys.modules, {"graspologic_native": ModuleType("graspologic_native")}
                ), mock.patch.object(
                    subcluster_command.network_clustering,
                    "leiden_partition",
                    return_value=np.array([1, 1, 1, 2, 2, 2, -1]),
                ) as partition:
                    _succeeded, failed, _output = run_command(
                        subcluster_command, viewer, args
                    )
                failed.assert_not_called()
                partition.assert_called_once()
                self.assertEqual(partition.call_args.args[3:5], (resolution, min_size))
        for args, line in (
            (["cluster_1", "jaccard", "0.5", "3"], "(Param=0.5, MinSize=3)"),
            (["cluster_1", "mcl", "3", "3"], "(Param=3.0, MinSize=3)"),
        ):
            with self.subTest(args=args):
                _succeeded, failed, output = run_command(
                    subcluster_command, self.viewer(), args
                )
                failed.assert_not_called()
                self.assertIn(line, output)

    def test_help_is_unchanged(self):
        for args in (["help"], ["-h"], ["--help"], ["help", "mcl"], []):
            with self.subTest(args=args):
                viewer = self.viewer()
                succeeded, failed, _output = run_command(subcluster_command, viewer, args)
                failed.assert_not_called()
                succeeded.assert_called_once()
                viewer._save_state.assert_not_called()

    def test_clear_without_extras_still_clears(self):
        viewer = self.viewer()
        succeeded, failed, _output = run_command(subcluster_command, viewer, ["clear"])
        failed.assert_not_called()
        succeeded.assert_called_once()
        self.assertEqual(viewer.group_labels[2], {"mine"})


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
                ["cluster_01"],
                "Error: Write the cluster as cluster_1, without leading zeros.",
            ),
            (
                clustered,
                ["cluster_1", "jaccard", "0.5", "3", "foo"],
                "Error: Unrecognized subcluster argument 'foo'.\n"
                "Usage: subcluster <CLUSTER_NAME> [MODE] [PARAM_1] [MIN_SIZE]",
            ),
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
            "Isolated nodes (no edge within the cluster) are alike in all three modes",
            "singleton subclusters when MIN_SIZE is 1, Noise when it is larger",
            "cluster_01 is refused",
            "When no subcluster reaches MIN_SIZE, nothing changes",
            "Anything after the arguments above (or after clear) is refused",
        ):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
