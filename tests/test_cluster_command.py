"""The cluster command (commands/cluster.py): Leiden, Jaccard and MCL
partitions, their parameters and defaults, and the size order clusters are
numbered in."""
import importlib
import io
import os
import sys
import unittest
import warnings
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Command_Engine.command_succeeded() imports Viewer_Command_Portal (and with it
# PySide6) on first use. Imported here, that first import cannot happen inside
# a sys.modules window below, whose exit would drop the modules again.
import Viewer_Command_Portal  # noqa: F401
from commands import cluster
from tests.command_fixtures import BARBELL_EDGES, network_viewer, run_command


class ClusterNumberingTests(unittest.TestCase):
    def test_leiden_filters_small_communities_and_keeps_isolates_as_noise(self):
        fake_leiden = mock.Mock(
            return_value=(
                None,
                {"0": 10, "1": 10, "2": 10, "3": 20, "4": 20},
            )
        )
        with mock.patch.dict(
            sys.modules,
            {"graspologic_native": SimpleNamespace(leiden=fake_leiden)},
        ):
            labels = cluster.network_clustering.leiden_partition(
                6,
                np.asarray([[0, 1], [1, 2], [3, 4]], dtype=np.int32),
                None,
                resolution=1.0,
                min_size=3,
            )

        np.testing.assert_array_equal(labels, [1, 1, 1, -1, -1, -1])
        self.assertEqual(fake_leiden.call_args.kwargs["seed"], 42)

    def test_fast_jaccard_filter_uses_closed_neighbour_union(self):
        # A triangle 0-1-2 with node 3 hanging from node 2, as CSR rows:
        # 0: 1 2, 1: 0 2, 2: 0 1 3, 3: 2. The closed neighbourhoods are
        # {0, 1, 2} for 0 and 1, {0, 1, 2, 3} for 2 and {2, 3} for 3, so the
        # edges score 3/3, 3/4 and 2/4.
        edges = np.asarray([[0, 1], [0, 2], [2, 3]], dtype=np.int32)
        indptr = np.asarray([0, 2, 4, 7, 8], dtype=np.int32)
        indices = np.asarray([1, 2, 0, 2, 0, 1, 3, 2], dtype=np.int32)

        for threshold, expected in (
            (0.4, [True, True, True]),
            (0.5, [True, True, True]),
            (0.6, [True, True, False]),
            (0.75, [True, True, False]),
            (0.8, [True, False, False]),
            (1.0, [True, False, False]),
        ):
            with self.subTest(threshold=threshold):
                result = cluster.network_clustering.fast_jaccard_filter(
                    edges, indptr, indices, threshold
                )
                np.testing.assert_array_equal(result, expected)

    def test_clusters_are_numbered_from_largest_to_smallest(self):
        labels = np.array([9, 9, 9, 4, 4, -1, 7, 7, 7, 7, 2])

        renumbered = cluster.renumber_clusters_by_size(labels)

        np.testing.assert_array_equal(
            renumbered,
            [2, 2, 2, 3, 3, -1, 1, 1, 1, 1, 4],
        )
        np.testing.assert_array_equal(
            labels,
            [9, 9, 9, 4, 4, -1, 7, 7, 7, 7, 2],
        )

    def test_equal_sizes_use_lowest_member_node_index(self):
        labels = np.array([8, 8, 3, 3, -1, 5, 5])

        renumbered = cluster.renumber_clusters_by_size(labels)

        np.testing.assert_array_equal(renumbered, [1, 1, 2, 2, -1, 3, 3])

    def test_noise_only_input_remains_noise(self):
        labels = np.full(4, -1, dtype=int)

        renumbered = cluster.renumber_clusters_by_size(labels)

        np.testing.assert_array_equal(renumbered, labels)

    def test_cluster_command_canonicalizes_algorithm_labels(self):
        raw_labels = np.array([4, 4, 4, 9, 9, 2, 2, 2, 2, -1])
        viewer = SimpleNamespace(
            n_nodes=len(raw_labels),
            edges=np.array([[0, 1]], dtype=np.int32),
            console_text=SimpleNamespace(text=""),
            current_colors=np.zeros((len(raw_labels), 4), dtype=float),
            _save_state=mock.Mock(),
            update_nodes=mock.Mock(),
        )

        with mock.patch.dict(
            sys.modules,
            {"graspologic_native": SimpleNamespace()},
        ), mock.patch.object(
            cluster.network_clustering,
            "leiden_partition",
            return_value=raw_labels,
        ) as leiden_partition, mock.patch("builtins.print"):
            cluster.run(viewer, ["leiden", "1.0", "1"])

        np.testing.assert_array_equal(
            viewer.cluster_labels,
            [2, 2, 2, 3, 3, 1, 1, 1, 1, -1],
        )
        leiden_partition.assert_called_once()
        self.assertEqual(leiden_partition.call_args.kwargs["seed"], 42)
        viewer._save_state.assert_called_once_with()
        viewer.update_nodes.assert_called_once_with()


class JaccardAndMCLModeTests(unittest.TestCase):
    """cluster jaccard and cluster mcl on the barbell (BARBELL_EDGES) plus an
    isolated node 6, and on other small graphs."""

    def cluster(self, args, n_nodes=7, edges=BARBELL_EDGES, edge_scores=None):
        viewer = network_viewer(n_nodes, edges, edge_scores)
        succeeded, failed, _output = run_command(cluster, viewer, args)
        failed.assert_not_called()
        return viewer, succeeded

    def test_jaccard_keeps_edges_at_or_above_the_threshold(self):
        # Closed neighbourhoods: 0-1 and 4-5 score 3/3 = 1, the four triangle
        # edges that touch the bridge 3/4, and the bridge 2/6 = 1/3.
        cases = [
            # Every edge, the bridge included, is kept at or below 1/3.
            ("0", "3", [1, 1, 1, 1, 1, 1, -1]),
            ("0.2", "3", [1, 1, 1, 1, 1, 1, -1]),
            ("0.33", "3", [1, 1, 1, 1, 1, 1, -1]),
            # Above 1/3 the bridge is cut; 3/4 itself is kept.
            ("0.34", "3", [1, 1, 1, 2, 2, 2, -1]),
            ("0.5", "3", [1, 1, 1, 2, 2, 2, -1]),
            ("0.75", "3", [1, 1, 1, 2, 2, 2, -1]),
            # Only 0-1 and 4-5 (1) remain above 3/4.
            ("0.76", "2", [1, 1, -1, -1, 2, 2, -1]),
            # Edges that score 1 are kept at the highest threshold; every other
            # node, the isolated one included, is a singleton that a MIN_SIZE
            # of 1 keeps.
            ("1", "1", [1, 1, 3, 4, 2, 2, 5]),
        ]
        for threshold, min_size, labels in cases:
            with self.subTest(threshold=threshold, min_size=min_size):
                viewer, succeeded = self.cluster(["jaccard", threshold, min_size])
                np.testing.assert_array_equal(viewer.cluster_labels, labels)
                self.assertEqual(
                    viewer.last_cluster_params,
                    (f"JACCARD_{float(threshold)}", int(min_size)),
                )
                clusters = len(set(labels) - {-1})
                noun = "cluster" if clusters == 1 else "clusters"
                succeeded.assert_called_once_with(
                    viewer, f"Done! Found {clusters} {noun} via JACCARD."
                )
                viewer._save_state.assert_called_once_with()

    def test_mcl_splits_the_barbell_at_its_bridge(self):
        cases = [
            ("2.0", "3", [1, 1, 1, 2, 2, 2, -1]),
            # MCL keeps the isolated node as its own cluster.
            ("2.0", "1", [1, 1, 1, 2, 2, 2, 3]),
            ("1.5", "1", [1, 1, 1, 2, 2, 2, 3]),
            ("4", "1", [1, 1, 1, 2, 2, 2, 3]),
            # Near 1, inflation no longer separates the triangles: the
            # connected barbell stays one cluster.
            ("1.2", "1", [1, 1, 1, 1, 1, 1, 2]),
        ]
        for inflation, min_size, labels in cases:
            with self.subTest(inflation=inflation, min_size=min_size):
                viewer, succeeded = self.cluster(["MCL", inflation, min_size])
                np.testing.assert_array_equal(viewer.cluster_labels, labels)
                self.assertEqual(
                    viewer.last_cluster_params, (f"MCL_{float(inflation)}", int(min_size))
                )
                clusters = len(set(labels) - {-1})
                succeeded.assert_called_once_with(
                    viewer, f"Done! Found {clusters} clusters via MCL."
                )

    def test_mcl_does_not_depend_on_the_scale_of_the_scores(self):
        # The path 0-...-5 scored with identities (0-1) and with scores 300
        # times larger, as -log10 of E-values are: each node's self-loop
        # takes its strongest edge, so the two scales cluster alike.
        path = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
        identities = np.array([0.9, 0.8, 0.7, 0.8, 0.9])
        for scale in (1, 300, 0.01):
            with self.subTest(scale=scale):
                viewer, _succeeded = self.cluster(
                    ["mcl", "2", "1"], n_nodes=6, edges=path, edge_scores=identities * scale
                )
                np.testing.assert_array_equal(viewer.cluster_labels, [1, 1, 1, 2, 2, 2])

    def test_jaccard_keeps_pairs_stars_and_chains_out_of_noise(self):
        # An isolated pair (index 1), a star (1/2 per spoke) and a chain (2/3,
        # 1/2, 2/3) all score above 0 on closed neighbourhoods, and node 10
        # has no edge.
        edges = ((0, 1), (2, 3), (2, 4), (2, 5), (6, 7), (7, 8), (8, 9))
        cases = [
            ("0.5", "2", [3, 3, 1, 1, 1, 1, 2, 2, 2, 2, -1]),
            ("0.6", "2", [1, 1, -1, -1, -1, -1, 2, 2, 3, 3, -1]),
            ("1", "2", [1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
        ]
        for threshold, min_size, labels in cases:
            with self.subTest(threshold=threshold):
                viewer, _succeeded = self.cluster(
                    ["jaccard", threshold, min_size], n_nodes=11, edges=edges
                )
                np.testing.assert_array_equal(viewer.cluster_labels, labels)

    def test_mcl_inflation_must_lie_in_the_documented_range(self):
        # The bounds 1.1 and 10.0 are allowed.
        for inflation, labels in (
            ("1.1", [1, 1, 1, 1, 1, 1, 2]),
            ("10.0", [1, 1, 1, 2, 2, 2, 3]),
        ):
            with self.subTest(inflation=inflation):
                viewer, _succeeded = self.cluster(["mcl", inflation, "1"])
                np.testing.assert_array_equal(viewer.cluster_labels, labels)

        for inflation in ("1.0", "10.01", "0", "-2", "nan", "inf"):
            with self.subTest(inflation=inflation):
                viewer = network_viewer(7, BARBELL_EDGES)
                succeeded, failed, output = run_command(
                    cluster, viewer, ["mcl", inflation, "1"]
                )
                message = (
                    "Error: MCL inflation must be between 1.1 and 10.0; "
                    f"got {float(inflation)}."
                )
                failed.assert_called_once_with(viewer, message)
                succeeded.assert_not_called()
                self.assertEqual(viewer.console_text.text, message)
                self.assertIn(message, output)
                self.assertNotIn("Running MCL", output)
                self.assertFalse(hasattr(viewer, "cluster_labels"))
                viewer._save_state.assert_not_called()
                viewer.update_nodes.assert_not_called()

    def test_mcl_weights_edges_by_their_scores(self):
        # The 4-cycle 0-1-2-3-0 splits along its weak edges.
        cycle = ((0, 1), (1, 2), (2, 3), (3, 0))
        cases = [
            ([1.0, 0.01, 1.0, 0.01], [1, 1, 2, 2]),
            ([0.01, 1.0, 0.01, 1.0], [1, 2, 2, 1]),
            (None, [1, 1, 1, 1]),
        ]
        for scores, labels in cases:
            with self.subTest(scores=scores):
                viewer, _succeeded = self.cluster(
                    ["mcl", "2", "1"], n_nodes=4, edges=cycle, edge_scores=scores
                )
                np.testing.assert_array_equal(viewer.cluster_labels, labels)

    def test_edge_orientation_does_not_matter(self):
        reversed_edges = tuple((v, u) for u, v in BARBELL_EDGES)
        for args in (["jaccard", "0.5", "3"], ["mcl", "2", "3"]):
            with self.subTest(args=args):
                viewer, _succeeded = self.cluster(args, edges=reversed_edges)
                np.testing.assert_array_equal(
                    viewer.cluster_labels, [1, 1, 1, 2, 2, 2, -1]
                )

    def test_larger_clusters_get_lower_numbers(self):
        # A triangle on nodes 0-2 and a K4 on 3-6: the K4 is found second but
        # is the larger cluster. Every edge of a clique scores 1.
        triangle_and_k4 = (
            (0, 1), (0, 2), (1, 2),
            (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6),
        )
        for args in (["jaccard", "0.2", "3"], ["mcl", "2", "3"]):
            with self.subTest(args=args):
                viewer, _succeeded = self.cluster(args, edges=triangle_and_k4)
                np.testing.assert_array_equal(
                    viewer.cluster_labels, [2, 2, 2, 1, 1, 1, 1]
                )

    def test_defaults_are_threshold_0_2_inflation_2_and_min_size_10(self):
        for mode, params in (("jaccard", ("JACCARD_0.2", 10)), ("mcl", ("MCL_2.0", 10))):
            with self.subTest(mode=mode):
                viewer, succeeded = self.cluster([mode])
                # Every group of the barbell is below the default MIN_SIZE.
                np.testing.assert_array_equal(viewer.cluster_labels, [-1] * 7)
                self.assertEqual(viewer.last_cluster_params, params)
                succeeded.assert_called_once_with(
                    viewer, f"Done! Found 0 clusters via {mode.upper()}."
                )

    def test_networks_without_edges_keep_each_node_alone(self):
        for mode in ("jaccard", "mcl"):
            for n_nodes, labels in ((1, [1]), (3, [1, 2, 3])):
                with self.subTest(mode=mode, n_nodes=n_nodes):
                    viewer, _succeeded = self.cluster(
                        [mode, "2" if mode == "mcl" else "0.2", "1"],
                        n_nodes=n_nodes,
                        edges=(),
                    )
                    np.testing.assert_array_equal(viewer.cluster_labels, labels)

    def test_nodes_take_their_clusters_colors_and_noise_is_grey(self):
        viewer, _succeeded = self.cluster(["jaccard", "0.5", "3"])
        colors = viewer.current_colors
        np.testing.assert_array_equal(colors[6], (0.8, 0.8, 0.8, 0.4))
        for members in ((0, 1, 2), (3, 4, 5)):
            for node in members:
                np.testing.assert_array_equal(colors[node], colors[members[0]])
            self.assertEqual(colors[members[0]][3], 1.0)
        self.assertFalse(np.array_equal(colors[0], colors[3]))
        viewer.update_nodes.assert_called_once_with()

    def test_parameter_errors_leave_the_viewer_unchanged(self):
        cases = [
            (["mcl", "two"], "Error: Parameter must be a number."),
            (["jaccard", "0.2", "big"], "Error: Min Size must be an integer."),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                viewer = network_viewer(7, BARBELL_EDGES)
                succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_called_once_with(viewer, message)
                succeeded.assert_not_called()
                self.assertFalse(hasattr(viewer, "cluster_labels"))
                viewer._save_state.assert_not_called()


PATH_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4))


class ParameterValidationTests(unittest.TestCase):
    """Parameters the algorithms cannot use are refused before any change."""

    def assert_refused(self, args, message, **viewer_options):
        viewer = network_viewer(7, BARBELL_EDGES, **viewer_options)
        with mock.patch.object(
            cluster.network_clustering, "leiden_partition"
        ) as leiden, mock.patch.object(
            cluster.network_clustering, "jaccard_partition"
        ) as jaccard, mock.patch.object(
            cluster.network_clustering, "markov_clusters"
        ) as mcl:
            succeeded, failed, output = run_command(cluster, viewer, args)
        failed.assert_called_once_with(viewer, message)
        succeeded.assert_not_called()
        self.assertEqual(viewer.console_text.text, message)
        self.assertEqual(output.count(message), 1)
        for kernel in (leiden, jaccard, mcl):
            kernel.assert_not_called()
        self.assertFalse(hasattr(viewer, "cluster_labels"))
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_leiden_resolution_must_be_finite_and_above_zero(self):
        for resolution in ("nan", "inf", "-inf", "0", "-1", "-0.5"):
            message = (
                "Error: Leiden resolution must be a finite number above 0; "
                f"got {float(resolution)}."
            )
            for args in (["leiden", resolution, "1"], [resolution, "1"], [resolution]):
                with self.subTest(args=args):
                    self.assert_refused(args, message)

    def test_jaccard_threshold_must_lie_in_zero_to_one(self):
        for threshold in ("nan", "inf", "1.5", "-0.1", "-inf"):
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
                ["1.0", min_size],
                ["jaccard", "0.2", min_size],
                ["mcl", "2", min_size],
            ):
                with self.subTest(args=args):
                    self.assert_refused(args, message)


class ExtraArgumentTests(unittest.TestCase):
    """An argument after those the command takes is refused, naming the first
    one, before any change; the arguments it does take are read as before."""

    USAGE = "Usage: cluster [MODE] [PARAM_1] [MIN_SIZE]"

    def assert_refused(self, args, argument, usage=USAGE):
        viewer = network_viewer(7, BARBELL_EDGES)
        with mock.patch.object(
            cluster.network_clustering, "leiden_partition"
        ) as leiden, mock.patch.object(
            cluster.network_clustering, "jaccard_partition"
        ) as jaccard, mock.patch.object(
            cluster.network_clustering, "markov_clusters"
        ) as mcl:
            succeeded, failed, output = run_command(cluster, viewer, args)
        first_line = f"Error: Unrecognized cluster argument '{argument}'."
        failed.assert_called_once_with(viewer, f"{first_line}\n{usage}")
        succeeded.assert_not_called()
        # The console line shows the first line; the terminal all of it.
        self.assertEqual(viewer.console_text.text, first_line)
        self.assertEqual(output.count(first_line), 1)
        self.assertIn(usage, output)
        for kernel in (leiden, jaccard, mcl):
            kernel.assert_not_called()
        self.assertFalse(hasattr(viewer, "cluster_labels"))
        viewer._save_state.assert_not_called()
        viewer.update_nodes.assert_not_called()

    def test_extra_tokens_after_each_form_are_refused_naming_the_first(self):
        cases = [
            (["leiden", "1.0", "10", "foo"], "foo"),
            (["leiden", "1.0", "10", "foo", "bar"], "foo"),
            (["Leiden", "1.0", "10", "Foo"], "Foo"),
            (["mcl", "2", "3", "x"], "x"),
            (["MCL", "2", "3", "x", "y", "z"], "x"),
            (["jaccard", "0.5", "3", "extra", "tokens"], "extra"),
            (["jaccard", "0.5", "3", "4"], "4"),
            # MODE omitted: PARAM_1 and MIN_SIZE take two arguments.
            (["1.0", "10", "foo"], "foo"),
            (["1.0", "10", "foo", "bar"], "foo"),
            (["1.5", "3", "leiden"], "leiden"),
        ]
        for args, argument in cases:
            with self.subTest(args=args):
                self.assert_refused(args, argument)

    def test_extra_tokens_after_list_are_refused(self):
        for args, argument in ((["list", "foo"], "foo"), (["LIST", "1", "2"], "1")):
            with self.subTest(args=args):
                self.assert_refused(args, argument, "Usage: cluster list")

    def test_a_parameter_error_before_the_extra_token_is_reported_first(self):
        viewer = network_viewer(7, BARBELL_EDGES)
        succeeded, failed, _output = run_command(
            cluster, viewer, ["jaccard", "many", "3", "extra"]
        )
        failed.assert_called_once_with(viewer, "Error: Parameter must be a number.")
        succeeded.assert_not_called()

    def test_every_form_without_extras_is_read_as_before(self):
        cases = [
            ([], 1.0, 10),
            (["leiden"], 1.0, 10),
            (["leiden", "1.5"], 1.5, 10),
            (["leiden", "1.5", "4"], 1.5, 4),
            (["1.5"], 1.5, 10),
            (["1.5", "4"], 1.5, 4),
            (["LEIDEN", "0.5", "1"], 0.5, 1),
        ]
        for args, resolution, min_size in cases:
            with self.subTest(args=args):
                viewer = network_viewer(7, BARBELL_EDGES)
                with mock.patch.dict(
                    sys.modules, {"graspologic_native": SimpleNamespace()}
                ), mock.patch.object(
                    cluster.network_clustering,
                    "leiden_partition",
                    return_value=np.array([1, 1, 1, 2, 2, 2, -1]),
                ) as partition:
                    _succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_not_called()
                partition.assert_called_once()
                self.assertEqual(partition.call_args.args[3:5], (resolution, min_size))
        for args, params in (
            (["jaccard"], ("JACCARD_0.2", 10)),
            (["jaccard", "0.5"], ("JACCARD_0.5", 10)),
            (["jaccard", "0.5", "3"], ("JACCARD_0.5", 3)),
            (["mcl"], ("MCL_2.0", 10)),
            (["mcl", "3"], ("MCL_3.0", 10)),
            (["mcl", "3", "3"], ("MCL_3.0", 3)),
        ):
            with self.subTest(args=args):
                viewer = network_viewer(7, BARBELL_EDGES)
                _succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_not_called()
                self.assertEqual(viewer.last_cluster_params, params)

    def test_help_and_list_are_unchanged(self):
        for args in (["help"], ["-h"], ["--help"], ["help", "leiden"]):
            with self.subTest(args=args):
                viewer = network_viewer(7, BARBELL_EDGES)
                succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_not_called()
                succeeded.assert_called_once()
                viewer._save_state.assert_not_called()
        viewer = network_viewer(7, BARBELL_EDGES, cluster_labels=[1, 1, 1, 2, 2, 2, -1])
        succeeded, failed, _output = run_command(cluster, viewer, ["list"])
        failed.assert_not_called()
        succeeded.assert_called_once_with(viewer, "Listed 2 clusters in console.")


class IsolatedNodeTests(unittest.TestCase):
    """A node with no edge is a singleton cluster with a MIN_SIZE of 1 and Noise
    above it, in Leiden as in MCL and Jaccard."""

    def setUp(self):
        try:
            import graspologic_native  # noqa: F401
        except ImportError:
            self.skipTest("graspologic_native is not installed")

    def cluster(self, args, n_nodes=7, edges=BARBELL_EDGES):
        viewer = network_viewer(n_nodes, edges)
        _succeeded, failed, _output = run_command(cluster, viewer, args)
        failed.assert_not_called()
        return viewer

    def test_all_three_modes_treat_the_isolated_node_alike(self):
        # Node 6 of the barbell network has no edge.
        modes = (("leiden", "1.0"), ("mcl", "2.0"), ("jaccard", "0.5"))
        for min_size, labels in (
            ("1", [1, 1, 1, 2, 2, 2, 3]),
            ("2", [1, 1, 1, 2, 2, 2, -1]),
            ("3", [1, 1, 1, 2, 2, 2, -1]),
        ):
            for mode, param in modes:
                with self.subTest(mode=mode, min_size=min_size):
                    viewer = self.cluster([mode, param, min_size])
                    np.testing.assert_array_equal(viewer.cluster_labels, labels)

    def test_a_leiden_singleton_is_grey_only_when_it_is_noise(self):
        kept = self.cluster(["leiden", "1.0", "1"])
        self.assertEqual(kept.current_colors[6][3], 1.0)
        self.assertFalse(np.array_equal(kept.current_colors[6], kept.current_colors[0]))
        noise = self.cluster(["leiden", "1.0", "2"])
        np.testing.assert_array_equal(noise.current_colors[6], (0.8, 0.8, 0.8, 0.4))

    def test_singletons_are_numbered_last_by_size_then_lowest_node_index(self):
        # Nodes 0 and 5 have no edge; 1-2 and 3-4 are pairs. The pairs are
        # larger, so they come first; the singletons follow in node order.
        for mode, param in (("leiden", "1.0"), ("mcl", "2.0")):
            with self.subTest(mode=mode):
                viewer = self.cluster([mode, param, "1"], 6, ((1, 2), (3, 4)))
                np.testing.assert_array_equal(
                    viewer.cluster_labels, [3, 1, 1, 2, 2, 4]
                )

    def test_a_network_of_isolated_nodes_in_leiden(self):
        viewer = self.cluster(["leiden", "1.0", "1"], 3, ())
        np.testing.assert_array_equal(viewer.cluster_labels, [1, 2, 3])
        viewer = self.cluster(["leiden", "1.0", "2"], 3, ())
        np.testing.assert_array_equal(viewer.cluster_labels, [-1, -1, -1])

    def test_the_report_counts_the_singletons(self):
        viewer = network_viewer(7, BARBELL_EDGES)
        succeeded, _failed, output = run_command(cluster, viewer, ["leiden", "1.0", "1"])
        succeeded.assert_called_once_with(viewer, "Done! Found 3 clusters via LEIDEN.")
        self.assertIn("Cluster 3", output)


class MCLEdgeScoreTests(unittest.TestCase):
    """MCL refuses edge scores it cannot use; Leiden is left as it was."""

    CYCLE = ((0, 1), (1, 2), (2, 3), (3, 0))

    def test_scores_of_zero_or_below_and_non_finite_scores_are_refused(self):
        for scores, bad in (
            ([1.0, 0.0, 1.0, 1.0], 1),
            ([1.0, -0.3, 1.0, 1.0], 1),
            ([float("nan"), 1.0, 1.0, 1.0], 1),
            ([1.0, 1.0, float("inf"), 1.0], 1),
            ([-1.0, -2.0, float("nan"), 1.0], 3),
        ):
            with self.subTest(scores=scores):
                viewer = network_viewer(4, self.CYCLE, scores)
                succeeded, failed, output = run_command(cluster, viewer, ["mcl", "1.5", "1"])
                noun = "edge" if bad == 1 else "edges"
                first_line = (
                    "Error: MCL needs every edge score to be finite and above 0, "
                    f"but found {bad} {noun} with a score that is not."
                )
                failed.assert_called_once()
                reported = failed.call_args.args[1]
                self.assertEqual(reported.splitlines()[0], first_line)
                self.assertIn("MCL cannot use a score of 0 or below", reported)
                succeeded.assert_not_called()
                # The console line shows the first line; the terminal all of it.
                self.assertEqual(viewer.console_text.text, first_line)
                self.assertEqual(output.count(first_line), 1)
                self.assertIn("MCL cannot use a score of 0 or below", output)
                self.assertNotIn("Running MCL (Inflation", output)
                self.assertFalse(hasattr(viewer, "cluster_labels"))
                viewer._save_state.assert_not_called()

    def test_positive_scores_and_unweighted_networks_still_cluster(self):
        for scores in ([1.0, 0.01, 1.0, 0.01], None):
            with self.subTest(scores=scores):
                viewer = network_viewer(4, self.CYCLE, scores)
                _succeeded, failed, _output = run_command(cluster, viewer, ["mcl", "2", "1"])
                failed.assert_not_called()
                self.assertTrue(hasattr(viewer, "cluster_labels"))

    def test_leiden_is_not_refused_for_such_scores(self):
        viewer = network_viewer(4, self.CYCLE, [1.0, -0.3, 0.0, 1.0])
        with mock.patch.dict(
            sys.modules, {"graspologic_native": SimpleNamespace()}
        ), mock.patch.object(
            cluster.network_clustering,
            "leiden_partition",
            return_value=np.array([1, 1, 1, 1]),
        ) as partition:
            _succeeded, failed, _output = run_command(cluster, viewer, ["leiden", "1", "1"])
        failed.assert_not_called()
        partition.assert_called_once()


class OverlappingMCLClusterTests(unittest.TestCase):
    def test_a_cluster_left_below_min_size_by_an_overlap_is_noise(self):
        # MCL gives the path 0-1-2-3-4 the overlapping clusters (0, 1, 2) and
        # (2, 3, 4). The later one takes node 2, leaving (0, 1) below 3.
        viewer = network_viewer(5, PATH_EDGES)
        _succeeded, failed, _output = run_command(cluster, viewer, ["mcl", "2.0", "3"])
        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.cluster_labels, [-1, -1, 1, 1, 1])

    def test_overlapping_clusters_that_stay_large_enough_keep_last_wins(self):
        viewer = network_viewer(5, PATH_EDGES)
        _succeeded, failed, _output = run_command(cluster, viewer, ["mcl", "2.0", "2"])
        failed.assert_not_called()
        # (2, 3, 4) is larger, so it is cluster 1; (0, 1) is cluster 2.
        np.testing.assert_array_equal(viewer.cluster_labels, [2, 2, 1, 1, 1])

    def test_label_mcl_clusters_applies_the_minimum_to_final_sizes(self):
        labels = cluster.label_mcl_clusters([(0, 1, 2), (2, 3, 4)], 6, 3)
        np.testing.assert_array_equal(labels, [-1, -1, 2, 2, 2, -1])
        # A cluster below the minimum on its own never takes nodes from another.
        labels = cluster.label_mcl_clusters([(0, 1, 2), (2, 3)], 4, 3)
        np.testing.assert_array_equal(labels, [1, 1, 1, -1])
        # Without overlap every cluster of enough nodes is kept.
        labels = cluster.label_mcl_clusters([(0, 1), (2,), (3, 4, 5)], 6, 2)
        np.testing.assert_array_equal(labels, [1, 1, -1, 2, 2, 2])


class JaccardWithoutNumbaTests(unittest.TestCase):
    def test_jaccard_does_not_need_numba_to_run(self):
        viewer = network_viewer(7, BARBELL_EDGES)
        with mock.patch.object(cluster.network_clustering, "NUMBA_AVAILABLE", False):
            succeeded, failed, output = run_command(cluster, viewer, ["jaccard", "0.5", "3"])
        failed.assert_not_called()
        self.assertNotIn("Numba", output)
        np.testing.assert_array_equal(viewer.cluster_labels, [1, 1, 1, 2, 2, 2, -1])
        succeeded.assert_called_once_with(viewer, "Done! Found 2 clusters via JACCARD.")


class ClusterListTests(unittest.TestCase):
    def listing(self, stream=None):
        viewer = network_viewer(7, BARBELL_EDGES, cluster_labels=[1, 1, 1, 2, 2, 2, -1])
        stream = io.StringIO() if stream is None else stream
        with redirect_stdout(stream):
            cluster.run(viewer, ["list"])
        return stream.getvalue()

    def test_the_header_and_noise_row_print_once(self):
        output = self.listing()
        self.assertEqual(output.count("Current Cluster Statistics"), 1)
        self.assertEqual(output.count("Noise (Unclustered)"), 1)
        self.assertEqual(output.count("Cluster 1"), 1)
        self.assertEqual(output.count("Cluster 2"), 1)

    def test_colour_codes_are_printed_only_for_a_terminal(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True

        escape = chr(27) + "["
        self.assertNotIn(escape, self.listing())
        self.assertIn("● Noise (Unclustered)", self.listing())
        colored = self.listing(Terminal())
        self.assertIn(escape + "38;2;204;204;204m", colored)
        self.assertIn(escape + "0m", colored)


class ClusterModuleTests(unittest.TestCase):
    def test_pyplot_is_not_imported(self):
        self.assertFalse(hasattr(cluster, "plt"))

    def test_help_states_what_is_clustered(self):
        output = io.StringIO()
        with redirect_stdout(output):
            cluster.print_help()
        text = " ".join(output.getvalue().split())
        for line in (
            "Clustering uses every loaded edge and node, regardless of the "
            "similarity slider and hidden nodes",
            "Jaccard compares closed neighbourhoods",
            "a node counts as its own neighbour",
            "an edge outside every triangle still scores above 0",
            "MCL gives every node a self-loop as heavy as its strongest edge",
            "Isolated nodes (no edge) are alike in all three modes",
            "singleton clusters when MIN_SIZE is 1, Noise when it is larger",
            "Anything after the arguments above (or after list) is refused",
        ):
            self.assertIn(line, text)

    def test_the_windows_ansi_switch_is_made_once_per_process(self):
        flag = "_WINDOWS_ANSI_ENABLED"
        had_flag = flag in vars(cluster)
        try:
            vars(cluster).pop(flag, None)
            with mock.patch.object(sys, "platform", "win32"), mock.patch(
                "os.system"
            ) as system:
                importlib.reload(cluster)
                importlib.reload(cluster)
            system.assert_called_once_with("")
        finally:
            # Leave the module as it was imported.
            if had_flag:
                vars(cluster)[flag] = True
            else:
                vars(cluster).pop(flag, None)

    def test_warning_filters_are_left_as_they_were(self):
        # A first run imports MCL's libraries, which add filters of their own.
        run_command(cluster, network_viewer(7, BARBELL_EDGES), ["mcl", "2", "1"])
        with warnings.catch_warnings():
            from scipy.sparse import SparseEfficiencyWarning

            warnings.simplefilter("always", SparseEfficiencyWarning)
            before = list(warnings.filters)
            viewer = network_viewer(7, BARBELL_EDGES)
            _succeeded, failed, _output = run_command(
                cluster, viewer, ["mcl", "2", "1"]
            )
            failed.assert_not_called()
            self.assertEqual(warnings.filters, before)


class NetworksWithoutEdgesTests(unittest.TestCase):
    def test_an_empty_edge_list_clusters_in_every_mode(self):
        for edges in ([], np.zeros(0, dtype=np.int32), np.zeros((0, 2), dtype=np.int32)):
            for mode, param, labels in (
                ("mcl", "2", [1, 2, 3]),
                ("jaccard", "0.2", [1, 2, 3]),
                ("leiden", "1.0", [1, 2, 3]),
            ):
                with self.subTest(mode=mode, edges=type(edges).__name__):
                    viewer = network_viewer(3, ())
                    viewer.edges = edges
                    with mock.patch.dict(
                        sys.modules, {"graspologic_native": SimpleNamespace()}
                    ):
                        _succeeded, failed, _output = run_command(
                            cluster, viewer, [mode, param, "1"]
                        )
                    failed.assert_not_called()
                    np.testing.assert_array_equal(viewer.cluster_labels, labels)

    def test_without_edges_every_node_is_noise_above_a_min_size_of_one(self):
        for mode, param in (("mcl", "2"), ("jaccard", "0.2"), ("leiden", "1.0")):
            with self.subTest(mode=mode):
                viewer = network_viewer(3, ())
                with mock.patch.dict(
                    sys.modules, {"graspologic_native": SimpleNamespace()}
                ):
                    _succeeded, failed, _output = run_command(
                        cluster, viewer, [mode, param, "2"]
                    )
                failed.assert_not_called()
                np.testing.assert_array_equal(viewer.cluster_labels, [-1, -1, -1])

    def test_an_empty_edge_list_with_scores_clusters_with_mcl(self):
        viewer = network_viewer(3, (), edge_scores=[])
        _succeeded, failed, _output = run_command(cluster, viewer, ["mcl", "2", "1"])
        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.cluster_labels, [1, 2, 3])


class SharedGroupNameWarningTests(unittest.TestCase):
    WARNING = (
        "Warning: custom groups and clusters now share {count} name{plural}: {names}. "
        "Selecting them with #name# is ambiguous until the group is removed."
    )

    def cluster(self, groups):
        viewer = network_viewer(7, BARBELL_EDGES)
        viewer.group_labels = [set(names) for names in groups]
        succeeded, failed, output = run_command(cluster, viewer, ["jaccard", "0.5", "3"])
        failed.assert_not_called()
        return viewer, succeeded, output

    def test_a_group_named_like_a_new_cluster_is_reported_and_kept(self):
        groups = [{"cluster_1", "mine"}, set(), {"cluster_2"}, set(), set(), set(), {"cluster_3"}]
        viewer, succeeded, output = self.cluster(groups)
        warning = self.WARNING.format(count=2, plural="s", names="cluster_1, cluster_2")
        message = f"Done! Found 2 clusters via JACCARD. {warning}"
        succeeded.assert_called_once_with(viewer, message)
        self.assertEqual(viewer.console_text.text, message)
        self.assertIn(warning, output)
        # Groups are neither renamed nor removed; cluster_3 is no cluster.
        self.assertEqual(viewer.group_labels, [set(names) for names in groups])

    def test_one_shared_name_reads_in_the_singular(self):
        viewer, succeeded, _output = self.cluster([{"cluster_2"}] + [set()] * 6)
        warning = self.WARNING.format(count=1, plural="", names="cluster_2")
        succeeded.assert_called_once_with(
            viewer, f"Done! Found 2 clusters via JACCARD. {warning}"
        )

    def test_other_groups_do_not_warn(self):
        viewer, succeeded, _output = self.cluster(
            [{"cluster_3", "cluster_01", "mine"}] + [set()] * 6
        )
        succeeded.assert_called_once_with(viewer, "Done! Found 2 clusters via JACCARD.")


class GeneratedSubclusterGroupTests(unittest.TestCase):
    """cluster replaces the clusters the generated subcluster_N_M groups describe,
    so it removes those groups, after the undo snapshot and only when it runs."""

    REMOVED = "Removed {count} generated subcluster group{plural} of the previous clustering."
    # The barbell's two triangles are the new clusters 1 and 2; node 6 is noise.
    ARGS = ["jaccard", "0.5", "3"]
    OLD_CLUSTERS = [1, 1, 1, 1, 2, 2, -1]
    CUSTOM = {"subcluster_0_2", "subcluster_001_2", "subcluster_1_002", "subcluster_1_0", "alpha"}

    def stale_groups(self):
        """Generated groups of an earlier clustering beside custom groups, per node."""
        return [
            {"subcluster_1_1", "alpha"},
            {"subcluster_1_1", "subcluster_0_2"},
            {"subcluster_1_2", "subcluster_001_2"},
            {"subcluster_12_34"},
            {"subcluster_1_002"},
            {"subcluster_1_0", "subcluster_2_1"},
            set(),
        ]

    def viewer(self, groups):
        viewer = network_viewer(7, BARBELL_EDGES, cluster_labels=self.OLD_CLUSTERS)
        viewer.group_labels = [set(names) for names in groups]
        return viewer

    def test_generated_groups_are_removed_and_the_report_says_so(self):
        viewer = self.viewer(self.stale_groups())

        succeeded, failed, output = run_command(cluster, viewer, self.ARGS)

        failed.assert_not_called()
        self.assertEqual(
            viewer.group_labels,
            [{"alpha"}, {"subcluster_0_2"}, {"subcluster_001_2"}, set(),
             {"subcluster_1_002"}, {"subcluster_1_0"}, set()],
        )
        # subcluster_1_1, subcluster_1_2, subcluster_12_34 and subcluster_2_1:
        # each group counts once, however many nodes carried it.
        removed = self.REMOVED.format(count=4, plural="s")
        message = f"Done! Found 2 clusters via JACCARD. {removed}"
        succeeded.assert_called_once_with(viewer, message)
        self.assertEqual(viewer.console_text.text, message)
        self.assertIn(message, output)
        viewer.update_nodes.assert_called_once_with()

    def test_one_removed_group_reads_in_the_singular(self):
        viewer = self.viewer([{"subcluster_3_1"}] + [set()] * 6)

        succeeded, _failed, _output = run_command(cluster, viewer, self.ARGS)

        removed = self.REMOVED.format(count=1, plural="")
        succeeded.assert_called_once_with(viewer, f"Done! Found 2 clusters via JACCARD. {removed}")

    def test_the_report_is_unchanged_when_there_is_nothing_to_remove(self):
        # Custom groups with lookalike names are kept and are not reported.
        for groups in ([set()] * 7, [set(self.CUSTOM)] + [set()] * 6):
            with self.subTest(groups=groups):
                viewer = self.viewer(groups)
                succeeded, _failed, output = run_command(cluster, viewer, self.ARGS)
                succeeded.assert_called_once_with(viewer, "Done! Found 2 clusters via JACCARD.")
                self.assertNotIn("Removed", output)
                self.assertEqual(viewer.group_labels, [set(names) for names in groups])

    def test_lookalike_and_custom_groups_are_kept(self):
        viewer = self.viewer([self.CUSTOM | {"subcluster_5_5", "cluster_9"}] + [set()] * 6)

        run_command(cluster, viewer, self.ARGS)

        self.assertEqual(viewer.group_labels[0], self.CUSTOM | {"cluster_9"})

    def test_the_removal_is_reported_before_the_shared_name_warning(self):
        viewer = self.viewer([{"subcluster_1_1", "cluster_2"}] + [set()] * 6)

        succeeded, _failed, _output = run_command(cluster, viewer, self.ARGS)

        warning = SharedGroupNameWarningTests.WARNING.format(count=1, plural="", names="cluster_2")
        removed = self.REMOVED.format(count=1, plural="")
        succeeded.assert_called_once_with(
            viewer, f"Done! Found 2 clusters via JACCARD. {removed} {warning}"
        )
        self.assertEqual(viewer.group_labels[0], {"cluster_2"})

    def test_the_groups_go_after_the_undo_snapshot(self):
        viewer = self.viewer(self.stale_groups())
        snapshots = []
        viewer._save_state = mock.Mock(
            side_effect=lambda: snapshots.append([set(groups) for groups in viewer.group_labels])
        )

        run_command(cluster, viewer, self.ARGS)

        viewer._save_state.assert_called_once_with()
        self.assertEqual(snapshots, [self.stale_groups()])
        self.assertNotEqual(viewer.group_labels, snapshots[0])

    def test_undo_brings_the_groups_back(self):
        from EMAPSSN_Viewer import MainViewer

        n_nodes = 7
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = n_nodes
        viewer.full_headers = [f"n{node}" for node in range(n_nodes)]
        viewer.edges = np.array(BARBELL_EDGES, dtype=np.int32)
        viewer.pos = np.zeros((n_nodes, 2), dtype=np.float32)
        viewer.visible_mask = np.ones(n_nodes, dtype=bool)
        viewer.current_colors = np.zeros((n_nodes, 4))
        viewer.current_sizes = np.full(n_nodes, 10.0, dtype=np.float32)
        viewer.current_shapes = np.full(n_nodes, "disc", dtype=object)
        viewer.node_render_order = np.arange(n_nodes, dtype=np.int32)
        viewer.cluster_labels = np.array(self.OLD_CLUSTERS)
        viewer.last_cluster_params = ("LEIDEN_1.0", 10)
        viewer.group_labels = [set(groups) for groups in self.stale_groups()]
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

        _succeeded, failed, _output = run_command(cluster, viewer, self.ARGS)

        failed.assert_not_called()
        np.testing.assert_array_equal(viewer.cluster_labels, [1, 1, 1, 2, 2, 2, -1])
        self.assertNotIn("subcluster_1_1", set().union(*viewer.group_labels))
        self.assertEqual(len(viewer.position_history), 1)

        with redirect_stdout(io.StringIO()):
            self.assertTrue(viewer._do_undo())

        self.assertEqual(viewer.group_labels, self.stale_groups())
        np.testing.assert_array_equal(viewer.cluster_labels, self.OLD_CLUSTERS)
        self.assertEqual(viewer.last_cluster_params, ("LEIDEN_1.0", 10))

    def test_a_refused_or_failed_run_leaves_the_groups_alone(self):
        cases = [
            (["mcl", "1.0", "1"], {}),        # inflation out of range
            (["jaccard", "1.5", "1"], {}),    # threshold out of range
            (["leiden", "0", "1"], {}),       # resolution not above 0
            (["jaccard", "0.2", "0"], {}),    # MIN_SIZE below 1
            (["jaccard", "0.2", "big"], {}),  # MIN_SIZE not an integer
            (["jaccard", "many"], {}),        # parameter not a number
            (["hierarchical"], {}),           # unknown mode
            (["leiden"], {"graspologic_native": None}),  # library missing
        ]
        for args, modules in cases:
            with self.subTest(args=args):
                viewer = self.viewer(self.stale_groups())
                with mock.patch.dict(sys.modules, modules):
                    succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_called_once()
                succeeded.assert_not_called()
                self.assertEqual(viewer.group_labels, self.stale_groups())
                np.testing.assert_array_equal(viewer.cluster_labels, self.OLD_CLUSTERS)
                viewer._save_state.assert_not_called()

    def test_an_mcl_run_that_cannot_use_the_edge_scores_leaves_the_groups_alone(self):
        viewer = self.viewer(self.stale_groups())
        viewer.edge_scores = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        succeeded, failed, _output = run_command(cluster, viewer, ["mcl", "2", "1"])

        failed.assert_called_once()
        succeeded.assert_not_called()
        self.assertEqual(viewer.group_labels, self.stale_groups())
        viewer._save_state.assert_not_called()

    def test_list_and_help_leave_the_groups_alone(self):
        for args in (["list"], ["help"]):
            with self.subTest(args=args):
                viewer = self.viewer(self.stale_groups())
                _succeeded, failed, _output = run_command(cluster, viewer, args)
                failed.assert_not_called()
                self.assertEqual(viewer.group_labels, self.stale_groups())
                viewer._save_state.assert_not_called()

    def test_a_viewer_without_groups_clusters_as_before(self):
        for group_labels in (None, "absent"):
            with self.subTest(group_labels=group_labels):
                viewer = network_viewer(7, BARBELL_EDGES)
                if group_labels is None:
                    viewer.group_labels = None
                succeeded, failed, _output = run_command(cluster, viewer, self.ARGS)
                failed.assert_not_called()
                succeeded.assert_called_once_with(viewer, "Done! Found 2 clusters via JACCARD.")

    def test_the_help_and_catalog_say_cluster_removes_the_groups(self):
        printed = io.StringIO()
        with redirect_stdout(printed):
            cluster.print_help()
        self.assertIn(
            "removes the subcluster groups (subcluster_N_M)", " ".join(printed.getvalue().split())
        )

        from desktop.Command_Metadata import COMMAND_METADATA

        action = COMMAND_METADATA["cluster"]["arguments"][0]["description"]
        self.assertIn("removes the generated subcluster_N_M groups", action)


if __name__ == "__main__":
    unittest.main()
