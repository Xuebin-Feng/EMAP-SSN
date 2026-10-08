"""The cluster command (commands/cluster.py): Leiden, Jaccard and MCL
partitions, their parameters and defaults, and the size order clusters are
numbered in."""
import os
import sys
import unittest
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

    def test_fast_jaccard_filter_uses_neighbour_union(self):
        edges = np.asarray([[0, 1], [0, 3]], dtype=np.int32)
        indptr = np.asarray([0, 2, 4, 7, 8], dtype=np.int32)
        indices = np.asarray([1, 2, 0, 2, 0, 1, 3, 2], dtype=np.int32)

        result = cluster.network_clustering.fast_jaccard_filter(
            edges, indptr, indices, 0.4
        )

        np.testing.assert_array_equal(result, [False, True])

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
        cases = [
            # Every triangle edge (1/3 and 1/4) is kept and the bridge (0) cut.
            ("0.2", "3", [1, 1, 1, 2, 2, 2, -1]),
            ("0.25", "3", [1, 1, 1, 2, 2, 2, -1]),
            # Only 0-1 and 4-5 (1/3) remain above 0.26.
            ("0.26", "2", [1, 1, -1, -1, 2, 2, -1]),
            # A threshold of 0 keeps the bridge too.
            ("0", "3", [1, 1, 1, 1, 1, 1, -1]),
            # No edge reaches 1, so every node, the isolated one included, is
            # a singleton that a MIN_SIZE of 1 keeps.
            ("1", "1", [1, 2, 3, 4, 5, 6, 7]),
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
                succeeded.assert_called_once_with(
                    viewer, f"Done! Found {clusters} clusters via JACCARD."
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
        for args in (["jaccard", "0.2", "3"], ["mcl", "2", "3"]):
            with self.subTest(args=args):
                viewer, _succeeded = self.cluster(args, edges=reversed_edges)
                np.testing.assert_array_equal(
                    viewer.cluster_labels, [1, 1, 1, 2, 2, 2, -1]
                )

    def test_larger_clusters_get_lower_numbers(self):
        # A triangle on nodes 0-2 and a K4 on 3-6: the K4 is found second but
        # is the larger cluster. Its Jaccard indices are 1/2, the triangle's 1/3.
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
        viewer, _succeeded = self.cluster(["jaccard", "0.2", "3"])
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


if __name__ == "__main__":
    unittest.main()
