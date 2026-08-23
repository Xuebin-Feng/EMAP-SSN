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

from commands import cluster


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


if __name__ == "__main__":
    unittest.main()
