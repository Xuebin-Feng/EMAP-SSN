"""utilities.Threshold_Tree: single linkage over a network's edges, strongest first.

The reference is scipy's single-linkage dendrogram of a complete network: with the distance
``top - score``, the groups it gives at every cut height are the groups the edges at or above
the matching score connect.
"""

import pathlib
import sys
import time
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The kernels are compiled (or loaded from the cache) on first use; importing the module here keeps that
# out of any sys.modules window another test opens.
from utilities.Threshold_Tree import groups_at, threshold_tree


def sorted_edges(pairs, scores):
    """The edges and scores in descending score order, as EdgeOrder holds them."""
    pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    return pairs[order], scores[order]


def components(count, pairs, scores, threshold):
    """The connected components of the edges scoring at least ``threshold``, as a label for each node (union-find, plainly)."""
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for (a, b), score in zip(pairs, scores):
        if score >= threshold:
            parent[find(int(a))] = find(int(b))
    return np.array([find(node) for node in range(count)])


def same_partition(first, second):
    """Whether two label arrays split the items into the same groups."""
    first, second = np.asarray(first), np.asarray(second)
    if len(first) != len(second):
        return False
    mapping, reverse = {}, {}
    for a, b in zip(first.tolist(), second.tolist()):
        if mapping.setdefault(a, b) != b or reverse.setdefault(b, a) != a:
            return False
    return True


def tree_partition(tree, count, threshold):
    """The groups of the network's nodes at ``threshold``, as a label for each node, read off the tree."""
    labels = groups_at(tree, threshold)
    by_node = np.empty(count, dtype=np.int64)
    by_node[tree.leaf_order] = labels
    return by_node


def check_post_order(test, tree):
    """The arrays are a valid post-order tree: children before parents, the root last, each node's leaves one block."""
    n = tree.node_count
    test.assertEqual(int(tree.parent[n - 1]), -1)
    test.assertTrue(np.all(tree.parent[:-1] > np.arange(n - 1)))
    test.assertTrue(np.all(tree.parent[:-1] < n))
    children = np.bincount(tree.parent[tree.parent >= 0], minlength=n)
    leaves = np.flatnonzero(children == 0)
    test.assertEqual(len(leaves), tree.leaf_count)
    test.assertEqual(sorted(tree.leaf_order.tolist()), list(range(tree.leaf_count)))
    # Every internal node has at least two children (a tie merges them; nothing is left with one).
    test.assertTrue(np.all(children[children > 0] >= 2) or n == 1)
    # The leaves under a node are one run of leaf numbers.
    leaf_number = np.full(n, -1)
    leaf_number[leaves] = np.arange(len(leaves))
    first = np.where(leaf_number >= 0, leaf_number, n + 1)
    last = np.where(leaf_number >= 0, leaf_number, -1)
    size = np.where(leaf_number >= 0, 1, 0)
    for node in range(n - 1):
        parent = tree.parent[node]
        first[parent] = min(first[parent], first[node])
        last[parent] = max(last[parent], last[node])
        size[parent] += size[node]
    test.assertTrue(np.array_equal(last - first + 1, size))
    # Heights never rise towards the root, and a branch is as long as the heights it spans.
    parents = tree.parent[:-1]
    test.assertTrue(np.all(tree.height[:-1] >= tree.height[parents]))
    test.assertTrue(np.allclose(tree.length[:-1], tree.height[:-1] - tree.height[parents]))
    test.assertEqual(float(tree.length[-1]), 0.0)


class AgainstScipy(unittest.TestCase):
    def test_complete_networks_give_the_groups_scipy_gives_at_every_cut(self):
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        for seed, count in ((1, 6), (2, 15), (3, 40), (4, 120)):
            rng = np.random.default_rng(seed)
            # Distinct scores: whole numbers, so that heights compare exactly.
            scores = rng.permutation(count * (count - 1) // 2).astype(np.float32) + 1
            pairs = [(a, b) for a in range(count) for b in range(a + 1, count)]
            matrix = np.zeros((count, count))
            top = float(scores.max())
            for (a, b), score in zip(pairs, scores):
                matrix[a, b] = matrix[b, a] = top - score
            reference = linkage(squareform(matrix), method="single")
            edges, ordered = sorted_edges(pairs, scores)
            tree = threshold_tree(edges, ordered, count)
            check_post_order(self, tree)
            self.assertEqual(tree.groups, 1)
            self.assertEqual(tree.joins, count - 1)
            with self.subTest(count=count):
                # Every distinct join height of the reference, and one either side of it.
                for height in sorted(set(reference[:, 2].tolist())):
                    for distance in (height - 0.5, height, height + 0.5):
                        expected = fcluster(reference, distance, criterion="distance")
                        got = tree_partition(tree, count, top - distance)
                        self.assertTrue(same_partition(expected, got), (count, distance))
                # The root is the lowest join, the leaves stand at the top score.
                self.assertEqual(tree.top, top)
                self.assertAlmostEqual(tree.bottom, top - float(reference[:, 2].max()), places=4)

    def test_sparse_networks_give_the_components_at_every_threshold(self):
        for seed in range(6):
            rng = np.random.default_rng(10 + seed)
            count = int(rng.integers(5, 60))
            pair_count = int(rng.integers(1, 3 * count))
            pairs = rng.integers(0, count, (pair_count, 2))
            scores = rng.integers(1, 8, pair_count).astype(np.float32)  # many ties
            edges, ordered = sorted_edges(pairs, scores)
            tree = threshold_tree(edges, ordered, count)
            check_post_order(self, tree)
            for threshold in (0.5, 1, 2, 3, 4, 5, 6, 7, 7.5):
                with self.subTest(seed=seed, threshold=threshold):
                    expected = components(count, edges, ordered, threshold)
                    self.assertTrue(same_partition(expected, tree_partition(tree, count, threshold)))


class Shapes(unittest.TestCase):
    def test_ties_make_one_branch_point_with_several_children(self):
        # 0-1, 0-2, 0-3 all at 5, then 3-4 at 2.
        edges, scores = sorted_edges([(0, 1), (0, 2), (0, 3), (3, 4)], [5, 5, 5, 2])
        tree = threshold_tree(edges, scores, 5)
        check_post_order(self, tree)
        self.assertEqual(tree.node_count, 5 + 2)
        children = np.bincount(tree.parent[tree.parent >= 0], minlength=tree.node_count)
        self.assertEqual(sorted(children[children > 0].tolist()), [2, 4])
        top_clade = int(np.flatnonzero(children == 4)[0])
        self.assertEqual(float(tree.height[top_clade]), 5.0)
        self.assertEqual(float(tree.height[-1]), 2.0)

    def test_a_tie_that_joins_two_larger_groups_is_still_one_node(self):
        # (0,1) and (2,3) at 9, then (1,2) and (3,4) and (0,5) at 4.
        edges, scores = sorted_edges([(0, 1), (2, 3), (1, 2), (3, 4), (0, 5)], [9, 9, 4, 4, 4])
        tree = threshold_tree(edges, scores, 6)
        check_post_order(self, tree)
        children = np.bincount(tree.parent[tree.parent >= 0], minlength=tree.node_count)
        self.assertEqual(int(children[-1]), 4)  # the two pairs and the two singletons
        self.assertEqual(tree.node_count, 6 + 3)

    def test_leaf_lengths_run_from_the_top_score_to_the_join(self):
        edges, scores = sorted_edges([(0, 1), (1, 2)], [8, 3])
        tree = threshold_tree(edges, scores, 3)
        leaf = np.flatnonzero(np.bincount(tree.parent[tree.parent >= 0], minlength=tree.node_count) == 0)
        for node in leaf:
            self.assertEqual(float(tree.height[node]), 8.0)
        self.assertEqual(tree.groups, 1)
        self.assertEqual(tree.root_score, 3.0)
        # The leaf that joins last is the longest branch: from 8 down to 3.
        lengths = sorted(float(tree.length[node]) for node in leaf)
        self.assertEqual(lengths, [0.0, 0.0, 5.0])

    def test_a_forest_hangs_its_groups_from_a_root_below_the_lowest_score(self):
        # Groups {0,1,2} (scores 9 and 6), {3,4} (score 4), and 5 alone.
        edges, scores = sorted_edges([(0, 1), (1, 2), (3, 4)], [9, 6, 4])
        tree = threshold_tree(edges, scores, 6)
        check_post_order(self, tree)
        self.assertEqual(tree.groups, 3)
        self.assertLess(tree.root_score, 4.0)
        self.assertAlmostEqual(tree.root_score, 4.0 - 0.05 * 5.0, places=5)
        children = np.bincount(tree.parent[tree.parent >= 0], minlength=tree.node_count)
        self.assertEqual(int(children[-1]), 3)
        # The cut at any score never joins the groups that share no edge.
        for threshold in (0.0, 4.0, 6.0):
            labels = tree_partition(tree, 6, threshold)
            self.assertNotEqual(labels[0], labels[3])
            self.assertNotEqual(labels[3], labels[5])
        self.assertEqual(len(set(tree_partition(tree, 6, 4.0).tolist())), 3)

    def test_nan_scores_join_nothing(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        scores = np.array([5.0, np.nan, 2.0], dtype=np.float32)
        tree = threshold_tree(edges, scores, 4)
        check_post_order(self, tree)
        self.assertEqual(tree.joins, 2)
        self.assertEqual(tree.groups, 2)
        self.assertTrue(np.all(np.isfinite(tree.height)) and np.all(np.isfinite(tree.length)))

    def test_all_nan_scores_make_a_flat_forest(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        tree = threshold_tree(edges, np.array([np.nan, np.nan], dtype=np.float32), 3)
        check_post_order(self, tree)
        self.assertEqual(tree.groups, 3)
        self.assertEqual(tree.node_count, 4)
        self.assertTrue(np.all(np.isfinite(tree.length)))

    def test_one_node_is_a_tree_of_one_leaf(self):
        tree = threshold_tree(np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32), 1)
        self.assertEqual((tree.node_count, tree.leaf_count, tree.groups), (1, 1, 1))
        self.assertEqual(int(tree.parent[0]), -1)
        self.assertEqual(tree.leaf_order.tolist(), [0])

    def test_no_edges_leave_every_node_a_group(self):
        tree = threshold_tree(np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32), 4)
        check_post_order(self, tree)
        self.assertEqual(tree.groups, 4)
        self.assertEqual(tree.node_count, 5)
        self.assertTrue(np.all(np.isfinite(tree.length)))

    def test_disconnected_nodes_hang_from_the_forest_root(self):
        edges, scores = sorted_edges([(0, 1)], [3])
        tree = threshold_tree(edges, scores, 4)
        check_post_order(self, tree)
        self.assertEqual(tree.groups, 3)
        self.assertEqual(int(np.bincount(tree.parent[tree.parent >= 0], minlength=tree.node_count)[-1]), 3)

    def test_self_edges_and_repeated_edges_join_nothing(self):
        edges, scores = sorted_edges([(0, 0), (0, 1), (1, 0), (1, 1)], [9, 8, 7, 6])
        tree = threshold_tree(edges, scores, 2)
        check_post_order(self, tree)
        self.assertEqual((tree.joins, tree.groups), (1, 1))

    def test_no_scores_means_every_edge_scores_the_same(self):
        tree = threshold_tree(np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32), None, 5)
        check_post_order(self, tree)
        self.assertEqual(tree.groups, 2)

    def test_edges_outside_the_network_are_refused(self):
        with self.assertRaises(ValueError):
            threshold_tree(np.array([[0, 5]], dtype=np.int32), np.array([1.0], dtype=np.float32), 3)
        with self.assertRaises(ValueError):
            threshold_tree(np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32), 0)

    def test_a_chain_as_deep_as_it_is_long_is_no_trouble(self):
        count = 30000
        pairs = [(node, node + 1) for node in range(count - 1)]
        edges, scores = sorted_edges(pairs, np.arange(count - 1, 0, -1))
        tree = threshold_tree(edges, scores, count)
        check_post_order(self, tree)
        self.assertEqual(tree.node_count, 2 * count - 1)

    def test_float64_scores_are_taken_too(self):
        edges, scores = sorted_edges([(0, 1), (1, 2)], [2, 1])
        tree = threshold_tree(edges, scores.astype(np.float64), 3)
        self.assertEqual(tree.groups, 1)


class Speed(unittest.TestCase):
    def test_44000_nodes_and_many_edges_where_the_network_connects_last(self):
        count, edge_count = 44000, 8_000_000
        rng = np.random.default_rng(42)
        edges = rng.integers(0, count - 1, (edge_count, 2)).astype(np.int32)
        edges[-1] = (0, count - 1)  # the last edge is what connects the last node
        scores = np.linspace(1.0, 0.0, edge_count, dtype=np.float32)
        threshold_tree(edges[:100], scores[:100], count, check=False)  # compiled, loaded
        started = time.perf_counter()
        tree = threshold_tree(edges, scores, count, check=False)
        elapsed = time.perf_counter() - started
        self.assertEqual((tree.groups, tree.joins), (1, count - 1))
        self.assertEqual(tree.node_count, 2 * count - 1)
        # 8M edges take about 25 ms here; the 48.7M edges of a very dense network about 0.14 s.
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
