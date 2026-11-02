"""Embedding_MSA saves its guide tree as Newick beside the alignment.

UPGMA trees keep their root and halve merge heights into branch lengths;
neighbor-joining trees carry their Saitou-Nei branch lengths and are rooted
at the midpoint of their longest path. Noise-perturbed replicates label each
split with the percentage of replicates that hold it, and neighbor-joining
replicates enter the consensus matrix as path lengths over their branches.
"""

import gc
import io
import pathlib
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
UTILITIES_DIR = PROJECT_ROOT / "src" / "utilities"
TOOLS_DIR = PROJECT_ROOT / "src" / "tools"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import Embedding_MSA
from tests.msa_fixtures import (
    MODEL,
    UPGMA,
    GuideTreeBuilt,
    InProcessPool,
    NetworkFixture,
    builder_settings,
)


NEIGHBOR_JOINING = "Neighbor-joining (Slow)"


def parse_newick(text):
    """Return (name, parent, length, label) for each node; the root is first."""
    nodes = [["", -1, 0.0, ""]]
    stack = []
    current = 0
    position = 0
    while text[position] != ";":
        char = text[position]
        if char == "(":
            stack.append(current)
            nodes.append(["", current, 0.0, ""])
            current = len(nodes) - 1
            position += 1
        elif char == ",":
            nodes.append(["", stack[-1], 0.0, ""])
            current = len(nodes) - 1
            position += 1
        elif char == ")":
            current = stack.pop()
            position += 1
        elif char == ":":
            value = re.match(r"[^,();]+", text[position + 1:]).group()
            nodes[current][2] = float(value)
            position += 1 + len(value)
        elif char == "'":
            end = position + 1
            name = []
            while text[end] != "'" or text[end + 1] == "'":
                name.append(text[end])
                end += 2 if text[end] == "'" else 1
            nodes[current][0] = "".join(name)
            position = end + 1
        else:
            value = re.match(r"[^,():;']+", text[position:]).group()
            nodes[current][0] = value
            position += len(value)
    has_children = {parent for _, parent, _, _ in nodes}
    for index, node in enumerate(nodes):
        # A label after ")" is an internal node's support, not a name.
        if index in has_children and node[0]:
            node[3], node[0] = node[0], ""
    return nodes


def leaf_path_lengths(nodes, names):
    """Return the matrix of path lengths between the named leaves."""
    depth = {}
    for index, (_, parent, length, _) in enumerate(nodes):
        depth[index] = (depth[parent] if parent >= 0 else 0.0) + (length if parent >= 0 else 0.0)
    leaf_of = {node[0]: index for index, node in enumerate(nodes) if node[0]}

    def ancestors(index):
        chain = []
        while index >= 0:
            chain.append(index)
            index = nodes[index][1]
        return chain

    chains = {name: ancestors(leaf_of[name]) for name in names}
    paths = np.zeros((len(names), len(names)))
    for i, first in enumerate(names):
        above = set(chains[first])
        for j, second in enumerate(names):
            meet = next(node for node in chains[second] if node in above)
            paths[i, j] = depth[leaf_of[first]] + depth[leaf_of[second]] - 2 * depth[meet]
    return paths


def tree_path_lengths(parent, length, num_seqs):
    """Return leaf path lengths from parent and branch-length arrays."""
    depth = np.zeros(len(parent))
    order = []
    for node in range(len(parent)):
        chain = []
        while node >= 0 and node not in order:
            chain.append(node)
            node = parent[node]
        order.extend(reversed(chain))
    for node in order:
        if parent[node] >= 0:
            depth[node] = depth[parent[node]] + length[node]
    paths = np.zeros((num_seqs, num_seqs))
    for i in range(num_seqs):
        above = set()
        node = i
        while node >= 0:
            above.add(node)
            node = parent[node]
        for j in range(num_seqs):
            meet = j
            while meet not in above:
                meet = parent[meet]
            paths[i, j] = depth[i] + depth[j] - 2 * depth[meet]
    return paths


def random_additive_distances(count, seed):
    """Return the leaf path lengths of a random binary tree."""
    rng = np.random.default_rng(seed)
    parent = np.full(2 * count - 1, -1)
    length = np.zeros(2 * count - 1)
    pool = list(range(count))
    for new in range(count, 2 * count - 1):
        for _ in range(2):
            child = pool.pop(int(rng.integers(len(pool))))
            parent[child] = new
            length[child] = rng.uniform(0.05, 1.0)
        pool.append(new)
    return tree_path_lengths(parent, length, count)


def random_nj_replicates(count, trees, rng):
    """Return neighbor-joining trees and raw branch lengths of random distances.

    Random distances fit no tree, so some raw lengths come out negative.
    """
    return [
        Embedding_MSA.neighbor_joining_condensed(
            rng.random(count * (count - 1) // 2), count, return_branch_lengths=True
        )
        for _ in range(trees)
    ]


def summed_path_lengths(replicates, count):
    """Return the brute-force sum of each replicate's clamped path lengths."""
    total = np.zeros((count, count))
    for tree, lengths in replicates:
        parent, length = Embedding_MSA.guide_tree_branches(tree, count, lengths)
        total += tree_path_lengths(parent, length, count)
    return total


def condensed_indices(count, first, second):
    """Return the condensed-matrix index of each pair, in either order."""
    i = np.minimum(first, second).astype(np.int64)
    j = np.maximum(first, second).astype(np.int64)
    return count * i - i * (i + 1) // 2 + j - i - 1


class PathLengthConsensusTests(unittest.TestCase):
    @staticmethod
    def every_ordered_pair(count):
        # Half of the pairs list the larger index first.
        edge_i, edge_j = np.nonzero(~np.eye(count, dtype=bool))
        return edge_i.astype(np.int32), edge_j.astype(np.int32)

    @staticmethod
    def add_full(replicates, count):
        accumulator = np.zeros(count * (count - 1) // 2, dtype=np.float32)
        for tree, lengths in replicates:
            Embedding_MSA.add_replicate_to_consensus(
                tree, lengths, count, True, accumulator, None, None, None
            )
        return accumulator

    @staticmethod
    def add_sparse(replicates, count, edge_i, edge_j):
        accumulator = np.zeros(edge_i.size, dtype=np.float32)
        for tree, lengths in replicates:
            Embedding_MSA.add_replicate_to_consensus(
                tree, lengths, count, False, None, edge_i, edge_j, accumulator
            )
        return accumulator

    def test_full_sums_match_brute_force_paths(self):
        rng = np.random.default_rng(42)
        for count in (2, 3, 17, 40):
            with self.subTest(count=count):
                replicates = random_nj_replicates(count, 4, rng)
                np.testing.assert_allclose(
                    self.add_full(replicates, count),
                    squareform(summed_path_lengths(replicates, count), checks=False),
                    rtol=1e-6,
                    atol=1e-6,
                )
        # The clamped branches were exercised.
        self.assertLess(min(lengths.min() for _, lengths in replicates), 0.0)

    def test_sparse_sums_match_brute_force_paths(self):
        rng = np.random.default_rng(42)
        for count in (2, 3, 17, 40):
            with self.subTest(count=count):
                replicates = random_nj_replicates(count, 4, rng)
                edge_i, edge_j = self.every_ordered_pair(count)
                np.testing.assert_allclose(
                    self.add_sparse(replicates, count, edge_i, edge_j),
                    summed_path_lengths(replicates, count)[edge_i, edge_j],
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_negative_branches_are_clamped_before_summing(self):
        # Clamped, the branches above leaves 0-3 are 0, 0.8, 0.4 and 0, and
        # both inner branches 0. Unclamped, the path from 0 to 2 would be 0.1.
        tree = np.array([[0, 1, 1.0, 2], [2, 3, 1.0, 2], [4, 5, 1.0, 4]], dtype=float)
        lengths = np.array([[-0.2, 1.0], [0.5, -0.1], [-0.3, 0.1]])
        expected = np.array([0.8, 0.4, 0.0, 1.2, 0.8, 0.4], dtype=np.float32)
        edge_i, edge_j = np.triu_indices(4, k=1)
        np.testing.assert_allclose(self.add_full([(tree, lengths)], 4), expected, atol=1e-7)
        np.testing.assert_allclose(
            self.add_sparse([(tree, lengths)], 4, edge_i.astype(np.int32), edge_j.astype(np.int32)),
            expected,
            atol=1e-7,
        )

    def test_upgma_replicates_still_add_merge_heights(self):
        count = 30
        tree = sch.linkage(np.random.default_rng(42).random(count * (count - 1) // 2), method="average")
        full = np.zeros(count * (count - 1) // 2, dtype=np.float32)
        Embedding_MSA.add_replicate_to_consensus(tree, None, count, True, full, None, None, None)
        expected_full = np.zeros_like(full)
        Embedding_MSA.accumulate_full_cophenetic(tree, count, expected_full)
        self.assertEqual(full.tobytes(), expected_full.tobytes())

        edge_i, edge_j = self.every_ordered_pair(count)
        sparse = np.zeros(edge_i.size, dtype=np.float32)
        Embedding_MSA.add_replicate_to_consensus(
            tree, None, count, False, None, edge_i, edge_j, sparse
        )
        self.assertEqual(
            sparse.tobytes(),
            Embedding_MSA.compute_sparse_cophenetic(tree, count, edge_i, edge_j).tobytes(),
        )


class NeighborJoiningBranchLengthTests(unittest.TestCase):
    def test_additive_distances_come_back_as_path_lengths(self):
        distances = random_additive_distances(30, seed=42)
        tree, lengths = Embedding_MSA.neighbor_joining_condensed(
            squareform(distances), 30, return_branch_lengths=True
        )
        parent, length = Embedding_MSA.guide_tree_branches(tree, 30, lengths)
        np.testing.assert_allclose(tree_path_lengths(parent, length, 30), distances, atol=1e-9)
        # NJ recovers every branch of a tree metric, so none needed clamping.
        self.assertGreater(lengths.min(), 0.0)

    def test_the_linkage_matrix_does_not_change(self):
        condensed = np.random.default_rng(42).random(45).astype(np.float32)
        tree, _ = Embedding_MSA.neighbor_joining_condensed(
            condensed, 10, return_branch_lengths=True
        )
        self.assertEqual(
            tree.tobytes(), Embedding_MSA.neighbor_joining_condensed(condensed, 10).tobytes()
        )

    def test_the_last_join_splits_its_distance_evenly(self):
        tree, lengths = Embedding_MSA.neighbor_joining_condensed(
            np.array([3.0]), 2, return_branch_lengths=True
        )
        np.testing.assert_array_equal(lengths, [[1.5, 1.5]])


class GuideTreeBranchTests(unittest.TestCase):
    def test_upgma_paths_equal_merge_heights(self):
        condensed = np.random.default_rng(42).random(66)
        tree = sch.linkage(condensed, method="average")
        parent, length = Embedding_MSA.guide_tree_branches(tree, 12)
        np.testing.assert_allclose(
            squareform(tree_path_lengths(parent, length, 12), checks=False),
            sch.cophenet(tree),
        )

    def test_a_negative_branch_hands_its_length_to_its_sibling(self):
        tree = np.array([[0, 1, 1.0, 2], [2, 3, 1.0, 3], [4, 5, 1.0, 5]], dtype=float)
        lengths = np.array([[-0.2, 1.0], [0.5, -0.1], [-0.3, 0.1]])
        _, length = Embedding_MSA.guide_tree_branches(tree, 4, lengths)
        np.testing.assert_allclose(length[[0, 1, 2, 3, 4, 5]], [0.0, 0.8, 0.4, 0.0, 0.0, 0.0])


class MidpointRootTests(unittest.TestCase):
    def rerooted_paths(self, children, branch, root, num_seqs):
        parent = np.full(len(children), -1)
        for node, kids in enumerate(children):
            for kid in kids:
                parent[kid] = node
        self.assertEqual(parent[root], -1)
        return parent, tree_path_lengths(parent, branch, num_seqs)

    def test_the_root_halves_the_longest_path_and_keeps_every_distance(self):
        distances = random_additive_distances(25, seed=7)
        tree, lengths = Embedding_MSA.neighbor_joining_condensed(
            squareform(distances), 25, return_branch_lengths=True
        )
        parent, length = Embedding_MSA.guide_tree_branches(tree, 25, lengths)
        children, branch, root = Embedding_MSA.midpoint_rooted_children(parent, length, 25)
        new_parent, paths = self.rerooted_paths(children, branch, root, 25)

        np.testing.assert_allclose(paths, distances, atol=1e-9)
        # Every leaf hangs from the root once, and no node is left with one child.
        reached = [node for node in range(len(children)) if node == root or new_parent[node] >= 0]
        self.assertEqual(sorted(node for node in reached if not children[node]), list(range(25)))
        self.assertTrue(all(len(children[node]) != 1 for node in reached))
        # The root sits halfway along the longest path.
        depth = np.zeros(len(children))
        stack = [root]
        while stack:
            node = stack.pop()
            for kid in children[node]:
                depth[kid] = depth[node] + branch[kid]
                stack.append(kid)
        self.assertAlmostEqual(depth[:25].max(), distances.max() / 2)
        sides = [
            max(depth[leaf] for leaf in range(25) if self.below(new_parent, leaf, kid))
            for kid in children[root]
        ]
        self.assertAlmostEqual(sides[0], sides[1])

    @staticmethod
    def below(parent, node, ancestor):
        while node >= 0:
            if node == ancestor:
                return True
            node = parent[node]
        return False

    def test_two_leaves_hang_from_a_root_between_them(self):
        tree, lengths = Embedding_MSA.neighbor_joining_condensed(
            np.array([4.0]), 2, return_branch_lengths=True
        )
        parent, length = Embedding_MSA.guide_tree_branches(tree, 2, lengths)
        children, branch, root = Embedding_MSA.midpoint_rooted_children(parent, length, 2)
        self.assertEqual(sorted(children[root]), [0, 1])
        np.testing.assert_allclose(branch[[0, 1]], [2.0, 2.0])

    def test_a_tree_of_zero_lengths_keeps_its_root(self):
        tree = np.array([[0, 1, 0.0, 2], [2, 3, 0.0, 3]], dtype=float)
        parent, length = Embedding_MSA.guide_tree_branches(tree, 3, np.zeros((2, 2)))
        children, branch, root = Embedding_MSA.midpoint_rooted_children(parent, length, 3)
        self.assertEqual(root, 4)
        self.assertEqual(children[4], [2, 3])


class SplitSupportTests(unittest.TestCase):
    def support(self, final, replicates, num_seqs, nj=False):
        keys = Embedding_MSA.guide_tree_leaf_keys(num_seqs, 42)
        replicate_keys = np.sort(np.concatenate([
            Embedding_MSA.replicate_split_keys(tree, num_seqs, keys) for tree in replicates
        ]))
        if nj:
            parent, length = Embedding_MSA.guide_tree_branches(
                final, num_seqs, np.ones((num_seqs - 1, 2))
            )
            children, _, root = Embedding_MSA.midpoint_rooted_children(parent, length, num_seqs)
        else:
            children, root = Embedding_MSA.rooted_children(
                Embedding_MSA.guide_tree_branches(final, num_seqs)[0]
            )
        support = Embedding_MSA.split_support(
            children, root, num_seqs, keys, replicate_keys, len(replicates)
        )
        clades = {}
        stack = [root]
        order = []
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(children[node])
        for node in reversed(order):
            clades[node] = frozenset([node]) if node < num_seqs else frozenset().union(
                *(clades[kid] for kid in children[node])
            )
        return {clades[node]: support[node] for node in order if support[node] >= 0}

    def test_a_split_in_one_of_two_replicates_scores_fifty(self):
        # ((0,1),(2,3)),4 against ((0,2),(1,3)),4.
        first = np.array([[0, 1, 1, 2], [2, 3, 1, 2], [5, 6, 2, 4], [4, 7, 3, 5]], dtype=float)
        second = np.array([[0, 2, 1, 2], [1, 3, 1, 2], [5, 6, 2, 4], [4, 7, 3, 5]], dtype=float)
        support = self.support(first, [first, second], 5)
        self.assertEqual(support, {frozenset({0, 1}): 50, frozenset({2, 3}): 50})

    def test_splits_count_wherever_the_replicate_is_rooted(self):
        # The same unrooted tree, rooted inside (0,1) in one replicate and
        # beside 4 in the other: both hold the splits {0,1} | {2,3,4} and
        # {0,1,2} | {3,4}.
        beside_four = np.array([[0, 1, 1, 2], [2, 5, 2, 3], [3, 6, 3, 4], [4, 7, 4, 5]], dtype=float)
        inside = np.array([[3, 4, 1, 2], [2, 5, 2, 3], [1, 6, 3, 4], [0, 7, 4, 5]], dtype=float)
        support = self.support(beside_four, [beside_four, inside], 5, nj=True)
        # The midpoint falls between {0,1,2} and {3,4}, so both root children
        # carry that split.
        self.assertEqual(support, {
            frozenset({0, 1}): 100, frozenset({0, 1, 2}): 100, frozenset({3, 4}): 100,
        })

    def test_split_keys_skip_single_leaf_splits(self):
        tree = np.array([[0, 1, 1, 2], [2, 5, 2, 3], [3, 6, 3, 4], [4, 7, 4, 5]], dtype=float)
        keys = Embedding_MSA.guide_tree_leaf_keys(5, 42)
        # {0,1} and {0,1,2} | {3,4}; {0,1,2,3} splits off leaf 4.
        self.assertEqual(Embedding_MSA.linkage_split_keys(tree, 5, keys).size, 2)


class NewickTextTests(unittest.TestCase):
    def test_upgma_tree_text(self):
        tree = sch.linkage(np.array([2.0, 6.0, 6.0]), method="average")
        parent, length = Embedding_MSA.guide_tree_branches(tree, 3)
        children, root = Embedding_MSA.rooted_children(parent)
        text = Embedding_MSA.newick_text(children, length, root, ["a", "b", "c"])
        self.assertEqual(text, "(c:3,(a:1,b:1):2);")

    def test_support_labels_follow_internal_nodes(self):
        tree = sch.linkage(np.array([2.0, 6.0, 6.0]), method="average")
        parent, length = Embedding_MSA.guide_tree_branches(tree, 3)
        children, root = Embedding_MSA.rooted_children(parent)
        support = np.array([-1, -1, -1, 87, -1])
        text = Embedding_MSA.newick_text(children, length, root, ["a", "b", "c"], support)
        self.assertEqual(text, "(c:3,(a:1,b:1)87:2);")

    def test_labels_with_newick_punctuation_are_quoted(self):
        for label, written in (
            ("sp|P12345|NAME_HUMAN", "sp|P12345|NAME_HUMAN"),
            ("seq 1", "'seq 1'"),
            ("O'Brien", "'O''Brien'"),
            ("x(1):2;3,4[5]", "'x(1):2;3,4[5]'"),
        ):
            with self.subTest(label=label):
                self.assertEqual(Embedding_MSA.newick_label(label), written)
                text = Embedding_MSA.newick_text([[], [], [0, 1]], np.ones(3), 2, [label, "b"])
                self.assertEqual(parse_newick(text)[1][0], label)

    def test_a_deep_caterpillar_tree_writes_without_recursion(self):
        count = 5000
        tree = np.zeros((count - 1, 4))
        tree[0] = [0, 1, 1.0, 2]
        for step in range(1, count - 1):
            tree[step] = [step + 1, count + step - 1, step + 1.0, step + 2]
        parent, length = Embedding_MSA.guide_tree_branches(tree, count)
        children, root = Embedding_MSA.rooted_children(parent)
        text = Embedding_MSA.newick_text(children, length, root, [f"s{i}" for i in range(count)])
        self.assertEqual(text.count("("), count - 1)
        self.assertTrue(text.endswith(";"))

    def test_one_sequence_is_its_own_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tree.nwk"
            Embedding_MSA.write_guide_tree(path, np.zeros((0, 4)), ["only one"])
            self.assertEqual(path.read_text(encoding="utf-8"), "'only one';\n")


class BuilderGuideTreeTests(unittest.TestCase):
    def run_builder(self, fixture, **settings):
        values = builder_settings(fixture, **settings)
        seen = {}
        real_write = Embedding_MSA.write_guide_tree

        def record_write(path, linkage_matrix, headers, nj_branch_lengths, *args):
            seen["path"] = pathlib.Path(path)
            seen["tree"] = np.array(linkage_matrix)
            seen["headers"] = list(headers)
            seen["nj"] = None if nj_branch_lengths is None else np.array(nj_branch_lengths)
            return real_write(path, linkage_matrix, headers, nj_branch_lengths, *args)

        def stop(*args, **kwargs):
            raise GuideTreeBuilt

        with mock.patch.multiple(
            Embedding_MSA, benchmark_msa_devices=stop, write_guide_tree=record_write, **values
        ), mock.patch.object(Embedding_MSA.mp, "set_start_method"), mock.patch.object(
            Embedding_MSA.mp, "Pool", InProcessPool
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:
                Embedding_MSA.run_msa_builder()
            except GuideTreeBuilt:
                pass
        gc.collect()
        seen["text"] = seen["path"].read_text(encoding="utf-8")
        return seen

    def test_upgma_tree_is_saved_beside_the_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NetworkFixture(directory, seed=4)
            seen = self.run_builder(fixture, TREE_METHOD=UPGMA)
            expected_path = fixture.msa_dir / f"Fixture_[{MODEL}]_guide_tree.nwk"

        self.assertEqual(seen["path"], expected_path)
        self.assertTrue(seen["text"].endswith(";\n"))
        self.assertEqual(seen["text"].count("\n"), 1)
        nodes = parse_newick(seen["text"])
        names = [node[0] for node in nodes if node[0]]
        self.assertEqual(sorted(names), sorted(seen["headers"]))
        # Without perturbed replicates no node carries a label.
        self.assertFalse(any(node[3] for node in nodes))
        np.testing.assert_allclose(
            squareform(leaf_path_lengths(nodes, seen["headers"]), checks=False),
            sch.cophenet(seen["tree"]),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_neighbor_joining_tree_keeps_its_branch_lengths_and_labels_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = NetworkFixture(directory, seed=6)
            seen = self.run_builder(
                fixture, TREE_METHOD=NEIGHBOR_JOINING, BOOTSTRAP_TREE=True, NUM_TREES=4
            )

        nodes = parse_newick(seen["text"])
        count = len(seen["headers"])
        parent, length = Embedding_MSA.guide_tree_branches(seen["tree"], count, seen["nj"])
        np.testing.assert_allclose(
            leaf_path_lengths(nodes, seen["headers"]),
            tree_path_lengths(parent, length, count),
            rtol=1e-5,
            atol=1e-6,
        )
        labels = [int(node[3]) for node in nodes if node[3]]
        self.assertTrue(labels)
        # Four replicates give multiples of 25 per cent.
        self.assertTrue(set(labels) <= {0, 25, 50, 75, 100})

    def test_neighbor_joining_consensus_averages_replicate_path_lengths(self):
        real_join = Embedding_MSA.neighbor_joining_condensed
        real_finalize = Embedding_MSA.finalize_cophenetic_consensus
        joins = []
        observed = {}

        def record_join(D_condensed, num_seqs, *args, **kwargs):
            result = real_join(D_condensed, num_seqs, *args, **kwargs)
            joins.append((np.array(D_condensed), result))
            return result

        def record_finalize(distance_matrix, num_seqs, edge_i, edge_j, *args):
            if edge_i is not None:
                observed["pairs"] = (np.array(edge_i), np.array(edge_j))
            return real_finalize(distance_matrix, num_seqs, edge_i, edge_j, *args)

        # A complete network averages every pair; an incomplete one only the
        # observed pairs.
        for complete in (True, False):
            joins.clear()
            observed.clear()
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as directory:
                fixture = NetworkFixture(directory, complete=complete, seed=42)
                with mock.patch.multiple(
                    Embedding_MSA,
                    neighbor_joining_condensed=record_join,
                    finalize_cophenetic_consensus=record_finalize,
                ):
                    seen = self.run_builder(
                        fixture, TREE_METHOD=NEIGHBOR_JOINING, BOOTSTRAP_TREE=True, NUM_TREES=3
                    )

                count = len(seen["headers"])
                *replicates, (final_input, _) = joins
                self.assertEqual(len(replicates), 3)
                mean = summed_path_lengths([tree for _, tree in replicates], count) / 3
                if complete:
                    self.assertNotIn("pairs", observed)
                    np.testing.assert_allclose(
                        final_input, squareform(mean, checks=False), rtol=1e-5, atol=1e-6
                    )
                else:
                    edge_i, edge_j = observed["pairs"]
                    self.assertLess(edge_i.size, count * (count - 1) // 2)
                    np.testing.assert_allclose(
                        final_input[condensed_indices(count, edge_i, edge_j)],
                        mean[edge_i, edge_j],
                        rtol=1e-5,
                        atol=1e-6,
                    )


if __name__ == "__main__":
    unittest.main()
