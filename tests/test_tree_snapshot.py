# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A tree held in an inspection snapshot: the clades and names it answers (web_ui.tree_snapshot).

The tree of most tests is tests.tree_fixtures.SMALL_TREE, opened in a Viewer as an uploaded tree, so every number
can be worked out by hand: six leaves A to F (network nodes 0 to 5), the nodes X and Y (6, 7) that no leaf names, and
the clades 2 (A B), 6 (D E), 7 (C D E) and 9 (the root).
"""

import json
import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
for folder in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from desktop.Viewer_Inspection import ViewerInspectionError  # noqa: E402
from tests.tree_fixtures import SMALL_TREE, give_edges, make_viewer, snapshot_viewer, upload_tree  # noqa: E402
from web_ui import tree_snapshot  # noqa: E402

HEADERS = list("ABCDEF") + ["X", "Y"]
# E is hidden; the clusters are A B: 0, C D E: 1, F: noise, X Y: 2.
VISIBLE = [True, True, True, True, False, True, True, True]
CLUSTERS = [0, 0, 1, 1, 1, -1, 2, 2]
METADATA = {
    "Org": {"type": "text", "values": np.array(["x", "x", "y", "y", "y", "z", "q", "q"], dtype=object)},
    "Num": {"type": "number", "values": np.arange(8, dtype=float)},
}


def rows_of(page):
    return page["rows"]


class SmallTreeCase(unittest.TestCase):
    """A snapshot of the small tree opened as an uploaded one."""

    headers = HEADERS
    tree = None
    upload = {}

    def setUp(self):
        count = len(self.headers)
        self.viewer, self.service = snapshot_viewer(
            self.headers, visible=VISIBLE[:count] if count == 8 else None, clusters=self.clusters(count),
            metadata=self.metadata(count))
        self.assertTrue(upload_tree(self.viewer, self.tree, **self.upload)["ok"])
        self.store = self.service.snapshots
        self.sid = self.service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})

    def clusters(self, count):
        return CLUSTERS[:count] if count == 8 else None

    def metadata(self, count):
        return METADATA if count == 8 else {}

    def query(self, action="query_clades", **arguments):
        return self.store.execute(action, self.sid, **arguments)

    def clades(self, **arguments):
        return rows_of(self.query(**arguments))

    def subset(self, expression):
        return self.store.execute("create_subset", self.sid, scope="all", expression=expression)["subset_id"]


class CladeRowTests(SmallTreeCase):
    def test_the_rows_of_every_clade_in_the_default_order(self):
        rows = self.clades()
        self.assertEqual(rows, [
            {"clade": 9, "parent": None, "depth": 0, "height": 0.0, "branch_length": None, "support": None,
             "leaves": {"total": 6, "in_network": 6, "not_in_network": 0, "hidden": 1},
             "top_clusters": [[1, 3], [0, 2], [-1, 1]]},
            {"clade": 7, "parent": 9, "depth": 1, "height": 3.0, "branch_length": 3.0, "support": 80,
             "leaves": {"total": 3, "in_network": 3, "not_in_network": 0, "hidden": 1}, "top_clusters": [[1, 3]]},
            {"clade": 2, "parent": 9, "depth": 1, "height": 2.0, "branch_length": 2.0, "support": 90,
             "leaves": {"total": 2, "in_network": 2, "not_in_network": 0, "hidden": 0}, "top_clusters": [[0, 2]]},
            {"clade": 6, "parent": 7, "depth": 2, "height": 5.0, "branch_length": 2.0, "support": 70,
             "leaves": {"total": 2, "in_network": 2, "not_in_network": 0, "hidden": 1}, "top_clusters": [[1, 2]]},
        ])

    def test_a_leaf_is_never_a_row(self):
        self.assertEqual(sorted(row["clade"] for row in self.clades()), [2, 6, 7, 9])

    def test_the_default_order_is_most_leaves_first_then_clade_number(self):
        self.assertEqual([row["clade"] for row in self.clades()], [9, 7, 2, 6])

    def test_tree_order_has_each_clade_before_the_clades_below_it_left_to_right(self):
        self.assertEqual([row["clade"] for row in self.clades(order="tree")], [9, 2, 7, 6])

    def test_within_gives_the_clades_below_one(self):
        self.assertEqual([row["clade"] for row in self.clades(within=9)], [7, 2, 6])
        self.assertEqual([row["clade"] for row in self.clades(within=7)], [6])
        self.assertEqual(self.clades(within=6), [])
        self.assertEqual([row["clade"] for row in self.clades(within=9, order="tree")], [2, 7, 6])

    def test_within_takes_only_a_clade_of_the_tree(self):
        for bad in (0, 3, 10, -1, "7"):
            with self.subTest(within=bad), self.assertRaisesRegex(ViewerInspectionError, "within must be the number of a clade"):
                self.clades(within=bad)

    def test_the_leaf_limits_count_the_leaves_a_clade_holds_in_the_tree(self):
        self.assertEqual([row["clade"] for row in self.clades(min_leaves=3)], [9, 7])
        self.assertEqual([row["clade"] for row in self.clades(max_leaves=2)], [2, 6])
        self.assertEqual([row["clade"] for row in self.clades(min_leaves=3, max_leaves=3)], [7])
        self.assertEqual(self.clades(min_leaves=7), [])
        for name in ("min_leaves", "max_leaves"):
            with self.assertRaisesRegex(ViewerInspectionError, "at least 1"):
                self.clades(**{name: 0})

    def test_the_limits_and_within_combine(self):
        self.assertEqual([row["clade"] for row in self.clades(within=9, min_leaves=2, max_leaves=2)], [2, 6])

    def test_a_page_continues_from_its_cursor(self):
        first = self.query(limit=3)
        self.assertEqual([row["clade"] for row in first["rows"]], [9, 7, 2])
        last = self.query(limit=3, cursor=first["next_cursor"])
        self.assertEqual([row["clade"] for row in last["rows"]], [6])
        self.assertTrue(last["complete"])
        with self.assertRaisesRegex(ViewerInspectionError, "Invalid cursor"):
            self.query(limit=3, min_leaves=2, cursor=first["next_cursor"])

    def test_an_unknown_argument_is_refused(self):
        with self.assertRaisesRegex(ViewerInspectionError, "Unknown query_clades argument: colour"):
            self.query(colour="red")


class ColumnTests(SmallTreeCase):
    def test_a_text_column_gives_the_top_category_and_a_number_column_the_median(self):
        by_clade = {row["clade"]: row["columns"] for row in self.clades(columns=["Org", "Num"])}
        self.assertEqual(by_clade[9], {"Org": {"top": ["y", 3]}, "Num": {"median": 2.5}})
        self.assertEqual(by_clade[7], {"Org": {"top": ["y", 3]}, "Num": {"median": 3.0}})
        self.assertEqual(by_clade[2], {"Org": {"top": ["x", 2]}, "Num": {"median": 0.5}})
        self.assertEqual(by_clade[6], {"Org": {"top": ["y", 2]}, "Num": {"median": 3.5}})

    def test_no_columns_adds_no_field(self):
        self.assertTrue(all("columns" not in row for row in self.clades()))

    def test_an_unknown_column_is_refused(self):
        with self.assertRaisesRegex(ViewerInspectionError, "Unknown metadata column"):
            self.clades(columns=["Nope"])


class ColumnTieTests(SmallTreeCase):
    def metadata(self, count):
        return {
            "Org": {"type": "text", "values": np.array(["b", "a", "b", "a", "", "z", "q", "q"], dtype=object)},
            "Num": {"type": "number", "values": np.array([1.0, np.nan, np.inf, 4.0, np.nan, 9.0, 0, 0])},
        }

    def test_equal_counts_give_the_first_category_by_text_and_blanks_and_bad_numbers_are_left_out(self):
        by_clade = {row["clade"]: row["columns"] for row in self.clades(columns=["Org", "Num"])}
        # Clade 2 holds A (b) and B (a): a tie, so "a".
        self.assertEqual(by_clade[2]["Org"], {"top": ["a", 1]})
        # Clade 6 holds D (a) and E (blank): one value.
        self.assertEqual(by_clade[6]["Org"], {"top": ["a", 1]})
        # Clade 2: 1.0 and a missing value; clade 6: 4.0 and a missing one; clade 7: C is infinite (left out), D and E.
        self.assertEqual(by_clade[2]["Num"], {"median": 1.0})
        self.assertEqual(by_clade[6]["Num"], {"median": 4.0})
        self.assertEqual(by_clade[7]["Num"], {"median": 4.0})


class NotInNetworkTests(SmallTreeCase):
    """A network without F and Y: the leaf F is the tree's only."""

    headers = list("ABCDE") + ["X"]

    def test_leaves_the_network_lacks_are_counted_and_not_in_any_cluster_or_column(self):
        rows = {row["clade"]: row for row in self.clades()}
        self.assertEqual(rows[9]["leaves"], {"total": 6, "in_network": 5, "not_in_network": 1, "hidden": 0})
        self.assertEqual(rows[9]["top_clusters"], [])  # no clusters in this Viewer
        self.assertEqual(rows[7]["leaves"], {"total": 3, "in_network": 3, "not_in_network": 0, "hidden": 0})

    def test_the_lists_of_names(self):
        self.assertEqual(rows_of(self.query("query_tree_names", list="leaves_not_in_network")), [{"name": "F", "leaf": 8}])
        self.assertEqual(rows_of(self.query("query_tree_names", list="network_nodes_not_in_tree")), [{"name": "X", "index": 5}])
        for name in ("leaves_not_in_alignment", "ambiguous_names"):
            self.assertEqual(rows_of(self.query("query_tree_names", list=name)), [])


class HiddenAndClusterTests(SmallTreeCase):
    def clusters(self, count):
        return [5, 5, 5, 7, 7, -1, 5, 5]

    def test_hidden_counts_the_network_leaves_the_snapshot_has_not_visible(self):
        rows = {row["clade"]: row["leaves"]["hidden"] for row in self.clades()}
        self.assertEqual(rows, {9: 1, 7: 1, 2: 0, 6: 1})

    def test_top_clusters_are_at_most_three_by_count_then_label_with_noise_as_minus_one(self):
        rows = {row["clade"]: row["top_clusters"] for row in self.clades()}
        self.assertEqual(rows[9], [[5, 3], [7, 2], [-1, 1]])
        self.assertEqual(rows[7], [[7, 2], [5, 1]])
        self.assertEqual(rows[2], [[5, 2]])


class FourClustersTests(SmallTreeCase):
    def clusters(self, count):
        return [0, 1, 2, 3, 3, 3, 9, 9]

    def test_only_three_are_listed_and_a_tie_goes_to_the_lower_label(self):
        rows = {row["clade"]: row["top_clusters"] for row in self.clades()}
        self.assertEqual(rows[9], [[3, 3], [0, 1], [1, 1]])


class ContainsTests(SmallTreeCase):
    def contains(self, expression, **arguments):
        return rows_of(self.query(contains=self.subset(expression), **arguments))

    def test_the_leaves_of_a_clade_are_that_clade(self):
        (row,) = self.contains('"D" | "E"')
        self.assertEqual(row["clade"], 6)
        self.assertEqual((row["in_subset"], row["outside_subset"], row["subset_not_in_tree"]), (2, 0, 0))
        self.assertEqual(row["leaves"], {"total": 2, "in_network": 2, "not_in_network": 0, "hidden": 1})
        (row,) = self.contains('"A" | "B"')
        self.assertEqual((row["clade"], row["outside_subset"]), (2, 0))

    def test_leaves_that_do_not_form_a_clade_are_held_by_the_smallest_clade_with_the_others_outside(self):
        (row,) = self.contains('"A" | "D"')
        self.assertEqual((row["clade"], row["in_subset"], row["outside_subset"]), (9, 2, 4))
        (row,) = self.contains('"C" | "E"')
        self.assertEqual((row["clade"], row["in_subset"], row["outside_subset"]), (7, 2, 1))
        (row,) = self.contains('"C" | "F"')
        self.assertEqual((row["clade"], row["in_subset"], row["outside_subset"]), (9, 2, 4))

    def test_one_leaf_is_held_by_its_parent_clade(self):
        for expression, clade in (('"A"', 2), ('"C"', 7), ('"E"', 6), ('"F"', 9)):
            with self.subTest(expression=expression):
                self.assertEqual(self.contains(expression)[0]["clade"], clade)

    def test_nodes_that_are_not_leaves_of_the_tree_are_counted_and_left_out(self):
        (row,) = self.contains('"D" | "X" | "Y"')
        self.assertEqual((row["clade"], row["in_subset"], row["subset_not_in_tree"]), (6, 1, 2))
        (row,) = self.contains('"A" | "B" | "Y"')
        self.assertEqual((row["clade"], row["outside_subset"], row["subset_not_in_tree"]), (2, 0, 1))

    def test_a_subset_with_no_leaf_of_the_tree_is_refused(self):
        with self.assertRaisesRegex(ViewerInspectionError, "No node of the subset is a leaf"):
            self.contains('"X" | "Y"')

    def test_an_unknown_subset_and_other_filters_are_refused(self):
        with self.assertRaisesRegex(ViewerInspectionError, "Unknown subset"):
            self.query(contains="nope")
        subset = self.subset('"A"')
        for extra in ({"within": 9}, {"min_leaves": 1}, {"max_leaves": 9}, {"order": "tree"}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ViewerInspectionError, "takes no within"):
                self.query(contains=subset, **extra)

    def test_the_row_takes_the_columns_too(self):
        (row,) = self.contains('"D" | "E"', columns=["Org"])
        self.assertEqual(row["columns"], {"Org": {"top": ["y", 2]}})

    def test_the_hidden_leaves_of_the_subset_count_like_any(self):
        (row,) = self.contains('"E"')
        self.assertEqual((row["clade"], row["in_subset"]), (6, 1))


class UnaryChainTests(SmallTreeCase):
    """A node with one child holds the same leaves as its child: the smaller number, the deeper node, is the smaller clade."""

    # A B (A B)=2 then a one-child node 3 above it, then C, and the root 5 over 3 and C. Leaves A B C = nodes 0 1 4.
    tree = {"labels": ["A", "B", "C"], "parent": [2, 2, 3, 5, 5, -1], "length": [1, 1, 1, 1, 1, 0], "support": [-1] * 6}
    headers = ["A", "B", "C"]

    def test_the_deeper_of_two_nodes_with_the_same_leaves_is_the_smallest_clade(self):
        (row,) = rows_of(self.query(contains=self.subset('"A" | "B"')))
        self.assertEqual(row["clade"], 2)
        self.assertEqual([r["clade"] for r in self.clades()], [5, 2, 3])


class SupportAndNameTests(SmallTreeCase):
    tree = {**SMALL_TREE, "clade_names": {"7": "Mammals", "9": "All"}}

    def test_support_and_the_files_clade_names_are_shown(self):
        rows = {row["clade"]: row for row in self.clades()}
        self.assertEqual(rows[7]["name"], "Mammals")
        self.assertEqual(rows[9]["name"], "All")
        self.assertNotIn("name", rows[2])
        self.assertEqual((rows[2]["support"], rows[6]["support"], rows[9]["support"]), (90, 70, None))


class NoLengthsTests(SmallTreeCase):
    upload = {"has_lengths": False, "length": [0] * 10}

    def test_height_and_branch_length_are_null_without_lengths(self):
        for row in self.clades():
            self.assertIsNone(row["height"])
            self.assertIsNone(row["branch_length"])
        self.assertEqual({row["depth"] for row in self.clades()}, {0, 1, 2})


class RoundingTests(SmallTreeCase):
    tree = {**SMALL_TREE, "length": [1, 1, 1.23456789, 1, 1, 1, 123456.789, 0.000123456789, 5, 0]}

    def test_numbers_have_four_significant_digits(self):
        rows = {row["clade"]: row for row in self.clades()}
        self.assertEqual(rows[2]["branch_length"], 1.235)
        self.assertEqual(rows[6]["branch_length"], 123500.0)
        self.assertEqual(rows[7]["branch_length"], 0.0001235)
        self.assertEqual(rows[6]["height"], 123500.0)  # 123456.789 + 0.000123456789, rounded
        self.assertEqual(self.clades(columns=["Num"])[0]["columns"]["Num"]["median"], 2.5)

    def test_significant_digits(self):
        self.assertEqual(tree_snapshot.significant(0.123456), 0.1235)
        self.assertEqual(tree_snapshot.significant(98765.4321), 98770.0)
        self.assertEqual(tree_snapshot.significant(np.float32(3.14159274)), 3.142)
        self.assertEqual(tree_snapshot.significant(7), 7.0)
        for missing in (None, float("nan"), float("inf")):
            self.assertIsNone(tree_snapshot.significant(missing))


class NameListTests(unittest.TestCase):
    """The ambiguous leaves of an opened tree and the leaves the loaded alignment lacks."""

    def test_ambiguous_leaves_are_listed_with_their_names(self):
        headers = ["Homo_sapiens", "Mus_musculus", "Rattus"]
        # (Homo sapiens, Homo  sapiens)=2, (Mus_musculus, Rattus)=5, and the root 6; both Homo labels reach one header.
        tree = {"labels": ["Homo sapiens", "Homo  sapiens", "Mus_musculus", "Rattus"],
                "parent": [2, 2, 6, 5, 5, 6, -1], "length": [1, 1, 1, 1, 1, 1, 0], "support": [-1] * 7}
        viewer, service = snapshot_viewer(headers)
        self.assertTrue(upload_tree(viewer, tree)["ok"])
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        run = lambda name: service.snapshots.execute("query_tree_names", sid, list=name)["rows"]  # noqa: E731
        self.assertEqual(run("ambiguous_names"), [{"name": "Homo sapiens", "leaf": 0}, {"name": "Homo  sapiens", "leaf": 1}])
        self.assertEqual([row["name"] for row in run("leaves_not_in_network")], ["Homo sapiens", "Homo  sapiens"])
        # The header both labels reach is named by neither.
        self.assertEqual(run("network_nodes_not_in_tree"), [{"name": "Homo_sapiens", "index": 0}])
        self.assertEqual(run("leaves_not_in_alignment"), [])

    def test_a_list_must_be_named_and_known(self):
        viewer, service = snapshot_viewer(HEADERS)
        upload_tree(viewer)
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        for arguments in ({}, {"list": "everything"}, {"list": "leaves_not_in_network", "extra": 1}):
            with self.subTest(arguments=arguments), self.assertRaises(ViewerInspectionError):
                service.snapshots.execute("query_tree_names", sid, **arguments)

    def test_the_lists_page_like_every_read(self):
        headers = ["h%d" % i for i in range(59)] + ["L0"]
        labels = ["L%d" % i for i in range(60)]
        # A star: 60 leaves under one root, none named in the network.
        tree = {"labels": labels, "parent": [60] * 60 + [-1], "length": [1] * 60 + [0], "support": [-1] * 61}
        viewer, service = snapshot_viewer(headers)
        upload_tree(viewer, tree)
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        seen, cursor = [], None
        while True:
            page = service.snapshots.execute("query_tree_names", sid, list="leaves_not_in_network", limit=25, cursor=cursor)
            seen += [row["name"] for row in page["rows"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual(seen, labels[1:])  # the one leaf the network names is not listed
        self.assertEqual(len(service.snapshots.execute("query_tree_names", sid, list="network_nodes_not_in_tree", limit=500)["rows"]), 59)

    def test_a_long_name_is_cut_and_says_so(self):
        headers = ["short", "B"]
        tree = {"labels": ["A" * 900, "B"], "parent": [2, 2, -1], "length": [1, 1, 0], "support": [-1] * 3}
        viewer, service = snapshot_viewer(headers)
        upload_tree(viewer, tree)
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        row = service.snapshots.execute("query_tree_names", sid, list="leaves_not_in_network")["rows"][0]
        self.assertEqual((len(row["name"]), row["truncated"]), (500, True))


# ---- The arithmetic, against a plain walk of random trees -------------------------------------------------------------


def random_tree(leaves, seed, unary=0.15):
    """(parent, length) of a random tree of ``leaves`` leaves in post-order, some of its nodes with one child."""
    rng = np.random.default_rng(seed)
    # Built recursively into nested tuples first, then numbered in post-order.
    counter = [0]

    def shape(count):
        if count == 1:
            counter[0] += 1
            return ("leaf", counter[0] - 1)
        cut = int(rng.integers(1, count))
        node = ("node", [shape(cut), shape(count - cut)])
        if rng.random() < unary:
            node = ("node", [node])
        return node

    root = shape(leaves)
    nodes = []

    def number(item):
        if item[0] == "leaf":
            nodes.append((None, item[1], []))
            return len(nodes) - 1
        kids = [number(child) for child in item[1]]
        nodes.append((kids, None, []))
        return len(nodes) - 1

    number(root)
    parent = [-1] * len(nodes)
    for index, (kids, _, _) in enumerate(nodes):
        for kid in kids or ():
            parent[kid] = index
    length = [float(x) for x in rng.integers(1, 9, size=len(nodes))]
    length[-1] = 0.0
    return parent, length


def walk(parent):
    """The reference answers by plain recursion: leaf sets (as sorted leaf numbers), depth and block start of every node."""
    count = len(parent)
    kids = [[] for _ in range(count)]
    for node, up in enumerate(parent):
        if up >= 0:
            kids[up].append(node)
    leaf_number = {}
    leaf_sets = [None] * count
    for node in range(count):  # post-order: children first
        if not kids[node]:
            leaf_number[node] = len(leaf_number)
            leaf_sets[node] = {leaf_number[node]}
        else:
            leaf_sets[node] = set().union(*(leaf_sets[k] for k in kids[node]))
    depth = [0] * count
    for node in range(count - 2, -1, -1):
        depth[node] = depth[parent[node]] + 1
    return kids, leaf_sets, depth


class IndexAgainstAWalkTests(unittest.TestCase):
    def test_leaf_ranges_depth_and_distance_match_a_plain_walk_on_random_trees(self):
        for seed in range(40):
            with self.subTest(seed=seed):
                parent, length = random_tree(int(np.random.default_rng(seed).integers(2, 70)), seed)
                index = tree_snapshot.tree_index(parent, length)
                kids, leaf_sets, depth = walk(parent)
                for node in range(len(parent)):
                    leaves = sorted(leaf_sets[node])
                    self.assertEqual((int(index["first_leaf"][node]), int(index["last_leaf"][node])), (leaves[0], leaves[-1]))
                    self.assertEqual(leaves, list(range(leaves[0], leaves[-1] + 1)))
                    self.assertEqual(int(index["depth"][node]), depth[node])
                    total, up = 0.0, node
                    while parent[up] >= 0:
                        total += length[up]
                        up = parent[up]
                    self.assertAlmostEqual(float(index["distance"][node]), total)
                    self.assertEqual(bool(index["is_leaf"][node]), not kids[node])
                    # The block of a node's subtree runs from `start` to the node.
                    block = {n for n in range(len(parent)) if leaf_sets[n] <= leaf_sets[node] and n <= node}
                    self.assertEqual(int(index["start"][node]), min(block))

    def test_the_smallest_clade_holding_some_leaves_matches_a_plain_search(self):
        rng = np.random.default_rng(7)
        for seed in range(25):
            parent, length = random_tree(int(rng.integers(2, 60)), 100 + seed)
            index = tree_snapshot.tree_index(parent, length)
            kids, leaf_sets, _ = walk(parent)
            leaf_count = len(index["leaf_node"])
            for _ in range(20):
                chosen = rng.choice(leaf_count, size=int(rng.integers(1, min(leaf_count, 6) + 1)), replace=False)
                holding = [n for n in range(len(parent)) if kids[n] and set(chosen.tolist()) <= leaf_sets[n]]
                expected = min(holding, key=lambda n: (len(leaf_sets[n]), n))
                self.assertEqual(tree_snapshot.smallest_clade(index, chosen), expected)
        self.assertIsNone(tree_snapshot.smallest_clade(index, []))

    def test_a_tree_of_one_node_has_no_clade_to_hold_a_leaf(self):
        index = tree_snapshot.tree_index([-1], [0])
        self.assertIsNone(tree_snapshot.smallest_clade(index, [0]))

    def test_a_deep_tree_is_indexed_without_recursion(self):
        leaves = 15000  # a caterpillar: leaf 0, leaf 1, a join, leaf 2, a join of that and the one before, ...
        parent = [2, 2, -1]
        for _ in range(leaves - 2):
            join = len(parent) + 1
            parent[-1] = join  # the root so far hangs from the new join
            parent.append(join)  # the new leaf
            parent.append(-1)  # the new join is the root
        index = tree_snapshot.tree_index(parent, np.ones(len(parent)))
        self.assertEqual(int(index["is_leaf"].sum()), leaves)
        self.assertEqual(int(index["depth"][0]), leaves - 1)
        self.assertEqual(float(index["distance"][0]), leaves - 1)
        self.assertEqual((int(index["first_leaf"][-1]), int(index["last_leaf"][-1])), (0, leaves - 1))
        self.assertEqual(tree_snapshot.smallest_clade(index, [0, 1]), 2)


# ---- The threshold tree: its joining root is not a split ------------------------------------------------------------


class ThresholdRowTests(unittest.TestCase):
    def snapshot(self, edges, scores, count=6):
        headers = ["n%d" % i for i in range(count)]
        viewer = give_edges(make_viewer(None, headers), edges, scores)
        viewer, service = snapshot_viewer(headers, base=viewer, clusters=[0, 0, 0, 1, 1, 1][:count])
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "threshold"}})
        return service, sid

    def test_a_network_in_two_groups_hangs_from_a_joining_root_that_is_marked(self):
        service, sid = self.snapshot([(0, 1), (1, 2), (3, 4), (4, 5)], [0.9, 0.8, 0.7, 0.6])
        rows = service.snapshots.execute("query_clades", sid)["rows"]
        root = max(rows, key=lambda row: row["leaves"]["total"])
        self.assertEqual((root["leaves"]["total"], root["parent"], root["artificial"]), (6, None, True))
        children = [row for row in rows if row["parent"] == root["clade"]]
        self.assertEqual(len(children), 2)
        self.assertTrue(all(row.get("artificial") for row in children))
        below = [row for row in rows if row["parent"] not in (None, root["clade"])]
        self.assertTrue(below and all("artificial" not in row for row in below))
        summary = service.snapshots.execute("get_summary", sid)["parts"]["tree"]["captured"]
        self.assertEqual((summary["root"], summary["threshold"]["forest"], summary["threshold"]["groups"]), ("artificial", True, 2))

    def test_a_connected_network_has_no_artificial_clade(self):
        service, sid = self.snapshot([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], [0.9, 0.8, 0.7, 0.6, 0.5])
        rows = service.snapshots.execute("query_clades", sid)["rows"]
        self.assertTrue(all("artificial" not in row for row in rows))
        summary = service.snapshots.execute("get_summary", sid)["parts"]["tree"]["captured"]
        self.assertEqual(summary["root"], "network")
        # Every node of the network is a leaf, and the clades are the joins.
        self.assertEqual(summary["counts"]["clades"], summary["counts"]["nodes"] - 6)
        self.assertTrue(all(row["leaves"]["not_in_network"] == 0 for row in rows))

    def test_the_rows_are_json_with_no_numpy_types(self):
        service, sid = self.snapshot([(0, 1), (1, 2), (3, 4), (4, 5)], [0.9, 0.8, 0.7, 0.6])
        json.dumps(service.snapshots.execute("query_clades", sid), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
