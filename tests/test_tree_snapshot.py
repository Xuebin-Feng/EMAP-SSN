# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""A tree held in an inspection snapshot: the clades and names it answers (web_ui.tree_snapshot).

The tree of most tests is tests.tree_fixtures.SMALL_TREE, opened in a Viewer as an uploaded tree, so every number
can be worked out by hand: six leaves A to F (network nodes 0 to 5), the nodes X and Y (6, 7) that no leaf names, and
the clades 2 (A B), 6 (D E), 7 (C D E) and 9 (the root).
"""

import base64
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
for folder in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import Command_Engine  # noqa: E402
from commands import color, select  # noqa: E402
from desktop.Viewer_Inspection import ViewerInspectionError  # noqa: E402
from tests.command_fixtures import reported_outcomes  # noqa: E402
from tests.tree_fixtures import (  # noqa: E402
    SMALL_TREE, give_edges, headers_of, make_alignment_folder, make_viewer, neighbor_joining_tree, snapshot_viewer,
    upload_tree,
)
from web_ui import tree_backend, tree_snapshot  # noqa: E402

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

    def test_outside_the_subset_counts_the_network_leaves_only(self):
        (row,) = rows_of(self.query(contains=self.subset('"A" | "D"')))
        self.assertEqual((row["clade"], row["leaves"]["total"], row["in_subset"], row["outside_subset"]), (9, 6, 2, 3))

    def test_the_lists_of_names(self):
        self.assertEqual(rows_of(self.query("query_tree_names", list="leaves_not_in_network")), [{"name": "F", "leaf": 8}])
        self.assertEqual(rows_of(self.query("query_tree_names", list="network_nodes_not_in_tree")), [{"name": "X", "index": 5}])
        for name in ("leaves_not_in_alignment", "ambiguous_names"):
            self.assertEqual(rows_of(self.query("query_tree_names", list=name)), [])


class HiddenWithLeavesOutsideTests(SmallTreeCase):
    """Node 0 (leaf A) is hidden, and leaf F is no node of the network: F is not hidden, it is not there."""

    headers = list("ABCDE") + ["X"]

    def test_a_leaf_outside_the_network_is_not_counted_hidden(self):
        self.viewer.visible_mask = np.array([False, True, True, True, True, True])
        sid = self.service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        rows = {row["clade"]: row["leaves"] for row in self.store.execute("query_clades", sid)["rows"]}
        self.assertEqual(rows[9], {"total": 6, "in_network": 5, "not_in_network": 1, "hidden": 1})
        self.assertEqual(rows[2]["hidden"], 1)
        self.assertEqual(rows[7]["hidden"], 0)


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


# ---- #clade:N# in expressions: the live tree and the snapshot's -----------------------------------------------------------


def expected_clades(state):
    """{clade: the network nodes of its leaves, sorted}, worked out by walking the payload's parents (not by the index)."""
    parent = np.frombuffer(base64.b64decode(state["parent"]), dtype="<i4")
    viewer_node = np.frombuffer(base64.b64decode(state["viewer_node"]), dtype="<i4")
    children = {}
    for node, above in enumerate(parent.tolist()):
        if above >= 0:
            children.setdefault(above, []).append(node)
    leaf_nodes = [node for node in range(len(parent)) if node not in children]
    ordinal = {node: position for position, node in enumerate(leaf_nodes)}

    def leaves_below(node):
        found, stack = [], [node]
        while stack:
            current = stack.pop()
            if current in children:
                stack.extend(children[current])
            else:
                found.append(current)
        return found

    return {
        node: sorted(int(viewer_node[ordinal[leaf]]) for leaf in leaves_below(node) if viewer_node[ordinal[leaf]] >= 0)
        for node in children
    }


class CladeLabelCase(unittest.TestCase):
    """A Viewer with each of the three trees, and the commands that take ``#clade:N#`` run over it."""

    GUIDE_HEADERS = headers_of(9)

    def setUp(self):
        patcher = mock.patch.dict(tree_backend._TREE_CACHE, {"key": None, "tree": None})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def viewer(self, source):
        """A Viewer whose network has the nodes of ``source``'s tree and that keeps ``source`` as its tree."""
        if source == "uploaded":
            viewer, service = snapshot_viewer(HEADERS, clusters=CLUSTERS)
            self.assertTrue(upload_tree(viewer)["ok"])
        elif source == "guide":
            folder = tempfile.TemporaryDirectory()
            self.addCleanup(folder.cleanup)
            msa, _ = make_alignment_folder(folder.name, self.GUIDE_HEADERS)
            viewer, service = snapshot_viewer(self.GUIDE_HEADERS, base=make_viewer(msa, self.GUIDE_HEADERS))
        else:
            headers = ["n%d" % i for i in range(7)]
            viewer = give_edges(make_viewer(None, headers), [(0, 1), (1, 2), (3, 4), (4, 5), (2, 3), (5, 6)],
                                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
            viewer, service = snapshot_viewer(headers, base=viewer)
        viewer._tree_source = source
        viewer.service = service
        return viewer

    def state(self, viewer, source):
        return tree_backend._source_state(viewer, source)[0]

    def evaluate(self, viewer, expression):
        with contextlib.redirect_stdout(io.StringIO()):
            return Command_Engine.parse_advanced_expression(
                expression, None, None, viewer.full_headers, viewer.cluster_labels, viewer.group_labels,
                label_providers=Command_Engine.label_providers(viewer))

    def run_select(self, viewer, *args):
        """Run select quietly over ``viewer``; returns (outcomes, failures, messages the portal was told, printed text)."""
        viewer.console_text = SimpleNamespace(text="")
        viewer.update_selection_visual = mock.Mock()
        told = []
        output = io.StringIO()
        with reported_outcomes() as (succeeded, failed), contextlib.redirect_stdout(output), \
                mock.patch("Viewer_Command_Portal.report", side_effect=lambda *a, **k: told.append(str(k.get("message")))):
            select.run(viewer, list(args))
        return succeeded, failed, told, output.getvalue()


class LiveCladeTests(CladeLabelCase):
    def test_a_clade_is_the_network_nodes_of_its_leaves_for_each_source(self):
        for source in ("guide", "threshold", "uploaded"):
            viewer = self.viewer(source)
            expected = expected_clades(self.state(viewer, source))
            self.assertTrue(len(expected) >= 3, source)
            for clade, nodes in expected.items():
                with self.subTest(source=source, clade=clade):
                    self.assertEqual(np.flatnonzero(self.evaluate(viewer, f"#clade:{clade}#")).tolist(), nodes)

    def test_the_small_tree_by_hand(self):
        viewer = self.viewer("uploaded")
        for clade, nodes in {2: [0, 1], 6: [3, 4], 7: [2, 3, 4], 9: [0, 1, 2, 3, 4, 5]}.items():
            self.assertEqual(np.flatnonzero(self.evaluate(viewer, f"#clade:{clade}#")).tolist(), nodes)

    def test_select_selects_exactly_the_clades_network_leaves_and_the_result_names_the_tree(self):
        names = {"guide": "the MSA Guide Tree", "threshold": "the network threshold tree", "uploaded": "the tree file small.nwk"}
        for source, name in names.items():
            viewer = self.viewer(source)
            clade, nodes = sorted(expected_clades(self.state(viewer, source)).items())[1]
            with self.subTest(source=source):
                viewer.selected_indices = []
                succeeded, failed, told, printed = self.run_select(viewer, f"#clade:{clade}#")
                failed.assert_not_called()
                succeeded.assert_called_once()
                self.assertEqual(sorted(viewer.selected_indices), nodes)
                self.assertIn(f"Clade {clade} of {name}.", told)
                self.assertIn(f"Clade {clade} of {name}.", printed)

    def test_the_tree_used_is_the_one_tree_view_keeps(self):
        viewer = self.viewer("uploaded")
        # The same network also has edges: the threshold tree is another tree with other clades.
        give_edges(viewer, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        resolve = viewer.web_plugin_registry.label_providers()["clade"]
        mask, note = resolve("7")
        self.assertEqual((np.flatnonzero(mask).tolist(), str(note)), ([2, 3, 4], "Clade 7 of the tree file small.nwk."))
        viewer._tree_source = "threshold"
        expected = expected_clades(self.state(viewer, "threshold"))
        clade = sorted(expected)[0]
        mask, note = resolve(str(clade))
        self.assertEqual((np.flatnonzero(mask).tolist(), str(note)), (expected[clade], f"Clade {clade} of the network threshold tree."))

    def test_an_unknown_clade_changes_nothing_and_says_which_tree_has_none(self):
        viewer = self.viewer("guide")
        internal = expected_clades(self.state(viewer, "guide"))
        leaf = next(node for node in range(self.state(viewer, "guide")["counts"]["nodes"]) if node not in internal)
        for text in (str(leaf), "999", "x", "-1", "07", "1.5", ""):
            with self.subTest(clade=text):
                viewer.selected_indices = [1]
                succeeded, failed, told, _ = self.run_select(viewer, f"#clade:{text}#")
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertEqual(viewer.selected_indices, [1])
                self.assertTrue(any("clade" in str(call.args[1]).lower() for call in failed.call_args_list))
        viewer.selected_indices = [1]
        _, failed, _, _ = self.run_select(viewer, "#clade:999#")
        self.assertIn("There is no clade 999 in the MSA Guide Tree", str(failed.call_args.args[1]))

    def test_an_expression_with_one_good_and_one_unknown_clade_changes_nothing(self):
        viewer = self.viewer("uploaded")
        viewer.selected_indices = [5]
        succeeded, failed, told, _ = self.run_select(viewer, "#clade:7#|#clade:8#")
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertEqual(viewer.selected_indices, [5])
        self.assertFalse([text for text in told if text.startswith("Clade")])

    def test_a_tree_that_cannot_be_shown_fails_in_words(self):
        viewer = self.viewer("guide")
        viewer.alignment = None
        viewer.selected_indices = [1]
        succeeded, failed, _, _ = self.run_select(viewer, "#clade:3#")
        succeeded.assert_not_called()
        self.assertIn("cannot be shown", str(failed.call_args.args[1]))
        self.assertEqual(viewer.selected_indices, [1])

    def test_a_group_named_like_a_clade_label_fails_naming_both(self):
        viewer = self.viewer("uploaded")
        viewer.group_labels = [["clade:7"]] + [[] for _ in range(len(HEADERS) - 1)]
        viewer.selected_indices = [5]
        succeeded, failed, _, _ = self.run_select(viewer, "#clade:7#")
        succeeded.assert_not_called()
        self.assertIn("custom group", str(failed.call_args.args[1]))
        self.assertIn("clade:7", str(failed.call_args.args[1]))
        self.assertEqual(viewer.selected_indices, [5])

    def test_clusters_noise_and_groups_read_as_before(self):
        viewer = self.viewer("uploaded")
        self.assertEqual(np.flatnonzero(self.evaluate(viewer, "#Cluster_1#")).tolist(), [2, 3, 4])
        self.assertEqual(np.flatnonzero(self.evaluate(viewer, "#Noise#")).tolist(), [5])
        self.assertEqual(np.flatnonzero(self.evaluate(viewer, "#clade:7# & #Cluster_1#")).tolist(), [2, 3, 4])
        self.assertEqual(np.flatnonzero(self.evaluate(viewer, "#clade:9# & !#clade:7#")).tolist(), [0, 1, 5])

    def test_color_takes_a_clade_label_too(self):
        viewer = self.viewer("uploaded")
        count = len(HEADERS)
        viewer.current_colors = np.ones((count, 4))
        viewer.current_sizes = np.full(count, 5.0)
        for name in ("_save_state", "promote_nodes", "update_nodes", "update_edges"):
            setattr(viewer, name, mock.Mock())
        viewer.update_selection_visual = mock.Mock()
        viewer.selected_indices = []
        viewer.console_text = SimpleNamespace(text="")
        told = []
        with reported_outcomes() as (succeeded, failed), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch("Viewer_Command_Portal.report", side_effect=lambda *a, **k: told.append(str(k.get("message")))):
            color.run(viewer, ["#clade:2#", "red"])
        failed.assert_not_called()
        self.assertEqual(np.flatnonzero(viewer.current_colors[:, 1] == 0).tolist(), [0, 1])
        self.assertIn("Clade 2 of the tree file small.nwk.", told)


class SnapshotCladeTests(SmallTreeCase):
    def members(self, subset):
        return self.query("query_nodes", subset_id=subset, fields=["index"])["rows"]

    def test_a_clade_label_makes_the_subset_of_its_network_leaves(self):
        for clade, nodes in {2: [0, 1], 6: [3, 4], 7: [2, 3, 4], 9: [0, 1, 2, 3, 4, 5]}.items():
            with self.subTest(clade=clade):
                self.assertEqual([row["index"] for row in self.members(self.subset(f"#clade:{clade}#"))], nodes)

    def test_the_subset_size_is_the_leaf_count_query_clades_gives(self):
        for row in self.clades():
            with self.subTest(clade=row["clade"]):
                matched = self.store.execute("create_subset", self.sid, scope="all", expression=f"#clade:{row['clade']}#")["matched_count"]
                self.assertEqual(matched, row["leaves"]["in_network"])

    def test_it_combines_with_other_labels_and_the_scope(self):
        self.assertEqual([row["index"] for row in self.members(self.subset("#clade:7# & !#Cluster_0#"))], [2, 3, 4])
        scoped = self.store.execute("create_subset", self.sid, scope="visible", expression="#clade:7#")
        self.assertEqual(scoped["matched_count"], 2)  # E is hidden

    def test_an_unknown_clade_is_refused_in_words(self):
        for text in ("3", "0", "10", "x"):
            with self.subTest(clade=text), self.assertRaisesRegex(ViewerInspectionError, "clade"):
                self.subset(f"#clade:{text}#")
        with self.assertRaisesRegex(ViewerInspectionError, "There is no clade 3 in the tree file small.nwk"):
            self.subset("#clade:3#")

    def test_without_the_tree_captured_it_says_to_recapture(self):
        sid = self.service.capture_snapshot()
        with self.assertRaisesRegex(ViewerInspectionError, r"Recapture get_summary with include_tree=true for #clade:\.\.\.# labels"):
            self.store.execute("create_subset", sid, scope="all", expression="#clade:7#")
        # Every other label still works in that snapshot.
        subset = self.store.execute("create_subset", sid, scope="all", expression="#Cluster_1#")
        self.assertEqual(subset["matched_count"], 3)

    def test_the_snapshots_tree_is_the_one_it_froze_not_the_one_tree_view_keeps_now(self):
        give_edges(self.viewer, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        self.viewer._tree_source = "threshold"
        self.assertEqual([row["index"] for row in self.members(self.subset("#clade:7#"))], [2, 3, 4])

    def test_a_group_named_like_the_label_is_refused_naming_both(self):
        self.viewer.group_labels = [["clade:7"]] + [[] for _ in range(7)]
        sid = self.service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        with self.assertRaisesRegex(ViewerInspectionError, "custom group"):
            self.store.execute("create_subset", sid, scope="all", expression="#clade:7#")


class LabelRegistrationTests(unittest.TestCase):
    def registry(self):
        from web_ui.Plugin_Manager import WebPluginRegistry
        return WebPluginRegistry(SimpleNamespace(name="viewer"))

    def test_a_namespace_resolves_with_the_viewer_and_the_value(self):
        registry = self.registry()
        seen = []
        registry.register_label_namespace("demo", "part", lambda viewer, value: seen.append((viewer.name, value)) or "mask")
        self.assertEqual(list(registry.label_providers()), ["part"])
        self.assertEqual(registry.label_providers()["part"]("3"), "mask")
        self.assertEqual(seen, [("viewer", "3")])

    def test_a_bad_registration_is_refused(self):
        from web_ui.Plugin_Manager import WebPluginError
        registry = self.registry()
        for namespace, resolve in (("Part", print), ("", print), ("a:b", print), (3, print), ("part", None)):
            with self.subTest(namespace=namespace), self.assertRaises(WebPluginError):
                registry.register_label_namespace("demo", namespace, resolve)
        registry.register_label_namespace("demo", "part", print)
        registry.register_label_namespace("demo", "part", print)  # its own again replaces it
        with self.assertRaisesRegex(WebPluginError, "already registered by plugin 'demo'"):
            registry.register_label_namespace("other", "part", print)

    def test_a_snapshot_part_declares_its_namespaces_and_a_namespace_has_one_part(self):
        from web_ui.Plugin_Manager import WebPluginError
        registry = self.registry()
        declare = lambda name, **kw: registry.register_snapshot_part(
            "demo", name, overview=dict, capture=dict, queries={}, summary_arguments={f"include_{name}": "switch"}, **kw)
        declare("one", labels={"part": lambda frozen, snapshot, value: None})
        self.assertIn("part", registry.snapshot_parts()["one"].labels)
        with self.assertRaisesRegex(WebPluginError, "already declared by another part"):
            declare("two", labels={"part": lambda frozen, snapshot, value: None})
        for labels in ({"Bad": print}, {"part2": None}, ["part"]):
            with self.subTest(labels=labels), self.assertRaises(WebPluginError):
                declare("three", labels=labels)

    def test_a_rolled_back_registration_takes_its_namespace_with_it(self):
        registry = self.registry()
        saved = registry.snapshot()
        registry.register_label_namespace("demo", "part", print)
        registry.restore(saved)
        self.assertEqual(registry.label_providers(), {})


if __name__ == "__main__":
    unittest.main()


class AboutRootTests(CladeLabelCase):
    """Every page of query_clades names the root its clades assume."""

    def pages(self, source):
        viewer = self.viewer(source)
        service = viewer.service
        sid = service.capture_snapshot(parts={"tree": {"tree_source": source}})
        first = service.snapshots.execute("query_clades", sid, limit=1)
        second = service.snapshots.execute("query_clades", sid, limit=1, cursor=first["next_cursor"])
        subset = service.snapshots.execute("create_subset", sid, scope="all")["subset_id"]
        held = service.snapshots.execute("query_clades", sid, contains=subset)
        summary = service.snapshots.execute("get_summary", sid)["parts"]["tree"]["captured"]
        return summary, [first, second, held]

    def test_each_source_says_its_root_on_every_page(self):
        expected = {"guide": "file", "uploaded": "file", "threshold": "network"}
        for source, root in expected.items():
            summary, pages = self.pages(source)
            self.assertEqual(summary["root"], root, source)
            for page in pages:
                with self.subTest(source=source):
                    self.assertEqual(page["about"], {"root": root, "rooted": summary["rooted"]})
                    self.assertLess(len(json.dumps(page["about"])), 60)

    def test_an_unrooted_guide_tree_says_midpoint(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        linkage, lengths = neighbor_joining_tree(9)
        msa, _ = make_alignment_folder(folder.name, self.GUIDE_HEADERS, linkage=linkage, nj_branch_lengths=lengths)
        viewer, service = snapshot_viewer(self.GUIDE_HEADERS, base=make_viewer(msa, self.GUIDE_HEADERS))
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "guide"}})
        page = service.snapshots.execute("query_clades", sid)
        self.assertEqual(page["about"], {"root": "midpoint", "rooted": False})

    def test_a_threshold_forest_says_artificial(self):
        headers = ["n%d" % i for i in range(6)]
        viewer = give_edges(make_viewer(None, headers), [(0, 1), (1, 2), (3, 4), (4, 5)], [0.9, 0.8, 0.7, 0.6])
        viewer, service = snapshot_viewer(headers, base=viewer)
        sid = service.capture_snapshot(parts={"tree": {"tree_source": "threshold"}})
        page = service.snapshots.execute("query_clades", sid)
        self.assertEqual(page["about"]["root"], "artificial")
