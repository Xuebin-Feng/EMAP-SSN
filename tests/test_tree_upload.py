# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's uploaded tree (src/web_ui/tree_backend.py): the tree_upload action and its route.

The page reads a tree file and sends the tree's arrays; the Viewer never trusts them. Every malformed payload is refused
whole, with a note, and leaves the tree that was kept (and the choice of source) as it was. A tree that is taken is matched
to the network by full sequence name, kept for the Viewer's session in memory, and listed in the chooser by its file name.
"""

import base64
import json
import http.client
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.tree_fixtures import page_headers, server_viewer  # noqa: E402
from web_ui import Web_Server, tree_backend  # noqa: E402
from web_ui.Plugin_Manager import ApiRequest  # noqa: E402


def balanced(count):
    """The arrays of a balanced binary tree over ``count`` leaves, nodes in post-order: (parent, length, support)."""
    parent, length, support = [], [], []

    def build(low, high):
        if high - low == 1:
            parent.append(-1)
            length.append(1.0)
            support.append(-1)
            return len(parent) - 1
        middle = (low + high) // 2
        left, right = build(low, middle), build(middle, high)
        parent.append(-1)
        length.append(0.5)
        support.append(90)
        node = len(parent) - 1
        parent[left] = parent[right] = node
        return node

    build(0, count)
    length[-1] = 0.0
    support[-1] = -1
    return parent, length, support


def payload(names, /, **overrides):
    """A tree_upload request for a balanced tree over ``names``; each key of ``overrides`` replaces the one it names."""
    labels = names
    parent, length, support = balanced(len(labels))
    data = {
        "action": "tree_upload", "name": "primates.nwk", "format": "Newick", "tree_name": "", "labels": list(labels),
        "parent": parent, "length": length, "support": support, "rooted": True, "root_stated": False,
    }
    data.update(overrides)
    return data


def packed(values, dtype):
    with np.errstate(over="ignore"):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def viewer_over(headers):
    return SimpleNamespace(full_headers=list(headers))


HEADERS = ["sp|Q1", "sp|Q2", "sp|Q3", "sp|Q4", "sp|Q5", "tr|X1", "tr|X2", "only_in_the_network"]


def unpack(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


class RefusalTests(unittest.TestCase):
    """Every payload that is not a tree is refused with a note, and nothing is changed."""

    def setUp(self):
        self.viewer = viewer_over(HEADERS)
        self.labels = HEADERS[:5]
        result = tree_backend.handle_tree_upload(self.viewer, payload(self.labels, name="first.nwk"))
        self.assertTrue(result["ok"], result)
        self.kept = self.viewer._tree_upload
        self.viewer._tree_source = "guide"

    def refused(self, data, note):
        result = tree_backend.handle_tree_upload(self.viewer, data)
        self.assertFalse(result["ok"], result)
        self.assertFalse(result["changed"])
        self.assertEqual(result["note"], note, result)
        self.assertIn(result["note"], tree_backend.UPLOAD_REFUSALS)
        # Atomic: the tree that was kept, and the choice of source, are as they were.
        self.assertIs(self.viewer._tree_upload, self.kept)
        self.assertEqual(self.viewer._tree_source, "guide")
        return result

    def test_a_request_that_is_not_an_object_is_refused(self):
        for data in (None, [], "tree", 7):
            with self.subTest(data=data):
                self.refused(data, "bad_request")

    def test_labels_must_be_a_list_of_names(self):
        self.refused(payload(self.labels, labels="abc"), "bad_labels")
        self.refused(payload(self.labels, labels=self.labels[:4] + [5]), "bad_labels")
        self.refused(payload(self.labels, labels=self.labels[:4] + [None]), "bad_labels")
        self.refused(payload(self.labels, labels=self.labels[:4] + [""]), "empty_label")
        self.refused(payload(self.labels, labels=[]), "no_leaves")
        self.refused(payload(["only"]), "too_few_leaves")

    def test_two_leaves_with_one_name_refuse_the_tree(self):
        result = self.refused(payload(["a", "b", "a", "c"]), "duplicate_label")
        self.assertEqual((result["detail"]["label"], result["detail"]["first"], result["detail"]["second"]), ("a", 0, 2))

    def test_the_arrays_must_have_one_entry_for_each_node(self):
        data = payload(self.labels)
        self.refused({**data, "length": data["length"][:-1]}, "size_mismatch")
        self.refused({**data, "support": data["support"] + [-1]}, "size_mismatch")
        self.refused({**data, "parent": data["parent"][:-1]}, "size_mismatch")
        # Fewer nodes than leaves cannot be a tree of these leaves.
        self.refused(payload(self.labels, parent=[-1], length=[0.0], support=[-1]), "leaf_count_mismatch")
        self.refused({**data, "parent": packed(data["parent"], "<i4")[:-4]}, "size_mismatch")

    def test_the_arrays_must_be_numbers(self):
        data = payload(self.labels)
        for key, value in (("parent", "not base64!"), ("parent", {"a": 1}), ("parent", 4), ("length", ["x"] * 9),
                           ("support", [None] * 9), ("parent", [True] * 9), ("parent", [0.5] * 9), ("support", [1.5] * 9)):
            with self.subTest(key=key, value=str(value)[:20]):
                self.refused({**data, key: value}, "bad_arrays")
        # Bytes that are not a whole number of entries.
        self.refused({**data, "parent": base64.b64encode(b"123456").decode()}, "size_mismatch")

    def test_the_nodes_must_be_one_tree_in_post_order(self):
        parent, length, support = balanced(5)
        count = len(parent)
        cases = {
            "a node whose parent is not after it": list(parent[:3]) + [3] + list(parent[4:]),
            "a node that is its own parent": [0] + list(parent[1:]),
            "a cycle": [1, 0] + list(parent[2:]),
            "two roots": list(parent[:-2]) + [-1, -1],
            "no root": list(parent[:-1]) + [0],
            "a parent past the end": list(parent[:2]) + [count] + list(parent[3:]),
            "a parent below -1": [-5] + list(parent[1:]),
            "the root not last": [parent[-1]] + list(parent[1:-1]) + [parent[0]],
        }
        for name, bad in cases.items():
            with self.subTest(name):
                self.refused(payload(self.labels, parent=bad, length=length, support=support), "not_post_order")

    def test_the_leaves_of_the_tree_are_as_many_as_the_names(self):
        parent, length, support = balanced(5)
        # Seven names for a tree of five leaves.
        self.refused(payload(HEADERS[:7], parent=parent, length=length, support=support), "leaf_count_mismatch")
        # An internal node given a leaf's place: a chain, whose single leaf is the first node.
        chain = list(range(1, 9)) + [-1]
        self.refused(payload(self.labels, parent=chain, length=[1.0] * 9, support=[-1] * 9), "leaf_count_mismatch")

    def test_a_branch_length_is_finite_and_not_negative(self):
        data = payload(self.labels)
        for bad in (float("nan"), float("inf"), float("-inf"), -0.5, 1e39):
            with self.subTest(bad=bad):
                length = list(data["length"])
                length[2] = bad
                self.refused({**data, "length": length}, "bad_length")
                self.refused({**data, "length": packed(length, "<f4")}, "bad_length")

    def test_a_support_is_minus_one_or_a_percentage(self):
        data = payload(self.labels)
        for bad in (-2, 101, 127, 1000):
            with self.subTest(bad=bad):
                support = list(data["support"])
                support[5] = bad
                self.refused({**data, "support": support}, "bad_support")
        support = np.array(data["support"], dtype="i1")
        support[5] = -128
        self.refused({**data, "support": packed(support, "i1")}, "bad_support")

    def test_rooting_is_true_false_or_nothing(self):
        for bad in ("yes", 1, [True]):
            self.refused(payload(self.labels, rooted=bad), "bad_rooted")
        self.refused(payload(self.labels, root_stated="yes"), "bad_rooted")

    def test_the_files_own_words_must_be_text_and_a_few(self):
        self.refused(payload(self.labels, name=7), "bad_text")
        self.refused(payload(self.labels, format=["Newick"]), "bad_text")
        self.refused(payload(self.labels, dialects="NHX"), "bad_text")
        self.refused(payload(self.labels, dialects=["NHX"] * 30), "bad_text")
        self.refused(payload(self.labels, has_lengths="yes"), "bad_text")
        self.refused(payload(self.labels, warnings="none"), "bad_warnings")
        self.refused(payload(self.labels, warnings=[{"code": 3}]), "bad_warnings")
        self.refused(payload(self.labels, warnings=[{"code": "x", "count": -1}]), "bad_warnings")
        self.refused(payload(self.labels, warnings=[{"code": "x", "count": "many"}]), "bad_warnings")
        self.refused(payload(self.labels, warnings=[{"code": "x"}] * 30), "bad_warnings")

    def test_clade_names_must_name_internal_nodes_with_text(self):
        self.refused(payload(self.labels, clade_names="Primates"), "bad_clade_names")
        self.refused(payload(self.labels, clade_names={"0": "a leaf"}), "bad_clade_names")
        self.refused(payload(self.labels, clade_names={"99": "nowhere"}), "bad_clade_names")
        self.refused(payload(self.labels, clade_names={"-1": "nowhere"}), "bad_clade_names")
        self.refused(payload(self.labels, clade_names={"x": "not a node"}), "bad_clade_names")
        self.refused(payload(self.labels, clade_names={"2": 5}), "bad_clade_names")
        self.refused(payload(self.labels, clade_names=[[2, "ok"], [True, "boolean"]]), "bad_clade_names")

    def test_a_tree_of_more_leaves_than_the_limit_is_refused(self):
        with mock.patch.object(tree_backend, "MAX_LEAVES", 4):
            result = self.refused(payload(self.labels), "too_many_leaves")
        self.assertEqual(result["detail"]["limit"], 4)
        # A tree with that many leaves is taken.
        with mock.patch.object(tree_backend, "MAX_LEAVES", 5):
            self.assertTrue(tree_backend.handle_tree_upload(self.viewer, payload(self.labels, name="second.nwk"))["ok"])

    def test_a_tree_with_more_than_twice_the_limit_of_nodes_is_refused(self):
        # A chain of single-child nodes over two leaves: few leaves, many nodes.
        parent = [2, 2] + list(range(3, 12)) + [-1]
        count = len(parent)
        with mock.patch.object(tree_backend, "MAX_LEAVES", 3):
            self.refused(payload(["a", "b"], parent=parent, length=[1.0] * count, support=[-1] * count), "too_many_nodes")

    def test_a_name_longer_than_the_limit_is_refused(self):
        with mock.patch.object(tree_backend, "MAX_LABEL_CHARACTERS", 10):
            result = self.refused(payload(["a" * 11, "b", "c"]), "label_too_long")
            self.assertEqual((result["detail"]["leaf"], result["detail"]["limit"]), (0, 10))
            self.assertTrue(tree_backend.handle_tree_upload(self.viewer, payload(["a" * 10, "b", "c"], name="third.nwk"))["ok"])
            self.kept = self.viewer._tree_upload
            self.viewer._tree_source = "guide"
        # Clade names have the limit too.
        with mock.patch.object(tree_backend, "MAX_LABEL_CHARACTERS", 5):
            self.refused(payload(["a", "b", "c"], clade_names={"3": "much too long"}), "label_too_long")

    def test_a_payload_larger_than_the_servers_limit_is_refused(self):
        with mock.patch.object(Web_Server, "MAX_ACTION_BYTES", 20):
            result = self.refused(payload(self.labels), "too_large")
        self.assertEqual(result["detail"]["limit"], 20)
        # The limit is the web server's own: nothing larger than it could have been read.
        self.assertEqual(tree_backend._action_limit(), Web_Server.MAX_ACTION_BYTES)

    def test_a_refusal_before_the_first_tree_keeps_nothing_and_changes_no_source(self):
        viewer = viewer_over(HEADERS)
        result = tree_backend.handle_tree_upload(viewer, payload(["a", "a"]))
        self.assertFalse(result["ok"])
        self.assertFalse(hasattr(viewer, "_tree_upload"))
        self.assertFalse(hasattr(viewer, "_tree_source"))

    def test_the_lists_and_their_bytes_are_the_same_tree(self):
        data = payload(self.labels)
        as_bytes = {**data, "parent": packed(data["parent"], "<i4"), "length": packed(data["length"], "<f4"),
                    "support": packed(data["support"], "i1")}
        first = tree_backend.validate_upload(data)
        second = tree_backend.validate_upload(as_bytes)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.parent.dtype, np.dtype("<i4"))
        self.assertEqual(first.length.dtype, np.dtype("<f4"))
        self.assertEqual(first.support.dtype, np.dtype("i1"))

    def test_one_internal_node_with_one_child_is_a_tree(self):
        # A unary node is no problem: the leaves are the nodes with no child.
        parent = [2, 2, 3, -1]
        valid = tree_backend.validate_upload(payload(["a", "b"], parent=parent, length=[1, 1, 1, 0], support=[-1] * 4))
        self.assertEqual(len(valid.parent), 4)


class MatchTests(unittest.TestCase):
    """A leaf names a node by its full sequence name: exactly, or after the Viewer's own header rules. Nothing else."""

    def match(self, labels, headers):
        node, report = tree_backend.match_labels(labels, headers)
        return node.tolist(), report

    def test_an_exact_name_matches(self):
        nodes, report = self.match(["sp|Q2", "sp|Q1", "sp|Q9"], ["sp|Q1", "sp|Q2", "sp|Q3"])
        self.assertEqual(nodes, [1, 0, -1])
        self.assertEqual((report["matched"], report["exact"], report["sanitized"], report["unmatched"]), (2, 2, 0, 1))
        self.assertEqual(report["network_nodes"], 3)
        self.assertEqual(report["network_nodes_not_in_tree"], 1)

    def test_a_raw_fasta_header_matches_through_the_header_rules(self):
        headers = ["sp|Q1|NAME_HUMAN_Some_protein_[Homo_sapiens]".replace("[", "(").replace("]", ")")]
        raw = "sp|Q1|NAME_HUMAN Some  protein [Homo_sapiens]"
        nodes, report = self.match([raw, "other"], headers)
        self.assertEqual(nodes, [0, -1])
        self.assertEqual((report["exact"], report["sanitized"]), (0, 1))

    def test_each_step_is_counted(self):
        headers = ["Homo_sapiens", "Pan_troglodytes", "Mus_musculus"]
        nodes, report = self.match(["Homo_sapiens", "Pan troglodytes", "Gorilla"], headers)
        self.assertEqual(nodes, [0, 1, -1])
        self.assertEqual((report["exact"], report["sanitized"], report["unmatched"]), (1, 1, 1))

    def test_no_accession_no_case_no_underscore_matching(self):
        headers = ["sp|Q1|NAME_HUMAN Something", "Homo_sapiens", "WP_000001.1_hypothetical_protein"]
        headers = [text.replace(" ", "_") for text in headers]
        for label in ("sp|Q1|NAME_HUMAN", "Q1", "WP_000001.1", "WP_000001", "homo_sapiens", "HOMO_SAPIENS", "Homo-sapiens", "Homo", " "):
            with self.subTest(label=label):
                nodes, report = self.match([label, "zz"], headers)
                self.assertEqual(nodes, [-1, -1])
                self.assertEqual(report["matched"], 0)

    def test_two_labels_reaching_one_header_by_the_rules_are_both_left_out_and_listed(self):
        headers = ["Homo_sapiens", "Mus_musculus", "Rattus"]
        labels = ["Homo sapiens", "Homo  sapiens", "Mus_musculus", "Rattus"]
        nodes, report = self.match(labels, headers)
        self.assertEqual(nodes, [-1, -1, 1, 2])
        self.assertEqual(report["ambiguous"], 2)
        self.assertEqual(report["ambiguous_labels"], ["Homo sapiens", "Homo  sapiens"])
        self.assertEqual((report["matched"], report["unmatched"]), (2, 2))

    def test_a_label_reaching_by_the_rules_a_header_an_exact_label_has_is_left_out(self):
        nodes, report = self.match(["Homo_sapiens", "Homo sapiens", "x"], ["Homo_sapiens", "Other"])
        self.assertEqual(nodes, [0, -1, -1])
        self.assertEqual((report["exact"], report["sanitized"], report["ambiguous"]), (1, 0, 1))

    def test_a_header_the_network_repeats_belongs_to_its_first_node(self):
        nodes, _report = self.match(["twin"], ["twin", "twin"])
        self.assertEqual(nodes, [0])

    def test_the_lists_of_a_report_are_capped(self):
        labels = [f"Name {k}" for k in range(tree_backend.MAX_REPORTED_LABELS + 40)] * 1
        headers = ["Name_%d" % k for k in range(len(labels))]
        # Every label reaches its header only through the rules, and two of them reach the same one.
        labels[1] = labels[0].replace("0", "0 ")
        nodes, report = self.match(labels, headers)
        self.assertEqual(report["ambiguous"], 2)
        # The page lists the unmatched leaves itself; the Viewer sends only a bounded list of the ambiguous ones.
        self.assertLessEqual(len(report["ambiguous_labels"]), tree_backend.MAX_REPORTED_LABELS)


class StateTests(unittest.TestCase):
    """What is kept, what the page is sent, and which tree is shown."""

    def setUp(self):
        self.viewer = viewer_over(HEADERS)
        self.labels = ["sp|Q3", "sp|Q1", "sp|Q2", "elsewhere", "sp|Q5"]

    def upload(self, labels=None, **overrides):
        result = tree_backend.handle_tree_upload(self.viewer, payload(labels or self.labels, **overrides))
        self.assertTrue(result["ok"], result)
        return result

    def test_an_upload_is_kept_and_becomes_the_tree_shown(self):
        result = self.upload(name="mine.nwk", tree_name="a tree", format="NEXUS", rooted=False, root_stated=True)
        self.assertEqual((result["changed"], result["note"], result["status"]), (True, None, "ok"))
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual((state["status"], state["source"]), ("ok", "uploaded"))
        self.assertEqual(self.viewer._tree_source, "uploaded")
        self.assertEqual(state["labels"], self.labels)
        self.assertEqual(state["name"], "a tree")
        self.assertIs(state["rooted"], False)
        self.assertEqual(state["uploaded"]["file"], "mine.nwk")
        self.assertEqual(state["uploaded"]["format"], "NEXUS")
        self.assertIs(state["uploaded"]["root_stated"], True)
        # The nodes the leaves stand for, and -1 for the one that is not a node of the network.
        self.assertEqual(unpack(state["viewer_node"], "<i4").tolist(), [2, 0, 1, -1, 4])
        parent = unpack(state["parent"], "<i4")
        self.assertEqual((len(parent), int(parent[-1])), (9, -1))
        self.assertEqual(state["counts"], {
            "leaves": 5, "nodes": 9, "leaves_in_network": 4, "leaves_not_in_network": 1,
            "network_nodes": 8, "network_nodes_not_in_tree": 4,
        })
        codes = {warning["code"]: warning.get("count") for warning in state["warnings"]}
        self.assertEqual(codes, {"leaves_not_in_network": 1, "network_nodes_not_in_tree": 4})

    def test_the_wire_arrays_are_what_was_sent(self):
        data = payload(self.labels)
        self.upload()
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual(unpack(state["parent"], "<i4").tolist(), data["parent"])
        self.assertEqual(unpack(state["length"], "<f4").tolist(), data["length"])
        self.assertEqual(unpack(state["support"], "i1").tolist(), data["support"])

    def test_the_chooser_lists_the_file_by_its_name_once_there_is_one(self):
        before = tree_backend.tree_state(self.viewer)
        self.assertEqual([entry["id"] for entry in before["sources"]], ["threshold", "guide"])
        self.upload(name="mine.nwk")
        entries = tree_backend.tree_state(self.viewer)["sources"]
        self.assertEqual([entry["id"] for entry in entries], ["threshold", "guide", "uploaded"])
        self.assertEqual(entries[2], {"id": "uploaded", "available": True, "reason": None, "name": "mine.nwk"})

    def test_the_tree_survives_asking_again_and_choosing_other_sources(self):
        self.upload()
        for _ in range(2):
            self.assertEqual(tree_backend.tree_state(self.viewer)["source"], "uploaded")
        # Another source can be asked for, and the Viewer keeps what was chosen last; the upload stays listed and kept.
        other = tree_backend.tree_state(self.viewer, "guide")
        self.assertEqual(other["source"], "guide")
        self.assertEqual(other["status"], "no_alignment")
        self.assertEqual(self.viewer._tree_source, "uploaded")
        again = tree_backend.tree_state(self.viewer, "uploaded")
        self.assertEqual((again["source"], again["status"]), ("uploaded", "ok"))

    def test_the_uploaded_tree_is_asked_for_before_there_is_one(self):
        state = tree_backend.tree_state(self.viewer, "uploaded")
        self.assertEqual((state["status"], state["source"]), ("no_upload", "uploaded"))
        self.assertFalse(hasattr(self.viewer, "_tree_source"))

    def test_another_tree_replaces_the_one_before(self):
        self.upload(name="one.nwk")
        second = ["sp|Q1", "sp|Q2", "sp|Q4"]
        result = self.upload(second, name="two.nwk")
        self.assertTrue(result["changed"])
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual(state["labels"], second)
        self.assertEqual([entry["name"] for entry in state["sources"] if entry["id"] == "uploaded"], ["two.nwk"])

    def test_the_same_tree_again_is_no_change_but_is_shown_again(self):
        self.upload(name="one.nwk")
        tree_backend.tree_state(self.viewer, "threshold")
        self.viewer._tree_source = "threshold"
        result = tree_backend.handle_tree_upload(self.viewer, payload(self.labels, name="one.nwk"))
        self.assertEqual((result["ok"], result["changed"], result["note"]), (True, False, "unchanged"))
        self.assertEqual(self.viewer._tree_source, "uploaded")
        # The same tree under another name is another upload.
        self.assertTrue(tree_backend.handle_tree_upload(self.viewer, payload(self.labels, name="copy.nwk"))["changed"])

    def test_nothing_is_written_to_disk(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            before = set(os.listdir(folder))
            here = os.getcwd()
            os.chdir(folder)
            try:
                self.upload()
                tree_backend.tree_state(self.viewer)
            finally:
                os.chdir(here)
            self.assertEqual(set(os.listdir(folder)), before)

    def test_a_tree_none_of_whose_leaves_is_a_node_is_not_drawn_and_says_how_the_names_differ(self):
        labels = [f"Mus_musculus_{k}" for k in range(7)]
        result = self.upload(labels, name="renamed.treefile")
        self.assertEqual(result["status"], "no_overlap")
        self.assertEqual(result["match"]["matched"], 0)
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual((state["status"], state["source"]), ("no_overlap", "uploaded"))
        self.assertNotIn("parent", state)
        self.assertEqual(state["match"]["tree_examples"], labels[:5])
        self.assertEqual(state["match"]["network_examples"], HEADERS[:5])
        self.assertEqual(state["counts"]["leaves"], 7)
        self.assertEqual(state["uploaded"]["file"], "renamed.treefile")
        # It is still listed, and still the tree the Viewer keeps, so that the card is what the page shows after a reload.
        self.assertEqual(self.viewer._tree_source, "uploaded")
        self.assertTrue(any(entry["id"] == "uploaded" and entry["available"] for entry in state["sources"]))

    def test_the_match_follows_the_network_the_viewer_holds(self):
        self.upload()
        self.viewer.full_headers = ["sp|Q1", "x", "y"]
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual(unpack(state["viewer_node"], "<i4").tolist(), [-1, 0, -1, -1, -1])
        self.viewer.full_headers = ["nothing", "alike"]
        self.assertEqual(tree_backend.tree_state(self.viewer)["status"], "no_overlap")

    def test_the_reader_warnings_and_the_files_facts_come_back(self):
        self.upload(warnings=[{"code": "negative_lengths", "count": 2}, {"code": "no_semicolon"}], dialects=["NHX"],
                    support_name="bootstrap", clade_names={"8": "Group X"}, has_lengths=False)
        state = tree_backend.tree_state(self.viewer)
        self.assertEqual(state["warnings"][0], {"code": "negative_lengths", "count": 2})
        self.assertEqual(state["warnings"][1], {"code": "no_semicolon"})
        self.assertFalse(state["has_lengths"])
        info = state["uploaded"]
        self.assertEqual((info["dialects"], info["support_name"], info["clade_names"]), (["NHX"], "bootstrap", 1))

    def test_two_labels_of_one_header_are_listed_among_the_warnings(self):
        self.upload(["only in the network", "only  in the network", "tr|X1", "tr|X1 "])
        state = tree_backend.tree_state(self.viewer)
        codes = {warning["code"]: warning.get("count") for warning in state["warnings"]}
        # The first two reach one header by the rules, and the last reaches by the rules the header "tr|X1" has exactly.
        self.assertEqual(codes["ambiguous_names"], 3)
        self.assertEqual(state["match"]["ambiguous_labels"], ["only in the network", "only  in the network", "tr|X1 "])
        self.assertEqual(unpack(state["viewer_node"], "<i4").tolist(), [-1, -1, 5, -1])
        self.assertEqual((state["match"]["matched"], state["match"]["ambiguous"]), (1, 3))

    def test_the_names_the_page_shows_are_text_with_no_control_characters(self):
        self.upload(name="line\nbreak\u0000.nwk", tree_name="a\tb")
        info = tree_backend.tree_state(self.viewer)["uploaded"]
        self.assertEqual((info["file"], info["tree_name"]), ("line break .nwk", "a b"))

    def test_a_long_file_name_is_cut(self):
        self.upload(name="x" * 1000)
        self.assertEqual(len(tree_backend.tree_state(self.viewer)["uploaded"]["file"]), tree_backend.MAX_TEXT)


class RouteTests(unittest.TestCase):
    """The route the page posts to, the action beside it, and the web server's rules for both."""

    def setUp(self):
        self.viewer = server_viewer()
        self.viewer.full_headers = list(HEADERS)
        tree_backend.register_backend(self.viewer.web_plugin_registry, self.viewer)
        self.registry = self.viewer.web_plugin_registry

    def post(self, data):
        body = json.dumps(data).encode("utf-8")
        request = ApiRequest("POST", tree_backend.UPLOAD_ROUTE, "", {}, body)
        return self.registry.api_routes[("POST", tree_backend.UPLOAD_ROUTE)].handler(request)

    def test_the_action_and_the_route_are_registered_with_the_servers_limits(self):
        self.assertIn("tree_upload", self.registry.actions)
        route = self.registry.api_routes[("POST", tree_backend.UPLOAD_ROUTE)]
        self.assertEqual(route.max_body, Web_Server.MAX_ACTION_BYTES)
        self.assertEqual(route.content_types, ("application/json",))

    def test_the_action_does_what_the_handler_does(self):
        data = payload(HEADERS[:5])
        result = self.registry.actions["tree_upload"](data)
        self.assertEqual((result["ok"], result["changed"]), (True, True))
        self.assertEqual(self.viewer._tree_source, "uploaded")
        refused = self.registry.actions["tree_upload"]({**data, "parent": [0]})
        self.assertFalse(refused["ok"])

    def test_the_route_answers_with_the_tree_it_made_the_pages(self):
        status, reply = self.post(payload(HEADERS[:5], name="mine.nwk"))
        self.assertEqual(status, 200)
        self.assertTrue(reply["ok"])
        self.assertEqual((reply["state"]["status"], reply["state"]["source"]), ("ok", "uploaded"))
        self.assertEqual(reply["state"]["labels"], HEADERS[:5])
        self.assertEqual(reply["match"]["matched"], 5)

    def test_the_route_answers_a_refusal_as_data_and_keeps_nothing(self):
        status, reply = self.post(payload(["a", "a", "b"]))
        self.assertEqual(status, 200)
        self.assertEqual((reply["ok"], reply["note"]), (False, "duplicate_label"))
        self.assertNotIn("state", reply)
        self.assertFalse(hasattr(self.viewer, "_tree_upload"))
        # A body that is no JSON is a refusal too.
        request = ApiRequest("POST", tree_backend.UPLOAD_ROUTE, "", {}, b"\xff\xfe not json")
        status, reply = self.registry.api_routes[("POST", tree_backend.UPLOAD_ROUTE)].handler(request)
        self.assertEqual((status, reply["ok"], reply["note"]), (200, False, "bad_request"))

    def test_the_web_server_applies_its_own_rules(self):
        server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, server)
        port = server.server_address[1]
        body = json.dumps(payload(HEADERS[:5])).encode("utf-8")

        def post(headers, data=body):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                connection.request("POST", tree_backend.UPLOAD_ROUTE, body=data, headers={"Host": f"127.0.0.1:{port}", **headers})
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        same = {**page_headers(server), "Content-Type": "application/json"}
        status, answer = post(same)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(answer)["ok"])
        self.assertEqual(self.viewer._tree_source, "uploaded")
        self.viewer._tree_source = "guide"
        # A page of another origin, a request that is not of the page, and a body of another type or none change nothing.
        for headers in ({**same, "Origin": "http://evil.example"}, {**same, "Sec-Fetch-Site": "cross-site"}):
            status, _ = post(headers)
            self.assertEqual(status, 403)
        self.assertEqual(post({**page_headers(server), "Content-Type": "text/plain"})[0], 415)
        self.assertEqual(post(same, data=b"")[0], 413)
        self.assertEqual(self.viewer._tree_source, "guide")


if __name__ == "__main__":
    unittest.main()
