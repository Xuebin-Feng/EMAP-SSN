# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View's backend (src/web_ui/tree_backend.py): which tree the page is sent, how it fits, and how it is told.

The tree is the one the Embedding MSA tool saved for the alignment the Viewer loaded, found from the
alignment's name alone: the page never names a file. The tests write real tree files with the tool's own
writer (tests/tree_fixtures.py). A mismatch between the tree and the alignment or the network is a warning the
page shows, never an error; the page's own words are the page's, so the backend answers a status, counts and
warning codes.
"""

import base64
import datetime
import json
import os
import pathlib
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for folder in (SRC_DIR, PROJECT_ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tests.tree_fixtures import (  # noqa: E402
    ALIGNMENT_NAME,
    SMALL_TREE,
    TREE_NAME,
    headers_of,
    give_edges,
    http_get,
    make_alignment_folder,
    make_viewer,
    node_colors,
    page_headers,
    random_linkage,
    save_guide_tree,
    server_viewer,
    snapshot_viewer,
    upload_tree,
    writer,
)
from web_ui import tree_backend  # noqa: E402


def unpack(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype=dtype)


class FolderTestCase(unittest.TestCase):
    """A test with a temporary folder in self.folder, and a clean tree cache."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = pathlib.Path(folder.name)
        patcher = mock.patch.dict(tree_backend._TREE_CACHE, {"key": None, "tree": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def viewer_for(self, headers, network=None, **options):
        """A Viewer over an alignment folder holding the tree of headers; the network is headers unless given."""
        msa, tree = make_alignment_folder(self.folder, headers, **options)
        return make_viewer(msa, headers if network is None else network), msa, tree


class WhichTreeTests(FolderTestCase):
    """What the Viewer says when there is a tree, and when there is not and why."""

    def test_the_tree_of_the_loaded_alignment_is_sent_whole(self):
        headers = headers_of(40)
        viewer, msa, tree = self.viewer_for(headers)
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["alignment"], ALIGNMENT_NAME)
        self.assertEqual(state["tree_path"], os.path.abspath(tree))
        self.assertEqual(state["name"], "guide_tree")
        self.assertIs(state["rooted"], True)
        self.assertIs(state["has_lengths"], True)
        self.assertEqual((state["counts"]["leaves"], state["counts"]["nodes"]), (40, 79))
        self.assertEqual(sorted(state["labels"]), sorted(headers))
        self.assertEqual(state["warnings"], [])
        self.assertEqual(state["notes"]["ALIGNMENT"], ALIGNMENT_NAME)

    def test_the_arrays_are_the_trees_in_the_wire_format_the_page_reads(self):
        headers = headers_of(25)
        viewer, _msa, _tree = self.viewer_for(headers)
        state = tree_backend.tree_state(viewer)
        parent = unpack(state["parent"], "<i4")
        length = unpack(state["length"], "<f4")
        support = unpack(state["support"], "i1")
        self.assertEqual((len(parent), len(length), len(support)), (49, 49, 49))
        # Post-order: every parent after its child, the root last.
        self.assertTrue(np.all(parent[:-1] > np.arange(48)))
        self.assertEqual(int(parent[-1]), -1)
        self.assertTrue(np.all(length >= 0))
        self.assertTrue(np.all(support == -1))
        viewer_node = unpack(state["viewer_node"], "<i4")
        self.assertEqual(len(viewer_node), 25)
        # Each leaf names the node of the network that has its header.
        for label, node in zip(state["labels"], viewer_node):
            self.assertEqual(headers[node], label)

    def test_leaves_are_matched_to_network_nodes_by_header_in_any_order(self):
        headers = headers_of(30)
        network = list(reversed(headers))
        viewer, _msa, _tree = self.viewer_for(headers, network=network)
        state = tree_backend.tree_state(viewer)
        for label, node in zip(state["labels"], unpack(state["viewer_node"], "<i4")):
            self.assertEqual(network[node], label)

    def test_split_support_labels_are_sent(self):
        headers = headers_of(30)
        tool = writer()
        keys = tool.guide_tree_leaf_keys(30, 42)
        first = random_linkage(30, seed=1)
        replicate_keys = np.sort(np.concatenate([tool.replicate_split_keys(first, 30, keys)] * 2))
        viewer, _msa, _tree = self.viewer_for(
            headers, linkage=first, leaf_keys=keys, replicate_keys=replicate_keys, num_trees=2)
        support = unpack(tree_backend.tree_state(viewer)["support"], "i1")
        self.assertTrue(np.any(support == 100))
        self.assertTrue(np.all((support == -1) | ((support >= 0) & (support <= 100))))

    def test_no_alignment_loaded_has_no_tree(self):
        self.assertEqual(tree_backend.guide_state(SimpleNamespace(alignment=None)), {"status": "no_alignment"})
        self.assertEqual(tree_backend.guide_state(SimpleNamespace()), {"status": "no_alignment"})
        for msa in (None, "", "   "):
            with self.subTest(msa=msa):
                self.assertEqual(tree_backend.guide_state(make_viewer(msa, ["a"])), {"status": "no_alignment"})

    def test_an_alignment_that_failed_to_load_is_not_one(self):
        # The tree follows the alignment the Viewer holds: a failed one holds nothing, though its path is set.
        headers = headers_of(10)
        msa, _tree = make_alignment_folder(self.folder, headers)
        self.assertEqual(tree_backend.tree_state(make_viewer(msa, headers, loaded=False))["status"], "no_alignment")

    def test_an_alignment_the_tool_did_not_name_has_no_tree_and_the_page_is_told_its_name(self):
        for name in ("mafft_result.fasta", "Set_[model]_alignment_edited.fasta", "Set_[model]_alignment.fa"):
            with self.subTest(name=name):
                msa = self.folder / name
                msa.write_text(">x\nA\n", encoding="utf-8")
                self.assertEqual(tree_backend.guide_state(make_viewer(msa, ["a"])),
                                 {"status": "unnamed_alignment", "alignment": name})

    def test_the_names_a_tree_is_not_guessed_from_are_not_tried(self):
        # A tree beside the alignment, or one named after the stem, is not the tool's: no guessing.
        headers = headers_of(10)
        msa = self.folder / "mafft_result.fasta"
        msa.write_text(">x\nA\n", encoding="utf-8")
        make_alignment_folder(self.folder, headers, alignment="other_alignment.fasta", tree="mafft_result_guide_tree.nex")
        self.assertEqual(tree_backend.tree_state(make_viewer(msa, headers))["status"], "unnamed_alignment")

    def test_a_missing_tree_says_which_file_it_looked_for(self):
        headers = headers_of(10)
        viewer, msa, _tree = self.viewer_for(headers, tree=None)
        state = tree_backend.guide_state(viewer)
        self.assertEqual(state, {
            "status": "no_tree",
            "alignment": ALIGNMENT_NAME,
            "tree_path": os.path.abspath(self.folder / "Guide_Trees" / TREE_NAME),
        })

    def test_a_tree_beside_the_alignment_instead_of_in_guide_trees_is_not_taken(self):
        headers = headers_of(10)
        viewer, _msa, tree = self.viewer_for(headers)
        tree.rename(self.folder / TREE_NAME)
        self.assertEqual(tree_backend.tree_state(viewer)["status"], "no_tree")

    def test_a_sparse_alignment_has_the_tree_of_its_fasta(self):
        headers = headers_of(10)
        _viewer, msa, tree = self.viewer_for(headers)
        sparse = self.folder / "Set_[model]_alignment_sparse.h5"
        sparse.write_bytes(b"")
        state = tree_backend.tree_state(make_viewer(sparse, headers))
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["tree_path"], os.path.abspath(tree))
        # The tree names the FASTA it was made for; the sparse file stands for it, so no warning.
        self.assertEqual(state["warnings"], [])

    def test_a_file_that_is_not_a_tree_is_reported_with_the_readers_reason(self):
        headers = headers_of(10)
        viewer, _msa, tree = self.viewer_for(headers)
        tree.write_text("#NEXUS\nBEGIN TREES;\n  TREE t = ((a,b);\nEND;\n", encoding="utf-8")
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "unreadable")
        self.assertEqual(state["tree_path"], os.path.abspath(tree))
        self.assertIn("(", state["reason"])
        self.assertNotIn("labels", state)

    def test_a_folder_where_the_tree_should_be_is_unreadable_not_a_crash(self):
        headers = headers_of(10)
        viewer, _msa, tree = self.viewer_for(headers, tree=None)
        (self.folder / "Guide_Trees" / TREE_NAME).mkdir(parents=True)
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "unreadable")
        self.assertTrue(state["reason"])

    def test_a_tree_none_of_whose_leaves_is_in_the_network_is_not_drawn(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(20), network=headers_of(20, prefix="tr|Z"))
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "no_overlap")
        self.assertEqual(state["counts"]["leaves"], 20)
        self.assertEqual(state["counts"]["leaves_in_network"], 0)
        self.assertEqual(state["counts"]["network_nodes_not_in_tree"], 20)
        self.assertNotIn("parent", state)

    def test_a_tree_is_never_read_by_a_path_the_page_supplies(self):
        # The route takes no parameter: a query naming a file changes nothing.
        headers = headers_of(10)
        viewer, _msa, _tree = self.viewer_for(headers)
        request = SimpleNamespace(method="GET", path=tree_backend.DATA_ROUTE, query="path=C:/Windows/win.ini&file=x", headers={}, body=b"")
        status, payload = tree_backend.handle_tree_data(viewer, request)
        self.assertEqual((status, payload["status"]), (200, "ok"))


class LocateTreeTests(FolderTestCase):
    """Where the tree should be, which the tree command says without parsing it on the Viewer's thread."""

    def test_each_place_the_tree_may_or_may_not_be(self):
        headers = headers_of(10)
        self.assertEqual(tree_backend.locate_tree(SimpleNamespace(alignment=None)), {"status": "no_alignment"})
        unnamed = self.folder / "mafft_result.fasta"
        unnamed.write_text(">x\nA\n", encoding="utf-8")
        self.assertEqual(tree_backend.locate_tree(make_viewer(unnamed, headers)),
                         {"status": "unnamed_alignment", "alignment": "mafft_result.fasta"})
        viewer, _msa, tree = self.viewer_for(headers, tree=None)
        expected = os.path.abspath(self.folder / "Guide_Trees" / TREE_NAME)
        self.assertEqual(tree_backend.locate_tree(viewer),
                         {"status": "no_tree", "alignment": ALIGNMENT_NAME, "tree_path": expected})
        save_guide_tree(expected, headers)
        self.assertEqual(tree_backend.locate_tree(viewer),
                         {"status": "present", "alignment": ALIGNMENT_NAME, "tree_path": expected})

    def test_whatever_is_there_is_present_and_nothing_is_read(self):
        headers = headers_of(10)
        viewer, _msa, tree = self.viewer_for(headers)
        tree.write_text("not a tree", encoding="utf-8")
        with mock.patch.object(tree_backend, "read_guide_tree", side_effect=AssertionError("read the file")):
            self.assertEqual(tree_backend.locate_tree(viewer)["status"], "present")
        # tree_state is what finds out that it is no tree.
        self.assertEqual(tree_backend.tree_state(viewer)["status"], "unreadable")

    def test_an_alignment_that_failed_to_load_has_no_place_for_a_tree(self):
        headers = headers_of(10)
        msa, _tree = make_alignment_folder(self.folder, headers)
        self.assertEqual(tree_backend.locate_tree(make_viewer(msa, headers, loaded=False)), {"status": "no_alignment"})


class HowTheTreeFitsTests(FolderTestCase):
    """The warnings: a tree made for another alignment, an older alignment, leaves the network lacks, and so on."""

    def codes(self, state):
        return [warning["code"] for warning in state["warnings"]]

    def test_a_tree_that_fits_has_no_warning(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(20))
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["warnings"], [])
        self.assertEqual(state["counts"], {
            "leaves": 20, "nodes": 39, "leaves_in_network": 20, "leaves_not_in_network": 0,
            "network_nodes": 20, "network_nodes_not_in_tree": 0, "leaves_not_in_alignment": 0,
        })

    def test_a_tree_made_for_another_alignment_is_drawn_with_a_warning(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(20), notes=[("ALIGNMENT", "Other_[model]_alignment.fasta")])
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["warnings"], [
            {"code": "alignment_differs", "tree": "Other_[model]_alignment.fasta", "loaded": ALIGNMENT_NAME}])

    def test_the_alignment_note_is_compared_without_regard_to_case(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(10), notes=[("ALIGNMENT", ALIGNMENT_NAME.upper())])
        self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [])

    def test_a_tree_without_the_alignment_note_is_checked_by_its_set_and_model(self):
        headers = headers_of(10)
        viewer, _msa, _tree = self.viewer_for(headers, notes=[("SEQUENCE_SET", "Set"), ("MODEL", "model")])
        self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [])
        viewer, _msa, _tree = self.viewer_for(headers, notes=[("SEQUENCE_SET", "Set"), ("MODEL", "esm2")])
        self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [
            {"code": "name_differs", "tree": "Set_[esm2]", "loaded": "Set_[model]"}])

    def test_a_tree_with_no_notes_at_all_is_taken_as_it_is(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(10), notes=[])
        state = tree_backend.tree_state(viewer)
        self.assertEqual((state["status"], state["warnings"]), ("ok", []))

    def written_note(self, seconds):
        return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(seconds))

    def test_an_alignment_older_than_its_tree_is_warned_about(self):
        now = time.time()
        viewer, msa, _tree = self.viewer_for(
            headers_of(10), notes=[("ALIGNMENT", ALIGNMENT_NAME), ("WRITTEN", self.written_note(now))])
        os.utime(msa, (now - 3600, now - 3600))
        self.assertEqual(self.codes(tree_backend.tree_state(viewer)), ["alignment_older"])

    def test_an_alignment_as_new_as_its_tree_or_newer_is_not(self):
        now = time.time()
        viewer, msa, _tree = self.viewer_for(
            headers_of(10), notes=[("ALIGNMENT", ALIGNMENT_NAME), ("WRITTEN", self.written_note(now - 7200))])
        for modified in (now, now - 7200, now - 7200 + 1.5):
            with self.subTest(modified=modified - now):
                os.utime(msa, (modified, modified))
                tree_backend._TREE_CACHE.update(key=None, tree=None)
                self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [])

    def test_a_written_time_that_cannot_be_read_gives_no_warning(self):
        viewer, msa, _tree = self.viewer_for(
            headers_of(10), notes=[("ALIGNMENT", ALIGNMENT_NAME), ("WRITTEN", "yesterday afternoon")])
        os.utime(msa, (1, 1))
        self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [])

    def test_the_written_notes_forms(self):
        local = time.mktime((2026, 11, 5, 14, 3, 22, 0, 0, -1))
        self.assertAlmostEqual(tree_backend.written_time("2026-11-05 14:03:22"), local, delta=1)
        east = datetime.datetime(2026, 11, 5, 19, 3, 22, tzinfo=datetime.timezone.utc).timestamp()
        self.assertAlmostEqual(tree_backend.written_time("2026-11-05 14:03:22 -0500"), east, delta=1)
        # A zone written as a name, as some systems write it, leaves the time of day.
        self.assertAlmostEqual(tree_backend.written_time("2026-11-05 14:03:22 Eastern Standard Time"), local, delta=1)
        for text in (None, 5, "", "never", "2026-13-45 99:99:99"):
            self.assertIsNone(tree_backend.written_time(text))

    def test_leaves_that_are_not_nodes_of_the_network_are_counted_and_warned_about(self):
        headers = headers_of(30)
        network = headers[:20] + headers_of(5, prefix="tr|Y")
        viewer, _msa, _tree = self.viewer_for(headers, network=network)
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "ok")
        counts = state["counts"]
        self.assertEqual((counts["leaves"], counts["leaves_in_network"], counts["leaves_not_in_network"]), (30, 20, 10))
        self.assertEqual((counts["network_nodes"], counts["network_nodes_not_in_tree"]), (25, 5))
        self.assertEqual(state["warnings"], [
            {"code": "leaves_not_in_network", "count": 10}, {"code": "network_nodes_not_in_tree", "count": 5}])
        viewer_node = unpack(state["viewer_node"], "<i4")
        self.assertEqual(int(np.count_nonzero(viewer_node < 0)), 10)

    def test_leaves_whose_sequence_the_loaded_alignment_lacks_are_warned_about(self):
        headers = headers_of(20)
        msa, _tree = make_alignment_folder(self.folder, headers)
        aligned = [True] * 15 + [False] * 5
        state = tree_backend.tree_state(make_viewer(msa, headers, aligned=aligned))
        self.assertEqual(state["counts"]["leaves_not_in_alignment"], 5)
        self.assertEqual(state["warnings"], [{"code": "leaves_not_in_alignment", "count": 5}])

    def test_the_notes_sent_are_bounded(self):
        notes = [(f"N{i}", "x" * 5000) for i in range(100)]
        viewer, _msa, _tree = self.viewer_for(headers_of(10), notes=notes)
        sent = tree_backend.tree_state(viewer)["notes"]
        self.assertLessEqual(len(sent), tree_backend.MAX_NOTES)
        self.assertTrue(all(len(value) <= tree_backend.MAX_NOTE_VALUE for value in sent.values()))


class CacheTests(FolderTestCase):
    def test_the_file_is_parsed_once_until_it_changes(self):
        headers = headers_of(20)
        viewer, msa, tree = self.viewer_for(headers)
        with mock.patch.object(tree_backend, "read_guide_tree", wraps=tree_backend.read_guide_tree) as read:
            first = tree_backend.tree_state(viewer)
            again = tree_backend.tree_state(viewer)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(first["labels"], again["labels"])
            # The tool ran again: a new file (another size and time) is read afresh.
            make_alignment_folder(self.folder, headers_of(25))
            changed = tree_backend.tree_state(make_viewer(msa, headers_of(25)))
            self.assertEqual(read.call_count, 2)
            self.assertEqual(changed["counts"]["leaves"], 25)

    def test_how_the_tree_fits_is_worked_out_afresh_from_the_cached_tree(self):
        headers = headers_of(20)
        viewer, msa, _tree = self.viewer_for(headers)
        self.assertEqual(tree_backend.tree_state(viewer)["warnings"], [])
        smaller = make_viewer(msa, headers[:10])
        state = tree_backend.tree_state(smaller)
        self.assertEqual([warning["code"] for warning in state["warnings"]], ["leaves_not_in_network"])


class RouteTests(FolderTestCase):
    """The route over the real web server, with the same-origin guard in front of it."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from web_ui import Web_Server

        self.http_get, self.page_headers = http_get, page_headers
        sessions = tempfile.TemporaryDirectory()
        self.addCleanup(sessions.cleanup)
        environment = mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": sessions.name})
        environment.start()
        self.addCleanup(environment.stop)
        headers = headers_of(30)
        msa, _tree = make_alignment_folder(self.folder, headers)
        self.viewer = server_viewer()
        fixture = make_viewer(msa, headers)
        self.viewer.alignment, self.viewer.full_headers, self.viewer.n_nodes = fixture.alignment, fixture.full_headers, 30
        tree_backend.register_backend(self.viewer.web_plugin_registry, self.viewer)
        self.server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)

    def test_the_page_gets_the_trees_state_as_json(self):
        import json

        status, headers, body = self.http_get(self.server, tree_backend.DATA_ROUTE, self.page_headers(self.server))
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        payload = json.loads(body)
        self.assertEqual((payload["status"], payload["counts"]["leaves"]), ("ok", 30))

    def test_the_source_is_chosen_in_the_query_and_a_bad_one_is_a_bad_request(self):
        import json

        give_edges(self.viewer, [(n, n + 1) for n in range(29)], np.arange(29, 0, -1))
        route = tree_backend.DATA_ROUTE
        status, _headers, body = self.http_get(self.server, route + "?source=threshold", self.page_headers(self.server))
        payload = json.loads(body)
        self.assertEqual((status, payload["status"], payload["source"]), (200, "ok", "threshold"))
        status, _headers, body = self.http_get(self.server, route, self.page_headers(self.server))
        self.assertEqual(json.loads(body)["source"], "threshold")
        for query in ("?source=upload", "?source=", "?source=guide%00"):
            with self.subTest(query=query):
                status, _headers, body = self.http_get(self.server, route + query, self.page_headers(self.server))
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))
        status, _headers, body = self.http_get(self.server, route + "?source=guide", self.page_headers(self.server))
        self.assertEqual(json.loads(body)["source"], "guide")

    def test_a_request_from_another_origin_is_refused(self):
        status, _headers, _body = self.http_get(
            self.server, tree_backend.DATA_ROUTE, {**self.page_headers(self.server), "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_the_route_answers_get_only(self):
        # POST is not registered for it: the server falls through to its own routes, which do not know it.
        route = self.viewer.web_plugin_registry.api_routes
        self.assertEqual([key for key in route if key[1] == tree_backend.DATA_ROUTE], [("GET", tree_backend.DATA_ROUTE)])

    def mismatch(self, query, **extra):
        return self.http_get(self.server, tree_backend.MISMATCH_ROUTE + query, {**self.page_headers(self.server), **extra})

    def test_the_mismatch_names_come_as_json_for_the_source_asked(self):
        import json

        self.viewer.full_headers = self.viewer.full_headers[:25]
        status, headers, body = self.mismatch("?source=guide")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        payload = json.loads(body)
        self.assertEqual(payload["leaves_not_in_network"]["count"], 5)
        self.assertEqual(payload["network_nodes_not_in_tree"], {"count": 0, "names": []})

    def test_a_missing_repeated_or_unknown_mismatch_source_is_a_bad_request(self):
        import json

        for query in ("", "?source=", "?source=upload", "?source=guide&source=guide", "?file=C:/Windows/win.ini", "?source=guide%00"):
            with self.subTest(query=query):
                status, _headers, body = self.mismatch(query)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(body))

    def test_a_mismatch_request_from_another_origin_is_refused_and_only_get_is_answered(self):
        status, _headers, _body = self.mismatch("?source=guide", Origin="http://evil.example")
        self.assertEqual(status, 403)
        keys = [key for key in self.viewer.web_plugin_registry.api_routes if key[1] == tree_backend.MISMATCH_ROUTE]
        self.assertEqual(keys, [("GET", tree_backend.MISMATCH_ROUTE)])

    def give_colors(self):
        from tests.tree_fixtures import node_colors

        self.viewer.current_colors = node_colors(30, {2: "#ff0000"})
        self.viewer.visible_mask = np.ones(30, dtype=bool)
        self.viewer.cluster_labels = np.arange(30, dtype=np.int32) % 3

    def test_the_page_gets_the_colours_as_json(self):
        import json

        self.give_colors()
        status, headers, body = self.http_get(self.server, tree_backend.COLORS_ROUTE, self.page_headers(self.server))
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        payload = json.loads(body)
        self.assertEqual((payload["status"], payload["count"]), ("ok", 30))
        self.assertEqual(list(unpack(payload["rgba"], np.uint8).reshape(-1, 4)[2]), [255, 0, 0, 255])
        self.assertEqual(payload["version"], tree_backend.colors_snapshot(self.viewer)["version"])

    def test_a_viewer_with_no_colours_answers_no_colours(self):
        import json

        status, _headers, body = self.http_get(self.server, tree_backend.COLORS_ROUTE, self.page_headers(self.server))
        answer = json.loads(body)
        self.assertEqual((status, answer["status"], answer["version"]), (200, "no_colors", 0))
        self.assertEqual(answer["marks"]["selected"], [0xFF, 0xAA, 0x00, 255])

    def test_a_request_for_the_colours_from_another_origin_is_refused(self):
        self.give_colors()
        status, _headers, _body = self.http_get(
            self.server, tree_backend.COLORS_ROUTE, {**self.page_headers(self.server), "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_the_colours_route_answers_get_only(self):
        route = self.viewer.web_plugin_registry.api_routes
        self.assertEqual([key for key in route if key[1] == tree_backend.COLORS_ROUTE], [("GET", tree_backend.COLORS_ROUTE)])

    def test_the_page_gets_the_selection_as_json(self):
        import json

        self.viewer.selected_indices = [9, 2, 2]
        self.viewer.selected_node_idx = 5
        status, headers, body = self.http_get(self.server, tree_backend.SELECTION_ROUTE, self.page_headers(self.server))
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("application/json"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        payload = json.loads(body)
        self.assertEqual((payload["status"], payload["count"], payload["focus"]), ("ok", 2, 5))
        self.assertEqual(list(unpack(payload["indices"], "<i4")), [2, 9])

    def test_a_request_for_the_selection_from_another_origin_is_refused(self):
        status, _headers, _body = self.http_get(
            self.server, tree_backend.SELECTION_ROUTE, {**self.page_headers(self.server), "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_the_selection_route_answers_get_only(self):
        route = self.viewer.web_plugin_registry.api_routes
        self.assertEqual([key for key in route if key[1] == tree_backend.SELECTION_ROUTE], [("GET", tree_backend.SELECTION_ROUTE)])

    def test_what_the_page_was_sent_is_not_announced_to_it(self):
        self.give_colors()
        _status, _headers, body = self.http_get(self.server, tree_backend.COLORS_ROUTE, self.page_headers(self.server))
        import json

        watch = self.viewer._tree_color_watch
        self.assertEqual((watch.latest, watch.announced), (json.loads(body)["version"],) * 2)


class ColorStateTests(unittest.TestCase):
    """What the page is told of the nodes' colours: their bytes, which are hidden, their clusters, and a version that names it."""

    def viewer(self, count=10, painted=None, hidden=(), clusters=None):
        return make_viewer(None, headers_of(count), colors=node_colors(count, painted), hidden=hidden, clusters=clusters)

    def test_every_node_has_its_colour_as_four_bytes(self):
        state = tree_backend.colors_state(self.viewer(painted={1: "#ff0000", 4: "#00ff80"}))
        rgba = unpack(state["rgba"], np.uint8).reshape(-1, 4)
        self.assertEqual((state["status"], state["count"], rgba.shape), ("ok", 10, (10, 4)))
        self.assertEqual(list(rgba[1]), [255, 0, 0, 255])
        self.assertEqual(list(rgba[4]), [0, 255, 128, 255])
        # What no command has coloured is the Viewer's initial node colour, which the page is told so as to leave it plain.
        self.assertEqual(state["plain"], [0x44, 0x88, 0xFF, 255])
        self.assertEqual(list(rgba[0]), state["plain"])

    def test_a_colour_is_rounded_to_the_nearest_byte(self):
        viewer = self.viewer(count=2)
        viewer.current_colors[0] = (0.3957, 0.8039, 0.2, 1.0)  # 100.9 and 204.99 of 255
        rgba = unpack(tree_backend.colors_state(viewer)["rgba"], np.uint8).reshape(-1, 4)
        self.assertEqual(list(rgba[0]), [101, 205, 51, 255])

    def test_the_plain_colour_is_the_initial_node_colour_the_viewer_is_set_to(self):
        import EMAPSSN_Config as cfg

        with mock.patch.object(cfg, "INITIAL_NODE_COLOR", "#102030"):
            state = tree_backend.colors_state(self.viewer(count=3))
        self.assertEqual(state["plain"], [0x10, 0x20, 0x30, 255])

    def test_hidden_nodes_are_one_bit_each_least_significant_first(self):
        state = tree_backend.colors_state(self.viewer(count=10, hidden=[2, 9]))
        bits = np.unpackbits(unpack(state["hidden"], np.uint8), bitorder="little")
        self.assertEqual(len(unpack(state["hidden"], np.uint8)), 2)
        self.assertEqual(list(bits[:10]), [0, 0, 1, 0, 0, 0, 0, 0, 0, 1])

    def test_clusters_come_with_the_colours_when_the_viewer_has_them(self):
        state = tree_backend.colors_state(self.viewer(count=5, clusters=[3, 3, -1, 7, 0]))
        self.assertEqual(state["cluster_type"], "i2")
        self.assertEqual(list(unpack(state["cluster"], "<i2")), [3, 3, -1, 7, 0])
        absent = tree_backend.colors_state(self.viewer(count=5))
        self.assertEqual((absent["cluster"], absent["cluster_type"]), (None, None))

    def test_cluster_numbers_that_do_not_fit_two_bytes_take_four(self):
        state = tree_backend.colors_state(self.viewer(count=4, clusters=[0, 40000, -1, 2**31 - 1]))
        self.assertEqual(state["cluster_type"], "i4")
        self.assertEqual(list(unpack(state["cluster"], "<i4")), [0, 40000, -1, 2**31 - 1])
        edge = tree_backend.colors_state(self.viewer(count=3, clusters=[-32768, 32767, 5]))
        self.assertEqual((edge["cluster_type"], list(unpack(edge["cluster"], "<i2"))), ("i2", [-32768, 32767, 5]))

    def test_a_viewer_with_no_colours_says_so(self):
        for viewer in (make_viewer(None, headers_of(4)),
                       SimpleNamespace(n_nodes=4, current_colors=np.zeros((3, 4), np.float32)),
                       SimpleNamespace(n_nodes=4, current_colors=np.zeros((4, 3), np.float32)),
                       SimpleNamespace(n_nodes=0, current_colors=np.zeros((0, 4), np.float32))):
            with self.subTest(viewer=vars(viewer).keys()):
                state = tree_backend.colors_state(viewer)
                # The marks are the Viewer's settings, and are there with no colours too.
                self.assertEqual((state["status"], state["version"], sorted(state["marks"])),
                                 ("no_colors", 0, ["connected", "connected_on", "selected"]))

    def test_a_visible_mask_of_the_wrong_length_hides_nothing(self):
        viewer = self.viewer(count=6)
        viewer.visible_mask = np.array([False, True])
        state = tree_backend.colors_state(viewer)
        self.assertEqual(list(np.unpackbits(unpack(state["hidden"], np.uint8), bitorder="little")[:6]), [0] * 6)

    def test_the_version_names_what_is_sent_and_nothing_else(self):
        viewer = self.viewer(count=20, painted={3: "red"}, clusters=[1] * 20)
        base = tree_backend.colors_state(viewer)["version"]
        self.assertEqual(tree_backend.colors_state(viewer)["version"], base)
        # A colour that differs by less than a byte is no change.
        viewer.current_colors[0, 0] += 1e-4
        self.assertEqual(tree_backend.colors_state(viewer)["version"], base)
        for change in ("colour", "hidden", "cluster"):
            with self.subTest(change=change):
                other = self.viewer(count=20, painted={3: "red"}, clusters=[1] * 20)
                if change == "colour":
                    other.current_colors[7] = (0, 1, 0, 1)
                elif change == "hidden":
                    other.visible_mask[7] = False
                else:
                    other.cluster_labels[7] = 2
                self.assertNotEqual(tree_backend.colors_state(other)["version"], base)
        # The colours the selection is outlined in are sent with them, so a change of those is a change.
        import EMAPSSN_Config as cfg

        for name in ("HOVER_COLOR", "CONNECTED_NODE_COLOR"):
            with self.subTest(setting=name), mock.patch.object(cfg, name, "#123456"):
                self.assertNotEqual(tree_backend.colors_state(viewer)["version"], base)

    def test_the_state_is_a_copy_of_what_the_viewer_held(self):
        viewer = self.viewer(count=4)
        snapshot = tree_backend.colors_snapshot(viewer)
        viewer.current_colors[:] = 0
        viewer.visible_mask[:] = False
        self.assertEqual(int(snapshot["rgba"][0, 0]), 0x44)
        self.assertEqual(int(np.unpackbits(snapshot["hidden"], bitorder="little")[:4].sum()), 0)

    def test_the_colours_of_forty_four_thousand_nodes_are_a_quarter_of_a_megabyte(self):
        import json

        count = 44000
        viewer = make_viewer(None, ["n%d" % i for i in range(count)], colors=node_colors(count), clusters=np.arange(count) % 200)
        state = tree_backend.colors_state(viewer)
        self.assertEqual(len(unpack(state["rgba"], np.uint8)), 4 * count)
        self.assertLess(len(json.dumps(state)), 400_000)


class ColorKeyPayloadTests(unittest.TestCase):
    """What the page is told of what the colours mean: the colour key's layers and its per-node arrays."""

    @staticmethod
    def keyed(count=12, title="Clusters", labels=("Cluster 1", "Cluster 2")):
        from desktop import Node_Colors

        def declare(viewer):
            red, blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
            viewer.current_colors[0:4] = red
            viewer.current_colors[4:6] = blue
            Node_Colors.declare_categories(viewer, title, [(labels[0], red), (labels[1], blue)], [0, 0, 0, 0, 1, 1], nodes=range(6))

        return make_viewer(None, headers_of(count), colors=node_colors(count), key=declare)

    def test_the_key_travels_with_the_colours_as_layers_and_arrays(self):
        state = tree_backend.colors_state(self.keyed())
        key = state["key"]
        (layer,) = key["layers"]
        self.assertEqual((layer["kind"], layer["title"]), ("categories", "Clusters"))
        self.assertEqual([(c["label"], c["rgba"]) for c in layer["categories"]],
                         [("Cluster 1", [255, 0, 0, 255]), ("Cluster 2", [0, 0, 255, 255])])
        self.assertEqual((key["owner_type"], key["category_type"]), ("i2", "i2"))
        self.assertEqual(list(unpack(key["owner"], "<i2")), [0] * 6 + [-1] * 6)
        self.assertEqual(list(unpack(key["category"], "<i2")), [0, 0, 0, 0, 1, 1] + [-1] * 6)
        bits = np.unpackbits(unpack(key["labelled"], np.uint8), bitorder="little")
        self.assertEqual(list(bits[:12]), [1] * 6 + [0] * 6)
        self.assertIsNone(key["values"])
        json.dumps(state)

    def test_a_node_that_only_changed_alpha_stays_labelled_in_the_payload(self):
        viewer = self.keyed()
        viewer.current_colors[0] = (1.0, 0.0, 0.0, 0.5)  # still red, fainter
        viewer.current_colors[1] = (1.0, 0.0, 0.1, 1.0)  # another color
        bits = np.unpackbits(unpack(tree_backend.colors_state(viewer)["key"]["labelled"], np.uint8), bitorder="little")
        self.assertEqual(list(bits[:6]), [1, 0, 1, 1, 1, 1])

    def test_two_runs_of_an_extendable_layer_are_one_layer_in_the_payload(self):
        from desktop import Node_Colors

        def declare(viewer):
            red, blue = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
            for nodes, color, label in (([0, 1, 2], red, "a → red"), ([3, 4], blue, "b → blue"), ([5], red, "a → red")):
                viewer.current_colors[nodes] = color
                Node_Colors.declare_categories(viewer, "Colors", [(label, color)], [0] * len(nodes), nodes=nodes,
                                               extend=True)

        key = tree_backend.colors_state(make_viewer(None, headers_of(8), colors=node_colors(8), key=declare))["key"]
        (layer,) = key["layers"]
        self.assertEqual(layer["title"], "Colors")
        self.assertEqual([c["label"] for c in layer["categories"]], ["a → red", "b → blue"])
        self.assertEqual(list(unpack(key["owner"], "<i2")), [0] * 6 + [-1] * 2)
        self.assertEqual(list(unpack(key["category"], "<i2")), [0, 0, 0, 1, 1, 0, -1, -1])

    def test_a_viewer_with_no_key_sends_none(self):
        state = tree_backend.colors_state(make_viewer(None, headers_of(5), colors=node_colors(5)))
        self.assertEqual((state["status"], state["key"]), ("ok", None))

    def test_a_gradient_sends_its_values_and_the_layer_ids_become_positions(self):
        import matplotlib as mpl

        from desktop import Node_Colors

        def declare(viewer):
            cmap = mpl.colormaps["viridis"]
            viewer.current_colors[0:3] = cmap([0.0, 0.5, 1.0])
            Node_Colors.declare_categories(viewer, "First", [("A", tuple(viewer.current_colors[3]))], [0], nodes=[3])
            Node_Colors.declare_gradient(viewer, "Length", "viridis", cmap, 10.0, 20.0, [0, 1, 2], [10.0, 15.0, 20.0],
                                         extras=[("No value", (0.8, 0.8, 0.8, 0.4), [4])])

        viewer = make_viewer(None, headers_of(6), colors=node_colors(6), key=declare)
        viewer.current_colors[4] = (0.8, 0.8, 0.8, 0.4)
        key = tree_backend.colors_state(viewer)["key"]
        self.assertEqual([layer["kind"] for layer in key["layers"]], ["categories", "gradient"])
        gradient = key["layers"][1]
        self.assertEqual((gradient["scheme"], gradient["min"], gradient["max"], len(gradient["stops"])), ("viridis", 10.0, 20.0, 16))
        self.assertEqual(gradient["extras"][0]["label"], "No value")
        self.assertEqual(list(unpack(key["owner"], "<i2")), [1, 1, 1, 0, 1, -1])
        self.assertEqual(list(unpack(key["category"], "<i2")), [-1, -1, -1, 0, 0, -1])
        values = unpack(key["values"], "<f4")
        self.assertEqual(list(values[:3]), [10.0, 15.0, 20.0])
        self.assertTrue(np.isnan(values[3:]).all())

    def test_a_key_that_cannot_be_read_is_no_key_and_never_stops_the_colours(self):
        viewer = self.keyed()
        viewer.color_key_owner = np.zeros(3, dtype=np.int32)
        state = tree_backend.colors_state(viewer)
        self.assertEqual((state["status"], state["key"]), ("ok", None))

    def test_the_version_follows_the_key_alone(self):
        base = tree_backend.colors_state(self.keyed())["version"]
        self.assertEqual(tree_backend.colors_state(self.keyed())["version"], base)
        # The same colours with another meaning: a title, a label, or no key at all.
        self.assertNotEqual(tree_backend.colors_state(self.keyed(title="Subclusters"))["version"], base)
        self.assertNotEqual(tree_backend.colors_state(self.keyed(labels=("Cluster 1", "Noise")))["version"], base)
        unkeyed = self.keyed()
        for name in ("color_key_layers", "color_key_owner", "color_key_category", "color_key_rgba", "color_key_values"):
            setattr(unkeyed, name, None)
        self.assertEqual(np.array_equal(self.keyed().current_colors, unkeyed.current_colors), True)
        self.assertNotEqual(tree_backend.colors_state(unkeyed)["version"], base)
        # Two nodes swapping categories keeps every colour but one.
        swapped = self.keyed()
        swapped.color_key_category[0], swapped.color_key_category[4] = 1, 0
        self.assertNotEqual(tree_backend.colors_snapshot(swapped)["version"], base)

    def test_titles_and_labels_are_kept_to_a_length(self):
        key = tree_backend.colors_state(self.keyed(title="T" * 5000, labels=("L" * 5000, "x")))["key"]
        self.assertEqual(len(key["layers"][0]["title"]), tree_backend.KEY_TEXT_LIMIT)
        self.assertEqual(len(key["layers"][0]["categories"][0]["label"]), tree_backend.KEY_TEXT_LIMIT)

    def test_the_key_of_forty_four_thousand_nodes_with_a_gradient_is_about_half_a_megabyte(self):
        import matplotlib as mpl

        from desktop import Node_Colors

        count = 44000

        def declare(viewer):
            cmap = mpl.colormaps["viridis"]
            values = np.linspace(0.0, 1.0, count)
            viewer.current_colors[:] = cmap(values)
            Node_Colors.declare_gradient(viewer, "Length", "viridis", cmap, 0.0, 1.0, np.arange(count), values)

        viewer = make_viewer(None, ["n%d" % i for i in range(count)], colors=node_colors(count), key=declare)
        key = tree_backend.colors_state(viewer)["key"]
        sizes = {name: len(key[name]) for name in ("owner", "category", "labelled", "values")}
        self.assertEqual(sizes["labelled"], len(base64.b64encode(bytes((count + 7) // 8))))
        self.assertLess(sum(sizes.values()), 500_000)


class MarksTests(unittest.TestCase):
    """The colours the selected nodes and their neighbours are outlined in are the Viewer's settings, read when asked."""

    def test_they_are_the_highlight_and_connected_node_colours_of_the_viewers_settings(self):
        import EMAPSSN_Config as cfg

        marks = tree_backend.marks_state(SimpleNamespace())
        self.assertEqual(marks["selected"], [0xFF, 0xAA, 0x00, 255])
        self.assertEqual(marks["connected"], [0xFF, 0x00, 0x00, 255])
        with mock.patch.object(cfg, "HOVER_COLOR", "#102030"), mock.patch.object(cfg, "CONNECTED_NODE_COLOR", "#405060"):
            marks = tree_backend.marks_state(SimpleNamespace())
        self.assertEqual((marks["selected"], marks["connected"]), ([0x10, 0x20, 0x30, 255], [0x40, 0x50, 0x60, 255]))

    def test_the_colours_answer_comes_with_them_and_follows_the_settings(self):
        import EMAPSSN_Config as cfg

        viewer = make_viewer(None, headers_of(4), colors=node_colors(4))
        self.assertEqual(tree_backend.colors_state(viewer)["marks"]["selected"], [0xFF, 0xAA, 0x00, 255])
        with mock.patch.object(cfg, "HOVER_COLOR", "#0a0b0c"):
            self.assertEqual(tree_backend.colors_state(viewer)["marks"]["selected"], [10, 11, 12, 255])

    def test_connected_nodes_are_identified_by_the_viewers_own_rule(self):
        import EMAPSSN_Config as cfg
        from EMAPSSN_Viewer import MainViewer

        viewer = make_viewer(None, headers_of(4), colors=node_colors(4))
        viewer._connected_node_identification_enabled = MainViewer._connected_node_identification_enabled.__get__(viewer)
        # The boundary colour differs from the connected node colour: on. The same colour: off.
        self.assertTrue(tree_backend.marks_state(viewer)["connected_on"])
        with mock.patch.object(cfg, "NODE_BOUNDARY_COLOR", cfg.CONNECTED_NODE_COLOR):
            self.assertFalse(tree_backend.marks_state(viewer)["connected_on"])
            self.assertEqual(tree_backend.colors_state(viewer)["marks"]["connected_on"], False)
        # A Viewer that cannot say, or whose rule fails, is taken not to.
        self.assertFalse(tree_backend.marks_state(SimpleNamespace())["connected_on"])
        viewer._connected_node_identification_enabled = mock.Mock(side_effect=RuntimeError("no"))
        self.assertFalse(tree_backend.marks_state(viewer)["connected_on"])


def network_viewer(count, edges, selected=()):
    """A real MainViewer with just the network a neighbour query reads: nodes, edges and a selection."""
    from EMAPSSN_Viewer import MainViewer

    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = count
    viewer.edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    viewer.selected_indices = list(selected)
    viewer.selected_node_idx = None
    viewer.visible_mask = np.ones(count, dtype=bool)
    return viewer


class ConnectedNodeTests(unittest.TestCase):
    """The nodes next to the selected ones: what the Viewer outlines in the connected colour, and only those."""

    def test_the_answer_is_the_viewers_own_neighbour_set(self):
        rng = np.random.default_rng(42)
        count = 400
        edges = rng.integers(0, count, size=(1500, 2))
        for selected in ([], [3], [5, 7, 9], list(rng.choice(count, 60, replace=False)), list(range(count))):
            with self.subTest(selected=len(selected)):
                viewer = network_viewer(count, edges, selected)
                nodes = viewer._selected_node_indices()
                expected = viewer._connected_to_selected_indices(nodes) if len(nodes) else np.empty(0)
                found = tree_backend.neighbours_of_nodes(viewer, nodes)
                self.assertEqual(found.dtype, np.int32)
                self.assertEqual(list(found), list(expected))

    def test_the_selected_nodes_are_not_among_their_own_neighbours(self):
        viewer = network_viewer(5, [[0, 1], [1, 2], [2, 3]], [1, 2])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(viewer, [1, 2])), [0, 3])

    def test_a_node_reached_by_many_is_named_once_and_a_node_with_no_edge_has_no_neighbours(self):
        viewer = network_viewer(6, [[0, 5], [1, 5], [2, 5]], [0, 1, 2, 4])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(viewer, [0, 1, 2, 4])), [5])

    def test_reading_in_pieces_gives_the_same_answer(self):
        rng = np.random.default_rng(42)
        edges = rng.integers(0, 300, size=(2000, 2))
        viewer = network_viewer(300, edges)
        nodes = np.arange(300)
        whole = tree_backend.neighbours_of_nodes(viewer, nodes)
        for chunk in (1, 7, 100, 1000):
            with self.subTest(chunk=chunk), mock.patch.object(tree_backend, "NEIGHBOUR_CHUNK", chunk):
                self.assertEqual(list(tree_backend.neighbours_of_nodes(viewer, nodes)), list(whole))

    def test_nodes_out_of_range_and_a_viewer_with_no_index_have_no_neighbours(self):
        viewer = network_viewer(4, [[0, 1]])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(viewer, [-3, 99])), [])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(viewer, [])), [])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(SimpleNamespace(), [0])), [])
        self.assertEqual(list(tree_backend.neighbours_of_nodes(network_viewer(4, []), [0, 1])), [])

    def test_it_leaves_the_viewers_own_neighbour_counts_alone(self):
        # The web server's thread asks; the Viewer's thread keeps its counts as the selection changes.
        viewer = network_viewer(6, [[0, 1], [1, 2], [2, 3]], [0])
        viewer._connected_to_selected_indices(viewer._selected_node_indices())
        tracker = viewer._selection_neighbours
        before = (tracker.touches.copy(), tracker.selected.copy())
        tree_backend.neighbours_of_nodes(viewer, [2, 3, 4])
        self.assertIs(viewer._selection_neighbours, tracker)
        self.assertTrue(np.array_equal(tracker.touches, before[0]) and np.array_equal(tracker.selected, before[1]))

    def selection(self, viewer, on=True):
        import EMAPSSN_Config as cfg
        from EMAPSSN_Viewer import MainViewer

        viewer._connected_node_identification_enabled = MainViewer._connected_node_identification_enabled.__get__(viewer)
        if on:
            return tree_backend.selection_state(viewer)
        with mock.patch.object(cfg, "NODE_BOUNDARY_COLOR", cfg.CONNECTED_NODE_COLOR):
            return tree_backend.selection_state(viewer)

    def test_the_selection_answer_names_the_neighbours_while_identification_is_on(self):
        viewer = network_viewer(6, [[0, 1], [1, 2], [2, 3], [4, 5]], [1])
        state = self.selection(viewer)
        self.assertEqual((state["connected_on"], state["connected_count"]), (True, 2))
        self.assertEqual(list(unpack(state["connected"], "<i4")), [0, 2])
        self.assertEqual(list(unpack(state["indices"], "<i4")), [1])

    def test_with_identification_off_it_names_none_as_the_viewer_outlines_none(self):
        viewer = network_viewer(6, [[0, 1], [1, 2]], [1])
        state = self.selection(viewer, on=False)
        self.assertEqual((state["connected_on"], state["connected_count"], state["connected"]), (False, 0, ""))

    def test_with_nothing_selected_there_are_no_neighbours(self):
        state = self.selection(network_viewer(6, [[0, 1], [1, 2]]))
        self.assertEqual((state["connected_on"], state["count"], state["connected_count"]), (True, 0, 0))

    def test_edges_are_never_sent(self):
        viewer = network_viewer(6, [[0, 1], [1, 2]], [1])
        state = self.selection(viewer)
        self.assertEqual(sorted(state), ["connected", "connected_count", "connected_on", "count", "focus", "indices", "status"])
        # Nodes, four bytes each: the neighbours are as long as their count says.
        self.assertEqual(len(base64.b64decode(state["connected"])), 4 * state["connected_count"])

    def test_a_viewer_that_cannot_name_neighbours_answers_without_them(self):
        state = tree_backend.selection_state(FakeSelectingViewer(count=10, selected=[1, 2]))
        self.assertEqual((state["status"], state["connected_on"], state["connected_count"]), ("ok", False, 0))

    def test_forty_four_thousand_selected_nodes_are_read_in_a_blink(self):
        count = 44000
        rng = np.random.default_rng(42)
        edges = rng.integers(0, count, size=(count * 20, 2))
        viewer = network_viewer(count, edges, range(count))
        viewer._adjacency_index()
        started = time.perf_counter()
        found = tree_backend.neighbours_of_nodes(viewer, np.arange(count))
        elapsed = time.perf_counter() - started
        self.assertEqual(len(found), 0)  # everything selected: nothing is left to be next to
        viewer.selected_indices = list(range(0, count, 2))
        started = time.perf_counter()
        half = tree_backend.neighbours_of_nodes(viewer, np.arange(0, count, 2))
        elapsed = max(elapsed, time.perf_counter() - started)
        self.assertGreater(len(half), count // 3)
        self.assertLess(elapsed, 3.0)


class FakeTimer:
    """A timer that fires when the test says, in the place of threading.Timer."""

    def __init__(self, delay, callback):
        self.delay, self.callback, self.started, self.daemon = delay, callback, False, False

    def start(self):
        self.started = True

    def fire(self):
        self.callback()


class ColorWatchTests(unittest.TestCase):
    """The listener that tells a connected Tree View page the colours changed: once per burst, and only when they did."""

    def setUp(self):
        self.timers = []

        def make_timer(delay, callback):
            self.timers.append(FakeTimer(delay, callback))
            return self.timers[-1]

        self.clients = {"tree"}
        self.viewer = make_viewer(None, headers_of(12), colors=node_colors(12), clusters=[0] * 12)
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: client in self.clients)
        self.viewer.broadcast_event = mock.Mock()
        self.watch = tree_backend.ColorWatch(self.viewer, timer_factory=make_timer)

    def version(self):
        return tree_backend.colors_snapshot(self.viewer)["version"]

    def test_a_change_is_announced_once_with_its_version(self):
        self.watch.served(self.version())
        self.viewer.current_colors[3] = (1, 0, 0, 1)
        self.watch()
        self.assertEqual(len(self.timers), 1)
        self.assertTrue(self.timers[0].started and self.timers[0].daemon)
        self.assertEqual(self.timers[0].delay, tree_backend.COLOR_COALESCE_SECONDS)
        self.viewer.broadcast_event.assert_not_called()
        self.timers[0].fire()
        self.viewer.broadcast_event.assert_called_once_with({"type": "tree_colors_changed", "version": self.version()})

    def test_a_burst_of_changes_is_one_event_with_the_last_version(self):
        self.watch.served(self.version())
        for node in (1, 2, 3):
            self.viewer.current_colors[node] = (node / 10, 0, 0, 1)
            self.watch()
        self.assertEqual(len(self.timers), 1)
        self.timers[0].fire()
        self.viewer.broadcast_event.assert_called_once_with({"type": "tree_colors_changed", "version": self.version()})

    def test_a_change_that_is_undone_before_the_event_announces_nothing(self):
        self.watch.served(self.version())
        before = self.viewer.current_colors[5].copy()
        self.viewer.current_colors[5] = (0, 1, 0, 1)
        self.watch()
        self.viewer.current_colors[5] = before
        self.watch()
        self.timers[0].fire()
        self.viewer.broadcast_event.assert_not_called()

    def test_drawing_the_nodes_again_in_the_same_colours_announces_nothing(self):
        # What the pointer moving over the nodes does: the hover colour is painted on a copy, never in current_colors.
        self.watch.served(self.version())
        for _ in range(5):
            self.watch()
        self.assertEqual(self.timers, [])
        self.viewer.broadcast_event.assert_not_called()

    def test_the_colours_the_page_was_sent_are_not_announced_to_it(self):
        self.viewer.current_colors[2] = (0, 0, 1, 1)
        self.watch.served(self.version())
        self.watch()
        self.assertEqual(self.timers, [])

    def test_hiding_a_node_and_changing_a_cluster_are_changes_too(self):
        self.watch.served(self.version())
        self.viewer.visible_mask[4] = False
        self.watch()
        self.timers[0].fire()
        self.viewer.cluster_labels[4] = 9
        self.watch()
        self.timers[1].fire()
        self.assertEqual(self.viewer.broadcast_event.call_count, 2)

    def test_the_next_change_after_an_event_is_announced_again(self):
        self.watch.served(self.version())
        for node in (1, 2):
            self.viewer.current_colors[node] = (1, 1, 0, 1)
            self.watch()
            self.timers[-1].fire()
        self.assertEqual(self.viewer.broadcast_event.call_count, 2)
        versions = [call.args[0]["version"] for call in self.viewer.broadcast_event.call_args_list]
        self.assertNotEqual(versions[0], versions[1])

    def test_nothing_is_done_while_no_tree_page_is_connected(self):
        self.clients = {"esmfold", "meta"}
        self.viewer.current_colors[1] = (1, 0, 0, 1)
        self.watch()
        self.assertEqual(self.timers, [])
        self.viewer.web_server = None
        self.watch()
        del self.viewer.web_server
        self.watch()
        self.assertEqual(self.timers, [])

    def test_no_colour_is_computed_for_a_viewer_that_nobody_is_watching(self):
        self.clients = set()
        with mock.patch.object(tree_backend, "colors_snapshot", side_effect=AssertionError("worked out for nobody")):
            self.watch()

    def test_the_event_is_small_and_carries_no_colour(self):
        self.watch.served(self.version())
        self.viewer.current_colors[3] = (1, 0, 0, 1)
        self.watch()
        self.timers[0].fire()
        event = self.viewer.broadcast_event.call_args.args[0]
        self.assertEqual(sorted(event), ["type", "version"])


class FollowColorsTests(unittest.TestCase):
    def test_the_viewer_gets_one_listener_however_often_the_backend_is_registered(self):
        from web_ui.Plugin_Manager import WebPluginRegistry

        viewer = SimpleNamespace()
        registry = WebPluginRegistry(viewer)
        for _ in range(3):
            tree_backend.register_backend(registry, viewer)
        self.assertEqual(len(viewer.node_update_listeners), 1)
        self.assertIs(viewer.node_update_listeners[0], viewer._tree_color_watch)
        self.assertEqual(registry.api_routes[("GET", "/api/tree/colors")].plugin_id, "tree")

    def test_listeners_that_are_there_already_are_kept(self):
        mine = lambda: None  # noqa: E731
        viewer = SimpleNamespace(node_update_listeners=[mine])
        tree_backend.follow_colors(viewer)
        self.assertEqual(viewer.node_update_listeners[0], mine)
        self.assertEqual(len(viewer.node_update_listeners), 2)


class FakeSelectingViewer:
    """The little of a Viewer that the selection actions touch, which records what they do to it, in order."""

    def __init__(self, count=10, hidden=(), selected=(), focus=None):
        self.n_nodes = count
        self.visible_mask = np.ones(count, dtype=bool)
        self.visible_mask[list(hidden)] = False
        self.selected_indices = list(selected)
        self.selected_node_idx = focus
        self.calls = []

    def _note_hand_selection(self):
        self.calls.append(("note", list(self.selected_indices)))

    def update_selection_visual(self, broadcast=True):
        self.calls.append(("visual", list(self.selected_indices)))

    def _show_selection_status(self):
        self.calls.append(("status",))

    def apply_left_click_focus(self, index):
        self.selected_node_idx = index
        self.calls.append(("focus", index))

    def clear_left_click_focus(self):
        self.selected_node_idx = None
        self.calls.append(("clear",))


class TreeSelectActionTests(unittest.TestCase):
    """tree_select does what a box selection does, and refuses what is not well formed without touching the selection."""

    def select(self, viewer, indices, mode="replace"):
        return tree_backend.handle_tree_select(viewer, {"action": "tree_select", "indices": indices, "mode": mode})

    def test_replace_makes_the_nodes_the_selection_as_a_box_selection_does(self):
        viewer = FakeSelectingViewer(selected=[1, 2])
        result = self.select(viewer, [5, 3, 4])
        self.assertEqual(result, {"ok": True, "changed": True, "selected": 3, "note": None})
        self.assertEqual(viewer.selected_indices, [3, 4, 5])
        # The selection is noted as made by hand before the nodes are drawn and the pages told, then the status line says it.
        self.assertEqual([call[0] for call in viewer.calls], ["note", "visual", "status"])
        self.assertEqual(viewer.calls[0][1], [3, 4, 5])

    def test_the_mode_is_replace_when_none_is_given(self):
        viewer = FakeSelectingViewer(selected=[1])
        self.assertTrue(tree_backend.handle_tree_select(viewer, {"indices": [7]})["changed"])
        self.assertEqual(viewer.selected_indices, [7])

    def test_add_joins_the_selection_and_toggle_swaps_the_nodes_named(self):
        viewer = FakeSelectingViewer(selected=[1, 2])
        self.select(viewer, [2, 3], "add")
        self.assertEqual(viewer.selected_indices, [1, 2, 3])
        self.select(viewer, [3, 4, 1], "toggle")
        self.assertEqual(viewer.selected_indices, [2, 4])

    def test_an_empty_list_replaces_the_selection_with_none_and_adds_nothing(self):
        viewer = FakeSelectingViewer(selected=[1, 2])
        self.assertTrue(self.select(viewer, [], "add")["ok"])
        self.assertEqual(viewer.selected_indices, [1, 2])
        self.assertEqual(self.select(viewer, [])["selected"], 0)
        self.assertEqual(viewer.selected_indices, [])
        self.assertEqual(viewer.calls[-1], ("status",))

    def test_repeats_count_once(self):
        viewer = FakeSelectingViewer()
        self.assertEqual(self.select(viewer, [4, 4, 2, 4, 2])["selected"], 2)
        self.assertEqual(viewer.selected_indices, [2, 4])
        viewer = FakeSelectingViewer(selected=[2])
        self.select(viewer, [4, 4], "toggle")
        self.assertEqual(viewer.selected_indices, [2, 4])

    def test_hidden_nodes_are_left_out(self):
        viewer = FakeSelectingViewer(hidden=[3, 4])
        self.assertEqual(self.select(viewer, [2, 3, 4, 5])["selected"], 2)
        self.assertEqual(viewer.selected_indices, [2, 5])

    def test_a_request_of_hidden_nodes_only_succeeds_with_a_note_and_changes_nothing(self):
        viewer = FakeSelectingViewer(hidden=[3, 4], selected=[1])
        for mode in ("replace", "add", "toggle"):
            with self.subTest(mode=mode):
                self.assertEqual(self.select(viewer, [3, 4], mode),
                                 {"ok": True, "changed": False, "selected": 1, "note": "nothing_selectable"})
        self.assertEqual((viewer.selected_indices, viewer.calls), ([1], []))

    def test_a_request_that_leaves_the_selection_as_it_was_changes_nothing(self):
        viewer = FakeSelectingViewer(selected=[1, 2])
        for indices, mode in (([1, 2], "replace"), ([2], "add"), ([], "toggle"), ([], "add")):
            with self.subTest(indices=indices, mode=mode):
                result = self.select(viewer, indices, mode)
                self.assertEqual((result["ok"], result["changed"], result["note"]), (True, False, "unchanged"))
        self.assertEqual(viewer.calls, [])
        # An empty selection cleared by an empty replace is the same.
        viewer = FakeSelectingViewer()
        self.assertEqual(self.select(viewer, [])["note"], "unchanged")
        self.assertEqual(viewer.calls, [])

    def test_a_request_that_is_not_well_formed_is_refused_and_changes_nothing(self):
        cases = {
            "indices missing": {"mode": "replace"},
            "indices a tuple": {"indices": (1, 2)},
            "indices a string": {"indices": "12"},
            "indices a number": {"indices": 3},
            "indices None": {"indices": None},
            "indices a dict": {"indices": {"1": 2}},
            "a boolean": {"indices": [1, True]},
            "numpy booleans": {"indices": [np.bool_(True)]},
            "a float": {"indices": [1, 2.0]},
            "a string number": {"indices": [1, "2"]},
            "None in it": {"indices": [1, None]},
            "a nested list": {"indices": [[1]]},
            "negative": {"indices": [1, -1]},
            "the count itself": {"indices": [1, 10]},
            "far out": {"indices": [1, 10**12]},
            "more than the network has": {"indices": [0] * 11},
            "mode unknown": {"indices": [1], "mode": "xor"},
            "mode in capitals": {"indices": [1], "mode": "Replace"},
            "mode a number": {"indices": [1], "mode": 1},
            "mode None": {"indices": [1], "mode": None},
        }
        for name, data in cases.items():
            with self.subTest(name):
                viewer = FakeSelectingViewer(selected=[7])
                result = tree_backend.handle_tree_select(viewer, {"action": "tree_select", **data})
                self.assertFalse(result["ok"])
                self.assertFalse(result["changed"])
                self.assertEqual((viewer.selected_indices, viewer.calls), ([7], []))

    def test_a_request_that_is_not_a_dict_is_refused(self):
        for data in (None, [], "indices", 3):
            with self.subTest(data=data):
                viewer = FakeSelectingViewer(selected=[7])
                self.assertFalse(tree_backend.handle_tree_select(viewer, data)["ok"])
                self.assertEqual((viewer.selected_indices, viewer.calls), ([7], []))

    def test_one_bad_number_among_good_ones_refuses_them_all(self):
        viewer = FakeSelectingViewer(selected=[7])
        self.assertEqual(self.select(viewer, [1, 2, 3, 99])["note"], "bad_indices")
        self.assertEqual(viewer.selected_indices, [7])

    def test_the_most_a_network_can_name_is_taken(self):
        viewer = FakeSelectingViewer(count=1000)
        result = self.select(viewer, list(range(1000)))
        self.assertEqual((result["ok"], result["selected"]), (True, 1000))
        self.assertEqual(viewer.selected_indices, list(range(1000)))

    def test_numpy_integers_are_whole_numbers_too(self):
        viewer = FakeSelectingViewer()
        self.assertTrue(self.select(viewer, [np.int64(3), np.int32(4)])["changed"])
        self.assertEqual(viewer.selected_indices, [3, 4])
        self.assertTrue(all(type(value) is int for value in viewer.selected_indices))

    def test_a_viewer_with_no_network_refuses(self):
        self.assertFalse(self.select(FakeSelectingViewer(count=0), [0])["ok"])
        viewer = FakeSelectingViewer()
        viewer.visible_mask = None
        self.assertEqual(self.select(viewer, [1])["note"], "no_network")


class TreeFocusActionTests(unittest.TestCase):
    """tree_focus puts the click focus on a node that is shown, as a click on the node does."""

    def focus(self, viewer, index):
        return tree_backend.handle_tree_focus(viewer, {"action": "tree_focus", "index": index})

    def test_a_node_that_is_shown_gets_the_focus(self):
        viewer = FakeSelectingViewer(selected=[1])
        self.assertEqual(self.focus(viewer, 4), {"ok": True, "changed": True, "note": None})
        self.assertEqual((viewer.calls, viewer.selected_node_idx, viewer.selected_indices), ([("focus", 4)], 4, [1]))

    def test_numpy_integers_are_taken(self):
        viewer = FakeSelectingViewer()
        self.assertTrue(self.focus(viewer, np.int64(2))["changed"])
        self.assertEqual(viewer.calls, [("focus", 2)])

    def test_what_is_not_a_node_of_the_network_changes_nothing(self):
        for index in (True, False, 1.0, "1", -1, 10, 10**12, [1], {"i": 1}):
            with self.subTest(index=index):
                viewer = FakeSelectingViewer(focus=3)
                self.assertFalse(self.focus(viewer, index)["ok"])
                self.assertEqual((viewer.calls, viewer.selected_node_idx), ([], 3))

    def test_a_hidden_node_is_refused(self):
        viewer = FakeSelectingViewer(hidden=[4])
        self.assertEqual(self.focus(viewer, 4), {"ok": False, "changed": False, "note": "hidden"})
        self.assertEqual(viewer.calls, [])

    def test_a_missing_index_is_refused_not_taken_for_none(self):
        viewer = FakeSelectingViewer(focus=3)
        self.assertFalse(tree_backend.handle_tree_focus(viewer, {"action": "tree_focus"})["ok"])
        self.assertEqual((viewer.calls, viewer.selected_node_idx), ([], 3))

    def test_none_takes_the_focus_off_and_leaves_the_selection(self):
        viewer = FakeSelectingViewer(selected=[1, 2], focus=3)
        self.assertEqual(self.focus(viewer, None), {"ok": True, "changed": True, "note": None})
        self.assertEqual((viewer.calls, viewer.selected_node_idx, viewer.selected_indices), ([("clear",)], None, [1, 2]))

    def test_none_with_no_focus_succeeds_with_a_note(self):
        viewer = FakeSelectingViewer()
        self.assertEqual(self.focus(viewer, None), {"ok": True, "changed": False, "note": "unchanged"})
        self.assertEqual(viewer.calls, [])

    def test_a_request_that_is_not_a_dict_is_refused(self):
        viewer = FakeSelectingViewer()
        self.assertFalse(tree_backend.handle_tree_focus(viewer, [3])["ok"])
        self.assertEqual(viewer.calls, [])


class SelectionStateTests(unittest.TestCase):
    """/api/tree/selection: what a page opened after a selection was made is told."""

    def test_the_selection_comes_sorted_without_repeats_and_in_range(self):
        viewer = FakeSelectingViewer(count=10, selected=[7, 3, 3, 12, -1, 5], focus=4)
        state = tree_backend.selection_state(viewer)
        self.assertEqual((state["status"], state["count"], state["focus"]), ("ok", 3, 4))
        self.assertEqual(list(unpack(state["indices"], "<i4")), [3, 5, 7])

    def test_no_selection_and_no_focus(self):
        state = tree_backend.selection_state(FakeSelectingViewer())
        self.assertEqual((state["count"], state["indices"], state["focus"]), (0, "", None))

    def test_a_focus_that_is_not_a_node_is_none(self):
        for focus in (-1, 10, True, "3", 2.5):
            with self.subTest(focus=focus):
                self.assertIsNone(tree_backend.selection_state(FakeSelectingViewer(focus=focus))["focus"])

    def test_a_viewer_with_no_network_has_no_selection(self):
        viewer = FakeSelectingViewer(count=0, selected=[0])
        self.assertEqual(tree_backend.selection_state(viewer)["count"], 0)

    def test_a_whole_networks_selection_is_four_bytes_a_node(self):
        viewer = FakeSelectingViewer(count=44000, selected=list(range(44000)))
        state = tree_backend.selection_state(viewer)
        self.assertEqual(state["count"], 44000)
        self.assertEqual(len(base64.b64decode(state["indices"])), 44000 * 4)


class RegistrationOfSelectionTests(unittest.TestCase):
    def test_the_two_actions_and_the_route_are_the_tree_plugins(self):
        from web_ui.Plugin_Manager import WebPluginRegistry

        viewer = FakeSelectingViewer()
        registry = WebPluginRegistry(viewer)
        tree_backend.register_backend(registry, viewer)
        self.assertEqual(registry.api_routes[("GET", "/api/tree/selection")].plugin_id, "tree")
        self.assertEqual(sorted(name for name in registry.actions if name.startswith("tree_")), ["tree_focus", "tree_select", "tree_upload"])
        # The registered handler is the one for this viewer, given the posted dict.
        registry.actions["tree_select"]({"action": "tree_select", "indices": [2], "mode": "replace"})
        self.assertEqual(viewer.selected_indices, [2])
        registry.actions["tree_focus"]({"action": "tree_focus", "index": 3})
        self.assertEqual(viewer.selected_node_idx, 3)

    def test_registering_again_replaces_them_without_error(self):
        from web_ui.Plugin_Manager import WebPluginRegistry

        viewer = FakeSelectingViewer()
        registry = WebPluginRegistry(viewer)
        for _ in range(2):
            tree_backend.register_backend(registry, viewer)

    def test_no_other_plugin_can_take_the_names(self):
        from web_ui.Plugin_Manager import WebPluginError, WebPluginRegistry, register_bundled_pages

        viewer = FakeSelectingViewer()
        registry = register_bundled_pages(WebPluginRegistry(viewer))
        tree_backend.register_backend(registry, viewer)
        with self.assertRaisesRegex(WebPluginError, "already registered by plugin 'tree'"):
            registry.register_action("meta", "tree_select", lambda data: None)


class RegistrationTests(unittest.TestCase):
    """The plugin registers its route, and its Console button the way the other pages do."""

    def test_the_backend_registers_its_route_as_the_tree_plugins(self):
        from web_ui.Plugin_Manager import WebPluginRegistry

        registry = WebPluginRegistry(SimpleNamespace())
        tree_backend.register_backend(registry, SimpleNamespace())
        route = registry.api_routes[("GET", "/api/tree/data")]
        self.assertEqual(route.plugin_id, "tree")
        # The same plugin registering again, as its command does, replaces the route and raises nothing.
        tree_backend.register_backend(registry, SimpleNamespace())

    def test_activation_registers_the_button_with_its_icon_and_page(self):
        viewer = SimpleNamespace(add_console_button=mock.Mock(), open_page=mock.Mock(), sidebar_buttons_to_persist=[])
        tree_backend.activate(viewer)
        viewer.add_console_button.assert_called_once()
        arguments = viewer.add_console_button.call_args.kwargs
        self.assertEqual(arguments["name"], "treeViewBtn")
        self.assertEqual(arguments["label"], "Tree View")
        self.assertEqual(arguments["icon"], "list-tree")
        arguments["callback"]()
        viewer.open_page.assert_called_once_with("tree")

    def test_the_command_that_brings_the_button_back_is_persisted_once(self):
        viewer = SimpleNamespace(add_console_button=mock.Mock(), open_page=mock.Mock())
        for _ in range(3):
            tree_backend.activate(viewer)
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["tree"])

    def test_a_viewer_with_no_console_button_registration_is_left_alone(self):
        viewer = SimpleNamespace(open_page=mock.Mock())
        tree_backend.activate(viewer)
        self.assertFalse(hasattr(viewer, "sidebar_buttons_to_persist"))

    def test_the_compatibility_wrapper_registers_and_activates(self):
        viewer = SimpleNamespace(add_console_button=mock.Mock(), open_page=mock.Mock(), web_action_handlers={})
        tree_backend.register(viewer)
        self.assertIn("tree", viewer.web_plugin_registry.registered_plugins)
        self.assertIn(("GET", tree_backend.DATA_ROUTE), viewer.web_plugin_registry.api_routes)
        viewer.add_console_button.assert_called_once()


class TreeSourceTests(FolderTestCase):
    """The trees the page can choose from: the guide tree, and the tree of the network's own edges (single linkage)."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.dict(tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def network(self, count=12, **options):
        """A Viewer with a network of count nodes in a chain (scores falling along it) and no alignment."""
        headers = headers_of(count)
        pairs = [(node, node + 1) for node in range(count - 1)]
        scores = np.arange(count - 1, 0, -1, dtype=np.float32)
        return give_edges(SimpleNamespace(full_headers=headers, n_nodes=count, alignment=None), pairs, scores, **options)

    def test_without_a_guide_tree_the_default_is_the_tree_of_the_edges(self):
        viewer = self.network()
        state = tree_backend.tree_state(viewer)
        self.assertEqual((state["status"], state["source"]), ("ok", "threshold"))
        self.assertEqual([(s["id"], s["available"], s["reason"]) for s in state["sources"]],
                         [("threshold", True, None), ("guide", False, "no_alignment")])
        counts = state["counts"]
        self.assertEqual((counts["leaves"], counts["nodes"], counts["leaves_in_network"]), (12, 23, 12))
        self.assertIs(state["rooted"], True)
        self.assertEqual(state["warnings"], [])
        self.assertTrue(np.all(unpack(state["support"], "i1") == -1))

    def test_the_leaves_say_which_node_of_the_network_each_is(self):
        viewer = self.network()
        state = tree_backend.tree_state(viewer)
        node = unpack(state["viewer_node"], "<i4")
        self.assertEqual(sorted(node.tolist()), list(range(12)))
        self.assertEqual(state["labels"], [viewer.full_headers[n] for n in node])
        parent = unpack(state["parent"], "<i4")
        self.assertTrue(np.all(parent[:-1] > np.arange(len(parent) - 1)))
        self.assertEqual(int(parent[-1]), -1)
        score = unpack(state["score"], "<f4")
        length = unpack(state["length"], "<f4")
        self.assertEqual(len(score), len(parent))
        self.assertTrue(np.allclose(length[:-1], score[:-1] - score[parent[:-1]]))
        self.assertEqual(float(score[0]), 11.0)  # leaves stand at the highest score
        self.assertEqual(float(score[-1]), 1.0)  # the root joins at the lowest

    def test_the_details_say_what_the_tree_is_built_from(self):
        viewer = self.network(threshold=4.5)
        info = tree_backend.tree_state(viewer)["threshold"]
        self.assertEqual((info["edges"], info["groups"], info["forest"]), (11, 1, False))
        self.assertEqual((info["top"], info["bottom"], info["current"]), (11.0, 1.0, 4.5))
        self.assertEqual((info["kept"], info["cutoff"], info["score_kind"]), ("threshold", 1.0, "similarity"))
        self.assertGreaterEqual(info["build_seconds"], 0)
        self.assertEqual(tree_backend.tree_state(self.network(evalue=True))["threshold"]["score_kind"], "blast")

    def test_a_manifold_layout_says_the_tree_is_of_the_kept_edges(self):
        import EMAPSSN_Config as cfg

        viewer = self.network()
        with mock.patch.object(cfg, "UMAP_MODE", True), mock.patch.object(cfg, "UMAP_NEIGHBORS", 7):
            info = tree_backend.tree_state(viewer)["threshold"]
        self.assertEqual((info["kept"], info["k"]), ("top_k", 7))
        self.assertNotIn("cutoff", info)

    def test_nodes_that_share_no_edge_make_a_forest_the_page_is_told_of(self):
        viewer = give_edges(SimpleNamespace(full_headers=headers_of(6), n_nodes=6, alignment=None),
                            [(0, 1), (2, 3)], [5, 3])
        state = tree_backend.tree_state(viewer)
        info = state["threshold"]
        self.assertEqual((info["forest"], info["groups"]), (True, 4))
        self.assertLess(info["root_score"], info["bottom"])

    def test_a_network_without_edges_has_no_threshold_tree(self):
        for edge_layer, reason in ((None, "no_edges"), (SimpleNamespace(order=None), "no_edges")):
            with self.subTest(reason=reason):
                viewer = SimpleNamespace(full_headers=headers_of(5), n_nodes=5, alignment=None, edge_layer=edge_layer)
                state = tree_backend.tree_state(viewer)
                self.assertEqual(state["status"], "no_alignment")  # both unavailable: the guide's words
                self.assertEqual(tree_backend.tree_state(viewer, "threshold")["status"], reason)
                self.assertEqual(state["sources"][0], {"id": "threshold", "available": False, "reason": reason})
        self.assertEqual(tree_backend.tree_state(SimpleNamespace(alignment=None), "threshold")["status"], "no_network")

    def test_a_guide_tree_is_the_default_when_the_alignment_has_one(self):
        headers = headers_of(20)
        viewer, _msa, _tree = self.viewer_for(headers)
        give_edges(viewer, [(n, n + 1) for n in range(19)], np.arange(19, 0, -1))
        state = tree_backend.tree_state(viewer)
        self.assertEqual((state["status"], state["source"]), ("ok", "guide"))
        self.assertEqual([(s["id"], s["available"]) for s in state["sources"]], [("threshold", True), ("guide", True)])
        self.assertEqual(state["name"], "guide_tree")

    def test_the_source_asked_for_is_kept_for_the_session(self):
        headers = headers_of(20)
        viewer, _msa, _tree = self.viewer_for(headers)
        give_edges(viewer, [(n, n + 1) for n in range(19)], np.arange(19, 0, -1))
        self.assertEqual(tree_backend.tree_state(viewer, "threshold")["source"], "threshold")
        self.assertEqual(tree_backend.tree_state(viewer)["source"], "threshold")
        self.assertEqual(tree_backend.tree_state(viewer, "guide")["source"], "guide")
        self.assertEqual(tree_backend.tree_state(viewer)["source"], "guide")

    def test_a_kept_source_that_cannot_be_shown_any_more_gives_way_to_the_default(self):
        viewer = self.network()
        self.assertEqual(tree_backend.tree_state(viewer, "threshold")["source"], "threshold")
        viewer.edge_layer = None
        state = tree_backend.tree_state(viewer)
        self.assertEqual(state["status"], "no_alignment")
        self.assertEqual(state["source"], "guide")

    def test_a_source_asked_for_that_cannot_be_shown_says_why_and_is_not_kept(self):
        viewer = self.network()
        state = tree_backend.tree_state(viewer, "guide")
        self.assertEqual((state["status"], state["source"]), ("no_alignment", "guide"))
        self.assertEqual(tree_backend.tree_state(viewer)["source"], "threshold")

    def test_an_unknown_source_fails_and_changes_nothing(self):
        viewer = self.network()
        tree_backend.tree_state(viewer, "threshold")
        for bad in ("upload", "", "Threshold", "threshold ", "../guide", 5):
            with self.subTest(source=bad), self.assertRaises(ValueError):
                tree_backend.tree_state(viewer, bad)
        self.assertEqual(viewer._tree_source, "threshold")

    def test_the_tree_is_built_once_for_a_network_and_again_for_a_new_one(self):
        from utilities import Threshold_Tree

        viewer = self.network()
        with mock.patch.object(Threshold_Tree, "threshold_tree", wraps=Threshold_Tree.threshold_tree) as build:
            first = tree_backend.tree_state(viewer)
            second = tree_backend.tree_state(viewer)
            self.assertEqual(build.call_count, 1)
            self.assertIs(first["labels"], second["labels"])
            # The slider moves without a new tree: only what the page reads of it changes.
            viewer.current_slider_threshold = 7.0
            self.assertEqual(tree_backend.tree_state(viewer)["threshold"]["current"], 7.0)
            self.assertEqual(build.call_count, 1)
            give_edges(viewer, [(0, 1)], [3.0])
            tree_backend.tree_state(viewer)
            self.assertEqual(build.call_count, 2)

    def test_the_cache_does_not_keep_a_replaced_networks_edges_alive(self):
        import gc

        viewer = self.network()
        tree_backend.tree_state(viewer)
        reference = tree_backend._THRESHOLD_CACHE["order"]
        viewer.edge_layer = None
        gc.collect()
        self.assertIsNone(reference())

    def test_a_44000_node_network_is_built_within_the_budget(self):
        count, edge_count = 44000, 3_000_000
        rng = np.random.default_rng(42)
        pairs = rng.integers(0, count, (edge_count, 2))
        scores = rng.random(edge_count).astype(np.float32)
        viewer = give_edges(SimpleNamespace(full_headers=headers_of(count), n_nodes=count, alignment=None), pairs, scores)
        started = time.perf_counter()
        state = tree_backend.tree_state(viewer)
        elapsed = time.perf_counter() - started
        self.assertEqual(state["status"], "ok")
        self.assertEqual(state["counts"]["leaves"], count)
        self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()


class MismatchTests(FolderTestCase):
    """The names the guide tree's warnings count (the lists Show Details shows)."""

    def test_each_list_names_what_its_warning_counts(self):
        headers = headers_of(30)
        network = headers[:20] + headers_of(5, prefix="tr|Y")
        viewer, _msa, _tree = self.viewer_for(headers, network=network)
        viewer.alignment.aligned_node_mask[[0, 3, 22]] = False
        state = tree_backend.mismatch_state(viewer, "guide")
        self.assertEqual((state["source"], state["status"]), ("guide", "ok"))
        self.assertEqual(sorted(state["leaves_not_in_network"]["names"]), sorted(headers[20:]))
        self.assertEqual(state["network_nodes_not_in_tree"]["names"], network[20:])
        # Only leaves that are nodes of the network can lack a sequence: node 22 is not a leaf of the tree.
        self.assertEqual(sorted(state["leaves_not_in_alignment"]["names"]), sorted([network[0], network[3]]))
        self.assertEqual(state["ambiguous_names"], {"count": 0, "names": []})
        for key in tree_backend.MISMATCH_LISTS:
            self.assertEqual(state[key]["count"], len(state[key]["names"]))
        warned = {item["code"]: item["count"] for item in tree_backend.tree_state(viewer)["warnings"]}
        self.assertEqual(warned, {"leaves_not_in_network": 10, "network_nodes_not_in_tree": 5, "leaves_not_in_alignment": 2})

    def test_a_tree_that_fits_has_empty_lists(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(12))
        state = tree_backend.mismatch_state(viewer, "guide")
        self.assertEqual(state["status"], "ok")
        self.assertEqual([state[key]["names"] for key in tree_backend.MISMATCH_LISTS], [[], [], [], []])

    def test_a_guide_tree_that_cannot_be_shown_says_why_with_empty_lists(self):
        viewer, _msa, _tree = self.viewer_for(headers_of(12), tree=None)
        state = tree_backend.mismatch_state(viewer, "guide")
        self.assertEqual(state["status"], "no_tree")
        self.assertEqual([state[key]["count"] for key in tree_backend.MISMATCH_LISTS], [0, 0, 0, 0])
        self.assertEqual(tree_backend.mismatch_state(SimpleNamespace(alignment=None), "guide")["status"], "no_alignment")

    def test_the_threshold_tree_has_every_node_as_a_leaf_so_its_lists_are_empty(self):
        headers = headers_of(10)
        viewer = give_edges(SimpleNamespace(full_headers=headers, n_nodes=10, alignment=None),
                            [(n, n + 1) for n in range(9)], np.arange(9, 0, -1, dtype=np.float32))
        state = tree_backend.mismatch_state(viewer, "threshold")
        self.assertEqual(state["status"], "ok")
        self.assertEqual([state[key]["names"] for key in tree_backend.MISMATCH_LISTS], [[], [], [], []])
        self.assertEqual(tree_backend.mismatch_state(SimpleNamespace(full_headers=[]), "threshold")["status"], "no_network")

    def test_an_unknown_source_raises(self):
        for bad in ("upload", "", "Guide", "../guide", 5, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                tree_backend.mismatch_state(SimpleNamespace(), bad)


class SnapshotPartTests(FolderTestCase):
    """The tree in an inspection snapshot: what the summary says of it, and what a capture freezes of each source."""

    def setUp(self):
        super().setUp()
        for cache, value in ((tree_backend._THRESHOLD_CACHE, {"order": None, "count": 0, "built": None}),):
            patcher = mock.patch.dict(cache, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def guide_viewer(self, headers=None, network=None, aligned=None, **options):
        headers = headers_of(12) if headers is None else headers
        msa, tree = make_alignment_folder(self.folder, headers, **options)
        viewer = make_viewer(msa, headers if network is None else network, aligned=aligned)
        count = len(viewer.full_headers)
        viewer, service = snapshot_viewer(viewer.full_headers, base=viewer, clusters=[i % 3 for i in range(count)])
        return viewer, service, tree

    def chain_viewer(self, count=12):
        headers = headers_of(count)
        pairs = [(node, node + 1) for node in range(count - 1)]
        viewer = give_edges(make_viewer(None, headers), pairs, np.arange(count - 1, 0, -1, dtype=np.float32))
        viewer, service = snapshot_viewer(headers, base=viewer, clusters=[i % 3 for i in range(count)])
        return viewer, service

    def captured(self, service, **options):
        sid = service.capture_snapshot(parts={"tree": options})
        return sid, service.snapshots.execute("get_summary", sid)["parts"]["tree"]["captured"]

    def test_the_part_is_declared_by_the_tree_plugin_with_its_switch_option_queries_and_capability(self):
        viewer, _service = snapshot_viewer(headers_of(3))
        part = viewer.web_plugin_registry.snapshot_parts()["tree"]
        self.assertEqual(part.plugin_id, "tree")
        self.assertEqual(part.capability, "tree_snapshots_v1")
        self.assertEqual(dict(part.summary_arguments), {"include_tree": "switch", "tree_source": "option"})
        self.assertEqual(sorted(part.queries), ["query_clades", "query_tree_names"])
        self.assertEqual(viewer.web_plugin_registry.part_queries(), {"query_clades": "tree", "query_tree_names": "tree"})
        tree_backend.register_backend(viewer.web_plugin_registry, viewer)  # again, as the command does

    def test_the_overview_says_what_can_be_captured_and_why_not_without_reading_any_tree(self):
        with mock.patch.object(tree_backend, "_read_cached", side_effect=AssertionError("read a tree")), \
                mock.patch.object(tree_backend, "_build_threshold", side_effect=AssertionError("built a tree")):
            viewer, _service = snapshot_viewer(headers_of(3))
            self.assertEqual(tree_backend.snapshot_overview(viewer), {
                "available": [], "opens_on": None,
                "unavailable": {"guide": "no_alignment", "threshold": "no_edges", "uploaded": "no_upload"}})
            msa, _tree = make_alignment_folder(self.folder / "with", headers_of(6))
            viewer, _service = snapshot_viewer(headers_of(6), base=make_viewer(msa, headers_of(6)))
            self.assertEqual(tree_backend.snapshot_overview(viewer), {
                "available": ["guide"], "opens_on": "guide", "unavailable": {"threshold": "no_edges", "uploaded": "no_upload"}})
            viewer = give_edges(viewer, [(0, 1)], [1.0])
            self.assertTrue(upload_tree(viewer, tree={"labels": ["a", "b"], "parent": [2, 2, -1], "length": [1, 1, 0], "support": [-1] * 3})["ok"])
            viewer._tree_source = "threshold"
            self.assertEqual(tree_backend.snapshot_overview(viewer), {"available": ["guide", "threshold", "uploaded"], "opens_on": "threshold", "unavailable": {}})
            msa, _tree = make_alignment_folder(self.folder / "renamed", headers_of(6), alignment="mine.fasta")
            viewer, _service = snapshot_viewer(headers_of(6), base=make_viewer(msa, headers_of(6)))
            self.assertEqual(tree_backend.snapshot_overview(viewer)["unavailable"]["guide"], "unnamed_alignment")
            msa, _tree = make_alignment_folder(self.folder / "bare", headers_of(6), tree=None)
            viewer, _service = snapshot_viewer(headers_of(6), base=make_viewer(msa, headers_of(6)))
            self.assertEqual(tree_backend.snapshot_overview(viewer)["unavailable"]["guide"], "no_tree")

    def test_a_summary_without_the_tree_lists_the_overview_only(self):
        viewer, service, _tree = self.guide_viewer()
        sid = service.capture_snapshot()
        summary = service.snapshots.execute("get_summary", sid)
        self.assertEqual(summary["parts"]["tree"]["available"], ["guide"])
        self.assertNotIn("captured", summary["parts"]["tree"])
        self.assertNotIn("tree_snapshots_v1", summary["capabilities"])
        self.assertLess(len(json.dumps(summary["parts"])), 400)

    def test_the_guide_tree_is_frozen_with_what_the_page_is_sent(self):
        viewer, service, _tree_path = self.guide_viewer()
        sid, summary = self.captured(service, tree_source="guide")
        state = tree_backend.tree_state(viewer, "guide")
        part = service.snapshots.items[sid]["data"]["parts"]["tree"]
        for key, dtype in (("parent", "<i4"), ("length", "<f4"), ("support", "i1"), ("viewer_node", "<i4")):
            np.testing.assert_array_equal(part[key], unpack(state[key], dtype), err_msg=key)
        self.assertEqual(part["labels"], state["labels"])
        self.assertEqual(summary["source"], "guide")
        self.assertEqual(summary["file"], state["tree_path"])
        self.assertEqual(summary["counts"], {**state["counts"], "clades": state["counts"]["nodes"] - 12})
        self.assertEqual((summary["rooted"], summary["has_lengths"], summary["warnings"], summary["threshold"]), (state["rooted"], state["has_lengths"], [], None))
        self.assertEqual(summary["root"], "file" if state["rooted"] is True else "midpoint")
        self.assertRegex(summary["fingerprint"], r"^[0-9a-f]{8}$")
        self.assertEqual(summary["fingerprint"], tree_backend._fingerprint(part["parent"], part["length"], part["support"], part["labels"]))
        rows = service.snapshots.execute("query_clades", sid, limit=500)["rows"]
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0]["leaves"], {"total": 12, "in_network": 12, "not_in_network": 0, "hidden": 0})

    def test_the_default_is_the_tree_the_page_opens_on_and_a_capture_never_changes_the_choice(self):
        viewer, service, _tree = self.guide_viewer()
        self.assertEqual(self.captured(service)[1]["source"], "guide")
        give_edges(viewer, [(n, n + 1) for n in range(11)], np.arange(11, 0, -1, dtype=np.float32))
        viewer._tree_source = "threshold"
        self.assertEqual(self.captured(service)[1]["source"], "threshold")
        self.assertEqual(self.captured(service, tree_source="guide")[1]["source"], "guide")
        self.assertEqual(viewer._tree_source, "threshold")  # the page's choice is the user's, not the snapshot's
        viewer._tree_source = None
        self.captured(service, tree_source="threshold")
        self.assertIsNone(viewer._tree_source)

    def test_support_of_the_guide_tree_is_split_stability(self):
        viewer, service, _tree = self.guide_viewer(notes=None)
        _sid, summary = self.captured(service, tree_source="guide")
        state = tree_backend.guide_state(viewer)
        has = bool((unpack(state["support"], "i1") >= 0).any())
        self.assertEqual(summary["support_kind"], "split_stability" if has else None)

    def test_the_threshold_tree_is_frozen_with_its_threshold_and_a_root_that_is_the_networks(self):
        viewer, service = self.chain_viewer()
        sid, summary = self.captured(service, tree_source="threshold")
        state = tree_backend.threshold_state(viewer)
        part = service.snapshots.items[sid]["data"]["parts"]["tree"]
        np.testing.assert_array_equal(part["parent"], unpack(state["parent"], "<i4"))
        np.testing.assert_array_equal(part["viewer_node"], unpack(state["viewer_node"], "<i4"))
        self.assertEqual((summary["source"], summary["name"], summary["file"], summary["rooted"], summary["root"]), ("threshold", None, None, True, "network"))
        self.assertEqual((summary["has_lengths"], summary["support_kind"], summary["warnings"]), (True, None, []))
        self.assertEqual(summary["threshold"]["groups"], 1)
        self.assertIs(summary["threshold"]["forest"], False)
        self.assertEqual(summary["threshold"]["kept"], state["threshold"]["kept"])
        self.assertEqual(summary["counts"]["leaves"], 12)
        self.assertEqual(self.captured(service)[1]["source"], "threshold")  # no guide tree: it is the one the page opens on

    def test_an_opened_tree_is_frozen_with_its_file_support_and_clade_names(self):
        headers = list("ABCDEF")
        viewer, service = snapshot_viewer(headers)
        self.assertTrue(upload_tree(viewer, tree={**SMALL_TREE, "clade_names": {"7": "Mammals"}})["ok"])
        sid, summary = self.captured(service)
        self.assertEqual((summary["source"], summary["file"], summary["name"], summary["root"], summary["rooted"]), ("uploaded", "small.nwk", None, "file", True))
        self.assertEqual((summary["support_kind"], summary["has_lengths"]), ("file", True))
        self.assertEqual(service.snapshots.items[sid]["data"]["parts"]["tree"]["clade_names"], {7: "Mammals"})
        viewer, service = snapshot_viewer(headers)
        upload_tree(viewer, tree={**SMALL_TREE, "support": [-1] * 10}, rooted=False)
        self.assertEqual(self.captured(service)[1]["support_kind"], None)
        self.assertEqual(self.captured(service)[1]["rooted"], False)

    def test_the_fit_warnings_are_reported_as_codes_and_counts_and_the_lists_name_them(self):
        headers = headers_of(12)
        viewer, service, _tree = self.guide_viewer(headers, network=headers[:10] + ["outsider"])
        sid, summary = self.captured(service, tree_source="guide")
        self.assertEqual(summary["warnings"], [{"code": "leaves_not_in_network", "count": 2}, {"code": "network_nodes_not_in_tree", "count": 1}])
        self.assertEqual(summary["counts"]["leaves_not_in_network"], 2)
        names = lambda kind: service.snapshots.execute("query_tree_names", sid, list=kind)["rows"]  # noqa: E731
        self.assertEqual([row["name"] for row in names("leaves_not_in_network")], headers[10:])
        self.assertTrue(all("index" not in row and "leaf" in row for row in names("leaves_not_in_network")))
        self.assertEqual(names("network_nodes_not_in_tree"), [{"name": "outsider", "index": 10}])
        # The same lists the page's details show.
        mismatch = tree_backend.mismatch_state(viewer, "guide")
        self.assertEqual([row["name"] for row in names("leaves_not_in_network")], mismatch["leaves_not_in_network"]["names"])
        self.assertEqual([row["name"] for row in names("network_nodes_not_in_tree")], mismatch["network_nodes_not_in_tree"]["names"])

    def test_leaves_the_alignment_lacks_are_listed(self):
        headers = headers_of(12)
        aligned = [True] * 12
        aligned[3] = aligned[7] = False
        viewer, service, _tree = self.guide_viewer(headers, aligned=aligned)
        sid, summary = self.captured(service, tree_source="guide")
        self.assertEqual(summary["warnings"], [{"code": "leaves_not_in_alignment", "count": 2}])
        rows = service.snapshots.execute("query_tree_names", sid, list="leaves_not_in_alignment")["rows"]
        self.assertEqual(sorted(row["name"] for row in rows), [headers[3], headers[7]])
        self.assertEqual(sorted(row["index"] for row in rows), [3, 7])
        self.assertEqual([row["name"] for row in rows], tree_backend.mismatch_state(viewer, "guide")["leaves_not_in_alignment"]["names"])

    def test_a_source_that_cannot_be_shown_is_refused_in_words_and_nothing_is_kept(self):
        from desktop.Viewer_Inspection import ViewerInspectionError

        viewer, service, _tree = self.guide_viewer(tree=None)
        cases = (("guide", "no_tree", "has no guide tree file"), ("threshold", "no_edges", "no edges"), ("uploaded", "no_upload", "no tree file has been opened"))
        for source, status, words in cases:
            with self.subTest(source=source), self.assertRaises(ViewerInspectionError) as raised:
                service.capture_snapshot(parts={"tree": {"tree_source": source}})
            self.assertIn(f"The {source} tree cannot be shown", str(raised.exception))
            self.assertIn(f"({status})", str(raised.exception))
            self.assertIn(words, str(raised.exception))
        self.assertEqual(len(service.snapshots.items), 0)
        with self.assertRaisesRegex(ViewerInspectionError, "The guide tree cannot be shown"):
            service.capture_snapshot(parts={"tree": {}})

    def test_the_refusal_names_the_trees_that_can_be_shown(self):
        from desktop.Viewer_Inspection import ViewerInspectionError

        viewer, service = self.chain_viewer()
        with self.assertRaisesRegex(ViewerInspectionError, r"The guide tree cannot be shown: no alignment is loaded \(no_alignment\)\. Trees that can be shown: threshold\."):
            service.capture_snapshot(parts={"tree": {"tree_source": "guide"}})

    def test_a_tree_that_matches_nothing_and_a_file_that_cannot_be_read_are_refused_with_the_reason(self):
        from desktop.Viewer_Inspection import ViewerInspectionError

        viewer, service = snapshot_viewer(["p", "q"])
        upload_tree(viewer, tree={"labels": ["a", "b"], "parent": [2, 2, -1], "length": [1, 1, 0], "support": [-1] * 3})
        with self.assertRaisesRegex(ViewerInspectionError, r"none of the tree's leaves is a node of the network \(no_overlap\)"):
            service.capture_snapshot(parts={"tree": {"tree_source": "uploaded"}})
        viewer, service, tree_path = self.guide_viewer()
        pathlib.Path(tree_path).write_text("not a tree", encoding="utf-8")
        with self.assertRaisesRegex(ViewerInspectionError, r"cannot be read \(unreadable: "):
            service.capture_snapshot(parts={"tree": {"tree_source": "guide"}})

    def test_an_unknown_source_is_refused(self):
        viewer, _service = snapshot_viewer(headers_of(3))
        with self.assertRaisesRegex(ValueError, "does not know a tree called nope"):
            tree_backend.capture_snapshot_part(viewer, {"tree_source": "nope"}, 10**9)

    def test_a_tree_too_big_for_the_budget_is_refused(self):
        viewer, _service = self.chain_viewer()
        with self.assertRaisesRegex(ValueError, "too large for the snapshot memory budget"):
            tree_backend.capture_snapshot_part(viewer, {"tree_source": "threshold"}, 100)

    def test_what_is_frozen_cannot_be_changed_by_the_viewer_afterwards(self):
        viewer, service = self.chain_viewer()
        sid, _summary = self.captured(service, tree_source="threshold")
        part = service.snapshots.items[sid]["data"]["parts"]["tree"]
        self.assertFalse(part["parent"].flags.writeable)
        before = service.snapshots.execute("query_clades", sid, limit=500)["rows"]
        viewer.edge_layer = None  # the network goes away
        self.assertEqual(service.snapshots.execute("query_clades", sid, limit=500)["rows"], before)
