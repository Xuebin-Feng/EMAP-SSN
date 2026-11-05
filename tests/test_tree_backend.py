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
    TREE_NAME,
    headers_of,
    http_get,
    make_alignment_folder,
    make_viewer,
    page_headers,
    random_linkage,
    server_viewer,
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
        self.assertEqual(tree_backend.tree_state(SimpleNamespace(alignment=None)), {"status": "no_alignment"})
        self.assertEqual(tree_backend.tree_state(SimpleNamespace()), {"status": "no_alignment"})
        for msa in (None, "", "   "):
            with self.subTest(msa=msa):
                self.assertEqual(tree_backend.tree_state(make_viewer(msa, ["a"])), {"status": "no_alignment"})

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
                self.assertEqual(tree_backend.tree_state(make_viewer(msa, ["a"])),
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
        state = tree_backend.tree_state(viewer)
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

    def test_a_request_from_another_origin_is_refused(self):
        status, _headers, _body = self.http_get(
            self.server, tree_backend.DATA_ROUTE, {**self.page_headers(self.server), "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_the_route_answers_get_only(self):
        # POST is not registered for it: the server falls through to its own routes, which do not know it.
        route = self.viewer.web_plugin_registry.api_routes
        self.assertEqual([key for key in route if key[1] == tree_backend.DATA_ROUTE], [("GET", tree_backend.DATA_ROUTE)])


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
        self.assertEqual(arguments["icon"], "layers")
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


if __name__ == "__main__":
    unittest.main()
