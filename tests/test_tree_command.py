# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tree command (commands/tree.py): it opens Tree View and adds its Console button, and reports what happened.

It takes no argument but help and --register-only (which brings the Console button back when a session
loads), says why when the page cannot open (a Viewer with no window), runs at once even while commands wait,
and is a catalogued command whose agent reply waits for approval. A valid request that has nothing to show succeeds
with a note: the page opens on its empty state and the result names what is missing; a page that is open
already is a request that changes nothing.
"""

from __future__ import annotations

from pathlib import Path
import sys
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
for folder in (SRC_DIR, SRC_DIR.parent):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import Command_Declarations  # noqa: E402
import Command_Engine  # noqa: E402
from commands import tree as tree_command  # noqa: E402
from desktop.Command_Metadata import AGENT_POLICY, COMMAND_METADATA, agent_policy  # noqa: E402
from tests.command_fixtures import run_command  # noqa: E402
from tests.tree_fixtures import ALIGNMENT_NAME, TREE_NAME, make_viewer  # noqa: E402


def viewer(opens=True, alignment="with tree"):
    """A Viewer whose open_page answers ``opens``, and which has a Console to add a button to.

    alignment: "with tree" (an alignment of the tool's name whose tree file is there; the default, so that a
    test of the opening says nothing about a missing tree), "no tree", "unnamed" (a name the tool did not
    give), or None for no alignment loaded.
    """
    fake = SimpleNamespace(
        console_text=SimpleNamespace(text=""),
        open_page=mock.Mock(return_value=opens),
        add_console_button=mock.Mock(),
        alignment=None,
    )
    if alignment is not None:
        folder = tempfile.TemporaryDirectory()
        fake.folder = folder
        name = "mafft_result.fasta" if alignment == "unnamed" else ALIGNMENT_NAME
        msa = Path(folder.name) / name
        msa.write_text(">x\nA\n", encoding="utf-8")
        if alignment == "with tree":
            (msa.parent / "Guide_Trees").mkdir()
            (msa.parent / "Guide_Trees" / TREE_NAME).write_text("#NEXUS\n", encoding="utf-8")
        fake.alignment = make_viewer(msa, ["a"]).alignment
        fake.expected_tree = os.path.abspath(msa.parent / "Guide_Trees" / TREE_NAME)
    return fake


class TreeViewerTestCase(unittest.TestCase):
    """A test case whose fake Viewers' temporary folders go with the test."""

    def viewer(self, *arguments, **options):
        fake = viewer(*arguments, **options)
        if hasattr(fake, "folder"):
            self.addCleanup(fake.folder.cleanup)
        return fake


class OpeningTests(TreeViewerTestCase):
    def test_it_opens_the_page_and_says_so(self):
        fake = self.viewer()
        succeeded, failed, _printed = run_command(tree_command, fake, [])
        fake.open_page.assert_called_once_with("tree")
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Opened Tree View.")

    def test_it_adds_the_console_button_and_keeps_it_for_the_session(self):
        fake = self.viewer()
        run_command(tree_command, fake, [])
        fake.add_console_button.assert_called_once()
        self.assertEqual(fake.add_console_button.call_args.kwargs["name"], "treeViewBtn")
        self.assertEqual(fake.sidebar_buttons_to_persist, ["tree"])

    def test_a_command_from_the_portal_never_waits_for_someone_to_close_a_dialog(self):
        fake = self.viewer()
        with mock.patch.object(tree_command, "CURRENT", SimpleNamespace(get=lambda: object())):
            run_command(tree_command, fake, [])
        fake.open_page.assert_called_once_with("tree", show_existing_dialog=False)

    def test_a_page_that_is_open_already_is_no_failure(self):
        # Nothing is left to open, which a valid request may find: it succeeds, with the fact as its note.
        fake = self.viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=True):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        failed.assert_not_called()
        self.assertEqual(str(succeeded.call_args.args[1]), "Tree View is already open in your browser.")

    def test_a_viewer_with_no_window_says_where_the_page_is_served(self):
        fake = self.viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=False), \
                mock.patch.object(tree_command, "is_headless", return_value=True), \
                mock.patch.object(tree_command, "page_address", return_value="http://127.0.0.1:1/tree.html") as address:
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        address.assert_called_once_with(fake, "/tree.html")
        failed.assert_not_called()
        self.assertIn("http://127.0.0.1:1/tree.html", str(succeeded.call_args.args[1]))
        self.assertIn("no window", str(succeeded.call_args.args[1]))
        self.assertNotIn("guide tree", str(succeeded.call_args.args[1]))

    def test_a_page_that_cannot_open_for_another_reason_fails_plainly(self):
        fake = self.viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=False), \
                mock.patch.object(tree_command, "is_headless", return_value=False):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        succeeded.assert_not_called()
        self.assertEqual(failed.call_args.args[1], "Could not open Tree View.")


class NoTreeTests(TreeViewerTestCase):
    """With no tree to show the page opens all the same, and the result says what is missing."""

    def run_tree(self, **options):
        fake = self.viewer(**options)
        succeeded, failed, _printed = run_command(tree_command, fake, [])
        failed.assert_not_called()
        fake.open_page.assert_called_once_with("tree")
        return fake, str(succeeded.call_args.args[1])

    def test_no_alignment_loaded(self):
        _fake, said = self.run_tree(alignment=None)
        self.assertEqual(said, "Opened Tree View. No alignment is loaded, so there is no guide tree to show.")

    def test_an_alignment_the_tool_did_not_name(self):
        _fake, said = self.run_tree(alignment="unnamed")
        self.assertEqual(said, "Opened Tree View. The loaded alignment, mafft_result.fasta, was not named by the Embedding "
                               "MSA tool, so it has no guide tree.")

    def test_an_alignment_with_no_tree_file_names_the_file_the_page_looks_for(self):
        fake, said = self.run_tree(alignment="no tree")
        self.assertEqual(said, "Opened Tree View. No guide tree was found for the loaded alignment; Tree View looks for "
                               f"{fake.expected_tree}.")
        self.assertTrue(os.path.isabs(fake.expected_tree))

    def test_a_tree_that_is_there_has_no_note_whatever_it_holds(self):
        # Whether it can be read is for the page to say: reading it here would parse it on the Viewer's thread.
        _fake, said = self.run_tree(alignment="with tree")
        self.assertEqual(said, "Opened Tree View.")

    def test_an_open_page_with_no_tree_says_both(self):
        fake = self.viewer(opens=False, alignment="no tree")
        with mock.patch.object(tree_command, "page_is_open", return_value=True):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        failed.assert_not_called()
        self.assertEqual(str(succeeded.call_args.args[1]), "Tree View is already open in your browser. No guide tree was "
                         f"found for the loaded alignment; Tree View looks for {fake.expected_tree}.")

    def test_a_viewer_with_no_window_says_so_and_what_is_missing(self):
        fake = self.viewer(opens=False, alignment=None)
        with mock.patch.object(tree_command, "page_is_open", return_value=False), \
                mock.patch.object(tree_command, "is_headless", return_value=True), \
                mock.patch.object(tree_command, "page_address", return_value="http://127.0.0.1:1/tree.html"):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        failed.assert_not_called()
        self.assertEqual(str(succeeded.call_args.args[1]), "A Viewer with no window opens no browser page; Tree View is "
                         "served at http://127.0.0.1:1/tree.html. No alignment is loaded, so there is no guide tree to show.")

    def test_registering_the_button_says_nothing_of_the_tree(self):
        fake = self.viewer(alignment=None)
        succeeded, _failed, _printed = run_command(tree_command, fake, ["--register-only"])
        self.assertEqual(str(succeeded.call_args.args[1]), "Registered Tree View.")

    def test_the_catalog_says_the_page_opens_without_a_tree(self):
        text = COMMAND_METADATA["tree"]["summary"]
        self.assertIn("still opens", text)
        self.assertIn("already", text)


class ArgumentTests(TreeViewerTestCase):
    def test_register_only_adds_the_button_and_opens_nothing(self):
        fake = self.viewer()
        succeeded, failed, _printed = run_command(tree_command, fake, ["--register-only"])
        fake.open_page.assert_not_called()
        fake.add_console_button.assert_called_once()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Registered Tree View.")

    def test_arguments_are_refused_with_the_usage_and_nothing_opens(self):
        for args in (["file.nex"], ["--register-only", "x"], ["open"]):
            with self.subTest(args=args):
                fake = self.viewer()
                succeeded, failed, printed = run_command(tree_command, fake, args)
                fake.open_page.assert_not_called()
                fake.add_console_button.assert_not_called()
                succeeded.assert_not_called()
                self.assertIn(f"takes no arguments, not '{args[0]}'", str(failed.call_args.args[1]))
                self.assertIn("Usage: tree", str(failed.call_args.args[1]))
                self.assertIn("Usage: tree", printed)

    def test_the_page_names_no_file_so_no_argument_can(self):
        fake = self.viewer()
        run_command(tree_command, fake, ["C:/secret.nex"])
        fake.open_page.assert_not_called()

    def test_help_prints_in_every_spelling_and_opens_nothing(self):
        for word in ("help", "-h", "--help", "HELP"):
            with self.subTest(word=word):
                fake = self.viewer()
                succeeded, failed, printed = run_command(tree_command, fake, [word])
                fake.open_page.assert_not_called()
                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], "Help information printed to the Console.")
                self.assertIn("Usage: tree", printed)
                self.assertIn("Guide_Trees", printed)


class DeclarationTests(unittest.TestCase):
    def test_it_is_catalogued_and_an_agent_reply_that_opens_it_waits_for_approval(self):
        # It opens a page in the user's browser, as meta and esmfold do.
        self.assertIn("tree", COMMAND_METADATA)
        entry = COMMAND_METADATA["tree"]
        self.assertTrue(entry["summary"])
        self.assertTrue(entry["opens_interface"])
        self.assertFalse(entry["writes_files"])
        self.assertEqual(agent_policy("tree"), "approve")
        self.assertEqual(AGENT_POLICY["tree"]["default"], "approve")

    def test_the_catalog_entry_says_the_page_names_no_file(self):
        text = COMMAND_METADATA["tree"]["summary"]
        self.assertIn("Guide_Trees", text)
        self.assertIn("names no file", text)

    def test_it_runs_at_once_even_while_commands_wait(self):
        for args in ([], ["help"], ["--register-only"]):
            for source in ("typed", "script", "portal"):
                self.assertEqual(Command_Engine.queue_kind("tree", args, source), Command_Engine.INSTANT)

    def test_a_cache_may_ask_for_its_button_back_by_name(self):
        # sidebar_buttons_to_persist lists "tree", and the Viewer runs `tree --register-only` for it.
        self.assertTrue(Command_Declarations.registers_console_button("tree"))

    def test_the_page_it_opens_is_the_one_the_plugin_declares(self):
        from web_ui.Plugin_Manager import bundled_pages

        page = bundled_pages().page("tree")
        self.assertEqual((page.path, page.client, page.themed, page.csp), ("/tree.html", "tree", True, ()))


if __name__ == "__main__":
    unittest.main()
