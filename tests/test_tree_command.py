# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tree command (commands/tree.py): it opens Tree View and adds its Console button, and reports what happened.

It takes no argument but help and --register-only (which brings the Console button back when a session
loads), says why when the page cannot open (already open, a Viewer with no window), runs at once even while
commands wait, and is a catalogued command whose agent reply is refused.
"""

from __future__ import annotations

from pathlib import Path
import sys
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


def viewer(opens=True):
    """A Viewer whose open_page answers ``opens``, and which has a Console to add a button to."""
    return SimpleNamespace(
        console_text=SimpleNamespace(text=""),
        open_page=mock.Mock(return_value=opens),
        add_console_button=mock.Mock(),
    )


class OpeningTests(unittest.TestCase):
    def test_it_opens_the_page_and_says_so(self):
        fake = viewer()
        succeeded, failed, _printed = run_command(tree_command, fake, [])
        fake.open_page.assert_called_once_with("tree")
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Opened Tree View.")

    def test_it_adds_the_console_button_and_keeps_it_for_the_session(self):
        fake = viewer()
        run_command(tree_command, fake, [])
        fake.add_console_button.assert_called_once()
        self.assertEqual(fake.add_console_button.call_args.kwargs["name"], "treeViewBtn")
        self.assertEqual(fake.sidebar_buttons_to_persist, ["tree"])

    def test_a_command_from_the_portal_never_waits_for_someone_to_close_a_dialog(self):
        fake = viewer()
        with mock.patch.object(tree_command, "CURRENT", SimpleNamespace(get=lambda: object())):
            run_command(tree_command, fake, [])
        fake.open_page.assert_called_once_with("tree", show_existing_dialog=False)

    def test_a_page_that_is_open_already_is_reported_as_such(self):
        fake = viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=True):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        succeeded.assert_not_called()
        self.assertEqual(failed.call_args.args[1], "Tree View is already open in your browser.")

    def test_a_viewer_with_no_window_says_where_the_page_is_served(self):
        fake = viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=False), \
                mock.patch.object(tree_command, "is_headless", return_value=True), \
                mock.patch.object(tree_command, "page_address", return_value="http://127.0.0.1:1/tree.html") as address:
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        address.assert_called_once_with(fake, "/tree.html")
        failed.assert_not_called()
        self.assertIn("http://127.0.0.1:1/tree.html", str(succeeded.call_args.args[1]))
        self.assertIn("no window", str(succeeded.call_args.args[1]))

    def test_a_page_that_cannot_open_for_another_reason_fails_plainly(self):
        fake = viewer(opens=False)
        with mock.patch.object(tree_command, "page_is_open", return_value=False), \
                mock.patch.object(tree_command, "is_headless", return_value=False):
            succeeded, failed, _printed = run_command(tree_command, fake, [])
        succeeded.assert_not_called()
        self.assertEqual(failed.call_args.args[1], "Could not open Tree View.")


class ArgumentTests(unittest.TestCase):
    def test_register_only_adds_the_button_and_opens_nothing(self):
        fake = viewer()
        succeeded, failed, _printed = run_command(tree_command, fake, ["--register-only"])
        fake.open_page.assert_not_called()
        fake.add_console_button.assert_called_once()
        failed.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Registered Tree View.")

    def test_arguments_are_refused_with_the_usage_and_nothing_opens(self):
        for args in (["file.nex"], ["--register-only", "x"], ["open"]):
            with self.subTest(args=args):
                fake = viewer()
                succeeded, failed, printed = run_command(tree_command, fake, args)
                fake.open_page.assert_not_called()
                fake.add_console_button.assert_not_called()
                succeeded.assert_not_called()
                self.assertIn(f"takes no arguments, not '{args[0]}'", str(failed.call_args.args[1]))
                self.assertIn("Usage: tree", str(failed.call_args.args[1]))
                self.assertIn("Usage: tree", printed)

    def test_the_page_names_no_file_so_no_argument_can(self):
        fake = viewer()
        run_command(tree_command, fake, ["C:/secret.nex"])
        fake.open_page.assert_not_called()

    def test_help_prints_in_every_spelling_and_opens_nothing(self):
        for word in ("help", "-h", "--help", "HELP"):
            with self.subTest(word=word):
                fake = viewer()
                succeeded, failed, printed = run_command(tree_command, fake, [word])
                fake.open_page.assert_not_called()
                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], "Help information printed to the Console.")
                self.assertIn("Usage: tree", printed)
                self.assertIn("Guide_Trees", printed)


class DeclarationTests(unittest.TestCase):
    def test_it_is_catalogued_and_an_agent_reply_may_not_open_it(self):
        # A window for the person at the Viewer, as the Console is: the Agent's instructions, which the size
        # ceiling of test_agent_prompt leaves no room in, have no item for it.
        self.assertIn("tree", COMMAND_METADATA)
        entry = COMMAND_METADATA["tree"]
        self.assertTrue(entry["summary"])
        self.assertTrue(entry["opens_interface"])
        self.assertFalse(entry["writes_files"])
        self.assertEqual(agent_policy("tree"), "refuse")
        self.assertEqual(AGENT_POLICY["tree"]["default"], "refuse")

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
