# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The console command (commands/console.py): it opens the Console window through the Viewer and reports what happened, refuses arguments and the cases where no window can open, prints its own help, runs at once even while commands wait, and is a catalogued command that the VR viewer can load without Qt."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Command_Engine  # noqa: E402
from commands import console as console_command  # noqa: E402
from desktop.Command_Metadata import AGENT_POLICY, COMMAND_METADATA, agent_policy  # noqa: E402
from tests.command_fixtures import run_command  # noqa: E402


def viewer(result):
    """A Viewer whose open_console_window gives ``result``."""
    return SimpleNamespace(
        console_text=SimpleNamespace(text=""),
        open_console_window=mock.Mock(return_value=result),
    )


class RunTests(unittest.TestCase):
    def test_it_opens_the_console_and_says_so(self):
        fake = viewer("opened")
        succeeded, failed, printed = run_command(console_command, fake, [])
        fake.open_console_window.assert_called_once_with()
        succeeded.assert_called_once()
        self.assertEqual(succeeded.call_args.args[1], "Opened the Console.")
        failed.assert_not_called()
        self.assertEqual(fake.console_text.text, "Opened the Console.")

    def test_a_viewer_with_no_window_says_where_the_output_is_instead(self):
        succeeded, failed, printed = run_command(console_command, viewer("no_window"), [])
        succeeded.assert_not_called()
        self.assertIn("shows none", failed.call_args.args[1])
        self.assertIn("log file", failed.call_args.args[1])
        self.assertIn("shows none", printed)

    def test_a_viewer_started_with_console_says_its_output_is_in_the_terminal(self):
        succeeded, failed, _ = run_command(console_command, viewer("not_kept"), [])
        succeeded.assert_not_called()
        self.assertIn("terminal it was started from", failed.call_args.args[1])

    def test_a_viewer_without_the_window_at_all_is_refused_plainly(self):
        succeeded, failed, _ = run_command(console_command, SimpleNamespace(console_text=SimpleNamespace(text="")), [])
        succeeded.assert_not_called()
        self.assertIn("no window of that kind", failed.call_args.args[1])

    def test_arguments_are_refused_with_the_usage_and_nothing_opens(self):
        fake = viewer("opened")
        succeeded, failed, printed = run_command(console_command, fake, ["clear"])
        fake.open_console_window.assert_not_called()
        succeeded.assert_not_called()
        self.assertIn("takes no arguments, not 'clear'", failed.call_args.args[1])
        self.assertIn("Usage: console", failed.call_args.args[1])
        self.assertIn("Usage: console", printed)

    def test_help_prints_in_every_spelling_and_opens_nothing(self):
        for word in ("help", "-h", "--help", "HELP"):
            with self.subTest(word=word):
                fake = viewer("opened")
                succeeded, failed, printed = run_command(console_command, fake, [word])
                fake.open_console_window.assert_not_called()
                failed.assert_not_called()
                self.assertEqual(succeeded.call_args.args[1], "Help information printed to the Console.")
                self.assertIn("Usage: console", printed)
                self.assertIn("Examples:", printed)


class CatalogTests(unittest.TestCase):
    def test_it_is_catalogued_and_an_agent_reply_may_not_run_it(self):
        # The Console is a window for the person at the Viewer; the Agent prompt (a small model's context) has
        # no item for it and names it among the refused commands.
        self.assertIn("console", COMMAND_METADATA)
        self.assertTrue(COMMAND_METADATA["console"]["summary"])
        self.assertEqual(agent_policy("console"), "refuse")
        self.assertEqual(AGENT_POLICY["console"]["default"], "refuse")

    def test_it_runs_at_once_even_while_commands_wait(self):
        for args in ([], ["help"]):
            for source in ("typed", "script", "portal"):
                self.assertEqual(Command_Engine.queue_kind("console", args, source), Command_Engine.INSTANT)

    def test_the_vr_viewer_can_load_it_because_it_imports_no_qt(self):
        tree = ast.parse((SRC_DIR / "commands" / "console.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])
        self.assertTrue(imported <= {"Command_Engine", "utilities"}, imported)


if __name__ == "__main__":
    unittest.main()
