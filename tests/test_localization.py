# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Command feedback is English where it is recorded and translatable where it is shown.

The console line on the Viewer canvas is the one place a Message is translated.
The terminal and the command portal (MCP clients, the agent page) keep its
English text. These tests install a stand-in translator that marks every
template, so text that skips or wrongly takes the translation is visible.
"""

import ast
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Command_Engine
from utilities import Localization
from utilities.Localization import Message, display_text


def mark(template):
    """A stand-in translation that keeps the template's placeholders."""
    return f"«{template}»"


class MessageTests(unittest.TestCase):
    def test_message_is_english_until_displayed(self):
        message = Message("Found {count} nodes.", count=3)
        self.assertEqual(str(message), "Found 3 nodes.")
        self.assertEqual(message.display(), "Found 3 nodes.")
        previous = Localization.set_translator(mark)
        self.addCleanup(Localization.set_translator, previous)
        self.assertEqual(str(message), "Found 3 nodes.")
        self.assertEqual(message.display(), "«Found 3 nodes.»")
        self.assertEqual(f"{message}", "Found 3 nodes.")

    def test_set_translator_returns_the_one_it_replaces(self):
        previous = Localization.set_translator(mark)
        try:
            self.assertIs(Localization.set_translator(None), mark)
        finally:
            Localization.set_translator(previous)

    def test_display_text_leaves_plain_text_alone(self):
        previous = Localization.set_translator(mark)
        self.addCleanup(Localization.set_translator, previous)
        self.assertEqual(display_text("Already formatted."), "Already formatted.")
        self.assertEqual(display_text(Message("Done.")), "«Done.»")


class ConsoleLineTests(unittest.TestCase):
    def setUp(self):
        self.viewer = SimpleNamespace(
            console_text=SimpleNamespace(text=""),
            update_console_background=mock.Mock(),
        )
        previous = Localization.set_translator(mark)
        self.addCleanup(Localization.set_translator, previous)

    def test_show_status_translates_the_console_line(self):
        Command_Engine.show_status(self.viewer, Message("Found {count} nodes.", count=3))
        self.assertEqual(self.viewer.console_text.text, "«Found 3 nodes.»")
        Command_Engine.show_status(self.viewer, "Plain text.")
        self.assertEqual(self.viewer.console_text.text, "Plain text.")

    def test_print_help_keeps_the_terminal_and_the_record_english(self):
        message = Message("Saved {name}.\nDetails follow.", name="out.svg")
        terminal = io.StringIO()
        with mock.patch("Viewer_Command_Portal.report") as report, \
                contextlib.redirect_stdout(terminal):
            Command_Engine.print_help(self.viewer, message)
        self.assertEqual(self.viewer.console_text.text, "«Saved out.svg.")
        self.viewer.update_console_background.assert_called_once()
        self.assertIn("Saved out.svg.\nDetails follow.", terminal.getvalue())
        self.assertNotIn("«", terminal.getvalue())
        self.assertEqual(
            str(report.call_args.kwargs["message"]), "Saved out.svg.\nDetails follow."
        )

    def test_command_outcomes_record_english(self):
        with mock.patch("Viewer_Command_Portal.report") as report:
            Command_Engine.command_failed(
                self.viewer, Message("No node named {name}.", name="x")
            )
            Command_Engine.command_succeeded(
                self.viewer, Message("Selected {count} nodes.", count=2)
            )
        failed, succeeded = report.call_args_list
        self.assertEqual(failed.args[1], "No node named x.")
        self.assertEqual(str(succeeded.args[1]), "Selected 2 nodes.")


def console_line_writes(tree):
    """Yield (function name, line) for every assignment to a console_text's text."""
    def writes_console_line(target):
        if not (isinstance(target, ast.Attribute) and target.attr == "text"):
            return False
        owner = target.value
        return (
            (isinstance(owner, ast.Attribute) and owner.attr == "console_text")
            or (isinstance(owner, ast.Name) and owner.id == "console_text")
        )

    def visit(node, function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        if any(writes_console_line(target) for target in targets):
            yield function, node.lineno
        for child in ast.iter_child_nodes(node):
            yield from visit(child, function)

    yield from visit(tree, None)


class ConsoleLineWriteGuardTests(unittest.TestCase):
    def test_only_show_status_writes_the_console_line(self):
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for function, line in console_line_writes(tree):
                if path.name == "Command_Engine.py" and function == "show_status":
                    continue
                offenders.append(f"{path.relative_to(SRC).as_posix()}:{line}")
        self.assertEqual(
            offenders,
            [],
            "Write the Viewer's console line through Command_Engine.show_status, "
            "the one place its text is translated.",
        )


if __name__ == "__main__":
    unittest.main()
