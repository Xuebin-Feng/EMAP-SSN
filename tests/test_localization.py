# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Command feedback is English where it is recorded and translatable where it is shown.

The console line on the Viewer canvas is the one place a Message is translated.
The terminal and the command portal (MCP clients, the agent page) keep its
English text. These tests install a stand-in translator that marks every
template, so text that skips or wrongly takes the translation is visible.

utilities.Localization also holds the Qt-free half of the translation
machinery, tested here: the pseudo-language's text, the placeholders a
translation must keep, and reading a catalog.
"""

import ast
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import Command_Engine
from utilities import Localization
from utilities.Localization import (
    CatalogMessage,
    Message,
    display_text,
    fill_ins,
    is_pseudo_translated,
    pseudo_translate,
    read_catalog,
)


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

    def test_a_translation_that_cannot_be_filled_in_shows_the_english_sentence(self):
        message = Message("Found {count} nodes.", count=3)
        for broken in ("{total} Knoten gefunden.", "{0} Knoten gefunden.", "{count Knoten gefunden."):
            with self.subTest(translation=broken):
                previous = Localization.set_translator(lambda template, shown=broken: shown)
                try:
                    self.assertEqual(message.display(), "Found 3 nodes.")
                finally:
                    Localization.set_translator(previous)


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


class PseudoTranslationTests(unittest.TestCase):
    def test_letters_take_accents_vowels_double_and_the_text_is_bracketed(self):
        self.assertEqual(pseudo_translate("Save"), "[Šååṽéé]")
        self.assertEqual(pseudo_translate("OK"), "[ÖÖĶ]")
        self.assertEqual(pseudo_translate("Consistency Check"), "[Çööñšîîšţééñçý Çĥééçķ]")
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.assertFalse(set(letters) & set(pseudo_translate(letters)), "every letter changes")

    def test_what_the_program_fills_in_or_reads_back_stays_as_written(self):
        cases = {
            "{count} nodes saved to {name}.": ["{count}", "{name}"],
            "Mean {value:.2f}, as {label!r} at {0}": ["{value:.2f}", "{label!r}", "{0}"],
            "Use {{braces}} literally": ["{{", "}}"],
            "%n position(s)": ["%n"],
            "%1 of %2 and %L3": ["%1", "%2", "%L3"],
            "%s at %(rate).1f%% done": ["%s", "%(rate).1f", "%%"],
            "<b>Bold</b> and <a href='https://x.org/a'>a link</a>": ["<b>", "</b>", "<a href='https://x.org/a'>", "</a>"],
            "Fish &amp; chips &#169; &#xA9;": ["&amp;", "&#169;", "&#xA9;"],
            "Save && Run": ["&&"],
            "See https://example.org/page now": ["https://example.org/page"],
            "FASTA files (*.fasta *.fa);;All files (*)": ["*.fasta", "*.fa", "*"],
        }
        for text, kept in cases.items():
            with self.subTest(text=text):
                pseudo = pseudo_translate(text)
                for piece in kept:
                    self.assertIn(piece, pseudo)
                self.assertEqual(fill_ins(pseudo), fill_ins(text))
                self.assertTrue(is_pseudo_translated(pseudo))
        template = pseudo_translate("{count} nodes saved to {name}.")
        self.assertEqual(template.format(count=3, name="out.svg"), "[3 ñööđééš šååṽééđ ţöö out.svg.]")

    def test_text_grows_about_as_much_as_a_translation(self):
        labels = [
            "Save & Run", "Export", "Consistency Check", "Similarity Threshold:",
            "Node Color", "Pick", "ON", "OFF", "Save Directories", "Layout Device",
            "Embedding Model", "Alignment Reference", "Show Labels", "Exit",
        ]
        growth = [len(pseudo_translate(label)) - 2 - len(label) for label in labels]
        self.assertTrue(all(extra > 0 for extra in growth), growth)
        self.assertGreaterEqual(sum(growth) / sum(map(len, labels)), 0.3)

    def test_spaces_stay_outside_the_brackets(self):
        self.assertEqual(pseudo_translate("  Name: "), "  [Ñååṁéé:] ")
        self.assertEqual(pseudo_translate("Line one\nLine two"), "[Ļîîñéé ööñéé\nĻîîñéé ţŵöö]")
        for blank in ("", "   ", "\n"):
            self.assertEqual(pseudo_translate(blank), blank)

    def test_only_whole_bracketed_text_counts_as_pseudo_translated(self):
        self.assertTrue(is_pseudo_translated(" [Šååṽéé] "))
        self.assertFalse(is_pseudo_translated("Save"))
        self.assertFalse(is_pseudo_translated("[Šååṽéé]: 3"), "text added after the catalog's")
        self.assertFalse(is_pseudo_translated(""))


class FillInTests(unittest.TestCase):
    def test_lists_the_placeholders_a_translation_must_keep(self):
        self.assertEqual(fill_ins("{total} of {count}"), ["{count}", "{total}"])
        self.assertEqual(fill_ins("{{literal}} and {0:>4}"), ["{0:>4}"])
        self.assertEqual(fill_ins("%n file(s), %1 and %L2"), ["%1", "%L2", "%n"])
        self.assertEqual(fill_ins("%s is 100%% and %(name)d"), ["%(name)d", "%s"])
        self.assertEqual(fill_ins("<b>Bold</b> &amp; *.fasta"), [])


class CatalogReadingTests(unittest.TestCase):
    CATALOG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="de" sourcelanguage="en">
<context>
    <name>Panel</name>
    <message>
        <location filename="../panel.py" line="+7"/>
        <source>Save</source>
        <translation>Speichern</translation>
    </message>
    <message>
        <source>Open</source>
        <comment>a file</comment>
        <translation type="unfinished"></translation>
    </message>
    <message numerus="yes">
        <source>%n file(s)</source>
        <translation>
            <numerusform>%n Datei</numerusform>
            <numerusform>%n Dateien</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>Message</name>
    <message>
        <source>Saved {name} &amp; more &lt;now&gt;.</source>
        <translation type="vanished">{name} gespeichert.</translation>
    </message>
</context>
</TS>
"""

    def test_reads_every_message_with_its_context_comment_forms_and_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "emapssn_de.ts"
            path.write_text(self.CATALOG, encoding="utf-8")
            messages = read_catalog(path)
        self.assertEqual(messages, [
            CatalogMessage("Panel", "Save", "", False, ("Speichern",), ""),
            CatalogMessage("Panel", "Open", "a file", False, ("",), "unfinished"),
            CatalogMessage("Panel", "%n file(s)", "", True, ("%n Datei", "%n Dateien"), ""),
            CatalogMessage("Message", "Saved {name} & more <now>.", "", False,
                           ("{name} gespeichert.",), "vanished"),
        ])
        self.assertEqual(messages[1].key, ("Panel", "Open", "a file"))


if __name__ == "__main__":
    unittest.main()
