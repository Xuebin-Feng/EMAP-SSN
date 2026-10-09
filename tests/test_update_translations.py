# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The catalog update command (src/resources/languages/Update_Translations.py).

It collects every text marked for translation, Message templates included,
into emapssn.ts and each language's catalog, keeps the translations already
made, compiles each language's catalog for the program to load, and refuses
texts no catalog can list. These tests run it, with Qt's lupdate and
lrelease, on a small made-up project in a temporary folder. The last test
runs --check on the real code, so a text marked without running the update
fails the suite.
"""

import ast
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
for folder in (SRC, SRC / "resources" / "languages"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtWidgets import QApplication, QWidget

from desktop.Desktop_App import install_translations
from utilities.Localization import MESSAGE_CONTEXT, Message, read_catalog
import Update_Translations
from Update_Translations import prepare_source, update_catalogs

PANEL = '''from PySide6.QtCore import QCoreApplication, QT_TRANSLATE_NOOP
from PySide6.QtWidgets import QLabel, QWidget

CHOICES = [QT_TRANSLATE_NOOP("Choices", "Circle"), QT_TRANSLATE_NOOP("Choices", "Square")]


class Panel(QWidget):
    def build(self, count, choice, folders):
        self.save = QLabel(self.tr("Save"))
        self.files = QLabel(QCoreApplication.translate("Panel", "%n file(s)", None, count))
        self.shared = QLabel(QCoreApplication.translate("Config", "Shared sentence"))
        self.choice = QLabel(QCoreApplication.translate("Choices", choice))
        self.folders = QLabel(QCoreApplication.translate("Config", "%n folder(s)", None, len(folders)))
'''

COMMANDS = '''from utilities.Localization import Message


def execute(viewer, count, name):
    saved = Message("Saved {count} nodes to {name}.", count=count, name=name)
    joined = Message(
        "Two literals "
        "joined: {name}",
        name=name,
    )
    escaped = Message("Tab\\there, a \\"quote\\", a back\\\\slash and\\na new line {count}", count=count)
    wide = Message("🤖 Agent and α-amylase {name}", name=name)
    nested = Message("Outer {inner}", inner=Message("Inner {count}", count=count))
    removed = Message("Removed %n group(s) from {name}.", n=count, name=name)
    return saved, joined, escaped, wide, nested, removed
'''

EXPECTED_TEXTS = {
    ("Choices", "Circle"), ("Choices", "Square"),
    ("Config", "Shared sentence"), ("Config", "%n folder(s)"),
    ("Panel", "Save"), ("Panel", "%n file(s)"),
    (MESSAGE_CONTEXT, "Saved {count} nodes to {name}."),
    (MESSAGE_CONTEXT, "Two literals joined: {name}"),
    (MESSAGE_CONTEXT, 'Tab\there, a "quote", a back\\slash and\na new line {count}'),
    (MESSAGE_CONTEXT, "🤖 Agent and α-amylase {name}"),
    (MESSAGE_CONTEXT, "Outer {inner}"),
    (MESSAGE_CONTEXT, "Inner {count}"),
    (MESSAGE_CONTEXT, "Removed %n group(s) from {name}."),
}
COUNTED_TEXTS = {"%n file(s)", "%n folder(s)", "Removed %n group(s) from {name}."}


def texts(catalog):
    return {(message.context, message.source) for message in read_catalog(catalog)}


def translate_in(catalog, source, translation):
    """Translate source in a .ts file, as a translator saving in Qt Linguist would."""
    text = catalog.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(<source>" + re.escape(source) + r'</source>\s*<translation) type="unfinished"></translation>'
    )
    text, count = pattern.subn(lambda match: f"{match.group(1)}>{translation}</translation>", text)
    assert count == 1, (source, count)
    catalog.write_text(text, encoding="utf-8")


class PrepareSourceTests(unittest.TestCase):
    def test_each_message_template_is_shown_to_lupdate_on_its_own_line(self):
        text = (
            "def show(viewer, n):\n"
            "    first = Message(\"One {n}\", n=n); second = Localization.Message(\"Two\")\n"
            "    third = Message(\n"
            "        \"Three \"\n"
            "        \"lines {n}\",\n"
            "        n=n,\n"
            "    )\n"
            "    return isinstance(first, Message), Message(\"Outer {x}\", x=Message(\"Inner\"))\n"
        )
        prepared, problems = prepare_source(text, "show.py")
        self.assertEqual(problems, [])
        marker = Update_Translations.message_marker
        self.assertEqual(prepared.splitlines(), [
            "def show(viewer, n):",
            f"    first = Message({marker('One {n}')}\"One {{n}}\", n=n); "
            f"second = Localization.Message({marker('Two')}\"Two\")",
            f"    third = Message({marker('Three lines {n}')}",
            "        \"Three \"",
            "        \"lines {n}\",",
            "        n=n,",
            "    )",
            f"    return isinstance(first, Message), Message({marker('Outer {x}')}\"Outer {{x}}\", "
            f"x=Message({marker('Inner')}\"Inner\"))",
        ])

    def test_markers_escape_what_a_python_string_must(self):
        self.assertEqual(
            Update_Translations.message_marker('Tab\t"q" back\\slash\nnew'),
            'QT_TRANSLATE_NOOP("Message", "Tab\\t\\"q\\" back\\\\slash\\nnew"), ',
        )

    def test_a_counted_template_is_shown_to_lupdate_with_a_count(self):
        marker = Update_Translations.message_marker
        self.assertEqual(
            marker("Removed %n group(s).", counted=True),
            'QCoreApplication.translate("Message", "Removed %n group(s).", None, 0), ',
        )
        prepared, problems = prepare_source(
            'def show(viewer, groups):\n    return Message("Removed %n group(s).", n=len(groups))\n', "show.py"
        )
        self.assertEqual(problems, [])
        self.assertIn(f'Message({marker("Removed %n group(s).", counted=True)}"Removed', prepared)

    def test_the_count_of_a_counted_translate_is_shown_to_lupdate_as_written(self):
        text = (
            "def build(self, folders, rows):\n"
            "    a = QCoreApplication.translate(\"Config\", \"%n 🧬 folder(s)\", None, len(folders))\n"
            "    b = QCoreApplication.translate(\n"
            "        \"Config\", \"%n row(s) in {name}\", None,\n"
            "        sum(\n"
            "            1 for row in rows\n"
            "        ), ).format(name=Message(\"Inner\"))\n"
            "    c = QCoreApplication.translate(\"Config\", \"%n cell(s)\", None, 3)\n"
            "    d = translate(\"Config\", \"%n file(s)\", None, len(folders))\n"
        )
        prepared, problems = prepare_source(text, "build.py")
        self.assertEqual(problems, [])
        # What follows a count stays on its line, so every text keeps its line number.
        self.assertEqual(prepared.splitlines(), [
            "def build(self, folders, rows):",
            "    a = QCoreApplication.translate(\"Config\", \"%n 🧬 folder(s)\", None, 0)",
            "    b = QCoreApplication.translate(",
            "        \"Config\", \"%n row(s) in {name}\", None,",
            "        0",
            "",
            f", ).format(name=Message({Update_Translations.message_marker('Inner')}\"Inner\"))",
            "    c = QCoreApplication.translate(\"Config\", \"%n cell(s)\", None, 3)",
            "    d = translate(\"Config\", \"%n file(s)\", None, 0)",
        ])

    def test_texts_lupdate_would_drop_or_miscount_are_refused(self):
        text = (
            "def build(self, n, values):\n"
            "    self.tr(\"%n file(s)\", n=n)\n"
            "    QCoreApplication.translate(\"Panel\", \"Save\", disambiguation=None)\n"
            "    self.tr(\"%n file(s)\")\n"
            "    QCoreApplication.translate(\"Panel\", \"%n file(s)\", None)\n"
            "    QT_TRANSLATE_NOOP(\"Panel\", \"%n file(s)\")\n"
            "    self.tr(\"%n files\", \"\", n)\n"
            "    QCoreApplication.translate(\"Panel\", \"Step %n\", None, n)\n"
            "    Message(\"Removed %n group(s).\", count=n)\n"
            "    Message(\"Removed %n groups.\", n=n)\n"
            "    Message(\"Removed %n group(s).\", **values)\n"
        )
        _, problems = prepare_source(text, "build.py")
        self.assertEqual([problem.split(":")[1] for problem in problems],
                         ["2", "3", "4", "5", "6", "7", "8", "9", "10"])
        self.assertIn("names its arguments", problems[0])
        self.assertIn("but no count", problems[2])
        self.assertIn("only Desktop_App's translate() shows its English plural", problems[5])
        self.assertIn("without an English plural ending", problems[6])
        self.assertIn("needs the count as n=", problems[7])

    def test_a_text_whose_lines_a_backslash_joins_is_refused(self):
        text = (
            'def build(self, value):\n'
            '    translate("Panel", """First line \\\n'
            'same line""")\n'
            '    Message("Saved \\\n'
            'here.")\n'
            '    translate("Panel", """A real backslash \\\\\n'
            'and a new line""")\n'
            '    translate("Panel", "Joined "  \\\n'
            '              "outside the text")\n'
        )
        _, problems = prepare_source(text, "join.py")
        self.assertEqual([problem.split(":")[1] for problem in problems], ["2", "4"])
        self.assertTrue(all("backslash" in problem for problem in problems))

    def test_texts_filled_in_before_translation_are_refused(self):
        text = (
            "def build(self, n, value, choice):\n"
            "    self.tr(f\"Built {n}\")\n"
            "    self.tr(\"Saved %d\" % n)\n"
            "    self.tr(\"Saved {}\".format(n))\n"
            "    QCoreApplication.translate(\"Panel\", \"Saved \" + value)\n"
            "    QT_TRANSLATE_NOOP(\"Panel\", f\"Shape {n}\")\n"
            "    Message(f\"Saved {n}\")\n"
            "    Message(value)\n"
        )
        _, problems = prepare_source(text, "build.py")
        self.assertEqual([problem.split(":")[1] for problem in problems], ["2", "3", "4", "5", "6", "7", "8"])
        self.assertTrue(all(problem.startswith("build.py:") for problem in problems))

    def test_plain_marked_texts_and_other_translate_methods_pass(self):
        text = (
            "def build(self, choice, table):\n"
            "    self.tr(\"Save\")\n"
            "    QCoreApplication.translate(\"Panel\", \"%n file(s)\", None, 3)\n"
            "    QCoreApplication.translate(\"Choices\", choice)\n"
            "    \"abc\".translate(table)\n"
            "    b\"abc\".translate(table, b\"a\")\n"
        )
        prepared, problems = prepare_source(text, "fine.py")
        self.assertEqual((prepared, problems), (text, []))

    def test_a_file_that_is_not_python_is_reported_not_raised(self):
        _, problems = prepare_source("def broken(:\n", "broken.py")
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("broken.py:1: not valid Python"))


class UpdateCommandTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.src = Path(folder.name) / "src"
        (self.src / "commands").mkdir(parents=True)
        self.translations = self.src / "resources" / "languages"
        self.translations.mkdir(parents=True)
        (self.src / "panel.py").write_text(PANEL, encoding="utf-8")
        (self.src / "commands" / "demo.py").write_text(COMMANDS, encoding="utf-8")
        self.lines = []

    def update(self, **options):
        self.lines.clear()
        return update_catalogs(self.src, self.translations, report=self.lines.append, **options)

    def snapshot(self):
        return {path: path.read_bytes() for path in self.src.rglob("*") if path.is_file()}

    def test_collects_marked_texts_and_message_templates_without_touching_the_code(self):
        code = self.snapshot()
        self.assertEqual(self.update(), 0, self.lines)
        self.assertEqual(texts(self.translations / "emapssn.ts"), EXPECTED_TEXTS)
        numerus = {message.source for message in read_catalog(self.translations / "emapssn.ts") if message.numerus}
        self.assertEqual(numerus, COUNTED_TEXTS)
        self.assertEqual({path: data for path, data in self.snapshot().items() if path.suffix == ".py"}, code)
        self.assertIn("Updated emapssn.ts: 13 texts, 7 of them console messages.", self.lines)
        self.assertNotIn("location", (self.translations / "emapssn.ts").read_text(encoding="utf-8"))

    def test_a_language_keeps_its_translations_through_updates(self):
        self.assertEqual(self.update(add=["de"]), 0, self.lines)
        german = self.translations / "emapssn_de.ts"
        self.assertIn('language="de"', german.read_text(encoding="utf-8"))
        self.assertTrue(german.with_suffix(".qm").is_file())
        self.assertEqual(texts(german), EXPECTED_TEXTS)
        translate_in(german, "Save", "Speichern")
        translate_in(german, "Saved {count} nodes to {name}.", "{count} Knoten in {name} gespeichert.")
        self.assertEqual(self.update(), 0, self.lines)
        self.assertIn("emapssn_de.ts: 2 of 13 texts translated.", self.lines)

        # A text that leaves the code leaves the list; its translation is kept.
        panel = self.src / "panel.py"
        panel.write_text(panel.read_text(encoding="utf-8").replace('self.tr("Save")', 'self.tr("Store")'),
                         encoding="utf-8")
        self.assertEqual(self.update(), 0, self.lines)
        self.assertIn(("Panel", "Store"), texts(self.translations / "emapssn.ts"))
        self.assertNotIn(("Panel", "Save"), texts(self.translations / "emapssn.ts"))
        kept = {message.source: message for message in read_catalog(german)}
        self.assertEqual(kept["Save"].translations, ("Speichern",))
        self.assertIn(kept["Save"].status, ("vanished", "obsolete"))

        # The compiled catalog is what the program loads.
        app = QApplication.instance() or QApplication([])
        installed = install_translations(app, "de", catalog_dir=self.translations)
        self.addCleanup(installed.remove)
        message = Message("Saved {count} nodes to {name}.", count=3, name="a.svg")
        self.assertEqual(message.display(), "3 Knoten in a.svg gespeichert.")

    def test_check_finds_a_stale_template_and_changes_nothing(self):
        self.assertEqual(self.update(add=["de"]), 0, self.lines)
        self.assertEqual(self.update(check=True), 0, self.lines)
        self.assertIn("emapssn.ts lists the code's 13 texts, 7 of them console messages.", self.lines)
        commands = self.src / "commands" / "demo.py"
        commands.write_text(
            commands.read_text(encoding="utf-8") + '\n\ndef later():\n    return Message("A new text.")\n',
            encoding="utf-8",
        )
        before = self.snapshot()
        self.assertEqual(self.update(check=True), 1)
        self.assertIn(f"emapssn.ts lacks [{MESSAGE_CONTEXT}] 'A new text.': run the update.", self.lines)
        self.assertEqual(self.snapshot(), before)

    def test_check_finds_a_catalog_translated_but_not_compiled(self):
        self.assertEqual(self.update(add=["de"]), 0, self.lines)
        translate_in(self.translations / "emapssn_de.ts", "Save", "Speichern")
        self.assertEqual(self.update(check=True), 1)
        self.assertIn("emapssn_de.qm does not match emapssn_de.ts: run the update.", self.lines)
        self.assertEqual(self.update(), 0, self.lines)
        self.assertEqual(self.update(check=True), 0, self.lines)

    def test_a_translation_that_loses_a_placeholder_is_reported(self):
        self.assertEqual(self.update(add=["de"]), 0, self.lines)
        translate_in(self.translations / "emapssn_de.ts", "Saved {count} nodes to {name}.", "Knoten gespeichert.")
        self.assertEqual(self.update(), 1)
        self.assertTrue(any("'Knoten gespeichert.'" in line and "['{count}', '{name}']" in line
                            for line in self.lines), self.lines)

    def test_texts_no_catalog_can_list_fail_the_update(self):
        (self.src / "late.py").write_text(
            "def late(self, n):\n    return self.tr(f\"Built {n}\")\n", encoding="utf-8"
        )
        self.assertEqual(self.update(), 1)
        self.assertTrue(any(line.startswith("late.py:2: tr() gets a text that is already filled in")
                            for line in self.lines), self.lines)

    def test_the_commands_own_folder_is_not_collected(self):
        tool = 'def tool(self, n):\n    return self.tr(f"Tool {n}"), self.tr("Tool text")\n'
        (self.translations / "tool.py").write_text(tool, encoding="utf-8")
        self.assertEqual(self.update(), 0, self.lines)
        self.assertEqual(texts(self.translations / "emapssn.ts"), EXPECTED_TEXTS)
        (self.src / "tool.py").write_text(tool, encoding="utf-8")
        self.assertEqual(self.update(), 1, "the same file elsewhere is collected and checked")

    def test_a_language_must_be_a_language_code(self):
        for language in ("german", "DE", "de-DE", "../de"):
            with self.subTest(language=language), self.assertRaises(ValueError):
                self.update(add=[language])
        self.assertEqual(sorted(path.name for path in self.translations.iterdir()), [])

    def test_add_and_check_cannot_be_combined(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            Update_Translations.main(["--check", "--add", "de"])


class RepositoryCatalogTests(unittest.TestCase):
    def test_the_catalogs_list_every_text_the_code_marks(self):
        lines = []
        status = update_catalogs(check=True, report=lines.append)
        self.assertEqual(status, 0, "Run python src/resources/languages/Update_Translations.py:\n" + "\n".join(lines))

    def test_the_catalog_lists_each_marked_text_as_python_reads_it(self):
        # lupdate parses the code itself. A text it reads otherwise than Python,
        # as with an escape or a joined line, would be listed but never found.
        listed = {message.key for message in read_catalog(Update_Translations.LANGUAGES_DIR / "emapssn.ts")}

        def plain(node):
            return isinstance(node, ast.Constant) and isinstance(node.value, str)

        missing, checked = [], 0
        for path in sorted(SRC.rglob("*.py")):
            if Update_Translations.LANGUAGES_DIR in path.parents:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = Update_Translations._called_name(node)
                args = node.args
                if name in ("translate", "QT_TRANSLATE_NOOP") and len(args) >= 2 and plain(args[0]) and plain(args[1]):
                    comment = args[2].value if len(args) > 2 and plain(args[2]) else ""
                    key = (args[0].value, args[1].value, comment)
                elif name == "Message" and args and plain(args[0]):
                    key = (MESSAGE_CONTEXT, args[0].value, "")
                else:
                    continue
                checked += 1
                if key not in listed:
                    missing.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}: {key[1][:60]!r}")
        self.assertGreater(checked, 500)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
