# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Loading translations (Desktop_App section 7): what a window shows in a language.

A window shows its text in the language installed before it is built.
English installs no catalog. The pseudo-language is made from the template
catalog, so a text the catalog doesn't list stays English, as it would in a
real language. A real language loads Qt's own catalogs, then EMAP-SSN's,
which wins where both translate a text. The Viewer's console messages come
from the same catalogs, while their recorded text stays English. Windows
mark their text with Desktop_App.translate, which shows a counted text no
catalog translates in its English plural, whatever is installed.

These tests write small catalogs to a temporary folder and compile them with
Qt's lrelease. The last class checks that every window installs the
translations before it builds anything that shows text.
"""

import ast
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
for folder in (SRC, SRC / "resources" / "languages"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtCore import QCoreApplication, QLibraryInfo
from PySide6.QtGui import QRawFont
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QWidget

import Command_Engine
from desktop.Desktop_App import (
    DESKTOP_FONT_DIR,
    PSEUDO_LANGUAGE,
    PSEUDO_TRANSLATION_VARIABLE,
    UI_BOLD_FILE,
    UI_REGULAR_FILE,
    install_translations,
    installed_language,
    startup_language,
    translate,
)
from utilities import Localization
from utilities.Localization import MESSAGE_CONTEXT, Message, pseudo_translate
import Update_Translations

SAVED = "Saved {count} nodes to {name}."
REMOVED = "Removed %n group(s) from {name}."

TEMPLATE = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" sourcelanguage="en">
<context>
    <name>Config</name>
    <message>
        <source>Shared sentence</source>
        <translation type="unfinished"></translation>
    </message>
</context>
<context>
    <name>{MESSAGE_CONTEXT}</name>
    <message>
        <source>{SAVED}</source>
        <translation type="unfinished"></translation>
    </message>
    <message numerus="yes">
        <source>{REMOVED}</source>
        <translation type="unfinished">
            <numerusform></numerusform>
        </translation>
    </message>
</context>
<context>
    <name>Panel</name>
    <message>
        <source>Save</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <source>Open</source>
        <comment>a file</comment>
        <translation type="unfinished"></translation>
    </message>
    <message numerus="yes">
        <source>%n file(s)</source>
        <translation type="unfinished">
            <numerusform></numerusform>
        </translation>
    </message>
</context>
</TS>
"""

GERMAN = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="de" sourcelanguage="en">
<context>
    <name>{MESSAGE_CONTEXT}</name>
    <message>
        <source>{SAVED}</source>
        <translation>{{count}} Knoten in {{name}} gespeichert.</translation>
    </message>
    <message numerus="yes">
        <source>{REMOVED}</source>
        <translation>
            <numerusform>%n Gruppe aus {{name}} entfernt.</numerusform>
            <numerusform>%n Gruppen aus {{name}} entfernt.</numerusform>
        </translation>
    </message>
</context>
<context>
    <name>Panel</name>
    <message>
        <source>Save</source>
        <translation>Speichern</translation>
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
    <name>QPlatformTheme</name>
    <message>
        <source>Cancel</source>
        <translation>Stornieren</translation>
    </message>
</context>
</TS>
"""


class Panel(QWidget):
    """A widget whose tr() looks texts up under the context "Panel"."""


def button_texts(*buttons):
    """The texts of Qt's standard buttons, sorted."""
    box = QDialogButtonBox()
    for button in buttons:
        box.addButton(button)
    return sorted(button.text() for button in box.buttons())


def mark(template, n=-1):
    return f"«{template}»"


class TranslationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.folder_object = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.folder_object.name)
        (cls.folder / "emapssn.ts").write_text(TEMPLATE, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.folder_object.cleanup()

    def setUp(self):
        # Every test starts and ends English, whatever a failure left behind.
        previous = Localization.set_translator(None)
        self.addCleanup(Localization.set_translator, previous)
        self.addCleanup(install_translations, self.app, None)

    def assert_english(self):
        self.assertIsNone(installed_language())
        panel = Panel()
        self.assertEqual(panel.tr("Save"), "Save")
        self.assertEqual(QCoreApplication.translate("Config", "Shared sentence"), "Shared sentence")
        self.assertEqual(Message(SAVED, count=3, name="a.svg").display(), "Saved 3 nodes to a.svg.")
        self.assertEqual(button_texts(QDialogButtonBox.StandardButton.Cancel), ["Cancel"])
        self.assertEqual([translate("Panel", "%n file(s)", None, count) for count in (1, 3)], ["1 file", "3 files"])
        self.assertEqual(Message(REMOVED, n=1, name="a.h5").display(), "Removed 1 group from a.h5.")


class StartupLanguageTests(unittest.TestCase):
    def test_english_unless_the_pseudo_language_is_asked_for(self):
        self.assertIsNone(startup_language({}))
        for value in ("1", "true", "Yes", " on "):
            self.assertEqual(startup_language({PSEUDO_TRANSLATION_VARIABLE: value}), PSEUDO_LANGUAGE)
        for value in ("", "0", "false", "no", "de"):
            self.assertIsNone(startup_language({PSEUDO_TRANSLATION_VARIABLE: value}))


class EnglishTests(TranslationTestCase):
    def test_english_installs_no_catalog(self):
        for language in (None, "", "en"):
            with self.subTest(language=language):
                self.assertIsNone(install_translations(self.app, language, catalog_dir=self.folder))
                self.assert_english()

    def test_english_takes_out_a_language_installed_before(self):
        install_translations(self.app, PSEUDO_LANGUAGE, catalog_dir=self.folder)
        install_translations(self.app, None, catalog_dir=self.folder)
        self.assert_english()


class PseudoLanguageTests(TranslationTestCase):
    def setUp(self):
        super().setUp()
        self.installed = install_translations(self.app, PSEUDO_LANGUAGE, catalog_dir=self.folder)

    def test_every_catalog_text_is_pseudo_translated(self):
        panel = Panel()
        self.assertEqual(panel.tr("Save"), "[Šååṽéé]")
        self.assertEqual(panel.tr("Open", "a file"), pseudo_translate("Open"))
        self.assertEqual(panel.tr("%n file(s)", "", 3), "[3 ƒîîļéé(š)]")
        self.assertEqual(
            QCoreApplication.translate("Config", "Shared sentence"), pseudo_translate("Shared sentence")
        )

    def test_text_the_catalog_does_not_list_stays_english(self):
        panel = Panel()
        count = 3
        self.assertEqual(panel.tr("Not listed"), "Not listed")
        self.assertEqual(panel.tr(f"Built {count}"), "Built 3")
        self.assertEqual(panel.tr("Open"), "Open", "the catalog lists Open only as 'a file'")
        self.assertEqual(QCoreApplication.translate("Elsewhere", "Save"), "Save")
        # Qt's own text belongs to Qt's catalogs, which no pseudo-language has.
        self.assertEqual(button_texts(QDialogButtonBox.StandardButton.Cancel), ["Cancel"])

    def test_console_messages_are_shown_translated_and_recorded_in_english(self):
        message = Message(SAVED, count=3, name="out.svg")
        self.assertEqual(message.display(), "[Šååṽééđ 3 ñööđééš ţöö out.svg.]")
        self.assertEqual(str(message), "Saved 3 nodes to out.svg.")
        viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
        Command_Engine.show_status(viewer, message)
        self.assertEqual(viewer.console_text.text, "[Šååṽééđ 3 ñööđééš ţöö out.svg.]")
        Command_Engine.show_status(viewer, Message("A template the catalog lacks."))
        self.assertEqual(viewer.console_text.text, "A template the catalog lacks.")
        counted = Message(REMOVED, n=2, name="a.h5")
        self.assertEqual(counted.display(), pseudo_translate(REMOVED).replace("%n", "2").format(name="a.h5"))
        self.assertEqual(str(counted), "Removed 2 groups from a.h5.")
        self.assertEqual(translate("Panel", "%n file(s)", None, 1), "[1 ƒîîļéé(š)]")

    def test_remove_restores_english_and_the_previous_message_translator(self):
        self.installed.remove()
        self.assert_english()
        Localization.set_translator(mark)
        installed = install_translations(self.app, PSEUDO_LANGUAGE, catalog_dir=self.folder)
        installed.remove()
        self.assertEqual(Localization.translate("Done."), "«Done.»")

    def test_a_second_install_replaces_the_first(self):
        second = install_translations(self.app, PSEUDO_LANGUAGE, catalog_dir=self.folder)
        self.assertIsNot(second, self.installed)
        self.assertEqual(installed_language(), PSEUDO_LANGUAGE)
        self.assertEqual(Panel().tr("Save"), "[Šååṽéé]")
        second.remove()
        self.assert_english()


class LanguageTests(TranslationTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        catalog = cls.folder / "emapssn_de.ts"
        catalog.write_text(GERMAN, encoding="utf-8")
        Update_Translations.run_qt_tool("lrelease", [catalog, "-qm", catalog.with_suffix(".qm")])
        qt_folder = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
        cls.qt_has_german = (qt_folder / "qtbase_de.qm").is_file()

    def test_texts_and_console_messages_come_from_the_languages_catalog(self):
        installed = install_translations(self.app, "de", catalog_dir=self.folder)
        self.assertEqual((installed.language, installed_language()), ("de", "de"))
        panel = Panel()
        self.assertEqual(panel.tr("Save"), "Speichern")
        self.assertEqual(panel.tr("%n file(s)", "", 1), "1 Datei")
        self.assertEqual(panel.tr("%n file(s)", "", 3), "3 Dateien")
        self.assertEqual(panel.tr("Not translated yet"), "Not translated yet")
        message = Message(SAVED, count=3, name="a.svg")
        self.assertEqual(message.display(), "3 Knoten in a.svg gespeichert.")
        self.assertEqual(str(message), "Saved 3 nodes to a.svg.")
        installed.remove()
        self.assert_english()

    def test_counted_texts_take_the_languages_plural_forms_or_else_english_ones(self):
        install_translations(self.app, "de", catalog_dir=self.folder)
        self.assertEqual([translate("Panel", "%n file(s)", None, count) for count in (1, 3)], ["1 Datei", "3 Dateien"])
        self.assertEqual(translate("Panel", "%n folder(s)", None, 3), "3 folders", "German lacks it: English")
        self.assertEqual(translate("Panel", "Save"), "Speichern")
        for count, german in ((1, "1 Gruppe aus a.h5 entfernt."), (4, "4 Gruppen aus a.h5 entfernt.")):
            with self.subTest(count=count):
                self.assertEqual(Message(REMOVED, n=count, name="a.h5").display(), german)
        self.assertEqual(str(Message(REMOVED, n=1, name="a.h5")), "Removed 1 group from a.h5.")

    def test_qts_own_text_comes_from_qts_catalog_and_ours_wins_where_both_translate(self):
        if not self.qt_has_german:
            self.skipTest("this PySide6 has no German catalog of its own")
        install_translations(self.app, "de", catalog_dir=self.folder)
        buttons = button_texts(QDialogButtonBox.StandardButton.Ok, QDialogButtonBox.StandardButton.Close)
        self.assertEqual(buttons, ["OK", "Schließen"])
        self.assertEqual(button_texts(QDialogButtonBox.StandardButton.Cancel), ["Stornieren"])

    def test_a_language_without_a_catalog_installs_nothing(self):
        with self.assertRaisesRegex(LookupError, "'fr'"):
            install_translations(self.app, "fr", catalog_dir=self.folder)
        self.assert_english()


class PseudoLetterFontTests(unittest.TestCase):
    def test_the_bundled_ui_font_draws_every_pseudo_letter(self):
        QApplication.instance() or QApplication([])
        letters = sorted(set(pseudo_translate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")) - {"[", "]"})
        for face in (UI_REGULAR_FILE, UI_BOLD_FILE):
            with self.subTest(face=face):
                font = QRawFont(str(DESKTOP_FONT_DIR / face), 12)
                self.assertTrue(font.isValid())
                # PySide6 misreads a non-ASCII str here, so pass the code point.
                self.assertFalse(font.supportsCharacter(ord("中")), "the check can fail")
                self.assertEqual([letter for letter in letters if not font.supportsCharacter(ord(letter))], [])


def calls_in(tree, name):
    """Line numbers of the calls to name (a function or a method) in tree."""
    def called(node):
        function = node.func
        return getattr(function, "id", None) == name or getattr(function, "attr", None) == name

    return sorted(node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call) and called(node))


class StartupWiringTests(unittest.TestCase):
    """Each window installs translations before it shows any text.

    Windows set their text once, as they are built, so text built before
    the translations are installed would stay English.
    """

    def assert_installed_between(self, tree, before, after):
        installs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "install_translations"
        ]
        self.assertEqual(len(installs), 1)
        self.assertEqual(ast.unparse(installs[0].args[1]), "startup_language()")
        line = installs[0].lineno
        for name in before:
            self.assertTrue(calls_in(tree, name), name)
            self.assertLess(calls_in(tree, name)[0], line, f"{name} must come first")
        for name in after:
            self.assertTrue(calls_in(tree, name), name)
            self.assertGreater(calls_in(tree, name)[0], line, f"{name} must come after")

    def test_config_and_tools_install_right_after_creating_the_application(self):
        for script, window in (("EMAPSSN_Config.py", "ConfigGUI"), ("EMAPSSN_Tools.py", "ToolsGUI")):
            with self.subTest(script=script):
                tree = ast.parse((SRC / script).read_text(encoding="utf-8"))
                self.assert_installed_between(
                    tree, before=["QApplication"],
                    after=["SingleInstanceController", "configure_qt_application_fonts", window],
                )

    def test_viewer_installs_as_soon_as_its_canvas_has_created_the_application(self):
        tree = ast.parse((SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        viewer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainViewer")
        methods = {node.name: node for node in viewer.body if isinstance(node, ast.FunctionDef)}
        # The main window, with the sidebar and its toggle, is built in _build_main_window.
        for widget in ("QMainWindow", "QPushButton"):
            self.assertTrue(calls_in(methods["_build_main_window"], widget), widget)
        self.assert_installed_between(
            methods["__init__"], before=["SceneCanvas", "instance"],
            after=["create_hud", "QLabel", "_build_main_window"],
        )

    def test_the_viewer_takes_its_language_face_before_it_draws_any_text(self):
        # VisPy draws a text in one face, so Chinese needs its face for all of it.
        tree = ast.parse((SRC / "EMAPSSN_Viewer.py").read_text(encoding="utf-8"))
        viewer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MainViewer")
        init = next(node for node in viewer.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        faces = [
            node for node in ast.walk(init)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "vispy_language_face"
        ]
        self.assertEqual(len(faces), 1)
        self.assertEqual(ast.unparse(faces[0].args[0]), "installed_language()")
        line, hud = faces[0].lineno, calls_in(init, "create_hud")[0]
        self.assertLess(calls_in(init, "install_translations")[0], line)
        self.assertLess(line, hud)
        assigned = {
            target.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Assign) and line <= node.lineno < hud
            for target in node.targets
            if isinstance(target, ast.Attribute)
        }
        self.assertLessEqual({"vispy_ui_face", "vispy_monospace_face"}, assigned)


if __name__ == "__main__":
    unittest.main()
