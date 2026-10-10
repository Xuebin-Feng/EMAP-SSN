# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The tool help pages and their translations (src/utilities/Help_Pages.py).

A translation of src/tools/tool_descriptions/<name>.md is <name>.<language>.md
beside it, and its first line names the English page's SHA-256. The Tools
window shows a tab's help in the window's language while that line matches,
and the English page otherwise; the update command lists the translations
that fell behind. These tests use made-up pages in a temporary folder, and
the Tools window on a copy of the real ones.
"""

import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, SRC / "resources" / "languages", ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from PySide6.QtWidgets import QApplication, QTextBrowser

from utilities import Help_Pages
import Update_Translations

# As a Windows checkout writes the English pages: with CRLF.
ENGLISH = "# 🧬 Demo Tool (`Demo.py`)\r\n\r\nThe English help.\r\n"
GERMAN = "# 🧬 Demo-Werkzeug (`Demo.py`)\n\nDie deutsche Hilfe.\n"


class HelpPageTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.page = self.folder / "Demo.md"
        self.page.write_bytes(ENGLISH.encode("utf-8"))

    def translate(self, marker=None, language="de", body=GERMAN):
        path = Help_Pages.translation_path(self.page, language)
        path.write_bytes(((marker or Help_Pages.translation_marker(self.page)) + "\n" + body).encode("utf-8"))
        return path

    def test_a_current_translation_shows_without_its_first_line(self):
        self.translate()
        self.assertEqual(Help_Pages.help_page_text(self.page, "de"), GERMAN)
        english = ENGLISH.replace("\r\n", "\n")
        for language in (None, "", "fr", "pseudo"):
            with self.subTest(language=language):
                self.assertEqual(Help_Pages.help_page_text(self.page, language), english)
        self.assertEqual(Help_Pages.stale_translations(self.folder), [])

    def test_the_english_page_shows_once_it_changes(self):
        translation = self.translate()
        revised = ENGLISH.replace("help.", "help, revised.")
        self.page.write_bytes(revised.encode("utf-8"))
        self.assertEqual(Help_Pages.help_page_text(self.page, "de"), revised.replace("\r\n", "\n"))
        self.assertEqual(Help_Pages.stale_translations(self.folder), [(translation, self.page)])
        self.translate()
        self.assertEqual(Help_Pages.help_page_text(self.page, "de"), GERMAN)

    def test_a_translation_naming_another_page_or_none_is_not_shown(self):
        other = self.folder / "Other.md"
        other.write_bytes(ENGLISH.encode("utf-8"))
        english = ENGLISH.replace("\r\n", "\n")
        for marker in (Help_Pages.translation_marker(other), "<!-- A comment -->", "# No marker"):
            with self.subTest(marker=marker):
                translation = self.translate(marker)
                self.assertEqual(Help_Pages.help_page_text(self.page, "de"), english)
                self.assertEqual(Help_Pages.stale_translations(self.folder), [(translation, self.page)])

    def test_line_endings_leave_the_marker_alone(self):
        marker = Help_Pages.translation_marker(self.page)
        self.assertRegex(marker, r"^<!-- Translation of Demo\.md, sha256 [0-9a-f]{64} -->$")
        self.page.write_bytes(ENGLISH.replace("\r\n", "\n").encode("utf-8"))
        self.assertEqual(Help_Pages.translation_marker(self.page), marker)
        translation = self.translate()
        translation.write_bytes(translation.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(Help_Pages.help_page_text(self.page, "de"), GERMAN)

    def test_a_translation_whose_page_is_gone_is_listed(self):
        translation = self.translate()
        self.page.unlink()
        self.assertEqual(Help_Pages.stale_translations(self.folder), [(translation, self.page)])

    def test_translations_are_told_apart_by_name(self):
        names = {
            "Demo.md": False, "Demo.zh_CN.md": True, "Demo.de.md": True, "Demo.sr_Latn_RS.md": True,
            "Demo.notes.md": False, "Demo.DE.md": False, "Demo.md.bak": False,
        }
        for name, expected in names.items():
            with self.subTest(name=name):
                self.assertEqual(Help_Pages.is_translation(name), expected)

    def test_the_update_lists_the_pages_tools_shows_in_english(self):
        pages = self.folder / "src" / "tools" / "tool_descriptions"
        pages.mkdir(parents=True)
        for name in ("Current", "Behind"):
            (pages / f"{name}.md").write_text(f"# {name}\n", encoding="utf-8")
            marker = Help_Pages.translation_marker(pages / f"{name}.md")
            (pages / f"{name}.zh_CN.md").write_text(marker + "\n# 译文\n", encoding="utf-8")
        (pages / "Behind.md").write_text("# Behind, revised\n", encoding="utf-8")
        (pages / "Gone.zh_CN.md").write_text("# 译文\n", encoding="utf-8")
        lines = Update_Translations.help_page_reports(self.folder / "src")
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(lines[0].startswith("Behind.zh_CN.md was translated from another version of Behind.md"))
        self.assertTrue(lines[0].endswith("\n  " + Help_Pages.translation_marker(pages / "Behind.md")))
        self.assertEqual(lines[1], "Gone.zh_CN.md translates Gone.md, which no longer exists, so Tools never shows it.")
        self.assertEqual(Update_Translations.help_page_reports(self.folder), [])


class ToolsHelpPageTests(unittest.TestCase):
    """The Tools window on a copy of the real help pages."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import EMAPSSN_Tools
        from tests.tools_gui_fixtures import isolated_tools_project

        isolated_tools_project(self)
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        self.pages = folder / "pages"
        shutil.copytree(Help_Pages.HELP_PAGES_DIR, self.pages)
        for patcher in (mock.patch("EMAPSSN_Tools.ResponsiveTextBrowser", QTextBrowser),
                        mock.patch.object(Help_Pages, "HELP_PAGES_DIR", self.pages)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tools = EMAPSSN_Tools

    def shown_help(self, window, language):
        """The help the Embedding MSA tab shows with language installed."""
        index = next(index for index in range(window.tabs.count())
                     if window.tabs.widget(index).property("descriptionKey") == "Embedding_MSA")
        with mock.patch.object(self.tools, "installed_language", return_value=language):
            window.on_tab_changed(index)
        return window.script_desc_text.toPlainText()

    def test_a_tab_shows_its_help_in_the_windows_language(self):
        english = self.pages / "Embedding_MSA.md"
        Help_Pages.translation_path(english, "de").write_text(
            Help_Pages.translation_marker(english) + "\n# Einbettungs-MSA (`Embedding_MSA.py`)\n\nDeutsche Hilfe.\n",
            encoding="utf-8",
        )
        window = self.tools.ToolsGUI()
        self.addCleanup(window.deleteLater)
        german = self.shown_help(window, "de")
        self.assertIn("Deutsche Hilfe.", german)
        self.assertNotIn("Translation of", german)
        self.assertIn("Embedding Multiple Sequence Alignment", self.shown_help(window, None))
        chinese = self.shown_help(window, "zh_CN")
        self.assertIn("嵌入多序列比对", chinese, "the shipped Simplified Chinese page")
        for shown in (self.shown_help(window, None), chinese):
            self.assertNotIn("Deutsche Hilfe.", shown)
        # Once the English page changes, it shows until the translations catch up.
        english.write_text(english.read_text(encoding="utf-8") + "\nOne more line.\n", encoding="utf-8")
        for language in ("de", "zh_CN"):
            with self.subTest(language=language):
                self.assertIn("One more line.", self.shown_help(window, language))

    def test_tool_titles_come_from_the_english_pages(self):
        titles = self.tools.get_tool_titles()
        self.assertEqual(titles["Embedding_MSA.py"], "🧬 Embedding Multiple Sequence Alignment")
        english = self.pages / "Embedding_MSA.md"
        # Listed after the English page, so its heading would win if it counted.
        Help_Pages.translation_path(english, "zh_CN").write_text(
            Help_Pages.translation_marker(english) + "\n# 🧬 嵌入多序列比对 (`Embedding_MSA.py`)\n", encoding="utf-8"
        )
        self.assertEqual(self.tools.get_tool_titles(), titles)


def page_structure(text):
    """What a help page's translation keeps of its English page.

    Its outline (heading levels in order), code spans and blocks, table
    rows, links, math and HTML tags: the words change, nothing else does.
    """
    lines = text.splitlines()
    displayed = re.findall(r"\$\$.*?\$\$", text, re.S)
    inline = re.findall(r"\$[^$\n]+\$", re.sub(r"\$\$.*?\$\$", "", text, flags=re.S))
    return {
        "outline": [len(match.group(1)) for line in lines if (match := re.match(r"(#{1,6}) ", line))],
        "code": sorted(re.findall(r"`[^`\n]+`", text)),
        "code blocks": text.count("```"),
        "table rows": sum(1 for line in lines if line.lstrip().startswith("|")),
        "links": sorted(re.findall(r"\]\(([^)\s]+)\)", text)),
        "math": sorted(displayed) + sorted(inline),
        "markup": sorted(re.findall(r"</?[a-z]+\b[^>]*>", text)),
    }


class TranslatedHelpPageTests(unittest.TestCase):
    """The translated help pages that ship: current, and their English pages' structure."""

    def test_each_translation_keeps_its_pages_structure(self):
        translations = sorted(page for page in Help_Pages.HELP_PAGES_DIR.glob("*.md") if Help_Pages.is_translation(page))
        self.assertTrue(translations)
        self.assertEqual(Help_Pages.stale_translations(), [])
        for translation in translations:
            name, language = translation.name.split(".")[:2]
            english = translation.with_name(f"{name}.md")
            with self.subTest(page=translation.name):
                shown = Help_Pages.help_page_text(english, language)
                self.assertNotEqual(shown, english.read_text(encoding="utf-8"), "Tools shows the translation")
                expected, found = page_structure(english.read_text(encoding="utf-8")), page_structure(shown)
                for part in expected:
                    self.assertEqual(found[part], expected[part], part)

    def test_a_tools_heading_reads_as_its_card_title(self):
        from utilities.Localization import LANGUAGES_DIR, read_catalog

        heading = re.compile(r"^# (.+) \(`([\w.]+\.py)`\)$", re.M)
        for translation in sorted(Help_Pages.HELP_PAGES_DIR.glob("*.md")):
            if not Help_Pages.is_translation(translation):
                continue
            name, language = translation.name.split(".")[:2]
            titles = {message.source: message.translations[0] for message in read_catalog(
                LANGUAGES_DIR / f"emapssn_{language}.ts") if message.context == "Tools" and message.translations}
            english = (Help_Pages.HELP_PAGES_DIR / f"{name}.md").read_text(encoding="utf-8")
            with self.subTest(page=translation.name):
                expected = [(titles[title], script) for title, script in heading.findall(english)]
                self.assertEqual(heading.findall(translation.read_text(encoding="utf-8")), expected)


class HelpPanelPageTests(unittest.TestCase):
    """The page around the help: a language with a bundled font draws its script in it."""

    def test_the_page_names_its_language_and_brings_its_font(self):
        import EMAPSSN_Tools
        from desktop.Desktop_App import LANGUAGE_FONTS, language_web_font_css

        page = EMAPSSN_Tools.ResponsiveTextBrowser.page_html
        chinese = page("<p>帮助</p>", "zh_CN")
        self.assertIn('<html lang="zh-CN">', chinese)
        self.assertIn("<p>帮助</p>", chinese)
        fonts = language_web_font_css("zh_CN", "fonts/desktop/")
        self.assertIn(fonts, chinese)
        # Relative to the panel's baseUrl, src/resources/.
        for path in LANGUAGE_FONTS["zh_CN"].files:
            self.assertIn(f"url('fonts/desktop/{path}')", fonts)
            self.assertTrue((Path(EMAPSSN_Tools._SRC_DIR) / "resources" / "fonts" / "desktop" / path).is_file())
        for language in (None, "pseudo"):
            with self.subTest(language=language):
                shown = page("<p>Help</p>", language)
                self.assertIn("<html>", shown)
                self.assertNotIn("@font-face", shown)

    def test_the_panel_shows_its_page_in_the_windows_language(self):
        import EMAPSSN_Tools
        from PySide6.QtWebEngineWidgets import QWebEngineView

        QApplication.instance() or QApplication([])
        # The page the panel hands Chromium, taken without starting Chromium.
        browser = EMAPSSN_Tools.ResponsiveTextBrowser.__new__(EMAPSSN_Tools.ResponsiveTextBrowser)
        for language in ("zh_CN", None):
            with self.subTest(language=language), \
                    mock.patch.object(EMAPSSN_Tools, "installed_language", return_value=language), \
                    mock.patch.object(QWebEngineView, "setHtml") as shown:
                browser.setHtml("<p>帮助</p>")
                html, base_url = shown.call_args.args
                self.assertEqual(html, EMAPSSN_Tools.ResponsiveTextBrowser.page_html("<p>帮助</p>", language))
                self.assertEqual(Path(base_url.toLocalFile()), Path(EMAPSSN_Tools._SRC_DIR) / "resources")


if __name__ == "__main__":
    unittest.main()
