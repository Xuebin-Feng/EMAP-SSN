# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The translated documentation pages and the check that keeps them in step with the English ones.

docs/list_of_commands.html has a translation per language at
docs/<language>/list_of_commands.html, such as docs/zh_CN/. The translation is the
English page with its readable text translated and nothing else changed, and each of
its sections carries data-source-sha256, the fingerprint of the English section it
was reviewed against. docs/Docs_Translation.py compares the pages, lists the sections
whose English changed, and stamps a section as reviewed.

The first class runs that check on the real pages: it fails, saying what to update,
when the English page changes and the translation does not follow. The other classes
change copies of the pages in a temporary folder and show the check notice each kind
of change. The last class checks the Chinese font the translation draws with.
"""

import contextlib
import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Set before Qt WebEngine starts Chromium, as tests/test_page_translations.py does.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))

import Docs_Translation as T  # noqa: E402

PAGE = "list_of_commands"
ENGLISH = DOCS / (PAGE + ".html")
CHINESE = DOCS / "zh_CN" / (PAGE + ".html")


def translations():
    """(language code, path) of every translated copy of the page under docs/."""
    return [(folder.name, folder / (PAGE + ".html")) for folder in sorted(DOCS.iterdir())
            if folder.is_dir() and (folder / (PAGE + ".html")).is_file()]


class RealPagesTests(unittest.TestCase):
    def test_the_translations_follow_the_english_page(self):
        self.assertIn("zh_CN", [language for language, _ in translations()])
        for language, path in translations():
            with self.subTest(language=language):
                report = T.check_page(ENGLISH, path, language)
                self.assertTrue(report.ok, "\n" + report.format())

    def test_the_english_page_has_the_sections_the_translations_are_matched_by(self):
        page = T.Page(ENGLISH)
        self.assertEqual(page.problems, [])
        keys = page.keys
        self.assertEqual(len(keys), len(set(keys)))
        for key in ("title", "aside", "docs-intro", "expr-interpreter", "cmd-zoom", "cmd-agent", "empty-search-state", "script"):
            self.assertIn(key, keys)
        self.assertGreaterEqual(sum(1 for key, element in page.units if element.tag == "section"), 20)

    def test_every_rule_for_text_that_is_never_translated_matches_something(self):
        counts = T.Page(ENGLISH).verbatim_rule_counts()
        for (description, _), count in zip(T.VERBATIM_RULES, counts):
            with self.subTest(rule=description):
                self.assertGreater(count, 0)
        self.assertGreater(counts[0], 300)  # the page's <code> snippets

    def test_each_page_has_a_language_switch_that_reaches_the_other(self):
        english, chinese = T.Page(ENGLISH), T.Page(CHINESE)
        self.assertEqual(english.html.attrs["lang"], "en")
        self.assertEqual(chinese.html.attrs["lang"], "zh-CN")
        expected = {
            ENGLISH: ("en", "zh-CN", "zh_CN/list_of_commands.html", CHINESE),
            CHINESE: ("zh-CN", "en", "../list_of_commands.html", ENGLISH),
        }
        for page, (own, other, href, target) in ((english, expected[ENGLISH]), (chinese, expected[CHINESE])):
            with self.subTest(page=page.path.name, own=own):
                switch = next(e for e in page.root.walk() if "lang-switch" in e.classes)
                options = {option.attrs["lang"]: option for option in switch.elements()}
                self.assertEqual(sorted(options), sorted(["en", "zh-CN"]))
                self.assertEqual(options[own].tag, "span")
                self.assertEqual(options[own].attrs.get("aria-current"), "page")
                self.assertIn("current", options[own].classes)
                self.assertEqual(options[other].tag, "a")
                self.assertEqual(options[other].attrs["href"], href)
                self.assertEqual(options[other].attrs["hreflang"], other)
                self.assertNotIn("current", options[other].classes)
                self.assertTrue((page.path.parent / href).resolve().samefile(target))

    def test_both_pages_load_the_latin_and_the_chinese_font_stylesheets(self):
        for page in (T.Page(ENGLISH), T.Page(CHINESE)):
            with self.subTest(page=page.path.parent.name):
                hrefs = [e.attrs["href"] for e in page.head.walk() if e.tag == "link" and e.attrs.get("rel") == "stylesheet"]
                resolved = [(page.path.parent / href).resolve() for href in hrefs]
                self.assertEqual(
                    resolved, [(DOCS / "fonts" / "fonts.css").resolve(), (DOCS / "fonts" / "fonts_zh_CN.css").resolve()]
                )
                for path in resolved:
                    self.assertTrue(path.is_file(), path)

    def test_the_chinese_page_links_the_quick_start_beside_the_english_page(self):
        chinese = T.Page(CHINESE)
        links = [e.attrs["href"] for e in chinese.root.walk() if e.tag == "a" and "quickstart" in e.attrs.get("href", "")]
        self.assertEqual(links, ["../quickstart.html"])
        self.assertTrue((CHINESE.parent / links[0]).resolve().is_file())

    def test_every_link_to_a_section_reaches_a_section_of_the_page(self):
        for page in (T.Page(ENGLISH), T.Page(CHINESE)):
            ids = {e.attrs["id"] for e in page.root.walk() if "id" in e.attrs}
            for element in page.root.walk():
                href = element.attrs.get("href", "")
                if element.tag == "a" and href.startswith("#"):
                    with self.subTest(page=page.path.parent.name, href=href):
                        self.assertIn(href[1:], ids)

    def test_the_chinese_page_is_really_translated(self):
        english, chinese = T.Page(ENGLISH), T.Page(CHINESE)

        def prose(page):
            words = 0
            for node in page.body.walk():
                if node.tag in ("script", "style") or T._is_verbatim(node):
                    continue
                for child in node.children:
                    if isinstance(child, str):
                        words += len(re.findall(r"[A-Za-z]{4,} [a-z]{3,} [A-Za-z]{3,}", child))
            return words

        # Runs of three English words outside code: the English page has hundreds, the translation
        # only what stays English (names, lists of colormaps), never a sentence.
        self.assertGreater(prose(english), 500)
        self.assertLess(prose(chinese), 10)


class TemporaryPages(unittest.TestCase):
    """Copies of the real pages in a temporary folder, to change."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.docs = Path(folder.name)
        (self.docs / "zh_CN").mkdir()
        self.english = self.docs / (PAGE + ".html")
        self.chinese = self.docs / "zh_CN" / (PAGE + ".html")
        shutil.copyfile(ENGLISH, self.english)
        shutil.copyfile(CHINESE, self.chinese)

    def edit(self, path, old, new, count=1):
        text = path.read_bytes().decode("utf-8")
        self.assertEqual(text.count(old), count, "the test's edit no longer matches the page: %r" % old[:60])
        path.write_bytes(text.replace(old, new).encode("utf-8"))

    def sub(self, path, pattern, replacement, count=1):
        text = path.read_bytes().decode("utf-8")
        new, n = re.subn(pattern, replacement, text, count=count, flags=re.S)
        self.assertEqual(n, count, "the test's edit no longer matches the page: %r" % pattern[:60])
        path.write_bytes(new.encode("utf-8"))

    def check(self):
        return T.check_page(self.english, self.chinese, "zh_CN")

    def problems(self):
        return "\n".join(self.check().problems)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = T.main(list(argv), docs=self.docs)
        return code, out.getvalue(), err.getvalue()

    def test_the_copies_start_out_matching(self):
        self.assertTrue(self.check().ok, self.check().format())


class EnglishChangesTests(TemporaryPages):
    SENTENCE = "exactly the specified numeric width"

    def test_a_changed_english_sentence_makes_only_its_section_stale(self):
        self.edit(self.english, self.SENTENCE, "exactly the given numeric width")
        report = self.check()
        self.assertEqual(report.problems, [])
        self.assertEqual(report.stale, ["cmd-zoom"])
        self.assertEqual(report.unstamped, [])

    def test_the_failure_message_says_what_to_update_and_how(self):
        self.edit(self.english, self.SENTENCE, "exactly the given numeric width")
        message = self.check().format()
        for expected in (
            "cmd-zoom", "docs/list_of_commands.html", "docs/zh_CN/list_of_commands.html",
            "python docs/Docs_Translation.py stamp", "python docs/Docs_Translation.py show",
        ):
            self.assertIn(expected, message)

    def test_a_changed_code_snippet_in_english_makes_its_section_stale_and_unmatched(self):
        self.edit(self.english, "<code>reset hide</code> command", "<code>reset hidden</code> command")
        report = self.check()
        self.assertEqual(report.stale, ["cmd-hide"])
        self.assertIn("cmd-hide", "\n".join(report.problems))
        self.assertIn("never translated", "\n".join(report.problems))

    def test_a_new_english_section_is_named(self):
        self.edit(
            self.english,
            '<div class="empty-search-state" id="empty-search-state">',
            '<section class="command-card" id="cmd-new"><p>New.</p></section>\n<div class="empty-search-state" id="empty-search-state">',
        )
        self.assertIn("lacks these sections of the English page: cmd-new", self.problems())

    def test_an_english_section_that_is_gone_is_named(self):
        self.sub(self.english, r'<section class="command-card" id="cmd-agent">.*?</section>', "")
        self.assertIn("has sections the English page lacks: cmd-agent", self.problems())

    def test_an_english_element_beside_the_sections_without_an_id_is_named(self):
        self.edit(
            self.english,
            '<div class="empty-search-state" id="empty-search-state">',
            '<p>Loose paragraph.</p>\n<div class="empty-search-state" id="empty-search-state">',
        )
        self.assertIn("has no id", self.problems())

    def test_a_changed_english_title_attribute_and_script_string_make_theirs_stale(self):
        self.edit(self.english, 'title="Toggle Sidebar"', 'title="Toggle the sidebar"')
        self.edit(self.english, '"Copied!"', '"Copied to clipboard!"')
        report = self.check()
        self.assertEqual(sorted(report.stale), ["aside", "script"])
        self.assertIn("TRANSLATED_SCRIPT_STRINGS", "\n".join(report.problems))

    def test_re_indenting_and_rewrapping_the_english_page_changes_nothing(self):
        self.edit(self.english, "Sets the camera rectangle to exactly", "Sets the   camera\r\n      rectangle   to\r\n  exactly")
        self.edit(self.english, '<section class="command-card" id="cmd-save">', '\r\n\r\n        <section class="command-card"\r\n   id="cmd-save">')
        self.assertTrue(self.check().ok, self.check().format())

    def test_restyling_both_pages_alike_makes_nothing_stale(self):
        for path in (self.english, self.chinese):
            text = path.read_bytes().decode("utf-8")
            self.assertIn("copy-btn", text)
            path.write_bytes(text.replace("copy-btn", "copy-button").encode("utf-8"))
        report = self.check()
        self.assertEqual((report.problems, report.stale, report.unstamped), ([], [], []))

    def test_restyling_only_the_english_page_is_noticed_without_making_text_stale(self):
        self.edit(self.english, "--accent: #4f46e5;", "--accent: #0f766e;")
        report = self.check()
        self.assertEqual(report.stale, [])
        self.assertIn("<style>", "\n".join(report.problems))
        self.assertIn("stylesheet link", "\n".join(report.problems))

    def test_a_shared_stylesheet_for_both_pages_passes(self):
        for path, href in ((self.english, "list_of_commands.css"), (self.chinese, "../list_of_commands.css")):
            text = path.read_bytes().decode("utf-8")
            style = re.search(r"<style>(.*?)</style>", text, re.S)
            (self.docs / "list_of_commands.css").write_text(style.group(1), encoding="utf-8")
            path.write_bytes(text.replace(style.group(0), '<link href="%s" rel="stylesheet">' % href).encode("utf-8"))
        report = self.check()
        self.assertEqual((report.problems, report.stale, report.unstamped), ([], [], []))

    def test_a_shared_stylesheet_the_translation_does_not_link_is_noticed(self):
        text = self.english.read_bytes().decode("utf-8")
        style = re.search(r"<style>(.*?)</style>", text, re.S)
        self.english.write_bytes(text.replace(style.group(0), '<link href="list_of_commands.css" rel="stylesheet">').encode("utf-8"))
        self.assertIn("outside its sections", self.problems())

    def test_a_renamed_class_the_rules_rely_on_is_named(self):
        for path in (self.english, self.chinese):
            path.write_bytes(path.read_bytes().replace(b"syntax-content", b"syntax-text"))
        self.assertIn("class syntax-content", self.problems())


class TranslationChangesTests(TemporaryPages):
    def test_a_missing_code_snippet_is_noticed_in_its_section(self):
        self.edit(self.chinese, "<code>COMMAND help</code>、", "")
        problems = self.problems()
        self.assertIn("Section 'docs-intro'", problems)
        self.assertIn("the tags differ", problems)

    def test_a_changed_code_snippet_is_noticed_in_its_section(self):
        self.edit(self.chinese, "<code>reset hide</code>", "<code>reset unhide</code>")
        problems = self.problems()
        self.assertIn("Section 'cmd-hide'", problems)
        self.assertIn("'reset hide' in English and 'reset unhide' in the translation", problems)

    def test_a_translated_command_name_or_syntax_line_is_noticed(self):
        self.edit(self.chinese, '<div class="syntax-content">zoom &lt;width&gt;</div>', '<div class="syntax-content">缩放 &lt;width&gt;</div>')
        self.edit(self.chinese, '<span class="command-card-name">save</span>', '<span class="command-card-name">保存</span>')
        problems = self.problems()
        self.assertIn("Section 'cmd-zoom'", problems)
        self.assertIn("Section 'cmd-save'", problems)

    def test_a_translated_example_line_is_noticed(self):
        self.edit(self.chinese, "<span>zoom 500</span>", "<span>缩放 500</span>")
        self.assertIn("Section 'cmd-zoom'", self.problems())

    def test_a_translated_command_in_the_sidebar_list_is_noticed(self):
        self.edit(self.chinese, '<a href="#cmd-zoom"><span>zoom</span></a>', '<a href="#cmd-zoom"><span>缩放</span></a>')
        self.assertIn("Section 'aside'", self.problems())

    def test_a_changed_id_class_or_link_is_noticed(self):
        self.edit(self.chinese, 'id="cmd-undoredo"', 'id="cmd-undo"')
        problems = self.problems()
        self.assertIn("lacks these sections of the English page: cmd-undoredo", problems)
        self.assertIn("has sections the English page lacks: cmd-undo", problems)

    def test_a_changed_class_or_link_target_is_noticed(self):
        self.edit(self.chinese, '<a href="#cmd-zoom">', '<a href="#cmd-zoom2">')
        self.sub(self.chinese, r'<div class="callout">', '<div class="callout warn">', count=1)
        problems = self.problems()
        self.assertIn("Section 'aside'", problems)
        self.assertIn("Section 'expr-interpreter'", problems)

    def test_sections_in_another_order_are_noticed(self):
        self.edit(self.chinese, 'id="cmd-zoom"', 'id="cmd-tmp"')
        self.edit(self.chinese, 'id="cmd-save"', 'id="cmd-zoom"')
        self.edit(self.chinese, 'id="cmd-tmp"', 'id="cmd-save"')
        self.assertIn("another order", self.problems())

    def test_a_changed_style_or_head_is_noticed(self):
        self.edit(self.chinese, "--accent: #4f46e5;", "--accent: #4f46e6;")
        self.assertIn("outside its sections", self.problems())

    def test_a_missing_stylesheet_link_is_noticed(self):
        self.edit(self.chinese, '<link href="../fonts/fonts_zh_CN.css" rel="stylesheet">', "")
        self.assertIn("outside its sections", self.problems())

    def test_a_changed_script_is_noticed(self):
        self.edit(self.chinese, "const originalText = btnElement.innerText;", "const originalText = btnElement.innerHTML;")
        problems = self.problems()
        self.assertIn("Section 'script'", problems)
        self.assertIn("its code differs", problems)

    def test_a_script_string_left_in_english_is_noticed(self):
        self.edit(self.chinese, '"已复制！"', '"Copied!"')
        self.assertIn("still English", self.problems())

    def test_a_script_string_changed_that_is_not_for_readers_is_noticed(self):
        self.edit(self.chinese, "'highlight-card'", "'highlight-card2'", count=3)
        self.assertIn("add it to TRANSLATED_SCRIPT_STRINGS", self.problems())

    def test_a_missing_or_wrong_html_language_is_noticed(self):
        self.edit(self.chinese, '<html lang="zh-CN">', '<html lang="en">')
        self.assertIn('should declare <html lang="zh-CN">', self.problems())

    def test_the_translatable_attributes_must_stay_but_may_differ(self):
        self.edit(self.chinese, 'placeholder="搜索命令…"', 'placeholder="找命令"')
        self.assertTrue(self.check().ok, self.check().format())
        self.edit(self.chinese, ' placeholder="找命令"', "")
        self.assertIn("Section 'aside'", self.problems())

    def test_a_missing_fingerprint_is_listed(self):
        self.sub(self.chinese, r'(<section class="command-card" id="cmd-run") data-source-sha256="[0-9a-f]{64}"', r"\1")
        report = self.check()
        self.assertEqual(report.unstamped, ["cmd-run"])
        self.assertEqual((report.problems, report.stale), ([], []))

    def test_a_fingerprint_that_is_not_the_sections_is_stale(self):
        self.sub(self.chinese, r'(<section class="command-card" id="cmd-run" data-source-sha256=")[0-9a-f]{64}', r"\g<1>" + "0" * 64)
        self.assertEqual(self.check().stale, ["cmd-run"])


class LanguageSwitchTests(TemporaryPages):
    def test_a_page_without_a_switch_is_named(self):
        self.sub(self.chinese, r'<nav class="lang-switch".*?</nav>', "")
        self.assertIn("list_of_commands.html has no language switch", self.problems())

    def test_a_link_to_a_file_that_is_not_the_other_page_is_named(self):
        self.edit(self.chinese, 'href="../list_of_commands.html" title="English"', 'href="../quickstart.html" title="English"')
        self.assertIn("the switch's link for en points at '../quickstart.html'", self.problems())

    def test_a_link_to_a_page_that_does_not_exist_is_named(self):
        self.edit(self.english, 'href="zh_CN/list_of_commands.html"', 'href="zh_CN/missing.html"')
        self.assertIn("points at 'zh_CN/missing.html'", self.problems())

    def test_the_switch_must_mark_its_own_language(self):
        self.edit(self.chinese, '<span class="lang-option current" lang="zh-CN" aria-current="page">', '<span class="lang-option" lang="zh-CN">')
        self.assertIn("must mark zh-CN, the page's own language, as current", self.problems())

    def test_the_switch_must_not_mark_the_other_language(self):
        self.edit(self.english, '<a class="lang-option" lang="zh-CN"', '<a class="lang-option current" lang="zh-CN"')
        self.assertIn("must be a link that is not marked current", self.problems())

    def test_the_switch_may_follow_the_reader_s_anchor(self):
        self.edit(self.english, 'href="zh_CN/list_of_commands.html"', 'href="zh_CN/list_of_commands.html#cmd-label"')
        self.assertTrue(self.check().ok, self.check().format())

    def test_the_switch_does_not_make_the_sidebar_stale(self):
        self.edit(self.english, 'title="Simplified Chinese">简体中文<', 'title="Chinese (Simplified)">简体中文<')
        self.assertTrue(self.check().ok, self.check().format())


class UpdateWorkflowTests(TemporaryPages):
    SENTENCE = "exactly the specified numeric width"
    TRANSLATED = "恰好等于指定的数值宽度"

    def test_translating_a_changed_section_and_stamping_it_brings_the_pages_back_in_step(self):
        self.edit(self.english, self.SENTENCE, "exactly the given numeric width")
        before = self.chinese.read_bytes().decode("utf-8")
        self.assertEqual(self.check().stale, ["cmd-zoom"])
        self.edit(self.chinese, self.TRANSLATED, "恰好等于给定的数值宽度")
        self.assertEqual(self.check().stale, ["cmd-zoom"])  # editing the text does not record the review
        self.assertEqual(T.stamp_sections(self.english, self.chinese, ["cmd-zoom"]), ["cmd-zoom"])
        self.assertTrue(self.check().ok, self.check().format())
        after = self.chinese.read_bytes().decode("utf-8")
        old = re.findall(r'data-source-sha256="([0-9a-f]{64})"', before)
        new = re.findall(r'data-source-sha256="([0-9a-f]{64})"', after)
        self.assertEqual(len(old), len(new))
        self.assertEqual(len([1 for a, b in zip(old, new) if a != b]), 1)

    def test_stamping_a_section_that_does_not_exist_changes_nothing(self):
        before = self.chinese.read_bytes()
        with self.assertRaises(KeyError):
            T.stamp_sections(self.english, self.chinese, ["cmd-zoom", "cmd-nonexistent"])
        self.assertEqual(self.chinese.read_bytes(), before)

    def test_stamping_adds_a_fingerprint_to_a_section_that_has_none(self):
        self.sub(self.chinese, r' data-source-sha256="[0-9a-f]{64}"', "", count=36)
        self.assertEqual(len(self.check().unstamped), 36)
        T.stamp_sections(self.english, self.chinese, self.check().unstamped)
        self.assertTrue(self.check().ok, self.check().format())

    def test_the_fingerprints_are_those_of_the_english_sections(self):
        stamps = T.english_fingerprints(self.english)
        page = T.Page(self.chinese)
        for key, element in page.units:
            with self.subTest(section=key):
                self.assertEqual(element.attrs.get("data-source-sha256"), stamps[key])

    def test_the_command_line_lists_stale_sections_and_stamps_them(self):
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("matches", out)
        self.edit(self.english, self.SENTENCE, "exactly the given numeric width")
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("cmd-zoom", out)
        self.assertIn("stamp", out)
        code, out, _ = self.run_main("stamp")
        self.assertEqual(code, 1)
        self.assertIn("cmd-zoom", out)
        code, out, _ = self.run_main("stamp", "--all")
        self.assertEqual(code, 0)
        self.assertIn("Stamped 1 section(s): cmd-zoom", out)
        self.assertEqual(self.run_main()[0], 0)

    def test_the_command_line_refuses_to_stamp_while_the_pages_differ_in_more_than_text(self):
        self.edit(self.chinese, "<code>reset hide</code>", "<code>reset unhide</code>")
        self.edit(self.english, self.SENTENCE, "exactly the given numeric width")
        before = self.chinese.read_bytes()
        code, out, err = self.run_main("stamp", "--all")
        self.assertEqual(code, 1)
        self.assertIn("cmd-hide", out)
        self.assertIn("Nothing was stamped", err)
        self.assertEqual(self.chinese.read_bytes(), before)

    def test_the_command_line_shows_both_sections_and_the_font_characters(self):
        code, out, _ = self.run_main("show", "cmd-zoom")
        self.assertEqual(code, 0)
        self.assertIn("===== English: cmd-zoom =====", out)
        self.assertIn("===== Translation: cmd-zoom =====", out)
        self.assertIn("Sets the camera rectangle", out)
        self.assertIn("相机矩形", out)
        listed = self.docs / "chars.txt"
        code, out, _ = self.run_main("characters", "--out", str(listed))
        self.assertEqual(code, 0)
        characters = listed.read_bytes().decode("utf-8")
        self.assertEqual(characters, T.cjk_characters(self.english, self.chinese))
        self.assertIn("简体中文", "".join(c for c in "简体中文" if c in characters))

    def test_a_missing_page_is_a_usage_error(self):
        self.chinese.unlink()
        code, _, err = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("No such page", err)


class BrowserTests(unittest.TestCase):
    """The pages in Qt WebEngine, opened from disk as a reader opens them (file://)."""

    @classmethod
    def setUpClass(cls):
        try:
            from PySide6.QtCore import QEventLoop, QTimer, QUrl
            from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
            from PySide6.QtWebEngineWidgets import QWebEngineView
            from PySide6.QtWidgets import QApplication
        except ImportError as error:
            raise unittest.SkipTest("needs PySide6 with Qt WebEngine: %s" % error)
        cls.QEventLoop, cls.QTimer, cls.QUrl = QEventLoop, QTimer, QUrl
        cls.app = QApplication.instance() or QApplication([])
        cls.profile = QWebEngineProfile()  # off the record: nothing is written to disk
        cls.view = QWebEngineView()
        cls.page = QWebEnginePage(cls.profile, cls.view)
        cls.view.setPage(cls.page)
        cls.view.resize(1200, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.page.deleteLater()
        cls.view.deleteLater()
        cls.app.processEvents()
        del cls.page, cls.view
        cls.profile.deleteLater()

    def pump(self, milliseconds):
        loop = self.QEventLoop()
        self.QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def run_js(self, code):
        result = {}
        loop = self.QEventLoop()

        def done(value):
            result["value"] = value
            loop.quit()

        self.page.runJavaScript(code, done)
        self.QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.assertIn("value", result, "the page did not answer: " + code)
        return result["value"]

    def load(self, url):
        loop = self.QEventLoop()
        self.view.loadFinished.connect(lambda ok: loop.quit())
        self.view.load(url)
        self.QTimer.singleShot(20000, loop.quit)
        loop.exec()
        self.pump(600)
        self.assertEqual(self.run_js("document.readyState"), "complete")

    def open(self, path, fragment=None):
        url = self.QUrl.fromLocalFile(str(path))
        if fragment:
            url.setFragment(fragment)
        self.load(url)

    def scroll_to(self, section):
        self.run_js("document.documentElement.style.scrollBehavior = 'auto'; document.getElementById('%s').scrollIntoView()" % section)
        self.pump(300)

    def switch_link(self):
        """The language switch's link as a reader pressing it would find it."""
        return self.run_js(
            "var link = document.querySelector('a.lang-option'); link.dispatchEvent(new Event('pointerdown')); link.getAttribute('href')"
        )

    def follow_switch(self):
        self.load(self.view.url().resolved(self.QUrl(self.switch_link())))

    def facts(self):
        import json

        return json.loads(self.run_js(
            "JSON.stringify({lang: document.documentElement.lang, hash: location.hash, "
            "theme: document.documentElement.getAttribute('data-theme'), "
            "highlighted: Array.from(document.querySelectorAll('.highlight-card')).map(e => e.id)})"
        ))

    def test_the_switch_carries_the_reader_s_place_to_the_other_language_and_back(self):
        self.open(ENGLISH, "cmd-label")
        self.assertEqual(self.facts()["lang"], "en")
        self.assertEqual(self.switch_link(), "zh_CN/list_of_commands.html#cmd-label")
        for section, anchor in (("cmd-print", "#cmd-print"), ("cmd-query", "#cmd-query"), ("expr-boolean", "#expr-boolean"), ("docs-intro", "")):
            with self.subTest(section=section):
                self.scroll_to(section)
                self.assertEqual(self.switch_link(), "zh_CN/list_of_commands.html" + anchor)
        self.scroll_to("cmd-meta")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["hash"]), ("zh-CN", "#cmd-meta"))
        self.assertEqual(facts["highlighted"], ["cmd-meta"])
        self.assertEqual(self.switch_link(), "../list_of_commands.html#cmd-meta")
        self.scroll_to("cmd-logo")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["hash"]), ("en", "#cmd-logo"))

    def test_the_switch_finds_the_section_at_the_end_of_the_page_too(self):
        self.open(CHINESE, "cmd-agent")
        self.run_js("window.scrollTo(0, document.documentElement.scrollHeight)")
        self.pump(300)
        self.assertEqual(self.switch_link(), "../list_of_commands.html#cmd-agent")

    def test_the_chosen_theme_goes_with_the_reader(self):
        self.open(ENGLISH)
        self.run_js("setTheme('dark')")
        self.follow_switch()
        self.assertEqual(self.facts()["theme"], "dark")
        self.run_js("setTheme('light')")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["theme"]), ("en", "light"))

    def test_the_search_finds_chinese_text_and_says_so_when_nothing_matches(self):
        self.open(CHINESE)

        def search(text):
            self.run_js(
                "var box = document.getElementById('search-input'); box.value = %r; box.dispatchEvent(new Event('input'))" % text
            )
            return {
                section: self.run_js("document.getElementById('%s').style.display" % section)
                for section in ("cmd-cluster", "cmd-zoom", "empty-search-state")
            }

        shown = search("聚类")
        self.assertEqual((shown["cmd-cluster"], shown["cmd-zoom"], shown["empty-search-state"]), ("block", "none", "none"))
        shown = search("相机矩形")
        self.assertEqual((shown["cmd-cluster"], shown["cmd-zoom"]), ("none", "block"))
        shown = search("没有这样的命令zzz")
        self.assertEqual(shown["empty-search-state"], "block")
        self.assertIn("没有匹配的命令", self.run_js("document.getElementById('empty-search-state').textContent"))
        search("")
        self.assertEqual(self.run_js("document.getElementById('cmd-zoom').style.display"), "block")

    def test_each_page_loads_the_chinese_faces_from_the_local_subsets_when_it_draws_chinese(self):
        for page in (CHINESE, ENGLISH):
            with self.subTest(page=page.parent.name):
                self.open(page)
                self.pump(600)
                loaded = self.run_js(
                    "Array.from(document.fonts).filter(f => f.status === 'loaded' && f.unicodeRange.indexOf('4E00') >= 0)"
                    ".map(f => f.family.replace(/[\"']/g, '') + ' ' + f.weight).join(',')"
                )
                self.assertIn("Noto Sans 400", loaded)
                if page == CHINESE:
                    self.assertIn("Noto Sans 700", loaded)


class ChineseFontTests(unittest.TestCase):
    """The pages draw Chinese from woff2 subsets of the application's Noto Sans SC, listed in fonts_zh_CN.css."""

    FONTS = DOCS / "fonts"

    @classmethod
    def setUpClass(cls):
        try:
            import brotli  # noqa: F401  (fontTools reads woff2 through it)
            from fontTools.ttLib import TTFont
        except ImportError as error:
            raise unittest.SkipTest("reading woff2 needs fontTools and brotli: %s" % error)
        cls.TTFont = TTFont
        cls.css = (cls.FONTS / "fonts_zh_CN.css").read_text(encoding="utf-8")
        cls.faces = []
        for block in re.findall(r"@font-face\s*\{(.*?)\}", re.sub(r"/\*.*?\*/", "", cls.css, flags=re.S), re.S):
            cls.faces.append({
                "family": re.search(r"font-family:\s*'([^']+)'", block).group(1),
                "weight": int(re.search(r"font-weight:\s*(\d+)\s*;", block).group(1)),
                "file": re.search(r"url\(([^)]+)\)", block).group(1),
                "ranges": re.search(r"unicode-range:\s*([^;]+);", block).group(1),
            })

    def used_characters(self):
        return set(T.cjk_characters(ENGLISH, *[path for _, path in translations()]))

    def test_the_stylesheet_adds_a_regular_and_a_bold_face_to_the_page_family(self):
        self.assertEqual([(face["family"], face["weight"]) for face in self.faces], [("Noto Sans", 400), ("Noto Sans", 700)])
        self.assertEqual({face["file"] for face in self.faces}, {"NotoSansSC-Regular.woff2", "NotoSansSC-Bold.woff2"})
        self.assertNotIn("http", self.css)
        self.assertEqual(len(re.findall(r"font-family: 'Noto Sans';", (self.FONTS / "fonts.css").read_text(encoding="utf-8"))), 8)

    def test_the_unicode_range_is_the_one_the_check_extracts_characters_by(self):
        for face in self.faces:
            ranges = []
            for part in face["ranges"].split(","):
                low, _, high = part.strip().removeprefix("U+").partition("-")
                ranges.append((int(low, 16), int(high or low, 16)))
            self.assertEqual(tuple(ranges), T.CJK_RANGES)

    def test_each_subset_covers_every_chinese_character_on_the_pages(self):
        used = self.used_characters()
        self.assertGreater(len(used), 600)
        for face in self.faces:
            path = self.FONTS / face["file"]
            with self.subTest(file=face["file"]):
                self.assertTrue(path.is_file())
                font = self.TTFont(path)
                self.assertEqual(font.flavor, "woff2")
                cmap = font.getBestCmap()
                missing = sorted(char for char in used if ord(char) not in cmap)
                self.assertEqual(
                    missing, [],
                    "The font subset lacks %d character(s) the pages use: %s. Rebuild it as docs/fonts/LICENSE.fonts "
                    "describes, starting with: python docs/Docs_Translation.py characters --out characters.txt"
                    % (len(missing), "".join(missing)),
                )

    def test_each_subset_keeps_to_the_characters_the_pages_use(self):
        used = self.used_characters()
        for face in self.faces:
            with self.subTest(file=face["file"]):
                cmap = self.TTFont(self.FONTS / face["file"]).getBestCmap()
                extra = [chr(code) for code in cmap if any(low <= code <= high for low, high in T.CJK_RANGES) and chr(code) not in used]
                self.assertLessEqual(len(extra), 20, "Rebuild the subset: it carries %d unused characters." % len(extra))
                self.assertLess((self.FONTS / face["file"]).stat().st_size, 200_000)

    def test_both_subsets_are_the_applications_noto_sans_sc(self):
        for face, source in zip(self.faces, ("Regular", "Bold")):
            with self.subTest(file=face["file"]):
                names = self.TTFont(self.FONTS / face["file"])["name"]
                family = names.getDebugName(1)
                self.assertEqual(family, "Noto Sans SC")
                self.assertIn("Adobe", names.getDebugName(0))
                full = self.TTFont(ROOT / "src" / "resources" / "fonts" / "desktop" / "noto" / "NotoSansSC" / ("NotoSansSC-%s.ttf" % source))
                cmap, source_cmap = self.TTFont(self.FONTS / face["file"]).getBestCmap(), full.getBestCmap()
                for code in cmap:
                    self.assertIn(code, source_cmap)

    def test_the_licence_notes_name_the_font_and_each_file_by_its_hash(self):
        notes = (self.FONTS / "LICENSE.fonts").read_text(encoding="utf-8")
        self.assertIn("Noto Sans SC", notes)
        self.assertIn("pyftsubset", notes)
        for face in self.faces:
            digest = hashlib.sha256((self.FONTS / face["file"]).read_bytes()).hexdigest()
            self.assertIn("%s  %s" % (digest, face["file"]), notes, "LICENSE.fonts lists a different %s" % face["file"])
        licence = (self.FONTS / "LICENSE.Noto").read_text(encoding="utf-8")
        self.assertIn("Noto Sans SC", licence)
        self.assertIn("SIL OPEN FONT LICENSE Version 1.1", licence)
        self.assertEqual(licence, (ROOT / "src" / "resources" / "fonts" / "LICENSE.Noto").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
