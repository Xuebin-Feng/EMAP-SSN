# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The translated documentation pages and the check that keeps them in step with the English ones.

The English pages are docs/<page>.html (quickstart.html and list_of_commands.html). A
language's translation of one is in its language's folder, as
src/resources/languages/<language>/docs/<page>.html, such as the Chinese ones in zh_CN/. The
translation is the English page with its readable text translated and nothing else changed,
and each of its sections carries data-source-sha256, the fingerprint of the English section
it was reviewed against. src/resources/languages/Docs_Translation.py compares the pages,
lists the sections whose English changed, stamps a section as reviewed, and writes what depends
on which languages exist: each page's language dropdown and each language's language.css (the
font of its pages, made from its language.json).

The first class runs that check on the real pages: it fails, saying what to update, when an
English page changes and a translation does not follow. The next classes change copies of the
pages, laid out as the project is (docs/ and src/resources/languages/zh_CN/docs/), in a temporary
folder and show the check notice each kind of change, for the command reference and for the Quick
Start, a page without <section>s. Then come the discovery of translated pages under the language
folders, the generated parts, and the browser. The last class checks the font the Chinese
pages draw with.
"""

import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Set before Qt WebEngine starts Chromium, as tests/test_page_translations.py does.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LANGUAGES = ROOT / "src" / "resources" / "languages"
for folder in (LANGUAGES, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import Docs_Translation as T  # noqa: E402
from utilities.Language_Packs import unicode_ranges  # noqa: E402

PAGES = ("list_of_commands", "quickstart")
ZH = LANGUAGES / "zh_CN" / "docs"
COMMANDS = "list_of_commands"
QUICK = "quickstart"
UP = "../../../../../docs/"  # from a translation's folder to docs/


def real(page):
    """(English path, Chinese path) of a page."""
    return DOCS / (page + ".html"), ZH / (page + ".html")


_CARD = re.compile(r'<section class="command-card" id="(cmd-[a-z0-9-]+)"[^>]*>(.*?)</section>', re.S)
_LEAD = re.compile(r"<p><strong>[^<]*</strong>([^<]*)")


def _once(text, snippet):
    return text.count(snippet) == 1


def _lead_phrase(card, text, chinese):
    """The start of the card's first paragraph, as little of it as occurs once in the page, or None."""
    match = _LEAD.search(card)
    prose = match.group(1).strip() if match else ""
    if chinese:
        candidates = (prose[:length] for length in range(6, len(prose) + 1))
    else:
        words = prose.split()
        candidates = (" ".join(words[:count]) for count in range(4, len(words) + 1))
    for phrase in candidates:
        if "&" not in phrase and _once(text, phrase):
            return phrase
    return None


def sample_cards(english_text, chinese_text, wanted=3):
    """The first `wanted` command cards of the command reference that the tests can change by name.

    The sync machinery's tests need a few cards to edit: the sentence at the start of a
    card, its syntax line, its example, a <code> snippet, its name and its link in the
    sidebar. They take those from the pages as they are, so a rewording of any card's
    text never has to be copied into a test. A card qualifies when each of those occurs
    exactly once in both pages. Each is a namespace: key, name, syntax, example, code (in
    both pages, as HTML), lead (the English phrase) and zh_lead (the Chinese one).
    """
    english = {key: card for key, card in _CARD.findall(english_text)}
    chinese = {key: card for key, card in _CARD.findall(chinese_text)}
    found = []
    for key, card in english.items():
        if key not in chinese or not _once(english_text, 'id="%s"' % key) or not _once(chinese_text, 'id="%s"' % key):
            continue
        name = re.search(r'<span class="command-card-name">([^<]+)</span>', card)
        syntax = re.search(r'<div class="syntax-content">([^<]+)</div>', card)
        example = re.search(r'<li class="example-item"><span>([^<]+)</span>', card)
        if not (name and syntax and example):
            continue
        name, syntax, example = name.group(1), syntax.group(1), example.group(1)
        pieces = (
            '<span class="command-card-name">%s</span>' % name, '<div class="syntax-content">%s</div>' % syntax,
            "<span>%s</span>" % example, '<a href="#%s"><span>%s</span></a>' % (key, name),
        )
        if not all(_once(english_text, piece) and _once(chinese_text, piece) for piece in pieces):
            continue
        code = next((c for c in re.findall(r"<code>([^<]+)</code>", card)
                     if _once(english_text, "<code>%s</code>" % c) and _once(chinese_text, "<code>%s</code>" % c)), None)
        lead = _lead_phrase(card, english_text, chinese=False)
        zh_lead = _lead_phrase(chinese[key], chinese_text, chinese=True)
        if code and lead and zh_lead:
            found.append(types.SimpleNamespace(
                key=key, name=name, syntax=syntax, example=example, code=code, lead=lead, zh_lead=zh_lead))
        if len(found) == wanted:
            return found
    raise AssertionError("the command reference has too few cards the tests can change by name: %d" % len(found))


def samples():
    """Three sample cards of the real command reference (see sample_cards)."""
    english, chinese = (path.read_bytes().decode("utf-8") for path in real(COMMANDS))
    return sample_cards(english, chinese)


class RealPagesTests(unittest.TestCase):
    def test_every_english_page_has_a_chinese_translation_that_follows_it(self):
        found = [(page, code) for page, code, _, _ in T.translations()]
        self.assertEqual(found, [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])
        for page, code, english, translated in T.translations():
            with self.subTest(page=page, language=code):
                report = T.check_page(english, translated, code)
                self.assertTrue(report.ok, "\n" + report.format())

    def test_the_whole_check_passes_and_every_rule_it_relies_on_still_matches_something(self):
        self.assertEqual(T.unused_rules(), [])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(T.main([]), 0, out.getvalue())
        self.assertEqual(out.getvalue().count("matches"), 2)

    def test_the_sample_cards_the_sync_tests_change_come_from_the_pages(self):
        # The machinery tests below edit three cards picked at run time, not quoted: the pick
        # must stay possible, and different cards must be picked.
        first, second, third = samples()
        self.assertEqual(len({first.key, second.key, third.key}), 3)
        english = (DOCS / "list_of_commands.html").read_bytes().decode("utf-8")
        chinese = (ZH / "list_of_commands.html").read_bytes().decode("utf-8")
        for card in (first, second, third):
            self.assertEqual(english.count(card.lead), 1, card.key)
            self.assertEqual(chinese.count(card.zh_lead), 1, card.key)
            self.assertEqual(english.count("<code>%s</code>" % card.code), 1, card.key)

    def test_the_command_reference_has_the_sections_the_translations_are_matched_by(self):
        page = T.Page(DOCS / (COMMANDS + ".html"))
        self.assertEqual(page.problems, [])
        keys = page.keys
        self.assertEqual(len(keys), len(set(keys)))
        for key in ("title", "icons", "topbar", "sidebar", "docs-intro", "expr-interpreter", "cmd-zoom", "cmd-agent", "empty-search-state", "script"):
            self.assertIn(key, keys)
        self.assertGreaterEqual(sum(1 for key, element in page.units if element.tag == "section"), 20)

    def test_the_quick_start_is_matched_by_its_dividers_and_what_follows_them(self):
        page = T.Page(DOCS / (QUICK + ".html"))
        self.assertEqual(page.problems, [])
        self.assertEqual(
            page.keys,
            ["title", "icons", "topbar", "sidebar", "page-header", "overview", "workflow", "gui-tools", "gui-config",
             "gui-viewer", "hands-on-tutorial", "best-practices", "next-steps", "script"],
        )
        tags = {key: [e.tag + ("." + e.attrs.get("class", "").split()[0] if e.attrs.get("class") else "")
                      for e in page.unit_members(key)] for key in page.keys}
        # The header and the note before the first divider make one section, named by the header's class.
        self.assertEqual(tags["page-header"], ["header.page-header", "div.callout"])
        # A section is its divider and the cards, figures and paragraphs up to the next divider.
        self.assertEqual(tags["overview"], ["div.section-divider", "div.manual-card"])
        self.assertGreaterEqual(len(tags["gui-config"]), 4)
        # The footer after the last card belongs to the last section.
        self.assertEqual(tags["next-steps"][-1], "footer")
        for key in ("overview", "workflow", "gui-tools", "gui-config", "gui-viewer", "hands-on-tutorial", "best-practices", "next-steps"):
            self.assertEqual(page.unit(key).attrs["id"], key)

    def test_every_rule_for_text_that_is_never_translated_matches_something(self):
        counts = T.Page(DOCS / (COMMANDS + ".html")).verbatim_rule_counts()
        for (description, _), count in zip(T.VERBATIM_RULES, counts):
            with self.subTest(rule=description):
                self.assertGreater(count, 0)
        self.assertGreater(counts[0], 300)  # the page's <code> snippets
        quick = T.Page(DOCS / (QUICK + ".html")).verbatim_rule_counts()
        self.assertGreater(quick[0], 100)  # and the Quick Start's

    def test_each_page_has_a_language_dropdown_that_reaches_the_pages_of_the_other_language(self):
        values = {
            COMMANDS: {
                "en": {"en": "list_of_commands.html", "zh-CN": "../src/resources/languages/zh_CN/docs/list_of_commands.html"},
                "zh-CN": {"en": UP + "list_of_commands.html", "zh-CN": "list_of_commands.html"},
            },
            QUICK: {
                "en": {"en": "quickstart.html", "zh-CN": "../src/resources/languages/zh_CN/docs/quickstart.html"},
                "zh-CN": {"en": UP + "quickstart.html", "zh-CN": "quickstart.html"},
            },
        }
        for page in PAGES:
            for own, path in (("en", real(page)[0]), ("zh-CN", real(page)[1])):
                with self.subTest(page=page, own=own):
                    parsed = T.Page(path)
                    self.assertEqual(parsed.html.attrs["lang"], own)
                    switch = next(e for e in parsed.root.walk() if "lang-switch" in e.classes)
                    select = next(e for e in switch.walk() if e.tag == "select")
                    self.assertTrue(select.attrs["aria-label"].strip())
                    options = {option.attrs["lang"]: option for option in select.elements() if option.tag == "option"}
                    self.assertEqual(list(options), ["en", "zh-CN"])
                    self.assertEqual([option.text() for option in options.values()], ["English", "简体中文"])
                    for lang, option in options.items():
                        self.assertEqual(option.attrs["data-lang"], lang.replace("-", "_"))
                        self.assertEqual(option.attrs["value"], values[page][own][lang])
                        self.assertEqual("selected" in option.attrs, lang == own)
                        self.assertTrue((path.parent / option.attrs["value"]).resolve().is_file())

    def test_each_page_links_the_shared_stylesheets_and_a_translation_also_its_languages_font(self):
        for page in PAGES:
            for translated, path in ((False, real(page)[0]), (True, real(page)[1])):
                with self.subTest(page=page, translated=translated):
                    parsed = T.Page(path)
                    hrefs = [e.attrs["href"] for e in parsed.head.walk() if e.tag == "link" and e.attrs.get("rel") == "stylesheet"]
                    resolved = [(path.parent / href).resolve() for href in hrefs]
                    expected = [(DOCS / "fonts" / "fonts.css").resolve(), (DOCS / "assets" / "docs.css").resolve()]
                    if translated:
                        expected.append((ZH / "language.css").resolve())
                    self.assertEqual(resolved, expected)
                    for target in resolved:
                        self.assertTrue(target.is_file(), target)
                    scripts = [(path.parent / e.attrs["src"]).resolve() for e in parsed.head.walk() if e.tag == "script"]
                    self.assertEqual(scripts, [(DOCS / "assets" / "docs.js").resolve()])

    def test_the_pages_of_a_language_link_each_other_not_the_english_ones(self):
        for translated, folder in ((False, DOCS), (True, ZH)):
            for page, other in ((COMMANDS, QUICK), (QUICK, COMMANDS)):
                with self.subTest(translated=translated, page=page):
                    parsed = T.Page(folder / (page + ".html"))
                    links = {e.attrs["href"] for e in parsed.root.walk() if e.tag == "a" and e.attrs.get("href", "").startswith(other)}
                    self.assertEqual(links, {other + ".html"})
                    self.assertEqual((folder / (other + ".html")).resolve(), (folder / (other + ".html")).resolve())
                    self.assertTrue((folder / (other + ".html")).is_file())

    def test_every_link_to_a_section_reaches_a_section_of_the_page(self):
        for page in PAGES:
            for path in real(page):
                parsed = T.Page(path)
                ids = {e.attrs["id"] for e in parsed.root.walk() if "id" in e.attrs}
                for element in parsed.root.walk():
                    href = element.attrs.get("href", "")
                    if element.tag == "a" and href.startswith("#"):
                        with self.subTest(page=path.relative_to(ROOT).as_posix(), href=href):
                            self.assertIn(href[1:], ids)

    def test_the_chinese_pages_are_really_translated(self):
        def prose(parsed):
            words = 0
            for node in parsed.body.walk():
                if node.tag in ("script", "style") or T._is_verbatim(node):
                    continue
                for child in node.children:
                    if isinstance(child, str):
                        words += len(re.findall(r"[A-Za-z]{4,} [a-z]{3,} [A-Za-z]{3,}", child))
            return words

        # Runs of three English words outside code: the English page has hundreds, the translation
        # only what stays English (names, lists of colormaps), never a sentence.
        for page, minimum in ((COMMANDS, 500), (QUICK, 150)):
            with self.subTest(page=page):
                self.assertGreater(prose(T.Page(real(page)[0])), minimum)
                self.assertLess(prose(T.Page(real(page)[1])), 10)

    def test_a_figure_caption_gives_the_label_the_english_screenshot_shows_beside_its_translation(self):
        page = (ZH / "quickstart.html").read_text(encoding="utf-8")
        for translated in ("保存并运行（Save &amp; Run）", "导出布局设置（Export Layout Settings）", "一致性检查（Consistency Check）",
                           "缩放（Zoom）", "隐藏节点（Hidden Nodes）", "选择框（Selection Box）"):
            self.assertIn(translated, page)
        # The pictures are the English screenshots, the same files as the English page's.
        english = T.Page(DOCS / "quickstart.html")
        chinese = T.Page(ZH / "quickstart.html")
        pictures = [(e.attrs["src"], e.attrs["alt"]) for e in english.root.walk() if e.tag == "img" and "alt" in e.attrs]
        translated = [(e.attrs["src"], e.attrs["alt"]) for e in chinese.root.walk() if e.tag == "img" and "alt" in e.attrs]
        self.assertEqual([src for src, _ in translated], [UP + src for src, _ in pictures])
        for (_, alt), (_, zh) in zip(pictures, translated):
            if alt:
                self.assertTrue(re.search(r"[一-鿿]", zh), zh)

    def test_the_generated_parts_are_up_to_date(self):
        self.assertEqual(T.generate(check=True), [])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(T.main(["generate", "--check"]), 0, out.getvalue() + err.getvalue())


class TemporaryPages(unittest.TestCase):
    """Copies of the real pages, laid out as the project is, in a temporary folder, to change."""

    PAGE = COMMANDS

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.docs = self.root / "docs"
        self.languages = self.root / "src" / "resources" / "languages"
        self.zh = self.languages / "zh_CN"
        self.docs.mkdir()
        (self.zh / "docs").mkdir(parents=True)
        shutil.copyfile(LANGUAGES / "zh_CN" / "language.json", self.zh / "language.json")
        shutil.copyfile(ZH / "language.css", self.zh / "docs" / "language.css")
        for page in PAGES:
            shutil.copyfile(real(page)[0], self.docs / (page + ".html"))
            shutil.copyfile(real(page)[1], self.zh / "docs" / (page + ".html"))
        self.english = self.docs / (self.PAGE + ".html")
        self.chinese = self.zh / "docs" / (self.PAGE + ".html")
        # Three command cards picked from the copies: card A for a sentence, a syntax line, an
        # example and a sidebar link; B for a snippet and a name; C for a whole section.
        self.a, self.b, self.c = samples()

    def reworded(self, phrase):
        """phrase with a word added: a change to a section's text that makes it stale."""
        return phrase + " indeed"

    def edit(self, path, old, new, count=1):
        text = path.read_bytes().decode("utf-8")
        self.assertEqual(text.count(old), count, "the test's edit no longer matches the page: %r" % old[:60])
        path.write_bytes(text.replace(old, new).encode("utf-8"))

    def sub(self, path, pattern, replacement, count=1):
        text = path.read_bytes().decode("utf-8")
        new, n = re.subn(pattern, replacement, text, count=count, flags=re.S)
        self.assertEqual(n, count, "the test's edit no longer matches the page: %r" % pattern[:60])
        path.write_bytes(new.encode("utf-8"))

    def check(self, page=None):
        page = page or self.PAGE
        return T.check_page(self.docs / (page + ".html"), self.zh / "docs" / (page + ".html"), "zh_CN")

    def problems(self):
        return "\n".join(self.check().problems)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = T.main(list(argv), docs=self.docs, languages_dir=self.languages)
        return code, out.getvalue(), err.getvalue()

    def test_the_copies_start_out_matching(self):
        for page in PAGES:
            self.assertTrue(self.check(page).ok, self.check(page).format())


class EnglishChangesTests(TemporaryPages):
    def test_a_changed_english_sentence_makes_only_its_section_stale(self):
        self.edit(self.english, self.a.lead, self.reworded(self.a.lead))
        report = self.check()
        self.assertEqual(report.problems, [])
        self.assertEqual(report.stale, [self.a.key])
        self.assertEqual(report.unstamped, [])

    def test_the_failure_message_says_what_to_update_and_how(self):
        self.edit(self.english, self.a.lead, self.reworded(self.a.lead))
        message = self.check().format()
        for expected in (
            self.a.key, "list_of_commands.html", "python src/resources/languages/Docs_Translation.py stamp",
            "python src/resources/languages/Docs_Translation.py show", "--page list_of_commands --language zh_CN",
        ):
            self.assertIn(expected, message)

    def test_a_changed_code_snippet_in_english_makes_its_section_stale_and_unmatched(self):
        self.edit(self.english, "<code>%s</code>" % self.b.code, "<code>%s_changed</code>" % self.b.code)
        report = self.check()
        self.assertEqual(report.stale, [self.b.key])
        self.assertIn(self.b.key, "\n".join(report.problems))
        self.assertIn("never translated", "\n".join(report.problems))

    def test_a_new_english_section_is_named(self):
        self.edit(
            self.english,
            '<div class="empty-search-state" id="empty-search-state">',
            '<section class="command-card" id="cmd-new"><p>New.</p></section>\n<div class="empty-search-state" id="empty-search-state">',
        )
        self.assertIn("lacks these sections of the English page: cmd-new", self.problems())

    def test_an_english_section_that_is_gone_is_named(self):
        self.sub(self.english, r'<section class="command-card" id="%s">.*?</section>' % self.c.key, "")
        self.assertIn("has sections the English page lacks: %s" % self.c.key, self.problems())

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
        self.assertEqual(sorted(report.stale), ["script", "topbar"])
        self.assertIn("TRANSLATED_SCRIPT_STRINGS", "\n".join(report.problems))

    def test_a_changed_english_theme_label_makes_the_top_bar_stale(self):
        self.edit(self.english, 'data-label-dark="Dark Mode"', 'data-label-dark="Night Mode"')
        report = self.check()
        self.assertEqual((report.problems, report.stale), ([], ["topbar"]))

    def test_a_theme_label_attribute_no_english_page_has_is_named(self):
        for name in PAGES:
            for path in (self.docs / (name + ".html"), self.zh / "docs" / (name + ".html")):
                text = path.read_bytes().decode("utf-8")
                path.write_bytes(re.sub(r' data-label-dark="[^"]*"', "", text).encode("utf-8"))
        messages = "\n".join(T.unused_rules(self.docs))
        self.assertIn("TRANSLATED_ATTRIBUTES in src/resources/languages/Docs_Translation.py lists 'data-label-dark'", messages)

    def test_re_indenting_and_rewrapping_the_english_page_changes_nothing(self):
        self.edit(self.english, self.a.lead, "   \r\n      ".join(self.a.lead.split(" ")))
        card = '<section class="command-card" id="%s">' % self.b.key
        self.edit(self.english, card, '\r\n\r\n        <section class="command-card"\r\n   id="%s">' % self.b.key)
        self.assertTrue(self.check().ok, self.check().format())

    def test_restyling_both_pages_alike_makes_nothing_stale(self):
        for path in (self.english, self.chinese):
            text = path.read_bytes().decode("utf-8")
            self.assertIn("copy-btn", text)
            path.write_bytes(text.replace("copy-btn", "copy-button").encode("utf-8"))
        report = self.check()
        self.assertEqual((report.problems, report.stale, report.unstamped), ([], [], []))

    def test_restyling_only_the_english_page_is_noticed_without_making_text_stale(self):
        self.edit(self.english, 'href="assets/docs.css"', 'href="assets/docs-teal.css"')
        report = self.check()
        self.assertEqual(report.stale, [])
        self.assertIn("<style>", "\n".join(report.problems))
        self.assertIn("stylesheet link", "\n".join(report.problems))

    def test_a_shared_stylesheet_is_compared_by_the_file_each_page_names(self):
        self.edit(self.english, 'href="assets/docs.css"', 'href="./assets/docs.css"')
        self.edit(self.chinese, 'href="%sassets/docs.css"' % UP, 'href="%sassets/../assets/docs.css"' % UP)
        report = self.check()
        self.assertEqual((report.problems, report.stale, report.unstamped), ([], [], []))

    def test_a_shared_stylesheet_the_translation_does_not_link_is_noticed(self):
        self.edit(self.chinese, '<link href="%sassets/docs.css" rel="stylesheet">' % UP, "")
        self.assertIn("outside its sections", self.problems())

    def test_a_shared_script_the_translation_does_not_load_is_noticed(self):
        self.edit(self.chinese, '<script src="%sassets/docs.js"></script>' % UP, "")
        self.assertIn("outside its sections", self.problems())

    def test_a_renamed_class_the_rules_rely_on_is_named(self):
        for name in PAGES:
            for path in (self.docs / (name + ".html"), self.zh / "docs" / (name + ".html")):
                path.write_bytes(path.read_bytes().replace(b"syntax-content", b"syntax-text"))
        self.assertIn("class syntax-content", "\n".join(T.unused_rules(self.docs)))


class TranslationChangesTests(TemporaryPages):
    def test_a_missing_code_snippet_is_noticed_in_its_section(self):
        self.edit(self.chinese, "<code>COMMAND help</code>、", "")
        problems = self.problems()
        self.assertIn("Section 'docs-intro'", problems)
        self.assertIn("the tags differ", problems)

    def test_a_changed_code_snippet_is_noticed_in_its_section(self):
        self.edit(self.chinese, "<code>%s</code>" % self.b.code, "<code>%s_changed</code>" % self.b.code)
        problems = self.problems()
        self.assertIn("Section '%s'" % self.b.key, problems)
        self.assertIn("'%s' in English and '%s_changed' in the translation" % (self.b.code, self.b.code), problems)

    def test_a_translated_command_name_or_syntax_line_is_noticed(self):
        self.edit(self.chinese, '<div class="syntax-content">%s</div>' % self.a.syntax,
                  '<div class="syntax-content">译 %s</div>' % self.a.syntax)
        self.edit(self.chinese, '<span class="command-card-name">%s</span>' % self.b.name,
                  '<span class="command-card-name">译%s</span>' % self.b.name)
        problems = self.problems()
        self.assertIn("Section '%s'" % self.a.key, problems)
        self.assertIn("Section '%s'" % self.b.key, problems)

    def test_a_translated_example_line_is_noticed(self):
        self.edit(self.chinese, "<span>%s</span>" % self.a.example, "<span>译 %s</span>" % self.a.example)
        self.assertIn("Section '%s'" % self.a.key, self.problems())

    def test_a_translated_command_in_the_sidebar_list_is_noticed(self):
        link = '<a href="#%s"><span>%s</span></a>' % (self.a.key, self.a.name)
        self.edit(self.chinese, link, '<a href="#%s"><span>译%s</span></a>' % (self.a.key, self.a.name))
        self.assertIn("Section 'sidebar'", self.problems())

    def test_a_changed_id_class_or_link_is_noticed(self):
        self.edit(self.chinese, 'id="%s"' % self.c.key, 'id="%s-renamed"' % self.c.key)
        problems = self.problems()
        self.assertIn("lacks these sections of the English page: %s" % self.c.key, problems)
        self.assertIn("has sections the English page lacks: %s-renamed" % self.c.key, problems)

    def test_a_changed_class_or_link_target_is_noticed(self):
        self.edit(self.chinese, '<a href="#%s">' % self.a.key, '<a href="#%s2">' % self.a.key)
        self.sub(self.chinese, r'<div class="callout callout-warn">', '<div class="callout callout-warn extra">', count=1)
        problems = self.problems()
        self.assertIn("Section 'sidebar'", problems)
        self.assertIn("Section 'expr-interpreter'", problems)

    def test_sections_in_another_order_are_noticed(self):
        self.edit(self.chinese, 'id="%s"' % self.a.key, 'id="cmd-tmp"')
        self.edit(self.chinese, 'id="%s"' % self.b.key, 'id="%s"' % self.a.key)
        self.edit(self.chinese, 'id="cmd-tmp"', 'id="%s"' % self.b.key)
        self.assertIn("another order", self.problems())

    def test_a_changed_head_is_noticed(self):
        self.edit(self.chinese, '<meta name="theme-color" content="#18181b"', '<meta name="theme-color" content="#18181c"')
        self.assertIn("outside its sections", self.problems())

    def test_a_missing_language_font_link_is_noticed(self):
        self.edit(self.chinese, '<link href="language.css" rel="stylesheet">', "")
        problems = self.problems()
        self.assertIn("must link its language's font", problems)
        self.assertNotIn("outside its sections", problems)  # the link is the one the pages do not share

    def test_a_language_font_link_to_a_file_that_is_not_there_is_noticed(self):
        (self.zh / "docs" / "language.css").unlink()
        self.assertIn("which does not exist beside it", self.problems())

    def test_a_language_font_link_for_a_language_without_a_font_is_noticed(self):
        manifest = json.loads((self.zh / "language.json").read_text(encoding="utf-8"))
        del manifest["font"]
        (self.zh / "language.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.assertIn("names no font with a web_range for it", self.problems())

    def test_a_broken_language_json_is_named(self):
        (self.zh / "language.json").write_text("{not json", encoding="utf-8")
        self.assertIn("language.json can't be used", self.problems())

    def test_a_second_stylesheet_link_that_is_not_the_languages_is_noticed(self):
        self.edit(self.chinese, '<link href="language.css" rel="stylesheet">', '<link href="language.css" rel="stylesheet"><link href="extra.css" rel="stylesheet">')
        self.assertIn("outside its sections", self.problems())

    def test_a_changed_script_is_noticed(self):
        self.edit(self.chinese, "const originalText = btnElement.innerText;", "const originalText = btnElement.innerHTML;")
        problems = self.problems()
        self.assertIn("Section 'script'", problems)
        self.assertIn("its code differs", problems)

    def test_a_script_string_left_in_english_is_noticed(self):
        self.edit(self.chinese, '"已复制！"', '"Copied!"')
        self.assertIn("still English", self.problems())

    def test_a_theme_label_left_in_english_is_noticed(self):
        self.edit(self.chinese, 'data-label-dark="深色模式"', 'data-label-dark="Dark Mode"')
        self.assertIn('data-label-dark="Dark Mode" is shown to readers by docs/assets/docs.js but is still English', self.problems())
        self.edit(self.chinese, 'data-label-light="浅色模式"', 'data-label-light="Light Mode"')
        self.assertIn('data-label-light="Light Mode" is shown to readers', self.problems())

    def test_a_missing_theme_label_is_noticed(self):
        self.edit(self.chinese, ' data-label-light="浅色模式"', "")
        problems = self.problems()
        self.assertIn("Section 'topbar'", problems)
        self.assertIn("the tags differ", problems)

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
        self.assertIn("Section 'sidebar'", self.problems())

    def test_a_missing_fingerprint_is_listed(self):
        self.sub(self.chinese, r'(<section class="command-card" id="%s") data-source-sha256="[0-9a-f]{64}"' % self.c.key, r"\1")
        report = self.check()
        self.assertEqual(report.unstamped, [self.c.key])
        self.assertEqual((report.problems, report.stale), ([], []))

    def test_a_fingerprint_that_is_not_the_sections_is_stale(self):
        self.sub(self.chinese, r'(<section class="command-card" id="%s" data-source-sha256=")[0-9a-f]{64}' % self.c.key,
                 r"\g<1>" + "0" * 64)
        self.assertEqual(self.check().stale, [self.c.key])

    def test_a_link_to_the_other_page_must_reach_the_page_in_the_pages_own_language(self):
        # The Chinese command reference's Quick Start tab goes to the Chinese Quick Start beside it.
        self.edit(self.chinese, 'href="quickstart.html"', 'href="%squickstart.html"' % UP)
        self.assertIn("in its own language", self.problems())
        # ... and the English page's tab to the English one.
        self.edit(self.chinese, 'href="%squickstart.html"' % UP, 'href="quickstart.html"')
        self.edit(self.english, 'href="quickstart.html"', 'href="../src/resources/languages/zh_CN/docs/quickstart.html"')
        self.assertIn("in its own language", self.problems())


class QuickStartChangesTests(TemporaryPages):
    """The Quick Start has no <section>s: its sections are each divider with the elements that follow it."""

    PAGE = QUICK
    SENTENCE = "Moves the view across the network."

    def test_a_changed_english_sentence_names_its_section(self):
        self.edit(self.english, self.SENTENCE, "Moves the view over the network.")
        report = self.check()
        self.assertEqual((report.problems, report.stale, report.unstamped), ([], ["gui-viewer"], []))
        message = report.format()
        for expected in ("gui-viewer", "quickstart.html", "Update it in", "--page quickstart --language zh_CN"):
            self.assertIn(expected, message)
        self.assertNotIn("list_of_commands", message)

    def test_a_changed_sentence_in_a_card_a_note_and_a_footer_marks_the_section_each_belongs_to(self):
        for old, new, section in (
            ("Bridging Sequence Space: Macroscopic Clusters to Microscopic Residues", "Bridging Sequence Space", "overview"),
            ("About This Manual vs. Other Documents", "About This Manual", "page-header"),
            ("Licensed under Apache-2.0", "Licensed under the Apache License", "next-steps"),
            ("Step 1: Ingest &amp; Sanitize Your Protein Sequences", "Step 1: Ingest Your Protein Sequences", "hands-on-tutorial"),
        ):
            with self.subTest(section=section):
                self.setUp()
                self.edit(self.english, old, new)
                self.assertEqual(self.check().stale, [section])

    def test_a_changed_code_snippet_is_stale_and_unmatched(self):
        self.edit(self.english, "<code>color K188 red</code>", "<code>color K188 blue</code>", count=3)
        report = self.check()
        self.assertEqual(report.stale, ["overview", "gui-config", "hands-on-tutorial"])
        self.assertIn("never translated", "\n".join(report.problems))

    def test_a_card_added_after_a_divider_belongs_to_its_section(self):
        self.edit(
            self.english,
            '<!-- SECTION 3: TOOLS GUI -->',
            '<div class="manual-card"><p>A new card.</p></div>\n<!-- SECTION 3: TOOLS GUI -->',
        )
        report = self.check()
        self.assertEqual(report.stale, ["workflow"])
        self.assertIn("workflow", "\n".join(report.problems))

    def test_a_new_divider_with_an_id_is_a_new_section(self):
        self.edit(
            self.english,
            '<!-- SECTION 8: RELATED REFERENCES -->',
            '<div id="extra" class="section-divider"><span>7b. Extra</span></div>\n<!-- SECTION 8: RELATED REFERENCES -->',
        )
        self.assertIn("lacks these sections of the English page: extra", self.problems())

    def test_a_divider_that_loses_its_id_is_named(self):
        self.edit(self.english, '<div id="next-steps" class="section-divider">', '<div class="section-divider">')
        self.assertIn("has sections the English page lacks: next-steps", self.problems())

    def test_a_translated_command_in_a_snippet_is_noticed_in_its_section(self):
        self.edit(self.chinese, "<code>color K188 red</code>", "<code>颜色 K188 red</code>", count=3)
        problems = self.problems()
        for section in ("overview", "gui-config", "hands-on-tutorial"):
            self.assertIn("Section '%s'" % section, problems)

    def test_the_screenshots_must_stay_the_english_ones(self):
        self.edit(self.chinese, 'src="%sassets/emapssn_viewer_gui.png"' % UP, 'src="%sassets/other.png"' % UP)
        self.assertIn("Section 'gui-viewer'", self.problems())

    def test_a_changed_header_is_noticed_in_the_page_header_section(self):
        self.edit(self.chinese, '<header class="page-header"', '<header class="page-head"')
        self.assertIn("lacks these sections of the English page: page-header", self.problems())

    def test_a_text_node_left_in_english_outside_the_translated_paragraphs_is_not_hidden(self):
        # The check compares tags and code, not prose; the real translation is checked for English prose
        # by RealPagesTests. A tag lost from a card shows.
        self.edit(self.chinese, '<span class="badge badge-core">基础</span>', "基础")
        self.assertIn("Section 'overview'", self.problems())

    def test_stamping_and_showing_by_section_and_page_on_the_command_line(self):
        self.edit(self.english, self.SENTENCE, "Moves the view over the network.")
        code, out, _ = self.run_main("--page", QUICK)
        self.assertEqual(code, 1)
        self.assertIn("gui-viewer", out)
        self.assertNotIn("list_of_commands", out)
        code, out, _ = self.run_main("show", "gui-viewer")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("===== English: gui-viewer"), 1)
        self.assertIn("Moves the view over the network.", out)
        self.assertIn("在网络中移动视图。", out)
        code, out, _ = self.run_main("stamp", "gui-viewer")
        self.assertEqual(code, 0)
        self.assertIn("Stamped quickstart/zh_CN: gui-viewer", out)
        self.assertEqual(self.run_main()[0], 0)

    def test_a_section_both_pages_have_needs_the_page_named(self):
        for name in PAGES:
            self.edit(self.docs / (name + ".html"), 'title="Toggle Sidebar"', 'title="Toggle the sidebar"')
        code, out, err = self.run_main("show", "topbar")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("===== English: topbar"), 2)
        code, _, err = self.run_main("stamp", "topbar")
        self.assertEqual(code, 2)
        self.assertIn("name one with --page and --language", err)
        code, out, _ = self.run_main("stamp", "topbar", "--page", QUICK)
        self.assertEqual(code, 0)
        self.assertEqual(self.check().stale, [])
        self.assertEqual(self.check(COMMANDS).stale, ["topbar"])
        code, out, _ = self.run_main("stamp", "--all")
        self.assertEqual(code, 0)
        self.assertIn("list_of_commands/zh_CN: topbar", out)
        self.assertEqual(self.run_main()[0], 0)


class LanguageSwitchTests(TemporaryPages):
    def test_a_page_without_a_switch_is_named(self):
        self.sub(self.chinese, r'<div class="lang-switch">.*?</select>\s*</div>', "")
        self.assertIn("list_of_commands.html has no language switch", self.problems())

    def test_a_select_without_a_label_is_named(self):
        self.edit(self.chinese, ' aria-label="语言" title="语言"', ' title="语言"')
        self.assertIn("needs an aria-label", self.problems())

    def test_an_option_that_leads_to_a_file_that_is_not_the_other_page_is_named(self):
        self.edit(self.chinese, 'value="%slist_of_commands.html"' % UP, 'value="%squickstart.html"' % UP)
        self.assertIn("the switch's option for en has the value '%squickstart.html'" % UP, self.problems())

    def test_an_option_that_leads_to_a_page_that_does_not_exist_is_named(self):
        self.edit(self.english, 'value="../src/resources/languages/zh_CN/docs/list_of_commands.html"', 'value="zh_CN/missing.html"')
        self.assertIn("has the value 'zh_CN/missing.html'", self.problems())

    def test_the_switch_must_select_its_own_language(self):
        self.edit(self.chinese, ' data-lang="zh_CN" selected>', ' data-lang="zh_CN">')
        self.assertIn("must have zh-CN, the page's own language, selected", self.problems())

    def test_the_switch_must_not_select_the_other_language(self):
        self.edit(self.english, 'data-lang="zh_CN">', 'data-lang="zh_CN" selected>')
        self.assertIn("option for zh-CN must not be the selected one", self.problems())

    def test_the_switch_may_follow_the_reader_s_anchor(self):
        self.edit(self.english, 'docs/list_of_commands.html"', 'docs/list_of_commands.html#cmd-label"')
        self.assertTrue(self.check().ok, self.check().format())

    def test_the_switch_does_not_make_the_sidebar_stale(self):
        self.edit(self.english, 'aria-label="Language" title="Language"', 'aria-label="Language switch" title="Choose a language"')
        self.assertTrue(self.check().ok, self.check().format())

    def test_a_language_missing_from_a_switch_is_named(self):
        self.sub(self.english, r'\s*<option value="\.\./src/resources/languages/zh_CN/docs/list_of_commands.html"[^>]*>[^<]*</option>', "")
        self.assertIn("should list ['en', 'zh-CN'] in this order", self.problems())

    def test_the_options_must_sit_between_the_markers_generate_rewrites(self):
        marker = "<!-- languages:begin -->"
        self.edit(self.chinese, marker, "")
        self.assertIn("must all sit between <!-- languages:begin --> and <!-- languages:end -->", self.problems())
        self.edit(self.chinese, '<option value="%slist_of_commands.html"' % UP, marker + '<option value="%slist_of_commands.html"' % UP)
        self.assertTrue(self.check().ok, self.check().format())
        # An option outside the block is not rewritten with it.
        self.edit(self.english, "<!-- languages:end -->", "")
        self.edit(self.english, "</select>", "<!-- languages:end --></select>")
        self.assertTrue(self.check().ok, self.check().format())
        self.edit(self.english, "<!-- languages:begin -->", "")
        self.edit(self.english, "<select ", "<!-- languages:begin --><select ")
        self.assertIn("must all sit between", self.problems())

    def test_the_options_must_be_in_the_pages_order(self):
        self.sub(
            self.chinese,
            r'(\s*)(<option value="[^"]*list_of_commands.html"[^>]*>English</option>)(\s*)(<option value="list_of_commands.html"[^>]*>简体中文</option>)',
            r"\1\4\3\2",
        )
        self.assertIn("should list ['en', 'zh-CN'] in this order", self.problems())

    def test_a_language_folder_added_must_be_listed_by_every_page_that_it_holds(self):
        fr = self.languages / "fr" / "docs"
        fr.mkdir(parents=True)
        shutil.copyfile(self.chinese, fr / (COMMANDS + ".html"))
        problems = self.problems()
        self.assertIn("should list ['en', 'fr', 'zh-CN'] in this order", problems)
        self.assertEqual(problems.count("should list ['en', 'fr', 'zh-CN']"), 2)  # both pages
        self.assertEqual(self.check(QUICK).problems, [])  # the Quick Start is not translated into French
        self.assertEqual(len(T.generate(self.docs, self.languages, check=True)), 3)
        T.generate(self.docs, self.languages)
        self.assertTrue(self.check().ok, self.check().format())


class UpdateWorkflowTests(TemporaryPages):
    def test_translating_a_changed_section_and_stamping_it_brings_the_pages_back_in_step(self):
        self.edit(self.english, self.a.lead, self.reworded(self.a.lead))
        before = self.chinese.read_bytes().decode("utf-8")
        self.assertEqual(self.check().stale, [self.a.key])
        self.edit(self.chinese, self.a.zh_lead, self.a.zh_lead + "确实")
        self.assertEqual(self.check().stale, [self.a.key])  # editing the text does not record the review
        self.assertEqual(T.stamp_sections(self.english, self.chinese, [self.a.key]), [self.a.key])
        self.assertTrue(self.check().ok, self.check().format())
        after = self.chinese.read_bytes().decode("utf-8")
        old = re.findall(r'data-source-sha256="([0-9a-f]{64})"', before)
        new = re.findall(r'data-source-sha256="([0-9a-f]{64})"', after)
        self.assertEqual(len(old), len(new))
        self.assertEqual(len([1 for a, b in zip(old, new) if a != b]), 1)

    def test_stamping_a_section_that_does_not_exist_changes_nothing(self):
        before = self.chinese.read_bytes()
        with self.assertRaises(KeyError):
            T.stamp_sections(self.english, self.chinese, [self.a.key, "cmd-nonexistent"])
        self.assertEqual(self.chinese.read_bytes(), before)

    def test_stamping_adds_a_fingerprint_to_a_section_that_has_none(self):
        stamped = len(re.findall(r' data-source-sha256="[0-9a-f]{64}"', self.chinese.read_text(encoding="utf-8")))
        self.assertGreater(stamped, 30)
        self.sub(self.chinese, r' data-source-sha256="[0-9a-f]{64}"', "", count=stamped)
        self.assertEqual(len(self.check().unstamped), stamped)
        T.stamp_sections(self.english, self.chinese, self.check().unstamped)
        self.assertTrue(self.check().ok, self.check().format())

    def test_the_fingerprints_are_those_of_the_english_sections(self):
        for page in PAGES:
            stamps = T.english_fingerprints(self.docs / (page + ".html"))
            parsed = T.Page(self.zh / "docs" / (page + ".html"))
            for key, element in parsed.units:
                with self.subTest(page=page, section=key):
                    self.assertEqual(element.attrs.get("data-source-sha256"), stamps[key])

    def test_the_command_line_lists_stale_sections_and_stamps_them(self):
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertEqual(out.count("matches"), 2)
        self.edit(self.english, self.a.lead, self.reworded(self.a.lead))
        code, out, _ = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn(self.a.key, out)
        self.assertIn("stamp", out)
        code, out, _ = self.run_main("stamp")
        self.assertEqual(code, 1)
        self.assertIn(self.a.key, out)
        code, out, _ = self.run_main("stamp", "--all")
        self.assertEqual(code, 0)
        self.assertIn("Stamped list_of_commands/zh_CN: %s" % self.a.key, out)
        self.assertEqual(self.run_main()[0], 0)

    def test_the_command_line_refuses_to_stamp_while_the_pages_differ_in_more_than_text(self):
        self.edit(self.chinese, "<code>%s</code>" % self.b.code, "<code>%s_changed</code>" % self.b.code)
        self.edit(self.english, self.a.lead, self.reworded(self.a.lead))
        before = self.chinese.read_bytes()
        code, out, err = self.run_main("stamp", "--all")
        self.assertEqual(code, 1)
        self.assertIn(self.b.key, out)
        self.assertIn("Nothing was stamped", err)
        self.assertEqual(self.chinese.read_bytes(), before)

    def test_the_command_line_shows_both_sections(self):
        code, out, _ = self.run_main("show", self.a.key)
        self.assertEqual(code, 0)
        self.assertIn("===== English: %s" % self.a.key, out)
        self.assertIn("===== Translation: %s" % self.a.key, out)
        self.assertIn(self.a.lead, out)
        self.assertIn(self.a.zh_lead, out)

    def test_a_page_or_language_with_no_translation_is_a_usage_error(self):
        for argv in (["--page", "missing"], ["--language", "fr"]):
            code, _, err = self.run_main(*argv)
            self.assertEqual(code, 2)
            self.assertIn("No translated page", err)

    def test_a_section_no_page_has_is_a_usage_error(self):
        code, _, err = self.run_main("show", "no-such-section")
        self.assertEqual(code, 2)
        self.assertIn("No such section", err)

    def test_a_page_the_language_does_not_translate_is_left_out_of_the_check_and_of_the_dropdown(self):
        (self.zh / "docs" / (COMMANDS + ".html")).unlink()
        self.assertEqual([(page, code) for page, code, _, _ in T.translations(self.docs, self.languages)], [(QUICK, "zh_CN")])
        # The Chinese Quick Start's links to the command reference, which its language does not have, must
        # reach the English page instead.
        self.assertIn("in its own language", "\n".join(self.check(QUICK).problems))
        quick = self.zh / "docs" / (QUICK + ".html")
        text = quick.read_bytes().decode("utf-8")
        quick.write_bytes(text.replace('href="list_of_commands.html"', 'href="%slist_of_commands.html"' % UP).encode("utf-8"))
        self.assertEqual(self.check(QUICK).problems, [])
        code, out, _ = self.run_main()
        self.assertEqual((code, out.count("matches")), (0, 1))
        # The English command reference still offers the language until generate takes the option out.
        self.assertEqual([path.name for path, _ in T.generate(self.docs, self.languages, check=True)], ["list_of_commands.html"])
        T.generate(self.docs, self.languages)
        select = next(e for e in T.Page(self.english).root.walk() if e.tag == "select")
        self.assertEqual([o.attrs["data-lang"] for o in select.elements() if o.tag == "option"], ["en"])


class DiscoveryTests(TemporaryPages):
    """Translated pages are found in the language folders, by the English pages in docs/."""

    def found(self):
        return [(page, code) for page, code, _, _ in T.translations(self.docs, self.languages)]

    def test_each_language_folder_that_holds_an_english_page_is_found_for_it(self):
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])
        (self.languages / "de" / "docs").mkdir(parents=True)
        shutil.copyfile(self.chinese, self.languages / "de" / "docs" / (QUICK + ".html"))
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "de"), (QUICK, "zh_CN")])
        paths = {(page, code): (english, translated) for page, code, english, translated in T.translations(self.docs, self.languages)}
        self.assertEqual(paths[(QUICK, "de")], (self.docs / "quickstart.html", self.languages / "de" / "docs" / "quickstart.html"))

    def test_a_page_in_docs_beside_the_english_ones_is_not_a_translation(self):
        # Translations used to be docs/<language>/<page>.html.
        (self.docs / "fr").mkdir()
        shutil.copyfile(self.chinese, self.docs / "fr" / (COMMANDS + ".html"))
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])

    def test_a_folder_that_is_not_named_as_a_language_or_has_no_docs_is_left_out(self):
        for name in ("notes", "pt_br", "x"):
            (self.languages / name / "docs").mkdir(parents=True)
            shutil.copyfile(self.chinese, self.languages / name / "docs" / (COMMANDS + ".html"))
        (self.languages / "de").mkdir()
        (self.languages / "pt_BR" / "help").mkdir(parents=True)
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])

    def test_a_folder_named_like_a_page_is_no_page_and_a_page_needs_its_exact_name(self):
        (self.languages / "de" / "docs" / (QUICK + ".html")).mkdir(parents=True)
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])
        (self.languages / "es" / "docs").mkdir(parents=True)
        shutil.copyfile(self.chinese, self.languages / "es" / "docs" / "List_Of_Commands.html")
        self.assertEqual(self.found(), [(COMMANDS, "zh_CN"), (QUICK, "zh_CN")])

    def test_an_english_page_with_no_translation_is_not_checked(self):
        for path in (self.zh / "docs").glob("*.html"):
            path.unlink()
        self.assertEqual(self.found(), [])
        code, _, err = self.run_main()
        self.assertEqual((code, "No translated page" in err), (0, True))  # said, but nothing to check is not a failure

    def test_the_languages_a_page_is_offered_in_are_those_that_hold_it(self):
        (self.languages / "de" / "docs").mkdir(parents=True)
        shutil.copyfile(self.chinese, self.languages / "de" / "docs" / (QUICK + ".html"))
        self.assertEqual([code for code, _ in T.page_languages(self.docs / "quickstart.html", self.languages)], ["en", "de", "zh_CN"])
        self.assertEqual([code for code, _ in T.page_languages(self.docs / "list_of_commands.html", self.languages)], ["en", "zh_CN"])


class GenerateTests(TemporaryPages):
    """The language dropdowns and the language.css files, made from the language folders."""

    def pages(self):
        return [self.docs / (page + ".html") for page in PAGES] + [self.zh / "docs" / (page + ".html") for page in PAGES]

    def snapshot(self):
        return {path: path.read_bytes() for path in [*self.pages(), self.zh / "docs" / "language.css"]}

    def test_the_copies_are_as_generate_writes_them(self):
        self.assertEqual(T.generate(self.docs, self.languages, check=True), [])
        before = self.snapshot()
        self.assertEqual(T.generate(self.docs, self.languages), [])
        self.assertEqual(self.snapshot(), before)

    def test_a_dropdown_that_lacks_a_language_is_found_by_check_and_rewritten_by_generate(self):
        before = self.snapshot()
        path = self.docs / "quickstart.html"
        self.sub(path, r'\s*<option value="\.\./src/resources/languages/zh_CN/docs/quickstart.html"[^>]*>[^<]*</option>', "")
        stale = T.generate(self.docs, self.languages, check=True)
        self.assertEqual([(p, "dropdown" in w) for p, w in stale], [(path, True)])
        self.assertNotEqual(path.read_bytes(), before[path])  # --check changed nothing
        code, out, err = self.run_main("generate", "--check")
        self.assertEqual(code, 1)
        self.assertIn("quickstart.html: its language dropdown does not list", out)
        self.assertIn("Run `python src/resources/languages/Docs_Translation.py generate`", err)
        code, out, _ = self.run_main("generate")
        self.assertEqual(code, 0)
        self.assertIn("Wrote 1 file", out)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.run_main("generate", "--check")[0], 0)

    def test_a_stale_dropdown_is_rewritten_on_every_page_keeping_each_pages_line_endings(self):
        crlf = self.docs / "quickstart.html"
        crlf.write_bytes(crlf.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.edit(crlf, "English</option>", "Inglés</option>")
        self.edit(self.chinese, "简体中文</option>", "中文</option>")
        self.assertEqual(len(T.generate(self.docs, self.languages)), 2)
        text = crlf.read_bytes()
        self.assertIn(b">English</option>", text)
        self.assertNotIn(b"\n", text.replace(b"\r\n", b""))  # still all CRLF
        self.assertIn("简体中文</option>", self.chinese.read_text(encoding="utf-8"))

    def test_a_language_folder_added_is_listed_on_every_page_it_holds_and_only_there(self):
        fr = self.languages / "fr" / "docs"
        fr.mkdir(parents=True)
        (self.languages / "fr" / "language.json").write_text(json.dumps({"name": "Français"}), encoding="utf-8")
        for page in PAGES:
            shutil.copyfile(self.zh / "docs" / (page + ".html"), fr / (page + ".html"))
        self.edit(fr / (COMMANDS + ".html"), '<html lang="zh-CN">', '<html lang="fr">')
        (fr / (QUICK + ".html")).unlink()  # a language may translate only some pages
        stale = T.generate(self.docs, self.languages)
        self.assertEqual(sorted(path.relative_to(self.root).as_posix() for path, _ in stale), sorted([
            "docs/list_of_commands.html", "src/resources/languages/fr/docs/list_of_commands.html",
            "src/resources/languages/zh_CN/docs/list_of_commands.html",
        ]))
        for path in (self.english, self.chinese, fr / (COMMANDS + ".html")):
            parsed = T.Page(path)
            select = next(e for e in parsed.root.walk() if e.tag == "select")
            options = [o for o in select.elements() if o.tag == "option"]
            self.assertEqual([o.attrs["data-lang"] for o in options], ["en", "fr", "zh_CN"])
            self.assertEqual([o.text() for o in options], ["English", "Français", "简体中文"])
            self.assertEqual([o.attrs["lang"] for o in options], ["en", "fr", "zh-CN"])
            self.assertEqual([o.attrs["value"] for o in options if "selected" in o.attrs],
                             [{"docs": "list_of_commands.html", "fr": "list_of_commands.html", "zh_CN": "list_of_commands.html"}[
                                 "docs" if path == self.english else path.parent.parent.name]])
            for option in options:
                self.assertTrue((path.parent / option.attrs["value"]).resolve().is_file(), option.attrs["value"])
        # The Quick Start is not offered in French.
        quick = T.Page(self.docs / "quickstart.html")
        select = next(e for e in quick.root.walk() if e.tag == "select")
        self.assertEqual([o.attrs["data-lang"] for o in select.elements() if o.tag == "option"], ["en", "zh_CN"])
        # A language whose script the core fonts cover has no font, so no language.css is written.
        self.assertFalse((fr / "language.css").exists())
        self.assertEqual(self.check().problems, [])
        self.assertEqual(T.generate(self.docs, self.languages, check=True), [])

    def test_the_font_rules_are_made_from_language_json_as_the_viewers_pages_get_theirs(self):
        css = (self.zh / "docs" / "language.css").read_text(encoding="utf-8")
        manifest = json.loads((self.zh / "language.json").read_text(encoding="utf-8"))
        font = manifest["font"]
        rules = re.findall(r"@font-face \{[^}]*\}", css)
        self.assertEqual(len(rules), 2)
        for rule, weight, path in zip(rules, (400, 700), font["files"]):
            self.assertIn("font-family: 'EMAP-SSN Language';", rule)
            self.assertIn("font-weight: %d;" % weight, rule)
            self.assertIn("url('../../../fonts/desktop/%s') format('truetype')" % path, rule)
            self.assertIn("unicode-range: %s;" % font["web_range"], rule)
        # The url is relative to the stylesheet, so the file really is where the rule says.
        for path in font["files"]:
            self.assertTrue((ZH / ".." / ".." / ".." / "fonts" / "desktop" / path).resolve().is_file())

    def test_a_language_css_that_is_stale_is_found_and_rewritten(self):
        css = self.zh / "docs" / "language.css"
        before = css.read_bytes()
        css.write_bytes(before.replace(b"font-weight: 700", b"font-weight: 600"))
        stale = T.generate(self.docs, self.languages, check=True)
        self.assertEqual([(p.name, "font rules" in w) for p, w in stale], [("language.css", True)])
        self.assertEqual(self.run_main("generate", "--check")[0], 1)
        self.assertEqual(self.run_main("generate")[0], 0)
        self.assertEqual(css.read_bytes(), before)  # the file's own line ends are kept

    def test_a_language_css_keeps_the_line_ends_the_checkout_gave_it(self):
        css = self.zh / "docs" / "language.css"
        for ends in (b"\r\n", b"\n"):
            text = css.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ends)
            css.write_bytes(text.replace(b"font-weight: 700", b"font-weight: 600"))
            self.assertEqual(self.run_main("generate")[0], 0)
            self.assertEqual(css.read_bytes(), text)
            # Line ends alone are no difference: a CRLF checkout is as generate would write it.
            self.assertEqual(T.generate(self.docs, self.languages, check=True), [])

    def test_a_language_css_that_is_missing_is_written(self):
        (self.zh / "docs" / "language.css").unlink()
        self.assertEqual([p.name for p, _ in T.generate(self.docs, self.languages, check=True)], ["language.css"])
        T.generate(self.docs, self.languages)
        written = (self.zh / "docs" / "language.css").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(written, (ZH / "language.css").read_bytes().replace(b"\r\n", b"\n"))

    def test_changing_the_font_in_language_json_changes_language_css_with_it(self):
        manifest = json.loads((self.zh / "language.json").read_text(encoding="utf-8"))
        manifest["font"]["web_range"] = "U+4E00-9FFF"
        (self.zh / "language.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.assertEqual([p.name for p, _ in T.generate(self.docs, self.languages, check=True)], ["language.css"])
        T.generate(self.docs, self.languages)
        self.assertEqual((self.zh / "docs" / "language.css").read_text(encoding="utf-8").count("unicode-range: U+4E00-9FFF;"), 2)

    def test_a_language_that_names_no_font_has_its_stale_language_css_removed(self):
        manifest = json.loads((self.zh / "language.json").read_text(encoding="utf-8"))
        del manifest["font"]
        (self.zh / "language.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        stale = T.generate(self.docs, self.languages, check=True)
        self.assertEqual([p.name for p, _ in stale], ["language.css"])
        self.assertTrue((self.zh / "docs" / "language.css").exists())
        T.generate(self.docs, self.languages)
        self.assertFalse((self.zh / "docs" / "language.css").exists())

    def test_a_language_json_that_cannot_be_used_is_named_and_fails_generate(self):
        (self.zh / "language.json").write_text("{not json", encoding="utf-8")
        stale = T.generate(self.docs, self.languages, check=True)
        self.assertTrue(any("can't be used" in what for _, what in stale))
        code, out, _ = self.run_main("generate")
        self.assertEqual(code, 1)
        self.assertIn("can't be used", out)

    def test_a_page_without_the_block_for_the_dropdown_is_named(self):
        self.edit(self.docs / "quickstart.html", "<!-- languages:begin -->", "")
        stale = T.generate(self.docs, self.languages, check=True)
        self.assertEqual([(p.name, "no <!-- languages:begin -->" in w) for p, w in stale], [("quickstart.html", True)])
        code, out, _ = self.run_main("generate")
        self.assertEqual(code, 1)
        self.assertIn("quickstart.html: has no <!-- languages:begin -->", out)


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

    ENGLISH = DOCS / "list_of_commands.html"
    CHINESE = ZH / "list_of_commands.html"
    ENGLISH_QUICK = DOCS / "quickstart.html"
    CHINESE_QUICK = ZH / "quickstart.html"

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

    OTHER_LANGUAGE = "Array.from(document.querySelector('.lang-switch select').options).find(o => !o.defaultSelected).value"

    def switch_link(self):
        """Where the language dropdown would take the reader now, from the place being read."""
        return self.run_js("window.docsLanguageTarget(%s)" % self.OTHER_LANGUAGE)

    def follow_switch(self):
        """Chooses the other language in the dropdown, as a reader does, and waits for its page."""
        loop = self.QEventLoop()
        self.view.loadFinished.connect(lambda ok: loop.quit())
        self.run_js(
            "var select = document.querySelector('.lang-switch select'); select.value = %s; "
            "select.dispatchEvent(new Event('change')); 1" % self.OTHER_LANGUAGE
        )
        self.QTimer.singleShot(20000, loop.quit)
        loop.exec()
        self.pump(600)
        self.assertEqual(self.run_js("document.readyState"), "complete")

    def facts(self):
        return json.loads(self.run_js(
            "JSON.stringify({lang: document.documentElement.lang, hash: location.hash, "
            "theme: document.documentElement.getAttribute('data-theme'), "
            "highlighted: Array.from(document.querySelectorAll('.highlight-card')).map(e => e.id)})"
        ))

    def test_the_switch_carries_the_reader_s_place_to_the_other_language_and_back(self):
        to_chinese = "../src/resources/languages/zh_CN/docs/list_of_commands.html"
        self.open(self.ENGLISH, "cmd-label")
        self.assertEqual(self.facts()["lang"], "en")
        self.assertEqual(self.switch_link(), to_chinese + "#cmd-label")
        for section, anchor in (("cmd-print", "#cmd-print"), ("cmd-query", "#cmd-query"), ("expr-boolean", "#expr-boolean"), ("docs-intro", "")):
            with self.subTest(section=section):
                self.scroll_to(section)
                self.assertEqual(self.switch_link(), to_chinese + anchor)
        self.scroll_to("cmd-meta")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["hash"]), ("zh-CN", "#cmd-meta"))
        self.assertEqual(facts["highlighted"], ["cmd-meta"])
        self.assertEqual(self.switch_link(), UP + "list_of_commands.html#cmd-meta")
        self.scroll_to("cmd-logo")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["hash"]), ("en", "#cmd-logo"))

    def test_the_quick_start_switch_carries_the_reader_s_place_too(self):
        self.open(self.ENGLISH_QUICK)
        self.assertEqual(self.switch_link(), "../src/resources/languages/zh_CN/docs/quickstart.html")
        for section in ("gui-tools", "gui-viewer", "best-practices"):
            with self.subTest(section=section):
                self.scroll_to(section)
                self.assertEqual(self.switch_link(), "../src/resources/languages/zh_CN/docs/quickstart.html#" + section)
        self.scroll_to("hands-on-tutorial")
        self.follow_switch()
        facts = self.facts()
        self.assertEqual((facts["lang"], facts["hash"]), ("zh-CN", "#hands-on-tutorial"))
        self.assertEqual(self.switch_link(), UP + "quickstart.html#hands-on-tutorial")
        self.follow_switch()
        self.assertEqual(self.facts()["lang"], "en")

    def test_the_dropdown_shows_the_pages_own_language_again_after_going_back(self):
        for page, own in ((self.ENGLISH, "en"), (self.CHINESE, "zh-CN"), (self.ENGLISH_QUICK, "en"), (self.CHINESE_QUICK, "zh-CN")):
            with self.subTest(page=page.relative_to(ROOT).as_posix()):
                self.open(page)
                self.run_js("document.querySelector('.lang-switch select').value = %s; 1" % self.OTHER_LANGUAGE)
                self.run_js("window.dispatchEvent(new Event('pageshow')); 1")
                self.assertEqual(
                    self.run_js("document.querySelector('.lang-switch select').selectedOptions[0].getAttribute('lang')"), own
                )

    def test_the_dropdown_is_a_labelled_select_a_keyboard_can_reach(self):
        for page in (self.ENGLISH, self.CHINESE, self.ENGLISH_QUICK, self.CHINESE_QUICK):
            with self.subTest(page=page.relative_to(ROOT).as_posix()):
                self.open(page)
                self.assertTrue(self.run_js("document.querySelector('.lang-switch select').getAttribute('aria-label').length > 0"))
                self.assertTrue(self.run_js("document.querySelector('.lang-switch select').tabIndex === 0"))
                self.run_js("document.querySelector('.lang-switch select').focus(); 1")
                self.assertEqual(self.run_js("document.activeElement.tagName"), "SELECT")
                # The field draws a chevron, as the windows' combo boxes do.
                self.assertIn("data:image/svg+xml", self.run_js("getComputedStyle(document.querySelector('.lang-switch select')).backgroundImage"))

    def test_the_switch_finds_the_section_at_the_end_of_the_page_too(self):
        self.open(self.CHINESE, "cmd-agent")
        self.run_js("window.scrollTo(0, document.documentElement.scrollHeight)")
        self.pump(300)
        self.assertEqual(self.switch_link(), UP + "list_of_commands.html#cmd-agent")

    def choose_theme(self, theme):
        """Presses the page's theme button until the page shows `theme`."""
        if self.facts()["theme"] != theme:
            self.run_js("document.getElementById('theme-toggle-btn').click()")
        self.assertEqual(self.facts()["theme"], theme)

    def test_the_chosen_theme_goes_with_the_reader(self):
        for english in (self.ENGLISH, self.ENGLISH_QUICK):
            with self.subTest(page=english.name):
                self.open(english)
                self.choose_theme("dark")
                self.follow_switch()
                facts = self.facts()
                self.assertEqual((facts["lang"], facts["theme"]), ("zh-CN", "dark"))
                # The button speaks the page's own language.
                self.assertEqual(self.run_js("document.querySelector('#theme-toggle-btn .theme-label').textContent"), "浅色模式")
                self.choose_theme("light")
                self.follow_switch()
                facts = self.facts()
                self.assertEqual((facts["lang"], facts["theme"]), ("en", "light"))
                self.assertEqual(self.run_js("document.querySelector('#theme-toggle-btn .theme-label').textContent"), "Dark Mode")

    def test_the_search_finds_chinese_text_and_says_so_when_nothing_matches(self):
        self.open(self.CHINESE)

        # Two cards picked from the pages; each search is a phrase of one card's Chinese text,
        # which no other part of the page holds.
        first, second, _third = samples()

        def search(text):
            self.run_js(
                "var box = document.getElementById('search-input'); box.value = %r; box.dispatchEvent(new Event('input'))" % text
            )
            return {
                section: self.run_js("document.getElementById('%s').style.display" % section)
                for section in (first.key, second.key, "empty-search-state")
            }

        shown = search(second.zh_lead)
        self.assertEqual((shown[second.key], shown[first.key], shown["empty-search-state"]), ("block", "none", "none"))
        shown = search(first.zh_lead)
        self.assertEqual((shown[second.key], shown[first.key]), ("none", "block"))
        shown = search("没有这样的命令zzz")
        self.assertEqual(shown["empty-search-state"], "block")
        self.assertIn("没有匹配的命令", self.run_js("document.getElementById('empty-search-state').textContent"))
        search("")
        self.assertEqual(self.run_js("document.getElementById('%s').style.display" % first.key), "block")

    def test_the_quick_start_search_filters_its_topics_in_chinese(self):
        self.open(self.CHINESE_QUICK)
        self.run_js("var box = document.getElementById('search-input'); box.value = '教程'; box.dispatchEvent(new Event('input')); 1")
        shown = self.run_js(
            "Array.from(document.querySelectorAll('.menu-item')).filter(e => e.style.display !== 'none').map(e => e.getAttribute('data-target')).join(',')"
        )
        self.assertEqual(shown, "hands-on-tutorial")
        # A category with nothing left under it goes too.
        self.assertEqual(
            self.run_js("Array.from(document.querySelectorAll('.menu-category')).filter(e => e.style.display !== 'none').length"), 1
        )

    def test_each_page_draws_chinese_in_the_language_font_and_an_english_page_loads_none(self):
        for page, chinese in ((self.CHINESE, True), (self.CHINESE_QUICK, True), (self.ENGLISH, False), (self.ENGLISH_QUICK, False)):
            with self.subTest(page=page.relative_to(ROOT).as_posix()):
                self.open(page)
                self.pump(600)
                loaded = self.run_js(
                    "Array.from(document.fonts).filter(f => f.status === 'loaded' && f.unicodeRange.indexOf('4E00') >= 0)"
                    ".map(f => f.family.replace(/[\"']/g, '') + ' ' + f.weight).sort().join(',')"
                )
                self.assertEqual(loaded, "EMAP-SSN Language 400,EMAP-SSN Language 700" if chinese else "")


class ChineseFontTests(unittest.TestCase):
    """The Chinese pages draw with the application's own Noto Sans SC, which language.json names."""

    @classmethod
    def setUpClass(cls):
        try:
            from fontTools.ttLib import TTFont
        except ImportError as error:
            raise unittest.SkipTest("reading the font needs fontTools: %s" % error)
        cls.TTFont = TTFont
        cls.font = json.loads((LANGUAGES / "zh_CN" / "language.json").read_text(encoding="utf-8"))["font"]

    def used_characters(self):
        return set(T.cjk_characters(*[english for _, _, english, _ in T.translations()], *[path for _, _, _, path in T.translations()]))

    def test_the_characters_the_check_lists_are_those_the_language_font_is_asked_for(self):
        self.assertEqual(set(T.CJK_RANGES), set(unicode_ranges(self.font["web_range"])))

    def test_both_faces_cover_every_chinese_character_on_the_pages(self):
        used = self.used_characters()
        self.assertGreater(len(used), 600)
        for path in self.font["files"]:
            with self.subTest(file=path):
                cmap = self.TTFont(LANGUAGES.parent / "fonts" / "desktop" / path).getBestCmap()
                missing = sorted(char for char in used if ord(char) not in cmap)
                self.assertEqual(
                    missing, [],
                    "The language font lacks %d character(s) a page uses: %s. Use a character the font has (it is cut "
                    "down to GB2312; see src/resources/fonts/desktop/README.md)." % (len(missing), "".join(missing)),
                )

    def test_the_documentation_fonts_hold_no_chinese_of_their_own_any_more(self):
        fonts = DOCS / "fonts"
        self.assertEqual([p.name for p in fonts.iterdir() if "SC" in p.name or "zh_CN" in p.name], [])
        for name in ("LICENSE.fonts", "LICENSE.Noto"):
            self.assertNotIn("Noto Sans SC", (fonts / name).read_text(encoding="utf-8").split("Chinese (and any other script")[0])
        # The application's own licence still names it, as the pages draw in its files.
        self.assertIn("Noto Sans SC", (LANGUAGES.parent / "fonts" / "LICENSE.Noto").read_text(encoding="utf-8"))
        licence = (fonts / "LICENSE.Noto").read_text(encoding="utf-8")
        body = "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007"
        self.assertEqual(
            licence[licence.index(body):],
            (LANGUAGES.parent / "fonts" / "LICENSE.Noto").read_text(encoding="utf-8")[
                (LANGUAGES.parent / "fonts" / "LICENSE.Noto").read_text(encoding="utf-8").index(body):],
        )


if __name__ == "__main__":
    unittest.main()
