# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The documentation pages' shared look: docs/assets/docs.css and docs.js.

The stylesheet holds the Studio colours of the desktop windows and the Viewer's
web pages, copied (a static page can't import them), so these tests keep the
copies equal to desktop.Studio_Theme and to agent.html. The pages themselves must
link the shared files and stay self-contained: nothing from src/, nothing online.
"""
import os
from pathlib import Path
import re
import sys
import time
import unittest

os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import QUrl  # noqa: E402
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.Studio_Theme import LUCIDE_DIR, theme_tokens, web_page_properties  # noqa: E402

DOCS = ROOT / "docs"
STYLESHEET = DOCS / "assets" / "docs.css"
SCRIPT = DOCS / "assets" / "docs.js"
PAGES = ("quickstart.html",)
DECLARATION = re.compile(r"^\s*(--[a-z0-9-]+)\s*:\s*(.*?);\s*$", re.MULTILINE)

# The custom properties the pages' CSS reads from Studio_Theme.web_page_properties.
STUDIO_PROPERTIES = (
    "--bg-main", "--bg-card", "--bg-muted", "--bg-inset", "--border-color", "--field-border",
    "--field-border-hover", "--text-primary", "--text-soft", "--text-secondary", "--text-subtle",
    "--text-disabled", "--input-bg", "--primary", "--primary-hover", "--primary-pressed",
    "--primary-text", "--accent", "--accent-soft", "--selection", "--text-badge-bg",
    "--text-badge-border", "--text-badge-text", "--code-bg", "--code-border", "--code-text",
    "--code-inline-bg", "--code-inline-text", "--scroll", "--scroll-hover", "--select-chevron",
)
# The same names in Studio_Theme.TOKENS (the windows are light only): custom property -> token.
STUDIO_LIGHT_TOKENS = {
    "--track": "track",
    "--tint": "tint",
    "--tint-hover": "tint_hover",
    "--separator": "separator",
}
# What the Viewer's own pages (agent.html) hold the same way: the status colours and shapes.
PAGE_OWN_PROPERTIES = (
    "--success", "--success-soft", "--warning", "--warning-soft", "--danger", "--danger-soft",
    "--danger-border", "--shadow-card", "--shadow-control",
)
PAGE_OWN_ROOT_ONLY = ("--radius", "--radius-card")


def stylesheet_block(source, selector):
    """The custom properties of the stylesheet's (first) block for selector, by name."""
    start = source.index(selector + " {")
    return dict(DECLARATION.findall(source[start:source.index("\n}", start)]))


def page_block(page, selector):
    source = (ROOT / "src" / "web_ui" / page).read_text(encoding="utf-8")
    start = source.index(selector + " {")
    return dict(DECLARATION.findall(source[start:source.index("}", start)]))


def read(path):
    return Path(path).read_text(encoding="utf-8")


class StyleSheetColourTests(unittest.TestCase):
    BLOCKS = ((":root", False), ('[data-theme="dark"]', True))

    def test_the_studio_colours_are_the_default_theme_colours(self):
        source = read(STYLESHEET)
        for selector, dark in self.BLOCKS:
            with self.subTest(block=selector):
                own = stylesheet_block(source, selector)
                studio = web_page_properties("purple", dark=dark)
                for name in STUDIO_PROPERTIES:
                    self.assertIn(name, own, name)
                    self.assertEqual(own[name], studio[name], name)

    def test_the_tab_bar_tint_and_separator_are_the_windows(self):
        own = stylesheet_block(read(STYLESHEET), ":root")
        tokens = theme_tokens("purple")
        for name, token in STUDIO_LIGHT_TOKENS.items():
            with self.subTest(name=name):
                self.assertEqual(own[name], tokens[token])
        dark = stylesheet_block(read(STYLESHEET), '[data-theme="dark"]')
        for name in STUDIO_LIGHT_TOKENS:
            self.assertIn(name, dark)

    def test_the_status_colours_are_the_viewer_pages(self):
        source = read(STYLESHEET)
        for selector, dark in self.BLOCKS:
            with self.subTest(block=selector):
                own = stylesheet_block(source, selector)
                page = page_block("agent.html", selector)
                names = PAGE_OWN_PROPERTIES + (() if dark else PAGE_OWN_ROOT_ONLY)
                for name in names:
                    self.assertEqual(own[name], page[name], name)

    def test_the_theme_colour_and_the_status_colours_come_in_both_modes(self):
        light = stylesheet_block(read(STYLESHEET), ":root")
        dark = stylesheet_block(read(STYLESHEET), '[data-theme="dark"]')
        # Only what the light block has, besides sizes and fonts, may be missing from dark.
        shared_only_in_light = {"--radius", "--radius-card", "--font-family", "--font-mono",
                                "--topbar-height", "--sidebar-width"}
        self.assertEqual(sorted(set(light) - set(dark) - shared_only_in_light), [])

    def test_every_colour_the_rules_read_is_declared(self):
        source = read(STYLESHEET)
        declared = set(stylesheet_block(source, ":root")) | {"--icon-copy"}
        used = set(re.findall(r"var\((--[a-z0-9-]+)", source))
        self.assertEqual(sorted(used - declared), [])

    def test_only_the_weights_the_windows_use_are_set(self):
        # The bundled language fonts have no Medium face, so 500 would draw as bold.
        weights = set(re.findall(r"font-weight:\s*(\d+)", read(STYLESHEET)))
        self.assertLessEqual(weights, {"400", "600"})

    def test_the_fonts_are_the_vendored_noto_sets(self):
        source = read(STYLESHEET)
        self.assertIn("--font-family: 'Noto Sans'", source)
        self.assertIn("--font-mono: 'Noto Sans Mono'", source)
        for name in ("Inter", "Fira Code", "fonts.googleapis.com", "fonts.gstatic.com"):
            self.assertNotIn(name, source)


class PageTests(unittest.TestCase):
    EXTERNAL = re.compile(r"""(?:src|href|action)\s*=\s*["']?\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//|@import""", re.I)

    def pages(self):
        for name in PAGES:
            yield name, read(DOCS / name)

    def test_each_page_links_the_shared_style_and_script_and_has_none_of_its_own(self):
        for name, html in self.pages():
            with self.subTest(page=name):
                self.assertIn('<link href="fonts/fonts.css" rel="stylesheet">', html)
                self.assertIn('<link href="assets/docs.css" rel="stylesheet">', html)
                self.assertIn('<script src="assets/docs.js"></script>', html)
                self.assertNotIn("<style", html)
                # The theme and its remembered choice are the shared script's.
                self.assertNotIn("localStorage", html)
                self.assertIn('name="viewport"', html)
                self.assertEqual(len(re.findall(r'<meta name="theme-color"', html)), 2)

    def test_the_pages_are_self_contained_and_offline(self):
        for name, html in self.pages():
            with self.subTest(page=name):
                self.assertIsNone(self.EXTERNAL.search(html))
                for name_of_file in ("fonts.googleapis.com", "fonts.gstatic.com", "cdn."):
                    self.assertNotIn(name_of_file, html)
                for target in re.findall(r'<(?:link|script|img)\b[^>]*?(?:src|href)="([^"#?]+)"', html):
                    self.assertTrue((DOCS / target).is_file(), target)
        for name in ("docs.css", "docs.js"):
            with self.subTest(file=name):
                text = read(DOCS / "assets" / name)
                self.assertIsNone(self.EXTERNAL.search(text))
                # Only data: URLs and the vendored fonts' own sheet are loaded from a stylesheet.
                for url in re.findall(r"url\(\s*[\"']?([^\"')]+)", text):
                    self.assertTrue(url.startswith("data:"), url)

    def test_every_in_page_link_has_a_target(self):
        for name, html in self.pages():
            with self.subTest(page=name):
                ids = set(re.findall(r'\bid="([^"]+)"', html))
                for anchor in re.findall(r'href="#([^"]+)"', html):
                    self.assertIn(anchor, ids)
                for target in re.findall(r'data-target="([^"]+)"', html):
                    self.assertIn(target, ids)

    def test_the_quick_start_keeps_the_anchors_other_pages_link_to(self):
        ids = set(re.findall(r'\bid="([^"]+)"', read(DOCS / "quickstart.html")))
        for anchor in ("overview", "workflow", "gui-tools", "gui-config", "gui-viewer",
                       "hands-on-tutorial", "best-practices", "next-steps"):
            self.assertIn(anchor, ids)

    def test_the_top_bar_names_both_pages_and_marks_the_one_shown(self):
        for name, html in self.pages():
            with self.subTest(page=name):
                bar = html[html.index('<header class="topbar">'):html.index("</header>")]
                tabs = re.findall(r'<a class="tab" href="([^"]+)"( aria-current="page")?>([^<]+)</a>', bar)
                self.assertEqual([tab[0] for tab in tabs], ["quickstart.html", "list_of_commands.html"])
                self.assertEqual([tab[0] for tab in tabs if tab[1]], [name])
                self.assertIn('id="theme-toggle-btn"', bar)
                self.assertIn('id="toggle-sidebar-btn"', bar)
                self.assertIn('id="sidebar"', html)

    def test_each_inline_icon_is_the_bundled_lucide_icon(self):
        for name, html in self.pages():
            with self.subTest(page=name):
                symbols = dict(re.findall(r'<symbol id="i-([a-z0-9-]+)"[^>]*>(.*?)</symbol>', html, re.S))
                self.assertTrue(symbols)
                for icon, shapes in symbols.items():
                    svg = (LUCIDE_DIR / f"{icon}.svg").read_text(encoding="utf-8")
                    inner = svg[svg.index(">", svg.index("<svg")) + 1:svg.rindex("</svg>")]
                    self.assertEqual(re.sub(r"\s", "", shapes), re.sub(r"\s", "", inner), icon)
                for used in re.findall(r'<use href="#i-([a-z0-9-]+)"', html):
                    self.assertIn(used, symbols)
                self.assertIn("ISC licence", html)
                # A symbol the page never draws is dead weight, except the theme button's other icon.
                for icon in symbols:
                    if icon not in ("sun", "moon"):
                        self.assertIn(f'href="#i-{icon}"', html, icon)


class ScriptTests(unittest.TestCase):
    def test_the_remembered_theme_key_and_its_guard(self):
        source = read(SCRIPT)
        self.assertIn('"emapssn_docs_theme"', source)
        # Storage can be missing or throw: every access is in a try block.
        lines = source.splitlines()
        accesses = [number for number, line in enumerate(lines) if "localStorage." in line]
        self.assertTrue(accesses)
        for number in accesses:
            self.assertTrue("try" in " ".join(lines[max(0, number - 1):number + 1]), lines[number])


class BrowserTests(unittest.TestCase):
    """The quick start in Qt WebEngine: the theme, the sidebar and the narrow layout."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.view = QWebEngineView()
        cls.view.resize(1000, 700)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()

    def js(self, source):
        result = []
        self.view.page().runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + 8
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(result, source)
        return result[0]

    def open(self, width=1000):
        self.view.resize(width, 700)
        loaded = []
        callback = loaded.append
        self.view.loadFinished.connect(callback)
        self.view.load(QUrl.fromLocalFile(str(DOCS / "quickstart.html")))
        deadline = time.monotonic() + 15
        while not loaded and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.view.loadFinished.disconnect(callback)
        self.assertEqual(loaded, [True])
        self.js("localStorage.removeItem('emapssn_docs_theme')")

    def test_the_theme_button_switches_and_remembers_the_theme(self):
        self.open()
        first = self.js("document.documentElement.getAttribute('data-theme')")
        self.assertIn(first, ("light", "dark"))
        other = "dark" if first == "light" else "light"
        self.js("document.getElementById('theme-toggle-btn').click(); 1")
        self.assertEqual(self.js("document.documentElement.getAttribute('data-theme')"), other)
        self.assertEqual(self.js("localStorage.getItem('emapssn_docs_theme')"), other)
        # The button names the theme a click switches to, with its icon.
        label = self.js("document.querySelector('#theme-toggle-btn .theme-label').textContent")
        self.assertEqual(label, "Light Mode" if other == "dark" else "Dark Mode")
        self.assertEqual(
            self.js("document.querySelector('#theme-toggle-btn use').getAttribute('href')"),
            "#i-sun" if other == "dark" else "#i-moon",
        )
        # The page paints with the theme's colours.
        painted = self.js("getComputedStyle(document.body).backgroundColor")
        self.assertEqual(painted, "rgb(9, 9, 11)" if other == "dark" else "rgb(250, 250, 250)")
        # A reload keeps the choice.
        self.open_keeping_storage()
        self.assertEqual(self.js("document.documentElement.getAttribute('data-theme')"), other)

    def open_keeping_storage(self):
        loaded = []
        callback = loaded.append
        self.view.loadFinished.connect(callback)
        self.view.reload()
        deadline = time.monotonic() + 15
        while not loaded and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.view.loadFinished.disconnect(callback)
        self.assertEqual(loaded, [True])

    def test_a_phone_has_no_horizontal_scroll_and_a_sidebar_drawer(self):
        for width in (320, 390, 768):
            with self.subTest(width=width):
                self.open(width)
                self.assertLessEqual(
                    self.js("document.documentElement.scrollWidth - document.documentElement.clientWidth"), 0
                )
        self.open(390)
        right = "Math.round(document.getElementById('sidebar').getBoundingClientRect().right)"
        self.assertEqual(self.js(right), 0)
        self.js("document.getElementById('toggle-sidebar-btn').click(); 1")
        self.wait_for(f"{right} > 200")
        self.assertEqual(self.js("document.getElementById('toggle-sidebar-btn').getAttribute('aria-expanded')"), "true")
        # Choosing a section closes the drawer.
        self.js("document.querySelector('.menu-item a').click(); 1")
        self.assertEqual(self.js("document.body.classList.contains('sidebar-open')"), False)
        self.wait_for(f"{right} === 0")

    def wait_for(self, expression, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.js(expression):
                return
            time.sleep(0.03)
        self.fail("Timed out: " + expression)

    def test_the_sidebar_highlights_the_section_being_read(self):
        self.open(1000)
        self.js("document.documentElement.style.scrollBehavior = 'auto'; document.getElementById('gui-config').scrollIntoView(); 1")
        self.wait_for("(document.querySelector('.menu-item.active') || document.body).getAttribute('data-target') === 'gui-config'")


if __name__ == "__main__":
    unittest.main()
