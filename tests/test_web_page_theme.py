# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's Agent and Meta Data pages in the Viewer's theme colour.

Each page sets its own values of the properties that take the theme colour
(desktop.Studio_Theme.WEB_PAGE_PROPERTIES), which are Purple's, and the
Viewer's web server ends the page's head with the theme colour's
(web_ui.Web_Server.page_in_theme_color), in any language. The Fold View page
is Mol*'s own interface and keeps its look.
"""

import http.client
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from http.server import ThreadingHTTPServer  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from desktop.Desktop_App import install_translations  # noqa: E402
from desktop.Studio_Theme import (  # noqa: E402
    DEFAULT_THEME_COLOR,
    WEB_PAGE_PROPERTIES,
    set_theme_color,
    web_page_properties,
    web_theme_css,
)
from web_ui.Web_Server import MIME_TYPES, WebServerHandler, content_security_policy, page_in_theme_color  # noqa: E402
from tests.translation_fixtures import pseudo_language  # noqa: E402

PAGES = SRC / "web_ui"
THEMED = ("agent.html", "meta.html")
DECLARATION = re.compile(r"^\s*(--[a-z0-9-]+)\s*:\s*(.*?);\s*$", re.MULTILINE)


class PageHandler(WebServerHandler):
    """Serves the bundled pages through the Viewer's own serve_file."""

    def do_GET(self):
        file = PAGES / self.path.lstrip("/")
        if file.name not in (*THEMED, "esmfold.html"):
            self.send_error(404)
            return
        self.serve_file(str(file), MIME_TYPES[file.suffix])


def declared(page, selector):
    """The custom properties the page's own block for selector declares, with their values."""
    source = (PAGES / page).read_text(encoding="utf-8")
    start = source.index(selector + " {")
    return dict(DECLARATION.findall(source[start:source.index("}", start)]))


class ServedPageThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
        self.server.viewer = SimpleNamespace()
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        # A colour other than the pages' own, Purple.
        set_theme_color("teal")
        self.addCleanup(set_theme_color, DEFAULT_THEME_COLOR)

    def get(self, page):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        try:
            connection.request("GET", f"/{page}")
            response = connection.getresponse()
            return response, response.read()
        finally:
            connection.close()

    def assert_themed(self, page):
        response, body = self.get(page)
        self.assertEqual(response.status, 200)
        shown = body.decode("utf-8")
        head, end, _ = shown.partition("</head>")
        styles = f"<style>\n{web_theme_css('teal')}\n</style>\n"
        self.assertTrue(end and head.endswith(styles), head[-300:])
        self.assertEqual(shown.count(web_theme_css("teal")), 1)
        self.assertEqual(response.getheader("Content-Security-Policy"), content_security_policy(page, body))
        return body

    def test_the_agent_and_meta_data_pages_end_their_head_with_the_theme_colour(self):
        for page in THEMED:
            with self.subTest(page=page):
                body = self.assert_themed(page)
                # An inline style leaves the policy as the page's own bytes have it.
                self.assertEqual(content_security_policy(page, body),
                                 content_security_policy(page, (PAGES / page).read_bytes()))

    def test_a_page_in_another_language_takes_the_theme_colour_too(self):
        for language in ("pseudo", "zh_CN"):
            with self.subTest(language=language):
                if language == "pseudo":
                    pseudo_language(self, self.app)
                else:
                    install_translations(self.app, language)
                    self.addCleanup(install_translations, self.app, None)
                for page in THEMED:
                    with self.subTest(page=page):
                        self.assertNotEqual(self.assert_themed(page), (PAGES / page).read_bytes())

    def test_the_fold_view_page_keeps_its_look(self):
        response, body = self.get("esmfold.html")
        self.assertEqual(body, (PAGES / "esmfold.html").read_bytes())
        self.assertEqual(response.getheader("Content-Security-Policy"), content_security_policy("esmfold.html", body))

    def test_only_the_viewers_own_pages_take_it(self):
        page = b"<html><head><style>p {}</style></head><body></body></html>"
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(page_in_theme_color(os.path.join(folder, "agent.html"), page), page)
        script = SRC / "resources" / "agent" / "attachments.js"
        self.assertEqual(page_in_theme_color(str(script), script.read_bytes()), script.read_bytes())
        self.assertEqual(page_in_theme_color(str(PAGES / "esmfold.html"), page), page)
        self.assertIn(web_theme_css(), page_in_theme_color(str(PAGES / "meta.html"), page).decode("utf-8"))


class PageFallbackTests(unittest.TestCase):
    """A page looks the same whether or not the server adds the theme colour's styles."""

    BLOCKS = ((":root", False), ('[data-theme="dark"]', True))

    def test_the_pages_own_values_are_purples(self):
        for page in THEMED:
            source = (PAGES / page).read_text(encoding="utf-8")
            used = set(re.findall(r"var\((--[a-z0-9-]+)", source))
            for selector, dark in self.BLOCKS:
                with self.subTest(page=page, block=selector):
                    purple = web_page_properties("purple", dark=dark)
                    own = {name: value for name, value in declared(page, selector).items() if name in purple}
                    self.assertTrue(own)
                    self.assertEqual(own, {name: purple[name] for name in own})
                    # Each theme property the page draws with has a value of its own.
                    self.assertEqual(sorted(used & set(purple) - set(own)), [])

    def test_every_theme_property_is_one_a_page_declares(self):
        properties = {entry[0] for entry in WEB_PAGE_PROPERTIES}
        for selector, _dark in self.BLOCKS:
            with self.subTest(block=selector):
                pages = set().union(*(declared(page, selector) for page in THEMED))
                self.assertEqual(sorted(properties - pages), [])


if __name__ == "__main__":
    unittest.main()
