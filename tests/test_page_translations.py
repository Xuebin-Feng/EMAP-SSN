# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The web pages the Viewer serves, shown in the Viewer's language.

A page marks its texts (web_ui/Page_Texts.py), the update command files
them in the catalogs (tests/test_update_translations.py), and the Viewer's
web server writes each one's translation in place of its English as it
serves the page or one of its scripts. These tests read and translate
pages, serve them, and open the Agent page in Qt WebEngine under the
pseudo-language, where every ASCII letter a catalog supplies is accented:
a plain one the page shows, once the values a test filled in are taken
out, came from no catalog. The messages the Viewer sends the page are
checked the same way. The ESMFold page stays English: it is Mol*'s own
interface, which has no translations.
"""

import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import urlsplit

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Set before Qt WebEngine starts Chromium, as tests/test_agent_composer.py does.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for folder in (SRC, ROOT):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from http.server import ThreadingHTTPServer

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from utilities.Localization import Message, display_text, pseudo_translate
from web_ui import Page_Texts
from web_ui.Page_Texts import (
    PAGE_CONTEXTS,
    language_tag,
    page_context,
    read_page,
    translated_file,
    translated_source,
)
from web_ui.Web_Server import MIME_TYPES, WebServerHandler, content_security_policy, page_in_viewer_language
from tests.agent_fixtures import image_bytes
from tests.translation_fixtures import pseudo_language, unmarked_page_texts

LINK = re.compile(r"https?://\S+")


def pseudo(context, text):
    """The pseudo-language as translated_file's translate: the context is the catalog's business."""
    return pseudo_translate(text)


class ReadPageTests(unittest.TestCase):
    def texts(self, source, suffix=".html"):
        page = read_page(source, suffix)
        self.assertEqual(page.problems, [])
        for text in page.texts:
            self.assertTrue(source[text.start:text.end])
        return [(text.kind, text.text) for text in page.texts]

    def test_a_marked_elements_content_is_one_text_with_its_markup(self):
        source = '<p>\r\n<b data-i18n>\r\n   Range <code>min-max</code>,\r\n   say  </b> left</p>'
        page = read_page(source, ".html")
        (text,) = page.texts
        self.assertEqual((text.kind, text.text, text.line), ("content", "Range <code>min-max</code>, say", 3))
        self.assertEqual(source[text.start:text.end], "Range <code>min-max</code>,\r\n   say")
        self.assertEqual(page.unmarked, [(4, "left")])

    def test_a_text_attribute_with_a_letter_is_a_text_wherever_it_is(self):
        source = ('<div title="Close &amp; go" class="Plain" data-note="Plain">'
                  "<input placeholder='Ask...' value=\"Plain\"><img alt=Logo>"
                  '<span aria-label="Two&#10;lines" title="✕"></span></div>')
        self.assertEqual(self.texts(source), [
            ("attribute", "Close & go"), ("attribute", "Ask..."), ("attribute", "Logo"), ("attribute", "Two\nlines"),
        ])

    def test_attributes_inside_a_marked_element_belong_to_its_text(self):
        source = '<p data-i18n title="Own">Say <abbr title="Inner">it</abbr></p>'
        self.assertEqual(self.texts(source), [("attribute", "Own"), ("content", 'Say <abbr title="Inner">it</abbr>')])

    def test_a_script_marks_its_texts_with_t(self):
        script = (
            "// t('A comment')\n"
            "/* t(\"Another\") */\n"
            "const a = t(\"Plain\"), b = t('Single {name}', {name: 'Plain value'});\n"
            "const c = t(\"Say \\\"hi\\\"\\u0021\\x21\"), d = obj.t('Method'), e = t;\n"
            "const f = `Tab ${t(\"Inside a template\")} ${`nested ${t('Deeper')}`}`;\n"
            "const g = text.replace(/t\\(\"x/g, '') / 2;\n"
            "function t(text) { return text; }\n"
            "const h = 'it\\'s t(\"not\") marked';\n"
        )
        self.assertEqual(self.texts(script, ".js"), [
            ("script", "Plain"), ("script", "Single {name}"), ("script", 'Say "hi"!!'),
            ("script", "Inside a template"), ("script", "Deeper"),
        ])
        page = read_page(f"<script>\n{script}</script><p data-i18n>Last</p>", ".html")
        self.assertEqual([(text.kind, text.text, text.line) for text in page.texts][-2:],
                         [("script", "Deeper", 6), ("content", "Last", 10)])

    def test_unmarked_text_is_what_an_element_shows_outside_marked_ones(self):
        source = ("<title>Page</title><style>p { content: 'Style'; }</style>"
                  "<script>const s = 'Script';</script><p translate=\"no\">Name <b>Bold</b></p>"
                  "<template><p>Template</p></template><p data-i18n>Marked</p><p>  Plain\n text </p>&times;")
        self.assertEqual(read_page(source, ".html").unmarked, [(1, "Page"), (1, "Plain text")])

    def test_texts_no_catalog_can_list_are_problems(self):
        source = (
            "<br data-i18n>\n"
            "<p data-i18n>Outer <b data-i18n>inner</b></p>\n"
            "<p data-i18n> <span class=\"x\"></span> </p>\n"
            "<p data-i18n>%n files</p>\n"
            "<script>\nt(`Template ${x}`); t(name); t('Sum ' + x); t();\n</script>\n"
            "<script>const broken = 'never closed;</script>\n"
        )
        problems = read_page(source, ".html").problems
        self.assertEqual([line for line, _ in problems], [1, 2, 3, 4, 6, 6, 6, 6, 8])
        self.assertIn("<br> has none", problems[0][1])
        self.assertIn("inside another marked element", problems[1][1])
        self.assertIn("holds no text", problems[2][1])
        self.assertIn("can't be counted", problems[3][1])
        self.assertTrue(all("plain string literal" in explanation for _, explanation in problems[4:8]))
        self.assertIn("can't be read", problems[8][1])


class TranslatedFileTests(unittest.TestCase):
    def test_each_text_is_written_as_its_place_needs(self):
        source = ('<html lang="en"><p data-i18n>Say <b>it</b></p><i title=\'Quote\'></i>'
                  '<script>t("Script");</script></html>')
        translations = {"Say <b>it</b>": "Sag <b>es</b>", "Quote": 'A "q" & <b>', "Script": "</script><b>"}
        self.assertEqual(
            translated_source(source, ".html", translations.get),
            '<html lang="en"><p data-i18n>Sag <b>es</b></p><i title="A &quot;q&quot; &amp; &lt;b&gt;"></i>'
            '<script>t("\\u003c/script>\\u003cb>");</script></html>',
        )
        script = "a = t('Line'); b = t(\"Sep\");"
        self.assertEqual(translated_source(script, ".js", {"Line": "x\u2028y", "Sep": 'q"'}.get),
                         'a = t("x\\u2028y"); b = t("q\\"");')

    def test_only_the_pages_files_are_translated(self):
        page = SRC / "web_ui" / "agent.html"
        body = page.read_bytes()
        self.assertEqual(translated_file(page, body, None, pseudo), body)
        for other in (SRC / "web_ui" / "esmfold.html", SRC / "web_ui" / "page_text.js",
                      SRC / "resources" / "agent" / "marked.umd.js"):
            with self.subTest(other=other.name):
                self.assertIsNone(page_context(other))
                self.assertEqual(translated_file(other, other.read_bytes(), "pseudo", pseudo), other.read_bytes())
        self.assertIsNone(page_context(ROOT / "agent.html"))
        self.assertEqual({page_context(SRC / relative) for relative in PAGE_CONTEXTS}, set(PAGE_CONTEXTS.values()))

    def test_a_real_language_names_itself_in_the_page(self):
        page = SRC / "web_ui" / "agent.html"
        body = page.read_bytes()
        self.assertIn('<html lang="en">', body.decode("utf-8"))
        shown = translated_file(page, body, "zh_CN", lambda context, text: text).decode("utf-8")
        self.assertIn('<html lang="zh-CN">', shown)
        self.assertIn('<html lang="en">', translated_file(page, body, "pseudo", pseudo).decode("utf-8"))
        self.assertEqual([language_tag(code) for code in ("zh_CN", "de", "pt_BR", "sr_Latn_RS", "pseudo", None)],
                         ["zh-CN", "de", "pt-BR", "sr-Latn-RS", None, None])

    def test_every_page_file_reads_without_problems(self):
        for relative in PAGE_CONTEXTS:
            path = SRC / relative
            with self.subTest(page=relative):
                page = read_page(path.read_text(encoding="utf-8"), path.suffix)
                self.assertEqual(page.problems, [])
                self.assertGreater(len(page.texts), 10)


class PageCheckTests(unittest.TestCase):
    """unmarked_page_texts (tests/translation_fixtures.py) on the real pages and a made-up one."""

    def test_every_text_the_pages_show_is_marked(self):
        for relative in PAGE_CONTEXTS:
            path = SRC / relative
            with self.subTest(page=relative):
                self.assertEqual(unmarked_page_texts(path.read_text(encoding="utf-8"), path.suffix), [])

    def test_the_check_finds_each_way_a_script_shows_text(self):
        script = "\n".join([
            "a.textContent = 'Set';",
            "a.innerHTML = `<b class=\"x\">Template</b> ${name}`;",
            "a.title = flag === 'compared' ? t('Marked') : 'Either';",
            "a.placeholder = t('Marked {x}', {x: 'Value'});",
            "a.setAttribute('aria-label', 'Labelled'); a.setAttribute('class', 'not text');",
            "row.append('Appended ', code);",
            "window.confirm('Confirmed?'); throw new Error(`Failed ${t('Inner')}`);",
            "columns.push({title: 'Column', field: 'Not shown', placeholder: \"Empty\"});",
            "a.innerHTML = '<div class=\"no letters shown\"></div>'; a.textContent = x.replaceAll('_', ' ');",
            "a.title = url || 'https://example.org/link'; console.error('Developer');",
            "a.textContent = name.replace('Prefix', '');",
        ])
        self.assertEqual(sorted(text for _, text in unmarked_page_texts(script, ".js")), sorted([
            "Set", "<b class=\"x\">Template</b> ", "Either", "Value", "Labelled", "Appended ", "Confirmed?",
            "Failed ", "Column", "Empty",
        ]))


class FixtureHandler(WebServerHandler):
    """Serves the files a test names, through the Viewer's own serve_file."""

    files = {}
    body = None

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/agent_resource/model_card.json":
            # Never the real file, which holds the user's API keys.
            status, body = type(self).body
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        file = type(self).files.get(path)
        if file is None:
            self.send_error(404)
            return
        self.serve_file(str(file), MIME_TYPES.get(file.suffix, "application/octet-stream"))


class AgentPageHandler(FixtureHandler):
    files = {
        "/agent.html": SRC / "web_ui" / "agent.html",
        "/page_text.js": SRC / "web_ui" / "page_text.js",
        "/agent_resource/attachments.js": SRC / "resources" / "agent" / "attachments.js",
        "/agent_resource/marked.umd.js": SRC / "resources" / "agent" / "marked.umd.js",
        "/fonts/fonts.css": SRC / "resources" / "fonts" / "fonts.css",
        "/esmfold.html": SRC / "web_ui" / "esmfold.html",
    }


def start_server(test_case, handler, viewer=None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.viewer = viewer or SimpleNamespace(communicator=SimpleNamespace(handle_action=lambda data: None))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
        server.shutdown()
        server.server_close()
        thread.join()

    test_case.addCleanup(stop)
    return server


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response, response.read()
    finally:
        connection.close()


class ServedPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_pages_come_in_the_viewers_language_with_a_policy_for_what_is_served(self):
        server = start_server(self, AgentPageHandler)
        english = {path: request(server, "GET", path)[1] for path in AgentPageHandler.files}
        for path, file in AgentPageHandler.files.items():
            self.assertEqual(english[path], file.read_bytes(), path)
        pseudo_language(self, self.app)
        for path, file in AgentPageHandler.files.items():
            response, body = request(server, "GET", path)
            with self.subTest(path=path):
                if page_context(file) is None:
                    self.assertEqual(body, file.read_bytes())
                    continue
                self.assertNotEqual(body, file.read_bytes())
                shown = body.decode("utf-8")
                self.assertIn(pseudo_translate("Clear Chat") if path == "/agent.html"
                              else pseudo_translate("Remove attachment"), shown)
                if path == "/agent.html":
                    self.assertEqual(response.getheader("Content-Security-Policy"),
                                     content_security_policy("agent.html", body))

    def test_a_page_that_cannot_be_translated_comes_in_english(self):
        page = SRC / "web_ui" / "agent.html"
        with mock.patch("web_ui.Web_Server.translated_file", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "x")), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(page_in_viewer_language(str(page), b"body"), b"body")
        self.assertIn("Could not show agent.html in the Viewer's language", printed.getvalue())

    def test_the_image_check_answers_in_the_viewers_language(self):
        viewer = SimpleNamespace(communicator=SimpleNamespace(handle_action=lambda data: None))
        server = start_server(self, AgentPageHandler, viewer)
        pseudo_language(self, self.app)
        headers = {"Host": f"127.0.0.1:{server.server_port}", "Content-Type": "application/octet-stream"}
        _, body = request(server, "POST", "/api/agent/image", b"not an image", headers)
        self.assert_translated(json.loads(body)["error"])
        _, body = request(server, "POST", "/api/agent/image", b"", headers)
        self.assert_translated(json.loads(body)["error"])

    def assert_translated(self, text, *values):
        for value in values:
            text = text.replace(str(value), "")
        self.assertEqual(re.findall("[A-Za-z]", text), [], text)


class RecordingPage(QWebEnginePage):
    """A page that keeps the text of each alert() and confirm(), and answers confirm() with answer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dialogs = []
        self.answer = False

    def javaScriptAlert(self, origin, message):
        self.dialogs.append(message)

    def javaScriptConfirm(self, origin, message):
        self.dialogs.append(message)
        return self.answer


# The texts every element shows, its title, placeholder, screen-reader
# name and image description, and the page's title.
SHOWN_TEXTS = """(() => {
    const shown = [document.title];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (!node.parentElement.closest('script, style')) shown.push(node.textContent);
    }
    for (const element of document.querySelectorAll('[title], [placeholder], [aria-label], [alt]')) {
        for (const name of ['title', 'placeholder', 'aria-label', 'alt']) {
            if (element.hasAttribute(name)) shown.push(element.getAttribute(name));
        }
    }
    return JSON.stringify(shown.filter(text => text.trim()));
})()"""

# Card names and every value the conversation fills in hold no letter, so
# whatever letter shows came from the page. The page shows data of its own
# too: the JSON the Custom Options field shows as its example, and the
# names of the cards it starts with or adds, which it saves in
# model_card.json and so leaves English.
PAGE_DATA = ('{\n  "reasoning_effort": "low"\n}', "Ollama (Local)", "LM Studio (Local)", "Llama.cpp (Local)",
             "New Model")
CARDS = {"cards": [
    {"id": "first", "name": "1", "url": "http://localhost:1/v1", "model": "2", "api_key": "", "temperature": 0},
    {"id": "second", "name": "", "url": "http://localhost:3/v1", "model": "4", "api_key": "", "temperature": 0},
]}


class AgentPageTests(unittest.TestCase):
    """The Agent page in Qt WebEngine under the pseudo-language, in each state it can show."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.view = QWebEngineView()
        cls.page = RecordingPage(cls.view)
        cls.view.setPage(cls.page)
        script = QWebEngineScript()
        script.setInjectionPoint(QWebEngineScript.DocumentCreation)
        script.setWorldId(QWebEngineScript.MainWorld)
        script.setSourceCode(
            "window.EventSource = class { close() {} };"
            "Object.defineProperty(navigator, 'clipboard', {value: {writeText: () => Promise.resolve()}});"
        )
        cls.page.scripts().insert(script)
        cls.view.resize(1200, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()

    def setUp(self):
        self.actions = []

        def handle(data):
            if data.get("action") == "fail":
                raise RuntimeError("The test asked for a failed action.")
            self.actions.append(data)

        self.server = start_server(self, AgentPageHandler, SimpleNamespace(communicator=SimpleNamespace(
            handle_action=handle)))
        pseudo_language(self, self.app)
        self.page.dialogs.clear()
        self.page.answer = False
        self.shown = []

    def js(self, source):
        result = []
        self.page.runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + 5
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(result, source)
        return result[0]

    def wait_for(self, source):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.js(source):
                return
            self.app.processEvents()
            time.sleep(0.02)
        self.fail("Timed out: " + source)

    def open(self, cards=CARDS, status=200):
        AgentPageHandler.body = (status, cards if isinstance(cards, bytes) else json.dumps(cards).encode())
        loaded = []
        callback = loaded.append
        self.view.loadFinished.connect(callback)
        self.view.load(QUrl(f"http://127.0.0.1:{self.server.server_port}/agent.html"))
        deadline = time.monotonic() + 10
        while not loaded and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.view.loadFinished.disconnect(callback)
        self.assertEqual(loaded, [True])
        self.wait_for("document.getElementById('capture-viewer-btn').onclick !== null")

    def collect(self):
        self.shown += json.loads(self.js(SHOWN_TEXTS))

    def run_and_collect(self, *sources):
        for source in sources:
            self.js(source)
            self.collect()

    def assert_translated(self, *values):
        self.assertTrue(self.shown)
        values = sorted({str(value) for value in values + PAGE_DATA}, key=len, reverse=True)
        for text in self.shown:
            left = LINK.sub("", text)
            for value in values:
                left = left.replace(value, "")
            self.assertEqual(re.findall("[A-Za-z]", left), [], text)

    def test_the_page_as_it_opens_and_connects(self):
        self.open()
        self.assertEqual(self.js("document.documentElement.lang"), "en")
        self.collect()
        self.run_and_collect(
            "handleServerEvent({type: 'init', data: {llm_loaded: false, llm_history: []}})",
            "handleServerEvent({type: 'init', data: {llm_loaded: true, llm_model_name: '5', llm_history: []}})",
            "eventSource.onopen()",
            "eventSource.onerror()",
            "toggleTheme()",
            "toggleTheme()",
        )
        self.assertEqual(self.js("t('{a} {{b}} {c}', {a: 1})"), "1 {b} {c}")
        self.assert_translated()

    def test_the_model_cards(self):
        self.open()
        self.run_and_collect(
            "document.getElementById('models-panel-btn').click()",
            "document.querySelectorAll('.model-card-header').forEach(header => header.click())",
            "document.querySelector('.show-key-btn').click()",
            "showCardsSaved()",
            "modelCards[0].name = '6'; activateCard(modelCards[0])",
            "modelCards[1].temperature = 1; activateCard(modelCards[1])",
            "addCard()",
            "removeCard(2); removeCard(1); removeCard(0)",
        )
        self.assertEqual(self.js("modelCards.length"), 1)
        self.assert_translated()

    def test_a_card_file_the_page_cannot_use(self):
        for body, status in ((b"{", 200), (b"[]", 200), (b"{}", 500)):
            with self.subTest(body=body, status=status):
                self.open(body, status)
                self.run_and_collect("handleServerEvent({type: 'init', data: {llm_loaded: false, llm_history: []}})")
                self.assertEqual(self.js("document.querySelectorAll('#chat-log .msg-error').length"), 1)
        self.assert_translated("500")

    def test_a_conversation(self):
        self.open()
        self.run_and_collect(
            "handleServerEvent({type: 'init', data: {llm_loaded: true, llm_model_name: '5', llm_history: []}})",
            "updateLLMState({llm_loaded: false}); updateLLMState({llm_loaded: true, llm_model_name: '7'})",
            "setAgentThinking('8')",
            "setAgentThinking('8', true)",
            "appendUserMsg('9', [{data_url: 'data:image/png;base64,AAAA'}])",
            "appendAgentMsg('10', '11', {total: 12, prompt: 13, completion: 14, requests: 15}, '16', '17')",
            "document.querySelector('.thought-box-header').click()",
            "document.querySelector('.copy-btn').click()",
            "appendErrorMsg('18')",
        )
        self.wait_for("document.querySelector('.copy-btn').textContent !== " + json.dumps(pseudo_translate("Copy")))
        self.collect()
        statuses = ("queued", "running", "succeeded", "failed", "cancelled", "skipped", "awaiting_user_input")
        for status in statuses:
            self.run_and_collect(f"activeCommandRequest = 'request'; showCommandStatus('{status}', ["
                                 "{approval: 'pending', status: 'awaiting_user_input', command: '19', command_id: 'c'}])")
        self.assertEqual(self.js("document.querySelectorAll('.command-approval button').length"), 2)
        self.run_and_collect("sendAction({action: 'fail'})")
        self.wait_for("document.querySelectorAll('#chat-log .msg-error').length === 2")
        self.collect()
        self.page.answer = False
        self.js("document.getElementById('clear-chat-btn').click()")
        self.page.answer = True
        self.run_and_collect("document.getElementById('clear-chat-btn').click()")
        self.assertEqual(len(self.page.dialogs), 2)
        self.shown += self.page.dialogs
        self.assert_translated("500")

    def test_attachments(self):
        self.open()
        self.js("handleServerEvent({type: 'init', data: {llm_loaded: true, llm_model_name: '5', llm_history: []}})")
        self.run_and_collect(
            "reserveAttachment('20')",
            "addImageFiles([new File(['21'], '', {type: 'text/plain'})])",
        )
        self.wait_for("document.getElementById('attachment-notice').textContent !== ''")
        self.collect()
        self.run_and_collect(
            "captureViewerAttachment()",
            "finishCapture({capture_token: Array.from(captureRequests.keys())[0], error: '22'})",
            "for (let i = 0; i < 11; i++) reserveAttachment(String(i))",
            "pendingSubmission = {submission_id: 'one', query: '', attachments: []};"
            "submissionTransportFailed(pendingSubmission)",
        )
        self.assert_translated()

    def test_the_esmfold_page_stays_english(self):
        # Mol* has no translations, so its page isn't served in another language.
        response, body = request(self.server, "GET", "/esmfold.html")
        self.assertEqual(body, (SRC / "web_ui" / "esmfold.html").read_bytes())
        self.assertNotIn("web_ui/esmfold.html", PAGE_CONTEXTS)


class AgentMessageTests(unittest.TestCase):
    """What the Viewer sends the Agent page, under the pseudo-language: its own text translated."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        pseudo_language(self, self.app)

    def assert_translated(self, text, *values):
        self.assertTrue(text)
        left = str(text)
        for value in sorted({str(value) for value in values}, key=len, reverse=True):
            left = left.replace(value, "")
        self.assertEqual(re.findall("[A-Za-z]", left), [], text)

    def test_the_image_checks(self):
        from web_ui.agent_images import inspect_source, validate_attachments

        ico = io.BytesIO()
        from PIL import Image
        Image.new("RGB", (32, 32)).save(ico, format="ICO", sizes=[(16, 16), (32, 32)])
        sources = [
            b"", b"<html/>", b'<svg xmlns="http://www.w3.org/2000/svg"><animate/></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg"><style>a { animation: x; }</style></svg>',
            b'<svg xmlns="http://www.w3.org/2000/svg" onload="x"/>', b"<svg", image_bytes("GIF", animated=True),
            ico.getvalue(), b"garbage",
        ]
        for raw in sources:
            with self.subTest(raw=raw[:20]), self.assertRaises(ValueError) as caught:
                inspect_source(raw)
            self.assert_translated(display_text(caught.exception))
        large = "data:image/png;base64," + __import__("base64").b64encode(image_bytes(size=(1700, 10))).decode()
        for attachments in ([{}] * 11, ["x"], [{"data_url": "data:text/plain,x"}], [{"data_url": large}]):
            with self.subTest(attachments=str(attachments)[:40]), self.assertRaises(ValueError) as caught:
                validate_attachments(attachments)
            self.assert_translated(display_text(caught.exception))

    def test_the_agent_turns_errors(self):
        from web_ui import agent_backend as agent

        events = []
        viewer = SimpleNamespace(broadcast_event=events.append, llm_loaded=False)
        agent.run_web_agent_query(viewer, "1", submission_id=2)
        agent.run_web_agent_query(viewer, "1", submission_id="a")
        viewer.llm_loaded = True
        viewer._agent_busy = True
        agent.run_web_agent_query(viewer, "1", submission_id="b")
        viewer._agent_busy = False
        agent.run_web_agent_query(viewer, " ", submission_id="c")
        agent.run_web_agent_query(viewer, "1", attachments="x", submission_id="d")
        with mock.patch.object(agent.os.path, "exists", return_value=False):
            agent.run_web_agent_query(viewer, "1", submission_id="e")
        with mock.patch("builtins.open", side_effect=OSError("3")):
            agent.run_web_agent_query(viewer, "1", submission_id="f")
        agent.on_web_worker_finished(viewer, "1", "", "", "", Message("No response received from LLM."))
        self.assertEqual([event["type"] for event in events], ["agent_error"] * 8)
        prompt = os.path.join(agent._SRC_DIR, "resources", "agent", "system_prompt.md")
        for event in events:
            self.assert_translated(event["error"], prompt, "3")

    def test_the_model_cards_and_commands_errors(self):
        from web_ui import agent_backend as agent

        events = []
        viewer = SimpleNamespace(broadcast_event=events.append, llm_loaded=False)
        cause = agent.ModelCardsError(Message("4"))
        with contextlib.redirect_stdout(io.StringIO()):
            agent.handle_set_backend(viewer, {"card": {}})
            with mock.patch.object(agent, "load_model_cards", side_effect=cause):
                agent.handle_set_backend(viewer, {"card_id": "x"})
            with mock.patch.object(agent, "load_model_cards", return_value=[]):
                agent.handle_set_backend(viewer, {"card_id": "x"})
            agent.handle_save_model_cards(viewer, {"cards": [], "save_id": "5"})
            with mock.patch.object(agent, "_read_model_card_document", side_effect=cause):
                agent.handle_check_model_cards(viewer, {})
        portal = SimpleNamespace(decide=mock.Mock(side_effect=ValueError(Message(
            "That command is not waiting to be run or discarded."))))
        with mock.patch("Viewer_Command_Portal.get_portal", return_value=portal):
            agent.handle_decide_agent_command(viewer, {})
        self.assertEqual([event["type"] for event in events], ["backend_state"] * 3 + ["model_cards_error"] * 2
                         + ["agent_command_error"])
        for event in events:
            self.assert_translated(event["error"])

    def test_the_viewer_capture_errors(self):
        from Viewer_Visual_State import capture_view

        def failing(message):
            def render():
                raise RuntimeError(message)
            return SimpleNamespace(render=render)

        portal = SimpleNamespace(get=lambda request_id: {"status": "running"})
        cases = [
            (SimpleNamespace(canvas=None), None, {}),
            (SimpleNamespace(canvas=failing("6")), "x", {}),
            (SimpleNamespace(canvas=failing("6")), None, {"QT_QPA_PLATFORM": "windows"}),
            (SimpleNamespace(canvas=failing("OpenGL context 6")), None, {"QT_QPA_PLATFORM": "offscreen"}),
        ]
        for viewer, request_id, environment in cases:
            with self.subTest(environment=environment, request_id=request_id), \
                    mock.patch.dict(os.environ, environment), \
                    mock.patch("Viewer_Command_Portal.get_portal", return_value=portal), \
                    self.assertRaises(ValueError) as caught:
                capture_view(viewer, request_id)
            self.assertTrue(str(caught.exception).startswith(("Viewer canvas", "Associated command")))
            self.assert_translated(display_text(caught.exception), "OpenGL context 6", "6")

    def test_the_command_portal_errors_an_agent_reply_can_meet(self):
        from Viewer_Command_Portal import ViewerCommandPortal as Portal

        # A stand-in with the portal's state: a real portal tees the process's output.
        portal = SimpleNamespace(requests={}, submissions={}, queue=[], _closed=False)
        portal._resolve_request = lambda request_id=None, submission_id=None: Portal._resolve_request(
            portal, request_id, submission_id)
        for commands in ([], ["x" * 8193]):
            with self.subTest(commands=len(commands)), self.assertRaises(ValueError) as caught:
                Portal.submit(portal, "a", commands)
            self.assert_translated(display_text(caught.exception), "1..100", "8192")
        portal.queue = [None] * 100
        with self.assertRaises(ValueError) as caught:
            Portal.submit(portal, "b", ["zoom 1"])
        self.assertIn("submission_id", str(caught.exception))
        self.assert_translated(display_text(caught.exception), "submission_id")
        portal.requests = {"r": {"commands": []}}
        for request_id in ("r", "missing"):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError) as caught:
                Portal.decide(portal, request_id, "c", True)
            self.assert_translated(display_text(caught.exception))
        portal._closed = True
        with self.assertRaises(ValueError) as caught:
            Portal.submit(portal, "c", ["zoom 1"])
        self.assert_translated(display_text(caught.exception))

    def test_the_agent_worker_hands_on_its_error(self):
        from web_ui import agent_backend as agent

        worker = agent.AgentWorker("server", "http://localhost:1", "1", "", "2", [], 0, None)
        emitted = []
        worker.finished.connect(lambda *values: emitted.append(values))
        failure = ValueError(Message(
            "Model request failed (HTTP {code}): {response}. Image messages require a vision-capable "
            "model/provider.", code=500, response="8",
        ))
        with mock.patch.object(agent, "call_api", side_effect=failure):
            worker.run()
        with mock.patch.object(agent, "call_api", return_value=None):
            worker.run()
        self.assertEqual(len(emitted), 2)
        self.assertIs(emitted[0][3], failure)
        for *_, error in emitted:
            self.assert_translated(display_text(error), "500", "8")

    def test_a_model_request_failure_and_the_analysis_step(self):
        import urllib.error

        from web_ui import agent_backend as agent

        failure = urllib.error.HTTPError("http://localhost:1", 500, "7", {}, io.BytesIO(b"8"))
        with mock.patch.object(agent.urllib.request, "urlopen", side_effect=failure), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught:
            agent.call_api("http://localhost:1", "9", "", "10")
        self.assertIn("Model request failed (HTTP 500): 8.", str(caught.exception))
        self.assert_translated(display_text(caught.exception), "500", "8")

        events = []
        viewer = SimpleNamespace(broadcast_event=events.append, llm_model_name="11", llm_backend="server")
        with mock.patch.object(agent, "RefinementWorker") as worker:
            agent.start_refinement_worker(viewer, "1", "2", "3", "4", "{}", "")
        worker.return_value.start.assert_called_once()
        self.assertEqual(events, [{"type": "agent_thinking", "model_name": "11", "analyzing": True}])


if __name__ == "__main__":
    unittest.main()
