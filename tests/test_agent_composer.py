"""Exercise the real Agent page (composer and model cards) in Qt WebEngine against a local HTTP fixture.

Run separately from suites that create a QCoreApplication. Set
AGENT_UI_SCREENSHOTS to a directory to save the responsive-layout checks.
"""
import base64
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

# tests/__init__ already selects the offscreen platform; this must be set
# before Qt WebEngine starts Chromium.
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView
from http.server import ThreadingHTTPServer
from web_ui.Web_Server import MIME_TYPES, WebServerHandler
from tests.agent_fixtures import image_bytes

ROOT = Path(__file__).resolve().parents[1]
# The agent resources served by exact name. model_card.json is never among
# them: the real one holds the user's API keys.
AGENT_RESOURCES = {
    path.name for path in (ROOT / 'src/resources/agent').iterdir() if path.name != 'model_card.json'
}


class FixtureHandler(WebServerHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        name = path.rsplit('/', 1)[-1]
        if name == 'model_card.json':
            self._send_json(200, {'cards': [{'id': 'test', 'name': 'Test vision model', 'url': 'http://test', 'model': 'test'}]})
            return
        if path == '/agent':
            file = ROOT / 'src/web_ui/agent.html'
        elif path == '/page_text.js':
            file = ROOT / 'src/web_ui/page_text.js'
        elif path.startswith('/agent_resource/') and name in AGENT_RESOURCES:
            file = ROOT / 'src/resources/agent' / name
        else:
            self.send_error(404)
            return
        # serve_file sends the page's Content-Security-Policy, so every test
        # here also runs under the real policy.
        self.serve_file(str(file), MIME_TYPES[file.suffix])


class ComposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.actions = []
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
        cls.server.viewer = SimpleNamespace(communicator=SimpleNamespace(handle_action=cls.actions.append))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.view = QWebEngineView()
        script = QWebEngineScript()
        script.setInjectionPoint(QWebEngineScript.DocumentCreation)
        script.setWorldId(QWebEngineScript.MainWorld)
        script.setSourceCode('window.EventSource = class { close() {} };')
        cls.view.page().scripts().insert(script)
        cls.view.resize(1000, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def js(self, source):
        result = []
        self.view.page().runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + 5
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.assertTrue(result, source)
        return result[0]

    def wait_for(self, source):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.js(source):
                return
            self.app.processEvents()
            time.sleep(.02)
        detail = self.js("document.getElementById('attachment-notice')?.textContent || ''")
        self.fail('Timed out: ' + source + '; ' + str(detail))

    def setUp(self):
        self.actions.clear()
        self.open_page()

    def open_page(self):
        loaded = []
        callback = lambda ok: loaded.append(ok)
        self.view.loadFinished.connect(callback)
        self.view.load(QUrl(f'http://127.0.0.1:{self.server.server_port}/agent'))
        deadline = time.monotonic() + 10
        while not loaded and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.view.loadFinished.disconnect(callback)
        self.assertEqual(loaded, [True])
        self.wait_for("typeof modelCards !== 'undefined' && modelCards.length === 1 && document.getElementById('capture-viewer-btn').onclick !== null")
        self.js("updateLLMState({llm_loaded:true, llm_model_name:'Test vision model'}, true)")

    def file_expression(self, fmt='PNG', animated=False, size=(20, 10), name='image.png'):
        encoded = base64.b64encode(image_bytes(fmt, animated, size)).decode()
        mime = {'PNG': 'png', 'JPEG': 'jpeg', 'GIF': 'gif', 'WEBP': 'webp', 'BMP': 'bmp', 'AVIF': 'avif'}[fmt]
        return f"new File([Uint8Array.from(atob('{encoded}'), c=>c.charCodeAt(0))], {json.dumps(name)}, {{type:'image/{mime}'}})"

    def test_capture_remove_correlation_and_send(self):
        self.js("document.getElementById('chat-input-field').value='Look at these'; captureViewerAttachment(); captureViewerAttachment()")
        self.wait_for('captureRequests.size === 2')
        encoded = base64.b64encode(image_bytes()).decode()
        self.js(f"handleServerEvent({{type:'agent_capture',capture_token:'another-browser',image_base64:'{encoded}'}})")
        self.assertEqual(self.js("document.querySelectorAll('#attachment-tray img').length"), 0)
        self.js(f"Array.from(captureRequests.keys()).forEach(token => handleServerEvent({{type:'agent_capture',capture_token:token,image_base64:'{encoded}'}}))")
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 0)
        self.assertEqual(self.js("document.querySelectorAll('#attachment-tray img').length"), 2)
        self.js("document.querySelector('#attachment-tray button').click()")
        self.assertEqual(self.js('pendingAttachments.length'), 1)
        self.js('sendAgentQuery(); sendAgentQuery()')
        self.wait_for('pendingSubmission !== null')
        self.js("handleServerEvent({type:'agent_accepted',submission_id:pendingSubmission.submission_id})")
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 1)
        self.assertEqual(self.js('pendingAttachments.length'), 0)
        self.assertTrue(self.js("document.getElementById('chat-input-field').disabled"))
        self.js("handleServerEvent({type:'agent_response',submission_id:pendingSubmission.submission_id,explanation:'Image received'})")
        self.wait_for("!document.getElementById('chat-send-btn').disabled")
        queries = [item for item in self.actions if item['action'] == 'agent_query']
        self.assertEqual(len(queries), 1)
        self.assertEqual(len(queries[0]['attachments']), 1)

    def test_drop_paste_conversion_rejection_and_order(self):
        first = self.file_expression(size=(2000, 1000), name='first.png')
        second = self.file_expression('JPEG', name='second.jpg')
        self.js(f"var dt = new DataTransfer(); dt.items.add({first}); dt.items.add({second}); document.getElementById('chat-composer').dispatchEvent(new DragEvent('drop', {{dataTransfer:dt,bubbles:true,cancelable:true}}))")
        self.wait_for('pendingAttachments.length === 2 && pendingAttachments.every(i=>!i.processing)')
        self.assertEqual(self.js('pendingAttachments.map(i=>i.name).join(",")'), 'first.png,second.jpg')
        self.wait_for("document.querySelector('#attachment-tray img').naturalWidth === 1600")
        self.assertEqual(self.js("document.querySelector('#attachment-tray img').naturalHeight"), 800)
        gif = self.file_expression('GIF', True, name='animation.gif')
        self.js(f"addImageFiles([{gif}, new File(['text'], 'note.txt', {{type:'text/plain'}})])")
        self.wait_for('pendingAttachments.every(i=>!i.processing)')
        self.assertEqual(self.js('pendingAttachments.length'), 2)
        self.assertTrue(self.js("document.getElementById('attachment-notice').textContent.length > 0"))
        self.js(f"var clip = new DataTransfer(); clip.items.add({second}); document.getElementById('chat-input-field').dispatchEvent(new ClipboardEvent('paste', {{clipboardData:clip,bubbles:true,cancelable:true}}))")
        self.wait_for('pendingAttachments.length === 3 && pendingAttachments.every(i=>!i.processing)')
        self.assertTrue(self.js("var plain = new DataTransfer(); plain.setData('text/plain','hello'); document.getElementById('chat-input-field').dispatchEvent(new ClipboardEvent('paste',{clipboardData:plain,bubbles:true,cancelable:true}))"))

    def test_failure_retry_history_and_clear(self):
        self.js(f'addImageFiles([{self.file_expression()}])')
        self.wait_for('pendingAttachments.length === 1 && !pendingAttachments[0].processing')
        self.js('sendAgentQuery()')
        original_id = self.js('pendingSubmission.submission_id')
        self.js('submissionTransportFailed(pendingSubmission); sendAgentQuery()')
        self.assertEqual(self.js('pendingSubmission.submission_id'), original_id)
        self.js("handleServerEvent({type:'agent_accepted',submission_id:pendingSubmission.submission_id}); handleServerEvent({type:'agent_error',submission_id:pendingSubmission.submission_id,error:'Vision unsupported'})")
        self.assertEqual(self.js('pendingAttachments.length'), 1)
        self.assertEqual(self.js("document.querySelectorAll('#chat-log .msg-user').length"), 0)
        self.js("renderHistory([{role:'user',content:'old text'},{role:'user',content:'',attachments:pendingAttachments}])")
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 1)
        self.js("window.confirm = message => { window.clearWarning = message; return false; }; document.getElementById('clear-chat-btn').click()")
        self.assertEqual(self.js('pendingAttachments.length'), 1)
        self.assertIn('will be lost', self.js('window.clearWarning'))
        self.js("window.confirm = () => true; document.getElementById('clear-chat-btn').click()")
        self.assertEqual(self.js('pendingAttachments.length'), 0)
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 0)

    def test_a_turn_dropped_with_its_model_stops_the_spinner_and_returns_the_draft(self):
        # The event is the one agent_backend sends when the model is switched
        # or turned off during a turn, whether from this page or the console.
        self.js("document.getElementById('chat-input-field').value = 'select cluster 1'; sendAgentQuery()")
        self.wait_for('pendingSubmission !== null')
        self.js("handleServerEvent({type:'agent_accepted',submission_id:pendingSubmission.submission_id}); handleServerEvent({type:'agent_thinking',model_name:'Test vision model',submission_id:pendingSubmission.submission_id})")
        self.assertTrue(self.js("document.getElementById('thinking-bubble') !== null"))
        self.assertEqual(self.js("document.getElementById('chat-input-field').value"), '')

        self.js("handleServerEvent({type:'agent_error',submission_id:pendingSubmission.submission_id,error:'The agent turn was stopped because the model was changed or switched off.'}); handleServerEvent({type:'backend_state',llm_loaded:true,llm_model_name:'Other model'})")
        self.assertTrue(self.js("document.getElementById('thinking-bubble') === null"))
        self.assertTrue(self.js('pendingSubmission === null'))
        self.assertEqual(self.js("document.getElementById('chat-input-field').value"), 'select cluster 1')
        self.assertIn('stopped', self.js("document.querySelector('#chat-log .msg-error').textContent"))
        self.assertFalse(self.js("document.getElementById('chat-send-btn').disabled"))

    def test_limits_clear_during_processing_and_layout(self):
        self.js(f'addImageFiles(Array.from({{length:11}},()=>{self.file_expression()}))')
        self.wait_for('pendingAttachments.length === 10 && pendingAttachments.every(i=>!i.processing)')
        self.assertIn('10', self.js("document.getElementById('attachment-notice').textContent"))
        for width in (1000, 390):
            self.view.resize(width, 800)
            self.wait_for(f'window.innerWidth === {width}')
            QTest.qWait(250)  # Allow Chromium's compositor to present the new size.
            self.assertTrue(self.js("document.getElementById('chat-composer').getBoundingClientRect().right <= innerWidth"))
            self.assertTrue(self.js("document.getElementById('chat-send-btn').getBoundingClientRect().right <= innerWidth"))
            self.assertTrue(self.js("document.getElementById('chat-send-btn').getBoundingClientRect().bottom <= innerHeight"))
            folder = os.environ.get('AGENT_UI_SCREENSHOTS')
            if folder:
                Path(folder).mkdir(parents=True, exist_ok=True)
                self.app.processEvents()
                self.view.grab().save(str(Path(folder) / f'agent-composer-{width}.png'))
        self.js('clearAttachmentDraft(); captureViewerAttachment(); clearAttachmentDraft()')
        self.assertEqual(self.js('captureRequests.size'), 0)
        self.assertEqual(self.js('pendingAttachments.length'), 0)

    def test_reconnect_restores_sent_images_without_duplicates(self):
        self.js(f'addImageFiles([{self.file_expression()}])')
        self.wait_for('pendingAttachments.length === 1 && !pendingAttachments[0].processing')
        self.js("sendAgentQuery(); handleServerEvent({type:'agent_accepted',submission_id:pendingSubmission.submission_id})")
        self.js("handleServerEvent({type:'init',data:{llm_loaded:true,llm_model_name:'Test vision model',llm_history:[]}})")
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 1)
        self.js("var completed = [{role:'user',content:'',attachments:pendingSubmission.attachments,submission_id:pendingSubmission.submission_id},{role:'assistant',content:'seen'}]; handleServerEvent({type:'init',data:{llm_loaded:true,llm_model_name:'Test vision model',llm_history:completed}})")
        self.assertTrue(self.js('pendingSubmission === null'))
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 1)
        self.assertFalse(self.js("document.getElementById('chat-send-btn').disabled"))

    def test_buttons_stretch_with_multiline_input(self):
        for width in (1000, 390):
            self.view.resize(width, 800)
            self.wait_for(f'window.innerWidth === {width}')
            self.js("var input = document.getElementById('chat-input-field'); input.value = 'First line\\nSecond line\\nThird line\\nFourth line'; input.dispatchEvent(new Event('input'))")
            self.assertTrue(self.js("['capture-viewer-btn','chat-send-btn'].every(id => Math.abs(document.getElementById(id).getBoundingClientRect().height - document.getElementById('chat-input-field').getBoundingClientRect().height) < 1)"))
            # Capture Viewer: a camera and a one-line label, or where the label
            # would squeeze the input, the camera alone, named by its title.
            capture = json.loads(self.js("""JSON.stringify((() => {
                const button = document.getElementById('capture-viewer-btn');
                const label = button.querySelector('.btn-label');
                return {
                    shown: label.getClientRects().length > 0,
                    lines: label.getBoundingClientRect().height / parseFloat(getComputedStyle(label).fontSize),
                    icon: button.querySelector('use').getAttribute('href'),
                    names: [button.title, button.getAttribute('aria-label')],
                    input: document.getElementById('chat-input-field').getBoundingClientRect().width,
                };
            })())"""))
            self.assertEqual(capture['shown'], width == 1000)
            if capture['shown']:
                self.assertLess(capture['lines'], 1.6)
            self.assertEqual(capture['icon'], '#i-camera')
            self.assertEqual(capture['names'], ['Capture Viewer', 'Capture Viewer'])
            self.assertGreater(capture['input'], 600 if width == 1000 else 180)

    def test_explanation_markup_is_inert(self):
        explanation = '\n\n'.join([
            'Raw <img src="missing.png" onerror="window.pwned = 1"> stays text.',
            '<div onmouseover="window.pwned = 2">Block HTML</div>',
            '[script](javascript:window.pwned=3) [entity](javascript&#58;window.pwned=4) '
            '[case](JaVaScRiPt:window.pwned=5) <javascript:window.pwned=6> '
            '[data](data:text/html,hi) [vb](vbscript:msgbox(1))',
            '![bad image](javascript:window.pwned=7) ![svg image](data:image/svg+xml,%3Csvg/%3E)',
            '[docs](https://example.org/docs) [local](help/page.html)',
            '```python\nprint("<b>")\n```',
            '| a | b |\n|---|:-:|\n| 1 | 2 |',
        ])
        # runJavaScript only converts primitive results, so the report is JSON.
        report = """(() => {
            const elements = Array.from(document.querySelectorAll('#chat-log *'));
            const message = document.querySelector('#chat-log .msg-agent');
            return JSON.stringify({
                handlers: elements.flatMap(el => el.getAttributeNames().filter(name => name.startsWith('on'))),
                schemes: elements.flatMap(el => ['href', 'src'].filter(name => el.hasAttribute(name))
                    .map(name => new URL(el.getAttribute(name), document.baseURI).protocol)),
                links: Array.from(message.querySelectorAll('a')).map(a => a.getAttribute('href')),
                images: message.querySelectorAll('img').length,
                code: message.querySelector('pre code.language-python')?.textContent,
                centered: Array.from(message.querySelectorAll('[align=center]')).map(cell => cell.textContent),
                text: message.textContent,
                pwned: window.pwned ?? null,
            });
        })()"""
        deliveries = {
            'response': "handleServerEvent({type:'agent_response',explanation:%s})",
            'history': "handleServerEvent({type:'init',data:{llm_loaded:true,llm_model_name:'Test vision model',"
                       "llm_history:[{role:'assistant',explanation:%s}]}})",
        }
        for path, event in deliveries.items():
            with self.subTest(path=path):
                self.js("document.getElementById('chat-log').replaceChildren(); window.pwned = undefined")
                self.js(event % json.dumps(explanation))
                QTest.qWait(300)  # A rendered <img onerror> would fire once its load fails.
                result = json.loads(self.js(report))
                self.assertEqual(result['handlers'], [])
                self.assertLessEqual(set(result['schemes']), {'http:', 'https:'})
                self.assertEqual(result['links'], ['https://example.org/docs', 'help/page.html'])
                self.assertEqual(result['images'], 0)
                self.assertIsNone(result['pwned'])
                for shown in ('<img src="missing.png" onerror="window.pwned = 1">',
                              '<div onmouseover="window.pwned = 2">Block HTML</div>',
                              'script entity case javascript:window.pwned=6 data vb',
                              'bad image svg image'):
                    self.assertIn(shown, result['text'])
                self.assertEqual(result['code'], 'print("<b>")\n')
                self.assertEqual(result['centered'], ['b', '2'])

    def test_content_security_policy_blocks_injected_markup(self):
        # Markup that bypassed the renderer still cannot run script: the page's
        # inline <script> is allowed by hash, injected handlers and URLs are not.
        self.js("window.violations = []; document.addEventListener('securitypolicyviolation', e => violations.push(e.effectiveDirective));"
                "document.body.insertAdjacentHTML('beforeend', '<img src=\"missing.png\" onerror=\"window.pwned = 1\"><a id=\"bad\" href=\"javascript:window.pwned = 2\">x</a>');"
                "document.getElementById('bad').click()")
        self.wait_for('violations.length >= 2')
        self.assertFalse(self.js("'pwned' in window"))
        self.assertTrue(self.js("violations.every(directive => directive.startsWith('script-src'))"))

    def test_the_theme_button_switches_the_palette_and_its_icon(self):
        state = """JSON.stringify({
            theme: document.documentElement.dataset.theme,
            label: document.getElementById('theme-toggle-btn').textContent,
            icon: document.querySelector('#theme-toggle-btn use').getAttribute('href'),
            page: getComputedStyle(document.body).backgroundColor,
            card: getComputedStyle(document.querySelector('header')).backgroundColor,
            primary: getComputedStyle(document.documentElement).getPropertyValue('--primary').trim(),
        })"""
        # The page remembers the choice for the next page this view loads.
        self.addCleanup(self.js, "localStorage.removeItem('ssn_theme')")
        self.js("document.documentElement.dataset.theme = 'light'; updateThemeBtn('light')")
        light = json.loads(self.js(state))
        self.js("document.getElementById('theme-toggle-btn').click()")
        dark = json.loads(self.js(state))
        self.assertEqual(light, {'theme': 'light', 'label': 'Dark Mode', 'icon': '#i-moon',
                                 'page': 'rgb(250, 250, 250)', 'card': 'rgb(255, 255, 255)', 'primary': '#18181b'})
        self.assertEqual(dark, {'theme': 'dark', 'label': 'Light Mode', 'icon': '#i-sun',
                                'page': 'rgb(9, 9, 11)', 'card': 'rgb(24, 24, 27)', 'primary': '#fafafa'})
        self.js("document.getElementById('theme-toggle-btn').click()")
        self.assertEqual(json.loads(self.js(state))['theme'], 'light')

    def test_a_saved_theme_is_in_place_before_the_page_can_be_drawn(self):
        # The theme chosen last, here or on the Metadata page, is set when the
        # page's <body> first exists, so a dark page never shows light first.
        probe = QWebEngineScript()
        probe.setInjectionPoint(QWebEngineScript.DocumentCreation)
        probe.setWorldId(QWebEngineScript.MainWorld)
        probe.setSourceCode(
            "new MutationObserver((records, observer) => {"
            "  if (!document.body) return;"
            "  window.themeWhenDrawable = document.documentElement.getAttribute('data-theme');"
            "  observer.disconnect();"
            "}).observe(document, {childList: true, subtree: true});"
        )
        self.view.page().scripts().insert(probe)
        self.addCleanup(self.view.page().scripts().remove, probe)
        self.addCleanup(self.js, "localStorage.removeItem('ssn_theme')")
        for theme, icon in (('dark', '#i-sun'), ('light', '#i-moon')):
            self.js(f"localStorage.setItem('ssn_theme', '{theme}')")
            self.open_page()
            with self.subTest(theme=theme):
                self.assertEqual(self.js('window.themeWhenDrawable'), theme)
                self.assertEqual(self.js("document.querySelector('#theme-toggle-btn use').getAttribute('href')"), icon)

    def test_every_icon_draws_a_symbol_of_the_pages_sprite(self):
        # The page in the states that show icons: the cards panel with a card
        # open and its key shown, a reply with its reasoning and commands, and
        # the next turn in progress.
        self.js("document.getElementById('models-panel-btn').click();"
                "document.querySelector('.model-card-header').click();"
                "document.querySelector('.show-key-btn').click();"
                "appendAgentMsg('Done', 'select 1', null, '', 'Because');"
                "document.querySelector('.thought-box-header').click();"
                "setAgentThinking('Test vision model');"
                "reserveAttachment('shot.png')")
        # runJavaScript only converts primitive results, so the report is JSON.
        report = json.loads(self.js("""JSON.stringify({
            symbols: Array.from(document.querySelectorAll('.icon-sprite symbol'), symbol => '#' + symbol.id),
            icons: Array.from(document.querySelectorAll('svg.icon use'))
                .filter(use => use.closest('svg').getClientRects().length)
                .map(use => [use.getAttribute('href'), use.getBBox().width]),
            keyButton: document.querySelector('.show-key-btn').title,
            removeAttachment: document.querySelector('#attachment-tray button use').getAttribute('href'),
            text: document.body.innerText,
        })"""))
        shown = {href for href, _ in report['icons']}
        self.assertLessEqual({'#i-bot', '#i-moon', '#i-settings', '#i-x', '#i-plus', '#i-save', '#i-send',
                              '#i-menu', '#i-trash-2', '#i-chevron-down', '#i-eye-off', '#i-copy',
                              '#i-loader-circle', '#i-camera'}, shown)
        self.assertEqual(report['removeAttachment'], '#i-x')
        self.assertLessEqual(shown, set(report['symbols']))
        for href, width in report['icons']:
            with self.subTest(icon=href):
                self.assertGreater(width, 0)  # Its symbol was found, and has shapes to draw.
        self.assertEqual(report['keyButton'], 'Hide')
        for glyph in '🤖🌙☀⚙✕×💾✓☰⏳▾▴▶▼':
            self.assertNotIn(glyph, report['text'])


class ModelCardFixtureHandler(FixtureHandler):
    """Serve model_card.json as a test sets it: bytes, an HTTP error status, or None (404)."""
    model_card = None

    def do_GET(self):
        if self.path.split('?')[0].rsplit('/', 1)[-1] != 'model_card.json':
            super().do_GET()
            return
        body = type(self).model_card
        if not isinstance(body, bytes):
            self.send_error(body or 404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ModelCardPageTests(unittest.TestCase):
    """What the page sends the Viewer: it never saves its cards over a model_card.json
    it could not load, activates a model by its saved card's id, and runs or
    discards an agent's command only when the user presses Run or Discard."""
    DEFAULT_NAMES = ['Ollama (Local)', 'LM Studio (Local)', 'Llama.cpp (Local)']
    js = ComposerTests.js
    wait_for = ComposerTests.wait_for

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), ModelCardFixtureHandler)
        cls.server.viewer = SimpleNamespace(communicator=SimpleNamespace(handle_action=lambda data: None))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.view = QWebEngineView()
        # Actions are recorded as the page sends them, so a save made while the
        # page loads is already listed once its controls are set up.
        script = QWebEngineScript()
        script.setInjectionPoint(QWebEngineScript.DocumentCreation)
        script.setWorldId(QWebEngineScript.MainWorld)
        script.setSourceCode(
            'window.EventSource = class { close() {} };'
            'window.sentActions = [];'
            '(() => { const pageFetch = window.fetch; window.fetch = (url, options) => {'
            ' if (url === "/api/action") sentActions.push(JSON.parse(options.body));'
            ' return pageFetch(url, options); }; })();'
        )
        cls.view.page().scripts().insert(script)
        cls.view.resize(1000, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def open_page(self, model_card):
        ModelCardFixtureHandler.model_card = model_card
        loaded = []
        callback = lambda ok: loaded.append(ok)
        self.view.loadFinished.connect(callback)
        self.view.load(QUrl(f'http://127.0.0.1:{self.server.server_port}/agent'))
        deadline = time.monotonic() + 10
        while not loaded and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.view.loadFinished.disconnect(callback)
        self.assertEqual(loaded, [True])
        # The controls are set up once loadModelCards has finished.
        self.wait_for("document.getElementById('capture-viewer-btn').onclick !== null")

    def state(self):
        return json.loads(self.js("""JSON.stringify({
            actions: sentActions,
            cards: modelCards.map(card => card.name),
            errors: Array.from(document.querySelectorAll('#chat-log .msg-error'), e => e.textContent),
            button: document.getElementById('save-cards-btn').textContent,
            buttonIcon: document.querySelector('#save-cards-btn use').getAttribute('href'),
        })"""))

    def test_a_file_that_cannot_be_loaded_is_never_saved_over(self):
        unusable = {
            # Chromium's own message would quote this file, sk-SECRET included.
            'invalid JSON': (b'{"cards": [{"name": "My API", "api_key": sk-SECRET}]}', 'it is not valid JSON'),
            'array root': (b'[{"name": "My API", "api_key": "sk-SECRET"}]', 'must contain a JSON object'),
            'cards not a list': (b'{"cards": {"0": {"name": "My API"}}}', 'must contain a JSON object'),
            'server error': (500, 'the Viewer answered HTTP 500'),
        }
        for label, (model_card, reason) in unusable.items():
            with self.subTest(label):
                self.open_page(model_card)
                self.js("document.getElementById('save-cards-btn').click()")
                clicked = self.state()
                # Connecting renders the history into a cleared chat log.
                self.js("handleServerEvent({type:'init',data:{llm_loaded:false,llm_history:[]}})")
                connected = self.state()

                self.assertEqual(clicked['actions'], [])
                self.assertEqual(clicked['cards'], self.DEFAULT_NAMES)
                self.assertEqual((clicked['button'], clicked['buttonIcon']), ('Save', '#i-save'))
                # Then only the request for the Viewer's own parse error, which
                # gives the position without quoting the file.
                self.assertEqual(connected['actions'], [{'action': 'check_model_cards'}])
                for state in (clicked, connected):
                    self.assertEqual(len(state['errors']), 1)
                    self.assertIn(reason, state['errors'][0])
                    self.assertIn('will not be saved', state['errors'][0])
                    self.assertNotIn('SECRET', state['errors'][0])

    def test_without_saved_cards_the_defaults_are_saved(self):
        for model_card in (None, b'{"cards": []}', b'{}'):
            with self.subTest(model_card=model_card):
                self.open_page(model_card)
                state = self.state()

                [save] = state['actions']
                self.assertEqual(save['action'], 'save_model_cards')
                self.assertEqual([card['name'] for card in save['cards']], self.DEFAULT_NAMES)
                # agent_backend's defaults have these ids, so they activate
                # before this save completes.
                self.assertEqual([card['id'] for card in save['cards']], ['ollama', 'lmstudio', 'llamacpp'])
                self.assertTrue(save['save_id'])
                self.assertEqual(state['errors'], [])

    def test_saved_cards_are_shown_and_a_save_is_confirmed_by_the_viewer(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API', 'url': 'https://api.example/v1'}]}).encode())
        self.assertEqual(self.state()['actions'], [])
        self.assertEqual(self.state()['cards'], ['My API'])

        self.js("setAgentThinking('Test model'); document.getElementById('save-cards-btn').click()")
        [save] = self.state()['actions']
        self.assertEqual([card['name'] for card in save['cards']], ['My API'])
        self.assertEqual([self.state()['button'], self.state()['buttonIcon']], ['Save', '#i-save'])
        self.js("handleServerEvent({type:'model_cards_saved',save_id:'another page'})")
        self.assertEqual([self.state()['button'], self.state()['buttonIcon']], ['Save', '#i-save'])
        self.js(f"handleServerEvent({{type:'model_cards_saved',save_id:{json.dumps(save['save_id'])}}})")
        self.assertEqual([self.state()['button'], self.state()['buttonIcon']], ['Saved!', '#i-check'])
        self.assertTrue(self.js("document.getElementById('save-cards-btn').classList.contains('is-saved')"))

        self.js("handleServerEvent({type:'model_cards_error',save_id:'x',error:'Model cards were not saved: disk full'})")
        self.assertEqual(self.state()['errors'], ['Error: Model cards were not saved: disk full'])
        self.assertTrue(self.js("document.getElementById('thinking-bubble') !== null"))

    def system_messages(self):
        return json.loads(self.js("JSON.stringify(Array.from(document.querySelectorAll('#chat-log .msg-system'), e => e.textContent))"))

    def test_a_model_is_activated_by_its_saved_card_id(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API', 'url': 'https://api.example/v1'}]}).encode())
        toggle = "document.getElementById('agent-toggle-input').click()"
        self.js(toggle)
        self.js(toggle)
        self.assertEqual(self.state()['actions'], [{'action': 'set_backend', 'card_id': 'mine'},
                                                   {'action': 'set_backend', 'card_id': None}])
        self.assertEqual(self.system_messages(), [])

        # The Viewer uses the saved card, so unsaved edits are pointed out.
        self.js("modelCards[0].url = 'http://other.example/v1'")
        self.js(toggle)
        self.assertEqual(self.state()['actions'][-1], {'action': 'set_backend', 'card_id': 'mine'})
        self.assertEqual(len(self.system_messages()), 1)
        self.assertIn('Using the saved settings of "My API"', self.system_messages()[0])
        self.js(toggle)
        self.js("document.getElementById('save-cards-btn').click()")
        save = self.state()['actions'][-1]
        self.js(f"handleServerEvent({{type:'model_cards_saved',save_id:{json.dumps(save['save_id'])}}})")
        self.js(toggle)
        self.assertEqual(len(self.system_messages()), 1)

        # The Viewer refuses a card it has not saved.
        self.js("handleServerEvent({type:'backend_state', llm_loaded:false, error:'This model is not saved yet.'})")
        self.assertEqual(self.state()['errors'], ['Error: This model is not saved yet.'])
        self.assertFalse(self.js("document.getElementById('agent-toggle-input').checked"))

    def test_a_card_saved_without_an_id_is_given_one(self):
        # A hand-written file; the Viewer can activate the card once it is saved.
        self.open_page(json.dumps({'cards': [{'name': 'Hand-written', 'url': 'http://localhost:1234/v1'}]}).encode())
        self.js("document.getElementById('agent-toggle-input').click()")
        [action] = self.state()['actions']
        self.assertEqual(action['action'], 'set_backend')
        self.assertIsInstance(action.get('card_id'), str)
        self.assertEqual(self.js('modelCards[0].id'), action['card_id'])

    def test_a_command_waiting_for_approval_is_run_or_discarded(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API'}]}).encode())
        waiting = {'command_id': 'c1', 'command': 'save <img src=x onerror=alert(1)>.h5',
                   'status': 'awaiting_user_input', 'approval': 'pending'}

        def update(request_id, *commands):
            event = {'type': 'command_request_updated', 'request_id': request_id,
                     'status': 'awaiting_user_input', 'commands': list(commands)}
            self.js(f'handleServerEvent({json.dumps(event)})')

        def approval():
            return json.loads(self.js("""JSON.stringify(Array.from(document.querySelectorAll('.command-approval'), row => ({
                text: row.textContent, images: row.querySelectorAll('img').length,
                buttons: Array.from(row.querySelectorAll('button'), b => [b.textContent, b.disabled])})))"""))

        self.js("handleServerEvent({type:'agent_command_request', request_id:'r1', status:'queued'})")
        update('another request', waiting)
        self.assertEqual(approval(), [])
        done = {'command_id': 'c0', 'command': 'select "a"', 'status': 'succeeded', 'approval': None}
        update('r1', done, waiting)
        [row] = approval()
        # The command is the model's text: shown, never parsed as markup.
        self.assertIn('save <img src=x onerror=alert(1)>.h5', row['text'])
        self.assertEqual(row['images'], 0)
        self.assertEqual(row['buttons'], [['Run', False], ['Discard', False]])

        self.js("document.querySelector('.command-approval .btn-primary').click()")
        self.assertEqual(self.state()['actions'][-1], {'action': 'decide_agent_command', 'request_id': 'r1', 'command_id': 'c1', 'run': True})
        self.assertEqual(approval()[0]['buttons'], [['Run', True], ['Discard', True]])

        update('r1', done, {**waiting, 'command_id': 'c2', 'command': 'print'})
        self.js("document.querySelectorAll('.command-approval button')[1].click()")
        self.assertEqual(self.state()['actions'][-1], {'action': 'decide_agent_command', 'request_id': 'r1', 'command_id': 'c2', 'run': False})

        # A page opened while a command waits shows it from the Viewer's state.
        init = {'type': 'init', 'data': {'llm_loaded': True, 'llm_history': [], 'agent_request_id': 'r1',
                'agent_command_request': {'request_id': 'r1', 'status': 'awaiting_user_input', 'commands': [waiting]}}}
        self.js(f'handleServerEvent({json.dumps(init)})')
        self.assertEqual([button for button, _ in approval()[0]['buttons']], ['Run', 'Discard'])


if __name__ == '__main__':
    unittest.main()
