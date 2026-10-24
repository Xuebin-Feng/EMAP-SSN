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
        # Purple's colours, the default theme colour's: its accents on the greys every colour shares.
        self.assertEqual(light, {'theme': 'light', 'label': 'Dark Mode', 'icon': '#i-moon',
                                 'page': 'rgb(250, 250, 250)', 'card': 'rgb(255, 255, 255)', 'primary': '#5b5bd6'})
        self.assertEqual(dark, {'theme': 'dark', 'label': 'Light Mode', 'icon': '#i-sun',
                                'page': 'rgb(9, 9, 11)', 'card': 'rgb(24, 24, 27)', 'primary': '#8289ff'})
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

    def card_inputs(self, card_index):
        """The text fields of a card in the panel: name, URL, model, temperature."""
        return f"document.querySelectorAll('.model-card')[{card_index}].querySelectorAll('.card-field input')"

    def type_into(self, input_expression, text):
        self.js(f"(() => {{ const field = {input_expression}; field.value = {json.dumps(text)}; field.dispatchEvent(new Event('input')); }})()")

    def test_a_blank_temperature_is_kept_blank_and_zero_is_kept_zero(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API', 'temperature': 0.3}, {'id': 'two', 'name': 'Second'}]}).encode())
        shown = json.loads(self.js(f"JSON.stringify([0, 1].map(i => {{ const field = {self.card_inputs('i')}[3]; return [field.value, field.placeholder]; }}))"))
        self.assertEqual(shown, [['0.3', 'blank: not sent'], ['', 'blank: not sent']])

        field = f"{self.card_inputs(0)}[3]"
        for text, expected in (('', None), ('0', 0), ('0.7', 0.7), ('', None)):
            with self.subTest(text=text):
                self.type_into(field, text)
                self.assertEqual(json.loads(self.js('JSON.stringify(modelCards[0].temperature)')), expected)
        self.type_into(field, '0.2')
        self.js("document.getElementById('save-cards-btn').click()")
        [save] = [a for a in self.state()['actions'] if a['action'] == 'save_model_cards']
        self.assertEqual([card.get('temperature') for card in save['cards']], [0.2, None])

        self.js("document.getElementById('add-card-btn').click()")
        self.assertEqual(json.loads(self.js('JSON.stringify(modelCards[2].temperature)')), None)
        self.assertEqual(self.js(f"{self.card_inputs(2)}[3].value"), '')

    def test_the_viewer_telling_the_page_about_a_card_is_not_undone_by_the_next_save(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API', 'url': 'https://api.example/v1', 'temperature': 0.0}]}).encode())
        note = "document.querySelector('.model-card .card-note')"
        self.assertTrue(self.js(f'{note}.hidden'))
        update = {'type': 'model_card_updated', 'card_id': 'mine', 'fields': {'omit_sampling': True}}
        self.js(f'handleServerEvent({json.dumps(update)})')
        self.assertFalse(self.js(f'{note}.hidden'))
        self.assertIn('rejected sampling parameters', self.js(f'{note}.textContent'))
        self.js("handleServerEvent({type:'model_card_updated', card_id:'nobody', fields:{omit_sampling:true}})")

        # The change is the saved card's too, so activating it is not "using the saved settings".
        self.js("document.getElementById('agent-toggle-input').click()")
        self.assertEqual(self.system_messages(), [])
        self.js("document.getElementById('save-cards-btn').click()")
        [save] = [a for a in self.state()['actions'] if a['action'] == 'save_model_cards']
        self.assertEqual(save['cards'][0]['omit_sampling'], True)

        # Typing a temperature asks for it to be sent again.
        self.type_into(f"{self.card_inputs(0)}[3]", '0.5')
        self.assertTrue(self.js(f'{note}.hidden'))
        self.assertNotIn('omit_sampling', json.loads(self.js('JSON.stringify(modelCards[0])')))

    CARD = {'cards': [{'id': 'mine', 'name': 'My API'}]}

    @staticmethod
    def command(command_id, text, status, approval=None):
        return {'command_id': command_id, 'command': text, 'status': status, 'approval': approval}

    def request_event(self, request_id, *commands, status='awaiting_user_input', kind='command_request_updated'):
        event = {'type': kind, 'request_id': request_id, 'status': status, 'commands': list(commands)}
        self.js(f'handleServerEvent({json.dumps(event)})')

    def request_cards(self):
        return json.loads(self.js("""JSON.stringify(Array.from(document.querySelectorAll('#chat-log .command-request'), card => ({
            request: card.dataset.requestId, status: card.dataset.status,
            commands: Array.from(card.querySelectorAll('.command-item'), item => [
                item.querySelector('.command-text').textContent, item.dataset.status,
                item.querySelector('.command-status').textContent, item.querySelectorAll('.command-approval').length])})))"""))

    IN_VIEW = """(element => { const log = document.getElementById('chat-log').getBoundingClientRect();
        const box = element.getBoundingClientRect(); return box.top >= log.top - 1 && box.bottom <= log.bottom + 1; })"""

    def fill_the_log(self, count=40):
        self.js(f"for (let i = 0; i < {count}; i++) appendSystemMsg('Earlier message ' + i + ' of the conversation.')")
        self.assertTrue(self.js("document.getElementById('chat-log').scrollHeight > document.getElementById('chat-log').clientHeight"))

    def test_each_request_has_its_own_card_with_a_status_for_every_command(self):
        self.open_page(json.dumps(self.CARD).encode())
        select = self.command('a1', 'select "a"', 'succeeded')
        save = self.command('a2', 'save x.h5', 'awaiting_user_input', 'pending')
        printing = self.command('a3', 'print', 'queued')
        self.request_event('r1', select, save, printing, kind='agent_command_request')

        [card] = self.request_cards()
        self.assertEqual((card['request'], card['status']), ('r1', 'awaiting_user_input'))
        self.assertEqual(card['commands'], [['select "a"', 'succeeded', 'succeeded', 0],
                                            ['save x.h5', 'awaiting_user_input', 'waiting for your approval', 1],
                                            ['print', 'queued', 'queued', 0]])
        # There is no single line of status any more.
        self.assertEqual(self.js("document.querySelectorAll('#command-request-status').length"), 0)
        self.assertEqual(self.js("document.querySelector('.command-request').getAttribute('aria-label')"), 'Viewer commands')

        # An update changes the statuses and keeps the rows, and the buttons, that it leaves as they were.
        self.js("window.firstRow = document.querySelector('.command-item'); window.runButton = document.querySelector('.command-approval button')")
        self.request_event('r1', select, save, {**printing, 'status': 'skipped'})
        self.assertTrue(self.js("document.querySelector('.command-item') === window.firstRow && "
                                "document.querySelector('.command-approval button') === window.runButton"))
        self.assertEqual(self.request_cards()[0]['commands'][2][1:3], ['skipped', 'skipped'])

        self.request_event('r1', select, {**save, 'status': 'running', 'approval': 'approved'}, printing, status='running')
        self.assertEqual(self.request_cards()[0]['commands'][1], ['save x.h5', 'running', 'running', 0])
        self.request_event('r1', select, {**save, 'status': 'failed', 'approval': 'approved'},
                           {**printing, 'status': 'skipped'}, status='failed')
        [card] = self.request_cards()
        self.assertEqual((card['status'], [c[1] for c in card['commands']]), ('failed', ['succeeded', 'failed', 'skipped']))
        self.assertEqual(len(card['commands']), 3)

    def test_a_later_requests_buttons_are_in_view_and_have_the_focus(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.request_event('r1', self.command('a1', 'select "a"', 'succeeded'), status='succeeded', kind='agent_command_request')
        self.fill_the_log()
        self.js("document.getElementById('chat-log').scrollTop = 0")
        self.request_event('r2', self.command('b1', 'save y.h5', 'awaiting_user_input', 'pending'), kind='agent_command_request')

        # Two cards, the second one below the conversation and showing its own buttons.
        self.assertEqual([card['request'] for card in self.request_cards()], ['r1', 'r2'])
        self.assertTrue(self.js(f"{self.IN_VIEW}(document.querySelector('.command-request[data-request-id=r2] .command-approval'))"))
        self.assertEqual(self.js("document.querySelectorAll('.command-request[data-request-id=r1] .command-approval').length"), 0)
        # A later update of the first request, which finished meanwhile, goes to its own card; one of a request
        # this page never showed is not this page's.
        self.request_event('r1', self.command('a1', 'select "a"', 'failed'), status='failed')
        self.request_event('r0', self.command('z1', 'zoom 100', 'running'), status='running')
        self.assertEqual([(card['request'], card['commands'][0][1]) for card in self.request_cards()], [('r1', 'failed'), ('r2', 'awaiting_user_input')])
        # The row has the focus, not Run: a key pressed at that moment approves nothing.
        focused = json.loads(self.js("""JSON.stringify({row: document.activeElement.classList.contains('command-approval'),
            tag: document.activeElement.tagName, label: document.activeElement.getAttribute('aria-label'),
            role: document.activeElement.getAttribute('role'), run: document.activeElement.querySelector('button').textContent})"""))
        self.assertEqual(focused, {'row': True, 'tag': 'DIV', 'label': 'The agent wants to run save y.h5', 'role': 'group', 'run': 'Run'})
        self.assertEqual(self.state()['actions'], [])

    def test_the_buttons_come_into_view_without_taking_the_focus_from_the_cards_panel(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.fill_the_log()
        self.js("document.getElementById('chat-log').scrollTop = 0;"
                "document.getElementById('models-panel-btn').click();"
                "document.querySelector('.model-card-header').click();"
                "document.querySelector('.model-card .card-field input').focus()")
        self.request_event('r1', self.command('a1', 'save y.h5', 'awaiting_user_input', 'pending'), kind='agent_command_request')
        self.assertTrue(self.js(f"{self.IN_VIEW}(document.querySelector('.command-approval'))"))
        self.assertTrue(self.js("document.activeElement.matches('.model-card .card-field input')"))

    def test_a_page_opened_while_a_command_waits_shows_it_in_view_with_the_focus(self):
        self.open_page(json.dumps(self.CARD).encode())
        history = [item for number in range(25) for item in (
            {'role': 'user', 'content': f'question {number}'},
            {'role': 'assistant', 'content': f'answer {number}', 'explanation': f'answer {number}', 'commands': '', 'terminal_output': ''})]
        request = {'request_id': 'r9', 'status': 'awaiting_user_input',
                   'commands': [self.command('c1', 'save z.h5', 'awaiting_user_input', 'pending')]}
        init = {'type': 'init', 'data': {'llm_loaded': True, 'llm_history': history, 'agent_request_id': 'r9', 'agent_command_request': request}}
        self.js(f'handleServerEvent({json.dumps(init)})')
        self.assertTrue(self.js(f"{self.IN_VIEW}(document.querySelector('.command-approval'))"))
        self.assertTrue(self.js("document.activeElement.classList.contains('command-approval')"))
        # The same request, shown again by a later event, keeps its one card and one pair of buttons.
        self.request_event('r9', *request['commands'])
        self.assertEqual((len(self.request_cards()), self.js("document.querySelectorAll('.command-approval').length")), (1, 1))

    def test_a_card_whose_log_was_cleared_is_made_again(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.request_event('r1', self.command('a1', 'select "a"', 'running'), status='running', kind='agent_command_request')
        self.js("document.getElementById('chat-log').innerHTML = ''")
        self.request_event('r1', self.command('a1', 'select "a"', 'succeeded'), status='succeeded')
        self.assertEqual(self.request_cards(), [{'request': 'r1', 'status': 'succeeded',
                                                 'commands': [['select "a"', 'succeeded', 'succeeded', 0]]}])

    def start_turn(self, text='hello'):
        """The page with a model loaded and a message of its own sent, accepted and being thought about."""
        self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[]}})")
        self.js(f"document.getElementById('chat-input-field').value = {json.dumps(text)}; sendAgentQuery()")
        self.wait_for('pendingSubmission !== null')
        self.js("handleServerEvent({type:'agent_accepted', submission_id: pendingSubmission.submission_id, query: pendingSubmission.query, attachments: []});"
                "handleServerEvent({type:'agent_thinking', model_name:'m', submission_id: pendingSubmission.submission_id})")

    STOP = "document.getElementById('chat-stop-btn')"

    def stop_button(self):
        return json.loads(self.js(f"""JSON.stringify({{
            shown: {self.STOP}.getClientRects().length > 0, hidden: {self.STOP}.hidden, disabled: {self.STOP}.disabled,
            label: {self.STOP}.getAttribute('aria-label'), title: {self.STOP}.title, text: {self.STOP}.textContent.trim(),
            icon: {self.STOP}.querySelector('use').getAttribute('href'), role: {self.STOP}.tagName,
            send: document.getElementById('chat-send-btn').disabled, input: document.getElementById('chat-input-field').disabled}})"""))

    def test_stop_shows_during_a_turn_and_asks_the_viewer_to_stop_it(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[]}})")
        idle = self.stop_button()
        self.assertEqual((idle['shown'], idle['hidden']), (False, True))
        self.assertEqual((idle['send'], idle['input']), (False, False))

        self.start_turn()
        running = self.stop_button()
        self.assertEqual((running['shown'], running['disabled'], running['send'], running['input']), (True, False, True, True))
        # A button with a name and a title whatever its width, an icon from the page's sprite, and a label.
        self.assertEqual((running['label'], running['title'], running['text'], running['icon'], running['role']),
                         ('Stop', 'Stop', 'Stop', '#i-x', 'BUTTON'))
        submission = self.js('pendingSubmission.submission_id')

        self.js(f"{self.STOP}.click(); {self.STOP}.click(); stopTurn(); stopTurn()")
        stops = [a for a in self.state()['actions'] if a['action'] == 'cancel_turn']
        # Once, whatever the clicks, naming the turn this page started; and the button waits for the Viewer's answer.
        self.assertEqual(stops, [{'action': 'cancel_turn', 'submission_id': submission}])
        self.assertTrue(self.stop_button()['disabled'])

        self.js(f"handleServerEvent({{type:'agent_stopped', submission_id:{json.dumps(submission)}}})")
        stopped = self.stop_button()
        self.assertEqual((stopped['shown'], stopped['disabled'], stopped['send'], stopped['input']), (False, False, False, False))
        self.assertTrue(self.js("document.getElementById('thinking-bubble') === null"))
        self.assertEqual(self.system_messages()[-1], 'Stopped. Commands that had already run were not undone.')
        # The message goes back to the draft, as for a turn that failed.
        self.assertEqual(self.js("document.getElementById('chat-input-field').value"), 'hello')
        self.assertEqual(self.js("document.querySelectorAll('#chat-log .msg-user').length"), 0)
        self.assertTrue(self.js('pendingSubmission === null'))

    def test_a_turn_started_from_the_terminal_locks_the_composer_and_can_be_stopped(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[]}})")
        self.js("handleServerEvent({type:'agent_thinking', model_name:'m'})")
        running = self.stop_button()
        self.assertEqual((running['shown'], running['send'], running['input']), (True, True, True))
        self.js(f"{self.STOP}.click()")
        self.assertEqual([a for a in self.state()['actions'] if a['action'] == 'cancel_turn'], [{'action': 'cancel_turn'}])
        self.js("handleServerEvent({type:'agent_stopped'})")
        stopped = self.stop_button()
        self.assertEqual((stopped['shown'], stopped['send'], stopped['input']), (False, False, False))
        self.assertTrue(self.js("document.getElementById('thinking-bubble') === null"))

    def test_stop_goes_when_the_turn_ends_any_other_way(self):
        self.open_page(json.dumps(self.CARD).encode())
        for ending in ("{type:'agent_response', explanation:'Done'}", "{type:'agent_error', error:'The server refused.'}",
                       "{type:'backend_state', llm_loaded:false}"):
            with self.subTest(ending=ending):
                self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[]}});"
                        "handleServerEvent({type:'agent_thinking', model_name:'m'})")
                self.assertTrue(self.stop_button()['shown'])
                self.js(f'handleServerEvent({ending})')
                self.assertFalse(self.stop_button()['shown'])

    def test_a_page_opened_during_a_turn_shows_stop(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[], agent_busy:true}})")
        running = self.stop_button()
        self.assertEqual((running['shown'], running['send'], running['input']), (True, True, True))
        self.assertTrue(self.js("document.getElementById('thinking-bubble') !== null"))
        # A reconnect that finds the Viewer idle takes it away.
        self.js("handleServerEvent({type:'init', data:{llm_loaded:true, llm_model_name:'m', llm_history:[], agent_busy:false}})")
        self.assertFalse(self.stop_button()['shown'])

    def test_stop_can_be_pressed_again_when_its_request_did_not_get_through(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.start_turn()
        self.js("window.fetch = () => Promise.resolve({ok: false, status: 500})")
        self.js(f"{self.STOP}.click()")
        self.wait_for(f"!{self.STOP}.disabled")
        self.assertEqual(self.state()['errors'], ['Error: Request failed (HTTP 500).'])
        self.assertTrue(self.stop_button()['shown'])

    def test_stop_fits_the_composer_at_any_width(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.start_turn()
        for width in (1000, 390):
            self.view.resize(width, 800)
            self.wait_for(f'window.innerWidth === {width}')
            geometry = json.loads(self.js(f"""JSON.stringify((() => {{
                const stop = {self.STOP}, label = stop.querySelector('.btn-label');
                return {{right: stop.getBoundingClientRect().right, window: innerWidth, labelShown: label.getClientRects().length > 0,
                         input: document.getElementById('chat-input-field').getBoundingClientRect().width,
                         height: stop.getBoundingClientRect().height, send: document.getElementById('chat-send-btn').getBoundingClientRect().height}};
            }})())"""))
            self.assertLessEqual(geometry['right'], geometry['window'])
            self.assertEqual(geometry['labelShown'], width == 1000)
            self.assertGreater(geometry['input'], 120)
            self.assertAlmostEqual(geometry['height'], geometry['send'], delta=1)
        self.view.resize(1000, 800)

    NOTICE = {'card_id': 'mine', 'name': 'Provider', 'host': 'api.provider.example', 'origin': 'https://api.provider.example',
              'plain_http_key': False, 'ref': {'id': 'mine', 'name': 'Provider', 'url': 'https://api.provider.example/v1'}}

    def show_notice(self, **changes):
        event = {'type': 'backend_state', 'llm_loaded': False, 'data_use': {**self.NOTICE, **changes}}
        self.js(f'handleServerEvent({json.dumps(event)})')

    def notice(self):
        return json.loads(self.js("""JSON.stringify(Array.from(document.querySelectorAll('#data-use-notice'), notice => ({
            role: notice.getAttribute('role'), labelled: document.getElementById(notice.getAttribute('aria-labelledby'))?.textContent,
            title: notice.querySelector('.data-use-title').textContent,
            paragraphs: Array.from(notice.querySelectorAll(':scope > p'), p => p.textContent),
            items: Array.from(notice.querySelectorAll('li'), li => li.textContent),
            warnings: Array.from(notice.querySelectorAll('.is-warning'), w => w.textContent),
            buttons: Array.from(notice.querySelectorAll('button'), b => [b.textContent, b.className.includes('btn-primary')]),
            focused: document.activeElement === notice})))"""))

    def test_a_notice_says_what_a_remote_card_receives_and_continue_or_cancel_answers_it(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'Provider', 'url': 'https://api.provider.example/v1'}]}).encode())
        self.js("handleServerEvent({type:'init', data:{llm_loaded:false, llm_history:[]}})")
        self.show_notice()
        [notice] = self.notice()
        # The wording the user sees before a card first sends anything to a server that is not on this computer.
        self.assertEqual(notice['role'], 'group')
        self.assertEqual(notice['labelled'], 'Send data to api.provider.example?')
        self.assertEqual(notice['paragraphs'], [
            'With every message, the model card "Provider" sends the following to api.provider.example, a server that is not on this computer.',
            "api.provider.example's own terms and privacy policy apply to what it receives. "
            "You are not asked again for this card unless its address changes."])
        self.assertEqual(notice['items'], [
            "the agent's instructions",
            'a summary of the open session: file names, metadata column names, selection counts and the reference header',
            'your messages, and the output of the commands the agent runs',
            'any images you attach'])
        self.assertEqual((notice['warnings'], notice['buttons']), ([], [['Continue', True], ['Cancel', False]]))
        # It is where the user is looking, and has the keyboard, which reaches the buttons next.
        self.assertTrue(notice['focused'])
        self.assertTrue(self.js(f"{self.IN_VIEW}(document.getElementById('data-use-notice'))"))
        self.assertFalse(self.js("document.getElementById('agent-toggle-input').checked"))

        # Another event for the same notice replaces it; Continue accepts, once, for the card named.
        self.show_notice()
        self.assertEqual(len(self.notice()), 1)
        self.js("document.querySelector('#data-use-notice .btn-primary').click()")
        self.assertEqual(self.notice(), [])
        self.assertEqual(self.state()['actions'], [{'action': 'set_backend', 'card_id': 'mine', 'accept_data_use': True}])

        self.show_notice()
        self.js("document.querySelectorAll('#data-use-notice button')[1].click()")
        self.assertEqual(self.notice(), [])
        self.assertEqual(self.state()['actions'][-1], {'action': 'decline_data_use'})

        # The Viewer answering without a notice takes a shown one away.
        self.show_notice()
        self.js("handleServerEvent({type:'backend_state', llm_loaded:true, llm_model_name:'big'})")
        self.assertEqual(self.notice(), [])

    def test_the_notice_of_a_card_whose_key_would_cross_the_network_in_the_clear_says_so(self):
        self.open_page(json.dumps(self.CARD).encode())
        self.show_notice(plain_http_key=True, host='gpu-box.lan:1234', name='LAN box')
        [notice] = self.notice()
        self.assertEqual(notice['warnings'], ['This card would also send its API key over plain http://, which anyone on the '
                                              'network path could read. Use an https:// address.'])
        self.assertEqual(self.notice()[0]['title'], 'Send data to gpu-box.lan:1234?')

    def test_a_page_opened_while_a_notice_waits_shows_it(self):
        self.open_page(json.dumps(self.CARD).encode())
        init = {'type': 'init', 'data': {'llm_loaded': False, 'llm_history': [], 'data_use': self.NOTICE}}
        self.js(f'handleServerEvent({json.dumps(init)})')
        [notice] = self.notice()
        self.assertEqual(notice['title'], 'Send data to api.provider.example?')

    def test_a_card_the_page_knows_by_name_only_is_accepted_by_that_name(self):
        self.open_page(json.dumps({'cards': [{'name': 'Hand-written', 'url': 'https://api.provider.example/v1'}]}).encode())
        self.show_notice(card_id=None, name='Hand-written')
        self.js("document.querySelector('#data-use-notice .btn-primary').click()")
        [action] = self.state()['actions']
        self.assertEqual((action['action'], action['accept_data_use']), ('set_backend', True))
        self.assertEqual(action['card_id'], self.js('modelCards[0].id'))

    def test_a_warning_from_the_viewer_is_shown_when_a_card_is_switched_on(self):
        self.open_page(json.dumps(self.CARD).encode())
        text = "Warning: Model card 'LAN box' sends its API key over plain http:// to gpu-box.lan:1234, which is not on this computer."
        self.js(f"handleServerEvent({{type:'backend_state', llm_loaded:true, llm_model_name:'m', warning:{json.dumps(text)}}})")
        shown = json.loads(self.js("""JSON.stringify(Array.from(document.querySelectorAll('#chat-log .msg-warning'), w => [w.textContent, w.getAttribute('role'),
            w.querySelector('use').getAttribute('href')]))"""))
        self.assertEqual(shown, [[text, 'note', '#i-triangle-alert']])

    def test_the_card_editor_warns_about_a_key_sent_in_the_clear_to_another_computer(self):
        self.open_page(json.dumps({'cards': [{'id': 'mine', 'name': 'My API', 'url': 'http://gpu-box.lan:1234/v1', 'api_key': 'sk-FAKE'}]}).encode())
        warning = "document.querySelectorAll('.model-card .card-note.is-warning')[0]"
        text = ('This card would send its API key over plain http:// to gpu-box.lan:1234, which is not on this computer: '
                'anyone on the network path could read it. Use an https:// address.')
        self.assertFalse(self.js(f'{warning}.hidden'))
        self.assertEqual(self.js(f'{warning}.textContent'), text)
        self.assertEqual(self.js(f"{warning}.getAttribute('role')"), 'note')

        url, key = f"{self.card_inputs(0)}[1]", "document.querySelector('.model-card input[type=password]')"
        for typed_url, typed_key, shown in (
            ('https://gpu-box.lan:1234/v1', 'sk-FAKE', False),      # encrypted
            ('http://localhost:1234/v1', 'sk-FAKE', False),         # this computer
            ('http://127.0.0.1:1234/v1', 'sk-FAKE', False),
            ('http://[::1]:1234/v1', 'sk-FAKE', False),
            ('http://192.168.0.7/v1', '', False),                   # nothing to send
            ('http://192.168.0.7/v1', 'sk-FAKE', True),
            ('http://localhost.example.com/v1', 'sk-FAKE', True),
            ('not a url', 'sk-FAKE', False),
        ):
            with self.subTest(url=typed_url, key=typed_key):
                self.type_into(url, typed_url)
                self.type_into(key, typed_key)
                self.assertEqual(not self.js(f'{warning}.hidden'), shown)
        self.type_into(url, 'http://192.168.0.7/v1')
        self.assertEqual(self.js(f'{warning}.textContent'), text.replace('gpu-box.lan:1234', '192.168.0.7'))

    def test_a_card_without_a_key_or_with_a_local_address_shows_no_warning(self):
        self.open_page(json.dumps({'cards': [{'id': 'a', 'name': 'A', 'url': 'http://localhost:1234/v1', 'api_key': 'sk-FAKE'},
                                             {'id': 'b', 'name': 'B', 'url': 'http://gpu-box.lan/v1'}]}).encode())
        self.assertEqual(self.js("Array.from(document.querySelectorAll('.model-card .card-note.is-warning')).filter(n => !n.hidden).length"), 0)

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
