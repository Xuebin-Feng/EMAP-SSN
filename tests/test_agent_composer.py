"""Exercise the real composer in Qt WebEngine against a local HTTP fixture.

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

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtWebEngineCore import QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView
from http.server import ThreadingHTTPServer
from web_ui.Web_Server import WebServerHandler
from tests.test_agent_images import image_bytes

ROOT = Path(__file__).resolve().parents[1]


class FixtureHandler(WebServerHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/agent_resource/model_card.json':
            self._send_json(200, {'cards': [{'id': 'test', 'name': 'Test vision model', 'url': 'http://test', 'model': 'test'}]})
            return
        if path == '/agent':
            file = ROOT / 'src/web_ui/agent.html'
        elif path.startswith('/agent_resource/'):
            file = ROOT / 'src/resources/agent' / path.rsplit('/', 1)[-1]
        else:
            self.send_error(404)
            return
        data = file.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html' if file.suffix == '.html' else 'text/javascript')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


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
        self.js("document.getElementById('clear-chat-btn').click()")
        self.assertEqual(self.js('pendingAttachments.length'), 0)
        self.assertEqual(self.js("document.querySelectorAll('#chat-log img').length"), 0)

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


if __name__ == '__main__':
    unittest.main()
