"""Same-origin enforcement for the Viewer web server's unauthenticated routes.

A browser sends a cross-origin POST whose body is ``text/plain`` without
asking the server first, so before this guard any page the user visited
could drive ``/api/action`` blind, and a rebound DNS name could read
``/api/events``. The header sets below are the ones Chromium and urllib
were observed to send, so a change that breaks a real caller fails here.

The guard covers every route, so its tests live here rather than in
tests/test_web_server.py or the inspection-route tests built on
tests/viewer_fixtures.SnapshotHTTPFixture.
"""
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from desktop.Viewer_Inspection import ViewerInspectionService

ACTION = json.dumps({'action': 'save_molstar_session', 'session': {'planted': True}}).encode()


def page_headers(port, **extra):
    """What Chromium sends for a same-origin fetch from a bundled page."""
    return {'Host': f'127.0.0.1:{port}', 'Origin': f'http://127.0.0.1:{port}',
            'Sec-Fetch-Site': 'same-origin', 'Sec-Fetch-Mode': 'cors',
            'Content-Type': 'application/json', **extra}


def cross_site_headers(port, **extra):
    """What a page on another site sends; no preflight, so this really arrives."""
    return {'Host': f'127.0.0.1:{port}', 'Origin': 'https://evil.example',
            'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'no-cors',
            'Content-Type': 'text/plain;charset=UTF-8', **extra}


class WebOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from web_ui import Web_Server
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = mock.patch.dict(os.environ, {'SSN_VIEWER_SESSION_DIR': self.directory.name})
        env.start()
        self.addCleanup(env.stop)
        self.viewer = SimpleNamespace(
            n_nodes=2, full_headers=['a', 'b'],
            metadata={'Org': {'type': 'text', 'values': np.array(['x', 'y'], dtype=object)}},
            edges=[], visible_mask=[True, False], selected_indices=[],
            group_labels=[['g'], []], cluster_labels=None, web_plugin_registry=None)
        self.viewer.viewer_inspection = ViewerInspectionService(self.viewer)
        self.dispatched = []
        self.viewer.handle_web_action = self.dispatched.append
        self.viewer.communicator = Web_Server.QtCommunicator(self.viewer)
        self.viewer.get_initial_web_state = lambda: {'rows': [{'Node ID': 'a', 'Org': 'x'}]}
        self.server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)
        self.port = self.server.server_address[1]

    def exchange(self, method, path, headers, body=b'', pause=0.0):
        """Send headers, then the body after a pause; return (status, headers, payload)."""
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in headers.items():
                connection.putheader(name, value)
            if body:
                connection.putheader('Content-Length', str(len(body)))
            connection.endheaders()
            time.sleep(pause)
            if body:
                connection.send(body)
            time.sleep(pause)
            response = connection.getresponse()
            # An unframed reply is a stream that never ends (the SSE route, and
            # any file served without a length), so take only its first line.
            raw = (response.read() if response.headers.get('Content-Length')
                   else response.readline())
            try:
                return response.status, response.headers, json.loads(raw)
            except ValueError:
                return response.status, response.headers, raw
        finally:
            connection.close()

    def settle(self):
        """Let the queued Qt dispatch run; the HTTP reply does not wait for it."""
        for _ in range(20):
            self.app.processEvents()

    def test_cross_site_requests_never_reach_the_viewer(self):
        # Both flavours reproduce an attack that worked before this guard: a
        # no-cors fetch, and a text/plain form post, which arrive without any
        # preflight the server could have refused.
        form = dict(cross_site_headers(self.port), **{
            'Sec-Fetch-Mode': 'navigate', 'Sec-Fetch-Dest': 'iframe', 'Content-Type': 'text/plain'})
        cases = [('/api/action', cross_site_headers(self.port), ACTION),
                 ('/api/action', form, b'{"action":"clear_history","x":"="}'),
                 ('/api/agent/image', cross_site_headers(self.port), b'\x89PNG\r\n\x1a\n')]
        for path, headers, body in cases:
            with self.subTest(path=path, mode=headers['Sec-Fetch-Mode']):
                status, _, payload = self.exchange('POST', path, headers, body)
                self.assertEqual((status, payload), (403, {'error': 'Cross-origin requests are not accepted.'}))
        # A same-site page on another port is still a different origin.
        sibling = dict(cross_site_headers(self.port),
                       **{'Origin': f'http://127.0.0.1:{self.port + 1}', 'Sec-Fetch-Site': 'same-site'})
        self.assertEqual(self.exchange('POST', '/api/action', sibling, ACTION)[0], 403)
        self.settle()
        self.assertEqual(self.dispatched, [])

    def test_cross_site_embeds_carry_no_origin_to_check(self):
        # A cross-site <img>, <script> or <iframe> is a plain GET: the browser
        # sends Sec-Fetch-Site but no Origin, so only that header distinguishes
        # it from a bundled page loading its own resources.
        denied = {'error': 'Cross-site requests are not accepted.'}
        embeds = [('/structures/model.pdb', 'image'), ('/agent.html', 'iframe'),
                  ('/agent_resource/model_card.json', 'script')]
        for path, destination in embeds:
            with self.subTest(destination=destination):
                headers = {'Host': f'127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'cross-site',
                           'Sec-Fetch-Mode': 'no-cors', 'Sec-Fetch-Dest': destination}
                self.assertEqual(self.exchange('GET', path, headers)[::2], (403, denied))

    def test_rebound_host_cannot_read_viewer_state(self):
        # DNS rebinding reaches the socket with a foreign name in Host and no
        # Origin, because the page believes it is already same-origin.
        rebound = {'Host': f'rebind.evil.example:{self.port}', 'Sec-Fetch-Site': 'same-origin'}
        denied = {'error': 'Requests must address this Viewer as 127.0.0.1 or localhost.'}
        for path in ('/api/events?client=meta', '/agent.html', '/meta.html'):
            with self.subTest(path=path):
                self.assertEqual(self.exchange('GET', path, rebound)[::2], (403, denied))
        # Token-bearing inspection routes are covered by the same guard.
        token = dict(rebound, **{'Authorization': 'Bearer ' + self.server.inspection_token,
                                 'Content-Type': 'application/json'})
        self.assertEqual(self.exchange('POST', '/api/mcp/v1/data', token, b'{"action":"get_summary"}')[0], 403)

    def test_bundled_pages_and_local_tools_still_work(self):
        # urllib sends no Origin and no Sec-Fetch-*, the way the ESMFold
        # worker's notify_server posts a finished structure.
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}/api/action',
                                         data=ACTION, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual((response.status, json.loads(response.read())), (200, {'status': 'ok'}))
        # The bundled pages, at either spelling of the loopback address.
        for host in (f'127.0.0.1:{self.port}', f'localhost:{self.port}'):
            with self.subTest(host=host):
                headers = page_headers(self.port, Host=host, Origin=f'http://{host}')
                self.assertEqual(self.exchange('POST', '/api/action', headers, ACTION)[::2],
                                 (200, {'status': 'ok'}))
                page = self.exchange('GET', '/agent.html', {'Host': host, 'Sec-Fetch-Site': 'none'})
                self.assertEqual(page[0], 200)
        # The SSE stream a bundled page opens on load.
        stream = self.exchange('GET', '/api/events?client=meta',
                               {'Host': f'127.0.0.1:{self.port}', 'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual(stream[0], 200)
        self.assertIn(b'"Node ID": "a"', stream[2])
        self.settle()
        self.assertEqual(self.dispatched, [{'action': 'save_molstar_session', 'session': {'planted': True}}] * 3)

    def test_real_attachment_and_session_payloads_are_accepted(self):
        # The bundled pages legitimately post far more than the inspection
        # endpoints' 64 KiB: agent.html sends base64 PNG attachments (a Viewer
        # capture is routinely hundreds of KB, up to 10 per message), and
        # esmfold.html re-posts the whole Mol* session after every fold, which
        # grows by roughly 8.5 KB per structure.
        import base64
        import io
        from PIL import Image
        noise = np.random.default_rng(0).integers(0, 256, (300, 300, 3), dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(noise).save(buffer, format='PNG')
        data_url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        self.assertGreater(len(data_url), 65536, 'fixture must exceed the inspection cap')
        query = json.dumps({'action': 'agent_query', 'query': 'Look at this',
                            'submission_id': 'realistic',
                            'attachments': [{'name': 'Viewer capture', 'data_url': data_url}]}).encode()
        session = json.dumps({'action': 'save_molstar_session',
                              'session': {'entries': ['x' * 8500] * 15}}).encode()
        for label, payload in (('agent_query', query), ('save_molstar_session', session)):
            with self.subTest(action=label, bytes=len(payload)):
                reply = self.exchange('POST', '/api/action', page_headers(self.port), payload)
                self.assertEqual(reply[::2], (200, {'status': 'ok'}))
        self.settle()
        self.assertEqual([item['action'] for item in self.dispatched],
                         ['agent_query', 'save_molstar_session'])

    def test_action_cap_covers_what_the_agent_layer_accepts(self):
        # A transport cap below the agent's own limit would reject messages
        # validate_attachments would then have accepted, so the two must move
        # together; this pins them.
        from web_ui.Web_Server import MAX_ACTION_BYTES
        from web_ui.agent_images import MAX_ATTACHMENTS, MAX_IMAGE_BYTES
        per_attachment = MAX_IMAGE_BYTES * 4 // 3 + 100
        self.assertGreater(MAX_ACTION_BYTES, MAX_ATTACHMENTS * per_attachment)

    def test_action_requires_a_json_content_type(self):
        # Setting this header cross-origin forces a preflight the server never
        # answers, so it bars the attack independently of Origin and Sec-Fetch.
        for content_type in ('text/plain;charset=UTF-8', 'application/x-www-form-urlencoded', ''):
            with self.subTest(content_type=content_type):
                headers = page_headers(self.port, **{'Content-Type': content_type})
                status, _, payload = self.exchange('POST', '/api/action', headers, ACTION)
                self.assertEqual((status, payload), (400, {
                    'error': 'Action requests must use Content-Type: application/json'}))
        self.settle()
        self.assertEqual(self.dispatched, [])

    def test_action_body_framing_is_answered_not_crashed(self):
        # These framings all used to reach int(self.headers['Content-Length'])
        # outside the handler's try block. That header's lookup yields None
        # rather than raising, so a missing one crashed the thread with
        # TypeError and the connection dropped without any reply at all.
        from web_ui.Web_Server import MAX_ACTION_BYTES
        size = {'error': f'Action request must be 1..{MAX_ACTION_BYTES} bytes'}
        # An over-cap length is refused on the declared size alone, so no body
        # is ever sent; the reply asks the client to close rather than reading
        # the hundreds of megabytes the header claims are coming.
        cases = [({}, b'', 400, size, None),
                 ({'Content-Length': str(MAX_ACTION_BYTES + 1)}, b'', 400, size, 'close'),
                 ({'Content-Length': 'abc'}, b'', 400,
                  {'error': "invalid literal for int() with base 10: 'abc'"}, 'close'),
                 ({'Transfer-Encoding': 'chunked'}, b'', 400, size, 'close')]
        for framing, body, status, payload, connection in cases:
            with self.subTest(framing=framing):
                headers = {key: value for key, value in page_headers(self.port).items()}
                headers.update(framing)
                reply = self.exchange('POST', '/api/action', headers, body)
                self.assertEqual((reply[0], reply[2], reply[1]['Connection']),
                                 (status, payload, connection))
        self.assertEqual(self.exchange('POST', '/api/action', page_headers(self.port), b'not json')[0], 400)
        self.settle()
        self.assertEqual(self.dispatched, [])

    def test_rejections_drain_the_request_body(self):
        # Replying before reading the body makes Windows reset the connection
        # once the body arrives, and the client loses the reply entirely
        # (ConnectionAbortedError [WinError 10053]). Sending the body after a
        # pause, by which time the reply has gone out, reproduces it every time.
        for size in (1, 65536):
            with self.subTest(size=size, reply='403 cross-site'):
                status, headers, payload = self.exchange(
                    'POST', '/api/action', cross_site_headers(self.port), b'x' * size, pause=0.05)
                self.assertEqual((status, payload['error'], headers['Connection']),
                                 (403, 'Cross-origin requests are not accepted.', None))
            with self.subTest(size=size, reply='400 content type'):
                headers = page_headers(self.port, **{'Content-Type': 'text/plain'})
                reply = self.exchange('POST', '/api/action', headers, b'x' * size, pause=0.05)
                self.assertEqual((reply[0], reply[1]['Connection']), (400, None))
        # The accepted size and the drained size are deliberately different: a
        # body may be hundreds of megabytes, but only a small one is worth
        # reading just to retire it, so a larger rejected body is told to close
        # instead. No body bytes are sent here, so the reply always arrives.
        headers = page_headers(self.port, **{'Content-Type': 'text/plain', 'Content-Length': '70000'})
        reply = self.exchange('POST', '/api/action', headers)
        self.assertEqual((reply[0], reply[1]['Connection']), (400, 'close'))


if __name__ == '__main__':
    unittest.main()
