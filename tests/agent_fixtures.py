"""Fixtures shared by the in-Viewer agent tests (test_agent_images.py, test_agent_composer.py,
test_agent_backend.py and test_agent_prompt.py): images, and a model provider on this computer."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

from PIL import Image


def image_bytes(fmt='PNG', animated=False, size=(20, 10)):
    out = io.BytesIO()
    kwargs = {'save_all': True, 'append_images': [Image.new('RGB', size, 'blue')], 'duration': 100} if animated else {}
    Image.new('RGB', size, 'red').save(out, format=fmt, **kwargs)
    return out.getvalue()


def completion(text='ok', **extra):
    """The body of an OpenAI-style chat completion that answers text."""
    return {'choices': [{'message': {'content': text, **extra}}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}


class FakeProvider:
    """A model provider on this computer, so no test sends a request anywhere else.

    respond(payload) answers each POST to /v1/chat/completions with a (status,
    body) pair, body being a dict (sent as JSON), text or bytes; the default
    answers completion(). requests lists what arrived, as dicts of "path",
    "headers" and "payload". Use it as a context manager; url is the base URL a
    model card holds. Any fake key a test sends is made up for it.
    """

    def __init__(self, respond=None):
        self.respond = respond or (lambda payload: (200, completion()))
        self.requests = []
        self._lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # The body is read before answering: a reply sent to a client
                # that is still sending can be lost on Windows.
                length = int(self.headers.get('Content-Length') or 0)
                payload = json.loads(self.rfile.read(length) or b'{}')
                with provider._lock:
                    provider.requests.append({'path': self.path, 'headers': dict(self.headers), 'payload': payload})
                status, body = provider.respond(payload)
                if isinstance(body, dict):
                    body = json.dumps(body)
                data = body.encode() if isinstance(body, str) else body
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass  # The client gave up (a stopped turn); nothing is left to tell it.

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/v1'

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exception):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()
