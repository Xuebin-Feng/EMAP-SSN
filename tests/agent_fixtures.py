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


class Stalled:
    """A reply that sends its headers and the start of its body, then waits for FakeProvider.release."""

    def __init__(self, start=b'{"choices": '):
        self.start = start


class FakeProvider:
    """A model provider on this computer, so no test sends a request anywhere else.

    respond(payload) answers each POST to /v1/chat/completions with a (status,
    body) pair, body being a dict (sent as JSON), text, bytes or Stalled; the
    default answers completion(). tls, a server ssl.SSLContext, makes it an HTTPS
    server (see self_signed_tls). requests lists what arrived, as dicts of
    "path", "headers" and "payload". hold(), called by respond, makes a reply
    wait until release is set (leaving the with block sets it); arrived is set
    when a request has come in. Use it as a context manager; url is the base URL
    a model card holds. Any fake key a test sends is made up for it.
    """

    def __init__(self, respond=None, tls=None):
        self.respond = respond or (lambda payload: (200, completion()))
        self.scheme = 'https' if tls else 'http'
        self.requests = []
        self.arrived = threading.Event()
        self.release = threading.Event()
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
                provider.arrived.set()
                status, body = provider.respond(payload)
                stalled = body if isinstance(body, Stalled) else None
                if isinstance(body, dict):
                    body = json.dumps(body)
                data = stalled.start if stalled else body.encode() if isinstance(body, str) else body
                try:
                    self.send_response(status)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data) + (1000 if stalled else 0)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    if stalled:
                        provider.release.wait(30)
                except OSError:
                    pass  # The client gave up (a stopped turn); nothing is left to tell it.

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        if tls:
            self.server.socket = tls.wrap_socket(self.server.socket, server_side=True)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f'{self.scheme}://127.0.0.1:{self.server.server_port}/v1'

    def hold(self):
        """Called by respond: wait, up to 30 s, until release is set."""
        self.release.wait(30)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exception):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()


def self_signed_tls(folder):
    """(server context, client context) for 127.0.0.1 with a certificate made here for the test, or None
    where the cryptography package is not installed. The client context trusts only that certificate."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None
    import datetime
    import ipaddress
    import os
    import ssl

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(minutes=5))
                   .not_valid_after(now + datetime.timedelta(days=1))
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), False)
                   .sign(key, hashes.SHA256()))
    cert_file, key_file = os.path.join(folder, 'cert.pem'), os.path.join(folder, 'key.pem')
    with open(cert_file, 'wb') as handle:
        handle.write(certificate.public_bytes(serialization.Encoding.PEM))
    with open(key_file, 'wb') as handle:
        handle.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(cert_file, key_file)
    return server, ssl.create_default_context(cafile=cert_file)
