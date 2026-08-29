# Copyright 2026 Xuebin Feng
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import http.server
import errno
import socket
import threading
import queue
import json
import os
import re
from urllib.parse import parse_qs, urlsplit
from PySide6 import QtCore


LOOPBACK_HOST = "127.0.0.1"

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        return super().default(obj)

class QtCommunicator(QtCore.QObject):
    action_signal = QtCore.Signal(dict)
    
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.action_signal.connect(self.dispatch_action)
        
    def handle_action(self, data):
        # Called from HTTP thread to safely execute on the main thread via Qt Signal
        self.action_signal.emit(data)
        
    def dispatch_action(self, data):
        # Delegate all web action execution to the viewer's registered action handlers
        if hasattr(self.viewer, "handle_web_action"):
            self.viewer.handle_web_action(data)

class ThreadSafeHTTPServer(http.server.ThreadingHTTPServer):
    def server_bind(self):
        _configure_address_reuse(self, platform_name=os.name)
        super().server_bind()

    def __init__(self, server_address, RequestHandlerClass, viewer):
        super().__init__(server_address, RequestHandlerClass)
        self.viewer = viewer
        self.serve_thread = None
        self.serve_error = None
        self._lifecycle_lock = threading.Lock()
        self._stopping = False
        self._stopped = False
        self.event_queues = []
        self.event_clients = {}
        self.queues_lock = threading.Lock()
        registry = getattr(viewer, "web_plugin_registry", None)
        self.static_routes = (
            registry.static_routes if registry is not None else {}
        )

        # Vendored fonts are a shared resource rather than one panel's asset, so
        # they are registered here instead of by an individual backend. This
        # keeps the web UI working without network access.
        self.static_routes["/fonts/"] = os.path.join(
            os.path.dirname(BASE_DIR), "resources", "fonts"
        )

    def register_event_queue(self, event_queue, client_id=None):
        with self.queues_lock:
            self.event_queues.append(event_queue)
            if client_id:
                self.event_clients.setdefault(client_id, set()).add(event_queue)
                pending = getattr(self.viewer, "_web_ui_pending_opens", None)
                if isinstance(pending, dict):
                    pending.pop(client_id, None)

    def unregister_event_queue(self, event_queue, client_id=None):
        with self.queues_lock:
            if event_queue in self.event_queues:
                self.event_queues.remove(event_queue)
            if client_id:
                client_queues = self.event_clients.get(client_id)
                if client_queues is not None:
                    client_queues.discard(event_queue)
                    if not client_queues:
                        self.event_clients.pop(client_id, None)

    def has_event_client(self, client_id):
        with self.queues_lock:
            return bool(self.event_clients.get(client_id))

    def handle_error(self, request, client_address):
        # Suppress traceback print for socket/connection abortions when browser tabs close
        import sys
        exc_type, exc_value, _ = sys.exc_info()
        if exc_type in (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            return
        super().handle_error(request, client_address)


def _configure_address_reuse(server, *, platform_name):
    """Apply portable loopback address-ownership policy before ``bind``."""
    if platform_name == "nt":
        server.allow_reuse_address = False
        exclusive_option = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive_option is not None:
            server.socket.setsockopt(socket.SOL_SOCKET, exclusive_option, 1)
        return
    server.allow_reuse_address = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # src/web_ui/

MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".md": "text/plain",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf"
}


def event_client_from_path(path):
    """Return a bounded client label from an SSE request URL, if present."""
    values = parse_qs(urlsplit(path).query).get("client", [])
    if len(values) != 1:
        return None
    client_id = values[0].strip().lower()
    if re.fullmatch(r"[a-z0-9_-]{1,64}", client_id) is None:
        return None
    return client_id

class WebServerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silences console log spam
        pass

    def do_GET(self):
        clean_path = self.path.split('?', 1)[0]
        if clean_path == "/api/events":
            self.handle_sse(event_client_from_path(self.path))
            return

        # Strip query parameters (e.g. ?v=1)
        if clean_path == "/" or clean_path == "":
            clean_path = "/index.html"

        # Check dynamically registered static routes first (e.g. for agent files)
        for route_prefix, local_dir in self.server.static_routes.items():
            if clean_path.startswith(route_prefix):
                rel = clean_path[len(route_prefix):]
                normalized = os.path.normpath(rel)
                if normalized.startswith("..") or os.path.isabs(normalized):
                    self.send_error(403, "Forbidden")
                    return
                filepath = os.path.normpath(os.path.join(local_dir, normalized))
                if not filepath.startswith(os.path.normpath(local_dir)):
                    self.send_error(403, "Forbidden")
                    return
                if not os.path.isfile(filepath):
                    self.send_error(404, "File Not Found")
                    return
                ext = os.path.splitext(filepath)[1].lower()
                self.serve_file(filepath, MIME_TYPES.get(ext, "application/octet-stream"))
                return

        # Fallback to serving public files inside BASE_DIR (src/web_ui)
        safe_rel_path = clean_path.lstrip("/")
        normalized = os.path.normpath(safe_rel_path)
        if normalized.startswith("..") or os.path.isabs(normalized):
            self.send_error(403, "Forbidden")
            return

        filepath = os.path.normpath(os.path.join(BASE_DIR, normalized))
        if not filepath.startswith(os.path.normpath(BASE_DIR)):
            self.send_error(403, "Forbidden")
            return

        if not os.path.isfile(filepath):
            self.send_error(404, "File Not Found")
            return

        # Determine MIME type dynamically
        ext = os.path.splitext(filepath)[1].lower()
        content_type = MIME_TYPES.get(ext, "application/octet-stream")
        self.serve_file(filepath, content_type)

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def handle_sse(self, client_id=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        
        q = queue.Queue()
        self.server.register_event_queue(q, client_id)
        
        try:
            # Send initial configuration
            initial_data = self.server.viewer.get_initial_web_state()
            self.wfile.write(f"data: {json.dumps({'type': 'init', 'data': initial_data}, cls=NumpyEncoder)}\n\n".encode('utf-8'))
            self.wfile.flush()
            
            while True:
                try:
                    event = q.get(timeout=1.0)
                    self.wfile.write(f"data: {json.dumps(event, cls=NumpyEncoder)}\n\n".encode('utf-8'))
                    self.wfile.flush()
                except queue.Empty:
                    # Send a keep-alive ping to prevent timeouts
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except Exception:
            pass
        finally:
            self.server.unregister_event_queue(q, client_id)

    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                self.server.viewer.communicator.handle_action(data)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok"}, cls=NumpyEncoder).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))

def _address_unavailable(error):
    return (
        error.errno in {errno.EADDRINUSE, errno.EACCES}
        or getattr(error, "winerror", None) in {10013, 10048}
    )


def _serve(server):
    try:
        server.serve_forever()
    except Exception as error:
        server.serve_error = error
        print(f"WebServer serving thread stopped unexpectedly: {error}")


def is_running(server):
    """Return whether ``server`` still owns an open socket and serving thread."""
    if server is None or getattr(server, "_stopping", False):
        return False
    thread = getattr(server, "serve_thread", None)
    server_socket = getattr(server, "socket", None)
    try:
        socket_open = server_socket is not None and server_socket.fileno() >= 0
    except (OSError, ValueError):
        socket_open = False
    return bool(thread is not None and thread.is_alive() and socket_open)


def stop_server(server):
    """Stop and close ``server`` once; repeated calls are safe."""
    if server is None:
        return
    lifecycle_lock = getattr(server, "_lifecycle_lock", None)
    if lifecycle_lock is None:
        lifecycle_lock = threading.Lock()
        server._lifecycle_lock = lifecycle_lock
    with lifecycle_lock:
        if getattr(server, "_stopping", False) or getattr(server, "_stopped", False):
            return
        server._stopping = True

    thread = getattr(server, "serve_thread", None)
    try:
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            server.shutdown()
    finally:
        try:
            server.server_close()
        finally:
            server._stopped = True


def start_server(viewer, preferred_port=8000):
    """Start on the traditional port, falling back when another Viewer owns it."""
    try:
        server = ThreadSafeHTTPServer(
            (LOOPBACK_HOST, preferred_port), WebServerHandler, viewer
        )
    except OSError as error:
        if not _address_unavailable(error):
            raise
        server = ThreadSafeHTTPServer((LOOPBACK_HOST, 0), WebServerHandler, viewer)
    thread = threading.Thread(
        target=_serve,
        args=(server,),
        daemon=True,
        name=f"SSNWebServer:{server.server_address[1]}",
    )
    server.serve_thread = thread
    try:
        thread.start()
    except Exception:
        server.server_close()
        raise
    return server


def ensure_server(viewer, server, preferred_port=8000):
    """Return a live server, replacing a stopped instance when necessary."""
    if is_running(server):
        return server
    stop_server(server)
    return start_server(viewer, preferred_port=preferred_port)
