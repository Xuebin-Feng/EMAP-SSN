# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
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

import base64
import http.server
import errno
import hashlib
import hmac
import socket
import threading
import queue
import json
import os
import re
import secrets
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit
from PySide6 import QtCore, QtWidgets

from desktop.Viewer_Inspection import (
    ViewerInspectionError,
    ViewerInspectionService,
)
from utilities.Viewer_Sessions import (
    SESSION_PROTOCOL_VERSION,
    publish_viewer_session,
    remove_viewer_session,
)


LOOPBACK_HOST = "127.0.0.1"
# Spellings a browser or a local tool may legitimately use to reach this
# server. A request naming anything else was pointed here by a DNS record
# the Viewer does not control, which is how a rebound name reads loopback
# data that is otherwise unreachable from the web.
LOOPBACK_HOST_NAMES = ("127.0.0.1", "localhost")
# /api/action reads its whole body into memory before parsing it, so it needs
# a bound -- but not the inspection endpoints' 64 KiB. The bundled pages post
# far more through here: agent.html sends base64 PNG attachments, and
# esmfold.html re-posts the entire Mol* session after every folded structure.
# The ceiling belongs to agent_images.validate_attachments, which accepts
# MAX_ATTACHMENTS data URLs of MAX_IMAGE_BYTES * 4 // 3 + 100 bytes each; a
# transport cap under that would reject messages the agent layer would then
# have accepted. tests/test_viewer_web_origin.py fails if the two drift apart.
MAX_ACTION_BYTES = 300 * 1024 * 1024

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


class _InspectionCall:
    def __init__(self, method_name, arguments):
        self.method_name = method_name
        self.arguments = arguments
        self.event = threading.Event()
        self.result = None
        self.error = None


class QtInspectionBridge(QtCore.QObject):
    """Run read-only snapshots on the Viewer's owning Qt thread."""

    inspection_signal = QtCore.Signal(object)

    def __init__(self, viewer):
        super().__init__()
        self.service = getattr(viewer, "viewer_inspection", None)
        if self.service is None:
            self.service = ViewerInspectionService(viewer)
        self.inspection_signal.connect(
            self._dispatch,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )

    def request(self, method_name, *, timeout=5.0, **arguments):
        if QtCore.QThread.currentThread() == self.thread():
            return getattr(self.service, method_name)(**arguments)
        call = _InspectionCall(method_name, arguments)
        self.inspection_signal.emit(call)
        if not call.event.wait(timeout):
            raise TimeoutError("The Viewer did not answer the inspection request in time.")
        if call.error is not None:
            raise call.error
        return call.result

    @QtCore.Slot(object)
    def _dispatch(self, call):
        try:
            call.result = getattr(self.service, call.method_name)(**call.arguments)
        except Exception as error:
            call.error = error
        finally:
            call.event.set()

class ThreadSafeHTTPServer(http.server.ThreadingHTTPServer):
    def server_bind(self):
        _configure_address_reuse(self, platform_name=os.name)
        super().server_bind()

    def __init__(self, server_address, RequestHandlerClass, viewer):
        super().__init__(server_address, RequestHandlerClass)
        self.viewer = viewer
        self.inspection_bridge = QtInspectionBridge(viewer)
        self.inspection_token = secrets.token_urlsafe(32)
        from utilities.Viewer_Sessions import ensure_viewer_identity
        self.inspection_session_id = ensure_viewer_identity(viewer)
        self.inspection_started_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.inspection_descriptor = None
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

# Content-Security-Policy sent with every HTML page. Inline <script> blocks
# run only by their SHA-256 hashes, computed from the file being served, so
# markup injected into a page (event-handler attributes, javascript: URLs)
# cannot run script. Chat attachments and Mol* snapshots are data:/blob:
# images; images from other hosts, such as one named in a model's reply, are
# not loaded.
PAGE_POLICY = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self' 'unsafe-inline'",
    "img-src": "'self' data: blob:",
    "connect-src": "'self'",
    "object-src": "'none'",
    "base-uri": "'none'",
    "form-action": "'none'",
    "frame-ancestors": "'none'",
}
# Mol* compiles code while it loads: its MP4 encoder is built with embind,
# which calls new Function, and compiles WebAssembly (also allowed by
# 'unsafe-eval') fetched from a data: URL. Mol* also contacts public servers:
# it lists remote states on load and downloads structures, maps and saved
# states on request.
PAGE_POLICY_ADDITIONS = {
    "esmfold.html": {"script-src": "'unsafe-eval'", "connect-src": "https: data:"},
}
INLINE_SCRIPT = re.compile(
    rb"<script\b(?![^>]*\ssrc\s*=)[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL
)


def content_security_policy(page_name, body):
    """Return the Content-Security-Policy for the HTML page ``body``."""
    directives = dict(PAGE_POLICY)
    for name, sources in PAGE_POLICY_ADDITIONS.get(page_name, {}).items():
        directives[name] += " " + sources
    for script in INLINE_SCRIPT.findall(body):
        # Browsers hash a script's text after the HTML parser has turned CRLF
        # and CR into LF.
        text = script.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = base64.b64encode(hashlib.sha256(text).digest()).decode("ascii")
        directives["script-src"] += f" 'sha256-{digest}'"
    return "; ".join(f"{name} {sources}" for name, sources in directives.items())


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
        if self._reject_foreign_request():
            return
        parsed_path = urlsplit(self.path)
        clean_path = parsed_path.path
        if clean_path.startswith("/api/mcp/v1/"):
            self.handle_mcp_inspection(clean_path, parsed_path.query)
            return
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

    def _send_json(self, status, payload, *, headers=None):
        body = json.dumps(payload, cls=NumpyEncoder, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        try:
            self.wfile.flush()
        except Exception:
            pass

    def _inspection_authorized(self):
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.inspection_token}"
        return hmac.compare_digest(supplied, expected)

    def _local_addresses(self):
        """Return the ``host:port`` and origin spellings naming this server."""
        port = self.server.server_address[1]
        hosts = {f"{name}:{port}" for name in LOOPBACK_HOST_NAMES}
        return hosts, {f"http://{host}" for host in hosts}

    def _foreign_request(self):
        """Return why this request is not from a page this Viewer served.

        A browser sends a cross-origin POST carrying a ``text/plain`` body
        without asking the server first, so without this check any page the
        user visits could drive ``/api/action`` blind. ``Origin`` and
        ``Sec-Fetch-Site`` are set by the browser and cannot be forged by a
        page's scripts, while local callers such as the ESMFold worker send
        neither, so an absent header stays allowed. ``Host`` is always
        present, so it can be required to match: that is what stops a
        rebound DNS name from reading ``/api/events`` or the bundled pages.
        """
        hosts, origins = self._local_addresses()
        origin = self.headers.get("Origin")
        if origin is not None and origin not in origins:
            return "Cross-origin requests are not accepted."
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in {"same-origin", "none"}:
            return "Cross-site requests are not accepted."
        if self.headers.get("Host") not in hosts:
            return "Requests must address this Viewer as 127.0.0.1 or localhost."
        return None

    def _reject_foreign_request(self):
        """Answer 403 and return whether the request was rejected."""
        reason = self._foreign_request()
        if reason is None:
            return False
        # Nothing has read the body at this point, so drain it before
        # replying; Windows resets the connection otherwise and the client
        # loses this 403 (see _discard_request_body).
        self._send_json(403, {"error": reason}, headers=self._discard_request_body())
        return True

    def _discard_request_body(self, limit=65536):
        """Read and drop an unused request body; return extra reply headers.

        Windows resets the connection (RST instead of FIN) when a socket is
        closed with unread received data, so the client can lose an early
        reply such as a 401. A body that is chunked, has an invalid length
        or exceeds ``limit`` stays unread, and the reply gets
        ``Connection: close`` instead.
        """
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if self.headers.get("Transfer-Encoding") or not 0 <= length <= limit:
            return {"Connection": "close"}
        self.rfile.read(length)
        return {}

    def handle_mcp_inspection(self, clean_path, query_string):
        if not self._inspection_authorized():
            self._send_json(
                401,
                {"error": "Missing or invalid Viewer inspection token."},
                headers={"WWW-Authenticate": "Bearer", **self._discard_request_body()},
            )
            return
        try:
            if clean_path == "/api/mcp/v1/session":
                payload = {
                    "protocol_version": SESSION_PROTOCOL_VERSION,
                    "inspection_capabilities": ["snapshots_v1", "commands_v1", "capture_view_v1", "alignment_snapshots_v1", "node_projection_v1"],
                    "inputs": self.server.inspection_bridge.service._input_paths(),
                    "session_id": self.server.inspection_session_id,
                    "session_alias": self.server.viewer.inspection_session_alias,
                    "pid": os.getpid(),
                    "started_at": self.server.inspection_started_at,
                    "launch_id": os.environ.get("SSN_VIEWER_LAUNCH_ID"),
                }
            elif clean_path == "/api/mcp/v1/summary":
                payload = self.server.inspection_bridge.request("get_summary")
            elif clean_path == "/api/mcp/v1/nodes":
                query = parse_qs(query_string, keep_blank_values=True)
                raw_columns = query.get("columns")
                columns = None
                if raw_columns is not None:
                    columns = []
                    for value in raw_columns:
                        columns.extend(
                            column.strip()
                            for column in value.split(",")
                            if column.strip()
                        )
                payload = self.server.inspection_bridge.request(
                    "query_nodes",
                    scope=query.get("scope", ["all"])[0],
                    offset=query.get("offset", [0])[0],
                    limit=query.get("limit", [100])[0],
                    columns=columns,
                )
            else:
                self._send_json(404, {"error": "Inspection endpoint not found."})
                return
        except ViewerInspectionError as error:
            self._send_json(400, {"error": str(error)})
            return
        except TimeoutError as error:
            self._send_json(503, {"error": str(error)})
            return
        except Exception as error:
            self._send_json(500, {"error": f"Viewer inspection failed: {error}"})
            return
        self._send_json(200, payload)

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} Not Found")
            return
        with open(filepath, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        if content_type == "text/html":
            self.send_header(
                "Content-Security-Policy",
                content_security_policy(os.path.basename(filepath), body),
            )
        self.end_headers()
        self.wfile.write(body)

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
        if self._reject_foreign_request():
            return
        if self.path == "/api/mcp/v1/commands":
            if not self._inspection_authorized():
                self._send_json(401, {"error": "Missing or invalid Viewer inspection token."},
                                headers=self._discard_request_body())
                return
            body = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Command request must be 1..65536 bytes")
                body = self.rfile.read(length)
                request = json.loads(body.decode("utf-8"))
                from mcp_server.core.Workflow_Dispatch import REGISTRY
                action = request['action']
                workflow = 'emapssn_viewer_control' if action == 'execute_commands' else 'emapssn_viewer_data'
                if action not in {'execute_commands', 'get_command_request', 'list_command_requests', 'read_command_output', 'capture_view', 'get_command_catalog'}:
                    raise ValueError('Unknown command portal action')
                arguments = REGISTRY[workflow][action].model.model_validate(request.get('arguments', {})).model_dump()
                arguments.pop('session_id', None)
                result = self.server.inspection_bridge.request('command_action', action=action, arguments=arguments)
                self._send_json(200, result)
            except TimeoutError:
                self._send_json(503, {'error': 'Viewer response timed out. Retry submission with the SAME submission_id; it may already be queued.'})
            except (ValueError, TypeError, KeyError) as error:
                self._send_json(400, {'error': str(error)[:1000]},
                                headers=self._discard_request_body() if body is None else None)
            return
        if self.path == "/api/mcp/v1/data":
            if not self._inspection_authorized():
                self._send_json(401, {"error": "Missing or invalid Viewer inspection token."},
                                headers=self._discard_request_body())
                return
            body = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ViewerInspectionError("Inspection request must be 1..65536 bytes")
                body = self.rfile.read(length)
                request = json.loads(body.decode("utf-8"))
                action = request["action"]
                from mcp_server.core.Workflow_Dispatch import REGISTRY
                if action not in {"get_residue_distribution", "get_summary", "describe_fields", "create_subset", "summarize_subset", "query_nodes", "read_value"}:
                    raise ViewerInspectionError("Unknown read-only inspection action")
                arguments = REGISTRY["emapssn_viewer_data"][action].model.model_validate(request.get("arguments", {})).model_dump()
                arguments.pop("session_id", None)
                service = self.server.inspection_bridge.service
                sid = (self.server.inspection_bridge.request("capture_snapshot", **{key: True for key in ("include_visual", "include_alignment") if arguments.pop(key, False)})
                       if action == "get_summary" else arguments.pop("snapshot_id"))
                payload = service.snapshots.execute(action, sid, **arguments)
                self._send_json(200, payload)
            except TimeoutError:
                self._send_json(503, {"error": "Viewer snapshot capture timed out; retry when idle."})
            except (ValueError, TypeError, KeyError) as error:
                self._send_json(400, {"error": str(error)[:1000]},
                                headers=self._discard_request_body() if body is None else None)
            return
        if self.path == "/api/mcp/v1/shutdown":
            # No shutdown reply uses the request body; drain it before any.
            closing = self._discard_request_body()
            if not self._inspection_authorized():
                self._send_json(
                    401,
                    {"error": "Missing or invalid Viewer inspection token."},
                    headers={"WWW-Authenticate": "Bearer", **closing},
                )
                return
            qapp = QtWidgets.QApplication.instance()
            if qapp is None:
                self._send_json(503, {"error": "Viewer application is unavailable."}, headers=closing)
                return
            queued = QtCore.QMetaObject.invokeMethod(
                qapp, "quit", QtCore.Qt.ConnectionType.QueuedConnection
            )
            if not queued:
                self._send_json(503, {"error": "Could not queue Viewer shutdown."}, headers=closing)
                return
            self._send_json(202, {"status": "accepted", "message": "Viewer shutdown queued."},
                            headers=closing)
            return

        if self.path == "/api/agent/image":
            from web_ui.agent_images import MAX_IMAGE_BYTES, inspect_source
            body = None
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > MAX_IMAGE_BYTES:
                    self.close_connection = True
                    self._send_json(413, {'error': 'Each image must be nonempty and at most 20 MiB.'},
                                    headers=self._discard_request_body(MAX_IMAGE_BYTES))
                    return
                body = self.rfile.read(length)
                self._send_json(200, inspect_source(body))
            except ValueError as error:
                self._send_json(400, {'error': str(error)},
                                headers=self._discard_request_body(MAX_IMAGE_BYTES) if body is None else None)
            return

        if self.path == "/api/action":
            body = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_ACTION_BYTES:
                    raise ValueError(f"Action request must be 1..{MAX_ACTION_BYTES} bytes")
                # A cross-origin caller cannot set this content type without a
                # CORS preflight, which this server never answers. That makes it
                # a barrier independent of Origin and Sec-Fetch-Site, built from
                # a request this server deliberately does not handle.
                media_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
                if media_type != "application/json":
                    raise ValueError("Action requests must use Content-Type: application/json")
                body = self.rfile.read(length)
                data = json.loads(body.decode("utf-8"))
                self.server.viewer.communicator.handle_action(data)
                self._send_json(200, {"status": "ok"})
            except ValueError as error:
                # Drain only a small body: the point is to stop Windows
                # resetting the connection before a short reply lands. Draining
                # up to MAX_ACTION_BYTES would instead read hundreds of
                # megabytes to answer a 400, so anything larger gets
                # Connection: close and the client hangs up.
                self._send_json(
                    400,
                    {"error": str(error)[:1000]},
                    headers=self._discard_request_body() if body is None else None,
                )
            except Exception as error:
                self._send_json(500, {"error": str(error)[:1000]})
            return

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
            descriptor = getattr(server, "inspection_descriptor", None)
            if descriptor is not None:
                remove_viewer_session(descriptor)
                server.inspection_descriptor = None
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
    try:
        server.inspection_descriptor = publish_viewer_session(
            session_id=server.inspection_session_id,
            pid=os.getpid(),
            host=LOOPBACK_HOST,
            port=int(server.server_address[1]),
            token=server.inspection_token,
            started_at=server.inspection_started_at,
        )
    except Exception as error:
        # The existing browser server remains useful even when discovery cannot
        # be published (for example, a locked-down temporary directory).
        print(f"Warning: Could not publish Viewer inspection session: {error}")
    return server


def ensure_server(viewer, server, preferred_port=8000):
    """Return a live server, replacing a stopped instance when necessary."""
    if is_running(server):
        return server
    stop_server(server)
    return start_server(viewer, preferred_port=preferred_port)
