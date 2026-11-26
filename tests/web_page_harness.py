# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared, backend-free helpers for Qt WebEngine page tests and local HTTP checks.

Holds the HTTP helpers (``http_get``, ``page_headers``) and ``ConsolePage`` (a
``QWebEnginePage`` subclass that captures JavaScript console messages) without
importing any plugin backend, so unrelated test sections stay decoupled.

Not collected by unittest; import with ``from tests.web_page_harness import ...``.
"""

import http.client
import os

__all__ = ("ConsolePage", "http_get", "page_headers")


def http_get(server, path, headers=None):
    """GET path from the real server; returns (status, headers, body bytes)."""
    port = server.server_address[1]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        for name, value in {"Host": f"127.0.0.1:{port}", **(headers or {})}.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


def page_headers(server):
    """What Chromium sends for a same-origin fetch from a bundled page."""
    port = server.server_address[1]
    return {"Origin": f"http://127.0.0.1:{port}", "Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}


def _build_console_page():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # Set before Qt WebEngine starts Chromium.
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
    from PySide6.QtWebEngineCore import QWebEnginePage

    class ConsolePage(QWebEnginePage):
        """A page that keeps what its scripts write to the console, as (level, message, line, source)."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.messages = []

        def javaScriptConsoleMessage(self, level, message, line, source):
            self.messages.append((level.name, message, line, source))

    ConsolePage.__qualname__ = "ConsolePage"
    return ConsolePage


def __getattr__(name):
    if name == "ConsolePage":
        cls = _build_console_page()
        globals()["ConsolePage"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
