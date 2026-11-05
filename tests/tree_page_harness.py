# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Tree View page tests' shared harness: tree.html in Qt WebEngine over the real web server.

The page is opened offscreen (no window is shown on the desktop) over the Viewer's own web server, which serves
the page and its scripts, and the tree backend's /api/tree/data answers from a temporary folder holding an
alignment and the tree file the writer saves for it (tests/tree_fixtures.py). The page's canvas is checked through
the page's own state and its pixels: ``TreeView.state`` holds the model, the layout and the view, and
``TreeView.lastStats`` what the last frame drew.

Not collected by unittest; import with ``from tests.tree_page_harness import ...``.
"""

import json
import os
import tempfile
import time
from unittest import mock

# esmfold_page_harness sets the offscreen platform and the GPU flag before Qt WebEngine starts Chromium.
from tests.esmfold_page_harness import ConsolePage
from tests import tree_fixtures
from tests.tree_fixtures import http_get, page_headers, server_viewer  # noqa: F401

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication

from web_ui import Web_Server, tree_backend


# Events the page's own handlers answer, dispatched as a browser would send them, at a point of the canvas.
EVENTS = """
window.fire = {
    canvas() { return document.getElementById('tree-canvas'); },
    // Where, in the canvas's own pixels, the page finds the pointer that these helpers put at (x, y): an event's
    // position is a whole number of pixels, and the canvas can lie on a fraction of one (under a banner of text).
    local(x, y) {
        const rect = this.canvas().getBoundingClientRect();
        const event = new MouseEvent('mousemove', {clientX: rect.left + x, clientY: rect.top + y});
        return [event.clientX - rect.left, event.clientY - rect.top];
    },
    wheel(deltaY, x, y, options = {}) {
        const rect = this.canvas().getBoundingClientRect();
        const event = new WheelEvent('wheel', {deltaY, deltaMode: 0, clientX: rect.left + x, clientY: rect.top + y,
            bubbles: true, cancelable: true, ...options});
        this.canvas().dispatchEvent(event);
        return event.defaultPrevented;
    },
    pointer(type, x, y, options = {}) {
        const rect = this.canvas().getBoundingClientRect();
        this.canvas().dispatchEvent(new PointerEvent(type, {pointerId: 7, button: 0, buttons: type === 'pointerup' ? 0 : 1,
            clientX: rect.left + x, clientY: rect.top + y, bubbles: true, cancelable: true, ...options}));
    },
    move(x, y) {
        const rect = this.canvas().getBoundingClientRect();
        this.canvas().dispatchEvent(new PointerEvent('pointermove', {pointerId: 7, button: -1, buttons: 0,
            clientX: rect.left + x, clientY: rect.top + y, bubbles: true}));
    },
    dblclick(x, y) {
        const rect = this.canvas().getBoundingClientRect();
        this.canvas().dispatchEvent(new MouseEvent('dblclick', {clientX: rect.left + x, clientY: rect.top + y, bubbles: true}));
    },
    key(key, options = {}) {
        const event = new KeyboardEvent('keydown', {key, bubbles: true, cancelable: true, ...options});
        this.canvas().dispatchEvent(event);
        return event.defaultPrevented;
    },
    click(x, y) {
        this.pointer('pointerdown', x, y);
        this.pointer('pointerup', x, y);
    },
    // The colour of the canvas pixel at (x, y), CSS pixels.
    pixel(x, y) {
        const ratio = this.canvas().width / this.canvas().clientWidth;
        return Array.from(this.canvas().getContext('2d').getImageData(Math.round(x * ratio), Math.round(y * ratio), 1, 1).data);
    },
};
"""


class TreePageMixin:
    """A TestCase's Qt WebEngine view of tree.html. Put it before unittest.TestCase.

    start_page_server(viewer) serves the page over a Viewer; open_page() loads it and waits until it shows a tree or
    says why it has none; js(source) and script(body) run JavaScript in it, and drawn() redraws and returns the frame's
    statistics.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.app = QApplication.instance() or QApplication([])
        cls.view = QWebEngineView()
        cls.console = ConsolePage(cls.view)
        cls.view.setPage(cls.console)
        cls.view.resize(1300, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.setUrl(QUrl("about:blank"))
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.console.messages.clear()
        # The tree last read belongs to the test that read it.
        patcher = mock.patch.dict(tree_backend._TREE_CACHE, {"key": None, "tree": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.2)
        super().tearDown()

    # ---- The server and the page ----------------------------------------------------------------

    def make_folder(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return folder.name

    def start_page_server(self, viewer):
        """The real web server over ``viewer``, with Tree View's backend registered; returns the server."""
        sessions = tempfile.TemporaryDirectory()
        self.addCleanup(sessions.cleanup)
        environment = mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": sessions.name})
        environment.start()
        self.addCleanup(environment.stop)
        tree_backend.register_backend(viewer.web_plugin_registry, viewer)
        server = Web_Server.start_server(viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, server)
        self.server = server
        # What the Viewer does with an event: put it in the queue of every page that listens.
        viewer.web_server = server

        def broadcast_event(event):
            with server.queues_lock:
                queues = list(server.event_queues)
            for queue in queues:
                queue.put(event)

        viewer.broadcast_event = broadcast_event
        return server

    def serve_headers(self, headers, *, network=None, folder=None, colors=None, hidden=None, clusters=None, **options):
        """Write the alignment and tree of ``headers``, and serve the page over a Viewer holding them.

        colors, hidden and clusters are what the Viewer holds of its nodes' colours, which are hidden, and their clusters
        (tree_fixtures.make_viewer).
        """
        folder = folder or self.make_folder()
        msa, tree = tree_fixtures.make_alignment_folder(folder, headers, **options)
        viewer = server_viewer()
        fixture = tree_fixtures.make_viewer(msa, headers if network is None else network,
                                            colors=colors, hidden=hidden, clusters=clusters)
        viewer.alignment, viewer.full_headers, viewer.n_nodes = fixture.alignment, fixture.full_headers, fixture.n_nodes
        for name in ("current_colors", "visible_mask", "cluster_labels"):
            if hasattr(fixture, name):
                setattr(viewer, name, getattr(fixture, name))
        self.viewer, self.msa, self.tree_file = viewer, msa, tree
        return self.start_page_server(viewer)

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

    def js(self, source, timeout=20):
        result = []
        self.view.page().runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(result, source)
        return result[0]

    def script(self, body, timeout=30):
        """The JSON result of the async JavaScript ``body``, which may await."""
        self.js("window.__result = null; (async () => { try { window.__result = JSON.stringify(await (async () => {"
                + body + "})()); } catch (error) { window.__result = 'ERROR ' + error.message; } })()")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.js("window.__result")
            if value:
                self.assertFalse(value.startswith("ERROR"), value)
                return json.loads(value)
            self.pump(0.05)
        self.fail("Timed out: " + body)

    def state(self, expression):
        """The JSON value of the JavaScript ``expression``."""
        return self.script(f"return {expression}")

    def wait_for(self, expression, timeout=20):
        """Wait until the JavaScript ``expression`` is true; fail with it otherwise."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.js(f"Boolean({expression})"):
                return
            self.pump(0.05)
        self.fail("Never true: " + expression)

    def load_page(self, path="/tree.html"):
        loaded = []
        self.view.loadFinished.connect(loaded.append)
        self.view.load(QUrl(f"http://127.0.0.1:{self.server.server_address[1]}{path}"))
        deadline = time.monotonic() + 30
        while not loaded and time.monotonic() < deadline:
            self.pump(0.02)
        self.view.loadFinished.disconnect(loaded.append)
        self.assertEqual(loaded, [True])

    def open_page(self, expect="tree"):
        """Load the page and wait until it shows ``expect`` (the body's data-view: tree, empty or loading)."""
        self.load_page()
        self.wait_for(f"document.body.dataset.view === {json.dumps(expect)}")
        self.js(EVENTS)

    def drawn(self):
        """Draw a frame now, and return its statistics."""
        return self.script("return TreeView.drawNow()")

    def console_errors(self):
        """What the page's scripts wrote to the console as errors or warnings since the test began."""
        return [(level, message) for level, message, _, _ in self.console.messages
                if level in ("ErrorMessageLevel", "WarningMessageLevel")]
