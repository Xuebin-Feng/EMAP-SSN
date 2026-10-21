# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Fold View page tests' shared harness: the real web server, Qt WebEngine and the bundled Mol*.

esmfold.html is opened in Qt WebEngine over the real Viewer web server, which
serves the page, its scripts, Mol* and a folder of fixture structures
(tests/esmfold_layer_fixtures.py). The offscreen engine has no WebGL here, so
Mol* has no canvas and a page is checked through its state and its DOM, not
pixels; tests/test_esmfold_shell_webgl.py runs the checks that need a canvas
in a child process that has WebGL, and skips where there is none.

Not collected by unittest; import with ``from tests.esmfold_page_harness import ...``.
"""

import http.client
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Set before Qt WebEngine starts Chromium, as tests/test_agent_composer.py does.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication


class ConsolePage(QWebEnginePage):
    """A page that keeps what its scripts write to the console, as (level, message, line, source)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        self.messages.append((level.name, message, line, source))


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


def make_viewer():
    """The little a Viewer must have for the real web server to run over it."""
    from desktop.Viewer_Inspection import ViewerInspectionService
    from web_ui import Web_Server

    viewer = SimpleNamespace(
        n_nodes=2, full_headers=["a", "b"], metadata={}, edges=[], visible_mask=[True, True], selected_indices=[],
        group_labels=[[], []], cluster_labels=None, web_plugin_registry=None)
    viewer.viewer_inspection = ViewerInspectionService(viewer)
    viewer.communicator = Web_Server.QtCommunicator(viewer)
    viewer.get_initial_web_state = lambda: {"rows": []}
    viewer.handle_web_action = lambda data: None
    return viewer


class FoldViewServerMixin:
    """setUp for a TestCase: a folder of structures, and the real web server over it.

    Gives self.directory (what /structures/ serves), self.outer (a folder around
    it that nothing may reach), self.viewer and self.server. Put it before
    unittest.TestCase.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        super().setUp()
        from web_ui import Web_Server

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.outer = folder.name
        self.directory = os.path.join(self.outer, "structures")
        os.makedirs(self.directory)
        sessions = tempfile.TemporaryDirectory()
        self.addCleanup(sessions.cleanup)
        environment = mock.patch.dict(os.environ, {"SSN_VIEWER_SESSION_DIR": sessions.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.viewer = make_viewer()
        self.server = Web_Server.start_server(self.viewer, preferred_port=0)
        self.addCleanup(Web_Server.stop_server, self.server)
        self.server.static_routes["/structures/"] = self.directory
        self.server.static_routes["/esmfold/"] = str(SRC / "resources" / "esmfold")

    def get_json(self, path, headers=None):
        status, response_headers, body = http_get(
            self.server, path, page_headers(self.server) if headers is None else headers)
        try:
            return status, response_headers, json.loads(body)
        except ValueError:
            return status, response_headers, body


class FoldViewPageMixin:
    """The page in a Qt WebEngine view, and the Viewer's side of its conversation.

    Put it before the class that starts the server (self.server, self.viewer).
    The Viewer's side: a restore_session answer to the page's load request
    (self.session is what it restores), every saved session in self.saved, and
    every other action the page posts in self.actions.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.view = QWebEngineView()
        cls.console = ConsolePage(cls.view)
        cls.view.setPage(cls.console)
        cls.view.resize(1300, 800)
        cls.view.show()

    @classmethod
    def tearDownClass(cls):
        cls.view.close()
        cls.view.deleteLater()
        cls.app.processEvents()

    def setUp(self):
        super().setUp()
        self.session = None
        self.saved = []
        self.actions = []
        self.console.messages.clear()
        self.viewer.handle_web_action = self.handle_action

    def tearDown(self):
        self.view.setUrl(QUrl("about:blank"))
        self.pump(0.3)

    def pump(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

    def handle_action(self, data):
        """The Viewer's side of the page's session handshake and its saves."""
        if data.get("action") == "load_molstar_session":
            self.broadcast({"type": "restore_session", "session": self.session})
        elif data.get("action") == "save_molstar_session":
            self.saved.append(data["session"])
        else:
            self.actions.append(data)

    def broadcast(self, event):
        with self.server.queues_lock:
            for queue in list(self.server.event_queues):
                queue.put(event)

    def js(self, source, timeout=20):
        result = []
        self.view.page().runJavaScript(source, lambda value: result.append(value))
        deadline = time.monotonic() + timeout
        while not result and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertTrue(result, source)
        return result[0]

    def script(self, body, timeout=60):
        """The JSON result of the async JavaScript `body`, which may await."""
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

    def console_errors(self):
        """What the page's scripts wrote to the console as errors or warnings since the test began.

        Mol* itself complains three times when the engine gives it no WebGL
        (it cannot make a canvas, and a call that reads the canvas fails);
        those are left out, as the page's job then is to say so (data-view
        "webgl"), and a test that has WebGL does not have them to leave out.
        """
        return [(level, message) for level, message, _, source in self.console.messages
                if level in ("ErrorMessageLevel", "WarningMessageLevel")
                and not (source.endswith("/esmfold/molstar.js")
                         and ("Could not create a WebGL rendering context" in message
                              or "Cannot read properties of undefined (reading 'reprCount')" in message))]

    def load_page(self, path="/esmfold.html"):
        loaded = []
        self.view.loadFinished.connect(loaded.append)
        self.view.load(QUrl(f"http://127.0.0.1:{self.server.server_address[1]}{path}"))
        deadline = time.monotonic() + 30
        while not loaded and time.monotonic() < deadline:
            self.pump(0.02)
        self.view.loadFinished.disconnect(loaded.append)
        self.assertEqual(loaded, [True])

    def open_page(self):
        self.load_page()
        # Mol* is up and the page has had its session answer.
        self.script("""
            for (let i = 0; i < 600 && !(typeof plugin !== 'undefined' && plugin && !isRestoring); i++) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            if (isRestoring) throw new Error('the page never finished restoring');
            return 1;
            """, timeout=90)
        self.js(HELPERS)

    def push(self, name):
        """Send the page a folded structure, as the Viewer does, and wait until it is loaded."""
        before = self.script("return loadedStructures().length")
        self.broadcast({"type": "esmfold_pdb", "node_id": name[:-4], "pdb_url": f"/structures/{name}"})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.script("await structureQueue; return loadedStructures().length") > before:
                break
            self.pump(0.1)
        else:
            self.fail(f"{name} was not loaded")
        self.script("await structureQueue; await cardNow(); return 1")


# Test-side readers of the page's Mol* state. Everything they return is plain JSON.
HELPERS = """
// Without WebGL, Mol* has no canvas, and a snapshot that includes the camera cannot be taken.
(() => {
    const snapshot = plugin.state.getSnapshot.bind(plugin.state);
    plugin.state.getSnapshot = params => snapshot({camera: !!plugin.canvas3d, ...params});
})();
window.layerState = function (index) {
    const structure = loadedStructures()[index];
    const model = structure.model.properties.cell.transform.params;
    const props = structure.properties.cell.transform.params;
    return {active: activeLayerId(structure), file: structureFile(structure),
            isMvs: model.properties['mvs-is-mvs-model'].isMvs,
            annotations: model.properties['mvs-annotations'].annotations.map(
                a => ({id: a.id, url: a.source.params.url, schema: a.schema})),
            modelAttach: model.autoAttach, tooltips: props.properties['mvs-annotation-tooltips'].tooltips,
            structureAttach: props.autoAttach,
            themes: structure.components.flatMap(c => c.representations.map(r => r.cell.transform.params.colorTheme))};
};
window.reprThemes = function () {
    return loadedStructures().map(s => s.components[0].representations[0].cell.transform.params.colorTheme);
};
window.colors = function (index, themeName, params) {
    const L = molstar.lib.structure;
    const data = loadedStructures()[index].cell.obj.data;
    const theme = plugin.representation.structure.themes.colorThemeRegistry.create(themeName, {structure: data}, params);
    const location = L.StructureElement.Location.create(data);
    const out = {};
    for (const unit of data.units) {
        location.unit = unit;
        for (const element of unit.elements) {
            location.element = element;
            if (L.StructureProperties.atom.label_atom_id(location) !== 'CA') continue;
            out[L.StructureProperties.residue.auth_seq_id(location)] =
                '#' + theme.color(location, false).toString(16).padStart(6, '0');
        }
    }
    return out;
};
// What Mol* shows for the pointer over residue `group` (counted from 0) of a structure.
window.tooltip = async function (index, group) {
    const repr = loadedStructures()[index].components[0].representations[0].cell.obj.data.repr;
    const loci = repr.getLoci({objectId: repr.renderObjects[0].id, instanceId: 0, groupId: group});
    plugin.managers.interactivity.lociHighlights.highlightOnly({loci, repr});
    await new Promise(resolve => setTimeout(resolve, 200));
    const labels = plugin.managers.lociLabels.getLabels().map(String);
    plugin.managers.interactivity.lociHighlights.clearHighlights();
    return labels;
};
window.fit = function (index) {
    const structure = loadedStructures()[index];
    const trace = s => readCaTrace((s.transform || s).cell.obj.data);
    const a = trace(loadedStructures()[0]), b = trace(structure);
    let sum = 0;
    const n = Math.min(a.coords.length, b.coords.length);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) sum += (a.coords[i][d] - b.coords[i][d]) ** 2;
    const node = structure.transform ? structure.transform.cell.transform.params : null;
    return {transformed: !!structure.transform, rmsd: Math.sqrt(sum / n), matrix: node ? Array.from(node.transform.params.data) : null};
};
// The Color by card as a person sees it: its buttons, the legend of the chosen coloring, the notes.
window.card = function () {
    const body = document.getElementById('color-body');
    const chips = [...body.querySelectorAll('[role="radio"]')];
    const legend = body.querySelector('#color-legend');
    const compact = document.getElementById('canvas-legend');
    const text = node => (node ? node.textContent : null);
    const chosen = chips.find(chip => chip.getAttribute('aria-checked') === 'true');
    return {
        title: text(document.getElementById('h-color')),
        chips: chips.map(chip => [chip.dataset.coloring, chip.firstChild.textContent, text(chip.querySelector('.cov')) || '']),
        selected: chosen ? chosen.dataset.coloring : null,
        legendTitle: legend ? legend.querySelector('.t').firstChild.textContent : null,
        legendUnit: legend ? text(legend.querySelector('.t small')) : null,
        bar: !!(legend && legend.querySelector('.ramp')),
        ends: legend ? [...legend.querySelectorAll('.ticks span')].map(s => s.textContent) : [],
        keys: legend ? [...legend.querySelectorAll('.band')].map(k => k.textContent) : [],
        noData: legend ? text(legend.querySelector('.nodata span')) : null,
        caution: legend ? text(legend.querySelector('.caution')) : null,
        source: legend ? text(legend.querySelector('.src summary span')) : null,
        description: legend ? text(legend.querySelector('.src p')) : null,
        layerNotes: [...body.querySelectorAll('.empty-layers div > div')].map(n => n.textContent),
        compact: {shown: compact.dataset.shown, title: text(compact.querySelector('.t')), ends: [...compact.querySelectorAll('.ticks span')].map(s => s.textContent)}
    };
};
window.cardNow = async function () { await renderColorCard(true); return card(); };
"""
