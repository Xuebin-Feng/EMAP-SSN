"""Viewer, command-portal and inspection fixtures shared by several test modules.

The fixtures are mixins, not TestCase subclasses, so importing them never makes
unittest discovery collect and run another module's tests a second time (the
imported TestCase classes used to run 2-3 times per suite).
"""
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PySide6 import QtWidgets  # noqa: E402
import Command_Engine as ce  # noqa: E402
from Viewer_Command_Portal import ViewerCommandPortal, bind  # noqa: E402
from desktop.Viewer_Inspection import ViewerInspectionService  # noqa: E402


class Viewer:
    def __init__(self, root):
        self.command_history = []
        self.history_file = str(Path(root) / 'history.txt')
        self.console_text = SimpleNamespace(text='')
        self.console_bg = SimpleNamespace(visible=False)
        self.tooltip = SimpleNamespace(text='', visible=False)
        self.events = []
        self.n_nodes = 3
        self.full_headers = ['one', 'two', 'three']
        self.visible_mask = np.ones(3, bool)
        self.selected_indices = []
        self.cluster_labels = np.array([0, 0, 1])
        self.group_labels = [set(), set(), set()]
        self.metadata = {}
        self.pos = np.array([[0., 0.], [1., 1.], [2., 2.]])
        self.edges = np.array([[0, 1], [1, 2]])
        self.edge_scores = np.array([.5, .8])
        self.current_slider_threshold = .4
        self.current_colors = np.ones((3, 4))
        self.current_sizes = np.ones(3)
        self.current_shapes = np.array(['disc'] * 3, dtype=object)
        self.saved = 0
    def broadcast_event(self, event): self.events.append(event)
    def broadcast_metadata_state(self): pass
    def update_console_background(self): pass
    def update_nodes(self): pass
    def update_selection_visual(self): pass
    def update_edges(self): pass
    def _save_state(self): self.saved += 1
    def promote_nodes(self, indices): pass
    def set_background_job_status(self, message): pass
    def process_command(self, command, record_history=True):
        with bind(None): ce.execute_command(self, command, record_history)


class PortalFixture:
    """A real ViewerCommandPortal around the fake Viewer, and finish() to await a request."""
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.viewer = Viewer(self.directory.name)
        self.portal = ViewerCommandPortal(self.viewer)
        self.viewer.command_portal = self.portal
        self.addCleanup(self.portal.shutdown)

    def finish(self, request_id, timeout=30):
        """Process Qt events until the request is terminal; fail the test after ``timeout`` s."""
        deadline = time.monotonic() + timeout
        while True:
            request = self.portal.get(request_id)
            if request['status'] in {'succeeded', 'failed', 'cancelled'}:
                return request
            if time.monotonic() >= deadline:
                self.fail(f'Command request timed out after {timeout} s in status {request["status"]!r}: {request}')
            self.app.processEvents()
            time.sleep(.005)


class SnapshotHTTPFixture:
    """A live Viewer web server (isolated session directory) and raw HTTP helpers."""
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        from web_ui import Web_Server
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        env=mock.patch.dict(os.environ, {'SSN_VIEWER_SESSION_DIR':self.directory.name}); env.start(); self.addCleanup(env.stop)
        self.viewer=SimpleNamespace(n_nodes=2,full_headers=['a','b'],metadata={'Org,名':{'type':'text','values':np.array(['x','y'],dtype=object)}},
                                    edges=[],visible_mask=[True,False],selected_indices=[],group_labels=[['g'],[]],cluster_labels=None,web_plugin_registry=None)
        self.viewer.viewer_inspection=ViewerInspectionService(self.viewer)
        self.server=Web_Server.start_server(self.viewer,preferred_port=0)
        self.addCleanup(Web_Server.stop_server,self.server)
        self.url=f'http://127.0.0.1:{self.server.server_address[1]}/api/mcp/v1/data'
    def request(self,action,args=None,token=True):
        outcome={}
        def work():
            request=urllib.request.Request(self.url,data=json.dumps({'action':action,'arguments':args or {}},ensure_ascii=False).encode('utf-8'),
                    headers={'Authorization':'Bearer '+self.server.inspection_token} if token else {})
            try:
                with urllib.request.urlopen(request,timeout=10) as response:
                    data=response.read(); outcome.update(payload=json.loads(data),size=len(data),status=response.status)
            except urllib.error.HTTPError as error:
                outcome.update(status=error.code,payload=json.loads(error.read()))
            except Exception as error: outcome['error']=error
        thread=threading.Thread(target=work); thread.start()
        deadline=time.monotonic()+12
        while thread.is_alive() and time.monotonic()<deadline:
            self.app.processEvents(); thread.join(.01)
        thread.join(.1)
        self.assertFalse(thread.is_alive())
        if 'error' in outcome: raise outcome['error']
        return outcome
    def exchange(self, method, path, headers, body=b'', pause=0.0):
        """Send the headers, then the body after a pause; return (status, headers, payload)."""
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=10)
        try:
            connection.putrequest(method, path)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            time.sleep(pause)
            connection.send(body)
            time.sleep(pause)
            response = connection.getresponse()
            return response.status, response.headers, json.loads(response.read())
        finally:
            connection.close()


class SnapshotFixture:
    """A captured inspection snapshot of a four-node fake Viewer."""
    def setUp(self):
        self.v = SimpleNamespace(n_nodes=4, full_headers=['same', 'same', 'third', 'fourth'],
            visible_mask=np.array([True,False,True,True]), selected_indices=[],
            cluster_labels=np.array([0,0,-1,1]), group_labels=[['a','b'],['a'],[],['b']],
            metadata={'Length': {'type':'number','values':np.array([1.,2.,np.nan,np.inf])},
                      'Org,名': {'type':'text','values':np.array(['x','y','x',''],dtype=object)}}, edges=[], current_slider_threshold=.5)
        self.service=ViewerInspectionService(self.v)
        self.sid=self.service.capture_snapshot()
        self.store=self.service.snapshots
    def call(self, action, **kwargs): return self.store.execute(action,self.sid,**kwargs)


# The four states a Viewer's windows can be in: where no user is at the Viewer
# (headless, or hidden and not shown) nothing may open a dialog, page, terminal or
# file manager; where one is (hidden and shown, or normal) everything works as ever.
WINDOW_STATES = ('headless', 'hidden_unshown', 'hidden_shown', 'normal')
NO_WINDOW_STATES = ('headless', 'hidden_unshown')


def apply_window_state(viewer, state):
    """Make a stand-in ``viewer`` answer shows_windows() as a real Viewer in ``state`` does.

    It is MainViewer's own method, reading the same attributes (headless and the
    window controller), so a change to how the Viewer decides reaches these tests.
    """
    import types
    from EMAPSSN_Viewer import MainViewer
    shown = state in ('hidden_shown', 'normal')
    viewer.headless = state == 'headless'
    viewer.window_mode = {'hidden_unshown': 'hidden', 'hidden_shown': 'hidden'}.get(state, state)
    viewer.window_control = SimpleNamespace(shows_windows=lambda: shown)
    viewer.shows_windows = types.MethodType(MainViewer.shows_windows, viewer)
    return viewer


# =====================================================================
# Undo/redo stand-in Viewer and quiet runner (shared across viewer/undo tests)
# =====================================================================

UNDO_BLUE = [0.0, 0.0, 1.0, 1.0]


def make_undo_viewer():
    """A four-node MainViewer holding every field a full history state captures."""
    from EMAPSSN_Viewer import MainViewer

    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 4
    viewer.full_headers = ["n0", "n1", "n2", "n3"]
    viewer.pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32
    )
    viewer.visible_mask = np.ones(4, dtype=bool)
    viewer.current_colors = np.tile(UNDO_BLUE, (4, 1)).astype(np.float32)
    viewer.current_sizes = np.full(4, 10.0, dtype=np.float32)
    viewer.current_shapes = np.full(4, "disc", dtype=object)
    viewer.node_render_order = np.arange(4, dtype=np.int32)
    viewer.cluster_labels = None
    viewer.last_cluster_params = None
    viewer.group_labels = [set(), set(), set(), set()]
    viewer.metadata = {
        "Length": {"type": "number", "values": np.array([10, 20, 30, 40])}
    }
    viewer._cacheable_attrs = {"sidebar_buttons_to_persist", "custom_scores"}
    viewer.sidebar_buttons_to_persist = []
    viewer.custom_scores = np.array([1.0, 2.0, 3.0, 4.0])
    viewer.selected_indices = []
    viewer.position_history = []
    viewer.redo_stack = []
    viewer.console_text = SimpleNamespace(text="")
    viewer.update_selection_visual = mock.Mock()
    viewer.update_edges = mock.Mock()
    viewer.broadcast_event = mock.Mock()
    # A page is connected, so the state is built and broadcast.
    viewer.web_server = SimpleNamespace(
        queues_lock=threading.Lock(), event_queues=[object()]
    )
    return viewer


def quietly(action):
    """Run an action without its console print reaching the test log."""
    import io
    from contextlib import redirect_stdout

    with redirect_stdout(io.StringIO()):
        return action()


# =====================================================================
# Edge-layer helpers (shared by test_viewer_edges and test_low_resource_mode)
# =====================================================================

_RECORDING_EDGE_CLASSES = None


def _recording_edge_classes():
    global _RECORDING_EDGE_CLASSES
    if _RECORDING_EDGE_CLASSES is None:
        from vispy.scene.visuals import create_visual_node
        from desktop.Viewer_Edges import EdgeSegmentsVisual

        class RecordingSegmentsVisual(EdgeSegmentsVisual):
            """EdgeSegmentsVisual that also keeps what it was asked to upload."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.uploaded = {}
                self.uploaded_grades = {}

            def set_segments(self, key, edges, **kwargs):
                self.uploaded[key] = np.array(edges, copy=True)
                super().set_segments(key, edges, **kwargs)

            def _upload_grades(self, key, vertex_grades):
                self.uploaded_grades[key] = np.array(vertex_grades, copy=True)
                super()._upload_grades(key, vertex_grades)

            def drawn(self, key):
                return self.uploaded[key][: self.segment_count]

        _RECORDING_EDGE_CLASSES = (RecordingSegmentsVisual, create_visual_node(RecordingSegmentsVisual))
    return _RECORDING_EDGE_CLASSES


def random_network(seed, nodes=60, edges=400):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, nodes, (edges, 2)).astype(np.int32)
    scores = rng.random(edges).astype(np.float32)
    scores[::5] = 0.5  # ties
    positions = (rng.random((nodes, 2)) * 100).astype(np.float32)
    return pairs, scores, positions


def recording_layer(edges, scores, positions, **kwargs):
    from desktop import Viewer_Edges
    from desktop.Viewer_Edges import ViewerEdges

    _visual_cls, recording_segments = _recording_edge_classes()
    with mock.patch.object(Viewer_Edges, "EdgeSegments", recording_segments):
        return ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                           use_position_texture=True, **kwargs)


def make_edge_viewer(test_case):
    """A five-node MainViewer wired with recording_layer and default edge flags patched off."""
    import EMAPSSN_Config as cfg
    from EMAPSSN_Viewer import MainViewer

    for name in ("UMAP_MODE", "LOW_RESOURCE_DRAWING", "FEWER_EDGES_WHILE_MOVING"):
        patcher = mock.patch.object(cfg, name, False)
        patcher.start()
        test_case.addCleanup(patcher.stop)
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 5
    viewer.edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    viewer.edge_scores = np.array([0.9, 0.9, 0.1, 0.6], dtype=np.float32)
    viewer.pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
    viewer.visible_mask = np.ones(5, dtype=bool)
    viewer.current_slider_threshold = 0.5
    viewer.selected_indices = []
    viewer.is_multi_dragging = False
    viewer.edge_layer = recording_layer(viewer.edges, viewer.edge_scores, viewer.pos)
    return viewer


# =====================================================================
# Controls card / widget test helpers (shared across viewer widget tests)
# =====================================================================

MANIFOLD_NOTE = "Nodes keep their layout positions"


def shipped_actions(manifold=False):
    """The Viewer's list of actions for a physics layout, or with manifold for a UMAP or PaCMAP one,
    as its source passes it to translate()."""
    import ast

    viewer_path = Path(SRC_DIR) / "EMAPSSN_Viewer.py"
    tree = ast.parse(viewer_path.read_text(encoding="utf-8"))
    found = [node.args[1].value for node in ast.walk(tree)
             if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "translate"
                 and len(node.args) == 2 and isinstance(node.args[1], ast.Constant)
                 and str(node.args[1].value).startswith("[ENTER]"))]
    wanted = [text for text in found if (MANIFOLD_NOTE in text) == manifold]
    if len(found) != 2 or len(wanted) != 1:
        raise AssertionError("The Viewer no longer lists its actions once for each kind of layout")
    return wanted[0]


def chinese_actions(source):
    """The Chinese catalog's text for source."""
    import xml.etree.ElementTree as ElementTree

    zh_catalog = Path(SRC_DIR) / "resources" / "languages" / "zh_CN" / "emapssn_zh_CN.ts"
    for message in ElementTree.parse(zh_catalog).getroot().iter("message"):
        if message.findtext("source") == source:
            return message.findtext("translation")
    raise AssertionError("The Chinese catalog has no text for the Viewer's actions")


def flush(app):
    for _ in range(4):
        app.processEvents()


class RecordingCanvas(QtWidgets.QWidget):
    """Stands in for the VisPy canvas the card sits on: it keeps every mouse and key event it is sent.

    ``keys`` are the Qt keys pressed on it, and ``on_copy``, when set, is
    called for each Copy key press it receives, as the Viewer's command line
    copies its own text when it is open.
    """

    def __init__(self):
        from PySide6.QtCore import Qt

        super().__init__()
        self.seen = []
        self.keys = []
        self.on_copy = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _keep(self, kind, event):
        self.seen.append(kind)
        event.accept()

    def mousePressEvent(self, event):
        self._keep("press", event)

    def mouseReleaseEvent(self, event):
        self._keep("release", event)

    def mouseMoveEvent(self, event):
        self._keep("move", event)

    def mouseDoubleClickEvent(self, event):
        self._keep("double", event)

    def wheelEvent(self, event):
        self._keep("wheel", event)

    def contextMenuEvent(self, event):
        self._keep("context", event)

    def keyPressEvent(self, event):
        from PySide6.QtGui import QKeySequence

        self.keys.append(event.key())
        if self.on_copy is not None and event.matches(QKeySequence.StandardKey.Copy):
            self.on_copy()
        self._keep("key", event)
