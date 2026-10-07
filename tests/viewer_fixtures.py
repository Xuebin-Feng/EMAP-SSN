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
