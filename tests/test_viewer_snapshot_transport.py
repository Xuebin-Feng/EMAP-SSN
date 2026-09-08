"""Authenticated snapshot transport, including Qt/worker ownership boundaries."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mcp_server.viewer.Viewer_Inspection import ViewerInspectionService
from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerError

class SnapshotHTTPTests(unittest.TestCase):
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
    def test_auth_validation_and_immutable_readonly_data(self):
        existing_files = sorted(str(p.relative_to(self.directory.name)) for p in Path(self.directory.name).rglob('*'))
        self.assertEqual(self.request('get_summary',token=False)['status'],401)
        for action,args in [('close_session',{}),('get_summary',{'unexpected':1}),('query_nodes',{'snapshot_id':'invalid'})]:
            self.assertEqual(self.request(action,args)['status'],400)
        summary=self.request('get_summary')
        self.assertEqual(summary['status'],200,summary)
        sid=summary['payload']['snapshot_id']
        self.viewer.metadata['Org,名']['values'][0]='changed'
        result=self.request('query_nodes',{'snapshot_id':sid,'columns':['Org,名'],'max_bytes':1024})
        self.assertEqual(result['status'],200,result)
        self.assertEqual(result['payload']['rows'][0]['metadata']['Org,名'],'x')
        self.assertLessEqual(result['size'],1024)
        subset=self.request('create_subset',{'snapshot_id':sid,'scope':'all','expression':'"a"'})
        self.assertEqual(subset['status'],200,subset)
        self.assertEqual(subset['payload']['matched_count'],1)
        self.assertEqual(self.viewer.selected_indices,[])
        self.assertEqual(self.request('read_value',{'snapshot_id':sid,'index':0,'field':'metadata','column':'Org,名'})['payload']['text'],'"x"')
        self.assertEqual(self.request('create_subset',{'snapshot_id':sid,'scope':'all','expression':'@file@'})['status'],400)
        self.assertEqual(self.request('query_nodes', {'snapshot_id':sid, 'cursor':'malformed'})['status'],400)
        self.assertEqual(sorted(str(p.relative_to(self.directory.name)) for p in Path(self.directory.name).rglob('*')), existing_files)
    def test_capture_on_qt_and_aggregation_off_qt(self):
        service=self.viewer.viewer_inspection; events=[]
        capture=service.capture_snapshot; execute=service.snapshots.execute
        def capture_checked(): events.append(('capture',threading.get_ident())); return capture()
        def execute_checked(*args,**kwargs): events.append(('execute',threading.get_ident())); return execute(*args,**kwargs)
        with mock.patch.object(service,'capture_snapshot',side_effect=capture_checked), mock.patch.object(service.snapshots,'execute',side_effect=execute_checked):
            result=self.request('get_summary')
            self.assertEqual(result['status'],200,result)
            result=self.request('summarize_subset',{'snapshot_id':result['payload']['snapshot_id'],'columns':['Org,名']})
            self.assertEqual(result['status'],200,result)
        self.assertEqual(events[0],('capture',threading.get_ident()))
        self.assertTrue(all(t!=threading.get_ident() for kind,t in events if kind=='execute'))

class SnapshotClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_older_viewer_requires_restart_without_changing_selection(self):
        client=MCPViewerClient(); client.connected_session_id='keep'
        with mock.patch('mcp_server.viewer.Viewer_Client.select_viewer_session',return_value=SimpleNamespace()), mock.patch.object(client,'_request',return_value={}) as request:
            with self.assertRaisesRegex(MCPViewerError,'restart'): await client.get_summary('explicit')
            self.assertEqual(request.call_count,1)
        self.assertEqual(client.connected_session_id,'keep')
    async def test_session_pages_no_snapshot_or_secrets(self):
        # A live selection outside the returned page must remain selected.
        client=MCPViewerClient(); client.connected_session_id='019'
        sessions=[SimpleNamespace(session_id=str(i).zfill(3),pid=i,started_at='now',token='secret') for i in range(20)]
        with mock.patch('mcp_server.viewer.Viewer_Client.discover_viewer_sessions',return_value=sessions), mock.patch.object(client,'_request',return_value={'inputs':{},'inspection_capabilities':['snapshots_v1']}) as request:
            first=await client.list_sessions(limit=5,max_bytes=1024)
            self.assertLessEqual(len(json.dumps(first,ensure_ascii=False,separators=(',',':')).encode()),1024)
            second=await client.list_sessions(offset=first['next_offset'],limit=5,max_bytes=1024)
            self.assertNotEqual(first['sessions'][0]['session_id'],second['sessions'][0]['session_id'])
            self.assertTrue(all(call.args[1]=='/api/mcp/v1/session' for call in request.call_args_list))
        self.assertEqual(client.connected_session_id,'019')
        self.assertNotIn('secret',json.dumps(first))
