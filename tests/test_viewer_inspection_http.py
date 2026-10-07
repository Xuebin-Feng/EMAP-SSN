"""Viewer inspection over HTTP (web_ui.Web_Server /api/mcp/v1/data) and the MCP client side.

Covers authentication, request framing and early rejections, snapshot isolation
and Qt/worker ownership on the server, and MCPViewerClient's capability checks.
"""
import asyncio
import http.client
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
from desktop.Viewer_Inspection import ViewerInspectionService
from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerError
from tests.sparse_alignment import sparse_alignment
from tests.viewer_fixtures import SnapshotHTTPFixture

class SnapshotHTTPTests(SnapshotHTTPFixture, unittest.TestCase):
    def test_alignment_capture_distribution_and_projection_over_http(self):
        self.url = self.url.replace('/commands', '/data')
        self.viewer.alignment = SimpleNamespace(
            aln=sparse_alignment((f'row{index}', 'A') for index in range(len(self.viewer.full_headers))),
            viewer_to_aln=np.arange(len(self.viewer.full_headers)),
            label_to_col={'1883':0}, col_to_label={0:'1883'},
            resolved_ref_full='ref', msa_file='fixture.fasta')
        captured = self.request('get_summary', {'include_alignment': True})
        self.assertEqual(captured['status'], 200, captured)
        sid = captured['payload']['snapshot_id']
        result = self.request('get_residue_distribution', {'snapshot_id':sid, 'positions':['1883']})
        self.assertEqual(result['status'], 200, result)
        self.assertEqual(result['payload']['rows'][0]['residues'][0]['fraction'], 1)
        page = self.request('query_nodes', {'snapshot_id':sid, 'fields':['node_id']})
        self.assertEqual(page['status'], 200, page)
        self.assertTrue(all(set(row)=={'node_id'} for row in page['payload']['rows']))
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
    def test_early_rejections_drain_request_body(self):
        # Replying before reading the body made the server reset (RST) the
        # connection once the body arrived, and Windows then dropped the 401:
        # ConnectionAbortedError [WinError 10053]. Sending the body after a
        # pause, by which time such a reply has gone out, reproduces it every time.
        denied = {'error': 'Missing or invalid Viewer inspection token.'}
        endpoints = [('GET', '/api/mcp/v1/session', 'Bearer'), ('POST', '/api/mcp/v1/data', None),
                     ('POST', '/api/mcp/v1/commands', None), ('POST', '/api/mcp/v1/shutdown', 'Bearer')]
        for method, path, challenge in endpoints:
            for size in (1, 65536):
                with self.subTest(method=method, path=path, size=size):
                    status, headers, payload = self.exchange(
                        method, path, {'Content-Length': str(size)}, b'x' * size, pause=0.05)
                    self.assertEqual((status, payload, headers['WWW-Authenticate'], headers['Connection']),
                                     (401, denied, challenge, None))
        for _ in range(10):  # the urllib request that failed under full-suite load
            self.assertEqual(self.request('get_summary', {'padding': 'x' * 9000}, token=False)['status'], 401)
    def test_undrainable_body_replies_close_connection(self):
        # Chunked, invalid and oversized bodies stay unread, so the reply asks
        # the client to close. No body bytes are sent here, so every reply
        # arrives intact, and its status and payload are unchanged.
        denied = {'error': 'Missing or invalid Viewer inspection token.'}
        size_error = {'error': 'Inspection request must be 1..65536 bytes'}
        token = {'Authorization': 'Bearer ' + self.server.inspection_token}
        cases = [
            ('/api/mcp/v1/data', {}, {'Content-Length': '65537'}, 401, denied, 'close'),
            ('/api/mcp/v1/shutdown', {}, {'Content-Length': '-1'}, 401, denied, 'close'),
            ('/api/mcp/v1/data', token, {'Content-Length': '65537'}, 400, size_error, 'close'),
            ('/api/mcp/v1/data', token, {'Content-Length': '0'}, 400, size_error, None),
            ('/api/mcp/v1/commands', token, {'Content-Length': 'abc'}, 400,
             {'error': "invalid literal for int() with base 10: 'abc'"}, 'close'),
            ('/api/mcp/v1/commands', token, {'Transfer-Encoding': 'chunked'}, 400,
             {'error': 'Command request must be 1..65536 bytes'}, 'close'),
            ('/api/agent/image', {}, {'Content-Length': str(20 * 1024 * 1024 + 1)}, 413,
             {'error': 'Each image must be nonempty and at most 20 MiB.'}, 'close'),
        ]
        for path, auth, framing, status, payload, connection in cases:
            with self.subTest(path=path, authorized=bool(auth), framing=framing):
                reply = self.exchange('POST', path, {**auth, **framing})
                self.assertEqual((reply[0], reply[2], reply[1]['Connection']), (status, payload, connection))
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
    async def test_attached_log_error_directs_to_command_output(self):
        client = MCPViewerClient()
        with mock.patch('mcp_server.viewer.Viewer_Client.select_viewer_session', return_value=SimpleNamespace(launch_id=None)):
            with self.assertRaisesRegex(MCPViewerError, 'read_command_output'):
                await client.read_log('attached')
    async def test_scientific_capabilities_require_upgrade(self):
        client = MCPViewerClient()
        with mock.patch('mcp_server.viewer.Viewer_Client.select_viewer_session', return_value=SimpleNamespace()), mock.patch.object(client, '_request', return_value={'inspection_capabilities':['snapshots_v1']}) as request:
            for action, args in [('get_summary', {'include_alignment':True}), ('query_nodes', {'fields':['node_id']}), ('get_residue_distribution', {'positions':['1']})]:
                with self.assertRaisesRegex(MCPViewerError, 'upgrade and restart'):
                    await client.inspect_data(action, args, 'explicit')
            self.assertEqual(request.call_count, 3)
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
