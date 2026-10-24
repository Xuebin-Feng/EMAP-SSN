"""Real HTTP/Qt portal transport and the web agent's shared result path."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tests.viewer_fixtures import PortalFixture, SnapshotHTTPFixture
from Viewer_Command_Portal import get_portal


class PortalHTTPTests(SnapshotHTTPFixture, unittest.TestCase):
    def test_attached_viewer_query_output_without_launch_capture(self):
        from tests.sparse_alignment import load_manager, write_fasta
        msa = str(Path(self.directory.name) / 'query.fasta')
        write_fasta(msa, [(h, 'AC') for h in self.viewer.full_headers])
        self.viewer.alignment = load_manager(msa, self.viewer.full_headers, self.viewer.full_headers[0])
        self.viewer.active_reference = self.viewer.full_headers[0]
        result = self.request('execute_commands', {'submission_id':'query-output-0123456789abcdef0123456789', 'commands':'query [1]'})
        self.assertEqual(result['status'], 200, result)
        self.app.processEvents()
        output = self.request('read_command_output', {'submission_id':'query-output-0123456789abcdef0123456789'})
        self.assertEqual(output['status'], 200, output)
        self.assertIn('100.0%', output['payload']['text'])
        self.assertIn('Pos 1', output['payload']['text'])
    def setUp(self):
        super().setUp()
        self.url = self.url.replace('/data', '/commands')
        v = self.viewer
        v.command_history = []
        v.history_file = self.directory.name + '/history.txt'
        v.console_text = SimpleNamespace(text='')
        v.broadcast_metadata_state = lambda: None
        v.broadcast_event = lambda event: None
        v.canvas = SimpleNamespace(render=lambda: np.zeros((30, 40, 4), dtype=np.uint8))

    def test_catalog_help_requires_no_queued_command(self):
        portal = get_portal(self.viewer)
        before = len(portal.requests)
        general = self.request('get_command_catalog', {})['payload']
        detailed = self.request('get_command_catalog', {'command': 'reset'})['payload']
        self.assertTrue(all(entry['help'] is None for entry in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['commands'][0]['syntax'])
        self.assertIn('order, layer', detailed['commands'][0]['help'])
        self.assertIn('summary', detailed['commands'][0])
        self.assertIn('targets', [a['name'] for a in detailed['commands'][0]['arguments']])
        self.assertEqual(len(portal.requests), before)

    def test_submission_ids_are_checked_and_a_conflict_is_named_over_http(self):
        short = self.request('execute_commands', {'submission_id': 'task-1', 'commands': ['zoom help']})
        self.assertEqual(short['status'], 400, short)
        self.assertIn('new random UUID', short['payload']['error'])
        self.assertEqual(len(getattr(get_portal(self.viewer), 'requests', {})), 0)
        sid = 'two-clients-one-id-0123456789abcdef01'
        first = self.request('execute_commands', {'submission_id': sid, 'commands': ['zoom help']})
        self.assertEqual(first['status'], 200, first)
        clash = self.request('execute_commands', {'submission_id': sid, 'commands': ['select help']})
        self.assertEqual(clash['status'], 400, clash)
        self.assertIn(f"request {first['payload']['request_id']}", clash['payload']['error'])
        self.assertIn('different payload', clash['payload']['error'])
        self.app.processEvents()
        self.assertEqual(len(get_portal(self.viewer).requests), 1)

    def test_identifier_lookup_over_http(self):
        request = self.request('execute_commands', {'submission_id': 'lookup' + '-0' * 16, 'commands': ['zoom help']})['payload']
        self.app.processEvents()
        for action in ('get_command_request', 'read_command_output'):
            by_request = self.request(action, {'request_id': request['request_id']})
            by_submission = self.request(action, {'submission_id': 'lookup' + '-0' * 16})
            self.assertEqual(by_request['payload'], by_submission['payload'])
            for args in ({}, {'submission_id': ''}, {'submission_id': 12},
                         {'submission_id': 'lookup' + '-0' * 16, 'request_id': request['request_id']}):
                self.assertEqual(self.request(action, args)['status'], 400)

    def test_command_http_auth_retry_results_capture_and_catalog(self):
        args = {'submission_id': 'http-retry-0123456789abcdef0123456789', 'commands': ['zoom help', 'unknown_command', 'zoom help']}
        self.assertEqual(self.request('execute_commands', args, token=False)['status'], 401)
        response = self.request('execute_commands', args)
        self.assertEqual(response['status'], 200, response)
        request_id = response['payload']['request_id']
        self.app.processEvents()
        result = self.request('get_command_request', {'request_id': request_id})['payload']
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual([c['status'] for c in result['commands']], ['succeeded', 'failed', 'skipped'])
        self.assertEqual(self.request('execute_commands', args)['payload']['request_id'], request_id)
        self.assertEqual(self.request('execute_commands', args | {'commands': ['zoom help']})['status'], 400)
        self.assertEqual(self.request('execute_commands', args | {'source': 'manual'})['status'], 400)
        self.assertIn('Usage: zoom', self.request('read_command_output', {'request_id': request_id})['payload']['text'])
        catalog = self.request('get_command_catalog', {'command': 'select'})['payload']
        self.assertIn('select', catalog['commands'][0]['help'])
        capture = self.request('capture_view', {'request_id': request_id})['payload']
        self.assertEqual(base64.b64decode(capture['image_base64'])[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(capture['width'], 40)


class BusyViewerTests(SnapshotHTTPFixture, unittest.TestCase):
    """The GUI thread is busy while an MCP client submits: the Viewer's 503 reaches the client, and the retry is safe.

    This test thread plays the GUI thread and answers no inspection call until it processes events, as a
    Viewer does while it computes. The bridge's wait is made instant, so nothing sleeps; the limit it was
    given is recorded instead.
    """

    def setUp(self):
        super().setUp()
        import threading
        from web_ui import Web_Server
        v = self.viewer
        v.command_history = []
        v.history_file = self.directory.name + '/history.txt'
        v.console_text = SimpleNamespace(text='')
        v.broadcast_metadata_state = lambda: None
        v.broadcast_event = lambda event: None
        v.canvas = SimpleNamespace(render=lambda: np.zeros((30, 40, 4), dtype=np.uint8))
        self.limits = []
        limits = self.limits

        class GiveUpAtOnce(threading.Event):
            def wait(self, timeout=None):
                limits.append(timeout)
                return super().wait(0)

        class BusyCall(Web_Server._InspectionCall):
            def __init__(self, method_name, arguments):
                super().__init__(method_name, arguments)
                self.event = GiveUpAtOnce()

        self.busy_patch = mock.patch.object(Web_Server, '_InspectionCall', BusyCall)
        self.busy_patch.start()
        self.addCleanup(self.busy_patch.stop)
        # A command queued while the GUI was busy runs on a later test's event loop unless the
        # portal is shut down; it would write its history into the folder this test removed.
        self.addCleanup(lambda: getattr(self.viewer, 'command_portal', None) is None or self.viewer.command_portal.shutdown())
        self.session = SimpleNamespace(base_url=f'http://127.0.0.1:{self.server.server_address[1]}',
                                       token=self.server.inspection_token)

    def submit(self, submission_id, commands=('zoom help',)):
        """What an MCP client sends, from a thread, while this one stays busy; return the outcome."""
        from mcp_server.viewer.Viewer_Client import MCPViewerClient
        client, outcome = MCPViewerClient(), {}

        def work():
            try:
                outcome['result'] = client._request(self.session, '/api/mcp/v1/commands', {
                    'action': 'execute_commands',
                    'arguments': {'submission_id': submission_id, 'commands': list(commands)}})
            except Exception as error:
                outcome['error'] = error

        thread = threading.Thread(target=work)
        thread.start()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        return outcome

    def test_a_busy_gui_answers_503_with_the_retry_advice_and_the_command_is_queued_once(self):
        from utilities.Viewer_Sessions import VIEWER_BRIDGE_SECONDS, VIEWER_REQUEST_SECONDS
        from mcp_server.viewer.Viewer_Client import MCPViewerTimeout
        sid = 'busy-' + 'a' * 32
        busy = self.submit(sid)
        error = busy['error']
        self.assertIsInstance(error, MCPViewerTimeout)
        self.assertIn('SAME submission_id', str(error))
        self.assertIn('wait_command_request', str(error))
        self.assertNotIn('Could not inspect the Viewer', str(error))
        # The bridge gave up at its own limit, which is shorter than the client's.
        self.assertEqual(self.limits, [VIEWER_BRIDGE_SECONDS])
        self.assertLess(self.limits[0], VIEWER_REQUEST_SECONDS)
        self.assertFalse(hasattr(self.viewer, 'command_portal'))  # the GUI thread has not run it yet
        # The GUI thread frees up: the command that timed out is queued, and runs once.
        self.app.processEvents()
        portal = get_portal(self.viewer)
        self.assertEqual(len(portal.requests), 1)
        first = next(iter(portal.requests.values()))
        self.assertEqual(first['submission_id'], sid)
        # The client does what the message says, now that the GUI thread answers: the SAME submission_id
        # returns the request that is already there.
        self.busy_patch.stop()
        self.url = self.url.replace('/data', '/commands')
        retry = self.request('execute_commands', {'submission_id': sid, 'commands': ['zoom help']})
        self.assertEqual(retry['status'], 200, retry)
        self.assertEqual(retry['payload']['request_id'], first['request_id'])
        self.assertEqual(len(portal.requests), 1)

    def test_a_new_submission_id_after_a_timeout_would_run_the_command_twice(self):
        """Why the message insists on the SAME id: the timed-out call is queued, so a new id adds a second request."""
        self.assertIn('error', self.submit('first-' + 'b' * 32))
        self.assertIn('error', self.submit('second-' + 'b' * 32))
        self.app.processEvents()
        self.assertEqual(len(get_portal(self.viewer).requests), 2)

    def test_every_gui_bound_route_gives_up_at_the_bridge_limit(self):
        from utilities.Viewer_Sessions import VIEWER_BRIDGE_SECONDS
        from mcp_server.viewer.Viewer_Client import MCPViewerClient, MCPViewerTimeout
        client = MCPViewerClient()
        for endpoint, body in (('/api/mcp/v1/summary', None),
                               ('/api/mcp/v1/data', {'action': 'get_summary', 'arguments': {}}),
                               ('/api/mcp/v1/commands', {'action': 'get_command_request', 'arguments': {'submission_id': 'x'}}),
                               ('/api/mcp/v1/window', {'action': 'show'})):
            with self.subTest(endpoint=endpoint):
                self.limits.clear()
                outcome = {}

                def work():
                    try:
                        client._request(self.session, endpoint, body)
                    except Exception as error:
                        outcome['error'] = error

                thread = threading.Thread(target=work)
                thread.start()
                thread.join(10)
                self.assertFalse(thread.is_alive())
                self.assertIsInstance(outcome['error'], MCPViewerTimeout)
                self.assertEqual(self.limits, [VIEWER_BRIDGE_SECONDS])
        self.app.processEvents()


class AgentAdapterTests(PortalFixture, unittest.TestCase):
    def test_web_agent_waits_for_portal_and_uses_actual_outcomes(self):
        from web_ui import agent_backend as agent
        v = self.viewer
        v._agent_generation = 1
        v._agent_busy = True
        with mock.patch.object(agent, 'start_refinement_worker') as refine:
            # A command the catalog does not know would wait for approval.
            agent.on_web_worker_finished(v, 'test', 'Command: select "one"\nCommand: select {Missing=1}\nCommand: select "two"', '', '{}', '', 1)
            refine.assert_not_called()
            result = self.finish(v._agent_request_id)
            self.assertEqual(result['status'], 'failed')
            refine.assert_called_once()
            feedback = json.loads(refine.call_args.args[4])
            self.assertEqual(feedback['result']['status'], 'failed')
            self.assertEqual(v.selected_indices, [0])

    def test_a_command_that_writes_waits_for_the_agent_page(self):
        from web_ui import agent_backend as agent
        v = self.viewer
        v._agent_generation = 1
        with mock.patch.object(agent, 'start_refinement_worker') as refine:
            agent.on_web_worker_finished(v, 'test', 'Command: select "one"\nCommand: save mine.h5', '', '{}', '', 1)
            request_id = v._agent_request_id
            for _ in range(200):
                if self.portal.get(request_id)['status'] == 'awaiting_user_input':
                    break
                self.app.processEvents()
            waiting = self.portal.get(request_id)['commands'][1]
            self.assertEqual((waiting['command'], waiting['status'], waiting['approval']),
                             ('save mine.h5', 'awaiting_user_input', 'pending'))
            refine.assert_not_called()
            agent.handle_decide_agent_command(v, {'request_id': request_id, 'command_id': waiting['command_id'], 'run': False})
            self.assertEqual(self.finish(request_id)['status'], 'cancelled')
            refine.assert_called_once()
            self.assertEqual(json.loads(refine.call_args.args[4])['result']['commands'][1]['status'], 'cancelled')

    def test_stale_agent_callback_cannot_submit(self):
        from web_ui import agent_backend as agent
        self.viewer._agent_generation = 2
        agent.on_web_worker_finished(self.viewer, 'test', 'Command: select "one"', '', '{}', '', 1)
        self.assertEqual(len(self.portal.requests), 0)

    def test_reset_does_not_cancel_submitted_request(self):
        from web_ui import agent_backend as agent
        self.viewer._agent_generation = 1
        with mock.patch.object(agent, 'start_refinement_worker') as refine:
            agent.on_web_worker_finished(self.viewer, 'test', 'Command: select "one"', '', '{}', '', 1)
            request_id = self.viewer._agent_request_id
            agent._invalidate_agent_turn(self.viewer)
            self.assertEqual(self.finish(request_id)['status'], 'succeeded')
            refine.assert_not_called()


class NativeImageTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_mcp_image_and_structured_metadata(self):
        import EMAPSSN_MCP_Server as server
        from mcp.types import CallToolResult
        with mock.patch.object(server, 'dispatch', new=mock.AsyncMock(return_value={'image_base64': 'aGVsbG8=', 'width': 40, 'height': 30})):
            result = await server.emapssn_viewer_data(None, 'capture_view', {})
        self.assertIsInstance(result, CallToolResult)
        self.assertEqual(result.content[1].type, 'image')
        self.assertEqual(result.structured_content['width'], 40)


if __name__ == '__main__': unittest.main()
