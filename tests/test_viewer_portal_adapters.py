"""Real HTTP/Qt portal transport and the web agent's shared result path."""
import asyncio
import base64
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tests.test_viewer_snapshot_transport import SnapshotHTTPTests
from tests.test_viewer_command_portal import PortalTests
from Viewer_Command_Portal import get_portal


class PortalHTTPTests(SnapshotHTTPTests):
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

    # Inherited snapshot-specific tests use their original route.
    def test_auth_validation_and_immutable_readonly_data(self):
        self.url = self.url.replace('/commands', '/data')
        super().test_auth_validation_and_immutable_readonly_data()

    def test_capture_on_qt_and_aggregation_off_qt(self):
        self.url = self.url.replace('/commands', '/data')
        super().test_capture_on_qt_and_aggregation_off_qt()

    def test_catalog_help_requires_no_queued_command(self):
        portal = get_portal(self.viewer)
        before = len(portal.requests)
        general = self.request('get_command_catalog', {})['payload']
        detailed = self.request('get_command_catalog', {'command': 'reset'})['payload']
        self.assertTrue(all(entry['help'] is None for entry in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['commands'][0]['syntax'])
        self.assertIn('order, layer', detailed['commands'][0]['help'])
        self.assertEqual(len(portal.requests), before)

    def test_command_http_auth_retry_results_capture_and_catalog(self):
        args = {'submission_id': 'http-retry', 'commands': ['zoom help', 'unknown_command', 'zoom help']}
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


class AgentAdapterTests(PortalTests):
    def test_web_agent_waits_for_portal_and_uses_actual_outcomes(self):
        from web_ui import agent_backend as agent
        v = self.viewer
        v._agent_generation = 1
        v._agent_busy = True
        with mock.patch.object(agent, 'start_refinement_worker') as refine:
            agent.on_web_worker_finished(v, 'test', 'Command: select "one"\nCommand: unknown_command\nCommand: select "two"', '', '{}', '', 1)
            refine.assert_not_called()
            result = self.finish(v._agent_request_id)
            self.assertEqual(result['status'], 'failed')
            refine.assert_called_once()
            feedback = json.loads(refine.call_args.args[4])
            self.assertEqual(feedback['result']['status'], 'failed')
            self.assertEqual(v.selected_indices, [0])

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
