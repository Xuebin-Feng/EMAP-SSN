"""Catalog metadata and read-only command follow-up contracts."""
import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests import test_viewer_command_portal as fixtures
from desktop.Command_Metadata import COMMAND_METADATA, get_command_metadata
from desktop.Viewer_Inspection import command_catalog
from Viewer_Command_Portal import ViewerCommandPortal
from mcp_server.core.Workflow_Dispatch import REGISTRY, dispatch
from mcp_server.viewer import Viewer_Operations as ops
from mcp.server.mcpserver.exceptions import ToolError


class MetadataTests(unittest.TestCase):
    def test_complete_independent_metadata_and_finite_choice_sources(self):
        catalog = command_catalog()['commands']
        self.assertEqual(set(COMMAND_METADATA), {c['command'] for c in catalog})
        for entry in catalog:
            name = entry['command']
            with self.subTest(command=name):
                self.assertTrue(entry['summary'])
                self.assertIsNone(entry['help'])
                self.assertEqual(len({a['name'] for a in entry['arguments']}), len(entry['arguments']))
                source = (Path('src/commands') / (name + '.py')).read_text(encoding='utf-8')
                literals = {n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
                for arg in entry['arguments']:
                    self.assertEqual(set(arg), {'name', 'description', 'choices'})
                    self.assertTrue(arg['description'])
                    tokens = []
                    for item in arg['choices']:
                        self.assertEqual(set(item), {'value', 'aliases'})
                        self.assertIsInstance(item['aliases'], list)
                        tokens += [item['value'], *item['aliases']]
                    self.assertEqual(len(tokens), len(set(tokens)))
                    # Finite choices must be actual parser literals, not extracted
                    # from prose help. Reset plural normalization is tested below.
                    if name != 'reset':
                        self.assertTrue(set(tokens) <= literals, (name, arg['name'], set(tokens) - literals))
        copy = get_command_metadata('reset')
        copy['arguments'][0]['choices'][0]['aliases'].append('not-a-target')
        self.assertNotIn('not-a-target', repr(command_catalog('reset')))

    def test_discovery_does_not_import_handlers(self):
        with mock.patch('importlib.import_module', side_effect=AssertionError('handler imported')):
            self.assertTrue(command_catalog()['commands'])


class LookupTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.PortalTests.setUpClass.__func__)
    setUp = fixtures.PortalTests.setUp
    finish = fixtures.PortalTests.finish

    def test_reset_choices_and_aliases_match_behavior(self):
        self.viewer.original_pos = self.viewer.pos.copy()
        for item in get_command_metadata('reset')['arguments'][0]['choices']:
            for token in [item['value'], *item['aliases']]:
                with self.subTest(token=token), redirect_stdout(io.StringIO()):
                    result = self.finish(self.portal.submit(token, 'reset ' + token)['request_id'])
                    self.assertEqual(result['status'], 'succeeded', result)
                    text = result['commands'][0]['messages'][0]['text']
                    expected = {'hide': 'hidden', 'order': 'node order'}.get(item['value'], item['value'])
                    self.assertEqual(text, 'Reset successful: ' + expected + '.')

    def test_status_output_paging_and_retry_equivalence(self):
        with redirect_stdout(io.StringIO()):
            request = self.portal.submit('submission', ['select help', 'zoom help'])
            self.finish(request['request_id'])
        for offset in (0, 1, 2):
            self.assertEqual(self.portal.get(request['request_id'], offset=offset, limit=1),
                             self.portal.get(submission_id='submission', offset=offset, limit=1))
        for stream in ('stdout', 'stderr'):
            self.portal.append_output(request['request_id'], stream, 'abcdefghijk')
            self.assertEqual(self.portal.read_output(request['request_id'], stream, 4, 4),
                             self.portal.read_output(submission_id='submission', stream=stream, offset=4, limit=4))
        self.assertEqual(self.portal.submit('submission', ['select help', 'zoom help'])['request_id'], request['request_id'])
        with self.assertRaisesRegex(ValueError, 'different payload'):
            self.portal.submit('submission', 'select help')
        self.assertEqual(len(self.portal.requests), 1)

    def test_invalid_unknown_evicted_and_isolated_lookups_never_submit(self):
        invalid = [{}, {'request_id': 'a', 'submission_id': 'b'}, {'request_id': ''},
                   {'submission_id': ' '}, {'request_id': 12}, {'submission_id': False},
                   {'request_id': []}, {'submission_id': {}}]
        for handler in (self.portal.get, self.portal.read_output):
            for arguments in invalid:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    handler(**arguments)
            with self.assertRaisesRegex(ValueError, 'Unknown submission_id'):
                handler(submission_id='unknown')
        self.assertEqual(len(self.portal.requests), 0)
        with redirect_stdout(io.StringIO()):
            request = self.portal.submit('old', 'select help')
            self.finish(request['request_id'])
        other = ViewerCommandPortal(fixtures.Viewer(self.directory.name))
        self.addCleanup(other.shutdown)
        with self.assertRaisesRegex(ValueError, 'Unknown submission_id'):
            other.get(submission_id='old')
        self.portal.history_limit = 1
        with redirect_stdout(io.StringIO()):
            self.finish(self.portal.submit('new', 'select help')['request_id'])
        for handler in (self.portal.get, self.portal.read_output):
            with self.assertRaisesRegex(ValueError, 'known but its result was evicted'):
                handler(submission_id='old')
        self.assertEqual(len(self.portal.requests), 1)


class MCPFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_validation_and_schema_before_transport(self):
        import jsonschema
        for action in ('get_command_request', 'read_command_output'):
            model = REGISTRY['emapssn_viewer_data'][action].model
            schema = model.model_json_schema()
            invalid = [{}, {'request_id': 'r', 'submission_id': 's'}, {'request_id': ''},
                       {'submission_id': ' '}, {'request_id': 1}, {'submission_id': False}]
            with mock.patch.object(ops, '_portal_call') as transport:
                for arguments in invalid:
                    with self.subTest(action=action, arguments=arguments):
                        with self.assertRaises(ToolError):
                            await dispatch('emapssn_viewer_data', action, arguments, None)
                        with self.assertRaises(jsonschema.ValidationError):
                            jsonschema.validate(arguments, schema)
                transport.assert_not_called()
            for arguments in ({'request_id': 'r'}, {'submission_id': 's'}, {'request_id': None, 'submission_id': 's'}):
                jsonschema.validate(arguments, schema)
                with mock.patch.object(ops, '_portal_call', new_callable=mock.AsyncMock, return_value={}) as transport:
                    await dispatch('emapssn_viewer_data', action, arguments, None)
                    forwarded = transport.call_args.args[2]
                    for key, value in arguments.items():
                        self.assertEqual(forwarded[key], value)

    async def test_execution_followup_uses_returned_session_on_new_and_retry(self):
        result = {'request_id': 'r', 'session_id': 'origin', 'submission_id': 's', 'status': 'queued'}
        with mock.patch.object(ops, '_portal_call', new_callable=mock.AsyncMock, return_value=result):
            for _ in range(2):
                response = await dispatch('emapssn_viewer_control', 'execute_commands',
                                          {'submission_id': 's', 'commands': 'select help'}, None)
                step = response['next_step']
                self.assertEqual(step, {'tool': 'emapssn_viewer_data', 'action': 'get_command_request',
                                        'arguments': {'request_id': 'r', 'session_id': 'origin'}})
                REGISTRY[step['tool']][step['action']].model.model_validate(step['arguments'])
                self.assertNotIn('next_step', result)
        self.assertNotIn('execute_commands', REGISTRY['emapssn_viewer_data'])

    async def test_session_selection_and_reconnect_keep_lookup_local(self):
        from mcp_server.viewer.Viewer_Client import MCPViewerClient
        # Exercise the actual client routing while keeping transport read-only.
        client = MCPViewerClient(Path.cwd())
        selected = []
        def discover(target, **kwargs):
            selected.append(target)
            return SimpleNamespace(session_id=target)
        def request(session, endpoint, payload=None):
            if endpoint.endswith('/session'):
                return {'inspection_capabilities': ['commands_v1']}
            self.assertEqual(payload['arguments'], {'submission_id': 's'})
            return {'session_id': session.session_id, 'request_id': session.session_id + '-r'}
        with mock.patch('mcp_server.viewer.Viewer_Client.select_viewer_session', side_effect=discover), \
                mock.patch.object(client, '_request', side_effect=request):
            client.connected_session_id = 'first'
            original = await client.command_action('get_command_request', {'submission_id': 's'})
            client.connected_session_id = 'second'
            other = await client.command_action('get_command_request', {'submission_id': 's'})
            explicit = await client.command_action('get_command_request', {'submission_id': 's'}, 'first')
            client.connected_session_id = None
            client.connected_session_id = 'first'
            reconnected = await client.command_action('get_command_request', {'submission_id': 's'})
        self.assertEqual(original, explicit)
        self.assertEqual(original, reconnected)
        self.assertNotEqual(original, other)
        self.assertEqual(selected, ['first', 'second', 'first', 'first'])


if __name__ == '__main__':
    unittest.main()
