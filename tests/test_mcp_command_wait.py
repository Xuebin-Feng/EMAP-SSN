"""wait_command_request: the Viewer-request counterpart of the pipeline's wait_job.

A fake clock stands in for time and sleeping, and a scripted portal for the Viewer, so every case
below decides exactly when the request changes, with no real waiting.
"""
import asyncio
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from EMAPSSN_MCP_Server import mcp
from mcp_server.core.Workflow_Dispatch import REGISTRY, dispatch
from mcp_server.pipeline.Pipeline_Jobs import MAX_WAIT_SECONDS
from mcp_server.viewer import Viewer_Operations as ops
from utilities.Viewer_Sessions import VIEWER_BRIDGE_SECONDS


class FakeClock:
    """Time that moves only when the code under test sleeps."""

    def __init__(self):
        self.now = 1000.0
        self.pauses = []

    def monotonic(self):
        return self.now

    async def pause(self, seconds):
        self.pauses.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def record(status='running', commands=None, request_id='r1', session_id='origin'):
    commands = commands if commands is not None else [command('select "a"', status)]
    return {'request_id': request_id, 'session_id': session_id, 'submission_id': 's', 'status': status,
            'commands': commands, 'total_commands': len(commands), 'next_offset': len(commands), 'complete': True}


def command(text, status, jobs=()):
    return {'command': text, 'command_id': text, 'status': status, 'jobs': list(jobs), 'messages': [], 'artifacts': []}


def job(job_id='2', status='running', **extra):
    return {'job_id': job_id, 'status': status, **extra}


class Script:
    """A portal whose get_command_request answers follow ``states``: (seconds since the first read, record).

    A state that is an exception is raised instead, after the time the Viewer's bridge waits before it
    answers 503 (a read of a busy Viewer takes that long to fail).
    """
    BUSY_READ_SECONDS = VIEWER_BRIDGE_SECONDS

    def __init__(self, clock, states):
        self.clock, self.states, self.start, self.calls = clock, states, None, []

    async def __call__(self, ctx, action, arguments, session_id):
        if self.start is None:
            self.start = self.clock.now
        self.calls.append((action, arguments, session_id))
        elapsed = self.clock.now - self.start
        current = self.states[0][1]
        for at, state in self.states:
            if elapsed >= at:
                current = state
        if isinstance(current, Exception):
            self.clock.now += self.BUSY_READ_SECONDS
            raise current
        return dict(current)


class Progress:
    """A context that records the progress notifications the handler sends."""

    def __init__(self, clock):
        self.clock, self.sent = clock, []

    async def report_progress(self, progress, total=None, message=None):
        self.sent.append((progress, total, message, self.clock.now))


class WaitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patch in (mock.patch.object(ops, '_now', self.clock.monotonic),
                      mock.patch.object(ops, '_pause', self.clock.pause)):
            patch.start()
            self.addCleanup(patch.stop)

    async def wait(self, states, ctx=None, **arguments):
        script = Script(self.clock, states)
        ctx = ctx or Progress(self.clock)
        with mock.patch.object(ops, '_portal_call', script):
            return await ops.wait_command_request(ctx, **{'request_id': 'r1', **arguments}), script, ctx

    async def test_a_request_that_has_ended_is_returned_at_once_in_every_ending_state(self):
        for status in ('succeeded', 'failed', 'cancelled', 'skipped', 'awaiting_user_input'):
            with self.subTest(status=status):
                result, script, ctx = await self.wait([(0, record(status))])
                self.assertEqual(result['status'], status)
                self.assertFalse(result['waiting'])
                self.assertEqual(result['waited_seconds'], 0)
                self.assertNotIn('next_step', result)
                self.assertEqual(len(script.calls), 1)
                self.assertEqual(ctx.sent, [])
        self.assertEqual(self.clock.pauses, [])

    async def test_it_waits_while_queued_or_running_and_returns_the_whole_record_when_it_ends(self):
        final = record('succeeded', [command('cluster', 'succeeded', [job('2', 'succeeded', output_path='out.txt')])])
        result, script, _ = await self.wait([(0, record('queued')), (0.5, record('running')), (6.0, final)])
        self.assertFalse(result['waiting'])
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['commands'], final['commands'])
        self.assertEqual(result['waited_seconds'], round(sum(self.clock.pauses), 2))
        self.assertGreaterEqual(result['waited_seconds'], 6.0)
        self.assertLess(result['waited_seconds'], 6.0 + max(seconds for _, seconds in ops.WAIT_POLL_SECONDS))
        self.assertTrue(all(call[0] == 'get_command_request' for call in script.calls))

    async def test_after_the_timeout_it_returns_waiting_and_says_how_to_wait_again(self):
        # 2.95 s is not a whole number of read intervals: the last sleep is cut short.
        result, script, _ = await self.wait([(0, record('running'))], timeout_seconds=2.95, session_id='viewer-9')
        self.assertTrue(result['waiting'])
        self.assertEqual(result['status'], 'running')
        self.assertEqual(result['waited_seconds'], 2.95)
        self.assertEqual(result['next_step'], {'tool': 'emapssn_viewer_data', 'action': 'wait_command_request',
                                               'arguments': {'request_id': 'r1', 'session_id': 'origin'}})
        # It never sleeps past its deadline, and the last read is at the deadline.
        self.assertAlmostEqual(sum(self.clock.pauses), 2.95)
        self.assertEqual(script.calls[-1][2], 'viewer-9')

    async def test_a_zero_timeout_reads_once(self):
        result, script, ctx = await self.wait([(0, record('running'))], timeout_seconds=0)
        self.assertTrue(result['waiting'])
        self.assertEqual((len(script.calls), self.clock.pauses, ctx.sent), (1, [], []))

    async def test_a_second_wait_finishes_what_the_first_left_running(self):
        states = [(0, record('running')), (7.0, record('succeeded'))]
        first, _, _ = await self.wait(states, timeout_seconds=3)
        self.assertTrue(first['waiting'])
        second, _, _ = await self.wait([(0, record('running')), (4.0, record('succeeded'))], timeout_seconds=30)
        self.assertEqual((second['status'], second['waiting']), ('succeeded', False))

    async def test_the_reads_are_exactly_get_command_request_with_the_callers_paging(self):
        result, script, _ = await self.wait([(0, record('succeeded'))], submission_id=None, offset=3, limit=7,
                                            command_id='c', artifact_offset=2, artifact_limit=5)
        action, arguments, session = script.calls[0]
        self.assertEqual(action, 'get_command_request')
        self.assertEqual(arguments, {'request_id': 'r1', 'submission_id': None, 'offset': 3, 'limit': 7,
                                     'command_id': 'c', 'artifact_offset': 2, 'artifact_limit': 5})
        self.assertIsNone(session)

    async def test_the_reads_get_less_frequent_as_the_wait_goes_on(self):
        _, _, _ = await self.wait([(0, record('running'))], timeout_seconds=50)
        self.assertEqual(self.clock.pauses[0], 0.1)
        self.assertAlmostEqual(self.clock.pauses[-1], 0.5, places=3)
        self.assertEqual(sorted({round(seconds, 2) for seconds in self.clock.pauses}, reverse=True), [0.5, 0.25, 0.1])
        self.assertLess(len(self.clock.pauses), 140)  # 4 reads a second for 50 s would be 200

    async def test_a_busy_viewer_does_not_end_the_wait(self):
        busy = ops.ViewerBusy('The Viewer did not answer in time.')
        states = [(0, record('running')), (1.0, busy), (13.0, record('succeeded'))]
        result, script, _ = await self.wait(states)
        self.assertEqual((result['status'], result['waiting']), ('succeeded', False))
        self.assertNotIn('viewer_busy', result)
        self.assertGreaterEqual(result['waited_seconds'], 13.0)
        self.assertGreaterEqual(len(script.calls), 5)  # the busy stretch cost reads of five seconds each

    async def test_a_viewer_still_busy_at_the_deadline_returns_the_last_record_marked_busy(self):
        busy = ops.ViewerBusy('The Viewer did not answer in time.')
        result, _, _ = await self.wait([(0, record('running')), (0.5, busy)], timeout_seconds=10)
        self.assertEqual((result['status'], result['waiting'], result['viewer_busy']), ('running', True, True))
        self.assertGreaterEqual(result['waited_seconds'], 10.0)
        self.assertEqual(result['next_step']['action'], 'wait_command_request')

    async def test_a_viewer_busy_from_the_start_is_an_error_once_the_wait_is_over(self):
        busy = ops.ViewerBusy('The Viewer did not answer in time.')
        with self.assertRaisesRegex(ToolError, 'did not answer in time') as caught:
            await self.wait([(0, busy)], timeout_seconds=9)
        self.assertIsInstance(caught.exception, ops.ViewerBusy)
        self.assertGreaterEqual(self.clock.now - 1000.0, 9.0)  # it kept asking until the deadline

    async def test_the_heartbeat_says_when_the_viewer_is_busy(self):
        busy = ops.ViewerBusy('The Viewer did not answer in time.')
        _, _, ctx = await self.wait([(0, record('running')), (0.5, busy)], timeout_seconds=25)
        self.assertTrue(ctx.sent[0][2].startswith('running: 0 of 1 commands finished'))
        self.assertTrue(any(message.endswith('(the Viewer is busy)') for _, _, message, _ in ctx.sent[1:]))

    async def test_a_submission_the_viewer_never_saw_says_it_is_safe_to_send_again(self):
        error = ToolError('Unknown submission_id in this Viewer')
        with mock.patch.object(ops, '_portal_call', side_effect=error):
            with self.assertRaisesRegex(ToolError, 'Unknown submission_id in this Viewer. It was never queued here, so '
                                                   'execute_commands with the same submission_id is safe to send again'):
                await ops.wait_command_request(Progress(self.clock), submission_id='s')
        evicted = ToolError('Submission is known but its result was evicted from this Viewer')
        with mock.patch.object(ops, '_portal_call', side_effect=evicted):
            with self.assertRaises(ToolError) as caught:
                await ops.wait_command_request(Progress(self.clock), submission_id='s')
        self.assertEqual(str(caught.exception), str(evicted))

    async def test_an_error_from_the_viewer_ends_the_wait(self):
        with mock.patch.object(ops, '_portal_call', side_effect=ToolError('Unknown submission_id in this Viewer')):
            with self.assertRaisesRegex(ToolError, 'Unknown submission_id'):
                await ops.wait_command_request(Progress(self.clock), submission_id='s')


class ProgressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = FakeClock()
        for patch in (mock.patch.object(ops, '_now', self.clock.monotonic),
                      mock.patch.object(ops, '_pause', self.clock.pause)):
            patch.start()
            self.addCleanup(patch.stop)

    async def run_wait(self, states, timeout=50):
        ctx = Progress(self.clock)
        with mock.patch.object(ops, '_portal_call', Script(self.clock, states)):
            await ops.wait_command_request(ctx, request_id='r1', timeout_seconds=timeout)
        return ctx.sent

    async def test_the_first_state_is_reported_at_once_and_a_change_follows_within_a_second(self):
        queued = record('queued', [command('cluster', 'queued')])
        running = record('running', [command('cluster', 'running', [job('4', 'running', position=1)])])
        sent = await self.run_wait([(0, queued), (0.3, running), (4.0, record('succeeded'))])
        self.assertEqual(sent[0][0], 0.0)
        self.assertIn('queued: 0 of 1 commands finished; queued: cluster', sent[0][2])
        self.assertIn('running: 0 of 1 commands finished; running: cluster (job 4 running at 1)', sent[1][2])
        self.assertGreaterEqual(sent[1][3] - sent[0][3], ops.PROGRESS_MIN_SECONDS)
        self.assertLess(sent[1][3] - sent[0][3], ops.PROGRESS_MIN_SECONDS + 0.5)

    async def test_a_jobs_progress_text_is_in_the_message(self):
        step = lambda text: record('running', [command('cluster', 'running', [job('4', 'running', progress=text)])])
        sent = await self.run_wait([(0, step('pass 1 of 3')), (2.0, step('pass 2 of 3')), (5.0, record('succeeded'))])
        self.assertTrue([m for _, _, m, _ in sent if 'pass' in m][0].endswith('(job 4 running, pass 1 of 3)'))
        self.assertTrue(any('pass 2 of 3' in message for _, _, message, _ in sent))

    async def test_changes_inside_the_minimum_interval_are_held_not_dropped(self):
        step = lambda n: record('running', [command('cluster', 'running', [job('4', 'running', progress=str(n))])])
        sent = await self.run_wait([(0, step(1)), (0.2, step(2)), (0.4, step(3)), (3.0, record('succeeded'))])
        messages = [message for _, _, message, _ in sent]
        self.assertTrue(messages[0].endswith('running, 1)'))
        self.assertTrue(messages[1].endswith('running, 3)'), messages)  # 2 was superseded before it was due
        self.assertEqual(len(messages), 2)

    async def test_a_silent_request_gets_a_heartbeat_about_every_ten_seconds(self):
        sent = await self.run_wait([(0, record('running'))], timeout=35)
        times = [stamp for _, _, _, stamp in sent]
        self.assertEqual(len(sent), 4)
        for earlier, later in zip(times, times[1:]):
            self.assertGreaterEqual(later - earlier, ops.HEARTBEAT_SECONDS)
            self.assertLess(later - earlier, ops.HEARTBEAT_SECONDS + 0.6)
        self.assertEqual(len({message for _, _, message, _ in sent}), 1)

    async def test_progress_values_only_rise_and_total_is_the_timeout(self):
        sent = await self.run_wait([(0, record('running'))], timeout=35)
        values = [value for value, _, _, _ in sent]
        self.assertEqual(values, sorted(set(values)))
        self.assertTrue(all(total == 35 for _, total, _, _ in sent))

    async def test_nothing_is_sent_when_the_request_has_already_ended(self):
        self.assertEqual(await self.run_wait([(0, record('succeeded'))]), [])

    def test_a_long_command_and_a_nested_request_read_clearly(self):
        text = 'x' * 100
        line = ops._wait_line(record('running', [command('a', 'succeeded'), command(text, 'running'), command('c', 'queued')]))
        self.assertEqual(line, f"running: 1 of 3 commands finished; running: {'x' * 57}...")
        # Only queued commands left: the next one is named.
        self.assertEqual(ops._wait_line(record('running', [command('a', 'succeeded'), command('b', 'queued')])),
                         'running: 1 of 2 commands finished; queued: b')
        self.assertEqual(ops._wait_line(record('succeeded', [command('a', 'succeeded')])), 'succeeded: 1 of 1 commands finished')


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    """Through the real MCP server: the action is discoverable, validated, and its progress reaches the client."""

    def setUp(self):
        self.clock = FakeClock()
        for patch in (mock.patch.object(ops, '_now', self.clock.monotonic),
                      mock.patch.object(ops, '_pause', self.clock.pause)):
            patch.start()
            self.addCleanup(patch.stop)

    async def call(self, client, arguments, **kwargs):
        return await client.call_tool('emapssn_viewer_data', {'action': 'wait_command_request', 'arguments': arguments},
                                      **kwargs)

    async def test_the_client_gets_progress_only_when_it_asks_and_the_record_when_done(self):
        states = [(0, record('running')), (12.0, record('succeeded'))]
        received = []

        async def on_progress(progress, total, message):
            received.append((progress, total, message))

        async with Client(mcp) as client:
            with mock.patch.object(ops, '_portal_call', Script(self.clock, states)):
                with_token = await self.call(client, {'request_id': 'r1'}, progress_callback=on_progress)
            self.assertFalse(with_token.is_error)
            self.assertEqual(with_token.structured_content['status'], 'succeeded')
            self.assertFalse(with_token.structured_content['waiting'])
            self.assertGreaterEqual(len(received), 2)  # the start, and a heartbeat after ten seconds
            self.assertTrue(all(total == 30.0 for _, total, _ in received))
            self.clock.now = 1000.0
            received.clear()
            with mock.patch.object(ops, '_portal_call', Script(self.clock, states)):
                without_token = await self.call(client, {'request_id': 'r1'})
            self.assertEqual(without_token.structured_content['status'], 'succeeded')
            self.assertEqual(received, [])

    async def test_the_schema_bounds_the_timeout_and_requires_exactly_one_identifier(self):
        description = await dispatch('emapssn_viewer_data', 'describe', {'action': 'wait_command_request'}, None)
        timeout = description['arguments_schema']['properties']['timeout_seconds']
        self.assertEqual((timeout['default'], timeout['minimum'], timeout['maximum']), (30.0, 0, MAX_WAIT_SECONDS))
        self.assertEqual(MAX_WAIT_SECONDS, 50.0)
        self.assertEqual(description['example']['arguments']['timeout_seconds'], 30)
        with mock.patch.object(ops, '_portal_call') as transport:
            for arguments in ({}, {'request_id': 'r', 'submission_id': 's'}, {'request_id': ''},
                              {'request_id': 'r', 'timeout_seconds': 50.5}, {'request_id': 'r', 'timeout_seconds': -1},
                              {'request_id': 'r', 'timeout_seconds': 'soon'}, {'request_id': 'r', 'limit': 101},
                              {'request_id': 'r', 'poll': 1}):
                with self.subTest(arguments=arguments):
                    with self.assertRaises(ToolError):
                        await dispatch('emapssn_viewer_data', 'wait_command_request', arguments, None)
            transport.assert_not_called()

    async def test_it_is_a_read_only_action_that_executes_nothing(self):
        self.assertNotIn('wait_command_request', REGISTRY['emapssn_viewer_control'])
        self.assertIn('wait_command_request', REGISTRY['emapssn_viewer_data'])
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        self.assertTrue(tools['emapssn_viewer_data'].annotations.read_only_hint)
        calls = []

        async def portal(ctx, action, arguments, session_id):
            calls.append(action)
            return record('succeeded')

        with mock.patch.object(ops, '_portal_call', portal):
            await dispatch('emapssn_viewer_data', 'wait_command_request', {'submission_id': 's'}, Progress(self.clock))
        self.assertEqual(calls, ['get_command_request'])


class ContractTests(unittest.TestCase):
    def test_the_states_that_end_a_wait_are_the_portals_terminal_ones_and_awaiting_input(self):
        from Viewer_Command_Portal import TERMINAL
        self.assertEqual(ops.FINISHED_STATES, frozenset(TERMINAL))
        self.assertEqual(ops.REQUEST_ENDED, frozenset(TERMINAL) | {'awaiting_user_input'})

    def test_the_tool_descriptions_say_to_wait_not_to_poll(self):
        async def describe(action):
            return (await dispatch('emapssn_viewer_control' if action == 'execute_commands' else 'emapssn_viewer_data',
                                   'describe', {'action': action}, None))['description']

        execute = asyncio.run(describe('execute_commands'))
        read = asyncio.run(describe('get_command_request'))
        wait = asyncio.run(describe('wait_command_request'))
        self.assertIn("emapssn_viewer_data(action='wait_command_request')", execute)
        self.assertIn('never polling', execute)
        self.assertNotIn('Poll get_command_request', execute)
        self.assertIn("emapssn_viewer_data(action='wait_command_request')", read)
        self.assertIn('Report completion only when status is succeeded', wait)
        self.assertIn('waiting=true', wait)

    def test_execute_commands_points_at_the_wait(self):
        result = {'request_id': 'r', 'session_id': 'origin', 'status': 'queued'}

        async def run():
            with mock.patch.object(ops, '_portal_call', new_callable=mock.AsyncMock, return_value=result):
                return await ops.execute_viewer_commands(SimpleNamespace(), 'x' * 36, 'select help')

        step = asyncio.run(run())['next_step']
        self.assertEqual(step, {'tool': 'emapssn_viewer_data', 'action': 'wait_command_request',
                                'arguments': {'request_id': 'r', 'session_id': 'origin'}})
        REGISTRY[step['tool']][step['action']].model.model_validate(step['arguments'])


if __name__ == '__main__':
    unittest.main()
