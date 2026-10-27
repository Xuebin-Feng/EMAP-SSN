"""Viewer command portal (Viewer_Command_Portal) and command worker tracking (Viewer_Worker_Tracking).

Covers equivalence with manual commands, completion, isolation, retries and
lookups, bounded feedback, submission limits, shutdown, the review of commands
from Agent replies, and failures of detached workers and command scripts.
"""
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6 import QtWidgets
import Command_Engine as ce
from Viewer_Command_Portal import ViewerCommandPortal, CURRENT, bind, user_interaction
from tests.command_fixtures import end_scheduler, scratch_commands
from tests.viewer_fixtures import NO_WINDOW_STATES, WINDOW_STATES, PortalFixture, Viewer, apply_window_state
from Background_Job_Scheduler import BackgroundJobScheduler
from Viewer_Visual_State import edge_stages, capture_view
from desktop.Command_Metadata import get_command_metadata


def fake_test_command(run, command='test'):
    """Resolve ``command`` (by default `test`) to a module whose run() is ``run``.

    The stand-in is the real command's module with its run() replaced, so it
    keeps what the module declares about how it is dispatched (a stop command
    stays one); a command with no module of its own has only run(). Every other
    import_module call, including the real command modules the Command_Engine
    imports for the commands that follow, is left unchanged.
    """
    real_import = importlib.import_module

    def import_module(name, package=None):
        if name == f'commands.{command}':
            try:
                real = real_import(name, package)
            except ImportError:
                return SimpleNamespace(run=run)
            # Without __spec__ and the other dunders: the engine reloads a module that has a spec.
            kept = {key: value for key, value in vars(real).items() if not key.startswith('__')}
            return SimpleNamespace(**{**kept, 'run': run})
        return real_import(name, package)
    return mock.patch('importlib.import_module', side_effect=import_module)


def worker_tracker(context, root):
    """A WorkerTracker whose status files live under ``root``; its poll timer is stopped.

    WorkerTracker derives its folder from its own file (<project>/temp/command_workers).
    """
    import Viewer_Worker_Tracking
    with mock.patch.object(Viewer_Worker_Tracking, '__file__', str(Path(root) / 'src' / 'Viewer_Worker_Tracking.py')):
        tracker = Viewer_Worker_Tracking.WorkerTracker(context)
    tracker.timer.stop()
    return tracker



class PortalTests(PortalFixture, unittest.TestCase):

    def test_manual_and_portal_selection_color_hide(self):
        manual = Viewer(self.directory.name)
        commands = ['select "one"', 'color "one" red', 'hide "two"']
        for command in commands: manual.process_command(command)
        request = self.portal.submit('equivalence', commands)
        result = self.finish(request['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual(manual.selected_indices, self.viewer.selected_indices)
        np.testing.assert_array_equal(manual.visible_mask, self.viewer.visible_mask)
        np.testing.assert_array_equal(manual.current_colors, self.viewer.current_colors)

    def test_explicit_failures_stop_batch_and_retry(self):
        r = self.portal.submit('retry', ['select', 'select "one"'])
        self.assertEqual(self.portal.submit('retry', ['select', 'select "one"'])['request_id'], r['request_id'])
        with self.assertRaises(ValueError): self.portal.submit('retry', ['select "two"'])
        result = self.finish(r['request_id'])
        self.assertEqual([c['status'] for c in result['commands']], ['failed', 'skipped'])
        self.assertEqual(self.viewer.selected_indices, [])
        for command in ['not_a_command', 'agent "nested"', 'select {Missing=1}', 'cluster topology nonsense']:
            result = self.finish(self.portal.submit(command, command)['request_id'])
            self.assertEqual(result['status'], 'failed', result)

    def test_no_match_and_help_are_successful(self):
        for command in ['select "absent"', 'select help']:
            result = self.finish(self.portal.submit(command, command)['request_id'])
            self.assertEqual(result['status'], 'succeeded', result)

    def test_wait_for_jobs_manual_input_and_stop_on_failure(self):
        scheduler = BackgroundJobScheduler(self.viewer)
        self.addCleanup(end_scheduler, scheduler)
        release = threading.Event()
        self.addCleanup(release.set)
        def worker(payload):
            release.wait(2)
            print('attributed-worker-output')
            raise ValueError('expected child failure')
        def run(viewer, args):
            scheduler.enqueue('test', 'test', None, worker, str(Path(self.directory.name) / 'out'))
            ce.command_succeeded(viewer)
        with fake_test_command(run):
            r = self.portal.submit('background', ['test', 'select "two"'])
            self.portal.pump()
        self.assertEqual(self.portal.get(r['request_id'])['status'], 'running')
        self.viewer.process_command('select "one"')
        self.assertEqual(self.viewer.selected_indices, [0])
        release.set()
        result = self.finish(r['request_id'])
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['commands'][1]['status'], 'skipped')
        text = self.portal.read_output(r['request_id'])['text']
        self.assertIn('attributed-worker-output', text)
        self.assertNotIn('Selected', text)

    def test_nested_children_and_model_recursion(self):
        def run(viewer, args):
            CURRENT.get().children(['select "one"', 'agent "recursive"', 'select "two"'])
            ce.command_succeeded(viewer)
        with fake_test_command(run):
            r = self.portal.submit('nested', 'test')
            # Only dispatch the parent under the mock.
            record = self.portal.requests[r['request_id']]['commands'][0]
            from Viewer_Command_Portal import ExecutionContext
            record.update(dispatched=True, status='running')
            with bind(ExecutionContext(self.portal, r['request_id'], record)): run(self.viewer, [])
        result = self.finish(r['request_id'])
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(self.viewer.selected_indices, [0])
        self.assertEqual(result['commands'][-1]['status'], 'skipped')

    def test_output_bounds_and_evicted_submission_not_reexecuted(self):
        r = self.portal.submit('old', 'select help')
        self.finish(r['request_id'])
        self.portal.max_output_bytes = 16
        self.portal.append_output(r['request_id'], 'stdout', 'x' * 100)
        page = self.portal.read_output(r['request_id'])
        self.assertTrue(page['truncated'])
        self.assertLessEqual(len(page['text'].encode()), 16)
        self.portal.history_limit = 1
        self.finish(self.portal.submit('new', 'select help')['request_id'])
        with self.assertRaisesRegex(ValueError, 'already executed'):
            self.portal.submit('old', 'select help')

    def test_handler_without_outcome_never_claims_success(self):
        with fake_test_command(lambda v,a: None):
            r = self.portal.submit('silent', 'test')
            self.portal.pump()
        self.assertEqual(self.portal.get(r['request_id'])['status'], 'failed')

    def test_headless_dialog_fails_before_open(self):
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '1'}), mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName') as dialog:
            result = self.finish(self.portal.submit('dialog', 'alignment')['request_id'])
        self.assertEqual(result['status'], 'failed', result)
        dialog.assert_not_called()

    def test_visual_counts_and_capture_preserve_state(self):
        cfg = SimpleNamespace(UMAP_MODE=True)
        self.viewer.selected_indices = [0]
        edges, counts = edge_stages(self.viewer, cfg)
        self.assertEqual(counts['rendered_edge_count'], 1)
        self.viewer.canvas = SimpleNamespace(render=lambda: np.zeros((100, 2000, 4), dtype=np.uint8))
        colors = self.viewer.current_colors.copy()
        result = capture_view(self.viewer)
        self.assertEqual(result['width'], 1600)
        self.assertEqual(result['height'], 80)
        np.testing.assert_array_equal(colors, self.viewer.current_colors)

    def test_headless_capture_failure_names_the_missing_opengl_context(self):
        # Qt's offscreen platform has no OpenGL context on Windows; say so and
        # name the remedy, but leave unrelated capture failures unchanged.
        def failing(message):
            def render():
                raise RuntimeError(message)
            return SimpleNamespace(render=render)
        no_context = 'Using glBindFramebuffer with no OpenGL context.'
        for platform, message, hinted in (('offscreen', no_context, True), ('offscreen', 'disk full', False),
                                          ('windows', no_context, False)):
            with self.subTest(platform=platform, message=message), \
                    mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': platform}), \
                    mock.patch('Viewer_Visual_State.sys', SimpleNamespace(platform='win32')):
                self.viewer.canvas = failing(message)
                with self.assertRaises(ValueError) as caught:
                    capture_view(self.viewer)
                self.assertTrue(str(caught.exception).startswith(f'Viewer canvas capture failed: {message}'))
                self.assertEqual('start a Viewer in hidden or normal mode' in str(caught.exception), hinted)

    def test_a_hidden_viewer_that_fell_back_headless_names_its_missing_display_when_capture_fails(self):
        reason = 'no display was found (DISPLAY and WAYLAND_DISPLAY are not set)'
        self.viewer.window_fallback_reason = reason
        self.viewer.headless = True
        self.viewer.canvas = SimpleNamespace(render=lambda: (_ for _ in ()).throw(RuntimeError('no OpenGL context here')))
        with mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': 'offscreen'}), self.assertRaises(ValueError) as caught:
            capture_view(self.viewer)
        self.assertTrue(str(caught.exception).endswith(f'because {reason}.'), str(caught.exception))
        self.assertNotIn('start a Viewer in hidden or normal mode', str(caught.exception))

    def test_a_linux_headless_viewer_with_no_display_names_it_whatever_the_render_error_says(self):
        # With no X server Qt's offscreen platform has no OpenGL at all; VisPy reports it as an
        # IndexError from its own framebuffer stack, not as a missing context.
        for message in ('no OpenGL context here', 'list index out of range'):
            with self.subTest(message=message):
                self.viewer.canvas = SimpleNamespace(render=lambda: (_ for _ in ()).throw(RuntimeError(message)))
                with mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': 'offscreen'}, clear=True), \
                        mock.patch('Viewer_Visual_State.sys', SimpleNamespace(platform='linux')), \
                        mock.patch('utilities.Viewer_Display.sys', SimpleNamespace(platform='linux')), \
                        self.assertRaises(ValueError) as caught:
                    capture_view(self.viewer)
                self.assertIn(f'Viewer canvas capture failed: {message}', str(caught.exception))
                self.assertTrue(str(caught.exception).endswith(
                    'because no display was found (DISPLAY and WAYLAND_DISPLAY are not set).'), str(caught.exception))
        # With a display the offscreen platform may render, and an unrelated failure keeps its own text.
        self.viewer.canvas = SimpleNamespace(render=lambda: (_ for _ in ()).throw(RuntimeError('disk full')))
        with mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': 'offscreen', 'DISPLAY': ':0'}, clear=True), \
                mock.patch('Viewer_Visual_State.sys', SimpleNamespace(platform='linux')), \
                mock.patch('utilities.Viewer_Display.sys', SimpleNamespace(platform='linux')), \
                self.assertRaises(ValueError) as caught:
            capture_view(self.viewer)
        self.assertEqual(str(caught.exception), 'Viewer canvas capture failed: disk full')

    def test_a_file_dialog_opens_only_where_a_user_is_at_the_viewer_command_or_not(self):
        from Viewer_Command_Portal import CommandInteractionError
        # Run as a portal command: the environment of this offscreen run says headless, the Viewer says otherwise.
        for state in WINDOW_STATES:
            with self.subTest(state=state, bound=True):
                apply_window_state(self.viewer, state)
                self.viewer.canvas = SimpleNamespace(native=None)
                with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '1', 'QT_QPA_PLATFORM': 'offscreen'}), \
                        mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('', '')) as dialog:
                    result = self.finish(self.portal.submit(f'state-{state}', 'alignment')['request_id'])
                if state in NO_WINDOW_STATES:
                    self.assertEqual(result['status'], 'failed', result)
                    dialog.assert_not_called()
                else:
                    self.assertEqual(result['status'], 'cancelled', result)
                    dialog.assert_called_once()
        # Not run by the portal: still the Viewer's state that decides, whatever the environment says.
        for state in WINDOW_STATES:
            with self.subTest(state=state, bound=False):
                viewer = apply_window_state(SimpleNamespace(), state)
                with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '1', 'QT_QPA_PLATFORM': 'offscreen'}):
                    if state in NO_WINDOW_STATES:
                        with self.assertRaisesRegex(CommandInteractionError, 'requires a visible Viewer'):
                            with user_interaction(viewer, 'Pick a file'):
                                self.fail('the dialog must not open')
                    else:
                        with user_interaction(viewer, 'Pick a file'):
                            pass
        # A stand-in with nothing to answer for it is windowed outside a portal command, as before.
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '1', 'QT_QPA_PLATFORM': 'offscreen'}):
            with user_interaction(SimpleNamespace(), 'Pick a file'):
                pass

    def test_an_unbound_picker_command_fails_cleanly_in_a_viewer_with_no_window(self):
        import commands.alignment as alignment_command
        for state in WINDOW_STATES:
            with self.subTest(state=state):
                viewer = apply_window_state(SimpleNamespace(canvas=SimpleNamespace(native=None),
                                                            console_text=SimpleNamespace(text='')), state)
                with mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('', '')) as dialog, \
                        mock.patch.object(ce, 'command_failed') as failed, \
                        mock.patch.object(ce, 'command_cancelled') as cancelled, \
                        mock.patch.object(ce, 'print_help'), redirect_stdout(io.StringIO()):
                    alignment_command.run(viewer, [])
                if state in NO_WINDOW_STATES:
                    dialog.assert_not_called()
                    failed.assert_called_once()
                    self.assertIn('requires a visible Viewer', str(failed.call_args.args[1]))
                    cancelled.assert_not_called()
                else:
                    dialog.assert_called_once()
                    failed.assert_not_called()
                    cancelled.assert_called_once()

    def test_a_hidden_viewer_refuses_dialogs_until_its_window_is_shown(self):
        self.viewer.window_mode = 'hidden'
        self.viewer.shows_windows = lambda: False
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '0', 'QT_QPA_PLATFORM': ''}), \
                mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName') as dialog:
            result = self.finish(self.portal.submit('hidden-dialog', 'alignment')['request_id'])
        self.assertEqual(result['status'], 'failed', result)
        self.assertIn('Show the window first', json.dumps(result))
        dialog.assert_not_called()

    def test_a_shown_hidden_viewer_opens_dialogs_even_on_the_offscreen_platform(self):
        # The Viewer answers for itself: the environment of a hidden launch may still say offscreen.
        self.viewer.shows_windows = lambda: True
        self.viewer.canvas = SimpleNamespace(native=None)
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '1', 'QT_QPA_PLATFORM': 'offscreen'}), \
                mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', return_value=('', '')) as dialog:
            result = self.finish(self.portal.submit('shown-dialog', 'alignment')['request_id'])
        self.assertEqual(result['status'], 'cancelled', result)
        dialog.assert_called_once()

    def test_visible_dialog_reports_waiting_then_cancelled(self):
        def choose(*args):
            self.assertEqual(self.portal.get(request_id)['status'], 'awaiting_user_input')
            return '', ''
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '0', 'QT_QPA_PLATFORM': ''}), mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', side_effect=choose):
            self.viewer.canvas = SimpleNamespace(native=None)
            request_id = self.portal.submit('visible-dialog', 'alignment')['request_id']
            result = self.finish(request_id)
        self.assertEqual(result['status'], 'cancelled', result)

    def test_worker_final_failed_status_keeps_partial_results(self):
        from Viewer_Command_Portal import ExecutionContext
        import psutil
        request_id = self.portal.submit('worker-status', 'test')['request_id']
        record = self.portal.requests[request_id]['commands'][0]
        record.update(dispatched=True, outcome='succeeded', status='running')
        context = ExecutionContext(self.portal, request_id, record)
        tracker = worker_tracker(context, self.directory.name)
        tracker.path.write_text(json.dumps({'pid': os.getpid(), 'created': psutil.Process().create_time(),
            'status': 'failed', 'message': 'One of two structures failed', 'artifacts': ['partial.pdb'],
            'result': {'succeeded': 1, 'total': 2}}))
        tracker.poll()
        result = self.finish(request_id)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['commands'][0]['jobs'][0]['result']['succeeded'], 1)
        self.assertEqual(len(result['commands'][0]['artifacts']), 1)

    def test_script_background_children(self):
        from Viewer_Command_Portal import ExecutionContext
        from Viewer_Worker_Tracking import ScriptTracker
        script = Path(self.directory.name) / 'commands.py'
        script.write_text('print(\'select "one"\')\nprint(\'agent "recursive"\')\n')
        request_id = self.portal.submit('script', 'run')['request_id']
        record = self.portal.requests[request_id]['commands'][0]
        record.update(dispatched=True, outcome='succeeded', status='running')
        ScriptTracker(ExecutionContext(self.portal, request_id, record), str(script))
        result = self.finish(request_id)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(self.viewer.selected_indices, [0])


class LookupTests(PortalFixture, unittest.TestCase):
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

    def test_reusing_an_id_with_another_payload_names_the_request_that_holds_it(self):
        sid = 'shared-by-two-clients-0123456789abcdef'
        with redirect_stdout(io.StringIO()):
            request = self.portal.submit(sid, 'select help')
            self.finish(request['request_id'])
        with self.assertRaises(ValueError) as caught:
            self.portal.submit(sid, 'zoom help')
        message = str(caught.exception)
        self.assertIn(repr(sid), message)
        self.assertIn('different payload', message)
        self.assertIn(f"request {request['request_id']}", message)
        self.assertIn('succeeded', message)
        self.assertIn(self.portal.requests[request['request_id']]['submitted_at'], message)
        self.assertIn('nothing was run', message)
        self.assertIn('new random UUID', message)
        # Another source is another payload too, and an identical retry still returns the request.
        with self.assertRaisesRegex(ValueError, 'different payload'):
            self.portal.submit(sid, 'select help', source='web_agent')
        self.assertEqual(self.portal.submit(sid, 'select help')['request_id'], request['request_id'])
        self.assertEqual(len(self.portal.requests), 1)

    def test_a_conflict_with_an_evicted_request_says_so(self):
        sid = 'evicted-then-reused-0123456789abcdef01'
        with redirect_stdout(io.StringIO()):
            request = self.portal.submit(sid, 'select help')
            self.finish(request['request_id'])
        del self.portal.requests[request['request_id']]
        with self.assertRaises(ValueError) as caught:
            self.portal.submit(sid, 'zoom help')
        self.assertIn(f"request {request['request_id']}, whose result has since been evicted", str(caught.exception))
        with self.assertRaisesRegex(ValueError, 'evicted'):
            self.portal.submit(sid, 'select help')

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
        other = ViewerCommandPortal(Viewer(self.directory.name))
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


class PortalLimitTests(PortalFixture, unittest.TestCase):
    """Submission limits, the queue bound, shutdown, and byte-exact output pages."""

    def test_invalid_submissions_are_rejected_before_queueing(self):
        invalid = [('empty id', '', 'select help'), ('129-character id', 'x' * 129, 'select help'),
                   ('numeric id', 7, 'select help'), ('missing id', None, 'select help'),
                   ('no commands', 'none', []), ('101 commands', 'many', ['select help'] * 101),
                   ('tuple', 'tuple', ('select help',)), ('blank command', 'blank', ['select help', '   ']),
                   ('two lines', 'lines', ['select "one"\nselect "two"']),
                   ('carriage return', 'return', ['select help\r']), ('8193 characters', 'long', ['x' * 8193]),
                   ('not text', 'number', [5])]
        for label, submission_id, commands in invalid:
            with self.subTest(label), self.assertRaises(ValueError):
                self.portal.submit(submission_id, commands)
        self.assertEqual((self.portal.requests, self.portal.queue, self.portal.submissions), ({}, [], {}))
        # The limits themselves are accepted; nothing runs, as no Qt events are processed.
        self.portal.submit('x' * 128, ['select help'] * 100)
        self.portal.submit('longest command', 'x' * 8192)
        self.assertEqual(len(self.portal.queue), 2)

    def test_full_queue_rejects_new_submissions_but_answers_retries(self):
        for index in range(100):
            self.portal.submit(f'queued-{index}', 'select help')
        with self.assertRaisesRegex(ValueError, 'queue is full'):
            self.portal.submit('one too many', 'select help')
        self.assertNotIn('one too many', self.portal.submissions)
        self.assertEqual(len(self.portal.requests), 100)
        self.assertEqual(self.portal.submit('queued-0', 'select help')['request_id'], self.portal.queue[0])

    def test_shutdown_cancels_queued_requests_and_nested_children(self):
        first = self.portal.submit('first', ['select "one"', 'select "two"'])['request_id']
        second = self.portal.submit('second', 'select "three"')['request_id']
        top, sibling = self.portal.requests[first]['commands']
        child, grandchild = self.portal.new_command('select "two"'), self.portal.new_command('select "three"')
        finished = self.portal.new_command('zoom help')
        finished.update(status='succeeded', outcome='succeeded')
        child['children'].append(grandchild)
        top['children'] += [child, finished]
        self.portal.shutdown()
        self.assertEqual(self.portal.queue, [])
        for record in (top, sibling, child, grandchild, *self.portal.requests[second]['commands']):
            self.assertEqual((record['status'], record['outcome']), ('cancelled', 'cancelled'), record)
            self.assertIsNotNone(record['finished_at'])
        self.assertEqual((finished['status'], finished['outcome']), ('succeeded', 'succeeded'))
        for request_id in (first, second):
            self.assertEqual(self.portal.get(request_id)['status'], 'cancelled')
            self.assertEqual([e['status'] for e in self.viewer.events if e['request_id'] == request_id][-1],
                             'cancelled')
        for submission_id, commands in (('later', 'select help'), ('first', ['select "one"', 'select "two"'])):
            with self.assertRaisesRegex(ValueError, 'shutting down'):
                self.portal.submit(submission_id, commands)
        self.app.processEvents()
        self.assertEqual(self.viewer.selected_indices, [])

    def test_output_pages_never_split_a_character(self):
        request_id = self.portal.submit('output', 'select help')['request_id']
        dna = '\U0001F9EC'  # four UTF-8 bytes
        text = 'ab\u540dc' + dna * 2 + 'end'
        self.portal.append_output(request_id, 'stdout', text)
        pages, offset = [], 0
        for _ in range(len(text.encode('utf-8'))):
            page = self.portal.read_output(request_id, offset=offset, limit=4)
            size = len(page['text'].encode('utf-8'))
            self.assertLessEqual(size, 4)
            self.assertEqual((page['offset'], page['next_offset']), (offset, offset + size))
            pages.append(page['text'])
            offset = page['next_offset']
            if page['eof']:
                break
        self.assertEqual(pages, ['ab', '\u540dc', dna, dna, 'end'])

    def test_the_colours_a_command_printed_are_not_in_the_output_a_client_reads(self):
        # The Console window shows them; an agent or an MCP client gets the text, even when a sequence is cut
        # between two writes.
        esc = chr(27)
        request_id = self.portal.submit('output', 'select help')['request_id']
        self.portal.append_output(request_id, 'stdout', f'| {esc}[38;2;1;2;3m●{esc}[0m Cluster 0\n')
        self.portal.append_output(request_id, 'stdout', f'{esc}[3')
        self.portal.append_output(request_id, 'stdout', '1mred{0}[0m and {0}[2Kmore\n'.format(esc))
        self.portal.append_output(request_id, 'stderr', f'{esc}[1mError: x{esc}[0m\n')
        page = self.portal.read_output(request_id, limit=4096)
        self.assertEqual(page['text'], '| ● Cluster 0\nred and more\n')
        self.assertEqual(self.portal.read_output(request_id, 'stderr', limit=4096)['text'], 'Error: x\n')

    def test_a_tee_over_the_capture_stream_still_gets_colours_from_commands(self):
        # Commands print to the portal's pass-through, which wraps the capture's stream: they colour for it.
        from Viewer_Command_Portal import _Tee
        from utilities import Console_Capture
        from utilities.Terminal_Colors import colours_enabled

        own = io.StringIO()
        with mock.patch.object(Console_Capture, '_capture', mock.Mock(text_streams=(own,))):
            self.assertTrue(colours_enabled(_Tee(own, 'stdout')))
            self.assertFalse(colours_enabled(_Tee(io.StringIO(), 'stdout')))

    def test_output_limit_holds_the_longest_character(self):
        # A page must fit any UTF-8 character (up to four bytes). With limit 1
        # or 2, 'a\u540db' read from offset 1 returned '' and next_offset 1 with
        # eof False, so a pager never advanced.
        request_id = self.portal.submit('output', 'select help')['request_id']
        self.portal.append_output(request_id, 'stdout', 'a\u540db')
        for limit in (0, 1, 2, 3):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'Invalid output page bounds'):
                self.portal.read_output(request_id, offset=1, limit=limit)
        page = self.portal.read_output(request_id, offset=1, limit=4)
        self.assertEqual((page['text'], page['offset'], page['next_offset'], page['eof']), ('\u540db', 1, 5, True))


class AgentReviewTests(PortalFixture, unittest.TestCase):
    """An Agent reply's commands run, wait for the user's Run or Discard, or are
    refused, as the catalog's agent policy says; other sources are not reviewed."""

    def stand_in(self, command):
        """Replace ``command`` with one that records its arguments and succeeds."""
        calls = []
        def run(viewer, args):
            calls.append(args)
            ce.command_succeeded(viewer, f'{command} stand-in ran.')
        patcher = fake_test_command(run, command)
        patcher.start()
        self.addCleanup(patcher.stop)
        return calls

    def wait_until(self, request_id, status):
        deadline = time.monotonic() + 10
        while (request := self.portal.get(request_id))['status'] != status:
            if time.monotonic() >= deadline:
                self.fail(f'The request never reached {status!r}: {request}')
            self.app.processEvents()
            time.sleep(.005)
        return request

    def test_view_commands_run_at_once(self):
        r = self.portal.submit('view', ['select "one"', 'color "one" red'], source='web_agent')
        result = self.finish(r['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual(self.viewer.selected_indices, [0])
        self.assertEqual([c['approval'] for c in result['commands']], [None, None])

    def test_a_command_needing_approval_waits_then_runs(self):
        saves = self.stand_in('save')
        r = self.portal.submit('approve', ['select "one"', 'save x.h5', 'select "two"'], source='web_agent')
        waiting = self.wait_until(r['request_id'], 'awaiting_user_input')
        self.assertEqual([(c['status'], c['approval']) for c in waiting['commands']],
                         [('succeeded', None), ('awaiting_user_input', 'pending'), ('queued', None)])
        # The Agent page shows its Run and Discard buttons from this event.
        published = [e for e in self.viewer.events if e.get('type') == 'command_request_updated']
        self.assertEqual(published[-1]['commands'][1], {
            'command_id': waiting['commands'][1]['command_id'], 'command': 'save x.h5',
            'status': 'awaiting_user_input', 'approval': 'pending'})
        for _ in range(20):
            self.app.processEvents()
            time.sleep(.005)
        self.assertEqual((saves, self.viewer.selected_indices), ([], [0]))

        self.portal.decide(r['request_id'], waiting['commands'][1]['command_id'], True)
        result = self.finish(r['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual((saves, self.viewer.selected_indices), ([['x.h5']], [1]))
        self.assertEqual(result['commands'][1]['approval'], 'approved')

    def test_a_discarded_command_is_cancelled_and_the_rest_skipped(self):
        saves = self.stand_in('save')
        r = self.portal.submit('discard', ['save x.h5', 'select "two"'], source='web_agent')
        waiting = self.wait_until(r['request_id'], 'awaiting_user_input')
        self.portal.decide(r['request_id'], waiting['commands'][0]['command_id'], False)
        result = self.finish(r['request_id'])
        self.assertEqual(result['status'], 'cancelled', result)
        self.assertEqual([c['status'] for c in result['commands']], ['cancelled', 'skipped'])
        self.assertEqual(result['commands'][0]['messages'][-1]['text'], 'Discarded in the Agent page.')
        self.assertEqual((saves, self.viewer.selected_indices), ([], []))

    def test_agent_commands_are_refused_without_running(self):
        calls = self.stand_in('agent')
        for command in ('agent off', 'agent <Local>', 'agent'):
            with self.subTest(command=command):
                result = self.finish(self.portal.submit(command, [command, 'select "one"'], source='web_agent')['request_id'])
                self.assertEqual([c['status'] for c in result['commands']], ['failed', 'skipped'])
                self.assertIn('Agent replies cannot run this command', result['commands'][0]['messages'][-1]['text'])
        self.assertEqual((calls, self.viewer.selected_indices), ([], []))

    def test_other_sources_are_not_reviewed(self):
        saves = self.stand_in('save')
        result = self.finish(self.portal.submit('mcp', 'save x.h5')['request_id'])
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual(saves, [['x.h5']])
        self.assertNotIn('approval', result['commands'][0])

    def test_only_the_command_that_is_waiting_can_be_decided(self):
        self.stand_in('save')
        r = self.portal.submit('decide', ['save a.h5', 'save b.h5'], source='web_agent')
        first, second = (c['command_id'] for c in self.wait_until(r['request_id'], 'awaiting_user_input')['commands'])
        for request_id, command_id, run in ((r['request_id'], second, True),  # not reached yet
                                            (r['request_id'], first, 'yes'),
                                            ('unknown', first, True)):
            with self.subTest(command_id=command_id, run=run), self.assertRaises(ValueError):
                self.portal.decide(request_id, command_id, run)
        self.portal.decide(r['request_id'], first, True)
        with self.assertRaises(ValueError):
            self.portal.decide(r['request_id'], first, True)

    def test_discarding_an_abandoned_turn_releases_the_queue(self):
        saves = self.stand_in('save')
        r = self.portal.submit('abandoned', ['save a.h5', 'print', 'select "one"'], source='web_agent')
        self.wait_until(r['request_id'], 'awaiting_user_input')
        later = self.portal.submit('later', 'select "two"')
        self.portal.discard_pending(r['request_id'])
        result = self.finish(r['request_id'])
        self.assertEqual([c['status'] for c in result['commands']], ['cancelled', 'cancelled', 'skipped'])
        self.assertEqual(self.finish(later['request_id'])['status'], 'succeeded')
        self.assertEqual((saves, self.viewer.selected_indices), ([], [1]))


    def test_a_command_nobody_can_approve_is_refused_at_once(self):
        # A Viewer that shows no window (headless, or hidden and not shown) with no Agent page open has
        # nobody to press Run: the command fails now, the rest of the request is skipped, and the
        # queue is not held for a decision that cannot come.
        saves = self.stand_in('save')
        silent = {'headless': lambda: setattr(self.viewer, 'headless', True),
                  'hidden, not shown': lambda: setattr(self.viewer, 'shows_windows', lambda: False)}
        for label, make_silent in silent.items():
            with self.subTest(label):
                make_silent()
                r = self.portal.submit(f'nobody {label}', ['select "one"', 'save x.h5', 'select "two"'], source='web_agent')
                result = self.finish(r['request_id'])
                self.assertEqual(result['status'], 'failed', result)
                self.assertEqual([c['status'] for c in result['commands']], ['succeeded', 'failed', 'skipped'])
                self.assertEqual([c.get('approval') for c in result['commands']], [None, 'refused', None])
                message = result['commands'][1]['messages'][-1]['text']
                self.assertIn('Viewer shows no window', message)
                self.assertIn('nobody can', message)
                later = self.finish(self.portal.submit(f'later {label}', 'select "two"')['request_id'])
                self.assertEqual(later['status'], 'succeeded', later)
                self.assertEqual(self.portal.queue, [])
        self.assertEqual(saves, [])
        # Commands that need no approval are not affected, and nothing waited or was announced.
        self.assertFalse([e for e in self.viewer.events if e.get('status') == 'awaiting_user_input'])

    def test_an_open_agent_page_or_a_shown_window_can_approve(self):
        saves = self.stand_in('save')
        self.viewer.headless = True
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: client == 'agent')
        r = self.portal.submit('page open', ['save x.h5'], source='web_agent')
        waiting = self.wait_until(r['request_id'], 'awaiting_user_input')
        self.assertEqual(waiting['commands'][0]['approval'], 'pending')
        self.portal.decide(r['request_id'], waiting['commands'][0]['command_id'], True)
        self.assertEqual(self.finish(r['request_id'])['status'], 'succeeded')
        self.assertEqual(saves, [['x.h5']])

        # A page for some other client does not count, and a window that is shown does.
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: client == 'meta')
        refused = self.finish(self.portal.submit('meta page only', ['save y.h5'], source='web_agent')['request_id'])
        self.assertEqual(refused['commands'][0]['approval'], 'refused')
        del self.viewer.headless
        self.viewer.shows_windows = lambda: True
        r = self.portal.submit('window shown', ['save z.h5'], source='web_agent')
        self.wait_until(r['request_id'], 'awaiting_user_input')
        self.portal.discard_pending(r['request_id'])
        self.assertEqual(self.finish(r['request_id'])['status'], 'cancelled')

    def test_the_portal_announces_a_command_when_it_starts_waiting_once(self):
        self.stand_in('save')
        announced = []
        self.portal.approval_needed.connect(lambda request_id, command_id: announced.append((request_id, command_id)))
        r = self.portal.submit('announce', ['select "one"', 'save a.h5', 'save b.h5'], source='web_agent')
        first = self.wait_until(r['request_id'], 'awaiting_user_input')['commands'][1]
        for _ in range(20):  # Further passes over the queue do not announce it again.
            self.portal.pump()
            self.app.processEvents()
            time.sleep(.005)
        self.assertEqual(announced, [(r['request_id'], first['command_id'])])
        self.portal.decide(r['request_id'], first['command_id'], True)
        deadline = time.monotonic() + 10
        while len(announced) < 2 and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        second = self.portal.get(r['request_id'])['commands'][2]
        self.assertEqual(announced[1], (r['request_id'], second['command_id']))
        self.portal.discard_pending(r['request_id'])
        self.finish(r['request_id'])
        # A command that needs no review is never announced.
        before = len(announced)
        self.finish(self.portal.submit('plain', 'select "two"', source='web_agent')['request_id'])
        self.assertEqual(len(announced), before)


class WorkerTrackingTests(PortalFixture, unittest.TestCase):
    """Detached workers and command scripts that end without a usable result fail their request."""

    def tracked_command(self, submission_id):
        from Viewer_Command_Portal import ExecutionContext
        request_id = self.portal.submit(submission_id, 'test')['request_id']
        record = self.portal.requests[request_id]['commands'][0]
        record.update(dispatched=True, outcome='succeeded', status='running')
        return request_id, ExecutionContext(self.portal, request_id, record)

    def assert_worker_failed(self, request_id, message):
        result = self.finish(request_id)
        self.assertEqual(result['status'], 'failed', result)
        job = result['commands'][0]['jobs'][0]
        self.assertEqual((job['status'], job['message']), ('failed', message))

    def test_worker_that_ends_without_a_final_status_fails_the_request(self):
        import psutil
        import Viewer_Worker_Tracking
        created = psutil.Process().create_time()
        # A live worker that has not finished is left running.
        request_id, context = self.tracked_command('alive')
        tracker = worker_tracker(context, self.directory.name)
        tracker.path.write_text(json.dumps({'pid': os.getpid(), 'created': created, 'status': 'running'}))
        tracker.poll()
        self.portal.pump()
        self.assertEqual(self.portal.get(request_id)['status'], 'running')
        self.assertEqual(context.record['jobs'][0]['status'], 'running')
        # The worker process has exited.
        with mock.patch.object(Viewer_Worker_Tracking.psutil, 'Process', side_effect=psutil.NoSuchProcess(os.getpid())):
            tracker.poll()
        self.assert_worker_failed(request_id, 'Worker exited before reporting a final result')
        # The recorded PID now belongs to a process started at another time.
        request_id, context = self.tracked_command('pid reused')
        tracker = worker_tracker(context, self.directory.name)
        tracker.path.write_text(json.dumps({'pid': os.getpid(), 'created': created - 10, 'status': 'running'}))
        tracker.poll()
        self.assert_worker_failed(request_id, 'Worker exited before reporting a final result')

    def test_worker_that_never_reports_startup_fails_after_60_seconds(self):
        import Viewer_Worker_Tracking
        request_id, context = self.tracked_command('silent')
        tracker = worker_tracker(context, self.directory.name)
        for elapsed, status in ((59, 'queued'), (61, 'failed')):
            clock = SimpleNamespace(monotonic=lambda: tracker.created + elapsed)
            with mock.patch.object(Viewer_Worker_Tracking, 'time', clock):
                tracker.poll()
            self.assertEqual(context.record['jobs'][0]['status'], status)
        self.assert_worker_failed(
            request_id, 'Worker did not report startup within 60 seconds; inspect its terminal before retrying')

    def test_a_stopped_script_is_cancelled_and_prepares_no_commands(self):
        import Viewer_Worker_Tracking
        from Viewer_Worker_Tracking import ScriptTracker
        script = Path(self.directory.name) / 'waiting.py'
        script.write_text('import time\nprint(\'select "one"\')\ntime.sleep(120)\n')
        request_id, context = self.tracked_command('stopped script')
        tracker = ScriptTracker(context, str(script))

        self.assertEqual(Viewer_Worker_Tracking.stop_command_scripts(), 1)

        result = self.finish(request_id)
        job = result['commands'][0]['jobs'][0]
        self.assertEqual((job['status'], job['message']), ('cancelled', 'Python command script stopped'))
        self.assertEqual(self.portal.get(request_id)['total_commands'], 1)  # no child commands
        self.assertEqual(self.viewer.selected_indices, [])
        self.assertTrue(tracker._script.stopped)

    def test_script_that_exits_with_an_error_prepares_no_commands(self):
        from Viewer_Worker_Tracking import ScriptTracker
        script = Path(self.directory.name) / 'failing.py'
        script.write_text('import sys\nprint(\'select "one"\')\nsys.exit(3)\n')
        request_id, context = self.tracked_command('failing script')
        ScriptTracker(context, str(script))
        self.assert_worker_failed(request_id, 'Python command script exited with code 3')
        self.assertEqual(self.portal.get(request_id)['total_commands'], 1)  # no child commands
        self.assertEqual(self.viewer.selected_indices, [])
        self.assertIn('select "one"', self.portal.read_output(request_id)['text'])


class HiddenWorkerFixture(PortalFixture):
    """A worker with no window (a headless Viewer's) is ended by the Viewer, and its failures name its log.

    Everyone else can close a worker's console; nobody can close a window that does not exist.
    """

    # Starts a child of its own, records both process IDs, and runs until it is ended.
    TREE_SCRIPT = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(sys.argv[1], 'w').write(f'{os.getpid()} {child.pid}')\n"
        "time.sleep(120)\n"
    )

    def setUp(self):
        super().setUp()
        import Viewer_Worker_Tracking
        # No Viewer's workers, and no earlier test's application hook.
        for name, value in (('_hidden_workers', set()), ('_hidden_quit_hooked', False)):
            patcher = mock.patch.object(Viewer_Worker_Tracking, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tracked_command(self, submission_id='hidden'):
        from Viewer_Command_Portal import ExecutionContext
        request_id = self.portal.submit(submission_id, 'test')['request_id']
        record = self.portal.requests[request_id]['commands'][0]
        record.update(dispatched=True, outcome='succeeded', status='running')
        return request_id, ExecutionContext(self.portal, request_id, record)

    def end_leftovers(self, process):
        import psutil
        try:
            root = psutil.Process(process.pid)
            targets = [root, *root.children(recursive=True)]
        except psutil.Error:
            targets = []
        for target in targets:
            try:
                target.kill()
            except psutil.Error:
                pass
        process.wait(timeout=30)

    def hidden_worker(self, code, *arguments):
        """Start a windowless worker as the esmfold command does; return (process, log path)."""
        from utilities.Terminal_Launcher import launch_without_window
        script = Path(self.directory.name) / 'worker.py'
        script.write_text(code, encoding='utf-8')
        log_path = Path(self.directory.name) / 'logs' / 'worker.log'
        process = launch_without_window([sys.executable, str(script), *map(str, arguments)],
                                        cwd=self.directory.name, log_path=log_path)
        self.addCleanup(self.end_leftovers, process)
        return process, str(log_path)

    def started_tree(self):
        """A running hidden worker and its child; return (process, log path, [psutil.Process of both])."""
        import psutil
        pids = Path(self.directory.name) / 'pids.txt'
        process, log = self.hidden_worker(self.TREE_SCRIPT, pids)
        deadline = time.monotonic() + 60
        while True:
            fields = pids.read_text().split() if pids.exists() else []
            if len(fields) == 2:
                break
            self.assertLess(time.monotonic(), deadline, 'The hidden worker never started its child')
            time.sleep(0.05)
        processes = [psutil.Process(int(field)) for field in fields]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        return process, log, processes

    def assert_all_ended(self, process, processes):
        import psutil
        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [], 'A process the hidden worker started is still running')
        self.assertIsNotNone(process.poll())

class HiddenWorkerTests(HiddenWorkerFixture, unittest.TestCase):
    """A worker with no window (a headless Viewer's) is ended by the Viewer, and its failures name its log.

    Everyone else can close a worker's console; nobody can close a window that does not exist.
    """

    def test_stopping_the_workers_ends_each_with_everything_it_started(self):
        import Viewer_Worker_Tracking
        process, _, processes = self.started_tree()
        Viewer_Worker_Tracking.track_hidden_worker(process)

        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 1)

        self.assert_all_ended(process, processes)
        # Nothing is left to stop, and a worker that ended by itself is not counted.
        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 0)
        finished, _ = self.hidden_worker('print("done")\n')
        finished.wait(timeout=60)
        Viewer_Worker_Tracking.track_hidden_worker(finished)
        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 0)

    def test_quitting_the_application_stops_the_hidden_workers(self):
        import Viewer_Worker_Tracking
        process, _, processes = self.started_tree()
        # A stand-in application: emitting the real aboutToQuit would run every other test's hooks.
        application = SimpleNamespace(aboutToQuit=mock.Mock())
        with mock.patch.object(Viewer_Worker_Tracking.QtCore.QCoreApplication, 'instance', return_value=application):
            Viewer_Worker_Tracking.track_hidden_worker(process)
            Viewer_Worker_Tracking.track_hidden_worker(process)  # hooked once, however many workers
        application.aboutToQuit.connect.assert_called_once_with(Viewer_Worker_Tracking.stop_hidden_workers)

        # What the signal then runs.
        Viewer_Worker_Tracking.stop_hidden_workers()

        self.assert_all_ended(process, processes)

    def test_a_job_that_fails_ends_its_hidden_worker_and_names_the_log(self):
        import Viewer_Worker_Tracking
        process, log, processes = self.started_tree()
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.attach_hidden(process, log)
        # The worker never reports startup.
        clock = SimpleNamespace(monotonic=lambda: tracker.created + 61)
        with mock.patch.object(Viewer_Worker_Tracking, 'time', clock):
            tracker.poll()

        result = self.finish(request_id)
        self.assertEqual(result['status'], 'failed', result)
        job = result['commands'][0]['jobs'][0]
        self.assertEqual(job['message'],
                         f'Worker did not report startup within 60 seconds; inspect its log {log} before retrying')
        self.assert_all_ended(process, processes)

    def test_a_worker_that_dies_at_startup_fails_the_job_at_once_with_its_log(self):
        process, log = self.hidden_worker('raise ImportError("no torch here")\n')
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.attach_hidden(process, log)

        deadline = time.monotonic() + 60
        while context.record['jobs'][0]['status'] != 'failed':
            self.assertLess(time.monotonic(), deadline, 'The dead worker was never reported')
            tracker.poll()
            time.sleep(0.05)

        self.assertLess(time.monotonic() - tracker.created, 60)
        result = self.finish(request_id)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['commands'][0]['jobs'][0]['message'],
                         f'Worker exited with code 1 before reporting startup; its output is in {log}')
        # The traceback that a terminal would have shown is where the message points.
        self.assertIn('ImportError: no torch here', Path(log).read_text(encoding='utf-8'))

    def test_a_status_file_in_flight_is_not_mistaken_for_a_dead_worker(self):
        # The worker has exited but its status file exists (it may be mid-replace
        # and unreadable for a moment): the next poll reads its real report.
        import psutil
        process = SimpleNamespace(poll=lambda: 0, returncode=0)
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.attach_hidden(process, 'worker.log')
        tracker.path.write_text('{"half"')  # not valid JSON yet

        tracker.poll()
        self.assertEqual(context.record['jobs'][0]['status'], 'queued')

        tracker.path.write_text(json.dumps({'pid': os.getpid(), 'created': psutil.Process().create_time(),
                                            'status': 'succeeded', 'message': 'Structure prediction completed'}))
        tracker.poll()
        self.assertEqual(self.finish(request_id)['status'], 'succeeded')

    def test_a_hidden_worker_that_exited_without_a_final_status_names_the_log(self):
        import psutil
        import Viewer_Worker_Tracking
        process = mock.Mock()
        process.poll.return_value = 0
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.attach_hidden(process, 'worker.log')
        tracker.path.write_text(json.dumps({'pid': os.getpid(), 'created': psutil.Process().create_time() - 10,
                                            'status': 'running'}))

        with mock.patch.object(Viewer_Worker_Tracking, '_end_process_tree') as end_tree:
            tracker.poll()

        self.assertEqual(self.finish(request_id)['commands'][0]['jobs'][0]['message'],
                         'Worker exited before reporting a final result; its output is in worker.log')
        # It had ended: no process tree is touched, so a reused PID is safe.
        end_tree.assert_not_called()

    def test_a_worker_with_a_terminal_is_left_to_its_window(self):
        import Viewer_Worker_Tracking
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        self.assertIsNone(tracker.hidden)

        with mock.patch.object(Viewer_Worker_Tracking, '_end_process_tree') as end_tree:
            tracker.fail('Worker did not report startup within 60 seconds; inspect its terminal before retrying')

        end_tree.assert_not_called()


class StartedWorkerTests(HiddenWorkerFixture, unittest.TestCase):
    """Workers a command started, in a console or hidden: `esmfold stop` ends them, and none leaves its input file.

    The worker is a real process; its input file is a real private file the registry is told about.
    """

    def setUp(self):
        super().setUp()
        import Viewer_Worker_Tracking
        for name, value in (('_started_workers', []), ('_started_quit_hooked', True)):
            patcher = mock.patch.object(Viewer_Worker_Tracking, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(Viewer_Worker_Tracking.stop_workers, 'test over')

    def input_file(self, name='input.json'):
        path = Path(self.directory.name) / name
        path.write_text('[["one", "ACDE"]]', encoding='utf-8')
        return str(path)

    def tracked_worker(self, code, *, tracker=None, input_path=None, output_dir=None, extra=()):
        """A real worker running code, entered in the registry; its input file's name is its first argument."""
        import Viewer_Worker_Tracking
        input_path = input_path or self.input_file()
        process, log = self.hidden_worker(code, input_path, *extra)
        worker = Viewer_Worker_Tracking.track_worker(
            process, input_path=input_path, output_dir=output_dir, tracker=tracker)
        return process, log, worker, input_path

    def test_stop_ends_the_worker_and_its_children_and_cancels_its_job(self):
        import Viewer_Worker_Tracking
        process, log, processes = self.started_tree()
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.timer.start(250)  # As a command's tracker polls while its job runs.
        self.addCleanup(tracker.timer.stop)
        Viewer_Worker_Tracking.track_worker(process, input_path=self.input_file(), tracker=tracker)

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('ESMFold worker stopped'), 1)

        self.assert_all_ended(process, processes)
        result = self.finish(request_id)
        self.assertEqual(result['status'], 'cancelled', result)
        job = result['commands'][0]['jobs'][0]
        self.assertEqual((job['status'], job['message']), ('cancelled', 'ESMFold worker stopped'))
        self.assertFalse(tracker.timer.isActive())
        # The tracker does not turn the ended worker into a failure afterwards.
        tracker.poll()
        self.assertEqual(self.portal.get(request_id)['status'], 'cancelled')

    def test_stop_reports_a_structure_saved_just_before_it_ended_the_worker(self):
        import psutil
        import Viewer_Worker_Tracking
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.timer.start(600000)  # It would not read the status file for ten minutes.
        self.addCleanup(tracker.timer.stop)
        process, log, worker, input_path = self.tracked_worker('import time\ntime.sleep(120)\n', tracker=tracker)
        tracker.path.write_text(json.dumps({
            'pid': process.pid, 'created': psutil.Process(process.pid).create_time(), 'status': 'running',
            'message': 'Folded one', 'artifacts': ['one.pdb'], 'result': {'succeeded': 1, 'total': 2}}))

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('ESMFold worker stopped'), 1)

        result = self.finish(request_id)
        self.assertEqual(result['status'], 'cancelled', result)
        self.assertEqual(len(result['commands'][0]['artifacts']), 1)
        job = result['commands'][0]['jobs'][0]
        self.assertEqual((job['status'], job['message'], job['result']),
                         ('cancelled', 'ESMFold worker stopped', {'succeeded': 1, 'total': 2}))

    def test_a_worker_its_status_file_names_is_found_without_its_input_file(self):
        import psutil
        import Viewer_Worker_Tracking
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.timer.start(600000)
        self.addCleanup(tracker.timer.stop)
        window, _ = self.hidden_worker('pass\n')  # The window's process, ended at once.
        window.wait(timeout=60)
        worker, _ = self.hidden_worker('import time\ntime.sleep(120)\n')
        # The worker has told the Viewer who it is (the first status it writes).
        tracker.identity = (worker.pid, psutil.Process(worker.pid).create_time())
        Viewer_Worker_Tracking.track_worker(window, tracker=tracker)

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('stopped'), 1)

        self.assertIsNotNone(worker.poll())
        self.assertEqual(self.finish(request_id)['status'], 'cancelled')

    def test_stop_ends_only_what_the_registry_started(self):
        import Viewer_Worker_Tracking
        process, log, worker, input_path = self.tracked_worker('import time\ntime.sleep(120)\n')
        other = Path(self.directory.name) / 'other.py'
        other.write_text('import time\ntime.sleep(120)\n', encoding='utf-8')
        bystander = subprocess.Popen([sys.executable, str(other), self.input_file('someone_elses.json')])
        self.addCleanup(lambda: (bystander.kill(), bystander.wait(timeout=10)))

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('stopped'), 1)

        self.assertIsNotNone(process.poll())
        self.assertIsNone(bystander.poll())
        self.assertEqual(Viewer_Worker_Tracking.stop_workers('stopped'), 0)

    def test_a_worker_is_found_by_its_input_file_after_its_window_handed_it_over(self):
        import Viewer_Worker_Tracking
        input_path = self.input_file()
        script = Path(self.directory.name) / 'worker.py'
        script.write_text('import time\ntime.sleep(120)\n', encoding='utf-8')
        window = Path(self.directory.name) / 'window.py'
        window.write_text('import subprocess, sys\nsubprocess.Popen(sys.argv[1:])\n', encoding='utf-8')
        from utilities.Terminal_Launcher import launch_without_window
        process = launch_without_window(
            [sys.executable, str(window), sys.executable, str(script), input_path],
            cwd=self.directory.name, log_path=Path(self.directory.name) / 'window.log')
        process.wait(timeout=30)  # The window's process is gone; the worker it started is not.
        self.addCleanup(self.end_leftovers, process)
        Viewer_Worker_Tracking.track_worker(process, input_path=input_path)
        import psutil
        found = [p for p in psutil.process_iter(['cmdline']) if input_path in (p.info['cmdline'] or ())]
        self.assertTrue(found)
        self.addCleanup(lambda: [p.kill() for p in found if p.is_running()])

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('stopped'), 1)

        _, alive = psutil.wait_procs(found, timeout=10)
        self.assertEqual(alive, [])
        self.assertFalse(os.path.exists(input_path))

    def test_stop_deletes_the_half_written_structure_and_keeps_the_finished_one(self):
        import Viewer_Worker_Tracking
        structures = Path(self.directory.name) / 'structures'
        structures.mkdir()
        code = (
            "import os, sys, time\n"
            "folder = sys.argv[2]\n"
            "open(os.path.join(folder, 'one.pdb'), 'w').write('ATOM\\n')\n"
            "open(os.path.join(folder, 'two.pdb.%d.partial' % os.getpid()), 'w').write('ATO')\n"
            "open(os.path.join(folder, 'ready'), 'w').close()\n"
            "time.sleep(120)\n")
        process, log, worker, input_path = self.tracked_worker(code, output_dir=str(structures), extra=(structures,))
        deadline = time.monotonic() + 60
        while not (structures / 'ready').exists():
            self.assertLess(time.monotonic(), deadline, 'The stand-in worker never wrote its files')
            time.sleep(0.05)
        (structures / 'ready').unlink()
        # A partial file of someone else's process, which stays.
        (structures / 'three.pdb.1.partial').write_text('x')

        self.assertEqual(Viewer_Worker_Tracking.stop_workers('stopped'), 1)

        self.assertEqual(sorted(p.name for p in structures.iterdir()), ['one.pdb', 'three.pdb.1.partial'])
        self.assertFalse(os.path.exists(input_path))

    def test_a_worker_that_died_before_reading_its_input_leaves_no_file(self):
        process, log, worker, input_path = self.tracked_worker('raise ImportError("no torch here")\n')
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        tracker.attach_hidden(process, log)
        worker.tracker = tracker
        tracker.worker = worker
        process.wait(timeout=60)
        self.assertTrue(os.path.exists(input_path))  # It never read it.

        deadline = time.monotonic() + 60
        while context.record['jobs'][0]['status'] != 'failed':
            self.assertLess(time.monotonic(), deadline, 'The dead worker was never reported')
            tracker.poll()
            time.sleep(0.05)

        self.assertFalse(os.path.exists(input_path))

    def test_a_worker_still_running_keeps_its_input_whatever_its_job_says(self):
        import Viewer_Worker_Tracking
        request_id, context = self.tracked_command()
        tracker = worker_tracker(context, self.directory.name)
        process, log, worker, input_path = self.tracked_worker('import time\ntime.sleep(120)\n', tracker=tracker)
        tracker.worker = worker
        # A console worker that is slow to start: the job fails after 60 s, the worker may still start.
        clock = SimpleNamespace(monotonic=lambda: tracker.created + 61)
        with mock.patch.object(Viewer_Worker_Tracking, 'time', clock):
            tracker.poll()

        self.assertEqual(context.record['jobs'][0]['status'], 'failed')
        self.assertIsNone(process.poll())
        self.assertTrue(os.path.exists(input_path))
        self.assertFalse(worker.tidy())

    def test_the_next_worker_tidies_the_input_of_one_that_ended_unread(self):
        import Viewer_Worker_Tracking
        first, _, _, first_input = self.tracked_worker('print("never reads its input")\n')
        first.wait(timeout=60)
        self.assertTrue(os.path.exists(first_input))
        second_input = self.input_file('second.json')

        process, _, _, _ = self.tracked_worker(
            'import time\ntime.sleep(120)\n', input_path=second_input)

        self.assertFalse(os.path.exists(first_input))
        self.assertTrue(os.path.exists(second_input))
        self.assertEqual(len(Viewer_Worker_Tracking._started_workers), 1)

    def test_quitting_deletes_the_input_of_a_hidden_worker_it_ended(self):
        import Viewer_Worker_Tracking
        process, log, worker, input_path = self.tracked_worker('import time\ntime.sleep(120)\n')
        Viewer_Worker_Tracking.track_hidden_worker(process)
        self.assertTrue(os.path.exists(input_path))

        # What the application's aboutToQuit signal runs for a hidden worker.
        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 1)

        self.assertIsNotNone(process.poll())
        self.assertFalse(os.path.exists(input_path))

    def test_the_quit_hook_for_every_worker_is_connected_once(self):
        import Viewer_Worker_Tracking
        application = SimpleNamespace(aboutToQuit=mock.Mock())
        with mock.patch.object(Viewer_Worker_Tracking, '_started_quit_hooked', False), \
                mock.patch.object(Viewer_Worker_Tracking.QtCore.QCoreApplication, 'instance', return_value=application):
            process = SimpleNamespace(poll=lambda: 0, pid=0)
            Viewer_Worker_Tracking.track_worker(process)
            Viewer_Worker_Tracking.track_worker(process)
        application.aboutToQuit.connect.assert_called_once_with(Viewer_Worker_Tracking.tidy_workers)


class JobProgressTests(PortalFixture, unittest.TestCase):
    """A job entry's `progress`: one plain line, copied in by whatever feeds the job, kept only when it changes."""

    def running_job(self):
        """Submit a request whose command starts a job that stays running; return (request id, its context)."""
        started = {}

        def run(viewer, args):
            context = CURRENT.get()
            context.add_job('job-1')
            context.job_event('job-1', 'running')
            started['context'] = context
            ce.command_succeeded(viewer)

        stack = fake_test_command(run)
        stack.start()
        self.addCleanup(stack.stop)
        request_id = self.portal.submit('progress', 'test')['request_id']
        deadline = time.monotonic() + 30
        while 'context' not in started:
            self.assertLess(time.monotonic(), deadline, 'The request never started its job')
            self.app.processEvents()
            time.sleep(0.005)
        return request_id, started['context']

    def entry(self, request_id):
        return self.portal.get(request_id)['commands'][0]['jobs'][0]

    def test_a_job_has_no_progress_until_something_reports_one(self):
        request_id, context = self.running_job()
        self.assertIsNone(self.entry(request_id)['progress'])
        for blank in (None, '', '   \n '):
            self.assertFalse(context.job_progress('job-1', blank))
        self.assertIsNone(self.entry(request_id)['progress'])

    def test_progress_appears_in_the_record_and_changes_only_when_the_text_does(self):
        request_id, context = self.running_job()
        self.assertTrue(context.job_progress('job-1', 'pass 1 of 3'))
        self.assertEqual(self.entry(request_id)['progress'], 'pass 1 of 3')
        self.assertFalse(context.job_progress('job-1', 'pass 1 of 3'))
        self.assertTrue(context.job_progress('job-1', 'pass 2 of 3'))
        self.assertEqual(self.entry(request_id)['progress'], 'pass 2 of 3')
        self.assertEqual(self.entry(request_id)['status'], 'running')

    def test_progress_is_a_plain_one_line_string_of_bounded_length(self):
        from Viewer_Command_Portal import MAX_PROGRESS_CHARACTERS
        request_id, context = self.running_job()
        context.job_progress('job-1', 'step\n  4 of 9\t(50%)\r\n')
        self.assertEqual(self.entry(request_id)['progress'], 'step 4 of 9 (50%)')
        context.job_progress('job-1', 'x' * 5000)
        self.assertEqual(self.entry(request_id)['progress'], 'x' * MAX_PROGRESS_CHARACTERS)
        context.job_progress('job-1', 42)
        self.assertEqual(self.entry(request_id)['progress'], '42')
        self.assertIsInstance(self.entry(request_id)['progress'], str)

    def test_progress_for_an_unknown_job_is_ignored_and_other_jobs_keep_theirs(self):
        request_id, context = self.running_job()
        context.add_job('job-2')
        self.assertFalse(context.job_progress('missing', 'text'))
        context.job_progress('job-1', 'one')
        context.job_progress('job-2', 'two')
        jobs = self.portal.get(request_id)['commands'][0]['jobs']
        self.assertEqual([(job['job_id'], job['progress']) for job in jobs], [('job-1', 'one'), ('job-2', 'two')])

    def test_the_last_progress_stays_in_the_entry_when_the_job_ends(self):
        request_id, context = self.running_job()
        context.job_progress('job-1', 'pass 3 of 3')
        context.job_event('job-1', 'succeeded', 'done')
        self.assertEqual(self.entry(request_id)['status'], 'succeeded')
        self.assertEqual(self.entry(request_id)['progress'], 'pass 3 of 3')


class StopRequestTests(PortalFixture, unittest.TestCase):
    """A request that holds only stop commands runs at once, even while the request at the head waits for its job."""

    def head_waiting_for_a_job(self):
        """Submit a request whose command starts a job that stays running; return (request id, finish_job)."""
        started = {}

        def run(viewer, args):
            context = CURRENT.get()
            context.add_job('job-1')
            context.job_event('job-1', 'running')
            started['context'] = context
            ce.command_succeeded(viewer)

        stack = fake_test_command(run)
        stack.start()
        self.addCleanup(stack.stop)
        request_id = self.portal.submit('head', 'test')['request_id']
        deadline = time.monotonic() + 30
        while 'context' not in started:
            self.assertLess(time.monotonic(), deadline, 'The head request never started its job')
            self.app.processEvents()
            time.sleep(0.005)
        self.assertEqual(self.portal.get(request_id)['status'], 'running')
        return request_id, lambda status='succeeded': started['context'].job_event('job-1', status)

    def stop_commands(self, name='esmfold'):
        """Stand-ins for `esmfold stop` and `run stop`, which record that they ran."""
        calls = []

        def run(viewer, args):
            calls.append((name, list(args)))
            ce.command_succeeded(viewer, 'stopped')

        stack = fake_test_command(run, command=name)
        stack.start()
        self.addCleanup(stack.stop)
        return calls

    def test_a_stop_request_runs_while_the_head_waits_and_the_head_goes_on_when_its_job_ends(self):
        head, end_job = self.head_waiting_for_a_job()
        calls = self.stop_commands('esmfold')

        stop = self.finish(self.portal.submit('stop', 'esmfold stop')['request_id'])

        self.assertEqual(stop['status'], 'succeeded', stop)
        self.assertEqual(calls, [('esmfold', ['stop'])])
        self.assertEqual(self.portal.get(head)['status'], 'running')
        end_job('cancelled')
        self.assertEqual(self.finish(head)['status'], 'cancelled')

    def test_run_stop_runs_ahead_too_and_case_and_spacing_do_not_matter(self):
        head, end_job = self.head_waiting_for_a_job()
        run_calls = self.stop_commands('run')

        self.finish(self.portal.submit('a', 'Run  STOP')['request_id'])

        self.assertEqual(run_calls, [('run', ['STOP'])])
        end_job()
        self.assertEqual(self.finish(head)['status'], 'succeeded')

    def test_several_stops_run_in_order_and_other_requests_keep_their_place(self):
        head, end_job = self.head_waiting_for_a_job()
        calls = self.stop_commands('esmfold')
        with fake_test_command(lambda viewer, args: ce.command_succeeded(viewer), command='select'):
            later = self.portal.submit('later', 'select "one"')['request_id']
            first = self.finish(self.portal.submit('s1', 'esmfold stop')['request_id'])
            second = self.finish(self.portal.submit('s2', ['esmfold stop', 'esmfold stop'])['request_id'])

            self.assertEqual((first['status'], second['status']), ('succeeded', 'succeeded'))
            self.assertEqual(len(calls), 3)
            # The request that is not a stop waits behind the head, as before.
            self.assertEqual(self.portal.get(later)['status'], 'queued')
            end_job()
            self.assertEqual(self.finish(head)['status'], 'succeeded')
            self.assertEqual(self.finish(later)['status'], 'succeeded')

    def test_only_a_request_of_nothing_but_stops_runs_ahead(self):
        head, end_job = self.head_waiting_for_a_job()
        calls = self.stop_commands('esmfold')
        self.stop_commands('run')
        submitted = {
            'mixed': self.portal.submit('mixed', ['esmfold stop', 'select "one"'])['request_id'],
            'extra word': self.portal.submit('extra', 'esmfold stop large')['request_id'],
            'bare': self.portal.submit('bare', 'esmfold')['request_id'],
        }
        for _ in range(30):
            self.app.processEvents()
            time.sleep(0.005)

        for label, request_id in submitted.items():
            with self.subTest(label):
                self.assertEqual(self.portal.get(request_id)['status'], 'queued')
        self.assertEqual(calls, [])
        end_job()
        for request_id in submitted.values():
            self.finish(request_id)

    def test_a_plugin_that_declares_it_acts_on_jobs_runs_ahead_without_any_change_to_the_portal(self):
        record = ("import Command_Engine\n"
                  "def run(viewer, args):\n"
                  "    Command_Engine.SCRATCH_CALLS.append((__name__.split('.')[-1], list(args)))\n"
                  "    Command_Engine.command_succeeded(viewer)\n")
        calls = []
        plugins = {
            'zzdecl_halt': "def acts_on_jobs(args):\n    return args == ['halt']\n" + record,
            'zzdecl_always': "ACTS_ON_JOBS = True\n" + record,
            'zzdecl_plain': record,
        }
        with scratch_commands(plugins), mock.patch.object(ce, 'SCRATCH_CALLS', calls, create=True):
            head, end_job = self.head_waiting_for_a_job()
            ahead = [self.portal.submit('halt', 'zzdecl_halt halt')['request_id'],
                     self.portal.submit('always', 'ZZDECL_ALWAYS anything at all')['request_id']]
            behind = {
                'other arguments': self.portal.submit('other', 'zzdecl_halt now')['request_id'],
                'declares nothing': self.portal.submit('plain', 'zzdecl_plain halt')['request_id'],
                'mixed': self.portal.submit('mixed', ['zzdecl_halt halt', 'zzdecl_plain'])['request_id'],
            }
            for request_id in ahead:
                self.assertEqual(self.finish(request_id)['status'], 'succeeded')
            for _ in range(30):
                self.app.processEvents()
                time.sleep(0.005)

            self.assertEqual(calls, [('zzdecl_halt', ['halt']), ('zzdecl_always', ['anything', 'at', 'all'])])
            for label, request_id in behind.items():
                with self.subTest(label):
                    self.assertEqual(self.portal.get(request_id)['status'], 'queued')
            self.assertEqual(self.portal.get(head)['status'], 'running')
            end_job()
            for request_id in (head, *behind.values()):
                self.finish(request_id)

    def test_a_stop_waiting_for_an_agent_review_does_not_run_ahead(self):
        head, end_job = self.head_waiting_for_a_job()
        calls = self.stop_commands('esmfold')
        request = self.portal.submit('agent stop', 'esmfold stop', source='web_agent')['request_id']
        for _ in range(30):
            self.app.processEvents()
            time.sleep(0.005)
        self.assertEqual(self.portal.get(request)['status'], 'queued')
        self.assertEqual(calls, [])

        command_id = self.portal.requests[request]['commands'][0]['command_id']
        # Not yet its turn to be reviewed: it waits behind the head. Once the head ends, it asks.
        end_job()
        self.finish(head)
        deadline = time.monotonic() + 30
        while self.portal.get(request)['status'] != 'awaiting_user_input':
            self.assertLess(time.monotonic(), deadline, self.portal.get(request))
            self.app.processEvents()
            time.sleep(0.005)
        self.portal.decide(request, command_id, True)
        self.assertEqual(self.finish(request)['status'], 'succeeded')
        self.assertEqual(calls, [('esmfold', ['stop'])])

    def test_a_head_that_waits_for_the_user_does_not_let_a_stop_run_ahead(self):
        # Nothing is running to stop: the head has not started, it waits for its Agent review.
        calls = self.stop_commands('esmfold')
        waiting = self.portal.submit('review', 'run', source='web_agent')['request_id']
        stop = self.portal.submit('stop', 'esmfold stop')['request_id']
        for _ in range(30):
            self.app.processEvents()
            time.sleep(0.005)

        self.assertEqual(self.portal.get(waiting)['status'], 'awaiting_user_input')
        self.assertEqual(self.portal.get(stop)['status'], 'queued')
        self.assertEqual(calls, [])
        self.portal.discard_pending(waiting)
        self.finish(waiting)
        self.assertEqual(self.finish(stop)['status'], 'succeeded')


class FreedLineTests(PortalFixture, unittest.TestCase):
    """A request whose remaining work is only background lane jobs lets the next request run, and ends when its jobs do."""

    def setUp(self):
        super().setUp()
        self.scheduler = BackgroundJobScheduler(self.viewer)
        self.addCleanup(end_scheduler, self.scheduler)
        self.release = {}
        self.begun = {}
        self.completed = []
        self.portal.completed.connect(self.completed.append)

    def worker(self, payload):
        self.begun[payload].set()
        if not self.release[payload].wait(30):
            raise RuntimeError('never released')
        if payload.startswith('fails'):
            raise ValueError('expected job failure')
        return {'message': f'done {payload}'}

    def command(self, lanes=None, releases=True):
        """Make `test NAME` queue one job named NAME, in the lane lanes[NAME] (default: the files lane)."""
        lanes = lanes or {}

        def run(viewer, args):
            name = args[0]
            self.begun[name], self.release[name] = threading.Event(), threading.Event()
            options = {'lane': lanes[name]} if name in lanes else {}
            if not releases:
                options['releases_request'] = False
            self.scheduler.enqueue('test', f'test {name}', name, self.worker,
                                   str(Path(self.directory.name) / f'{name}.out'), **options)
            ce.command_succeeded(viewer)

        stack = fake_test_command(run)
        stack.start()
        self.addCleanup(stack.stop)

    def settle(self, condition, message, timeout=30):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, message)
            self.app.processEvents()
            time.sleep(0.005)

    def running(self, name):
        return lambda: name in self.begun and self.begun[name].is_set()

    def idle(self, turns=20):
        for _ in range(turns):
            self.app.processEvents()
            time.sleep(0.005)

    def status(self, request_id):
        return self.portal.get(request_id)['status']

    def test_two_requests_with_a_job_in_each_lane_overlap(self):
        self.command(lanes={'fold': 'folding'})
        first = self.portal.submit('one', 'test file')['request_id']
        second = self.portal.submit('two', 'test fold')['request_id']

        # The second runs while the first's job still does: the first no longer holds the head.
        self.settle(self.running('fold'), 'the second request never ran beside the first')
        self.assertTrue(self.begun['file'].is_set())
        self.assertFalse(self.release['file'].is_set())
        self.assertEqual((self.status(first), self.status(second)), ('running', 'running'))

        self.release['fold'].set()
        self.assertEqual(self.finish(second)['status'], 'succeeded')
        self.assertEqual(self.status(first), 'running', 'it ends when its own job does')
        self.release['file'].set()
        self.assertEqual(self.finish(first)['status'], 'succeeded')
        self.assertEqual(sorted(self.completed), sorted([first, second]))

    def test_a_parked_request_ends_with_its_status_its_artifacts_and_one_completed_signal(self):
        self.command()
        request_id = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle(10)
        self.assertEqual(self.status(request_id), 'running')
        self.assertEqual(self.completed, [], 'completed fires when the jobs end, not when the line is freed')

        self.release['a'].set()
        result = self.finish(request_id)

        self.assertEqual(result['status'], 'succeeded', result)
        self.assertTrue(result['finished_at'])
        self.assertEqual(self.completed, [request_id])
        [command] = result['commands']
        self.assertEqual(command['status'], 'succeeded')
        [job] = command['jobs']
        self.assertEqual((job['status'], job['kind'], job['lane']), ('succeeded', 'test', 'files'))
        self.assertTrue(job['finished_at'])
        self.assertIn('done a', job['message'])
        self.assertEqual([Path(path).name for path in command['artifacts']], ['a.out'])
        self.assertEqual(self.portal.queue, [])

    def test_a_parked_requests_failed_job_fails_the_request(self):
        self.command()
        request_id = self.portal.submit('one', 'test fails')['request_id']
        self.settle(self.running('fails'), 'the job never ran')
        self.release['fails'].set()

        result = self.finish(request_id)

        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual(result['commands'][0]['status'], 'failed')
        self.assertIn('expected job failure', result['commands'][0]['jobs'][0]['message'])
        self.assertEqual(self.completed, [request_id])

    def wait_for(self, request_id, timeout=30, release=None, after=4):
        """Run wait_command_request on the real portal, with Qt events pumped as the Viewer pumps them.

        Time is fake (each pause moves it by its length), so nothing waits for real. After ``after`` pauses the
        job named ``release`` is let go. Returns (the wait's result, the pauses it took, its progress calls).
        """
        import asyncio
        from mcp_server.viewer import Viewer_Operations as ops
        clock, pauses, notes = [0.0], [], []

        async def portal_call(ctx, action, arguments, session_id):
            self.assertEqual(action, 'get_command_request')
            return self.portal.get(**arguments)

        async def pause(seconds):
            pauses.append(seconds)
            clock[0] += seconds
            self.app.processEvents()
            if release is not None and len(pauses) == after:
                self.release[release].set()
            await asyncio.sleep(0.005)  # the job's worker thread runs meanwhile

        async def report_progress(progress, total=None, message=None):
            notes.append(message)

        context = SimpleNamespace(report_progress=report_progress)
        with mock.patch.object(ops, '_portal_call', portal_call), mock.patch.object(ops, '_pause', pause), \
                mock.patch.object(ops, '_now', lambda: clock[0]):
            result = asyncio.run(ops.wait_command_request(context, request_id=request_id, timeout_seconds=timeout))
        return result, pauses, notes

    def test_a_wait_on_a_parked_request_returns_when_its_jobs_end_not_when_the_line_is_freed(self):
        self.command()
        request_id = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.settle(lambda: not self.portal.queue, 'the request never left the head of the line')
        self.assertEqual(self.status(request_id), 'running')

        result, pauses, notes = self.wait_for(request_id, release='a')

        self.assertGreaterEqual(len(pauses), 4, 'it returned while the parked request was still running')
        self.assertEqual((result['status'], result['waiting']), ('succeeded', False))
        [command] = result['commands']
        [job] = command['jobs']
        self.assertEqual((job['status'], job['kind'], job['lane']), ('succeeded', 'test', 'files'))
        self.assertEqual([Path(path).name for path in command['artifacts']], ['a.out'])
        self.assertTrue(notes and 'job' in notes[0], notes)
        self.assertEqual(self.completed, [request_id])

    def test_a_wait_on_a_parked_request_times_out_while_its_job_runs_and_the_next_call_finishes_it(self):
        self.command()
        request_id = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')

        first, _, _ = self.wait_for(request_id, timeout=3)

        self.assertEqual((first['status'], first['waiting']), ('running', True))
        self.assertEqual(first['next_step']['action'], 'wait_command_request')
        self.assertEqual(first['next_step']['arguments']['request_id'], request_id)
        self.assertFalse(self.release['a'].is_set())

        second, _, _ = self.wait_for(request_id, release='a')

        self.assertEqual((second['status'], second['waiting']), ('succeeded', False))

    def test_a_wait_on_a_parked_request_whose_job_fails_returns_the_failure(self):
        self.command()
        request_id = self.portal.submit('one', 'test fails')['request_id']
        self.settle(self.running('fails'), 'the job never ran')

        result, _, _ = self.wait_for(request_id, release='fails')

        self.assertEqual((result['status'], result['waiting']), ('failed', False))
        self.assertIn('expected job failure', result['commands'][0]['jobs'][0]['message'])

    def test_a_request_behind_a_parked_one_is_waited_for_on_its_own(self):
        self.command(lanes={'b': 'folding'})
        first = self.portal.submit('one', 'test a')['request_id']
        second = self.portal.submit('two', 'test b')['request_id']
        self.settle(self.running('b'), 'the second request never ran beside the first')

        result, _, _ = self.wait_for(second, release='b')

        self.assertEqual((result['request_id'], result['status']), (second, 'succeeded'))
        self.assertEqual(self.status(first), 'running')
        self.release['a'].set()
        self.assertEqual(self.finish(first)['status'], 'succeeded')

    def test_a_parked_requests_job_removed_while_it_waits_cancels_the_request(self):
        self.command()
        self.portal.submit('hold', 'test hold')
        self.settle(self.running('hold'), 'the first job never ran')
        waiting = self.portal.submit('wait', 'test waits')['request_id']
        self.settle(lambda: bool(self.portal.get(waiting)['commands'][0]['jobs']), 'the second job was never queued')

        job_id = int(self.portal.get(waiting)['commands'][0]['jobs'][0]['job_id'])
        self.assertTrue(self.scheduler.cancel(job_id))

        result = self.finish(waiting)
        self.assertEqual(result['status'], 'cancelled', result)
        self.assertEqual(result['commands'][0]['jobs'][0]['status'], 'cancelled')
        self.release['hold'].set()

    def test_a_parked_requests_running_job_stopped_cancels_the_request(self):
        started = threading.Event()

        def worker(payload, job):
            started.set()
            while True:
                job.raise_if_cancelled()
                time.sleep(0.005)

        def run(viewer, args):
            self.scheduler.enqueue('test', 'test x', 'x', worker, str(Path(self.directory.name) / 'x.out'))
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            request_id = self.portal.submit('one', 'test x')['request_id']
            self.settle(started.is_set, 'the job never ran')
            self.assertTrue(self.scheduler.cancel(1))
            result = self.finish(request_id)

        self.assertEqual(result['status'], 'cancelled', result)

    def test_a_job_that_does_not_release_its_request_keeps_the_head_as_before(self):
        self.command(releases=False)
        first = self.portal.submit('one', 'test a')['request_id']
        second = self.portal.submit('two', 'test b')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle()

        self.assertEqual((self.status(first), self.status(second)), ('running', 'queued'))
        self.assertNotIn('b', self.begun)
        self.release['a'].set()
        self.assertEqual(self.finish(first)['status'], 'succeeded')
        self.settle(self.running('b'), 'the second never ran')
        self.release['b'].set()
        self.assertEqual(self.finish(second)['status'], 'succeeded')

    def test_a_job_that_does_not_declare_the_release_holds_the_head(self):
        def run(viewer, args):
            context = CURRENT.get()
            context.add_job('queued-1', command='print')
            context.job_event('queued-1', 'running')
            self.held = context
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            first = self.portal.submit('one', 'test a')['request_id']
            second = self.portal.submit('two', 'select "one"')['request_id']
            self.idle(30)
            self.assertEqual((self.status(first), self.status(second)), ('running', 'queued'))
            self.held.job_event('queued-1', 'succeeded')
            self.assertEqual(self.finish(first)['status'], 'succeeded')
            self.assertEqual(self.finish(second)['status'], 'succeeded')

    def test_work_in_the_command_queue_keeps_the_head_and_is_no_job(self):
        # The command queue's entry is the command's own `queue`, not an entry of `jobs`: it has no number.
        def run(viewer, args):
            context = CURRENT.get()
            context.join_queue()
            context.queue_event('running')
            self.held = context
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            first = self.portal.submit('one', 'test a')['request_id']
            second = self.portal.submit('two', 'select "one"')['request_id']
            self.idle(30)
            self.assertEqual((self.status(first), self.status(second)), ('running', 'queued'))
            command = self.portal.get(first)['commands'][0]
            self.assertEqual((command['jobs'], command['queue']['status']), ([], 'running'))
            self.held.queue_place(1)
            self.held.queue_progress('pass 2 of 3')
            self.assertEqual(self.portal.get(first)['commands'][0]['queue']['position'], 1)
            self.assertEqual(self.portal.get(first)['commands'][0]['queue']['progress'], 'pass 2 of 3')
            self.held.queue_event('succeeded')
            self.assertEqual(self.finish(first)['status'], 'succeeded')
            self.assertEqual(self.finish(second)['status'], 'succeeded')
            self.assertEqual(self.portal.get(first)['commands'][0]['queue']['status'], 'succeeded')

    def test_a_command_the_queue_cancels_ends_cancelled_with_its_message(self):
        def run(viewer, args):
            context = CURRENT.get()
            context.join_queue()
            self.held = context
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            first = self.portal.submit('one', 'test a')['request_id']
            self.idle(30)
            self.assertEqual(self.status(first), 'running')
            self.held.queue_event('cancelled', 'Cancelled by undo before it finished.')
            result = self.finish(first)
            self.assertEqual(result['status'], 'cancelled', result)
            self.assertEqual(result['commands'][0]['queue']['message'], 'Cancelled by undo before it finished.')
            self.assertEqual(result['commands'][0]['messages'][-1]['text'], 'Cancelled by undo before it finished.')

    def test_a_request_with_work_awaiting_the_user_keeps_the_head(self):
        def run(viewer, args):
            context = CURRENT.get()
            context.add_job('ask-1', releases_request=True)
            context.job_event('ask-1', 'awaiting_user_input')
            self.held = context
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            first = self.portal.submit('one', 'test a')['request_id']
            second = self.portal.submit('two', 'select "one"')['request_id']
            self.idle(30)
            self.assertEqual((self.status(first), self.status(second)), ('awaiting_user_input', 'queued'))
            self.held.job_event('ask-1', 'succeeded')
            self.assertEqual(self.finish(first)['status'], 'succeeded')
            self.assertEqual(self.finish(second)['status'], 'succeeded')

    def test_a_request_with_more_commands_after_the_job_keeps_the_head(self):
        self.command()
        first = self.portal.submit('one', ['test a', 'select "two"'])['request_id']
        second = self.portal.submit('two', 'select "one"')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle()
        self.assertEqual((self.status(first), self.status(second)), ('running', 'queued'))
        self.release['a'].set()
        self.assertEqual(self.finish(first)['status'], 'succeeded')
        self.assertEqual(self.finish(second)['status'], 'succeeded')

    def test_a_command_with_children_still_to_run_keeps_the_head(self):
        # A script's commands run after its job; the next request must not slip in between.
        def run(viewer, args):
            context = CURRENT.get()
            self.begun['a'], self.release['a'] = threading.Event(), threading.Event()
            self.scheduler.enqueue('test', 'test a', 'a', self.worker, str(Path(self.directory.name) / 'a.out'))
            context.children(['select "two"'])
            ce.command_succeeded(viewer)

        with fake_test_command(run):
            first = self.portal.submit('one', 'test a')['request_id']
            second = self.portal.submit('two', 'select "one"')['request_id']
            self.settle(self.running('a'), 'the job never ran')
            self.idle()
            self.assertEqual((self.status(first), self.status(second)), ('running', 'queued'))
            self.release['a'].set()
            self.assertEqual(self.finish(first)['status'], 'succeeded')
            self.assertEqual(self.finish(second)['status'], 'succeeded')

    def test_closing_the_portal_cancels_a_parked_request(self):
        self.command()
        request_id = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle(10)
        self.assertEqual(self.status(request_id), 'running')
        self.portal.shutdown()
        self.assertEqual(self.status(request_id), 'cancelled')
        self.release['a'].set()



    # --- An Agent reply's commands that need approval, beside requests that left the head of the line ---

    def save_stand_in(self):
        """Replace `save`, a command that needs approval, with one that records its arguments."""
        calls = []

        def run(viewer, args):
            calls.append(args)
            ce.command_succeeded(viewer, 'save stand-in ran.')

        stack = fake_test_command(run, 'save')
        stack.start()
        self.addCleanup(stack.stop)
        return calls

    def test_a_reply_behind_a_parked_request_is_refused_at_once_where_nobody_can_approve_it(self):
        saves = self.save_stand_in()
        self.command()
        self.viewer.headless = True
        parked = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle(10)
        self.assertEqual((self.status(parked), self.portal.queue), ('running', []))

        reply = self.portal.submit('two', ['select "one"', 'save x.h5', 'select "two"'], source='web_agent')['request_id']
        result = self.finish(reply)
        self.assertEqual(result['status'], 'failed', result)
        self.assertEqual([(c['status'], c.get('approval')) for c in result['commands']],
                         [('succeeded', None), ('failed', 'refused'), ('skipped', None)])
        self.assertIn('nobody can', result['commands'][1]['messages'][-1]['text'])
        self.assertEqual(saves, [])
        # The parked request's job ran on, and the line is free for what comes next.
        self.assertEqual(self.status(parked), 'running')
        self.assertEqual(self.finish(self.portal.submit('three', 'select "two"')['request_id'])['status'], 'succeeded')
        self.release['a'].set()
        self.assertEqual(self.finish(parked)['status'], 'succeeded')

    def test_a_reply_behind_a_parked_request_waits_for_approval_where_it_can_be_given(self):
        saves = self.save_stand_in()
        self.command()
        self.viewer.headless = True
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: client == 'agent')
        announced = []
        self.portal.approval_needed.connect(lambda request_id, command_id: announced.append(command_id))
        parked = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        self.idle(10)

        reply = self.portal.submit('two', ['select "one"', 'save x.h5'], source='web_agent')['request_id']
        self.settle(lambda: self.status(reply) == 'awaiting_user_input', 'the command never waited for approval')
        waiting = self.portal.get(reply)['commands'][1]
        self.assertEqual((waiting['status'], waiting['approval']), ('awaiting_user_input', 'pending'))
        self.assertEqual(announced, [waiting['command_id']])
        self.assertEqual(self.status(parked), 'running')

        # The parked request ending changes nothing for the one waiting, which is decided as ever.
        self.release['a'].set()
        self.assertEqual(self.finish(parked)['status'], 'succeeded')
        self.idle(10)
        self.assertEqual((self.status(reply), saves, announced), ('awaiting_user_input', [], [waiting['command_id']]))
        self.portal.decide(reply, waiting['command_id'], True)
        self.assertEqual(self.finish(reply)['status'], 'succeeded')
        self.assertEqual(saves, [['x.h5']])

    def test_a_command_approved_into_a_background_job_frees_the_line_for_the_next_reply(self):
        saves = self.save_stand_in()
        self.command()
        self.viewer.shows_windows = lambda: True
        # `test` is no command of the catalog, so a reply's `test a` waits for approval; once it is
        # run, its job holds only its own request, which leaves the head of the line.
        first = self.portal.submit('one', 'test a', source='web_agent')['request_id']
        self.settle(lambda: self.status(first) == 'awaiting_user_input', 'the first command never waited')
        self.portal.decide(first, self.portal.get(first)['commands'][0]['command_id'], True)
        self.settle(self.running('a'), 'the approved job never ran')
        self.idle(10)
        self.assertEqual((self.status(first), self.portal.queue), ('running', []))

        second = self.portal.submit('two', 'save y.h5', source='web_agent')['request_id']
        self.settle(lambda: self.status(second) == 'awaiting_user_input', 'the second command never waited')
        self.portal.decide(second, self.portal.get(second)['commands'][0]['command_id'], True)
        self.assertEqual(self.finish(second)['status'], 'succeeded')
        self.assertEqual((saves, self.status(first)), ([['y.h5']], 'running'))
        self.release['a'].set()
        self.assertEqual(self.finish(first)['status'], 'succeeded')

    def test_a_stopped_replys_waiting_command_is_discarded_beside_a_parked_request(self):
        self.save_stand_in()
        self.command()
        self.viewer.shows_windows = lambda: True
        parked = self.portal.submit('one', 'test a')['request_id']
        self.settle(self.running('a'), 'the job never ran')
        reply = self.portal.submit('two', 'save x.h5', source='web_agent')['request_id']
        self.settle(lambda: self.status(reply) == 'awaiting_user_input', 'the command never waited')
        self.portal.discard_pending(reply)
        self.assertEqual(self.finish(reply)['status'], 'cancelled')
        # Discarding the reply's command leaves the parked request's job alone.
        self.assertEqual(self.status(parked), 'running')
        self.assertFalse(self.release['a'].is_set())
        self.release['a'].set()
        self.assertEqual(self.finish(parked)['status'], 'succeeded')

if __name__ == '__main__': unittest.main()
