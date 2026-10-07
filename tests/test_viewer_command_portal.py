"""Viewer command portal (Viewer_Command_Portal) and command worker tracking (Viewer_Worker_Tracking).

Covers equivalence with manual commands, completion, isolation, retries and
lookups, bounded feedback, submission limits, shutdown, and failures of
detached workers and command scripts.
"""
import importlib
import io
import json
import os
from pathlib import Path
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
from tests.viewer_fixtures import PortalFixture, Viewer
from Background_Job_Scheduler import BackgroundJobScheduler
from Viewer_Visual_State import edge_stages, capture_view
from desktop.Command_Metadata import get_command_metadata


def fake_test_command(run):
    """Resolve the command `test` to a module whose run() is ``run``.

    Every other import_module call, including the real command modules the
    Command_Engine imports for the commands that follow, is left unchanged.
    """
    real_import = importlib.import_module

    def import_module(name, package=None):
        if name == 'commands.test':
            return SimpleNamespace(run=run)
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
        self.addCleanup(scheduler.shutdown)
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
        cfg = SimpleNamespace(UMAP_MODE=True, LOW_RESOURCE_MODE=False)
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
                    mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': platform}):
                self.viewer.canvas = failing(message)
                with self.assertRaises(ValueError) as caught:
                    capture_view(self.viewer)
                self.assertTrue(str(caught.exception).startswith(f'Viewer canvas capture failed: {message}'))
                self.assertEqual('relaunch it in normal mode' in str(caught.exception), hinted)

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


if __name__ == '__main__': unittest.main()
