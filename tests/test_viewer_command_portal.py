"""Command equivalence, completion, isolation, and bounded Viewer feedback."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PySide6 import QtWidgets
import Command_Engine as ce
from Viewer_Command_Portal import ViewerCommandPortal, CURRENT, bind, user_interaction
from Background_Job_Scheduler import BackgroundJobScheduler
from Viewer_Visual_State import edge_stages, capture_view


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


class PortalTests(unittest.TestCase):
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

    def finish(self, request_id):
        deadline = time.monotonic() + 5
        while self.portal.get(request_id)['status'] not in {'succeeded', 'failed', 'cancelled'} and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        return self.portal.get(request_id)

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
        with mock.patch('importlib.import_module', return_value=SimpleNamespace(run=run)):
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
        with mock.patch('importlib.import_module', return_value=SimpleNamespace(run=run)):
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
        with mock.patch('importlib.import_module', return_value=SimpleNamespace(run=lambda v,a: None)):
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

    def test_visible_dialog_reports_waiting_then_cancelled(self):
        def choose(*args):
            self.assertEqual(self.portal.get(request_id)['status'], 'awaiting_user_input')
            return '', ''
        with mock.patch.dict(os.environ, {'SSN_VIEWER_HEADLESS': '0', 'QT_QPA_PLATFORM': ''}), mock.patch.object(QtWidgets.QFileDialog, 'getOpenFileName', side_effect=choose):
            self.viewer.canvas = SimpleNamespace(native=None)
            request_id = self.portal.submit('visible-dialog', 'alignment')['request_id']
            result = self.finish(request_id)
        self.assertEqual(result['status'], 'cancelled', result)

    def test_worker_final_result_and_unexpected_exit(self):
        from Viewer_Command_Portal import ExecutionContext
        from Viewer_Worker_Tracking import WorkerTracker
        import psutil
        request_id = self.portal.submit('worker-status', 'test')['request_id']
        record = self.portal.requests[request_id]['commands'][0]
        record.update(dispatched=True, outcome='succeeded', status='running')
        context = ExecutionContext(self.portal, request_id, record)
        tracker = WorkerTracker(context)
        tracker.timer.stop()
        tracker.path = Path(self.directory.name) / 'status.json'
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


if __name__ == '__main__': unittest.main()
