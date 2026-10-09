"""Outcome summaries, help and artifacts that Viewer commands report through the real command portal."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from tests.viewer_fixtures import PortalFixture
from desktop.Viewer_Inspection import command_catalog
from Viewer_Command_Portal import ExecutionContext
from Viewer_Command_Portal import CURRENT


class FeedbackTests(PortalFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        import EMAPSSN_Config as cfg
        # Some commands create their output folders before anything else, even
        # for `help`; the defaults are relative to the working directory.
        for name, folder in (('METADATA_DIR', 'metadata'), ('ANALYSIS_RESULT_DIR', 'results')):
            patcher = mock.patch.object(cfg, name, str(Path(self.directory.name) / folder))
            patcher.start()
            self.addCleanup(patcher.stop)

    def execute(self, command, expected='succeeded'):
        with redirect_stdout(io.StringIO()):
            result = self.finish(self.portal.submit(str(len(self.portal.requests)), command)['request_id'])
        self.assertEqual(result['status'], expected, result)
        return result['commands'][0]

    def success_text(self, record):
        summaries = [m['text'] for m in record['messages'] if m['status'] == 'succeeded']
        self.assertEqual(len(summaries), 1, record)
        return summaries[0]

    def test_every_catalogued_command_reports_help_success(self):
        self.viewer.alignment = SimpleNamespace(aln=[object()], has_reference=True)
        with mock.patch('web_ui.agent_backend.register'), mock.patch('web_ui.meta_backend.register'):
            for entry in command_catalog()['commands']:
                with self.subTest(command=entry['command']):
                    record = self.execute(entry['command'] + ' help')
                    self.assertEqual(self.success_text(record), 'Help information printed to the terminal.')
                    self.assertEqual(len(record['messages']), 1, record)

    def test_reset_reports_actual_targets_and_retry_does_not_repeat(self):
        self.viewer.current_colors[:] = 0
        record = self.execute('reset colors shapes order hide')
        self.assertEqual(self.success_text(record), 'Reset successful: colors, shapes, node order, hidden.')
        np.testing.assert_array_equal(self.viewer.current_shapes, ['disc'] * 3)
        np.testing.assert_array_equal(self.viewer.visible_mask, [True] * 3)
        saved = self.viewer.saved
        retried = self.portal.submit('0', 'reset colors shapes order hide')
        self.assertEqual(retried['commands'][0]['messages'], record['messages'])
        self.assertEqual(self.viewer.saved, saved)

    def test_visual_summaries_and_no_matches(self):
        for command, expected in [
            ('color "one" red', 'Applied: 1 node (red)'),
            ('color "absent" red', 'No nodes matched'),
            ('select "one"', 'Selected 1'),
            ('hide "two"', 'Hidden 1'),
            ('group "one" example', "Groups Applied: 1 node -> 'example'"),
            ('group "absent" example', 'No nodes matched'),
            ('group list', 'Listed 1'),
            ('cluster list', 'Listed 2'),
            ('subcluster clear', 'No subcluster groups to clear'),
        ]:
            with self.subTest(command=command):
                record = self.execute(command)
                self.assertIn(expected, self.success_text(record))
                texts = [m['text'] for m in record['messages']]
                self.assertEqual(len(texts), len(set(texts)), record)

    def test_failures_never_append_success_messages(self):
        for command in ('color', 'select', 'select {Missing=1}', 'query', 'logo',
                        'label', 'run', 'alignment', 'reference absent', 'zoom invalid'):
            with self.subTest(command=command):
                record = self.execute(command, 'failed')
                self.assertFalse(any(m['status'] == 'succeeded' for m in record['messages']), record)

    def test_undo_redo_distinguish_noop_from_change(self):
        for name in ('undo', 'redo'):
            for changed in (True, False):
                setattr(self.viewer, '_do_' + name, mock.Mock(return_value=changed))
                summary = self.success_text(self.execute(name))
                self.assertEqual(summary, f'{name.title()} successful.' if changed else f'Nothing to {name}.')

    def test_interfaces_and_registration(self):
        self.viewer.open_agent_ui = mock.Mock()
        self.viewer.open_metadata_ui = mock.Mock()
        with mock.patch('web_ui.agent_backend.register') as agent_register, \
                mock.patch('web_ui.meta_backend.register') as meta_register, \
                mock.patch('web_ui.esmfold_backend.register') as esmfold_register:
            self.assertIn('Opened the Agent interface', self.success_text(self.execute('agent')))
            self.assertIn('Opened the metadata interface', self.success_text(self.execute('meta')))
            self.assertIn('Registered', self.success_text(self.execute('agent --register-only')))
            self.assertIn('Registered', self.success_text(self.execute('meta --register-only')))
            self.assertIn('Registered', self.success_text(self.execute('esmfold --register-only')))
        # The stand-ins, not the real registrations, were reached.
        self.assertEqual((agent_register.call_count, meta_register.call_count, esmfold_register.call_count),
                         (2, 2, 1))

    def test_metadata_file_summary_and_artifact(self):
        self.viewer.metadata = {'Example': {'type': 'number', 'values': np.arange(3)}}
        target = Path(self.directory.name) / 'metadata.csv'
        with mock.patch('EMAPSSN_Config.METADATA_DIR', self.directory.name, create=True):
            record = self.execute('meta download metadata.csv')
            self.assertIn(str(target), self.success_text(record))
            self.assertEqual(record['artifacts'], [str(target)])
            self.assertTrue(target.is_file())
            self.viewer.metadata = {}
            record = self.execute('meta download metadata.csv', 'failed')
        self.assertFalse(any(m['status'] == 'succeeded' for m in record['messages']))

    def test_metadata_download_refuses_a_path(self):
        """A portal command (web agent, MCP) cannot write outside the metadata directory."""
        self.viewer.metadata = {'Example': {'type': 'number', 'values': np.arange(3)}}
        target = Path(self.directory.name) / 'escaped.csv'
        meta_dir = Path(self.directory.name) / 'Meta_Data'
        with mock.patch('EMAPSSN_Config.METADATA_DIR', str(meta_dir), create=True):
            for name in (str(target), r'..\escaped.csv'):
                with self.subTest(name=name):
                    record = self.execute(f'meta download {name}', 'failed')
                    self.assertEqual(record['artifacts'], [])
                    self.assertIn('path separators', record['messages'][-1]['text'])
        self.assertFalse(target.exists())
        self.assertEqual(list(meta_dir.iterdir()), [])

    def test_summary_promotion_keeps_artifacts_and_bounds(self):
        record = self.portal.new_command('test')
        context = ExecutionContext(self.portal, 'test', record)
        context.report(message='Saved output.')
        context.report('succeeded', 'Saved output.', Path(self.directory.name) / 'out')
        self.assertEqual(len(record['messages']), 1)
        self.assertEqual(len(record['artifacts']), 1)
        context.report('succeeded', 'x' * 3000)
        self.assertTrue(record['messages'][-1]['truncated'])
        self.assertEqual(len(record['messages'][-1]['text']), 2048)

    def test_label_and_logo_submission_remains_pending_until_job_completion(self):
        from tests.sparse_alignment import load_manager, write_fasta
        import EMAPSSN_Config as cfg
        msa = str(Path(self.directory.name) / 'alignment.fasta')
        write_fasta(msa, [('one', 'AC'), ('two', 'AC'), ('three', 'AD')])
        self.viewer.alignment = load_manager(msa, self.viewer.full_headers, 'one')
        self.viewer.active_reference = 'one'
        self.viewer.alignment_offset = 0
        contexts = []
        def enqueue(**kwargs):
            context = CURRENT.get()
            context.add_job('test-job', output_path=kwargs['output_path'])
            contexts.append(context)
        self.viewer.background_job_scheduler = SimpleNamespace(
            is_output_path_reserved=lambda path: False, enqueue=enqueue)
        with mock.patch.object(cfg, 'resolve_directory_path', return_value=self.directory.name):
            for command in ('logo [1] result.svg', 'label clusters result.xlsx'):
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    request = self.portal.submit(command, command)
                    self.portal.pump()
                    pending = self.portal.get(request['request_id'])
                    self.assertEqual(pending['status'], 'running', pending)
                    self.assertTrue(pending['complete'])
                    self.assertIn('Queued', self.success_text(pending['commands'][0]))
                    contexts[-1].job_event('test-job', 'succeeded', 'Artifact generated.', [])
                    result = self.finish(request['request_id'])
                    self.assertEqual(result['status'], 'succeeded', result)


if __name__ == '__main__':
    unittest.main()
