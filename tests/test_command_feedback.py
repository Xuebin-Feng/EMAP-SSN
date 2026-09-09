"""Agent-facing help and outcome summaries through the real command portal."""
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from tests import test_viewer_command_portal as fixtures
from desktop.Viewer_Inspection import command_catalog, _command_syntax
from Viewer_Command_Portal import ExecutionContext
from Viewer_Command_Portal import CURRENT


class CatalogTests(unittest.TestCase):
    def test_usage_sections_do_not_collect_examples_or_prose(self):
        text = '''Usage: reset colors
                  reset sizes
                  reset colors

Description:
  reset is reversible
Examples:
  reset shapes
Usage:
reset hide
'''
        self.assertEqual(_command_syntax(text, 'reset'),
                         ['reset colors', 'reset sizes', 'reset hide'])
        self.assertEqual(_command_syntax('No usage available.', 'reset'), [])
        self.assertEqual(_command_syntax('Usage:\n  <target>', 'reset'), [])

    def test_catalog_is_read_only_and_preserves_detailed_help(self):
        with mock.patch('Command_Engine.execute_command', side_effect=AssertionError('executed')):
            general = command_catalog()
            detailed = command_catalog('reset')['commands'][0]
        self.assertEqual(len(general['commands']), 24)
        self.assertTrue(all(c['help'] is None and isinstance(c['syntax'], list)
                            for c in general['commands']))
        self.assertIn('reset <TARGET_1> [TARGET_2] ...', detailed['syntax'])
        for target in ('colors', 'sizes', 'shapes', 'clusters', 'groups', 'hide', 'network', 'order, layer'):
            self.assertIn(target, detailed['help'])
        self.assertIn('arguments={"command":"reset"}', general['note'])
        with self.assertRaisesRegex(ValueError, 'Unknown command'):
            command_catalog('missing')


class FeedbackTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.PortalTests.setUpClass.__func__)
    setUp = fixtures.PortalTests.setUp
    finish = fixtures.PortalTests.finish

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
        with mock.patch('commands.agent.register'), mock.patch('commands.meta.register'):
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
            ('color "one" red', '1 nodes'),
            ('color "absent" red', 'No nodes matched'),
            ('select "one"', 'Selected 1'),
            ('hide "two"', 'Hidden 1'),
            ('group "one" example', '1 nodes'),
            ('group "absent" example', 'No nodes matched'),
            ('group list', 'Listed 1'),
            ('cluster list', 'Listed 2'),
            ('subcluster clear', 'Cleared all subcluster groups'),
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
        with mock.patch('commands.agent.register'), mock.patch('commands.meta.register'), \
                mock.patch('commands.esmfold.esmfold_backend.register'):
            self.assertIn('Opened the Agent interface', self.success_text(self.execute('agent')))
            self.assertIn('Opened the metadata interface', self.success_text(self.execute('meta')))
            self.assertIn('Registered', self.success_text(self.execute('agent --register-only')))
            self.assertIn('Registered', self.success_text(self.execute('meta --register-only')))
            self.assertIn('Registered', self.success_text(self.execute('esmfold --register-only')))

    def test_metadata_file_summary_and_artifact(self):
        self.viewer.metadata = {'Example': {'type': 'number', 'values': np.arange(3)}}
        target = Path(self.directory.name) / 'metadata.csv'
        record = self.execute(f'meta download {target}')
        self.assertIn(str(target), self.success_text(record))
        self.assertEqual(record['artifacts'], [str(target)])
        self.assertTrue(target.is_file())
        self.viewer.metadata = {}
        record = self.execute(f'meta download {target}', 'failed')
        self.assertFalse(any(m['status'] == 'succeeded' for m in record['messages']))

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
        from tests.test_incomplete_alignment_commands import load_manager, write_fasta
        import EMAPSSN_Config as cfg
        import Cache_Manifest as cache_manifest
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
        with mock.patch.object(cfg, 'resolve_directory_path', return_value=self.directory.name), \
                mock.patch.object(cache_manifest, 'validate_network_schema', return_value={}):
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
