"""In-Viewer agent backend (web_ui/agent_backend): model-card activation, loading and saving."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from web_ui import agent_backend as agent


class CardActivationTests(unittest.TestCase):
    def activate(self, url, model=''):
        """Activate a card with urlopen mocked and return (viewer, probed URLs).

        Like a real OpenAI-compatible server, the mock serves /models on the
        API base only and answers 404 everywhere else."""
        probed = []

        def urlopen(request, timeout=None):
            probed.append(request.full_url)
            if request.full_url != 'http://localhost:1234/v1/models':
                raise urllib.error.HTTPError(request.full_url, 404, 'Not Found', {}, None)
            response = mock.MagicMock()
            response.__enter__.return_value.status = 200
            response.__enter__.return_value.read.return_value = json.dumps({'data': [{'id': 'qwen3-8b'}]}).encode()
            return response

        viewer = SimpleNamespace()
        card = {'name': 'LM Studio', 'url': url, 'model': model}
        with mock.patch.object(agent.urllib.request, 'urlopen', side_effect=urlopen):
            self.assertTrue(agent.activate_agent_from_card(viewer, card, quiet=True))
        return viewer, probed

    def test_blank_model_is_probed_on_the_api_base(self):
        for url in ('http://localhost:1234/v1/chat/completions', 'http://localhost:1234/v1/chat/completions/',
                    'http://localhost:1234/v1', 'http://localhost:1234/v1/'):
            with self.subTest(url=url):
                viewer, probed = self.activate(url)
                self.assertEqual(probed, ['http://localhost:1234/v1/models'])
                self.assertEqual(viewer.llm_url, 'http://localhost:1234/v1')
                self.assertEqual(viewer.llm_model_name, 'qwen3-8b')

    def test_named_model_is_not_probed(self):
        viewer, probed = self.activate('http://localhost:1234/v1/chat/completions', model='my-model')
        self.assertEqual(probed, [])
        self.assertEqual((viewer.llm_url, viewer.llm_model_name), ('http://localhost:1234/v1', 'my-model'))


class ModelCardTests(unittest.TestCase):
    """model_card.json holds the user's cards and API keys; no save may wipe them.

    _SRC_DIR points at a temporary tree, so the real file is never read.
    """
    SAVED = {'cards': [{'id': 'mine', 'name': 'My API', 'url': 'https://api.example/v1',
                        'model': 'big', 'api_key': 'sk-SECRET', 'temperature': 0.2}],
             'note': 'kept'}
    DEFAULT_IDS = ['ollama', 'lmstudio', 'llamacpp']

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name, 'resources', 'agent')
        self.folder.mkdir(parents=True)
        self.path = self.folder / 'model_card.json'
        patcher = mock.patch.object(agent, '_SRC_DIR', directory.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def viewer():
        viewer = SimpleNamespace(events=[])
        viewer.broadcast_event = viewer.events.append
        return viewer

    def save(self, data):
        """Run the Agent page's save action; return (broadcast events, terminal output)."""
        viewer, output = self.viewer(), io.StringIO()
        with redirect_stdout(output):
            agent.handle_save_model_cards(viewer, {'action': 'save_model_cards', **data})
        return viewer.events, output.getvalue()

    def test_saving_replaces_the_cards_and_keeps_other_keys(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        cards = [{'id': 'new', 'name': 'Local', 'url': 'http://localhost:1234/v1', 'api_key': ''}]

        events, _ = self.save({'cards': cards, 'save_id': 's1'})

        self.assertEqual(events, [{'type': 'model_cards_saved', 'save_id': 's1'}])
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), {'cards': cards, 'note': 'kept'})
        self.assertEqual(agent.load_model_cards(), cards)
        self.assertEqual(os.listdir(self.folder), ['model_card.json'])

    def test_without_saved_cards_the_defaults_are_loaded_and_a_save_creates_the_file(self):
        self.assertEqual([card['id'] for card in agent.load_model_cards()], self.DEFAULT_IDS)
        for document in ({}, {'cards': []}):
            with self.subTest(document=document):
                self.path.write_text(json.dumps(document), encoding='utf-8')
                self.assertEqual([card['id'] for card in agent.load_model_cards()], self.DEFAULT_IDS)

        self.path.unlink()
        events, _ = self.save({'cards': [{'id': 'new', 'name': 'Local'}], 'save_id': 'first'})
        self.assertEqual(events, [{'type': 'model_cards_saved', 'save_id': 'first'}])
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), {'cards': [{'id': 'new', 'name': 'Local'}]})

    def test_a_request_without_cards_to_save_is_refused(self):
        # A request without "cards" used to save {"cards": []}, wiping every card.
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        before = self.path.read_bytes()
        for request in ({}, {'cards': None}, {'cards': []}, {'cards': 'My API'},
                        {'cards': {'0': {'name': 'My API'}}}, {'cards': ['My API']}):
            with self.subTest(request=request):
                events, output = self.save({**request, 'save_id': 's2'})

                self.assertEqual(self.path.read_bytes(), before)
                [event] = events
                self.assertEqual((event['type'], event['save_id']), ('model_cards_error', 's2'))
                self.assertIn('"cards" list of objects', event['error'])
                self.assertIn(event['error'], output)

    def test_a_file_that_cannot_be_loaded_is_kept_and_reported(self):
        unusable = {
            'trailing comma': (b'{"cards": [{"name": "My API", "api_key": "sk-SECRET"},]}',
                               r'is not valid JSON \(.*line 1 column \d+'),
            'unquoted value': (b'{"cards": [{"name": "My API", "api_key": sk-SECRET}]}', 'is not valid JSON'),
            'array root': (b'[{"name": "My API", "api_key": "sk-SECRET"}]', 'must contain a JSON object'),
            'cards not a list': (b'{"cards": {"0": {"api_key": "sk-SECRET"}}}', 'must contain a JSON object'),
            'not UTF-8': (b'{"cards": [], "api_key": "sk-SECRET\xff"}', 'could not be read'),
        }
        for label, (content, reason) in unusable.items():
            with self.subTest(label):
                self.path.write_bytes(content)
                events, output = self.save({'cards': [{'id': 'new', 'name': 'Local'}], 'save_id': 's3'})
                checker = self.viewer()
                agent.handle_check_model_cards(checker, {'action': 'check_model_cards'})

                self.assertEqual(self.path.read_bytes(), content)
                self.assertEqual(os.listdir(self.folder), ['model_card.json'])
                [event] = events
                self.assertEqual(event['type'], 'model_cards_error')
                self.assertRegex(event['error'], reason)
                self.assertIn('left unchanged', event['error'])
                self.assertIn(event['error'], output)
                [check] = checker.events
                self.assertEqual(check['type'], 'model_cards_error')
                self.assertRegex(check['error'], reason)
                for message in (event['error'], check['error']):
                    self.assertNotIn('SECRET', message)
                with self.assertRaisesRegex(agent.ModelCardsError, reason):
                    agent.load_model_cards()

        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        checker = self.viewer()
        agent.handle_check_model_cards(checker, {'action': 'check_model_cards'})
        self.assertEqual(checker.events, [])

    def test_a_failed_write_keeps_the_previous_file(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        before = self.path.read_bytes()

        def fail_midway(document, handle, **options):
            handle.write('{"cards": ')
            raise OSError(28, 'No space left on device')

        with mock.patch.object(json, 'dump', side_effect=fail_midway):
            events, _ = self.save({'cards': [{'id': 'new', 'name': 'Local'}], 'save_id': 's4'})

        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(os.listdir(self.folder), ['model_card.json'])
        [event] = events
        self.assertEqual(event['type'], 'model_cards_error')
        self.assertIn('No space left on device', event['error'])

    def interrupted_save(self):
        """Stop a save as a hard kill would, between writing its copy and the
        rename, and return the copy it leaves behind (API keys included)."""
        class Killed(BaseException):
            pass

        created = []
        mkstemp = tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            descriptor, name = mkstemp(*args, **kwargs)
            created.append(Path(name))
            return descriptor, name

        with mock.patch.object(tempfile, 'mkstemp', recording_mkstemp), \
                mock.patch.object(os, 'replace', side_effect=Killed), \
                mock.patch.object(os, 'unlink'), self.assertRaises(Killed):
            agent.save_model_cards([{'id': 'new', 'name': 'Local', 'api_key': 'sk-SECRET'}])
        [leftover] = created
        self.assertTrue(leftover.is_file())
        return leftover

    def test_an_interrupted_save_leaves_its_copy_only_until_it_is_stale(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        leftover = self.interrupted_save()

        # A recent copy may belong to a save still in progress, so it stays.
        self.assertEqual(agent.load_model_cards(), self.SAVED['cards'])
        self.assertTrue(leftover.exists())

        stale = time.time() - 2 * agent._STALE_PARTIAL_SECONDS
        os.utime(leftover, (stale, stale))
        with mock.patch('builtins.open', wraps=open) as opened:
            self.assertEqual(agent.load_model_cards(), self.SAVED['cards'])
        self.assertFalse(leftover.exists())
        self.assertNotIn(leftover.name, str(opened.call_args_list))

        leftover = self.interrupted_save()
        os.utime(leftover, (stale, stale))
        events, _ = self.save({'cards': [{'id': 'new', 'name': 'Local'}], 'save_id': 's5'})
        self.assertEqual(events, [{'type': 'model_cards_saved', 'save_id': 's5'}])
        self.assertEqual(os.listdir(self.folder), ['model_card.json'])

    def test_an_interrupted_save_copy_cannot_be_committed(self):
        relative = f'src/resources/agent/{self.interrupted_save().name}'
        try:
            result = subprocess.run(['git', '-C', str(ROOT), 'check-ignore', '--no-index', '--quiet', '--', relative],
                                    capture_output=True)
        except OSError:
            self.skipTest('git is not installed')
        if result.returncode not in (0, 1):
            self.skipTest('not a git checkout')
        self.assertEqual(result.returncode, 0, f'{relative} is not git-ignored')

    def test_agent_command_reports_an_unusable_file_when_it_needs_the_cards(self):
        from commands import agent as agent_command
        self.path.write_text('{"cards": [{"name": "My API"},]}', encoding='utf-8')
        engine = agent_command.Command_Engine

        def run(args, llm_loaded):
            viewer = SimpleNamespace(llm_loaded=llm_loaded)
            with mock.patch.object(agent_command, 'register'), \
                    mock.patch.object(agent_command, 'activate_agent_from_card') as activate, \
                    mock.patch.object(agent_command, 'run_web_agent_query') as query, \
                    mock.patch.object(engine, 'print_help'), \
                    mock.patch.object(engine, 'command_failed') as failed, \
                    mock.patch.object(engine, 'command_succeeded') as succeeded:
                agent_command.run(viewer, args)
            return activate, query, failed, succeeded

        for args, llm_loaded in ((['<My', 'API>'], True), (['hello'], False)):
            with self.subTest(args=args, llm_loaded=llm_loaded):
                activate, query, failed, succeeded = run(args, llm_loaded)
                activate.assert_not_called()
                query.assert_not_called()
                succeeded.assert_not_called()
                self.assertIn('is not valid JSON', failed.call_args.args[1])

        # A running agent needs no cards to take a message.
        activate, query, failed, succeeded = run(['hello'], True)
        query.assert_called_once()
        failed.assert_not_called()
        succeeded.assert_called_once()

    def test_terminal_activation_reports_an_unusable_file(self):
        self.path.write_text('[1]', encoding='utf-8')
        with mock.patch.object(agent.Command_Engine, 'print_help') as print_help, \
                mock.patch.object(agent, 'activate_agent_from_card') as activate:
            self.assertFalse(agent.activate_agent(SimpleNamespace(llm_loaded=False)))
        activate.assert_not_called()
        self.assertIn('must contain a JSON object', print_help.call_args.args[1])
