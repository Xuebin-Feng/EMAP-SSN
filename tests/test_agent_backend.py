"""In-Viewer agent backend (web_ui/agent_backend): model-card activation, loading and saving,
activation by saved card id, and running or discarding commands an Agent reply left waiting."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PySide6 import QtCore
from web_ui import agent_backend as agent
from web_ui.agent_prompt import PLACEHOLDERS
from tests.agent_fixtures import FakeProvider, Stalled, completion, self_signed_tls
from tests.viewer_fixtures import PortalFixture


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
                self.assertIn('is not valid JSON', str(failed.call_args.args[1]))

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
        self.assertIn('must contain a JSON object', str(print_help.call_args.args[1]))


class SetBackendTests(unittest.TestCase):
    """The Agent page activates only a saved model card, named by its id.

    A request's own card fields never take effect, so no request can point the
    agent at a server the user did not save. _SRC_DIR points at a temporary tree.
    """
    SAVED = {'cards': [{'id': 'mine', 'name': 'My API', 'url': 'http://localhost:11006/v1',
                        'model': 'big', 'api_key': 'sk-SECRET'}]}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        folder = Path(directory.name, 'resources', 'agent')
        folder.mkdir(parents=True)
        self.path = folder / 'model_card.json'
        # Activating a card without a model name probes its server, so none answers.
        for patcher in (mock.patch.object(agent, '_SRC_DIR', directory.name),
                        mock.patch.object(agent.urllib.request, 'urlopen', side_effect=urllib.error.URLError('offline'))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def set_backend(self, data, viewer=None):
        """Run the Agent page's set_backend action; return the viewer and its backend_state event."""
        viewer = viewer or SimpleNamespace()
        viewer.events = []
        viewer.broadcast_event = viewer.events.append
        with redirect_stdout(io.StringIO()):
            agent.handle_set_backend(viewer, {'action': 'set_backend', **data})
        return viewer, [e for e in viewer.events if e['type'] == 'backend_state'][-1]

    def test_a_saved_card_is_activated_by_its_id(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        viewer, event = self.set_backend({'card_id': 'mine', 'card': {
            'id': 'mine', 'url': 'http://attacker.example/v1', 'model': 'theirs', 'api_key': ''}})
        self.assertEqual((viewer.llm_url, viewer.llm_model_name, viewer.llm_api_key),
                         ('http://localhost:11006/v1', 'big', 'sk-SECRET'))
        self.assertTrue(event['llm_loaded'])
        self.assertNotIn('error', event)

    def test_without_saved_cards_a_default_card_is_activated_by_its_id(self):
        viewer, event = self.set_backend({'card_id': 'ollama'})
        self.assertEqual((viewer.llm_url, event['llm_loaded']), ('http://localhost:11434/v1', True))

    def test_a_card_that_is_not_saved_is_never_activated(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        cases = {
            'unsaved id': ({'card_id': 'new'}, 'not saved yet'),
            'a card from an older page': ({'card': {'url': 'http://attacker.example/v1'}}, 'Reload the Agent page'),
            'an id that is not text': ({'card_id': 7}, 'Reload the Agent page'),
        }
        for label, (data, error) in cases.items():
            with self.subTest(label):
                viewer, _ = self.set_backend({'card_id': 'mine'})
                viewer, event = self.set_backend(data, viewer)
                self.assertFalse(event['llm_loaded'])
                self.assertIn(error, event['error'])
                self.assertFalse(hasattr(viewer, 'llm_url'))

    def test_an_unusable_file_is_reported(self):
        self.path.write_text('{"cards": [', encoding='utf-8')
        _, event = self.set_backend({'card_id': 'mine'})
        self.assertFalse(event['llm_loaded'])
        self.assertIn('Model cards could not be loaded', event['error'])

    def test_no_id_deactivates(self):
        self.path.write_text(json.dumps(self.SAVED), encoding='utf-8')
        viewer, _ = self.set_backend({'card_id': 'mine'})
        viewer, event = self.set_backend({'card_id': None}, viewer)
        self.assertFalse(event['llm_loaded'])
        self.assertNotIn('error', event)


class AgentCommandApprovalTests(unittest.TestCase):
    """The page's Run or Discard reaches the portal; an abandoned turn discards its waiting commands."""

    def test_the_decision_reaches_the_portal_and_a_refusal_is_shown(self):
        portal = mock.Mock()
        viewer = SimpleNamespace(events=[], command_portal=portal)
        viewer.broadcast_event = viewer.events.append
        agent.handle_decide_agent_command(viewer, {'action': 'decide_agent_command', 'request_id': 'r', 'command_id': 'c', 'run': True})
        portal.decide.assert_called_once_with('r', 'c', True)
        self.assertEqual(viewer.events, [])

        portal.decide.side_effect = ValueError('That command is not waiting to be run or discarded.')
        agent.handle_decide_agent_command(viewer, {'request_id': 'r', 'command_id': 'c', 'run': False})
        self.assertEqual(viewer.events, [{'type': 'agent_command_error', 'error': 'That command is not waiting to be run or discarded.'}])

    def test_an_abandoned_turn_discards_its_waiting_commands(self):
        portal = mock.Mock()
        viewer = SimpleNamespace(command_portal=portal, _agent_request_id='r1', llm_loaded=False)
        agent.deactivate_agent(viewer, quiet=True)
        portal.discard_pending.assert_called_once_with('r1')
        self.assertIsNone(viewer._agent_request_id)

        # A request already evicted from the portal's history is no error.
        portal.discard_pending.side_effect = ValueError('evicted')
        viewer._agent_request_id = 'r2'
        agent.deactivate_agent(viewer, quiet=True)
        self.assertIsNone(viewer._agent_request_id)


class TurnAcceptanceTests(unittest.TestCase):
    """run_web_agent_query says whether it accepted the turn, and why not."""

    @staticmethod
    def viewer(**attributes):
        viewer = SimpleNamespace(events=[], **{'llm_loaded': True, 'llm_backend': 'server', 'llm_model_name': 'm', **attributes})
        viewer.broadcast_event = viewer.events.append
        return viewer

    def submit(self, viewer, query, **options):
        reasons = []
        accepted = agent.run_web_agent_query(viewer, query, on_rejected=reasons.append, **options)
        return accepted, reasons

    def test_a_turn_that_is_not_accepted_returns_false_with_its_reason(self):
        for label, query, setup, reason in (
            ('busy', 'hello', {'_agent_busy': True}, 'already has an active agent turn'),
            ('not loaded', 'hello', {'llm_loaded': False}, 'LLM is not loaded'),
            ('empty', '', {}, 'Enter a message'),
            ('blank', '  ', {}, 'Enter a message'),
        ):
            with self.subTest(label):
                viewer = self.viewer(**setup)
                accepted, reasons = self.submit(viewer, query)
                self.assertIs(accepted, False)
                self.assertIn(reason, str(reasons[0]))
                # The Agent page is told as before.
                [event] = viewer.events
                self.assertEqual(event['type'], 'agent_error')
                self.assertIn(reason, event['error'])

    def test_other_rejections_return_false(self):
        accepted, reasons = self.submit(self.viewer(), 'hello', attachments='x')
        self.assertIs(accepted, False)
        self.assertIn('at most 10 image attachments', str(reasons[0]))

        accepted, reasons = self.submit(self.viewer(), 'hello', submission_id=7)
        self.assertIs(accepted, False)
        self.assertEqual(str(reasons[0]), 'Invalid submission ID.')

        with mock.patch.object(agent.os.path, 'exists', return_value=False):
            accepted, reasons = self.submit(self.viewer(), 'hello')
        self.assertIs(accepted, False)
        self.assertIn('System prompt file missing', str(reasons[0]))

        with mock.patch('builtins.open', side_effect=OSError('denied')):
            accepted, reasons = self.submit(self.viewer(), 'hello')
        self.assertIs(accepted, False)
        self.assertIn('Could not read prompt file: denied', str(reasons[0]))

    def test_a_rejection_needs_no_callback(self):
        viewer = self.viewer(llm_loaded=False)
        self.assertIs(agent.run_web_agent_query(viewer, 'hello'), False)
        self.assertEqual([event['type'] for event in viewer.events], ['agent_error'])

    def accept(self, viewer, query='hello', **options):
        with mock.patch.object(agent, 'get_viewer_session_context', return_value=''), \
                mock.patch.object(agent, 'load_agent_history', return_value=[]), \
                mock.patch.object(agent, 'AgentWorker') as worker:
            reasons = []
            accepted = agent.run_web_agent_query(viewer, query, on_rejected=reasons.append, **options)
        return accepted, reasons, worker

    def test_an_accepted_turn_returns_true(self):
        viewer = self.viewer()
        accepted, reasons, worker = self.accept(viewer, submission_id='s1')
        self.assertIs(accepted, True)
        self.assertEqual(reasons, [])
        worker.return_value.start.assert_called_once_with()
        self.assertEqual([event['type'] for event in viewer.events], ['agent_thinking', 'agent_accepted'])
        self.assertTrue(viewer._agent_busy)

    def test_a_repeated_submission_reports_its_first_outcome(self):
        viewer = self.viewer()
        self.assertTrue(self.accept(viewer, submission_id='s1')[0])
        self.assertTrue(agent.run_web_agent_query(viewer, 'hello', submission_id='s1'))
        viewer = self.viewer(_agent_busy=True)
        self.assertFalse(self.submit(viewer, 'hello', submission_id='s2')[0])
        viewer._agent_busy = False
        self.assertFalse(agent.run_web_agent_query(viewer, 'hello', submission_id='s2'))

    def test_a_model_error_after_acceptance_reaches_the_terminal(self):
        viewer = self.viewer(_agent_busy=True)
        output = io.StringIO()
        with redirect_stdout(output):
            agent.on_web_worker_finished(viewer, 'hello', '', '', '', ValueError('The server refused the request.'))
        self.assertIn('The server refused the request.', output.getvalue())
        self.assertFalse(viewer._agent_busy)
        self.assertEqual([event['type'] for event in viewer.events], ['agent_error'])


class CardRefusalTests(unittest.TestCase):
    """A card that cannot be used leaves the model already loaded in place."""

    def loaded_viewer(self):
        viewer = SimpleNamespace(events=[])
        viewer.broadcast_event = viewer.events.append
        with mock.patch.object(agent.Command_Engine, 'print_help'):
            self.assertTrue(agent.activate_agent_from_card(
                viewer, {'name': 'Old', 'url': 'http://localhost:11001/v1', 'model': 'old-model', 'temperature': 0.5}, quiet=True))
        return viewer

    def assert_old_model_kept(self, viewer):
        self.assertTrue(viewer.llm_loaded)
        self.assertEqual((viewer.llm_url, viewer.llm_model_name, viewer.llm_temperature),
                         ('http://localhost:11001/v1', 'old-model', 0.5))

    def test_a_card_without_a_url_keeps_the_loaded_model(self):
        viewer = self.loaded_viewer()
        with mock.patch.object(agent.Command_Engine, 'print_help') as print_help:
            self.assertFalse(agent.activate_agent_from_card(viewer, {'name': 'New', 'model': 'x'}, quiet=True))
        self.assert_old_model_kept(viewer)
        self.assertIn("Model card 'New' has no URL", str(print_help.call_args.args[1]))

    def test_a_card_with_a_temperature_that_is_not_a_number_keeps_the_loaded_model(self):
        for temperature in ('warm', [0.2], {'t': 1}):
            with self.subTest(temperature=temperature):
                viewer = self.loaded_viewer()
                card = {'name': 'New', 'url': 'http://localhost:11002/v1', 'model': 'x', 'temperature': temperature}
                with mock.patch.object(agent.Command_Engine, 'print_help') as print_help:
                    self.assertFalse(agent.activate_agent_from_card(viewer, card, quiet=True))
                self.assert_old_model_kept(viewer)
                self.assertIn("Model card 'New' has an invalid temperature", str(print_help.call_args.args[1]))

    def test_a_usable_card_still_replaces_the_loaded_model(self):
        viewer = self.loaded_viewer()
        viewer.llm_history = [{'role': 'user', 'content': 'earlier'}]
        card = {'name': 'New', 'url': 'http://localhost:11002/v1', 'model': 'x', 'temperature': '0.25'}
        self.assertTrue(agent.activate_agent_from_card(viewer, card, quiet=True))
        self.assertEqual((viewer.llm_url, viewer.llm_model_name, viewer.llm_temperature, viewer.llm_loaded),
                         ('http://localhost:11002/v1', 'x', 0.25, True))
        self.assertEqual(viewer.llm_history, [])


class DroppedTurnTests(unittest.TestCase):
    """The Agent page's spinner waits for a reply; a turn dropped with its model gets one."""

    @staticmethod
    def running_viewer(submission_id='s1'):
        turn = {'submission_id': submission_id, 'attachments': []}
        viewer = SimpleNamespace(events=[], llm_loaded=True, llm_backend='server', llm_model_name='m',
                                 _agent_busy=True, _agent_turn=turn, _agent_generation=3)
        viewer.broadcast_event = viewer.events.append
        return viewer

    def test_deactivating_during_a_turn_ends_it_on_the_page(self):
        viewer = self.running_viewer()
        self.assertTrue(agent.deactivate_agent(viewer, quiet=True))
        [event] = viewer.events
        self.assertEqual((event['type'], event['submission_id']), ('agent_error', 's1'))
        self.assertIn('stopped', event['error'])
        # A page that reconnects and asks again about the submission gets the same answer.
        self.assertEqual(viewer._agent_submissions['s1'], event)
        self.assertEqual((viewer._agent_busy, viewer._agent_generation), (False, 4))

    def test_switching_models_during_a_turn_ends_it_once(self):
        viewer = self.running_viewer()
        card = {'name': 'New', 'url': 'http://localhost:11002/v1', 'model': 'x'}
        self.assertTrue(agent.activate_agent_from_card(viewer, card, quiet=True))
        self.assertEqual([event['type'] for event in viewer.events], ['agent_error'])

    def test_a_refused_card_does_not_end_the_turn(self):
        viewer = self.running_viewer()
        with mock.patch.object(agent.Command_Engine, 'print_help'):
            self.assertFalse(agent.activate_agent_from_card(viewer, {'name': 'New'}, quiet=True))
        self.assertEqual(viewer.events, [])
        self.assertEqual((viewer._agent_busy, viewer._agent_generation), (True, 3))

    def test_an_idle_agent_sends_nothing(self):
        viewer = self.running_viewer()
        viewer._agent_busy = False
        agent.deactivate_agent(viewer, quiet=True)
        self.assertEqual(viewer.events, [])

    def test_clearing_the_history_leaves_the_pages_own_reset_alone(self):
        viewer = self.running_viewer()
        with mock.patch.object(agent, 'save_agent_history'):
            agent.handle_clear_history(viewer, {})
        self.assertEqual(viewer.events, [])
        self.assertFalse(viewer._agent_busy)

    def test_deactivation_says_whether_a_model_was_unloaded(self):
        viewer = SimpleNamespace(llm_loaded=True, llm_backend='server', llm_model_name='m')
        with mock.patch.object(agent.Command_Engine, 'print_help'):
            self.assertTrue(agent.deactivate_agent(viewer))
            self.assertFalse(agent.deactivate_agent(viewer))
        self.assertFalse(viewer.llm_loaded)


class AgentCommandTests(unittest.TestCase):
    """The `agent` command: what it reports, and what the Agent page is told."""
    CARDS = {'cards': [{'id': 'a', 'name': 'Alpha', 'url': 'http://alpha.example/v1', 'model': 'alpha-1',
                        'data_use_accepted': 'http://alpha.example'},
                       {'id': 'b', 'name': 'Beta', 'url': 'http://beta.example/v1', 'model': 'beta-1',
                        'data_use_accepted': 'http://beta.example'}]}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.agent_folder = Path(directory.name, 'resources', 'agent')
        self.agent_folder.mkdir(parents=True)
        (self.agent_folder / 'model_card.json').write_text(json.dumps(self.CARDS), encoding='utf-8')
        (self.agent_folder / 'system_prompt.md').write_text('Prompt. ' + ' '.join(PLACEHOLDERS), encoding='utf-8')
        from commands import agent as agent_command
        self.command = agent_command
        engine = agent_command.Command_Engine
        patchers = [mock.patch.object(agent, '_SRC_DIR', directory.name),
                    mock.patch.object(agent, 'get_viewer_session_context', return_value=''),
                    mock.patch.object(agent, 'load_agent_history', return_value=[]),
                    mock.patch.object(agent, 'AgentWorker'),
                    mock.patch.object(agent_command, 'register'),
                    mock.patch.object(engine, 'print_help'),
                    mock.patch.object(engine, 'command_failed'),
                    mock.patch.object(engine, 'command_succeeded')]
        started = [patcher.start() for patcher in patchers]
        for patcher in patchers:
            self.addCleanup(patcher.stop)
        self.print_help, self.failed, self.succeeded = started[-3:]

    @staticmethod
    def viewer(**attributes):
        viewer = SimpleNamespace(events=[], **attributes)
        viewer.broadcast_event = viewer.events.append
        return viewer

    def run_command(self, viewer, *args):
        with redirect_stdout(io.StringIO()) as output:
            self.command.run(viewer, list(args))
        return output.getvalue()

    def states(self, viewer):
        return [event for event in viewer.events if event['type'] == 'backend_state']

    def test_a_message_that_starts_with_help_is_sent(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m')
        with mock.patch.object(self.command, 'run_web_agent_query', return_value=True) as query:
            output = self.run_command(viewer, 'help', 'me', 'select', 'cluster', '1')
        self.assertEqual(query.call_args.args[1], 'help me select cluster 1')
        self.assertNotIn('LLM Agent CLI Portal', output)
        self.succeeded.assert_called_once()

    def test_help_alone_prints_the_help(self):
        for word in ('help', 'HELP', '-h', '--help'):
            with self.subTest(word=word):
                viewer = self.viewer()
                with mock.patch.object(self.command, 'run_web_agent_query') as query:
                    output = self.run_command(viewer, word)
                query.assert_not_called()
                self.assertIn('LLM Agent CLI Portal', output)

    def test_a_turn_the_backend_rejects_is_reported_as_a_failure(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m', _agent_busy=True)
        self.run_command(viewer, 'hello')
        self.succeeded.assert_not_called()
        reason = 'This Viewer already has an active agent turn.'
        self.assertEqual(str(self.failed.call_args.args[1]), f'Error: {reason}')
        # The reason also reaches the terminal and the console line.
        self.assertEqual(str(self.print_help.call_args.args[1]), f'Error: {reason}')

    def test_an_empty_message_and_a_missing_prompt_file_are_failures(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m')
        self.run_command(viewer, '""')
        self.assertIn('Enter a message', str(self.failed.call_args.args[1]))
        self.succeeded.assert_not_called()

        (self.agent_folder / 'system_prompt.md').unlink()
        self.failed.reset_mock()
        self.run_command(viewer, 'hello')
        self.assertIn('System prompt file missing', str(self.failed.call_args.args[1]))
        self.succeeded.assert_not_called()

    def test_an_accepted_message_is_still_a_success(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m')
        self.run_command(viewer, 'hello')
        self.failed.assert_not_called()
        self.assertEqual(self.succeeded.call_args.args[1], 'Submitted the message to the Agent backend.')

    def test_a_failed_automatic_activation_is_reported_and_sends_nothing(self):
        (self.agent_folder / 'model_card.json').write_text(
            json.dumps({'cards': [{'id': 'a', 'name': 'Alpha', 'url': '', 'model': 'm'}]}), encoding='utf-8')
        viewer = self.viewer(llm_loaded=False)
        with mock.patch.object(self.command, 'run_web_agent_query') as query:
            self.run_command(viewer, 'hello')
        query.assert_not_called()
        self.succeeded.assert_not_called()
        self.failed.assert_called_once()
        self.assertEqual(self.states(viewer), [])

    def test_automatic_activation_tells_the_agent_page(self):
        viewer = self.viewer(llm_loaded=False)
        self.run_command(viewer, 'hello')
        [state] = self.states(viewer)
        self.assertEqual((state['llm_loaded'], state['llm_model_name']), (True, 'alpha-1'))
        self.succeeded.assert_called_once()

    def test_choosing_a_model_tells_the_agent_page(self):
        viewer = self.viewer()
        self.run_command(viewer, '<Beta>')
        [state] = self.states(viewer)
        self.assertEqual((state['llm_loaded'], state['llm_model_name']), (True, 'beta-1'))
        self.assertIn("Activated Agent model 'Beta'", self.succeeded.call_args.args[1])

    def test_a_refused_model_card_is_a_failure_and_the_page_hears_nothing(self):
        (self.agent_folder / 'model_card.json').write_text(
            json.dumps({'cards': [{'id': 'a', 'name': 'Alpha', 'url': '', 'model': 'm'}]}), encoding='utf-8')
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_url='http://localhost:11005/v1', llm_model_name='old')
        self.run_command(viewer, '<Alpha>')
        self.failed.assert_called_once()
        self.succeeded.assert_not_called()
        self.assertEqual((viewer.llm_loaded, viewer.llm_model_name), (True, 'old'))
        self.assertEqual(viewer.events, [])

    def test_turning_the_agent_off_tells_the_page_and_ends_a_running_turn(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m',
                             _agent_busy=True, _agent_turn={'submission_id': 's1', 'attachments': []})
        self.run_command(viewer, 'off')
        self.assertEqual([event['type'] for event in viewer.events], ['agent_error', 'backend_state'])
        self.assertFalse(self.states(viewer)[0]['llm_loaded'])
        self.assertEqual(self.succeeded.call_args.args[1], 'Agent deactivated.')

    def test_switching_models_during_a_turn_ends_the_turn_before_the_new_state(self):
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_model_name='m',
                             _agent_busy=True, _agent_turn={'submission_id': 's1', 'attachments': []})
        self.run_command(viewer, '<Beta>')
        self.assertEqual([event['type'] for event in viewer.events], ['agent_error', 'backend_state'])

    def test_turning_off_an_agent_that_is_off_reports_that(self):
        viewer = self.viewer(llm_loaded=False)
        self.run_command(viewer, 'off')
        self.assertEqual(self.succeeded.call_args.args[1], 'Agent is already inactive.')
        self.assertEqual(viewer.events, [])

    def open_viewer(self, opened, connected=False):
        viewer = self.viewer(open_agent_ui=mock.Mock(return_value=opened))
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: connected and client_id == 'agent')
        return viewer

    def test_the_page_is_reported_opened_only_when_it_opened(self):
        for opened, connected, outcome, text in (
            (True, False, self.succeeded, 'Opened the Agent interface'),
            (False, True, self.succeeded, 'already open'),
            (False, False, self.failed, 'was not opened'),
        ):
            with self.subTest(opened=opened, connected=connected):
                self.succeeded.reset_mock()
                self.failed.reset_mock()
                self.run_command(self.open_viewer(opened, connected))
                outcome.assert_called_once()
                self.assertIn(text, outcome.call_args.args[1])

    def test_a_command_from_the_portal_never_waits_on_an_already_open_dialog(self):
        from Viewer_Command_Portal import CURRENT
        viewer = self.open_viewer(False, connected=True)
        self.run_command(viewer)
        viewer.open_agent_ui.assert_called_once_with()

        token = CURRENT.set(object())
        self.addCleanup(CURRENT.reset, token)
        viewer = self.open_viewer(False, connected=True)
        self.run_command(viewer)
        viewer.open_agent_ui.assert_called_once_with(show_existing_dialog=False)


class SamplingParameterTests(unittest.TestCase):
    """A blank temperature is not sent; a provider that refuses sampling parameters is asked once more without them."""
    KEY = 'sk-FAKE-KEY-for-tests-0000'

    def call(self, provider, **options):
        report = {}
        output = io.StringIO()
        with redirect_stdout(output):
            result = agent.call_api(provider.url, 'm', 'system', 'question', report=report, **options)
        return result, report, output.getvalue()

    @staticmethod
    def refuse(*names, status=400):
        """A provider that answers `status` while the request holds one of names, and accepts it otherwise."""
        def respond(payload):
            if any(name in payload for name in names):
                return status, {'error': {'message': "Unsupported value: 'temperature' does not support 0.0 with this model."}}
            return 200, completion('fine')
        return respond

    def test_a_blank_temperature_is_not_sent_and_a_number_is(self):
        with FakeProvider() as provider:
            self.call(provider)
            self.call(provider, temperature=0.0)
            self.call(provider, temperature=0.7)
        self.assertEqual([('temperature' in r['payload'], r['payload'].get('temperature')) for r in provider.requests],
                         [(False, None), (True, 0.0), (True, 0.7)])

    def test_a_card_with_a_blank_temperature_is_activated_without_one(self):
        for label, value in (('missing', None), ('null', None), ('empty', ''), ('spaces', '  ')):
            card = {'name': 'Remote', 'url': 'http://localhost:11008/v1', 'model': 'm'}
            if label != 'missing':
                card['temperature'] = value
            with self.subTest(label):
                viewer = SimpleNamespace()
                self.assertTrue(agent.activate_agent_from_card(viewer, card, quiet=True))
                self.assertIsNone(viewer.llm_temperature)
        for value, expected in ((0.0, 0.0), (0, 0.0), ('0.3', 0.3), (1, 1.0)):
            with self.subTest(value=value):
                viewer = SimpleNamespace()
                self.assertTrue(agent.activate_agent_from_card(viewer, {'url': 'http://localhost:11004/v1', 'model': 'm', 'temperature': value}, quiet=True))
                self.assertEqual(viewer.llm_temperature, expected)
        for value in ('warm', float('nan'), float('inf'), [1]):
            with self.subTest(value=value), mock.patch.object(agent.Command_Engine, 'print_help'):
                self.assertFalse(agent.activate_agent_from_card(SimpleNamespace(), {'url': 'http://localhost:11004/v1', 'temperature': value}, quiet=True))

    def test_the_local_default_cards_keep_a_temperature_of_zero(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(agent, '_SRC_DIR', folder):
            defaults = agent.load_model_cards()
        self.assertEqual([card['temperature'] for card in defaults], [0.0, 0.0, 0.0])

    def test_a_refusal_naming_a_sampling_parameter_is_retried_once_without_them(self):
        options = {'top_p': 0.9, 'top_k': 5, 'reasoning_effort': 'low'}
        with FakeProvider(self.refuse('temperature', 'top_p', 'top_k')) as provider:
            result, report, output = self.call(provider, temperature=0.0, options=options)
        first, second = (r['payload'] for r in provider.requests)
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual({k: first[k] for k in ('temperature', 'top_p', 'top_k')}, {'temperature': 0.0, 'top_p': 0.9, 'top_k': 5})
        self.assertFalse({'temperature', 'top_p', 'top_k'} & set(second))
        # What is not a sampling parameter, and the whole conversation, are sent again as they were.
        self.assertEqual(second['reasoning_effort'], 'low')
        self.assertEqual(second['messages'], first['messages'])
        self.assertEqual((result['content'], report), ('fine', {'sampling_rejected': True}))
        self.assertIn('sending the request again without them', output)

    def test_each_sampling_parameter_alone_triggers_the_retry(self):
        for name in ('temperature', 'top_p', 'top_k'):
            with self.subTest(name), FakeProvider(self.refuse(name)) as provider:
                _, report, _ = self.call(provider, temperature=0.5 if name == 'temperature' else None,
                                         options={name: 1} if name != 'temperature' else None)
                self.assertEqual(len(provider.requests), 2)
                self.assertTrue(report['sampling_rejected'])

    def test_other_failures_are_not_retried(self):
        cases = {
            'a 400 about something else': (lambda payload: (400, 'image input unsupported'), {'temperature': 0.0}),
            'a 500 naming temperature': (self.refuse('temperature', status=500), {'temperature': 0.0}),
            'a 429': (lambda payload: (429, 'rate limit: slow down'), {'temperature': 0.0}),
            'a 400 naming temperature when none was sent': (self.refuse('model'), {}),
        }
        for label, (respond, options) in cases.items():
            with self.subTest(label), FakeProvider(respond) as provider:
                with self.assertRaises(ValueError) as caught:
                    self.call(provider, **options)
                self.assertEqual(len(provider.requests), 1)
                self.assertIn('Model request failed (HTTP', str(caught.exception))

    def test_a_second_refusal_ends_the_request(self):
        always = lambda payload: (400, "temperature is not supported; top_p is not supported either")
        with FakeProvider(always) as provider:
            with self.assertRaises(ValueError) as caught:
                self.call(provider, temperature=0.0)
        self.assertEqual(len(provider.requests), 2)
        self.assertIn('HTTP 400', str(caught.exception))

    def test_the_key_is_never_printed_or_shown(self):
        echoing = lambda payload: (400 if 'temperature' in payload else 401, f"Incorrect API key provided: {self.KEY}; temperature too")
        for options in ({'temperature': 0.0}, {}):
            with self.subTest(options=options), FakeProvider(echoing) as provider:
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(ValueError) as caught:
                    agent.call_api(provider.url, 'm', 's', 'q', api_key=self.KEY, **options)
                shown = output.getvalue() + str(caught.exception) + repr(caught.exception.args)
                self.assertNotIn(self.KEY, shown)
                self.assertIn('Incorrect API key provided', shown)
                # The key was sent as the provider's credential, and only there.
                self.assertTrue(all(r['headers'].get('Authorization') == f'Bearer {self.KEY}' for r in provider.requests))
                self.assertNotIn(self.KEY, json.dumps([r['payload'] for r in provider.requests]))


class RememberedSamplingTests(unittest.TestCase):
    """The card remembers a provider's refusal: this session, the card file, and the Agent page's own copy."""
    CARD = {'id': 'mine', 'name': 'My API', 'url': 'http://localhost:11007/v1', 'model': 'm', 'api_key': 'sk-FAKE-0000',
            'temperature': 0.0, 'options': {'top_p': 0.9, 'reasoning_effort': 'low'}, 'extra': 'kept'}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name, 'resources', 'agent')
        self.folder.mkdir(parents=True)
        self.path = self.folder / 'model_card.json'
        self.path.write_text(json.dumps({'cards': [dict(self.CARD), {'id': 'other', 'name': 'Other'}], 'note': 'kept'}), encoding='utf-8')
        patcher = mock.patch.object(agent, '_SRC_DIR', directory.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = SimpleNamespace(events=[])
        self.viewer.broadcast_event = self.viewer.events.append
        self.assertTrue(agent.activate_agent_from_card(self.viewer, dict(self.CARD), quiet=True))

    def saved(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_a_refusal_is_remembered_in_the_session_and_on_the_saved_card(self):
        self.assertEqual((self.viewer.llm_temperature, self.viewer.llm_options), (0.0, {'top_p': 0.9, 'reasoning_effort': 'low'}))
        agent._apply_call_notes(self.viewer, {'sampling_rejected': True}, self.viewer.llm_card_ref)

        self.assertEqual((self.viewer.llm_temperature, self.viewer.llm_options), (None, {'reasoning_effort': 'low'}))
        document = self.saved()
        self.assertEqual(document['note'], 'kept')
        self.assertEqual(document['cards'][0], {**self.CARD, 'omit_sampling': True})
        self.assertEqual(document['cards'][1], {'id': 'other', 'name': 'Other'})
        self.assertEqual(os.listdir(self.folder), ['model_card.json'])
        # The page keeps its own copy of the cards and saves them whole: it is told.
        self.assertEqual(self.viewer.events, [{'type': 'model_card_updated', 'card_id': 'mine', 'fields': {'omit_sampling': True}}])

        # After a restart the card still sends nothing: even the temperature a card holds.
        restarted = SimpleNamespace()
        self.assertTrue(agent.activate_agent_from_card(restarted, agent.load_model_cards()[0], quiet=True))
        self.assertEqual((restarted.llm_temperature, restarted.llm_options), (None, {'reasoning_effort': 'low'}))

    def test_nothing_is_remembered_without_a_refusal(self):
        before = self.path.read_bytes()
        for report in ({}, {'sampling_rejected': False}, None, mock.Mock()):
            agent._apply_call_notes(self.viewer, report, self.viewer.llm_card_ref)
        self.assertEqual((self.path.read_bytes(), self.viewer.events, self.viewer.llm_temperature), (before, [], 0.0))

    def test_a_refusal_met_after_the_model_was_switched_updates_only_the_card(self):
        old_ref = self.viewer.llm_card_ref
        self.assertTrue(agent.activate_agent_from_card(self.viewer, {'id': 'other', 'name': 'Other', 'url': 'http://localhost:11004/v1', 'temperature': 0.2}, quiet=True))
        agent._apply_call_notes(self.viewer, {'sampling_rejected': True}, old_ref)
        self.assertEqual(self.viewer.llm_temperature, 0.2)
        self.assertTrue(self.saved()['cards'][0]['omit_sampling'])
        self.assertNotIn('omit_sampling', self.saved()['cards'][1])

    def test_a_card_without_an_id_is_found_by_its_name_and_address(self):
        card = {'name': 'Hand-written', 'url': 'http://localhost:11009/v1', 'temperature': 0.0}
        self.path.write_text(json.dumps({'cards': [card, {'name': 'Hand-written', 'url': 'http://localhost:11010/v1'}]}), encoding='utf-8')
        viewer = SimpleNamespace(events=[])
        viewer.broadcast_event = viewer.events.append
        agent.activate_agent_from_card(viewer, card, quiet=True)
        agent._apply_call_notes(viewer, {'sampling_rejected': True}, viewer.llm_card_ref)
        cards = self.saved()['cards']
        self.assertEqual([c.get('omit_sampling') for c in cards], [True, None])

    def test_a_card_that_cannot_be_saved_still_stops_the_session_sending_them(self):
        output = io.StringIO()
        with mock.patch.object(agent, 'save_model_cards', side_effect=OSError('disk full')), redirect_stdout(output):
            agent._apply_call_notes(self.viewer, {'sampling_rejected': True}, self.viewer.llm_card_ref)
        self.assertIsNone(self.viewer.llm_temperature)
        self.assertEqual(self.viewer.events, [])
        self.assertIn('could not be updated (disk full)', output.getvalue())
        self.assertNotIn('sk-FAKE-0000', output.getvalue())
        self.assertNotIn('omit_sampling', self.saved()['cards'][0])

    def test_a_worker_that_met_a_refusal_hands_it_to_the_turn(self):
        refusing = SamplingParameterTests.refuse('temperature', 'top_p')
        with FakeProvider(refusing) as provider:
            card = {**self.CARD, 'url': provider.url}
            self.assertTrue(agent.activate_agent_from_card(self.viewer, card, quiet=True))
            viewer = self.viewer
            worker = agent.AgentWorker('server', viewer.llm_url, 'm', 'system', 'hello', [], viewer.llm_temperature,
                                       viewer.llm_api_key, viewer.llm_options, card_ref=viewer.llm_card_ref)
            emitted = []
            worker.finished.connect(lambda *values: emitted.append(values))
            with redirect_stdout(io.StringIO()):
                worker.run()
        self.assertEqual((emitted[0][0], len(provider.requests)), ('fine', 2))
        viewer._agent_busy, viewer._agent_generation = True, 0
        with mock.patch.object(agent, 'save_and_broadcast_agent_response'):
            agent.on_web_worker_finished(viewer, 'hello', *emitted[0], generation=0, worker=worker)
        self.assertEqual((viewer.llm_temperature, viewer.llm_options), (None, {'reasoning_effort': 'low'}))
        self.assertTrue(self.saved()['cards'][0]['omit_sampling'])


class AgentSnapshotTests(unittest.TestCase):
    """A turn's snapshot must not push an MCP client's out of the Viewer's shared snapshot store."""

    def setUp(self):
        from desktop.Viewer_Inspection import ViewerInspectionService
        self.viewer = SimpleNamespace(
            n_nodes=4, full_headers=['A', 'B', 'C', 'D'], visible_mask=[True, True, True, True], selected_indices=[1],
            edges=[(0, 1)], metadata={'score': {'type': 'numeric', 'values': [1.0, 2.0, 3.0, 4.0]}},
            cluster_labels=[1, 1, 2, 2], group_labels=[set(), set(), set(), set()], current_slider_threshold=0.5)
        config = SimpleNamespace(NODE_FASTA_FILE=None, INPUT_HDF5='network.h5', MSA_FILE='', TARGET_CACHE_PATH=None,
                                 TARGET_CACHE_FILE=None, SIMILARITY_THRESHOLD=0.1)
        self.service = ViewerInspectionService(self.viewer, config)
        self.viewer.viewer_inspection = self.service

    def test_a_turn_reads_the_snapshot_it_sends_and_leaves_the_shared_store_alone(self):
        text = agent.get_viewer_session_context(self.viewer)
        self.assertTrue(text.startswith('\n--- ACTIVE VIEWER SNAPSHOT ---\n'))
        document = json.loads(text.split('---\n', 1)[1])
        self.assertEqual((document['summary']['node_count'], document['summary']['selected_node_count']), (4, 1))
        self.assertIn('score', json.dumps(document['fields']))
        self.assertEqual(len(self.service.snapshots.items), 0)

    def test_an_mcp_snapshot_stays_retrievable_after_several_agent_turns(self):
        # The store holds two snapshots; both MCP clients' must survive every turn.
        first, second = self.service.capture_snapshot(), self.service.capture_snapshot()
        for _ in range(5):
            self.assertIn('ACTIVE VIEWER SNAPSHOT', agent.get_viewer_session_context(self.viewer))
        for snapshot in (first, second):
            self.assertEqual(self.service.snapshots.execute('get_summary', snapshot)['node_count'], 4)
        self.assertEqual(len(self.service.snapshots.items), 2)

    def test_a_viewer_without_an_inspection_service_gets_one(self):
        del self.viewer.viewer_inspection
        self.assertIn('ACTIVE VIEWER SNAPSHOT', agent.get_viewer_session_context(self.viewer))
        self.assertIsNotNone(self.viewer.viewer_inspection)


class ApprovalNoticeTests(PortalFixture, unittest.TestCase):
    """A command the user must approve, started from the terminal or while the page is closed, is announced."""

    def setUp(self):
        super().setUp()
        self.viewer.shows_windows = lambda: True
        self.viewer.open_agent_ui = mock.Mock(return_value=True)
        self.viewer.llm_loaded, self.viewer.llm_backend = True, 'server'
        patcher = mock.patch.object(agent.Command_Engine, 'print_help')
        self.print_help = patcher.start()
        self.addCleanup(patcher.stop)
        # What the model is asked after the commands ended is not under test, and no request is sent.
        patcher = mock.patch.object(agent, 'start_refinement_worker')
        patcher.start()
        self.addCleanup(patcher.stop)

    def announcements(self):
        return [str(call.args[1]) for call in self.print_help.call_args_list if 'needs your approval' in str(call.args[1])]

    def reply(self, text):
        self.viewer._agent_busy = True
        agent.on_web_worker_finished(self.viewer, 'question', text, '', '{}', '')
        return self.viewer._agent_request_id

    def settle(self, until=lambda: False, seconds=0.3):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not until():
            self.app.processEvents()
            time.sleep(0.005)

    def test_the_terminal_is_told_and_the_page_opens_when_no_page_is_open(self):
        request_id = self.reply('command: select "one"\ncommand: save x.h5')
        self.settle(self.announcements, seconds=5)
        [message] = self.announcements()
        self.assertIn('The agent wants to run save x.h5, which needs your approval.', message)
        self.viewer.open_agent_ui.assert_called_once_with(show_existing_dialog=False)
        self.assertEqual(self.portal.get(request_id)['status'], 'awaiting_user_input')
        # The page, once open, is sent the request with its commands.
        [event] = [e for e in self.viewer.events if e['type'] == 'agent_command_request']
        self.assertEqual([(c['command'], c['approval']) for c in event['commands']], [('select "one"', None), ('save x.h5', 'pending')])

    def test_an_open_page_needs_no_notice(self):
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: client == 'agent')
        self.reply('command: save x.h5')
        self.settle()
        self.assertEqual(self.announcements(), [])
        self.viewer.open_agent_ui.assert_not_called()

    def test_a_viewer_without_windows_is_not_asked_to_open_a_page(self):
        # The command waits only if the Agent page is open: the page is then served to a browser by hand.
        self.viewer.shows_windows = lambda: False
        self.viewer.web_server = SimpleNamespace(has_event_client=lambda client: False)
        request_id = self.reply('command: save x.h5')
        self.settle()
        self.viewer.open_agent_ui.assert_not_called()
        self.assertEqual(self.announcements(), [])
        result = self.portal.get(request_id)
        self.assertEqual((result['status'], result['commands'][0]['approval']), ('failed', 'refused'))

    def test_the_page_is_opened_only_where_the_viewer_shows_windows(self):
        record = {'approval': 'pending', 'status': 'awaiting_user_input', 'command': 'save x.h5'}
        portal = mock.Mock()
        portal.get.return_value = {'commands': [record]}
        for shows, opens in ((True, True), (False, False)):
            with self.subTest(shows_windows=shows):
                self.viewer.shows_windows = lambda shows=shows: shows
                self.viewer.open_agent_ui.reset_mock()
                self.print_help.reset_mock()
                self.viewer._agent_request_id = 'r'
                agent._announce_approval(self.viewer, portal, 'r', 'c')
                # The terminal is told either way; the page is opened only for a user who can see it.
                self.assertEqual(len(self.announcements()), 1)
                self.assertEqual(self.viewer.open_agent_ui.called, opens)

    def test_a_waiting_command_of_another_request_is_not_announced(self):
        record = {'approval': 'pending', 'status': 'awaiting_user_input', 'command': 'save x.h5'}
        portal = mock.Mock()
        portal.get.return_value = {'commands': [record]}
        self.viewer._agent_request_id = 'the turn running now'
        agent._announce_approval(self.viewer, portal, 'a turn that was stopped', 'c')
        self.assertEqual(self.announcements(), [])
        agent._announce_approval(self.viewer, portal, 'the turn running now', 'c')
        self.assertEqual(len(self.announcements()), 1)

    def test_nothing_is_announced_for_a_turn_that_was_stopped(self):
        request_id = self.reply('command: save x.h5')
        agent._invalidate_agent_turn(self.viewer)  # Before the portal's first pass over the queue.
        self.settle()
        self.assertEqual(self.announcements(), [])
        self.viewer.open_agent_ui.assert_not_called()
        self.assertEqual(self.portal.get(request_id)['status'], 'cancelled')

    def test_a_command_decided_in_the_meantime_or_of_another_request_is_not_announced(self):
        request = self.portal.get(self.reply('command: save y.h5'))
        command = request['commands'][0]
        self.settle(self.announcements, seconds=5)
        self.print_help.reset_mock()
        self.portal.decide(request['request_id'], command['command_id'], True)
        agent._announce_approval(self.viewer, self.portal, request['request_id'], command['command_id'])
        self.assertEqual(self.announcements(), [])
        agent._announce_approval(self.viewer, self.portal, 'another request', command['command_id'])
        self.assertEqual(self.announcements(), [])
        self.portal.discard_pending(request['request_id'])


class RequestAbortTests(unittest.TestCase):
    """A stopped turn's model request ends at once: the worker's blocking read is shut down, not waited out."""

    @staticmethod
    def call_in_thread(provider, handle, **options):
        outcome = {}

        def work():
            try:
                outcome['result'] = agent.call_api(provider.url, 'm', 'system', 'question', handle=handle, **options)
            except BaseException as error:
                outcome['error'] = error
            outcome['ended'] = time.monotonic()

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread, outcome

    def assert_ends_when_aborted(self, provider, handle, thread, outcome):
        self.assertTrue(provider.arrived.wait(5), 'the request never arrived')
        started = time.monotonic()
        handle.abort()
        thread.join(5)
        self.assertFalse(thread.is_alive(), 'the worker is still blocked in its read')
        self.assertIsInstance(outcome.get('error'), agent.RequestAborted, outcome)
        # The timeout is 300 s: this is the abort, not the wait running out.
        self.assertLess(outcome['ended'] - started, 2)

    def test_a_request_waiting_for_the_reply_ends_when_aborted(self):
        def respond(payload):
            provider.hold()
            return 200, completion()
        with FakeProvider(respond) as provider:
            handle = agent.RequestHandle()
            thread, outcome = self.call_in_thread(provider, handle)
            self.assert_ends_when_aborted(provider, handle, thread, outcome)

    def test_a_reply_that_stalls_in_its_body_ends_when_aborted(self):
        with FakeProvider(lambda payload: (200, Stalled())) as provider:
            handle = agent.RequestHandle()
            thread, outcome = self.call_in_thread(provider, handle)
            self.assert_ends_when_aborted(provider, handle, thread, outcome)

    def over_tls(self):
        """A temporary folder's certificate: (server context, client context), or the test is skipped."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        contexts = self_signed_tls(folder.name)
        if contexts is None:
            self.skipTest('the cryptography package is not installed')
        server, client = contexts
        # http.client asks ssl for the default context when it is given none: here, the one that trusts the test certificate.
        patcher = mock.patch('ssl._create_default_https_context', return_value=client)
        patcher.start()
        self.addCleanup(patcher.stop)
        return server

    def test_a_request_over_tls_ends_when_aborted(self):
        server = self.over_tls()
        stalls = {'before the headers': lambda payload: (provider.hold(), (200, completion()))[1],
                  'in the body': lambda payload: (200, Stalled())}
        for label, respond in stalls.items():
            with self.subTest(label), FakeProvider(respond, tls=server) as provider:
                handle = agent.RequestHandle()
                thread, outcome = self.call_in_thread(provider, handle)
                self.assert_ends_when_aborted(provider, handle, thread, outcome)

    def test_a_request_over_tls_that_is_not_aborted_works_through_the_handle(self):
        server = self.over_tls()
        with FakeProvider(tls=server) as provider:
            result = agent.call_api(provider.url, 'm', 's', 'q', handle=agent.RequestHandle())
        self.assertEqual((result['content'], provider.url[:8]), ('ok', 'https://'))

    def test_a_request_aborted_before_it_starts_sends_nothing(self):
        with FakeProvider() as provider:
            handle = agent.RequestHandle()
            handle.abort()
            with self.assertRaises(agent.RequestAborted):
                agent.call_api(provider.url, 'm', 's', 'q', handle=handle)
        self.assertEqual(provider.requests, [])

    def test_a_connection_made_after_the_abort_is_shut_down_at_once(self):
        import socket
        ours, theirs = socket.socketpair()
        self.addCleanup(ours.close)
        self.addCleanup(theirs.close)
        handle = agent.RequestHandle()
        handle.abort()
        with self.assertRaises(agent.RequestAborted):
            handle.attach(ours)
        theirs.settimeout(2)
        self.assertEqual(theirs.recv(1), b'')

    def test_a_request_that_is_not_aborted_works_through_the_handle(self):
        refusing = SamplingParameterTests.refuse('temperature')
        with FakeProvider(refusing) as provider:
            handle = agent.RequestHandle()
            report = {}
            with redirect_stdout(io.StringIO()):
                result = agent.call_api(provider.url, 'm', 's', 'q', temperature=0.0, handle=handle, report=report)
            # The retry without sampling parameters goes through the same handle.
            self.assertEqual((result['content'], len(provider.requests), report), ('fine', 2, {'sampling_rejected': True}))
            handle.abort()  # Too late to matter, and no error.
            self.assertTrue(handle.aborted)

    def test_an_error_from_the_provider_is_still_the_providers_error(self):
        with FakeProvider(lambda payload: (500, 'overloaded')) as provider:
            with redirect_stdout(io.StringIO()), self.assertRaises(ValueError) as caught:
                agent.call_api(provider.url, 'm', 's', 'q', handle=agent.RequestHandle())
        self.assertIn('HTTP 500', str(caught.exception))

    def test_a_server_that_is_not_there_is_still_an_ordinary_error(self):
        with FakeProvider() as provider:
            url = provider.url
        with self.assertRaises(OSError):  # urllib's URLError: not mistaken for an abort.
            agent.call_api(url, 'm', 's', 'q', handle=agent.RequestHandle())


class StopTurnTests(unittest.TestCase):
    """The page's Stop button: the turn is dropped, its request ended, its approvals discarded, and the page told."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    @staticmethod
    def running_viewer(submission_id='s1', **attributes):
        turn = {'submission_id': submission_id, 'attachments': []}
        viewer = SimpleNamespace(events=[], llm_loaded=True, llm_backend='server', llm_model_name='m',
                                 _agent_busy=True, _agent_turn=turn, _agent_generation=3,
                                 _agent_request_id='r1', _agent_handle=agent.RequestHandle(),
                                 command_portal=mock.Mock(), **attributes)
        viewer.broadcast_event = viewer.events.append
        return viewer

    def test_stopping_drops_the_turn_ends_its_request_and_discards_its_approvals(self):
        viewer = self.running_viewer()
        handle = viewer._agent_handle
        agent.handle_cancel_turn(viewer, {'action': 'cancel_turn', 'submission_id': 's1'})
        self.assertEqual(viewer.events, [{'type': 'agent_stopped', 'submission_id': 's1'}])
        self.assertEqual((viewer._agent_busy, viewer._agent_generation, viewer._agent_request_id, viewer._agent_handle),
                         (False, 4, None, None))
        self.assertTrue(handle.aborted)
        viewer.command_portal.discard_pending.assert_called_once_with('r1')
        # A page that reconnects and asks again about the submission is told it was stopped, and is not accepted.
        self.assertEqual(viewer._agent_submissions['s1'], viewer.events[0])
        self.assertIs(agent.run_web_agent_query(viewer, 'hello', submission_id='s1'), False)
        self.assertEqual(viewer.events[-1], {'type': 'agent_stopped', 'submission_id': 's1'})

    def test_a_turn_started_elsewhere_is_stopped_without_a_submission_id(self):
        viewer = self.running_viewer(submission_id=None)
        agent.handle_cancel_turn(viewer, {})
        self.assertEqual(viewer.events, [{'type': 'agent_stopped'}])
        self.assertFalse(viewer._agent_busy)

    def test_a_late_click_stops_nothing(self):
        # No turn is running.
        viewer = self.running_viewer()
        viewer._agent_busy = False
        agent.handle_cancel_turn(viewer, {'submission_id': 's1'})
        self.assertEqual((viewer.events, viewer._agent_generation, viewer._agent_handle.aborted), ([], 3, False))
        # The turn running is another one than the page's.
        viewer = self.running_viewer(submission_id='s2')
        agent.handle_cancel_turn(viewer, {'submission_id': 's1'})
        self.assertEqual((viewer.events, viewer._agent_busy, viewer._agent_handle.aborted), ([], True, False))
        viewer.command_portal.discard_pending.assert_not_called()

    def test_every_way_of_dropping_a_turn_ends_its_request(self):
        for label, drop in (
            ('stop', lambda viewer: agent.handle_cancel_turn(viewer, {})),
            ('switched off', lambda viewer: agent.deactivate_agent(viewer, quiet=True)),
            ('history cleared', lambda viewer: agent.handle_clear_history(viewer, {})),
            ('another model', lambda viewer: agent.activate_agent_from_card(viewer, {'name': 'New', 'url': 'http://localhost:11003/v1', 'model': 'x'}, quiet=True)),
        ):
            with self.subTest(label), mock.patch.object(agent, 'save_agent_history'):
                viewer = self.running_viewer()
                handle = viewer._agent_handle
                drop(viewer)
                self.assertTrue(handle.aborted)

    def test_the_worker_blocked_on_a_slow_model_ends_when_the_turn_is_stopped(self):
        def respond(payload):
            provider.hold()
            return 200, completion('too late')
        with FakeProvider(respond) as provider:
            viewer = self.running_viewer()
            worker = agent.AgentWorker('server', provider.url, 'm', 'system', 'question', [], None, None)
            viewer._agent_handle = worker.handle
            emitted = []
            worker.finished.connect(lambda *values: emitted.append(values))
            worker.start()  # A real QThread, as in the Viewer.
            self.assertTrue(provider.arrived.wait(5))
            started = time.monotonic()
            agent.handle_cancel_turn(viewer, {})
            deadline = time.monotonic() + 5
            while not emitted and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            self.assertTrue(worker.wait(5000), 'the worker thread is still blocked in its read')
            self.assertLess(time.monotonic() - started, 2)
        [(text, _, _, error)] = emitted
        self.assertEqual(text, '')
        self.assertIsInstance(error, agent.RequestAborted)
        # The error reaches the turn after it was dropped: nothing more is said, sent or saved.
        events = len(viewer.events)
        with mock.patch.object(agent, 'save_and_broadcast_agent_response') as saved, \
                mock.patch.object(agent, 'start_refinement_worker') as explained:
            agent.on_web_worker_finished(viewer, 'question', *emitted[0], generation=3, turn=None, worker=worker)
        self.assertEqual((len(viewer.events), saved.called, explained.called), (events, False, False))
        viewer.command_portal.submit.assert_not_called()

    def test_a_turn_the_viewer_started_can_be_stopped_while_the_model_thinks(self):
        # Through run_web_agent_query, which is what tells the Viewer which request to end.
        def respond(payload):
            provider.hold()
            return 200, completion('too late')
        with FakeProvider(respond) as provider:
            viewer = SimpleNamespace(events=[], llm_loaded=True, llm_backend='server', llm_url=provider.url, llm_model_name='m')
            viewer.broadcast_event = viewer.events.append
            with mock.patch.object(agent, 'get_viewer_session_context', return_value=''), \
                    mock.patch.object(agent, 'load_agent_history', return_value=[]):
                self.assertTrue(agent.run_web_agent_query(viewer, 'hello', submission_id='s1'))
            worker = viewer._web_agent_worker
            self.assertTrue(provider.arrived.wait(5))
            agent.handle_cancel_turn(viewer, {'submission_id': 's1'})
            self.assertTrue(worker.wait(5000), 'the worker thread is still waiting for the model')
        self.assertTrue(worker.handle.aborted)
        self.assertEqual([event['type'] for event in viewer.events], ['agent_thinking', 'agent_accepted', 'agent_stopped'])

    def test_stopping_during_the_explain_call_drops_its_answer(self):
        viewer = self.running_viewer()
        viewer.llm_history = []
        with mock.patch.object(agent, 'RefinementWorker') as worker_class, \
                mock.patch.object(agent, 'save_and_broadcast_agent_response') as saved:
            agent.start_refinement_worker(viewer, 'question', 'text', 'save x.h5', '{}', '{}', '', turn=viewer._agent_turn)
            worker = worker_class.return_value
            self.assertIs(viewer._agent_handle, worker.handle)
            finished = worker.finished.connect.call_args.args[0]
            agent.handle_cancel_turn(viewer, {})
            worker.handle.abort.assert_called_once_with()
            finished('an answer nobody asked for any more', '', '{}', '')
        saved.assert_not_called()
        self.assertEqual([event['type'] for event in viewer.events], ['agent_thinking', 'agent_stopped'])


class StopWithPortalTests(PortalFixture, unittest.TestCase):
    """What stopping does to a turn whose commands wait for approval in the real portal."""

    def setUp(self):
        super().setUp()
        self.viewer.shows_windows = lambda: True
        self.viewer.open_agent_ui = mock.Mock(return_value=True)
        self.viewer.llm_loaded, self.viewer.llm_backend = True, 'server'
        self.viewer._agent_busy, self.viewer._agent_turn, self.viewer._agent_generation = True, {'submission_id': 's1', 'attachments': []}, 0
        patcher = mock.patch.object(agent.Command_Engine, 'print_help')
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(agent, 'start_refinement_worker')
        self.explain = patcher.start()
        self.addCleanup(patcher.stop)

    def reply_with_a_waiting_command(self):
        agent.on_web_worker_finished(self.viewer, 'save it', 'command: select "one"\ncommand: save x.h5\ncommand: select "two"',
                                     '', '{}', '', generation=0, turn=self.viewer._agent_turn)
        request_id = self.viewer._agent_request_id
        deadline = time.monotonic() + 5
        while self.portal.get(request_id)['status'] != 'awaiting_user_input' and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        return request_id

    def settle(self, seconds=0.4):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)

    def test_stopping_discards_the_waiting_command_frees_the_queue_and_never_asks_the_model_again(self):
        request_id = self.reply_with_a_waiting_command()
        agent.handle_cancel_turn(self.viewer, {'submission_id': 's1'})
        self.settle()
        result = self.portal.get(request_id)
        self.assertEqual([(c['command'], c['status']) for c in result['commands']],
                         [('select "one"', 'succeeded'), ('save x.h5', 'cancelled'), ('select "two"', 'skipped')])
        self.assertEqual(self.portal.queue, [])
        self.explain.assert_not_called()
        self.assertEqual([e['type'] for e in self.viewer.events if e['type'].startswith('agent_')],
                         ['agent_command_request', 'agent_stopped'])

    def test_the_second_model_call_does_start_when_the_user_discards_instead(self):
        # The positive control of the test above: the same flow, ended by the user's Discard.
        request_id = self.reply_with_a_waiting_command()
        command = self.portal.get(request_id)['commands'][1]
        self.portal.decide(request_id, command['command_id'], False)
        deadline = time.monotonic() + 5
        while not self.explain.called and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.explain.assert_called_once()


class CardOriginTests(unittest.TestCase):
    """Where a card sends its requests, and whether that is this computer."""

    def test_which_servers_are_on_this_computer(self):
        cases = {
            'http://localhost:11434/v1': (False, 'http://localhost:11434', 'localhost:11434'),
            'http://LOCALHOST/v1': (False, 'http://localhost', 'localhost'),
            'http://127.0.0.1:1234/v1': (False, 'http://127.0.0.1:1234', '127.0.0.1:1234'),
            'http://127.5.6.7/v1': (False, 'http://127.5.6.7', '127.5.6.7'),
            'http://[::1]:8080/v1': (False, 'http://[::1]:8080', '[::1]:8080'),
            'http://0.0.0.0:8080/v1': (False, 'http://0.0.0.0:8080', '0.0.0.0:8080'),
            'http://ollama.localhost:11434/v1': (False, 'http://ollama.localhost:11434', 'ollama.localhost:11434'),
            'https://api.openai.com/v1': (True, 'https://api.openai.com', 'api.openai.com'),
            'https://API.Example.com:8443/v1/chat/completions': (True, 'https://api.example.com:8443', 'api.example.com:8443'),
            'http://192.168.1.20:11434/v1': (True, 'http://192.168.1.20:11434', '192.168.1.20:11434'),
            'http://10.0.0.5/v1': (True, 'http://10.0.0.5', '10.0.0.5'),
            'http://gpu-box.local:1234/v1': (True, 'http://gpu-box.local:1234', 'gpu-box.local:1234'),
            'http://[2001:db8::7]/v1': (True, 'http://[2001:db8::7]', '[2001:db8::7]'),
            # A name that only starts like a local one is not one.
            'http://localhost.example.com/v1': (True, 'http://localhost.example.com', 'localhost.example.com'),
            'http://127.0.0.1.example.com/v1': (True, 'http://127.0.0.1.example.com', '127.0.0.1.example.com'),
            # What the user typed before @ is not part of where the data goes.
            'https://user:secret@api.example.com/v1': (True, 'https://api.example.com', 'api.example.com'),
        }
        for url, (remote, origin, host) in cases.items():
            with self.subTest(url=url):
                found = agent.card_origin({'url': url})
                self.assertEqual((found['remote'], found['origin'], found['host']), (remote, origin, host))
        for url in ('', None, 'localhost:11434/v1', 'ftp://example.com/', 'not a url', 'http://host:badport/v1', '/v1'):
            with self.subTest(url=url):
                self.assertIsNone(agent.card_origin({'url': url}))

    def test_a_key_over_plain_http_to_another_computer_is_flagged(self):
        flagged = lambda url, key: agent.card_origin({'url': url, 'api_key': key})['plain_http_key']
        self.assertTrue(flagged('http://gpu-box:1234/v1', 'sk-FAKE'))
        self.assertFalse(flagged('http://gpu-box:1234/v1', ''))          # no key to send
        self.assertFalse(flagged('https://gpu-box:1234/v1', 'sk-FAKE'))  # encrypted
        self.assertFalse(flagged('http://localhost:1234/v1', 'sk-FAKE'))  # never leaves this computer


class DataUseNoticeTests(unittest.TestCase):
    """A card whose server is not on this computer is not switched on until the user has seen what it receives."""
    REMOTE = {'id': 'remote', 'name': 'Provider', 'url': 'https://api.provider.example/v1', 'model': 'big', 'api_key': 'sk-FAKE-0000'}
    LOCAL = {'id': 'local', 'name': 'Local', 'url': 'http://localhost:11434/v1', 'model': 'small'}
    ORIGIN = 'https://api.provider.example'

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name, 'resources', 'agent')
        self.folder.mkdir(parents=True)
        self.path = self.folder / 'model_card.json'
        self.path.write_text(json.dumps({'cards': [dict(self.REMOTE), dict(self.LOCAL)], 'note': 'kept'}), encoding='utf-8')
        patcher = mock.patch.object(agent, '_SRC_DIR', directory.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(agent.Command_Engine, 'print_help')
        self.print_help = patcher.start()
        self.addCleanup(patcher.stop)
        # No probe for a blank model name and no request may leave this computer.
        patcher = mock.patch.object(agent.urllib.request, 'urlopen', side_effect=AssertionError('a request was sent'))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = SimpleNamespace(events=[])
        self.viewer.broadcast_event = self.viewer.events.append

    def saved(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def set_backend(self, **data):
        self.viewer.events.clear()
        agent.handle_set_backend(self.viewer, {'action': 'set_backend', **data})
        return [event for event in self.viewer.events if event['type'] == 'backend_state'][-1]

    def test_a_remote_card_is_refused_until_accepted_and_nothing_is_sent(self):
        blank_model = {**self.REMOTE, 'model': ''}
        for card in (self.REMOTE, blank_model):
            with self.subTest(model=card['model']):
                viewer = SimpleNamespace()
                self.assertFalse(agent.activate_agent_from_card(viewer, card, quiet=True))
                self.assertFalse(getattr(viewer, 'llm_loaded', False))
                self.assertEqual(viewer.llm_pending_data_use, {
                    'card_id': 'remote', 'ref': {'id': 'remote', 'name': 'Provider', 'url': 'https://api.provider.example/v1'},
                    'name': 'Provider', 'host': 'api.provider.example', 'origin': self.ORIGIN, 'plain_http_key': False})
        self.assertTrue(agent.activate_agent_from_card(SimpleNamespace(), self.LOCAL, quiet=True))

    def test_the_acceptance_is_for_the_server_it_was_given_for(self):
        for accepted, url, activates in (
            (self.ORIGIN, 'https://api.provider.example/v1', True),
            (self.ORIGIN, 'https://api.provider.example/other/path', True),   # the same server
            (self.ORIGIN, 'http://api.provider.example/v1', False),           # no longer encrypted
            (self.ORIGIN, 'https://api.provider.example:8443/v1', False),     # another port
            (self.ORIGIN, 'https://elsewhere.example/v1', False),             # another server
            ('https://other.example', 'https://api.provider.example/v1', False),
            (None, 'https://api.provider.example/v1', False),
        ):
            with self.subTest(accepted=accepted, url=url):
                card = {**self.REMOTE, 'url': url, 'model': 'big', 'data_use_accepted': accepted}
                with mock.patch.object(agent.Command_Engine, 'print_help'):
                    self.assertEqual(agent.activate_agent_from_card(SimpleNamespace(), card, quiet=True), activates)

    def test_a_refused_card_leaves_the_model_loaded_as_it_was(self):
        self.assertTrue(agent.activate_agent_from_card(self.viewer, self.LOCAL, quiet=True))
        self.assertFalse(agent.activate_agent_from_card(self.viewer, self.REMOTE, quiet=True))
        self.assertEqual((self.viewer.llm_loaded, self.viewer.llm_model_name), (True, 'small'))
        # Switching to a card that is let through clears the notice.
        self.assertTrue(self.viewer.llm_pending_data_use)
        self.assertTrue(agent.activate_agent_from_card(self.viewer, self.LOCAL, quiet=True))
        self.assertIsNone(self.viewer.llm_pending_data_use)

    def test_the_page_is_asked_then_continue_saves_the_acceptance_and_switches_the_model_on(self):
        state = self.set_backend(card_id='remote')
        self.assertFalse(state['llm_loaded'])
        self.assertEqual(state['data_use'], self.viewer.llm_pending_data_use)
        self.assertEqual((state['data_use']['name'], state['data_use']['host'], state['data_use']['plain_http_key']),
                         ('Provider', 'api.provider.example', False))
        self.assertNotIn('data_use_accepted', self.saved()['cards'][0])

        state = self.set_backend(card_id='remote', accept_data_use=True)
        self.assertEqual((state['llm_loaded'], state['llm_model_name'], 'data_use' in state), (True, 'big', False))
        document = self.saved()
        self.assertEqual(document['cards'][0], {**self.REMOTE, 'data_use_accepted': self.ORIGIN})
        self.assertEqual(document['cards'][1], self.LOCAL)
        self.assertEqual(document['note'], 'kept')
        self.assertEqual(os.listdir(self.folder), ['model_card.json'])
        # The page keeps its own copy of the cards, and is told, so its next Save keeps this.
        self.assertIn({'type': 'model_card_updated', 'card_id': 'remote', 'fields': {'data_use_accepted': self.ORIGIN}}, self.viewer.events)
        # Never asked again for this card: not now, and not after a restart.
        self.assertEqual(self.set_backend(card_id=None)['llm_loaded'], False)
        self.assertTrue(self.set_backend(card_id='remote')['llm_loaded'])
        restarted = SimpleNamespace()
        self.assertTrue(agent.activate_agent_from_card(restarted, agent.load_model_cards()[0], quiet=True))

    def test_continue_counts_only_for_the_card_the_notice_was_about(self):
        # No notice was shown, so a request that says it was accepted accepts nothing.
        state = self.set_backend(card_id='remote', accept_data_use=True)
        self.assertEqual((state['llm_loaded'], bool(state.get('data_use'))), (False, True))
        self.assertNotIn('data_use_accepted', self.saved()['cards'][0])
        # The notice is about the remote card; accepting it for another card does not carry over.
        other = {**self.REMOTE, 'id': 'other', 'name': 'Another', 'url': 'https://another.example/v1'}
        self.path.write_text(json.dumps({'cards': [dict(self.REMOTE), other]}), encoding='utf-8')
        state = self.set_backend(card_id='other', accept_data_use=True)
        self.assertEqual((state['llm_loaded'], state['data_use']['name']), (False, 'Another'))
        self.assertNotIn('data_use_accepted', self.saved()['cards'][1])

    def test_cancel_clears_the_notice_and_the_model_loaded_stays(self):
        self.assertTrue(self.set_backend(card_id='local')['llm_loaded'])
        self.assertIn('data_use', self.set_backend(card_id='remote'))
        self.viewer.events.clear()
        agent.handle_decline_data_use(self.viewer, {})
        [state] = self.viewer.events
        self.assertEqual((state['llm_loaded'], state['llm_model_name'], 'data_use' in state), (True, 'small', False))
        self.assertNotIn('data_use_accepted', self.saved()['cards'][0])
        # A cancelled notice cannot be accepted afterwards: the notice is asked again, and the model is still the local one.
        state = self.set_backend(card_id='remote', accept_data_use=True)
        self.assertEqual((state['llm_model_name'], 'data_use' in state), ('small', True))
        self.assertNotIn('data_use_accepted', self.saved()['cards'][0])
        # Switching the agent off clears one too.
        self.assertIn('data_use', self.set_backend(card_id='remote'))
        self.assertNotIn('data_use', self.set_backend(card_id=None))

    def test_a_page_opened_while_a_notice_waits_is_sent_it(self):
        self.set_backend(card_id='remote')
        with mock.patch.object(agent, 'load_agent_history', return_value=[]):
            state = agent._extend_initial_web_state(self.viewer, {})
        self.assertEqual(state['data_use'], self.viewer.llm_pending_data_use)
        self.assertNotIn('warning', state)

    def test_the_terminal_is_told_which_card_waits_and_the_page_is_shown_the_notice(self):
        from commands import agent as agent_command
        cases = {'a page open': (True, True, False), 'windows shown, no page': (True, False, True), 'no windows': (False, False, False)}
        for label, (windows, page, opens) in cases.items():
            with self.subTest(label):
                viewer = SimpleNamespace(events=[], shows_windows=lambda windows=windows: windows, open_agent_ui=mock.Mock(return_value=True),
                                         web_server=SimpleNamespace(has_event_client=lambda client, page=page: page))
                viewer.broadcast_event = viewer.events.append
                self.assertFalse(agent.activate_agent_from_card(viewer, self.REMOTE, quiet=True))
                message = agent.review_data_use(viewer)
                self.assertIn("Model card 'Provider' sends your data to api.provider.example", str(message))
                self.assertEqual(viewer.open_agent_ui.called, opens)
                self.assertEqual([e['data_use']['name'] for e in viewer.events if e['type'] == 'backend_state'], ['Provider'])
        self.assertIsNone(agent.review_data_use(SimpleNamespace()))

    def test_the_command_fails_with_the_reason_and_does_not_switch_the_model_on(self):
        from commands import agent as agent_command
        engine = agent_command.Command_Engine
        self.path.write_text(json.dumps({'cards': [dict(self.REMOTE)]}), encoding='utf-8')
        for args in (['<Provider>'], ['select', 'cluster', '1']):
            with self.subTest(args=args), mock.patch.object(engine, 'command_failed') as failed, \
                    mock.patch.object(engine, 'command_succeeded') as succeeded, \
                    mock.patch.object(agent_command, 'register'), \
                    mock.patch.object(agent_command, 'run_web_agent_query') as query:
                viewer = SimpleNamespace(events=[], shows_windows=lambda: True, open_agent_ui=mock.Mock(return_value=True))
                viewer.broadcast_event = viewer.events.append
                with redirect_stdout(io.StringIO()):
                    agent_command.run(viewer, args)
                failed.assert_called_once()
                self.assertIn('not on this computer', str(failed.call_args.args[1]))
                succeeded.assert_not_called()
                query.assert_not_called()
                viewer.open_agent_ui.assert_called_once_with(show_existing_dialog=False)
                self.assertFalse(getattr(viewer, 'llm_loaded', False))

    def test_a_card_that_sends_a_key_over_plain_http_is_warned_about_each_time_it_is_switched_on(self):
        card = {'id': 'lan', 'name': 'LAN box', 'url': 'http://gpu-box.lan:1234/v1', 'model': 'm', 'api_key': 'sk-FAKE-0000'}
        self.path.write_text(json.dumps({'cards': [{**card, 'data_use_accepted': 'http://gpu-box.lan:1234'}]}), encoding='utf-8')
        for _ in range(2):
            self.print_help.reset_mock()
            state = self.set_backend(card_id='lan')
            self.assertTrue(state['llm_loaded'])
            self.assertIn('sends its API key over plain http:// to gpu-box.lan:1234', state['warning'])
            self.assertNotIn('sk-FAKE-0000', state['warning'])
            self.assertIn('plain http://', str(self.print_help.call_args.args[1]))
            self.set_backend(card_id=None)
        with mock.patch.object(agent, 'load_agent_history', return_value=[]):
            self.set_backend(card_id='lan')
            self.assertIn('warning', agent._extend_initial_web_state(self.viewer, {}))
        # No warning for a card that cannot send a key that way, nor once it is switched off.
        self.assertNotIn('warning', self.set_backend(card_id='local'))
        self.assertNotIn('warning', self.set_backend(card_id=None))
        # The notice for such a card says so too.
        self.path.write_text(json.dumps({'cards': [card]}), encoding='utf-8')
        self.assertTrue(self.set_backend(card_id='lan')['data_use']['plain_http_key'])


class SnapshotPathsTests(unittest.TestCase):
    """The snapshot the model gets names files, not the folders of the user's computer; MCP's summary is unchanged."""
    DIRECTORY = 'C:\\Users\\Someone\\Private Lab\\Project'

    def viewer(self):
        from desktop.Viewer_Inspection import ViewerInspectionService
        viewer = SimpleNamespace(
            n_nodes=2, full_headers=['A', 'B'], visible_mask=[True, True], selected_indices=[], edges=[(0, 1)],
            metadata={'score': {'type': 'numeric', 'values': [1.0, 2.0]}}, cluster_labels=[1, 1], group_labels=[set(), set()],
            current_slider_threshold=0.5)
        config = SimpleNamespace(
            NODE_FASTA_FILE=self.DIRECTORY + '\\Input_Files\\set.fasta', INPUT_HDF5='/home/someone/lab/network.h5',
            MSA_FILE='alignment.fasta', EMBED_FILE=self.DIRECTORY + '\\Embeddings\\set.h5',
            TARGET_CACHE_PATH=self.DIRECTORY + '\\Cache_Files\\version_00.h5', SIMILARITY_THRESHOLD=0.1)
        viewer.viewer_inspection = ViewerInspectionService(viewer, config)
        return viewer

    def test_the_model_gets_file_names_and_mcp_keeps_the_paths(self):
        viewer = self.viewer()
        text = agent.get_viewer_session_context(viewer)
        document = json.loads(text.split('---\n', 1)[1])
        self.assertEqual(document['summary']['inputs'], {
            'node_fasta': 'set.fasta', 'network_hdf5': 'network.h5', 'msa': 'alignment.fasta',
            'embedding': 'set.h5', 'layout_cache': 'version_00.h5'})
        for private in ('Someone', 'Private Lab', '/home/', 'C:\\\\Users', 'Input_Files'):
            self.assertNotIn(private, text)
        # The summary MCP clients read is the service's own, and still has them.
        service = viewer.viewer_inspection
        snapshot = service.capture_snapshot()
        inputs = service.snapshots.execute('get_summary', snapshot)['inputs']
        self.assertEqual(inputs['node_fasta'], self.DIRECTORY + '\\Input_Files\\set.fasta')
        self.assertEqual(inputs['network_hdf5'], '/home/someone/lab/network.h5')
        self.assertEqual(service.get_summary()['inputs']['layout_cache'], self.DIRECTORY + '\\Cache_Files\\version_00.h5')

    def test_the_embedding_file_and_the_generation_parameters_lose_their_folders_too(self):
        summary = {
            'inputs': {'node_fasta': '/data/set.fasta', 'msa': None, 'embedding': '', 'layout_cache': 'Cache_Files\\a\\v.h5'},
            'embedding': {'available': True, 'file': 'D:/emb/set.h5', 'model': 'esm'},
            'provenance': {'status': 'valid', 'generation_parameters': {
                'input': 'C:\\data\\x.h5', 'steps': 3, 'nested': [{'path': '/home/u/y.fasta', 'name': 'a/b not a path'}],
                'mount': '\\\\server\\share\\z.h5'}},
            'metadata_preview': [{'name': '/looks/like/a/path', 'type': 'text'}],
            'alignment': {'reference': '/not/a/path|header'},
        }
        result = agent.agent_summary(summary)
        self.assertEqual(result['inputs'], {'node_fasta': 'set.fasta', 'msa': None, 'embedding': '', 'layout_cache': 'v.h5'})
        self.assertEqual(result['embedding'], {'available': True, 'file': 'set.h5', 'model': 'esm'})
        self.assertEqual(result['provenance']['generation_parameters'],
                         {'input': 'x.h5', 'steps': 3, 'nested': [{'path': 'y.fasta', 'name': 'a/b not a path'}], 'mount': 'z.h5'})
        # Names a command must spell exactly are left alone: only paths of the user's folders change.
        self.assertEqual(result['metadata_preview'], summary['metadata_preview'])
        self.assertEqual(result['alignment'], summary['alignment'])
        self.assertEqual(summary['inputs']['node_fasta'], '/data/set.fasta')  # the argument is not changed
        self.assertEqual(agent.agent_summary({'node_count': 3}), {'node_count': 3})
