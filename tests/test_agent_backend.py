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
    SAVED = {'cards': [{'id': 'mine', 'name': 'My API', 'url': 'https://api.example/v1',
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
                         ('https://api.example/v1', 'big', 'sk-SECRET'))
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
                viewer, {'name': 'Old', 'url': 'http://old.example/v1', 'model': 'old-model', 'temperature': 0.5}, quiet=True))
        return viewer

    def assert_old_model_kept(self, viewer):
        self.assertTrue(viewer.llm_loaded)
        self.assertEqual((viewer.llm_url, viewer.llm_model_name, viewer.llm_temperature),
                         ('http://old.example/v1', 'old-model', 0.5))

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
                card = {'name': 'New', 'url': 'http://new.example/v1', 'model': 'x', 'temperature': temperature}
                with mock.patch.object(agent.Command_Engine, 'print_help') as print_help:
                    self.assertFalse(agent.activate_agent_from_card(viewer, card, quiet=True))
                self.assert_old_model_kept(viewer)
                self.assertIn("Model card 'New' has an invalid temperature", str(print_help.call_args.args[1]))

    def test_a_usable_card_still_replaces_the_loaded_model(self):
        viewer = self.loaded_viewer()
        viewer.llm_history = [{'role': 'user', 'content': 'earlier'}]
        card = {'name': 'New', 'url': 'http://new.example/v1', 'model': 'x', 'temperature': '0.25'}
        self.assertTrue(agent.activate_agent_from_card(viewer, card, quiet=True))
        self.assertEqual((viewer.llm_url, viewer.llm_model_name, viewer.llm_temperature, viewer.llm_loaded),
                         ('http://new.example/v1', 'x', 0.25, True))
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
        card = {'name': 'New', 'url': 'http://new.example/v1', 'model': 'x'}
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
    CARDS = {'cards': [{'id': 'a', 'name': 'Alpha', 'url': 'http://alpha.example/v1', 'model': 'alpha-1'},
                       {'id': 'b', 'name': 'Beta', 'url': 'http://beta.example/v1', 'model': 'beta-1'}]}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.agent_folder = Path(directory.name, 'resources', 'agent')
        self.agent_folder.mkdir(parents=True)
        (self.agent_folder / 'model_card.json').write_text(json.dumps(self.CARDS), encoding='utf-8')
        (self.agent_folder / 'system_prompt.md').write_text('Prompt.', encoding='utf-8')
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
        viewer = self.viewer(llm_loaded=True, llm_backend='server', llm_url='http://old/v1', llm_model_name='old')
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
