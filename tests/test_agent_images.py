"""Image validation, multimodal transport, history and retry regression tests."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tests.agent_fixtures import image_bytes
from web_ui import agent_backend as agent
from web_ui.agent_images import inspect_source, validate_attachments, message_content, history_messages


def attachment():
    return {'name': 'test.png', 'data_url': 'data:image/png;base64,' + base64.b64encode(image_bytes()).decode()}


class ImageTests(unittest.TestCase):
    def test_static_formats_and_animation(self):
        for fmt in ('PNG', 'JPEG', 'WEBP', 'GIF', 'BMP', 'AVIF'):
            with self.subTest(format=fmt):
                self.assertTrue(inspect_source(image_bytes(fmt))['mime_type'].startswith('image/'))
        for fmt in ('PNG', 'GIF', 'WEBP', 'TIFF', 'AVIF'):
            with self.subTest(animated=fmt), self.assertRaisesRegex(ValueError, 'multi-frame'):
                inspect_source(image_bytes(fmt, animated=True))

    def test_invalid_sources_svg_and_limits(self):
        for raw in (b'', b'not an image', b'%PDF-1.7', b'x' * (20 * 1024 * 1024 + 1)):
            with self.assertRaises(ValueError):
                inspect_source(raw)
        self.assertEqual(inspect_source(b'<svg xmlns="http://www.w3.org/2000/svg"><style>rect { fill: red; }</style><rect width="10" height="10"/></svg>')['mime_type'], 'image/svg+xml')
        for content in ('<animate/>', '<style>@keyframes spin {}</style>', '<image href="data:image/gif;base64,abc"/>'):
            with self.assertRaises(ValueError):
                inspect_source(f'<svg>{content}</svg>'.encode())

    def test_backend_validates_normalized_content(self):
        item = attachment()
        self.assertEqual(validate_attachments([item]), [item])
        self.assertEqual(validate_attachments(None), [])
        for invalid in ([item] * 11, ['bad'], [{'data_url': 'https://example.com/a.png'}], [{'data_url': 'data:image/png;base64,@@@@'}]):
            with self.assertRaises(ValueError):
                validate_attachments(invalid)
        for raw in (image_bytes('JPEG'), image_bytes('PNG', animated=True), image_bytes(size=(1601, 1))):
            with self.assertRaises(ValueError):
                validate_attachments([{'data_url': 'data:image/png;base64,' + base64.b64encode(raw).decode()}])

    def test_api_preserves_text_images_and_history_despite_options(self):
        items = [attachment(), attachment()]
        prior = history_messages([{'role': 'user', 'content': 'before', 'attachments': items}])
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices': [{'message': {'content': 'seen'}}]}).encode()
        with mock.patch.object(agent.urllib.request, 'urlopen', return_value=response) as send:
            agent.call_api('http://local/v1', 'vision', 'system', message_content('', items), history=prior, options={'messages': []})
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(len(payload['messages'][-1]['content']), 2)
        self.assertEqual(payload['messages'][1]['content'][1]['image_url']['url'], items[0]['data_url'])
        self.assertEqual(message_content('plain'), 'plain')

    def test_provider_error_is_visible_without_retry(self):
        error = urllib.error.HTTPError('http://local', 400, 'Bad request', {}, io.BytesIO(b'image input unsupported'))
        with mock.patch.object(agent.urllib.request, 'urlopen', side_effect=error) as send:
            with self.assertRaisesRegex(ValueError, 'image input unsupported'):
                agent.call_api('http://local', 'text-only', '', message_content('', [attachment()]))
        self.assertEqual(send.call_count, 1)

    def test_refinement_receives_same_images_and_prior_history(self):
        items = [attachment()]
        worker = agent.RefinementWorker('server', 'http://local', 'vision', 'question', 'zoom', '{}', 0, None, attachments=items, history=[{'role': 'user', 'content': 'prior'}])
        with mock.patch.object(agent, 'call_api', return_value={'content': 'answer'}) as call:
            worker.run()
        self.assertEqual(call.call_args.args[3][-1]['image_url']['url'], items[0]['data_url'])
        self.assertEqual(call.call_args.kwargs['history'][0]['content'], 'prior')

    def test_history_roundtrip_and_legacy_messages(self):
        viewer = SimpleNamespace(llm_history=[{'role': 'user', 'content': 'legacy'} for _ in range(12)], broadcast_event=mock.Mock())
        items = [attachment()]
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(agent, 'get_agent_history_path', return_value=str(Path(folder) / 'history.json')):
            agent.save_and_broadcast_agent_response(viewer, '', 'answer', '', '', '{}', turn={'attachments': items, 'submission_id': 'one'})
            restored = agent.load_agent_history(viewer)
        self.assertEqual(restored[-2]['attachments'], items)
        self.assertEqual(len(restored), 14)
        self.assertEqual(history_messages(restored)[0]['content'], 'legacy')
        self.assertEqual(history_messages(restored)[-2]['content'][0]['type'], 'image_url')

    def test_query_acceptance_and_duplicate_retry(self):
        viewer = SimpleNamespace(llm_loaded=True, llm_backend='server', llm_model_name='vision', llm_history=[], broadcast_event=mock.Mock())
        worker = mock.Mock()
        with mock.patch.object(agent, 'AgentWorker', return_value=worker) as create, mock.patch.object(agent, 'get_viewer_session_context', return_value=''), mock.patch.object(agent, 'load_agent_history', return_value=[]):
            agent.run_web_agent_query(viewer, '', [attachment()], 'same-id')
            agent.run_web_agent_query(viewer, '', [attachment()], 'same-id')
        create.assert_called_once()
        worker.start.assert_called_once()
        self.assertEqual(create.call_args.args[4][0]['type'], 'image_url')
        self.assertEqual(viewer.broadcast_event.call_args.args[0]['type'], 'agent_accepted')
        self.assertEqual(viewer.broadcast_event.call_args.args[0]['submission_id'], 'same-id')

    def test_invalid_query_does_not_start_worker(self):
        viewer = SimpleNamespace(llm_loaded=True, broadcast_event=mock.Mock())
        with mock.patch.object(agent, 'AgentWorker') as create:
            agent.run_web_agent_query(viewer, 'question', [{'data_url': 'bad'}], 'invalid')
        create.assert_not_called()
        self.assertEqual(viewer.broadcast_event.call_args.args[0]['type'], 'agent_error')
        self.assertEqual(viewer.broadcast_event.call_args.args[0]['submission_id'], 'invalid')


class ReplyTimeoutTests(unittest.TestCase):
    def test_a_model_reply_waits_for_the_reply_timeout(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({'choices': [{'message': {'content': 'slow'}}]}).encode()
        with mock.patch.object(agent.urllib.request, 'urlopen', return_value=response) as send:
            self.assertEqual(agent.call_api('http://local/v1', 'slow', 'system', 'question')['content'], 'slow')
        self.assertEqual(send.call_args.kwargs['timeout'], agent.AGENT_REPLY_TIMEOUT_SECONDS)
        self.assertEqual(agent.AGENT_REPLY_TIMEOUT_SECONDS, 300.0)


if __name__ == '__main__':
    unittest.main()


def saved_turn(number, *, images=0, explanation=None, commands=(), output_padding=0, submission_id=None):
    """The two entries a finished turn leaves in the saved history (agent_backend.save_and_broadcast_agent_response)."""
    explanation = explanation or f'answer {number}'
    results = [{'command': command, 'status': status, 'messages': [{'status': status, 'text': message}] if message else []}
               for command, status, message in commands]
    terminal_output = json.dumps({'result': {'status': 'failed', 'commands': results},
                                  'output': {'stdout': {'text': f'node_header_{number} ' + 'x' * output_padding}, 'stderr': {'text': ''}}})
    shown = '\n'.join(command for command, _, _ in commands)
    content = explanation
    if commands:
        content += f'\n\nCommands submitted:\n```\n{shown}\n```\n\nTerminal output:\n```\n{terminal_output}\n```'
    user = {'role': 'user', 'content': f'question {number}'}
    if images:
        user['attachments'] = [attachment() for _ in range(images)]
    if submission_id:
        user['submission_id'] = submission_id
    return [user, {'role': 'assistant', 'content': content, 'explanation': explanation, 'commands': shown,
                   'terminal_output': terminal_output if commands else '', 'tokens': None, 'reasoning': ''}]


def conversation(count, **options):
    return [message for number in range(count) for message in saved_turn(number, **options)]


def texts(messages):
    return [message['content'] if isinstance(message['content'], str) else
            ' '.join(part.get('text', '<image>') for part in message['content']) for message in messages]


class HistoryBoundTests(unittest.TestCase):
    """A request re-sends the last few turns within a token budget; older turns lose their images and command output."""

    def test_the_limits_are_named_and_what_the_user_was_told(self):
        from web_ui import agent_images
        self.assertEqual((agent_images.HISTORY_MAX_TURNS, agent_images.HISTORY_TOKEN_BUDGET, agent_images.CHARS_PER_TOKEN),
                         (10, 8000, 4))
        self.assertEqual([agent_images.estimate_tokens(text) for text in ('', 'x', 'x' * 4, 'x' * 5, 'x' * 8000)], [0, 1, 1, 2, 2000])

    def test_at_most_ten_turns_are_sent(self):
        history = conversation(25)
        sent = history_messages(history)
        self.assertEqual(len(sent), 20)
        self.assertEqual([m['content'] for m in sent if m['role'] == 'user'], [f'question {n}' for n in range(15, 25)])
        self.assertEqual([m['role'] for m in sent], ['user', 'assistant'] * 10)
        # Fewer turns than the limit are all sent; a smaller limit is a smaller number of turns.
        self.assertEqual(len(history_messages(conversation(4))), 8)
        self.assertEqual(len(history_messages(history, max_turns=3)), 6)
        self.assertEqual(history_messages(history, max_turns=0), [])
        self.assertEqual(history_messages([]), [])

    def test_the_newest_turn_is_whole_and_an_older_turn_is_a_summary(self):
        commands = [('select "a"', 'succeeded', ''), ('save x.h5', 'cancelled', 'Discarded in the Agent page.'), ('print', 'skipped', '')]
        history = saved_turn(0, images=2, commands=commands, output_padding=50) + saved_turn(1, images=1, commands=commands)
        older_user, older_assistant, newest_user, newest_assistant = history_messages(history)

        # An older turn: the user's text and a note of the images, the explanation and one line per command.
        self.assertEqual(older_user['content'], 'question 0\n[2 image(s) attached here are not sent again]')
        self.assertEqual(older_assistant['content'], 'answer 0\n\nCommands run:\n- select "a": succeeded\n'
                                                     '- save x.h5: cancelled (Discarded in the Agent page.)\n- print: skipped')
        self.assertNotIn('node_header_0', older_assistant['content'])
        self.assertNotIn('Terminal output', older_assistant['content'])
        # The newest: as it was saved, images and command output included.
        self.assertEqual([part['type'] for part in newest_user['content']], ['text', 'image_url'])
        self.assertEqual(newest_assistant['content'], history[3]['content'])
        self.assertIn('node_header_1', newest_assistant['content'])
        # No image of an older turn is anywhere in the request.
        self.assertFalse(any(isinstance(m['content'], list) and any(p['type'] == 'image_url' for p in m['content'])
                             for m in (older_user, older_assistant)))

    def test_older_turns_are_added_while_they_fit_the_budget(self):
        from web_ui import agent_images
        history = [message for number in range(10) for message in saved_turn(number, explanation='x' * 6000)]  # 1,500 tokens each
        sent = history_messages(history)
        # The newest and the four before it: a sixth would take the request past 8,000 tokens.
        self.assertEqual([m['content'] for m in sent if m['role'] == 'user'], [f'question {n}' for n in range(5, 10)])
        self.assertLessEqual(agent_images._tokens(sent), agent_images.HISTORY_TOKEN_BUDGET)
        self.assertGreater(agent_images._tokens(sent) + 1500, agent_images.HISTORY_TOKEN_BUDGET)
        self.assertEqual(len(history_messages(history, budget_tokens=100_000)), 20)
        # An image counts for what it costs: five images of the newest turn leave no room for an older one.
        with_images = history[:-2] + saved_turn(9, explanation='x' * 6000, images=5)
        sent = history_messages(with_images)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sum(isinstance(part, dict) and part['type'] == 'image_url' for part in sent[0]['content']), 5)
        # A turn that does not fit ends the history: a smaller one before it is not skipped to.
        mixed = (saved_turn(0) + saved_turn(1, explanation='x' * 40000) + saved_turn(2))
        self.assertEqual([m['content'] for m in history_messages(mixed) if m['role'] == 'user'], ['question 2'])

    def test_an_oversized_newest_turn_is_summarized_but_keeps_its_images(self):
        commands = [('select "a"', 'succeeded', '')]
        history = saved_turn(0) + saved_turn(1, images=1, commands=commands, output_padding=100_000)
        sent = history_messages(history)
        self.assertEqual(len(sent), 4)
        user, assistant = sent[2:]
        self.assertEqual([part['type'] for part in user['content']], ['text', 'image_url'])
        self.assertEqual(assistant['content'], 'answer 1\n\nCommands run:\n- select "a": succeeded')
        self.assertLess(sum(len(str(m['content'])) for m in sent if m['role'] == 'assistant'), 200)

    def test_the_saved_history_is_not_changed(self):
        import copy
        history = conversation(15, images=1, commands=[('select "a"', 'succeeded', '')])
        before = copy.deepcopy(history)
        history_messages(history)
        self.assertEqual(history, before)

    def test_a_summary_lists_the_first_commands_and_cuts_long_messages(self):
        commands = [(f'color #cluster_{n}# red', 'failed', 'because ' + 'y' * 300) if n == 0 else (f'color #cluster_{n}# red', 'succeeded', '')
                    for n in range(30)]
        older = history_messages(saved_turn(0, commands=commands) + saved_turn(1))[1]['content']
        lines = older.split('\n\nCommands run:\n')[1].split('\n')
        self.assertEqual(len(lines), 21)
        self.assertEqual(lines[0], f"- color #cluster_0# red: failed (because {'y' * 152})")  # 160 characters of the message
        self.assertEqual(lines[19], '- color #cluster_19# red: succeeded')
        self.assertEqual(lines[20], '- and 10 more commands')

    def test_odd_histories_are_still_sent(self):
        # Output that is not the portal's JSON: the lines the model sent. None at all: just the explanation.
        history = saved_turn(0, commands=[('select "a"', 'succeeded', '')])
        history[1]['terminal_output'] = 'not json {'
        older = history_messages(history + saved_turn(1))[1]['content']
        self.assertEqual(older, 'answer 0\n\nCommands submitted:\nselect "a"')
        self.assertEqual(history_messages(saved_turn(0) + saved_turn(1))[1]['content'], 'answer 0')
        # Entries saved before the explanation and command fields, and a history that starts with an answer.
        legacy = [{'role': 'assistant', 'content': 'orphan'},
                  {'role': 'user', 'content': 'old'},
                  {'role': 'assistant', 'content': 'old answer\n\nCommands submitted:\n```\nx\n```\n\nTerminal output:\n```\n{"secret": 1}\n```'},
                  {'role': 'user', 'content': 'unanswered'}]
        sent = history_messages(legacy + saved_turn(5))
        self.assertEqual(texts(sent)[:5], ['orphan', 'old', 'old answer\n\nCommands submitted:\n```\nx\n```', 'unanswered', 'question 5'])
        self.assertNotIn('secret', ' '.join(texts(sent)))


class RequestsUseTheBoundTests(unittest.TestCase):
    """The history a request holds is bounded, and the saved one stays whole."""

    def test_the_first_call_the_explain_call_and_the_saved_file(self):
        history = conversation(30, commands=[('select "a"', 'succeeded', '')])
        viewer = SimpleNamespace(llm_loaded=True, llm_backend='server', llm_model_name='m', llm_history=history, events=[])
        viewer.broadcast_event = viewer.events.append
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(agent, 'get_agent_history_path', return_value=str(Path(folder) / 'history.json')), \
                mock.patch.object(agent, 'get_viewer_session_context', return_value=''), \
                mock.patch.object(agent, 'AgentWorker') as worker, mock.patch.object(agent, 'RefinementWorker') as refinement:
            self.assertTrue(agent.run_web_agent_query(viewer, 'next question'))
            sent = worker.call_args.args[5]
            agent.start_refinement_worker(viewer, 'next question', 'text', 'select "b"', '{}', '{}', '')
            explained = refinement.call_args.kwargs['history']
            agent.save_and_broadcast_agent_response(viewer, 'next question', 'answer', '', '', '{}')
            saved = json.loads((Path(folder) / 'history.json').read_text(encoding='utf-8'))
        for messages in (sent, explained):
            self.assertEqual(len(messages), 20)
            self.assertEqual(messages[-2]['content'], 'question 29')
        # The history in memory, the page's view of it and the file on disk hold every message.
        self.assertEqual(len(viewer.llm_history), 62)
        self.assertEqual(len(saved), 62)
        self.assertEqual(len(viewer.events[-1]['llm_history']), 62)
