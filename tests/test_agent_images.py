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
from PIL import Image
from web_ui import agent_backend as agent
from web_ui.agent_images import inspect_source, validate_attachments, message_content, history_messages


def image_bytes(fmt='PNG', animated=False, size=(20, 10)):
    out = io.BytesIO()
    kwargs = {'save_all': True, 'append_images': [Image.new('RGB', size, 'blue')], 'duration': 100} if animated else {}
    Image.new('RGB', size, 'red').save(out, format=fmt, **kwargs)
    return out.getvalue()


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


if __name__ == '__main__':
    unittest.main()
