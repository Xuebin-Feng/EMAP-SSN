"""In-Viewer agent backend (web_ui/agent_backend): model-card activation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
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
