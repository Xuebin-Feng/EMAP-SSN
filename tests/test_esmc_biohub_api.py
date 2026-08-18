"""Tests for ESMC 6B use of shared Biohub credentials."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resources.pLM_models import esmc_6b_api


class FakeForgeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.token = kwargs["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}


class ESMCBiohubAPITests(unittest.TestCase):
    def test_load_model_uses_shared_settings_and_esmc_mapping(self):
        forge_module = types.ModuleType("esm.sdk.forge")
        forge_module.ESMCForgeInferenceClient = FakeForgeClient
        settings = {
            "ESM_API_TOKEN": "shared-token",
            "ESM_API_URL": "https://biohub.example",
            "ESM3_MODEL": "esm3-large-2024-03",
        }

        def load_from_terminal(*, prompt_callback):
            self.assertEqual(prompt_callback(), "terminal-token")
            return settings

        with (
            mock.patch.dict(sys.modules, {"esm.sdk.forge": forge_module}),
            mock.patch.object(
                esmc_6b_api,
                "_terminal_token_prompt",
                return_value="terminal-token",
            ) as terminal_prompt,
            mock.patch.object(
                esmc_6b_api.Biohub_API,
                "load_api_settings",
                side_effect=load_from_terminal,
            ) as load_settings,
        ):
            client = esmc_6b_api.load_model("esmc_6b", None)

        load_settings.assert_called_once()
        terminal_prompt.assert_called_once_with(False)
        self.assertEqual(client.kwargs["model"], "esmc-6b-2024-12")
        self.assertEqual(client.kwargs["url"], settings["ESM_API_URL"])
        self.assertEqual(client.kwargs["token"], settings["ESM_API_TOKEN"])
        self.assertEqual(client._ssn_biohub_settings, settings)

    def test_authentication_failure_refreshes_once_then_returns_embedding(self):
        settings = {
            "ESM_API_TOKEN": "old-token",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        refreshed = dict(settings, ESM_API_TOKEN="new-token")
        client = SimpleNamespace(
            _ssn_biohub_settings=settings,
            _ssn_auth_refresh_attempted=False,
            token="old-token",
            headers={"Authorization": "Bearer old-token"},
        )
        expected = np.ones((4, 3), dtype=np.float32)

        with (
            mock.patch.object(
                esmc_6b_api,
                "_request_embedding",
                side_effect=[
                    esmc_6b_api.Biohub_API.BiohubAuthenticationError(401),
                    expected,
                ],
            ) as request,
            mock.patch.object(
                esmc_6b_api.Biohub_API,
                "refresh_api_token",
                return_value=refreshed,
            ) as refresh,
        ):
            result = esmc_6b_api.get_embedding(
                "ACDE",
                client,
                None,
                np.float32,
            )

        np.testing.assert_array_equal(result, expected)
        self.assertEqual(request.call_count, 2)
        refresh.assert_called_once()
        self.assertEqual(client.headers["Authorization"], "Bearer new-token")

    def test_second_authentication_failure_is_not_retried(self):
        settings = {
            "ESM_API_TOKEN": "old-token",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        client = SimpleNamespace(
            _ssn_biohub_settings=settings,
            _ssn_auth_refresh_attempted=False,
            token="old-token",
            headers={"Authorization": "Bearer old-token"},
        )

        with (
            mock.patch.object(
                esmc_6b_api,
                "_request_embedding",
                side_effect=[
                    esmc_6b_api.Biohub_API.BiohubAuthenticationError(401),
                    esmc_6b_api.Biohub_API.BiohubAuthenticationError(403),
                ],
            ) as request,
            mock.patch.object(
                esmc_6b_api.Biohub_API,
                "refresh_api_token",
                return_value=dict(settings, ESM_API_TOKEN="new-token"),
            ) as refresh,
        ):
            with self.assertRaises(esmc_6b_api.Biohub_API.BiohubAuthenticationError):
                esmc_6b_api.get_embedding("ACDE", client, None, np.float32)

        self.assertEqual(request.call_count, 2)
        refresh.assert_called_once()


if __name__ == "__main__":
    unittest.main()
