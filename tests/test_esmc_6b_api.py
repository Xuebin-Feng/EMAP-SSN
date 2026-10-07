"""The esmc_6b_api pLM plugin: ESMC 6B embeddings through the Biohub API.

Credential storage itself is covered by test_biohub_api.
"""

import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    LogitsConfig,
    LogitsOutput,
)


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


class FakeInferenceClient:
    """ESMCForgeInferenceClient replying with real esm.sdk.api results.

    encode() returns the queued encode_replies (such as ESMProteinError) first,
    then a protein tensor; logits() returns logits_reply or the embeddings.
    """

    def __init__(self, embeddings, encode_replies=(), logits_reply=None):
        self.embeddings = embeddings
        self.encode_replies = list(encode_replies)
        self.logits_reply = logits_reply
        self.encoded = []
        self.configs = []

    def encode(self, protein):
        self.encoded.append(protein)
        if self.encode_replies:
            return self.encode_replies.pop(0)
        return ESMProteinTensor(sequence=torch.zeros(len(protein.sequence) + 2))

    def logits(self, protein_tensor, config):
        self.configs.append(config)
        if self.logits_reply is not None:
            return self.logits_reply
        return LogitsOutput(embeddings=self.embeddings)


def bf16_rows(count, width=3):
    """Rows 0..count-1 of consecutive integers as a (1, count, width) bf16 batch."""
    values = torch.arange(count * width, dtype=torch.float32)
    return values.reshape(1, count, width).to(torch.bfloat16)


# Rows 1..4 of bf16_rows(6): the residues of "ACDE" without BOS and EOS rows.
ACDE_ROWS = np.arange(18, dtype=np.float32).reshape(6, 3)[1:5]


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


class EmbeddingRequestTests(unittest.TestCase):
    """_request_embedding and the retries of get_embedding, through a fake
    client that returns the real esm.sdk.api result types."""

    AuthError = esmc_6b_api.Biohub_API.BiohubAuthenticationError

    def test_bf16_embeddings_drop_boundary_rows_in_the_requested_dtype(self):
        for dtype in (np.float32, np.float16):
            with self.subTest(dtype=dtype.__name__):
                client = FakeInferenceClient(bf16_rows(6))
                result = esmc_6b_api._request_embedding("ACDE", client, dtype)
                self.assertEqual(result.dtype, dtype)
                np.testing.assert_array_equal(result, ACDE_ROWS)
                self.assertEqual(client.encoded, [ESMProtein(sequence="ACDE")])
                self.assertEqual(
                    client.configs,
                    [LogitsConfig(sequence=True, return_embeddings=True)],
                )

    def test_embedding_rows_must_match_the_sequence(self):
        cases = {
            5: "API embedding length does not match the cleaned sequence length (3 != 4).",
            1: "API returned an unexpected embedding shape: (1, 3)",
        }
        for rows, message in cases.items():
            with self.subTest(rows=rows):
                client = FakeInferenceClient(bf16_rows(rows))
                with self.assertRaises(RuntimeError) as raised:
                    esmc_6b_api._request_embedding("ACDE", client, np.float32)
                self.assertEqual(str(raised.exception), message)

    def test_sdk_errors_become_authentication_or_runtime_errors(self):
        for status in (401, 403):
            with self.subTest(stage="encode", status=status):
                client = FakeInferenceClient(
                    bf16_rows(6), encode_replies=[ESMProteinError(status, "denied")]
                )
                with self.assertRaises(self.AuthError) as raised:
                    esmc_6b_api._request_embedding("ACDE", client, np.float32)
                self.assertEqual(raised.exception.error_code, status)
                self.assertEqual(client.configs, [])
            with self.subTest(stage="logits", status=status):
                client = FakeInferenceClient(
                    bf16_rows(6), logits_reply=ESMProteinError(status, "denied")
                )
                with self.assertRaises(self.AuthError) as raised:
                    esmc_6b_api._request_embedding("ACDE", client, np.float32)
                self.assertEqual(raised.exception.error_code, status)

        failures = {
            "encode": (
                {"encode_replies": [ESMProteinError(500, "server busy")]},
                "API encode error (code 500): server busy",
            ),
            "logits": (
                {"logits_reply": ESMProteinError(500, "server busy")},
                "API logits error (code 500): server busy",
            ),
        }
        for stage, (replies, message) in failures.items():
            with self.subTest(stage=stage, status=500):
                client = FakeInferenceClient(bf16_rows(6), **replies)
                with self.assertRaises(RuntimeError) as raised:
                    esmc_6b_api._request_embedding("ACDE", client, np.float32)
                # BiohubAuthenticationError is a RuntimeError subclass too.
                self.assertIs(type(raised.exception), RuntimeError)
                self.assertEqual(str(raised.exception), message)

    def run_with_retries(self, client):
        """Run get_embedding with recorded sleeps; return its result or error."""
        sleep = mock.Mock()
        with mock.patch.object(
            esmc_6b_api, "time", SimpleNamespace(sleep=sleep)
        ), redirect_stdout(io.StringIO()) as stdout:
            try:
                result = esmc_6b_api.get_embedding("ACDE", client, None, np.float32)
            except RuntimeError as error:
                result = error
        return result, sleep, stdout.getvalue()

    def test_transient_failures_retry_after_growing_delays(self):
        client = FakeInferenceClient(
            bf16_rows(6),
            encode_replies=[ESMProteinError(500, "busy"), ESMProteinError(503, "down")],
        )
        result, sleep, output = self.run_with_retries(client)
        np.testing.assert_array_equal(result, ACDE_ROWS)
        self.assertEqual(sleep.call_args_list, [mock.call(5), mock.call(10)])
        self.assertEqual(len(client.encoded), 3)
        self.assertIn(
            "Temporary ESMC 6B API error (attempt 1/3): API encode error "
            "(code 500): busy. Retrying in 5 seconds...",
            output,
        )
        self.assertIn(
            "Temporary ESMC 6B API error (attempt 2/3): API encode error "
            "(code 503): down. Retrying in 10 seconds...",
            output,
        )

    def test_three_failures_give_up_with_the_last_error_as_cause(self):
        client = FakeInferenceClient(
            bf16_rows(6),
            encode_replies=[ESMProteinError(500, name) for name in ("one", "two", "three")],
        )
        error, sleep, _output = self.run_with_retries(client)
        self.assertIs(type(error), RuntimeError)
        self.assertEqual(str(error), "ESMC 6B API request failed after 3 attempts.")
        self.assertEqual(str(error.__cause__), "API encode error (code 500): three")
        self.assertEqual(sleep.call_args_list, [mock.call(5), mock.call(10)])
        self.assertEqual(len(client.encoded), 3)


if __name__ == "__main__":
    unittest.main()
