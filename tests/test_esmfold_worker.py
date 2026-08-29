"""Mocked tests for local/remote ESMFold worker behavior."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resources.esmfold import esmfold_worker


class FakeProtein:
    def __init__(self, sequence=None):
        self.sequence = sequence


class FakeProteinError:
    def __init__(self, error_code, error_msg="rejected"):
        self.error_code = error_code
        self.error_msg = error_msg

    def __str__(self):
        return self.error_msg


class FakeGenerationConfig:
    def __init__(self, track, num_steps):
        self.track = track
        self.num_steps = num_steps


class FakeOutput:
    def __init__(self, pdb="ATOM\n"):
        self.plddt = None
        self.pdb = pdb

    def to_pdb_string(self):
        return self.pdb


class FakeClient:
    def __init__(self, results):
        self.results = list(results)
        self.closed = False
        self.token = "old-token"
        self.headers = {"Authorization": "Bearer old-token"}

    def generate(self, protein, config):
        return self.results.pop(0)

    def close(self):
        self.closed = True


def fake_esm_modules():
    esm_module = types.ModuleType("esm")
    sdk_module = types.ModuleType("esm.sdk")
    api_module = types.ModuleType("esm.sdk.api")
    api_module.ESMProtein = FakeProtein
    api_module.ESMProteinError = FakeProteinError
    api_module.GenerationConfig = FakeGenerationConfig
    esm_module.sdk = sdk_module
    sdk_module.api = api_module
    return {
        "esm": esm_module,
        "esm.sdk": sdk_module,
        "esm.sdk.api": api_module,
    }


class ESMFoldWorkerTests(unittest.TestCase):
    def test_argument_parser_preserves_local_compatibility_and_large_mode(self):
        local = esmfold_worker.parse_arguments(["input.json", "structures", "cuda"])
        large = esmfold_worker.parse_arguments(
            ["input.json", "structures", "--mode", "large"]
        )

        self.assertEqual(local.mode, "local")
        self.assertEqual(local.device, "cuda")
        self.assertEqual(local.action_url, esmfold_worker.DEFAULT_ACTION_URL)
        self.assertEqual(large.mode, "large")
        self.assertIsNone(large.device)

        custom = esmfold_worker.parse_arguments(
            [
                "input.json",
                "structures",
                "--action-url",
                "http://127.0.0.1:49123/api/action",
            ]
        )
        self.assertEqual(
            custom.action_url,
            "http://127.0.0.1:49123/api/action",
        )

    def test_notify_server_posts_to_the_supplied_instance_url(self):
        action_url = "http://127.0.0.1:49123/api/action"
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen") as urlopen:
            esmfold_worker.notify_server("node_1", "node_1.pdb", action_url)

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, action_url)
        self.assertEqual(
            request.get_header("Content-type"),
            "application/json",
        )

    def test_large_client_uses_hidden_worker_terminal_prompt(self):
        settings = {
            "ESM_API_TOKEN": "terminal-token",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        remote_client = mock.Mock()
        sdk_module = types.ModuleType("esm.sdk")
        sdk_module.client = mock.Mock(return_value=remote_client)

        def load_from_terminal(*, prompt_callback):
            self.assertEqual(prompt_callback(), "terminal-token")
            return settings

        with (
            mock.patch.dict(sys.modules, {"esm.sdk": sdk_module}),
            mock.patch.object(
                esmfold_worker,
                "_terminal_token_prompt",
                return_value="terminal-token",
            ) as terminal_prompt,
            mock.patch.object(
                esmfold_worker.Biohub_API,
                "load_api_settings",
                side_effect=load_from_terminal,
            ) as load_settings,
        ):
            client, loaded_settings = esmfold_worker._load_large_client()

        load_settings.assert_called_once()
        terminal_prompt.assert_called_once_with(False)
        sdk_module.client.assert_called_once_with(
            model=settings["ESM3_MODEL"],
            url=settings["ESM_API_URL"],
            token=settings["ESM_API_TOKEN"],
        )
        self.assertIs(client, remote_client)
        self.assertEqual(loaded_settings, settings)

    def test_large_prediction_writes_model_specific_pdb_and_closes_client(self):
        client = FakeClient([FakeOutput()])
        settings = {
            "ESM_API_TOKEN": "hidden",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        notifier = mock.Mock()

        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.dict(sys.modules, fake_esm_modules()),
                mock.patch.object(
                    esmfold_worker,
                    "_load_large_client",
                    return_value=(client, settings),
                ),
                mock.patch.object(esmfold_worker, "_load_local_model") as local_loader,
            ):
                completed, failed = esmfold_worker.run_predictions(
                    [["node/name", "ACDE"]],
                    structures_dir,
                    mode="large",
                    notifier=notifier,
                )

            expected_name = "node_name_esm3-large-2024-03.pdb"
            self.assertEqual((completed, failed), (1, False))
            self.assertTrue(os.path.exists(os.path.join(structures_dir, expected_name)))
            notifier.assert_called_once_with("node/name", expected_name)
            local_loader.assert_not_called()
            self.assertTrue(client.closed)

    def test_authentication_failure_refreshes_once_and_retries_current_request(self):
        client = FakeClient([FakeProteinError(401), FakeOutput()])
        settings = {
            "ESM_API_TOKEN": "old-token",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        refreshed = dict(settings, ESM_API_TOKEN="new-token")
        auth_state = {"settings": settings, "refresh_attempted": False}

        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(
                esmfold_worker.Biohub_API,
                "refresh_api_token",
                return_value=refreshed,
            ) as refresh,
        ):
            result = esmfold_worker._generate_remote_structure(
                client,
                FakeProtein("ACDE"),
                FakeGenerationConfig("structure", 8),
                auth_state,
            )

        self.assertIsInstance(result, FakeOutput)
        refresh.assert_called_once()
        self.assertTrue(auth_state["refresh_attempted"])
        self.assertEqual(client.headers["Authorization"], "Bearer new-token")

    def test_second_authentication_failure_stops_without_another_prompt(self):
        client = FakeClient([FakeProteinError(401), FakeProteinError(403)])
        settings = {
            "ESM_API_TOKEN": "old-token",
            "ESM_API_URL": "https://biohub.ai",
            "ESM3_MODEL": "esm3-large-2024-03",
        }
        auth_state = {"settings": settings, "refresh_attempted": False}

        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(
                esmfold_worker.Biohub_API,
                "refresh_api_token",
                return_value=dict(settings, ESM_API_TOKEN="new-token"),
            ) as refresh,
        ):
            with self.assertRaises(esmfold_worker.Biohub_API.BiohubAuthenticationError):
                esmfold_worker._generate_remote_structure(
                    client,
                    FakeProtein("ACDE"),
                    FakeGenerationConfig("structure", 8),
                    auth_state,
                )
        refresh.assert_called_once()

    def test_main_binds_the_action_url_without_changing_notifier_shape(self):
        action_url = "http://127.0.0.1:49123/api/action"
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(
                    esmfold_worker,
                    "_load_nodes",
                    return_value=[["node_1", "ACDE"]],
                ),
                mock.patch.object(
                    esmfold_worker,
                    "run_predictions",
                    return_value=(1, False),
                ) as run_predictions,
                mock.patch.object(esmfold_worker, "notify_server") as notify_server,
            ):
                result = esmfold_worker.main(
                    [
                        "input.json",
                        structures_dir,
                        "--action-url",
                        action_url,
                    ]
                )

                notifier = run_predictions.call_args.kwargs["notifier"]
                notifier("node_1", "node_1.pdb")

        self.assertEqual(result, 0)
        notify_server.assert_called_once_with("node_1", "node_1.pdb", action_url)


if __name__ == "__main__":
    unittest.main()
