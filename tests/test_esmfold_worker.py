"""Mocked tests for local/remote ESMFold worker behavior."""

import fnmatch
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy
import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import Viewer_Worker_Tracking
from resources.esmfold import esmfold_worker
from tests.held_file_fixtures import held_file_error, transient_hold
from utilities import Job_Cancellation
from web_ui import esmfold_backend
from web_ui.Plugin_Manager import WebPluginRegistry


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
    def __init__(self, pdb="ATOM\n", plddt=None):
        self.plddt = plddt
        self.pdb = pdb
        self.plddt_at_write = None

    def to_pdb_string(self):
        self.plddt_at_write = self.plddt
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


# Runs the worker script as __main__ in a process where `import torch` raises
# ImportError, as it does for a missing torch, and where esm imports.
# argv: worker script, then the worker's arguments.
TORCH_UNIMPORTABLE_RUNNER = """
import runpy, sys, types
worker, *arguments = sys.argv[1:]
sys.modules["torch"] = None
for name in ("esm", "esm.sdk", "esm.sdk.api"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["esm.sdk.api"].ESMProtein = object
sys.modules["esm.sdk.api"].ESMProteinError = object
sys.modules["esm.sdk.api"].GenerationConfig = object
sys.argv = [worker, *arguments]
runpy.run_path(worker, run_name="__main__")
"""


class ESMFoldWorkerTests(unittest.TestCase):
    def test_argument_parser_preserves_local_compatibility_and_large_mode(self):
        local = esmfold_worker.parse_arguments(["input.json", "structures", "cuda"])
        large = esmfold_worker.parse_arguments(
            ["input.json", "structures", "--mode", "large"]
        )

        self.assertEqual(local.mode, "local")
        self.assertEqual(local.device, "cuda")
        # Standalone runs notify nobody and keep their input unless asked.
        self.assertIsNone(local.action_url)
        self.assertFalse(local.delete_input)
        self.assertFalse(local.skip_existing)
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
        self.assertEqual(urlopen.call_args.kwargs["timeout"], esmfold_worker.NOTIFY_TIMEOUT_SECONDS)

    def test_notify_server_without_url_sends_nothing(self):
        # Formerly an omitted URL posted to 127.0.0.1:8000, which may belong
        # to an unrelated Viewer.
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen") as urlopen:
            esmfold_worker.notify_server("node_1", "node_1.pdb")
            esmfold_worker.notify_server("node_1", "node_1.pdb", None)
        urlopen.assert_not_called()

    def test_input_json_is_deleted_only_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "records.json")
            for delete in (False, True):
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write('[["node_1", "ACDE"]]')
                self.assertEqual(esmfold_worker._load_nodes(path, delete=delete), [["node_1", "ACDE"]])
                self.assertEqual(os.path.exists(path), not delete)
            self.assertEqual(
                esmfold_worker.parse_arguments(["records.json", "--delete-input"]).delete_input, True
            )

    def test_a_torch_that_cannot_be_imported_leaves_no_input_file_behind(self):
        # The worker used to import torch before reading its arguments, so a
        # torch that is missing or crashes on import left the private file the
        # Viewer handed over (--delete-input) in the temporary folder.
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "records.json")
            status_path = os.path.join(directory, "status.json")
            Path(input_path).write_text('[["node_1", "ACDE"]]', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "-c", TORCH_UNIMPORTABLE_RUNNER, esmfold_worker.__file__,
                    input_path, os.path.join(directory, "structures"), "cpu",
                    "--delete-input", "--noninteractive", "--portal-status", status_path,
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
            )
            self.assertFalse(os.path.exists(input_path), result.stdout + result.stderr)
            # The failure is still reported: on the terminal, in the exit code and to the Viewer.
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("Error initializing ESM3 prediction:", result.stdout)
            status = json.loads(Path(status_path).read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertIn("torch", status["message"])

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
            mock.patch.object(esmfold_worker, "can_ask", return_value=True),
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
            mock.patch.object(esmfold_worker, "can_ask", return_value=True),
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
                mock.patch.object(esmfold_worker, "notify_job"),
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

    def test_notify_job_posts_the_event_the_page_shows_and_warns_only_once(self):
        action_url = "http://127.0.0.1:49123/api/action"
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen") as urlopen:
            esmfold_worker.notify_job("progress", action_url, index=2, total=3, name="node_b")
            esmfold_worker.notify_job("started", None, total=3)
        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, action_url)
        self.assertEqual(json.loads(request.data.decode("utf-8")),
                         {"action": "esmfold_job", "event": "progress", "index": 2, "total": 3, "name": "node_b"})
        # A Viewer that cannot be reached costs one warning, not one for each event.
        esmfold_worker._job_notice_failed = False
        output = io.StringIO()
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen", side_effect=OSError("refused")), redirect_stdout(output):
            for index in (1, 2, 3):
                esmfold_worker.notify_job("progress", action_url, index=index, total=3)
        esmfold_worker._job_notice_failed = False
        self.assertEqual(output.getvalue().count("Warning: Could not report progress"), 1)

    def test_run_predictions_reports_each_structure_it_starts_and_each_that_fails(self):
        events = []
        model = mock.Mock()

        def generate(protein, config):
            if protein.sequence == "DE":
                raise RuntimeError("out of memory")
            return FakeOutput(pdb=f"NEW {protein.sequence}\n")

        model.generate.side_effect = generate
        with tempfile.TemporaryDirectory() as structures_dir:
            with open(os.path.join(structures_dir, "node_a.pdb"), "w", encoding="utf-8") as handle:
                handle.write("OLD\n")
            with (
                mock.patch.dict(sys.modules, fake_esm_modules()),
                mock.patch.object(esmfold_worker, "_load_local_model", return_value=model),
                redirect_stdout(io.StringIO()),
            ):
                result = esmfold_worker.run_predictions(
                    [["node_a", "AC"], ["node_b", "DE"], ["node_c", "FG"]], structures_dir, mode="local",
                    target_device="cpu", notifier=mock.Mock(), skip_existing=True,
                    job_event=lambda event, **fields: events.append((event, fields)),
                )
        self.assertEqual(result, (2, True))
        # The kept structure is not started; the others are, in order, and the failure says why.
        self.assertEqual(events, [
            ("progress", {"index": 2, "total": 3, "name": "node_b"}),
            ("structure_failed", {"name": "node_b", "reason": "out of memory"}),
            ("progress", {"index": 3, "total": 3, "name": "node_c"}),
        ])

    def _main_events(self, run_result=None, load_error=None):
        """The events main() reports to the Viewer for a run whose predictions end as run_result says."""
        action_url = "http://127.0.0.1:49123/api/action"
        events = []
        loaded = mock.patch.object(esmfold_worker, "_load_nodes", return_value=[["node_1", "ACDE"], ["node_2", "FG"]])
        if load_error:
            loaded = mock.patch.object(esmfold_worker, "_load_nodes", side_effect=load_error)
        with tempfile.TemporaryDirectory() as structures_dir, loaded, \
                mock.patch.object(esmfold_worker, "run_predictions", return_value=run_result) as run_predictions, \
                mock.patch.object(esmfold_worker, "notify_job", side_effect=lambda event, url, **fields: events.append((event, url, fields))), \
                redirect_stdout(io.StringIO()):
            result = esmfold_worker.main(["input.json", structures_dir, "--action-url", action_url, "--noninteractive"])
        return result, events, run_predictions

    def test_main_reports_a_run_to_the_viewer_from_its_start_to_its_end(self):
        url = "http://127.0.0.1:49123/api/action"
        result, events, run_predictions = self._main_events((2, False))
        self.assertEqual(result, 0)
        self.assertEqual(events, [
            ("started", url, {"total": 2, "names": ["node_1", "node_2"]}),
            ("finished", url, {"succeeded": 2, "total": 2, "failed": False}),
        ])
        # run_predictions was told how to report each structure.
        self.assertIn("job_event", run_predictions.call_args.kwargs)
        result, events, _ = self._main_events((1, True))
        self.assertEqual(result, 1)
        self.assertEqual(events[-1], ("finished", url, {"succeeded": 1, "total": 2, "failed": True, "log": None}))

    def test_main_reports_a_run_that_could_not_start(self):
        url = "http://127.0.0.1:49123/api/action"
        result, events, _ = self._main_events(load_error=ValueError("no records"))
        self.assertEqual(result, 1)
        self.assertEqual(events, [("failed", url, {"reason": "no records", "log": None})])

    def _run_local(self, records, structures_dir, **options):
        model = mock.Mock()
        model.generate.side_effect = lambda protein, config: FakeOutput(pdb=f"NEW {protein.sequence}\n")
        notifier = mock.Mock()
        output = io.StringIO()
        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(esmfold_worker, "_load_local_model", return_value=model),
            redirect_stdout(output),
        ):
            result = esmfold_worker.run_predictions(
                records, structures_dir, mode="local", target_device="cpu", notifier=notifier, **options
            )
        return result, model, notifier, output.getvalue()

    def test_skip_existing_keeps_structures_and_counts_them_completed(self):
        with tempfile.TemporaryDirectory() as structures_dir:
            existing = os.path.join(structures_dir, "node_a.pdb")
            with open(existing, "w", encoding="utf-8") as handle:
                handle.write("OLD\n")
            result, model, notifier, output = self._run_local(
                [["node_a", "AC"], ["node_b", "DE"]], structures_dir, skip_existing=True
            )
            self.assertEqual(result, (2, False))
            with open(existing, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "OLD\n")
            with open(os.path.join(structures_dir, "node_b.pdb"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "NEW DE\n")
            self.assertEqual(model.generate.call_count, 1)
            notifier.assert_called_once_with("node_b", "node_b.pdb")
            self.assertIn("Keeping existing structure for node_a", output)

    def test_replacements_and_shared_filenames_are_reported(self):
        with tempfile.TemporaryDirectory() as structures_dir:
            with open(os.path.join(structures_dir, "a_b.pdb"), "w", encoding="utf-8") as handle:
                handle.write("OLD\n")
            result, _, _, output = self._run_local([["a/b", "AC"], ["a_b", "DE"]], structures_dir)
            self.assertEqual(result, (2, False))
            self.assertIn(f"replacing existing structure {os.path.join(structures_dir, 'a_b.pdb')}", output)
            self.assertIn("'a_b' and 'a/b' both map to a_b.pdb", output)
            with open(os.path.join(structures_dir, "a_b.pdb"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "NEW DE\n")

    def test_failed_write_keeps_the_previous_structure_and_no_partial_file(self):
        with tempfile.TemporaryDirectory() as structures_dir:
            target = os.path.join(structures_dir, "node.pdb")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("OLD\n")
            with mock.patch.object(esmfold_worker.os, "replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    esmfold_worker._write_prediction(FakeOutput("NEW\n"), "node", structures_dir)
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "OLD\n")
            self.assertEqual(os.listdir(structures_dir), ["node.pdb"])

    def test_write_retries_while_windows_holds_the_target_open(self):
        for winerror in (32, 33, 5):
            with self.subTest(winerror=winerror), tempfile.TemporaryDirectory() as structures_dir:
                with transient_hold(2, winerror) as (attempts, pauses):
                    esmfold_worker._write_prediction(FakeOutput("NEW\n"), "node", structures_dir)
                with open(os.path.join(structures_dir, "node.pdb"), encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "NEW\n")
                self.assertEqual(len(attempts), 3)
                self.assertEqual(len(pauses), 2)
                self.assertEqual(os.listdir(structures_dir), ["node.pdb"])

    def test_a_hold_that_never_ends_fails_the_structure_saying_so_and_keeps_the_previous_one(self):
        with tempfile.TemporaryDirectory() as structures_dir:
            target = os.path.join(structures_dir, "node.pdb")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("OLD\n")
            with transient_hold(10 ** 6) as (attempts, pauses), self.assertRaises(PermissionError) as raised:
                esmfold_worker._write_prediction(FakeOutput("NEW\n"), "node", structures_dir)
            # The worker reports str(error) to the Viewer, as the failed structure's reason.
            self.assertIn("another program kept the file in use", str(raised.exception))
            self.assertIn("being used by another process", str(raised.exception))
            with open(target, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "OLD\n")
            self.assertEqual(os.listdir(structures_dir), ["node.pdb"])

    def test_a_partial_file_that_stays_held_is_left_to_be_deleted_when_the_worker_ends(self):
        leftovers = []
        with tempfile.TemporaryDirectory() as structures_dir:
            real_remove = os.remove

            def remove(name):
                if name.endswith(".partial"):
                    raise held_file_error(32, name)
                real_remove(name)

            with transient_hold(10 ** 6), \
                    mock.patch.object(Job_Cancellation.os, "remove", side_effect=remove), \
                    mock.patch.object(Job_Cancellation, "leave_partial", side_effect=leftovers.append), \
                    self.assertRaises(PermissionError):
                esmfold_worker._write_prediction(FakeOutput("NEW\n"), "node", structures_dir)
            self.assertEqual([os.path.basename(path) for path in leftovers], [f"node.pdb.{os.getpid()}.partial"])

    def test_the_worker_process_loads_the_shared_rename_without_qt(self):
        # The worker is a process of its own: the helper must import there with nothing of the Viewer's.
        result = subprocess.run(
            [sys.executable, "-I", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from resources.esmfold import esmfold_worker as w; "
             "print(w.replace_file.__module__, 'PySide6' in sys.modules)", str(SRC_DIR)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        self.assertEqual(result.stdout.split(), ["utilities.Job_Cancellation", "False"], result.stdout + result.stderr)


class ViewerNotificationTests(unittest.TestCase):
    """What notify_server posts is what the Viewer's ESMFold backend acts on."""

    def posted_payload(self, node_id, pdb_filename):
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen") as urlopen:
            esmfold_worker.notify_server(node_id, pdb_filename, "http://127.0.0.1:49123/api/action")
        return json.loads(urlopen.call_args.args[0].data.decode("utf-8"))

    def dispatch(self, payload):
        """Deliver ``payload`` as MainViewer.handle_web_action does; return the broadcasts."""
        viewer = types.SimpleNamespace(broadcast_event=mock.Mock())
        registry = WebPluginRegistry(viewer)
        with tempfile.TemporaryDirectory() as structures_dir, mock.patch.object(
            esmfold_backend, "get_structures_directory", return_value=structures_dir
        ):
            esmfold_backend.register_backend(registry, viewer)
        with redirect_stdout(io.StringIO()):
            registry.actions[payload["action"]](payload)
        return [call.args[0] for call in viewer.broadcast_event.call_args_list]

    def test_folded_structure_reaches_the_page_as_a_structure_url(self):
        self.assertEqual(
            self.dispatch(self.posted_payload("node/1", "node_1.pdb")),
            [{"type": "esmfold_pdb", "node_id": "node/1", "pdb_url": "/structures/node_1.pdb"}],
        )

    def test_job_events_reach_the_page_as_the_worker_posts_them(self):
        with mock.patch.object(esmfold_worker.urllib.request, "urlopen") as urlopen:
            esmfold_worker.notify_job("progress", "http://127.0.0.1:49123/api/action", index=1, total=2, name="node/1")
        payload = json.loads(urlopen.call_args.args[0].data.decode("utf-8"))
        self.assertEqual(
            self.dispatch(payload),
            [{"type": "esmfold_job", "event": "progress", "index": 1, "total": 2, "name": "node/1"}],
        )

    def test_notification_without_a_filename_is_not_broadcast(self):
        payload = self.posted_payload("node/1", "node_1.pdb")
        del payload["pdb_filename"]
        self.assertEqual(self.dispatch(payload), [])

    def test_download_and_authentication_events_pass_their_numbers_on(self):
        self.assertEqual(
            self.dispatch({"action": "esmfold_job", "event": "downloading", "percent": 42, "run": "r1"}),
            [{"type": "esmfold_job", "event": "downloading", "percent": 42, "run": "r1"}])
        self.assertEqual(
            self.dispatch({"action": "esmfold_job", "event": "authentication", "status": 401, "run": "r1"}),
            [{"type": "esmfold_job", "event": "authentication", "status": 401, "run": "r1"}])

    def test_a_percent_that_is_not_a_number_is_dropped(self):
        for value in (True, -1, "50", 10 ** 12, None):
            with self.subTest(value=value):
                [message] = self.dispatch({"action": "esmfold_job", "event": "downloading", "percent": value})
                self.assertNotIn("percent", message)


class LocalModelDataRootTests(unittest.TestCase):
    """Only ESM3 is redirected; every other model keeps esm's own repositories."""

    def test_esm3_is_local_first_and_other_models_use_the_package_mapping(self):
        package_calls = []

        def package_data_root(model_type):
            package_calls.append(model_type)
            return Path("package") / model_type

        modules = {name: types.ModuleType(name) for name in (
            "esm", "esm.pretrained", "esm.utils", "esm.utils.constants",
            "esm.utils.constants.esm3", "esm.models", "esm.models.esm3",
        )}
        for name, module in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, module)
        modules["esm.utils.constants.esm3"].data_root = package_data_root
        loaded = object()
        esm3_class = mock.Mock()
        esm3_class.from_pretrained.return_value.to.return_value = loaded
        modules["esm.models.esm3"].ESM3 = esm3_class
        # A stub rather than mock.patch("huggingface_hub.snapshot_download"):
        # resolving that target imports the real package (httpx, tqdm, ...)
        # inside the patched sys.modules, which then drops those modules and
        # leaves orphan copies behind. A stub also can never download weights.
        download = mock.Mock(return_value="cached-esm3")
        modules["huggingface_hub"] = types.ModuleType("huggingface_hub")
        modules["huggingface_hub"].snapshot_download = download

        with mock.patch.dict(sys.modules, modules):
            self.assertIs(esmfold_worker._load_local_model("cpu"), loaded)
            data_root = modules["esm.pretrained"].data_root
            self.assertEqual(data_root("esm3"), Path("cached-esm3"))
            download.assert_called_once_with(
                repo_id="biohub/esm3-sm-open-v1", local_files_only=True
            )
            self.assertEqual(data_root("esmc-300"), Path("package") / "esmc-300")
            self.assertEqual(package_calls, ["esmc-300"])
            self.assertEqual(download.call_count, 1)
        esm3_class.from_pretrained.assert_called_once_with("esm3_sm_open_v1")


class WritePredictionScaleTests(unittest.TestCase):
    """pLDDT must reach to_pdb_string() unscaled.

    esm >= 3.4 multiplies ProteinChain.confidence by PLDDT_B_FACTOR_SCALE
    (100.0) itself when building the atom array. Pre-scaling here as well
    pushes B-factors to ~10000, which overflows the six-column PDB
    temperature-factor field and makes biotite raise BadStructureError.
    """

    def test_write_prediction_does_not_rescale_plddt(self):
        confidence = numpy.array([0.31, 0.58, 0.95], dtype=numpy.float32)
        output = FakeOutput(plddt=confidence.copy())

        with tempfile.TemporaryDirectory() as structures_dir:
            esmfold_worker._write_prediction(output, "node/name", structures_dir)

        numpy.testing.assert_allclose(output.plddt_at_write, confidence)
        numpy.testing.assert_allclose(output.plddt, confidence)


class PortalStatusTests(unittest.TestCase):
    """Writing the Viewer Command Portal status file must not change what the
    worker reports about its structures.

    On Windows, os.replace raises PermissionError (WinError 5) while another
    process has the status file open: the Viewer's WorkerTracker reads it every
    250 ms, and external monitors may read it too.
    """

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.status_path = os.path.join(self.directory, "status.json")
        # main() rebinds these globals; stopping the patch restores them, so
        # other tests still see a standalone worker.
        portal = mock.patch.multiple(
            esmfold_worker,
            _portal_path=self.status_path,
            _portal_noninteractive=True,
            _portal_state={"artifacts": []},
        )
        portal.start()
        self.addCleanup(portal.stop)
        # The holds these tests simulate are Windows errors, whatever platform the tests run on.
        windows = mock.patch.object(Job_Cancellation, "_on_windows", return_value=True)
        windows.start()
        self.addCleanup(windows.stop)

    def read_status(self):
        with open(self.status_path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_status_write_retries_while_the_file_is_held_open(self):
        real_replace = os.replace
        attempts = []

        def replace(source, target):
            attempts.append(target)
            if len(attempts) <= 3:
                raise held_file_error(5, target)
            real_replace(source, target)

        esmfold_worker._portal_state["artifacts"].append("node_1.pdb")
        with mock.patch.object(esmfold_worker.os, "replace", side_effect=replace), \
                mock.patch.object(esmfold_worker.time, "sleep") as sleep:
            esmfold_worker.portal_event("running", "Folded node_1", succeeded=1, total=2)

        self.assertEqual(attempts, [self.status_path] * 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(
            self.read_status(),
            {
                "artifacts": ["node_1.pdb"],
                "pid": os.getpid(),
                "created": psutil.Process().create_time(),
                "status": "running",
                "message": "Folded node_1",
                "result": {"succeeded": 1, "total": 2},
            },
        )
        self.assertEqual(os.listdir(self.directory), ["status.json"])

    def test_status_write_waits_for_a_reader_to_close_the_file(self):
        esmfold_worker.portal_event("running", "Worker started")
        # Hold the file open as WorkerTracker.poll() does while reading it; the
        # retry's pause is where the reader lets go.
        reader = open(self.status_path, "rb")
        self.addCleanup(reader.close)
        with mock.patch.object(
            esmfold_worker.time, "sleep", side_effect=lambda seconds: reader.close()
        ) as sleep:
            esmfold_worker.portal_event("running", "Folded node_1", succeeded=1, total=1)

        self.assertEqual(self.read_status()["message"], "Folded node_1")
        if os.name == "nt":
            # Windows refused the replacement while the reader had the file open.
            self.assertGreaterEqual(sleep.call_count, 1)

    def test_unwritable_status_warns_and_final_statuses_wait_longer(self):
        attempts = []

        def replace(source, target):
            attempts.append(target)
            raise held_file_error(5, target)

        output = io.StringIO()
        with mock.patch.object(esmfold_worker.os, "replace", side_effect=replace), \
                mock.patch.object(esmfold_worker.time, "sleep"), redirect_stdout(output):
            esmfold_worker.portal_event("running", "Folded node_1", succeeded=1, total=2)
            progress_attempts = len(attempts)
            esmfold_worker.portal_event("succeeded", "Structure prediction completed", succeeded=2, total=2)
        final_attempts = len(attempts) - progress_attempts

        self.assertEqual(progress_attempts, esmfold_worker.PORTAL_PROGRESS_ATTEMPTS)
        self.assertEqual(final_attempts, esmfold_worker.PORTAL_FINAL_ATTEMPTS)
        self.assertGreater(final_attempts, progress_attempts)
        self.assertEqual(output.getvalue().count("Warning: Could not update the worker status file"), 2)
        # Neither a status file nor a stale .partial is left behind.
        self.assertEqual(os.listdir(self.directory), [])

    def test_unwritable_progress_update_does_not_fail_a_saved_structure(self):
        # A monitor holds the status file open for the whole run and lets go
        # before the final status. In the field report, an ESM3-large run saved
        # every structure but reported one as a WinError 5 failure.
        real_replace = os.replace
        real_run_predictions = esmfold_worker.run_predictions
        held = []

        def replace(source, target):
            if held and target == self.status_path:
                raise held_file_error(5, target)
            real_replace(source, target)

        def run_while_held(*args, **kwargs):
            held.append(True)
            try:
                return real_run_predictions(*args, **kwargs)
            finally:
                held.clear()

        records = os.path.join(self.directory, "records.json")
        with open(records, "w", encoding="utf-8") as handle:
            json.dump([["node_a", "AC"], ["node_b", "DE"]], handle)
        structures_dir = os.path.join(self.directory, "structures")
        model = mock.Mock()
        model.generate.side_effect = lambda protein, config: FakeOutput()
        output = io.StringIO()
        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(esmfold_worker, "_load_local_model", return_value=model),
            mock.patch.object(esmfold_worker, "run_predictions", side_effect=run_while_held),
            mock.patch.object(esmfold_worker.os, "replace", side_effect=replace),
            mock.patch.object(esmfold_worker.time, "sleep"),
            # main() wraps sys.stdout and sys.stderr; these restore them.
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            result = esmfold_worker.main(
                [records, structures_dir, "cpu", "--portal-status", self.status_path, "--noninteractive"]
            )

        self.assertEqual(result, 0)
        status = self.read_status()
        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["result"], {"succeeded": 2, "total": 2})
        self.assertEqual(
            [os.path.basename(path) for path in status["artifacts"]],
            ["node_a.pdb", "node_b.pdb"],
        )
        self.assertEqual(sorted(os.listdir(structures_dir)), ["node_a.pdb", "node_b.pdb"])
        self.assertNotIn("Error folding", output.getvalue())
        # The warnings reach the log the Viewer forwards as the worker's output.
        with open(self.status_path + ".stdout", encoding="utf-8") as handle:
            self.assertEqual(
                handle.read().count("Warning: Could not update the worker status file (running)"),
                2,
            )


def pdb_text(sequence, chain="A", first=1):
    """The backbone atoms of a structure of ``sequence`` as ESM writes them: one chain, residues numbered on."""
    lines = []
    serial = 1
    for offset, _ in enumerate(sequence):
        for atom in ("N", "CA", "C"):
            lines.append(
                f"ATOM  {serial:5d}  {atom:<3} ALA {chain}{first + offset:4d}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 50.00           C  "
            )
            serial += 1
    return "\n".join(lines) + "\nEND\n"


def conservation(values, label="Agreement"):
    """A property as a provider (utilities.Residue_Data) returns it, with ``values`` for its residues."""
    return {
        "label": label, "unit": "%", "range": [0, 100], "description": "Test data.",
        "values": list(values),
        "source": {"kind": "msa", "file": "family.fasta", "alignment_id": "sha256:" + "0" * 64,
                   "columns": list(range(len(values))), "sequences": 7},
    }


def extra_for(header, *value_lists):
    """The third item of a node's record: its header and one property per list of values."""
    names = ("information_content", "same_residue_pct")
    return {"header": header, "properties": {
        name: conservation(values, label=name) for name, values in zip(names, value_lists)
    }}


class ResidueDataFileTests(unittest.TestCase):
    """The sidecar a structure gets: its name, its numbering, its content, and how it is written."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def write_structure(self, name="node_a.pdb", sequence="ACDE", **numbering):
        Path(self.directory, name).write_text(pdb_text(sequence, **numbering), encoding="utf-8")
        return name

    def read(self, name):
        return json.loads(Path(self.directory, name).read_text(encoding="utf-8"))

    def test_the_file_is_named_for_the_pdb_actually_written(self):
        for pdb, expected in (
            ("XP_123.1.pdb", "XP_123.1.residues.json"),
            ("XP_123.1_esm3-large-2024-03.pdb", "XP_123.1_esm3-large-2024-03.residues.json"),
            ("name.PDB", "name.residues.json"),
            ("no_extension", "no_extension.residues.json"),
        ):
            with self.subTest(pdb=pdb):
                self.assertEqual(esmfold_worker.residue_data_filename(pdb), expected)

    def test_residue_numbering_is_read_from_the_structure(self):
        numbering = esmfold_worker.pdb_residue_numbering
        self.assertEqual(numbering(pdb_text("ACDE"), 4), ("A", 1))
        self.assertEqual(numbering(pdb_text("ACDE", chain="B", first=5), 4), ("B", 5))
        self.assertEqual(numbering(pdb_text("ACDE", first=-2), 4), ("A", -2))
        # No atoms to read: the numbering ESM writes.
        self.assertEqual(numbering("ATOM\n", 4), ("A", 1))
        self.assertEqual(numbering("", 4), ("A", 1))

    def test_residue_numbering_that_cannot_index_the_values_is_refused(self):
        numbering = esmfold_worker.pdb_residue_numbering
        gap = pdb_text("AC").replace("   2    ", "   3    ")
        two_chains = pdb_text("AC") + pdb_text("AC", chain="B")
        insertion = pdb_text("AC").replace("   2    ", "   2A   ")
        for label, text, length, reason in (
            ("a gap in the numbers", gap, 2, "not the 2 of the sequence"),
            ("two chains", two_chains, 2, "2 chains"),
            ("an insertion code", insertion, 2, "insertion codes"),
            ("another length", pdb_text("ACD"), 4, "not the 4 of the sequence"),
        ):
            with self.subTest(label), self.assertRaisesRegex(ValueError, reason):
                numbering(text, length)

    def test_properties_without_one_value_per_residue_are_dropped(self):
        kept, problems = esmfold_worker.checked_properties(
            {"good": conservation([1, 2, 3]), "short": conservation([1, 2]), "none": {"values": None}, "odd": 7}, 3
        )
        self.assertEqual(list(kept), ["good"])
        self.assertEqual(len(problems), 3)
        self.assertIn("short has 2 values for 3 residues", problems)

    def test_the_file_holds_the_documented_fields(self):
        name = self.write_structure("node_a.pdb", "ACDE", chain="B", first=5)
        extra = extra_for("node_a some description", [10.0, None, 30.5, 40.25], [90.0, 80.0, None, 60.0])
        output = io.StringIO()
        with redirect_stdout(output):
            written = esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra)

        self.assertEqual(written, "node_a.residues.json")
        document = self.read(written)
        self.assertEqual(
            list(document),
            ["format", "version", "structure", "header", "sequence", "residue_numbering", "properties"],
        )
        self.assertEqual(document["format"], "emapssn-residue-data")
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["structure"], "node_a.pdb")
        self.assertEqual(document["header"], "node_a some description")
        self.assertEqual(document["sequence"], "ACDE")
        self.assertEqual(document["residue_numbering"], {"chain": "B", "first": 5})
        self.assertEqual(list(document["properties"]), ["information_content", "same_residue_pct"])
        self.assertEqual(document["properties"]["information_content"]["values"], [10.0, None, 30.5, 40.25])
        self.assertEqual(document["properties"]["same_residue_pct"], extra["properties"]["same_residue_pct"])
        self.assertEqual(sorted(os.listdir(self.directory)), ["node_a.pdb", "node_a.residues.json"])

    def test_the_header_defaults_to_the_record_name(self):
        name = self.write_structure()
        with redirect_stdout(io.StringIO()):
            written = esmfold_worker._write_residue_data(
                self.directory, name, "node_a", "ACDE", {"properties": {"p": conservation([1, 2, 3, 4])}}
            )
        self.assertEqual(self.read(written)["header"], "node_a")

    def test_non_ascii_headers_are_kept_as_text(self):
        name = self.write_structure()
        extra = {"header": "α-amylase 淀粉酶", "properties": {"p": conservation([1, 2, 3, 4])}}
        with redirect_stdout(io.StringIO()):
            written = esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra)
        self.assertEqual(self.read(written)["header"], "α-amylase 淀粉酶")

    def test_nothing_is_written_without_a_property(self):
        name = self.write_structure()
        for label, extra in (
            ("no extra", None), ("not an object", [1, 2]), ("no properties", {"header": "h"}),
            ("empty properties", {"properties": {}}), ("properties not an object", {"properties": [1]}),
        ):
            with self.subTest(label):
                self.assertIsNone(esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra))
        self.assertEqual(os.listdir(self.directory), ["node_a.pdb"])

    def test_a_property_of_the_wrong_length_is_dropped_with_a_warning(self):
        name = self.write_structure()
        extra = {"properties": {"good": conservation([1, 2, 3, 4]), "bad": conservation([1, 2])}}
        output = io.StringIO()
        with redirect_stdout(output):
            written = esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra)
        self.assertEqual(list(self.read(written)["properties"]), ["good"])
        self.assertIn("Dropped a residue-data property for node_a: bad has 2 values for 4 residues", output.getvalue())

        only_bad = {"properties": {"bad": conservation([1, 2])}}
        with redirect_stdout(io.StringIO()):
            self.assertIsNone(esmfold_worker._write_residue_data(self.directory, name, "node_b", "ACDE", only_bad))
        self.assertNotIn("node_b.residues.json", os.listdir(self.directory))

    def test_a_structure_that_is_not_this_sequence_gets_no_file(self):
        extra = extra_for("h", [1, 2, 3, 4])
        for label, pdb in (("another length", "node_a.pdb"), ("missing", "gone.pdb")):
            with self.subTest(label):
                self.write_structure("node_a.pdb", "ACDEFG")
                output = io.StringIO()
                with redirect_stdout(output):
                    written = esmfold_worker._write_residue_data(self.directory, pdb, "node_a", "ACDE", extra)
                self.assertIsNone(written)
                self.assertIn("Warning: Could not write the residue data for node_a", output.getvalue())
        self.assertEqual(sorted(os.listdir(self.directory)), ["node_a.pdb"])

    def test_values_that_are_not_json_leave_no_file_and_no_partial(self):
        name = self.write_structure()
        for label, bad in (("nan", float("nan")), ("infinity", float("inf")), ("an object", object())):
            with self.subTest(label):
                extra = {"properties": {"p": conservation([1, 2, 3, bad])}}
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertIsNone(esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra))
                self.assertIn("Could not write the residue data", output.getvalue())
        self.assertEqual(os.listdir(self.directory), ["node_a.pdb"])

    def test_the_file_is_published_whole_or_not_at_all(self):
        name = self.write_structure()
        extra = extra_for("h", [1, 2, 3, 4])
        previous = Path(self.directory, "node_a.residues.json")
        previous.write_text('{"old": true}', encoding="utf-8")
        seen = []

        def replace(source, target):
            seen.append((os.path.basename(source), os.path.exists(target)))
            raise OSError("disk full")

        with mock.patch.object(esmfold_worker.os, "replace", side_effect=replace), \
                mock.patch.object(esmfold_worker.time, "sleep"), redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra))

        # Written under a .partial name first, and never over the old file in place.
        self.assertEqual(seen, [(f"node_a.residues.json.{os.getpid()}.partial", True)])
        self.assertEqual(previous.read_text(encoding="utf-8"), '{"old": true}')
        self.assertEqual(sorted(os.listdir(self.directory)), ["node_a.pdb", "node_a.residues.json"])
        self.assertIn("disk full", output.getvalue())

    def test_the_partial_name_is_the_one_esmfold_stop_deletes(self):
        # StartedWorker._remove_partial_structures removes <output dir>/*.<pid>.partial of a worker it ended.
        name = self.write_structure()
        names = []

        def replace(source, target):
            names.append(os.path.basename(source))
            raise OSError("interrupted")

        with mock.patch.object(esmfold_worker.os, "replace", side_effect=replace), \
                mock.patch.object(esmfold_worker.time, "sleep"), redirect_stdout(io.StringIO()):
            esmfold_worker._write_residue_data(self.directory, name, "node_a", "ACDE", extra_for("h", [1, 2, 3, 4]))
        [partial] = names
        pid = os.getpid()
        self.assertTrue(fnmatch.fnmatch(partial, f"*.{pid}.partial"), partial)
        Path(self.directory, partial).write_text("{", encoding="utf-8")
        worker = Viewer_Worker_Tracking.StartedWorker(mock.Mock(), output_dir=self.directory)
        worker.pids.add(pid)
        worker._remove_partial_structures()
        self.assertEqual(os.listdir(self.directory), ["node_a.pdb"])


class ResidueDataPredictionTests(unittest.TestCase):
    """run_predictions with records that carry per-residue data."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.structures = directory.name

    def run_local(self, records, **options):
        model = mock.Mock()
        model.generate.side_effect = lambda protein, config: FakeOutput(pdb=pdb_text(protein.sequence))
        events = []
        output = io.StringIO()

        def notifier(node_id, pdb_filename):
            sidecar = esmfold_worker.residue_data_filename(pdb_filename)
            # What the page finds when it is told: the structure, and its data if there is any.
            events.append(("notified", node_id, os.path.exists(os.path.join(self.structures, sidecar))))

        def saved(path):
            events.append(("saved", os.path.basename(path)))

        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(esmfold_worker, "_load_local_model", return_value=model),
            redirect_stdout(output),
        ):
            result = esmfold_worker.run_predictions(
                records, self.structures, mode="local", target_device="cpu",
                notifier=notifier, residue_data_saved=saved, **options,
            )
        return result, model, events, output.getvalue()

    def files(self):
        return sorted(os.listdir(self.structures))

    def read(self, name):
        return json.loads(Path(self.structures, name).read_text(encoding="utf-8"))

    def test_the_old_pair_form_writes_a_structure_and_nothing_else(self):
        result, model, events, output = self.run_local([["node_a", "ACDE"], ["node_b", "DEFG"]])
        self.assertEqual(result, (2, False))
        self.assertEqual(self.files(), ["node_a.pdb", "node_b.pdb"])
        self.assertEqual(events, [("notified", "node_a", False), ("notified", "node_b", False)])
        self.assertNotIn("residue data", output)

    def test_a_record_with_data_gets_its_file_before_the_viewer_is_told(self):
        extra = extra_for("node_a first protein", [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        result, _, events, output = self.run_local([["node_a", "ACDE", extra], ["node_b", "DEFG"]])

        self.assertEqual(result, (2, False))
        self.assertEqual(self.files(), ["node_a.pdb", "node_a.residues.json", "node_b.pdb"])
        # The file exists when the notifier runs, and is reported as saved right after it.
        self.assertEqual(events, [
            ("notified", "node_a", True), ("saved", "node_a.residues.json"), ("notified", "node_b", False),
        ])
        document = self.read("node_a.residues.json")
        self.assertEqual(document["structure"], "node_a.pdb")
        self.assertEqual(document["header"], "node_a first protein")
        self.assertEqual(document["sequence"], "ACDE")
        self.assertEqual(document["residue_numbering"], {"chain": "A", "first": 1})
        self.assertEqual(document["properties"]["same_residue_pct"]["values"], [5.0, 6.0, 7.0, 8.0])
        self.assertIn("Saved residue data to:", output)

    def test_the_file_of_a_large_run_carries_the_models_suffix_of_its_pdb(self):
        client = FakeClient([FakeOutput(pdb=pdb_text("ACDE"))])
        settings = {"ESM_API_TOKEN": "hidden", "ESM_API_URL": "https://biohub.ai", "ESM3_MODEL": "esm3-large-2024-03"}
        extra = extra_for("node/name x", [1, 2, 3, 4])
        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(esmfold_worker, "_load_large_client", return_value=(client, settings)),
            redirect_stdout(io.StringIO()),
        ):
            esmfold_worker.run_predictions(
                [["node/name", "ACDE", extra]], self.structures, mode="large", notifier=mock.Mock()
            )
        self.assertEqual(self.files(), ["node_name_esm3-large-2024-03.pdb", "node_name_esm3-large-2024-03.residues.json"])
        self.assertEqual(self.read("node_name_esm3-large-2024-03.residues.json")["structure"], "node_name_esm3-large-2024-03.pdb")

    def test_a_record_that_is_neither_a_pair_nor_a_triple_is_invalid(self):
        result, _, _, output = self.run_local([["node_a", "ACDE", {}, "extra"], ["node_b"], "node_c", ["node_d", "DEFG"]])
        self.assertEqual(result, (1, True))
        self.assertEqual(output.count("Error: Invalid node record"), 3)
        self.assertEqual(self.files(), ["node_d.pdb"])

    def test_data_that_is_not_an_object_is_ignored_and_the_structure_still_folded(self):
        result, _, events, output = self.run_local([["node_a", "ACDE", [1, 2, 3, 4]]])
        self.assertEqual(result, (1, False))
        self.assertEqual(self.files(), ["node_a.pdb"])
        self.assertIn("Warning: Ignoring the residue data of node_a", output)

    def test_a_failure_to_write_the_data_does_not_fail_the_structure(self):
        extra = extra_for("h", [1, 2, 3])  # Three values for four residues.
        result, _, events, output = self.run_local([["node_a", "ACDE", extra]])
        self.assertEqual(result, (1, False))
        self.assertEqual(self.files(), ["node_a.pdb"])
        self.assertEqual(events, [("notified", "node_a", False)])

    def test_skip_existing_refreshes_the_data_of_a_structure_it_keeps(self):
        Path(self.structures, "node_a.pdb").write_text(pdb_text("ACDE"), encoding="utf-8")
        Path(self.structures, "node_a.residues.json").write_text('{"old": true}', encoding="utf-8")
        extra = extra_for("node_a now", [9.0, 9.0, 9.0, 9.0], [1.0, 1.0, 1.0, 1.0])
        before = Path(self.structures, "node_a.pdb").read_bytes()

        result, model, events, output = self.run_local([["node_a", "ACDE", extra], ["node_b", "DEFG"]], skip_existing=True)

        self.assertEqual(result, (2, False))
        self.assertEqual(model.generate.call_count, 1)  # Only node_b was folded.
        self.assertEqual(Path(self.structures, "node_a.pdb").read_bytes(), before)
        document = self.read("node_a.residues.json")
        self.assertEqual(document["header"], "node_a now")
        self.assertEqual(document["properties"]["information_content"]["values"], [9.0, 9.0, 9.0, 9.0])
        # The kept structure is not announced again, but its refreshed data is reported saved.
        self.assertEqual(events, [("saved", "node_a.residues.json"), ("notified", "node_b", False)])
        self.assertIn("Keeping existing structure for node_a", output)
        self.assertIn("Refreshed the residue data of the kept structure from the alignment now loaded: node_a.residues.json", output)
        self.assertIn("Kept 1 existing structure(s) without predicting them again.", output)
        self.assertIn("Refreshed the residue data of 1 of them.", output)

    def test_skip_existing_without_data_leaves_the_old_file_as_it_is(self):
        Path(self.structures, "node_a.pdb").write_text(pdb_text("ACDE"), encoding="utf-8")
        Path(self.structures, "node_a.residues.json").write_text('{"old": true}', encoding="utf-8")
        result, _, events, output = self.run_local([["node_a", "ACDE"]], skip_existing=True)
        self.assertEqual(result, (1, False))
        self.assertEqual(self.read("node_a.residues.json"), {"old": True})
        self.assertEqual(events, [])
        self.assertNotIn("Refreshed", output)

    def test_skip_existing_refreshes_nothing_for_a_structure_of_another_sequence(self):
        Path(self.structures, "node_a.pdb").write_text(pdb_text("ACDEFG"), encoding="utf-8")
        result, _, events, output = self.run_local([["node_a", "ACDE", extra_for("h", [1, 2, 3, 4])]], skip_existing=True)
        self.assertEqual(result, (1, False))
        self.assertEqual(self.files(), ["node_a.pdb"])
        self.assertEqual(events, [])
        self.assertIn("Warning: Could not write the residue data for node_a", output)

    def test_a_refold_without_data_leaves_an_earlier_file_beside_the_new_structure(self):
        # Data from an earlier alignment is not deleted: a run with no alignment writes none, and removes none.
        extra = extra_for("h", [1, 2, 3, 4])
        self.run_local([["node_a", "ACDE", extra]])
        self.run_local([["node_a", "ACDE"]])
        self.assertEqual(self.files(), ["node_a.pdb", "node_a.residues.json"])


class ResidueDataMainTests(unittest.TestCase):
    """main(): the input file in both forms, and what the job reports."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name
        self.status_path = os.path.join(self.directory, "status.json")
        self.structures = os.path.join(self.directory, "structures")
        portal = mock.patch.multiple(
            esmfold_worker, _portal_path=self.status_path, _portal_noninteractive=True, _portal_state={"artifacts": []},
        )
        portal.start()
        self.addCleanup(portal.stop)

    def run_main(self, records, *arguments):
        path = os.path.join(self.directory, "records.json")
        Path(path).write_text(json.dumps(records), encoding="utf-8")
        model = mock.Mock()
        model.generate.side_effect = lambda protein, config: FakeOutput(pdb=pdb_text(protein.sequence))
        with (
            mock.patch.dict(sys.modules, fake_esm_modules()),
            mock.patch.object(esmfold_worker, "_load_local_model", return_value=model),
            redirect_stdout(io.StringIO()) as output,
            redirect_stderr(io.StringIO()),
        ):
            result = esmfold_worker.main(
                [path, self.structures, "cpu", "--portal-status", self.status_path, "--noninteractive", *arguments]
            )
        with open(self.status_path, encoding="utf-8") as handle:
            return result, json.load(handle), model, output.getvalue()

    def test_the_input_file_holds_pairs_and_triples_alike(self):
        path = os.path.join(self.directory, "records.json")
        records = [["a", "AC"], ["b", "DE", {"header": "b x", "properties": {"p": {"values": [1, 2]}}}]]
        Path(path).write_text(json.dumps(records), encoding="utf-8")
        self.assertEqual(esmfold_worker._load_nodes(path), records)
        self.assertEqual(esmfold_worker._load_nodes(path, delete=True), records)
        self.assertFalse(os.path.exists(path))

    def test_the_artifacts_list_each_structure_then_its_data_and_the_counts_stay_per_structure(self):
        records = [
            ["node_a", "ACDE", extra_for("node_a x", [1, 2, 3, 4], [4, 3, 2, 1])],
            ["node_b", "DEFG"],
            ["node_c", "GHIK", extra_for("node_c x", [5, 6, 7, 8])],
        ]
        result, status, model, output = self.run_main(records)

        self.assertEqual(result, 0)
        self.assertEqual(status["status"], "succeeded")
        names = [os.path.basename(path) for path in status["artifacts"]]
        self.assertEqual(
            names, ["node_a.pdb", "node_a.residues.json", "node_b.pdb", "node_c.pdb", "node_c.residues.json"]
        )
        self.assertTrue(all(os.path.isabs(path) for path in status["artifacts"]))
        # Three structures were folded; the two data files are not counted among them.
        self.assertEqual(status["result"], {"succeeded": 3, "total": 3})
        self.assertEqual(status["message"], "Structure prediction completed; per-residue data saved for 2 structure(s)")
        self.assertEqual(sorted(os.listdir(self.structures)), sorted(names))

    def test_a_run_without_data_reports_as_before(self):
        result, status, _, _ = self.run_main([["node_a", "AC"], ["node_b", "DE"]])
        self.assertEqual(result, 0)
        self.assertEqual(status["message"], "Structure prediction completed")
        self.assertEqual([os.path.basename(path) for path in status["artifacts"]], ["node_a.pdb", "node_b.pdb"])

    def test_skip_existing_reports_the_refreshed_file_as_an_artifact(self):
        os.makedirs(self.structures)
        Path(self.structures, "node_a.pdb").write_text(pdb_text("ACDE"), encoding="utf-8")
        result, status, model, output = self.run_main(
            [["node_a", "ACDE", extra_for("node_a x", [1, 2, 3, 4])]], "--skip-existing"
        )
        self.assertEqual(result, 0)
        model.generate.assert_not_called()
        self.assertEqual([os.path.basename(path) for path in status["artifacts"]], ["node_a.residues.json"])
        self.assertEqual(status["result"], {"succeeded": 1, "total": 1})
        self.assertIn("Refreshed the residue data of 1 of them.", output)


class NeverAskTests(unittest.TestCase):
    """The worker has no window and no input: it never waits for someone to answer, and Biohub's rejection ends the run cleanly."""

    SETTINGS = {"ESM_API_TOKEN": "saved-secret-token", "ESM_API_URL": "https://biohub.ai", "ESM3_MODEL": "esm3-large-2024-03"}

    def setUp(self):
        portal = mock.patch.multiple(
            esmfold_worker, _portal_path=None, _portal_noninteractive=False, _portal_state={"artifacts": []})
        portal.start()
        self.addCleanup(portal.stop)

    def terminal(self, is_terminal):
        stdin = mock.Mock()
        stdin.isatty.return_value = is_terminal
        return mock.patch.object(sys, "stdin", stdin)

    def test_it_can_ask_only_where_a_terminal_can_answer(self):
        with self.terminal(True):
            self.assertTrue(esmfold_worker.can_ask())
            with mock.patch.object(esmfold_worker, "_portal_noninteractive", True):
                self.assertFalse(esmfold_worker.can_ask(), "--noninteractive says there is no one")
        with self.terminal(False):
            self.assertFalse(esmfold_worker.can_ask(), "no terminal, whatever the flag")
        with mock.patch.object(sys, "stdin", None):
            self.assertFalse(esmfold_worker.can_ask())

    def test_it_never_waits_for_input_without_a_terminal(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("the worker waited for input")

        for flag in (True, False):
            with self.subTest(noninteractive=flag), self.terminal(False), \
                    mock.patch.object(esmfold_worker, "_portal_noninteractive", flag), \
                    mock.patch("builtins.input", forbidden), \
                    mock.patch.object(esmfold_worker.getpass, "getpass", forbidden):
                esmfold_worker.portal_pause("Press Enter")
                with self.assertRaises(RuntimeError):
                    esmfold_worker._terminal_token_prompt(False)
                with self.assertRaises(RuntimeError):
                    esmfold_worker._terminal_token_prompt(True)

    def test_run_by_hand_in_a_terminal_it_still_pauses_and_prompts(self):
        with self.terminal(True), mock.patch("builtins.input") as pause, \
                mock.patch.object(esmfold_worker.getpass, "getpass", return_value="typed") as prompt:
            esmfold_worker.portal_pause("Press Enter")
            self.assertEqual(esmfold_worker._terminal_token_prompt(False), "typed")
        pause.assert_called_once_with("Press Enter")
        prompt.assert_called_once()

    def test_a_rejected_token_ends_the_run_without_refreshing_when_no_one_can_be_asked(self):
        client = FakeClient([FakeProteinError(401)])
        auth_state = {"settings": dict(self.SETTINGS), "refresh_attempted": False}
        with self.terminal(False), mock.patch.dict(sys.modules, fake_esm_modules()), \
                mock.patch.object(esmfold_worker.Biohub_API, "refresh_api_token") as refresh:
            with self.assertRaises(esmfold_worker.Biohub_API.BiohubAuthenticationError) as caught:
                esmfold_worker._generate_remote_structure(
                    client, FakeProtein("ACDE"), FakeGenerationConfig("structure", 8), auth_state)
        refresh.assert_not_called()
        self.assertEqual(caught.exception.error_code, 401)
        self.assertNotIn("saved-secret-token", str(caught.exception))

    def test_a_rejection_in_the_middle_keeps_what_was_saved_and_is_not_counted_as_a_failed_structure(self):
        client = FakeClient([FakeOutput(), FakeProteinError(403)])
        events = []
        with tempfile.TemporaryDirectory() as structures_dir, self.terminal(False), \
                mock.patch.dict(sys.modules, fake_esm_modules()), \
                mock.patch.object(esmfold_worker, "_load_large_client", return_value=(client, dict(self.SETTINGS))), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(esmfold_worker.Biohub_API.BiohubAuthenticationError):
                esmfold_worker.run_predictions(
                    [["node_a", "ACDE"], ["node_b", "ACDF"], ["node_c", "ACDG"]], structures_dir, mode="large",
                    notifier=mock.Mock(), job_event=lambda event, **fields: events.append((event, fields)))
            saved = sorted(os.listdir(structures_dir))
        self.assertEqual(saved, ["node_a_esm3-large-2024-03.pdb"])
        self.assertEqual([event for event, _ in events], ["progress", "progress"], "no structure_failed for the rejection")
        self.assertTrue(client.closed)

    def run_main_rejected(self, status_path):
        events = []
        output = io.StringIO()
        error = esmfold_worker.Biohub_API.BiohubAuthenticationError(401, "Biohub rejected the saved API token.")
        with tempfile.TemporaryDirectory() as structures_dir, \
                mock.patch.object(esmfold_worker, "_load_nodes", return_value=[["node_1", "ACDE"], ["node_2", "FG"]]), \
                mock.patch.object(esmfold_worker, "run_predictions", side_effect=error), \
                mock.patch.object(esmfold_worker, "notify_job",
                                  side_effect=lambda event, url, **fields: events.append((event, fields))), \
                redirect_stdout(output):
            result = esmfold_worker.main([
                "input.json", structures_dir, "--action-url", "http://127.0.0.1:49123/api/action",
                "--noninteractive", "--run-id", "run1", "--portal-status", status_path])
        return result, events, output.getvalue()

    def test_main_exits_with_its_own_code_and_says_authentication_not_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            status_path = os.path.join(directory, "status.json")
            result, events, output = self.run_main_rejected(status_path)
            with open(status_path, encoding="utf-8") as handle:
                status = json.load(handle)
        self.assertEqual(result, esmfold_worker.EXIT_AUTHENTICATION)
        self.assertEqual(esmfold_worker.EXIT_AUTHENTICATION, 3)
        self.assertEqual([event for event, _ in events], ["started", "authentication"])
        self.assertEqual(events[-1][1], {"status": 401, "run": "run1"})
        # A status that waits for the user, never failed: the Viewer reads failed as the end of the job.
        self.assertEqual(status["status"], "awaiting_user_input")
        self.assertEqual(status["result"]["authentication"], 401)
        self.assertNotIn("saved-secret-token", output + json.dumps(status))


class DownloadProgressTests(unittest.TestCase):
    """The weights of the local model are downloaded once; the Viewer is told how far it has got."""

    @classmethod
    def setUpClass(cls):
        # Imported here, not for the first time inside a patched sys.modules, which would drop the package again
        # afterwards and leave orphan copies of its modules behind.
        import tqdm.auto  # noqa: F401

    def bars(self, reports):
        """The class, with the clock the worker reads in self.now."""
        self.now = 100.0
        patcher = mock.patch("time.monotonic", lambda: self.now)
        patcher.start()
        self.addCleanup(patcher.stop)
        return esmfold_worker.download_progress_class(reports.append)

    def test_the_percent_of_the_bytes_is_reported_not_going_back_and_not_too_often(self):
        reports = []
        progress = self.bars(reports)
        # Named as huggingface_hub names its bars; the count of files is no byte count.
        files = progress(total=3, unit="it", name="files", file=io.StringIO(), mininterval=0, miniters=1)
        files.update(1)
        self.assertEqual(reports, [], "a bar that does not count bytes says nothing")
        bar = progress(total=1000, unit="B", name="huggingface_hub.snapshot_download.transfer", file=io.StringIO(),
                       mininterval=0, miniters=1)
        self.assertEqual(reports, [0], "the first look")
        self.now = 100.1
        bar.update(100)
        self.assertEqual(reports, [0], "10% so soon after the last report is not reported")
        self.now = 100.7
        bar.update(100)
        self.assertEqual(reports, [0, 20])
        self.now = 100.8
        bar.update(100)
        self.assertEqual(reports, [0, 20], "30% 0.1 s later is not reported")
        # More files start: the total grows and the percent of the bytes falls, which is not reported.
        self.now = 101.5
        bar.total = 4000
        bar.update(100)
        self.assertEqual(reports, [0, 20], "the percent never goes back")
        self.now = 101.6
        bar.update(3600)
        self.assertEqual(reports, [0, 20, 100], "the end is reported")
        bar.close()
        files.close()

    def test_the_first_download_hands_the_class_to_huggingface_and_reports_its_events(self):
        events = []
        reached = {}

        def snapshot_download(**options):
            if options.get("local_files_only"):
                raise FileNotFoundError("not cached")
            reached["options"] = options
            bar = options["tqdm_class"](total=100, unit="B", name="x", file=io.StringIO())
            bar.update(100)
            bar.close()
            return "downloaded-esm3"

        modules = self.model_modules(snapshot_download)
        with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            esmfold_worker._load_local_model("cpu", lambda event, **fields: events.append((event, fields)))
            root = modules["esm.pretrained"].data_root("esm3")
        self.assertEqual(root, Path("downloaded-esm3"))
        self.assertEqual(reached["options"]["repo_id"], "biohub/esm3-sm-open-v1")
        self.assertEqual(events[-1], ("downloading", {"percent": 100}))

    def test_a_cached_model_reports_nothing_and_without_an_event_function_nothing_changes(self):
        downloads = []
        modules = self.model_modules(lambda **options: downloads.append(options) or "cached-esm3")
        with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            esmfold_worker._load_local_model("cpu", lambda event, **fields: self.fail(event))
            modules["esm.pretrained"].data_root("esm3")
        self.assertEqual(downloads, [{"repo_id": "biohub/esm3-sm-open-v1", "local_files_only": True}])

    def test_hugging_face_refusing_access_is_explained_and_other_errors_pass_through(self):
        class Refused(Exception):
            response = types.SimpleNamespace(status_code=401)

        def refusing(**options):
            if options.get("local_files_only"):
                raise FileNotFoundError("not cached")
            raise Refused("401 Client Error")

        modules = self.model_modules(refusing)
        with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            esmfold_worker._load_local_model("cpu")
            with self.assertRaises(RuntimeError) as caught:
                modules["esm.pretrained"].data_root("esm3")
        text = str(caught.exception)
        self.assertIn("Hugging Face refused access", text)
        self.assertIn("hf auth login", text)
        self.assertIn("esmfold large", text)

        def broken(**options):
            if options.get("local_files_only"):
                raise FileNotFoundError("not cached")
            raise ConnectionError("no network")

        modules = self.model_modules(broken)
        with mock.patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            esmfold_worker._load_local_model("cpu")
            with self.assertRaises(ConnectionError):
                modules["esm.pretrained"].data_root("esm3")

    @staticmethod
    def model_modules(snapshot_download):
        """The stubs LocalModelDataRootTests uses, so that nothing is downloaded or imported for real."""
        modules = {name: types.ModuleType(name) for name in (
            "esm", "esm.pretrained", "esm.utils", "esm.utils.constants",
            "esm.utils.constants.esm3", "esm.models", "esm.models.esm3",
        )}
        for name, module in modules.items():
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, module)
        modules["esm.utils.constants.esm3"].data_root = lambda model_type: Path("package") / model_type
        modules["esm.models.esm3"].ESM3 = mock.Mock()
        modules["huggingface_hub"] = types.ModuleType("huggingface_hub")
        modules["huggingface_hub"].snapshot_download = snapshot_download
        return modules


if __name__ == "__main__":
    unittest.main()
