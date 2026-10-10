"""Mocked tests for local/remote ESMFold worker behavior."""

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

from resources.esmfold import esmfold_worker
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
        real_replace = os.replace
        attempts = []

        def replace(source, target):
            attempts.append(target)
            if len(attempts) < 3:
                raise PermissionError("in use")
            real_replace(source, target)

        with tempfile.TemporaryDirectory() as structures_dir:
            with mock.patch.object(esmfold_worker.os, "replace", side_effect=replace), \
                    mock.patch.object(esmfold_worker.time, "sleep"):
                esmfold_worker._write_prediction(FakeOutput("NEW\n"), "node", structures_dir)
            with open(os.path.join(structures_dir, "node.pdb"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "NEW\n")
        self.assertEqual(len(attempts), 3)


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

    def test_notification_without_a_filename_is_not_broadcast(self):
        payload = self.posted_payload("node/1", "node_1.pdb")
        del payload["pdb_filename"]
        self.assertEqual(self.dispatch(payload), [])


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

    def read_status(self):
        with open(self.status_path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_status_write_retries_while_the_file_is_held_open(self):
        real_replace = os.replace
        attempts = []

        def replace(source, target):
            attempts.append(target)
            if len(attempts) <= 3:
                raise PermissionError(13, "Access is denied", target)
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
            raise PermissionError(13, "Access is denied", target)

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
                raise PermissionError(13, "Access is denied", target)
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


if __name__ == "__main__":
    unittest.main()
