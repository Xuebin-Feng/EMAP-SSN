"""The esmfold command: routing, what it reports, the folds it queues, and esmfold stop.

A fold is a job of the background scheduler's folding lane. The command fixes which nodes to fold, and
their sequences and per-residue data, when it is given, saves them to a private input file, reserves the
files the worker will write, and queues the job; the job launches the worker with no window when its turn
comes. These tests drive the command against a scheduler that only records what it is given (what the
command decides), then the job against fake processes (what a fold does), and the whole against the real
scheduler, the real portal and real worker processes (test_esmfold_lane has the lane's own behaviour).
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import psutil
from PySide6 import QtCore

import Viewer_Worker_Tracking
from Background_Job_Scheduler import BackgroundJobScheduler
from Viewer_Command_Portal import CURRENT, ExecutionContext, bind
from commands import esmfold as esmfold_command
from tests.command_fixtures import end_scheduler
from tests.sparse_alignment import sparse_alignment
from tests.viewer_fixtures import NO_WINDOW_STATES, WINDOW_STATES, PortalFixture, apply_window_state
from utilities import Hardware_Acceleration as Hardware_Utils
from utilities.Localization import display_text
from web_ui import Browser_Page
from web_ui.Plugin_Manager import bundled_pages


# What every run says about per-residue data while no alignment is loaded, as the stand-in Viewers here are.
NO_ALIGNMENT = "No per-residue alignment data is saved: no alignment is loaded."
ADDRESS = "http://127.0.0.1:49123"
URL = f"{ADDRESS}/api/action"
# Where the command module under test says the project is, as it computes it.
COMMAND_SRC = os.path.dirname(os.path.dirname(os.path.abspath(esmfold_command.__file__)))
COMMAND_PROJECT = os.path.dirname(COMMAND_SRC)


def opening_pages(viewer):
    """viewer, which opens the pages the bundled plugins declare through the Viewer's own open_page."""
    from EMAPSSN_Viewer import MainViewer

    viewer._open_web_ui = lambda *args, **kwargs: Browser_Page.open_browser_page(viewer, *args, **kwargs)
    viewer.web_plugin_registry = bundled_pages()
    viewer.open_page = lambda client, **kwargs: MainViewer.open_page(viewer, client, **kwargs)
    return viewer


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class CapturingScheduler:
    """What the command asks of the background scheduler: who holds a path, and the jobs queued.

    It runs nothing: ``jobs`` keeps the arguments of each enqueue, the first of which is the job's
    FoldRun (``payload``). ``held`` names the jobs that hold a path, as the reservations of the real one do.
    """

    def __init__(self):
        self.jobs = []
        self.held = {}
        self.reserved = []
        self.cancelled = []
        self.calls = 0

    @staticmethod
    def key(path):
        return os.path.normcase(os.path.abspath(path))

    def reservation_holder(self, path):
        return self.held.get(self.key(path))

    def enqueue(self, **job):
        number = len(self.jobs) + 1
        self.jobs.append(job)
        paths = [job["output_path"], *job.get("extra_paths", ())]
        self.reserved.append(paths)
        for path in paths:
            self.held[self.key(path)] = number
        return number

    def snapshot(self):
        return [
            SimpleNamespace(
                job=SimpleNamespace(job_id=index + 1, command_name=job["command_name"], description=job["description"]),
                lane=job["lane"], running=index == 0, position=index + 1)
            for index, job in enumerate(self.jobs)
        ]

    def cancel_many(self, job_ids):
        self.calls += 1
        self.cancelled.extend(job_ids)
        return list(job_ids)

    @property
    def run(self):
        """The FoldRun of the last job queued."""
        return self.jobs[-1]["payload"]


def make_viewer(count=0, scheduler=True, **extra):
    headers = [f"node_{index} description" for index in range(count)]
    viewer = SimpleNamespace(
        selected_indices=list(range(count)),
        selected_node_idx=None,
        full_headers=headers,
        sequences_map={header: "ACDE" for header in headers},
        console_text=SimpleNamespace(text=""),
        get_web_url=mock.Mock(return_value=URL),
        background_job_scheduler=CapturingScheduler() if scheduler is True else scheduler,
    )
    for name, value in extra.items():
        setattr(viewer, name, value)
    return viewer


class CommandCase(unittest.TestCase):
    """Runs the command with the page, the folder and the engine's reports stubbed."""

    def setUp(self):
        # The job launches the worker, never the command: a stand-in that is called fails the test.
        patcher = mock.patch.object(
            esmfold_command, "launch_without_window",
            side_effect=AssertionError("The command must queue the fold, not start the worker."),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(Viewer_Worker_Tracking, "track_worker")
        self.track_worker = patcher.start()
        self.addCleanup(patcher.stop)
        # Every ready credential lookup stays on this machine's own files: none is read here.
        patcher = mock.patch.object(esmfold_command, "_credentials_state", return_value=("saved", None))
        self.credentials = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(esmfold_command, "_model_suffix", side_effect=lambda large: "esm3-large-2024-03" if large else None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(esmfold_command.RUNS.clear)

    def run_reporting(self, viewer, arguments, *, opened=True, output=None, structures=None):
        """Run the command; return (succeeded, failed) and leave the job it queued on viewer's scheduler.

        ``output`` collects what the command prints. The private input file the run saved is deleted
        when the test ends, as the scheduler's cleanup would.
        """
        engine = esmfold_command.Command_Engine
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            folder = structures or enter(tempfile.TemporaryDirectory())
            self.structures_dir = folder
            enter(mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory", return_value=folder))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui", return_value=opened))
            succeeded = enter(mock.patch.object(engine, "command_succeeded"))
            failed = enter(mock.patch.object(engine, "command_failed"))
            enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
            enter(redirect_stdout(output if output is not None else io.StringIO()))
            try:
                esmfold_command.run(viewer, arguments)
            finally:
                scheduler = viewer.background_job_scheduler
                for job in getattr(scheduler, "jobs", ()):
                    job["payload"].discard_input()
        return succeeded, failed

    def read_records(self, run):
        with open(run.input_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def queued(self, viewer, arguments, **options):
        """Run the command and return its FoldRun, with the input file kept readable until the test ends."""
        outputs = options.pop("output", None)
        succeeded, failed = self.run_reporting(viewer, arguments, output=outputs, **options)
        failed.assert_not_called()
        return viewer.background_job_scheduler.run, succeeded


class RoutingTests(CommandCase):
    def test_bare_command_without_selection_registers_and_opens_ui(self):
        viewer = make_viewer()
        with (
            mock.patch.object(esmfold_command.esmfold_backend, "register") as register,
            mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui") as open_esmfold_ui,
        ):
            esmfold_command.run(viewer, [])

        register.assert_called_once_with(viewer)
        open_esmfold_ui.assert_called_once_with(viewer)
        self.assertEqual(viewer.background_job_scheduler.jobs, [])

    def test_keywords_without_selection_report_error(self):
        for arguments in (["multi"], ["large"], ["large", "multi"]):
            with self.subTest(arguments=arguments):
                viewer = make_viewer()
                with (
                    mock.patch.object(esmfold_command.esmfold_backend, "register") as register,
                    mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui") as open_esmfold_ui,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    esmfold_command.run(viewer, arguments)

                register.assert_not_called()
                open_esmfold_ui.assert_not_called()
                self.assertEqual(viewer.background_job_scheduler.jobs, [])
                self.assertIn("Error: No nodes selected.", output.getvalue())

    def test_local_single_queues_a_fold_whose_worker_arguments_are_the_legacy_ones(self):
        viewer = make_viewer(1)
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")) as get_device:
            run, _ = self.queued(viewer, [])
        get_device.assert_called_once_with()
        command = run.command()
        # The worker owns and deletes the private input file only because the Viewer passes --delete-input.
        self.assertEqual(command[2:], [
            run.input_path, os.path.abspath(self.structures_dir), "cuda", "--delete-input", "--action-url", URL,
            "--noninteractive", "--run-id", run.run_id,
        ])
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(os.path.abspath(command[1]), os.path.join(COMMAND_SRC, "resources", "esmfold", "esmfold_worker.py"))
        self.assertNotIn("--mode", command)
        self.assertNotIn("--skip-existing", command)

    def test_the_records_are_fixed_when_the_command_is_given_and_the_input_file_holds_them(self):
        viewer = make_viewer(1)
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")):
            succeeded, failed = self.run_reporting(viewer, [])
        run = viewer.background_job_scheduler.run
        self.assertEqual(run.records, [["node_0", "ACDE"]])
        # Saved when the command ran: the file existed then (it is deleted by the helper afterwards).
        self.assertTrue(run.input_path.endswith(".json"))
        self.assertFalse(os.path.exists(run.input_path))

    def test_large_single_bypasses_hardware_and_selects_remote_mode(self):
        viewer = make_viewer(1)
        with mock.patch.object(Hardware_Utils, "get_optimal_device") as get_device:
            run, _ = self.queued(viewer, ["large"])
        get_device.assert_not_called()
        command = run.command()
        self.assertEqual(command[-8:], ["--delete-input", "--mode", "large", "--action-url", URL,
                                        "--noninteractive", "--run-id", run.run_id])
        # The Viewer neither loads nor forwards credentials to the worker.
        self.assertNotIn("ESM_API_TOKEN", " ".join(command))
        self.assertFalse(hasattr(esmfold_command, "Biohub_API"))

    def test_large_multi_accepts_both_keyword_orders(self):
        for arguments in (["large", "multi"], ["multi", "large"]):
            with self.subTest(arguments=arguments):
                viewer = make_viewer(2)
                run, _ = self.queued(viewer, arguments)
                self.assertEqual([record[0] for record in run.records], ["node_0", "node_1"])
                self.assertEqual(run.mode, "large")
                self.assertEqual(viewer.background_job_scheduler.jobs[0]["description"], "esmfold large multi -> 2 structures")

    def test_multiple_nodes_without_multi_are_rejected(self):
        viewer = make_viewer(2)
        with redirect_stdout(io.StringIO()) as output:
            esmfold_command.run(viewer, ["large"])
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.assertIn("Multiple nodes selected", output.getvalue())

    def test_unknown_and_duplicate_keywords_are_rejected(self):
        for arguments in (["larger"], ["large", "large"], ["multi", "multi"]):
            with self.subTest(arguments=arguments):
                viewer = make_viewer(1)
                with redirect_stdout(io.StringIO()) as output:
                    esmfold_command.run(viewer, arguments)
                self.assertEqual(viewer.background_job_scheduler.jobs, [])
                self.assertIn("Usage: esmfold [large] [multi]", output.getvalue())

    def test_an_unavailable_web_server_queues_nothing(self):
        viewer = make_viewer(1)
        viewer.get_web_url.side_effect = RuntimeError("bind failed")
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory", return_value=structures_dir),
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.QMessageBox, "critical") as critical,
            ):
                esmfold_command.run(viewer, ["large"])

        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        critical.assert_called_once()
        self.assertEqual(viewer.console_text.text, "Error: Viewer web server unavailable.")

    def test_a_viewer_without_a_scheduler_folds_nothing(self):
        viewer = make_viewer(1, scheduler=None)
        engine = esmfold_command.Command_Engine
        with mock.patch.object(engine, "command_failed") as failed, redirect_stdout(io.StringIO()):
            esmfold_command.run(viewer, ["large"])
        failed.assert_called_once_with(viewer, "Error: The background job scheduler is unavailable.")
        self.assertEqual(viewer.console_text.text, "Error: The background job scheduler is unavailable.")

    def test_the_duplicate_keyword_error_reaches_the_console_as_a_message(self):
        viewer = make_viewer(1)
        shown = []
        with (
            mock.patch.object(esmfold_command.Command_Engine, "show_status", side_effect=lambda v, message: shown.append(message)),
            redirect_stdout(io.StringIO()),
        ):
            esmfold_command.run(viewer, ["large", "large"])
        [message] = shown
        self.assertEqual(str(message), "Error: Duplicate esmfold keyword: large")
        self.assertIsInstance(message.values["error"], esmfold_command.Message)

    def test_stop_is_in_the_help(self):
        output = io.StringIO()
        with redirect_stdout(output):
            esmfold_command.print_help()
        self.assertIn("esmfold stop", output.getvalue())
        self.assertNotIn("console window", output.getvalue())
        self.assertNotIn("Only one folding worker", output.getvalue())

    def test_the_command_never_starts_a_terminal(self):
        self.assertFalse(hasattr(esmfold_command, "launch_in_terminal"))
        source = Path(esmfold_command.__file__).read_text(encoding="utf-8")
        self.assertNotIn("launch_in_terminal", source)
        self.assertNotIn("HoldMode", source)


class QueuedFoldTests(CommandCase):
    """What the command hands the scheduler."""

    def test_the_job_names_its_lane_description_cleanup_and_log(self):
        viewer = make_viewer(1)
        run, _ = self.queued(viewer, ["large"])
        job = viewer.background_job_scheduler.jobs[0]
        self.assertEqual(job["command_name"], "esmfold")
        self.assertEqual(job["lane"], "folding")
        self.assertEqual(display_text(job["lane_title"]), "Folding")
        self.assertEqual(job["description"], "esmfold large -> node_0")
        self.assertIs(job["worker"], esmfold_command.fold_job)
        self.assertIs(job["payload"], run)
        self.assertEqual(job["log_path"], run.log_path)
        self.assertEqual(job["cleanup"].__func__, esmfold_command.FoldRun.discard_input)
        self.assertTrue(job["allow_overwrite"], "a refold replaces the structure that is there")
        self.assertEqual(run.job_id, 1)
        self.assertIn(run.run_id, esmfold_command.RUNS)

    def test_every_structure_and_every_residue_data_file_is_reserved(self):
        viewer = make_viewer(2)
        run, _ = self.queued(viewer, ["large", "multi"])
        folder = os.path.abspath(self.structures_dir)
        names = ["node_0_esm3-large-2024-03.pdb", "node_0_esm3-large-2024-03.residues.json",
                 "node_1_esm3-large-2024-03.pdb", "node_1_esm3-large-2024-03.residues.json"]
        [paths] = viewer.background_job_scheduler.reserved
        self.assertEqual([os.path.relpath(path, folder) for path in paths], names)

    def test_a_local_fold_reserves_the_names_without_a_model_suffix(self):
        viewer = make_viewer(1)
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cpu")):
            self.queued(viewer, [])
        [paths] = viewer.background_job_scheduler.reserved
        self.assertEqual([os.path.basename(path) for path in paths], ["node_0.pdb", "node_0.residues.json"])

    def test_a_node_another_fold_holds_is_left_out_and_the_outcome_names_the_job(self):
        viewer = make_viewer(3)
        scheduler = viewer.background_job_scheduler
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cpu")):
            self.queued(make_viewer(1, scheduler=scheduler), [], structures=self.enterContext(tempfile.TemporaryDirectory()))
            first_folder = self.structures_dir
            # The same folder: node_0 is held by job 1.
            run, succeeded = self.queued(viewer, ["multi"], structures=first_folder)
        self.assertEqual([record[0] for record in run.records], ["node_1", "node_2"])
        message = succeeded.call_args.args[1]
        self.assertIn("Left out 1 selected node another fold holds: node_0 (#1).", message)
        self.assertEqual(len(scheduler.jobs), 2)

    def test_a_run_whose_every_node_is_held_is_refused_and_leaves_no_file(self):
        viewer = make_viewer(2)
        scheduler = viewer.background_job_scheduler
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cpu")):
            folder = self.enterContext(tempfile.TemporaryDirectory())
            self.run_reporting(viewer, ["multi"], structures=folder)
            before = set(os.listdir(tempfile.gettempdir()))
            succeeded, failed = self.run_reporting(viewer, ["multi"], structures=folder)
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("Every selected node is already queued or being folded: node_0 (#1), node_1 (#1).",
                      failed.call_args.args[1])
        self.assertTrue(viewer.console_text.text.startswith("Error:"))
        self.assertEqual(len(scheduler.jobs), 1, "nothing more was queued")

    def test_nodes_that_map_to_one_file_name_in_one_run_stay_in_that_run(self):
        # The worker warns that only one structure can be kept under the name; the run still holds each node.
        viewer = make_viewer(2)
        viewer.full_headers = ["same description", "same other"]
        viewer.sequences_map = {"same description": "ACDE", "same other": "ACDF"}
        run, _ = self.queued(viewer, ["large", "multi"])
        self.assertEqual([record[0] for record in run.records], ["same", "same"])

    def test_the_input_file_is_deleted_when_the_scheduler_refuses_the_job(self):
        viewer = make_viewer(1)
        scheduler = viewer.background_job_scheduler
        scheduler.enqueue = mock.Mock(side_effect=FileExistsError("reserved"))
        created = []
        real_write = esmfold_command.FoldRun.write_input

        def write_input(run):
            real_write(run)
            created.append(run.input_path)

        with mock.patch.object(esmfold_command.FoldRun, "write_input", write_input):
            succeeded, failed = self.run_reporting(viewer, ["large"])
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("Could not queue the fold", failed.call_args.args[1])
        self.assertEqual(len(created), 1)
        self.assertFalse(os.path.exists(created[0]))
        self.assertEqual(esmfold_command.RUNS, {})

    def test_the_outcome_says_it_is_queued_with_its_job_and_log(self):
        viewer = make_viewer(1)
        run, succeeded = self.queued(viewer, ["large"])
        message = succeeded.call_args.args[1]
        self.assertEqual(
            message,
            f"Queued Biohub ESM3 folding for 1 structure(s) as job #1. {NO_ALIGNMENT} "
            f"The worker runs without a window; its output is in {run.log_path}.",
        )
        self.assertEqual(
            viewer.console_text.text, f"Queued 1 structure for folding with Biohub ESM3 as job #1. {NO_ALIGNMENT}")

    def test_a_second_fold_says_where_it_waits(self):
        viewer = make_viewer(1)
        self.run_reporting(viewer, ["large"])
        _run, succeeded = self.queued(make_viewer(1, scheduler=viewer.background_job_scheduler), ["large"],
                                      structures=self.enterContext(tempfile.TemporaryDirectory()))
        self.assertIn("as job #2. It is number 2 in the folding line.", succeeded.call_args.args[1])

    def test_a_local_run_needs_no_credentials(self):
        self.credentials.return_value = ("missing", None)
        with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cpu")), \
                mock.patch.object(esmfold_command.token_dialog, "ask_token",
                                  side_effect=AssertionError("a local fold asks for no token")):
            self.queued(make_viewer(1), [])


SECRET = "typed-secret-token-0123"


class TokenTests(CommandCase):
    """esmfold large without a saved token asks for it in a dialog; the token goes nowhere but the settings file."""

    def setUp(self):
        super().setUp()
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.settings_path = os.path.join(directory, "Biohub_API.json")
        from resources import Biohub_API
        self.Biohub_API = Biohub_API
        patcher = mock.patch.object(Biohub_API, "API_SETTINGS_FILE", self.settings_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.credentials.return_value = ("missing", None)
        self.asked = []

        def ask_token(parent, **options):
            self.asked.append(options)
            return self.answer

        self.answer = SECRET
        patcher = mock.patch.object(esmfold_command.token_dialog, "ask_token", side_effect=ask_token)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(esmfold_command.Command_Engine, "command_cancelled")
        self.cancelled = patcher.start()
        self.addCleanup(patcher.stop)

    def viewer(self, state="normal", count=1):
        from tests.viewer_fixtures import apply_window_state
        return apply_window_state(make_viewer(count), state)

    def test_the_token_is_asked_for_saved_and_the_fold_queued_without_showing_it_anywhere(self):
        viewer, output = self.viewer(), io.StringIO()
        run, succeeded = self.queued(viewer, ["large"], output=output)

        self.assertEqual(self.asked, [{}], "asked once, for a token, not for a replacement")
        with open(self.settings_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        self.assertEqual(saved["ESM_API_TOKEN"], SECRET)
        self.assertEqual(saved["ESM3_MODEL"], "esm3-large-2024-03")
        everything = "\n".join([
            output.getvalue(), str(succeeded.call_args), viewer.console_text.text, json.dumps(run.records),
            " ".join(run.command()), str(run.failures), str(run.log_path),
        ])
        self.assertNotIn(SECRET, everything)
        self.assertEqual(len(viewer.background_job_scheduler.jobs), 1)
        self.cancelled.assert_not_called()

    def test_a_token_already_saved_asks_for_nothing(self):
        self.credentials.return_value = ("saved", None)
        self.queued(self.viewer(), ["large"])
        self.assertEqual(self.asked, [])

    def test_cancelling_the_dialog_queues_nothing_and_leaves_no_file(self):
        self.answer = None
        viewer = self.viewer()
        succeeded, failed = self.run_reporting(viewer, ["large"])

        succeeded.assert_not_called()
        failed.assert_not_called()
        self.cancelled.assert_called_once()
        self.assertEqual(str(self.cancelled.call_args.args[1]), "Biohub API token entry was cancelled; nothing was queued.")
        self.assertEqual(viewer.background_job_scheduler.jobs, [])
        self.assertEqual(len(esmfold_command.RUNS), 0)
        self.assertFalse(os.path.exists(self.settings_path))

    def test_a_viewer_with_no_windows_fails_at_once_and_opens_no_dialog(self):
        for state in NO_WINDOW_STATES:
            with self.subTest(state=state):
                self.asked.clear()
                viewer = self.viewer(state)
                succeeded, failed = self.run_reporting(viewer, ["large"])
                succeeded.assert_not_called()
                failed.assert_called_once()
                text = failed.call_args.args[1]
                self.assertIn("Could not ask for the Biohub API token", text)
                self.assertIn("requires a visible Viewer and user input", text)
                if state == "hidden_unshown":
                    self.assertIn("Show the window first", text)
                self.assertEqual(self.asked, [], "no dialog where nobody could answer it")
                self.assertEqual(viewer.background_job_scheduler.jobs, [])
                self.assertFalse(os.path.exists(self.settings_path))

    def test_a_settings_file_that_cannot_be_used_is_refused_not_written_over(self):
        self.credentials.return_value = ("invalid", self.Biohub_API.BiohubSettingsError("Could not read valid Biohub API JSON."))
        viewer = self.viewer()
        succeeded, failed = self.run_reporting(viewer, ["large"])
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("The Biohub API settings cannot be used: Could not read valid Biohub API JSON.", failed.call_args.args[1])
        self.assertEqual(self.asked, [])
        self.assertEqual(viewer.background_job_scheduler.jobs, [])

    def test_nothing_is_asked_when_every_node_is_already_queued(self):
        viewer = self.viewer()
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.credentials.return_value = ("saved", None)
        self.run_reporting(viewer, ["large"], structures=folder)
        self.credentials.return_value = ("missing", None)
        succeeded, failed = self.run_reporting(viewer, ["large"], structures=folder)
        succeeded.assert_not_called()
        self.assertIn("Every selected node is already queued or being folded", failed.call_args.args[1])
        self.assertEqual(self.asked, [], "no one is asked for a token in vain")

    def test_a_token_that_cannot_be_saved_fails_the_command_without_naming_it(self):
        viewer = self.viewer()
        error = self.Biohub_API.BiohubSettingsError(f"Could not write Biohub API settings to '{self.settings_path}'.")
        with mock.patch.object(esmfold_command.token_dialog, "save_token", side_effect=error):
            succeeded, failed = self.run_reporting(viewer, ["large"])
        succeeded.assert_not_called()
        failed.assert_called_once()
        self.assertIn("Could not save the Biohub API token", failed.call_args.args[1])
        self.assertNotIn(SECRET, failed.call_args.args[1])
        self.assertEqual(viewer.background_job_scheduler.jobs, [])


class PrepareAgainTests(unittest.TestCase):
    """What is got ready for the second worker after Biohub rejected the token: the first worker's traces go, the records shrink."""

    class Tracker:
        """Stands in for the first worker's WorkerTracker: it keeps what was asked of it."""

        def __init__(self):
            self.calls = []
            self.timer = SimpleNamespace(stop=lambda: self.calls.append("stop"))

        def poll(self):
            self.calls.append("poll")

        def deleteLater(self):
            self.calls.append("deleted")

    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        patcher = mock.patch.object(Viewer_Worker_Tracking, "__file__", str(root / "src" / "Viewer_Worker_Tracking.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        # Each prepare_again() writes a new input file and leaves the old one to the worker that read it,
        # so every file written is removed here, not only the first and the last.
        self.inputs = []
        write_input = esmfold_command.FoldRun.write_input

        def recording_write_input(run):
            write_input(run)
            self.inputs.append(run.input_path)

        patcher = mock.patch.object(esmfold_command.FoldRun, "write_input", recording_write_input)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: [esmfold_command._discard_worker_input(path) for path in self.inputs])
        records = [["one", "AC", {"x": 1}], ["two", "DE", {"x": 2}], ["three", "FG", {"x": 3}]]
        self.run_ = esmfold_command.FoldRun(
            records, [(record[0], record[1]) for record in records], mode="large", device=None,
            structures_dir=str(root), action_url="", portal=True)
        Path(self.run_.status_path).parent.mkdir(parents=True, exist_ok=True)
        self.run_.write_input()
        self.addCleanup(self.run_.discard_input)
        self.first_input = self.run_.input_path
        self.addCleanup(esmfold_command._discard_worker_input, self.first_input)

    def test_the_first_workers_status_files_go_so_that_the_second_workers_tracker_does_not_read_them(self):
        for suffix in ("", ".partial", ".stdout", ".stderr"):
            Path(self.run_.status_path + suffix).write_text("the first worker's", encoding="utf-8")
        self.run_.prepare_again()
        for suffix in ("", ".partial", ".stdout", ".stderr"):
            self.assertFalse(os.path.exists(self.run_.status_path + suffix), suffix or "the status file")

    def test_the_records_become_those_not_folded_in_a_new_input_file(self):
        self.run_.folded.update({"one"})
        self.run_.prepare_again()
        self.assertEqual([record[0] for record in self.run_.records], ["two", "three"])
        self.assertNotEqual(self.run_.input_path, self.first_input, "the first was the first worker's to read and delete")
        with open(self.run_.input_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), [["two", "DE", {"x": 2}], ["three", "FG", {"x": 3}]])

    def test_the_first_tracker_takes_a_last_look_and_stops_and_the_link_to_its_worker_is_cut(self):
        tracker = self.Tracker()
        self.run_.tracker, self.run_.process, self.run_.worker = tracker, object(), object()
        self.run_.prepare_again()
        self.assertEqual(tracker.calls, ["poll", "stop", "deleted"])
        self.assertEqual((self.run_.tracker, self.run_.process, self.run_.worker), (None, None, None))

    def test_the_second_worker_gets_a_tracker_of_its_own(self):
        started = []
        self.run_.job = SimpleNamespace()
        with mock.patch.object(esmfold_command.FoldRun, "started", autospec=True,
                               side_effect=lambda run, job: started.append(job)):
            self.run_.prepare_again()
        self.assertEqual(started, [self.run_.job])
        self.run_.job = None
        self.run_.prepare_again()  # no job yet: nothing to start, and no error


class ReportingTests(CommandCase):
    def test_nodes_without_a_sequence_are_named_in_the_outcome(self):
        viewer = make_viewer(3)
        del viewer.sequences_map["node_1 description"]
        succeeded, failed = self.run_reporting(viewer, ["large", "multi"])

        failed.assert_not_called()
        [(_, message)] = [call.args for call in succeeded.call_args_list]
        self.assertTrue(message.startswith(
            "Queued Biohub ESM3 folding for 2 structure(s) as job #1. "
            f"Skipped 1 selected node without a sequence. (node_1) {NO_ALIGNMENT}"), message)
        self.assertEqual(
            viewer.console_text.text,
            f"Queued 2 structures for folding with Biohub ESM3 as job #1. Skipped 1 selected node without a sequence. {NO_ALIGNMENT}",
        )

    def test_many_skipped_nodes_are_counted_without_listing_them(self):
        viewer = make_viewer(9)
        for index in range(1, 9):
            del viewer.sequences_map[f"node_{index} description"]
        succeeded, _ = self.run_reporting(viewer, ["multi", "large"])

        [(_, message)] = [call.args for call in succeeded.call_args_list]
        self.assertIn("Skipped 8 selected nodes without a sequence. " + NO_ALIGNMENT, message)
        self.assertNotIn("node_1", message)

    def test_each_failure_is_reported_once(self):
        def selected(count, sequences=True):
            viewer = make_viewer(count)
            if not sequences:
                viewer.sequences_map.clear()
            return viewer

        cases = {
            "no node": (make_viewer(), ["multi"], "No nodes selected"),
            "several nodes": (selected(2), [], "Multiple nodes selected"),
            "no sequence": (selected(1, sequences=False), ["large"], "Could not retrieve sequences"),
        }
        for label, (viewer, arguments, text) in cases.items():
            with self.subTest(label):
                succeeded, failed = self.run_reporting(viewer, arguments)
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn(text, failed.call_args.args[1])
                self.assertTrue(viewer.console_text.text.startswith("Error:"), viewer.console_text.text)

    def test_an_unopened_page_is_mentioned_after_the_job_was_queued(self):
        viewer = make_viewer(1, web_server=SimpleNamespace(has_event_client=lambda client_id: False))
        succeeded, _ = self.run_reporting(viewer, ["large"], opened=False)
        self.assertEqual(len(viewer.background_job_scheduler.jobs), 1)
        self.assertIn("The structure viewer page was not opened", succeeded.call_args.args[1])

        # A page that is already open needs no remark.
        viewer = make_viewer(1, web_server=SimpleNamespace(has_event_client=lambda client_id: client_id == "esmfold"))
        succeeded, _ = self.run_reporting(viewer, ["large"], opened=False)
        self.assertNotIn("not opened", succeeded.call_args.args[1])

    def clicked_viewer(self):
        """Two nodes, none selected, the second one clicked: a click is a highlight, not a selection."""
        viewer = make_viewer(2)
        viewer.selected_indices = []
        viewer.selected_node_idx = 1
        return viewer

    def test_a_clicked_node_is_not_folded_and_a_plain_esmfold_opens_the_structure_viewer(self):
        viewer = self.clicked_viewer()
        output = io.StringIO()
        with (
            mock.patch.object(esmfold_command.esmfold_backend, "register") as register,
            mock.patch.object(Hardware_Utils, "get_optimal_device") as get_device,
        ):
            succeeded, failed = self.run_reporting(viewer, [], output=output)

        failed.assert_not_called()
        self.assertEqual(viewer.background_job_scheduler.jobs, [])  # nothing is folded
        get_device.assert_not_called()
        self.assertEqual(succeeded.call_args.args[1], "Opened the structure viewer; no folding job was requested.")
        self.assertNotIn("clicked", output.getvalue() + succeeded.call_args.args[1] + viewer.console_text.text)

    def test_the_forms_that_fold_fail_with_nothing_selected_whether_or_not_a_node_was_clicked(self):
        for clicked in (False, True):
            for arguments in (["multi"], ["large"], ["large", "multi"]):
                with self.subTest(clicked=clicked, arguments=arguments):
                    viewer = self.clicked_viewer()
                    if not clicked:
                        viewer.selected_node_idx = None
                    succeeded, failed = self.run_reporting(viewer, arguments)

                    self.assertEqual(viewer.background_job_scheduler.jobs, [])
                    succeeded.assert_not_called()
                    self.assertIn("No nodes selected", failed.call_args.args[1])
                    self.assertTrue(viewer.console_text.text.startswith("Error:"), viewer.console_text.text)

    def test_a_selection_is_folded_and_a_node_clicked_besides_is_left_alone(self):
        viewer = make_viewer(2)
        viewer.selected_indices = [0]
        viewer.selected_node_idx = 1
        output = io.StringIO()
        run, succeeded = self.queued(viewer, ["large"], output=output)

        self.assertEqual(run.records, [["node_0", "ACDE"]])
        self.assertNotIn("clicked", output.getvalue() + succeeded.call_args.args[1] + viewer.console_text.text)

    # --- Per-residue data from the loaded alignment ---

    def alignment_viewer(self):
        """Three nodes, "ACDE" each. The alignment has node_0 as it is, node_1 under another sequence, and not node_2."""
        viewer = make_viewer(3)
        aln = sparse_alignment([("a", "AC-DE"), ("b", "ACGDE")])
        viewer.alignment = SimpleNamespace(
            aln=aln, msa_file="D:/alignments/family.fasta", viewer_to_aln=np.array([0, 1, -1])
        )
        return viewer

    def worker_records(self, viewer, arguments=("large", "multi"), **options):
        """Run the command; return (the records the worker will be given, the outcome, the output)."""
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            for patch in options.get("patches", ()):
                stack.enter_context(patch)
            succeeded, failed = self.run_reporting(viewer, list(arguments), output=output)
        failed.assert_not_called()
        return viewer.background_job_scheduler.run.records, succeeded.call_args.args[1], output.getvalue()

    def test_nodes_in_the_loaded_alignment_get_their_data_in_the_workers_input(self):
        viewer = self.alignment_viewer()
        records, outcome, output = self.worker_records(viewer)

        name, sequence, extra = records[0]
        self.assertEqual((name, sequence), ("node_0", "ACDE"))
        self.assertEqual(extra["header"], "node_0 description")
        self.assertEqual(list(extra["properties"]), ["information_content", "same_residue_pct"])
        agreement = extra["properties"]["same_residue_pct"]
        self.assertEqual(len(agreement["values"]), 4)
        self.assertEqual(agreement["source"]["file"], "family.fasta")
        self.assertEqual(agreement["source"]["columns"], [0, 1, 3, 4])
        self.assertEqual(agreement["source"]["sequences"], 2)
        self.assertEqual(records[1:], [["node_1", "ACDE"], ["node_2", "ACDE"]])

        self.assertTrue(outcome.startswith(
            "Queued Biohub ESM3 folding for 3 structure(s) as job #1. "
            "Per-residue alignment data from family.fasta will be saved beside 1 structure: "
            "Information Content, Residue Agreement. "
            "No per-residue alignment data for 1 structure: the sequence differs from the alignment's. (node_1) "
            "No per-residue alignment data for 1 structure: not in the alignment. (node_2)"), outcome)
        self.assertEqual(
            viewer.console_text.text,
            "Queued 3 structures for folding with Biohub ESM3 as job #1. "
            "Per-residue alignment data from family.fasta will be saved beside 1 structure: "
            "Information Content, Residue Agreement. "
            "No per-residue alignment data for 1 structure: the sequence differs from the alignment's. "
            "No per-residue alignment data for 1 structure: not in the alignment.",
        )
        self.assertIn("Note: No per-residue alignment data for node_1: its sequence (4 residues) differs", output)
        self.assertIn("Note: No per-residue alignment data for node_2: it is not in the alignment.", output)

    def test_the_data_is_the_alignment_s_at_the_time_of_the_command_not_at_the_time_of_the_job(self):
        viewer = self.alignment_viewer()
        records, _, _ = self.worker_records(viewer)
        run = viewer.background_job_scheduler.run
        viewer.alignment = None   # unloaded while the job waits
        self.assertEqual(run.records, records)
        self.assertEqual(len(run.records[0]), 3)

    def test_every_node_with_data_needs_no_other_sentence(self):
        viewer = self.alignment_viewer()
        viewer.alignment.viewer_to_aln = np.array([0, 0, 0])
        records, outcome, _ = self.worker_records(viewer)
        self.assertEqual([len(record) for record in records], [3, 3, 3])
        self.assertIn(
            "Per-residue alignment data from family.fasta will be saved beside 3 structures: "
            "Information Content, Residue Agreement.", outcome)

    def test_an_alignment_with_no_rows_loaded_counts_as_no_alignment(self):
        viewer = make_viewer(2)
        viewer.alignment = SimpleNamespace(aln=None, msa_file="x.fasta")
        records, outcome, _ = self.worker_records(viewer)
        self.assertEqual(records, [["node_0", "ACDE"], ["node_1", "ACDE"]])
        self.assertIn("No per-residue alignment data is saved: no alignment is loaded.", outcome)

    def test_a_provider_that_fails_costs_the_data_and_not_the_fold(self):
        def broken(inputs, node_index, sequence):
            raise RuntimeError("boom")

        viewer = self.alignment_viewer()
        patch = mock.patch.dict(esmfold_command.Residue_Data.PROVIDERS, {"broken": broken}, clear=True)
        records, outcome, output = self.worker_records(viewer, patches=[patch])

        self.assertEqual([len(record) for record in records], [2, 2, 2])
        self.assertIn("Warning: Could not compute per-residue data for node_0: RuntimeError: boom", output)
        self.assertIn(
            "No per-residue alignment data for 3 structures: it could not be computed. (node_0, node_1, node_2)", outcome)

    def test_a_failure_to_read_the_alignment_costs_the_data_and_not_the_fold(self):
        viewer = self.alignment_viewer()
        patch = mock.patch.object(
            esmfold_command.Command_Engine, "get_alignment_mapping", side_effect=RuntimeError("bad mapping"))
        records, outcome, output = self.worker_records(viewer, patches=[patch])

        self.assertEqual([len(record) for record in records], [2, 2, 2])
        self.assertIn("Warning: Could not read the loaded alignment for per-residue data: bad mapping", output)
        self.assertIn("No per-residue alignment data for 3 structures: it could not be computed.", outcome)
        self.assertNotIn("no alignment is loaded", outcome)

    # --- Opening the page ---

    def open_page(self, opened, connected=False):
        viewer = make_viewer()
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: connected and client_id == "esmfold")
        engine = esmfold_command.Command_Engine
        with (
            mock.patch.object(esmfold_command.esmfold_backend, "register"),
            mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui", return_value=opened) as open_ui,
            mock.patch.object(engine, "command_succeeded") as succeeded,
            mock.patch.object(engine, "command_failed") as failed,
        ):
            esmfold_command.run(viewer, [])
        return open_ui, succeeded, failed

    def test_the_page_is_reported_opened_only_when_it_opened(self):
        _, succeeded, failed = self.open_page(True)
        self.assertIn("Opened the structure viewer", succeeded.call_args.args[1])
        failed.assert_not_called()

        _, succeeded, failed = self.open_page(False, connected=True)
        self.assertIn("already open", succeeded.call_args.args[1])
        self.assertNotIn("Opened", succeeded.call_args.args[1])
        failed.assert_not_called()

        _, succeeded, failed = self.open_page(False)
        succeeded.assert_not_called()
        self.assertIn("was not opened", failed.call_args.args[1])

    def test_a_portal_command_never_waits_on_an_already_open_dialog(self):
        open_ui, _, _ = self.open_page(False, connected=True)
        open_ui.assert_called_once_with(mock.ANY)

        token = CURRENT.set(object())
        self.addCleanup(CURRENT.reset, token)
        open_ui, _, _ = self.open_page(False, connected=True)
        open_ui.assert_called_once_with(mock.ANY, show_existing_dialog=False)

    def test_opening_the_viewer_page_is_never_refused_by_a_fold_that_runs(self):
        viewer = make_viewer()
        viewer.background_job_scheduler.jobs.append({"command_name": "esmfold", "description": "x", "lane": "folding",
                                                     "payload": None, "output_path": "x"})
        _, succeeded, failed = self.open_page(True)
        failed.assert_not_called()
        self.assertIn("Opened the structure viewer", succeeded.call_args.args[1])


class WindowStateTests(unittest.TestCase):
    """No fold ever has a window of its own; the page opens only where a user is at the Viewer.

    The page opens through the Viewer's real opener, so a page that would open reaches the opener
    mock; only the scheduler, the worker's launcher and the dialogs are stubbed.
    """

    def make_viewer(self, count=1, *, headless=True, connected=False):
        viewer = make_viewer(
            count,
            get_web_url=lambda path="/": f"{ADDRESS}/{str(path).lstrip('/')}",
            web_server=SimpleNamespace(has_event_client=lambda client_id: connected and client_id == "esmfold"),
            update_console_background=lambda: None,
        )
        if headless is not None:
            viewer.headless = headless
        return opening_pages(viewer)

    def run_esmfold(self, viewer, arguments):
        """Run the command with only the scheduler stubbed; return what happened."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        engine = esmfold_command.Command_Engine
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            enter(mock.patch.object(Viewer_Worker_Tracking, "__file__", str(root / "src" / "Viewer_Worker_Tracking.py")))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory",
                                    return_value=str(root / "structures")))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
            enter(mock.patch.object(esmfold_command, "_credentials_state", return_value=("saved", None)))
            enter(mock.patch.object(esmfold_command, "_model_suffix", return_value=None))
            if "large" not in arguments:
                enter(mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")))
            hidden = enter(mock.patch.object(
                esmfold_command, "launch_without_window",
                side_effect=AssertionError("The job launches the worker, not the command.")))
            browser = enter(mock.patch.object(Browser_Page.Desktop_Open, "open_url", return_value=True))
            critical = enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
            information = enter(mock.patch.object(Browser_Page.QMessageBox, "information"))
            succeeded = enter(mock.patch.object(engine, "command_succeeded"))
            failed = enter(mock.patch.object(engine, "command_failed"))
            enter(redirect_stdout(output))
            try:
                esmfold_command.run(viewer, arguments)
            finally:
                for job in viewer.background_job_scheduler.jobs:
                    job["payload"].discard_input()
            enter(mock.patch.object(esmfold_command, "RUNS", esmfold_command.RUNS))
        esmfold_command.RUNS.clear()
        return SimpleNamespace(
            root=root, output=output.getvalue(), hidden=hidden, browser=browser, critical=critical,
            information=information, succeeded=succeeded, failed=failed, scheduler=viewer.background_job_scheduler,
        )

    def outcome(self, run):
        run.failed.assert_not_called()
        [call] = run.succeeded.call_args_list
        return call.args[1]

    def test_the_fold_is_queued_whatever_the_state_and_only_the_page_follows_whether_a_user_is_there(self):
        for state in WINDOW_STATES:
            with self.subTest(state=state):
                viewer = apply_window_state(self.make_viewer(headless=None), state)
                run = self.run_esmfold(viewer, [])
                self.assertEqual(len(run.scheduler.jobs), 1)
                run.hidden.assert_not_called()
                if state in NO_WINDOW_STATES:
                    run.browser.assert_not_called()
                    run.critical.assert_not_called()
                    run.information.assert_not_called()
                    self.assertIn(f"served at {ADDRESS}/esmfold.html", self.outcome(run))
                else:
                    run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new_tab=True)
                    self.assertNotIn("served at", self.outcome(run))
                self.assertIn("The worker runs without a window; its output is in", self.outcome(run))

    def test_the_bare_command_opens_the_page_or_gives_its_address_by_state(self):
        for state in WINDOW_STATES:
            with self.subTest(state=state):
                run = self.run_esmfold(apply_window_state(self.make_viewer(0, headless=None), state), [])
                if state in NO_WINDOW_STATES:
                    run.browser.assert_not_called()
                    self.assertEqual(
                        self.outcome(run),
                        f"A Viewer with no window opens no browser page; the structure viewer is served at "
                        f"{ADDRESS}/esmfold.html. No folding job was requested.",
                    )
                else:
                    run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new_tab=True)
                    self.assertEqual(self.outcome(run), "Opened the structure viewer; no folding job was requested.")

    def test_a_connected_structure_viewer_needs_no_address(self):
        run = self.run_esmfold(self.make_viewer(connected=True), ["large"])
        run.browser.assert_not_called()
        run.information.assert_not_called()
        message = self.outcome(run)
        self.assertIn("The worker runs without a window; its output is in", message)
        self.assertNotIn("served at", message)

    def test_skipped_nodes_are_still_reported(self):
        viewer = self.make_viewer(2)
        del viewer.sequences_map["node_1 description"]
        run = self.run_esmfold(viewer, ["large", "multi"])
        self.assertTrue(self.outcome(run).startswith(
            "Queued Biohub ESM3 folding for 1 structure(s) as job #1. "
            f"Skipped 1 selected node without a sequence. (node_1) {NO_ALIGNMENT} The worker runs without a window"))

    def test_no_page_opens_when_nothing_is_selected_either(self):
        run = self.run_esmfold(self.make_viewer(0), [])
        run.browser.assert_not_called()
        self.assertEqual(run.scheduler.jobs, [])
        self.assertEqual(
            self.outcome(run),
            f"A Viewer with no window opens no browser page; the structure viewer is served at {ADDRESS}/esmfold.html. "
            "No folding job was requested.",
        )

    def test_an_open_page_is_reported_and_an_unavailable_web_server_fails(self):
        run = self.run_esmfold(self.make_viewer(0, connected=True), [])
        self.assertEqual(self.outcome(run), "The structure viewer is already open; no folding job was requested.")
        # Not even the dialog that says so: nobody is at a headless Viewer to close it.
        run.information.assert_not_called()

        viewer = self.make_viewer(0)
        viewer.get_web_url = mock.Mock(side_effect=RuntimeError("bind failed"))
        run = self.run_esmfold(viewer, [])
        run.browser.assert_not_called()
        run.succeeded.assert_not_called()
        run.failed.assert_called_once()
        self.assertIn("was not opened", run.failed.call_args.args[1])

    def test_no_dialog_opens_when_the_web_server_is_unavailable_in_a_viewer_with_no_windows(self):
        viewer = self.make_viewer()
        viewer.get_web_url = mock.Mock(side_effect=RuntimeError("bind failed"))
        run = self.run_esmfold(viewer, ["large"])
        self.assertEqual(run.scheduler.jobs, [])
        run.critical.assert_not_called()
        run.failed.assert_called_once()
        self.assertIn("web server is unavailable", run.failed.call_args.args[1])

    def test_the_web_server_failure_shows_its_dialog_in_a_windowed_viewer_run_by_hand(self):
        viewer = self.make_viewer(headless=False)
        viewer.get_web_url = mock.Mock(side_effect=RuntimeError("bind failed"))
        run = self.run_esmfold(viewer, ["large"])
        run.critical.assert_called_once()
        self.assertEqual(run.scheduler.jobs, [])


class FakeProcess:
    """A worker process that ends when a test says so (or is ended)."""

    def __init__(self, code=0, ends=True):
        self.pid = 4321
        self.returncode = None
        self.code = code
        self.release = __import__("threading").Event()
        if ends:
            self.release.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if not self.release.wait(timeout):
            raise subprocess.TimeoutExpired("worker", timeout)
        self.returncode = self.code
        return self.code

    def end(self, code=-15):
        self.code = code
        self.release.set()


class FoldJobTests(unittest.TestCase):
    """What the job does with the worker: launch it with no window, wait, stop it, say how it ended."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        patcher = mock.patch.object(Viewer_Worker_Tracking, "__file__", str(self.root / "src" / "Viewer_Worker_Tracking.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.viewer = make_viewer(1)
        self.process = FakeProcess()
        self.launched = []
        self.stopped = []
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        enter = stack.enter_context

        def launch(command, **options):
            self.launched.append((command, options))
            return self.process

        enter(mock.patch.object(esmfold_command, "launch_without_window", side_effect=launch))
        enter(mock.patch.object(Viewer_Worker_Tracking, "track_hidden_worker"))
        self.track = enter(mock.patch.object(Viewer_Worker_Tracking, "track_worker", return_value=self.worker()))
        self.run = esmfold_command.FoldRun(
            [["node_0", "ACDE"]], [("node_0", "ACDE")], mode="local", device="cpu",
            structures_dir=str(self.root / "structures"), action_url=URL, portal=False)
        self.run.write_input()
        self.addCleanup(self.run.discard_input)

    def worker(self):
        worker = mock.Mock()
        worker.stop.side_effect = lambda message: (self.stopped.append(str(message)), self.process.end())
        return worker

    def handle(self):
        from utilities.Job_Cancellation import JobHandle
        return JobHandle()

    def run_job(self, handle=None):
        handle = handle or self.handle()
        return esmfold_command.fold_job(self.run, handle), handle

    def test_the_worker_is_launched_with_no_window_its_output_in_a_log_and_the_project_as_its_folder(self):
        result, handle = self.run_job()
        [(command, options)] = self.launched
        self.assertEqual(command, self.run.command())
        self.assertEqual(os.path.abspath(options["cwd"]), COMMAND_PROJECT)
        self.assertEqual(options["log_path"], self.run.log_path)
        self.assertEqual(Path(self.run.log_path).parent, self.root / "temp" / "command_workers")
        self.assertEqual(handle.log_path, self.run.log_path)
        self.assertEqual(str(result["message"]), "Folded 1 structure.")

    def test_the_worker_is_registered_for_stop_and_for_the_viewer_closing(self):
        with mock.patch.object(Viewer_Worker_Tracking, "track_hidden_worker") as hidden:
            self.run_job()
        hidden.assert_called_once_with(self.process)
        self.track.assert_called_once_with(
            self.process, input_path=self.run.input_path, output_dir=self.run.structures_dir, tracker=None)

    def test_a_job_told_to_stop_ends_the_worker_through_the_function_it_registered(self):
        self.process = FakeProcess(ends=False)
        handle = self.handle()
        outcome = []
        thread = __import__("threading").Thread(
            target=lambda: outcome.append(self.attempt(handle)), daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not self.launched and time.monotonic() < deadline:
            time.sleep(0.005)
        while handle._stoppers == [] and time.monotonic() < deadline:
            time.sleep(0.005)
        handle.request()
        thread.join(10)
        self.assertEqual(self.stopped, ["ESMFold worker stopped"])
        self.assertEqual(outcome, ["cancelled"])

    def attempt(self, handle):
        from utilities.Job_Cancellation import JobCancelled
        try:
            esmfold_command.fold_job(self.run, handle)
        except JobCancelled:
            return "cancelled"
        except Exception as error:
            return f"failed: {error}"
        return "succeeded"

    def test_a_job_told_to_stop_before_it_launches_launches_nothing(self):
        handle = self.handle()
        handle.request()
        self.assertEqual(self.attempt(handle), "cancelled")
        self.assertEqual(self.launched, [])

    def test_a_worker_that_cannot_be_started_fails_the_job_with_the_reason(self):
        with mock.patch.object(esmfold_command, "launch_without_window", side_effect=OSError("no python")):
            outcome = self.attempt(self.handle())
        self.assertEqual(outcome, "failed: Could not launch the folding worker without a window:\nno python")

    def test_a_worker_that_ends_badly_fails_the_job_with_what_it_said(self):
        self.process = FakeProcess(code=1)
        self.run.note_event({"event": "structure_failed", "name": "node_0", "reason": "CUDA ran out of memory."})
        self.run.note_event({"event": "structure_failed", "name": "node_1", "reason": "again"})
        outcome = self.attempt(self.handle())
        self.assertEqual(outcome, "failed: Folding node_0 failed: CUDA ran out of memory. 1 more structure failed too.")

    def test_a_worker_that_reported_a_reason_or_nothing_says_that(self):
        self.process = FakeProcess(code=1)
        self.run.note_event({"event": "failed", "reason": "the input was unreadable"})
        self.assertEqual(self.attempt(self.handle()), "failed: The folding worker failed: the input was unreadable")
        self.run.reason = None
        self.process = FakeProcess(code=9)
        self.assertEqual(self.attempt(self.handle()),
                         f"failed: The folding worker stopped (exit code 9); its output is in {self.run.log_path}.")

    def test_the_progress_the_worker_reports_reaches_the_jobs_handle(self):
        self.process = FakeProcess(ends=False)
        handle = self.handle()
        thread = __import__("threading").Thread(target=lambda: self.attempt(handle), daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while self.run.handle is None and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(display_text(handle.progress_text), "Starting the folding worker…")
        self.run.note_event({"event": "started", "total": 3})
        self.assertEqual(display_text(handle.progress_text), "Getting the model ready…")
        self.run.note_event({"event": "progress", "index": 2, "total": 3, "name": "node_b"})
        self.assertEqual(display_text(handle.progress_text), "Folding 2 of 3: node_b")
        self.process.end(0)
        thread.join(10)

    def test_events_reach_only_the_run_this_viewer_started(self):
        esmfold_command._remember(self.run)
        self.addCleanup(esmfold_command.RUNS.clear)
        self.run.handle = self.handle()
        esmfold_command.note_job_event(self.viewer, {"event": "progress", "index": 1, "total": 2, "name": "x", "run": self.run.run_id})
        self.assertEqual(display_text(self.run.handle.progress_text), "Folding 1 of 2: x")
        esmfold_command.note_job_event(self.viewer, {"event": "progress", "index": 2, "total": 2, "name": "y", "run": "someoneelse"})
        esmfold_command.note_job_event(self.viewer, {"event": "progress", "index": 2, "total": 2, "name": "z"})
        self.assertEqual(display_text(self.run.handle.progress_text), "Folding 1 of 2: x")
        esmfold_command.note_structure_folded(self.viewer, self.run.run_id, "node_0")
        esmfold_command.note_structure_folded(self.viewer, "someoneelse", "node_9")
        self.assertEqual(self.run.folded, {"node_0"})

    def test_only_the_last_runs_are_remembered(self):
        esmfold_command.RUNS.clear()
        self.addCleanup(esmfold_command.RUNS.clear)
        runs = [esmfold_command.FoldRun([], [], mode="local", device="cpu", structures_dir="x",
                                        action_url=URL, portal=False)
                for _ in range(esmfold_command.MAX_RUNS_KEPT + 3)]
        for run in runs:
            esmfold_command._remember(run)
        self.assertEqual(list(esmfold_command.RUNS), [run.run_id for run in runs[3:]])

    def test_the_private_input_is_deleted_by_the_cleanup_and_the_cleanup_may_run_twice(self):
        self.assertTrue(os.path.exists(self.run.input_path))
        self.run.discard_input()
        self.assertFalse(os.path.exists(self.run.input_path))
        self.run.discard_input()

    def test_a_portal_run_gets_a_status_file_and_a_plain_run_does_not(self):
        plain = self.run.command()
        self.assertNotIn("--portal-status", plain)
        portal = esmfold_command.FoldRun([], [], mode="large", device=None, structures_dir="x",
                                         action_url=URL, portal=True)
        portal.input_path = "input.json"
        command = portal.command()
        self.assertEqual(command[command.index("--portal-status") + 1], portal.status_path)
        self.assertTrue(Path(portal.status_path).name.startswith("esmfold-"))
        self.assertEqual(Path(portal.status_path).stem, Path(portal.log_path).stem)

    def test_a_retried_run_asks_the_worker_to_keep_the_structures_it_saved(self):
        self.run.skip_existing = True
        self.assertEqual(self.run.command()[-1], "--skip-existing")


class StopTests(CommandCase):
    """esmfold stop: the running fold ends, the folds that wait leave the line."""

    def run_stop(self, arguments, *, ended=0, folds=()):
        """Run `esmfold stop ...` on a viewer whose scheduler holds ``folds`` ((job id, running) pairs)."""
        viewer = make_viewer()
        scheduler = viewer.background_job_scheduler
        scheduler.snapshot = lambda: [
            SimpleNamespace(job=SimpleNamespace(job_id=job_id, command_name="esmfold", description="esmfold"),
                            lane=lane, running=running, position=index + 1)
            for index, (job_id, running, lane) in enumerate(folds)
        ]
        engine = esmfold_command.Command_Engine
        with (
            mock.patch.object(Viewer_Worker_Tracking, "stop_workers", return_value=ended) as stop_workers,
            mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui") as open_ui,
            mock.patch.object(engine, "command_succeeded") as succeeded,
            mock.patch.object(engine, "command_failed") as failed,
            redirect_stdout(io.StringIO()) as output,
        ):
            esmfold_command.run(viewer, arguments)
        open_ui.assert_not_called()
        return SimpleNamespace(
            viewer=viewer, scheduler=scheduler, stop_workers=stop_workers, succeeded=succeeded, failed=failed,
            output=output.getvalue())

    def test_stop_ends_the_running_fold_and_says_how_many(self):
        for arguments in (["stop"], ["STOP"], ["Stop"]):
            with self.subTest(arguments=arguments):
                run = self.run_stop(arguments, folds=[(4, True, "folding"), (5, True, "folding")])
                self.assertEqual(run.scheduler.cancelled, [4, 5])
                run.stop_workers.assert_called_once_with("ESMFold worker stopped")
                run.failed.assert_not_called()
                text = "Stopped 2 ESMFold workers; structures already saved are kept."
                self.assertEqual(str(run.succeeded.call_args.args[1]), text)
                self.assertEqual(run.viewer.console_text.text, text)
                self.assertIn(text, run.output)

        run = self.run_stop(["stop"], folds=[(4, True, "folding")])
        self.assertEqual(str(run.succeeded.call_args.args[1]), "Stopped 1 ESMFold worker; structures already saved are kept.")

    def test_stop_takes_the_folds_that_wait_out_of_the_line_and_says_so(self):
        run = self.run_stop(["stop"], folds=[(4, True, "folding"), (6, False, "folding"), (7, False, "folding")])
        self.assertEqual(run.scheduler.cancelled, [4, 6, 7])
        self.assertEqual(run.scheduler.calls, 1, "all in one step, so the next fold cannot start as the first ends")
        self.assertEqual(
            str(run.succeeded.call_args.args[1]),
            "Stopped 1 ESMFold worker; structures already saved are kept. Removed 2 queued folds.")
        run = self.run_stop(["stop"], folds=[(6, False, "folding")])
        self.assertEqual(str(run.succeeded.call_args.args[1]), "Removed 1 queued fold.")

    def test_stop_leaves_the_jobs_of_other_lanes_and_commands_alone(self):
        run = self.run_stop(["stop"], folds=[(1, True, "files"), (2, False, "files"), (3, True, "folding")])
        self.assertEqual(run.scheduler.cancelled, [3])

    def test_stop_ends_a_worker_that_was_started_some_other_way_too(self):
        run = self.run_stop(["stop"], ended=1)
        self.assertEqual(run.scheduler.cancelled, [])
        self.assertEqual(str(run.succeeded.call_args.args[1]), "Stopped 1 ESMFold worker; structures already saved are kept.")

    def test_stop_with_nothing_running_or_waiting_succeeds_with_a_note(self):
        run = self.run_stop(["stop"])
        run.failed.assert_not_called()
        self.assertEqual(str(run.succeeded.call_args.args[1]), "No ESMFold worker is running.")
        self.assertEqual(run.viewer.console_text.text, "No ESMFold worker is running.")

    def test_stop_takes_no_other_keyword(self):
        for arguments in (["stop", "large"], ["stop", "multi"], ["stop", "stop"]):
            with self.subTest(arguments=arguments):
                run = self.run_stop(arguments, folds=[(4, True, "folding")])
                self.assertEqual(run.scheduler.cancelled, [])
                run.stop_workers.assert_not_called()
                run.succeeded.assert_not_called()
                run.failed.assert_called_once_with(run.viewer, "Error: Stop cannot be combined with other keywords.")
                self.assertIn("Usage: esmfold [large] [multi]", run.output)
                self.assertTrue(run.viewer.console_text.text.startswith("Error:"), run.viewer.console_text.text)

        # As a keyword after others, it is still not a keyword.
        viewer = make_viewer(1)
        with redirect_stdout(io.StringIO()):
            esmfold_command.run(viewer, ["large", "stop"])
        self.assertEqual(viewer.console_text.text, "Error: Unknown esmfold keyword: stop")


# Runs the real worker's main() with the folding replaced by a stand-in that
# writes a one-line PDB per record and tells main()'s notifier, so the status
# file, the mirrored output and the exit code are the real worker's.
# argv: the project's src folder, then the worker's own arguments.
STAND_IN_FOLDING = """
import os, sys
src, *arguments = sys.argv[1:]
sys.path.insert(0, src)
from resources.esmfold import esmfold_worker as worker

def run_predictions(nodes, structures_dir, *, notifier, **options):
    print("Folding with a stand-in model: \\u03b1-amylase")
    for node_id, sequence in nodes:
        name = worker.prediction_filename(node_id)
        with open(os.path.join(structures_dir, name), "w") as handle:
            handle.write("ATOM\\n")
        notifier(node_id, name)
    return len(nodes), False

worker.run_predictions = run_predictions
worker.notify_server = lambda *args, **options: None
raise SystemExit(worker.main(arguments))
"""

# Starts a child of its own, records both process IDs, and runs until it is ended.
STAND_IN_HANGING = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open({pids!r}, "w") as handle:
    handle.write(f"{{os.getpid()}} {{child.pid}}")
time.sleep(120)
"""

# Folds the first record, half writes the second (<name>.<pid>.partial, as the worker does while it
# writes a structure), starts a child of its own, records both process IDs, and runs until it is ended.
STAND_IN_FOLD_THEN_HANG = """
import os, subprocess, sys, time
src, *arguments = sys.argv[1:]
sys.path.insert(0, src)
from resources.esmfold import esmfold_worker as worker

def run_predictions(nodes, structures_dir, *, notifier, **options):
    first, second = nodes
    name = worker.prediction_filename(first[0])
    with open(os.path.join(structures_dir, name), "w") as handle:
        handle.write("ATOM\\n")
    notifier(first[0], name)
    partial = worker.prediction_filename(second[0]) + f".{{os.getpid()}}.partial"
    with open(os.path.join(structures_dir, partial), "w") as handle:
        handle.write("ATO")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open({pids!r}, "w") as handle:
        handle.write(f"{{os.getpid()}} {{child.pid}}")
    time.sleep(120)
    return 2, False

worker.run_predictions = run_predictions
worker.notify_server = lambda *args, **options: None
raise SystemExit(worker.main(arguments))
"""

# Folds each record as a stand-in, saves its residue data with the real worker's writer, and reports it as
# the real run_predictions does: the data file is written before the notifier, and reported saved after it.
STAND_IN_FOLDING_WITH_DATA = """
import os, sys
src, *arguments = sys.argv[1:]
sys.path.insert(0, src)
from resources.esmfold import esmfold_worker as worker

def run_predictions(nodes, structures_dir, *, notifier, residue_data_saved, **options):
    for node_id, sequence, extra in nodes:
        name = worker.prediction_filename(node_id)
        with open(os.path.join(structures_dir, name), "w") as handle:
            handle.write("ATOM\\n")
        sidecar = worker._write_residue_data(structures_dir, name, node_id, sequence, extra)
        notifier(node_id, name)
        residue_data_saved(os.path.join(structures_dir, sidecar))
    return len(nodes), False

worker.run_predictions = run_predictions
worker.notify_server = lambda *args, **options: None
raise SystemExit(worker.main(arguments))
"""

# Folds the first record with its data file, half writes the second's structure and data file
# (<name>.<pid>.partial, as the worker does while it writes), starts a child, and runs until it is ended.
STAND_IN_FOLD_WITH_DATA_THEN_HANG = """
import os, subprocess, sys, time
src, *arguments = sys.argv[1:]
sys.path.insert(0, src)
from resources.esmfold import esmfold_worker as worker

def run_predictions(nodes, structures_dir, *, notifier, residue_data_saved, **options):
    first, second = nodes
    name = worker.prediction_filename(first[0])
    with open(os.path.join(structures_dir, name), "w") as handle:
        handle.write("ATOM\\n")
    sidecar = worker._write_residue_data(structures_dir, name, first[0], first[1], first[2])
    notifier(first[0], name)
    residue_data_saved(os.path.join(structures_dir, sidecar))
    second_name = worker.prediction_filename(second[0])
    for partial in (second_name, worker.residue_data_filename(second_name)):
        with open(os.path.join(structures_dir, partial + f".{{os.getpid()}}.partial"), "w") as handle:
            handle.write("{{")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open({pids!r}, "w") as handle:
        handle.write(f"{{os.getpid()}} {{child.pid}}")
    time.sleep(120)
    return 2, False

worker.run_predictions = run_predictions
worker.notify_server = lambda *args, **options: None
raise SystemExit(worker.main(arguments))
"""


class ProcessCase(PortalFixture, unittest.TestCase):
    """The folding lane as real processes: the real scheduler, portal and tracker, the real worker's
    status reporting, and a real process tree."""

    def setUp(self):
        super().setUp()
        self.root = Path(self.directory.name)
        self.inputs = []
        for name, value in (
            ("_hidden_workers", set()), ("_hidden_quit_hooked", True),
            ("_started_workers", []), ("_started_quit_hooked", True),
        ):
            patcher = mock.patch.object(Viewer_Worker_Tracking, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            Viewer_Worker_Tracking, "__file__", str(self.root / "src" / "Viewer_Worker_Tracking.py"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.scheduler = BackgroundJobScheduler(self.viewer)
        self.viewer.background_job_scheduler = self.scheduler
        self.addCleanup(self.end_workers)
        self.addCleanup(esmfold_command.RUNS.clear)
        self.scripts = 0
        self.current_script = None
        self.script_of = {}
        self.address = ADDRESS
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        enter = stack.enter_context
        enter(mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory",
                                return_value=str(self.root / "structures")))
        enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
        enter(mock.patch.object(esmfold_command, "_credentials_state", return_value=("saved", None)))
        enter(mock.patch.object(esmfold_command, "_model_suffix", return_value=None))
        self.browser = enter(mock.patch.object(Browser_Page.Desktop_Open, "open_url"))
        self.critical = enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
        real_launch = esmfold_command.launch_without_window
        real_write = esmfold_command.FoldRun.write_input

        def write_input(run):
            real_write(run)
            # [python, worker script, input, structures, ...]: the stand-in takes the worker's place. The real
            # worker deletes its private input (--delete-input); a stand-in that fails or hangs before reading
            # it leaves the file for the test to remove.
            self.inputs.append(run.input_path)
            self.script_of[run.run_id] = self.current_script

        def launch(command, **options):
            run_id = command[command.index("--run-id") + 1]
            return real_launch([command[0], str(self.script_of[run_id]), SRC_DIR, *command[2:]], **options)

        enter(mock.patch.object(esmfold_command.FoldRun, "write_input", autospec=True, side_effect=write_input))
        enter(mock.patch.object(esmfold_command, "launch_without_window", side_effect=launch))

    def end_workers(self):
        end_scheduler(self.scheduler, 5.0)
        Viewer_Worker_Tracking.stop_hidden_workers()
        Viewer_Worker_Tracking.stop_workers("test over")
        for path in self.inputs:
            if os.path.exists(path):
                os.unlink(path)

    def make_viewer(self, headless=True, selected=(0,), state=None):
        """The Viewer the commands run in; ``state`` is one of tests.viewer_fixtures.WINDOW_STATES."""
        viewer = self.viewer
        viewer.headless = headless
        viewer.selected_indices = list(selected)
        viewer.selected_node_idx = None
        viewer.sequences_map = {"one": "ACDE", "two": "ACDF", "three": "ACDG"}
        viewer.full_headers = ["one", "two", "three"]
        viewer.get_web_url = lambda path="/": f"{self.address}/{str(path).lstrip('/')}"
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: False)
        opening_pages(viewer)
        if state is not None:
            apply_window_state(viewer, state)
        return viewer

    def run_in_portal(self, script_text, arguments=("large",), selected=(0,), name=None, state=None):
        """Run esmfold as a portal command whose worker is script_text; return (request id, context, mocks).

        The job launches the worker when its turn comes, after the command has returned, so the stand-ins
        for the launcher and the folder (setUp) stay in place for the whole test.
        """
        self.scripts += 1
        script = self.root / f"stand_in_{self.scripts}.py"
        script.write_text(script_text, encoding="utf-8")
        self.current_script = script
        name = name or f"fold-{self.scripts}"
        request_id = self.portal.submit(name, "test")["request_id"]
        record = self.portal.requests[request_id]["commands"][0]
        record.update(dispatched=True, outcome="succeeded", status="running")
        context = ExecutionContext(self.portal, request_id, record)
        with redirect_stdout(io.StringIO()):
            with bind(context):
                esmfold_command.run(self.make_viewer(selected=selected, state=state), list(arguments))
        return request_id, context, SimpleNamespace(browser=self.browser, critical=self.critical)

    def run_typed(self, script_text, arguments=("large",), selected=(0,), state=None):
        """Run esmfold as the user types it, with no portal request: a fold with no status file and no tracker."""
        self.scripts += 1
        script = self.root / f"stand_in_{self.scripts}.py"
        script.write_text(script_text, encoding="utf-8")
        self.current_script = script
        with redirect_stdout(io.StringIO()):
            with bind(None):
                esmfold_command.run(self.make_viewer(selected=selected, state=state), list(arguments))

    def logs(self):
        return sorted((self.root / "temp" / "command_workers").glob("*.log"))

    def wait_for(self, condition, message, timeout=60):
        deadline = time.monotonic() + timeout
        while not condition():
            self.assertLess(time.monotonic(), deadline, message)
            self.app.processEvents()
            time.sleep(0.02)

    def jobs(self, request_id):
        return self.portal.get(request_id)["commands"][0]["jobs"]


class FoldProcessTests(ProcessCase):
    def test_a_fold_reports_through_the_portal_and_writes_its_log(self):
        request_id, context, mocks = self.run_in_portal(STAND_IN_FOLDING)

        result = self.finish(request_id)

        self.assertEqual(result["status"], "succeeded", result)
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["job_id"], "1", "the scheduler's job is the portal's job: one entry, not two")
        self.assertEqual(job["result"], {"succeeded": 1, "total": 1})
        pdb = os.path.abspath(self.root / "structures" / "one.pdb")
        self.assertEqual(result["commands"][0]["artifacts"], [pdb])
        self.assertTrue(os.path.exists(pdb))
        # The worker's own output still reaches the portal, and the log holds all of it.
        self.assertIn("Successfully completed 1 / 1", self.portal.read_output(request_id)["text"])
        [log] = self.logs()
        text = log.read_text(encoding="utf-8")
        self.assertIn("Folding with a stand-in model: α-amylase", text)
        self.assertIn("Successfully completed 1 / 1", text)
        # The outcome tells where the log is and where the page is; nothing opened.
        messages = [entry["text"] for entry in context.record["messages"]]
        self.assertTrue(any(f"its output is in {log}" in text and f"{ADDRESS}/esmfold.html" in text for text in messages), messages)
        mocks.browser.assert_not_called()
        mocks.critical.assert_not_called()

    def with_alignment(self):
        """The Viewer holds an alignment with nodes one and two, so both get per-residue data."""
        aln = sparse_alignment([("a", "AC-DE"), ("b", "AC-DF")])
        self.viewer.alignment = SimpleNamespace(
            aln=aln, msa_file="D:/alignments/family.fasta", viewer_to_aln=np.array([0, 1, -1]))

    def test_a_fold_with_an_alignment_reports_each_structure_and_its_data_file_as_artifacts(self):
        self.with_alignment()
        request_id, context, _ = self.run_in_portal(
            STAND_IN_FOLDING_WITH_DATA, arguments=("large", "multi"), selected=(0, 1))

        result = self.finish(request_id)

        self.assertEqual(result["status"], "succeeded", result)
        structures = self.root / "structures"
        names = ["one.pdb", "one.residues.json", "two.pdb", "two.residues.json"]
        self.assertEqual(result["commands"][0]["artifacts"], [os.path.abspath(structures / name) for name in names])
        self.assertEqual(sorted(path.name for path in structures.iterdir()), names)
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(job["result"], {"succeeded": 2, "total": 2})
        document = json.loads((structures / "one.residues.json").read_text(encoding="utf-8"))
        self.assertEqual(document["format"], "emapssn-residue-data")
        self.assertEqual((document["structure"], document["header"], document["sequence"]), ("one.pdb", "one", "ACDE"))
        self.assertEqual(document["properties"]["same_residue_pct"]["values"], [100.0, 100.0, 100.0, 50.0])

    def test_a_worker_that_fails_to_start_fails_the_job_with_its_output_in_the_log(self):
        request_id, context, _ = self.run_in_portal("raise ImportError('no torch here')\n")

        result = self.finish(request_id)

        self.assertEqual(result["status"], "failed", result)
        [log] = self.logs()
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(job["status"], "failed")
        self.assertIn(f"its output is in {log}", job["message"])
        self.assertIn("ImportError: no torch here", log.read_text(encoding="utf-8"))
        # The card keeps the failure, with the log to open.
        (failure,) = self.scheduler.failures()
        self.assertEqual((failure.lane, failure.log_path), ("folding", str(log)))
        self.assertIn("exit code 1", display_text(failure.error))

    def test_the_run_is_followed_by_one_tracker_that_reports_into_the_scheduler_s_job(self):
        request_id, context, _ = self.run_in_portal(STAND_IN_FOLDING)
        self.finish(request_id)
        [job] = context.record["jobs"]
        self.assertEqual(job["job_id"], "1")
        self.assertFalse(any(entry["job_id"].startswith("worker-") for entry in context.record["jobs"]))


class FoldStopProcessTests(ProcessCase):
    def start_folding(self, script=STAND_IN_FOLD_THEN_HANG, arguments=("large", "multi"), selected=(0, 1), name=None):
        """Start a fold of two nodes whose worker folds one and hangs on the second; return its pieces."""
        pids = self.root / f"pids{self.scripts + 1}.txt"
        request_id, context, _ = self.run_in_portal(
            script.format(pids=str(pids)), arguments=arguments, selected=selected, name=name)
        self.wait_for(lambda: len(pids.read_text().split() if pids.exists() else []) == 2,
                      "The stand-in worker never started its child")
        processes = [psutil.Process(int(field)) for field in pids.read_text().split()]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        # The tracker reads the worker's status: the folded structure is reported before the stop.
        expected = 2 if script is STAND_IN_FOLD_WITH_DATA_THEN_HANG else 1
        self.wait_for(lambda: len(self.portal.requests[request_id]["commands"][0]["artifacts"]) >= expected,
                      "The first structure was never reported")
        return request_id, context, processes

    def stop_through_the_portal(self, text="esmfold stop"):
        """Submit `esmfold stop` behind the fold request, which still waits for its worker."""
        request = self.portal.submit("stop-" + text.replace(" ", "-") + str(time.monotonic()), text)
        return self.finish(request["request_id"])

    def assert_stopped(self, request_id, processes, kept=("one.pdb",)):
        structures = self.root / "structures"
        pdb = os.path.abspath(structures / "one.pdb")
        # The fold request ends cancelled, not failed, with its job cancelled and the finished structure kept.
        result = self.finish(request_id)
        self.assertEqual(result["status"], "cancelled", result)
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(result["commands"][0]["artifacts"], [os.path.abspath(structures / name) for name in kept])
        self.assertTrue(os.path.exists(pdb))
        # The half-written structure, and its half-written data, leave no file, under any name.
        self.assertEqual(sorted(path.name for path in structures.iterdir()), sorted(kept))
        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])
        self.assertFalse([path for path in self.inputs if os.path.exists(path)])
        self.assertEqual(list((self.root / "temp" / "command_workers").glob("*.partial")), [])
        self.wait_for(lambda: self.scheduler.queue_depth == 0, "The scheduler never let the fold go")
        for suffix in ("one", "two"):
            self.assertFalse(self.scheduler.is_output_path_reserved(str(structures / f"{suffix}.pdb")))

    def test_a_fold_is_stopped_while_its_own_request_waits(self):
        request_id, context, processes = self.start_folding()
        self.assertEqual(self.portal.get(request_id)["status"], "running")

        stop = self.stop_through_the_portal()

        self.assertEqual(stop["status"], "succeeded", stop)
        [message] = [entry["text"] for entry in stop["commands"][0]["messages"]]
        self.assertEqual(message, "Stopped 1 ESMFold worker; structures already saved are kept.")
        self.assert_stopped(request_id, processes)
        # A second stop finds nothing running.
        again = self.stop_through_the_portal("esmfold  STOP")
        self.assertEqual(again["status"], "succeeded", again)
        self.assertEqual(again["commands"][0]["messages"][-1]["text"], "No ESMFold worker is running.")

    def test_stop_keeps_the_saved_data_file_and_leaves_no_partial_one(self):
        aln = sparse_alignment([("a", "AC-DE"), ("b", "AC-DF")])
        self.viewer.alignment = SimpleNamespace(
            aln=aln, msa_file="D:/alignments/family.fasta", viewer_to_aln=np.array([0, 1, -1]))
        request_id, context, processes = self.start_folding(script=STAND_IN_FOLD_WITH_DATA_THEN_HANG)
        structures = self.root / "structures"
        # Before the stop: one finished structure with its data, and the half-written pair of the second.
        listed = sorted(path.name for path in structures.iterdir())
        self.assertEqual([name for name in listed if not name.endswith(".partial")], ["one.pdb", "one.residues.json"])
        self.assertEqual(len([name for name in listed if name.endswith(".partial")]), 2)

        stop = self.stop_through_the_portal()

        self.assertEqual(stop["status"], "succeeded", stop)
        self.assert_stopped(request_id, processes, kept=("one.pdb", "one.residues.json"))
        json.loads((structures / "one.residues.json").read_text(encoding="utf-8"))  # Whole, not truncated.

    def test_the_session_card_s_stop_does_what_esmfold_stop_does(self):
        # Removing the running fold from the card is the scheduler's cancel: the same function ends the worker.
        request_id, context, processes = self.start_folding()
        (entry,) = [entry for entry in self.scheduler.snapshot() if entry.running]
        self.assertEqual(entry.lane, "folding")
        self.assertEqual(entry.log_path, str(self.logs()[0]))
        self.assertTrue(self.scheduler.cancel(entry.job.job_id))
        self.assert_stopped(request_id, processes)

    def test_a_worker_is_stopped_before_it_reports_startup(self):
        request_id, context, _ = self.run_in_portal("import time\ntime.sleep(120)\n")
        self.wait_for(lambda: bool(self.scheduler.snapshot()) and self.scheduler.snapshot()[0].running
                      and esmfold_command.RUNS and next(iter(esmfold_command.RUNS.values())).process is not None,
                      "The worker never started")
        run = next(iter(esmfold_command.RUNS.values()))
        self.wait_for(lambda: run.tracker is not None, "The tracker never started")
        self.assertIsNone(run.tracker.identity)  # It has reported nothing.

        stop = self.stop_through_the_portal()

        self.assertEqual(stop["status"], "succeeded", stop)
        result = self.finish(request_id)
        self.assertEqual(result["status"], "cancelled", result)
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(job["status"], "cancelled")
        self.assertIsNotNone(run.process.poll())
        self.wait_for(lambda: not run.tracker.timer.isActive(), "The tracker kept polling")
        self.assertFalse([path for path in self.inputs if os.path.exists(path)])
        # Nothing later turns the cancelled job into a failure.
        for _ in range(5):
            run.tracker.poll()
        self.assertEqual(self.portal.get(request_id)["status"], "cancelled")

    def test_a_worker_that_already_finished_is_not_stopped(self):
        request_id, context, mocks = self.run_in_portal(STAND_IN_FOLDING)
        self.assertEqual(self.finish(request_id)["status"], "succeeded")

        stop = self.stop_through_the_portal()

        self.assertEqual(stop["commands"][0]["messages"][-1]["text"], "No ESMFold worker is running.")
        self.assertTrue(os.path.exists(self.root / "structures" / "one.pdb"))

    def test_only_the_workers_this_viewer_started_are_stopped(self):
        # A worker of another Viewer: the same script and the same shape of command line.
        script = self.root / "other_worker.py"
        script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")
        other = subprocess.Popen([sys.executable, str(script), str(self.root / "someone_elses_input.json")])
        self.addCleanup(lambda: (other.kill(), other.wait(timeout=10)))
        request_id, context, processes = self.start_folding()

        self.stop_through_the_portal()

        self.assert_stopped(request_id, processes)
        self.assertIsNone(other.poll())

    def test_closing_the_viewer_ends_the_worker_and_what_it_started_within_the_bound(self):
        pids = self.root / "pids-close.txt"
        request_id, context, _ = self.run_in_portal(STAND_IN_HANGING.format(pids=str(pids)))
        self.wait_for(lambda: len(pids.read_text().split() if pids.exists() else []) == 2, "No child started")
        processes = [psutil.Process(int(field)) for field in pids.read_text().split()]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        run = next(iter(esmfold_command.RUNS.values()))
        self.assertIsNone(run.process.poll())

        started = time.monotonic()
        self.scheduler.shutdown()                      # what the window's close and the application's quit run
        self.assertLess(time.monotonic() - started, 6.0)

        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])
        self.assertIsNotNone(run.process.poll())

    def test_closing_the_viewer_through_the_quit_hook_ends_the_worker_too(self):
        pids = self.root / "pids-hook.txt"
        self.run_in_portal(STAND_IN_HANGING.format(pids=str(pids)))
        self.wait_for(lambda: len(pids.read_text().split() if pids.exists() else []) == 2, "No child started")
        processes = [psutil.Process(int(field)) for field in pids.read_text().split()]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        # What the application's aboutToQuit signal runs besides the scheduler's shutdown.
        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 1)
        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])


class FoldQueueProcessTests(ProcessCase):
    """Folds queue behind each other, and a waiting fold is as good as any job for an MCP client to wait on.

    The portal runs one request at a time, so a fold that waits for another comes from a request that
    finds one already going: here, one the user typed.
    """

    def hold_the_lane(self, name="first"):
        """A typed fold whose worker hangs, so that what is queued after it waits; returns its processes."""
        pids = self.root / f"{name}.txt"
        self.run_typed(STAND_IN_HANGING.format(pids=str(pids)), selected=(0,))
        self.wait_for(lambda: len(pids.read_text().split() if pids.exists() else []) == 2, "No child started")
        processes = [psutil.Process(int(field)) for field in pids.read_text().split()]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        return processes

    def test_a_fold_requested_while_another_runs_waits_and_runs_when_it_ends(self):
        processes = self.hold_the_lane()
        second, context, _ = self.run_in_portal(STAND_IN_FOLDING, selected=(1,), name="second")

        # Not refused: it waits, as a job of its own, and the request keeps running for it.
        self.wait_for(lambda: self.jobs(second) and self.jobs(second)[0].get("position") == 2,
                      "The place in the line was never recorded")
        job = self.jobs(second)[0]
        self.assertEqual((job["status"], job["position"], job["waiting_for"]), ("queued", 2, "1"))
        self.assertEqual(self.portal.get(second)["commands"][0]["status"], "running", "the command waits for its fold")
        self.assertFalse((self.root / "structures" / "two.pdb").exists())
        lanes = {(entry.job.job_id, entry.running, entry.position) for entry in self.scheduler.snapshot()}
        self.assertEqual(lanes, {(1, True, 1), (2, False, 2)})

        # The first fold ends; the second runs and completes the request.
        for process in processes:
            process.kill()
        result = self.finish(second)
        self.assertEqual(result["status"], "succeeded", result)
        self.assertTrue((self.root / "structures" / "two.pdb").exists())
        self.assertEqual(self.jobs(second)[0]["status"], "succeeded")

    def test_a_waiting_fold_removed_before_it_starts_never_runs_and_leaves_no_input_file(self):
        self.hold_the_lane()
        second, context, _ = self.run_in_portal(STAND_IN_FOLDING, selected=(1,), name="second")
        self.wait_for(lambda: bool(self.jobs(second)), "The fold was never queued")
        waiting_input = next(run.input_path for run in esmfold_command.RUNS.values() if run.job_id == 2)
        self.assertTrue(os.path.exists(waiting_input))

        self.assertTrue(self.scheduler.cancel(2))

        result = self.finish(second)
        self.assertEqual(result["status"], "cancelled", result)
        self.assertEqual(self.jobs(second)[0]["status"], "cancelled")
        self.assertFalse(os.path.exists(waiting_input), "the private input of a removed fold is deleted")
        self.assertFalse(self.scheduler.is_output_path_reserved(str(self.root / "structures" / "two.pdb")))
        self.assertFalse((self.root / "structures" / "two.pdb").exists())

    def test_jobs_cancel_reaches_a_fold_by_the_number_it_shares_with_every_job(self):
        from commands import jobs as jobs_command
        processes = self.hold_the_lane()
        second, _, _ = self.run_in_portal(STAND_IN_FOLDING, selected=(1,), name="second")
        self.wait_for(lambda: bool(self.jobs(second)), "The fold was never queued")
        entry = self.jobs(second)[0]
        # The scheduler's own number, as jobs lists it: not a worker-<hex> that nothing could name.
        self.assertEqual(entry["job_id"], "2")
        self.assertEqual((entry["kind"], entry["lane"], entry["lane_title"]), ("esmfold", "folding", "Folding"))

        def jobs(*words):
            output = io.StringIO()
            with redirect_stdout(output), bind(None):
                jobs_command.run(self.make_viewer(), list(words))
            return output.getvalue()

        listing = jobs()
        self.assertIn("#1     running", listing)
        self.assertIn("#2     waiting", listing)
        self.assertIn("(folding)", listing)

        # The waiting fold is cancelled, the running one only told to stop: it is not "cancelled" while its
        # worker still ends.
        said = jobs("cancel", "1", "2")
        self.assertIn("Cancelled 1 job: #2 ", said)
        self.assertIn("Told 1 running job to stop: #1 ", said)
        self.assertEqual(self.finish(second)["status"], "cancelled")
        self.wait_for(lambda: self.scheduler.snapshot() == [], "The line never emptied")
        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])

    def test_esmfold_stop_ends_the_running_fold_and_removes_the_waiting_ones(self):
        processes = self.hold_the_lane()
        self.run_typed(STAND_IN_FOLDING, selected=(1,))
        third, _, _ = self.run_in_portal(STAND_IN_FOLDING, selected=(2,), name="third")
        self.wait_for(lambda: len(self.scheduler.snapshot()) == 3, "The folds were never queued")
        inputs = [run.input_path for run in esmfold_command.RUNS.values()]

        request = self.portal.submit("stop-all", "esmfold stop")
        stop = self.finish(request["request_id"])

        self.assertEqual(stop["status"], "succeeded", stop)
        self.assertEqual(
            stop["commands"][0]["messages"][-1]["text"],
            "Stopped 1 ESMFold worker; structures already saved are kept. Removed 2 queued folds.")
        self.assertEqual(self.finish(third)["status"], "cancelled")
        self.wait_for(lambda: self.scheduler.snapshot() == [], "The line never emptied")
        self.assertFalse([path for path in inputs + self.inputs if os.path.exists(path)])
        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])
        self.assertEqual(list((self.root / "structures").glob("*.pdb")), [])

    def test_the_card_lists_the_folds_in_their_own_lane_beside_the_files(self):
        self.hold_the_lane()
        self.run_in_portal(STAND_IN_FOLDING, selected=(1,), name="second")
        from desktop.Viewer_Jobs import job_lanes

        # A file job runs at the same time, in the files lane.
        gate = __import__("threading").Event()
        self.addCleanup(gate.set)
        self.scheduler.enqueue("print", "print -> map.png", None, lambda payload: gate.wait(30) and {},
                               str(self.root / "map.png"))

        def both_run():
            seen = {lane.name: lane for lane in job_lanes(self.scheduler)}
            return set(seen) == {"files", "folding"} and [row.state for row in seen["files"].rows] == ["running"]

        self.wait_for(both_run, "The file job never ran beside the fold")
        lanes = {lane.name: lane for lane in job_lanes(self.scheduler)}
        self.assertEqual([row.state for row in lanes["folding"].rows], ["running", "queued"])
        self.assertEqual([row.state for row in lanes["files"].rows], ["running"])
        self.assertEqual(display_text(lanes["folding"].title), "Folding")
        self.assertEqual(lanes["folding"].rows[0].description, "esmfold large -> one")
        self.assertEqual(lanes["folding"].rows[0].log_path, str(self.logs()[0]))


# Runs the real worker's main() with the folding replaced by a stand-in: the first run folds its first record and
# meets Biohub's rejection of the token at its second (raising what the real client does), and every later run
# folds what it is given. It writes the node ids each run was given to a file, and posts what it folds to the Viewer
# through the real notifier. REJECT_EVERY_RUN makes every run meet the rejection at its first record.
STAND_IN_REJECTED = """
import os, sys
src, *arguments = sys.argv[1:]
sys.path.insert(0, src)
from resources.esmfold import esmfold_worker as worker

MARKER = MARKER_PATH
GIVEN = GIVEN_PATH
EVERY_RUN = REJECT_EVERY_RUN

def run_predictions(nodes, structures_dir, *, notifier, **options):
    first_run = not os.path.exists(MARKER)
    open(MARKER, "w").close()
    with open(GIVEN, "a") as handle:
        handle.write(",".join(node[0] for node in nodes) + "\\n")
    rejected_at = 0 if EVERY_RUN and not first_run else 1
    for index, node in enumerate(nodes):
        if (first_run or EVERY_RUN) and index == rejected_at:
            raise worker.Biohub_API.BiohubAuthenticationError(401, "Biohub rejected the saved API token.")
        name = worker.prediction_filename(node[0])
        with open(os.path.join(structures_dir, name), "w") as handle:
            handle.write("ATOM\\n")
        notifier(node[0], name)
    return len(nodes), False

worker.run_predictions = run_predictions
raise SystemExit(worker.main(arguments))
"""

OLD_TOKEN = "old-saved-token-4567"
NEW_TOKEN = "replacement-token-8901"


class ActionServer:
    """Receives the worker's posts to /api/action as the Viewer's web server does and hands them to the esmfold backend."""

    def __init__(self, viewer):
        import http.server
        import threading
        backend = esmfold_command.esmfold_backend
        handlers = {"structure_folded": backend.handle_structure_folded, "esmfold_job": backend.handle_esmfold_job}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                handler = handlers.get(body.get("action"))
                if handler is not None:
                    handler(viewer, body)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *arguments):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.address = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class RejectedTokenProcessTests(ProcessCase):
    """Biohub rejects the saved token: the Viewer asks for another in a dialog and folds the rest with it."""

    def setUp(self):
        super().setUp()
        from resources import Biohub_API
        self.Biohub_API = Biohub_API
        self.settings_path = str(self.root / "Biohub_API.json")
        patcher = mock.patch.object(Biohub_API, "API_SETTINGS_FILE", self.settings_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        Biohub_API.write_api_settings(
            {"ESM_API_TOKEN": OLD_TOKEN, "ESM_API_URL": "https://biohub.example", "ESM3_MODEL": "esm3-large-2024-03"},
            self.settings_path)
        self.server = ActionServer(self.viewer)
        self.addCleanup(self.server.close)
        self.address = self.server.address
        self.marker = self.root / "worker-ran-once"
        self.given = self.root / "given.txt"
        self.replies = []
        self.dialogs = []
        patcher = mock.patch.object(esmfold_command.token_dialog, "ask_token", side_effect=self.ask_token)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask_token(self, parent, *, replacement=False, shown=None):
        context = CURRENT.get()
        status = context.portal.requests[context.request_id]["status"] if context is not None else None
        self.dialogs.append({"replacement": replacement, "request_status": status})
        answer = self.replies.pop(0)
        return answer(shown) if callable(answer) else answer

    def script(self, every_run=False):
        return (STAND_IN_REJECTED.replace("MARKER_PATH", repr(str(self.marker)))
                .replace("GIVEN_PATH", repr(str(self.given))).replace("REJECT_EVERY_RUN", repr(every_run)))

    def fold_three(self, **options):
        script = self.script(options.pop("every_run", False))
        return self.run_in_portal(script, arguments=("large", "multi"), selected=(0, 1, 2),
                                  state=options.pop("state", "normal"))

    def given_lines(self):
        return self.given.read_text().split() if self.given.exists() else []

    def saved_settings(self):
        return self.Biohub_API.read_api_settings(self.settings_path)

    def everything_shown(self, request_id):
        """Every text a request, its output, its log and the saved status files hold."""
        parts = [json.dumps(self.portal.get(request_id, limit=100)),
                 self.portal.read_output(request_id, "stdout")["text"], self.portal.read_output(request_id, "stderr")["text"]]
        parts += [path.read_text(encoding="utf-8", errors="replace") for path in self.logs()]
        parts += [path.read_text(encoding="utf-8", errors="replace")
                  for path in (self.root / "temp" / "command_workers").glob("*") if path.suffix != ".log" and path.is_file()]
        return "\n".join(parts)

    def test_the_token_is_replaced_in_a_dialog_and_only_the_structures_not_yet_folded_run_again(self):
        self.replies = [NEW_TOKEN]
        request_id, _, _ = self.fold_three()

        result = self.finish(request_id)

        self.assertEqual(result["status"], "succeeded", result)
        # Asked once, as a replacement, while the request waited for the user: not as a failure.
        self.assertEqual(self.dialogs, [{"replacement": True, "request_status": "awaiting_user_input"}])
        self.assertEqual(self.given_lines(), ["one,two,three", "two,three"], "the folded structure is not folded again")
        self.assertEqual(sorted(path.name for path in (self.root / "structures").glob("*.pdb")),
                         ["one.pdb", "three.pdb", "two.pdb"])
        saved = self.saved_settings()
        self.assertEqual((saved["ESM_API_TOKEN"], saved["ESM_API_URL"], saved["ESM3_MODEL"]),
                         (NEW_TOKEN, "https://biohub.example", "esm3-large-2024-03"), "the URL and model are kept")
        # One job, one entry, from queued to succeeded: the second worker is the same job.
        [job] = result["commands"][0]["jobs"]
        self.assertEqual((job["job_id"], job["status"]), ("1", "succeeded"))
        self.assertEqual(self.finish(request_id)["commands"][0]["status"], "succeeded")
        # What the first worker saved and what the second did are artifacts of the one request: each worker has its tracker.
        self.assertEqual(sorted(os.path.basename(path) for path in result["commands"][0]["artifacts"]),
                         ["one.pdb", "three.pdb", "two.pdb"])
        shown = self.everything_shown(request_id)
        self.assertNotIn(NEW_TOKEN, shown)
        self.assertNotIn(OLD_TOKEN, shown)
        self.assertEqual(self.scheduler.snapshot(), [])

    def test_a_replacement_that_is_rejected_too_ends_the_job_and_keeps_what_was_folded(self):
        self.replies = [NEW_TOKEN]
        request_id, _, _ = self.fold_three(every_run=True)

        result = self.finish(request_id)

        self.assertEqual(result["status"], "failed", result)
        self.assertIn("Biohub rejected the replacement token too", result["commands"][0]["jobs"][0]["message"])
        self.assertEqual(len(self.dialogs), 1, "asked once, never again")
        self.assertEqual(sorted(path.name for path in (self.root / "structures").glob("*.pdb")), ["one.pdb"])
        self.assertNotIn(NEW_TOKEN, self.everything_shown(request_id))

    def test_cancelling_the_dialog_fails_the_job_and_leaves_the_saved_token_alone(self):
        self.replies = [None]
        request_id, _, _ = self.fold_three()

        result = self.finish(request_id)

        self.assertEqual(result["status"], "failed", result)
        self.assertIn("no replacement was entered", result["commands"][0]["jobs"][0]["message"])
        self.assertEqual(self.saved_settings()["ESM_API_TOKEN"], OLD_TOKEN)
        self.assertEqual(self.given_lines(), ["one,two,three"], "no second worker")
        self.assertEqual(sorted(path.name for path in (self.root / "structures").glob("*.pdb")), ["one.pdb"])

    def test_a_viewer_with_no_windows_cannot_ask_and_the_job_fails_saying_so(self):
        for state in NO_WINDOW_STATES:
            with self.subTest(state=state):
                self.marker.unlink(missing_ok=True)
                self.given.unlink(missing_ok=True)
                self.dialogs.clear()
                request_id, _, _ = self.fold_three(state=state)
                result = self.finish(request_id)
                self.assertEqual(result["status"], "failed", result)
                message = result["commands"][0]["jobs"][0]["message"]
                self.assertIn("a replacement cannot be asked for", message)
                self.assertIn("requires a visible Viewer and user input", message)
                self.assertEqual(self.dialogs, [])
                self.assertEqual(self.saved_settings()["ESM_API_TOKEN"], OLD_TOKEN)
                self.assertEqual(self.given_lines(), ["one,two,three"])

    def test_a_stop_while_the_dialog_is_open_closes_it_and_cancels_the_job(self):
        closed = []

        class Dialog:
            rejected = False

            def reject(self):
                closed.append(True)
                self.rejected = True

        def answer(shown):
            dialog = Dialog()
            shown(dialog)
            # The user types `esmfold stop` in the Viewer while the dialog is open: it runs in this event loop.
            QtCore.QTimer.singleShot(0, lambda: esmfold_command.run(self.make_viewer(state="normal"), ["stop"]))
            deadline = time.monotonic() + 30
            while not dialog.rejected and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.005)
            return None

        self.replies = [answer]
        request_id, _, _ = self.fold_three()

        result = self.finish(request_id)

        self.assertEqual(result["status"], "cancelled", result)
        self.assertEqual(closed, [True], "the dialog was closed, not left open")
        self.assertEqual(self.given_lines(), ["one,two,three"])
        self.wait_for(lambda: self.scheduler.snapshot() == [], "The line never emptied")

    def test_a_typed_fold_asks_too_and_runs_again_with_no_portal_request(self):
        self.replies = [NEW_TOKEN]
        self.run_typed(self.script(), arguments=("large", "multi"), selected=(0, 1, 2), state="normal")

        self.wait_for(lambda: self.given_lines() == ["one,two,three", "two,three"] and not self.scheduler.snapshot(),
                      "The fold never ran again")

        self.assertEqual(self.dialogs, [{"replacement": True, "request_status": None}])
        self.assertEqual(self.saved_settings()["ESM_API_TOKEN"], NEW_TOKEN)
        self.assertEqual(sorted(path.name for path in (self.root / "structures").glob("*.pdb")),
                         ["one.pdb", "three.pdb", "two.pdb"])


if __name__ == "__main__":
    unittest.main()
