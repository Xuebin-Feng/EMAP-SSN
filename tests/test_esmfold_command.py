"""Regression tests for local and Biohub-backed ESMFold command routing."""

import contextlib
import io
import json
import os
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

import psutil

import Viewer_Worker_Tracking
from Viewer_Command_Portal import CURRENT, ExecutionContext, bind
from commands import esmfold as esmfold_command
from tests.viewer_fixtures import NO_WINDOW_STATES, WINDOW_STATES, PortalFixture, apply_window_state
from utilities import Hardware_Acceleration as Hardware_Utils
from web_ui import Browser_Page


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class ESMFoldCommandTests(unittest.TestCase):
    def setUp(self):
        # These Viewers are not headless, so the worker must never start without a
        # window; a stand-in that is called fails the test instead of starting a worker.
        patcher = mock.patch.object(
            esmfold_command,
            "launch_without_window",
            side_effect=AssertionError("A windowed esmfold must start its worker in a terminal."),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_viewer(self, count=0):
        headers = [f"node_{index} description" for index in range(count)]
        return SimpleNamespace(
            selected_indices=list(range(count)),
            selected_node_idx=None,
            full_headers=headers,
            sequences_map={header: "ACDE" for header in headers},
            console_text=SimpleNamespace(text=""),
            get_web_url=mock.Mock(
                return_value="http://127.0.0.1:49123/api/action"
            ),
        )

    def remove_worker_input(self, launch):
        if not launch.called:
            return
        command = launch.call_args.args[0]
        try:
            os.unlink(command[2])
        except OSError:
            pass

    def test_bare_command_without_selection_registers_and_opens_ui(self):
        viewer = self.make_viewer()
        with (
            mock.patch.object(esmfold_command.esmfold_backend, "register") as register,
            mock.patch.object(
                esmfold_command.esmfold_backend,
                "open_esmfold_ui",
            ) as open_esmfold_ui,
        ):
            esmfold_command.run(viewer, [])

        register.assert_called_once_with(viewer)
        open_esmfold_ui.assert_called_once_with(viewer)

    def test_keywords_without_selection_report_error(self):
        for arguments in (["multi"], ["large"], ["large", "multi"]):
            with self.subTest(arguments=arguments):
                viewer = self.make_viewer()
                with (
                    mock.patch.object(
                        esmfold_command.esmfold_backend,
                        "register",
                    ) as register,
                    mock.patch.object(
                        esmfold_command.esmfold_backend,
                        "open_esmfold_ui",
                    ) as open_esmfold_ui,
                    mock.patch.object(
                        esmfold_command,
                        "launch_in_terminal",
                    ) as launch,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    esmfold_command.run(viewer, arguments)

                register.assert_not_called()
                open_esmfold_ui.assert_not_called()
                launch.assert_not_called()
                self.assertIn("Error: No nodes selected.", output.getvalue())

    def test_local_single_uses_hardware_and_legacy_worker_arguments(self):
        viewer = self.make_viewer(1)
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(
                    Hardware_Utils,
                    "get_optimal_device",
                    return_value=FakeDevice("cuda"),
                ) as get_device,
                mock.patch.object(
                    esmfold_command.esmfold_backend,
                    "get_structures_directory",
                    return_value=structures_dir,
                ),
                mock.patch.object(
                    esmfold_command,
                    "launch_in_terminal",
                ) as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(
                    esmfold_command.esmfold_backend,
                    "open_esmfold_ui",
                ) as open_esmfold_ui,
            ):
                esmfold_command.run(viewer, [])

            try:
                get_device.assert_called_once_with()
                command = launch.call_args.args[0]
                # The worker owns and deletes the private input file only
                # because the Viewer passes --delete-input.
                self.assertEqual(
                    command[-4:],
                    [
                        "cuda",
                        "--delete-input",
                        "--action-url",
                        "http://127.0.0.1:49123/api/action",
                    ],
                )
                self.assertNotIn("--mode", command)
                with open(command[2], "r", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), [["node_0", "ACDE"]])
                open_esmfold_ui.assert_called_once_with(
                    viewer,
                    show_existing_dialog=False,
                )
            finally:
                self.remove_worker_input(launch)

    def test_large_single_bypasses_hardware_and_selects_remote_mode(self):
        viewer = self.make_viewer(1)
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(
                    Hardware_Utils,
                    "get_optimal_device",
                ) as get_device,
                mock.patch.object(
                    esmfold_command.esmfold_backend,
                    "get_structures_directory",
                    return_value=structures_dir,
                ),
                mock.patch.object(
                    esmfold_command,
                    "launch_in_terminal",
                ) as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(
                    esmfold_command.esmfold_backend,
                    "open_esmfold_ui",
                ) as open_esmfold_ui,
            ):
                esmfold_command.run(viewer, ["large"])

            try:
                get_device.assert_not_called()
                launch.assert_called_once()
                command = launch.call_args.args[0]
                self.assertEqual(
                    command[-5:],
                    [
                        "--delete-input",
                        "--mode",
                        "large",
                        "--action-url",
                        "http://127.0.0.1:49123/api/action",
                    ],
                )
                # The worker's terminal asks for Biohub credentials; the
                # Viewer neither loads nor forwards them.
                self.assertNotIn("ESM_API_TOKEN", " ".join(command))
                self.assertFalse(hasattr(esmfold_command, "Biohub_API"))
                open_esmfold_ui.assert_called_once_with(
                    viewer,
                    show_existing_dialog=False,
                )
            finally:
                self.remove_worker_input(launch)

    def test_large_multi_accepts_both_keyword_orders(self):
        for arguments in (["large", "multi"], ["multi", "large"]):
            with self.subTest(arguments=arguments):
                viewer = self.make_viewer(2)
                with tempfile.TemporaryDirectory() as structures_dir:
                    with (
                        mock.patch.object(
                            esmfold_command.esmfold_backend,
                            "get_structures_directory",
                            return_value=structures_dir,
                        ),
                        mock.patch.object(
                            esmfold_command,
                            "launch_in_terminal",
                        ) as launch,
                        mock.patch.object(esmfold_command.esmfold_backend, "register"),
                        mock.patch.object(
                            esmfold_command.esmfold_backend,
                            "open_esmfold_ui",
                        ) as open_esmfold_ui,
                    ):
                        esmfold_command.run(viewer, arguments)

                    try:
                        command = launch.call_args.args[0]
                        with open(command[2], "r", encoding="utf-8") as handle:
                            records = json.load(handle)
                        self.assertEqual([record[0] for record in records], ["node_0", "node_1"])
                        self.assertEqual(
                            command[-5:],
                            [
                                "--delete-input",
                                "--mode",
                                "large",
                                "--action-url",
                                "http://127.0.0.1:49123/api/action",
                            ],
                        )
                        open_esmfold_ui.assert_called_once_with(
                            viewer,
                            show_existing_dialog=False,
                        )
                    finally:
                        self.remove_worker_input(launch)

    def test_multiple_nodes_without_multi_are_rejected(self):
        viewer = self.make_viewer(2)
        with (
            mock.patch.object(esmfold_command, "launch_in_terminal") as launch,
            redirect_stdout(io.StringIO()) as output,
        ):
            esmfold_command.run(viewer, ["large"])
        launch.assert_not_called()
        self.assertIn("Multiple nodes selected", output.getvalue())

    def test_unknown_and_duplicate_keywords_are_rejected(self):
        for arguments in (["larger"], ["large", "large"], ["multi", "multi"]):
            with self.subTest(arguments=arguments):
                viewer = self.make_viewer(1)
                with (
                    mock.patch.object(esmfold_command, "launch_in_terminal") as launch,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    esmfold_command.run(viewer, arguments)
                launch.assert_not_called()
                self.assertIn("Usage: esmfold [large] [multi]", output.getvalue())

    def test_unavailable_web_server_prevents_worker_launch(self):
        viewer = self.make_viewer(1)
        viewer.get_web_url.side_effect = RuntimeError("bind failed")
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(
                    esmfold_command.esmfold_backend,
                    "get_structures_directory",
                    return_value=structures_dir,
                ),
                mock.patch.object(esmfold_command, "launch_in_terminal") as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.QMessageBox, "critical") as critical,
            ):
                esmfold_command.run(viewer, ["large"])

        launch.assert_not_called()
        critical.assert_called_once()
        self.assertEqual(viewer.console_text.text, "Error: Viewer web server unavailable.")


    # --- What the command reports -------------------------------------------

    def run_reporting(self, viewer, arguments, *, launch=None, opened=True, process=None, output=None):
        """Run the command with the worker and the page stubbed; return (succeeded, failed, launch).

        process is the terminal process the stubbed launch returns, and output
        collects what the command prints.
        """
        engine = esmfold_command.Command_Engine
        launch_options = {"side_effect": launch} if launch else {}
        if process is not None:
            launch_options["return_value"] = process
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory", return_value=structures_dir),
                mock.patch.object(esmfold_command, "launch_in_terminal", **launch_options) as launch_mock,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui", return_value=opened),
                mock.patch.object(engine, "command_succeeded") as succeeded,
                mock.patch.object(engine, "command_failed") as failed,
                mock.patch.object(esmfold_command.QMessageBox, "critical"),
                redirect_stdout(output if output is not None else io.StringIO()),
            ):
                try:
                    esmfold_command.run(viewer, arguments)
                finally:
                    self.remove_worker_input(launch_mock)
        return succeeded, failed, launch_mock

    def test_nodes_without_a_sequence_are_named_in_the_outcome(self):
        viewer = self.make_viewer(3)
        del viewer.sequences_map["node_1 description"]
        succeeded, failed, _ = self.run_reporting(viewer, ["large", "multi"])

        failed.assert_not_called()
        [(_, message)] = [call.args for call in succeeded.call_args_list]
        self.assertEqual(
            message,
            "Started Biohub ESM3 folding for 2 structure(s); waiting for the worker. "
            "Skipped 1 selected node without a sequence. (node_1)",
        )
        self.assertEqual(
            viewer.console_text.text,
            "Spawning separate console to fold 2 structures with Biohub ESM3... "
            "Skipped 1 selected node without a sequence.",
        )

    def test_many_skipped_nodes_are_counted_without_listing_them(self):
        viewer = self.make_viewer(9)
        for index in range(1, 9):
            del viewer.sequences_map[f"node_{index} description"]
        succeeded, _, _ = self.run_reporting(viewer, ["multi", "large"])

        [(_, message)] = [call.args for call in succeeded.call_args_list]
        self.assertTrue(message.endswith("Skipped 8 selected nodes without a sequence."), message)
        self.assertNotIn("node_1", message)

    def test_a_run_that_skips_nothing_reports_as_before(self):
        viewer = self.make_viewer(2)
        succeeded, failed, _ = self.run_reporting(viewer, ["large", "multi"])

        failed.assert_not_called()
        self.assertEqual(
            succeeded.call_args.args[1],
            "Started Biohub ESM3 folding for 2 structure(s); waiting for the worker.",
        )
        self.assertEqual(
            viewer.console_text.text,
            "Spawning separate console to fold 2 structures with Biohub ESM3...",
        )

    def test_an_unopened_page_is_mentioned_after_the_job_started(self):
        viewer = self.make_viewer(1)
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: False)
        succeeded, _, launch = self.run_reporting(viewer, ["large"], opened=False)
        launch.assert_called_once()
        self.assertIn("The structure viewer page was not opened", succeeded.call_args.args[1])

        # A page that is already open needs no remark.
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: client_id == "esmfold")
        succeeded, _, _ = self.run_reporting(viewer, ["large"], opened=False)
        self.assertNotIn("not opened", succeeded.call_args.args[1])

    def test_each_failure_is_reported_once(self):
        def selected(count, sequences=True):
            viewer = self.make_viewer(count)
            if not sequences:
                viewer.sequences_map.clear()
            return viewer

        def refused_launch(command, **options):
            raise OSError("no terminal")

        cases = {
            "no node": (self.make_viewer(), ["multi"], None, "No nodes selected"),
            "several nodes": (selected(2), [], None, "Multiple nodes selected"),
            "no sequence": (selected(1, sequences=False), ["large"], None, "Could not retrieve sequences"),
            "terminal": (selected(1), ["large"], refused_launch, "no terminal"),
        }
        for label, (viewer, arguments, launch, text) in cases.items():
            with self.subTest(label):
                succeeded, failed, _ = self.run_reporting(viewer, arguments, launch=launch)
                succeeded.assert_not_called()
                failed.assert_called_once()
                self.assertIn(text, failed.call_args.args[1])
                # The console line shows its own, shorter sentence.
                self.assertTrue(viewer.console_text.text.startswith("Error:"), viewer.console_text.text)

    # --- Nothing selected: the node last clicked ----------------------------

    CLICKED = "Nothing is selected, so the last clicked node is folded: node_1."

    def clicked_viewer(self):
        viewer = self.make_viewer(2)
        viewer.selected_indices = []
        viewer.selected_node_idx = 1
        return viewer

    def test_the_last_clicked_node_is_folded_and_every_report_says_so(self):
        for arguments in ([], ["large"]):
            with self.subTest(arguments=arguments):
                viewer = self.clicked_viewer()
                output = io.StringIO()
                with mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")):
                    succeeded, failed, launch = self.run_reporting(viewer, arguments, output=output)

                failed.assert_not_called()
                launch.assert_called_once()
                self.assertIn(self.CLICKED, output.getvalue())
                self.assertTrue(succeeded.call_args.args[1].endswith(self.CLICKED), succeeded.call_args.args[1])
                self.assertTrue(viewer.console_text.text.startswith(self.CLICKED), viewer.console_text.text)
                self.assertIn("Spawning separate console to fold 1 structure", viewer.console_text.text)

    def test_the_clicked_node_is_what_the_worker_folds(self):
        viewer = self.clicked_viewer()
        folded = []

        def launch(command, **options):
            with open(command[2], "r", encoding="utf-8") as handle:
                folded.extend(json.load(handle))

        self.run_reporting(viewer, ["large"], launch=launch)
        self.assertEqual(folded, [["node_1", "ACDE"]])

    def test_the_note_names_the_node_by_its_first_word(self):
        viewer = self.clicked_viewer()
        succeeded, _, _ = self.run_reporting(viewer, ["large"])
        self.assertNotIn("description", succeeded.call_args.args[1])

    def test_a_selection_wins_over_the_clicked_node_and_needs_no_note(self):
        viewer = self.make_viewer(2)
        viewer.selected_indices = [0]
        viewer.selected_node_idx = 1
        output = io.StringIO()
        succeeded, _, _ = self.run_reporting(viewer, ["large"], output=output)

        self.assertNotIn("Nothing is selected", output.getvalue())
        self.assertNotIn("Nothing is selected", succeeded.call_args.args[1])
        self.assertNotIn("Nothing is selected", viewer.console_text.text)

    def test_nothing_selected_and_nothing_clicked_still_fails_as_before(self):
        viewer = self.make_viewer(2)
        viewer.selected_indices = []
        succeeded, failed, launch = self.run_reporting(viewer, ["large"])
        launch.assert_not_called()
        succeeded.assert_not_called()
        self.assertIn("No nodes selected", failed.call_args.args[1])

    def test_the_note_is_not_printed_for_a_run_that_is_refused(self):
        viewer = self.clicked_viewer()
        viewer.esmfold_worker = (self.running_process(), None)
        output = io.StringIO()
        self.run_reporting(viewer, ["large"], output=output)
        self.assertNotIn("Nothing is selected", output.getvalue())

    # --- One worker at a time ----------------------------------------------

    REFUSED = (
        "An esmfold worker is still running. Wait for it to finish, or close its "
        "console window, before folding again."
    )

    def running_process(self, running=True):
        process = mock.Mock()
        process.poll.return_value = None if running else 0
        return process

    def test_a_second_run_is_refused_while_the_first_worker_runs(self):
        viewer = self.make_viewer(1)
        first, _, launch = self.run_reporting(viewer, ["large"], process=self.running_process())
        launch.assert_called_once()
        first.assert_called_once()

        output = io.StringIO()
        succeeded, failed, launch = self.run_reporting(viewer, ["large"], output=output)
        launch.assert_not_called()
        succeeded.assert_not_called()
        failed.assert_called_once_with(viewer, f"Error: {self.REFUSED}")
        self.assertIn(f"Error: {self.REFUSED}", output.getvalue())
        self.assertEqual(viewer.console_text.text, f"Error: {self.REFUSED}")

    def test_the_refusal_comes_before_any_hardware_or_sequence_work(self):
        viewer = self.make_viewer(1)
        viewer.esmfold_worker = (self.running_process(), None)
        with mock.patch.object(Hardware_Utils, "get_optimal_device") as get_device:
            _, failed, launch = self.run_reporting(viewer, [])
        get_device.assert_not_called()
        launch.assert_not_called()
        failed.assert_called_once()

    def test_a_finished_worker_no_longer_blocks_a_run(self):
        viewer = self.make_viewer(1)
        self.run_reporting(viewer, ["large"], process=self.running_process(running=False))

        succeeded, failed, launch = self.run_reporting(viewer, ["large"], process=self.running_process())
        launch.assert_called_once()
        failed.assert_not_called()
        succeeded.assert_called_once()

    def test_a_tracked_worker_blocks_a_run_until_its_tracker_stops(self):
        # A terminal that hands its window to another process exits at once,
        # so for a run the MCP or agent page started only the tracker knows.
        viewer = self.make_viewer(1)
        tracker = SimpleNamespace(timer=SimpleNamespace(isActive=mock.Mock(return_value=True)))
        viewer.esmfold_worker = (self.running_process(running=False), tracker)

        _, failed, launch = self.run_reporting(viewer, ["large"])
        launch.assert_not_called()
        failed.assert_called_once()

        tracker.timer.isActive.return_value = False
        succeeded, failed, launch = self.run_reporting(viewer, ["large"])
        launch.assert_called_once()
        failed.assert_not_called()

    def test_a_tracked_worker_that_reported_its_result_is_done_though_its_terminal_stays_open(self):
        # A failed worker waits for Enter in its terminal, which is not computation.
        viewer = self.make_viewer(1)
        tracker = SimpleNamespace(timer=SimpleNamespace(isActive=lambda: False))
        viewer.esmfold_worker = (self.running_process(), tracker)
        self.assertFalse(esmfold_command._worker_running(viewer))

    def test_a_deleted_tracker_leaves_the_judgement_to_the_process(self):
        def deleted():
            raise RuntimeError("Internal C++ object already deleted.")

        viewer = self.make_viewer(1)
        tracker = SimpleNamespace(timer=SimpleNamespace(isActive=deleted))
        viewer.esmfold_worker = (self.running_process(running=False), tracker)
        self.assertFalse(esmfold_command._worker_running(viewer))
        viewer.esmfold_worker = (self.running_process(), tracker)
        self.assertTrue(esmfold_command._worker_running(viewer))

    def test_opening_the_viewer_page_is_not_refused_while_a_worker_runs(self):
        viewer = self.make_viewer()
        viewer.esmfold_worker = (self.running_process(), None)
        succeeded, failed, launch = self.run_reporting(viewer, [])
        launch.assert_not_called()
        failed.assert_not_called()
        self.assertIn("Opened the structure viewer", succeeded.call_args.args[1])

    def test_a_launch_that_failed_leaves_no_worker_to_wait_for(self):
        def refused_launch(command, **options):
            raise OSError("no terminal")

        viewer = self.make_viewer(1)
        self.run_reporting(viewer, ["large"], launch=refused_launch)
        self.assertFalse(hasattr(viewer, "esmfold_worker"))
        self.assertFalse(esmfold_command._worker_running(viewer))

    def test_the_refusal_reaches_the_console_as_a_message(self):
        viewer = self.make_viewer(1)
        viewer.esmfold_worker = (self.running_process(), None)
        shown = []
        with (
            mock.patch.object(esmfold_command.Command_Engine, "show_status", side_effect=lambda v, message: shown.append(message)),
            mock.patch.object(esmfold_command.Command_Engine, "command_failed"),
            redirect_stdout(io.StringIO()),
        ):
            esmfold_command.run(viewer, ["large"])
        [message] = shown
        self.assertEqual(str(message), f"Error: {self.REFUSED}")
        self.assertIsInstance(message.values["error"], esmfold_command.Message)

    def test_the_worker_input_is_removed_when_no_worker_started(self):
        for label, failure in (("terminal unavailable", OSError("no terminal")), ("unexpected", RuntimeError("boom"))):
            with self.subTest(label):
                created = []

                def launch(command, **options):
                    created.append(command[2])
                    self.assertTrue(os.path.exists(command[2]))
                    raise failure

                viewer = self.make_viewer(1)
                if isinstance(failure, RuntimeError):
                    with self.assertRaises(RuntimeError):
                        self.run_reporting(viewer, ["large"], launch=launch)
                else:
                    self.run_reporting(viewer, ["large"], launch=launch)
                [path] = created
                self.assertFalse(os.path.exists(path))

    def test_the_duplicate_keyword_error_reaches_the_console_as_a_message(self):
        viewer = self.make_viewer(1)
        shown = []
        with (
            mock.patch.object(esmfold_command.Command_Engine, "show_status", side_effect=lambda v, message: shown.append(message)),
            redirect_stdout(io.StringIO()),
        ):
            esmfold_command.run(viewer, ["large", "large"])
        [message] = shown
        self.assertEqual(str(message), "Error: Duplicate esmfold keyword: large")
        self.assertIsInstance(message.values["error"], esmfold_command.Message)

    # --- Opening the page --------------------------------------------------

    def open_page(self, opened, connected=False):
        viewer = self.make_viewer()
        viewer.web_server = SimpleNamespace(
            has_event_client=lambda client_id: connected and client_id == "esmfold"
        )
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
        from Viewer_Command_Portal import CURRENT

        open_ui, _, _ = self.open_page(False, connected=True)
        open_ui.assert_called_once_with(mock.ANY)

        token = CURRENT.set(object())
        self.addCleanup(CURRENT.reset, token)
        open_ui, _, _ = self.open_page(False, connected=True)
        open_ui.assert_called_once_with(mock.ANY, show_existing_dialog=False)


ADDRESS = "http://127.0.0.1:49123"
# Where the command module under test says the project is, as it computes it.
COMMAND_SRC = os.path.dirname(os.path.dirname(os.path.abspath(esmfold_command.__file__)))
COMMAND_PROJECT = os.path.dirname(COMMAND_SRC)


def remove_worker_input(launch):
    """Delete the private input file of the worker a stubbed launch was asked to start."""
    if launch.called:
        try:
            os.unlink(launch.call_args.args[0][2])
        except OSError:
            pass


class HeadlessESMFoldTests(unittest.TestCase):
    """In a headless Viewer (no screen) esmfold opens no console window and no browser page.

    The page opens through the Viewer's real opener, so a page that would
    open reaches the webbrowser mock; only the launchers are stubbed.
    """

    def make_viewer(self, count=1, *, headless=True, connected=False):
        headers = [f"node_{index} description" for index in range(count)]
        viewer = SimpleNamespace(
            selected_indices=list(range(count)),
            selected_node_idx=None,
            full_headers=headers,
            sequences_map={header: "ACDE" for header in headers},
            console_text=SimpleNamespace(text=""),
            get_web_url=lambda path="/": f"{ADDRESS}/{str(path).lstrip('/')}",
            web_server=SimpleNamespace(
                has_event_client=lambda client_id: connected and client_id == "esmfold"
            ),
            update_console_background=lambda: None,
        )
        if headless is not None:
            viewer.headless = headless
        viewer._open_web_ui = lambda *args, **kwargs: Browser_Page.open_browser_page(
            viewer, *args, **kwargs
        )
        return viewer

    def run_esmfold(self, viewer, arguments, *, launch=None, tracker=None):
        """Run the command with only the launchers stubbed; return what happened."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        process = mock.Mock()
        process.poll.return_value = None  # A running worker.
        hidden_options = {"side_effect": launch} if launch else {"return_value": process}
        engine = esmfold_command.Command_Engine
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            # The worker folder (status files, logs) is <project>/temp/command_workers.
            enter(mock.patch.object(
                Viewer_Worker_Tracking, "__file__", str(root / "src" / "Viewer_Worker_Tracking.py")
            ))
            enter(mock.patch.object(
                esmfold_command.esmfold_backend, "get_structures_directory",
                return_value=str(root / "structures"),
            ))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
            if "large" not in arguments:
                enter(mock.patch.object(
                    Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")
                ))
            if tracker is not None:
                enter(mock.patch.object(Viewer_Worker_Tracking, "WorkerTracker", return_value=tracker))
            hidden = enter(mock.patch.object(esmfold_command, "launch_without_window", **hidden_options))
            terminal = enter(mock.patch.object(esmfold_command, "launch_in_terminal", return_value=process))
            track = enter(mock.patch.object(Viewer_Worker_Tracking, "track_hidden_worker"))
            browser = enter(mock.patch.object(Browser_Page.webbrowser, "open", return_value=True))
            critical = enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
            information = enter(mock.patch.object(Browser_Page.QMessageBox, "information"))
            succeeded = enter(mock.patch.object(engine, "command_succeeded"))
            failed = enter(mock.patch.object(engine, "command_failed"))
            enter(redirect_stdout(output))
            try:
                esmfold_command.run(viewer, arguments)
            finally:
                remove_worker_input(hidden)
                remove_worker_input(terminal)
        return SimpleNamespace(
            root=root, process=process, output=output.getvalue(), hidden=hidden, terminal=terminal,
            track=track, browser=browser, critical=critical, information=information,
            succeeded=succeeded, failed=failed,
        )

    def outcome(self, run):
        run.failed.assert_not_called()
        [call] = run.succeeded.call_args_list
        return call.args[1]

    def test_the_worker_runs_hidden_with_its_output_in_a_log(self):
        for arguments, tail in (
            ([], ["cuda", "--delete-input", "--action-url", f"{ADDRESS}/api/action", "--noninteractive"]),
            (["large"], ["--delete-input", "--mode", "large", "--action-url", f"{ADDRESS}/api/action", "--noninteractive"]),
        ):
            with self.subTest(arguments=arguments):
                viewer = self.make_viewer()
                run = self.run_esmfold(viewer, arguments)

                run.hidden.assert_called_once()
                command = run.hidden.call_args.args[0]
                options = run.hidden.call_args.kwargs
                self.assertEqual(command[-len(tail):], tail)
                self.assertEqual(os.path.abspath(options["cwd"]), COMMAND_PROJECT)
                # Without a tracker the log is in the worker folder, named for the run.
                log_path = Path(options["log_path"])
                self.assertEqual(log_path.parent, run.root / "temp" / "command_workers")
                self.assertTrue(log_path.name.startswith("esmfold-") and log_path.suffix == ".log")
                # No terminal, no page, no dialog; the Viewer ends the worker when it quits.
                run.terminal.assert_not_called()
                run.browser.assert_not_called()
                run.critical.assert_not_called()
                run.information.assert_not_called()
                run.track.assert_called_once_with(run.process)
                self.assertEqual(viewer.esmfold_worker, (run.process, None))
                self.assertEqual(viewer.console_text.text, "")
                self.assertEqual(
                    self.outcome(run),
                    "Started "
                    + ("Biohub ESM3" if arguments else "local ESM3")
                    + " folding for 1 structure(s); waiting for the worker."
                    + f" The worker has no window; its output is in {log_path}."
                    + f" A Viewer with no window opens no browser page; the structure viewer is served at {ADDRESS}/esmfold.html.",
                )
                self.assertIn(f"without a window; its output goes to {log_path}", run.output)

    def test_the_worker_and_the_page_follow_whether_a_user_is_at_the_viewer(self):
        for state in WINDOW_STATES:
            with self.subTest(state=state):
                viewer = apply_window_state(self.make_viewer(headless=None), state)
                run = self.run_esmfold(viewer, [])
                if state in NO_WINDOW_STATES:
                    run.hidden.assert_called_once()
                    run.terminal.assert_not_called()
                    run.browser.assert_not_called()
                    run.critical.assert_not_called()
                    run.information.assert_not_called()
                    self.assertIn("The worker has no window", self.outcome(run))
                    self.assertIn(f"served at {ADDRESS}/esmfold.html", self.outcome(run))
                else:
                    run.hidden.assert_not_called()
                    run.terminal.assert_called_once()
                    run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new=2)
                    self.assertEqual(
                        self.outcome(run),
                        "Started local ESM3 folding for 1 structure(s); waiting for the worker.",
                    )

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
                    run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new=2)
                    self.assertEqual(self.outcome(run), "Opened the structure viewer; no folding job was requested.")

    def test_a_launch_failure_shows_its_dialog_only_where_a_user_is(self):
        def refused(command, **options):
            raise OSError("no python")

        for state in NO_WINDOW_STATES:
            with self.subTest(state=state):
                run = self.run_esmfold(apply_window_state(self.make_viewer(headless=None), state), ["large"], launch=refused)
                run.critical.assert_not_called()
                run.succeeded.assert_not_called()

    def test_a_connected_structure_viewer_needs_no_address(self):
        viewer = self.make_viewer(connected=True)
        run = self.run_esmfold(viewer, ["large"])

        run.browser.assert_not_called()
        run.information.assert_not_called()
        message = self.outcome(run)
        self.assertIn("The worker has no window; its output is in", message)
        self.assertNotIn("served at", message)

    def test_skipped_nodes_are_still_reported(self):
        viewer = self.make_viewer(2)
        del viewer.sequences_map["node_1 description"]
        run = self.run_esmfold(viewer, ["large", "multi"])

        self.assertTrue(
            self.outcome(run).startswith(
                "Started Biohub ESM3 folding for 1 structure(s); waiting for the worker. "
                "Skipped 1 selected node without a sequence. (node_1) The worker has no window"
            )
        )

    def test_a_tracked_run_keeps_the_status_arguments_and_records_the_log_beside_the_status_file(self):
        root = Path(tempfile.gettempdir()) / "unused"
        tracker = SimpleNamespace(
            path=root / "temp" / "command_workers" / "worker-abc.json",
            timer=SimpleNamespace(isActive=lambda: True),
            attach_hidden=mock.Mock(),
            fail=mock.Mock(),
        )
        token = CURRENT.set(object())
        self.addCleanup(CURRENT.reset, token)
        viewer = self.make_viewer()
        run = self.run_esmfold(viewer, ["large"], tracker=tracker)

        command = run.hidden.call_args.args[0]
        # --portal-status first, then --noninteractive, as in a windowed run.
        self.assertEqual(command[-3:], ["--portal-status", str(tracker.path), "--noninteractive"])
        log_path = root / "temp" / "command_workers" / "worker-abc.log"
        self.assertEqual(Path(run.hidden.call_args.kwargs["log_path"]), log_path)
        tracker.attach_hidden.assert_called_once_with(run.process, str(log_path))
        tracker.fail.assert_not_called()
        self.assertEqual(viewer.esmfold_worker, (run.process, tracker))
        run.browser.assert_not_called()

    def test_no_page_opens_when_nothing_is_selected_either(self):
        viewer = self.make_viewer(0)
        run = self.run_esmfold(viewer, [])

        run.browser.assert_not_called()
        run.hidden.assert_not_called()
        run.terminal.assert_not_called()
        self.assertEqual(
            self.outcome(run),
            f"A Viewer with no window opens no browser page; the structure viewer is served at {ADDRESS}/esmfold.html. "
            "No folding job was requested.",
        )

    def test_an_open_page_is_reported_and_an_unavailable_web_server_fails(self):
        run = self.run_esmfold(self.make_viewer(0, connected=True), [])
        self.assertEqual(
            self.outcome(run), "The structure viewer is already open; no folding job was requested."
        )
        # Not even the dialog that says so: nobody is at a headless Viewer to close it.
        run.information.assert_not_called()

        viewer = self.make_viewer(0)
        viewer.get_web_url = mock.Mock(side_effect=RuntimeError("bind failed"))
        run = self.run_esmfold(viewer, [])
        run.browser.assert_not_called()
        run.succeeded.assert_not_called()
        run.failed.assert_called_once()
        self.assertIn("was not opened", run.failed.call_args.args[1])

    def test_no_dialog_opens_when_the_worker_cannot_start(self):
        def refused(command, **options):
            raise OSError("no python")

        viewer = self.make_viewer()
        run = self.run_esmfold(viewer, ["large"], launch=refused)

        run.succeeded.assert_not_called()
        run.critical.assert_not_called()
        run.track.assert_not_called()
        run.failed.assert_called_once_with(
            viewer, "Error: Could not launch the ESMFold worker without a window:\nno python"
        )
        self.assertFalse(hasattr(viewer, "esmfold_worker"))
        self.assertFalse(os.path.exists(run.hidden.call_args.args[0][2]))

    def test_no_dialog_opens_when_the_web_server_is_unavailable(self):
        viewer = self.make_viewer()
        viewer.get_web_url = mock.Mock(side_effect=RuntimeError("bind failed"))
        run = self.run_esmfold(viewer, ["large"])

        run.hidden.assert_not_called()
        run.critical.assert_not_called()
        run.failed.assert_called_once()
        self.assertIn("web server is unavailable", run.failed.call_args.args[1])

    def test_a_second_run_is_refused_without_naming_a_console_window(self):
        viewer = self.make_viewer()
        worker = mock.Mock()
        worker.poll.return_value = None
        viewer.esmfold_worker = (worker, None)
        run = self.run_esmfold(viewer, ["large"])

        run.hidden.assert_not_called()
        run.succeeded.assert_not_called()
        run.failed.assert_called_once_with(
            viewer, "Error: An esmfold worker is still running. Wait for it to finish before folding again."
        )

    # --- A windowed Viewer is as it was ------------------------------------

    def test_a_windowed_viewer_still_uses_the_terminal_and_opens_the_page(self):
        for headless in (False, None):  # None: a stand-in with no flag at all.
            with self.subTest(headless=headless):
                viewer = self.make_viewer(headless=headless)
                run = self.run_esmfold(viewer, [])

                run.hidden.assert_not_called()
                run.track.assert_not_called()
                run.terminal.assert_called_once()
                command = run.terminal.call_args.args[0]
                self.assertEqual(
                    command,
                    [
                        sys.executable,
                        os.path.join(COMMAND_SRC, "resources", "esmfold", "esmfold_worker.py"),
                        command[2],
                        os.path.abspath(str(run.root / "structures")),
                        "cuda",
                        "--delete-input",
                        "--action-url",
                        f"{ADDRESS}/api/action",
                    ],
                )
                options = run.terminal.call_args.kwargs
                self.assertEqual(os.path.abspath(options["cwd"]), COMMAND_PROJECT)
                self.assertEqual(options["hold"], esmfold_command.HoldMode.NEVER)
                self.assertEqual(options["title"], "SSN ESMFold")
                run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new=2)
                self.assertEqual(
                    self.outcome(run),
                    "Started local ESM3 folding for 1 structure(s); waiting for the worker.",
                )
                self.assertEqual(
                    viewer.console_text.text,
                    "Spawning separate console to fold 1 structure with local ESM3...",
                )
                self.assertIn("in a separate console...", run.output)
                self.assertEqual(viewer.esmfold_worker, (run.process, None))

    def test_a_windowed_viewer_keeps_the_noninteractive_rule_for_tracked_runs(self):
        # --noninteractive follows the environment for a run the portal tracks,
        # as before the headless flag existed; an untracked one never gets it.
        for environment, expected in (
            ({"SSN_VIEWER_HEADLESS": "1", "QT_QPA_PLATFORM": ""}, ["--portal-status", "status.json", "--noninteractive"]),
            ({"SSN_VIEWER_HEADLESS": "0", "QT_QPA_PLATFORM": "offscreen"}, ["--portal-status", "status.json", "--noninteractive"]),
            ({"SSN_VIEWER_HEADLESS": "0", "QT_QPA_PLATFORM": ""}, ["--portal-status", "status.json"]),
        ):
            with self.subTest(environment=environment):
                tracker = SimpleNamespace(
                    path="status.json", timer=SimpleNamespace(isActive=lambda: True),
                    attach_hidden=mock.Mock(), fail=mock.Mock(),
                )
                token = CURRENT.set(object())
                self.addCleanup(CURRENT.reset, token)
                with mock.patch.dict(os.environ, environment):
                    run = self.run_esmfold(self.make_viewer(headless=False), ["large"], tracker=tracker)

                run.hidden.assert_not_called()
                command = run.terminal.call_args.args[0]
                self.assertEqual(command[-len(expected):], expected)
                tracker.attach_hidden.assert_not_called()
                self.assertEqual(self.outcome(run), "Started Biohub ESM3 folding for 1 structure(s); waiting for the worker.")

    def test_a_windowed_viewer_still_opens_the_page_when_nothing_is_selected(self):
        run = self.run_esmfold(self.make_viewer(0, headless=False), [])

        run.browser.assert_called_once_with(f"{ADDRESS}/esmfold.html", new=2)
        run.hidden.assert_not_called()
        self.assertEqual(
            self.outcome(run), "Opened the structure viewer; no folding job was requested."
        )

    def test_a_windowed_launch_failure_still_shows_the_dialog(self):
        def refused(command, **options):
            raise OSError("no terminal")

        viewer = self.make_viewer(headless=False)
        engine = esmfold_command.Command_Engine
        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            root = Path(enter(tempfile.TemporaryDirectory()))
            enter(mock.patch.object(
                esmfold_command.esmfold_backend, "get_structures_directory",
                return_value=str(root),
            ))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
            enter(mock.patch.object(Hardware_Utils, "get_optimal_device", return_value=FakeDevice("cuda")))
            enter(mock.patch.object(esmfold_command, "launch_in_terminal", side_effect=refused))
            enter(mock.patch.object(
                esmfold_command, "launch_without_window",
                side_effect=AssertionError("A windowed esmfold must start its worker in a terminal."),
            ))
            enter(mock.patch.object(engine, "command_failed"))
            critical = enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
            enter(redirect_stdout(io.StringIO()))
            esmfold_command.run(viewer, [])

        critical.assert_called_once()


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


class HeadlessESMFoldProcessTests(PortalFixture, unittest.TestCase):
    """The headless worker as real processes: the real launcher, the real worker's
    status reporting, the real tracker and a real process tree."""

    def setUp(self):
        super().setUp()
        self.root = Path(self.directory.name)
        for name, value in (("_hidden_workers", set()), ("_hidden_quit_hooked", True)):
            patcher = mock.patch.object(Viewer_Worker_Tracking, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(
            Viewer_Worker_Tracking, "__file__", str(self.root / "src" / "Viewer_Worker_Tracking.py")
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.end_workers)

    def end_workers(self):
        Viewer_Worker_Tracking.stop_hidden_workers()

    def make_viewer(self):
        viewer = self.viewer
        viewer.headless = True
        viewer.selected_indices = [0]
        viewer.selected_node_idx = None
        viewer.sequences_map = {"one": "ACDE"}
        viewer.get_web_url = lambda path="/": f"{ADDRESS}/{str(path).lstrip('/')}"
        viewer.web_server = SimpleNamespace(has_event_client=lambda client_id: False)
        viewer._open_web_ui = lambda *args, **kwargs: Browser_Page.open_browser_page(
            viewer, *args, **kwargs
        )
        return viewer

    def run_in_portal(self, script_text):
        """Run `esmfold large` as a portal command whose worker is script_text; return (request id, context, mocks)."""
        script = self.root / "stand_in.py"
        script.write_text(script_text, encoding="utf-8")
        request_id = self.portal.submit("headless-fold", "test")["request_id"]
        record = self.portal.requests[request_id]["commands"][0]
        record.update(dispatched=True, outcome="succeeded", status="running")
        context = ExecutionContext(self.portal, request_id, record)
        real_launch = esmfold_command.launch_without_window

        def launch(command, **options):
            # [python, worker script, input, structures, ...]: the stand-in takes the worker's
            # place. The real worker deletes its private input (--delete-input); a stand-in
            # that fails or hangs before reading it leaves the file for the test to remove.
            self.addCleanup(lambda path=command[2]: os.path.exists(path) and os.unlink(path))
            return real_launch([command[0], str(script), SRC_DIR, *command[2:]], **options)

        with contextlib.ExitStack() as stack:
            enter = stack.enter_context
            enter(mock.patch.object(
                esmfold_command.esmfold_backend, "get_structures_directory",
                return_value=str(self.root / "structures"),
            ))
            enter(mock.patch.object(esmfold_command.esmfold_backend, "register"))
            enter(mock.patch.object(esmfold_command, "launch_without_window", side_effect=launch))
            terminal = enter(mock.patch.object(esmfold_command, "launch_in_terminal"))
            browser = enter(mock.patch.object(Browser_Page.webbrowser, "open"))
            critical = enter(mock.patch.object(esmfold_command.QMessageBox, "critical"))
            enter(redirect_stdout(io.StringIO()))
            with bind(context):
                esmfold_command.run(self.make_viewer(), ["large"])
        self.assertIsNotNone(getattr(self.viewer, "esmfold_worker", None))
        self.addCleanup(self.viewer.esmfold_worker[1].timer.stop)
        return request_id, context, SimpleNamespace(terminal=terminal, browser=browser, critical=critical)

    def logs(self):
        return sorted((self.root / "temp" / "command_workers").glob("*.log"))

    def test_a_hidden_fold_reports_through_the_portal_and_writes_its_log(self):
        request_id, context, mocks = self.run_in_portal(STAND_IN_FOLDING)

        result = self.finish(request_id)

        self.assertEqual(result["status"], "succeeded", result)
        [job] = result["commands"][0]["jobs"]
        self.assertEqual((job["status"], job["message"]), ("succeeded", "Structure prediction completed"))
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
        self.assertEqual(log.stem, os.path.splitext(os.path.basename(job_status_path(self.root)))[0])
        # The outcome tells where the log is and where the page is; nothing opened.
        messages = [entry["text"] for entry in context.record["messages"]]
        self.assertTrue(any(f"its output is in {log}" in text and f"{ADDRESS}/esmfold.html" in text for text in messages), messages)
        mocks.terminal.assert_not_called()
        mocks.browser.assert_not_called()
        mocks.critical.assert_not_called()

    def test_a_worker_that_fails_to_start_is_reported_with_its_traceback_in_the_log(self):
        request_id, context, _ = self.run_in_portal("raise ImportError('no torch here')\n")

        result = self.finish(request_id)

        self.assertEqual(result["status"], "failed", result)
        [log] = self.logs()
        [job] = result["commands"][0]["jobs"]
        self.assertEqual(
            job["message"], f"Worker exited with code 1 before reporting startup; its output is in {log}"
        )
        self.assertIn("ImportError: no torch here", log.read_text(encoding="utf-8"))

    def test_closing_the_viewer_ends_the_hidden_worker_and_what_it_started(self):
        pids = self.root / "pids.txt"
        request_id, context, _ = self.run_in_portal(STAND_IN_HANGING.format(pids=str(pids)))
        deadline = time.monotonic() + 60
        while len(pids.read_text().split() if pids.exists() else []) != 2:
            self.assertLess(time.monotonic(), deadline, "The stand-in worker never started its child")
            time.sleep(0.05)
        processes = [psutil.Process(int(field)) for field in pids.read_text().split()]
        self.addCleanup(lambda: [p.kill() for p in processes if p.is_running()])
        process = self.viewer.esmfold_worker[0]
        self.assertIsNone(process.poll())

        # What the application's aboutToQuit signal runs.
        self.assertEqual(Viewer_Worker_Tracking.stop_hidden_workers(), 1)

        _, alive = psutil.wait_procs(processes, timeout=10)
        self.assertEqual(alive, [])
        self.assertIsNotNone(process.poll())


def job_status_path(root):
    """The one worker status file under root's worker folder."""
    [path] = (Path(root) / "temp" / "command_workers").glob("worker-*.json")
    return str(path)


if __name__ == "__main__":
    unittest.main()
