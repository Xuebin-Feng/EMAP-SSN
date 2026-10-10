"""Regression tests for local and Biohub-backed ESMFold command routing."""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from commands import esmfold as esmfold_command
from utilities import Hardware_Acceleration as Hardware_Utils


class FakeDevice:
    def __init__(self, device_type):
        self.type = device_type

    def __str__(self):
        return self.type


class ESMFoldCommandTests(unittest.TestCase):
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

    def run_reporting(self, viewer, arguments, *, launch=None, opened=True):
        """Run the command with the worker and the page stubbed; return (succeeded, failed, launch)."""
        engine = esmfold_command.Command_Engine
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(esmfold_command.esmfold_backend, "get_structures_directory", return_value=structures_dir),
                mock.patch.object(esmfold_command, "launch_in_terminal", **({"side_effect": launch} if launch else {})) as launch_mock,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui", return_value=opened),
                mock.patch.object(engine, "command_succeeded") as succeeded,
                mock.patch.object(engine, "command_failed") as failed,
                mock.patch.object(esmfold_command.QMessageBox, "critical"),
                redirect_stdout(io.StringIO()),
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


if __name__ == "__main__":
    unittest.main()
