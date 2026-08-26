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
from utilities import Hardware_Utils


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
                    esmfold_command.cfg,
                    "STRUCTURES_DIR",
                    structures_dir,
                    create=True,
                ),
                mock.patch.object(
                    esmfold_command,
                    "launch_in_terminal",
                ) as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui"),
            ):
                esmfold_command.run(viewer, [])

            try:
                get_device.assert_called_once_with()
                command = launch.call_args.args[0]
                self.assertEqual(command[-1], "cuda")
                self.assertNotIn("--mode", command)
                with open(command[2], "r", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), [["node_0", "ACDE"]])
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
                    esmfold_command.cfg,
                    "STRUCTURES_DIR",
                    structures_dir,
                    create=True,
                ),
                mock.patch.object(
                    esmfold_command,
                    "launch_in_terminal",
                ) as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui"),
            ):
                esmfold_command.run(viewer, ["large"])

            try:
                get_device.assert_not_called()
                command = launch.call_args.args[0]
                self.assertEqual(command[-2:], ["--mode", "large"])
                self.assertNotIn("ESM_API_TOKEN", " ".join(command))
            finally:
                self.remove_worker_input(launch)

    def test_large_multi_accepts_both_keyword_orders(self):
        for arguments in (["large", "multi"], ["multi", "large"]):
            with self.subTest(arguments=arguments):
                viewer = self.make_viewer(2)
                with tempfile.TemporaryDirectory() as structures_dir:
                    with (
                        mock.patch.object(
                            esmfold_command.cfg,
                            "STRUCTURES_DIR",
                            structures_dir,
                            create=True,
                        ),
                        mock.patch.object(
                            esmfold_command,
                            "launch_in_terminal",
                        ) as launch,
                        mock.patch.object(esmfold_command.esmfold_backend, "register"),
                        mock.patch.object(
                            esmfold_command.esmfold_backend,
                            "open_esmfold_ui",
                        ),
                    ):
                        esmfold_command.run(viewer, arguments)

                    try:
                        command = launch.call_args.args[0]
                        with open(command[2], "r", encoding="utf-8") as handle:
                            records = json.load(handle)
                        self.assertEqual([record[0] for record in records], ["node_0", "node_1"])
                        self.assertEqual(command[-2:], ["--mode", "large"])
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

    def test_large_command_defers_credentials_to_worker_terminal(self):
        viewer = self.make_viewer(1)
        with tempfile.TemporaryDirectory() as structures_dir:
            with (
                mock.patch.object(
                    esmfold_command.cfg,
                    "STRUCTURES_DIR",
                    structures_dir,
                    create=True,
                ),
                mock.patch.object(esmfold_command, "launch_in_terminal") as launch,
                mock.patch.object(esmfold_command.esmfold_backend, "register"),
                mock.patch.object(esmfold_command.esmfold_backend, "open_esmfold_ui"),
            ):
                esmfold_command.run(viewer, ["large"])

            try:
                launch.assert_called_once()
                self.assertFalse(hasattr(esmfold_command, "Biohub_API"))
                self.assertEqual(launch.call_args.args[0][-2:], ["--mode", "large"])
            finally:
                self.remove_worker_input(launch)


if __name__ == "__main__":
    unittest.main()
