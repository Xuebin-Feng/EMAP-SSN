import io
import importlib.util
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESET_PATH = os.path.join(SRC_DIR, "commands", "reset.py")

command_engine_stub = types.ModuleType("Command_Engine")
command_engine_stub.execute_reset = mock.Mock()
spec = importlib.util.spec_from_file_location("reset_command_under_test", RESET_PATH)
reset_command = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {"Command_Engine": command_engine_stub}):
    spec.loader.exec_module(reset_command)


class ResetCommandHelpTests(unittest.TestCase):
    def test_help_matches_terminal_help_convention_without_resetting(self):
        viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
        output = io.StringIO()

        with redirect_stdout(output), mock.patch.object(
            reset_command.Command_Engine, "execute_reset"
        ) as execute_reset:
            reset_command.run(viewer, ["help"])

        execute_reset.assert_not_called()
        self.assertEqual(
            viewer.console_text.text,
            "Help information printed to the terminal",
        )
        help_text = output.getvalue()
        self.assertIn("Network Reset Tool", help_text)
        self.assertIn("reset <TARGET_1> [TARGET_2]", help_text)
        self.assertIn("order, layer", help_text)
        self.assertIn("one undoable action", help_text)

    def test_all_supported_help_flags_use_the_help_path(self):
        for flag in ("help", "-h", "--help"):
            with self.subTest(flag=flag):
                viewer = SimpleNamespace(console_text=SimpleNamespace(text=""))
                with mock.patch.object(reset_command, "print_help") as print_help, \
                     mock.patch.object(
                         reset_command.Command_Engine, "execute_reset"
                     ) as execute_reset:
                    reset_command.run(viewer, [flag])

                print_help.assert_called_once_with()
                execute_reset.assert_not_called()
                self.assertEqual(
                    viewer.console_text.text,
                    "Help information printed to the terminal",
                )

    def test_non_help_arguments_still_delegate_to_reset_engine(self):
        viewer = SimpleNamespace()

        with mock.patch.object(
            reset_command.Command_Engine, "execute_reset"
        ) as execute_reset:
            reset_command.run(viewer, ["colors", "sizes"])

        execute_reset.assert_called_once_with(viewer, ["colors", "sizes"])


if __name__ == "__main__":
    unittest.main()
