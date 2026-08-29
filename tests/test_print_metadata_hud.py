import importlib.util
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def load_print_command():
    command_engine = types.ModuleType("Command_Engine")
    config = types.ModuleType("SSN_Config")
    config.ANALYSIS_RESULT_DIR = "Analysis_Results"
    config.SEQUENCE_SET = "test_sequences"
    config.resolve_directory_path = lambda value: value

    utilities = types.ModuleType("utilities")
    utilities.__path__ = []
    application_windows = types.ModuleType("utilities.Application_Windows")
    application_windows.open_in_file_manager = mock.Mock()

    spec = importlib.util.spec_from_file_location(
        "print_command_under_test", os.path.join(SRC_DIR, "commands", "print.py")
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "Command_Engine": command_engine,
            "SSN_Config": config,
            "utilities": utilities,
            "utilities.Application_Windows": application_windows,
        },
    ):
        spec.loader.exec_module(module)
    return module


print_command = load_print_command()


class PrintMetadataHUDTests(unittest.TestCase):
    def test_metadata_hud_is_hidden_during_capture_then_restored(self):
        metadata_visual = SimpleNamespace(visible=True)
        viewer = SimpleNamespace(
            canvas=SimpleNamespace(
                bgcolor="white",
                update=mock.Mock(),
            ),
            console_bg=SimpleNamespace(visible=True),
            console_text=SimpleNamespace(text="previous message"),
            hidden_text=SimpleNamespace(visible=True),
            hud_displays={
                "meta_display": SimpleNamespace(text_visual=metadata_visual)
            },
            instr_text=SimpleNamespace(visible=True),
            tooltip=SimpleNamespace(visible=True),
            view=SimpleNamespace(
                camera=SimpleNamespace(rect="original rect", aspect=1.0)
            ),
            zoom_text=SimpleNamespace(visible=True),
        )

        def capture_while_hud_is_hidden(_viewer, _is_transparent):
            self.assertFalse(metadata_visual.visible)
            return np.zeros((2, 2, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            print_command, "PRINT_DIRECTORY", temp_dir
        ), mock.patch.object(
            print_command, "_capture_tile", side_effect=capture_while_hud_is_hidden
        ), mock.patch.object(
            print_command.mpimg, "imsave"
        ), mock.patch.object(
            print_command, "open_in_file_manager"
        ), mock.patch.object(
            print_command.app, "process_events"
        ):
            print_command.run(viewer, ["metadata_hud"])

        self.assertTrue(metadata_visual.visible)


if __name__ == "__main__":
    unittest.main()
