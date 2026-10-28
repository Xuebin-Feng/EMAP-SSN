"""Metadata web backend (src/web_ui/meta_backend.py): the node selection and
highlight actions the Metadata UI sends to the Viewer, and the format its
export dialog saves in."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_ui.Plugin_Manager import WebPluginRegistry  # noqa: E402
from tests.viewer_fixtures import NO_WINDOW_STATES, WINDOW_STATES, apply_window_state  # noqa: E402


class MetadataHighlightActionTests(unittest.TestCase):
    def make_viewer(self):
        return type(
            "MetadataViewer",
            (),
            {
                "n_nodes": 3,
                "visible_mask": np.array([True, True, False], dtype=bool),
                "apply_left_click_focus": mock.Mock(),
                "clear_left_click_focus": mock.Mock(),
            },
        )()

    def test_select_accepts_only_visible_integer_node_indices(self):
        from web_ui import meta_backend

        viewer = self.make_viewer()
        self.assertTrue(meta_backend.handle_select_node(viewer, {"index": 1}))
        viewer.apply_left_click_focus.assert_called_once_with(1)

        for invalid in (True, 1.0, "1", -1, 2, 3, None):
            with self.subTest(index=invalid):
                self.assertFalse(
                    meta_backend.handle_select_node(viewer, {"index": invalid})
                )
        viewer.apply_left_click_focus.assert_called_once_with(1)

    def test_registered_clear_action_only_delegates_click_focus_clear(self):
        from web_ui import meta_backend

        viewer = self.make_viewer()
        registry = WebPluginRegistry(viewer)
        meta_backend.register_backend(registry, viewer)

        self.assertIn("select", registry.actions)
        self.assertIn("clear_selection", registry.actions)
        self.assertTrue(registry.actions["clear_selection"]({}))
        viewer.clear_left_click_focus.assert_called_once_with()

    def test_activation_wrapper_does_not_duplicate_viewer_row_broadcast(self):
        from web_ui import meta_backend

        viewer = SimpleNamespace(
            left_click_highlight_indices=[1],
            broadcast_event=mock.Mock(),
            add_console_button=mock.Mock(),
            open_page=mock.Mock(),
            sidebar_buttons_to_persist=[],
        )
        original_mouse_press = mock.Mock(
            side_effect=lambda _event: viewer.broadcast_event(
                {"type": "highlight_row", "index": 1}
            )
        )
        viewer.on_mouse_press = original_mouse_press

        meta_backend.activate(viewer)
        viewer.on_mouse_press(SimpleNamespace(button=1, modifiers=[]))

        original_mouse_press.assert_called_once()
        viewer.broadcast_event.assert_called_once_with(
            {"type": "highlight_row", "index": 1}
        )
        self.assertIsNone(viewer.left_click_highlight_indices)


class MetadataConsoleButtonTests(unittest.TestCase):
    """The metadata page declares its own button for the Console's button row, and the command that persists."""

    def test_activation_registers_the_button_with_its_icon_and_page(self):
        from web_ui import meta_backend

        viewer = SimpleNamespace(add_console_button=mock.Mock(), open_page=mock.Mock(), sidebar_buttons_to_persist=[])
        meta_backend.activate(viewer)
        viewer.add_console_button.assert_called_once()
        arguments = viewer.add_console_button.call_args.kwargs
        self.assertEqual(arguments["name"], "metaDataBtn")
        self.assertEqual(arguments["label"], "Meta Data")
        self.assertEqual(arguments["icon"], "chart-column")
        self.assertEqual(arguments["tooltip"], "Open Metadata Spreadsheet in browser")
        # A click opens the page its manifest declares, with no argument.
        arguments["callback"]()
        viewer.open_page.assert_called_once_with("meta")

    def test_the_command_to_register_it_again_is_persisted_once_however_often_it_is_activated(self):
        from web_ui import meta_backend

        viewer = SimpleNamespace(add_console_button=mock.Mock(), open_page=mock.Mock())
        for _ in range(3):
            meta_backend.activate(viewer)
        self.assertEqual(viewer.sidebar_buttons_to_persist, ["meta"])  # the attribute is made when it is missing
        self.assertEqual(viewer.add_console_button.call_count, 3)  # the Viewer's registry takes a repeat as no change

    def test_a_viewer_with_no_console_button_registration_is_left_alone(self):
        from web_ui import meta_backend

        viewer = SimpleNamespace(open_page=mock.Mock())
        meta_backend.activate(viewer)
        self.assertFalse(hasattr(viewer, "sidebar_buttons_to_persist"))
        self.assertFalse(hasattr(viewer, "add_sidebar_button"))

    def test_the_backend_no_longer_reaches_for_the_sidebar(self):
        source = (SRC_DIR / "web_ui" / "meta_backend.py").read_text(encoding="utf-8")
        self.assertNotIn("add_sidebar_button", source)


class MetadataExportFormatTests(unittest.TestCase):
    """The export dialog's chosen filter gives the file its format by its
    pattern, which every language keeps, not by its name, which a language
    translates."""

    def export(self, chosen_file, chosen_filter):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog, QFileDialog
        from web_ui import meta_backend

        class ChosenFileDialog:
            """Stands in for the modal QFileDialog, with a file and a filter chosen."""

            FileMode, AcceptMode = QFileDialog.FileMode, QFileDialog.AcceptMode

            def __init__(self, parent):
                pass

            def windowFlags(self):
                return Qt.WindowType.Dialog

            def exec(self):
                return QDialog.DialogCode.Accepted

            def selectedFiles(self):
                return [chosen_file]

            def selectedNameFilter(self):
                return chosen_filter

            def __getattr__(self, name):
                return lambda *args: None

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        with mock.patch.object(meta_backend.QtWidgets, "QFileDialog", ChosenFileDialog), \
                mock.patch.object(meta_backend.cfg, "METADATA_DIR", folder.name, create=True), \
                mock.patch.object(meta_backend, "download_metadata") as download:
            meta_backend.handle_export_metadata(SimpleNamespace(main_window=mock.Mock()), {})
        download.assert_called_once()
        return download.call_args.args[1]

    def test_the_filters_pattern_gives_the_extension(self):
        for chosen_file, chosen_filter, saved in (
            ("table", "CSV Files (*.csv)", "table.csv"),
            ("table", "Excel Files (*.xlsx)", "table.xlsx"),
            # Filters as a language may name them.
            ("table", "Valeurs séparées (*.csv)", "table.csv"),
            ("table", "Classeur (*.xlsx)", "table.xlsx"),
            ("table.xls", "Classeur (*.xlsx)", "table.xls"),
            ("table.csv", "Valeurs séparées (*.csv)", "table.csv"),
        ):
            with self.subTest(chosen_filter=chosen_filter, chosen_file=chosen_file):
                self.assertEqual(self.export(chosen_file, chosen_filter), saved)


class MetadataFileDialogWindowStateTests(unittest.TestCase):
    """The page's import and export open a modal file dialog: only where a user can answer it.

    Elsewhere the action fails with a message instead of blocking the Qt thread for good.
    """

    def run_handler(self, handler, state):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QDialog, QFileDialog
        from web_ui import meta_backend

        opened = []

        class RejectingFileDialog:
            FileMode, AcceptMode = QFileDialog.FileMode, QFileDialog.AcceptMode

            def __init__(self, parent):
                opened.append(parent)

            def windowFlags(self):
                return Qt.WindowType.Dialog

            def exec(self):
                return QDialog.DialogCode.Rejected

            def __getattr__(self, name):
                return lambda *args: None

        viewer = apply_window_state(SimpleNamespace(main_window=mock.Mock()), state)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        engine = meta_backend.Command_Engine
        with mock.patch.object(meta_backend.QtWidgets, "QFileDialog", RejectingFileDialog), \
                mock.patch.object(meta_backend.cfg, "METADATA_DIR", folder.name, create=True), \
                mock.patch.object(engine, "command_failed") as failed, \
                mock.patch("builtins.print"):
            handler(viewer, {})
        return opened, failed

    def test_import_and_export_need_a_user_at_the_viewer(self):
        from web_ui import meta_backend
        for handler, noun in ((meta_backend.handle_import_metadata, "Importing"),
                              (meta_backend.handle_export_metadata, "Exporting")):
            for state in WINDOW_STATES:
                with self.subTest(handler=handler.__name__, state=state):
                    opened, failed = self.run_handler(handler, state)
                    if state in NO_WINDOW_STATES:
                        self.assertEqual(opened, [])
                        failed.assert_called_once()
                        self.assertEqual(
                            str(failed.call_args.args[1]),
                            f"{noun} metadata opens a file dialog, which a Viewer that shows no windows cannot offer.",
                        )
                    else:
                        self.assertEqual(len(opened), 1)
                        failed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
