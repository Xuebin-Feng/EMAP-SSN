"""Metadata web backend (src/web_ui/meta_backend.py): the node selection and
highlight actions the Metadata UI sends to the Viewer."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web_ui.Plugin_Manager import WebPluginRegistry  # noqa: E402


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
            add_sidebar_button=mock.Mock(),
            open_metadata_ui=mock.Mock(),
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


if __name__ == "__main__":
    unittest.main()
