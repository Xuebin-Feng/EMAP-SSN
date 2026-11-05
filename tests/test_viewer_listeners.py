# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's two generic hooks for pages that follow it (EMAPSSN_Viewer.MainViewer).

Whatever changes the nodes' colours or visibility, in whatever command, ends by drawing the nodes again, so a
page that follows them adds a listener to ``node_update_listeners`` and is called each time. Loading an alignment
says so to every page, with an ``alignment_changed`` event. Neither hook names a command or a page.
"""

import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from EMAPSSN_Viewer import MainViewer  # noqa: E402


class FakeMarkers:
    def __init__(self):
        self.visible = True
        self.data = None

    def set_data(self, **kwargs):
        self.data = kwargs

    def set_gl_state(self, *args, **kwargs):
        pass


def renderable_viewer(node_count=3):
    """A MainViewer with just what update_nodes reads, drawing into a marker visual that keeps what it is given."""
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = node_count
    viewer.node_render_order = np.arange(node_count, dtype=np.int32)
    viewer.visible_mask = np.ones(node_count, dtype=bool)
    viewer.selected_indices = []
    viewer.selected_node_idx = None
    viewer.left_click_highlight_indices = None
    viewer.edges = np.empty((0, 2), dtype=np.int32)
    viewer.pos = np.column_stack((np.arange(node_count), np.zeros(node_count))).astype(np.float32)
    viewer.current_colors = np.zeros((node_count, 4), dtype=np.float32)
    viewer.current_sizes = np.ones(node_count, dtype=np.float32)
    viewer.current_shapes = np.full(node_count, "disc", dtype=object)
    viewer.markers = FakeMarkers()
    viewer.canvas = SimpleNamespace(update=mock.Mock())
    viewer._update_hud_elements = mock.Mock()
    return viewer


class NodeUpdateListenerTests(unittest.TestCase):
    def test_a_listener_is_called_each_time_the_nodes_are_drawn(self):
        viewer = renderable_viewer()
        calls = []
        viewer.node_update_listeners = [lambda: calls.append("first")]
        viewer.update_nodes()
        viewer.update_nodes()
        self.assertEqual(calls, ["first", "first"])

    def test_listeners_are_called_in_the_order_they_were_added_and_after_the_drawing(self):
        viewer = renderable_viewer()
        seen = []
        viewer.node_update_listeners = [
            lambda: seen.append(("one", viewer.markers.data is not None, viewer.canvas.update.call_count)),
            lambda: seen.append(("two", viewer.markers.data is not None, viewer.canvas.update.call_count)),
        ]
        viewer.update_nodes()
        self.assertEqual(seen, [("one", True, 1), ("two", True, 1)])

    def test_a_listener_sees_the_colours_the_nodes_were_drawn_in(self):
        viewer = renderable_viewer()
        viewer.current_colors[1] = (1.0, 0.0, 0.0, 1.0)
        found = []
        viewer.node_update_listeners = [lambda: found.append(viewer.current_colors.copy())]
        viewer.update_nodes()
        np.testing.assert_array_equal(found[0][1], (1.0, 0.0, 0.0, 1.0))

    def test_a_listener_that_fails_stops_neither_the_others_nor_the_drawing(self):
        viewer = renderable_viewer()
        calls = []

        def broken():
            raise RuntimeError("nobody was listening")

        viewer.node_update_listeners = [broken, lambda: calls.append("after")]
        with mock.patch("builtins.print") as printed:
            viewer.update_nodes()
        self.assertEqual(calls, ["after"])
        self.assertIsNotNone(viewer.markers.data)
        self.assertIn("nobody was listening", " ".join(str(call.args[0]) for call in printed.call_args_list))

    def test_a_viewer_with_no_listeners_draws_as_it_did(self):
        viewer = renderable_viewer()
        viewer.update_nodes()
        self.assertIsNotNone(viewer.markers.data)
        viewer.node_update_listeners = []
        viewer.update_nodes()

    def test_a_listener_added_while_the_nodes_are_drawn_is_called_next_time(self):
        viewer = renderable_viewer()
        calls = []
        viewer.node_update_listeners = [lambda: viewer.node_update_listeners.append(lambda: calls.append("late"))]
        viewer.update_nodes()
        self.assertEqual(calls, [])
        viewer.node_update_listeners.pop(0)
        viewer.update_nodes()
        self.assertEqual(calls, ["late"])

    def test_a_new_viewer_starts_with_an_empty_list(self):
        source = pathlib.Path(SRC_DIR, "EMAPSSN_Viewer.py").read_text(encoding="utf-8")
        self.assertIn("self.node_update_listeners = []", source)


class AlignmentChangedTests(unittest.TestCase):
    def viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers = ["a", "b"]
        viewer.active_reference = None
        viewer.alignment_offset = 0
        viewer.broadcast_event = mock.Mock()
        return viewer

    def test_loading_the_alignment_tells_every_page(self):
        viewer = self.viewer()
        with mock.patch("Alignment_Manager.Alignment_Manager", return_value="the alignment") as manager:
            viewer.load_global_alignment()
        manager.assert_called_once()
        self.assertEqual(viewer.alignment, "the alignment")
        viewer.broadcast_event.assert_called_once_with({"type": "alignment_changed"})

    def test_the_event_comes_after_the_new_alignment_is_in_place(self):
        viewer = self.viewer()
        held = []
        viewer.broadcast_event = lambda event: held.append(viewer.alignment)
        with mock.patch("Alignment_Manager.Alignment_Manager", return_value="the alignment"):
            viewer.load_global_alignment()
        self.assertEqual(held, ["the alignment"])

    def test_a_viewer_with_no_web_server_loads_the_alignment_all_the_same(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.full_headers, viewer.active_reference, viewer.alignment_offset = ["a"], None, 0
        with mock.patch("Alignment_Manager.Alignment_Manager", return_value="the alignment"):
            viewer.load_global_alignment()
        self.assertEqual(viewer.alignment, "the alignment")

    def test_an_object_that_cannot_broadcast_loads_the_alignment_all_the_same(self):
        # What the alignment command's tests bind the loader to: the load is not a failure for want of a web UI.
        viewer = SimpleNamespace(full_headers=["a"], active_reference=None, alignment_offset=0)
        with mock.patch("Alignment_Manager.Alignment_Manager", return_value="the alignment"):
            MainViewer.load_global_alignment(viewer)
        self.assertEqual(viewer.alignment, "the alignment")

    def test_a_reference_change_that_keeps_the_rows_says_so_too(self):
        viewer = self.viewer()
        viewer.alignment = "the old alignment"
        with mock.patch("Alignment_Manager.Alignment_Manager", return_value="the new numbering"):
            viewer.load_global_alignment(reuse_loaded=True)
        viewer.broadcast_event.assert_called_once_with({"type": "alignment_changed"})


if __name__ == "__main__":
    unittest.main()
