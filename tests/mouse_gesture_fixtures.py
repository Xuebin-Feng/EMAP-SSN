# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Viewer mouse and keyboard gesture tests.

Provides synthetic VisPy event helpers, real VisPy Markers with identity pixel
transforms, RecordingEdges, GestureTestCase, ViewTransform, and ZoomTimer without
importing from any test_* module.
"""
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6 import QtWidgets  # noqa: E402
from vispy import app as vispy_app, scene  # noqa: E402

import EMAPSSN_Config as cfg  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from tests.timer_fixtures import FakeTimer  # noqa: E402

# The nodes lie 40 px apart on y = 0, and a click on a node's own place picks it.
NODE = [(40.0 * index, 0.0) for index in range(6)]
SHIFT = ["Shift"]
CTRL = ["Control"]


def distinct_outlines():
    return mock.patch.multiple(
        cfg,
        CONNECTED_NODE_COLOR="red",
        NODE_BOUNDARY_COLOR="black",
        NODE_BOUNDARY_WIDTH=1.5,
        HOVER_COLOR="#ffaa00",
    )


class IdentityTransforms:
    """A TransformSystem, and a node transform, whose scene, canvas and
    framebuffer coordinates are one."""

    @staticmethod
    def map(points):
        points = np.asarray(points, dtype=float)
        flat = np.atleast_2d(points)[:, :2]
        mapped = np.column_stack([flat, np.zeros(len(flat)), np.ones(len(flat))])
        return mapped[0] if points.ndim == 1 else mapped

    imap = map

    def get_transform(self, map_from, map_to):
        return self


class PlacedMarkers:
    """Real VisPy markers, picked through IdentityTransforms: scene units are canvas pixels."""

    def __init__(self):
        object.__setattr__(self, "_markers", scene.visuals.Markers())
        object.__setattr__(self, "transforms", IdentityTransforms())

    def __getattr__(self, name):
        return getattr(self._markers, name)

    def __setattr__(self, name, value):
        setattr(self._markers, name, value)


def make_dragged_node_viewer():
    """A MainViewer with six disc nodes 40 px apart on y = 0, real VisPy Markers, and identity pick."""
    rng = np.random.default_rng(42)
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = 6
    _ = rng.normal(0, 1, (6, 2))
    viewer.edges = np.array([[0, 1], [1, 2], [2, 3], [5, 6]], dtype=np.int32)
    viewer.current_colors = rng.random((6, 4)).astype(np.float32)
    viewer.current_colors[:, 3] = 1.0
    _ = rng.choice([6.0, 10.0, 15.0, 25.0], 6)
    viewer.current_shapes = np.array(["disc"] * 6, dtype=object)
    viewer.visible_mask = np.ones(6, dtype=bool)
    viewer.node_render_order = rng.permutation(6).astype(np.int32)
    viewer.pos = np.array([[0, 0], [40, 0], [80, 0], [120, 0], [160, 0], [200, 0]],
                          dtype=np.float32)  # float32, as the Viewer loads them
    viewer.current_sizes = np.full(6, 10.0, np.float32)
    viewer.selected_indices, viewer.selected_node_idx = [], None
    viewer.left_click_highlight_indices = None
    viewer.hovered_node_idx = None
    viewer.markers = PlacedMarkers()
    transforms = IdentityTransforms()
    viewer.canvas = SimpleNamespace(update=mock.Mock(), pixel_scale=1.0,
                                    scene=SimpleNamespace(node_transform=lambda _scene: transforms))
    viewer.view = SimpleNamespace(scene=object())
    viewer.edge_layer = None
    viewer.hud_displays = {}
    viewer.tooltip = SimpleNamespace(text="")
    viewer.full_headers = [f"N{i}" for i in range(6)]
    viewer.is_multi_dragging = viewer.is_box_selecting = False
    viewer.console_mode = False
    viewer.selection_box = SimpleNamespace(set_data=mock.Mock(), visible=False)
    viewer._hud_timer = mock.Mock()
    viewer._update_hud_elements = mock.Mock()
    for name in ("_save_state", "apply_left_click_focus", "clear_left_click_focus",
                 "_show_click_status", "broadcast_event", "_refresh_session_card"):
        setattr(viewer, name, mock.Mock())
    viewer.update_nodes()
    return viewer


def event(x=0.0, y=0.0, button=None, buttons=(), modifiers=(), delta=(0.0, 0.0)):
    return SimpleNamespace(pos=np.array([float(x), float(y)]), button=button, buttons=list(buttons),
                           modifiers=list(modifiers), handled=False, delta=delta)


def escape():
    return SimpleNamespace(key="Escape", text="", modifiers=[], handled=False)


class RecordingEdges:
    """An edge layer that keeps the positions it was last given."""

    def __init__(self):
        self.positions = None
        self.visible = True
        self.calls = 0

    def show_threshold(self, positions, visible_mask, threshold):
        self.positions = np.array(positions, dtype=float)
        self.calls += 1
        return 3

    def show_edges(self, positions, edges):
        return self.show_threshold(positions, None, 0.0)

    def set_opacity_by_score(self, enabled):
        return False

    def note_motion(self):
        pass


class GestureTestCase(unittest.TestCase):
    """A Viewer to drag nodes in, and the events that do it."""

    DOUBLE_CLICK_MS = 40

    def setUp(self):
        FakeTimer.created = []
        for patcher in (mock.patch.object(cfg, "UMAP_MODE", False, create=True),
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", False),
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", False),
                        distinct_outlines(),     # a selected node's outline differs
                        mock.patch.object(vispy_app, "Timer", FakeTimer),
                        mock.patch.object(QtWidgets.QApplication, "doubleClickInterval",
                                          return_value=self.DOUBLE_CLICK_MS),
                        mock.patch("builtins.print")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_viewer(self, selected=()):
        viewer = make_dragged_node_viewer()
        del viewer._save_state          # the real one: it snapshots into position_history
        viewer.position_history = []
        viewer.redo_stack = []
        viewer.edge_layer = RecordingEdges()
        viewer.current_slider_threshold = 0.0
        viewer.update_console_background = mock.Mock()
        viewer.selected_indices = list(selected)
        viewer.selection_epoch = 0
        viewer._drag_in_click, viewer._drag_turn, viewer._wheel_turn = False, None, None   # as __init__ has them
        viewer._remove_on_click = viewer._header_copy_node = None
        viewer.update_selection_visual()
        viewer.broadcast_event.reset_mock()
        viewer.edge_layer.calls = 0
        return viewer

    @staticmethod
    def press(viewer, x, y, button=2, modifiers=()):
        ev = event(x, y, button=button, buttons=[button], modifiers=modifiers)
        viewer.on_mouse_press(ev)
        return ev

    @staticmethod
    def move(viewer, x, y, buttons=(2,), modifiers=()):
        ev = event(x, y, buttons=buttons, modifiers=modifiers)
        viewer.on_mouse_move(ev)
        return ev

    @staticmethod
    def wheel(viewer, delta, x=0.0, y=0.0, buttons=(2,), modifiers=()):
        ev = event(x, y, buttons=buttons, modifiers=modifiers, delta=delta)
        viewer.on_mouse_wheel(ev)
        return ev

    @staticmethod
    def release(viewer, x, y, button=2, modifiers=()):
        ev = event(x, y, button=button, modifiers=modifiers)
        viewer.on_mouse_release(ev)
        return ev

    @staticmethod
    def double_click(viewer, x, y, button=1, modifiers=()):
        ev = event(x, y, button=button, buttons=[button], modifiers=modifiers)
        viewer.on_mouse_double_click(ev)
        return ev

    @staticmethod
    def esc(viewer):
        ev = escape()
        viewer.on_key_press(ev)
        return ev

    @staticmethod
    def key_release(viewer, key):
        ev = SimpleNamespace(key=key, text="", modifiers=[], handled=False)
        viewer.on_key_release(ev)
        return ev

    @staticmethod
    def rotated(points, about, degrees):
        angle = np.radians(degrees)
        rel = np.asarray(points, dtype=float) - about
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.column_stack([rel[:, 0] * cos_a - rel[:, 1] * sin_a,
                                rel[:, 0] * sin_a + rel[:, 1] * cos_a]) + about

    @staticmethod
    def places(viewer, nodes):
        return viewer.pos[list(nodes), :2].astype(float)

    @staticmethod
    def status(viewer):
        return [str(call.args[0]) for call in viewer._show_click_status.call_args_list]

    def assert_picture_matches(self, viewer):
        """The markers, the edge layer and the pick show the nodes where they are."""
        order = viewer._submitted_marker_node_order
        np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2], viewer.pos[order, :2])
        np.testing.assert_allclose(viewer.edge_layer.positions[:, :2], viewer.pos[:, :2])
        self.assertIsNotNone(viewer._marker_picker)
        for node in range(len(viewer.pos)):
            self.assertEqual(viewer.pick_hovered_node(viewer.pos[node, :2]), node)

    def assert_places(self, viewer, nodes, expected, atol=1e-3):
        np.testing.assert_allclose(self.places(viewer, nodes), np.asarray(expected, dtype=float), atol=atol)


class ViewTransform:
    """A node transform for a view that has panned and zoomed: the canvas place c is the place c * scale + offset
    in the network (a vector of four, as VisPy's transforms and IdentityTransforms give)."""

    def __init__(self, scale, offset):
        self.scale, self.offset = float(scale), np.asarray(offset, dtype=float)

    def map(self, point):
        place = np.asarray(point, dtype=float)[:2] * self.scale + self.offset
        return np.array([place[0], place[1], 0.0, 1.0])

    def imap(self, point):
        place = (np.asarray(point, dtype=float)[:2] - self.offset) / self.scale
        return np.array([place[0], place[1], 0.0, 1.0])


class ZoomTimer:
    """A QTimer's start, stop and timeout, for the smooth wheel zoom, fired by the test."""

    def __init__(self):
        self.active = False
        self.timeout = SimpleNamespace(connect=self._connect)
        self._slot = None

    def _connect(self, slot):
        self._slot = slot

    def isActive(self):
        return self.active

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def fire(self):
        self._slot()
