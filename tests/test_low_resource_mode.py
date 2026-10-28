# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Low Resource Mode by category, and Show Frame Rate (FPS) (desktop/Viewer_State.py,
EMAPSSN_Config.py, EMAPSSN_Viewer.py, desktop/Viewer_Fps.py).

The Visual Effects tab's Low Resource Mode section has a switch per category
(Fewer Edges Drawn While Moving, No Zoom Animation, Update Colors
After Mouse Release, Update Web Pages After Mouse Release) and Show Frame Rate
(FPS), all off by default. These tests check the old single switch's migration
in settings files and settings documents; that each category's helper reads
its own switch and the Viewer asks the helpers; that a node drag never leaves
edges out and moves them with the nodes, Update Colors After Mouse Release
putting the colouring off to the release; what each switch still does in a
Viewer opened in UMAP mode; and the frame-rate display: off by default, left
of the sidebar toggle in every sidebar state, resting when idle, counting
only the canvas's own frames and never in a capture.
"""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import EMAPSSN_Config as cfg  # noqa: E402
import EMAPSSN_Viewer  # noqa: E402
from EMAPSSN_Viewer import MainViewer, _sidebar_room  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402
from desktop import Viewer_Edges, Viewer_State  # noqa: E402
from desktop.Viewer_Fps import FPS_GAP, FPS_IDLE_S, FpsOverlay, FrameRate, fps_text  # noqa: E402
from desktop.Viewer_Motion import MotionDetail  # noqa: E402
from desktop.Viewer_State import (  # noqa: E402
    DEFAULTS, LEGACY_LOW_RESOURCE_KEY, LOW_RESOURCE_DEFAULTS, VISUAL_PROFILE_DEFAULTS,
    decode_document, encode_document, fewer_edges_while_moving, get_viewer_settings_schema,
    low_resource_animations, low_resource_drawing, low_resource_webserver,
    migrate_low_resource_mode, normalize_viewer_settings,
)
from Viewer_Visual_State import edge_stages, edges_follow_selection  # noqa: E402
from tests import test_viewer_edges as edge_tests  # noqa: E402
from tests import test_viewer_markers as marker_tests  # noqa: E402
from tests.theme_fixture import apply_theme_for_class  # noqa: E402

CATEGORIES = ("FEWER_EDGES_WHILE_MOVING", "LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_DRAWING",
              "LOW_RESOURCE_WEBSERVER")
SWITCHES = CATEGORIES + ("SHOW_FPS",)
OLD_CATEGORIES = tuple(LOW_RESOURCE_DEFAULTS)   # what LOW_RESOURCE_MODE stood for
HELPERS = {
    "FEWER_EDGES_WHILE_MOVING": fewer_edges_while_moving,
    "LOW_RESOURCE_ANIMATIONS": low_resource_animations,
    "LOW_RESOURCE_DRAWING": low_resource_drawing,
    "LOW_RESOURCE_WEBSERVER": low_resource_webserver,
}


class FakeTimer:
    def __init__(self, callback):
        self.callback = callback

    def start(self):
        pass


# =====================================================================
# Settings: defaults, the old switch, documents
# =====================================================================

class SettingsTests(unittest.TestCase):
    def test_every_category_and_show_fps_start_off(self):
        self.assertEqual(OLD_CATEGORIES, ("LOW_RESOURCE_ANIMATIONS", "LOW_RESOURCE_DRAWING",
                                          "LOW_RESOURCE_WEBSERVER"))
        for key in SWITCHES:
            with self.subTest(key=key):
                self.assertIs(VISUAL_PROFILE_DEFAULTS[key], False)
                self.assertIs(DEFAULTS[key], False)
                self.assertIs(getattr(cfg, key), False)
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, VISUAL_PROFILE_DEFAULTS)
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, DEFAULTS)
        self.assertFalse(hasattr(cfg, LEGACY_LOW_RESOURCE_KEY))

    def test_the_old_switch_sets_the_three_categories_it_stood_for(self):
        for old in (True, False):
            with self.subTest(old=old):
                migrated = migrate_low_resource_mode({LEGACY_LOW_RESOURCE_KEY: old, "NODE_SIZE": 12})
                self.assertEqual(migrated, {"NODE_SIZE": 12, **{key: old for key in OLD_CATEGORIES}})
        # Fewer Edges Drawn While Moving keeps its own value, or its default (off).
        kept = migrate_low_resource_mode({LEGACY_LOW_RESOURCE_KEY: True, "FEWER_EDGES_WHILE_MOVING": True})
        self.assertEqual(kept, {"FEWER_EDGES_WHILE_MOVING": True, **{key: True for key in OLD_CATEGORIES}})

    def test_categories_already_there_win_over_the_old_switch(self):
        migrated = migrate_low_resource_mode({LEGACY_LOW_RESOURCE_KEY: True, "LOW_RESOURCE_DRAWING": False})
        self.assertEqual(migrated, {"LOW_RESOURCE_DRAWING": False})

    def test_the_config_module_takes_the_old_switch_as_the_categories(self):
        # As it reads viewer_settings.json at import, or a Viewer's settings.
        names = SWITCHES + (LEGACY_LOW_RESOURCE_KEY,)
        saved = {name: getattr(cfg, name) for name in names if hasattr(cfg, name)}
        try:
            cfg.apply_viewer_settings({LEGACY_LOW_RESOURCE_KEY: True})
            self.assertEqual({key: getattr(cfg, key) for key in SWITCHES},
                             {"FEWER_EDGES_WHILE_MOVING": False, "SHOW_FPS": False,
                              **{key: True for key in OLD_CATEGORIES}})
            self.assertFalse(hasattr(cfg, LEGACY_LOW_RESOURCE_KEY))
        finally:
            for name in names:
                if name in saved:
                    setattr(cfg, name, saved[name])
                elif hasattr(cfg, name):
                    delattr(cfg, name)

    def test_motion_detail_off_makes_no_thinned_copies_and_says_why(self):
        reports = []
        detail = MotionDetail(setting=False, report=reports.append, gpu_timer=SimpleNamespace(available=True))
        self.assertFalse(detail.enabled)
        detail.report_once(True)
        self.assertIn("Fewer Edges Drawn While Moving is off", reports[0])
        with mock.patch.dict(os.environ, {"SSN_EDGE_MOTION_LOD": "0"}):
            self.assertFalse(MotionDetail(setting=True).enabled)
            self.assertEqual(MotionDetail(setting=True).off_by, "environment")
        self.assertTrue(MotionDetail(setting=True).enabled)
        with mock.patch.object(Viewer_Edges, "MIN_SUBSET_SEGMENTS", 10):
            for enabled, levels in ((False, (1,)), (True, (1, 4, 16, 64))):
                edges, scores, positions = edge_tests.random_network(3, edges=900)
                layer = edge_tests.recording_layer(edges, scores, positions,
                                                   motion=MotionDetail(enabled=enabled))
                layer.show_threshold(positions, None, 0.0)
                with self.subTest(enabled=enabled):
                    self.assertEqual(tuple(layer.visual.available_levels()), levels)

    def test_settings_without_the_old_switch_pass_through(self):
        values = {"NODE_SIZE": 12, "LOW_RESOURCE_DRAWING": True}
        self.assertIs(migrate_low_resource_mode(values), values)
        self.assertEqual(migrate_low_resource_mode([1]), [1])

    def test_an_old_document_decodes_as_the_categories(self):
        current = encode_document("viewer", DEFAULTS)
        self.assertEqual({key: current["visualization"][key] for key in SWITCHES},
                         {key: False for key in SWITCHES})
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, current["visualization"])
        for old in (True, False):
            with self.subTest(old=old):
                document = json.loads(json.dumps(current))
                for key in SWITCHES:
                    del document["visualization"][key]
                document["visualization"][LEGACY_LOW_RESOURCE_KEY] = old
                values = decode_document(document, "viewer")
                self.assertEqual({key: values[key] for key in OLD_CATEGORIES},
                                 {key: old for key in OLD_CATEGORIES})
                self.assertIs(values["FEWER_EDGES_WHILE_MOVING"], False)
                self.assertIs(values["SHOW_FPS"], False)
                self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, values)
        # A document with the categories keeps them.
        document = json.loads(json.dumps(current))
        document["visualization"]["LOW_RESOURCE_WEBSERVER"] = True
        document["visualization"][LEGACY_LOW_RESOURCE_KEY] = False
        values = decode_document(document, "viewer")
        self.assertIs(values["LOW_RESOURCE_WEBSERVER"], True)
        self.assertIs(values["LOW_RESOURCE_DRAWING"], False)

    def test_an_old_overlay_supplies_the_categories(self):
        from utilities.Headless_Settings import overlay_edits
        edits = overlay_edits({"schema_version": 2, "kind": "viewer",
                               "visualization": {LEGACY_LOW_RESOURCE_KEY: True}}, "viewer")
        self.assertEqual(edits, {key: True for key in OLD_CATEGORIES})

    def test_flat_settings_with_the_old_switch_normalize(self):
        with tempfile.TemporaryDirectory() as root:
            values = normalize_viewer_settings(
                {"TARGET_CACHE_PATH": "", "NODE_FASTA_FILE": "set.fasta", "INPUT_HDF5": "network.h5",
                 "MSA_FILE": "", "UMAP_MODE": False, "SIMILARITY_THRESHOLD": 0.1,
                 "TOP_EDGE_PERCENT": None, LEGACY_LOW_RESOURCE_KEY: "true"},
                root, require_cache=False)
        self.assertEqual({key: values[key] for key in OLD_CATEGORIES}, {key: True for key in OLD_CATEGORIES})
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, values)

    def test_the_schema_lists_the_categories_and_tolerates_the_old_switch(self):
        section = get_viewer_settings_schema()["properties"]["visualization"]
        for key in SWITCHES:
            with self.subTest(key=key):
                self.assertEqual(section["properties"][key]["type"], "boolean")
                self.assertIs(section["properties"][key]["default"], False)
                self.assertNotIn(key, section["required"])
        legacy = section["properties"][LEGACY_LOW_RESOURCE_KEY]
        self.assertTrue(legacy["deprecated"])
        self.assertEqual(legacy["type"], "boolean")
        self.assertNotIn(LEGACY_LOW_RESOURCE_KEY, section["required"])


# =====================================================================
# One helper per category, and the Viewer asks the helpers
# =====================================================================

class CategoryHelperTests(unittest.TestCase):
    def test_each_helper_reads_only_its_own_switch(self):
        for on in SWITCHES:
            configuration = SimpleNamespace(**{key: key == on for key in SWITCHES})
            for key, helper in HELPERS.items():
                with self.subTest(on=on, helper=helper.__name__):
                    self.assertIs(helper(configuration), key == on)
        # A configuration without them (VR's settings) has every category off.
        for helper in HELPERS.values():
            self.assertIs(helper(SimpleNamespace()), False)

    def test_the_viewer_asks_each_category_only_where_its_effects_are(self):
        """Each category's helper is called in the handlers of its own effects,
        and nothing in the Viewer reads the switches themselves."""
        expected = {
            "fewer_edges_while_moving": {"draw_network"},
            "low_resource_animations": {"_camera_mouse_event", "_sidebar_can_slide"},
            "low_resource_drawing": {"on_slider_value_changed", "_on_slider_released", "_redraw_moved_nodes",
                                     "_end_wheel_turn", "on_mouse_move", "on_mouse_release"},
            "low_resource_webserver": {"on_mouse_press", "on_mouse_move"},
        }
        found = {name: set() for name in expected}
        sources = {path: (SRC / path).read_text(encoding="utf-8")
                   for path in ("EMAPSSN_Viewer.py", "Viewer_Visual_State.py")}
        for path, source in sources.items():
            tree = ast.parse(source)
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef):
                    continue
                for node in ast.walk(function):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", None) in expected:
                        found[node.func.id].add(function.name)
            for key in SWITCHES[:-1] + (LEGACY_LOW_RESOURCE_KEY,):
                with self.subTest(path=path, key=key):
                    self.assertNotIn(f"'{key}'", source)
                    self.assertNotIn(f'"{key}"', source)
                    self.assertNotIn(f"cfg.{key}", source)
        self.assertEqual(found, expected)


# =====================================================================
# Node drags: no edge left out, Update Colors After Mouse Release colours on release
# =====================================================================

class NodeDragTests(unittest.TestCase):
    """A right-drag of selected nodes, on real markers and a real edge layer."""

    def make_viewer(self, motion):
        viewer = marker_tests.make_viewer()
        viewer.visible_mask[:] = True
        viewer.edge_scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        viewer.current_slider_threshold = 0.0
        viewer.selected_node_idx = None
        viewer.edge_layer = edge_tests.recording_layer(
            viewer.edges, viewer.edge_scores, viewer.pos,
            motion=MotionDetail(enabled=motion, timer_factory=FakeTimer, report=lambda text: None))
        transforms = marker_tests.IdentityTransforms()
        viewer.canvas = SimpleNamespace(update=mock.Mock(), pixel_scale=1.0,
                                        scene=SimpleNamespace(node_transform=lambda _scene: transforms))
        viewer.view = SimpleNamespace(scene=object())
        viewer.is_box_selecting = False
        viewer.broadcast_event = mock.Mock()
        viewer.selected_indices = [1, 2]
        viewer.update_selection_visual()
        # The press: the drag starts from the selection as it is.
        viewer.is_multi_dragging = True
        viewer._drag_selection_unsent = False
        viewer.drag_start_mouse = np.zeros(2)
        viewer.drag_start_nodes_pos = viewer.pos[viewer.selected_indices, :2].copy()
        outlines = viewer.node_outlines
        viewer.outline_calls = []
        viewer.node_outlines = lambda *a, **k: viewer.outline_calls.append(1) or outlines(*a, **k)
        return viewer

    @staticmethod
    def move(viewer, x, y):
        viewer.on_mouse_move(SimpleNamespace(pos=np.array([x, y], dtype=float), buttons=[2],
                                             modifiers=[], handled=False))

    @staticmethod
    def release(viewer):
        viewer.on_mouse_release(SimpleNamespace(pos=np.zeros(2), button=2, buttons=[],
                                                modifiers=[], handled=False))

    def test_the_edges_follow_the_dragged_nodes_and_none_is_left_out(self):
        for drawing in (False, True):
            for motion in (False, True):
                with self.subTest(drawing=drawing, motion=motion), \
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing), \
                        mock.patch.object(cfg, "UMAP_MODE", False):
                    viewer = self.make_viewer(motion)
                    layer = viewer.edge_layer
                    for step in range(1, 4):
                        uploads = layer.position_upload_count
                        self.move(viewer, 2.0 * step, -1.0 * step)
                        self.assertEqual(layer.position_upload_count, uploads + 1)
                        np.testing.assert_allclose(viewer.pos[[1, 2], :2],
                                                   viewer.drag_start_nodes_pos + [2.0 * step, -1.0 * step])
                        # The edge layer has the nodes where they are now.
                        np.testing.assert_array_equal(np.asarray(layer._positions)[:, :2], viewer.pos[:, :2])
                        self.assertEqual(layer.drawn_edge_count, 4)
                        self.assertTrue(layer.visible)
                        # So do the markers.
                        order = viewer._submitted_marker_node_order
                        np.testing.assert_allclose(viewer.markers._data["a_position"][:, :2],
                                                   viewer.pos[order, :2])
                        self.assertEqual(layer.motion.active, motion)
                    self.release(viewer)
                    self.assertEqual(layer.drawn_edge_count, 4)
                    np.testing.assert_array_equal(np.asarray(layer._positions)[:, :2], viewer.pos[:, :2])

    def test_node_and_edge_drawing_colours_the_selection_on_release(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing), \
                    mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing), \
                    mock.patch.object(cfg, "UMAP_MODE", False):
                viewer = self.make_viewer(motion=True)
                colours = viewer.markers._data["a_bg_color"].copy()
                for step in range(1, 4):
                    self.move(viewer, float(step), 0.0)
                    self.assertEqual(len(viewer.outline_calls), 0 if drawing else step)
                    # The colours stay those the press gave the markers.
                    np.testing.assert_array_equal(viewer.markers._data["a_bg_color"], colours)
                    if drawing:
                        self.assertIsNone(viewer._marker_picker)   # remade on release
                self.release(viewer)
                self.assertEqual(len(viewer.outline_calls), 1 if drawing else 4)
                self.assertIsNotNone(viewer._marker_picker)

    def test_a_drag_after_a_hover_change_sends_every_marker(self):
        with mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", True), \
                mock.patch.object(cfg, "UMAP_MODE", False):
            viewer = self.make_viewer(motion=False)
            viewer.hovered_node_idx = 3
            viewer._show_hover_change(None)
            self.assertIsNone(viewer._submitted_marker_data)
            viewer.outline_calls.clear()      # the hover's own check of the outlines
            self.move(viewer, 1.0, 0.0)
            self.assertEqual(len(viewer.outline_calls), 1)   # update_nodes, in full


class BoxSelectionTests(unittest.TestCase):
    """A right-button selection box tells the web pages (the Meta Data sheet)
    of its selection at each move, or, under Update Web Pages After Mouse Release, once on
    release; Update Colors After Mouse Release selects only on release, so it sends once
    there either way."""

    def run_box(self, webserver, drawing, umap=False):
        maker = marker_tests.DraggedNodePickTests("test_a_dragged_node_can_be_dragged_again")
        event = marker_tests.DraggedNodePickTests.event
        with mock.patch.object(cfg, "UMAP_MODE", umap), \
                mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", webserver), \
                mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
            viewer = maker.make_viewer()        # nodes at (0, 0), (40, 0), (80, 0) ...
            viewer.on_mouse_press(event(30, -10, button=2, buttons=[2]))
            self.assertTrue(viewer.is_box_selecting)
            sent, per_move = [], []
            viewer.broadcast_event = sent.append
            for x in (60, 90):
                viewer.on_mouse_move(event(x, 10, buttons=[2]))
                per_move.append(len(sent))
            viewer.on_mouse_release(event(90, 10, button=2))
        self.assertEqual(sorted(viewer.selected_indices), [1, 2])
        self.assertTrue(all(message["type"] == "selection_changed" for message in sent))
        return sent, per_move

    def test_each_move_tells_the_web_pages_without_webserver_communication(self):
        sent, per_move = self.run_box(webserver=False, drawing=False)
        self.assertEqual(per_move, [1, 2])
        self.assertEqual(len(sent), 2)                          # nothing more on release
        self.assertEqual(sorted(sent[-1]["indices"]), [1, 2])

    def test_webserver_communication_tells_them_once_on_release(self):
        for drawing in (False, True):
            with self.subTest(drawing=drawing):
                sent, per_move = self.run_box(webserver=True, drawing=drawing)
                self.assertEqual(per_move, [0, 0])
                self.assertEqual(len(sent), 1)
                self.assertEqual(sorted(sent[0]["indices"]), [1, 2])

    def test_node_and_edge_drawing_alone_sends_once_on_release(self):
        sent, per_move = self.run_box(webserver=False, drawing=True)
        self.assertEqual(per_move, [0, 0])
        self.assertEqual(len(sent), 1)


# =====================================================================
# UMAP mode (Layout Method set to UMAP or PaCMAP)
# =====================================================================

class UmapModeTests(unittest.TestCase):
    """What each switch still does in a Viewer opened in UMAP mode.

    There the Viewer draws only the selected nodes' edges (edges_follow_selection);
    the threshold slider is built as always and filters them by score; a right-button
    selection box works as always; and a selected node cannot be dragged
    (on_mouse_press) or turned with Ctrl + wheel (on_mouse_wheel). So:

    * Fewer Edges Drawn While Moving still thins a selection of MIN_SUBSET_SEGMENTS
      edges (250,000) or more, and is never in effect for fewer;
    * No Zoom Animation and Show Frame Rate (FPS) never look at the mode;
    * Update Colors After Mouse Release keeps the threshold slider and the selection
      box, and loses the node drag;
    * Update Web Pages After Mouse Release keeps the selection box, and loses the
      node drag.

    No switch has lost every effect, so the Config greys none of them, and its tips
    say which parts do not apply (test_emapssn_config_low_resource).
    """

    def test_the_viewer_reads_the_mode_only_where_these_effects_depend_on_it(self):
        # UMAP and PaCMAP layouts behave alike: both keep each node's strongest edges, draw
        # the selected nodes' edges and cannot be dragged. The Viewer asks
        # desktop.Viewer_State.knn_topology_layout for that; only the cache lookup, which
        # must name the layout, reads UMAP_MODE and LAYOUT_METHOD themselves.
        reads, asks = {}, {}
        for path in ("EMAPSSN_Viewer.py", "Viewer_Visual_State.py"):
            source = (SRC / path).read_text(encoding="utf-8")
            for function in ast.walk(ast.parse(source)):
                if isinstance(function, ast.FunctionDef):
                    body = ast.get_source_segment(source, function)
                    if "UMAP_MODE" in body or "LAYOUT_METHOD" in body:
                        reads.setdefault(path, set()).add(function.name)
                    if "knn_topology_layout" in body:
                        asks.setdefault(path, set()).add(function.name)
        # The cache lookup, the drag, and the selection's edges: nothing else reads the mode, so
        # nothing else makes a switch's effect differ there. A new reader needs this list
        # (and the Config's greying) thought through again.
        self.assertEqual(reads, {"EMAPSSN_Viewer.py": {"load_and_simulate"}})
        self.assertEqual(asks, {
            "EMAPSSN_Viewer.py": {"on_mouse_press", "on_mouse_wheel"},
            "Viewer_Visual_State.py": {"edge_stages", "edges_follow_selection"},
        })
        for path in sorted((SRC / "desktop").glob("Viewer_*.py")):
            if path.name != "Viewer_State.py":              # the settings, not behaviour
                with self.subTest(path=path.name):
                    source = path.read_text(encoding="utf-8")
                    self.assertNotIn("UMAP_MODE", source)
                    self.assertNotIn("LAYOUT_METHOD", source)

    def test_the_predicate_is_true_for_a_umap_or_pacmap_layout_and_not_for_physics(self):
        for settings, expected in (
            ({"UMAP_MODE": False}, False),
            ({"UMAP_MODE": False, "LAYOUT_METHOD": "pacmap"}, False),   # the method means nothing without the mode
            ({"UMAP_MODE": True}, True),
            ({"UMAP_MODE": True, "LAYOUT_METHOD": "umap"}, True),
            ({"UMAP_MODE": True, "LAYOUT_METHOD": "pacmap"}, True),
            ({}, False),
        ):
            with self.subTest(settings=settings):
                self.assertIs(Viewer_State.knn_topology_layout(SimpleNamespace(**settings)), expected)

    def test_the_zoom_and_sidebar_animations_and_the_frame_rate_ignore_the_mode(self):
        import inspect
        for function in (MainViewer._camera_mouse_event, MainViewer._sidebar_can_slide):
            with self.subTest(function=function.__name__):
                source = inspect.getsource(function)
                self.assertNotIn("UMAP", source)
                self.assertIn("low_resource_animations(cfg)", source)
        # The frame rate is drawn by desktop/Viewer_Fps.py, which the scan above shows never reads
        # the mode either, and the Viewer reads it only in the two functions listed there.

    def test_a_selected_node_cannot_be_dragged_so_a_press_sends_the_selection_at_once(self):
        maker = marker_tests.DraggedNodePickTests("test_a_dragged_node_can_be_dragged_again")
        event = marker_tests.DraggedNodePickTests.event
        for umap in (False, True):
            for webserver in (False, True):
                with self.subTest(umap=umap, webserver=webserver), \
                        mock.patch.object(cfg, "UMAP_MODE", umap), \
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", webserver):
                    viewer = maker.make_viewer()
                    viewer.selected_indices = [1]
                    viewer.on_mouse_press(event(40, 0, button=2, buttons=[2]))
                    self.assertEqual(viewer.is_multi_dragging, not umap)
                    # Update Web Pages After Mouse Release holds back only a node drag's selection.
                    held_back = webserver and not umap
                    self.assertEqual(viewer.broadcast_event.call_count, 0 if held_back else 1)
                    viewer.on_mouse_release(event(40, 0, button=2))
                    self.assertEqual(viewer.broadcast_event.call_count, 1)

    def test_a_selection_box_still_follows_the_two_mouse_release_switches(self):
        box = BoxSelectionTests("test_each_move_tells_the_web_pages_without_webserver_communication")
        for webserver, drawing in ((False, False), (True, False), (True, True), (False, True)):
            with self.subTest(webserver=webserver, drawing=drawing):
                sent, per_move = box.run_box(webserver, drawing, umap=True)
                expected_sent, expected_moves = box.run_box(webserver, drawing, umap=False)
                self.assertEqual(per_move, expected_moves)
                self.assertEqual(len(sent), len(expected_sent))
                self.assertEqual(sorted(sent[-1]["indices"]), [1, 2])
        # The two ways the web pages hear of the box: at each move, or once on release.
        self.assertEqual(box.run_box(False, False, umap=True)[1], [1, 2])
        self.assertEqual(box.run_box(True, False, umap=True)[1], [0, 0])
        self.assertEqual(len(box.run_box(True, False, umap=True)[0]), 1)

    def make_umap_viewer(self):
        """A Viewer in UMAP mode with edges (0,1) .9, (1,2) .9, (2,3) .1, (3,4) .6 and nodes 1, 2 selected."""
        maker = edge_tests.UpdateEdgesTests("test_the_threshold_and_hidden_nodes_choose_the_prefix")
        viewer = maker.make_viewer()
        self.addCleanup(maker.doCleanups)
        patcher = mock.patch.object(cfg, "UMAP_MODE", True)
        patcher.start()
        self.addCleanup(patcher.stop)
        viewer.selected_indices = [1, 2]
        viewer.slider_down = False
        viewer.slider = SimpleNamespace(isSliderDown=lambda: viewer.slider_down)
        viewer.threshold_box = SimpleNamespace(value_at=lambda position: position / 100.0,
                                               show_value=mock.Mock())
        return viewer

    def test_the_threshold_slider_still_filters_the_selections_edges(self):
        viewer = self.make_umap_viewer()
        self.assertTrue(edges_follow_selection(viewer, cfg))
        for threshold, drawn in ((0.0, 3), (0.5, 2), (0.95, 0)):
            with self.subTest(threshold=threshold):
                viewer.current_slider_threshold = threshold
                viewer.update_edges()
                self.assertEqual(viewer.edge_layer.drawn_edge_count, drawn)

    def test_update_colors_after_mouse_release_redraws_the_selections_edges_on_release_of_the_slider(self):
        for drawing in (False, True):
            viewer = self.make_umap_viewer()       # its patches come first, then the switch's
            with self.subTest(drawing=drawing), mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing):
                viewer.update_edges()
                self.assertEqual(viewer.edge_layer.drawn_edge_count, 2)       # threshold .5
                viewer.slider_down = True
                viewer.on_slider_value_changed(5)                              # to threshold .05
                self.assertAlmostEqual(viewer.current_slider_threshold, 0.05)
                viewer.threshold_box.show_value.assert_called_with(0.05)       # the box follows either way
                self.assertEqual(viewer.edge_layer.drawn_edge_count, 2 if drawing else 3)
                viewer.slider_down = False
                viewer._on_slider_released()
                self.assertEqual(viewer.edge_layer.drawn_edge_count, 3)

    def test_a_large_selection_gets_thinned_copies_only_while_fewer_edges_is_on(self):
        edges, scores, positions = edge_tests.random_network(3, nodes=60, edges=900)
        with mock.patch.object(Viewer_Edges, "MIN_SUBSET_SEGMENTS", 100), \
                mock.patch.object(cfg, "UMAP_MODE", True):
            for enabled in (False, True):
                for selected, thinned in ((list(range(60)), True), ([0], False)):
                    with self.subTest(fewer_edges=enabled, selected=len(selected)):
                        layer = edge_tests.recording_layer(edges, scores, positions,
                                                           motion=MotionDetail(enabled=enabled))
                        viewer = MainViewer.__new__(MainViewer)
                        viewer.n_nodes, viewer.edges, viewer.edge_scores = 60, edges, scores
                        viewer.pos, viewer.visible_mask = positions, np.ones(60, dtype=bool)
                        viewer.current_slider_threshold = 0.0
                        viewer.selected_indices = selected
                        viewer.edge_layer = layer
                        viewer._refresh_session_card = mock.Mock()
                        viewer.update_edges()
                        self.assertEqual(layer.drawn_edge_count >= 100, thinned)
                        levels = (1, 4, 16, 64) if enabled and thinned else (1,)
                        self.assertEqual(tuple(layer.visual.available_levels()), levels)


# =====================================================================
# Show Frame Rate (FPS)
# =====================================================================

class FrameRateTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.frames = FrameRate(clock=lambda: self.now[0])

    def run_frames(self, count, interval):
        for _ in range(count):
            self.now[0] += interval
            self.frames.drew(1)
            self.assertTrue(self.frames.shown())

    def test_only_frames_the_canvas_drew_count(self):
        # The window composing for a widget over the canvas (the display's own text).
        self.assertFalse(self.frames.shown())
        self.frames.drew(16)
        self.assertTrue(self.frames.shown())
        self.assertEqual(self.frames.detail, 16)
        self.assertFalse(self.frames.shown())

    def test_the_rate_of_the_last_second(self):
        self.assertIsNone(self.frames.rate())
        self.run_frames(100, 1 / 50)
        self.assertAlmostEqual(self.frames.rate(), 50.0, delta=0.5)
        self.run_frames(60, 1 / 30)
        self.assertAlmostEqual(self.frames.rate(), 30.0, delta=0.5)

    def test_idle_once_the_frames_stop_and_not_between_slow_ones(self):
        self.run_frames(30, 1 / 60)
        self.now[0] += FPS_IDLE_S * 0.9
        self.assertIsNotNone(self.frames.rate())
        self.now[0] += FPS_IDLE_S * 0.2
        self.assertIsNone(self.frames.rate())
        self.assertIsNone(self.frames.rate())   # not the old rate again
        # The first frame after a pause starts afresh: no rate from the gap.
        self.now[0] += 3.0
        self.frames.drew(1)
        self.frames.shown()
        self.assertIsNone(self.frames.rate())
        # A slow pan, 1.5 frames a second, is never idle between its frames.
        self.run_frames(4, 1 / 1.5)
        self.now[0] += 0.6
        self.assertAlmostEqual(self.frames.rate(), 1.5, delta=0.05)

    def test_the_texts(self):
        self.assertEqual(fps_text(None), "Idle")
        self.assertEqual(fps_text(58.4, 3.14), "58 fps · GPU 3.1 ms")
        self.assertEqual(fps_text(4.26, None, 16), "4.3 fps · edges 1/16")
        self.assertEqual(fps_text(60.0, None, 1), "60 fps")
        self.assertEqual(fps_text(240.0, 0.042), "240 fps · GPU 0.04 ms")


class FakeGpuTimer:
    def __init__(self):
        self.timed = []

    def time(self, draw, tag, on_result):
        self.timed.append(tag)
        draw()
        on_result(tag, 2.5)


class FpsOverlayTests(unittest.TestCase):
    CANVAS_SIZE = (1200, 800)

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        apply_theme_for_class(cls, cls.app)

    def overlay(self, **kwargs):
        canvas = SimpleNamespace(native=QWidget(), update=mock.Mock(), size=self.CANVAS_SIZE, **kwargs)
        self.addCleanup(canvas.native.deleteLater)
        self.now = [10.0]
        overlay = FpsOverlay(canvas, lambda: 16, clock=lambda: self.now[0], gpu_timer=FakeGpuTimer())
        overlay.attach()
        return canvas, overlay

    def test_it_rests_when_idle_and_never_redraws_the_canvas(self):
        canvas, overlay = self.overlay()
        self.assertEqual(overlay.text(), "Idle")
        self.assertFalse(overlay.timer.isActive())
        overlay.frame_shown()                     # a composition with nothing drawn
        self.assertFalse(overlay.timer.isActive())
        for _ in range(20):
            self.now[0] += 1 / 40
            overlay.frames.drew(16)
            overlay.frame_shown()
        self.assertTrue(overlay.timer.isActive())
        overlay.refresh()
        self.assertTrue(overlay.text().startswith("40 fps"), overlay.text())
        self.assertIn("edges 1/16", overlay.text())
        self.now[0] += 1.0
        overlay.refresh()
        self.assertEqual(overlay.text(), "Idle")
        self.assertFalse(overlay.timer.isActive())   # nothing runs while idle
        canvas.update.assert_not_called()

    def test_captures_are_not_frames(self):
        drawn = []
        canvas, overlay = self.overlay(_draw_scene=lambda *a, **k: drawn.append(1))
        with mock.patch.object(Viewer_Edges, "_draws_into_window", return_value=False):
            canvas._draw_scene()          # canvas.render(): print, MCP captures
        self.assertEqual(drawn, [1])
        self.assertIsNone(overlay.frames._pending)
        self.assertEqual(overlay._gpu.timed, [])
        with mock.patch.object(Viewer_Edges, "_draws_into_window", return_value=True):
            canvas._draw_scene()          # a frame into the window
        self.assertEqual(drawn, [1, 1])
        self.assertEqual(overlay.frames._pending, 16)
        self.assertEqual(overlay.frames.gpu_ms, 2.5)

    # --- in the Viewer ---------------------------------------------------

    def make_viewer(self, show_fps=True, buttons=True):
        viewer = MainViewer.__new__(MainViewer)
        viewer.canvas = SimpleNamespace(native=QWidget(), size=self.CANVAS_SIZE)
        viewer._build_main_window()
        self.addCleanup(viewer.main_window.deleteLater)
        self.addCleanup(viewer.main_window.close)
        viewer._update_hud_elements = lambda: None
        with mock.patch.object(cfg, "SHOW_FPS", show_fps):
            viewer._build_fps_overlay()
        if buttons:
            viewer.add_sidebar_button("metaDataBtn", "Meta Data", lambda: None, icon="chart-column")
        viewer.set_sidebar_visible(False, animate=False)
        viewer.main_window.resize(*self.CANVAS_SIZE)
        viewer.main_window.show()
        self.app.processEvents()
        return viewer

    def assert_by_the_toggle(self, viewer):
        overlay, toggle = viewer.fps_overlay.geometry(), viewer.toggle_sidebar_btn.geometry()
        self.assertEqual(overlay.x() + overlay.width(), toggle.x() - FPS_GAP)
        self.assertEqual(2 * overlay.y() + overlay.height(), 2 * toggle.y() + toggle.height())
        self.assertLessEqual(viewer._command_line_right(), overlay.x() - FPS_GAP)

    def test_off_by_default(self):
        self.assertIs(cfg.SHOW_FPS, False)
        viewer = self.make_viewer(show_fps=False)
        self.assertIsNone(viewer.fps_overlay)
        self.assertEqual(viewer._command_line_right(), self.CANVAS_SIZE[0] - 20)

    def test_left_of_the_sidebar_toggle_open_closed_and_sliding(self):
        # The sidebar is off (SIDEBAR_ENABLED); its kept code puts the frame rate by its toggle.
        for instant in (False, True):
            with self.subTest(instant=instant), mock.patch.object(cfg, "LOW_RESOURCE_ANIMATIONS", instant), \
                    mock.patch.object(EMAPSSN_Viewer, "SIDEBAR_ENABLED", True):
                viewer = self.make_viewer()
                self.assertFalse(viewer.fps_overlay.isHidden())
                self.assert_by_the_toggle(viewer)
                for visible in (True, False):
                    viewer.set_sidebar_visible(visible)
                    self.assertEqual(viewer._sidebar_slide_width is not None, not instant)
                    deadline = time.monotonic() + 3.0
                    while viewer._sidebar_slide_width is not None and time.monotonic() < deadline:
                        self.assert_by_the_toggle(viewer)   # at every frame of the slide
                        self.app.processEvents()
                        time.sleep(0.005)
                    self.assertIsNone(viewer._sidebar_slide_width)
                    self.assertEqual(_sidebar_room(viewer), viewer._panel_w if visible else 0)
                    self.assert_by_the_toggle(viewer)
                # Its text changes keep its right edge by the toggle.
                viewer.fps_overlay.show_text(fps_text(123.0, 45.6, 64))
                self.assert_by_the_toggle(viewer)

    def test_where_the_toggle_would_be_when_it_is_hidden(self):
        # Always, now that the sidebar is off: whether or not its code was given buttons.
        for buttons in (False, True):
            with self.subTest(sidebar_buttons=buttons):
                viewer = self.make_viewer(buttons=buttons)
                self.assertTrue(viewer.toggle_sidebar_btn.isHidden())
                geometry = viewer.fps_overlay.geometry()
                self.assertEqual(geometry.x() + geometry.width(), self.CANVAS_SIZE[0] - 10)
                self.assertLessEqual(viewer._command_line_right(), geometry.x() - FPS_GAP)


_FPS_SCRIPT = textwrap.dedent(
    """
    import sys
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from vispy import app, gloo, scene
    from vispy.gloo import gl
    app.use_app("pyside6")
    from desktop import Viewer_Edges as VE
    from desktop.Viewer_Fps import FpsOverlay
    canvas = scene.SceneCanvas(size=(200, 160), bgcolor="white", show=False, config=dict(stencil_size=8))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-5, 105), y=(-5, 105))
    rng = np.random.default_rng(42)
    pos = (rng.random((80, 2)) * 100).astype(np.float32)
    edges = rng.integers(0, 80, (600, 2)).astype(np.int32)
    layer = VE.ViewerEdges(view.scene, edges, rng.random(600).astype(np.float32), pos, (0, 0, 0, 0.3), 1.5)
    layer.show_threshold(pos, None, 0.0)
    layer.visible = True
    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_data(pos=pos, size=8, face_color="blue")
    before = canvas.render()
    if (before[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    overlay = FpsOverlay(canvas, lambda: 16)
    overlay.attach()
    overlay.show_text("88 fps · GPU 8.8 ms · edges 1/16")
    overlay.place(190, 25)
    overlay.show()
    app.process_events()
    after = canvas.render()
    differs = int((before != after).any(axis=2).sum())
    pending_after_capture = overlay.frames._pending
    shape = before.shape[:2]
    window = gloo.FrameBuffer(color=gloo.RenderBuffer(shape))
    packed = gloo.RenderBuffer(shape, format=0x88F0)
    window.depth_buffer = packed
    window.stencil_buffer = packed
    with mock.patch.object(VE, "_draws_into_window", lambda canvas: True):
        for _ in range(6):
            canvas.push_fbo(window, (0, 0), canvas.size)
            try:
                canvas._draw_scene()
                canvas.context.flush_commands()
            finally:
                canvas.pop_fbo()
            gl.glFinish()
    print("FPS", f"capture_differs={differs}", f"pending_after_capture={pending_after_capture}",
          f"pending_after_frames={overlay.frames._pending}", f"timer={overlay._gpu.available}",
          f"gpu_ms={overlay.frames.gpu_ms}", flush=True)
    """
)


class FpsCaptureTests(unittest.TestCase):
    """On a real GL canvas: a capture (canvas.render(), as print and MCP
    captures take) is the same with the display shown, and counts as no
    frame; frames drawn as into the window count, with their GPU time."""

    def test_the_display_is_never_in_a_capture(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = edge_tests.run_gl_script(self, _FPS_SCRIPT, scale)
                line = next(line for line in output.splitlines() if line.startswith("FPS "))
                values = dict(item.split("=", 1) for item in line.split()[1:])
                self.assertEqual(values["capture_differs"], "0", output)
                self.assertEqual(values["pending_after_capture"], "None", output)
                self.assertEqual(values["pending_after_frames"], "16", output)
                if values["timer"] == "True":
                    self.assertGreater(float(values["gpu_ms"]), 0.0, output)


if __name__ == "__main__":
    unittest.main()
