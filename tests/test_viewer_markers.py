"""The Viewer's node markers on a hover (desktop/Viewer_Markers.py).

A hover recolours two markers instead of sending every marker again; these
tests check that the marker buffer it leaves is byte for byte the one
MainViewer.update_nodes would send, that any other change since update_nodes
sends every marker again, that a uniform shape is passed as one symbol name,
and that the VisPy buffer layout written into is the expected one. The GL
test draws hover frames both ways and compares their pixels.
"""
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg  # noqa: E402
import vispy  # noqa: E402
from vispy import scene  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from desktop import Viewer_Markers  # noqa: E402
from desktop.Viewer_Markers import HoverMarkers, marker_layout_problem, uniform_symbol  # noqa: E402

SHAPES = ["disc", "square", "diamond", "triangle_up", "star", "ring", "cross", "x", "clobber", "arrow"]


def make_viewer(node_count=14, shapes=None):
    """A MainViewer with only what update_nodes draws, on real VisPy Markers."""
    rng = np.random.default_rng(42)
    viewer = MainViewer.__new__(MainViewer)
    viewer.n_nodes = node_count
    viewer.pos = (rng.normal(0, 1, (node_count, 2)) * 3).astype(np.float32)
    viewer.edges = np.array([[0, 1], [1, 2], [2, 3], [5, 6]], dtype=np.int32)
    viewer.current_colors = rng.random((node_count, 4)).astype(np.float32)
    viewer.current_colors[:, 3] = 1.0
    viewer.current_sizes = rng.choice([6.0, 10.0, 15.0, 25.0], node_count).astype(np.float32)
    if shapes is None:
        shapes = [SHAPES[i % len(SHAPES)] for i in range(node_count)]
    viewer.current_shapes = np.array(shapes, dtype=object)
    viewer.visible_mask = np.ones(node_count, dtype=bool)
    viewer.visible_mask[4] = False
    viewer.node_render_order = rng.permutation(node_count).astype(np.int32)
    viewer.selected_indices = [1, 7]
    viewer.selected_node_idx = 9          # its click ring is drawn before it
    viewer.left_click_highlight_indices = None
    viewer.hovered_node_idx = None
    viewer.markers = scene.visuals.Markers()
    viewer.canvas = SimpleNamespace(update=mock.Mock())
    viewer._update_hud_elements = mock.Mock()
    return viewer


def distinct_outlines():
    patcher = mock.patch.multiple(cfg, CONNECTED_NODE_COLOR="red", NODE_BOUNDARY_COLOR="black",
                                  NODE_BOUNDARY_WIDTH=1.5, HOVER_COLOR="#ffaa00")
    return patcher


class VispyLayoutTests(unittest.TestCase):
    def test_vispy_is_the_release_whose_marker_buffer_the_hover_writes(self):
        self.assertEqual(
            vispy.__version__, Viewer_Markers.VISPY_VERSION,
            "VisPy changed: check that MarkersVisual still keeps one record a marker in _data "
            "(fields MARKER_FIELDS, a_bg_color the face colour) and uploads it whole to _vbo, "
            "then update VISPY_VERSION in desktop/Viewer_Markers.py.",
        )

    def test_markers_hold_the_expected_records(self):
        markers = scene.visuals.Markers()
        markers.set_data(pos=np.zeros((3, 2), np.float32), face_color=(0.1, 0.2, 0.3, 1.0),
                         size=np.array([1.0, 2.0, 3.0], np.float32), symbol="square")
        self.assertIsNone(marker_layout_problem(markers, 3))
        np.testing.assert_allclose(markers._data["a_bg_color"][1], (0.1, 0.2, 0.3, 1.0), rtol=1e-6)
        self.assertIn("4", marker_layout_problem(markers, 4))
        self.assertIsNotNone(marker_layout_problem(SimpleNamespace(), 3))
        broken = SimpleNamespace(_data=markers._data, _vbo=None)
        self.assertIsNotNone(marker_layout_problem(broken, 3))


class UniformSymbolTests(unittest.TestCase):
    def test_one_name_only_when_every_marker_has_it(self):
        self.assertEqual(uniform_symbol(np.array(["disc"] * 5, dtype=object)), "disc")
        self.assertIsNone(uniform_symbol(np.array(["disc", "disc", "square"], dtype=object)))
        self.assertIsNone(uniform_symbol(np.array([], dtype=object)))
        self.assertIsNone(uniform_symbol(np.array([None, None], dtype=object)))

    def test_one_name_draws_what_the_list_draws(self):
        for name in ("disc", "o", "square", "star"):
            one, listed = scene.visuals.Markers(), scene.visuals.Markers()
            one.set_data(pos=np.zeros((4, 2), np.float32), symbol=name)
            listed.set_data(pos=np.zeros((4, 2), np.float32), symbol=[name] * 4)
            self.assertEqual(one._data.tobytes(), listed._data.tobytes(), name)

    def test_update_nodes_passes_one_name_for_a_uniform_shape(self):
        for shapes, expected in ((["star"] * 14, "star"), (None, list)):
            with self.subTest(shapes=expected), distinct_outlines():
                viewer = make_viewer(shapes=shapes)
                with mock.patch.object(viewer.markers, "set_data", wraps=viewer.markers.set_data) as set_data:
                    viewer.update_nodes()
                symbol = set_data.call_args.kwargs["symbol"]
                if expected is list:
                    self.assertIsInstance(symbol, list)
                else:
                    self.assertEqual(symbol, expected)


class HoverMarkersTests(unittest.TestCase):
    def setUp(self):
        patcher = distinct_outlines()
        patcher.start()
        self.addCleanup(patcher.stop)
        HoverMarkers._layout_reported = False
        self.addCleanup(setattr, HoverMarkers, "_layout_reported", False)
        self.viewer = make_viewer()
        self.viewer.update_nodes()

    def hover(self, node):
        """Hover node through the hover path; return whether every marker was sent."""
        viewer = self.viewer
        previous = viewer.hovered_node_idx
        viewer.hovered_node_idx = node
        with mock.patch.object(viewer, "update_nodes", wraps=viewer.update_nodes) as full:
            viewer._show_hover_change(previous)
        return full.called

    def full_buffer(self):
        """The marker buffer update_nodes sends for the viewer as it is now."""
        self.viewer.update_nodes()
        return self.viewer.markers._data.tobytes()

    def test_a_hover_leaves_the_buffer_update_nodes_would_send(self):
        viewer = self.viewer
        order = list(viewer._submitted_marker_node_order)
        # A node drawn under others, one drawn over it, the clicked node (two
        # markers: its ring and itself), a hidden node (no marker), and none.
        covered, above = order[1], order[-1]
        for node in (covered, above, 9, covered, 4, None, 9, None):
            with self.subTest(node=node):
                with mock.patch.object(viewer.markers._vbo, "set_subdata",
                                       wraps=viewer.markers._vbo.set_subdata) as sent:
                    sent_whole = self.hover(node)
                self.assertFalse(sent_whole)
                self.assertLessEqual(sent.call_count, 2)
                self.assertTrue(all(len(call.args[0]) <= 2 for call in sent.call_args_list))
                patched = viewer.markers._data.tobytes()
                self.assertEqual(patched, self.full_buffer())

    def test_the_hover_colour_on_a_click_ring_has_half_the_opacity(self):
        viewer = self.viewer
        viewer.canvas.update.reset_mock()
        viewer._update_hud_elements.reset_mock()
        self.assertFalse(self.hover(9))
        slots = np.flatnonzero(viewer._submitted_marker_node_order == 9)
        ring = slots[viewer._submitted_marker_ring_mask[slots]][0]
        node = slots[~viewer._submitted_marker_ring_mask[slots]][0]
        face = viewer.markers._data["a_bg_color"]
        np.testing.assert_allclose(face[node], (1.0, 2 / 3, 0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(face[ring], (1.0, 2 / 3, 0.0, 0.5), atol=1e-6)
        viewer.canvas.update.assert_called()
        viewer._update_hud_elements.assert_called()

    def test_any_other_change_since_update_nodes_sends_every_marker(self):
        def recolour(viewer):
            viewer.current_colors[3] = (0.0, 0.0, 0.0, 1.0)

        def resize(viewer):
            viewer.current_sizes[3] *= 2

        def move(viewer):
            viewer.pos[3] += 1.0

        def hide(viewer):
            viewer.visible_mask[6] = False

        def reshape(viewer):
            viewer.current_shapes[3] = "triangle_down"

        def select(viewer):
            viewer.selected_indices = [2]

        def click(viewer):
            viewer.selected_node_idx = 11

        def promote(viewer):
            viewer.node_render_order = viewer.node_render_order[::-1].copy()

        def capture(viewer):
            viewer.transient_marks_hidden = True

        def resend(viewer):
            viewer.markers.set_data(pos=np.zeros((2, 2), np.float32))

        def outline_setting(viewer):
            cfg.NODE_BOUNDARY_WIDTH = 0.5

        for change in (recolour, resize, move, hide, reshape, select, click, promote,
                       capture, resend, outline_setting):
            with self.subTest(change=change.__name__):
                self.setUp()
                change(self.viewer)
                self.assertTrue(self.hover(self.viewer._submitted_marker_node_order[0]))

    def test_an_unexpected_buffer_layout_sends_every_marker_and_says_so_once(self):
        reports = []
        with mock.patch.object(Viewer_Markers, "marker_layout_problem", return_value="changed"), \
                mock.patch("builtins.print", reports.append):
            self.viewer.update_nodes()
            self.viewer.update_nodes()
            self.assertTrue(self.hover(self.viewer._submitted_marker_node_order[0]))
            self.assertTrue(self.hover(None))
        said = [text for text in reports if "hover updates send every marker" in str(text)]
        self.assertEqual(len(said), 1)
        self.assertIn("changed", said[0])

    def test_mouse_moves_over_nodes_go_through_the_hover_path(self):
        viewer = self.viewer
        transform = SimpleNamespace(map=lambda point: np.asarray(point, float)[:2])
        viewer.canvas = SimpleNamespace(update=mock.Mock(),
                                        scene=SimpleNamespace(node_transform=lambda _scene: transform))
        viewer.view = SimpleNamespace(scene=None)
        viewer.is_multi_dragging = viewer.is_box_selecting = False

        def move():
            viewer.on_mouse_move(SimpleNamespace(pos=np.zeros(2), buttons=[], modifiers=(), handled=False))
        with mock.patch.object(viewer, "_show_hover_change") as shown, \
                mock.patch.object(viewer, "pick_hovered_node", return_value=6):
            move()
            shown.assert_called_once_with(None)
            self.assertEqual(viewer.hovered_node_idx, 6)
            move()                                  # still on node 6: nothing to show
            shown.assert_called_once()
        with mock.patch.object(viewer, "_show_hover_change") as shown, \
                mock.patch.object(viewer, "pick_hovered_node", return_value=None):
            move()
            shown.assert_called_once_with(6)
            self.assertIsNone(viewer.hovered_node_idx)


class FakeTransforms:
    """A TransformSystem whose visual coordinates are framebuffer pixels
    divided by scale, y up, and canvas pixels framebuffer pixels divided by
    pixel_scale, y down, on a framebuffer height pixels tall."""

    def __init__(self, scale=1.0, pixel_scale=1.0, height=400.0):
        self.scale, self.pixel_scale, self.height = scale, pixel_scale, height

    def get_transform(self, map_from, map_to):
        if (map_from, map_to) == ("visual", "framebuffer"):
            return SimpleNamespace(map=self._visual_to_fb, imap=self._fb_to_visual)
        if (map_from, map_to) == ("canvas", "framebuffer"):
            return SimpleNamespace(map=lambda p: np.array(
                [p[0] * self.pixel_scale, self.height - p[1] * self.pixel_scale, 0.0, 1.0]))
        raise AssertionError((map_from, map_to))

    def _visual_to_fb(self, points):
        points = np.atleast_2d(np.asarray(points, float))
        return np.column_stack([points[:, :2] * self.scale, np.zeros(len(points)), np.ones(len(points))])

    def _fb_to_visual(self, point):
        return np.array([point[0] / self.scale, point[1] / self.scale, 0.0, 1.0])


class MarkerPickerTests(unittest.TestCase):
    """The pick against the markers' drawn shapes (desktop.Viewer_Markers.MarkerPicker)."""

    def picker(self, nodes, positions, sizes, shapes, outlines=None, rings=None, antialias=1.0):
        count = len(nodes)
        node_positions = np.zeros((max(nodes, default=-1) + 1, 2), np.float32)
        for node, position in zip(nodes, positions):
            node_positions[node] = position
        return Viewer_Markers.MarkerPicker(
            np.array(nodes, dtype=np.int64), np.zeros(count, bool) if rings is None else np.array(rings),
            node_positions, np.array(sizes, np.float32),
            np.full(count, 0.0, np.float32) if outlines is None else np.array(outlines, np.float32),
            Viewer_Markers.symbol_codes_of(shapes), antialias)

    def pick(self, picker, x, y, pixel_scale=1.0, height=400.0):
        """Pick at framebuffer pixel (x, y), y up, as a cursor at that pixel's centre."""
        transforms = FakeTransforms(pixel_scale=pixel_scale, height=height)
        cursor = ((x + 0.5) / pixel_scale, (height - y - 0.5) / pixel_scale)
        return picker.pick(transforms, cursor, pixel_scale)

    def test_the_topmost_covering_node_wins_over_a_nearer_centre(self):
        # Node 1, small, sits under node 2, big and drawn later, near its centre.
        picker = self.picker([1, 2], [(100, 100), (110, 100)], [6, 40], ["disc", "disc"])
        self.assertEqual(self.pick(picker, 100, 100), 2)
        # Drawn the other way round, the small one is on top where it is drawn.
        picker = self.picker([2, 1], [(110, 100), (100, 100)], [40, 6], ["disc", "disc"])
        self.assertEqual(self.pick(picker, 100, 100), 1)
        self.assertEqual(self.pick(picker, 106, 100), 2)

    def test_the_reach_is_the_drawn_shape_and_its_outline(self):
        # A disc 20 across with a 4-pixel outline draws out to 10 + 2 + 1 (antialias).
        picker = self.picker([7], [(100, 100)], [20], ["disc"], outlines=[4])
        self.assertEqual(self.pick(picker, 112, 100), 7)       # centre 112.5: 12.5 out
        margin = Viewer_Markers.HOVER_MARGIN_PX
        self.assertEqual(self.pick(picker, 113, 100), 7)       # 13.5: within the margin
        self.assertIsNone(self.pick(picker, 100 + int(13 + margin) + 1, 100))
        # At display scale 2 the same node is twice as many pixels across:
        # 20 + 4 + 1 out, and the margin is twice as wide too.
        self.assertEqual(self.pick(picker, 124, 100, pixel_scale=2.0), 7)
        self.assertEqual(self.pick(picker, 126, 100, pixel_scale=2.0), 7)
        self.assertIsNone(self.pick(picker, 100 + int(25 + 2 * margin) + 1, 100, pixel_scale=2.0))

    def test_shapes_cover_what_they_draw(self):
        # A square's corner is drawn, a disc's is not; a ring's hole is not.
        square = self.picker([3], [(100, 100)], [20], ["square"])
        disc = self.picker([3], [(100, 100)], [20], ["disc"])
        ring = self.picker([4, 3], [(100, 100), (100, 100)], [6, 40], ["disc", "ring"])
        self.assertEqual(self.pick(square, 110, 110), 3)
        self.assertIsNone(self.pick(disc, 110, 110))
        self.assertEqual(self.pick(ring, 100, 100), 4)          # through the hole, the node below
        self.assertEqual(self.pick(ring, 115, 100), 3)          # on the band
        # A triangle points up (framebuffer y points up): its base corners are
        # drawn, the same offsets above its centre are not.
        triangle = self.picker([5], [(100, 100)], [30], ["triangle_up"])
        self.assertEqual(self.pick(triangle, 112, 88), 5)
        self.assertIsNone(self.pick(triangle, 112, 112))
        down = self.picker([5], [(100, 100)], [30], ["triangle_down"])
        self.assertIsNone(self.pick(down, 112, 88))
        self.assertEqual(self.pick(down, 112, 111), 5)

    def test_a_whole_pixel_cursor_names_the_pixel_below_and_right_of_it(self):
        # A square 4 pixels across, centred on framebuffer (100, 100), y up:
        # its rows are framebuffer y 98..101, canvas rows 298..301 (y down).
        for scale in (1.0, 2.0):
            with self.subTest(display_scale=scale):
                picker = self.picker([3], [(100, 100)], [4 / scale], ["square"], antialias=0.0)
                transforms = FakeTransforms(pixel_scale=scale, height=400.0)
                with mock.patch.object(Viewer_Markers, "HOVER_MARGIN_PX", 0.0):
                    top = picker.pick(transforms, (100 / scale, 298 / scale), scale)
                    above = picker.pick(transforms, (100 / scale, 297 / scale), scale)
                    bottom = picker.pick(transforms, (100 / scale, 301 / scale), scale)
                    below = picker.pick(transforms, (100 / scale, 302 / scale), scale)
                self.assertEqual((top, above, bottom, below), (3, None, 3, None))

    def test_click_rings_are_not_nodes(self):
        picker = self.picker([8, 8, 9], [(100, 100), (100, 100), (130, 100)], [60, 30, 10],
                             ["disc", "disc", "disc"], rings=[True, False, False])
        self.assertEqual(self.pick(picker, 100, 100), 8)
        self.assertIsNone(self.pick(picker, 125, 100 + 20))     # on the ring alone
        self.assertEqual(self.pick(picker, 128, 100), 9)

    def test_the_tree_outlives_changes_that_move_no_node(self):
        first = self.picker([1, 2], [(100, 100), (110, 100)], [6, 40], ["disc", "disc"])
        self.assertEqual(self.pick(first, 100, 100), 2)
        # The same positions drawn in another order (a selection raised node 1).
        again = Viewer_Markers.MarkerPicker(np.array([2, 1]), np.zeros(2, bool), first.node_positions,
                                            np.array([40, 6], np.float32), np.zeros(2, np.float32),
                                            first.codes, 1.0, previous=first)
        self.assertIs(again._tree, first._tree)
        self.assertEqual(self.pick(again, 100, 100), 1)
        moved = first.node_positions.copy()
        moved[1, 0] += 50
        after = Viewer_Markers.MarkerPicker(np.array([2, 1]), np.zeros(2, bool), moved,
                                            np.array([40, 6], np.float32), np.zeros(2, np.float32),
                                            first.codes, 1.0, previous=again)
        self.assertIsNone(after._tree)
        self.assertEqual(self.pick(after, 100, 100), 2)
        self.assertEqual(self.pick(after, 150, 100), 1)

    def test_hidden_nodes_are_never_picked(self):
        # Node 0 is hidden: it has a position but no marker.
        picker = Viewer_Markers.MarkerPicker(np.array([1]), np.zeros(1, bool),
                                             np.array([[100, 100], [140, 100]], np.float32),
                                             np.array([20.0], np.float32), np.zeros(1, np.float32),
                                             Viewer_Markers.symbol_codes_of(["disc"]), 1.0)
        self.assertIsNone(self.pick(picker, 100, 100))
        self.assertEqual(self.pick(picker, 140, 100), 1)

    def test_no_markers_pick_nothing(self):
        self.assertIsNone(self.pick(self.picker([], np.zeros((0, 2)), [], []), 1, 1))

    def test_the_viewer_picks_with_its_markers(self):
        with distinct_outlines():
            viewer = make_viewer()
            viewer.update_nodes()
        picker = viewer._marker_picker
        np.testing.assert_array_equal(picker.nodes, viewer._submitted_marker_node_order[
            ~viewer._submitted_marker_ring_mask])
        np.testing.assert_array_equal(picker.positions, viewer.markers._data["a_position"][
            ~viewer._submitted_marker_ring_mask, :2])
        np.testing.assert_array_equal(picker.codes, viewer.markers._data["a_symbol"][
            ~viewer._submitted_marker_ring_mask].astype(np.int64))
        viewer.canvas = SimpleNamespace(pixel_scale=2.0)
        with mock.patch.object(picker, "pick", return_value=3) as pick:
            self.assertEqual(viewer.pick_hovered_node((1.0, 2.0)), 3)
        self.assertEqual(pick.call_args.args[1:], ((1.0, 2.0), 2.0))
        viewer.markers.visible = False
        self.assertIsNone(viewer.pick_hovered_node((1.0, 2.0)))


_HOVER_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    os.environ["SSN_VIEWER_EXPLICIT_SETTINGS"] = "1"
    from vispy import app, gloo, scene
    app.use_app("pyside6")
    import EMAPSSN_Viewer as V
    from desktop import Viewer_Edges as VE
    cfg = V.cfg
    cfg.NODE_SIZE, cfg.NODE_BOUNDARY_WIDTH = 15, 2.0
    cfg.CONNECTED_NODE_COLOR, cfg.NODE_BOUNDARY_COLOR = "red", "black"
    rng = np.random.default_rng(42)
    n = 40
    shapes = ["disc", "square", "diamond", "triangle_up", "star", "ring", "cross", "x", "clobber", "arrow"]
    v = V.MainViewer.__new__(V.MainViewer)
    v.n_nodes = n
    v.pos = (rng.normal(0, 1, (n, 2)) * 4).astype(np.float32)
    v.edges = rng.integers(0, n, (300, 2)).astype(np.int32)
    v.edge_scores = rng.random(300).astype(np.float32)
    v.current_colors = rng.random((n, 4)).astype(np.float32)
    v.current_colors[:, 3] = 1.0
    v.current_sizes = rng.choice([6.0, 10.0, 15.0, 25.0], n).astype(np.float32)
    v.current_shapes = np.array([shapes[i % len(shapes)] for i in range(n)], dtype=object)
    v.visible_mask = np.ones(n, dtype=bool)
    v.visible_mask[4] = False
    v.node_render_order = rng.permutation(n).astype(np.int32)
    v.selected_indices = [1, 7]
    v.selected_node_idx = 9
    v.left_click_highlight_indices = None
    v.hovered_node_idx = None
    v.current_slider_threshold = 0.0
    v.canvas = scene.SceneCanvas(size=(200, 160), bgcolor="white", show=False, config=dict(stencil_size=8))
    v.view = v.canvas.central_widget.add_view()
    v.view.camera = scene.PanZoomCamera(aspect=1)
    v.view.camera.set_range(x=(-12, 12), y=(-10, 10))
    v._update_hud_elements = lambda *args, **kwargs: None
    v._refresh_session_card = lambda: None
    cfg.EDGE_COLOR, cfg.EDGE_ALPHA, cfg.EDGE_WIDTH = "grey", 0.1, 1.5
    v.edge_layer = VE.ViewerEdges(v.view.scene, v.edges, v.edge_scores, v.pos, (0.5, 0.5, 0.5, 0.1), 1.5,
                                  image_cache=VE.EdgeImageCache(enabled=True, report=lambda text: None))
    v.markers = scene.visuals.Markers(parent=v.view.scene)
    v.update_edges()
    v.update_nodes()
    if (v.canvas.render()[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    print("SCALE", v.canvas.pixel_scale, flush=True)
    shape = v.canvas.render().shape[:2]
    window = gloo.FrameBuffer(color=gloo.RenderBuffer(shape))
    packed = gloo.RenderBuffer(shape, format=0x88F0)
    window.depth_buffer = packed
    window.stencil_buffer = packed
    cache = v.edge_layer.image_cache

    def frame():
        # A frame drawn as into the window, where the edge cache serves.
        with mock.patch.object(VE, "_draws_into_window", lambda canvas: True):
            v.canvas.push_fbo(window, (0, 0), v.canvas.size)
            try:
                v.canvas._draw_scene()
                return window.read()
            finally:
                v.canvas.pop_fbo()

    frame()
    order = list(v._submitted_marker_node_order)
    pictures = cache.render_count
    patched = whole = 0
    for node in (order[1], order[-1], 9, order[1], 4, None, 9, order[len(order) // 2], None):
        previous, v.hovered_node_idx = v.hovered_node_idx, node
        calls = []
        with mock.patch.object(v, "update_nodes", side_effect=lambda: calls.append(1) or V.MainViewer.update_nodes(v)):
            v._show_hover_change(previous)
        patched += not calls
        hovered = frame()
        V.MainViewer.update_nodes(v)          # every marker sent, as before
        sent = frame()
        differing = int((hovered != sent).any(axis=2).sum())
        whole += differing
        print("HOVER", node, int(not calls), differing, flush=True)
    print("RESULT", patched, whole, cache.render_count - pictures, flush=True)
    """
)


_PICK_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    os.environ["SSN_VIEWER_EXPLICIT_SETTINGS"] = "1"
    import matplotlib.colors as mcolors
    from vispy import app, scene
    app.use_app("pyside6")
    import EMAPSSN_Viewer as V
    from desktop import Viewer_Markers as VM
    cfg = V.cfg
    cfg.NODE_SIZE, cfg.NODE_BOUNDARY_WIDTH = 15, 2.0
    cfg.CONNECTED_NODE_COLOR, cfg.NODE_BOUNDARY_COLOR = "red", "black"
    shapes = ["disc", "square", "diamond", "triangle_up", "triangle_down", "star", "ring", "cross",
              "x", "vbar", "hbar", "arrow", "tailed_arrow", "clobber", "cross_lines"]
    contexts = []

    def scene_viewer(kind):
        rng = np.random.default_rng(42)
        n = 60
        v = V.MainViewer.__new__(V.MainViewer)
        v.n_nodes = n
        v.pos = (rng.normal(0, 1, (n, 2)) * 6).astype(np.float32)
        v.edges = np.array([[5, 6], [9, 10]], dtype=np.int32)
        v.current_colors = np.tile(mcolors.to_rgba(cfg.INITIAL_NODE_COLOR), (n, 1)).astype(np.float32)
        if kind == "mixed":
            v.current_sizes = rng.choice([6.0, 10.0, 15.0, 22.0, 30.0], n).astype(np.float32)
            v.current_shapes = np.array([shapes[i % len(shapes)] if i % 3 else "disc" for i in range(n)], dtype=object)
        else:
            v.current_sizes = np.full(n, 15.0, dtype=np.float32)
            v.current_shapes = np.full(n, "disc", dtype=object)
        v.visible_mask = np.ones(n, dtype=bool)
        v.visible_mask[[3, 17]] = False
        v.node_render_order = rng.permutation(n).astype(np.int32)
        v.selected_indices = [5, 9, 40]
        v.hovered_node_idx = None
        v.selected_node_idx = 12
        v.left_click_highlight_indices = None
        v.canvas = scene.SceneCanvas(size=(300, 240), bgcolor="white", show=False, config=dict(stencil_size=8))
        contexts.append(v.canvas)
        v.view = v.canvas.central_widget.add_view()
        v.view.camera = scene.PanZoomCamera(aspect=1)
        v._update_hud_elements = lambda *args, **kwargs: None
        v.markers = scene.visuals.Markers(parent=v.view.scene)
        v.update_nodes()
        v.view.camera.rect = (-20, -16, 40, 32)
        return v

    def id_image(v):
        # The same markers, face and outline in each node's ID colour, blending
        # off: each pixel holds the topmost node whose drawn shape covers it.
        data = v.markers._data
        keep = ~np.asarray(v._submitted_marker_ring_mask)
        ids = np.asarray(v._submitted_marker_node_order)[keep] + 1
        rgba = np.stack([(ids & 255) / 255.0, ((ids >> 8) & 255) / 255.0, np.zeros(len(ids)), np.ones(len(ids))], axis=1)
        marks = scene.visuals.Markers(parent=v.view.scene)
        marks.set_data(pos=data["a_position"][keep, :2], face_color=rgba, edge_color=rgba,
                       size=data["a_size"][keep], edge_width=data["a_edgewidth"][keep],
                       symbol=[v.current_shapes[i] for i in np.asarray(v._submitted_marker_node_order)[keep]])
        marks.set_gl_state(blend=False, depth_test=False)
        v.markers.visible = False
        image = v.canvas.render()
        v.markers.visible, marks.visible = True, False
        under = image[..., 0].astype(int) + (image[..., 1].astype(int) << 8) - 1
        under[image[..., 2] != 0] = -1
        return under

    def excess(v, node, pixel):
        picker = v._marker_picker
        i = int(np.flatnonzero(picker.nodes == node)[0])
        to_fb = v.markers.transforms.get_transform("visual", "framebuffer")
        centre = np.asarray(to_fb.map(picker.positions[i:i + 1]))[0, :2]
        scale = v.canvas.pixel_scale
        r = VM.marker_distance([picker.codes[i]], [pixel[0] - centre[0]], [centre[1] - pixel[1]],
                               [picker.sizes[i] * scale])[0]
        return r - (0.5 * picker.edge_widths[i] * scale + picker.antialias)

    first = True
    for kind in ("mixed", "discs"):
        v = scene_viewer(kind)
        if first:
            if (v.canvas.render()[..., 0] < 250).sum() == 0:
                sys.exit(3)
            print("CONTEXT OK", flush=True)
            print("SCALE", v.canvas.pixel_scale, flush=True)
            first = False
        scale = v.canvas.pixel_scale
        under = id_image(v)
        height, width = under.shape
        rows, cols = np.nonzero(under >= 0)
        c2f = v.markers.transforms.get_transform("canvas", "framebuffer")
        reach = int(np.ceil(VM.HOVER_MARGIN_PX * scale)) + 1
        points = on_nodes = wrong = ties = background_picks = background_far = 0
        step = int(2 * scale)
        for py in range(max(0, rows.min() - 10), min(height, rows.max() + 10), step):
            for px in range(max(0, cols.min() - 10), min(width, cols.max() + 10), step):
                truth = int(under[py, px])
                picked = v.pick_hovered_node(((px + 0.5) / scale, (py + 0.5) / scale))
                picked = -1 if picked is None else picked
                points += 1
                if truth >= 0:
                    on_nodes += 1
                    if picked != truth:
                        pixel = np.floor(np.asarray(c2f.map(np.array([(px + 0.5) / scale, (py + 0.5) / scale])))[:2]) + 0.5
                        if min(abs(excess(v, node, pixel)) for node in (truth, picked) if node >= 0) < 0.01:
                            ties += 1
                        else:
                            wrong += 1
                elif picked >= 0:
                    background_picks += 1
                    window = under[max(0, py - reach):py + reach + 1, max(0, px - reach):px + reach + 1]
                    background_far += not (window == picked).any()
        print("GRID", kind, points, on_nodes, wrong, ties, background_picks, background_far, flush=True)
    """
)


def run_gl_script(test, script, scale):
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", QT_ENABLE_HIGHDPI_SCALING="0",
                       QT_SCALE_FACTOR=str(scale))
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, SRC_DIR],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=environment, timeout=300,
        )
    except subprocess.TimeoutExpired:
        test.skipTest("the GL render probe timed out")
    if "CONTEXT OK" not in result.stdout:
        test.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr[-2000:])
    return result.stdout + result.stderr[-2000:]


class HoverPixelTests(unittest.TestCase):
    """Hover frames drawn after the two-marker update equal those drawn after
    update_nodes sends every marker, pixel for pixel, at display scales 1 and
    2, with ten shapes, three outline widths, a click ring and nodes drawn
    over each other; and a hover never redraws the edges' cached picture."""

    def test_hover_frames_equal_frames_with_every_marker_sent(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _HOVER_SCRIPT, scale)
                self.assertIn(f"SCALE {float(scale)}", output)
                result = next(line.split()[1:] for line in output.splitlines() if line.startswith("RESULT"))
                patched, differing, pictures = map(int, result)
                self.assertEqual(patched, 9, output)
                self.assertEqual(differing, 0, output)
                self.assertEqual(pictures, 0, output)


class PickPixelTests(unittest.TestCase):
    """The hover pick against what is drawn, at display scales 1 and 2.

    Overlapping nodes of five sizes, fifteen shapes and two outline widths,
    a click ring, raised selected nodes and hidden nodes, then plain discs, are
    drawn once more as an ID image (each pixel holding the topmost node whose
    drawn shape covers it). On a grid of cursor positions, over a node the pick
    is that node, but where the pixel's centre lies on a shape's very edge
    (within 0.01 px, where the GPU's float32 and NumPy's float64 can differ);
    over no node it is no node or one drawn within the hover margin.
    """

    def test_the_pick_is_the_topmost_node_drawn_under_the_cursor(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _PICK_SCRIPT, scale)
                self.assertIn(f"SCALE {float(scale)}", output)
                grids = [line.split()[1:] for line in output.splitlines() if line.startswith("GRID ")]
                self.assertEqual([grid[0] for grid in grids], ["mixed", "discs"], output)
                for kind, points, on_nodes, wrong, ties, background_picks, far in grids:
                    with self.subTest(scene=kind):
                        self.assertGreater(int(on_nodes), 1000, output)
                        self.assertEqual(int(wrong), 0, output)
                        self.assertLessEqual(int(ties), int(on_nodes) // 1000, output)
                        self.assertEqual(int(far), 0, output)


if __name__ == "__main__":
    unittest.main()
