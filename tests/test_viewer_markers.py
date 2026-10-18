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
        viewer.canvas = SimpleNamespace(update=mock.Mock(), scene=None)
        node = int(viewer._submitted_visible_node_order[-1])
        transform = SimpleNamespace(map=lambda point: np.asarray(point, float)[:2],
                                    inverse=SimpleNamespace(map=lambda point: np.asarray(point, float)[:2]))
        viewer.canvas.scene = SimpleNamespace(node_transform=lambda _scene: transform)
        viewer.view = SimpleNamespace(scene=None)
        viewer.is_multi_dragging = viewer.is_box_selecting = False
        with mock.patch.object(viewer, "_show_hover_change") as shown:
            viewer.on_mouse_move(SimpleNamespace(pos=viewer.pos[node].astype(float), buttons=[],
                                                 modifiers=(), handled=False))
            shown.assert_called_once_with(None)
            self.assertEqual(viewer.hovered_node_idx, node)


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


if __name__ == "__main__":
    unittest.main()
