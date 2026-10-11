"""The desktop Viewer's edge layer (desktop/Viewer_Edges.py) and update_edges.

The edges are sorted by descending score once, so a threshold keeps a prefix
of them, and their buffer holds node numbers that the vertex shader resolves
through a texture of node positions. These tests check that the drawn edges
are exactly those edge_stages keeps, and that each kind of change re-sends only
what it must: a threshold change nothing, a node move the position texture,
hiding nodes the edge buffer once. The last test renders the old Line visual
and the new one on a real GL canvas and compares their pixels.
"""
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from vispy.scene.visuals import create_visual_node


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import EMAPSSN_Config as cfg  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from Viewer_Visual_State import edge_stages, edges_follow_selection  # noqa: E402
from desktop import Viewer_Edges  # noqa: E402
from desktop.Viewer_Edges import (  # noqa: E402
    EdgeOrder,
    EdgeSegmentsVisual,
    TEXTURE_WIDTH,
    ViewerEdges,
    _contiguous_line_positions,
    _descending_order,
    _edge_segment_positions,
    position_texels,
    position_texture_enabled,
)


def random_network(seed, nodes=60, edges=400):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, nodes, (edges, 2)).astype(np.int32)
    scores = rng.random(edges).astype(np.float32)
    scores[::5] = 0.5  # ties
    positions = (rng.random((nodes, 2)) * 100).astype(np.float32)
    return pairs, scores, positions


def as_rows(edges):
    """Edges as a sorted list of (source, target) rows, to compare as multisets."""
    return sorted(map(tuple, np.asarray(edges).reshape(-1, 2).tolist()))


class RecordingSegmentsVisual(EdgeSegmentsVisual):
    """EdgeSegmentsVisual that also keeps what it was asked to upload."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uploaded = {}

    def set_segments(self, key, edges):
        self.uploaded[key] = np.array(edges, copy=True)
        super().set_segments(key, edges)

    def drawn(self, key):
        return self.uploaded[key][: self.segment_count]


RecordingSegments = create_visual_node(RecordingSegmentsVisual)


def recording_layer(edges, scores, positions, **kwargs):
    with mock.patch.object(Viewer_Edges, "EdgeSegments", RecordingSegments):
        return ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                           use_position_texture=True, **kwargs)


class EdgeOrderTests(unittest.TestCase):
    def test_prefix_counts_match_the_array_comparison(self):
        edges, scores, _ = random_network(1)
        order = EdgeOrder(edges, scores)
        exact = [float(value) for value in scores[:20]]
        thresholds = exact + [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0,
                              float("inf"), float("-inf"), float("nan")]
        for threshold in thresholds:
            for kind in (float, np.float32, np.float64):
                value = kind(threshold)
                with self.subTest(threshold=threshold, kind=kind.__name__):
                    expected = int((scores >= value).sum())
                    self.assertEqual(order.count_at_or_above(value), expected)
                    np.testing.assert_array_equal(
                        as_rows(order.edges[:expected]), as_rows(edges[scores >= value])
                    )

    def test_float64_scores_compare_in_float64(self):
        edges, scores, _ = random_network(2)
        scores64 = scores.astype(np.float64) + 1e-9
        order = EdgeOrder(edges, scores64)
        for threshold in (0.5, 0.5 + 1e-9, float(scores64[3])):
            self.assertEqual(order.count_at_or_above(threshold), int((scores64 >= threshold).sum()))

    def test_order_is_descending_and_stable(self):
        scores = np.array([0.5, 0.9, -0.0, 0.5, np.inf, 0.0, -2.0, 0.9, -np.inf, 0.5],
                          dtype=np.float32)
        order, ordered = _descending_order(scores)
        # +0.0 sorts before -0.0; the two compare equal, so a threshold keeps both or neither.
        self.assertEqual(order.tolist(), [4, 1, 7, 0, 3, 9, 5, 2, 6, 8])
        np.testing.assert_array_equal(ordered.view(np.uint32), scores[order].view(np.uint32))

    def test_float32_fast_path_matches_the_generic_sort(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=5000).astype(np.float32)
        scores[rng.integers(0, 5000, 800)] = 0.25
        scores[:10] = [np.inf, -np.inf, 0.0, -0.0, np.nan, 1e-40, -1e-40, -np.nan, 3.0, -3.0]
        fast, ordered = _descending_order(scores)
        generic, generic_scores = _descending_order(scores.astype(np.float64))
        np.testing.assert_array_equal(fast, generic)
        self.assertEqual(len(fast), len(scores) - 2)
        # The scores rebuilt from the sort keys are the scores, bit for bit.
        np.testing.assert_array_equal(ordered.view(np.uint32), scores[fast].view(np.uint32))
        np.testing.assert_array_equal(ordered, generic_scores)

    def test_nan_scores_are_never_drawn(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        scores = np.array([np.nan, 0.4, np.nan], dtype=np.float32)
        order = EdgeOrder(edges, scores)
        self.assertEqual(as_rows(order.edges), [(1, 2)])
        self.assertEqual(order.count_at_or_above(float("-inf")), 1)

    def test_without_scores_every_edge_passes(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        for scores in (None, np.empty(0, dtype=np.float32)):
            order = EdgeOrder(edges, scores)
            self.assertEqual(order.count_at_or_above(5.0), 2)
            np.testing.assert_array_equal(order.edges, edges)

    def test_mismatched_scores_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must match the edges"):
            EdgeOrder(np.array([[0, 1], [1, 2]]), np.array([0.5], dtype=np.float32))


class PositionTextureTests(unittest.TestCase):
    def test_texels_hold_each_node_at_its_row_and_column(self):
        for count in (0, 1, TEXTURE_WIDTH, TEXTURE_WIDTH + 1, 2 * TEXTURE_WIDTH + 1,
                      3 * TEXTURE_WIDTH + 5):
            positions = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
            texels = position_texels(positions)
            rows = len(texels)
            with self.subTest(count=count):
                self.assertEqual(texels.shape, (rows, TEXTURE_WIDTH, 4))
                self.assertEqual(texels.dtype, np.float32)
                self.assertEqual(rows & (rows - 1), 0)  # a power of two
                self.assertGreaterEqual(rows * TEXTURE_WIDTH, count)
                self.assertLess((rows // 2) * TEXTURE_WIDTH, max(count, 1))
                for node in ({0, count // 2, count - 1} if count else ()):
                    np.testing.assert_array_equal(
                        texels[node // TEXTURE_WIDTH, node % TEXTURE_WIDTH],
                        [positions[node, 0], positions[node, 1], 0.0, 0.0],
                    )
                self.assertEqual(np.count_nonzero(texels.reshape(-1, 4)[count:]), 0)

    def test_switch_and_node_limit(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(Viewer_Edges.POSITION_TEXTURE_ENVIRONMENT_VARIABLE, None)
            self.assertTrue(position_texture_enabled(1000))
            self.assertTrue(position_texture_enabled((1 << 24) - 1))
            self.assertFalse(position_texture_enabled(1 << 24))
            for value in ("0", "off", "False", " no "):
                os.environ[Viewer_Edges.POSITION_TEXTURE_ENVIRONMENT_VARIABLE] = value
                self.assertFalse(position_texture_enabled(1000), value)
            os.environ[Viewer_Edges.POSITION_TEXTURE_ENVIRONMENT_VARIABLE] = "1"
            self.assertTrue(position_texture_enabled(1000))


class ViewerEdgesTests(unittest.TestCase):
    def test_drawn_edges_are_those_edge_stages_keeps(self):
        edges, scores, positions = random_network(4)
        layer = recording_layer(edges, scores, positions)
        rng = np.random.default_rng(5)
        configuration = SimpleNamespace(UMAP_MODE=False, LOW_RESOURCE_MODE=False)
        for step in range(40):
            visible = rng.random(len(positions)) > 0.25 if step % 3 else np.ones(len(positions), bool)
            threshold = float(rng.choice([0.0, 0.5, float(scores[step]), rng.random(), 1.5]))
            viewer = SimpleNamespace(edges=edges, edge_scores=scores, visible_mask=visible,
                                     n_nodes=len(positions), current_slider_threshold=threshold,
                                     selected_indices=[])
            expected, counts = edge_stages(viewer, configuration)
            with self.subTest(step=step):
                drawn = layer.show_threshold(positions, visible, threshold)
                self.assertEqual(drawn, counts["rendered_edge_count"])
                self.assertEqual(layer.drawn_edge_count, drawn)
                self.assertEqual(as_rows(layer.visual.drawn("threshold")), as_rows(expected))

    def test_threshold_changes_upload_nothing(self):
        edges, scores, positions = random_network(6)
        layer = recording_layer(edges, scores, positions)
        layer.show_threshold(positions, None, 0.0)
        self.assertEqual((layer.upload_count, layer.position_upload_count), (1, 1))
        for threshold in (0.3, 0.5, 0.99, 2.0, 0.0, 0.5):
            drawn = layer.show_threshold(positions, None, threshold)
            self.assertEqual(drawn, int((scores >= threshold).sum()))
            self.assertEqual(layer.visual.segment_count, drawn)
        self.assertEqual((layer.upload_count, layer.position_upload_count), (1, 1))

    def test_moving_nodes_uploads_only_positions(self):
        edges, scores, positions = random_network(7)
        layer = recording_layer(edges, scores, positions)
        layer.show_threshold(positions, None, 0.4)
        moved = positions.copy()
        for step in range(5):
            moved[step] += 3.0
            layer.show_threshold(moved, None, 0.4)
        self.assertEqual(layer.position_upload_count, 6)
        self.assertEqual(layer.upload_count, 1)
        # The same positions again send nothing.
        layer.show_threshold(moved.copy(), None, 0.4)
        self.assertEqual(layer.position_upload_count, 6)

    def test_hiding_nodes_rebuilds_the_edges_once(self):
        edges, scores, positions = random_network(8)
        layer = recording_layer(edges, scores, positions)
        visible = np.ones(len(positions), dtype=bool)
        layer.show_threshold(positions, visible, 0.2)
        visible[:10] = False
        for threshold in (0.2, 0.6, 0.1):
            drawn = layer.show_threshold(positions, visible.copy(), threshold)
            keep = (scores >= threshold) & visible[edges[:, 0]] & visible[edges[:, 1]]
            self.assertEqual(drawn, int(keep.sum()))
        self.assertEqual(layer.upload_count, 2)
        visible[:] = True
        layer.show_threshold(positions, visible, 0.2)
        self.assertEqual(layer.upload_count, 3)
        # A mask with every node visible is the same as no mask.
        layer.show_threshold(positions, None, 0.2)
        self.assertEqual(layer.upload_count, 3)

    def test_explicit_edges_upload_per_array_and_keep_the_threshold_buffer(self):
        edges, scores, positions = random_network(9)
        layer = recording_layer(edges, scores, positions)
        layer.show_threshold(positions, None, 0.3)
        subset = edges[:17]
        self.assertEqual(layer.show_edges(positions, subset), 17)
        layer.show_edges(positions, subset)
        self.assertEqual(layer.upload_count, 2)
        np.testing.assert_array_equal(layer.visual.drawn("explicit"), subset)
        other = edges[20:25]
        self.assertEqual(layer.show_edges(positions, other), 5)
        self.assertEqual(layer.upload_count, 3)
        self.assertEqual(layer.show_edges(positions, edges[:0]), 0)
        self.assertEqual(layer.visual.segment_count, 0)
        # Back to the threshold: its buffer is still on the GPU.
        drawn = layer.show_threshold(positions, None, 0.3)
        self.assertEqual(drawn, int((scores >= 0.3).sum()))
        self.assertEqual(layer.upload_count, 4)
        self.assertNotIn("explicit", layer.visual._buffers)

    def test_non_finite_positions_are_refused_until_fixed(self):
        edges, scores, positions = random_network(10)
        broken = positions.copy()
        broken[3, 1] = np.nan
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 1), 1.0,
                                    use_position_texture=texture)
                for _ in range(2):  # refused every time, not only the first
                    with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                        layer.show_threshold(broken, None, 0.0)
                layer.show_threshold(positions, None, 0.0)
                self.assertEqual(layer.drawn_edge_count, len(edges))
                self.assertEqual(layer.position_upload_count, 1 if texture else 0)

    def test_out_of_range_endpoints_are_refused(self):
        positions = np.zeros((3, 2), dtype=np.float32)
        for bad in ([[0, 3]], [[-1, 0]]):
            with self.assertRaisesRegex(ValueError, "below the node count"):
                ViewerEdges(None, np.array(bad, dtype=np.int32), None, positions,
                            (0, 0, 0, 1), 1.0, use_position_texture=True)

    def test_line_fallback_uploads_the_drawn_coordinates(self):
        edges, scores, positions = random_network(11)
        layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 1), 1.0,
                            use_position_texture=False)
        self.assertFalse(layer.uses_position_texture)
        uploads = []
        original = layer.visual.set_data

        def record(*args, **kwargs):
            uploads.append(kwargs["pos"])
            return original(*args, **kwargs)

        layer.visual.set_data = record
        visible = np.ones(len(positions), dtype=bool)
        visible[:5] = False
        for threshold in (0.4, 0.4, 0.8):
            layer.show_threshold(positions, visible, threshold)
        self.assertEqual(len(uploads), 2)  # the repeated threshold sends nothing
        keep = (scores >= 0.8) & visible[edges[:, 0]] & visible[edges[:, 1]]
        expected = _edge_segment_positions(positions, edges[keep])
        self.assertEqual(sorted(map(tuple, uploads[-1].reshape(-1, 4).tolist())),
                         sorted(map(tuple, expected.reshape(-1, 4).tolist())))
        moved = positions + 1.0
        layer.show_threshold(moved, visible, 0.8)
        self.assertEqual(len(uploads), 3)
        layer.show_edges(moved, edges[:4])
        np.testing.assert_array_equal(uploads[-1], _edge_segment_positions(moved, edges[:4]))

    def test_line_positions_are_finite_contiguous_float32(self):
        source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)[:, ::-1]
        result = _contiguous_line_positions(source)

        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, [[2.0, 1.0], [4.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            _contiguous_line_positions([[0.0, float("nan")]])

    def test_edge_segments_match_per_edge_endpoint_order(self):
        rng = np.random.default_rng(7)
        for columns in (2, 3):
            positions = rng.normal(size=(50, columns)).astype(np.float32)
            edges = rng.integers(0, 50, size=(200, 2)).astype(np.int32)
            reference = []
            for source, target in edges:  # The former draw_network loop.
                reference.append(positions[source])
                reference.append(positions[target])
            result = _edge_segment_positions(positions, edges)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(result.flags.c_contiguous)
            np.testing.assert_array_equal(result, np.asarray(reference))
        empty = _edge_segment_positions(positions, np.empty((0, 2), dtype=np.int32))
        self.assertEqual(empty.shape, (0, 3))


class UpdateEdgesTests(unittest.TestCase):
    def make_viewer(self):
        for name in ("UMAP_MODE", "LOW_RESOURCE_MODE"):
            patcher = mock.patch.object(cfg, name, False)
            patcher.start()
            self.addCleanup(patcher.stop)
        viewer = MainViewer.__new__(MainViewer)
        viewer.n_nodes = 5
        viewer.edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
        viewer.edge_scores = np.array([0.9, 0.9, 0.1, 0.6], dtype=np.float32)
        viewer.pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32)
        viewer.visible_mask = np.ones(5, dtype=bool)
        viewer.current_slider_threshold = 0.5
        viewer.selected_indices = []
        viewer.is_multi_dragging = False
        viewer.edge_layer = recording_layer(viewer.edges, viewer.edge_scores, viewer.pos)
        return viewer

    def test_the_threshold_and_hidden_nodes_choose_the_prefix(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        self.assertTrue(viewer.edge_layer.visible)
        self.assertEqual(as_rows(viewer.edge_layer.visual.drawn("threshold")),
                         [(0, 1), (1, 2), (3, 4)])
        viewer.current_slider_threshold = 0.95
        viewer.update_edges()
        self.assertFalse(viewer.edge_layer.visible)
        viewer.current_slider_threshold = 0.0
        viewer.visible_mask[2] = False
        viewer.update_edges()
        self.assertEqual(as_rows(viewer.edge_layer.visual.drawn("threshold")), [(0, 1), (3, 4)])

    def test_selection_changes_send_nothing_outside_umap_mode(self):
        viewer = self.make_viewer()
        viewer.update_edges()
        for selection in ([0], [0, 2], [1], []):
            viewer.selected_indices = selection
            viewer.update_edges()
        self.assertEqual(viewer.edge_layer.upload_count, 1)
        self.assertEqual(viewer.edge_layer.position_upload_count, 1)

    def test_dragging_nodes_sends_only_positions(self):
        viewer = self.make_viewer()
        viewer.selected_indices = [1]
        viewer.update_edges()
        viewer.is_multi_dragging = True
        for step in range(1, 4):
            viewer.pos[1, 1] = step
            viewer.update_edges()
        self.assertEqual(viewer.edge_layer.position_upload_count, 4)
        self.assertEqual(viewer.edge_layer.upload_count, 1)

    def test_umap_mode_draws_the_selection_edges(self):
        viewer = self.make_viewer()
        viewer.current_slider_threshold = 0.0
        viewer.selected_indices = [0]
        with mock.patch.object(cfg, "UMAP_MODE", True):
            self.assertTrue(edges_follow_selection(viewer, cfg))
            viewer.update_edges()
            self.assertEqual(as_rows(viewer.edge_layer.visual.drawn("explicit")), [(0, 1)])
            first = viewer.edge_layer.upload_count
            viewer.selected_indices = [2]
            viewer.update_edges()
            self.assertGreater(viewer.edge_layer.upload_count, first)
            self.assertEqual(as_rows(viewer.edge_layer.visual.drawn("explicit")), [(1, 2), (2, 3)])
            viewer.selected_indices = []
            viewer.update_edges()
            self.assertFalse(viewer.edge_layer.visible)

    def test_low_resource_drag_leaves_out_the_dragged_nodes_edges(self):
        viewer = self.make_viewer()
        viewer.current_slider_threshold = 0.0
        viewer.selected_indices = [1]
        viewer.is_multi_dragging = True
        with mock.patch.object(cfg, "LOW_RESOURCE_MODE", True):
            self.assertTrue(edges_follow_selection(viewer, cfg))
            viewer.update_edges()
            self.assertEqual(as_rows(viewer.edge_layer.visual.drawn("explicit")), [(2, 3), (3, 4)])
            viewer.is_multi_dragging = False
            self.assertFalse(edges_follow_selection(viewer, cfg))
            viewer.update_edges()
            self.assertEqual(len(viewer.edge_layer.visual.drawn("threshold")), 4)
            # Dragging with nothing selected keeps every edge, as edge_stages does.
            viewer.is_multi_dragging = True
            viewer.selected_indices = []
            self.assertFalse(edges_follow_selection(viewer, cfg))
            viewer.update_edges()
            self.assertEqual(len(viewer.edge_layer.visual.drawn("threshold")), 4)

    def test_no_edge_layer_is_a_no_op(self):
        viewer = self.make_viewer()
        viewer.edge_layer = None
        viewer.update_edges()

    def test_draw_network_builds_the_layer_from_the_edge_settings(self):
        viewer = self.make_viewer()
        viewer.view = SimpleNamespace(scene=object())
        viewer.canvas = SimpleNamespace(scene=object())
        viewer.vispy_ui_face = None
        viewer._tooltip_font_size_points = lambda: 9
        viewer.update_nodes = mock.Mock()
        with mock.patch.object(cfg, "EDGE_COLOR", "#336699"), \
                mock.patch.object(cfg, "EDGE_ALPHA", 0.25), \
                mock.patch.object(cfg, "EDGE_WIDTH", 2.0), \
                mock.patch("EMAPSSN_Viewer.scene"), \
                mock.patch("EMAPSSN_Viewer.ViewerEdges") as layer_class:
            viewer.draw_network()
            args, kwargs = layer_class.call_args
            self.assertIs(args[0], viewer.view.scene)
            self.assertIs(args[1], viewer.edges)
            self.assertIs(args[2], viewer.edge_scores)
            self.assertIs(args[3], viewer.pos)
            self.assertEqual(kwargs["width"], 2.0)
            np.testing.assert_allclose(kwargs["color"], (0.2, 0.4, 0.6, 0.25))
            self.assertIs(viewer.edge_layer, layer_class.return_value)

            viewer.edges = np.empty((0, 2), dtype=np.int32)
            viewer.draw_network()
            self.assertIsNone(viewer.edge_layer)
            self.assertEqual(layer_class.call_count, 1)


_RENDER_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from vispy import app, scene
    from desktop.Viewer_Edges import ViewerEdges, _edge_segment_positions
    app.use_app("pyside6")
    canvas = scene.SceneCanvas(size=(240, 200), bgcolor="white", show=False)
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-5, 105), y=(-5, 105))
    rng = np.random.default_rng(42)
    nodes, count = 3000, 30000
    pos = (rng.random((nodes, 2)) * 100).astype(np.float32)
    edges = rng.integers(0, nodes, (count, 2)).astype(np.int32)
    scores = rng.random(count).astype(np.float32)
    scores[::7] = 0.5
    color = (0.1, 0.2, 0.7, 0.3)
    line = scene.visuals.Line(pos=_edge_segment_positions(pos, edges[:10]), connect="segments",
                              color=color, width=1.5, parent=view.scene)
    line.set_gl_state("translucent", depth_test=False)
    if (canvas.render()[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    layer = ViewerEdges(view.scene, edges, scores, pos, color, 1.5, use_position_texture=True)
    hidden = np.ones(nodes, dtype=bool)
    hidden[rng.choice(nodes, 400, replace=False)] = False
    moved = pos.copy()
    moved[:500] += 6.0
    mismatches = 0
    for visible, threshold, positions in [(None, 0.0, pos), (None, 0.5, pos), (hidden, 0.8, pos),
                                          (hidden, 0.3, moved), (None, 0.95, moved)]:
        drawn = layer.show_threshold(positions, visible, threshold)
        mask = np.ones(nodes, bool) if visible is None else visible
        old_edges = edges[(scores >= threshold) & mask[edges[:, 0]] & mask[edges[:, 1]]]
        layer.visible, line.visible = drawn > 0, False
        new = canvas.render()
        line.set_data(pos=_edge_segment_positions(positions, old_edges))
        layer.visible, line.visible = False, True
        old = canvas.render()
        if drawn != len(old_edges) or not np.array_equal(new, old):
            mismatches += 1
            print("MISMATCH", threshold, drawn, len(old_edges), int((new != old).any(axis=2).sum()))
    print("DONE", mismatches, flush=True)
    """
)


class EdgeShaderPixelTests(unittest.TestCase):
    def test_texture_shader_draws_the_same_pixels_as_the_line_visual(self):
        environment = dict(os.environ)
        # tests/__init__.py defaults Qt to offscreen, which has no OpenGL on
        # Windows; a hidden canvas on the native platform renders.
        environment.pop("QT_QPA_PLATFORM", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, "-c", _RENDER_SCRIPT, SRC_DIR],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=180,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("the GL render probe timed out")
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
        self.assertIn("DONE 0", result.stdout, result.stdout + result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
