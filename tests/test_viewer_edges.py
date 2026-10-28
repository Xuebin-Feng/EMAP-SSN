"""The desktop Viewer's edge layer (desktop/Viewer_Edges.py) and update_edges.

The edges are sorted by descending score once, so a threshold keeps a prefix
of them, and their buffer holds node numbers that the vertex shader resolves
through a texture of node positions. These tests check that the drawn edges
are exactly those edge_stages keeps, and that each kind of change re-sends only
what it must: a threshold change nothing, a node move the position texture,
hiding nodes the edge buffer once. The GL tests render the old Line visual
and the new one on a real GL canvas and compare their pixels, and check that
the stencil overdraw limit (EdgeStencilLimit) leaves every pixel as it was.

Edge Opacity's Auto switch makes each edge's alpha follow its score: the tests
check the grades and the map the shader gets, that nothing is made or uploaded
until it is first on or by a threshold move, that the explicit (selection)
edges and the fallback ignore it, and, on a real GL canvas, the alpha each
edge is drawn with.
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
    SCORE_ALPHA_MAX,
    SCORE_ALPHA_MIN,
    SCORE_GRADES,
    ScoreOpacity,
    TEXTURE_WIDTH,
    ViewerEdges,
    _contiguous_line_positions,
    _descending_order,
    _edge_segment_positions,
    position_texels,
    position_texture_enabled,
    score_grades,
    score_opacity,
)
from desktop.Viewer_Motion import MotionDetail  # noqa: E402


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
        self.uploaded_grades = {}

    def set_segments(self, key, edges, **kwargs):
        self.uploaded[key] = np.array(edges, copy=True)
        super().set_segments(key, edges, **kwargs)

    def _upload_grades(self, key, vertex_grades):
        self.uploaded_grades[key] = np.array(vertex_grades, copy=True)
        super()._upload_grades(key, vertex_grades)

    def drawn(self, key):
        return self.uploaded[key][: self.segment_count]


RecordingSegments = create_visual_node(RecordingSegmentsVisual)


class NoGpuTimer:
    """MotionDetail's GPU timer when there is none to ask: draws are just drawn."""
    available = False

    def time(self, draw, tag, on_result):
        draw()


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
        configuration = SimpleNamespace(UMAP_MODE=False)
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
        for name in ("UMAP_MODE", "LOW_RESOURCE_DRAWING", "FEWER_EDGES_WHILE_MOVING"):
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

    def test_a_node_drag_leaves_no_edge_out_whatever_the_categories(self):
        # Low Resource Mode once hid the dragged nodes' edges; no category does now.
        viewer = self.make_viewer()
        viewer.current_slider_threshold = 0.0
        viewer.selected_indices = [1]
        viewer.is_multi_dragging = True
        for drawing in (False, True):
            for motion in (False, True):
                with self.subTest(drawing=drawing, motion=motion), \
                        mock.patch.object(cfg, "LOW_RESOURCE_DRAWING", drawing), \
                        mock.patch.object(cfg, "FEWER_EDGES_WHILE_MOVING", motion), \
                        mock.patch.object(cfg, "LOW_RESOURCE_ANIMATIONS", True), \
                        mock.patch.object(cfg, "LOW_RESOURCE_WEBSERVER", True):
                    self.assertFalse(edges_follow_selection(viewer, cfg))
                    active, counts = edge_stages(viewer, cfg)
                    self.assertEqual(as_rows(active), [(0, 1), (1, 2), (2, 3), (3, 4)])
                    self.assertEqual(counts["rendered_edge_count"], 4)
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


class ScoreOpacityTests(unittest.TestCase):
    """The grades a score is kept as, and the opacity a grade gets (the shader's uniforms)."""

    def test_the_grades_span_the_range_in_65536_steps(self):
        scores = np.array([0.0, 0.25, 0.5, 1.0, 0.125], dtype=np.float32)
        grades = score_grades(scores, 0.0, 1.0)
        self.assertEqual(grades.dtype, np.uint16)
        self.assertEqual(SCORE_GRADES, 65536)
        self.assertEqual(grades.tolist(), [0, 16384, 32768, 65535, 8192])
        # The range need not start at 0, and scores outside it are clamped.
        outside = score_grades(np.array([-1.0, 2.0, 2.5, 3.0, 9.0]), 2.0, 3.0)
        self.assertEqual(outside.tolist(), [0, 0, 32768, 65535, 65535])

    def test_a_high_threshold_still_has_a_thousand_grades(self):
        # 100 thousand scores over the network's range, a threshold keeping the top 2 %.
        scores = np.linspace(0.0, 1.0, 100_000, dtype=np.float32)
        kept = score_grades(scores[scores >= 0.98], 0.0, 1.0)
        self.assertGreater(len(np.unique(kept)), 1000)

    def test_grades_do_not_depend_on_the_chunk_or_the_dtype(self):
        rng = np.random.default_rng(1)
        scores = rng.normal(size=1000).astype(np.float32)
        whole = score_grades(scores, float(scores.min()), float(scores.max()))
        with mock.patch.object(Viewer_Edges, "_GRADE_CHUNK", 7):
            np.testing.assert_array_equal(
                score_grades(scores, float(scores.min()), float(scores.max())), whole)
        np.testing.assert_array_equal(
            score_grades(scores.astype(np.float64), float(scores.min()), float(scores.max())), whole)

    def test_scores_without_a_span_have_grade_zero(self):
        scores = np.full(5, 0.4, dtype=np.float32)
        for low, high in ((0.4, 0.4), (0.5, 0.4), (float("nan"), 1.0), (0.0, float("inf"))):
            with self.subTest(low=low, high=high):
                self.assertEqual(score_grades(scores, low, high).tolist(), [0] * 5)
        self.assertEqual(len(score_grades(np.empty(0), 0.0, 1.0)), 0)

    def test_the_lowest_score_is_15_percent_and_the_highest_85(self):
        self.assertEqual((SCORE_ALPHA_MIN, SCORE_ALPHA_MAX), (0.15, 0.85))
        opacity = score_opacity(0.2, 0.9, 0.0, 1.0)
        low, high = score_grades(np.array([0.2, 0.9]), 0.0, 1.0)
        alphas = opacity.alphas([low, 0.5 * (int(low) + int(high)), high])
        np.testing.assert_allclose(alphas, [0.15, 0.5, 0.85], atol=2e-5)
        # Straight between them, and held at the ends beyond them.
        np.testing.assert_allclose(opacity.alphas([0, 65535]), [0.15, 0.85])
        grades = np.linspace(low, high, 9)
        np.testing.assert_allclose(np.diff(opacity.alphas(grades)), 0.7 / 8, atol=2e-5)

    def test_the_uniforms_are_four_numbers(self):
        opacity = score_opacity(0.25, 0.75, 0.0, 1.0)
        self.assertIsInstance(opacity, ScoreOpacity)
        self.assertEqual(len(opacity), 4)
        grade_low, grade_scale, alpha_low, alpha_high = opacity
        self.assertAlmostEqual(grade_low, 0.25 * 65535)
        self.assertAlmostEqual(1.0 / grade_scale, 0.5 * 65535)
        self.assertEqual((alpha_low, alpha_high), (0.15, 0.85))

    def test_equal_scores_are_all_85_percent(self):
        for opacity in (score_opacity(0.4, 0.4, 0.4, 0.4), score_opacity(0.4, 0.4, 0.0, 1.0),
                        # A spread of under half a grade is none.
                        score_opacity(0.4, 0.4 + 1e-6, 0.0, 1.0),
                        score_opacity(0.0, 0.0, float("nan"), 1.0)):
            with self.subTest(opacity=opacity):
                np.testing.assert_allclose(opacity.alphas([0, 20000, 65535]), 0.85)


class OpacityByScoreLayerTests(unittest.TestCase):
    """ViewerEdges.set_opacity_by_score: what it makes, when, and what draws."""

    def make(self, count=400, seed=21, **kwargs):
        edges, scores, positions = random_network(seed, edges=count)
        layer = recording_layer(edges, scores, positions, **kwargs)
        return layer, edges, scores, positions

    def grades_of(self, layer):
        scores = layer.order.scores
        return score_grades(scores, scores[-1], scores[0])

    def test_nothing_is_made_until_the_opacity_first_follows_the_score(self):
        layer, _, _, positions = self.make()
        layer.show_threshold(positions, None, 0.3)
        layer.show_threshold(positions, None, 0.6)
        self.assertEqual((layer.score_upload_count, layer.visual.grade_bytes()), (0, 0))
        self.assertIsNone(layer._grades)
        self.assertIsNone(layer.visual._score_opacity)
        self.assertTrue(layer.set_opacity_by_score(True))
        # Asking makes nothing either: the next show_threshold does.
        self.assertEqual((layer.score_upload_count, layer.visual.grade_bytes()), (0, 0))
        layer.show_threshold(positions, None, 0.3)
        self.assertEqual(layer.score_upload_count, 1)
        # A float32 at each of an edge's two vertices.
        self.assertEqual(layer.visual.grade_bytes(), 8 * len(layer.order))
        expected = np.repeat(self.grades_of(layer).astype(np.float32), 2)
        np.testing.assert_array_equal(layer.visual.uploaded_grades["threshold"], expected)
        self.assertIsNotNone(layer.visual._score_opacity)

    def test_a_threshold_move_remaps_and_uploads_nothing(self):
        layer, _, scores, positions = self.make()
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.0)
        counts = (layer.upload_count, layer.score_upload_count, layer.position_upload_count)
        buffers = (layer.visual._buffers["threshold"], layer.visual._grade_buffers["threshold"])
        seen = []
        for threshold in (0.3, 0.5, 0.8, 0.97, 0.0, 0.5):
            layer.show_threshold(positions, None, threshold)
            opacity = layer.visual._score_opacity
            seen.append(opacity)
            self.assertEqual(opacity, layer.opacity_map(threshold))
            # The lowest score at or above the threshold is 15 %, the top score 85 %.
            drawn = scores[scores >= threshold]
            low, high = score_grades(np.array([drawn.min(), drawn.max()]), scores.min(), scores.max())
            np.testing.assert_allclose(opacity.alphas([low, high]), [0.15, 0.85], atol=2e-3)
        self.assertEqual(len(set(seen)), 5)  # a move remaps; the same threshold again is the same map
        self.assertEqual(seen[1], seen[-1])
        self.assertEqual(
            (layer.upload_count, layer.score_upload_count, layer.position_upload_count), counts)
        self.assertIs(layer.visual._buffers["threshold"], buffers[0])
        self.assertIs(layer.visual._grade_buffers["threshold"], buffers[1])

    def thinned(self):
        """A layer that keeps thinned copies of even 400 edges."""
        patcher = mock.patch.object(Viewer_Edges, "MIN_SUBSET_SEGMENTS", 10)
        patcher.start()
        self.addCleanup(patcher.stop)
        return self.make(motion=MotionDetail(enabled=True, gpu_timer=NoGpuTimer()))

    def test_the_shader_gets_the_map_and_the_thinning(self):
        layer, _, scores, positions = self.thinned()
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.4)
        visual = layer.visual
        for k in (1, 4, 16):
            visual.detail = k
            visual._bound = None
            visual._bind()
            self.assertEqual(visual.shared_program["u_thinning"], float(k))
            self.assertEqual(visual.shared_program["u_alpha_map"], tuple(layer.opacity_map(0.4)))
        # The colour's own alpha is not what is drawn.
        self.assertEqual(visual.shared_program["u_color"], (0, 0, 0, 0.5))

    def test_a_thinned_copy_keeps_the_grades_of_its_edges(self):
        layer, _, _, positions = self.thinned()
        layer.show_threshold(positions, None, 0.0)
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.0)
        grades = self.grades_of(layer)
        recorded = layer.visual.uploaded_grades
        np.testing.assert_array_equal(recorded["threshold"], np.repeat(grades.astype(np.float32), 2))
        for k in (4, 16, 64):
            np.testing.assert_array_equal(
                recorded[("threshold", k)], np.repeat(grades[::k].astype(np.float32), 2))
            self.assertEqual(layer.visual._grade_buffers[("threshold", k)].size,
                             layer.visual._buffers[("threshold", k)].size)

    def test_hiding_nodes_rebuilds_the_grades_with_the_edges(self):
        layer, _, _, positions = self.make()
        layer.set_opacity_by_score(True)
        visible = np.ones(len(positions), dtype=bool)
        visible[:12] = False
        # The grades are made after the nodes were hidden, ...
        layer.show_threshold(positions, visible, 0.2)
        grades = self.grades_of(layer)
        edges = layer.order.edges
        keep = visible[edges[:, 0]] & visible[edges[:, 1]]
        np.testing.assert_array_equal(
            layer.visual.uploaded_grades["threshold"], np.repeat(grades[keep].astype(np.float32), 2))
        # ... and follow every later change of what is visible.
        for hidden in (slice(20, 30), slice(0, 0), slice(5, 40)):
            visible[:] = True
            visible[hidden] = False
            layer.show_threshold(positions, visible.copy(), 0.2)
            keep = visible[edges[:, 0]] & visible[edges[:, 1]]
            np.testing.assert_array_equal(
                layer.visual.uploaded_grades["threshold"],
                np.repeat(grades[keep].astype(np.float32), 2))
            self.assertEqual(layer.visual.grade_bytes(), 8 * int(keep.sum()))
        # Hiding costs no new map, and the opacity still comes from the edges at the threshold.
        self.assertEqual(layer.visual._score_opacity, layer.opacity_map(0.2))

    def test_the_opacity_comes_from_the_threshold_not_from_which_nodes_are_hidden(self):
        layer, _, _, positions = self.make()
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.3)
        shown = layer.visual._score_opacity
        visible = np.ones(len(positions), dtype=bool)
        visible[:30] = False
        layer.show_threshold(positions, visible, 0.3)
        self.assertEqual(layer.visual._score_opacity, shown)

    def test_a_selections_edges_are_drawn_at_the_colours_alpha_and_make_no_grades(self):
        layer, edges, _, positions = self.make()
        layer.set_opacity_by_score(True)
        layer.show_edges(positions, edges[:30])
        self.assertEqual((layer.score_upload_count, layer.visual.grade_bytes()), (0, 0))
        self.assertIsNone(layer._grades)
        self.assertIsNone(layer.visual._score_opacity)
        self.assertNotIn("explicit", layer.visual._grade_buffers)
        self.assertEqual(layer.visual.edge_style(), ((0.0, 0.0, 0.0, 0.5), 1.0))
        layer.show_threshold(positions, None, 0.3)
        self.assertEqual(layer.score_upload_count, 1)
        self.assertIsNotNone(layer.visual._score_opacity)
        bytes_ = layer.visual.grade_bytes()
        layer.show_edges(positions, edges[:30])
        layer.show_edges(positions, edges[5:25])
        self.assertIsNone(layer.visual._score_opacity)
        self.assertNotIn("explicit", layer.visual._grade_buffers)
        self.assertEqual((layer.score_upload_count, layer.visual.grade_bytes()), (1, bytes_))

    def test_the_shaders_follow_the_opacity_even_when_it_is_on_from_the_first_show(self):
        # The scene node's own opacity (1.0) is left alone, and a layer that is
        # never shown plain first still gets the shaders of an alpha by score.
        layer, _, _, positions = self.make()
        visual = layer.visual
        self.assertEqual(visual.opacity, 1.0)
        self.assertNotIn("v_alpha", visual.shared_program._fcode)
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.3)
        self.assertIn("v_alpha", visual.shared_program._fcode)
        self.assertIn("a_grade", visual.shared_program._vcode)
        self.assertEqual(visual.opacity, 1.0)
        layer.set_opacity_by_score(False)
        layer.show_threshold(positions, None, 0.3)
        self.assertNotIn("v_alpha", visual.shared_program._fcode)
        self.assertNotIn("a_grade", visual.shared_program._vcode)
        self.assertEqual(visual.opacity, 1.0)

    def test_turning_it_off_draws_plain_and_back_on_uploads_nothing_new(self):
        layer, _, _, positions = self.make()
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.3)
        self.assertFalse(layer.set_opacity_by_score(False))
        layer.show_threshold(positions, None, 0.3)
        self.assertIsNone(layer.visual._score_opacity)
        self.assertEqual(layer.visual.edge_style(), ((0.0, 0.0, 0.0, 0.5), 1.0))
        self.assertIsNone(layer.opacities_of(np.array([0.5]), 0.3))
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.3)
        self.assertIsNotNone(layer.visual._score_opacity)
        self.assertEqual(layer.score_upload_count, 1)

    def test_the_fallback_draws_at_the_sliders_alpha_and_says_why_once(self):
        edges, scores, positions = random_network(11)
        said = []
        layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                            use_position_texture=False, report=said.append)
        for _ in range(3):
            self.assertFalse(layer.set_opacity_by_score(True))
            layer.show_threshold(positions, None, 0.3)
        self.assertEqual(len(said), 1)
        self.assertIn("coordinate fallback", said[0])
        self.assertIn("Edge Opacity", said[0])
        self.assertFalse(layer.opacity_by_score)
        self.assertIsInstance(layer.visual, Viewer_Edges.LimitedLine)
        np.testing.assert_allclose(layer.visual.color, (0, 0, 0, 0.5))
        self.assertIsNone(layer.opacities_of(scores, 0.3))
        # Turning it off says nothing more.
        layer.set_opacity_by_score(False)
        self.assertEqual(len(said), 1)

    def test_edges_without_scores_are_drawn_at_the_sliders_alpha_and_it_says_so(self):
        edges, _, positions = random_network(12)
        said = []
        layer = ViewerEdges(None, edges, None, positions, (0, 0, 0, 0.5), 1.0,
                            use_position_texture=True, report=said.append)
        self.assertFalse(layer.set_opacity_by_score(True))
        self.assertFalse(layer.set_opacity_by_score(True))
        self.assertEqual(len(said), 1)
        self.assertIn("no scores", said[0])
        self.assertEqual(layer.show_threshold(positions, None, 0.3), len(edges))
        self.assertIsNone(layer.visual._score_opacity)

    def test_opacities_of_scores_follow_the_map_the_layer_draws(self):
        layer, _, scores, positions = self.make()
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.4)
        drawn = scores[scores >= 0.4]
        alphas = layer.opacities_of(drawn, 0.4)
        self.assertAlmostEqual(float(alphas.min()), 0.15, delta=2e-3)
        self.assertAlmostEqual(float(alphas.max()), 0.85, delta=1e-6)
        self.assertTrue((np.diff(alphas[np.argsort(drawn)]) >= -1e-12).all())


class OpacityGenerationTests(unittest.TestCase):
    """What the image cache keys a picture by: the opacity map is part of it."""

    def layer(self):
        edges, scores, positions = random_network(31)
        layer = recording_layer(edges, scores, positions)
        layer.show_threshold(positions, None, 0.0)
        return layer, positions

    def test_the_opacity_bumps_the_generation_and_changes_the_style(self):
        layer, positions = self.layer()
        visual = layer.visual
        style = visual.edge_style()
        layer.set_opacity_by_score(True)
        generation = visual.generation
        layer.show_threshold(positions, None, 0.0)
        self.assertGreater(visual.generation, generation)
        self.assertNotEqual(visual.edge_style(), style)
        self.assertEqual(visual.edge_style()[:2], style)
        # A new map is a new picture; the same one again is not.
        for threshold in (0.4, 0.7):
            before = visual.generation, visual.edge_style()
            layer.show_threshold(positions, None, threshold)
            self.assertGreater(visual.generation, before[0])
            self.assertNotEqual(visual.edge_style(), before[1])
        before = visual.generation, visual.edge_style()
        layer.show_threshold(positions, None, 0.7)
        self.assertEqual((visual.generation, visual.edge_style()), before)
        # Turned off, the picture is the plain one again.
        layer.set_opacity_by_score(False)
        layer.show_threshold(positions, None, 0.7)
        self.assertGreater(visual.generation, before[0])
        self.assertEqual(visual.edge_style(), style)

    def test_the_cache_key_changes_with_auto_and_with_the_map(self):
        layer, positions = self.layer()
        visual = layer.visual

        def key():
            # The generation is held still, so that only the style can tell the keys apart.
            proxy = SimpleNamespace(
                generation=3, edge_style=visual.edge_style, stencil_limit=SimpleNamespace(enabled=True),
                transforms=SimpleNamespace(get_transform=lambda map_from, map_to: FakeTransform()))
            canvas = SimpleNamespace(physical_size=(400, 300), pixel_scale=1.0,
                                     bgcolor=SimpleNamespace(rgba=(1.0, 1.0, 1.0, 1.0)))
            return Viewer_Edges._picture_key(proxy, canvas)

        keys = [key()]
        layer.set_opacity_by_score(True)
        layer.show_threshold(positions, None, 0.0)
        keys.append(key())
        layer.show_threshold(positions, None, 0.5)
        keys.append(key())
        layer.set_opacity_by_score(False)
        layer.show_threshold(positions, None, 0.5)
        keys.append(key())
        self.assertEqual(keys[3], keys[0])
        self.assertEqual(len(set(keys[:3])), 3)

    def test_new_grades_bump_the_generation(self):
        layer, positions = self.layer()
        layer.set_opacity_by_score(True)
        generation = layer.visual.generation
        layer.show_threshold(positions, None, 0.0)
        self.assertGreater(layer.visual.generation, generation)
        hidden = np.ones(len(positions), dtype=bool)
        hidden[:5] = False
        generation = layer.visual.generation
        layer.show_threshold(positions, hidden, 0.0)
        self.assertGreater(layer.visual.generation, generation)
        with self.assertRaisesRegex(ValueError, "grades for the"):
            layer.visual.set_grades("threshold", np.zeros(3, dtype=np.uint16))


class FakeGL:
    """vispy.gloo.gl's answers for framebuffer_stencil_bits, set per test.

    attachment maps an attachment name to (object type, stencil size), or to
    an exception to raise; errors lists the glGetError values to return, in
    turn, after which it returns 0.
    """

    GL_RENDERER = 0x1F01

    def __init__(self, binding=0, attachment=None, stencil_bits=0, errors=(), raises=None):
        self.binding = binding
        self.attachment = attachment or {}
        self.stencil_bits = stencil_bits
        self.errors = list(errors)
        self.raises = raises
        self.asked = []

    def glGetParameter(self, name):
        if self.raises is not None:
            raise self.raises
        if name == Viewer_Edges._GL_FRAMEBUFFER_BINDING:
            return self.binding
        if name == Viewer_Edges._GL_STENCIL_BITS:
            return self.stencil_bits
        if name == self.GL_RENDERER:
            return "Fake Renderer"
        raise AssertionError(name)

    def glGetFramebufferAttachmentParameter(self, target, attachment, pname):
        self.asked.append(attachment)
        answer = self.attachment.get(attachment, (Viewer_Edges._GL_NONE, 0))
        if isinstance(answer, Exception):
            raise answer
        kind, size = answer
        return kind if pname == Viewer_Edges._GL_ATTACHMENT_OBJECT_TYPE else size

    def glGetError(self):
        return self.errors.pop(0) if self.errors else 0


GL_RENDERBUFFER = 0x8D41
GL_FRAMEBUFFER_DEFAULT = 0x8218
GL_INVALID_ENUM = 0x0500


class FramebufferStencilBitsTests(unittest.TestCase):
    def test_a_framebuffer_object_is_asked_about_its_stencil_attachment(self):
        gl = FakeGL(binding=7, attachment={Viewer_Edges._GL_STENCIL_ATTACHMENT: (GL_RENDERBUFFER, 8)})
        self.assertEqual(Viewer_Edges.framebuffer_stencil_bits(gl), 8)
        self.assertEqual(set(gl.asked), {Viewer_Edges._GL_STENCIL_ATTACHMENT})

    def test_the_default_framebuffer_is_asked_about_its_stencil_buffer(self):
        gl = FakeGL(binding=0, attachment={Viewer_Edges._GL_STENCIL: (GL_FRAMEBUFFER_DEFAULT, 8)})
        self.assertEqual(Viewer_Edges.framebuffer_stencil_bits(gl), 8)
        self.assertEqual(set(gl.asked), {Viewer_Edges._GL_STENCIL})

    def test_no_stencil_attachment_is_zero_bits(self):
        gl = FakeGL(binding=7, stencil_bits=8)
        self.assertEqual(Viewer_Edges.framebuffer_stencil_bits(gl), 0)

    def test_an_unanswered_question_falls_back_to_the_old_one(self):
        # A driver that refuses the attachment question (GL_INVALID_ENUM).
        gl = FakeGL(binding=0, attachment={Viewer_Edges._GL_STENCIL: (GL_FRAMEBUFFER_DEFAULT, 8)},
                    errors=[GL_INVALID_ENUM], stencil_bits=4)
        self.assertEqual(Viewer_Edges.framebuffer_stencil_bits(gl), 4)

    def test_failures_are_zero_bits_and_leave_no_gl_error_behind(self):
        cases = {
            "both refused": FakeGL(errors=[GL_INVALID_ENUM, GL_INVALID_ENUM, GL_INVALID_ENUM]),
            "raises": FakeGL(raises=RuntimeError("no context")),
            "attachment raises": FakeGL(attachment={Viewer_Edges._GL_STENCIL: RuntimeError("no function")}),
        }
        for name, gl in cases.items():
            with self.subTest(name):
                self.assertEqual(Viewer_Edges.framebuffer_stencil_bits(gl), 0)
                self.assertEqual(gl.errors, [])


class FakeCanvas:
    """What EdgeStencilLimit reads of a canvas: its framebuffer stack and context."""

    def __init__(self, framebuffer=None):
        self._fb_stack = [] if framebuffer is None else [(framebuffer, (0, 0), (10, 10))]
        self.context = SimpleNamespace(flush_commands=mock.Mock())


class EdgeStencilLimitTests(unittest.TestCase):
    def decide(self, bits):
        reports = []
        limit = Viewer_Edges.EdgeStencilLimit(enabled=True, report=reports.append)
        canvas = FakeCanvas()
        bits_mock = mock.Mock(side_effect=bits) if isinstance(bits, Exception) else mock.Mock(return_value=bits)
        with mock.patch.object(Viewer_Edges, "framebuffer_stencil_bits", bits_mock), \
                mock.patch.object(Viewer_Edges, "_gl_renderer", return_value="Fake Renderer"):
            first = limit.applies(canvas)
            second = limit.applies(canvas)
        return first, second, reports, bits_mock, canvas

    def test_the_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(Viewer_Edges.STENCIL_ENVIRONMENT_VARIABLE, None)
            self.assertTrue(Viewer_Edges.stencil_limit_enabled())
            for value in ("0", "off", "False", " no "):
                os.environ[Viewer_Edges.STENCIL_ENVIRONMENT_VARIABLE] = value
                self.assertFalse(Viewer_Edges.stencil_limit_enabled(), value)
                self.assertFalse(Viewer_Edges.EdgeStencilLimit().enabled, value)
            os.environ[Viewer_Edges.STENCIL_ENVIRONMENT_VARIABLE] = "1"
            self.assertTrue(Viewer_Edges.EdgeStencilLimit().enabled)

    def test_a_window_with_8_stencil_bits_draws_with_the_limit_and_says_so_once(self):
        first, second, reports, bits, canvas = self.decide(8)
        self.assertTrue(first)
        self.assertTrue(second)
        bits.assert_called_once()
        canvas.context.flush_commands.assert_called_once()
        self.assertEqual(len(reports), 1)
        self.assertIn("overdraw limit on", reports[0])
        self.assertIn("Fake Renderer", reports[0])

    def test_a_window_with_fewer_bits_or_an_unanswered_question_draws_as_before(self):
        for bits in (0, 1, 4, 7, RuntimeError("no context")):
            with self.subTest(bits=bits):
                first, second, reports, _, _ = self.decide(bits)
                self.assertFalse(first)
                self.assertFalse(second)
                self.assertEqual(len(reports), 1)
                self.assertIn("overdraw limit off", reports[0])
                self.assertIn("Fake Renderer", reports[0])

    def test_each_window_is_asked_for_itself(self):
        limit = Viewer_Edges.EdgeStencilLimit(enabled=True, report=lambda _text: None)
        first, second = FakeCanvas(), FakeCanvas()
        with mock.patch.object(Viewer_Edges, "framebuffer_stencil_bits", side_effect=[8, 0]), \
                mock.patch.object(Viewer_Edges, "_gl_renderer", return_value="Fake"):
            self.assertTrue(limit.applies(first))
            self.assertFalse(limit.applies(second))
            self.assertTrue(limit.applies(first))

    def test_a_framebuffer_object_decides_by_its_own_stencil_buffer(self):
        # canvas.render() (print, captures) attaches colour and depth only.
        limit = Viewer_Edges.EdgeStencilLimit(enabled=True, report=self.fail)
        with mock.patch.object(Viewer_Edges, "framebuffer_stencil_bits", side_effect=AssertionError):
            self.assertFalse(limit.applies(FakeCanvas(SimpleNamespace(stencil_buffer=None))))
            self.assertTrue(limit.applies(FakeCanvas(SimpleNamespace(stencil_buffer=object()))))

    def test_off_or_without_a_canvas_it_never_applies(self):
        limit = Viewer_Edges.EdgeStencilLimit(enabled=False, report=self.fail)
        with mock.patch.object(Viewer_Edges, "framebuffer_stencil_bits", side_effect=AssertionError):
            self.assertFalse(limit.applies(FakeCanvas()))
            self.assertFalse(limit.applies(FakeCanvas(SimpleNamespace(stencil_buffer=object()))))
            self.assertFalse(Viewer_Edges.EdgeStencilLimit(enabled=True, report=self.fail).applies(None))


class DrawWithLimitTests(unittest.TestCase):
    def draw(self, applies, fail=False):
        visual = mock.Mock()
        limit = SimpleNamespace(applies=mock.Mock(return_value=applies))
        drawn = mock.Mock(side_effect=RuntimeError("draw failed") if fail else None)
        with mock.patch.object(Viewer_Edges, "get_current_canvas", return_value="canvas"), \
                mock.patch.object(Viewer_Edges.gloo, "set_state") as set_state:
            if fail:
                with self.assertRaises(RuntimeError):
                    Viewer_Edges._draw_with_limit(visual, limit, drawn)
            else:
                Viewer_Edges._draw_with_limit(visual, limit, drawn)
        limit.applies.assert_called_once_with("canvas")
        drawn.assert_called_once()
        return visual, set_state

    def test_a_limited_draw_counts_in_the_stencil_and_turns_it_off_after(self):
        for fail in (False, True):
            with self.subTest(draw_fails=fail):
                visual, set_state = self.draw(True, fail)
                state = visual.update_gl_state.call_args.kwargs
                self.assertEqual(state["stencil_test"], True)
                self.assertEqual(state["stencil_func"], ("greater", Viewer_Edges.STENCIL_LIMIT, 0xFF))
                self.assertEqual(state["stencil_op"], ("keep", "keep", "incr"))
                self.assertEqual(state["dither"], False)
                set_state.assert_called_once_with(stencil_test=False, dither=True)

    def test_an_unlimited_draw_keeps_the_stencil_test_off(self):
        visual, set_state = self.draw(False)
        visual.update_gl_state.assert_called_once_with(stencil_test=False)
        set_state.assert_not_called()

    def test_the_limit_counts_to_what_8_bits_hold(self):
        self.assertEqual(Viewer_Edges.STENCIL_LIMIT, 255)
        self.assertEqual(Viewer_Edges.STENCIL_BITS_NEEDED, 8)

    def test_both_edge_visuals_draw_with_the_layers_limit(self):
        edges, scores, positions = random_network(9)
        limit = Viewer_Edges.EdgeStencilLimit(enabled=True, report=lambda _text: None)
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                                    use_position_texture=texture, stencil_limit=limit)
                self.assertIs(layer.stencil_limit, limit)
                self.assertIs(layer.visual.stencil_limit, limit)
                expected = Viewer_Edges.EdgeSegmentsVisual if texture else Viewer_Edges.LimitedLine
                self.assertIsInstance(layer.visual, expected)
                with mock.patch.object(Viewer_Edges, "_draw_with_limit") as draw:
                    layer.visual.draw()
                self.assertIs(draw.call_args.args[0], layer.visual)
                self.assertIs(draw.call_args.args[1], limit)
        default = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0)
        self.assertIsInstance(default.stencil_limit, Viewer_Edges.EdgeStencilLimit)

    def test_the_viewer_canvas_asks_for_an_8_bit_stencil_buffer(self):
        import ast
        import EMAPSSN_Viewer
        with open(EMAPSSN_Viewer.__file__, encoding="utf-8") as source:
            tree = ast.parse(source.read())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and getattr(node.func, "attr", None) == "SceneCanvas"]
        self.assertEqual(len(calls), 1)
        config = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "config")
        self.assertEqual(ast.literal_eval(config.keywords[0].value), Viewer_Edges.STENCIL_BITS_NEEDED)
        self.assertEqual(config.keywords[0].arg, "stencil_size")


class EdgeGenerationTests(unittest.TestCase):
    """Every change to what the edges draw bumps their visual's generation,
    so the image cache draws a new picture; what leaves it alone does not."""

    def layer(self, texture):
        edges, scores, positions = random_network(11)
        layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                            use_position_texture=texture)
        layer.show_threshold(positions, None, 0.0)
        layer.visible = True
        return layer, positions

    def assert_bumps(self, layer, change, bumps=True):
        before = layer.visual.generation
        change()
        if bumps:
            self.assertGreater(layer.visual.generation, before)
        else:
            self.assertEqual(layer.visual.generation, before)

    def test_changes_to_the_drawn_edges_bump_it(self):
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer, positions = self.layer(texture)
                moved = positions.copy()
                moved[3] += 1.0
                hidden = np.ones(len(positions), dtype=bool)
                hidden[5] = False
                explicit = np.array([[0, 1], [2, 3]], dtype=np.int32)
                self.assert_bumps(layer, lambda: layer.show_threshold(positions, None, 0.5))
                self.assert_bumps(layer, lambda: layer.show_threshold(moved, None, 0.5))
                self.assert_bumps(layer, lambda: layer.show_threshold(moved, hidden, 0.5))
                self.assert_bumps(layer, lambda: layer.show_edges(moved, explicit))
                self.assert_bumps(layer, lambda: layer.show_threshold(moved, hidden, 0.5))
                self.assert_bumps(layer, lambda: setattr(layer, "visible", False))
                self.assert_bumps(layer, lambda: setattr(layer, "visible", True))

    def test_new_segments_of_the_same_count_bump_it(self):
        # A UMAP selection swapped for another with as many edges.
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer, positions = self.layer(texture)
                layer.show_edges(positions, np.array([[0, 1], [2, 3]], dtype=np.int32))
                self.assert_bumps(layer, lambda: layer.show_edges(
                    positions, np.array([[4, 5], [6, 7]], dtype=np.int32)))
        layer, _ = self.layer(True)
        self.assert_bumps(layer, lambda: layer.visual.set_segments("threshold", np.array([[1, 2]])))
        self.assert_bumps(layer, lambda: layer.visual.drop_segments("threshold"))

    def test_the_same_edges_again_leave_it(self):
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer, positions = self.layer(texture)
                self.assert_bumps(layer, lambda: layer.show_threshold(positions.copy(), None, 0.0), False)
                self.assert_bumps(layer, lambda: setattr(layer, "visible", True), False)

    def test_colour_alpha_and_width_bump_it(self):
        layer, _ = self.layer(True)
        self.assert_bumps(layer, lambda: layer.visual.set_color((0, 0, 0, 0.25)))
        self.assert_bumps(layer, lambda: layer.visual.set_color((1, 0, 0, 0.25)))
        self.assert_bumps(layer, lambda: layer.visual.set_width(2.0))
        self.assertEqual(layer.visual.edge_style(), ((1.0, 0.0, 0.0, 0.25), 2.0))
        line, _ = self.layer(False)
        self.assert_bumps(line, lambda: line.visual.set_data(color=(0, 0, 0, 0.25)))
        self.assert_bumps(line, lambda: line.visual.set_data(width=2.0))

    def test_a_hover_or_an_unchanged_selection_leaves_it(self):
        viewer = UpdateEdgesTests.make_viewer(self)
        viewer.edge_layer = ViewerEdges(None, viewer.edges, viewer.edge_scores, viewer.pos,
                                        (0, 0, 0, 0.5), 1.0, use_position_texture=True)
        viewer.update_edges()
        viewer.n_nodes = len(viewer.pos)
        viewer.node_render_order = np.arange(viewer.n_nodes, dtype=np.int32)
        viewer.selected_node_idx = None
        viewer.left_click_highlight_indices = None
        viewer.current_colors = np.zeros((viewer.n_nodes, 4), dtype=np.float32)
        viewer.current_sizes = np.ones(viewer.n_nodes, dtype=np.float32)
        viewer.current_shapes = np.full(viewer.n_nodes, "disc", dtype=object)
        from vispy import scene
        viewer.markers = scene.visuals.Markers()
        viewer.canvas = SimpleNamespace(update=mock.Mock())
        viewer._update_hud_elements = mock.Mock()
        viewer._refresh_session_card = mock.Mock()
        viewer.broadcast_event = mock.Mock()
        layer = viewer.edge_layer
        for node in (2, 3, None):
            viewer.hovered_node_idx = node
            self.assert_bumps(layer, viewer.update_nodes, False)
        self.assert_bumps(layer, viewer.update_selection_visual, False)
        viewer.current_slider_threshold = 0.0
        self.assert_bumps(layer, viewer.update_edges)


class FakeTransform:
    def __init__(self, scale=1.0, shift=0.0):
        self.scale, self.shift = scale, shift

    def map(self, points):
        points = np.asarray(points, dtype=float)
        return np.column_stack([points * self.scale + self.shift,
                                np.zeros(len(points)), np.ones(len(points))])


class PictureKeyTests(unittest.TestCase):
    def setUp(self):
        self.transform = FakeTransform()
        self.visual = SimpleNamespace(
            generation=3, edge_style=lambda: ((0, 0, 0, 0.1), 1.0),
            transforms=SimpleNamespace(get_transform=lambda map_from, map_to: self.transform),
            stencil_limit=SimpleNamespace(enabled=True),
        )
        self.canvas = SimpleNamespace(physical_size=(400, 300), pixel_scale=2.0,
                                      bgcolor=SimpleNamespace(rgba=(1.0, 1.0, 1.0, 1.0)))

    def key(self):
        return Viewer_Edges._picture_key(self.visual, self.canvas)

    def test_the_key_holds_still_while_nothing_changes(self):
        self.assertEqual(self.key(), self.key())

    def test_every_input_changes_the_key(self):
        changes = {
            "generation": lambda: setattr(self.visual, "generation", 4),
            "style": lambda: setattr(self.visual, "edge_style", lambda: ((0, 0, 0, 0.2), 1.0)),
            "camera pan": lambda: setattr(self.transform, "shift", 0.5),
            "camera zoom": lambda: setattr(self.transform, "scale", 1.5),
            "framebuffer size": lambda: setattr(self.canvas, "physical_size", (800, 600)),
            "display scale": lambda: setattr(self.canvas, "pixel_scale", 1.0),
            "background": lambda: setattr(self.canvas, "bgcolor", SimpleNamespace(rgba=(0, 0, 0, 1))),
            "overdraw limit": lambda: setattr(self.visual.stencil_limit, "enabled", False),
            "limit value": lambda: mock.patch.object(Viewer_Edges, "STENCIL_LIMIT", 64).start(),
        }
        for name, change in changes.items():
            with self.subTest(name):
                self.setUp()
                before = self.key()
                change()
                self.assertNotEqual(self.key(), before)
                mock.patch.stopall()


class FakeGlooObject:
    """A VisPy GL object that records whether it was deleted."""

    made = []

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.deleted = False
        self.stencil_buffer = None
        self.values = {}
        FakeGlooObject.made.append(self)

    def delete(self):
        self.deleted = True

    def __setitem__(self, name, value):
        self.values[name] = value

    def draw(self, mode):
        self.values["drawn"] = self.values.get("drawn", 0) + 1


class FakeGloo:
    Texture2D = FrameBuffer = RenderBuffer = VertexBuffer = Program = FakeGlooObject

    def __init__(self):
        self.states = []

    def set_state(self, *args, **kwargs):
        self.states.append((args, kwargs))


class FakeWindowCanvas:
    """What the image cache uses of a canvas, drawing into its window."""

    def __init__(self, physical_size=(400, 300), refuse=()):
        self.physical_size = physical_size
        self.size = (200, 150)
        self.bgcolor = "white"
        self._fb_stack = []
        self.refuse = set(refuse)  # stencil formats whose framebuffer is incomplete
        self.closers = []
        self.events = SimpleNamespace(close=SimpleNamespace(connect=self.closers.append))
        self.context = SimpleNamespace(flush_commands=self.flush, clear=mock.Mock())
        self.pushed = []

    def push_fbo(self, fbo, offset, size):
        self._fb_stack.append((fbo, offset, size))
        self.pushed.append(fbo)

    def pop_fbo(self):
        self._fb_stack.pop()

    def flush(self):
        if self._fb_stack:
            stencil = self._fb_stack[-1][0].stencil_buffer
            kind = None if stencil is None else stencil.kwargs.get("format")
            if kind in self.refuse:
                raise RuntimeError(f"framebuffer incomplete with {kind}")


class EdgeImageCacheTests(unittest.TestCase):
    def setUp(self):
        FakeGlooObject.made = []
        self.gloo = FakeGloo()
        self.gl = SimpleNamespace(glGetError=mock.Mock(return_value=0), GL_RENDERER=0x1F01,
                                  glGetParameter=lambda name: "Fake Renderer")
        for patcher in (mock.patch.object(Viewer_Edges, "gloo", self.gloo),
                        mock.patch.object(Viewer_Edges, "_gl", lambda: self.gl)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.reports = []
        self.cache = Viewer_Edges.EdgeImageCache(enabled=True, report=self.reports.append)

    def objects(self, kind):
        return [item for item in FakeGlooObject.made if item.kwargs.get("internalformat") == kind]

    def test_it_serves_only_the_window(self):
        canvas = FakeWindowCanvas()
        self.assertTrue(self.cache.serves(canvas))
        canvas._fb_stack.append((object(), (0, 0), (1, 1)))  # canvas.render(): print, captures
        self.assertFalse(self.cache.serves(canvas))
        self.assertFalse(self.cache.serves(None))
        self.assertFalse(Viewer_Edges.EdgeImageCache(enabled=False).serves(FakeWindowCanvas()))
        with mock.patch.dict(os.environ, {Viewer_Edges.CACHE_ENVIRONMENT_VARIABLE: "0"}):
            self.assertFalse(Viewer_Edges.EdgeImageCache().enabled)

    def test_a_key_draws_its_picture_once_and_frames_copy_it(self):
        canvas = FakeWindowCanvas()
        draw_edges = mock.Mock()
        for key in ("a", "a", "a", "b", "b"):
            self.assertTrue(self.cache.draw(canvas, key, draw_edges))
        self.assertEqual(draw_edges.call_count, 2)
        self.assertEqual((self.cache.render_count, self.cache.reuse_count), (2, 3))
        picture = [item for item in FakeGlooObject.made if "drawn" in item.values]
        self.assertEqual(picture[0].values["drawn"], 5)
        self.assertEqual(picture[0].values["u_size"], (400.0, 300.0))
        # Blending off for the copy, then the state the edges leave behind.
        self.assertEqual(self.gloo.states[0][1]["blend"], False)
        self.assertEqual(self.gloo.states[1], (("translucent",), {"depth_test": False, "dither": True}))
        texture = self.objects("rgba8")
        self.assertEqual(len(texture), 1)
        self.assertEqual(texture[0].kwargs["shape"], (300, 400, 4))
        self.assertEqual(self.cache.stencil_kind, "8-bit stencil")
        self.assertEqual(self.cache.bytes, 400 * 300 * 5)
        self.assertEqual(len(self.reports), 1)
        self.assertIn("image cache on", self.reports[0])

    def test_a_resize_frees_the_old_picture_and_draws_a_new_one(self):
        canvas = FakeWindowCanvas()
        draw_edges = mock.Mock()
        self.cache.draw(canvas, "a", draw_edges)
        old = self.objects("rgba8")[0]
        canvas.physical_size = (800, 600)
        self.cache.draw(canvas, "a", draw_edges)
        self.assertTrue(old.deleted)
        self.assertEqual(self.objects("rgba8")[-1].kwargs["shape"], (600, 800, 4))
        self.assertEqual(draw_edges.call_count, 2)

    def test_closing_the_window_frees_the_picture(self):
        canvas = FakeWindowCanvas()
        self.cache.draw(canvas, "a", mock.Mock())
        texture = self.objects("rgba8")[0]
        self.assertEqual(len(canvas.closers), 1)
        canvas.closers[0](None)
        self.assertTrue(texture.deleted)
        self.assertEqual(self.cache.bytes, 0)

    def test_a_refused_stencil_falls_back_to_a_packed_one_then_none(self):
        for refused, kind in (({"stencil"}, "depth24/stencil8"), ({"stencil", 0x88F0}, "no stencil")):
            with self.subTest(refused=refused):
                self.setUp()
                self.assertTrue(self.cache.draw(FakeWindowCanvas(refuse=refused), "a", mock.Mock()))
                self.assertEqual(self.cache.stencil_kind, kind)
                self.assertFalse(self.cache.failed)

    def test_failures_draw_directly_from_then_on_and_say_so_once(self):
        def refuse_all():
            return FakeWindowCanvas(refuse={"stencil", 0x88F0, None})

        def out_of_memory():
            # GL_OUT_OF_MEMORY while the edges are drawn into the picture.
            self.gl.glGetError.side_effect = [0x0505, 0]
        cases = {
            "no framebuffer": (refuse_all, lambda: None),
            "texture refused": (FakeWindowCanvas, lambda: None),
            "edges fail to draw": (FakeWindowCanvas, mock.Mock(side_effect=RuntimeError("draw failed"))),
            "GL error": (FakeWindowCanvas, out_of_memory),
        }
        for name, (make_canvas, draw_edges) in cases.items():
            with self.subTest(name):
                self.setUp()
                canvas = make_canvas()
                if name == "texture refused":
                    self.gloo.Texture2D = mock.Mock(side_effect=MemoryError("out of memory"))
                self.assertFalse(self.cache.draw(canvas, "a", draw_edges))
                self.assertTrue(self.cache.failed)
                self.assertFalse(self.cache.serves(FakeWindowCanvas()))
                said = [text for text in self.reports if "image cache off" in text]
                self.assertEqual(len(said), 1)
                self.assertIn("Fake Renderer", said[0])
                self.assertTrue(all(item.deleted for item in self.objects("rgba8")))
                self.assertEqual(canvas._fb_stack, [])

    def test_a_gl_error_after_the_copy_keeps_the_frame_and_stops_the_cache(self):
        canvas = FakeWindowCanvas()
        self.cache.draw(canvas, "a", mock.Mock())
        self.gl.glGetError.side_effect = [0x0505, 0]
        self.assertTrue(self.cache.draw(canvas, "a", mock.Mock()))
        self.assertTrue(self.cache.failed)
        self.assertIn("image cache off", self.reports[-1])

    def test_draw_edges_uses_the_cache_when_it_serves(self):
        visual = SimpleNamespace(image_cache=mock.Mock(), stencil_limit=None)
        drawn = mock.Mock()
        with mock.patch.object(Viewer_Edges, "get_current_canvas", return_value="canvas"), \
                mock.patch.object(Viewer_Edges, "_picture_key", return_value="key"), \
                mock.patch.object(Viewer_Edges, "_draw_with_limit") as direct:
            for serves, copied, direct_draws in ((True, True, 0), (True, False, 1), (False, None, 2)):
                visual.image_cache.serves.return_value = serves
                visual.image_cache.draw.return_value = copied
                Viewer_Edges._draw_edges(visual, drawn)
                self.assertEqual(direct.call_count, direct_draws)
            self.assertEqual(visual.image_cache.draw.call_count, 2)
            self.assertEqual(visual.image_cache.draw.call_args.args[:2], ("canvas", "key"))
            # A cached picture is drawn with the overdraw limit too.
            visual.image_cache.draw.call_args.args[2]()
            direct.assert_called_with(visual, None, drawn)

    def test_both_edge_visuals_draw_through_the_layers_cache(self):
        edges, scores, positions = random_network(12)
        cache = Viewer_Edges.EdgeImageCache(enabled=True, report=lambda _text: None)
        for texture in (True, False):
            with self.subTest(position_texture=texture):
                layer = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0,
                                    use_position_texture=texture, image_cache=cache)
                self.assertIs(layer.image_cache, cache)
                self.assertIs(layer.visual.image_cache, cache)
        default = ViewerEdges(None, edges, scores, positions, (0, 0, 0, 0.5), 1.0)
        self.assertIsInstance(default.image_cache, Viewer_Edges.EdgeImageCache)


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


_STENCIL_SCRIPT = textwrap.dedent(
    """
    import sys
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from vispy import app, gloo, scene
    from vispy.gloo import gl
    from desktop import Viewer_Edges as VE
    app.use_app("pyside6")
    canvas = scene.SceneCanvas(size=(200, 160), bgcolor="white", show=False,
                               config=dict(stencil_size=8))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-5, 105), y=(-5, 105))
    rng = np.random.default_rng(42)
    # Two tight groups joined by 2000 edges, so the pixels between them lie
    # under all of them; a 50-node clique; loose edges over everything.
    pos = np.vstack([rng.normal((20, 50), 1.5, (30, 2)), rng.normal((80, 52), 1.5, (30, 2)),
                     rng.normal((50, 20), 6, (50, 2)), rng.random((150, 2)) * 100]).astype(np.float32)
    ci, cj = np.triu_indices(50, 1)
    edges = np.vstack([np.stack([rng.integers(0, 30, 2000), rng.integers(30, 60, 2000)], axis=1),
                       np.stack([ci + 60, cj + 60], axis=1),
                       rng.integers(110, 260, (400, 2))]).astype(np.int32)
    scores = rng.random(len(edges)).astype(np.float32)
    limit = VE.EdgeStencilLimit(enabled=True, report=lambda text: None)
    layers = {texture: VE.ViewerEdges(view.scene, edges, scores, pos, (0, 0, 0, 0.1), 1.0,
                                      use_position_texture=texture, stencil_limit=limit)
              for texture in (True, False)}
    for layer in layers.values():
        layer.show_threshold(pos, None, -1.0)
        layer.visible = False
    # Nodes drawn over the saturated middle of the bundle and over the clique.
    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_data(pos=np.array([[50, 51], [44, 51], [56, 51], [50, 20]], dtype=np.float32),
                     face_color=(0.27, 0.53, 1.0, 1.0), edge_color=(0, 0, 0, 1), size=10, edge_width=0.5)
    markers.set_gl_state("translucent", depth_test=False)
    layers[True].visible = True
    if (canvas.render()[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    print("SCALE", canvas.pixel_scale, flush=True)
    shape = canvas.render().shape[:2]
    plain = gloo.FrameBuffer(color=gloo.RenderBuffer(shape), depth=gloo.RenderBuffer(shape))
    stenciled = gloo.FrameBuffer(color=gloo.RenderBuffer(shape))
    packed = gloo.RenderBuffer(shape, format=0x88F0)  # GL_DEPTH24_STENCIL8
    stenciled.depth_buffer = packed
    stenciled.stencil_buffer = packed

    def bits(fbo):
        canvas.push_fbo(fbo, (0, 0), canvas.size)
        canvas.context.flush_commands()
        result = VE.framebuffer_stencil_bits(gl)
        canvas.pop_fbo()
        canvas.context.flush_commands()
        return result

    print("BITS", bits(plain), bits(stenciled), flush=True)

    def draw(fbo, bg):
        canvas.push_fbo(fbo, (0, 0), canvas.size)
        try:
            canvas._draw_scene(bgcolor=bg)
            return fbo.read()
        finally:
            canvas.pop_fbo()

    colours = {"black": (0, 0, 0), "grey": (0.5, 0.5, 0.5), "orange": (0.9, 0.3, 0.1), "white": (1, 1, 1)}
    cases = inexact = engaged = contrasting = printed_differ = 0
    for texture, layer in layers.items():
        for other in layers.values():
            other.visible = other is layer
        for alpha in (0.01, 0.1, 0.5, 1.0):
            for name, rgb in colours.items():
                if texture:
                    layer.visual.set_color((*rgb, alpha))
                else:
                    layer.visual.set_data(color=(*rgb, alpha))
                for bg in ("white", "black"):
                    cases += 1
                    limit.enabled = False
                    everything = draw(stenciled, bg)
                    printed_everything = canvas.render(bgcolor=bg)
                    limit.enabled = True
                    limited = draw(stenciled, bg)
                    if (limited != everything).any():
                        inexact += 1
                        print("INEXACT", texture, alpha, name, bg, int((limited != everything).any(axis=2).sum()))
                    with mock.patch.object(VE, "STENCIL_LIMIT", 8):
                        # A limit too low to be exact shows the stencil really counts.
                        if alpha == 0.1 and (name, bg) not in {("white", "white"), ("black", "black")}:
                            contrasting += 1
                            engaged += bool((draw(stenciled, bg) != everything).any())
                        # canvas.render(), as print draws, has no stencil buffer: untouched.
                        if (canvas.render(bgcolor=bg) != printed_everything).any():
                            printed_differ += 1
    print("STENCIL", cases, inexact, engaged, contrasting, printed_differ, flush=True)
    """
)


def run_gl_script(test, script, scale):
    """Run script in a subprocess on the native GL at display scale (1 or 2)
    and return its output, or skip test when it gets no OpenGL context."""
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    for name in (Viewer_Edges.STENCIL_ENVIRONMENT_VARIABLE, Viewer_Edges.CACHE_ENVIRONMENT_VARIABLE):
        environment.pop(name, None)
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


_CACHE_SCRIPT = textwrap.dedent(
    """
    import sys
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from vispy import app, gloo, scene
    from vispy.gloo import gl
    from desktop import Viewer_Edges as VE
    app.use_app("pyside6")
    canvas = scene.SceneCanvas(size=(200, 160), bgcolor="white", show=False,
                               config=dict(stencil_size=8))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(-5, 105), y=(-5, 105))
    rng = np.random.default_rng(42)
    pos = np.vstack([rng.normal((20, 50), 1.5, (30, 2)), rng.normal((80, 52), 1.5, (30, 2)),
                     rng.random((150, 2)) * 100]).astype(np.float32)
    edges = np.vstack([np.stack([rng.integers(0, 30, 2000), rng.integers(30, 60, 2000)], axis=1),
                       rng.integers(60, 210, (400, 2))]).astype(np.int32)
    scores = rng.random(len(edges)).astype(np.float32)
    reports = []
    # The settings some saved profiles use: grey edges 1.5 wide at alpha 0.1,
    # nodes 15 across with 2-pixel outlines.
    cache = VE.EdgeImageCache(enabled=True, report=reports.append)
    limit = VE.EdgeStencilLimit(enabled=True, report=reports.append)
    layers = {texture: VE.ViewerEdges(view.scene, edges, scores, pos, (0.5, 0.5, 0.5, 0.1), 1.5,
                                      use_position_texture=texture, stencil_limit=limit,
                                      image_cache=cache)
              for texture in (True, False)}
    for layer in layers.values():
        layer.show_threshold(pos, None, -1.0)
        layer.visible = False
    markers = scene.visuals.Markers(parent=view.scene)
    colours = np.tile([0.27, 0.53, 1.0, 1.0], (len(pos), 1)).astype(np.float32)

    def set_nodes():
        markers.set_data(pos=pos, face_color=colours, edge_color=(0, 0, 0, 1), size=15, edge_width=2.0)

    set_nodes()
    markers.set_gl_state("translucent", depth_test=False)
    hud = scene.visuals.Text("Hidden Nodes: 0", pos=(150, 140), color="grey", parent=canvas.scene)
    layers[True].visible = True
    if (canvas.render()[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    print("SCALE", canvas.pixel_scale, flush=True)
    shape = canvas.render().shape[:2]
    # The window's own framebuffer, as QOpenGLWidget makes it: colour plus a
    # packed depth/stencil buffer. Frames drawn into it count as the window's.
    window = gloo.FrameBuffer(color=gloo.RenderBuffer(shape))
    packed = gloo.RenderBuffer(shape, format=0x88F0)
    window.depth_buffer = packed
    window.stencil_buffer = packed

    def frame(cached=True):
        enabled, cache.enabled = cache.enabled, cached
        with mock.patch.object(VE, "_draws_into_window", lambda canvas: True):
            canvas.push_fbo(window, (0, 0), canvas.size)
            try:
                canvas._draw_scene()
                return window.read()
            finally:
                canvas.pop_fbo()
                cache.enabled = enabled

    results = []

    def check(name):
        before = cache.render_count
        cached = frame()
        drew = cache.render_count - before
        direct = frame(cached=False)
        results.append((name, drew, int((cached != direct).any(axis=2).sum())))

    for texture, layer in layers.items():
        for other in layers.values():
            other.visible = other is layer
        check("first")
        check("unchanged")
        colours[7] = (1.0, 0.67, 0.0, 1.0)
        set_nodes()
        check("hover")
        colours[7] = (0.27, 0.53, 1.0, 1.0)
        set_nodes()
        hud.text = "Hidden Nodes: 3"
        check("hud text")
        rect = view.camera.rect
        view.camera.rect = (rect.left + 4, rect.bottom, rect.width, rect.height)
        check("camera pan")
        view.camera.rect = (rect.left + 4, rect.bottom, rect.width * 0.7, rect.height * 0.7)
        check("camera zoom")
        layer.show_threshold(pos, None, 0.5)
        check("threshold")
        moved = pos.copy()
        moved[:20] += 4.0
        layer.show_threshold(moved, None, 0.5)
        check("node positions")
        hidden = np.ones(len(pos), dtype=bool)
        hidden[40:55] = False
        layer.show_threshold(moved, hidden, 0.5)
        check("hidden nodes")
        layer.show_edges(moved, edges[:900])
        check("explicit edges")
        if texture:
            layer.visual.set_color((0.2, 0.2, 0.2, 0.3))
        else:
            layer.visual.set_data(color=(0.2, 0.2, 0.2, 0.3))
        check("colour and alpha")
        if texture:
            layer.visual.set_width(2.5)
        else:
            layer.visual.set_data(width=2.5)
        check("width")
        layer.visible = False
        check("edges hidden")
        layer.visible = True
        check("edges shown")
        canvas.bgcolor = "black"
        check("background")
        canvas.bgcolor = "white"
        limit.enabled = False
        check("overdraw limit off")
        limit.enabled = True
        check("overdraw limit on")
        # Captures and print draw through canvas.render(): never the cache.
        counts = (cache.render_count, cache.reuse_count)
        printed = canvas.render()
        cache.enabled = False
        printed_direct = canvas.render()
        cache.enabled = True
        results.append(("capture", int(counts != (cache.render_count, cache.reuse_count)),
                        int((printed != printed_direct).any(axis=2).sum())))
    # Edge Opacity's Auto switch: the picture the cache holds is drawn with the alpha by score,
    # and a new threshold, new hidden nodes or Auto turned off draw a new one.
    layer = layers[True]
    for other in layers.values():
        other.visible = other is layer
    layer.visual.set_color((0.5, 0.5, 0.5, 0.1))
    layer.show_threshold(pos, None, -1.0)
    layer.set_opacity_by_score(True)
    layer.show_threshold(pos, None, -1.0)
    check("auto on")
    layer.show_threshold(pos, None, 0.5)
    check("auto threshold")
    hidden = np.ones(len(pos), dtype=bool)
    hidden[40:55] = False
    layer.show_threshold(pos, hidden, 0.5)
    check("auto hidden")
    layer.set_opacity_by_score(False)
    layer.show_threshold(pos, hidden, 0.5)
    check("auto off")
    for name, drew, differ in results:
        print("CASE", name.replace(" ", "_"), drew, differ, flush=True)

    # Failures: no framebuffer this GL accepts, then a GL error while the
    # picture is drawn. Either way the frame is drawn directly, and said once.
    layer = layers[True]
    for other in layers.values():
        other.visible = other is layer
    for failure in ("framebuffer", "gl error"):
        cache = VE.EdgeImageCache(enabled=True, report=reports.append)
        layer.visual.image_cache = cache
        reported = len(reports)
        if failure == "framebuffer":
            patch = mock.patch.object(VE, "_CACHE_STENCILS", ((0x1234, 1),))
        else:
            original = VE._draw_with_limit

            def failing(visual, limit, draw, original=original):
                original(visual, limit, draw)
                top = canvas._fb_stack[-1][0] if canvas._fb_stack else None
                if top is not None and top is cache._framebuffer:
                    gl.glEnable(0x1234)  # GL_INVALID_ENUM, inside the picture's draw
            patch = mock.patch.object(VE, "_draw_with_limit", failing)
        with patch:
            fallen = frame()
        direct = frame(cached=False)
        said = sum("image cache off" in text for text in reports[reported:])
        print("FAIL", failure.replace(" ", "_"), int(cache.failed), said,
              int((fallen != direct).any(axis=2).sum()), int(cache.bytes), flush=True)
    """
)


class EdgeImageCachePixelTests(unittest.TestCase):
    """The image cache on a real GL canvas, at display scales 1 and 2.

    Frames drawn into a framebuffer like the window's (colour plus packed
    depth/stencil) with the cache equal those drawn without it, pixel for
    pixel. A hover or a HUD change copies the picture without drawing the
    edges; every change to the camera, the edges, their style, the
    background or the overdraw limit draws a new picture; canvas.render()
    never touches the cache; and a failure draws directly and says so once.
    """

    TRIGGERS = ("camera_pan", "camera_zoom", "threshold", "node_positions", "hidden_nodes",
                "explicit_edges", "colour_and_alpha", "width", "edges_shown", "background",
                "overdraw_limit_off", "overdraw_limit_on")
    COPIES = ("unchanged", "hover", "hud_text")
    AUTO = ("auto_on", "auto_threshold", "auto_hidden", "auto_off")

    def test_cached_frames_equal_direct_ones(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _CACHE_SCRIPT, scale)
                cases = [line.split()[1:] for line in output.splitlines() if line.startswith("CASE ")]
                failures = [line.split()[1:] for line in output.splitlines() if line.startswith("FAIL ")]
                self.assertEqual(len(cases), 2 * (3 + len(self.TRIGGERS) + len(self.COPIES)) + len(self.AUTO), output)
                for name, drew, differ in cases:
                    with self.subTest(case=name):
                        self.assertEqual(int(differ), 0, output)
                        if name in self.COPIES or name in ("capture", "edges_hidden"):
                            self.assertEqual(int(drew), 0, output)
                        else:
                            self.assertEqual(int(drew), 1, output)
                self.assertEqual(len(failures), 2, output)
                for name, failed, reports, differ, cache_bytes in failures:
                    with self.subTest(failure=name):
                        self.assertEqual((failed, reports, differ, cache_bytes), ("1", "1", "0", "0"), output)


class EdgeStencilLimitPixelTests(unittest.TestCase):
    """The overdraw limit on a real GL canvas, at display scales 1 and 2.

    In a framebuffer with a stencil buffer, every edge path, alpha, colour and
    background draws the same pixels with the limit as without it, nodes drawn
    over saturated edges included; a limit of 8 changes them, so the stencil
    really counts. canvas.render(), whose framebuffer has no stencil buffer,
    as print's, is untouched even by a limit of 8.
    """

    def run_script(self, scale):
        output = run_gl_script(self, _STENCIL_SCRIPT, scale)
        lines = {line.split()[0]: line.split()[1:] for line in output.splitlines() if line.strip()}
        self.assertIn("STENCIL", lines, output)
        return lines, output

    def test_the_limit_leaves_every_pixel_as_it_was(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                lines, output = self.run_script(scale)
                self.assertEqual(float(lines["SCALE"][0]), scale)
                self.assertEqual(lines["BITS"], ["0", "8"])
                cases, inexact, engaged, contrasting, printed_differ = map(int, lines["STENCIL"])
                self.assertEqual(cases, 64)
                self.assertEqual(inexact, 0, output)
                self.assertEqual(contrasting, 12)
                self.assertEqual(engaged, contrasting, output)
                self.assertEqual(printed_differ, 0, output)


_OPACITY_SCRIPT = textwrap.dedent(
    """
    import sys
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    from vispy import app, gloo, scene
    from desktop import Viewer_Edges as VE
    from desktop import Viewer_Motion as VM
    app.use_app("pyside6")
    WIDTH, HEIGHT = 200, 130
    canvas = scene.SceneCanvas(size=(WIDTH, HEIGHT), bgcolor="white", show=False, config=dict(stencil_size=8))
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.set_range(x=(0, WIDTH), y=(0, HEIGHT), margin=0)

    def rows(count, step):
        # Edge i is a horizontal line along its own row, on a pixel's middle.
        pos = np.zeros((2 * count, 2), dtype=np.float32)
        for i in range(count):
            pos[2 * i] = (20, 4.5 + step * i)
            pos[2 * i + 1] = (180, 4.5 + step * i)
        return pos, np.array([[2 * i, 2 * i + 1] for i in range(count)], dtype=np.int32)

    def measure(image, i, step):
        # The darkest pixel at the middle of edge i's row: 255 less what the edge added to the white.
        scale = canvas.pixel_scale
        row = int(round((HEIGHT - (4.5 + step * i)) * scale))
        return int(image[max(0, row - 2):row + 2, int(100 * scale), 0].min())

    def differing(a, b):
        return int((a != b).any(axis=2).sum())

    COLOR = (0.0, 0.0, 0.0, 0.3)
    N, STEP = 8, 12
    pos, edges = rows(N, STEP)
    scores = np.array([0.2, 0.9, 0.5, 1.0, 0.0, 0.7, 0.35, 0.6], dtype=np.float32)
    layers = []

    def make(edges_, scores_, pos_, **options):
        layer = VE.ViewerEdges(view.scene, edges_, scores_, pos_, COLOR, 1.0, use_position_texture=True, **options)
        layer.show_threshold(pos_, None, -1.0)
        layer.visible = False
        layers.append(layer)
        return layer

    def only(layer):
        for other in layers:
            other.visible = other is layer

    plain = make(edges, scores, pos)
    only(plain)
    plain.visible = True
    if (canvas.render()[..., 0] < 250).sum() == 0:
        sys.exit(3)
    print("CONTEXT OK", flush=True)
    print("SCALE", canvas.pixel_scale, flush=True)

    def expected(alpha):
        return 255.0 * (1.0 - alpha)

    def alpha_of(score, drawn):
        return 0.15 + 0.7 * (score - drawn.min()) / (drawn.max() - drawn.min())

    # Plain, as before; then Auto on, then off again: the same pixels as plain.
    layer = make(edges, scores, pos)
    only(layer)
    before = canvas.render()
    layer.set_opacity_by_score(True)
    layer.show_threshold(pos, None, 0.0)
    levels = canvas.render()
    for i in range(N):
        print("ROW levels", i, measure(levels, i, STEP), round(expected(alpha_of(scores[i], scores)), 2), flush=True)
    uploads_before = (layer.upload_count, layer.score_upload_count, layer.position_upload_count)
    buffers_before = (layer.visual._buffers["threshold"], layer.visual._grade_buffers["threshold"])
    layer.show_threshold(pos, None, 0.5)
    half = canvas.render()
    drawn = scores[scores >= 0.5]
    for i in range(N):
        wanted = alpha_of(scores[i], drawn) if scores[i] >= 0.5 else 0.0
        print("ROW threshold", i, measure(half, i, STEP), round(expected(wanted), 2), flush=True)
    layer.show_threshold(pos, None, 0.0)
    again = canvas.render()
    print("UPLOADS", int(uploads_before == (layer.upload_count, layer.score_upload_count, layer.position_upload_count)),
          int(buffers_before[0] is layer.visual._buffers["threshold"]),
          int(buffers_before[1] is layer.visual._grade_buffers["threshold"]),
          layer.score_upload_count, differing(levels, again), flush=True)
    layer.set_opacity_by_score(False)
    layer.show_threshold(pos, None, -1.0)
    print("CASE off_again", differing(before, canvas.render()), flush=True)

    # Hiding nodes leaves the others' alpha as it was: the map comes from the threshold.
    layer.set_opacity_by_score(True)
    layer.show_threshold(pos, None, 0.0)
    visible = np.ones(2 * N, dtype=bool)
    visible[[4, 5, 10, 11]] = False
    layer.show_threshold(pos, visible, 0.0)
    hidden = canvas.render()
    for i in range(N):
        gone = i in (2, 5)
        print("ROW hidden", i, measure(hidden, i, STEP),
              255 if gone else round(expected(alpha_of(scores[i], scores)), 2), flush=True)

    # Scores that are all the same are all 85 %.
    equal = make(edges, np.full(N, 0.4, dtype=np.float32), pos)
    equal.set_opacity_by_score(True)
    equal.show_threshold(pos, None, 0.0)
    only(equal)
    flat = canvas.render()
    for i in range(N):
        print("ROW equal", i, measure(flat, i, STEP), round(expected(0.85), 2), flush=True)

    # A selection's edges are drawn at the colour's alpha, whatever Auto says.
    auto = make(edges, scores, pos)
    auto.set_opacity_by_score(True)
    # One that has drawn by score before, and one that has not.
    used = make(edges, scores, pos)
    used.set_opacity_by_score(True)
    used.show_threshold(pos, None, 0.0)
    ordinary = make(edges, scores, pos)
    chosen = edges[::2]
    auto.show_edges(pos, chosen)
    used.show_edges(pos, chosen)
    ordinary.show_edges(pos, chosen)
    only(auto)
    first = canvas.render()
    only(used)
    third = canvas.render()
    only(ordinary)
    second = canvas.render()
    print("CASE explicit", differing(first, second), differing(third, second), int((first[..., 0] < 250).any()),
          auto.score_upload_count, used.score_upload_count, flush=True)

    # Moving: every k-th edge is drawn, each as opaque as k of them: 1 - (1 - alpha) ** k.
    class NoTimer:
        available = False

        def time(self, draw, tag, on_result):
            draw()

    VE.MIN_SUBSET_SEGMENTS = 8
    motion = VM.MotionDetail(enabled=True, gpu_timer=NoTimer(), report=lambda text: None)
    motion.level_for = lambda segments, levels, into_window: 4
    pos16, edges16 = rows(16, 7)
    scores16 = np.linspace(1.0, 0.0, 16).astype(np.float32)
    thin = make(edges16, scores16, pos16, motion=motion)
    thin.set_opacity_by_score(True)
    thin.show_threshold(pos16, None, 0.0)
    only(thin)
    moving = canvas.render()
    print("CASE thinned", int(thin.visual.detail), thin.visual.drawn_segments(), flush=True)
    for i in range(16):
        alpha = 0.85 - 0.7 * i / 15.0
        wanted = 1.0 - (1.0 - alpha) ** 4 if i % 4 == 0 else 0.0
        print("ROW motion", i, measure(moving, i, 7), round(expected(wanted), 2), flush=True)

    # The overdraw limit leaves every pixel as it was with the alpha following the score, and counts.
    rng = np.random.default_rng(42)
    dense = np.vstack([rng.normal((20, 50), 1.5, (30, 2)), rng.normal((80, 52), 1.5, (30, 2)),
                       rng.random((100, 2)) * 100]).astype(np.float32)
    dense_edges = np.vstack([np.stack([rng.integers(0, 30, 2500), rng.integers(30, 60, 2500)], axis=1),
                             rng.integers(60, 160, (300, 2))]).astype(np.int32)
    dense_scores = rng.random(len(dense_edges)).astype(np.float32)
    limit = VE.EdgeStencilLimit(enabled=True, report=lambda text: None)
    bundle = make(dense_edges, dense_scores, dense, stencil_limit=limit)
    bundle.set_opacity_by_score(True)
    bundle.show_threshold(dense, None, 0.0)
    shape = canvas.render().shape[:2]
    stenciled = gloo.FrameBuffer(color=gloo.RenderBuffer(shape))
    packed = gloo.RenderBuffer(shape, format=0x88F0)
    stenciled.depth_buffer = packed
    stenciled.stencil_buffer = packed

    def draw(bg):
        canvas.push_fbo(stenciled, (0, 0), canvas.size)
        try:
            canvas._draw_scene(bgcolor=bg)
            return stenciled.read()
        finally:
            canvas.pop_fbo()

    only(bundle)
    cases = inexact = contrasting = engaged = followed = 0
    for rgb in ((0.0, 0.0, 0.0), (0.9, 0.3, 0.1), (1.0, 1.0, 1.0)):
        bundle.visual.set_color((*rgb, 0.3))
        for bg in ("white", "black"):
            cases += 1
            # Edges the colour of the background show nothing.
            visible = not ((rgb == (0.0, 0.0, 0.0) and bg == "black") or (rgb == (1.0, 1.0, 1.0) and bg == "white"))
            contrasting += visible
            limit.enabled = False
            everything = draw(bg)
            limit.enabled = True
            if differing(draw(bg), everything):
                inexact += 1
            # Highest scores first, an edge takes up to 85 %: 8 fragments settle a pixel already, so
            # it takes a limit of 1 to see the stencil count.
            with mock.patch.object(VE, "STENCIL_LIMIT", 1):
                if visible and differing(draw(bg), everything):
                    engaged += 1
            bundle.set_opacity_by_score(False)
            bundle.show_threshold(dense, None, 0.0)
            if visible and differing(draw(bg), everything):
                followed += 1
            bundle.set_opacity_by_score(True)
            bundle.show_threshold(dense, None, 0.0)
    print("CASE limit", cases, inexact, contrasting, engaged, followed, flush=True)
    """
)


class EdgeOpacityPixelTests(unittest.TestCase):
    """Edge Opacity's Auto switch on a real GL canvas, at display scales 1 and 2.

    One edge to a row, on a white background, in black at the colour's alpha
    0.3: with Auto on, each is drawn at the alpha its score gives it, 15 % for
    the lowest score drawn and 85 % for the highest, whichever the threshold
    and the hidden nodes, and 85 % for all of them when their scores are equal.
    A threshold move changes the opacity with no upload and no new buffer; Auto
    turned off draws the plain picture again; a selection's edges ignore it;
    thinned edges are as opaque as k of them; the overdraw limit still leaves
    every pixel as it was.
    """

    TOLERANCE = 2.0

    def rows(self, output, name):
        return [(int(line.split()[2]), int(line.split()[3]), float(line.split()[4]))
                for line in output.splitlines() if line.startswith(f"ROW {name} ")]

    def case(self, output, name):
        for line in output.splitlines():
            if line.startswith(f"CASE {name} "):
                return [int(value) for value in line.split()[2:]]
        self.fail(f"no CASE {name}: {output}")

    def check_rows(self, output, name, count):
        rows = self.rows(output, name)
        self.assertEqual(len(rows), count, output)
        for index, measured, wanted in rows:
            with self.subTest(case=name, edge=index):
                self.assertAlmostEqual(measured, wanted, delta=self.TOLERANCE, msg=output)

    def test_the_alpha_follows_the_score_on_screen(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _OPACITY_SCRIPT, scale)
                self.assertIn(f"SCALE {float(scale)}", output)
                # The lowest score is 15 % (255 * 0.85 = 217), the highest 85 % (38), the rest between.
                levels = self.rows(output, "levels")
                self.assertEqual(len(levels), 8, output)
                by_score = sorted(zip([0.2, 0.9, 0.5, 1.0, 0.0, 0.7, 0.35, 0.6], levels))
                self.assertAlmostEqual(by_score[0][1][1], 216.75, delta=self.TOLERANCE)
                self.assertAlmostEqual(by_score[-1][1][1], 38.25, delta=self.TOLERANCE)
                self.assertAlmostEqual(by_score[3][1][1], 255 * (1 - (0.15 + 0.7 * 0.5)), delta=self.TOLERANCE)
                self.assertEqual([entry[1][1] for entry in by_score],
                                 sorted((entry[1][1] for entry in by_score), reverse=True), output)
                self.check_rows(output, "levels", 8)
                self.check_rows(output, "hidden", 8)
                self.check_rows(output, "equal", 8)

    def test_a_threshold_move_remaps_without_an_upload(self):
        output = run_gl_script(self, _OPACITY_SCRIPT, 1)
        # Edges below 0.5 are gone, the faintest left is 15 %, the top one 85 %.
        self.check_rows(output, "threshold", 8)
        faintest = [measured for _, measured, wanted in self.rows(output, "threshold") if wanted < 255]
        self.assertAlmostEqual(max(faintest), 216.75, delta=self.TOLERANCE)
        line = next(line for line in output.splitlines() if line.startswith("UPLOADS "))
        same_counts, same_edges, same_grades, score_uploads, changed_back = map(int, line.split()[1:])
        self.assertEqual((same_counts, same_edges, same_grades, score_uploads), (1, 1, 1, 1))
        self.assertEqual(changed_back, 0)  # back at the first threshold, the first picture again

    def test_auto_off_draws_the_plain_picture_again(self):
        output = run_gl_script(self, _OPACITY_SCRIPT, 1)
        self.assertEqual(self.case(output, "off_again"), [0])

    def test_a_selections_edges_ignore_auto(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _OPACITY_SCRIPT, scale)
                differ, differ_after_auto, drawn, score_uploads, uploads_after_auto = self.case(output, "explicit")
                # Drawn as plain edges are, whether or not the layer had drawn by score before,
                # and no grades are made for them.
                self.assertEqual((differ, differ_after_auto, drawn), (0, 0, 1), output)
                self.assertEqual((score_uploads, uploads_after_auto), (0, 1), output)

    def test_thinned_edges_are_as_opaque_as_the_edges_they_stand_for(self):
        output = run_gl_script(self, _OPACITY_SCRIPT, 1)
        self.assertEqual(self.case(output, "thinned"), [4, 4], output)
        self.check_rows(output, "motion", 16)
        drawn = [measured for index, measured, wanted in self.rows(output, "motion") if index % 4 == 0]
        self.assertEqual(len(drawn), 4)
        self.assertLess(drawn[0], 255 * (1 - 0.85) + self.TOLERANCE + 1)

    def test_the_overdraw_limit_leaves_every_pixel_as_it_was(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _OPACITY_SCRIPT, scale)
                cases, inexact, contrasting, engaged, followed = self.case(output, "limit")
                self.assertEqual((cases, contrasting), (6, 4))
                self.assertEqual(inexact, 0, output)
                # A limit of 1 does change them, so the stencil counts these fragments;
                self.assertEqual(engaged, contrasting, output)
                # and the alpha by score is a different picture from the colour's alpha.
                self.assertEqual(followed, contrasting, output)


if __name__ == "__main__":
    unittest.main()
