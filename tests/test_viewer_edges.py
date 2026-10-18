"""The desktop Viewer's edge layer (desktop/Viewer_Edges.py) and update_edges.

The edges are sorted by descending score once, so a threshold keeps a prefix
of them, and their buffer holds node numbers that the vertex shader resolves
through a texture of node positions. These tests check that the drawn edges
are exactly those edge_stages keeps, and that each kind of change re-sends only
what it must: a threshold change nothing, a node move the position texture,
hiding nodes the edge buffer once. The GL tests render the old Line visual
and the new one on a real GL canvas and compare their pixels, and check that
the stencil overdraw limit (EdgeStencilLimit) leaves every pixel as it was.
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
                    layer.visual.shared_program["u_color"] = (*rgb, alpha)
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


class EdgeStencilLimitPixelTests(unittest.TestCase):
    """The overdraw limit on a real GL canvas, at display scales 1 and 2.

    In a framebuffer with a stencil buffer, every edge path, alpha, colour and
    background draws the same pixels with the limit as without it, nodes drawn
    over saturated edges included; a limit of 8 changes them, so the stencil
    really counts. canvas.render(), whose framebuffer has no stencil buffer,
    as print's, is untouched even by a limit of 8.
    """

    def run_script(self, scale):
        environment = dict(os.environ)
        environment.pop("QT_QPA_PLATFORM", None)
        environment.pop(Viewer_Edges.STENCIL_ENVIRONMENT_VARIABLE, None)
        environment.update(PYTHONDONTWRITEBYTECODE="1", QT_ENABLE_HIGHDPI_SCALING="0",
                           QT_SCALE_FACTOR=str(scale))
        try:
            result = subprocess.run(
                [sys.executable, "-c", _STENCIL_SCRIPT, SRC_DIR],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.skipTest("the GL render probe timed out")
        if "CONTEXT OK" not in result.stdout:
            self.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
        lines = {line.split()[0]: line.split()[1:] for line in result.stdout.splitlines() if line.strip()}
        self.assertIn("STENCIL", lines, result.stdout + result.stderr[-2000:])
        return lines, result.stdout

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


if __name__ == "__main__":
    unittest.main()
