"""Fewer edges while the view moves (desktop/Viewer_Motion.py, desktop/Viewer_Edges.py).

While the user pans, zooms or drags nodes the edge layer draws every k-th
edge of the score order from thinned copies, with the opacity raised to keep
dense regions' tone, and every edge again once the view settles. These tests
check the copies against the strided score order through thresholds, hidden
nodes and selection lists; the opacity; the choice of k and its stability;
the settle timer; which inputs start a motion and that a camera moved
otherwise does not; that captures always draw every edge; and that the image
cache only ever holds full-detail pictures. The GL test draws motion and
settled frames on a real canvas.
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
from vispy.scene.visuals import create_visual_node  # noqa: E402
from EMAPSSN_Viewer import MainViewer  # noqa: E402
from desktop import Viewer_Edges, Viewer_Motion  # noqa: E402
from desktop.Viewer_Edges import EdgeSegmentsVisual, ViewerEdges  # noqa: E402
from desktop.Viewer_Motion import MotionDetail, compensated_alpha  # noqa: E402


class FakeTimer:
    """A settle timer the test fires by hand."""

    def __init__(self, callback):
        self.callback = callback
        self.starts = 0

    def start(self):
        self.starts += 1

    def fire(self):
        self.callback()


class FakeGpuTimer:
    """GPU timing the test feeds by hand: available, or not (before OpenGL 3.3)."""

    def __init__(self, available=True):
        self.available = available
        self.timed = []

    def time(self, draw, tag, on_result):
        self.timed.append(tag)
        draw()


def motion_detail(available=True, **kwargs):
    timers = []

    def factory(callback):
        timers.append(FakeTimer(callback))
        return timers[-1]
    reports = []
    detail = MotionDetail(enabled=True, report=reports.append, timer_factory=factory,
                          gpu_timer=FakeGpuTimer(available), **kwargs)
    return detail, timers, reports


class CompensatedAlphaTests(unittest.TestCase):
    def test_dense_regions_keep_their_tone(self):
        self.assertAlmostEqual(compensated_alpha(0.1, 1), 0.1)
        self.assertAlmostEqual(compensated_alpha(0.1, 4), 1 - 0.9 ** 4)
        self.assertAlmostEqual(compensated_alpha(0.5, 16), 1 - 0.5 ** 16)
        # k edges of the compensated opacity leave a pixel as k * n edges of
        # the original leave it, for any n.
        for alpha in (0.01, 0.1, 0.4):
            for k in (4, 16, 64):
                self.assertAlmostEqual((1 - compensated_alpha(alpha, k)) ** 3, (1 - alpha) ** (3 * k))
        self.assertEqual(compensated_alpha(1.0, 4), 1.0)
        self.assertEqual(compensated_alpha(0.0, 64), 0.0)
        self.assertEqual(compensated_alpha(1.5, 4), 1.0)


class MotionDetailTests(unittest.TestCase):
    SEGMENTS = 8_000_000
    ALL = (1, 4, 16, 64)

    def test_the_switch(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(Viewer_Motion.MOTION_ENVIRONMENT_VARIABLE, None)
            self.assertTrue(MotionDetail().enabled)
            for value in ("0", "off", "False", " no "):
                os.environ[Viewer_Motion.MOTION_ENVIRONMENT_VARIABLE] = value
                self.assertFalse(MotionDetail().enabled, value)

    def test_every_edge_unless_the_view_moves_into_the_window(self):
        detail, timers, _ = motion_detail()
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 1)
        detail.note()
        self.assertGreater(detail.level_for(self.SEGMENTS, self.ALL, True), 1)
        # A capture, print's or MCP's, draws into a framebuffer object.
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, False), 1)
        # A short buffer is cheap: always in full.
        self.assertEqual(detail.level_for(Viewer_Motion.MIN_SUBSET_SEGMENTS - 1, self.ALL, True), 1)
        timers[0].fire()
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 1)
        off = MotionDetail(enabled=False, gpu_timer=FakeGpuTimer())
        off.note()
        self.assertEqual(off.level_for(self.SEGMENTS, self.ALL, True), 1)

    def test_each_input_restarts_the_settle_timer_and_settling_redraws(self):
        detail, timers, _ = motion_detail()
        redraws = []
        detail.on_settle(lambda: redraws.append(1))
        for _ in range(3):
            detail.note()
        self.assertEqual(len(timers), 1)
        self.assertEqual(timers[0].starts, 3)
        self.assertTrue(detail.active)
        timers[0].fire()
        self.assertFalse(detail.active)
        self.assertEqual(redraws, [1])
        timers[0].fire()                      # a late timer after settling does nothing
        self.assertEqual(redraws, [1])
        self.assertEqual(Viewer_Motion.MOTION_SETTLE_MS, 150)

    def test_k_fits_the_last_full_frame_into_the_budget(self):
        budget = Viewer_Motion.MOTION_EDGE_BUDGET_MS
        for full_ms, expected in ((budget * 0.9, 1), (budget * 3.0, 4), (budget * 10.0, 16),
                                  (budget * 40.0, 64), (budget * 500.0, 64)):
            with self.subTest(full_ms=full_ms):
                detail, _, _ = motion_detail()
                detail._record((1, self.SEGMENTS), full_ms)        # a full-detail frame's GPU time
                detail.note()
                self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), expected)

    def test_without_gpu_timing_the_edge_count_chooses(self):
        per_frame = Viewer_Motion.FALLBACK_EDGES_PER_FRAME
        for segments, expected in ((per_frame, 1), (per_frame * 3, 4), (per_frame * 10, 16),
                                   (per_frame * 1000, 64)):
            with self.subTest(segments=segments):
                detail, _, _ = motion_detail(available=False)
                detail.note()
                self.assertEqual(detail.level_for(max(segments, Viewer_Motion.MIN_SUBSET_SEGMENTS),
                                                  self.ALL, True), max(expected, 1))

    def test_a_slow_motion_frame_raises_k_and_nothing_lowers_it_while_moving(self):
        budget = Viewer_Motion.MOTION_EDGE_BUDGET_MS
        detail, _, _ = motion_detail()
        detail._record((1, self.SEGMENTS), budget * 3.0)
        detail.note()
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 4)
        # One slow frame (the first after a switch, say) changes nothing.
        detail._record((4, self.SEGMENTS // 4), budget * 2.0)
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 4)
        detail._record((4, self.SEGMENTS // 4), budget * 0.5)
        detail._record((4, self.SEGMENTS // 4), budget * 2.0)
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 4)
        # The zoom brought longer edges: motion frames at k = 4 keep running long.
        detail._record((4, self.SEGMENTS // 4), budget * 2.0)
        self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 16)
        # Late results from k = 4 frames, a capture's full frame, and a cheap
        # k = 16 frame change nothing: only frames at the current k count.
        detail._record((4, self.SEGMENTS // 4), budget * 5.0)
        detail._record((4, self.SEGMENTS // 4), budget * 5.0)
        detail._record((1, self.SEGMENTS), budget * 5.0)
        detail._record((16, self.SEGMENTS // 16), budget * 0.01)
        for _ in range(10):
            self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 16)

    def test_a_new_motion_returns_to_more_detail_only_with_room_to_spare(self):
        budget = Viewer_Motion.MOTION_EDGE_BUDGET_MS
        margin = Viewer_Motion.MOTION_STEP_DOWN_MARGIN
        for full_ms, expected in ((4 * budget * (margin + 0.1), 16), (4 * budget * (margin - 0.1), 4)):
            with self.subTest(full_ms=full_ms):
                detail, timers, _ = motion_detail()
                detail._record((1, self.SEGMENTS), budget * 3.0)
                detail.note()
                detail.level_for(self.SEGMENTS, self.ALL, True)
                detail._record((4, self.SEGMENTS // 4), budget * 2.0)
                detail._record((4, self.SEGMENTS // 4), budget * 2.0)    # raised to 16
                self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), 16)
                timers[0].fire()
                self.assertEqual(detail.final_level, 16)
                detail._record((1, self.SEGMENTS), full_ms)              # the settled frame
                detail.note()
                self.assertEqual(detail.level_for(self.SEGMENTS, self.ALL, True), expected)

    def test_without_gpu_timing_long_frame_intervals_raise_k(self):
        now = [0.0]
        for interval_ms, expected in ((40.0, 16), (10.0, 4)):
            with self.subTest(interval_ms=interval_ms):
                detail, _, _ = motion_detail(available=False, clock=lambda: now[0])
                detail.note()
                segments = Viewer_Motion.FALLBACK_EDGES_PER_FRAME * 3      # k = 4 by count
                levels = [detail.level_for(segments, self.ALL, True)]
                for _ in range(5):
                    now[0] += interval_ms / 1e3
                    levels.append(detail.level_for(segments, self.ALL, True))
                self.assertEqual(levels[0], 4)
                self.assertEqual(levels[-1], expected)
                self.assertEqual(levels, sorted(levels))                  # never lower while moving

    def test_only_the_thinned_copies_there_are(self):
        detail, _, _ = motion_detail()
        detail._record((1, self.SEGMENTS), Viewer_Motion.MOTION_EDGE_BUDGET_MS * 40)
        detail.note()
        self.assertEqual(detail.level_for(self.SEGMENTS, (1, 4), True), 4)
        self.assertEqual(detail.level_for(self.SEGMENTS, (1,), True), 1)

    def test_draws_are_timed_with_their_level_and_segments(self):
        detail, _, _ = motion_detail()
        drawn = mock.Mock()
        detail.time_draw(drawn, 16, 500_000)
        drawn.assert_called_once_with()
        self.assertEqual(detail._gpu.timed, [(16, 500_000)])

    def test_it_says_once_what_it_does(self):
        cases = ((dict(), True, "motion detail on"), (dict(available=False), True, "edge count"),
                 (dict(), False, "coordinate fallback"))
        for kwargs, thinned, expected in cases:
            with self.subTest(expected=expected):
                detail, _, reports = motion_detail(**kwargs)
                detail.report_once(thinned)
                detail.report_once(thinned)
                self.assertEqual(len(reports), 1)
                self.assertIn(expected, reports[0])
        reports = []
        MotionDetail(enabled=False, report=reports.append, gpu_timer=FakeGpuTimer()).report_once(True)
        self.assertIn("SSN_EDGE_MOTION_LOD=0", reports[0])


class RecordingThinnedVisual(EdgeSegmentsVisual):
    """EdgeSegmentsVisual that keeps every buffer it uploads, thinned copies included."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uploaded = {}

    def _upload(self, key, node_numbers):
        self.uploaded[key] = np.array(node_numbers, copy=True).reshape(-1, 2).astype(np.int64)
        super()._upload(key, node_numbers)

    def bound_edges(self):
        """The edges the next draw draws, at the detail set."""
        self._bind()
        key, k, drawn = self._bound
        return self.uploaded[key if k == 1 else (key, k)][:drawn]


RecordingThinned = create_visual_node(RecordingThinnedVisual)


def random_network(seed, nodes=80, edges=900):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, nodes, (edges, 2)).astype(np.int32)
    scores = rng.random(edges).astype(np.float32)
    scores[::7] = 0.5
    positions = (rng.random((nodes, 2)) * 100).astype(np.float32)
    return pairs, scores, positions


class ThinnedCopyTests(unittest.TestCase):
    """Small networks, with MIN_SUBSET_SEGMENTS lowered so that they get thinned copies."""

    def setUp(self):
        for module in (Viewer_Edges, Viewer_Motion):
            patcher = mock.patch.object(module, "MIN_SUBSET_SEGMENTS", 10)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.edges, self.scores, self.positions = random_network(3)
        with mock.patch.object(Viewer_Edges, "EdgeSegments", RecordingThinned):
            self.layer = ViewerEdges(None, self.edges, self.scores, self.positions, (0, 0, 0, 0.1), 1.0,
                                     use_position_texture=True,
                                     motion=MotionDetail(enabled=True, gpu_timer=FakeGpuTimer()))

    def drawn_at(self, k):
        visual = self.layer.visual
        visual.detail = k
        return visual.bound_edges()

    def test_settling_redraws_the_edges(self):
        detail, timers, _ = motion_detail()
        with mock.patch.object(Viewer_Edges, "EdgeSegments", RecordingThinned), \
                mock.patch.object(RecordingThinned, "update") as update:
            layer = ViewerEdges(None, self.edges, self.scores, self.positions, (0, 0, 0, 0.1), 1.0,
                                use_position_texture=True, motion=detail)
            layer.show_threshold(self.positions, None, 0.0)
            layer.note_motion()
            update.reset_mock()
            timers[0].fire()
            update.assert_called_once_with()

    def test_each_copy_holds_every_kth_edge_of_what_is_drawn(self):
        rng = np.random.default_rng(4)
        nodes = len(self.positions)
        for step in range(12):
            visible = None if step % 3 == 0 else rng.random(nodes) > 0.3
            threshold = float(rng.choice([0.0, 0.5, rng.random(), 0.95]))
            with self.subTest(step=step):
                self.layer.show_threshold(self.positions, visible, threshold)
                full = self.drawn_at(1)
                self.assertEqual(len(full), self.layer.drawn_edge_count)
                for k in (4, 16, 64):
                    np.testing.assert_array_equal(self.drawn_at(k), full[::k])

    def test_selection_lists_are_thinned_too_unless_told_not(self):
        explicit = self.edges[200:700]
        self.layer.show_edges(self.positions, explicit)
        full = self.drawn_at(1)
        np.testing.assert_array_equal(full, explicit)
        for k in (4, 16, 64):
            np.testing.assert_array_equal(self.drawn_at(k), explicit[::k])
        self.layer.show_edges(self.positions, explicit.copy(), thinned=False)
        self.assertEqual(self.layer.visual.available_levels(), (1,))
        np.testing.assert_array_equal(self.drawn_at(16), explicit)
        # Back to the threshold: its copies are still there.
        self.layer.show_threshold(self.positions, None, 0.0)
        self.assertEqual(self.layer.visual.available_levels(), (1, 4, 16, 64))

    def test_a_thinned_draw_raises_the_opacity_and_full_detail_restores_it(self):
        self.layer.show_threshold(self.positions, None, 0.0)
        visual = self.layer.visual
        self.drawn_at(16)
        np.testing.assert_allclose(visual.shared_program["u_color"][3], compensated_alpha(0.1, 16))
        self.drawn_at(1)
        np.testing.assert_allclose(visual.shared_program["u_color"][3], 0.1)

    def test_short_buffers_have_no_copies(self):
        with mock.patch.object(Viewer_Edges, "MIN_SUBSET_SEGMENTS", 10_000):
            self.layer.show_threshold(self.positions, None, 0.25)
            self.layer.visual.set_segments("threshold", self.edges)
        self.assertEqual(self.layer.visual.available_levels(), (1,))

    def test_the_copies_add_a_third_to_the_buffer(self):
        self.layer.show_threshold(self.positions, None, 0.0)
        sizes = self.layer.visual._buffer_sizes
        extra = sum(sizes[("threshold", k)] for k in (4, 16, 64))
        self.assertAlmostEqual(extra / sizes["threshold"], 1 / 4 + 1 / 16 + 1 / 64, delta=0.01)

    def test_a_drop_frees_the_copies(self):
        self.layer.show_edges(self.positions, self.edges[:500])
        self.layer.visual.drop_segments("explicit")
        self.assertFalse([key for key in self.layer.visual._buffers if key[0:1] == ("explicit",)
                          or key == "explicit"])


class DrawEdgesTests(unittest.TestCase):
    """_draw_edges: motion frames draw thinned and around the cache; the rest as before."""

    def visual(self, k):
        motion = SimpleNamespace(level_for=mock.Mock(return_value=k), report_once=mock.Mock(),
                                 time_draw=mock.Mock(side_effect=lambda draw, k, n: draw()))
        cache = mock.Mock()
        cache.serves.return_value = True
        cache.draw.side_effect = lambda canvas, key, draw: draw() or True
        visual = SimpleNamespace(image_cache=cache, stencil_limit=None, motion_detail=motion,
                                 segment_count=900_000, available_levels=lambda: (1, 4, 16, 64),
                                 drawn_segments=lambda: 900_000 // k, detail=1)
        return visual, motion, cache

    def draw(self, visual, into_window=True):
        drawn = mock.Mock()
        with mock.patch.object(Viewer_Edges, "get_current_canvas", return_value="canvas"), \
                mock.patch.object(Viewer_Edges, "_draws_into_window", return_value=into_window), \
                mock.patch.object(Viewer_Edges, "_picture_key", return_value="key"), \
                mock.patch.object(Viewer_Edges, "_draw_with_limit", side_effect=lambda v, l, d: d()):
            Viewer_Edges._draw_edges(visual, drawn)
        return drawn

    def test_a_motion_frame_draws_every_kth_edge_without_the_cache(self):
        visual, motion, cache = self.visual(16)
        drawn = self.draw(visual)
        drawn.assert_called_once_with()
        self.assertEqual(visual.detail, 16)
        cache.draw.assert_not_called()
        self.assertEqual(motion.time_draw.call_args.args[1:], (16, 900_000 // 16))
        motion.level_for.assert_called_once_with(900_000, (1, 4, 16, 64), True)
        motion.report_once.assert_called_once_with(True)

    def test_a_still_frame_draws_in_full_through_the_cache(self):
        visual, motion, cache = self.visual(1)
        drawn = self.draw(visual)
        drawn.assert_called_once_with()
        self.assertEqual(visual.detail, 1)
        cache.draw.assert_called_once()
        self.assertEqual(motion.time_draw.call_args.args[1], 1)

    def test_a_capture_asks_for_a_capture(self):
        visual, motion, cache = self.visual(1)
        self.draw(visual, into_window=False)
        self.assertFalse(motion.level_for.call_args.args[2])
        motion.report_once.assert_not_called()


class SettledCacheTests(unittest.TestCase):
    """After a motion the cache draws a new full-detail picture, as the camera moved."""

    def test_the_picture_key_does_not_hold_the_detail(self):
        visual = SimpleNamespace(generation=1, edge_style=lambda: (), stencil_limit=None,
                                 transforms=SimpleNamespace(get_transform=lambda **k: SimpleNamespace(
                                     map=lambda p: np.column_stack([np.asarray(p), np.zeros(3), np.ones(3)]))),
                                 detail=1)
        canvas = SimpleNamespace(physical_size=(10, 10), pixel_scale=1.0, bgcolor=SimpleNamespace(rgba=(1, 1, 1, 1)))
        still = Viewer_Edges._picture_key(visual, canvas)
        visual.detail = 16
        self.assertEqual(Viewer_Edges._picture_key(visual, canvas), still)


class MotionTriggerTests(unittest.TestCase):
    """Input handlers start a motion; a camera moved by anything else does not."""

    def make_viewer(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.edge_layer = SimpleNamespace(note_motion=mock.Mock())
        camera = SimpleNamespace(interactive=True, zoom_factor=0.007,
                                 _scene_transform=SimpleNamespace(imap=lambda pos: pos))
        viewer.view = SimpleNamespace(camera=camera)
        viewer._original_camera_mouse_event = mock.Mock()
        viewer._wheel_zoom = SimpleNamespace(add=mock.Mock())
        viewer._hud_timer = SimpleNamespace(start=mock.Mock())
        return viewer

    @staticmethod
    def event(kind, buttons=(), button=None):
        return SimpleNamespace(type=kind, buttons=list(buttons), button=button, handled=False,
                               pos=np.zeros(2), delta=(0.0, 1.0), modifiers=())

    def test_pans_zooms_and_pinches_start_a_motion(self):
        for kind, buttons, starts in (("mouse_move", (1,), True), ("mouse_move", (), False),
                                      ("mouse_wheel", (), True), ("gesture_zoom", (), True),
                                      ("mouse_press", (1,), False), ("mouse_move", (2,), False)):
            with self.subTest(kind=kind, buttons=buttons), \
                    mock.patch.object(cfg, "LOW_RESOURCE_MODE", False, create=True):
                viewer = self.make_viewer()
                viewer._camera_mouse_event(self.event(kind, buttons))
                self.assertEqual(viewer.edge_layer.note_motion.called, starts)

    def test_each_smooth_zoom_frame_and_the_zoom_box_continue_a_motion(self):
        viewer = self.make_viewer()
        viewer._after_wheel_zoom_step()
        viewer.edge_layer.note_motion.assert_called_once_with()
        viewer._hud_timer.start.assert_called_once_with()
        viewer = self.make_viewer()
        viewer._return_keyboard_to_canvas = mock.Mock()
        with mock.patch("commands.zoom.set_zoom") as set_zoom:
            viewer._zoom_to_box(250.0)
        set_zoom.assert_called_once()
        viewer.edge_layer.note_motion.assert_called_once_with()

    def test_dragging_nodes_starts_a_motion(self):
        viewer = self.make_viewer()
        viewer.is_multi_dragging = True
        viewer.is_box_selecting = False
        viewer.pos = np.zeros((3, 2))
        viewer.selected_indices = [1]
        viewer.drag_start_mouse = np.zeros(2)
        viewer.drag_start_nodes_pos = np.zeros((1, 2))
        transform = SimpleNamespace(map=lambda pos: np.asarray(pos, float))
        viewer.canvas = SimpleNamespace(scene=SimpleNamespace(node_transform=lambda _scene: transform))
        viewer.view = SimpleNamespace(scene=None)
        viewer.update_selection_visual = mock.Mock()
        with mock.patch.object(cfg, "LOW_RESOURCE_MODE", False, create=True):
            viewer.on_mouse_move(SimpleNamespace(pos=np.ones(2), buttons=[2], modifiers=(), handled=False))
        viewer.edge_layer.note_motion.assert_called_once_with()

    def test_the_viewer_without_edges_ignores_motion(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.edge_layer = None
        viewer._note_view_motion()

    def test_a_low_resource_drag_list_is_not_thinned(self):
        viewer = MainViewer.__new__(MainViewer)
        viewer.edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        viewer.edge_scores = np.array([0.5, 0.6], dtype=np.float32)
        viewer.pos = np.zeros((3, 2), dtype=np.float32)
        viewer.visible_mask = np.ones(3, dtype=bool)
        viewer.current_slider_threshold = 0.0
        viewer.selected_indices = [0]
        viewer._refresh_session_card = mock.Mock()
        viewer.edge_layer = mock.Mock()
        viewer.edge_layer.show_edges.return_value = 1
        for low_resource, dragging, thinned in ((True, True, False), (False, False, True)):
            with self.subTest(low_resource=low_resource), \
                    mock.patch.object(cfg, "LOW_RESOURCE_MODE", low_resource, create=True), \
                    mock.patch.object(cfg, "UMAP_MODE", True, create=True):
                viewer.is_multi_dragging = dragging
                viewer.update_edges()
                self.assertEqual(viewer.edge_layer.show_edges.call_args.kwargs["thinned"], thinned)


_MOTION_SCRIPT = textwrap.dedent(
    """
    import ctypes
    import os
    import sys
    from types import SimpleNamespace
    from unittest import mock
    import numpy as np
    sys.path.insert(0, sys.argv[1])
    os.environ["SSN_VIEWER_EXPLICIT_SETTINGS"] = "1"
    from vispy import app, gloo, scene
    from vispy.gloo import gl
    app.use_app("pyside6")
    import EMAPSSN_Viewer as V
    from desktop import Viewer_Edges as VE, Viewer_Motion as VM
    VE.MIN_SUBSET_SEGMENTS = VM.MIN_SUBSET_SEGMENTS = 1000
    cfg = V.cfg
    cfg.NODE_SIZE, cfg.NODE_BOUNDARY_WIDTH, cfg.HOVER_COLOR = 15, 2.0, "#ffaa00"
    rng = np.random.default_rng(42)
    n = 160
    v = V.MainViewer.__new__(V.MainViewer)
    v.n_nodes = n
    # Two tight groups joined by 6000 edges, and loose edges over everything.
    v.pos = np.vstack([rng.normal((-30, 0), 2.0, (40, 2)), rng.normal((30, 2), 2.0, (40, 2)),
                       rng.random((80, 2)) * 80 - 40]).astype(np.float32)
    v.edges = np.vstack([np.stack([rng.integers(0, 40, 6000), rng.integers(40, 80, 6000)], axis=1),
                         rng.integers(80, 160, (1500, 2))]).astype(np.int32)
    v.edge_scores = rng.random(len(v.edges)).astype(np.float32)
    v.current_colors = np.tile([0.27, 0.53, 1.0, 1.0], (n, 1)).astype(np.float32)
    v.current_sizes = np.full(n, 15.0, dtype=np.float32)
    v.current_shapes = np.full(n, "disc", dtype=object)
    v.visible_mask = np.ones(n, dtype=bool)
    v.visible_mask[[100, 101]] = False
    v.node_render_order = np.arange(n, dtype=np.int32)
    v.selected_indices, v.selected_node_idx, v.left_click_highlight_indices = [], None, None
    v.hovered_node_idx = None
    v.current_slider_threshold = 0.2
    v.canvas = scene.SceneCanvas(size=(200, 160), bgcolor="white", show=False, config=dict(stencil_size=8))
    v.view = v.canvas.central_widget.add_view()
    v.view.camera = scene.PanZoomCamera(aspect=1)
    v.view.camera.set_range(x=(-45, 45), y=(-40, 40))
    v._update_hud_elements = lambda *args, **kwargs: None
    v._refresh_session_card = lambda: None
    timers = []

    class Timer:
        def __init__(self, callback):
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

    reports = []
    motion = VM.MotionDetail(enabled=True, report=reports.append, timer_factory=Timer)
    v.edge_layer = VE.ViewerEdges(v.view.scene, v.edges, v.edge_scores, v.pos, (0.5, 0.5, 0.5, 0.1), 1.5,
                                  image_cache=VE.EdgeImageCache(enabled=True, report=reports.append),
                                  motion=motion)
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
    cache, visual = v.edge_layer.image_cache, v.edge_layer.visual

    # Fragment counts, to see a motion frame draw a k-th of the edges.
    from vispy.gloo.gl import gl2
    gen = gl2._get_gl_func("glGenQueries", None, (ctypes.c_int, ctypes.POINTER(ctypes.c_uint)))
    begin = gl2._get_gl_func("glBeginQuery", None, (ctypes.c_uint, ctypes.c_uint))
    end = gl2._get_gl_func("glEndQuery", None, (ctypes.c_uint,))
    result = gl2._get_gl_func("glGetQueryObjectuiv", None, (ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)))
    query = (ctypes.c_uint * 1)()
    gen(1, query)
    VM.MOTION_EDGE_BUDGET_MS = 1e9       # no step-ups: k stays where the test puts it

    def frame(markers=True):
        v.markers.visible = markers
        with mock.patch.object(VE, "_draws_into_window", lambda canvas: True):
            v.canvas.push_fbo(window, (0, 0), v.canvas.size)
            try:
                begin(0x8914, query[0])          # GL_SAMPLES_PASSED
                v.canvas._draw_scene()
                v.canvas.context.flush_commands()
                end(0x8914)
                image = window.read()
            finally:
                v.canvas.pop_fbo()
                v.markers.visible = True
        count = ctypes.c_uint(0)
        result(query[0], 0x8866, ctypes.byref(count))
        return image, int(count.value)

    def edges_only():
        # The edges alone, drawn directly and without the overdraw limit, so
        # that their fragments are counted.
        limit = v.edge_layer.stencil_limit
        cache.enabled, limit.enabled = False, False
        try:
            return frame(markers=False)
        finally:
            cache.enabled, limit.enabled = True, True

    def without_lod():
        motion.enabled, cache.enabled = False, False
        try:
            return frame()[0]
        finally:
            motion.enabled, cache.enabled = True, True

    frame()
    full_edges, full_fragments = edges_only()
    out = {}
    rect = v.view.camera.rect
    v.edge_layer.note_motion()
    motion.level = 4
    pictures = cache.render_count, cache.reuse_count
    moving_edges, moving_fragments = edges_only()
    v.view.camera.rect = (rect.left + 3, rect.bottom, rect.width, rect.height)
    moving, _ = frame()
    out["cache_untouched"] = int((cache.render_count, cache.reuse_count) == pictures)
    out["drawn"] = f"{visual.segment_count}/{visual.drawn_segments()}"
    out["fragment_share"] = round(moving_fragments / max(full_fragments, 1), 3)
    out["motion_differs"] = int((moving != without_lod()).any(axis=2).sum())
    # The dense bundle keeps its tone: the mean grey where every edge saturates it.
    saturated = (full_edges[..., :3] < 140).all(axis=2)
    out["saturated_pixels"] = int(saturated.sum())
    out["tone_shift"] = round(float(abs(moving_edges[saturated][:, :3].astype(float).mean()
                                        - full_edges[saturated][:, :3].astype(float).mean())), 1)
    # A capture while moving (print, MCP) draws every edge.
    captured = v.canvas.render()
    motion.enabled = False
    captured_full = v.canvas.render()
    motion.enabled = True
    out["capture_differs"] = int((captured != captured_full).any(axis=2).sum())
    # A hover while moving: the two-marker update, against every marker sent.
    nodes = list(v._submitted_marker_node_order)
    hover_ok = 1
    for node in (nodes[5], nodes[50], None):
        previous, v.hovered_node_idx = v.hovered_node_idx, node
        calls = []
        with mock.patch.object(v, "update_nodes", side_effect=lambda: calls.append(1) or V.MainViewer.update_nodes(v)):
            v._show_hover_change(previous)
        hovered, _ = frame()
        V.MainViewer.update_nodes(v)
        sent, _ = frame()
        hover_ok &= int(not calls and not (hovered != sent).any())
    out["hover_while_moving"] = hover_ok
    # Settled: every edge again, from a new full-detail picture in the cache.
    before = cache.render_count
    timers[0].callback()
    settled, _ = frame()
    out["settled_picture"] = cache.render_count - before
    out["settled_differs"] = int((settled != without_lod()).any(axis=2).sum())
    out["settled_drawn"] = visual.drawn_segments() == visual.segment_count
    print("MOTION", " ".join(f"{key}={value}" for key, value in out.items()), flush=True)
    print("REPORTS", " | ".join(reports), flush=True)
    """
)


def run_gl_script(test, script, scale):
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    for name in ("SSN_EDGE_STENCIL_LIMIT", "SSN_EDGE_IMAGE_CACHE", Viewer_Motion.MOTION_ENVIRONMENT_VARIABLE):
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


class MotionPixelTests(unittest.TestCase):
    """Motion and settled frames on a real GL canvas, at display scales 1 and 2.

    A motion frame at k = 4 draws a quarter of the edges (a quarter of their
    fragments, give or take), keeps the dense bundle's tone, leaves the image
    cache alone and still shows a hover exactly as before; a capture taken
    while moving draws every edge; and once settled the frame is pixel for
    pixel the frame without motion detail, from a new cached picture.
    """

    def test_motion_frames_thin_and_settled_frames_are_whole(self):
        for scale in (1, 2):
            with self.subTest(display_scale=scale):
                output = run_gl_script(self, _MOTION_SCRIPT, scale)
                line = next(line for line in output.splitlines() if line.startswith("MOTION "))
                values = dict(item.split("=", 1) for item in line.split()[1:])
                full, drawn = map(int, values["drawn"].split("/"))
                self.assertEqual(drawn, -(-full // 4), output)
                self.assertAlmostEqual(float(values["fragment_share"]), 0.25, delta=0.06, msg=output)
                self.assertGreater(int(values["motion_differs"]), 0, output)
                self.assertLess(float(values["tone_shift"]), 12.0, output)
                self.assertEqual(values["cache_untouched"], "1", output)
                self.assertEqual(values["capture_differs"], "0", output)
                self.assertEqual(values["hover_while_moving"], "1", output)
                self.assertEqual(values["settled_differs"], "0", output)
                self.assertEqual(values["settled_picture"], "1", output)
                self.assertEqual(values["settled_drawn"], "True", output)
                self.assertIn("motion detail on", output)


if __name__ == "__main__":
    unittest.main()
