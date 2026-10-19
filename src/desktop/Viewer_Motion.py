# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Fewer edges while the view moves (MotionDetail).

While the user pans, zooms (the wheel, a pinch, the Zoom box) or drags nodes,
the Viewer draws every k-th edge of the score order, each edge's opacity
raised to 1 - (1 - a) ** k so that dense regions keep their tone. Once no
such input has come for MOTION_SETTLE_MS, it draws every edge again. Only the
input handlers start this (MotionDetail.note): a camera moved by print or a
command draws in full, and so does every draw into a framebuffer object
(canvas.render(): print, captures).

k is one of MOTION_LEVELS, chosen when a motion starts and only raised while
it lasts, so the detail never flickers between levels:

* GPU timestamps measure what the edges take to draw. A motion starts with
  the least k whose share of the last measured full-detail cost fits
  MOTION_EDGE_BUDGET_MS; k = 1, every edge, when the full frame fits already.
  MOTION_SLOW_FRAMES motion frames in a row that still run
  MOTION_STEP_UP_FACTOR over the budget raise k one level (one slow frame,
  such as the first after a switch, does not). A motion starts with more
  detail than the last one ended with only when that detail is expected to
  take at most MOTION_STEP_DOWN_MARGIN of the budget.
* Without GPU timestamps (before OpenGL 3.3), a motion starts with the least
  k that draws at most FALLBACK_EDGES_PER_FRAME edges, or the k the last
  motion ended with if higher, and frames FALLBACK_SLOW_FRAME_MS apart raise k.

Config's Fewer Edges While Moving switch (FEWER_EDGES_WHILE_MOVING)
turns it off, and so does SSN_EDGE_MOTION_LOD=0, whatever the switch says.
While it is off no thinned copies are made, so they take no memory.
"""

from __future__ import annotations

import ctypes
import os
import time
from collections import deque

MOTION_ENVIRONMENT_VARIABLE = "SSN_EDGE_MOTION_LOD"
_OFF_VALUES = {"0", "off", "false", "no"}
# Every k-th edge of the score order is drawn while the view moves.
MOTION_LEVELS = (1, 4, 16, 64)
# Full detail returns this long after the last pan, zoom or drag input.
MOTION_SETTLE_MS = 150
# The GPU time the edges may take in a motion frame: a 60 Hz frame has 16.7 ms,
# and the nodes, the text and the window's own drawing need the rest.
MOTION_EDGE_BUDGET_MS = 8.0
# Motion frames whose edges take this much more than the budget raise k,
# MOTION_SLOW_FRAMES of them in a row.
MOTION_STEP_UP_FACTOR = 1.25
MOTION_SLOW_FRAMES = 2
# A new motion draws more detail than the last one ended with only if that
# detail is expected to take at most this share of the budget.
MOTION_STEP_DOWN_MARGIN = 0.7
# Without GPU timestamps: the edges a motion frame may draw, and the frame
# interval past which k is raised.
FALLBACK_EDGES_PER_FRAME = 1_500_000
FALLBACK_SLOW_FRAME_MS = 25.0
# Edge buffers shorter than this are always drawn in full: they are cheap,
# and their thinned copies would not be worth the memory.
MIN_SUBSET_SEGMENTS = 250_000


def motion_detail_enabled():
    value = os.environ.get(MOTION_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value not in _OFF_VALUES


def compensated_alpha(alpha, k):
    """The opacity that k-times fewer edges need to give dense regions the
    tone every edge gives them: 1 - (1 - alpha) ** k."""
    alpha = min(max(float(alpha), 0.0), 1.0)
    return 1.0 - (1.0 - alpha) ** int(k)


# OpenGL 3.3 (ARB_timer_query) names.
_GL_TIMESTAMP = 0x8E28
_GL_QUERY_RESULT = 0x8866
_GL_QUERY_RESULT_AVAILABLE = 0x8867


class GpuTimer:
    """GPU timestamps around a draw, read back frames later without waiting.

    It loads glQueryCounter and friends through VisPy on first use; without
    them (before OpenGL 3.3), or if they raise a GL error, ``available``
    turns False and draws are just drawn.
    """

    def __init__(self):
        self.available = None
        self._verified = False
        self._pending = deque()
        self._free = []

    def _load(self):
        try:
            from vispy.gloo.gl import gl2
            from vispy.gloo import gl
            uint, uint_p = ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)
            self._gen = gl2._get_gl_func("glGenQueries", None, (ctypes.c_int, uint_p))
            self._counter = gl2._get_gl_func("glQueryCounter", None, (uint, uint))
            self._get_int = gl2._get_gl_func(
                "glGetQueryObjectiv", None, (uint, uint, ctypes.POINTER(ctypes.c_int)))
            self._get_u64 = gl2._get_gl_func(
                "glGetQueryObjectui64v", None, (uint, uint, ctypes.POINTER(ctypes.c_uint64)))
            self._gl = gl
            self.available = True
        except Exception:
            self.available = False

    def _query(self):
        if self._free:
            return self._free.pop()
        ids = (ctypes.c_uint * 2)()
        self._gen(2, ids)
        self._free.append(int(ids[1]))
        return int(ids[0])

    def time(self, draw, tag, on_result):
        """Draw, with GPU timestamps around it when it can; on_result(tag, ms)
        follows once the GPU has done it (at a later call)."""
        if self.available is None:
            self._load()
        if not self.available:
            draw()
            return
        self.poll(on_result)
        try:
            start, end = self._query(), self._query()
            self._counter(start, _GL_TIMESTAMP)
            # The first timestamp shows whether this GL takes them. Errors
            # are not looked for after the draw, which are the draw's own.
            if not self._verified:
                if self._gl.glGetError():
                    raise RuntimeError("timestamp query refused")
                self._verified = True
        except Exception:
            self._disable()
            draw()
            return
        draw()
        try:
            self._counter(end, _GL_TIMESTAMP)
        except Exception:
            self._disable()
            return
        self._pending.append((start, end, tag))

    def poll(self, on_result):
        """Hand on the timings the GPU has finished."""
        while self._pending:
            start, end, tag = self._pending[0]
            ready = ctypes.c_int(0)
            self._get_int(end, _GL_QUERY_RESULT_AVAILABLE, ctypes.byref(ready))
            if not ready.value:
                return
            self._pending.popleft()
            first, last = ctypes.c_uint64(0), ctypes.c_uint64(0)
            self._get_u64(start, _GL_QUERY_RESULT, ctypes.byref(first))
            self._get_u64(end, _GL_QUERY_RESULT, ctypes.byref(last))
            self._free.extend((start, end))
            if last.value >= first.value:
                on_result(tag, (last.value - first.value) / 1e6)

    def _disable(self):
        self.available = False
        self._pending.clear()
        try:
            for _ in range(16):
                if not self._gl.glGetError():
                    break
        except Exception:
            pass


class MotionDetail:
    """Which share of the edges to draw: all of them, or every k-th while the
    view moves (see the module's documentation for the rule)."""

    def __init__(self, enabled=None, report=None, clock=time.monotonic, timer_factory=None,
                 gpu_timer=None, setting=True):
        # setting is Config's switch; enabled, when given, decides alone.
        if enabled is None:
            self.off_by = ("environment" if not motion_detail_enabled()
                           else None if setting else "setting")
            self.enabled = self.off_by is None
        else:
            self.enabled = bool(enabled)
            self.off_by = None if self.enabled else "environment"
        self.active = False
        self.level = None                 # k for the motion now, chosen at its first frame
        self.final_level = 1              # k the last motion ended with
        self._report = print if report is None else report
        self._reported = False
        self._clock = clock
        self._timer_factory = timer_factory or _qt_settle_timer
        self._timer = None
        self._listeners = []
        self._gpu = GpuTimer() if gpu_timer is None else gpu_timer
        self._ns_per_segment = None       # the last measured GPU cost of one drawn edge
        self._last_frame = None
        self._intervals = deque(maxlen=3)
        self._slow_frames = 0

    # --- input -----------------------------------------------------------

    def note(self):
        """A pan, zoom or drag input: draw fewer edges until it settles."""
        if not self.enabled:
            return
        if not self.active:
            self.active = True
            self.level = None
            self._last_frame = None
            self._intervals.clear()
            self._slow_frames = 0
        if self._timer is None:
            self._timer = self._timer_factory(self.settle)
        self._timer.start()

    def settle(self):
        """The view has stopped: every edge again, from the next frame."""
        if not self.active:
            return
        self.active = False
        self.final_level = self.level or 1
        self.level = None
        for listener in list(self._listeners):
            listener()

    def on_settle(self, listener):
        self._listeners.append(listener)

    # --- per frame -------------------------------------------------------

    def level_for(self, segments, levels, into_window):
        """The k to draw a buffer of segments edges with, given the levels it
        has thinned copies for, into the window (True) or a capture."""
        if not (self.enabled and self.active and into_window) or segments < MIN_SUBSET_SEGMENTS:
            return 1
        if self.level is None:
            self.level = self._first_level(segments)
        elif self._gpu.available is False:
            self._check_frame_interval()
        usable = [k for k in levels if k <= self.level] or [1]
        return max(usable)

    def _first_level(self, segments):
        if self._ns_per_segment is None:
            # No GPU timing yet: the edge count, and the last motion's outcome.
            level = next((k for k in MOTION_LEVELS if segments / k <= FALLBACK_EDGES_PER_FRAME),
                         MOTION_LEVELS[-1])
            return max(level, self.final_level)
        full_ms = self._ns_per_segment * segments / 1e6
        level = next((k for k in MOTION_LEVELS if full_ms / k <= MOTION_EDGE_BUDGET_MS),
                     MOTION_LEVELS[-1])
        if level < self.final_level and full_ms / level > MOTION_STEP_DOWN_MARGIN * MOTION_EDGE_BUDGET_MS:
            level = self.final_level
        return level

    def _raise_level(self):
        higher = [k for k in MOTION_LEVELS if k > (self.level or 1)]
        if higher:
            self.level = higher[0]
            self._intervals.clear()
            self._last_frame = None
            self._slow_frames = 0

    def _check_frame_interval(self):
        now = self._clock()
        if self._last_frame is not None:
            self._intervals.append((now - self._last_frame) * 1e3)
        self._last_frame = now
        if len(self._intervals) == self._intervals.maxlen and \
                sorted(self._intervals)[1] > FALLBACK_SLOW_FRAME_MS:
            self._raise_level()

    def time_draw(self, draw, k, segments):
        """Draw the edges (k-th of them, segments drawn), timing the GPU's work."""
        if not self.enabled or segments < 1:
            draw()
            return
        self._gpu.time(draw, (k, segments), self._record)

    def _record(self, tag, gpu_ms):
        k, segments = tag
        self._ns_per_segment = gpu_ms * 1e6 / segments
        if not (self.active and self.level is not None and k == self.level):
            return
        if gpu_ms > MOTION_STEP_UP_FACTOR * MOTION_EDGE_BUDGET_MS:
            self._slow_frames += 1
            if self._slow_frames >= MOTION_SLOW_FRAMES:
                self._raise_level()
        else:
            self._slow_frames = 0

    # --- report ----------------------------------------------------------

    def report_once(self, thinned):
        """Say once what motion does: thinned is whether the edge visual can draw thinned copies."""
        if self._reported:
            return
        self._reported = True
        if not self.enabled and self.off_by == "setting":
            self._report("Edges: motion detail off (Fewer Edges While Moving is off in the Config).")
        elif not self.enabled:
            self._report(f"Edges: motion detail off ({MOTION_ENVIRONMENT_VARIABLE}=0).")
        elif not thinned:
            self._report("Edges: motion detail off: the coordinate fallback draws every edge.")
        else:
            how = {True: "GPU timer", False: "edge count and frame times", None: "GPU timer if available"}[
                self._gpu.available]
            self._report(
                f"Edges: motion detail on (every 4th, 16th or 64th edge while the view moves, "
                f"every edge {MOTION_SETTLE_MS} ms after; chosen by {how})."
            )


def _qt_settle_timer(callback):
    """A single-shot Qt timer that calls callback MOTION_SETTLE_MS after its last start()."""
    from PySide6 import QtCore
    timer = QtCore.QTimer()
    timer.setSingleShot(True)
    timer.setInterval(MOTION_SETTLE_MS)
    timer.timeout.connect(callback)
    return timer
