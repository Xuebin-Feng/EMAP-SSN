# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""How the desktop Viewer draws its edges.

Two choices keep the threshold slider and node dragging cheap on networks of
tens of millions of edges:

* The edges are sorted once by descending score. The edges at or above any
  threshold are then a prefix of that order, so a slider move binary-searches
  the prefix length and draws that many edges of a buffer uploaded once.
* The edge buffer holds node numbers, not coordinates. The vertex shader looks
  each node's position up in a small float texture, so moving nodes re-sends
  that texture alone (16 bytes a node), whatever the edge count.

Hiding or showing nodes rebuilds the buffer once, keeping the score order.
UMAP mode and a low-resource drag draw edges chosen by the selection instead;
ViewerEdges.show_edges uploads those as given.

All edges share one colour and alpha, so the order they are blended in leaves
every pixel the same as drawing them in their loaded order.

With SSN_EDGE_POSITION_TEXTURE=0, or a network of 2**24 nodes or more (a
float vertex attribute numbers nodes exactly only below that), the edges are
drawn by VisPy's Line visual from coordinates gathered on the CPU, as before:
the prefix still spares the filtering, but every slider or node move uploads
the drawn edges' coordinates again.

Inside a dense cluster each pixel can lie under thousands of edges, but an
8-bit pixel stops changing long before: the stencil buffer counts the edge
fragments each pixel takes and drops those past STENCIL_LIMIT before they are
blended (EdgeStencilLimit), which leaves every pixel as it was. It needs a
framebuffer with an 8-bit stencil buffer; without one, or with
SSN_EDGE_STENCIL_LIMIT=0, the edges are drawn as before.

Most frames change only the nodes or the HUD: a hover, a click, typing. The
window keeps the edges' last picture in a texture (EdgeImageCache) and copies
it while the camera, the window and the edges stay as they were, so those
frames cost the nodes alone. Print and captures always draw the edges.
"""

from __future__ import annotations

import os
import weakref

import numpy as np
from vispy import gloo, scene
from vispy.gloo.context import get_current_canvas
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual


POSITION_TEXTURE_ENVIRONMENT_VARIABLE = "SSN_EDGE_POSITION_TEXTURE"
STENCIL_ENVIRONMENT_VARIABLE = "SSN_EDGE_STENCIL_LIMIT"
CACHE_ENVIRONMENT_VARIABLE = "SSN_EDGE_IMAGE_CACHE"
_OFF_VALUES = {"0", "off", "false", "no"}
# The most edge fragments a pixel takes. All edges share one colour and alpha,
# and 8-bit blending leaves a pixel unchanged once alpha times its distance
# from the edge colour is under half a level: within about 190 layers for any
# alpha above 1/510 (below that an edge changes no pixel at all). So 255, all
# an 8-bit stencil counts, leaves every pixel as drawing every edge does.
STENCIL_LIMIT = 255
# The stencil bits the limit needs: it counts to STENCIL_LIMIT.
STENCIL_BITS_NEEDED = 8
# Node numbers travel to the GPU as float32 vertex attributes, exact below 2**24.
MAX_TEXTURE_NODES = 1 << 24
# Texels per row of the position texture. Both texture sides are powers of two,
# so the shader's texel arithmetic is exact.
TEXTURE_WIDTH = 2048
# ViewerEdges' visibility key before the first visible-edge upload; None means
# every node is visible.
_NOT_BUILT = object()


def position_texture_enabled(node_count):
    value = os.environ.get(POSITION_TEXTURE_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value not in _OFF_VALUES and node_count < MAX_TEXTURE_NODES


def stencil_limit_enabled():
    value = os.environ.get(STENCIL_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value not in _OFF_VALUES


# OpenGL 3.0 / OpenGL ES 2.0 names for asking a framebuffer about its stencil buffer.
_GL_FRAMEBUFFER = 0x8D40
_GL_FRAMEBUFFER_BINDING = 0x8CA6
_GL_STENCIL = 0x1802             # the stencil buffer of the default framebuffer
_GL_STENCIL_ATTACHMENT = 0x8D20  # that of a framebuffer object
_GL_ATTACHMENT_OBJECT_TYPE = 0x8CD0
_GL_ATTACHMENT_STENCIL_SIZE = 0x2217
_GL_NONE = 0
# glGetIntegerv's older question, answered by compatibility profiles only.
_GL_STENCIL_BITS = 0x0D57


def _clear_gl_errors(gl):
    """Take back the errors a failed question left, so VisPy's next check
    does not report them against a draw."""
    for _ in range(16):
        if not gl.glGetError():
            return


def framebuffer_stencil_bits(gl):
    """The stencil bits of the framebuffer bound now, or 0 when it has none
    or the driver cannot say. gl is vispy.gloo.gl, with a context current."""
    bits = 0
    try:
        bound = gl.glGetParameter(_GL_FRAMEBUFFER_BINDING)
        attachment = _GL_STENCIL_ATTACHMENT if bound else _GL_STENCIL
        kind = gl.glGetFramebufferAttachmentParameter(
            _GL_FRAMEBUFFER, attachment, _GL_ATTACHMENT_OBJECT_TYPE)
        if not gl.glGetError():
            if kind == _GL_NONE:
                return 0
            bits = gl.glGetFramebufferAttachmentParameter(
                _GL_FRAMEBUFFER, attachment, _GL_ATTACHMENT_STENCIL_SIZE)
            if not gl.glGetError() and isinstance(bits, int) and bits > 0:
                return bits
        _clear_gl_errors(gl)
        bits = gl.glGetParameter(_GL_STENCIL_BITS)
        if gl.glGetError() or not isinstance(bits, int) or bits < 0:
            bits = 0
    except Exception:
        bits = 0
    finally:
        try:
            _clear_gl_errors(gl)
        except Exception:
            pass
    return int(bits)


def _gl_renderer(gl):
    """GL_RENDERER of the current context, for the report."""
    try:
        return gl.glGetParameter(gl.GL_RENDERER) or "unknown renderer"
    except Exception as error:
        return f"unknown renderer: {error}"


def _pushed_framebuffer(canvas):
    """The framebuffer object canvas.render() or push_fbo draws into, or None
    for the window's own framebuffer."""
    stack = getattr(canvas, '_fb_stack', None)
    return stack[-1][0] if stack else None


class EdgeStencilLimit:
    """Whether the edges draw with the stencil overdraw limit (STENCIL_LIMIT).

    It is decided for the framebuffer they are drawn into. A window's is asked
    for its stencil bits on the first draw, and the choice is reported once
    (report, print by default) with the renderer, so a log shows it. A
    framebuffer object, as canvas.render() makes for print and captures, has a
    stencil buffer only when its maker attached one; render()'s has none, so
    prints are drawn as before.
    """

    def __init__(self, enabled=None, report=print):
        self.enabled = stencil_limit_enabled() if enabled is None else bool(enabled)
        self._report = report
        self._windows = weakref.WeakKeyDictionary()

    def applies(self, canvas):
        if not self.enabled or canvas is None:
            return False
        framebuffer = _pushed_framebuffer(canvas)
        if framebuffer is not None:
            return getattr(framebuffer, 'stencil_buffer', None) is not None
        try:
            decided = self._windows.get(canvas)
        except TypeError:
            return False
        if decided is None:
            decided = self._windows[canvas] = self._decide_for_window(canvas)
        return decided

    def _decide_for_window(self, canvas):
        from vispy.gloo import gl
        try:
            # Run what VisPy has queued, so the window's framebuffer is the one bound.
            canvas.context.flush_commands()
            bits = framebuffer_stencil_bits(gl)
        except Exception:
            bits = 0
        renderer = _gl_renderer(gl)
        if bits >= STENCIL_BITS_NEEDED:
            self._report(f"Edges: overdraw limit on ({bits} stencil bits; {renderer}).")
            return True
        self._report(
            f"Edges: overdraw limit off, the window has {bits} stencil bits "
            f"({renderer}); edges are drawn without it."
        )
        return False


def _limited_state():
    """The edges' GL state with the limit: a fragment passes while its pixel
    has taken fewer than STENCIL_LIMIT, and counts itself. Dithering is off,
    so a pixel's last value is the one blending settles on."""
    return dict(
        stencil_test=True,
        stencil_func=('greater', STENCIL_LIMIT, 0xFF),
        stencil_op=('keep', 'keep', 'incr'),
        stencil_mask=0xFF,
        dither=False,
    )


# What the edges leave behind after a limited draw: GL's defaults, which every
# other visual draws with.
_AFTER_LIMITED_STATE = dict(stencil_test=False, dither=True)
_UNLIMITED_STATE = dict(stencil_test=False)


def _draw_with_limit(visual, limit, draw):
    """Draw visual with the overdraw limit where it applies, then turn the
    stencil test off again, so markers and text never inherit it."""
    limited = limit is not None and limit.applies(get_current_canvas())
    visual.update_gl_state(**(_limited_state() if limited else _UNLIMITED_STATE))
    try:
        draw()
    finally:
        if limited:
            gloo.set_state(**_AFTER_LIMITED_STATE)


def image_cache_enabled():
    value = os.environ.get(CACHE_ENVIRONMENT_VARIABLE, "").strip().lower()
    return value not in _OFF_VALUES


def _draws_into_window(canvas):
    """Whether canvas draws into its window's own framebuffer now, rather than
    into a framebuffer object pushed for canvas.render() (print, captures)."""
    return _pushed_framebuffer(canvas) is None


# The cached picture, copied 1:1: each fragment reads the texel under it.
_PICTURE_VERTEX_SHADER = """
attribute vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

_PICTURE_FRAGMENT_SHADER = """
uniform sampler2D u_picture;
uniform vec2 u_size;
void main() {
    gl_FragColor = texture2D(u_picture, gl_FragCoord.xy / u_size);
}
"""

# Stencil buffers to try for the cache's framebuffer, smallest first: an
# 8-bit stencil alone (1 byte a pixel), which some drivers refuse, then
# GL_DEPTH24_STENCIL8 (4 bytes), then none, which only drops the overdraw
# limit while the edges are drawn into the cache.
_CACHE_STENCILS = (("stencil", 1), (0x88F0, 4), (None, 0))


class EdgeImageCache:
    """The edge layer's last picture over the background, reused while
    nothing it shows changes.

    A frame that changes only the nodes or the HUD (a hover, a click halo,
    typing, a status line, the sidebar sliding) copies the picture instead of
    drawing every edge again. The picture is drawn when its key changes (see
    _picture_key): the camera, the framebuffer's size and scale, the
    background, the edges themselves (EdgeSegmentsVisual.generation) or the
    overdraw limit. The Viewer draws its edges first, straight after the
    clear, so the copy, with blending off, leaves the framebuffer exactly as
    drawing them does.

    It serves only frames drawn into the window's own framebuffer: a draw into
    a pushed framebuffer object, as canvas.render() makes for print and
    captures, always draws the edges themselves. If the framebuffer or texture
    cannot be made, or GL reports an error, it says so once (report) and the
    edges are drawn directly from then on. SSN_EDGE_IMAGE_CACHE=0 turns it off.
    ``render_count`` counts pictures drawn, ``reuse_count`` frames that copied one.
    """

    def __init__(self, enabled=None, report=print):
        self.enabled = image_cache_enabled() if enabled is None else bool(enabled)
        self.failed = False
        self.render_count = 0
        self.reuse_count = 0
        self.bytes = 0
        self.stencil_kind = None
        self._report = report
        self._reported = False
        self._texture = None
        self._framebuffer = None
        self._stencil = None
        self._picture = None
        self._shape = None
        self._key = None
        self._canvas = None

    def serves(self, canvas):
        return (self.enabled and not self.failed and canvas is not None
                and _draws_into_window(canvas))

    def draw(self, canvas, key, draw_edges):
        """Copy the picture for key into the window's framebuffer, drawing it
        first with draw_edges when key is new. Returns False, having drawn
        nothing, when the cache cannot serve; the caller then draws directly."""
        try:
            shape = (int(round(canvas.physical_size[1])), int(round(canvas.physical_size[0])))
            if self._framebuffer is None or shape != self._shape or canvas is not self._canvas:
                self._allocate(canvas, shape)
            if key != self._key:
                self._key = None
                canvas.push_fbo(self._framebuffer, (0, 0), canvas.size)
                try:
                    canvas.context.clear(color=canvas.bgcolor, depth=True, stencil=True)
                    draw_edges()
                finally:
                    canvas.pop_fbo()
                self._check_gl(canvas)
                self._key = key
                self.render_count += 1
            else:
                self.reuse_count += 1
        except Exception as error:
            self._fail(canvas, error)
            return False
        try:
            gloo.set_state(blend=False, depth_test=False, stencil_test=False,
                           cull_face=False, dither=False)
            self._picture['u_size'] = (float(shape[1]), float(shape[0]))
            self._picture.draw('triangle_strip')
            # The state the edges' own draw leaves for the visuals after them.
            gloo.set_state('translucent', depth_test=False, dither=True)
        except Exception as error:
            self._fail(canvas, error)
            return False
        try:
            self._check_gl(canvas)
        except Exception as error:
            # The copy is drawn, so this frame keeps it; the next ones draw
            # the edges directly.
            self._fail(canvas, error)
        return True

    def _allocate(self, canvas, shape):
        """A colour texture the size of the window's framebuffer, with the
        smallest stencil buffer this GL accepts, checked complete now."""
        self.free()
        texture = gloo.Texture2D(shape=shape + (4,), internalformat='rgba8',
                                 interpolation='nearest', wrapping='clamp_to_edge')
        refusals = []
        for stencil_format, stencil_bytes in _CACHE_STENCILS:
            framebuffer = gloo.FrameBuffer(color=texture)
            stencil = None
            if stencil_format is not None:
                stencil = gloo.RenderBuffer(shape, format=stencil_format)
                framebuffer.stencil_buffer = stencil
            try:
                # VisPy checks a framebuffer's completeness when it is first bound.
                canvas.push_fbo(framebuffer, (0, 0), canvas.size)
                try:
                    canvas.context.flush_commands()
                finally:
                    canvas.pop_fbo()
                    canvas.context.flush_commands()
                self._check_gl(canvas)
            except Exception as error:
                refusals.append(f"{stencil_format}: {error}")
                for buffer in (framebuffer, stencil):
                    if buffer is not None:
                        buffer.delete()
                continue
            self._texture, self._framebuffer, self._stencil = texture, framebuffer, stencil
            self.stencil_kind = {"stencil": "8-bit stencil", None: "no stencil"}.get(
                stencil_format, "depth24/stencil8")
            break
        else:
            texture.delete()
            raise RuntimeError("no framebuffer for the cache: " + "; ".join(refusals))
        self.bytes = shape[0] * shape[1] * (4 + stencil_bytes)
        self._shape = shape
        if self._canvas is not canvas:
            self._canvas = canvas
            close = getattr(getattr(canvas, 'events', None), 'close', None)
            if close is not None:
                close.connect(lambda _event: self.free())
        if self._picture is None:
            self._picture = gloo.Program(_PICTURE_VERTEX_SHADER, _PICTURE_FRAGMENT_SHADER)
            self._picture['a_position'] = gloo.VertexBuffer(
                np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32))
        self._picture['u_picture'] = self._texture
        if not self._reported:
            self._reported = True
            self._report(
                f"Edges: image cache on ({shape[1]}x{shape[0]}, {self.bytes / 2**20:.0f} MB, "
                f"{self.stencil_kind}; {_gl_renderer(_gl())})."
            )

    def _check_gl(self, canvas):
        canvas.context.flush_commands()
        error = _gl().glGetError()
        if error:
            _clear_gl_errors(_gl())
            raise RuntimeError(f"GL error 0x{error:04x}")

    def _fail(self, canvas, error):
        self.failed = True
        try:
            self.free()
        except Exception:
            pass
        try:
            _clear_gl_errors(_gl())
        except Exception:
            pass
        self._report(
            f"Edges: image cache off, {error} ({_gl_renderer(_gl())}); "
            "edges are drawn directly."
        )

    def free(self):
        """Release the texture and framebuffer; the next frame makes new ones."""
        for name in ('_framebuffer', '_stencil', '_texture'):
            item = getattr(self, name)
            setattr(self, name, None)
            if item is not None:
                item.delete()
        self._shape = None
        self._key = None
        self.bytes = 0


def _gl():
    from vispy.gloo import gl
    return gl


def _picture_key(visual, canvas):
    """Everything the edges' picture depends on, besides what bumps
    visual.generation: the visual's mapping to the framebuffer, the
    framebuffer's size and scale, the background and the overdraw limit."""
    transform = visual.transforms.get_transform(map_from='visual', map_to='render')
    corners = np.asarray(transform.map(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
                         dtype=np.float64)
    limit = getattr(visual, 'stencil_limit', None)
    return (
        visual.generation,
        visual.edge_style(),
        tuple(corners.ravel().tolist()),
        tuple(float(value) for value in canvas.physical_size),
        float(canvas.pixel_scale),
        tuple(float(value) for value in canvas.bgcolor.rgba),
        None if limit is None else (limit.enabled, STENCIL_LIMIT),
    )


def _draw_edges(visual, draw):
    """Draw the edges: from the image cache when it serves this frame,
    otherwise directly, with the overdraw limit where it applies."""
    canvas = get_current_canvas()
    cache = getattr(visual, 'image_cache', None)
    limit = getattr(visual, 'stencil_limit', None)
    if cache is not None and cache.serves(canvas):
        try:
            key = _picture_key(visual, canvas)
        except Exception as error:
            cache._fail(canvas, error)
        else:
            if cache.draw(canvas, key, lambda: _draw_with_limit(visual, limit, draw)):
                return
    _draw_with_limit(visual, limit, draw)


def position_texels(positions):
    """Lay node ``i``'s (x, y) into texel (row i // TEXTURE_WIDTH, column i % TEXTURE_WIDTH).

    Returns a (rows, TEXTURE_WIDTH, 4) float32 RGBA array, rows a power of two;
    the unused channels and texels are zero.
    """
    positions = np.asarray(positions, dtype=np.float32)[:, :2]
    count = len(positions)
    rows = 1 << max(0, -(-count // TEXTURE_WIDTH) - 1).bit_length()
    texels = np.zeros((rows, TEXTURE_WIDTH, 4), dtype=np.float32)
    texels.reshape(-1, 4)[:count, :2] = positions
    return texels


def _contiguous_line_positions(positions):
    """Normalize VisPy line vertices before uploading them to the GPU."""
    result = np.ascontiguousarray(positions, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] not in (2, 3):
        raise ValueError(
            "Line positions must have shape (N, 2) or (N, 3); "
            f"received {result.shape}."
        )
    if not np.isfinite(result).all():
        raise ValueError("Line positions contain NaN or infinite values.")
    return result


def _edge_segment_positions(positions, edges):
    """Return one Line upload holding both endpoints of every edge.

    Rows ``2k`` and ``2k + 1`` are the source and target of edge ``k``, the
    vertex order of VisPy's ``connect='segments'``. One fancy-indexing gather
    replaces a per-edge Python loop, which took seconds on multi-million-edge
    networks during startup.
    """
    positions = np.asarray(positions)
    pairs = np.asarray(edges, dtype=np.intp).reshape(-1, 2)
    return _contiguous_line_positions(
        positions[pairs].reshape(-1, positions.shape[1])
    )


def _gather_edges(edges, rows):
    """Return ``edges[rows]`` for an (E, 2) int32 array.

    Each edge's two int32 node numbers are read as one int64, which makes the
    random gather about four times faster than indexing rows of two.
    """
    pairs = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
    return pairs.view(np.int64).reshape(-1)[rows].view(np.int32).reshape(-1, 2)


def _descending_float32_order(scores):
    """Stable descending order of float32 scores and the sorted scores.

    Each score's bits become an unsigned key that sorts like the float; the
    key, inverted for descending order, fills the high 32 bits and the edge
    number the low ones. One sort of those 64-bit values (NumPy's vectorized
    sort) is the stable order: 1.3 s for 48.7 million scores, against 4.2 s
    for argsort. The sorted scores come back from the high bits, with no
    gather. NaN scores are kept; the caller drops them.
    """
    bits = scores.view(np.uint32)
    ascending = np.where(bits >> 31, ~bits, bits | np.uint32(0x80000000))
    packed = (~ascending).astype(np.uint64) << np.uint64(32)
    del ascending
    packed |= np.arange(len(scores), dtype=np.uint64)
    packed.sort()
    order = (packed & np.uint64(0xFFFFFFFF)).astype(np.intp)
    ascending = ~(packed >> np.uint64(32)).astype(np.uint32)
    del packed
    sorted_bits = np.where(ascending >> 31, ascending & np.uint32(0x7FFFFFFF), ~ascending)
    return order, sorted_bits.view(np.float32)


def _descending_order(scores):
    """Stable descending order of numeric scores, NaN scores left out.

    Returns the order and the scores in that order.
    """
    scores = np.asarray(scores)
    if scores.dtype == np.float32 and len(scores) < (1 << 32):
        order, ordered = _descending_float32_order(np.ascontiguousarray(scores))
    else:
        count = len(scores)
        # A stable ascending sort of the reversed scores, read backwards, is
        # a stable descending sort of the scores.
        order = (count - 1) - np.argsort(scores[::-1], kind="stable")[::-1]
        ordered = scores[order]
    if np.issubdtype(ordered.dtype, np.floating):
        valid = ~np.isnan(ordered)
        if not valid.all():
            order, ordered = order[valid], ordered[valid]
    return order, ordered


class EdgeOrder:
    """The network's edges by descending score, for prefix thresholds.

    ``edges`` and ``scores`` hold the drawable edges in that order: every edge
    whose score can pass a threshold (NaN never does). Without scores, every
    edge passes, as in Viewer_Visual_State.edge_stages.
    """

    def __init__(self, edges, scores=None):
        edges = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 2)
        if scores is None or len(scores) == 0:
            self.edges = edges
            self.scores = None
            return
        scores = np.asarray(scores)
        if len(scores) != len(edges):
            raise ValueError(
                f"Edge scores ({len(scores)}) must match the edges ({len(edges)})."
            )
        order, self.scores = _descending_order(scores)
        self.edges = _gather_edges(edges, order)

    def __len__(self):
        return len(self.edges)

    def count_at_or_above(self, threshold):
        """Return how many leading edges have ``score >= threshold``.

        Each probe compares one score with the threshold as NumPy compares the
        whole array in edge_stages, under the same type promotion, so the count
        is the one ``scores >= threshold`` gives.
        """
        if self.scores is None:
            return len(self.edges)
        scores = self.scores
        low, high = 0, len(scores)
        while low < high:
            middle = (low + high) // 2
            if scores[middle] >= threshold:
                low = middle + 1
            else:
                high = middle
        return low


_VERTEX_SHADER = """
uniform sampler2D u_node_xy;
uniform float u_texture_width;
// 1 / texture width and 1 / texture height, both powers of two.
uniform vec2 u_texel_size;
attribute float a_node;

void main() {
    // Node n is the texel in column n mod width of row n div width. Scaling
    // by a power of two is exact, so floor() finds the row of every node
    // number below 2**24, and the texture is sampled at texel centres.
    float row = floor(a_node * u_texel_size.x);
    float column = a_node - row * u_texture_width;
    vec2 xy = texture2DLod(u_node_xy, (vec2(column, row) + 0.5) * u_texel_size, 0.0).xy;
    gl_Position = $transform(vec4(xy, 0.0, 1.0));
}
"""

_FRAGMENT_SHADER = """
uniform vec4 u_color;

void main() {
    gl_FragColor = u_color;
}
"""


class EdgeSegmentsVisual(Visual):
    """Line segments between nodes whose positions live in a float texture.

    Draws the first ``count`` segments of one of its node-number buffers,
    each two vertices ``(source, target)``, in one colour, like VisPy's
    Line visual with ``connect='segments'`` and the 'gl' method.
    """

    def __init__(self, color=(0.0, 0.0, 0.0, 1.0), width=1.0):
        Visual.__init__(self, vcode=_VERTEX_SHADER, fcode=_FRAGMENT_SHADER)
        self._draw_mode = 'lines'
        self._index_buffer = None
        self._width = float(width)
        self._texture = None
        self._bounds = None
        self._buffers = {}
        self._buffer_sizes = {}
        self._segment_count = 0
        self._drawn = None
        # The EdgeStencilLimit it draws with; None draws every fragment.
        self.stencil_limit = None
        # The EdgeImageCache it draws through; None draws directly.
        self.image_cache = None
        # Bumped by every change to what it draws (positions, segments, the
        # drawn count, colour), so the image cache knows its picture is old.
        self.generation = 0
        self._color = tuple(float(value) for value in color)
        self.shared_program['u_color'] = self._color
        self.set_gl_state('translucent')

    @property
    def segment_count(self):
        return self._segment_count

    def edge_style(self):
        return (self._color, self._width)

    def set_color(self, color):
        self._color = tuple(float(value) for value in color)
        self.shared_program['u_color'] = self._color
        self.generation += 1
        self.update()

    def set_width(self, width):
        self._width = float(width)
        self.generation += 1
        self.update()

    def draw(self):
        _draw_edges(self, lambda: Visual.draw(self))

    def set_node_positions(self, positions):
        """Upload node ``i``'s (x, y) to texel ``i``; positions must be finite."""
        positions = np.asarray(positions, dtype=np.float32)[:, :2]
        count = len(positions)
        texels = position_texels(positions)
        rows = len(texels)
        if self._texture is not None and self._texture.shape[:2] == (rows, TEXTURE_WIDTH):
            self._texture.set_data(texels)
        else:
            self._texture = gloo.Texture2D(
                texels, internalformat='rgba32f',
                interpolation='nearest', wrapping='clamp_to_edge',
            )
            self.shared_program['u_node_xy'] = self._texture
            self.shared_program['u_texture_width'] = float(TEXTURE_WIDTH)
            self.shared_program['u_texel_size'] = (1.0 / TEXTURE_WIDTH, 1.0 / rows)
        if count:
            low = positions.min(axis=0)
            high = positions.max(axis=0)
            self._bounds = [(float(low[d]), float(high[d])) for d in range(2)]
        else:
            self._bounds = None
        self.generation += 1
        self.update()

    def set_segments(self, key, edges):
        """Upload the (E, 2) node numbers of ``edges`` as buffer ``key``."""
        node_numbers = np.ascontiguousarray(edges, dtype=np.float32).reshape(-1)
        buffer = self._buffers.get(key)
        if buffer is None or len(node_numbers) == 0:
            # A fresh buffer each time it empties, as a zero-size upload is
            # not one every GL driver accepts.
            buffer = gloo.VertexBuffer(node_numbers if len(node_numbers) else
                                       np.zeros(2, dtype=np.float32))
            self._buffers[key] = buffer
        else:
            buffer.set_data(node_numbers)
        self._buffer_sizes[key] = len(node_numbers) // 2
        self.generation += 1

    def drop_segments(self, key):
        """Free buffer ``key``; the next set_segments for it uploads afresh."""
        buffer = self._buffers.pop(key, None)
        if buffer is not None:
            buffer.delete()
        self._buffer_sizes.pop(key, None)
        self.generation += 1

    def draw_segments(self, key, count):
        """Draw the first ``count`` segments of buffer ``key`` from now on."""
        available = self._buffer_sizes.get(key, 0)
        count = max(0, min(int(count), available))
        self._segment_count = count
        if (key, count) != self._drawn:
            self._drawn = (key, count)
            self.generation += 1
        if count:
            # A view of the buffer's first 2 * count vertices sets the vertex
            # count VisPy passes to glDrawArrays; nothing is uploaded.
            self.shared_program['a_node'] = self._buffers[key][:2 * count]
        self.update()

    def _prepare_transforms(self, view):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._segment_count == 0 or self._texture is None or self._width == 0:
            return False
        # The line state VisPy's 'gl' Line method sets, so other visuals'
        # state cannot leak into the edges.
        width = self.transforms.pixel_scale * self._width
        self.update_gl_state(line_smooth=False, line_width=max(width, 1.0))

    def _compute_bounds(self, axis, view):
        if self._bounds is None or axis > 1:
            return None
        return self._bounds[axis]


EdgeSegments = create_visual_node(EdgeSegmentsVisual)


class LimitedLine(scene.visuals.Line):
    """VisPy's Line, for the coordinate fallback, drawn with an EdgeStencilLimit
    and through an EdgeImageCache, as EdgeSegmentsVisual is."""

    def __init__(self, *args, stencil_limit=None, image_cache=None, **kwargs):
        # Set before VisPy freezes the visual's attributes.
        self.stencil_limit = stencil_limit
        self.image_cache = image_cache
        self.generation = 0
        super().__init__(*args, **kwargs)

    def edge_style(self):
        # Colour and width reach the Line through set_data, which bumps generation.
        return ()

    def set_data(self, *args, **kwargs):
        self.generation += 1
        return super().set_data(*args, **kwargs)

    def draw(self):
        _draw_edges(self, super().draw)


class ViewerEdges:
    """The Viewer's edge layer: the score order and the visual drawing it.

    show_threshold draws the edges at or above a threshold whose endpoints are
    both visible; show_edges draws a given edge list. Each uploads only what
    changed since the previous call: node positions, the visible edges, or
    the explicit list. ``upload_count`` counts edge-buffer uploads, and
    ``position_upload_count`` node-position uploads. Either visual draws with
    ``stencil_limit`` (an EdgeStencilLimit unless one is given) and through
    ``image_cache`` (an EdgeImageCache unless one is given).
    """

    def __init__(self, parent, edges, scores, positions, color, width,
                 name='network_edges', use_position_texture=None, stencil_limit=None,
                 image_cache=None):
        self.order = EdgeOrder(edges, scores)
        node_count = len(positions)
        if len(self.order) and (
            int(self.order.edges.min()) < 0
            or int(self.order.edges.max()) >= node_count
        ):
            raise ValueError(
                f"Edge endpoints must be node indices below the node count ({node_count})."
            )
        if use_position_texture is None:
            use_position_texture = position_texture_enabled(node_count)
        self.uses_position_texture = bool(use_position_texture)
        self.stencil_limit = EdgeStencilLimit() if stencil_limit is None else stencil_limit
        self.image_cache = EdgeImageCache() if image_cache is None else image_cache
        if self.uses_position_texture:
            self.visual = EdgeSegments(color=color, width=width, parent=parent, name=name)
            self.visual.stencil_limit = self.stencil_limit
            self.visual.image_cache = self.image_cache
        else:
            self.visual = LimitedLine(
                connect='segments', color=color, width=width, parent=parent, name=name,
                stencil_limit=self.stencil_limit, image_cache=self.image_cache,
            )
        self.visual.set_gl_state('translucent', depth_test=False)
        # Nothing is uploaded before the first show_threshold or show_edges.
        self.visual.visible = False
        self.upload_count = 0
        self.position_upload_count = 0
        self.drawn_edge_count = 0
        self._positions = None
        self._visibility = _NOT_BUILT
        self._visible_counts = None
        self._visible_edges = None
        self._explicit = None
        self._drawn_key = None

    @property
    def visible(self):
        return self.visual.visible

    @visible.setter
    def visible(self, value):
        if bool(value) != bool(self.visual.visible):
            self.visual.generation += 1
        self.visual.visible = value

    def _upload_positions(self, positions):
        """Send the node positions if they changed; return whether they did."""
        if self._positions is not None and np.array_equal(positions, self._positions):
            return False
        xy = np.asarray(positions, dtype=np.float32)[:, :2]
        if not np.isfinite(xy).all():
            raise ValueError("Node positions contain NaN or infinite values.")
        self._positions = np.array(positions, copy=True)
        if self.uses_position_texture:
            self.visual.set_node_positions(xy)
            self.position_upload_count += 1
        return True

    def _update_visibility(self, visible_mask):
        """Rebuild the visible edges, in score order, when the mask changed."""
        if visible_mask is None:
            key = None
        else:
            visible = np.asarray(visible_mask, dtype=bool)
            key = None if visible.all() else visible.tobytes()
        if self._visibility is not _NOT_BUILT and self._visibility == key:
            return False
        edges = self.order.edges
        if key is None:
            visible_edges, counts = edges, None
        else:
            keep = visible[edges[:, 0]] & visible[edges[:, 1]]
            visible_edges = edges[keep]
            counts = np.cumsum(keep, dtype=np.int64 if len(keep) >= (1 << 31) else np.int32)
        self._visibility = key
        self._visible_counts = counts
        if self.uses_position_texture:
            self.visual.set_segments('threshold', visible_edges)
            self.upload_count += 1
            self._visible_edges = None
        else:
            self._visible_edges = visible_edges
        return True

    def show_threshold(self, positions, visible_mask, threshold):
        """Draw the edges scoring at least ``threshold`` between visible nodes.

        Returns how many edges are drawn.
        """
        if self._explicit is not None:
            self._explicit = None
            if self.uses_position_texture:
                self.visual.drop_segments('explicit')
        moved = self._upload_positions(positions)
        rebuilt = self._update_visibility(visible_mask)
        qualified = self.order.count_at_or_above(threshold)
        counts = self._visible_counts
        drawn = qualified if counts is None else (int(counts[qualified - 1]) if qualified else 0)
        if self.uses_position_texture:
            self.visual.draw_segments('threshold', drawn)
        else:
            key = ('threshold', self._visibility, drawn)
            if moved or rebuilt or key != self._drawn_key:
                self._upload_line(positions, self._visible_edges[:drawn])
                self._drawn_key = key
        self.drawn_edge_count = drawn
        return drawn

    def show_edges(self, positions, edges):
        """Draw exactly ``edges``, an (E, 2) array of node indices.

        The list is uploaded again whenever a different array object is passed.
        """
        moved = self._upload_positions(positions)
        changed = edges is not self._explicit
        self._explicit = edges
        drawn = len(edges)
        if self.uses_position_texture:
            if changed:
                self.visual.set_segments('explicit', edges)
                self.upload_count += 1
            self.visual.draw_segments('explicit', drawn)
        elif moved or changed or self._drawn_key != ('explicit',):
            self._upload_line(positions, edges)
            self._drawn_key = ('explicit',)
        self.drawn_edge_count = drawn
        return drawn

    def _upload_line(self, positions, edges):
        if len(edges):
            self.visual.set_data(pos=_edge_segment_positions(np.asarray(positions)[:, :2], edges))
            self.upload_count += 1
