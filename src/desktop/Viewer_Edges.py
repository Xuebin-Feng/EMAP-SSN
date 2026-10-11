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
"""

from __future__ import annotations

import os

import numpy as np
from vispy import gloo, scene
from vispy.scene.visuals import create_visual_node
from vispy.visuals import Visual


POSITION_TEXTURE_ENVIRONMENT_VARIABLE = "SSN_EDGE_POSITION_TEXTURE"
_OFF_VALUES = {"0", "off", "false", "no"}
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
        self.shared_program['u_color'] = tuple(float(value) for value in color)
        self.set_gl_state('translucent')

    @property
    def segment_count(self):
        return self._segment_count

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

    def drop_segments(self, key):
        """Free buffer ``key``; the next set_segments for it uploads afresh."""
        buffer = self._buffers.pop(key, None)
        if buffer is not None:
            buffer.delete()
        self._buffer_sizes.pop(key, None)

    def draw_segments(self, key, count):
        """Draw the first ``count`` segments of buffer ``key`` from now on."""
        available = self._buffer_sizes.get(key, 0)
        count = max(0, min(int(count), available))
        self._segment_count = count
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


class ViewerEdges:
    """The Viewer's edge layer: the score order and the visual drawing it.

    show_threshold draws the edges at or above a threshold whose endpoints are
    both visible; show_edges draws a given edge list. Each uploads only what
    changed since the previous call: node positions, the visible edges, or
    the explicit list. ``upload_count`` counts edge-buffer uploads, and
    ``position_upload_count`` node-position uploads.
    """

    def __init__(self, parent, edges, scores, positions, color, width,
                 name='network_edges', use_position_texture=None):
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
        if self.uses_position_texture:
            self.visual = EdgeSegments(color=color, width=width, parent=parent, name=name)
        else:
            self.visual = scene.visuals.Line(
                connect='segments', color=color, width=width, parent=parent, name=name,
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
