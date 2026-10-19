# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's node markers on a hover: what changes, and how little to send.

MainViewer.update_nodes sends every marker attribute again, which a colour,
selection, hide or size change needs. A hover changes one face colour (two,
with the node it leaves), so HoverMarkers rewrites only those markers'
records in VisPy's marker buffer. That buffer is VisPy's own (MarkersVisual's
``_data`` and ``_vbo``, as in VisPy VISPY_VERSION): HoverMarkers checks its
layout when update_nodes sends the markers, and if it is not the expected one
it says so once and every hover sends the markers whole, as before.

uniform_symbol lets update_nodes pass one symbol name when every node has
the same shape, which spares VisPy a Python lookup per node.

MarkerPicker finds the node the cursor is on as the markers are drawn: the
topmost one whose drawn shape covers the pixel under the cursor, by the
signed distances VisPy's marker shaders use, at the display's scale.
"""

from __future__ import annotations

import functools

import numpy as np

# The VisPy release whose MarkersVisual buffer layout this module writes into
# (tests/test_viewer_markers pins it).
VISPY_VERSION = "0.17.0"
MARKER_FIELDS = ('a_position', 'a_fg_color', 'a_bg_color', 'a_size', 'a_edgewidth', 'a_symbol')


def uniform_symbol(shapes):
    """The one symbol every marker has, or None when they differ (or there are none)."""
    if len(shapes) == 0:
        return None
    first = shapes[0]
    if not isinstance(first, str):
        return None
    return first if bool(np.all(shapes == first)) else None


def marker_layout_problem(markers, count):
    """Why VisPy's marker buffer is not the layout HoverMarkers writes into,
    or None when it is: a structured array ``_data`` of MARKER_FIELDS, one
    record a marker, and a vertex buffer ``_vbo`` of the same records."""
    data = getattr(markers, '_data', None)
    vbo = getattr(markers, '_vbo', None)
    if not isinstance(data, np.ndarray) or data.dtype.names != MARKER_FIELDS:
        return f"its marker records are {getattr(getattr(data, 'dtype', None), 'names', None)}"
    if data['a_bg_color'].shape[1:] != (4,) or data.dtype['a_bg_color'].base != np.float32:
        return "its face colours are not four float32 values"
    if len(data) != count:
        return f"it holds {len(data)} markers, not {count}"
    if vbo is None or not hasattr(vbo, 'set_subdata') or getattr(vbo, 'size', None) != count \
            or getattr(vbo, 'dtype', None) != data.dtype:
        return "its vertex buffer does not hold the marker records"
    return None


class HoverMarkers:
    """What update_nodes sent, so that a hover can rewrite two face colours.

    capture() records the inputs update_nodes drew from. apply() first checks
    that they are still what the markers show (colours, sizes, shapes,
    positions, hidden nodes, outlines, draw order, click rings); a hover then
    recolours the markers of the node it leaves and the node it reaches, each
    as update_nodes colours it, and sends those records alone. When anything
    else has changed, or the buffer layout is not VisPy's expected one, apply()
    returns False and the caller sends the markers whole.
    """

    _layout_reported = False

    def __init__(self, viewer, colors_source, sizes, shapes, outlines, identify_connected,
                 left_clicked, marker_node_order, marker_ring_mask, report=None):
        report = print if report is None else report
        markers = viewer.markers
        self.problem = marker_layout_problem(markers, len(marker_node_order))
        if self.problem is not None and not HoverMarkers._layout_reported:
            HoverMarkers._layout_reported = True
            report(f"Nodes: hover updates send every marker, as VisPy's marker buffer is not "
                   f"the expected one ({self.problem}; VisPy {_vispy_version()}).")
        self._data = getattr(markers, '_data', None)
        self._colors = np.array(colors_source, copy=True)
        self._sizes = np.array(sizes, copy=True)
        self._shapes = np.array(shapes, copy=True)
        self._positions = np.array(viewer.pos, copy=True)
        self._visible = np.array(viewer.visible_mask, dtype=bool, copy=True)
        self._outlines = tuple(np.array(part, copy=True) for part in outlines)
        self._identify = bool(identify_connected)
        self._left_clicked = np.array(left_clicked, copy=True)
        self._draw_order = np.array(viewer._submitted_visible_node_order, copy=True)
        self.marker_node_order = np.array(marker_node_order, copy=True)
        self.marker_ring_mask = np.array(marker_ring_mask, dtype=bool, copy=True)

    def _unchanged(self, viewer):
        import EMAPSSN_Config as cfg
        if getattr(viewer, 'transient_marks_hidden', False):
            return False
        if getattr(viewer.markers, '_data', None) is not self._data:
            return False
        colors = viewer.current_colors
        if colors.dtype != self._colors.dtype or not np.array_equal(colors, self._colors):
            return False
        sizes = getattr(viewer, 'current_sizes', cfg.NODE_SIZE)
        if not np.array_equal(np.asarray(sizes), self._sizes):
            return False
        if not np.array_equal(viewer.pos, self._positions):
            return False
        if not np.array_equal(np.asarray(viewer.visible_mask, dtype=bool), self._visible):
            return False
        shapes = getattr(viewer, 'current_shapes', None)
        if shapes is None or len(shapes) != len(self._shapes) or not bool(np.all(shapes == self._shapes)):
            return False
        identify = viewer._connected_node_identification_enabled()
        if identify != self._identify:
            return False
        outlines = viewer.node_outlines(identify)
        if not all(np.array_equal(now, then) for now, then in zip(outlines, self._outlines)):
            return False
        if not np.array_equal(viewer._left_click_node_indices(), self._left_clicked):
            return False
        return np.array_equal(viewer.visible_node_render_order(identify_connected=identify),
                              self._draw_order)

    def apply(self, viewer, left, reached):
        """Show the hover moving from node ``left`` to node ``reached`` (either
        may be None). Returns False, having changed nothing, when the markers
        must be sent whole instead."""
        if self.problem is not None or not self._unchanged(viewer):
            return False
        import EMAPSSN_Config as cfg
        import matplotlib.colors as mcolors
        from vispy.color import ColorArray
        data = self._data
        vbo = viewer.markers._vbo
        for node in {left, reached} - {None}:
            slots = np.flatnonzero(self.marker_node_order == node)
            if len(slots) == 0:
                continue  # a hidden node has no marker
            # The face colours update_nodes gives these markers: the node's
            # colour, the hover colour while it is hovered, and half the
            # opacity on a click ring.
            rows = np.repeat(viewer.current_colors[node:node + 1], len(slots), axis=0)
            if node == reached:
                rows[:] = mcolors.to_rgba(cfg.HOVER_COLOR)
            rows[self.marker_ring_mask[slots], 3] *= 0.5
            data['a_bg_color'][slots] = ColorArray(rows).rgba
            # A click ring sits right before its node, so the slots run on.
            first, last = int(slots.min()), int(slots.max())
            vbo.set_subdata(data[first:last + 1], offset=first, copy=True)
        viewer.markers.update()
        return True


def _vispy_version():
    try:
        import vispy
        return vispy.__version__
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Which node the cursor is on.
#
# VisPy draws each marker as a point sprite T = size + 4 * (outline + 1.5 *
# antialias) device pixels across, and its fragment shader keeps the pixels
# where the symbol's signed distance r is at most half the outline width plus
# the antialias band (vispy/visuals/markers.py). The distances below are those
# shaders' arithmetic, with (dx, dy) = (gl_PointCoord - 0.5) * T: the pixel
# centre less the marker's centre in framebuffer pixels, with y pointing down
# (VisPy's framebuffer coordinates point it up).

_SQRT2 = np.sqrt(2.0)
_SQRT3 = np.sqrt(3.0)


def _disc(dx, dy, s):
    return np.hypot(dx, dy) - s / 2.0


def _arrow(dx, dy, s):
    ady = np.abs(dy)
    r1 = np.abs(dx) + ady - s / 2.0
    r2 = dx + 0.25 * s + ady - s / 2.0
    return np.maximum(r1, -r2) / _SQRT2


def _ring(dx, dy, s):
    length = np.hypot(dx, dy)
    return np.maximum(length - s / 2.0, -(length - s / 4.0))


def _clobber(dx, dy, s):
    pi = 3.14159265358979323846264
    radius = 0.32 * s
    shift = 0.36 / _SQRT3 * s
    x, y = dx, dy - 0.25 * shift
    angles = (-pi / 2, -pi / 2 + 2 * pi / 3, -pi / 2 + 4 * pi / 3)
    return np.minimum.reduce([np.hypot(x - np.cos(t) * shift, y - np.sin(t) * shift) - radius
                              for t in angles])


def _square(dx, dy, s):
    return np.maximum(np.abs(dx), np.abs(dy)) - s / 2.0


def _x(dx, dy, s):
    u, w = np.abs((dx + dy) / _SQRT2), np.abs((dy - dx) / _SQRT2)
    vbar = np.maximum(u - s / 6.0, w - s / 2.0)
    hbar = np.maximum(w - s / 6.0, u - s / 2.0)
    return np.minimum(vbar, hbar)


def _diamond(dx, dy, s):
    return (np.abs(dx) + np.abs(dy) - s / 2.0) / _SQRT2


def _vbar(dx, dy, s):
    return np.maximum(np.abs(dx) - s / 6.0, np.abs(dy) - s / 2.0)


def _hbar(dx, dy, s):
    return np.maximum(np.abs(dy) - s / 6.0, np.abs(dx) - s / 2.0)


def _cross(dx, dy, s):
    return np.minimum(_vbar(dx, dy, s), _hbar(dx, dy, s))


def _tailed_arrow(dx, dy, s):
    half = s / 2.0
    ady = np.abs(dy)
    arrow = np.maximum(np.abs(dx) + ady - half, -(dx + 0.25 * s + ady - half))
    upper_bottom = ady - s / 8.0 / _SQRT2
    left = -dx - half
    right = dx + ady - half
    bar = np.maximum(upper_bottom, left)
    slanted = right >= bar
    bar = np.where(slanted, right, bar)
    scale = np.where(slanted, _SQRT2, 1.0)
    return np.where(arrow <= bar, arrow / _SQRT2, bar / scale)


def _triangle(dx, dy, s, sign):
    height = sign * s * _SQRT3 / 2.0
    bottom = sign * (dy - height / 2.0)
    rotated = _SQRT3 / 2.0 * dx - 0.5 * (dy - height / 6.0) + height / 6.0
    counter = -_SQRT3 / 2.0 * dx - 0.5 * (dy - height / 6.0) + height / 6.0
    edges = np.maximum(sign * (rotated - height / 2.0), sign * (counter - height / 2.0))
    return np.maximum(edges, bottom)


def _star(dx, dy, s):
    pi2_5 = 3.141592653589 * 2.0 / 5.0
    pi2_20 = 3.141592653589 / 10.0
    shift_y = -0.05 * s
    top = -s / 2.0
    star = None
    for i in range(5):
        if i == 0:
            spike_x, spike_y = dx, dy + shift_y
        else:
            angle = i * pi2_5
            spike_x = np.cos(angle) * dx - np.sin(angle) * (dy + shift_y)
            spike_y = np.sin(angle) * dx + np.cos(angle) * (dy + shift_y)
        rot18x = np.cos(pi2_20) * spike_x - np.sin(pi2_20) * (spike_y - top)
        rot_18x = np.cos(pi2_20) * spike_x + np.sin(pi2_20) * (spike_y - top)
        spike = np.maximum(spike_y - s / 10.0, np.maximum(rot18x, -rot_18x))
        star = spike if star is None else np.minimum(star, spike)
    return star


def _cross_lines(dx, dy, s):
    vbar = np.maximum(np.abs(dx), np.abs(dy) - s / 2.0)
    hbar = np.maximum(np.abs(dy), np.abs(dx) - s / 2.0)
    return np.minimum(vbar, hbar)


_DISTANCES = {
    'disc': _disc, 'arrow': _arrow, 'ring': _ring, 'clobber': _clobber, 'square': _square,
    'x': _x, 'diamond': _diamond, 'vbar': _vbar, 'hbar': _hbar, 'cross': _cross,
    'tailed_arrow': _tailed_arrow,
    'triangle_up': lambda dx, dy, s: _triangle(dx, dy, s, 1.0),
    'triangle_down': lambda dx, dy, s: _triangle(dx, dy, s, -1.0),
    'star': _star, 'cross_lines': _cross_lines,
}


@functools.lru_cache(maxsize=1)
def _symbol_codes():
    """VisPy's shader code for each symbol name, and the name for each code."""
    from vispy.visuals.markers import symbol_shader_values, symbol_shaders
    return dict(symbol_shader_values), {i: name for i, name in enumerate(symbol_shaders)}


def marker_distance(codes, dx, dy, size):
    """VisPy's signed distance of each pixel offset (dx, dy) from its marker's
    outline: symbol code codes, drawn size device pixels across."""
    _, names = _symbol_codes()
    codes = np.asarray(codes).astype(np.int64)
    dx, dy, size = (np.asarray(value, dtype=np.float64) for value in (dx, dy, size))
    result = np.full(dx.shape, np.inf)
    for code in np.unique(codes):
        chosen = codes == code
        result[chosen] = _DISTANCES[names[int(code)]](dx[chosen], dy[chosen], size[chosen])
    return result


def symbol_codes_of(shapes):
    """The shader's symbol code for each shape name."""
    values, _ = _symbol_codes()
    single = uniform_symbol(shapes)
    if single is not None:
        return np.full(len(shapes), values[single], dtype=np.int64)
    return np.array([values[name] for name in shapes], dtype=np.int64)


# Over no node, the cursor still hovers a node whose drawn shape is within
# this many logical pixels: about the slack the old circular pick gave a
# default node (its radius was two thirds of the node size).
HOVER_MARGIN_PX = 1.0


@functools.lru_cache(maxsize=8)
def _margin_rings(radius):
    """Whole-pixel offsets within radius of a pixel, grouped by distance,
    nearest first: [(0, 0)], then the four at 1, the four at sqrt(2), and so on."""
    reach = int(np.floor(radius))
    ys, xs = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    offsets = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    distances = np.round(np.hypot(offsets[:, 0], offsets[:, 1]), 9)
    keep = distances <= radius + 1e-9
    offsets, distances = offsets[keep], distances[keep]
    return tuple(offsets[distances == value] for value in np.unique(distances))


class MarkerPicker:
    """Which node the cursor is on, as the markers are drawn.

    The node whose drawn shape covers the pixel under the cursor and is drawn
    last (on top) wins; click rings, a clicked node's halo, are not nodes.
    Over no node, the topmost node drawn at the nearest pixel within
    HOVER_MARGIN_PX wins. Candidates come from a k-d tree of the markers, so a pick costs the
    markers near the cursor, not every node.
    """

    def __init__(self, marker_node_order, marker_ring_mask, node_positions, sizes, edge_widths,
                 codes, antialias, previous=None):
        keep = ~np.asarray(marker_ring_mask, dtype=bool)
        self.slots = np.flatnonzero(keep)
        self.nodes = np.asarray(marker_node_order)[keep]
        # Every node's position, by node number, as the markers are sent (float32):
        # a copy, never a view of the Viewer's positions, which a drag moves in
        # place. Through a view the next picker would see no node moved and keep
        # a tree of the old places (moved nodes could then not be picked).
        self.node_positions = np.array(np.asarray(node_positions)[:, :2], dtype=np.float32)
        self.positions = self.node_positions[self.nodes]
        self.sizes = np.asarray(sizes, dtype=np.float32)[keep]
        self.edge_widths = np.asarray(edge_widths, dtype=np.float32)[keep]
        self.codes = np.asarray(codes)[keep].astype(np.int64)
        self.antialias = float(antialias)
        self._marker_of_node = np.full(len(self.node_positions), -1, dtype=np.int64)
        self._marker_of_node[self.nodes] = np.arange(len(self.nodes))
        # The k-d tree indexes nodes by position, so a recolour, a selection or
        # a hide keeps the last one; only moved nodes build it again.
        self._tree = None
        if (previous is not None and previous._tree is not None
                and np.array_equal(previous.node_positions, self.node_positions)):
            self._tree = previous._tree

    def _candidates(self, point, radius):
        """The drawn markers (indices into self.nodes) whose node lies within radius of point."""
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.node_positions.astype(np.float64))
        nodes = np.asarray(self._tree.query_ball_point(point, radius), dtype=np.int64)
        markers = self._marker_of_node[nodes] if len(nodes) else nodes
        return markers[markers >= 0]

    def pick(self, transforms, cursor, pixel_scale):
        """The node under cursor (canvas pixels, as mouse events give them),
        or None. transforms is the markers' TransformSystem."""
        if len(self.nodes) == 0:
            return None
        to_framebuffer = transforms.get_transform('visual', 'framebuffer')
        canvas_to_framebuffer = transforms.get_transform('canvas', 'framebuffer')
        scale = float(pixel_scale)
        # The device pixel under the cursor: the one whose top-left corner a
        # whole-pixel cursor position names, as on screen; then its centre in
        # framebuffer pixels.
        device = np.floor(np.asarray(cursor, dtype=np.float64)[:2] * scale)
        pixel = np.asarray(canvas_to_framebuffer.map((device + 0.5) / scale))[:2]
        # How far a marker's pixels reach: half its sprite's diagonal.
        sprite = self.sizes.max() * scale + 4.0 * (self.edge_widths.max() * scale + 1.5 * self.antialias)
        reach = sprite / 2.0 * _SQRT2 + HOVER_MARGIN_PX * scale
        origin, unit_x, unit_y = (np.asarray(row)[:2] for row in to_framebuffer.map(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])))
        per_unit = min(np.hypot(*(unit_x - origin)), np.hypot(*(unit_y - origin)))
        if not np.isfinite(per_unit) or per_unit <= 0:
            return None
        point = np.asarray(to_framebuffer.imap(pixel))[:2]
        found = self._candidates(point, reach / per_unit)
        if len(found) == 0:
            return None
        centres = np.asarray(to_framebuffer.map(self.positions[found]))[:, :2]
        size = self.sizes[found] * scale
        outline = self.edge_widths[found] * scale
        total = size + 4.0 * (outline + 1.5 * self.antialias)
        slots = self.slots[found]
        # The pixel under the cursor first, then the pixels around it within
        # the margin, nearest first: the topmost node drawn at the nearest of
        # them is the one hovered.
        for offsets in _margin_rings(HOVER_MARGIN_PX * scale):
            px = pixel[0] + offsets[:, 0][None, :]
            py = pixel[1] + offsets[:, 1][None, :]
            dx = px - centres[:, 0][:, None]
            dy = centres[:, 1][:, None] - py
            distance = marker_distance(np.repeat(self.codes[found][:, None], len(offsets), axis=1),
                                       dx, dy, np.repeat(size[:, None], len(offsets), axis=1))
            reach_px = (0.5 * outline + self.antialias)[:, None]
            half = (total / 2.0)[:, None]
            covered = ((size > 0)[:, None] & (np.abs(dx) < half) & (np.abs(dy) < half)
                       & (distance <= reach_px))
            hit = covered.any(axis=1)
            if hit.any():
                return int(self.nodes[found[hit][np.argmax(slots[hit])]])
        return None
