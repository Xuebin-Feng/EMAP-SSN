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
"""

from __future__ import annotations

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
