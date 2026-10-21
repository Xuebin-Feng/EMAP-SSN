# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering filters and read-only canvas capture."""
import base64
from datetime import datetime, timezone
import io
import os
import sys
import uuid
import numpy as np

from desktop.Viewer_State import knn_topology_layout
from utilities.Localization import JoinedMessage, Message


def edges_follow_selection(viewer, configuration):
    """Whether edge_stages filters the edges it keeps by the selection.

    A UMAP or PaCMAP layout (knn_topology_layout) keeps only the selected
    nodes' edges. Otherwise the threshold and hidden nodes alone decide, and
    the Viewer draws a prefix of its score-sorted edges; a node drag draws
    them all, following the nodes.
    """
    return knn_topology_layout(configuration)


def edge_stages(viewer, configuration, cache=False):
    edges = np.asarray(getattr(viewer, 'edges', ()), dtype=np.int32).reshape(-1, 2)
    threshold = getattr(viewer, 'current_slider_threshold', getattr(configuration, 'SIMILARITY_THRESHOLD', 0.0))
    scores = getattr(viewer, 'edge_scores', ())
    visible = np.asarray(getattr(viewer, "visible_mask", [True] * getattr(viewer, "n_nodes", len(getattr(viewer, "full_headers", ())))), dtype=bool)
    key = (id(getattr(viewer, 'edges', None)), id(getattr(viewer, 'edge_scores', None)), len(edges), threshold, visible.tobytes()) if cache else None
    cached = getattr(viewer, '_edge_filter_cache', None) if cache else None
    if cached is not None and cached[0] == key:
        active, qualified_count = cached[1], cached[2]
    else:
        qualified = np.asarray(scores) >= threshold if len(scores) else np.ones(len(edges), dtype=bool)
        endpoint = visible[edges[:, 0]] & visible[edges[:, 1]]
        active = edges[qualified & endpoint]
        qualified_count = int(qualified.sum())
        if cache:
            viewer._edge_filter_cache = (key, active, qualified_count)
    visible_count = len(active)
    selection = getattr(viewer, 'selected_indices', ())
    if knn_topology_layout(configuration):
        active = active[np.isin(active[:, 0], selection) | np.isin(active[:, 1], selection)]
    return active, {'loaded_edge_count': len(edges), 'threshold_qualified_edge_count': qualified_count,
                    'visible_endpoint_edge_count': visible_count, 'rendered_edge_count': len(active)}


def visual_overview(viewer, configuration):
    counts = edge_stages(viewer, configuration)[1]
    camera = getattr(getattr(viewer, 'view', None), 'camera', None)
    rect = getattr(camera, 'rect', None)
    counts['camera_rectangle'] = list(map(float, (rect.left, rect.bottom, rect.width, rect.height))) if rect is not None else None
    counts['canvas_dimensions'] = list(getattr(getattr(viewer, 'canvas', None), 'size', ()))
    counts['rendered_edge_count'] = counts['rendered_edge_count'] if getattr(viewer, 'edge_layer', None) is not None else 0
    return counts


def opengl_failure_note(viewer, error):
    """Why a headless Viewer could not render, a Message, when the cause is its missing OpenGL context; else None.

    Qt's offscreen platform has no OpenGL context on Windows and macOS, and on
    Linux only through an X server, so a headless Viewer there cannot render
    (capture_view, print). A hidden-mode Viewer has one, unless it found no
    display and fell back to headless. With a known cause (the fallback's reason,
    or on Linux no display at all) any failure to render is that cause, whatever
    VisPy's error says; otherwise only an error naming the missing OpenGL context
    is. Other failures keep their own text.
    """
    if not (os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or getattr(viewer, 'headless', False)):
        return None
    reason = getattr(viewer, 'window_fallback_reason', None)
    if reason is None and sys.platform.startswith('linux'):
        from utilities.Viewer_Display import display_unavailable_reason
        reason = display_unavailable_reason()
    if reason:
        return Message('This Viewer has no OpenGL context to render with, because {reason}.', reason=reason)
    if 'OpenGL context' not in str(error):
        return None
    return Message(
        'This headless Viewer has no OpenGL context to render with;'
        ' start a Viewer in hidden or normal mode to render the view.'
    )


def capture_view(viewer, request_id=None):
    """The canvas as a PNG, for MCP clients and the Agent page.

    Its errors hold Messages: MCP clients get str(error), in English, and
    the Agent page shows them translated.
    """
    from PIL import Image
    canvas = getattr(viewer, 'canvas', None)
    if canvas is None:
        raise ValueError(Message('Viewer canvas is unavailable'))
    if request_id is not None:
        from Viewer_Command_Portal import get_portal, TERMINAL
        request = get_portal(viewer).get(request_id)
        if request['status'] not in TERMINAL:
            raise ValueError(Message('Associated command request has not finished'))
    try:
        pixels = canvas.render()
        result = Image.fromarray(pixels)
        original = list(result.size)
        result.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        result.save(out, format='PNG')
    except Exception as error:
        message = Message('Viewer canvas capture failed: {error}', error=error)
        note = opengl_failure_note(viewer, error)
        if note is not None:
            message = JoinedMessage([message, note])
        raise ValueError(message) from error
    return {'capture_id': uuid.uuid4().hex, 'captured_at': datetime.now(timezone.utc).isoformat(),
            'session_id': getattr(viewer, 'inspection_session_id', None), 'request_id': request_id,
            'observation': 'Current canvas; manual changes may postdate command completion.',
            'mime_type': 'image/png', 'width': result.width, 'height': result.height,
            'original_dimensions': original, 'image_base64': base64.b64encode(out.getvalue()).decode('ascii')}
