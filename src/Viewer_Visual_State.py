# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Shared rendering filters and read-only canvas capture."""
import base64
from datetime import datetime, timezone
import io
import uuid
import numpy as np


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
    if getattr(configuration, 'LOW_RESOURCE_MODE', False) and getattr(viewer, 'is_multi_dragging', False) and len(selection):
        active = active[~(np.isin(active[:, 0], selection) | np.isin(active[:, 1], selection))]
    if getattr(configuration, 'UMAP_MODE', False):
        active = active[np.isin(active[:, 0], selection) | np.isin(active[:, 1], selection)]
    return active, {'loaded_edge_count': len(edges), 'threshold_qualified_edge_count': qualified_count,
                    'visible_endpoint_edge_count': visible_count, 'rendered_edge_count': len(active)}


def visual_overview(viewer, configuration):
    counts = edge_stages(viewer, configuration)[1]
    camera = getattr(getattr(viewer, 'view', None), 'camera', None)
    rect = getattr(camera, 'rect', None)
    counts['camera_rectangle'] = list(map(float, (rect.left, rect.bottom, rect.width, rect.height))) if rect is not None else None
    counts['canvas_dimensions'] = list(getattr(getattr(viewer, 'canvas', None), 'size', ()))
    counts['rendered_edge_count'] = counts['rendered_edge_count'] if getattr(viewer, 'line_visual', None) is not None else 0
    return counts


def capture_view(viewer, request_id=None):
    from PIL import Image
    canvas = getattr(viewer, 'canvas', None)
    if canvas is None:
        raise ValueError('Viewer canvas is unavailable')
    if request_id is not None:
        from Viewer_Command_Portal import get_portal, TERMINAL
        request = get_portal(viewer).get(request_id)
        if request['status'] not in TERMINAL:
            raise ValueError('Associated command request has not finished')
    try:
        pixels = canvas.render()
        result = Image.fromarray(pixels)
        original = list(result.size)
        result.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        result.save(out, format='PNG')
    except Exception as error:
        raise ValueError(f'Viewer canvas capture failed: {error}') from error
    return {'capture_id': uuid.uuid4().hex, 'captured_at': datetime.now(timezone.utc).isoformat(),
            'session_id': getattr(viewer, 'inspection_session_id', None), 'request_id': request_id,
            'observation': 'Current canvas; manual changes may postdate command completion.',
            'mime_type': 'image/png', 'width': result.width, 'height': result.height,
            'original_dimensions': original, 'image_base64': base64.b64encode(out.getvalue()).decode('ascii')}
