# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The screen's DPI for the Viewer's canvas, without VisPy's unguarded divisions.

VisPy's macOS and Windows versions of get_dpi divide the screen's pixels by its
size in millimetres. A display that reports 0 mm raised ZeroDivisionError inside
SceneCanvas and the Viewer died. The canvas is therefore given its DPI here:
the number VisPy would have used when the measurement works, 96.0 when it does
not (with one console line). A DPI the user set through VisPy's own
configuration is left alone (None is returned, as VisPy then uses that).
"""

import math

FALLBACK_DPI = 96.0


def canvas_dpi(get_dpi=None, configured=None):
    """The dpi= argument for SceneCanvas.

    get_dpi and configured are parameters for the tests; by default VisPy's own
    get_dpi and vispy.config['dpi'] are used.
    """
    if configured is None:
        try:
            from vispy.util.config import config as vispy_config
            configured = vispy_config['dpi']
        except Exception:
            configured = None
    if configured is not None:
        return None
    if get_dpi is None:
        try:
            from vispy.util.dpi import get_dpi
        except Exception:
            get_dpi = None
    value = None
    if get_dpi is not None:
        try:
            value = get_dpi(raise_error=False)
        except Exception:
            value = None
    try:
        usable = (value is not None and not isinstance(value, bool)
                  and math.isfinite(float(value)) and float(value) > 0)
    except (TypeError, ValueError):
        usable = False
    if usable:
        return value
    print(f"The screen reported no usable size; using {FALLBACK_DPI:.0f} dpi for the display.")
    return FALLBACK_DPI
