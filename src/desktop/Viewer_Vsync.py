# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Vertical sync for the Viewer's canvas.

The canvas is a QOpenGLWidget (VisPy's Qt backend). It draws into a
framebuffer object of its own; Qt then composes that into the top-level
window's surface and swaps that surface's buffers. So the swap interval that
paces the screen is the top-level window's (its QSurfaceFormat), not the
canvas widget's: VisPy's `vsync` argument sets the widget's own format, and
the widget's context never swaps a surface of its own. (Measured with a
QMainWindow around a VisPy canvas: widget format 0 beside window format 1.)
The window takes QSurfaceFormat.defaultFormat()'s interval, 1 unless the
application changes it; the Viewer leaves that default alone, as QtWebEngine
and the other windows share it.

The Viewer therefore does both, for itself alone: the canvas is made with
vsync (vsync_enabled), and its window is given the same swap interval
(apply_window_swap_interval): nothing when the window already has it, which
is the usual case, and otherwise (SSN_VIEWER_VSYNC=0) the window's platform
surface is made again with the interval wanted, before the window is shown.

SSN_VIEWER_VSYNC=0 (or off, false, no) turns vertical sync off, to compare
panning with it on. A driver's own override (such as the NVIDIA control
panel's Vertical sync setting) wins over either.

Neither touches canvas.render(): print, MCP captures and the edge image
cache draw into framebuffer objects and read them back, and never swap.
"""

from __future__ import annotations

import os

VSYNC_ENVIRONMENT_VARIABLE = "SSN_VIEWER_VSYNC"
_OFF_VALUES = {"0", "off", "false", "no"}


def vsync_enabled(environ=None):
    """Whether the Viewer syncs to the screen's refresh: yes unless SSN_VIEWER_VSYNC is 0/off/false/no."""
    environ = os.environ if environ is None else environ
    return str(environ.get(VSYNC_ENVIRONMENT_VARIABLE, "")).strip().lower() not in _OFF_VALUES


def apply_window_swap_interval(window, enabled):
    """Give the top-level QWidget window the swap interval 1 (enabled) or 0, for this window alone.

    The window's platform surface is made again only if its interval differs, and only while it is not
    shown. Returns True if the window has the interval wanted, False if it could not be given it
    (the window keeps what it had)."""
    wanted = 1 if enabled else 0
    try:
        from PySide6.QtGui import QSurfaceFormat
        if (window.windowHandle() is None
                and QSurfaceFormat.defaultFormat().swapInterval() == wanted):
            return True                     # the window will take the default's interval when it is made
        window.winId()                      # creates the platform window, not shown
        handle = window.windowHandle()
        if handle is None:
            return False
        if handle.format().swapInterval() == wanted:
            return True
        if window.isVisible():
            return False
        fmt = QSurfaceFormat(handle.requestedFormat())
        fmt.setSwapInterval(wanted)
        handle.destroy()
        handle.setFormat(fmt)
        handle.create()
        return handle.format().swapInterval() == wanted
    except Exception as error:
        print(f"Warning: Could not set the window's vertical sync: {error}")
        return False
