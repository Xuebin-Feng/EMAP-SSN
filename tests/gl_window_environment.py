# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The environment of the child process in a GL test that counts scene draws on a real VisPy window.

The test modules set QT_QPA_PLATFORM to "offscreen" for themselves, so the child
needs it taken away to get a real window. Which window it gets then is decided
here, as the Viewer decides it: on Linux the Viewer runs VisPy through xcb (X, or
XWayland under Wayland), because Qt's Wayland plugin makes an OpenGL ES context that
VisPy's desktop shaders do not suit (EMAPSSN_Viewer._configure_linux_vispy_platform).
A child left to Qt picks native Wayland on a Wayland session, where the Viewer never
runs and a canvas cannot even render. Windows and macOS keep Qt's own choice.

Not collected by unittest; import with ``from tests.gl_window_environment import ...``.
"""
import os
import sys
import unittest

X_SOCKET_DIR = "/tmp/.X11-unix"    # an X server listens at X<n> here for display :<n>


def gl_window_environment():
    """A copy of os.environ for a child that opens a real window; raises SkipTest when Linux has no X display.

    The test runner may have removed DISPLAY (to keep a run off the desktop, which Qt's Wayland plugin
    reaches through the default wayland-0 socket all the same). X has a default of its own, display :0,
    so a missing DISPLAY falls back to it when its socket exists: that is where XWayland listens.
    """
    environment = dict(os.environ)
    environment.pop("QT_QPA_PLATFORM", None)
    environment.update(PYTHONDONTWRITEBYTECODE="1", QT_ENABLE_HIGHDPI_SCALING="0")
    if sys.platform.startswith("linux"):
        if not environment.get("DISPLAY"):
            if not os.path.exists(os.path.join(X_SOCKET_DIR, "X0")):
                raise unittest.SkipTest(
                    "no X display to open the test window on (DISPLAY is not set, and nothing listens on :0): "
                    "the Viewer runs VisPy through xcb, not on Qt's Wayland plugin")
            environment["DISPLAY"] = ":0"
        environment["QT_QPA_PLATFORM"] = "xcb"
    return environment
