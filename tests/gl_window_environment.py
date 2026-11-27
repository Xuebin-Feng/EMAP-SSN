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
import subprocess
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


_NATIVE_SCALE = []


def native_display_scale():
    """The factor a real window's canvas has at QT_SCALE_FACTOR=1 with QT_ENABLE_HIGHDPI_SCALING=0: 1.0 everywhere but macOS.

    A Retina Mac gives its windows a device pixel ratio of 2 whatever those variables say, so a canvas
    the test asks to run at display scale 2 reports a pixel_scale of 4 there. The tests multiply the scale
    they ask for by this, read from Qt on the real screen; on Windows and Linux it is 1.0 without asking.
    """
    if not _NATIVE_SCALE:
        value = 1.0
        if sys.platform == "darwin":
            environment = dict(os.environ)
            environment.pop("QT_QPA_PLATFORM", None)
            environment.update(QT_ENABLE_HIGHDPI_SCALING="0", QT_SCALE_FACTOR="1")
            code = ("from PySide6.QtGui import QGuiApplication\n"
                    "app = QGuiApplication([])\n"
                    "print('NATIVE', app.primaryScreen().devicePixelRatio())\n")
            try:
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", env=environment, timeout=60)
                found = [line.split()[1] for line in result.stdout.splitlines() if line.startswith("NATIVE ")]
                value = float(found[0]) if found else 1.0
            except (subprocess.TimeoutExpired, OSError, ValueError):
                value = 1.0
        _NATIVE_SCALE.append(value)
    return _NATIVE_SCALE[0]


def run_gl_script(test, script, *, args=(), scale=None, unset=(), timeout=300):
    """Run ``script`` in a child process on a real GL window and return its output.

    Builds the child environment via :func:`gl_window_environment` (so Linux uses
    ``xcb`` and skips cleanly when no X display exists), removes any environment
    variables in ``unset``, and sets ``QT_SCALE_FACTOR`` when ``scale`` is given.
    Skips ``test`` if the probe times out or gets no OpenGL context (no ``CONTEXT OK``
    marker in stdout), and asserts a zero exit status otherwise.
    """
    environment = gl_window_environment()
    for name in unset:
        environment.pop(name, None)
    if scale is not None:
        environment["QT_SCALE_FACTOR"] = str(scale)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=environment, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        test.skipTest("the GL render probe timed out")
    if "CONTEXT OK" not in result.stdout:
        test.skipTest(f"no OpenGL context to render with (exit {result.returncode})")
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr[-2000:])
    return result.stdout + result.stderr[-2000:]
