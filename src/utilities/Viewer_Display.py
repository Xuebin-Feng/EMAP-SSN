# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Which window mode a Viewer launch gets, and why a hidden one may run offscreen.

A Viewer runs in one of three window modes, named as in the MCP ``mode``:

* ``normal``: a normal Viewer, its window on screen from the start.
* ``hidden``: Qt's native platform, with the main window kept off screen
  (``Qt.WA_DontShowOnScreen``). The window has a real OpenGL context, so
  ``capture_view`` and ``print`` work, and it can be shown later.
* ``headless``: Qt's offscreen platform. It has no native window, and on Windows,
  macOS and a Linux without a reachable X server no OpenGL context either.

``SSN_VIEWER_WINDOW=hidden`` asks for the second. This module decides whether
that is possible on this host, before Qt or any OpenGL library is imported, so it
only uses the standard library.
"""

from __future__ import annotations

import os
import subprocess
import sys

WINDOW_ENV = "SSN_VIEWER_WINDOW"
FALLBACK_ENV = "SSN_VIEWER_WINDOW_FALLBACK"
HEADLESS_ENV = "SSN_VIEWER_HEADLESS"
MAC_NO_DOCK_ENV = "QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM"

# A hidden Viewer whose native platform cannot make an OpenGL context prints
# this line and exits with this code; the MCP client then launches it offscreen.
HIDDEN_UNAVAILABLE_MARKER = "Viewer: hidden window unavailable"
HIDDEN_UNAVAILABLE_EXIT_CODE = 75


def _text(value):
    return str(value or "").strip().lower()


def _macos_session_name():
    """The launchd session this process runs in ("Aqua" for a graphical login), or None."""
    try:
        done = subprocess.run(["/bin/launchctl", "managername"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    name = done.stdout.strip()
    return name if done.returncode == 0 and name else None


def display_unavailable_reason(environ=None, platform_name=None, macos_session=None):
    """Why Qt's native platform cannot open here, or None when it can.

    Linux needs an X server (or XWayland: the Viewer runs VisPy through xcb in a
    Wayland session) or a Wayland compositor; without either Qt aborts at start.
    macOS needs a graphical login session (an SSH login has none). Windows has no
    such case: a process with no desktop session fails later, and the hidden
    Viewer reports that itself.
    """
    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    if platform_name.startswith("linux"):
        has_x11 = bool(environ.get("DISPLAY"))
        has_wayland = bool(environ.get("WAYLAND_DISPLAY"))
        if not has_x11 and not has_wayland:
            return "no display was found (DISPLAY and WAYLAND_DISPLAY are not set)"
        selected = _text(environ.get("QT_QPA_PLATFORM"))
        # EMAPSSN_Viewer._configure_linux_vispy_platform moves Wayland to xcb.
        uses_xcb = selected.startswith("wayland") or (
            not selected and _text(environ.get("XDG_SESSION_TYPE")) == "wayland")
        if uses_xcb and not has_x11:
            return "the Wayland session has no XWayland display (DISPLAY is not set), which the Viewer's OpenGL needs"
    elif platform_name == "darwin":
        session = (macos_session or _macos_session_name)()
        if session is not None and session != "Aqua":
            return f"this process is not in a graphical login session (launchd session {session!r})"
    return None


def resolve_window_mode(requested, environ=None, platform_name=None):
    """The window mode a launch of kind ``requested`` gets, as (mode, fallback_reason).

    ``requested`` is "normal", "headless" or "hidden"; mode is one of them.
    fallback_reason is set only when a hidden launch has to run headless.
    """
    if requested == "hidden":
        reason = display_unavailable_reason(environ, platform_name)
        return ("headless", reason) if reason else ("hidden", None)
    return requested, None


def apply_hidden_environment(environ=None, platform_name=None):
    """Prepare the environment of a Viewer started with SSN_VIEWER_WINDOW=hidden.

    Call before Qt is imported. Returns the window mode: "hidden", or
    "headless" (the offscreen platform) with FALLBACK_ENV set to the reason.
    Returns None and changes nothing unless the Viewer was asked to be hidden
    and is not already headless.
    """
    environ = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    if environ.get(WINDOW_ENV) != "hidden" or _text(environ.get(HEADLESS_ENV)) in {"1", "true", "yes"}:
        return None
    mode, reason = resolve_window_mode("hidden", environ, platform_name)
    if mode == "headless":
        environ["QT_QPA_PLATFORM"] = "offscreen"
        environ[FALLBACK_ENV] = reason
        return mode
    # An inherited offscreen platform (a test run, a CI shell) is not what a
    # hidden Viewer wants; normal mode drops it the same way.
    if _text(environ.get("QT_QPA_PLATFORM")) == "offscreen":
        environ.pop("QT_QPA_PLATFORM")
    if platform_name == "darwin":
        # Qt turns every process it starts into a Dock-icon application; a hidden
        # Viewer stays a background application until its window is shown
        # (desktop.Viewer_Window switches the activation policy then).
        environ.setdefault(MAC_NO_DOCK_ENV, "1")
    return mode
