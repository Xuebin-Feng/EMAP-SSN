# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Shared main-window presentation helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from PySide6 import QtCore


def _activate_window(window):
    """Request foreground focus without leaving the window always-on-top."""
    if window is None or not window.isVisible():
        return

    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()

    handle = window.windowHandle()
    if handle is not None:
        handle.requestActivate()

    if sys.platform == "win32":
        _activate_windows_native(window)


def _activate_windows_native(window):
    """Put a Windows main window in front once, then restore normal Z-order."""
    try:
        import ctypes

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32
        swp_no_move_or_size = 0x0001 | 0x0002
        hwnd_topmost = -1
        hwnd_not_topmost = -2
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetWindowPos(
            hwnd, hwnd_topmost, 0, 0, 0, 0, swp_no_move_or_size
        )
        user32.SetWindowPos(
            hwnd, hwnd_not_topmost, 0, 0, 0, 0, swp_no_move_or_size
        )
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    except (AttributeError, OSError, TypeError, ValueError):
        # Qt activation above remains the portable fallback.
        pass


def _signal_launcher_ready(window):
    """Atomically tell a desktop launcher that its Qt window is ready."""
    if window is None or not window.isVisible():
        return False

    ready_value = os.environ.pop("SSN_GUI_READY_FILE", None)
    if not ready_value:
        return False

    ready_path = Path(ready_value)
    temporary_path = ready_path.with_name(
        f".{ready_path.name}.{os.getpid()}.tmp"
    )
    try:
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text("ready\n", encoding="utf-8")
        os.replace(temporary_path, ready_path)
    except OSError as error:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        print(f"Warning: Could not signal GUI readiness: {error}", file=sys.stderr)
        return False
    return True


def _activate_and_signal(window):
    """Perform the final foreground request, then acknowledge GUI readiness."""
    _activate_window(window)
    _signal_launcher_ready(window)


def show_window_in_front(window):
    """Show a main window and make best-effort foreground requests at startup."""
    window.show()
    QtCore.QTimer.singleShot(
        0, lambda active_window=window: _activate_window(active_window)
    )
    QtCore.QTimer.singleShot(
        100, lambda active_window=window: _activate_and_signal(active_window)
    )
