# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0

"""Shared main-window presentation helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
import time

from PySide6 import QtCore, QtNetwork


def _single_instance_server_name(application_id):
    """Return a short, stable, per-user local-server name for an application."""
    normalized_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(application_id)).strip("-")
    if not normalized_id:
        raise ValueError("application_id must contain at least one usable character")
    user_identity = os.path.normcase(str(Path.home().resolve()))
    user_digest = hashlib.sha256(user_identity.encode("utf-8")).hexdigest()[:12]
    return f"ssn-{normalized_id}-{user_digest}"


def _notify_local_server(server_name, timeout_ms):
    socket = QtNetwork.QLocalSocket()
    socket.connectToServer(server_name)
    if not socket.waitForConnected(max(0, int(timeout_ms))):
        socket.abort()
        return False
    socket.write(b"activate\n")
    socket.flush()
    socket.waitForBytesWritten(min(max(0, int(timeout_ms)), 500))
    socket.disconnectFromServer()
    return True


def notify_existing_instance(application_id, timeout_ms=750):
    """Activate an existing application without claiming a new instance."""
    return _notify_local_server(
        _single_instance_server_name(application_id), timeout_ms
    )


class SingleInstanceController(QtCore.QObject):
    """Own one application instance and route duplicate launches to its window."""

    def __init__(self, application_id, parent=None):
        super().__init__(parent)
        self.server_name = _single_instance_server_name(application_id)
        lock_path = Path(QtCore.QDir.tempPath()) / f"{self.server_name}.lock"
        self._lock = QtCore.QLockFile(str(lock_path))
        self._server = QtNetwork.QLocalServer(self)
        self._server.setSocketOptions(QtNetwork.QLocalServer.UserAccessOption)
        self._server.newConnection.connect(self._accept_connections)
        self._activation_callback = None
        self._activation_pending = False
        self._owns_server = False

    @property
    def owns_server(self):
        return self._owns_server

    def acquire_or_notify(self, timeout_ms=30000):
        """Claim the instance name, or ask the owner to activate and return False."""
        if not self._lock.tryLock(0):
            deadline = time.monotonic() + (max(0, timeout_ms) / 1000.0)
            while True:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if self._notify_existing_instance(min(remaining_ms, 500)):
                    return False
                if remaining_ms <= 0:
                    break
                QtCore.QThread.msleep(min(100, remaining_ms))
            raise RuntimeError(
                "Another instance is running, but its window could not be activated."
            )

        # The process lock makes stale local-server cleanup safe: no live peer
        # can own the same application ID while this process holds the lock.
        if self._server.listen(self.server_name):
            self._owns_server = True
            return True
        QtNetwork.QLocalServer.removeServer(self.server_name)
        if self._server.listen(self.server_name):
            self._owns_server = True
            return True
        self._lock.unlock()
        raise RuntimeError(
            f"Could not establish the single-instance channel "
            f"'{self.server_name}': {self._server.errorString()}"
        )

    def _notify_existing_instance(self, timeout_ms):
        return _notify_local_server(self.server_name, timeout_ms)

    def set_activation_callback(self, callback):
        """Register the callback used to restore and focus the primary window."""
        if callback is not None and not callable(callback):
            raise TypeError("activation callback must be callable or None")
        self._activation_callback = callback
        if callback is not None and self._activation_pending:
            self._activation_pending = False
            QtCore.QTimer.singleShot(0, callback)

    def _accept_connections(self):
        accepted = False
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                break
            accepted = True
            socket.disconnectFromServer()
            socket.deleteLater()
        if not accepted:
            return
        if self._activation_callback is None:
            self._activation_pending = True
        else:
            QtCore.QTimer.singleShot(0, self._activation_callback)

    def close(self):
        """Release the owned local server during normal application shutdown."""
        if self._owns_server:
            self._server.close()
            self._owns_server = False
            self._lock.unlock()


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
