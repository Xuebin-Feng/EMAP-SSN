# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The Viewer's main window in its three modes, and showing a hidden one.

``normal``: the window is on screen from the start (desktop.Desktop_App
show_window_in_front). ``headless``: Qt's offscreen platform; there is no native
window to show. ``hidden``: Qt's native platform, with the window "shown" under
``Qt.WA_DontShowOnScreen``, which keeps it off screen, out of the taskbar and the
Dock, and gives its VisPy canvas a real OpenGL context, so ``capture_view`` and
``print`` work. ``ViewerWindow.show()`` later moves it on screen (hide(), clear
the attribute, show(): no native window is rebuilt and the OpenGL context is
kept) and ``hide()`` moves it off again.

A shown hidden window does not take the keyboard: it is shown without
activating, pinned above other windows on Windows until the user first
activates it, and its taskbar button flashes. Whether the platform let it become
the foreground window is part of the state, because Windows and Wayland
compositors may refuse.

``shows_windows()`` is the one question the rest of the Viewer asks before it
opens a dialog, a message box, a browser page or a terminal: true for a normal
Viewer and for a hidden one while its window is on screen.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

from PySide6 import QtCore, QtWidgets

NORMAL = "normal"
HIDDEN = "hidden"
HEADLESS = "headless"
MODES = (NORMAL, HIDDEN, HEADLESS)


class WindowUnavailable(RuntimeError):
    """The window cannot be shown or hidden in this Viewer's mode or on this host."""


class WindowsNative:
    """The few Win32 calls the window needs, so tests can stand in for them."""

    _SWP_NOSIZE = 0x0001
    _SWP_NOMOVE = 0x0002
    _SWP_NOACTIVATE = 0x0010
    _HWND_TOPMOST = -1
    _HWND_NOTOPMOST = -2

    def __init__(self):
        self._user32 = None

    def _api(self):
        if self._user32 is None:
            from ctypes import wintypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, wintypes.UINT]
            user32.SetWindowPos.restype = wintypes.BOOL
            user32.GetForegroundWindow.restype = wintypes.HWND
            self._user32 = user32
        return self._user32

    def _place(self, hwnd, after):
        from ctypes import wintypes
        flags = self._SWP_NOMOVE | self._SWP_NOSIZE | self._SWP_NOACTIVATE
        return bool(self._api().SetWindowPos(wintypes.HWND(hwnd), wintypes.HWND(after), 0, 0, 0, 0, flags))

    def topmost(self, hwnd):
        """Put the window above all non-topmost windows without activating it."""
        return self._place(hwnd, self._HWND_TOPMOST)

    def not_topmost(self, hwnd):
        return self._place(hwnd, self._HWND_NOTOPMOST)

    def foreground(self):
        """The handle of the foreground window, or 0."""
        return int(ctypes.cast(self._api().GetForegroundWindow(), ctypes.c_void_p).value or 0)


class _ObjcRuntime:
    """Messages to Objective-C objects through libobjc (no PyObjC)."""

    def __init__(self):
        path = ctypes.util.find_library("objc")
        if not path:
            raise OSError("libobjc was not found")
        self._lib = ctypes.cdll.LoadLibrary(path)
        self._lib.objc_getClass.restype = ctypes.c_void_p
        self._lib.objc_getClass.argtypes = [ctypes.c_char_p]
        self._lib.sel_registerName.restype = ctypes.c_void_p
        self._lib.sel_registerName.argtypes = [ctypes.c_char_p]

    def cls(self, name):
        return self._lib.objc_getClass(name.encode("ascii"))

    def send(self, receiver, selector, *args, restype=ctypes.c_void_p, argtypes=()):
        prototype = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
        message = prototype(("objc_msgSend", self._lib))
        return message(receiver, self._lib.sel_registerName(selector.encode("ascii")), *args)


class MacActivation:
    """A hidden Viewer's Dock icon: absent while hidden, present while shown.

    Qt turns every process into a Dock-icon application unless
    QT_MAC_DISABLE_FOREGROUND_APPLICATION_TRANSFORM is set (the hidden launch
    sets it, utilities.Viewer_Display). The process then starts as a plain
    background application, and this class makes it an accessory application
    (no Dock icon, may own windows) while hidden and a regular one (Dock icon)
    while shown. Every call is guarded: a failure prints a warning and leaves the
    window as it is, never raising. Not tested on a real Mac.
    """

    REGULAR = 0       # NSApplicationActivationPolicyRegular
    ACCESSORY = 1     # NSApplicationActivationPolicyAccessory

    def __init__(self, runtime=None, report=print):
        self._runtime = runtime
        self._report = report

    def _objc(self):
        if self._runtime is None:
            self._runtime = _ObjcRuntime()
        return self._runtime

    def _policy(self, policy, name):
        try:
            runtime = self._objc()
            application = runtime.send(runtime.cls("NSApplication"), "sharedApplication")
            done = runtime.send(application, "setActivationPolicy:", policy,
                                restype=ctypes.c_bool, argtypes=(ctypes.c_long,))
            if not done:
                self._report(f"Warning: macOS refused the {name} activation policy; the Dock icon is unchanged.")
            return bool(done)
        except Exception as error:
            self._report(f"Warning: could not set the {name} activation policy ({error}); the Dock icon is unchanged.")
            return False

    def accessory(self):
        """Hide the Dock icon (a hidden Viewer)."""
        return self._policy(self.ACCESSORY, "accessory")

    def regular(self):
        """Show the Dock icon (a shown Viewer)."""
        return self._policy(self.REGULAR, "regular")

    def order_front(self, view):
        """Order the window of the NSView ``view`` in front without making it key."""
        try:
            runtime = self._objc()
            window = runtime.send(view, "window")
            if window:
                runtime.send(window, "orderFrontRegardless")
                return True
        except Exception as error:
            self._report(f"Warning: could not order the Viewer window to the front ({error}).")
        return False


class _ActivationWatcher(QtCore.QObject):
    """Calls ``callback`` once, when the user takes the window.

    That is when the platform makes the window the foreground window
    (``is_foreground``), or when a key or mouse button is pressed in it. Qt also
    reports an activation when Windows activates the window inside its own
    thread but leaves another program in the foreground; that is not the user.
    """

    _USER = (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.KeyPress)

    def __init__(self, window, callback, is_foreground):
        super().__init__(window)
        self._window = window
        self._callback = callback
        self._is_foreground = is_foreground
        QtWidgets.QApplication.instance().installEventFilter(self)

    def cancel(self):
        window, self._window = self._window, None
        if window is not None:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.removeEventFilter(self)
        self.deleteLater()

    def _in_window(self, watched):
        window = self._window
        return watched is window or (isinstance(watched, QtWidgets.QWidget) and watched.window() is window)

    def eventFilter(self, watched, event):
        if self._window is None:
            return False
        kind = event.type()
        taken = False
        if kind == QtCore.QEvent.Type.WindowActivate and watched is self._window:
            taken = self._is_foreground()
        elif kind in self._USER and self._in_window(watched):
            taken = True
        if taken:
            callback = self._callback
            self.cancel()
            callback()
        return False


class ViewerWindow:
    """The main window of a Viewer, its mode and its on-screen state."""

    def __init__(self, window, mode, *, requested=None, fallback_reason=None, platform=None,
                 native=None, mac=None, alert=None):
        if mode not in MODES:
            raise ValueError(f"mode: expected one of {', '.join(MODES)}.")
        self.window = window
        self.mode = mode
        self.requested = requested or mode
        self.fallback_reason = fallback_reason
        self.platform = sys.platform if platform is None else platform
        self._native = native if native is not None else (WindowsNative() if self.platform == "win32" else None)
        self._mac = mac if mac is not None else (MacActivation() if self.platform == "darwin" else None)
        self._alert = alert or QtWidgets.QApplication.alert
        self._shown = False          # a hidden Viewer's window is on screen
        self._pinned = False         # topmost until first activated (Windows)
        self._watcher = None

    # ---- the question the rest of the Viewer asks

    def shows_windows(self):
        """Whether the user is at this Viewer: dialogs, pages and terminals may open."""
        return self.mode == NORMAL or (self.mode == HIDDEN and self._shown)

    # ---- startup

    def open(self):
        """Open the window the way this mode does, at the point a Viewer is ready to be seen."""
        if self.mode == NORMAL:
            from desktop.Desktop_App import show_window_in_front
            show_window_in_front(self.window)
        elif self.mode == HIDDEN:
            if self._mac is not None:
                self._mac.accessory()
            # "Shown" for Qt, so the canvas gets its OpenGL context and resizes, but never mapped.
            self.window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            self.window.show()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.processEvents()

    # ---- show and hide

    def show(self):
        """Put the window on screen, on top but without the keyboard; return the state."""
        if self.mode == HEADLESS:
            raise WindowUnavailable(self._headless_reason("show"))
        window = self.window
        if self.mode == HIDDEN and not self._shown:
            if self._mac is not None:
                self._mac.regular()
            # A show() on a window that is already "shown" under the attribute does nothing.
            window.hide()
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)
        window.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        try:
            if window.isMinimized():
                window.showNormal()
            else:
                window.show()
        finally:
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self._shown = True
        self._present()
        return self.state()

    def hide(self):
        """Take a hidden-mode window off screen again; return the state."""
        if self.mode != HIDDEN:
            if self.mode == HEADLESS:
                raise WindowUnavailable(self._headless_reason("hide"))
            raise WindowUnavailable(
                "Only a Viewer started with mode='hidden' can be hidden; this one is a normal Viewer, "
                "whose window the user closes (which ends the Viewer).")
        if self._shown:
            self._unpin()
            window = self.window
            window.hide()
            window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.show()
            self._shown = False
            if self._mac is not None:
                self._mac.accessory()
        return self.state()

    def _present(self):
        """Raise a window that was just shown, without taking the keyboard."""
        window = self.window
        if self._native is not None and not self._foreground():
            # Windows may refuse the foreground to a program the user is not working in.
            # Above the other windows until the user takes it is the next best thing.
            if self._native.topmost(int(window.winId())):
                self._pinned = True
                if self._watcher is None:
                    self._watcher = _ActivationWatcher(window, self._unpin, self._foreground)
        if self._mac is not None:
            self._mac.order_front(int(window.winId()))
        try:
            self._alert(window, 0)   # flashes the taskbar button, bounces the Dock icon
        except Exception as error:
            print(f"Warning: could not flash the Viewer window: {error}")

    def _unpin(self):
        """Stop keeping the window above others: the user has taken it."""
        watcher, self._watcher = self._watcher, None
        if watcher is not None:
            watcher.cancel()
        if self._pinned and self._native is not None:
            self._pinned = False
            try:
                self._native.not_topmost(int(self.window.winId()))
            except Exception as error:
                print(f"Warning: could not release the Viewer window's top position: {error}")

    def _headless_reason(self, action):
        text = "This Viewer runs headless (Qt's offscreen platform), which has no window to " + action + "."
        if self.fallback_reason:
            text += f" It was asked to be hidden but fell back to headless because {self.fallback_reason}."
        else:
            text += " Start it with mode='hidden' to be able to show it."
        return text

    # ---- state

    def _foreground(self):
        window = self.window
        if self._native is not None:
            return self._native.foreground() == int(window.winId())
        return bool(window.isActiveWindow())

    def state(self):
        """What a client may know of the window: mode, whether it is on screen and in front."""
        shown = self.shows_windows()
        window = self.window
        app = QtWidgets.QApplication.instance()
        return {
            "mode": self.mode,
            "requested_mode": self.requested,
            "fallback_reason": self.fallback_reason,
            "platform": app.platformName() if app is not None else None,
            "visible": bool(shown and window.isVisible()),
            "minimized": bool(shown and window.isMinimized()),
            "foreground": bool(shown and self._foreground()),
            "on_top_until_activated": self._pinned,
            "can_show": self.mode != HEADLESS,
            "can_hide": self.mode == HIDDEN,
        }
