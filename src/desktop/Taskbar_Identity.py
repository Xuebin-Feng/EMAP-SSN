# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""What a window tells the Windows taskbar about itself: its AppUserModelID, its small icon, its name, and how to start it again.

The windows (desktop/Desktop_App.py re-exports every name here, which is where they come from for the main windows
and for opt_vr) and the start-up splash (desktop/Splash.py, which is on screen long before the windows' heavy imports
and so cannot import Desktop_App) set the same properties, so the splash and the window it waits for are one entry in
the taskbar. This module imports only ctypes, QtCore and the standard library.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import uuid

from PySide6.QtCore import QEvent, QObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# The AppUserModelID of each window's process and of its main window. They are deliberately not the ID of any
# shortcut: a window whose ID a shortcut carries (one on the desktop, as the installers make them) gets that
# shortcut's icon and name in the taskbar, whatever icon and name the window sets, and the shortcuts use the large
# art. A window sets its ID for its process before its first window (set_app_user_model_id) and for its main window
# with its own icon, name and start command (TaskbarIdentity), so the taskbar says "EMAP-SSN" and not "Python",
# shows the small icon, keeps each kind of window in a group of its own, and offers a pin that starts the window
# again. The Viewer, which cannot be started again on its own, sets only its process's.
APP_USER_MODEL_IDS = {
    "config": "ca.utoronto.emapssn.config.window",
    "tools": "ca.utoronto.emapssn.tools.window",
    "viewer": "ca.utoronto.emapssn.viewer",
}

# The launcher the desktop shortcuts start (install.bat builds it from src/bin/Launcher.cs), and its batch file.
LAUNCHER_EXE = PROJECT_ROOT / "EMAP-SSN.exe"
LAUNCHER_BATCH = PROJECT_ROOT / "src" / "bin" / "EMAPSSN_Desktop_Launcher.bat"


def set_app_user_model_id(app_id):
    """Give this process its taskbar identity (Windows only; call it before the first window exists).

    Returns whether Windows accepted it; anywhere else it does nothing and returns False.
    """
    if sys.platform != "win32":
        return False
    try:
        function = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        function.argtypes = [ctypes.c_wchar_p]
        function.restype = ctypes.c_long
        return function(app_id) == 0
    except (AttributeError, OSError):
        return False


def launcher_relaunch_command(word, batch_file=LAUNCHER_BATCH, batch_argument=None):
    """The command line that starts a window the way its desktop shortcut does (what a pinned copy of the window runs).

    ``word`` names the window to EMAP-SSN.exe ("viewer" is Config, "tools" is Tools; src/bin/Launcher.cs lists
    them). Where install.bat could not build that, the shortcut runs ``batch_file`` in a console with
    ``batch_argument`` (the word, when None), and so does this.
    """
    if LAUNCHER_EXE.is_file():
        return f'"{LAUNCHER_EXE}" {word}'
    argument = word if batch_argument is None else batch_argument
    batch = f'"{batch_file}" {argument}'.strip()
    shell = os.environ.get("COMSPEC") or "cmd.exe"
    return f'"{shell}" /d /c "{batch}"'


# What a window tells the taskbar through its own properties (SHGetPropertyStoreForWindow), which are
# System.AppUserModel.* in propkey.h: one format ID, a property ID for each.
_TASKBAR_PROPERTY_FORMAT = "9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"
_TASKBAR_RELAUNCH_COMMAND, _TASKBAR_RELAUNCH_ICON, _TASKBAR_RELAUNCH_NAME, _TASKBAR_ID = 2, 3, 4, 5
_PROPERTY_STORE_IID = "886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"
_VT_LPWSTR = 31
# The slots of IPropertyStore's table of methods that are used.
_PROPERTY_STORE_RELEASE, _PROPERTY_STORE_SET_VALUE = 2, 6


class _Guid(ctypes.Structure):
    _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16), ("data3", ctypes.c_uint16),
                ("data4", ctypes.c_ubyte * 8)]


class _PropertyKey(ctypes.Structure):
    _fields_ = [("format_id", _Guid), ("property_id", ctypes.c_uint32)]


class _PropVariant(ctypes.Structure):
    # A PROPVARIANT that holds a string: the type, three reserved words, the pointer, and the room of the union's
    # other members, which makes it as large as Windows' on both 32 and 64 bits.
    _fields_ = [("type", ctypes.c_uint16), ("reserved", ctypes.c_uint16 * 3), ("pointer", ctypes.c_void_p),
                ("room", ctypes.c_void_p)]


def _guid(text):
    return _Guid.from_buffer_copy(uuid.UUID(text).bytes_le)


def _set_window_texts(hwnd, texts):
    """Set text properties of a native window in order: ``texts`` is (property ID, text) pairs. Returns whether all took."""
    shell32, shlwapi, ole32 = ctypes.windll.shell32, ctypes.windll.shlwapi, ctypes.windll.ole32
    shell32.SHGetPropertyStoreForWindow.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
    shell32.SHGetPropertyStoreForWindow.restype = ctypes.c_long
    shlwapi.SHStrDupW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
    shlwapi.SHStrDupW.restype = ctypes.c_long
    ole32.PropVariantClear.argtypes = [ctypes.POINTER(_PropVariant)]
    ole32.PropVariantClear.restype = ctypes.c_long
    store = ctypes.c_void_p()
    if shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(_guid(_PROPERTY_STORE_IID)), ctypes.byref(store)) != 0:
        return False
    methods = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    set_value = ctypes.WINFUNCTYPE(
        ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_PropertyKey), ctypes.POINTER(_PropVariant)
    )(methods[_PROPERTY_STORE_SET_VALUE])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(methods[_PROPERTY_STORE_RELEASE])
    accepted = True
    try:
        for property_id, text in texts:
            value = _PropVariant(type=_VT_LPWSTR)
            pointer = ctypes.c_void_p()
            if shlwapi.SHStrDupW(text, ctypes.byref(pointer)) != 0:
                return False
            value.pointer = pointer.value
            key = _PropertyKey(_guid(_TASKBAR_PROPERTY_FORMAT), property_id)
            accepted = set_value(store, ctypes.byref(key), ctypes.byref(value)) == 0 and accepted
            ole32.PropVariantClear(ctypes.byref(value))  # the property store keeps a copy
    finally:
        release(store)
    return accepted


def set_window_taskbar_identity(window, app_id, icon_path, display_name, relaunch_command):
    """Give a main window its own taskbar identity (Windows only; the window needs a native handle, which this makes).

    ``app_id`` is the window's ID, ``icon_path`` the .ico its taskbar button shows and a pinned copy of it,
    ``display_name`` what the taskbar calls it and a pinned copy, and ``relaunch_command`` the command line (as
    ShellExecute runs it) that a pinned copy starts; Windows ignores the icon and the name unless it has the command
    too. A window that cannot be started again on its own (the Viewer) has ``relaunch_command`` None and gets its ID
    only. Returns whether Windows accepted them all; anywhere else it does nothing and returns False.
    """
    if sys.platform != "win32":
        return False
    try:
        # The ID goes last, which is when the taskbar looks at the rest.
        texts = () if relaunch_command is None else (
            (_TASKBAR_RELAUNCH_COMMAND, relaunch_command),
            (_TASKBAR_RELAUNCH_ICON, f"{icon_path},0"),
            (_TASKBAR_RELAUNCH_NAME, display_name),
        )
        return _set_window_texts(int(window.winId()), (*texts, (_TASKBAR_ID, app_id)))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


class TaskbarIdentity(QObject):
    """Keeps a main window's taskbar identity (set_window_taskbar_identity) on its native handle.

    A window makes one while it is built: TaskbarIdentity(self, app_id, icon_path, display_name,
    launcher_relaunch_command(...)); so does the start-up splash of that window, with the same arguments, so that the
    two are one group. The properties are set when Qt gives the window its native handle, which is
    before it is first shown, and again whenever Qt gives it another. Off Windows it does nothing. The icon must
    be an .ico.
    """

    def __init__(self, window, app_id, icon_path, display_name, relaunch_command):
        super().__init__(window)
        self._window = window
        self._arguments = (app_id, icon_path, display_name, relaunch_command)
        if sys.platform == "win32":
            window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._window and event.type() == QEvent.Type.WinIdChange:
            set_window_taskbar_identity(self._window, *self._arguments)
        return False
