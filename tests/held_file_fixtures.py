"""A file another program holds, as the tests of the save commands simulate it.

On Windows a virus scan, an indexer or a sync client may have a file open for a
moment, and a rename or a deletion of that file then fails (WinError 32, 33 or
5). utilities.Job_Cancellation waits and tries again.

held_file_error() is the error, whatever the platform the test runs on.
transient_hold() makes os.replace raise it a few times, as if Windows were the
platform, and records the waits the retry asks for instead of making them.
hold_without_delete_sharing() really holds a file on Windows, as a scanner does.
"""
import errno
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utilities import Job_Cancellation  # noqa: E402


def held_file_error(winerror=32, filename=None):
    """The PermissionError Windows raises for a file another program holds: WinError 32, 33 or 5."""
    error = PermissionError(
        errno.EACCES, "The process cannot access the file because it is being used by another process", filename
    )
    error.winerror = winerror  # Only Windows gives an OSError this attribute itself.
    return error


@contextmanager
def transient_hold(failures, winerror=32):
    """os.replace fails ``failures`` times as Windows does for a held file, and then renames as usual.

    The platform reads as Windows meanwhile, and the waits are not made but listed. Yields
    (the targets os.replace was asked for, the waits in seconds).
    """
    real_replace = os.replace
    attempts, pauses = [], []

    def replace(source, target):
        attempts.append(target)
        if len(attempts) <= failures:
            raise held_file_error(winerror, source)
        real_replace(source, target)

    def pause(seconds):
        pauses.append(seconds)
        return False

    with mock.patch.object(Job_Cancellation, "_on_windows", return_value=True), \
            mock.patch.object(Job_Cancellation, "_pause", side_effect=pause), \
            mock.patch.object(Job_Cancellation.os, "replace", side_effect=replace):
        yield attempts, pauses


@contextmanager
def hold_without_delete_sharing(path, seconds=None):
    """Hold ``path`` open for the block, on Windows, as a virus scan does: it can still be read and written, but
    not renamed, replaced or deleted. With ``seconds``, a timer lets go of it after that long; the block's end
    lets go of it at the latest. The file is held when the block starts.
    """
    import ctypes
    from ctypes import wintypes

    generic_read, share_read_write, open_existing = 0x80000000, 0x1 | 0x2, 3
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.CreateFileW(os.fspath(path), generic_read, share_read_write, None, open_existing, 0, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    lock, released = threading.Lock(), []

    def release():
        with lock:
            if not released:
                released.append(True)
                kernel32.CloseHandle(handle)

    timer = threading.Timer(seconds, release) if seconds is not None else None
    if timer is not None:
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()
        release()
