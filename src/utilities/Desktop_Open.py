# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Open a file, a folder or a web page in the program the user's desktop has for it.

Every "open this folder", "open this log" and "open this page" button goes through
here. On Windows that is what it always was: Qt's QDesktopServices (and, for the
Viewer's own pages, the webbrowser module). On Linux and macOS the program that
opens the target (the desktop's xdg-open or gio, macOS's open, or the user's
BROWSER) is started with /dev/null for stdin, stdout and stderr, in a session of
its own.

Why not leave it to Qt or webbrowser: both start the opener with the caller's own
descriptors. A window that captures its output (utilities/Console_Capture.py)
has pipes there, and a file manager, an editor or a browser that this window
started would hold them for as long as it ran: the capture's close would wait for
it at exit, and it would die of SIGPIPE on its next write once the window had
gone. Nothing in this module needs Qt on Linux and macOS; Windows loads it when
it is first asked.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading

# The programs that open a target, in the order they are tried, as (program, arguments before the target).
_MACOS_OPENERS = (("open", ()),)
_OTHER_OPENERS = (("xdg-open", ()), ("gio", ("open",)))


def _reap(process: subprocess.Popen) -> None:
    """Collect the opener's exit status when it ends, so that it never lingers as a zombie."""
    threading.Thread(target=process.wait, name="desktop-opener-reaper", daemon=True).start()


def _start(command: list[str]) -> bool:
    """Start ``command`` with no standard streams, in a session of its own; whether it started.

    This is the one place a program is started here, which is where the tests'
    desktop guard (tests/desktop_openers.py) stands.
    """
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, ValueError):
        return False
    _reap(process)
    return True


def _browser_commands(url: str, environment=None) -> list[list[str]]:
    """The commands the user's BROWSER names for ``url``, as the webbrowser module reads it.

    BROWSER is a list of commands (separated as PATH is); a ``%s`` in one stands
    for the address, and without one the address goes last.
    """
    environment = os.environ if environment is None else environment
    commands = []
    for entry in environment.get("BROWSER", "").split(os.pathsep):
        if not entry.strip():
            continue
        try:
            words = shlex.split(entry)
        except ValueError:
            continue
        if not words:
            continue
        if any("%s" in word for word in words):
            words = [word.replace("%s", url) for word in words]
        else:
            words.append(url)
        commands.append(words)
    return commands


def _desktop_commands(target: str, platform_name: str, which=None) -> list[list[str]]:
    """The commands that open ``target`` on this desktop, those whose program is installed."""
    which = which or shutil.which
    openers = _MACOS_OPENERS if platform_name == "darwin" else _OTHER_OPENERS
    return [[program, *before, target] for program, before in openers if which(program)]


def _open_with_the_desktop(target: str, *, browser: bool, platform_name: str | None = None) -> bool:
    """Linux and macOS: start the first opener that starts; a web address tries BROWSER first."""
    platform_name = platform_name or sys.platform
    if target.startswith('-'):
        return False  # an opener would read it as an option
    commands = _browser_commands(target) if browser else []
    commands += _desktop_commands(target, platform_name)
    return any(_start(command) for command in commands)


def open_path(path) -> bool:
    """Open a file or a folder in the program the desktop has for it; whether one was started."""
    path = os.fspath(path)
    if sys.platform == "win32":
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        return bool(QDesktopServices.openUrl(QUrl.fromLocalFile(path)))
    return _open_with_the_desktop(os.path.abspath(path), browser=False)


def open_url(url: str, *, new_tab: bool = False) -> bool:
    """Open a web address in the user's browser; whether a program was started.

    ``new_tab`` is what the Viewer's own pages ask for, and on Windows they went
    through the webbrowser module, which the other addresses (a licence link)
    never did; that stays as it was.
    """
    url = str(url)
    if sys.platform == "win32":
        if new_tab:
            import webbrowser

            return bool(webbrowser.open(url, new=2))
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        return bool(QDesktopServices.openUrl(QUrl(url)))
    return _open_with_the_desktop(url, browser=True)


__all__ = ["open_path", "open_url"]
