# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The interpreter a window runs under, so that the macOS Dock and menu bar show the window's name.

macOS names a running program after its program file, and the managed environment's file is ``.venv/bin/python``,
so every window was called "python". A symbolic link in the same folder that points at ``python`` and carries the
window's name (``.venv/bin/EMAP-SSN``, ``.venv/bin/EMAP-SSN Tools``) gives the window that name and runs the same
interpreter and environment.

Only window processes are started through a link: the launcher's splash, the window itself and the Viewer a window
starts. In a process started that way ``sys.executable`` is the link, so ``restore_real_python`` puts the real
interpreter back at the start of each window script, and everything the window starts after that (tools, the layout
generator, spawned workers, the MCP server's jobs) runs ``.venv/bin/python`` as it always did. The text of the
commands Config and Tools show to the user is unchanged for the same reason.

The link is made by setup and by every launcher that finds it missing (a ``uv sync`` that rebuilds the environment
removes it), only inside the project's ``.venv/bin``, and a link that cannot be made means plain ``python``: it never
fails a start. Linux and Windows are not touched. The shell scripts do the same in ``ssn_window_python``
(install.sh); tests/test_launcher_scripts.py checks that both name the windows alike.

Standard library only: the launcher imports this before anything heavy is loaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The window kinds of desktop/Desktop_Launcher.py and what each is called in the Dock: the names of the macOS
# bundles (EMAP-SSN.app, EMAP-SSN Tools.app). A Viewer ("session") is a window of the first.
WINDOW_NAMES = {"viewer": "EMAP-SSN", "session": "EMAP-SSN", "tools": "EMAP-SSN Tools"}
LINK_NAMES = frozenset(WINDOW_NAMES.values())
REAL_NAME = "python"


def venv_bin(root) -> Path:
    return Path(root) / ".venv" / "bin"


def ensure_link(directory, name: str) -> Path | None:
    """The link ``directory/name`` to ``python`` in the same folder, made if it is missing; None if it cannot be had.

    A link that points elsewhere, or nowhere, is replaced. A real file of that name is left alone.
    """
    directory = Path(directory)
    link = directory / name
    try:
        if not (directory / REAL_NAME).is_file():
            return None
        if os.path.islink(link):
            if os.readlink(link) == REAL_NAME and link.is_file():
                return link
            os.unlink(link)
        elif os.path.lexists(link):
            return None
        os.symlink(REAL_NAME, link)
    except OSError:
        # Two launches may make it at once: whichever came second finds it done.
        try:
            if os.path.islink(link) and os.readlink(link) == REAL_NAME and link.is_file():
                return link
        except OSError:
            pass
        return None
    return link


def _in_venv_bin(python: Path, root) -> bool:
    if python.name != REAL_NAME and python.name not in LINK_NAMES:
        return False
    try:
        return os.path.samefile(python.parent, venv_bin(root))
    except OSError:
        return False


def window_python(root, kind: str, python=None, platform_name: str | None = None) -> str:
    """The interpreter to start the window of ``kind`` with: its link on macOS, else ``python`` unchanged.

    ``python`` (default: this process's) is the interpreter the caller would have used. Only the project's own
    managed interpreter is replaced, and a link that cannot be made leaves it as it was.
    """
    python = sys.executable if python is None else python
    name = WINDOW_NAMES.get(kind)
    if (platform_name or sys.platform) != "darwin" or name is None:
        return str(python)
    if not _in_venv_bin(Path(python), root):
        return str(python)
    link = ensure_link(venv_bin(root), name)
    return str(python) if link is None else str(link)


def restore_real_python(platform_name: str | None = None) -> None:
    """In a window process started through a link, make ``sys.executable`` the real ``python`` again."""
    if (platform_name or sys.platform) != "darwin":
        return
    current = Path(sys.executable)
    if current.name in LINK_NAMES:
        real = current.with_name(REAL_NAME)
        if real.is_file():
            sys.executable = str(real)
