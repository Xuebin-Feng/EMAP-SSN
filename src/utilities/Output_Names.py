# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
r"""Plain file names for the files Viewer commands write.

A command that takes an output name writes it into one configured folder, and
``os.path.join(folder, name)`` discards that folder whenever the name is a path
of its own. On Windows ``os.path.isabs`` misses two such forms, ``C:name``
(relative to drive C's current directory) and ``\name`` (the current drive's
root), and ``\\host\share\name`` opens a network connection that offers the
user's credentials. Commands reach the Viewer from the console, the web agent
and MCP clients alike, so a name is accepted only when it cannot be a path.

A command that writes several files also needs their names to stay apart:
in a folder that ignores letter case, "Alpha" and "alpha" are one file.
"""
import os
import re
import sys
import tempfile

from utilities.Localization import Message

# Characters Windows does not allow in a file name. ':' would also select an
# NTFS alternate data stream ("name.txt:stream"), hidden inside another file.
_UNSUPPORTED_CHARACTERS = re.compile(r'[<>:"|?*\x00-\x1f]')


def validate_output_basename(filename):
    """Return ``filename`` without surrounding whitespace if it is a plain name.

    Raises ValueError for an empty name, ``.`` or ``..``, a path separator, or a
    character Windows does not allow in file names. The error holds a Message:
    str(error) is English, and a window shows it translated.
    """
    filename = str(filename).strip()
    if not filename:
        raise ValueError(Message("Filename cannot be empty."))
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(Message("Filename must not include a directory or path separators."))
    if _UNSUPPORTED_CHARACTERS.search(filename):
        raise ValueError(Message("Filename contains unsupported characters: '{file}'.", file=filename))
    return filename


def first_case_collision(names):
    """Return the first two of ``names`` that differ only in letter case, or None."""
    first_name_by_lower = {}
    for name in names:
        first_name = first_name_by_lower.setdefault(name.lower(), name)
        if first_name != name:
            return first_name, name
    return None


def folder_ignores_case(path):
    """Return whether names differing only in letter case are one file in ``path``.

    Case sensitivity belongs to the volume, and on Windows to each folder, not
    to the OS: APFS can be case-sensitive, and Linux mounts FAT, NTFS and SMB
    volumes that ignore case. ``path`` need not exist yet, so its nearest
    existing folder is probed, whose file system and case setting a new
    subfolder takes. The probe makes and removes one empty file; if it can't,
    the answer is what Windows and macOS (ignore case) and Linux (doesn't)
    do by default.
    """
    folder = os.path.abspath(path)
    while not os.path.isdir(folder):
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    try:
        handle, probe = tempfile.mkstemp(prefix="ssn-case-probe-", dir=folder)
    except OSError:
        return os.name == "nt" or sys.platform == "darwin"
    try:
        os.close(handle)
        # The random name holds lower-case letters, so its upper-case form
        # differs and can only exist as this same file.
        return os.path.exists(os.path.join(folder, os.path.basename(probe).upper()))
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
