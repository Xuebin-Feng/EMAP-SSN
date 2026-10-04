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
"""
import re

# Characters Windows does not allow in a file name. ':' would also select an
# NTFS alternate data stream ("name.txt:stream"), hidden inside another file.
_UNSUPPORTED_CHARACTERS = re.compile(r'[<>:"|?*\x00-\x1f]')


def validate_output_basename(filename):
    """Return ``filename`` without surrounding whitespace if it is a plain name.

    Raises ValueError for an empty name, ``.`` or ``..``, a path separator, or a
    character Windows does not allow in file names.
    """
    filename = str(filename).strip()
    if not filename:
        raise ValueError("Filename cannot be empty.")
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("Filename must not include a directory or path separators.")
    if _UNSUPPORTED_CHARACTERS.search(filename):
        raise ValueError(f"Filename contains unsupported characters: '{filename}'.")
    return filename
