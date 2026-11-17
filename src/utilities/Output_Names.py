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
import uuid

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


def new_partial_file(folder, prefix, suffix):
    """Make the empty file an output is written into before it replaces the output; return its path.

    The name is ``prefix``, a random part and ``suffix``, in ``folder``, and the file is made with
    O_EXCL, so it is this call's own, as tempfile.mkstemp's is. Unlike mkstemp's, which are for
    their owner alone (mode 0600), it is made as any file a program saves is: the umask decides who
    may read it. The output os.replace makes of it keeps that mode, so a logo, a label workbook or
    an exported FASTA can be read by whoever can read a printed PNG beside it.
    """
    path = os.path.join(folder, f"{prefix}{uuid.uuid4().hex}{suffix}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    os.close(descriptor)
    return path


def publish_without_overwrite(staged, destination, claim_name_first=None):
    """Make ``destination`` hold the staged file without ever replacing an existing file.

    Raises FileExistsError if ``destination`` exists (``staged`` is then left where it is). On
    success the staged name is gone. A hard link is tried first, as it refuses to replace a file
    in one step. Drives without hard links (exFAT, FAT32, some network shares) refuse it with
    some other OSError; any such error switches to a second way that does not overwrite either:
    where a rename refuses an existing destination (Windows) it is a rename, elsewhere (a rename
    replaces silently there) the name is first claimed with O_CREAT | O_EXCL and then replaced.
    ``claim_name_first`` chooses between the two (default: not on Windows). If the second way
    fails too, its error is raised, chained from the link's.
    """
    if claim_name_first is None:
        claim_name_first = os.name != "nt"
    try:
        os.link(staged, destination)
    except FileExistsError:
        raise
    except OSError as link_error:
        try:
            if claim_name_first:
                descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o666)
                os.close(descriptor)
                try:
                    os.replace(staged, destination)
                except BaseException:
                    try:
                        os.unlink(destination)
                    except OSError:
                        pass
                    raise
            else:
                os.rename(staged, destination)
        except OSError as error:
            raise error from link_error
        return
    os.unlink(staged)


_SYSTEM_LEFTOVER_NAMES = {"thumbs.db", "desktop.ini"}


def is_system_leftover(name):
    """Return whether ``name`` is a file the operating system leaves in a folder it was shown.

    Finder makes ``.DS_Store`` and, on drives that cannot hold extra file data, AppleDouble files
    (``._name``); Explorer makes ``Thumbs.db`` and ``desktop.ini``. None of them is the user's
    data, so a folder that holds nothing else counts as empty. The Windows names match in any
    letter case, as that file system treats them.
    """
    return name == ".DS_Store" or name.startswith("._") or name.lower() in _SYSTEM_LEFTOVER_NAMES


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
