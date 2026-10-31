# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Where NCBI BLAST+ is, and how it is run.

One lookup for everything that needs blastp and makeblastdb: the
Align_Substitution_Matrix tool, the benchmark, the MCP capabilities report and
the Tools window. The BLASTP Directory setting is looked at only when
``find_executables`` is called, never when this module is imported, because a
tool script that Save & Run or an MCP job starts loads its settings after its
own imports.

* A BLASTP Directory that is set must hold both programs. Nothing else is
  tried when it does not, so a typo is an error and not a quiet switch to
  another BLAST+.
* A blank one searches the PATH, then (on Windows) the newest NCBI installer
  folder under Program Files, by version number, or (elsewhere) the usual Unix
  folders.

Every BLAST+ process this program starts gets ``blast_environment()``, which
turns BLAST+'s usage reporting off.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import NamedTuple

PROGRAMS = ("blastp", "makeblastdb")

# The folders searched on Linux and macOS when the PATH has no BLAST+.
UNIX_FOLDERS = (
    "/usr/local/ncbi/blast/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/homebrew/bin",
)

# BLAST+ phones home with usage statistics unless told not to. The value is
# read as a boolean. DO_NOT_TRACK and NCBI_USAGE_REPORT_ENABLED, when a user has
# set them, take precedence over it inside BLAST+, so this module neither sets
# nor clears them.
USAGE_REPORT_VARIABLE = "BLAST_USAGE_REPORT"

SOURCE_SETTING = "BLASTP Directory"
SOURCE_PATH = "PATH"
STANDARD_FOLDER = "standard install folder"


class BlastError(RuntimeError):
    """BLAST+ cannot be found or run; the message says why in plain words."""


class BlastExecutables(NamedTuple):
    """The two programs to run and where the lookup found them."""

    blastp: str
    makeblastdb: str
    source: str  # "BLASTP Directory", "PATH" or "<folder> (standard install folder)"

    @property
    def directory(self):
        return os.path.dirname(self.blastp)


def is_windows():
    return sys.platform == "win32"


def program_file(name):
    """The file name of a BLAST+ program on this platform."""
    return name + ".exe" if is_windows() else name


def _paths_in(folder):
    return tuple(os.path.join(folder, program_file(name)) for name in PROGRAMS)


def _usable(path):
    return os.path.isfile(path) and (is_windows() or os.access(path, os.X_OK))


def _join(names):
    names = [f"'{name}'" for name in names]
    return " and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + f" and {names[-1]}"


def blast_environment(environ=None):
    """The environment for a BLAST+ process: the current one, with usage reporting off."""
    environment = dict(os.environ if environ is None else environ)
    environment[USAGE_REPORT_VARIABLE] = "false"
    return environment


def _from_setting(folder):
    if not os.path.isdir(folder):
        raise BlastError(
            f"The {SOURCE_SETTING} '{folder}' is not a folder that exists. "
            f"Correct it, or leave it blank to search for BLAST+ automatically."
        )
    paths = _paths_in(folder)
    missing = [os.path.basename(path) for path in paths if not os.path.isfile(path)]
    if missing:
        hint = ""
        inner = os.path.join(folder, "bin")
        if all(_usable(path) for path in _paths_in(inner)):
            hint = f" Its 'bin' folder holds them: use '{inner}'."
        raise BlastError(
            f"The {SOURCE_SETTING} '{folder}' does not contain {_join(missing)}.{hint} "
            f"Correct it, or leave it blank to search for BLAST+ automatically."
        )
    if not is_windows():
        not_runnable = [os.path.basename(path) for path in paths if not os.access(path, os.X_OK)]
        if not_runnable:
            raise BlastError(
                f"{_join(not_runnable)} in the {SOURCE_SETTING} '{folder}' "
                f"{'is' if len(not_runnable) == 1 else 'are'} not executable."
            )
    return BlastExecutables(*paths, SOURCE_SETTING)


def _on_path():
    blastp = shutil.which("blastp")
    if not blastp:
        return None
    blastp = os.path.abspath(blastp)  # the workers run in another folder
    beside = os.path.join(os.path.dirname(blastp), program_file("makeblastdb"))
    makeblastdb = beside if _usable(beside) else shutil.which("makeblastdb")
    if not makeblastdb:
        return None
    return BlastExecutables(blastp, os.path.abspath(makeblastdb), SOURCE_PATH)


def windows_ncbi_root():
    """The folder the NCBI installer puts each BLAST+ version in."""
    return os.path.join(os.environ.get("ProgramFiles") or r"C:\Program Files", "NCBI")


def _version_key(name):
    match = re.search(r"\d+(?:\.\d+)*", name)
    return tuple(int(part) for part in match.group().split(".")) if match else ()


def windows_install_folders(root=None):
    """The ``bin`` folders under the NCBI installer folder that hold both programs, newest version first.

    The folders are named for their version (blast-2.17.0+), and 2.17 is newer
    than 2.9, so they are ordered by their numbers and not by their letters.
    """
    root = windows_ncbi_root() if root is None else root
    try:
        names = os.listdir(root)
    except OSError:
        return []
    names.sort(key=lambda name: (_version_key(name), name.lower()), reverse=True)
    bins = (os.path.join(root, name, "bin") for name in names)
    return [folder for folder in bins if all(os.path.isfile(path) for path in _paths_in(folder))]


def _standard_folders():
    if is_windows():
        return windows_install_folders()
    return [folder for folder in UNIX_FOLDERS if all(_usable(path) for path in _paths_in(folder))]


def find_automatic():
    """What the search of a blank BLASTP Directory uses right now, or None. Never raises, never runs BLAST."""
    found = _on_path()
    if found:
        return found
    folders = _standard_folders()
    if folders:
        return BlastExecutables(*_paths_in(folders[0]), f"{folders[0]} ({STANDARD_FOLDER})")
    return None


def describe_automatic():
    """One line for a placeholder: where the automatic search would look right now, or that it found nothing."""
    found = find_automatic()
    if found is None:
        return "BLAST+ not found"
    if found.source == SOURCE_PATH:
        return f"{found.directory} ({SOURCE_PATH})"
    return found.source


def _not_found_message():
    place = "the standard NCBI install folder" if is_windows() else "the usual BLAST+ folders"
    message = f"NCBI BLAST+ (blastp and makeblastdb) was not found on the PATH or in {place}."
    blastp = shutil.which("blastp")
    if blastp:
        message += f" blastp is at '{blastp}', but makeblastdb is not beside it or on the PATH."
    return f"{message} Install BLAST+ and set the {SOURCE_SETTING} to the folder that holds both."


def find_executables(blastp_dir=""):
    """blastp and makeblastdb for a BLASTP Directory setting; BlastError when the setting or the search fails.

    ``blastp_dir`` is the setting as the settings loader left it (a relative
    value is already absolute). Blank means search automatically.
    """
    folder = "" if blastp_dir is None else str(blastp_dir).strip()
    if folder:
        return _from_setting(folder)
    found = find_automatic()
    if found is None:
        raise BlastError(_not_found_message())
    return found


def query_version(executable, timeout=30):
    """The first non-empty line ``executable -version`` prints, such as "blastp: 2.17.0+"."""
    try:
        completed = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=blast_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BlastError(f"Unable to query BLASTP version using '{executable}': {error}") from error
    output = completed.stdout or completed.stderr or ""
    version = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not version:
        raise BlastError(f"BLASTP at '{executable}' returned no version information.")
    return version


_versions = {}


def cached_version(executable, timeout=30):
    """``query_version``, asked once per file: a replaced file (another modification time or size) is asked again."""
    try:
        status = os.stat(executable)
        key = (os.path.abspath(executable), status.st_mtime_ns, status.st_size)
    except OSError:
        key = (executable, None, None)
    if key not in _versions:
        _versions[key] = query_version(executable, timeout)
    return _versions[key]
