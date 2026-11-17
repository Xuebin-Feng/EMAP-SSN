"""When a test may run git.

On a Mac without Apple's command-line tools, /usr/bin/git is a stub that opens an
"Install Command Line Developer Tools" dialog instead of running. A test that wants
git therefore asks here first, and skips with the reason this gives:

    from tests.git_availability import git_executable_for

    git = git_executable_for(PROJECT_ROOT)      # raises unittest.SkipTest
    subprocess.run([git, "-C", str(PROJECT_ROOT), ...])

Git is usable when (1) an executable is found; (2) on macOS, an executable that is
/usr/bin/git has command-line tools behind it (`xcode-select -p` exits 0, which shows
no dialog); and, for a question about a repository, (3) the folder holds a .git entry
(looked for without running git: a downloaded archive has none).

GitAvailabilityTests in tests/test_run_sections.py (its guard test) fails for a test file that runs git
without going through this module.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

APPLE_STUB = "/usr/bin/git"
XCODE_SELECT_TIMEOUT = 10


def _tools_installed(platform, run):
    try:
        done = run(["xcode-select", "-p"], capture_output=True, timeout=XCODE_SELECT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def git_problem(root=None, *, platform=None, which=None, run=None, has_git_entry=None):
    """Why git cannot be used here (a sentence), or None when it can.

    root: a folder whose repository the test asks about, or None for a test that only
    needs the executable. The other arguments stand in for the host in tests.
    """
    platform = sys.platform if platform is None else platform
    which = shutil.which if which is None else which
    run = subprocess.run if run is None else run
    found = which("git")
    if not found:
        return "git is not installed"
    if platform == "darwin" and posixpath.normpath(found) == APPLE_STUB and not _tools_installed(platform, run):
        return "/usr/bin/git is Apple's stub and the command-line tools are not installed"
    if root is not None:
        if has_git_entry is None:
            has_git_entry = (Path(root) / ".git").exists()
        if not has_git_entry:
            return "this folder is not a git work tree"
    return None


def git_executable_for(root=None):
    """The git executable to run, or a skip of the calling test."""
    problem = git_problem(root)
    if problem:
        raise unittest.SkipTest(problem)
    return shutil.which("git")
