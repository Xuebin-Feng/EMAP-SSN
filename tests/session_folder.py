"""The private folder a test run works in, and getting rid of it again.

A root test process (one `python -m unittest`, or one section of
tests/run_sections.py) works inside one folder, `emapssn-test-sessions-*` in the
system temp folder:

    <session>/                owner.json, app_settings.json, gpu_kernels/, ...
    <session>/temp/           the process's own temp folder: TMP, TEMP, TMPDIR and
                              tempfile.tempdir point here, so whatever a test (or a
                              child it starts) leaves in "the temp folder" stays
                              inside the session and goes with it

That keeps the developer's temp folder clean even when a test forgets its own
clean-up, and makes a leak visible: what is in <session>/temp when the process
ends was left behind (leaked() lists it).

Who removes the folder:

* the runner makes the folder, names it in EMAPSSN_TEST_SESSION_DIR, and removes it
  after the section's process has ended, however it ended (a timeout, a crash, a
  kill), so a process that never got to run its exit handlers leaves nothing;
* a process nobody told removes its own folder as it exits, retrying while Windows
  still holds something open;
* a process that starts sweeps the folders of processes that have ended without
  clearing up (owner.json names the process; a folder from before owner.json
  counts as stale when nothing has touched it for a day).
"""
import json
import os
import shutil
import stat
import sys
import tempfile
import time

PREFIX = "emapssn-test-sessions-"
SESSION_ENV = "EMAPSSN_TEST_SESSION_DIR"
OWNER_FILE = "owner.json"
TEMP_NAME = "temp"
STALE_SECONDS = 24 * 3600  # untouched this long and no owner.json: nobody is using it
REMOVE_SECONDS = 5.0  # how long a folder that Windows still holds is retried


def _process_started(pid):
    """When process `pid` began (seconds since the epoch), or None where that cannot be read."""
    try:
        import psutil

        return round(psutil.Process(pid).create_time(), 2)
    except Exception:
        return None


def adopt(path):
    """Make `path` this process's session folder: its owner.json and its temp folder. Return the temp folder."""
    temp = os.path.join(path, TEMP_NAME)
    os.makedirs(temp, exist_ok=True)
    owner = {"pid": os.getpid(), "started": _process_started(os.getpid())}
    with open(os.path.join(path, OWNER_FILE), "w", encoding="utf-8") as handle:
        json.dump(owner, handle)
    return temp


def create(label=""):
    """Make a new session folder in the system temp folder and return its path."""
    return tempfile.mkdtemp(prefix=PREFIX + (label + "-" if label else ""))


def temp_environment(temp, environ=None):
    """The variables that make `temp` the temp folder of a process and its children.

    Python looks at TMPDIR, then TEMP, then TMP; Windows programs read TMP and TEMP.
    TMPDIR is set wherever it counts: on POSIX, and where the environment has one.
    A test that gives a child another temp folder must set all three.
    """
    environ = os.environ if environ is None else environ
    names = ["TMP", "TEMP"]
    if os.name != "nt" or "TMPDIR" in environ:
        names.append("TMPDIR")
    return {name: temp for name in names}


def use_temp(temp):
    """Point this process, and the children it starts, at `temp` as the temp folder."""
    os.environ.update(temp_environment(temp))
    tempfile.tempdir = temp


def leaked(path):
    """What is left in the session's temp folder, as names (empty when nothing is)."""
    try:
        return sorted(os.listdir(os.path.join(path, TEMP_NAME)))
    except OSError:
        return []


def _clear_read_only(function, path, *_):
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except OSError:
        pass  # rmtree's next pass reports it


def remove_tree(path, seconds=REMOVE_SECONDS):
    """Remove the folder, retrying while something still holds part of it. Return None
    once it is gone, else the last error."""
    deadline = time.monotonic() + seconds
    pause = 0.05
    while True:
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_clear_read_only)
            else:
                shutil.rmtree(path, onerror=_clear_read_only)
        except FileNotFoundError:
            return None
        except OSError as error:
            problem = error
        else:
            return None
        if not os.path.exists(path):
            return None
        if time.monotonic() >= deadline:
            return problem
        time.sleep(pause)
        pause = min(pause * 2, 0.5)


def _alive(owner):
    """Whether the process an owner.json names is still running (True when that cannot be told)."""
    pid, started = owner.get("pid"), owner.get("started")
    if not isinstance(pid, int):
        return True
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        if started is None:
            return True
        return abs(psutil.Process(pid).create_time() - started) < 1.0
    except ImportError:
        return True  # cannot tell, so it counts as running
    except Exception:
        return True


def stale(path, now=None):
    """Whether no process is using the session folder at `path` any longer."""
    now = time.time() if now is None else now
    try:
        with open(os.path.join(path, OWNER_FILE), encoding="utf-8") as handle:
            owner = json.load(handle)
    except FileNotFoundError:
        try:
            return now - os.stat(path).st_mtime > STALE_SECONDS
        except OSError:
            return False
    except (OSError, ValueError):
        return False
    return isinstance(owner, dict) and not _alive(owner)


def sweep(base=None, keep=()):
    """Remove the session folders whose processes have ended, except those in `keep`.
    Return the names removed."""
    base = base or tempfile.gettempdir()
    removed = []
    try:
        names = [name for name in os.listdir(base) if name.startswith(PREFIX)]
    except OSError:
        return removed
    keep = {os.path.normcase(os.path.abspath(path)) for path in keep}
    for name in names:
        path = os.path.join(base, name)
        if os.path.normcase(os.path.abspath(path)) in keep or not os.path.isdir(path):
            continue
        if stale(path) and remove_tree(path, seconds=0.5) is None:
            removed.append(name)
    return removed
