"""Keep a test run off the user's desktop: no test may open a folder, a file, a browser
page or a console window for real.

The program reaches the desktop through a handful of calls, which tests/__init__.py
replaces in every test process:

* QDesktopServices.openUrl, which the Windows opener (utilities/Desktop_Open.py) ends
  in for every "open this folder or file" button, and through it open_in_file_manager;
* Desktop_Open._start, the one place the Linux and macOS opener starts a program
  (xdg-open, gio, open or the user's BROWSER);
* webbrowser.open, open_new and open_new_tab, the Viewer's browser pages;
* os.startfile, the Windows fallback of open_in_file_manager;
* subprocess.Popen of an opener program (open, xdg-open, gio, explorer, ...) or with
  CREATE_NEW_CONSOLE, which makes a console window.

A replacement does nothing and says it worked, so the code under test carries on,
and records the call. A test that did not expect it then fails, naming the opener,
its target and the line that called it; a call that arrives with no test running
(a leftover thread) is printed when the process exits. A test that wants to see an
opener called replaces it with its own spy, as before, or, to see the traps
themselves, collects the calls with expecting().

Only processes that import the tests package are covered. A child process a test
starts without it keeps the real calls; the Viewers and Configs the tests start
run headless or offscreen, which asks for no windows.
"""
import atexit
import contextlib
import importlib.abc
import importlib.machinery
import os
import subprocess
import sys
import threading
import unittest
import webbrowser

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIS_FILE = os.path.normcase(os.path.abspath(__file__))
# Files that only pass the call on: the site a failure names is the line that asked for it.
_PASS_THROUGH = frozenset({_THIS_FILE, os.path.normcase(os.path.join(_PROJECT_ROOT, "src", "utilities", "Desktop_Open.py"))})
_LOCK = threading.Lock()
_UNEXPECTED = []  # reached with no test expecting it, not yet reported
_EXPECTING = []  # one list per open expecting() block, innermost last
_REAL_POPEN = subprocess.Popen
_OPENER_PROGRAMS = frozenset(
    {"open", "xdg-open", "gio", "kde-open", "kde-open5", "gnome-open", "explorer", "explorer.exe"}
)
_CREATE_NEW_CONSOLE = 0x10


class DesktopOpenerError(OSError):
    """A program that would open something on the desktop was started in a test."""


# ----------------------------------------------------------------------------
# Recording


def _origin():
    """The test that is running and the project line that made the call, from the stack."""
    test, site = "", ""
    frame = sys._getframe(2)
    while frame is not None:
        path = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
        if not site and path not in _PASS_THROUGH and path.startswith(os.path.normcase(_PROJECT_ROOT)):
            site = f"{os.path.relpath(frame.f_code.co_filename, _PROJECT_ROOT)}:{frame.f_lineno} in {frame.f_code.co_name}"
        if not test:
            owner = frame.f_locals.get("self")
            if isinstance(owner, unittest.TestCase):
                test = owner.id()
        frame = frame.f_back
    return test, site


def _record(kind, target):
    test, site = _origin()
    entry = f"{kind}({target!r}), called from {site or 'outside the project'}" + (f" in {test}" if test else "")
    with _LOCK:
        (_EXPECTING[-1] if _EXPECTING else _UNEXPECTED).append(entry)


def unexpected_calls():
    """The openers reached so far with no test expecting them (a copy)."""
    with _LOCK:
        return list(_UNEXPECTED)


@contextlib.contextmanager
def expecting():
    """Collect the openers reached inside the block instead of failing the test.

    The list it yields holds one description per call.
    """
    calls = []
    with _LOCK:
        _EXPECTING.append(calls)
    try:
        yield calls
    finally:
        with _LOCK:
            _EXPECTING.remove(calls)


# ----------------------------------------------------------------------------
# The replacements


def _open_url(url, *arguments, **keywords):
    text = url.toString() if hasattr(url, "toString") else str(url)
    _record("QDesktopServices.openUrl", text)
    return True


def _web_open(url, new=0, autoraise=True):
    _record("webbrowser.open", url)
    return True


def _web_open_new(url):
    _record("webbrowser.open_new", url)
    return True


def _web_open_new_tab(url):
    _record("webbrowser.open_new_tab", url)
    return True


def _start_file(path, *arguments, **keywords):
    _record("os.startfile", os.fspath(path))


def _program_of(arguments):
    """The program name a Popen call starts, lower case without its folder."""
    if isinstance(arguments, (str, bytes, os.PathLike)):
        first = os.fsdecode(arguments).strip().split(" ")[0]
    else:
        try:
            first = os.fsdecode(next(iter(arguments)))
        except (StopIteration, TypeError, ValueError):
            return ""
    return os.path.basename(first.replace("\\", "/")).lower()


class _GuardedPopen(_REAL_POPEN):
    """subprocess.Popen that refuses the programs that open something on the desktop."""

    def __init__(self, args, *positional, **keywords):
        program = _program_of(args)
        flags = keywords.get("creationflags") or 0
        if program in _OPENER_PROGRAMS or (os.name == "nt" and flags & _CREATE_NEW_CONSOLE):
            self._child_created = False  # so that __del__ has nothing to wait for
            _record("subprocess.Popen", args if program in _OPENER_PROGRAMS else f"{args!r} with CREATE_NEW_CONSOLE")
            raise DesktopOpenerError(f"a test may not start {program or 'a console'}, which opens a window")
        super().__init__(args, *positional, **keywords)


def _start_opener(command):
    _record("Desktop_Open", " ".join(str(word) for word in command))
    return True


_REAL_OPENER_START = None


def _guard_opener(module):
    global _REAL_OPENER_START
    if module._start is not _start_opener:
        _REAL_OPENER_START = module._start
        module._start = _start_opener


def real_opener_start():
    """Desktop_Open._start as it was before the trap, for the test of what it starts."""
    return _REAL_OPENER_START


class _GuardedLoader(importlib.abc.Loader):
    """Execute a module, then run its guard on it (replace QDesktopServices.openUrl, Desktop_Open._start)."""

    def __init__(self, loader, guard):
        self._loader = loader
        self._guard = guard

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        self._guard(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _GuardedFinder(importlib.abc.MetaPathFinder):
    """Guard a module when it is first imported, whenever that is: the call sites only name the module."""

    def __init__(self, guards):
        self._guards = guards  # module name -> function that replaces what opens something in it

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._guards:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _GuardedLoader(spec.loader, self._guards[fullname])
        return spec


def _guard_qt(module):
    services = getattr(module, "QDesktopServices", None)
    if services is not None:
        services.openUrl = staticmethod(_open_url)


# ----------------------------------------------------------------------------
# Failing the test that reached an opener


def _guard_results():
    original = unittest.TestResult.stopTest

    def stop_test(self, test):
        original(self, test)
        with _LOCK:
            reached = list(_UNEXPECTED)
            del _UNEXPECTED[:]
        if reached:
            message = "The test reached the desktop through code that opens things for the user:\n  " + "\n  ".join(reached)
            self.addFailure(test, (AssertionError, AssertionError(message), None))

    stop_test.guards_desktop = True
    unittest.TestResult.stopTest = stop_test


def _report_at_exit():
    reached = unexpected_calls()
    if reached:
        print("Desktop openers reached while no test was running:\n  " + "\n  ".join(reached),
              file=sys.stderr, flush=True)


def install():
    """Replace the openers in this process; calling it again does nothing."""
    if getattr(unittest.TestResult.stopTest, "guards_desktop", False):
        return
    webbrowser.open = _web_open
    webbrowser.open_new = _web_open_new
    webbrowser.open_new_tab = _web_open_new_tab
    if hasattr(os, "startfile"):
        os.startfile = _start_file
    subprocess.Popen = _GuardedPopen
    guards = {"PySide6.QtGui": _guard_qt, "utilities.Desktop_Open": _guard_opener}
    waiting = {}
    for name, guard in guards.items():
        loaded = sys.modules.get(name)
        if loaded is not None:
            guard(loaded)
        else:
            waiting[name] = guard
    if waiting:
        sys.meta_path.insert(0, _GuardedFinder(waiting))
    _guard_results()
    atexit.register(_report_at_exit)
