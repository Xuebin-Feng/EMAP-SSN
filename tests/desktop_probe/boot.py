# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The first thing the desktop probe runs in a process: decide whether to act, and if so, start the probe.

The .pth that probe.py installs imports the stub (stub.py) when the gate file exists, and the stub calls
``start`` here. This module acts only in a process that runs one of this checkout's entry points (ENTRY_POINTS).
It stays out of pip, uv, the test suite (the root test process marks itself and its children in
EMAPSSN_TEST_ENVIRONMENT, see tests/__init__.py) and every other script. It uses the standard library only; Qt
is not imported here. A thread waits for the program's own QApplication and hands over to runtime.py, which then
works in the GUI thread.

Nothing here may change what the program does, so a failure is written to a log and swallowed.
"""

from __future__ import annotations

import atexit
import importlib
import os
import sys
import threading
import time
import types

PACKAGE = "emapssn_desktop_probe"
TEST_ENVIRONMENT = "EMAPSSN_TEST_ENVIRONMENT"  # set by tests/__init__.py in the root test process and its children
KILL_SWITCH = "EMAPSSN_DESKTOP_PROBE"           # =0 keeps the probe out of a process, whatever the gate says

# The entry scripts of a checkout (path under the checkout, in lower case) and the kind the probe calls each.
ENTRY_POINTS = {
    "src/emapssn_config.py": "config",
    "src/emapssn_tools.py": "tools",
    "src/emapssn_viewer.py": "viewer",
    "src/layout_cache_generator.py": "layout",
    "src/emapssn_mcp_server.py": "mcp",
    "src/desktop/desktop_launcher.py": "splash",
    "src/desktop/desktop_launcher_monitor.py": "monitor",
}

# How long the thread that waits for QApplication sleeps between looks: short while a program starts, longer
# for a process that never makes one (the MCP server, the monitor).
POLL_SCHEDULE = ((30.0, 0.005), (300.0, 0.05), (float("inf"), 0.25))

_started = None


def load_package(directory=None):
    """Register this folder as the package ``emapssn_desktop_probe``, so that its modules import one another.

    The folder has no __init__.py on purpose (unittest discovery must not walk into it); a module with a
    __path__ is all a package needs. sys.path is not touched.
    """
    existing = sys.modules.get(PACKAGE)
    if existing is not None:
        return existing
    directory = directory or os.path.dirname(os.path.abspath(__file__))
    package = types.ModuleType(PACKAGE)
    package.__path__ = [directory]
    package.__file__ = os.path.join(directory, "__init__.py")
    sys.modules[PACKAGE] = package
    return package


def entry_kind(checkout, argv0):
    """The kind of ``argv0`` when it is an entry script of ``checkout``, else None."""
    if not argv0 or argv0 in ("-c", "-m", "-"):
        return None
    try:
        script = os.path.realpath(os.path.abspath(argv0))
        root = os.path.realpath(checkout)
        relative = os.path.relpath(script, root)
    except (OSError, ValueError):  # a script on another drive is not in the checkout
        return None
    return ENTRY_POINTS.get(relative.replace("\\", "/").lower())


def decide(checkout, argv, environ):
    """(kind, detail) when the probe should run in this process, else None. ``detail`` is the launcher's mode."""
    if environ.get(TEST_ENVIRONMENT) or environ.get(KILL_SWITCH) == "0":
        return None
    kind = entry_kind(checkout, argv[0] if argv else "")
    if kind is None:
        return None
    detail = argv[1] if kind == "splash" and len(argv) > 1 and not argv[1].startswith("-") else None
    if kind in ("config", "tools") and "--headless" in argv:
        # The MCP server runs Config and Tools without a window for exports and layouts, and starts a Viewer
        # as "EMAPSSN_Config.py --headless launch-viewer": Config's script then runs the Viewer's in-process.
        position = argv.index("--headless") + 1
        operation = argv[position] if position < len(argv) and not argv[position].startswith("-") else None
        detail = f"headless {operation}" if operation else "headless"
        if operation == "launch-viewer":
            kind = "viewer"
    return kind, detail


def _log_boot_failure(checkout, error):
    try:
        import traceback

        directory = os.path.join(checkout, "temp", "desktop_probe")
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "boot-errors.log"), "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] pid {os.getpid()} {sys.argv[:1]}\n")
            handle.write("".join(traceback.format_exception(type(error), error, error.__traceback__)) + "\n")
    except Exception:
        pass


class Boot:
    """The probe's start in one process: the work the program's main thread must not wait for happens in ``run``.

    The main thread decides (a few string comparisons), starts the thread and goes on. The thread imports the
    probe's modules, makes the recorder and its files, notes who started the process, and then waits for the
    program's QApplication. ``finish`` is what an exit calls: it waits for the recorder, which an exit that comes
    very early would otherwise beat.
    """

    def __init__(self, checkout, kind, detail, argv):
        self.checkout = str(checkout)
        self.kind = kind
        self.detail = detail
        self.argv = list(argv)
        self.pid = os.getpid()
        self.boot_wall_ms = int(time.time() * 1000)
        self.boot_mono_ms = round(time.monotonic() * 1000, 1)
        self.recorder = None
        self.ready = threading.Event()  # set when the recorder is made, or when it could not be

    def run(self):
        try:
            load_package(os.path.dirname(os.path.abspath(__file__)))
            from emapssn_desktop_probe import recorder, store

            gate = store.read_gate(self.checkout)
            if gate is None:  # the gate went away between the stub's look and now
                return
            self.recorder = recorder.Recorder(self.checkout, self.kind, gate, self.argv, detail=self.detail,
                                              boot_wall_ms=self.boot_wall_ms, boot_mono_ms=self.boot_mono_ms)
            self.recorder.prepare()
        except Exception as error:  # the program goes on without the probe
            _log_boot_failure(self.checkout, error)
            self.recorder = None
            return
        finally:
            self.ready.set()
        _wait_for_qt(self.recorder)

    def event(self, name, **fields):
        if self.recorder is not None and os.getpid() == self.pid:
            self.recorder.event(name, **fields)

    def finish(self, how):
        self.ready.wait(2.0)
        if self.recorder is not None:
            self.recorder.finish(how)


def start(checkout, argv=None, environ=None):
    """Start the probe in this process when it should run here; the Boot when it did, else None."""
    global _started
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    if _started is not None:
        return _started
    try:
        decision = decide(checkout, argv, environ)
        if decision is None:
            return None
        kind, detail = decision
        _started = Boot(checkout, kind, detail, argv)
        _install_exit_records(_started)
        threading.Thread(target=_started.run, name="desktop-probe-wait", daemon=True).start()
        return _started
    except Exception as error:  # the program goes on without the probe
        _log_boot_failure(str(checkout), error)
        return None


def _install_exit_records(boot):
    """Write the exit event on a normal exit and before os._exit, which skips the exit handlers."""
    atexit.register(boot.finish, "exit")
    original = os._exit

    def recorded_exit(code):
        if os.getpid() == boot.pid:  # a forked worker's os._exit is not this process's exit
            try:
                boot.ready.wait(2.0)
                boot.event("exit_code", code=code)
                boot.finish("os._exit")
            except Exception:
                pass
        original(code)

    os._exit = recorded_exit


def _wait_for_qt(recorder):
    """Wait, in the thread of its own, for the program's QApplication, then hand over to the runtime in the GUI thread.

    Only the module table and QCoreApplication.instance() are looked at here, and the thread ends as soon as it
    has handed over. It is a daemon, so it never delays the program's exit.
    """
    started = time.monotonic()
    try:
        # What the timeline needs to join the launches up, read here so the main thread does not pay for it.
        from emapssn_desktop_probe import store

        try:
            recorder.update_process(created=store.own_start_time(), ancestors=store.lineage())
            recorder.add_details()
        except Exception:
            pass  # only the timeline and the record need it
        while not (recorder.disabled or recorder.stopping):
            core = sys.modules.get("PySide6.QtCore")
            if core is not None:
                application = core.QCoreApplication.instance()
                if application is not None:
                    _hand_over(recorder, core, application)
                    return
            elapsed = time.monotonic() - started
            time.sleep(next(delay for limit, delay in POLL_SCHEDULE if elapsed < limit))
    except Exception as error:
        recorder.fail("waiting for Qt", error)


def _hand_over(recorder, core, application):
    """Import the runtime (here, in the thread) and have the GUI thread attach to the application."""
    runtime = importlib.import_module(PACKAGE + ".runtime")
    runtime.attach(recorder, core, application)
