# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""The first stage of a Windows desktop launch: the splash, the dependency check, and the hand-off to the monitor.

    python -u src\\desktop\\Desktop_Launcher.py viewer|tools [--console]
    python -u src\\desktop\\Desktop_Launcher.py session [--settings FILE --delete-settings]

EMAP-SSN.exe (src/bin/Launcher.cs, built by install.bat) starts it with a hidden console. "viewer"
is the EMAP-SSN Configuration window, as in EMAPSSN_Desktop_Launcher.bat, and "tools" is Tools. It

1. brings the window forward and ends if one is already open for this checkout (no splash);
2. ends quietly if another launch of the same window is under way (a launcher lock), so a double-click
   starts one window;
3. shows the splash (desktop/Splash.py), within about a third of a second of the click;
4. checks the environment (Install_Dependencies.py --check-only) and shows the lines it prints, live, on
   the splash. If the environment needs setup or repair it hands over to the old visible console
   (EMAPSSN_Desktop_Launcher.bat), which does that, as before;
5. starts desktop/Desktop_Launcher_Monitor.py detached, which runs the window with a hidden console and
   keeps its output in application.log, and follows what the window reports (utilities/Startup_Status.py);
6. closes the splash when the window is ready (gui.ready), unless the user has opened its details: then it
   says Ready and stays until Close is pressed. If the window ends first, it opens them and turns into
   an error view that shows the log's last lines.

"session" is a Viewer that Config starts: no check and no lock (Config checked, and many Viewers may be
open), the same splash and monitor, and the arguments after it go to the Viewer. The monitor reports a
crash after the window opened in a dialog, since no terminal is open to show it.

--console runs the old visible-console launch in this console, with no splash, exactly as the batch file
does. Imports only PySide6 and the standard library, so it is up before the heavy modules load.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

T_START = time.time()

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import QCoreApplication, QLockFile, QObject, QProcess, QProcessEnvironment, QTimer, Qt, Signal
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtWidgets import QApplication

from desktop import Desktop_Launcher_Monitor as Monitor
from desktop import Splash as SplashModule
from desktop.Splash import SplashWindow, StatusFeed
from utilities import Startup_Status as Status

# The launcher's kinds, as the batch file names them ("viewer" is Config), and what each is called elsewhere.
APPLICATION_IDS = {"viewer": "SSN_Config", "tools": "SSN_Tools"}  # the single-instance channels
SPLASH_KINDS = {"viewer": "config", "tools": "tools", "session": "viewer"}
KINDS = tuple(SPLASH_KINDS)
BATCH_FILE = Path("src") / "bin" / "EMAPSSN_Desktop_Launcher.bat"
INSTALLER = Path("src") / "Install_Dependencies.py"
VENV_PYTHON = Path(".venv") / "Scripts" / "python.exe"
REQUIRED_PYTHON = (3, 13)
STARTUP_TIMEOUT_SECONDS = Monitor.STARTUP_TIMEOUT_SECONDS
POLL_MS = 30
MARKS_FILE_VARIABLE = "SSN_LAUNCH_MARKS_FILE"  # a file that gets the launch's timings as JSON when it ends
HIDDEN_VARIABLE = "SSN_SPLASH_HIDDEN"          # 1: draw the splash nowhere (the timing runs and the tests)
EXIT_FALLBACK = 3  # the launcher failed before it started anything: the stub falls back to the visible console

CREATE_NEW_CONSOLE = 0x00000010
CREATE_NO_WINDOW = 0x08000000

# What the splash says while the launcher itself works.
CHECKED_TEXT = Status.QT_TRANSLATE_NOOP("Splash", "Starting…")


# ---------------------------------------------------------------------------------------------
# Plain helpers (no Qt)
# ---------------------------------------------------------------------------------------------

def single_instance_server_name(application_id, checkout_root, home=None):
    """The name of the single-instance channel of a window, as desktop.Desktop_App gives it (per user and checkout).

    tests/test_desktop_launcher.py compares the two: this module does not import Desktop_App, which
    would cost a tenth of a second before the splash shows.
    """
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(application_id)).strip("-")
    identity = "\n".join(
        os.path.normcase(str(Path(path).resolve())) for path in (Path.home() if home is None else home, checkout_root)
    )
    return f"ssn-{normalized}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]}"


def activate_existing_instance(kind, root, timeout_ms=300):
    """Ask the window of this checkout that is already open to come forward; True if there is one."""
    if kind not in APPLICATION_IDS:
        return False
    socket = QLocalSocket()
    socket.connectToServer(single_instance_server_name(APPLICATION_IDS[kind], root))
    if not socket.waitForConnected(timeout_ms):
        socket.abort()
        return False
    socket.write(b"activate\n")
    socket.flush()
    socket.waitForBytesWritten(timeout_ms)
    socket.disconnectFromServer()
    return True


def venv_python(root):
    return Path(root) / VENV_PYTHON


def find_uv(environment=None):
    """The uv executable, as the batch file looks for it (on PATH, else ~/.local/bin), or None."""
    environment = os.environ if environment is None else environment
    found = shutil.which("uv", path=environment.get("PATH"))
    if found:
        return found
    candidate = Path(environment.get("USERPROFILE") or Path.home()) / ".local" / "bin" / "uv.exe"
    return str(candidate) if candidate.is_file() else None


def check_command(root, python, uv):
    """The dependency check, as EMAPSSN.bat runs it (-u, so its lines reach the splash as they are printed)."""
    return [str(python), "-u", str(Path(root) / INSTALLER), "--check-only", "--uv-executable", str(uv),
            "--venv", str(Path(root) / ".venv")]


def visible_console_command(root, kind):
    """The old launch: the batch file in a console of its own, which does any setup and shows what it does."""
    return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(Path(root) / BATCH_FILE), kind]


def open_visible_console(root, kind, *, popen=subprocess.Popen):
    """Start the old visible-console launch (a new console window on Windows)."""
    flags = CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    return popen(visible_console_command(root, kind), cwd=str(root), creationflags=flags) if flags else \
        popen(visible_console_command(root, kind), cwd=str(root))


def lock_path(root, kind):
    return Path(root) / "temp" / f"launch_{kind}.lock"


def acquire_launch_lock(root, kind):
    """Take the lock of this window's launches; returns the QLockFile, or None if another launch holds it.

    A lock whose owner has gone (its process ended) is taken over; there is no time limit on a live one.
    """
    path = lock_path(root, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(path))
    lock.setStaleLockTime(0)
    return lock if lock.tryLock(0) else None


def new_state_dir(root, kind):
    """A folder of this launch's own under temp/, as the batch file makes: <kind>_<token>."""
    state = Path(root) / "temp" / f"{kind}_{uuid.uuid4().hex}"
    state.mkdir(parents=True, exist_ok=True)
    return state


def read_exit_code(state_dir):
    try:
        return int((Path(state_dir) / "application.exit").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------------------------

def _guarded(method):
    """Run a step of the launch; if it raises, end the launch rather than leave the event loop waiting for ever.

    An exception in a slot never reaches main(), so nothing else would end a launch that broke. If nothing
    was started yet (no monitor), the stub is asked for the visible console (EXIT_FALLBACK), which shows what
    is wrong; once the monitor runs, the launch is over, and 0 says it was handled.
    """

    @functools.wraps(method)
    def run(self, *arguments):
        try:
            return method(self, *arguments)
        except Exception:
            import traceback

            traceback.print_exc()
            self.marks["exception"] = traceback.format_exc()[-600:]
            self.finish(0 if self.started_anything else EXIT_FALLBACK)

    return run


class Launch(QObject):
    """One launch: runs on the event loop, and ``finished(code)`` ends it."""

    finished = Signal(int)

    def __init__(self, kind, root, *, python=None, uv=None, app_args=(), show=True, check=None, timeout=None,
                 popen=subprocess.Popen, environment=None):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown launch kind {kind!r}")
        self.kind = kind
        self.root = Path(root)
        self.python = str(python or venv_python(root))
        self.uv = uv
        self.app_args = tuple(app_args)
        self.show = show
        self.check = (kind != "session") if check is None else check
        self.timeout = STARTUP_TIMEOUT_SECONDS if timeout is None else timeout
        self.popen = popen
        self.environment = environment
        self.marks = {}
        self.splash = None
        self.lock = None
        self.state_dir = None
        self.monitor = None
        self.check_process = None
        self._check_buffer = bytearray()
        self.check_lines = []
        self._deadline = None
        self._done = False
        self.started_anything = False
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_start)

    def mark(self, name):
        """Note the time of a step: milliseconds since this process started; marks["start_epoch"] is when that was."""
        self.marks.setdefault("start_epoch", T_START)
        self.marks[name] = round((time.time() - T_START) * 1000)

    # --- the steps
    @_guarded
    def begin(self):
        """Steps 1 to 3 (they end the launch quietly when there is nothing to do), then the check."""
        self.mark("begin")
        if self.kind in APPLICATION_IDS:
            if activate_existing_instance(self.kind, self.root):
                self.mark("activated_existing")
                return self.finish(0)
            self.lock = acquire_launch_lock(self.root, self.kind)
            if self.lock is None:
                self.mark("launch_in_progress")
                return self.finish(0)
        self.open_splash()
        if self.check:
            return self.start_check()
        return self.start_monitor()

    def open_splash(self):
        language = SplashModule.splash_language()
        SplashModule.load_fonts(language)
        self._translator = SplashModule.install_catalog(language)
        self.splash = SplashWindow(SPLASH_KINDS[self.kind], language)
        self.splash.closed.connect(self._splash_closed)
        self.splash.set_status(CHECKED_TEXT if not self.check else Status.CHECKING_ENVIRONMENT,
                               SplashModule.FRACTION_CHECKING)
        if not self.show:
            self.splash.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        self.splash.show()
        QApplication.processEvents()
        self.mark("splash_shown")

    def needs_setup(self):
        """Why the environment cannot be used as it is, or None: the old console does the setup."""
        if not Path(self.python).is_file():
            return "the managed Python is missing"
        if sys.version_info[:2] != REQUIRED_PYTHON and Path(self.python).resolve() == Path(sys.executable).resolve():
            return "the managed Python is not version 3.13"
        if not self.uv:
            self.uv = find_uv(self.environment)
        if not self.uv:
            return "uv is not installed"
        return None

    def start_check(self):
        reason = self.needs_setup()
        if reason:
            return self.hand_over_to_console(reason)
        process = QProcess(self)
        process.setProgram(self.python)
        command = check_command(self.root, self.python, self.uv)
        process.setArguments(command[1:])
        process.setWorkingDirectory(str(self.root))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.setProcessEnvironment(self._environment())
        if hasattr(process, "setCreateProcessArgumentsModifier") and sys.platform == "win32":
            process.setCreateProcessArgumentsModifier(_no_window)
        process.readyReadStandardOutput.connect(self._check_output)
        process.finished.connect(self._check_finished)
        process.errorOccurred.connect(self._check_failed)
        self.check_process = process
        process.start()
        self.mark("check_started")

    def _environment(self):
        environment = QProcessEnvironment.systemEnvironment()
        for name in Monitor._MANAGED_ENVIRONMENT_PATH_OVERRIDES:
            environment.remove(name)
        return environment

    @_guarded
    def _check_output(self):
        self._check_buffer.extend(bytes(self.check_process.readAllStandardOutput()))
        while b"\n" in self._check_buffer:
            line, _, rest = bytes(self._check_buffer).partition(b"\n")
            self._check_buffer[:] = rest
            text = line.decode("utf-8", "replace").strip()
            if text:
                self.check_lines.append(text)
                fraction = min(SplashModule.FRACTION_CHECKED - 0.04,
                               SplashModule.FRACTION_CHECKING + 0.03 * len(self.check_lines))
                self.splash.show_text(text, fraction)
                self.splash.add_output(text)  # and the section Show details opens

    @_guarded
    def _check_failed(self, error):
        # A check that starts and then crashes also ends in _check_finished; one that cannot start ends here.
        if error == QProcess.ProcessError.FailedToStart and not self._done:
            self.hand_over_to_console("the environment check could not run")

    @_guarded
    def _check_finished(self, code, status):
        self.mark("check_finished")
        self._check_output()
        if self._done:
            return
        if code != 0 or status != QProcess.ExitStatus.NormalExit:
            return self.hand_over_to_console("the environment needs setup or repair")
        # Another launch may have opened the window while this one checked.
        if self.kind in APPLICATION_IDS and activate_existing_instance(self.kind, self.root):
            self.mark("activated_existing")
            return self.finish(0)
        self.splash.set_status(CHECKED_TEXT, SplashModule.FRACTION_CHECKED)
        self.start_monitor()

    def hand_over_to_console(self, reason):
        """The environment cannot be used as it is: the old launch, in a visible console, repairs it and starts the window."""
        self.mark("handed_over_to_console")
        self.marks["console_reason"] = reason
        try:
            open_visible_console(self.root, self.kind, popen=self.popen)
        except OSError as error:
            self.marks["console_failed"] = str(error)
            return self.finish(EXIT_FALLBACK)
        self.finish(0)

    def start_monitor(self):
        self.state_dir = new_state_dir(self.root, self.kind)
        self.feed = StatusFeed(self.state_dir / "startup.status")
        self.splash.feed = self.feed
        self.splash.timer.start()
        # What the window prints is in application.log, which the monitor makes and the window's capture writes
        # as it prints: the splash follows that file into its section.
        self.splash.follow_log(self.state_dir / "application.log")
        try:
            self.monitor = Monitor.launch_detached_monitor(
                self.kind, self.state_dir, failure_report="dialog", app_args=self.app_args,
                project_root=self.root, python=self.python,
            )
        except (OSError, ValueError) as error:
            self.started_anything = False
            self.splash.show_error([str(error)], None, reason=QCoreApplication.translate(
                "Splash", "The window could not be started."))
            return
        self.started_anything = True
        self.mark("monitor_started")
        self._deadline = time.monotonic() + self.timeout
        self._poll.start(POLL_MS)

    @_guarded
    def _poll_start(self):
        if self._done or self.splash is None or self.splash.error:
            return
        ready = self.state_dir / "gui.ready"
        if ready.is_file():
            return self.window_ready()
        code = read_exit_code(self.state_dir)
        if code is not None:
            return self.window_ended(code)
        if self.monitor is not None and self.monitor.poll() is not None:
            # The monitor ended without the window's exit file: it failed, or was stopped.
            code = read_exit_code(self.state_dir)
            if code is None and not ready.is_file():
                return self.window_ended(Monitor.STARTUP_LAUNCH_FAILED, monitor_failed=True)
        if time.monotonic() > self._deadline:
            self.show_failure(Monitor.STARTUP_TIMEOUT, QCoreApplication.translate(
                "Splash", "{window} did not open a window within {minutes} minutes.").format(
                window=self._window_name(), minutes=int(self.timeout // 60)))

    def _window_name(self):
        return f"{SplashModule.PRODUCT} {SplashModule.tr(SplashModule.SUBTITLES[SPLASH_KINDS[self.kind]])}"

    def window_ready(self):
        self.mark("gui_ready")
        try:  # the monitor waits for this before it reports a crash, as it waits for the batch file's
            (self.state_dir / "terminal.dismissed").write_text("dismissed\n", encoding="utf-8")
        except OSError:
            pass
        self._poll.stop()
        if self.splash.details_open:
            self.mark("ready_kept_open")
            self.splash.show_ready()  # the user is watching: it stays until Close is pressed
        else:
            self.splash.close()

    def window_ended(self, code, monitor_failed=False):
        self.mark("window_ended_before_ready")
        self.marks["exit_code"] = code
        if code == 0 and not monitor_failed:
            self.splash.close()  # it handed over to a window that was already open, or has nothing to show
            return
        self.show_failure(code)

    def show_failure(self, code, reason=None):
        """The error view: the section opens by itself, with why the window did not open and what it printed."""
        self.marks["failure_code"] = code
        log = None if self.state_dir is None else self.state_dir / "application.log"
        self.splash.sync_log()  # what it printed last
        self.splash.show_error(None, log, reason=reason)
        self._poll.stop()

    def _splash_closed(self):
        # Whether the window opened or the error view was read, this launch handled what happened: 0.
        self.finish(0)

    def finish(self, code):
        if self._done:
            return code
        self._done = True
        self._poll.stop()
        self.mark("finished")
        if self.splash is not None:
            self.splash.timer.stop()
            self.splash.hide()
        if self.lock is not None:
            self.lock.unlock()
            self.lock = None
        marks = os.environ.get(MARKS_FILE_VARIABLE)
        if marks:
            try:
                Path(marks).write_text(json.dumps({**self.marks, "check_lines": self.check_lines}, ensure_ascii=False),
                                       encoding="utf-8")
            except OSError:
                pass
        self.finished.emit(code)
        return code


def _no_window(arguments):
    arguments.flags |= CREATE_NO_WINDOW


def run_console(root, kind):
    """--console: the old launch, in this console, with no splash."""
    return subprocess.call(visible_console_command(root, kind), cwd=str(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("--console", action="store_true", help="the old visible-console launch, with no splash")
    parser.add_argument("--root", default=str(SRC_DIR.parent), help="the project folder (tests)")
    parser.add_argument("--python", default=None, help="the interpreter that runs the window (tests)")
    parser.add_argument("--hidden", action="store_true", help="draw the splash nowhere (tests)")
    # What this does not know goes to the window: a session's --settings FILE --delete-settings are the Viewer's.
    args, window_arguments = parser.parse_known_args(argv)
    args.app_args = window_arguments
    if args.console:
        return run_console(args.root, args.kind)
    if args.kind != "session" and args.app_args:
        parser.error(f"unrecognized arguments: {' '.join(args.app_args)} (only session takes arguments for the window)")
    application = QApplication(sys.argv[:1])
    hidden = args.hidden or os.environ.get(HIDDEN_VARIABLE) == "1"
    launch = Launch(args.kind, args.root, python=args.python, app_args=args.app_args, show=not hidden)
    result = []
    launch.finished.connect(lambda code: (result.append(code), application.quit()))
    QTimer.singleShot(0, launch.begin)
    application.exec()
    return result[0] if result else EXIT_FALLBACK


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit:
        raise
    except Exception:  # nothing started yet that anyone could see: let the stub open the visible console
        import traceback

        traceback.print_exc()
        code = EXIT_FALLBACK
    raise SystemExit(code)
