# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Correlate detached command workers without owning their terminal lifetime."""
import json
import os
from pathlib import Path
import time
import uuid
import subprocess
import sys
import threading
import weakref
import psutil
from PySide6 import QtCore


def worker_directory():
    """The folder of the command workers' status files and logs (<project>/temp/command_workers)."""
    return Path(__file__).resolve().parent.parent / 'temp' / 'command_workers'


class WorkerTracker(QtCore.QObject):
    def __init__(self, context):
        super().__init__(context.portal)
        self.context = context
        self.job_id = 'worker-' + uuid.uuid4().hex
        self.path = worker_directory() / (self.job_id + '.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.created = time.monotonic()
        self.identity = None
        self.previous = None
        # (process, log path) of a worker that runs with no window of its own, else None.
        self.hidden = None
        self.log_offsets = {'stdout': 0, 'stderr': 0}
        context.add_job(self.job_id)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(250)

    def attach_hidden(self, process, log_path):
        """Record the windowless process that runs this job, and the log its output goes to.

        Nobody can close a window that does not exist, so a job that fails
        here takes its worker down with it, and its failures name the log in
        place of the terminal a person would have looked at.
        """
        self.hidden = (process, log_path)

    def fail(self, message):
        self.context.job_event(self.job_id, 'failed', message)
        self.timer.stop()
        if self.hidden is not None:
            end_hidden_worker(self.hidden[0])

    def poll(self):
        for stream in self.log_offsets:
            try:
                with Path(str(self.path) + '.' + stream).open('rb') as handle:
                    handle.seek(self.log_offsets[stream])
                    data = handle.read(32768)
                    self.log_offsets[stream] = handle.tell()
                self.context.portal.append_output(self.context.request_id, stream, data.decode('utf-8', errors='replace'))
            except OSError:
                pass
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            data = None
        if data is not None:
            if self.identity is None:
                self.identity = (data['pid'], data['created'])
            if (data['pid'], data['created']) != self.identity:
                self.fail('Worker identity changed unexpectedly')
                return
            if data != self.previous:
                self.previous = data
                self.context.job_event(self.job_id, data['status'], data.get('message'), data.get('artifacts', ()))
                for job in self.context.record['jobs']:
                    if job['job_id'] == self.job_id:
                        job['result'] = data.get('result', {})
            if data['status'] in {'succeeded', 'failed', 'cancelled'}:
                self.timer.stop()
                return
        if self.identity is not None:
            try:
                process = psutil.Process(self.identity[0])
                alive = process.create_time() == self.identity[1] and process.is_running()
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                note = '' if self.hidden is None else f'; its output is in {self.hidden[1]}'
                self.fail('Worker exited before reporting a final result' + note)
        elif self.hidden is not None and self.hidden[0].poll() is not None and not self.path.exists():
            # Typically an import error, whose traceback only the log holds.
            self.fail(f'Worker exited with code {self.hidden[0].returncode} before reporting startup; '
                      f'its output is in {self.hidden[1]}')
        elif time.monotonic() - self.created > 60:
            where = 'its terminal' if self.hidden is None else f'its log {self.hidden[1]}'
            self.fail(f'Worker did not report startup within 60 seconds; inspect {where} before retrying')


# The Python command scripts this process runs, until each ends.
_running_scripts = set()
_running_lock = threading.Lock()
_quit_hooked = False


def _end_process_tree(pid, timeout=2.0):
    """End the process pid and every process it started.

    The tree is read while its root still runs: on Windows a venv's python.exe
    runs the real interpreter as its child, and a process whose parent has
    gone can no longer be found from the root.
    """
    try:
        root = psutil.Process(pid)
        targets = [root, *root.children(recursive=True)]
    except psutil.Error:
        return
    for target in targets:
        try:
            target.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(targets, timeout=timeout)
    for target in alive:
        try:
            target.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=timeout)


def end_hidden_worker(process):
    """End the windowless worker process and every process it started.

    Returns whether it was still running. It is the Popen that is asked, not a
    bare PID: a worker that has ended may have left its PID to another process.
    """
    if process.poll() is not None:
        return False
    _end_process_tree(process.pid)
    return True


# The windowless workers this process started, until each ends or is stopped.
_hidden_workers = set()
_hidden_lock = threading.Lock()
_hidden_quit_hooked = False


def track_hidden_worker(process):
    """End process, and every process it started, when the application quits.

    A worker with no window can't be closed by anyone, so unless something
    stops it, closing the Viewer would leave it running unseen.
    """
    global _hidden_quit_hooked
    with _hidden_lock:
        _hidden_workers.difference_update([worker for worker in _hidden_workers if worker.poll() is not None])
        _hidden_workers.add(process)
    application = QtCore.QCoreApplication.instance()
    if application is not None and not _hidden_quit_hooked:
        application.aboutToQuit.connect(stop_hidden_workers)
        _hidden_quit_hooked = True


def stop_hidden_workers():
    """End every windowless worker this process started; return how many were running."""
    with _hidden_lock:
        workers = list(_hidden_workers)
        _hidden_workers.clear()
    return sum(end_hidden_worker(worker) for worker in workers)


class CommandScript:
    """A Python command script running on a daemon thread, which stop() ends.

    When it ends, finished.emit gets the script's CompletedProcess, or the
    exception that kept it from running. A script stopped by stop() has
    stopped set, and its result means nothing.
    """

    def __init__(self, path, finished):
        self.stopped = False
        self._process = None
        self._done = False
        self._lock = threading.Lock()
        with _running_lock:
            _running_scripts.add(self)
        threading.Thread(
            target=self._work, args=(os.path.abspath(path), finished),
            name='Viewer-command-script', daemon=True,
        ).start()

    def _work(self, path, finished):
        # Decoded as UTF-8, so the child must print UTF-8, not the Windows ANSI code page.
        environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        try:
            with self._lock:
                if self.stopped:
                    raise RuntimeError('The Python script was stopped before it started.')
                # No stdin: a script calling input() would otherwise wait on the Viewer's own.
                self._process = subprocess.Popen(
                    [sys.executable, path], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace',
                    env=environment, cwd=os.path.dirname(path))
            stdout, stderr = self._process.communicate()
            result = subprocess.CompletedProcess(self._process.args, self._process.returncode, stdout, stderr)
        except Exception as error:
            result = error
        finally:
            with self._lock:
                self._done = True
            with _running_lock:
                _running_scripts.discard(self)
        try:
            finished.emit(result)
        except RuntimeError:
            pass  # The receiving Qt object is already deleted; nothing is left to hand the result to.

    def stop(self):
        """End the script and every process it started; return whether it was still running."""
        with self._lock:
            if self.stopped or self._done:
                return False
            self.stopped = True
            process = self._process
        if process is not None:
            _end_process_tree(process.pid)
        return True


def stop_command_scripts():
    """End every Python command script this process runs; return how many were running."""
    with _running_lock:
        scripts = list(_running_scripts)
    return sum(script.stop() for script in scripts)


def start_command_script(path, finished):
    """Run the Python command script at path on a daemon thread, and emit finished with its result.

    The result is the script's CompletedProcess, or the exception that kept it
    from running. The script runs in its own folder, so a relative path in it
    names a file beside the script wherever the Viewer was started; path is made
    absolute first, as the folder change would otherwise move it. The returned
    CommandScript stops it; so do `run stop` and the application quitting.
    """
    global _quit_hooked
    application = QtCore.QCoreApplication.instance()
    if application is not None and not _quit_hooked:
        application.aboutToQuit.connect(stop_command_scripts)
        _quit_hooked = True
    return CommandScript(path, finished)


class ScriptTracker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def __init__(self, context, path):
        super().__init__(context.portal)
        self.context = context
        self.job_id = 'script-' + uuid.uuid4().hex
        context.add_job(self.job_id, status_detail='Executing selected Python command script')
        self.finished.connect(self._finish, QtCore.Qt.ConnectionType.QueuedConnection)
        context.job_event(self.job_id, 'running')
        self._script = start_command_script(path, self.finished)

    def _finish(self, result):
        if self._script.stopped:
            self.context.job_event(self.job_id, 'cancelled', 'Python command script stopped')
            return
        if isinstance(result, Exception):
            self.context.job_event(self.job_id, 'failed', str(result))
            return
        for stream in ('stdout', 'stderr'):
            self.context.portal.append_output(self.context.request_id, stream, getattr(result, stream))
        if result.returncode:
            self.context.job_event(self.job_id, 'failed', f'Python command script exited with code {result.returncode}')
            return
        from Command_Engine import script_command
        commands = [script_command(line) for line in result.stdout.splitlines()]
        commands = [line for line in commands if line and line.split()[0].lower() != 'run']
        try:
            self.context.children(commands)
        except ValueError as error:
            self.context.job_event(self.job_id, 'failed', str(error))
            return
        self.context.job_event(self.job_id, 'succeeded', f'Prepared {len(commands)} child commands')


class TypedScriptRun(QtCore.QObject):
    """A Python command script that a typed `run` runs in the background.

    The Viewer stays usable while the script runs, since it only prints the
    commands to run. When it ends, a queued signal hands its result to the GUI
    thread, where on_finished(viewer, result) reports a failure or runs the
    commands. The Viewer's command_script_run names the run from its start until
    its commands have run, which is how a second run is refused.

    Nothing here keeps the Viewer alive, and nothing touches a Viewer that has
    closed: a run abandoned when the window closes or the application quits
    stops its script and drops its result, and the thread is a daemon. A
    script `run stop` ends leaves nothing to report either.
    """
    finished = QtCore.Signal(object)

    def __init__(self, viewer, path, on_finished):
        super().__init__()
        self._viewer = weakref.ref(viewer)
        self._on_finished = on_finished
        self._abandoned = False
        self.finished.connect(self._finish, QtCore.Qt.ConnectionType.QueuedConnection)
        application = QtCore.QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.abandon)
        # The canvas of a stand-in Viewer has no events. vispy keeps a bound method weakly.
        close = getattr(getattr(getattr(viewer, 'canvas', None), 'events', None), 'close', None)
        if close is not None:
            close.connect(self.abandon)
        viewer.command_script_run = self
        self._script = start_command_script(path, self.finished)

    def abandon(self, event=None):
        """Stop the script and drop its result: the Viewer is closing."""
        self._abandoned = True
        self._script.stop()

    def _finish(self, result):
        viewer = self._viewer()
        if viewer is None:
            return
        if self._abandoned or self._script.stopped:
            self._release(viewer)
            return
        if getattr(viewer, '_command_dispatch_active', False):
            # Another command is mid-dispatch, and its event processing delivered this signal. Its
            # outcome and the Viewer's state are not ours to touch: wait for it to end, as typed input does.
            QtCore.QTimer.singleShot(20, lambda: self._finish(result))
            return
        from Viewer_Command_Portal import bind
        # As Command_Engine.execute_command does: input typed while the commands run, and the command
        # portal's next request, wait until they have run.
        viewer._command_dispatch_active = True
        try:
            with bind(None):
                self._on_finished(viewer, result)
        finally:
            viewer._command_dispatch_active = False
            self._release(viewer)

    def _release(self, viewer):
        if getattr(viewer, 'command_script_run', None) is self:
            viewer.command_script_run = None
