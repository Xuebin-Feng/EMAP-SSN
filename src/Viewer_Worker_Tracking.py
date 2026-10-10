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


class WorkerTracker(QtCore.QObject):
    def __init__(self, context):
        super().__init__(context.portal)
        self.context = context
        self.job_id = 'worker-' + uuid.uuid4().hex
        self.path = Path(__file__).resolve().parent.parent / 'temp' / 'command_workers' / (self.job_id + '.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.created = time.monotonic()
        self.identity = None
        self.previous = None
        self.log_offsets = {'stdout': 0, 'stderr': 0}
        context.add_job(self.job_id)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(250)

    def fail(self, message):
        self.context.job_event(self.job_id, 'failed', message)
        self.timer.stop()

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
                self.fail('Worker exited before reporting a final result')
        elif time.monotonic() - self.created > 60:
            self.fail('Worker did not report startup within 60 seconds; inspect its terminal before retrying')


def start_command_script(path, finished):
    """Run the Python command script at path on a daemon thread, and emit finished with its result.

    The result is the script's CompletedProcess, or the exception that kept it
    from running. The script runs in its own folder, so a relative path in it
    names a file beside the script wherever the Viewer was started; path is made
    absolute first, as the folder change would otherwise move it.
    """
    path = os.path.abspath(path)
    # Decoded as UTF-8, so the child must print UTF-8, not the Windows ANSI code page.
    environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    def work():
        try:
            # No stdin: a script calling input() would otherwise wait on the Viewer's own.
            result = subprocess.run([sys.executable, path], stdin=subprocess.DEVNULL, capture_output=True,
                text=True, encoding='utf-8', errors='replace', env=environment, cwd=os.path.dirname(path))
        except Exception as error:
            result = error
        try:
            finished.emit(result)
        except RuntimeError:
            pass  # The receiving Qt object is already deleted; nothing is left to hand the result to.
    threading.Thread(target=work, name='Viewer-command-script', daemon=True).start()


class ScriptTracker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def __init__(self, context, path):
        super().__init__(context.portal)
        self.context = context
        self.job_id = 'script-' + uuid.uuid4().hex
        context.add_job(self.job_id, status_detail='Executing selected Python command script')
        self.finished.connect(self._finish, QtCore.Qt.ConnectionType.QueuedConnection)
        context.job_event(self.job_id, 'running')
        start_command_script(path, self.finished)

    def _finish(self, result):
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
    drops its result, and the thread is a daemon.
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
        start_command_script(path, self.finished)

    def abandon(self, event=None):
        """Drop the script's result: the Viewer is closing."""
        self._abandoned = True

    def _finish(self, result):
        viewer = self._viewer()
        if viewer is None:
            return
        if self._abandoned:
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
