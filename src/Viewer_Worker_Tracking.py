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


class ScriptTracker(QtCore.QObject):
    finished = QtCore.Signal(object)

    def __init__(self, context, path):
        super().__init__(context.portal)
        self.context = context
        self.job_id = 'script-' + uuid.uuid4().hex
        context.add_job(self.job_id, status_detail='Executing selected Python command script')
        self.finished.connect(self._finish, QtCore.Qt.ConnectionType.QueuedConnection)
        def work():
            try:
                result = subprocess.run([sys.executable, path], capture_output=True,
                    text=True, encoding='utf-8', errors='replace')
            except Exception as error:
                result = error
            self.finished.emit(result)
        context.job_event(self.job_id, 'running')
        threading.Thread(target=work, name='Viewer-command-script', daemon=True).start()

    def _finish(self, result):
        if isinstance(result, Exception):
            self.context.job_event(self.job_id, 'failed', str(result))
            return
        for stream in ('stdout', 'stderr'):
            self.context.portal.append_output(self.context.request_id, stream, getattr(result, stream))
        if result.returncode:
            self.context.job_event(self.job_id, 'failed', f'Python command script exited with code {result.returncode}')
            return
        commands = [line.split('//')[0].strip() for line in result.stdout.splitlines()]
        commands = [line for line in commands if line and line.split()[0].lower() != 'run']
        try:
            self.context.children(commands)
        except ValueError as error:
            self.context.job_event(self.job_id, 'failed', str(error))
            return
        self.context.job_event(self.job_id, 'succeeded', f'Prepared {len(commands)} child commands')
