# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""Viewer-owned command execution. No MCP or model dependency."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import uuid

from PySide6 import QtCore

TERMINAL = {'succeeded', 'failed', 'cancelled', 'skipped'}
CURRENT = ContextVar('viewer_command_context', default=None)


def now():
    return datetime.now(timezone.utc).isoformat()


class CommandInteractionError(RuntimeError):
    pass


class _Tee:
    """Permanent pass-through; only explicitly bound execution contexts capture."""
    def __init__(self, stream, name):
        self.stream, self.name = stream, name

    def write(self, text):
        result = self.stream.write(text)
        context = CURRENT.get()
        if context is not None:
            context.portal.append_output(context.request_id, self.name, text)
        return result

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def install_output_capture():
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if not isinstance(stream, _Tee):
            setattr(sys, name, _Tee(stream, name))


@contextmanager
def bind(context):
    token = CURRENT.set(context)
    try:
        yield
    finally:
        CURRENT.reset(token)


def report(status=None, message=None, artifact=None, viewer=None):
    context = CURRENT.get()
    if context is not None and (viewer is None or viewer is context.portal.viewer):
        context.report(status, message, artifact)


@contextmanager
def user_interaction(viewer, description):
    context = CURRENT.get()
    if context is not None and context.portal.viewer is viewer:
        if os.environ.get('SSN_VIEWER_HEADLESS') == '1' or os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
            raise CommandInteractionError(f'{description} requires a visible Viewer and user input.')
        context.record['status'] = 'awaiting_user_input'
        context.portal.requests[context.request_id]['status'] = 'awaiting_user_input'
        context.report(message=description)
        context.portal.publish(context.request_id)
    try:
        yield
    finally:
        if context is not None and context.portal.viewer is viewer:
            context.record['status'] = 'running'
            context.portal.requests[context.request_id]['status'] = 'running'
            context.portal.publish(context.request_id)


class ExecutionContext:
    def __init__(self, portal, request_id, record):
        self.portal, self.request_id, self.record = portal, request_id, record

    def report(self, status=None, message=None, artifact=None):
        if status is not None and self.record.get('outcome') not in {'failed', 'cancelled'}:
            self.record['outcome'] = status
        if message:
            messages = self.record['messages']
            raw = str(message).encode('utf-8')
            text = raw[:2048].decode('utf-8', errors='replace')
            # A handler may print a status immediately before declaring success.
            # Promote that informational entry instead of repeating the summary.
            if (status == 'succeeded' and messages and
                    messages[-1]['status'] is None and
                    messages[-1]['text'] == text and not messages[-1]['truncated'] and
                    len(raw) <= 2048):
                messages[-1]['status'] = status
            elif len(messages) < 100 and sum(len(m['text'].encode('utf-8')) for m in messages) < 8192:
                messages.append({'status': status, 'text': raw[:2048].decode('utf-8', errors='replace'), 'truncated': len(raw) > 2048})
            else:
                self.record['messages_truncated'] = True
        if artifact:
            path = os.path.abspath(os.fspath(artifact))
            if path not in self.record['artifacts']:
                self.record['artifacts'].append(path)

    def add_job(self, job_id, **extra):
        job = dict(job_id=str(job_id), status='queued', **extra)
        self.record['jobs'].append(job)
        return job

    def job_event(self, job_id, status, message=None, artifacts=()):
        for job in self.record['jobs']:
            if job['job_id'] == str(job_id):
                job['status'] = status
                if message:
                    job['message'] = str(message)[:1024]
                    self.portal.append_output(self.request_id, 'stderr' if status == 'failed' else 'stdout', str(message) + '\n')
                for artifact in artifacts:
                    self.report(artifact=artifact)
                if status in {'failed', 'cancelled'}:
                    self.report(status, message)
                break
        self.portal.changed.emit(self.request_id)

    def children(self, commands):
        if len(commands) > 1000 or any(not isinstance(c, str) or not c.strip() or len(c) > 8192 or '\n' in c or '\r' in c for c in commands):
            raise ValueError('A command script may emit at most 1000 nonempty single-line commands, each at most 8192 characters')
        self.record['children'].extend(self.portal.new_command(c) for c in commands)


class ViewerCommandPortal(QtCore.QObject):
    changed = QtCore.Signal(str)
    completed = QtCore.Signal(str)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.requests = OrderedDict()
        self.submissions = {}
        self.queue = []
        self.output = {}
        self.output_lock = threading.RLock()
        self.max_output_bytes = 16 * 1024**2
        self.history_limit = 100
        self._busy = False
        self._closed = False
        self.changed.connect(self._changed, QtCore.Qt.ConnectionType.QueuedConnection)
        install_output_capture()

    @staticmethod
    def new_command(command):
        return dict(command_id=uuid.uuid4().hex, command=command, status='queued',
                    submitted_at=now(), started_at=None, finished_at=None, outcome=None,
                    messages=[], artifacts=[], jobs=[], children=[], dispatched=False)

    def submit(self, submission_id, commands, source='mcp'):
        if self._closed:
            raise ValueError('Viewer command portal is shutting down.')
        if not isinstance(submission_id, str) or not 1 <= len(submission_id) <= 128:
            raise ValueError('submission_id must be a client-generated string of 1..128 characters')
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
            raise ValueError('commands must contain 1..100 command strings')
        if any(not isinstance(c, str) or not c.strip() or '\n' in c or '\r' in c or len(c) > 8192 for c in commands):
            raise ValueError('Each command must be a nonempty single line of at most 8192 characters')
        signature = hashlib.sha256(json.dumps([commands, source], ensure_ascii=False).encode()).hexdigest()
        if submission_id in self.submissions:
            old_signature, request_id = self.submissions[submission_id]
            if old_signature != signature:
                raise ValueError('submission_id already used with a different payload')
            if request_id not in self.requests:
                raise ValueError('Submission already executed; its result was evicted. It will not be executed again.')
            return self.get(request_id)
        if len(self.queue) >= 100:
            raise ValueError('Viewer command queue is full; retry later with the same submission_id')
        request_id = uuid.uuid4().hex
        self.requests[request_id] = dict(request_id=request_id, submission_id=submission_id,
            session_id=getattr(self.viewer, 'inspection_session_id', None), source=source,
            status='queued', submitted_at=now(), finished_at=None,
            commands=[self.new_command(c.strip()) for c in commands])
        self.submissions[submission_id] = (signature, request_id)
        with self.output_lock:
            self.output[request_id] = {s: {'data': b'', 'base': 0} for s in ('stdout', 'stderr')}
        self.queue.append(request_id)
        self.publish(request_id)
        QtCore.QTimer.singleShot(0, self.pump)
        return self.get(request_id)

    def _advance(self, request_id, record):
        if record['status'] in TERMINAL:
            return True
        if not record['dispatched']:
            import Command_Engine
            record.update(status='running', started_at=now(), dispatched=True)
            context = ExecutionContext(self, request_id, record)
            with bind(context):
                try:
                    Command_Engine.execute_command(self.viewer, record['command'])
                except Exception as error:
                    context.report('failed', str(error))
            if record['outcome'] is None:
                context.report('failed', 'Command handler did not report an explicit outcome.')
            self.publish(request_id)
        if any(j['status'] not in TERMINAL for j in record['jobs']):
            record['status'] = ('awaiting_user_input' if any(j['status'] == 'awaiting_user_input' for j in record['jobs']) else 'running')
            return False
        for child in record['children']:
            if record['outcome'] in {'failed', 'cancelled'}:
                self._skip(child)
            elif not self._advance(request_id, child):
                return False
            elif child['status'] in {'failed', 'cancelled'}:
                record['outcome'] = child['status']
        record.update(status=record['outcome'], finished_at=now())
        return True

    @staticmethod
    def _skip(record):
        if record['status'] == 'queued':
            record.update(status='skipped', finished_at=now())

    def pump(self):
        if self._busy or self._closed or not self.queue or getattr(self.viewer, '_command_dispatch_active', False):
            if self.queue and not self._closed:
                QtCore.QTimer.singleShot(20, self.pump)
            return
        self._busy = True
        try:
            request_id = self.queue[0]
            request = self.requests[request_id]
            request['status'] = 'running'
            failure = None
            for record in request['commands']:
                if failure:
                    self._skip(record)
                elif not self._advance(request_id, record):
                    request['status'] = record['status']
                    self.publish(request_id)
                    return
                elif record['status'] in {'failed', 'cancelled'}:
                    failure = record['status']
            request.update(status=failure or 'succeeded', finished_at=now())
            self.queue.pop(0)
            self.publish(request_id)
            self.completed.emit(request_id)
            self._prune()
            QtCore.QTimer.singleShot(0, self.pump)
        finally:
            self._busy = False

    def _changed(self, request_id):
        self.publish(request_id)
        self.pump()

    def publish(self, request_id):
        request = self.requests.get(request_id)
        if request and hasattr(self.viewer, 'broadcast_event'):
            self.viewer.broadcast_event({'type': 'command_request_updated',
                'request_id': request_id, 'status': request['status'],
                'commands': [{'command': c['command'], 'status': c['status']} for c in request['commands']]})

    def _resolve_request(self, request_id=None, submission_id=None):
        if (request_id is None) == (submission_id is None):
            raise ValueError('Supply exactly one of request_id or submission_id')
        value = request_id if request_id is not None else submission_id
        if not isinstance(value, str) or not value.strip():
            raise ValueError('request_id or submission_id must be a nonempty string')
        if submission_id is not None:
            if submission_id not in self.submissions:
                raise ValueError('Unknown submission_id in this Viewer')
            request_id = self.submissions[submission_id][1]
            if request_id not in self.requests:
                raise ValueError('Submission is known but its result was evicted from this Viewer')
        if request_id not in self.requests:
            raise ValueError('Command request is unknown or evicted from this Viewer')
        return request_id

    def get(self, request_id=None, offset=0, limit=25, command_id=None, artifact_offset=0, artifact_limit=25, submission_id=None):
        request_id = self._resolve_request(request_id, submission_id)
        if offset < 0 or not 1 <= limit <= 100 or artifact_offset < 0 or not 1 <= artifact_limit <= 100:
            raise ValueError('Invalid command page bounds')
        request = self.requests[request_id]
        result = {k: deepcopy(v) for k, v in request.items() if k != 'commands'}
        records = []
        def flatten(commands, parent=None):
            for command in commands:
                records.append({k: deepcopy(v) for k, v in command.items() if k not in {'children', 'dispatched', 'outcome'}} | {'parent_command_id': parent})
                flatten(command['children'], command['command_id'])
        flatten(request['commands'])
        if command_id is not None:
            records = [r for r in records if r['command_id'] == command_id]
            if not records:
                raise ValueError('Unknown command_id for this request')
        for record in records:
            artifacts = record['artifacts']
            record.update(artifact_count=len(artifacts), artifacts=artifacts[artifact_offset:artifact_offset+artifact_limit],
                next_artifact_offset=min(len(artifacts), artifact_offset+artifact_limit), artifacts_complete=artifact_offset+artifact_limit >= len(artifacts))
        page = []
        for record in records[offset:offset + limit]:
            if len(json.dumps(page + [record], ensure_ascii=False).encode()) > 60000:
                if not page:
                    raise ValueError('Command record exceeds response budget; reduce artifact_limit and request a specific command_id')
                break
            page.append(record)
        end = offset + len(page)
        return result | {'commands': page, 'total_commands': len(records), 'next_offset': end,
                         'complete': end >= len(records)}

    def list_requests(self, offset=0, limit=25):
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError('Invalid request page bounds')
        records = [{k: r[k] for k in ('request_id', 'submission_id', 'source', 'status', 'submitted_at', 'finished_at')} for r in reversed(self.requests.values())]
        return {'requests': records[offset:offset + limit], 'total': len(records),
                'next_offset': min(offset + limit, len(records)), 'complete': offset + limit >= len(records)}

    def append_output(self, request_id, stream, text):
        with self.output_lock:
            if request_id not in self.output:
                return
            slot = self.output[request_id][stream]
            slot['data'] += text.encode('utf-8', errors='replace')
            total = sum(len(s['data']) for streams in self.output.values() for s in streams.values())
            excess = max(0, total - self.max_output_bytes)
            for streams in self.output.values():
                for s in streams.values():
                    n = min(excess, len(s['data']))
                    s['data'] = s['data'][n:]
                    s['base'] += n
                    excess -= n

    def read_output(self, request_id=None, stream='stdout', offset=0, limit=8192, submission_id=None):
        request_id = self._resolve_request(request_id, submission_id)
        if stream not in {'stdout', 'stderr'} or offset < 0 or not 1 <= limit <= 32768:
            raise ValueError('Invalid output page bounds')
        with self.output_lock:
            if request_id not in self.output:
                raise ValueError('Command output is unknown or evicted')
            slot = self.output[request_id][stream]
            start = min(max(offset, slot['base']), slot['base'] + len(slot['data']))
            data = slot['data'][start-slot['base']:start-slot['base']+limit]
            # Byte cursors remain exact; avoid splitting a Unicode character at the page end.
            if start-slot['base']+len(data) < len(slot['data']):
                while data:
                    try:
                        data.decode('utf-8'); break
                    except UnicodeDecodeError as error:
                        if error.reason == 'unexpected end of data': data = data[:error.start]
                        else: break
            end = start + len(data)
            return {'request_id': request_id, 'stream': stream, 'text': data.decode('utf-8', errors='replace'),
                'offset': start, 'next_offset': end, 'truncated': slot['base'] > 0,
                'available_from': slot['base'], 'eof': end >= slot['base'] + len(slot['data'])}

    def _prune(self):
        finished = [key for key, r in self.requests.items() if r['status'] in TERMINAL]
        for key in finished[:-self.history_limit]:
            del self.requests[key]
            with self.output_lock:
                self.output.pop(key, None)

    def shutdown(self):
        self._closed = True
        for request_id in self.queue:
            request = self.requests[request_id]
            request.update(status='cancelled', finished_at=now())
            def abandon(records):
                for record in records:
                    if record['status'] not in TERMINAL:
                        record.update(status='cancelled', finished_at=now(), outcome='cancelled')
                    abandon(record['children'])
            abandon(request['commands'])
            self.publish(request_id)
        self.queue.clear()


def get_portal(viewer):
    portal = getattr(viewer, 'command_portal', None)
    if portal is None:
        portal = viewer.command_portal = ViewerCommandPortal(viewer)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(portal.shutdown)
    return portal
