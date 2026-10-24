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

from desktop.Viewer_Window import viewer_shows_windows as _viewer_shows_windows
from utilities.Localization import Message

TERMINAL = {'succeeded', 'failed', 'cancelled', 'skipped'}
CURRENT = ContextVar('viewer_command_context', default=None)
# Commands from these sources are checked against the catalog's agent policy
# (desktop.Command_Metadata) before they run. A command needing approval waits
# in awaiting_user_input until decide() runs or discards it, and a refused one
# fails without running. Commands a run script emits are the user's own.
REVIEWED_SOURCES = {'web_agent'}
APPROVAL = {'run': None, 'approve': 'pending', 'refuse': 'refused'}
REFUSED = 'Agent replies cannot run this command; type it in the Viewer terminal instead.'
# The most commands a run script may hold, typed or sent through the portal.
MAX_SCRIPT_COMMANDS = 1000
# The longest `progress` line a job entry keeps.
MAX_PROGRESS_CHARACTERS = 256
# Commands that end background work, as their lower-case words. The request at the
# head of the queue waits for its jobs to end, so a stop command queued behind it
# would wait as long: a request that holds only these runs at once instead
# (ViewerCommandPortal._run_stop_requests). So do the jobs commands, which list
# and cancel the commands the Viewer's command queue holds (Command_Queue).
STOP_COMMANDS = frozenset({('esmfold', 'stop'), ('run', 'stop')})
CONTROL_COMMANDS = frozenset({'jobs'})


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


def viewer_shows_windows(viewer):
    """Whether the user is at ``viewer``: false for a headless Viewer and a hidden one not shown.

    The Viewer answers for itself (MainViewer.shows_windows); a stand-in without
    that method counts as headless when its flag or the environment says so.
    """
    return _viewer_shows_windows(viewer, environment=True)


@contextmanager
def user_interaction(viewer, description):
    """Wrap a dialog that needs the user, and refuse it where nobody can answer it.

    A Viewer that shows no windows (headless, or hidden and not shown) fails the
    command, whether or not a portal command is running it. A command from the
    portal also reports awaiting_user_input while the dialog is open.
    """
    context = CURRENT.get()
    bound = context is not None and context.portal.viewer is viewer
    # Without a portal command, a stand-in viewer is judged by its own flag; the environment
    # of the process speaks for the Viewer only when a portal command runs in it.
    if not _viewer_shows_windows(viewer, environment=bound):
        hint = ''
        if getattr(viewer, 'window_mode', None) == 'hidden':
            hint = " Show the window first (emapssn_viewer_control action='show_session')."
        raise CommandInteractionError(f'{description} requires a visible Viewer and user input.{hint}')
    if bound:
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

    def add_job(self, job_id, *, kind=None, lane=None, lane_title=None, description=None, progress=None,
                cancellable=False, stopping=False, stop_requested=False, releases_request=False, **extra):
        """Add a job to the request's record.

        ``kind`` names what the job is, ``lane`` and ``lane_title`` the line it waits in
        (None for a job that has none), ``description`` says what it does and ``progress``
        how far it is. ``cancellable`` is whether a stop request would do anything now,
        ``stopping`` whether the job was told to stop and has not ended, and ``stop_requested``
        whether it was told to stop and finished anyway. ``releases_request`` says that
        the request that gave the job need not keep its place at the head of the portal's
        line while only jobs like this are left (see ViewerCommandPortal._frees_the_line).
        The times the job started and finished are set by job_event.
        """
        job = dict(job_id=str(job_id), status='queued', kind=kind, lane=lane, lane_title=lane_title,
                   description=description, progress=progress, cancellable=cancellable, stopping=stopping,
                   stop_requested=stop_requested, releases_request=releases_request,
                   started_at=None, finished_at=None, **extra)
        self.record['jobs'].append(job)
        return job

    def job_event(self, job_id, status, message=None, artifacts=()):
        for job in self.record['jobs']:
            if job['job_id'] == str(job_id):
                job['status'] = status
                if status == 'running' and job.get('started_at') is None:
                    job['started_at'] = now()
                if status in {'succeeded', 'failed', 'cancelled'}:
                    job['finished_at'] = now()
                    job['stopping'] = False
                    job['cancellable'] = False
                if message:
                    job['message'] = str(message)[:1024]
                    self.portal.append_output(self.request_id, 'stderr' if status == 'failed' else 'stdout', str(message) + '\n')
                for artifact in artifacts:
                    self.report(artifact=artifact)
                if status in {'failed', 'cancelled'}:
                    self.report(status, message)
                break
        self.portal.changed.emit(self.request_id)

    def job_progress(self, job_id, text):
        """Show what a job reports as it goes in its entry, as `progress`; return whether the entry changed.

        `progress` (None in a new entry) is one plain line, here no markup and at most MAX_PROGRESS_CHARACTERS.
        The command queue's computing commands are fed through this; the background scheduler writes the same
        field for its lane jobs itself (Background_Job_Scheduler._update_records). Call it on the Viewer's
        thread, which also reads the record, and throttle there: this keeps only a line that differs.
        """
        text = ' '.join(str(text).split())[:MAX_PROGRESS_CHARACTERS] if text is not None else ''
        for job in self.record['jobs']:
            if job['job_id'] == str(job_id):
                if text and job.get('progress') != text:
                    job['progress'] = text
                    return True
                break
        return False

    def children(self, commands):
        if len(commands) > MAX_SCRIPT_COMMANDS or any(not isinstance(c, str) or not c.strip() or len(c) > 8192 or '\n' in c or '\r' in c for c in commands):
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
        # Requests that left the head of the line while only their background jobs were left.
        self._parked = set()
        self.changed.connect(self._changed, QtCore.Qt.ConnectionType.QueuedConnection)
        install_output_capture()

    @staticmethod
    def new_command(command):
        return dict(command_id=uuid.uuid4().hex, command=command, status='queued',
                    submitted_at=now(), started_at=None, finished_at=None, outcome=None,
                    messages=[], artifacts=[], jobs=[], children=[], dispatched=False)

    def submit(self, submission_id, commands, source='mcp'):
        # The errors an agent reply can meet hold Messages, which the Agent
        # page shows translated; str() keeps them English for MCP clients.
        if self._closed:
            raise ValueError(Message('Viewer command portal is shutting down.'))
        if not isinstance(submission_id, str) or not 1 <= len(submission_id) <= 128:
            raise ValueError('submission_id must be a client-generated string of 1..128 characters')
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list) or not 1 <= len(commands) <= 100:
            raise ValueError(Message('commands must contain 1..100 command strings'))
        if any(not isinstance(c, str) or not c.strip() or '\n' in c or '\r' in c or len(c) > 8192 for c in commands):
            raise ValueError(Message('Each command must be a nonempty single line of at most 8192 characters'))
        signature = hashlib.sha256(json.dumps([commands, source], ensure_ascii=False).encode()).hexdigest()
        if submission_id in self.submissions:
            old_signature, request_id = self.submissions[submission_id]
            if old_signature != signature:
                first = self.requests.get(request_id)
                made = (f"request {request_id}, {first['status']}, submitted {first['submitted_at']}" if first
                        else f"request {request_id}, whose result has since been evicted")
                raise ValueError(
                    f"submission_id {submission_id!r} was already used with a different payload ({made}). Ids are shared "
                    "by every client of this Viewer, and nothing was run. Use a new random UUID for each different "
                    "submission, and the same one only to retry an identical payload.")
            if request_id not in self.requests:
                raise ValueError('Submission already executed; its result was evicted. It will not be executed again.')
            return self.get(request_id)
        if len(self.queue) >= 100:
            raise ValueError(Message('Viewer command queue is full; retry later with the same submission_id'))
        records = [self.new_command(c.strip()) for c in commands]
        if source in REVIEWED_SOURCES:
            from desktop.Command_Metadata import agent_policy
            for record in records:
                record['approval'] = APPROVAL[agent_policy(record['command'])]
        request_id = uuid.uuid4().hex
        self.requests[request_id] = dict(request_id=request_id, submission_id=submission_id,
            session_id=getattr(self.viewer, 'inspection_session_id', None), source=source,
            status='queued', submitted_at=now(), finished_at=None, commands=records)
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
            if record.get('approval') == 'pending':
                record['status'] = 'awaiting_user_input'
                return False
            import Command_Engine
            record.update(status='running', started_at=now(), dispatched=True)
            context = ExecutionContext(self, request_id, record)
            if record.get('approval') == 'refused':
                context.report('failed', REFUSED)
            else:
                with bind(context):
                    try:
                        Command_Engine.execute_command(self.viewer, record['command'])
                    except Exception as error:
                        context.report('failed', str(error))
            # A command that waits its turn in the Viewer's command queue reports when it has run.
            if record['outcome'] is None and not any(j['status'] not in TERMINAL for j in record['jobs']):
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
                    if self._frees_the_line(request, record):
                        self._park(request_id)
                    elif record['dispatched']:
                        self._run_stop_requests()
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

    @staticmethod
    def _frees_the_line(request, record):
        """Whether the request at the head may let the next one run: all that is left of it is background jobs.

        That is so when ``record`` is its last command, which has no children, is
        not waiting for the user, and every job it has not finished declared at
        enqueue that it releases its request (a file or a fold in a lane of the
        background scheduler, which runs beside the commands). A request with
        more commands after it, with work in the Viewer's command queue, or
        awaiting user input keeps its place, as before.
        """
        pending = [job for job in record['jobs'] if job['status'] not in TERMINAL]
        return (bool(pending) and all(job.get('releases_request') for job in pending)
                and record is request['commands'][-1] and not record['children']
                and record['status'] == 'running')

    def _park(self, request_id):
        """Take the head request out of the line while its jobs run; the line goes on with the next request.

        Its status stays running. A job event sweeps it (_sweep), and it ends
        when its jobs have, with completed emitted then.
        """
        self.queue.remove(request_id)
        self._parked.add(request_id)
        QtCore.QTimer.singleShot(0, self.pump)

    def _sweep(self, request_id):
        """A job of a parked request reported: finish the request if its jobs have all ended."""
        request = self.requests[request_id]
        record = request['commands'][-1]
        if not self._advance(request_id, record):
            request['status'] = record['status']
            return
        self._parked.discard(request_id)
        request.update(status=record['status'] if record['status'] in {'failed', 'cancelled'} else 'succeeded',
                       finished_at=now())
        self.publish(request_id)
        self.completed.emit(request_id)
        self._prune()

    @staticmethod
    def _is_stop(record):
        """Whether record is a stop or jobs command that may run now: not still waiting for an Agent review."""
        words = tuple(record['command'].lower().split())
        return ((words in STOP_COMMANDS or (bool(words) and words[0] in CONTROL_COMMANDS))
                and record.get('approval') in (None, 'approved'))

    def _run_stop_requests(self):
        """Run the queued requests that hold only stop commands, while the request at the head waits for its jobs.

        `esmfold stop` and `run stop` end the very jobs that keep the head waiting, so
        behind it they could never run. The head stays where it is: its job ends
        cancelled, and the queue goes on from there.
        """
        for request_id in list(self.queue[1:]):
            request = self.requests[request_id]
            if request['status'] != 'queued' or not all(self._is_stop(record) for record in request['commands']):
                continue
            request['status'] = 'running'
            failure = None
            finished = True
            for record in request['commands']:
                if failure:
                    self._skip(record)
                elif not self._advance(request_id, record):
                    finished = False  # A stop command has no jobs to wait for; the queue takes it from here.
                    break
                elif record['status'] in {'failed', 'cancelled'}:
                    failure = record['status']
            if not finished:
                continue
            request.update(status=failure or 'succeeded', finished_at=now())
            self.queue.remove(request_id)
            self.publish(request_id)
            self.completed.emit(request_id)
            self._prune()

    def _changed(self, request_id):
        if request_id in self._parked:
            self._sweep(request_id)
        self.publish(request_id)
        self.pump()

    def decide(self, request_id, command_id, run):
        """Run or discard the command a reviewed request is waiting to run."""
        request_id = self._resolve_request(request_id)
        if not isinstance(run, bool):
            raise ValueError('run must be true or false')
        record = next((c for c in self.requests[request_id]['commands'] if c['command_id'] == command_id), None)
        if record is None or record.get('approval') != 'pending' or record['status'] != 'awaiting_user_input':
            raise ValueError(Message('That command is not waiting to be run or discarded.'))
        if run:
            record.update(approval='approved', status='queued')
        else:
            self._discard(request_id, record)
        self.changed.emit(request_id)

    def discard_pending(self, request_id):
        """Discard every command of a request that still waits for approval."""
        request_id = self._resolve_request(request_id)
        records = [c for c in self.requests[request_id]['commands'] if c.get('approval') == 'pending']
        for record in records:
            self._discard(request_id, record)
        if records:
            self.changed.emit(request_id)

    def _discard(self, request_id, record):
        record.update(approval='discarded', status='cancelled', finished_at=now())
        ExecutionContext(self, request_id, record).report('cancelled', 'Discarded in the Agent page.')

    def publish(self, request_id):
        request = self.requests.get(request_id)
        if request and hasattr(self.viewer, 'broadcast_event'):
            self.viewer.broadcast_event({'type': 'command_request_updated',
                'request_id': request_id, 'status': request['status'],
                'commands': [{'command_id': c['command_id'], 'command': c['command'], 'status': c['status'],
                              'approval': c.get('approval')} for c in request['commands']]})

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
            raise ValueError(Message('Command request is unknown or evicted from this Viewer'))
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
        # A page must hold the longest UTF-8 character (4 bytes); a smaller
        # limit can return an empty page that never advances the cursor.
        if stream not in {'stdout', 'stderr'} or offset < 0 or not 4 <= limit <= 32768:
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
        for request_id in [*self.queue, *self._parked]:
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
        self._parked.clear()


def get_portal(viewer):
    portal = getattr(viewer, 'command_portal', None)
    if portal is None:
        portal = viewer.command_portal = ViewerCommandPortal(viewer)
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(portal.shutdown)
    return portal
