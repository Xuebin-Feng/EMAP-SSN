# Copyright 2026 Xuebin Feng
# SPDX-License-Identifier: Apache-2.0
"""The Viewer's command queue: commands run one at a time, in the order they were given.

Each command runs on the network as the commands given before it left it, as
if they had been typed one by one, and the Viewer keeps responding meanwhile.

* **Given** (dispatch, GUI thread): Command_Engine hands the queue a command
  that reads or changes the network. A module with ``plan(viewer, args)``
  checks its arguments now, so a typo is refused at once; plan returns None
  when it refused (or answered at once) and has reported the outcome, or a
  plan. The selection the command will use is bound now (see Job).
* **Turn** (GUI thread): when every command given before it has finished,
  ``start(viewer, plan)`` checks what the network must have (a cluster, a
  metadata column) and gathers copies of what can change. It returns None
  when the command ends there, a no-op or a refusal that it has reported.
* **Compute**: ``compute(inputs, progress)`` works on the copies, with no
  viewer and no Qt. A plan whose ``heavy`` is true computes on the queue's
  worker thread (NumPy, Numba, graspologic and SciPy release the GIL, so the
  GUI keeps running); a light one, here at its turn.
* **Apply** (GUI thread, one step): ``apply(viewer, plan, result)`` saves the
  undo state once, writes the result and redraws. It waits while nodes are
  dragged or a box selection is drawn.

A module without plan() (not yet split) runs its whole run(viewer, args) on
the GUI thread at its turn. When nothing waits, a command that is not heavy
runs at once, as it always did, without becoming a job.

A failure at a command's turn cancels the commands given after it from the same
place: typed at the command line, or by a script typed there. The command
portal (MCP, the Agent page) sends one command of a request at a time and skips
the rest of a request itself.
"""
from __future__ import annotations

import collections
import itertools
import queue as queue_module
import threading
import time
import traceback
import weakref

import numpy as np
from PySide6 import QtCore

import Command_Engine
from utilities.Localization import JoinedMessage, Message

# A module without plan() is run whole, by its run(viewer, args), at its turn.
LEGACY = object()
WAITING = ('queued', 'running', 'applying')
# Commands from these places form one line of work: a failure cancels the later ones.
COMMAND_LINE_SOURCES = ('typed', 'script')
_NO_SELECTION = np.empty(0, dtype=np.int64)
# How often the status line shows a computing command's time and progress.
STATUS_REFRESH_MS = 500


class CommandCancelled(Exception):
    """Raised in a compute step whose command was cancelled, at its next progress report."""


def _epoch(viewer):
    return getattr(viewer, 'selection_epoch', 0)


def _selection_array(indices):
    return np.asarray(list(indices) if indices is not None else (), dtype=np.int64)


class Progress:
    """What a compute step may report from its thread: progress('text') and cancelled.

    A call after the command was cancelled raises CommandCancelled, which ends
    the step; its result would be thrown away anyway.
    """

    def __init__(self, job):
        self._job = job

    @property
    def cancelled(self):
        return self._job.cancel_requested

    def __call__(self, text=None):
        if self._job.cancel_requested:
            raise CommandCancelled()
        if text is not None:
            self._job.progress_text = str(text)


def no_progress(text=None):
    """The progress callback of a command run at once (run(viewer, args)): it reports nothing."""


class Job:
    """One command given to the queue, from the moment it is given until it has finished.

    The selection it uses (Command_Engine.command_selection) is the one that
    stood when it was given: a snapshot of the live selection, or, when the
    command given before it was still waiting and nothing was selected by hand
    since, the selection that command leaves (``previous``). While nothing is
    selected by hand after it is given, that is the live selection itself at
    its turn. A selection the command makes (set_command_selection) becomes
    the live one only if nothing was selected by hand after it was given.
    """

    def __init__(self, queue, viewer, job_id, line, name, args, module, plan, source, context, silent):
        self.queue = queue
        self._viewer = weakref.ref(viewer)
        self.job_id = job_id
        self.line = line
        self.name = name
        self.args = args
        self.module = module
        self.plan = plan
        self.source = source
        self.context = context
        self.silent = silent
        self.heavy = plan is not LEGACY and bool(getattr(plan, 'heavy', False))
        # Its feedback goes to the status line while a command is typed (Command_Engine.show_status).
        self.deferred = True
        self.status = 'queued'
        self.finished = False
        self.started = False
        self.outcome = None
        self.message = None
        self.cancel_requested = False
        self.cancelled_by_request = False
        self.progress_text = None
        self.given_at = time.monotonic()
        self.started_at = None
        self.finished_at = None
        self.result = None
        self.traceback = None
        # The selection: see the class docstring.
        self.dispatch_epoch = _epoch(viewer)
        self.snapshot = None
        self.previous = None
        self.selection_in = None
        self.selection_out = None

    @property
    def viewer(self):
        return self._viewer()

    def label(self):
        """'#14 color red': how messages name the job."""
        return f"#{self.job_id} {self.line}"

    # --- the selection ---------------------------------------------------

    def resolve_selection(self):
        """The selection this job was given, found once (iteratively: chains of cancelled jobs can be long)."""
        pending = []
        job = self
        while job.selection_in is None:
            if job.snapshot is not None:
                job.selection_in, job.snapshot, job.previous = job.snapshot, None, None
                break
            previous = job.previous
            if previous is None:
                job.selection_in = _NO_SELECTION
                break
            if previous.selection_out is not None:
                job.selection_in, job.previous = previous.selection_out, None
                break
            pending.append(job)
            job = previous
        for later in reversed(pending):
            previous = later.previous
            later.selection_in = previous.selection_out if previous.selection_out is not None else previous.selection_in
            later.previous = None
        return self.selection_in

    def bound_selection(self, viewer):
        """The selection this command uses when the live one is not it, else None."""
        if viewer is not self.viewer or _epoch(viewer) == self.dispatch_epoch:
            return None
        return self.resolve_selection()

    def leave_selection(self, viewer, indices):
        """Record the selection this command made; return whether it may become the live one."""
        if viewer is not self.viewer:
            return True
        self.selection_out = _selection_array(indices)
        return _epoch(viewer) == self.dispatch_epoch

    # --- outcome ---------------------------------------------------------

    def note_outcome(self, status):
        """Keep the outcome the command reported: the latest, except that a failure or cancellation stays."""
        if status is not None and self.outcome not in ('failed', 'cancelled'):
            self.outcome = status

    def seconds(self):
        start = self.started_at if self.started_at is not None else self.given_at
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - start)


class ThreadExecutor:
    """Runs compute steps one at a time, in order, on one daemon thread."""

    def __init__(self):
        self._tasks = queue_module.Queue()
        self._thread = None
        self._lock = threading.Lock()

    def submit(self, job, compute, done):
        """Run compute() on the thread, then done(job, result, error) there."""
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name='SSN-Command-Compute', daemon=True)
                self._thread.start()
        self._tasks.put((job, compute, done))

    def _loop(self):
        while True:
            job, compute, done = self._tasks.get()
            if job is None:
                return
            _run_compute(job, compute, done)

    def shutdown(self):
        self._tasks.put((None, None, None))


def _run_compute(job, compute, done):
    try:
        result, error = compute(), None
    except Exception as exception:
        result, error = None, exception
        job.traceback = traceback.format_exc()
    try:
        done(job, result, error)
    except RuntimeError:
        pass  # The queue's Qt object is deleted: the Viewer closed.


class ManualExecutor:
    """An executor for tests: a compute step runs only when finish() is called, on the calling thread."""

    def __init__(self):
        self.tasks = []

    def submit(self, job, compute, done):
        self.tasks.append((job, compute, done))

    def finish(self, job=None):
        """Run the oldest compute step, or job's; return its job."""
        index = 0 if job is None else next(i for i, task in enumerate(self.tasks) if task[0] is job)
        task = self.tasks.pop(index)
        _run_compute(*task)
        return task[0]

    def shutdown(self):
        self.tasks.clear()


class CommandQueue(QtCore.QObject):
    """The Viewer's command queue (see the module docstring). Lives on the GUI thread."""

    changed = QtCore.Signal()
    _computed = QtCore.Signal(object, object, object)

    HISTORY = 20

    def __init__(self, viewer, *, executor=None, job_numbers=None):
        super().__init__()
        self._viewer = weakref.ref(viewer)
        self.executor = executor if executor is not None else ThreadExecutor()
        self._numbers = job_numbers if job_numbers is not None else itertools.count(1)
        self.jobs = []
        self.finished = collections.deque(maxlen=self.HISTORY)
        # True while a job's step runs on the GUI thread: undo and redo then belong to that job.
        self.in_turn = False
        self._last_given = None
        self._last_given_epoch = None
        self._closed = False
        self._computed.connect(self._on_computed, QtCore.Qt.ConnectionType.QueuedConnection)
        self._wake = QtCore.QTimer(self)
        self._wake.setSingleShot(True)
        self._wake.timeout.connect(self._resume)
        self._status_timer = QtCore.QTimer(self)
        self._status_timer.setInterval(STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self._show_running)

    # --- what waits ----------------------------------------------------------

    def pending(self):
        """The jobs given and not finished, in the order given."""
        return list(self.jobs)

    def uses_module(self, command_name):
        """Whether a waiting or running job still has to run command_name's module."""
        return any(job.name == command_name for job in self.jobs)

    def find(self, job_id):
        return next((job for job in self.jobs if job.job_id == job_id), None)

    def running(self):
        """The job whose turn it is, or None."""
        return next((job for job in self.jobs if job.status != 'queued'), None)

    # --- given -------------------------------------------------------------

    def dispatch(self, module, name, args, line, *, source='typed', silent=False):
        """Take a command given now; return its Job, or None when it ran at once, was refused or answered."""
        if self._closed:
            raise RuntimeError(Message("The command queue is shutting down."))
        viewer = self._viewer()
        from Viewer_Command_Portal import CURRENT
        context = CURRENT.get()
        planner = getattr(module, 'plan', None)
        plan = LEGACY if planner is None else planner(viewer, list(args))
        if plan is None:
            return None
        heavy = plan is not LEGACY and bool(getattr(plan, 'heavy', False))
        if not self.jobs and not heavy:
            # Nothing waits: it runs now, as it always did.
            if plan is LEGACY:
                Command_Engine._run_module(viewer, module, name, list(args), silent)
            else:
                self._run_planned_now(viewer, module, plan)
            return None

        job = Job(self, viewer, next(self._numbers), line, name, list(args), module, plan, source, context, silent)
        if self.jobs and self._last_given is not None and self._last_given_epoch == job.dispatch_epoch:
            job.previous = self._last_given
        else:
            job.snapshot = _selection_array(getattr(viewer, 'selected_indices', None))
        ahead = self.jobs[-1] if self.jobs else None
        self.jobs.append(job)
        self._last_given, self._last_given_epoch = job, job.dispatch_epoch
        if context is not None:
            context.add_job(str(job.job_id), command=line, kind='command', description=line, cancellable=True)
        self._update_places()
        if ahead is not None:
            message = Message("#{job} {command}: waiting for #{ahead} {ahead_command}.",
                              job=job.job_id, command=line, ahead=ahead.job_id, ahead_command=ahead.line)
            print(message)
            if not silent and hasattr(viewer, 'console_text'):
                Command_Engine.show_status(viewer, message)
            if context is not None:
                context.report(message=message)
        self.changed.emit()
        self._pump(from_dispatch=True)
        return job

    def _run_planned_now(self, viewer, module, plan):
        """A light planned command given while nothing waits: all its steps, now."""
        inputs = module.start(viewer, plan)
        if inputs is None:
            return
        module.apply(viewer, plan, module.compute(inputs, no_progress))
        viewer.broadcast_metadata_state()

    # --- turns -------------------------------------------------------------

    def _busy(self, viewer, from_dispatch=False):
        """Whether a turn must wait: nodes are being dragged, a box drawn, or another command dispatched."""
        return (getattr(viewer, 'is_multi_dragging', False) or getattr(viewer, 'is_box_selecting', False)
                or (not from_dispatch and getattr(viewer, '_command_dispatch_active', False)))

    def _schedule(self, milliseconds=0):
        if not self._closed:
            self._wake.start(milliseconds)

    def resume(self):
        """Go on with the queue now, as after a drag ends (the Viewer calls it on mouse release)."""
        self._schedule(0)

    def _resume(self):
        head = self.jobs[0] if self.jobs else None
        if head is not None and head.status == 'applying':
            self._try_apply(head)
        else:
            self._pump()

    def _pump(self, from_dispatch=False):
        if self._closed or self.in_turn or not self.jobs:
            return
        viewer = self._viewer()
        if viewer is None:
            return
        head = self.jobs[0]
        if head.status != 'queued':
            return  # Its turn has come: it computes, or waits to apply.
        if self._busy(viewer, from_dispatch):
            self._schedule(20)
            return
        self._start(head, viewer)
        if self.jobs and self.jobs[0].status == 'queued':
            # One light command per event-loop pass, so input and painting go on between them.
            self._schedule(0)

    def _step(self, job, viewer, step):
        """Run one GUI-thread step of job as the command itself: its context, selection and outcome."""
        from Viewer_Command_Portal import bind
        was_active = getattr(viewer, '_command_dispatch_active', False)
        viewer._command_dispatch_active = True
        self.in_turn = True
        value = None
        try:
            with bind(job.context), Command_Engine.command_job(job), Command_Engine.recorded_outcome() as outcome:
                try:
                    value = step()
                except Exception as error:
                    Command_Engine._report_command_error(viewer, error, job.silent)
            job.note_outcome(outcome['status'])
        finally:
            self.in_turn = False
            viewer._command_dispatch_active = was_active
        return value

    def _start(self, job, viewer):
        job.status = 'running'
        job.started = True
        job.started_at = time.monotonic()
        job.resolve_selection()
        self._event(job, 'running')
        self._update_places()
        self.changed.emit()
        if job.plan is LEGACY:
            self._step(job, viewer, lambda: Command_Engine.run_command_module(
                viewer, job.module, job.name, list(job.args), job.silent))
            self._finish(job, viewer)
            return
        inputs = self._step(job, viewer, lambda: job.module.start(viewer, job.plan))
        if job.finished:
            return
        if inputs is None or job.outcome in ('failed', 'cancelled'):
            self._finish(job, viewer)
            return
        if not job.heavy:
            result = self._step(job, viewer, lambda: job.module.compute(inputs, Progress(job)))
            if job.outcome in ('failed', 'cancelled'):
                self._finish(job, viewer)
                return
            self._apply(job, viewer, result)
            return
        self._show_running()
        self._status_timer.start()
        self.executor.submit(job, lambda: self._compute(job, inputs), self._computed.emit)

    @staticmethod
    def _compute(job, inputs):
        """The compute step, on the worker thread: printed output goes to the command's portal record."""
        from Viewer_Command_Portal import bind
        with bind(job.context):
            return job.module.compute(inputs, Progress(job))

    def _on_computed(self, job, result, error):
        viewer = self._viewer()
        if job.finished or viewer is None or self._closed:
            self._schedule(0)  # A cancelled command's result is thrown away.
            return
        if error is not None:
            def report():
                raise error  # Reported, with its traceback from the worker, as a command's error is.
            self._step(job, viewer, report)
            self._finish(job, viewer)
            return
        job.result = result
        job.status = 'applying'
        self._event(job, 'applying')
        self._try_apply(job)

    def _try_apply(self, job):
        viewer = self._viewer()
        if job.finished or viewer is None:
            return
        if self._busy(viewer):
            self._schedule(20)
            return
        result, job.result = job.result, None
        self._apply(job, viewer, result)

    def _apply(self, job, viewer, result):
        def apply():
            try:
                job.module.apply(viewer, job.plan, result)
                viewer.broadcast_metadata_state()
            finally:
                Command_Engine._refresh_info_card(viewer)
        self._step(job, viewer, apply)
        self._finish(job, viewer)

    # --- finished ------------------------------------------------------------

    def _finish(self, job, viewer, status=None, message=None):
        if job.finished:
            return
        if status is None:
            if job.outcome is None:
                # As the command portal says of a handler that reports nothing.
                self._step(job, viewer, lambda: Command_Engine.command_failed(
                    viewer, 'Command handler did not report an explicit outcome.'))
            status = job.outcome if job.outcome in ('succeeded', 'failed', 'cancelled') else 'failed'
        self._copy_progress(job)  # what it last reported, though no status tick came after it
        job.finished = True
        job.status = status
        job.message = message
        job.finished_at = time.monotonic()
        if job.started and viewer is not None and _epoch(viewer) == job.dispatch_epoch:
            # Nothing was selected by hand since it was given: what it leaves is the live selection.
            job.selection_out = _selection_array(getattr(viewer, 'selected_indices', None))
        if job in self.jobs:
            self.jobs.remove(job)
        self.finished.append(job)
        self._event(job, status, message)
        if status in ('failed', 'cancelled') and not job.cancelled_by_request:
            self._cancel_after_failure(job, viewer)
        if not any(waiting.status == 'running' and waiting.heavy for waiting in self.jobs):
            self._status_timer.stop()
        self._update_places()
        self._show_finished(job, viewer)
        self.changed.emit()
        self._schedule(0)

    def _event(self, job, status, message=None):
        """Tell the command portal record that gave job how it stands."""
        if job.context is None:
            return
        job.context.job_event(str(job.job_id), status, message)

    def _update_places(self):
        """Each waiting job's place in the line, in its portal record: position and the job ahead of it."""
        for position, job in enumerate(self.jobs, 1):
            if job.context is None:
                continue
            for entry in job.context.record['jobs']:
                if entry['job_id'] == str(job.job_id):
                    entry['position'] = position
                    entry['waiting_for'] = str(self.jobs[position - 2].job_id) if position > 1 else None

    def _cancel_after_failure(self, failed, viewer):
        """A command from the command line failed: cancel the later commands from there, which assumed it worked."""
        if failed.source not in COMMAND_LINE_SOURCES:
            return
        later = [job for job in self.jobs if job.source in COMMAND_LINE_SOURCES and job.status == 'queued']
        if not later:
            return
        reason = Message("Cancelled: #{job}, given before it, failed.", job=failed.job_id)
        for job in later:
            job.cancel_requested = True
            self._finish(job, viewer, 'cancelled', reason)
        note = Message("Cancelled %n command(s) given after #{job}, which failed: {commands}. "
                       "Press Up at the command line to recall the ones you typed.",
                       n=len(later), job=failed.job_id, commands=", ".join(job.label() for job in later))
        print(note)
        if viewer is not None and hasattr(viewer, 'set_background_job_status'):
            viewer.set_background_job_status(note)

    # --- cancelling --------------------------------------------------------

    def cancel(self, job, reason=None):
        """Cancel a job that waits, computes or waits to apply; return whether it was not finished.

        A computation that cannot stop at once runs to its end on the worker
        thread and its result is thrown away; the queue goes on at once.
        """
        if job.finished:
            return False
        job.cancel_requested = True
        job.cancelled_by_request = True
        self._finish(job, self._viewer(), 'cancelled',
                     reason if reason is not None else Message("Cancelled before it finished."))
        return True

    def cancel_newest(self, count=1, reason=None):
        """Cancel the newest count jobs that are not finished; return them, newest first."""
        cancelled = []
        while len(cancelled) < count and self.jobs:
            job = self.jobs[-1]
            self.cancel(job, reason)
            cancelled.append(job)
        return cancelled

    def cancel_all(self, reason=None):
        return self.cancel_newest(len(self.jobs), reason)

    def take_back(self, count=1):
        """Undo by hand while commands wait: cancel the newest count of them.

        Returns (jobs cancelled, Message saying so), or ([], None) when nothing waits.
        """
        cancelled = self.cancel_newest(count, Message("Cancelled by undo before it finished."))
        if not cancelled:
            return [], None
        message = Message("Undo: cancelled %n command(s) that had not finished: {commands}.",
                          n=len(cancelled), commands=", ".join(job.label() for job in cancelled))
        print(message)
        return cancelled, message

    def shutdown(self):
        """The Viewer closes: cancel every job, and run nothing more."""
        if self._closed:
            return
        self._closed = True
        self._wake.stop()
        self._status_timer.stop()
        viewer = self._viewer()
        for job in list(self.jobs):
            job.cancel_requested = True
            job.cancelled_by_request = True
            job.finished = True
            job.status = 'cancelled'
            job.finished_at = time.monotonic()
            self.jobs.remove(job)
            self.finished.append(job)
            try:
                self._event(job, 'cancelled', Message("Cancelled: the Viewer closed."))
            except RuntimeError:
                pass
        self.executor.shutdown()
        if viewer is not None:
            self.changed.emit()

    # --- the status line -----------------------------------------------------

    @staticmethod
    def _copy_progress(job):
        """Show the progress text a compute step reports in the job's command portal entry, as `progress`."""
        if job.context is not None and job.progress_text:
            job.context.job_progress(str(job.job_id), job.progress_text)

    def _show_running(self):
        viewer = self._viewer()
        job = self.running()
        if job is not None and job.status == 'running':
            # The step reports from its worker thread; the portal record is read on this one, so the text
            # is copied here, once per tick of the status timer: that is the throttle.
            self._copy_progress(job)
        if viewer is None or job is None or job.status != 'running' or not hasattr(viewer, 'set_background_job_status'):
            return
        seconds = f"{job.seconds():.0f}"
        if job.progress_text:
            parts = [Message("Running #{job} {command}: {progress} ({seconds} s).", job=job.job_id,
                             command=job.line, progress=job.progress_text, seconds=seconds)]
        else:
            parts = [Message("Running #{job} {command} ({seconds} s).", job=job.job_id, command=job.line,
                             seconds=seconds)]
        waiting = len(self.jobs) - 1
        if waiting:
            parts.append(Message("%n more command(s) waiting.", n=waiting))
        viewer.set_background_job_status(JoinedMessage(parts) if len(parts) > 1 else parts[0])

    def _show_finished(self, job, viewer):
        if viewer is None or not hasattr(viewer, 'set_background_job_status') or self._closed:
            return
        if job.cancelled_by_request or not job.started:
            return  # Cancellation says so itself.
        seconds = f"{job.seconds():.1f}"
        if job.status == 'succeeded':
            message = Message("#{job} {command} finished in {seconds} s.", job=job.job_id, command=job.line, seconds=seconds)
        else:
            message = Message("#{job} {command} failed after {seconds} s.", job=job.job_id, command=job.line, seconds=seconds)
        if self.jobs:
            message = JoinedMessage([message, Message("%n more command(s) waiting.", n=len(self.jobs))])
        viewer.set_background_job_status(message)
