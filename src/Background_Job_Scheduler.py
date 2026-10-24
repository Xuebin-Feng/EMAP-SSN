# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Viewer's background jobs: lanes of work that run beside each other, each in the order of its own line.

A command that has slow work to do hands it to the scheduler with enqueue() and
says, in that call, everything the scheduler and the Session card need to know
about it: the lane it goes in, its description, the output paths it reserves, the
function that cleans up after it, and a log it writes. Nothing here, and nothing
in the card, names a command, so a command module (a plugin) can use the
scheduler without the scheduler being changed.

* **Lanes.** A lane is a line of jobs run one at a time, in order, on a daemon
  thread of its own; the lanes run side by side. "files" is the default lane. A
  job names another lane, and the first job to name it creates it;
  register_lane() gives a lane its title for the card.
* **The line.** Each lane's waiting jobs are a list the GUI thread may rearrange
  (move) or remove from (cancel) while the lane's thread takes the next job.
* **Stopping.** A running job is told to stop (cancel) and ends at the next check
  point of its worker (utilities.Job_Cancellation), which deletes its partial
  files; or, for a job that waits on a process, at the stop function the worker
  registered with its JobHandle.
* **Closing.** shutdown() stops every running job, waits a moment, and deletes
  the partial files of a job that did not stop.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import itertools
import os
import threading
import time
import traceback
from typing import Any, Callable, Mapping
import weakref

from PySide6 import QtCore
from desktop.Desktop_App import open_in_file_manager
from desktop.Viewer_Window import viewer_shows_windows
from utilities.Job_Cancellation import GuiCall, JobCancelled, JobHandle, bound_to
from utilities.Localization import Message


Worker = Callable[..., Mapping[str, Any]]

# How long closing the Viewer waits for the running jobs to reach a check point and stop.
SHUTDOWN_WAIT_SECONDS = 2.5
# The lane a job goes in unless it names another, and its title.
DEFAULT_LANE = "files"
# How many failed jobs are kept for the card to show, until each is dismissed.
KEPT_FAILURES = 5


def _takes_handle(worker):
    """Whether the worker has a second parameter named ``job``, which is the job's JobHandle."""
    try:
        parameters = list(inspect.signature(worker).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = [parameter for parameter in parameters
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 2 and positional[1].name == "job"


@dataclass(frozen=True)
class BackgroundJob:
    """One job, as the command that queued it described it.

    ``reserved_paths`` are every output the job reserves, ``output_path`` first;
    ``cleanup()`` is called once when the job leaves the scheduler, however it
    ends; ``on_started(job)`` is called on the GUI thread when the job begins to
    run, and ``on_finished(job, outcome, error)`` once its end is reported,
    outcome being "succeeded", "failed" or "cancelled" and error the error of a
    failed job; ``log_path`` is a log the job writes.
    """
    job_id: int
    command_name: str
    description: str
    payload: Any
    worker: Worker
    output_path: str
    command_context: Any = None
    lane: str = DEFAULT_LANE
    reserved_paths: tuple = ()
    cleanup: Callable[[], None] | None = None
    on_started: Callable[..., None] | None = None
    on_finished: Callable[..., None] | None = None
    log_path: str | None = None


@dataclass(frozen=True)
class JobStatus:
    """One unfinished job as snapshot() reports it.

    ``position`` is its place in its lane's line, the running job being 1;
    ``started_at`` is the running job's time.monotonic() start, None for a
    waiting one; ``stopping`` is true for the running job once it was told to
    stop; ``progress`` is what the running job says it is doing; ``log_path``
    the log it writes, when it names one.
    """
    job: BackgroundJob
    running: bool
    position: int
    started_at: float | None = None
    stopping: bool = False
    lane: str = DEFAULT_LANE
    progress: Any = None
    log_path: str | None = None


@dataclass(frozen=True)
class FailedJob:
    """A job that failed, kept for the card to show with its reason and, if it wrote one, its log."""
    job_id: int
    description: str
    lane: str
    error: Any
    log_path: str | None


class _Lane:
    """One line of jobs and the thread that runs them. Every field is guarded by the scheduler's lock."""

    def __init__(self, name, title, lock):
        self.name = name
        self.title = title
        # The jobs waiting, in the order they will run; the running one is not in it.
        self.queued = []
        self.active = None
        self.handle = None
        self.started = None
        self.stopping = False
        self.thread = None
        self.wake = threading.Condition(lock)


class BackgroundJobScheduler(QtCore.QObject):
    """Run command jobs in lanes: one job at a time within a lane, the lanes side by side.

    Everything that reads or changes a lane's line, or chooses its next job,
    happens under one lock, so a job is either still in its line, where it can
    be moved or removed, or running, where it cannot be moved and can only be
    told to stop. A lane's thread takes the job at the front of its own line
    and no other, so rearranging one lane never touches another's.
    """

    job_started = QtCore.Signal(object)
    job_succeeded = QtCore.Signal(object, object, float)
    # The error itself, not its text, so a Message it holds shows translated.
    job_failed = QtCore.Signal(object, object, str, float)
    # The job, and whether it had started: a job removed from its line, or one stopped while it ran.
    job_cancelled = QtCore.Signal(object, bool)
    # A line or a running job changed: sent once the new state stands, for whatever lists the jobs.
    changed = QtCore.Signal()
    # A job asks for a function to be run on the GUI thread (JobHandle.call_in_gui).
    _gui_requested = QtCore.Signal(object)

    def __init__(self, viewer, job_numbers=None):
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._lock = threading.Lock()
        self._accepting = True
        self._lanes = {}
        # Job numbers: the Viewer's command queue draws from the same count, so
        # every job the Viewer lists has a number of its own.
        self._job_numbers = job_numbers if job_numbers is not None else itertools.count(1)
        # Every output path a job not yet finished reserves, and the job.
        self._reserved = {}
        self._cleaned = set()
        self._failures = []
        # The log a job that failed named while it ran, until its failure is reported.
        self._logs = {}
        # The jobs that were told to stop and finished anyway, until their end is reported.
        self._not_stopped = set()

        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self.job_started.connect(self._report_started, queued)
        self.job_succeeded.connect(self._report_succeeded, queued)
        self.job_failed.connect(self._report_failed, queued)
        self.job_cancelled.connect(self._report_cancelled, queued)
        self.changed.connect(self._update_records, queued)
        self._gui_requested.connect(self._run_gui_call, queued)

        with self._lock:
            default = self._lane(DEFAULT_LANE, Message("Files"))
            self._start(default)
        # The default lane's thread, which the tests wait on.
        self._thread = default.thread

    @staticmethod
    def _output_key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    # --- lanes ---

    def _lane(self, name, title=None):
        """The lane called name, made if there is none; call with the lock held."""
        lane = self._lanes.get(name)
        if lane is None:
            lane = self._lanes[name] = _Lane(name, title if title is not None else name, self._lock)
        elif title is not None:
            lane.title = title
        return lane

    def _start(self, lane):
        """Start the lane's thread if it has none; call with the lock held."""
        if lane.thread is None:
            lane.thread = threading.Thread(
                target=self._run_lane, args=(lane,),
                name="SSN-Background-Jobs" if lane.name == DEFAULT_LANE else f"SSN-Background-Jobs-{lane.name}",
                daemon=True,
            )
            lane.thread.start()

    def register_lane(self, name, title):
        """Give the lane its title for the Session card (a Message or text), making the lane if it is new."""
        with self._lock:
            self._lane(name, title)

    def lanes(self):
        """(name, title) of every lane, in the order the lanes were made."""
        with self._lock:
            return [(lane.name, lane.title) for lane in self._lanes.values()]

    @property
    def queue_depth(self):
        """Return the number of running and queued jobs."""
        with self._lock:
            return sum(len(lane.queued) + (lane.active is not None) for lane in self._lanes.values())

    def _line(self, lane):
        """The lane's unfinished jobs in the order of its line, the running one first; call with the lock held."""
        return ([lane.active] if lane.active is not None else []) + lane.queued

    def listing(self):
        """The jobs not yet finished, lane after lane, each in the order of its line, as (job, running) pairs."""
        with self._lock:
            return [(job, job is lane.active) for lane in self._lanes.values() for job in self._line(lane)]

    def snapshot(self):
        """The unfinished jobs, lane after lane, each lane in the order of its line, as JobStatus entries, read at one moment."""
        with self._lock:
            entries = []
            for lane in self._lanes.values():
                position = 1
                if lane.active is not None:
                    handle = lane.handle
                    entries.append(JobStatus(
                        lane.active, True, position, lane.started, lane.stopping, lane.name,
                        handle.progress_text if handle is not None else None,
                        (handle.log_path if handle is not None else None) or lane.active.log_path,
                    ))
                    position += 1
                for job in lane.queued:
                    entries.append(JobStatus(job, False, position, None, False, lane.name, None, job.log_path))
                    position += 1
            return entries

    def failures(self):
        """The jobs that failed lately, oldest first, as FailedJob entries, until each is dismissed."""
        with self._lock:
            return list(self._failures)

    def dismiss_failure(self, job_id):
        """Forget a failed job the card showed; return whether there was one."""
        with self._lock:
            kept = [failure for failure in self._failures if failure.job_id != job_id]
            found = len(kept) != len(self._failures)
            self._failures = kept
        if found:
            self.changed.emit()
        return found

    def log_path_of(self, job_id):
        """The log a job names, by its number: a job waiting or running, or one that failed lately; else None."""
        with self._lock:
            for lane in self._lanes.values():
                for job in self._line(lane):
                    if job.job_id == job_id:
                        handle = lane.handle if job is lane.active else None
                        return (handle.log_path if handle is not None else None) or job.log_path
            for failure in self._failures:
                if failure.job_id == job_id:
                    return failure.log_path
        return None

    def reservation_holder(self, path):
        """The number of the job that reserves path, or None."""
        with self._lock:
            return self._reserved.get(self._output_key(path))

    def _find(self, job_id):
        """(lane, index of the waiting job or None for the running one) of an unfinished job, or (None, None)."""
        for lane in self._lanes.values():
            for index, job in enumerate(lane.queued):
                if job.job_id == job_id:
                    return lane, index
            if lane.active is not None and lane.active.job_id == job_id:
                return lane, None
        return None, None

    # --- the line ---

    def cancel(self, job_id):
        """Take a job out of its line, or tell the running one to stop; return whether there was one to.

        A waiting job is gone at once: its outputs are released and it is reported
        cancelled. The running job is told to stop and ends at the next check
        point of its worker, which deletes its unfinished files; it is reported
        cancelled then. A worker with no check point finishes its file as it
        would have, and is reported as completed. A job already told to stop
        is not told again.
        """
        return bool(self.cancel_many([job_id]))

    def cancel_many(self, job_ids):
        """Cancel several jobs as one step; return the numbers of those that there was something to cancel.

        The waiting ones are taken out of their lines before any running one is
        told to stop, in a single step: a lane's next job could otherwise start
        in between, when the running job ended at once, and run although it was
        among those to cancel.
        """
        removed = []
        handles = []
        cancelled = []
        with self._lock:
            for job_id in job_ids:
                lane, index = self._find(job_id)
                if lane is None:
                    continue
                if index is not None:
                    job = lane.queued.pop(index)
                    self._release(job)
                    removed.append(job)
                elif not lane.stopping:
                    lane.stopping = True
                    if lane.handle is not None:
                        handles.append(lane.handle)
                else:
                    continue
                cancelled.append(job_id)
        for job in removed:
            self.job_cancelled.emit(job, False)
            self._clean_up(job)
        for handle in handles:
            handle.request()
        if cancelled:
            self.changed.emit()
        return cancelled

    def move(self, job_id, offset):
        """Move a waiting job ``offset`` places along its lane's line (negative is earlier); return whether it moved.

        It stops at the ends of the waiting jobs, and the running job never
        moves and is never passed: a job that is running, or gone, does not move.
        """
        with self._lock:
            lane, index = self._find(job_id)
            if lane is None or index is None:
                return False
            target = max(0, min(len(lane.queued) - 1, index + int(offset)))
            if target == index:
                return False
            lane.queued.insert(target, lane.queued.pop(index))
        self.changed.emit()
        return True

    def is_output_path_reserved(self, path):
        key = self._output_key(path)
        with self._lock:
            return key in self._reserved

    def _release(self, job):
        """Free the paths the job reserved; call with the lock held."""
        for path in job.reserved_paths:
            self._reserved.pop(self._output_key(path), None)

    def _clean_up(self, job):
        """Call the job's cleanup function, once, for whatever reason the job left the scheduler."""
        if job.cleanup is None:
            return
        with self._lock:
            if job.job_id in self._cleaned:
                return
            self._cleaned.add(job.job_id)
        try:
            job.cleanup()
        except Exception as error:  # A cleanup that fails must not stop the lane.
            print(f"Warning: Could not clean up after background job #{job.job_id}: {error}")

    def enqueue(
        self,
        command_name,
        description,
        payload,
        worker,
        output_path,
        allow_overwrite=False,
        *,
        lane=DEFAULT_LANE,
        lane_title=None,
        extra_paths=(),
        cleanup=None,
        on_started=None,
        on_finished=None,
        log_path=None,
        front=False,
        releases_request=True,
    ):
        """Reserve the outputs and put one immutable job at the end of its lane's line (the front, with front=True).

        ``output_path`` and ``extra_paths`` are everything the job writes: each
        is reserved until the job ends, and a second job may not reserve one.
        ``output_path`` must not exist, unless allow_overwrite. ``lane`` names
        the line the job goes in, made if it is new, with ``lane_title`` as its
        title for the card. ``cleanup`` is called once when the job leaves
        the scheduler, however it ends; ``on_started(job)`` on the GUI thread when
        it begins to run and ``on_finished(job, outcome, error)`` when its end is
        reported; ``log_path`` names a log the job writes. A worker with a second
        parameter named ``job`` is given the job's JobHandle. ``releases_request``
        (the default, for a job that runs in a lane) lets the command request that
        queued the job give up its place at the head of the command portal's line
        while only such jobs are left of it: the next request runs at once and this
        one ends, with its status and artifacts, when its jobs have.
        """
        from Viewer_Command_Portal import CURRENT
        command_context = CURRENT.get()
        output_path = os.path.abspath(os.fspath(output_path))
        paths = tuple(dict.fromkeys(
            os.path.abspath(os.fspath(path)) for path in (output_path, *extra_paths)
        ))
        with self._lock:
            if not self._accepting:
                raise RuntimeError(Message("The background job scheduler is shutting down."))
            if not allow_overwrite and os.path.exists(output_path):
                raise FileExistsError(Message("Output file already exists: {path}", path=output_path))
            for path in paths:
                if self._output_key(path) in self._reserved:
                    raise FileExistsError(Message(
                        "Output file is already reserved by a background job: {path}", path=path
                    ))

            target = self._lane(lane, lane_title)
            job_id = next(self._job_numbers)
            queue_position = len(target.queued) + (target.active is not None) + 1
            job = BackgroundJob(
                job_id=job_id,
                command_name=str(command_name),
                description=str(description),
                payload=payload,
                worker=worker,
                output_path=output_path,
                command_context=command_context,
                lane=target.name,
                reserved_paths=paths,
                cleanup=cleanup,
                on_started=on_started,
                on_finished=on_finished,
                log_path=None if log_path is None else os.fspath(log_path),
            )
            for path in paths:
                self._reserved[self._output_key(path)] = job_id
            if front:
                target.queued.insert(0, job)
                queue_position = 1 + (target.active is not None)
            else:
                target.queued.append(job)
            if command_context is not None:
                command_context.add_job(
                    job_id, output_path=output_path, kind=job.command_name, lane=target.name,
                    lane_title=None if target.title is None else str(target.title),
                    description=job.description, cancellable=True, releases_request=releases_request,
                )
            self._start(target)
            target.wake.notify()
        self._update_records()
        self.changed.emit()

        message = Message(
            "Queued background job #{job}: {command} (position {position}) -> {file}",
            job=job_id,
            command=job.command_name,
            position=queue_position,
            file=os.path.basename(output_path),
        )
        print(message)
        self._set_viewer_status(message)
        return job_id

    def shutdown(self, wait=SHUTDOWN_WAIT_SECONDS):
        """Close the lines: discard the waiting jobs and stop the running ones, within ``wait`` seconds.

        Each running job is told to stop and given up to ``wait`` seconds (all
        of them together) to reach a check point, where its worker deletes its
        partial files and the job ends. A job that has not stopped by then is
        left to its daemon thread, which ends with the process; its partial
        files are deleted here instead, the ones its worker named
        (JobHandle.note_partial), and its cleanup function is called, so that
        closing the Viewer in the middle of a write leaves no half-written file
        whatever the job's thread is doing. A file it renames into place
        afterwards cannot appear: the partial file it would rename is gone.
        """
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            discarded, handles, threads = [], [], []
            for lane in self._lanes.values():
                for job in lane.queued:
                    self._release(job)
                    discarded.append(job)
                lane.queued.clear()
                if lane.handle is not None:
                    handles.append(lane.handle)
                if lane.thread is not None:
                    threads.append(lane.thread)
                lane.wake.notify_all()
        for job in discarded:
            self._clean_up(job)
        for handle in handles:
            handle.request()
        deadline = time.monotonic() + max(0.0, wait)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            stuck = [(lane.active, lane.handle) for lane in self._lanes.values() if lane.active is not None]
        for job, handle in stuck:
            for path in handle.partial_paths() if handle is not None else ():
                try:
                    os.remove(path)
                except OSError:
                    pass
            self._clean_up(job)

    def _run_lane(self, lane):
        """The lane's thread: take the job at the front of the line, run it, and report how it ended."""
        while True:
            with self._lock:
                while self._accepting and not lane.queued:
                    lane.wake.wait()
                if not self._accepting:
                    return
                job = lane.queued.pop(0)
                handle = JobHandle(on_change=self._announce_change, call_in_gui=self._post_gui_call)
                lane.active = job
                lane.handle = handle
                lane.started = time.monotonic()
                lane.stopping = False

            started_at = time.perf_counter()
            self.job_started.emit(job)
            self.changed.emit()
            try:
                from Viewer_Command_Portal import bind
                with bound_to(handle), bind(job.command_context):
                    if _takes_handle(job.worker):
                        result = dict(job.worker(job.payload, job=handle))
                    else:
                        result = dict(job.worker(job.payload))
                elapsed = time.perf_counter() - started_at
                with self._lock:
                    accepting = self._accepting
                    if handle.requested:
                        self._not_stopped.add(job.job_id)
                if accepting:
                    self.job_succeeded.emit(job, result, elapsed)
            except JobCancelled:
                with self._lock:
                    accepting = self._accepting
                if accepting:
                    self.job_cancelled.emit(job, True)
            except Exception as error:
                elapsed = time.perf_counter() - started_at
                detail = traceback.format_exc()
                with self._lock:
                    accepting = self._accepting
                    if handle.log_path:
                        self._logs[job.job_id] = handle.log_path
                if accepting:
                    self.job_failed.emit(job, error, detail, elapsed)
            finally:
                with self._lock:
                    lane.active = None
                    lane.handle = None
                    lane.started = None
                    lane.stopping = False
                    self._release(job)
                    accepting = self._accepting
                self._clean_up(job)
                if accepting:
                    self.changed.emit()

    def _post_gui_call(self, function):
        """Have the GUI thread run function; return the GuiCall that says when it has."""
        call = GuiCall(function)
        self._gui_requested.emit(call)
        return call

    @QtCore.Slot(object)
    def _run_gui_call(self, call):
        call.run()

    def _announce_change(self):
        """A running job reported progress or a log: the card is told, once the signal is delivered."""
        if self._accepting:
            self.changed.emit()

    def _set_viewer_status(self, message):
        viewer = self._viewer_ref()
        if viewer is None or not hasattr(viewer, "set_background_job_status"):
            return
        viewer.set_background_job_status(message)

    @QtCore.Slot()
    def _update_records(self):
        """Bring each unfinished job's portal record up to date: its place, progress and whether it is stopping.

        The place is its position in its lane's line and the job ahead of it. A
        running job also gives its progress, and whether a stop request would
        do anything (not once it was told to stop). The portal is told of a
        record only when something in it changed.
        """
        with self._lock:
            lines = []
            for lane in self._lanes.values():
                line = list(self._line(lane))
                state = {}
                if lane.active is not None:
                    handle = lane.handle
                    progress = handle.progress_text if handle is not None else None
                    state = {"progress": None if progress is None else str(progress), "stopping": lane.stopping}
                lines.append((line, lane.active, state))
        for line, active, state in lines:
            for position, job in enumerate(line, 1):
                context = job.command_context
                if context is None:
                    continue
                fresh = {"position": position, "waiting_for": str(line[position - 2].job_id) if position > 1 else None}
                if job is active:
                    fresh.update(state, cancellable=not state["stopping"])
                changed = False
                for entry in context.record['jobs']:
                    if entry['job_id'] == str(job.job_id):
                        for key, value in fresh.items():
                            if key not in entry or entry[key] != value:
                                entry[key] = value
                                changed = True
                if changed:
                    context.portal.changed.emit(context.request_id)

    def _finished(self, job, outcome, error=None):
        """Tell the job's own on_finished, on the GUI thread, how it ended."""
        if job.on_finished is None:
            return
        try:
            job.on_finished(job, outcome, error)
        except Exception as problem:  # What a command does with the news must not stop the reports.
            print(f"Warning: Background job #{job.job_id}'s follow-up failed: {problem}")

    def _started(self, job):
        """Tell the job's own on_started, on the GUI thread, that it runs."""
        if job.on_started is None:
            return
        try:
            job.on_started(job)
        except Exception as problem:
            print(f"Warning: Background job #{job.job_id}'s start-up failed: {problem}")

    @QtCore.Slot(object)
    def _report_started(self, job):
        if job.command_context is not None:
            job.command_context.job_event(job.job_id, "running")
        message = Message(
            "Running background job #{job}: {description}", job=job.job_id, description=job.description
        )
        print(message)
        self._set_viewer_status(message)
        self._started(job)

    @QtCore.Slot(object, object, float)
    def _report_succeeded(self, job, result, elapsed):
        detail = result.get("message") or Message("Saved {path}", path=job.output_path)
        with self._lock:
            not_stopped = job.job_id in self._not_stopped
            self._not_stopped.discard(job.job_id)
        # A stop request is honoured at the worker's check points: past its last one, the job completes.
        if not_stopped:
            message = Message(
                "Background job #{job} was told to stop but had no step left to stop at, and completed in {seconds}s: {detail}",
                job=job.job_id, seconds=f"{elapsed:.1f}", detail=detail,
            )
        else:
            message = Message(
                "Background job #{job} completed in {seconds}s: {detail}",
                job=job.job_id, seconds=f"{elapsed:.1f}", detail=detail,
            )
        print(f"\n{message}")
        self._set_viewer_status(message)
        if not_stopped and job.command_context is not None:
            for entry in job.command_context.record['jobs']:
                if entry['job_id'] == str(job.job_id):
                    entry['stop_requested'] = True

        reveal_directory = result.get("reveal_directory")
        # A job asks for its folder to be shown. Only a Viewer that is there and shows windows
        # has a user to show it to: one that is gone (or closing) has none.
        if reveal_directory:
            viewer = self._viewer_ref()
            if viewer is not None and viewer_shows_windows(viewer):
                try:
                    open_in_file_manager(reveal_directory)
                except Exception as error:
                    print(f"Could not reveal output directory: {error}")

        if job.command_context is not None:
            job.command_context.job_event(job.job_id, "succeeded", message, [job.output_path])
        self._finished(job, "succeeded")

    @QtCore.Slot(object, bool)
    def _report_cancelled(self, job, started):
        if started:
            message = Message("Background job #{job} was stopped before it finished: {description}",
                              job=job.job_id, description=job.description)
        else:
            message = Message("Background job #{job} was cancelled before it started: {description}",
                              job=job.job_id, description=job.description)
        print(message)
        self._set_viewer_status(message)
        if job.command_context is not None:
            job.command_context.job_event(job.job_id, 'cancelled', str(message))
        self._finished(job, "cancelled")

    @QtCore.Slot(object, object, str, float)
    def _report_failed(self, job, error, detail, elapsed):
        """Report a failed job: its error shows translated, and MCP clients get str(error)."""
        message = Message(
            "Background job #{job} failed after {seconds}s ({command}): {error}",
            job=job.job_id,
            seconds=f"{elapsed:.1f}",
            command=job.command_name,
            error=error,
        )
        print(f"\n{message}")
        print(detail)
        self._set_viewer_status(message)
        with self._lock:
            log_path = self._logs.pop(job.job_id, None) or job.log_path
            self._failures.append(FailedJob(job.job_id, job.description, job.lane, error, log_path))
            del self._failures[:-KEPT_FAILURES]
        if job.command_context is not None:
            job.command_context.portal.append_output(job.command_context.request_id, 'stderr', detail)
            job.command_context.job_event(job.job_id, 'failed', str(error))
        self.changed.emit()
        self._finished(job, "failed", error)
