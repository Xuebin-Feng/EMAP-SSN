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

"""One sequential, viewer-owned queue for expensive command artifacts."""

from __future__ import annotations

from dataclasses import dataclass
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
from utilities.Job_Cancellation import CancelToken, JobCancelled, bound_to
from utilities.Localization import Message


Worker = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class BackgroundJob:
    job_id: int
    command_name: str
    description: str
    payload: Any
    worker: Worker
    output_path: str
    command_context: Any = None


@dataclass(frozen=True)
class JobStatus:
    """One unfinished job as snapshot() reports it.

    ``position`` is its place in the line, the running job being 1;
    ``started_at`` is the running job's time.monotonic() start, None for a
    waiting one; ``stopping`` is true for the running job once it was told to stop.
    """
    job: BackgroundJob
    running: bool
    position: int
    started_at: float | None = None
    stopping: bool = False


class BackgroundJobScheduler(QtCore.QObject):
    """Run mixed command jobs one at a time on one daemon thread, in the order of the line.

    The line is a list the GUI thread may rearrange (move) while the thread
    takes jobs from its front: every change to the list, and the choice of the
    next job, happen under one lock, so a job is either still in the line,
    where it can be moved or removed, or running, where it cannot be moved and
    can only be told to stop (cancel). A running job stops at the next check
    point of its worker (utilities.Job_Cancellation.raise_if_cancelled); the
    worker deletes what it had written, and the job ends as cancelled.
    """

    job_started = QtCore.Signal(object)
    job_succeeded = QtCore.Signal(object, object, float)
    # The error itself, not its text, so a Message it holds shows translated.
    job_failed = QtCore.Signal(object, object, str, float)
    # The job, and whether it had started: a job removed from the line, or one stopped while it ran.
    job_cancelled = QtCore.Signal(object, bool)
    # The line or the running job changed: sent once the new state stands, for whatever lists the jobs.
    changed = QtCore.Signal()

    def __init__(self, viewer, job_numbers=None):
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._accepting = True
        # The jobs waiting, in the order they will run; the running one is not in it.
        self._queued = []
        self._active_job = None
        self._active_token = None
        self._active_started = None
        self._stopping = False
        # Job numbers: the Viewer's command queue draws from the same count, so
        # every job the Viewer lists has a number of its own.
        self._job_numbers = job_numbers if job_numbers is not None else itertools.count(1)
        self._reserved_output_paths = set()

        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self.job_started.connect(self._report_started, queued)
        self.job_succeeded.connect(self._report_succeeded, queued)
        self.job_failed.connect(self._report_failed, queued)
        self.job_cancelled.connect(self._report_cancelled, queued)
        self.changed.connect(self._update_places, queued)

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="SSN-Background-Jobs",
            daemon=True,
        )
        self._thread.start()

    @staticmethod
    def _output_key(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @property
    def queue_depth(self):
        """Return the number of running and queued jobs."""
        with self._lock:
            return len(self._queued) + (self._active_job is not None)

    def _line(self):
        """The unfinished jobs in the order of the line, the running one first; call with the lock held."""
        return ([self._active_job] if self._active_job is not None else []) + self._queued

    def listing(self):
        """The jobs not yet finished, in the order of the line, as (job, running) pairs."""
        with self._lock:
            return [(job, job is self._active_job) for job in self._line()]

    def snapshot(self):
        """The unfinished jobs in the order of the line, as JobStatus entries, read at one moment."""
        with self._lock:
            entries = []
            if self._active_job is not None:
                entries.append(JobStatus(self._active_job, True, 1, self._active_started, self._stopping))
            first = len(entries) + 1
            entries.extend(JobStatus(job, False, first + index) for index, job in enumerate(self._queued))
            return entries

    def cancel(self, job_id):
        """Take a job out of the line, or tell the running one to stop; return whether there was one to.

        A waiting job is gone at once: its output is released and it is reported
        cancelled. The running job is told to stop and ends at the next check
        point of its worker, which deletes its unfinished file; it is reported
        cancelled then. A worker with no check point finishes its file as it
        would have, and is reported as completed. A job already told to stop
        is not told again.
        """
        removed = None
        with self._lock:
            index = next((i for i, job in enumerate(self._queued) if job.job_id == job_id), None)
            if index is not None:
                removed = self._queued.pop(index)
                self._reserved_output_paths.discard(self._output_key(removed.output_path))
            elif (self._active_job is not None and self._active_job.job_id == job_id
                  and not self._stopping):
                self._stopping = True
                self._active_token.request()
            else:
                return False
        if removed is not None:
            self.job_cancelled.emit(removed, False)
        self.changed.emit()
        return True

    def move(self, job_id, offset):
        """Move a waiting job ``offset`` places along the line (negative is earlier); return whether it moved.

        It stops at the ends of the waiting jobs, and the running job never
        moves and is never passed: a job that is running, or gone, does not move.
        """
        with self._lock:
            index = next((i for i, job in enumerate(self._queued) if job.job_id == job_id), None)
            if index is None:
                return False
            target = max(0, min(len(self._queued) - 1, index + int(offset)))
            if target == index:
                return False
            self._queued.insert(target, self._queued.pop(index))
        self.changed.emit()
        return True

    def is_output_path_reserved(self, path):
        key = self._output_key(path)
        with self._lock:
            return key in self._reserved_output_paths

    def enqueue(
        self,
        command_name,
        description,
        payload,
        worker,
        output_path,
        allow_overwrite=False,
    ):
        """Reserve an output and put one immutable job at the end of the line."""
        from Viewer_Command_Portal import CURRENT
        command_context = CURRENT.get()
        output_path = os.path.abspath(os.fspath(output_path))
        output_key = self._output_key(output_path)
        with self._lock:
            if not self._accepting:
                raise RuntimeError(Message("The background job scheduler is shutting down."))
            if not allow_overwrite and os.path.exists(output_path):
                raise FileExistsError(Message("Output file already exists: {path}", path=output_path))
            if output_key in self._reserved_output_paths:
                raise FileExistsError(Message(
                    "Output file is already reserved by a background job: {path}", path=output_path
                ))

            job_id = next(self._job_numbers)
            queue_position = len(self._queued) + (self._active_job is not None) + 1
            job = BackgroundJob(
                job_id=job_id,
                command_name=str(command_name),
                description=str(description),
                payload=payload,
                worker=worker,
                output_path=output_path,
                command_context=command_context,
            )
            self._reserved_output_paths.add(output_key)
            self._queued.append(job)
            if command_context is not None:
                command_context.add_job(job_id, output_path=output_path)
            self._wake.notify()
        self._update_places()
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

    def shutdown(self):
        """Discard queued jobs and tell the running one to stop, without waiting for the daemon worker."""
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            for job in self._queued:
                self._reserved_output_paths.discard(self._output_key(job.output_path))
            self._queued.clear()
            if self._active_token is not None:
                self._active_token.request()
            self._wake.notify_all()

    def _worker_loop(self):
        while True:
            with self._lock:
                while self._accepting and not self._queued:
                    self._wake.wait()
                if not self._accepting:
                    return
                job = self._queued.pop(0)
                token = CancelToken()
                self._active_job = job
                self._active_token = token
                self._active_started = time.monotonic()
                self._stopping = False

            started_at = time.perf_counter()
            self.job_started.emit(job)
            self.changed.emit()
            try:
                from Viewer_Command_Portal import bind
                with bound_to(token), bind(job.command_context):
                    result = dict(job.worker(job.payload))
                elapsed = time.perf_counter() - started_at
                with self._lock:
                    accepting = self._accepting
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
                if accepting:
                    self.job_failed.emit(job, error, detail, elapsed)
            finally:
                with self._lock:
                    self._active_job = None
                    self._active_token = None
                    self._active_started = None
                    self._stopping = False
                    self._reserved_output_paths.discard(
                        self._output_key(job.output_path)
                    )
                    accepting = self._accepting
                if accepting:
                    self.changed.emit()

    def _set_viewer_status(self, message):
        viewer = self._viewer_ref()
        if viewer is None or not hasattr(viewer, "set_background_job_status"):
            return
        viewer.set_background_job_status(message)

    @QtCore.Slot()
    def _update_places(self):
        """Each unfinished job's place in the line, in its portal record: position and the job ahead of it."""
        with self._lock:
            line = list(self._line())
        for position, job in enumerate(line, 1):
            context = job.command_context
            if context is None:
                continue
            for entry in context.record['jobs']:
                if entry['job_id'] == str(job.job_id):
                    entry['position'] = position
                    entry['waiting_for'] = str(line[position - 2].job_id) if position > 1 else None

    @QtCore.Slot(object)
    def _report_started(self, job):
        if job.command_context is not None:
            job.command_context.job_event(job.job_id, "running")
        message = Message(
            "Running background job #{job}: {description}", job=job.job_id, description=job.description
        )
        print(message)
        self._set_viewer_status(message)

    @QtCore.Slot(object, object, float)
    def _report_succeeded(self, job, result, elapsed):
        detail = result.get("message") or Message("Saved {path}", path=job.output_path)
        message = Message(
            "Background job #{job} completed in {seconds}s: {detail}",
            job=job.job_id,
            seconds=f"{elapsed:.1f}",
            detail=detail,
        )
        print(f"\n{message}")
        self._set_viewer_status(message)

        reveal_directory = result.get("reveal_directory")
        # print and label ask for their folder to be shown; where no user is at the Viewer, none is.
        if reveal_directory:
            viewer = self._viewer_ref()
            if viewer is None or viewer_shows_windows(viewer):
                try:
                    open_in_file_manager(reveal_directory)
                except Exception as error:
                    print(f"Could not reveal output directory: {error}")

        if job.command_context is not None:
            job.command_context.job_event(job.job_id, "succeeded", message, [job.output_path])

    @QtCore.Slot(object, bool)
    def _report_cancelled(self, job, started):
        if started:
            message = Message("Background job #{job} was stopped before it finished; no file was written: {description}",
                              job=job.job_id, description=job.description)
        else:
            message = Message("Background job #{job} was cancelled before it started: {description}",
                              job=job.job_id, description=job.description)
        print(message)
        self._set_viewer_status(message)
        if job.command_context is not None:
            job.command_context.job_event(job.job_id, 'cancelled', str(message))

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
        if job.command_context is not None:
            job.command_context.portal.append_output(job.command_context.request_id, 'stderr', detail)
            job.command_context.job_event(job.job_id, 'failed', str(error))
