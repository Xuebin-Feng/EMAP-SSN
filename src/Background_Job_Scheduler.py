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
import queue
import threading
import time
import traceback
from typing import Any, Callable, Mapping
import weakref

from PySide6 import QtCore
from desktop.Desktop_App import open_in_file_manager
from desktop.Viewer_Window import viewer_shows_windows
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


class BackgroundJobScheduler(QtCore.QObject):
    """Run mixed command jobs in strict FIFO order on one daemon thread."""

    job_started = QtCore.Signal(object)
    job_succeeded = QtCore.Signal(object, object, float)
    # The error itself, not its text, so a Message it holds shows translated.
    job_failed = QtCore.Signal(object, object, str, float)
    job_cancelled = QtCore.Signal(object)

    _STOP = object()

    def __init__(self, viewer, job_numbers=None):
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._accepting = True
        self._active_job = None
        self._outstanding_count = 0
        # Job numbers: the Viewer's command queue draws from the same count, so
        # every job the Viewer lists has a number of its own.
        self._job_numbers = job_numbers if job_numbers is not None else itertools.count(1)
        self._reserved_output_paths = set()
        # The jobs not yet finished, in the order queued, and those cancelled before they started.
        self._waiting = {}
        self._cancelled = set()

        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self.job_started.connect(self._report_started, queued)
        self.job_succeeded.connect(self._report_succeeded, queued)
        self.job_failed.connect(self._report_failed, queued)
        self.job_cancelled.connect(self._report_cancelled, queued)

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
            return self._outstanding_count

    def listing(self):
        """The jobs not yet finished, oldest first, as (job, running) pairs."""
        with self._lock:
            active = self._active_job
            return [(job, job is active) for job in self._waiting.values()]

    def cancel(self, job_id):
        """Cancel a job that has not started; return whether there was one.

        A job that writes its file already is left to finish.
        """
        with self._lock:
            job = self._waiting.get(job_id)
            if job is None or job is self._active_job or job_id in self._cancelled:
                return False
            self._cancelled.add(job_id)
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
        """Reserve an output and append one immutable job to the FIFO queue."""
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
            queue_position = self._outstanding_count + 1
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
            self._outstanding_count += 1
            self._waiting[job_id] = job
            if command_context is not None:
                command_context.add_job(job_id, output_path=output_path)
            self._jobs.put(job)

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
        """Discard queued jobs without waiting for the daemon worker."""
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False

        while True:
            try:
                item = self._jobs.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BackgroundJob):
                with self._lock:
                    self._reserved_output_paths.discard(
                        self._output_key(item.output_path)
                    )
                    self._outstanding_count -= 1
                    self._waiting.pop(item.job_id, None)
            self._jobs.task_done()
        self._jobs.put(self._STOP)

    def _worker_loop(self):
        while True:
            job = self._jobs.get()
            if job is self._STOP:
                self._jobs.task_done()
                return

            with self._lock:
                cancelled = job.job_id in self._cancelled
                if not self._accepting or cancelled:
                    self._reserved_output_paths.discard(
                        self._output_key(job.output_path)
                    )
                    self._outstanding_count -= 1
                    self._waiting.pop(job.job_id, None)
                    self._cancelled.discard(job.job_id)
                    self._jobs.task_done()
                    if cancelled and self._accepting:
                        self.job_cancelled.emit(job)
                    continue
                self._active_job = job

            started_at = time.perf_counter()
            self.job_started.emit(job)
            try:
                from Viewer_Command_Portal import bind
                with bind(job.command_context):
                    result = dict(job.worker(job.payload))
                elapsed = time.perf_counter() - started_at
                with self._lock:
                    accepting = self._accepting
                if accepting:
                    self.job_succeeded.emit(job, result, elapsed)
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
                    self._reserved_output_paths.discard(
                        self._output_key(job.output_path)
                    )
                    self._outstanding_count -= 1
                    self._waiting.pop(job.job_id, None)
                self._jobs.task_done()

    def _set_viewer_status(self, message):
        viewer = self._viewer_ref()
        if viewer is None or not hasattr(viewer, "set_background_job_status"):
            return
        viewer.set_background_job_status(message)

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

    @QtCore.Slot(object)
    def _report_cancelled(self, job):
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
