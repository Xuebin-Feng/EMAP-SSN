# Copyright 2026 Xuebin Feng
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
import os
import queue
import threading
import time
import traceback
from typing import Any, Callable, Mapping
import weakref

from PySide6 import QtCore
from utilities.Application_Windows import open_in_file_manager


Worker = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class BackgroundJob:
    job_id: int
    command_name: str
    description: str
    payload: Any
    worker: Worker
    output_path: str


class BackgroundJobScheduler(QtCore.QObject):
    """Run mixed command jobs in strict FIFO order on one daemon thread."""

    job_started = QtCore.Signal(object)
    job_succeeded = QtCore.Signal(object, object, float)
    job_failed = QtCore.Signal(object, str, str, float)

    _STOP = object()

    def __init__(self, viewer):
        super().__init__()
        self._viewer_ref = weakref.ref(viewer)
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._accepting = True
        self._active_job = None
        self._outstanding_count = 0
        self._next_job_id = 1
        self._reserved_output_paths = set()

        queued = QtCore.Qt.ConnectionType.QueuedConnection
        self.job_started.connect(self._report_started, queued)
        self.job_succeeded.connect(self._report_succeeded, queued)
        self.job_failed.connect(self._report_failed, queued)

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
        output_path = os.path.abspath(os.fspath(output_path))
        output_key = self._output_key(output_path)
        with self._lock:
            if not self._accepting:
                raise RuntimeError("The background job scheduler is shutting down.")
            if not allow_overwrite and os.path.exists(output_path):
                raise FileExistsError(f"Output file already exists: {output_path}")
            if output_key in self._reserved_output_paths:
                raise FileExistsError(
                    f"Output file is already reserved by a background job: {output_path}"
                )

            job_id = self._next_job_id
            self._next_job_id += 1
            queue_position = self._outstanding_count + 1
            job = BackgroundJob(
                job_id=job_id,
                command_name=str(command_name),
                description=str(description),
                payload=payload,
                worker=worker,
                output_path=output_path,
            )
            self._reserved_output_paths.add(output_key)
            self._outstanding_count += 1
            self._jobs.put(job)

        message = (
            f"Queued background job #{job_id}: {job.command_name} "
            f"(position {queue_position}) -> {os.path.basename(output_path)}"
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
            self._jobs.task_done()
        self._jobs.put(self._STOP)

    def _worker_loop(self):
        while True:
            job = self._jobs.get()
            if job is self._STOP:
                self._jobs.task_done()
                return

            with self._lock:
                if not self._accepting:
                    self._reserved_output_paths.discard(
                        self._output_key(job.output_path)
                    )
                    self._outstanding_count -= 1
                    self._jobs.task_done()
                    continue
                self._active_job = job

            started_at = time.perf_counter()
            self.job_started.emit(job)
            try:
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
                    self.job_failed.emit(job, str(error), detail, elapsed)
            finally:
                with self._lock:
                    self._active_job = None
                    self._reserved_output_paths.discard(
                        self._output_key(job.output_path)
                    )
                    self._outstanding_count -= 1
                self._jobs.task_done()

    def _set_viewer_status(self, message):
        viewer = self._viewer_ref()
        if viewer is None or not hasattr(viewer, "console_text"):
            return
        viewer.console_text.text = str(message)
        if hasattr(viewer, "update_console_background"):
            viewer.update_console_background()

    @QtCore.Slot(object)
    def _report_started(self, job):
        message = f"Running background job #{job.job_id}: {job.description}"
        print(message)
        self._set_viewer_status(message)

    @QtCore.Slot(object, object, float)
    def _report_succeeded(self, job, result, elapsed):
        detail = result.get("message") or f"Saved {job.output_path}"
        message = (
            f"Background job #{job.job_id} completed in {elapsed:.1f}s: {detail}"
        )
        print(f"\n{message}")
        self._set_viewer_status(message)

        reveal_directory = result.get("reveal_directory")
        if reveal_directory:
            try:
                open_in_file_manager(reveal_directory)
            except Exception as error:
                print(f"Could not reveal output directory: {error}")

    @QtCore.Slot(object, str, str, float)
    def _report_failed(self, job, error, detail, elapsed):
        message = (
            f"Background job #{job.job_id} failed after {elapsed:.1f}s "
            f"({job.command_name}): {error}"
        )
        print(f"\n{message}")
        print(detail)
        self._set_viewer_status(message)
