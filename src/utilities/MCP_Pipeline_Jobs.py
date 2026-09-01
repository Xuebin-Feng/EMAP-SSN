# Copyright 2026 Xuebin Feng
# Author affiliation: University of Toronto
# SPDX-License-Identifier: Apache-2.0

"""Server-local FIFO execution for MCP-started pipeline tools."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid

from utilities.Tool_Execution import (
    ToolInvocation,
    prepare_headless_invocation,
    resolve_tool_directories,
)


TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled"})
JOB_STATUSES = (
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
)


class PipelineJobError(RuntimeError):
    """Raised when an MCP pipeline job request cannot be completed."""


class PipelineQueueFullError(PipelineJobError):
    """Raised when the server-local pending queue is full."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class PipelineJob:
    job_id: str
    tool_id: str
    invocation: ToolInvocation
    created_at: str
    stdout_path: str
    stderr_path: str
    output_locations: dict[str, str]
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    failure_message: str | None = None
    cancellation_requested: bool = False
    completion: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
        compare=False,
    )


class PipelineJobManager:
    """Own one bounded FIFO and every subprocess launched from it."""

    def __init__(
        self,
        project_root,
        *,
        python_executable=None,
        max_pending=16,
        history_limit=100,
        termination_grace=5.0,
        temporary_parent=None,
    ):
        self.project_root = os.path.abspath(os.fspath(project_root))
        self.python_executable = python_executable or sys.executable
        self.max_pending = int(max_pending)
        self.history_limit = int(history_limit)
        self.termination_grace = float(termination_grace)
        if self.max_pending < 1:
            raise ValueError("max_pending must be at least 1.")
        if self.history_limit < 1:
            raise ValueError("history_limit must be at least 1.")
        if self.termination_grace < 0:
            raise ValueError("termination_grace cannot be negative.")

        parent = (
            os.path.abspath(os.fspath(temporary_parent))
            if temporary_parent is not None
            else os.path.join(
                tempfile.gettempdir(),
                "sequence_similarity_network_viewer",
                "mcp_jobs",
            )
        )
        os.makedirs(parent, mode=0o700, exist_ok=True)
        self.temporary_root = tempfile.mkdtemp(prefix="server-", dir=parent)
        try:
            os.chmod(self.temporary_root, 0o700)
        except OSError:
            pass

        self._jobs: dict[str, PipelineJob] = {}
        self._pending: deque[str] = deque()
        self._terminal_order: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._termination_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._current_job_id: str | None = None
        self._current_process: asyncio.subprocess.Process | None = None
        self._closed = False

    async def start(self):
        if self._closed:
            raise PipelineJobError("The pipeline job manager is closed.")
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(),
                name="SSNMCPPipelineFIFO",
            )

    async def submit(self, tool_id, settings_source):
        await self.start()
        async with self._lock:
            if self._closed:
                raise PipelineJobError("The pipeline job manager is closed.")
            if len(self._pending) >= self.max_pending:
                raise PipelineQueueFullError(
                    f"The pipeline queue already has {self.max_pending} pending jobs."
                )

        job_id = str(uuid.uuid4())
        job_directory = os.path.join(self.temporary_root, job_id)
        os.makedirs(job_directory, mode=0o700)
        try:
            invocation = prepare_headless_invocation(
                tool_id,
                settings_source,
                self.project_root,
                python_executable=self.python_executable,
                snapshot_directory=job_directory,
            )
            output_locations = resolve_tool_directories(
                invocation.tool,
                invocation.settings_path,
                self.project_root,
                output_only=True,
            )
            stdout_path = os.path.join(job_directory, "stdout.log")
            stderr_path = os.path.join(job_directory, "stderr.log")
            for path in (stdout_path, stderr_path):
                with open(path, "wb"):
                    pass
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
            job = PipelineJob(
                job_id=job_id,
                tool_id=invocation.tool.tool_id,
                invocation=invocation,
                created_at=_utc_now(),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                output_locations=output_locations,
            )
        except Exception:
            self._remove_job_directory(job_directory)
            raise

        async with self._lock:
            if self._closed:
                self._remove_job_directory(job_directory)
                raise PipelineJobError("The pipeline job manager is closed.")
            if len(self._pending) >= self.max_pending:
                self._remove_job_directory(job_directory)
                raise PipelineQueueFullError(
                    f"The pipeline queue already has {self.max_pending} pending jobs."
                )
            self._jobs[job_id] = job
            self._pending.append(job_id)
            payload = self._job_payload_locked(job)
            self._wake.set()
            return payload

    async def list_jobs(self, *, limit=100):
        try:
            limit = int(limit)
        except (TypeError, ValueError) as error:
            raise PipelineJobError("Job list limit must be an integer.") from error
        if not 1 <= limit <= 100:
            raise PipelineJobError("Job list limit must be between 1 and 100.")
        async with self._lock:
            jobs = list(self._jobs.values())[-limit:]
            return [self._job_payload_locked(job) for job in reversed(jobs)]

    async def get_job(self, job_id):
        async with self._lock:
            job = self._require_job_locked(job_id)
            return self._job_payload_locked(job)

    async def read_log(self, job_id, stream, *, offset=0, limit=65536):
        if stream not in {"stdout", "stderr"}:
            raise PipelineJobError("Log stream must be 'stdout' or 'stderr'.")
        try:
            offset = int(offset)
            limit = int(limit)
        except (TypeError, ValueError) as error:
            raise PipelineJobError("Log offset and limit must be integers.") from error
        if offset < 0:
            raise PipelineJobError("Log offset cannot be negative.")
        if not 1 <= limit <= 262144:
            raise PipelineJobError("Log limit must be between 1 and 262144 bytes.")
        async with self._lock:
            job = self._require_job_locked(job_id)
            path = job.stdout_path if stream == "stdout" else job.stderr_path
        return await asyncio.to_thread(
            self._read_log_page,
            job_id,
            stream,
            path,
            offset,
            limit,
        )

    async def cancel(self, job_id):
        process = None
        queued_payload = None
        evicted_directories = []
        async with self._lock:
            job = self._require_job_locked(job_id)
            if job.status in TERMINAL_JOB_STATUSES:
                return self._job_payload_locked(job)
            job.cancellation_requested = True
            if job.status == "queued":
                try:
                    self._pending.remove(job.job_id)
                except ValueError:
                    pass
                job.status = "cancelled"
                job.finished_at = _utc_now()
                evicted_directories = self._finish_job_locked(job)
                queued_payload = self._job_payload_locked(job)
            else:
                job.status = "cancelling"
                if self._current_job_id == job.job_id:
                    process = self._current_process

        for directory in evicted_directories:
            await asyncio.to_thread(self._remove_job_directory, directory)
        if queued_payload is not None:
            return queued_payload

        if process is not None:
            await self._terminate_process_tree(process)
        try:
            await asyncio.wait_for(
                job.completion.wait(),
                timeout=self.termination_grace + 5.0,
            )
        except TimeoutError:
            pass
        return await self.get_job(job_id)

    async def wait_for_terminal(self, job_id, *, timeout=30.0):
        async with self._lock:
            job = self._require_job_locked(job_id)
            completion = job.completion
        await asyncio.wait_for(completion.wait(), timeout=timeout)
        return await self.get_job(job_id)

    async def close(self):
        evicted_directories = []
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            for job_id in list(self._pending):
                job = self._jobs[job_id]
                job.cancellation_requested = True
                job.status = "cancelled"
                job.finished_at = _utc_now()
                evicted_directories.extend(self._finish_job_locked(job))
            self._pending.clear()
            process = self._current_process
            if self._current_job_id is not None:
                current = self._jobs[self._current_job_id]
                current.cancellation_requested = True
                if current.status not in TERMINAL_JOB_STATUSES:
                    current.status = "cancelling"
            self._wake.set()

        for directory in evicted_directories:
            await asyncio.to_thread(self._remove_job_directory, directory)
        if process is not None:
            await self._terminate_process_tree(process)
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(
                    self._worker_task,
                    timeout=self.termination_grace + 5.0,
                )
            except TimeoutError:
                self._worker_task.cancel()
                try:
                    await self._worker_task
                except asyncio.CancelledError:
                    pass
        await asyncio.to_thread(self._remove_job_directory, self.temporary_root)

    async def _worker(self):
        while True:
            await self._wake.wait()
            async with self._lock:
                if not self._pending:
                    self._wake.clear()
                    if self._closed:
                        return
                    continue
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                if job.status != "queued":
                    continue
                job.status = "running"
                job.started_at = _utc_now()
                self._current_job_id = job_id

            process = None
            evicted_directories = []
            try:
                with open(job.stdout_path, "ab", buffering=0) as stdout_handle, open(
                    job.stderr_path,
                    "ab",
                    buffering=0,
                ) as stderr_handle:
                    process = await asyncio.create_subprocess_exec(
                        *job.invocation.argv,
                        cwd=job.invocation.cwd,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        **self._process_group_options(),
                    )
                    async with self._lock:
                        self._current_process = process
                        cancel_now = job.cancellation_requested
                        if cancel_now:
                            job.status = "cancelling"
                    if cancel_now:
                        await self._terminate_process_tree(process)
                    return_code = await process.wait()
                async with self._lock:
                    job.exit_code = return_code
                    job.finished_at = _utc_now()
                    if job.cancellation_requested:
                        job.status = "cancelled"
                    elif return_code == 0:
                        job.status = "succeeded"
                    else:
                        job.status = "failed"
                        job.failure_message = (
                            f"The pipeline process exited with code {return_code}."
                        )
                    evicted_directories = self._finish_job_locked(job)
            except asyncio.CancelledError:
                if process is not None:
                    await self._terminate_process_tree(process)
                async with self._lock:
                    job.cancellation_requested = True
                    job.status = "cancelled"
                    job.finished_at = _utc_now()
                    evicted_directories = self._finish_job_locked(job)
                raise
            except Exception as error:
                async with self._lock:
                    job.status = (
                        "cancelled" if job.cancellation_requested else "failed"
                    )
                    job.finished_at = _utc_now()
                    job.failure_message = (
                        None
                        if job.cancellation_requested
                        else f"Could not run the pipeline process: {error}"
                    )
                    evicted_directories = self._finish_job_locked(job)
            finally:
                async with self._lock:
                    if self._current_job_id == job_id:
                        self._current_job_id = None
                        self._current_process = None
                    if self._pending:
                        self._wake.set()
                    elif self._closed:
                        self._wake.set()
                for directory in evicted_directories:
                    await asyncio.to_thread(self._remove_job_directory, directory)

    def _job_payload_locked(self, job):
        queue_position = None
        if job.status == "queued":
            try:
                queue_position = list(self._pending).index(job.job_id) + 1
            except ValueError:
                queue_position = None
        return {
            "job_id": job.job_id,
            "tool_id": job.tool_id,
            "status": job.status,
            "queue_position": queue_position,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "exit_code": job.exit_code,
            "failure_message": job.failure_message,
            "cancellation_requested": job.cancellation_requested,
            "settings_snapshot": job.invocation.settings_path,
            "stdout_log": job.stdout_path,
            "stderr_log": job.stderr_path,
            "output_locations": dict(job.output_locations),
        }

    def _require_job_locked(self, job_id):
        try:
            return self._jobs[str(job_id)]
        except KeyError as error:
            raise PipelineJobError(f"Unknown pipeline job ID: {job_id}") from error

    def _finish_job_locked(self, job):
        if not job.completion.is_set():
            job.completion.set()
            self._terminal_order.append(job.job_id)
        evicted = []
        while len(self._terminal_order) > self.history_limit:
            oldest_id = self._terminal_order.popleft()
            oldest = self._jobs.pop(oldest_id, None)
            if oldest is not None:
                evicted.append(os.path.dirname(oldest.stdout_path))
        return evicted

    @staticmethod
    def _read_log_page(job_id, stream, path, offset, limit):
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as handle:
                handle.seek(min(offset, size))
                data = handle.read(limit)
        except OSError as error:
            raise PipelineJobError(f"Could not read the {stream} log: {error}") from error
        next_offset = min(offset, size) + len(data)
        return {
            "job_id": job_id,
            "stream": stream,
            "offset": min(offset, size),
            "next_offset": next_offset,
            "size": size,
            "eof": next_offset >= size,
            "text": data.decode("utf-8", errors="replace"),
        }

    @staticmethod
    def _process_group_options():
        if os.name == "nt":
            return {
                "creationflags": getattr(
                    subprocess,
                    "CREATE_NEW_PROCESS_GROUP",
                    0,
                )
            }
        return {"start_new_session": True}

    async def _terminate_process_tree(self, process):
        async with self._termination_lock:
            if process.returncode is not None:
                return
            if os.name == "nt":
                taskkill_code = await self._taskkill(process.pid, force=False)
                if taskkill_code != 0 and process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        return
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return
            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=self.termination_grace,
                )
                return
            except TimeoutError:
                pass
            if os.name == "nt":
                await self._taskkill(process.pid, force=True)
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        return
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    return
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                pass

    @staticmethod
    async def _taskkill(pid, *, force):
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await process.wait()

    def _remove_job_directory(self, path):
        target = os.path.abspath(os.fspath(path))
        root = os.path.abspath(self.temporary_root)
        try:
            within_root = os.path.commonpath((root, target)) == root
        except ValueError:
            within_root = False
        if target != root and not within_root:
            raise PipelineJobError(
                f"Refusing to remove an MCP job directory outside {root}."
            )
        shutil.rmtree(target, ignore_errors=True)


__all__ = [
    "JOB_STATUSES",
    "PipelineJobError",
    "PipelineJobManager",
    "PipelineQueueFullError",
    "TERMINAL_JOB_STATUSES",
]
