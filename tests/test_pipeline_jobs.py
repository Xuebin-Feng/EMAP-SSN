"""Pipeline job processes (mcp_server.pipeline.Pipeline_Jobs).

Runs real processes. Cancelling a job, or closing the manager while a job runs,
must end every process the job started, not only the one the manager launched.
On Windows that one is usually a venv's python.exe redirector, so a tool's own
children run two levels below it.
"""
import asyncio
import json
import pathlib
import sys
import tempfile
import time
import unittest
from unittest import mock

import psutil


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager  # noqa: E402
from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    ToolInvocation,
    create_settings_snapshot,
    get_tool_spec,
)


TERMINATION_GRACE = 1.0

# The child prints its own PID because behind a venv redirector it is not the
# PID Popen reports, and the file appears only once the whole tree is running.
SPAWNING_JOB = """
import json
import os
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", "import os, time; print(os.getpid(), flush=True); time.sleep(60)"],
    stdout=subprocess.PIPE,
)
pids = {"job": os.getpid(), "child": child.pid, "sleeper": int(child.stdout.readline())}
with open(sys.argv[1] + ".partial", "w", encoding="utf-8") as handle:
    json.dump(pids, handle)
os.replace(sys.argv[1] + ".partial", sys.argv[1])
time.sleep(60)
""".lstrip()


def _alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _describe(process):
    try:
        return f"{process.pid} {' '.join(process.cmdline())}"
    except psutil.Error:
        return str(process.pid)


def _survivors(processes, timeout):
    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in processes if _alive(process)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


def _kill(processes):
    # psutil refuses to signal a PID that a newer process has reused.
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    _survivors(processes, 5.0)


class JobProcessTreeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.temporary_path = pathlib.Path(temporary.name)
        self.script = self.temporary_path / "spawning_job.py"
        self.script.write_text(SPAWNING_JOB, encoding="utf-8")
        self.pid_file = self.temporary_path / "pids.json"
        patcher = mock.patch(
            "mcp_server.pipeline.Pipeline_Jobs.prepare_headless_invocation",
            side_effect=self._prepare_invocation,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.manager = PipelineJobManager(
            PROJECT_ROOT,
            termination_grace=TERMINATION_GRACE,
            temporary_parent=self.temporary_path / "jobs",
        )
        self.addAsyncCleanup(self.manager.close)
        await self.manager.start()

    def _prepare_invocation(
        self,
        tool_id,
        settings_source,
        project_root,
        *,
        python_executable=None,
        snapshot_directory=None,
    ):
        spec = get_tool_spec(tool_id)
        snapshot = create_settings_snapshot(
            spec,
            settings_source,
            snapshot_directory=snapshot_directory,
        )
        # The manager's own interpreter, as in production: a venv's
        # python.exe redirector when the tests run from the project venv.
        return ToolInvocation(
            tool=spec,
            argv=(python_executable or sys.executable, "-u", str(self.script), str(self.pid_file)),
            cwd=str(self.temporary_path),
            settings_path=snapshot,
            owns_settings_snapshot=True,
        )

    async def _start_spawning_job(self):
        job = await self.manager.submit(
            "sanitize_sequences",
            {
                "DIRECTORIES": {"FASTA_DIR": str(self.temporary_path / "outputs")},
                "Sanitize_Sequences.py": {},
            },
        )
        deadline = time.monotonic() + 60
        while not self.pid_file.exists():
            if time.monotonic() > deadline:
                log = await self.manager.read_log(job["job_id"], "stderr")
                self.fail("The job never recorded its processes:\n" + log["text"])
            await asyncio.sleep(0.05)
        pids = json.loads(self.pid_file.read_text(encoding="utf-8"))
        processes = []
        self.addCleanup(_kill, processes)
        for pid in dict.fromkeys(pids.values()):
            processes.append(psutil.Process(pid))
        return job, processes

    async def _assert_all_ended(self, processes, started):
        remaining = TERMINATION_GRACE + 5.0 - (time.monotonic() - started)
        survivors = await asyncio.to_thread(_survivors, processes, max(remaining, 0.0))
        self.assertEqual(
            [],
            [_describe(process) for process in survivors],
            "Processes the job started outlived it.",
        )

    async def test_cancel_ends_the_processes_a_job_started(self):
        job, processes = await self._start_spawning_job()
        started = time.monotonic()
        cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        await self._assert_all_ended(processes, started)

    async def test_close_ends_the_processes_a_running_job_started(self):
        job, processes = await self._start_spawning_job()
        started = time.monotonic()
        await self.manager.close()
        self.assertEqual((await self.manager.get_job(job["job_id"]))["status"], "cancelled")
        await self._assert_all_ended(processes, started)


if __name__ == "__main__":
    unittest.main()
