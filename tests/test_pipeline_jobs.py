"""Pipeline job processes (mcp_server.pipeline.Pipeline_Jobs).

Runs real processes. Cancelling a job, or closing the manager while a job runs,
must end every process the job started, not only the one the manager launched.
On Windows that one is usually a venv's python.exe redirector, so a tool's own
children run two levels below it. taskkill finds a tree through WMI, which
cancels queries under memory pressure, so the stop must not rely on it.
"""
import asyncio
import json
import os
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

from mcp_server.pipeline import Pipeline_Jobs  # noqa: E402
from mcp_server.pipeline.Pipeline_Jobs import PipelineJobManager  # noqa: E402
from tools.tool_helpers.Tool_Pipeline import (  # noqa: E402
    ToolInvocation,
    create_settings_snapshot,
    get_tool_spec,
)


TERMINATION_GRACE = 1.0

SLEEPER = "import os, time; print(os.getpid(), flush=True); time.sleep(60)"
# What taskkill does when WMI cancels the query that lists the tree: it exits
# with an error and stops nothing. (The message stands in for taskkill's own.)
FAILED_TASKKILL = (1, "ERROR: the process query was cancelled (WMI 0x80041032).")
SIGTERM_IGNORING_SLEEPER = (
    "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "print(os.getpid(), flush=True); time.sleep(60)"
)

# The job runs the child code it is given. The child prints its own PID because
# behind a venv redirector it is not the PID Popen reports, and the file appears
# only once the whole tree is running.
SPAWNING_JOB = """
import json
import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", sys.argv[2]], stdout=subprocess.PIPE)
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
            argv=(
                python_executable or sys.executable,
                "-u",
                str(self.script),
                str(self.pid_file),
                self.child_code,
            ),
            cwd=str(self.temporary_path),
            settings_path=snapshot,
            owns_settings_snapshot=True,
        )

    async def _start_spawning_job(self, child_code=SLEEPER):
        self.child_code = child_code
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
        # Waiting until the processes end by themselves is not stopping them.
        self.assertGreater(remaining, 0.0, "Stopping the job outlasted its grace period by 5 s.")
        survivors = await asyncio.to_thread(_survivors, processes, remaining)
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

    async def test_cancel_ends_a_process_that_ignores_sigterm(self):
        # On POSIX the job's leader exits at SIGTERM within the grace period,
        # so the member that ignores SIGTERM must still get SIGKILL.
        job, processes = await self._start_spawning_job(SIGTERM_IGNORING_SLEEPER)
        started = time.monotonic()
        cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        await self._assert_all_ended(processes, started)

    @unittest.skipUnless(os.name == "nt", "taskkill is used on Windows only.")
    async def test_cancel_ends_the_whole_tree_when_taskkill_fails(self):
        job, processes = await self._start_spawning_job()
        taskkill = mock.AsyncMock(return_value=FAILED_TASKKILL)
        with mock.patch.object(PipelineJobManager, "_taskkill", taskkill):
            started = time.monotonic()
            cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["failure_message"])
        # The graceful attempt still comes first.
        self.assertFalse(taskkill.await_args_list[0].kwargs["force"])
        await self._assert_all_ended(processes, started)

    def _shorten_stop(self):
        # The stop gives up on processes that will not end; there is no need to wait long.
        self.manager.termination_grace = 0.2
        patcher = mock.patch.object(Pipeline_Jobs, "_FORCED_STOP_WAIT", 0.5)
        patcher.start()
        self.addCleanup(patcher.stop)

    @unittest.skipUnless(os.name == "nt", "taskkill is used on Windows only.")
    async def test_cancel_reports_processes_it_could_not_stop(self):
        job, processes = await self._start_spawning_job()
        self._shorten_stop()
        taskkill = mock.AsyncMock(return_value=FAILED_TASKKILL)
        with mock.patch.object(PipelineJobManager, "_taskkill", taskkill), \
                mock.patch.object(psutil.Process, "kill", side_effect=psutil.AccessDenied()), \
                self.assertLogs(Pipeline_Jobs.__name__, "WARNING") as logs:
            cancelled = await self.manager.cancel(job["job_id"])
        # The launched process is still running too, so the job cannot finish.
        self.assertEqual(cancelled["status"], "cancelling")
        message = cancelled["failure_message"]
        self.assertIsNotNone(message)
        self.assertIn("still running after a forced stop", message)
        for process in processes:
            self.assertIn(f" {process.pid}", message)
        self.assertIn(job["job_id"], "\n".join(logs.output))

    @unittest.skipUnless(os.name == "nt", "taskkill is used on Windows only.")
    async def test_cancel_without_a_process_list_reports_what_taskkill_said(self):
        # Only if psutil cannot read the tree does the stop fall back to taskkill,
        # and when that fails too, to killing the launched process on its own.
        job, processes = await self._start_spawning_job()
        self._shorten_stop()
        taskkill = mock.AsyncMock(return_value=FAILED_TASKKILL)
        with mock.patch.object(PipelineJobManager, "_taskkill", taskkill), \
                mock.patch.object(psutil.Process, "children", side_effect=psutil.AccessDenied()), \
                self.assertLogs(Pipeline_Jobs.__name__, "WARNING"):
            cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIn(FAILED_TASKKILL[1], cancelled["failure_message"])
        self.assertIn("killed on its own", cancelled["failure_message"])
        self.assertEqual([False, True, True], [call.kwargs["force"] for call in taskkill.await_args_list])

    @unittest.skipIf(os.name == "nt", "Windows stops console processes only by force.")
    async def test_cancel_lets_group_members_clean_up_within_the_grace_period(self):
        # multiprocessing's resource tracker outlives the leader the same way,
        # to unlink the semaphores the dead processes leave behind.
        marker = self.temporary_path / "cleaned-up"
        cleaning_sleeper = (
            "import os, signal, sys, time\n"
            "def stop(*_):\n"
            "    time.sleep(0.3)\n"
            f"    open({str(marker)!r}, 'w').close()\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "print(os.getpid(), flush=True)\n"
            "time.sleep(60)\n"
        )
        # A long grace costs nothing here: cancel returns once the group is empty.
        self.manager.termination_grace = 10.0
        job, processes = await self._start_spawning_job(cleaning_sleeper)
        cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertTrue(marker.exists(), "A group member was killed before it finished cleaning up.")
        self.assertEqual([], [_describe(process) for process in processes if _alive(process)])


if __name__ == "__main__":
    unittest.main()
