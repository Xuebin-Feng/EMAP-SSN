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
import signal
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
SIGTERM, SIGKILL = 15, 9
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


    @unittest.skipIf(os.name == "nt", "Process groups are POSIX.")
    async def test_cancel_on_a_system_that_answers_eperm_for_unreaped_members(self):
        # macOS answers EPERM, not success, when a group's only members are
        # zombies, and ESRCH once they are reaped. Stand in for that: the first
        # probe that finds the group empty answers EPERM, the next ESRCH.
        job, processes = await self._start_spawning_job(SIGTERM_IGNORING_SLEEPER)
        killpg = os.killpg
        answers = []

        def macos_killpg(group, signal_number):
            try:
                killpg(group, signal_number)
            except ProcessLookupError:
                if signal_number or answers:
                    raise
                answers.append("EPERM")
                raise PermissionError(1, "Operation not permitted") from None

        started = time.monotonic()
        with mock.patch.object(Pipeline_Jobs.os, "killpg", side_effect=macos_killpg):
            cancelled = await self.manager.cancel(job["job_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["failure_message"])
        await self._assert_all_ended(processes, started)


class FakeProcess:
    """The part of asyncio's Process that stopping a process group uses."""

    pid = 4242

    def __init__(self, returncode=None):
        self.returncode = returncode

    async def wait(self):
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class StuckProcess(FakeProcess):
    """A launched process that is alive: no return code ever comes."""

    async def wait(self):
        await asyncio.sleep(60)


class ProcessGroupStopTests(unittest.IsolatedAsyncioTestCase):
    """Stopping a process group when killpg answers as macOS does.

    macOS answers EPERM, where Linux answers success for zombies and ESRCH
    only for an empty group, when a group has members but none this user may
    signal: for a group the manager started, zombies not yet reaped. Nothing
    of the job runs then, so EPERM ends a wait and a send; success from a
    probe means a live member and must never be read as an end.
    """

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.manager = PipelineJobManager(
            PROJECT_ROOT, termination_grace=0.2, temporary_parent=pathlib.Path(temporary.name) / "jobs")
        for name, value in (("_FORCED_STOP_WAIT", 0.3), ("_LEADER_REAP_WAIT", 0.05)):
            patcher = mock.patch.object(Pipeline_Jobs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Windows has no SIGKILL; the stand-in killpg never sends anything.
        patcher = mock.patch.object(Pipeline_Jobs.signal, "SIGKILL", SIGKILL, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []

    def killpg(self, *script):
        """A killpg whose n-th call does what script[n] says (the last one repeats):
        None answers success, an exception class or instance is raised."""
        script = list(script)

        def fake(group, signal_number):
            self.calls.append(signal_number)
            step = script[min(len(self.calls), len(script)) - 1]
            if step is not None:
                raise step
        return mock.patch.object(Pipeline_Jobs.os, "killpg", fake, create=True)

    async def test_a_probe_that_answers_eperm_with_the_leader_ended_means_the_group_is_over(self):
        with self.killpg(PermissionError):
            self.assertTrue(await PipelineJobManager._wait_for_process_group(
                1, time.monotonic() + 5, FakeProcess(returncode=0)))

    async def test_a_probe_that_answers_eperm_with_the_leader_reaped_soon_means_it_is_over(self):
        with self.killpg(PermissionError):
            self.assertTrue(await PipelineJobManager._wait_for_process_group(
                1, time.monotonic() + 5, FakeProcess()))

    async def test_a_probe_that_answers_eperm_with_the_leader_alive_is_not_an_end(self):
        with self.killpg(PermissionError):
            self.assertFalse(await PipelineJobManager._wait_for_process_group(
                1, time.monotonic() + 0.3, StuckProcess()))
        self.assertGreater(len(self.calls), 1, "it keeps polling until the deadline")

    async def test_a_probe_that_answers_eperm_without_a_leader_is_not_an_end(self):
        with self.killpg(PermissionError):
            self.assertFalse(await PipelineJobManager._wait_for_process_group(1, time.monotonic() + 0.2))

    async def test_a_probe_that_answers_esrch_means_the_group_is_over(self):
        with self.killpg(ProcessLookupError):
            self.assertTrue(await PipelineJobManager._wait_for_process_group(1, time.monotonic() + 5))

    async def test_a_probe_that_answers_success_means_a_member_lives(self):
        with self.killpg(None):
            self.assertFalse(await PipelineJobManager._wait_for_process_group(1, time.monotonic() + 0.2))
        self.assertGreater(len(self.calls), 1, "it polls until the deadline")

    async def test_a_member_that_ignores_sigterm_gets_sigkill_then_eperm_then_esrch(self):
        # SIGTERM sent; the probes find the member alive until the deadline;
        # SIGKILL sent; the probe then answers EPERM (zombie), then ESRCH.
        sigterm, sigkill = SIGTERM, SIGKILL

        def fake(group, signal_number):
            self.calls.append(signal_number)
            if sigkill in self.calls and signal_number == 0:
                raise PermissionError(1, "Operation not permitted")

        with mock.patch.object(Pipeline_Jobs.os, "killpg", fake, create=True):
            problem = await self.manager._stop_process_group(FakeProcess())
        self.assertEqual(problem, "")
        self.assertEqual([c for c in self.calls if c], [sigterm, sigkill])

    async def test_a_live_member_after_sigkill_is_reported(self):
        with self.killpg(None):
            problem = await self.manager._stop_process_group(FakeProcess())
        self.assertEqual(problem, "processes of group 4242 still running after SIGKILL")

    async def test_eperm_on_sigterm_ends_the_stop_without_sigkill(self):
        with self.killpg(PermissionError):
            self.assertEqual(await self.manager._stop_process_group(FakeProcess()), "")
        self.assertEqual(self.calls, [SIGTERM])

    async def test_eperm_on_sigkill_still_waits_for_the_leader(self):
        # The group answers success to SIGTERM and its probe, then EPERM to SIGKILL:
        # the leader is not reaped yet, so the stop waits for it.
        sigkill = SIGKILL

        def fake(group, signal_number):
            self.calls.append(signal_number)
            if signal_number == sigkill:
                raise PermissionError(1, "Operation not permitted")

        process = FakeProcess()
        with mock.patch.object(Pipeline_Jobs.os, "killpg", fake, create=True):
            self.assertEqual(await self.manager._stop_process_group(process), "")
        self.assertEqual(process.returncode, -15)

    async def test_eperm_everywhere_with_a_live_leader_is_never_reported_as_stopped(self):
        # A sandbox that refuses signals: SIGTERM, probes and SIGKILL all answer EPERM.
        with self.killpg(PermissionError):
            problem = await self.manager._stop_process_group(StuckProcess())
        self.assertIn("refused (EPERM)", problem)
        self.assertEqual(self.calls[0], SIGTERM)
        self.assertEqual(self.calls[-1], SIGKILL)

    async def test_eperm_on_sigkill_reports_a_leader_that_will_not_end(self):
        sigkill = SIGKILL

        def fake(group, signal_number):
            if signal_number == sigkill:
                raise PermissionError(1, "Operation not permitted")

        with mock.patch.object(Pipeline_Jobs.os, "killpg", fake, create=True):
            problem = await self.manager._stop_process_group(StuckProcess())
        self.assertEqual(
            problem, "process 4242 still running: signalling its process group was refused (EPERM)")


class LogPageTests(unittest.TestCase):
    def test_pages_end_on_character_boundaries(self):
        text = "é" * 10 + "名😀x"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stdout.log")
            pathlib.Path(path).write_bytes(text.encode())
            offset, pages = 0, []
            while True:
                page = PipelineJobManager._read_log_page("job", "stdout", path, offset, 5)
                pages.append(page["text"])
                offset = page["next_offset"]
                if page["eof"]:
                    break
            self.assertEqual("".join(pages), text)
            self.assertNotIn("�", "".join(pages))

    def test_a_log_that_ends_mid_character_gives_it_whole_once_it_grows(self):
        character = "😀".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stdout.log")
            pathlib.Path(path).write_bytes(b"ab" + character[:2])
            running = PipelineJobManager._read_log_page("job", "stdout", path, 0, 100, True)
            self.assertEqual((running["text"], running["next_offset"], running["eof"]), ("ab", 2, True))
            held = PipelineJobManager._read_log_page("job", "stdout", path, 2, 100, True)
            self.assertEqual((held["text"], held["next_offset"], held["eof"]), ("", 2, True))
            with open(path, "ab") as handle:
                handle.write(character[2:] + b"c")
            grown = PipelineJobManager._read_log_page("job", "stdout", path, 2, 100, True)
            self.assertEqual((grown["text"], grown["next_offset"], grown["eof"]), ("😀c", 7, True))

    def test_a_finished_log_that_ends_mid_character_still_reaches_its_end(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stdout.log")
            pathlib.Path(path).write_bytes(b"ab" + "😀".encode()[:2])
            first = PipelineJobManager._read_log_page("job", "stdout", path, 0, 100, False)
            second = PipelineJobManager._read_log_page("job", "stdout", path, first["next_offset"], 100, False)
            self.assertEqual((first["text"], first["eof"]), ("ab", False))
            self.assertEqual((second["text"], second["next_offset"], second["eof"]), ("�", 4, True))

    def test_invalid_bytes_are_still_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stdout.log")
            pathlib.Path(path).write_bytes(b"a\xffb")
            page = PipelineJobManager._read_log_page("job", "stdout", path, 0, 2)
            self.assertEqual(page["text"] + PipelineJobManager._read_log_page("job", "stdout", path, 2, 2)["text"], "a�b")

class ExitDescriptionTests(unittest.TestCase):
    def test_a_negative_code_on_posix_names_the_signal(self):
        self.assertEqual(Pipeline_Jobs.exit_description(-signal.SIGABRT.value, "linux"),
            f"was ended by signal {signal.SIGABRT.value} (SIGABRT)")
        self.assertEqual(Pipeline_Jobs.exit_description(-signal.SIGTERM.value, "darwin"),
            f"was ended by signal {signal.SIGTERM.value} (SIGTERM)")

    def test_an_unknown_signal_number_is_given_bare(self):
        self.assertEqual(Pipeline_Jobs.exit_description(-200, "linux"), "was ended by signal 200")

    def test_windows_and_positive_codes_are_unchanged(self):
        self.assertEqual(Pipeline_Jobs.exit_description(-6, "win32"), "exited with code -6")
        self.assertEqual(Pipeline_Jobs.exit_description(1, "linux"), "exited with code 1")
        self.assertEqual(Pipeline_Jobs.exit_description(1, "win32"), "exited with code 1")

    def test_the_benchmark_message_names_the_signal_too(self):
        self.assertIn(
            f"was ended by signal {signal.SIGTERM.value} (SIGTERM)",
            Pipeline_Jobs._benchmark_failure({"stages": []}, -signal.SIGTERM.value, "linux"),
        )


class WatchedToolFailureMessageTests(unittest.TestCase):
    """A watched tool that crashed ends its parent with 128+n; the job names signal n again (stand-in codes only)."""

    WATCHED = str(SRC_DIR / "tools" / "Embedding_MSA.py")
    UNWATCHED = str(SRC_DIR / "tools" / "Sanitize_Sequences.py")

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("SSN_TOOL_SUPERVISOR", None)

    @staticmethod
    def invocation(script):
        return ToolInvocation(tool=None, argv=(sys.executable, "-u", script), cwd=".", settings_path="")

    def message(self, script, code, platform="linux"):
        return Pipeline_Jobs.process_failure_message(self.invocation(script), code, platform)

    def test_a_watched_tool_ended_by_a_crash_signal_names_the_signal(self):
        self.assertEqual(self.message(self.WATCHED, 128 + signal.SIGABRT.value),
                         f"The pipeline process was ended by signal {signal.SIGABRT.value} (SIGABRT).")
        self.assertEqual(self.message(self.WATCHED, 128 + signal.SIGSEGV.value, "darwin"),
                         f"The pipeline process was ended by signal {signal.SIGSEGV.value} (SIGSEGV).")

    def test_an_unwatched_tool_with_the_same_code_says_the_code(self):
        self.assertEqual(self.message(self.UNWATCHED, 134), "The pipeline process exited with code 134.")

    def test_a_watched_tool_with_an_ordinary_code_says_the_code(self):
        for code in (1, 2, 128):
            with self.subTest(code=code):
                self.assertEqual(self.message(self.WATCHED, code), f"The pipeline process exited with code {code}.")

    def test_a_negative_code_and_a_stopped_tool_are_as_before(self):
        self.assertEqual(self.message(self.WATCHED, -signal.SIGTERM.value),
                         f"The pipeline process was ended by signal {signal.SIGTERM.value} (SIGTERM).")
        self.assertEqual(self.message(self.UNWATCHED, -signal.SIGABRT.value),
                         f"The pipeline process was ended by signal {signal.SIGABRT.value} (SIGABRT).")
        self.assertEqual(self.message(self.WATCHED, 137), "The pipeline process exited with code 137.")

    def test_windows_is_unchanged(self):
        self.assertEqual(self.message(self.WATCHED, 134, "win32"), "The pipeline process exited with code 134.")

    def test_the_off_switch_leaves_the_code(self):
        os.environ["SSN_TOOL_SUPERVISOR"] = "0"
        self.assertEqual(self.message(self.WATCHED, 134), "The pipeline process exited with code 134.")

    def test_the_benchmark_message_is_unchanged_by_a_128_plus_code(self):
        self.assertIn("exited with code 134",
                      Pipeline_Jobs._benchmark_failure({"stages": []}, 134, "linux"))


if __name__ == "__main__":
    unittest.main()
